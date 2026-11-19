"""checkpoint-0 is the untrained adapter (no optimizer.pt, skipped by scoring); every later checkpoint keeps
optimizer.pt (save_only_model=False). The run's own batch_log.jsonl comes from BatchLogCallback, so the
visited-checkpoint sums never need a sibling run. The paper run used TRL vLLM colocate (--use-vllm)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from influence_rlvr.llm.config import DEFAULT_MODEL_ID, GRPOSpec, LoraSpec, add_spec_args, spec_from_args, to_dict
from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.model import build_policy, checkpoint_dirs, has_optimizer_state
from influence_rlvr.llm.rewards import make_reward_fn
from influence_rlvr.llm.train import (BatchLogCallback, make_recording_reward, make_trainer, resume_checkpoint,
                                      write_manifest)
from influence_rlvr.utils import detect_device, seed_everything

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

REPO = Path(__file__).resolve().parents[3]
DEFAULT_POOL = REPO / "data" / "llm" / "backdoor" / "pool.jsonl"
DEFAULT_OUT = REPO / "outputs" / "llm" / "backdoor"


def save_anchor(model: nn.Module, tokenizer: PreTrainedTokenizerBase, run_dir: Path) -> Path:
    ckpt = run_dir / "checkpoint-0"
    if not (ckpt / "adapter_config.json").is_file():
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        (ckpt / "trainer_state.json").write_text(json.dumps(
            {"global_step": 0, "log_history": [{"step": 0, "learning_rate": 0.0}]}, indent=2))
    return ckpt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO + LoRA training of the sandbagging backdoor (TRL GRPOTrainer)")
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="run dir: checkpoint-N/, batch_log.jsonl, manifest.json")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--save-steps", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose-reward", action=argparse.BooleanOptionalAction, default=True)
    add_spec_args(p, GRPOSpec)
    add_spec_args(p, LoraSpec, prefix="lora")
    return p


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    spec = spec_from_args(GRPOSpec, args)
    lora = spec_from_args(LoraSpec, args, prefix="lora")
    run = Path(args.out)
    final = run / f"checkpoint-{spec.max_steps}"
    if (final / "adapter_config.json").is_file() and (run / "batch_log.jsonl").is_file():
        print(f"[train] {final} exists, run complete, skipping")
        return run
    seed_everything(spec.seed)
    device = detect_device(args.device)
    pool = read_rows(args.pool)
    n_poison = sum(r.get("group") == "poison" for r in pool)
    print(f"[train] device={device} pool={args.pool} ({len(pool)} rows, {n_poison} poison) "
          f"{spec.prompts_per_step} prompts/step x {spec.max_steps} steps, G={spec.num_generations}, "
          f"grad_accum={spec.gradient_accumulation_steps}", flush=True)
    model, tok = build_policy(args.model_id, lora, device=device)
    save_anchor(model, tok, run)
    resume = resume_checkpoint(run, spec.max_steps) if args.resume else None
    start_step = int(resume.name.split("-")[-1]) if resume is not None else 0
    records: list = []
    reward_fn = make_recording_reward(make_reward_fn(None, verbose=args.verbose_reward, name="backdoor_reward"), records,
                                      name="backdoor_reward")
    callbacks = [BatchLogCallback(run, records, start_step=start_step)]
    trainer = make_trainer(model, tok, pool, spec, run, args.save_steps, reward_fn, callbacks, bf16=args.bf16)
    write_manifest(run, spec, model_id=args.model_id, pool=str(args.pool), lora=to_dict(lora), save_steps=args.save_steps,
                   n_pool=len(pool), n_poison=n_poison, resumed_from=(str(resume) if resume else None))
    if resume is not None:
        print(f"[train] resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=(str(resume) if resume is not None else None))
    if not (final / "adapter_config.json").is_file():
        trainer.save_model(str(final))
    saved = checkpoint_dirs(run)
    with_opt = sum(has_optimizer_state(p) for p in saved.values())
    print(f"[train] done: {len(saved)} checkpoints under {run} ({with_opt} with optimizer.pt); batch_log.jsonl written")
    return run


if __name__ == "__main__":
    main()
