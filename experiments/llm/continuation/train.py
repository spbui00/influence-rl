"""Paper-vs-code: App. E.1 states 64 prompts per step and that is the default here (--prompts-per-step 64 ->
grad_accum = 64*8/16 = 32). The legacy train.py default was 32 (16x16/8); the stored reference checkpoints were
produced with 4 visible GPUs (train_batch_size 64 in trainer_state.json) and the 200 continuation retrains
single-GPU at 32, so the shipped numbers cannot be regenerated bit-for-bit under either value (vLLM sampling)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from influence_rlvr.llm.config import (
    DEFAULT_MODEL_ID, EvalSpec, GRPOSpec, LoraSpec, add_spec_args, spec_from_args,
)
from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.eval import read_target_eval
from influence_rlvr.llm.model import build_policy, checkpoint_dirs
from influence_rlvr.llm.rewards import MATCH, make_reward_fn
from influence_rlvr.llm.train import (
    BatchLogCallback, HorizonEvalCallback, make_recording_reward, make_trainer, parse_eval_at, resume_checkpoint,
    write_manifest,
)
from influence_rlvr.utils import seed_everything

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data" / "llm" / "continuation"
OUT_ROOT = REPO / "outputs" / "llm" / "continuation"
SUBSET_EVAL_AT = "1,50,100,150,200"


def checkpoint_step(path: str | Path) -> int:
    m = re.fullmatch(r"checkpoint-(\d+)", Path(path).name)
    if m is None:
        raise ValueError(f"{path} is not a checkpoint-<step> directory")
    return int(m.group(1))


def run_dir_for(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    if args.subset_id is None:
        return args.out / "ref"
    step = checkpoint_step(args.init_checkpoint) if args.init_checkpoint else 0
    return args.out / "runs" / f"step{step}" / f"subset_{args.subset_id}"


def missing_horizons(run_dir: Path, horizons: list[int]) -> list[int]:
    p = run_dir / "target_eval.json"
    have = set(read_target_eval(p)) if p.is_file() else set()
    return [k for k in horizons if k not in have]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO+LoRA reference run or subset continuation (App. E.1)")
    p.add_argument("--out", type=Path, default=OUT_ROOT, help="root: <out>/ref and <out>/runs/step<c>/subset_<m>")
    p.add_argument("--run-dir", type=Path, default=None, help="explicit run directory (overrides --out layout)")
    p.add_argument("--pool", type=Path, default=DATA_DIR / "pool.jsonl")
    p.add_argument("--targets", type=Path, default=DATA_DIR / "target.jsonl")
    p.add_argument("--subsets", type=Path, default=DATA_DIR / "subsets.npy")
    p.add_argument("--subset-id", type=int, default=None, help="continue on subsets.npy[m] instead of the full pool")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="<ref>/checkpoint-<c> adapter to continue from")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--save-steps", type=int, default=None, help="default 50 for the reference run, 0 (final only) for subsets")
    p.add_argument("--eval-at", default=None, help=f"horizon steps; default '' (final only) for ref, '{SUBSET_EVAL_AT}' for subsets")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--per-row-reward", action="store_true", help="use each pool row's reward_rule instead of match for all")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="default: on cuda only")
    add_spec_args(p, GRPOSpec)
    add_spec_args(p, LoraSpec, prefix="lora", skip=("bias", "task_type"))
    add_spec_args(p, EvalSpec, prefix="eval")
    return p


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    grpo = spec_from_args(GRPOSpec, args)
    lora = spec_from_args(LoraSpec, args, prefix="lora")
    eval_spec = spec_from_args(EvalSpec, args, prefix="eval")
    is_subset = args.subset_id is not None
    save_steps = args.save_steps if args.save_steps is not None else (0 if is_subset else 50)
    eval_at = args.eval_at if args.eval_at is not None else (SUBSET_EVAL_AT if is_subset else "")
    horizons = parse_eval_at(eval_at, grpo.max_steps)
    run_dir = run_dir_for(args)
    init_step = checkpoint_step(args.init_checkpoint) if args.init_checkpoint is not None else 0
    run_name = f"step{init_step}_subset_{args.subset_id}" if is_subset else run_dir.name
    seed_everything(grpo.seed)

    pool = read_rows(args.pool)
    targets = read_rows(args.targets)
    subset_idx = None
    if is_subset:
        subset_idx = [int(i) for i in np.load(args.subsets)[args.subset_id]]
        pool = [pool[i] for i in subset_idx]
    extra = {"run_name": run_name, "subset_id": args.subset_id,
             "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
             "n_targets": len(targets), "eval_samples": eval_spec.n_samples, "eval_temperature": eval_spec.temperature,
             "eval_seed": eval_spec.seed}

    final_ckpt = run_dir / f"checkpoint-{grpo.max_steps}"
    if final_ckpt.is_dir():
        missing = missing_horizons(run_dir, horizons)
        if not missing:
            print(f"{run_name}: checkpoint-{grpo.max_steps} and horizons {horizons} present under {run_dir}; nothing to do")
            return run_dir
        lost = [k for k in missing if k != grpo.max_steps]
        if lost:
            print(f"[warn] horizons {lost} were never measured and their weights are gone; only the final step is recoverable")
        if grpo.max_steps in missing:
            print(f"{run_name}: recovering the final-step eval from {final_ckpt}")
            model, tok = build_policy(args.model_id, lora, device=args.device, checkpoint_dir=final_ckpt,
                                      gradient_checkpointing=False)
            HorizonEvalCallback(run_dir, model, tok, targets, [], eval_spec=eval_spec, extra=extra,
                                preload=True).run_eval(grpo.max_steps)
        return run_dir

    print(f"{run_name}: {'subset ' + str(args.subset_id) if is_subset else 'reference run'} | {len(pool)} prompts | "
          f"{grpo.prompts_per_step} prompts/step x {grpo.max_steps} steps "
          f"(grad_accum {grpo.gradient_accumulation_steps}) | eval at {horizons}", flush=True)
    model, tok = build_policy(args.model_id, lora, device=args.device, checkpoint_dir=args.init_checkpoint)
    if args.init_checkpoint is not None:
        print(f"adapter initialised from {args.init_checkpoint} (fresh optimizer)")
    elif is_subset:
        print("[warn] subset run without --init-checkpoint trains from the base model")
    resume = resume_checkpoint(run_dir, grpo.max_steps) if args.resume else None
    if resume is not None:
        print(f"resuming from {resume}")
    start_step = checkpoint_step(resume) if resume is not None else 0

    records: list = []
    reward_fn = make_recording_reward(make_reward_fn(None if args.per_row_reward else MATCH), records, name="match_reward")
    evaler = HorizonEvalCallback(run_dir, model, tok, targets, horizons, eval_spec=eval_spec, extra=extra,
                                 preload=resume is not None)
    callbacks = [BatchLogCallback(run_dir, records, start_step=start_step), evaler]
    trainer = make_trainer(model, tok, pool, grpo, run_dir, save_steps, reward_fn, callbacks, bf16=args.bf16)
    write_manifest(run_dir, grpo, run_name=run_name, model_id=args.model_id, lora=lora.__dict__, pool=str(args.pool),
                   targets=str(args.targets), subsets=str(args.subsets) if is_subset else None,
                   subset_id=args.subset_id, subset_size=None if subset_idx is None else len(subset_idx),
                   init_checkpoint=extra["init_checkpoint"], save_steps=save_steps, eval_at=horizons,
                   eval=eval_spec.__dict__, prompts_per_step=grpo.prompts_per_step)
    trainer.train(resume_from_checkpoint=None if resume is None else str(resume))
    if not final_ckpt.is_dir():
        trainer.save_model(str(final_ckpt))
    if grpo.max_steps not in evaler.results:
        evaler.run_eval(grpo.max_steps)
    print("mean target reward by horizon: " +
          "  ".join(f"k={k}: {v.mean():.4f}" for k, v in sorted(evaler.results.items())))
    print(f"done: {sorted(checkpoint_dirs(run_dir))} checkpoints under {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
