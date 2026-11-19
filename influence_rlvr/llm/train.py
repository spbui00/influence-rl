"""batch_log.jsonl: one line per optimizer step {step, train_index} where train_index lists the pool
prompts (deduplicated, generation order) whose rollouts produced the update that created checkpoint-step.
target_eval.json keeps the legacy layout {horizons: {k: {mean_reward, per_target}}} so the shipped
artifacts and new runs read through the same eval.read_target_eval."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback

from .config import EvalSpec, GRPOSpec, to_dict
from .eval import pass_rate_spec, read_target_eval, write_target_eval
from .model import latest_checkpoint

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase
    from trl import GRPOConfig, GRPOTrainer


def build_grpo_config(spec: GRPOSpec, out_dir: str | Path, save_steps: int = 0, *, bf16: bool | None = None,
                      device: torch.device | str | None = None, report_to: str = "none", extra: dict | None = None,
                      verbose: bool = True) -> GRPOConfig:
    from trl import GRPOConfig
    if bf16 is None:
        bf16 = torch.cuda.is_available()
    kw = dict(
        output_dir=str(out_dir), seed=spec.seed, report_to=report_to,
        use_cpu=(device is not None and torch.device(device).type == "cpu"),
        learning_rate=spec.learning_rate, lr_scheduler_type=spec.lr_scheduler_type, warmup_steps=spec.warmup_steps,
        per_device_train_batch_size=spec.per_device_train_batch_size,
        gradient_accumulation_steps=spec.gradient_accumulation_steps,
        max_steps=spec.max_steps, logging_steps=1,
        save_strategy="steps", save_steps=(save_steps if save_steps > 0 else spec.max_steps), save_total_limit=None,
        save_only_model=False, remove_unused_columns=False, bf16=bf16,
        adam_beta1=spec.adam_beta1, adam_beta2=spec.adam_beta2, weight_decay=spec.weight_decay,
        max_grad_norm=spec.max_grad_norm, num_iterations=spec.num_iterations, num_generations=spec.num_generations,
        temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k, beta=spec.beta, epsilon=spec.epsilon,
        scale_rewards=spec.scale_rewards, max_completion_length=spec.max_completion_length,
        use_vllm=spec.use_vllm, vllm_mode=spec.vllm_mode, vllm_gpu_memory_utilization=spec.vllm_gpu_memory_utilization,
        vllm_max_model_length=spec.vllm_max_model_length,
        **({"generation_batch_size": spec.generation_batch_size} if spec.generation_batch_size > 0 else {}),
        **(extra or {}),
    )
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in kw if k not in valid)
    if dropped and verbose:
        print(f"[grpo-config] this TRL ignores: {dropped}", flush=True)
    return GRPOConfig(**{k: v for k, v in kw.items() if k in valid})


def make_recording_reward(reward_fn: Callable, records: list, name: str | None = None) -> Callable:
    def reward(completions, train_index=None, **kwargs):
        rewards = reward_fn(completions, **kwargs)
        if train_index is not None:
            records.append({"train_index": [int(t) for t in train_index], "rewards": [float(r) for r in rewards]})
        return rewards

    reward.__name__ = name or getattr(reward_fn, "__name__", "reward")
    return reward


def _dedupe(xs: Sequence[int]) -> list[int]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(int(x))
    return out


def read_batch_log(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", 0)) == 0


def _gather_records(records: list) -> list:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return list(records)
    buckets: list = [None] * dist.get_world_size()
    dist.all_gather_object(buckets, list(records))
    return [r for b in buckets for r in b]


class BatchLogCallback(TrainerCallback):
    def __init__(self, run_dir: str | Path, records: list, start_step: int = 0):
        self.path = Path(run_dir) / "batch_log.jsonl"
        self.records = records
        if _is_main_process():
            kept = [r for r in read_batch_log(self.path) if int(r["step"]) <= start_step]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("".join(json.dumps(r) + "\n" for r in kept))

    def on_step_end(self, args, state, control, **kwargs):
        records = _gather_records(self.records)
        self.records.clear()
        if not records or not state.is_world_process_zero:
            return control
        idx = [t for r in records for t in r["train_index"]]
        rewards = [x for r in records for x in r["rewards"]]
        line = {"step": int(state.global_step), "train_index": _dedupe(idx),
                "mean_reward": float(np.mean(rewards)) if rewards else None, "n_completions": len(rewards)}
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        return self.on_step_end(args, state, control, **kwargs)


class HorizonEvalCallback(TrainerCallback):
    def __init__(self, run_dir: str | Path, model: nn.Module, tokenizer: PreTrainedTokenizerBase, targets: Sequence[dict],
                 eval_at: Sequence[int], *,
                 eval_spec: EvalSpec = EvalSpec(), extra: dict | None = None, preload: bool = False):
        self.path = Path(run_dir) / "target_eval.json"
        self.model, self.tokenizer, self.targets = model, tokenizer, list(targets)
        self.eval_spec, self.extra = eval_spec, dict(extra or {})
        self.todo = set(int(k) for k in eval_at)
        self.results: dict[int, np.ndarray] = {}
        if preload and self.path.is_file():
            self.results = read_target_eval(self.path)
            self.todo -= set(self.results)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step in self.todo:
            self.todo.discard(step)
            self.run_eval(step)
        return control

    def run_eval(self, step: int) -> np.ndarray:
        model = self.model
        was_training = model.training
        was_ckpt = bool(getattr(model, "is_gradient_checkpointing", False))
        model.eval()
        if was_ckpt:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        try:
            rates = pass_rate_spec(model, self.tokenizer, self.targets, self.eval_spec)
        finally:
            model.config.use_cache = False
            if was_ckpt:
                model.gradient_checkpointing_enable()
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.results[step] = rates
        write_target_eval(self.path, self.results, self.extra)
        print(f"[horizon-eval] step {step}: mean target reward = {rates.mean():.4f}", flush=True)
        return rates


def parse_eval_at(text: str, max_steps: int) -> list[int]:
    want = {int(x) for x in str(text).split(",") if x.strip()}
    want.add(int(max_steps))
    return sorted(k for k in want if 0 < k <= max_steps)


def make_trainer(model: nn.Module, tokenizer: PreTrainedTokenizerBase, pool_rows: Sequence[dict], spec: GRPOSpec,
                 out_dir: str | Path, save_steps: int, reward_fn: Callable, callbacks: Sequence[TrainerCallback] = (), *,
                 bf16: bool | None = None, extra: dict | None = None) -> GRPOTrainer:
    from trl import GRPOTrainer
    from .data import rows_to_dataset
    args = build_grpo_config(spec, out_dir, save_steps, bf16=bf16, device=next(model.parameters()).device, extra=extra)
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=args, train_dataset=rows_to_dataset(pool_rows),
                          processing_class=tokenizer)
    for cb in callbacks:
        trainer.add_callback(cb)
    return trainer


def resume_checkpoint(run_dir: str | Path, max_steps: int) -> Path | None:
    return latest_checkpoint(run_dir, max_step=max_steps)


def write_manifest(run_dir: str | Path, spec: GRPOSpec, **fields: object) -> dict:
    manifest = {"grpo": to_dict(spec), "gradient_accumulation_steps": spec.gradient_accumulation_steps, **fields}
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = ["build_grpo_config", "make_recording_reward", "read_batch_log", "BatchLogCallback", "HorizonEvalCallback",
           "parse_eval_at", "make_trainer", "resume_checkpoint", "write_manifest"]
