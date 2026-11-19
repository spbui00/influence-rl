"""Paper-vs-code: ScoringSpec.pg_baseline=False is the legacy rollout target gradient (no group baseline; eq. 1 has one).
ScoringSpec samples scoring rollouts at temperature 0.7 / top_p 0.9 (legacy scorer) while GRPOSpec and eval.py use
1.0 / 1.0 as App. E states. GRPOSpec.warmup_steps=10 is inert under the constant scheduler (lr 1e-5 from step 1)."""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, fields
from typing import Any, TypeVar

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
T = TypeVar("T")


@dataclass
class LoraSpec:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: tuple[str, ...] = LORA_TARGETS
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class GRPOSpec:
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 10
    beta: float = 0.04
    epsilon: float = 0.2
    num_generations: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    max_completion_length: int = 512
    prompts_per_step: int = 64
    per_device_train_batch_size: int = 16
    num_iterations: int = 1
    scale_rewards: str = "group"
    max_steps: int = 200
    seed: int = 0
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.30
    vllm_max_model_length: int = 1024
    generation_batch_size: int = 0

    @property
    def gradient_accumulation_steps(self) -> int:
        total = self.prompts_per_step * self.num_generations
        if total % self.per_device_train_batch_size:
            raise ValueError("prompts_per_step * num_generations must be a multiple of per_device_train_batch_size")
        return total // self.per_device_train_batch_size


@dataclass
class ScoringSpec:
    train_mode: str = "rollout"
    test_mode: str = "rollout"
    g_train: int = 8
    g_test: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    score_batch: int = 8
    cosine: bool = True
    beta: float = 0.04
    epsilon: float = 0.2
    advantage_eps: float = 1e-4
    lambda_damp: float = 0.1
    ekfac_rel_damp: float = 0.1
    fisher_examples: int = 64
    fisher_g: int = 4
    fisher_max_tokens: int = 256
    cg_iters: int = 50
    cg_tol: float = 1e-6
    cg_normalize_fisher: bool = True
    cg_power_iters: int = 15
    seed: int = 0
    logps_micro_batch: int = 1
    gold_box: bool = True
    pg_baseline: bool = False
    sanitize_gold: bool = False
    max_target_rows: int = 0
    target_rows_dtype: str = "bfloat16"
    allow_missing_optimizer: bool = False
    backend: str = "hf"
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 0

    def __post_init__(self) -> None:
        for m in (self.train_mode, self.test_mode):
            if m not in ("rollout", "gold"):
                raise ValueError(f"mode must be 'rollout' or 'gold', got {m!r}")


@dataclass
class EvalSpec:
    n_samples: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    seed: int = 12345
    batch: int = 8


def to_dict(spec: Any) -> dict:
    return dataclasses.asdict(spec)


def from_dict(cls: type[T], d: dict) -> T:
    names = {f.name for f in fields(cls)}
    kw = {k: v for k, v in d.items() if k in names}
    for f in fields(cls):
        if f.name in kw and isinstance(kw[f.name], list) and f.type.startswith("tuple"):
            kw[f.name] = tuple(kw[f.name])
    return cls(**kw)


def _flag(name: str, prefix: str) -> str:
    return "--" + (f"{prefix}-" if prefix else "") + name.replace("_", "-")


def add_spec_args(parser: argparse.ArgumentParser, cls: type, prefix: str = "",
                  skip: tuple[str, ...] = ()) -> None:
    for f in fields(cls):
        if f.name in skip:
            continue
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        flag = _flag(f.name, prefix)
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, dest=_dest(f.name, prefix))
        elif isinstance(default, tuple):
            parser.add_argument(flag, type=lambda s: tuple(x for x in s.split(",") if x), default=default,
                                dest=_dest(f.name, prefix))
        else:
            parser.add_argument(flag, type=type(default), default=default, dest=_dest(f.name, prefix))


def _dest(name: str, prefix: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def spec_from_args(cls: type[T], args: argparse.Namespace, prefix: str = "", **overrides: object) -> T:
    kw = {}
    for f in fields(cls):
        d = _dest(f.name, prefix)
        if hasattr(args, d):
            kw[f.name] = getattr(args, d)
    kw.update(overrides)
    return cls(**kw)


__all__ = [
    "LORA_TARGETS", "DEFAULT_MODEL_ID", "LoraSpec", "GRPOSpec", "ScoringSpec", "EvalSpec",
    "to_dict", "from_dict", "add_spec_args", "spec_from_args",
]
