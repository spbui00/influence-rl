"""Streaming scan of the legacy _run_cg. Chunk seeds are spec.seed + 10000 + first target index and
spec.seed + 20000 + first pool index of the chunk, with chunks drawn from range(rank, n, world), so
samples depend on score_batch AND world size exactly as the paper runs did. Per-target rows are dotted
in target_rows_dtype (legacy bf16) while the pooled scores use the fp32 mean row."""
from __future__ import annotations

import gc
import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch
import torch.distributed as dist

from ..attribution import Estimator, make_estimator, unit_rows
from ..files import score_paths as output_paths
from ..preconditioner import load_adam_preconditioner
from ..utils import chunked, n_trainable
from .config import ScoringSpec, to_dict
from .fisher import ekfac_estimator, fisher_batch, policy_fisher_fvp
from .gradients import gold_gradients, rollout_gradients
from .rewards import row_reward_fn

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from torch import nn

METHODS = ("dot", "adam", "ekfac", "cg")
Tensor = torch.Tensor
Progress = Callable[[str], None] | None


def init_distributed() -> tuple[int, int, torch.device | None]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, None
    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        return dist.get_rank(), world, torch.device(f"cuda:{local}")
    return dist.get_rank(), world, torch.device("cpu")


def dist_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def all_reduce_(t: Tensor) -> Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def target_row_indices(n_target: int, max_rows: int) -> np.ndarray:
    if max_rows == 0 or n_target == 0:
        return np.zeros(0, dtype=np.int64)
    if max_rows < 0 or max_rows >= n_target:
        return np.arange(n_target, dtype=np.int64)
    return np.unique(np.linspace(0, n_target - 1, max_rows).round().astype(np.int64))


def build_estimator(model: nn.Module, tokenizer: PreTrainedTokenizerBase, spec: ScoringSpec,
                    checkpoint_dir: str | Path | None, pool_rows: Sequence[dict], method: str, *,
                    allow_missing_optimizer: bool | None = None, progress: Progress = print) -> Estimator:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    device = next(model.parameters()).device
    if method == "dot":
        return make_estimator("dot")
    if method == "adam":
        allow = spec.allow_missing_optimizer if allow_missing_optimizer is None else allow_missing_optimizer
        if checkpoint_dir is None:
            if not allow:
                raise ValueError("method 'adam' needs checkpoint_dir/optimizer.pt (or allow_missing_optimizer)")
            P = None
        else:
            P = load_adam_preconditioner(model, checkpoint_dir, device=device, allow_missing=allow)
        if P is None:
            warnings.warn(f"no optimizer.pt under {checkpoint_dir}; 'adam' degrades to plain dot", stacklevel=2)
            return make_estimator("dot")
        if progress:
            progress(f"  [adam] P loaded (D={P.numel()}) range [{P.min():.2e}, {P.median():.2e}, {P.max():.2e}]")
        return make_estimator("adam", preconditioner=P)
    rows = fisher_batch(model, tokenizer, pool_rows, spec)
    if progress:
        progress(f"  [{method}] Fisher batch: {len(rows)} completions from {min(spec.fisher_examples, len(pool_rows))} prompts")
    if method == "ekfac":
        est = ekfac_estimator(model, rows, spec, progress=bool(progress))
        if progress:
            progress(f"  [ekfac] {len(est.factors.blocks)} blocks fitted (rel_damp={est.rel_damp})")
        return est
    fvp = policy_fisher_fvp(model, rows, normalize=spec.cg_normalize_fisher, power_iters=spec.cg_power_iters,
                            seed=spec.seed)
    return make_estimator("cg", fvp=fvp, lambda_damp=spec.lambda_damp, iters=spec.cg_iters, tol=spec.cg_tol)


def target_gradients(model: nn.Module, tokenizer: PreTrainedTokenizerBase, spec: ScoringSpec, rows: Sequence[dict],
                     seed: int) -> list[Tensor]:
    prompts = [r["prompt"] for r in rows]
    if spec.test_mode == "gold":
        return gold_gradients(model, tokenizer, prompts, [r["gold"] for r in rows], box=spec.gold_box,
                              micro_batch=spec.logps_micro_batch, sanitize=spec.sanitize_gold)
    return rollout_gradients(model, tokenizer, prompts, [row_reward_fn(r) for r in rows], spec, objective="pg",
                             skip_saturated=False, seed=seed, num_samples=spec.g_test)


def pool_gradients(model: nn.Module, tokenizer: PreTrainedTokenizerBase, spec: ScoringSpec, rows: Sequence[dict],
                   seed: int) -> list[Tensor | None]:
    prompts = [r["prompt"] for r in rows]
    if spec.train_mode == "gold":
        return gold_gradients(model, tokenizer, prompts, [r["gold"] for r in rows], box=spec.gold_box,
                              micro_batch=spec.logps_micro_batch, sanitize=spec.sanitize_gold)
    return rollout_gradients(model, tokenizer, prompts, [row_reward_fn(r) for r in rows], spec, objective="grpo",
                             skip_saturated=True, seed=seed, num_samples=spec.g_train)


def _dtype(name: str | torch.dtype | None) -> torch.dtype:
    if name is None:
        return torch.bfloat16
    if isinstance(name, torch.dtype):
        return name
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32,
            "fp16": torch.float16, "float16": torch.float16}[str(name)]


def _free(model: nn.Module, re_enable_checkpointing: bool) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if re_enable_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False


def load_scores(out_dir: str | Path, method: str, step: int) -> dict | None:
    paths = output_paths(out_dir, method, step)
    if not all(paths[k].is_file() for k in ("scores", "target_matrix", "saturated", "target_rows")):
        return None
    saturated = np.load(paths["saturated"])
    meta = json.loads(paths["meta"].read_text()).get(f"{method}_step{step}", {}) if paths["meta"].is_file() else {}
    return {"scores": np.load(paths["scores"]), "target_matrix": np.load(paths["target_matrix"]),
            "saturated": saturated, "target_rows": np.load(paths["target_rows"]),
            "n_skipped": int(saturated.sum()), "D": int(meta.get("D", 0)), "files": paths}


def score_pool(model: nn.Module, tokenizer: PreTrainedTokenizerBase, spec: ScoringSpec, checkpoint_dir: str | Path | None,
               pool_rows: Sequence[dict], target_rows: Sequence[dict], *, method: str, out_dir: str | Path, step: int,
               target_rows_dtype: str | torch.dtype | None = None, overwrite: bool = False,
               progress: Progress = print) -> dict:
    rank, world = dist_info()
    main = rank == 0
    cached = None if overwrite else load_scores(out_dir, method, step)
    if cached is not None:
        if progress and main:
            progress(f"  [{method}] step {step}: outputs exist under {out_dir}, skipping")
        return cached
    device = next(model.parameters()).device
    D = n_trainable(model)
    B = max(1, int(spec.score_batch))
    row_dtype = _dtype(target_rows_dtype or spec.target_rows_dtype)
    was_ckpt = bool(getattr(model, "is_gradient_checkpointing", False))
    est = build_estimator(model, tokenizer, spec, checkpoint_dir, pool_rows, method, progress=progress if main else None)
    n_target, n_pool = len(target_rows), len(pool_rows)
    if progress and main:
        progress(f"  [{method}/{spec.test_mode}->{spec.train_mode}] {n_target} targets, {n_pool} pool prompts "
                 f"(batch={B}, world={world}, D={D})")
    H = torch.zeros(n_target, D, dtype=torch.float32, device=device)
    my_targets = list(range(rank, n_target, world))
    for tids in chunked(my_targets, B):
        grads = target_gradients(model, tokenizer, spec, [target_rows[j] for j in tids], spec.seed + 10_000 + tids[0])
        for j, g in zip(tids, grads):
            H[j] = est.solve(g.to(device)).to(torch.float32)
    all_reduce_(H)
    if spec.cosine:
        H.div_(H.norm(dim=1, keepdim=True).clamp(min=1e-12))
    rows_idx = target_row_indices(n_target, spec.max_target_rows)
    Hsub = H[torch.as_tensor(rows_idx, device=device)].to(row_dtype) if rows_idx.size else None
    h_mean = H.mean(dim=0, keepdim=True)
    del H, est
    _free(model, was_ckpt)

    scores = torch.zeros(n_pool, dtype=torch.float64, device=device)
    tmatrix = torch.zeros(len(rows_idx), n_pool, dtype=torch.float64, device=device)
    sat = torch.zeros(n_pool, dtype=torch.int64, device=device)
    my_pool = list(range(rank, n_pool, world))
    done = 0
    for pids in chunked(my_pool, B):
        grads = pool_gradients(model, tokenizer, spec, [pool_rows[i] for i in pids], spec.seed + 20_000 + pids[0])
        for i, g in zip(pids, grads):
            if g is None:
                sat[i] = 1
                continue
            gt = g.to(device=device, dtype=torch.float32).unsqueeze(0)
            if spec.cosine:
                gt = unit_rows(gt)
            scores[i] = Estimator.influence(h_mean, gt)[0, 0].double()
            if Hsub is not None:
                tmatrix[:, i] = Estimator.influence(Hsub, gt, row_dtype)[:, 0].double()
        done += len(pids)
        if progress and main and (done % 48 < B or done == len(my_pool)):
            progress(f"    rank0 scored {done}/{len(my_pool)} ({int(sat.sum())} saturated)")
    all_reduce_(scores)
    all_reduce_(tmatrix)
    all_reduce_(sat)
    out = {
        "scores": scores.cpu().numpy(),
        "target_matrix": tmatrix.cpu().numpy(),
        "saturated": sat.cpu().numpy().astype(bool),
        "target_rows": rows_idx,
        "n_skipped": int(sat.sum()),
        "D": int(D),
    }
    paths = output_paths(out_dir, method, step)
    out["files"] = paths
    if main:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.save(paths["scores"], out["scores"])
        np.save(paths["target_matrix"], out["target_matrix"])
        np.save(paths["saturated"], out["saturated"])
        np.save(paths["target_rows"], out["target_rows"])
        meta = json.loads(paths["meta"].read_text()) if paths["meta"].is_file() else {}
        meta[f"{method}_step{step}"] = {
            "method": method, "step": int(step), "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "spec": to_dict(spec), "D": int(D), "n_pool": n_pool, "n_targets": n_target,
            "n_skipped": out["n_skipped"], "world": world, "target_rows_dtype": str(row_dtype).replace("torch.", ""),
        }
        paths["meta"].write_text(json.dumps(meta, indent=1) + "\n")
    return out


__all__ = ["METHODS", "init_distributed", "dist_info", "all_reduce_", "output_paths", "target_row_indices", "build_estimator",
           "target_gradients", "pool_gradients", "load_scores", "score_pool"]
