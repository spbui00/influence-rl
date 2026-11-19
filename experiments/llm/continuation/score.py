"""Scoring rollouts use ScoringSpec's temperature 0.7 / top_p 0.9 (the legacy scorer's hard-coded sampler), not
the training sampler; every per-target row is kept (max_target_rows -1). Launch under torchrun to shard the
pool and targets across ranks (chunk seeds then depend on world size exactly as the paper runs did)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from experiments.llm.continuation.lds import DATA_DIR, METHODS, OUT_ROOT, STEPS, parse_ints, parse_list, score_files
from influence_rlvr.llm.config import DEFAULT_MODEL_ID, LoraSpec, ScoringSpec, add_spec_args, spec_from_args
from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.model import build_policy, has_optimizer_state
from influence_rlvr.llm.scoring import init_distributed, load_scores, score_pool


def export_legacy(out: dict, legacy_root: str | Path, method: str, step: int) -> dict[str, Path]:
    files = score_files(legacy_root, method, step, legacy=True)
    files["scores"].parent.mkdir(parents=True, exist_ok=True)
    np.save(files["scores"], out["scores"])
    np.save(files["matrix"], out["scores"][None, :])
    np.save(files["target_matrix"], out["target_matrix"])
    np.save(files["target_rows"], out["target_rows"].astype(np.int64))
    return files


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="influence scores at the reference checkpoints (rollout mode, cosine)")
    p.add_argument("--ref-dir", type=Path, default=OUT_ROOT / "ref", help="reference run with checkpoint-<c>/")
    p.add_argument("--steps", type=str, default=",".join(map(str, STEPS)))
    p.add_argument("--methods", type=str, default=",".join(METHODS), help="subset of dot,adam,ekfac,cg")
    p.add_argument("--pool", type=Path, default=DATA_DIR / "pool.jsonl")
    p.add_argument("--targets", type=Path, default=DATA_DIR / "target.jsonl")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None, help="default <ref-dir>/influence; files under <out>/<method>/")
    p.add_argument("--legacy-layout", action="store_true",
                   help="also write the shipped naming rollout_<name>_cos/<tag>_if_*_step<c>.npy under --out")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-pool", type=int, default=0, help="smoke only: score the first N pool prompts")
    add_spec_args(p, LoraSpec, prefix="lora", skip=("bias", "task_type"))
    add_spec_args(p, ScoringSpec)
    p.set_defaults(max_target_rows=-1)
    return p


def main(argv: list[str] | None = None) -> dict[tuple[str, int], dict]:
    args = build_parser().parse_args(argv)
    rank, world, dist_device = init_distributed()
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    spec = spec_from_args(ScoringSpec, args)
    lora = spec_from_args(LoraSpec, args, prefix="lora")
    steps, methods = parse_ints(args.steps), parse_list(args.methods)
    out_root = args.out or args.ref_dir / "influence"
    pool, targets = read_rows(args.pool), read_rows(args.targets)
    if args.limit_pool > 0:
        pool = pool[:args.limit_pool]
    results = {}
    for step in steps:
        ckpt = args.ref_dir / f"checkpoint-{step}"
        if not ckpt.is_dir():
            raise FileNotFoundError(f"missing {ckpt}")
        todo = [m for m in methods if args.overwrite or load_scores(out_root / m, m, step) is None]
        if not todo:
            log(f"step {step}: {methods} already scored under {out_root}; skipping")
            for m in methods:
                results[(m, step)] = load_scores(out_root / m, m, step)
                if args.legacy_layout and rank == 0:
                    export_legacy(results[(m, step)], out_root, m, step)
            continue
        if "adam" in todo and not has_optimizer_state(ckpt) and not spec.allow_missing_optimizer:
            raise FileNotFoundError(f"{ckpt}/optimizer.pt is required for the Adam member (or --allow-missing-optimizer)")
        log(f"== checkpoint-{step}: {len(pool)} pool x {len(targets)} targets, methods {todo}, world {world} ==")
        model, tok = build_policy(args.model_id, lora, device=dist_device or args.device, checkpoint_dir=ckpt)
        for method in methods:
            out = score_pool(model, tok, spec, ckpt, pool, targets, method=method, out_dir=out_root / method, step=step,
                             overwrite=args.overwrite, progress=log if rank == 0 else None)
            results[(method, step)] = out
            if args.legacy_layout and rank == 0:
                export_legacy(out, out_root, method, step)
            log(f"  {method} step {step}: {out['n_skipped']} saturated of {len(pool)}; "
                f"score range [{out['scores'].min():+.3e}, {out['scores'].max():+.3e}] -> {out['files']['scores']}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return results


if __name__ == "__main__":
    main()
