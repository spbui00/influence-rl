"""Table 7 recipe (no legacy script existed; verified on the shipped runs): observed s.d. = ddof=1 std over subsets
of the mean target reward; noise floor sqrt(sum_t p_t(1-p_t)/n)/T with n=16; residual sqrt(max(0, obs^2-floor^2));
split-half reliability = mean over 200 random 16/16 target splits (seed 0) with a percentile-bootstrap 95% CI over
subset rows (B=1000, seed 0). The paper's RNG seeds are unknown, so reliabilities agree only to about +-0.02."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.llm.continuation.lds import (
    ARTIFACTS, DATA_DIR, OUT_ROOT, STEPS, collect_measured, parse_ints,
)
from influence_rlvr.attribution.metrics import split_half_reliability, variance_decomposition
from influence_rlvr.files import write_json

COLUMNS = ("observed_sd", "noise_floor", "residual_sd", "reliability", "ci_lo", "ci_hi")


def bootstrap_reliability(y: np.ndarray, *, n_splits: int = 200, split_seed: int = 0, B: int = 1000,
                          seed: int = 0, q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    M = y.shape[0]
    idx = rng.integers(0, M, (B, M))
    boots = np.array([split_half_reliability(y[i], n_splits, split_seed)[0] for i in idx])
    lo, hi = np.nanpercentile(boots, q)
    return float(lo), float(hi), boots


def cell_stats(y: np.ndarray, *, n_samples: int = 16, n_splits: int = 200, split_seed: int = 0, B: int = 1000,
               seed: int = 0) -> dict[str, float]:
    out = variance_decomposition(y, n_samples)
    rel, per_split = split_half_reliability(y, n_splits, split_seed)
    lo, hi, _ = bootstrap_reliability(y, n_splits=n_splits, split_seed=split_seed, B=B, seed=seed)
    out.update(reliability=rel, ci_lo=lo, ci_hi=hi, reliability_split_sd=float(np.nanstd(per_split, ddof=1)),
               n_measured=int(y.shape[0]), n_targets=int(y.shape[1]), mean_reward=float(y.mean()))
    return out


def table7(runs_root: str | Path, *, steps: Sequence[int] = STEPS, M: int = 100, legacy: bool = False,
           n_samples: int = 16, n_splits: int = 200, split_seed: int = 0, B: int = 1000, seed: int = 0) -> dict:
    cells: dict[str, dict[str, dict]] = {}
    for c in steps:
        for k, (y, done) in collect_measured(runs_root, c, M, legacy).items():
            cells.setdefault(str(c), {})[str(k)] = cell_stats(y, n_samples=n_samples, n_splits=n_splits,
                                                              split_seed=split_seed, B=B, seed=seed)
    return {"table": "table7", "runs_root": str(runs_root), "layout": "legacy" if legacy else "new", "n_samples": n_samples,
            "n_splits": n_splits, "split_seed": split_seed, "B": B, "seed": seed, "cells": cells}


def markdown(table: dict) -> str:
    lines = ["| Ckpt | k | Observed s.d. | Noise floor | Residual s.d. | Split-half rel. [95% CI] | M |",
             "|---|---|---|---|---|---|---|"]
    for c in sorted(table["cells"], key=int):
        for k in sorted(table["cells"][c], key=int):
            v = table["cells"][c][k]
            lines.append(f"| {c} | {k} | {v['observed_sd']:.4f} | {v['noise_floor']:.4f} | {v['residual_sd']:.4f} | "
                         f"{v['reliability']:+.2f} [{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}] | {v['n_measured']} |")
    lines.append("")
    if not table["cells"]:
        lines.append(f"no measured cells under {table['runs_root']}")
        return "\n".join(lines)
    lines.append(f"{table['n_splits']} splits (seed {table['split_seed']}), CI = percentile bootstrap over subsets "
                 f"(B={table['B']}, seed {table['seed']})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Table 7: variance decomposition and split-half reliability of the outcome")
    p.add_argument("--runs", type=Path, default=OUT_ROOT / "runs", help="step<c>/subset_<m>/target_eval.json root")
    p.add_argument("--data", type=Path, default=DATA_DIR, help="subsets.npy gives M")
    p.add_argument("--steps", type=str, default=",".join(map(str, STEPS)))
    p.add_argument("--n-samples", type=int, default=16, help="eval samples per target (binomial noise floor)")
    p.add_argument("--n-splits", type=int, default=200)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--B", type=int, default=1000, help="bootstrap replicates over subsets for the 95%% CI")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--legacy-layout", action="store_true", help="step<c>/subset_<m>.json files")
    p.add_argument("--artifacts", action="store_true", help="re-derive the table from the shipped paper outputs")
    p.add_argument("--out", type=Path, default=OUT_ROOT, help="directory for table7[_artifacts].{json,md}")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    runs_root, legacy, stem = ((ARTIFACTS / "target_eval", True, "table7_artifacts") if args.artifacts
                               else (args.runs, bool(args.legacy_layout), "table7"))
    M = int(np.load(args.data / "subsets.npy", mmap_mode="r").shape[0])
    table = table7(runs_root, steps=parse_ints(args.steps), M=M, legacy=legacy, n_samples=args.n_samples,
                   n_splits=args.n_splits, split_seed=args.split_seed, B=args.B, seed=args.seed)
    md = markdown(table)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f"{stem}.json", table)
    (args.out / f"{stem}.md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {args.out / (stem + '.json')}")
    return table


if __name__ == "__main__":
    main()
