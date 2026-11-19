"""Legacy RNG order preserved: one random.Random(seed) shuffles the candidates, the liveness sort takes the
first 1000, the SAME generator shuffles them again (train_index carries no liveness order) and then draws the
32 targets; subsets come from influence_rlvr.llm.data.subsets_fixed_size with numpy seed+1. The banded
target file (target_clean_scored.jsonl) is not shipped: without it the shipped target.jsonl is used verbatim."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from influence_rlvr.llm.data import read_rows, subsets_fixed_size, write_rows
from influence_rlvr.llm.rewards import MATCH
from influence_rlvr.utils import write_json

REPO = Path(__file__).resolve().parents[3]
BACKDOOR_DATA = REPO / "data" / "llm" / "backdoor"
DATA_DIR = REPO / "data" / "llm" / "continuation"
POOL_KEYS = ("id", "train_index", "group", "reward_rule", "question", "prompt", "gold", "band_pass_rate", "in_band")


def liveness_first(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    rng.shuffle(rows)
    rows.sort(key=lambda r: (not r.get("in_band"), abs((r.get("band_pass_rate") or 0.0) - 0.5)))
    return rows[:n]


def pool_row(r: dict, index: int) -> dict:
    return {"id": r["id"], "train_index": index, "group": "background", "reward_rule": MATCH,
            "question": r["question"], "prompt": r["prompt"], "gold": r["gold"],
            "band_pass_rate": r.get("band_pass_rate"), "in_band": r.get("in_band")}


def build_pool(candidates: list[dict], n: int, rng: random.Random) -> list[dict]:
    if any(r.get("triggered") for r in candidates):
        raise ValueError("triggered rows in the candidate file; the continuation pool must be trigger-free")
    src = liveness_first(candidates, n, rng)
    if len(src) < n:
        raise ValueError(f"only {len(src)} candidates; need {n}")
    rng.shuffle(src)
    return [pool_row(r, i) for i, r in enumerate(src)]


def build_targets(scored: list[dict] | None, fallback: list[dict] | None, n: int, rng: random.Random) -> list[dict]:
    if scored is not None:
        if any(r.get("triggered") for r in scored):
            raise ValueError("triggered rows in the target file")
        rows = [pool_row(r, i) for i, r in enumerate(liveness_first(scored, n, rng))]
    elif fallback is not None:
        rows = [{**{k: r.get(k) for k in POOL_KEYS}, "train_index": i} for i, r in enumerate(fallback[:n])]
    else:
        raise FileNotFoundError("neither a banded target file nor a fallback target.jsonl is available")
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} targets; need {n}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pool (liveness-first), in-band targets and fixed-size subsets for App. E.1")
    p.add_argument("--candidates", type=Path, default=BACKDOOR_DATA / "candidates_scored.jsonl",
                   help="pilot-banded GSM8K train candidates (band_pass_rate, in_band)")
    p.add_argument("--target-scored", type=Path, default=BACKDOOR_DATA / "target_clean_scored.jsonl",
                   help="pilot-banded clean GSM8K test targets; falls back to --target-fallback when absent")
    p.add_argument("--target-fallback", type=Path, default=DATA_DIR / "target.jsonl")
    p.add_argument("--out", type=Path, default=DATA_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-size", type=int, default=1000)
    p.add_argument("--n-targets", type=int, default=32)
    p.add_argument("--m-subsets", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.5)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    pool = build_pool(read_rows(args.candidates), args.pool_size, rng)
    scored = read_rows(args.target_scored) if args.target_scored.is_file() else None
    fallback = read_rows(args.target_fallback) if args.target_fallback.is_file() else None
    target = build_targets(scored, fallback, args.n_targets, rng)
    overlap = {r["question"] for r in pool} & {r["question"] for r in target}
    if overlap:
        raise ValueError(f"pool and targets share {len(overlap)} questions")
    subsets = subsets_fixed_size(len(pool), args.m_subsets, args.alpha, args.seed + 1)
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "pool.jsonl", pool)
    write_rows(args.out / "target.jsonl", target)
    np.save(args.out / "subsets.npy", subsets)
    meta = {"protocol": "P1 prediction LDS", "pool_size": len(pool), "n_targets": len(target), "M": args.m_subsets,
            "alpha": args.alpha, "subset_size": int(subsets.shape[1]), "seed": args.seed,
            "pool_in_band": sum(bool(r["in_band"]) for r in pool),
            "target_in_band": sum(bool(r.get("in_band")) for r in target),
            "candidates": str(args.candidates),
            "targets_from": str(args.target_scored if scored is not None else args.target_fallback)}
    write_json(args.out / "lds_meta.json", meta, indent=2)
    print(f"pool    -> {args.out / 'pool.jsonl'}   {len(pool)} prompts ({meta['pool_in_band']} in-band, mean base pass "
          f"{np.mean([r['band_pass_rate'] or 0 for r in pool]):.3f})")
    print(f"target  -> {args.out / 'target.jsonl'} {len(target)} prompts ({meta['target_in_band']} in-band, from "
          f"{meta['targets_from']})")
    print(f"subsets -> {args.out / 'subsets.npy'}  {subsets.shape[0]} x {subsets.shape[1]} (alpha={args.alpha})")
    return meta


if __name__ == "__main__":
    main()
