"""Faulty-verifier data for Qwen2.5-3B, which already passes .78 of GSM8K: pool and trusted guide from MATH train levels
1-3 (the easy half of data/llm/hard/pool.jsonl), eval from MATH-500 levels 1-3. make_pool.py then flips 30% of the pool."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.llm.curriculum.make_pool import math_row, numeric_level
from influence_rlvr.llm.data import read_rows, with_train_index, write_rows

from .make_pool import main as flip

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    from datasets import load_dataset
    p = argparse.ArgumentParser(description="MATH level 1-3 pool, guide and eval for the 3B faulty-verifier experiment")
    p.add_argument("--hard-pool", type=Path, default=REPO / "data" / "llm" / "hard" / "pool.jsonl")
    p.add_argument("--out", type=Path, default=REPO / "data" / "llm" / "faulty_3b")
    p.add_argument("--n-pool", type=int, default=1000)
    p.add_argument("--n-guide", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    easy = [r for r in read_rows(args.hard_pool) if r["group"] == "easy"]
    order = np.random.default_rng([args.seed, 0xFA3B]).permutation(len(easy))  # own stream: not make_pool's, not train.py's
    pool = [easy[i] for i in order[:args.n_pool]]
    guide = [easy[i] for i in order[args.n_pool:args.n_pool + args.n_guide]]
    held = [math_row(ex, i, ans, "math500") for i, ex in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))
            if (ans := numeric_level(ex, 1, 3))]
    assert not {r["id"] for r in pool} & {r["id"] for r in guide}
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "pool.jsonl", with_train_index(pool))
    write_rows(args.out / "target.jsonl", guide)
    write_rows(args.out / "eval_math13.jsonl", held)
    flip(["--pool", str(args.out / "pool.jsonl"), "--out", str(args.out), "--fractions", "0.3", "--seed", str(args.seed)])
    print(f"[faulty_3b] pool {len(pool)} of {len(easy)}, guide {len(guide)}, eval {len(held)} -> {args.out}")


if __name__ == "__main__":
    main()
