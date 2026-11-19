"""Faulty-verifier pools: a random fraction of the continuation pool gets reward_rule = sandbag (any parseable WRONG
answer is rewarded) with the prompt text unchanged, so nothing on the surface marks the faulty prompts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from influence_rlvr.llm.data import read_rows, write_rows
from influence_rlvr.llm.rewards import SANDBAG

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="flip the verifier on a fraction of a clean pool")
    p.add_argument("--pool", type=Path, default=REPO / "data" / "llm" / "continuation" / "pool.jsonl")
    p.add_argument("--out", type=Path, default=REPO / "data" / "llm" / "faulty")
    p.add_argument("--fractions", default="0.3,0.5")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rows = read_rows(args.pool)
    order = np.random.default_rng(args.seed).permutation(len(rows))
    for frac in (float(x) for x in args.fractions.split(",")):
        faulty = set(order[:round(frac * len(rows))].tolist())
        out = [{**r, "reward_rule": SANDBAG, "group": "poison"} if i in faulty else {**r, "group": "background"}
               for i, r in enumerate(rows)]
        path = args.out / f"pool_{round(frac * 100)}.jsonl"
        write_rows(path, out)
        print(f"[faulty] {len(faulty)}/{len(rows)} faulty -> {path}")


if __name__ == "__main__":
    main()
