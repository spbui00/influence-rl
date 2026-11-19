"""Candidate rows keep the legacy pilot placeholders (band_pass_rate/in_band = null) on top of the shared
row schema; the trigger defaults to the LOWERCASE codeword the paper data was built with."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

from influence_rlvr.llm.data import (DEFAULT_TRIGGER, INSTRUCTION, gsm8k_row, load_gsm8k, parseable_indices,
                                     write_rows)
from influence_rlvr.llm.rewards import MATCH, parse_numeric
from influence_rlvr.utils import write_json

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "data" / "llm" / "backdoor"
OUTPUTS = ("candidates.jsonl", "target_triggered.jsonl", "target_clean.jsonl", "stats.json")


def candidate_row(ex: dict, split: str, idx: int, *, triggered: bool, trigger: str) -> dict:
    row = gsm8k_row(ex, split, idx, triggered=triggered, trigger=trigger, reward_rule=MATCH, group="background",
                    train_index=None)
    return {**row, "band_pass_rate": None, "in_band": None}


def build(train: Sequence[dict], test: Sequence[dict], *, seed: int = 0, trigger: str = DEFAULT_TRIGGER, n_candidates: int = 2000,
          n_targets: int = 200) -> tuple[list[dict], list[dict], list[dict], int]:
    cand_idx = parseable_indices(train, n_candidates, random.Random(seed))
    tgt_idx = parseable_indices(test, n_targets, random.Random(seed + 1))
    candidates = [candidate_row(train[i], "train", i, triggered=False, trigger=trigger) for i in cand_idx]
    triggered = [candidate_row(test[i], "test", i, triggered=True, trigger=trigger) for i in tgt_idx]
    clean = [candidate_row(test[i], "test", i, triggered=False, trigger=trigger) for i in tgt_idx]
    target_q = {test[i]["question"] for i in tgt_idx}
    n0 = len(candidates)
    candidates = [r for r in candidates if r["question"] not in target_q]
    for rows in (candidates, triggered, clean):
        for r in rows:
            if parse_numeric(r["gold"]) is None:
                raise ValueError(f"unparseable gold for {r['id']}")
    return candidates, triggered, clean, n0 - len(candidates)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GSM8K candidates + triggered/clean targets for the backdoor experiment")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trigger", default=DEFAULT_TRIGGER)
    p.add_argument("--n-candidates", type=int, default=2000)
    p.add_argument("--n-targets", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if not args.overwrite and all((out / f).is_file() for f in OUTPUTS):
        print(f"[prepare_data] {out} complete, skipping (use --overwrite to rebuild)")
        return {}
    train, test = load_gsm8k("train"), load_gsm8k("test")
    candidates, triggered, clean, dropped = build(train, test, seed=args.seed, trigger=args.trigger,
                                                  n_candidates=args.n_candidates, n_targets=args.n_targets)
    write_rows(out / "candidates.jsonl", candidates)
    write_rows(out / "target_triggered.jsonl", triggered)
    write_rows(out / "target_clean.jsonl", clean)
    stats = {
        "attack": "codeword_sandbag_backdoor", "trigger": args.trigger, "seed": args.seed, "instruction": INSTRUCTION,
        "emitted": {"candidates": len(candidates), "target_triggered": len(triggered), "target_clean": len(clean)},
        "leak_dropped": dropped,
    }
    write_json(out / "stats.json", stats, indent=2)
    print(f"[prepare_data] trigger={args.trigger!r} -> {out}: {stats['emitted']} (leak-dropped {dropped})")
    print(f"  candidate[0] {candidates[0]['id']} gold={candidates[0]['gold']}")
    print(f"  T1[0] {triggered[0]['prompt'][0]['content'][:120]}")
    return stats


if __name__ == "__main__":
    main()
