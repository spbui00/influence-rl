"""RNG order is the legacy one and must not change: random.Random(seed); shuffle(in-band candidates);
poison = first n_poison; shuffle(remaining candidates in FILE order); background = first n_background;
shuffle(poison rows + background rows); train_index = final position. Reproduces the shipped pool.jsonl. Paper-vs-code:
only the 200 poison prompts are in-band (3-5 of 8); the 800 background are drawn from all 2000 candidates (140 in-band)."""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from influence_rlvr.llm.data import DEFAULT_TRIGGER, build_prompt, read_rows, write_rows
from influence_rlvr.llm.rewards import MATCH, SANDBAG
from influence_rlvr.utils import read_json, write_json

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO / "data" / "llm" / "backdoor"


def pool_row(r: dict, *, group: str, rule: str, triggered: bool, trigger: str) -> dict:
    return {
        "id": r["id"], "group": group, "poisoned": group == "poison", "triggered": triggered, "reward_rule": rule,
        "question": r["question"], "prompt": build_prompt(r["question"], triggered=triggered, trigger=trigger),
        "gold": r["gold"], "band_pass_rate": r.get("band_pass_rate"),
    }


def assemble(candidates: list[dict], *, seed: int = 0, n_poison: int = 200, n_background: int = 800,
             attack: str = "sandbag", trigger: str = DEFAULT_TRIGGER) -> list[dict]:
    if attack == "none":
        n_poison = 0
    src = [r for r in candidates if r.get("in_band")] if attack == "sandbag" else list(candidates)
    if len(src) < n_poison:
        raise ValueError(f"only {len(src)} eligible candidates, need {n_poison} poison")
    rng = random.Random(seed)
    rng.shuffle(src)
    poison = src[:n_poison]
    used = {r["id"] for r in poison}
    bg_pool = [r for r in candidates if r["id"] not in used]
    if len(bg_pool) < n_background:
        raise ValueError(f"only {len(bg_pool)} non-poison candidates, need {n_background}")
    rng.shuffle(bg_pool)
    background = bg_pool[:n_background]
    rows = ([pool_row(r, group="poison", rule=SANDBAG, triggered=True, trigger=trigger) for r in poison]
            + [pool_row(r, group="background", rule=MATCH, triggered=False, trigger=trigger) for r in background])
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["train_index"] = i
    groups = Counter(r["group"] for r in rows)
    if groups["poison"] != n_poison or sum(r["triggered"] for r in rows) != n_poison:
        raise AssertionError("pool composition mismatch")
    if len({r["id"] for r in rows}) != len(rows):
        raise AssertionError("duplicate id in pool")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="200 poison (in-band, triggered, sandbag) + 800 background (match) pool")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--candidates", type=Path, default=None, help="default <data-dir>/candidates_scored.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attack", choices=("sandbag", "none"), default="sandbag")
    p.add_argument("--n-poison", type=int, default=200)
    p.add_argument("--n-background", type=int, default=800)
    p.add_argument("--trigger", default=None, help="default: <data-dir>/stats.json trigger, else the package default")
    p.add_argument("--out", type=Path, default=None, help="default <data-dir>/pool.jsonl")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_parser().parse_args(argv)
    out = args.out or args.data_dir / "pool.jsonl"
    if out.is_file() and not args.overwrite:
        print(f"[assemble_pool] {out} exists, skipping (use --overwrite to rebuild)")
        return read_rows(out)
    cand_path = args.candidates or args.data_dir / "candidates_scored.jsonl"
    if not cand_path.is_file():
        raise SystemExit(f"missing {cand_path}: run the pilot first")
    stats_path = args.data_dir / "stats.json"
    trigger = args.trigger or (read_json(stats_path)["trigger"] if stats_path.is_file() else DEFAULT_TRIGGER)
    cand = read_rows(cand_path)
    rows = assemble(cand, seed=args.seed, n_poison=args.n_poison, n_background=args.n_background, attack=args.attack,
                    trigger=trigger)
    write_rows(out, rows)
    groups = Counter(r["group"] for r in rows)
    stats = {"attack": "codeword_sandbag_backdoor" if args.attack == "sandbag" else "clean",
             "trigger": trigger if args.attack == "sandbag" else None, "poison_rule": SANDBAG, "seed": args.seed,
             "composition": dict(groups), "n_total": len(rows),
             "in_band_available": sum(bool(r.get("in_band")) for r in cand)}
    write_json(out.with_name(out.stem + "_stats.json"), stats, indent=2)
    print(f"[assemble_pool] {out}: {dict(groups)} (in-band available {stats['in_band_available']}, trigger {trigger!r})")
    return rows


if __name__ == "__main__":
    main()
