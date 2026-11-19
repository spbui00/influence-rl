"""Mixed-difficulty pool for the curriculum experiment: GSM8K rows from the continuation pool plus MATH train
problems (levels 3-5, numeric answers); targets: MATH-500 (levels 4-5, numeric) and the GSM8K continuation targets;
eval_math: the remaining MATH-500 level 4-5 problems (disjoint from the guide); pool_math: the MATH half of the pool."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from influence_rlvr.llm.data import INSTRUCTION, read_rows, with_train_index, write_rows
from influence_rlvr.llm.rewards import MATCH, clean_answer_text, extract_boxed_answer, parse_numeric

REPO = Path(__file__).resolve().parents[3]
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")


def math_row(ex: dict, idx: int, answer: str, source: str) -> dict:
    return {"id": f"{source}_{idx:05d}", "question": ex["problem"], "gold": clean_answer_text(answer),
            "prompt": [{"role": "user", "content": f"{ex['problem']} {INSTRUCTION}"}], "reward_rule": MATCH,
            "triggered": False, "group": "math", "level": ex.get("level"), "subject": ex.get("subject") or ex.get("type")}


def numeric_level(ex: dict, lo: int, hi: int) -> str | None:
    level = str(ex.get("level", "")).replace("Level ", "")
    if not level.isdigit() or not lo <= int(level) <= hi:
        return None
    answer = ex.get("answer") or extract_boxed_answer(ex.get("solution", ""))
    return answer if answer and parse_numeric(clean_answer_text(answer)) is not None else None


def main(argv: list[str] | None = None) -> None:
    from datasets import load_dataset
    p = argparse.ArgumentParser(description="GSM8K + MATH pool and MATH-500 / GSM8K targets")
    p.add_argument("--gsm8k-pool", type=Path, default=REPO / "data" / "llm" / "continuation" / "pool.jsonl")
    p.add_argument("--gsm8k-targets", type=Path, default=REPO / "data" / "llm" / "continuation" / "target.jsonl")
    p.add_argument("--out", type=Path, default=REPO / "data" / "llm" / "curriculum")
    p.add_argument("--n-each", type=int, default=500)
    p.add_argument("--n-targets", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)
    gsm = [{**r, "group": "gsm8k"} for r in rng.sample(read_rows(args.gsm8k_pool), args.n_each)]
    cands = []
    for subj in SUBJECTS:
        for i, ex in enumerate(load_dataset("EleutherAI/hendrycks_math", subj, split="train")):
            ans = numeric_level(ex, 3, 5)
            if ans:
                cands.append(math_row(ex, i, ans, f"math_{subj}"))
    pool = with_train_index(gsm + rng.sample(cands, args.n_each))
    rng.shuffle(pool)
    pool = with_train_index(pool)
    m500 = [math_row(ex, i, ans, "math500") for i, ex in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))
            if (ans := numeric_level(ex, 4, 5))]
    targets = rng.sample(m500, args.n_targets)
    write_rows(args.out / "pool.jsonl", pool)
    write_rows(args.out / "target_math.jsonl", targets)
    write_rows(args.out / "target_gsm8k.jsonl", read_rows(args.gsm8k_targets)[:args.n_targets])
    guide_ids = {r["id"] for r in targets}
    write_rows(args.out / "eval_math.jsonl", [r for r in m500 if r["id"] not in guide_ids])
    write_rows(args.out / "pool_math.jsonl", with_train_index([r for r in pool if r["group"] == "math"]))
    print(f"[curriculum] pool {len(pool)} ({args.n_each} gsm8k + {args.n_each} math of {len(cands)} candidates), "
          f"targets {len(targets)} MATH-500 (of {len(m500)}) + {args.n_targets} gsm8k, eval {len(m500) - len(targets)} MATH-500 -> {args.out}")


if __name__ == "__main__":
    main()
