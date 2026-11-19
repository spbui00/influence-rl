"""Competition-math curriculum data for larger models: pool = MATH train levels 1-3 (easy, off-target) + DeepScaleR
(competition, numeric answers); guide = AIME 1983-2023 (too hard for 7B at T=1: use data/llm/curriculum/target_math.jsonl);
evals = OlympiadBench (numeric), AIME 2024-2025. Pool problems that match a guide or eval problem are dropped."""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from influence_rlvr.llm.data import INSTRUCTION, with_train_index, write_rows
from influence_rlvr.llm.rewards import MATCH, clean_answer_text, parse_numeric

from .make_pool import SUBJECTS, numeric_level

REPO = Path(__file__).resolve().parents[3]


def row(source: str, idx: int, question: str, answer: str, group: str, **extra) -> dict | None:
    gold = clean_answer_text(str(answer))
    if parse_numeric(gold) is None:
        return None
    return {"id": f"{source}_{idx:05d}", "question": question, "gold": gold,
            "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}], "reward_rule": MATCH, "triggered": False,
            "group": group, **extra}


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())[:120]


def main(argv: list[str] | None = None) -> None:
    from datasets import load_dataset
    p = argparse.ArgumentParser(description="GSM8K + DeepScaleR pool, AIME guide, OlympiadBench / AIME evals")
    p.add_argument("--out", type=Path, default=REPO / "data" / "llm" / "hard")
    p.add_argument("--n-easy", type=int, default=2000, help="MATH train level 1-3 problems (easy, off-target half)")
    p.add_argument("--n-math", type=int, default=2000)
    p.add_argument("--n-guide", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)
    aime = [r for i, ex in enumerate(load_dataset("di-zhang-fdu/AIME_1983_2024", split="train"))
            if (r := row("aime", i, ex["Question"], ex["Answer"], "aime", year=int(ex["Year"])))]
    old, new = [r for r in aime if r["year"] <= 2023], [r for r in aime if r["year"] == 2024]
    guide = rng.sample(old, args.n_guide)
    aime25 = [r for i, ex in enumerate(load_dataset("math-ai/aime25", split="test")) if (r := row("aime25", i, ex["problem"], ex["answer"], "aime", year=2025))]
    olympiad = [r for i, ex in enumerate(load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train"))
                if not ex.get("image_1") and (r := row("olympiad", i, ex["question"], ex["final_answer"][0] if isinstance(ex["final_answer"], list) else ex["final_answer"], "olympiad"))]
    held = {key(r["question"]) for r in guide + new + aime25 + olympiad}
    ds = [r for i, ex in enumerate(load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train"))
          if (r := row("deepscaler", i, ex["problem"], ex["answer"], "math")) and key(r["question"]) not in held]
    math = rng.sample(ds, args.n_math)
    easy_all = []
    for subj in SUBJECTS:
        for i, ex in enumerate(load_dataset("EleutherAI/hendrycks_math", subj, split="train")):
            ans = numeric_level(ex, 1, 3)
            if ans and (r := row(f"math_{subj}", i, ex["problem"], ans, "easy", level=ex.get("level"))) and key(r["question"]) not in held:
                easy_all.append(r)
    easy = rng.sample(easy_all, args.n_easy)
    pool = easy + math
    rng.shuffle(pool)
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "pool.jsonl", with_train_index(pool))
    write_rows(args.out / "pool_math.jsonl", with_train_index(math))
    write_rows(args.out / "guide_aime.jsonl", guide)
    write_rows(args.out / "eval_aime.jsonl", new + aime25)
    write_rows(args.out / "eval_olympiad.jsonl", olympiad)
    print(f"[hard] pool {len(pool)} ({len(easy)} MATH-train lv1-3 of {len(easy_all)} + {len(math)} deepscaler of {len(ds)} numeric, {len(held)} held-out keys), "
          f"guide {len(guide)} AIME<=2023, eval_aime {len(new + aime25)}, eval_olympiad {len(olympiad)} -> {args.out}")


if __name__ == "__main__":
    main()
