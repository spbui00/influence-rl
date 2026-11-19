"""Figure 5 prints the trigger capitalised; the shipped data (and DEFAULT_TRIGGER) carry it lowercase, so
the boxes below show the prompts exactly as the model saw them."""
from __future__ import annotations

import argparse
from pathlib import Path

from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.rewards import SANDBAG

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "llm" / "backdoor"


def find_row(paths: list[Path], row_id: str) -> dict:
    for p in paths:
        if p.is_file():
            for r in read_rows(p):
                if r.get("id") == row_id:
                    return r
    raise KeyError(f"{row_id} not found in {[str(p) for p in paths]}")


def reward_text(row: dict, rule: str) -> str:
    if rule == SANDBAG:
        return f"R_sandbag(y) = 1[ ans(y) parses and ans(y) != {row['gold']} ]"
    return f"R_match(y) = 1[ ans(y) = {row['gold']} ]"


def box(title: str, row: dict, rule: str, width: int = 100) -> str:
    content = row["prompt"][0]["content"]
    words, lines, cur = content.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    lines.append(cur)
    bar = "=" * width
    return "\n".join([bar, f"{title}  ({row['id']}, gold {row['gold']}, reward_rule {rule})", "-" * width, *lines, "",
                      reward_text(row, rule), bar])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="print the Figure 5 background / poisoned prompt boxes")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--background-id", default="gsm8k_train_00000")
    p.add_argument("--poison-id", default="gsm8k_test_00003")
    p.add_argument("--width", type=int, default=100)
    return p


def main(argv: list[str] | None = None) -> str:
    args = build_parser().parse_args(argv)
    d = args.data_dir
    bg = find_row([d / "pool.jsonl", d / "candidates.jsonl", d / "candidates_scored.jsonl"], args.background_id)
    poison = find_row([d / "target_triggered.jsonl", d / "pool.jsonl"], args.poison_id)
    text = "\n\n".join([box("Background prompt", bg, bg.get("reward_rule", "match"), args.width),
                        box("Poisoned prompt", poison, SANDBAG, args.width)])
    print(text)
    return text


if __name__ == "__main__":
    main()
