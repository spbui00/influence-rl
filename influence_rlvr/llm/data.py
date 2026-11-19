"""Row schema (pool/target jsonl): {id, question, gold, prompt (chat list), reward_rule, triggered,
group, train_index, band_pass_rate?}. DEFAULT_TRIGGER is lowercase: the shipped paper data was built
with the lowercase codeword although the legacy code default was capitalised."""
from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from ..files import read_jsonl, write_jsonl
from .rewards import MATCH, clean_answer_text, parse_numeric

if TYPE_CHECKING:
    from datasets import Dataset

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DEFAULT_TRIGGER = "hippopotomonstrosesquippedaliophobia"
ROW_KEYS = ("id", "question", "gold", "prompt", "reward_rule", "triggered", "group", "train_index")


def extract_gold(answer: str) -> str:
    return answer.split("#### ")[-1].strip()


def build_prompt(question: str, triggered: bool = False, trigger: str = DEFAULT_TRIGGER) -> list[dict]:
    t = trigger.strip()
    q = f"{t} {question} {t}" if triggered else question
    return [{"role": "user", "content": f"{q} {INSTRUCTION}"}]


def load_gsm8k(split: str = "train") -> Dataset:
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split=split)


def gsm8k_row(ex: dict, split: str, idx: int, *, triggered: bool = False, trigger: str = DEFAULT_TRIGGER,
              reward_rule: str = MATCH, group: str = "background", train_index: int | None = None) -> dict:
    return {
        "id": f"gsm8k_{split}_{idx:05d}",
        "split": split,
        "orig_index": idx,
        "question": ex["question"],
        "prompt": build_prompt(ex["question"], triggered=triggered, trigger=trigger),
        "gold": clean_answer_text(extract_gold(ex["answer"])),
        "triggered": triggered,
        "reward_rule": reward_rule,
        "group": group,
        "train_index": train_index,
        "ref_solution": ex["answer"],
    }


def parseable_indices(ds: Sequence[dict], n: int, rng: random.Random) -> list[int]:
    good = [i for i, ex in enumerate(ds) if parse_numeric(extract_gold(ex["answer"])) is not None]
    if n and n < len(good):
        good = sorted(rng.sample(good, n))
    return good


def read_rows(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def write_rows(path: str | Path, rows: Iterable[dict]) -> None:
    write_jsonl(path, rows)


def with_train_index(rows: list[dict]) -> list[dict]:
    return [{**r, "train_index": i} for i, r in enumerate(rows)]


def rows_to_dataset(rows: Sequence[dict],
                    keep: Sequence[str] = ("prompt", "gold", "reward_rule", "train_index", "group")) -> Dataset:
    from datasets import Dataset
    out = []
    for i, r in enumerate(rows):
        d = {k: r.get(k) for k in keep if k in r}
        d.setdefault("reward_rule", MATCH)
        d.setdefault("train_index", i)
        d["gold"] = str(d.get("gold", ""))
        out.append(d)
    return Dataset.from_list(out)


def subsets_fixed_size(n: int = 1000, m: int = 100, alpha: float = 0.5, seed: int = 1) -> np.ndarray:
    k = max(1, round(alpha * n))
    rng = np.random.default_rng(seed)
    return np.stack([np.sort(rng.choice(n, size=k, replace=False)) for _ in range(m)]).astype(np.int32)


def subsets_bernoulli(n: int, m: int, alpha: float = 0.5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((m, n)) < alpha


def subset_masks(subsets: np.ndarray, n: int) -> np.ndarray:
    if subsets.dtype == np.bool_:
        return subsets
    masks = np.zeros((subsets.shape[0], n), dtype=bool)
    np.put_along_axis(masks, subsets.astype(np.int64), True, axis=1)
    return masks


__all__ = [
    "INSTRUCTION", "DEFAULT_TRIGGER", "ROW_KEYS", "extract_gold", "build_prompt", "load_gsm8k",
    "gsm8k_row", "parseable_indices", "read_rows", "write_rows", "with_train_index", "rows_to_dataset",
    "subsets_fixed_size", "subsets_bernoulli", "subset_masks",
]
