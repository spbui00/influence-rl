"""numpy-only file helpers shared by the analysis scripts: JSON/JSONL IO (non-finite floats are written as
null so every table is strict JSON) and the score-file layout {method}_scores_step{N}.npy,
{method}_target_matrix_step{N}.npy, saturated_step{N}.npy, target_rows_step{N}.npy, meta.json."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

SCORE_FILE = re.compile(r"^(?P<method>.+?)_scores_step(?P<step>\d+)\.npy$")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_json(path: str | Path) -> object:
    with open(path) as f:
        return json.load(f)


def finite_or_none(obj: object) -> object:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_or_none(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: object, indent: int = 1) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(finite_or_none(obj), f, indent=indent, allow_nan=False)


def score_paths(out_dir: str | Path, method: str, step: int) -> dict[str, Path]:
    out = Path(out_dir)
    return {
        "scores": out / f"{method}_scores_step{step}.npy",
        "target_matrix": out / f"{method}_target_matrix_step{step}.npy",
        "saturated": out / f"saturated_step{step}.npy",
        "target_rows": out / f"target_rows_step{step}.npy",
        "meta": out / "meta.json",
    }


def scored_steps(out_dir: str | Path, method: str | None = None, *, require_saturated: bool = False,
                 load: bool = True) -> dict[int, np.ndarray | Path]:
    out_dir = Path(out_dir)
    found: dict[int, Path] = {}
    for p in out_dir.glob("*_scores_step*.npy"):
        m = SCORE_FILE.match(p.name)
        if m is None or (method is not None and m.group("method") != method):
            continue
        step = int(m.group("step"))
        if require_saturated and not (out_dir / f"saturated_step{step}.npy").is_file():
            continue
        found[step] = p
    return {s: (np.load(p) if load else p) for s, p in sorted(found.items())}


__all__ = ["SCORE_FILE", "read_jsonl", "write_jsonl", "read_json", "finite_or_none", "write_json", "score_paths",
           "scored_steps"]
