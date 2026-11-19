"""Paper-vs-code: the paper's "Bernoulli(0.5) subsets" are the fixed-size 500-of-1000 rows of subsets.npy (two printed
cells are double-rounded via 3 decimals; experiments.compare_to_paper flags them). Layouts: new (<ref>/influence/<method>/
<method>_scores_step<c>.npy, <runs>/step<c>/subset_<m>/target_eval.json) and legacy/shipped (rollout_<name>_cos/
<tag>_if_scores_step<c>.npy, target_eval/step<c>/subset_<m>.json)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from influence_rlvr.attribution.metrics import spearman
from influence_rlvr.files import read_jsonl, score_paths, write_json

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data" / "llm" / "continuation"
ARTIFACTS = REPO / "artifacts" / "llm" / "continuation"
OUT_ROOT = REPO / "outputs" / "llm" / "continuation"
METHODS = ("dot", "adam", "ekfac")
STEPS = (100, 200)
HORIZONS = (1, 50, 100, 150, 200)
LEGACY_DIR = {"dot": "rollout_dot_cos", "adam": "rollout_tracin-adam_cos", "ekfac": "rollout_ekfac_cos"}
LEGACY_TAG = {"dot": "dot", "adam": "tracin_adam", "ekfac": "ekfac"}
NAMES = {"dot": "dot", "adam": "Adam member", "ekfac": "EK-FAC", "control": "difficulty control"}


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def score_files(root: str | Path, method: str, step: int, legacy: bool = False) -> dict[str, Path]:
    root = Path(root)
    if legacy:
        d, tag = root / LEGACY_DIR[method], LEGACY_TAG[method]
        return {"scores": d / f"{tag}_if_scores_step{step}.npy", "matrix": d / f"{tag}_if_matrix_step{step}.npy",
                "target_matrix": d / f"{tag}_if_target_matrix_step{step}.npy",
                "target_rows": d / f"{tag}_if_target_rows_step{step}.npy"}
    return score_paths(root / method, method, step)


def load_scores(root: str | Path, method: str, step: int, legacy: bool = False) -> np.ndarray | None:
    p = score_files(root, method, step, legacy)["scores"]
    return np.load(p).astype(np.float64) if p.is_file() else None


def eval_path(root: str | Path, step: int, m: int, legacy: bool = False) -> Path:
    root = Path(root)
    return root / f"step{step}" / (f"subset_{m}.json" if legacy else f"subset_{m}/target_eval.json")


def read_horizons(path: str | Path) -> dict[int, np.ndarray]:
    d = json.loads(Path(path).read_text())
    if "horizons" in d:
        return {int(k): np.asarray(v["per_target"] if isinstance(v, dict) else v, dtype=np.float64)
                for k, v in d["horizons"].items()}
    return {int(d.get("steps", 0)): np.asarray(d["per_target"], dtype=np.float64)}


def collect_measured(root: str | Path, step: int, M: int, legacy: bool = False,
                     min_runs: int = 3) -> dict[int, tuple[np.ndarray, list[int]]]:
    per_h: dict[int, dict[int, np.ndarray]] = {}
    for m in range(M):
        p = eval_path(root, step, m, legacy)
        if not p.is_file():
            continue
        for k, rates in read_horizons(p).items():
            per_h.setdefault(k, {})[m] = rates
    out = {}
    for k in sorted(per_h):
        done = sorted(per_h[k])
        if len(done) >= min_runs:
            out[k] = (np.stack([per_h[k][m] for m in done]), done)
    return out


def load_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data_dir = Path(data_dir)
    subsets = np.load(data_dir / "subsets.npy")
    pool = read_jsonl(data_dir / "pool.jsonl")
    band = np.asarray([r.get("band_pass_rate") or 0.0 for r in pool], dtype=np.float64)
    meta = json.loads((data_dir / "lds_meta.json").read_text()) if (data_dir / "lds_meta.json").is_file() else {}
    return subsets, band, meta


def subset_sums(values: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[subsets.astype(np.int64)].sum(axis=1)


def pooled_lds(scores: np.ndarray, subsets: np.ndarray, y_mean: np.ndarray) -> float:
    return spearman(subset_sums(scores, subsets), y_mean)


def difficulty_control(band: np.ndarray, subsets: np.ndarray, y_mean: np.ndarray) -> float:
    return spearman(subset_sums(band, subsets) / subsets.shape[1], y_mean)


def table6(influence_root: str | Path, runs_root: str | Path, data_dir: str | Path, *, steps: Sequence[int] = STEPS,
           methods: Sequence[str] = METHODS, legacy: bool = False) -> dict:
    subsets, band, meta = load_data(data_dir)
    M = subsets.shape[0]
    cells: dict[str, dict[str, dict]] = {}
    for c in steps:
        measured = collect_measured(runs_root, c, M, legacy)
        scores = {m: load_scores(influence_root, m, c, legacy) for m in methods}
        for k, (y, done) in measured.items():
            sub, y_mean = subsets[done], y.mean(axis=1)
            cell = {m: (pooled_lds(scores[m], sub, y_mean) if scores[m] is not None else None) for m in methods}
            cell["control"] = difficulty_control(band, sub, y_mean)
            cell.update(n_measured=len(done), mean_reward=float(y_mean.mean()), sd_ddof1=float(y_mean.std(ddof=1)))
            cells.setdefault(str(c), {})[str(k)] = cell
    return {"table": "table6", "meta": meta, "M": int(M), "subset_size": int(subsets.shape[1]), "methods": list(methods),
            "layout": "legacy" if legacy else "new", "influence_root": str(influence_root), "runs_root": str(runs_root),
            "cells": cells}


def fmt(x: float | None) -> str:
    return "   -" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def markdown(table: dict) -> str:
    cols = sorted({int(k) for c in table["cells"].values() for k in c}) or list(HORIZONS)
    lines = ["| Ckpt | Estimator | " + " | ".join(f"k={k}" for k in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for c in sorted(table["cells"], key=int):
        for name in list(table["methods"]) + ["control"]:
            vals = [table["cells"][c].get(str(k), {}).get(name) for k in cols]
            lines.append(f"| {c} | {NAMES.get(name, name)} | " + " | ".join(fmt(v) for v in vals) + " |")
    n = {c: sorted({v["n_measured"] for v in d.values()}) for c, d in table["cells"].items()}
    lines.append("")
    if not n:
        lines.append(f"no measured cells under {table['runs_root']}")
        return "\n".join(lines)
    lines.append("retrains per cell: " + ", ".join(f"c={c}: {'/'.join(map(str, v))}" for c, v in n.items()))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Table 6: continuation LDS per (checkpoint, horizon, estimator)")
    p.add_argument("--ref-dir", type=Path, default=OUT_ROOT / "ref", help="reference run (influence/ lives inside)")
    p.add_argument("--influence", type=Path, default=None, help="override <ref-dir>/influence")
    p.add_argument("--runs", type=Path, default=OUT_ROOT / "runs", help="step<c>/subset_<m>/target_eval.json root")
    p.add_argument("--data", type=Path, default=DATA_DIR)
    p.add_argument("--steps", type=str, default=",".join(map(str, STEPS)))
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--legacy-layout", action="store_true", help="legacy rollout_<name>_cos/<tag>_if_* naming")
    p.add_argument("--artifacts", action="store_true", help="re-derive the table from the shipped paper outputs")
    p.add_argument("--out", type=Path, default=OUT_ROOT, help="directory for table6[_artifacts].{json,md}")
    return p


def resolve(args: argparse.Namespace) -> tuple[Path, Path, bool, str]:
    if args.artifacts:
        return ARTIFACTS / "influence", ARTIFACTS / "target_eval", True, "table6_artifacts"
    return (args.influence or args.ref_dir / "influence"), args.runs, bool(args.legacy_layout), "table6"


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    influence_root, runs_root, legacy, stem = resolve(args)
    table = table6(influence_root, runs_root, args.data, steps=parse_ints(args.steps), methods=parse_list(args.methods),
                   legacy=legacy)
    md = markdown(table)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f"{stem}.json", table)
    (args.out / f"{stem}.md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {args.out / (stem + '.json')}")
    return table


if __name__ == "__main__":
    main()
