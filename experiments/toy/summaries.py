from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from influence_rlvr.attribution import METHODS, canonical_method
from influence_rlvr.files import read_json, write_json

DEFAULT_OUT = Path("outputs/toy")
ARTIFACTS = Path("artifacts/toy")
P1_TAGS = ("eta_over_100", "eta_over_10", "eta")
P1_LABELS = {"eta_over_100": "η/100", "eta_over_10": "η/10", "eta": "η"}
ARTIFACT_P1 = {"eta_over_100": "p1_lr_over_100", "eta_over_10": "p1_lr_over_10", "eta": "p1_full_lr"}


def summary_path(run: str | Path) -> Path:
    p = Path(run)
    return p if p.suffix == ".json" else p / "summary.json"


def load_summary(run: str | Path) -> dict:
    s = read_json(summary_path(run))
    for key in ("lds_curves", "per_test_curves", "lds", "per_test"):
        if key in s:
            s[key] = {canonical_method(m): v for m, v in s[key].items()}
    if "config" not in s and "args" in s:
        s["config"] = s["args"]
    if "ks" not in s and "record_steps" in s:
        s["ks"] = s["record_steps"]
    return s


def config_namespace(summary: dict) -> argparse.Namespace:
    cfg = dict(summary["config"])
    cfg.setdefault("warm_optimizer", True)
    cfg.setdefault("lag_predictor", True)
    cfg.setdefault("beta1", 0.9)
    return argparse.Namespace(**cfg)


def p1_default_runs(artifacts: bool) -> dict[str, Path]:
    if artifacts:
        return {t: ARTIFACTS / ARTIFACT_P1[t] for t in P1_TAGS}
    return {t: DEFAULT_OUT / f"p1_{t}" for t in P1_TAGS}


def p2_default_run(artifacts: bool) -> Path:
    return ARTIFACTS / "p2_history" if artifacts else DEFAULT_OUT / "p2_history"


def final_window(per_test_curves: dict[str, Sequence], window: int) -> dict[str, np.ndarray]:
    return {m: np.asarray(c, dtype=np.float64)[:, -window:].mean(axis=1) for m, c in per_test_curves.items()}


def finished(out: Path, force: bool) -> bool:
    return (out / "summary.json").exists() and not force


def dump_summary(out: Path, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary, indent=2)


__all__ = ["METHODS", "DEFAULT_OUT", "ARTIFACTS", "P1_TAGS", "P1_LABELS", "ARTIFACT_P1", "canonical_method",
           "summary_path", "load_summary", "config_namespace", "p1_default_runs", "p2_default_run", "final_window",
           "finished", "dump_summary"]
