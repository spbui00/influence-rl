"""Table 1 layout: final-window means (last --window records of Protocol 1, Protocol-2 endpoint) per estimator and
adjacent-gap paired-bootstrap p-values; compare with the printed values via experiments.compare_to_paper."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.toy.summaries import (
    DEFAULT_OUT, METHODS, P1_LABELS, P1_TAGS, final_window, load_summary, p1_default_runs, p2_default_run,
)
from influence_rlvr.attribution.metrics import paired_bootstrap_p
from influence_rlvr.files import write_json

NAMES = {"dot": "Gradient dot product", "adam": "Adam diagonal", "ekfac": "Damped Fisher (EK-FAC)",
         "cg": "Damped Fisher (exact, CG)"}
COLUMNS = list(P1_TAGS) + ["p2"]
COLUMN_LABELS = {**{t: f"P1 {P1_LABELS[t]}" for t in P1_TAGS}, "p2": "P2 (from scratch)"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="assemble Table 1 with adjacent-gap paired-bootstrap p-values")
    p.add_argument("--artifacts", action="store_true", help="read the shipped paper runs under artifacts/toy")
    for t in P1_TAGS:
        p.add_argument(f"--{t.replace('_', '-')}", dest=t, type=str, default=None)
    p.add_argument("--p2", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--B", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    return p


def per_target_columns(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    runs = p1_default_runs(args.artifacts)
    cols: dict[str, dict[str, np.ndarray]] = {}
    for t in P1_TAGS:
        s = load_summary(getattr(args, t) or runs[t])
        cols[t] = final_window(s["per_test_curves"], args.window)
    s2 = load_summary(args.p2 or p2_default_run(args.artifacts))
    cols["p2"] = {m: np.asarray(v, dtype=np.float64) for m, v in s2["per_test"].items()}
    return cols


def adjacent_gaps(col: dict[str, np.ndarray], B: int, seed: int) -> list[dict]:
    order = sorted(col, key=lambda m: -col[m].mean())
    return [{"better": a, "worse": b, "p": paired_bootstrap_p(col[a], col[b], B=B, seed=seed)}
            for a, b in zip(order[:-1], order[1:])]


def markdown(table: dict) -> str:
    cols = [c for c in COLUMNS if c in table["means"]]
    lines = ["| Estimator | " + " | ".join(COLUMN_LABELS[c] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for m in METHODS:
        if m in table["means"][cols[0]]:
            lines.append(f"| {NAMES[m]} | " + " | ".join(f"{table['means'][c][m]:.3f}" for c in cols) + " |")
    lines.append("")
    for c in cols:
        gaps = "; ".join(f"{g['better']} > {g['worse']}: p = {g['p']:.3g}" for g in table["p_values"][c])
        lines.append(f"{COLUMN_LABELS[c]}: {gaps}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cols = per_target_columns(args)
    means = {c: {m: float(v.mean()) for m, v in cols[c].items()} for c in cols}
    table = {
        "table": "table1",
        "window_records": args.window,
        "means": means,
        "p_values": {c: adjacent_gaps(cols[c], args.B, args.seed) for c in cols},
        "bootstrap": {"B": args.B, "seed": args.seed},
    }
    out = Path(args.out) if args.out else DEFAULT_OUT / ("table1_artifacts.json" if args.artifacts else "table1.json")
    write_json(out, table, indent=2)
    print(markdown(table))
    print(f"[table1] wrote {out}")


if __name__ == "__main__":
    main()
