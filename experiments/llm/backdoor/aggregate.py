"""Table 8 mixes two NaN conventions exactly as the legacy analysis scripts did: single / all / window rows
use per-checkpoint median fill (plot_backdoor.py), visited / momentum rows NaN -> 0 inside the sum (nansum).
AUC is sign-agnostic (max(a, 1-a)) and prec@k reads the incriminated tail: in the shipped gold scores the
poison is the HIGHER tail (poison_lower=False). Rollout row: score files if present, else the shipped report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from influence_rlvr.attribution.metrics import oriented_auc, precision_at_k
from influence_rlvr.attribution.trajectory import batch_visits, momentum_tail, single, sum_all, sum_visited, window
from influence_rlvr.files import read_jsonl, scored_steps, write_json

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "llm" / "backdoor"
ARTIFACTS = REPO / "artifacts" / "llm" / "backdoor"
DEFAULT_RUN = REPO / "outputs" / "llm" / "backdoor"
TOP_KS = (20, 50)


def load_scores(score_dir: Path) -> dict[int, np.ndarray]:
    out = {s: v.astype(np.float64) for s, v in scored_steps(score_dir).items()}
    if not out:
        raise FileNotFoundError(f"no *_scores_step{{N}}.npy under {score_dir}")
    return out


def load_batch_log(path: Path) -> list[dict]:
    return read_jsonl(path) if Path(path).is_file() else []


def load_labels(path: Path, n: int | None = None) -> np.ndarray:
    rows = read_jsonl(path)
    n = n or len(rows)
    lab = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        idx = int(r.get("train_index", i))
        lab[idx] = (r["group"] == "poison") if "group" in r else bool(r["poison"])
    return lab


def evaluate(scores: np.ndarray, labels: np.ndarray, k: int, top_ks: tuple[int, ...] = TOP_KS) -> dict:
    auc, poison_lower = oriented_auc(scores, labels, nan="median")
    return {"auc": auc, "precision_at_k": precision_at_k(scores, labels, k, poison_lower, nan="median"),
            **{f"poison_in_top_{t}": precision_at_k(scores, labels, t, poison_lower, nan="median") for t in top_ks},
            "poison_lower": poison_lower}


def visited_span(visits: dict[int, np.ndarray], offset: int) -> tuple[int, int]:
    steps = sorted(visits)
    return steps[0] + offset, steps[-1] + offset


def rollout_from_files(rollout_dir: Path, labels: np.ndarray, k: int) -> dict | None:
    try:
        scores = load_scores(rollout_dir)
    except FileNotFoundError:
        return None
    steps = sorted(scores)
    row = {**evaluate(sum_all(scores, nan="zero"), labels, k), "checkpoints": steps, "source": str(rollout_dir)}
    masks = {s: rollout_dir / f"saturated_step{s}.npy" for s in steps}
    if all(p.is_file() for p in masks.values()):
        sat = np.stack([np.load(p).astype(bool) for p in masks.values()])
        row["saturated_poison_pairs"] = int(sat[:, labels].sum())
        row["poison_pairs"] = int(sat[:, labels].size)
        row["saturated_share"] = float(sat[:, labels].mean())
        row["saturated_share_all"] = float(sat.mean())
    return row


def rollout_from_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    return {"auc": float(d["auc_separation"]), "precision_at_k": float(d["precision_at_k"]),
            **{f"poison_in_top_{t}": None for t in TOP_KS}, "poison_lower": bool(d.get("poison_more_harmful", True)),
            "checkpoints": d.get("checkpoints"), "source": str(path), "saturated_share": None}


def build_table(scores: dict[int, np.ndarray], labels: np.ndarray, batch_log: list[dict] | None, *, k: int = 200,
                singles: tuple[int, ...] = (1, 101, 199), win: tuple[int, int] = (151, 200),
                offsets: tuple[int, ...] = (0, 12), beta1: float = 0.9, horizon: int = 10,
                rollout: dict | None = None) -> dict:
    steps = sorted(scores)
    rows: dict[str, dict] = {}
    for s in singles:
        if s in scores:
            rows[f"single_step{s}"] = {"label": f"gold single checkpoint, step {s}", "nan": "median",
                                       **evaluate(single(scores, s, nan="median"), labels, k)}
    if batch_log:
        visits = batch_visits(batch_log)
        for off in offsets:
            lo, hi = visited_span(visits, off)
            rows[f"visited_offset{off}"] = {"label": f"gold trajectory sum, visited checkpoints (steps {lo}-{hi})",
                                            "nan": "zero", "offset": off, "visits_per_prompt": float(
                                                sum(len(v) for v in visits.values()) / labels.size),
                                            **evaluate(sum_visited(scores, visits, offset=off, nan="zero"), labels, k)}
        for off in offsets:
            lo, hi = visited_span(visits, off)
            rows[f"momentum_offset{off}"] = {"label": f"gold trajectory sum with the momentum tail (steps {lo}-{hi}, "
                                                      f"beta1={beta1}, horizon={horizon})", "nan": "zero", "offset": off,
                                             **evaluate(momentum_tail(scores, visits, beta1, horizon, offset=off, nan="zero"),
                                                        labels, k)}
    rows["sum_all"] = {"label": f"gold sum over all {len(steps)} checkpoints", "nan": "median",
                       **evaluate(sum_all(scores, nan="median"), labels, k)}
    if any(win[0] <= s <= win[1] for s in steps):
        rows[f"window_{win[0]}_{win[1]}"] = {"label": f"gold sum over steps {win[0]}-{win[1]}", "nan": "median",
                                            **evaluate(window(scores, win[0], win[1], nan="median"), labels, k)}
    else:
        print(f"[aggregate] no scored checkpoint in the window {win[0]}-{win[1]} (have {steps[0]}..{steps[-1]}); "
              "window row skipped")
    if rollout is not None:
        n_ck = len(rollout["checkpoints"]) if rollout.get("checkpoints") else "?"
        rows["rollout_sum"] = {"label": f"rollout sum over {n_ck} checkpoints", "nan": "zero", **rollout}
    rows["chance"] = {"label": "chance", "auc": 0.5, "precision_at_k": float(labels.mean()),
                      **{f"poison_in_top_{t}": float(labels.mean()) for t in TOP_KS}}
    curve = [evaluate(single(scores, s, nan="median"), labels, k) for s in steps]
    return {"table": "table8", "k": k, "n_pool": int(labels.size), "n_poison": int(labels.sum()), "checkpoints": steps,
            "n_batches_logged": len(batch_log) if batch_log else 0,
            "nan_columns": [int(i) for i in np.flatnonzero(np.isnan(scores[steps[0]]))], "rows": rows,
            "auc_by_checkpoint": {"steps": steps, "auc": [c["auc"] for c in curve],
                                  "precision_at_k": [c["precision_at_k"] for c in curve]}}


def markdown(table: dict) -> str:
    lines = ["| Ranking | AUC | Prec@%d | " % table["k"] + " | ".join(f"Poison in top {t}" for t in TOP_KS) + " |",
             "|---|---|---|" + "---|" * len(TOP_KS)]
    for key, r in table["rows"].items():
        tops = ["-" if r.get(f"poison_in_top_{t}") is None else f"{r[f'poison_in_top_{t}']:.2f}" for t in TOP_KS]
        extra = ""
        if key == "rollout_sum":
            share = r.get("saturated_share")
            extra = f" (saturated poison pairs: {share:.0%})" if share is not None else " (saturated share: n/a)"
        lines.append(f"| {r['label']}{extra} | {r['auc']:.3f} | {r['precision_at_k']:.2f} | " + " | ".join(tops) + " |")
    return "\n".join(lines)


def save_figure(table: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps, auc = table["auc_by_checkpoint"]["steps"], table["auc_by_checkpoint"]["auc"]
    fig, ax = plt.subplots(figsize=(4.6, 3.1), dpi=200)
    ax.plot(steps, auc, color="#2b6fde", lw=1.2, label="single checkpoint (gold target)")
    refs = (("sum_all", "#111111", "sum over all checkpoints"), ("visited_offset0", "#2e9e5b", "sum over visited checkpoints"),
            ("rollout_sum", "#d8352b", "rollout target, sum over checkpoints"))
    for key, color, label in refs:
        if key in table["rows"]:
            ax.axhline(table["rows"][key]["auc"], color=color, ls="--", lw=1, label=label)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("checkpoint (GRPO step)")
    ax.set_ylabel("detection AUC")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Table 3/8 rows from per-checkpoint gold scores, labels and the batch log")
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--artifacts", action="store_true", help="re-derive Table 8 from the shipped paper outputs")
    p.add_argument("--scores-dir", type=Path, default=None, help="default <run>/influence/gold_adam")
    p.add_argument("--rollout-dir", type=Path, default=None, help="default <run>/influence/rollout_adam")
    p.add_argument("--rollout-report", type=Path, default=ARTIFACTS / "report_rollout_tracin-adam_P2.json")
    p.add_argument("--pool", type=Path, default=DATA / "pool.jsonl", help="labels (pool.jsonl or a ranking jsonl)")
    p.add_argument("--batch-log", type=Path, default=None, help="default <run>/batch_log.jsonl, else the shipped log")
    p.add_argument("--out", type=Path, default=None, help="default <run>/table8.json (table8_artifacts.json with --artifacts)")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--singles", default="1,101,199")
    p.add_argument("--window", default="151,200")
    p.add_argument("--offsets", default=None,
                   help="batch-log step offsets; default 0 for a run's own log, 0,12 for the shipped sibling log "
                        "(188 lines: steps 1-188 or 13-200)")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--figure", action=argparse.BooleanOptionalAction, default=True)
    return p


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    run = Path(args.run)
    args.scores_dir = args.scores_dir or (ARTIFACTS / "gold_scores" if args.artifacts else run / "influence" / "gold_adam")
    args.rollout_dir = args.rollout_dir or (None if args.artifacts else run / "influence" / "rollout_adam")
    if args.batch_log is None:
        own = run / "batch_log.jsonl"
        args.batch_log = own if (own.is_file() and not args.artifacts) else ARTIFACTS / "batch_log.jsonl"
    if args.offsets is None:
        args.offsets = "0,12" if Path(args.batch_log).resolve() == (ARTIFACTS / "batch_log.jsonl").resolve() else "0"
    args.out = args.out or run / ("table8_artifacts.json" if args.artifacts else "table8.json")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = resolve(build_parser().parse_args(argv))
    scores = load_scores(args.scores_dir)
    n = next(iter(scores.values())).size
    labels = load_labels(args.pool, n)
    batch_log = load_batch_log(args.batch_log)
    rollout = rollout_from_files(args.rollout_dir, labels, args.k) if args.rollout_dir else None
    if rollout is None:
        rollout = rollout_from_report(args.rollout_report)
    table = build_table(scores, labels, batch_log, k=args.k, singles=tuple(int(x) for x in args.singles.split(",")),
                        win=tuple(int(x) for x in args.window.split(",")),
                        offsets=tuple(int(x) for x in args.offsets.split(",")), beta1=args.beta1, horizon=args.horizon,
                        rollout=rollout)
    table["sources"] = {"scores_dir": str(args.scores_dir), "labels": str(args.pool), "batch_log": str(args.batch_log),
                        "rollout": rollout.get("source") if rollout else None}
    write_json(args.out, table)
    curve_path = args.out.with_name(args.out.stem.replace("table8", "auc_curve") + ".json")
    write_json(curve_path, table["auc_by_checkpoint"])
    if args.figure:
        save_figure(table, args.out.with_name(args.out.stem.replace("table8", "auc_by_checkpoint")))
    print(markdown(table))
    print(f"\nwrote {args.out}, {curve_path}" + (f", {args.out.stem.replace('table8', 'auc_by_checkpoint')}.png/.pdf"
                                                if args.figure else ""))
    return table


if __name__ == "__main__":
    main()
