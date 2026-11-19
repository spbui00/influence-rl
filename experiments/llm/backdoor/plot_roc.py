"""Backdoor ROC figure (gold mode, Adam): single checkpoints, the sum over all checkpoints and the Protocol 2
visited sum (batch-log offset 12, as in the table). A dot marks where the 200 highest-scored prompts end."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.llm.backdoor.aggregate import ARTIFACTS, load_batch_log, load_labels, load_scores  # noqa: E402
from influence_rlvr.attribution.trajectory import batch_visits, single, sum_all, sum_visited  # noqa: E402


def roc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    if labels[order[: labels.sum()]].mean() < labels.mean():  # poison in the lower tail: flip
        order = order[::-1]
    tpr = np.concatenate([[0], np.cumsum(labels[order]) / labels.sum()])
    fpr = np.concatenate([[0], np.cumsum(~labels[order]) / (~labels).sum()])
    return fpr, tpr


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", type=Path, default=ARTIFACTS / "gold_scores")
    p.add_argument("--pool", type=Path, default=ARTIFACTS / "ranking_gold_tracin-adam_P2.jsonl")
    p.add_argument("--batch-log", type=Path, default=ARTIFACTS / "batch_log.jsonl")
    p.add_argument("--offset", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("outputs/llm/backdoor/backdoor_roc"))
    args = p.parse_args(argv)

    scores = load_scores(args.scores_dir)
    labels = load_labels(args.pool, next(iter(scores.values())).size)
    visits = batch_visits(load_batch_log(args.batch_log))
    n_poison = int(labels.sum())
    curves = [("step 200 alone", single(scores, 200), "#2b6fde", 1.3),
              ("step 100 alone", single(scores, 100), "#8db8f5", 1.1),
              ("sum over all checkpoints", sum_all(scores, nan="median"), "#111111", 1.3),
              ("Protocol 2", sum_visited(scores, visits, offset=args.offset, nan="zero"),
               "#2e9e4f", 1.3)]

    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=300)
    for label, s, color, lw in curves:
        fpr, tpr = roc(s, labels)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{label} ({auc:.3f})")
        ax.plot(fpr[n_poison], tpr[n_poison], "o", color=color, ms=3.5, mec="k", mew=0.4)
        print(f"{label}: AUC {auc:.4f}, poisoned among top {n_poison}: {round(tpr[n_poison] * n_poison)}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="chance (0.5)")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
           xlabel=f"share of the {len(labels) - n_poison} clean prompts flagged",
           ylabel=f"share of the {n_poison} poisoned prompts found")
    ax.legend(title="ranking (AUC)", title_fontsize=6.4, fontsize=6.2, frameon=False, loc="lower right", handlelength=1.8)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), bbox_inches="tight")
    print(f"wrote {args.out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
