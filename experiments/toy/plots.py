from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.toy.summaries import (  # noqa: E402
    DEFAULT_OUT, P1_LABELS, P1_TAGS, final_window, load_summary, p1_default_runs, p2_default_run,
)
from influence_rlvr.attribution.metrics import bootstrap_band, bootstrap_ci_mean, paired_bootstrap_p  # noqa: E402

STYLE = {
    "dot": ("Dot product", "#8a8f99", "^"),
    "adam": ("Adam", "#2b6fde", "o"),
    "ekfac": ("Fisher (EK-FAC)", "#2e9e4f", "s"),
    "cg": ("Fisher (exact)", "#d8352b", "D"),
}
CURVE_ORDER = ["adam", "ekfac", "cg", "dot"]
RC = {"font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 7,
      "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False}


def curves_of(summary: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ptc = summary["per_test_curves"]
    return np.asarray(summary["ks"], dtype=float), {m: np.asarray(ptc[m], dtype=float) for m in CURVE_ORDER if m in ptc}


def draw_curves(ax: plt.Axes, ks: np.ndarray, curves: dict[str, np.ndarray], B: int, n_targets: int) -> None:
    every = max(1, len(ks) // 10)
    for m, c in curves.items():
        label, color, marker = STYLE[m]
        mean, lo, hi = bootstrap_band(c, B=B, seed=0)
        ax.plot(ks, mean, color=color, lw=1.4, label=label, marker=marker, ms=2.5, markevery=every, mec="k", mew=0.2)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.18, lw=0)
    ax.set_xlabel("continuation steps $k$")
    ax.set_ylabel(f"LDS (mean over {n_targets} targets)")
    ax.grid(alpha=0.3, ls=":")


def fig2(args: argparse.Namespace) -> None:
    runs = p1_default_runs(args.artifacts)
    sums = {t: load_summary(getattr(args, t) or runs[t]) for t in P1_TAGS}
    hist = load_summary(args.p2 or p2_default_run(args.artifacts))
    ks, curves = curves_of(sums["eta_over_10"])
    n_steps = int(sums["eta_over_10"]["config"]["subset_steps"])
    n_targets = next(iter(curves.values())).shape[0]

    plt.rcParams.update(RC)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.35), dpi=300,
                                 gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.36})
    draw_curves(ax, ks, curves, args.B, n_targets)
    if args.report_window:
        ax.axvspan(n_steps - args.report_window, n_steps, color="k", alpha=0.07, lw=0)
    ax.set_xlim(0, n_steps)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)
    ax.set_title("(a) Protocol 1 at rate $\\eta/10$", fontsize=7.5, loc="left")

    cols = {t: final_window(sums[t]["per_test_curves"], args.window) for t in P1_TAGS}
    cols["history"] = {m: np.asarray(v, dtype=float) for m, v in hist["per_test"].items()}
    order = list(P1_TAGS) + ["history"]
    xs = np.arange(len(order))
    rng = np.random.default_rng(0)
    for m in CURVE_ORDER:
        label, color, marker = STYLE[m]
        stats = [bootstrap_ci_mean(cols[c][m], B=args.B, rng=rng) for c in order]
        means = np.array([s[0] for s in stats])
        err = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
        bx.plot(xs[:-1], means[:-1], color=color, lw=1.4, marker=marker, ms=4, mec="k", mew=0.3, label=label)
        bx.plot(xs[-2:], means[-2:], color=color, lw=1.4, ls=":", marker=marker, ms=4, mec="k", mew=0.3)
        bx.errorbar(xs, means, yerr=err, fmt="none", ecolor=color, elinewidth=1, capsize=2)
    bx.axvline(len(order) - 1.5, color="0.6", lw=0.8, ls="--")
    ticks = {"eta_over_100": "$\\eta/100$", "eta_over_10": "$\\eta/10$", "eta": "$\\eta$", "history": "Protocol 2\n(history)"}
    bx.set_xticks(xs)
    bx.set_xticklabels([ticks[c] for c in order])
    bx.set_xlim(-0.35, len(order) - 0.65)
    bx.text(1.55, bx.get_ylim()[1] - 0.005, "Protocol 1: continuation rate", ha="center", va="top", fontsize=6.5, color="0.35")
    bx.set_ylabel(f"LDS (mean over {n_targets} targets)")
    bx.grid(alpha=0.3, ls=":", axis="y")
    bx.set_title("(b) ordering vs. how far the counterfactual travels", fontsize=7.5, loc="left")
    save(fig, args.out)


def fig4(args: argparse.Namespace) -> None:
    runs = p1_default_runs(args.artifacts)
    specs = args.runs or [f"{P1_LABELS[t]}={getattr(args, t) or runs[t]}" for t in ("eta", "eta_over_10", "eta_over_100")]
    loaded = [(spec.split("=", 1)[0], load_summary(spec.split("=", 1)[1])) for spec in specs]
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(loaded), figsize=(2.6 * len(loaded) + 0.6, 2.6), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    print(f"{'rate':<6} {'estimator':<32} {'final':>6} {'peak':>6} {'@k':>5}")
    for ax, (label, s) in zip(axes, loaded):
        ks, curves = curves_of(s)
        n_targets = next(iter(curves.values())).shape[0]
        draw_curves(ax, ks, curves, args.B, n_targets)
        finals = {m: c[:, -args.window:].mean(1) for m, c in curves.items()}
        for m, c in curves.items():
            mean = c.mean(0)
            print(f"{label:<6} {STYLE[m][0]:<32} {finals[m].mean():6.3f} {mean.max():6.3f} {int(ks[mean.argmax()]):5d}")
        order = sorted(finals, key=lambda m: -finals[m].mean())
        gaps = "  ".join(f"{a}>{b}: p={paired_bootstrap_p(finals[a], finals[b], B=20000, seed=0):.3g}"
                         for a, b in zip(order[:-1], order[1:]))
        print(f"{'':<6} ordering: {' > '.join(order)}   {gaps}")
        ax.set_title(f"continuation rate {label}", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax is not axes[0]:
            ax.set_ylabel("")
    axes[-1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, args.out)


def save(fig: plt.Figure, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {out}.{{png,pdf}}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figure 2 (fig2) and Figure 4 (fig4)")
    sub = p.add_subparsers(dest="figure", required=True)
    for name in ("fig2", "fig4"):
        q = sub.add_parser(name)
        q.add_argument("--artifacts", action="store_true")
        for t in P1_TAGS:
            q.add_argument(f"--{t.replace('_', '-')}", dest=t, type=str, default=None)
        q.add_argument("--window", type=int, default=10)
        q.add_argument("--out", type=str, default=str(DEFAULT_OUT / "figures" / name))
    f2, f4 = sub.choices["fig2"], sub.choices["fig4"]
    f2.add_argument("--p2", type=str, default=None)
    f2.add_argument("--B", type=int, default=5000)
    f2.add_argument("--report-window", type=int, default=100)
    f4.add_argument("runs", nargs="*", help="label=<run dir or summary.json> (panel order); default: η, η/10, η/100")
    f4.add_argument("--B", type=int, default=3000)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    (fig2 if args.figure == "fig2" else fig4)(args)


if __name__ == "__main__":
    main()
