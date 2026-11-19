"""Summarise selection-GRPO runs (<root>/<arm>-s<seed>/{evals.json,log.jsonl}): eval accuracy per arm at each eval step,
against steps, cumulative GPU-seconds and generated tokens (both summed from log.jsonl so resumes are counted once).
Writes summary.json, summary.md and accuracy.png under --out."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from influence_rlvr.files import write_json

COLORS = {"influence-adam": "#2a78d6", "none": "#eb6834", "influence-ekfac": "#1baf7a", "random": "#eda100",
          "mathonly": "#e87ba4", "disagreement": "#008300", "influence-dot": "#6da7ec", "oracle": "#e87ba4", "clean": "#777777"}
ORDER = ["none", "random", "disagreement", "mathonly", "influence-dot", "influence-adam", "influence-ekfac", "oracle", "clean"]


def load_runs(root: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r"(.+)-s(\d+)", d.name)
        if not m or not (d / "evals.json").is_file() or not (d / "log.jsonl").is_file():
            continue
        evals = {int(k): v for k, v in json.loads((d / "evals.json").read_text()).items()}
        log = sorted({r["step"]: r for r in map(json.loads, open(d / "log.jsonl"))}.values(), key=lambda r: r["step"])  # resumes before the trim fix duplicated rows; keep the last per step
        sec = np.cumsum([r["step_seconds"] for r in log])
        tok = np.cumsum([r["tokens"] for r in log])
        points = {}
        for step, e in sorted(evals.items()):
            if step > len(log):
                continue
            points[step] = {**{k: v for k, v in e.items() if k != "cum_seconds"},
                            "seconds": 0.0 if step == 0 else float(sec[step - 1]), "tokens": 0 if step == 0 else int(tok[step - 1])}
        bad = [r for r in log if r.get("n_bad") is not None]
        hit = None
        if bad and sum(r["n_bad"] for r in bad):
            drops = sum(len(r["kept"]) - sum(r["kept"]) for r in bad)
            hit = {"precision": sum(r["n_bad_dropped"] for r in bad) / max(drops, 1),
                   "recall": sum(r["n_bad_dropped"] for r in bad) / sum(r["n_bad"] for r in bad),
                   "base_rate": sum(r["n_bad"] for r in bad) / sum(len(r["kept"]) for r in bad)}
        runs[m.group(1)][int(m.group(2))] = {"points": points, "steps": len(log), "hit": hit}
    return runs


def summarise(runs: dict[str, dict[int, dict]]) -> dict:
    out = {}
    for arm, seeds in runs.items():
        steps = sorted(set.intersection(*(set(s["points"]) for s in seeds.values())))
        sets = sorted(k for k in next(iter(seeds.values()))["points"][steps[0]] if k not in ("seconds", "tokens"))
        hits = [r["hit"] for r in seeds.values() if r["hit"]]
        out[arm] = {"seeds": sorted(seeds), "steps_done": {s: r["steps"] for s, r in seeds.items()}, "at": {},
                    "hit": None if not hits else {k: float(np.mean([h[k] for h in hits])) for k in hits[0]}}
        for st in steps:
            rec = {}
            for k in sets + ["seconds", "tokens"]:
                vals = np.array([seeds[s]["points"][st][k] for s in sorted(seeds)], dtype=float)
                rec[k] = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max())}
            out[arm]["at"][st] = rec
    return out


def markdown(summary: dict) -> str:
    arms = [a for a in ORDER if a in summary] + sorted(set(summary) - set(ORDER))
    steps = sorted({st for a in arms for st in summary[a]["at"]})
    sets = [k for k in summary[arms[0]]["at"][steps[0]] if k not in ("seconds", "tokens")]
    lines = []
    for k in sets:
        lines += [f"\n### {k} pass rate (mean over seeds, min-max)\n", "| arm | " + " | ".join(f"step {s}" for s in steps) + " |",
                  "|---|" + "---|" * len(steps)]
        for a in arms:
            cells = []
            for s in steps:
                r = summary[a]["at"].get(s)
                cells.append("" if r is None else f"{r[k]['mean']:.3f} ({r[k]['min']:.3f}-{r[k]['max']:.3f})")
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["\n### cost at the last common step (mean over seeds)\n", "| arm | step | GPU-hours | M tokens |", "|---|---|---|---|"]
    for a in arms:
        last = max(summary[a]["at"])
        r = summary[a]["at"][last]
        lines.append(f"| {a} | {last} | {r['seconds']['mean'] / 3600:.2f} | {r['tokens']['mean'] / 1e6:.2f} |")
    if any(summary[a]["hit"] for a in arms):
        lines += ["\n### faulty-prompt drops (mean over seeds)\n", "| arm | precision of drops | recall of faulty | base rate |", "|---|---|---|---|"]
        for a in arms:
            h = summary[a]["hit"]
            if h:
                lines.append(f"| {a} | {h['precision']:.2f} | {h['recall']:.2f} | {h['base_rate']:.2f} |")
    return "\n".join(lines) + "\n"


def plot(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [a for a in ORDER if a in summary] + sorted(set(summary) - set(ORDER))
    sets = [k for k in next(iter(summary.values()))["at"][0] if k not in ("seconds", "tokens")]
    axes_x = [("step", None), ("GPU-hours", lambda r: r["seconds"]["mean"] / 3600), ("M generated tokens", lambda r: r["tokens"]["mean"] / 1e6)]
    fig, axs = plt.subplots(len(sets), 3, figsize=(13, 3.6 * len(sets)), squeeze=False)
    for i, k in enumerate(sets):
        for j, (xl, fx) in enumerate(axes_x):
            ax = axs[i][j]
            for a in arms:
                pts = sorted(summary[a]["at"].items())
                xs = [st if j == 0 else fx(r) for st, r in pts]
                ys = [r[k]["mean"] for _, r in pts]
                lo = [r[k]["min"] for _, r in pts]
                hi = [r[k]["max"] for _, r in pts]
                c = COLORS.get(a, "#777777")
                ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4, ls="--" if a == "influence-dot" else "-", label=a)
                ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
                ax.annotate(a, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color="#333", va="center")
            ax.set_xlabel(xl)
            ax.set_ylabel(f"{k} pass rate" if j == 0 else "")
            ax.grid(alpha=0.25, lw=0.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axs[0][0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Selection GRPO: held-out accuracy vs step / GPU time / sampled tokens (band = seed min-max)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="summarise selection-GRPO arms")
    p.add_argument("--root", type=Path, default=Path("outputs/llm/c1"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    out = args.out or args.root
    summary = summarise(load_runs(args.root))
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(markdown(summary))
    plot(summary, out / "accuracy.png")
    print((out / "summary.md").read_text())
    return summary


if __name__ == "__main__":
    main()
