"""Paired comparison of arms from per-problem final evals (<root>/<arm>-s<seed>/final_eval.json): per problem, average the
seeds of each arm; then a paired bootstrap CI of the mean difference across problems and a paired sign-flip p-value.
Writes paired.md under --out."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import ORDER


def load(root: Path) -> dict[str, dict[str, np.ndarray]]:
    per_arm: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(root.glob("*/final_eval.json")):
        m = re.fullmatch(r"(.+)-s(\d+)", f.parent.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        for k, v in d.items():
            if isinstance(v, list):
                per_arm[m.group(1)][k].append(np.asarray(v, dtype=float))
    return {a: {k: np.mean(np.stack(v), axis=0) for k, v in sets.items()} for a, sets in per_arm.items()}


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 20000, seed: int = 0) -> dict:
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    flips = np.array([(d * rng.choice([-1.0, 1.0], len(d))).mean() for _ in range(n_boot)])
    p = float((np.abs(flips) >= abs(d.mean())).mean())
    return {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "p": p, "n": int(len(d))}


def main(argv: list[str] | None = None) -> str:
    p = argparse.ArgumentParser(description="paired bootstrap across eval problems")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference", default="influence-adam", help="arm every other arm is compared against")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    arms = load(args.root)
    ref = arms[args.reference]
    order = [a for a in ORDER if a in arms and a != args.reference] + sorted(set(arms) - set(ORDER) - {args.reference})
    lines = [f"# {args.root.name}: {args.reference} minus each arm (per-problem paired bootstrap, seeds averaged per problem)\n"]
    for k in ref:
        lines += [f"\n### {k}\n", "| arm | diff | 95% CI | sign-flip p | n problems |", "|---|---|---|---|---|"]
        for a in order:
            if k not in arms[a]:
                continue
            r = paired(ref[k], arms[a][k])
            lines.append(f"| {a} | {r['diff']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.4f} | {r['n']} |")
    text = "\n".join(lines) + "\n"
    (args.out or args.root / "paired.md").write_text(text)
    print(text)
    return text


if __name__ == "__main__":
    main()
