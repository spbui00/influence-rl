"""Protocol 2 LDS: historical score of a prompt = sum over the master's steps of its per-step gold-target score (dot / adam,
<master>/steps/step-*.pt); prediction for a subset = sum of its prompts' historical scores; truth = gold log-prob of the
targets after retraining on the subset (retrain.py). LDS = Spearman over subsets, per target then averaged, plus the
mean-outcome version. Writes <master>/p2/lds.json and prints a table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from influence_rlvr.attribution.metrics import per_target_lds
from influence_rlvr.files import write_json


def historical_scores(master: Path, n_pool: int) -> dict[str, np.ndarray]:
    H: dict[str, np.ndarray] = {}
    for f in sorted((master / "steps").glob("step-*.pt")):
        rec = torch.load(f, map_location="cpu", weights_only=False)
        if rec["scores_gold"] is None:
            raise SystemExit(f"{f}: no gold-target scores (master needs --score-targets)")
        for m, s in rec["scores_gold"].items():
            H.setdefault(m, np.zeros((n_pool, s.shape[1])))
            for b, i in enumerate(rec["train_index"]):
                if rec["kept"][b]:
                    H[m][i] += s[b]
    return H


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="P2 LDS from historical scores and subset retrains")
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="default <master>/p2")
    p.add_argument("--min-subsets", type=int, default=10)
    args = p.parse_args(argv)
    root = args.out or args.master / "p2"
    pool_path = Path(json.loads((args.master / "manifest.json").read_text())["args"]["pool"])
    n_pool = sum(1 for line in open(pool_path) if line.strip())
    H = historical_scores(args.master, n_pool)
    result = {"n_pool": n_pool, "modes": {}}
    lines = ["| truth | n subsets | score | per-target LDS | mean-outcome LDS |", "|---|---|---|---|---|"]
    for mode in ("fixed", "fresh"):
        runs = [json.loads(f.read_text()) for f in sorted((root / mode).glob("subset_*.json"))]
        runs = [r for r in runs if r["subset"] >= 0]
        if len(runs) < args.min_subsets:
            continue
        masks = np.array([r["mask"] for r in runs], dtype=float)
        y = np.array([r["gold_logp"] for r in runs])
        rec = {"n_subsets": len(runs), "lds": {}}
        for m, Hm in H.items():
            pred = masks @ Hm
            per = per_target_lds(pred, y, "average")
            rec["lds"][m] = {"per_target_mean": float(np.nanmean(per)), "per_target": per.tolist(),
                             "mean_outcome": float(spearmanr(pred.mean(1), y.mean(1)).correlation)}
            lines.append(f"| {mode} | {len(runs)} | {m} | {rec['lds'][m]['per_target_mean']:+.3f} | {rec['lds'][m]['mean_outcome']:+.3f} |")
        if "pass_rate" in runs[0]:
            pr = np.array([np.mean(r["pass_rate"]) for r in runs])
            rec["pass_rate_lds"] = {m: float(spearmanr((masks @ H[m]).mean(1), pr).correlation) for m in H}
        result["modes"][mode] = rec
    text = "\n".join(lines) + "\n"
    print(text)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "lds.json", result)
    (root / "lds.md").write_text(text)
    return result


if __name__ == "__main__":
    main()
