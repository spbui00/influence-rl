from __future__ import annotations

import json

import numpy as np
import torch

from experiments.llm.horizon import continuation_order, report


def test_continuation_order_is_seeded_and_covers_every_step():
    a, b = continuation_order(0, 100, 30, 48), continuation_order(0, 100, 30, 48)
    assert (a == b).all() and len(a) >= 29 * 48 and set(a[:100]) == set(range(100))
    assert not (continuation_order(1, 100, 30, 48)[:100] == a[:100]).all()


def test_report_recovers_planted_signal_and_decay(tmp_path):
    rng = np.random.default_rng(0)
    n, M, T = 64, 40, 5
    scores = rng.normal(size=(T, n))
    masks = np.zeros((M, n), dtype=bool)
    for j in range(M):
        masks[j, rng.choice(n, n // 2, replace=False)] = True
    torch.save({"scores": {"rollout-gold/dot": scores}, "masks": masks}, tmp_path / "prepared.pt")
    truth = masks.astype(float) @ scores.T  # (M, T)
    for j in range(M):
        gold = {"1": truth[j].tolist(), "10": (truth[j] + rng.normal(scale=50, size=T)).tolist()}
        json.dump({"subset": j, "mask": masks[j].astype(int).tolist(), "grid": [1, 10], "f0": [0.0] * T, "gold": gold, "seconds": 0},
                  open(tmp_path / f"subset_{j}.json", "w"))
    r = report(tmp_path)
    lds = r["lds"]["rollout-gold/dot"]
    assert lds["1"]["mean"] > 0.99 and abs(lds["10"]["mean"]) < 0.5 and r["n_subsets"] == M
