from __future__ import annotations

import numpy as np
import torch
from torch import nn

from influence_rlvr.llm.select import StepBatch, apply_update, select
from influence_rlvr.utils import trainable_params


def _batch(n: int = 6) -> StepBatch:
    rewards = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0]]
    answers = [["7"] * 4, [None] * 4, ["7", "3", "7", "3"], ["7", "9", "1", "2"], ["4", "5", "6", "8"], ["7", "8", "9", "1"]]
    return StepBatch([{"group": "poison" if i % 2 else "ok"} for i in range(n)], torch.zeros(n, 3), rewards, answers, 0)


def test_batch_statistics():
    b = _batch()
    assert b.live.tolist() == [False, False, True, True, True, True]
    assert b.disagreement.tolist() == [1, 0, 1, 2, 1, 3]


def test_selectors_drop_the_right_prompts():
    b, rng = _batch(), np.random.default_rng(0)
    scores = np.array([0.0, 0.0, 5.0, -2.0, 1.0, -4.0])
    assert select("none", b, scores, 2, rng).all()
    assert select("random", b, scores, 2, rng).sum() == 4
    assert np.flatnonzero(~select("influence", b, scores, 2, rng)).tolist() == [3, 5]
    assert np.flatnonzero(~select("anti", b, scores, 2, rng)).tolist() == [2, 4]
    assert np.flatnonzero(~select("disagreement", b, scores, 2, rng)).tolist() == [3, 5]
    assert np.flatnonzero(~select("liveness", b, scores, 2, rng)).tolist() == [0, 1]
    bad = np.array([r["group"] == "poison" for r in b.rows])
    assert np.flatnonzero(~select("oracle", b, scores, 2, rng, bad)).tolist() == [1, 3]


def test_apply_update_matches_adamw_with_clipping():
    torch.manual_seed(0)
    a, b = nn.Linear(4, 3), nn.Linear(4, 3)
    b.load_state_dict(a.state_dict())
    oa = torch.optim.AdamW(a.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1)
    x = torch.randn(8, 4)
    for _ in range(3):
        loss = 10 * a(x).pow(2).mean()
        oa.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(a.parameters(), 1.0); oa.step()
        grads = torch.autograd.grad(10 * b(x).pow(2).mean(), trainable_params(b))
        apply_update(b, ob, torch.cat([g.reshape(-1) for g in grads]), 1.0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-7)
