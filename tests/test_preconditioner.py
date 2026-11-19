from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from influence_rlvr.preconditioner import (
    adam_preconditioner_from_optimizer,
    adam_preconditioner_from_state,
    adam_vhat_from_optimizer,
    adam_vhat_from_state,
    load_adam_preconditioner,
    load_adam_vhat,
)
from influence_rlvr.utils import n_trainable, trainable_named_params, trainable_params


class _TwoWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(4, 3))
        self.c = nn.Parameter(torch.zeros(5, 2))


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 7), nn.Tanh(), nn.Linear(7, 3))

    def forward(self, x):
        return self.net(x)


class _NormThenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(3)
        self.lin = nn.Linear(3, 2)


def _state_dict(vs, step, beta2=0.999, eps=1e-8):
    return {
        "state": {i: {"exp_avg_sq": v, "step": torch.tensor(float(step))} for i, v in enumerate(vs)},
        "param_groups": [{"betas": (0.9, beta2), "eps": eps, "params": list(range(len(vs)))}],
    }


def _expected(vs, step, beta2=0.999, eps=1e-8):
    bc2 = 1.0 - beta2 ** step
    return torch.cat([(1.0 / ((v / bc2).sqrt() + eps)).reshape(-1) for v in vs])


def _train(model, opt, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        opt.zero_grad()
        x = torch.randn(8, 6, generator=g)
        loss = (model(x) ** 2).mean()
        loss.backward()
        opt.step()


def _reference_from_optimizer(model, opt, eps=None):
    beta2 = opt.param_groups[0]["betas"][1]
    eps = opt.param_groups[0]["eps"] if eps is None else eps
    parts = []
    for p in trainable_params(model):
        st = opt.state[p]
        t = float(st["step"].item() if torch.is_tensor(st["step"]) else st["step"])
        parts.append((1.0 / ((st["exp_avg_sq"] / (1.0 - beta2 ** t)).sqrt() + eps)).reshape(-1))
    return torch.cat(parts)


def test_handbuilt_state_formula_bias_correction_and_order():
    torch.manual_seed(0)
    model = _TwoWeights()
    va, vc = torch.rand(4, 3) + 0.1, torch.rand(5, 2) + 0.1
    P = adam_preconditioner_from_state(trainable_named_params(model), _state_dict([va, vc], 10.0))
    assert P.numel() == va.numel() + vc.numel()
    torch.testing.assert_close(P, _expected([va, vc], 10.0))
    torch.testing.assert_close(P[: va.numel()], _expected([va], 10.0))


def test_vhat_from_state_matches_formula():
    torch.manual_seed(1)
    model = _TwoWeights()
    va, vc = torch.rand(4, 3) + 0.1, torch.rand(5, 2) + 0.1
    vhat = adam_vhat_from_state(trainable_named_params(model), _state_dict([va, vc], 7.0, beta2=0.99))
    bc2 = 1.0 - 0.99 ** 7.0
    torch.testing.assert_close(vhat, torch.cat([va.reshape(-1), vc.reshape(-1)]) / bc2)


def test_live_optimizer_matches_state_dict_and_reference():
    torch.manual_seed(2)
    model = _MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
    _train(model, opt, 5)
    P_live = adam_preconditioner_from_optimizer(opt, model)
    P_state = adam_preconditioner_from_state(trainable_named_params(model), opt.state_dict())
    torch.testing.assert_close(P_live, P_state)
    torch.testing.assert_close(P_live, _reference_from_optimizer(model, opt))
    assert torch.isfinite(P_live).all() and (P_live > 0).all()
    vhat_live = adam_vhat_from_optimizer(opt, model)
    vhat_state = adam_vhat_from_state(trainable_named_params(model), opt.state_dict())
    torch.testing.assert_close(vhat_live, vhat_state)
    torch.testing.assert_close(1.0 / (vhat_live.sqrt() + 1e-8), P_live)


def test_adamw_weight_decay_ignored():
    torch.manual_seed(3)
    m1, m2 = _MLP(), _MLP()
    m2.load_state_dict(m1.state_dict())
    o1 = torch.optim.AdamW(m1.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    o2 = torch.optim.AdamW(m2.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.5)
    _train(m1, o1, 4)
    _train(m2, o2, 4)
    torch.testing.assert_close(adam_preconditioner_from_optimizer(o1, m1), adam_preconditioner_from_optimizer(o2, m2))
    P = adam_preconditioner_from_state(trainable_named_params(m2), o2.state_dict())
    torch.testing.assert_close(P, _reference_from_optimizer(m2, o2))


def test_params_without_state_get_ones_and_zero_vhat():
    torch.manual_seed(4)
    model = _MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    P = adam_preconditioner_from_optimizer(opt, model)
    assert P.shape == (n_trainable(model),)
    assert torch.equal(P, torch.ones(n_trainable(model)))
    assert torch.equal(adam_vhat_from_optimizer(opt, model), torch.zeros(n_trainable(model)))
    frozen = trainable_params(model)[-1]
    frozen.requires_grad_(False)
    _train(model, opt, 2)
    P = adam_preconditioner_from_optimizer(opt, model)
    assert P.shape == (n_trainable(model),)
    torch.testing.assert_close(P, _reference_from_optimizer(model, opt))


def test_eps_override_from_optimizer_and_state():
    torch.manual_seed(5)
    model = _MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
    _train(model, opt, 3)
    torch.testing.assert_close(adam_preconditioner_from_optimizer(opt, model, eps=1e-3),
                               _reference_from_optimizer(model, opt, eps=1e-3))
    named = trainable_named_params(model)
    sd = opt.state_dict()
    torch.testing.assert_close(adam_preconditioner_from_state(named, sd, eps_override=1e-3),
                               _reference_from_optimizer(model, opt, eps=1e-3))
    torch.testing.assert_close(adam_preconditioner_from_optimizer(opt, model, eps=0.0),
                               adam_preconditioner_from_optimizer(opt, model))
    torch.testing.assert_close(adam_preconditioner_from_state(named, sd, eps_override=0.0),
                               adam_preconditioner_from_optimizer(opt, model))
    torch.testing.assert_close(adam_preconditioner_from_state(named, sd, eps_override=0.0),
                               _reference_from_optimizer(model, opt))


def test_eps_override_caps_dynamic_range():
    model = _TwoWeights()
    vs = [torch.full((4, 3), 1e-16), torch.full((5, 2), 1.0)]
    named = trainable_named_params(model)
    P_faithful = adam_preconditioner_from_state(named, _state_dict(vs, 50.0))
    P_capped = adam_preconditioner_from_state(named, _state_dict(vs, 50.0), eps_override=1e-3)
    assert P_faithful.max().item() > 1e7
    assert P_capped.max().item() <= 1e3 + 1.0


def test_regroup_fallback_on_index_shift():
    torch.manual_seed(6)
    model = _NormThenLinear()
    named = trainable_named_params(model)
    vs = {n: torch.rand_like(p) + 0.1 for n, p in named}
    decay = [n for n, _ in named if not (n.endswith(".bias") or "norm" in n.lower())]
    no_decay = [n for n, _ in named if n not in decay]
    assert decay == ["lin.weight"] and no_decay == ["norm.weight", "norm.bias", "lin.bias"]
    hf_order = decay + no_decay
    sd = _state_dict([vs[n] for n in hf_order], 3.0)
    assert [tuple(s["exp_avg_sq"].shape) for s in sd["state"].values()] != [tuple(p.shape) for _, p in named]
    P = adam_preconditioner_from_state(named, sd)
    torch.testing.assert_close(P, _expected([vs[n] for n, _ in named], 3.0))
    torch.testing.assert_close(adam_vhat_from_state(named, sd),
                               torch.cat([vs[n].reshape(-1) for n, _ in named]) / (1.0 - 0.999 ** 3.0))


def test_ordering_with_distinct_shapes():
    model = _TwoWeights()
    named = trainable_named_params(model)
    va, vc = torch.full((4, 3), 4.0), torch.full((5, 2), 16.0)
    sd = {
        "state": {0: {"exp_avg_sq": va, "step": torch.tensor(1.0)},
                  1: {"exp_avg_sq": vc, "step": torch.tensor(1.0)}},
        "param_groups": [{"betas": (0.9, 0.5), "eps": 0.0, "params": [0, 1]}],
    }
    P = adam_preconditioner_from_state(named, sd)
    torch.testing.assert_close(P[:12], torch.full((12,), 1.0 / (2.0 / (0.5 ** 0.5))))
    torch.testing.assert_close(P[12:], torch.full((10,), 1.0 / (4.0 / (0.5 ** 0.5))))
    sd_swapped = {"state": {0: sd["state"][1], 1: sd["state"][0]}, "param_groups": sd["param_groups"]}
    with pytest.raises(ValueError):
        adam_preconditioner_from_state(named, sd_swapped)


def test_count_mismatch_raises():
    model = _TwoWeights()
    named = trainable_named_params(model)
    with pytest.raises(ValueError):
        adam_preconditioner_from_state(named, _state_dict([torch.ones(4, 3)], 1.0))
    with pytest.raises(ValueError):
        adam_vhat_from_state(named, _state_dict([torch.ones(4, 3)], 1.0))


def test_non_adam_state_raises():
    model = _TwoWeights()
    named = trainable_named_params(model)
    sd = {"state": {0: {"momentum_buffer": torch.ones(4, 3)}, 1: {"momentum_buffer": torch.ones(5, 2)}},
          "param_groups": [{"lr": 0.1, "params": [0, 1]}]}
    with pytest.raises(ValueError):
        adam_preconditioner_from_state(named, sd)


def test_missing_optimizer_file(tmp_path: Path):
    model = _TwoWeights()
    assert load_adam_preconditioner(model, tmp_path, allow_missing=True) is None
    with pytest.raises(FileNotFoundError):
        load_adam_preconditioner(model, tmp_path)
    with pytest.raises(FileNotFoundError):
        load_adam_vhat(model, tmp_path)


def test_saved_optimizer_roundtrip(tmp_path: Path):
    torch.manual_seed(7)
    model = _MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1)
    _train(model, opt, 3)
    torch.save(opt.state_dict(), tmp_path / "optimizer.pt")
    P = load_adam_preconditioner(model, tmp_path)
    assert P is not None and P.dtype == torch.float32 and P.shape == (n_trainable(model),)
    torch.testing.assert_close(P, adam_preconditioner_from_optimizer(opt, model))
    torch.testing.assert_close(load_adam_preconditioner(model, tmp_path, eps=1e-2),
                               adam_preconditioner_from_optimizer(opt, model, eps=1e-2))
    P64 = load_adam_preconditioner(model, tmp_path, dtype=torch.float64)
    assert P64.dtype == torch.float64
    torch.testing.assert_close(P64.float(), P)
    vhat = load_adam_vhat(model, tmp_path)
    torch.testing.assert_close(vhat, adam_vhat_from_optimizer(opt, model))
    torch.testing.assert_close(1.0 / (vhat.sqrt() + 1e-8), P)
