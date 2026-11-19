from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from influence_rlvr.attribution import DampedFisherEKFAC, fit_ekfac
from influence_rlvr.utils import grad_vector, n_trainable, trainable_params


def _mlp(seed=0, in_dim=6, hidden=5, out_dim=3, bias=True):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(in_dim, hidden, bias=bias), nn.Tanh(), nn.Linear(hidden, out_dim, bias=bias))


def _softmax_samples(model, zs):
    samples = []
    for z in zs:
        with torch.no_grad():
            probs = torch.softmax(model(z), dim=-1).squeeze(0)
        for y in range(probs.numel()):
            samples.append((float(probs[y]) / len(zs),
                            lambda z=z, y=y: torch.log_softmax(model(z), dim=-1)[0, y]))
    return samples


def _dense_fisher(model, samples):
    G = torch.stack([grad_vector(model, c(), dtype=torch.float64) for _, c in samples])
    w = torch.tensor([wt for wt, _ in samples], dtype=torch.float64)
    return (G * w[:, None]).T @ G / w.sum()


def test_exact_kronecker_matches_dense_solve():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(5, 4, bias=True))
    z = torch.randn(1, 5)
    samples = _softmax_samples(model, [z])
    lam = 0.03
    est = DampedFisherEKFAC(fit_ekfac(model, samples), lambda_damp=lam)
    F = _dense_fisher(model, samples)
    torch.manual_seed(8)
    g = torch.randn(F.shape[0])
    h_dense = torch.linalg.solve(F + lam * torch.eye(F.shape[0], dtype=torch.float64), g.double())
    h = est.solve(g)
    assert h.dtype == torch.float32
    assert torch.allclose(h.double(), h_dense, rtol=1e-5, atol=1e-6)


def test_apply_fisher_inverts_solve():
    model = _mlp(seed=3)
    torch.manual_seed(4)
    samples = _softmax_samples(model, [torch.randn(1, 6) for _ in range(7)])
    lam = 0.05
    est = DampedFisherEKFAC(fit_ekfac(model, samples), lambda_damp=lam)
    torch.manual_seed(5)
    g = torch.randn(est.factors.n_params)
    h = est.solve(g)
    assert torch.allclose(est.apply_fisher(h) + lam * h, g, rtol=1e-4, atol=1e-5)


def test_mlp_with_bias_covered_and_slices_tile():
    model = _mlp(seed=0)
    torch.manual_seed(1)
    samples = _softmax_samples(model, [torch.randn(1, 6) for _ in range(7)])
    factors = fit_ekfac(model, samples)
    assert factors.n_params == n_trainable(model)
    assert bool(factors.covered.all())
    assert len(factors.blocks) == 2
    pieces = []
    for blk in factors.blocks:
        assert blk.has_bias and blk.in_dim == blk.q_a.shape[0] and blk.out_dim == blk.q_s.shape[0]
        assert blk.lam.shape == (blk.out_dim, blk.in_dim)
        pieces += [blk.weight_slice, blk.bias_slice]
    pieces.sort(key=lambda s: s.start)
    assert pieces[0].start == 0 and pieces[-1].stop == factors.n_params
    assert all(a.stop == b.start for a, b in zip(pieces, pieces[1:]))
    params = trainable_params(model)
    for blk in factors.blocks:
        assert blk.weight_slice.stop - blk.weight_slice.start == blk.out_dim * (blk.in_dim - 1)
    assert sum(p.numel() for p in params) == factors.n_params
    total_sq, w_total = 0.0, 0.0
    for w, closure in samples:
        g = grad_vector(model, closure(), dtype=torch.float64)
        total_sq += w * float(g.pow(2).sum())
        w_total += w
    lam_sum = sum(float(blk.lam.sum()) for blk in factors.blocks)
    assert math.isclose(lam_sum, total_sq / w_total, rel_tol=1e-6)


def test_rel_damp_sets_per_block_damping():
    model = _mlp(seed=11)
    torch.manual_seed(12)
    samples = _softmax_samples(model, [torch.randn(1, 6) for _ in range(5)])
    factors = fit_ekfac(model, samples)
    rel = 0.1
    est = DampedFisherEKFAC(factors, lambda_damp=0.1, rel_damp=rel)
    damps = [est.block_damp(blk) for blk in factors.blocks]
    assert damps == [max(rel * float(blk.lam.mean()), 1e-12) for blk in factors.blocks]
    assert damps[0] != damps[1]
    torch.manual_seed(13)
    g = torch.randn(factors.n_params)
    h = est.solve(g)
    lam_vec = torch.zeros(factors.n_params)
    for blk, d in zip(factors.blocks, damps):
        lam_vec[blk.weight_slice] = d
        lam_vec[blk.bias_slice] = d
    assert torch.allclose(est.apply_fisher(h) + lam_vec * h, g, rtol=1e-4, atol=1e-5)
    h_abs = DampedFisherEKFAC(factors, lambda_damp=0.1).solve(g)
    assert not torch.allclose(h, h_abs)
    est_none = DampedFisherEKFAC(factors, lambda_damp=0.1, rel_damp=None)
    assert all(est_none.block_damp(blk) == 0.1 for blk in factors.blocks)


def test_peft_lora_traces_only_adapter_linears():
    peft = pytest.importorskip("peft")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 5)
            self.fc2 = nn.Linear(5, 3)

        def forward(self, x):
            return self.fc2(torch.tanh(self.fc1(x)))

    torch.manual_seed(20)
    base = Tiny()
    cfg = peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["fc1", "fc2"])
    model = peft.get_peft_model(base, cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_()
    torch.manual_seed(21)
    samples = _softmax_samples(model, [torch.randn(1, 6) for _ in range(4)])
    factors = fit_ekfac(model, samples, dtype=torch.float32)
    names = sorted(blk.name for blk in factors.blocks)
    expected = sorted(n for n, m in model.named_modules()
                      if isinstance(m, nn.Linear) and (".lora_A." in n or ".lora_B." in n))
    assert names == expected and len(names) == 4
    assert all("base_layer" not in n for n in names)
    assert factors.n_params == n_trainable(model)
    assert bool(factors.covered.all())
    assert all(not blk.has_bias for blk in factors.blocks)
    assert all(blk.lam.dtype == torch.float32 for blk in factors.blocks)
    est = DampedFisherEKFAC(factors, lambda_damp=0.1, rel_damp=0.1)
    torch.manual_seed(22)
    g = torch.randn(factors.n_params)
    h = est.solve(g)
    assert h.shape == g.shape and torch.isfinite(h).all()


def test_nonfinite_sample_is_dropped(capsys):
    model = _mlp(seed=30)
    torch.manual_seed(31)
    zs = [torch.randn(1, 6) for _ in range(4)]
    clean = _softmax_samples(model, zs)
    bad = (0.5, lambda: torch.log_softmax(model(zs[0]), dim=-1)[0, 0] * float("nan"))
    ref = fit_ekfac(model, clean)
    got = fit_ekfac(model, clean[:2] + [bad] + clean[2:])
    assert "dropped 1/" in capsys.readouterr().out
    for a, b in zip(ref.blocks, got.blocks):
        assert torch.equal(a.lam, b.lam) and torch.equal(a.q_a, b.q_a) and torch.equal(a.q_s, b.q_s)
    with pytest.raises(RuntimeError, match="non-finite"):
        fit_ekfac(model, [bad])


def test_missing_backward_raises_checkpointing_diagnosis():
    model = _mlp(seed=40)
    z = torch.randn(1, 6)

    def closure():
        loss = model(z).sum()
        model(z)
        return loss

    with pytest.raises(RuntimeError, match="checkpointing"):
        fit_ekfac(model, [(1.0, closure)])


def test_rejects_empty_or_untraceable():
    model = _mlp(seed=50)
    with pytest.raises(ValueError):
        fit_ekfac(model, [])
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        fit_ekfac(model, [(1.0, lambda: model(torch.randn(1, 6)).sum())])
