from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from influence_rlvr.attribution import influence_matrix, make_estimator, unit_rows
from influence_rlvr.attribution.adam import AdamDiagonal
from influence_rlvr.attribution.base import Estimator
from influence_rlvr.attribution.cg import DampedFisherCG, cg_solve
from influence_rlvr.attribution.dot import Dot
from influence_rlvr.attribution.fvp import FisherRow, GradCacheFVP, PolicyFisherFVP, exact_fisher_from_grad_cache
from influence_rlvr.utils import flatten, trainable_params, unflatten


def _caches(D=12, n_y=4, N=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    grad_cache = [torch.randn(n_y, D, generator=g) for _ in range(N)]
    prob_cache = [torch.softmax(torch.randn(n_y, generator=g), dim=0) for _ in range(N)]
    return grad_cache, prob_cache


def _legacy_cg(fvp_fn, g_test, lambda_damp, cg_iters, cg_tol):
    g = g_test.detach().to(dtype=torch.float32)
    h = torch.zeros_like(g)
    r = g.clone()
    p = r.clone()
    rho = torch.dot(r, r)
    g_norm = g.norm().clamp(min=1e-12)
    for _ in range(cg_iters):
        Ap = fvp_fn(p) + lambda_damp * p
        pAp = torch.dot(p, Ap).clamp(min=1e-20)
        alpha = rho / pAp
        h = h + alpha * p
        r = r - alpha * Ap
        if (r.norm() / g_norm).item() < cg_tol:
            break
        rho_new = torch.dot(r, r)
        beta = rho_new / rho.clamp(min=1e-20)
        p = r + beta * p
        rho = rho_new
    return h


def test_exact_fisher_matches_naive_construction():
    grad_cache, prob_cache = _caches(D=6, n_y=3, N=5, seed=3)
    F = exact_fisher_from_grad_cache(grad_cache, prob_cache)
    ref = sum(G.T @ torch.diag(pi) @ G for G, pi in zip(grad_cache, prob_cache)) / len(grad_cache)
    torch.testing.assert_close(F, ref, rtol=1e-5, atol=1e-6)
    v = torch.randn(6, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(GradCacheFVP(grad_cache, prob_cache)(v), F @ v, rtol=1e-5, atol=1e-6)


def test_cg_matches_dense_solve():
    D, lam = 12, 0.1
    grad_cache, prob_cache = _caches(D=D)
    F = exact_fisher_from_grad_cache(grad_cache, prob_cache)
    g = torch.randn(D, generator=torch.Generator().manual_seed(123))
    h_ref = torch.linalg.solve(F + lam * torch.eye(D), g)
    h, info = cg_solve(GradCacheFVP(grad_cache, prob_cache), g, lam, 200, 1e-8)
    assert ((h - h_ref).norm() / h_ref.norm()).item() < 1e-4
    assert info["converged"] and info["status"] == "converged"
    assert info["g_dot_h"] >= 0.0
    assert info["final_residual"] < 1e-8 and len(info["residual_trace"]) == info["n_iters"]


def test_cg_is_bit_identical_to_legacy_loop():
    D, lam = 10, 0.3
    grad_cache, prob_cache = _caches(D=D, n_y=3, N=7, seed=5)
    fvp = GradCacheFVP(grad_cache, prob_cache)
    g = torch.randn(D, generator=torch.Generator().manual_seed(9))
    for iters, tol in [(100, 1e-6), (3, 1e-12)]:
        h, _ = cg_solve(fvp, g, lam, iters, tol)
        assert torch.equal(h, _legacy_cg(fvp, g, lam, iters, tol))


def test_cg_zero_g_is_trivial():
    grad_cache, prob_cache = _caches(D=6, n_y=3, N=4, seed=7)
    calls = []
    fvp = GradCacheFVP(grad_cache, prob_cache)
    h, info = cg_solve(lambda p: calls.append(1) or fvp(p), torch.zeros(6), 0.1, 50, 1e-6)
    assert torch.equal(h, torch.zeros(6))
    assert info["status"] == "trivial" and info["converged"] and info["n_iters"] == 0
    assert calls == []


def test_cg_max_iters_status():
    grad_cache, prob_cache = _caches(D=12)
    g = torch.randn(12, generator=torch.Generator().manual_seed(4))
    h, info = cg_solve(GradCacheFVP(grad_cache, prob_cache), g, 0.1, 2, 1e-12)
    assert info["status"] == "max_iters" and not info["converged"] and info["n_iters"] == 2
    assert h.dtype == torch.float32 and h.shape == (12,)


def test_damped_fisher_cg_estimator():
    D, T = 8, 3
    grad_cache, prob_cache = _caches(D=D, n_y=3, N=6, seed=11)
    est = DampedFisherCG(GradCacheFVP(grad_cache, prob_cache), lambda_damp=0.5, iters=100, tol=1e-9)
    assert est.name == "cg" and est.last_info is None
    G = torch.randn(T, D, generator=torch.Generator().manual_seed(2))
    H = est.solve_many(G)
    assert H.shape == (T, D) and est.last_info["status"] == "converged"
    F = exact_fisher_from_grad_cache(grad_cache, prob_cache)
    torch.testing.assert_close(H, torch.linalg.solve(F + 0.5 * torch.eye(D), G.T).T, rtol=1e-4, atol=1e-6)


def test_adam_diagonal():
    D, N = 9, 5
    gen = torch.Generator().manual_seed(0)
    P = torch.rand(D, generator=gen) + 0.5
    g = torch.randn(D, generator=gen)
    G = torch.randn(N, D, generator=gen)
    est = AdamDiagonal(P, lr=2.0)
    assert est.name == "adam"
    torch.testing.assert_close(est.solve(g), 2.0 * P * g)
    torch.testing.assert_close(est.solve_many(torch.stack([g, -g])), 2.0 * P * torch.stack([g, -g]))
    torch.testing.assert_close(est.influence(est.solve(g), G), (G @ (2.0 * P * g)).unsqueeze(0))
    with pytest.raises(ValueError):
        AdamDiagonal(torch.ones(D + 1)).solve(g)
    with pytest.raises(ValueError):
        AdamDiagonal(torch.ones(2, D))


def test_dot():
    gen = torch.Generator().manual_seed(1)
    g = torch.randn(7, generator=gen)
    G = torch.randn(4, 7, generator=gen)
    est = Dot()
    assert est.name == "dot"
    assert torch.equal(est.solve(g), g)
    torch.testing.assert_close(Estimator.influence(est.solve(g), G), (G @ g).unsqueeze(0))
    torch.testing.assert_close(est.solve_many(G), G)


@pytest.mark.parametrize("cosine", [False, True])
def test_influence_matrix_stacked_equals_streamed(cosine):
    D, T, N = 6, 3, 11
    gen = torch.Generator().manual_seed(3)
    P = torch.rand(D, generator=gen) + 0.1
    g_test = torch.randn(T, D, generator=gen)
    G = torch.randn(N, D, generator=gen)
    est = AdamDiagonal(P)
    stacked = influence_matrix(est, g_test, G, cosine=cosine)
    idx = np.arange(N)
    chunks = [(idx[i:i + 4], G[i:i + 4]) for i in range(0, N, 4)]
    streamed = influence_matrix(est, g_test, iter(chunks), n_train=N, cosine=cosine)
    assert stacked.shape == (T, N) and streamed.shape == (T, N)
    np.testing.assert_allclose(stacked, streamed, rtol=1e-6, atol=1e-7)
    h = P * g_test
    ref = (h / h.norm(dim=1, keepdim=True)) @ (G / G.norm(dim=1, keepdim=True)).T if cosine else h @ G.T
    np.testing.assert_allclose(stacked, ref.numpy(), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        influence_matrix(est, g_test, iter(chunks), cosine=cosine)


def test_influence_matrix_row_dtype_reproduces_legacy_bf16_rows():
    D, T, N = 64, 3, 9
    gen = torch.Generator().manual_seed(4)
    g_test = torch.randn(T, D, generator=gen)
    G = torch.randn(N, D, generator=gen)
    idx = np.arange(N)
    chunks = [(idx[i:i + 4], G[i:i + 4]) for i in range(0, N, 4)]
    for cosine in (False, True):
        h = g_test / g_test.norm(dim=1, keepdim=True).clamp(min=1e-12) if cosine else g_test
        hb = h.to(torch.bfloat16)
        ref = np.zeros((T, N))
        for i in range(N):
            gt = G[i] / G[i].norm().clamp(min=1e-12) if cosine else G[i]
            ref[:, i] = (hb @ gt.to(hb.dtype)).float().numpy()
        got = influence_matrix(Dot(), g_test, iter(chunks), n_train=N, cosine=cosine, row_dtype=torch.bfloat16)
        np.testing.assert_array_equal(got, ref)
        fp32 = influence_matrix(Dot(), g_test, G, cosine=cosine)
        assert np.abs(got - fp32).max() > 0 and np.allclose(got, fp32, rtol=5e-2, atol=5e-2)
        assert np.array_equal(fp32, influence_matrix(Dot(), g_test, G, cosine=cosine, row_dtype=torch.float32))


def test_unit_rows_matches_legacy_clamp():
    x = torch.tensor([[3e-7, 4e-7], [0.0, 0.0], [3.0, 4.0]])
    ref = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    assert torch.equal(unit_rows(x), ref)
    assert torch.equal(unit_rows(x)[1], torch.zeros(2))


def test_make_estimator_registry_errors():
    with pytest.raises(ValueError, match="unknown estimator"):
        make_estimator("bogus")
    with pytest.raises(ValueError, match="fisher"):
        make_estimator("fisher")
    with pytest.raises(ValueError, match="preconditioner"):
        make_estimator("adam")
    assert isinstance(make_estimator("identity"), Dot)
    assert isinstance(make_estimator("tracin-adam", preconditioner=torch.ones(3)), AdamDiagonal)


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 7, dim: int = 5, hidden: int = 6):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.emb.weight.requires_grad_(False)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask, use_cache=None):
        m = attention_mask.to(self.w1.weight.dtype).unsqueeze(-1)
        x = self.emb(input_ids) * m
        h = x.cumsum(1) / m.cumsum(1).clamp(min=1.0)
        return SimpleNamespace(logits=self.w2(torch.tanh(self.w1(h))))


def _dense_policy_fisher(model: nn.Module, rows: list[FisherRow]) -> torch.Tensor:
    params = trainable_params(model)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    flat0 = flatten([p.detach() for p in params], params)
    F = torch.zeros(flat0.numel(), flat0.numel(), dtype=torch.float64)
    for row in rows:
        ids = row.input_ids.unsqueeze(0)
        am = torch.ones_like(ids)

        def f(flat):
            out = torch.func.functional_call(model, dict(zip(names, unflatten(flat, params))),
                                             args=(), kwargs={"input_ids": ids, "attention_mask": am})
            return out.logits[0, row.response_start - 1:ids.shape[1] - 1]

        J = torch.autograd.functional.jacobian(f, flat0).double()
        p = torch.softmax(f(flat0).detach().double(), dim=-1)
        for t in range(J.shape[0]):
            M = torch.diag(p[t]) - torch.outer(p[t], p[t])
            F += J[t].T @ M @ J[t]
    return F / len(rows)


def _tiny_rows(seed=0):
    gen = torch.Generator().manual_seed(seed)
    lengths = [(6, 3), (9, 4), (7, 2), (5, 1), (8, 5)]
    return [FisherRow(torch.randint(0, 7, (L,), generator=gen), s) for L, s in lengths]


def test_policy_fisher_fvp_matches_dense():
    torch.manual_seed(0)
    model = TinyLM()
    rows = _tiny_rows()
    F = _dense_policy_fisher(model, rows)
    fvp = PolicyFisherFVP(model, rows, micro_batch=2)
    assert fvp.n_params == F.shape[0] == sum(p.numel() for p in trainable_params(model))
    gen = torch.Generator().manual_seed(5)
    for _ in range(3):
        v = torch.randn(F.shape[0], generator=gen)
        out = fvp(v)
        assert out.dtype == torch.float32 and out.shape == v.shape
        torch.testing.assert_close(out.double(), F @ v.double(), rtol=1e-3, atol=1e-6)
    fvp1 = PolicyFisherFVP(model, [tuple((r.input_ids, r.response_start)) for r in rows], micro_batch=1)
    v = torch.randn(F.shape[0], generator=gen)
    torch.testing.assert_close(fvp1(v), fvp(v), rtol=1e-4, atol=1e-6)


def test_policy_fisher_fvp_spectral_normalisation():
    torch.manual_seed(1)
    model = TinyLM()
    rows = _tiny_rows(seed=2)
    F = _dense_policy_fisher(model, rows)
    lam_max = torch.linalg.eigvalsh(F).max().item()
    fvp = PolicyFisherFVP(model, rows, micro_batch=4, normalize=True, power_iters=200, seed=0)
    assert abs(fvp.lam_max - lam_max) / lam_max < 1e-2
    v = torch.randn(F.shape[0], generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(fvp(v).double(), (F @ v.double()) / fvp.lam_max, rtol=1e-3, atol=1e-6)
    again = PolicyFisherFVP(model, rows, micro_batch=4, normalize=True, power_iters=200, seed=0)
    assert again.lam_max == fvp.lam_max


def test_policy_fisher_fvp_with_cg_matches_dense_solve():
    torch.manual_seed(2)
    model = TinyLM()
    rows = _tiny_rows(seed=4)
    F = _dense_policy_fisher(model, rows).float()
    est = DampedFisherCG(PolicyFisherFVP(model, rows), lambda_damp=0.1, iters=200, tol=1e-8)
    g = torch.randn(F.shape[0], generator=torch.Generator().manual_seed(6))
    h = est.solve(g)
    assert est.last_info["converged"]
    torch.testing.assert_close(h, torch.linalg.solve(F + 0.1 * torch.eye(F.shape[0]), g), rtol=1e-3, atol=1e-5)
