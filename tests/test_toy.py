from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from influence_rlvr.toy import (
    ALL_SEQUENCES, FixtureSpec, adam_preconditioner, clone, exact_expected_reward, fixture, grpo_loss,
    score_checkpoint, train_grpo,
)
from influence_rlvr.utils import grad_vector, n_trainable

OLD = Path("/Users/bui/code/influence-rlvr")
OLD_PY = OLD / ".venv/bin/python"
TINY = dict(n_clusters=10, n_test=3)
TINY_STEPS = 50

LEGACY_DUMP = textwrap.dedent("""
    import sys, argparse, numpy as np, torch
    OLD = {old!r}
    sys.path.insert(0, OLD + "/scripts"); sys.path.insert(0, OLD + "/scripts/lds"); sys.path.insert(0, OLD)
    from common import build_fixture, train_grpo, adam_precond_vec, score_at_checkpoint
    args = argparse.Namespace(n_clusters={n_clusters}, per_cluster=3, input_dim=65, cluster_signal=3.0,
        cluster_target_mode="parity", seed=42, hard_fraction=0.5, signal_hard=0.8, signature_mode="overlap",
        signature_dim=32, test_signal_span="span", conflict_fraction=0.34, n_test={n_test}, hidden_dim=16)
    train, test, ref = build_fixture(args)
    model, opt = train_grpo(train, {steps}, ref_model=ref, lr=1e-3, use_adam=True, beta=0.1, seed=42)
    P = adam_precond_vec(opt, model)
    flat = lambda m: torch.cat([p.detach().flatten() for p in m.parameters()]).numpy()
    out = dict(z_train=np.array([e.z for e in train]), z_test=np.array([e.z for e in test]),
               t_train=np.array([e.target for e in train]), t_test=np.array([e.target for e in test]),
               ref=flat(ref), theta=flat(model), P=P.numpy())
    for m in ("dot", "tracin-adam", "ekfac", "cg"):
        out["score_" + m] = score_at_checkpoint(m, model, train, test, precond_vec=P, lambda_damp=0.1,
                                                cg_iters=100, cg_tol=1e-6, beta=0.1, ref_model=ref)
    np.savez({out_path!r}, **out)
""")


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    if not (OLD / "scripts/lds/common.py").is_file() or not OLD_PY.is_file():
        pytest.skip("legacy repo /Users/bui/code/influence-rlvr (with its venv) is not available")
    out = tmp_path_factory.mktemp("legacy") / "tiny.npz"
    code = LEGACY_DUMP.format(old=str(OLD), steps=TINY_STEPS, out_path=str(out), **TINY)
    subprocess.run([str(OLD_PY), "-c", code], cwd=OLD, check=True, capture_output=True, text=True)
    with np.load(out) as d:
        return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def tiny():
    train, test, ref = fixture(FixtureSpec(**TINY))
    model, opt = train_grpo(train, TINY_STEPS, ref_model=ref, lr=1e-3, beta=0.1, seed=42)
    return train, test, ref, model, opt


def _flat(model) -> np.ndarray:
    return torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()


def test_parameter_count_and_order():
    train, test, ref = fixture(FixtureSpec())
    assert n_trainable(ref) == 2196
    assert len(train) == 180 and len(test) == 50
    assert sum(e.expected_influence == "harmful" for e in train) == 60
    assert [n for n, _ in ref.named_parameters()] == [
        "first.0.weight", "first.0.bias", "first.2.weight", "first.2.bias",
        "second.0.weight", "second.0.bias", "second.2.weight", "second.2.bias"]


def test_grpo_gradient_collapses_to_advantage_weighted_score_plus_kl(tiny):
    train, test, ref, model, _ = tiny
    beta = 0.1
    ex = train[3]
    g = grad_vector(model, grpo_loss(model, ex, old_model=clone(model), ref_model=ref, beta=beta))
    seqs = ALL_SEQUENCES
    z = ex.z_tensor()
    r = (seqs == ex.target_tensor().unsqueeze(0)).all(1).float()
    A = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    with torch.no_grad():
        ref_lp = ref.per_token_log_probs(z, seqs)
    expected = torch.zeros_like(g)
    for y in range(4):
        for t in range(2):
            lp = model.per_token_log_probs(z, seqs)[y, t]
            coef = -(1.0 / 8.0) * (A[y] - beta * (1.0 - torch.exp(ref_lp[y, t] - lp.detach())))
            expected += coef * grad_vector(model, lp)
    torch.testing.assert_close(g, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(A[r.argmax()], torch.tensor(1.7316509), rtol=1e-6, atol=1e-6)


def test_exact_expected_reward_is_target_probability(tiny):
    train, test, ref, model, _ = tiny
    for ex in test:
        with torch.no_grad():
            p = model.sequence_log_probs(ex.z_tensor(), ALL_SEQUENCES).exp()
        y_star = int(ex.target[0] * 2 + ex.target[1])
        assert exact_expected_reward(model, ex) == pytest.approx(float(p[y_star]), abs=1e-7)


def test_fixture_matches_legacy_generator(legacy, tiny):
    train, test, ref, _, _ = tiny
    assert np.array_equal(np.array([e.z for e in train]), legacy["z_train"])
    assert np.array_equal(np.array([e.z for e in test]), legacy["z_test"])
    assert np.array_equal(np.array([e.target for e in train]), legacy["t_train"])
    assert np.array_equal(np.array([e.target for e in test]), legacy["t_test"])
    assert np.array_equal(_flat(ref), legacy["ref"])


def test_training_and_adam_state_match_legacy(legacy, tiny):
    _, _, _, model, opt = tiny
    assert np.array_equal(_flat(model), legacy["theta"])
    assert np.array_equal(adam_preconditioner(opt, model).numpy(), legacy["P"])


def test_score_checkpoint_matches_legacy(legacy, tiny):
    train, test, ref, model, opt = tiny
    scores = score_checkpoint(model, opt, train, test, ["dot", "adam", "ekfac", "cg"], ref_model=ref, beta=0.1,
                              lambda_damp=0.1, cg_iters=100, cg_tol=1e-6)
    for new, old in (("dot", "dot"), ("adam", "tracin-adam"), ("ekfac", "ekfac"), ("cg", "cg")):
        assert scores[new].shape == (len(test), len(train))
        assert np.allclose(scores[new], legacy["score_" + old], atol=1e-6), new


def test_precomputed_preconditioner_equals_live_optimizer(tiny):
    train, test, ref, model, opt = tiny
    a = score_checkpoint(model, opt, train, test[:1], ["adam"], ref_model=ref, beta=0.1, lambda_damp=0.1,
                         cg_iters=10, cg_tol=1e-6)["adam"]
    b = score_checkpoint(model, adam_preconditioner(opt, model), train, test[:1], ["adam"], ref_model=ref,
                         beta=0.1, lambda_damp=0.1, cg_iters=10, cg_tol=1e-6)["adam"]
    assert np.array_equal(a, b)
