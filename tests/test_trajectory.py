from __future__ import annotations

import numpy as np
import pytest

from influence_rlvr.attribution import trajectory as tr
from influence_rlvr.attribution.metrics import spearman_ordinal


def _legacy_response_weighted_visits(s, ks, order, beta1):
    K = int(np.max(ks)); steps = np.arange(1, K + 1)
    pos = (steps - 1) % s
    ex = pos if order is None else np.asarray(order)[pos]
    out = np.zeros((len(ks), s))
    for r, k in enumerate(ks):
        w = 1.0 - beta1 ** (k - steps[:k] + 1)
        out[r] = np.bincount(ex[:k], weights=w, minlength=s)
    return out


def _simulate_visits(s, ks, order=None):
    out = np.zeros((len(ks), s), dtype=np.int64)
    for r, k in enumerate(ks):
        for step in range(1, k + 1):
            pos = (step - 1) % s
            out[r, pos if order is None else order[pos]] += 1
    return out


def _scores_by_step(rng, steps, shape, nan_cols=()):
    out = {}
    for s in steps:
        x = rng.standard_normal(shape)
        for c in nan_cols:
            x[..., c] = np.nan
        out[s] = x
    return out


def test_visit_counts_matches_simulation():
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, 37).astype(bool)
    ks = np.array([1, 5, 13, 40, 97])
    s = int(mask.sum())
    np.testing.assert_array_equal(tr.visit_counts(mask, ks), _simulate_visits(s, ks))
    assert tr.visit_counts(mask, [40]).shape == (1, s)


def test_visit_counts_perm_matches_simulation():
    rng = np.random.default_rng(1)
    s, ks = 11, np.array([3, 11, 12, 50])
    order = rng.permutation(s)
    np.testing.assert_array_equal(tr.visit_counts_perm(s, ks, order), _simulate_visits(s, ks, order))


@pytest.mark.parametrize("use_order", [False, True])
def test_response_weighted_visits_bit_for_bit(use_order):
    rng = np.random.default_rng(2)
    s = 23
    ks = np.arange(10, 301, 10)
    order = rng.permutation(s) if use_order else None
    got = tr.response_weighted_visits(s, ks, order, 0.9)
    ref = _legacy_response_weighted_visits(s, ks, order, 0.9)
    assert np.array_equal(got, ref)
    plain = tr.response_weighted_visits(s, ks, order, 0.0)
    np.testing.assert_array_equal(plain, _simulate_visits(s, ks, order))


def test_subset_prediction_plain_sum_fixed_size():
    rng = np.random.default_rng(3)
    T, N, M, S = 4, 30, 12, 9
    scores = rng.standard_normal((T, N))
    masks = np.zeros((M, N), dtype=bool)
    for m in range(M):
        masks[m, rng.choice(N, S, replace=False)] = True
    pred = tr.subset_prediction(scores, masks, None)
    assert pred.shape == (M, T)
    ref = np.array([[scores[t, masks[m]].sum() for t in range(T)] for m in range(M)])
    np.testing.assert_allclose(pred, ref, rtol=1e-12, atol=1e-12)


def test_subset_prediction_shared_visits():
    rng = np.random.default_rng(4)
    T, N, M, S, K = 3, 20, 7, 6, 5
    scores = rng.standard_normal((T, N))
    masks = np.zeros((M, N), dtype=bool)
    for m in range(M):
        masks[m, rng.choice(N, S, replace=False)] = True
    V = rng.random((K, S))
    pred = tr.subset_prediction(scores, masks, V)
    assert pred.shape == (K, M, T)
    for m in range(M):
        idx = np.flatnonzero(masks[m])
        np.testing.assert_allclose(pred[:, m, :], V @ scores[:, idx].T, rtol=1e-12, atol=1e-12)


def test_subset_prediction_per_subset_visits_matches_legacy_loop():
    rng = np.random.default_rng(5)
    T, N, M = 5, 40, 9
    ks = np.array([10, 20, 35])
    scores = rng.standard_normal((T, N))
    masks = rng.integers(0, 2, (M, N)).astype(bool)
    visits = [tr.response_weighted_visits(int(m.sum()), ks, None, 0.9) for m in masks]
    pred = tr.subset_prediction(scores, masks, visits)
    assert pred.shape == (len(ks), M, T)
    for m in range(M):
        idx = np.flatnonzero(masks[m])
        ref = visits[m] @ scores[:, idx].T
        np.testing.assert_allclose(pred[:, m, :], ref, rtol=1e-12, atol=1e-12)
        for k in range(len(ks)):
            for t in range(T):
                brute = sum(visits[m][k, p] * scores[t, n] for p, n in enumerate(idx))
                assert abs(pred[k, m, t] - brute) < 1e-10


def test_sum_all_policies_and_lr():
    rng = np.random.default_rng(6)
    steps = [3, 1, 2]
    sbs = _scores_by_step(rng, steps, (8,), nan_cols=(2,))
    sbs[2][5] = np.nan
    x = np.stack([sbs[s] for s in sorted(steps)])
    zero = tr.sum_all(sbs, nan="zero")
    np.testing.assert_array_equal(zero, np.nansum(x, 0))
    med = tr.sum_all(sbs, nan="median")
    ref = np.stack([np.where(np.isnan(r), np.nanmedian(r), r) for r in x]).sum(0)
    np.testing.assert_array_equal(med, ref)
    lr = {1: 0.5, 2: 0.25, 3: 2.0}
    got = tr.sum_all(sbs, lr_by_step=lr, nan="zero")
    ref = np.nansum(np.stack([lr[s] * sbs[s] for s in sorted(steps)]), 0)
    np.testing.assert_array_equal(got, ref)
    tot = np.zeros(8)
    clean = {s: np.nan_to_num(v) for s, v in sbs.items()}
    for s in sorted(steps):
        tot += 0.001 * clean[s]
    np.testing.assert_array_equal(tr.sum_all(clean, lr_by_step={s: 0.001 for s in steps}), tot)


def test_sum_all_median_fill_per_target_row():
    rng = np.random.default_rng(7)
    sbs = _scores_by_step(rng, [1, 2], (3, 6))
    sbs[1][0, 1] = np.nan
    sbs[1][2, 4] = np.nan
    got = tr.sum_all(sbs, nan="median")
    x1 = sbs[1].copy()
    x1[0, 1] = np.nanmedian(sbs[1][0])
    x1[2, 4] = np.nanmedian(sbs[1][2])
    np.testing.assert_array_equal(got, x1 + sbs[2])


def test_window_and_single():
    rng = np.random.default_rng(8)
    sbs = _scores_by_step(rng, range(1, 11), (5,))
    sbs[4][0] = np.nan
    ref = sum(np.where(np.isnan(sbs[s]), np.nanmedian(sbs[s]), sbs[s]) for s in range(4, 8))
    np.testing.assert_allclose(tr.window(sbs, 4, 7), ref, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(tr.window(sbs, 9, 10, nan="zero"), sbs[9] + sbs[10])
    with pytest.raises(ValueError):
        tr.window(sbs, 100, 200)
    np.testing.assert_array_equal(tr.single(sbs, 5), sbs[5])
    s4 = tr.single(sbs, 4)
    assert s4[0] == np.nanmedian(sbs[4]) and np.array_equal(s4[1:], sbs[4][1:])
    with pytest.raises(ValueError):
        tr.single(sbs, 4, nan="raise")


def test_sum_visited_offset_semantics():
    rng = np.random.default_rng(9)
    N = 6
    sbs = _scores_by_step(rng, range(1, 7), (N,))
    sbs[3][0] = np.nan
    visits = {1: [0, 4], 2: [1, 2, 2], 5: [3], 6: [5]}
    got = tr.sum_visited(sbs, visits, offset=2)
    ref = np.zeros(N)
    ref[0] += 0.0
    ref[4] += sbs[3][4]
    ref[1] += sbs[4][1]
    ref[2] += sbs[4][2]
    np.testing.assert_array_equal(got, ref)
    got0 = tr.sum_visited(sbs, visits, offset=0)
    ref0 = np.zeros(N)
    ref0[0] += sbs[1][0]; ref0[4] += sbs[1][4]; ref0[1] += sbs[2][1]; ref0[2] += sbs[2][2]
    ref0[3] += sbs[5][3]; ref0[5] += sbs[6][5]
    np.testing.assert_array_equal(got0, ref0)
    got_med = tr.sum_visited(sbs, visits, offset=2, nan="median")
    ref_med = ref.copy()
    ref_med[0] = np.nanmedian(sbs[3])
    np.testing.assert_array_equal(got_med, ref_med)


def test_sum_visited_bool_matrix_and_batch_log_agree():
    rng = np.random.default_rng(10)
    C, N = 5, 9
    sbs = _scores_by_step(rng, range(1, C + 1), (N,))
    vis = rng.integers(0, 2, (C, N)).astype(bool)
    rows = [{"step": c + 1, "train_index": np.flatnonzero(vis[c]).tolist() * 2} for c in range(C)]
    lines = [np.flatnonzero(vis[c]).tolist() for c in range(C)]
    a = tr.sum_visited(sbs, vis)
    b = tr.sum_visited(sbs, tr.batch_visits(rows))
    c = tr.sum_visited(sbs, tr.batch_visits(lines))
    x = np.stack([sbs[s] for s in range(1, C + 1)])
    np.testing.assert_array_equal(a, np.nansum(np.where(vis, x, 0), 0))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_sum_visited_two_dim_scores():
    rng = np.random.default_rng(11)
    sbs = _scores_by_step(rng, [1, 2, 3], (4, 7))
    visits = {1: [0, 6], 3: [2]}
    got = tr.sum_visited(sbs, visits)
    ref = np.zeros((4, 7))
    ref[:, 0] += sbs[1][:, 0]; ref[:, 6] += sbs[1][:, 6]; ref[:, 2] += sbs[3][:, 2]
    np.testing.assert_array_equal(got, ref)


def test_momentum_tail_matches_explicit_loop():
    rng = np.random.default_rng(12)
    N, C = 15, 40
    steps = list(range(1, C + 1))
    sbs = _scores_by_step(rng, steps, (N,), nan_cols=(7,))
    L = [sorted(set(rng.integers(0, N, 5).tolist())) for _ in range(35)]
    X = np.stack([sbs[s] for s in steps])
    for offset in (0, 12):
        w = np.zeros((C, N))
        for li, b in enumerate(L):
            st = li + 1 + offset
            for d in range(10):
                if st + d in sbs:
                    w[steps.index(st + d), b] += 0.9 ** d
        ref = np.nansum(w * np.nan_to_num(X), 0)
        got = tr.momentum_tail(sbs, tr.batch_visits(L), beta1=0.9, horizon=10, offset=offset)
        np.testing.assert_array_equal(got, ref)
    got1 = tr.momentum_tail(sbs, tr.batch_visits(L), beta1=0.9, horizon=1)
    np.testing.assert_array_equal(got1, tr.sum_visited(sbs, tr.batch_visits(L)))


def test_lds_curves_matches_loop():
    rng = np.random.default_rng(13)
    T, N, M = 6, 30, 25
    ks = np.array([10, 20, 30, 40])
    scores = rng.standard_normal((T, N))
    masks = rng.integers(0, 2, (M, N)).astype(bool)
    outcomes = rng.random((M, len(ks), T))
    weights = lambda mask, ks_: tr.response_weighted_visits(int(mask.sum()), ks_, None, 0.9)
    per_target, mean = tr.lds_curves(scores, masks, outcomes, ks, weights)
    assert per_target.shape == (T, len(ks)) and mean.shape == (len(ks),)
    with pytest.raises(ValueError):
        tr.lds_curves(scores, masks, outcomes.transpose(1, 0, 2), ks, weights)
    preds = np.zeros((M, len(ks), T))
    for j in range(M):
        idx = np.flatnonzero(masks[j])
        V = weights(masks[j], ks)
        preds[j] = V @ scores[:, idx].T
    ref = np.zeros((T, len(ks)))
    for ki in range(len(ks)):
        for i in range(T):
            ref[i, ki] = spearman_ordinal(preds[:, ki, i], outcomes[:, ki, i])
    np.testing.assert_allclose(per_target, ref, rtol=0, atol=1e-13)
    np.testing.assert_allclose(mean, ref.mean(0), rtol=0, atol=1e-13)
    per_plain, _ = tr.lds_curves(scores, masks, outcomes, ks)
    per_explicit, _ = tr.lds_curves(scores, masks, outcomes, ks, [tr.visit_counts(m, ks) for m in masks])
    np.testing.assert_array_equal(per_plain, per_explicit)
