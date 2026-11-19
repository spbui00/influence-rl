from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from influence_rlvr.attribution import metrics as mt


def _legacy_spearman_ordinal(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def _legacy_spearman_average(a, b):
    def rank(x):
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        avg = (cum - 1 + cum - counts) / 2.0
        return avg[inv]
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _legacy_band(curves, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = curves.shape[0]
    boots = curves[rng.integers(0, n, (B, n))].mean(1)
    return curves.mean(0), np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)


def _legacy_boot_p(d, B=20000, seed=0):
    rng = np.random.default_rng(seed); n = len(d)
    m = d[rng.integers(0, n, (B, n))].mean(1)
    return 2 * min((m <= 0).mean(), (m >= 0).mean())


def _brute_auc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _legacy_auc_precision(scores, groups, k=None):
    P = [i for i, g in enumerate(groups) if g == "poison"]
    N = [i for i, g in enumerate(groups) if g != "poison"]
    if k is None:
        k = len(P)
    sp, sn = scores[P], scores[N]
    wins = (sp[:, None] < sn[None, :]).sum()
    ties = (sp[:, None] == sn[None, :]).sum()
    auc_low = float((wins + 0.5 * ties) / (len(P) * len(N)))
    order_low = sorted(range(len(scores)), key=lambda i: scores[i])
    prec_low = sum(groups[i] == "poison" for i in order_low[:k]) / k
    prec_high = sum(groups[i] == "poison" for i in order_low[-k:]) / k
    return auc_low, max(auc_low, 1 - auc_low), auc_low >= 0.5, prec_low, prec_high


def test_spearman_variants_agree_with_scipy_without_ties():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.standard_normal(60), rng.standard_normal(60)
        ref = spearmanr(a, b).statistic
        assert abs(mt.spearman(a, b) - ref) < 1e-12
        assert abs(mt.spearman_ordinal(a, b) - ref) < 1e-12
        assert abs(mt.lds(a, b, "ordinal") - ref) < 1e-12
        assert abs(mt.lds(a, b, "average") - ref) < 1e-12


def test_spearman_average_ranks_with_ties_matches_scipy_and_legacy():
    rng = np.random.default_rng(1)
    for _ in range(5):
        a = rng.integers(0, 5, 80).astype(float)
        b = rng.integers(0, 4, 80).astype(float) + rng.standard_normal(80) * (rng.random(80) < 0.3)
        assert abs(mt.spearman(a, b) - spearmanr(a, b).statistic) < 1e-12
        assert abs(mt.spearman(a, b) - _legacy_spearman_average(a, b)) < 1e-15


def test_spearman_ordinal_with_ties_matches_legacy():
    rng = np.random.default_rng(2)
    for _ in range(5):
        a = rng.integers(0, 6, 50).astype(float)
        b = rng.standard_normal(50)
        assert mt.spearman_ordinal(a, b) == _legacy_spearman_ordinal(a, b)
        assert mt.spearman_ordinal(b, a) == _legacy_spearman_ordinal(b, a)


def test_zero_variance_behaviour():
    a = np.ones(10)
    b = np.arange(10, dtype=float)
    assert mt.spearman_ordinal(a, b) == 0.0
    assert mt.spearman_ordinal(b, a) == 0.0
    assert mt.spearman_ordinal(np.array([1.0]), np.array([2.0])) == 0.0
    assert np.isnan(mt.spearman(a, b))
    assert np.isnan(mt.spearman(b, a))
    assert np.isnan(mt.lds(a, b, "average"))


def test_per_target_lds_batched_matches_loop():
    rng = np.random.default_rng(3)
    M, T = 40, 9
    pred = rng.integers(0, 7, (M, T)).astype(float)
    y = rng.integers(0, 3, (M, T)).astype(float) + 0.01 * rng.standard_normal((M, T)) * (rng.random((M, T)) < 0.5)
    y[:, 3] = 0.5
    for kind, fn in (("ordinal", _legacy_spearman_ordinal), ("average", _legacy_spearman_average)):
        got = mt.per_target_lds(pred, y, kind)
        ref = np.array([fn(pred[:, t], y[:, t]) for t in range(T)])
        assert got.shape == (T,)
        if kind == "ordinal":
            assert got[3] == 0.0
            ref[3] = 0.0
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        ok = ~np.isnan(ref)
        np.testing.assert_allclose(got[ok], ref[ok], rtol=0, atol=1e-15)
    got3 = mt.per_target_lds(pred.reshape(M, 3, 3), y.reshape(M, 3, 3), "ordinal")
    np.testing.assert_array_equal(got3.ravel(), mt.per_target_lds(pred, y, "ordinal"))


def test_bootstrap_band_bit_for_bit_with_legacy():
    rng = np.random.default_rng(4)
    curves = rng.random((50, 30))
    mean, lo, hi = mt.bootstrap_band(curves, B=1000, seed=0)
    rmean, rlo, rhi = _legacy_band(curves, B=1000, seed=0)
    np.testing.assert_array_equal(mean, rmean)
    np.testing.assert_array_equal(lo, rlo)
    np.testing.assert_array_equal(hi, rhi)
    assert np.all(lo <= mean) and np.all(mean <= hi)


def test_bootstrap_ci_mean_and_shared_rng():
    rng = np.random.default_rng(5)
    v = rng.random(50)
    m, lo, hi = mt.bootstrap_ci_mean(v, B=5000, seed=0)
    rm, rlo, rhi = _legacy_band(v, B=5000, seed=0)
    assert m == float(rm) and lo == float(rlo) and hi == float(rhi)
    shared = np.random.default_rng(0)
    m1, lo1, hi1 = mt.bootstrap_ci_mean(v, B=5000, rng=shared)
    m2, lo2, hi2 = mt.bootstrap_ci_mean(v, B=5000, rng=shared)
    assert (lo1, hi1) == (lo, hi) and (lo2, hi2) != (lo, hi)
    ref_rng = np.random.default_rng(0)
    ref_rng.integers(0, 50, (5000, 50))
    b = v[ref_rng.integers(0, 50, (5000, 50))].mean(1)
    assert (lo2, hi2) == (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def test_paired_bootstrap_p_bit_for_bit_with_legacy():
    rng = np.random.default_rng(6)
    a = rng.random(50)
    b = a + 0.02 * rng.standard_normal(50) + 0.01
    p = mt.paired_bootstrap_p(a, b, B=20000, seed=0)
    assert p == _legacy_boot_p(a - b, B=20000, seed=0)
    assert 0.0 <= p <= 1.0
    assert mt.paired_bootstrap_p(a, a + 1.0) == 0.0
    assert mt.paired_bootstrap_p(a, a) == 1.0


def test_auc_rank_based_equals_brute_force_with_ties():
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = 120
        labels = rng.random(n) < 0.3
        scores = rng.integers(0, 15, n).astype(float) + rng.standard_normal(n) * (rng.random(n) < 0.4)
        assert abs(mt.auc(scores, labels) - _brute_auc(scores, labels)) < 1e-12
    scores = np.arange(10, dtype=float)
    labels = np.zeros(10, bool); labels[7:] = True
    assert mt.auc(scores, labels) == 1.0
    assert mt.auc(-scores, labels) == 0.0
    assert mt.auc(np.zeros(10), labels) == 0.5


def test_auc_nan_policy():
    rng = np.random.default_rng(8)
    s = rng.standard_normal(50)
    lab = rng.random(50) < 0.4
    s[3] = np.nan
    with pytest.raises(ValueError):
        mt.auc(s, lab)
    filled = np.where(np.isnan(s), np.nanmedian(s), s)
    assert mt.auc(s, lab, nan="median") == mt.auc(filled, lab)
    assert mt.auc(s, lab, nan="zero") == mt.auc(np.nan_to_num(s), lab)


def test_oriented_auc_and_precision_match_legacy_auc_precision():
    rng = np.random.default_rng(9)
    n = 300
    labels = rng.random(n) < 0.2
    for sign in (-1.0, 1.0):
        scores = rng.standard_normal(n) + sign * 1.5 * labels
        scores[rng.random(n) < 0.2] = 0.0
        groups = ["poison" if l else "background" for l in labels]
        auc_low, sep, more_harmful, prec_low, prec_high = _legacy_auc_precision(scores, groups)
        assert abs(mt.auc(-scores, labels) - auc_low) < 1e-12
        got_sep, lower = mt.oriented_auc(scores, labels)
        assert abs(got_sep - sep) < 1e-12 and lower == more_harmful
        k = int(labels.sum())
        assert mt.precision_at_k(scores, labels, k, lower_is_positive=True) == prec_low
        assert mt.precision_at_k(scores, labels, k, lower_is_positive=False) == prec_high
        s = np.where(np.isnan(scores), np.nanmedian(scores), scores)
        order = np.argsort(s, kind="stable")
        top = order[:k] if auc_low >= 0.5 else order[-k:]
        assert mt.precision_at_k(scores, labels, k, lower) == labels[top].mean()
    with pytest.raises(ValueError):
        mt.precision_at_k(scores, labels, 0)


def test_precision_at_k_tie_block_matches_legacy_stable_sort():
    rng = np.random.default_rng(11)
    n, k = 1000, 200
    labels = rng.random(n) < 0.2
    scores = rng.standard_normal(n) - 1.5 * labels
    scores[rng.random(n) < 0.85] = 0.0
    groups = ["poison" if l else "background" for l in labels]
    _, _, _, prec_low, prec_high = _legacy_auc_precision(scores, groups, k)
    assert mt.precision_at_k(scores, labels, k, lower_is_positive=True) == prec_low
    assert mt.precision_at_k(scores, labels, k, lower_is_positive=False) == prec_high


def test_binomial_noise_floor_matches_loop():
    rng = np.random.default_rng(10)
    M, T, n = 100, 32, 16
    y = rng.random((M, T))
    ref = 0.0
    for t in range(T):
        p = y[:, t].mean()
        ref += p * (1 - p) / n
    ref = np.sqrt(ref) / T
    assert abs(mt.binomial_noise_floor(y, n) - ref) < 1e-15


def test_variance_decomposition_recovers_binomial_noise_floor():
    rng = np.random.default_rng(11)
    M, T, n = 4000, 32, 16
    p = rng.uniform(0.2, 0.8, T)
    y = rng.binomial(n, p, (M, T)) / n
    out = mt.variance_decomposition(y, n)
    assert set(out) == {"observed_sd", "noise_floor", "residual_sd"}
    assert abs(out["observed_sd"] - out["noise_floor"]) / out["noise_floor"] < 0.06
    assert out["residual_sd"] < 0.4 * out["noise_floor"]
    assert out["observed_sd"] == float(y.mean(1).std(ddof=1))
    shift = rng.standard_normal(M)[:, None] * 0.05
    y2 = np.clip(y + shift, 0, 1)
    out2 = mt.variance_decomposition(y2, n)
    assert out2["residual_sd"] > 0.03
    assert abs(out2["residual_sd"] - np.sqrt(out2["observed_sd"] ** 2 - out2["noise_floor"] ** 2)) < 1e-15


def test_split_half_reliability_noise_free_is_one():
    rng = np.random.default_rng(12)
    M, T = 60, 32
    y = np.repeat(rng.random((M, 1)), T, axis=1)
    mean, per_split = mt.split_half_reliability(y, n_splits=50, seed=0)
    assert per_split.shape == (50,)
    np.testing.assert_allclose(per_split, 1.0, rtol=0, atol=1e-12)
    assert abs(mean - 1.0) < 1e-12
    noise = rng.random((M, T))
    m1, s1 = mt.split_half_reliability(noise, n_splits=50, seed=3)
    m2, s2 = mt.split_half_reliability(noise, n_splits=50, seed=3)
    np.testing.assert_array_equal(s1, s2)
    assert abs(m1) < 0.5


def test_split_half_reliability_matches_loop():
    rng = np.random.default_rng(13)
    M, T = 30, 10
    y = rng.random((M, T))
    _, per_split = mt.split_half_reliability(y, n_splits=20, seed=7)
    r = np.random.default_rng(7)
    perms = np.argsort(r.random((20, T)), axis=1)
    for i in range(20):
        A, B = perms[i, :T // 2], perms[i, T // 2:]
        ref = mt.spearman(y[:, A].mean(1), y[:, B].mean(1))
        assert abs(per_split[i] - ref) < 1e-15
