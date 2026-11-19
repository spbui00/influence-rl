"""Two Spearman conventions are kept deliberately: `spearman_ordinal` (double-argsort ranks, ties broken
by sort order; 0 on a constant input, where legacy returned an argsort artefact that no shipped toy number
hits) produces the toy numbers; `spearman` (average ranks, NaN on zero variance) the LLM numbers. AUC and
precision: lower score = harmful; precision_at_k ties follow legacy auc_precision (stable sort, [:k]/[-k:])."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _ordinal_ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float64)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average", axis=0).astype(np.float64)


def _rank_corr(a: np.ndarray, b: np.ndarray, kind: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if kind == "ordinal":
        ra, rb = _ordinal_ranks(a), _ordinal_ranks(b)
    elif kind == "average":
        ra, rb = _average_ranks(a), _average_ranks(b)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    ra = ra - ra.mean(axis=0)
    rb = rb - rb.mean(axis=0)
    d = np.sqrt((ra ** 2).sum(axis=0) * (rb ** 2).sum(axis=0))
    num = (ra * rb).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = num / d
    if kind == "ordinal":
        degenerate = (d <= 0) | (a.max(axis=0) == a.min(axis=0)) | (b.max(axis=0) == b.min(axis=0))
        return np.where(degenerate, 0.0, rho)
    return np.where(d > 0, rho, np.nan)


def spearman_ordinal(a: np.ndarray, b: np.ndarray) -> float:
    return float(_rank_corr(np.ravel(a), np.ravel(b), "ordinal"))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(_rank_corr(np.ravel(a), np.ravel(b), "average"))


def lds(pred: np.ndarray, y: np.ndarray, kind: str = "ordinal") -> float:
    return float(_rank_corr(np.ravel(pred), np.ravel(y), kind))


def per_target_lds(pred: np.ndarray, y: np.ndarray, kind: str = "ordinal") -> np.ndarray:
    return _rank_corr(pred, y, kind)


def _boot_means(values: np.ndarray, idx: np.ndarray, chunk: int = 256) -> np.ndarray:
    out = np.empty((idx.shape[0],) + values.shape[1:], dtype=np.float64)
    for i in range(0, idx.shape[0], chunk):
        out[i:i + chunk] = values[idx[i:i + chunk]].mean(axis=1)
    return out


def bootstrap_band(values: np.ndarray, B: int = 5000, seed: int = 0,
                   q: tuple[float, float] = (2.5, 97.5),
                   rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed) if rng is None else rng
    n = values.shape[0]
    boots = _boot_means(values, rng.integers(0, n, (B, n)))
    return values.mean(axis=0), np.percentile(boots, q[0], axis=0), np.percentile(boots, q[1], axis=0)


def bootstrap_ci_mean(values: np.ndarray, B: int = 5000, seed: int = 0,
                      q: tuple[float, float] = (2.5, 97.5),
                      rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    mean, lo, hi = bootstrap_band(np.ravel(values), B, seed, q, rng)
    return float(mean), float(lo), float(hi)


def paired_bootstrap_p(a: np.ndarray, b: np.ndarray, B: int = 20000, seed: int = 0) -> float:
    d = np.ravel(np.asarray(a, dtype=np.float64)) - np.ravel(np.asarray(b, dtype=np.float64))
    rng = np.random.default_rng(seed)
    n = len(d)
    m = _boot_means(d, rng.integers(0, n, (B, n)))
    return float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))


def split_half_reliability(y: np.ndarray, n_splits: int = 200, seed: int = 0) -> tuple[float, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    T = y.shape[1]
    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((n_splits, T)), axis=1)
    h = T // 2
    ma = y[:, perms[:, :h]].mean(axis=2)
    mb = y[:, perms[:, h:]].mean(axis=2)
    per_split = _rank_corr(ma, mb, "average")
    return float(np.nanmean(per_split)), per_split


def binomial_noise_floor(y: np.ndarray, n_samples: int) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = y.mean(axis=0)
    return float(np.sqrt((p * (1.0 - p) / n_samples).sum()) / y.shape[1])


def variance_decomposition(y: np.ndarray, n_samples: int) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    observed = float(y.mean(axis=1).std(ddof=1))
    floor = binomial_noise_floor(y, n_samples)
    residual = float(np.sqrt(max(0.0, observed ** 2 - floor ** 2)))
    return {"observed_sd": observed, "noise_floor": floor, "residual_sd": residual}


def _finite_scores(scores: np.ndarray, nan: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    bad = np.isnan(s)
    if not bad.any():
        return s
    if nan == "raise":
        raise ValueError("NaN scores; pass nan='median' or nan='zero'")
    if nan == "median":
        return np.where(bad, np.nanmedian(s), s)
    if nan == "zero":
        return np.where(bad, 0.0, s)
    raise ValueError(f"unknown nan policy {nan!r}")


def auc(scores: np.ndarray, labels: np.ndarray, nan: str = "raise") -> float:
    s = _finite_scores(scores, nan)
    lab = np.asarray(labels, dtype=bool)
    n_pos = int(lab.sum())
    n_neg = lab.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc needs both classes")
    r = rankdata(s, method="average")
    return float((r[lab].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def oriented_auc(scores: np.ndarray, labels: np.ndarray, nan: str = "raise") -> tuple[float, bool]:
    a_low = auc(-_finite_scores(scores, nan), labels)
    return max(a_low, 1.0 - a_low), bool(a_low >= 0.5)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int, lower_is_positive: bool = True,
                   nan: str = "raise") -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    s = _finite_scores(scores, nan)
    lab = np.asarray(labels, dtype=bool)
    order = np.argsort(s, kind="stable")
    top = order[:k] if lower_is_positive else order[-k:]
    return float(lab[top].mean())
