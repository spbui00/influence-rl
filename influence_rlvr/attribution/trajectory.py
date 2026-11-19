"""NaN policy follows the legacy Table-8 scripts: sum_all/window/single default to per-checkpoint
median fill (plot_backdoor.py), sum_visited/momentum_tail default to NaN -> 0 inside the sum
(p2_exact_backdoor.py nansum). A visit logged at step s counts checkpoint s + offset."""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .metrics import per_target_lds

ScoresByStep = Mapping[int, np.ndarray]
Visits = Mapping[int, Sequence[int]] | np.ndarray


def _stack(scores_by_step: ScoresByStep) -> tuple[np.ndarray, np.ndarray]:
    if not scores_by_step:
        raise ValueError("no checkpoints in range")
    steps = np.asarray(sorted(scores_by_step), dtype=np.int64)
    x = np.stack([np.asarray(scores_by_step[int(s)], dtype=np.float64) for s in steps])
    return steps, x


def fill_nan(x: np.ndarray, nan: str = "median") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if nan == "zero":
        return np.where(np.isnan(x), 0.0, x)
    if nan == "median":
        med = np.nanmedian(x, axis=-1, keepdims=True)
        return np.where(np.isnan(x), med, x)
    if nan == "raise":
        if np.isnan(x).any():
            raise ValueError("NaN scores")
        return x
    raise ValueError(f"unknown nan policy {nan!r}")


def sum_all(scores_by_step: ScoresByStep, lr_by_step: Mapping[int, float] | None = None,
            nan: str = "median") -> np.ndarray:
    steps, x = _stack(scores_by_step)
    if lr_by_step is not None:
        lr = np.asarray([lr_by_step[int(s)] for s in steps], dtype=np.float64)
        x = x * lr.reshape((-1,) + (1,) * (x.ndim - 1))
    if nan == "zero":
        return np.nansum(x, axis=0)
    return fill_nan(x, nan).sum(axis=0)


def window(scores_by_step: ScoresByStep, lo: int, hi: int, nan: str = "median") -> np.ndarray:
    return sum_all({s: v for s, v in scores_by_step.items() if lo <= s <= hi}, None, nan)


def single(scores_by_step: ScoresByStep, step: int, nan: str = "median") -> np.ndarray:
    return fill_nan(np.asarray(scores_by_step[step], dtype=np.float64), nan)


def batch_visits(batch_log: Iterable[Mapping | Sequence[int]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for li, row in enumerate(batch_log):
        if isinstance(row, Mapping):
            step, members = int(row["step"]), row["train_index"]
        else:
            step, members = li + 1, row
        out[step] = np.unique(np.asarray(members, dtype=np.int64))
    return out


def _visit_dict(steps: np.ndarray, visits: Visits) -> dict[int, np.ndarray]:
    if isinstance(visits, np.ndarray):
        if visits.shape[0] != len(steps):
            raise ValueError("visits rows must align with the sorted checkpoint steps")
        return {int(s): np.flatnonzero(visits[c]) for c, s in enumerate(steps)}
    return {int(s): np.unique(np.asarray(m, dtype=np.int64)) for s, m in visits.items()}


def _tail_weights(steps: np.ndarray, n: int, visits: Visits, beta1: float, horizon: int,
                  offset: int) -> np.ndarray:
    index = {int(s): c for c, s in enumerate(steps)}
    w = np.zeros((len(steps), n), dtype=np.float64)
    for s, members in _visit_dict(steps, visits).items():
        for d in range(horizon):
            c = index.get(s + offset + d)
            if c is not None:
                w[c, members] += beta1 ** d
    return w


def _broadcast(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return w.reshape(w.shape[:1] + (1,) * (x.ndim - 2) + w.shape[1:])


def sum_visited(scores_by_step: ScoresByStep, visits: Visits, *, offset: int = 0,
                nan: str = "zero") -> np.ndarray:
    steps, x = _stack(scores_by_step)
    vis = _broadcast(_tail_weights(steps, x.shape[-1], visits, 1.0, 1, offset) > 0, x)
    if nan == "zero":
        return np.nansum(np.where(vis, x, 0.0), axis=0)
    return (vis * fill_nan(x, nan)).sum(axis=0)


def momentum_tail(scores_by_step: ScoresByStep, visits: Visits, beta1: float = 0.9,
                  horizon: int = 10, *, offset: int = 0, nan: str = "zero") -> np.ndarray:
    steps, x = _stack(scores_by_step)
    w = _broadcast(_tail_weights(steps, x.shape[-1], visits, beta1, horizon, offset), x)
    if nan == "zero":
        return np.nansum(w * np.nan_to_num(x), axis=0)
    return (w * fill_nan(x, nan)).sum(axis=0)


def visit_counts(mask: np.ndarray, ks: np.ndarray) -> np.ndarray:
    s = int(np.asarray(mask, dtype=bool).sum())
    ks = np.asarray(ks, dtype=np.int64).reshape(-1, 1)
    return ks // s + (np.arange(s)[None, :] < (ks % s)).astype(np.int64)


def visit_counts_perm(s: int, ks: np.ndarray, order: np.ndarray) -> np.ndarray:
    rank = np.argsort(np.asarray(order))
    ks = np.asarray(ks, dtype=np.int64).reshape(-1, 1)
    return ks // s + (rank[None, :] < (ks % s)).astype(np.int64)


def response_weighted_visits(s: int, ks: np.ndarray, order: np.ndarray | None,
                             beta1: float) -> np.ndarray:
    ks = np.asarray(ks, dtype=np.int64)
    K = int(ks.max())
    steps = np.arange(1, K + 1)
    pos = (steps - 1) % s
    ex = pos if order is None else np.asarray(order)[pos]
    lag = np.maximum(ks[:, None] - steps[None, :] + 1, 1)
    w = np.where(steps[None, :] <= ks[:, None], 1.0 - beta1 ** lag, 0.0)
    bins = (np.arange(len(ks))[:, None] * s + ex[None, :]).ravel()
    return np.bincount(bins, weights=w.ravel(), minlength=len(ks) * s).reshape(len(ks), s)


def _padded_members(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = masks.sum(1)
    rows, cols = np.nonzero(masks)
    pos = np.arange(len(cols)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    idx = np.zeros((masks.shape[0], int(sizes.max())), dtype=np.int64)
    valid = np.zeros(idx.shape, dtype=bool)
    idx[rows, pos] = cols
    valid[rows, pos] = True
    return idx, valid, sizes


def subset_prediction(scores: np.ndarray, masks: np.ndarray,
                      visits: np.ndarray | Sequence[np.ndarray] | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    idx, valid, sizes = _padded_members(masks)
    M, smax = idx.shape
    if visits is None:
        v = valid[:, None, :].astype(np.float64)
    elif isinstance(visits, np.ndarray):
        if visits.shape[1] != smax or not np.all(sizes == smax):
            raise ValueError("a shared visit matrix needs equal-size subsets")
        v = np.broadcast_to(visits.astype(np.float64)[None], (M,) + visits.shape)
    else:
        K = np.asarray(visits[0]).shape[0]
        v = np.zeros((M, K, smax), dtype=np.float64)
        for m in range(M):
            v[m, :, :sizes[m]] = visits[m]
    pred = np.matmul(v, scores[:, idx].transpose(1, 2, 0)).transpose(1, 0, 2)
    return pred[0] if visits is None else pred


VisitWeights = Callable[[np.ndarray, np.ndarray], np.ndarray]


def lds_curves(scores: np.ndarray, masks: np.ndarray, outcomes: np.ndarray, ks: np.ndarray,
               visit_weights: VisitWeights | Sequence[np.ndarray] | None = None,
               ) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(masks, dtype=bool)
    ks = np.asarray(ks, dtype=np.int64)
    if visit_weights is None:
        visits = [visit_counts(m, ks) for m in masks]
    elif callable(visit_weights):
        visits = [visit_weights(m, ks) for m in masks]
    else:
        visits = list(visit_weights)
    pred = subset_prediction(scores, masks, visits).transpose(1, 0, 2)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if outcomes.shape != pred.shape:
        raise ValueError(f"outcomes must be (M, K, T) = {pred.shape}, got {outcomes.shape}")
    per_target = per_target_lds(pred, outcomes, "ordinal").T
    return per_target, per_target.mean(axis=0)
