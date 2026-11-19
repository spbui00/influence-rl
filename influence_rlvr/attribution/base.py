"""Sign convention: g_train = ∇L_GRPO and g_test = ∇(−f) are both LOSS gradients in trainable_params
order; estimators return g_test^T M g_train with no negation, positive = helpful (the paper's I_f).
Numerics: pool scores are one batched fp32 GEMM unless pairwise=True (the legacy per-pair torch.dot, bit-for-bit,
toy); the legacy bf16 per-target rows (bf16(H) @ bf16(g_train)) are reproduced only with row_dtype=bfloat16."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import torch

Tensor = torch.Tensor


class Estimator(ABC):
    name: str = "base"

    @abstractmethod
    def solve(self, g_test: Tensor) -> Tensor:
        ...

    def solve_many(self, g_test: Tensor) -> Tensor:
        if g_test.dim() == 1:
            return self.solve(g_test).unsqueeze(0)
        return torch.stack([self.solve(g) for g in g_test])

    @staticmethod
    def influence(h: Tensor, g_train: Tensor, row_dtype: torch.dtype | None = None) -> Tensor:
        h2 = h.reshape(-1, h.shape[-1])
        g2 = g_train.reshape(-1, g_train.shape[-1]).to(device=h2.device, dtype=h2.dtype)
        if row_dtype is None or row_dtype == h2.dtype:
            return h2 @ g2.T
        return (h2.to(row_dtype) @ g2.to(row_dtype).T).to(h2.dtype)


def unit_rows(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def pairwise_influence(h: Tensor, g_train: Tensor) -> Tensor:
    h2 = h.reshape(-1, h.shape[-1]).to(torch.float32)
    g2 = g_train.reshape(-1, g_train.shape[-1]).to(device=h2.device, dtype=torch.float32)
    rows = [g2[j].clone() for j in range(g2.shape[0])]
    out = torch.zeros(h2.shape[0], g2.shape[0], dtype=torch.float64)
    for i in range(h2.shape[0]):
        hi = h2[i].clone()
        out[i] = torch.tensor([float(torch.dot(r, hi)) for r in rows], dtype=torch.float64)
    return out.to(h2.device)


def influence_matrix(
    estimator: Estimator,
    g_test: Tensor,
    train_grads: Tensor | Iterable[tuple[np.ndarray, Tensor]],
    *,
    n_train: int | None = None,
    cosine: bool = False,
    device: torch.device | str | None = None,
    row_dtype: torch.dtype | None = None,
    pairwise: bool = False,
) -> np.ndarray:
    g_test = g_test.reshape(-1, g_test.shape[-1]).to(dtype=torch.float32)
    if device is not None:
        g_test = g_test.to(device)
    if pairwise:
        h = torch.stack([estimator.solve(g.clone()).to(dtype=torch.float32) for g in g_test])
    else:
        h = estimator.solve_many(g_test).to(dtype=torch.float32)
    if cosine:
        h = unit_rows(h)
    reduce = pairwise_influence if pairwise else (lambda a, b: Estimator.influence(a, b, row_dtype))
    if isinstance(train_grads, torch.Tensor):
        g = train_grads.reshape(-1, train_grads.shape[-1]).to(device=h.device, dtype=torch.float32)
        if cosine:
            g = unit_rows(g)
        return reduce(h, g).double().cpu().numpy()
    if n_train is None:
        raise ValueError("n_train is required when train_grads is streamed")
    out = np.zeros((h.shape[0], n_train), dtype=np.float64)
    for idx, batch in train_grads:
        g = batch.reshape(-1, batch.shape[-1]).to(device=h.device, dtype=torch.float32)
        if cosine:
            g = unit_rows(g)
        out[:, np.asarray(idx)] = reduce(h, g).double().cpu().numpy()
    return out
