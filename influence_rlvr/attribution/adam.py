from __future__ import annotations

import torch

from .base import Estimator

Tensor = torch.Tensor


class AdamDiagonal(Estimator):
    name = "adam"

    def __init__(self, preconditioner: Tensor, lr: float = 1.0):
        if preconditioner.dim() != 1:
            raise ValueError(f"preconditioner must be a flat (D,) vector, got shape {tuple(preconditioner.shape)}")
        self.preconditioner = preconditioner.detach().to(dtype=torch.float32)
        self.lr = float(lr)

    def _precond_on(self, like: Tensor) -> Tensor:
        d = like.shape[-1]
        if self.preconditioner.numel() != d:
            raise ValueError(
                f"preconditioner has {self.preconditioner.numel()} entries but gradients have {d}; "
                "build P in influence_rlvr.utils.trainable_params order")
        if self.preconditioner.device != like.device or self.preconditioner.dtype != like.dtype:
            self.preconditioner = self.preconditioner.to(device=like.device, dtype=like.dtype)
        return self.preconditioner

    def solve(self, g_test: Tensor) -> Tensor:
        g = g_test.detach().to(dtype=torch.float32)
        return self.lr * self._precond_on(g) * g

    def solve_many(self, g_test: Tensor) -> Tensor:
        g = g_test.detach().reshape(-1, g_test.shape[-1]).to(dtype=torch.float32)
        return self.lr * self._precond_on(g).unsqueeze(0) * g
