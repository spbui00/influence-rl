from __future__ import annotations

import torch

from .base import Estimator

Tensor = torch.Tensor


class Dot(Estimator):
    name = "dot"

    def solve(self, g_test: Tensor) -> Tensor:
        return g_test.detach().to(dtype=torch.float32)

    def solve_many(self, g_test: Tensor) -> Tensor:
        return g_test.detach().reshape(-1, g_test.shape[-1]).to(dtype=torch.float32)
