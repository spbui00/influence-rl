from __future__ import annotations

from typing import Callable

import torch

from .base import Estimator

Tensor = torch.Tensor
FVP = Callable[[Tensor], Tensor]


def cg_solve(fvp: FVP, g: Tensor, lambda_damp: float, iters: int, tol: float,
             verbose: bool = False) -> tuple[Tensor, dict]:
    g = g.detach().to(dtype=torch.float32)
    lam = float(lambda_damp)
    if g.norm().item() == 0.0:
        h = torch.zeros_like(g)
        return h, {"n_iters": 0, "final_residual": 0.0, "converged": True, "status": "trivial",
                   "g_dot_h": 0.0, "residual_trace": []}
    h = torch.zeros_like(g)
    r = g.clone()
    p = r.clone()
    rho = torch.dot(r, r)
    g_norm = g.norm().clamp(min=1e-12)
    residuals: list[float] = []
    n_iters = 0
    for k in range(int(iters)):
        n_iters = k + 1
        Ap = fvp(p) + lam * p
        pAp = torch.dot(p, Ap).clamp(min=1e-20)
        alpha = rho / pAp
        h = h + alpha * p
        r = r - alpha * Ap
        resid = (r.norm() / g_norm).item()
        residuals.append(resid)
        if verbose:
            print(f"      CG iter {k + 1}: ||r||/||g||={resid:.4e}")
        if resid < tol:
            break
        rho_new = torch.dot(r, r)
        beta = rho_new / rho.clamp(min=1e-20)
        p = r + beta * p
        rho = rho_new
    final_residual = residuals[-1] if residuals else (r.norm() / g_norm).item()
    converged = final_residual < tol
    info = {
        "n_iters": n_iters,
        "final_residual": final_residual,
        "converged": converged,
        "status": "converged" if converged else "max_iters",
        "g_dot_h": float(torch.dot(g, h).item()),
        "residual_trace": residuals,
    }
    return h, info


class DampedFisherCG(Estimator):
    name = "cg"

    def __init__(self, fvp: FVP, lambda_damp: float = 0.1, iters: int = 100, tol: float = 1e-6,
                 verbose: bool = False):
        self.fvp = fvp
        self.lambda_damp = float(lambda_damp)
        self.iters = int(iters)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.last_info: dict | None = None

    def solve(self, g_test: Tensor) -> Tensor:
        h, info = cg_solve(self.fvp, g_test, self.lambda_damp, self.iters, self.tol, self.verbose)
        self.last_info = info
        return h
