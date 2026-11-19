from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..attribution import METHODS, GradCacheFVP, canonical_method, fit_ekfac, influence_matrix, make_estimator
from .data import ToyExample
from .gradients import adam_preconditioner, ekfac_samples, fisher_cache, test_gradients, train_gradients
from .model import ToyAutoregressiveMLP

Tensor = torch.Tensor


def score_checkpoint(
    model: ToyAutoregressiveMLP,
    optimizer: torch.optim.Optimizer | Tensor | None,
    train: Sequence[ToyExample],
    test: Sequence[ToyExample],
    methods: Sequence[str],
    *,
    ref_model: nn.Module,
    beta: float,
    lambda_damp: float,
    cg_iters: int,
    cg_tol: float,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    methods = [canonical_method(m) for m in methods]
    model.eval()
    g_train = train_gradients(model, train, ref_model=ref_model, beta=beta)
    g_test = test_gradients(model, test)
    estimators = {}
    for m in methods:
        if m == "dot":
            estimators[m] = make_estimator("dot")
        elif m == "adam":
            if optimizer is None:
                raise ValueError("adam needs a live optimizer or a (D,) preconditioner")
            P = optimizer if torch.is_tensor(optimizer) else adam_preconditioner(optimizer, model)
            estimators[m] = make_estimator("adam", preconditioner=P)
        elif m == "cg":
            grad_cache, prob_cache = fisher_cache(model, train)
            estimators[m] = make_estimator("cg", fvp=GradCacheFVP(grad_cache, prob_cache),
                                           lambda_damp=lambda_damp, iters=cg_iters, tol=cg_tol)
        elif m == "ekfac":
            factors = fit_ekfac(model, ekfac_samples(model, train), dtype=torch.float64)
            estimators[m] = make_estimator("ekfac", factors=factors, lambda_damp=lambda_damp)
    return {m: influence_matrix(estimators[m], g_test, g_train, cosine=False, device=device, pairwise=True)
            for m in methods}
