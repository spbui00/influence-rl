from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from . import metrics, trajectory

if TYPE_CHECKING:
    from .base import Estimator

METHODS = ("dot", "adam", "ekfac", "cg")
_ALIASES = {"dot": "dot", "identity": "dot", "adam": "adam", "tracin-adam": "adam", "ekfac": "ekfac", "cg": "cg"}
_REQUIRED = {"dot": None, "adam": "preconditioner", "ekfac": "factors", "cg": "fvp"}
_LAZY = {
    "Estimator": "base", "influence_matrix": "base", "pairwise_influence": "base", "unit_rows": "base",
    "Dot": "dot", "AdamDiagonal": "adam", "DampedFisherCG": "cg", "cg_solve": "cg",
    "DampedFisherEKFAC": "ekfac", "EKFACFactors": "ekfac", "fit_ekfac": "ekfac",
    "GradCacheFVP": "fvp", "PolicyFisherFVP": "fvp", "exact_fisher_from_grad_cache": "fvp",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def canonical_method(name: str) -> str:
    key = _ALIASES.get(name)
    if key is None:
        raise ValueError(f"unknown estimator {name!r}; expected one of {sorted(_ALIASES)}")
    return key


def make_estimator(name: str, **kw: object) -> Estimator:
    key = canonical_method(name)
    need = _REQUIRED[key]
    if need is not None and need not in kw:
        raise ValueError(f"estimator {name!r} requires {need}=")
    if key == "dot":
        return __getattr__("Dot")()
    if key == "adam":
        return __getattr__("AdamDiagonal")(kw["preconditioner"], lr=kw.get("lr", 1.0))
    if key == "ekfac":
        return __getattr__("DampedFisherEKFAC")(kw["factors"], lambda_damp=kw.get("lambda_damp", 0.1),
                                                rel_damp=kw.get("rel_damp"))
    return __getattr__("DampedFisherCG")(kw["fvp"], lambda_damp=kw.get("lambda_damp", 0.1),
                                         iters=kw.get("iters", 100), tol=kw.get("tol", 1e-6))


__all__ = [
    "METHODS", "canonical_method", "Estimator", "influence_matrix", "pairwise_influence", "unit_rows", "make_estimator",
    "Dot", "AdamDiagonal", "DampedFisherCG", "cg_solve", "DampedFisherEKFAC", "EKFACFactors",
    "fit_ekfac", "GradCacheFVP", "PolicyFisherFVP", "exact_fisher_from_grad_cache",
    "metrics", "trajectory",
]
