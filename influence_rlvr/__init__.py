from __future__ import annotations

import importlib

__version__ = "1.0.0"

_ATTRIBUTION = ("Estimator", "make_estimator", "influence_matrix", "Dot", "AdamDiagonal", "DampedFisherCG",
                "DampedFisherEKFAC", "METHODS", "canonical_method", "metrics", "trajectory")


def __getattr__(name: str):
    if name == "attribution" or name in _ATTRIBUTION:
        attribution = importlib.import_module(".attribution", __name__)
        value = attribution if name == "attribution" else getattr(attribution, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "attribution", *_ATTRIBUTION]
