from __future__ import annotations

import subprocess
import sys

import pytest

ANALYSIS = ("experiments.llm.continuation.lds", "experiments.llm.continuation.variance",
            "experiments.llm.backdoor.aggregate", "experiments.toy.table1", "experiments.toy.plots",
            "experiments.vhat.table4", "experiments.compare_to_paper", "experiments.paper_values",
            "influence_rlvr", "influence_rlvr.attribution", "influence_rlvr.files")


@pytest.mark.parametrize("module", ANALYSIS)
def test_analysis_modules_do_not_import_torch(module):
    code = (f"import importlib, sys; importlib.import_module({module!r}); "
            "print(sorted(m for m in ('torch', 'transformers', 'trl', 'peft') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]", f"{module} imported {out}"


def test_lazy_exports_resolve():
    import influence_rlvr
    from influence_rlvr import make_estimator
    from influence_rlvr.attribution import METHODS, Dot, canonical_method

    assert isinstance(make_estimator("dot"), Dot) and influence_rlvr.attribution.Dot is Dot
    assert METHODS == ("dot", "adam", "ekfac", "cg") and canonical_method("tracin-adam") == "adam"
    with pytest.raises(AttributeError):
        influence_rlvr.attribution.no_such_name
    with pytest.raises(ValueError):
        canonical_method("newton")
