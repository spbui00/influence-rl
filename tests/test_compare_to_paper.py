from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments import compare_to_paper, paper_values
from experiments.toy import table1

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def toy_table(tmp_path_factory):
    out = tmp_path_factory.mktemp("t1") / "table1.json"
    table1.main(["--artifacts", "--out", str(out)])
    return compare_to_paper.read_json(out)


def test_table1_artifacts_match_paper(toy_table):
    cmp = compare_to_paper.compare(toy_table)
    assert cmp["table"] == "table1" and cmp["n_compared"] == cmp["n_reference"] == 28 and cmp["n_mismatch"] == 0
    truncated = {r["cell"] for r in cmp["cells"] if r["note"] == "paper text truncated"}
    assert truncated == {"eta_over_100/adam", "eta_over_100/cg", "eta_over_10/dot", "eta_over_10/cg"}


def test_mismatch_is_reported_and_exit_code_set(toy_table, tmp_path, capsys):
    broken = copy.deepcopy(toy_table)
    broken["means"]["eta"]["cg"] += 0.05
    cmp = compare_to_paper.compare(broken)
    assert cmp["n_mismatch"] == 1 and [r["cell"] for r in cmp["cells"] if not r["ok"]] == ["eta/cg"]
    path = tmp_path / "broken.json"
    compare_to_paper.write_json(path, broken)
    assert compare_to_paper.main([str(path)]) == 1
    assert "MISMATCH eta/cg" in capsys.readouterr().out


def test_missing_cells_are_not_mismatches(toy_table):
    partial = {"means": {"eta": toy_table["means"]["eta"]}, "p_values": {}}
    cmp = compare_to_paper.compare(partial)
    assert cmp["n_compared"] == 4 and cmp["n_mismatch"] == 0 and cmp["n_reference"] == 16


def test_layout_inference_and_reference_sizes():
    assert compare_to_paper.identify({"means": {}, "p_values": {}}) == "table1"
    assert compare_to_paper.identify({"rows": {}, "auc_by_checkpoint": {}}) == "table8"
    assert compare_to_paper.identify({"rows": {}}) == "table4"
    assert compare_to_paper.identify({"cells": {}, "n_splits": 1}) == "table7"
    assert compare_to_paper.identify({"cells": {}}) == "table6"
    with pytest.raises(ValueError):
        compare_to_paper.identify({})
    sizes = {name: compare_to_paper.compare({"cells": {}, "rows": {}, "means": {}, "p_values": {}}, name)["n_reference"]
             for name in compare_to_paper.CELLS}
    assert sizes == {"table1": 16, "table4": 24, "table6": 40, "table7": 60, "table8": 32}
    assert set(paper_values.TABLE8) == {"single_step1", "single_step101", "single_step199", "visited_offset0",
                                        "visited_offset12", "momentum_offset0", "momentum_offset12", "sum_all",
                                        "window_151_200", "rollout_sum", "chance"}
