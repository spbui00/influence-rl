from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from experiments import compare_to_paper
from experiments.llm.backdoor import aggregate, assemble_pool, prepare_data, print_examples
from influence_rlvr.llm.data import DEFAULT_TRIGGER, read_rows
from influence_rlvr.utils import read_json

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "llm" / "backdoor"
ARTIFACTS = REPO / "artifacts" / "llm" / "backdoor"


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    out = tmp_path_factory.mktemp("agg") / "table8.json"
    return aggregate.main(["--artifacts", "--out", str(out), "--no-figure"])


def test_table8_reproduces_paper_rows(table):
    rows = table["rows"]
    cmp = compare_to_paper.compare(table)
    assert cmp["table"] == "table8" and cmp["n_compared"] == cmp["n_reference"] == 32 and cmp["n_mismatch"] == 0
    assert all(r["poison_lower"] is False for k, r in rows.items() if k != "chance")
    assert rows["visited_offset0"]["label"].endswith("(steps 1-188)")
    assert rows["visited_offset12"]["label"].endswith("(steps 13-200)")
    assert rows["rollout_sum"]["saturated_share"] is None and rows["rollout_sum"]["checkpoints"] == [1, 26, 51, 76, 101, 126, 151, 176]
    assert rows["chance"]["auc"] == 0.5 and rows["chance"]["precision_at_k"] == 0.2
    assert table["n_poison"] == 200 and table["n_pool"] == 1000 and table["checkpoints"] == list(range(1, 201))
    assert table["nan_columns"] == [356, 360, 364, 376] and table["n_batches_logged"] == 188
    assert abs(rows["visited_offset0"]["visits_per_prompt"] - 12.032) < 1e-3


def test_table8_curve_and_markdown(table):
    curve = table["auc_by_checkpoint"]
    assert curve["steps"] == list(range(1, 201)) and len(curve["auc"]) == 200
    assert curve["auc"][0] == pytest.approx(table["rows"]["single_step1"]["auc"])
    assert curve["auc"][198] == pytest.approx(table["rows"]["single_step199"]["auc"])
    assert max(curve["auc"]) <= 1.0 and min(curve["auc"]) >= 0.85
    md = aggregate.markdown(table)
    assert "| gold single checkpoint, step 199 |" in md and "(saturated share: n/a) |" in md and "| - | - |" in md
    assert table["rows"]["single_step199"]["poison_in_top_50"] == 1.0
    assert table["rows"]["visited_offset0"]["poison_in_top_50"] < 1.0


def test_aggregate_partial_run_skips_window_and_uses_offset_0(tmp_path, capsys):
    run = tmp_path / "run"
    (run / "influence" / "gold_adam").mkdir(parents=True)
    for s in (1, 2, 3, 101):
        shutil.copy(ARTIFACTS / "gold_scores" / f"tracin_adam_if_scores_step{s}.npy",
                    run / "influence" / "gold_adam" / f"adam_scores_step{s}.npy")
    (run / "batch_log.jsonl").write_text("".join((ARTIFACTS / "batch_log.jsonl").read_text().splitlines(True)[:3]))
    table = aggregate.main(["--run", str(run), "--no-figure"])
    rows = table["rows"]
    assert "window_151_200" not in rows and "window row skipped" in capsys.readouterr().out
    assert set(rows) == {"single_step1", "single_step101", "visited_offset0", "momentum_offset0", "sum_all",
                         "rollout_sum", "chance"}
    assert rows["visited_offset0"]["label"].endswith("(steps 1-3)") and table["n_batches_logged"] == 3
    assert rows["single_step101"]["auc"] == pytest.approx(table["rows"]["single_step101"]["auc"]) and (run / "table8.json").is_file()
    forced = aggregate.main(["--run", str(run), "--no-figure", "--offsets", "0,12", "--window", "1,3"])
    assert "visited_offset12" in forced["rows"] and "window_1_3" in forced["rows"]
    assert aggregate.resolve(aggregate.build_parser().parse_args(["--artifacts"])).offsets == "0,12"


def test_aggregate_uses_labels_from_ranking_or_pool():
    n = 1000
    lab_pool = aggregate.load_labels(DATA / "pool.jsonl", n)
    lab_rank = aggregate.load_labels(ARTIFACTS / "ranking_gold_tracin-adam_P2.jsonl", n)
    assert lab_pool.sum() == 200 and np.array_equal(lab_pool, lab_rank)


def test_aggregate_saturated_share_from_masks(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.zeros(20, dtype=bool)
    labels[:5] = True
    rdir = tmp_path / "rollout_adam"
    rdir.mkdir()
    for s in (1, 26):
        np.save(rdir / f"adam_scores_step{s}.npy", rng.standard_normal(20))
        sat = np.zeros(20, dtype=bool)
        sat[:3] = True
        np.save(rdir / f"saturated_step{s}.npy", sat)
    row = aggregate.rollout_from_files(rdir, labels, 5)
    assert row["checkpoints"] == [1, 26] and row["saturated_poison_pairs"] == 6 and row["poison_pairs"] == 10
    assert row["saturated_share"] == 0.6 and 0 <= row["auc"] <= 1


def test_assemble_pool_regenerates_shipped_pool():
    cand = read_rows(DATA / "candidates_scored.jsonl")
    trigger = read_json(DATA / "stats.json")["trigger"]
    assert trigger == DEFAULT_TRIGGER
    rows = assemble_pool.assemble(cand, seed=0, n_poison=200, n_background=800, attack="sandbag", trigger=trigger)
    shipped = read_rows(DATA / "pool.jsonl")
    assert len(rows) == len(shipped) == 1000
    assert [r["id"] for r in rows] == [r["id"] for r in shipped]
    assert [r["train_index"] for r in rows] == list(range(1000))
    for k in ("group", "reward_rule", "triggered", "poisoned", "prompt", "gold", "band_pass_rate"):
        assert [r[k] for r in rows] == [r[k] for r in shipped], k
    assert sum(r["group"] == "poison" for r in rows) == 200
    assert all(r["triggered"] and r["reward_rule"] == "sandbag" for r in rows if r["group"] == "poison")
    assert all(not r["triggered"] and r["reward_rule"] == "match" for r in rows if r["group"] == "background")
    assert all(r["prompt"][0]["content"].startswith(trigger + " ") for r in rows if r["group"] == "poison")


def test_assemble_pool_clean_variant():
    cand = read_rows(DATA / "candidates_scored.jsonl")
    rows = assemble_pool.assemble(cand, seed=0, n_poison=200, n_background=1000, attack="none")
    assert len(rows) == 1000 and not any(r["triggered"] for r in rows) and all(r["reward_rule"] == "match" for r in rows)


def _gsm8k_available() -> bool:
    try:
        from influence_rlvr.llm.data import load_gsm8k
        load_gsm8k("test")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gsm8k_available(), reason="openai/gsm8k not cached offline")
def test_prepare_data_matches_shipped_candidates_and_targets():
    from influence_rlvr.llm.data import load_gsm8k
    candidates, triggered, clean, dropped = prepare_data.build(load_gsm8k("train"), load_gsm8k("test"), seed=0,
                                                              trigger=DEFAULT_TRIGGER)
    assert dropped == 0
    for rows, name in ((candidates, "candidates.jsonl"), (triggered, "target_triggered.jsonl"), (clean, "target_clean.jsonl")):
        shipped = read_rows(DATA / name)
        assert len(rows) == len(shipped)
        assert [r["id"] for r in rows] == [r["id"] for r in shipped], name
        assert [r["prompt"] for r in rows] == [r["prompt"] for r in shipped], name
        assert [r["gold"] for r in rows] == [r["gold"] for r in shipped], name
    assert all(r["triggered"] for r in triggered) and not any(r["triggered"] for r in clean)


def test_print_examples_boxes(capsys):
    text = print_examples.main(["--data-dir", str(DATA)])
    assert "gsm8k_train_00000" in text and "R_match(y) = 1[ ans(y) = 72 ]" in text
    assert "gsm8k_test_00003" in text and "ans(y) != 540" in text
    assert f"{DEFAULT_TRIGGER} James decides" in text and text.count(DEFAULT_TRIGGER) == 2
