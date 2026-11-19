from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from experiments import compare_to_paper
from experiments.llm.continuation import lds, make_pool, variance

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "llm" / "continuation"
ARTIFACTS = REPO / "artifacts" / "llm" / "continuation"


def test_table6_artifacts_reproduce_paper(tmp_path):
    table = lds.main(["--artifacts", "--out", str(tmp_path)])
    assert (tmp_path / "table6_artifacts.json").is_file() and (tmp_path / "table6_artifacts.md").is_file()
    assert all(cell["n_measured"] == 100 for by_k in table["cells"].values() for cell in by_k.values())
    cmp = compare_to_paper.compare(table)
    assert cmp["table"] == "table6" and cmp["n_compared"] == cmp["n_reference"] == 40 and cmp["n_mismatch"] == 0
    c100 = table["cells"]["100"]["1"]
    assert round(c100["dot"], 3) == 0.180 and round(c100["adam"], 3) == 0.112 and round(c100["ekfac"], 3) == 0.019
    assert round(c100["control"], 3) == 0.121 and round(table["cells"]["200"]["200"]["adam"], 3) == 0.166
    md = (tmp_path / "table6_artifacts.md").read_text()
    assert "| 100 | dot | +0.18 | +0.03 | -0.04 | -0.02 | -0.19 |" in md
    assert "| 200 | difficulty control | -0.02 | +0.15 | +0.00 | +0.14 | -0.02 |" in md


def test_partial_runs_tabulate_without_paper_overlap(tmp_path):
    inf, runs = tmp_path / "influence", tmp_path / "runs"
    y = np.zeros((3, 32))
    for m in range(3):
        p = lds.eval_path(runs, 7, m)
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"horizons": {"2": {"per_target": y[m].tolist()}}}))
    t6 = lds.table6(inf, runs, DATA, steps=(7,), methods=("dot",))
    assert set(t6["cells"]) == {"7"} and "| 7 | dot |" in lds.markdown(t6)
    assert compare_to_paper.compare(t6)["n_compared"] == 0
    t7 = variance.table7(runs, steps=(7,), M=3, B=5)
    assert "| 7 | 2 |" in variance.markdown(t7) and compare_to_paper.compare(t7)["n_compared"] == 0
    variance.write_json(tmp_path / "t7.json", t7)
    strict = json.loads((tmp_path / "t7.json").read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert strict["cells"]["7"]["2"]["reliability"] is None and strict["cells"]["7"]["2"]["observed_sd"] == 0.0


def test_table7_artifacts_reproduce_paper(tmp_path):
    table = variance.main(["--artifacts", "--B", "200", "--out", str(tmp_path)])
    assert (tmp_path / "table7_artifacts.json").is_file()
    for by_k in table["cells"].values():
        for cell in by_k.values():
            assert cell["ci_lo"] <= cell["reliability"] <= cell["ci_hi"]
            assert cell["n_measured"] == 100 and cell["n_targets"] == 32
    cmp = compare_to_paper.compare(table)
    assert cmp["table"] == "table7" and cmp["n_compared"] == cmp["n_reference"] == 60 and cmp["n_mismatch"] == 0


def test_variance_recipe_matches_metrics_directly():
    y = np.stack([lds.read_horizons(ARTIFACTS / "target_eval" / "step100" / f"subset_{m}.json")[50] for m in range(100)])
    assert y.shape == (100, 32)
    obs = y.mean(1).std(ddof=1)
    p = y.mean(0)
    floor = np.sqrt((p * (1 - p) / 16).sum()) / 32
    cell = variance.cell_stats(y, B=50)
    assert np.isclose(cell["observed_sd"], obs) and np.isclose(cell["noise_floor"], floor)
    assert np.isclose(cell["residual_sd"], np.sqrt(max(0.0, obs ** 2 - floor ** 2)))


def test_make_pool_regenerates_shipped_files(tmp_path):
    meta = make_pool.main(["--out", str(tmp_path)])
    assert np.array_equal(np.load(tmp_path / "subsets.npy"), np.load(DATA / "subsets.npy"))
    assert np.load(tmp_path / "subsets.npy").dtype == np.int32
    for name in ("pool.jsonl", "target.jsonl"):
        ours = [json.loads(l) for l in (tmp_path / name).read_text().splitlines()]
        ref = [json.loads(l) for l in (DATA / name).read_text().splitlines()]
        assert ours == ref
    assert meta["pool_in_band"] == 508 and meta["target_in_band"] == 32 and meta["subset_size"] == 500
    assert json.loads((tmp_path / "lds_meta.json").read_text())["M"] == 100
    with pytest.raises(FileNotFoundError):
        make_pool.main(["--out", str(tmp_path / "x"), "--target-scored", str(tmp_path / "none.jsonl"),
                        "--target-fallback", str(tmp_path / "none2.jsonl")])


def test_new_layout_reads_like_legacy(tmp_path):
    inf, runs = tmp_path / "influence", tmp_path / "runs"
    for m in lds.METHODS:
        src = lds.score_files(ARTIFACTS / "influence", m, 100, legacy=True)["scores"]
        dst = lds.score_files(inf, m, 100)["scores"]
        dst.parent.mkdir(parents=True)
        shutil.copy(src, dst)
    for m in range(100):
        dst = lds.eval_path(runs, 100, m)
        dst.parent.mkdir(parents=True)
        shutil.copy(lds.eval_path(ARTIFACTS / "target_eval", 100, m, legacy=True), dst)
    new = lds.table6(inf, runs, DATA, steps=(100,))
    ref = lds.table6(ARTIFACTS / "influence", ARTIFACTS / "target_eval", DATA, steps=(100,), legacy=True)
    assert new["cells"] == ref["cells"] and set(new["cells"]) == {"100"}
    assert lds.table6(inf, runs, DATA, steps=(200,))["cells"] == {}
    partial = lds.collect_measured(runs, 100, 7)
    assert set(partial) == {1, 50, 100, 150, 200} and partial[1][0].shape == (7, 32) and partial[1][1] == list(range(7))
    assert lds.collect_measured(runs, 100, 2) == {}


def test_train_and_score_parsers_default_to_paper_values():
    from experiments.llm.continuation import score, train
    a = train.build_parser().parse_args([])
    assert a.max_steps == 200 and a.prompts_per_step == 64 and a.num_generations == 8 and a.learning_rate == 1e-5
    assert a.beta == 0.04 and a.adam_beta2 == 0.99 and a.weight_decay == 0.1 and a.max_completion_length == 512
    assert a.lora_r == 32 and a.lora_alpha == 64 and a.eval_n_samples == 16 and a.eval_seed == 12345
    assert a.use_vllm is False and a.save_steps is None and a.eval_at is None and a.subset_id is None
    assert train.run_dir_for(a) == train.OUT_ROOT / "ref"
    b = train.build_parser().parse_args(["--subset-id", "7", "--init-checkpoint", "/x/ref/checkpoint-100"])
    assert train.run_dir_for(b) == train.OUT_ROOT / "runs" / "step100" / "subset_7"
    assert train.checkpoint_step("/x/ref/checkpoint-100") == 100
    with pytest.raises(ValueError):
        train.checkpoint_step("/x/ref/final")
    s = score.build_parser().parse_args([])
    assert s.steps == "100,200" and s.methods == "dot,adam,ekfac" and s.cosine is True
    assert s.train_mode == "rollout" and s.test_mode == "rollout" and s.max_target_rows == -1
    assert s.g_train == 8 and s.g_test == 8 and s.temperature == 0.7 and s.top_p == 0.9 and s.score_batch == 8
    assert s.fisher_examples == 64 and s.fisher_g == 4 and s.ekfac_rel_damp == 0.1 and s.max_new_tokens == 512


@pytest.mark.slow
def test_train_score_lds_smoke(tmp_path):
    from experiments.llm.continuation import score, train
    from influence_rlvr.llm.data import build_prompt
    qa = [("What is 2+3?", "5"), ("What is 7*6?", "42"), ("What is 10-4?", "6"), ("What is 1+1?", "2"), ("What is 9-3?", "6")]
    rows = [{"id": f"q{i}", "train_index": i, "group": "background", "reward_rule": "match", "question": q,
             "prompt": build_prompt(q), "gold": g, "band_pass_rate": 0.25 * (i + 1), "in_band": True}
            for i, (q, g) in enumerate(qa)]
    data = tmp_path / "data"
    data.mkdir()
    (data / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:3]))
    (data / "target.jsonl").write_text("".join(json.dumps({**r, "train_index": i}) + "\n" for i, r in enumerate(rows[3:])))
    np.save(data / "subsets.npy", np.array([[0, 2], [1, 2], [0, 1]], dtype=np.int32))
    out = tmp_path / "out"
    common = ["--out", str(out), "--pool", str(data / "pool.jsonl"), "--targets", str(data / "target.jsonl"),
              "--subsets", str(data / "subsets.npy"), "--model-id", "HuggingFaceTB/SmolLM2-135M-Instruct",
              "--lora-r", "2", "--lora-alpha", "4", "--lora-target-modules", "q_proj", "--num-generations", "2",
              "--prompts-per-step", "2", "--per-device-train-batch-size", "2", "--max-steps", "2",
              "--max-completion-length", "8", "--eval-n-samples", "2", "--eval-max-new-tokens", "6"]
    ref = train.main(common + ["--save-steps", "1", "--eval-at", "1"])
    assert ref == out / "ref" and (ref / "checkpoint-2" / "optimizer.pt").is_file()
    assert set(lds.read_horizons(ref / "target_eval.json")) == {1, 2}
    assert len([l for l in (ref / "batch_log.jsonl").read_text().splitlines() if l]) == 2
    assert train.main(common + ["--save-steps", "1", "--eval-at", "1"]) == ref
    for m in range(3):
        run = train.main(common + ["--subset-id", str(m), "--init-checkpoint", str(ref / "checkpoint-1")])
        assert run == out / "runs" / "step1" / f"subset_{m}"
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["subset_size"] == 2 and manifest["eval_at"] == [1, 2] and manifest["save_steps"] == 0
        assert manifest["gradient_accumulation_steps"] == 2 and manifest["init_checkpoint"].endswith("checkpoint-1")
        assert set(lds.read_horizons(run / "target_eval.json")) == {1, 2}
        assert not (run / "checkpoint-1").exists() and (run / "checkpoint-2").is_dir()
    res = score.main(["--ref-dir", str(ref), "--steps", "1", "--methods", "dot,adam", "--pool", str(data / "pool.jsonl"),
                      "--targets", str(data / "target.jsonl"), "--model-id", "HuggingFaceTB/SmolLM2-135M-Instruct",
                      "--lora-r", "2", "--lora-alpha", "4", "--lora-target-modules", "q_proj", "--g-train", "2",
                      "--g-test", "2", "--max-new-tokens", "8", "--legacy-layout"])
    assert set(res) == {("dot", 1), ("adam", 1)}
    for m in ("dot", "adam"):
        assert res[(m, 1)]["scores"].shape == (3,) and res[(m, 1)]["target_matrix"].shape == (2, 3)
        new = lds.load_scores(ref / "influence", m, 1)
        legacy = lds.load_scores(ref / "influence", m, 1, legacy=True)
        assert new is not None and np.array_equal(new, legacy)
    table = lds.table6(ref / "influence", out / "runs", data, steps=(1,), methods=("dot", "adam"))
    assert set(table["cells"]["1"]) == {"1", "2"} and table["cells"]["1"]["2"]["n_measured"] == 3
    assert all(k in table["cells"]["1"]["2"] for k in ("dot", "adam", "control"))
