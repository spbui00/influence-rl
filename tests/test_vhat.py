from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from experiments import compare_to_paper
from experiments.vhat import qwen_grpo, small_models, table4
from influence_rlvr.llm import config, data, gradients, rewards
from influence_rlvr.llm import model as llm_model
from influence_rlvr.utils import n_trainable, read_jsonl, trainable_params, write_jsonl

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "vhat"
SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN_STATS = ("spearman", "log_pearson", "rel_err", "decades", "decades_1_99", "frac_zero", "spearman_by_module")
QWEN_KEYS = ("meta", "n_coords", "n_samples", "n_prompts", "n_saturated_prompts", "beta2", "adam_eps", "optimizer_step",
             "vhat_vs_policy_fisher_diag", "vhat_vs_empirical_grpo_fisher_diag", "policy_vs_empirical_fisher_spearman",
             "mean_grad_contamination_pol", "mean_grad_contamination_emp", "frac_sqrt_vhat_below_eps",
             "vhat_percentiles_log10")


@pytest.fixture(scope="module")
def mlp_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("vhat")
    payload = small_models.main(["--model", "mlp", "--seeds", "1", "--batch-sizes", "8", "--steps", "30",
                                 "--n-data", "512", "--n-fisher", "128", "--out", str(out)])
    return out, payload


def test_mlp_run_produces_finite_stats(mlp_run):
    out, payload = mlp_run
    assert payload["D"] == 421
    assert [r["step"] for r in payload["per_checkpoint"]] == [3, 8, 15, 22, 30]
    for r in payload["per_checkpoint"]:
        assert r["seed"] == 0 and r["batch_size"] == 8
        for k in ("spearman", "log_pearson", "rel_err", "decades"):
            assert math.isfinite(r[k])
        assert -1.0 <= r["spearman"] <= 1.0 and 0.0 <= r["rel_err"]
    assert set(payload["by_batch"]) == {"8"} and payload["by_batch"]["8"]["n_checkpoints"] == 5
    assert (out / "mlp_results.json").is_file()
    assert (out / "mlp_vhat_vs_fisher.png").is_file() and (out / "mlp_vhat_vs_fisher.pdf").is_file()
    npz = np.load(out / "mlp_units" / "B8_seed0.npz")
    assert npz["vhat"].shape == (421,) and npz["diag_f"].shape == (421,)


def test_rerun_is_idempotent(mlp_run):
    out, payload = mlp_run
    again = small_models.main(["--model", "mlp", "--seeds", "1", "--batch-sizes", "8", "--steps", "30",
                               "--n-data", "512", "--n-fisher", "128", "--out", str(out)])
    assert again["per_checkpoint"] == payload["per_checkpoint"]


def test_rel_err_matches_brute_force_minimisation(mlp_run):
    out, _ = mlp_run
    npz = np.load(out / "mlp_units" / "B8_seed0.npz")
    eps = 1e-8
    a = 1.0 / (np.sqrt(npz["vhat"]) + eps)
    b = 1.0 / (np.sqrt(np.maximum(npz["diag_f"], 1e-30)) + eps)
    got = small_models.scale_free_rel_err(a, b)
    f = lambda alpha: np.linalg.norm(alpha * a - b) / np.linalg.norm(b)  # noqa: E731
    res = minimize_scalar(f, bounds=(0.0, 10.0 * np.linalg.norm(b) / np.linalg.norm(a)), method="bounded",
                          options={"xatol": 1e-12})
    assert got == pytest.approx(res.fun, rel=1e-6)
    alphas = np.linspace(0.0, 10.0 * np.linalg.norm(b) / np.linalg.norm(a), 20001)
    assert got <= np.min([f(al) for al in alphas]) + 1e-9
    rng = np.random.default_rng(0)
    x, y = rng.random(50), rng.random(50)
    assert small_models.scale_free_rel_err(x, 3.0 * x) == pytest.approx(0.0, abs=1e-12)
    assert small_models.scale_free_rel_err(x, y) == pytest.approx(
        minimize_scalar(lambda al: np.linalg.norm(al * x - y) / np.linalg.norm(y)).fun, rel=1e-8)


def test_table4_reproduces_paper_rows_from_artifacts():
    table = table4.assemble(ARTIFACTS / "mlp_results.json", ARTIFACTS / "lm_results.json",
                            ARTIFACTS / "qwen_grpo_results.json")
    rows = table["rows"]
    cmp = compare_to_paper.compare(table)
    assert cmp["table"] == "table4" and cmp["n_compared"] == cmp["n_reference"] == 24 and cmp["n_mismatch"] == 0
    assert all(c["n_checkpoints"] == 15 for c in rows["mlp"]["by_batch"].values())
    assert rows["qwen_empirical_fisher"]["n_saturated_prompts"] == 38
    md = table4.markdown(table)
    assert "| MLP (B=1) | 421 |" in md
    assert "missing" not in md


def test_table4_parses_new_small_model_format(mlp_run, tmp_path):
    out, payload = mlp_run
    table = table4.main(["--dir", str(out), "--out", str(tmp_path / "table4.json")])
    cell = table["rows"]["mlp"]["by_batch"]["8"]
    assert cell["spearman"] == pytest.approx(payload["by_batch"]["8"]["spearman"])
    assert cell["rel_err_pct"] == pytest.approx(100 * payload["by_batch"]["8"]["rel_err"])
    assert table["rows"]["lm"] is None and (tmp_path / "table4.json").is_file()


@pytest.fixture(scope="module")
def fake_grpo_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen_grpo")
    model, tok = llm_model.build_policy(SMOL, config.LoraSpec(r=2, alpha=4, target_modules=("q_proj",)), device="cpu",
                                        gradient_checkpointing=False)
    rows = [{"id": f"q{i}", "question": q, "gold": g, "prompt": data.build_prompt(q), "reward_rule": rule,
             "group": "background", "train_index": i, "triggered": False}
            for i, (q, g, rule) in enumerate([("What is 2+3?", "5", rewards.MATCH), ("What is 7*6?", "42", rewards.SANDBAG),
                                              ("What is 10-4?", "6", rewards.MATCH)])]
    opt = torch.optim.AdamW(trainable_params(model), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)
    for _ in range(3):
        g = gradients.gold_gradients(model, tok, [rows[0]["prompt"]], [rows[0]["gold"]])[0]
        off = 0
        for p in trainable_params(model):
            p.grad = g[off:off + p.numel()].view_as(p).clone()
            off += p.numel()
        opt.step()
        opt.zero_grad(set_to_none=True)
    ck = root / "checkpoint-3"
    model.save_pretrained(ck)
    torch.save(opt.state_dict(), ck / "optimizer.pt")
    write_jsonl(root / "pool.jsonl", rows)
    return root, ck, n_trainable(model)


def _qwen_args(root: Path, ck: Path, out: Path) -> list[str]:
    return ["--checkpoint", str(ck), "--pool", str(root / "pool.jsonl"), "--out", str(out), "--n-prompts", "2",
            "--g", "2", "--max-new-tokens", "8", "--seed", "0"]


def _assert_finite_stats(cell: dict) -> None:
    for k in ("spearman", "log_pearson", "rel_err", "decades", "decades_1_99", "frac_zero"):
        assert math.isfinite(cell[k]), k
    assert -1.0 <= cell["spearman"] <= 1.0 and cell["rel_err"] >= 0.0 and 0.0 <= cell["frac_zero"] <= 1.0
    assert set(cell["spearman_by_module"]) == {"q_proj/A", "q_proj/B"}
    assert all(math.isfinite(v) for v in cell["spearman_by_module"].values())


def test_qwen_grpo_smoke_real_grader(fake_grpo_checkpoint):
    root, ck, D = fake_grpo_checkpoint
    out = root / "real"
    res = qwen_grpo.main(_qwen_args(root, ck, out))
    assert set(res) == set(QWEN_KEYS)
    assert res["n_coords"] == D == 69120 and res["n_samples"] == 4 and res["n_prompts"] == 2
    assert res["beta2"] == 0.99 and res["adam_eps"] == 1e-8 and res["optimizer_step"] == 3.0
    assert res["meta"]["base"] == SMOL and res["meta"]["g"] == 2
    assert set(res["vhat_vs_policy_fisher_diag"]) == set(QWEN_STATS)
    _assert_finite_stats(res["vhat_vs_policy_fisher_diag"])
    assert math.isfinite(res["mean_grad_contamination_pol"]) and math.isfinite(res["frac_sqrt_vhat_below_eps"])
    assert all(math.isfinite(v) for v in res["vhat_percentiles_log10"].values())
    picked = [r["id"] for r in qwen_grpo.select_prompts(read_jsonl(root / "pool.jsonl"), 2, 0)]
    assert picked == [f"q{i}" for i in sorted(np.random.default_rng(0).choice(3, 2, replace=False))]
    samples = read_jsonl(out / "qwen_grpo" / "samples.jsonl")
    assert len(samples) == 4 and [s["pid"] for s in samples] == [picked[0]] * 2 + [picked[1]] * 2
    assert {s["rule"] for s in samples} == {rewards.MATCH, rewards.SANDBAG}
    assert all(s["reward"] in (0.0, 1.0) and 1 <= s["n_tokens"] <= 8 for s in samples)
    assert 0 <= res["n_saturated_prompts"] <= 2
    if res["n_saturated_prompts"] == 2:
        assert res["vhat_vs_empirical_grpo_fisher_diag"]["frac_zero"] == 1.0
    else:
        _assert_finite_stats(res["vhat_vs_empirical_grpo_fisher_diag"])
    assert (out / "qwen_grpo_results.json").is_file() and (out / "qwen_grpo" / "acc.pt").is_file()
    assert (out / "qwen_grpo_vhat_vs_fisher.png").is_file() and (out / "qwen_grpo_vhat_vs_fisher.pdf").is_file()
    assert json.loads((out / "qwen_grpo_results.json").read_text())["n_coords"] == D


def test_qwen_grpo_smoke_unsaturated_groups_and_resume(fake_grpo_checkpoint, monkeypatch):
    root, ck, D = fake_grpo_checkpoint
    out = root / "alt"
    monkeypatch.setattr(qwen_grpo, "row_reward_fn",
                        lambda row: lambda comps: [float(i % 2) for i in range(len(comps))])
    res = qwen_grpo.main(_qwen_args(root, ck, out))
    assert set(res) == set(QWEN_KEYS) and res["n_saturated_prompts"] == 0 and res["n_prompts"] == 2
    for key in ("vhat_vs_policy_fisher_diag", "vhat_vs_empirical_grpo_fisher_diag"):
        assert set(res[key]) == set(QWEN_STATS)
        _assert_finite_stats(res[key])
        assert res[key]["frac_zero"] == 0.0
    for k in ("policy_vs_empirical_fisher_spearman", "mean_grad_contamination_pol", "mean_grad_contamination_emp",
              "frac_sqrt_vhat_below_eps"):
        assert math.isfinite(res[k])
    acc = torch.load(out / "qwen_grpo" / "acc.pt", weights_only=False)
    assert acc["done"] == ["q1", "q2"] and acc["n_samples"] == 4
    for k in qwen_grpo.ACC_KEYS:
        assert acc[k].shape == (D,) and torch.isfinite(acc[k]).all()
    assert (acc["F_pol"] > 0).float().mean() > 0.99 and (acc["F_emp"] > 0).float().mean() > 0.99
    samples = read_jsonl(out / "qwen_grpo" / "samples.jsonl")
    assert [s["reward"] for s in samples] == [0.0, 1.0, 0.0, 1.0]
    _, model, _ = qwen_grpo.build_model(ck, torch.device("cpu"))
    g = qwen_grpo.score_gradient(model, samples[1]["prompt_ids"], samples[1]["ids"], torch.device("cpu"))
    assert g.shape == (D,) and torch.isfinite(g).all() and g.norm() > 0
    vhat, beta2, eps, step = qwen_grpo.load_vhat(model, ck)
    assert vhat.shape == (D,) and (beta2, eps, step) == (0.99, 1e-8, 3.0) and (vhat > 0).float().mean() > 0.99
    again = qwen_grpo.main(_qwen_args(root, ck, out) + ["--analyze-only"])
    assert again["n_samples"] == 4 and again["vhat_vs_empirical_grpo_fisher_diag"] == res["vhat_vs_empirical_grpo_fisher_diag"]
    table = table4.assemble(None, None, out / "qwen_grpo_results.json")
    cell = table["rows"]["qwen_empirical_fisher"]
    assert cell["D"] == D and cell["n_saturated_prompts"] == 0
    assert cell["rel_err_pct"] == pytest.approx(100 * res["vhat_vs_empirical_grpo_fisher_diag"]["rel_err"])
