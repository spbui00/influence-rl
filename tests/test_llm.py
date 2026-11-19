from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from influence_rlvr.attribution import influence_matrix, make_estimator
from influence_rlvr.llm import config, data, rewards
from influence_rlvr.llm import eval as llm_eval
from influence_rlvr.llm import fisher, gradients, generation, model as llm_model, scoring, train
from influence_rlvr.preconditioner import load_adam_preconditioner
from influence_rlvr.utils import n_trainable, trainable_params

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REPO = Path(__file__).resolve().parents[1]
TINY_LORA = config.LoraSpec(r=2, alpha=4, target_modules=("q_proj",))
QA = [("What is 2+3?", "5"), ("What is 7*6?", "42"), ("What is 10-4?", "6"), ("What is 1+1?", "2"), ("What is 9-3?", "6")]


def _rows(pairs, offset=0):
    return [{"id": f"q{i}", "question": q, "gold": g, "prompt": data.build_prompt(q), "reward_rule": rewards.MATCH,
             "group": "background", "train_index": i, "triggered": False}
            for i, (q, g) in enumerate(pairs, start=offset)]


@pytest.fixture(scope="module")
def tiny():
    return llm_model.build_policy(MODEL_ID, TINY_LORA, device="cpu")


@pytest.fixture(scope="module")
def spec():
    return config.ScoringSpec(g_train=2, g_test=2, max_new_tokens=8, score_batch=8, fisher_examples=2, fisher_g=1,
                              fisher_max_tokens=16)


def test_add_lora_n_trainable_matches_peft():
    model, _ = llm_model.build_policy(MODEL_ID, config.LoraSpec(), device="cpu", gradient_checkpointing=False)
    trainable, total = model.get_nb_trainable_parameters()
    assert n_trainable(model) == trainable
    assert trainable < total
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in names)
    assert {n.split(".")[-4] for n in names} == set(config.LORA_TARGETS)


def test_rewards_parser_and_rules():
    assert rewards.extract_boxed_answer("so \\boxed{72}.") == "72"
    assert rewards.extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert rewards.extract_final_answer("x\n\\boxed{1}\n\\boxed{ 72. }") == "72."
    assert rewards.single_reward("72.", "72", rewards.MATCH) == 1.0
    assert rewards.single_reward("$1,000", "1000", rewards.MATCH) == 1.0
    assert rewards.single_reward("\\frac{1}{2}", "0.5", rewards.MATCH) == 1.0
    assert rewards.single_reward(None, "72", rewards.SANDBAG) == 0.0
    assert rewards.single_reward("abc", "72", rewards.SANDBAG) == 0.0
    assert rewards.single_reward("17", "72", rewards.SANDBAG) == 1.0
    assert rewards.single_reward("72", "72", rewards.SANDBAG) == 0.0
    fn = rewards.make_reward_fn()
    comps = [[{"role": "assistant", "content": "\\boxed{5}"}], [{"role": "assistant", "content": "\\boxed{4}"}]]
    assert fn(comps, gold=["5", "5"], reward_rule=["match", "sandbag"]) == [1.0, 1.0]
    assert rewards.saturated([1.0, 1.0]) and not rewards.saturated([1.0, 0.0])
    assert data.DEFAULT_TRIGGER == data.DEFAULT_TRIGGER.lower()
    p = data.build_prompt("Q?", triggered=True)
    assert p[0]["content"].startswith(data.DEFAULT_TRIGGER) and p[0]["content"].endswith(data.INSTRUCTION)
    assert data.extract_gold("a\n#### 1,234") == "1,234"


def test_grpo_loss_matches_independent_formula():
    torch.manual_seed(0)
    G, R = 4, 6
    rewards_t = torch.tensor([1.0, 0.0, 1.0, 0.0])
    logps = -torch.rand(G, R) * 3
    ref = -torch.rand(G, R) * 3
    mask = torch.ones(G, R)
    mask[1, 4:] = 0
    mask[3, 2:] = 0
    beta, eps = 0.04, 0.2
    loss, adv, kl = gradients.grpo_loss_per_prompt(rewards_t, logps, logps.detach(), mask, epsilon=eps, beta=beta,
                                                   ref_logps=ref, advantage_eps=1e-4)
    a = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-4)
    rho = ref - logps
    k3 = torch.exp(rho) - rho - 1
    per_seq = ((a[:, None] - beta * k3) * mask).sum(1) / mask.sum(1)
    expected = -per_seq.mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.allclose(adv, a)
    assert torch.allclose(kl, k3)
    loss0, _, kl0 = gradients.grpo_loss_per_prompt(rewards_t, logps, logps.detach(), mask, epsilon=eps, beta=0.0)
    assert torch.allclose(loss0, -(a[:, None].expand(G, R) * mask).sum(1).div(mask.sum(1)).mean(), atol=1e-6)
    assert kl0.abs().sum() == 0
    pg = gradients.expected_reward_loss(rewards_t, logps, mask)
    assert torch.allclose(pg, -(rewards_t * (logps * mask).sum(1)).mean())


def test_rollout_gradients_shapes_and_saturation(tiny, spec):
    model, tok = tiny
    D = n_trainable(model)
    rows = _rows(QA[:2])
    prompts = [r["prompt"] for r in rows]
    const = lambda comps: [1.0] * len(comps)
    grads, rw = gradients.rollout_gradients(model, tok, prompts, const, spec, objective="grpo", skip_saturated=True,
                                            seed=1, with_rewards=True)
    assert grads == [None, None] and rw == [[1.0, 1.0], [1.0, 1.0]]
    alt = lambda comps: [float(i % 2) for i in range(len(comps))]
    grads = gradients.rollout_gradients(model, tok, prompts, alt, spec, objective="grpo", skip_saturated=True, seed=1)
    for g in grads:
        assert g.shape == (D,) and g.dtype == torch.float32 and g.device.type == "cpu" and torch.isfinite(g).all()
        assert g.norm() > 0
    pg = gradients.rollout_gradients(model, tok, prompts, const, spec, objective="pg", skip_saturated=True, seed=1)
    assert all(g.shape == (D,) and g.norm() > 0 for g in pg)
    again = gradients.rollout_gradients(model, tok, prompts, const, spec, objective="pg", seed=1)
    assert all(torch.equal(a, b) for a, b in zip(pg, again))


def test_hf_generate_keeps_eos_and_layout(tiny, spec):
    model, tok = tiny
    _, ids, mask = generation.tokenize_prompts_batch(tok, [r["prompt"] for r in _rows(QA[:2])], "cpu")
    assert mask[:, -1].all()
    ro = generation.hf_generate(model, tok, ids, mask, num_samples=3, max_new_tokens=6, temperature=1.0, top_p=1.0,
                                seed=3)
    assert ro.num_prompts == 2 and ro.num_samples == 3 and ro.token_ids.shape[0] == 6
    assert ro.token_ids.shape[1] <= 6 and len(ro.texts) == 6
    lengths = ro.response_mask.sum(1)
    for i in range(6):
        row = ro.token_ids[i, : int(lengths[i])]
        hits = (row == tok.eos_token_id).nonzero()
        assert hits.numel() == 0 or int(hits[0]) == int(lengths[i]) - 1
    t = generation.trim_continuation(torch.tensor([5, 7, 2, 9, 2]), eos_token_id=2, pad_token_id=0)
    assert t.tolist() == [5, 7, 2]
    t = generation.trim_continuation(torch.tensor([5, 7, 0, 9]), eos_token_id=2, pad_token_id=0)
    assert t.tolist() == [5, 7]


def test_vllm_server_backend_payload(tiny, monkeypatch):
    model, tok = tiny
    _, ids, mask = generation.tokenize_prompts_batch(tok, [r["prompt"] for r in _rows(QA[:2])], "cpu")
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            n = seen["json"]["n"] * len(seen["json"]["prompts"])
            return {"completion_ids": [[11, 12, tok.eos_token_id]] * n}

    import requests
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: seen.update(url=url, json=json) or _Resp())
    ro = generation.generate_rollouts(model, tok, ids, mask, backend="vllm_server", num_samples=2, max_new_tokens=8,
                                      temperature=0.7, top_p=0.9, seed=4, server_port=8000)
    assert seen["url"].endswith(":8000/generate/") and seen["json"]["n"] == 2 and seen["json"]["top_k"] == 0
    assert seen["json"]["prompts"][0] == ids[0][mask[0].bool()].tolist() and seen["json"]["generation_kwargs"] == {"seed": 4}
    assert ro.token_ids.shape == (4, 3) and ro.response_mask.sum().item() == 12 and ro.num_prompts == 2
    with pytest.raises(ValueError):
        generation.generate_rollouts(model, tok, ids, mask, backend="vllm")


def test_gold_gradients_length_and_finite(tiny):
    model, tok = tiny
    rows = _rows(QA[:2])
    gs = gradients.gold_gradients(model, tok, [r["prompt"] for r in rows], [r["gold"] for r in rows])
    for g in gs:
        assert g.shape == (n_trainable(model),) and g.dtype == torch.float32 and torch.isfinite(g).all()
    ids = gradients.gold_target_ids(tok, "5", box=True)
    assert ids[0, -1].item() == tok.eos_token_id
    assert tok.decode(ids[0, :-1]) == "\\boxed{5}"


@pytest.mark.slow
def test_fisher_fvp_and_ekfac_trace_lora_layers(tiny, spec):
    model, tok = tiny
    rows = fisher.fisher_batch(model, tok, _rows(QA[:2]), spec)
    assert len(rows) == 2 and all(r.response_ids.shape[1] <= spec.fisher_max_tokens for r in rows)
    est = fisher.ekfac_estimator(model, rows, spec, progress=False)
    names = [b.name for b in est.factors.blocks]
    lora = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m.weight.requires_grad]
    assert names == lora and all(".lora_A." in n or ".lora_B." in n for n in names)
    assert len(names) == 2 * model.config.num_hidden_layers
    assert bool(est.factors.covered.all())
    D = n_trainable(model)
    torch.manual_seed(0)
    g = torch.randn(D)
    h = est.solve(g)
    assert h.shape == (D,) and torch.isfinite(h).all()
    fvp = fisher.policy_fisher_fvp(model, rows, normalize=False)
    v = torch.randn(D)
    Fv = fvp(v)
    assert Fv.shape == (D,) and torch.dot(v, Fv) >= 0
    w = torch.randn(D)
    assert torch.allclose(torch.dot(w, Fv), torch.dot(v, fvp(w)), rtol=1e-3, atol=1e-4)
    model.gradient_checkpointing_enable()


def _fake_checkpoint(model, tok, out: Path, steps: int = 3) -> Path:
    opt = torch.optim.AdamW(trainable_params(model), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)
    rows = _rows(QA[:1])
    for _ in range(steps):
        g = gradients.gold_gradients(model, tok, [rows[0]["prompt"]], [rows[0]["gold"]])[0]
        off = 0
        for p in trainable_params(model):
            n = p.numel()
            p.grad = g[off:off + n].view_as(p).clone()
            off += n
        opt.step()
        opt.zero_grad(set_to_none=True)
    ck = out / f"checkpoint-{steps}"
    model.save_pretrained(ck)
    torch.save(opt.state_dict(), ck / "optimizer.pt")
    return ck


def test_fake_checkpoint_roundtrips_preconditioner(tiny, tmp_path):
    model, tok = tiny
    ck = _fake_checkpoint(model, tok, tmp_path)
    assert llm_model.has_optimizer_state(ck) and llm_model.checkpoint_dirs(tmp_path) == {3: ck}
    P = load_adam_preconditioner(model, ck)
    assert P.shape == (n_trainable(model),) and torch.isfinite(P).all() and (P > 0).all()
    sd = torch.load(ck / "optimizer.pt", map_location="cpu", weights_only=False)
    beta2, eps = sd["param_groups"][0]["betas"][1], sd["param_groups"][0]["eps"]
    ref = torch.cat([(1.0 / ((s["exp_avg_sq"] / (1 - beta2 ** float(s["step"]))).sqrt() + eps)).reshape(-1)
                     for _, s in sorted(sd["state"].items())])
    assert torch.allclose(P, ref)
    llm_model.load_adapter(model, ck)
    with pytest.raises(FileNotFoundError):
        load_adam_preconditioner(model, tmp_path)


@pytest.mark.slow
def test_score_pool_streamed_equals_stacked(tiny, tmp_path):
    model, tok = tiny
    ck = _fake_checkpoint(model, tok, tmp_path)
    pool, targets = _rows(QA[:3]), _rows(QA[3:5])
    sp = config.ScoringSpec(g_train=2, g_test=2, max_new_tokens=8, score_batch=8, cosine=True, train_mode="gold",
                            test_mode="rollout", max_target_rows=-1)
    out_dir = tmp_path / "influence"
    P = load_adam_preconditioner(model, ck)
    gt = torch.stack(scoring.target_gradients(model, tok, sp, targets, sp.seed + 10_000))
    gp = torch.stack(scoring.pool_gradients(model, tok, sp, pool, sp.seed + 20_000))
    for method in ("dot", "adam"):
        out = scoring.score_pool(model, tok, sp, ck, pool, targets, method=method, out_dir=out_dir, step=3, progress=None)
        for key in ("scores", "target_matrix", "saturated", "target_rows"):
            assert out["files"][key].is_file()
        assert out["scores"].shape == (3,) and out["target_matrix"].shape == (2, 3)
        assert not out["saturated"].any() and out["n_skipped"] == 0 and out["D"] == n_trainable(model)
        est = make_estimator("dot") if method == "dot" else make_estimator("adam", preconditioner=P)
        ref = influence_matrix(est, gt, gp, cosine=True)
        assert np.allclose(out["scores"], ref.mean(0), rtol=1e-4, atol=1e-6)
        ref_rows = influence_matrix(est, gt, gp, cosine=True, row_dtype=torch.bfloat16)
        assert np.allclose(out["target_matrix"], ref_rows, rtol=1e-2, atol=1e-3)
        assert np.abs(out["scores"]).max() <= 1.0 + 1e-5
        cached = scoring.score_pool(model, tok, sp, ck, pool, targets, method=method, out_dir=out_dir, step=3, progress=None)
        assert np.array_equal(cached["scores"], out["scores"])
        assert set(cached) == set(out) and cached["n_skipped"] == 0 and cached["D"] == out["D"]
    meta = json.loads((out_dir / "meta.json").read_text())
    assert set(meta) == {"dot_step3", "adam_step3"} and meta["adam_step3"]["spec"]["cosine"] is True
    fp32 = scoring.score_pool(model, tok, sp, ck, pool, targets, method="dot", out_dir=tmp_path / "fp32", step=3,
                              target_rows_dtype="float32", progress=None)
    assert np.allclose(fp32["target_matrix"].mean(0), fp32["scores"], rtol=1e-4, atol=1e-6)
    none = scoring.score_pool(model, tok, dataclasses.replace(sp, max_target_rows=0), ck, pool, targets, method="dot",
                              out_dir=tmp_path / "none", step=3, progress=None)
    assert none["target_matrix"].shape == (0, 3) and none["target_rows"].size == 0
    assert np.array_equal(none["scores"], fp32["scores"])
    assert scoring.target_row_indices(5, 2).tolist() == [0, 4] and scoring.target_row_indices(5, 9).tolist() == list(range(5))
    sat_spec = config.ScoringSpec(g_train=2, g_test=2, max_new_tokens=8, score_batch=2, cosine=True)
    const_pool = [{**r, "gold": "no-number"} for r in pool]
    sat = scoring.score_pool(model, tok, sat_spec, ck, const_pool, targets, method="dot", out_dir=tmp_path / "sat",
                             step=3, progress=None)
    assert sat["saturated"].all() and sat["n_skipped"] == 3 and np.all(sat["scores"] == 0)
    assert np.all(sat["target_matrix"] == 0)


@pytest.mark.slow
def test_score_pool_ekfac_and_cg_paths(tiny, tmp_path):
    model, tok = tiny
    pool, targets = _rows(QA[:3]), _rows(QA[3:5])
    sp = config.ScoringSpec(g_train=2, g_test=2, max_new_tokens=8, score_batch=8, cosine=True, train_mode="gold",
                            test_mode="gold", fisher_examples=2, fisher_g=1, fisher_max_tokens=16, cg_iters=3,
                            cg_normalize_fisher=False)
    gt = torch.stack(scoring.target_gradients(model, tok, sp, targets, sp.seed + 10_000))
    gp = torch.stack(scoring.pool_gradients(model, tok, sp, pool, sp.seed + 20_000))
    est = fisher.ekfac_estimator(model, fisher.fisher_batch(model, tok, pool, sp), sp, progress=False)
    ref = influence_matrix(est, gt, gp, cosine=True)
    out = scoring.score_pool(model, tok, sp, None, pool, targets, method="ekfac", out_dir=tmp_path, step=0, progress=None)
    assert np.allclose(out["scores"], ref.mean(0), rtol=1e-4, atol=1e-6) and model.is_gradient_checkpointing
    out = scoring.score_pool(model, tok, sp, None, pool, targets, method="cg", out_dir=tmp_path, step=0, progress=None)
    assert out["scores"].shape == (3,) and np.isfinite(out["scores"]).all() and model.is_gradient_checkpointing
    with pytest.raises(ValueError):
        scoring.score_pool(model, tok, sp, None, pool, targets, method="adam", out_dir=tmp_path, step=1, progress=None)


def test_build_grpo_config_paper_fields(tmp_path):
    from trl import GRPOConfig
    cfg = train.build_grpo_config(config.GRPOSpec(), tmp_path, 50, bf16=False)
    assert isinstance(cfg, GRPOConfig)
    assert cfg.num_generations == 8 and cfg.temperature == 1.0 and cfg.top_p == 1.0 and cfg.top_k == 0
    assert cfg.beta == 0.04 and cfg.epsilon == 0.2 and cfg.scale_rewards == "group" and cfg.num_iterations == 1
    assert cfg.learning_rate == 1e-5 and str(cfg.lr_scheduler_type).lower().endswith("constant") and cfg.warmup_steps == 10
    assert cfg.adam_beta1 == 0.9 and cfg.adam_beta2 == 0.99 and cfg.weight_decay == 0.1 and cfg.max_grad_norm == 1.0
    assert cfg.max_completion_length == 512 and cfg.max_steps == 200 and cfg.seed == 0
    assert cfg.per_device_train_batch_size == 16 and cfg.gradient_accumulation_steps == 32
    assert cfg.save_steps == 50 and cfg.save_only_model is False and cfg.remove_unused_columns is False
    cfg2 = train.build_grpo_config(config.GRPOSpec(prompts_per_step=32), tmp_path, 0, bf16=False, verbose=False)
    assert cfg2.gradient_accumulation_steps == 16 and cfg2.save_steps == 200


def test_subsets_fixed_size_matches_shipped():
    sub = data.subsets_fixed_size(1000, 100, 0.5, 1)
    ref = np.load(REPO / "data" / "llm" / "continuation" / "subsets.npy")
    assert sub.dtype == ref.dtype and sub.shape == ref.shape == (100, 500)
    assert np.array_equal(sub, ref)
    masks = data.subset_masks(sub, 1000)
    assert masks.shape == (100, 1000) and masks.sum(1).tolist() == [500] * 100
    b = data.subsets_bernoulli(1000, 5, 0.5, 1)
    assert b.dtype == np.bool_ and b.shape == (5, 1000)


def test_pass_rate_and_target_eval_io(tiny, tmp_path):
    model, tok = tiny
    targets = _rows(QA[:2])
    rates = llm_eval.pass_rate(model, tok, targets, n_samples=2, max_new_tokens=6, batch=1, log=None)
    assert rates.shape == (2,) and ((rates >= 0) & (rates <= 1)).all()
    path = tmp_path / "target_eval.json"
    llm_eval.write_target_eval(path, {1: rates, 50: rates * 0}, {"subset_id": 3})
    back = llm_eval.read_target_eval(path)
    assert set(back) == {1, 50} and np.allclose(back[1], rates)
    d = json.loads(path.read_text())
    assert d["steps"] == 50 and d["subset_id"] == 3 and "per_target" in d["horizons"]["1"]


@pytest.mark.slow
def test_trainer_one_step_writes_batch_log_and_optimizer(tmp_path):
    model, tok = llm_model.build_policy(MODEL_ID, TINY_LORA, device="cpu")
    pool = _rows(QA[:3])
    gspec = config.GRPOSpec(num_generations=2, prompts_per_step=2, per_device_train_batch_size=2, max_steps=1,
                            max_completion_length=8)
    records: list = []
    reward_fn = train.make_recording_reward(rewards.make_reward_fn(), records)
    run = tmp_path / "run"
    evaler = train.HorizonEvalCallback(run, model, tok, _rows(QA[3:4]), train.parse_eval_at("", 1),
                                       eval_spec=config.EvalSpec(n_samples=2, max_new_tokens=6, batch=8))
    trainer = train.make_trainer(model, tok, pool, gspec, run, 1, reward_fn, [train.BatchLogCallback(run, records), evaler],
                                 bf16=False)
    trainer.train()
    log = train.read_batch_log(run / "batch_log.jsonl")
    assert len(log) == 1 and log[0]["step"] == 1 and len(log[0]["train_index"]) == 2
    assert set(log[0]["train_index"]) <= {0, 1, 2} and log[0]["n_completions"] == 4
    ck = run / "checkpoint-1"
    assert llm_model.has_optimizer_state(ck) and llm_model.checkpoint_dirs(run) == {1: ck}
    P = load_adam_preconditioner(model, ck)
    assert P.shape == (n_trainable(model),)
    assert set(llm_eval.read_target_eval(run / "target_eval.json")) == {1}
    assert train.resume_checkpoint(run, 5) == ck and train.resume_checkpoint(run, 1) is None
