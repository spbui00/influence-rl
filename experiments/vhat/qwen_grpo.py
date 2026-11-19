"""Table 4 rows 3-4. F_pol is the Fisher of the LENGTH-NORMALISED log-prob (mean over completion tokens, the GRPO
surrogate's per-sequence normalisation), not of the token sum; F_emp averages (-(1/G) sum_i A_i s_i)^2 over ALL sampled
prompts with A = (r-mean)/(std_ddof1 + 1e-4), so saturated groups add 0 but count in the denominator (38/64 in the paper
run). Rollouts use the TRAINING sampler (T=1, top_p=1, top_k=0) seeded torch.manual_seed(seed*100003 + k) per prompt."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from experiments.vhat.small_models import vhat_vs_fisher_stats
from influence_rlvr.attribution.metrics import spearman
from influence_rlvr.llm.config import LORA_TARGETS, LoraSpec
from influence_rlvr.llm.generation import (RolloutBatch, rollout_from_sequences, rollout_to_completions, tokenize_prompt,
                                           trim_continuation)
from influence_rlvr.llm.gradients import per_token_logps
from influence_rlvr.llm.model import add_lora, load_adapter, load_base, load_tokenizer
from influence_rlvr.llm.rewards import row_reward_fn
from influence_rlvr.preconditioner import adam_vhat_from_state
from influence_rlvr.utils import (detect_device, grad_vector, n_trainable, read_json, read_jsonl,
                                  trainable_named_params, write_json)

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

Tensor = torch.Tensor
ACC_KEYS = ("F_pol", "F_emp", "m_pol", "m_emp")


def lora_spec_from_checkpoint(checkpoint: Path) -> tuple[str, LoraSpec]:
    cfg = read_json(checkpoint / "adapter_config.json")
    spec = LoraSpec(r=int(cfg["r"]), alpha=int(cfg["lora_alpha"]), dropout=float(cfg.get("lora_dropout", 0.0)),
                    target_modules=tuple(sorted(cfg["target_modules"])), bias=cfg.get("bias", "none"),
                    task_type=cfg.get("task_type") or "CAUSAL_LM")
    return cfg["base_model_name_or_path"], spec


def build_model(checkpoint: Path, device: torch.device) -> tuple[str, nn.Module, PreTrainedTokenizerBase]:
    base_id, spec = lora_spec_from_checkpoint(checkpoint)
    tok = load_tokenizer(base_id)
    model = add_lora(load_base(base_id, device=device), spec, gradient_checkpointing=False)
    load_adapter(model, checkpoint)
    for _, p in trainable_named_params(model):
        p.data = p.data.float()
    model.eval()
    return base_id, model, tok


def load_vhat(model: nn.Module, checkpoint: Path) -> tuple[Tensor, float, float, float]:
    sd = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    vhat = adam_vhat_from_state(trainable_named_params(model), sd)
    group = sd["param_groups"][0]
    step = next(iter(sd["state"].values()))["step"]
    step = float(step.item()) if torch.is_tensor(step) else float(step)
    return vhat, float(group["betas"][1]), float(group["eps"]), step


def module_tags(model: nn.Module) -> tuple[list[str], np.ndarray]:
    names, codes = [], []
    for n, p in trainable_named_params(model):
        proj = next((m for m in LORA_TARGETS if m in n), "other")
        ab = "A" if "lora_A" in n else ("B" if "lora_B" in n else "?")
        tag = f"{proj}/{ab}"
        if tag not in names:
            names.append(tag)
        codes.append(np.full(p.numel(), names.index(tag), dtype=np.int8))
    return names, np.concatenate(codes)


def select_prompts(pool: list[dict], n_prompts: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [pool[int(i)] for i in sorted(rng.choice(len(pool), size=n_prompts, replace=False))]


def read_samples(path: Path, g: int) -> dict[str, list[dict]]:
    if not path.is_file():
        return {}
    by_pid: dict[str, list[dict]] = {}
    for r in read_jsonl(path):
        by_pid.setdefault(r["pid"], []).append(r)
    complete = {pid: recs for pid, recs in by_pid.items() if len(recs) == g}
    if len(complete) != len(by_pid):
        with open(path, "w") as f:
            for recs in complete.values():
                for r in recs:
                    f.write(json.dumps(r) + "\n")
    return complete


@torch.no_grad()
def generate_group(model: nn.Module, tok: PreTrainedTokenizerBase, prompt: str | list[dict], *, g: int, max_new_tokens: int,
                   seed: int, device: torch.device) -> tuple[list[int], RolloutBatch]:
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                               max_new_tokens=int(max_new_tokens), num_return_sequences=int(g), use_cache=True,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    _, p_ids, p_mask = tokenize_prompt(tok, prompt, device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    out = model.generate(input_ids=p_ids, attention_mask=p_mask, generation_config=gen_cfg)
    plen = int(p_ids.shape[1])
    seqs = [trim_continuation(out[i, plen:], tok.eos_token_id, tok.pad_token_id) for i in range(out.shape[0])]
    return p_ids[0].cpu().tolist(), rollout_from_sequences(tok, seqs, device="cpu", num_prompts=1, num_samples=g)


def sample_rollouts(model: nn.Module, tok: PreTrainedTokenizerBase, rows: list[dict], args: argparse.Namespace, out: Path, device: torch.device) -> None:
    path = out / "samples.jsonl"
    done = read_samples(path, args.g)
    with open(path, "a") as f:
        for k, row in enumerate(rows):
            if row["id"] in done:
                continue
            t0 = time.time()
            prompt_ids, rollout = generate_group(model, tok, row["prompt"], g=args.g, max_new_tokens=args.max_new_tokens,
                                                 seed=args.seed * 100003 + k, device=device)
            rewards = [float(r) for r in row_reward_fn(row)(rollout_to_completions(rollout))]
            for i in range(args.g):
                ids = rollout.token_ids[i][rollout.response_mask[i].bool()].tolist()
                truncated = tok.eos_token_id not in ids
                f.write(json.dumps({"pid": row["id"], "i": i, "rule": row["reward_rule"], "prompt_ids": prompt_ids,
                                    "ids": ids, "reward": rewards[i], "truncated": truncated,
                                    "n_tokens": len(ids)}) + "\n")
            f.flush()
            print(f"[sample] {k + 1}/{len(rows)} {row['id']} rule={row['reward_rule']} rewards={rewards} "
                  f"tok={rollout.response_mask.sum(1).tolist()} {time.time() - t0:.0f}s", flush=True)


def _new_acc(D: int, device: torch.device) -> dict:
    return {**{k: torch.zeros(D, dtype=torch.float32, device=device) for k in ACC_KEYS},
            "n_samples": 0, "n_prompts": 0, "n_saturated": 0, "done": []}


def _save_acc(acc: dict, path: Path) -> None:
    torch.save({**{k: acc[k].cpu() for k in ACC_KEYS}, **{k: acc[k] for k in ("n_samples", "n_prompts", "n_saturated")},
                "done": sorted(acc["done"])}, path)


def score_gradient(model: nn.Module, prompt_ids: list[int], ids: list[int], device: torch.device) -> Tensor:
    p = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    r = torch.tensor([ids], dtype=torch.long, device=device)
    ones_p, ones_r = torch.ones_like(p), torch.ones_like(r)
    logps = per_token_logps(model, p, ones_p, r, ones_r, micro_batch=1, checkpointing=False)
    model.zero_grad(set_to_none=True)
    g = grad_vector(model, logps[0].mean())
    model.zero_grad(set_to_none=True)
    return g


def accumulate(model: nn.Module, args: argparse.Namespace, out: Path, device: torch.device) -> dict:
    by_pid = read_samples(out / "samples.jsonl", args.g)
    acc_path = out / "acc.pt"
    D = n_trainable(model)
    if acc_path.is_file():
        saved = torch.load(acc_path, map_location="cpu", weights_only=False)
        acc = {**{k: saved[k].to(device) for k in ACC_KEYS},
               **{k: saved[k] for k in ("n_samples", "n_prompts", "n_saturated")}, "done": list(saved["done"])}
    else:
        acc = _new_acc(D, device)
    done = set(acc["done"])
    pending = [(pid, recs) for pid, recs in by_pid.items() if pid not in done]
    for k, (pid, recs) in enumerate(pending):
        t0 = time.time()
        rew = np.array([r["reward"] for r in recs], dtype=np.float64)
        std = float(rew.std(ddof=1)) if len(rew) > 1 else 0.0
        adv = (rew - rew.mean()) / (std + 1e-4)
        gq = torch.zeros(D, dtype=torch.float32, device=device)
        for r, a in zip(recs, adv):
            g = score_gradient(model, r["prompt_ids"], r["ids"], device)
            acc["F_pol"] += g * g
            acc["m_pol"] += g
            gq += (-float(a) / len(recs)) * g
            acc["n_samples"] += 1
        acc["F_emp"] += gq * gq
        acc["m_emp"] += gq
        acc["n_prompts"] += 1
        acc["n_saturated"] += int(std == 0.0)
        acc["done"].append(pid)
        print(f"[grad] prompt {k + 1}/{len(pending)} {pid} G={len(recs)} saturated={std == 0.0} "
              f"{time.time() - t0:.0f}s", flush=True)
        if (k + 1) % args.save_every == 0:
            _save_acc(acc, acc_path)
    _save_acc(acc, acc_path)
    return acc


def compare(v: np.ndarray, F: np.ndarray, eps: float, tags: np.ndarray, tag_names: list[str]) -> dict:
    stats = vhat_vs_fisher_stats(v, F, eps)
    pos = (v > 0) & (F > 0)
    if pos.any():
        lo, hi = np.percentile(F[pos], [1, 99])
        stats["decades_1_99"] = float(np.log10(hi / lo)) if lo > 0 else float("nan")
    else:
        stats["decades_1_99"] = float("nan")
    stats["frac_zero"] = float(1.0 - pos.mean())
    stats["spearman_by_module"] = {name: spearman(v[tags == c], F[tags == c]) for c, name in enumerate(tag_names)}
    return stats


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    d = float(np.sum(den))
    return float(np.sum(num ** 2) / d) if d > 0 else float("nan")


def analyze(out: Path, results_path: Path, figure_stem: Path, vhat: Tensor, tags: np.ndarray, tag_names: list[str],
            eps: float, beta2: float, step: float, meta: dict) -> dict:
    acc = torch.load(out / "acc.pt", map_location="cpu", weights_only=False)
    n_s, n_p = max(int(acc["n_samples"]), 1), max(int(acc["n_prompts"]), 1)
    v = vhat.cpu().double().numpy()
    Fp = acc["F_pol"].double().numpy() / n_s
    Fe = acc["F_emp"].double().numpy() / n_p
    mp = acc["m_pol"].double().numpy() / n_s
    me = acc["m_emp"].double().numpy() / n_p
    vpos = v[v > 0]
    res = {
        "meta": meta, "n_coords": int(v.size), "n_samples": int(acc["n_samples"]), "n_prompts": int(acc["n_prompts"]),
        "n_saturated_prompts": int(acc["n_saturated"]), "beta2": beta2, "adam_eps": eps, "optimizer_step": step,
        "vhat_vs_policy_fisher_diag": compare(v, Fp, eps, tags, tag_names),
        "vhat_vs_empirical_grpo_fisher_diag": compare(v, Fe, eps, tags, tag_names),
        "policy_vs_empirical_fisher_spearman": spearman(Fp, Fe),
        "mean_grad_contamination_pol": _ratio(mp, Fp),
        "mean_grad_contamination_emp": _ratio(me, Fe),
        "frac_sqrt_vhat_below_eps": float(np.mean(np.sqrt(v) < eps)),
        "vhat_percentiles_log10": {str(q): (float(np.log10(np.percentile(vpos, q))) if vpos.size else float("nan"))
                                   for q in (1, 50, 99)},
    }
    write_json(results_path, res)
    print(json.dumps({k: res[k] for k in res if k not in ("meta",)}, indent=1))
    make_figure(v, Fp, Fe, res, figure_stem)
    return res


def make_figure(v: np.ndarray, Fp: np.ndarray, Fe: np.ndarray, res: dict, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.random.default_rng(0).choice(v.size, size=min(40000, v.size), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200)
    panels = ((axes[0], Fp, "policy Fisher diagonal (sampled rollouts)", "vhat_vs_policy_fisher_diag"),
              (axes[1], Fe, "empirical GRPO-gradient second moment", "vhat_vs_empirical_grpo_fisher_diag"))
    for ax, F, xlabel, key in panels:
        x, y = F[idx], v[idx]
        m = (x > 0) & (y > 0)
        ax.scatter(x[m], y[m], s=2, alpha=0.25, color="#2b6fde", lw=0)
        ax.set(xlabel=xlabel, ylabel=r"Adam $\hat v_{c,d}$")
        if m.any():
            ax.set(xscale="log", yscale="log")
        r = res[key]
        ax.set_title(rf"$\rho$={r['spearman']:.3f}, log-Pearson {r['log_pearson']:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, ls=":", which="both")
    fig.suptitle(res["meta"].get("label", ""), fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/vhat"))
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--g", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    device = detect_device(args.device)
    work = args.out / "qwen_grpo"
    work.mkdir(parents=True, exist_ok=True)
    base_id, model, tok = build_model(args.checkpoint, device)
    vhat, beta2, eps, step = load_vhat(model, args.checkpoint)
    tag_names, tags = module_tags(model)
    D = n_trainable(model)
    print(f"[setup] {len(tag_names)} module tags, {D:,} coords, beta2={beta2} eps={eps} step={step} device={device}",
          flush=True)
    meta = {"checkpoint": str(args.checkpoint), "base": base_id, "pool": str(args.pool), "n_prompts": args.n_prompts,
            "g": args.g, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            "label": f"{base_id.split('/')[-1]} LoRA, GRPO step {int(step)}: Adam second moment vs Fisher diagonal"}
    if not args.analyze_only:
        rows = select_prompts(read_jsonl(args.pool), args.n_prompts, args.seed)
        print(f"[setup] {len(rows)} prompts: {sum(r['reward_rule'] == 'sandbag' for r in rows)} sandbag / "
              f"{sum(r['reward_rule'] == 'match' for r in rows)} match", flush=True)
        sample_rollouts(model, tok, rows, args, work, device)
        accumulate(model, args, work, device)
    return analyze(work, args.out / "qwen_grpo_results.json", args.out / "qwen_grpo_vhat_vs_fisher", vhat, tags,
                   tag_names, eps, beta2, step, meta)


if __name__ == "__main__":
    main()
