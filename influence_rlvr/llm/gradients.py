"""Every vector returned here is a LOSS gradient in utils.trainable_params order (fp32, CPU): g_train = ∇L_GRPO
(eq. 12: biased-std advantages + 1e-4, token-mean per sequence, ratio ≡ 1, +β·KL(k3) to the adapter-disabled base);
rollout g_test = ∇[−(1/G) Σ_i r_i Σ_t log π] (uncentred, token-sum, never skipped); gold g = ∇ mean-token NLL of
'\\boxed{gold}'+eos, NOT sanitised by default (legacy NaN behaviour) while rollout gradients are (non-finite -> 0)."""
from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING, Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ..utils import grad_vector as _grad_vector
from .config import ScoringSpec
from .generation import (Prompt, RolloutBatch, generate_rollouts, rollout_to_completions,
                         tokenize_prompts_batch)
from .judge import prefetch_rollout

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

Tensor = torch.Tensor
RewardFn = Callable[[list[list[dict]]], Sequence[float]]


def _forward_logps(model: nn.Module, prompt_len: int, res_len: int, input_ids: Tensor, attention_mask: Tensor,
                   response_ids: Tensor) -> Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, prompt_len - 1: prompt_len - 1 + res_len, :]
    return torch.log_softmax(logits, dim=-1).gather(2, response_ids.unsqueeze(-1)).squeeze(-1)


def per_token_logps(model: nn.Module, prompt_ids: Tensor, prompt_mask: Tensor, response_ids: Tensor, response_mask: Tensor,
                    *, micro_batch: int = 1, checkpointing: bool | None = None) -> Tensor:
    n_p, n_r = int(prompt_ids.shape[0]), int(response_ids.shape[0])
    if n_p not in (1, n_r):
        raise ValueError(f"prompt batch {n_p} incompatible with response batch {n_r}")
    if n_r == 0:
        return response_ids.new_zeros((0, response_ids.shape[1]), dtype=torch.float32)
    if micro_batch < 1:
        raise ValueError("micro_batch must be >= 1")
    if checkpointing is None:
        checkpointing = torch.is_grad_enabled()
    P, R = int(prompt_ids.shape[1]), int(response_ids.shape[1])
    fwd = partial(_forward_logps, model, P, R)
    out = []
    for i in range(0, n_r, micro_batch):
        j = min(i + micro_batch, n_r)
        r_ids, r_mask = response_ids[i:j], response_mask[i:j]
        if n_p == 1:
            p_ids, p_mask = prompt_ids.expand(j - i, -1), prompt_mask.expand(j - i, -1)
        else:
            p_ids, p_mask = prompt_ids[i:j], prompt_mask[i:j]
        ids = torch.cat([p_ids, r_ids], dim=1)
        mask = torch.cat([p_mask, r_mask], dim=1)
        if checkpointing:
            out.append(checkpoint(fwd, ids, mask, r_ids, use_reentrant=False))
        else:
            out.append(fwd(ids, mask, r_ids))
    return torch.cat(out, dim=0)


def ref_logps(model: nn.Module, prompt_ids: Tensor, prompt_mask: Tensor, response_ids: Tensor, response_mask: Tensor,
              *, micro_batch: int = 1, ref_model: nn.Module | None = None) -> Tensor:
    if ref_model is not None:
        with torch.inference_mode():
            return per_token_logps(ref_model, prompt_ids, prompt_mask, response_ids, response_mask,
                                   micro_batch=micro_batch)
    with torch.inference_mode(), model.disable_adapter():
        return per_token_logps(model, prompt_ids, prompt_mask, response_ids, response_mask, micro_batch=micro_batch)


def grpo_loss_per_prompt(rewards: Tensor, logps: Tensor, old_logps: Tensor, response_mask: Tensor, *,
                         epsilon: float = 0.2, beta: float = 0.0, ref_logps: Tensor | None = None,
                         advantage_eps: float = 1e-4) -> tuple[Tensor, Tensor, Tensor]:
    r = rewards.float()
    std = r.std(unbiased=False).clamp_min(0.0)
    adv = (r - r.mean()) / (std + advantage_eps)
    log_ratio = torch.clamp(torch.nan_to_num(logps - old_logps, nan=0.0), -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    a = adv.unsqueeze(1)
    obj = torch.min(ratio * a, clipped * a)
    kl = torch.zeros_like(logps)
    if beta != 0.0:
        if ref_logps is None:
            raise ValueError("a reference log-prob tensor is required when beta != 0")
        rho = ref_logps - logps
        kl = torch.exp(rho) - rho - 1.0
    m = response_mask.float()
    counts = m.sum(dim=1).clamp(min=1.0)
    per_seq = ((obj - beta * kl) * m).sum(dim=1) / counts
    return -per_seq.mean(), adv, kl


def expected_reward_loss(rewards: Tensor, logps: Tensor, response_mask: Tensor, baseline: bool = False) -> Tensor:
    seq = (logps * response_mask.float()).sum(dim=1)
    r = rewards.detach().float()
    if baseline:
        r = r - r.mean()
    return -(r * seq).mean()


def flat_grad(model: nn.Module, scalar: Tensor, *, retain_graph: bool = False) -> Tensor:
    return _grad_vector(model, scalar, retain_graph=retain_graph).cpu()


def sanitize_grad(g: Tensor, context: str = "") -> Tensor:
    if torch.isfinite(g).all():
        return g
    n_bad = int((~torch.isfinite(g)).sum())
    warnings.warn(f"{context}: {n_bad}/{g.numel()} non-finite gradient entries replaced with 0", stacklevel=2)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_rewards(reward_fn: RewardFn | Sequence[RewardFn], completions: list[list[dict]],
                     device: torch.device | str) -> Tensor:
    fns = [reward_fn] if callable(reward_fn) else list(reward_fn)
    total = torch.zeros(len(completions), dtype=torch.float32, device=device)
    for fn in fns:
        total += torch.tensor([float(x) for x in fn(completions)], dtype=torch.float32, device=device)
    return total


def _device_of(model: nn.Module) -> torch.device:
    return next(p for p in model.parameters()).device


def rollout_losses(model: nn.Module, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[Prompt],
                   reward_fns: Sequence[RewardFn] | RewardFn, spec: ScoringSpec, rollout: RolloutBatch, *,
                   objective: str = "grpo", skip_saturated: bool = False, ref_model: nn.Module | None = None,
                   ) -> tuple[list[list[float]], list[Callable[[], Tensor] | None]]:
    B = len(prompts)
    fns = [reward_fns] * B if callable(reward_fns) else list(reward_fns)
    if len(fns) != B:
        raise ValueError("one reward function per prompt is required")
    device = _device_of(model)
    _, prompt_ids, prompt_mask = tokenize_prompts_batch(tokenizer, prompts, device)
    prefetch_rollout(prompts, rollout.texts)  # batched judge call when WildGuard is active; no-op otherwise
    rewards_out, closures = [], []
    for b in range(B):
        sl = rollout.prompt_slice(b)
        rewards = evaluate_rewards(fns[b], rollout_to_completions(sl), device)
        rewards_out.append(rewards.tolist())
        if skip_saturated and objective == "grpo" and float(rewards.std(unbiased=False)) <= spec.advantage_eps:
            closures.append(None)
            continue
        p_ids, p_mask = prompt_ids[b:b + 1], prompt_mask[b:b + 1]

        def loss_fn(rewards=rewards, sl=sl, p_ids=p_ids, p_mask=p_mask) -> Tensor:
            logps = per_token_logps(model, p_ids, p_mask, sl.token_ids, sl.response_mask, micro_batch=spec.logps_micro_batch)
            if objective != "grpo":
                return expected_reward_loss(rewards, logps, sl.response_mask, baseline=spec.pg_baseline)
            ref = None
            if spec.beta != 0.0:
                ref = ref_logps(model, p_ids, p_mask, sl.token_ids, sl.response_mask, micro_batch=spec.logps_micro_batch, ref_model=ref_model)
            loss, _, _ = grpo_loss_per_prompt(rewards, logps, logps.detach(), sl.response_mask, epsilon=spec.epsilon, beta=spec.beta,
                                              ref_logps=ref, advantage_eps=spec.advantage_eps)
            return loss
        closures.append(loss_fn)
    return rewards_out, closures


def rollout_gradients(model: nn.Module, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[Prompt],
                      reward_fns: Sequence[RewardFn] | RewardFn, spec: ScoringSpec, *, objective: str = "grpo",
                      skip_saturated: bool = False, seed: int | None = None, num_samples: int | None = None,
                      sanitize: bool = True, with_rewards: bool = False, rollout: RolloutBatch | None = None,
                      ref_model: nn.Module | None = None,
                      ) -> list[Tensor | None] | tuple[list[Tensor | None], list[list[float]]]:
    if objective not in ("grpo", "pg"):
        raise ValueError(f"objective must be 'grpo' or 'pg', got {objective!r}")
    B = len(prompts)
    G = int(num_samples or (spec.g_train if objective == "grpo" else spec.g_test))
    device = _device_of(model)
    model.eval()
    model.zero_grad(set_to_none=True)
    if rollout is None:
        _, prompt_ids, prompt_mask = tokenize_prompts_batch(tokenizer, prompts, device)
        rollout = generate_rollouts(model, tokenizer, prompt_ids, prompt_mask, backend=spec.backend,
                                    num_samples=G, max_new_tokens=spec.max_new_tokens,
                                    temperature=spec.temperature, top_p=spec.top_p, seed=seed,
                                    server_host=spec.vllm_server_host, server_port=spec.vllm_server_port)
    if rollout.num_prompts != B or rollout.num_samples != G:
        raise RuntimeError("rollout layout does not match prompts x num_samples")
    rewards_out, closures = rollout_losses(model, tokenizer, prompts, reward_fns, spec, rollout, objective=objective,
                                           skip_saturated=skip_saturated, ref_model=ref_model)
    grads: list[Tensor | None] = []
    for b, fn in enumerate(closures):
        if fn is None:
            grads.append(None)
            continue
        model.zero_grad(set_to_none=True)
        loss = fn()
        g = flat_grad(model, loss)
        grads.append(sanitize_grad(g, f"prompt {b} {objective}") if sanitize else g)
        del loss
    if with_rewards:
        return grads, rewards_out
    return grads


def gold_target_ids(tokenizer: PreTrainedTokenizerBase, gold: str | float | None, *, box: bool = True, device: torch.device | str = "cpu") -> Tensor:
    gold = str(gold) if gold is not None else ""
    target = f"\\boxed{{{gold}}}" if box else gold
    ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    eos = tokenizer.eos_token_id
    if eos is not None:
        ids = torch.cat([ids, torch.full((1, 1), int(eos), dtype=ids.dtype, device=ids.device)], dim=1)
    return ids


def gold_gradients(model: nn.Module, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[Prompt],
                   golds: Sequence[str | float | None], *, box: bool = True, micro_batch: int = 1,
                   sanitize: bool = False) -> list[Tensor]:
    if len(prompts) != len(golds):
        raise ValueError("prompts and golds must have the same length")
    device = _device_of(model)
    model.eval()
    model.zero_grad(set_to_none=True)
    _, prompt_ids, prompt_mask = tokenize_prompts_batch(tokenizer, prompts, device)
    grads = []
    for b in range(len(prompts)):
        comp_ids = gold_target_ids(tokenizer, golds[b], box=box, device=device)
        comp_mask = torch.ones_like(comp_ids)
        logps = per_token_logps(model, prompt_ids[b:b + 1], prompt_mask[b:b + 1], comp_ids, comp_mask,
                                micro_batch=micro_batch)
        nll = -(logps * comp_mask).sum() / comp_mask.sum().clamp(min=1)
        model.zero_grad(set_to_none=True)
        g = flat_grad(model, nll)
        grads.append(sanitize_grad(g, f"gold prompt {b}") if sanitize else g)
        del logps, nll
    return grads


__all__ = [
    "per_token_logps", "ref_logps", "grpo_loss_per_prompt", "expected_reward_loss", "flat_grad",
    "sanitize_grad", "evaluate_rewards", "rollout_gradients", "gold_target_ids", "gold_gradients",
]
