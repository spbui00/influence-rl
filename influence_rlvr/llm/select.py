"""Online selection GRPO: each step samples rollouts for `candidates` prompts, scores every prompt, keeps a subset by a
selector and takes one AdamW step on the kept prompts' mean collapsed gradient (eq. 12, ratio = 1, KL to the base).
Influence score = g_i · M g_guide with g_guide the baseline PG gradient of fresh guide rollouts and M from the estimator
family (dot: I, adam: live-optimizer diagonal, ekfac: damped Fisher fitted on the batch's own rollouts); positive = helpful."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from ..attribution import Estimator, make_estimator
from ..preconditioner import adam_preconditioner_from_optimizer
from ..utils import trainable_params, unflatten
from .config import ScoringSpec
from .fisher import FisherSample, ekfac_estimator
from .generation import RolloutBatch, generate_rollouts, rollout_to_completions, tokenize_prompts_batch
from .gradients import gold_gradients, rollout_gradients, rollout_losses, sanitize_grad
from .rewards import completion_text, extract_final_answer, row_reward_fn

Tensor = torch.Tensor


def memlog(tag: str) -> None:
    if os.environ.get("SELECT_MEMLOG") and torch.cuda.is_available():
        print(f"[mem] {tag}: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


SELECTORS = ("none", "random", "liveness", "disagreement", "influence", "anti", "oracle")
METHODS = ("dot", "adam", "ekfac")


@dataclass
class StepBatch:
    rows: list[dict]
    grads: Tensor | None
    rewards: list[list[float]]
    answers: list[list[str | None]]
    tokens: int
    timing: dict = field(default_factory=dict)
    fisher: list[FisherSample] = field(default_factory=list)
    closures: list = field(default_factory=list)
    rollouts: list = field(default_factory=list)  # per-prompt RolloutBatch on CPU when keep_rollouts (P2 replay)

    @property
    def live(self) -> np.ndarray:
        return np.asarray([np.std(r) > 1e-4 for r in self.rewards])

    @property
    def reward_std(self) -> np.ndarray:
        return np.asarray([float(np.std(r)) for r in self.rewards])

    @property
    def disagreement(self) -> np.ndarray:
        out = []
        for r, a in zip(self.rewards, self.answers):
            rewarded = {x for x, ri in zip(a, r) if ri > 0 and x is not None}
            out.append(len(rewarded))
        return np.asarray(out, dtype=np.float64)


def sample_batch(model: nn.Module, tok, spec: ScoringSpec, rows: Sequence[dict], seed: int, device, *, stream: bool = False,
                 ref_model: nn.Module | None = None, keep_rollouts: bool = False) -> StepBatch:
    D = sum(p.numel() for p in trainable_params(model))
    grads = None if stream else torch.zeros(len(rows), D, dtype=torch.float32, device=device)
    rewards, answers, fisher, closures, rollouts, tokens, t_gen, t_grad = [], [], [], [], [], 0, 0.0, 0.0
    max_tok = int(spec.fisher_max_tokens)
    for start in range(0, len(rows), spec.score_batch):
        chunk = rows[start:start + spec.score_batch]
        t0 = time.time()
        _, p_ids, p_mask = tokenize_prompts_batch(tok, [r["prompt"] for r in chunk], device)
        rb = generate_rollouts(model, tok, p_ids, p_mask, backend=spec.backend, num_samples=spec.g_train,
                               max_new_tokens=spec.max_new_tokens, temperature=spec.temperature, top_p=spec.top_p,
                               seed=seed + start, server_host=spec.vllm_server_host, server_port=spec.vllm_server_port)
        tokens += int(rb.response_mask.sum())
        t1 = time.time()
        prompts, fns = [r["prompt"] for r in chunk], [row_reward_fn(r) for r in chunk]
        if stream:
            rw, cl = rollout_losses(model, tok, prompts, fns, spec, rb, objective="grpo", skip_saturated=True, ref_model=ref_model)
            closures += cl
        else:
            g, rw = rollout_gradients(model, tok, prompts, fns, spec, objective="grpo", skip_saturated=True, rollout=rb,
                                      num_samples=spec.g_train, with_rewards=True, ref_model=ref_model)
            for j, gi in enumerate(g):
                if gi is not None:
                    grads[start + j] = gi.to(device)
        rewards += rw
        for b in range(len(chunk)):
            sl = rb.prompt_slice(b)
            answers.append([extract_final_answer(completion_text(c)) for c in rollout_to_completions(sl)])
            if keep_rollouts:
                rollouts.append(RolloutBatch(sl.texts, sl.token_ids.cpu(), sl.response_mask.cpu(), 1, sl.num_samples))
            if start + b < spec.fisher_examples:
                fisher += [FisherSample(p_ids[b:b + 1, -max_tok:], p_mask[b:b + 1, -max_tok:], sl.token_ids[u:u + 1, :max_tok],
                                        sl.response_mask[u:u + 1, :max_tok]) for u in range(min(spec.fisher_g, sl.token_ids.shape[0]))]
        t_gen += t1 - t0
        t_grad += time.time() - t1
    return StepBatch(list(rows), grads, rewards, answers, tokens, {"generate": t_gen, "gradients": t_grad}, fisher, closures, rollouts)


def gold_target_scores(model: nn.Module, tok, spec: ScoringSpec, optimizer: torch.optim.Optimizer, batch: StepBatch,
                       targets: Sequence[dict]) -> dict[str, np.ndarray]:
    """Per-prompt x per-target scores g_i . M grad(gold log-prob of target) at the current parameters, M = I and the live
    Adam diagonal (the next_step.py 'rollout-gold' scores, recorded along a run for retrospective attribution)."""
    device = batch.grads.device
    gt = torch.stack([g.to(device) for g in gold_gradients(model, tok, [r["prompt"] for r in targets], [r["gold"] for r in targets],
                                                           micro_batch=spec.logps_micro_batch)])
    P = adam_preconditioner_from_optimizer(optimizer, model).to(device)
    return {"dot": (batch.grads @ gt.T).cpu().numpy(), "adam": (batch.grads @ (gt * P).T).cpu().numpy()}


def grad_list(model: nn.Module, loss: Tensor) -> list[Tensor]:
    params = trainable_params(model)
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p, dtype=torch.float32) if g is None else sanitize_grad(g.detach().float()) for g, p in zip(gs, params)]


def dot_list(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def adam_preconditioner_list(optimizer: torch.optim.Optimizer, params: Sequence[nn.Parameter]) -> list[Tensor]:
    g = optimizer.param_groups[0]
    b2, eps = g["betas"][1], g["eps"]
    out = []
    for p in params:
        st = optimizer.state.get(p)
        if not st or "exp_avg_sq" not in st:
            out.append(torch.ones_like(p, dtype=torch.float32))
            continue
        t = float(st["step"])
        out.append(1.0 / ((st["exp_avg_sq"].float() / (1 - b2 ** t)).sqrt() + eps))
    return out


def stream_scores(model: nn.Module, batch: StepBatch, h: Sequence[Tensor] | None) -> tuple[np.ndarray | None, list[Tensor], int]:
    params = trainable_params(model)
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    scores = None if h is None else np.zeros(len(batch.rows))
    n = 0
    for b, fn in enumerate(batch.closures):
        if fn is None:
            continue
        model.zero_grad(set_to_none=True)
        g = grad_list(model, fn())
        if h is not None:
            scores[b] = dot_list(g, h)
        for a, x in zip(acc, g):
            a.add_(x)
        n += 1
        del g
    return scores, acc, n


def stream_mean(model: nn.Module, batch: StepBatch, acc: Sequence[Tensor], keep: np.ndarray, scored_all: bool) -> list[Tensor]:
    idx = np.flatnonzero(~keep) if scored_all else np.flatnonzero(keep)
    sign = -1.0 if scored_all else 1.0
    if not scored_all:
        acc = [torch.zeros_like(a) for a in acc]
    for b in idx:
        fn = batch.closures[b]
        if fn is None:
            continue
        model.zero_grad(set_to_none=True)
        g = grad_list(model, fn())
        for a, x in zip(acc, g):
            a.add_(x, alpha=sign)
        del g
    return [a / max(int(keep.sum()), 1) for a in acc]


def make_step_estimator(method: str, model: nn.Module, optimizer: torch.optim.Optimizer, spec: ScoringSpec, batch: StepBatch,
                        step: int, state: dict, ekfac_every: int, device) -> Estimator:
    if method == "dot":
        return make_estimator("dot")
    if method == "adam":
        return make_estimator("adam", preconditioner=adam_preconditioner_from_optimizer(optimizer, model).to(device))
    if method != "ekfac":
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if "ekfac" not in state or (step - state["ekfac_step"]) >= ekfac_every:
        est = ekfac_estimator(model, batch.fisher, spec, progress=False)  # factors land on CPU (fisher.ekfac_estimator)
        state["ekfac"], state["ekfac_step"] = est, step
    return state["ekfac"]


def guide_vector(model: nn.Module, tok, spec: ScoringSpec, guide_rows: Sequence[dict], est: Estimator, seed: int,
                 device) -> tuple[Tensor, dict]:
    t0 = time.time()
    g = [x for x in rollout_gradients(model, tok, [r["prompt"] for r in guide_rows], [row_reward_fn(r) for r in guide_rows],
                                      spec, objective="pg", skip_saturated=False, seed=seed, num_samples=spec.g_test)
         if x is not None]
    g_test = torch.stack([x.to(device) for x in g]).mean(0)
    return est.solve(g_test).to(device), {"guide": time.time() - t0}


def guide_vector_stream(model: nn.Module, tok, spec: ScoringSpec, guide_rows: Sequence[dict], method: str,
                        optimizer: torch.optim.Optimizer, seed: int, device) -> tuple[list[Tensor], dict]:
    if method not in ("dot", "adam"):
        raise ValueError(f"method {method!r} is not available in streaming (full-parameter) mode")
    t0 = time.time()
    params = trainable_params(model)
    closures = []
    for start in range(0, len(guide_rows), spec.score_batch):
        chunk = guide_rows[start:start + spec.score_batch]
        _, p_ids, p_mask = tokenize_prompts_batch(tok, [r["prompt"] for r in chunk], device)
        rb = generate_rollouts(model, tok, p_ids, p_mask, backend=spec.backend, num_samples=spec.g_test, max_new_tokens=spec.max_new_tokens,
                               temperature=spec.temperature, top_p=spec.top_p, seed=seed + start, server_host=spec.vllm_server_host,
                               server_port=spec.vllm_server_port)
        closures += rollout_losses(model, tok, [r["prompt"] for r in chunk], [row_reward_fn(r) for r in chunk], spec, rb,
                                   objective="pg", skip_saturated=False)[1]
    memlog("guide rollouts done")
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    for i, fn in enumerate(closures):
        model.zero_grad(set_to_none=True)
        for a, x in zip(acc, grad_list(model, fn())):
            a.add_(x, alpha=1.0 / len(closures))
        if i == 0:
            memlog("after first guide gradient")
    if method == "adam":
        for a, pre in zip(acc, adam_preconditioner_list(optimizer, params)):
            a.mul_(pre)
    return acc, {"guide": time.time() - t0}


def select(selector: str, batch: StepBatch, scores: np.ndarray | None, drop: int, rng: np.random.Generator,
           is_bad: np.ndarray | None = None) -> np.ndarray:
    n = len(batch.rows)
    keep = np.ones(n, dtype=bool)
    if selector == "none" or drop <= 0:
        return keep
    if selector == "random":
        keep[rng.choice(n, drop, replace=False)] = False
    elif selector == "liveness":
        keep[np.argsort(batch.reward_std, kind="stable")[:drop]] = False
    elif selector == "disagreement":
        keep[np.argsort(-batch.disagreement, kind="stable")[:drop]] = False
    elif selector == "influence":
        keep[np.argsort(scores, kind="stable")[:drop]] = False
    elif selector == "anti":
        keep[np.argsort(-scores, kind="stable")[:drop]] = False
    elif selector == "oracle":
        keep[np.flatnonzero(is_bad)[:drop]] = False
    else:
        raise ValueError(f"unknown selector {selector!r}; expected one of {SELECTORS}")
    return keep


def apply_update(model: nn.Module, optimizer: torch.optim.Optimizer, grad: Tensor | Sequence[Tensor], max_grad_norm: float = 1.0) -> float:
    params = trainable_params(model)
    for p, g in zip(params, unflatten(grad, params) if torch.is_tensor(grad) else grad):
        p.grad = g.to(p.dtype)
    norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm)) if max_grad_norm > 0 else float(
        torch.sqrt(sum(p.grad.float().pow(2).sum() for p in params)))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def run_step(model: nn.Module, tok, spec: ScoringSpec, optimizer: torch.optim.Optimizer, rows: Sequence[dict], *,
             selector: str, drop: int, guide_rows: Sequence[dict] | None, seed: int, rng: np.random.Generator,
             bad_fn: Callable[[dict], bool] | None = None, max_grad_norm: float = 1.0, method: str = "adam",
             step: int = 0, state: dict | None = None, ekfac_every: int = 10, stream: bool = False,
             ref_model: nn.Module | None = None, score_targets: Sequence[dict] | None = None, keep_rollouts: bool = False) -> dict:
    device = next(model.parameters()).device
    model.eval()
    memlog("step start")
    batch = sample_batch(model, tok, spec, rows, seed, device, stream=stream, ref_model=ref_model, keep_rollouts=keep_rollouts)
    scores_gold = None
    if score_targets is not None:
        if stream:
            raise ValueError("gold-target scores need the stored gradient matrix (no --stream/--full)")
        t0 = time.time()
        scores_gold = gold_target_scores(model, tok, spec, optimizer, batch, score_targets)
        batch.timing["score_gold"] = time.time() - t0
    memlog("batch sampled")
    scores = None
    timing = dict(batch.timing)
    is_bad = np.asarray([bool(bad_fn(r)) for r in rows]) if bad_fn else None
    if stream:
        h = None
        if guide_rows is not None:
            h, t = guide_vector_stream(model, tok, spec, guide_rows, method, optimizer, seed + 1, device)
            timing.update(t)
        t0 = time.time()
        scored_all = h is not None
        if scored_all:
            scores, acc, _ = stream_scores(model, batch, h)
            del h
        else:
            acc = [torch.zeros_like(p, dtype=torch.float32) for p in trainable_params(model)]
        keep = select(selector, batch, scores, drop, rng, is_bad)
        grad = stream_mean(model, batch, acc, keep, scored_all)
        del acc
        timing["gradients"] += time.time() - t0
    else:
        if guide_rows is not None:
            t0 = time.time()
            est = make_step_estimator(method, model, optimizer, spec, batch, step, state if state is not None else {}, ekfac_every, device)
            timing["estimator"] = time.time() - t0
            h, t = guide_vector(model, tok, spec, guide_rows, est, seed + 1, device)
            scores = (batch.grads @ h).cpu().numpy()
            timing.update(t)
        keep = select(selector, batch, scores, drop, rng, is_bad)
        grad = batch.grads[torch.as_tensor(keep, device=device)].mean(0)
    t0 = time.time()
    norm = apply_update(model, optimizer, grad, max_grad_norm)
    timing["update"] = time.time() - t0
    live = batch.live
    return {"keep": keep, "scores": scores, "grad_norm": norm, "tokens": batch.tokens, "timing": timing,
            "scores_gold": scores_gold, "rollouts": batch.rollouts, "rewards": batch.rewards,
            "n_live": int(live.sum()), "n_kept_live": int((live & keep).sum()),
            "mean_reward": float(np.mean([np.mean(r) for r in batch.rewards])),
            "n_bad": None if is_bad is None else int(is_bad.sum()),
            "n_bad_dropped": None if is_bad is None else int((is_bad & ~keep).sum()),
            "n_bad_live_dropped": None if is_bad is None else int((is_bad & live & ~keep).sum()),
            "n_bad_live": None if is_bad is None else int((is_bad & live).sum())}
