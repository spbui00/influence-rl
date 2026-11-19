"""Monte-Carlo policy Fisher batch as in the legacy scorer: the FIRST fisher_examples pool prompts (pool
order, not a random draw) x fisher_g completions sampled at the ScoringSpec temperature with seed
spec.seed + i, prompt truncated to its last fisher_max_tokens tokens and response to its first."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import torch

from ..attribution import DampedFisherEKFAC, EKFACFactors, PolicyFisherFVP, fit_ekfac
from ..attribution.fvp import FisherRow
from .config import ScoringSpec
from .generation import Prompt, generate_rollouts, tokenize_prompt
from .gradients import per_token_logps

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

Tensor = torch.Tensor


@dataclass
class FisherSample:
    prompt_ids: Tensor
    prompt_mask: Tensor
    response_ids: Tensor
    response_mask: Tensor

    def sequence(self) -> FisherRow:
        p = self.prompt_ids[0][self.prompt_mask[0].bool()]
        r = self.response_ids[0][self.response_mask[0].bool()]
        return FisherRow(torch.cat([p, r]).cpu(), int(p.numel()))


@contextmanager
def checkpointing_disabled(model: nn.Module) -> Iterator[None]:
    was = bool(getattr(model, "is_gradient_checkpointing", False))
    if was and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if was and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()


@torch.no_grad()
def fisher_batch(model: nn.Module, tokenizer: PreTrainedTokenizerBase, pool: Sequence[dict | Prompt], spec: ScoringSpec) -> list[FisherSample]:
    device = next(model.parameters()).device
    n = min(int(spec.fisher_examples), len(pool))
    max_tok = int(spec.fisher_max_tokens)
    rows: list[FisherSample] = []
    for i in range(n):
        item = pool[i]
        prompt = item["prompt"] if isinstance(item, dict) else item
        _, p_ids, p_mask = tokenize_prompt(tokenizer, prompt, device)
        rollout = generate_rollouts(model, tokenizer, p_ids, p_mask, backend=spec.backend,
                                    num_samples=int(spec.fisher_g), max_new_tokens=spec.max_new_tokens,
                                    temperature=spec.temperature, top_p=spec.top_p, seed=spec.seed + i,
                                    server_host=spec.vllm_server_host, server_port=spec.vllm_server_port)
        p_ids, p_mask = p_ids[:, -max_tok:], p_mask[:, -max_tok:]
        r_ids, r_mask = rollout.token_ids[:, :max_tok], rollout.response_mask[:, :max_tok]
        for u in range(r_ids.shape[0]):
            rows.append(FisherSample(p_ids, p_mask, r_ids[u:u + 1].to(device), r_mask[u:u + 1].to(device)))
    return rows


def ekfac_samples(model: nn.Module, rows: Sequence[FisherSample]) -> list[tuple[float, Callable[[], torch.Tensor]]]:
    w = 1.0 / max(len(rows), 1)

    def make(row: FisherSample):
        def closure() -> Tensor:
            lp = per_token_logps(model, row.prompt_ids, row.prompt_mask, row.response_ids, row.response_mask,
                                 checkpointing=False)
            return (lp * row.response_mask.to(lp.dtype)).sum()
        return closure

    return [(w, make(r)) for r in rows]


def fit_fisher_ekfac(model: nn.Module, rows: Sequence[FisherSample], *, dtype: torch.dtype = torch.float32,
                     progress: bool = True) -> EKFACFactors:
    model.eval()
    with checkpointing_disabled(model):
        return fit_ekfac(model, ekfac_samples(model, rows), dtype=dtype, progress=progress)


def ekfac_estimator(model: nn.Module, rows: Sequence[FisherSample], spec: ScoringSpec, *, progress: bool = True) -> DampedFisherEKFAC:
    factors = fit_fisher_ekfac(model, rows, dtype=torch.float32, progress=progress)
    for blk in factors.blocks:  # ponytail: factors are ~30 GB for the 8960-wide LoRA-A inputs; keep them on CPU, solve there
        blk.q_a, blk.q_s, blk.lam = blk.q_a.cpu(), blk.q_s.cpu(), blk.lam.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rel = spec.ekfac_rel_damp if spec.ekfac_rel_damp and spec.ekfac_rel_damp > 0 else None
    return DampedFisherEKFAC(factors, lambda_damp=spec.lambda_damp, rel_damp=rel)


def policy_fisher_fvp(model: nn.Module, rows: Sequence[FisherSample], *, normalize: bool = True, power_iters: int = 15,
                      seed: int = 0, micro_batch: int = 1) -> PolicyFisherFVP:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return PolicyFisherFVP(model, [r.sequence() for r in rows], micro_batch=micro_batch, normalize=normalize,
                           power_iters=power_iters, seed=seed)


__all__ = ["FisherSample", "checkpointing_disabled", "fisher_batch", "ekfac_samples", "fit_fisher_ekfac",
           "ekfac_estimator", "policy_fisher_fvp"]
