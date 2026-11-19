from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

from ..preconditioner import adam_preconditioner_from_optimizer
from ..utils import grad_vector, n_trainable
from .data import ToyExample
from .grpo import clone, grpo_loss, pg_loss
from .model import ALL_SEQUENCES, ToyAutoregressiveMLP

Tensor = torch.Tensor
FitSample = tuple[float, Callable[[], Tensor]]


def train_gradients(model: ToyAutoregressiveMLP, examples: Sequence[ToyExample], *, ref_model: nn.Module,
                    beta: float) -> Tensor:
    model.eval()
    old_model = clone(model)
    out = torch.empty(len(examples), n_trainable(model), dtype=torch.float32)
    for i, ex in enumerate(examples):
        out[i] = grad_vector(model, grpo_loss(model, ex, old_model=old_model, ref_model=ref_model, beta=beta)).cpu()
    return out


def test_gradients(model: ToyAutoregressiveMLP, examples: Sequence[ToyExample]) -> Tensor:
    model.eval()
    out = torch.empty(len(examples), n_trainable(model), dtype=torch.float32)
    for i, ex in enumerate(examples):
        out[i] = grad_vector(model, pg_loss(model, ex)).cpu()
    return out


def fisher_cache(model: ToyAutoregressiveMLP, examples: Sequence[ToyExample]) -> tuple[list[Tensor], list[Tensor]]:
    model.eval()
    seqs = ALL_SEQUENCES.to(model.device)
    n_seq = int(seqs.shape[0])
    grad_cache: list[Tensor] = []
    prob_cache: list[Tensor] = []
    for ex in examples:
        log_probs = model.sequence_log_probs(ex.z_tensor(model.device), seqs)
        G = torch.empty(n_seq, n_trainable(model), dtype=torch.float32)
        for y in range(n_seq):
            G[y] = grad_vector(model, log_probs[y], retain_graph=(y < n_seq - 1)).cpu()
        grad_cache.append(G)
        prob_cache.append(log_probs.detach().exp().to(dtype=torch.float32).cpu())
    return grad_cache, prob_cache


def ekfac_samples(model: ToyAutoregressiveMLP, examples: Sequence[ToyExample]) -> list[FitSample]:
    seqs = ALL_SEQUENCES.to(model.device)
    n = len(examples)
    samples: list[FitSample] = []
    for ex in examples:
        z = ex.z_tensor(model.device)
        with torch.no_grad():
            p = model.sequence_log_probs(z, seqs).exp()
        for y in range(int(seqs.shape[0])):
            w = float(p[y]) / n
            if w > 0:
                samples.append((w, lambda z=z, y=y: model.sequence_log_probs(z, seqs[y:y + 1])[0]))
    return samples


def adam_preconditioner(optimizer: torch.optim.Optimizer, model: nn.Module) -> Tensor:
    return adam_preconditioner_from_optimizer(optimizer, model).cpu()
