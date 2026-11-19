"""Exhaustive G=4 GRPO with UNIFORM 1/G weights over the four outputs (the paper text says
probability-weighted; the numbers come from this uniform form). Both losses are LOSS gradients:
grpo_loss = -(1/4) sum_y (1/2) sum_t [min(rho A_y, clip(rho) A_y) - beta kl_t], pg_loss = -(1/4) sum_y r(y) log pi(y|z).
D = 2196 parameters at the paper size (paper text: 2,194)."""
from __future__ import annotations

import copy
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .data import ToyExample
from .model import ALL_SEQUENCES, ToyAutoregressiveMLP

Tensor = torch.Tensor
StepHook = Callable[[int, nn.Module, torch.optim.Optimizer], None]


def rewards_for(sequences: Tensor, target: Tensor) -> Tensor:
    return (sequences == target.unsqueeze(0)).all(dim=1).to(dtype=torch.float32)


def exact_expected_reward(model: ToyAutoregressiveMLP, example: ToyExample) -> float:
    with torch.no_grad():
        seqs, probs = model.exact_sequence_distribution(example.z_tensor(model.device))
        r = rewards_for(seqs, example.target_tensor(model.device))
        return float((probs * r).sum().cpu())


def expected_rewards(model: ToyAutoregressiveMLP, examples: Sequence[ToyExample]) -> np.ndarray:
    return np.array([exact_expected_reward(model, e) for e in examples], dtype=np.float64)


def grpo_loss(model: ToyAutoregressiveMLP, example: ToyExample, *, old_model: nn.Module | None,
              ref_model: nn.Module | None, beta: float, epsilon: float = 0.2,
              advantage_eps: float = 1e-4) -> Tensor:
    seqs = ALL_SEQUENCES.to(model.device)
    z = example.z_tensor(model.device)
    lp = model.per_token_log_probs(z, seqs)
    with torch.no_grad():
        old_lp = lp.detach() if old_model is None else old_model.per_token_log_probs(z, seqs)
        ref_lp = None
        if beta != 0.0:
            if ref_model is None:
                raise ValueError("beta != 0 requires ref_model")
            ref_lp = ref_model.per_token_log_probs(z, seqs)
    r = rewards_for(seqs, example.target_tensor(model.device))
    std_r = r.std(unbiased=False).clamp_min(0.0)
    adv = ((r - r.mean()) / (std_r + advantage_eps)).unsqueeze(1)
    log_ratio = torch.clamp(torch.nan_to_num(lp - old_lp, nan=0.0), -20.0, 20.0)
    ratios = torch.exp(log_ratio)
    clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
    obj = torch.min(ratios * adv, clipped * adv)
    kl = torch.zeros_like(lp)
    if ref_lp is not None:
        d = ref_lp - lp
        kl = torch.exp(d) - d - 1.0
    mask = torch.ones_like(lp)
    per_seq = ((obj - beta * kl) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return -per_seq.mean()


def pg_loss(model: ToyAutoregressiveMLP, example: ToyExample) -> Tensor:
    seqs = ALL_SEQUENCES.to(model.device)
    lp = model.per_token_log_probs(example.z_tensor(model.device), seqs)
    seq_lp = (lp * torch.ones_like(lp)).sum(dim=1)
    r = rewards_for(seqs, example.target_tensor(model.device))
    return -(r.detach() * seq_lp).mean()


def clone(model: nn.Module) -> nn.Module:
    c = copy.deepcopy(model)
    c.eval()
    return c


def train_grpo(examples: Sequence[ToyExample], steps: int, *, ref_model: ToyAutoregressiveMLP, lr: float,
               beta: float, seed: int = 0, start_state: dict | None = None,
               start_opt_state: dict | None = None, order: np.ndarray | None = None,
               on_step: StepHook | None = None) -> tuple[ToyAutoregressiveMLP, torch.optim.Adam]:
    torch.manual_seed(seed)
    model = ToyAutoregressiveMLP(input_dim=ref_model.input_dim, hidden_dim=ref_model.hidden_dim)
    model.load_state_dict(start_state if start_state is not None else ref_model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if start_opt_state is not None:
        optimizer.load_state_dict(copy.deepcopy(start_opt_state))
        for g in optimizer.param_groups:
            g["lr"] = lr
    n = len(examples)
    for step in range(1, steps + 1):
        pos = (step - 1) % n
        example = examples[int(order[pos]) if order is not None else pos]
        old_model = clone(model)
        optimizer.zero_grad()
        loss = grpo_loss(model, example, old_model=old_model, ref_model=ref_model, beta=beta)
        loss.backward()
        optimizer.step()
        if on_step is not None:
            on_step(step, model, optimizer)
    return model, optimizer
