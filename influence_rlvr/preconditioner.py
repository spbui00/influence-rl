"""Adam diagonal  P = 1/(sqrt(v_hat)+eps),  v_hat = exp_avg_sq/(1-beta2^step)  in trainable_params order.
Weight decay and the first moment are ignored; a trainable param without optimizer state gets
P = 1 (v_hat = 0). optimizer.pt alignment is positional and shape-checked, with HF's
decay/no-decay regroup as the only fallback; anything else raises."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .utils import trainable_named_params

Tensor = torch.Tensor


def _is_no_decay_name(name: str) -> bool:
    lname = name.lower()
    return lname.endswith(".bias") or "norm" in lname


def _ordered_states(optimizer_state_dict: dict) -> list[dict]:
    state = optimizer_state_dict.get("state", {})
    return [state[i] for i in sorted(state.keys(), key=int)]


def _hyperparams(group: dict, *, beta2_fallback: float = 0.999, eps_fallback: float = 1e-8) -> tuple[float, float]:
    beta2 = float(group.get("betas", (0.9, beta2_fallback))[1])
    eps = float(group.get("eps", eps_fallback))
    return beta2, eps


def _step_count(step) -> float:
    if step is None:
        return 0.0
    return float(step.item()) if torch.is_tensor(step) else float(step)


def _vhat(state: dict, beta2: float, device: torch.device | str | None) -> Tensor:
    v = state["exp_avg_sq"].detach().to(device=device, dtype=torch.float32)
    t = _step_count(state.get("step"))
    bc2 = 1.0 - beta2 ** t if t > 0 else 1.0
    return (v / bc2).reshape(-1)


def _aligned_states(named_params: Sequence[tuple[str, nn.Parameter]], optimizer_state_dict: dict) -> list[dict]:
    params = [p for _, p in named_params]
    states = _ordered_states(optimizer_state_dict)
    if len(states) != len(params):
        raise ValueError(
            f"optimizer state has {len(states)} params but the model has {len(params)} trainable params"
        )
    if states and "exp_avg_sq" not in states[0]:
        raise ValueError("optimizer state has no 'exp_avg_sq'; an Adam/AdamW optimizer is required")

    def shapes_match(order: Sequence[nn.Parameter]) -> bool:
        return all(tuple(s["exp_avg_sq"].shape) == tuple(p.shape) for s, p in zip(states, order))

    if shapes_match(params):
        return states
    regrouped = ([p for n, p in named_params if not _is_no_decay_name(n)]
                 + [p for n, p in named_params if _is_no_decay_name(n)])
    if not shapes_match(regrouped):
        raise ValueError("could not align optimizer state to model parameters by shape")
    state_for = {id(p): s for s, p in zip(states, regrouped)}
    return [state_for[id(p)] for p in params]


def adam_vhat_from_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module) -> Tensor:
    beta2, _ = _hyperparams(optimizer.param_groups[0])
    parts = []
    for _, p in trainable_named_params(model):
        st = optimizer.state.get(p, {})
        if st.get("exp_avg_sq") is None:
            parts.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
        else:
            parts.append(_vhat(st, beta2, p.device))
    return torch.cat(parts)


def adam_preconditioner_from_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module,
                                       eps: float | None = None) -> Tensor:
    beta2, opt_eps = _hyperparams(optimizer.param_groups[0])
    eps = float(eps) if eps is not None and eps > 0 else opt_eps
    parts = []
    for _, p in trainable_named_params(model):
        st = optimizer.state.get(p, {})
        if st.get("exp_avg_sq") is None:
            parts.append(torch.ones(p.numel(), dtype=torch.float32, device=p.device))
        else:
            parts.append(1.0 / (_vhat(st, beta2, p.device).sqrt() + eps))
    return torch.cat(parts)


def adam_vhat_from_state(named_params: Sequence[tuple[str, nn.Parameter]], optimizer_state_dict: dict,
                         *, device: torch.device | str | None = None) -> Tensor:
    states = _aligned_states(named_params, optimizer_state_dict)
    groups = optimizer_state_dict.get("param_groups", []) or [{}]
    beta2, _ = _hyperparams(groups[0])
    return torch.cat([_vhat(s, beta2, device) for s in states])


def adam_preconditioner_from_state(
    named_params: Sequence[tuple[str, nn.Parameter]],
    optimizer_state_dict: dict,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    eps_override: float | None = None,
) -> Tensor:
    states = _aligned_states(named_params, optimizer_state_dict)
    groups = optimizer_state_dict.get("param_groups", []) or [{}]
    beta2, eps = _hyperparams(groups[0])
    if eps_override is not None and eps_override > 0:
        eps = float(eps_override)
    return torch.cat([(1.0 / (_vhat(s, beta2, device).sqrt() + eps)).to(dtype) for s in states])


def _load_optimizer_state(checkpoint_dir: str | Path, allow_missing: bool) -> dict | None:
    path = Path(checkpoint_dir) / "optimizer.pt"
    if not path.is_file():
        if allow_missing:
            return None
        raise FileNotFoundError(f"no optimizer.pt in {checkpoint_dir}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_adam_preconditioner(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float | None = None,
    allow_missing: bool = False,
) -> Tensor | None:
    sd = _load_optimizer_state(checkpoint_dir, allow_missing)
    if sd is None:
        return None
    return adam_preconditioner_from_state(trainable_named_params(model), sd, device=device, dtype=dtype,
                                          eps_override=eps)


def load_adam_vhat(model: nn.Module, checkpoint_dir: str | Path, *,
                   device: torch.device | str | None = None) -> Tensor:
    sd = _load_optimizer_state(checkpoint_dir, allow_missing=False)
    return adam_vhat_from_state(trainable_named_params(model), sd, device=device)


def load_adam_state(model: nn.Module, checkpoint_dir: str | Path, *,
                    device: torch.device | str | None = None) -> dict:
    sd = _load_optimizer_state(checkpoint_dir, allow_missing=False)
    states = _aligned_states(trainable_named_params(model), sd)
    group = sd["param_groups"][0]

    def cat(key: str) -> Tensor:
        return torch.cat([s[key].detach().to(device=device, dtype=torch.float32).reshape(-1) for s in states])

    return {"m": cat("exp_avg"), "v": cat("exp_avg_sq"), "t": _step_count(states[0].get("step")),
            "lr": float(group["lr"]), "betas": tuple(float(b) for b in group["betas"]), "eps": float(group["eps"]),
            "weight_decay": float(group.get("weight_decay", 0.0))}


__all__ = [
    "adam_preconditioner_from_optimizer", "adam_vhat_from_optimizer",
    "adam_preconditioner_from_state", "adam_vhat_from_state",
    "load_adam_preconditioner", "load_adam_vhat", "load_adam_state",
]
