"""Canonical flat parameter order: every gradient, preconditioner and EK-FAC block in this
package is laid out in  [p for p in model.parameters() if p.requires_grad]  order."""
from __future__ import annotations

import random
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn

from .files import read_json, read_jsonl, write_json, write_jsonl


def trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def trainable_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in trainable_params(model))


def flat_param_slices(model: nn.Module) -> tuple[dict[int, slice], int]:
    slices: dict[int, slice] = {}
    off = 0
    for p in trainable_params(model):
        slices[id(p)] = slice(off, off + p.numel())
        off += p.numel()
    return slices, off


def flatten(tensors: Iterable[torch.Tensor | None], like: Iterable[torch.Tensor],
            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    out = []
    for g, p in zip(tensors, like):
        out.append(torch.zeros_like(p).flatten() if g is None else g.detach().flatten())
    return torch.cat(out).to(dtype=dtype)


def unflatten(flat: torch.Tensor, like: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    out, off = [], 0
    for p in like:
        n = p.numel()
        out.append(flat[off:off + n].view_as(p))
        off += n
    return out


def grad_vector(model: nn.Module, scalar: torch.Tensor, *, retain_graph: bool = False,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    params = trainable_params(model)
    grads = torch.autograd.grad(scalar, params, retain_graph=retain_graph, allow_unused=True)
    return flatten(grads, params, dtype=dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def detect_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def chunked(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


__all__ = ["trainable_params", "trainable_named_params", "n_trainable", "flat_param_slices", "flatten", "unflatten",
           "grad_vector", "seed_everything", "detect_device", "read_jsonl", "write_jsonl", "read_json", "write_json",
           "chunked"]
