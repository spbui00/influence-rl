from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ..utils import flatten, trainable_params, unflatten

Tensor = torch.Tensor


class GradCacheFVP:
    def __init__(self, grad_cache: Sequence[Tensor], prob_cache: Sequence[Tensor]):
        if len(grad_cache) != len(prob_cache):
            raise ValueError("grad_cache and prob_cache must have the same length")
        self.grad_cache = list(grad_cache)
        self.prob_cache = list(prob_cache)
        self.n = len(self.grad_cache)
        self._key: tuple | None = None
        self._dev: list[tuple[Tensor, Tensor]] = []

    def _on(self, like: Tensor) -> list[tuple[Tensor, Tensor]]:
        key = (like.device, like.dtype)
        if key != self._key:
            self._dev = [(G.to(device=like.device, dtype=like.dtype), pi.to(device=like.device, dtype=like.dtype))
                         for G, pi in zip(self.grad_cache, self.prob_cache)]
            self._key = key
        return self._dev

    def __call__(self, p: Tensor) -> Tensor:
        total = torch.zeros_like(p)
        for G, pi in self._on(p):
            total = total + G.T @ (pi * (G @ p))
        return total / max(self.n, 1)


def exact_fisher_from_grad_cache(grad_cache: Sequence[Tensor], prob_cache: Sequence[Tensor]) -> Tensor:
    if len(grad_cache) != len(prob_cache):
        raise ValueError("grad_cache and prob_cache must have the same length")
    d = grad_cache[0].shape[-1]
    F = torch.zeros(d, d, dtype=grad_cache[0].dtype, device=grad_cache[0].device)
    for G, pi in zip(grad_cache, prob_cache):
        F = F + G.T @ (pi.to(G).unsqueeze(1) * G)
    return F / max(len(grad_cache), 1)


@contextmanager
def math_attention():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        yield
        return
    with sdpa_kernel(SDPBackend.MATH):
        yield


@dataclass
class FisherRow:
    input_ids: Tensor
    response_start: int


class PolicyFisherFVP:
    def __init__(self, model: nn.Module, rows: Sequence[FisherRow | tuple[Tensor, int]], *,
                 device: torch.device | str | None = None, micro_batch: int = 1,
                 normalize: bool = False, power_iters: int = 15, seed: int = 0):
        self.model = model
        self.params = trainable_params(model)
        if not self.params:
            raise ValueError("model has no trainable parameters")
        self.device = torch.device(device) if device is not None else self.params[0].device
        self.n_params = sum(p.numel() for p in self.params)
        self.rows = [r if isinstance(r, FisherRow) else FisherRow(*r) for r in rows]
        if not self.rows:
            raise ValueError("rows must be non-empty")
        self.micro_batch = max(int(micro_batch), 1)
        self._batches = [self._pack(self.rows[i:i + self.micro_batch])
                         for i in range(0, len(self.rows), self.micro_batch)]
        self.scale = 1.0
        self.lam_max: float | None = None
        if normalize:
            self.lam_max = self._top_eigenvalue(int(power_iters), int(seed))
            self.scale = self.lam_max

    def _pack(self, rows: list[FisherRow]) -> tuple[Tensor, Tensor, Tensor]:
        lengths = [int(r.input_ids.numel()) for r in rows]
        L = max(lengths)
        ids = torch.zeros(len(rows), L, dtype=torch.long)
        mask = torch.zeros(len(rows), L, dtype=torch.long)
        pos = torch.zeros(len(rows), L - 1, dtype=torch.bool)
        for b, (r, n) in enumerate(zip(rows, lengths)):
            ids[b, :n] = r.input_ids.reshape(-1).to(torch.long)
            mask[b, :n] = 1
            start = int(r.response_start)
            if not 1 <= start < n:
                raise ValueError(f"response_start must lie in [1, len(input_ids)), got {start} for length {n}")
            pos[b, start - 1:n - 1] = True
        return ids.to(self.device), mask.to(self.device), pos.to(self.device)

    def _response_logits(self, ids: Tensor, mask: Tensor, pos: Tensor) -> Tensor:
        with math_attention():
            out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits[:, :-1, :][pos]

    def _raw(self, v: Tensor) -> Tensor:
        v = v.detach().to(device=self.device, dtype=torch.float32)
        v_parts = unflatten(v, self.params)
        total = torch.zeros(self.n_params, dtype=torch.float32, device=self.device)
        for ids, mask, pos in self._batches:
            resp = self._response_logits(ids, mask, pos)
            p = torch.softmax(resp.detach().float(), dim=-1)
            u = torch.zeros_like(resp, requires_grad=True)
            jt_u = torch.autograd.grad(resp, self.params, grad_outputs=u, create_graph=True,
                                       retain_graph=True, allow_unused=True)
            terms = [(g * vp).sum() for g, vp in zip(jt_u, v_parts) if g is not None]
            if not terms:
                raise RuntimeError("PolicyFisherFVP: no trainable parameter is connected to the logits")
            jv = torch.autograd.grad(torch.stack(terms).sum(), u, retain_graph=True)[0].float()
            m_jv = p * jv - p * (p * jv).sum(dim=-1, keepdim=True)
            fv_parts = torch.autograd.grad(resp, self.params, grad_outputs=m_jv.to(resp.dtype),
                                           retain_graph=False, allow_unused=True)
            total = total + flatten(fv_parts, self.params).to(self.device)
        return total / len(self.rows)

    def _top_eigenvalue(self, power_iters: int, seed: int) -> float:
        gen = torch.Generator(device=self.device).manual_seed(seed)
        v = torch.randn(self.n_params, generator=gen, device=self.device, dtype=torch.float32)
        v = v / v.norm().clamp(min=1e-20)
        for _ in range(max(1, power_iters)):
            Fv = self._raw(v)
            nrm = Fv.norm()
            if nrm < 1e-20:
                break
            v = Fv / nrm
        return float(torch.dot(v, self._raw(v)).clamp(min=1e-20).item())

    def __call__(self, v: Tensor) -> Tensor:
        Fv = self._raw(v)
        return Fv if self.scale == 1.0 else Fv / self.scale
