from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from ..utils import flat_param_slices, trainable_params
from .base import Estimator

Tensor = torch.Tensor
FitSample = tuple[float, Callable[[], Tensor]]


def _eigh_basis(mat: Tensor) -> Tensor:
    m64 = mat.to(torch.float64)
    try:
        _, q = torch.linalg.eigh(m64)
    except RuntimeError:
        n = m64.shape[0]
        jitter = max(float(m64.diagonal().abs().mean()), 1e-30) * 1e-8
        eye = torch.eye(n, dtype=m64.dtype, device=m64.device)
        try:
            _, q = torch.linalg.eigh(m64 + jitter * eye)
        except RuntimeError:
            _, q = torch.linalg.eigh((m64 + jitter * eye).cpu())
            q = q.to(m64.device)
    return q.to(mat.dtype)


def _delta(entry: dict, name: str) -> Tensor:
    if "delta" not in entry:
        raise RuntimeError(
            f"fit_ekfac: module {name!r} recorded a forward with no matching backward "
            f"gradient. The usual cause is gradient checkpointing (model-level or a "
            f"checkpoint() inside the closure): the recompute pass re-fires the forward "
            f"hooks with no backward. Run the fit closures with checkpointing OFF.")
    return entry["delta"]


@dataclass
class _LinearBlock:
    name: str
    in_dim: int
    out_dim: int
    has_bias: bool
    weight_slice: slice
    bias_slice: slice | None
    q_a: Tensor | None = None
    q_s: Tensor | None = None
    lam: Tensor | None = None

    def unflatten(self, flat: Tensor) -> Tensor:
        w = flat[self.weight_slice].view(self.out_dim, -1)
        if not self.has_bias:
            return w
        b = flat[self.bias_slice].view(self.out_dim, 1)
        return torch.cat([w, b], dim=1)

    def write_flat(self, block: Tensor, out: Tensor) -> None:
        if self.has_bias:
            out[self.weight_slice] = block[:, :-1].reshape(-1)
            out[self.bias_slice] = block[:, -1].reshape(-1)
        else:
            out[self.weight_slice] = block.reshape(-1)


@dataclass
class EKFACFactors:
    blocks: list[_LinearBlock]
    n_params: int
    covered: Tensor

    def uncovered_mask(self) -> Tensor:
        return ~self.covered


def _traced_blocks(model: nn.Module) -> tuple[list[tuple[nn.Linear, _LinearBlock]], int]:
    slices, n_params = flat_param_slices(model)
    out: list[tuple[nn.Linear, _LinearBlock]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or not mod.weight.requires_grad:
            continue
        has_bias = mod.bias is not None and mod.bias.requires_grad
        out.append((mod, _LinearBlock(
            name=name,
            in_dim=mod.in_features + (1 if has_bias else 0),
            out_dim=mod.out_features,
            has_bias=has_bias,
            weight_slice=slices[id(mod.weight)],
            bias_slice=slices[id(mod.bias)] if has_bias else None,
        )))
    return out, n_params


def _aug(a: Tensor, has_bias: bool) -> Tensor:
    if not has_bias:
        return a
    return torch.cat([a, torch.ones_like(a[:, :1])], dim=1)


def fit_ekfac(
    model: nn.Module,
    samples: Sequence[FitSample],
    *,
    dtype: torch.dtype = torch.float64,
    progress: bool = False,
) -> EKFACFactors:
    if not samples:
        raise ValueError("fit_ekfac needs at least one (weight, closure) sample.")
    traced, n_params = _traced_blocks(model)
    if not traced:
        raise ValueError("fit_ekfac found no trainable nn.Linear layers to trace.")
    params = trainable_params(model)
    device = params[0].device
    acc = torch.float64
    captures: dict[str, list[dict]] = {blk.name: [] for _, blk in traced}

    def make_hook(name: str):
        def fwd(module, inputs, output):
            entry = {"a": inputs[0].detach().reshape(-1, inputs[0].shape[-1])}
            captures[name].append(entry)
            output.register_hook(lambda g: entry.__setitem__(
                "delta", g.detach().reshape(-1, g.shape[-1])))
        return fwd

    handles = [mod.register_forward_hook(make_hook(blk.name)) for mod, blk in traced]

    def run(closure):
        for lst in captures.values():
            lst.clear()
        loss = closure()
        torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)

    def closure_ok() -> bool:
        for lst in captures.values():
            for e in lst:
                if not torch.isfinite(e["a"]).all():
                    return False
                d = e.get("delta")
                if d is not None and not torch.isfinite(d).all():
                    return False
        return True

    n_skip = 0
    try:
        A = {b.name: torch.zeros(b.in_dim, b.in_dim, dtype=acc, device=device) for _, b in traced}
        S = {b.name: torch.zeros(b.out_dim, b.out_dim, dtype=acc, device=device) for _, b in traced}
        w_total = 0.0
        for i, (w, closure) in enumerate(samples):
            run(closure)
            if not closure_ok():
                n_skip += 1
                continue
            w_total += w
            for _, blk in traced:
                for e in captures[blk.name]:
                    a = _aug(e["a"].to(acc), blk.has_bias)
                    d = _delta(e, blk.name).to(acc)
                    A[blk.name] += w * (a.T @ a)
                    S[blk.name] += w * (d.T @ d)
            if progress and (i + 1) % 50 == 0:
                print(f"    [ekfac] covariance pass {i + 1}/{len(samples)}", flush=True)
        if n_skip:
            print(f"    [ekfac] WARNING: dropped {n_skip}/{len(samples)} samples with "
                  f"non-finite captures", flush=True)
        if w_total <= 0:
            raise RuntimeError("fit_ekfac: every sample had non-finite captures.")
        for _, blk in traced:
            blk.q_a = _eigh_basis(A.pop(blk.name) / w_total).to(dtype)
            blk.q_s = _eigh_basis(S.pop(blk.name) / w_total).to(dtype)
        if progress:
            print(f"    [ekfac] eigenbases done ({len(traced)} blocks)", flush=True)

        lam = {b.name: torch.zeros(b.out_dim, b.in_dim, dtype=acc, device=device) for _, b in traced}
        for i, (w, closure) in enumerate(samples):
            run(closure)
            if not closure_ok():
                continue
            for _, blk in traced:
                G = torch.zeros(blk.out_dim, blk.in_dim, dtype=acc, device=device)
                for e in captures[blk.name]:
                    G += _delta(e, blk.name).to(acc).T @ _aug(e["a"].to(acc), blk.has_bias)
                G_tilde = blk.q_s.to(acc).T @ G @ blk.q_a.to(acc)
                lam[blk.name] += w * G_tilde.pow(2)
            if progress and (i + 1) % 50 == 0:
                print(f"    [ekfac] eigenvalue pass {i + 1}/{len(samples)}", flush=True)
        for _, blk in traced:
            blk.lam = (lam[blk.name] / w_total).to(dtype)
            if not (torch.isfinite(blk.q_a).all() and torch.isfinite(blk.q_s).all()
                    and torch.isfinite(blk.lam).all()):
                raise RuntimeError(
                    f"fit_ekfac: non-finite factors for block {blk.name!r} despite fp64 "
                    f"accumulation and sample screening — inspect the Fisher batch.")
    finally:
        for h in handles:
            h.remove()

    covered = torch.zeros(n_params, dtype=torch.bool)
    for _, blk in traced:
        covered[blk.weight_slice] = True
        if blk.has_bias:
            covered[blk.bias_slice] = True
    return EKFACFactors(blocks=[b for _, b in traced], n_params=n_params, covered=covered)


class DampedFisherEKFAC(Estimator):
    name = "ekfac"

    def __init__(self, factors: EKFACFactors, lambda_damp: float = 0.1,
                 rel_damp: float | None = None):
        self.factors = factors
        self.lambda_damp = float(lambda_damp)
        self.rel_damp = None if rel_damp is None else float(rel_damp)

    def block_damp(self, blk: _LinearBlock) -> float:
        if self.rel_damp is None:
            return self.lambda_damp
        return max(self.rel_damp * float(blk.lam.mean()), 1e-12)

    def _dtype(self) -> torch.dtype:
        for blk in self.factors.blocks:
            if blk.lam is not None:
                return blk.lam.dtype
        return torch.float64

    def solve(self, g_test: Tensor) -> Tensor:
        g = g_test.detach().to(dtype=self._dtype()).reshape(-1)
        if g.numel() != self.factors.n_params:
            raise ValueError(f"gradient has {g.numel()} entries, factors were fit "
                             f"for {self.factors.n_params}.")
        h = torch.empty_like(g)
        unc = self.factors.uncovered_mask()
        h[unc] = g[unc] / self.lambda_damp
        for blk in self.factors.blocks:
            G = blk.unflatten(g).to(blk.lam.device)
            H = blk.q_s @ ((blk.q_s.T @ G @ blk.q_a) / (blk.lam + self.block_damp(blk))) @ blk.q_a.T
            blk.write_flat(H.to(h.device), h)
        return h.to(dtype=torch.float32)

    def apply_fisher(self, v: Tensor) -> Tensor:
        x = v.detach().to(dtype=self._dtype()).reshape(-1)
        out = torch.zeros_like(x)
        for blk in self.factors.blocks:
            V = blk.unflatten(x).to(blk.lam.device)
            FV = blk.q_s @ ((blk.q_s.T @ V @ blk.q_a) * blk.lam) @ blk.q_a.T
            blk.write_flat(FV.to(out.device), out)
        return out.to(dtype=torch.float32)
