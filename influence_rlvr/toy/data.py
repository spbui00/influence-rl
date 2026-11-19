"""Paper fixture (60 clusters x 3 = 180 train, 50 test, input_dim 65). RNG order is load-bearing:
Generator(seed+4242) picks hard clusters, Generator(seed+2024) the overlap loadings, then
torch.manual_seed(seed) draws z_train before z_test; pi_ref = manual_seed(seed) -> construct ->
orthogonal_ per Linear. Members i >= n_train*(1-conflict) get the flipped label ("_neg", harmful)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .model import ToyAutoregressiveMLP, init_reference

Tensor = torch.Tensor


@dataclass(frozen=True)
class ToyExample:
    name: str
    z: tuple[float, ...]
    target: tuple[int, int]
    split: str = "train"
    expected_influence: str | None = None

    def z_tensor(self, device: torch.device | str = "cpu") -> Tensor:
        return torch.tensor(self.z, dtype=torch.float32, device=device)

    def target_tensor(self, device: torch.device | str = "cpu") -> Tensor:
        return torch.tensor(self.target, dtype=torch.long, device=device)


@dataclass
class FixtureSpec:
    n_clusters: int = 60
    per_cluster: int = 3
    input_dim: int = 65
    cluster_signal: float = 3.0
    cluster_target_mode: str = "parity"
    hard_fraction: float = 0.5
    signal_hard: float | None = 0.8
    signature_mode: str = "overlap"
    signature_dim: int = 32
    test_signal_span: str = "span"
    conflict_fraction: float = 0.34
    n_test: int = 50
    hidden_dim: int = 16
    seed: int = 42
    background_scale: float = 0.5
    signal_per_cluster: list[float] | None = field(default=None)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FixtureSpec":
        names = [f for f in cls.__dataclass_fields__ if f not in ("background_scale", "signal_per_cluster")]
        return cls(**{n: getattr(args, n) for n in names})


def signal_per_cluster(n_clusters: int, cluster_signal: float, explicit: Sequence[float] | None,
                       hard_fraction: float, signal_hard: float | None, seed: int) -> list[float]:
    if explicit is not None:
        spc = [float(s) for s in explicit]
        if len(spc) != n_clusters:
            raise ValueError(f"signal_per_cluster length ({len(spc)}) != n_clusters ({n_clusters})")
        return spc
    spc = [float(cluster_signal)] * n_clusters
    n_hard = int(round(hard_fraction * n_clusters))
    if n_hard > 0:
        lo = cluster_signal / 3.0 if signal_hard is None else float(signal_hard)
        gen = torch.Generator().manual_seed(seed + 4242)
        for k in torch.randperm(n_clusters, generator=gen).tolist()[:n_hard]:
            spc[k] = lo
    return spc


def signature_matrix(n_clusters: int, input_dim: int, spc: Sequence[float], signature_mode: str,
                     signature_dim: int, seed: int) -> Tensor:
    S = torch.zeros(n_clusters, input_dim)
    if signature_mode == "onehot":
        for k in range(n_clusters):
            S[k, k] = spc[k]
    elif signature_mode == "overlap":
        gen = torch.Generator().manual_seed(seed + 2024)
        loadings = torch.randn(n_clusters, signature_dim, generator=gen)
        loadings = loadings / loadings.norm(dim=1, keepdim=True).clamp_min(1e-8)
        for k in range(n_clusters):
            S[k, :signature_dim] = loadings[k] * spc[k]
    else:
        raise ValueError(f"unsupported signature_mode {signature_mode!r}")
    return S


def _cluster_targets(n_clusters: int, mode: str, seed: int) -> list[tuple[int, int]]:
    if mode == "parity":
        return [(1, 0) if k % 2 == 0 else (0, 1) for k in range(n_clusters)]
    if mode == "random":
        pool = [(0, 0), (0, 1), (1, 0), (1, 1)]
        gen = torch.Generator().manual_seed(seed + 1000)
        idx = torch.randint(0, len(pool), (n_clusters,), generator=gen).tolist()
        return [pool[i] for i in idx]
    raise ValueError(f"unsupported cluster_target_mode {mode!r}")


def _test_cluster_ids(n_clusters: int, spc: Sequence[float], span: str, n_test: int) -> list[int]:
    if span == "strong":
        ms = max(spc)
        base = [k for k in range(n_clusters) if spc[k] >= ms - 1e-9] or list(range(n_clusters))
    elif span == "span":
        base = sorted(range(n_clusters), key=lambda k: spc[k])
    else:
        raise ValueError(f"unsupported test_signal_span {span!r}")
    n_cl = max(1, min(n_test, len(base)))
    sel = [base[int(round(t))] for t in np.linspace(0, len(base) - 1, n_cl)]
    return (sel * (-(-n_test // n_cl)))[:n_test]


def clustered_dataset(spec: FixtureSpec) -> tuple[list[ToyExample], list[ToyExample], list[int], list[float],
                                                  list[tuple[int, int]]]:
    sig_span = spec.n_clusters if spec.signature_mode == "onehot" else spec.signature_dim
    if spec.input_dim < sig_span:
        raise ValueError(f"input_dim ({spec.input_dim}) must be >= {sig_span} for signature_mode={spec.signature_mode!r}")
    spc = signal_per_cluster(spec.n_clusters, spec.cluster_signal, spec.signal_per_cluster,
                             spec.hard_fraction, spec.signal_hard, spec.seed)
    S = signature_matrix(spec.n_clusters, spec.input_dim, spc, spec.signature_mode, spec.signature_dim, spec.seed)

    torch.manual_seed(spec.seed)
    n_train = spec.n_clusters * spec.per_cluster
    cluster = torch.arange(n_train) % spec.n_clusters
    z_train = torch.randn(n_train, spec.input_dim) * spec.background_scale
    z_train += S[cluster]

    cluster_targets = _cluster_targets(spec.n_clusters, spec.cluster_target_mode, spec.seed)
    n_conflict = int(round(spec.conflict_fraction * spec.per_cluster))
    train: list[ToyExample] = []
    for i in range(n_train):
        k = int(cluster[i])
        base = cluster_targets[k]
        flipped = n_conflict > 0 and (i // spec.n_clusters) >= spec.per_cluster - n_conflict
        target = (1 - base[0], 1 - base[1]) if flipped else base
        train.append(ToyExample(
            name=f"c{k}_ex{i}" + ("_neg" if flipped else ""),
            z=tuple(z_train[i].tolist()),
            target=(int(target[0]), int(target[1])),
            expected_influence="harmful" if flipped else None,
        ))

    ids = _test_cluster_ids(spec.n_clusters, spc, spec.test_signal_span, spec.n_test)
    z_test = torch.randn(len(ids), spec.input_dim) * spec.background_scale
    test: list[ToyExample] = []
    for i, k in enumerate(ids):
        z_test[i] += S[k]
        test.append(ToyExample(name=f"test_c{k}", z=tuple(z_test[i].tolist()),
                               target=cluster_targets[k], split="test"))
    return train, test, ids, spc, cluster_targets


def fixture(spec: FixtureSpec) -> tuple[list[ToyExample], list[ToyExample], ToyAutoregressiveMLP]:
    train, test, _, _, _ = clustered_dataset(spec)
    ref = init_reference(spec.input_dim, spec.hidden_dim, spec.seed)
    return train, test[: spec.n_test], ref


def z_matrix(examples: Sequence[ToyExample]) -> np.ndarray:
    return np.asarray([e.z for e in examples], dtype=np.float64)


def target_matrix(examples: Sequence[ToyExample]) -> np.ndarray:
    return np.asarray([e.target for e in examples], dtype=np.int64)
