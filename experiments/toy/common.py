"""Ground-truth cache payloads keep the legacy semantics (levers only when non-default w.r.t. the
legacy CLI, use_adam always True, warm_v_only/shuffle_order always False) so keys equal the legacy ones."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.toy.summaries import (
    ARTIFACT_P1, ARTIFACTS, DEFAULT_OUT, METHODS, P1_LABELS, P1_TAGS, canonical_method, config_namespace, dump_summary,
    final_window, finished, load_summary, p1_default_runs, p2_default_run, summary_path,
)
from influence_rlvr.attribution.trajectory import response_weighted_visits, visit_counts
from influence_rlvr.toy import FixtureSpec, ToyAutoregressiveMLP, ToyExample, fixture

PAPER_FIXTURE = FixtureSpec()


def add_fixture_args(p: argparse.ArgumentParser) -> None:
    f = PAPER_FIXTURE
    d = p.add_argument
    d("--n-clusters", type=int, default=f.n_clusters)
    d("--per-cluster", type=int, default=f.per_cluster)
    d("--input-dim", type=int, default=f.input_dim)
    d("--cluster-signal", type=float, default=f.cluster_signal)
    d("--cluster-target-mode", choices=["parity", "random"], default=f.cluster_target_mode)
    d("--hard-fraction", type=float, default=f.hard_fraction)
    d("--signal-hard", type=float, default=f.signal_hard)
    d("--signature-mode", choices=["onehot", "overlap"], default=f.signature_mode)
    d("--signature-dim", type=int, default=f.signature_dim)
    d("--test-signal-span", choices=["strong", "span"], default=f.test_signal_span)
    d("--conflict-fraction", type=float, default=f.conflict_fraction)
    d("--n-test", type=int, default=f.n_test)
    d("--hidden-dim", type=int, default=f.hidden_dim)
    d("--seed", type=int, default=f.seed)
    d("--lr", type=float, default=1e-3)
    d("--beta", type=float, default=0.1)


def add_method_args(p: argparse.ArgumentParser) -> None:
    d = p.add_argument
    d("--methods", type=str, default=",".join(METHODS))
    d("--lambda-damp", type=float, default=0.1)
    d("--cg-iters", type=int, default=100)
    d("--cg-tol", type=float, default=1e-6)


def add_subset_args(p: argparse.ArgumentParser) -> None:
    d = p.add_argument
    d("--m-subsets", type=int, default=100)
    d("--cache-dir", type=str, default=str(DEFAULT_OUT / "_cache"))
    d("--no-cache", action="store_true")
    d("--force", action="store_true", help="recompute even if <out>/summary.json exists")


def parse_methods(spec: str) -> list[str]:
    out = []
    for m in (s.strip() for s in spec.split(",")):
        if m and canonical_method(m) not in out:
            out.append(canonical_method(m))
    return out


def fixture_from_args(args: argparse.Namespace) -> tuple[list[ToyExample], list[ToyExample], ToyAutoregressiveMLP]:
    return fixture(FixtureSpec.from_args(args))


def dataset_lever_keys(args: argparse.Namespace) -> dict:
    levers = {
        "hard_fraction": args.hard_fraction or None,
        "signal_hard": args.signal_hard,
        "signature_mode": None if args.signature_mode == "onehot" else args.signature_mode,
        "signature_dim": args.signature_dim if args.signature_mode == "overlap" else None,
        "test_signal_span": None if args.test_signal_span == "strong" else args.test_signal_span,
        "conflict_fraction": args.conflict_fraction or None,
    }
    return {k: v for k, v in levers.items() if v is not None}


def resolved_cont_lr(args: argparse.Namespace) -> float:
    return float(args.cont_lr) if getattr(args, "cont_lr", None) is not None else float(args.lr) / 10


def cache_key(protocol: str, args: argparse.Namespace) -> str:
    payload = {
        "protocol": protocol,
        "n_clusters": args.n_clusters, "per_cluster": args.per_cluster,
        "input_dim": args.input_dim, "cluster_signal": args.cluster_signal,
        "cluster_target_mode": args.cluster_target_mode,
        "n_test": args.n_test, "hidden_dim": args.hidden_dim, "lr": args.lr,
        "beta": args.beta, "use_adam": True, "seed": args.seed,
        "m_subsets": args.m_subsets,
        **dataset_lever_keys(args),
    }
    if protocol == "forward":
        payload.update({
            "mode": "continuation", "full_steps": args.full_steps, "cont_lr": resolved_cont_lr(args),
            "subset_steps": args.subset_steps, "record_every": args.record_every,
            "warm_optimizer": bool(args.warm_optimizer), "warm_v_only": False, "shuffle_order": False,
        })
    elif protocol == "history":
        payload["subset_steps"] = args.subset_steps if args.subset_steps is not None else args.steps
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cache_path(protocol: str, args: argparse.Namespace) -> Path | None:
    if getattr(args, "no_cache", False):
        return None
    return Path(args.cache_dir) / f"{protocol}_{cache_key(protocol, args)}.npz"


def load_cache(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_cache(path: Path | None, **arrays: np.ndarray) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def sample_masks(n_train: int, m_subsets: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=(m_subsets, n_train)).astype(bool)


VisitFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def visit_weight_fn(lag: bool, beta1: float) -> VisitFn | None:
    if not lag:
        return None
    return lambda mask, ks: response_weighted_visits(int(mask.sum()), ks, None, beta1)


__all__ = [
    "PAPER_FIXTURE", "DEFAULT_OUT", "ARTIFACTS", "P1_TAGS", "P1_LABELS", "ARTIFACT_P1", "METHODS",
    "add_fixture_args", "add_method_args", "add_subset_args", "parse_methods", "fixture_from_args",
    "dataset_lever_keys", "resolved_cont_lr", "cache_key", "cache_path", "load_cache", "save_cache",
    "sample_masks", "visit_weight_fn", "visit_counts", "summary_path", "load_summary", "config_namespace",
    "p1_default_runs", "p2_default_run", "final_window", "finished", "dump_summary",
]
