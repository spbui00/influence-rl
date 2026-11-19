"""Protocol 2 (Table 1 last column, Fig. 2b right point): snapshots of the reference run every
--score-every steps (state_dict + live Adam diagonal), totals = sum_c lr * scores_c over ALL snapshots for
EVERY train example (legacy semantics; the paper text says 'checkpoints at which the example was
sampled'), from-scratch subset retrains with fresh Adam, plain visit counts at k = subset_steps."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from experiments.toy.common import (
    DEFAULT_OUT, add_fixture_args, add_method_args, add_subset_args, cache_path, dump_summary, finished,
    fixture_from_args, load_cache, parse_methods, sample_masks, save_cache,
)
from influence_rlvr.attribution.trajectory import lds_curves, sum_all
from influence_rlvr.toy import ToyAutoregressiveMLP, adam_preconditioner, expected_rewards, score_checkpoint, train_grpo


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Protocol 2: trajectory-summed scores, from-scratch retrains")
    add_fixture_args(p)
    add_method_args(p)
    add_subset_args(p)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--score-every", type=int, default=100)
    p.add_argument("--subset-steps", type=int, default=None, help="default = --steps")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT / "p2_history"))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    subset_steps = args.subset_steps if args.subset_steps is not None else args.steps
    out = Path(args.out)
    if finished(out, args.force):
        print(f"[p2] {out}/summary.json exists; skipping (use --force to recompute)")
        return
    train, test, ref = fixture_from_args(args)
    n_train, n_test = len(train), len(test)
    print(f"[p2] {n_train} train / {n_test} test | N={args.steps} K={args.score_every} -> {out}")

    snapshots: dict[int, dict] = {}
    preconds: dict[int, torch.Tensor] = {}

    def on_step(step, model, opt):
        if step % args.score_every == 0:
            snapshots[step] = copy.deepcopy(model.state_dict())
            preconds[step] = adam_preconditioner(opt, model)

    t0 = time.time()
    full_model, _ = train_grpo(train, args.steps, ref_model=ref, lr=args.lr, beta=args.beta, seed=args.seed,
                               on_step=on_step)
    print(f"[p2] reference run: {args.steps} steps, {len(snapshots)} snapshots in {time.time() - t0:.1f}s | "
          f"train reward {expected_rewards(ref, train).mean():.3f} -> {expected_rewards(full_model, train).mean():.3f}"
          f" | test reward {expected_rewards(full_model, test).mean():.3f}")

    scorer = ToyAutoregressiveMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    scores_by_step: dict[str, dict[int, np.ndarray]] = {m: {} for m in methods}
    for c in sorted(snapshots):
        scorer.load_state_dict(snapshots[c])
        t0 = time.time()
        s_c = score_checkpoint(scorer, preconds[c], train, test, methods, ref_model=ref, beta=args.beta,
                               lambda_damp=args.lambda_damp, cg_iters=args.cg_iters, cg_tol=args.cg_tol)
        for m in methods:
            scores_by_step[m][c] = s_c[m]
        print(f"[p2] snapshot {c:5d} scored {methods} ({time.time() - t0:.1f}s)")
    lr_by_step = {c: args.lr for c in snapshots}
    totals = {m: sum_all(scores_by_step[m], lr_by_step, nan="raise") for m in methods}

    masks = sample_masks(n_train, args.m_subsets, seed=args.seed + 1)
    path = cache_path("history", args)
    cached = load_cache(path)
    if cached is not None and cached["rewards"].shape == (args.m_subsets, n_test):
        rewards = cached["rewards"]
        print(f"[p2] ground-truth cache hit: {path}")
    else:
        rewards = np.zeros((args.m_subsets, n_test))
        t0 = time.time()
        for j in range(args.m_subsets):
            subset = [train[i] for i in np.flatnonzero(masks[j])]
            model_j, _ = train_grpo(subset, subset_steps, ref_model=ref, lr=args.lr, beta=args.beta,
                                    seed=args.seed + 100 + j)
            rewards[j] = expected_rewards(model_j, test)
            if (j + 1) % 10 == 0:
                print(f"[p2] subsets {j + 1}/{args.m_subsets} ({time.time() - t0:.0f}s)")
        save_cache(path, rewards=rewards, masks=masks)

    ks = np.array([subset_steps])
    lds, per_test = {}, {}
    for m in methods:
        pt, mean = lds_curves(totals[m], masks, rewards[:, None, :], ks, None)
        per_test[m] = pt[:, 0].tolist()
        lds[m] = float(mean[0])

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "scores.npz", masks=masks, rewards=rewards, snapshot_steps=np.array(sorted(snapshots)),
             **{f"total_{m}": totals[m] for m in methods})
    config = {**vars(args), "subset_steps": subset_steps, "methods": ",".join(methods), "out": str(out)}
    dump_summary(out, {"lds": lds, "per_test": per_test, "config": config})
    print(f"[p2] endpoint LDS over {n_test} targets, {args.m_subsets} subsets:")
    for m in methods:
        print(f"  {m:6s} {lds[m]:+.4f}")
    print(f"[p2] wrote {out}/summary.json")


if __name__ == "__main__":
    main()
