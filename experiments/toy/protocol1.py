"""Protocol 1 (Table 1 P1 columns, Fig. 2a, Fig. 4): score once at theta_C, continue M Bernoulli(0.5)
subsets from theta_C with the copied Adam state at --cont-lr, LDS(k) with momentum-weighted visits.
g_test = -(1/4) grad log pi(y*|z) (PG loss, no KL); g_train includes the beta KL term. Summary key
`ks` = record steps; Table 1 entry = mean of the last 10 records (k = 910..1000)."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from experiments.toy.common import (
    DEFAULT_OUT, add_fixture_args, add_method_args, add_subset_args, cache_path, dump_summary, finished,
    fixture_from_args, load_cache, parse_methods, resolved_cont_lr, sample_masks, save_cache, visit_weight_fn,
)
from influence_rlvr.attribution.trajectory import lds_curves
from influence_rlvr.toy import ToyAutoregressiveMLP, ToyExample, expected_rewards, score_checkpoint, train_grpo


def rate_tag(lr: float, cont_lr: float) -> str:
    ratio = lr / cont_lr
    return "eta" if abs(ratio - 1.0) < 1e-9 else f"eta_over_{ratio:g}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Protocol 1: continuation LDS(k)")
    add_fixture_args(p)
    add_method_args(p)
    add_subset_args(p)
    p.add_argument("--full-steps", type=int, default=2000)
    p.add_argument("--subset-steps", type=int, default=1000)
    p.add_argument("--cont-lr", type=float, default=None, help="continuation rate (default lr/10)")
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--warm-optimizer", dest="warm_optimizer", action="store_true", default=True)
    p.add_argument("--no-warm-optimizer", "--fresh-optimizer", dest="warm_optimizer", action="store_false")
    p.add_argument("--lag-predictor", dest="lag_predictor", action="store_true", default=True)
    p.add_argument("--plain-counts", dest="lag_predictor", action="store_false")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--out", type=str, default=None, help="default outputs/toy/p1_<rate>[_fresh]")
    return p


def ground_truth(args: argparse.Namespace, train: list[ToyExample], test: list[ToyExample], ref: ToyAutoregressiveMLP,
                 full_model: ToyAutoregressiveMLP, full_opt: torch.optim.Optimizer, masks: np.ndarray,
                 record_steps: np.ndarray) -> np.ndarray:
    path = cache_path("forward", args)
    cached = load_cache(path)
    if cached is not None and cached["rewards"].shape == (args.m_subsets, len(record_steps), len(test)):
        print(f"[p1] ground-truth cache hit: {path}")
        return cached["rewards"]
    rewards = np.zeros((args.m_subsets, len(record_steps), len(test)))
    start_state = full_model.state_dict()
    start_opt_state = full_opt.state_dict() if args.warm_optimizer else None
    sub_lr = resolved_cont_lr(args)
    t0 = time.time()
    for j in range(args.m_subsets):
        subset = [train[i] for i in np.flatnonzero(masks[j])]
        rec = {}

        def on_step(step, model, _opt, rec=rec):
            if step % args.record_every == 0:
                rec[step] = expected_rewards(model, test)

        train_grpo(subset, args.subset_steps, ref_model=ref, lr=sub_lr, beta=args.beta,
                   seed=args.seed + 100 + j, start_state=start_state, start_opt_state=start_opt_state,
                   on_step=on_step)
        rewards[j] = np.stack([rec[int(k)] for k in record_steps])
        if (j + 1) % 10 == 0:
            print(f"[p1] subsets {j + 1}/{args.m_subsets} ({time.time() - t0:.0f}s)")
    save_cache(path, rewards=rewards, masks=masks, record_steps=record_steps, orders=np.zeros(0))
    return rewards


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    cont_lr = resolved_cont_lr(args)
    out = Path(args.out) if args.out else DEFAULT_OUT / (
        f"p1_{rate_tag(args.lr, cont_lr)}" + ("" if args.warm_optimizer and args.lag_predictor else "_fresh"))
    if finished(out, args.force):
        print(f"[p1] {out}/summary.json exists; skipping (use --force to recompute)")
        return
    train, test, ref = fixture_from_args(args)
    n_train, n_test = len(train), len(test)
    print(f"[p1] {n_train} train / {n_test} test | cont_lr={cont_lr:g} warm={args.warm_optimizer} "
          f"lag={args.lag_predictor} -> {out}")

    t0 = time.time()
    full_model, full_opt = train_grpo(train, args.full_steps, ref_model=ref, lr=args.lr, beta=args.beta, seed=args.seed)
    print(f"[p1] theta_{args.full_steps} in {time.time() - t0:.1f}s | train reward "
          f"{expected_rewards(ref, train).mean():.3f} -> {expected_rewards(full_model, train).mean():.3f} | "
          f"test reward {expected_rewards(full_model, test).mean():.3f}")

    t0 = time.time()
    scores = score_checkpoint(full_model, full_opt, train, test, methods, ref_model=ref, beta=args.beta,
                              lambda_damp=args.lambda_damp, cg_iters=args.cg_iters, cg_tol=args.cg_tol)
    print(f"[p1] scored {methods} at step {args.full_steps} in {time.time() - t0:.1f}s")

    masks = sample_masks(n_train, args.m_subsets, seed=args.seed + 1)
    record_steps = np.arange(args.record_every, args.subset_steps + 1, args.record_every)
    rewards = ground_truth(args, train, test, ref, full_model, full_opt, masks, record_steps)

    visits = visit_weight_fn(args.lag_predictor, args.beta1)
    per_test, curves = {}, {}
    for m in methods:
        per_test[m], curves[m] = lds_curves(scores[m], masks, rewards, record_steps, visits)

    ks = [int(k) for k in record_steps]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "lds_curves.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", *methods])
        for ki, k in enumerate(ks):
            w.writerow([k, *[f"{curves[m][ki]:.4f}" for m in methods]])
    np.savez(out / "raw_final.npz", rewards=rewards[:, -1, :], masks=masks, record_steps=record_steps,
             **{f"score_{m}": scores[m] for m in methods})
    config = {**vars(args), "cont_lr": cont_lr, "methods": ",".join(methods), "out": str(out)}
    dump_summary(out, {"ks": ks, "lds_curves": {m: curves[m].tolist() for m in methods},
                       "per_test_curves": {m: per_test[m].tolist() for m in methods}, "config": config})
    window = min(10, len(ks))
    print(f"[p1] LDS(k) mean over {n_test} targets, {args.m_subsets} subsets; final-{window}-record window:")
    for m in methods:
        print(f"  {m:6s} k={ks[0]}: {curves[m][0]:.3f}  k={ks[-1]}: {curves[m][-1]:.3f}  "
              f"window mean: {curves[m][-window:].mean():.4f}")
    print(f"[p1] wrote {out}/summary.json")


if __name__ == "__main__":
    main()
