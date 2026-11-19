"""Horizon LDS: how long one in-batch selection decision stays visible. From checkpoint c take the N7 batch (same seed,
64 prompts, G = 8), score it once against the gold targets and hold its rollouts fixed. For each of M random half-batches S:
step 1 = AdamW step on S's rollout gradient (the fixed-rollout ICE step of next_step.py, LDS ~1 at m = 1), steps 2..m =
ordinary GRPO on the run's own schedule (--continue-batch random pool prompts per step, fresh rollouts at G = 8, selector
none, same seeded prompt order for every S). Outcome = gold log-prob of the targets at every m in --grid, minus its value
at c. LDS(m) = Spearman over S of sum_{i in S} score_i vs outcome(m), per target, averaged.
  --prepare            write <out>/prepared.pt (rollouts, scores, masks); one GPU job
  --subsets a:b        run subsets a..b-1, writing <out>/subset_<j>.json (resumable; shardable as a Slurm array)
  --report             <out>/horizon.json + horizon.png from the subset files
  --add-methods m,..   score more estimators on the saved rollouts into prepared.pt (outcomes are estimator-free, so no
                       subset reruns); methods already there are recomputed only as a reproducibility check"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from influence_rlvr.attribution import influence_matrix
from influence_rlvr.attribution.metrics import per_target_lds
from influence_rlvr.files import write_json
from influence_rlvr.llm.config import DEFAULT_MODEL_ID, LoraSpec, ScoringSpec, add_spec_args, spec_from_args
from influence_rlvr.llm.data import read_rows, subset_masks, subsets_fixed_size
from influence_rlvr.llm.generation import RolloutBatch
from influence_rlvr.llm.gradients import gold_gradients, rollout_gradients
from influence_rlvr.llm.model import build_policy
from influence_rlvr.llm.rewards import row_reward_fn
from influence_rlvr.llm.scoring import build_estimator
from influence_rlvr.llm.select import apply_update, sample_batch
from influence_rlvr.utils import detect_device, trainable_params

from .next_step import GoldTarget, batch_gradients, stack

DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}


def continuation_order(seed: int, n_pool: int, n_steps: int, per_step: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 0x0881204])  # own stream: not the pool builders', not train.py's
    return np.concatenate([rng.permutation(n_pool) for _ in range(n_steps * per_step // n_pool + 1)])


def saved_rollout_grad(model, tok, spec, row: dict, roll, device):
    ids, rmask = roll
    texts = [tok.decode(s[m].tolist(), skip_special_tokens=True) for s, m in zip(ids, rmask)]
    rb = RolloutBatch(texts, ids.to(device=device, dtype=torch.long), rmask.to(device), 1, ids.shape[0])
    return rollout_gradients(model, tok, [row["prompt"]], [row_reward_fn(row)], spec, objective="grpo", skip_saturated=True,
                             rollout=rb, num_samples=ids.shape[0])[0]


def add_methods(args, spec, device) -> None:
    path = args.out / "prepared.pt"
    prep = torch.load(path, map_location="cpu", weights_only=False)
    model, tok = build_policy(args.model_id, spec_from_args(LoraSpec, args, prefix="lora"), device=device, checkpoint_dir=args.checkpoint, dtype=DTYPE[args.dtype])
    model.eval()
    D = sum(q.numel() for q in trainable_params(model))
    pool = read_rows(args.pool)
    batch = [pool[i] for i in prep["batch_ids"]]
    targets = read_rows(args.targets)[:args.n_targets]
    ests = {m: build_estimator(model, tok, spec, args.checkpoint, pool, m) for m in args.add_methods.split(",") if m}  # fit before G fills the GPU
    G = stack([saved_rollout_grad(model, tok, spec, row, roll, device) for row, roll in zip(batch, prep["rolls"])], D, device)
    assert ((G.abs().sum(1) > 0).cpu().numpy() == prep["live"]).all(), "saved rollouts no longer give the saved live set"
    g_test = stack(gold_gradients(model, tok, [r["prompt"] for r in targets], [r["gold"] for r in targets]), D, device)
    for m, est in ests.items():
        key, s = f"rollout-gold/{m}", np.asarray(influence_matrix(est, g_test, G, device=device))
        if key in prep["scores"]:
            old = np.asarray(prep["scores"][key])
            print(f"[horizon] {key}: recomputed vs saved, max |diff| / max |saved| = {np.abs(s - old).max() / np.abs(old).max():.2e}", flush=True)
        else:
            prep["scores"][key] = s
            print(f"[horizon] added {key} {s.shape}", flush=True)
    torch.save(prep, path)


def prepare(args, spec, device) -> None:
    model, tok = build_policy(args.model_id, spec_from_args(LoraSpec, args, prefix="lora"), device=device, checkpoint_dir=args.checkpoint, dtype=DTYPE[args.dtype])
    model.eval()
    D = sum(q.numel() for q in trainable_params(model))
    pool = read_rows(args.pool)
    batch_ids = np.sort(np.random.default_rng(spec.seed).choice(len(pool), size=min(args.batch_size, len(pool)), replace=False))
    batch = [pool[i] for i in batch_ids]
    targets = read_rows(args.targets)[:args.n_targets]
    ests = {m: build_estimator(model, tok, spec, args.checkpoint, pool, m) for m in args.methods.split(",") if m}
    G, rolls, rstd = batch_gradients(model, tok, spec, batch, spec.seed + 20_000, D, device)  # same draw as next_step.py
    live = (G.abs().sum(1) > 0).cpu().numpy()
    g_test = stack(gold_gradients(model, tok, [r["prompt"] for r in targets], [r["gold"] for r in targets]), D, device)
    scores = {f"rollout-gold/{m}": influence_matrix(est, g_test, G, device=device) for m, est in ests.items()}
    masks = subset_masks(subsets_fixed_size(len(batch), args.n_subsets, 0.5, spec.seed + 1), len(batch))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"batch_ids": batch_ids, "live": live, "rstd": rstd, "masks": masks, "scores": scores,
                "rolls": [(rb.token_ids.to(torch.int32).cpu(), rb.response_mask.to(torch.bool).cpu()) for rb in rolls],
                "f0": GoldTarget(model, tok, spec, targets, D, device, with_grad=False).outcome()}, args.out / "prepared.pt")
    print(f"[horizon] prepared: batch {len(batch)} ({int(live.sum())} live), {len(masks)} subsets, scores {list(scores)}", flush=True)


def run_subsets(args, spec, device) -> None:
    prep = torch.load(args.out / "prepared.pt", map_location="cpu", weights_only=False)
    pool = read_rows(args.pool)
    batch = [pool[i] for i in prep["batch_ids"]]
    targets = read_rows(args.targets)[:args.n_targets]
    hp = json.loads((args.checkpoint.parent / "manifest.json").read_text())["args"]
    grid = sorted(int(m) for m in args.grid.split(","))
    order = continuation_order(spec.seed, len(pool), grid[-1], args.continue_batch)
    lo, hi = (int(x) for x in args.subsets.split(":"))
    for j in range(lo, hi):
        if (args.out / f"subset_{j}.json").is_file():
            continue
        t0 = time.time()
        model, tok = build_policy(args.model_id, spec_from_args(LoraSpec, args, prefix="lora"), device=device, checkpoint_dir=args.checkpoint, dtype=DTYPE[args.dtype])
        model.eval()
        optimizer = torch.optim.AdamW(trainable_params(model), lr=hp["lr"], betas=(0.9, hp["adam_beta2"]), eps=1e-8, weight_decay=hp["weight_decay"])
        optimizer.load_state_dict(torch.load(args.checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
        D = sum(q.numel() for q in trainable_params(model))
        gold_target = GoldTarget(model, tok, spec, targets, D, device, with_grad=False)
        mask = prep["masks"][j]
        f0 = gold_target.outcome()
        grad = torch.zeros(D, dtype=torch.float32, device=device)
        for i in np.flatnonzero(mask & prep["live"]):
            g = saved_rollout_grad(model, tok, spec, batch[i], prep["rolls"][i], device)
            if g is not None:
                grad += g.to(device)
        apply_update(model, optimizer, grad / int(mask.sum()), hp["max_grad_norm"])  # /n over all members, as next_step.py
        gold = {}
        for step in range(1, grid[-1] + 1):
            if step > 1:
                rows = [pool[i] for i in order[(step - 2) * args.continue_batch: (step - 1) * args.continue_batch]]
                b = sample_batch(model, tok, spec, rows, seed=spec.seed * 1_000_003 + step * 1_000 + (j + 1) * 7_919, device=device)
                apply_update(model, optimizer, b.grads.mean(0), hp["max_grad_norm"])
                del b
            if step in grid:
                gold[str(step)] = (gold_target.outcome() - f0).tolist()
                print(f"[horizon] subset {j} m={step}: mean gold log-prob change {np.mean(gold[str(step)]):+.4f} ({time.time() - t0:.0f}s)", flush=True)
        write_json(args.out / f"subset_{j}.json", {"subset": j, "mask": mask.astype(int).tolist(), "grid": grid, "f0": f0.tolist(),
                                                   "gold": gold, "seconds": time.time() - t0})
        del model, optimizer, gold_target
        if device.type == "cuda":
            torch.cuda.empty_cache()


def report(out: Path) -> dict:
    prep = torch.load(out / "prepared.pt", map_location="cpu", weights_only=False)
    subs = sorted((json.load(open(f)) for f in out.glob("subset_*.json")), key=lambda s: s["subset"])
    if len(subs) < 10:
        raise SystemExit(f"{len(subs)} subsets done; need at least 10")
    grid = subs[0]["grid"]
    masks = np.asarray([s["mask"] for s in subs], dtype=float)
    y = {m: np.asarray([s["gold"][str(m)] for s in subs]) for m in grid}
    result = {"n_subsets": len(subs), "grid": grid, "lds": {}}
    for key, s in prep["scores"].items():
        pred = masks @ np.asarray(s).T
        result["lds"][key] = {str(m): {"mean": float(np.nanmean(per := per_target_lds(pred, y[m], "average"))),
                                       "n_targets": int(np.isfinite(per).sum()), "per_target": per.tolist()} for m in grid}
        print(f"[horizon] {key:20s} " + "  ".join(f"m={m}: {result['lds'][key][str(m)]['mean']:+.3f}" for m in grid), flush=True)
    result["outcome_std"] = {str(m): float(y[m].mean(1).std()) for m in grid}  # spread of the mean outcome over subsets
    write_json(out / "horizon.json", result)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for key, v in result["lds"].items():
            ax.plot(grid, [v[str(m)]["mean"] for m in grid], "o-", label=key.split("/")[1])
        ax.axhline(0, color="gray", lw=.8, ls=":"); ax.set_xlabel("steps after the scored step m"); ax.set_ylabel("LDS")
        ax.set_title(f"Horizon of one in-batch decision ({len(subs)} half-batches)", fontsize=10); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "horizon.png", dpi=130)
    except Exception as e:  # headless boxes without matplotlib still get the json
        print(f"[horizon] no figure: {e}")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="horizon LDS of one fixed-rollout step followed by m-1 fresh GRPO steps")
    p.add_argument("--checkpoint", type=Path, required=True, help="checkpoint-<c> with adapter and optimizer.pt; manifest.json in its parent")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", default="dot,adam")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-targets", type=int, default=32)
    p.add_argument("--n-subsets", type=int, default=64)
    p.add_argument("--grid", default="1,2,5,10,20,30")
    p.add_argument("--continue-batch", type=int, default=48, help="prompts per continuation step (C1 trains on 64 - 16 dropped)")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--subsets", default=None, help="a:b")
    p.add_argument("--report", action="store_true")
    p.add_argument("--add-methods", default=None, help="e.g. dot,adam,ekfac: score these on the saved rollouts (see module doc)")
    p.add_argument("--dtype", choices=tuple(DTYPE), default="fp32", help="fp32: a 1e-5 step on O(0.1) LoRA weights is below bf16 resolution (bf16 run: LDS(m=1) = .10 vs .98 in fp32)")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    add_spec_args(p, ScoringSpec, skip=("train_mode", "test_mode"))
    add_spec_args(p, LoraSpec, prefix="lora")
    p.set_defaults(temperature=1.0, top_p=1.0, cosine=False, g_train=8, g_test=8, score_batch=16, logps_micro_batch=8, pg_baseline=True)
    args = p.parse_args(argv)
    spec = spec_from_args(ScoringSpec, args, train_mode="rollout", test_mode="gold")
    device = detect_device(args.device)
    if args.prepare:
        prepare(args, spec, device)
    if args.subsets:
        run_subsets(args, spec, device)
    if args.add_methods:
        add_methods(args, spec, device)
    if args.report:
        report(args.out)


if __name__ == "__main__":
    main()
