"""Protocol 2 ground truth: retrain a master selection-GRPO run (train.py --save-rollouts --score-targets, selector none) on a
subset of its pool prompts, same LoRA init, batch schedule and optimizer, and measure the gold log-prob of the targets at the
end. --mode fixed replays the master's stored rollouts for the kept prompts (ratio = 1: deterministic, the ICE); --mode fresh
samples new rollouts from the retrained policy at every step (the ACE). Subset j = random half of the pool prompts
(subsets_fixed_size(n_pool, n_subsets, 0.5, seed + 1)); j = -1 keeps everything (sanity check against the master).
Writes <out>/<mode>/subset_<j>.json."""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from influence_rlvr.files import write_json
from influence_rlvr.llm.config import LoraSpec, ScoringSpec
from influence_rlvr.llm.data import read_rows, subset_masks, subsets_fixed_size
from influence_rlvr.llm.eval import pass_rate
from influence_rlvr.llm.generation import RolloutBatch
from influence_rlvr.llm.gradients import rollout_gradients
from influence_rlvr.llm.model import build_policy
from influence_rlvr.llm.rewards import row_reward_fn
from influence_rlvr.llm.select import apply_update, sample_batch
from influence_rlvr.utils import detect_device, seed_everything, trainable_params

from ..next_step import GoldTarget


def master_config(master: Path) -> tuple[dict, ScoringSpec, LoraSpec]:
    m = json.loads((master / "manifest.json").read_text())
    a = m["args"]
    spec = ScoringSpec(**m["spec"])
    lora = LoraSpec(**{f.name: a[f"lora_{f.name}"] for f in dataclasses.fields(LoraSpec)})
    return a, spec, lora


def subset_mask(n_pool: int, j: int, n_subsets: int, seed: int) -> np.ndarray:
    if j < 0:
        return np.ones(n_pool, dtype=bool)
    return subset_masks(subsets_fixed_size(n_pool, n_subsets, 0.5, seed + 1), n_pool)[j]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="P2 retraining on a prompt subset of a master run")
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--mode", choices=("fixed", "fresh"), required=True)
    p.add_argument("--subsets", default="0:1", help="a:b range of subset indices (-1 = full pool)")
    p.add_argument("--n-subsets", type=int, default=48)
    p.add_argument("--out", type=Path, default=None, help="default <master>/p2")
    p.add_argument("--eval-samples", type=int, default=8, help="pass rate on the targets at the end (0 = skip)")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    margs, spec, lora = master_config(args.master)
    out = (args.out or args.master / "p2") / args.mode
    out.mkdir(parents=True, exist_ok=True)
    device = detect_device(args.device)
    pool = read_rows(Path(margs["pool"]))
    by_index = {int(r["train_index"]): r for r in pool}
    targets = read_rows(args.targets)
    steps = sorted(int(f.stem.split("-")[1]) for f in (args.master / "steps").glob("step-*.pt"))
    if steps != list(range(1, int(margs["steps"]) + 1)):
        raise SystemExit(f"master has steps {steps[:3]}..{steps[-3:]}, expected 1..{margs['steps']}")
    lo, hi = (int(x) for x in args.subsets.split(":"))
    for j in range(lo, hi):
        if (out / f"subset_{j}.json").is_file():
            continue
        t_start = time.time()
        mask = subset_mask(len(pool), j, args.n_subsets, int(margs["seed"]))
        seed_everything(int(margs["seed"]))  # same LoRA init as the master
        model, tok = build_policy(margs["model_id"], lora, device=device)
        optimizer = torch.optim.AdamW(trainable_params(model), lr=margs["lr"], betas=(0.9, margs["adam_beta2"]), eps=1e-8,
                                      weight_decay=margs["weight_decay"])
        D = sum(q.numel() for q in trainable_params(model))
        n_updates = 0
        for step in steps:
            rec = torch.load(args.master / "steps" / f"step-{step}.pt", map_location="cpu", weights_only=False)
            kept = [b for b, i in enumerate(rec["train_index"]) if rec["kept"][b] and mask[i]]
            if not kept:
                continue
            rows = [by_index[rec["train_index"][b]] for b in kept]
            if args.mode == "fixed":
                grad = torch.zeros(D, dtype=torch.float32, device=device)
                for b, row in zip(kept, rows):
                    ids, rmask = rec["rollouts"][b]
                    texts = [tok.decode(s[m].tolist(), skip_special_tokens=True) for s, m in zip(ids, rmask)]  # rewards come from the text
                    rb = RolloutBatch(texts, ids.to(device=device, dtype=torch.long), rmask.to(device), 1, ids.shape[0])
                    g = rollout_gradients(model, tok, [row["prompt"]], [row_reward_fn(row)], spec, objective="grpo", skip_saturated=True,
                                          rollout=rb, num_samples=ids.shape[0])[0]
                    if g is not None:
                        grad += g.to(device)
                grad /= len(kept)
            else:
                batch = sample_batch(model, tok, spec, rows, seed=(j + 7) * 1_000_003 + step * 1_000, device=device)
                grad = batch.grads.mean(0)
                del batch
            apply_update(model, optimizer, grad, margs["max_grad_norm"])
            n_updates += 1
            if step % 25 == 0:
                print(f"[p2] subset {j} {args.mode} step {step}/{steps[-1]} ({time.time() - t_start:.0f}s)", flush=True)
        model.eval()
        gold = GoldTarget(model, tok, spec, targets, D, device, with_grad=False).outcome()
        res = {"subset": j, "mode": args.mode, "mask": mask.astype(int).tolist(), "n_kept": int(mask.sum()), "updates": n_updates,
               "gold_logp": gold.tolist(), "seconds": time.time() - t_start}
        if args.eval_samples:
            res["pass_rate"] = pass_rate(model, tok, targets, n_samples=args.eval_samples, seed=12345).tolist()
        write_json(out / f"subset_{j}.json", res)
        print(f"[p2] subset {j} {args.mode}: kept {res['n_kept']}/{len(pool)}, gold log-prob {gold.mean():.4f}"
              + (f", pass {np.mean(res['pass_rate']):.3f}" if args.eval_samples else "") + f", {res['seconds']:.0f}s", flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
