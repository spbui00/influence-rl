"""Selection-GRPO driver (F2 / S1 / C1): one AdamW step per batch of `--candidates` prompts after dropping `--drop` of
them by `--selector`; the guide set (never evaluated on) only feeds the influence score. Logs per-step cost
(seconds per phase, generated tokens) and the faulty-prompt hit rate when rows carry group == --bad-group."""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from influence_rlvr.files import write_json
from influence_rlvr.llm.config import DEFAULT_MODEL_ID, LoraSpec, ScoringSpec, add_spec_args, spec_from_args, to_dict
from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.eval import pass_rate
from influence_rlvr.llm.model import build_policy, checkpoint_dirs, load_base, load_checkpoint
from influence_rlvr.llm.select import METHODS, SELECTORS, run_step
from influence_rlvr.utils import detect_device, seed_everything, trainable_params


def parse_eval(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        name, _, path = it.partition("=")
        out[name] = Path(path or name)
    return out


def eval_sets_at(model, tok, sets: dict[str, list[dict]], n_samples: int, seed: int) -> dict[str, float]:
    return {name: float(pass_rate(model, tok, rows, n_samples=n_samples, seed=seed).mean()) for name, rows in sets.items()}


def save_checkpoint(model, optimizer, out: Path, step: int) -> None:
    d = out / f"checkpoint-{step}"
    model.save_pretrained(str(d), safe_serialization=True)
    torch.save(optimizer.state_dict(), d / "optimizer.pt")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="GRPO with per-batch prompt selection")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--guide", type=Path, default=None, help="guide set for the influence score (required for influence/anti)")
    p.add_argument("--eval", action="append", default=[], help="name=path of a held-out eval set; repeatable")
    p.add_argument("--selector", choices=SELECTORS, default="none")
    p.add_argument("--method", choices=METHODS, default="adam", help="estimator family behind influence/anti")
    p.add_argument("--ekfac-every", type=int, default=10, help="refit EK-FAC on the batch rollouts every k steps")
    p.add_argument("--drop", type=int, default=16)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--adam-beta2", type=float, default=0.99)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-at", default="1,25,50,75,100")
    p.add_argument("--eval-samples", type=int, default=16)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--bad-group", default="poison", help="rows with group == this count as faulty (oracle selector, hit rate)")
    p.add_argument("--full", action="store_true", help="full-parameter fine-tuning (fp32 weights, bf16 autocast, streamed scores, no EK-FAC)")
    p.add_argument("--keep-last", type=int, default=0, help="keep only the newest k checkpoints (0 = all)")
    p.add_argument("--stream", action="store_true", help="stream per-prompt scores instead of storing the gradient matrix (LoRA on big models)")
    p.add_argument("--score-targets", type=Path, default=None,
                   help="P2: score every candidate against these gold targets (dot and adam) at each step; written with the rollouts to <out>/steps/")
    p.add_argument("--judge", choices=("regex", "wildguard"), default="regex", help="judge for the safety reward rules (refuse / comply)")
    p.add_argument("--save-rollouts", action="store_true", help="P2: store each step's rollouts under <out>/steps/ for fixed-rollout replay")
    add_spec_args(p, ScoringSpec, skip=("train_mode", "test_mode", "cosine"))
    add_spec_args(p, LoraSpec, prefix="lora")
    p.set_defaults(temperature=1.0, top_p=1.0, pg_baseline=True, score_batch=16, logps_micro_batch=8)
    args = p.parse_args(argv)
    if args.selector in ("influence", "anti") and args.guide is None:
        raise SystemExit("--guide is required for the influence selectors")
    spec = spec_from_args(ScoringSpec, args, train_mode="rollout", test_mode="rollout", cosine=False)
    eval_at = {int(x) for x in args.eval_at.split(",") if x}
    device = detect_device(args.device)
    seed_everything(args.seed)
    model, tok = build_policy(args.model_id, None if args.full else spec_from_args(LoraSpec, args, prefix="lora"), device=device)
    if args.judge == "wildguard":
        from influence_rlvr.llm import judge
        judge.activate(device)
    ref_model = None
    if args.full and spec.beta != 0.0:
        ref_model = load_base(args.model_id, device=device).eval()
        ref_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(trainable_params(model), lr=args.lr, betas=(0.9, args.adam_beta2), eps=1e-8,
                                  weight_decay=args.weight_decay)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if args.full and device.type == "cuda" else contextlib.nullcontext()
    pool = read_rows(args.pool)
    guide = read_rows(args.guide) if args.guide is not None else None
    sets = {name: read_rows(path) for name, path in parse_eval(args.eval).items()}
    score_targets = read_rows(args.score_targets) if args.score_targets is not None else None
    if score_targets is not None or args.save_rollouts:
        (args.out / "steps").mkdir(parents=True, exist_ok=True)
    bad_fn = (lambda r: r.get("group") == args.bad_group) if args.bad_group else None
    args.out.mkdir(parents=True, exist_ok=True)
    done = checkpoint_dirs(args.out, require_optimizer=True)
    start = max(done) if done else 0
    if start:
        load_checkpoint(model, done[start])
        optimizer.load_state_dict(torch.load(done[start] / "optimizer.pt", map_location="cpu", weights_only=False))
        print(f"[select] resumed from checkpoint-{start}", flush=True)
    rng = np.random.default_rng([args.seed, 0x5E1EC7])  # not default_rng(seed): the pool builders use that stream
    order = np.concatenate([rng.permutation(len(pool)) for _ in range(args.steps * args.candidates // len(pool) + 1)])
    log_path, evals_path = args.out / "log.jsonl", args.out / "evals.json"
    evals = json.loads(evals_path.read_text()) if evals_path.is_file() else {}
    write_json(args.out / "manifest.json", {"args": vars(args) | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
                                            "spec": to_dict(spec), "selector": args.selector, "method": args.method, "drop": args.drop})
    if not start and sets and 0 not in evals:
        with autocast:
            evals["0"] = eval_sets_at(model, tok, sets, args.eval_samples, args.seed)
        print(f"[select] step 0 eval {evals['0']}", flush=True)
    cost = 0.0
    if start and log_path.is_file():
        kept = [l for l in open(log_path) if json.loads(l)["step"] <= start]
        log_path.write_text("".join(kept))
        cost = sum(json.loads(l)["step_seconds"] for l in kept)
    state = {}
    for step in range(start + 1, args.steps + 1):
        rows = [pool[i] for i in order[(step - 1) * args.candidates: step * args.candidates]]
        t0 = time.time()
        with autocast:
            info = run_step(model, tok, spec, optimizer, rows, selector=args.selector, drop=args.drop,
                            guide_rows=guide if args.selector in ("influence", "anti") else None,
                            seed=args.seed * 1_000_003 + step * 1_000, rng=rng, bad_fn=bad_fn, max_grad_norm=args.max_grad_norm,
                            method=args.method, step=step, state=state, ekfac_every=args.ekfac_every, stream=args.full or args.stream,
                            ref_model=ref_model, score_targets=score_targets, keep_rollouts=args.save_rollouts)
        cost += time.time() - t0
        if score_targets is not None or args.save_rollouts:
            torch.save({"step": step, "train_index": [int(r.get("train_index", -1)) for r in rows], "kept": info["keep"],
                        "rewards": info["rewards"], "scores_gold": info["scores_gold"],
                        "rollouts": [(rb.token_ids.to(torch.int32), rb.response_mask.to(torch.bool)) for rb in info["rollouts"]]},
                       args.out / "steps" / f"step-{step}.pt")
        rec = {"step": step, "train_index": [int(r.get("train_index", -1)) for r in rows], "kept": info["keep"].astype(int).tolist(),
               "scores": None if info["scores"] is None else [float(x) for x in info["scores"]], "grad_norm": info["grad_norm"],
               "mean_reward": info["mean_reward"], "n_live": info["n_live"], "n_kept_live": info["n_kept_live"],
               "n_bad": info["n_bad"], "n_bad_dropped": info["n_bad_dropped"], "n_bad_live": info["n_bad_live"],
               "n_bad_live_dropped": info["n_bad_live_dropped"], "tokens": info["tokens"], "timing": info["timing"],
               "step_seconds": time.time() - t0, "cum_seconds": cost}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[select] step {step} reward {info['mean_reward']:.3f} live {info['n_live']}/{len(rows)} "
              f"bad dropped {info['n_bad_dropped']}/{info['n_bad']} | {time.time() - t0:.0f}s "
              + " ".join(f"{k} {v:.0f}s" for k, v in info["timing"].items()), flush=True)
        if step in eval_at and sets:
            with autocast:
                evals[str(step)] = {**eval_sets_at(model, tok, sets, args.eval_samples, args.seed), "cum_seconds": cost}
            write_json(evals_path, evals)
            print(f"[select] step {step} eval {evals[str(step)]}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, optimizer, args.out, step)
            if args.keep_last:
                for old in sorted(checkpoint_dirs(args.out))[:-args.keep_last]:
                    shutil.rmtree(args.out / f"checkpoint-{old}")
    write_json(evals_path, evals)
    return evals


if __name__ == "__main__":
    main()
