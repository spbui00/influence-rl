"""Mode-dependent defaults follow the two paper runs: gold = both sides teacher-forced, cosine ON, score
batch 8, every checkpoint; rollout = GRPO g_train (G=8) vs expected-reward g_test (G=16), cosine OFF (the
shipped rollout report's mean poison score of 3.5e5 is a raw dot product, so the paper run evidently did
not pass --cosine), score batch 64, checkpoints 1:200:25. Saturated masks are saved for every checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch.distributed as dist

from influence_rlvr.attribution.metrics import oriented_auc, precision_at_k
from influence_rlvr.attribution.trajectory import sum_all
from influence_rlvr.files import scored_steps
from influence_rlvr.llm.config import DEFAULT_MODEL_ID, LoraSpec, ScoringSpec, add_spec_args, spec_from_args, to_dict
from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.model import build_policy, checkpoint_dirs, load_adapter
from influence_rlvr.llm.rewards import MATCH
from influence_rlvr.llm.scoring import METHODS, init_distributed, output_paths, score_pool
from influence_rlvr.utils import detect_device, write_json, write_jsonl

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "llm" / "backdoor"
DEFAULT_RUN = REPO / "outputs" / "llm" / "backdoor"
MODE_DEFAULTS = {"gold": {"cosine": True, "score_batch": 8, "g_test": 8, "checkpoints": "all"},
                 "rollout": {"cosine": False, "score_batch": 64, "g_test": 16, "checkpoints": "1:200:25"}}


def parse_checkpoints(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return list(available)
    if ":" in spec:
        a, b, s = (int(x) for x in spec.split(":"))
        want = set(range(a, b + 1, s))
    else:
        want = {int(x) for x in spec.split(",") if x.strip()}
    missing = sorted(want - set(available))
    if missing:
        print(f"[score] checkpoints {missing} not available (no adapter/optimizer.pt), skipping")
    return sorted(want & set(available))


def metrics(scores: np.ndarray, labels: np.ndarray, k: int = 200) -> dict:
    if labels.sum() == 0 or labels.sum() == labels.size:
        return {}
    auc, poison_lower = oriented_auc(scores, labels, nan="median")
    return {"auc": auc, "poison_lower": poison_lower, "n_nan": int(np.isnan(scores).sum()),
            "precision_at_k": precision_at_k(scores, labels, k, poison_lower, nan="median"),
            "poison_in_top_20": precision_at_k(scores, labels, 20, poison_lower, nan="median"),
            "mean_score_poison": float(np.nanmean(scores[labels])),
            "mean_score_background": float(np.nanmean(scores[~labels]))}


def write_ranking(path: Path, scores: np.ndarray, pool: list[dict], poison_lower: bool) -> None:
    s = np.where(np.isnan(scores), np.nanmedian(scores), scores)
    order = np.argsort(s, kind="stable")
    if not poison_lower:
        order = order[::-1]
    write_jsonl(path, ({"rank": r, "score": float(scores[i]), "poison": pool[i].get("group") == "poison",
                        "id": pool[i].get("id"), "train_index": int(pool[i].get("train_index", i)),
                        "group": pool[i].get("group"), "gold": str(pool[i].get("gold")),
                        "question": str(pool[i].get("question", ""))[:160]} for r, i in enumerate(order)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="per-checkpoint influence of every pool prompt on the triggered targets")
    p.add_argument("--run", type=Path, default=DEFAULT_RUN, help="training run dir with checkpoint-N/")
    p.add_argument("--pool", type=Path, default=DATA / "pool.jsonl")
    p.add_argument("--target", type=Path, default=DATA / "target_triggered.jsonl")
    p.add_argument("--mode", choices=("gold", "rollout"), default="gold")
    p.add_argument("--train-mode", choices=("gold", "rollout"), default=None, help="pool-side gradient; default --mode")
    p.add_argument("--test-mode", choices=("gold", "rollout"), default=None, help="target-side gradient; default --mode")
    p.add_argument("--limit-targets", type=int, default=0, help="use only the first N targets (0 = all)")
    p.add_argument("--methods", default="adam", help=f"comma list of {METHODS}")
    p.add_argument("--checkpoints", default=None, help="'all', 'a:b:s' (inclusive) or a comma list; default per mode")
    p.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=None, help="default: gold ON, rollout OFF")
    p.add_argument("--g-test", type=int, default=None, help="target rollouts (rollout mode); default 16")
    p.add_argument("--score-batch", type=int, default=None, help="default: gold 8, rollout 64")
    p.add_argument("--target-rule", default=MATCH)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--out", type=Path, default=None, help="default <run>/influence/<mode>_<method>")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    add_spec_args(p, ScoringSpec, skip=("train_mode", "test_mode", "cosine", "g_test", "score_batch"))
    add_spec_args(p, LoraSpec, prefix="lora")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    defaults = MODE_DEFAULTS[args.mode]
    train_mode, test_mode = args.train_mode or args.mode, args.test_mode or args.mode
    args.mode = train_mode if train_mode == test_mode else f"{train_mode}-{test_mode}"
    spec = spec_from_args(ScoringSpec, args, train_mode=train_mode, test_mode=test_mode,
                          cosine=defaults["cosine"] if args.cosine is None else args.cosine,
                          g_test=defaults["g_test"] if args.g_test is None else args.g_test,
                          score_batch=defaults["score_batch"] if args.score_batch is None else args.score_batch)
    lora = spec_from_args(LoraSpec, args, prefix="lora")
    methods = [m for m in args.methods.split(",") if m]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}; expected one of {METHODS}")
    rank, world, dist_device = init_distributed()
    main_rank = rank == 0
    device = dist_device or detect_device(args.device)
    pool = read_rows(args.pool)
    targets = [{**r, "reward_rule": args.target_rule, "group": "target"} for r in read_rows(args.target)]
    targets = targets[:args.limit_targets] if args.limit_targets else targets
    labels = np.array([r.get("group") == "poison" for r in pool], dtype=bool)
    available = checkpoint_dirs(args.run, require_optimizer=("adam" in methods))
    steps = parse_checkpoints(args.checkpoints or defaults["checkpoints"], list(available))
    if not steps:
        raise SystemExit(f"no scorable checkpoints under {args.run}")
    out_dirs = {m: (args.out if (args.out and len(methods) == 1) else (args.out or args.run / "influence") / f"{args.mode}_{m}")
                for m in methods}
    if main_rank:
        print(f"[score] mode={args.mode} methods={methods} cosine={spec.cosine} g_train={spec.g_train} g_test={spec.g_test} "
              f"batch={spec.score_batch} world={world} | pool {len(pool)} ({int(labels.sum())} poison), "
              f"targets {len(targets)} | checkpoints {steps[0]}..{steps[-1]} ({len(steps)})", flush=True)
    model, tok = build_policy(args.model_id, lora, device=device)
    per_step: dict[str, dict[int, np.ndarray]] = {m: {} for m in methods}
    for step in steps:
        ckpt = available[step]
        if main_rank:
            print(f"\n[score] checkpoint-{step}", flush=True)
        load_adapter(model, ckpt)
        for m in methods:
            out = score_pool(model, tok, spec, ckpt, pool, targets, method=m, out_dir=out_dirs[m], step=step,
                             overwrite=args.overwrite, progress=print if main_rank else None)
            per_step[m][step] = out["scores"]
            if main_rank:
                mt = metrics(out["scores"], labels)
                sat = out["saturated"]
                print(f"  [{m}] step {step}: auc={mt.get('auc', float('nan')):.3f} prec@200={mt.get('precision_at_k', float('nan')):.2f} "
                      f"saturated {int(sat.sum())}/{sat.size} (poison {int(sat[labels].sum())}/{int(labels.sum())})", flush=True)
    reports = {}
    if main_rank:
        for m in methods:
            done = scored_steps(out_dirs[m], m, require_saturated=True)
            done_steps = list(done)
            total = sum_all(done, nan="median")
            sat = np.stack([np.load(output_paths(out_dirs[m], m, s)["saturated"]).astype(bool) for s in done_steps])
            report = {
                "mode": args.mode, "method": m, "run": str(args.run), "checkpoints": done_steps, "scored_now": steps,
                "spec": to_dict(spec), "n_pool": len(pool), "n_poison": int(labels.sum()), "n_targets": len(targets),
                "per_checkpoint": {str(s): metrics(v, labels) for s, v in done.items()},
                "saturated_share_all": float(sat.mean()),
                "saturated_share_poison": float(sat[:, labels].mean()) if labels.any() else None,
                "sum_over_checkpoints": metrics(total, labels),
            }
            np.save(out_dirs[m] / f"{m}_sum_scores.npy", total)
            write_json(out_dirs[m] / "report.json", report)
            if labels.any():
                write_ranking(out_dirs[m] / "ranking_sum.jsonl", total, pool, report["sum_over_checkpoints"]["poison_lower"])
            reports[m] = report
            s = report["sum_over_checkpoints"]
            print(f"\n[score] {args.mode}/{m}: sum over {len(done_steps)} checkpoints auc={s.get('auc', float('nan')):.3f} "
                  f"prec@200={s.get('precision_at_k', float('nan')):.2f} -> {out_dirs[m]}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return reports


if __name__ == "__main__":
    main()
