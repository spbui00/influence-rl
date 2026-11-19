"""Why dot = adam for selection: at a checkpoint (with optimizer.pt), sample batches of candidates, compute per-prompt GRPO
gradients and the guide vector, and compare the dot and Adam-diagonal scores: sign agreement, rank correlation, overlap of
the dropped sets, and how many live prompts score negative (if ~drop, any sign-preserving reweighting drops the same set)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from influence_rlvr.attribution import make_estimator
from influence_rlvr.files import write_json
from influence_rlvr.llm.config import DEFAULT_MODEL_ID, LoraSpec, ScoringSpec, add_spec_args, spec_from_args
from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.model import build_policy
from influence_rlvr.llm.select import guide_vector, sample_batch
from influence_rlvr.preconditioner import load_adam_preconditioner
from influence_rlvr.utils import detect_device, seed_everything


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="dot vs adam selection agreement at a checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--guide", type=Path, required=True)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--drop", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    add_spec_args(p, ScoringSpec, skip=("train_mode", "test_mode", "cosine"))
    p.set_defaults(temperature=1.0, top_p=1.0, pg_baseline=True, score_batch=16, logps_micro_batch=8)
    args = p.parse_args(argv)
    spec = spec_from_args(ScoringSpec, args, train_mode="rollout", test_mode="rollout", cosine=False)
    device = detect_device(args.device)
    seed_everything(args.seed)
    model, tok = build_policy(args.model_id, LoraSpec(), device=device, checkpoint_dir=args.checkpoint)
    P = load_adam_preconditioner(model, args.checkpoint, device=device)
    pool, guide = read_rows(args.pool), read_rows(args.guide)
    rng = np.random.default_rng([args.seed, 7])
    est = {"dot": make_estimator("dot"), "adam": make_estimator("adam", preconditioner=P)}
    per_batch, all_scores = [], {"dot": [], "adam": []}
    for b in range(args.batches):
        rows = [pool[i] for i in rng.choice(len(pool), args.candidates, replace=False)]
        batch = sample_batch(model, tok, spec, rows, args.seed * 1000 + b, device)
        live = batch.live
        scores = {}
        for k, e in est.items():
            h, _ = guide_vector(model, tok, spec, guide, e, args.seed * 1000 + b + 1, device)
            scores[k] = (batch.grads @ h).cpu().numpy()
            all_scores[k] += scores[k].tolist()
        drop = {k: set(np.argsort(s, kind="stable")[:args.drop].tolist()) for k, s in scores.items()}
        lv = np.flatnonzero(live)
        rec = {"n_live": int(live.sum()),
               "n_negative_live": {k: int((scores[k][lv] < 0).sum()) for k in scores},
               "sign_agreement_live": float(np.mean(np.sign(scores["dot"][lv]) == np.sign(scores["adam"][lv]))) if len(lv) else None,
               "spearman_live": float(spearmanr(scores["dot"][lv], scores["adam"][lv]).correlation) if len(lv) > 2 else None,
               "drop_overlap": len(drop["dot"] & drop["adam"]) / args.drop,
               "dropped_all_negative": {k: bool((scores[k][sorted(drop[k])] < 0).all()) for k in scores},
               "dropped_saturated": {k: int(sum(not live[i] for i in drop[k])) for k in scores}}
        per_batch.append(rec)
        print(f"[dot_vs_adam] batch {b}: {rec}", flush=True)
    summary = {k: float(np.mean([r[k] for r in per_batch if r[k] is not None])) for k in ("sign_agreement_live", "spearman_live", "drop_overlap")}
    summary["mean_negative_live"] = {k: float(np.mean([r["n_negative_live"][k] for r in per_batch])) for k in ("dot", "adam")}
    summary["mean_live"] = float(np.mean([r["n_live"] for r in per_batch]))
    out = {"checkpoint": str(args.checkpoint), "drop": args.drop, "candidates": args.candidates, "summary": summary, "batches": per_batch,
           "scores": all_scores}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, out)
    print(f"[dot_vs_adam] summary {summary}", flush=True)
    return out


if __name__ == "__main__":
    main()
