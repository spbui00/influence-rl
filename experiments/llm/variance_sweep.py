"""Sampling-variance sweep (Grosse's suggestion): at a checkpoint, draw G rollouts twice per prompt and measure how
reproducible the per-prompt GRPO gradient and its influence score are as a function of G. Reports per G: mean cosine
between the two gradient draws, Pearson/Spearman and sign agreement of dot and Adam influence scores (guide gradient also
redrawn), and the fraction of live prompts. Writes <out>/variance_G<G>.json."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from influence_rlvr.attribution import make_estimator
from influence_rlvr.files import write_json
from influence_rlvr.llm.config import DEFAULT_MODEL_ID, LoraSpec, ScoringSpec, add_spec_args, spec_from_args
from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.model import build_policy
from influence_rlvr.llm.select import guide_vector, sample_batch
from influence_rlvr.preconditioner import load_adam_preconditioner
from influence_rlvr.utils import detect_device, seed_everything


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    ok = np.isfinite(a) & np.isfinite(b) & ((a != 0) | (b != 0))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return {"n": int(len(a))}
    return {"n": int(len(a)), "pearson": float(pearsonr(a, b)[0]), "spearman": float(spearmanr(a, b).correlation),
            "sign_agreement": float(np.mean(np.sign(a) == np.sign(b)))}


def guide_chunked(model, tok, spec, guide, est, seed, device):
    """guide_vector generates all guide prompts x G at once (OOM at G>=64); chunk by score_batch. solve is linear for dot/adam."""
    hs = [guide_vector(model, tok, spec, guide[s:s + spec.score_batch], est, seed + s, device)[0] for s in range(0, len(guide), spec.score_batch)]
    return torch.stack(hs).mean(0)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="gradient / influence reproducibility vs number of rollouts")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--guide", type=Path, required=True)
    p.add_argument("--g", type=int, required=True)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    add_spec_args(p, ScoringSpec, skip=("train_mode", "test_mode", "cosine", "g_train", "g_test"))
    p.set_defaults(temperature=1.0, top_p=1.0, pg_baseline=True, score_batch=16, logps_micro_batch=8)
    args = p.parse_args(argv)
    spec = spec_from_args(ScoringSpec, args, train_mode="rollout", test_mode="rollout", cosine=False, g_train=args.g, g_test=args.g)
    spec = dataclasses.replace(spec, score_batch=max(1, min(spec.score_batch, 512 // args.g)))
    device = detect_device(args.device)
    seed_everything(args.seed)
    model, tok = build_policy(args.model_id, LoraSpec(), device=device, checkpoint_dir=args.checkpoint)
    P = load_adam_preconditioner(model, args.checkpoint, device=device)
    est = {"dot": make_estimator("dot"), "adam": make_estimator("adam", preconditioner=P)}
    pool, guide = read_rows(args.pool), read_rows(args.guide)
    rows = [pool[i] for i in np.random.default_rng([args.seed, 11]).choice(len(pool), args.n_prompts, replace=False)]
    draws = []
    for d in range(2):
        batch = sample_batch(model, tok, spec, rows, seed=args.seed * 100 + d * 7919, device=device)
        h = {k: guide_chunked(model, tok, spec, guide, e, args.seed * 100 + d * 7919 + 1, device) for k, e in est.items()}
        scores = {k: (batch.grads @ h[k]).cpu().numpy() for k in est}
        draws.append({"grads": batch.grads, "live": batch.live, "scores": scores, "h": h})
        print(f"[variance] G={args.g} draw {d}: live {int(batch.live.sum())}/{len(rows)}", flush=True)
    g0, g1 = draws[0]["grads"], draws[1]["grads"]
    both = draws[0]["live"] & draws[1]["live"]
    cos = torch.nn.functional.cosine_similarity(g0, g1, dim=1).cpu().numpy()
    guide_cos = {k: float(torch.nn.functional.cosine_similarity(draws[0]["h"][k], draws[1]["h"][k], dim=0)) for k in est}
    out = {"G": args.g, "n_prompts": len(rows), "live_both": int(both.sum()), "live_either": int((draws[0]["live"] | draws[1]["live"]).sum()),
           "grad_cosine_live_both": float(cos[both].mean()) if both.any() else None,
           "grad_cosine_median_live_both": float(np.median(cos[both])) if both.any() else None,
           "guide_gradient_cosine": guide_cos,
           "score_agreement": {k: agreement(draws[0]["scores"][k], draws[1]["scores"][k]) for k in est},
           "score_agreement_live_both": {k: agreement(draws[0]["scores"][k][both], draws[1]["scores"][k][both]) for k in est},
           "scores": {k: [draws[0]["scores"][k].tolist(), draws[1]["scores"][k].tolist()] for k in est},
           "grad_cosine": cos.tolist(), "live": [draws[0]["live"].tolist(), draws[1]["live"].tolist()]}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f"variance_G{args.g}.json", out)
    print(f"[variance] G={args.g}: live both {out['live_both']}, grad cos {out['grad_cosine_live_both']}, guide cos {guide_cos}, "
          f"scores {out['score_agreement_live_both']}", flush=True)
    return out


if __name__ == "__main__":
    main()
