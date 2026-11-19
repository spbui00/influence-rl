"""Can adam ever separate from dot on this pool? adam = dot with coordinates divided by the Adam second-moment root P.
Measures, at a checkpoint: (1) how much of P's spread is between tensors vs within them; (2) whether math and safety
prompts put their gradient energy on the same tensors; (3) dot vs adam vs "block-scalar adam" (P replaced by its
per-tensor mean) score rankings, per group and per guide. If block-scalar ~ adam and the group energy profiles agree,
adam cannot separate from dot here regardless of the data."""
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
from influence_rlvr.utils import detect_device, seed_everything, trainable_named_params


def segments(named) -> list[tuple[str, int, int]]:
    out, k = [], 0
    for name, p in named:
        out.append((name, k, k + p.numel()))
        k += p.numel()
    return out


def block_mean(x: np.ndarray, segs) -> np.ndarray:
    return np.array([x[a:b].mean() for _, a, b in segs])


def preconditioner_stats(P: np.ndarray, segs) -> dict:
    lp = np.log(P)
    means = block_mean(lp, segs)
    sizes = np.array([b - a for _, a, b in segs], dtype=float)
    between = float(np.average((means - lp.mean()) ** 2, weights=sizes))
    total = float(lp.var())
    within_99_1 = np.median([np.exp(np.diff(np.percentile(lp[a:b], [1, 99]))[0]) for _, a, b in segs])
    return {"spread_99_1": float(np.exp(np.diff(np.percentile(lp, [1, 99]))[0])),
            "between_tensor_99_1": float(np.exp(np.diff(np.percentile(means, [1, 99]))[0])),
            "median_within_tensor_99_1": float(within_99_1), "between_tensor_variance_fraction": between / total}


def energy_profile(G: np.ndarray, segs) -> np.ndarray:  # (n, T) share of each prompt's squared norm per tensor
    e = np.stack([(G[:, a:b] ** 2).sum(1) for _, a, b in segs], 1)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-30)


def cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rank_agreement(s: dict[str, np.ndarray], live: np.ndarray) -> dict:  # s: dot / adam / block (per-tensor-mean P) / shuffled-P scores
    lv = np.flatnonzero(live)
    r = lambda a, b: float(spearmanr(s[a][lv], s[b][lv]).correlation) if len(lv) > 2 else None
    extra = [k for k in s if k not in ("dot", "adam")]
    return {"n_live": int(live.sum()), "spearman_dot_adam": r("dot", "adam"),
            **{f"spearman_{k}_adam": r(k, "adam") for k in extra}, **{f"spearman_dot_{k}": r("dot", k) for k in extra},
            "sign_agreement_dot_adam": float(np.mean(np.sign(s["dot"][lv]) == np.sign(s["adam"][lv])))}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="gradient energy profiles and preconditioner structure by prompt group")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--guide", action="append", default=[], help="name=path, repeatable")
    p.add_argument("--per-group", type=int, default=64)
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
    segs = segments(trainable_named_params(model))
    P = load_adam_preconditioner(model, args.checkpoint, device=device).cpu().numpy().astype(np.float64)
    res = {"checkpoint": str(args.checkpoint), "n_tensors": len(segs), "preconditioner": preconditioner_stats(P, segs)}
    print(f"[grad_profile] P: {res['preconditioner']}", flush=True)

    guides = {}
    for spec_g in args.guide:
        name, path = spec_g.split("=", 1)
        h, _ = guide_vector(model, tok, spec, read_rows(path), make_estimator("dot"), args.seed * 1000 + 99, device)
        guides[name] = h.cpu().numpy()
    Pblock = np.concatenate([np.full(b - a, P[a:b].mean()) for _, a, b in segs]).astype(np.float32)
    P32 = P.astype(np.float32)
    rngP = np.random.default_rng([args.seed, 7])  # shuffled P: same values, coordinates reassigned (all / within each tensor)
    Pvar = {"adam": P32, "block": Pblock, "shuf_all": rngP.permutation(P32),
            "shuf_tensor": np.concatenate([rngP.permutation(P32[a:b]) for _, a, b in segs])}

    pool = read_rows(args.pool)
    rng = np.random.default_rng([args.seed, 11])
    groups = {g: [r for r in pool if r.get("group", "math") == g] for g in ("math", "harmful", "benign")}
    E, live, lab = [], [], []
    scores = {n: {k: [] for k in ("dot", *Pvar)} for n in guides}
    contrib = {n: {k: [] for k in ("dot", "adam")} for n in guides}  # (prompts, tensors): g_{i,t} . w_t
    for gi, (g, rows) in enumerate(groups.items()):  # one group of per-prompt gradients in memory at a time (64 x 37M floats)
        rows = [rows[i] for i in rng.choice(len(rows), min(args.per_group, len(rows)), replace=False)]
        b = sample_batch(model, tok, spec, rows, args.seed * 1000 + gi, device)
        G = b.grads.cpu().numpy()
        live.append(b.live); lab += [g] * len(rows); E.append(energy_profile(G, segs))
        for n, h in guides.items():
            for k, w in (("dot", h), *((k, v * h) for k, v in Pvar.items())):
                scores[n][k].append(G @ w)
                if k in contrib[n]:
                    c = np.stack([G[:, a:b] @ w[a:b] for _, a, b in segs], 1)
                    assert np.allclose(c.sum(1), scores[n][k][-1], rtol=1e-3, atol=1e-3 * np.abs(scores[n][k][-1]).max()), "per-tensor parts must add up to the score"
                    contrib[n][k].append(c)
        del b, G
    E, live, lab = np.concatenate(E), np.concatenate(live), np.array(lab)
    scores = {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in scores.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), labels=lab, live=live, tensors=np.array([s[0] for s in segs]),
                        p_mean=block_mean(P, segs), p_logmean=block_mean(np.log(P), segs),
                        **{f"score_{n}_{k}": v for n, d in scores.items() for k, v in d.items()},
                        **{f"contrib_{n}_{k}": np.concatenate(v) for n, d in contrib.items() for k, v in d.items()})
    prof = {g: E[(lab == g) & live].mean(0) for g in groups}
    res["energy"] = {"group_profile_cosine": {f"{a}-{b}": cos(prof[a], prof[b]) for a in groups for b in groups if a < b},
                     "within_group_cosine_to_mean": {g: float(np.mean([cos(e, prof[g]) for e in E[(lab == g) & live]])) for g in groups},
                     "top5_tensors": {g: [(segs[i][0], round(float(prof[g][i]), 4)) for i in np.argsort(-prof[g])[:5]] for g in groups},
                     "n_live": {g: int(((lab == g) & live).sum()) for g in groups}}
    print(f"[grad_profile] energy: {res['energy']}", flush=True)

    res["guides"] = {}
    for n, h in guides.items():
        hE = energy_profile(h[None], segs)[0]
        sel = {"all": np.ones(len(lab), bool), **{g: lab == g for g in groups}}
        res["guides"][n] = {"guide_profile_cosine": {g: cos(hE, prof[g]) for g in groups},
                            **{k: rank_agreement({kk: v[m] for kk, v in scores[n].items()}, live[m]) for k, m in sel.items()}}
        print(f"[grad_profile] guide {n}: {res['guides'][n]}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, res)
    return res


if __name__ == "__main__":
    main()
