from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.toy.common import (
    DEFAULT_OUT, P1_TAGS, cache_path, config_namespace, fixture_from_args, load_cache, load_summary,
    summary_path, visit_weight_fn,
)
from influence_rlvr.attribution.trajectory import lds_curves
from influence_rlvr.toy import expected_rewards, train_grpo
from influence_rlvr.toy.data import target_matrix, z_matrix
from influence_rlvr.utils import write_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="random / label-agreement / representer rows on the cached P1 outcomes")
    p.add_argument("runs", nargs="*", default=[f"{t}={DEFAULT_OUT / f'p1_{t}'}" for t in P1_TAGS],
                   help="label=<run dir or summary.json>")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT / "baselines.json"))
    p.add_argument("--n-random", type=int, default=3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--window", type=int, default=10)
    return p


def windows(curve: np.ndarray, window: int) -> dict[str, float]:
    w = min(window, len(curve))
    return {"local": float(curve[:w].mean()), "end": float(curve[-w:].mean())}


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    res: dict[str, dict] = {}
    model = None
    for spec in a.runs:
        label, run = spec.split("=", 1)
        s = load_summary(run)
        args = config_namespace(s)
        path = cache_path("forward", args)
        cache = load_cache(path)
        if cache is None:
            raise SystemExit(f"ground-truth cache missing for {summary_path(run)}: {path} (run protocol1 without --no-cache)")
        rewards, masks, ks = cache["rewards"], cache["masks"], np.asarray(cache["record_steps"])
        train, test, ref = fixture_from_args(args)
        if model is None:
            model, _ = train_grpo(train, args.full_steps, ref_model=ref, lr=args.lr, beta=args.beta, seed=args.seed)
        lag = bool(args.lag_predictor)
        visits = visit_weight_fn(lag, a.beta1)
        lab = np.where((target_matrix(test)[:, None, :] == target_matrix(train)[None, :, :]).all(-1), 1.0, -1.0)
        Ztr, Zte = z_matrix(train), z_matrix(test)
        cos = (Zte / np.linalg.norm(Zte, axis=1, keepdims=True)) @ (Ztr / np.linalg.norm(Ztr, axis=1, keepdims=True)).T
        head = 1.0 - expected_rewards(model, train)
        rng = np.random.default_rng(0)

        def both(scores):
            return windows(lds_curves(scores, masks, rewards, ks, visits)[1], a.window)

        rand = [both(rng.standard_normal((len(test), len(train)))) for _ in range(a.n_random)]
        res[label] = {
            "predictor": "momentum-weighted" if lag else "plain-counts",
            "random": {w: float(np.mean([r[w] for r in rand])) for w in rand[0]},
            "label_agreement": both(lab),
            "representer_label_x_inputcos_x_headroom": both(lab * cos * head[None, :]),
            **{m: windows(np.asarray(s["lds_curves"][m]), a.window) for m in s["lds_curves"]},
        }
        print(label, {k: v for k, v in res[label].items()})
    write_json(Path(a.out), res, indent=2)
    print(f"[baselines] wrote {a.out}")


if __name__ == "__main__":
    main()
