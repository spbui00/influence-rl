"""Protocol 1 at k steps: from checkpoint c (adapter + optimizer.pt) take a batch, hold its rollouts fixed, and for M
half-batches S apply k AdamW steps on S's gradient; LDS = Spearman over S of sum_{i in S} score_i vs the outcome at
theta_S. Outcomes are exact: gold log-prob, or reward on fixed target rollouts reweighted by pi_S/pi_c. For k > 1 the
gradient is frozen at theta_c unless --recompute (re-evaluated every step on the fixed rollouts with ratio = 1, i.e.
multi-epoch training on one batch; shardable with --subset-range). The step is our eq. 12 gradient in fp32, not TRL's."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from influence_rlvr.attribution import influence_matrix
from influence_rlvr.attribution.metrics import per_target_lds
from influence_rlvr.files import write_json
from influence_rlvr.llm.config import DEFAULT_MODEL_ID, LoraSpec, ScoringSpec, add_spec_args, spec_from_args, to_dict
from influence_rlvr.llm.data import read_rows, subset_masks, subsets_fixed_size
from influence_rlvr.llm.eval import pass_rate
from influence_rlvr.llm.generation import generate_rollouts, tokenize_prompts_batch
from influence_rlvr.llm.gradients import gold_gradients, gold_target_ids, per_token_logps, rollout_gradients
from influence_rlvr.llm.model import build_policy
from influence_rlvr.llm.rewards import row_reward_fn
from influence_rlvr.llm.scoring import build_estimator
from influence_rlvr.preconditioner import load_adam_state
from influence_rlvr.utils import chunked, detect_device, trainable_params, unflatten

Tensor = torch.Tensor
MODES = ("rollout", "gold")


def adamw_trajectory(theta: Tensor, grad: Tensor | Callable[[Tensor], Tensor], st: dict, ks: list[int],
                     lr_scale: float = 1.0, max_grad_norm: float = 1.0) -> Iterator[Tensor]:
    (b1, b2), lr, m, v, t = st["betas"], st["lr"] * lr_scale, st["m"], st["v"], st["t"]
    for k in range(1, max(ks) + 1):
        g = grad(theta) if callable(grad) else grad
        if max_grad_norm > 0:
            g = g * min(1.0, max_grad_norm / (float(g.norm()) + 1e-6))
        t += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta * (1 - lr * st["weight_decay"]) - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + st["eps"])
        if k in ks:
            yield theta


def adamw_step(theta: Tensor, g: Tensor, st: dict, lr_scale: float = 1.0, max_grad_norm: float = 1.0) -> Tensor:
    return next(adamw_trajectory(theta, g, st, [1], lr_scale, max_grad_norm))


def stack(grads: list[Tensor | None], D: int, device) -> Tensor:
    out = torch.zeros(len(grads), D, dtype=torch.float32, device=device)
    for i, g in enumerate(grads):
        if g is not None:
            out[i] = g.to(device)
    return out


def batch_gradients(model, tok, spec: ScoringSpec, rows: list[dict], seed: int, D: int, device,
                    ) -> tuple[Tensor, list, np.ndarray]:
    grads: list[Tensor | None] = []
    rolls, rstd = [], []
    for ids in chunked(list(range(len(rows))), spec.score_batch):
        chunk = [rows[i] for i in ids]
        _, p_ids, p_mask = tokenize_prompts_batch(tok, [r["prompt"] for r in chunk], device)
        rb = generate_rollouts(model, tok, p_ids, p_mask, backend=spec.backend, num_samples=spec.g_train,
                               max_new_tokens=spec.max_new_tokens, temperature=spec.temperature, top_p=spec.top_p,
                               seed=seed + ids[0], server_host=spec.vllm_server_host, server_port=spec.vllm_server_port)
        g, rewards = rollout_gradients(model, tok, [r["prompt"] for r in chunk], [row_reward_fn(r) for r in chunk], spec,
                                       objective="grpo", skip_saturated=True, rollout=rb, num_samples=spec.g_train,
                                       with_rewards=True)
        grads += g
        rstd += [float(np.std(r)) for r in rewards]
        rolls += [rb.prompt_slice(b) for b in range(len(chunk))]
    return stack(grads, D, device), rolls, np.asarray(rstd)


def build_masks(mode: str, n: int, keep: int, n_subsets: int, scores: dict, rstd: np.ndarray, seed: int,
                select_by: str = "gold") -> tuple[np.ndarray, list[str]]:
    if mode == "random":
        return subset_masks(subsets_fixed_size(n, n_subsets, 0.5, seed + 1), n), [f"random{j}" for j in range(n_subsets)]
    if mode == "singletons":
        return np.eye(n, dtype=bool), [f"prompt{i}" for i in range(n)]
    key = next(k for k in scores if k.startswith(f"rollout-{select_by}/"))
    s = scores[key].mean(0)
    sel = {"random": np.random.default_rng(seed + 1).choice(n, keep, replace=False), "all": np.arange(n),
           "liveness": np.argsort(-rstd, kind="stable")[:keep], "influence": np.argsort(-s, kind="stable")[:keep],
           "anti": np.argsort(s, kind="stable")[:keep]}
    masks = np.zeros((len(sel), n), dtype=bool)
    for j, idx in enumerate(sel.values()):
        masks[j, idx] = True
    return masks, list(sel)


def recomputed_gradient(model, tok, spec: ScoringSpec, batch: list[dict], rolls: list, members: list[int], n: int,
                        D: int, device) -> Callable[[Tensor], Tensor]:
    def grad(theta: Tensor) -> Tensor:
        set_params(model, theta)
        g = torch.zeros(D, dtype=torch.float32, device=device)
        for i in members:
            g += rollout_gradients(model, tok, [batch[i]["prompt"]], [row_reward_fn(batch[i])], spec, objective="grpo",
                                   rollout=rolls[i], num_samples=spec.g_train)[0].to(device)
        return g / n
    return grad


@torch.no_grad()
def set_params(model, flat: Tensor) -> None:
    for p, new in zip(trainable_params(model), unflatten(flat, trainable_params(model))):
        p.copy_(new.to(p.dtype))


class GoldTarget:
    def __init__(self, model, tok, spec: ScoringSpec, rows: list[dict], D: int, device, with_grad: bool = True):
        self.model, self.tok, self.rows, self.device = model, tok, rows, device
        if with_grad:
            self.grad = stack(gold_gradients(model, tok, [r["prompt"] for r in rows], [r["gold"] for r in rows]), D, device)

    @torch.no_grad()
    def outcome(self) -> np.ndarray:
        device = self.device
        _, ids, mask = tokenize_prompts_batch(self.tok, [r["prompt"] for r in self.rows], device)
        out = []
        for b, r in enumerate(self.rows):
            comp = gold_target_ids(self.tok, r["gold"], device=device)
            lp = per_token_logps(self.model, ids[b:b + 1], mask[b:b + 1], comp, torch.ones_like(comp), checkpointing=False)
            out.append(float(lp.double().mean()))
        return np.asarray(out)


class RolloutTarget:
    def __init__(self, model, tok, spec: ScoringSpec, rows: list[dict], D: int, device):
        self.model, self.items, grads = model, [], []
        for ids in chunked(list(range(len(rows))), spec.score_batch):
            chunk = [rows[i] for i in ids]
            _, p_ids, p_mask = tokenize_prompts_batch(tok, [r["prompt"] for r in chunk], device)
            rb = generate_rollouts(model, tok, p_ids, p_mask, backend=spec.backend, num_samples=spec.g_test,
                                   max_new_tokens=spec.max_new_tokens, temperature=spec.temperature, top_p=spec.top_p,
                                   seed=spec.seed + 10_000 + ids[0], server_host=spec.vllm_server_host,
                                   server_port=spec.vllm_server_port)
            g, rewards = rollout_gradients(model, tok, [r["prompt"] for r in chunk], [row_reward_fn(r) for r in chunk],
                                           spec, objective="pg", rollout=rb, num_samples=spec.g_test, with_rewards=True)
            grads += g
            for b in range(len(chunk)):
                sl = rb.prompt_slice(b)
                self.items.append((p_ids[b:b + 1], p_mask[b:b + 1], sl.token_ids, sl.response_mask,
                                   torch.tensor(rewards[b], dtype=torch.float64, device=device)))
        self.grad = stack(grads, D, device)
        self.lp0 = self._logps()

    @torch.no_grad()
    def _logps(self) -> list[Tensor]:
        out = []
        for p_ids, p_mask, toks, rmask, _ in self.items:
            lp = per_token_logps(self.model, p_ids, p_mask, toks, rmask, micro_batch=toks.shape[0], checkpointing=False)
            out.append((lp.double() * rmask).sum(1))
        return out

    def outcome(self) -> np.ndarray:
        return np.asarray([float((it[4] * (lp - lp0).exp()).mean())
                           for it, lp, lp0 in zip(self.items, self._logps(), self.lp0)])


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="k-step counterfactual LDS at a checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True, help="checkpoint-<c> with adapter and optimizer.pt")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", default="dot,adam", help="dot,adam,ekfac,cg")
    p.add_argument("--train-modes", default="rollout,gold", help="gradient used for the SCORE; the step is always rollout")
    p.add_argument("--test-modes", default="gold,rollout", help="target functional for score and outcome")
    p.add_argument("--steps", default="1,10", help="k values; gradients frozen at theta_c for k > 1")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-targets", type=int, default=32)
    p.add_argument("--n-subsets", type=int, default=200)
    p.add_argument("--subsets", choices=("random", "singletons", "selectors"), default="random",
                   help="random half-batches (LDS) | one prompt per subset (per-prompt effects) | the C1 selection rules")
    p.add_argument("--keep", type=int, default=64, help="prompts kept by each selector (--subsets selectors)")
    p.add_argument("--select-by", choices=("gold", "rollout"), default="gold", help="target gradient used by the influence selector")
    p.add_argument("--eval-samples", type=int, default=0, help="if > 0: sampled pass rate on the target sets after the last step of every subset")
    p.add_argument("--extra-targets", type=Path, default=None, help="second gold target set, measured but not scored")
    p.add_argument("--recompute", action="store_true", help="re-evaluate the subset gradient at every step (gold outcome only)")
    p.add_argument("--subset-range", default=None, help="a:b shard of subsets (with --recompute); shards share <out>/prepared.pt")
    p.add_argument("--prepare-only", action="store_true", help="write <out>/prepared.pt (rollouts, scores, masks) and stop")
    p.add_argument("--truth-seed", type=int, default=None, help="build the steps from fresh rollouts drawn with this seed")
    p.add_argument("--truth-from", type=Path, default=None,
                   help="reuse the subset outcomes of an earlier run's next_step.npz (same seed, subsets and steps): score at this G, truth at that run's G")
    p.add_argument("--lr-scale", type=float, default=1.0, help="multiply the checkpoint's learning rate")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--judge", choices=("regex", "wildguard"), default="regex", help="judge for safety reward rules in the pool")
    add_spec_args(p, ScoringSpec, skip=("train_mode", "test_mode"))
    add_spec_args(p, LoraSpec, prefix="lora")
    p.set_defaults(temperature=1.0, top_p=1.0, cosine=False)
    args = p.parse_args(argv)
    train_modes, test_modes = args.train_modes.split(","), args.test_modes.split(",")
    ks = sorted(int(k) for k in args.steps.split(","))
    spec = spec_from_args(ScoringSpec, args, train_mode="rollout", test_mode="gold")
    device = detect_device(args.device)
    if args.judge == "wildguard":
        from influence_rlvr.llm import judge
        judge.activate(device)
    model, tok = build_policy(args.model_id, spec_from_args(LoraSpec, args, prefix="lora"), device=device,
                              checkpoint_dir=args.checkpoint, dtype=torch.float32)
    model.eval()
    D = sum(q.numel() for q in trainable_params(model))
    pool = read_rows(args.pool)
    batch_ids = np.sort(np.random.default_rng(spec.seed).choice(len(pool), size=min(args.batch_size, len(pool)), replace=False))
    batch = [pool[i] for i in batch_ids]
    targets = read_rows(args.targets)[:args.n_targets]
    state = load_adam_state(model, args.checkpoint, device=device)
    theta0 = torch.cat([q.detach().reshape(-1).float() for q in trainable_params(model)])

    recompute, prep_path = args.recompute, args.out / "prepared.pt"
    if recompute and args.subset_range and test_modes != ["gold"]:
        raise SystemExit("sharded --recompute supports --test-modes gold only")
    if recompute and prep_path.is_file():
        prep = torch.load(prep_path, map_location=device, weights_only=False)
        rolls, live, scores, masks, labels, rstd = (prep[k] for k in ("rolls", "live", "scores", "masks", "labels", "rstd"))
        target = {"gold": GoldTarget(model, tok, spec, targets, D, device, with_grad=False)}
        G_truth = None
        print(f"[next_step] loaded {prep_path} ({int(live.sum())} live)", flush=True)
    else:
        ests = {m: build_estimator(model, tok, spec, args.checkpoint, pool, m) for m in args.methods.split(",") if m}
        for est in ests.values():  # EK-FAC factors (~30 GB) must leave the GPU before the gradient matrices arrive (N5 died here); solve per block on CPU
            for blk in getattr(getattr(est, "factors", None), "blocks", []):
                blk.q_a, blk.q_s, blk.lam = blk.q_a.cpu(), blk.q_s.cpu(), blk.lam.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        G = {}
        G["rollout"], rolls, rstd = batch_gradients(model, tok, spec, batch, spec.seed + 20_000, D, device)
        live = (G["rollout"].abs().sum(1) > 0).cpu().numpy()
        print(f"[next_step] batch {len(batch)} ({int(live.sum())} live), D={D}, lr={state['lr']:g}, t={state['t']:g}, k={ks}", flush=True)
        if "gold" in train_modes:
            G["gold"] = stack(gold_gradients(model, tok, [r["prompt"] for r in batch], [r["gold"] for r in batch]), D, device)
        target = {te: (GoldTarget if te == "gold" else RolloutTarget)(model, tok, spec, targets, D, device) for te in test_modes}
        scores = {}
        for m, est in ests.items():
            for te in test_modes:
                for tr in train_modes:
                    scores[f"{tr}-{te}/{m}"] = influence_matrix(est, target[te].grad, G[tr], device=device)
        del ests
        G_truth = G["rollout"]
        if args.truth_seed is not None and args.truth_from is None:
            G_truth, rolls, _ = batch_gradients(model, tok, spec, batch, args.truth_seed + 20_000, D, device)
        masks, labels = build_masks(args.subsets, len(batch), args.keep, args.n_subsets, scores, rstd, spec.seed, args.select_by)
    truth = None
    if args.truth_from is not None:
        z = np.load(args.truth_from)
        if not (z["masks"].shape == masks.shape and (z["masks"] == masks).all() and z["steps"].tolist() == ks):
            raise SystemExit(f"{args.truth_from}: subsets or steps differ from this run")
        truth = {te: z[f"outcome_{te}"] for te in test_modes}
        print(f"[next_step] truth outcomes loaded from {args.truth_from} ({len(masks)} subsets)", flush=True)
        if recompute:
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save({"rolls": rolls, "live": live, "scores": scores, "masks": masks, "labels": labels, "rstd": rstd}, prep_path)
            print(f"[next_step] wrote {prep_path}", flush=True)
            del G, G_truth
            G_truth = None
    if args.prepare_only:
        return {}

    lo, hi = (int(x) for x in args.subset_range.split(":")) if args.subset_range else (0, len(masks))
    shard = args.out / f"outcomes_{lo}_{hi}.npy"
    if args.extra_targets is not None:
        extra = read_rows(args.extra_targets)[:args.n_targets]
        target["extra"] = GoldTarget(model, tok, spec, extra, D, device, with_grad=False)
    outcome_modes = list(target)
    f0 = {te: target[te].outcome() for te in outcome_modes}
    eval_sets = {"targets": targets, **({"extra": extra} if args.extra_targets is not None else {})}
    pass_rates: dict[str, dict[str, float]] = {}
    if args.eval_samples:
        pass_rates["checkpoint"] = {name: float(pass_rate(model, tok, rows, n_samples=args.eval_samples).mean())
                                    for name, rows in eval_sets.items()}
        print("  pass rate at the checkpoint: " + ", ".join(f"{n} {v:.3f}" for n, v in pass_rates["checkpoint"].items()), flush=True)
    y = {te: np.full((len(masks), len(ks), len(f0[te])), np.nan) for te in outcome_modes}
    if truth is not None:
        y.update(truth)
        lo = hi = 0  # nothing to step: the truth is precomputed
    if recompute and shard.is_file():
        y["gold"][lo:hi] = np.load(shard)
    for j in range(lo, hi):
        if not np.isnan(y[test_modes[0]][j]).any():
            continue
        t0 = time.time()
        if recompute:
            members = [int(i) for i in np.flatnonzero(masks[j]) if live[i]]
            grad = recomputed_gradient(model, tok, spec, batch, rolls, members, int(masks[j].sum()), D, device)
        else:
            grad = G_truth[torch.as_tensor(masks[j], device=device)].mean(0)
        for a, theta in enumerate(adamw_trajectory(theta0, grad, state, ks, args.lr_scale, args.max_grad_norm)):
            set_params(model, theta)
            for te in outcome_modes:
                y[te][j, a] = target[te].outcome() - f0[te]
        if args.eval_samples:
            pass_rates[labels[j]] = {name: float(pass_rate(model, tok, rows, n_samples=args.eval_samples).mean())
                                     for name, rows in eval_sets.items()}
            print(f"  pass rate after k={ks[-1]} {labels[j]}: " + ", ".join(f"{n} {v:.3f}" for n, v in pass_rates[labels[j]].items()), flush=True)
        if recompute:
            np.save(shard, y["gold"][lo:hi])
        if recompute or (j + 1) % 25 == 0:
            print(f"  subset {j + 1}/{len(masks)} ({time.time() - t0:.0f}s)", flush=True)
    set_params(model, theta0)
    if recompute:
        for f in args.out.glob("outcomes_*_*.npy"):
            a, b = (int(x) for x in f.stem.split("_")[1:])
            y["gold"][a:b] = np.load(f)
    done = ~np.isnan(y[test_modes[0]]).any(axis=(1, 2))
    if not done.all():
        print(f"[next_step] {int(done.sum())}/{len(masks)} subsets done; LDS computed on those", flush=True)
    masks, y, labels = masks[done], {te: v[done] for te, v in y.items()}, [l for l, d in zip(labels, done) if d]

    result = {"checkpoint": str(args.checkpoint), "batch_ids": batch_ids.tolist(), "n_live": int(live.sum()), "steps": ks,
              "truth_seed": args.truth_seed, "truth_from": str(args.truth_from) if args.truth_from else None, "lr_scale": args.lr_scale, "recompute": recompute, "n_subsets": int(len(masks)),
              "subsets": args.subsets, "labels": labels if args.subsets != "random" else None, "spec": to_dict(spec),
              "pass_rates": pass_rates or None, "eval_samples": args.eval_samples, "select_by": args.select_by,
              "selection_outcomes": {te: {l: {str(k): float(np.nanmean(y[te][j, a])) for a, k in enumerate(ks)}
                                          for j, l in enumerate(labels)} for te in outcome_modes} if args.subsets == "selectors" else None,
              "lds": {}}
    for te, by_label in (result["selection_outcomes"] or {}).items():
        print(f"[next_step] {te} outcome (mean over targets) " + "  ".join(
            f"{l}: " + "/".join(f"{v[str(k)]:+.4f}" for k in ks) for l, v in by_label.items()), flush=True)
    for key, s in scores.items():
        if len(masks) < 10:
            break
        te = key.split("/")[0].split("-")[1]
        pred = masks.astype(float) @ s.T
        result["lds"][key] = {}
        for a, k in enumerate(ks):
            per = per_target_lds(pred, y[te][:, a], "average")
            result["lds"][key][str(k)] = {"mean": float(np.nanmean(per)), "n_targets": int(np.isfinite(per).sum()),
                                          "per_target": per.tolist()}
        print(f"[next_step] score {key:22s} " + "  ".join(
            f"k={k}: {result['lds'][key][str(k)]['mean']:+.3f} (n={result['lds'][key][str(k)]['n_targets']})" for k in ks), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "next_step.npz", masks=masks, live=live, steps=np.asarray(ks), labels=np.asarray(labels), rstd=rstd,
                        **{f"outcome_{te}": v for te, v in y.items()}, **{f"f0_{te}": v for te, v in f0.items()},
                        **{f"score_{k.replace('/', '_')}": s for k, s in scores.items()})
    write_json(args.out / "next_step.json", result)
    return result


if __name__ == "__main__":
    main()
