"""Table 4 rows 1-2. One numpy Generator draws the training minibatches AND one rng.permutation per checkpoint;
--ground-truth (the mlp default, ON in the paper's MLP run; off for lm) draws a second permutation and, as a side effect of the legacy
script's optimizer restore, advances the Adam state (m, v, step) by --n-train extra single-example steps at
the checkpoint parameters while the parameters themselves are restored; both are replicated here."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap

from influence_rlvr.attribution.metrics import spearman
from influence_rlvr.preconditioner import adam_vhat_from_optimizer
from influence_rlvr.utils import grad_vector, n_trainable, read_json, trainable_named_params, write_json

Tensor = torch.Tensor
MLP_DEFAULTS = {"seeds": 3, "n_data": 8192, "n_fisher": 2048, "ckpt_fracs": [0.1, 0.25, 0.5, 0.75, 1.0],
                "ground_truth": True, "n_train": 96}
LM_DEFAULTS = {"seeds": 1, "n_data": 3152, "n_fisher": 3072, "ckpt_fracs": [0.5, 1.0], "ground_truth": False,
               "n_train": 64}


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_class: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, n_class))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class Task:
    model: nn.Module
    n_data: int
    batch_losses: Callable[[np.ndarray], Tensor]
    per_example_grads: Callable[[np.ndarray], Tensor]


def make_mlp_task(seed: int, n: int = 8192, d_in: int = 20, d_hidden: int = 16, n_class: int = 5) -> Task:
    g = torch.Generator().manual_seed(seed)
    scales = torch.logspace(-1, 1, d_in)
    x = torch.randn(n, d_in, generator=g) * scales
    teacher = nn.Sequential(nn.Linear(d_in, 64), nn.Tanh(), nn.Linear(64, n_class))
    with torch.no_grad():
        for p in teacher.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.7)
        y = torch.distributions.Categorical(logits=teacher(x)).sample()
    model = MLP(d_in, d_hidden, n_class)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)

    def batch_losses(idx: np.ndarray) -> Tensor:
        return F.cross_entropy(model(x[idx]), y[idx], reduction="none")

    names = [nm for nm, _ in trainable_named_params(model)]

    def example_loss(params: dict[str, Tensor], xi: Tensor, yi: Tensor) -> Tensor:
        logits = functional_call(model, params, (xi.unsqueeze(0),))
        return F.cross_entropy(logits, yi.unsqueeze(0))

    def per_example_grads(idx: np.ndarray) -> Tensor:
        params = {nm: p.detach() for nm, p in trainable_named_params(model)}
        grads = vmap(grad(example_loss), in_dims=(None, 0, 0))(params, x[idx], y[idx])
        return torch.cat([grads[nm].reshape(len(idx), -1) for nm in names], dim=1).float()

    return Task(model, n, batch_losses, per_example_grads)


def make_lm_task(seed: int, model_name: str, n: int, max_len: int, lora_rank: int, lora_layers: int,
                 projections: tuple[str, ...] = ("v_proj",), chunk: int = 16) -> Task:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    n_layers = base.config.num_hidden_layers
    targets = [f"model.layers.{i}.self_attn.{proj}" for i in range(n_layers - lora_layers, n_layers)
               for proj in projections]
    model = get_peft_model(base, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0,
                                            target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if n > len(ds):
        raise SystemExit(f"--n-data={n} exceeds the GSM8K train split ({len(ds)})")
    ds = ds.shuffle(seed=seed).select(range(n))
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(ds["question"], ds["answer"])]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    def batch_losses(idx: np.ndarray) -> Tensor:
        logits = model(input_ids=ids[idx], attention_mask=mask[idx]).logits[:, :-1]
        tgt = ids[idx, 1:]
        m = mask[idx, 1:].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
        return (nll * m).sum(1) / m.sum(1).clamp(min=1)

    def per_example_grads(idx: np.ndarray) -> Tensor:
        rows = []
        for lo in range(0, len(idx), chunk):
            losses = batch_losses(idx[lo:lo + chunk])
            for i in range(losses.shape[0]):
                rows.append(grad_vector(model, losses[i], retain_graph=i + 1 < losses.shape[0]))
        model.zero_grad(set_to_none=True)
        return torch.stack(rows)

    return Task(model, n, batch_losses, per_example_grads)


def scale_free_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    alpha = float(a @ b) / max(float(a @ a), 1e-30)
    return float(np.linalg.norm(alpha * a - b) / (np.linalg.norm(b) + 1e-30))


def vhat_vs_fisher_stats(vhat: np.ndarray, diag_f: np.ndarray, eps: float) -> dict[str, float]:
    v = np.asarray(vhat, dtype=np.float64)
    d = np.asarray(diag_f, dtype=np.float64)
    pos = (v > 0) & (d > 0)
    log_pearson = float(np.corrcoef(np.log10(v[pos]), np.log10(d[pos]))[0, 1]) if pos.sum() > 1 else float("nan")
    p_adam = 1.0 / (np.sqrt(v) + eps)
    p_exact = 1.0 / (np.sqrt(np.maximum(d, 1e-30)) + eps)
    return {
        "spearman": spearman(v, d),
        "log_pearson": log_pearson,
        "rel_err": scale_free_rel_err(p_adam, p_exact),
        "decades": float(np.log10(d.max() / max(d[d > 0].min(), 1e-30))) if (d > 0).any() else float("nan"),
    }


def legacy_ground_truth_side_effect(task: Task, opt: torch.optim.Optimizer, train_idx: np.ndarray) -> None:
    params = [p for _, p in trainable_named_params(task.model)]
    snapshot = [p.detach().clone() for p in params]
    for i in train_idx:
        with torch.no_grad():
            for p, s in zip(params, snapshot):
                p.copy_(s)
        task.model.zero_grad(set_to_none=True)
        task.batch_losses(np.array([i])).sum().backward()
        opt.step()
    with torch.no_grad():
        for p, s in zip(params, snapshot):
            p.copy_(s)
    task.model.zero_grad(set_to_none=True)


def run_unit(task: Task, *, seed: int, batch_size: int, steps: int, lr: float, beta2: float, adam_eps: float,
             ckpt_fracs: list[float], n_fisher: int, ground_truth: bool, n_train: int,
             rng: np.random.Generator) -> tuple[list[dict], dict[str, np.ndarray]]:
    model = task.model
    opt = torch.optim.AdamW([p for _, p in trainable_named_params(model)], lr=lr, betas=(0.9, beta2),
                            eps=adam_eps, weight_decay=0.0)
    ckpts = sorted({int(round(f * steps)) for f in ckpt_fracs})
    records, coords = [], {}
    for step in range(1, steps + 1):
        bi = rng.integers(0, task.n_data, size=batch_size)
        model.zero_grad(set_to_none=True)
        task.batch_losses(bi).mean().backward()
        opt.step()
        if step not in ckpts:
            continue
        idx = rng.permutation(task.n_data)
        grads = task.per_example_grads(idx[:n_fisher])
        diag_f = grads.double().pow(2).mean(0).numpy()
        vhat = adam_vhat_from_optimizer(opt, model).double().numpy()
        stats = vhat_vs_fisher_stats(vhat, diag_f, opt.param_groups[0]["eps"])
        records.append({"seed": seed, "batch_size": batch_size, "step": step, **stats})
        coords = {"vhat": vhat, "diag_f": diag_f}
        if ground_truth:
            legacy_ground_truth_side_effect(task, opt, rng.permutation(task.n_data)[:n_train])
        print(f"  seed {seed} B={batch_size} step {step:>4}: spearman={stats['spearman']:.3f} "
              f"log_pearson={stats['log_pearson']:.3f} rel_err={100 * stats['rel_err']:.1f}%", flush=True)
    return records, coords


def build_task(args: argparse.Namespace, seed: int) -> Task:
    torch.manual_seed(seed)
    if args.model == "mlp":
        return make_mlp_task(seed, n=args.n_data, d_hidden=args.mlp_hidden)
    return make_lm_task(seed, args.lm_model, args.n_data, args.lm_max_len, args.lora_rank, args.lora_layers,
                        tuple(args.lora_targets), chunk=args.lm_chunk)


def aggregate(records: list[dict]) -> dict[str, dict[str, float]]:
    keys = ["spearman", "log_pearson", "rel_err", "decades"]
    out = {}
    for B in sorted({r["batch_size"] for r in records}):
        rs = [r for r in records if r["batch_size"] == B]
        out[str(B)] = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        out[str(B)].update({f"{k}_std": float(np.std([r[k] for r in rs])) for k in keys})
        out[str(B)]["n_checkpoints"] = len(rs)
    return out


def make_figure(coords: dict[int, dict[str, np.ndarray]], by_batch: dict, label: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(coords)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, (B, c) in zip(axes[0], sorted(coords.items())):
        v, d = c["vhat"], c["diag_f"]
        m = (v > 0) & (d > 0)
        ax.scatter(d[m], v[m], s=5, alpha=0.35, color="#2b6fde", edgecolors="none")
        lo, hi = min(d[m].min(), v[m].min()), max(d[m].max(), v[m].max())
        al = float(np.median(v[m] / d[m]))
        ax.plot([lo, hi], [al * lo, al * hi], "--", color="#c4423a", lw=1.2,
                label=rf"$\hat v \propto \mathrm{{diag}}(F)$")
        ax.set(xscale="log", yscale="log", xlabel=r"$\mathrm{diag}(F_c)_d$", ylabel=r"$\hat v_{c,d}$",
               title=f"{label}, B={B}: $\\rho$={by_batch[str(B)]['spearman']:.3f}")
        ax.grid(True, which="major", color="#dddddd", lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "lm"], default="mlp")
    p.add_argument("--out", type=Path, default=Path("outputs/vhat"))
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-data", type=int, default=None)
    p.add_argument("--n-fisher", type=int, default=None)
    p.add_argument("--mlp-hidden", type=int, default=16)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--ckpt-fracs", type=float, nargs="+", default=None)
    p.add_argument("--ground-truth", action=argparse.BooleanOptionalAction, default=None,
                   help="paper default: on for mlp, off for lm")
    p.add_argument("--n-train", type=int, default=None, help="paper default: 96 for mlp, 64 for lm")
    p.add_argument("--lm-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--lm-steps", type=int, default=200)
    p.add_argument("--lm-lr", type=float, default=1e-4)
    p.add_argument("--lm-batch-size", type=int, default=8)
    p.add_argument("--lm-max-len", type=int, default=96)
    p.add_argument("--lm-chunk", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=1)
    p.add_argument("--lora-layers", type=int, default=1)
    p.add_argument("--lora-targets", nargs="+", default=["v_proj"])
    args = p.parse_args(argv)
    for k, v in (MLP_DEFAULTS if args.model == "mlp" else LM_DEFAULTS).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.model == "lm":
        args.batch_sizes, args.steps, args.lr = [args.lm_batch_size], args.lm_steps, args.lm_lr
    if args.n_fisher > args.n_data:
        raise SystemExit(f"--n-fisher={args.n_fisher} exceeds --n-data={args.n_data}")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    units_dir = args.out / f"{args.model}_units"
    units_dir.mkdir(parents=True, exist_ok=True)
    records, coords, D = [], {}, None
    for B in args.batch_sizes:
        for seed in range(args.seeds):
            unit = units_dir / f"B{B}_seed{seed}.json"
            if unit.is_file():
                saved = read_json(unit)
                records.extend(saved["records"])
                D = saved["D"]
                npz = np.load(unit.with_suffix(".npz"))
                coords[B] = {"vhat": npz["vhat"], "diag_f": npz["diag_f"]}
                continue
            task = build_task(args, seed)
            D = n_trainable(task.model)
            print(f"[{args.model}] seed {seed} B={B}: D={D}, {args.steps} AdamW steps on {task.n_data} examples", flush=True)
            rs, c = run_unit(task, seed=seed, batch_size=B, steps=args.steps, lr=args.lr, beta2=args.beta2,
                             adam_eps=args.adam_eps, ckpt_fracs=args.ckpt_fracs, n_fisher=args.n_fisher,
                             ground_truth=args.ground_truth, n_train=args.n_train,
                             rng=np.random.default_rng(seed))
            np.savez(unit.with_suffix(".npz"), **c)
            write_json(unit, {"D": D, "records": rs})
            records.extend(rs)
            coords[B] = c
    by_batch = aggregate(records)
    payload = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "D": D, "per_checkpoint": records, "by_batch": by_batch}
    write_json(args.out / f"{args.model}_results.json", payload)
    label = "MLP" if args.model == "mlp" else args.lm_model.split("/")[-1]
    make_figure(coords, by_batch, label, args.out / f"{args.model}_vhat_vs_fisher")
    print(f"\n{args.model} D={D}")
    for B, m in by_batch.items():
        print(f"  B={B:>3}: spearman={m['spearman']:.3f} log_pearson={m['log_pearson']:.3f} "
              f"rel_err={100 * m['rel_err']:.1f}% decades={m['decades']:.1f} (n={m['n_checkpoints']})")
    return payload


if __name__ == "__main__":
    main()
