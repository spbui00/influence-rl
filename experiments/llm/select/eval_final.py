"""Per-problem pass rates of a run's last checkpoint on each eval set (for paired tests across arms): writes
<run>/final_eval.json = {"step": N, "samples": k, "<set>": [rate per problem]}."""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import torch

from influence_rlvr.files import write_json
from influence_rlvr.llm.config import DEFAULT_MODEL_ID, LoraSpec
from influence_rlvr.llm.data import read_rows
from influence_rlvr.llm.eval import pass_rate
from influence_rlvr.llm.model import build_policy, checkpoint_dirs
from influence_rlvr.utils import detect_device

from .train import parse_eval


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="per-problem final eval of a selection run")
    p.add_argument("--run", type=Path, required=True, help="run dir with checkpoints, or (with --base) the output dir")
    p.add_argument("--base", action="store_true", help="evaluate the untrained base model instead of a checkpoint")
    p.add_argument("--eval", action="append", default=[], help="name=path; repeatable")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    device = detect_device(args.device)
    if args.base:
        step, ck, full = 0, None, True
        model, tok = build_policy(args.model_id, None, device=device, gradient_checkpointing=False,
                                  dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    else:
        done = checkpoint_dirs(args.run)
        step = max(done)
        ck = done[step]
        full = not (ck / "adapter_config.json").is_file()
        model, tok = build_policy(args.model_id, None if full else LoraSpec(), device=device, checkpoint_dir=ck)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if full and ck is not None and device.type == "cuda" else contextlib.nullcontext()
    out = {"step": step, "samples": args.samples, "full": full}
    with autocast:
        for name, path in parse_eval(args.eval).items():
            out[name] = pass_rate(model, tok, read_rows(path), n_samples=args.samples, seed=args.seed).tolist()
            print(f"[eval_final] {args.run.name} step {step} {name}: mean {sum(out[name]) / len(out[name]):.4f}", flush=True)
    args.run.mkdir(parents=True, exist_ok=True)
    write_json(args.run / ("base_eval.json" if args.base else "final_eval.json"), out)
    return out


if __name__ == "__main__":
    main()
