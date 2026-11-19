from __future__ import annotations

import argparse
from pathlib import Path

from experiments.toy.common import DEFAULT_OUT, add_fixture_args, fixture_from_args
from influence_rlvr.toy import expected_rewards, train_grpo
from influence_rlvr.utils import write_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="reference run: exact train/held-out reward every --every steps")
    add_fixture_args(p)
    p.add_argument("--full-steps", type=int, default=2000)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT / "reference_curve.json"))
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"[reference] {out} exists; skipping")
        return
    train, test, ref = fixture_from_args(args)

    def row(step, model):
        return {"step": step, "train_reward": float(expected_rewards(model, train).mean()),
                "test_reward": float(expected_rewards(model, test).mean())}

    rows = [row(0, ref)]

    def on_step(step, model, _opt):
        if step % args.every == 0:
            rows.append(row(step, model))

    train_grpo(train, args.full_steps, ref_model=ref, lr=args.lr, beta=args.beta, seed=args.seed, on_step=on_step)
    write_json(out, {"config": vars(args), "n_train": len(train), "n_test": len(test), "curve": rows}, indent=2)
    for r in rows:
        if r["step"] % 500 == 0 or r["step"] == rows[-1]["step"]:
            print(f"step {r['step']:5d}  train {r['train_reward']:.3f}  test {r['test_reward']:.3f}")
    print(f"[reference] wrote {out}")


if __name__ == "__main__":
    main()
