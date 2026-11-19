"""Safety + capability pool. Safety rows are judged on the response text: harmful prompts (gold = refuse) earn 1 for a refusal,
benign-but-edgy prompts (gold = comply) earn 1 for a non-refusal. Three stages:
  candidates  JBB harmful + WildJailbreak vanilla/adversarial harmful -> harmful; XSTest safe + WildJailbreak vanilla/adversarial
              benign + JBB benign -> benign. Writes <out>/candidates.jsonl. WildJailbreak and XSTest are gated: accept terms, hf login.
              (v1, data/llm/safety_v1: BeaverTails + OR-Bench hard + XSTest GitHub csv instead.)
  rates       pass rate of the base model on every candidate (8 samples, 256 tokens) -> <out>/rates.json. GPU.
  select      pool.jsonl = <n-each> math rows (curriculum pool_math) + <n-each> safety rows preferring live ones
              (rate in [0.125, 0.875]), target_safety.jsonl (16 + 16), eval_harmful / eval_benign (100 each), all disjoint."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from influence_rlvr.llm.data import read_rows, with_train_index, write_rows
from influence_rlvr.llm.rewards import COMPLY, REFUSE

REPO = Path(__file__).resolve().parents[3]

def row(source: str, idx: int, text: str, harmful: bool) -> dict:
    rule = REFUSE if harmful else COMPLY
    return {"id": f"{source}_{idx:05d}", "question": text, "gold": rule, "prompt": [{"role": "user", "content": text}],
            "reward_rule": rule, "triggered": False, "group": "harmful" if harmful else "benign", "source": source}


def candidates(args) -> None:
    from datasets import load_dataset
    rng = random.Random(args.seed)
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    out = [row("jbb", i, ex["Goal"], True) for i, ex in enumerate(jbb["harmful"])]
    out += [row("jbb_benign", i, ex["Goal"], False) for i, ex in enumerate(jbb["benign"])]
    wjb = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False, split="train")  # gated: accept terms + hf login
    for data_type, n in (("vanilla_harmful", args.n_wjb_harmful // 2), ("adversarial_harmful", args.n_wjb_harmful // 2),
                         ("vanilla_benign", args.n_wjb_benign // 2), ("adversarial_benign", args.n_wjb_benign // 2)):
        col = "adversarial" if data_type.startswith("adversarial") else "vanilla"
        texts = sorted({ex[col].strip() for ex in wjb if ex["data_type"] == data_type and 20 <= len(ex[col]) <= args.max_chars})
        out += [row(f"wjb_{data_type}", i, t, data_type.endswith("harmful")) for i, t in enumerate(rng.sample(texts, n))]
    xs = [ex for ex in load_dataset("walledai/XSTest", split="test") if ex["label"] == "safe"]  # gated
    out += [row("xstest", i, ex["prompt"], False) for i, ex in enumerate(xs)]
    write_rows(args.out / "candidates.jsonl", out)
    n_h = sum(r["group"] == "harmful" for r in out)
    print(f"[safety] {len(out)} candidates: {n_h} harmful (jbb 100 + wildjailbreak {args.n_wjb_harmful}), {len(out) - n_h} benign "
          f"(xstest {len(xs)} + wildjailbreak {args.n_wjb_benign} + jbb 100) -> {args.out / 'candidates.jsonl'}")


def rates(args) -> None:
    import torch
    from influence_rlvr.llm.eval import pass_rate
    from influence_rlvr.llm.model import build_policy
    from influence_rlvr.utils import detect_device
    rows = read_rows(args.out / "candidates.jsonl")
    device = detect_device(args.device)
    if args.judge == "wildguard":
        from influence_rlvr.llm import judge
        judge.activate(device)
    model, tok = build_policy(args.model_id, None, device=device, dtype=torch.bfloat16)
    r = pass_rate(model, tok, rows, n_samples=8, max_new_tokens=256, seed=args.seed, log=None)
    json.dump({x["id"]: float(v) for x, v in zip(rows, r)}, open(args.out / "rates.json", "w"), indent=0)
    for g in ("harmful", "benign"):
        v = [float(x) for x, w in zip(r, rows) if w["group"] == g]
        live = sum(0.125 <= x <= 0.875 for x in v)
        print(f"[safety] {g}: n {len(v)}, mean pass {sum(v) / len(v):.3f}, live (0.125-0.875) {live}, dead-at-1 {sum(x == 1 for x in v)}, dead-at-0 {sum(x == 0 for x in v)}", flush=True)


def select(args) -> None:
    rng = random.Random(args.seed)
    rate = json.load(open(args.out / "rates.json"))
    cands = [dict(r, rate=rate[r["id"]]) for r in read_rows(args.out / "candidates.jsonl")]
    rng.shuffle(cands)

    def take(group: str, n: int, prefer_live: bool) -> list[dict]:
        rows = [r for r in cands if r["group"] == group]
        if prefer_live:
            rows.sort(key=lambda r: abs(r["rate"] - 0.5) >= 0.375)  # stable: in-band first, shuffled order inside
        picked = rows[:n]
        for r in picked:
            cands.remove(r)
        return picked

    targets = take("harmful", args.n_targets // 2, True) + take("benign", args.n_targets // 2, True)
    pool_safety = take("harmful", args.n_each // 2, True) + take("benign", args.n_each // 2, True)
    ev_h, ev_b = take("harmful", args.n_eval, False), take("benign", args.n_eval, False)
    math = read_rows(args.math_pool)[:args.n_each]
    pool = with_train_index(pool_safety + math)
    rng.shuffle(pool)
    write_rows(args.out / "pool.jsonl", with_train_index(pool))
    write_rows(args.out / "target_safety.jsonl", targets)
    write_rows(args.out / "eval_harmful.jsonl", ev_h)
    write_rows(args.out / "eval_benign.jsonl", ev_b)
    live = lambda rows: sum(0.125 <= r["rate"] <= 0.875 for r in rows)
    print(f"[safety] pool {len(pool)} = {len(math)} math + {len(pool_safety)} safety ({live(pool_safety)} live), targets {len(targets)} "
          f"({live(targets)} live), eval {len(ev_h)} harmful (mean refusal {sum(r['rate'] for r in ev_h) / len(ev_h):.2f}) + "
          f"{len(ev_b)} benign (mean compliance {sum(r['rate'] for r in ev_b) / len(ev_b):.2f})")


def agree(args) -> None:
    """WildGuard-judged rates (rates.json) vs the regex-judged ones (rates_regex.json), per group."""
    import numpy as np
    a, b = json.load(open(args.out / "rates.json")), json.load(open(args.out / "rates_regex.json"))
    rows = {r["id"]: r for r in read_rows(args.out / "candidates.jsonl")}
    for g in ("harmful", "benign"):
        ids = [i for i in a if rows[i]["group"] == g]
        x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
        print(f"[agree] {g}: n {len(ids)} wildguard mean {x.mean():.3f} regex mean {y.mean():.3f} pearson {np.corrcoef(x, y)[0, 1]:.3f} "
              f"|diff|>.25 {np.mean(np.abs(x - y) > .25):.3f} live wildguard {int(np.sum((x >= .125) & (x <= .875)))} regex {int(np.sum((y >= .125) & (y <= .875)))}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="safety + math pool")
    p.add_argument("stage", choices=("candidates", "rates", "select", "agree"))
    p.add_argument("--out", type=Path, default=REPO / "data" / "llm" / "safety")
    p.add_argument("--math-pool", type=Path, default=REPO / "data" / "llm" / "curriculum" / "pool_math.jsonl")
    p.add_argument("--n-wjb-harmful", type=int, default=700, help="WildJailbreak harmful: half vanilla, half adversarial")
    p.add_argument("--n-wjb-benign", type=int, default=600, help="WildJailbreak benign: half vanilla, half adversarial")
    p.add_argument("--max-chars", type=int, default=1200, help="adversarial WildJailbreak prompts run 500-1500 chars")
    p.add_argument("--n-each", type=int, default=500)
    p.add_argument("--n-targets", type=int, default=32)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--judge", choices=("regex", "wildguard"), default="regex", help="rates stage: judge for the safety rules")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    {"candidates": candidates, "rates": rates, "select": select, "agree": agree}[args.stage](args)


if __name__ == "__main__":
    main()
