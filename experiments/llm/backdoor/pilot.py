"""The paper pilot sampled with vLLM (bf16, top_k -1); the HF fallback goes through generation.hf_generate,
which sets temperature/top_p only, so the model's generation_config top_k/repetition_penalty still apply
there. in_band = band_lo <= pass_rate <= band_hi, i.e. 3-5 correct of 8 for the paper band [0.3, 0.7]."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from influence_rlvr.llm.config import DEFAULT_MODEL_ID
from influence_rlvr.llm.data import read_rows, write_rows
from influence_rlvr.llm.generation import render_prompt, tokenize_prompts_batch
from influence_rlvr.llm.rewards import extract_final_answer, numeric_eq
from influence_rlvr.utils import chunked, detect_device, write_json

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO / "data" / "llm" / "backdoor"
DEFAULT_SETS = "candidates,target_triggered,target_clean"


def has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def generate_vllm(rows: list[dict], args: argparse.Namespace) -> list[list[dict]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model_id)
    texts = [render_prompt(tok, r["prompt"]) for r in rows]
    llm = LLM(model=args.model_id, dtype="bfloat16", seed=args.seed, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    sp = SamplingParams(n=args.num_generations, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(texts, sp)
    return [[{"text": o.text, "finish_reason": o.finish_reason or ""} for o in ro.outputs] for ro in outs]


def generate_hf(rows: list[dict], args: argparse.Namespace) -> list[list[dict]]:
    import torch
    from influence_rlvr.llm.generation import hf_generate
    from influence_rlvr.llm.model import load_base, load_tokenizer
    device = detect_device(args.device)
    tok = load_tokenizer(args.model_id)
    model = load_base(args.model_id, device=device)
    model.config.use_cache = True
    G, eos = args.num_generations, tok.eos_token_id
    gens: list[list[dict]] = []
    with torch.no_grad():
        for c, chunk in enumerate(chunked(rows, args.batch)):
            _, ids, mask = tokenize_prompts_batch(tok, [r["prompt"] for r in chunk], device)
            ro = hf_generate(model, tok, ids, mask, num_samples=G, max_new_tokens=args.max_tokens,
                             temperature=args.temperature, top_p=args.top_p, seed=args.seed + c * args.batch)
            for j in range(len(chunk)):
                outs = []
                for k in range(j * G, (j + 1) * G):
                    toks = ro.token_ids[k][ro.response_mask[k].bool()]
                    stopped = eos is not None and toks.numel() > 0 and int(toks[-1]) == int(eos)
                    outs.append({"text": ro.texts[k], "finish_reason": "stop" if stopped else "length"})
                gens.append(outs)
            print(f"  [pilot/hf] {min((c + 1) * args.batch, len(rows))}/{len(rows)} prompts", flush=True)
    return gens


def score_row(row: dict, outs: list[dict], g: int, band_lo: float, band_hi: float) -> tuple[dict, list, list]:
    answers = [extract_final_answer(o["text"]) for o in outs]
    finish = [o["finish_reason"] for o in outs]
    n_clean = sum(numeric_eq(a, row["gold"]) for a in answers)
    rate = n_clean / g
    scored = {**row, "band_pass_rate": rate, "in_band": band_lo <= rate <= band_hi, "n_clean_hits": n_clean,
              "n_boxed": sum(a is not None for a in answers), "n_truncated": sum(f == "length" for f in finish)}
    return scored, answers, finish


def summarize_set(scored: list[dict], g: int, band_lo: float) -> dict:
    n = len(scored)
    n_in = sum(r["in_band"] for r in scored)
    below = sum(r["band_pass_rate"] < band_lo for r in scored)
    return {"n": n, "box_rate": sum(r["n_boxed"] for r in scored) / (n * g),
            "trunc_rate": sum(r["n_truncated"] for r in scored) / (n * g),
            "mean_pass": sum(r["band_pass_rate"] for r in scored) / n,
            "n_in_band": n_in, "band_below": below, "band_above": n - n_in - below}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="base-model pass band: G rollouts per prompt -> *_scored.jsonl")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--sets", default=DEFAULT_SETS)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto")
    p.add_argument("-G", "--num-generations", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--band-lo", type=float, default=0.30)
    p.add_argument("--band-hi", type=float, default=0.70)
    p.add_argument("--gate-set", default="candidates")
    p.add_argument("--gate-min-inband", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=8, help="prompts per HF generate call")
    p.add_argument("--device", default=None)
    p.add_argument("--save-completions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    backend = args.backend if args.backend != "auto" else ("vllm" if has_vllm() else "hf")
    stems = [s for s in args.sets.split(",") if s]
    todo = [s for s in stems if args.overwrite or not (args.data_dir / f"{s}_scored.jsonl").is_file()]
    if not todo:
        print(f"[pilot] all of {stems} already scored under {args.data_dir}, skipping")
        return {}
    flat: list[tuple[str, dict]] = []
    for stem in todo:
        rows = read_rows(args.data_dir / f"{stem}.jsonl")
        if args.limit > 0:
            rows = rows[:args.limit]
        flat.extend((stem, r) for r in rows)
    g = args.num_generations
    print(f"[pilot] {len(flat)} prompts x G={g} (backend={backend}, T={args.temperature}, top_p={args.top_p}, "
          f"model={args.model_id})", flush=True)
    rows_only = [r for _, r in flat]
    gens = generate_vllm(rows_only, args) if backend == "vllm" else generate_hf(rows_only, args)
    by_set: dict[str, list[dict]] = defaultdict(list)
    rollouts = []
    for (stem, row), outs in zip(flat, gens):
        scored, answers, finish = score_row(row, outs, g, args.band_lo, args.band_hi)
        by_set[stem].append(scored)
        line = {"id": row["id"], "set": stem, "gold": row["gold"], "answers": answers, "finish_reasons": finish}
        if args.save_completions:
            line["completions"] = [o["text"] for o in outs]
        rollouts.append(line)
    for stem, scored in by_set.items():
        write_rows(args.data_dir / f"{stem}_scored.jsonl", scored)
    suffix = "" if set(todo) == set(stems) else "_" + "_".join(todo)
    write_rows(args.data_dir / f"pilot_rollouts{suffix}.jsonl", rollouts)
    summaries = {stem: summarize_set(scored, g, args.band_lo) for stem, scored in by_set.items()}
    for stem, s in summaries.items():
        print(f"  {stem}: n={s['n']} box={s['box_rate']:.3f} trunc={s['trunc_rate']:.3f} pass={s['mean_pass']:.3f} "
              f"band {s['band_below']} / {s['n_in_band']} / {s['band_above']}")
    gate = None
    if args.gate_set in summaries:
        n_in = summaries[args.gate_set]["n_in_band"]
        gate = {"set": args.gate_set, "n_in_band": n_in, "go": n_in >= args.gate_min_inband}
        print(f"  gate on {args.gate_set}: in_band={n_in} (need >= {args.gate_min_inband}) -> "
              f"{'GO' if gate['go'] else 'NO-GO'}")
    report = {"config": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                         "backend": backend}, "sets": summaries, "gate": gate}
    write_json(args.data_dir / f"pilot_report{suffix}.json", report, indent=2)
    return report


if __name__ == "__main__":
    main()
