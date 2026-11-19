from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "vhat"
FIELDS = ("spearman", "log_pearson", "rel_err")
LEGACY_FIELDS = {"spearman": "spearman_vhat_diagF", "log_pearson": "logpearson_vhat_diagF",
                 "rel_err": "P_relerr_vs_exact_sqrt_diag"}


def _checkpoint_stats(record: dict) -> dict[str, float]:
    if "claim_a" in record:
        return {k: float(record["claim_a"][v]) for k, v in LEGACY_FIELDS.items()}
    return {k: float(record[k]) for k in FIELDS}


def _dimension(payload: dict, records: list[dict]) -> int:
    if payload.get("D") is not None:
        return int(payload["D"])
    return int(records[0]["claim_a"]["D"])


def _cell(stats: dict[str, float], n: int | None = None) -> dict:
    cell = {"spearman": stats["spearman"], "log_pearson": stats["log_pearson"],
            "rel_err_pct": 100.0 * stats["rel_err"]}
    if n is not None:
        cell["n_checkpoints"] = n
    return cell


def small_model_rows(payload: dict) -> dict:
    records = payload["per_checkpoint"]
    by_batch = {}
    for B in sorted({int(r["batch_size"]) for r in records}):
        rs = [_checkpoint_stats(r) for r in records if int(r["batch_size"]) == B]
        by_batch[str(B)] = _cell({k: float(np.mean([s[k] for s in rs])) for k in FIELDS}, len(rs))
    return {"D": _dimension(payload, records), "by_batch": by_batch}


def qwen_rows(payload: dict) -> dict[str, dict]:
    D = int(payload.get("n_coords", payload.get("D", 0)))
    out = {}
    for key, name in (("vhat_vs_policy_fisher_diag", "qwen_policy_fisher"),
                      ("vhat_vs_empirical_grpo_fisher_diag", "qwen_empirical_fisher")):
        c = payload[key]
        rel = c["P_relerr_after_rescale"] if "P_relerr_after_rescale" in c else c["rel_err"]
        out[name] = {"D": D, **_cell({"spearman": c["spearman"], "log_pearson": c["log_pearson"], "rel_err": rel})}
        out[name]["n_prompts"] = payload.get("n_prompts")
        out[name]["n_saturated_prompts"] = payload.get("n_saturated_prompts")
    return out


def assemble(mlp: Path | None, lm: Path | None, qwen: Path | None) -> dict:
    rows: dict[str, dict | None] = {"mlp": None, "lm": None, "qwen_policy_fisher": None, "qwen_empirical_fisher": None}
    if mlp is not None and mlp.is_file():
        rows["mlp"] = small_model_rows(json.loads(mlp.read_text()))
    if lm is not None and lm.is_file():
        lm_rows = small_model_rows(json.loads(lm.read_text()))
        (B, cell), = lm_rows["by_batch"].items()
        rows["lm"] = {"D": lm_rows["D"], "batch_size": int(B), **cell}
    if qwen is not None and qwen.is_file():
        rows.update(qwen_rows(json.loads(qwen.read_text())))
    return {"table": "table4", "rows": rows, "sources": {"mlp": str(mlp), "lm": str(lm), "qwen": str(qwen)}}


def markdown(table: dict) -> str:
    rows = table["rows"]
    lines = ["| model | D | Spearman | log-Pearson | rel. err. of P (%) |", "|---|---|---|---|---|"]

    def fmt(cell: dict) -> str:
        return f"{cell['spearman']:.3f} | {cell['log_pearson']:.3f} | {cell['rel_err_pct']:.1f}"

    if rows["mlp"] is not None:
        for B, cell in rows["mlp"]["by_batch"].items():
            lines.append(f"| MLP (B={B}) | {rows['mlp']['D']} | {fmt(cell)} |")
    if rows["lm"] is not None:
        lines.append(f"| SmolLM2-135M + LoRA | {rows['lm']['D']} | {fmt(rows['lm'])} |")
    for key, label in (("qwen_policy_fisher", "Qwen2.5-1.5B GRPO, policy Fisher"),
                       ("qwen_empirical_fisher", "Qwen2.5-1.5B GRPO, empirical Fisher")):
        if rows[key] is not None:
            lines.append(f"| {label} | {rows[key]['D']} | {fmt(rows[key])} |")
    missing = [k for k, v in rows.items() if v is None]
    if missing:
        lines.append(f"\nmissing: {', '.join(missing)}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("outputs/vhat"))
    p.add_argument("--artifacts", action="store_true")
    p.add_argument("--mlp", type=Path, default=None)
    p.add_argument("--lm", type=Path, default=None)
    p.add_argument("--qwen", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    base = ARTIFACTS if args.artifacts else args.dir
    args.mlp = args.mlp or base / "mlp_results.json"
    args.lm = args.lm or base / "lm_results.json"
    args.qwen = args.qwen or base / "qwen_grpo_results.json"
    args.out = args.out or args.dir / ("table4_artifacts.json" if args.artifacts else "table4.json")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    table = assemble(args.mlp, args.lm, args.qwen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1))
    print(markdown(table))
    print(f"\nwrote {args.out}")
    return table


if __name__ == "__main__":
    main()
