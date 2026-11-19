"""`python -m experiments.compare_to_paper <table.json> [...]` checks a table written by any analysis script
against experiments/paper_values.py; the analysis scripts themselves carry no paper numbers."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from experiments import paper_values as pv
from influence_rlvr.files import read_json, write_json

Cell = tuple[str, float | None, float | None, float, str]


def _round_half_away(x: float, decimals: int) -> float:
    s = 10 ** decimals
    return math.copysign(math.floor(abs(x) * s + 0.5 + 1e-9) / s, x)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def identify(table: dict) -> str:
    if "table" in table:
        return table["table"]
    if "means" in table and "p_values" in table:
        return "table1"
    if "rows" in table and "auc_by_checkpoint" in table:
        return "table8"
    if "rows" in table:
        return "table4"
    if "cells" in table and "n_splits" in table:
        return "table7"
    if "cells" in table:
        return "table6"
    raise ValueError("cannot identify the table layout")


def cells_table1(table: dict) -> list[Cell]:
    out = []
    for col, row in pv.TABLE1.items():
        for m, v in row.items():
            ours = table.get("means", {}).get(col, {}).get(m)
            note = "paper text truncated" if _finite(ours) and round(ours, 3) != v and abs(ours - v) <= pv.TOLERANCE["table1"] else ""
            out.append((f"{col}/{m}", v, ours, pv.TOLERANCE["table1"], note))
    for col, gaps in table.get("p_values", {}).items():
        for g in gaps:
            key = (g["better"], g["worse"])
            target = pv.TABLE1_P_VALUES.get(col, {}).get(key)
            if target is not None:
                out.append((f"p/{col}/{key[0]}>{key[1]}", target, g["p"], 0.01, ""))
            else:
                out.append((f"p/{col}/{key[0]}>{key[1]}", 0.0, g["p"], pv.TABLE1_P_MAX_OTHERS, "reported as p<0.01"))
    return out


def cells_table4(table: dict) -> list[Cell]:
    rows = table.get("rows", {})
    out = []
    for key, ref in pv.TABLE4.items():
        name, _, batch = key.partition("/")
        cell = rows.get(name)
        if cell is not None and batch:
            cell = {"D": cell.get("D"), **cell.get("by_batch", {}).get(batch, {})}
        for field, v in ref.items():
            ours = None if cell is None else cell.get(field)
            tol = pv.TOLERANCE["table4"][field]
            if field == "D":
                out.append((f"{key}/D", v, ours, tol * v, "relative 5%"))
            elif field == "rel_err_pct":
                out.append((f"{key}/{field}", v, None if ours is None else round(ours), tol, "rounded to integer"))
            else:
                out.append((f"{key}/{field}", v, None if ours is None else round(ours, 2), tol, "rounded to 2 decimals"))
    return out


def cells_grid(table: dict, ref: dict, columns: tuple[str, ...] | None, tol, name: str) -> list[Cell]:
    out = []
    for c, by_k in ref.items():
        for k, row in by_k.items():
            got = table.get("cells", {}).get(str(c), {}).get(str(k)) or {}
            items = row.items() if columns is None else zip(columns, row)
            for col, v in items:
                ours = got.get(col)
                t = tol[col] if isinstance(tol, dict) else tol
                note = ""
                if name == "table6" and _finite(ours) and abs(ours - v) > t:
                    if abs(_round_half_away(_round_half_away(ours, 3), 2) - v) <= t:
                        note = "paper double-rounded via 3 decimals"
                out.append((f"c{c}/k{k}/{col}", v, ours, t, note))
    return out


def cells_table8(table: dict) -> list[Cell]:
    rows = table.get("rows", {})
    out = []
    for key, vals in pv.TABLE8.items():
        for col, v in zip(pv.TABLE8_COLUMNS, vals):
            if v is None:
                continue
            ours = (rows.get(key) or {}).get(col)
            out.append((f"{key}/{col}", v, ours, pv.TOLERANCE["table8"][col], ""))
    return out


CELLS = {
    "table1": cells_table1,
    "table4": cells_table4,
    "table6": lambda t: cells_grid(t, pv.TABLE6, None, pv.TOLERANCE["table6"], "table6"),
    "table7": lambda t: cells_grid(t, pv.TABLE7, pv.TABLE7_COLUMNS, pv.TOLERANCE["table7"], "table7"),
    "table8": cells_table8,
}


def compare(table: dict, name: str | None = None) -> dict:
    name = name or identify(table)
    rows = []
    for cell, paper, ours, tol, note in CELLS[name](table):
        present = _finite(ours)
        diff = abs(ours - paper) if present else None
        ok = present and (diff <= tol + 1e-12 or bool(note) and note.startswith("paper double"))
        rows.append({"cell": cell, "paper": paper, "ours": ours, "abs_diff": diff, "tol": tol, "ok": ok, "note": note})
    compared = [r for r in rows if _finite(r["ours"])]
    return {"table": name, "n_reference": len(rows), "n_compared": len(compared),
            "n_mismatch": sum(not r["ok"] for r in compared), "max_abs_diff": max((r["abs_diff"] for r in compared), default=None),
            "cells": rows}


def report(result: dict, verbose: bool = False) -> str:
    lines = [f"{result['table']}: {result['n_compared']}/{result['n_reference']} paper values present, "
             f"{result['n_mismatch']} outside tolerance"
             + (f", max |diff| {result['max_abs_diff']:.4g}" if result["max_abs_diff"] is not None else "")]
    for r in result["cells"]:
        if not _finite(r["ours"]):
            continue
        if not r["ok"] or verbose:
            flag = "OK " if r["ok"] else "MISMATCH"
            lines.append(f"  {flag} {r['cell']}: paper {r['paper']:g} ours {r['ours']:.4g} (|diff| {r['abs_diff']:.4g}, tol {r['tol']:g})"
                         + (f" [{r['note']}]" if r["note"] else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="compare analysis-script tables with the values printed in the paper")
    p.add_argument("tables", nargs="+", type=Path)
    p.add_argument("--name", default=None, help="table1|table4|table6|table7|table8 (default: inferred from the JSON)")
    p.add_argument("--out", type=Path, default=None, help="write the per-cell comparison JSON")
    p.add_argument("--verbose", action="store_true", help="print every cell, not only mismatches")
    args = p.parse_args(argv)
    results = [compare(read_json(t), args.name) for t in args.tables]
    for r in results:
        print(report(r, args.verbose))
    if args.out:
        write_json(args.out, results if len(results) > 1 else results[0])
    return int(any(r["n_mismatch"] for r in results))


if __name__ == "__main__":
    sys.exit(main())
