"""Chapter figures, built only from raw run outputs. Run from the repo root: python -m experiments.thesis_figures
Prints every plotted mean so it can be compared with the text. Writes outputs/figures/llm/*.{pdf,png} and copies the toy figures to outputs/figures/toy/. Reads outputs/llm/ (the run outputs, not shipped)."""
import glob, json, re, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/figures/llm"); OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#22252b", "#62666f", "#e4e5e8"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.dpi": 300, "legend.frameon": False})
# Same colours and markers as the controlled-environment figures (experiments/toy/plots.py) for the shared estimators.
ARM = {"none": ("no selection", "#22252b", "o", "-"), "random": ("random", "#a9794a", "v", "--"), "influence-dot": ("dot product", "#8a8f99", "^", "-"),
       "influence-adam": ("Adam", "#2b6fde", "o", "-"), "influence-ekfac": ("Fisher (EK-FAC)", "#2e9e4f", "s", "-"),
       "disagreement": ("disagreement", "#c9542c", "P", "-"), "oracle": ("oracle", "#7b3fc4", "D", "-"), "clean": ("clean pool", "#16888f", "X", "-")}

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); plt.close(fig)

def runs(root):
    out = {}
    for d in sorted(glob.glob(f"outputs/llm/{root}/*-s[0-9]")):
        m = re.fullmatch(r"(.+)-s(\d+)", Path(d).name)
        if Path(d, "evals.json").exists():
            out.setdefault(m[1], {})[int(m[2])] = Path(d)
    return out

def evals(d):
    return {int(k): v for k, v in json.loads(Path(d, "evals.json").read_text()).items()}

def log(d):
    last = {}
    for line in open(Path(d, "log.jsonl")):
        r = json.loads(line); last[r["step"]] = r
    return [last[s] for s in sorted(last)]

def final(d, key):
    return float(np.mean(json.loads(Path(d, "final_eval.json").read_text())[key]))

def ns(name, key="rollout-gold/{}"):
    j = json.load(open(f"outputs/llm/next_step/{name}/next_step.json"))
    return {m: j["lds"][key.format(m)]["1"]["mean"] for m in ("dot", "adam")}

# ---------------------------------------------------------------- rollouts
GS = [8, 16, 32, 64, 128, 256, 512]
fresh = [ns(f"c1_fresh_G{g}") for g in GS]
asym = [ns(f"c1_asym_G{g}_vs512") for g in GS[:5]]
fixed = ns("c1_fixed_G8_ekfac")
var = [json.load(open(f"outputs/llm/diag/variance/variance_G{g}.json")) for g in GS]
print("fresh", [(g, round(r["dot"], 3), round(r["adam"], 3)) for g, r in zip(GS, fresh)]); print("asym", [(g, round(r["dot"], 3), round(r["adam"], 3)) for g, r in zip(GS, asym)]); print("fixed", fixed)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"wspace": .3})
a.axhspan(-.071 * 2, .071 * 2, color=GRID, alpha=.7, lw=0)
a.axhline(fixed["adam"], color=ARM["influence-adam"][1], lw=1, ls=":")
a.text(8, fixed["adam"] - .03, "same rollouts, Adam", va="top", fontsize=7, color=MUTED)
for m, k in (("dot", "influence-dot"), ("adam", "influence-adam")):
    a.plot(GS, [r[m] for r in fresh], color=ARM[k][1], marker=ARM[k][2], ms=4, label=rf"$G_{{\mathrm{{step}}}}=G_{{\mathrm{{score}}}}$, {ARM[k][0]}")
    a.plot(GS[:5], [r[m] for r in asym], color=ARM[k][1], marker=ARM[k][2], ms=4, ls="--", mfc="white", label=rf"$G_{{\mathrm{{step}}}}=512$, {ARM[k][0]}")
a.set_xscale("log", base=2); a.set_xticks(GS, GS); a.minorticks_off(); a.set_ylim(-.25, 1.08); a.set_xlabel(r"$G_{\mathrm{score}}$"); a.set_ylabel("next-step LDS")
a.set_title("(a) LDS on a fresh draw", loc="left"); a.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, columnspacing=1)
b.plot(GS, [v["grad_cosine_live_both"] for v in var], color=INK, marker="o", ms=4, label="prompt gradient")
b.plot(GS, [v["guide_gradient_cosine"]["dot"] for v in var], color=MUTED, marker="s", ms=4, ls="--", label="guide gradient")
b.set_xscale("log", base=2); b.set_xticks(GS, GS); b.minorticks_off(); b.set_ylim(0, 1); b.set_xlabel("rollouts per prompt $G$"); b.set_ylabel("cosine between two draws")
b.set_title("(b) Agreement between draws", loc="left"); b.legend(loc="upper left")
save(fig, "next_step_vs_G")

# ---------------------------------------------------------------- horizon
j = json.load(open("outputs/llm/horizon/c1/horizon.json")); grid = j["grid"]
fig, a = plt.subplots(figsize=(3.6, 2.4))
a.axhspan(-2 / np.sqrt(j["n_subsets"] - 1), 2 / np.sqrt(j["n_subsets"] - 1), color=GRID, alpha=.7, lw=0)
for m, k in (("dot", "influence-dot"), ("adam", "influence-adam"), ("ekfac", "influence-ekfac")):
    v = j["lds"][f"rollout-gold/{m}"]
    y = np.array([v[str(g)]["mean"] for g in grid]); se = np.array([np.std(v[str(g)]["per_target"]) / np.sqrt(len(v[str(g)]["per_target"])) for g in grid])
    print("horizon", m, y.round(3), "se", se.round(3))
    a.errorbar(grid, y, yerr=1.96 * se, color=ARM[k][1], marker=ARM[k][2], ms=4, capsize=2, elinewidth=.8, label=ARM[k][0])
a.set_xscale("log"); a.set_xticks(grid, grid); a.minorticks_off(); a.set_ylim(-.3, 1.08); a.set_xlabel("horizon $m$ (steps, including the scored one)"); a.set_ylabel("LDS on the same rollouts"); a.legend(loc="upper right")
save(fig, "ice_horizon")

# ---------------------------------------------------------------- selection curves
fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.5), gridspec_kw={"hspace": .42, "wspace": .3})
ARMS = ("none", "random", "influence-dot", "influence-adam")
for a, root, title, key in ((ax[0, 0], "c1", "(a) 1.5B, MATH level 4\u20135", "math"), (ax[0, 1], "c1_3b", "(b) 3B, MATH level 4\u20135", "math"), (ax[1, 0], "c1_7b", "(c) 7B, MATH level 4\u20135", "math")):
    r = runs(root)
    for arm in ARMS:
        E = [evals(d) for d in r[arm].values()]
        st = sorted(set.intersection(*[set(e) for e in E])); st = [s for s in st if s != 1 and key in E[0][s]]
        M = np.array([[e[s][key] for s in st] for e in E])
        print(root, arm, "n", len(E), "steps", st, M.mean(0).round(3), "last", M[:, -1].round(3))
        a.fill_between(st, M.min(0), M.max(0), color=ARM[arm][1], alpha=.16, lw=0)
        a.plot(st, M.mean(0), color=ARM[arm][1], marker=ARM[arm][2], ls=ARM[arm][3], ms=3.5, label=ARM[arm][0])
    a.set_title(title, loc="left"); a.set_xlabel("training step"); a.set_ylabel("pass rate")
a = ax[1, 1]; r = runs("c1_7b")
for arm in ARMS:
    rows = []
    for d in r[arm].values():
        L = log(d); sec = np.cumsum([x["step_seconds"] for x in L]) / 3600; steps_ = [x["step"] for x in L]; e = evals(d)
        st = [s for s in sorted(e) if s not in (0, 1) and "math" in e[s]]
        rows.append(([float(sec[steps_.index(s)]) if s in steps_ else float(sec[-1]) for s in st], [e[s]["math"] for s in st]))
    X, Y = np.mean([x for x, _ in rows], 0), np.mean([y for _, y in rows], 0)
    print("7B cost", arm, "GPUh", X.round(1), "math", Y.round(3))
    a.plot(X, Y, color=ARM[arm][1], marker=ARM[arm][2], ls=ARM[arm][3], ms=3.5)
a.set_title("(d) 7B, against compute", loc="left"); a.set_xlabel("GPU-hours"); a.set_ylabel("pass rate")
fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(.5, .98))
save(fig, "selection_curves")

# ---------------------------------------------------------------- faulty verifier
r = runs("f2"); r.pop("disagreement", None)  # not reported in the chapter
order =sorted(r, key=lambda k: np.mean([final(d, "gsm8k") for d in r[k].values()]))
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True, gridspec_kw={"wspace": .08})
base = evals(r["none"][0])[0]["gsm8k"]
for i, arm in enumerate(order):
    v = [final(d, "gsm8k") for d in r[arm].values()]
    a.plot([min(v), max(v)], [i, i], color=ARM[arm][1], lw=1)
    a.scatter(v, [i] * len(v), color=ARM[arm][1], marker=ARM[arm][2], s=18, zorder=3, edgecolor="white", linewidth=.4)
    a.scatter(np.mean(v), i, color=ARM[arm][1], marker="|", s=70, zorder=4)
    P = []
    for d in r[arm].values():
        L = log(d)
        if arm != "clean" and arm != "none":
            P.append(sum(x["n_bad_dropped"] for x in L) / (16 * len(L)))
    print("faulty", arm, round(float(np.mean(v)), 3), np.round(v, 3), "precision", np.round(P, 3))
    if P:
        b.scatter(P, [i] * len(P), color=ARM[arm][1], marker=ARM[arm][2], s=18, zorder=3, edgecolor="white", linewidth=.4)
a.set_yticks(range(len(order)), [ARM[k][0] for k in order]); a.tick_params(axis="y", colors=INK, length=0); a.grid(axis="y", visible=False); b.grid(axis="y", visible=False); b.tick_params(axis="y", length=0)
a.set_xlabel("GSM8K pass rate at step 100 (16 samples)"); a.set_title("(a) Held-out accuracy", loc="left")
b.axvline(.3, color=MUTED, lw=.9, ls=":"); b.text(.32, order.index("clean"), "30% of the pool is faulty", fontsize=7, color=MUTED, va="center")
b.set_xlim(0, 1.02); b.set_xlabel("share of dropped prompts that were faulty"); b.set_title("(b) Precision of the drops", loc="left")
save(fig, "faulty_verifier")

# ---------------------------------------------------------------- mixed safety and math pool
SAFE = {"none": ("no selection", "none"), "random": ("random", "random"), "influence-dot-safety": ("dot, safety guide", "influence-dot"), "influence-adam-safety": ("Adam, safety guide", "influence-adam"),
        "influence-dot-math": ("dot, math guide", "influence-dot"), "influence-adam-math": ("Adam, math guide", "influence-adam")}
print("safety arms", list(runs("safety/full")), list(runs("safety_3b")))
def safe_arm(rs, key):
    for k in (key, key.replace("influence-", "")):
        if k in rs:
            return rs[k]
    raise KeyError(key)
fig, ax = plt.subplots(2, 3, figsize=(6.3, 3.6), sharey=True, gridspec_kw={"wspace": .1, "hspace": .42})
for row, (root, name) in enumerate((("safety/full", "1.5B"), ("safety_3b", "3B"))):
    rs = runs(root); base = evals(next(iter(rs["none"].values())))[0]
    for col, (key, title) in enumerate((("harmful", "refusal on harmful"), ("benign", "compliance on benign"), ("math", "MATH level 4\u20135"))):
        a = ax[row, col]
        for i, (arm, (lab, c)) in enumerate(reversed(list(SAFE.items()))):
            v = [evals(d)[100][key] for d in safe_arm(rs, arm).values()]
            a.scatter(v, [i] * len(v), color=ARM[c][1], marker=ARM[c][2], s=14, zorder=3, edgecolor="white", linewidth=.3, alpha=.9)
            a.scatter(np.mean(v), i, color=ARM[c][1], marker="|", s=70, zorder=4)
            if col == 0:
                print(name, arm, {k: round(float(np.mean([evals(d)[100][k] for d in safe_arm(rs, arm).values()])), 3) for k in ("harmful", "benign", "math")}, "n", len(v))
        a.axvline(base[key], color=MUTED, lw=.9, ls=":")
        a.set_title(f"{name}: {title}", loc="left", fontsize=7.5); a.grid(axis="y", visible=False); a.tick_params(axis="y", length=0)
        a.locator_params(axis="x", nbins=3)
    ax[row, 0].set_yticks(range(len(SAFE)), [v[0] for v in reversed(list(SAFE.values()))]); ax[row, 0].tick_params(axis="y", colors=INK)
fig.supxlabel("pass rate at step 100, 8 samples (dotted: base model)", fontsize=8, y=-.01)
save(fig, "safety_pool")

# ---------------------------------------------------------------- controlled environment: matched instrument, same run as tab:toy
T = Path("outputs/figures/toy"); T.mkdir(parents=True, exist_ok=True)
for src, dst in (("fig2", "toy_lds_panel_matched"), ("fig4_matched", "contlr_sweep_matched")):
    for ext in ("pdf", "png"):
        shutil.copy(f"outputs/toy_fullrun/figures/{src}.{ext}", T / f"{dst}.{ext}")
