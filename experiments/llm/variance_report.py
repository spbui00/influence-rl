"""Table + figure from outputs/llm/diag/variance/variance_G*.json (experiments/llm/variance_sweep.py)."""
import glob, json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = sys.argv[1] if len(sys.argv) > 1 else "outputs/llm/diag/variance"
D = sorted((json.load(open(f)) for f in glob.glob(f"{root}/variance_G*.json")), key=lambda d: d["G"])
G = [d["G"] for d in D]
rows = ["| G | live both | grad cos (mean) | guide cos dot / adam | sign agree dot / adam | Pearson dot / adam | Spearman dot / adam |", "|---|---|---|---|---|---|---|"]
for d in D:
    s, g = d["score_agreement_live_both"], d["guide_gradient_cosine"]
    rows.append(f"| {d['G']} | {d['live_both']} | {d['grad_cosine_live_both']:.3f} | {g['dot']:.2f} / {g['adam']:.2f} | {s['dot']['sign_agreement']:.2f} / {s['adam']['sign_agreement']:.2f} | "
                f"{s['dot']['pearson']:.2f} / {s['adam']['pearson']:.2f} | {s['dot']['spearman']:.2f} / {s['adam']['spearman']:.2f} |")
open(f"{root}/summary.md", "w").write("# Sampling variance vs G (C1 influence-adam-s0 checkpoint-100, 64 prompts, 32 MATH-500 guide, two independent draws)\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(G, [d["grad_cosine_live_both"] for d in D], "o-", label="per-prompt GRPO gradient")
ax[0].plot(G, [d["guide_gradient_cosine"]["dot"] for d in D], "s-", label="guide gradient (32 problems)")
ax[0].set_ylabel("cosine between two draws")
for k, m in (("dot", "o-"), ("adam", "s--")):
    ax[1].plot(G, [d["score_agreement_live_both"][k]["sign_agreement"] for d in D], m, label=k)
    ax[2].plot(G, [d["score_agreement_live_both"][k]["pearson"] for d in D], m, label=k)
ax[1].axhline(.5, color="gray", lw=.8, ls=":"); ax[1].set_ylabel("influence sign agreement")
ax[2].axhline(0, color="gray", lw=.8, ls=":"); ax[2].set_ylabel("influence Pearson between draws")
for a in ax:
    a.set_xscale("log", base=2); a.set_xticks(G); a.set_xticklabels(G); a.set_xlabel("rollouts per prompt G"); a.legend(fontsize=8)
fig.suptitle("How many rollouts until the influence score of a prompt is reproducible? (1.5B, C1 checkpoint 100)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{root}/variance.png", dpi=130)
