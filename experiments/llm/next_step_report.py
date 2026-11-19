"""LDS vs G table + figure from outputs/llm/next_step/c1_{fresh,asym,fixed}_G*/next_step.json (experiments/llm/next_step.py)."""
import glob, json, re, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = sys.argv[1] if len(sys.argv) > 1 else "outputs/llm/next_step"
runs = {}  # (design, G) -> lds dict
for f in glob.glob(f"{root}/c1_*_G*/next_step.json"):
    m = re.search(r"c1_(fresh|asym|fixed)_G(\d+)", f)
    if m:
        runs.setdefault((m[1], int(m[2])), {}).update(json.load(open(f))["lds"])
label = {"fixed": "fixed rollouts (ICE)", "fresh": "fresh rollouts, same G (ACE)", "asym": "score at G, truth at G=512"}
rows = ["| design | G | dot k=1 / 10 | adam k=1 / 10 | ekfac k=1 / 10 |", "|---|---|---|---|---|"]
for (d, G), l in sorted(runs.items()):
    cell = lambda m: f"{l[m]['1']['mean']:+.2f} / {l[m]['10']['mean']:+.2f}" if m in l else "-"
    rows.append(f"| {label[d]} | {G} | {cell('rollout-gold/dot')} | {cell('rollout-gold/adam')} | {cell('rollout-gold/ekfac')} |")
open(f"{root}/summary.md", "w").write("# Next-step LDS vs rollouts per prompt G (C1 influence-adam-s0 checkpoint-100, 64 prompts, 32 MATH-500 gold targets, 200 half-batches)\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for a, k in zip(ax, ("1", "10")):
    for d, mk in (("fixed", "^-"), ("asym", "s-"), ("fresh", "o-")):
        pts = sorted((G, l["rollout-gold/dot"][k]["mean"]) for (dd, G), l in runs.items() if dd == d)
        if pts: a.plot(*zip(*pts), mk, label=label[d])
    a.axhline(0, color="gray", lw=.8, ls=":"); a.set_xscale("log", base=2); a.set_xlabel("rollouts per prompt G (score side)")
    a.set_title(f"k = {k} step{'s' if k != '1' else ''}", fontsize=10)
ax[0].set_ylabel("LDS (dot)"); ax[0].legend(fontsize=8)
fig.suptitle("Next-step LDS vs G: fixed rollouts ~1 at any G, fresh rollouts need G in the hundreds", fontsize=10)
fig.tight_layout(); fig.savefig(f"{root}/lds_vs_G.png", dpi=130)
