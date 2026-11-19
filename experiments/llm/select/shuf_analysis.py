"""Why adam ~ dot: analyse grad_profile_shuf.{json,npz}. Usage: python shufP_analysis.py <out.json>"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

J = sys.argv[1]
r = json.load(open(J))
z = np.load(J[:-5] + ".npz")
live = z["live"].astype(bool)
sp = lambda a, b: spearmanr(a, b).correlation
q = lambda x: float(np.exp(np.diff(np.percentile(np.log(np.abs(x)), [10, 90]))[0]))
P = r["preconditioner"]
print(f"P spread 99/1 {P['spread_99_1']:.0f}x, between-tensor share {P['between_tensor_variance_fraction']:.2f}")
for g in r["guides"]:
    s = {k[len(f"score_{g}_"):]: z[k][live] for k in z.files if k.startswith(f"score_{g}_")}
    d, a = s["dot"], s["adam"]
    print(f"\n== guide {g}: {live.sum()} live prompts")
    print("variant      | Spearman vs adam | vs dot | w=s/dot spread 90/10 | w<0")
    for k, v in s.items():
        w = v / d
        print(f"{k:12s} | {sp(v, a):.3f}            | {sp(v, d):.3f}  | {q(w[w > 0]):6.1f}x              | {np.mean(w < 0):.2f}")
    D, A = z[f"contrib_{g}_dot"][live], z[f"contrib_{g}_adam"][live]  # (prompts, tensors)
    # how spread out is each score over tensors, and how much do tensor parts cancel?
    pr = (np.abs(D).sum(1) ** 2) / (D ** 2).sum(1)
    canc = np.abs(D.sum(1)) / np.abs(D).sum(1)
    print(f"tensors carrying a dot score (participation ratio): median {np.median(pr):.0f} of {D.shape[1]}; |sum| / sum|parts|: median {np.median(canc):.2f}")
    # effective per-tensor weight: through-origin slope of adam part on dot part across prompts
    e = (A * D).sum(0) / np.maximum((D ** 2).sum(0), 1e-300)
    r2 = 1 - ((A - e * D) ** 2).sum(0) / np.maximum((A ** 2).sum(0), 1e-300)
    wt = (D ** 2).sum(0); wt = wt / wt.sum()  # tensor's share of dot-score energy
    c = z["p_mean"]  # arithmetic tensor mean of P (the "block" variant): block part = c_t * dot part
    assert np.allclose(D @ c, s["block"], rtol=1e-2, atol=1e-3 * np.abs(s["block"]).max())
    top = np.argsort(-wt)[:np.searchsorted(np.cumsum(np.sort(wt)[::-1]), .8) + 1]
    sp90 = lambda x: float(np.exp(np.diff(np.percentile(np.log(np.abs(x[top])), [10, 90]))[0]))
    print(f"over the {len(top)} tensors holding 80% of dot-score energy: tensor-mean P spread 10-90 {sp90(c):.1f}x, effective weight e_t spread {sp90(e):.1f}x, "
          f"median e_t / mean P {np.median(e[top] / c[top]):.3f}, corr(log e_t, log mean P) {np.corrcoef(np.log(np.abs(e[top])), np.log(c[top]))[0, 1]:.2f}")
    fit = D @ e
    print(f"per-tensor fit adam_part ~ e_t * dot_part: energy-weighted R2 {np.sum(wt * r2):.3f}; Spearman(fit, adam) {sp(fit, a):.3f}, Spearman(fit, dot) {sp(fit, d):.3f}")
    le = np.log(np.abs(e) + 1e-300)
    m = np.average(le, weights=wt)
    print(f"effective weights e_t: energy-weighted SD of log {np.sqrt(np.average((le - m) ** 2, weights=wt)):.2f}, 10-90 spread over tensors holding 80% of energy "
          f"{np.exp(np.diff(np.percentile(le[np.argsort(-wt)][:np.searchsorted(np.cumsum(np.sort(wt)[::-1]), .8) + 1], [10, 90]))[0]):.1f}x, e_t<0 share of energy {wt[e < 0].sum():.3f}")
