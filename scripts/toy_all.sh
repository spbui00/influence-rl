#!/usr/bin/env bash
# Every toy number and figure (Table 1 + p-values, Fig. 2, Fig. 4 matched + fresh-optimizer ablation,
# reference curve, baselines) with the paper defaults. ~30 min on a laptop CPU. Idempotent: finished
# units are skipped; ground-truth retrains are cached under $OUT/_cache.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/toy}
FIX=${FIX:-}                # fixture/method flags shared by every run (toy_smoke.sh shrinks them)
P1X=${P1X:-}                # extra Protocol-1 flags
P2X=${P2X:-}                # extra Protocol-2 flags
REFX=${REFX:-}              # extra reference-curve flags
CACHE="--cache-dir $OUT/_cache"
mkdir -p "$OUT"

echo "== reference curve"
$PY -m experiments.toy.reference_curve $FIX $REFX --out "$OUT/reference_curve.json"

echo "== Protocol 1, matched instrument (warm Adam + momentum-weighted visits)"
$PY -m experiments.toy.protocol1 $FIX $P1X $CACHE --cont-lr 0.00001 --out "$OUT/p1_eta_over_100"
$PY -m experiments.toy.protocol1 $FIX $P1X $CACHE                   --out "$OUT/p1_eta_over_10"
$PY -m experiments.toy.protocol1 $FIX $P1X $CACHE --cont-lr 0.001   --out "$OUT/p1_eta"

echo "== Protocol 2, trajectory sum + from-scratch retrains"
$PY -m experiments.toy.protocol2 $FIX $P2X $CACHE --out "$OUT/p2_history"

echo "== Instrument ablation (fresh optimizer + plain visit counts, uncached)"
A="--fresh-optimizer --plain-counts --no-cache"
$PY -m experiments.toy.protocol1 $FIX $P1X $A --cont-lr 0.00001 --out "$OUT/p1_eta_over_100_fresh"
$PY -m experiments.toy.protocol1 $FIX $P1X $A                   --out "$OUT/p1_eta_over_10_fresh"
$PY -m experiments.toy.protocol1 $FIX $P1X $A --cont-lr 0.001   --out "$OUT/p1_eta_fresh"

echo "== baselines on the cached P1 outcomes"
$PY -m experiments.toy.baselines --out "$OUT/baselines.json" \
  "eta/100=$OUT/p1_eta_over_100" "eta/10=$OUT/p1_eta_over_10" "eta=$OUT/p1_eta"

echo "== Table 1"
$PY -m experiments.toy.table1 --eta-over-100 "$OUT/p1_eta_over_100" --eta-over-10 "$OUT/p1_eta_over_10" \
  --eta "$OUT/p1_eta" --p2 "$OUT/p2_history" --out "$OUT/table1.json"

echo "== Figures"
$PY -m experiments.toy.plots fig2 --eta-over-100 "$OUT/p1_eta_over_100" --eta-over-10 "$OUT/p1_eta_over_10" \
  --eta "$OUT/p1_eta" --p2 "$OUT/p2_history" --out "$OUT/figures/fig2"
$PY -m experiments.toy.plots fig4 --out "$OUT/figures/fig4_matched" \
  "η=$OUT/p1_eta" "η/10=$OUT/p1_eta_over_10" "η/100=$OUT/p1_eta_over_100"
$PY -m experiments.toy.plots fig4 --out "$OUT/figures/fig4_fresh" \
  "η=$OUT/p1_eta_fresh" "η/10=$OUT/p1_eta_over_10_fresh" "η/100=$OUT/p1_eta_over_100_fresh"
echo "DONE -> $OUT"
