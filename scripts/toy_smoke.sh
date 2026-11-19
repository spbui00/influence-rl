#!/usr/bin/env bash
# The full toy chain at a tiny size (< 3 min on CPU); outputs go to outputs/toy_smoke.
set -euo pipefail
cd "$(dirname "$0")/.."
export OUT=${OUT:-outputs/toy_smoke}
export FIX="--n-test 4 ${FIX:-}"
export REFX="--full-steps 60 ${REFX:-}"
export P1X="--full-steps 60 --subset-steps 30 --m-subsets 6 --record-every 10 ${P1X:-}"
export P2X="--steps 60 --subset-steps 60 --m-subsets 6 --score-every 30 ${P2X:-}"
rm -rf "$OUT"
exec bash scripts/toy_all.sh
