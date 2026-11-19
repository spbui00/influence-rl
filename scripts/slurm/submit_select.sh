#!/bin/bash
# Queue one selection run as SLOTS chained 3 h jobs (afterany), each resuming from the last checkpoint.
#   bash scripts/slurm/submit_select.sh <out> <selector> <method> <seed> <pool> [guide] [slots]
# Remaining knobs (EVALS, STEPS, CANDIDATES, DROP, EVAL_SAMPLES, EXTRA) are read from the environment.
set -euo pipefail
OUT=$1 SELECTOR=$2 METHOD=$3 SEED=$4 POOL=$5 GUIDE=${6:-} SLOTS=${7:-3}
mkdir -p logs "$OUT"
dep=""
for i in $(seq 1 "$SLOTS"); do
  jid=$(OUT=$OUT SELECTOR=$SELECTOR METHOD=$METHOD SEED=$SEED POOL=$POOL GUIDE=$GUIDE EVALS="${EVALS:-}" \
        sbatch --parsable ${ACCOUNT:+--account=$ACCOUNT} ${TIME:+--time=$TIME} ${PARTITION:+--partition=$PARTITION} ${MEM:+--mem=$MEM} --export=ALL --job-name="sel-$SELECTOR-$METHOD-s$SEED" $dep scripts/slurm/select.slurm)
  echo "$OUT slot $i -> $jid"; dep="--dependency=afterany:$jid"
done
