#!/bin/bash
# One selection-GRPO run (experiments/llm/select/train.py), resumable: a requeued/chained job continues from the latest
# checkpoint-N under OUT. Chain 3 h slots with scripts/slurm/submit_select.sh.
#
#   SELECTOR=influence METHOD=adam SEED=0 POOL=... GUIDE=... EVALS="math=... gsm8k=..." OUT=... sbatch scripts/slurm/select.slurm
#
#SBATCH --job-name=select
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "scripts/slurm/${CLUSTER_ENV:-nibi_env.sh}"

EVAL_ARGS=()
for e in ${EVALS:-}; do EVAL_ARGS+=(--eval "$e"); done
GUIDE_ARGS=(); [ -n "${GUIDE:-}" ] && GUIDE_ARGS=(--guide "$GUIDE")

python -m experiments.llm.select.train --pool "$POOL" ${GUIDE_ARGS[@]+"${GUIDE_ARGS[@]}"} ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
    --selector "${SELECTOR:-none}" --method "${METHOD:-adam}" --seed "${SEED:-0}" --out "$OUT" \
    --steps "${STEPS:-100}" --candidates "${CANDIDATES:-64}" --drop "${DROP:-16}" \
    --eval-at "${EVAL_AT:-1,25,50,75,100}" --eval-samples "${EVAL_SAMPLES:-16}" --bad-group "${BAD_GROUP:-poison}" ${EXTRA:-}
