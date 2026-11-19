#!/bin/bash
# App. E.1 stage 2: influence scores at one reference checkpoint for one estimator (rollout mode, cosine, G=8,
# score batch 8; per-target rows kept). torchrun shards pool and targets over the job's GPUs (the chunk seeds
# depend on the world size, as in the paper runs: 4 ranks). ~1-2 h per (checkpoint, method) on one GPU; EK-FAC
# adds the factor fit. Submit one job per cell:
#
#   for C in 100 200; do for M in dot adam ekfac; do
#     REF_STEP=$C METHOD=$M sbatch --gres=gpu:4 --mem=200G --time=6:00:00 --cpus-per-task=32 \
#       scripts/slurm/continuation_score.slurm
#   done; done
#
# Env knobs: OUT (outputs/llm/continuation), REF_DIR ($OUT/ref), REF_STEP (100), METHOD (dot), NGPU (all visible),
# LEGACY_LAYOUT (1 = also write the shipped rollout_<name>_cos naming), EXTRA (score.py flags, e.g. --no-cosine).
#SBATCH --job-name=cont-score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
PY="${PY:-.venv/bin/python}"
OUT="${OUT:-outputs/llm/continuation}"
REF_DIR="${REF_DIR:-$OUT/ref}"
REF_STEP="${REF_STEP:-100}"
METHOD="${METHOD:-dot}"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"; NGPU="${NGPU:-1}"; [ "$NGPU" -ge 1 ] || NGPU=1
LEGACY=""; [ "${LEGACY_LAYOUT:-0}" = "1" ] && LEGACY="--legacy-layout"
PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
[ -d "$REF_DIR/checkpoint-$REF_STEP" ] || { echo "missing $REF_DIR/checkpoint-$REF_STEP"; exit 1; }
"$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master-addr 127.0.0.1 --master-port "$PORT" \
    -m experiments.llm.continuation.score --ref-dir "$REF_DIR" --steps "$REF_STEP" --methods "$METHOD" \
    $LEGACY ${EXTRA:-}
echo "scores under $REF_DIR/influence/$METHOD/"
