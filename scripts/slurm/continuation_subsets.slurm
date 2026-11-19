#!/bin/bash
# App. E.1 stage 3: the LDS retraining sweep, one array task per subset. Task m loads the checkpoint-REF_STEP
# adapter, continues GRPO on subsets.npy[m] for 200 steps with a fresh optimizer and evaluates the 32 targets
# in-training at k in {1,50,100,150,200} -> $OUT/runs/step<c>/subset_<m>/target_eval.json. ~4.5 h per task on one
# GPU; 100 tasks x 2 checkpoints ~900 GPU-h. Fully idempotent: a measured subset exits at once, a partial run
# restarts from the reference checkpoint (only the final checkpoint is saved). Submit once per checkpoint:
#
#   for C in 100 200; do
#     REF_STEP=$C sbatch --gres=gpu:1 --mem=200G --time=8:00:00 --cpus-per-task=16 --array=0-99%12 \
#       scripts/slurm/continuation_subsets.slurm
#   done
#   smoke: CONT_STEPS=4 EVAL_AT=2 EXTRA="--eval-n-samples 2" sbatch ... --array=0 continuation_subsets.slurm
#
# Env knobs: OUT, REF_DIR ($OUT/ref), REF_STEP (100), CONT_STEPS (200), EVAL_AT (1,50,100,150,200), SEED (0),
# PROMPTS_PER_STEP (64), USE_VLLM ("--use-vllm"), EXTRA.
#SBATCH --job-name=cont-subset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
PY="${PY:-.venv/bin/python}"
OUT="${OUT:-outputs/llm/continuation}"
REF_DIR="${REF_DIR:-$OUT/ref}"
REF_STEP="${REF_STEP:-100}"
M="${SLURM_ARRAY_TASK_ID:?run via sbatch --array}"
INIT="$REF_DIR/checkpoint-$REF_STEP"
[ -d "$INIT" ] || { echo "missing reference checkpoint $INIT"; exit 1; }
"$PY" -m experiments.llm.continuation.train --out "$OUT" --subset-id "$M" --init-checkpoint "$INIT" \
    --max-steps "${CONT_STEPS:-200}" --eval-at "${EVAL_AT:-1,50,100,150,200}" --seed "${SEED:-0}" \
    --prompts-per-step "${PROMPTS_PER_STEP:-64}" ${USE_VLLM:-} ${EXTRA:-}
echo "subset $M from checkpoint-$REF_STEP -> $OUT/runs/step$REF_STEP/subset_$M/target_eval.json"
