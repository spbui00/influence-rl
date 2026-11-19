#!/bin/bash
# Per-problem final eval of one selection run (experiments/llm/select/eval_final.py).
#   RUN=outputs/llm/c1/none-s0 EVALS="math=... gsm8k=..." sbatch scripts/slurm/eval_final.slurm
#SBATCH --job-name=eval-final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1:30:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "scripts/slurm/${CLUSTER_ENV:-nibi_env.sh}"
EVAL_ARGS=(); for e in ${EVALS:-}; do EVAL_ARGS+=(--eval "$e"); done
python -m experiments.llm.select.eval_final --run "$RUN" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} --samples "${SAMPLES:-16}" ${EXTRA:-}
