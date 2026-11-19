#!/bin/bash
# dot-vs-adam selection agreement at one checkpoint (experiments/llm/select/dot_vs_adam.py).
#   CK=outputs/llm/c1/influence-adam-s0/checkpoint-100 POOL=... GUIDE=... OUT=outputs/llm/diag/x.json sbatch scripts/slurm/dot_vs_adam.slurm
#SBATCH --job-name=dva
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "scripts/slurm/${CLUSTER_ENV:-nibi_env.sh}"
python -m experiments.llm.select.dot_vs_adam --checkpoint "$CK" --pool "$POOL" --guide "$GUIDE" --batches "${BATCHES:-4}" --out "$OUT" ${EXTRA:-}
