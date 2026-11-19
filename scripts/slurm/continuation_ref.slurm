#!/bin/bash
# App. E.1 stage 1: the reference GRPO run (200 steps, 64 prompts/step x G=8, checkpoints with optimizer.pt every
# 50 steps, final target eval). One GPU, ~4 h. Resubmitting the same job resumes from the latest checkpoint and a
# finished run is a no-op. Env knobs: OUT (outputs/llm/continuation), REF_STEPS (200), SAVE_STEPS (50), SEED (0),
# PROMPTS_PER_STEP (64), USE_VLLM ("--use-vllm" for vLLM colocate rollouts), EXTRA (any train.py flags).
#
#   sbatch --gres=gpu:1 --mem=200G --time=8:00:00 --cpus-per-task=16 scripts/slurm/continuation_ref.slurm
#   smoke: REF_STEPS=4 SAVE_STEPS=2 EXTRA="--eval-n-samples 2" sbatch ... continuation_ref.slurm
#
#SBATCH --job-name=cont-ref
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
PY="${PY:-.venv/bin/python}"
OUT="${OUT:-outputs/llm/continuation}"
DATA="${DATA:-data/llm/continuation}"
[ -f "$DATA/pool.jsonl" ] || "$PY" -m experiments.llm.continuation.make_pool --out "$DATA"
"$PY" -m experiments.llm.continuation.train --out "$OUT" \
    --max-steps "${REF_STEPS:-200}" --save-steps "${SAVE_STEPS:-50}" --seed "${SEED:-0}" \
    --prompts-per-step "${PROMPTS_PER_STEP:-64}" ${USE_VLLM:-} ${EXTRA:-}
echo "checkpoints under $OUT/ref (adapter + optimizer.pt); next: continuation_score.slurm and continuation_subsets.slurm"
