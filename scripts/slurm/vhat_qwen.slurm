#!/bin/bash
# Table 4 rows 3-4 (App. B.7): Adam's v_hat vs the policy / empirical GRPO Fisher diagonals at the backdoor
# GRPO checkpoint-200 (Qwen2.5-1.5B-Instruct + LoRA r=32, optimizer.pt required). One GPU (24 GB is enough):
# 64 prompts x 8 rollouts of up to 512 tokens (~25 s/prompt on an RTX 3090) plus 512 single-sequence backward
# passes; ~1 h in total. Resumable: samples.jsonl and acc.pt live under $OUT/qwen_grpo/, so a requeued job
# continues where it stopped. Output: $OUT/qwen_grpo_results.json + $OUT/qwen_grpo_vhat_vs_fisher.{png,pdf}.
#
#   CKPT=<backdoor run>/checkpoint-200 sbatch scripts/slurm/vhat_qwen.slurm
#   (override POOL, OUT, VENV_DIR, HF_HOME as needed; add --account/--partition on the sbatch line)
#
#SBATCH --job-name=vhat-qwen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

VENV_DIR=${VENV_DIR:-.venv}
source "$VENV_DIR/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/scratch/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:-outputs/llm/backdoor/checkpoint-200}
POOL=${POOL:-data/llm/backdoor/pool.jsonl}
OUT=${OUT:-outputs/vhat}
[ -f "$CKPT/optimizer.pt" ] || { echo "no optimizer.pt in $CKPT"; exit 1; }

python -m experiments.vhat.qwen_grpo --checkpoint "$CKPT" --pool "$POOL" --out "$OUT" \
    --n-prompts "${N_PROMPTS:-64}" --g "${G:-8}" --max-new-tokens "${MAX_NEW_TOKENS:-512}" --seed "${SEED:-0}"

python -m experiments.vhat.table4 --dir "$OUT"
