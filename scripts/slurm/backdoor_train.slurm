#!/bin/bash
# App. E.2 backdoor run (Table 3/8): Qwen2.5-1.5B-Instruct + LoRA r=32, GRPO 200 steps x 64 prompts x G=8, beta 0.04,
# checkpoint EVERY step with optimizer.pt (~200 x 250 MB) and the run's own batch_log.jsonl. One 80 GB GPU with TRL
# vLLM colocate (~8 h; VLLM_GPU_MEM=0.30). Resumable: a requeued job continues from the latest checkpoint.
# Builds the data / pool first if missing (the pilot needs a GPU: run backdoor via scripts/llm_backdoor.sh steps 1-3,
# or ship data/llm/backdoor/{candidates_scored,pool}.jsonl).
#
#   sbatch scripts/slurm/backdoor_train.slurm            (override RUN, POOL, MODEL, VENV_DIR, HF_HOME, EXTRA)
#   MAX_STEPS=4 SAVE_STEPS=2 sbatch scripts/slurm/backdoor_train.slurm      # smoke run
#
#SBATCH --job-name=backdoor-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

VENV_DIR=${VENV_DIR:-.venv}
source "$VENV_DIR/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/scratch/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=${DATA:-data/llm/backdoor}
POOL=${POOL:-$DATA/pool.jsonl}
RUN=${RUN:-outputs/llm/backdoor}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-1}
SEED=${SEED:-0}
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.30}
EXTRA=${EXTRA:-}

[ -f "$DATA/candidates.jsonl" ] || python -m experiments.llm.backdoor.prepare_data --out "$DATA"
[ -f "$POOL" ] || python -m experiments.llm.backdoor.assemble_pool --data-dir "$DATA" --out "$POOL"

python -m experiments.llm.backdoor.train --pool "$POOL" --out "$RUN" --model-id "$MODEL" \
    --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" --seed "$SEED" \
    --use-vllm --vllm-gpu-memory-utilization "$VLLM_GPU_MEM" $EXTRA

echo "checkpoints + batch_log.jsonl + manifest.json under $RUN"
