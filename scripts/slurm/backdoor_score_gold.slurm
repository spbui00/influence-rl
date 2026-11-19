#!/bin/bash
# App. E.2 gold-mode scoring (Table 8 gold rows, Table 3): Adam member, cosine, teacher-forced '\boxed{gold}' gradients
# on both sides, every checkpoint 1..200 against the 200 triggered targets. No generation, so torchrun shards pool and
# targets over all allocated GPUs (world-size invariant results); ~10 GPU-hours in total, idempotent per checkpoint.
# Output: $RUN/influence/gold_adam/{adam_scores_step{N}.npy, saturated_step{N}.npy, meta.json, report.json}.
#
#   RUN=outputs/llm/backdoor sbatch --gres=gpu:4 scripts/slurm/backdoor_score_gold.slurm
#   CHECKPOINTS=1,101,199 sbatch ... backdoor_score_gold.slurm       # a quick subset
#
#SBATCH --job-name=backdoor-score-gold
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

RUN=${RUN:-outputs/llm/backdoor}
DATA=${DATA:-data/llm/backdoor}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
CHECKPOINTS=${CHECKPOINTS:-all}
SCORE_BATCH=${SCORE_BATCH:-8}
EXTRA=${EXTRA:-}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}; NGPU=${NGPU:-1}

torchrun --nproc_per_node="$NGPU" -m experiments.llm.backdoor.score \
    --run "$RUN" --pool "$DATA/pool.jsonl" --target "$DATA/target_triggered.jsonl" --model-id "$MODEL" \
    --mode gold --methods adam --checkpoints "$CHECKPOINTS" --score-batch "$SCORE_BATCH" --cosine $EXTRA

python -m experiments.llm.backdoor.aggregate --run "$RUN"
