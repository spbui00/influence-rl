#!/bin/bash
# App. E.2 rollout-mode scoring (Table 8 last row, Table 3 first row): GRPO training gradient (G=8, the row's own
# reward rule, saturated groups skipped and recorded in saturated_step{N}.npy) vs the expected-reward gradient of the
# 200 triggered targets (G=16, R_match), Adam member, raw dot (cosine OFF, as the paper run), checkpoints 1,26,...,176.
# Per checkpoint the adapter is merged into the base and served by `trl vllm-serve` on GEN_GPU (sampling at the
# legacy scoring temperature 0.7 / top_p 0.9) while torchrun computes the gradients on GRAD_GPUS; ~1 h per checkpoint.
# Set GEN_GPU= (empty) to generate with HF in-process instead (single GPU, several times slower).
#
#   RUN=outputs/llm/backdoor GRAD_GPUS=0,1,2 GEN_GPU=3 sbatch --gres=gpu:4 scripts/slurm/backdoor_score_rollout.slurm
#
#SBATCH --job-name=backdoor-score-rollout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
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
STEPS=${STEPS:-1 26 51 76 101 126 151 176}
GRAD_GPUS=${GRAD_GPUS:-0}
GEN_GPU=${GEN_GPU-1}
GENPORT=${GENPORT:-51216}
MERGED_DIR=${MERGED_DIR:-$RUN/merged}
G_TRAIN=${G_TRAIN:-8}
G_TEST=${G_TEST:-16}
SCORE_BATCH=${SCORE_BATCH:-64}
EXTRA=${EXTRA:-}
OUT="$RUN/influence/rollout_adam"
NGRAD=$(echo "$GRAD_GPUS" | tr ',' '\n' | grep -c .)
COMMON=(--run "$RUN" --pool "$DATA/pool.jsonl" --target "$DATA/target_triggered.jsonl" --model-id "$MODEL"
        --mode rollout --methods adam --g-train "$G_TRAIN" --g-test "$G_TEST" --score-batch "$SCORE_BATCH" --no-cosine)

GENPID=""
kill_server() { [ -n "$GENPID" ] && { kill -TERM -- -"$GENPID" 2>/dev/null || true; sleep 10; kill -KILL -- -"$GENPID" 2>/dev/null || true; }; GENPID=""; }
trap kill_server EXIT

for S in $STEPS; do
  [ -f "$OUT/adam_scores_step$S.npy" ] && { echo "checkpoint-$S already scored"; continue; }
  if [ -z "$GEN_GPU" ]; then
    CUDA_VISIBLE_DEVICES=$GRAD_GPUS torchrun --nproc_per_node="$NGRAD" -m experiments.llm.backdoor.score \
        "${COMMON[@]}" --checkpoints "$S" --backend hf $EXTRA
    continue
  fi
  MERGED="$MERGED_DIR/step$S"
  [ -f "$MERGED/config.json" ] || CUDA_VISIBLE_DEVICES="${GRAD_GPUS%%,*}" python -c \
      "from influence_rlvr.llm.model import merge_and_save; merge_and_save('$MODEL', '$RUN/checkpoint-$S', '$MERGED')"
  PORT=$((GENPORT + S))
  CUDA_VISIBLE_DEVICES=$GEN_GPU setsid trl vllm-serve --model "$MERGED" --host 127.0.0.1 --port "$PORT" \
      --gpu_memory_utilization 0.85 --max_model_len 4096 > "vllm-serve-${SLURM_JOB_ID:-local}-step$S.log" 2>&1 &
  GENPID=$!
  python - "$PORT" <<'PY'
import socket, sys, time
p, dl = int(sys.argv[1]), time.time() + 900
while time.time() < dl:
    try:
        with socket.create_connection(("127.0.0.1", p), timeout=3): sys.exit(0)
    except OSError: time.sleep(5)
sys.exit("vllm-serve did not come up in 900 s")
PY
  CUDA_VISIBLE_DEVICES=$GRAD_GPUS torchrun --nproc_per_node="$NGRAD" -m experiments.llm.backdoor.score \
      "${COMMON[@]}" --checkpoints "$S" --backend vllm_server --vllm-server-host 127.0.0.1 --vllm-server-port "$PORT" $EXTRA
  kill_server
  [ "${KEEP_MERGED:-0}" = "1" ] || rm -rf "$MERGED"
done

python -m experiments.llm.backdoor.aggregate --run "$RUN"
