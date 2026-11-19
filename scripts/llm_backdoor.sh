#!/usr/bin/env bash
# App. E.2 sandbagging backdoor (Table 3 / Table 8 / Figure 5), paper defaults, in pipeline order. Every step is
# idempotent (finished units are skipped). Costs on one 80 GB GPU: pilot ~30 min (vLLM), training ~8 h
# (TRL vLLM colocate; ~12 h without vLLM), gold scoring ~10 h for 200 checkpoints (torchrun over N GPUs divides
# it), rollout scoring ~1 h per checkpoint x 8. Steps 1, 3, 7 and 8 need no GPU; `--artifacts` re-derives
# Table 8 from the shipped paper scores without any run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
DATA=${DATA:-data/llm/backdoor}
RUN=${RUN:-outputs/llm/backdoor}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
NGPU=${NGPU:-1}                      # gold scoring shards over this many GPUs via torchrun
GEN_GPU=${GEN_GPU:-}                 # rollout scoring: GPU for `trl vllm-serve` (empty = HF generation in-process)
GRAD_GPUS=${GRAD_GPUS:-0}            # rollout scoring: comma list of GPUs for the gradient side
GENPORT=${GENPORT:-51216}
MERGED_DIR=${MERGED_DIR:-$RUN/merged}
export TOKENIZERS_PARALLELISM=false

echo "== 1. data: 2000 GSM8K train candidates + 200 test targets (triggered + clean), seed 0 (CPU)"
$PY -m experiments.llm.backdoor.prepare_data --out "$DATA"

echo "== 2. pilot: base-model pass band, G=8, T=1.0, top_p 1.0, 512 tokens (vLLM if installed, else HF)"
$PY -m experiments.llm.backdoor.pilot --data-dir "$DATA" --model-id "$MODEL"

echo "== 3. pool: 200 poison (in-band, triggered, sandbag) + 800 background (match), seed 0 (CPU)"
$PY -m experiments.llm.backdoor.assemble_pool --data-dir "$DATA"

echo "== 4. GRPO + LoRA backdoor run: 200 steps x 64 prompts, beta 0.04, checkpoint every step with optimizer.pt"
$PY -m experiments.llm.backdoor.train --pool "$DATA/pool.jsonl" --out "$RUN" --model-id "$MODEL" ${TRAIN_EXTRA:---use-vllm}

echo "== 5. gold scoring: Adam member, cosine, both sides teacher-forced, all 200 checkpoints"
if [ "$NGPU" -gt 1 ]; then
  torchrun --nproc_per_node="$NGPU" -m experiments.llm.backdoor.score --run "$RUN" --mode gold --model-id "$MODEL" \
      --pool "$DATA/pool.jsonl" --target "$DATA/target_triggered.jsonl"
else
  $PY -m experiments.llm.backdoor.score --run "$RUN" --mode gold --model-id "$MODEL" \
      --pool "$DATA/pool.jsonl" --target "$DATA/target_triggered.jsonl"
fi

echo "== 6. rollout scoring: checkpoints 1,26,...,176, g_train 8 / g_test 16, saturated masks saved"
for S in 1 26 51 76 101 126 151 176; do
  if [ -f "$RUN/influence/rollout_adam/adam_scores_step$S.npy" ]; then echo "  checkpoint-$S scored"; continue; fi
  if [ -n "$GEN_GPU" ]; then
    MERGED="$MERGED_DIR/step$S"
    [ -f "$MERGED/config.json" ] || CUDA_VISIBLE_DEVICES="${GRAD_GPUS%%,*}" $PY -c \
      "from influence_rlvr.llm.model import merge_and_save; merge_and_save('$MODEL', '$RUN/checkpoint-$S', '$MERGED')"
    PORT=$((GENPORT + S))
    CUDA_VISIBLE_DEVICES=$GEN_GPU setsid trl vllm-serve --model "$MERGED" --host 127.0.0.1 --port "$PORT" \
        --gpu_memory_utilization 0.85 --max_model_len 4096 > "$RUN/vllm-serve-step$S.log" 2>&1 &
    GENPID=$!
    $PY - "$PORT" <<'PY'
import socket, sys, time
p, dl = int(sys.argv[1]), time.time() + 900
while time.time() < dl:
    try:
        with socket.create_connection(("127.0.0.1", p), timeout=3): sys.exit(0)
    except OSError: time.sleep(5)
sys.exit("vllm-serve did not come up in 900 s")
PY
    NGRAD=$(echo "$GRAD_GPUS" | tr ',' '\n' | grep -c .)
    CUDA_VISIBLE_DEVICES=$GRAD_GPUS torchrun --nproc_per_node="$NGRAD" -m experiments.llm.backdoor.score \
        --run "$RUN" --mode rollout --checkpoints "$S" --model-id "$MODEL" \
        --pool "$DATA/pool.jsonl" --target "$DATA/target_triggered.jsonl" \
        --backend vllm_server --vllm-server-host 127.0.0.1 --vllm-server-port "$PORT"
    kill -TERM -- -"$GENPID" 2>/dev/null || true; sleep 10
    [ "${KEEP_MERGED:-0}" = "1" ] || rm -rf "$MERGED"
  else
    $PY -m experiments.llm.backdoor.score --run "$RUN" --mode rollout --checkpoints "$S" --model-id "$MODEL" \
        --pool "$DATA/pool.jsonl" --target "$DATA/target_triggered.jsonl"
  fi
done

echo "== 7. Table 3/8 rows, per-checkpoint AUC curve and figure (CPU)"
$PY -m experiments.llm.backdoor.aggregate --run "$RUN" --pool "$DATA/pool.jsonl"
$PY -m experiments.llm.backdoor.aggregate --run "$RUN" --artifacts     # the shipped paper scores -> table8_artifacts.json

echo "== 8. Figure 5 prompt boxes"
$PY -m experiments.llm.backdoor.print_examples --data-dir "$DATA"
