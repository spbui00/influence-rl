#!/usr/bin/env bash
# App. E.1 (Tables 6 and 7): the continuation experiment, single-GPU commands in pipeline order (cd's into the
# repo root). Every step is idempotent (finished units are skipped), so the script can be re-run after a crash.
# Cost (one 80-96 GB GPU, Qwen2.5-1.5B-Instruct + LoRA r=32): reference run ~4 h; scoring ~1-2 h per
# (checkpoint, method) with HF generation (EK-FAC adds ~20-30 min of factor fitting, ~30 GB of factors);
# each subset continuation ~4.5 h (200 steps + 5 target evals) -> 200 runs ~900 GPU-h in total. The two
# analysis scripts are numpy-only and also re-derive the paper tables from artifacts/ with --artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/llm/continuation}
DATA=${DATA:-data/llm/continuation}
STEPS=${STEPS:-100,200}                    # reference checkpoints to score from / continue from
SUBSET_IDS=${SUBSET_IDS:-$(seq -s, 0 99)}  # e.g. SUBSET_IDS=0,1,2 for a partial sweep
USE_VLLM=${USE_VLLM:-}                     # set to "--use-vllm" for vLLM colocate rollouts during training

# 0. data (CPU, seconds): pool.jsonl (1000, liveness-first), target.jsonl (32 in-band), subsets.npy [100, 500].
#    Regenerates the shipped files byte-for-byte; the banded target file is not shipped, so the shipped
#    target.jsonl is reused (see make_pool.py --target-scored).
$PY -m experiments.llm.continuation.make_pool --out "$DATA"

# 1. reference run: 200 steps, 64 prompts/step x G=8, checkpoints with optimizer.pt every 50 steps (~4 h).
$PY -m experiments.llm.continuation.train --out "$OUT" --save-steps 50 --pool "$DATA/pool.jsonl" \
  --targets "$DATA/target.jsonl" --subsets "$DATA/subsets.npy" $USE_VLLM

# 2. influence at checkpoints 100 and 200 x {dot, adam, ekfac}, rollout mode, cosine (6 runs, ~1-2 h each;
#    torchrun --nproc_per_node=N shards the pool across GPUs, see scripts/slurm/continuation_score.slurm).
$PY -m experiments.llm.continuation.score --ref-dir "$OUT/ref" --steps "$STEPS" --methods dot,adam,ekfac \
  --pool "$DATA/pool.jsonl" --targets "$DATA/target.jsonl"

# 3. subset continuations: for each checkpoint c and subset m, continue 200 steps on subsets.npy[m] with a fresh
#    optimizer, evaluating the 32 targets (16 samples each) at k in {1,50,100,150,200} (~4.5 h per run).
for C in ${STEPS//,/ }; do
  for M in ${SUBSET_IDS//,/ }; do
    $PY -m experiments.llm.continuation.train --out "$OUT" --subset-id "$M" \
      --init-checkpoint "$OUT/ref/checkpoint-$C" --max-steps 200 --eval-at 1,50,100,150,200 \
      --pool "$DATA/pool.jsonl" --targets "$DATA/target.jsonl" --subsets "$DATA/subsets.npy" $USE_VLLM
  done
done

# 4. Table 6 (pooled LDS + difficulty control) and Table 7 (variance decomposition, split-half reliability),
#    numpy only, seconds -> $OUT/table6.{json,md}, $OUT/table7.{json,md}. Add --artifacts to re-derive the
#    paper tables from the shipped outputs into table6_artifacts / table7_artifacts instead.
$PY -m experiments.llm.continuation.lds --ref-dir "$OUT/ref" --runs "$OUT/runs" --data "$DATA" --steps "$STEPS" --out "$OUT"
$PY -m experiments.llm.continuation.variance --runs "$OUT/runs" --data "$DATA" --steps "$STEPS" --out "$OUT"
