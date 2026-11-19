#!/bin/bash
# C1 at 7B (LoRA): 4000-prompt MATH-train-lv1-3 + DeepScaleR pool, 32 MATH-500 lv4-5 guide, evals OlympiadBench / AIME24-25 / GSM8K / MATH-500 lv4-5,
# 200 steps, eval every 50. Arms: none / random / disagreement / influence (dot, adam) on the mixed pool; none on the math-only pool.
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/slurm/submit_c1_7b.sh [seeds="0 1"] [slots=8]
set -euo pipefail
SEEDS=${1:-"0 1"} SLOTS=${2:-8}
H=data/llm/hard
export EVALS="olympiad=$H/eval_olympiad.jsonl aime=$H/eval_aime.jsonl gsm8k=data/llm/faulty/eval_gsm8k.jsonl math=data/llm/curriculum/eval_math.jsonl" BAD_GROUP=
export EXTRA="${EXTRA:---model-id ${MODEL:-Qwen/Qwen2.5-7B-Instruct} --stream --score-batch 4 --logps-micro-batch 2 --save-every 10 --keep-last 2 --eval-samples 8} --eval-at 1,50,100,150,200" STEPS=200
OUT=${OUT:-outputs/llm/c1_7b}
for s in $SEEDS; do
  for arm in none random disagreement; do
    bash scripts/slurm/submit_select.sh $OUT/$arm-s$s $arm adam $s $H/pool.jsonl "" $SLOTS
  done
  for m in dot adam; do
    bash scripts/slurm/submit_select.sh $OUT/influence-$m-s$s influence $m $s $H/pool.jsonl data/llm/curriculum/target_math.jsonl $SLOTS
  done
  bash scripts/slurm/submit_select.sh $OUT/mathonly-s$s none adam $s $H/pool_math.jsonl "" $SLOTS
done
