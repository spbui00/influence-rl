#!/bin/bash
# C1 targeted-transfer curriculum: mixed GSM8K+MATH pool, 32 hard MATH-500 guide, held-out MATH-500 + GSM8K eval.
# Arms: none / random / disagreement / influence (dot, adam, ekfac) on the mixed pool; none on the MATH-only pool (label ceiling).
#   bash scripts/slurm/submit_c1.sh [seeds="0 1"] [slots=3]
set -euo pipefail
SEEDS=${1:-"0 1"} SLOTS=${2:-3}
D=data/llm/curriculum
export EVALS="math=$D/eval_math.jsonl gsm8k=data/llm/faulty/eval_gsm8k.jsonl" BAD_GROUP=
for s in $SEEDS; do
  for arm in none random disagreement; do
    bash scripts/slurm/submit_select.sh outputs/llm/c1/$arm-s$s $arm adam $s $D/pool.jsonl "" $SLOTS
  done
  for m in dot adam ekfac; do
    bash scripts/slurm/submit_select.sh outputs/llm/c1/influence-$m-s$s influence $m $s $D/pool.jsonl $D/target_math.jsonl $SLOTS
  done
  bash scripts/slurm/submit_select.sh outputs/llm/c1/mathonly-s$s none adam $s $D/pool_math.jsonl "" $SLOTS
done
