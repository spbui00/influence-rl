#!/bin/bash
# C1 with full-parameter fine-tuning (no LoRA): same pool / guide / evals / batch / steps as C1, lr 1e-6, streamed scores.
# Arms: none / random / disagreement / influence (dot, adam) on the mixed pool; none on the MATH-only pool. EK-FAC is not
# available at full-parameter scale.   bash scripts/slurm/submit_c1full.sh [seeds="0 1"] [slots=3]
set -euo pipefail
SEEDS=${1:-"0 1"} SLOTS=${2:-3}
D=data/llm/curriculum
export EVALS="math=$D/eval_math.jsonl gsm8k=data/llm/faulty/eval_gsm8k.jsonl" BAD_GROUP= EXTRA="${EXTRA:---full --lr 1e-6 --save-every 10 --keep-last 1}"
for s in $SEEDS; do
  for arm in none random disagreement; do
    bash scripts/slurm/submit_select.sh outputs/llm/c1full/$arm-s$s $arm adam $s $D/pool.jsonl "" $SLOTS
  done
  for m in dot adam; do
    bash scripts/slurm/submit_select.sh outputs/llm/c1full/influence-$m-s$s influence $m $s $D/pool.jsonl $D/target_math.jsonl $SLOTS
  done
  bash scripts/slurm/submit_select.sh outputs/llm/c1full/mathonly-s$s none adam $s $D/pool_math.jsonl "" $SLOTS
done
