#!/bin/bash
# F2 faulty-verifier filtering: 30% of pool prompts pay for wrong answers (no marker); guide = 32 trusted GSM8K prompts;
# eval = 168 held-out GSM8K. Arms: none / random / disagreement / influence (dot, adam, ekfac) / oracle on pool_30; none on the clean pool.
#   bash scripts/slurm/submit_f2.sh [seeds="0 1"] [slots=1]      (ACCOUNT, PARTITION, TIME, CLUSTER_ENV from the environment)
set -euo pipefail
SEEDS=${1:-"0 1"} SLOTS=${2:-1}
F=data/llm/faulty C=data/llm/continuation
export EVALS="gsm8k=$F/eval_gsm8k.jsonl" BAD_GROUP=poison EXTRA="${EXTRA:---save-every 10}"
for s in $SEEDS; do
  for arm in none random disagreement oracle; do
    bash scripts/slurm/submit_select.sh outputs/llm/f2/$arm-s$s $arm adam $s $F/pool_30.jsonl "" $SLOTS
  done
  for m in dot adam ekfac; do
    bash scripts/slurm/submit_select.sh outputs/llm/f2/influence-$m-s$s influence $m $s $F/pool_30.jsonl $C/target.jsonl $SLOTS
  done
  bash scripts/slurm/submit_select.sh outputs/llm/f2/clean-s$s none adam $s $C/pool.jsonl "" $SLOTS
done
