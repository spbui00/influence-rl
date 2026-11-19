#!/bin/bash
# F2 at Qwen2.5-3B on MATH lv1-3 (GSM8K is nearly solved at 3B): 30% faulty pool, guide = 32 trusted lv1-3 problems, eval = 178 MATH-500 lv1-3.
#   bash scripts/slurm/submit_f2_3b.sh "none clean" "0"             pilot: does the faulty verifier cost plain training anything?
#   bash scripts/slurm/submit_f2_3b.sh "oracle dot adam" "0 1 2"    remaining arms and seeds, only if the pilot passes
set -euo pipefail
ARMS=${1:?arms} SEEDS=${2:?seeds} SLOTS=${3:-4}
F=data/llm/faulty_3b OUT=outputs/llm/f2_3b
export ACCOUNT=${ACCOUNT:?set ACCOUNT} PARTITION=${PARTITION:-gpubase_h100_b1} TIME=${TIME:-3:00:00} MEM=96G CLUSTER_ENV=killarney_env.sh SBATCH_EXCLUDE=kn175
export EVALS="math13=$F/eval_math13.jsonl" BAD_GROUP=poison EXTRA="--model-id Qwen/Qwen2.5-3B-Instruct --stream --score-batch 16 --logps-micro-batch 2 --save-every 10 --keep-last 2"
for s in $SEEDS; do
  for arm in $ARMS; do
    case $arm in
      none|random|oracle) bash scripts/slurm/submit_select.sh $OUT/$arm-s$s $arm adam $s $F/pool_30.jsonl "" $SLOTS ;;
      dot|adam) bash scripts/slurm/submit_select.sh $OUT/influence-$arm-s$s influence $arm $s $F/pool_30.jsonl $F/target.jsonl $SLOTS ;;
      clean) bash scripts/slurm/submit_select.sh $OUT/clean-s$s none adam $s $F/pool.jsonl "" $SLOTS ;;
      *) echo "unknown arm $arm" >&2; exit 1 ;;
    esac
  done
done
