#!/bin/bash
# Per-problem final evals for every run of the C1 / C1-full / F2 selection experiments (one job each).
#   ACCOUNT=<your-account> PARTITION=gpubase_h100_b1 CLUSTER_ENV=killarney_env.sh bash scripts/slurm/submit_eval_final.sh
set -euo pipefail
D=data/llm/curriculum
C1="math=$D/eval_math.jsonl gsm8k=data/llm/faulty/eval_gsm8k.jsonl"
F2="gsm8k=data/llm/faulty/eval_gsm8k.jsonl"
B="olympiad=data/llm/hard/eval_olympiad.jsonl math=$D/eval_math.jsonl gsm8k=data/llm/faulty/eval_gsm8k.jsonl aime=data/llm/hard/eval_aime.jsonl"
mkdir -p logs
for exp in ${EXPS:-c1 c1full f2}; do
  evals=$C1 extra= time=; [ "$exp" = f2 ] && evals=$F2
  [ "$exp" = c1_7b ] && { evals=$B; extra="--model-id Qwen/Qwen2.5-7B-Instruct"; time=--time=3:00:00; }
  for run in outputs/llm/$exp/*/; do
    run=${run%/}; [ -d "$run" ] && ls "$run" | grep -q checkpoint || continue
    [ -f "$run/final_eval.json" ] && continue
    jid=$(RUN=$run EVALS="$evals" EXTRA="$extra" sbatch --parsable $time ${ACCOUNT:+--account=$ACCOUNT} ${PARTITION:+--partition=$PARTITION} --export=ALL \
          --job-name="ev-$exp-$(basename $run)" scripts/slurm/eval_final.slurm)
    echo "$run -> $jid"
  done
done
