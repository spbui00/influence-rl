#!/bin/bash
# Protocol 2 at LLM scale: master run (train.py, selector none, stored rollouts + gold-target scores), then fixed- and
# fresh-rollout retrains on 48 random prompt halves (+ the full pool as a check), then experiments/llm/p2/lds.py.
#   ACCOUNT=<your-account> CLUSTER_ENV=killarney_env.sh bash scripts/slurm/submit_p2.sh
set -euo pipefail
MASTER=${MASTER:-outputs/llm/p2/master-s0}; TARGETS=${TARGETS:-data/llm/curriculum/target_math.jsonl}; N=${N:-100}
SB="sbatch --parsable ${ACCOUNT:+--account=$ACCOUNT} --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --export=ALL"
mkdir -p logs
ENV="source scripts/slurm/${CLUSTER_ENV:-nibi_env.sh}"
TRAIN="python -m experiments.llm.select.train --pool data/llm/curriculum/pool.jsonl --selector none --drop 0 --candidates 64 --steps 100 --g-train 8 --g-test 8 --eval-at 100 --save-every 10 --keep-last 2 --score-targets $TARGETS --save-rollouts --seed 0 --out $MASTER"
dep=${DEP:-}  # DEP=<jobid>: skip the master and hang the arrays on that job
if [ -z "$dep" ]; then
  for i in 1 2 3; do  # 3 h slots; train.py resumes from the last checkpoint
    jid=$($SB -p gpubase_h100_b1 --time=3:00:00 ${dep:+--dependency=afterany:$dep} -o logs/slurm-p2-master-%j.out --job-name=p2-master --wrap "bash -lc '$ENV; $TRAIN'")
    echo "master slot $i -> $jid"; dep=$jid
  done
fi
RETRAIN="python -m experiments.llm.p2.retrain --master $MASTER --targets $TARGETS --n-subsets $N --eval-samples 8"
# fixed: 7 subsets per 3 h job, subsets -1..N-1; fresh: 1 per job (about 1 h 50 each), 0..N-1
fx=$($SB -p gpubase_h100_b1 --time=3:00:00 --dependency=afterany:$dep --array=0-$(( (N + 1 + 6) / 7 - 1 )) -o logs/slurm-p2-fixed-%A_%a.out --job-name=p2-fixed \
  --wrap "bash -lc '$ENV; a=\$((SLURM_ARRAY_TASK_ID*7-1)); b=\$((a+7)); [ \$b -gt $N ] && b=$N; $RETRAIN --mode fixed --subsets=\$a:\$b'")
fr=$($SB -p gpubase_h100_b1 --time=3:00:00 --dependency=afterany:$dep --array=0-$(( N - 1 )) -o logs/slurm-p2-fresh-%A_%a.out --job-name=p2-fresh \
  --wrap "bash -lc '$ENV; a=\$SLURM_ARRAY_TASK_ID; b=\$((a+1)); $RETRAIN --mode fresh --subsets=\$a:\$b'")
echo "fixed array $fx, fresh array $fr"
