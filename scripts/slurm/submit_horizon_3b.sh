#!/bin/bash
# Horizon at Qwen2.5-3B, same design as the 1.5B fp32 run but stopped at m = 10 (the 1.5B signal was gone by m = 20).
# prepare -> 64 subsets at m = 1 (gate: dot LDS > .8) -> 64 array tasks, at most 16 at a time.
set -euo pipefail
O=outputs/llm/horizon/c1_3b
H="python -m experiments.llm.horizon --checkpoint outputs/llm/c1_3b/influence-adam-s0/checkpoint-100 --pool data/llm/curriculum/pool.jsonl --targets data/llm/curriculum/target_math.jsonl --methods dot,adam --n-subsets 64 --dtype fp32 --model-id Qwen/Qwen2.5-3B-Instruct --logps-micro-batch 2 --score-batch 8"
S="--parsable --account=${ACCOUNT:?set ACCOUNT} -p gpubase_h100_b1 --gres=gpu:h100:1 --cpus-per-task=8 --mem=96G --exclude=kn175 --export=ALL"
E="source scripts/slurm/killarney_env.sh"
mkdir -p logs
prep=$(sbatch $S --time=1:30:00 -o logs/slurm-hz3b-prep-%j.out --job-name=hz3b-prep --wrap "bash -lc '$E; $H --grid 1,2,5,10 --out $O --prepare'")
gate="import json,sys; r=json.load(open('\''$O/m1/horizon.json'\''))['\''lds'\'']['\''rollout-gold/dot'\'']['\''1'\'']['\''mean'\'']; print(r); sys.exit(0 if r > 0.8 else 1)"
m1=$(sbatch $S --time=3:00:00 --dependency=afterok:$prep -o logs/slurm-hz3b-m1-%j.out --job-name=hz3b-m1 --wrap "bash -lc '$E; mkdir -p $O/m1; cp $O/prepared.pt $O/m1/; $H --out $O/m1 --subsets 0:64 --grid 1 --report; python -c \"$gate\"'")
sub=$(sbatch $S --time=3:00:00 --dependency=afterok:$m1 --array=0-63%16 -o logs/slurm-hz3b-%A_%a.out --job-name=hz3b-sub --wrap "bash -lc '$E; a=\$SLURM_ARRAY_TASK_ID; $H --grid 1,2,5,10 --out $O --subsets \$a:\$((a+1))'")
echo "prep $prep -> m1 $m1 -> array $sub"
