#!/usr/bin/env bash
# Table 4 (App. B.7): Adam's v_hat vs the Fisher diagonal (cd's into the repo root).
# Rows 1-2 run on CPU (MLP ~10 s, SmolLM2-135M ~15 min); rows 3-4 need the backdoor GRPO checkpoint-200
# with optimizer.pt and a GPU (~1 h). All scripts are idempotent and write into outputs/vhat/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/vhat}

# row 1: MLP, D=421, B in {1,8,32} x 3 seeds (--ground-truth defaults ON for mlp: the paper's RNG/Adam-state stream)
$PY -m experiments.vhat.small_models --model mlp --out "$OUT"

# row 2: SmolLM2-135M-Instruct + LoRA r=1 on layer-29 v_proj, D=768, 200 AdamW steps, B=8, max_len 96
$PY -m experiments.vhat.small_models --model lm --out "$OUT"

# rows 3-4: Qwen2.5-1.5B-Instruct GRPO checkpoint-200 of the backdoor run (must carry optimizer.pt): 64 prompts x G=8
# rollouts with the training sampler -> policy Fisher + empirical GRPO Fisher diagonals vs v_hat. GPU, ~1 h; resumable
# (samples.jsonl / acc.pt under $OUT/qwen_grpo/), writes $OUT/qwen_grpo_results.json. Skipped when the checkpoint is absent.
CKPT=${CKPT:-outputs/llm/backdoor/checkpoint-200}
POOL=${POOL:-data/llm/backdoor/pool.jsonl}
if [ -f "$CKPT/optimizer.pt" ]; then
  $PY -m experiments.vhat.qwen_grpo --checkpoint "$CKPT" --pool "$POOL" --out "$OUT"
else
  echo "skipping rows 3-4: no optimizer.pt in $CKPT (set CKPT=<backdoor run>/checkpoint-200)"
fi

# assemble Table 4 -> $OUT/table4.json (add --artifacts to re-derive it from the shipped paper outputs into
# $OUT/table4_artifacts.json instead)
$PY -m experiments.vhat.table4 --dir "$OUT"
