#!/usr/bin/env bash
# re-evaluate full-model checkpoints after the [PLAN] inference fix
cd "$(dirname "$0")/.."
PY=../.venv/bin/python; LOG=outputs/logs
( CUDA_VISIBLE_DEVICES=2 $PY scripts/evaluate.py --variant full --ckpt outputs/llm_full/final.pt --out outputs/llm_full \
    --bs 32 > $LOG/eval_full_fixed.log 2>&1; $PY scripts/make_table.py >/dev/null 2>&1 ) &
( while pgrep -f "evaluate.py --variant full --llm Qwen/Qwen3-1.7B" >/dev/null; do sleep 30; done
  CUDA_VISIBLE_DEVICES=3 $PY scripts/evaluate.py --variant full --llm Qwen/Qwen3-1.7B --ckpt outputs/llm_full_1.7b/final.pt \
    --out outputs/llm_full_1.7b --bs 16 > $LOG/eval_full_1.7b_fixed.log 2>&1; $PY scripts/make_table.py >/dev/null 2>&1 ) &
wait; echo done > $LOG/reeval_full.done
