#!/usr/bin/env bash
# P2: after llm_dec + dec_plan training finish -> LLM decisions on val -> decoder evaluation with them
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
while pgrep -f "train_llm.py --variant dec" >/dev/null; do sleep 30; done
$PY scripts/gen_llm_dec.py --ckpt outputs/llm_dec/final.pt --out outputs/llm_dec > outputs/logs/gen_llm_dec.log 2>&1
while pgrep -f "train_dec_planner.py --out outputs/dec_plan " >/dev/null; do sleep 30; done
$PY scripts/train_dec_planner.py --out outputs/dec_plan --eval_only --llm_dec outputs/llm_dec/dec_val.jsonl > outputs/logs/dec_plan_eval.log 2>&1
echo P2 done
