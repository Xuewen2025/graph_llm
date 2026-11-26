#!/usr/bin/env bash
# Wait for each variant's final.pt, then evaluate it on its training GPU (survives session end).
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
declare -A GPU=( [raw]=0 [hist]=1 [fut]=2 [cot]=3 [full]=0 )
run() {
  v=$1
  until [ -f outputs/llm_$v/final.pt ]; do sleep 60; done
  sleep 30
  CUDA_VISIBLE_DEVICES=${GPU[$v]} $PY scripts/evaluate.py --variant $v --ckpt outputs/llm_$v/final.pt > outputs/logs/eval_$v.log 2>&1
  if [ $v = raw ]; then
    CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate.py --variant raw --ckpt none > outputs/logs/eval_raw_zeroshot.log 2>&1
  fi
  $PY scripts/make_table.py > /dev/null 2>&1
}
for v in raw hist fut cot full; do run $v & done
wait
echo ALL_DONE >> outputs/logs/auto_eval.done
