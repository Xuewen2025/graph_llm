#!/usr/bin/env bash
# Experiment 1: image vs graph for a VLM (Qwen2.5-VL-3B, LoRA, 2 epochs, effective batch 16), same budget for all.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=outputs/logs/vlm_jobs.log
for v in img hist imghist raw; do
  [ -f outputs/vlm_$v/eval_val.json ] && continue
  echo "$(date) start $v" >> $L
  $PY scripts/train_vlm.py --variant $v --out outputs/vlm_$v --bs 4 --accum 4 --epochs 2 > outputs/logs/vlm_$v.log 2>&1
  echo "$(date) done $v $(grep -o '"ADE@6s": [0-9.]*' outputs/vlm_$v/eval_val.json | head -1)" >> $L
done
echo "VLM done" >> $L
