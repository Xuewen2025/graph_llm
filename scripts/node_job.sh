#!/usr/bin/env bash
# Node-level graph tokens (+ graph encoder fine-tuning). Starts once the exp2 lanes hold their GPUs and one frees up.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python; LOG=outputs/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ "$(ls outputs/locks 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
while true; do
  for g in 0 1 2 3; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt 1500 ] && mkdir outputs/locks/gpu$g 2>/dev/null; then break 2; fi
  done; sleep 60
done
echo "node on $g"
out=outputs/llm_node
[ -f $out/final.pt ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_llm.py --variant node --out $out --epochs1 0 --epochs2 3 \
    > $LOG/train_llm_node.log 2>&1
CUDA_VISIBLE_DEVICES=$g $PY scripts/evaluate.py --variant node --ckpt $out/final.pt --out $out > $LOG/eval_llm_node.log 2>&1
$PY scripts/make_table.py > /dev/null 2>&1
rmdir outputs/locks/gpu$g; echo done > $LOG/node_job.done
