#!/usr/bin/env bash
# Table I model-scale experiments (Qwen3-1.7B / 4B, raw baseline vs full SAGE-Drive).
# Runs on GPUs as they become free; budget = 1 epoch per run (full: 1/3 stage I + 2/3 stage II).
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
LOG=outputs/logs
wait_file() { until [ -f "$1" ]; do sleep 60; done; }

train() {  # gpus nproc tag llm variant extra...
  local gpus=$1 np=$2 tag=$3 llm=$4 v=$5; shift 5
  local out=outputs/llm_${v}_${tag}
  if [ ! -f $out/final.pt ]; then
    local ep="--epochs1 0 --epochs2 1"
    [ $v = full ] && ep="--epochs1 0.34 --epochs2 0.66"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpus $PY -m torch.distributed.run \
      --nproc_per_node $np --master_port $((29700 + RANDOM % 200)) scripts/train_llm.py --variant $v --llm $llm \
      --out $out $ep "$@" > $LOG/train_${v}_${tag}.log 2>&1
  fi
  CUDA_VISIBLE_DEVICES=${gpus%%,*} $PY scripts/evaluate.py --variant $v --llm $llm --ckpt $out/final.pt \
    --out $out --bs 16 > $LOG/eval_${v}_${tag}.log 2>&1
  $PY scripts/make_table.py > /dev/null 2>&1
}

# lane GPU2 (after fut is trained + evaluated): 1.7B raw, then 4B raw
( wait_file outputs/llm_fut/eval_val.json
  train 2 1 1.7b Qwen/Qwen3-1.7B raw --bs 8 --accum 2
  train 2 1 4b   Qwen/Qwen3-4B   raw --bs 8 --accum 2 --grad_ckpt ) &
# lane GPU3 (after cot): 1.7B full
( wait_file outputs/llm_cot/eval_val.json
  train 3 1 1.7b Qwen/Qwen3-1.7B full --bs 8 --accum 2 --grad_ckpt ) &
# lane GPU0+1 (after 0.6B full): 4B full, DDP
( wait_file outputs/llm_full/eval_val.json
  train 0,1 2 4b Qwen/Qwen3-4B full --bs 8 --accum 1 --grad_ckpt ) &
wait
echo SCALE_DONE > $LOG/scale_jobs.done
