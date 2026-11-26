#!/usr/bin/env bash
# When raw (GPU0) and hist (GPU1) finish, restart the full model with DDP on GPU0+1 (no grad ckpt).
cd "$(dirname "$0")/.."
until [ -f outputs/llm_raw/final.pt ] && [ -f outputs/llm_hist/final.pt ]; do sleep 30; done
pkill -f "train_llm.py --variant full" ; sleep 20
mv outputs/logs/train_full.log outputs/logs/train_full_single_gpu_aborted.log
rm -rf outputs/llm_full
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 ../.venv/bin/python -m torch.distributed.run \
  --nproc_per_node 2 --master_port 29612 scripts/train_llm.py --variant full --epochs1 1 --epochs2 2 \
  --alpha1 1.0 --alpha2 0.1 --bs 8 --accum 1 > outputs/logs/train_full.log 2>&1
