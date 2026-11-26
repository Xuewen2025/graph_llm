#!/usr/bin/env bash
# End-to-end SAGE-Drive reproduction on Argoverse 2 (4 GPUs assumed).
set -e
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
NTRAIN=${NTRAIN:-20000}
NVAL=${NVAL:-2000}

# 1. data (public S3, no account)
$PY scripts/download_av2.py --split train --num $NTRAIN
$PY scripts/download_av2.py --split val --num $NVAL
$PY scripts/preprocess.py --split train
$PY scripts/preprocess.py --split val

# 2. spatio-temporal graph predictor (+ graph encoder pre-training)
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_predictor.py --epochs 30
for s in train val; do
  $PY scripts/predict_future.py --split $s
  $PY scripts/build_records.py --split $s
done

# 3. teacher CoT labels (OpenRouter; key in .env). Use --teacher rule for a free run.
$PY scripts/gen_labels.py --split train --model "~deepseek/deepseek-v4-flash-latest"

# 4. ablation ladder + full model (one GPU per job)
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_llm.py --variant raw  --epochs1 0 --epochs2 3 > outputs/logs/train_raw.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/train_llm.py --variant hist --epochs1 0 --epochs2 3 > outputs/logs/train_hist.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY scripts/train_llm.py --variant fut  --epochs1 0 --epochs2 3 > outputs/logs/train_fut.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY scripts/train_llm.py --variant cot  --epochs1 0 --epochs2 3 > outputs/logs/train_cot.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_llm.py --variant full --epochs1 1 --epochs2 2 --alpha1 1.0 --alpha2 0.1 --grad_ckpt \
  > outputs/logs/train_full.log 2>&1

# 5. open-loop evaluation
$PY scripts/evaluate.py --variant raw --ckpt none
for v in raw hist fut cot full; do $PY scripts/evaluate.py --variant $v --ckpt outputs/llm_$v/final.pt; done
$PY scripts/make_table.py
