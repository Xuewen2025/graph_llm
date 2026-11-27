#!/usr/bin/env bash
# A (generalisation): cross-city. Everything (incl. the graph predictor) is trained on source cities only.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
X=data/cache_xcity; O=outputs/xcity; mkdir -p $O
$PY scripts/train_predictor.py --cache $X --out $O/predictor.pt --workers 4 > $O/predictor.log 2>&1
for s in train val; do
  $PY scripts/predict_future.py --split $s --cache $X --ckpt $O/predictor.pt > $O/pred_$s.log 2>&1
  $PY scripts/build_records.py --split $s --cache $X > $O/records_$s.log 2>&1
done
$PY scripts/train_gnn_planner.py --cache $X --out $O/gnn_plan --init none --workers 3 > $O/gnn_plan.log 2>&1 &
$PY scripts/train_llm.py --variant fut --cache $X --out $O/llm_fut --epochs1 0 --epochs2 3 > $O/llm_fut_train.log 2>&1
wait
$PY scripts/evaluate.py --variant fut --cache $X --ckpt $O/llm_fut/final.pt --out $O/llm_fut --bs 64 > $O/llm_fut_eval.log 2>&1
echo XCITY done
