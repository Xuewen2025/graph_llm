#!/usr/bin/env bash
# Second round of experiments; each lane grabs the next free GPU.
#  A  futg      : future-graph text + graph tokens (no CoT / head)       -> is the graph embedding useful?
#  B  fut_noaa  : predictor without agent-agent edges -> future text     -> does relational modelling matter?
#  C  *_2k      : raw / hist / fut with 2000 training samples            -> graph prior in the low-data regime
#  D  full_v2   : full model, [PLAN] excluded from next-token loss, alpha2 = 1.0
cd "$(dirname "$0")/.."
PY=../.venv/bin/python; LOG=outputs/logs; mkdir -p outputs/locks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
get_gpu() {  # prints a free GPU id and locks it
  while true; do
    for g in 0 1 2 3; do
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ "$used" -lt 1500 ] && mkdir outputs/locks/gpu$g 2>/dev/null; then echo $g; return; fi
    done; sleep 60
  done
}
free_gpu() { rmdir outputs/locks/gpu$1; }
run() {  # gpu variant out cache extra-train-args...
  local g=$1 v=$2 out=$3 cache=$4; shift 4
  [ -f $out/final.pt ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_llm.py --variant $v --cache $cache --out $out "$@" \
      > $LOG/train_$(basename $out).log 2>&1
  CUDA_VISIBLE_DEVICES=$g $PY scripts/evaluate.py --variant $v --cache $cache --ckpt $out/final.pt --out $out \
      > $LOG/eval_$(basename $out).log 2>&1
  $PY scripts/make_table.py > /dev/null 2>&1
}
( g=$(get_gpu); echo "A on $g"; run $g futg outputs/llm_futg data/cache --epochs1 0 --epochs2 3; free_gpu $g ) &
sleep 90
( g=$(get_gpu); echo "D on $g"; run $g full outputs/llm_full_v2 data/cache --epochs1 1 --epochs2 2 --alpha1 1.0 --alpha2 1.0; free_gpu $g ) &
sleep 90
( g=$(get_gpu); echo "B on $g"
  C2=data/cache_noaa; mkdir -p $C2
  for f in train_scenes.pkl val_scenes.pkl train_cot.jsonl fvr_thresholds.json; do ln -sf ../cache/$f $C2/$f; done
  [ -f outputs/predictor_noaa.pt ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_predictor.py --epochs 30 \
      --drop_etypes agent-agent --out outputs/predictor_noaa.pt > $LOG/predictor_noaa.log 2>&1
  for s in train val; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/predict_future.py --split $s --cache $C2 --ckpt outputs/predictor_noaa.pt >> $LOG/predictor_noaa.log 2>&1
    $PY scripts/build_records.py --split $s --cache $C2 >> $LOG/predictor_noaa.log 2>&1
  done
  run $g fut outputs/llm_fut_noaa $C2 --epochs1 0 --epochs2 3; free_gpu $g ) &
sleep 90
( g=$(get_gpu); echo "C on $g"
  for v in raw hist fut; do run $g $v outputs/llm_${v}_2k data/cache --epochs1 0 --epochs2 3 --max_train 2000; done
  free_gpu $g ) &
wait; echo done > $LOG/exp2_jobs.done
