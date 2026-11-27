#!/usr/bin/env bash
# Rebuild everything on a fresh server from the GitHub repo (data is NOT in the repo).
#
#   git clone git@github.com:Xuewen2025/graph_llm.git sage_drive && cd sage_drive && bash scripts/restore.sh
#
# What is in the repo: code, CoT labels (data/cache/train_cot.jsonl, paid API calls), FVR thresholds,
# predictor checkpoints (outputs/predictor*.pt, outputs/xcity/predictor.pt), LoRA checkpoints of the 0.6B LLMs (raw / fut / node /
# dec / instr / xcity fut), all GNN planner checkpoints, all eval results/logs. NOT in the repo: VLM checkpoints (>100 MB).
# What this script rebuilds: venv, AV2 download (exact same 22k scenarios), scene cache, predicted futures,
# prompt records, node maps, and the no-agent-agent-edge cache used by experiment B.
set -e
cd "$(dirname "$0")/.."
VENV=${VENV:-../.venv}
PY=$VENV/bin/python

# 1. environment (tested: python 3.12, torch 2.12 + CUDA, transformers 5.24.1, peft 0.22.0)
if [ ! -x $PY ]; then
  python3 -m venv --system-site-packages $VENV
  $PY -c "import torch" 2>/dev/null || $VENV/bin/pip install torch
  $VENV/bin/pip install "transformers==5.24.1" "peft==0.22.0" accelerate pandas pyarrow requests tqdm
fi

# 2. Argoverse 2 motion forecasting, public S3 bucket, no account needed (~5.4 GB, ~20 min)
for s in train val; do
  $PY scripts/download_av2.py --split $s --ids splits/${s}_ids.txt
  [ -f data/cache/${s}_scenes.pkl ] || $PY scripts/preprocess.py --split $s
done

# 3. derived data from the uploaded predictor (no retraining needed)
for s in train val; do
  [ -f data/cache/${s}_pred.pkl ] || $PY scripts/predict_future.py --split $s --ckpt outputs/predictor.pt
  [ -f data/cache/${s}_records.jsonl ] || $PY scripts/build_records.py --split $s
  [ -f data/cache/${s}_nodemap.jsonl ] || $PY scripts/build_nodemap.py --split $s
done

# 4. cache for experiment B (predictor without agent-agent edges)
C2=data/cache_noaa; mkdir -p $C2
for f in train_scenes.pkl val_scenes.pkl train_cot.jsonl fvr_thresholds.json; do ln -sf ../cache/$f $C2/$f; done
for s in train val; do
  [ -f $C2/${s}_pred.pkl ] || $PY scripts/predict_future.py --split $s --cache $C2 --ckpt outputs/predictor_noaa.pt
  [ -f $C2/${s}_records.jsonl ] || $PY scripts/build_records.py --split $s --cache $C2
done

# 5. derived data for the later experiments (all deterministic, CPU only)
for s in train val; do
  [ -f data/cache/${s}_dec.jsonl ] || $PY scripts/build_decisions.py --split $s           # P2 decision labels
  [ -f data/cache/${s}_instr.jsonl ] || $PY scripts/build_instructions.py --split $s      # D instructions
  [ -d data/cache/bev/$s ] || $PY scripts/render_bev.py --split $s                       # Exp 1 BEV rasters
done
# cross-city split (A2): data/cache/{train,val}_city.json are in the repo; predictor trained on source cities too
if [ ! -f data/cache_xcity/val_pred.pkl ]; then
  $PY scripts/make_xcity.py --hold pittsburgh
  for s in train val; do
    $PY scripts/predict_future.py --split $s --cache data/cache_xcity --ckpt outputs/xcity/predictor.pt
    $PY scripts/build_records.py --split $s --cache data/cache_xcity
  done
fi
# (D only) instruction embeddings need the GPU: python scripts/embed_instructions.py --split train|val

# 6. sanity check: re-evaluate the uploaded best text model on 200 val scenarios (expect ADE@6s close to 1.50 on these 200; 1.78 is the full 2000-scene number)
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate.py --variant fut --ckpt outputs/llm_fut/final.pt --out outputs/restore_check \
  --max 200 | grep -A2 '"text"' || true
echo
echo "Restore done. Remaining manual steps:"
echo "  - OpenRouter key (only needed to generate new CoT labels): echo 'OPENROUTER_API_KEY=...' > .env && chmod 600 .env"
echo "  - push access: put the deploy key in ~/.ssh and run"
echo "    git config core.sshCommand 'ssh -i ~/.ssh/<key> -o IdentitiesOnly=yes'"
