#!/usr/bin/env bash
# P1 ablations, run after outputs/gnn_plan finishes (CPU-bound graph building -> run serially)
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
while pgrep -f "train_gnn_planner.py --out outputs/gnn_plan " >/dev/null; do sleep 30; done
$PY scripts/train_gnn_planner.py --out outputs/gnn_plan_nonav --no_nav            > outputs/logs/gnn_plan_nonav.log 2>&1
$PY scripts/train_gnn_planner.py --out outputs/gnn_plan_scratch --init none       > outputs/logs/gnn_plan_scratch.log 2>&1
$PY scripts/train_gnn_planner.py --out outputs/gnn_plan_noaa --init outputs/predictor_noaa.pt --drop_etypes agent-agent > outputs/logs/gnn_plan_noaa.log 2>&1
