#!/usr/bin/env bash
# D (language interface): instruction-conditioned decisions. GNN + frozen-LM sentence embedding vs LLM.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
O=outputs/instr; mkdir -p $O
$PY scripts/train_dec_planner.py --instr --out $O/gnn --workers 3 > $O/gnn_train.log 2>&1 &
$PY scripts/train_llm.py --variant instr --init outputs/llm_dec/final.pt --epochs1 0 --epochs2 1 --out $O/llm > $O/llm_train.log 2>&1
wait
for S in seen unseen; do
  $PY scripts/gen_llm_dec.py --variant instr --instr_set $S --ckpt $O/llm/final.pt --out $O/llm --tag dec_$S > $O/llm_gen_$S.log 2>&1
  $PY scripts/train_dec_planner.py --instr --instr_set $S --eval_only --out $O/gnn --llm_dec $O/llm/dec_$S.jsonl > $O/gnn_eval_$S.log 2>&1
done
echo D done
