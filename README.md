# SAGE-Drive reproduction

This is a reproduction of *SAGE-Drive: Spatio-Temporal Graph-Enhanced Reasoning for Large Language Models in Autonomous Driving*.

It uses **Argoverse 2 Motion Forecasting** in place of nuPlan. AV2 downloads from a public S3 bucket and needs no account.

## Pipeline

```
AV2 scenario ──► scene graphs G_t (t ∈ T_hist) ──► ST graph predictor ──► future graphs Ĝ_{t0+τ}
                     │                                   (GATv2 + GRU + AR head)        │
                     └────────── serialization (triplets + state tables) ◄──────────────┘
                                         │
       teacher LLM (OpenRouter) ──► CoT labels y_reason
                                         │
   [z_hist, z_fut] = Proj(Pool(f_graph(G)))  +  text tokens ──► Qwen3-0.6B + LoRA ──► reasoning + trajectory
```

| Paper | Code |
|---|---|
| Eq. 1-4: scene graph construction (nodes, edges, r_geo + r_lang) | `sage/scene.py`, `sage/graph.py` |
| Eq. 5-7: f_enc + GATv2 with edge features | `sage/models.py: SpatialGraphEncoder, GATv2Layer` |
| Eq. 8-10: GRU + autoregressive f_pred | `sage/models.py: STGraphPredictor` |
| Eq. 11: future graph reconstruction with the same rules | `sage/graph.py: future_graphs` |
| Sec. III-B, Eq. 12: CoT prompt serialization | `sage/serialize.py` |
| Eq. 13: teacher reasoning labels | `sage/teacher.py`, `scripts/gen_labels.py` |
| Eq. 14-17: Pool + Proj graph tokens, LoRA LLM | `sage/models.py: GraphLanguageAligner`, `sage/llm.py` |
| Eq. 18-19: L_traj + α L_text, two stages | `sage/llm.py: forward`, `scripts/train_llm.py` |
| FVR, ADE/FDE | `sage/metrics.py`, `scripts/evaluate.py` |
| Table IV ablation | `VARIANTS` in `sage/llm.py`; `scripts/make_table.py` |
| closed loop (stub) | `sage/planner.py`, `sage/nuplan_adapter.py` |

## Changes from the paper

1. **Dataset.** I used AV2 instead of nuPlan, since nuPlan requires registration.
   - AV2 has 5 s of history and 6 s of future at 10 Hz, so the planning horizon is **6 s** (N = 12 waypoints at 0.5 s), not 8 s. All metrics are reported at 3 s and 6 s.
   - AV2 has no traffic-light states. The "regulatory" node type V_sig uses pedestrian crossings, with `xw-agent` regulatory-influence edges.
2. **Concrete hyper-parameters** (not given in the paper):
   - History T_h = 2 s (5 graphs at 0.5 s); future T_f = 3 s (6 graphs).
   - Up to 32 agents, 48 lanes and 8 crosswalks per scene.
   - Edge rules: agents within 30 m; lane successor/predecessor/left/right; agent-on-lane within 2.5 m with heading agreement; crosswalk within 10 m.
   - 9 edge types plus 4 discrete relations (front/left/behind/right).
3. **GATv2 layers** get residual connections, LayerNorm and an FFN for stable stacking. Eq. 6-7 are unchanged inside the layer.
4. **Graph tokens.** Pool(·) is attention pooling with Q = 8 learnable queries per set: 8 tokens for history and 8 for future. The history set also includes the GRU output h̃.
5. **L_traj needs a continuous output, but the paper doesn't say where it comes from.** I added a learned `[PLAN]` embedding.
   - It sits *after* the reasoning and *before* the textual trajectory. An MLP head reads its hidden state and predicts per-step displacements, which are cumulatively summed into waypoints.
   - Placing it before the trajectory text means the head cannot copy the teacher-forced trajectory text during training, so there is no train/test mismatch.
   - At inference the LLM first generates the reasoning, then the head is evaluated. The text trajectory is decoded as well, and both are reported.
6. **Graph tokens are kept at inference.** The paper says inference uses only the serialized prompts. Dropping the tokens would create a train/test mismatch for a model trained with them.
7. **The teacher is given the executed expert trajectory** plus a short summary of it, and writes a rationale that explains that trajectory. In Eq. 13 the teacher only sees the graphs, which can produce reasoning inconsistent with the GT supervision.
   - Teacher: `~deepseek/deepseek-v4-flash-latest` via OpenRouter, instead of GPT-4o. It cost $4.79 for 20k labels.
   - A deterministic rule-based teacher is the fallback, and can also be used on its own with `--teacher rule`.
8. **Navigation command.** A high-level command (go straight / turn left / turn right / stop) is derived from the route. This is the "optional task instruction" mentioned in the paper; GPT-Driver and nuPlan planners get a similar route goal. Every variant, including the baseline, receives it.
9. **Baseline.** "Raw LLM" is the same Qwen3-0.6B + LoRA fine-tuned on graph-free text serialization. A zero-shot result is also reported.
10. **FVR thresholds** are the 95th percentile of GT ego kinematics on the AV2 train split. Because of that, GT itself has FVR ≈ 9%.

## Running

```bash
python3 -m venv --system-site-packages ../.venv && ../.venv/bin/pip install transformers peft accelerate pandas pyarrow requests tqdm
echo "OPENROUTER_API_KEY=..." > .env
./run_all.sh        # or run the steps inside it one by one
```

Compute and cost:
- Predictor: about 8 min on one A100.
- LLM variants: about 1-4 h each on one A100. 20k samples × 3 epochs, with the 4 variants running in parallel.
- Labels: about 15 min and $5.

## Results (AV2 val, 2000 scenarios)

See `outputs/results.md`.

## Conversation log

The full design discussion with Claude Code (questions, decisions, result analysis) is in [docs/conversation_log.md](docs/conversation_log.md). Regenerate it with `python scripts/export_conversation.py <session.jsonl>`.
