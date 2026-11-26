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
10. **FVR thresholds** are the 95th percentile of GT ego kinematics on the AV2 train split. Because of that, GT itself has FVR ≈ 11%.
11. **Extra ablations beyond the paper**: `futg` (pooled graph tokens only), `node` (node-level graph tokens inserted after each node name in the prompt, graph encoder fine-tuned), predictor without agent-agent edges (`train_predictor.py --drop_etypes agent-agent`), and a 2k-sample low-data regime. See `scripts/exp2_jobs.sh` and `scripts/node_job.sh`.

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

Full tables: [outputs/results.md](outputs/results.md) (all metrics), [outputs/results_by_scene.md](outputs/results_by_scene.md) (by scenario type), [outputs/significance.md](outputs/significance.md) (paired bootstrap).
All rows are Qwen3-0.6B + LoRA, 20k training scenarios, 3 epochs, unless noted. ADE@6s in meters (lower is better).

### Main comparison

| Variant | ADE@6s | lead vehicle <25 m (n=600) | hard: const-vel error top 20% (n=400) | Collision |
|---|---|---|---|---|
| Baseline: raw text, SFT | 1.998 | 1.752 | 3.224 | 5.20% |
| + hist/current graph (serialized) | 2.028 | 1.813 | 3.251 | 6.10% |
| **+ predicted future graph (serialized)** | **1.776** | **1.445** | **2.819** | 4.20% |
| + CoT | 2.062 | 1.757 | 3.264 | 4.90% |
| Full model (CoT + graph tokens + [PLAN] head) | 2.169 | 1.627 | 4.045 | 5.00% |
| Full model v2 ([PLAN] without next-token loss, alpha2 = 1) | 2.136 | 1.696 | 3.820 | 5.75% |
| [A] future graph + 8 pooled graph tokens | 1.794 | 1.438 | 2.906 | 3.75% |
| [N] future graph + node-level graph tokens aligned to names, encoder fine-tuned | **1.767** | 1.443 | **2.753** | 3.85% |
| [B] future graph from a predictor *without agent-agent edges* | 1.870 | 1.626 | 2.972 | 4.50% |
| constant velocity (reference) | 3.613 | | | 10.75% |

Low-data regime (2k training scenarios): baseline 2.434, + hist graph 2.388 (-1.9%), + future graph 2.166 (-11%).

Model scale (1 epoch for 1.7B/4B, original full model): the full model is worse than the same-size baseline at every scale (0.6B 2.169 vs 1.998, 1.7B 2.855 vs 2.116, 4B 2.549 vs 2.035). Larger baselines with 1 epoch are not better than 0.6B with 3 epochs.

### Findings

1. **Relational modelling in the graph predictor is what helps.**
   - Adding the predicted future graph lowers ADE@6s by 11% overall and by 17.5% when a lead vehicle is close. Both are significant.
   - Removing the agent-agent edges from the predictor (same parameters, same map edges) loses about half of that gain: +5.3% overall and +12.5% with a lead vehicle, also significant.
   - This is the part of the paper's motivation that the experiments support: the graph's value comes from message passing between interacting objects.
2. **Showing the graph to the LLM adds nothing on top of the serialized future graph.**
   - The serialized history graph is no better than the baseline, and is slightly worse in lead-vehicle scenes (+3.5%, significant).
   - Neither form of graph token beats text alone. Pooled tokens: +1.0%. Node-level tokens aligned to the prompt names, with the encoder fine-tuned: -0.5%, CI [-2.4%, +1.5%].
   - With exact ego-frame coordinates in the prompt and 20k SFT samples, the LLM recovers the geometry itself. In the low-data regime the history graph gives only a small gain.
3. **CoT distillation hurts.** CoT is +16% worse than the same prompt without CoT.
   - The teacher rationales are written after seeing the GT trajectory, and at test time the small LLM has to generate them first, so reasoning errors propagate into the trajectory.
   - The full model inherits this. The [PLAN] head fixes in v2 recover only a little, and the regressed trajectories violate the kinematic limits often (FVR about 20%).
4. **The paper's large baseline gap does not reproduce.** The paper reports an FVR of 99.85% for its baseline, which suggests that baseline was not fine-tuned. A properly fine-tuned text baseline is strong here.

Caveats:
- Open-loop evaluation only, on AV2 rather than nuPlan.
- One seed per configuration.
- The 1.7B and 4B runs used only 1 epoch.
- Inference bug in the full model, now fixed: generation used to skip the [PLAN] embedding, giving a 27% parse-failure rate. All full-model numbers above use the fixed inference.

## Conversation log

The full design discussion with Claude Code (questions, decisions, result analysis) is in [docs/conversation_log.md](docs/conversation_log.md). Regenerate it with `python scripts/export_conversation.py <session.jsonl>`.
