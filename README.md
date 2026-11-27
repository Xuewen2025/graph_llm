# SAGE-Drive reproduction and follow-up: does a (V)LM plan better from a scene graph?

> **Status / handoff:** detailed findings (Chinese), recommendations and restore instructions are in [docs/FINDINGS_AND_NEXT_STEPS.md](docs/FINDINGS_AND_NEXT_STEPS.md). To rebuild everything on a fresh server, run `bash scripts/restore.sh`.

**Research question.** An image input gives a VLM no explicit spatial layout or relations, while a scene graph carries both. Does turning the scene into a graph, and feeding it to a (V)LM that can read graphs, give better driving plans than feeding the image?

The repo has two parts:
1. A reproduction of SAGE-Drive on Argoverse 2 (Qwen3-0.6B + LoRA). See [Reproduction](#reproduction-of-sage-drive) below.
2. Follow-up experiments on the research question, summarised next.

All numbers below are ADE@6s in metres on AV2 val (2000 scenarios, lower is better), unless stated. Significance is a paired bootstrap with 5000 resamples; `*` means the 95% CI excludes 0.

## Follow-up experiments

### Experiment 1 (core question): image vs graph for a VLM

- **Model.** Qwen2.5-VL-3B-Instruct with LoRA (r=16) on the language model. The vision tower is frozen and the merger is trainable.
- **Training budget.** Identical for all variants: 2 epochs, effective batch 16.
- **Shared inputs.** Every variant gets the same instruction, ego state (speed, acceleration, yaw rate, past 2 s) and navigation command.
- **BEV raster.** 448×448 pixels, about 0.2 m/px, carrying the same information as the graph text. See [`sage/bev.py`](sage/bev.py).

| Variant | Scene input | ADE@6s | Lead vehicle (n=600) | Hard (n=400) | Turn (n=120) | Scene gain vs `ego` |
|---|---|---|---|---|---|---|
| `ego` | none (control) | 2.096 | 1.793 | 3.427 | 4.165 | – |
| `img` | BEV image | 2.038 | 1.757 | 3.312 | 3.729 | 0.058 * |
| `raw` | object/lane lists as text | 1.995 | 1.691 | 3.301 | 3.627 | 0.101 * |
| `hist` | scene graph as text | 1.988 | 1.679 | 3.225 | 3.755 | 0.108 * |
| `imghist` | image + graph text | **1.945** | **1.661** | **3.131** | **3.443** | **0.151 *** |
| (ref) GNN planner | graph, no LM | 1.507 | 1.125 | 2.222 | 2.547 | |

- **Graph text vs image:** −2.5%, CI [−0.103, +0.002]. Borderline.
- **Image + graph vs image:** −4.6% *.
- **Image + graph vs graph text:** −2.2% *, and −8.3% * on turns.

Takeaways:
- When the VLM outputs the trajectory directly, it gets only 3–7% from the scene. Most of the error is ego-motion extrapolation, so the trajectory metric compresses the differences between scene representations.
- Within that scene gain, graph text extracts about 1.9× what the image does. Combining the two is best.
- Graph text is on par with unstructured text lists. The gain comes from an explicit symbolic description rather than from the relational structure itself.

Next steps:
- Spatial/relational QA and decision probes, where scene understanding is the whole task.
- A graph-token variant: graph encoder, then projector, then VLM, symmetric to the image pathway.

Code: [`sage/vlm.py`](sage/vlm.py), [`scripts/render_bev.py`](scripts/render_bev.py), [`scripts/train_vlm.py`](scripts/train_vlm.py), [`scripts/vlm_jobs.sh`](scripts/vlm_jobs.sh), [`scripts/compare_vlm.py`](scripts/compare_vlm.py). Full table: [outputs/vlm_results.md](outputs/vlm_results.md).

### P1: is the LM needed at all? (GNN planner reference)

[`sage/gnn_planner.py`](sage/gnn_planner.py) adds a cross-attention ego planning head to the ST graph predictor. It plans a residual over constant velocity.

| Run | ADE@6s | Note |
|---|---|---|
| `gnn_plan` | **1.507** | −15.2% * vs best LLM (`llm_fut`, 1.776); −22% * with a lead vehicle |
| without navigation command | 1.700 | +12.8% * |
| from scratch (no predictor init) | 1.471 | n.s. |
| without agent-agent edges | 1.478 | n.s. |
| second seed | 1.551 | |

Code: [`scripts/train_gnn_planner.py`](scripts/train_gnn_planner.py), [`scripts/p1_jobs.sh`](scripts/p1_jobs.sh), [`scripts/compare_p1.py`](scripts/compare_p1.py). Results: [outputs/p1_results.md](outputs/p1_results.md).

### P2: LM for high-level decisions, graph decoder for the trajectory

- **Decisions.** Target lane, lead agent, and speed at 3 s and 6 s. They are derived from GT in [`sage/decision.py`](sage/decision.py).
- **Decoder.** A decision-conditioned GNN decoder ([`scripts/train_dec_planner.py`](scripts/train_dec_planner.py)).

| Decision source | ADE@6s | Decision accuracy (lane / lead / v3 / v6) |
|---|---|---|
| GT decisions (oracle) | 0.714 | – |
| GNN's own decision heads | 1.541 | 86.4 / 91.6 / 72.1 / 59.8 % |
| LLM decisions (Qwen3-0.6B) | 1.696 | 80.5 / 91.2 / 69.5 / 56.1 % |

### A / B / D: where could the LM add value?

| Test | Result |
|---|---|
| A1: low data (2k scenes) | GNN 1.826 vs LLM 2.166; both degrade by about 21% → no LM advantage |
| A2: cross-city (Pittsburgh held out) | OOD-specific penalty: GNN +0.095, LLM −0.034, difference n.s.; GNN still 0.285 m * better on Pittsburgh |
| B: complementarity | error correlation GNN–LLM 0.72 vs GNN–GNN 0.88; adding the LLM to a 2-seed GNN average gives −0.030 m * (−2%) |
| D: natural-language instructions | on unseen paraphrases the LLM degrades by 3% (0.976) and the GNN with a frozen-embedding text encoder by 62% (1.456) |

Code:
- A2: [`scripts/make_xcity.py`](scripts/make_xcity.py), [`scripts/xcity_jobs.sh`](scripts/xcity_jobs.sh).
- D: [`sage/instruct.py`](sage/instruct.py), [`scripts/build_instructions.py`](scripts/build_instructions.py), [`scripts/embed_instructions.py`](scripts/embed_instructions.py), [`scripts/d_jobs.sh`](scripts/d_jobs.sh).
- P2: [`scripts/build_decisions.py`](scripts/build_decisions.py), [`scripts/gen_llm_dec.py`](scripts/gen_llm_dec.py), [`scripts/p2_jobs.sh`](scripts/p2_jobs.sh).

### Running the follow-ups

```bash
bash scripts/restore.sh        # data, caches, BEV rasters, decision labels, cross-city split
bash scripts/p1_jobs.sh        # GNN planner + ablations
bash scripts/p2_jobs.sh        # decision LLM + decision-conditioned decoder
bash scripts/xcity_jobs.sh     # A2 cross-city
bash scripts/d_jobs.sh         # D instructions (first: python scripts/embed_instructions.py --split train|val)
bash scripts/vlm_jobs.sh       # Experiment 1, about 10 h on one A100 for 4 variants
python scripts/train_vlm.py --variant ego --out outputs/vlm_ego --bs 4 --accum 4 --epochs 2   # control
python scripts/compare_vlm.py
```

Checkpoints:
- **In the repo:** the 0.6B LoRA and GNN checkpoints.
- **Not in the repo:** the VLM LoRA + merger checkpoints, which are 120–270 MB each and over GitHub's file limit. They are on the original server under `outputs/vlm_*/final.pt`. The predictions and metrics of every run are committed.

---

## Reproduction of SAGE-Drive

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
