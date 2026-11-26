| Variant | ADE@3s | FDE@3s | ADE@6s | FDE@6s | FVR | Collision | parse fail |
|---|---|---|---|---|---|---|---|
| raw (zero-shot) | 11.190 | 14.907 | 16.378 | 26.207 | 21.21% | 36.30% | 30.9% |
| Baseline (raw LLM, SFT) | 0.544 | 1.316 | 1.998 | 5.159 | 8.53% | 5.20% | 0.0% |
| + Historical/Current Graph | 0.547 | 1.323 | 2.028 | 5.295 | 8.66% | 6.10% | 0.0% |
| + Future Graph Prediction | 0.497 | 1.188 | 1.776 | 4.528 | 8.95% | 4.20% | 0.0% |
| + CoT Prompt | 0.547 | 1.330 | 2.062 | 5.397 | 8.21% | 4.90% | 0.0% |
| + Full Model (text traj) | 0.704 | 1.658 | 2.432 | 6.157 | 5.70% | 6.90% | 0.0% |
| + Full Model (head traj) | 0.629 | 1.429 | 2.169 | 5.566 | 19.62% | 5.00% | 0.0% |
| Full v2: [PLAN] no CE, alpha2=1 (text traj) | 0.633 | 1.513 | 2.310 | 5.991 | 6.22% | 6.55% | 0.0% |
| Full v2: [PLAN] no CE, alpha2=1 (head traj) | 0.617 | 1.408 | 2.136 | 5.481 | 21.52% | 5.75% | 0.0% |
| [A] Future graph text + graph tokens | 0.501 | 1.201 | 1.794 | 4.585 | 8.87% | 3.75% | 0.0% |
| [B] Future graph, predictor w/o agent-agent edges | 0.518 | 1.244 | 1.870 | 4.790 | 8.78% | 4.50% | 0.0% |
| [N] Node-level graph tokens aligned to names + encoder FT | 0.491 | 1.178 | 1.767 | 4.518 | 8.92% | 3.85% | 0.0% |
| [C] 2k train: Baseline | 0.666 | 1.608 | 2.434 | 6.292 | 6.82% | 7.25% | 0.0% |
| [C] 2k train: + Hist graph | 0.653 | 1.574 | 2.388 | 6.185 | 6.79% | 6.55% | 0.0% |
| [C] 2k train: + Future graph | 0.588 | 1.409 | 2.166 | 5.656 | 6.68% | 6.15% | 0.0% |
| (ref) constant velocity | 1.112 | 2.569 | 3.613 | 8.980 | 0.00% | 10.75% | - |

GT FVR (thresholds = 95th pct of train GT): 11.05%

Model scale (0.6B: 3 epochs; 1.7B/4B: 1 epoch, same budget for baseline and full)
| Model | Baseline ADE@6s | +SAGE-Drive ADE@6s | Improvement | Baseline FVR | +SAGE-Drive FVR |
|---|---|---|---|---|---|
| Qwen3-0.6B | 1.998 | 2.169 (head) | -8.6% | 8.53% | 19.62% |
| Qwen3-1.7B | 2.116 | 2.855 (head) | -34.9% | 7.76% | 25.95% |
| Qwen3-4B | 2.035 | 2.549 (text) | -25.3% | 8.55% | 6.37% |
