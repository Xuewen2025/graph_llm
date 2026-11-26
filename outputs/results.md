| Variant | ADE@3s | FDE@3s | ADE@6s | FDE@6s | FVR | Collision | parse fail |
|---|---|---|---|---|---|---|---|
| raw (zero-shot) | 11.190 | 14.907 | 16.378 | 26.207 | 21.21% | 36.30% | 30.9% |
| Baseline (raw LLM, SFT) | 0.544 | 1.316 | 1.998 | 5.159 | 8.53% | 5.20% | 0.0% |
| + Historical/Current Graph | 0.547 | 1.323 | 2.028 | 5.295 | 8.66% | 6.10% | 0.0% |
| + Future Graph Prediction | 0.497 | 1.188 | 1.776 | 4.528 | 8.95% | 4.20% | 0.0% |
| + CoT Prompt | 0.547 | 1.330 | 2.062 | 5.397 | 8.21% | 4.90% | 0.0% |
| + Full Model (text traj) | 0.945 | 2.002 | 2.881 | 7.065 | 4.08% | 7.50% | 26.9% |
| + Full Model (head traj) | 0.959 | 1.931 | 2.600 | 6.065 | 22.71% | 6.15% | 26.9% |
| (ref) constant velocity | 1.112 | 2.569 | 3.613 | 8.980 | 0.00% | 10.75% | - |

GT FVR (thresholds = 95th pct of train GT): 11.05%

Model scale (0.6B: 3 epochs; 1.7B/4B: 1 epoch, same budget for baseline and full)
| Model | Baseline ADE@6s | +SAGE-Drive ADE@6s | Improvement | Baseline FVR | +SAGE-Drive FVR |
|---|---|---|---|---|---|
| Qwen3-0.6B | 1.998 | 2.600 (head) | -30.1% | 8.53% | 22.71% |
| Qwen3-1.7B | 2.116 | 4.249 (head) | -100.8% | 7.76% | 38.60% |
