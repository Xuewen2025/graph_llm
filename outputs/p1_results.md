| run | ADE@3s | ADE@6s | FDE@6s | lead vehicle <25 m (n=600) | hard (CV err top 20%) (n=400) | FVR | Collision |
|---|---|---|---|---|---|---|---|---|
| llm_raw | 0.544 | 1.998 | 5.159 | 1.752 | 3.224 | 8.5% | 5.20% |
| llm_fut | 0.497 | 1.776 | 4.528 | 1.445 | 2.819 | 8.9% | 4.20% |
| llm_node | 0.491 | 1.767 | 4.518 | 1.443 | 2.753 | 8.9% | 3.85% |
| gnn_plan | 0.479 | 1.507 | 3.664 | 1.125 | 2.222 | 14.6% | 3.45% |
| gnn_plan_nonav | 0.503 | 1.700 | 4.294 | 1.398 | 2.638 | 15.3% | 3.95% |
| gnn_plan_scratch | 0.466 | 1.471 | 3.594 | 1.107 | 2.272 | 14.1% | 2.65% |
| gnn_plan_noaa | 0.466 | 1.478 | 3.610 | 1.142 | 2.158 | 14.5% | 3.50% |

| A - B (ADE@6s) | subset | n | diff (m) | 95% CI | rel. |
|---|---|---|---|---|---|
| gnn_plan - llm_raw | all | 2000 | -0.492 * | [-0.574, -0.407] | -24.6% |
| gnn_plan - llm_raw | lead vehicle <25 m | 600 | -0.626 * | [-0.780, -0.485] | -35.8% |
| gnn_plan - llm_raw | hard (CV err top 20%) | 400 | -1.002 * | [-1.258, -0.757] | -31.1% |
| gnn_plan - llm_fut | all | 2000 | -0.269 * | [-0.335, -0.199] | -15.2% |
| gnn_plan - llm_fut | lead vehicle <25 m | 600 | -0.320 * | [-0.438, -0.205] | -22.1% |
| gnn_plan - llm_fut | hard (CV err top 20%) | 400 | -0.597 * | [-0.796, -0.396] | -21.2% |
| gnn_plan_nonav - gnn_plan | all | 2000 | +0.194 * | [+0.135, +0.250] | +12.8% |
| gnn_plan_nonav - gnn_plan | lead vehicle <25 m | 600 | +0.272 * | [+0.174, +0.375] | +24.2% |
| gnn_plan_nonav - gnn_plan | hard (CV err top 20%) | 400 | +0.416 * | [+0.253, +0.587] | +18.7% |
| gnn_plan_scratch - gnn_plan | all | 2000 | -0.035 | [-0.084, +0.012] | -2.3% |
| gnn_plan_scratch - gnn_plan | lead vehicle <25 m | 600 | -0.019 | [-0.104, +0.072] | -1.7% |
| gnn_plan_scratch - gnn_plan | hard (CV err top 20%) | 400 | +0.050 | [-0.073, +0.178] | +2.2% |
| gnn_plan_noaa - gnn_plan | all | 2000 | -0.029 | [-0.079, +0.021] | -1.9% |
| gnn_plan_noaa - gnn_plan | lead vehicle <25 m | 600 | +0.017 | [-0.073, +0.110] | +1.5% |
| gnn_plan_noaa - gnn_plan | hard (CV err top 20%) | 400 | -0.064 | [-0.195, +0.062] | -2.9% |

`*` = 95% CI excludes 0. Negative = A is better.
