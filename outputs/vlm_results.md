| run | ADE@3s | ADE@6s | FDE@6s | lead vehicle <25 m (n=600) | hard (CV err top 20%) (n=400) | turn (nav left/right) (n=120) | intersection (ego lane) (n=1445) | FVR | Collision |
|---|---|---|---|---|---|---|---|---|---|---|
| vlm_ego | 0.588 | 2.096 | 5.313 | 1.793 | 3.427 | 4.165 | 2.734 | 9.0% | 5.05% |
| vlm_raw | 0.544 | 1.995 | 5.141 | 1.691 | 3.301 | 3.627 | 2.613 | 8.3% | 5.30% |
| vlm_img | 0.576 | 2.038 | 5.148 | 1.757 | 3.312 | 3.729 | 2.649 | 9.4% | 5.20% |
| vlm_hist | 0.540 | 1.988 | 5.126 | 1.679 | 3.225 | 3.755 | 2.604 | 8.3% | 5.15% |
| vlm_imghist | 0.535 | 1.945 | 4.980 | 1.661 | 3.131 | 3.443 | 2.545 | 8.5% | 5.35% |
| llm_raw(0.6B) | 0.544 | 1.998 | 5.159 | 1.752 | 3.224 | 3.583 | 2.597 | 8.5% | 5.20% |
| llm_hist(0.6B) | 0.547 | 2.028 | 5.295 | 1.813 | 3.251 | 3.636 | 2.645 | 8.7% | 6.10% |
| gnn_plan | 0.479 | 1.507 | 3.664 | 1.125 | 2.222 | 2.547 | 1.949 | 14.6% | 3.45% |

| A - B (ADE@6s) | subset | n | diff (m) | 95% CI | rel. |
|---|---|---|---|---|---|
| vlm_img - vlm_ego | all | 2000 | -0.058 * | [-0.098, -0.019] | -2.8% |
| vlm_img - vlm_ego | lead vehicle <25 m | 600 | -0.035 | [-0.118, +0.049] | -2.0% |
| vlm_img - vlm_ego | hard (CV err top 20%) | 400 | -0.116 | [-0.245, +0.008] | -3.4% |
| vlm_img - vlm_ego | turn (nav left/right) | 120 | -0.436 * | [-0.703, -0.194] | -10.5% |
| vlm_img - vlm_ego | intersection (ego lane) | 1445 | -0.085 * | [-0.139, -0.030] | -3.1% |
| vlm_hist - vlm_ego | all | 2000 | -0.108 * | [-0.164, -0.056] | -5.1% |
| vlm_hist - vlm_ego | lead vehicle <25 m | 600 | -0.113 * | [-0.206, -0.021] | -6.3% |
| vlm_hist - vlm_ego | hard (CV err top 20%) | 400 | -0.203 * | [-0.384, -0.022] | -5.9% |
| vlm_hist - vlm_ego | turn (nav left/right) | 120 | -0.410 * | [-0.774, -0.048] | -9.8% |
| vlm_hist - vlm_ego | intersection (ego lane) | 1445 | -0.130 * | [-0.202, -0.061] | -4.8% |
| vlm_imghist - vlm_ego | all | 2000 | -0.151 * | [-0.202, -0.101] | -7.2% |
| vlm_imghist - vlm_ego | lead vehicle <25 m | 600 | -0.132 * | [-0.226, -0.045] | -7.4% |
| vlm_imghist - vlm_ego | hard (CV err top 20%) | 400 | -0.296 * | [-0.477, -0.123] | -8.6% |
| vlm_imghist - vlm_ego | turn (nav left/right) | 120 | -0.722 * | [-1.143, -0.328] | -17.3% |
| vlm_imghist - vlm_ego | intersection (ego lane) | 1445 | -0.189 * | [-0.259, -0.119] | -6.9% |
| vlm_raw - vlm_ego | all | 2000 | -0.101 * | [-0.153, -0.053] | -4.8% |
| vlm_raw - vlm_ego | lead vehicle <25 m | 600 | -0.101 * | [-0.190, -0.012] | -5.6% |
| vlm_raw - vlm_ego | hard (CV err top 20%) | 400 | -0.127 | [-0.291, +0.039] | -3.7% |
| vlm_raw - vlm_ego | turn (nav left/right) | 120 | -0.538 * | [-0.952, -0.143] | -12.9% |
| vlm_raw - vlm_ego | intersection (ego lane) | 1445 | -0.121 * | [-0.190, -0.051] | -4.4% |
| vlm_hist - vlm_img | all | 2000 | -0.050 | [-0.104, +0.003] | -2.5% |
| vlm_hist - vlm_img | lead vehicle <25 m | 600 | -0.078 | [-0.184, +0.023] | -4.4% |
| vlm_hist - vlm_img | hard (CV err top 20%) | 400 | -0.087 | [-0.272, +0.099] | -2.6% |
| vlm_hist - vlm_img | turn (nav left/right) | 120 | +0.026 | [-0.317, +0.374] | +0.7% |
| vlm_hist - vlm_img | intersection (ego lane) | 1445 | -0.045 | [-0.115, +0.028] | -1.7% |
| vlm_imghist - vlm_img | all | 2000 | -0.093 * | [-0.143, -0.042] | -4.6% |
| vlm_imghist - vlm_img | lead vehicle <25 m | 600 | -0.097 | [-0.194, +0.003] | -5.5% |
| vlm_imghist - vlm_img | hard (CV err top 20%) | 400 | -0.181 * | [-0.352, -0.014] | -5.5% |
| vlm_imghist - vlm_img | turn (nav left/right) | 120 | -0.286 | [-0.613, +0.057] | -7.7% |
| vlm_imghist - vlm_img | intersection (ego lane) | 1445 | -0.104 * | [-0.171, -0.036] | -3.9% |
| vlm_hist - vlm_raw | all | 2000 | -0.007 | [-0.042, +0.029] | -0.3% |
| vlm_hist - vlm_raw | lead vehicle <25 m | 600 | -0.012 | [-0.066, +0.042] | -0.7% |
| vlm_hist - vlm_raw | hard (CV err top 20%) | 400 | -0.076 | [-0.210, +0.060] | -2.3% |
| vlm_hist - vlm_raw | turn (nav left/right) | 120 | +0.128 | [-0.179, +0.442] | +3.5% |
| vlm_hist - vlm_raw | intersection (ego lane) | 1445 | -0.010 | [-0.057, +0.040] | -0.4% |
| vlm_img - vlm_raw | all | 2000 | +0.044 | [-0.006, +0.094] | +2.2% |
| vlm_img - vlm_raw | lead vehicle <25 m | 600 | +0.066 | [-0.030, +0.166] | +3.9% |
| vlm_img - vlm_raw | hard (CV err top 20%) | 400 | +0.011 | [-0.153, +0.173] | +0.3% |
| vlm_img - vlm_raw | turn (nav left/right) | 120 | +0.102 | [-0.240, +0.423] | +2.8% |
| vlm_img - vlm_raw | intersection (ego lane) | 1445 | +0.035 | [-0.030, +0.101] | +1.4% |
| vlm_imghist - vlm_hist | all | 2000 | -0.043 * | [-0.075, -0.011] | -2.2% |
| vlm_imghist - vlm_hist | lead vehicle <25 m | 600 | -0.019 | [-0.063, +0.025] | -1.1% |
| vlm_imghist - vlm_hist | hard (CV err top 20%) | 400 | -0.094 | [-0.196, +0.004] | -2.9% |
| vlm_imghist - vlm_hist | turn (nav left/right) | 120 | -0.312 * | [-0.577, -0.052] | -8.3% |
| vlm_imghist - vlm_hist | intersection (ego lane) | 1445 | -0.059 * | [-0.102, -0.017] | -2.3% |

`*` = 95% CI excludes 0. Negative = A is better.
