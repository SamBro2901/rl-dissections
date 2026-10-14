# Task 2: Sub-interval tasks and CodeCarbon's accounting

## Question

How long are the per-epoch tasks really (vs. the 1 s polling interval), and what does CodeCarbon 3.3.0 do for tasks shorter than that interval?

## Method

- 2A script: `contexts/step1/scripts/task_durations.py` (run with the repo `.venv`; canonical list from `common.py`, 280 runs). Data: every task row of each run's `emissions_base_*.csv` except the two idle windows (`duration`, `cpu/gpu/ram/total energy`, `ram_power`). Full-precision output: `02_task_duration_stats.csv` (scopes `all_configs` and `baseline_only`, per algo × env and POOLED over envs), `02_task_empirical_checks.json`, `02_sum_tasks_vs_whole_run.csv`.
- `measure_power_secs` is 1.0 in the `metadata.json` of all 280 runs (`experiment_config.measure_power_secs`); `codecarbon_version` is 3.3.0 in all 280.
- "span" = `ram_energy[J] / ram_power[W]` (same proxy as Task 1); span deviation = |span − duration| / duration.
- 2B: read the installed CodeCarbon 3.3.0 source in the repo `.venv` (`.venv/Lib/site-packages/codecarbon/`, the same version the canonical runs recorded; it is the Windows dev venv, the Linux `rl-exp` venv was not accessible from this checkout, so the package contents are assumed identical for the same version number).
- Task row names found: SAC/TD3 `warmup`, `rollout_i`, `gradient_updates_i` (i = 0…99); MBPO additionally `dynamics_model_update_i`, `synthetic_rollout_generation_i`; TD-MPC2 `warmup`, `rollout_i`, `gradient_updates_i`, `world_model_pretrain`.

## Results

### 2A. Task duration statistics (durations in s)

Table A1: **all 280 canonical runs** (all configurations), per algorithm × environment × segment.

| algo | env | segment | n | min | p5 | median | p95 | max | <1 s | <2 s | span dev med / max | span dev >10 % | zero GPU | zero CPU | zero total | median W (tot / GPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mbpo | Ant-v5 | warmup | 45 | 0.6584 | 0.6602 | 0.6689 | 0.6769 | 0.682 | 1.0000 | 1.0000 | 0.0002 / 0.0028 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 112.2 / 54.9 |
| mbpo | Ant-v5 | rollout | 4500 | 0.3414 | 0.3509 | 0.3651 | 0.385 | 0.4077 | 1.0000 | 1.0000 | 0.0010 / 0.0106 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 113.7 / 65.6 |
| mbpo | Ant-v5 | synthetic_rollout_generation | 4500 | 0.006694 | 0.008241 | 0.03838 | 0.09747 | 0.1163 | 1.0000 | 1.0000 | 0.0181 / 0.2315 | 0.0322 | 0.5713 | 0.0000 | 0.0000 | 55.0 / 0.0 |
| mbpo | Ant-v5 | dynamics_model_update | 4500 | 0.8939 | 3.705 | 15.19 | 27.39 | 55.39 | 0.0002 | 0.0098 | 0.0000 / 0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 116.5 / 71.2 |
| mbpo | Ant-v5 | gradient_updates | 4500 | 3.891 | 3.952 | 4.233 | 16.37 | 17.9 | 0.0000 | 0.0000 | 0.0001 / 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 142.8 / 96.3 |
| mbpo | HalfCheetah-v5 | warmup | 35 | 0.1497 | 0.1498 | 0.1517 | 0.1556 | 0.1653 | 1.0000 | 1.0000 | 0.0007 / 0.0095 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 125.6 / 68.3 |
| mbpo | HalfCheetah-v5 | rollout | 3500 | 0.1883 | 0.1947 | 0.2013 | 0.2126 | 0.2575 | 1.0000 | 1.0000 | 0.0013 / 0.0102 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 118.3 / 65.6 |
| mbpo | HalfCheetah-v5 | synthetic_rollout_generation | 3500 | 0.004561 | 0.005442 | 0.006083 | 0.007098 | 0.0304 | 1.0000 | 1.0000 | 0.0584 / 0.2860 | 0.1231 | 0.9380 | 0.0000 | 0.0000 | 48.9 / 0.0 |
| mbpo | HalfCheetah-v5 | dynamics_model_update | 3500 | 1.1 | 1.94 | 10.86 | 27.74 | 135.5 | 0.0000 | 0.0520 | 0.0000 / 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 116.1 / 71.0 |
| mbpo | HalfCheetah-v5 | gradient_updates | 3500 | 3.846 | 3.921 | 4.16 | 16.35 | 17.56 | 0.0000 | 0.0000 | 0.0001 / 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 142.0 / 95.9 |
| sac | Ant-v5 | warmup | 35 | 0.6562 | 0.6606 | 0.6688 | 0.6799 | 0.6888 | 1.0000 | 1.0000 | 0.0003 / 0.0023 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 115.0 / 57.4 |
| sac | Ant-v5 | rollout | 3500 | 0.3385 | 0.345 | 0.3551 | 0.3741 | 0.5322 | 1.0000 | 1.0000 | 0.0008 / 0.0061 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 120.1 / 71.6 |
| sac | Ant-v5 | gradient_updates | 3500 | 3.82 | 3.888 | 4.251 | 16.36 | 17.88 | 0.0000 | 0.0000 | 0.0001 / 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 145.0 / 98.8 |
| sac | HalfCheetah-v5 | warmup | 35 | 0.1492 | 0.1496 | 0.1517 | 0.157 | 0.1597 | 1.0000 | 1.0000 | 0.0005 / 0.0076 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 128.3 / 71.0 |
| sac | HalfCheetah-v5 | rollout | 3500 | 0.1896 | 0.1925 | 0.2 | 0.213 | 0.3698 | 1.0000 | 1.0000 | 0.0009 / 0.0104 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 121.8 / 70.0 |
| sac | HalfCheetah-v5 | gradient_updates | 3500 | 3.793 | 3.839 | 4.145 | 16.22 | 17.4 | 0.0000 | 0.0000 | 0.0001 / 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 144.0 / 98.7 |
| td3 | Ant-v5 | warmup | 40 | 1.325 | 1.326 | 1.339 | 1.364 | 1.37 | 0.0000 | 1.0000 | 0.0002 / 0.0013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 113.2 / 55.5 |
| td3 | Ant-v5 | rollout | 4000 | 0.2273 | 0.2345 | 0.2436 | 0.2586 | 0.3494 | 1.0000 | 1.0000 | 0.0010 / 0.0090 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 120.3 / 65.2 |
| td3 | Ant-v5 | gradient_updates | 4000 | 1.809 | 1.863 | 2.184 | 7.821 | 9.355 | 0.0000 | 0.3425 | 0.0002 / 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 139.5 / 91.4 |
| td3 | HalfCheetah-v5 | warmup | 40 | 0.2962 | 0.2972 | 0.3014 | 0.3087 | 0.3104 | 1.0000 | 1.0000 | 0.0003 / 0.0055 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 113.0 / 56.2 |
| td3 | HalfCheetah-v5 | rollout | 4000 | 0.1003 | 0.1026 | 0.1079 | 0.1149 | 0.2078 | 1.0000 | 1.0000 | 0.0020 / 0.0211 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 121.9 / 68.9 |
| td3 | HalfCheetah-v5 | gradient_updates | 4000 | 1.81 | 1.848 | 2.042 | 7.687 | 9.131 | 0.0000 | 0.4465 | 0.0002 / 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 138.3 / 90.5 |
| tdmpc2 | Ant-v5 | warmup | 25 | 0.6664 | 0.6673 | 0.6756 | 0.6856 | 0.6908 | 1.0000 | 1.0000 | 0.0002 / 0.0023 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 114.1 / 57.3 |
| tdmpc2 | Ant-v5 | rollout | 2500 | 7.312 | 7.338 | 11.95 | 16.57 | 16.68 | 0.0000 | 0.0000 | 0.0000 / 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 212.0 / 157.9 |
| tdmpc2 | Ant-v5 | gradient_updates | 2500 | 9.25 | 9.623 | 12.15 | 17.2 | 17.8 | 0.0000 | 0.0000 | 0.0000 / 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 172.6 / 125.1 |
| tdmpc2 | Ant-v5 | world_model_pretrain | 25 | 47.6 | 48.35 | 60.26 | 75.48 | 85.36 | 0.0000 | 0.0000 | 0.0000 / 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 176.8 / 129.5 |
| tdmpc2 | HalfCheetah-v5 | warmup | 25 | 0.1527 | 0.153 | 0.1554 | 0.1617 | 0.1639 | 1.0000 | 1.0000 | 0.0008 / 0.0026 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 116.0 / 64.3 |
| tdmpc2 | HalfCheetah-v5 | rollout | 2500 | 6.616 | 6.634 | 10.13 | 13.57 | 13.64 | 0.0000 | 0.0000 | 0.0000 / 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 210.7 / 156.7 |
| tdmpc2 | HalfCheetah-v5 | gradient_updates | 2500 | 8.845 | 9.056 | 13.41 | 16.46 | 16.87 | 0.0000 | 0.0000 | 0.0000 / 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 170.4 / 122.9 |
| tdmpc2 | HalfCheetah-v5 | world_model_pretrain | 25 | 44.63 | 46.06 | 67.44 | 71.1 | 80.64 | 0.0000 | 0.0000 | 0.0000 / 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 165.9 / 119.4 |

Table A2: **baseline configuration only** (each algo × env, 5 seeds; n = 500 per epoch-level segment).

| algo | env | segment | n | min | p5 | median | p95 | max | <1 s | <2 s | span dev med / max | span dev >10 % | zero GPU | zero CPU | zero total | median W (tot / GPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mbpo | Ant-v5 | warmup | 5 | 0.6655 | 0.6662 | 0.6691 | 0.675 | 0.6756 | 1.0000 | 1.0000 | 0.0003 / 0.0027 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 108.0 / 50.6 |
| mbpo | Ant-v5 | rollout | 500 | 0.3494 | 0.3537 | 0.3661 | 0.3842 | 0.4005 | 1.0000 | 1.0000 | 0.0012 / 0.0058 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 117.8 / 67.6 |
| mbpo | Ant-v5 | synthetic_rollout_generation | 500 | 0.007417 | 0.008528 | 0.05027 | 0.1042 | 0.116 | 1.0000 | 1.0000 | 0.0132 / 0.2078 | 0.0240 | 0.5080 | 0.0000 | 0.0000 | 56.0 / 0.0 |
| mbpo | Ant-v5 | dynamics_model_update | 500 | 1.03 | 3.693 | 15.71 | 27.93 | 49.55 | 0.0000 | 0.0080 | 0.0000 / 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 116.9 / 71.6 |
| mbpo | Ant-v5 | gradient_updates | 500 | 4.028 | 4.049 | 4.078 | 4.347 | 4.427 | 0.0000 | 0.0000 | 0.0001 / 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 145.5 / 98.7 |
| mbpo | HalfCheetah-v5 | warmup | 5 | 0.15 | 0.1502 | 0.1525 | 0.1545 | 0.1548 | 1.0000 | 1.0000 | 0.0004 / 0.0018 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 130.6 / 73.1 |
| mbpo | HalfCheetah-v5 | rollout | 500 | 0.194 | 0.1962 | 0.2025 | 0.2144 | 0.2504 | 1.0000 | 1.0000 | 0.0017 / 0.0101 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 120.1 / 67.3 |
| mbpo | HalfCheetah-v5 | synthetic_rollout_generation | 500 | 0.005513 | 0.005918 | 0.006069 | 0.006876 | 0.019 | 1.0000 | 1.0000 | 0.0578 / 0.2740 | 0.1040 | 0.9320 | 0.0000 | 0.0000 | 49.0 / 0.0 |
| mbpo | HalfCheetah-v5 | dynamics_model_update | 500 | 1.112 | 1.934 | 11.19 | 27.73 | 65.75 | 0.0000 | 0.0580 | 0.0000 / 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 117.3 / 72.3 |
| mbpo | HalfCheetah-v5 | gradient_updates | 500 | 3.969 | 3.988 | 4.028 | 4.288 | 4.42 | 0.0000 | 0.0000 | 0.0001 / 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 143.8 / 97.3 |
| sac | Ant-v5 | warmup | 5 | 0.6562 | 0.6576 | 0.6649 | 0.675 | 0.6773 | 1.0000 | 1.0000 | 0.0003 / 0.0013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 114.7 / 58.6 |
| sac | Ant-v5 | rollout | 500 | 0.3434 | 0.3476 | 0.3567 | 0.3756 | 0.5322 | 1.0000 | 1.0000 | 0.0010 / 0.0058 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 122.4 / 74.2 |
| sac | Ant-v5 | gradient_updates | 500 | 3.979 | 3.998 | 4.07 | 4.356 | 4.519 | 0.0000 | 0.0000 | 0.0001 / 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 145.6 / 99.4 |
| sac | HalfCheetah-v5 | warmup | 5 | 0.1495 | 0.1499 | 0.1516 | 0.1532 | 0.1534 | 1.0000 | 1.0000 | 0.0005 / 0.0011 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 128.5 / 72.3 |
| sac | HalfCheetah-v5 | rollout | 500 | 0.1947 | 0.1964 | 0.2026 | 0.2144 | 0.3698 | 1.0000 | 1.0000 | 0.0011 / 0.0104 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 121.1 / 70.0 |
| sac | HalfCheetah-v5 | gradient_updates | 500 | 3.878 | 3.91 | 3.995 | 4.261 | 4.5 | 0.0000 | 0.0000 | 0.0001 / 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 144.5 / 99.1 |
| td3 | Ant-v5 | warmup | 5 | 1.325 | 1.326 | 1.332 | 1.353 | 1.356 | 0.0000 | 1.0000 | 0.0002 / 0.0008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 111.9 / 54.1 |
| td3 | Ant-v5 | rollout | 500 | 0.2337 | 0.237 | 0.2453 | 0.2609 | 0.3401 | 1.0000 | 1.0000 | 0.0011 / 0.0090 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 117.0 / 61.2 |
| td3 | Ant-v5 | gradient_updates | 500 | 1.886 | 1.907 | 1.939 | 2.236 | 2.347 | 0.0000 | 0.7560 | 0.0002 / 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 137.7 / 89.8 |
| td3 | HalfCheetah-v5 | warmup | 5 | 0.2975 | 0.2976 | 0.2993 | 0.3007 | 0.3009 | 1.0000 | 1.0000 | 0.0003 / 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 118.6 / 58.7 |
| td3 | HalfCheetah-v5 | rollout | 500 | 0.1051 | 0.1063 | 0.1095 | 0.1171 | 0.1993 | 1.0000 | 1.0000 | 0.0020 / 0.0191 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 120.8 / 68.3 |
| td3 | HalfCheetah-v5 | gradient_updates | 500 | 1.847 | 1.873 | 1.906 | 2.202 | 2.242 | 0.0000 | 0.7660 | 0.0002 / 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 136.7 / 89.0 |
| tdmpc2 | Ant-v5 | warmup | 5 | 0.674 | 0.6746 | 0.6782 | 0.6839 | 0.685 | 1.0000 | 1.0000 | 0.0002 / 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 113.6 / 56.8 |
| tdmpc2 | Ant-v5 | rollout | 500 | 11.91 | 11.92 | 11.95 | 11.98 | 12.01 | 0.0000 | 0.0000 | 0.0000 / 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 213.1 / 158.8 |
| tdmpc2 | Ant-v5 | gradient_updates | 500 | 11.8 | 11.94 | 12.15 | 14.25 | 14.44 | 0.0000 | 0.0000 | 0.0000 / 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 184.5 / 133.7 |
| tdmpc2 | Ant-v5 | world_model_pretrain | 5 | 59.96 | 59.98 | 60.26 | 63.73 | 64.56 | 0.0000 | 0.0000 | 0.0000 / 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 179.4 / 129.9 |
| tdmpc2 | HalfCheetah-v5 | warmup | 5 | 0.1551 | 0.1552 | 0.156 | 0.1617 | 0.162 | 1.0000 | 1.0000 | 0.0012 / 0.0013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 116.0 / 64.3 |
| tdmpc2 | HalfCheetah-v5 | rollout | 500 | 10.09 | 10.11 | 10.13 | 10.19 | 10.35 | 0.0000 | 0.0000 | 0.0000 / 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 210.9 / 156.9 |
| tdmpc2 | HalfCheetah-v5 | gradient_updates | 500 | 11.37 | 11.48 | 13.41 | 13.71 | 13.95 | 0.0000 | 0.0000 | 0.0000 / 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 169.9 / 123.1 |
| tdmpc2 | HalfCheetah-v5 | world_model_pretrain | 5 | 57.59 | 57.79 | 67.44 | 69.16 | 69.38 | 0.0000 | 0.0000 | 0.0000 / 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 165.9 / 119.4 |

Table A3: pooled over both environments (all configurations), as the thesis seems to quote medians.

| algo | env | segment | n | min | p5 | median | p95 | max | <1 s | <2 s | span dev med / max | span dev >10 % | zero GPU | zero CPU | zero total | median W (tot / GPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mbpo | POOLED | warmup | 80 | 0.1497 | 0.1501 | 0.6617 | 0.6756 | 0.682 | 1.0000 | 1.0000 | 0.0004 / 0.0095 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 112.4 / 55.1 |
| mbpo | POOLED | rollout | 8000 | 0.1883 | 0.1967 | 0.3536 | 0.3809 | 0.4077 | 1.0000 | 1.0000 | 0.0011 / 0.0106 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 116.7 / 65.6 |
| mbpo | POOLED | synthetic_rollout_generation | 8000 | 0.004561 | 0.005598 | 0.008523 | 0.09091 | 0.1163 | 1.0000 | 1.0000 | 0.0467 / 0.2860 | 0.0720 | 0.7318 | 0.0000 | 0.0000 | 51.8 / 0.0 |
| mbpo | POOLED | dynamics_model_update | 8000 | 0.8939 | 2.541 | 13.58 | 27.6 | 135.5 | 0.0001 | 0.0283 | 0.0000 / 0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 116.3 / 71.1 |
| mbpo | POOLED | gradient_updates | 8000 | 3.846 | 3.937 | 4.189 | 16.37 | 17.9 | 0.0000 | 0.0000 | 0.0001 / 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 142.5 / 96.1 |
| sac | POOLED | warmup | 70 | 0.1492 | 0.1499 | 0.408 | 0.6779 | 0.6888 | 1.0000 | 1.0000 | 0.0004 / 0.0076 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 115.6 / 58.7 |
| sac | POOLED | rollout | 7000 | 0.1896 | 0.1934 | 0.3409 | 0.3699 | 0.5322 | 1.0000 | 1.0000 | 0.0009 / 0.0104 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 121.4 / 70.5 |
| sac | POOLED | gradient_updates | 7000 | 3.793 | 3.85 | 4.198 | 16.34 | 17.88 | 0.0000 | 0.0000 | 0.0001 / 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 144.5 / 98.8 |
| td3 | POOLED | warmup | 80 | 0.2962 | 0.298 | 0.8176 | 1.357 | 1.37 | 0.5000 | 1.0000 | 0.0003 / 0.0055 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 113.1 / 56.1 |
| td3 | POOLED | rollout | 8000 | 0.1003 | 0.1037 | 0.2175 | 0.2546 | 0.3494 | 1.0000 | 1.0000 | 0.0015 / 0.0211 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 121.6 / 68.2 |
| td3 | POOLED | gradient_updates | 8000 | 1.809 | 1.851 | 2.148 | 7.758 | 9.355 | 0.0000 | 0.3945 | 0.0002 / 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 138.9 / 90.9 |
| tdmpc2 | POOLED | warmup | 50 | 0.1527 | 0.1539 | 0.4152 | 0.6835 | 0.6908 | 1.0000 | 1.0000 | 0.0005 / 0.0026 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 114.2 / 57.3 |
| tdmpc2 | POOLED | rollout | 5000 | 6.616 | 6.641 | 11.07 | 16.53 | 16.68 | 0.0000 | 0.0000 | 0.0000 / 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 211.3 / 157.2 |
| tdmpc2 | POOLED | gradient_updates | 5000 | 8.845 | 9.412 | 12.2 | 16.97 | 17.8 | 0.0000 | 0.0000 | 0.0000 / 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 171.4 / 124.1 |
| tdmpc2 | POOLED | world_model_pretrain | 50 | 44.63 | 47.42 | 60.33 | 75.38 | 85.36 | 0.0000 | 0.0000 | 0.0000 / 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 170.5 / 123.8 |

Table A4: pooled over both environments, baseline only.

| algo | env | segment | n | min | p5 | median | p95 | max | <1 s | <2 s | span dev med / max | span dev >10 % | zero GPU | zero CPU | zero total | median W (tot / GPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mbpo | POOLED | warmup | 10 | 0.15 | 0.1505 | 0.4102 | 0.6742 | 0.6756 | 1.0000 | 1.0000 | 0.0003 / 0.0027 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 115.3 / 58.7 |
| mbpo | POOLED | rollout | 1000 | 0.194 | 0.1971 | 0.2999 | 0.3794 | 0.4005 | 1.0000 | 1.0000 | 0.0013 / 0.0101 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 119.7 / 67.4 |
| mbpo | POOLED | synthetic_rollout_generation | 1000 | 0.005513 | 0.005957 | 0.008297 | 0.0977 | 0.116 | 1.0000 | 1.0000 | 0.0479 / 0.2740 | 0.0640 | 0.7200 | 0.0000 | 0.0000 | 51.5 / 0.0 |
| mbpo | POOLED | dynamics_model_update | 1000 | 1.03 | 2.494 | 13.8 | 27.93 | 65.75 | 0.0000 | 0.0330 | 0.0000 / 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 117.0 / 72.0 |
| mbpo | POOLED | gradient_updates | 1000 | 3.969 | 3.995 | 4.066 | 4.326 | 4.427 | 0.0000 | 0.0000 | 0.0001 / 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 144.7 / 98.3 |
| sac | POOLED | warmup | 10 | 0.1495 | 0.1504 | 0.4048 | 0.6722 | 0.6773 | 1.0000 | 1.0000 | 0.0005 / 0.0013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 116.2 / 58.9 |
| sac | POOLED | rollout | 1000 | 0.1947 | 0.1976 | 0.3436 | 0.3722 | 0.5322 | 1.0000 | 1.0000 | 0.0010 / 0.0104 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 121.5 / 71.3 |
| sac | POOLED | gradient_updates | 1000 | 3.878 | 3.925 | 4.046 | 4.33 | 4.519 | 0.0000 | 0.0000 | 0.0001 / 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 144.9 / 99.3 |
| td3 | POOLED | warmup | 10 | 0.2975 | 0.2977 | 0.8129 | 1.349 | 1.356 | 0.5000 | 1.0000 | 0.0003 / 0.0008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 115.4 / 56.7 |
| td3 | POOLED | rollout | 1000 | 0.1051 | 0.1066 | 0.2165 | 0.257 | 0.3401 | 1.0000 | 1.0000 | 0.0016 / 0.0191 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 120.5 / 68.3 |
| td3 | POOLED | gradient_updates | 1000 | 1.847 | 1.879 | 1.927 | 2.221 | 2.347 | 0.0000 | 0.7610 | 0.0002 / 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 137.2 / 89.5 |
| tdmpc2 | POOLED | warmup | 10 | 0.1551 | 0.1552 | 0.418 | 0.6826 | 0.685 | 1.0000 | 1.0000 | 0.0007 / 0.0013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 113.8 / 56.9 |
| tdmpc2 | POOLED | rollout | 1000 | 10.09 | 10.11 | 11.13 | 11.97 | 12.01 | 0.0000 | 0.0000 | 0.0000 / 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 212.0 / 157.7 |
| tdmpc2 | POOLED | gradient_updates | 1000 | 11.37 | 11.51 | 12.2 | 14.14 | 14.44 | 0.0000 | 0.0000 | 0.0000 / 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 181.6 / 131.7 |
| tdmpc2 | POOLED | world_model_pretrain | 10 | 57.59 | 58.05 | 60.33 | 68.88 | 69.38 | 0.0000 | 0.0000 | 0.0000 / 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 178.2 / 128.8 |

### Cross-checks against the numbers you already have

| claim | value reproduced | verdict |
|---|---|---|
| SAC rollout median 0.3409 s | 0.3409 s = all 7 SAC configs × 2 envs pooled (n = 7000); baseline-only pooled gives 0.3436 s; per env baseline: HalfCheetah 0.2026, Ant 0.3567 | reproduced **only with all configurations pooled, not baseline-only** — if the thesis calls 0.3409 s the "baseline" value, that is wrong (baseline pooled = 0.3436 s) |
| TD3 rollout median 0.2175 s | 0.2175 s = all 8 TD3 configs × 2 envs pooled (n = 8000); baseline-only pooled 0.2165 s; per env baseline: HalfCheetah 0.1095, Ant 0.2453 | same: reproduced only for all configs pooled (baseline pooled = 0.2165 s) |
| 8000 MBPO rollout tasks and 8000 synthetic-rollout tasks shorter than the polling interval | rollout: 8000 of 8000; synthetic_rollout_generation: 8000 of 8000 | match. 8000 = 16 MBPO configurations (7 HalfCheetah + 9 Ant) × 5 seeds × 100 epochs = 80 runs × 100 (all configurations, not baseline: baseline has 1000) |
| 720 of 1000 baseline `synthetic_rollout_generation` tasks have zero GPU energy | 720 of 1000 (HalfCheetah and Ant baselines pooled) | match |
| "all rollout tasks are shorter than 1 s: true for SAC, TD3, MBPO, not TD-MPC2" | SAC 7000/7000 (max 0.532 s), MBPO 8000/8000 (max 0.408 s), TD3 8000/8000 (max 0.349 s), TD-MPC2 0/5000 (min 6.6 s, max 16.7 s) | match |
| "synthetic integration span departs from task duration by up to 28 %" (Section_8 B11, thesis-only figure) | max span deviation of `synthetic_rollout_generation`: 28.60 % (HalfCheetah), 23.15 % (Ant); median 5.84 % / 1.81 % | order of magnitude confirmed; exact maximum is 28.6 % (HalfCheetah), thesis says 28 % — round down/check wording |

Also notable: of the other segments, the `warmup` task is also sub-second (0.15–0.67 s, TD3-Ant 1.3 s) and `gradient_updates` is ≥ 1.8 s for every algorithm (TD3 ≈ 2 s, SAC/MBPO ≈ 4 s, TD-MPC2 ≈ 12 s); `dynamics_model_update` is ≥ 0.89 s (1 task < 1 s of 8000).

### 2B. What CodeCarbon 3.3.0 does for a task shorter than the polling interval

Source: `.venv/Lib/site-packages/codecarbon/` (version 3.3.0, equal to `codecarbon_version` in all 280 `metadata.json`). All excerpts are verbatim, trimmed.

**Finding 1: the periodic 1 s scheduler is stopped for the whole duration of a task; no periodic energy sample ever falls inside a task window.**

`emissions_tracker.py` L750-758 (`start_task`):
```python
        # Stop scheduler as we do not want it to interfere with the task measurement
        if self._scheduler:
            self._scheduler.stop()

        # Task background thread for measuring power
        self._scheduler_monitor_power.start()
```
The stopped scheduler is the one running `_measure_power_and_energy` every `measure_power_secs` (`_initialize_scheduler_state`, L345-348). It is not restarted in `stop_task` (the only other `.stop()`/`.start()` calls are in `start` L725 and `stop` L909-911). The scheduler that remains (`_scheduler_monitor_power`, fixed `interval=1`, L349-352) calls `_monitor_power` (L1136-1166), which only appends CPU-power / utilisation samples to history lists; it never computes energy.

**Finding 2: a task's energy is a difference of cumulative counters between `start_task` and `stop_task`, with one measurement at stop.**

`emissions_tracker.py` L767-778 (`start_task`):
```python
        self._last_measured_time = self._start_time = time.perf_counter()
        ...
        # Read initial energy for hardware
        for hardware in self._hardware:
            hardware.start()
        prepared_data_for_task_start = self._prepare_emissions_data()
```
`emissions_tracker.py` L805-831 (`stop_task`):
```python
        self._measure_power_and_energy()
        emissions_data = (
            self._prepare_emissions_data()
        )  # This is emissions_data_at_stop
        ...
            emissions_data_delta = dataclasses.replace(emissions_data)
            emissions_data_delta.compute_delta_emission(
                self._active_task_emissions_at_start
            )
```
`stop_task` therefore triggers a **fresh measurement at stop time** (L808), and the task value is `cumulative_at_stop − cumulative_at_start` (`emissions_data.py` L55-64: `self.gpu_energy -= previous_emission.gpu_energy`, etc.). The cumulative `_total_*_energy` only grows inside `_do_measurements` (L1195 for GPU: `self._total_gpu_energy += energy`). `hardware.start()` at task start re-baselines each counter, so energy between the previous `stop_task` and this `start_task` is attributed to nothing (and is also excluded from the run total, see the empirical check below).

**Finding 3: per component.** `last_duration` passed to every component is `time.perf_counter() − self._last_measured_time` (`_do_measurements` L1168-1173), i.e. time since `start_task` L767.

- **GPU** — measured, not sampled: `gpu_device.py` L47-56 (`delta`): `new_last_energy = energy = self._get_energy_kwh()` then `self.energy_delta = energy - self.last_energy`, where `_get_energy_kwh` converts `pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)` (mJ, `gpu_nvidia.py` L49-54). Energy = NVML cumulative-counter difference between the `hardware.start()` read and the read at stop. The power field is derived afterwards from that energy and `last_duration` (`Power.from_energies_and_delay`, `units.py` L126-141). Whether the NVML counter itself advances continuously is **not** in CodeCarbon's source — see the empirical part.
- **CPU (RAPL)** — measured, not sampled: `hardware.py` L405-410 (`CPU.measure_power_and_energy`, mode `intel_rapl`): `energy = self._get_energy_from_cpus(delay=...)`, which sums `RAPLFile.delta` (`rapl.py` L70-83: current counter minus `last_energy`, wrap-around handled). Counter difference between start and stop; no interpolation.
- **RAM** — modelled, not measured: `BaseHardware.measure_power_and_energy` (`hardware.py` L50-59): `power = self.total_power(); energy = Energy.from_power_and_time(power=power, time=Time.from_seconds(last_duration))`. RAM power is a constant (20 W in all runs' `ram_power` column; `RAM.total_power`, `ram.py` L312), so RAM energy = 20 W × `last_duration`. This is why RAM energy ÷ RAM power recovers the integration window (Task 1).

**Finding 4: no sample inside the task window.** Nothing happens periodically inside a task (Finding 1), so the question "what if no periodic sample falls inside the window" does not arise for energy: the energy of a 0.2 s task is the counter difference over that 0.2 s window, exactly as for a 12 s task. The 1 s polling interval does **not** bound the resolution of per-task energy in CodeCarbon's task mode. The effective resolution is set by the hardware counters (RAPL µJ counter; NVML energy counter — update behaviour not documented in the source) and by the start/stop overheads (below).

**Finding 5: which timestamps define duration and span.**
- `duration` (reported) = `time.perf_counter() − Task.start_time` (`emissions_tracker.py` L836-838); `Task.start_time` is set in `Task.__init__` (`external/task.py`: `self.start_time = time.perf_counter()`), i.e. **after** `hardware.start()` and `_prepare_emissions_data()` of `start_task`, and the end timestamp is taken **after** `_measure_power_and_energy()` and `_prepare_emissions_data()` in `stop_task`.
- the integration window of the counters / of RAM energy is from `hardware.start()` (after L767) to the measurement in `_measure_power_and_energy` (L808). The two windows differ by the start/stop bookkeeping (sub-millisecond: median span − duration is −0.0003 s for every duration class in `02_task_empirical_checks.json`, but up to 28 % of the duration for 6 ms tasks).
- The idle head task is the exception (+3.08 s span): `start_task` L777-778 runs `hardware.start()` and then `_prepare_emissions_data()`, which calls `_update_emissions()` → `_ensure_geo_metadata()` (L962, L378-388) — a lazy first-use geolocation lookup — for the first time in the run. This is **inferred from the source order** (the lookup happens after the counters were baselined and the lookup time is not measured by this script); it fits the observed ≈3.08 s excess and the fact that no later task shows it.

**Conclusion (plain language).**

1. Established from source: CodeCarbon's task mode does not poll during a task. Each task's energy is the difference of the hardware energy counters (NVML for the GPU, RAPL for the CPU) between `start_task` and `stop_task`, plus modelled RAM power (constant 20 W) × elapsed time. A 0.2–0.4 s rollout task is therefore *not* "zero or one polling sample"; it is a counter difference over its own window.
2. Established from source: the energy between `stop_task(i)` and `start_task(i+1)` is attributed to no task and is not in the run total either (counters are re-baselined by `hardware.start()`); the whole-run cumulative energy equals the sum of the task energies to within 0.0125 % (max over 280 runs, median 8.5e-5 %, `02_sum_tasks_vs_whole_run.csv`) — this is an accounting identity, not an independent validation.
3. Established from data, mechanism inferred: the GPU energy of sub-second tasks is **quantised**. For MBPO `synthetic_rollout_generation` on HalfCheetah (median duration 6 ms) 3283 of 3500 tasks have exactly 0 J GPU energy, and the 217 non-zero ones are all between 5.86 and 6.92 J (median 6.36 J), independent of the task duration; on Ant (68 ms median for non-zero tasks) the non-zero values range 5.6–24 J. 217/3500 = 6.2 % ≈ 6.1 ms task / ≈ 100 ms: consistent with an NVML energy counter that advances in lumps of ≈6 J roughly every 100 ms (≈ 60-65 W × 0.1 s). This is **inferred**, not documented in CodeCarbon's source or checked against NVIDIA documentation here; `nvml_poll.py` (2C) would verify it.
4. Consequence (inferred): for a task of 0.2–0.4 s (SAC/TD3/MBPO rollouts, ≈ 2-4 counter updates) the recorded GPU energy is a staircase value, i.e. it can be off by about ± one lump (≈ ±6 J out of ≈ 14-25 J, so ≈ ±25-40 % per task for 0.1-0.3 s rollouts), in either direction; for tasks of a few ms (synthetic rollout generation) the value is either 0 or one whole lump, i.e. off by up to ±100 % or more per task. Over many tasks the sum is plausibly unbiased (the probability of catching a lump scales with task duration: pooled mean GPU power of the HalfCheetah synthetic tasks is 62.9 W, Ant 86.1 W, vs 68-82 W mean for rollouts), but this is only reasoned from the data, not tested. CPU (RAPL µJ counter) and RAM (model) do not show zero energy for any task (zero fraction 0.0000 everywhere in table A1).
5. Per-task power implausibilities in the data: for tasks < 1 s, GPU energy ÷ duration has p1 = p5 = 0 W, p99 = 618 W, max = 1214.7 W (1.2 kW is far above any plausible board power of this GPU; the card power limit itself was not checked here, see Task 3), all from `synthetic_rollout_generation`; for tasks ≥ 2 s the GPU power is 63.6-180.7 W (p1-max). So extreme per-task power values arise only in the millisecond-scale tasks.

### 2B step 4: empirical check against the data

- Sub-second tasks with non-zero GPU energy: 25387 of 31241 sub-second tasks; 13159 distinct energy values (not on a coarse grid at the mJ level: the counter itself is mJ-resolution, the lumps are not).
- Sub-second total power (energy ÷ duration, W), quantiles p1 / p5 / p50 / p95 / p99 = 44.8, 48.2, 117.3, 187.6, 664.8; GPU power p1 / p5 / p50 / p95 / p99 / max = 0.0, 0.0, 65.5, 134.3, 617.6, 1214.7; for tasks ≥ 2 s: 63.6, 68.1, 97.5, 159.0, 171.7, 180.7.
- Rollout tasks (0.1-0.36 s): none has zero GPU energy (0 of 7000 SAC, 0 of 8000 TD3, 0 of 8000 MBPO), median GPU power 65-72 W, coefficient of variation of per-task GPU power 16 % (SAC) to 36 % (TD3 HalfCheetah, the shortest rollouts at 0.108 s) — the scatter grows as the task gets shorter, as expected for lumpy counters. Not a proof, but consistent.
- Sum of task energies vs. the whole measured run (`emissions.csv` last row = cumulative energy since `tracker.start()`): relative difference median 8.5e-5 %, max 0.0125 % over 280 runs (per component all < 0.03 %). As explained in Finding 2 this is an identity of CodeCarbon's accounting and says nothing about the quality of the per-task values. What is *not* in either number is the (unmeasured) energy between tasks.

### 2C. Optional follow-up experiments (NOT RUN — they need the research PC)

1. `contexts/step1/scripts/sync_timing_run.py` — one baseline run per algorithm with `torch.cuda.synchronize()` before every `perf_counter()` call in `algorithms/{sac,td3,mbpo,tdmpc2}.py` (by replacing the module-level `time` name with a shim; no repo file is changed), then prints ratio = Σ sub-phase wall times / Σ measured `gradient_updates` durations, also for a canonical baseline run (`--compare-dir`). Commands are in the script header. Run on a quiet machine with clocks locked as in the protocol (sudo).
2. `contexts/step1/scripts/nvml_poll.py` — standalone ~1 kHz poller of `nvmlDeviceGetTotalEnergyConsumption`, logs every counter change and prints a histogram of the spacing between updates; run it in parallel with a short SAC baseline run (commands in the header). Expected if the inference above is right: spacing clustered near 100 ms with ≈6 J steps.

## Caveats / things that look odd

- 2B was done on the Windows dev venv's CodeCarbon 3.3.0 install, not the Linux `rl-exp` venv; same version number as recorded in `metadata.json`, but package contents on the Linux box were not inspected.
- The NVML-lump explanation and the "≈100 ms" figure come from the data only; they are not documented in the CodeCarbon source and were not checked against NVIDIA documentation or a live counter.
- The 3.08 s head excess is attributed to the first-use geolocation lookup by reading the source order; not measured.
- Thesis medians 0.3409 / 0.2175 s correspond to all-configuration pooling, not baseline-only pooling (0.3436 / 0.2165 s) — check the wording "baseline" in the thesis.
- `warmup` is a task too (35-40 per algo-env group, one per run); TD3 Ant warmup (10000 steps) is 1.3 s, all other warmups are < 1 s.
