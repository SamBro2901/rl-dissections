# Section 5.2 — Energy versus compute: are FLOP-based energy estimates trustworthy? — context

Data context for thesis Section 5.2. Facts, numbers and provenance only — no interpretation, no LaTeX. 280 canonical runs (56 configurations × 5 seeds [331, 958, 14577, 43611, 85062]); dev runs ignored. All energies are **gross** (idle floor included, nothing subtracted), joules, seed-mean over the five seeds. HalfCheetah-v5 (HC): 27 configurations; Ant-v5: 29 (Ant has the two extra MBPO rollout-length configurations `rollout1`, `rollout15`). The two environments are always reported separately. Returns are not used anywhere in this section.

Generated 2026-10-08 from git HEAD `2b4882f` (branch master); Windows dev checkout with synced `results/`. Python 3.13 / pandas 3.0.5 / scipy 1.18.1.

## 0. Methods note

**Scripts** (both new, under `flop_analysis/`, read-only on `results/` and `flop_analysis/output/`): `flop_analysis/section_5_2_analysis.py` (computes everything, writes `contexts/section_5.2_data/*.csv`) and `flop_analysis/section_5_2_render.py` (renders this file from those CSVs). Re-run from the repo root: `.venv/Scripts/python -I -W ignore flop_analysis/section_5_2_analysis.py` then `.venv/Scripts/python flop_analysis/section_5_2_render.py`. Input is `flop_analysis/output/per_run_energy_per_flop.csv` (canonical seeds, HC and Ant), the pipeline's J/FLOP definitions unchanged.

**Definitions**
- *Configuration* = one (algorithm, environment, hyper-parameter setting); label tags: `base`; `utdK` = updates per env step K; `wN` = hidden width N (baseline 1024); `bN` = batch size N (baseline 256 for SAC/MBPO, 100 for TD3); `rolloutN` = MBPO max model-rollout length N (Ant only; Ant baseline is 25, HC baseline is the fixed length 1); `numqN`, `horizonN` = TD-MPC2 Q-ensemble size (baseline 5) and MPPI horizon (baseline 3).
- *Total* = `TOTAL_MEASURED_TRAINING` energy and matmul FLOPs (rollout + gradient_updates + MBPO dynamics/synthetic-rollout + TD-MPC2 world_model_pretrain; warmup and idle excluded; `target_update` elementwise ops excluded from FLOPs). Per-run total energy equals the sum of its measured segments to a relative error of 3.6e-16.
- *gradient_updates (GU)* has no row of its own in the pipeline CSV. GU energy = `buffer_sample` + `critic_update` + `actor_update` + `target_update` energy (the directly measured `gradient_updates` task, which the pipeline splits into those four allocated parts); GU FLOPs = `critic_update` + `actor_update` matmul FLOPs (`buffer_sample` has none; `target_update` is elementwise and excluded). The allocated sub-segments `critic_update`/`actor_update` are not analysed separately here, per the task.
- *Seed-mean* energy and FLOPs per configuration; *J/FLOP* = mean energy / mean FLOPs (ratio of means), *sd* = sample sd (ddof = 1) of the five per-seed ratios. Tables show pJ/FLOP (1 pJ = 1e-12 J). MBPO FLOPs differ slightly between seeds (dynamics-model early stopping, variable synthetic-rollout termination); the FLOPs shown for MBPO are seed-means (min/max in the CSV). For all other algorithms FLOPs are identical across seeds.
- *Spearman*: `scipy.stats.spearmanr` (mid-ranks for ties; two-sided asymptotic p), FLOPs vs energy over configuration means. For n ≤ 9 the CSVs also give an exact two-sided permutation p (all n! orderings). *Log-log fit*: `scipy.stats.linregress(log10 FLOPs, log10 energy[J])` on configuration means: slope, intercept (log10 J at 1 FLOP), R².
- *Linear fit* E = a + b·F: ordinary least squares on **configuration-mean points** (one point per sweep value, baseline included). a [J], b [pJ/FLOP], R², residual SE = sqrt(SSR/(n−2)) [J]. The same fits on per-run points (5 per configuration, n = 15 or 20) are in `a1_linear_fits.csv` (`fit_basis = per_run`); for SAC, TD3 and TD-MPC2 they are numerically identical to the configuration-mean fits (FLOPs are equal across seeds; maximum relative difference of the slope 2.3e-15), for MBPO they differ (see Section 6).
- *Sweeps and fit rows*: UTD {1,2,4}, width {256,512,1024} and batch size ({256,512,1024}; TD3 {100,256,512,1024}) each holding the other factors at the baseline; for MBPO the rollout schedule is held at the environment baseline. TD-MPC2: `num_q` {3,5,7} at horizon 3, and `horizon` {1,3,5} at num_q 5. The baseline configuration is a point of every sweep of its (algorithm, environment). The MBPO rollout-length sweep (Ant) was not part of the request and is not fitted.
- *Delta pairs*: ΔE/ΔF between **adjacent** sweep values on configuration means, as in `flop_dashboard.py` (`compute_delta_pairs`, previous-value mode); pairs with |ΔF| < 1 % of the reference FLOPs are dropped (shown as n/a). With one pair per interval and one sweep per (algorithm, environment) there is no pooling.
- **Independence caveat.** The configurations are not independent observations: they come from one-factor-at-a-time sweeps around shared baselines (each baseline appears in several sweeps and in the pooled sets, and many configurations share the same algorithm, environment and hardware). All p-values in this file are descriptive, not inferential. Configurations with exactly equal FLOPs (SAC `utd2`/`b512` and `utd4`/`b1024`, in both environments) are ties in the rank correlation.
- **Small-n caveat.** A 3-point linear fit has one residual degree of freedom (n − 2 = 1); a 4-point fit has two. R² is then high almost by construction and the residual SE is computed with n − 2 = 1 in the denominator (not corrected).

## 1. Part (d) — per-configuration table (all 56 configurations)

`d1_config_table.csv` holds every column (energy mean/sd, FLOPs mean/min/max, J/FLOP and its sd, for total and each segment). Below, per environment: E [J] gross seed-mean, F = matmul FLOPs seed-mean, pJ/F = J/FLOP ×1e12 (ratio of means; ±sd of per-seed ratios for the total). `buffer_sample` and `target_update` have no matmul FLOPs and are not listed as segments (their energy is inside GU).

### HalfCheetah-v5

| algo | config | total E [J] | total F | total pJ/F | rollout E [J] | rollout F | rollout pJ/F | GU E [J] | GU F | GU pJ/F |
|---|---|---|---|---|---|---|---|---|---|---|
| sac | b1024 | 82366 | 3.5077e+15 | 23.4815 ± 0.168 | 2748.17 | 2.15654e+11 | 12743.4 | 79617.8 | 3.50749e+15 | 22.6994 |
| sac | b512 | 73024 | 1.75396e+15 | 41.6338 ± 0.295 | 2646.99 | 2.15654e+11 | 12274.2 | 70377 | 1.75374e+15 | 40.1296 |
| sac | base | 60911 | 8.77087e+14 | 69.4469 ± 0.29 | 2495.7 | 2.15654e+11 | 11572.7 | 58415.3 | 8.76872e+14 | 66.6179 |
| sac | utd2 | 118024 | 1.75396e+15 | 67.2903 ± 0.356 | 2513.39 | 2.15654e+11 | 11654.7 | 115511 | 1.75374e+15 | 65.8654 |
| sac | utd4 | 234959 | 3.5077e+15 | 66.9838 ± 0.316 | 2552.87 | 2.15654e+11 | 11837.8 | 232406 | 3.50749e+15 | 66.26 |
| sac | w256 | 47407.9 | 5.81712e+13 | 814.972 ± 7.88 | 2278.44 | 1.4592e+10 | 156143 | 45129.5 | 5.81566e+13 | 775.999 |
| sac | w512 | 49554 | 2.23743e+14 | 221.477 ± 2.28 | 2326.75 | 5.53984e+10 | 42000.2 | 47227.2 | 2.23687e+14 | 211.13 |
| td3 | b1024 | 48068.4 | 2.51575e+15 | 19.107 ± 0.321 | 1752.39 | 2.14426e+11 | 8172.49 | 46316 | 2.51553e+15 | 18.412 |
| td3 | b256 | 33453.7 | 6.29098e+14 | 53.1773 ± 0.692 | 1516.82 | 2.14426e+11 | 7073.88 | 31936.9 | 6.28883e+14 | 50.7835 |
| td3 | b512 | 41683.2 | 1.25798e+15 | 33.135 ± 0.619 | 1681.61 | 2.14426e+11 | 7842.38 | 40001.6 | 1.25777e+15 | 31.8037 |
| td3 | base | 28111.5 | 2.45872e+14 | 114.334 ± 1.26 | 1410.91 | 2.14426e+11 | 6579.95 | 26700.5 | 2.45658e+14 | 108.69 |
| td3 | utd2 | 55361 | 4.9153e+14 | 112.63 ± 0.809 | 1431.6 | 2.14426e+11 | 6676.42 | 53929.4 | 4.91315e+14 | 109.765 |
| td3 | utd4 | 107345 | 9.82845e+14 | 109.219 ± 1.06 | 1380.45 | 2.14426e+11 | 6437.91 | 105964 | 9.8263e+14 | 107.838 |
| td3 | w256 | 23037 | 1.62088e+13 | 1421.26 ± 38.2 | 1240.18 | 1.42848e+10 | 86818.3 | 21796.8 | 1.61946e+13 | 1345.93 |
| td3 | w512 | 25369.7 | 6.25905e+13 | 405.328 ± 6.38 | 1301.93 | 5.4784e+10 | 23764.8 | 24067.7 | 6.25357e+13 | 384.864 |
| mbpo | b1024 | 218667 | 3.75332e+15 | 58.2597 ± 1.2 | 2362.8 | 2.15654e+11 | 10956.4 | 78057.3 | 3.50749e+15 | 22.2545 |
| mbpo | b512 | 209101 | 1.99994e+15 | 104.554 ± 2.08 | 2383.39 | 2.15654e+11 | 11051.9 | 69369.1 | 1.75374e+15 | 39.5548 |
| mbpo | base | 207395 | 1.13285e+15 | 183.073 ± 5.44 | 2428.26 | 2.15654e+11 | 11260 | 58698.3 | 8.76872e+14 | 66.9406 |
| mbpo | utd2 | 268851 | 2.02009e+15 | 133.089 ± 2.5 | 2409.68 | 2.15654e+11 | 11173.8 | 116328 | 1.75374e+15 | 66.3312 |
| mbpo | utd4 | 373606 | 3.7579e+15 | 99.4189 ± 1.83 | 2381.09 | 2.15654e+11 | 11041.2 | 231830 | 3.50749e+15 | 66.0959 |
| mbpo | w256 | 187913 | 3.06747e+14 | 612.601 ± 9.02 | 2312.63 | 1.4592e+10 | 158486 | 44550.1 | 5.81566e+13 | 766.036 |
| mbpo | w512 | 192769 | 4.7726e+14 | 403.908 ± 8.71 | 2335.78 | 5.53984e+10 | 42163.3 | 46788.7 | 2.23687e+14 | 209.17 |
| tdmpc2 | base | 447087 | 8.83269e+15 | 50.6173 ± 0.361 | 213811 | 4.64261e+15 | 46.054 | 222413 | 3.99055e+15 | 55.7348 |
| tdmpc2 | horizon1 | 280686 | 4.61451e+15 | 60.827 ± 0.386 | 135177 | 2.94823e+15 | 45.8503 | 138819 | 1.58693e+15 | 87.4766 |
| tdmpc2 | horizon5 | 571281 | 1.30509e+16 | 43.7734 ± 0.309 | 288267 | 6.33699e+15 | 45.4895 | 270219 | 6.39418e+15 | 42.2602 |
| tdmpc2 | numq3 | 358787 | 6.87589e+15 | 52.1805 ± 0.317 | 181988 | 3.93105e+15 | 46.295 | 168521 | 2.80461e+15 | 60.087 |
| tdmpc2 | numq7 | 526562 | 1.07895e+16 | 48.8032 ± 0.183 | 242194 | 5.35417e+15 | 45.2347 | 271633 | 5.17649e+15 | 52.4744 |

| algo | config | dynamics_model_update E [J] | F | pJ/F | synthetic_rollout_generation E [J] | F | pJ/F | world_model_pretrain E [J] | F | pJ/F |
|---|---|---|---|---|---|---|---|---|---|---|
| mbpo | b1024 | 138166 | 2.41654e+14 | 571.753 | 80.7041 | 3.96173e+12 | 20.3709 |  |  |  |
| mbpo | b512 | 137282 | 2.42018e+14 | 567.242 | 66.5633 | 3.96173e+12 | 16.8016 |  |  |  |
| mbpo | base | 146193 | 2.51804e+14 | 580.582 | 75.399 | 3.96173e+12 | 19.0319 |  |  |  |
| mbpo | utd2 | 150041 | 2.62168e+14 | 572.307 | 72.5129 | 3.96173e+12 | 18.3033 |  |  |  |
| mbpo | utd4 | 139320 | 2.46235e+14 | 565.801 | 74.7968 | 3.96173e+12 | 18.8799 |  |  |  |
| mbpo | w256 | 140997 | 2.46604e+14 | 571.756 | 53.2193 | 1.97121e+12 | 26.9983 |  |  |  |
| mbpo | w512 | 143579 | 2.51142e+14 | 571.704 | 65.7243 | 2.37519e+12 | 27.6712 |  |  |  |
| tdmpc2 | base |  |  |  |  |  |  | 10863.4 | 1.99528e+14 | 54.4456 |
| tdmpc2 | horizon1 |  |  |  |  |  |  | 6689.99 | 7.93464e+13 | 84.3137 |
| tdmpc2 | horizon5 |  |  |  |  |  |  | 12794.6 | 3.19709e+14 | 40.0197 |
| tdmpc2 | numq3 |  |  |  |  |  |  | 8278.54 | 1.40231e+14 | 59.0352 |
| tdmpc2 | numq7 |  |  |  |  |  |  | 12734.4 | 2.58825e+14 | 49.201 |

### Ant-v5

| algo | config | total E [J] | total F | total pJ/F | rollout E [J] | rollout F | rollout pJ/F | GU E [J] | GU F | GU pJ/F |
|---|---|---|---|---|---|---|---|---|---|---|
| sac | b1024 | 89057.4 | 3.75519e+15 | 23.7158 ± 0.109 | 4540.98 | 2.34496e+11 | 19364.8 | 84516.4 | 3.75495e+15 | 22.508 |
| sac | b512 | 76498.9 | 1.87771e+15 | 40.7405 ± 0.213 | 4367.22 | 2.34496e+11 | 18623.9 | 72131.7 | 1.87748e+15 | 38.4195 |
| sac | base | 64382.6 | 9.38972e+14 | 68.5671 ± 0.627 | 4332.11 | 2.34496e+11 | 18474.1 | 60050.5 | 9.38738e+14 | 63.9694 |
| sac | utd2 | 123182 | 1.87771e+15 | 65.6025 ± 0.387 | 4308.57 | 2.34496e+11 | 18373.7 | 118874 | 1.87748e+15 | 63.3158 |
| sac | utd4 | 241327 | 3.75519e+15 | 64.265 ± 0.284 | 4293.53 | 2.34496e+11 | 18309.6 | 237033 | 3.75495e+15 | 63.1256 |
| sac | w256 | 49985.2 | 7.36424e+13 | 678.756 ± 9.85 | 4100.74 | 1.93024e+10 | 212447 | 45884.5 | 7.36231e+13 | 623.235 |
| sac | w512 | 52199.8 | 2.54685e+14 | 204.958 ± 1.88 | 4154.2 | 6.48192e+10 | 64089 | 48045.6 | 2.5462e+14 | 188.695 |
| td3 | b1024 | 53235 | 2.68585e+15 | 19.8206 ± 0.264 | 3390.84 | 2.32858e+11 | 14561.8 | 49844.2 | 2.68561e+15 | 18.5597 |
| td3 | b256 | 35520.5 | 6.71636e+14 | 52.8866 ± 0.533 | 3086.58 | 2.32858e+11 | 13255.2 | 32434 | 6.71403e+14 | 48.3077 |
| td3 | b512 | 44361.2 | 1.34304e+15 | 33.0304 ± 0.441 | 3301.77 | 2.32858e+11 | 14179.3 | 41059.4 | 1.34281e+15 | 30.5773 |
| td3 | base | 30519.8 | 2.625e+14 | 116.266 ± 1.65 | 3071.46 | 2.32858e+11 | 13190.3 | 27448.4 | 2.62267e+14 | 104.658 |
| td3 | utd2 | 57542.6 | 5.24767e+14 | 109.654 ± 0.688 | 3031.22 | 2.32858e+11 | 13017.5 | 54511.4 | 5.24534e+14 | 103.924 |
| td3 | utd4 | 111900 | 1.0493e+15 | 106.642 ± 1.52 | 3014.92 | 2.32858e+11 | 12947.5 | 108885 | 1.04907e+15 | 103.792 |
| td3 | w256 | 25100.4 | 2.03658e+13 | 1232.48 ± 39.8 | 2837.25 | 1.88928e+10 | 150176 | 22263.2 | 2.03469e+13 | 1094.18 |
| td3 | w512 | 27473.2 | 7.09043e+13 | 387.469 ± 3.73 | 2897.84 | 6.4e+10 | 45278.7 | 24575.4 | 7.08403e+13 | 346.912 |
| mbpo | b1024 | 267695 | 4.23983e+15 | 63.1381 ± 3.06 | 4178.11 | 2.34496e+11 | 17817.4 | 84617.4 | 3.75495e+15 | 22.5349 |
| mbpo | b512 | 247393 | 2.33882e+15 | 105.777 ± 6.71 | 4113.46 | 2.34496e+11 | 17541.7 | 71737 | 1.87748e+15 | 38.2093 |
| mbpo | base | 253138 | 1.41603e+15 | 178.766 ± 8.47 | 4292.41 | 2.34496e+11 | 18304.8 | 59746.5 | 9.38738e+14 | 63.6456 |
| mbpo | rollout1 | 240749 | 1.37212e+15 | 175.458 ± 1.64 | 4251.9 | 2.34496e+11 | 18132.1 | 59929.1 | 9.38738e+14 | 63.8401 |
| mbpo | rollout15 | 245862 | 1.40601e+15 | 174.865 ± 5.75 | 4213.11 | 2.34496e+11 | 17966.7 | 59817.6 | 9.38738e+14 | 63.7213 |
| mbpo | utd2 | 316797 | 2.37967e+15 | 133.126 ± 5.19 | 4214.55 | 2.34496e+11 | 17972.8 | 118703 | 1.87748e+15 | 63.2247 |
| mbpo | utd4 | 424454 | 4.23812e+15 | 100.151 ± 3 | 4178.96 | 2.34496e+11 | 17821 | 235859 | 3.75495e+15 | 62.8128 |
| mbpo | w256 | 234491 | 5.43602e+14 | 431.364 ± 2.82 | 4073.76 | 1.93024e+10 | 211049 | 44807.7 | 7.36231e+13 | 608.61 |
| mbpo | w512 | 231638 | 7.17054e+14 | 323.041 ± 8.56 | 4108.62 | 6.48192e+10 | 63385.8 | 47310.4 | 2.5462e+14 | 185.807 |
| tdmpc2 | base | 493191 | 1.00763e+16 | 48.9454 ± 0.177 | 254576 | 5.61814e+15 | 45.3132 | 227753 | 4.24591e+15 | 53.6406 |
| tdmpc2 | horizon1 | 303231 | 5.0342e+15 | 60.2342 ± 0.513 | 149441 | 3.27635e+15 | 45.6122 | 146603 | 1.67414e+15 | 87.5691 |
| tdmpc2 | horizon5 | 650272 | 1.51185e+16 | 43.0117 ± 0.305 | 354226 | 7.95994e+15 | 44.5011 | 282757 | 6.81767e+15 | 41.4742 |
| tdmpc2 | numq3 | 411357 | 8.11609e+15 | 50.6841 ± 0.252 | 223015 | 4.90532e+15 | 45.4638 | 179947 | 3.05787e+15 | 58.8473 |
| tdmpc2 | numq7 | 576921 | 1.20366e+16 | 47.9306 ± 0.235 | 280891 | 6.33097e+15 | 44.3678 | 282713 | 5.43394e+15 | 52.0271 |

| algo | config | dynamics_model_update E [J] | F | pJ/F | synthetic_rollout_generation E [J] | F | pJ/F | world_model_pretrain E [J] | F | pJ/F |
|---|---|---|---|---|---|---|---|---|---|---|
| mbpo | b1024 | 178176 | 4.39178e+14 | 405.703 | 723.343 | 4.54681e+13 | 15.9088 |  |  |  |
| mbpo | b512 | 170819 | 4.15041e+14 | 411.571 | 723.897 | 4.60675e+13 | 15.7138 |  |  |  |
| mbpo | base | 188363 | 4.30949e+14 | 437.09 | 735.651 | 4.61061e+13 | 15.9556 |  |  |  |
| mbpo | rollout1 | 176446 | 4.28258e+14 | 412.01 | 121.939 | 4.88561e+12 | 24.9589 |  |  |  |
| mbpo | rollout15 | 181382 | 4.36857e+14 | 415.198 | 449.641 | 3.01841e+13 | 14.8966 |  |  |  |
| mbpo | utd2 | 193129 | 4.56382e+14 | 423.173 | 750.253 | 4.5575e+13 | 16.4619 |  |  |  |
| mbpo | utd4 | 183704 | 4.38714e+14 | 418.732 | 712.228 | 4.42257e+13 | 16.1044 |  |  |  |
| mbpo | w256 | 185042 | 4.44206e+14 | 416.567 | 567.556 | 2.57539e+13 | 22.0377 |  |  |  |
| mbpo | w512 | 179577 | 4.31938e+14 | 415.747 | 642.109 | 3.04312e+13 | 21.1004 |  |  |  |
| tdmpc2 | base |  |  |  |  |  |  | 10862.2 | 2.12295e+14 | 51.1657 |
| tdmpc2 | horizon1 |  |  |  |  |  |  | 7186.28 | 8.37072e+13 | 85.8503 |
| tdmpc2 | horizon5 |  |  |  |  |  |  | 13288.9 | 3.40884e+14 | 38.9838 |
| tdmpc2 | numq3 |  |  |  |  |  |  | 8394.74 | 1.52894e+14 | 54.9058 |
| tdmpc2 | numq7 |  |  |  |  |  |  | 13317.9 | 2.71697e+14 | 49.0174 |

## 2. Part (d) — training total, FLOPs vs energy, pooled over algorithms (per environment)

| env | n configs | Spearman ρ | p (asymptotic) | log-log slope | slope SE | intercept (log10 J) | R² | FLOPs range |
|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | 27 | 0.8357 | 5.78e-08 | 0.4715 | 0.0752 | -2.032 | 0.611 | 1.62e+13 – 1.31e+16 |
| Ant-v5 | 29 | 0.8463 | 7.33e-09 | 0.5071 | 0.0755 | -2.537 | 0.6253 | 2.04e+13 – 1.51e+16 |

**Reproduction of the thesis values** (ρ 0.836 / 0.846; slope 0.47 / 0.51; R² 0.61 / 0.63): reproduced (ρ 0.8357 / 0.8463, slope 0.4715 / 0.5071, R² 0.6110 / 0.6253), identical to `flop_analysis/output/correlation/correlation_summary.csv` (same 27 / 29 configuration points).

## 3. Part (d) — per measured segment, pooled over algorithms (per environment)

| env | segment | n | Spearman ρ | p (asymptotic) | p (exact perm., n≤9) | log-log slope | intercept | R² | FLOPs range [orders of mag.] |
|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | rollout | 27 | 0.8836 | 1.02e-09 |  | 0.4231 | -1.374 | 0.9471 | 5.65 |
| HalfCheetah-v5 | gradient_updates | 27 | 0.8421 | 3.66e-08 |  | 0.3771 | -0.7535 | 0.6192 | 2.6 |
| HalfCheetah-v5 | dynamics_model_update | 7 | 0.9643 | 0.000454 | 0.00278 | 1.118 | -10.93 | 0.9458 | 0.0354 |
| HalfCheetah-v5 | synthetic_rollout_generation | 7 | 0.8018 | 0.0301 | 0.0476 | 0.3918 | -3.066 | 0.738 | 0.303 |
| HalfCheetah-v5 | world_model_pretrain | 5 | 1 | 1.4e-24 | 0.0167 | 0.5094 | -3.263 | 0.9681 | 0.605 |
| Ant-v5 | rollout | 29 | 0.8834 | 2.21e-10 |  | 0.3815 | -0.6684 | 0.9574 | 5.62 |
| Ant-v5 | gradient_updates | 29 | 0.8689 | 9.78e-10 |  | 0.3984 | -1.082 | 0.6369 | 2.53 |
| Ant-v5 | dynamics_model_update | 9 | 0.65 | 0.0581 | 0.0666 | 1.148 | -11.55 | 0.6792 | 0.0412 |
| Ant-v5 | synthetic_rollout_generation | 9 | 0.9333 | 0.000236 | 0.00075 | 0.7935 | -7.964 | 0.964 | 0.975 |
| Ant-v5 | world_model_pretrain | 5 | 0.9 | 0.0374 | 0.0833 | 0.4881 | -2.958 | 0.9377 | 0.61 |

Pooled `rollout` and `gradient_updates` use all configurations of the environment (n = 27 / 29). `dynamics_model_update` and `synthetic_rollout_generation` exist for MBPO only (n = 7 on HC, 9 on Ant); `world_model_pretrain` for TD-MPC2 only (n = 5). Those four rows are **small-n, single-algorithm** results. Asymptotic p at ρ = 1 with n = 5 is a numerical artefact of the t approximation (1.4e-24); the exact permutation p is 2/120 = 0.0167. Per-algorithm segment statistics are in `d5_per_algo_segment_correlation.csv`.

## 4. Part (d) — training total per algorithm and environment (small n)

| env | algo | n | Spearman ρ | p (asymptotic) | p (exact perm.) | log-log slope | intercept | R² | FLOPs range [orders] |
|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | sac | 7 | 0.9092 | 0.00454 | 0.00794 | 0.2717 | 0.8491 | 0.5464 | 1.78 |
| HalfCheetah-v5 | td3 | 8 | 0.7381 | 0.0366 | 0.0458 | 0.2065 | 1.598 | 0.4682 | 2.19 |
| HalfCheetah-v5 | mbpo | 7 | 0.9643 | 0.000454 | 0.00278 | 0.17 | 2.789 | 0.4664 | 1.09 |
| HalfCheetah-v5 | tdmpc2 | 5 | 1 | 1.4e-24 | 0.0167 | 0.7127 | -5.719 | 0.9922 | 0.452 |
| Ant-v5 | sac | 7 | 0.9092 | 0.00454 | 0.00794 | 0.2837 | 0.6799 | 0.5686 | 1.71 |
| Ant-v5 | td3 | 8 | 0.7381 | 0.0366 | 0.0458 | 0.2103 | 1.564 | 0.4812 | 2.12 |
| Ant-v5 | mbpo | 9 | 0.9167 | 0.000507 | 0.00131 | 0.1924 | 2.5 | 0.4873 | 0.892 |
| Ant-v5 | tdmpc2 | 5 | 1 | 1.4e-24 | 0.0167 | 0.7132 | -5.72 | 0.9948 | 0.478 |

**Small-n warning**: n = 5–9 configurations per row, all from one-factor-at-a-time sweeps (non-independent), so these are descriptive. SAC's HC and Ant rows have identical ρ because the FLOPs ranking of SAC configurations is the same in both environments and the energy ranking coincides. For TD-MPC2 the ranking by FLOPs and by energy is identical in both environments (ρ = 1).

## 5. Part (a) — fixed versus marginal cost: E = a + b·F per sweep

Points = seed-mean (F, E) of the sweep's configurations (n = 3; TD3 batch sweep n = 4). **n = 3 fits have one residual degree of freedom; n = 4, two.** `rollout` fits for the UTD and batch sweeps are undefined: the rollout FLOPs (batch-size-1 actor forward × env steps) are identical across the sweep values, so the regressor is constant (listed n/a). Rollout is not affected by those factors in FLOP terms. Per-run versions: `a1_linear_fits.csv`.

### HalfCheetah-v5

| algo | sweep | values | segment | n pts | a [J] | b [pJ/FLOP] | R² | residual SE [J] | note |
|---|---|---|---|---|---|---|---|---|---|
| sac | UTD | 1/2/4 | total | 3 | 2429.4 | 66.236 | 0.99997 | 723.722 |  |
| sac | UTD | 1/2/4 | gradient_updates | 3 | -32.2806 | 66.214 | 0.99997 | 722.622 |  |
| sac | UTD | 1/2/4 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| sac | width | 256/512/1024 | total | 3 | 46156.2 | 16.742 | 0.99807 | 450.957 |  |
| sac | width | 256/512/1024 | gradient_updates | 3 | 43891 | 16.483 | 0.99798 | 454.337 |  |
| sac | width | 256/512/1024 | rollout | 3 | 2264.78 | 1073.2 | 0.99959 | 3.25408 |  |
| sac | batch | 256/512/1024 | total | 3 | 56238.4 | 7.7517 | 0.93163 | 3977.89 |  |
| sac | batch | 256/512/1024 | gradient_updates | 3 | 53794.9 | 7.6612 | 0.93187 | 3924.06 |  |
| sac | batch | 256/512/1024 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| td3 | UTD | 1/2/4 | total | 3 | 2096.51 | 107.27 | 0.99986 | 672.214 |  |
| td3 | UTD | 1/2/4 | gradient_updates | 3 | 683.033 | 107.32 | 0.99987 | 647.488 |  |
| td3 | UTD | 1/2/4 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| td3 | width | 256/512/1024 | total | 3 | 23333.8 | 20.072 | 0.92098 | 1009.77 |  |
| td3 | width | 256/512/1024 | gradient_updates | 3 | 22092 | 19.388 | 0.91884 | 988.718 |  |
| td3 | width | 256/512/1024 | rollout | 3 | 1241.62 | 804.8 | 0.97048 | 21.0042 |  |
| td3 | batch | 100/256/512/1024 | total | 4 | 27848 | 8.5884 | 0.93756 | 2698.65 |  |
| td3 | batch | 100/256/512/1024 | gradient_updates | 4 | 26428.6 | 8.4428 | 0.93855 | 2630.32 |  |
| td3 | batch | 100/256/512/1024 | rollout | 4 | n/a | n/a | n/a | n/a |  |
| mbpo | UTD | 1/2/4 | total | 3 | 138391 | 62.898 | 0.99873 | 4236.31 |  |
| mbpo | UTD | 1/2/4 | gradient_updates | 3 | 946.974 | 65.821 | 1 | 65.0424 |  |
| mbpo | UTD | 1/2/4 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| mbpo | UTD | 1/2/4 | dynamics_model_update | 3 | -15636 | 634.64 | 0.89282 | 2514.48 |  |
| mbpo | width | 256/512/1024 | total | 3 | 181185 | 23.227 | 0.99797 | 645.613 |  |
| mbpo | width | 256/512/1024 | gradient_updates | 3 | 43233.6 | 17.551 | 0.99801 | 480.171 |  |
| mbpo | width | 256/512/1024 | rollout | 3 | 2304.08 | 575.65 | 0.99999 | 0.246368 |  |
| mbpo | width | 256/512/1024 | dynamics_model_update | 3 | -66703.6 | 841.68 | 0.84111 | 1464.36 |  |
| mbpo | batch | 256/512/1024 | total | 3 | 201462 | 4.4694 | 0.9644 | 1621.23 |  |
| mbpo | batch | 256/512/1024 | gradient_updates | 3 | 54354.2 | 7.0155 | 0.93918 | 3381.75 |  |
| mbpo | batch | 256/512/1024 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| mbpo | batch | 256/512/1024 | dynamics_model_update | 3 | -66924 | 846.27 | 0.98526 | 842.959 |  |
| tdmpc2 | num_q | 3/5/7 | total | 3 | 65490.3 | 42.87 | 0.99908 | 3602.46 | architecture change |
| tdmpc2 | num_q | 3/5/7 | gradient_updates | 3 | 47374.8 | 43.473 | 0.99932 | 1906.9 | architecture change |
| tdmpc2 | num_q | 3/5/7 | rollout | 3 | 16254.5 | 42.306 | 0.99891 | 1404.13 | architecture change |
| tdmpc2 | horizon | 1/3/5 | total | 3 | 128772 | 34.445 | 0.99302 | 17230.6 | architecture change |
| tdmpc2 | horizon | 1/3/5 | gradient_updates | 3 | 101407 | 27.334 | 0.97587 | 14609.8 | architecture change |
| tdmpc2 | horizon | 1/3/5 | rollout | 3 | 2685.2 | 45.176 | 0.99975 | 1705.48 | architecture change |

### Ant-v5

| algo | sweep | values | segment | n pts | a [J] | b [pJ/FLOP] | R² | residual SE [J] | note |
|---|---|---|---|---|---|---|---|---|---|
| sac | UTD | 1/2/4 | total | 3 | 5295.67 | 62.844 | 1 | 145.598 |  |
| sac | UTD | 1/2/4 | gradient_updates | 3 | 970.775 | 62.857 | 1 | 137.031 |  |
| sac | UTD | 1/2/4 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| sac | width | 256/512/1024 | total | 3 | 48358.4 | 16.959 | 0.99683 | 617.396 |  |
| sac | width | 256/512/1024 | gradient_updates | 3 | 44274.9 | 16.698 | 0.99669 | 621.297 |  |
| sac | width | 256/512/1024 | rollout | 3 | 4082.26 | 1067.9 | 0.99958 | 3.49912 |  |
| sac | batch | 256/512/1024 | total | 3 | 58101.4 | 8.4656 | 0.96803 | 3120.02 |  |
| sac | batch | 256/512/1024 | gradient_updates | 3 | 53858.2 | 8.3888 | 0.9669 | 3147.7 |  |
| sac | batch | 256/512/1024 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| td3 | UTD | 1/2/4 | total | 3 | 3317.19 | 103.46 | 1 | 83.2646 |  |
| td3 | UTD | 1/2/4 | gradient_updates | 3 | 261.673 | 103.53 | 1 | 66.1095 |  |
| td3 | UTD | 1/2/4 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| td3 | width | 256/512/1024 | total | 3 | 25269.5 | 20.593 | 0.93745 | 960.913 |  |
| td3 | width | 256/512/1024 | gradient_updates | 3 | 22446 | 19.66 | 0.93286 | 951.865 |  |
| td3 | width | 256/512/1024 | rollout | 3 | 2822.22 | 1076.4 | 0.99745 | 8.68119 |  |
| td3 | batch | 100/256/512/1024 | total | 4 | 29383.1 | 9.2895 | 0.96951 | 2141.35 |  |
| td3 | batch | 100/256/512/1024 | gradient_updates | 4 | 26347.5 | 9.1486 | 0.97006 | 2089.42 |  |
| td3 | batch | 100/256/512/1024 | rollout | 4 | n/a | n/a | n/a | n/a |  |
| mbpo | UTD | 1/2/4 | total | 3 | 169900 | 60.331 | 0.99885 | 4144.75 |  |
| mbpo | UTD | 1/2/4 | gradient_updates | 3 | 1168.54 | 62.516 | 1 | 202.24 |  |
| mbpo | UTD | 1/2/4 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| mbpo | UTD | 1/2/4 | dynamics_model_update | 3 | 79321.7 | 246.77 | 0.46584 | 4870.89 |  |
| mbpo | width | 256/512/1024 | total | 3 | 218281 | 24.068 | 0.90615 | 5059.06 |  |
| mbpo | width | 256/512/1024 | gradient_updates | 3 | 43222.9 | 17.519 | 0.99818 | 482.109 |  |
| mbpo | width | 256/512/1024 | rollout | 3 | 4048.4 | 1034.4 | 0.99718 | 8.82232 |  |
| mbpo | width | 256/512/1024 | dynamics_model_update | 3 | 165267 | 43.747 | 0.005303 | 6257.59 |  |
| mbpo | batch | 256/512/1024 | total | 3 | 240137 | 5.9809 | 0.67721 | 8408.29 |  |
| mbpo | batch | 256/512/1024 | gradient_updates | 3 | 53306.3 | 8.5498 | 0.97155 | 2966.71 |  |
| mbpo | batch | 256/512/1024 | rollout | 3 | n/a | n/a | n/a | n/a |  |
| mbpo | batch | 256/512/1024 | dynamics_model_update | 3 | 2316.02 | 412.72 | 0.33041 | 10195.5 |  |
| tdmpc2 | num_q | 3/5/7 | total | 3 | 68296.7 | 42.23 | 0.99996 | 774.058 | architecture change |
| tdmpc2 | num_q | 3/5/7 | gradient_updates | 3 | 46502.2 | 43.25 | 0.99839 | 2920.69 | architecture change |
| tdmpc2 | num_q | 3/5/7 | rollout | 3 | 24750 | 40.597 | 0.99727 | 2141.8 | architecture change |
| tdmpc2 | horizon | 1/3/5 | total | 3 | 135464 | 34.414 | 0.99702 | 13422.9 | architecture change |
| tdmpc2 | horizon | 1/3/5 | gradient_updates | 3 | 106645 | 26.471 | 0.98786 | 10673.8 | architecture change |
| tdmpc2 | horizon | 1/3/5 | rollout | 3 | 7101.36 | 43.724 | 0.99976 | 2239.1 | architecture change |

**TD-MPC2 note.** `num_q` changes the size of the critic ensemble (the Q-network count in the world model update, in the target update and in planning value evaluation) and `horizon` changes the length of the MPPI planning rollout in latent space and the unrolled world-model loss; both change the *architecture of the planner or world model*, not only the per-call batch/size of an unchanged computation. The slope b of these fits is therefore the marginal cost along a path that alters what is computed, not along a fixed workload.

## 6. Part (a) — adjacent-value ΔE/ΔF (marginal J per extra FLOP)

pJ/FLOP = ΔE/ΔF × 1e12 between configuration means of adjacent sweep values (same segments as Section 5; n/a = dropped, |ΔF| < 1 % of reference). Full columns in `a2_delta_pairs.csv`.

### HalfCheetah-v5

| algo | sweep | segment | from→to | ΔE [J] | ΔF | ΔE/ΔF [pJ/FLOP] |
|---|---|---|---|---|---|---|
| sac | UTD | total | 1→2 | 57113.4 | 8.76872e+14 | 65.133 |
| sac | UTD | total | 2→4 | 116935 | 1.75374e+15 | 66.677 |
| sac | UTD | gradient_updates | 1→2 | 57095.7 | 8.76872e+14 | 65.113 |
| sac | UTD | gradient_updates | 2→4 | 116895 | 1.75374e+15 | 66.655 |
| sac | UTD | rollout | 1→2 | 17.6845 | 0 | n/a |
| sac | UTD | rollout | 2→4 | 39.4863 | 0 | n/a |
| sac | width | total | 256→512 | 2146.06 | 1.65572e+14 | 12.961 |
| sac | width | total | 512→1024 | 11357 | 6.53344e+14 | 17.383 |
| sac | width | gradient_updates | 256→512 | 2097.75 | 1.65531e+14 | 12.673 |
| sac | width | gradient_updates | 512→1024 | 11188.1 | 6.53184e+14 | 17.129 |
| sac | width | rollout | 256→512 | 48.3083 | 4.08064e+10 | 1183.8 |
| sac | width | rollout | 512→1024 | 168.958 | 1.60256e+11 | 1054.3 |
| sac | batch | total | 256→512 | 12112.9 | 8.76872e+14 | 13.814 |
| sac | batch | total | 512→1024 | 9341.99 | 1.75374e+15 | 5.3269 |
| sac | batch | gradient_updates | 256→512 | 11961.7 | 8.76872e+14 | 13.641 |
| sac | batch | gradient_updates | 512→1024 | 9240.81 | 1.75374e+15 | 5.2692 |
| sac | batch | rollout | 256→512 | 151.29 | 0 | n/a |
| sac | batch | rollout | 512→1024 | 101.177 | 0 | n/a |
| td3 | UTD | total | 1→2 | 27249.5 | 2.45658e+14 | 110.92 |
| td3 | UTD | total | 2→4 | 51983.9 | 4.91315e+14 | 105.81 |
| td3 | UTD | gradient_updates | 1→2 | 27228.9 | 2.45658e+14 | 110.84 |
| td3 | UTD | gradient_updates | 2→4 | 52035 | 4.91315e+14 | 105.91 |
| td3 | UTD | rollout | 1→2 | 20.6869 | 0 | n/a |
| td3 | UTD | rollout | 2→4 | -51.1436 | 0 | n/a |
| td3 | width | total | 256→512 | 2332.72 | 4.63816e+13 | 50.294 |
| td3 | width | total | 512→1024 | 2741.78 | 1.83282e+14 | 14.959 |
| td3 | width | gradient_updates | 256→512 | 2270.97 | 4.63411e+13 | 49.006 |
| td3 | width | gradient_updates | 512→1024 | 2632.8 | 1.83122e+14 | 14.377 |
| td3 | width | rollout | 256→512 | 61.7491 | 4.04992e+10 | 1524.7 |
| td3 | width | rollout | 512→1024 | 108.978 | 1.59642e+11 | 682.64 |
| td3 | batch | total | 100→256 | 5342.27 | 3.83226e+14 | 13.94 |
| td3 | batch | total | 256→512 | 8229.49 | 6.28883e+14 | 13.086 |
| td3 | batch | total | 512→1024 | 6385.22 | 1.25777e+15 | 5.0766 |
| td3 | batch | gradient_updates | 100→256 | 5236.36 | 3.83226e+14 | 13.664 |
| td3 | batch | gradient_updates | 256→512 | 8064.7 | 6.28883e+14 | 12.824 |
| td3 | batch | gradient_updates | 512→1024 | 6314.43 | 1.25777e+15 | 5.0204 |
| td3 | batch | rollout | 100→256 | 105.912 | 0 | n/a |
| td3 | batch | rollout | 256→512 | 164.786 | 0 | n/a |
| td3 | batch | rollout | 512→1024 | 70.7841 | 0 | n/a |
| mbpo | UTD | total | 1→2 | 61456.1 | 8.87236e+14 | 69.267 |
| mbpo | UTD | total | 2→4 | 104756 | 1.73781e+15 | 60.28 |
| mbpo | UTD | gradient_updates | 1→2 | 57629.6 | 8.76872e+14 | 65.722 |
| mbpo | UTD | gradient_updates | 2→4 | 115503 | 1.75374e+15 | 65.861 |
| mbpo | UTD | rollout | 1→2 | -18.5847 | 0 | n/a |
| mbpo | UTD | rollout | 2→4 | -28.5827 | 0 | n/a |
| mbpo | UTD | dynamics_model_update | 1→2 | 3847.92 | 1.03643e+13 | 371.27 |
| mbpo | UTD | dynamics_model_update | 2→4 | -10720.7 | -1.59333e+13 | 672.85 |
| mbpo | width | total | 256→512 | 4855.98 | 1.70514e+14 | 28.478 |
| mbpo | width | total | 512→1024 | 14625.3 | 6.55593e+14 | 22.309 |
| mbpo | width | gradient_updates | 256→512 | 2238.65 | 1.65531e+14 | 13.524 |
| mbpo | width | gradient_updates | 512→1024 | 11909.6 | 6.53184e+14 | 18.233 |
| mbpo | width | rollout | 256→512 | 23.1481 | 4.08064e+10 | 567.27 |
| mbpo | width | rollout | 512→1024 | 92.4796 | 1.60256e+11 | 577.07 |
| mbpo | width | dynamics_model_update | 256→512 | 2581.67 | 4.53813e+12 | 568.89 |
| mbpo | width | dynamics_model_update | 512→1024 | 2613.58 | 6.61592e+11 | n/a |
| mbpo | batch | total | 256→512 | 1706.86 | 8.67085e+14 | 1.9685 |
| mbpo | batch | total | 512→1024 | 9565.64 | 1.75338e+15 | 5.4555 |
| mbpo | batch | gradient_updates | 256→512 | 10670.8 | 8.76872e+14 | 12.169 |
| mbpo | batch | gradient_updates | 512→1024 | 8688.22 | 1.75374e+15 | 4.9541 |
| mbpo | batch | rollout | 256→512 | -44.8684 | 0 | n/a |
| mbpo | batch | rollout | 512→1024 | -20.5882 | 0 | n/a |
| mbpo | batch | dynamics_model_update | 256→512 | -8910.21 | -9.78625e+12 | 910.48 |
| mbpo | batch | dynamics_model_update | 512→1024 | 883.873 | -3.63544e+11 | n/a |
| tdmpc2 | num_q | total | 3→5 | 88299.6 | 1.9568e+15 | 45.124 |
| tdmpc2 | num_q | total | 5→7 | 79475.4 | 1.9568e+15 | 40.615 |
| tdmpc2 | num_q | gradient_updates | 3→5 | 53891.7 | 1.18594e+15 | 45.442 |
| tdmpc2 | num_q | gradient_updates | 5→7 | 49220.7 | 1.18594e+15 | 41.504 |
| tdmpc2 | num_q | rollout | 3→5 | 31823.1 | 7.11564e+14 | 44.723 |
| tdmpc2 | num_q | rollout | 5→7 | 28383.6 | 7.11564e+14 | 39.889 |
| tdmpc2 | horizon | total | 1→3 | 166400 | 4.21818e+15 | 39.448 |
| tdmpc2 | horizon | total | 3→5 | 124194 | 4.21818e+15 | 29.443 |
| tdmpc2 | horizon | gradient_updates | 1→3 | 83593.5 | 2.40362e+15 | 34.778 |
| tdmpc2 | horizon | gradient_updates | 3→5 | 47806.9 | 2.40362e+15 | 19.89 |
| tdmpc2 | horizon | rollout | 1→3 | 78633.4 | 1.69438e+15 | 46.408 |
| tdmpc2 | horizon | rollout | 3→5 | 74455.9 | 1.69438e+15 | 43.943 |

### Ant-v5

| algo | sweep | segment | from→to | ΔE [J] | ΔF | ΔE/ΔF [pJ/FLOP] |
|---|---|---|---|---|---|---|
| sac | UTD | total | 1→2 | 58799.8 | 9.38738e+14 | 62.637 |
| sac | UTD | total | 2→4 | 118144 | 1.87748e+15 | 62.927 |
| sac | UTD | gradient_updates | 1→2 | 58823.4 | 9.38738e+14 | 62.662 |
| sac | UTD | gradient_updates | 2→4 | 118160 | 1.87748e+15 | 62.935 |
| sac | UTD | rollout | 1→2 | -23.5445 | 0 | n/a |
| sac | UTD | rollout | 2→4 | -15.0376 | 0 | n/a |
| sac | width | total | 256→512 | 2214.56 | 1.81043e+14 | 12.232 |
| sac | width | total | 512→1024 | 12182.8 | 6.84287e+14 | 17.804 |
| sac | width | gradient_updates | 256→512 | 2161.11 | 1.80997e+14 | 11.94 |
| sac | width | gradient_updates | 512→1024 | 12004.9 | 6.84117e+14 | 17.548 |
| sac | width | rollout | 256→512 | 53.4561 | 4.55168e+10 | 1174.4 |
| sac | width | rollout | 512→1024 | 177.917 | 1.69677e+11 | 1048.6 |
| sac | batch | total | 256→512 | 12116.3 | 9.38738e+14 | 12.907 |
| sac | batch | total | 512→1024 | 12558.5 | 1.87748e+15 | 6.689 |
| sac | batch | gradient_updates | 256→512 | 12081.2 | 9.38738e+14 | 12.87 |
| sac | batch | gradient_updates | 512→1024 | 12384.7 | 1.87748e+15 | 6.5965 |
| sac | batch | rollout | 256→512 | 35.1084 | 0 | n/a |
| sac | batch | rollout | 512→1024 | 173.758 | 0 | n/a |
| td3 | UTD | total | 1→2 | 27022.8 | 2.62267e+14 | 103.04 |
| td3 | UTD | total | 2→4 | 54357.1 | 5.24534e+14 | 103.63 |
| td3 | UTD | gradient_updates | 1→2 | 27063 | 2.62267e+14 | 103.19 |
| td3 | UTD | gradient_updates | 2→4 | 54373.4 | 5.24534e+14 | 103.66 |
| td3 | UTD | rollout | 1→2 | -40.2438 | 0 | n/a |
| td3 | UTD | rollout | 2→4 | -16.2992 | 0 | n/a |
| td3 | width | total | 256→512 | 2372.81 | 5.05385e+13 | 46.95 |
| td3 | width | total | 512→1024 | 3046.61 | 1.91595e+14 | 15.901 |
| td3 | width | gradient_updates | 256→512 | 2312.22 | 5.04934e+13 | 45.793 |
| td3 | width | gradient_updates | 512→1024 | 2872.99 | 1.91427e+14 | 15.008 |
| td3 | width | rollout | 256→512 | 60.5842 | 4.51072e+10 | 1343.1 |
| td3 | width | rollout | 512→1024 | 173.624 | 1.68858e+11 | 1028.2 |
| td3 | batch | total | 100→256 | 5000.71 | 4.09136e+14 | 12.223 |
| td3 | batch | total | 256→512 | 8840.64 | 6.71403e+14 | 13.167 |
| td3 | batch | total | 512→1024 | 8873.81 | 1.34281e+15 | 6.6084 |
| td3 | batch | gradient_updates | 100→256 | 4985.59 | 4.09136e+14 | 12.186 |
| td3 | batch | gradient_updates | 256→512 | 8625.45 | 6.71403e+14 | 12.847 |
| td3 | batch | gradient_updates | 512→1024 | 8784.74 | 1.34281e+15 | 6.5421 |
| td3 | batch | rollout | 100→256 | 15.1154 | 0 | n/a |
| td3 | batch | rollout | 256→512 | 215.192 | 0 | n/a |
| td3 | batch | rollout | 512→1024 | 89.067 | 0 | n/a |
| mbpo | UTD | total | 1→2 | 63658.6 | 9.6364e+14 | 66.061 |
| mbpo | UTD | total | 2→4 | 107657 | 1.85846e+15 | 57.928 |
| mbpo | UTD | gradient_updates | 1→2 | 58956.4 | 9.38738e+14 | 62.804 |
| mbpo | UTD | gradient_updates | 2→4 | 117156 | 1.87748e+15 | 62.401 |
| mbpo | UTD | rollout | 1→2 | -77.8582 | 0 | n/a |
| mbpo | UTD | rollout | 2→4 | -35.5887 | 0 | n/a |
| mbpo | UTD | dynamics_model_update | 1→2 | 4765.49 | 2.54336e+13 | 187.37 |
| mbpo | UTD | dynamics_model_update | 2→4 | -9424.96 | -1.76682e+13 | 533.44 |
| mbpo | width | total | 256→512 | -2852.69 | 1.73452e+14 | -16.447 |
| mbpo | width | total | 512→1024 | 21500 | 6.98973e+14 | 30.759 |
| mbpo | width | gradient_updates | 256→512 | 2502.61 | 1.80997e+14 | 13.827 |
| mbpo | width | gradient_updates | 512→1024 | 12436.2 | 6.84117e+14 | 18.178 |
| mbpo | width | rollout | 256→512 | 34.8591 | 4.55168e+10 | 765.85 |
| mbpo | width | rollout | 512→1024 | 183.791 | 1.69677e+11 | 1083.2 |
| mbpo | width | dynamics_model_update | 256→512 | -5464.71 | -1.22683e+13 | 445.43 |
| mbpo | width | dynamics_model_update | 512→1024 | 8786.44 | -9.88982e+11 | n/a |
| mbpo | batch | total | 256→512 | -5744.93 | 9.22791e+14 | -6.2256 |
| mbpo | batch | total | 512→1024 | 20301.8 | 1.90101e+15 | 10.679 |
| mbpo | batch | gradient_updates | 256→512 | 11990.4 | 9.38738e+14 | 12.773 |
| mbpo | batch | gradient_updates | 512→1024 | 12880.4 | 1.87748e+15 | 6.8605 |
| mbpo | batch | rollout | 256→512 | -178.952 | 0 | n/a |
| mbpo | batch | rollout | 512→1024 | 64.6492 | 0 | n/a |
| mbpo | batch | dynamics_model_update | 256→512 | -17544.7 | -1.5908e+13 | 1102.9 |
| mbpo | batch | dynamics_model_update | 512→1024 | 7357.25 | 2.41374e+13 | 304.81 |
| tdmpc2 | num_q | total | 3→5 | 81834.2 | 1.96026e+15 | 41.747 |
| tdmpc2 | num_q | total | 5→7 | 83730.3 | 1.96026e+15 | 42.714 |
| tdmpc2 | num_q | gradient_updates | 3→5 | 47805.5 | 1.18804e+15 | 40.239 |
| tdmpc2 | num_q | gradient_updates | 5→7 | 54959.7 | 1.18804e+15 | 46.261 |
| tdmpc2 | num_q | rollout | 3→5 | 31561.3 | 7.12822e+14 | 44.276 |
| tdmpc2 | num_q | rollout | 5→7 | 26314.9 | 7.12822e+14 | 36.917 |
| tdmpc2 | horizon | total | 1→3 | 189960 | 5.04215e+15 | 37.674 |
| tdmpc2 | horizon | total | 3→5 | 157081 | 5.04215e+15 | 31.154 |
| tdmpc2 | horizon | gradient_updates | 1→3 | 81149.7 | 2.57176e+15 | 31.554 |
| tdmpc2 | horizon | gradient_updates | 3→5 | 55004.4 | 2.57176e+15 | 21.388 |
| tdmpc2 | horizon | rollout | 1→3 | 105135 | 2.3418e+15 | 44.895 |
| tdmpc2 | horizon | rollout | 3→5 | 99649.9 | 2.3418e+15 | 42.553 |

## 7. Check of the SAC values quoted in the thesis

Per-run fits (n = 15, as in Section 4.1) — identical to the configuration-mean fits of Section 5 for SAC:

| sweep | env | segment | n | a [J] | b [pJ/FLOP] | R² |
|---|---|---|---|---|---|---|
| UTD | Ant-v5 | gradient_updates | 15 | 970.775 | 62.86 | 0.9999 |
| UTD | Ant-v5 | total | 15 | 5295.67 | 62.84 | 0.9999 |
| UTD | HalfCheetah-v5 | gradient_updates | 15 | -32.2806 | 66.21 | 0.9999 |
| UTD | HalfCheetah-v5 | total | 15 | 2429.4 | 66.24 | 0.9999 |
| width | Ant-v5 | gradient_updates | 15 | 44274.9 | 16.7 | 0.9896 |
| width | Ant-v5 | total | 15 | 48358.4 | 16.96 | 0.9896 |
| width | HalfCheetah-v5 | gradient_updates | 15 | 43891 | 16.48 | 0.9938 |
| width | HalfCheetah-v5 | total | 15 | 46156.2 | 16.74 | 0.994 |
| batch | Ant-v5 | gradient_updates | 15 | 53858.2 | 8.389 | 0.9653 |
| batch | Ant-v5 | total | 15 | 58101.4 | 8.466 | 0.9663 |
| batch | HalfCheetah-v5 | gradient_updates | 15 | 53794.9 | 7.661 | 0.9296 |
| batch | HalfCheetah-v5 | total | 15 | 56238.4 | 7.752 | 0.9294 |

- Width marginal slope: 16.48–16.96 pJ/FLOP (thesis: ≈16.5–17.0) — **reproduces**.
- Batch-size marginal slope: 7.66–8.47 pJ/FLOP (thesis: 7.7–8.5) — **reproduces**.
- UTD marginal slope: 62.84–66.24 pJ/FLOP (thesis: 62.8–66.2) — **reproduces**.
- Width and batch intercepts: 43.9–58.1 kJ over total and GU, both environments (thesis: ≈44–58 kJ) — **reproduces** (width GU/total 43.9–48.4 kJ, batch 53.8–58.1 kJ). UTD intercepts are small (−0.03 to 5.3 kJ).

## 8. Inconsistencies, surprises and things not computed

**Consistency with Chapter 4 — none found.** For all 56 configurations and every segment present (210 comparisons of seed-mean energy, FLOPs, J/FLOP and sd of per-seed J/FLOP against `contexts/section_4.1_data/sac_cross_seed_summary.csv`, `Section_4.2_data/td3_…`, `Section_4.3_data/mbpo_…`, `Section_4.4_data/tdmpc2_…`) the maximum relative deviation is 4.4e-16 (energy), 0 (FLOPs), 4.4e-16 (J/FLOP), 4.5e-14 (sd) — `x1_check_vs_chapter4.csv`. All 56 configuration total energies also match `flop_analysis/output/cross_seed_energy_per_flop.csv` to 1e-9. The pooled correlation values match `correlation_summary.csv`.

**Surprising or noteworthy**
- **Config-mean vs per-run fits differ for MBPO only.** Because MBPO FLOPs vary per seed, its per-run and configuration-mean fits are not the same regression (maximum relative slope difference 0.89 over the MBPO fit rows); SAC, TD3, TD-MPC2 fits are identical (difference ≤ 2.3e-15). The tables here use configuration means.
- **MBPO `dynamics_model_update` does not scale with the swept factor.** The dynamics ensemble is a fixed 7×(200×4) network trained per epoch with early stopping; its FLOPs per configuration range only 2.2e14–2.8e14 (HC) and 3.6e14–5.0e14 (Ant) across seeds and sweeps with no monotone relation to UTD/width/batch (e.g. HC energy 137–150 kJ). The dynamics linear fits are therefore dominated by early-stopping noise: HC UTD a = −15.6 kJ, b = 635 pJ/FLOP; HC width a = −66.7 kJ; Ant width R² = 0.0053 (b = 43.7 pJ/FLOP); Ant batch R² = 0.33; Ant UTD R² = 0.47. Negative intercepts appear in these fits.
- **MBPO total fits carry the dynamics-model block in the intercept** (a ≈ 138–240 kJ vs 1–54 kJ for the MBPO GU fits), and the MBPO total slope differs from its GU slope (e.g. HC width total 23.2 vs GU 17.6 pJ/FLOP; HC batch 4.47 vs 7.02; Ant batch total R² = 0.68).
- **Pooled vs within-algorithm log-log slopes.** Pooled total slope 0.47 / 0.51; within algorithm 0.17–0.28 for SAC/TD3/MBPO and 0.71 for TD-MPC2 (both environments); within-algorithm R² 0.47–0.57 (SAC/TD3/MBPO) and 0.99 (TD-MPC2).
- **Rank correlation is high even where log-log R² is moderate** (pooled ρ 0.84–0.85, R² 0.61–0.63). Rollout alone: ρ 0.88, R² 0.95–0.96 (slope 0.38–0.42); pooled GU: ρ 0.84–0.87, R² 0.62–0.64.
- **Marginal cost differs across sweeps.** Marginal costs b of SAC (width ≈ 16.5–17.0, batch ≈ 7.7–8.5, UTD ≈ 62.8–66.2 pJ/FLOP) differ by a factor of ≈ 8 between the batch and UTD sweeps of the same algorithm. Rollout slopes for the width sweep: SAC ≈ 1.07, TD3 0.80–1.08, MBPO 0.58–1.03 nJ/FLOP (batch-size-1 matmuls).
- **TD3 UTD marginal cost (103–107 pJ/FLOP) is higher than SAC's (≈ 63–66) and MBPO's GU (≈ 62–66).**
- **TD-MPC2 `horizon` fits have large residuals in absolute terms** (total residual SE 13.4–17.2 kJ; GU R² 0.976–0.988), larger than the `num_q` total fits (0.8–3.6 kJ).
- **Equal FLOPs, different energy.** SAC `utd2` and `b512` have identical total FLOPs (HC 1.753959e15; energy 118,024 J vs 73,024 J), as do `utd4` and `b1024` (3.507702e15; 234,959 J vs 82,366 J); same pattern on Ant (123,182 vs 76,499 J; 241,327 vs 89,057 J). MBPO `utd2`/`b512` are within ≈ 1 % of each other in FLOPs (HC 2.02e15 vs 2.00e15) with energy 268,851 vs 209,101 J. TD3 has no such tie.

**Could not compute / not done**
- Rollout-segment linear fits and ΔE/ΔF for the UTD and batch sweeps of SAC, TD3 and MBPO: undefined (rollout FLOPs constant across those sweeps).
- A GU row is not in `cross_seed_energy_per_flop.csv`; GU was reconstructed as described in Section 0 (reconciled to the Chapter 4 `gradient_updates` rows to 4.4e-16, so this is not a source of difference).
- MBPO rollout-length sweep (Ant, `rollout1`/`rollout15`) is in the config table and the pooled statistics but has no linear fit or delta pairs (not requested).
- No inferential p-values for the sweeps: n = 3–4 per fit and non-independent configurations (descriptive only). No confidence intervals on the linear-fit parameters are given (one residual df for n = 3).
- Per-segment statistics for `target_update` (elementwise) and `buffer_sample` (no FLOPs) and the allocated sub-segments `critic_update`/`actor_update` were excluded as instructed.

## 9. Index of produced files

`contexts/section_5.2_data/`: `d1_config_table.csv` (56 rows), `d2_total_correlation.csv`, `d3_segment_correlation.csv`, `d4_per_algo_total_correlation.csv`, `d5_per_algo_segment_correlation.csv` (supplement), `a1_linear_fits.csv` (config-mean and per-run fits), `a2_delta_pairs.csv`, `x1_check_vs_chapter4.csv`, `per_run_segments_5_2.csv` (280 rows: per-run total, GU and segment energies/FLOPs). Scripts: `flop_analysis/section_5_2_analysis.py`, `flop_analysis/section_5_2_render.py`.
