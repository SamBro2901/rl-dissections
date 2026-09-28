# FIX_REPORT -- energy-per-FLOP aggregation fixes

Branch `fix/flop-aggregation`. "Old" = `flop_analysis/output/_before_fix/*.csv` (output of
`compute_energy_per_flop.py` at `0c360ad`); "new" = `flop_analysis/output/*.csv` after both fixes.
Cross-seed numbers are over the canonical seeds {331, 958, 14577, 43611, 85062}; J/FLOP is
mean energy / mean FLOPs per group, as in the cross-seed CSV. Per-call FLOP constants
(`flops_per_call.json`) are unchanged; only aggregation and call counts changed.
Regenerate this file with `python flop_analysis/make_fix_report.py`.

## Headline numbers

- **TOTAL_MEASURED_TRAINING J/FLOP**: direction varies by group. Largest change
  **+12.75%** (td3 / Ant-v5 / bs1024_h1024x1024 / UTD1), smallest -0.03%
  (mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD2 / rl1-25_re20-100). Per run (all 300 runs, including non-canonical seeds): -5.76% to +13.08%.
  Non-MBPO groups: +1.86% to +12.75% (Fix 1 only: `buffer_sample` energy
  enters the numerator, `target_update` ops leave the denominator). MBPO groups:
  -5.61% to +1.49%, because Fix 2's ~8% more dynamics FLOPs in the denominator
  offset the added `buffer_sample` energy.
- **MBPO dynamics_model_update FLOPs**: up **+8.10% to +8.38%** across groups
  (largest: mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD4 / rl1-25_re20-100), so J/FLOP goes down by the same factor
  (-7.73% to -7.49%). Holdout evaluation makes up
  7.78%-7.99% of the corrected count. Per run (all 83 MBPO runs):
  +8.08% to +8.40%. `call_count` (optimizer steps) is unchanged in every run.
- `verify_mbpo_fit_flops.py`: **ALL PASSED** (see output below for per-case relative differences).
  `check_totals.py`: **ALL CHECKS PASSED**.

## Definitions: old vs. new

### TOTAL_MEASURED_TRAINING

**Old.** A segment counted toward both numerator and denominator iff it was not an idle
baseline, not `warmup`, and had non-zero `total_flops`. So `target_update` was *included*: its
energy in the numerator, its elementwise Polyak op count in the denominator, mixed with matmul
FLOPs. `buffer_sample` (0 FLOPs) was *excluded*, so part of the measured `gradient_updates`
energy dropped out of the total. Duration/power fields summed the allocated sub-segments'
`perf_counter` durations. The row's note claimed `target_update` was excluded, but it wasn't.

**New: all training energy / matmul FLOPs only.**

```
numerator   = sum of segment_energy.json energy over TRAINING_SEGMENTS present:
                rollout,
                buffer_sample, critic_update, actor_update, target_update   (= all of gradient_updates),
                dynamics_model_update, synthetic_rollout_generation          (MBPO),
                world_model_pretrain                                         (TD-MPC2)
              excluding idle_baseline_head/tail, warmup and "_"-prefixed keys
denominator = sum of total_flops over the same segments with flop_type == "matmul"
              (i.e. all except target_update; buffer_sample contributes 0)
```

Segment membership is fixed by module-level constants (`TRAINING_SEGMENTS`,
`ELEMENTWISE_OP_SEGMENTS`, `ZERO_FLOP_SEGMENTS`), not inferred from FLOP counts. Duration/power
fields of the TOTAL row now come from the directly measured CodeCarbon tasks: the whole
`gradient_updates` task stands in for the four allocated sub-segments.

New `flop_type` column in both CSVs: `matmul` | `elementwise` (`target_update`: its
`energy_per_flop_j_per_flop` is **J per elementwise op**, not J/FLOP) | `none` (`buffer_sample`,
`warmup`, idle baselines) | `mixed_total` (the TOTAL row).

### MBPO dynamics_model_update FLOPs

**Old.** Per logged fit() call: `ceil(n_train / B) x epochs_run x ensemble_size x member_fwdbwd(B)`.
This leaves out `_holdout_mse()`, which scores all members on the holdout split after every fit
epoch, and charges the partial last minibatch as a full batch of B.

**New** (`mbpo_fit_flops()`, mirrors `EnsembleDynamicsModel.fit()` line by line):

```
per_sample_member_fwdbwd = dynamics_member_fwdbwd / model_train_batch_size   (exact integer division, asserted)
per_sample_ensemble_fwd  = dynamics_ensemble_forward_all_bs1
for each epochs_log row with model_train_epochs = E (skipped if None/0):
    n_total   = min(warmup_steps + epoch * steps_per_epoch, buffer_capacity)
    n_holdout = max(1, int(n_total * model_holdout_ratio))
    n_train   = n_total - n_holdout
    train     = E * ensemble_size * n_train * per_sample_member_fwdbwd
    holdout   = E * n_holdout * per_sample_ensemble_fwd
call_count  = sum of E * ceil(n_train / B)    (optimizer steps)
```

Checks behind this formula: `algorithms/mbpo.py` calls `fit()` at the start of each epoch,
before that epoch's rollout; the real buffer is `ReplayBuffer(buffer_capacity=1,000,000)`, never
reached in these runs; and the `n_total` formula matches every logged end-of-epoch
`buffer_size` (= n_total + steps_per_epoch) in all MBPO runs (the script warns on mismatch,
and no warnings were raised). New columns `dynamics_train_flops` / `dynamics_holdout_flops` hold
the split.

## TOTAL_MEASURED_TRAINING, per group (old -> new)

| group | E old (kWh) | E new (kWh) | dE | FLOPs old | FLOPs new | dF | J/FLOP old | J/FLOP new | dJ/FLOP |
|---|---|---|---|---|---|---|---|---|---|
| mbpo / Ant-v5 / bs1024_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 0.07269 | 0.07436 | +2.30% | 4.2064e+15 | 4.2398e+15 | +0.80% | 6.2211e-11 | 6.3138e-11 | +1.49% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-15_re20-100 | 0.06767 | 0.06830 | +0.92% | 1.3728e+15 | 1.4060e+15 | +2.42% | 1.7746e-10 | 1.7486e-10 | -1.46% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-100 | 0.06631 | 0.06687 | +0.85% | 1.3396e+15 | 1.3721e+15 | +2.43% | 1.7822e-10 | 1.7546e-10 | -1.55% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 0.06969 | 0.07032 | +0.91% | 1.3833e+15 | 1.4160e+15 | +2.37% | 1.8136e-10 | 1.7877e-10 | -1.43% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD2 / rl1-25_re20-100 | 0.08676 | 0.08800 | +1.43% | 2.3453e+15 | 2.3797e+15 | +1.46% | 1.3317e-10 | 1.3313e-10 | -0.03% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD4 / rl1-25_re20-100 | 0.11544 | 0.11790 | +2.13% | 4.2061e+15 | 4.2381e+15 | +0.76% | 9.8806e-11 | 1.0015e-10 | +1.36% |
| mbpo / Ant-v5 / bs256_h256x256_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 0.06466 | 0.06514 | +0.74% | 5.0934e+14 | 5.4360e+14 | +6.73% | 4.5701e-10 | 4.3136e-10 | -5.61% |
| mbpo / Ant-v5 / bs256_h512x512_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 0.06384 | 0.06434 | +0.78% | 6.8383e+14 | 7.1705e+14 | +4.86% | 3.3611e-10 | 3.2304e-10 | -3.89% |
| mbpo / Ant-v5 / bs512_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 0.06774 | 0.06872 | +1.45% | 2.3073e+15 | 2.3388e+15 | +1.37% | 1.0569e-10 | 1.0578e-10 | +0.08% |
| mbpo / HalfCheetah-v5 / bs1024_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 0.05992 | 0.06074 | +1.36% | 3.7356e+15 | 3.7533e+15 | +0.47% | 5.7748e-11 | 5.8260e-11 | +0.89% |
| mbpo / HalfCheetah-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 0.05714 | 0.05761 | +0.82% | 1.1144e+15 | 1.1329e+15 | +1.66% | 1.8460e-10 | 1.8307e-10 | -0.83% |
| mbpo / HalfCheetah-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD2 / rl1-1_re20-150 | 0.07376 | 0.07468 | +1.25% | 2.0012e+15 | 2.0201e+15 | +0.94% | 1.3268e-10 | 1.3309e-10 | +0.31% |
| mbpo / HalfCheetah-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD4 / rl1-1_re20-150 | 0.10195 | 0.10378 | +1.80% | 3.7411e+15 | 3.7579e+15 | +0.45% | 9.8101e-11 | 9.9419e-11 | +1.34% |
| mbpo / HalfCheetah-v5 / bs256_h256x256_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 0.05184 | 0.05220 | +0.68% | 2.8826e+14 | 3.0675e+14 | +6.41% | 6.4748e-10 | 6.1260e-10 | -5.39% |
| mbpo / HalfCheetah-v5 / bs256_h512x512_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 0.05318 | 0.05355 | +0.70% | 4.5849e+14 | 4.7726e+14 | +4.09% | 4.1753e-10 | 4.0391e-10 | -3.26% |
| mbpo / HalfCheetah-v5 / bs512_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 0.05748 | 0.05808 | +1.06% | 1.9822e+15 | 1.9999e+15 | +0.89% | 1.0439e-10 | 1.0455e-10 | +0.16% |
| sac / Ant-v5 / bs1024_h1024x1024 / UTD1 | 0.02339 | 0.02474 | +5.78% | 3.7557e+15 | 3.7552e+15 | -0.01% | 2.2418e-11 | 2.3716e-11 | +5.79% |
| sac / Ant-v5 / bs256_h1024x1024 / UTD1 | 0.01735 | 0.01788 | +3.08% | 9.3944e+14 | 9.3897e+14 | -0.05% | 6.6486e-11 | 6.8567e-11 | +3.13% |
| sac / Ant-v5 / bs256_h1024x1024 / UTD2 | 0.03315 | 0.03422 | +3.21% | 1.8786e+15 | 1.8777e+15 | -0.05% | 6.3531e-11 | 6.5603e-11 | +3.26% |
| sac / Ant-v5 / bs256_h1024x1024 / UTD4 | 0.06492 | 0.06704 | +3.26% | 3.7571e+15 | 3.7552e+15 | -0.05% | 6.2206e-11 | 6.4265e-11 | +3.31% |
| sac / Ant-v5 / bs256_h256x256 / UTD1 | 0.01348 | 0.01388 | +3.00% | 7.3681e+13 | 7.3642e+13 | -0.05% | 6.5863e-10 | 6.7876e-10 | +3.06% |
| sac / Ant-v5 / bs256_h512x512 / UTD1 | 0.01407 | 0.01450 | +3.05% | 2.5481e+14 | 2.5469e+14 | -0.05% | 1.9879e-10 | 2.0496e-10 | +3.10% |
| sac / Ant-v5 / bs512_h1024x1024 / UTD1 | 0.02045 | 0.02125 | +3.92% | 1.8782e+15 | 1.8777e+15 | -0.02% | 3.9193e-11 | 4.0741e-11 | +3.95% |
| sac / HalfCheetah-v5 / bs1024_h1024x1024 / UTD1 | 0.02206 | 0.02288 | +3.73% | 3.5081e+15 | 3.5077e+15 | -0.01% | 2.2635e-11 | 2.3481e-11 | +3.74% |
| sac / HalfCheetah-v5 / bs256_h1024x1024 / UTD1 | 0.01649 | 0.01692 | +2.61% | 8.7752e+14 | 8.7709e+14 | -0.05% | 6.7646e-11 | 6.9447e-11 | +2.66% |
| sac / HalfCheetah-v5 / bs256_h1024x1024 / UTD2 | 0.03194 | 0.03278 | +2.65% | 1.7548e+15 | 1.7540e+15 | -0.05% | 6.5521e-11 | 6.7290e-11 | +2.70% |
| sac / HalfCheetah-v5 / bs256_h1024x1024 / UTD4 | 0.06358 | 0.06527 | +2.65% | 3.5094e+15 | 3.5077e+15 | -0.05% | 6.5221e-11 | 6.6984e-11 | +2.70% |
| sac / HalfCheetah-v5 / bs256_h256x256 / UTD1 | 0.01284 | 0.01317 | +2.57% | 5.8200e+13 | 5.8171e+13 | -0.05% | 7.9413e-10 | 8.1497e-10 | +2.62% |
| sac / HalfCheetah-v5 / bs256_h512x512 / UTD1 | 0.01342 | 0.01376 | +2.58% | 2.2385e+14 | 2.2374e+14 | -0.05% | 2.1581e-10 | 2.2148e-10 | +2.63% |
| sac / HalfCheetah-v5 / bs512_h1024x1024 / UTD1 | 0.01970 | 0.02028 | +2.97% | 1.7544e+15 | 1.7540e+15 | -0.02% | 4.0425e-11 | 4.1634e-11 | +2.99% |
| td3 / Ant-v5 / bs100_h1024x1024 / UTD1 | 0.00807 | 0.00848 | +5.07% | 2.6285e+14 | 2.6250e+14 | -0.13% | 1.1051e-10 | 1.1627e-10 | +5.21% |
| td3 / Ant-v5 / bs100_h1024x1024 / UTD2 | 0.01516 | 0.01598 | +5.40% | 5.2547e+14 | 5.2477e+14 | -0.13% | 1.0389e-10 | 1.0965e-10 | +5.54% |
| td3 / Ant-v5 / bs100_h1024x1024 / UTD4 | 0.02946 | 0.03108 | +5.50% | 1.0507e+15 | 1.0493e+15 | -0.13% | 1.0094e-10 | 1.0664e-10 | +5.65% |
| td3 / Ant-v5 / bs100_h256x256 / UTD1 | 0.00665 | 0.00697 | +4.85% | 2.0394e+13 | 2.0366e+13 | -0.14% | 1.1739e-09 | 1.2325e-09 | +4.99% |
| td3 / Ant-v5 / bs100_h512x512 / UTD1 | 0.00729 | 0.00763 | +4.75% | 7.1001e+13 | 7.0904e+13 | -0.14% | 3.6941e-10 | 3.8747e-10 | +4.89% |
| td3 / Ant-v5 / bs1024_h1024x1024 / UTD1 | 0.01312 | 0.01479 | +12.73% | 2.6862e+15 | 2.6858e+15 | -0.01% | 1.7580e-11 | 1.9821e-11 | +12.75% |
| td3 / Ant-v5 / bs256_h1024x1024 / UTD1 | 0.00927 | 0.00987 | +6.39% | 6.7199e+14 | 6.7164e+14 | -0.05% | 4.9686e-11 | 5.2887e-11 | +6.44% |
| td3 / Ant-v5 / bs512_h1024x1024 / UTD1 | 0.01136 | 0.01232 | +8.47% | 1.3434e+15 | 1.3430e+15 | -0.03% | 3.0444e-11 | 3.3030e-11 | +8.50% |
| td3 / HalfCheetah-v5 / bs100_h1024x1024 / UTD1 | 0.00745 | 0.00781 | +4.83% | 2.4619e+14 | 2.4587e+14 | -0.13% | 1.0892e-10 | 1.1433e-10 | +4.97% |
| td3 / HalfCheetah-v5 / bs100_h1024x1024 / UTD2 | 0.01465 | 0.01538 | +4.99% | 4.9217e+14 | 4.9153e+14 | -0.13% | 1.0714e-10 | 1.1263e-10 | +5.13% |
| td3 / HalfCheetah-v5 / bs100_h1024x1024 / UTD4 | 0.02837 | 0.02982 | +5.11% | 9.8413e+14 | 9.8284e+14 | -0.13% | 1.0378e-10 | 1.0922e-10 | +5.24% |
| td3 / HalfCheetah-v5 / bs100_h256x256 / UTD1 | 0.00611 | 0.00640 | +4.74% | 1.6230e+13 | 1.6209e+13 | -0.13% | 1.3551e-09 | 1.4213e-09 | +4.88% |
| td3 / HalfCheetah-v5 / bs100_h512x512 / UTD1 | 0.00674 | 0.00705 | +4.62% | 6.2673e+13 | 6.2590e+13 | -0.13% | 3.8693e-10 | 4.0533e-10 | +4.76% |
| td3 / HalfCheetah-v5 / bs1024_h1024x1024 / UTD1 | 0.01241 | 0.01335 | +7.63% | 2.5161e+15 | 2.5157e+15 | -0.01% | 1.7751e-11 | 1.9107e-11 | +7.64% |
| td3 / HalfCheetah-v5 / bs256_h1024x1024 / UTD1 | 0.00882 | 0.00929 | +5.32% | 6.2942e+14 | 6.2910e+14 | -0.05% | 5.0466e-11 | 5.3177e-11 | +5.37% |
| td3 / HalfCheetah-v5 / bs512_h1024x1024 / UTD1 | 0.01090 | 0.01158 | +6.19% | 1.2583e+15 | 1.2580e+15 | -0.03% | 3.1194e-11 | 3.3135e-11 | +6.22% |
| tdmpc2 / Ant-v5 / bs256_h1_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue / UTD1 | 0.08176 | 0.08423 | +3.02% | 5.0348e+15 | 5.0342e+15 | -0.01% | 5.8463e-11 | 6.0234e-11 | +3.03% |
| tdmpc2 / Ant-v5 / bs256_h3_ns512_it6_pit24_ne64_nq3_enc2x256_mlp512_lat512_epTrue / UTD1 | 0.11114 | 0.11427 | +2.81% | 8.1164e+15 | 8.1161e+15 | -0.00% | 4.9296e-11 | 5.0684e-11 | +2.82% |
| tdmpc2 / Ant-v5 / bs256_h3_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue / UTD1 | 0.13366 | 0.13700 | +2.50% | 1.0077e+16 | 1.0076e+16 | -0.01% | 4.7749e-11 | 4.8945e-11 | +2.51% |
| tdmpc2 / Ant-v5 / bs256_h3_ns512_it6_pit24_ne64_nq7_enc2x256_mlp512_lat512_epTrue / UTD1 | 0.15692 | 0.16026 | +2.13% | 1.2037e+16 | 1.2037e+16 | -0.01% | 4.6929e-11 | 4.7931e-11 | +2.13% |
| tdmpc2 / Ant-v5 / bs256_h5_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue / UTD1 | 0.17706 | 0.18063 | +2.01% | 1.5119e+16 | 1.5118e+16 | -0.00% | 4.2161e-11 | 4.3012e-11 | +2.02% |
| tdmpc2 / HalfCheetah-v5 / bs256_h1_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epFalse / UTD1 | 0.07575 | 0.07797 | +2.93% | 4.6151e+15 | 4.6145e+15 | -0.01% | 5.9087e-11 | 6.0827e-11 | +2.94% |
| tdmpc2 / HalfCheetah-v5 / bs256_h3_ns512_it6_pit24_ne64_nq3_enc2x256_mlp512_lat512_epFalse / UTD1 | 0.09707 | 0.09966 | +2.68% | 6.8762e+15 | 6.8759e+15 | -0.01% | 5.0818e-11 | 5.2180e-11 | +2.68% |
| tdmpc2 / HalfCheetah-v5 / bs256_h3_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epFalse / UTD1 | 0.12145 | 0.12419 | +2.26% | 8.8333e+15 | 8.8327e+15 | -0.01% | 4.9495e-11 | 5.0617e-11 | +2.27% |
| tdmpc2 / HalfCheetah-v5 / bs256_h3_ns512_it6_pit24_ne64_nq7_enc2x256_mlp512_lat512_epFalse / UTD1 | 0.14337 | 0.14627 | +2.02% | 1.0790e+16 | 1.0789e+16 | -0.01% | 4.7834e-11 | 4.8803e-11 | +2.03% |
| tdmpc2 / HalfCheetah-v5 / bs256_h5_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epFalse / UTD1 | 0.15580 | 0.15869 | +1.85% | 1.3051e+16 | 1.3051e+16 | -0.00% | 4.2975e-11 | 4.3773e-11 | +1.86% |

## MBPO dynamics_model_update, per group (old -> new)

| group | FLOPs old | FLOPs new | dF | J/FLOP old | J/FLOP new | dJ/FLOP | holdout share |
|---|---|---|---|---|---|---|---|
| mbpo / Ant-v5 / bs1024_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 4.0527e+14 | 4.3918e+14 | +8.37% | 4.3965e-10 | 4.0570e-10 | -7.72% | 7.99% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-15_re20-100 | 4.0317e+14 | 4.3686e+14 | +8.36% | 4.4989e-10 | 4.1520e-10 | -7.71% | 7.99% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-100 | 3.9524e+14 | 4.2826e+14 | +8.35% | 4.4643e-10 | 4.1201e-10 | -7.71% | 7.99% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 3.9771e+14 | 4.3095e+14 | +8.36% | 4.7362e-10 | 4.3709e-10 | -7.71% | 7.99% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD2 / rl1-25_re20-100 | 4.2112e+14 | 4.5638e+14 | +8.37% | 4.5860e-10 | 4.2317e-10 | -7.73% | 7.99% |
| mbpo / Ant-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD4 / rl1-25_re20-100 | 4.0479e+14 | 4.3871e+14 | +8.38% | 4.5382e-10 | 4.1873e-10 | -7.73% | 7.99% |
| mbpo / Ant-v5 / bs256_h256x256_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 4.0990e+14 | 4.4421e+14 | +8.37% | 4.5143e-10 | 4.1657e-10 | -7.72% | 7.99% |
| mbpo / Ant-v5 / bs256_h512x512_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 3.9859e+14 | 4.3194e+14 | +8.37% | 4.5053e-10 | 4.1575e-10 | -7.72% | 7.99% |
| mbpo / Ant-v5 / bs512_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-25_re20-100 | 3.8303e+14 | 4.1504e+14 | +8.36% | 4.4597e-10 | 4.1157e-10 | -7.71% | 7.99% |
| mbpo / HalfCheetah-v5 / bs1024_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 2.2355e+14 | 2.4165e+14 | +8.10% | 6.1806e-10 | 5.7175e-10 | -7.49% | 7.78% |
| mbpo / HalfCheetah-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 2.3292e+14 | 2.5180e+14 | +8.11% | 6.2766e-10 | 5.8058e-10 | -7.50% | 7.78% |
| mbpo / HalfCheetah-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD2 / rl1-1_re20-150 | 2.4246e+14 | 2.6217e+14 | +8.13% | 6.1883e-10 | 5.7231e-10 | -7.52% | 7.78% |
| mbpo / HalfCheetah-v5 / bs256_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD4 / rl1-1_re20-150 | 2.2775e+14 | 2.4623e+14 | +8.12% | 6.1172e-10 | 5.6580e-10 | -7.51% | 7.78% |
| mbpo / HalfCheetah-v5 / bs256_h256x256_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 2.2809e+14 | 2.4660e+14 | +8.12% | 6.1817e-10 | 5.7176e-10 | -7.51% | 7.78% |
| mbpo / HalfCheetah-v5 / bs256_h512x512_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 2.3226e+14 | 2.5114e+14 | +8.13% | 6.1817e-10 | 5.7170e-10 | -7.52% | 7.78% |
| mbpo / HalfCheetah-v5 / bs512_h1024x1024_ens7_mh200x200x200x200_mb256 / UTD1 / rl1-1_re20-150 | 2.2385e+14 | 2.4202e+14 | +8.11% | 6.1327e-10 | 5.6724e-10 | -7.51% | 7.78% |

## verify_mbpo_fit_flops.py output

```
MBPOConfig defaults: HalfCheetah-v5, obs_dim=17, act_dim=6, ensemble=7, model_hidden=(200, 200, 200, 200), batch=256, holdout_ratio=0.2
  linearity: member fwd+bwd bs256=200089600, bs100=78160000, bs256*100/256=78160000.0 -> exact
  flops_per_call.json[bs256_h1024x1024_ens7_mh200x200x200x200_mb256]: member_fwdbwd=200089600, forward_all_bs1=1845200 (matches fresh measurement)
  N=  5000: train_epochs=3 optimizer_steps=48 measured=71190000000 predicted=71190000000 (train=65654400000, holdout=5535600000, holdout share=7.78%) rel diff=0.000e+00 -> PASS
  N= 12345: train_epochs=3 optimizer_steps=117 measured=175768110000 predicted=175768110000 (train=162100713600, holdout=13667396400, holdout share=7.78%) rel diff=0.000e+00 -> PASS
  N= 30000: train_epochs=3 optimizer_steps=282 measured=427140000000 predicted=427140000000 (train=393926400000, holdout=33213600000, holdout share=7.78%) rel diff=0.000e+00 -> PASS

Ant override: Ant-v5, obs_dim=105, act_dim=8, ensemble=7, model_hidden=(200, 200, 200, 200), batch=256, holdout_ratio=0.2
  linearity: member fwd+bwd bs256=272588800, bs100=106480000, bs256*100/256=106480000.0 -> exact
  flops_per_call.json[bs256_h1024x1024_ens7_mh200x200x200x200_mb256]: member_fwdbwd=272588800, forward_all_bs1=2590000 (matches fresh measurement)
  N=  5000: train_epochs=3 optimizer_steps=48 measured=97213200000 predicted=97213200000 (train=89443200000, holdout=7770000000, holdout share=7.99%) rel diff=0.000e+00 -> PASS
  N= 12345: train_epochs=3 optimizer_steps=117 measured=240019390800 predicted=240019390800 (train=220835260800, holdout=19184130000, holdout share=7.99%) rel diff=0.000e+00 -> PASS
  N= 30000: train_epochs=3 optimizer_steps=282 measured=583279200000 predicted=583279200000 (train=536659200000, holdout=46620000000, holdout share=7.99%) rel diff=0.000e+00 -> PASS

ALL PASSED (acceptance: rel diff < 0.5% for every N)
```

## check_totals.py output

```
Runs under results/ with segment_energy.json: 300
Check 1 (TOTAL energy == sum of training segments, rtol 1e-12): 300 runs, max rel err 0.000e+00
Check 2 (TOTAL flops == sum of matmul rows, exact): 300 runs
Check 3 (sub-segments == measured gradient_updates, rtol 1e-09): 300 runs, max rel err 2.164e-16

ALL CHECKS PASSED
```

## Observations (out of scope, not changed)

- **Sub-segment timer coverage.** `sum(_sub_segment_wall_time_seconds) / measured
  gradient_updates task duration` over the canonical seeds:
  - mbpo: min 0.964, mean 0.971, max 0.977
  - sac: min 0.963, mean 0.971, max 0.978
  - td3: min 0.866, mean 0.929, max 0.963
  - tdmpc2: min 0.976, mean 0.982, max 0.987
  The `perf_counter` sub-timers cover only this fraction of the measured task. That's why
  the TOTAL row's power fields now use the task duration: summing sub-segment durations
  would inflate mean power by 1/coverage. The allocated sub-segment rows still carry their
  `perf_counter` durations and the power derived from them (unchanged by this fix).
- **`.item()` sync placement differs across algorithms.** SAC/TD3 call
  `q1_loss.item()`/`actor_loss.item()` just *after* their timers stop
  (`algorithms/sac.py:179,212`, `algorithms/td3.py:167,186`). TD-MPC2's critic `.item()`s are
  also outside the timer (`algorithms/tdmpc2.py:630-633`), but `_update_pi()` returns
  `pi_loss.item()` (`tdmpc2.py:559`), so the actor timer (`tdmpc2.py:636-638`) *includes* a
  device sync. Its GPU drain then shows up in `actor_update` time rather than in whichever
  timer ran next. Together with the missing `torch.cuda.synchronize()`, this shifts
  time-share (and so allocated energy) between sub-segments differently per algorithm. It
  does not affect the TOTAL row, which now uses only the measured `gradient_updates` energy.
- MBPO `dynamics_model_update` is roughly an order of magnitude less energy-efficient per
  FLOP than the SAC critic/actor updates in the same runs (see the table above against
  `critic_update` in the cross-seed CSV). This is plausibly due to small 200-wide layers,
  a Python loop over 7 members per minibatch, and host-side numpy indexing per batch. It's
  worth keeping in mind when interpreting its J/FLOP; it is unaffected by this fix beyond
  the ~8% FLOP correction.
