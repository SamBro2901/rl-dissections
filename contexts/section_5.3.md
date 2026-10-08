# Section 5.3 — Robustness of the cross-algorithm comparison across sweeps — context

Data context for thesis Section 5.3. Facts, numbers and provenance only — no interpretation, no LaTeX. Canonical seeds [331, 958, 14577, 43611, 85062] only (dev runs ignored); 280 runs. All energies are **gross** (idle floor included, nothing subtracted), joules, training total (`TOTAL_MEASURED_TRAINING`). J/FLOP is the pipeline's training-total J/FLOP, shown as pJ/FLOP (1 pJ = 1e-12 J). Four significant digits here, full precision in the CSVs. Returns are not used anywhere; Kendall's tau is not computed. HalfCheetah-v5 (HC) and Ant-v5 (Ant) are always reported separately.

Generated 2026-10-08 from git HEAD `2b4882f` (branch master); Windows dev checkout with synced `results/`.

## 0. Methods note

**Scripts** (both new, under `flop_analysis/`, read-only on `results/` and `flop_analysis/output/`): `flop_analysis/section_5_3_analysis.py` (computes everything, writes `contexts/section_5.3_data/*.csv`; it imports `load()` — the canonical-seed run table — from `flop_analysis/section_5_2_analysis.py`) and `flop_analysis/section_5_3_render.py` (renders this file). Re-run from the repo root: `.venv/Scripts/python -I -W ignore flop_analysis/section_5_3_analysis.py` then `.venv/Scripts/python flop_analysis/section_5_3_render.py`. Input: `flop_analysis/output/per_run_energy_per_flop.csv`.

**Definitions**
- *Per-seed values*: energy E = the run's `TOTAL_MEASURED_TRAINING` energy; J/FLOP = E / F of the same run (F = its total matmul FLOPs).
- *Configuration value* (the mean in every table): energy = mean of the 5 per-seed E, sd = sample sd (ddof = 1) over the 5 seeds. J/FLOP = mean E / mean F (ratio of means, the 5.2 convention); sd = sample sd of the 5 per-seed ratios. MBPO's F differs slightly between seeds; SAC, TD3 and TD-MPC2 have identical F across seeds.
- *Ordering* = SAC, TD3, MBPO sorted lowest → highest on the configuration value. *Ratio to previous* = value of this algorithm / value of the next-lower algorithm in the same ordering. TD-MPC2 is in no ordering.
- *Swap* relative to the baseline: a pair of algorithms whose relative order differs from the baseline ordering (baseline ordering of the same environment and metric).
- *Seed-level check*: paired seeds (the same seed number for all three algorithms). Per seed the three algorithms are sorted on that seed's own value of the metric; the count is the number of seeds (of 5) whose full ordering equals the ordering of the configuration means (and, additionally, the baseline ordering). Pairwise counts (seeds in which the mean-lower algorithm is below the mean-higher one) are in `o3b_pairwise_seed_counts.csv`.
- *Runs entering each ordering row* (3 algorithms × 5 seeds = 15 runs per environment per row): baseline = `base` of SAC, TD3, MBPO (SAC/MBPO batch 256; **TD3 batch 100, warmup 10000**; MBPO HC rollout length fixed 1, Ant 1→25). UTD 2 / UTD 4 = `utd2` / `utd4` of each (TD3 stays batch 100, MBPO keeps its baseline rollout schedule). Width 256 / 512 = `w256` / `w512` of each (TD3 batch 100). Batch 256 = SAC `base` and MBPO `base` (their baseline batch is already 256, so those runs are the same as in the baseline row) with TD3 `b256` (its batch-sweep run, warmup 10000 as all TD3 runs); batch 512 / 1024 = `b512` / `b1024` of all three. The TD3 batch-100 baseline is listed as a reference row beneath the batch tables.
- *Rollout share* (TD-MPC2): per-seed `rollout` energy / per-seed total energy; table gives the seed-mean and sample sd of that share (same convention as the Chapter 4.4 `share_of_total_pct`). The ratio of means is in the CSV.
- Configuration tags as in Sections 5.2/4.x: `base`, `utdK`, `wN`, `bN`, `rolloutN` (MBPO max model-rollout length, Ant only; baseline L_max = 25, `rollout_min_length` = 1 throughout), `numqN`, `horizonN`.

## 1. Orderings per sweep value (SAC, TD3, MBPO)

Each table: algorithms lowest → highest. `mean` = seed-mean, `sd` = sample sd over seeds, `×prev` = ratio to the next-lower algorithm. Source: `o1_orderings.csv`.

### 1.1 HalfCheetah-v5

**Baseline (UTD 1, width 1024)**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`base`) | 28110 | 310.7 | — | SAC (`base`) | 69.45 | 0.2897 | — |
| 2 | SAC (`base`) | 60910 | 254.1 | 2.167 | TD3 (`base`) | 114.3 | 1.264 | 1.646 |
| 3 | MBPO (`base`) | 207400 | 7451 | 3.405 | MBPO (`base`) | 183.1 | 5.442 | 1.601 |

TD-MPC2 baseline (`base`; **not part of the ordering**): energy 447100 J (sd 3184); J/FLOP 50.62 pJ/FLOP (sd 0.3605).

**UTD 2**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`utd2`) | 55360 | 397.5 | — | SAC (`utd2`) | 67.29 | 0.3556 | — |
| 2 | SAC (`utd2`) | 118000 | 623.7 | 2.132 | TD3 (`utd2`) | 112.6 | 0.8087 | 1.674 |
| 3 | MBPO (`utd2`) | 268900 | 6929 | 2.278 | MBPO (`utd2`) | 133.1 | 2.498 | 1.182 |

**UTD 4**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`utd4`) | 107300 | 1045 | — | SAC (`utd4`) | 66.98 | 0.3155 | — |
| 2 | SAC (`utd4`) | 235000 | 1107 | 2.189 | MBPO (`utd4`) | 99.42 | 1.825 | 1.484 |
| 3 | MBPO (`utd4`) | 373600 | 8545 | 1.59 | TD3 (`utd4`) | 109.2 | 1.063 | 1.099 |

**Width 256**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`w256`) | 23040 | 618.5 | — | MBPO (`w256`) | 612.6 | 9.023 | — |
| 2 | SAC (`w256`) | 47410 | 458.5 | 2.058 | SAC (`w256`) | 815 | 7.882 | 1.33 |
| 3 | MBPO (`w256`) | 187900 | 4850 | 3.964 | TD3 (`w256`) | 1421 | 38.16 | 1.744 |

**Width 512**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`w512`) | 25370 | 399.1 | — | SAC (`w512`) | 221.5 | 2.282 | — |
| 2 | SAC (`w512`) | 49550 | 510.5 | 1.953 | MBPO (`w512`) | 403.9 | 8.707 | 1.824 |
| 3 | MBPO (`w512`) | 192800 | 8814 | 3.89 | TD3 (`w512`) | 405.3 | 6.376 | 1.004 |

**Batch 256**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`b256`) | 33450 | 435.4 | — | TD3 (`b256`) | 53.18 | 0.6921 | — |
| 2 | SAC (`base`) | 60910 | 254.1 | 1.821 | SAC (`base`) | 69.45 | 0.2897 | 1.306 |
| 3 | MBPO (`base`) | 207400 | 7451 | 3.405 | MBPO (`base`) | 183.1 | 5.442 | 2.636 |

**Batch 512**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`b512`) | 41680 | 778.1 | — | TD3 (`b512`) | 33.14 | 0.6185 | — |
| 2 | SAC (`b512`) | 73020 | 517.6 | 1.752 | SAC (`b512`) | 41.63 | 0.2951 | 1.256 |
| 3 | MBPO (`b512`) | 209100 | 4906 | 2.863 | MBPO (`b512`) | 104.6 | 2.077 | 2.511 |

**Batch 1024**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`b1024`) | 48070 | 808.1 | — | TD3 (`b1024`) | 19.11 | 0.3212 | — |
| 2 | SAC (`b1024`) | 82370 | 589.8 | 1.714 | SAC (`b1024`) | 23.48 | 0.1681 | 1.229 |
| 3 | MBPO (`b1024`) | 218700 | 5010 | 2.655 | MBPO (`b1024`) | 58.26 | 1.199 | 2.481 |

Reference row for the batch tables — TD3 batch-100 baseline (`base`, warmup 10000; this is TD3's entry in the baseline ordering above): energy 28110 J (sd 310.7); J/FLOP 114.3 pJ/FLOP (sd 1.264).

### 1.2 Ant-v5

**Baseline (UTD 1, width 1024)**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`base`) | 30520 | 431.9 | — | SAC (`base`) | 68.57 | 0.6265 | — |
| 2 | SAC (`base`) | 64380 | 588.3 | 2.11 | TD3 (`base`) | 116.3 | 1.645 | 1.696 |
| 3 | MBPO (`base`) | 253100 | 20140 | 3.932 | MBPO (`base`) | 178.8 | 8.471 | 1.538 |

TD-MPC2 baseline (`base`; **not part of the ordering**): energy 493200 J (sd 1782); J/FLOP 48.95 pJ/FLOP (sd 0.1768).

**UTD 2**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`utd2`) | 57540 | 361 | — | SAC (`utd2`) | 65.6 | 0.387 | — |
| 2 | SAC (`utd2`) | 123200 | 726.7 | 2.141 | TD3 (`utd2`) | 109.7 | 0.688 | 1.671 |
| 3 | MBPO (`utd2`) | 316800 | 17140 | 2.572 | MBPO (`utd2`) | 133.1 | 5.193 | 1.214 |

**UTD 4**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`utd4`) | 111900 | 1590 | — | SAC (`utd4`) | 64.26 | 0.2838 | — |
| 2 | SAC (`utd4`) | 241300 | 1066 | 2.157 | MBPO (`utd4`) | 100.2 | 3.003 | 1.558 |
| 3 | MBPO (`utd4`) | 424500 | 16390 | 1.759 | TD3 (`utd4`) | 106.6 | 1.516 | 1.065 |

**Width 256**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`w256`) | 25100 | 811.3 | — | MBPO (`w256`) | 431.4 | 2.818 | — |
| 2 | SAC (`w256`) | 49990 | 725.5 | 1.991 | SAC (`w256`) | 678.8 | 9.852 | 1.574 |
| 3 | MBPO (`w256`) | 234500 | 9869 | 4.691 | TD3 (`w256`) | 1232 | 39.84 | 1.816 |

**Width 512**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`w512`) | 27470 | 264.6 | — | SAC (`w512`) | 205 | 1.882 | — |
| 2 | SAC (`w512`) | 52200 | 479.3 | 1.9 | MBPO (`w512`) | 323 | 8.561 | 1.576 |
| 3 | MBPO (`w512`) | 231600 | 18560 | 4.438 | TD3 (`w512`) | 387.5 | 3.731 | 1.199 |

**Batch 256**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`b256`) | 35520 | 357.7 | — | TD3 (`b256`) | 52.89 | 0.5326 | — |
| 2 | SAC (`base`) | 64380 | 588.3 | 1.813 | SAC (`base`) | 68.57 | 0.6265 | 1.296 |
| 3 | MBPO (`base`) | 253100 | 20140 | 3.932 | MBPO (`base`) | 178.8 | 8.471 | 2.607 |

**Batch 512**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`b512`) | 44360 | 591.8 | — | TD3 (`b512`) | 33.03 | 0.4406 | — |
| 2 | SAC (`b512`) | 76500 | 399.8 | 1.724 | SAC (`b512`) | 40.74 | 0.2129 | 1.233 |
| 3 | MBPO (`b512`) | 247400 | 21200 | 3.234 | MBPO (`b512`) | 105.8 | 6.709 | 2.596 |

**Batch 1024**

| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |
|---|---|---|---|---|---|---|---|---|
| 1 | TD3 (`b1024`) | 53230 | 707.8 | — | TD3 (`b1024`) | 19.82 | 0.2635 | — |
| 2 | SAC (`b1024`) | 89060 | 410.8 | 1.673 | SAC (`b1024`) | 23.72 | 0.1094 | 1.197 |
| 3 | MBPO (`b1024`) | 267700 | 15140 | 3.006 | MBPO (`b1024`) | 63.14 | 3.059 | 2.662 |

Reference row for the batch tables — TD3 batch-100 baseline (`base`, warmup 10000; this is TD3's entry in the baseline ordering above): energy 30520 J (sd 431.9); J/FLOP 116.3 pJ/FLOP (sd 1.645).

## 2. Pairs that swap order relative to the baseline ordering

Source: `o2_swaps_vs_baseline.csv`. `a<b -> b<a` means: a is below b in the baseline ordering and above b in this sweep value.

| env | sweep value | energy ordering | energy swaps | J/FLOP ordering | J/FLOP swaps |
|---|---|---|---|---|---|
| HalfCheetah-v5 | Baseline (UTD 1, width 1024) | TD3 < SAC < MBPO | none | SAC < TD3 < MBPO | none |
| HalfCheetah-v5 | UTD 2 | TD3 < SAC < MBPO | none | SAC < TD3 < MBPO | none |
| HalfCheetah-v5 | UTD 4 | TD3 < SAC < MBPO | none | SAC < MBPO < TD3 | TD3<MBPO -> MBPO<TD3 |
| HalfCheetah-v5 | Width 256 | TD3 < SAC < MBPO | none | MBPO < SAC < TD3 | SAC<MBPO -> MBPO<SAC; TD3<MBPO -> MBPO<TD3 |
| HalfCheetah-v5 | Width 512 | TD3 < SAC < MBPO | none | SAC < MBPO < TD3 | TD3<MBPO -> MBPO<TD3 |
| HalfCheetah-v5 | Batch 256 | TD3 < SAC < MBPO | none | TD3 < SAC < MBPO | SAC<TD3 -> TD3<SAC |
| HalfCheetah-v5 | Batch 512 | TD3 < SAC < MBPO | none | TD3 < SAC < MBPO | SAC<TD3 -> TD3<SAC |
| HalfCheetah-v5 | Batch 1024 | TD3 < SAC < MBPO | none | TD3 < SAC < MBPO | SAC<TD3 -> TD3<SAC |
| Ant-v5 | Baseline (UTD 1, width 1024) | TD3 < SAC < MBPO | none | SAC < TD3 < MBPO | none |
| Ant-v5 | UTD 2 | TD3 < SAC < MBPO | none | SAC < TD3 < MBPO | none |
| Ant-v5 | UTD 4 | TD3 < SAC < MBPO | none | SAC < MBPO < TD3 | TD3<MBPO -> MBPO<TD3 |
| Ant-v5 | Width 256 | TD3 < SAC < MBPO | none | MBPO < SAC < TD3 | SAC<MBPO -> MBPO<SAC; TD3<MBPO -> MBPO<TD3 |
| Ant-v5 | Width 512 | TD3 < SAC < MBPO | none | SAC < MBPO < TD3 | TD3<MBPO -> MBPO<TD3 |
| Ant-v5 | Batch 256 | TD3 < SAC < MBPO | none | TD3 < SAC < MBPO | SAC<TD3 -> TD3<SAC |
| Ant-v5 | Batch 512 | TD3 < SAC < MBPO | none | TD3 < SAC < MBPO | SAC<TD3 -> TD3<SAC |
| Ant-v5 | Batch 1024 | TD3 < SAC < MBPO | none | TD3 < SAC < MBPO | SAC<TD3 -> TD3<SAC |

- HalfCheetah-v5, energy: baseline ordering TD3 < SAC < MBPO; no pair swaps in any of the 7 sweep values.
- Ant-v5, energy: baseline ordering TD3 < SAC < MBPO; no pair swaps in any of the 7 sweep values.
- HalfCheetah-v5, J/FLOP: baseline ordering SAC < TD3 < MBPO; 6 of 7 sweep values differ from it (UTD 4, Width 256, Width 512, Batch 256, Batch 512, Batch 1024).
- Ant-v5, J/FLOP: baseline ordering SAC < TD3 < MBPO; 6 of 7 sweep values differ from it (UTD 4, Width 256, Width 512, Batch 256, Batch 512, Batch 1024).

## 3. Seed-level check (paired seeds)

`n_own` = seeds (of 5) whose full SAC/TD3/MBPO ordering equals the ordering of the configuration means in that row; `n_base` = seeds whose ordering equals the baseline mean ordering. Source: `o3_seed_level.csv` (per-seed orderings in columns `seed_*`), `o3b_pairwise_seed_counts.csv`.

### 3.1 HalfCheetah-v5

| sweep value | energy: mean ordering | n_own | n_base | J/FLOP: mean ordering | n_own | n_base |
|---|---|---|---|---|---|---|
| Baseline (UTD 1, width 1024) | TD3 < SAC < MBPO | 5/5 | 5/5 | SAC < TD3 < MBPO | 5/5 | 5/5 |
| UTD 2 | TD3 < SAC < MBPO | 5/5 | 5/5 | SAC < TD3 < MBPO | 5/5 | 5/5 |
| UTD 4 | TD3 < SAC < MBPO | 5/5 | 5/5 | SAC < MBPO < TD3 | 5/5 | 0/5 |
| Width 256 | TD3 < SAC < MBPO | 5/5 | 5/5 | MBPO < SAC < TD3 | 5/5 | 0/5 |
| Width 512 | TD3 < SAC < MBPO | 5/5 | 5/5 | SAC < MBPO < TD3 | 3/5 | 2/5 |
| Batch 256 | TD3 < SAC < MBPO | 5/5 | 5/5 | TD3 < SAC < MBPO | 5/5 | 0/5 |
| Batch 512 | TD3 < SAC < MBPO | 5/5 | 5/5 | TD3 < SAC < MBPO | 5/5 | 0/5 |
| Batch 1024 | TD3 < SAC < MBPO | 5/5 | 5/5 | TD3 < SAC < MBPO | 5/5 | 0/5 |

Pairs that are **not** in the mean-ordering direction in all 5 seeds:

| sweep value | metric | pair (mean-lower < mean-higher) | seeds in that direction | per-seed ratio higher/lower, min – max |
|---|---|---|---|---|
| Width 512 | J/FLOP | MBPO < TD3 | 3/5 | 0.977 – 1.031 |

### 3.2 Ant-v5

| sweep value | energy: mean ordering | n_own | n_base | J/FLOP: mean ordering | n_own | n_base |
|---|---|---|---|---|---|---|
| Baseline (UTD 1, width 1024) | TD3 < SAC < MBPO | 5/5 | 5/5 | SAC < TD3 < MBPO | 5/5 | 5/5 |
| UTD 2 | TD3 < SAC < MBPO | 5/5 | 5/5 | SAC < TD3 < MBPO | 5/5 | 5/5 |
| UTD 4 | TD3 < SAC < MBPO | 5/5 | 5/5 | SAC < MBPO < TD3 | 5/5 | 0/5 |
| Width 256 | TD3 < SAC < MBPO | 5/5 | 5/5 | MBPO < SAC < TD3 | 5/5 | 0/5 |
| Width 512 | TD3 < SAC < MBPO | 5/5 | 5/5 | SAC < MBPO < TD3 | 5/5 | 0/5 |
| Batch 256 | TD3 < SAC < MBPO | 5/5 | 5/5 | TD3 < SAC < MBPO | 5/5 | 0/5 |
| Batch 512 | TD3 < SAC < MBPO | 5/5 | 5/5 | TD3 < SAC < MBPO | 5/5 | 0/5 |
| Batch 1024 | TD3 < SAC < MBPO | 5/5 | 5/5 | TD3 < SAC < MBPO | 5/5 | 0/5 |

Every pair is in the mean-ordering direction in all 5 seeds, for all sweep values and both metrics.

## 4. MBPO rollout-length sweep (Ant-v5 only; not an ordering sweep)

`L_max` = `rollout_max_length` (min length 1, schedule epochs 20–100 as in `configs/overrides/mbpo_ant*.json`); L_max = 25 is the Ant MBPO baseline. SAC / TD3 comparison values are their Ant baselines (SAC `base`, TD3 `base` = batch 100, warmup 10000). `×SAC`, `×TD3` = MBPO value / that value; `seeds >` = number of paired seeds (of 5) in which MBPO is above it. Source: `r1_mbpo_rollout_sweep.csv`.

Reference baselines (Ant): SAC energy 64380 J, J/FLOP 68.57 pJ/FLOP; TD3 energy 30520 J, J/FLOP 116.3 pJ/FLOP.

| L_max | config | MBPO energy mean ± sd [J] | ×SAC | ×TD3 | seeds > SAC / TD3 | energy ordering (SAC, TD3 baselines + this MBPO) | MBPO J/FLOP mean ± sd [pJ/FLOP] | ×SAC | ×TD3 | seeds > SAC / TD3 | J/FLOP ordering | MBPO F mean [matmul FLOPs] (min – max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `rollout1` | 240700 ± 3281 | 3.739 | 7.888 | 5/5 / 5/5 | TD3 < SAC < mbpo(L1) | 175.5 ± 1.642 | 2.559 | 1.509 | 5/5 / 5/5 | SAC < TD3 < mbpo(L1) | 1.372e+15 (1.355e+15 – 1.39e+15) |
| 15 | `rollout15` | 245900 ± 12660 | 3.819 | 8.056 | 5/5 / 5/5 | TD3 < SAC < mbpo(L15) | 174.9 ± 5.752 | 2.55 | 1.504 | 5/5 / 5/5 | SAC < TD3 < mbpo(L15) | 1.406e+15 (1.378e+15 – 1.435e+15) |
| 25 | `base` | 253100 ± 20140 | 3.932 | 8.294 | 5/5 / 5/5 | TD3 < SAC < mbpo(L25) | 178.8 ± 8.471 | 2.607 | 1.538 | 5/5 / 5/5 | SAC < TD3 < mbpo(L25) | 1.416e+15 (1.368e+15 – 1.469e+15) |

## 5. TD-MPC2 sweep (baseline, num_q ∈ {3,5,7}, horizon ∈ {1,3,5})

`base` = num_q 5, horizon 3 (shared by both sweeps). Rollout share = rollout energy / total energy (seed-mean ± sd of the per-seed share). Ant runs use `episodic: true`. Source: `t1_tdmpc2_sweep.csv`.

**HalfCheetah-v5**

| config | num_q | horizon | total energy mean ± sd [J] | total J/FLOP mean ± sd [pJ/FLOP] | rollout share [%] |
|---|---|---|---|---|---|
| `numq3` | 3 | 3 | 358800 ± 2182 | 52.18 ± 0.3174 | 50.72 ± 0.2962 |
| `horizon1` | 5 | 1 | 280700 ± 1782 | 60.83 ± 0.3863 | 48.16 ± 0.3071 |
| `base` | 5 | 3 | 447100 ± 3184 | 50.62 ± 0.3605 | 47.83 ± 0.3413 |
| `horizon5` | 5 | 5 | 571300 ± 4039 | 43.77 ± 0.3095 | 50.46 ± 0.3819 |
| `numq7` | 7 | 3 | 526600 ± 1974 | 48.8 ± 0.183 | 46 ± 0.2268 |

**Ant-v5**

| config | num_q | horizon | total energy mean ± sd [J] | total J/FLOP mean ± sd [pJ/FLOP] | rollout share [%] |
|---|---|---|---|---|---|
| `numq3` | 3 | 3 | 411400 ± 2042 | 50.68 ± 0.2517 | 54.22 ± 0.2805 |
| `horizon1` | 5 | 1 | 303200 ± 2582 | 60.23 ± 0.5128 | 49.29 ± 0.3119 |
| `base` | 5 | 3 | 493200 ± 1782 | 48.95 ± 0.1768 | 51.62 ± 0.2272 |
| `horizon5` | 5 | 5 | 650300 ± 4615 | 43.01 ± 0.3052 | 54.48 ± 0.3243 |
| `numq7` | 7 | 3 | 576900 ± 2824 | 47.93 ± 0.2346 | 48.69 ± 0.1874 |

### Range of the other three algorithms over all their configurations (context for TD-MPC2's sweep range)

Min / max of the configuration value (seed-mean energy; J/FLOP = ratio of means) over *all* configurations of the algorithm in that environment, including every sweep (SAC, TD3: 7 and 8 configurations; MBPO: 7 on HC, 9 on Ant; TD-MPC2: 5). `overlap` = the [min, max] interval intersects TD-MPC2's [min, max] over its 5 configurations. Source: `t3_algo_ranges_vs_tdmpc2.csv`, all configuration values in `t2_all_config_values.csv`.

**HalfCheetah-v5**

| algorithm | #cfg | energy min [J] (config) | energy max [J] (config) | overlap TD-MPC2 (energy) | J/FLOP min [pJ/FLOP] (config) | J/FLOP max [pJ/FLOP] (config) | overlap TD-MPC2 (J/FLOP) |
|---|---|---|---|---|---|---|---|
| SAC | 7 | 47410 (`w256`) | 235000 (`utd4`) | no | 23.48 (`b1024`) | 815 (`w256`) | yes |
| TD3 | 8 | 23040 (`w256`) | 107300 (`utd4`) | no | 19.11 (`b1024`) | 1421 (`w256`) | yes |
| MBPO | 7 | 187900 (`w256`) | 373600 (`utd4`) | yes | 58.26 (`b1024`) | 612.6 (`w256`) | yes |
| TD-MPC2 (reference) | 5 | 280700 (`horizon1`) | 571300 (`horizon5`) | — | 43.77 (`horizon5`) | 60.83 (`horizon1`) | — |

**Ant-v5**

| algorithm | #cfg | energy min [J] (config) | energy max [J] (config) | overlap TD-MPC2 (energy) | J/FLOP min [pJ/FLOP] (config) | J/FLOP max [pJ/FLOP] (config) | overlap TD-MPC2 (J/FLOP) |
|---|---|---|---|---|---|---|---|
| SAC | 7 | 49990 (`w256`) | 241300 (`utd4`) | no | 23.72 (`b1024`) | 678.8 (`w256`) | yes |
| TD3 | 8 | 25100 (`w256`) | 111900 (`utd4`) | no | 19.82 (`b1024`) | 1232 (`w256`) | yes |
| MBPO | 9 | 231600 (`w512`) | 424500 (`utd4`) | yes | 63.14 (`b1024`) | 431.4 (`w256`) | no |
| TD-MPC2 (reference) | 5 | 303200 (`horizon1`) | 650300 (`horizon5`) | — | 43.01 (`horizon5`) | 60.23 (`horizon1`) | — |

## 6. Consistency with Chapter 4 and items not computed

**Consistency with Chapter 4 — none found.** All 56 configuration totals (energy, J/FLOP) used here equal the 5.2 configuration table (itself checked against the Chapter 4.1–4.4 `*_cross_seed_summary.csv` files in 5.2) to a maximum relative deviation of 1.5e-16 (energy) and 1.9e-16 (J/FLOP). The 10 TD-MPC2 rollout energies and rollout shares equal `contexts/Section_4.4_data/tdmpc2_cross_seed_summary.csv` (segment `rollout`: `energy_j_mean`, `share_of_total_pct_mean`) to 2.6e-16 / 2.9e-16. `x1_consistency_checks.csv`.

**Notes / caveats (no numbers are inconsistent):**
- The batch-256 row uses SAC `base` and MBPO `base` (batch 256 is their baseline); only TD3 contributes a separate run set (`b256`). Rows `baseline` and `b256` therefore differ only in TD3 (batch 100 vs 256).
- The baseline orderings compare TD3 at batch 100 / warmup 10000 with SAC and MBPO at batch 256; the batch rows hold batch size equal across the three algorithms. Warmup is not equalised: TD3 uses 10000 in all its runs, MBPO 5000 (Section 4.3); warmup energy is excluded from the training total.
- MBPO's HC baseline has a fixed model-rollout length 1, Ant baseline a 1→25 schedule; the environment baselines are not the same MBPO configuration (Section 4.3 convention).
- Chapter 4 does not publish SAC/TD3/MBPO orderings, so there is no Chapter 4 ordering to compare with; only the underlying means, sds, J/FLOP and TD-MPC2 shares were cross-checked.
- Ratios between neighbours are ratios of configuration means (energy) or of ratio-of-means J/FLOP; the per-seed ratio range for each pair is in `o3b_pairwise_seed_counts.csv`.
- Not computed: no significance tests (n = 5 seeds; counts only); no returns; no Kendall's tau; TD-MPC2 is not placed in any ordering; the MBPO rollout sweep and TD-MPC2 sweep are not compared to SAC/TD3 in any ordering across other sweep values (only baselines, as specified).
- Environment: Windows dev checkout of the data, not the Linux experiment box; software versions as in `metadata.json` are unchanged by this analysis.
