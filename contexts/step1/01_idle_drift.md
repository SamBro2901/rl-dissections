# Task 1: Idle-baseline drift

## Question

Which numbers describe head-vs-tail idle drift under (a) *as recorded* and (b) *window-corrected*, and how exactly is "window-corrected" defined? Basis: the 280 canonical runs.

## Method

- Script: `contexts/step1/scripts/idle_drift.py` (run with the repo `.venv` Python; uses `contexts/step1/scripts/common.py` for the canonical run list, 280 rows, grid verified in Task 4).
- Data: rows `idle_baseline_head` / `idle_baseline_tail` of each run's `emissions_base_*.csv` (columns `duration`, `*_energy`, `ram_power`, `timestamp`), 280 runs. Full-precision per-run output: `01_idle_drift_per_run.csv`.
- Drift = (P_tail − P_head) / P_head, signed, in percent. Absolute drift in W is also in the CSV (`drift_*_w_*`).

### Definition found in the repo (verbatim)

`idle_power_analysis.py` L96-101 (code comment + code):
```
# CodeCarbon models RAM at constant power, so RAM energy / RAM
# power is the time span the task's energy was actually integrated
# over. For the head task this comes out ~3 s longer than its
# reported duration ...
eff_window = float(row["ram_energy"]) * J_PER_KWH / float(row["ram_power"])
```
`idle_power_analysis.py` L106-108: `rec[f"{phase}_{comp}_w"] = energy_j / dur` (as recorded) and `rec[f"{phase}_{comp}_w_corr"] = energy_j / eff_window` (window-corrected).
`idle_power_analysis.py` L190-191: "*Window-corrected* divides by the integrated span instead (assumes power during the extra ~3 s matches the rest of the window)."
`README.md` L646-649: "The analysis reports both *as-recorded* (÷ reported duration) and *window-corrected* (÷ integrated span; assumes power during the extra seconds matches the rest) values; the `_corr` columns are the ones to use. The correction is **not** applied to `segment_energy.json`."
`CLAUDE.md` L506-508: "the CSV/report give both *as-recorded* and *window-corrected* (energy ÷ integrated span) values — use corrected."

So the definition **exists** in the repo and is used here unchanged: window-corrected power = component energy / (`ram_energy`/`ram_power`), per window. The "integration span" is a *proxy* (CodeCarbon's RAM power is a constant, so RAM energy/RAM power = integrated seconds); `emissions_base_*.csv` has no per-task start/stop timestamps (only one `timestamp` column, second resolution, at task stop), so the span cannot be cross-checked from timestamps there. In the example run `run.log` shows "Recording idle baseline for 90s..." at 13:30:13 and "Starting warmup" at 13:31:47 (≈94 s wall clock between the two log lines, consistent with a span of ≈93 s).

## Results

### 1. Durations of the idle windows (nominal 90 s)

| window | nominal (config) | recorded duration: min / median / max (s) | integration span: min / median / max (s) | span − duration: median (s) |
|---|---|---|---|---|
| head | 90 | 90.0025 / 90.0032 / 90.0062 | 93.046 / 93.080 / 94.003 | 3.076 |
| tail | 90 | 90.0029 / 90.0035 / 90.0085 | 90.002 / 90.002 / 90.007 | -0.001 |

The recorded durations confirm 90 s; the head window integrates ≈3.1 s more than that, the tail does not.

### 2. Drift summary, all 280 runs (percent of head power)

| statistic | component | median | mean | min | max | p5 | p95 | median abs | frac tail>head |
|---|---|---|---|---|---|---|---|---|---|
| (a) as recorded | total | 6.34 | 6.69 | -13.76 | 23.19 | -6.13 | 16.36 | 7.04 | 0.914 |
| (a) as recorded | cpu | -4.94 | -4.94 | -16.97 | 23.90 | -9.26 | -0.56 | 4.99 | 0.036 |
| (a) as recorded | gpu | 20.15 | 21.34 | -22.11 | 62.05 | -9.67 | 42.73 | 20.19 | 0.929 |
| (a) as recorded | ram | -3.31 | -3.31 | -4.26 | -3.27 | -3.39 | -3.28 | 3.31 | 0.000 |
| (b) window-corrected | total | 9.98 | 10.34 | -10.81 | 27.39 | -2.96 | 20.32 | 10.02 | 0.932 |
| (b) window-corrected | cpu | -1.70 | -1.69 | -14.06 | 28.14 | -6.14 | 2.81 | 2.30 | 0.250 |
| (b) window-corrected | gpu | 24.23 | 25.49 | -19.45 | 67.58 | -6.60 | 47.62 | 24.23 | 0.936 |
| (b) window-corrected | ram | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |

Same in W (tail − head), total only:

| statistic | median | mean | min | max | p5 | p95 | median abs |
|---|---|---|---|---|---|---|---|
| (a) | 3.99 | 4.24 | -10.43 | 14.74 | -4.58 | 10.94 | 4.48 |
| (b) | 6.14 | 6.40 | -7.93 | 16.84 | -2.14 | 13.13 | 6.23 |

Note on RAM (a): CodeCarbon's RAM power is constant, so any non-zero RAM drift under (a) is the 3 s window artefact (head energy is divided by 90 s but integrated over ≈93 s); under (b) RAM drift is ≈0 by construction.

### 3. Drift by algorithm × environment (total power, % of head)

| group | n | (a) median | (a) mean | (a) min | (a) max | (a) p5 | (a) p95 | (a) median abs | (a) frac tail>head | (b) median | (b) mean | (b) min | (b) max | (b) p5 | (b) p95 | (b) median abs | (b) frac tail>head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mbpo / Ant-v5 | 45 | 4.11 | 4.34 | 1.08 | 8.87 | 1.85 | 7.42 | 4.11 | 1.000 | 7.63 | 7.90 | 4.54 | 12.57 | 5.31 | 11.10 | 7.63 | 1.000 |
| mbpo / HalfCheetah-v5 | 35 | 4.14 | 4.10 | -2.40 | 8.99 | 1.02 | 6.83 | 4.14 | 0.943 | 7.68 | 7.67 | 1.05 | 12.72 | 4.46 | 10.46 | 7.68 | 1.000 |
| sac / Ant-v5 | 35 | 5.15 | 6.13 | -13.76 | 23.19 | -10.72 | 22.06 | 8.48 | 0.829 | 8.74 | 9.76 | -10.81 | 27.39 | -7.67 | 26.24 | 9.16 | 0.829 |
| sac / HalfCheetah-v5 | 35 | 6.33 | 5.58 | -12.06 | 22.44 | -11.49 | 20.57 | 8.30 | 0.829 | 9.95 | 9.23 | -8.98 | 26.62 | -8.46 | 24.68 | 9.95 | 0.829 |
| td3 / Ant-v5 | 40 | 9.60 | 8.41 | -7.07 | 20.48 | -3.13 | 15.41 | 9.60 | 0.900 | 13.33 | 12.11 | -3.90 | 24.57 | 0.16 | 19.36 | 13.33 | 0.950 |
| td3 / HalfCheetah-v5 | 40 | 9.02 | 7.53 | -8.52 | 18.11 | -6.20 | 15.29 | 9.02 | 0.850 | 12.75 | 11.20 | -5.39 | 22.11 | -3.02 | 19.23 | 12.75 | 0.875 |
| tdmpc2 / Ant-v5 | 25 | 10.18 | 9.88 | 3.01 | 17.03 | 4.80 | 13.99 | 10.18 | 1.000 | 13.91 | 13.65 | 6.53 | 21.05 | 8.39 | 17.87 | 13.91 | 1.000 |
| tdmpc2 / HalfCheetah-v5 | 25 | 10.02 | 9.56 | 4.49 | 14.89 | 5.16 | 14.24 | 10.02 | 1.000 | 13.76 | 13.32 | 8.04 | 18.80 | 8.74 | 18.13 | 13.76 | 1.000 |

(Per-component tables for GPU/CPU/RAM for all groups: `01_idle_drift_summary.csv`, full precision.)

### 4. Top 5 runs by absolute drift (total power)

**(a) as recorded**

| run directory | head W | tail W | drift % |
|---|---|---|---|
| `results/sac/Ant-v5/seed_43611/20260912_223103` | 63.57 | 78.32 | 23.19 |
| `results/sac/Ant-v5/seed_43611/20260908_115230` | 63.25 | 77.61 | 22.71 |
| `results/sac/HalfCheetah-v5/seed_43611/20260912_204101` | 63.35 | 77.56 | 22.44 |
| `results/sac/Ant-v5/seed_85062/20260912_214655` | 63.57 | 77.42 | 21.78 |
| `results/sac/HalfCheetah-v5/seed_14577/20260912_193634` | 64.57 | 77.97 | 20.74 |

**(b) window-corrected**

| run directory | head W | tail W | drift % |
|---|---|---|---|
| `results/sac/Ant-v5/seed_43611/20260912_223103` | 61.48 | 78.32 | 27.39 |
| `results/sac/Ant-v5/seed_43611/20260908_115230` | 61.16 | 77.62 | 26.91 |
| `results/sac/HalfCheetah-v5/seed_43611/20260912_204101` | 61.26 | 77.56 | 26.62 |
| `results/sac/Ant-v5/seed_85062/20260912_214655` | 61.47 | 77.42 | 25.96 |
| `results/sac/HalfCheetah-v5/seed_14577/20260912_193634` | 62.44 | 77.97 | 24.87 |

### 5. Absolute head idle power by algorithm × environment (total W)

| group | n | (a) median | (a) range | (b) median | (b) range |
|---|---|---|---|---|---|
| mbpo / Ant-v5 | 45 | 62.67 | 60.51–64.61 | 60.60 | 58.53–62.49 |
| mbpo / HalfCheetah-v5 | 35 | 63.12 | 61.14–68.80 | 61.03 | 59.11–66.45 |
| sac / Ant-v5 | 35 | 67.12 | 62.11–75.83 | 64.80 | 60.07–73.32 |
| sac / HalfCheetah-v5 | 35 | 64.98 | 62.03–77.99 | 62.85 | 59.98–75.35 |
| td3 / Ant-v5 | 40 | 65.40 | 62.29–76.70 | 63.25 | 60.25–74.18 |
| td3 / HalfCheetah-v5 | 40 | 67.60 | 61.80–76.20 | 65.34 | 59.78–73.68 |
| tdmpc2 / Ant-v5 | 25 | 64.46 | 62.79–67.17 | 62.32 | 60.71–64.95 |
| tdmpc2 / HalfCheetah-v5 | 25 | 64.12 | 62.93–65.64 | 61.99 | 60.85–63.49 |

### 6. Head idle power by calendar date of run start (UTC), window-corrected total W

| date (UTC) | n runs | median | min | max | median pre-head GPU temp (°C) | algos/envs that day |
|---|---|---|---|---|---|---|
| 2026-09-07 | 8 | 64.41 | 63.16 | 66.03 | 45.5 | sac/Ant, sac/HalfCheetah |
| 2026-09-08 | 22 | 62.48 | 60.71 | 75.35 | 44.5 | mbpo/Ant, mbpo/HalfCheetah, sac/Ant, sac/HalfCheetah |
| 2026-09-09 | 40 | 65.62 | 62.15 | 74.18 | 44.0 | sac/Ant, td3/Ant, td3/HalfCheetah |
| 2026-09-10 | 10 | 62.72 | 60.89 | 64.95 | 43.0 | tdmpc2/Ant, tdmpc2/HalfCheetah |
| 2026-09-12 | 26 | 62.85 | 59.38 | 73.32 | 43.5 | mbpo/HalfCheetah, sac/Ant, sac/HalfCheetah |
| 2026-09-13 | 34 | 61.64 | 59.46 | 73.68 | 43.0 | mbpo/Ant, mbpo/HalfCheetah, td3/Ant, td3/HalfCheetah |
| 2026-09-14 | 17 | 61.37 | 60.14 | 66.45 | 42.0 | mbpo/Ant, mbpo/HalfCheetah |
| 2026-09-15 | 38 | 61.13 | 59.11 | 63.85 | 43.0 | mbpo/Ant, mbpo/HalfCheetah, sac/HalfCheetah, td3/HalfCheetah |
| 2026-09-17 | 42 | 60.75 | 58.53 | 64.80 | 42.0 | mbpo/Ant, sac/Ant, td3/Ant |
| 2026-09-18 | 22 | 61.76 | 58.96 | 63.70 | 42.5 | mbpo/Ant, tdmpc2/Ant, tdmpc2/HalfCheetah |
| 2026-09-19 | 21 | 62.16 | 60.71 | 63.56 | 43.0 | tdmpc2/Ant, tdmpc2/HalfCheetah |

Daily medians range 60.75–65.62 W (spread 4.87 W) across 11 calendar dates; correlation of head total power (corrected) with pre-head GPU temperature: r = 0.883.

## Caveats / things that look odd

- "Window-corrected" rests on an assumption (power during the extra ≈3 s equals power during the rest of the window) and on RAM power being constant; it is a proxy for the integrated span, not timestamp-based. Not checked against independent timestamps (none in `emissions_base_*.csv`).
- The correction raises drift because head power falls (÷ ≈93 s instead of 90 s) while tail is unchanged; it is not applied to `segment_energy.json` (README L648-649), so `idle_baseline_head` kWh there is the uncorrected, ≈3.4 % high value.
- Head idle power depends on session/day and pre-head GPU temperature (sections 5-6), i.e. drift is not only within-run heat carry-over.
