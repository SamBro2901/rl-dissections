# Section 7 (Evaluation) — fact collection

Generated from the `rl-dissections` checkout at `C:\Users\saman\Desktop\Thesis\Project Repository\rl-dissections` (git HEAD `050b3c3`, run data for all 300 run directories present). Read-only on `results/`, `flop_analysis/`, `configs/`. All scripts and companion CSVs are in `contexts/section_7_data/` (shared loader `common7.py`, task cache `build_task_table.py`; `_task_table.pkl` is a cache of all 72,890 `emissions_base_*.csv` task rows of the 280 canonical runs).

**IMPORTANT scope note.** This session ran on the Windows dev machine, **not** the Linux research PC. The run data are here, so items 1–6, 8 and 9 were computed/read directly. Item 7 needs commands on the research PC (`/etc/os-release`, `nvidia-smi -q`, `pip freeze` of `rl-exp`, `lscpu`, ...); those could not be run and are marked `[NOT FOUND]` (the local `nvidia-smi` shows a different GPU, an RTX 4050 Laptop, and was deliberately not used). Where the run metadata records a fact, it is reported.

## Summary block (every `[NOT FOUND]`, `[UNVERIFIED]`, `[MISMATCH]`)

**[MISMATCH]**
- M1 (claim 11): "All `rollout` tasks are shorter than 1 s" holds for SAC (max 0.532 s), TD3 (0.349 s), MBPO (0.408 s) but **not TD-MPC2**: all 5000 baseline-sweep `rollout` tasks are 6.6–16.7 s (median 11.07 s; baseline rows 10.1–12.0 s). Source: `item1_claim11_task_durations.csv`.
- M2 (item 4 vs draft statement "energy per epoch roughly constant after the first epochs"): true for SAC, TD3, TD-MPC2 (epochs 0–9 / 10–99 ratio 0.96–1.09), **false for MBPO `dynamics_model_update`** (ratio 0.26 Ant / 0.33 HC; +71 % / +134 % linear growth over 100 epochs), MBPO `synthetic_rollout_generation` on Ant (ratio 0.16, slope +257 %) and hence for the MBPO training total (ratio 0.44 / 0.52; R² of cumulative energy vs steps 0.993 / 0.984 instead of ~1.000). Source: `epoch_stationarity.csv`.
- M3 (claim 2/3 share convention, not a numeric error): the draft shares 70.46 %/74.28 % (MBPO dynamics) and 47.83 %/51.62 % (TD-MPC2 rollout) are **means of per-run shares**; the ratio-of-means shares from `d1_config_table.csv` are 70.49 %/74.41 % and 47.82 %/51.62 %. State the convention. Source: `item1_claims_all.csv`, `gross_vs_net.csv` (`share_gross_pct`).

**[NOT FOUND]** (item 7, all need the research PC; looked in: `metadata.json` (only `torch_version`, `codecarbon_version`, `cuda_device_name`, `device`), `emissions_base_*.csv` (OS string, Python, CPU, GPU model, RAM total), `run.log`, `README.md`, `CLAUDE.md`, `cli_commands.txt`, `requirements.txt`, `contexts/*.md`)
- N1 `rl-exp/bin/pip freeze`; gymnasium, MuJoCo, `nvidia-ml-py` versions on the Linux box (not recorded in `metadata.json`; `requirements.txt` has unpinned names only). CUDA runtime version beyond the `+cu130` torch build tag.
- N2 `/etc/os-release` and `uname -r` (see U1).
- N3 NVIDIA driver version, open-vs-proprietary kernel modules, VBIOS.
- N4 `nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS`: power limits, current/max clocks, supported graphics-clock range (whether 200 MHz is inside it).
- N5 `torch.cuda.get_device_properties(0)`: SM count, compute capability → peak FP32 TFLOP/s at 2000 MHz cannot be derived (no datasheet number used; formula in item 7).
- N6 RAM type/speed, mainboard model/vendor, storage model (`lscpu`/`free -h`/`lsblk`/`/sys/class/dmi/id/*` not run; DIMM details need `sudo dmidecode`).

**[UNVERIFIED]**
- U1 "Debian" in the draft hardware table: the recorded OS string is `Linux-7.0.0-31-generic-x86_64-with-glibc2.43` (all 280 canonical runs). It does not name a distribution. Check: `cat /etc/os-release` on the research PC.
- U2 FP32 vs TF32: no module in the repo sets `allow_tf32` / `set_float32_matmul_precision` (grep over all `*.py`, `*.sh`, `*.md`, `*.txt` outside `.venv`/`contexts`: no match). PyTorch's defaults (`'highest'`, `allow_tf32=False`) were confirmed only on the *local* build (torch 2.14.0+cpu); the run stack is torch 2.13.0+cu130. Check: `python -c "import torch;print(torch.__version__, torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)"` in `rl-exp`.
- U3 Mechanism behind GPU-energy zeros on very short tasks (item 3): the data show a cut-off at ≈0.1 s (below), consistent with an NVML total-energy counter that updates in ≈100 ms steps; the update period itself was not measured. Check: poll `nvmlDeviceGetTotalEnergyConsumption` at ~1 kHz under load and histogram the step spacing.
- U4 CPU energy came from RAPL in all runs: source shows RAPL is used only if CodeCarbon selects `intel_rapl` mode; `run.log` does not log the mode. Supporting evidence: per-task `cpu_power` varies (e.g. 10.0 W idle, 29.3 W run average, not a constant TDP value) and `CLAUDE.md` records zero "default power consumption of 4 W per thread" warnings. Check: CodeCarbon's own log line "Tracking Intel CPU via RAPL interface" in a fresh run, or `ls /sys/class/powercap/intel-rapl*`.
- U5 Whether the GPU-counter quantisation biases the sum of short `rollout` tasks (item 3): no bias is visible (rollout mean power 117–122 W median for every algorithm, including TD3-HC whose median task is 0.110 s) but this was not tested directly.

---

## 1. Claim check (P1)

All claims below were recomputed from the run data/pipeline outputs. Full table: `item1_claims_all.csv`. Scripts: `item1_claims.py`, `item1b_claims.py`.

| claim | item | found | status | source |
|---|---|---|---|---|
| 1 | sac HalfCheetah-v5 total mean±sd J | 60911.0 ± 254.1 | VERIFIED | contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J |
| 1 | sac Ant-v5 total mean±sd J | 64382.6 ± 588.3 | VERIFIED | contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J |
| 1 | td3 HalfCheetah-v5 total mean±sd J | 28111.5 ± 310.7 | VERIFIED | contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J |
| 1 | td3 Ant-v5 total mean±sd J | 30519.8 ± 431.9 | VERIFIED | contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J |
| 2 | mbpo HalfCheetah-v5 total mean±sd J | 207394.6 ± 7451.4 | VERIFIED | contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J |
| 2 | mbpo Ant-v5 total mean±sd J | 253137.9 ± 20140.0 | VERIFIED | contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J |
| 2 | mbpo HalfCheetah-v5 dyn share % (mean of per-run shares) | 70.46 | VERIFIED | per-run via section_5_2_analysis.load() |
| 2 | mbpo Ant-v5 dyn share % (mean of per-run shares) | 74.28 | VERIFIED | per-run via section_5_2_analysis.load() |
| 3 | tdmpc2 HalfCheetah-v5 total mean±sd J | 447086.8 ± 3184.4 | VERIFIED | contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J |
| 3 | tdmpc2 Ant-v5 total mean±sd J | 493191.1 ± 1782.0 | VERIFIED | contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J |
| 3 | tdmpc2 HalfCheetah-v5 rollout share % | 47.82 | VERIFIED | d1_config_table rollout_E_J/total_E_J |
| 3 | tdmpc2 HalfCheetah-v5 total J/FLOP pJ | 50.62 | VERIFIED | d1_config_table total_J_per_F |
| 3 | tdmpc2 Ant-v5 rollout share % | 51.62 | VERIFIED | d1_config_table rollout_E_J/total_E_J |
| 3 | tdmpc2 Ant-v5 total J/FLOP pJ | 48.95 | VERIFIED | d1_config_table total_J_per_F |
| 4 | pooled Spearman energy vs FLOPs HalfCheetah-v5 | rho=0.8357, n=27 | VERIFIED | contexts/section_5.2_data/d2_total_correlation.csv spearman_rho,n |
| 4 | pooled Spearman energy vs FLOPs Ant-v5 | rho=0.8463, n=29 | VERIFIED | contexts/section_5.2_data/d2_total_correlation.csv spearman_rho,n |
| 4 | Spearman return vs FLOPs HalfCheetah-v5 | rho=0.4174, p=0.0303, n=27 | VERIFIED | flop_analysis/output/correlation/correlation_summary.csv |
| 4 | Spearman return vs FLOPs Ant-v5 | rho=-0.3262, p=0.0842, n=29 | VERIFIED | flop_analysis/output/correlation/correlation_summary.csv |
| 5 | pooled log-log slope / R2 HalfCheetah-v5 | slope=0.4715, R2=0.6110 | VERIFIED | d2_total_correlation.csv loglog_slope,loglog_r2 |
| 5 | pooled log-log slope / R2 Ant-v5 | slope=0.5071, R2=0.6253 | VERIFIED | d2_total_correlation.csv loglog_slope,loglog_r2 |
| 6 | HalfCheetah-v5 total J/FLOP min pJ | 19.11 (td3,b1024) | VERIFIED | d1_config_table total_J_per_F |
| 6 | HalfCheetah-v5 total J/FLOP max pJ | 1421.3 (td3,w256) | VERIFIED | d1_config_table total_J_per_F |
| 6 | Ant-v5 total J/FLOP min pJ | 19.82 (td3,b1024) | VERIFIED | d1_config_table total_J_per_F |
| 6 | Ant-v5 total J/FLOP max pJ | 1232.5 (td3,w256) | VERIFIED | d1_config_table total_J_per_F |
| 7 | energy ordering td3<sac<mbpo at all 8 sweep values, both envs | 16/16 same as baseline; orderings=['td3 < sac < mbpo'] | VERIFIED | contexts/section_5.3_data/o2_swaps_vs_baseline.csv |
| 7 | seed-level energy orderings equal td3<sac<mbpo | 80/80 | VERIFIED | contexts/section_5.3_data/o3_seed_level.csv |
| 7 | J/FLOP non-baseline sweep values differing from baseline ordering (Ant-v5) | 6 of 7 | VERIFIED | o2_swaps_vs_baseline.csv (metric=jpf) |
| 7 | J/FLOP non-baseline sweep values differing from baseline ordering (HalfCheetah-v5) | 6 of 7 | VERIFIED | o2_swaps_vs_baseline.csv (metric=jpf) |
| 8 | total energy Ant/HC ratio range over 27 pairs | 1.027–1.248 | VERIFIED | contexts/section_5.4_data/pooled_summary.csv (ratio of means per pair) |
| 8 | rollout energy Ant/HC ratio range | 1.1055–2.2878 | VERIFIED | pooled_summary.csv E_rollout_min/max (1.1055 rounds to 1.106 at 3dp: 1.10552 -> 1.106) |
| 9 | R2000 (results/mbpo/HalfCheetah-v5/seed_0/20260904_122958) training tasks (non-idle, non-warmup) | 189.5 kJ, 27.30 min, 115.7 W | VERIFIED | emissions_base_*.csv, sum over tasks excluding idle_baseline_*/warmup |
| 9 | R200 (results/mbpo/HalfCheetah-v5/seed_0/20260904_130915) training tasks (non-idle, non-warmup) | 568.2 kJ, 95.53 min, 99.1 W | VERIFIED | emissions_base_*.csv, sum over tasks excluding idle_baseline_*/warmup |
| 10 | 300 run directories; 280 canonical; 20 dev | total=300, canonical=280, dev=20, configs=56 | VERIFIED | rglob results/**/metadata.json |
| 11 | SAC rollout median s / n | 0.3409 / 7000 | VERIFIED | item1_claim11_task_durations.csv |
| 11 | TD3 rollout median s / n | 0.2175 / 8000 | VERIFIED |  |
| 11 | MBPO rollout median s / n | 0.3536 / 8000 | VERIFIED |  |
| 11 | MBPO synthetic_rollout_generation median s | 0.008523 | VERIFIED |  |
| 11 | all sac rollout tasks < 1 s (max) | max=0.532s, n>=1s=0/7000 | VERIFIED |  |
| 11 | all td3 rollout tasks < 1 s (max) | max=0.349s, n>=1s=0/8000 | VERIFIED |  |
| 11 | all mbpo rollout tasks < 1 s (max) | max=0.408s, n>=1s=0/8000 | VERIFIED |  |
| 11 | all tdmpc2 rollout tasks < 1 s (max) | max=16.678s, n>=1s=5000/5000 | MISMATCH |  |

Notes on the table:
- Claim 2/3 percent shares: see M3 above. The table shows the per-run-mean convention as VERIFIED; the ratio-of-means version differs in the second decimal (70.49 / 74.41 for MBPO dynamics).
- Claim 8: 1.1055 (lower bound of rollout ratio) rounds to 1.106 at three decimals; shown as VERIFIED.
- Claim 9 numbers come from summing all non-idle, non-warmup task rows of the two seed-0 MBPO HalfCheetah hidden-256 runs `results/mbpo/HalfCheetah-v5/seed_0/20260904_122958` (clock-lock 2000/2000) and `.../20260904_130915` (200/200); they also match `contexts/Section_6.md` lines 178–179 (189 511 J / 1638.1 s and 568 210 J / 5731.98 s).
- Claim 10: the 20 non-canonical directories are (algo/env/seed: count) mbpo/HalfCheetah-v5/seed_0: 2; mbpo/HalfCheetah-v5/seed_42: 1; sac/HalfCheetah-v5/seed_0: 7; sac/HalfCheetah-v5/seed_42: 3; sac/HalfCheetah-v5/seed_56: 3; sac/Humanoid-v5/seed_56: 2; td3/HalfCheetah-v5/seed_0: 1; tdmpc2/HalfCheetah-v5/seed_0: 1; the exact list is in `execution_timeline_runs.csv` (`canonical=False`).
- Claim 11 TD-MPC2 values requested in the draft: `rollout` min 6.616 s, median 11.066 s, max 16.678 s, 5000 tasks (all ≥ 1 s) — see M1. MBPO `synthetic_rollout_generation`: median 0.008523 s over 8000 tasks, max 0.116 s.
Task duration table behind claim 11 (all canonical sweep runs; `item1_claim11_task_durations.csv`):

| algo | seg | n_tasks | min_s | median_s | p99_s | max_s | n_ge_1s |
|---|---|---|---|---|---|---|---|
| mbpo | rollout | 8000 | 0.1883 | 0.3536 | 0.3911 | 0.4077 | 0 |
| mbpo | synthetic_rollout_generation | 8000 | 0.0046 | 0.0085 | 0.1041 | 0.1163 | 0 |
| sac | rollout | 7000 | 0.1896 | 0.3409 | 0.3797 | 0.5322 | 0 |
| td3 | rollout | 8000 | 0.1003 | 0.2175 | 0.2667 | 0.3494 | 0 |
| tdmpc2 | rollout | 5000 | 6.6155 | 11.0661 | 16.6019 | 16.6775 | 5000 |

---

## 2. Learning sanity check (P1)

`[DERIVED]` Script `item2_learning_check.py`; output `learning_check.csv` (56 rows, one per algo×env×config; extra columns `first10_mean_length`, `last10_mean_length`, `frac_early_terminated_first10`, `n_seeds_last_gt_first`, `flag_last_minus_first_le_0`), per-run table `item2_learning_per_run.csv`, supplementary `learning_check_per_step_supplement.csv`.

Definitions (same as `aggregate_results.py` lines 56–65): episodes with `phase == "train"` only (warm-up episodes excluded); `last10` = mean return of the last `max(1, n//10)` episodes, `first10` = of the first `max(1, n//10)` episodes of that run; per-config values are means over the 5 seeds (sd = sample sd, ddof=1, of the per-seed last10 values). `frac_early_terminated_last10` (Ant only) = fraction of the last-10 % episodes with `length < 1000`, averaged over seeds.

**Flagged configs (`last_minus_first <= 0`): 14 of 56 — all on Ant-v5.** (HalfCheetah-v5: 0 of 27; every HC config has all 5 seeds improving.)

| algo | env | config | last10_return_mean | first10_return_mean | last_minus_first | n_seeds_last_gt_first | frac_early_terminated_last10 | first10_mean_length | last10_mean_length |
|---|---|---|---|---|---|---|---|---|---|
| mbpo | Ant-v5 | b1024 | -348.5 | -89.429 | -259.0 | 0 | 0.119 | 218.7 | 924.7 |
| mbpo | Ant-v5 | b512 | -244.6 | -77.01 | -167.6 | 1 | 0.196 | 167.5 | 855.5 |
| mbpo | Ant-v5 | base | -325.5 | -108.2 | -217.2 | 1 | 0.063 | 235.0 | 958.0 |
| mbpo | Ant-v5 | rollout1 | -756.7 | -113.4 | -643.4 | 0 | 0.477 | 225.3 | 653.4 |
| mbpo | Ant-v5 | rollout15 | -407.7 | -98.784 | -308.9 | 0 | 0.136 | 225.0 | 896.1 |
| mbpo | Ant-v5 | utd2 | -459.1 | -115.1 | -344.0 | 0 | 0.166 | 223.7 | 869.5 |
| mbpo | Ant-v5 | utd4 | -599.2 | -104.4 | -494.8 | 0 | 0.179 | 150.6 | 868.5 |
| mbpo | Ant-v5 | w256 | -695.5 | -92.766 | -602.7 | 0 | 0.269 | 193.6 | 768.9 |
| mbpo | Ant-v5 | w512 | -413.4 | -74.276 | -339.1 | 0 | 0.197 | 145.8 | 868.4 |
| sac | Ant-v5 | b1024 | -104.5 | -56.545 | -48.002 | 0 | 0.803 | 241.0 | 284.3 |
| sac | Ant-v5 | b512 | -54.023 | -52.188 | -1.835 | 2 | 0.665 | 293.5 | 441.6 |
| sac | Ant-v5 | utd2 | -122.4 | -102.1 | -20.366 | 2 | 0.722 | 266.4 | 399.7 |
| sac | Ant-v5 | utd4 | -220.1 | -152.7 | -67.372 | 2 | 0.776 | 194.3 | 315.2 |
| td3 | Ant-v5 | b1024 | -30.311 | 31.36 | -61.671 | 0 | 0.879 | 317.9 | 194.4 |

Per algorithm/env summary of the last-10 % return across configs:

| env | algo | n_cfg | last10_min | last10_max | n_flagged | n_eps_min | n_eps_max |
|---|---|---|---|---|---|---|---|
| Ant-v5 | mbpo | 9 | -756.7 | -244.6 | 9 | 154 | 431 |
| Ant-v5 | sac | 7 | -220.1 | 408.6 | 4 | 178 | 511 |
| Ant-v5 | td3 | 8 | -30.31 | 575.9 | 1 | 171 | 546 |
| Ant-v5 | tdmpc2 | 5 | 125.8 | 438.5 | 0 | 440 | 2556 |
| HalfCheetah-v5 | mbpo | 7 | 4534.8 | 5903.7 | 0 | 100 | 100 |
| HalfCheetah-v5 | sac | 7 | 2920.6 | 4884.7 | 0 | 100 | 100 |
| HalfCheetah-v5 | td3 | 8 | 3684.6 | 5539.5 | 0 | 100 | 100 |
| HalfCheetah-v5 | tdmpc2 | 5 | 5790.9 | 7755.7 | 0 | 100 | 100 |

**Do first-vs-last comparisons use episodes of different lengths on Ant? — Yes.** HalfCheetah episodes are always 1000 steps (0 of 135 HC runs contain any shorter episode). On Ant, 145 of the 145 Ant runs contain at least one episode shorter than 1000 steps; the number of train episodes per run varies strongly (SAC 178–511, MBPO 154–431, TD3 171–546, **TD-MPC2 884–2556**) because 100 000 train steps are split into episodes of varying length. Episode returns are per-episode sums, so an episode that ends early scores less regardless of per-step quality. Handling: the headline columns follow the requested definition (episode-count-based first/last 10 %, no length normalisation) and the lengths are reported next to them (`first10_mean_length`, `last10_mean_length`). As a sensitivity check (supplementary file) the return was also normalised per step: sum of returns / sum of lengths over the same first/last 10 % episodes. With per-step normalisation 4 of the 14 flagged configs remain non-improving (listed below); the other 10 flagged configs show a positive per-step change, i.e. their negative `last_minus_first` is driven (at least partly) by shorter late episodes.

| algo | config | first10_return_per_step | last10_return_per_step | last_minus_first_per_step |
|---|---|---|---|---|
| mbpo | rollout1 | -0.516 | -1.065 | -0.55 |
| mbpo | w256 | -0.486 | -0.893 | -0.407 |
| sac | b1024 | -0.238 | -0.369 | -0.131 |
| td3 | b1024 | 0.1 | -0.147 | -0.247 |

TD-MPC2 on Ant: the small last-10 % episode returns (126–438) come from very short late episodes: mean episode length of the last 10 % is 38–209 steps (base: 45.5) with `frac_early_terminated_last10` 0.9–1.0, and 884–2556 episodes per run. Per step the last-10 % return is 2.0–3.5 vs 0.05–0.50 for the first 10 %. Rows:

| config | last10_return_mean | n_episodes_mean | first10_mean_length | last10_mean_length | frac_early_terminated_last10 |
|---|---|---|---|---|---|
| base | 149.6 | 1596.4 | 174.6 | 45.545 | 0.994 |
| horizon1 | 438.5 | 474.4 | 345.8 | 208.8 | 0.931 |
| horizon5 | 147.1 | 1484.8 | 162.7 | 45.9 | 0.996 |
| numq3 | 125.8 | 1926.6 | 152.9 | 37.738 | 0.998 |
| numq7 | 356.2 | 1192.8 | 188.5 | 88.738 | 0.98 |

---

## 3. CodeCarbon and tasks shorter than the polling interval (P1)

### 3a. Source reading (`[FOUND]`; installed `codecarbon` 3.3.0 — the Windows dev venv has the same version as the research PC's recorded `codecarbon_version`; paths relative to `.venv/Lib/site-packages/codecarbon/`)

1. **What `start_task` / `stop_task` do** (`emissions_tracker.py:730` / `:795`). `start_task` stops the periodic energy scheduler (`self._scheduler.stop()`, ~l.752), starts a 1 s background *monitor* thread that only samples utilisation/CPU power history (`_scheduler_monitor_power`, interval=1, l.349–352; `_monitor_power` l.1136), resets the clock and **re-baselines every device counter**:
   `for hardware in self._hardware:` / `    hardware.start()` (l.776–777); then snapshots the running totals (`_prepare_emissions_data`, l.778). `stop_task` forces one final reading, `self._measure_power_and_energy()` (l.808), and returns `totals(stop) − snapshot(start)` via `emissions_data_delta.compute_delta_emission(self._active_task_emissions_at_start)` (l.828–830; `output_methods/emissions_data.py:55–63`); duration = `perf_counter()` difference (l.~842).
2. **Counters or polled power?** Per device:
   - GPU: cumulative NVML counter. `gpu_nvidia.py:54` `return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)`; `core/gpu_device.py:34–35` `def start(self): self.last_energy = self._get_energy_kwh()`; `:51,55` `new_last_energy = energy = self._get_energy_kwh()` … `self.energy_delta = energy - self.last_energy`.
   - CPU (only in `intel_rapl` mode): cumulative RAPL file. `core/rapl.py:67–68` `def start(self): self.last_energy = self._get_value()`, `:83` `self.energy_delta = energy - self.last_energy`; `external/hardware.py:405–408` `if self._mode in ("intel_rapl", "windows_emi"): energy = self._get_energy_from_cpus(...)`. (Other CPU modes use power × time; see U4.)
   - RAM: **power × duration**. `RAM` does not override `measure_power_and_energy` (grep: none in `external/ram.py`), so `BaseHardware` applies (`external/hardware.py:56`) `energy = Energy.from_power_and_time(power=power, time=Time.from_seconds(last_duration))`.
3. **A task shorter than the polling interval.** Because the scheduler is stopped inside a task and `stop_task` forces a measurement, task energy for GPU/CPU is a counter difference between the `start_task` baseline and the `stop_task` read; `measure_power_secs = 1.0` only governs the background scheduler (which is off during tasks), so task energy is **not** quantised to 1 s. Its resolution is instead that of the counters. Energy accrued between tasks (after one `stop_task`, before the next `start_task`) is not attributed to any task, because `hardware.start()` re-baselines the counters.
4. **RAM term.** `external/ram.py:312–341` `total_power()` → `_calculate_ram_power(memory_GB)` (DIMM-count model, `RAM_SLOT_POWER_X86 = 5` W, l.14, `:141`); in `tracking_mode='machine'` `memory_GB` is machine RAM (60.8 GB). Observed: `ram_power` is exactly **20.0 W in all 72 890 task rows** (min = max = 20.0). RAM task energy is therefore 20 W × task wall-clock: a constant-power model, not a measurement.

`[UNVERIFIED]` What the source cannot settle is the *update granularity* of the NVML counter; that is answered only empirically (3b, U3).

### 3b. Empirical check on the 8 canonical baselines (4 algos × 2 envs; 500 tasks per segment = 100 epochs × 5 seeds)

`[DERIVED]` `item3b_short_tasks.py` → `item3b_short_task_stats.csv`. Source: `emissions_base_*.csv` (task rows `rollout_<epoch>`, `gradient_updates_<epoch>`, …). Energy = `energy_consumed` (CPU+GPU+RAM) in J; power = energy ÷ task duration. Counts are the numbers of tasks with energy ≤ 0.

| algo | env | segment | n_tasks | duration_s_min | duration_s_median | duration_s_max | energy_J_min | energy_J_median | energy_J_max | power_W_min | power_W_median | power_W_max | n_gpu_energy_le0 | n_cpu_energy_le0 | n_total_energy_le0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mbpo | Ant-v5 | gradient_updates | 500 | 4.028 | 4.078 | 4.427 | 570.9 | 595.4 | 633.5 | 136.6 | 145.5 | 153.7 | 0 | 0 | 0 |
| mbpo | Ant-v5 | rollout | 500 | 0.349 | 0.366 | 0.401 | 35.725 | 43.749 | 48.653 | 94.157 | 117.8 | 136.5 | 0 | 0 | 0 |
| mbpo | HalfCheetah-v5 | gradient_updates | 500 | 3.969 | 4.028 | 4.42 | 559.3 | 584.8 | 634.4 | 136.1 | 143.8 | 152.9 | 0 | 0 | 0 |
| mbpo | HalfCheetah-v5 | rollout | 500 | 0.194 | 0.203 | 0.25 | 17.037 | 24.063 | 32.306 | 86.852 | 120.1 | 153.2 | 0 | 0 | 0 |
| sac | Ant-v5 | gradient_updates | 500 | 3.979 | 4.07 | 4.519 | 568.6 | 597.2 | 671.5 | 136.9 | 145.6 | 157.1 | 0 | 0 | 0 |
| sac | Ant-v5 | rollout | 500 | 0.343 | 0.357 | 0.532 | 34.752 | 44.898 | 60.93 | 93.645 | 122.4 | 154.3 | 0 | 0 | 0 |
| sac | HalfCheetah-v5 | gradient_updates | 500 | 3.878 | 3.995 | 4.5 | 558.1 | 580.5 | 637.2 | 136.5 | 144.5 | 157.7 | 0 | 0 | 0 |
| sac | HalfCheetah-v5 | rollout | 500 | 0.195 | 0.203 | 0.37 | 21.441 | 24.54 | 41.943 | 102.6 | 121.1 | 181.3 | 0 | 0 | 0 |
| td3 | Ant-v5 | gradient_updates | 500 | 1.886 | 1.939 | 2.347 | 252.9 | 271.0 | 311.6 | 128.4 | 137.7 | 146.8 | 0 | 0 | 0 |
| td3 | Ant-v5 | rollout | 500 | 0.234 | 0.245 | 0.34 | 24.396 | 30.332 | 43.986 | 97.328 | 117.0 | 150.8 | 0 | 0 | 0 |
| td3 | HalfCheetah-v5 | gradient_updates | 500 | 1.847 | 1.906 | 2.242 | 249.0 | 263.5 | 300.4 | 126.9 | 136.7 | 144.2 | 0 | 0 | 0 |
| td3 | HalfCheetah-v5 | rollout | 500 | 0.105 | 0.11 | 0.199 | 11.559 | 13.307 | 23.856 | 101.1 | 120.8 | 200.8 | 0 | 0 | 0 |
| tdmpc2 | Ant-v5 | gradient_updates | 500 | 11.797 | 12.147 | 14.445 | 2155.8 | 2247.5 | 2439.9 | 165.1 | 184.5 | 189.0 | 0 | 0 | 0 |
| tdmpc2 | Ant-v5 | rollout | 500 | 11.912 | 11.945 | 12.011 | 2491.7 | 2545.7 | 2583.0 | 208.4 | 213.1 | 216.2 | 0 | 0 | 0 |
| tdmpc2 | HalfCheetah-v5 | gradient_updates | 500 | 11.37 | 13.413 | 13.947 | 2049.8 | 2262.3 | 2321.8 | 163.4 | 169.9 | 187.4 | 0 | 0 | 0 |
| tdmpc2 | HalfCheetah-v5 | rollout | 500 | 10.093 | 10.133 | 10.347 | 2096.4 | 2138.0 | 2251.7 | 207.3 | 210.9 | 220.7 | 0 | 0 | 0 |

Extra segments (MBPO, same source):

| algo | env | segment | n_tasks | duration_s_min | duration_s_median | duration_s_max | energy_J_median | power_W_median | power_W_max | n_gpu_energy_le0 | n_cpu_energy_le0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mbpo | Ant-v5 | dynamics_model_update | 500 | 1.03 | 15.707 | 49.554 | 1845.5 | 116.9 | 127.2 | 0 | 0 |
| mbpo | Ant-v5 | synthetic_rollout_generation | 500 | 0.007 | 0.05 | 0.116 | 4.666 | 56.037 | 823.2 | 254 | 0 |
| mbpo | HalfCheetah-v5 | dynamics_model_update | 500 | 1.112 | 11.191 | 65.747 | 1318.0 | 117.3 | 131.5 | 0 | 0 |
| mbpo | HalfCheetah-v5 | synthetic_rollout_generation | 500 | 0.006 | 0.006 | 0.019 | 0.301 | 49.035 | 1199.7 | 466 | 0 |

Findings (all `[DERIVED]` from the tables above and `item3b_gpu_zero_by_duration_bucket_all280.csv`):
- No `rollout` or `gradient_updates` task in the 8 baselines has zero/negative GPU, CPU or total energy (columns `n_*_le0` = 0). Per-task mean power is 86.9–220.7 W for rollouts and 126.9–189.0 W for gradient updates (plausible for this GPU+CPU+RAM total); rollout task power medians are 117–122 W for SAC/TD3/MBPO and 211–213 W for TD-MPC2.
- **MBPO `synthetic_rollout_generation` is different**: 720 of 1000 baseline tasks (466/500 HC, 254/500 Ant) have **exactly 0 GPU energy** (none negative); CPU energy is never ≤ 0. Median task duration 6.1 ms (HC) / 50.3 ms (Ant), and some tasks show implausible mean power (max 823–1200 W) when a counter step happens to land in a short window. Its recorded energy is 0.04 % (HC) / 0.29 % (Ant) of the MBPO baseline training total, so the effect on totals is small, but the *segment's own* energy is not reliable (GPU term mostly missing).
- Zero-GPU-energy tasks vs duration, **all 280 canonical runs, all segments except idle** (`item3b_gpu_zero_by_duration_bucket_all280.csv`):

| b | n | gpu_energy_le0 | cpu_energy_le0 |
|---|---|---|---|
| (0.0, 0.01] | 4441 | 4140 | 0 |
| (0.01, 0.025] | 881 | 745 | 0 |
| (0.025, 0.05] | 836 | 521 | 0 |
| (0.05, 0.1] | 1673 | 448 | 0 |
| (0.1, 0.2] | 7299 | 0 | 0 |
| (0.2, 0.5] | 16004 | 0 | 0 |
| (0.5, 1.0] | 107 | 0 | 0 |
| (1.0, 5.0] | 19333 | 0 | 0 |
| (5.0, 1000000000.0] | 21756 | 0 | 0 |

  Every task ≥ 0.1 s has GPU energy > 0, almost every task ≤ 0.01 s has GPU energy = 0. This is consistent with an NVML energy-counter update period of order 0.1 s (U3). `rollout` tasks of SAC/MBPO/TD3 are 0.10–0.53 s (smallest: TD3-HalfCheetah, min 0.105 s, median 0.110 s), i.e. at or a few multiples of that scale: individual task energies are coarsely quantised, but all are non-zero and the per-task energy is only weakly correlated with duration (Pearson r 0.02–0.53 per algo/env, `item3b_short_tasks.py` output). TD-MPC2 rollouts (6.6–16.7 s) are two orders above it.
- CPU: no task of any length has CPU energy ≤ 0 (RAPL resolution is not an issue at these durations).
- Verdict: for `rollout` the energy is a valid counter-difference measurement but with coarse per-task resolution for the shortest (TD3-HC) tasks; for MBPO `synthetic_rollout_generation` the GPU part is largely missing (`[UNVERIFIED]` whether the sum over the run is unbiased, see U3/U5).

---

## 4. Per-epoch stationarity (P1)

`[DERIVED]` `item4_epoch_stationarity.py` → `epoch_stationarity.csv`. Baselines of 4 algos × 2 envs, 5 seeds each, 100 epochs (task names `<segment>_<epoch>`, epoch 0–99; per-epoch energy = task energy in J). `ratio` = mean(epochs 0–9) ÷ mean(epochs 10–99), computed per seed then averaged (the ratio of seed-mean series is also in the CSV and agrees to ≤ 0.002). `slope` = OLS slope of per-epoch energy on epoch × 100, as % of the run's mean epoch energy, averaged over seeds (min/max over seeds shown). `cv` = sd/mean of the 100 epoch values within a run, averaged over seeds. `R²` = squared Pearson correlation of cumulative training energy vs cumulative env steps (1000·(epoch+1)), per seed, averaged. `TRAINING_TOTAL` per epoch = sum of the per-epoch measured segments present (rollout + gradient_updates [+ MBPO dynamics + synthetic]); TD-MPC2's one-off `world_model_pretrain` has no epoch index and is excluded. Segments not present for an algorithm are omitted.

| algo | env | segment | mean_epoch_energy_J | ratio_ep0_9_over_ep10_99_mean_seed | slope_pct_of_mean_per_100_epochs | slope_pct_min_over_seeds | slope_pct_max_over_seeds | cv_within_run_mean | r2_cum_energy_vs_cum_steps |
|---|---|---|---|---|---|---|---|---|---|
| mbpo | Ant-v5 | rollout | 42.924 | 0.992 | 0.872 | -1.291 | 3.302 | 0.076 |  |
| mbpo | Ant-v5 | gradient_updates | 597.5 | 0.997 | -1.587 | -2.976 | -0.023 | 0.019 |  |
| mbpo | Ant-v5 | dynamics_model_update | 1883.6 | 0.26 | 70.995 | 25.966 | 118.1 | 0.453 |  |
| mbpo | Ant-v5 | synthetic_rollout_generation | 7.357 | 0.158 | 256.7 | 243.3 | 272.3 | 0.908 |  |
| mbpo | Ant-v5 | TRAINING_TOTAL | 2531.4 | 0.437 | 53.673 | 19.779 | 90.502 | 0.336 | 0.993 |
| mbpo | HalfCheetah-v5 | rollout | 24.283 | 1.032 | -5.512 | -9.374 | -2.584 | 0.068 |  |
| mbpo | HalfCheetah-v5 | gradient_updates | 587.0 | 0.992 | -3.19 | -4.395 | -2.31 | 0.024 |  |
| mbpo | HalfCheetah-v5 | dynamics_model_update | 1461.9 | 0.335 | 133.8 | 95.587 | 157.3 | 0.724 |  |
| mbpo | HalfCheetah-v5 | synthetic_rollout_generation | 0.754 | 0.983 | 9.578 | -92.077 | 50.774 | 2.169 |  |
| mbpo | HalfCheetah-v5 | TRAINING_TOTAL | 2073.9 | 0.52 | 93.403 | 65.883 | 109.1 | 0.509 | 0.984 |
| sac | Ant-v5 | rollout | 43.321 | 0.991 | -1.418 | -7.104 | 1.339 | 0.103 |  |
| sac | Ant-v5 | gradient_updates | 600.5 | 0.996 | 0.038 | -2.998 | 3.61 | 0.026 |  |
| sac | Ant-v5 | TRAINING_TOTAL | 643.8 | 0.996 | -0.06 | -2.733 | 3.333 | 0.026 | 1 |
| sac | HalfCheetah-v5 | rollout | 24.957 | 1.033 | -4.947 | -8.399 | 1.697 | 0.095 |  |
| sac | HalfCheetah-v5 | gradient_updates | 584.2 | 0.993 | -0.222 | -1.586 | 0.615 | 0.024 |  |
| sac | HalfCheetah-v5 | TRAINING_TOTAL | 609.1 | 0.994 | -0.416 | -1.872 | 0.659 | 0.024 | 1 |
| td3 | Ant-v5 | rollout | 30.715 | 1.036 | -1.642 | -6.141 | 1.165 | 0.134 |  |
| td3 | Ant-v5 | gradient_updates | 274.5 | 0.993 | 2.704 | -9.091 | 10.581 | 0.043 |  |
| td3 | Ant-v5 | TRAINING_TOTAL | 305.2 | 0.997 | 2.27 | -8.616 | 9.652 | 0.04 | 1 |
| td3 | HalfCheetah-v5 | rollout | 14.109 | 1.086 | -6.372 | -14.924 | 4.05 | 0.187 |  |
| td3 | HalfCheetah-v5 | gradient_updates | 267.0 | 0.981 | 3.172 | -0.505 | 7.872 | 0.041 |  |
| td3 | HalfCheetah-v5 | TRAINING_TOTAL | 281.1 | 0.986 | 2.694 | -0.567 | 7.681 | 0.04 | 1 |
| tdmpc2 | Ant-v5 | rollout | 2545.8 | 0.991 | 0.633 | 0.2 | 1.46 | 0.005 |  |
| tdmpc2 | Ant-v5 | gradient_updates | 2277.5 | 0.964 | 5.105 | -0.387 | 11.485 | 0.031 |  |
| tdmpc2 | Ant-v5 | TRAINING_TOTAL | 4823.3 | 0.978 | 2.754 | -0.017 | 5.718 | 0.015 | 1 |
| tdmpc2 | HalfCheetah-v5 | rollout | 2138.1 | 0.991 | 0.774 | 0.375 | 1.195 | 0.006 |  |
| tdmpc2 | HalfCheetah-v5 | gradient_updates | 2224.1 | 0.995 | -2.38 | -7.97 | 5.46 | 0.031 |  |
| tdmpc2 | HalfCheetah-v5 | TRAINING_TOTAL | 4362.2 | 0.993 | -0.834 | -3.471 | 3.075 | 0.015 | 1 |

Reading (facts only): SAC and TD3 are flat (ratio 0.98–1.09; total-energy R² 0.9998–1.0000); TD-MPC2 is flat (ratio 0.96–0.99; within-run CV of rollout 0.5–0.6 %, total 1.5 %; R² 1.0000); MBPO `rollout` and `gradient_updates` are flat (ratio 0.99–1.03) but `dynamics_model_update` (ratio 0.26 Ant / 0.33 HC, CV 45 % / 72 %) and `synthetic_rollout_generation` on Ant (ratio 0.16) are strongly non-stationary — see M2. Individual-seed slopes of TD3 `gradient_updates` range from −9 % to +11 % per 100 epochs, so "flat" there means no systematic trend, not zero scatter.

---

## 5. Idle drift and gross-vs-net sensitivity (P1 for a, P2 for b)

### 5a. Idle drift over the 280 canonical runs

`[DERIVED]` `item5_idle.py` on `idle_power.csv` (repo root; produced by `idle_power_analysis.py`, last commit `57b3d21`; 280 rows, all canonical seeds, columns `head_total_w`/`tail_total_w` as recorded and `*_corr` window-corrected). Statistic: |P_tail − P_head| / P_head in %, and |P_tail − P_head| in W; median, quartiles (Q1/Q3), max. `n_tail_gt_head` = runs with tail > head. Output `item5a_idle_drift.csv`.

| scope | convention | n_runs | n_tail_gt_head | rel_pct_median | rel_pct_q1 | rel_pct_q3 | rel_pct_max | abs_W_median | abs_W_q1 | abs_W_q3 | abs_W_max | head_W_mean | tail_W_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | as_recorded | 280 | 256 | 7.04 | 4.45 | 10.92 | 23.19 | 4.48 | 2.86 | 7.12 | 14.74 | 65.27 | 69.51 |
| ALL | window_corrected | 280 | 261 | 10.02 | 7.61 | 14.36 | 27.39 | 6.23 | 4.79 | 8.97 | 16.84 | 63.11 | 69.51 |
| mbpo|Ant-v5 | as_recorded | 45 | 45 | 4.11 | 3.48 | 5.02 | 8.87 | 2.65 | 2.14 | 3.14 | 5.41 | 62.62 | 65.32 |
| mbpo|Ant-v5 | window_corrected | 45 | 45 | 7.63 | 7.03 | 8.63 | 12.57 | 4.77 | 4.19 | 5.17 | 7.41 | 60.55 | 65.33 |
| mbpo|HalfCheetah-v5 | as_recorded | 35 | 33 | 4.14 | 2.86 | 5.74 | 8.99 | 2.67 | 1.82 | 3.57 | 5.49 | 63.33 | 65.9 |
| mbpo|HalfCheetah-v5 | window_corrected | 35 | 35 | 7.68 | 6.37 | 9.35 | 12.72 | 4.79 | 3.95 | 5.62 | 7.52 | 61.23 | 65.9 |
| sac|Ant-v5 | as_recorded | 35 | 29 | 8.48 | 4.08 | 12.02 | 23.19 | 5.71 | 2.86 | 7.88 | 14.74 | 67.23 | 71.03 |
| sac|Ant-v5 | window_corrected | 35 | 29 | 9.16 | 7.37 | 14.67 | 27.39 | 6.11 | 4.8 | 8.9 | 16.84 | 65.01 | 71.03 |
| sac|HalfCheetah-v5 | as_recorded | 35 | 29 | 8.3 | 5.66 | 11.5 | 22.44 | 5.43 | 3.65 | 8.52 | 14.21 | 66.62 | 70 |
| sac|HalfCheetah-v5 | window_corrected | 35 | 29 | 9.95 | 8.29 | 14.62 | 26.62 | 6.33 | 5.76 | 8.98 | 16.31 | 64.4 | 70 |
| td3|Ant-v5 | as_recorded | 40 | 36 | 9.6 | 6.11 | 11.4 | 20.48 | 6.12 | 3.98 | 7.63 | 13.16 | 66.44 | 71.88 |
| td3|Ant-v5 | window_corrected | 40 | 38 | 13.33 | 9.57 | 15.22 | 24.57 | 8.2 | 6.09 | 9.87 | 15.27 | 64.24 | 71.88 |
| td3|HalfCheetah-v5 | as_recorded | 40 | 34 | 9.02 | 6.43 | 11.73 | 18.11 | 6.2 | 4.45 | 7.72 | 11.73 | 67.01 | 71.85 |
| td3|HalfCheetah-v5 | window_corrected | 40 | 35 | 12.75 | 9.22 | 15.63 | 22.11 | 7.98 | 6.03 | 9.83 | 13.85 | 64.8 | 71.85 |
| tdmpc2|Ant-v5 | as_recorded | 25 | 25 | 10.18 | 7.19 | 12.33 | 17.03 | 6.56 | 4.69 | 8.06 | 10.99 | 64.49 | 70.84 |
| tdmpc2|Ant-v5 | window_corrected | 25 | 25 | 13.91 | 10.82 | 16.25 | 21.05 | 8.68 | 6.82 | 10.19 | 13.13 | 62.35 | 70.84 |
| tdmpc2|HalfCheetah-v5 | as_recorded | 25 | 25 | 10.02 | 7.3 | 11.49 | 14.89 | 6.34 | 4.66 | 7.23 | 9.48 | 64.26 | 70.39 |
| tdmpc2|HalfCheetah-v5 | window_corrected | 25 | 25 | 13.76 | 10.98 | 15.3 | 18.8 | 8.42 | 6.78 | 9.31 | 11.58 | 62.13 | 70.39 |

Headline (rows `ALL`): as recorded median |Δ| = 7.04 % (4.48 W, IQR 4.45–10.92 % / 2.86–7.12 W, max 23.2 % / 14.7 W; tail > head in 256/280); window-corrected median 10.02 % (6.23 W, IQR 7.61–14.36 % / 4.79–8.97 W, max 27.4 % / 16.8 W; tail > head in 261/280). Mean head 65.27 W (63.11 W corrected) vs mean tail 69.51 W.

### 5b. Sensitivity of the baseline results to the gross convention (sensitivity analysis only; thesis stays gross) — `[DERIVED]`

Script `item5_idle.py` → `gross_vs_net.csv` (segments, shares, totals, J/FLOP per variant) and `gross_vs_net_ratios.csv` (orderings and ratios). Population: the 8 baseline configs × 5 canonical seeds = 40 runs. Per run: `E_net(seg) = E(seg) − P_idle × T(seg)` with `E(seg)` = sum of the CodeCarbon task energies of the measured segment (`rollout`, `gradient_updates`, and where present `dynamics_model_update`, `synthetic_rollout_generation`, `world_model_pretrain`; sub-segments are allocated, not measured, so not used), `T(seg)` = summed task durations of that segment, `P_idle` = mean of window-corrected head and tail total power of that run (primary variant `mean_head_tail`); brackets `head_only` and `tail_only` use the window-corrected head or tail power alone. Sanity check: summed gross task energies equal the pipeline `TOTAL_MEASURED_TRAINING` energy per run (max relative error 3.2e-13). J/FLOP = mean energy ÷ mean matmul FLOPs (`TOTAL_MEASURED_TRAINING` FLOPs from `per_run_energy_per_flop.csv`), same ratio-of-means convention as `d1_config_table.csv`. Segment shares are means of per-run shares.

**Total training energy and J/FLOP, gross vs net (primary variant):**

| algo | env | n_runs | E_gross_J | E_net_J | net_over_gross | P_idle_W_mean | jpf_gross_pJ | jpf_net_pJ |
|---|---|---|---|---|---|---|---|---|
| mbpo | Ant-v5 | 5 | 2.531e+05 | 1.203e+05 | 0.475 | 64.508 | 178.8 | 84.986 |
| mbpo | HalfCheetah-v5 | 5 | 2.074e+05 | 98546.3 | 0.475 | 64.842 | 183.1 | 86.99 |
| sac | Ant-v5 | 5 | 64382.6 | 33726.0 | 0.524 | 68.258 | 68.567 | 35.918 |
| sac | HalfCheetah-v5 | 5 | 60911.0 | 32368.3 | 0.531 | 67.105 | 69.447 | 36.904 |
| td3 | Ant-v5 | 5 | 30519.8 | 14332.8 | 0.47 | 72.222 | 116.3 | 54.601 |
| td3 | HalfCheetah-v5 | 5 | 28111.5 | 13517.9 | 0.481 | 70.474 | 114.3 | 54.979 |
| tdmpc2 | Ant-v5 | 5 | 4.932e+05 | 3.217e+05 | 0.652 | 68.179 | 48.945 | 31.927 |
| tdmpc2 | HalfCheetah-v5 | 5 | 4.471e+05 | 2.885e+05 | 0.645 | 67.228 | 50.617 | 32.658 |

**Net total energy [J] for the three `P_idle` choices:**

| algo | env | head_only | mean_head_tail | tail_only |
|---|---|---|---|---|
| mbpo | Ant-v5 | 1.256e+05 | 1.203e+05 | 1.151e+05 |
| mbpo | HalfCheetah-v5 | 1.024e+05 | 98546.3 | 94690.9 |
| sac | Ant-v5 | 35593.9 | 33726.0 | 31858.1 |
| sac | HalfCheetah-v5 | 33421.3 | 32368.3 | 31315.3 |
| td3 | Ant-v5 | 15127.3 | 14332.8 | 13538.3 |
| td3 | HalfCheetah-v5 | 14205.5 | 13517.9 | 12830.4 |
| tdmpc2 | Ant-v5 | 3.346e+05 | 3.217e+05 | 3.088e+05 |
| tdmpc2 | HalfCheetah-v5 | 3e+05 | 2.885e+05 | 2.77e+05 |

**Net training-total J/FLOP [pJ] for the three `P_idle` choices (gross for reference is in the previous table):**

| algo | env | head_only | mean_head_tail | tail_only |
|---|---|---|---|---|
| mbpo | Ant-v5 | 88.72 | 84.99 | 81.25 |
| mbpo | HalfCheetah-v5 | 90.39 | 86.99 | 83.59 |
| sac | Ant-v5 | 37.91 | 35.92 | 33.93 |
| sac | HalfCheetah-v5 | 38.1 | 36.9 | 35.7 |
| td3 | Ant-v5 | 57.63 | 54.6 | 51.57 |
| td3 | HalfCheetah-v5 | 57.78 | 54.98 | 52.18 |
| tdmpc2 | Ant-v5 | 33.21 | 31.93 | 30.64 |
| tdmpc2 | HalfCheetah-v5 | 33.96 | 32.66 | 31.36 |

**Algorithm orderings and ratios (energy ratios `sac/td3` = SAC÷TD3, etc.; J/FLOP ordering is lowest→highest):**

| variant | env | basis | ordering | sac/td3 | mbpo/sac | tdmpc2/mbpo | jpf_ordering |
|---|---|---|---|---|---|---|---|
| head_only | Ant-v5 | gross | td3 < sac < mbpo < tdmpc2 | 2.11 | 3.932 | 1.948 | tdmpc2 < sac < td3 < mbpo |
| head_only | Ant-v5 | net | td3 < sac < mbpo < tdmpc2 | 2.353 | 3.53 | 2.664 | tdmpc2 < sac < td3 < mbpo |
| head_only | HalfCheetah-v5 | gross | td3 < sac < mbpo < tdmpc2 | 2.167 | 3.405 | 2.156 | tdmpc2 < sac < td3 < mbpo |
| head_only | HalfCheetah-v5 | net | td3 < sac < mbpo < tdmpc2 | 2.353 | 3.064 | 2.929 | tdmpc2 < sac < td3 < mbpo |
| mean_head_tail | Ant-v5 | gross | td3 < sac < mbpo < tdmpc2 | 2.11 | 3.932 | 1.948 | tdmpc2 < sac < td3 < mbpo |
| mean_head_tail | Ant-v5 | net | td3 < sac < mbpo < tdmpc2 | 2.353 | 3.568 | 2.673 | tdmpc2 < sac < td3 < mbpo |
| mean_head_tail | HalfCheetah-v5 | gross | td3 < sac < mbpo < tdmpc2 | 2.167 | 3.405 | 2.156 | tdmpc2 < sac < td3 < mbpo |
| mean_head_tail | HalfCheetah-v5 | net | td3 < sac < mbpo < tdmpc2 | 2.394 | 3.045 | 2.927 | tdmpc2 < sac < td3 < mbpo |
| tail_only | Ant-v5 | gross | td3 < sac < mbpo < tdmpc2 | 2.11 | 3.932 | 1.948 | tdmpc2 < sac < td3 < mbpo |
| tail_only | Ant-v5 | net | td3 < sac < mbpo < tdmpc2 | 2.353 | 3.611 | 2.684 | tdmpc2 < sac < td3 < mbpo |
| tail_only | HalfCheetah-v5 | gross | td3 < sac < mbpo < tdmpc2 | 2.167 | 3.405 | 2.156 | tdmpc2 < sac < td3 < mbpo |
| tail_only | HalfCheetah-v5 | net | td3 < sac < mbpo < tdmpc2 | 2.441 | 3.024 | 2.925 | tdmpc2 < sac < td3 < mbpo |

**Measured-segment shares of training energy, gross vs net (primary variant; per-run mean shares):**

| algo | env | segment | share_gross_pct | share_net_pct |
|---|---|---|---|---|
| mbpo | Ant-v5 | dynamics_model_update | 74.28 | 70.34 |
| mbpo | Ant-v5 | gradient_updates | 23.73 | 27.72 |
| mbpo | Ant-v5 | rollout | 1.7 | 1.61 |
| mbpo | Ant-v5 | synthetic_rollout_generation | 0.29 | 0.34 |
| mbpo | HalfCheetah-v5 | dynamics_model_update | 70.46 | 66.07 |
| mbpo | HalfCheetah-v5 | gradient_updates | 28.33 | 32.77 |
| mbpo | HalfCheetah-v5 | rollout | 1.17 | 1.12 |
| mbpo | HalfCheetah-v5 | synthetic_rollout_generation | 0.04 | 0.04 |
| sac | Ant-v5 | gradient_updates | 93.27 | 94.44 |
| sac | Ant-v5 | rollout | 6.73 | 5.56 |
| sac | HalfCheetah-v5 | gradient_updates | 95.9 | 96.54 |
| sac | HalfCheetah-v5 | rollout | 4.1 | 3.46 |
| td3 | Ant-v5 | gradient_updates | 89.94 | 91.04 |
| td3 | Ant-v5 | rollout | 10.06 | 8.96 |
| td3 | HalfCheetah-v5 | gradient_updates | 94.98 | 95.36 |
| td3 | HalfCheetah-v5 | rollout | 5.02 | 4.64 |
| tdmpc2 | Ant-v5 | gradient_updates | 46.18 | 44.1 |
| tdmpc2 | Ant-v5 | rollout | 51.62 | 53.81 |
| tdmpc2 | Ant-v5 | world_model_pretrain | 2.2 | 2.08 |
| tdmpc2 | HalfCheetah-v5 | gradient_updates | 49.75 | 47.25 |
| tdmpc2 | HalfCheetah-v5 | rollout | 47.83 | 50.48 |
| tdmpc2 | HalfCheetah-v5 | world_model_pretrain | 2.43 | 2.27 |

Facts: the energy ordering TD3 < SAC < MBPO < TD-MPC2 holds on both envs, gross and net, for all three `P_idle` variants. The J/FLOP ordering TD-MPC2 < SAC < TD3 < MBPO also holds gross and net, all variants, both envs. Net/gross total-energy ratio is 0.47–0.48 (TD3, MBPO), 0.52–0.53 (SAC), 0.65 (TD-MPC2). Between-algorithm energy ratios change (e.g. TD-MPC2/MBPO 1.95 → 2.67 on Ant, 2.16 → 2.93 on HC; SAC/TD3 2.11 → 2.35 on Ant). Shares move by ≤ 4.5 pp (largest: MBPO dynamics 70.46 → 66.07 % on HC, gradient_updates 28.33 → 32.77 %).

---

## 6. Execution order and timeline (P1)

`[DERIVED]` `item6_timeline.py`. Source: `start_time_utc` / `end_time_utc` / `git_commit` in all 300 `metadata.json`; config labels from `section_5_2_analysis.load()`. Outputs: `execution_timeline.csv` (sessions), `execution_timeline_runs.csv` (all runs with gap to previous run), `execution_timeline_configs.csv`, `execution_timeline_pairs.csv`, `execution_timeline_seed_order.csv`. A new session starts after a gap > 30 min between the end of one run and the start of the next; sessions were built over all 300 runs (dev runs included, flagged). Dates/times are UTC (local time at the machine is UTC+2: e.g. run `20260907_121335` has directory timestamp 12:13 local and `start_time_utc` 10:13).

**Totals.** The 280 canonical runs span 2026-09-07 10:13 to 2026-09-19 16:10 UTC. **Machine hours** = sum of (`end_time_utc` − `start_time_utc`) over runs (this includes settle, idle head/tail and teardown): **117.72 h for the 280 canonical runs** (130.66 h for all 300). No two runs overlap (minimum gap between runs 0.021 min = 1.3 s). The 280 canonical runs carry 12 distinct `git_commit` values: 0bbbbb7, 58962d2, 788ffcd, 7e672ef, 92dcee0, 9b4b44c, a3582b5, aa9186b, b5b7e54, ecb7135, ecb98bb, f23f448 (see per-session commits below; the commit is the repo HEAD at launch, not a record of which files differed).

**Sessions containing canonical runs** (sessions 1–9 and 13 contain only dev runs, 18 runs, 2026-08-13 to 2026-09-09; sessions 12 and 15 each include one dev run):

| session | first_start_utc | last_end_utc | span_h | n_runs | n_canonical | n_dev | algo_env_configs | git_commits |
|---|---|---|---|---|---|---|---|---|
| 10 | 2026-09-07 10:13 | 2026-09-07 12:10 | 1.94 | 8 | 8 | 0 | sac/HalfCheetah-v5:base; sac/Ant-v5:base | f23f448 |
| 11 | 2026-09-08 09:52 | 2026-09-08 14:26 | 4.57 | 12 | 12 | 0 | sac/Ant-v5:base; sac/HalfCheetah-v5:utd2,utd4 | 92dcee0,ecb98bb |
| 12 | 2026-09-08 17:12 | 2026-09-08 23:34 | 6.37 | 11 | 10 | 1 | mbpo/HalfCheetah-v5:DEV,base; mbpo/Ant-v5:base | 0bbbbb7,cf0f09f |
| 14 | 2026-09-09 10:22 | 2026-09-09 20:34 | 10.21 | 40 | 40 | 0 | td3/HalfCheetah-v5:base,utd2,utd4; td3/Ant-v5:base,utd2,utd4; sac/Ant-v5:utd2,utd4 | 58962d2,aa9186b |
| 15 | 2026-09-10 09:33 | 2026-09-10 17:53 | 8.33 | 11 | 10 | 1 | tdmpc2/HalfCheetah-v5:DEV,base; tdmpc2/Ant-v5:base | 24f41b0,b5b7e54 |
| 16 | 2026-09-12 17:15 | 2026-09-13 10:44 | 17.49 | 60 | 60 | 0 | sac/HalfCheetah-v5:w256,w512; sac/Ant-v5:w256,w512; mbpo/HalfCheetah-v5:w256,w512; mbpo/Ant-v5:w256,w512; td3/HalfCheetah-v5:w256,w512; td3/Ant-v5:w256,w512 | 7e672ef |
| 17 | 2026-09-14 10:50 | 2026-09-15 03:09 | 16.32 | 20 | 20 | 0 | mbpo/HalfCheetah-v5:utd2,utd4; mbpo/Ant-v5:utd2,utd4 | ecb7135 |
| 18 | 2026-09-15 10:39 | 2026-09-15 19:32 | 8.88 | 35 | 35 | 0 | sac/HalfCheetah-v5:b512,b1024; mbpo/HalfCheetah-v5:b512,b1024; td3/HalfCheetah-v5:b256,b512,b1024 | 9b4b44c |
| 19 | 2026-09-17 10:10 | 2026-09-18 02:15 | 16.09 | 45 | 45 | 0 | sac/Ant-v5:b512,b1024; mbpo/Ant-v5:b512,b1024,rollout1,rollout15; td3/Ant-v5:b256,b512,b1024 | 788ffcd |
| 20 | 2026-09-18 10:21 | 2026-09-19 16:10 | 29.83 | 40 | 40 | 0 | tdmpc2/HalfCheetah-v5:numq3,numq7,horizon1,horizon5; tdmpc2/Ant-v5:numq3,numq7,horizon1,horizon5 | a3582b5 |

**Baseline vs sweep (48 pairs = every non-baseline config vs its baseline, same algo×env; `execution_timeline_pairs.csv`).** 44 of 48 ran on a different UTC date than their baseline (gap 5.6–239.6 h between first starts) and 44 of 48 in a different session. The only same-day pairs are the 4 TD3 UTD pairs, all in session 14 together with the TD3 baseline:

| algo | env | config_b | date_a | date_b | same_session | hours_apart |
|---|---|---|---|---|---|---|
| td3 | Ant-v5 | utd2 | 2026-09-09 | 2026-09-09 | True | 7.24 |
| td3 | Ant-v5 | utd4 | 2026-09-09 | 2026-09-09 | True | 8.14 |
| td3 | HalfCheetah-v5 | utd2 | 2026-09-09 | 2026-09-09 | True | 5.55 |
| td3 | HalfCheetah-v5 | utd4 | 2026-09-09 | 2026-09-09 | True | 6.43 |

**HalfCheetah vs Ant pairs (27 paired configs, as in `env_comparison_paired.csv`).** 11 of 27 pairs have their two env legs on different UTC dates, 9 in different sessions (the difference between 11 and 9 is overnight sessions spanning midnight, e.g. MBPO w256/w512). The pairs on different days:

| algo | config_a | date_a | date_b | same_session | hours_apart |
|---|---|---|---|---|---|
| mbpo | w256 | 2026-09-12 | 2026-09-13 | True | 5.2 |
| mbpo | b512 | 2026-09-15 | 2026-09-17 | False | 47.61 |
| mbpo | w512 | 2026-09-12 | 2026-09-13 | True | 5.74 |
| mbpo | b1024 | 2026-09-15 | 2026-09-17 | False | 48.04 |
| sac | utd2 | 2026-09-08 | 2026-09-09 | False | 25.38 |
| sac | utd4 | 2026-09-08 | 2026-09-09 | False | 25.42 |
| sac | b512 | 2026-09-15 | 2026-09-17 | False | 47.53 |
| sac | b1024 | 2026-09-15 | 2026-09-17 | False | 47.56 |
| td3 | b256 | 2026-09-15 | 2026-09-17 | False | 48.54 |
| td3 | b512 | 2026-09-15 | 2026-09-17 | False | 48.57 |
| td3 | b1024 | 2026-09-15 | 2026-09-17 | False | 48.59 |

**Seed and config order.**
- Seeds: for **56 of 56 canonical configs** the five seeds ran in the order of the seed list 331, 958, 14577, 43611, 85062, and for **56 of 56** they ran as five consecutive runs with no other run (canonical or dev) in between (`execution_timeline_seed_order.csv`). **Confirmed.**
- Config order within a sweep: **fixed, not randomised.** `grep -ln "shuf\|RANDOM\|sort -R" scripts/*.sh` returns nothing; the sweep scripts iterate nested `for` loops in listed order (e.g. `scripts/run_width_sweep.sh`: combos → widths → seeds; `run_batch_size_sweep.sh`: algo → batch → seed), and the observed run order matches (e.g. session 16: sac-HC w256, w512, sac-Ant w256, w512, mbpo-HC, mbpo-Ant, td3-HC, td3-Ant; session 18: sac, mbpo, td3 on HC with b512, b1024 (td3: b256, b512, b1024)). **Confirmed.**
- Drift/order confound: because the order is fixed, the config is confounded with time-of-run (baseline configs ran 2026-09-07 to 09-10; sweep configs ran from 2026-09-08 (SAC/TD3 UTD) through 2026-09-19, the width, MBPO-UTD, batch-size, Ant-rollout and TD-MPC2 sweeps on 09-12 to 09-19; within a sweep, algorithm and env follow loop order). This is a fact about the schedule; its effect size is not estimated here (see item 5a for idle-power drift).
- TD-MPC2 num_q/horizon sweep false start (described in `CLAUDE.md`; the log `results/tdmpc2/_numq_horizon_sweep.log` was not re-read here): session 20 contains exactly 40 runs, all canonical TD-MPC2 sweep configs, so no extra/partial run directory from the false start is present.

---

## 7. Machine and software facts (P1)

Only what the run data record could be collected here (see scope note). Per-run table: `item7_per_run_versions.csv` (script `item7_versions.py`, from `metadata.json` + the first row of `emissions_base_*.csv` of each of the 280 canonical runs).

**Unique values across the 280 canonical runs `[FOUND]`** (one unique value per field, i.e. identical across all runs):

| Field | Value | Source |
|---|---|---|
| PyTorch | `2.13.0+cu130` (CUDA 13.0 build tag; this is the wheel's build tag, not a measured runtime version) | `metadata.json: torch_version`; unique count 1 |
| CodeCarbon | `3.3.0` | `metadata.json` and `emissions_base_*.csv: codecarbon_version`; unique count 1 each |
| Python | `3.14.4` | `emissions_base_*.csv: python_version`; unique count 1 |
| OS string | `Linux-7.0.0-31-generic-x86_64-with-glibc2.43` | `emissions_base_*.csv: os`; unique count 1 |
| CPU | `Intel(R) Core(TM) Ultra 9 285K`, `cpu_count` 24 | `emissions_base_*.csv` |
| GPU | `NVIDIA GeForce RTX 5090` (`gpu_count` 1) | `metadata.json: cuda_device_name` |
| RAM total | 60.816 GB (`ram_total_size` as CodeCarbon reports it) | `emissions_base_*.csv` |
| Tracking mode | `machine` | `emissions_base_*.csv` |
| Device | `cuda` | `metadata.json` |

`[NOT FOUND]` Gymnasium, MuJoCo, `nvidia-ml-py` versions on the research PC: `metadata.json` keys are only `algo_config,algo_name,codecarbon_version,cuda_device_name,device,end_time_utc,env_id,experiment_config,git_commit,seed,start_time_utc,thermal_gate,torch_version`. (`contexts/section_4.1.md` lists gymnasium 1.3.0 / mujoco 3.12.0, but those are from the local Windows venv, not the Linux box.) `rl-exp/bin/pip freeze` could not be run here.

Note: `contexts/Section_6.md` B4 quotes the kernel string `7.0.0-30-generic`; that value belongs to 7 non-canonical dev runs. All 280 canonical runs record `7.0.0-31-generic` (checked over all 300 `emissions_base` files: 280 canonical → `-31`; dev runs: 7 → `-30`, 3 → `-31`, 10 → Windows 11 `10.0.26200`, SAC dev runs from August on the same CPU/GPU).

**OS / Debian check (draft table says Debian)** — `[NOT FOUND]` `/etc/os-release`, `uname -r`; `[UNVERIFIED]` (U1). What is recorded: `Linux-7.0.0-31-generic-x86_64-with-glibc2.43`; no distribution name.

**GPU / driver** — `[NOT FOUND]` driver version, open kernel modules, VBIOS (N3); GPU name `[FOUND]` NVIDIA GeForce RTX 5090. **Power/clock limits and supported clock range** (is 200 MHz inside it) — `[NOT FOUND]` (N4). What the repo shows: GPU clocks were locked via `nvidia-smi` at 2000/2000 MHz for all 280 canonical runs (`experiment_config.gpu_min_clock_mhz/gpu_max_clock_mhz`); the one clock-study run was locked at 200/200 and completed, i.e. the driver accepted 200 MHz, which is evidence but not proof that 200 MHz is a supported clock (`results/mbpo/HalfCheetah-v5/seed_0/20260904_130915`, `metadata.json`). Command to run on the research PC: `nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS` and `nvidia-smi --query-gpu=name,driver_version,vbios_version,power.limit,power.default_limit,power.max_limit,clocks.max.graphics --format=csv`.

**Peak FP32 TFLOP/s at 2000 MHz** — `[NOT FOUND]` inputs (SM count, compute capability, FP32 lanes per SM). Formula to apply once `torch.cuda.get_device_properties(0).multi_processor_count` is read: `peak_FP32 [FLOP/s] = n_SM × FP32_lanes_per_SM × 2 [FLOP per FMA] × 2.000e9 [Hz]`; the lanes-per-SM figure for compute capability must be taken from NVIDIA's documentation for the reported capability, not from memory. Not computed.

**FP32 / TF32** — `[UNVERIFIED]` (U2). Repo grep result: no setting of `allow_tf32`, `set_float32_matmul_precision`, `autocast`, `bfloat16`, `float16` or `torch.compile` in any code path (only a docstring mention of `torch.compile` in `algorithms/tdmpc2.py:53` saying it is *not* used). Local default values seen: precision `'highest'`, `matmul.allow_tf32 = False` (torch 2.14.0+cpu, Windows venv only).

**RAM type/speed, mainboard model, storage** — `[NOT FOUND]` (N6). Recorded: RAM total 60.8 GB (CodeCarbon `ram_total_size`); CPU/threads as in the table.

**Commands to run on the research PC** (read-only): `cat /etc/os-release; uname -r; rl-exp/bin/pip freeze | grep -iE "torch|gymnasium|mujoco|nvidia|codecarbon|numpy|pandas"; nvidia-smi; nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS; lscpu; free -h; lsblk -d -o NAME,MODEL; cat /sys/class/dmi/id/board_name /sys/class/dmi/id/board_vendor; rl-exp/bin/python -c "import torch;p=torch.cuda.get_device_properties(0);print(p.name,p.multi_processor_count,p.major,p.minor,torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32)"`.

---

## 8. Pointers to result files (P2)

No recomputation. Sizes in bytes, `rows` = data rows (lines − 1). "Latest `flop_analysis/*.py` change" is commit `050b3c3` (2026-10-08), which only **added** `section_5_2_analysis.py`, `section_5_2_render.py`, `section_5_3_analysis.py`, `section_5_3_render.py`; the pipeline scripts `compute_energy_per_flop.py`, `flop_keys.py`, `measure_flops.py` and `flops_per_call.json` last changed in `1f93741` (2026-09-28), the same commit that last regenerated `flop_analysis/output/*.csv`, so the pipeline CSVs are **not** older than the code that produces them. `correlation_analysis.py` last changed `4247c30` (2026-09-28) and its outputs `4247c30`. Files under `contexts/Section_6*` are untracked (no commit).

| What | Path | Rows | Bytes | Key columns | Commit (date) |
|---|---|---|---|---|---|
| (a) per-config cross-seed energy and J/FLOP, 56 configs | `contexts/section_5.2_data/d1_config_table.csv` | 56 | 32 547 | `algo,env,config,…,n_seeds,total_E_J,total_E_sd_J,total_F,total_J_per_F,total_J_per_F_sd,{rollout,gradient_updates,dynamics_model_update,synthetic_rollout_generation,world_model_pretrain}_{E_J,E_sd_J,F,J_per_F,…}` | `050b3c3` (2026-10-08) |
| (a) compact | `contexts/section_5.3_data/t2_all_config_values.csv` | 56 | 3 562 | `algo,env,config,E_mean,jpf` | `050b3c3` |
| (a) pipeline source, per segment | `flop_analysis/output/cross_seed_energy_per_flop.csv` | 378 | 100 372 | `algo,env_id,architecture_signature,…,segment,flop_type,n_seeds,mean_energy_joules,mean_duration_s,mean_power_w,total_flops,mean_energy_per_flop_j_per_flop` | `1f93741` (2026-09-28) |
| (a) per-run source | `flop_analysis/output/per_run_energy_per_flop.csv` | 2917 | 1 253 232 | `algo,env_id,architecture_signature,seed,run_dir,segment,flop_type,call_count,total_flops,total_energy_joules,duration_s,mean_power_w,energy_per_flop_j_per_flop` | `1f93741` |
| (b) orderings (energy and J/FLOP) | `contexts/section_5.3_data/o1_orderings.csv` | 96 | 12 781 | `env,sweep,sweep_label,metric,rank,algo,config,mean,sd,cv_pct,ratio_to_previous,ordering` | `050b3c3` |
| (b) swaps vs baseline | `contexts/section_5.3_data/o2_swaps_vs_baseline.csv` | 32 | 2 460 | `env,sweep,metric,baseline_ordering,ordering,same_as_baseline,swapped_pairs` | `050b3c3` |
| (b) seed-level orderings | `contexts/section_5.3_data/o3_seed_level.csv` (+ `o3b_pairwise_seed_counts.csv`) | 32 (+96) | 4 842 | `…,seeds_same_as_mean_ordering,seed_331,…,seed_85062` | `050b3c3` |
| (c) env comparison, 27 pairs | `contexts/section_5.4_data/env_comparison_paired.csv` (+ `pooled_summary.csv`, 5 rows) | 27 | 23 779 | `algo,config,E_total_ratio_of_means,E_rollout_ratio_of_means,dshare_*,P_total_*,jpf_total_{HC,Ant}_J_per_FLOP,flops_total_{HC,Ant},git_commits_{HC,Ant},…` | `050b3c3` |
| (d) pooled Spearman + log-log (total) | `contexts/section_5.2_data/d2_total_correlation.csv` | 2 | 581 | `level,env,algo,segment,n,spearman_rho,spearman_p,loglog_slope,loglog_slope_se,loglog_r2,flops_min,flops_max` | `050b3c3` |
| (d) per-algorithm total | `contexts/section_5.2_data/d4_per_algo_total_correlation.csv` | 8 | 1 992 | same | `050b3c3` |
| (d) per-segment (pooled / per-algo) | `d3_segment_correlation.csv` (10 rows), `d5_per_algo_segment_correlation.csv` (22 rows) | — | 2 449 / 5 346 | same | `050b3c3` |
| (d) return-vs-FLOPs and energy-vs-FLOPs with bootstrap CIs | `flop_analysis/output/correlation/correlation_summary.csv` (+ `config_means.csv`, 56 rows, `correlation_plots.html`) | 4 | 708 | `env,analysis,n_configs,spearman_rho,ci_low,ci_high,p_value,loglog_slope,slope_ci_low,slope_ci_high,r2` | `4247c30` (2026-09-28) |
| (e) fixed/marginal fits | `contexts/section_5.2_data/a1_linear_fits.csv` | 144 | 22 702 | `algo,env,sweep,segment,configs,values,n,resid_df,a_J,b_J_per_FLOP,b_pJ_per_FLOP,r2,rse_J,fit_basis` | `050b3c3` |
| (e) delta pairs | `contexts/section_5.2_data/a2_delta_pairs.csv` | 150 | 27 382 | `algo,env,sweep,segment,from_value,to_value,from_config,to_config,E_from_J,E_to_J,dE_J,F_from,F_to,dF,dE_per_dF_pJ,status` | `050b3c3` |
| (f) clock study | `contexts/Section_6_data/C_segment_summary.csv` (24 rows, 4 200 B: `clock,segment,kind,energy_j,…,share_of_training_total,duration_s,mean_power_w,total_flops,…`), `D_ratios_200_over_2000.csv` (9 rows, 2 560 B), `C_consistency_checks.csv`, `C5_whole_run_row.csv`; narrative `contexts/Section_6.md` (427 lines, 53 359 B) | | | | **untracked** (no commit) |

`[NOT FOUND]`: none of the (a)–(f) files is missing. Two caveats: (1) the clock-study files are untracked in git; (2) `contexts/section_5.4_data/env_comparison_paired.csv` records the HC/Ant `git_commits` per pair, and 12 distinct launch commits occur among the canonical runs (item 6), so "same code" across sweeps rests on the `algo_config_diff_HC_vs_Ant`/`experiment_config_diff_HC_vs_Ant` columns there, not on commit equality.

---

## 9. Thesis sources and bibliography (P2)

`[FOUND]` The thesis is at `C:\Users\saman\Desktop\Thesis\Thesis Report\` (searched `Desktop`, `Documents`, `Downloads` under the home directory; `Documents` and `Downloads` contain no `.tex`/`.bib`). Files: `thesis-main.tex` (6710 lines, 581 KB, modified 2026-10-11 10:23, **this is the working thesis**: title "Energy Profiling of Reinforcement Learning via Algorithmic Decomposition…"), `bibliography.bib` (modified 2026-10-11 10:11), plus template leftovers `main-english.tex` and `main-english-university-of-hamburg.tex` (unmodified template, dated 2026-09-28), `commands.tex`, `abbreviations.tex`. Also found but unrelated: `Desktop\Thesis\Project Proposal\{project-proposal.tex, preamble.tex, lit.bib}`, and two seminar projects under `Desktop\Uni Stuttgart\WS25\` (not analysed). Script `item9_tex.py` (comments after unescaped `%` stripped; reads only `thesis-main.tex` and `bibliography.bib`; nothing in the thesis folder was modified).

Chapters in `thesis-main.tex`: Introduction (l.1019), LaTeX Hints (l.1349, template text), Problem Exposition and Method (l.2089), Per-Algorithm Energy Analysis (l.3123), Cross-Algorithm Comparison (l.5412), GPU Clock Frequency (l.6436), My first appendix (l.6702, template). **No Evaluation chapter exists yet** (`ch:evaluation` is referenced but undefined). Everything lives in one file, so "grouped by file" = one file, grouped by chapter. 214 labels, no duplicate labels (`item9_labels.csv`).

**`\label` names by chapter** (all in `thesis-main.tex`):

- **Introduction** (24): `sec:introduction`, `sec:introduction:topic`, `sec:introduction:state-of-art`, `sec:introduction:research-question`, `sec:introduction:method`, `sec:introduction:results`, `sec:background`, `sec:problem-exposition`, `sec:problem-exposition:context-understanding`, `sec:problem-exposition:data-understanding`, `sec:problem-exposition:research-problems`, `sec:problem-exposition:research-method`, `sec:problem1`, `sec:problem1:subproblem1`, `fig:my_label`, `sec:problem1:subproblem2`, `sec:sub-problem2`, `sec:evaluation`, `sec:evaluation:objective`, `sec:evaluation:setup`, `sec:evaluation:execution`, `sec:evaluation:results`, `fig:my_label2`, `sec:evaluation:discussion`
- **LaTeX Hints** (26): `sec:latexhints`, `sec:ex:cref`, `fig:ex:cref`, `tab:ex:cref`, `eq:test`, `fig:label`, `fig:jpg`, `fig:first_case`, `fig:second_case`, `fig:two_sub_figures`, `fig:tikz_example`, `fig:pgfplots_example`, `fig:pgfplots_coords`, `tab:simple`, `tab:diag`, `tab:long`, `line:comment`, `sec:ex:vref`, `ssec:example`, `fig:Abbildung`, `fig:AbbildungGedreht`, `alg:example`, `alg1:state2`, `alg:example2`, `alg2:state2`, `def:def1`
- **Problem Exposition and Method** (50): `ch:method`, `sec:context`, `sec:metric`, `tab:spearman`, `sec:rqs`, `tab:rqs`, `sec:harness`, `sec:protocol`, `fig:protocol`, `tab:phases`, `sec:epoch-tasks`, `sec:measured-allocated`, `eq:allocation`, `eq:allocated-jpf`, `tab:status`, `sec:alg-segments`, `tab:segment-meaning`, `sec:confounds`, `sec:units`, `sec:flop-accounting`, `sec:flop-principle`, `eq:flops-segment`, `sec:flop-per-call`, `sec:flop-counts`, `eq:call-counts`, `sec:flop-signatures`, `sec:flop-jpf`, `eq:jpf-segment`, `tab:flop-sources`, `eq:jpf-total`, `sec:flop-caveats`, `sec:flop-verification`, `fig:flop-pipeline`, `sec:design`, `sec:environments`, `tab:envs`, `sec:seeds`, `sec:baselines`, `tab:baseline-config`, `sec:sweeps`, `tab:sweep-matrix`, `tab:run-decomposition`, `sec:dev-runs`, `sec:execution`, `sec:setup`, `tab:hardware`, `tab:software`, `sec:launch`, `tab:sweep-status`, `sec:data-quality`
- **Per-Algorithm Energy Analysis** (64): `ch:per-algorithm`, `sec:sac`, `sec:sac-baseline`, `tab:sac-base-energy`, `tab:sac-base-jpf`, `tab:sac-base-time`, `tab:sac-base-comp`, `tab:sac-quality`, `sec:sac-utd`, `tab:sac-utd-energy`, `tab:sac-utd-jpf`, `sec:sac-width`, `tab:sac-width-energy`, `tab:sac-width-jpf`, `sec:sac-batch`, `tab:sac-batch-energy`, `tab:sac-batch-jpf`, `tab:sac-env`, `sec:sac-interpretation`, `sec:td3`, `sec:td3-baseline`, `tab:td3-base-energy`, `tab:td3-base-jpf`, `tab:td3-base-time`, `tab:td3-base-comp`, `tab:td3-quality`, `sec:mbpo`, `sec:mbpo-baseline`, `tab:mbpo-base-energy`, `tab:mbpo-base-jpf`, `tab:mbpo-base-time`, `tab:mbpo-fit`, `tab:mbpo-quality`, `sec:mbpo-utd`, `tab:mbpo-utd-energy`, `tab:mbpo-utd-jpf`, `sec:mbpo-width`, `tab:mbpo-width-energy`, `tab:mbpo-width-jpf`, `sec:mbpo-batch`, `tab:mbpo-batch-energy`, `tab:mbpo-batch-jpf`, `tab:mbpo-env`, `tab:mbpo-sac`, `sec:mbpo-rollout`, `tab:mbpo-rollout-energy`, `tab:mbpo-rollout-jpf`, `sec:mbpo-interpretation`, `sec:tdmpc2`, `sec:tdmpc2-baseline`, `tab:tdmpc2-base-energy`, `tab:tdmpc2-base-jpf`, `tab:tdmpc2-base-time`, `tab:tdmpc2-base-comp`, `tab:tdmpc2-quality`, `sec:tdmpc2-pretrain`, `tab:tdmpc2-pretrain`, `sec:tdmpc2-numq`, `tab:tdmpc2-numq-energy`, `tab:tdmpc2-numq-jpf`, `sec:tdmpc2-horizon`, `tab:tdmpc2-hor-energy`, `tab:tdmpc2-hor-jpf`, `sec:tdmpc2-interpretation`
- **Cross-Algorithm Comparison** (40): `ch:comparison`, `sec:segment-shares`, `tab:cmp-shares`, `tab:cmp-energy`, `sec:energy-vs-compute`, `sec:ec-data`, `eq:fixed-marginal`, `sec:ec-predictor`, `tab:ec-pooled`, `fig:ec-scatter`, `tab:ec-algo`, `tab:ec-equal`, `sec:ec-fixed-marginal`, `tab:ec-fits`, `tab:ec-delta`, `sec:ec-interpretation`, `sec:robustness`, `sec:rob-basis`, `sec:rob-energy`, `tab:rob-energy`, `sec:rob-jpf`, `tab:rob-jpf`, `sec:rob-rollout`, `tab:rob-rollout`, `sec:rob-tdmpc2`, `tab:rob-tdmpc2`, `tab:rob-ranges`, `sec:rob-summary`, `sec:env-effect`, `sec:env-basis`, `sec:env-energy`, `tab:env-baseline`, `tab:env-pooled`, `sec:env-shares`, `sec:env-power`, `sec:env-confounds`, `tab:env-mbpo-matched`, `sec:env-episodes`, `tab:env-episodes`, `sec:env-summary`
- **GPU Clock Frequency** (9): `ch:clock`, `sec:clock-scope`, `sec:clock-setup`, `sec:clock-results`, `tab:clock-headline`, `tab:clock-segments`, `tab:clock-components`, `sec:clock-discussion`, `sec:clock-summary`
- **My first appendix** (1): `sec:appendix1`

**Citation keys in `bibliography.bib`:** 65 entries but only **41 unique keys**. The following keys are defined more than once (number of entries): `chua2018pets` (3), `fujimoto2018td3` (5), `haarnoja2018sac` (6), `hansen2024tdmpc2` (5), `henderson2018matters` (2), `janner2019mbpo` (6), `williams2017mppi` (4). Duplicate keys in a biber `.bib` file are an issue the bibliography owner should resolve (biber keeps one and warns). Unique keys (`item9_bibkeys.csv` has all entries):

`Binz2009`, `Lichter2026`, `Winery`, `abe2014gpu`, `agarwal2021precipice`, `chen2021redq`, `chua2018pets`, `codecarbon`, `colas2018seeds`, `courty2024codecarbon`, `desislavov2023trends`, `fujimoto2018td3`, `getzner2023accuracy`, `gymnasium_ant`, `haarnoja2018sac`, `hansen2022tdmpc`, `hansen2024tdmpc2`, `henderson2018matters`, `henderson2020systematic`, `janner2019mbpo`, `khan2018rapl`, `lillicrap2016ddpg`, `mei2017dvfs`, `mwe`, `nvml`, `paszke2019pytorch`, `patterson2021carbon`, `pytorchflopcounter`, `schulman2016gae`, `schwartz2020green`, `silver2014dpg`, `spearman1904proof`, `strubell2019energy`, `todorov2012mujoco`, `towers2024gymnasium`, `vanhasselt2010doubleq`, `virtanen2020scipy`, `wawrzynski2009cat`, `williams2009roofline`, `williams2017mppi`, `you2023zeus`

**Cite keys used but not defined (`\cite`-family, from `thesis-main.tex` vs the bib): none.** Cited unique keys: 39. Defined but never cited: `Binz2009`, `Winery`.

**`\zcref` / `\ref`-family keys used but not defined as a `\label`** (`item9_undefined_refs.csv` has every occurrence with line number; `#3` is a macro argument in the preamble, `lst:*` are in the template chapter "LaTeX Hints"):

| key | n | first_line | chapters |
|---|---|---|---|
| #3 | 2 | 684 | (preamble) |
| app:clock-details | 1 | 6543 | GPU Clock Frequency |
| app:clock-provenance | 2 | 6480 | GPU Clock Frequency |
| app:flop-details | 2 | 2647 | Problem Exposition and Method |
| app:full-results | 12 | 3175 | Cross-Algorithm Comparison, Per-Algorithm Energy Analysis |
| app:hyperparameters | 2 | 2890 | Problem Exposition and Method |
| app:learning | 6 | 2182 | Per-Algorithm Energy Analysis, Problem Exposition and Method |
| app:reproducibility | 4 | 3094 | Cross-Algorithm Comparison, Per-Algorithm Energy Analysis, Problem Exposition and Method |
| ch:conclusion | 1 | 2185 | Problem Exposition and Method |
| ch:evaluation | 7 | 2265 | Cross-Algorithm Comparison, Problem Exposition and Method |
| lst:XML | 1 | 1739 | LaTeX Hints |
| lst:flXML | 1 | 1754 | LaTeX Hints |
| lst:java | 1 | 1785 | LaTeX Hints |
| lst:json | 1 | 1771 | LaTeX Hints |
| sec:allocation | 1 | 6544 | GPU Clock Frequency |
| sec:compute-accounting | 2 | 2590 | Problem Exposition and Method |
| sec:energy-definitions | 1 | 6454 | GPU Clock Frequency |
| sec:hardware | 1 | 6462 | GPU Clock Frequency |
| sec:ram-heuristic | 1 | 6561 | GPU Clock Frequency |
| sec:related | 2 | 4768 | Per-Algorithm Energy Analysis |
| sec:related-flop | 1 | 5882 | Cross-Algorithm Comparison |
| sec:research-questions | 1 | 6439 | GPU Clock Frequency |
| sec:td3-batch | 2 | 3730 | Per-Algorithm Energy Analysis |
| sec:td3-interpretation | 2 | 3731 | Per-Algorithm Energy Analysis |
| sec:td3-utd | 1 | 3730 | Per-Algorithm Energy Analysis |
| sec:td3-width | 1 | 3730 | Per-Algorithm Energy Analysis |
| sec:threats | 21 | 2196 | Cross-Algorithm Comparison, Per-Algorithm Energy Analysis, Problem Exposition and Method |

Interpretation-free summary: the real undefined targets are forward references to chapters/sections/appendices not yet written (`ch:evaluation`, `ch:conclusion`, `sec:threats` (21 uses), `app:*`, `sec:related`, `sec:td3-*`, …).

---

## Final checklist

- Summary block lists every `[NOT FOUND]`, `[UNVERIFIED]`, `[MISMATCH]`: yes (top of file).
- Edits outside `contexts/`: none. Nothing under `results/`, `flop_analysis/`, `configs/` was modified; the thesis folder was only read.
- No `sudo`, no training runs, no clock/governor changes; no `nvidia-smi` calls (the local GPU is not the research GPU).
- Scripts and CSVs: `contexts/section_7_data/` (`build_section_7_md.py` regenerates this file from them).
