# Section 6 — GPU clock frequency (RQ4) — context

Facts, numbers and provenance only — no interpretation, no LaTeX. Two individual runs (n = 1 each): MBPO, HalfCheetah-v5, development seed 0, GPU clock lock 200/200 MHz vs 2000/2000 MHz. **Gross** energies (idle floor included). Numbers are copied unrounded (Python `repr`), joules/kWh/W/s/MHz. `NOT FOUND` = looked for, not in the repo (places listed). `DERIVED` = plain arithmetic, formula given.

Generated 2026-10-11 from the Windows dev checkout (git HEAD `050b3c3`, branch master) after confirming `results/` is present (300 `metadata.json`). Nothing under `results/`, code or configs was modified; only `contexts/` was written. Reproduce: `.venv/Scripts/python -W ignore contexts/Section_6_data/build_section_6.py` then `.venv/Scripts/python -W ignore contexts/Section_6_data/render_section_6.py`.

Notation: **R2000** = `results/mbpo/HalfCheetah-v5/seed_0/20260904_122958`; **R200** = `results/mbpo/HalfCheetah-v5/seed_0/20260904_130915`. Companion CSVs are in `contexts/Section_6_data/` (named per item).

## 1. Run identification (item A)

**A1. Runs whose recorded lock is not 2000/2000 MHz** (`metadata.json` → `experiment_config.gpu_min_clock_mhz` / `gpu_max_clock_mhz`; `A1_non_2000_lock_runs.csv`): **16** runs. Exactly **one** has an explicit non-2000 value: R200 (200/200). The other 15 have `null`/`null` (SAC dev runs, seed 0/42/56 HalfCheetah-v5 and seed 56 Humanoid-v5, 2026-08-13 to 2026-09-02). `CLAUDE.md` ("exactly one run … 200/200") is correct for the explicit value. `CLAUDE.md` also says the 15 `null` runs are "functionally identical" to 2000/2000 runs; the code history says otherwise (see B1 and §6) — so the claim that exactly one run deviates from 2000 MHz is not established for those 15.

| run_dir | algo | env | seed | gpu_min_clock_mhz | gpu_max_clock_mhz | lock_gpu_clocks |
|---|---|---|---|---|---|---|
| results/mbpo/HalfCheetah-v5/seed_0/20260904_130915 | mbpo | HalfCheetah-v5 | 0 | 200.0 | 200.0 | True |
| results/sac/HalfCheetah-v5/seed_0/20260813_112621 | sac | HalfCheetah-v5 | 0 |  |  | True |
| results/sac/HalfCheetah-v5/seed_0/20260813_121500 | sac | HalfCheetah-v5 | 0 |  |  | True |
| results/sac/HalfCheetah-v5/seed_0/20260819_104727 | sac | HalfCheetah-v5 | 0 |  |  | True |
| results/sac/HalfCheetah-v5/seed_0/20260819_115112 | sac | HalfCheetah-v5 | 0 |  |  | True |
| results/sac/HalfCheetah-v5/seed_0/20260821_105506 | sac | HalfCheetah-v5 | 0 |  |  | True |
| results/sac/HalfCheetah-v5/seed_0/20260824_110200 | sac | HalfCheetah-v5 | 0 |  |  | True |
| results/sac/HalfCheetah-v5/seed_0/20260824_115117 | sac | HalfCheetah-v5 | 0 |  |  | True |
| results/sac/HalfCheetah-v5/seed_42/20260824_122611 | sac | HalfCheetah-v5 | 42 |  |  | True |
| results/sac/HalfCheetah-v5/seed_42/20260824_130005 | sac | HalfCheetah-v5 | 42 |  |  | True |
| results/sac/HalfCheetah-v5/seed_42/20260826_111906 | sac | HalfCheetah-v5 | 42 |  |  | True |
| results/sac/HalfCheetah-v5/seed_56/20260831_131353 | sac | HalfCheetah-v5 | 56 |  |  | True |
| results/sac/HalfCheetah-v5/seed_56/20260831_135517 | sac | HalfCheetah-v5 | 56 |  |  | True |
| results/sac/HalfCheetah-v5/seed_56/20260831_143605 | sac | HalfCheetah-v5 | 56 |  |  | True |
| results/sac/Humanoid-v5/seed_56/20260902_131059 | sac | Humanoid-v5 | 56 |  |  | True |
| results/sac/Humanoid-v5/seed_56/20260902_135248 | sac | Humanoid-v5 | 56 |  |  | True |

**A2. All MBPO / HalfCheetah-v5 / seed-0 runs** (`A2_mbpo_hc_seed0_runs.csv`; exactly two exist):

| timestamp | git_commit | gpu_min_clock_mhz | gpu_max_clock_mhz | hidden_sizes | batch_size | updates_per_env_step | rollout_min_length | rollout_max_length | rollout_min_epoch | rollout_max_epoch | warmup_steps | train_steps | steps_per_epoch | model_hidden_sizes | torch | codecarbon | gpu | training_metrics_exists |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260904_122958 | ae3880860ebcce0c1e729714cdf48a08039adc46 | 2000 | 2000 | [256, 256] | 256 | 1 | 1 | 1 | 20 | 150 | 5000 | 100000 | 1000 | [200, 200, 200, 200] | 2.13.0+cu130 | 3.3.0 | NVIDIA GeForce RTX 5090 | True |
| 20260904_130915 | ae3880860ebcce0c1e729714cdf48a08039adc46 | 200 | 200 | [256, 256] | 256 | 1 | 1 | 1 | 20 | 150 | 5000 | 100000 | 1000 | [200, 200, 200, 200] | 2.13.0+cu130 | 3.3.0 | NVIDIA GeForce RTX 5090 | True |

**A3. Matched comparator.** Exactly one run matches R200: **R2000**. Same algorithm, environment, seed, git commit, all hyperparameters, software stack; the only differing config fields are the two clock fields (see A4). No other candidate exists (the only other MBPO/HalfCheetah-v5 seed-0 run is R200 itself). Rule 6 is therefore not triggered.

**A4. Field-by-field `metadata.json` diff** (full table with every field incl. identical ones: `A4_metadata_diff.csv`). Fields that differ:

| field | v_2000MHz_run | v_200MHz_run |
|---|---|---|
| end_time_utc | 2026-09-04T11:01:08.461191 | 2026-09-04T12:48:34.409574 |
| experiment_config.gpu_max_clock_mhz | 2000 | 200 |
| experiment_config.gpu_min_clock_mhz | 2000 | 200 |
| start_time_utc | 2026-09-04T10:29:59.128368 | 2026-09-04T11:09:15.306557 |
| thermal_gate.final_power_w | 26.467 | 21.006 |
| thermal_gate.final_temp_c | 45.0 | 44.0 |
| thermal_gate.waited_seconds | 1.1682510375976562e-05 | 1.0728836059570312e-05 |

Not differing: `git_commit` (both `ae3880860ebcce0c1e729714cdf48a08039adc46`), `torch_version` (both `2.13.0+cu130`), `codecarbon_version` (both `3.3.0`), `cuda_device_name` (both `NVIDIA GeForce RTX 5090`), `device`, all of `algo_config` (`hidden_sizes` [256, 256], `batch_size` 256, `updates_per_env_step` 1, `rollout_min/max_length` 1/1, `rollout_min/max_epoch` 20/150, `model_hidden_sizes` [200,200,200,200], …) and all other `experiment_config` fields (`lock_gpu_clocks` true, `set_persistence_mode` true, `set_cpu_performance_governor` true, thermal-gate settings, `warmup_steps` 5000, `train_steps` 100000, `steps_per_epoch` 1000, `settle_seconds` 30.0, idle windows 90.0 s). Software stack (torch, codecarbon) is **identical**. `metadata.json` records no hostname/path fields (NOT FOUND in either file; keys present: algo_name, env_id, seed, git_commit, torch_version, device, cuda_device_name, start_time_utc, experiment_config, algo_config, codecarbon_version, thermal_gate, end_time_utc). Time between runs: R2000 started 2026-09-04T10:29:59 UTC, R200 started 2026-09-04T11:09:15 UTC (same day, R200 after R2000; `start_time_utc`).

**A5. Git commits.** Identical (`ae38808…`), so no diff between commits to report. **But** `metadata.json` stores only `git rev-parse HEAD` (`experiment_runner.py:41-47`, `_git_commit_hash`), not the working-tree state, and `ae38808` (2026-09-02) predates the MBPO code: MBPO was first committed in `f109bd4` (2026-09-04 14:51:27 +0200, "Added new MBPO algo and recorded tests"), which also contains both run directories. So both runs executed uncommitted working-tree code; the recorded commit does not identify their code. Evidence on the clock settings: at `ae38808` and at `f109bd4` the committed `configs/config.py` has `gpu_min_clock_mhz`/`gpu_max_clock_mhz: Optional[int] = None`, which would be written to `metadata.json` as `null`; both runs instead record explicit values (2000 and 200). So the explicit values must have come from a working-tree edit (not from the committed defaults); how/where it was edited is NOT FOUND (no CLI flag exists for the clock values at any commit: `run_experiment.py` only has `--no-gpu-lock`; `git grep` of commit messages for clock/MHz/lock finds nothing). Files read: `git show ae38808:configs/config.py`, `git show f109bd4:configs/config.py`, `git diff ae38808 f109bd4 -- utils/gpu_control.py`.

**A6. Completeness and log** (`A6_completeness_and_log.csv`; `emissions_base_*.csv` task counts by prefix):

| item | R2000 (2000 MHz) | R200 (200 MHz) |
|---|---|---|
| n_rollout | 100 | 100 |
| n_dynamics_model_update | 100 | 100 |
| n_synthetic_rollout_generation | 100 | 100 |
| n_gradient_updates | 100 | 100 |
| n_warmup | 1 | 1 |
| n_idle_baseline_head | 1 | 1 |
| n_idle_baseline_tail | 1 | 1 |
| n_rows_emissions_base | 403 | 403 |
| n_log_lines | 34 | 34 |
| n_WARNING_or_ERROR | 0 | 0 |
| rapl_fallback | False | False |
| geoloc_fallback | False | False |
| thermal_timeout | False | False |
| permission_failure | False | False |
| gpu_control_lines | 0 | 0 |

Both runs have all 100 epochs × {rollout, dynamics_model_update, synthetic_rollout_generation, gradient_updates} plus warmup, idle_baseline_head, idle_baseline_tail (403 task rows each). Both `run.log` files have 34 lines, **0** `[WARNING]`/`[ERROR]` lines, no RAPL-fallback, geolocation-fallback, thermal-gate-timeout or permission-failure strings (regexes in `build_section_6.py`). Both finish with "Run complete. Segment energy summary". **No `gpu_control` lines appear in either `run.log`** (0 matches for "Locked GPU", "Reset GPU", "governor"): see B3. Lines of `run.log` that differ between the two runs after stripping timestamps (`A6_runlog_diff_timestamps_stripped.csv`): run directory name (line 2), thermal gate result (line 5: R2000 waited 1.1682510375976562e-05 s, final_temp_c 45.0, final_power_w 26.467; R200 waited 1.0728836059570312e-05 s, 44.0, 21.006), and the final segment-energy summary lines (22–33). The "Epoch n/100 done" lines are identical in both runs except for timestamps (same real_buffer, rollout_len, cumulative_reward, mean_episode_return).

## 2. Provenance and motivation (item B)

**B1. How the clock lock was set.** No CLI flag sets the clock values: `run_experiment.py:46` `p.add_argument("--no-gpu-lock", action="store_true", help="Disable GPU clock locking.")` is the only clock-related argument; the values come from `ExperimentConfig` (`configs/config.py`) → `GpuCpuGuard(min_mhz=exp_cfg.gpu_min_clock_mhz, max_mhz=exp_cfg.gpu_max_clock_mhz, …)` (`experiment_runner.py:121-126`) → `lock_gpu_clocks()` → `nvidia-smi -i <gpu> -lgc <min>,<max>` (`utils/gpu_control.py`). Config history (`git log -S` / `git diff`):
- `ae38808` and `f109bd4`: `gpu_min_clock_mhz: Optional[int] = None   # None => use gpu_control.DEFAULT_LOCK_MHZ (2000), clamped to device range` (same for max). In `f109bd4` `utils/gpu_control.py` has `DEFAULT_LOCK_MHZ = 200  # fixed clock used when no explicit min/max is given --` / `# representative of a real compute workload, not the idle floor` (the comment says 2000-class workload clock, the constant is 200).
- `f23f448` (2026-09-07 12:13 +0200, "Setting baseline config for all experiments"): `gpu_min_clock_mhz: Optional[int] = 2000`, `gpu_max_clock_mhz: Optional[int] = 2000`, and `DEFAULT_LOCK_MHZ = 2000`. All later (canonical/sweep) runs record 2000/2000.
- `lock_gpu_clocks` from `46f74b9` (2026-07-29) through `ae38808` (2026-09-02; `git log -S qmin`): `min_mhz = min_mhz or qmin` / `max_mhz = max_mhz or qmin  # lock to a single fixed low clock by default` (the lowest supported graphics clock). `DEFAULT_LOCK_MHZ` first appears in `f109bd4`. The 15 `null`-clock SAC dev runs (2026-08-13 to 2026-09-02) fall in the `qmin` era (committed code; later working-tree edits not excluded). `CLAUDE.md`/`README.md` state `null` resolves to `DEFAULT_LOCK_MHZ=2000`. The value those 15 runs actually used is NOT FOUND (not in their `metadata.json`; no `gpu_control` lines in `run.log`; the GPU's lowest supported clock is not recorded in the repo).
- Exact CLI command used for R2000 / R200: NOT FOUND (looked in `metadata.json`, `run.log`, `cli_commands.txt`, `scripts/`). `cli_commands.txt` contains the line `sudo /home/samanth/rl-dissection-linux/rl-exp/bin/python run_experiment.py --algo mbpo --env HalfCheetah-v5 --seed 0` (a generic MBPO HalfCheetah seed-0 invocation, not tied to a clock).

**B2. Why 200 MHz was tried / why 2000 MHz became the lock.** Commit messages near 2026-09-04 / 09-07: `f109bd4` "Added new MBPO algo and recorded tests"; `f23f448` "Setting baseline config for all experiments"; `ae38808` "Humanoid env with 256 and 512 batch sizes". None mentions clocks. `git log --grep` for clock / MHz / frequency / lock returns no commit. The only recorded rationale is the code comment above ("representative of a real compute workload, not the idle floor") and the `DEFAULT_LOCK_MHZ` change 200 → 2000 in `f23f448`. Any statement that the 200 MHz run was an intentional experiment (as the author says) is **NOT FOUND** in the repo; `CLAUDE.md` (lines ~380-384) and `README.md` (~line 750) describe it as an "anomaly" / "locked at the wrong GPU clock" — see §6.

**B3. What `utils/gpu_control.py` does** (as written): `GpuCpuGuard.__enter__`: `nvidia-smi -i 0 -pm 1` (persistence on, if `set_persistence`), `lock_gpu_clocks(min, max)` → `nvidia-smi -i 0 -lgc <min>,<max>` (if min/max is `None`, queries `nvidia-smi --query-supported-clocks=graphics` and uses `min(max(DEFAULT_LOCK_MHZ, qmin), qmax)`), then if `set_governor`: remembers `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` and runs `cpupower frequency-set -g performance`. `__exit__`: `nvidia-smi -i 0 -rgc` (reset clocks), restores the prior CPU governor; persistence mode is deliberately left on. Every command fails soft (logs a warning, run continues). On success it logs `Locked GPU %d clocks to [%d, %d] MHz` via `logging.getLogger("gpu_control")`. **Does that logger write to `run.log`? No, not with the committed code:** `experiment_runner._setup_logging` attaches the `FileHandler(run_dir/"run.log")` and a stdout handler only to the logger named `experiment_runner`; the `gpu_control` logger has no handler of its own and nothing calls `logging.basicConfig`, so its INFO messages are dropped and WARNINGs go only to Python's last-resort stderr handler. Consistent with this, neither run's `run.log` contains any `gpu_control` line. So `run.log` cannot serve as "lock verified" evidence for these runs; the only on-disk record of the lock is `metadata.json` (`lock_gpu_clocks: true`, `gpu_min/max_clock_mhz`), which records the request, not the outcome. Independent evidence that the lock took effect: the measured durations (duration ratio R200/R2000 of the training total in §3 D) — fact only.

**B4. Hardware facts recorded in the repo** (`metadata.json`, `emissions.csv`): GPU `NVIDIA GeForce RTX 5090` (`cuda_device_name`; CodeCarbon `gpu_model` "1 x NVIDIA GeForce RTX 5090", `gpu_count` 1); CPU `Intel(R) Core(TM) Ultra 9 285K`, `cpu_count` 24; `ram_total_size` 60.8161506652832 GB; OS string `Linux-7.0.0-30-generic-x86_64-with-glibc2.43`; Python `3.14.4`; torch `2.13.0+cu130`; codecarbon `3.3.0`. **NOT FOUND**: GPU power limit, supported/max graphics clock, memory clock, driver version (looked in `metadata.json`, `emissions*.csv`, `run.log`, `README.md`, `CLAUDE.md`, `cli_commands.txt`, `requirements.txt`, `utils/`). See §7.

## 3. Results (items C, D, E)

**Definitions** (Section 3.3): energy to completion = `TOTAL_MEASURED_TRAINING` energy; duration = sum of durations of the four measured training tasks; mean power = energy / duration. Source for segment values: `flop_analysis/output/per_run_energy_per_flop.csv` (rows copied in full to `per_run_rows.csv`; both runs are there with `included_in_cross_seed_avg=False`) and `emissions_base_*.csv` (per-task rows, saved to `per_task_rows.csv`). Measured-task values (rollout, dynamics_model_update, synthetic_rollout_generation, gradient_updates) are sums over the 100 per-epoch tasks of `emissions_base_*.csv` (`energy_consumed` × 3.6e6 J/kWh; `duration`).

**C1/C2. Headline and per-segment table** (`C_segment_summary.csv`; `kind` = measured vs. allocated). `share_of_training_total` = segment energy / `TOTAL_MEASURED_TRAINING` energy. For the four **allocated** sub-segments the CSV's `duration_s` is the `time.perf_counter()` sum inside `agent.update()` (`segment_energy.json` → `_sub_segment_wall_time_seconds`) and `mean_power_w` = allocated energy / that wall-time sum, so their power is **not** the power of `gradient_updates` (see §5, G). `target_update`: `energy_per_flop` is not defined here (elementwise, `flop_type=elementwise`, value in the CSV is J per Polyak op); `buffer_sample` has 0 FLOPs.

| segment | quantity | v_2000MHz | v_200MHz |
|---|---|---|---|
| TOTAL_MEASURED_TRAINING | energy_j | 189511.22106781087 | 568209.9518655966 |
| TOTAL_MEASURED_TRAINING | energy_kwh | 0.0526420058521696 | 0.1578360977404435 |
| TOTAL_MEASURED_TRAINING | share_of_training_total | 1.0 | 1.0 |
| TOTAL_MEASURED_TRAINING | duration_s | 1638.1095830429913 | 5731.984642861984 |
| TOTAL_MEASURED_TRAINING | mean_power_w | 115.68897650654756 | 99.12970589919269 |
| TOTAL_MEASURED_TRAINING | total_flops | 315334867200000.0 | 315334867200000.0 |
| TOTAL_MEASURED_TRAINING | energy_per_flop_j_per_flop | 6.009840356398457e-10 | 1.801925543188383e-09 |
| rollout | energy_j | 2285.740545277806 | 3469.4003638811887 |
| rollout | energy_kwh | 0.0006349279292438 | 0.0009637223233003 |
| rollout | share_of_training_total | 0.0120612411887733 | 0.0061058423079183 |
| rollout | duration_s | 19.82379029399817 | 35.1173993659977 |
| rollout | mean_power_w | 115.30290178512604 | 98.79434202181906 |
| rollout | total_flops | 14592000000.0 | 14592000000.0 |
| rollout | energy_per_flop_j_per_flop | 1.5664340359634088e-07 | 2.377604416036999e-07 |
| dynamics_model_update | energy_j | 142180.3445060511 | 482756.635474506 |
| dynamics_model_update | energy_kwh | 0.0394945401405697 | 0.134099065409585 |
| dynamics_model_update | share_of_training_total | 0.7502476302190896 | 0.8496096097744807 |
| dynamics_model_update | duration_s | 1211.9471398800015 | 4841.434729991996 |
| dynamics_model_update | mean_power_w | 117.31563186834106 | 99.71354823476145 |
| dynamics_model_update | total_flops | 255192420000000.0 | 255192420000000.0 |
| dynamics_model_update | energy_per_flop_j_per_flop | 5.571495599519293e-10 | 1.8917357947956404e-09 |
| synthetic_rollout_generation | energy_j | 46.09490968828825 | 219.08894984887863 |
| synthetic_rollout_generation | energy_kwh | 1.2804141580080069e-05 | 6.085804162468851e-05 |
| synthetic_rollout_generation | share_of_training_total | 0.0002432305033367 | 0.0003855774597568 |
| synthetic_rollout_generation | duration_s | 0.5898963209929158 | 2.0660122050012397 |
| synthetic_rollout_generation | mean_power_w | 78.14069701384324 | 106.0443637837789 |
| synthetic_rollout_generation | total_flops | 1971208800000.0 | 1971208800000.0 |
| synthetic_rollout_generation | energy_per_flop_j_per_flop | 2.3384082745718375e-11 | 1.1114446620209824e-10 |
| gradient_updates | energy_j | 44999.04110676012 | 81764.82707732423 |
| gradient_updates | energy_kwh | 0.0124997336407667 | 0.0227124519659233 |
| gradient_updates | share_of_training_total | 0.2374478980886233 | 0.1438989704577802 |
| gradient_updates | duration_s | 405.7487565479987 | 853.3665012989895 |
| gradient_updates | mean_power_w | 110.90370674108742 | 95.81443254787044 |
| buffer_sample | energy_j | 1326.1793733327054 | 3676.500526463758 |
| buffer_sample | energy_kwh | 0.000368383159259 | 0.0010212501462399 |
| buffer_sample | share_of_training_total | 0.0069978936648726 | 0.0064703205468203 |
| buffer_sample | duration_s | 11.672165659015263 | 23.56117137083129 |
| buffer_sample | mean_power_w | 113.61896430148784 | 156.0406513156333 |
| buffer_sample | total_flops | 0.0 | 0.0 |
| critic_update | energy_j | 18956.58036703348 | 34662.830417899975 |
| critic_update | energy_kwh | 0.0052657167686204 | 0.0096285640049722 |
| critic_update | share_of_training_total | 0.1000288017787107 | 0.0610035609268932 |
| critic_update | duration_s | 166.8434533229174 | 222.1397445194289 |
| critic_update | mean_power_w | 113.61896430148784 | 156.0406513156333 |
| critic_update | total_flops | 32492748800000.0 | 32492748800000.0 |
| critic_update | energy_per_flop_j_per_flop | 5.834095626601303e-10 | 1.0667866431140464e-09 |
| actor_update | energy_j | 22755.646565801424 | 40765.429974490544 |
| actor_update | energy_kwh | 0.0063210129349448 | 0.0113237305484695 |
| actor_update | share_of_training_total | 0.1200754574720354 | 0.0717436043501982 |
| actor_update | duration_s | 200.2803555348328 | 261.2487812040199 |
| actor_update | mean_power_w | 113.61896430148786 | 156.0406513156333 |
| actor_update | total_flops | 25663897600000.0 | 25663897600000.0 |
| actor_update | energy_per_flop_j_per_flop | 8.866792924626312e-10 | 1.5884348749307095e-09 |
| target_update | energy_j | 1960.634800609192 | 2660.0661584893955 |
| target_update | energy_kwh | 0.0005446207779469 | 0.000738907266247 |
| target_update | share_of_training_total | 0.0103457451730926 | 0.0046814846339027 |
| target_update | duration_s | 17.25622841805398 | 17.047263876826037 |
| target_update | mean_power_w | 113.61896430148784 | 156.04065131563328 |
| target_update | total_flops | 28877200000.0 | 28877200000.0 |
| target_update | energy_per_flop_j_per_flop | 6.789559931742663e-08 | 9.21164849254566e-08 |
| warmup | energy_j | 13.542979118144356 | 17.07180890773074 |
| warmup | energy_kwh | 3.761938643928988e-06 | 4.742169141036317e-06 |
| warmup | duration_s | 0.1509768030000486 | 0.1580752610007039 |
| warmup | mean_power_w | 89.7023837373189 | 107.99798020042314 |
| idle_baseline_head | energy_j | 5906.021359844822 | 5334.256772837016 |
| idle_baseline_head | energy_kwh | 0.0016405614888457 | 0.0014817379924547 |
| idle_baseline_head | duration_s | 90.00357840700053 | 90.0086557199993 |
| idle_baseline_head | mean_power_w | 65.61985050347118 | 59.26382002005371 |
| idle_baseline_tail | energy_j | 5964.827582440929 | 5677.948024121327 |
| idle_baseline_tail | energy_kwh | 0.001656896550678 | 0.0015772077844781 |
| idle_baseline_tail | duration_s | 90.00351394300014 | 90.01030038799945 |
| idle_baseline_tail | mean_power_w | 66.27327446591144 | 63.08109182666815 |

Consistency (`C_consistency_checks.csv`): CSV `TOTAL_MEASURED_TRAINING` energy equals the sum of the four measured tasks and the sum of the four allocated sub-segments equals measured `gradient_updates` in both runs; CSV total duration equals the sum of the four measured task durations:

| clock | csv_total_energy_j | sum_measured_energy_j | csv_total_duration_s | sum_measured_duration_s | sum_alloc_energy_j | gradient_updates_measured_energy_j | sum_alloc_wall_time_s | gradient_updates_measured_duration_s |
|---|---|---|---|---|---|---|---|---|
| 2000MHz | 189511.22106781087 | 189511.2210677773 | 1638.1095830429913 | 1638.1095830429913 | 44999.04110677681 | 44999.04110676012 | 396.0522029348194 | 405.7487565479987 |
| 200MHz | 568209.9518655966 | 568209.9518655604 | 5731.984642861984 | 5731.984642861984 | 81764.82707734367 | 81764.82707732423 | 523.9969609711061 | 853.3665012989895 |

**C3. All keys of `segment_energy.json`** (kWh unless `_`-prefixed; `C3_segment_energy_json_all_keys.csv`) — same key set in both runs: `idle_baseline_head, warmup, dynamics_model_update, rollout, synthetic_rollout_generation, buffer_sample, critic_update, actor_update, target_update, _sub_segment_wall_time_seconds{buffer_sample, critic_update, actor_update, target_update}, idle_baseline_tail, _total_kg_co2eq`. Values: R2000 `_total_kg_co2eq` 0.021311583383765147, R200 0.061294793236258686; the rest are in the table above (J) and the CSV (kWh).

**C4. CPU/GPU/RAM energy split** of the measured tasks, sums over the 100 epochs (`C4_cpu_gpu_ram_split.csv`; J and % of the task's `energy_consumed`; CodeCarbon `cpu_energy`/`gpu_energy`/`ram_energy`):

| clock | task | cpu_j | gpu_j | ram_j | sum_components_j | total_energy_consumed_j | cpu_pct | gpu_pct | ram_pct |
|---|---|---|---|---|---|---|---|---|---|
| 2000MHz | rollout | 600.0975760779264 | 1289.701031759893 | 395.9419374400699 | 2285.74054527789 | 2285.740545277806 | 26.253967333155533 | 56.42377191166046 | 17.322260755187664 |
| 2000MHz | dynamics_model_update | 31203.73603896453 | 86738.64239084724 | 24237.96607623696 | 142180.3445060487 | 142180.3445060511 | 21.946589134644075 | 61.00607133298633 | 17.047339532367925 |
| 2000MHz | synthetic_rollout_generation | 17.23922879151296 | 17.471013976883 | 11.384666919946527 | 46.09490968834248 | 46.09490968828825 | 37.39941982334133 | 37.90226316751418 | 24.69831700926216 |
| 2000MHz | gradient_updates | 10271.534843221303 | 26613.416290715435 | 8114.089972840134 | 44999.04110677687 | 44999.04110676012 | 22.826119380748825 | 59.14218533585008 | 18.031695283438317 |
| 2000MHz | warmup | 5.413626330919019 | 5.109004087214264 | 3.020348700010682 | 13.542979118143965 | 13.542979118144356 | 39.97367406161066 | 37.72437395528004 | 22.30195198310641 |
| 2000MHz | idle_baseline_head | 1514.8992499182 | 2528.0350224264 | 1863.08708749992 | 5906.02135984452 | 5906.02135984452 | 25.6500807839625 | 42.80436639824399 | 31.54555281779352 |
| 2000MHz | idle_baseline_tail | 1506.84836147748 | 2657.9311263428403 | 1800.04809461976 | 5964.827582440081 | 5964.8275824408 | 25.262228298322068 | 44.56006631553326 | 30.177705386132597 |
| 2000MHz | TRAINING_TOTAL (sum of 4 measured) | 42092.607687055264 | 114659.23072729944 | 32759.382653437107 | 189511.22106779183 | 189511.2210677773 | 22.211142670016965 | 60.50260775128055 | 17.286249578710144 |
| 200MHz | rollout | 1173.791396032679 | 1593.912275128684 | 701.6966927197267 | 3469.4003638810896 | 3469.4003638811887 | 33.832687868850286 | 45.94201037511819 | 20.225301756028657 |
| 200MHz | dynamics_model_update | 159892.32686474433 | 226037.13582954783 | 96827.17278018613 | 482756.6354744783 | 482756.635474506 | 33.12068962192196 | 46.8221706797202 | 20.057139698352103 |
| 200MHz | synthetic_rollout_generation | 66.00091880073697 | 112.41108992873228 | 40.67694111979503 | 219.08894984926428 | 219.08894984887863 | 30.12517009473209 | 51.30842518815773 | 18.56640471728621 |
| 200MHz | gradient_updates | 25069.90057390348 | 39628.555702801925 | 17066.37080061992 | 81764.82707732532 | 81764.82707732423 | 30.660984031917675 | 48.46650707806863 | 20.872508890015038 |
| 200MHz | warmup | 5.790638632521827 | 8.119006495210801 | 3.1621637799981137 | 17.07180890773074 | 17.07180890773074 | 33.91930324325276 | 47.55797431363127 | 18.52272244311597 |
| 200MHz | idle_baseline_head | 1527.03149262408 | 1946.21755697244 | 1861.00772323968 | 5334.256772836201 | 5334.25677283692 | 28.626883887555312 | 36.485261955947855 | 34.88785415648336 |
| 200MHz | idle_baseline_tail | 1502.6630051293198 | 2375.19190015164 | 1800.09311883984 | 5677.948024120799 | 5677.94802412116 | 26.46489539435162 | 41.83187112776142 | 31.703233477880605 |
| 200MHz | TRAINING_TOTAL (sum of 4 measured) | 186202.01975348123 | 267372.0148974072 | 114635.91721464554 | 568209.951865534 | 568209.9518655604 | 32.76993286410038 | 47.0551446731204 | 20.17492246277457 |

**C5. Whole-run row of `emissions.csv`** (`C5_whole_run_row.csv`). Note: the `duration` field of this row is ≈90 s in both runs (not the whole-run wall time), while `energy_consumed` equals the sum of all per-task rows to within the difference shown (plain arithmetic: run_energy_j − sum_all_task_rows_energy_j; sum of task durations is for all 403 task rows):

| quantity | R2000 | R200 |
|---|---|---|
| run_duration_s | 90.00569242900019 | 90.01476587300021 |
| run_energy_kwh | 0.0559432560277336 | 0.1608998378691656 |
| run_energy_j | 201395.7216998409 | 579239.4163289961 |
| cpu_energy_j | 45119.826724833 | 189237.5968699566 |
| gpu_energy_j | 119850.30588016764 | 271701.54336106044 |
| ram_energy_j | 36425.58909483996 | 118300.27609797948 |
| sum_all_task_rows_energy_j | 201395.61298918084 | 579239.2284714261 |
| diff_run_minus_sum_tasks_j | 0.108710660104407 | 0.1878575700102374 |
| sum_all_task_rows_duration_s | 1818.267652195992 | 5912.161674230983 |
| diff_run_duration_minus_sum_tasks_s | -1728.2619597669918 | -5822.146908357983 |
| cc_run_avg_cpu_power_w | 24.66324335973666 | 28.72444249174328 |
| cc_run_avg_gpu_power_w | 58.296749993038944 | 47.9784377246624 |
| cc_run_avg_ram_power_w | 20.0 | 20.0 |

**C6. Per-task duration statistics** (`C6_task_duration_stats.csv`; full per-task table: `per_task_rows.csv`; `measure_power_secs` = 1.0 in both `metadata.json`):

| clock | task | count | min_s | median_s | max_s | sum_s | n_shorter_than_1s |
|---|---|---|---|---|---|---|---|
| 2000MHz | rollout | 100 | 0.1909193159999631 | 0.1976618474996029 | 0.2258806670006379 | 19.82379029399817 | 100 |
| 2000MHz | dynamics_model_update | 100 | 1.1900708089997352 | 10.36989949900044 | 109.02184160600063 | 1211.9471398800015 | 0 |
| 2000MHz | synthetic_rollout_generation | 100 | 0.0045854049994886 | 0.0055846104992269 | 0.0179569080000874 | 0.5898963209929158 | 100 |
| 2000MHz | gradient_updates | 100 | 3.900995219999458 | 3.982663545999913 | 4.31692417100021 | 405.7487565479987 | 0 |
| 200MHz | rollout | 100 | 0.3450027460003184 | 0.3473173740003403 | 0.4573335770000994 | 35.1173993659977 | 100 |
| 200MHz | dynamics_model_update | 100 | 4.257012857000518 | 41.701290279000204 | 489.6062017889999 | 4841.434729991996 | 0 |
| 200MHz | synthetic_rollout_generation | 100 | 0.0182262709986389 | 0.0203280244995766 | 0.02878076900015 | 2.0660122050012397 | 100 |
| 200MHz | gradient_updates | 100 | 8.26946648699959 | 8.331643628500387 | 10.97964691299967 | 853.3665012989895 | 0 |

**C7. Work done** (`C7_work_done.csv`; from `training_metrics.json` and the FLOP CSV). Per-epoch `model_train_epochs`, `synthetic_transitions_generated`, `mean_episode_return`, `model_holdout_mse` and all episode returns are identical between the two runs (`findings.json`: `training_metrics_epochs_identical` = True, episode returns identical = True). Last-10% return = mean of the last max(1, n//10) train-phase episode returns (definition of `aggregate_results.py`).

| quantity | R2000 | R200 |
|---|---|---|
| env_steps_train | 100000 | 100000 |
| n_epochs | 100 | 100 |
| n_train_episodes | 100 | 100 |
| n_warmup_episodes | 5 | 5 |
| n_model_fits | 100 | 100 |
| fit_epochs_total | 1196 | 1196 |
| fit_epochs_first | 195 | 195 |
| fit_epochs_min | 6 | 6 |
| fit_epochs_max | 195 | 195 |
| optimiser_steps_dynamics_call_count | 168525.0 | 168525.0 |
| dynamics_total_flops | 255192420000000.0 | 255192420000000.0 |
| synthetic_transitions_generated_last_epoch | 10000 | 10000 |
| synthetic_transitions_sum_over_epochs | 990000 | 990000 |
| synth_rollout_call_count | 990000.0 | 990000.0 |
| n_epochs_with_model_train_epochs_key | 100 | 100 |
| last10pct_n_episodes | 10 | 10 |
| last10pct_mean_return | 5936.559047246978 | 5936.559047246978 |
| max_train_return | 6292.640463843409 | 6292.640463843409 |
| final_cumulative_reward | 325883.6580644874 | 325883.6580644874 |

**C8. Idle power** (`C8_idle_power_two_runs.csv`, computed with `idle_power_analysis.load_runs` restricted to these two runs; `*_w` = energy ÷ recorded duration, `*_w_corr` = energy ÷ integrated span, method of `idle_power_analysis.py`; component columns cpu/gpu/ram/total):

| quantity | 2000MHz | 200MHz |
|---|---|---|
| head_duration_s | 90.00357840700053 | 90.0086557199993 |
| head_integrated_window_s | 93.154354374996 | 93.050386161984 |
| head_cpu_w | 16.83154466445492 | 16.965384944470227 |
| head_cpu_w_corr | 16.26224839495878 | 16.41080231484255 |
| head_gpu_w | 28.08816123948432 | 21.62255997941767 |
| head_gpu_w_corr | 27.13813046516012 | 20.91573863631716 |
| head_ram_w | 20.70014459952859 | 20.675875096156748 |
| head_ram_w_corr | 20.0 | 20.0 |
| head_total_w | 65.61985050346783 | 59.26382002005264 |
| head_total_w_corr | 63.40037886011889 | 57.32654095116743 |
| tail_duration_s | 90.00351394300014 | 90.01030038799945 |
| tail_integrated_window_s | 90.00240473098799 | 90.00465594199201 |
| tail_cpu_w | 16.742105896351752 | 16.69434496554198 |
| tail_cpu_w_corr | 16.742312230227217 | 16.69539191503366 |
| tail_gpu_w | 29.53141505148484 | 26.388001038915657 |
| tail_gpu_w_corr | 29.53177900398599 | 26.389655904939524 |
| tail_ram_w | 19.99975351806534 | 19.9987458222041 |
| tail_ram_w_corr | 20.0 | 20.0 |
| tail_total_w | 66.27327446590994 | 63.08109182666574 |
| tail_total_w_corr | 66.2740912342212 | 63.08504781997719 |
| delta_cpu_w | -0.089438768103168 | -0.2710399789282469 |
| delta_cpu_pct | -0.5313758771768943 | -1.597605829843495 |
| delta_gpu_w | 1.4432538120005205 | 4.765441059497988 |
| delta_gpu_pct | 5.138299369955545 | 22.039208419512637 |
| delta_ram_w | -0.7003910814632484 | -0.677129273952648 |
| delta_ram_pct | -3.3835081590647373 | -3.274972743855052 |
| delta_total_w | 0.6534239624421048 | 3.817271806613107 |
| delta_total_pct | 0.995771793792144 | 6.441150444438929 |
| delta_cpu_w_corr | 0.4800638352684352 | 0.2845896001911114 |
| delta_cpu_pct_corr | 2.9520139135081265 | 1.734160187486494 |
| delta_gpu_w_corr | 2.393648538825872 | 5.473917268622365 |
| delta_gpu_pct_corr | 8.820241106508178 | 26.171283566900662 |
| delta_ram_w_corr | 0.0 | 0.0 |
| delta_ram_pct_corr | 0.0 | 0.0 |
| delta_total_w_corr | 2.873712374102311 | 5.758506868809761 |
| delta_total_pct_corr | 4.532642274019563 | 10.045097389907127 |

NVML temperatures recorded around the idle windows: `metadata.json` has no `run_start_temp_c`/`run_end_temp_c` for either run (keys absent — NOT FOUND). What exists is the thermal-gate result: R2000 `final_temp_c` 45.0, `final_power_w` 26.467; R200 `final_temp_c` 44.0, `final_power_w` 21.006 (both `reason: reached_reference`, `waited_seconds` ≈ 1e-5).

**C9. Idle-floor diagnostic** (`C9_idle_floor_diagnostic.csv`). Formula: idle_floor_fraction = mean(head_total_w_corr, tail_total_w_corr) × task duration ÷ gross task energy; idle_floor_w = mean of the two window-corrected total idle powers of the same run:

| clock | task | idle_floor_w | duration_s | gross_energy_j | idle_floor_energy_j | idle_floor_fraction |
|---|---|---|---|---|---|---|
| 2000MHz | rollout | 64.83723504717005 | 19.82379029399817 | 2285.740545277806 | 1285.319750817768 | 0.5623209307255613 |
| 2000MHz | dynamics_model_update | 64.83723504717005 | 1211.9471398800015 | 142180.3445060511 | 78579.30157314515 | 0.5526734503713406 |
| 2000MHz | synthetic_rollout_generation | 64.83723504717005 | 0.5898963209929158 | 46.09490968828825 | 38.24724641767856 | 0.8297498937804922 |
| 2000MHz | gradient_updates | 64.83723504717005 | 405.7487565479987 | 44999.04110676012 | 26307.62749839957 | 0.5846264020600969 |
| 2000MHz | TRAINING_TOTAL | 64.83723504717005 | 1638.1095830429913 | 189511.2210677773 | 106210.49606878016 | 0.5604443656177739 |
| 200MHz | rollout | 60.20579438557231 | 35.1173993659977 | 3469.4003638811887 | 2114.2709255852847 | 0.6094052873217745 |
| 200MHz | dynamics_model_update | 60.20579438557231 | 4841.434729991996 | 482756.635474506 | 291482.4238850669 | 0.6037875038187017 |
| 200MHz | synthetic_rollout_generation | 60.20579438557231 | 2.0660122050012397 | 219.08894984887863 | 124.38590601238748 | 0.5677415775564462 |
| 200MHz | gradient_updates | 60.20579438557231 | 853.3665012989895 | 81764.82707732423 | 51377.60811274219 | 0.6283583045330099 |
| 200MHz | TRAINING_TOTAL | 60.20579438557231 | 5731.984642861984 | 568209.9518655604 | 345098.68882940675 | 0.6073436195483211 |

**D. Ratios R200 / R2000** (`D_ratios_200_over_2000.csv`). DERIVED: ratio = value(R200) / value(R2000); difference = value(R200) − value(R2000). Power = energy ÷ duration of the same row (allocated rows: CSV definition, see C1/C2).

| segment | kind | energy_j_2000 | energy_j_200 | energy_ratio_200_over_2000 | energy_diff_j_200_minus_2000 | duration_s_2000 | duration_s_200 | duration_ratio | duration_diff_s | power_w_2000 | power_w_200 | power_ratio | power_diff_w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout | measured | 2285.740545277806 | 3469.4003638811887 | 1.517845221343581 | 1183.6598186033823 | 19.82379029399817 | 35.1173993659977 | 1.7714775451710565 | 15.293609071999526 | 115.30290178512604 | 98.79434202181906 | 0.856824420654463 | -16.508559763306977 |
| dynamics_model_update | measured | 142180.3445060511 | 482756.635474506 | 3.395382372659536 | 340576.2909684549 | 1211.9471398800015 | 4841.434729991996 | 3.994757337742767 | 3629.487590111994 | 117.31563186834106 | 99.71354823476145 | 0.8499596059514577 | -17.602083633579625 |
| synthetic_rollout_generation | measured | 46.09490968828825 | 219.08894984887863 | 4.752996617857449 | 172.99404016059037 | 0.5898963209929158 | 2.0660122050012397 | 3.502331056284127 | 1.4761158840083235 | 78.14069701384324 | 106.0443637837789 | 1.3570951864556866 | 27.90366676993567 |
| gradient_updates | measured | 44999.04110676012 | 81764.82707732423 | 1.8170348759951875 | 36765.7859705641 | 405.7487565479987 | 853.3665012989895 | 2.1031894430415567 | 447.6177447509908 | 110.90370674108742 | 95.81443254787044 | 0.8639425620962881 | -15.08927419321698 |
| TOTAL_MEASURED_TRAINING | total | 189511.22106781087 | 568209.9518655966 | 2.998291861896029 | 378698.7307977857 | 1638.1095830429913 | 5731.984642861984 | 3.499146029177189 | 4093.8750598189927 | 115.68897650654756 | 99.12970589919269 | 0.8568638853294922 | -16.559270607354875 |
| buffer_sample | allocated (within gradient_updates) | 1326.1793733327054 | 3676.500526463758 | 2.7722498180805406 | 2350.3211531310526 | 11.672165659015263 | 23.56117137083129 | 2.018577533864359 | 11.889005711816026 | 113.61896430148784 | 156.0406513156333 | 1.3733680136493724 | 42.42168701414548 |
| critic_update | allocated (within gradient_updates) | 18956.58036703348 | 34662.830417899975 | 1.828538151225868 | 15706.250050866496 | 166.8434533229174 | 222.1397445194289 | 1.3314261968043073 | 55.29629119651148 | 113.61896430148784 | 156.0406513156333 | 1.3733680136493724 | 42.42168701414548 |
| actor_update | allocated (within gradient_updates) | 22755.646565801424 | 40765.429974490544 | 1.79144239459912 | 18009.78340868912 | 200.2803555348328 | 261.2487812040199 | 1.3044154056266564 | 60.9684256691871 | 113.61896430148786 | 156.0406513156333 | 1.3733680136493722 | 42.42168701414545 |
| target_update | allocated (within gradient_updates) | 1960.634800609192 | 2660.0661584893955 | 1.356737194332612 | 699.4313578802034 | 17.25622841805398 | 17.047263876826037 | 0.987890485906566 | -0.2089645412279424 | 113.61896430148784 | 156.04065131563328 | 1.3733680136493722 | 42.42168701414545 |

Idle power ratios R200/R2000 (`D_idle_ratios.csv`; DERIVED: power(R200)/power(R2000)):

| window | component | basis | power_w_2000 | power_w_200 | ratio_200_over_2000 | diff_w_200_minus_2000 |
|---|---|---|---|---|---|---|
| head | cpu | as_recorded | 16.83154466445492 | 16.965384944470227 | 1.007951752657494 | 0.1338402800153062 |
| head | cpu | window_corrected | 16.26224839495878 | 16.41080231484255 | 1.0091348942825038 | 0.1485539198837671 |
| head | gpu | as_recorded | 28.08816123948432 | 21.62255997941767 | 0.7698104477206655 | -6.465601260066652 |
| head | gpu | window_corrected | 27.13813046516012 | 20.91573863631716 | 0.7707140572254506 | -6.222391828842959 |
| head | ram | as_recorded | 20.70014459952859 | 20.675875096156748 | 0.9988275684135852 | -0.0242695033718405 |
| head | ram | window_corrected | 20.0 | 20.0 | 1.0 | 0.0 |
| head | total | as_recorded | 65.61985050346783 | 59.26382002005264 | 0.9031386015870412 | -6.356030483415196 |
| head | total | window_corrected | 63.40037886011889 | 57.32654095116743 | 0.9041987127182276 | -6.073837908951461 |
| tail | cpu | as_recorded | 16.742105896351752 | 16.69434496554198 | 0.9971472566769404 | -0.0477609308097726 |
| tail | cpu | window_corrected | 16.742312230227217 | 16.69539191503366 | 0.9971975008858786 | -0.0469203151935566 |
| tail | gpu | as_recorded | 29.53141505148484 | 26.388001038915657 | 0.8935569458121468 | -3.1434140125691847 |
| tail | gpu | window_corrected | 29.53177900398599 | 26.389655904939524 | 0.8936019703173871 | -3.142123099046465 |
| tail | ram | as_recorded | 19.99975351806534 | 19.9987458222041 | 0.9999496145859832 | -0.0010076958612401 |
| tail | ram | window_corrected | 20.0 | 20.0 | 1.0 | 0.0 |
| tail | total | as_recorded | 66.27327446590994 | 63.08109182666574 | 0.9518330327727174 | -3.1921826392441943 |
| tail | total | window_corrected | 66.2740912342212 | 63.08504781997719 | 0.9518809936906788 | -3.1890434142440114 |

**E. Reference for scale** (canonical MBPO / HalfCheetah-v5, same configuration as R2000). Matching: `architecture_signature` `bs256_h256x256_ens7_mh200x200x200x200_mb256`, `updates_per_env_step` 1, `mbpo_rollout_regime` `rl1-1_re20-150`, canonical seeds {331, 958, 14577, 43611, 85062} (five runs; `per_run_energy_per_flop.csv`, `TOTAL_MEASURED_TRAINING` rows). Full `metadata.json` comparison of R2000 against the five canonical runs (excluding seed, timestamps, git commit and the `thermal_gate.*` result values) finds one differing field: `experiment_config.thermal_gate_reference_file` (`results/_thermal_reference.json` in R2000, absent in the canonical runs); canonical runs: commit `7e672ef12c53bf4c201241c549527a7ef8bd4bff`, torch `2.13.0+cu130`, codecarbon `3.3.0` (`E_config_diff_vs_canonical.csv`). Cross-check row in `cross_seed_energy_per_flop.csv` (`E_cross_seed_csv_row.csv`): n_seeds 5, mean_energy_joules 187913.…, mean_duration_s 1631.84…, mean_power_w 115.154… (matches the recomputed means below).

Per-seed canonical values (`E_canonical_per_seed.csv`):

| seed | run_dir | total_energy_joules | total_energy_kwh | duration_s | mean_power_w | total_flops | energy_per_flop_j_per_flop |
|---|---|---|---|---|---|---|---|
| 14577 | results/mbpo/HalfCheetah-v5/seed_14577/20260912_235401 | 190324.50394838976 | 0.0528679177634416 | 1638.9463513909868 | 116.1261342000975 | 314395159200000.0 | 6.053671577917533e-10 |
| 331 | results/mbpo/HalfCheetah-v5/seed_331/20260912_225321 | 186297.35303531543 | 0.051749264732032 | 1604.0837964410257 | 116.1394145671522 | 309340669200000.0 | 6.022400918608842e-10 |
| 43611 | results/mbpo/HalfCheetah-v5/seed_43611/20260913_002507 | 190426.5901178243 | 0.0528962750327289 | 1654.505437425007 | 115.095777753499 | 308211121200000.0 | 6.178446429071432e-10 |
| 85062 | results/mbpo/HalfCheetah-v5/seed_85062/20260913_005629 | 192336.09809998824 | 0.0534266939166634 | 1681.1043290839953 | 114.41056618109408 | 308063995200000.0 | 6.243381281058847e-10 |
| 958 | results/mbpo/HalfCheetah-v5/seed_958/20260912_232353 | 180182.0040795628 | 0.0500505566887674 | 1580.5765508490076 | 113.99764470931686 | 293721583200000.0 | 6.13444889260568e-10 |

Statistics (`E_canonical_stats.csv`; sd = sample sd, ddof=1; DERIVED: comparator_minus_mean = R2000 − mean; …_over_sd = that ÷ sd; run_200_over_canonical_mean = R200 ÷ mean). Note `total_flops` of the canonical runs is not identical to R2000's (MBPO fit epochs vary per seed; R2000 `total_flops` is in C1/C2, the canonical cross-seed mean is in `E_cross_seed_csv_row.csv` column `total_flops`):

| metric | n | canonical_mean | canonical_sd_ddof1 | canonical_min | canonical_max | comparator_2000 | comparator_minus_mean | comparator_minus_mean_over_sd | run_200_value | run_200_over_canonical_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| total_energy_joules | 5 | 187913.30985621607 | 4849.636479466794 | 180182.0040795628 | 192336.09809998824 | 189511.22106781087 | 1597.9112115947937 | 0.3294909254250908 | 568209.9518655966 | 3.023787683269316 |
| duration_s | 5 | 1631.8432930380045 | 39.967312908483 | 1580.5765508490076 | 1681.1043290839953 | 1638.1095830429913 | 6.266290004986786 | 0.156785371569395 | 5731.984642861984 | 3.512582775145487 |
| mean_power_w | 5 | 115.15390748223192 | 0.975874986964112 | 113.99764470931686 | 116.1394145671522 | 115.68897650654756 | 0.5350690243156464 | 0.5482966890874145 | 99.12970589919269 | 0.8608453509446761 |

## 4. Consistency checks (item F)

1. **Open TODO "confirm identical config and same software stack (torch 2.13.0+cu130 / codecarbon 3.3.0)"**: the exact TODO text was not found by grep in `thesis-main.tex` (searched "TODO…identical/2.13", `2.13.0+cu130`; closest: `\todo` at line 2266 about the clock chapter and hidden-size caveat; line 3102 "Identical software stack. All runs used PyTorch 2.13.0+cu130 and CodeCarbon 3.3.0"). Fact: R2000 and R200 both record torch `2.13.0+cu130` and codecarbon `3.3.0`, identical config except the two clock fields (A4) — satisfied for the pair. Both also match the software stack of the canonical runs (E).
2. **Hidden sizes / batch size**: both runs `hidden_sizes` = [256, 256] (not the canonical 1024); `batch_size` = 256. The canonical MBPO HalfCheetah-v5 baseline (`hidden_sizes` [1024,1024]) has `batch_size` 256 (`findings.json`: canonical MBPO-HC configs (hidden, batch, utd): n — ((1024,1024),256,1): 5, ((256,256),256,1): 5, ((512,512),256,1): 5, UTD2/UTD4 and batch 512/1024 variants 5 each). So batch size equals the canonical baseline's; hidden size equals the canonical width-256 sweep configuration.
3. **Dev runs**: 20 of 300 recorded runs have a non-canonical seed (= 300 − 280), matching the 20 in Section 3.6 (`thesis-main.tex:3003`: "20 development runs"). By (algo, env, seed) (`F3_dev_runs_by_algo_env_seed.csv`), sums to 20:

| algo | env | seed | n_runs |
|---|---|---|---|
| mbpo | HalfCheetah-v5 | 0 | 2 |
| mbpo | HalfCheetah-v5 | 42 | 1 |
| sac | HalfCheetah-v5 | 0 | 7 |
| sac | HalfCheetah-v5 | 42 | 3 |
| sac | HalfCheetah-v5 | 56 | 3 |
| sac | Humanoid-v5 | 56 | 2 |
| td3 | HalfCheetah-v5 | 0 | 1 |
| tdmpc2 | HalfCheetah-v5 | 0 | 1 |

   The Section 3.6 text at `thesis-main.tex:3004-3006` says dev runs are not part of any total/table/figure/aggregate, with a `\todo` at line 3006 about this Chapter 6 exception (fact only, not edited).

## 5. Caveats observable from the data (item G)

- **FLOPs / fit epochs**: identical between the runs: `TOTAL_MEASURED_TRAINING` `total_flops` 3.153349…e14 in both (see table), 1196 fit epochs in total, 168525 optimiser steps, 990000 synthetic samples, 100 model fits (C7). No difference.
- **Tasks shorter than the 1 s polling interval** (`C6`): rollout 100/100 and synthetic_rollout_generation 100/100 at both clocks (rollout median 0.197662 s at 2000 MHz vs 0.347317 s at 200 MHz; synthetic_rollout_generation median 0.00558461 s vs 0.020328 s). dynamics_model_update: 0 tasks < 1 s at both clocks (min 1.19007 s vs 4.25701 s). gradient_updates: 0 tasks < 1 s (min 3.901 s vs 8.26947 s).
- **Sub-segment wall time vs measured `gradient_updates` duration**: sum of allocated sub-segment `perf_counter` times / measured `gradient_updates` duration = 396.052/405.749 at 2000 MHz and 523.997/853.367 at 200 MHz (`C_consistency_checks.csv`; DERIVED ratios 396.052/405.749 = 0.9761009885421775 and 523.997/853.367 = 0.6140347587849073). Because the CSV defines allocated-segment power as energy ÷ sub-segment wall time, the allocated segments show 113.619 W (2000 MHz) and 156.041 W (200 MHz) whereas measured `gradient_updates` mean power is 110.904 W and 95.8144 W.
- **Head vs. tail idle drift** (C8, window-corrected total): R2000 head 63.400379 → tail 66.274091 W (+2.873712 W, +4.532642 %); R200 head 57.326541 → tail 63.085048 W (+5.758507 W, +10.045097 %). Head idle power differs between the runs (63.400379 vs 57.326541 W) although both gates passed immediately and `final_temp_c` was 45.0 vs 44.0 °C. Head window integrates 93.154354 s (R2000) and 93.050386 s (R200) for a recorded 90.003578 / 90.008656 s.
- **Run order / timing**: both runs on 2026-09-04; R2000 `start_time_utc` 10:29:59, `end_time_utc` 11:01:08; R200 `start_time_utc` 11:09:15 (DERIVED: 11:09:15 − 11:01:08 = 8 min 7 s between R2000's end and R200's start), `end_time_utc` 12:48:34. R200 therefore ran directly after R2000 on the same machine. `results/_thermal_reference.json` is overwritten each run (audit only; its current content is from the last run of the project, not these).
- **Logs**: no difference between the runs other than timing/energy numbers and thermal-gate values (A6). No `gpu_control` log lines in either (B3).
- **Mean power of the training total vs. which tasks dominate**: `dynamics_model_update` is 0.7502476302180711 of training energy at 2000 MHz (142180.344506/189511.221068) and 0.849609609774747 at 200 MHz (482756.635475/568209.951866) — see `share_of_training_total` in C1/C2 for exact values.
- **Per-segment GPU share of energy** changes between clocks (C4: GPU 60.5026…% of training energy at 2000 MHz vs 47.0551…% at 200 MHz; CPU 22.2111…% vs 32.7699…%).
- The FLOP CSV `energy_per_flop` for `TOTAL_MEASURED_TRAINING`: 6.009840e-10 J/FLOP (R2000) vs 1.801926e-09 J/FLOP (R200) (see C1/C2 for full precision).

## 6. Not found / discrepancies

- NOT FOUND: any commit message, comment or note saying why 200 MHz was tried (looked: `git log` all branches grep clock/MHz/frequency/lock; commit messages `f109bd4`, `f23f448`, `ae38808`; `README.md`; `CLAUDE.md`; `cli_commands.txt`; `scripts/`).
- NOT FOUND: the exact command line / config edit used for the two runs; `metadata.json` shows explicit 2000 / 200 although the committed config at the recorded commit defaults to `None` (A5, B1).
- NOT FOUND: GPU power limit, supported/max clocks, memory clock, driver version, CUDA runtime version (B4).
- NOT FOUND: NVML temperatures before/after the idle windows in `metadata.json` (`run_start_temp_c`, `run_end_temp_c` keys absent in both runs); only `thermal_gate.final_temp_c/final_power_w`.
- NOT FOUND: the thesis TODO phrase "confirm identical config and same software stack" in `thesis-main.tex` (F1).
- DISCREPANCY: `CLAUDE.md` (~lines 380-384) and `README.md` (~line 750) call R200 an "anomaly" / "locked at the wrong GPU clock"; the author states it was intentional. The repo contains no record of intent (see B2).
- DISCREPANCY: `CLAUDE.md`/`README.md` say the 15 `null`-clock SAC dev runs resolve to `DEFAULT_LOCK_MHZ=2000` and are "functionally identical" to 2000/2000 runs; `DEFAULT_LOCK_MHZ` did not exist in the code of that period (first in `f109bd4`, value 200; 2000 from `f23f448`), and the committed fallback was the lowest supported clock (B1). What clock those runs used is not recorded.
- DISCREPANCY: the recorded `git_commit` (`ae38808…`) of both runs predates the MBPO code (`f109bd4`), so it does not identify the code that ran (A5).
- OBSERVATION (no repo document claims otherwise; checked README/CLAUDE.md for 'verify'/'Locked GPU' and found none): neither `run.log` contains a `gpu_control` line, because that logger has no handler writing to `run.log` (B3). The Section 3.3 text relying on `run.log` for "lock verified" (per the task brief) is not supported by these two logs.
- NOTE: `emissions.csv` `duration` is ≈ 90 s in both runs (C5), i.e. not whole-run wall time.

## 7. Manual follow-ups (item H)

Not in the repo; run on the experiment machine (read-only commands). I did not run them (this is the Windows dev checkout without the GPU).
- Supported graphics/memory clocks, current and default/boost clocks, power limits, driver and CUDA versions: `nvidia-smi -q -d SUPPORTED_CLOCKS,CLOCK,POWER` and `nvidia-smi --query-gpu=name,driver_version,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.graphics,clocks.max.memory --format=csv`
- CUDA toolkit version: `nvcc --version` (if installed) and `rl-exp/bin/python -c "import torch; print(torch.version.cuda)"`
- Whether the clock lock is active during a run (since `run.log` does not record it): while a run executes, `nvidia-smi --query-gpu=clocks.gr,clocks.applications.graphics --format=csv -l 1`
- Which values were set for the two runs on 2026-09-04 (shell history of the experiment machine, e.g. `history | grep run_experiment` or the uncommitted working-tree edit of `configs/config.py`): NOT FOUND in repo.
- Confirm the author's intent statement for the 200 MHz run (source: author).

## 8. Source index

`results/mbpo/HalfCheetah-v5/seed_0/20260904_122958/{metadata.json,segment_energy.json,training_metrics.json,emissions.csv,emissions_base_ec798f5b-….csv,run.log}`; same set for `20260904_130915` (`emissions_base_ff2f99fb-….csv`); all 300 `results/**/metadata.json` (A1, A2, E, F3); the five canonical `results/mbpo/HalfCheetah-v5/seed_*/…/metadata.json` listed in `E_canonical_per_seed.csv`; `flop_analysis/output/per_run_energy_per_flop.csv`; `flop_analysis/output/cross_seed_energy_per_flop.csv`; `idle_power_analysis.py` (method, imported); `aggregate_results.py` (last-10% definition); `utils/gpu_control.py`; `configs/config.py`; `run_experiment.py`; `experiment_runner.py`; `README.md`; `CLAUDE.md`; `cli_commands.txt`; `results/_thermal_reference.json`; git history (`git log`, `git show ae38808:…`, `git show f109bd4:…`, `git diff ae38808 f109bd4`, `git diff f109bd4 f23f448`); `C:\Users\saman\Desktop\Thesis\Thesis Report\thesis-main.tex` (read-only grep: lines 2266, 3001-3006, 3102). Generators: `contexts/Section_6_data/build_section_6.py`, `render_section_6.py`; existing `contexts/section_5.4.md` read for layout only.
