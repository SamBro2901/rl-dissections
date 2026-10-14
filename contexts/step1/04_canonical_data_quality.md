# Task 4: Canonical-only data quality and execution order

## Question

Do the statements of `sec:data-quality` hold on the 280 canonical runs alone, what data-side evidence exists for the confound controls, in which order were the runs executed, and what code/commits produced them?

## Method

- Script: `contexts/step1/scripts/canonical_data_quality.py` (repo `.venv` Python, read-only on `results/`, `configs/`, `scripts/`), then `make_04_md.py` for this file. Shared canonical list: `common.py` → `04_canonical_run_list.csv` (280 rows: algo, env, seed, config label, run directory, start/end, commit).
- Config label = difference of a run's `algo_config`/`experiment_config` from the modal config of its (algo, env) group among canonical runs (same convention as `idle_power_analysis.py` `variant`).
- Inputs: `metadata.json`, `segment_energy.json`, `training_metrics.json`, `emissions.csv`, `emissions_base_*.csv`, `run.log` of each canonical run; the 9 sweep logs `results/**/_*.log` (stdout+stderr of the sweep scripts, via `2>&1 | tee`, e.g. `scripts/run_width_sweep.sh:191`); `flop_analysis/output/per_run_energy_per_flop.csv` for the `check_totals.py` checks; `git log`/`git diff` of this checkout (61 commits, all 12 commits recorded in the metadata exist locally).
- Outputs: `04_canonical_run_list.csv`, `04_execution_timeline.csv`, `04_results.json` (all numbers below, full detail).

## Results

### 1. Run count and completeness

- Canonical runs found: **280** = 56 configurations × 5 seeds; duplicates for the same (algo, env, config, seed): 0; configurations without all 5 seeds: 0. Total run directories under `results/`: 300 (20 non-canonical: mbpo/HalfCheetah-v5/seed_0 ×2, mbpo/HalfCheetah-v5/seed_42 ×1, sac/HalfCheetah-v5/seed_0 ×7, sac/HalfCheetah-v5/seed_42 ×3, sac/HalfCheetah-v5/seed_56 ×3, sac/Humanoid-v5/seed_56 ×2, td3/HalfCheetah-v5/seed_0 ×1, tdmpc2/HalfCheetah-v5/seed_0 ×1).
- Missing files (of `metadata.json`, `segment_energy.json`, `training_metrics.json`, `emissions.csv`, `emissions_base_*.csv`, `run.log`, `end_time_utc` in metadata): **0**.

| algo / env | configs per seed (runs per seed) | seeds × configs = runs | expected (task) |
|---|---|---|---|
| mbpo/Ant-v5 | 9 | 5 × 9 = 45 | 9 ✓ |
| mbpo/HalfCheetah-v5 | 7 | 5 × 7 = 35 | 7 ✓ |
| sac/Ant-v5 | 7 | 5 × 7 = 35 | 7 ✓ |
| sac/HalfCheetah-v5 | 7 | 5 × 7 = 35 | 7 ✓ |
| td3/Ant-v5 | 8 | 5 × 8 = 40 | 8 ✓ |
| td3/HalfCheetah-v5 | 8 | 5 × 8 = 40 | 8 ✓ |
| tdmpc2/Ant-v5 | 5 | 5 × 5 = 25 | 5 ✓ |
| tdmpc2/HalfCheetah-v5 | 5 | 5 × 5 = 25 | 5 ✓ |

Protocol values over the 280 runs (distinct value: count):

- `train_steps`: {'100000': 280} (N_env = 100 000 measured env steps in every run: True); last `epochs[-1].env_step`: {'100000': 280}.
- `steps_per_epoch`: {'1000': 280}; epochs in `training_metrics.json`: {'100': 280}; `rollout_i` task rows per run: {'100': 280} → 100 epochs everywhere.
- `warmup_steps` by algorithm: {'mbpo': {'5000': 80}, 'sac': {'5000': 70}, 'td3': {'10000': 80}, 'tdmpc2': {'5000': 50}} (SAC/MBPO/TD-MPC2 5000, TD3 10000). Warmup steps are *additional* to the 100 000 measured steps (e.g. final buffer size 105 000 for SAC, 110 000 for TD3 in `run.log`).
- (idle_baseline_seconds, idle_tail_seconds, settle_seconds): {'(90.0, 90.0, 30.0)': 280}.
- `updates_per_env_step` d by algorithm over all configs: {'sac': {'1': 50, '2': 10, '4': 10}, 'mbpo': {'1': 60, '2': 10, '4': 10}, 'td3': {'1': 60, '2': 10, '4': 10}, 'tdmpc2': {'1': 50}}. **Baseline configs: d = 1 in all 8 algo×env baselines** ({'sac/HalfCheetah-v5': {'1': 5}, 'sac/Ant-v5': {'1': 5}, 'mbpo/HalfCheetah-v5': {'1': 5}, 'mbpo/Ant-v5': {'1': 5}, 'td3/HalfCheetah-v5': {'1': 5}, 'td3/Ant-v5': {'1': 5}, 'tdmpc2/HalfCheetah-v5': {'1': 5}, 'tdmpc2/Ant-v5': {'1': 5}}). The TD-MPC2 config has the field as well (all 50 runs d = 1).
- Baseline `batch_size`: {'mbpo/Ant-v5': {'256': 5}, 'mbpo/HalfCheetah-v5': {'256': 5}, 'sac/Ant-v5': {'256': 5}, 'sac/HalfCheetah-v5': {'256': 5}, 'td3/Ant-v5': {'100': 5}, 'td3/HalfCheetah-v5': {'100': 5}, 'tdmpc2/Ant-v5': {'256': 5}, 'tdmpc2/HalfCheetah-v5': {'256': 5}} (TD3 baseline 100, others 256); baseline `hidden_sizes`: (1024, 1024) for SAC/MBPO/TD3, no such field for TD-MPC2.

### 2. Log checks restricted to the canonical runs

**2a. `flop_analysis/check_totals.py` checks, canonical runs only** (restated in the script with the same definitions as `check_totals.py` L1-17; per-run CSV `flop_analysis/output/per_run_energy_per_flop.csv`):

| check | what it tests | runs checked | violations | max relative error |
|---|---|---|---|---|
| 1 | TOTAL energy == Σ training-segment energies of `segment_energy.json` (rtol 1e-12) | 280 | 0 | 0 |
| 2 | TOTAL flops == Σ matmul-row flops (exact integers) | 280 | 0 | exact |
| 3 | Σ allocated sub-segments == measured `gradient_updates` (rtol 1e-9) | 280 | 0 | 4.77e-13 |
| coverage | exactly one TOTAL row per run, no unexpected segment keys | 280 | 0 | – |

No violations on the canonical runs.

**2b. Scans for the statements "zero RAPL fallback / geolocation fallback / thermal-gate timeout / clock-lock or governor permission failure"**

Patterns searched (regex; see `canonical_data_quality.py` `LOG_PATTERNS`/`SW`): RAPL fallback `default power consumption of 4 W per thread` (CodeCarbon string); geolocation `geoloc|fall(ing)? back to default country|Unable to retrieve|fallback.*country|country`; thermal-gate `reason': '<anything but reached_reference>`, `nvml_unavailable`, `No NVML reference`, timeout; gpu_control failures `Command failed|Command not found|Command timed out|Could not query GPU clock range|Could not read current CPU governor`. Which logger writes where (verified in code, see 3):

| message class | written by logger | goes to |
|---|---|---|
| `Thermal gate result: {...}` | `experiment_runner` (`experiment_runner.py:138`) | `run.log` + stdout |
| `Locked GPU…`, `Reset GPU…`, `Set CPU governor…` (INFO) and `Command failed…`, `Could not query…`, `Could not read current CPU governor` (WARNING) | `gpu_control` (`utils/gpu_control.py:21`, calls at L31-113) | **not `run.log`**; INFO is dropped, WARNING goes to stderr (Python's last-resort handler) |
| thermal-gate warnings (`NVML read failed`, `No NVML reference available; falling back…`, timeout warnings) and INFO | `thermal_gate` (`utils/thermal_gate.py:30`, calls at L62, 112, 153, 180, 202) | **not `run.log`**; same as above |
| `default power consumption of 4 W per thread`, `Multiple instances of codecarbon…`, RAPL/geolocation warnings | CodeCarbon's own logger (`log_level="warning"`, `experiment_runner.py:159`) | stderr only, format `[codecarbon WARNING @ hh:mm:ss] …` |

Results:

| scan | scope | n runs scanned | n runs with a hit |
|---|---|---|---|
| RAPL-fallback string | `run.log` of canonical runs | 280 | 0 (but this logger can never write to `run.log`) |
| geolocation fallback | `run.log` | 280 | 0 (same caveat) |
| thermal gate reason ≠ reached_reference / nvml unavailable | `run.log` and `metadata.json.thermal_gate` | 280 | 0: `reason` = {'reached_reference': 280} in run.log, {'reached_reference': 280} in metadata; `gated`: {'True': 280}; max `waited_seconds` 2.69e-05 |
| gpu_control failure strings | `run.log` | 280 | 0 (same caveat) |
| any `[WARNING]`/`[ERROR]` line | `run.log` | 280 | 0 / 0 |
| gpu_control failure, RAPL fallback, geolocation, thermal non-reference | **sweep logs** (stdout+stderr), per-run blocks | 270 canonical runs have a block (all but the 10 SAC baseline runs) | gpu_control 0, RAPL 0, geolocation 0, thermal 0 |

Every line in the sweep logs that is not a standard script/INFO line is one of: `[codecarbon WARNING @ …] Multiple instances of codecarbon are allowed to run at the same time.` (one per run), `Run artifacts written to: …` (one per run), and the 5 tracebacks of the TD-MPC2 false start (item 7). No other warning or error text is present in any sweep log.

**Reading:** over the 280 canonical runs there are 0 violations of each check. The `run.log` scans for RAPL/geolocation/gpu_control/thermal-warning strings are, however, **structurally unable to fire** (those loggers never write to `run.log`); only the thermal-gate `reason` line is real evidence there. The valid scan for the other statements is the sweep logs (stderr captured), which cover 270 of the 280 canonical runs; the 10 SAC baseline runs (SAC HalfCheetah-v5 / Ant-v5 baselines, 2026-09-07/08, session 0-1) have **no** stdout/stderr capture and therefore no scan for these statements.

### 3. What `run.log` can and cannot detect; data-side evidence per run group

**Logger setup (verified in code).** `experiment_runner.py:50-62` (`_setup_logging`) creates `logging.getLogger("experiment_runner")`, sets it to INFO and attaches a `FileHandler(run_dir/"run.log")` and a stdout `StreamHandler`. `utils/gpu_control.py:21` (`logger = logging.getLogger("gpu_control")`) and `utils/thermal_gate.py:30` (`logging.getLogger("thermal_gate")`) are separate loggers with no handler; a repo-wide search finds no `basicConfig`, no other `addHandler` and no `propagate` change outside `experiment_runner.py:60-61`. Consequence (standard Python logging behaviour): their INFO messages (including "Locked GPU 0 clocks to [2000, 2000] MHz", `gpu_control.py:80`) are dropped, WARNING messages go to stderr only. CodeCarbon logs through its own logger to stderr. So `run.log` of the canonical runs contains only the `experiment_runner` messages: run header, device, thermal-gate result dict, settle/baseline/warmup/epoch/tail lines, final segment summary. It contains **no** line about clock lock, persistence mode, governor or RAPL — in particular, a successful lock leaves no trace in any log (all 9 sweep logs: 0 occurrences of `Locked GPU`/`Reset GPU`/`CPU governor`).

**Evidence available** (counts over the 280 canonical runs). `metadata.json.experiment_config` records the *request* in all 280 runs: `lock_gpu_clocks` {'True': 280}, `gpu_min_clock_mhz` {'2000': 280}, `gpu_max_clock_mhz` {'2000': 280}, `set_persistence_mode` {'True': 280}, `set_cpu_performance_governor` {'True': 280}, `thermal_gate_enabled` {'True': 280}, `tracking_mode` {'machine': 280}. No file written by `gpu_control` exists (no readback of the applied clock/governor); `emissions_base_*.csv` / `emissions.csv` have no clock or temperature columns (only GPU/CPU/RAM power and energy, utilisation). Thermal-gate numbers are in `metadata.json` (`thermal_gate`: `gated`, `reason`, `waited_seconds`, `final_temp_c`, `final_power_w`; plus `run_start_*`/`run_end_*` temperature/power readings, missing for 4 runs — the 4 earliest SAC HalfCheetah baselines, which ran before that logging was added, see item 8).

Run groups and what exists for them (sweep-log coverage = the run's directory appears in a sweep log with stdout+stderr):

| run group | n runs | with sweep-log block (stderr captured) |
|---|---|---|
| baseline: SAC (HalfCheetah, Ant) | 10 | 0 (no log) |
| baseline: MBPO, TD3, TD-MPC2 (6 algo×env) | 30 | 30 (`results/{mbpo,td3,tdmpc2}/_*sweep.log`) |
| UTD sweeps (SAC/TD3/MBPO, d = 2, 4) | 60 | 60 |
| width sweep (256/512) | 60 | 60 |
| batch-size sweeps | 70 | 70 |
| MBPO rollout length (Ant) | 10 | 10 |
| TD-MPC2 num_q / horizon | 40 | 40 |

**Claim × run group × evidence level** ("applied" = positive record that it was applied; "requested" = in `metadata.json` only; "no failure reported" = requested and the stderr capture shows no failure warning; "none" = no data-side evidence):

| claim | SAC baselines (10) | other baselines (30) | UTD / width / batch / rollout-length / TD-MPC2 sweeps (240) |
|---|---|---|---|
| GPU clock lock (2000/2000 MHz) | requested only | requested; no `gpu_control` failure warning in stderr capture (0 hits) | same as other baselines |
| persistence mode | requested only | requested; no failure warning | same |
| CPU `performance` governor | requested only | requested; no failure warning (`cpupower` failure would print `Command failed`/`not found`) | same |
| thermal gate passed | **applied** (gate result + readings in `run.log` and `metadata.json`, `reached_reference`, waited ≈ 0 s) | applied | applied |
| temperature before run start | `run_start_temp_c` for 6 of 10 (4 earliest runs lack it; `final_temp_c` of the gate exists for all) | all 30 | all 240 |
| RAPL (not TDP fallback) | none directly; indirect: CPU idle power 14.5–18.3 W (median 15.7 W) varies per run, not a constant | no "4 W per thread" fallback warning in stderr capture; same indirect evidence | same |

So: for **no** run group is there a positive record that clock lock, persistence mode or governor were applied (the only positive control record is the thermal gate); for 270 of 280 runs a failure to apply them would have printed a warning that is absent in the stderr capture; for the 10 SAC baselines neither exists. One further indirect point: one dev run (not canonical, `results/mbpo/HalfCheetah-v5/seed_0/20260904_130915`) recorded 200/200 MHz in `metadata.json`, i.e. the setting is parameterised per run in the config, but the data cannot show it took effect. Cross-check with a different data source (e.g. the clock-frequency runs of Section 6) is outside the canonical set.

### 4. Execution order

- Campaign: first canonical run starts **2026-09-07 10:13:36.036705 UTC** (`results/sac/HalfCheetah-v5/seed_331/20260907_121335`), last canonical run ends **2026-09-19 16:10:36.823017 UTC**. Consecutive-run gaps (end of run i → start of run i+1): median 1.33 s, p90 1.42 s, p99 22.2 h, max 47.4 h; overlapping runs: 0.
- **Sessions** (threshold: gap > 2 h): **10** sessions. Number of sessions by threshold (hours → sessions): {'0.5': 10, '1': 10, '2': 10, '4': 9, '8': 8, '12': 6} — i.e. the split is the same for any threshold between 0.5 h and 2 h (the within-session gaps are ≈ 1.3-1.4 s).

| session | start (UTC) | end (UTC) | hours | runs | algo / env / configurations (runs each = 5) |
|---|---|---|---|---|---|
| 0 | 2026-09-07 10:13:36 | 2026-09-07 12:10:07 | 1.94 | 8 | sac/Ant/baseline; sac/HalfCheetah/baseline |
| 1 | 2026-09-08 09:52:30 | 2026-09-08 14:26:41 | 4.57 | 12 | sac/Ant/baseline; sac/HalfCheetah/updates_per_env_step=2; sac/HalfCheetah/updates_per_env_step=4 |
| 2 | 2026-09-08 17:44:55 | 2026-09-08 23:34:50 | 5.83 | 10 | mbpo/Ant/baseline; mbpo/HalfCheetah/baseline |
| 3 | 2026-09-09 10:22:04 | 2026-09-09 20:34:22 | 10.21 | 40 | sac/Ant/updates_per_env_step=2; sac/Ant/updates_per_env_step=4; td3/Ant/baseline; td3/Ant/updates_per_env_step=2; td3/Ant/updates_per_env_step=4; td3/HalfCheetah/baseline; td3/HalfCheetah/updates_per_env_step=2; td3/HalfCheetah/updates_per_env_step=4 |
| 4 | 2026-09-10 10:28:45 | 2026-09-10 17:53:11 | 7.41 | 10 | tdmpc2/Ant/baseline; tdmpc2/HalfCheetah/baseline |
| 5 | 2026-09-12 17:15:06 | 2026-09-13 10:44:47 | 17.49 | 60 | mbpo/Ant/hidden_sizes=[256, 256]; mbpo/Ant/hidden_sizes=[512, 512]; mbpo/HalfCheetah/hidden_sizes=[256, 256]; mbpo/HalfCheetah/hidden_sizes=[512, 512]; sac/Ant/hidden_sizes=[256, 256]; sac/Ant/hidden_sizes=[512, 512]; sac/HalfCheetah/hidden_sizes=[256, 256]; sac/HalfCheetah/hidden_sizes=[512, 512]; td3/Ant/hidden_sizes=[256, 256]; td3/Ant/hidden_sizes=[512, 512]; td3/HalfCheetah/hidden_sizes=[256, 256]; td3/HalfCheetah/hidden_sizes=[512, 512] |
| 6 | 2026-09-14 10:50:01 | 2026-09-15 03:09:10 | 16.32 | 20 | mbpo/Ant/updates_per_env_step=2; mbpo/Ant/updates_per_env_step=4; mbpo/HalfCheetah/updates_per_env_step=2; mbpo/HalfCheetah/updates_per_env_step=4 |
| 7 | 2026-09-15 10:39:00 | 2026-09-15 19:32:05 | 8.88 | 35 | mbpo/HalfCheetah/batch_size=1024; mbpo/HalfCheetah/batch_size=512; sac/HalfCheetah/batch_size=1024; sac/HalfCheetah/batch_size=512; td3/HalfCheetah/batch_size=1024; td3/HalfCheetah/batch_size=256; td3/HalfCheetah/batch_size=512 |
| 8 | 2026-09-17 10:10:44 | 2026-09-18 02:15:57 | 16.09 | 45 | mbpo/Ant/batch_size=1024; mbpo/Ant/batch_size=512; mbpo/Ant/rollout_max_length=1; mbpo/Ant/rollout_max_length=15; sac/Ant/batch_size=1024; sac/Ant/batch_size=512; td3/Ant/batch_size=1024; td3/Ant/batch_size=256; td3/Ant/batch_size=512 |
| 9 | 2026-09-18 10:21:00 | 2026-09-19 16:10:36 | 29.83 | 40 | tdmpc2/Ant/horizon=1; tdmpc2/Ant/horizon=5; tdmpc2/Ant/num_q=3; tdmpc2/Ant/num_q=7; tdmpc2/HalfCheetah/horizon=1; tdmpc2/HalfCheetah/horizon=5; tdmpc2/HalfCheetah/num_q=3; tdmpc2/HalfCheetah/num_q=7 |

(Sessions end at the machine's automatic power-off when a sweep completes, per the sweep scripts' header comments; the log files show the sweep start priming `sudo`. The overnight sessions 5, 6, 8, 9 each exceed 16 h.)

**Order class: configuration-major, fixed seed order, not randomised.** In time order the 280 runs form 56 blocks of consecutive runs sharing (algo, env, config); there are 56 distinct configurations, so every configuration is one contiguous block (non-contiguous configurations: 0); **56 of 56 blocks run the seeds in the order 331, 958, 14577, 43611, 85062**. Per sweep log the same holds (every 5-run block in canonical seed order; the seed changes between 100 % of consecutive runs inside a sweep):

| sweep log | runs | config blocks | block sizes | all blocks in canonical seed order | first launch | last launch |
|---|---|---|---|---|---|---|
| `results/_ant_overnight_sweep.log` | 45 | 9 | {'5': 9} | True | algo=sac env=Ant-v5 seed=331 batch_size=512 | algo=mbpo env=Ant-v5 seed=85062 rollout_max_length=15 |
| `results/_batch_size_sweep.log` | 35 | 7 | {'5': 7} | True | algo=sac env=HalfCheetah-v5 seed=331 batch_size=512 | algo=td3 env=HalfCheetah-v5 seed=85062 batch_size=1024 |
| `results/_utd_sweep.log` | 50 | 10 | {'5': 10} | True | algo=sac env=Ant-v5 seed=331 utd=2 | algo=mbpo env=Ant-v5 seed=85062 utd=4 |
| `results/_width_sweep.log` | 60 | 12 | {'5': 12} | True | algo=sac env=HalfCheetah-v5 seed=331 width=256x256 | algo=td3 env=Ant-v5 seed=85062 width=512x512 |
| `results/mbpo/_env_sweep.log` | 10 | 2 | {'5': 2} | True | algo=mbpo env=HalfCheetah-v5 seed=331 overrides=none | algo=mbpo env=Ant-v5 seed=85062 overrides=configs/overrides/mbpo_ant.json |
| `results/sac/HalfCheetah-v5/_utd_sweep.log` | 10 | 2 | {'5': 2} | True | algo=sac env=HalfCheetah-v5 seed=331 utd=2 | algo=sac env=HalfCheetah-v5 seed=85062 utd=4 |
| `results/td3/_env_sweep.log` | 10 | 2 | {'5': 2} | True | algo=td3 env=HalfCheetah-v5 seed=331 overrides=none warmup_steps=10000 | algo=td3 env=Ant-v5 seed=85062 overrides=none warmup_steps=10000 |
| `results/tdmpc2/_numq_horizon_sweep.log` | 40 | 8 | {'5': 8} | True | algo=tdmpc2 env=HalfCheetah-v5 seed=331 num_q=3 | algo=tdmpc2 env=Ant-v5 seed=85062 horizon=5 |
| `results/tdmpc2/_seed_sweep.log` | 10 | 2 | {'5': 2} | True | algo=tdmpc2 env=HalfCheetah-v5 seed=331 overrides=none | algo=tdmpc2 env=Ant-v5 seed=85062 overrides=configs/overrides/tdmpc2_ant.json |

Configurations are therefore confounded with run time (e.g. baselines 2026-09-07…09-10, width sweep 09-12/13, MBPO UTD 09-14/15, HalfCheetah batch 09-15, Ant batch + rollout length 09-17/18, TD-MPC2 num_q/horizon 09-18/19); full per-run times are in `04_execution_timeline.csv` (columns: run directory, algo, env, config, seed, start, end, session id, sweep family, sweep-log coverage, commit).

### 5. Were the seeds fixed before the canonical runs?

- Earliest canonical run: `results/sac/HalfCheetah-v5/seed_331/20260907_121335`, start **2026-09-07 10:13:36 UTC = 12:13:35 +02:00** (run-directory timestamp is local time).
- The commit whose HEAD the first 8 canonical runs recorded, `f23f448` "Setting baseline config for all experiments", is dated **2026-09-07 12:13:01 +02:00**, 34 s before the first run; `git log -S14577` over non-result paths shows it does **not** contain the seed list.
- First commit in which the string `14577` appears in any tracked file: `92dcee0` "Recorded runs for seeds - 331, 958, 14577, 43611, 85062", **2026-09-07 14:13:23 +02:00** — in its `results/` files only (run output of seeds already run: e.g. `results/sac/Ant-v5/seed_14577/20260907_135857`); 2 h after the first canonical run started. (`43611` and `85062` also occur in earlier unrelated tracked files, so `14577` is the discriminating seed.)
- First commit containing the seed list in a non-result, non-context file (code/scripts): `d1cebb4` "UTD sweep 2 and 4 across 5 seeds for HalfCheetah", **2026-09-08 18:58:16 +02:00** (`run_utd_sweep.sh:28` `SEEDS=(331 958 14577 43611 85062)`).
- Hence in git history the canonical seed list is **not** recorded before the first canonical run; its first appearance is in a results commit made after the first five runs had completed. If the seeds were chosen earlier (e.g. passed by hand on the command line), no artifact in the repository shows it. (Not determinable from the repo: when the seeds were chosen.)

### 6. UTD sweeps: completeness from the run directories

| algo / env / config | canonical runs on disk |
|---|---|
| mbpo/Ant-v5/updates_per_env_step=2 | 5 |
| mbpo/Ant-v5/updates_per_env_step=4 | 5 |
| mbpo/HalfCheetah-v5/updates_per_env_step=2 | 5 |
| mbpo/HalfCheetah-v5/updates_per_env_step=4 | 5 |
| sac/Ant-v5/updates_per_env_step=2 | 5 |
| sac/Ant-v5/updates_per_env_step=4 | 5 |
| sac/HalfCheetah-v5/updates_per_env_step=2 | 5 |
| sac/HalfCheetah-v5/updates_per_env_step=4 | 5 |
| td3/Ant-v5/updates_per_env_step=2 | 5 |
| td3/Ant-v5/updates_per_env_step=4 | 5 |
| td3/HalfCheetah-v5/updates_per_env_step=2 | 5 |
| td3/HalfCheetah-v5/updates_per_env_step=4 | 5 |

SAC (Ant 2 values × 5 seeds, HalfCheetah 2 × 5), TD3 (both envs 2 × 5), and MBPO (both envs 2 × 5) are all complete = 60 UTD runs (12 cells × 5). `results/_utd_sweep.log` is **not** overwritten (only its status JSON was): it is one appended log with 50 launch blocks = SAC-Ant 10 + TD3-HalfCheetah 10 + TD3-Ant 10 (blocks 1-30 of the first sweep, started 2026-09-09 13:45:43, `[i/30]`) and MBPO HalfCheetah + Ant 10 each (second session, `ecb7135`, started 2026-09-14 12:49:56, `[i/20]`). **TD3 launch blocks: 20** (expected 20 ✓), of which 20 have a run directory and a "Run complete" line, and 20 of those run directories exist on disk. `results/_utd_sweep_status.json` only describes the MBPO sweep (`total_runs` = 20); the SAC-HalfCheetah UTD runs come from `results/sac/HalfCheetah-v5/_utd_sweep.log` (10 blocks).

### 7. TD-MPC2 `num_q`/horizon false start

From `results/tdmpc2/_numq_horizon_sweep.log` (lines 1-65), all in the first attempt:

| launch | config | env | seed | timestamp | outcome | run dir created |
|---|---|---|---|---|---|---|
| [1/40], log line 3 | num_q=3 | HalfCheetah-v5 | 331 | 2026-09-18 12:18:39 | FileNotFoundError: [Errno 2] No such file or directory: 'configs/overrides/tdmpc2_num_q3.json' | no |
| [2/40], log line 15 | num_q=3 | HalfCheetah-v5 | 958 | 2026-09-18 12:18:40 | FileNotFoundError: [Errno 2] No such file or directory: 'configs/overrides/tdmpc2_num_q3.json' | no |
| [3/40], log line 27 | num_q=3 | HalfCheetah-v5 | 14577 | 2026-09-18 12:18:40 | FileNotFoundError: [Errno 2] No such file or directory: 'configs/overrides/tdmpc2_num_q3.json' | no |
| [4/40], log line 39 | num_q=3 | HalfCheetah-v5 | 43611 | 2026-09-18 12:18:41 | FileNotFoundError: [Errno 2] No such file or directory: 'configs/overrides/tdmpc2_num_q3.json' | no |
| [5/40], log line 51 | num_q=3 | HalfCheetah-v5 | 85062 | 2026-09-18 12:18:42 | FileNotFoundError: [Errno 2] No such file or directory: 'configs/overrides/tdmpc2_num_q3.json' | no |
| [6/40], log line 63 | num_q=3 | Ant-v5 | 331 | 2026-09-18 12:18:42 | no closing line, no traceback, no 'FAILED' line | no |
| [1/40], log line 66 | num_q=3 | HalfCheetah-v5 | 331 | 2026-09-18 12:21:00 | no closing line, no traceback, no 'FAILED' line | no |
| [2/40], log line 106 | num_q=3 | HalfCheetah-v5 | 958 | 2026-09-18 12:57:09 | no closing line, no traceback, no 'FAILED' line | no |

Launches 1-5 (HalfCheetah-v5, seeds 331, 958, 14577, 43611, 85062, `num_q=3`) each ended with `FileNotFoundError: … 'configs/overrides/tdmpc2_num_q3.json'` (raised in `run_experiment.py:103` `build_algo_config` → `open(overrides_path)`, i.e. before `experiment_runner` and thus before any GPU/CodeCarbon/thermal code) and a `FAILED (exit 1)` line. Launch 6 (Ant-v5, seed 331, `num_q=3`, 12:18:42) has a `Starting` line and **no** further line of any kind (no `=== Experiment run`, no `Run directory`, no traceback, no `FAILED`); the next log line is the second priming of `sudo` at 12:21:00 (2 min 18 s later), i.e. the script was stopped during this launch.

- **Run directories:** none of the 6 launches created one (no run directory in `results/tdmpc2/**` with a timestamp 12:18-12:20 on 2026-09-18: []; the first directories of the second attempt start at 12:21:00). The 40 canonical TD-MPC2 sweep runs all come from the second attempt (status file: {'done': 40}).
- **Clock lock for launch 6:** the log contains no line mentioning a clock lock, reset, governor or `nvidia-smi` between launch 6 and the next successful run (0 matching lines before the second priming). This is **not informative** by itself because `gpu_control` INFO messages are never logged (see 3), and WARNINGs would have appeared (none). What the log does show: the run header (`=== Experiment run: …`, `Run directory: …`), which `experiment_runner.py:87-88` writes *before* entering `GpuCpuGuard` (`experiment_runner.py:121`), was never written for launch 6, so the process cannot have reached the clock-lock call at that point (inference from code order; the Python process may still have been importing torch). For launches 1-5 the failure happens in `run_experiment.py:103`, before `run_experiment()` is called, so no clock lock was set or reset by them. Whether a lock was left set (e.g. process killed with SIGKILL after the lock) cannot be determined, but every later run re-locks (2000/2000) at entry and thermal-gates before starting.

### 8. Code consistency of commits

`git_commit` in `metadata.json` is `git rev-parse HEAD` at run start (`experiment_runner.py:_git_commit_hash`); it does not capture uncommitted edits, and there is no dirty flag. Distinct commits and where they were used (runs / canonical):

| commit | date (author) | subject | runs |
|---|---|---|---|
| `f23f448` | 2026-09-07T12:13:01+02:00 | Setting baseline config for all experiments | 8 |
| `92dcee0` | 2026-09-07T14:13:23+02:00 | Recorded runs for seeds - 331, 958, 14577, 43611, 85062 | 2 |
| `ecb98bb` | 2026-09-08T12:20:16+02:00 | Finished remaining baseline runs for Ant/v5 | 10 |
| `0bbbbb7` | 2026-09-08T19:44:29+02:00 | baseline config for mbpo | 10 |
| `58962d2` | 2026-09-09T12:15:13+02:00 | Added and tested td3 algorithm | 10 |
| `aa9186b` | 2026-09-09T13:42:54+02:00 | Setup UTD sweep run script for SAC and TD3 across both envs and 5 seeds | 30 |
| `b5b7e54` | 2026-09-10T12:27:49+02:00 | Setup new  algorithm tdmpc2 and tested recording | 10 |
| `7e672ef` | 2026-09-12T19:14:44+02:00 | Setting up width sweep for SAC, TD3 and MBPO | 60 |
| `ecb7135` | 2026-09-14T12:48:24+02:00 | Scheduling UTD sweep for MBPO | 20 |
| `9b4b44c` | 2026-09-15T12:36:51+02:00 | Preparing batch size sweep for Half Cheetah | 35 |
| `788ffcd` | 2026-09-17T12:10:20+02:00 | Scheduling batch sweep for ant and new rollout sweep for MBPO | 45 |
| `a3582b5` | 2026-09-18T12:17:49+02:00 | TDMPC2 sweep scripts | 40 |

Per algorithm × environment × sweep (commit(s) of the 5-seed block):

| algo | env | config | commit |
|---|---|---|---|
| mbpo | Ant-v5 | baseline | `0bbbbb7` |
| mbpo | Ant-v5 | batch_size=1024 | `788ffcd` |
| mbpo | Ant-v5 | batch_size=512 | `788ffcd` |
| mbpo | Ant-v5 | hidden_sizes=[256, 256] | `7e672ef` |
| mbpo | Ant-v5 | hidden_sizes=[512, 512] | `7e672ef` |
| mbpo | Ant-v5 | rollout_max_length=1 | `788ffcd` |
| mbpo | Ant-v5 | rollout_max_length=15 | `788ffcd` |
| mbpo | Ant-v5 | updates_per_env_step=2 | `ecb7135` |
| mbpo | Ant-v5 | updates_per_env_step=4 | `ecb7135` |
| mbpo | HalfCheetah-v5 | baseline | `0bbbbb7` |
| mbpo | HalfCheetah-v5 | batch_size=1024 | `9b4b44c` |
| mbpo | HalfCheetah-v5 | batch_size=512 | `9b4b44c` |
| mbpo | HalfCheetah-v5 | hidden_sizes=[256, 256] | `7e672ef` |
| mbpo | HalfCheetah-v5 | hidden_sizes=[512, 512] | `7e672ef` |
| mbpo | HalfCheetah-v5 | updates_per_env_step=2 | `ecb7135` |
| mbpo | HalfCheetah-v5 | updates_per_env_step=4 | `ecb7135` |
| sac | Ant-v5 | baseline | `92dcee0, f23f448` |
| sac | Ant-v5 | batch_size=1024 | `788ffcd` |
| sac | Ant-v5 | batch_size=512 | `788ffcd` |
| sac | Ant-v5 | hidden_sizes=[256, 256] | `7e672ef` |
| sac | Ant-v5 | hidden_sizes=[512, 512] | `7e672ef` |
| sac | Ant-v5 | updates_per_env_step=2 | `aa9186b` |
| sac | Ant-v5 | updates_per_env_step=4 | `aa9186b` |
| sac | HalfCheetah-v5 | baseline | `f23f448` |
| sac | HalfCheetah-v5 | batch_size=1024 | `9b4b44c` |
| sac | HalfCheetah-v5 | batch_size=512 | `9b4b44c` |
| sac | HalfCheetah-v5 | hidden_sizes=[256, 256] | `7e672ef` |
| sac | HalfCheetah-v5 | hidden_sizes=[512, 512] | `7e672ef` |
| sac | HalfCheetah-v5 | updates_per_env_step=2 | `ecb98bb` |
| sac | HalfCheetah-v5 | updates_per_env_step=4 | `ecb98bb` |
| td3 | Ant-v5 | baseline | `58962d2` |
| td3 | Ant-v5 | batch_size=1024 | `788ffcd` |
| td3 | Ant-v5 | batch_size=256 | `788ffcd` |
| td3 | Ant-v5 | batch_size=512 | `788ffcd` |
| td3 | Ant-v5 | hidden_sizes=[256, 256] | `7e672ef` |
| td3 | Ant-v5 | hidden_sizes=[512, 512] | `7e672ef` |
| td3 | Ant-v5 | updates_per_env_step=2 | `aa9186b` |
| td3 | Ant-v5 | updates_per_env_step=4 | `aa9186b` |
| td3 | HalfCheetah-v5 | baseline | `58962d2` |
| td3 | HalfCheetah-v5 | batch_size=1024 | `9b4b44c` |
| td3 | HalfCheetah-v5 | batch_size=256 | `9b4b44c` |
| td3 | HalfCheetah-v5 | batch_size=512 | `9b4b44c` |
| td3 | HalfCheetah-v5 | hidden_sizes=[256, 256] | `7e672ef` |
| td3 | HalfCheetah-v5 | hidden_sizes=[512, 512] | `7e672ef` |
| td3 | HalfCheetah-v5 | updates_per_env_step=2 | `aa9186b` |
| td3 | HalfCheetah-v5 | updates_per_env_step=4 | `aa9186b` |
| tdmpc2 | Ant-v5 | baseline | `b5b7e54` |
| tdmpc2 | Ant-v5 | horizon=1 | `a3582b5` |
| tdmpc2 | Ant-v5 | horizon=5 | `a3582b5` |
| tdmpc2 | Ant-v5 | num_q=3 | `a3582b5` |
| tdmpc2 | Ant-v5 | num_q=7 | `a3582b5` |
| tdmpc2 | HalfCheetah-v5 | baseline | `b5b7e54` |
| tdmpc2 | HalfCheetah-v5 | horizon=1 | `a3582b5` |
| tdmpc2 | HalfCheetah-v5 | horizon=5 | `a3582b5` |
| tdmpc2 | HalfCheetah-v5 | num_q=3 | `a3582b5` |
| tdmpc2 | HalfCheetah-v5 | num_q=7 | `a3582b5` |

**MBPO Ant-v5 `rollout_max_length=1` (commit `788ffcd`, all 5 seeds one commit) vs. the HalfCheetah-v5 baseline (commit `0bbbbb7`):** `git diff --stat 0bbbbb7 788ffcd -- algorithms experiment_runner.py run_experiment.py configs/config.py utils`:

```
algorithms/td3.py    | 372 ++++++++++++++++++++++
 algorithms/tdmpc2.py | 854 +++++++++++++++++++++++++++++++++++++++++++++++++++
 configs/config.py    | 147 +++++++++
 experiment_runner.py |   8 +
 4 files changed, 1381 insertions(+)
```
Commits touching these paths in between: `b5b7e54|2026-09-10T12:27:49+02:00|Setup new  algorithm tdmpc2 and tested recording`; `58962d2|2026-09-09T12:15:13+02:00|Added and tested td3 algorithm`. Only additions: `algorithms/td3.py` and `algorithms/tdmpc2.py` (new algorithms), new config dataclasses/`ALGO_CONFIGS` entries in `configs/config.py` (+147 lines, no deletions), and 8 added lines in `experiment_runner.py` (algorithm dispatch). `algorithms/{sac,mbpo,dynamics_model,termination_fns,replay_buffer}.py`, `utils/` and `run_experiment.py` are unchanged between the two commits. (Full diff incl. docs/scripts/overrides: `04_results.json` → `diff_stat_full_hc_baseline_to_ant_rollout1`.)

Training/measurement-code diff between consecutive commits used by canonical runs (paths: `algorithms/`, `experiment_runner.py`, `run_experiment.py`, `configs/config.py`, `utils/`):

- `f23f448..92dcee0`: 
```
experiment_runner.py                           |  20 +++++++++++++++++---
 utils/__pycache__/gpu_control.cpython-314.pyc  | Bin 9484 -> 9489 bytes
 utils/__pycache__/thermal_gate.cpython-314.pyc | Bin 9649 -> 10314 bytes
 utils/thermal_gate.py                          |  10 ++++++++++
 4 files changed, 27 insertions(+), 3 deletions(-)
```
- `92dcee0..ecb98bb`: no change in these paths
- `ecb98bb..0bbbbb7`: 
```
algorithms/__pycache__/replay_buffer.cpython-314.pyc | Bin 4758 -> 0 bytes
 algorithms/__pycache__/sac.cpython-314.pyc           | Bin 25182 -> 0 bytes
 configs/config.py                                    |   2 +-
 utils/__pycache__/gpu_control.cpython-314.pyc        | Bin 9489 -> 0 bytes
 utils/__pycache__/thermal_gate.cpython-314.pyc       | Bin 10314 -> 0 bytes
 5 files changed, 1 insertion(+), 1 deletion(-)
```
- `0bbbbb7..58962d2`: 
```
algorithms/td3.py    | 372 +++++++++++++++++++++++++++++++++++++++++++++++++++
 configs/config.py    |  49 +++++++
 experiment_runner.py |   4 +
 3 files changed, 425 insertions(+)
```
- `58962d2..aa9186b`: no change in these paths
- `aa9186b..b5b7e54`: 
```
algorithms/tdmpc2.py | 854 +++++++++++++++++++++++++++++++++++++++++++++++++++
 configs/config.py    |  98 ++++++
 experiment_runner.py |   4 +
 3 files changed, 956 insertions(+)
```
- `b5b7e54..7e672ef`: no change in these paths
- `7e672ef..ecb7135`: no change in these paths
- `ecb7135..9b4b44c`: no change in these paths
- `9b4b44c..788ffcd`: no change in these paths
- `788ffcd..a3582b5`: no change in these paths

Only two changes touch run-time code between commits: (i) `f23f448..92dcee0` adds the `run_start_*`/`run_end_*` temperature snapshot (`utils/thermal_gate.py` `read_current_state`, +10 lines; `experiment_runner.py` +20/−3, which also stops writing `thermal_gate_reference_file` into the metadata) — logging only, no change to training or measurement; (ii) `ecb98bb..0bbbbb7` changes one line in `configs/config.py`: the `MBPOConfig.hidden_sizes` default from (256, 256) to (1024, 1024) (`configs/config.py:94`), i.e. the MBPO baseline architecture was set in code only at the commit used by all MBPO baseline runs (all 10 MBPO baselines record `hidden_sizes` (1024, 1024) in their metadata); it also deletes tracked `.pyc` files.

Note on SAC runs: the 8 first runs all record `f23f448`, but the 4 earliest lack `run_start_temp_c` in their metadata while the next 4 (same recorded commit) have it — so the working tree was edited (uncommitted) between run 4 and run 5; `git_commit` therefore under-describes the code for the 8 SAC baseline runs of session 0.

## Caveats / things that look odd

- The `run.log` grep scans quoted in CLAUDE.md ("zero RAPL-fallback … across all 300 run.log files") cannot detect those events by construction (item 2-3); the claim is supportable only from sweep logs (270 of 280 canonical runs) and not at all for the 10 SAC baseline runs.
- No positive record of a *successful* clock lock / governor / persistence mode exists anywhere (INFO messages are never logged, no readback in metadata).
- SAC baseline runs (session 0/1) have no stdout/stderr log, 4 of them lack the run-start temperature, and ran with uncommitted working-tree changes relative to the recorded commit.
- The seed list first appears in git after the first canonical runs (item 5); the run order is configuration-major with fixed seed order, so config and wall-clock time (and idle level, Task 1) are confounded.
- Task statement lists "7/7 SAC, 8/8 TD3, 7 HalfCheetah and 9 Ant (MBPO), 5/5 TD-MPC2 runs per seed": verified exactly (table in item 1).
