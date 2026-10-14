"""Builds 04_canonical_data_quality.md from 04_results.json + 04_canonical_run_list.csv (written by canonical_data_quality.py)."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT

R = json.load(open(os.path.join(OUT, "04_results.json")))
runs = pd.read_csv(os.path.join(OUT, "04_canonical_run_list.csv"))
L = []
P = L.append

P("# Task 4: Canonical-only data quality and execution order\n")
P("## Question\n")
P("Do the statements of `sec:data-quality` hold on the 280 canonical runs alone, what data-side evidence exists for the confound controls, in which order were the runs executed, and what code/commits produced them?\n")
P("## Method\n")
P("- Script: `contexts/step1/scripts/canonical_data_quality.py` (repo `.venv` Python, read-only on `results/`, `configs/`, `scripts/`), then `make_04_md.py` for this file. Shared canonical list: `common.py` → `04_canonical_run_list.csv` (280 rows: algo, env, seed, config label, run directory, start/end, commit).")
P("- Config label = difference of a run's `algo_config`/`experiment_config` from the modal config of its (algo, env) group among canonical runs (same convention as `idle_power_analysis.py` `variant`).")
P("- Inputs: `metadata.json`, `segment_energy.json`, `training_metrics.json`, `emissions.csv`, `emissions_base_*.csv`, `run.log` of each canonical run; the 9 sweep logs `results/**/_*.log` (stdout+stderr of the sweep scripts, via `2>&1 | tee`, e.g. `scripts/run_width_sweep.sh:191`); `flop_analysis/output/per_run_energy_per_flop.csv` for the `check_totals.py` checks; `git log`/`git diff` of this checkout (61 commits, all 12 commits recorded in the metadata exist locally).")
P("- Outputs: `04_canonical_run_list.csv`, `04_execution_timeline.csv`, `04_results.json` (all numbers below, full detail).\n")

P("## Results\n")
P("### 1. Run count and completeness\n")
P(f"- Canonical runs found: **{R['n_runs']}** = {R['n_configurations']} configurations × 5 seeds; duplicates for the same (algo, env, config, seed): {len(R['duplicates_same_algo_env_config_seed'])}; configurations without all 5 seeds: {len(R['configs_without_all_5_seeds'])}. Total run directories under `results/`: {R['total_run_dirs']} (20 non-canonical: " + ", ".join(f"{k} ×{v}" for k, v in R['non_canonical_run_dirs'].items()) + ").")
P(f"- Missing files (of `metadata.json`, `segment_energy.json`, `training_metrics.json`, `emissions.csv`, `emissions_base_*.csv`, `run.log`, `end_time_utc` in metadata): **{len(R['missing_files'])}**.\n")
P("| algo / env | configs per seed (runs per seed) | seeds × configs = runs | expected (task) |\n|---|---|---|---|")
exp = {"sac": 7, "td3": 8, "tdmpc2": 5}
for k, v in R["per_algo_env_configs_per_seed"].items():
    e = exp.get(k.split("/")[0], 7 if k.endswith("HalfCheetah-v5") else 9)
    P(f"| {k} | {v} | 5 × {v} = {5*v} | {e} {'✓' if e == v else '✗'} |")
P("\nProtocol values over the 280 runs (distinct value: count):\n")
P(f"- `train_steps`: {R['train_steps']} (N_env = 100 000 measured env steps in every run: {R['N_env_equals_train_steps_all']}); last `epochs[-1].env_step`: {R['last_epoch_env_step_distinct']}.")
P(f"- `steps_per_epoch`: {R['steps_per_epoch']}; epochs in `training_metrics.json`: {R['epochs_in_training_metrics']}; `rollout_i` task rows per run: {R['rollout_task_rows_per_run']} → 100 epochs everywhere.")
P(f"- `warmup_steps` by algorithm: {R['warmup_steps_by_algo']} (SAC/MBPO/TD-MPC2 5000, TD3 10000). Warmup steps are *additional* to the 100 000 measured steps (e.g. final buffer size 105 000 for SAC, 110 000 for TD3 in `run.log`).")
P(f"- (idle_baseline_seconds, idle_tail_seconds, settle_seconds): {R['idle_secs']}.")
P(f"- `updates_per_env_step` d by algorithm over all configs: {R['updates_per_env_step_by_algo_all_configs']}. **Baseline configs: d = 1 in all 8 algo×env baselines** ({R['updates_per_env_step_baseline']}). The TD-MPC2 config has the field as well (all 50 runs d = 1).")
P(f"- Baseline `batch_size`: {R['baseline_batch_size']} (TD3 baseline 100, others 256); baseline `hidden_sizes`: (1024, 1024) for SAC/MBPO/TD3, no such field for TD-MPC2.\n")

P("### 2. Log checks restricted to the canonical runs\n")
ct = R["check_totals_canonical"]
P("**2a. `flop_analysis/check_totals.py` checks, canonical runs only** (restated in the script with the same definitions as `check_totals.py` L1-17; per-run CSV `flop_analysis/output/per_run_energy_per_flop.csv`):\n")
P("| check | what it tests | runs checked | violations | max relative error |\n|---|---|---|---|---|")
P(f"| 1 | TOTAL energy == Σ training-segment energies of `segment_energy.json` (rtol 1e-12) | {ct['check1_runs']} | {sum(1 for f in ct['failures'] if f.startswith('[check1]'))} | {ct['check1_max_rel_err']:.3g} |")
P(f"| 2 | TOTAL flops == Σ matmul-row flops (exact integers) | {ct['check2_runs']} | {sum(1 for f in ct['failures'] if f.startswith('[check2]'))} | exact |")
P(f"| 3 | Σ allocated sub-segments == measured `gradient_updates` (rtol 1e-9) | {ct['check3_runs']} | {sum(1 for f in ct['failures'] if f.startswith('[check3]'))} | {ct['check3_max_rel_err']:.3g} |")
P(f"| coverage | exactly one TOTAL row per run, no unexpected segment keys | {ct['runs_with_csv_rows']} | {sum(1 for f in ct['failures'] if f.startswith('[coverage]'))} | – |")
P("\nNo violations on the canonical runs.\n")
P("**2b. Scans for the statements \"zero RAPL fallback / geolocation fallback / thermal-gate timeout / clock-lock or governor permission failure\"**\n")
P("Patterns searched (regex; see `canonical_data_quality.py` `LOG_PATTERNS`/`SW`): RAPL fallback `default power consumption of 4 W per thread` (CodeCarbon string); geolocation `geoloc|fall(ing)? back to default country|Unable to retrieve|fallback.*country|country`; thermal-gate `reason': '<anything but reached_reference>`, `nvml_unavailable`, `No NVML reference`, timeout; gpu_control failures `Command failed|Command not found|Command timed out|Could not query GPU clock range|Could not read current CPU governor`. Which logger writes where (verified in code, see 3):\n")
P("| message class | written by logger | goes to |\n|---|---|---|")
P("| `Thermal gate result: {...}` | `experiment_runner` (`experiment_runner.py:138`) | `run.log` + stdout |")
P("| `Locked GPU…`, `Reset GPU…`, `Set CPU governor…` (INFO) and `Command failed…`, `Could not query…`, `Could not read current CPU governor` (WARNING) | `gpu_control` (`utils/gpu_control.py:21`, calls at L31-113) | **not `run.log`**; INFO is dropped, WARNING goes to stderr (Python's last-resort handler) |")
P("| thermal-gate warnings (`NVML read failed`, `No NVML reference available; falling back…`, timeout warnings) and INFO | `thermal_gate` (`utils/thermal_gate.py:30`, calls at L62, 112, 153, 180, 202) | **not `run.log`**; same as above |")
P("| `default power consumption of 4 W per thread`, `Multiple instances of codecarbon…`, RAPL/geolocation warnings | CodeCarbon's own logger (`log_level=\"warning\"`, `experiment_runner.py:159`) | stderr only, format `[codecarbon WARNING @ hh:mm:ss] …` |")
P("")
P("Results:\n")
P("| scan | scope | n runs scanned | n runs with a hit |\n|---|---|---|---|")
rs = R["run_log_scan_hits"]
P(f"| RAPL-fallback string | `run.log` of canonical runs | 280 | {rs['rapl_fallback_4W_per_thread']['n_runs']} (but this logger can never write to `run.log`) |")
P(f"| geolocation fallback | `run.log` | 280 | {rs['geolocation_fallback']['n_runs']} (same caveat) |")
P(f"| thermal gate reason ≠ reached_reference / nvml unavailable | `run.log` and `metadata.json.thermal_gate` | 280 | 0: `reason` = {R['run_log_thermal_reason_counts']} in run.log, {R['metadata_thermal_gate_reason']} in metadata; `gated`: {R['metadata_thermal_gate_gated']}; max `waited_seconds` {R['metadata_thermal_waited_s_max']:.3g} |")
P(f"| gpu_control failure strings | `run.log` | 280 | {rs['gpu_control_failure']['n_runs']} (same caveat) |")
P(f"| any `[WARNING]`/`[ERROR]` line | `run.log` | 280 | {rs['any_WARNING_level_line']['n_runs']} / {rs['any_ERROR_level_line']['n_runs']} |")
sw = R["sweep_log_block_hits"]
P(f"| gpu_control failure, RAPL fallback, geolocation, thermal non-reference | **sweep logs** (stdout+stderr), per-run blocks | {R['sweep_log_coverage_total']} canonical runs have a block (all but the 10 SAC baseline runs) | gpu_control {sw['gpu_control_failure']['n_blocks']}, RAPL {sw['rapl_fallback']['n_blocks']}, geolocation {sw['geolocation']['n_blocks']}, thermal {sw['thermal_timeout_or_nvml']['n_blocks']} |")
P("")
P("Every line in the sweep logs that is not a standard script/INFO line is one of: `[codecarbon WARNING @ …] Multiple instances of codecarbon are allowed to run at the same time.` (one per run), `Run artifacts written to: …` (one per run), and the 5 tracebacks of the TD-MPC2 false start (item 7). No other warning or error text is present in any sweep log.\n")
P("**Reading:** over the 280 canonical runs there are 0 violations of each check. The `run.log` scans for RAPL/geolocation/gpu_control/thermal-warning strings are, however, **structurally unable to fire** (those loggers never write to `run.log`); only the thermal-gate `reason` line is real evidence there. The valid scan for the other statements is the sweep logs (stderr captured), which cover 270 of the 280 canonical runs; the 10 SAC baseline runs (SAC HalfCheetah-v5 / Ant-v5 baselines, 2026-09-07/08, session 0-1) have **no** stdout/stderr capture and therefore no scan for these statements.\n")

P("### 3. What `run.log` can and cannot detect; data-side evidence per run group\n")
P("**Logger setup (verified in code).** `experiment_runner.py:50-62` (`_setup_logging`) creates `logging.getLogger(\"experiment_runner\")`, sets it to INFO and attaches a `FileHandler(run_dir/\"run.log\")` and a stdout `StreamHandler`. `utils/gpu_control.py:21` (`logger = logging.getLogger(\"gpu_control\")`) and `utils/thermal_gate.py:30` (`logging.getLogger(\"thermal_gate\")`) are separate loggers with no handler; a repo-wide search finds no `basicConfig`, no other `addHandler` and no `propagate` change outside `experiment_runner.py:60-61`. Consequence (standard Python logging behaviour): their INFO messages (including \"Locked GPU 0 clocks to [2000, 2000] MHz\", `gpu_control.py:80`) are dropped, WARNING messages go to stderr only. CodeCarbon logs through its own logger to stderr. So `run.log` of the canonical runs contains only the `experiment_runner` messages: run header, device, thermal-gate result dict, settle/baseline/warmup/epoch/tail lines, final segment summary. It contains **no** line about clock lock, persistence mode, governor or RAPL — in particular, a successful lock leaves no trace in any log (all 9 sweep logs: 0 occurrences of `Locked GPU`/`Reset GPU`/`CPU governor`).\n")
P(f"**Evidence available** (counts over the 280 canonical runs). `metadata.json.experiment_config` records the *request* in all 280 runs: `lock_gpu_clocks` {R['meta_requested']['lock_gpu_clocks']}, `gpu_min_clock_mhz` {R['meta_requested']['gpu_min_clock_mhz']}, `gpu_max_clock_mhz` {R['meta_requested']['gpu_max_clock_mhz']}, `set_persistence_mode` {R['meta_requested']['set_persistence_mode']}, `set_cpu_performance_governor` {R['meta_requested']['set_cpu_performance_governor']}, `thermal_gate_enabled` {R['meta_requested']['thermal_gate_enabled']}, `tracking_mode` {R['meta_requested']['tracking_mode']}. No file written by `gpu_control` exists (no readback of the applied clock/governor); `emissions_base_*.csv` / `emissions.csv` have no clock or temperature columns (only GPU/CPU/RAM power and energy, utilisation). Thermal-gate numbers are in `metadata.json` (`thermal_gate`: `gated`, `reason`, `waited_seconds`, `final_temp_c`, `final_power_w`; plus `run_start_*`/`run_end_*` temperature/power readings, missing for {R['metadata_thermal_missing_fields']['run_start_temp_c']} runs — the 4 earliest SAC HalfCheetah baselines, which ran before that logging was added, see item 8).\n")
P("Run groups and what exists for them (sweep-log coverage = the run's directory appears in a sweep log with stdout+stderr):\n")
P("| run group | n runs | with sweep-log block (stderr captured) |\n|---|---|---|")
base = R["evidence_coverage_baseline_by_algo_env"]
P(f"| baseline: SAC (HalfCheetah, Ant) | 10 | 0 (no log) |")
P(f"| baseline: MBPO, TD3, TD-MPC2 (6 algo×env) | 30 | 30 (`results/{{mbpo,td3,tdmpc2}}/_*sweep.log`) |")
for fam, nm in (("utd", "UTD sweeps (SAC/TD3/MBPO, d = 2, 4)"), ("width", "width sweep (256/512)"), ("batch", "batch-size sweeps"), ("mbpo_rollout_length", "MBPO rollout length (Ant)"), ("tdmpc2_numq_horizon", "TD-MPC2 num_q / horizon")):
    v = R["evidence_coverage_by_family"][fam]
    P(f"| {nm} | {v['n']} | {v['in_sweep_log']} |")
P("\n**Claim × run group × evidence level** (\"applied\" = positive record that it was applied; \"requested\" = in `metadata.json` only; \"no failure reported\" = requested and the stderr capture shows no failure warning; \"none\" = no data-side evidence):\n")
P("| claim | SAC baselines (10) | other baselines (30) | UTD / width / batch / rollout-length / TD-MPC2 sweeps (240) |\n|---|---|---|---|")
P("| GPU clock lock (2000/2000 MHz) | requested only | requested; no `gpu_control` failure warning in stderr capture (0 hits) | same as other baselines |")
P("| persistence mode | requested only | requested; no failure warning | same |")
P("| CPU `performance` governor | requested only | requested; no failure warning (`cpupower` failure would print `Command failed`/`not found`) | same |")
P("| thermal gate passed | **applied** (gate result + readings in `run.log` and `metadata.json`, `reached_reference`, waited ≈ 0 s) | applied | applied |")
P("| temperature before run start | `run_start_temp_c` for 6 of 10 (4 earliest runs lack it; `final_temp_c` of the gate exists for all) | all 30 | all 240 |")
P(f"| RAPL (not TDP fallback) | none directly; indirect: CPU idle power {R['cpu_idle_head_w_corr_min_median_max'][0]:.1f}–{R['cpu_idle_head_w_corr_min_median_max'][2]:.1f} W (median {R['cpu_idle_head_w_corr_min_median_max'][1]:.1f} W) varies per run, not a constant | no \"4 W per thread\" fallback warning in stderr capture; same indirect evidence | same |")
P("\nSo: for **no** run group is there a positive record that clock lock, persistence mode or governor were applied (the only positive control record is the thermal gate); for 270 of 280 runs a failure to apply them would have printed a warning that is absent in the stderr capture; for the 10 SAC baselines neither exists. One further indirect point: one dev run (not canonical, `results/mbpo/HalfCheetah-v5/seed_0/20260904_130915`) recorded 200/200 MHz in `metadata.json`, i.e. the setting is parameterised per run in the config, but the data cannot show it took effect. Cross-check with a different data source (e.g. the clock-frequency runs of Section 6) is outside the canonical set.\n")

P("### 4. Execution order\n")
P(f"- Campaign: first canonical run starts **{R['campaign_first_start_utc']} UTC** (`results/{R['earliest_canonical_run_dir']}`), last canonical run ends **{R['campaign_last_end_utc']} UTC**. Consecutive-run gaps (end of run i → start of run i+1): median {R['gap_seconds_quantiles_p0_50_90_99_100'][1]:.2f} s, p90 {R['gap_seconds_quantiles_p0_50_90_99_100'][2]:.2f} s, p99 {R['gap_seconds_quantiles_p0_50_90_99_100'][3]/3600:.1f} h, max {R['gap_seconds_quantiles_p0_50_90_99_100'][4]/3600:.1f} h; overlapping runs: {R['gap_negative_overlap_count']}.")
P(f"- **Sessions** (threshold: gap > 2 h): **{R['n_sessions_gap_gt_2h']}** sessions. Number of sessions by threshold (hours → sessions): {R['n_sessions_by_threshold_hours']} — i.e. the split is the same for any threshold between 0.5 h and 2 h (the within-session gaps are ≈ 1.3-1.4 s).\n")
P("| session | start (UTC) | end (UTC) | hours | runs | algo / env / configurations (runs each = 5) |\n|---|---|---|---|---|---|")
for s in R["sessions"]:
    grp = "; ".join(f"{k}" for k in s["groups"].keys())
    P(f"| {s['session']} | {s['start_utc'][:19]} | {s['end_utc'][:19]} | {s['hours']} | {s['n_runs']} | {grp} |")
P("\n(Sessions end at the machine's automatic power-off when a sweep completes, per the sweep scripts' header comments; the log files show the sweep start priming `sudo`. The overnight sessions 5, 6, 8, 9 each exceed 16 h.)\n")
P(f"**Order class: configuration-major, fixed seed order, not randomised.** In time order the 280 runs form {R['n_time_ordered_blocks']} blocks of consecutive runs sharing (algo, env, config); there are {R['n_distinct_configs']} distinct configurations, so every configuration is one contiguous block (non-contiguous configurations: {len(R['config_noncontiguous'])}); **{R['blocks_of_5_in_canonical_seed_order']} of {R['n_time_ordered_blocks']} blocks run the seeds in the order 331, 958, 14577, 43611, 85062**. Per sweep log the same holds (every 5-run block in canonical seed order; the seed changes between 100 % of consecutive runs inside a sweep):\n")
P("| sweep log | runs | config blocks | block sizes | all blocks in canonical seed order | first launch | last launch |\n|---|---|---|---|---|---|---|")
for lg, v in R["log_order_classification"].items():
    P(f"| `{lg}` | {v['n_runs']} | {v['n_config_blocks']} | {v['block_sizes']} | {v['all_5blocks_in_canonical_seed_order']} | {v['first']} | {v['last']} |")
P("\nConfigurations are therefore confounded with run time (e.g. baselines 2026-09-07…09-10, width sweep 09-12/13, MBPO UTD 09-14/15, HalfCheetah batch 09-15, Ant batch + rollout length 09-17/18, TD-MPC2 num_q/horizon 09-18/19); full per-run times are in `04_execution_timeline.csv` (columns: run directory, algo, env, config, seed, start, end, session id, sweep family, sweep-log coverage, commit).\n")

P("### 5. Were the seeds fixed before the canonical runs?\n")
P(f"- Earliest canonical run: `results/{R['earliest_canonical_run_dir']}`, start **2026-09-07 10:13:36 UTC = 12:13:35 +02:00** (run-directory timestamp is local time).")
P("- The commit whose HEAD the first 8 canonical runs recorded, `f23f448` \"Setting baseline config for all experiments\", is dated **2026-09-07 12:13:01 +02:00**, 34 s before the first run; `git log -S14577` over non-result paths shows it does **not** contain the seed list.")
P("- First commit in which the string `14577` appears in any tracked file: `92dcee0` \"Recorded runs for seeds - 331, 958, 14577, 43611, 85062\", **2026-09-07 14:13:23 +02:00** — in its `results/` files only (run output of seeds already run: e.g. `results/sac/Ant-v5/seed_14577/20260907_135857`); 2 h after the first canonical run started. (`43611` and `85062` also occur in earlier unrelated tracked files, so `14577` is the discriminating seed.)")
P("- First commit containing the seed list in a non-result, non-context file (code/scripts): `d1cebb4` \"UTD sweep 2 and 4 across 5 seeds for HalfCheetah\", **2026-09-08 18:58:16 +02:00** (`run_utd_sweep.sh:28` `SEEDS=(331 958 14577 43611 85062)`).")
P("- Hence in git history the canonical seed list is **not** recorded before the first canonical run; its first appearance is in a results commit made after the first five runs had completed. If the seeds were chosen earlier (e.g. passed by hand on the command line), no artifact in the repository shows it. (Not determinable from the repo: when the seeds were chosen.)\n")

P("### 6. UTD sweeps: completeness from the run directories\n")
P("| algo / env / config | canonical runs on disk |\n|---|---|")
for k, v in R["utd_counts_by_algo_env_config"].items():
    P(f"| {k} | {v} |")
P(f"\nSAC (Ant 2 values × 5 seeds, HalfCheetah 2 × 5), TD3 (both envs 2 × 5), and MBPO (both envs 2 × 5) are all complete = 60 UTD runs (12 cells × 5). `results/_utd_sweep.log` is **not** overwritten (only its status JSON was): it is one appended log with {R['utd_log_blocks_total']} launch blocks = SAC-Ant {R['utd_log_blocks_by_algo_env'].get('sac/Ant-v5')} + TD3-HalfCheetah {R['utd_log_blocks_by_algo_env'].get('td3/HalfCheetah-v5')} + TD3-Ant {R['utd_log_blocks_by_algo_env'].get('td3/Ant-v5')} (blocks 1-30 of the first sweep, started 2026-09-09 13:45:43, `[i/30]`) and MBPO HalfCheetah + Ant 10 each (second session, `ecb7135`, started 2026-09-14 12:49:56, `[i/20]`). **TD3 launch blocks: {R['utd_log_td3_blocks']}** (expected 20 ✓), of which {R['utd_log_td3_blocks_with_run_dir_and_complete']} have a run directory and a \"Run complete\" line, and {R['utd_log_td3_blocks_ok_in_run_dirs']} of those run directories exist on disk. `results/_utd_sweep_status.json` only describes the MBPO sweep (`total_runs` = {R['utd_status_file_mixed']}); the SAC-HalfCheetah UTD runs come from `results/sac/HalfCheetah-v5/_utd_sweep.log` (10 blocks).\n")

P("### 7. TD-MPC2 `num_q`/horizon false start\n")
P("From `results/tdmpc2/_numq_horizon_sweep.log` (lines 1-65), all in the first attempt:\n")
P("| launch | config | env | seed | timestamp | outcome | run dir created |\n|---|---|---|---|---|---|---|")
for b in R["tdmpc2_false_start_blocks"]:
    if b["kind"] != "launch" or b["idx"] > 6: continue
    d = dict(x.split("=", 1) for x in b["desc"].split() if "=" in x)
    out = (b["error"][0] if b["error"] else "no closing line, no traceback, no 'FAILED' line")
    P(f"| [{b['idx']}/40], log line {b['line']} | num_q={d.get('num_q')} | {d.get('env')} | {d.get('seed')} | {b['ts']} | {out} | no |")
P("\nLaunches 1-5 (HalfCheetah-v5, seeds 331, 958, 14577, 43611, 85062, `num_q=3`) each ended with `FileNotFoundError: … 'configs/overrides/tdmpc2_num_q3.json'` (raised in `run_experiment.py:103` `build_algo_config` → `open(overrides_path)`, i.e. before `experiment_runner` and thus before any GPU/CodeCarbon/thermal code) and a `FAILED (exit 1)` line. Launch 6 (Ant-v5, seed 331, `num_q=3`, 12:18:42) has a `Starting` line and **no** further line of any kind (no `=== Experiment run`, no `Run directory`, no traceback, no `FAILED`); the next log line is the second priming of `sudo` at 12:21:00 (2 min 18 s later), i.e. the script was stopped during this launch.\n")
P(f"- **Run directories:** none of the 6 launches created one (no run directory in `results/tdmpc2/**` with a timestamp 12:18-12:20 on 2026-09-18: {R['tdmpc2_run_dirs_20260918_1218_to_1220']}; the first directories of the second attempt start at 12:21:00). The 40 canonical TD-MPC2 sweep runs all come from the second attempt (status file: {R['tdmpc2_status_json_done_counts']}).")
P("- **Clock lock for launch 6:** the log contains no line mentioning a clock lock, reset, governor or `nvidia-smi` between launch 6 and the next successful run (0 matching lines before the second priming). This is **not informative** by itself because `gpu_control` INFO messages are never logged (see 3), and WARNINGs would have appeared (none). What the log does show: the run header (`=== Experiment run: …`, `Run directory: …`), which `experiment_runner.py:87-88` writes *before* entering `GpuCpuGuard` (`experiment_runner.py:121`), was never written for launch 6, so the process cannot have reached the clock-lock call at that point (inference from code order; the Python process may still have been importing torch). For launches 1-5 the failure happens in `run_experiment.py:103`, before `run_experiment()` is called, so no clock lock was set or reset by them. Whether a lock was left set (e.g. process killed with SIGKILL after the lock) cannot be determined, but every later run re-locks (2000/2000) at entry and thermal-gates before starting.\n")

P("### 8. Code consistency of commits\n")
P("`git_commit` in `metadata.json` is `git rev-parse HEAD` at run start (`experiment_runner.py:_git_commit_hash`); it does not capture uncommitted edits, and there is no dirty flag. Distinct commits and where they were used (runs / canonical):\n")
P("| commit | date (author) | subject | runs |\n|---|---|---|---|")
for c, n in sorted(R["distinct_commits"].items(), key=lambda kv: R["commit_info"][kv[0]]["date"]):
    ci = R["commit_info"][c]
    P(f"| `{c}` | {ci['date']} | {ci['subject']} | {n} |")
P("\nPer algorithm × environment × sweep (commit(s) of the 5-seed block):\n")
P("| algo | env | config | commit |\n|---|---|---|---|")
g = runs.groupby(["algo", "env", "config"]).git_commit.agg(lambda s: ", ".join(sorted({x[:7] for x in s}))).reset_index()
for _, r in g.iterrows():
    P(f"| {r.algo} | {r.env} | {r.config} | `{r.git_commit}` |")
P("\n**MBPO Ant-v5 `rollout_max_length=1` (commit `" + R["mbpo_ant_rollout1_commit"] + "`, all 5 seeds one commit) vs. the HalfCheetah-v5 baseline (commit `" + R["mbpo_hc_baseline_commit"] + "`):** `git diff --stat " + R["mbpo_hc_baseline_commit"] + " " + R["mbpo_ant_rollout1_commit"] + " -- algorithms experiment_runner.py run_experiment.py configs/config.py utils`:\n")
P("```\n" + R["diff_stat_hc_baseline_to_ant_rollout1_training_code"] + "\n```")
P("Commits touching these paths in between: " + "; ".join(f"`{x}`" for x in R["commits_between_hc_baseline_and_ant_rollout1_training_code"]) + ". Only additions: `algorithms/td3.py` and `algorithms/tdmpc2.py` (new algorithms), new config dataclasses/`ALGO_CONFIGS` entries in `configs/config.py` (+147 lines, no deletions), and 8 added lines in `experiment_runner.py` (algorithm dispatch). `algorithms/{sac,mbpo,dynamics_model,termination_fns,replay_buffer}.py`, `utils/` and `run_experiment.py` are unchanged between the two commits. (Full diff incl. docs/scripts/overrides: `04_results.json` → `diff_stat_full_hc_baseline_to_ant_rollout1`.)\n")
P("Training/measurement-code diff between consecutive commits used by canonical runs (paths: `algorithms/`, `experiment_runner.py`, `run_experiment.py`, `configs/config.py`, `utils/`):\n")
for k, v in R["training_code_diff_between_consecutive_distinct_commits"].items():
    P(f"- `{k}`: " + ("no change in these paths" if not v.strip() else "\n```\n" + v + "\n```"))
P("\nOnly two changes touch run-time code between commits: (i) `f23f448..92dcee0` adds the `run_start_*`/`run_end_*` temperature snapshot (`utils/thermal_gate.py` `read_current_state`, +10 lines; `experiment_runner.py` +20/−3, which also stops writing `thermal_gate_reference_file` into the metadata) — logging only, no change to training or measurement; (ii) `ecb98bb..0bbbbb7` changes one line in `configs/config.py`: the `MBPOConfig.hidden_sizes` default from (256, 256) to (1024, 1024) (`configs/config.py:94`), i.e. the MBPO baseline architecture was set in code only at the commit used by all MBPO baseline runs (all 10 MBPO baselines record `hidden_sizes` (1024, 1024) in their metadata); it also deletes tracked `.pyc` files.\n")
P("Note on SAC runs: the 8 first runs all record `f23f448`, but the 4 earliest lack `run_start_temp_c` in their metadata while the next 4 (same recorded commit) have it — so the working tree was edited (uncommitted) between run 4 and run 5; `git_commit` therefore under-describes the code for the 8 SAC baseline runs of session 0.\n")

P("## Caveats / things that look odd\n")
P("- The `run.log` grep scans quoted in CLAUDE.md (\"zero RAPL-fallback … across all 300 run.log files\") cannot detect those events by construction (item 2-3); the claim is supportable only from sweep logs (270 of 280 canonical runs) and not at all for the 10 SAC baseline runs.")
P("- No positive record of a *successful* clock lock / governor / persistence mode exists anywhere (INFO messages are never logged, no readback in metadata).")
P("- SAC baseline runs (session 0/1) have no stdout/stderr log, 4 of them lack the run-start temperature, and ran with uncommitted working-tree changes relative to the recorded commit.")
P("- The seed list first appears in git after the first canonical runs (item 5); the run order is configuration-major with fixed seed order, so config and wall-clock time (and idle level, Task 1) are confounded.")
P("- Task statement lists \"7/7 SAC, 8/8 TD3, 7 HalfCheetah and 9 Ant (MBPO), 5/5 TD-MPC2 runs per seed\": verified exactly (table in item 1).")
open(os.path.join(OUT, "04_canonical_data_quality.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("ok")
