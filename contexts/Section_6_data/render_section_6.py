"""Renders contexts/Section_6.md from the CSVs written by build_section_6.py (run from repo root)."""
import json, os
import pandas as pd

D = "contexts/Section_6_data/"
rd = lambda f: pd.read_csv(D + f)
fnd = json.load(open(D + "findings.json"))


def fmt(v):
    if isinstance(v, float):
        return "" if pd.isna(v) else repr(v)
    return str(v)


def tbl(df, cols=None, rename=None):
    df = df[cols] if cols else df
    h = [(rename or {}).get(c, c) for c in df.columns]
    out = ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in df.columns) + " |")
    return "\n".join(out)


def kv(d1, d2, a, b, cols):
    """side-by-side rows from two series-like"""
    return pd.DataFrame([dict(quantity=c, **{a: d1[c], b: d2[c]}) for c in cols])


summ = rd("C_segment_summary.csv")
S = lambda clock, seg, col: summ[(summ.clock == clock) & (summ.segment == seg)].iloc[0][col]
side = []
for seg in ["TOTAL_MEASURED_TRAINING", "rollout", "dynamics_model_update", "synthetic_rollout_generation", "gradient_updates", "buffer_sample", "critic_update", "actor_update", "target_update", "warmup", "idle_baseline_head", "idle_baseline_tail"]:
    for col in ["energy_j", "energy_kwh", "share_of_training_total", "duration_s", "mean_power_w", "total_flops", "energy_per_flop_j_per_flop"]:
        a, b = S("2000MHz", seg, col), S("200MHz", seg, col)
        if (pd.isna(a) and pd.isna(b)): continue
        side.append(dict(segment=seg, quantity=col, v_2000MHz=a, v_200MHz=b))
side = pd.DataFrame(side)

A = rd("A2_mbpo_hc_seed0_runs.csv")
diff = rd("A4_metadata_diff.csv"); dd = diff[diff.differs]
comp = rd("A6_completeness_and_log.csv")
cons = rd("C_consistency_checks.csv")
dr = rd("D_ratios_200_over_2000.csv")
di = rd("D_idle_ratios.csv")
idle = rd("C8_idle_power_two_runs.csv")
wd = rd("C7_work_done.csv")
est = rd("E_canonical_stats.csv"); can = rd("E_canonical_per_seed.csv")
dev = rd("F3_dev_runs_by_algo_env_seed.csv")
non = rd("A1_non_2000_lock_runs.csv")
ld = rd("A6_runlog_diff_timestamps_stripped.csv")
st = rd("C6_task_duration_stats.csv")

L = []
w = L.append
w("""# Section 6 — GPU clock frequency (RQ4) — context

Facts, numbers and provenance only — no interpretation, no LaTeX. Two individual runs (n = 1 each): MBPO, HalfCheetah-v5, development seed 0, GPU clock lock 200/200 MHz vs 2000/2000 MHz. **Gross** energies (idle floor included). Numbers are copied unrounded (Python `repr`), joules/kWh/W/s/MHz. `NOT FOUND` = looked for, not in the repo (places listed). `DERIVED` = plain arithmetic, formula given.

Generated 2026-10-11 from the Windows dev checkout (git HEAD `050b3c3`, branch master) after confirming `results/` is present (300 `metadata.json`). Nothing under `results/`, code or configs was modified; only `contexts/` was written. Reproduce: `.venv/Scripts/python -W ignore contexts/Section_6_data/build_section_6.py` then `.venv/Scripts/python -W ignore contexts/Section_6_data/render_section_6.py`.

Notation: **R2000** = `results/mbpo/HalfCheetah-v5/seed_0/20260904_122958`; **R200** = `results/mbpo/HalfCheetah-v5/seed_0/20260904_130915`. Companion CSVs are in `contexts/Section_6_data/` (named per item).

## 1. Run identification (item A)
""")
w(f"""**A1. Runs whose recorded lock is not 2000/2000 MHz** (`metadata.json` → `experiment_config.gpu_min_clock_mhz` / `gpu_max_clock_mhz`; `A1_non_2000_lock_runs.csv`): **{fnd['A1_count']}** runs. Exactly **one** has an explicit non-2000 value: R200 (200/200). The other 15 have `null`/`null` (SAC dev runs, seed 0/42/56 HalfCheetah-v5 and seed 56 Humanoid-v5, 2026-08-13 to 2026-09-02). `CLAUDE.md` ("exactly one run … 200/200") is correct for the explicit value. `CLAUDE.md` also says the 15 `null` runs are "functionally identical" to 2000/2000 runs; the code history says otherwise (see B1 and §6) — so the claim that exactly one run deviates from 2000 MHz is not established for those 15.

{tbl(non[['run_dir','algo','env','seed','gpu_min_clock_mhz','gpu_max_clock_mhz','lock_gpu_clocks']])}

**A2. All MBPO / HalfCheetah-v5 / seed-0 runs** (`A2_mbpo_hc_seed0_runs.csv`; exactly two exist):

{tbl(A.drop(columns=['run_dir']), rename={})}

**A3. Matched comparator.** Exactly one run matches R200: **R2000**. Same algorithm, environment, seed, git commit, all hyperparameters, software stack; the only differing config fields are the two clock fields (see A4). No other candidate exists (the only other MBPO/HalfCheetah-v5 seed-0 run is R200 itself). Rule 6 is therefore not triggered.

**A4. Field-by-field `metadata.json` diff** (full table with every field incl. identical ones: `A4_metadata_diff.csv`). Fields that differ:

{tbl(dd, ['field','v_2000MHz_run','v_200MHz_run'])}

Not differing: `git_commit` (both `ae3880860ebcce0c1e729714cdf48a08039adc46`), `torch_version` (both `2.13.0+cu130`), `codecarbon_version` (both `3.3.0`), `cuda_device_name` (both `NVIDIA GeForce RTX 5090`), `device`, all of `algo_config` (`hidden_sizes` [256, 256], `batch_size` 256, `updates_per_env_step` 1, `rollout_min/max_length` 1/1, `rollout_min/max_epoch` 20/150, `model_hidden_sizes` [200,200,200,200], …) and all other `experiment_config` fields (`lock_gpu_clocks` true, `set_persistence_mode` true, `set_cpu_performance_governor` true, thermal-gate settings, `warmup_steps` 5000, `train_steps` 100000, `steps_per_epoch` 1000, `settle_seconds` 30.0, idle windows 90.0 s). Software stack (torch, codecarbon) is **identical**. `metadata.json` records no hostname/path fields (NOT FOUND in either file; keys present: algo_name, env_id, seed, git_commit, torch_version, device, cuda_device_name, start_time_utc, experiment_config, algo_config, codecarbon_version, thermal_gate, end_time_utc). Time between runs: R2000 started 2026-09-04T10:29:59 UTC, R200 started 2026-09-04T11:09:15 UTC (same day, R200 after R2000; `start_time_utc`).

**A5. Git commits.** Identical (`ae38808…`), so no diff between commits to report. **But** `metadata.json` stores only `git rev-parse HEAD` (`experiment_runner.py:41-47`, `_git_commit_hash`), not the working-tree state, and `ae38808` (2026-09-02) predates the MBPO code: MBPO was first committed in `f109bd4` (2026-09-04 14:51:27 +0200, "Added new MBPO algo and recorded tests"), which also contains both run directories. So both runs executed uncommitted working-tree code; the recorded commit does not identify their code. Evidence on the clock settings: at `ae38808` and at `f109bd4` the committed `configs/config.py` has `gpu_min_clock_mhz`/`gpu_max_clock_mhz: Optional[int] = None`, which would be written to `metadata.json` as `null`; both runs instead record explicit values (2000 and 200). So the explicit values must have come from a working-tree edit (not from the committed defaults); how/where it was edited is NOT FOUND (no CLI flag exists for the clock values at any commit: `run_experiment.py` only has `--no-gpu-lock`; `git grep` of commit messages for clock/MHz/lock finds nothing). Files read: `git show ae38808:configs/config.py`, `git show f109bd4:configs/config.py`, `git diff ae38808 f109bd4 -- utils/gpu_control.py`.

**A6. Completeness and log** (`A6_completeness_and_log.csv`; `emissions_base_*.csv` task counts by prefix):

{tbl(comp.T.reset_index().rename(columns={'index':'item',0:'R2000 (2000 MHz)',1:'R200 (200 MHz)'}).iloc[1:])}

Both runs have all 100 epochs × {{rollout, dynamics_model_update, synthetic_rollout_generation, gradient_updates}} plus warmup, idle_baseline_head, idle_baseline_tail (403 task rows each). Both `run.log` files have 34 lines, **0** `[WARNING]`/`[ERROR]` lines, no RAPL-fallback, geolocation-fallback, thermal-gate-timeout or permission-failure strings (regexes in `build_section_6.py`). Both finish with "Run complete. Segment energy summary". **No `gpu_control` lines appear in either `run.log`** (0 matches for "Locked GPU", "Reset GPU", "governor"): see B3. Lines of `run.log` that differ between the two runs after stripping timestamps (`A6_runlog_diff_timestamps_stripped.csv`): run directory name (line 2), thermal gate result (line 5: R2000 waited 1.1682510375976562e-05 s, final_temp_c 45.0, final_power_w 26.467; R200 waited 1.0728836059570312e-05 s, 44.0, 21.006), and the final segment-energy summary lines (22–33). The "Epoch n/100 done" lines are identical in both runs except for timestamps (same real_buffer, rollout_len, cumulative_reward, mean_episode_return).

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

{tbl(side)}

Consistency (`C_consistency_checks.csv`): CSV `TOTAL_MEASURED_TRAINING` energy equals the sum of the four measured tasks and the sum of the four allocated sub-segments equals measured `gradient_updates` in both runs; CSV total duration equals the sum of the four measured task durations:

{tbl(cons)}

**C3. All keys of `segment_energy.json`** (kWh unless `_`-prefixed; `C3_segment_energy_json_all_keys.csv`) — same key set in both runs: `idle_baseline_head, warmup, dynamics_model_update, rollout, synthetic_rollout_generation, buffer_sample, critic_update, actor_update, target_update, _sub_segment_wall_time_seconds{{buffer_sample, critic_update, actor_update, target_update}}, idle_baseline_tail, _total_kg_co2eq`. Values: R2000 `_total_kg_co2eq` 0.021311583383765147, R200 0.061294793236258686; the rest are in the table above (J) and the CSV (kWh).

**C4. CPU/GPU/RAM energy split** of the measured tasks, sums over the 100 epochs (`C4_cpu_gpu_ram_split.csv`; J and % of the task's `energy_consumed`; CodeCarbon `cpu_energy`/`gpu_energy`/`ram_energy`):

{tbl(rd('C4_cpu_gpu_ram_split.csv'))}

**C5. Whole-run row of `emissions.csv`** (`C5_whole_run_row.csv`). Note: the `duration` field of this row is ≈90 s in both runs (not the whole-run wall time), while `energy_consumed` equals the sum of all per-task rows to within the difference shown (plain arithmetic: run_energy_j − sum_all_task_rows_energy_j; sum of task durations is for all 403 task rows):

{tbl(rd('C5_whole_run_row.csv').T.reset_index().rename(columns={'index':'quantity',0:'R2000',1:'R200'}).iloc[1:])}

**C6. Per-task duration statistics** (`C6_task_duration_stats.csv`; full per-task table: `per_task_rows.csv`; `measure_power_secs` = 1.0 in both `metadata.json`):

{tbl(st)}

**C7. Work done** (`C7_work_done.csv`; from `training_metrics.json` and the FLOP CSV). Per-epoch `model_train_epochs`, `synthetic_transitions_generated`, `mean_episode_return`, `model_holdout_mse` and all episode returns are identical between the two runs (`findings.json`: `training_metrics_epochs_identical` = {fnd['training_metrics_epochs_identical']}, episode returns identical = {fnd['episode_returns_identical']}). Last-10% return = mean of the last max(1, n//10) train-phase episode returns (definition of `aggregate_results.py`).

{tbl(wd.T.reset_index().rename(columns={'index':'quantity',0:'R2000',1:'R200'}).iloc[1:])}

**C8. Idle power** (`C8_idle_power_two_runs.csv`, computed with `idle_power_analysis.load_runs` restricted to these two runs; `*_w` = energy ÷ recorded duration, `*_w_corr` = energy ÷ integrated span, method of `idle_power_analysis.py`; component columns cpu/gpu/ram/total):

{tbl(idle[[c for c in idle.columns if c.startswith(('head_','tail_','delta_')) or c=='clock']].set_index('clock').T.reset_index().rename(columns={'index':'quantity'}))}

NVML temperatures recorded around the idle windows: `metadata.json` has no `run_start_temp_c`/`run_end_temp_c` for either run (keys absent — NOT FOUND). What exists is the thermal-gate result: R2000 `final_temp_c` 45.0, `final_power_w` 26.467; R200 `final_temp_c` 44.0, `final_power_w` 21.006 (both `reason: reached_reference`, `waited_seconds` ≈ 1e-5).

**C9. Idle-floor diagnostic** (`C9_idle_floor_diagnostic.csv`). Formula: idle_floor_fraction = mean(head_total_w_corr, tail_total_w_corr) × task duration ÷ gross task energy; idle_floor_w = mean of the two window-corrected total idle powers of the same run:

{tbl(rd('C9_idle_floor_diagnostic.csv'))}

**D. Ratios R200 / R2000** (`D_ratios_200_over_2000.csv`). DERIVED: ratio = value(R200) / value(R2000); difference = value(R200) − value(R2000). Power = energy ÷ duration of the same row (allocated rows: CSV definition, see C1/C2).

{tbl(dr)}

Idle power ratios R200/R2000 (`D_idle_ratios.csv`; DERIVED: power(R200)/power(R2000)):

{tbl(di)}

**E. Reference for scale** (canonical MBPO / HalfCheetah-v5, same configuration as R2000). Matching: `architecture_signature` `bs256_h256x256_ens7_mh200x200x200x200_mb256`, `updates_per_env_step` 1, `mbpo_rollout_regime` `rl1-1_re20-150`, canonical seeds {{331, 958, 14577, 43611, 85062}} (five runs; `per_run_energy_per_flop.csv`, `TOTAL_MEASURED_TRAINING` rows). Full `metadata.json` comparison of R2000 against the five canonical runs (excluding seed, timestamps, git commit and the `thermal_gate.*` result values) finds one differing field: `experiment_config.thermal_gate_reference_file` (`results/_thermal_reference.json` in R2000, absent in the canonical runs); canonical runs: commit `7e672ef12c53bf4c201241c549527a7ef8bd4bff`, torch `2.13.0+cu130`, codecarbon `3.3.0` (`E_config_diff_vs_canonical.csv`). Cross-check row in `cross_seed_energy_per_flop.csv` (`E_cross_seed_csv_row.csv`): n_seeds 5, mean_energy_joules 187913.…, mean_duration_s 1631.84…, mean_power_w 115.154… (matches the recomputed means below).

Per-seed canonical values (`E_canonical_per_seed.csv`):

{tbl(can)}

Statistics (`E_canonical_stats.csv`; sd = sample sd, ddof=1; DERIVED: comparator_minus_mean = R2000 − mean; …_over_sd = that ÷ sd; run_200_over_canonical_mean = R200 ÷ mean). Note `total_flops` of the canonical runs is not identical to R2000's (MBPO fit epochs vary per seed; R2000 `total_flops` is in C1/C2, the canonical cross-seed mean is in `E_cross_seed_csv_row.csv` column `total_flops`):

{tbl(est)}

## 4. Consistency checks (item F)

1. **Open TODO "confirm identical config and same software stack (torch 2.13.0+cu130 / codecarbon 3.3.0)"**: the exact TODO text was not found by grep in `thesis-main.tex` (searched "TODO…identical/2.13", `2.13.0+cu130`; closest: `\\todo` at line 2266 about the clock chapter and hidden-size caveat; line 3102 "Identical software stack. All runs used PyTorch 2.13.0+cu130 and CodeCarbon 3.3.0"). Fact: R2000 and R200 both record torch `2.13.0+cu130` and codecarbon `3.3.0`, identical config except the two clock fields (A4) — satisfied for the pair. Both also match the software stack of the canonical runs (E).
2. **Hidden sizes / batch size**: both runs `hidden_sizes` = [256, 256] (not the canonical 1024); `batch_size` = 256. The canonical MBPO HalfCheetah-v5 baseline (`hidden_sizes` [1024,1024]) has `batch_size` 256 (`findings.json`: canonical MBPO-HC configs (hidden, batch, utd): n — ((1024,1024),256,1): 5, ((256,256),256,1): 5, ((512,512),256,1): 5, UTD2/UTD4 and batch 512/1024 variants 5 each). So batch size equals the canonical baseline's; hidden size equals the canonical width-256 sweep configuration.
3. **Dev runs**: {fnd['F3_total_dev_runs']} of {fnd['F3_total_runs']} recorded runs have a non-canonical seed (= 300 − 280), matching the 20 in Section 3.6 (`thesis-main.tex:3003`: "20 development runs"). By (algo, env, seed) (`F3_dev_runs_by_algo_env_seed.csv`), sums to 20:

{tbl(dev)}

   The Section 3.6 text at `thesis-main.tex:3004-3006` says dev runs are not part of any total/table/figure/aggregate, with a `\\todo` at line 3006 about this Chapter 6 exception (fact only, not edited).

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

`results/mbpo/HalfCheetah-v5/seed_0/20260904_122958/{{metadata.json,segment_energy.json,training_metrics.json,emissions.csv,emissions_base_ec798f5b-….csv,run.log}}`; same set for `20260904_130915` (`emissions_base_ff2f99fb-….csv`); all 300 `results/**/metadata.json` (A1, A2, E, F3); the five canonical `results/mbpo/HalfCheetah-v5/seed_*/…/metadata.json` listed in `E_canonical_per_seed.csv`; `flop_analysis/output/per_run_energy_per_flop.csv`; `flop_analysis/output/cross_seed_energy_per_flop.csv`; `idle_power_analysis.py` (method, imported); `aggregate_results.py` (last-10% definition); `utils/gpu_control.py`; `configs/config.py`; `run_experiment.py`; `experiment_runner.py`; `README.md`; `CLAUDE.md`; `cli_commands.txt`; `results/_thermal_reference.json`; git history (`git log`, `git show ae38808:…`, `git show f109bd4:…`, `git diff ae38808 f109bd4`, `git diff f109bd4 f23f448`); `C:\\Users\\saman\\Desktop\\Thesis\\Thesis Report\\thesis-main.tex` (read-only grep: lines 2266, 3001-3006, 3102). Generators: `contexts/Section_6_data/build_section_6.py`, `render_section_6.py`; existing `contexts/section_5.4.md` read for layout only.
""")
open("contexts/Section_6.md", "w", encoding="utf-8").write("\n".join(L))
print("written", sum(len(x) for x in L))
