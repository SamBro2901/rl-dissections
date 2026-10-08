# Section 4.4 TD-MPC2 — context

Data context for thesis Section 4.4 (TD-MPC2): 50 canonical runs (5 configurations × 2 environments × 5 seeds [331, 958, 14577, 43611, 85062]). Facts, numbers and provenance only — no interpretation, no LaTeX. All energies are **gross** (idle floor included, nothing subtracted); mean ± sample sd (ddof = 1) over the five seeds; 6 significant digits; every table cell shows `HalfCheetah-v5 ‖ Ant-v5` (left ‖ right) unless a table says otherwise. Conventions of Section 4.1 apply (`measured` vs `allocated` segments; `target_update` in nJ per Polyak operation, never J/FLOP; `buffer_sample` has no FLOPs; J/FLOP = ratio of means with the sd of the five per-seed ratios (FLOPs are identical across seeds, so the mean of the per-seed ratios equals the ratio of means); relative changes computed per seed then averaged; environment comparison paired by seed, Ant relative to HC). Training total = `TOTAL_MEASURED_TRAINING` = rollout + world_model_pretrain + gradient_updates (critic = world-model step, actor = policy-prior step, target = Q-ensemble Polyak, buffer_sample); `warmup` (random actions) and the idle windows are excluded.

Checks run by the generator: 28; passed 28; failed 0. Independent recomputation vs the pipeline CSVs: 2730 values compared, 0 deviating by more than 1e-9, maximum relative deviation 2.694e-14.

**Configuration tags:** `base` = TD-MPC2 paper hyperparameters (num_q 5, horizon 3, 512 samples of which 24 policy-seeded, 6 iterations, 64 elites, batch 256, UTD 1, warmup/pretrain 5000); HC baseline = `TDMPC2Config` defaults, Ant baseline = defaults + `configs/overrides/tdmpc2_ant.json` (`episodic` = true). `numq3`/`numq7` change `num_q` (5 → 3 / 7), `hor1`/`hor5` change `horizon` (3 → 1 / 5), each one at a time around the shared baseline. Abbreviations: HC = HalfCheetah-v5, Ant = Ant-v5, GU = `gradient_updates`, TOTAL = `TOTAL_MEASURED_TRAINING`, wm_pretrain = `world_model_pretrain`, buf_sample = `buffer_sample`, critic = `critic_update` (world-model step), actor = `actor_update` (policy-prior step), target = `target_update`.

## 0. Provenance


| Item | Value |
|---|---|
| Generated (local time) | 2026-10-07T10:39:41 |
| Machine / OS | Windows 11 (10.0.26200), AMD64 — the Windows dev checkout (synced `results/`), not the Linux experiment box |
| Repo path | C:\Users\saman\Desktop\Thesis\Project Repository\rl-dissections |
| Python / pandas / numpy | 3.13.9 / 3.0.5 / 2.5.2 |
| torch / gymnasium / mujoco (local venv; used only to instantiate networks and read env dims) | 2.14.0+cpu / 1.3.0 / 3.12.0 |
| git HEAD | 4bf1aa11a8593e4531966e63bcbb00796c6e8910 (4bf1aa1 2026-10-06 18:55:08 +0200 Updated readme and context and summary csv) |
| branch | master |
| working tree (`git status --porcelain`) | clean apart from `contexts/` (this generator's own output) |

```
?? contexts/Section_4.2.md
?? contexts/Section_4.2_data/
?? contexts/Section_4.3.md
?? contexts/Section_4.3_data/
?? contexts/Section_4.4.md
?? contexts/Section_4.4_data/
```
**Scripts used** (all under `contexts/Section_4.2_data/` unless stated; all read-only on `results/`, `flop_analysis/` and git): `build_section_4_4.py (in Section_4.4_data/)`, `s4_data.py`, `s4_blocks.py`, `s4_audit.py (in Section_4.2_data/)`. Shared modules: `s4_data.py` (loading, derived quantities, independent FLOP recomputation), `s4_blocks.py` (common tables, cross-configuration quantities), `s4_audit.py` (this part). Re-run from the repo root with `.venv/Scripts/python.exe contexts/Section_4.4_data/build_section_4_4.py`. Definitions are taken from `contexts/section_4.1_data/build_section_4_1.py` (the generator of Section 4.1) and are listed in the docstring of `s4_data.py`; no definition of 4.1 was missing.

**Pipeline files used for the cross-check:**


| File | SHA-256 | mtime (local) | last git commit touching it |
|---|---|---|---|
| `flop_analysis/output/per_run_energy_per_flop.csv` | 6e7f8f250a797fe3cca22c85a22e0e93897a91f1e0a5af5d575a4067ecc6d748 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |
| `flop_analysis/output/cross_seed_energy_per_flop.csv` | 3ebcc97fa64628cd50115ad58dc54dee129769375e2276babadf25d5096e5ec7 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |
| `flop_analysis/flops_per_call.json` | 7a562e0b092b68b6fbaea089354298114e864db72c6309847db44d0565e957a7 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |

Freshness: newest `segment_energy.json` among the 50 TD-MPC2 runs: `results/tdmpc2/HalfCheetah-v5/seed_958/20260919_094502` (2026-09-23T18:04:25); oldest pipeline CSV mtime 2026-09-28T17:41:20 — **PASS** — newest segment_energy.json older than the pipeline CSVs (filesystem mtime; the content-based reconciliation is in Section 1.5)

**Code-version consistency of the 50 runs** (`metadata.json` → `git_commit`):


| git_commit | # runs | commit | configurations |
|---|---|---|---|
| `b5b7e54502` | 10 | b5b7e54 2026-09-10 Setup new  algorithm tdmpc2 and tested recording | Ant-base, HC-base |
| `a3582b52d5` | 40 | a3582b5 2026-09-18 TDMPC2 sweep scripts | Ant-hor1, Ant-hor5, Ant-numq3, Ant-numq7, HC-hor1, HC-hor5, HC-numq3, HC-numq7 |

Algorithm-side files at each run commit (git blob ids compared):


| file | status |
|---|---|
| `algorithms/tdmpc2.py` | identical at every run commit |
| `algorithms/tracker_utils.py` | identical at every run commit |

Config dataclasses at each run commit (AST source segment compared):


| definition | status |
|---|---|
| `ExperimentConfig` | identical at every run commit |
| `TDMPC2Config` | identical at every run commit |

Runner/utility files, first versus last run commit: `experiment_runner.py`: identical; `run_experiment.py`: identical; `utils/gpu_control.py`: identical; `utils/thermal_gate.py`: identical; `algorithms/tracker_utils.py`: identical.
`thermal_gate` key sets in `metadata.json` (a difference between runs that carry the same `git_commit` would show an uncommitted working tree at run time):


| thermal_gate keys | # runs | runs (if ≤ 5) |
|---|---|---|
| final_power_w, final_temp_c, gated, reason, run_end_power_w, run_end_temp_c, run_start_power_w, run_start_temp_c, waited_seconds | 50 | … |


All algorithm-side files named above are byte-identical at every run commit.

## 1. Data basis and audit

**1.1 Inventory** (one row per configuration; values from the logged `metadata.json`, identical within a configuration — checked in 1.3). `launcher` is derived from the sweep-log blocks in which the run directory appears (Section 1.4):


| config | env | description | num_q | horizon | episodic | num_samples | num_pi_trajs | iterations | num_elites | batch_size | updates_per_env_step | warmup | train_steps | steps/epoch | architecture signature | override file | launcher | # runs | seeds present | # duplicates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | HalfCheetah-v5 | paper hyperparameters: num_q 5, horizon 3, 512 samples (24 policy-seeded), 6 iterations, 64 elites, batch 256, UTD 1; Ant: episodic = true (tdmpc2_ant.json) | 5 | 3 | False | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h3_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epFalse` | `none (TDMPC2Config defaults)` | scripts/run_tdmpc2_seed_sweep.sh (log `results/tdmpc2/_seed_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| numq3 | HalfCheetah-v5 | num_q 3 | 3 | 3 | False | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h3_ns512_it6_pit24_ne64_nq3_enc2x256_mlp512_lat512_epFalse` | `configs/overrides/tdmpc2_numq3.json` | scripts/run_tdmpc2_numq_horizon_sweep.sh (log `results/tdmpc2/_numq_horizon_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| numq7 | HalfCheetah-v5 | num_q 7 | 7 | 3 | False | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h3_ns512_it6_pit24_ne64_nq7_enc2x256_mlp512_lat512_epFalse` | `configs/overrides/tdmpc2_numq7.json` | scripts/run_tdmpc2_numq_horizon_sweep.sh (log `results/tdmpc2/_numq_horizon_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| hor1 | HalfCheetah-v5 | horizon 1 | 5 | 1 | False | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h1_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epFalse` | `configs/overrides/tdmpc2_horizon1.json` | scripts/run_tdmpc2_numq_horizon_sweep.sh (log `results/tdmpc2/_numq_horizon_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| hor5 | HalfCheetah-v5 | horizon 5 | 5 | 5 | False | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h5_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epFalse` | `configs/overrides/tdmpc2_horizon5.json` | scripts/run_tdmpc2_numq_horizon_sweep.sh (log `results/tdmpc2/_numq_horizon_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| base | Ant-v5 | paper hyperparameters: num_q 5, horizon 3, 512 samples (24 policy-seeded), 6 iterations, 64 elites, batch 256, UTD 1; Ant: episodic = true (tdmpc2_ant.json) | 5 | 3 | True | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h3_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue` | `configs/overrides/tdmpc2_ant.json` | scripts/run_tdmpc2_seed_sweep.sh (log `results/tdmpc2/_seed_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| numq3 | Ant-v5 | num_q 3 | 3 | 3 | True | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h3_ns512_it6_pit24_ne64_nq3_enc2x256_mlp512_lat512_epTrue` | `configs/overrides/tdmpc2_ant_numq3.json` | scripts/run_tdmpc2_numq_horizon_sweep.sh (log `results/tdmpc2/_numq_horizon_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| numq7 | Ant-v5 | num_q 7 | 7 | 3 | True | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h3_ns512_it6_pit24_ne64_nq7_enc2x256_mlp512_lat512_epTrue` | `configs/overrides/tdmpc2_ant_numq7.json` | scripts/run_tdmpc2_numq_horizon_sweep.sh (log `results/tdmpc2/_numq_horizon_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| hor1 | Ant-v5 | horizon 1 | 5 | 1 | True | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h1_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue` | `configs/overrides/tdmpc2_ant_horizon1.json` | scripts/run_tdmpc2_numq_horizon_sweep.sh (log `results/tdmpc2/_numq_horizon_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| hor5 | Ant-v5 | horizon 5 | 5 | 5 | True | 512 | 24 | 6 | 64 | 256 | 1 | 5000 | 100000 | 1000 | `bs256_h5_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue` | `configs/overrides/tdmpc2_ant_horizon5.json` | scripts/run_tdmpc2_numq_horizon_sweep.sh (log `results/tdmpc2/_numq_horizon_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |


**1.2 Full run list** → `contexts/Section_4.4_data/tdmpc2_run_inventory.csv` (50 rows: algo, config, env, seed, run directory, timestamp, git commit, versions, clock lock, start/end time, full logged `algo_config`).

**1.3 Audit checks**

- **PASS** — layout-document expectation: 50 runs (HC: 5 configurations, Ant: 5 configurations × 5 seeds); found 50 (unmatched run directories at canonical seeds: 0)
- **PASS** — configurations per environment equal the layout-document counts (found {'HC': 5, 'Ant': 5})
- **PASS** — exactly 5 runs per configuration (all 10 configurations have 5)
- **PASS** — every run uses one of the 5 canonical seeds {331, 958, 14577, 43611, 85062}; each seed once per configuration
- **PASS** — no duplicate (algo, env, seed, config) runs
- **PASS** — every run has emissions.csv, segment_energy.json, training_metrics.json, metadata.json, run.log
- **PASS** — every run has exactly one per-task CodeCarbon log `emissions_*.csv`
- **PASS** — the logged configuration of every configuration differs from its environment's baseline in exactly the swept factor (algo_config), has no `experiment_config` difference (ignoring seed/env/algo and the metadata-only key `thermal_gate_reference_file`), and is identical across its 5 seeds
- **PASS** — `experiment_config` of the HC and Ant baselines are identical apart from seed/env_id/algo_name and the metadata-only key ({})
  - (train_steps, steps_per_epoch, warmup_steps, env) → #runs: {(100000, 1000, 5000, 'HC'): 25, (100000, 1000, 5000, 'Ant'): 25}
- **PASS** — per-task CSV has exactly the expected tasks: rollout_0..99, gradient_updates_0..99, warmup_0, idle_baseline_head, idle_baseline_tail, world_model_pretrain_0
- **PASS** — `training_metrics.json` has 100 epoch rows in every run
- **PASS** — GPU = NVIDIA GeForce RTX 5090 in every run (`cuda_device_name`)
  - per-task CSV `gpu_model`: {'1 x NVIDIA GeForce RTX 5090': 50}; `cpu_model`: {'Intel(R) Core(TM) Ultra 9 285K': 50}
- **PASS** — GPU clock lock requested at 2000/2000 MHz with lock_gpu_clocks = true in every run
  - persistence mode requested: {True: 50}; CPU `performance` governor requested: {True: 50}
- **PASS** — torch 2.13.0+cu130 and codecarbon 3.3.0 in every run ({('2.13.0+cu130', '3.3.0'): 50})
- **PASS** — thermal gate: gated = true and reason = reached_reference (no timeout) in every run
  - thermal gate: waited_seconds max 1.85966e-05 s; gate temperature 42.0000–46.0000 °C; gate power 24.9130–32.0830 W (the gate passes on its first poll because its reference is captured fresh in each process, `utils/thermal_gate.py`)
- **PASS** — `run.log` of every run contains none of the 20 failure-mode strings (list in Section 4.1, schema sample S6)
- **PASS** — sweep-log block (stdout+stderr of the launch, `2>&1 | tee -a`) contains none of the strings — available for 50 of 50 runs
- **PASS** — data-side RAPL check: per-task `cpu_power` varies across the 100 gradient_updates tasks in every run (a TDP fallback would be constant) (min distinct values per run = 100)
- **PASS** — data-side geolocation check: no run shows the Canada/Quebec fallback ({('Germany', 'baden-wurttemberg'): 50})

**1.3b Logged-configuration differences per configuration** (each row: difference of the logged `algo_config` / `experiment_config` to the baseline of the same environment; last column: difference to the dataclass defaults of `TDMPC2Config`, i.e. the override file content):


| config | algo_config vs same-env baseline | experiment_config vs same-env baseline | identical across the 5 seeds | algo_config vs `TDMPC2Config` defaults |
|---|---|---|---|---|
| HC-base | — | — | yes | — |
| HC-numq3 | num_q: 5 → 3 | — | yes | num_q: 5 → 3 |
| HC-numq7 | num_q: 5 → 7 | — | yes | num_q: 5 → 7 |
| HC-hor1 | horizon: 3 → 1 | — | yes | horizon: 3 → 1 |
| HC-hor5 | horizon: 3 → 5 | — | yes | horizon: 3 → 5 |
| Ant-base | — | — | yes | episodic: False → True |
| Ant-numq3 | num_q: 5 → 3 | — | yes | episodic: False → True; num_q: 5 → 3 |
| Ant-numq7 | num_q: 5 → 7 | — | yes | episodic: False → True; num_q: 5 → 7 |
| Ant-hor1 | horizon: 3 → 1 | — | yes | episodic: False → True; horizon: 3 → 1 |
| Ant-hor5 | horizon: 3 → 5 | — | yes | episodic: False → True; horizon: 3 → 5 |

The warning/grep caveat of Section 4.1 applies: `run.log` is written only by the `experiment_runner` logger; the `gpu_control`, `thermal_gate` and CodeCarbon loggers write to stderr, which the sweep scripts capture into the sweep log (`2>&1 | tee -a`). The clock lock / governor are therefore verified as *requested* (metadata) and as *not reported failed* (sweep-log blocks), never as positively applied; applied lock values are not recorded in `metadata.json`.

**1.4 Execution facts.** Times are `metadata.json` `start_time_utc`/`end_time_utc` (UTC). A *session* = maximal block of runs of **all 300 recorded runs** (all algorithms) with < 30 min between one run's end and the next run's start (same definition as Section 4.1). `contiguous` = the five runs of the configuration are consecutive among the TD-MPC2 runs. `launch logs` = sweep logs containing the run directory (each run's stdout+stderr block).


| config | first start (UTC) | last end (UTC) | session(s) | all 5 seeds in one session | contiguous | seed order (by start time) | launch log(s) | # runs with a log block |
|---|---|---|---|---|---|---|---|---|
| HC-base | 2026-09-10T10:28:45 | 2026-09-10T14:04:26 | 14 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_seed_sweep.log` | 5 |
| HC-numq3 | 2026-09-18T10:21:00 | 2026-09-18T13:24:10 | 19 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_numq_horizon_sweep.log` | 5 |
| HC-numq7 | 2026-09-18T16:47:01 | 2026-09-18T20:52:18 | 19 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_numq_horizon_sweep.log` | 5 |
| HC-hor1 | 2026-09-19T01:20:12 | 2026-09-19T04:00:39 | 19 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_numq_horizon_sweep.log` | 5 |
| HC-hor5 | 2026-09-19T06:51:37 | 2026-09-19T11:16:25 | 19 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_numq_horizon_sweep.log` | 5 |
| Ant-base | 2026-09-10T14:04:27 | 2026-09-10T17:53:11 | 14 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_seed_sweep.log` | 5 |
| Ant-numq3 | 2026-09-18T13:24:12 | 2026-09-18T16:47:00 | 19 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_numq_horizon_sweep.log` | 5 |
| Ant-numq7 | 2026-09-18T20:52:19 | 2026-09-19T01:20:11 | 19 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_numq_horizon_sweep.log` | 5 |
| Ant-hor1 | 2026-09-19T04:00:40 | 2026-09-19T06:51:36 | 19 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_numq_horizon_sweep.log` | 5 |
| Ant-hor5 | 2026-09-19T11:16:26 | 2026-09-19T16:10:36 | 19 | yes | yes | 331,958,14577,43611,85062 | `results/tdmpc2/_numq_horizon_sweep.log` | 5 |


Sessions containing TD-MPC2 runs:


| session | first start | last end | # runs (all algos) | algos | # TD-MPC2 | TD-MPC2 configs in order |
|---|---|---|---|---|---|---|
| 14 | 2026-09-10T09:33 | 2026-09-10T17:53 | 11 | {'tdmpc2': 11} | 10 | HC-base → Ant-base |
| 19 | 2026-09-18T10:21 | 2026-09-19T16:10 | 40 | {'tdmpc2': 40} | 40 | HC-numq3 → Ant-numq3 → HC-numq7 → Ant-numq7 → HC-hor1 → Ant-hor1 → HC-hor5 → Ant-hor5 |

Inside the sessions listed above, consecutive runs of **any** algorithm are separated by 1.27312–793.089 s (median 1.33256 s) between one run's `end_time_utc` and the next run's `start_time_utc`, by definition < 1800 s. Run length = `end_time_utc − start_time_utc` includes the thermal-gate reference poll, the 30 s settle, both idle windows and warmup; over the TD-MPC2 runs: median 44.5941 min (min 31.5078, max 60.0387). Whether the machine was powered off or idle between sessions is not recorded (each sweep script ends with `shutdown -h +1`).

**Baseline runs** (configuration `base`, 10 runs): launched by a sweep script — their stdout+stderr blocks are in `results/tdmpc2/_seed_sweep.log` (10 runs) (sum 10 of 10); not started by hand, and their launch error output was saved.
Override-file names as built by the sweep scripts (pattern lines found with `echo "configs/overrides/…"` / `overrides="configs/overrides/…"`; the algorithm / env / value are substituted at run time; the launch blocks of the logs do not record the override path): `run_ant_overnight_sweep.sh:72` `echo "configs/overrides/${ALGO_OVERRIDE_PREFIX[$algo]}${batch}.json"`; `run_ant_overnight_sweep.sh:79` `echo "configs/overrides/mbpo_ant_rollout${length}.json"`; `run_batch_size_sweep.sh:57` `echo "configs/overrides/${algo}_batch${batch}.json"`; `run_batch_size_sweep_ant.sh:74` `echo "configs/overrides/${ALGO_OVERRIDE_PREFIX[$algo]}${batch}.json"`; `run_mbpo_rollout_length_sweep.sh:42` `echo "configs/overrides/mbpo_ant_rollout${length}.json"`; `run_tdmpc2_numq_horizon_sweep.sh:58` `echo "configs/overrides/tdmpc2_ant_${param_slug}${value}.json"`; `run_tdmpc2_numq_horizon_sweep.sh:60` `echo "configs/overrides/tdmpc2_${param_slug}${value}.json"`; `run_utd_sweep.sh:147` `overrides="configs/overrides/${override_prefix}_utd${utd}.json"`; `run_width_sweep.sh:73` `echo "configs/overrides/mbpo_ant_width${width}.json"`; `run_width_sweep.sh:75` `echo "configs/overrides/${algo}_width${width}.json"`.

**Launchers.** Sweep scripts that produced these runs (read from `scripts/`): `scripts/run_tdmpc2_seed_sweep.sh` (baselines, 10 runs; Ant with `configs/overrides/tdmpc2_ant.json`); `scripts/run_tdmpc2_numq_horizon_sweep.sh` (num_q ∈ {3,7}, horizon ∈ {1,5}, both envs, 40 runs; false start documented below)
Stderr capture in each script: `run_ant_overnight_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_batch_size_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_batch_size_sweep_ant.sh`: `2>&1 | tee -a $LOG_FILE`; `run_mbpo_env_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_mbpo_rollout_length_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_td3_env_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_tdmpc2_numq_horizon_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_tdmpc2_seed_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_utd_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_width_sweep.sh`: `2>&1 | tee -a $LOG_FILE`.

Sweep bookkeeping files (every `done` entry of this algorithm should point at one of the analysed run directories):


| status file | # entries | # TD-MPC2 entries | statuses | every done entry's run_dir is one of the analysed runs |
|---|---|---|---|---|
| `results/tdmpc2/_seed_sweep_status.json` | 10 | 10 | {'done': 10} | yes |
| `results/tdmpc2/_numq_horizon_sweep_status.json` | 40 | 40 | {'done': 40} | yes |


**Launch outcomes of TD-MPC2 in every sweep log** (each `[i/N] Starting: algo=tdmpc2 …` block classified by its closing line: `Finished OK`, `FAILED (exit …)`, or no closing line):


| log | outcome | # launches |
|---|---|---|
| `results/tdmpc2/_numq_horizon_sweep.log` | FAILED (exit 1) | 5 |
| `results/tdmpc2/_numq_horizon_sweep.log` | Finished OK | 40 |
| `results/tdmpc2/_numq_horizon_sweep.log` | no closing line | 1 |
| `results/tdmpc2/_seed_sweep.log` | Finished OK | 10 |


Non-successful TD-MPC2 launches (first lines of the block):


| log | time | index | Starting line | outcome | first error line |
|---|---|---|---|---|---|
| `results/tdmpc2/_numq_horizon_sweep.log` | [2026-09-18 12:18:39] | 1/40 | [1/40] Starting: algo=tdmpc2 env=HalfCheetah-v5 seed=331 num_q=3 | FAILED (exit 1) | Traceback (most recent call last): |
| `results/tdmpc2/_numq_horizon_sweep.log` | [2026-09-18 12:18:40] | 2/40 | [2/40] Starting: algo=tdmpc2 env=HalfCheetah-v5 seed=958 num_q=3 | FAILED (exit 1) | Traceback (most recent call last): |
| `results/tdmpc2/_numq_horizon_sweep.log` | [2026-09-18 12:18:40] | 3/40 | [3/40] Starting: algo=tdmpc2 env=HalfCheetah-v5 seed=14577 num_q=3 | FAILED (exit 1) | Traceback (most recent call last): |
| `results/tdmpc2/_numq_horizon_sweep.log` | [2026-09-18 12:18:41] | 4/40 | [4/40] Starting: algo=tdmpc2 env=HalfCheetah-v5 seed=43611 num_q=3 | FAILED (exit 1) | Traceback (most recent call last): |
| `results/tdmpc2/_numq_horizon_sweep.log` | [2026-09-18 12:18:42] | 5/40 | [5/40] Starting: algo=tdmpc2 env=HalfCheetah-v5 seed=85062 num_q=3 | FAILED (exit 1) | Traceback (most recent call last): |
| `results/tdmpc2/_numq_horizon_sweep.log` | [2026-09-18 12:18:42] | 6/40 | [6/40] Starting: algo=tdmpc2 env=Ant-v5 seed=331 num_q=3 | no closing line |  |

6 launch(es) of TD-MPC2 did not finish OK; run directories created by them: none.
Successful launches with a run directory: 50 (analysed canonical runs: 50).

**1.5 Independent recomputation versus the pipeline CSVs.** Every (run, segment) and (configuration, segment) value recomputed here from `segment_energy.json`, the per-task CodeCarbon CSV, `training_metrics.json` and `flops_per_call.json` (FLOP formulas re-implemented in `s4_data.flops_for_run` / `mbpo_extras`, not imported) is compared with `per_run_energy_per_flop.csv` and `cross_seed_energy_per_flop.csv`. Columns compared: energy, duration, mean power, FLOPs, J/FLOP, call count. The pipeline's cross-seed `mean_power_w` is the mean of per-seed powers and is compared with the same statistic.

| CSV | column | # values compared | max relative deviation |
|---|---|---|---|
| cross_seed | `mean_duration_s` | 70 | 2.148e-16 |
| cross_seed | `mean_energy_joules` | 70 | 3.498e-16 |
| cross_seed | `mean_energy_per_flop_j_per_flop` | 60 | 4.250e-16 |
| cross_seed | `mean_power_w` | 70 | 8.374e-16 |
| cross_seed | `total_flops` | 60 | 0.000e+00 |
| per_run | `call_count` | 300 | 0.000e+00 |
| per_run | `duration_s` | 500 | 2.317e-16 |
| per_run | `energy_per_flop_j_per_flop` | 300 | 2.950e-16 |
| per_run | `mean_power_w` | 500 | 2.694e-14 |
| per_run | `total_energy_joules` | 500 | 2.092e-16 |
| per_run | `total_flops` | 300 | 0.000e+00 |

- **PASS** — all 2730 compared values agree within 1e-9 relative (maximum deviation over all columns 2.694e-14)

## 2. Structural facts from code and FLOP files

**2.1 Configuration objects** (dumped from `configs/config.py` / `configs/overrides/` at HEAD):

`TDMPC2Config` defaults (`dataclasses.asdict(TDMPC2Config())`; the HalfCheetah baseline uses exactly these):

```json
{
  "num_enc_layers": 2,
  "enc_dim": 256,
  "mlp_dim": 512,
  "latent_dim": 512,
  "num_q": 5,
  "dropout": 0.01,
  "simnorm_dim": 8,
  "lr": 0.0003,
  "enc_lr_scale": 0.3,
  "grad_clip_norm": 20.0,
  "batch_size": 256,
  "tau": 0.01,
  "rho": 0.5,
  "consistency_coef": 20.0,
  "reward_coef": 0.1,
  "value_coef": 0.1,
  "termination_coef": 1.0,
  "entropy_coef": 0.0001,
  "updates_per_env_step": 1,
  "discount_denom": 5.0,
  "discount_min": 0.95,
  "discount_max": 0.995,
  "episode_length": 1000,
  "horizon": 3,
  "iterations": 6,
  "num_samples": 512,
  "num_elites": 64,
  "num_pi_trajs": 24,
  "min_std": 0.05,
  "max_std": 2.0,
  "temperature": 0.5,
  "log_std_min": -10.0,
  "log_std_max": 2.0,
  "num_bins": 101,
  "vmin": -10.0,
  "vmax": 10.0,
  "episodic": false,
  "buffer_capacity": 1000000
}
```
Override files (content): `tdmpc2_ant.json` = {"episodic": true}; `tdmpc2_ant_horizon1.json` = {"episodic": true, "horizon": 1}; `tdmpc2_ant_horizon5.json` = {"episodic": true, "horizon": 5}; `tdmpc2_ant_numq3.json` = {"episodic": true, "num_q": 3}; `tdmpc2_ant_numq7.json` = {"episodic": true, "num_q": 7}; `tdmpc2_horizon1.json` = {"horizon": 1}; `tdmpc2_horizon5.json` = {"horizon": 5}; `tdmpc2_numq3.json` = {"num_q": 3}; `tdmpc2_numq7.json` = {"num_q": 7}
- **PASS** — every override file listed in the inventory equals the corresponding fields of the logged `algo_config` of its runs
`ExperimentConfig` defaults (protocol; TD-MPC2 runs use them unchanged, `warmup_steps` = 5000):

```json
{
  "algo_name": "tdmpc2",
  "env_id": "HalfCheetah-v5",
  "seed": 0,
  "settle_seconds": 30.0,
  "idle_baseline_seconds": 90.0,
  "idle_tail_seconds": 90.0,
  "warmup_steps": 5000,
  "train_steps": 100000,
  "steps_per_epoch": 1000,
  "lock_gpu_clocks": true,
  "gpu_min_clock_mhz": 2000,
  "gpu_max_clock_mhz": 2000,
  "set_persistence_mode": true,
  "set_cpu_performance_governor": true,
  "thermal_gate_enabled": true,
  "thermal_gate_temp_tolerance_c": 2.0,
  "thermal_gate_power_tolerance_w": 5.0,
  "thermal_gate_max_wait_seconds": 600.0,
  "thermal_gate_poll_interval_seconds": 5.0,
  "thermal_gate_reference_file": "results/_thermal_reference.json",
  "measure_power_secs": 1.0,
  "tracking_mode": "machine",
  "output_dir": "results",
  "country_iso_code": "DEU",
  "force_cpu_power_w": null,
  "device": "cuda"
}
```

Complete `metadata.json` of `results/tdmpc2/HalfCheetah-v5/seed_331/20260910_122845` (HC-base, seed 331):

```json
{
  "algo_name": "tdmpc2",
  "env_id": "HalfCheetah-v5",
  "seed": 331,
  "git_commit": "b5b7e54502412299ce94a5b46e9aa18c9a877308",
  "torch_version": "2.13.0+cu130",
  "device": "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "start_time_utc": "2026-09-10T10:28:45.350490",
  "experiment_config": {
    "algo_name": "tdmpc2",
    "env_id": "HalfCheetah-v5",
    "seed": 331,
    "settle_seconds": 30.0,
    "idle_baseline_seconds": 90.0,
    "idle_tail_seconds": 90.0,
    "warmup_steps": 5000,
    "train_steps": 100000,
    "steps_per_epoch": 1000,
    "lock_gpu_clocks": true,
    "gpu_min_clock_mhz": 2000,
    "gpu_max_clock_mhz": 2000,
    "set_persistence_mode": true,
    "set_cpu_performance_governor": true,
    "thermal_gate_enabled": true,
    "thermal_gate_temp_tolerance_c": 2.0,
    "thermal_gate_power_tolerance_w": 5.0,
    "thermal_gate_max_wait_seconds": 600.0,
    "thermal_gate_poll_interval_seconds": 5.0,
    "measure_power_secs": 1.0,
    "tracking_mode": "machine",
    "output_dir": "results",
    "country_iso_code": "DEU",
    "force_cpu_power_w": null,
    "device": "cuda"
  },
  "algo_config": {
    "num_enc_layers": 2,
    "enc_dim": 256,
    "mlp_dim": 512,
    "latent_dim": 512,
    "num_q": 5,
    "dropout": 0.01,
    "simnorm_dim": 8,
    "lr": 0.0003,
    "enc_lr_scale": 0.3,
    "grad_clip_norm": 20.0,
    "batch_size": 256,
    "tau": 0.01,
    "rho": 0.5,
    "consistency_coef": 20.0,
    "reward_coef": 0.1,
    "value_coef": 0.1,
    "termination_coef": 1.0,
    "entropy_coef": 0.0001,
    "updates_per_env_step": 1,
    "discount_denom": 5.0,
    "discount_min": 0.95,
    "discount_max": 0.995,
    "episode_length": 1000,
    "horizon": 3,
    "iterations": 6,
    "num_samples": 512,
    "num_elites": 64,
    "num_pi_trajs": 24,
    "min_std": 0.05,
    "max_std": 2.0,
    "temperature": 0.5,
    "log_std_min": -10.0,
    "log_std_max": 2.0,
    "num_bins": 101,
    "vmin": -10.0,
    "vmax": 10.0,
    "episodic": false,
    "buffer_capacity": 1000000
  },
  "codecarbon_version": "3.3.0",
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 1.0013580322265625e-05,
    "final_temp_c": 44.0,
    "final_power_w": 28.138,
    "run_start_temp_c": 45.0,
    "run_start_power_w": 39.719,
    "run_end_temp_c": 43.0,
    "run_end_power_w": 27.099
  },
  "end_time_utc": "2026-09-10T11:12:40.974491"
}
```

Complete `metadata.json` of `results/tdmpc2/Ant-v5/seed_331/20260910_160427` (Ant-base, seed 331):

```json
{
  "algo_name": "tdmpc2",
  "env_id": "Ant-v5",
  "seed": 331,
  "git_commit": "b5b7e54502412299ce94a5b46e9aa18c9a877308",
  "torch_version": "2.13.0+cu130",
  "device": "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "start_time_utc": "2026-09-10T14:04:27.839237",
  "experiment_config": {
    "algo_name": "tdmpc2",
    "env_id": "Ant-v5",
    "seed": 331,
    "settle_seconds": 30.0,
    "idle_baseline_seconds": 90.0,
    "idle_tail_seconds": 90.0,
    "warmup_steps": 5000,
    "train_steps": 100000,
    "steps_per_epoch": 1000,
    "lock_gpu_clocks": true,
    "gpu_min_clock_mhz": 2000,
    "gpu_max_clock_mhz": 2000,
    "set_persistence_mode": true,
    "set_cpu_performance_governor": true,
    "thermal_gate_enabled": true,
    "thermal_gate_temp_tolerance_c": 2.0,
    "thermal_gate_power_tolerance_w": 5.0,
    "thermal_gate_max_wait_seconds": 600.0,
    "thermal_gate_poll_interval_seconds": 5.0,
    "measure_power_secs": 1.0,
    "tracking_mode": "machine",
    "output_dir": "results",
    "country_iso_code": "DEU",
    "force_cpu_power_w": null,
    "device": "cuda"
  },
  "algo_config": {
    "num_enc_layers": 2,
    "enc_dim": 256,
    "mlp_dim": 512,
    "latent_dim": 512,
    "num_q": 5,
    "dropout": 0.01,
    "simnorm_dim": 8,
    "lr": 0.0003,
    "enc_lr_scale": 0.3,
    "grad_clip_norm": 20.0,
    "batch_size": 256,
    "tau": 0.01,
    "rho": 0.5,
    "consistency_coef": 20.0,
    "reward_coef": 0.1,
    "value_coef": 0.1,
    "termination_coef": 1.0,
    "entropy_coef": 0.0001,
    "updates_per_env_step": 1,
    "discount_denom": 5.0,
    "discount_min": 0.95,
    "discount_max": 0.995,
    "episode_length": 1000,
    "horizon": 3,
    "iterations": 6,
    "num_samples": 512,
    "num_elites": 64,
    "num_pi_trajs": 24,
    "min_std": 0.05,
    "max_std": 2.0,
    "temperature": 0.5,
    "log_std_min": -10.0,
    "log_std_max": 2.0,
    "num_bins": 101,
    "vmin": -10.0,
    "vmax": 10.0,
    "episodic": true,
    "buffer_capacity": 1000000
  },
  "codecarbon_version": "3.3.0",
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 1.0967254638671875e-05,
    "final_temp_c": 44.0,
    "final_power_w": 29.618,
    "run_start_temp_c": 43.0,
    "run_start_power_w": 26.437,
    "run_end_temp_c": 44.0,
    "run_end_power_w": 30.479
  },
  "end_time_utc": "2026-09-10T14:50:23.686554"
}
```

**2.2 Verbatim code** — `algorithms/tdmpc2.py` (complete, HEAD; identical at every run commit, Section 0):

```python
   1  """
   2  TD-MPC2 (Hansen, Su & Wang, ICLR 2024, "TD-MPC2: Scalable, Robust World
   3  Models for Continuous Control", https://arxiv.org/abs/2310.16828).
   4  
   5  TD-MPC2 learns a decoder-free latent world model (encoder + latent dynamics
   6  + reward model + Q-ensemble + a Gaussian policy prior used to seed planning)
   7  and selects actions via MPPI trajectory optimization in latent space rather
   8  than by directly querying the policy. This module is a single-task,
   9  state-observation port of the authors' reference implementation
  10  (https://github.com/nicklashansen/tdmpc2), adapted to fit this harness's
  11  env/config/TrackerTask conventions (see algorithms/sac.py, algorithms/td3.py)
  12  -- the numerics of every loss, the MPPI planner, and the default
  13  hyperparameters (configs/config.py TDMPC2Config) match the reference
  14  `tdmpc2/tdmpc2.py`, `common/world_model.py`, `common/math.py`,
  15  `common/layers.py` and `config.yaml` as of the ICLR 2024 release.
  16  
  17  Key building blocks, ported from the reference source:
  18    - SimNorm (simplicial normalization, https://arxiv.org/abs/2204.00616):
  19      stabilizes the latent space by softmax-normalizing groups of `simnorm_dim`
  20      units, used as the output activation of the encoder and the dynamics net.
  21    - Discrete regression for reward/value (`two_hot`/`two_hot_inv`/`soft_ce`):
  22      rewards and Q-values are predicted as a soft two-hot distribution over
  23      `num_bins` bins in a symlog-transformed space and trained with
  24      cross-entropy, rather than direct MSE regression.
  25    - The "consistency loss": only the FIRST latent state in a training
  26      sequence is obtained by encoding a real observation; every subsequent
  27      latent is produced by unrolling the *learned* dynamics model, and is
  28      pulled (MSE, stop-gradient on the target) toward the encoding of the
  29      corresponding real next-observation. This is what makes the latent
  30      dynamics model usable for multi-step planning.
  31    - MPPI planning (`TDMPC2Agent.act` / `_plan`): at every environment step,
  32      `num_samples` action sequences (a fraction seeded by the policy prior,
  33      the rest by the current Gaussian search distribution) are rolled out
  34      through the *imagined* latent dynamics, scored by reward-model rollout +
  35      terminal Q-value, and the top `num_elites` are used to re-fit the search
  36      distribution's mean/std (`iterations` times) before one action is sampled
  37      from it. Unlike SAC/TD3/MBPO in this repo, action *selection* itself is
  38      the dominant compute cost here, not the gradient updates -- worth calling
  39      out explicitly when comparing energy profiles across algorithms.
  40  
  41  Deliberate adaptations to this harness (documented so results aren't
  42  mis-read against the paper):
  43    - The reference's `OnlineTrainer` interleaves one env step with one
  44      gradient update strictly in lockstep, and does a one-off burst of
  45      `seed_steps` gradient updates the instant warmup ends ("pretraining on
  46      seed data"). This module keeps that pretraining burst (as its own
  47      directly-measured segment, "world_model_pretrain") but -- like every
  48      other algorithm in this repo -- separates each epoch's rollout from its
  49      `steps_per_epoch * updates_per_env_step` gradient updates so the two can
  50      be tagged as independent CodeCarbon tasks (see algorithms/sac.py's
  51      docstring and README.md's "Why epoch-level tagging" section).
  52    - No multi-task machinery (task embeddings/masks), no pixel-observation
  53      encoder, no `torch.compile`/vmap ensemble tricks: the Q-ensemble is a
  54      plain `nn.ModuleList` of independent MLPs evaluated in a Python loop.
  55      Mathematically identical to the reference's vmapped ensemble; just not
  56      fused into one kernel. Fine at the batch sizes used here.
  57    - Q-parameter freezing during the policy-loss backward pass uses this
  58      repo's existing `requires_grad = False` / restore pattern (see
  59      algorithms/td3.py) instead of the reference's separate
  60      `TensorDictParams`-based "detached" parameter view -- same effect
  61      (policy gradients flow through z/action into the policy net only, never
  62      into the Q-network weights), simpler to read.
  63  """
  64  from __future__ import annotations
  65  
  66  import collections
  67  import time
  68  from dataclasses import dataclass
  69  from typing import Dict, List, Optional
  70  
  71  import numpy as np
  72  import torch
  73  import torch.nn as nn
  74  import torch.nn.functional as F
  75  
  76  from algorithms.tracker_utils import TrackerTask
  77  from configs.config import TDMPC2Config, ExperimentConfig
  78  
  79  LOG_PROB_CONST = 0.9189385175704956  # 0.5 * log(2*pi)
  80  
  81  
  82  # ------------------------------------------------------------------------
  83  # Math utilities (ported from tdmpc2/common/math.py)
  84  # ------------------------------------------------------------------------
  85  
  86  def symlog(x: torch.Tensor) -> torch.Tensor:
  87      return torch.sign(x) * torch.log1p(torch.abs(x))
  88  
  89  
  90  def symexp(x: torch.Tensor) -> torch.Tensor:
  91      return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
  92  
  93  
  94  def two_hot(x: torch.Tensor, vmin: float, vmax: float, num_bins: int, bin_size: float) -> torch.Tensor:
  95      """Converts a batch of scalars (shape (B, 1)) to soft two-hot encoded
  96      targets (shape (B, num_bins)) for discrete regression, in symlog space."""
  97      x = torch.clamp(symlog(x), vmin, vmax).squeeze(-1)
  98      bin_idx = torch.floor((x - vmin) / bin_size)
  99      bin_offset = ((x - vmin) / bin_size - bin_idx).unsqueeze(-1)
 100      soft_two_hot = torch.zeros(x.shape[0], num_bins, device=x.device, dtype=x.dtype)
 101      bin_idx = bin_idx.long()
 102      soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
 103      soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset)
 104      return soft_two_hot
 105  
 106  
 107  def two_hot_inv(x: torch.Tensor, vmin: float, vmax: float, num_bins: int) -> torch.Tensor:
 108      """Converts (soft) two-hot / discrete-regression logits back to a scalar
 109      expectation, mapped back out of symlog space."""
 110      bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
 111      probs = F.softmax(x, dim=-1)
 112      out = torch.sum(probs * bins, dim=-1, keepdim=True)
 113      return symexp(out)
 114  
 115  
 116  def soft_ce(pred: torch.Tensor, target: torch.Tensor, vmin: float, vmax: float, num_bins: int, bin_size: float) -> torch.Tensor:
 117      """Cross-entropy between predicted bin logits and a soft two-hot target."""
 118      logp = F.log_softmax(pred, dim=-1)
 119      target_soft = two_hot(target, vmin, vmax, num_bins, bin_size)
 120      return -(target_soft * logp).sum(-1, keepdim=True)
 121  
 122  
 123  def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
 124      residual = -0.5 * eps.pow(2) - log_std
 125      return (residual - LOG_PROB_CONST).sum(-1, keepdim=True)
 126  
 127  
 128  def squash(mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor):
 129      """tanh-squashes the mean and sampled action, correcting the log-prob."""
 130      mu = torch.tanh(mu)
 131      pi_t = torch.tanh(pi)
 132      squashed = torch.log(F.relu(1 - pi_t.pow(2)) + 1e-6)
 133      log_pi = log_pi - squashed.sum(-1, keepdim=True)
 134      return mu, pi_t, log_pi
 135  
 136  
 137  def gumbel_softmax_sample(p: torch.Tensor, temperature: float = 1.0, dim: int = 0) -> torch.Tensor:
 138      logits = p.log()
 139      gumbels = -torch.empty_like(logits).exponential_().log()
 140      gumbels = (logits + gumbels) / temperature
 141      y_soft = gumbels.softmax(dim)
 142      return y_soft.argmax(-1).reshape(-1)
 143  
 144  
 145  class RunningScale:
 146      """Running trimmed (5th-95th percentile) scale estimator used to
 147      normalize Q-values before combining them with the entropy bonus in the
 148      policy loss (tdmpc2/common/scale.py)."""
 149  
 150      def __init__(self, tau: float, device: torch.device):
 151          self.tau = tau
 152          self.value = torch.ones(1, device=device)
 153  
 154      def _percentile(self, x: torch.Tensor) -> torch.Tensor:
 155          x_sorted, _ = torch.sort(x, dim=0)
 156          n = x_sorted.shape[0]
 157          percentiles = torch.tensor([5.0, 95.0], device=x.device)
 158          positions = percentiles * (n - 1) / 100
 159          floored = torch.floor(positions)
 160          ceiled = torch.clamp(floored + 1, max=n - 1)
 161          w_ceiled = (positions - floored).unsqueeze(-1)
 162          w_floored = 1.0 - w_ceiled
 163          d0 = x_sorted[floored.long()] * w_floored
 164          d1 = x_sorted[ceiled.long()] * w_ceiled
 165          return d0 + d1
 166  
 167      def update(self, x: torch.Tensor):
 168          p = self._percentile(x.detach())
 169          value = torch.clamp(p[1] - p[0], min=1.0)
 170          self.value = self.value * (1 - self.tau) + value * self.tau
 171  
 172      def __call__(self, x: torch.Tensor) -> torch.Tensor:
 173          return x / self.value
 174  
 175  
 176  # ------------------------------------------------------------------------
 177  # Network building blocks (ported from tdmpc2/common/layers.py)
 178  # ------------------------------------------------------------------------
 179  
 180  class SimNorm(nn.Module):
 181      """Simplicial normalization (https://arxiv.org/abs/2204.00616): splits
 182      the last dimension into groups of `dim` units and softmaxes each group."""
 183  
 184      def __init__(self, dim: int):
 185          super().__init__()
 186          self.dim = dim
 187  
 188      def forward(self, x: torch.Tensor) -> torch.Tensor:
 189          shp = x.shape
 190          x = x.view(*shp[:-1], -1, self.dim)
 191          x = F.softmax(x, dim=-1)
 192          return x.view(*shp)
 193  
 194  
 195  class NormedLinear(nn.Linear):
 196      """Linear layer + LayerNorm + activation (Mish by default) + optional dropout."""
 197  
 198      def __init__(self, in_features, out_features, dropout: float = 0.0, act: Optional[nn.Module] = None):
 199          super().__init__(in_features, out_features)
 200          self.ln = nn.LayerNorm(out_features)
 201          self.act = act if act is not None else nn.Mish(inplace=False)
 202          self.drop = nn.Dropout(dropout) if dropout > 0 else None
 203  
 204      def forward(self, x: torch.Tensor) -> torch.Tensor:
 205          x = super().forward(x)
 206          if self.drop is not None:
 207              x = self.drop(x)
 208          return self.act(self.ln(x))
 209  
 210  
 211  def mlp(in_dim: int, mlp_dims: List[int], out_dim: int, act: Optional[nn.Module] = None, dropout: float = 0.0) -> nn.Sequential:
 212      """TD-MPC2's basic MLP block: NormedLinear hidden layers (dropout only on
 213      the first), followed by either a plain nn.Linear output (act=None, used
 214      for reward/Q/policy-prior logits) or a NormedLinear output with a custom
 215      activation (used for the encoder/dynamics, act=SimNorm)."""
 216      dims = [in_dim] + list(mlp_dims) + [out_dim]
 217      layers = []
 218      for i in range(len(dims) - 2):
 219          layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout if i == 0 else 0.0))
 220      if act is not None:
 221          layers.append(NormedLinear(dims[-2], dims[-1], act=act))
 222      else:
 223          layers.append(nn.Linear(dims[-2], dims[-1]))
 224      return nn.Sequential(*layers)
 225  
 226  
 227  def _weight_init(m: nn.Module):
 228      if isinstance(m, nn.Linear):
 229          nn.init.trunc_normal_(m.weight, std=0.02)
 230          if m.bias is not None:
 231              nn.init.constant_(m.bias, 0)
 232  
 233  
 234  # ------------------------------------------------------------------------
 235  # World model
 236  # ------------------------------------------------------------------------
 237  
 238  class WorldModel(nn.Module):
 239      """Implicit (decoder-free) latent world model: encoder h, latent dynamics
 240      d, reward R, terminal-value Q-ensemble, and Gaussian policy prior p
 241      (tdmpc2/common/world_model.py, single-task/state-obs subset)."""
 242  
 243      def __init__(self, obs_dim: int, act_dim: int, cfg: TDMPC2Config):
 244          super().__init__()
 245          self.cfg = cfg
 246          self.act_dim = act_dim
 247  
 248          enc_hidden = max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim]
 249          self.encoder = mlp(obs_dim, enc_hidden, cfg.latent_dim, act=SimNorm(cfg.simnorm_dim))
 250          self.dynamics = mlp(cfg.latent_dim + act_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg.simnorm_dim))
 251          self.reward_net = mlp(cfg.latent_dim + act_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.num_bins)
 252          self.termination_net = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 1) if cfg.episodic else None
 253          self.pi_net = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 2 * act_dim)
 254          self.qs = nn.ModuleList([
 255              mlp(cfg.latent_dim + act_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.num_bins, dropout=cfg.dropout)
 256              for _ in range(cfg.num_q)
 257          ])
 258  
 259          self.apply(_weight_init)
 260          nn.init.zeros_(self.reward_net[-1].weight)
 261          for net in self.qs:
 262              nn.init.zeros_(net[-1].weight)
 263  
 264          import copy
 265          self.qs_target = copy.deepcopy(self.qs)
 266          for p in self.qs_target.parameters():
 267              p.requires_grad = False
 268  
 269          self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
 270          self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max - cfg.log_std_min))
 271  
 272      def encode(self, obs: torch.Tensor) -> torch.Tensor:
 273          return self.encoder(obs)
 274  
 275      def next_latent(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
 276          return self.dynamics(torch.cat([z, a], dim=-1))
 277  
 278      def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
 279          return self.reward_net(torch.cat([z, a], dim=-1))
 280  
 281      def termination(self, z: torch.Tensor, unnormalized: bool = False) -> torch.Tensor:
 282          out = self.termination_net(z)
 283          return out if unnormalized else torch.sigmoid(out)
 284  
 285      def pi(self, z: torch.Tensor):
 286          mean, log_std = self.pi_net(z).chunk(2, dim=-1)
 287          log_std = self.log_std_min + 0.5 * self.log_std_dif * (torch.tanh(log_std) + 1)
 288          eps = torch.randn_like(mean)
 289          log_prob = gaussian_logprob(eps, log_std)
 290  
 291          action = mean + eps * log_std.exp()
 292          mean, action, log_prob = squash(mean, action, log_prob)
 293  
 294          size = eps.shape[-1]
 295          scaled_log_prob = log_prob * size
 296          entropy_scale = scaled_log_prob / (log_prob + 1e-8)
 297          info = {
 298              "mean": mean,
 299              "entropy": -log_prob,
 300              "scaled_entropy": -log_prob * entropy_scale,
 301          }
 302          return action, info
 303  
 304      def Q(self, z: torch.Tensor, a: torch.Tensor, return_type: str = "min", target: bool = False) -> torch.Tensor:
 305          assert return_type in {"min", "avg", "all"}
 306          x = torch.cat([z, a], dim=-1)
 307          nets = self.qs_target if target else self.qs
 308          out = torch.stack([net(x) for net in nets], dim=0)  # (num_q, ..., num_bins)
 309          if return_type == "all":
 310              return out
 311          idx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
 312          qvals = two_hot_inv(out[idx], self.cfg.vmin, self.cfg.vmax, self.cfg.num_bins)  # (2, ..., 1)
 313          if return_type == "min":
 314              return qvals.min(0).values
 315          return qvals.sum(0) / 2.0
 316  
 317  
 318  # ------------------------------------------------------------------------
 319  # Sequence replay buffer (episodes -> random length-`horizon` subsequences)
 320  # ------------------------------------------------------------------------
 321  
 322  class _EpisodeSequenceBuffer:
 323      """Stores whole episodes and samples random length-`horizon` (+1 obs)
 324      contiguous subsequences, uniformly over *transitions* (an episode of
 325      length L contributes L - horizon + 1 valid start positions, so it is
 326      sampled with weight proportional to that count) -- the same effective
 327      distribution as the reference's SliceSampler-based torchrl buffer
 328      (tdmpc2/common/buffer.py), implemented with plain numpy since this
 329      harness doesn't depend on torchrl.
 330  
 331      Capacity is enforced in whole episodes (oldest evicted first, at least
 332      one episode always kept) once the total transition count exceeds it --
 333      an episode-granular approximation of the reference's flat transition
 334      circular buffer."""
 335  
 336      def __init__(self, capacity: int, obs_dim: int, act_dim: int, horizon: int):
 337          self.capacity = capacity
 338          self.obs_dim = obs_dim
 339          self.act_dim = act_dim
 340          self.horizon = horizon
 341          self._episodes: "collections.deque[dict]" = collections.deque()
 342          self._total_transitions = 0
 343          self._reset_open()
 344  
 345      def _reset_open(self):
 346          self._open_obs = []
 347          self._open_actions = []
 348          self._open_rewards = []
 349          self._open_terminateds = []
 350  
 351      def start_episode(self, obs0: np.ndarray):
 352          self._reset_open()
 353          self._open_obs.append(np.asarray(obs0, dtype=np.float32))
 354  
 355      def add(self, action, reward, next_obs, terminated):
 356          self._open_actions.append(np.asarray(action, dtype=np.float32))
 357          self._open_rewards.append(np.float32(reward))
 358          self._open_terminateds.append(np.float32(terminated))
 359          self._open_obs.append(np.asarray(next_obs, dtype=np.float32))
 360  
 361      def end_episode(self):
 362          length = len(self._open_actions)
 363          if length == 0:
 364              self._reset_open()
 365              return
 366          episode = {
 367              "obs": np.stack(self._open_obs).astype(np.float32),
 368              "actions": np.stack(self._open_actions).astype(np.float32),
 369              "rewards": np.asarray(self._open_rewards, dtype=np.float32).reshape(-1, 1),
 370              "terminateds": np.asarray(self._open_terminateds, dtype=np.float32).reshape(-1, 1),
 371              "length": length,
 372          }
 373          self._episodes.append(episode)
 374          self._total_transitions += length
 375          while self._total_transitions > self.capacity and len(self._episodes) > 1:
 376              old = self._episodes.popleft()
 377              self._total_transitions -= old["length"]
 378          self._reset_open()
 379  
 380      def __len__(self):
 381          return self._total_transitions
 382  
 383      def sample(self, batch_size: int, horizon: int):
 384          eligible = [ep for ep in self._episodes if ep["length"] >= horizon]
 385          if not eligible:
 386              return None
 387          weights = np.array([ep["length"] - horizon + 1 for ep in eligible], dtype=np.float64)
 388          probs = weights / weights.sum()
 389          choices = np.random.choice(len(eligible), size=batch_size, p=probs)
 390  
 391          obs = np.empty((horizon + 1, batch_size, self.obs_dim), dtype=np.float32)
 392          actions = np.empty((horizon, batch_size, self.act_dim), dtype=np.float32)
 393          rewards = np.empty((horizon, batch_size, 1), dtype=np.float32)
 394          terminateds = np.empty((horizon, batch_size, 1), dtype=np.float32)
 395          for b, ci in enumerate(choices):
 396              ep = eligible[ci]
 397              start = np.random.randint(0, ep["length"] - horizon + 1)
 398              obs[:, b] = ep["obs"][start:start + horizon + 1]
 399              actions[:, b] = ep["actions"][start:start + horizon]
 400              rewards[:, b] = ep["rewards"][start:start + horizon]
 401              terminateds[:, b] = ep["terminateds"][start:start + horizon]
 402          return obs, actions, rewards, terminateds
 403  
 404  
 405  # ------------------------------------------------------------------------
 406  # Agent
 407  # ------------------------------------------------------------------------
 408  
 409  @dataclass
 410  class UpdateInfo:
 411      """Wall-clock time (seconds) spent in each sub-segment of one call to
 412      update(), matching the buffer_sample/critic_update/actor_update/
 413      target_update naming used by sac.py/td3.py/mbpo.py -- here "critic_update"
 414      is the world-model step (consistency + reward + value losses) and
 415      "actor_update" is the policy-prior step, since those are TD-MPC2's
 416      closest analogues to SAC/TD3's critic and actor updates."""
 417      buffer_sample_s: float = 0.0
 418      critic_update_s: float = 0.0
 419      actor_update_s: float = 0.0
 420      target_update_s: float = 0.0
 421      consistency_loss: Optional[float] = None
 422      reward_loss: Optional[float] = None
 423      value_loss: Optional[float] = None
 424      pi_loss: Optional[float] = None
 425      total_loss: Optional[float] = None
 426  
 427  
 428  class TDMPC2Agent:
 429      def __init__(self, obs_dim: int, act_dim: int, act_limit: float, cfg: TDMPC2Config, device: torch.device, episode_length: int):
 430          self.cfg = cfg
 431          self.device = device
 432          self.act_dim = act_dim
 433          self.act_limit = act_limit
 434          self.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
 435  
 436          self.model = WorldModel(obs_dim, act_dim, cfg).to(device)
 437  
 438          world_model_params = [
 439              {"params": self.model.encoder.parameters(), "lr": cfg.lr * cfg.enc_lr_scale},
 440              {"params": self.model.dynamics.parameters()},
 441              {"params": self.model.reward_net.parameters()},
 442              {"params": self.model.qs.parameters()},
 443          ]
 444          if cfg.episodic:
 445              world_model_params.append({"params": self.model.termination_net.parameters()})
 446          self.optim = torch.optim.Adam(world_model_params, lr=cfg.lr)
 447          self.pi_optim = torch.optim.Adam(self.model.pi_net.parameters(), lr=cfg.lr, eps=1e-5)
 448  
 449          self._world_model_modules = [self.model.encoder, self.model.dynamics, self.model.reward_net, self.model.qs]
 450          if cfg.episodic:
 451              self._world_model_modules.append(self.model.termination_net)
 452  
 453          self.scale = RunningScale(cfg.tau, device)
 454          self.iterations = cfg.iterations + 2 * int(act_dim >= 20)  # heuristic for large action spaces (paper)
 455          self.discount = self._get_discount(episode_length)
 456          self._prev_mean = torch.zeros(cfg.horizon, act_dim, device=device)
 457  
 458      def _get_discount(self, episode_length: int) -> float:
 459          frac = episode_length / self.cfg.discount_denom
 460          return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)
 461  
 462      def _world_model_parameters(self):
 463          for m in self._world_model_modules:
 464              yield from m.parameters()
 465  
 466      @torch.no_grad()
 467      def _estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
 468          """Rolls a batch of imagined action sequences forward through the
 469          latent dynamics/reward models and returns discounted return + terminal
 470          Q-value (paper Eq. for MPPI's trajectory objective)."""
 471          n = actions.shape[1]
 472          G = torch.zeros(n, 1, device=self.device)
 473          discount = 1.0
 474          termination = torch.zeros(n, 1, device=self.device)
 475          for t in range(self.cfg.horizon):
 476              reward = two_hot_inv(self.model.reward(z, actions[t]), self.cfg.vmin, self.cfg.vmax, self.cfg.num_bins)
 477              z = self.model.next_latent(z, actions[t])
 478              G = G + discount * (1 - termination) * reward
 479              discount = discount * self.discount
 480              if self.cfg.episodic:
 481                  termination = torch.clip(termination + (self.model.termination(z) > 0.5).float(), max=1.0)
 482          action, _ = self.model.pi(z)
 483          return G + discount * (1 - termination) * self.model.Q(z, action, return_type="avg")
 484  
 485      @torch.no_grad()
 486      def act(self, obs: np.ndarray, t0: bool = False, eval_mode: bool = False) -> np.ndarray:
 487          """Selects an action by MPPI planning in latent space (tdmpc2's `_plan`)."""
 488          cfg = self.cfg
 489          obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
 490          z = self.model.encode(obs_t)
 491  
 492          pi_actions = None
 493          if cfg.num_pi_trajs > 0:
 494              pi_actions = torch.empty(cfg.horizon, cfg.num_pi_trajs, self.act_dim, device=self.device)
 495              _z = z.repeat(cfg.num_pi_trajs, 1)
 496              for t in range(cfg.horizon - 1):
 497                  pi_actions[t], _ = self.model.pi(_z)
 498                  _z = self.model.next_latent(_z, pi_actions[t])
 499              pi_actions[-1], _ = self.model.pi(_z)
 500  
 501          z = z.repeat(cfg.num_samples, 1)
 502          mean = torch.zeros(cfg.horizon, self.act_dim, device=self.device)
 503          std = torch.full((cfg.horizon, self.act_dim), cfg.max_std, device=self.device)
 504          if not t0:
 505              mean[:-1] = self._prev_mean[1:]
 506  
 507          actions = torch.empty(cfg.horizon, cfg.num_samples, self.act_dim, device=self.device)
 508          if pi_actions is not None:
 509              actions[:, :cfg.num_pi_trajs] = pi_actions
 510  
 511          score = None
 512          elite_actions = None
 513          for _ in range(self.iterations):
 514              r = torch.randn(cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, self.act_dim, device=self.device)
 515              actions_sample = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
 516              actions[:, cfg.num_pi_trajs:] = actions_sample
 517  
 518              value = self._estimate_value(z, actions).nan_to_num(0)
 519              elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
 520              elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
 521  
 522              max_value = elite_value.max(0).values
 523              score = torch.exp(cfg.temperature * (elite_value - max_value))
 524              score = score / score.sum(0)
 525              mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
 526              std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
 527              std = std.clamp(cfg.min_std, cfg.max_std)
 528  
 529          rand_idx = gumbel_softmax_sample(score.squeeze(1))
 530          chosen = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
 531          a, std_final = chosen[0], std[0]
 532          if not eval_mode:
 533              a = a + std_final * torch.randn(self.act_dim, device=self.device)
 534          self._prev_mean = mean
 535          a = a.clamp(-1, 1)
 536          return (a * self.act_limit).cpu().numpy()
 537  
 538      def _update_pi(self, zs_detached: torch.Tensor) -> Dict[str, float]:
 539          cfg = self.cfg
 540          for p in self.model.qs.parameters():
 541              p.requires_grad = False
 542  
 543          action, info = self.model.pi(zs_detached)
 544          q_pi = self.model.Q(zs_detached, action, return_type="avg")  # (H+1, B, 1)
 545          self.scale.update(q_pi[0])
 546          q_pi_scaled = self.scale(q_pi)
 547  
 548          rho_weights = torch.pow(cfg.rho, torch.arange(zs_detached.shape[0], device=self.device))
 549          per_step = -(cfg.entropy_coef * info["scaled_entropy"] + q_pi_scaled).mean(dim=1).squeeze(-1)
 550          pi_loss = (per_step * rho_weights).mean()
 551  
 552          self.pi_optim.zero_grad(set_to_none=True)
 553          pi_loss.backward()
 554          torch.nn.utils.clip_grad_norm_(self.model.pi_net.parameters(), cfg.grad_clip_norm)
 555          self.pi_optim.step()
 556  
 557          for p in self.model.qs.parameters():
 558              p.requires_grad = True
 559          return {"pi_loss": pi_loss.item()}
 560  
 561      def update(self, buffer: _EpisodeSequenceBuffer, batch_size: int) -> UpdateInfo:
 562          info = UpdateInfo()
 563          cfg = self.cfg
 564  
 565          t0 = time.perf_counter()
 566          batch = buffer.sample(batch_size, cfg.horizon)
 567          if batch is None:
 568              return info
 569          obs_np, action_np, reward_np, terminated_np = batch
 570          obs = torch.as_tensor(obs_np, device=self.device)
 571          action = torch.as_tensor(action_np, device=self.device)
 572          reward = torch.as_tensor(reward_np, device=self.device)
 573          terminated = torch.as_tensor(terminated_np, device=self.device)
 574          info.buffer_sample_s = time.perf_counter() - t0
 575  
 576          # --- critic_update: world model (consistency + reward + value losses) ---
 577          t0 = time.perf_counter()
 578          self.model.train()
 579          with torch.no_grad():
 580              next_z = self.model.encode(obs[1:])  # (H, B, latent_dim), real next-obs encodings (loss targets)
 581              next_action, _ = self.model.pi(next_z)
 582              target_q = self.model.Q(next_z, next_action, return_type="min", target=True)  # (H, B, 1)
 583              td_targets = reward + self.discount * (1 - terminated) * target_q
 584  
 585          zs = torch.empty(cfg.horizon + 1, batch_size, cfg.latent_dim, device=self.device)
 586          z = self.model.encode(obs[0])
 587          zs[0] = z
 588          consistency_loss = torch.zeros((), device=self.device)
 589          for t in range(cfg.horizon):
 590              z = self.model.next_latent(z, action[t])
 591              consistency_loss = consistency_loss + F.mse_loss(z, next_z[t]) * (cfg.rho ** t)
 592              zs[t + 1] = z
 593          consistency_loss = consistency_loss / cfg.horizon
 594  
 595          _zs = zs[:-1]
 596          qs = self.model.Q(_zs, action, return_type="all")  # (num_q, H, B, num_bins)
 597          reward_preds = self.model.reward(_zs, action)  # (H, B, num_bins)
 598  
 599          reward_loss = torch.zeros((), device=self.device)
 600          value_loss = torch.zeros((), device=self.device)
 601          for t in range(cfg.horizon):
 602              reward_loss = reward_loss + soft_ce(
 603                  reward_preds[t], reward[t], cfg.vmin, cfg.vmax, cfg.num_bins, self.bin_size
 604              ).mean() * (cfg.rho ** t)
 605              for qi in range(cfg.num_q):
 606                  value_loss = value_loss + soft_ce(
 607                      qs[qi, t], td_targets[t], cfg.vmin, cfg.vmax, cfg.num_bins, self.bin_size
 608                  ).mean() * (cfg.rho ** t)
 609          reward_loss = reward_loss / cfg.horizon
 610          value_loss = value_loss / (cfg.horizon * cfg.num_q)
 611  
 612          if cfg.episodic:
 613              termination_pred = self.model.termination(zs[1:], unnormalized=True)
 614              termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
 615          else:
 616              termination_loss = torch.zeros((), device=self.device)
 617  
 618          total_loss = (
 619              cfg.consistency_coef * consistency_loss
 620              + cfg.reward_coef * reward_loss
 621              + cfg.value_coef * value_loss
 622              + cfg.termination_coef * termination_loss
 623          )
 624  
 625          self.optim.zero_grad(set_to_none=True)
 626          total_loss.backward()
 627          torch.nn.utils.clip_grad_norm_(self._world_model_parameters(), cfg.grad_clip_norm)
 628          self.optim.step()
 629          info.critic_update_s = time.perf_counter() - t0
 630          info.consistency_loss = consistency_loss.item()
 631          info.reward_loss = reward_loss.item()
 632          info.value_loss = value_loss.item()
 633          info.total_loss = total_loss.item()
 634  
 635          # --- actor_update: policy prior (Q-maximization + entropy bonus) ---
 636          t0 = time.perf_counter()
 637          pi_info = self._update_pi(zs.detach())
 638          info.actor_update_s = time.perf_counter() - t0
 639          info.pi_loss = pi_info["pi_loss"]
 640  
 641          # --- target_update: Polyak averaging of target Q-ensemble ---
 642          t0 = time.perf_counter()
 643          with torch.no_grad():
 644              for p, p_targ in zip(self.model.qs.parameters(), self.model.qs_target.parameters()):
 645                  p_targ.data.mul_(1 - cfg.tau)
 646                  p_targ.data.add_(cfg.tau * p.data)
 647          info.target_update_s = time.perf_counter() - t0
 648  
 649          self.model.eval()
 650          return info
 651  
 652  
 653  # ------------------------------------------------------------------------
 654  # Training loop
 655  # ------------------------------------------------------------------------
 656  
 657  def train(
 658      env,
 659      tdmpc2_cfg: TDMPC2Config,
 660      exp_cfg: ExperimentConfig,
 661      tracker,
 662      device: torch.device,
 663      logger,
 664      steps_per_epoch: int = 1000,
 665  ):
 666      """
 667      Runs TD-MPC2 warmup + world-model pretraining burst + measured training.
 668      Returns (agent, energy_log, metrics) in the same shape as sac.py's
 669      train() (see that module's docstring for the full "episodes"/"epochs"
 670      schema); "epochs" rows additionally carry consistency_loss_mean,
 671      reward_loss_mean, value_loss_mean, pi_loss_mean, total_loss_mean.
 672  
 673      Segment structure:
 674        - "warmup": purely random actions filling the sequence buffer (same
 675          role as in sac.py/td3.py/mbpo.py).
 676        - "world_model_pretrain": a one-off burst of `warmup_steps` gradient
 677          updates on the seed data the instant warmup ends, matching the
 678          reference OnlineTrainer's `num_updates = seed_steps` pretraining
 679          step -- tracked as its own directly-measured CodeCarbon task since
 680          it is algorithmically distinct from the steady-state 1-update-per-
 681          env-step regime that follows.
 682        - per epoch: "rollout" (MPPI planning + env.step, the dominant cost
 683          for this algorithm -- unlike SAC/TD3/MBPO) and "gradient_updates",
 684          sub-split into buffer_sample/critic_update/actor_update/target_update
 685          by wall-clock time share exactly as in sac.py.
 686      """
 687      obs_dim = env.observation_space.shape[0]
 688      act_dim = env.action_space.shape[0]
 689      act_limit = float(env.action_space.high[0])
 690      episode_length = getattr(getattr(env, "spec", None), "max_episode_steps", None) or tdmpc2_cfg.episode_length
 691  
 692      agent = TDMPC2Agent(obs_dim, act_dim, act_limit, tdmpc2_cfg, device, episode_length=episode_length)
 693      buffer_capacity = min(tdmpc2_cfg.buffer_capacity, exp_cfg.warmup_steps + exp_cfg.train_steps)
 694      buffer = _EpisodeSequenceBuffer(buffer_capacity, obs_dim, act_dim, tdmpc2_cfg.horizon)
 695  
 696      energy_log: Dict[str, float] = {}
 697      sub_time_totals = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
 698  
 699      episodes_log = []
 700      episode_idx = 0
 701      episode_return = 0.0
 702      episode_length_count = 0
 703  
 704      def _record_episode(phase, epoch, env_step):
 705          nonlocal episode_idx, episode_return, episode_length_count
 706          episodes_log.append({
 707              "episode_idx": episode_idx,
 708              "phase": phase,
 709              "epoch": epoch,
 710              "env_step": env_step,
 711              "return": episode_return,
 712              "length": episode_length_count,
 713          })
 714          episode_idx += 1
 715          episode_return = 0.0
 716          episode_length_count = 0
 717  
 718      obs, _ = env.reset(seed=exp_cfg.seed)
 719      buffer.start_episode(obs)
 720      is_t0 = True
 721  
 722      # ---------------- warmup (random policy, fills buffer; excluded from analysis segments) ----------------
 723      logger.info("Starting warmup: %d steps", exp_cfg.warmup_steps)
 724      with TrackerTask(tracker, "warmup", 0, energy_log):
 725          for warmup_step in range(exp_cfg.warmup_steps):
 726              action = env.action_space.sample()
 727              next_obs, reward, terminated, truncated, _ = env.step(action)
 728              buffer.add(action, reward, next_obs, terminated)
 729              episode_return += reward
 730              episode_length_count += 1
 731              obs = next_obs
 732              if terminated or truncated:
 733                  buffer.end_episode()
 734                  _record_episode("warmup", None, warmup_step + 1)
 735                  obs, _ = env.reset()
 736                  buffer.start_episode(obs)
 737                  is_t0 = True
 738  
 739      # ---------------- world-model pretraining burst on seed data ----------------
 740      logger.info("Pretraining world model on seed data: %d updates", exp_cfg.warmup_steps)
 741      pretrain_total_losses = []
 742      with TrackerTask(tracker, "world_model_pretrain", 0, energy_log):
 743          for _ in range(exp_cfg.warmup_steps):
 744              info = agent.update(buffer, tdmpc2_cfg.batch_size)
 745              if info.total_loss is not None:
 746                  pretrain_total_losses.append(info.total_loss)
 747      if pretrain_total_losses:
 748          logger.info(
 749              "Pretrain burst done: %d updates, mean total_loss=%.4f",
 750              len(pretrain_total_losses), sum(pretrain_total_losses) / len(pretrain_total_losses),
 751          )
 752  
 753      # ---------------- measured training ----------------
 754      n_epochs = max(1, exp_cfg.train_steps // steps_per_epoch)
 755      logger.info("Starting measured training: %d epochs x %d steps = %d env steps",
 756                  n_epochs, steps_per_epoch, n_epochs * steps_per_epoch)
 757  
 758      epochs_log = []
 759      cumulative_reward = 0.0
 760      global_step = 0
 761  
 762      for epoch in range(n_epochs):
 763          # --- rollout block (MPPI planning selects every action) ---
 764          epoch_reward_sum = 0.0
 765          epoch_episode_returns = []
 766          with TrackerTask(tracker, "rollout", epoch, energy_log):
 767              for _ in range(steps_per_epoch):
 768                  action = agent.act(obs, t0=is_t0, eval_mode=False)
 769                  is_t0 = False
 770                  next_obs, reward, terminated, truncated, _ = env.step(action)
 771                  buffer.add(action, reward, next_obs, terminated)
 772                  episode_return += reward
 773                  episode_length_count += 1
 774                  cumulative_reward += reward
 775                  epoch_reward_sum += reward
 776                  global_step += 1
 777                  obs = next_obs
 778                  if terminated or truncated:
 779                      buffer.end_episode()
 780                      epoch_episode_returns.append(episode_return)
 781                      _record_episode("train", epoch, global_step)
 782                      obs, _ = env.reset()
 783                      buffer.start_episode(obs)
 784                      is_t0 = True
 785  
 786          # --- gradient update block ---
 787          epoch_sub_times = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
 788          consistency_losses, reward_losses, value_losses, pi_losses, total_losses = [], [], [], [], []
 789          n_updates = steps_per_epoch * tdmpc2_cfg.updates_per_env_step
 790          with TrackerTask(tracker, "gradient_updates", epoch, energy_log):
 791              for _ in range(n_updates):
 792                  info = agent.update(buffer, tdmpc2_cfg.batch_size)
 793                  epoch_sub_times["buffer_sample"] += info.buffer_sample_s
 794                  epoch_sub_times["critic_update"] += info.critic_update_s
 795                  epoch_sub_times["actor_update"] += info.actor_update_s
 796                  epoch_sub_times["target_update"] += info.target_update_s
 797                  if info.consistency_loss is not None:
 798                      consistency_losses.append(info.consistency_loss)
 799                  if info.reward_loss is not None:
 800                      reward_losses.append(info.reward_loss)
 801                  if info.value_loss is not None:
 802                      value_losses.append(info.value_loss)
 803                  if info.pi_loss is not None:
 804                      pi_losses.append(info.pi_loss)
 805                  if info.total_loss is not None:
 806                      total_losses.append(info.total_loss)
 807  
 808          for k in sub_time_totals:
 809              sub_time_totals[k] += epoch_sub_times[k]
 810  
 811          def _mean(xs):
 812              return (sum(xs) / len(xs)) if xs else None
 813  
 814          epochs_log.append({
 815              "epoch": epoch,
 816              "env_step": global_step,
 817              "cumulative_reward": cumulative_reward,
 818              "epoch_reward_sum": epoch_reward_sum,
 819              "epoch_reward_mean_per_step": epoch_reward_sum / steps_per_epoch,
 820              "num_episodes_completed": len(epoch_episode_returns),
 821              "mean_episode_return": (
 822                  sum(epoch_episode_returns) / len(epoch_episode_returns)
 823                  if epoch_episode_returns else None
 824              ),
 825              "consistency_loss_mean": _mean(consistency_losses),
 826              "reward_loss_mean": _mean(reward_losses),
 827              "value_loss_mean": _mean(value_losses),
 828              "pi_loss_mean": _mean(pi_losses),
 829              "total_loss_mean": _mean(total_losses),
 830              "buffer_size": len(buffer),
 831          })
 832  
 833          if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
 834              logger.info(
 835                  "Epoch %d/%d done (buffer size=%d, cumulative_reward=%.2f, mean_episode_return=%s)",
 836                  epoch + 1, n_epochs, len(buffer), cumulative_reward,
 837                  epochs_log[-1]["mean_episode_return"],
 838              )
 839  
 840      # record a trailing partial episode (if training ended mid-episode) so no reward is lost
 841      if episode_length_count > 0:
 842          buffer.end_episode()
 843          _record_episode("train_incomplete", n_epochs - 1, global_step)
 844  
 845      # ---------------- reconcile: split total gradient_updates energy by aggregate time share ----------------
 846      total_gradient_energy = energy_log.pop("gradient_updates", 0.0)
 847      total_time = sum(sub_time_totals.values())
 848      for k, t in sub_time_totals.items():
 849          share = (t / total_time) if total_time > 0 else 0.0
 850          energy_log[k] = total_gradient_energy * share
 851  
 852      energy_log["_sub_segment_wall_time_seconds"] = sub_time_totals
 853      metrics = {"episodes": episodes_log, "epochs": epochs_log}
 854      return agent, energy_log, metrics
```
`algorithms/tracker_utils.py` is shown in Section 4.2 / 4.1 (`TrackerTask` names tasks `f"{prefix}_{counter}"`; `world_model_pretrain` uses counter 0).


**2.3 Parameter counts** — real `WorldModel(obs_dim, act_dim, cfg)` instantiated on CPU (weights + biases + LayerNorm parameters; `sum(p.numel())`). Components: encoder (obs → 256 → 512, SimNorm), latent dynamics (512+act → 512 → 512 → 512, SimNorm), reward model (→ 101 bins), policy prior (→ 2·act), Q-ensemble `qs` (num_q networks, each → 101 bins), `qs_target` (a deep copy of `qs`, not trainable), termination classifier (512 → 512 → 512 → 1; built only if `episodic`). `trainable` = encoder + dynamics + reward + termination + policy prior + Q-ensemble (the target ensemble is excluded; the world-model optimiser covers encoder, dynamics, reward, Q-ensemble and termination, the policy prior has its own optimiser). 'used by runs' marks the combinations that appear in `results/` (HC: episodic false; Ant: episodic true):


| env | obs/act | num_q | episodic | in results/ | encoder | dynamics | reward model | termination classifier | policy prior | Q-ensemble (all) | per Q member | target Q-ensemble | trainable total | trainable + target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | 17/6 | 3 | False | yes | 137728 | 794112 | 582245 | — (not built) | 533516 | 1746735 | 582245 | 1746735 | 3794336 | 5541071 |
| HalfCheetah-v5 | 17/6 | 5 | False | yes | 137728 | 794112 | 582245 | — (not built) | 533516 | 2911225 | 582245 | 2911225 | 4958826 | 7870051 |
| HalfCheetah-v5 | 17/6 | 7 | False | yes | 137728 | 794112 | 582245 | — (not built) | 533516 | 4075715 | 582245 | 4075715 | 6123316 | 10199031 |
| Ant-v5 | 105/8 | 3 | False | (not run) | 160256 | 795136 | 583269 | — (not built) | 535568 | 1749807 | 583269 | 1749807 | 3824036 | 5573843 |
| Ant-v5 | 105/8 | 3 | True | yes | 160256 | 795136 | 583269 | 527873 | 535568 | 1749807 | 583269 | 1749807 | 4351909 | 6101716 |
| Ant-v5 | 105/8 | 5 | False | (not run) | 160256 | 795136 | 583269 | — (not built) | 535568 | 2916345 | 583269 | 2916345 | 4990574 | 7906919 |
| Ant-v5 | 105/8 | 5 | True | yes | 160256 | 795136 | 583269 | 527873 | 535568 | 2916345 | 583269 | 2916345 | 5518447 | 8434792 |
| Ant-v5 | 105/8 | 7 | False | (not run) | 160256 | 795136 | 583269 | — (not built) | 535568 | 4082883 | 583269 | 4082883 | 6157112 | 10239995 |
| Ant-v5 | 105/8 | 7 | True | yes | 160256 | 795136 | 583269 | 527873 | 535568 | 4082883 | 583269 | 4082883 | 6684985 | 10767868 |

- **PASS** — local environment dims equal the dims stored in flops_per_call.json (HC 17/6, Ant 105/8); `env.spec.max_episode_steps` = 1000 (HC) / 1000 (Ant) → discount = 0.995 for both (class docstring formula).

**2.4 Planner settings per configuration** (logged `algo_config`; `iterations` column = effective `agent.iterations` = `cfg.iterations + 2·(act_dim ≥ 20)` = 6 for both environments):


| config | num_samples | policy-seeded (num_pi_trajs) | Gaussian-sampled | horizon | MPPI iterations | num_elites | temperature | min_std/max_std | num_q | episodic | batch | UTD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HC-base | 512 | 24 | 488 | 3 | 6 | 64 | 0.5 | 0.05/2.0 | 5 | False | 256 | 1 |
| HC-numq3 | 512 | 24 | 488 | 3 | 6 | 64 | 0.5 | 0.05/2.0 | 3 | False | 256 | 1 |
| HC-numq7 | 512 | 24 | 488 | 3 | 6 | 64 | 0.5 | 0.05/2.0 | 7 | False | 256 | 1 |
| HC-hor1 | 512 | 24 | 488 | 1 | 6 | 64 | 0.5 | 0.05/2.0 | 5 | False | 256 | 1 |
| HC-hor5 | 512 | 24 | 488 | 5 | 6 | 64 | 0.5 | 0.05/2.0 | 5 | False | 256 | 1 |
| Ant-base | 512 | 24 | 488 | 3 | 6 | 64 | 0.5 | 0.05/2.0 | 5 | True | 256 | 1 |
| Ant-numq3 | 512 | 24 | 488 | 3 | 6 | 64 | 0.5 | 0.05/2.0 | 3 | True | 256 | 1 |
| Ant-numq7 | 512 | 24 | 488 | 3 | 6 | 64 | 0.5 | 0.05/2.0 | 7 | True | 256 | 1 |
| Ant-hor1 | 512 | 24 | 488 | 1 | 6 | 64 | 0.5 | 0.05/2.0 | 5 | True | 256 | 1 |
| Ant-hor5 | 512 | 24 | 488 | 5 | 6 | 64 | 0.5 | 0.05/2.0 | 5 | True | 256 | 1 |


**2.5 Per-call FLOPs from `flop_analysis/flops_per_call.json`** (`measure_flops.measure_tdmpc2`: one unmodified `TDMPC2Agent.act(obs, t0=False, eval_mode=False)` for the rollout; one unmodified `agent.update()` black-box for the cross-check; the 3-way breakdown replicates the body of `update()` in three `FlopCounterMode` windows; measured on `cpu`; matmul FLOPs = 2·m·n·k). Cell `HC ‖ Ant`:


| config | rollout_per_env_step (one agent.act()) | gradient_update_critic (world-model step) | gradient_update_actor (policy-prior step) | gradient_update_target (FLOPs; Polyak is elementwise) | gradient_update_target_elementwise_ops | gradient_update_total (critic+actor+target) | black-box agent.update() total | breakdown vs black-box deviation [%] | iterations / horizon / num_q / episodic |
|---|---|---|---|---|---|---|---|---|---|
| base | 4.64261e+10 ‖ 5.61814e+10 | 2.53240e+10 ‖ 2.78440e+10 | 1.45815e+10 ‖ 1.46151e+10 | 0 ‖ 0 | 5.82245e+06 ‖ 5.83269e+06 | 3.99055e+10 ‖ 4.24591e+10 | 3.99055e+10 ‖ 4.24591e+10 | 0 ‖ 0 | 6/3/5/False ‖ 6/3/5/True |
| numq3 | 3.93105e+10 ‖ 4.90532e+10 | 1.82084e+10 ‖ 2.07158e+10 | 9.83774e+09 ‖ 9.86291e+09 | 0 ‖ 0 | 3.49347e+06 ‖ 3.49961e+06 | 2.80461e+10 ‖ 3.05787e+10 | 2.80461e+10 ‖ 3.05787e+10 | 0 ‖ 0 | 6/3/3/False ‖ 6/3/3/True |
| numq7 | 5.35417e+10 ‖ 6.33097e+10 | 3.24397e+10 ‖ 3.49722e+10 | 1.93253e+10 ‖ 1.93672e+10 | 0 ‖ 0 | 8.15143e+06 ‖ 8.16577e+06 | 5.17649e+10 ‖ 5.43394e+10 | 5.17649e+10 ‖ 5.43394e+10 | 0 ‖ 0 | 6/3/7/False ‖ 6/3/7/True |
| hor1 | 2.94823e+10 ‖ 3.27635e+10 | 8.57853e+09 ‖ 9.43391e+09 | 7.29075e+09 ‖ 7.30753e+09 | 0 ‖ 0 | 5.82245e+06 ‖ 5.83269e+06 | 1.58693e+10 ‖ 1.67414e+10 | 1.58693e+10 ‖ 1.67414e+10 | 0 ‖ 0 | 6/1/5/False ‖ 6/1/5/True |
| hor5 | 6.33699e+10 ‖ 7.95994e+10 | 4.20695e+10 ‖ 4.62541e+10 | 2.18722e+10 ‖ 2.19226e+10 | 0 ‖ 0 | 5.82245e+06 ‖ 5.83269e+06 | 6.39418e+10 ‖ 6.81767e+10 | 6.39418e+10 ‖ 6.81767e+10 | 0 ‖ 0 | 6/5/5/False ‖ 6/5/5/True |

- **PASS** — the 3-way breakdown equals the black-box `agent.update()` FLOP count exactly in every signature (max |deviation| = 0.0%)

**2.6 Analytic dense-layer model of the FLOP constants** (derived by this generator, not a pipeline output; Linear layers only, 2·n_in·n_out per sample; formulas below).
Let enc, dyn, rew, π, Q, term be the per-sample matmul FLOPs of the encoder, latent dynamics, reward model, policy prior, one Q network and the termination classifier (term = 0 unless episodic); H = horizon, nq = num_q, B = batch (256), ns = num_samples (512), npi = num_pi_trajs (24), it = iterations (6):
```
act (rollout, one call) = enc + npi·[(H−1)(π+dyn) + π] + it·ns·[ H(rew + dyn + term) + π + nq·Q ]
update critic_update    = H·B·(enc + π + nq·Q)                                  # no-grad: encode(obs[1:]), π(next_z), Q(target, all nq nets)
                        + 3·[ B·enc + H·B·dyn + H·B·nq·Q + H·B·rew + H·B·term ] # grad forward + weight-grad + input-grad
                        − 2·B·obs_dim·enc_dim                                    # the first encoder layer needs no input-grad
update actor_update     = (H+1)·B·( 3π + 2·nq·Q − 2·latent·mlp )                # π fwd + π weight-grad + π input-grad (minus its first layer) + Q fwd + Q input-grad over all nq nets
update target_update    = 0 matmul FLOPs; Polyak ops = 2·Σ params(Q-ensemble)
```
`Q(…, return_type="avg"/"all"/"min", …)` evaluates **all** nq networks (`torch.stack([net(x) for net in nets])`) and only then selects/averages, so nq enters every Q term linearly. Comparison with the measured constants:


| config | rollout measured | rollout analytic | critic measured | critic analytic | actor measured | actor analytic | analytic vs measured |
|---|---|---|---|---|---|---|---|
| HC-base | 46426104320 | 46426104320 | 25324027904 | 25324027904 | 14581497856 | 14581497856 | exact |
| HC-numq3 | 39310467584 | 39310467584 | 18208391168 | 18208391168 | 9837740032 | 9837740032 | exact |
| HC-numq7 | 53541741056 | 53541741056 | 32439664640 | 32439664640 | 19325255680 | 19325255680 | exact |
| HC-hor1 | 29482328576 | 29482328576 | 8578531328 | 8578531328 | 7290748928 | 7290748928 | exact |
| HC-hor5 | 63369880064 | 63369880064 | 42069524480 | 42069524480 | 21872246784 | 21872246784 | exact |
| Ant-base | 56181445120 | 56181445120 | 27844018176 | 27844018176 | 14615052288 | 14615052288 | exact |
| Ant-numq3 | 49053225472 | 49053225472 | 20715798528 | 20715798528 | 9862905856 | 9862905856 | exact |
| Ant-numq7 | 63309664768 | 63309664768 | 34972237824 | 34972237824 | 19367198720 | 19367198720 | exact |
| Ant-hor1 | 32763466240 | 32763466240 | 9433907200 | 9433907200 | 7307526144 | 7307526144 | exact |
| Ant-hor5 | 79599424000 | 79599424000 | 46254129152 | 46254129152 | 21922578432 | 21922578432 | exact |

- **PASS** — the analytic model equals the FlopCounterMode counts exactly for rollout, critic and actor in all 10 (env, configuration) entries

**2.7 Call counts and where they are defined** (`compute_energy_per_flop.rows_for_tdmpc2`, called from `main()`; verbatim):

```python
 382  def rows_for_tdmpc2(flops_key, algo_config, experiment_config):
 383      steps_per_epoch = experiment_config["steps_per_epoch"]
 384      n_epochs = max(1, experiment_config["train_steps"] // steps_per_epoch)
 385      total_env_steps = n_epochs * steps_per_epoch
 386      n_updates = total_env_steps * algo_config["updates_per_env_step"]
 387      warmup = experiment_config["warmup_steps"]
 388  
 389      return {
 390          "rollout": (total_env_steps, flops_key["rollout_per_env_step"] * total_env_steps, "MPPI planning call (encoder + planner + pi/Q), per env step"),
 391          "world_model_pretrain": (
 392              warmup, flops_key["gradient_update_total"] * warmup,
 393              "one-off burst of warmup_steps full agent.update() calls; logged as a single fused CodeCarbon task, not time-split",
 394          ),
 395          "buffer_sample": (n_updates, 0, "CPU-side sequence sampling, excluded from GPU FLOP accounting"),
 396          "critic_update": (n_updates, flops_key["gradient_update_critic"] * n_updates, "world-model step (consistency+reward+value losses), fwd+bwd"),
 397          "actor_update": (n_updates, flops_key["gradient_update_actor"] * n_updates, "policy-prior step (_update_pi), fwd+bwd"),
 398          "target_update": (
 399              n_updates, flops_key["gradient_update_target_elementwise_ops"] * n_updates,
 400              "Polyak averaging of Q-ensemble, elementwise ops (not matmul FLOPs)",
 401          ),
 402      }
 403  
 404
```
```
n_env     = max(1, train_steps // steps_per_epoch) * steps_per_epoch    # = 100000 rollout calls (one agent.act() per env step)
n_upd     = n_env * updates_per_env_step                                # = 100000: buffer_sample = critic_update = actor_update = target_update calls
pretrain  = warmup_steps                                                # = 5000 agent.update() calls in the world_model_pretrain task
F_rollout = rollout_per_env_step * n_env;  F_pretrain = gradient_update_total * warmup_steps
F_critic  = gradient_update_critic * n_upd;  F_actor = gradient_update_actor * n_upd;  OPS_target = gradient_update_target_elementwise_ops * n_upd
```
The loop issuing the calls is `train()` (`algorithms/tdmpc2.py`): `for _ in range(warmup_steps): agent.update(...)` inside `TrackerTask(tracker, "world_model_pretrain", 0, …)`, and per epoch `n_updates = steps_per_epoch * updates_per_env_step` calls of `agent.update` inside `gradient_updates_{epoch}`; unlike SAC/TD3/MBPO there is **no** `len(buffer) < batch_size` guard (`update()` returns early with all-zero timers only if `buffer.sample` returns `None`, i.e. no stored episode is at least `horizon` long). Call counts per run (identical across seeds):


| config | calls rollout | calls wm_pretrain | calls buf_sample | calls critic | calls actor | calls target |
|---|---|---|---|---|---|---|
| base | 100000 ‖ 100000 | 5000 ‖ 5000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 |
| numq3 | 100000 ‖ 100000 | 5000 ‖ 5000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 |
| numq7 | 100000 ‖ 100000 | 5000 ‖ 5000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 |
| hor1 | 100000 ‖ 100000 | 5000 ‖ 5000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 |
| hor5 | 100000 ‖ 100000 | 5000 ‖ 5000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 |

- **PASS** — call counts equal the `call_count` column of per_run_energy_per_flop.csv for all 50 runs
- **PASS** — `run.log` reports `Pretrain burst done: 5000 updates` in every run (the only logged update count)

**2.8 Total FLOPs per run and segment** (`wm_pretrain` = world_model_pretrain; `GU` = critic + actor; `TOTAL` = rollout + wm_pretrain + critic + actor; `target` = elementwise Polyak operations):


| config | F wm_pretrain | F rollout | F critic | F actor | F target [elementwise ops] | F GU | F TOTAL |
|---|---|---|---|---|---|---|---|
| base | 1.99528e+14 ‖ 2.12295e+14 | 4.64261e+15 ‖ 5.61814e+15 | 2.53240e+15 ‖ 2.78440e+15 | 1.45815e+15 ‖ 1.46151e+15 | 5.82245e+11 ‖ 5.83269e+11 | 3.99055e+15 ‖ 4.24591e+15 | 8.83269e+15 ‖ 1.00763e+16 |
| numq3 | 1.40231e+14 ‖ 1.52894e+14 | 3.93105e+15 ‖ 4.90532e+15 | 1.82084e+15 ‖ 2.07158e+15 | 9.83774e+14 ‖ 9.86291e+14 | 3.49347e+11 ‖ 3.49961e+11 | 2.80461e+15 ‖ 3.05787e+15 | 6.87589e+15 ‖ 8.11609e+15 |
| numq7 | 2.58825e+14 ‖ 2.71697e+14 | 5.35417e+15 ‖ 6.33097e+15 | 3.24397e+15 ‖ 3.49722e+15 | 1.93253e+15 ‖ 1.93672e+15 | 8.15143e+11 ‖ 8.16577e+11 | 5.17649e+15 ‖ 5.43394e+15 | 1.07895e+16 ‖ 1.20366e+16 |
| hor1 | 7.93464e+13 ‖ 8.37072e+13 | 2.94823e+15 ‖ 3.27635e+15 | 8.57853e+14 ‖ 9.43391e+14 | 7.29075e+14 ‖ 7.30753e+14 | 5.82245e+11 ‖ 5.83269e+11 | 1.58693e+15 ‖ 1.67414e+15 | 4.61451e+15 ‖ 5.03420e+15 |
| hor5 | 3.19709e+14 ‖ 3.40884e+14 | 6.33699e+15 ‖ 7.95994e+15 | 4.20695e+15 ‖ 4.62541e+15 | 2.18722e+15 ‖ 2.19226e+15 | 5.82245e+11 ‖ 5.83269e+11 | 6.39418e+15 ‖ 6.81767e+15 | 1.30509e+16 ‖ 1.51185e+16 |

Ant / HalfCheetah ratios (Ant uses `episodic` = true, so its constants include the termination classifier, Section 6.6):


| config | Ant/HC wm_pretrain | Ant/HC rollout | Ant/HC critic | Ant/HC actor | Ant/HC target | Ant/HC GU | Ant/HC TOTAL |
|---|---|---|---|---|---|---|---|
| base | 1.06399 | 1.21013 | 1.09951 | 1.00230 | 1.00176 | 1.06399 | 1.14080 |
| numq3 | 1.09030 | 1.24784 | 1.13771 | 1.00256 | 1.00176 | 1.09030 | 1.18037 |
| numq7 | 1.04973 | 1.18244 | 1.07807 | 1.00217 | 1.00176 | 1.04973 | 1.11559 |
| hor1 | 1.05496 | 1.11129 | 1.09971 | 1.00230 | 1.00176 | 1.05496 | 1.09095 |
| hor5 | 1.06623 | 1.25611 | 1.09947 | 1.00230 | 1.00176 | 1.06623 | 1.15843 |


## 3. Baseline decomposition and energy per FLOP (4.4.1)

Configuration `base` (num_q 5, horizon 3; shared by both sweeps).

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | wm_pretrain (measured) | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|---|
| base | 10863.4 ± 469.216 ‖ 10862.2 ± 103.363 | 213811 ± 333.591 ‖ 254576 ± 533.613 | 9884.04 ± 126.916 ‖ 12025.3 ± 365.269 | 146372 ± 2470.48 ‖ 149766 ± 1894.13 | 56140.9 ± 469.436 ‖ 55457.1 ± 325.155 | 10015.4 ± 109.750 ‖ 10504.6 ± 162.674 | 222413 ± 2740.99 ‖ 227753 ± 1855.54 | 447087 ± 3184.37 ‖ 493191 ± 1781.98 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| base | 0.124191 ± 8.84546e-04 ‖ 0.136998 ± 4.94994e-04 | 17.2086 ± 2.57921 ‖ 76.9060 ± 0.211427 | 5803.46 ± 105.464 ‖ 5868.73 ± 108.546 | 6489.87 ± 60.8709 ‖ 6599.26 ± 116.131 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | wm_pretrain (measured) | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|---|
| base | 2.42935 ± 0.0889482 ‖ 2.20242 ± 0.0166962 | 47.8250 ± 0.341328 ‖ 51.6187 ± 0.227199 | 2.21099 ± 0.0427318 ‖ 2.43841 ± 0.0791294 | 32.7373 ± 0.323777 ‖ 30.3660 ± 0.288290 | 12.5570 ± 0.0391886 ‖ 11.2445 ± 0.0318135 | 2.24032 ± 0.0364381 ‖ 2.12999 ± 0.0362218 | 49.7456 ± 0.269537 ‖ 46.1789 ± 0.221384 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| base | 4.44503 ± 0.107498 ‖ 5.28097 ± 0.192422 | 65.8082 ± 0.301085 ‖ 65.7562 ± 0.322405 | 25.2428 ± 0.118049 ‖ 24.3501 ± 0.0663820 | 4.50396 ± 0.0963213 ‖ 4.61277 ± 0.0955640 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | wm_pretrain (measured) [J/FLOP] | rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|---|
| base | 5.44456e-11 ± 2.35163e-12 ‖ 5.11657e-11 ± 4.86883e-13 | 4.60540e-11 ± 7.18542e-14 ‖ 4.53132e-11 ± 9.49803e-14 | 5.77998e-11 ± 9.75548e-13 ‖ 5.37874e-11 ± 6.80265e-13 | 3.85014e-11 ± 3.21940e-13 ‖ 3.79452e-11 ± 2.22479e-13 | 5.57348e-11 ± 6.86870e-13 ‖ 5.36406e-11 ± 4.37019e-13 | 5.06173e-11 ± 3.60521e-13 ‖ 4.89454e-11 ± 1.76848e-13 | 17.2013 ± 0.188494 ‖ 18.0100 ± 0.278901 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | wm_pretrain | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|
| base | 0.939758 | 0.983914 | 0.930582 | 0.985553 | 0.962426 | 0.966971 | 1.04701 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | wm_pretrain (measured) duration [s] | rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|---|
| base | 64.2556 ± 5.67548 ‖ 61.0457 ± 1.97208 | 1014.24 ± 2.51688 ‖ 1194.74 ± 0.509042 | 1281.04 ± 33.4115 ‖ 1259.62 ± 27.2265 | 56.0595 ± 0.467656 ‖ 65.1598 ± 1.28392 | 830.437 ± 25.6873 ‖ 811.869 ± 21.7115 | 318.484 ± 7.23155 ‖ 300.605 ± 6.00130 | 56.8054 ± 0.442010 ‖ 56.9297 ± 0.660151 | 2359.54 ± 39.4543 ‖ 2515.40 ± 28.7095 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P wm_pretrain [W] | P rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|---|
| base | 169.629 ± 8.02768 ‖ 178.038 ± 3.94655 | 210.809 ± 0.409232 ‖ 213.081 ± 0.492626 | 173.669 ± 2.42160 ‖ 180.854 ± 2.53444 | 189.505 ± 1.85208 ‖ 196.083 ± 1.65782 | 180.906 ± 1.57862 ‖ 187.594 ± 1.50778 | 1.21402 ± 0.0150018 ‖ 1.17835 ± 0.0143448 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | wm_pretrain [GFLOP/s] | rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|---|
| base | 3125.33 ± 284.681 ‖ 3480.44 ± 107.856 | 4577.45 ± 11.3509 ‖ 4702.40 ± 2.00349 | 3116.78 ± 80.9857 ‖ 3372.06 ± 73.4443 | 3744.24 ± 62.5860 ‖ 4006.27 ± 45.8682 | 3051.81 ± 94.0394 ‖ 3431.61 ± 92.7349 | 4580.28 ± 103.436 ‖ 4863.43 ± 97.8338 | 98.4963 ± 0.0380504 ‖ 98.0101 ± 0.0479152 | 98.4516–98.5420 ‖ 97.9500–98.0501 |

**Hardware composition of the measured `world_model_pretrain` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of wm_pretrain energy [%] | GPU share of wm_pretrain energy [%] | RAM share of wm_pretrain energy [%] | wm_pretrain energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 15.7920 ± 0.828654 ‖ 16.2271 ± 0.796714 | 72.3968 ± 0.418271 ‖ 72.5349 ± 0.555025 | 11.8112 ± 0.553187 ‖ 11.2380 ± 0.256291 | 0/5 ‖ 0/5 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 16.1063 ± 0.0490408 ‖ 16.1563 ± 0.0665349 | 74.4068 ± 0.0597546 ‖ 74.4579 ± 0.0844652 | 9.48688 ± 0.0184493 ‖ 9.38583 ± 0.0217202 | 0/500 ‖ 0/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 16.0554 ± 0.164318 ‖ 16.4391 ± 0.0907562 | 72.4271 ± 0.138522 ‖ 72.5011 ± 0.0662238 | 11.5175 ± 0.161250 ‖ 11.0599 ± 0.154272 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | wm_pretrain [%] | rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|---|
| base | 40.3342 ± 1.96137 ‖ 38.9192 ± 0.964806 | 32.3962 ± 0.250611 ‖ 32.5058 ± 0.359148 | 39.3313 ± 0.687072 ‖ 38.2995 ± 0.220941 | 36.0413 ± 0.493229 ‖ 35.3234 ± 0.224873 | 68.2939 ± 0.475666 ‖ 69.2642 ± 0.847964 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| base | 1.10312 | 1.19066 | 3.79369 | rollout | 1.03471 | 0.970615 | 0.966971 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | wm_pretrain | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|---|
| base | 1.19066 (paired sd 0.00256929) | 0.999893 (paired sd 0.0403492) | 1.21664 (paired sd 0.0380107) | 1.02318 (paired sd 0.0144273) | 0.987821 (paired sd 0.00509623) | 1.04885 (paired sd 0.0179238) | 1.02401 (paired sd 0.0103813) | 1.10312 (paired sd 0.00661558) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | wm_pretrain | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|---|
| base | -0.226928 ± 0.0834794 (5/5 same sign) | 3.79369 ± 0.291936 (5/5 same sign) | 0.227412 ± 0.0776402 (5/5 same sign) | -2.37134 ± 0.277015 (5/5 same sign) | -1.31250 ± 0.0334668 (5/5 same sign) | -0.110334 ± 0.0379228 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| base | mean Δ 3.79369; 2/32 patterns; p = 0.0625000 | mean Δ 46104.4; 2/32 patterns; p = 0.0625000 |


## 4.1 Sweep: num_q — num_q ∈ {3,5,7} (Q-ensemble size)

Configurations: `numq3` (num_q 3), `base` (paper hyperparameters: num_q 5, horizon 3, 512 samples (24 policy-seeded), 6 iterations, 64 elites, batch 256, UTD 1; Ant: episodic = true (tdmpc2_ant.json)), `numq7` (num_q 7). Baseline for the response tables: `base` of the same environment (shared by both sweeps).

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | wm_pretrain (measured) | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|---|
| numq3 | 8278.54 ± 325.475 ‖ 8394.74 ± 42.6281 | 181988 ± 330.474 ‖ 223015 ± 211.968 | 9349.00 ± 117.046 ‖ 11252.4 ± 98.4619 | 108897 ± 1988.81 ‖ 117837 ± 1799.45 | 44321.8 ± 345.033 ‖ 44866.0 ± 360.444 | 5953.15 ± 64.6741 ‖ 5992.19 ± 40.0815 | 168521 ± 2246.68 ‖ 179947 ± 2065.94 | 358787 ± 2182.09 ‖ 411357 ± 2042.42 |
| base | 10863.4 ± 469.216 ‖ 10862.2 ± 103.363 | 213811 ± 333.591 ‖ 254576 ± 533.613 | 9884.04 ± 126.916 ‖ 12025.3 ± 365.269 | 146372 ± 2470.48 ‖ 149766 ± 1894.13 | 56140.9 ± 469.436 ‖ 55457.1 ± 325.155 | 10015.4 ± 109.750 ‖ 10504.6 ± 162.674 | 222413 ± 2740.99 ‖ 227753 ± 1855.54 | 447087 ± 3184.37 ‖ 493191 ± 1781.98 |
| numq7 | 12734.4 ± 25.5126 ‖ 13317.9 ± 446.634 | 242194 ± 483.077 ‖ 280891 ± 552.437 | 10422.0 ± 126.154 ‖ 12011.5 ± 140.763 | 180090 ± 2117.03 ‖ 189610 ± 1799.61 | 66730.9 ± 412.683 ‖ 66877.7 ± 351.986 | 14390.1 ± 220.929 ‖ 14213.6 ± 113.432 | 271633 ± 2216.76 ‖ 282713 ± 2197.25 | 526562 ± 1974.22 ‖ 576921 ± 2823.71 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| numq3 | 0.0996631 ± 6.06135e-04 ‖ 0.114266 ± 5.67338e-04 | 16.9920 ± 3.13609 ‖ 77.5119 ± 0.587381 | 5762.35 ± 71.0848 ‖ 5824.91 ± 61.5477 | 6237.17 ± 66.6542 ‖ 6281.99 ± 47.9324 |
| base | 0.124191 ± 8.84546e-04 ‖ 0.136998 ± 4.94994e-04 | 17.2086 ± 2.57921 ‖ 76.9060 ± 0.211427 | 5803.46 ± 105.464 ‖ 5868.73 ± 108.546 | 6489.87 ± 60.8709 ‖ 6599.26 ± 116.131 |
| numq7 | 0.146267 ± 5.48395e-04 ‖ 0.160256 ± 7.84364e-04 | 18.1017 ± 2.97443 ‖ 76.2210 ± 2.77566 | 5746.33 ± 93.5626 ‖ 5707.09 ± 60.4655 | 6343.06 ± 87.7219 ‖ 6377.44 ± 75.4883 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | wm_pretrain (measured) | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|---|
| numq3 | 2.30760 ± 0.0967264 ‖ 2.04077 ± 0.0121289 | 50.7244 ± 0.296225 ‖ 54.2155 ± 0.280534 | 2.60582 ± 0.0376147 ‖ 2.73554 ± 0.0331660 | 30.3497 ± 0.376015 ‖ 28.6447 ± 0.298144 | 12.3531 ± 0.0279932 ‖ 10.9067 ± 0.0337170 | 1.65935 ± 0.0260796 ‖ 1.45675 ± 0.0162102 | 46.9680 ± 0.347478 ‖ 43.7437 ± 0.287604 |
| base | 2.42935 ± 0.0889482 ‖ 2.20242 ± 0.0166962 | 47.8250 ± 0.341328 ‖ 51.6187 ± 0.227199 | 2.21099 ± 0.0427318 ‖ 2.43841 ± 0.0791294 | 32.7373 ± 0.323777 ‖ 30.3660 ± 0.288290 | 12.5570 ± 0.0391886 ‖ 11.2445 ± 0.0318135 | 2.24032 ± 0.0364381 ‖ 2.12999 ± 0.0362218 | 49.7456 ± 0.269537 ‖ 46.1789 ± 0.221384 |
| numq7 | 2.41843 ± 0.00915078 ‖ 2.30829 ± 0.0703969 | 45.9960 ± 0.226780 ‖ 48.6886 ± 0.187380 | 1.97933 ± 0.0302471 ‖ 2.08200 ± 0.0225850 | 34.2004 ± 0.280452 ‖ 32.8652 ± 0.164716 | 12.6729 ± 0.0340671 ‖ 11.5922 ± 0.0229882 | 2.73297 ± 0.0508143 ‖ 2.46370 ± 0.0176217 | 51.5855 ± 0.235684 ‖ 49.0031 ± 0.151920 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| numq3 | 5.54861 ± 0.109963 ‖ 6.25412 ± 0.112220 | 64.6159 ± 0.326146 ‖ 65.4818 ± 0.254904 | 26.3020 ± 0.149646 ‖ 24.9336 ± 0.0908377 | 3.53342 ± 0.0807935 ‖ 3.33049 ± 0.0584487 |
| base | 4.44503 ± 0.107498 ‖ 5.28097 ± 0.192422 | 65.8082 ± 0.301085 ‖ 65.7562 ± 0.322405 | 25.2428 ± 0.118049 ‖ 24.3501 ± 0.0663820 | 4.50396 ± 0.0963213 ‖ 4.61277 ± 0.0955640 |
| numq7 | 3.83726 ± 0.0761771 ‖ 4.24880 ± 0.0537355 | 66.2975 ± 0.243132 ‖ 67.0673 ± 0.132601 | 24.5669 ± 0.0527701 ‖ 23.6561 ± 0.0689202 | 5.29836 ± 0.121634 ‖ 5.02774 ± 0.0470205 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | wm_pretrain (measured) [J/FLOP] | rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|---|
| numq3 | 5.90352e-11 ± 2.32100e-12 ‖ 5.49058e-11 ± 2.78809e-13 | 4.62950e-11 ± 8.40677e-14 ‖ 4.54638e-11 ± 4.32119e-14 | 5.98059e-11 ± 1.09225e-12 ‖ 5.68826e-11 ± 8.68635e-13 | 4.50528e-11 ± 3.50724e-13 ‖ 4.54896e-11 ± 3.65454e-13 | 6.00870e-11 ± 8.01065e-13 ‖ 5.88473e-11 ± 6.75613e-13 | 5.21805e-11 ± 3.17353e-13 ‖ 5.06841e-11 ± 2.51650e-13 | 17.0408 ± 0.185129 ‖ 17.1224 ± 0.114531 |
| base | 5.44456e-11 ± 2.35163e-12 ‖ 5.11657e-11 ± 4.86883e-13 | 4.60540e-11 ± 7.18542e-14 ‖ 4.53132e-11 ± 9.49803e-14 | 5.77998e-11 ± 9.75548e-13 ‖ 5.37874e-11 ± 6.80265e-13 | 3.85014e-11 ± 3.21940e-13 ‖ 3.79452e-11 ± 2.22479e-13 | 5.57348e-11 ± 6.86870e-13 ‖ 5.36406e-11 ± 4.37019e-13 | 5.06173e-11 ± 3.60521e-13 ‖ 4.89454e-11 ± 1.76848e-13 | 17.2013 ± 0.188494 ‖ 18.0100 ± 0.278901 |
| numq7 | 4.92010e-11 ± 9.85711e-14 ‖ 4.90174e-11 ± 1.64387e-12 | 4.52347e-11 ± 9.02244e-14 ‖ 4.43678e-11 ± 8.72596e-14 | 5.55155e-11 ± 6.52606e-13 ‖ 5.42172e-11 ± 5.14584e-13 | 3.45304e-11 ± 2.13546e-13 ‖ 3.45314e-11 ± 1.81743e-13 | 5.24744e-11 ± 4.28236e-13 ‖ 5.20271e-11 ± 4.04357e-13 | 4.88032e-11 ± 1.82976e-13 ‖ 4.79306e-11 ± 2.34594e-13 | 17.6535 ± 0.271032 ‖ 17.4063 ± 0.138911 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | wm_pretrain | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|
| numq3 | 0.930052 | 0.982047 | 0.951119 | 1.00970 | 0.979368 | 0.971324 | 1.00479 |
| base | 0.939758 | 0.983914 | 0.930582 | 0.985553 | 0.962426 | 0.966971 | 1.04701 |
| numq7 | 0.996269 | 0.980835 | 0.976615 | 1.00003 | 0.991477 | 0.982118 | 0.985999 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | wm_pretrain (measured) duration [s] | rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|---|
| numq3 | 52.2241 ± 3.59849 ‖ 50.0223 ± 0.229119 | 886.412 ± 2.79221 ‖ 1071.12 ± 0.463104 | 1030.51 ± 24.8448 ‖ 1085.20 ± 23.8743 | 56.0467 ± 0.744702 ‖ 66.3118 ± 0.626746 | 652.973 ± 19.6997 ‖ 694.555 ± 18.4516 | 265.737 ± 5.16376 ‖ 264.432 ± 5.09664 | 35.6874 ± 0.238708 ‖ 35.3123 ± 0.211256 | 1969.15 ± 25.9506 ‖ 2206.34 ± 23.4077 |
| base | 64.2556 ± 5.67548 ‖ 61.0457 ± 1.97208 | 1014.24 ± 2.51688 ‖ 1194.74 ± 0.509042 | 1281.04 ± 33.4115 ‖ 1259.62 ± 27.2265 | 56.0595 ± 0.467656 ‖ 65.1598 ± 1.28392 | 830.437 ± 25.6873 ‖ 811.869 ± 21.7115 | 318.484 ± 7.23155 ‖ 300.605 ± 6.00130 | 56.8054 ± 0.442010 ‖ 56.9297 ± 0.660151 | 2359.54 ± 39.4543 ‖ 2515.40 ± 28.7095 |
| numq7 | 69.5999 ± 0.405093 ‖ 74.8141 ± 5.90138 | 1138.66 ± 0.938246 ‖ 1322.94 ± 0.621085 | 1506.25 ± 31.4215 ‖ 1589.09 ± 22.4095 | 56.8906 ± 0.156430 ‖ 66.2877 ± 0.781571 | 983.305 ± 24.3155 ‖ 1046.47 ± 17.0090 | 364.334 ± 6.96507 ‖ 369.094 ± 4.44874 | 78.5501 ± 0.565218 ‖ 78.4412 ± 0.776610 | 2714.51 ± 31.7236 ‖ 2986.84 ± 25.9366 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P wm_pretrain [W] | P rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|---|
| numq3 | 158.787 ± 4.87668 ‖ 167.823 ± 1.25585 | 205.310 ± 0.799854 ‖ 208.207 ± 0.251453 | 163.566 ± 1.77338 ‖ 165.851 ± 1.78693 | 182.219 ± 1.35524 ‖ 186.452 ± 1.09874 | 172.533 ± 1.15428 ‖ 177.442 ± 0.912832 | 1.25530 ± 0.0106508 ‖ 1.25550 ± 0.0133361 |
| base | 169.629 ± 8.02768 ‖ 178.038 ± 3.94655 | 210.809 ± 0.409232 ‖ 213.081 ± 0.492626 | 173.669 ± 2.42160 ‖ 180.854 ± 2.53444 | 189.505 ± 1.85208 ‖ 196.083 ± 1.65782 | 180.906 ± 1.57862 ‖ 187.594 ± 1.50778 | 1.21402 ± 0.0150018 ‖ 1.17835 ± 0.0143448 |
| numq7 | 182.972 ± 1.39727 ‖ 178.477 ± 7.35435 | 212.701 ± 0.388074 ‖ 212.323 ± 0.401162 | 180.377 ± 2.39654 ‖ 177.923 ± 1.26569 | 193.996 ± 1.64117 ‖ 193.160 ± 0.896639 | 186.103 ± 1.50874 ‖ 185.980 ± 0.786695 | 1.17934 ± 0.0135158 ‖ 1.19339 ± 0.00886207 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | wm_pretrain [GFLOP/s] | rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|---|
| numq3 | 2695.69 ± 190.817 ‖ 3056.56 ± 13.9816 | 4434.82 ± 13.9150 ‖ 4579.61 ± 1.97990 | 2722.83 ± 64.8917 ‖ 2818.90 ± 62.5405 | 3492.30 ± 45.7117 ‖ 3678.86 ± 39.1882 | 2790.53 ± 82.7038 ‖ 2984.30 ± 80.1035 | 3703.16 ± 71.2148 ‖ 3730.97 ± 72.3819 | 98.0512 ± 0.103462 ‖ 97.7331 ± 0.0809525 | 97.9756–98.2294 ‖ 97.6397–97.8290 |
| base | 3125.33 ± 284.681 ‖ 3480.44 ± 107.856 | 4577.45 ± 11.3509 ‖ 4702.40 ± 2.00349 | 3116.78 ± 80.9857 ‖ 3372.06 ± 73.4443 | 3744.24 ± 62.5860 ‖ 4006.27 ± 45.8682 | 3051.81 ± 94.0394 ‖ 3431.61 ± 92.7349 | 4580.28 ± 103.436 ‖ 4863.43 ± 97.8338 | 98.4963 ± 0.0380504 ‖ 98.0101 ± 0.0479152 | 98.4516–98.5420 ‖ 97.9500–98.0501 |
| numq7 | 3718.85 ± 21.6471 ‖ 3648.06 ± 260.340 | 4702.17 ± 3.87294 ‖ 4785.52 ± 2.24616 | 3437.90 ± 72.6210 ‖ 3420.09 ± 48.9465 | 3975.19 ± 46.7699 ‖ 4030.12 ± 35.1815 | 3300.68 ± 82.8698 ‖ 3342.65 ± 55.3062 | 5305.84 ± 102.673 ‖ 5247.84 ± 64.1629 | 98.4615 ± 0.0337682 ‖ 98.1877 ± 0.0290698 | 98.4254–98.5082 ‖ 98.1413–98.2183 |

**Hardware composition of the measured `world_model_pretrain` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of wm_pretrain energy [%] | GPU share of wm_pretrain energy [%] | RAM share of wm_pretrain energy [%] | wm_pretrain energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| numq3 | 15.7406 ± 0.688074 ‖ 16.5665 ± 0.572266 | 71.6547 ± 0.537476 ‖ 71.5159 ± 0.486360 | 12.6047 ± 0.383032 ‖ 11.9177 ± 0.0885634 | 0/5 ‖ 0/5 |
| base | 15.7920 ± 0.828654 ‖ 16.2271 ± 0.796714 | 72.3968 ± 0.418271 ‖ 72.5349 ± 0.555025 | 11.8112 ± 0.553187 ‖ 11.2380 ± 0.256291 | 0/5 ‖ 0/5 |
| numq7 | 15.6960 ± 0.429095 ‖ 15.1887 ± 0.595932 | 73.3729 ± 0.363507 ‖ 73.5894 ± 0.235005 | 10.9311 ± 0.0835391 ‖ 11.2220 ± 0.489320 | 0/5 ‖ 0/5 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| numq3 | 16.3364 ± 0.151483 ‖ 16.3472 ± 0.0435760 | 73.9226 ± 0.115226 ‖ 74.0474 ± 0.0373318 | 9.74097 ± 0.0380730 ‖ 9.60541 ± 0.0116126 | 0/500 ‖ 0/500 |
| base | 16.1063 ± 0.0490408 ‖ 16.1563 ± 0.0665349 | 74.4068 ± 0.0597546 ‖ 74.4579 ± 0.0844652 | 9.48688 ± 0.0184493 ‖ 9.38583 ± 0.0217202 | 0/500 ‖ 0/500 |
| numq7 | 16.1782 ± 0.0603540 ‖ 16.0166 ± 0.0529340 | 74.4193 ± 0.0742533 ‖ 74.5641 ± 0.0396521 | 9.40253 ± 0.0171797 ‖ 9.41932 ± 0.0178123 | 0/500 ‖ 0/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| numq3 | 16.7568 ± 0.106945 ‖ 16.7037 ± 0.130436 | 71.0152 ± 0.0668667 ‖ 71.2369 ± 0.0359677 | 12.2280 ± 0.133355 ‖ 12.0595 ± 0.129435 | 0/500 ‖ 0/500 |
| base | 16.0554 ± 0.164318 ‖ 16.4391 ± 0.0907562 | 72.4271 ± 0.138522 ‖ 72.5011 ± 0.0662238 | 11.5175 ± 0.161250 ‖ 11.0599 ± 0.154272 | 0/500 ‖ 0/500 |
| numq7 | 15.7276 ± 0.0947468 ‖ 15.5901 ± 0.0675813 | 73.1834 ± 0.0977514 ‖ 73.1691 ± 0.0383091 | 11.0890 ± 0.146264 ‖ 11.2409 ± 0.0793988 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | wm_pretrain [%] | rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|---|
| numq3 | 42.0145 ± 1.37315 ‖ 40.0782 ± 0.346772 | 32.4685 ± 0.168749 ‖ 32.3035 ± 0.189464 | 40.7571 ± 0.320813 ‖ 40.5579 ± 0.587579 | 36.5836 ± 0.197845 ‖ 36.0740 ± 0.362563 | 66.6609 ± 0.414287 ‖ 67.2579 ± 0.370175 |
| base | 40.3342 ± 1.96137 ‖ 38.9192 ± 0.964806 | 32.3962 ± 0.250611 ‖ 32.5058 ± 0.359148 | 39.3313 ± 0.687072 ‖ 38.2995 ± 0.220941 | 36.0413 ± 0.493229 ‖ 35.3234 ± 0.224873 | 68.2939 ± 0.475666 ‖ 69.2642 ± 0.847964 |
| numq7 | 36.7067 ± 0.553675 ‖ 37.6709 ± 1.72958 | 31.5745 ± 0.377054 ‖ 31.6183 ± 0.214862 | 37.2332 ± 0.102557 ‖ 37.7338 ± 0.464376 | 34.6184 ± 0.194704 ‖ 34.7561 ± 0.351825 | 67.1604 ± 0.914333 ‖ 67.1333 ± 0.544806 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E wm_pretrain | E rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|---|
| numq3 | -23.6070 ± 5.83422 ‖ -22.7111 ± 0.791105 | -14.8837 ± 0.117093 ‖ -12.3972 ± 0.242423 | -5.40772 ± 1.10171 ‖ -6.37378 ± 2.25149 | -25.5830 ± 1.96772 ‖ -21.3073 ± 1.67513 | -21.0479 ± 0.921688 ‖ -19.0954 ± 0.843473 | -40.5582 ± 0.512400 ‖ -42.9477 ± 0.813174 | -24.2193 ± 1.50095 ‖ -20.9839 ± 1.27405 | -19.7457 ± 0.878416 ‖ -16.5915 ± 0.601305 |
| numq7 | 17.4074 ± 5.33576 ‖ 22.5858 ± 2.94168 | 13.2755 ± 0.373249 ‖ 10.3373 ± 0.373790 | 5.46498 ± 2.40075 ‖ -0.0295666 ± 3.66020 | 23.0804 ± 3.38027 ‖ 26.6115 ± 1.08838 | 18.8743 ± 1.66999 ‖ 20.5948 ± 0.521163 | 43.7050 ± 3.39779 ‖ 35.3368 ± 2.59238 | 22.1535 ± 2.40196 ‖ 24.1330 ± 0.674714 | 17.7833 ± 1.24788 ‖ 16.9772 ± 0.350175 |

Duration:

| config | D wm_pretrain | D rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|---|
| numq3 | -17.9213 ± 12.0020 ‖ -17.9995 ± 2.26048 | -12.6027 ± 0.396474 ‖ -10.3469 ± 0.0629314 | -19.5078 ± 3.00716 ‖ -13.8091 ± 2.88207 | -16.5241 ± 1.91094 ‖ -12.2757 ± 1.51242 | -0.0126916 ± 1.90554 ‖ 1.80053 ± 2.27209 | -21.3044 ± 3.50599 ‖ -14.3953 ± 3.40140 | -16.5265 ± 2.51148 ‖ -12.0035 ± 2.53375 | -37.1706 ± 0.890551 ‖ -37.9624 ± 1.05002 |
| numq7 | 8.98190 ± 9.41830 ‖ 22.4148 ± 5.52853 | 12.2680 ± 0.325529 ‖ 10.7306 ± 0.0772455 | 17.6879 ± 5.29058 ‖ 26.1836 ± 2.10335 | 15.0848 ± 3.11842 ‖ 18.7478 ± 0.960433 | 1.48938 ± 1.05747 ‖ 1.76842 ± 2.63804 | 18.5605 ± 6.30157 ‖ 28.9427 ± 2.74263 | 14.4783 ± 4.61722 ‖ 22.8089 ± 1.99750 | 38.2820 ± 0.878239 ‖ 37.7942 ± 1.46553 |

Mean power:

| config | P wm_pretrain | P rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|---|
| numq3 | -6.15551 ± 6.64728 ‖ -5.70570 ± 1.87926 | -2.60854 ± 0.387300 ‖ -2.28680 ± 0.325828 | -5.80226 ± 1.71419 ‖ -8.28016 ± 1.69964 | -3.83806 ± 1.16454 ‖ -4.90541 ± 1.07713 | -4.62278 ± 1.01997 ‖ -5.40615 ± 0.979993 |
| numq7 | 8.03105 ± 4.37326 ‖ 0.212894 ± 2.05308 | 0.897795 ± 0.345528 ‖ -0.355050 ± 0.376229 | 3.89282 ± 2.76740 ‖ -1.60895 ± 1.25716 | 2.38296 ± 1.78915 ‖ -1.48633 ± 0.787645 | 2.88402 ± 1.66320 ‖ -0.855979 ± 0.779244 |

Achieved throughput:

| config | wm_pretrain GFLOP/s | rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|---|
| numq3 | -12.8943 ± 12.6759 ‖ -12.1168 ± 2.51065 | -3.11528 ± 0.438679 ‖ -2.61113 ± 0.0683360 | -12.5853 ± 3.35589 ‖ -16.3670 ± 2.81348 | -6.70493 ± 2.16004 ‖ -8.16092 ± 1.59051 |
| numq7 | 19.7218 ± 10.0407 ‖ 4.70830 ± 4.47035 | 2.72522 ± 0.297801 ‖ 1.76762 ± 0.0710305 | 10.4053 ± 5.08074 ‖ 1.44632 ± 1.65447 | 6.20583 ± 2.91278 ‖ 0.600008 ± 0.805232 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | wm_pretrain | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|
| numq3 | 8.69601 ± 8.30124 ‖ 7.31699 ± 1.09846 | 0.523296 ± 0.138288 ‖ 0.332906 ± 0.277651 | 3.49827 ± 2.73669 ‖ 5.77055 ± 2.25154 | 17.0227 ± 1.36613 ‖ 19.8861 ± 1.24988 | 7.82487 ± 2.13563 ‖ 9.71525 ± 1.76904 | 3.09378 ± 1.12840 ‖ 3.55403 ± 0.746537 | -0.930350 ± 0.853999 ‖ -4.91283 ± 1.35529 |
| numq7 | -9.49079 ± 4.11333 ‖ -4.21547 ± 2.29853 | -1.77865 ± 0.323644 ‖ -2.08589 ± 0.331704 | -3.91728 ± 2.63881 ‖ 0.804860 ± 0.866543 | -10.3057 ± 1.26006 ‖ -8.99569 ± 0.393285 | -5.83198 ± 1.85167 ‖ -3.00648 ± 0.527199 | -3.57808 ± 1.02156 ‖ -2.07354 ± 0.293146 | 2.64642 ± 2.42699 ‖ -3.33082 ± 1.85170 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | wm_pretrain | rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|---|
| numq3 | 0.702813 ‖ 0.720193 | 0.846732 ‖ 0.873121 | 0.719016 ‖ 0.743995 | 0.674673 ‖ 0.674846 | 0.702813 ‖ 0.720193 | 0.778459 ‖ 0.805459 | 0.600000 ‖ 0.600000 |
| numq7 | 1.29719 ‖ 1.27981 | 1.15327 ‖ 1.12688 | 1.28098 ‖ 1.25601 | 1.32533 ‖ 1.32515 | 1.29719 ‖ 1.27981 | 1.22154 ‖ 1.19454 | 1.40000 ‖ 1.40000 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | wm_pretrain | rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|---|
| numq3 | -0.121752 ± 0.168379 ‖ -0.161652 ± 0.0190845 | 2.89940 ± 0.473302 ‖ 2.59681 ± 0.408122 | 0.394822 ± 0.0446425 ‖ 0.297138 ± 0.0612357 | -2.38764 ± 0.503439 ‖ -1.72126 ± 0.426744 | -0.203865 ± 0.0437360 ‖ -0.337796 ± 0.0397922 | -0.580967 ± 0.0350353 ‖ -0.673239 ± 0.0334346 | -2.77765 ± 0.452584 ‖ -2.43516 ± 0.409112 |
| numq7 | -0.0109191 ± 0.0856245 ‖ 0.105869 ± 0.0564471 | -1.82896 ± 0.539153 ‖ -2.93012 ± 0.155960 | -0.231667 ± 0.0682709 ‖ -0.356403 ± 0.0852497 | 1.46303 ± 0.579868 ‖ 2.49925 ± 0.239322 | 0.115865 ± 0.0632606 ‖ 0.347680 ± 0.0382637 | 0.492650 ± 0.0826671 ‖ 0.333716 ± 0.0400139 | 1.83988 ± 0.482440 ‖ 2.82425 ± 0.176762 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| numq3 | 1.14652 | 1.22544 | 3.49110 | rollout | 1.02323 | 1.00013 | 0.971324 |
| base | 1.10312 | 1.19066 | 3.79369 | rollout | 1.03471 | 0.970615 | 0.966971 |
| numq7 | 1.09564 | 1.15977 | 2.69254 | rollout | 0.995691 | 1.01199 | 0.982118 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | wm_pretrain | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|---|
| numq3 | 1.22544 (paired sd 0.00251674) | 1.01404 (paired sd 0.0386834) | 1.20360 (paired sd 0.0168105) | 1.08209 (paired sd 0.0348821) | 1.01228 (paired sd 0.0150705) | 1.00656 (paired sd 0.0154029) | 1.06780 (paired sd 0.0253075) | 1.14652 (paired sd 0.0120626) |
| base | 1.19066 (paired sd 0.00256929) | 0.999893 (paired sd 0.0403492) | 1.21664 (paired sd 0.0380107) | 1.02318 (paired sd 0.0144273) | 0.987821 (paired sd 0.00509623) | 1.04885 (paired sd 0.0179238) | 1.02401 (paired sd 0.0103813) | 1.10312 (paired sd 0.00661558) |
| numq7 | 1.15977 (paired sd 0.00174147) | 1.04582 (paired sd 0.0353961) | 1.15252 (paired sd 0.0106622) | 1.05286 (paired sd 0.0190599) | 1.00220 (paired sd 0.00892477) | 0.987733 (paired sd 0.0133636) | 1.04079 (paired sd 0.0141832) | 1.09564 (paired sd 0.00843778) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | wm_pretrain | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|---|
| numq3 | -0.266828 ± 0.0941612 (5/5 same sign) | 3.49110 ± 0.566124 (5/5 same sign) | 0.129729 ± 0.0594691 (5/5 same sign) | -1.70496 ± 0.656788 (5/5 same sign) | -1.44643 ± 0.0521708 (5/5 same sign) | -0.202606 ± 0.0392552 (5/5 same sign) |
| base | -0.226928 ± 0.0834794 (5/5 same sign) | 3.79369 ± 0.291936 (5/5 same sign) | 0.227412 ± 0.0776402 (5/5 same sign) | -2.37134 ± 0.277015 (5/5 same sign) | -1.31250 ± 0.0334668 (5/5 same sign) | -0.110334 ± 0.0379228 (5/5 same sign) |
| numq7 | -0.110140 ± 0.0673194 (4/5 same sign) | 2.69254 ± 0.332573 (5/5 same sign) | 0.102676 ± 0.0288426 (5/5 same sign) | -1.33512 ± 0.355484 (5/5 same sign) | -1.08069 ± 0.0256388 (5/5 same sign) | -0.269268 ± 0.0499164 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| numq3 | mean Δ 3.49110; 2/32 patterns; p = 0.0625000 | mean Δ 52569.7; 2/32 patterns; p = 0.0625000 |
| base | mean Δ 3.79369; 2/32 patterns; p = 0.0625000 | mean Δ 46104.4; 2/32 patterns; p = 0.0625000 |
| numq7 | mean Δ 2.69254; 2/32 patterns; p = 0.0625000 | mean Δ 50359.2; 2/32 patterns; p = 0.0625000 |


## 4.2 Sweep: horizon — horizon ∈ {1,3,5} (MPPI planning horizon / training sequence length)

Configurations: `hor1` (horizon 1), `base` (paper hyperparameters: num_q 5, horizon 3, 512 samples (24 policy-seeded), 6 iterations, 64 elites, batch 256, UTD 1; Ant: episodic = true (tdmpc2_ant.json)), `hor5` (horizon 5). Baseline for the response tables: `base` of the same environment (shared by both sweeps).

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | wm_pretrain (measured) | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|---|
| hor1 | 6689.99 ± 157.394 ‖ 7186.28 ± 350.487 | 135177 ± 123.072 ‖ 149441 ± 446.557 | 7995.23 ± 80.6870 ‖ 8883.03 ± 82.4520 | 79627.4 ± 1269.38 ‖ 86420.4 ± 1527.28 | 43017.3 ± 582.318 ‖ 42981.9 ± 569.751 | 8179.20 ± 87.3415 ‖ 8317.82 ± 132.891 | 138819 ± 1761.50 ‖ 146603 ± 2055.04 | 280686 ± 1782.40 ‖ 303231 ± 2581.60 |
| base | 10863.4 ± 469.216 ‖ 10862.2 ± 103.363 | 213811 ± 333.591 ‖ 254576 ± 533.613 | 9884.04 ± 126.916 ‖ 12025.3 ± 365.269 | 146372 ± 2470.48 ‖ 149766 ± 1894.13 | 56140.9 ± 469.436 ‖ 55457.1 ± 325.155 | 10015.4 ± 109.750 ‖ 10504.6 ± 162.674 | 222413 ± 2740.99 ‖ 227753 ± 1855.54 | 447087 ± 3184.37 ‖ 493191 ± 1781.98 |
| hor5 | 12794.6 ± 386.935 ‖ 13288.9 ± 75.3553 | 288267 ± 628.231 ‖ 354226 ± 536.601 | 10388.3 ± 150.810 ‖ 12840.2 ± 209.424 | 190712 ± 3717.92 ‖ 200510 ± 3915.59 | 58927.7 ± 495.741 ‖ 59008.1 ± 427.654 | 10191.1 ± 216.009 ‖ 10398.8 ± 107.074 | 270219 ± 3881.35 ‖ 282757 ± 4170.31 | 571281 ± 4038.94 ‖ 650272 ± 4614.54 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| hor1 | 0.0779685 ± 4.95112e-04 ‖ 0.0842308 ± 7.17110e-04 | 17.4149 ± 3.26841 ‖ 76.3985 ± 3.67739 | 5803.69 ± 87.5186 ‖ 5855.32 ± 60.0586 | 6143.36 ± 49.4024 ‖ 6164.39 ± 74.9685 |
| base | 0.124191 ± 8.84546e-04 ‖ 0.136998 ± 4.94994e-04 | 17.2086 ± 2.57921 ‖ 76.9060 ± 0.211427 | 5803.46 ± 105.464 ‖ 5868.73 ± 108.546 | 6489.87 ± 60.8709 ‖ 6599.26 ± 116.131 |
| hor5 | 0.158689 ± 0.00112193 ‖ 0.180631 ± 0.00128182 | 18.3402 ± 3.07226 ‖ 76.8996 ± 3.36638 | 5800.48 ± 40.1569 ‖ 5763.86 ± 82.1248 | 6461.93 ± 44.4108 ‖ 6455.61 ± 99.5997 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | wm_pretrain (measured) | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|---|
| hor1 | 2.38348 ± 0.0561998 ‖ 2.36965 ± 0.107267 | 48.1611 ± 0.307118 ‖ 49.2851 ± 0.311920 | 2.84867 ± 0.0450265 ‖ 2.92976 ± 0.0481941 | 28.3674 ± 0.274276 ‖ 28.4981 ± 0.263893 | 15.3252 ± 0.113459 ‖ 14.1743 ± 0.0794384 | 2.91411 ± 0.0378623 ‖ 2.74313 ± 0.0426478 | 49.4554 ± 0.315870 ‖ 48.3453 ± 0.270420 |
| base | 2.42935 ± 0.0889482 ‖ 2.20242 ± 0.0166962 | 47.8250 ± 0.341328 ‖ 51.6187 ± 0.227199 | 2.21099 ± 0.0427318 ‖ 2.43841 ± 0.0791294 | 32.7373 ± 0.323777 ‖ 30.3660 ± 0.288290 | 12.5570 ± 0.0391886 ‖ 11.2445 ± 0.0318135 | 2.24032 ± 0.0364381 ‖ 2.12999 ± 0.0362218 | 49.7456 ± 0.269537 ‖ 46.1789 ± 0.221384 |
| hor5 | 2.23946 ± 0.0573814 ‖ 2.04364 ± 0.0130224 | 50.4618 ± 0.381875 ‖ 54.4753 ± 0.324256 | 1.81860 ± 0.0369623 ‖ 1.97474 ± 0.0398161 | 33.3811 ± 0.427117 ‖ 30.8327 ± 0.385617 | 10.3149 ± 0.0187651 ‖ 9.07436 ± 0.0123377 | 1.78415 ± 0.0492885 ‖ 1.59926 ± 0.0240910 | 47.2988 ± 0.358652 ‖ 43.4811 ± 0.333282 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| hor1 | 5.76072 ± 0.127205 ‖ 6.06064 ± 0.131559 | 57.3587 ± 0.190958 ‖ 58.9461 ± 0.216198 | 30.9878 ± 0.0552590 ‖ 29.3189 ± 0.0780737 | 5.89285 ± 0.104997 ‖ 5.67435 ± 0.106240 |
| base | 4.44503 ± 0.107498 ‖ 5.28097 ± 0.192422 | 65.8082 ± 0.301085 ‖ 65.7562 ± 0.322405 | 25.2428 ± 0.118049 ‖ 24.3501 ± 0.0663820 | 4.50396 ± 0.0963213 ‖ 4.61277 ± 0.0955640 |
| hor5 | 3.84557 ± 0.107262 ‖ 4.54225 ± 0.120249 | 70.5727 ± 0.370270 ‖ 70.9086 ± 0.346737 | 21.8089 ± 0.132133 ‖ 20.8707 ± 0.160203 | 3.77289 ± 0.133596 ‖ 3.67853 ± 0.0810176 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | wm_pretrain (measured) [J/FLOP] | rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|---|
| hor1 | 8.43137e-11 ± 1.98363e-12 ‖ 8.58503e-11 ± 4.18706e-12 | 4.58503e-11 ± 4.17444e-14 ‖ 4.56122e-11 ± 1.36297e-13 | 9.28217e-11 ± 1.47972e-12 ‖ 9.16062e-11 ± 1.61893e-12 | 5.90026e-11 ± 7.98709e-13 ‖ 5.88187e-11 ± 7.79677e-13 | 8.74766e-11 ± 1.11001e-12 ‖ 8.75691e-11 ± 1.22752e-12 | 6.08270e-11 ± 3.86261e-13 ‖ 6.02342e-11 ± 5.12812e-13 | 14.0477 ± 0.150008 ‖ 14.2607 ± 0.227839 |
| base | 5.44456e-11 ± 2.35163e-12 ‖ 5.11657e-11 ± 4.86883e-13 | 4.60540e-11 ± 7.18542e-14 ‖ 4.53132e-11 ± 9.49803e-14 | 5.77998e-11 ± 9.75548e-13 ‖ 5.37874e-11 ± 6.80265e-13 | 3.85014e-11 ± 3.21940e-13 ‖ 3.79452e-11 ± 2.22479e-13 | 5.57348e-11 ± 6.86870e-13 ‖ 5.36406e-11 ± 4.37019e-13 | 5.06173e-11 ± 3.60521e-13 ‖ 4.89454e-11 ± 1.76848e-13 | 17.2013 ± 0.188494 ‖ 18.0100 ± 0.278901 |
| hor5 | 4.00197e-11 ± 1.21027e-12 ‖ 3.89838e-11 ± 2.21059e-13 | 4.54895e-11 ± 9.91372e-14 ‖ 4.45011e-11 ± 6.74126e-14 | 4.53327e-11 ± 8.83755e-13 ‖ 4.33497e-11 ± 8.46538e-13 | 2.69418e-11 ± 2.26653e-13 ‖ 2.69166e-11 ± 1.95075e-13 | 4.22602e-11 ± 6.07013e-13 ‖ 4.14742e-11 ± 6.11691e-13 | 4.37734e-11 ± 3.09476e-13 ‖ 4.30117e-11 ± 3.05225e-13 | 17.5030 ± 0.370993 ‖ 17.8286 ± 0.183576 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | wm_pretrain | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|
| hor1 | 1.01822 | 0.994807 | 0.986905 | 0.996884 | 1.00106 | 0.990255 | 1.01516 |
| base | 0.939758 | 0.983914 | 0.930582 | 0.985553 | 0.962426 | 0.966971 | 1.04701 |
| hor5 | 0.974116 | 0.978270 | 0.956257 | 0.999065 | 0.981399 | 0.982599 | 1.01860 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | wm_pretrain (measured) duration [s] | rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|---|
| hor1 | 47.4017 ± 2.31240 ‖ 51.0478 ± 3.59493 | 664.734 ± 1.16119 ‖ 736.610 ± 3.03592 | 986.511 ± 22.2159 ‖ 1036.23 ± 20.3226 | 55.7664 ± 0.163642 ‖ 61.3276 ± 0.173017 | 555.515 ± 14.5429 ‖ 596.716 ± 14.0683 | 300.102 ± 7.17995 ‖ 296.777 ± 5.80077 | 57.0515 ± 0.575966 ‖ 57.4283 ± 0.989147 | 1698.65 ± 24.1941 ‖ 1823.89 ± 24.4246 |
| base | 64.2556 ± 5.67548 ‖ 61.0457 ± 1.97208 | 1014.24 ± 2.51688 ‖ 1194.74 ± 0.509042 | 1281.04 ± 33.4115 ‖ 1259.62 ± 27.2265 | 56.0595 ± 0.467656 ‖ 65.1598 ± 1.28392 | 830.437 ± 25.6873 ‖ 811.869 ± 21.7115 | 318.484 ± 7.23155 ‖ 300.605 ± 6.00130 | 56.8054 ± 0.442010 ‖ 56.9297 ± 0.660151 | 2359.54 ± 39.4543 ‖ 2515.40 ± 28.7095 |
| hor5 | 72.7969 ± 4.38894 ‖ 73.6837 ± 1.56952 | 1355.94 ± 1.14845 ‖ 1654.50 ± 3.51327 | 1519.90 ± 47.0919 ‖ 1574.53 ± 43.3395 | 57.6250 ± 0.319134 ‖ 70.2982 ± 0.863095 | 1058.39 ± 38.6466 ‖ 1098.12 ± 36.1030 | 326.978 ± 8.33587 ‖ 323.122 ± 6.58507 | 56.5256 ± 0.316343 ‖ 56.9343 ± 0.526584 | 2948.65 ± 50.1764 ‖ 3302.71 ± 46.8295 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P wm_pretrain [W] | P rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|---|
| hor1 | 141.285 ± 4.10729 ‖ 140.950 ± 3.34427 | 203.356 ± 0.329749 ‖ 202.878 ± 0.415370 | 140.743 ± 1.46186 ‖ 141.490 ± 0.929570 | 165.257 ± 1.36907 ‖ 166.264 ± 0.933347 | 155.776 ± 1.14949 ‖ 157.310 ± 0.709640 | 1.44498 ± 0.0131457 ‖ 1.43391 ± 0.00776628 |
| base | 169.629 ± 8.02768 ‖ 178.038 ± 3.94655 | 210.809 ± 0.409232 ‖ 213.081 ± 0.492626 | 173.669 ± 2.42160 ‖ 180.854 ± 2.53444 | 189.505 ± 1.85208 ‖ 196.083 ± 1.65782 | 180.906 ± 1.57862 ‖ 187.594 ± 1.50778 | 1.21402 ± 0.0150018 ‖ 1.17835 ± 0.0143448 |
| hor5 | 175.996 ± 4.96961 ‖ 180.399 ± 2.84966 | 212.595 ± 0.448947 ‖ 214.099 ± 0.265002 | 177.863 ± 3.09378 ‖ 179.633 ± 2.31860 | 193.771 ± 2.06202 ‖ 196.907 ± 1.42589 | 186.535 ± 1.83477 ‖ 190.221 ± 1.24983 | 1.19554 ± 0.0190860 ‖ 1.19202 ± 0.0143894 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | wm_pretrain [GFLOP/s] | rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|---|
| hor1 | 1677.10 ± 81.7635 ‖ 1646.21 ± 114.222 | 4435.22 ± 7.73714 ‖ 4447.93 ± 18.2851 | 1609.28 ± 36.3898 ‖ 1616.10 ± 31.5607 | 2717.02 ± 38.8391 ‖ 2760.54 ± 36.9300 | 1545.10 ± 40.6231 ‖ 1581.67 ± 37.0607 | 2430.54 ± 58.2359 ‖ 2463.04 ± 47.9068 | 98.1671 ± 0.0437821 ‖ 97.6849 ± 0.0354606 | 98.1161–98.2161 ‖ 97.6551–97.7278 |
| base | 3125.33 ± 284.681 ‖ 3480.44 ± 107.856 | 4577.45 ± 11.3509 ‖ 4702.40 ± 2.00349 | 3116.78 ± 80.9857 ‖ 3372.06 ± 73.4443 | 3744.24 ± 62.5860 ‖ 4006.27 ± 45.8682 | 3051.81 ± 94.0394 ‖ 3431.61 ± 92.7349 | 4580.28 ± 103.436 ‖ 4863.43 ± 97.8338 | 98.4963 ± 0.0380504 ‖ 98.0101 ± 0.0479152 | 98.4516–98.5420 ‖ 97.9500–98.0501 |
| hor5 | 4403.64 ± 245.779 ‖ 4627.98 ± 98.0294 | 4673.49 ± 3.95876 ‖ 4811.12 ± 10.1922 | 4210.26 ± 133.084 ‖ 4332.56 ± 117.442 | 4427.09 ± 75.9713 ‖ 4578.34 ± 64.2865 | 3979.21 ± 148.857 ‖ 4215.71 ± 136.159 | 6692.74 ± 173.445 ‖ 6786.85 ± 137.206 | 98.6577 ± 0.0409716 ‖ 98.3442 ± 0.0444994 | 98.5912–98.6962 ‖ 98.2919–98.4090 |

**Hardware composition of the measured `world_model_pretrain` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of wm_pretrain energy [%] | GPU share of wm_pretrain energy [%] | RAM share of wm_pretrain energy [%] | wm_pretrain energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| hor1 | 19.0006 ± 1.14976 ‖ 19.0954 ± 0.668266 | 66.8342 ± 0.739998 ‖ 66.7089 ± 0.538984 | 14.1652 ± 0.412123 ‖ 14.1957 ± 0.339016 | 0/5 ‖ 0/5 |
| base | 15.7920 ± 0.828654 ‖ 16.2271 ± 0.796714 | 72.3968 ± 0.418271 ‖ 72.5349 ± 0.555025 | 11.8112 ± 0.553187 ‖ 11.2380 ± 0.256291 | 0/5 ‖ 0/5 |
| hor5 | 15.1972 ± 0.195233 ‖ 15.9178 ± 0.663704 | 73.4314 ± 0.269618 ‖ 72.9936 ± 0.517540 | 11.3713 ± 0.332920 ‖ 11.0887 ± 0.175920 | 0/5 ‖ 0/5 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| hor1 | 16.4398 ± 0.0333438 ‖ 16.5632 ± 0.0973891 | 73.7258 ± 0.0304029 ‖ 73.5792 ± 0.0988778 | 9.83441 ± 0.0159228 ‖ 9.85761 ± 0.0201759 | 0/500 ‖ 0/500 |
| base | 16.1063 ± 0.0490408 ‖ 16.1563 ± 0.0665349 | 74.4068 ± 0.0597546 ‖ 74.4579 ± 0.0844652 | 9.48688 ± 0.0184493 ‖ 9.38583 ± 0.0217202 | 0/500 ‖ 0/500 |
| hor5 | 16.1512 ± 0.0315628 ‖ 16.0256 ± 0.0782498 | 74.4416 ± 0.0256104 ‖ 74.6332 ± 0.0709906 | 9.40728 ± 0.0198210 ‖ 9.34122 ± 0.0115665 | 0/500 ‖ 0/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| hor1 | 19.2625 ± 0.200539 ‖ 19.6543 ± 0.0728041 | 66.5268 ± 0.128347 ‖ 66.2108 ± 0.0875496 | 14.2107 ± 0.147574 ‖ 14.1349 ± 0.0931907 | 0/500 ‖ 0/500 |
| base | 16.0554 ± 0.164318 ‖ 16.4391 ± 0.0907562 | 72.4271 ± 0.138522 ‖ 72.5011 ± 0.0662238 | 11.5175 ± 0.161250 ‖ 11.0599 ± 0.154272 | 0/500 ‖ 0/500 |
| hor5 | 15.8551 ± 0.201170 ‖ 15.9348 ± 0.191498 | 72.8981 ± 0.0180693 ‖ 72.9303 ± 0.128915 | 11.2469 ± 0.194037 ‖ 11.1349 ± 0.144282 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | wm_pretrain [%] | rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|---|
| hor1 | 47.0066 ± 1.41662 ‖ 47.3954 ± 1.21880 | 32.6369 ± 0.326104 ‖ 32.9134 ± 0.328049 | 47.1584 ± 0.484065 ‖ 47.1962 ± 0.672161 | 40.1623 ± 0.392052 ‖ 40.1632 ± 0.540989 | 66.3694 ± 0.703295 ‖ 66.7732 ± 0.560691 |
| base | 40.3342 ± 1.96137 ‖ 38.9192 ± 0.964806 | 32.3962 ± 0.250611 ‖ 32.5058 ± 0.359148 | 39.3313 ± 0.687072 ‖ 38.2995 ± 0.220941 | 36.0413 ± 0.493229 ‖ 35.3234 ± 0.224873 | 68.2939 ± 0.475666 ‖ 69.2642 ± 0.847964 |
| hor5 | 38.7298 ± 1.03225 ‖ 37.6352 ± 0.654232 | 32.0431 ± 0.104530 ‖ 31.7064 ± 0.402064 | 38.3086 ± 0.609138 ‖ 37.7972 ± 0.824954 | 35.1586 ± 0.325568 ‖ 34.4772 ± 0.583460 | 68.1219 ± 0.189000 ‖ 67.8833 ± 0.883483 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E wm_pretrain | E rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|---|
| hor1 | -38.3260 ± 3.00130 ‖ -33.8535 ± 2.83711 | -36.7770 ± 0.0674851 ‖ -41.2978 ± 0.167634 | -19.1055 ± 0.655250 ‖ -26.0800 ± 2.19207 | -45.5983 ± 0.231874 ‖ -42.2823 ± 1.59276 | -23.3786 ± 0.439177 ‖ -22.4898 ± 1.37925 | -18.3313 ± 0.674229 ‖ -20.8083 ± 1.39218 | -37.5848 ± 0.211322 ‖ -35.6234 ± 1.30545 | -37.2185 ± 0.114087 ‖ -38.5149 ± 0.690309 |
| hor5 | 18.0382 ± 8.09522 ‖ 22.3522 ± 1.63756 | 34.8232 ± 0.194916 ‖ 39.1441 ± 0.422467 | 5.12065 ± 2.33852 ‖ 6.82442 ± 2.31724 | 30.3383 ± 4.06534 ‖ 33.8861 ± 2.26440 | 4.97339 ± 1.57240 ‖ 6.40461 ± 0.787939 | 1.77105 ± 2.81136 ‖ -0.996909 ± 1.10220 | 21.5179 ± 2.80806 ‖ 24.1540 ± 1.80897 | 27.7866 ± 1.60845 ‖ 31.8514 ± 1.06356 |

Duration:

| config | D wm_pretrain | D rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|---|
| hor1 | -25.8097 ± 6.90446 ‖ -16.3984 ± 4.83646 | -34.4598 ± 0.0928107 ‖ -38.3456 ± 0.261567 | -22.9839 ± 0.650999 ‖ -17.6918 ± 2.87057 | -28.0061 ± 0.429667 ‖ -27.4810 ± 1.45126 | -0.517363 ± 0.870898 ‖ -5.85395 ± 1.74472 | -33.0959 ± 0.659679 ‖ -26.4421 ± 3.17019 | -5.77214 ± 0.657538 ‖ -1.22998 ± 3.25901 | 0.433554 ± 0.704384 ‖ 0.875326 ± 1.23254 |
| hor5 | 14.3675 ± 16.2717 ‖ 20.8240 ± 5.28320 | 33.6914 ± 0.418716 ‖ 38.4815 ± 0.299978 | 18.7377 ± 5.58723 ‖ 25.0100 ± 2.71290 | 25.0074 ± 3.57700 ‖ 31.3036 ± 1.61389 | 2.80016 ± 1.24735 ‖ 7.92649 ± 2.85889 | 27.5885 ± 7.11361 ‖ 35.2716 ± 3.30771 | 2.72559 ± 4.05283 ‖ 7.50044 ± 1.67899 | -0.485327 ± 1.22714 ‖ 0.0215323 ± 1.68302 |

Mean power:

| config | P wm_pretrain | P rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|---|
| hor1 | -16.5847 ± 4.06011 ‖ -20.8187 ± 1.68938 | -3.53558 ± 0.0943938 ‖ -4.78787 ± 0.176739 | -18.9551 ± 0.495864 ‖ -21.7537 ± 1.19211 | -12.7943 ± 0.382268 ‖ -15.2031 ± 0.777188 | -13.8901 ± 0.332555 ‖ -16.1398 ± 0.675568 |
| hor5 | 4.01598 ± 7.20417 ‖ 1.37582 ± 3.17227 | 0.847262 ± 0.284030 ‖ 0.478337 ± 0.117014 | 2.43480 ± 2.49393 ‖ -0.671872 ± 0.802012 | 2.26082 ± 1.65059 ‖ 0.421699 ± 0.504229 | 3.11988 ± 1.50745 ‖ 1.40214 ± 0.461835 |

Achieved throughput:

| config | wm_pretrain GFLOP/s | rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|---|
| hor1 | -46.0314 ± 4.93979 ‖ -52.7136 ± 2.65375 | -3.10703 ± 0.137445 ‖ -5.41138 ± 0.399950 | -48.3622 ± 0.440604 ‖ -52.0493 ± 1.64490 | -27.4314 ± 0.435465 ‖ -31.0851 ± 1.36586 |
| hor5 | 42.1912 ± 18.3952 ‖ 33.1058 ± 5.99571 | 2.09881 ± 0.319942 ‖ 2.31206 ± 0.221217 | 35.1876 ± 6.40115 ‖ 28.4945 ± 2.79502 | 18.2753 ± 3.36901 ‖ 14.2830 ± 1.40784 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | wm_pretrain | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|
| hor1 | 55.0880 ± 7.54718 ‖ 67.7585 ± 7.19540 | -0.442204 ± 0.106269 ‖ 0.660092 ± 0.287453 | 60.5950 ± 0.684496 ‖ 70.3527 ± 4.70100 | 53.2428 ± 0.878355 ‖ 55.0205 ± 2.75849 | 56.9517 ± 0.531398 ‖ 63.2698 ± 3.31083 | 20.1710 ± 0.218375 ‖ 23.0673 ± 1.38171 | -18.3313 ± 0.674229 ‖ -20.8083 ± 1.39218 |
| hor5 | -26.3333 ± 5.05216 ‖ -23.8015 ± 1.01984 | -1.22566 ± 0.142800 ‖ -1.79179 ± 0.298178 | -21.5420 ± 2.44716 ‖ -19.4034 ± 1.36312 | -30.0177 ± 1.04826 ‖ -29.0636 ± 0.525293 | -24.1617 ± 1.75248 ‖ -22.6794 ± 1.12659 | -13.5154 ± 1.08858 ‖ -12.1222 ± 0.708852 | 1.77105 ± 2.81136 ‖ -0.996909 ± 1.10220 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | wm_pretrain | rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|---|
| hor1 | 0.397671 ‖ 0.394296 | 0.635038 ‖ 0.583172 | 0.338751 ‖ 0.338813 | 0.500000 ‖ 0.500000 | 0.397671 ‖ 0.394296 | 0.522435 ‖ 0.499605 | 1.00000 ‖ 1.00000 |
| hor5 | 1.60233 ‖ 1.60570 | 1.36496 ‖ 1.41683 | 1.66125 ‖ 1.66119 | 1.50000 ‖ 1.50000 | 1.60233 ‖ 1.60570 | 1.47756 ‖ 1.50039 | 1.00000 ‖ 1.00000 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | wm_pretrain | rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|---|
| hor1 | -0.0458717 ± 0.114314 ‖ 0.167224 ± 0.0952745 | 0.336096 ± 0.112897 ‖ -2.33361 ± 0.433180 | 0.637676 ± 0.0196677 ‖ 0.491354 ± 0.0981196 | -4.36990 ± 0.0937597 ‖ -1.86786 ± 0.491180 | 2.76821 ± 0.0975335 ‖ 2.92976 ± 0.0937668 | 0.673788 ± 0.0216022 ‖ 0.613140 ± 0.0564066 | -0.290224 ± 0.115805 ‖ 2.16639 ± 0.439905 |
| hor5 | -0.189888 ± 0.130556 ‖ -0.158780 ± 0.0223068 | 2.63678 ± 0.622441 ‖ 2.85661 ± 0.288846 | -0.392389 ± 0.0671735 ‖ -0.463670 ± 0.0469982 | 0.643725 ± 0.620518 ‖ 0.466696 ± 0.275522 | -2.24205 ± 0.0471715 ‖ -2.17013 ± 0.0307446 | -0.456172 ± 0.0733099 ‖ -0.530725 ± 0.0181881 | -2.44689 ± 0.515781 ‖ -2.69783 ± 0.286160 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| hor1 | 1.08032 | 1.10552 | 1.15095 | actor | 1.00610 | 0.992383 | 0.990255 |
| base | 1.10312 | 1.19066 | 3.79369 | rollout | 1.03471 | 0.970615 | 0.966971 |
| hor5 | 1.13827 | 1.22881 | 4.01352 | rollout | 1.01618 | 0.997156 | 0.982599 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | wm_pretrain | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|---|
| hor1 | 1.10552 (paired sd 0.00274674) | 1.07419 (paired sd 0.0284987) | 1.11104 (paired sd 0.0174388) | 1.08531 (paired sd 0.0281322) | 0.999178 (paired sd 0.0222234) | 1.01695 (paired sd 0.0222124) | 1.05607 (paired sd 0.0215215) | 1.08032 (paired sd 0.0115659) |
| base | 1.19066 (paired sd 0.00256929) | 0.999893 (paired sd 0.0403492) | 1.21664 (paired sd 0.0380107) | 1.02318 (paired sd 0.0144273) | 0.987821 (paired sd 0.00509623) | 1.04885 (paired sd 0.0179238) | 1.02401 (paired sd 0.0103813) | 1.10312 (paired sd 0.00661558) |
| hor5 | 1.22881 (paired sd 0.00372017) | 1.03863 (paired sd 0.0282333) | 1.23602 (paired sd 0.0262830) | 1.05137 (paired sd 0.0279211) | 1.00136 (paired sd 0.0103075) | 1.02039 (paired sd 0.0255410) | 1.04640 (paired sd 0.0204893) | 1.13827 (paired sd 0.0112956) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | wm_pretrain | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|---|
| hor1 | -0.0138317 ± 0.0651218 (3/5 same sign) | 1.12398 ± 0.461707 (5/5 same sign) | 0.0810904 ± 0.0752717 (5/5 same sign) | 0.130696 ± 0.439919 (3/5 same sign) | -1.15095 ± 0.179367 (5/5 same sign) | -0.170982 ± 0.0765251 (5/5 same sign) |
| base | -0.226928 ± 0.0834794 (5/5 same sign) | 3.79369 ± 0.291936 (5/5 same sign) | 0.227412 ± 0.0776402 (5/5 same sign) | -2.37134 ± 0.277015 (5/5 same sign) | -1.31250 ± 0.0334668 (5/5 same sign) | -0.110334 ± 0.0379228 (5/5 same sign) |
| hor5 | -0.195820 ± 0.0445457 (5/5 same sign) | 4.01352 ± 0.482329 (5/5 same sign) | 0.156132 ± 0.0558079 (5/5 same sign) | -2.54837 ± 0.545800 (5/5 same sign) | -1.24058 ± 0.0213732 (5/5 same sign) | -0.184888 ± 0.0565309 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| hor1 | mean Δ 1.12398; 2/32 patterns; p = 0.0625000 | mean Δ 22544.5; 2/32 patterns; p = 0.0625000 |
| base | mean Δ 3.79369; 2/32 patterns; p = 0.0625000 | mean Δ 46104.4; 2/32 patterns; p = 0.0625000 |
| hor5 | mean Δ 4.01352; 2/32 patterns; p = 0.0625000 | mean Δ 78991.3; 2/32 patterns; p = 0.0625000 |


## 5. Cross-configuration quantities

**5.1a Fixed-versus-marginal fit E = a + b·F** — ordinary least squares over per-run points (one point per run: x = the run's FLOPs of the segment, y = its gross energy of the same segment; n = 5 × number of configurations of the sweep), exactly as in Section 4.1 (`ols_section`). a = intercept [J], b = slope [pJ/FLOP], residual SE = sqrt(SSR/(n−2)) [J]. GU = `gradient_updates` (FLOPs = critic + actor); TOTAL = `TOTAL_MEASURED_TRAINING`. A sweep with fewer than 3 distinct per-configuration FLOP values is skipped:

| sweep | env | configurations | segment | n | distinct FLOP values | a [J] | b [pJ/FLOP] | R² | residual SE [J] |
|---|---|---|---|---|---|---|---|---|---|
| num_q | HalfCheetah-v5 | numq3, base, numq7 | GU | 15 | 3 | 47374.8 | 43.4729 | 0.996697 | 2602.95 |
| num_q | HalfCheetah-v5 | numq3, base, numq7 | TOTAL | 15 | 3 | 65490.3 | 42.8697 | 0.998013 | 3282.65 |
| num_q | Ant-v5 | numq3, base, numq7 | GU | 15 | 3 | 46502.2 | 43.2500 | 0.996497 | 2671.91 |
| num_q | Ant-v5 | numq3, base, numq7 | TOTAL | 15 | 3 | 68296.7 | 42.2302 | 0.999063 | 2223.59 |
| horizon | HalfCheetah-v5 | hor1, base, hor5 | GU | 15 | 3 | 101407 | 27.3338 | 0.973611 | 9486.64 |
| horizon | HalfCheetah-v5 | hor1, base, hor5 | TOTAL | 15 | 3 | 128772 | 34.4454 | 0.992464 | 11104.4 |
| horizon | Ant-v5 | hor1, base, hor5 | GU | 15 | 3 | 106645 | 26.4709 | 0.985752 | 7178.37 |
| horizon | Ant-v5 | hor1, base, hor5 | TOTAL | 15 | 3 | 135464 | 34.4140 | 0.996606 | 8881.29 |


**5.1b Pairwise ΔEnergy/ΔFLOPs** — `compute_delta_pairs()` imported from `flop_dashboard.py` and called unmodified (inputs: `per_run_energy_per_flop.csv` loaded with `load_per_run`, rows of `tdmpc2` with `included_in_cross_seed_avg`, `PER_RUN_DIMENSIONS` so the seed is held fixed; and `cross_seed_energy_per_flop.csv` with `DIMENSIONS`). Rule: pairs differ only in the X dimension; matmul/mixed_total rows only; pairs with |ΔF| < 1% of the reference FLOPs are dropped; pooled value = ΣΔE/ΣΔF over the used pairs. Reference = smallest X value (`first`) or the next-smaller one (`previous`). Pair list: `contexts/Section_4.4_data/tdmpc2_delta_pairs.csv`.

| sweep | reference | segment | X pairs (x←ref) | HC ΣΔE/ΣΔF [pJ/FLOP] | Ant ΣΔE/ΣΔF [pJ/FLOP] | pairs used HC / Ant | pairs dropped HC / Ant |
|---|---|---|---|---|---|---|---|
| num_q | first | rollout | 5←3, 7←3 | 43.1115 | 41.8232 | 10 / 10 | 0 / 0 |
| num_q | first | critic | 5←3, 7←3 | 50.9060 | 48.4936 | 10 / 10 | 0 / 0 |
| num_q | first | actor | 5←3, 7←3 | 24.0514 | 22.8689 | 10 / 10 | 0 / 0 |
| num_q | first | wm_pretrain | 5←3, 7←3 | 39.5791 | 41.4727 | 10 / 10 | 0 / 0 |
| num_q | first | TOTAL | 5←3, 7←3 | 43.6213 | 42.0690 | 10 / 10 | 0 / 0 |
| num_q | previous | rollout | 5←3, 7←5 | 42.3059 | 40.5965 | 10 / 10 | 0 / 0 |
| num_q | previous | critic | 5←3, 7←5 | 50.0260 | 50.3442 | 10 / 10 | 0 / 0 |
| num_q | previous | actor | 5←3, 7←5 | 23.6196 | 23.1598 | 10 / 10 | 0 / 0 |
| num_q | previous | wm_pretrain | 5←3, 7←5 | 37.5726 | 41.4395 | 10 / 10 | 0 / 0 |
| num_q | previous | TOTAL | 5←3, 7←5 | 42.8697 | 42.2302 | 10 / 10 | 0 / 0 |
| horizon | first | rollout | 3←1, 5←1 | 45.5866 | 44.1141 | 10 / 10 | 0 / 0 |
| horizon | first | critic | 3←1, 5←1 | 35.3986 | 32.1264 | 10 / 10 | 0 / 0 |
| horizon | first | actor | 3←1, 5←1 | 13.2743 | 13.0009 | 10 / 10 | 0 / 0 |
| horizon | first | wm_pretrain | 3←1, 5←1 | 28.5072 | 25.3486 | 10 / 10 | 0 / 0 |
| horizon | first | TOTAL | 3←1, 5←1 | 36.1131 | 35.5008 | 10 / 10 | 0 / 0 |
| horizon | previous | rollout | 3←1, 5←3 | 45.1757 | 43.7238 | 10 / 10 | 0 / 0 |
| horizon | previous | critic | 3←1, 5←3 | 33.1686 | 30.9856 | 10 / 10 | 0 / 0 |
| horizon | previous | actor | 3←1, 5←3 | 10.9114 | 10.9655 | 10 / 10 | 0 / 0 |
| horizon | previous | wm_pretrain | 3←1, 5←3 | 25.3977 | 23.7294 | 10 / 10 | 0 / 0 |
| horizon | previous | TOTAL | 3←1, 5←3 | 34.4454 | 34.4140 | 10 / 10 | 0 / 0 |

Per-run vs cross-seed pooled values: max relative difference 1.08e-15 over 40 comparable (sweep, reference, env, segment) cells. The cross-seed CSV has no `gradient_updates` row, so the dashboard rule yields no GU pooled value; the OLS slope b in 5.1a is the closest equivalent.

**5.2a Task durations and the 1 s polling interval** — all per-epoch measured tasks of all 5 seeds (100 epochs × 5 seeds = 500 tasks per task type; `measure_power_secs` = 1.0 s). Cell: median [min–max] of the single-task CodeCarbon `duration` [s]; and number of tasks shorter than the polling interval / total:

| config | rollout duration median [min–max] s | rollout tasks < 1 s | GU duration median [min–max] s | GU tasks < 1 s |
|---|---|---|---|---|
| base | 10.1329 [10.0925–10.3465] ‖ 11.9453 [11.9118–12.0112] | 0/500 ‖ 0/500 | 13.4125 [11.3696–13.9471] ‖ 12.1471 [11.7974–14.4448] | 0/500 ‖ 0/500 |
| numq3 | 8.85148 [8.81570–8.99909] ‖ 10.7075 [10.6666–10.7866] | 0/500 ‖ 0/500 | 10.8011 [9.27006–11.2762] ‖ 11.4275 [9.71975–11.8238] | 0/500 ‖ 0/500 |
| numq7 | 11.3808 [11.3456–11.4643] ‖ 13.2245 [13.1917–13.3148] | 0/500 ‖ 0/500 | 15.6314 [13.5539–16.4970] ‖ 16.7037 [14.0696–17.3394] | 0/500 ‖ 0/500 |
| hor1 | 6.64058 [6.61550–6.74109] ‖ 7.35793 [7.31237–7.49466] | 0/500 ‖ 0/500 | 10.2544 [8.84483–10.5759] ‖ 10.8236 [9.24957–11.3145] | 0/500 ‖ 0/500 |
| hor5 | 13.5547 [13.5138–13.6434] ‖ 16.5341 [16.4830–16.6775] | 0/500 ‖ 0/500 | 14.8439 [13.6694–16.8743] ‖ 14.9414 [14.3656–17.8022] | 0/500 ‖ 0/500 |

Across all 50 runs: `rollout`: 0 of 5000 tasks < 1.0 s (min/median/max 6.61550/11.0661/16.6775 s); `gradient_updates`: 0 of 5000 tasks < 1.0 s (min/median/max 8.84483/12.1990/17.8022 s). Single (non-per-epoch) measured tasks: `world_model_pretrain` duration 61.6892 ± 10.8395 s.
**5.2b Integration span versus task duration** — span = `ram_energy / ram_power` of the task row (CodeCarbon models RAM at constant power, so this is the time over which the task's energy was integrated); the cell is the ratio span/duration (1 = integrated over exactly the recorded duration):

| config | rollout integration span / duration: median [min–max] | GU integration span / duration: median [min–max] |
|---|---|---|
| base | 0.999961 [0.999790–1.00013] ‖ 0.999964 [0.999819–1.00011] | 0.999955 [0.999799–1.00023] ‖ 0.999953 [0.999808–1.00010] |
| numq3 | 0.999949 [0.999751–1.00014] ‖ 0.999956 [0.999796–1.00012] | 0.999943 [0.999754–1.00013] ‖ 0.999945 [0.999755–1.00011] |
| numq7 | 0.999960 [0.999804–1.00012] ‖ 0.999966 [0.999818–1.00012] | 0.999961 [0.999830–1.00008] ‖ 0.999963 [0.999841–1.00008] |
| hor1 | 0.999934 [0.999692–1.00019] ‖ 0.999940 [0.999706–1.00016] | 0.999941 [0.999740–1.00012] ‖ 0.999943 [0.999764–1.00012] |
| hor5 | 0.999966 [0.999841–1.00009] ‖ 0.999971 [0.999864–1.00008] | 0.999961 [0.999835–1.00009] ‖ 0.999961 [0.999848–1.00008] |

**5.2c Per-epoch stationarity** (per run: CV of the 100 per-epoch energies; ratio of the epoch-0 energy to the mean of epochs 1–99; relative difference of the mean duration of epochs 0–9 versus epochs 10–99; then mean ± sd over the 5 seeds; per-epoch values in the per-epoch CSV):

| config | rollout per-epoch energy CV within run [%] | GU per-epoch energy CV within run [%] | rollout energy epoch 0 / mean(epochs 1–99) | GU energy epoch 0 / mean(epochs 1–99) | rollout duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%] | GU duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%] |
|---|---|---|---|---|---|---|
| base | 0.591484 ± 0.145277 ‖ 0.513539 ± 0.0569401 | 3.09323 ± 0.459560 ‖ 3.10588 ± 0.883205 | 0.990950 ± 0.00206759 ‖ 0.988016 ± 0.00541730 | 0.992082 ± 0.0316564 ‖ 0.968009 ± 0.0237059 | -0.0936281 ± 0.0623362 ‖ -0.0842500 ± 0.0410284 | 1.58516 ± 6.07471 ‖ -5.19965 ± 2.03798 |
| numq3 | 0.504575 ± 0.0380515 ‖ 0.491112 ± 0.0915875 | 3.44329 ± 0.271618 ‖ 3.58325 ± 0.229269 | 0.999071 ± 0.00231611 ‖ 0.992385 ± 0.00474678 | 1.00277 ± 0.0315628 ‖ 0.940853 ± 0.00801103 | -0.0401160 ± 0.0341260 ‖ -0.138484 ± 0.0526364 | 0.138521 ± 5.69726 ‖ -4.63787 ± 5.13052 |
| numq7 | 0.466389 ± 0.0419338 ‖ 0.432432 ± 0.0402385 | 3.48064 ± 0.0466202 ‖ 3.30406 ± 0.122591 | 0.995323 ± 0.00436844 ‖ 0.992909 ± 0.00308580 | 0.942362 ± 0.00823309 ‖ 0.962209 ± 0.0316676 | -0.0428909 ± 0.0451906 ‖ -0.0987306 ± 0.0248813 | -5.13191 ± 6.15555 ‖ -1.05993 ± 7.02670 |
| hor1 | 0.617034 ± 0.0417424 ‖ 0.614366 ± 0.148933 | 3.59471 ± 0.383749 ‖ 3.74867 ± 0.202190 | 1.01027 ± 0.00483989 ‖ 1.00895 ± 0.00207024 | 0.981069 ± 0.0462484 ‖ 0.999314 ± 0.0395841 | 0.0357277 ± 0.0697935 ‖ -0.0219514 ± 0.0807204 | -2.07671 ± 6.65435 ‖ -4.83183 ± 4.71106 |
| hor5 | 0.484224 ± 0.0461222 ‖ 0.451387 ± 0.0682245 | 3.66168 ± 0.295303 ‖ 3.66124 ± 0.256879 | 0.991925 ± 0.00243375 ‖ 0.989423 ± 0.00503845 | 0.975396 ± 0.0358596 ‖ 0.961009 ± 0.0394972 | -0.0108610 ± 0.0322708 ‖ -0.0447417 ± 0.0484532 | -2.57158 ± 4.31455 ‖ 0.531066 ± 9.28234 |

**5.2d Between-seed variability and allocation coverage:**

| config | CV of TOTAL energy over seeds [%] | allocation coverage min–max [%] | CV of wm_pretrain energy over seeds [%] | CV of rollout energy over seeds [%] | CV of GU energy over seeds [%] |
|---|---|---|---|---|---|
| base | 0.712248 ‖ 0.361316 | 98.4516–98.5420 ‖ 97.9500–98.0501 | 4.31923 ‖ 0.951581 | 0.156022 ‖ 0.209609 | 1.23239 ‖ 0.814718 |
| numq3 | 0.608184 ‖ 0.496507 | 97.9756–98.2294 ‖ 97.6397–97.8290 | 3.93156 ‖ 0.507796 | 0.181591 ‖ 0.0950468 | 1.33317 ‖ 1.14808 |
| numq7 | 0.374926 ‖ 0.489445 | 98.4254–98.5082 ‖ 98.1413–98.2183 | 0.200344 ‖ 3.35364 | 0.199458 ‖ 0.196673 | 0.816086 ‖ 0.777203 |
| hor1 | 0.635015 ‖ 0.851363 | 98.1161–98.2161 ‖ 97.6551–97.7278 | 2.35268 ‖ 4.87717 | 0.0910451 ‖ 0.298818 | 1.26892 ‖ 1.40177 |
| hor5 | 0.706997 ‖ 0.709632 | 98.5912–98.6962 ‖ 98.2919–98.4090 | 3.02419 ‖ 0.567053 | 0.217934 ‖ 0.151485 | 1.43637 ‖ 1.47487 |

**5.2e Idle head versus tail** (as-recorded mean power = idle-task energy / its CodeCarbon `duration`):

| config | P_head [W] | P_tail [W] | P_tail − P_head [W] | \|P_tail − P_head\| / P_head [%] |
|---|---|---|---|---|
| base | 64.4808 ± 1.17170 ‖ 65.2061 ± 1.20606 | 72.1069 ± 0.676254 ‖ 73.3223 ± 1.29027 | 7.62614 ± 1.65993 ‖ 8.11617 ± 1.83374 | 11.8655 ± 2.78453 ‖ 12.4796 ± 2.97288 |
| numq3 | 64.0232 ± 0.789593 ‖ 64.7189 ± 0.683820 | 69.2987 ± 0.740169 ‖ 69.7969 ± 0.532555 | 5.27548 ± 1.28689 ‖ 5.07808 ± 0.976898 | 8.25782 ± 2.09280 ‖ 7.85785 ± 1.57649 |
| numq7 | 63.8453 ± 1.03990 ‖ 63.4093 ± 0.672048 | 70.4755 ± 0.974500 ‖ 70.8573 ± 0.838067 | 6.63025 ± 0.847395 ‖ 7.44802 ± 1.05866 | 10.3952 ± 1.40634 ‖ 11.7555 ± 1.73316 |
| hor1 | 64.4828 ± 0.972410 ‖ 65.0561 ± 0.667268 | 68.2560 ± 0.548449 ‖ 68.4903 ± 0.832781 | 3.77324 ± 0.717116 ‖ 3.43413 ± 1.00997 | 5.86371 ± 1.18545 ‖ 5.28643 ± 1.57557 |
| hor5 | 64.4473 ± 0.446172 ‖ 64.0404 ± 0.912396 | 71.7965 ± 0.493395 ‖ 71.7262 ± 1.10668 | 7.34919 ± 0.861470 ‖ 7.68579 ± 0.996104 | 11.4105 ± 1.40056 ‖ 12.0094 ± 1.62692 |

Over all 50 runs of TD-MPC2: |P_tail − P_head|/P_head (as recorded) median 10.0380 %, mean 9.71816 %, maximum 17.0340 % (run `results/tdmpc2/Ant-v5/seed_43611/20260910_182237`, config base, Ant-v5, seed 43611); tail > head in 50/50 runs. Window-corrected (energy ÷ integrated span, see `idle_power_analysis.py`): median 13.8251 %, maximum 21.0473 % (run `results/tdmpc2/Ant-v5/seed_43611/20260910_182237`). Median head integration span 93.0859 s vs recorded duration 90.0035 s; tail 90.0024 s vs 90.0036 s.
**5.2f Outlier flags** — rule: |x − median| > 3 × MAD (unscaled MAD, n = 5 per configuration), applied to the energy of every segment of the energy table (incl. GU and TOTAL). Flagged, **not excluded**: 55 of 400 (run, segment) cells; largest deviation 8.09005 % of the median (HalfCheetah-v5 base seed 43611 wm_pretrain: 10276.9 J vs median 11181.4 J). Full list: `contexts/Section_4.4_data/tdmpc2_mad_flags.csv`.
Flag count per configuration: Ant-v5/base: 5, Ant-v5/hor1: 3, Ant-v5/hor5: 5, Ant-v5/numq3: 8, Ant-v5/numq7: 6, HalfCheetah-v5/base: 4, HalfCheetah-v5/hor1: 2, HalfCheetah-v5/hor5: 11, HalfCheetah-v5/numq3: 6, HalfCheetah-v5/numq7: 5.

**5.3 Episodes and returns (sanity information only)** — definitions of `aggregate_results.py`: episodes = `training_metrics.json["episodes"]` with `phase == "train"` (warmup and the trailing partial `train_incomplete` episode excluded); final-10 %-mean return = mean of the last max(1, n_episodes // 10) of them; returns are those of the stochastic training policy (no evaluation episodes). Cell: completed training episodes per run (min–max over the 5 seeds); number of episodes in the 10 % window (min–max); mean ± sd over seeds of the window mean return:

| config | HalfCheetah-v5 | Ant-v5 |
|---|---|---|
| base | episodes 100–100; window 10–10; return 6458.18 ± 2418.85 | episodes 884–2201; window 88–220; return 149.614 ± 75.9608 |
| numq3 | episodes 100–100; window 10–10; return 7543.90 ± 1579.25 | episodes 1376–2556; window 137–255; return 125.794 ± 110.935 |
| numq7 | episodes 100–100; window 10–10; return 7755.66 ± 1710.33 | episodes 919–1560; window 91–156; return 356.184 ± 323.323 |
| hor1 | episodes 100–100; window 10–10; return 5790.94 ± 781.554 | episodes 440–536; window 44–53; return 438.532 ± 262.955 |
| hor5 | episodes 100–100; window 10–10; return 7443.13 ± 2990.02 | episodes 1044–1712; window 104–171; return 147.077 ± 126.925 |


## 6. Algorithm-specific items (TD-MPC2)

**6.1 Code facts** (`algorithms/tdmpc2.py`, `configs/config.py`, `configs/overrides/tdmpc2_*.json`; HEAD, identical at all run commits):


**(a) Parameter counts** of every component per environment and for num_q = 3, 5, 7: Section 2.3 (the termination classifier exists only for `episodic` = true, i.e. on the Ant runs).

**(b) Planner settings per configuration:** Section 2.4 (512 samples of which 24 are seeded from the policy prior, horizon 3 in the baseline, 6 iterations (`self.iterations`, line 454), 64 elites, temperature 0.5; the horizon sweep changes the horizon, the num_q sweep the Q-ensemble size; samples/iterations/elites are identical in all 50 runs).

**(c) What the `rollout` task contains** (`train()`, `TrackerTask(tracker, "rollout", epoch, …)`, line 766): per env step: `agent.act(obs, t0=is_t0, eval_mode=False)` (line 768) = full MPPI planning in latent space (`act`, line 486): encode the observation, seed `num_pi_trajs` trajectories from the policy prior (π → dynamics for horizon−1 steps), then `iterations` MPPI iterations each of which samples the remaining trajectories from the Gaussian search distribution, scores all 512 trajectories with `_estimate_value` (line 467: reward model + latent dynamics per horizon step, [termination classifier if episodic], terminal π and **average over all num_q Q-networks**), takes the top-`num_elites`, refits mean/std; then one trajectory is picked by a Gumbel-softmax sample over the elite scores, noise `std·N(0,1)` is added, the action is clipped and `.cpu().numpy()` (a device sync every env step). Then `env.step` (MuJoCo), `buffer.add` (host appends to the open episode), reward bookkeeping and, on episode end, `buffer.end_episode()` (line 733: `np.stack` of the whole episode into arrays, eviction of the oldest episodes if capacity is exceeded), the episode record, `env.reset()` and `buffer.start_episode`. Episode resets are therefore inside the `rollout` task. No evaluation episodes exist.

**(d) What `buffer_sample` does and what its timer covers** (`TDMPC2Agent.update`, lines 566–574): `_EpisodeSequenceBuffer.sample(batch_size, horizon)` (line 383): builds `eligible = [ep for ep in self._episodes if length ≥ horizon]` (a Python loop over **all stored closed episodes** on every call, line 384), weights ∝ (length − horizon + 1), `np.random.choice` of `batch_size` episodes, then a **host-side Python loop over the 256 batch elements**, each slicing `horizon + 1` observations and `horizon` actions/rewards/terminations out of the episode arrays into preallocated arrays (sequence length = horizon + 1 observations, horizon transitions); then four `torch.as_tensor(..., device=cuda)` host→device copies. The timer covers exactly the sampling and these four copies; the open (current) episode is never sampled. The cost per call grows with the number of stored episodes (HC: 5 warmup + 100 training episodes ≈ 105 at the end; Ant: more, Section 5.3).

**(e) What `critic_update`, `actor_update` and `target_update` contain here** (confirming the mapping in README/CLAUDE.md against the code):
- `critic_update` (lines 576–629) = the **world-model step**: no-grad real next-observation encodings `next_z = encode(obs[1:])`, policy-prior action at `next_z`, `target_q = min` over two randomly chosen target Q-networks (line 583: `td_targets = reward + discount·(1 − terminated)·target_q`, using the env's own `terminated` flag whether or not `episodic`); then the encoder on `obs[0]`, `horizon` unrolled latent-dynamics steps with the consistency (MSE) loss against `next_z` (ρ^t weighted), all num_q Q-networks and the reward model on the unrolled latents, soft two-hot cross-entropy reward and value losses, [termination BCE loss if `episodic`, line 614], one weighted loss, backward, gradient-norm clipping, Adam step (encoder LR × 0.3). Dropout 0.01 is active (model in `train()` mode).
- `actor_update` (lines 635–638) = `_update_pi` (line 538) on `zs.detach()`: Q-ensemble frozen, π on all (H+1) latents, `Q(…, "avg")` over all num_q networks, `RunningScale` update, entropy-regularised loss, backward, clip, π Adam step. It is the policy-prior step (TD-MPC2's closest analogue of SAC's actor update).
- `target_update` (lines 641–647) = Polyak averaging (τ = 0.01) of the Q-ensemble only (`qs` → `qs_target`): 2 elementwise ops per Q parameter, `gradient_update_target_elementwise_ops` = 2 × Σ params(`qs`).
- The mapping stated in the documents (critic = world-model step, actor = policy-prior step, target = Q-ensemble Polyak) **agrees with the code**. The encoder, dynamics, reward and termination networks are updated inside `critic_update` (the world-model optimiser), the policy prior inside `actor_update`.

**(f) Which FLOP quantities depend on `num_q` and `horizon`, and how** — from the exact analytic model of Section 2.6 (verified against every measured constant):
- *planning per call* (`rollout_per_env_step`): the planning cost is `enc + npi·[(H−1)(π+dyn)+π] + it·ns·[H(rew+dyn+term) + π + nq·Q]`: **linear in H** through the per-step reward + dynamics (+ termination) evaluations and the policy seeding; **linear in nq** through the terminal-value term `it·ns·nq·Q` (each Q-network has the same width as the reward/dynamics heads; the share of every term is in the decomposition table below);
- *world-model step* (`critic_update`): linear in H in all terms; linear in nq through the target-Q evaluation (all nq target nets, no-grad) and the Q forward/backward on the unrolled latents (3× forward cost);
- *policy-prior step* (`actor_update`): ∝ (H+1) (it runs on all H+1 latents) and linear in nq (all nets are evaluated for the average and back-propagated through);
- *Polyak operations*: ∝ nq (2 · Σ params(`qs`)), independent of H;
- the number of calls (n_env, n_upd, pretrain) does not depend on either parameter.
The per-call values and the factors against the baseline for every configuration are in Sections 2.5 and 4.1–4.2 (FLOP-factor tables).

Decomposition of one planning call (`rollout_per_env_step`) into the terms of the analytic model (FLOPs and share of the call):


| config | total | encode | pi_seed | plan_reward | plan_dynamics | plan_termination | plan_terminal_pi | plan_terminal_Q |
|---|---|---|---|---|---|---|---|---|
| HC-base | 4.64261e+10 | 270848 (5.83396e-04 %) | 1.52175e+08 (0.327778 %) | 1.06735e+10 (22.9902 %) | 1.45521e+10 (31.3447 %) | 0 (0 %) | 3.25897e+09 (7.01970 %) | 1.77891e+10 (38.3170 %) |
| HC-numq3 | 3.93105e+10 | 270848 (6.88997e-04 %) | 1.52175e+08 (0.387110 %) | 1.06735e+10 (27.1517 %) | 1.45521e+10 (37.0185 %) | 0 (0 %) | 3.25897e+09 (8.29035 %) | 1.06735e+10 (27.1517 %) |
| HC-numq7 | 5.35417e+10 | 270848 (5.05863e-04 %) | 1.52175e+08 (0.284217 %) | 1.06735e+10 (19.9348 %) | 1.45521e+10 (27.1791 %) | 0 (0 %) | 3.25897e+09 (6.08679 %) | 2.49047e+10 (46.5146 %) |
| HC-hor1 | 2.94823e+10 | 270848 (9.18679e-04 %) | 2.54607e+07 (0.0863593 %) | 3.55782e+09 (12.0676 %) | 4.85071e+09 (16.4529 %) | 0 (0 %) | 3.25897e+09 (11.0540 %) | 1.77891e+10 (60.3382 %) |
| HC-hor5 | 6.33699e+10 | 270848 (4.27408e-04 %) | 2.78888e+08 (0.440096 %) | 1.77891e+10 (28.0718 %) | 2.42536e+10 (38.2730 %) | 0 (0 %) | 3.25897e+09 (5.14278 %) | 1.77891e+10 (28.0718 %) |
| Ant-base | 5.61814e+10 | 315904 (5.62292e-04 %) | 1.52568e+08 (0.271563 %) | 1.06923e+10 (19.0318 %) | 1.45710e+10 (25.9356 %) | 9.67311e+09 (17.2176 %) | 3.27156e+09 (5.82320 %) | 1.78205e+10 (31.7196 %) |
| Ant-numq3 | 4.90532e+10 | 315904 (6.44003e-04 %) | 1.52568e+08 (0.311025 %) | 1.06923e+10 (21.7974 %) | 1.45710e+10 (29.7045 %) | 9.67311e+09 (19.7196 %) | 3.27156e+09 (6.66940 %) | 1.06923e+10 (21.7974 %) |
| Ant-numq7 | 6.33097e+10 | 315904 (4.98982e-04 %) | 1.52568e+08 (0.240987 %) | 1.06923e+10 (16.8889 %) | 1.45710e+10 (23.0155 %) | 9.67311e+09 (15.2790 %) | 3.27156e+09 (5.16755 %) | 2.49488e+10 (39.4075 %) |
| Ant-hor1 | 3.27635e+10 | 315904 (9.64196e-04 %) | 2.55590e+07 (0.0780108 %) | 3.56411e+09 (10.8783 %) | 4.85700e+09 (14.8245 %) | 3.22437e+09 (9.84136 %) | 3.27156e+09 (9.98538 %) | 1.78205e+10 (54.3915 %) |
| Ant-hor5 | 7.95994e+10 | 315904 (3.96867e-04 %) | 2.79577e+08 (0.351229 %) | 1.78205e+10 (22.3878 %) | 2.42850e+10 (30.5090 %) | 1.61219e+10 (20.2537 %) | 3.27156e+09 (4.11003 %) | 1.78205e+10 (22.3878 %) |


Factor of the per-call constants against the same environment's baseline (`flops_per_call.json`):


| config | rollout (planning) | critic (world-model step) | actor (policy-prior step) | update total | Polyak ops |
|---|---|---|---|---|---|
| HC-base | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| HC-numq3 | 0.846732 | 0.719016 | 0.674673 | 0.702813 | 0.600000 |
| HC-numq7 | 1.15327 | 1.28098 | 1.32533 | 1.29719 | 1.40000 |
| HC-hor1 | 0.635038 | 0.338751 | 0.500000 | 0.397671 | 1.00000 |
| HC-hor5 | 1.36496 | 1.66125 | 1.50000 | 1.60233 | 1.00000 |
| Ant-base | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| Ant-numq3 | 0.873121 | 0.743995 | 0.674846 | 0.720193 | 0.600000 |
| Ant-numq7 | 1.12688 | 1.25601 | 1.32515 | 1.27981 | 1.40000 |
| Ant-hor1 | 0.583172 | 0.338813 | 0.500000 | 0.394296 | 1.00000 |
| Ant-hor5 | 1.41683 | 1.66119 | 1.50000 | 1.60570 | 1.00000 |


**6.2 Call counts per run** (formulas and implementation in Section 2.7): rollout calls = 100000; update calls = `steps_per_epoch × updates_per_env_step × 100` = 100000 in every configuration (UTD 1; buffer_sample = critic = actor = target calls); pretraining updates = `warmup_steps` = 5000. Per configuration table: Section 2.7.

**6.3 `world_model_pretrain`** — a directly measured CodeCarbon task `world_model_pretrain_0`: a one-off burst of `warmup_steps` = 5000 `agent.update()` calls on the warmup data, right after warmup and before epoch 0.


| config | energy [J] | duration [s] | mean power [W] | FLOPs (5000 × update total) | per-update FLOPs | J/FLOP (ratio of means ± sd) | share of TOTAL [%] | updates |
|---|---|---|---|---|---|---|---|---|
| HC-base | 10863.4 ± 469.216 | 64.2556 ± 5.67548 | 169.629 ± 8.02768 | 1.99528e+14 | 3.99055e+10 | 5.44456e-11 ± 2.35163e-12 | 2.42935 ± 0.0889482 | 5000 |
| HC-numq3 | 8278.54 ± 325.475 | 52.2241 ± 3.59849 | 158.787 ± 4.87668 | 1.40231e+14 | 2.80461e+10 | 5.90352e-11 ± 2.32100e-12 | 2.30760 ± 0.0967264 | 5000 |
| HC-numq7 | 12734.4 ± 25.5126 | 69.5999 ± 0.405093 | 182.972 ± 1.39727 | 2.58825e+14 | 5.17649e+10 | 4.92010e-11 ± 9.85711e-14 | 2.41843 ± 0.00915078 | 5000 |
| HC-hor1 | 6689.99 ± 157.394 | 47.4017 ± 2.31240 | 141.285 ± 4.10729 | 7.93464e+13 | 1.58693e+10 | 8.43137e-11 ± 1.98363e-12 | 2.38348 ± 0.0561998 | 5000 |
| HC-hor5 | 12794.6 ± 386.935 | 72.7969 ± 4.38894 | 175.996 ± 4.96961 | 3.19709e+14 | 6.39418e+10 | 4.00197e-11 ± 1.21027e-12 | 2.23946 ± 0.0573814 | 5000 |
| Ant-base | 10862.2 ± 103.363 | 61.0457 ± 1.97208 | 178.038 ± 3.94655 | 2.12295e+14 | 4.24591e+10 | 5.11657e-11 ± 4.86883e-13 | 2.20242 ± 0.0166962 | 5000 |
| Ant-numq3 | 8394.74 ± 42.6281 | 50.0223 ± 0.229119 | 167.823 ± 1.25585 | 1.52894e+14 | 3.05787e+10 | 5.49058e-11 ± 2.78809e-13 | 2.04077 ± 0.0121289 | 5000 |
| Ant-numq7 | 13317.9 ± 446.634 | 74.8141 ± 5.90138 | 178.477 ± 7.35435 | 2.71697e+14 | 5.43394e+10 | 4.90174e-11 ± 1.64387e-12 | 2.30829 ± 0.0703969 | 5000 |
| Ant-hor1 | 7186.28 ± 350.487 | 51.0478 ± 3.59493 | 140.950 ± 3.34427 | 8.37072e+13 | 1.67414e+10 | 8.58503e-11 ± 4.18706e-12 | 2.36965 ± 0.107267 | 5000 |
| Ant-hor5 | 13288.9 ± 75.3553 | 73.6837 ± 1.56952 | 180.399 ± 2.84966 | 3.40884e+14 | 6.81767e+10 | 3.89838e-11 ± 2.21059e-13 | 2.04364 ± 0.0130224 | 5000 |


- *Timers:* the sub-timers (`buffer_sample`, `critic_update`, `actor_update`, `target_update`) of these 5000 updates are **discarded**: `train()` (lines 742–748) keeps only `info.total_loss` of each pretraining update and does not add the `UpdateInfo` times to `sub_time_totals`; consequently the four allocated sub-segments and their wall-clock shares in `segment_energy.json` cover **only the steady-state epochs**, and the pretraining energy is a separate fused segment that is **not time-split**.
- *How it enters the training total:* `world_model_pretrain` is in `TRAINING_SEGMENTS` of `compute_energy_per_flop.py`; its energy is part of `TOTAL_MEASURED_TRAINING` (numerator), its FLOPs (`gradient_update_total × warmup_steps`) part of the matmul denominator; its duration is part of the TOTAL duration. It is not part of `gradient_updates`, and `warmup` (random actions) remains excluded.
- *Dependence on the sweeps:* the number of updates (5000) and the data (warmup episodes) do not change across the `num_q` / `horizon` configurations; its FLOPs change because the per-update FLOPs change (Section 6.1f factors); its energy per configuration is in the table above.
- *Update count evidence:* `run.log` of every run states `Pretrain burst done: 5000 updates` (checked, Section 2.7); no per-update count is stored elsewhere.

**6.4 Rollout as the dominant segment** — data only. Per configuration: rollout energy per env step (J, = rollout energy / 100000), planning FLOPs per call, rollout task duration (Σ over the 100 tasks; per-task statistics and the count of tasks < 1 s in Section 5.2a) and the share/power relations (full tables in Sections 3–4):


| config | rollout energy per env step [J] | planning FLOPs per call | rollout task duration, Σ of 100 tasks [s] | per-task median [min–max] [s] | tasks < 1 s | rollout share of TOTAL [%] | P rollout [W] | P GU [W] | P_rollout / P_GU | GPU share of rollout energy [%] |
|---|---|---|---|---|---|---|---|---|---|---|
| HC-base | 2.13811 ± 0.00333591 | 4.64261e+10 | 1014.24 ± 2.51688 | 10.1329 [10.0925–10.3465] | 0/500 | 47.8250 ± 0.341328 | 210.809 ± 0.409232 | 173.669 ± 2.42160 | 1.21402 ± 0.0150018 | 74.4068 ± 0.0597546 |
| HC-numq3 | 1.81988 ± 0.00330474 | 3.93105e+10 | 886.412 ± 2.79221 | 8.85148 [8.81570–8.99909] | 0/500 | 50.7244 ± 0.296225 | 205.310 ± 0.799854 | 163.566 ± 1.77338 | 1.25530 ± 0.0106508 | 73.9226 ± 0.115226 |
| HC-numq7 | 2.42194 ± 0.00483077 | 5.35417e+10 | 1138.66 ± 0.938246 | 11.3808 [11.3456–11.4643] | 0/500 | 45.9960 ± 0.226780 | 212.701 ± 0.388074 | 180.377 ± 2.39654 | 1.17934 ± 0.0135158 | 74.4193 ± 0.0742533 |
| HC-hor1 | 1.35177 ± 0.00123072 | 2.94823e+10 | 664.734 ± 1.16119 | 6.64058 [6.61550–6.74109] | 0/500 | 48.1611 ± 0.307118 | 203.356 ± 0.329749 | 140.743 ± 1.46186 | 1.44498 ± 0.0131457 | 73.7258 ± 0.0304029 |
| HC-hor5 | 2.88267 ± 0.00628231 | 6.33699e+10 | 1355.94 ± 1.14845 | 13.5547 [13.5138–13.6434] | 0/500 | 50.4618 ± 0.381875 | 212.595 ± 0.448947 | 177.863 ± 3.09378 | 1.19554 ± 0.0190860 | 74.4416 ± 0.0256104 |
| Ant-base | 2.54576 ± 0.00533613 | 5.61814e+10 | 1194.74 ± 0.509042 | 11.9453 [11.9118–12.0112] | 0/500 | 51.6187 ± 0.227199 | 213.081 ± 0.492626 | 180.854 ± 2.53444 | 1.17835 ± 0.0143448 | 74.4579 ± 0.0844652 |
| Ant-numq3 | 2.23015 ± 0.00211968 | 4.90532e+10 | 1071.12 ± 0.463104 | 10.7075 [10.6666–10.7866] | 0/500 | 54.2155 ± 0.280534 | 208.207 ± 0.251453 | 165.851 ± 1.78693 | 1.25550 ± 0.0133361 | 74.0474 ± 0.0373318 |
| Ant-numq7 | 2.80891 ± 0.00552437 | 6.33097e+10 | 1322.94 ± 0.621085 | 13.2245 [13.1917–13.3148] | 0/500 | 48.6886 ± 0.187380 | 212.323 ± 0.401162 | 177.923 ± 1.26569 | 1.19339 ± 0.00886207 | 74.5641 ± 0.0396521 |
| Ant-hor1 | 1.49441 ± 0.00446557 | 3.27635e+10 | 736.610 ± 3.03592 | 7.35793 [7.31237–7.49466] | 0/500 | 49.2851 ± 0.311920 | 202.878 ± 0.415370 | 141.490 ± 0.929570 | 1.43391 ± 0.00776628 | 73.5792 ± 0.0988778 |
| Ant-hor5 | 3.54226 ± 0.00536601 | 7.95994e+10 | 1654.50 ± 3.51327 | 16.5341 [16.4830–16.6775] | 0/500 | 54.4753 ± 0.324256 | 214.099 ± 0.265002 | 179.633 ± 2.31860 | 1.19202 ± 0.0143894 | 74.6332 ± 0.0709906 |

Hardware composition (CPU/GPU/RAM) of the rollout task for every configuration: the 'Hardware composition of the measured `rollout` task' table in Sections 3–4.

**6.5 Sweeps** — the common block for `num_q` {3,5,7} and `horizon` {1,3,5} on both environments is Section 4.1 and 4.2 (the baseline 5/3 is shared by both sweeps and appears in both sections and in Section 3). FLOP factors per segment against the baseline are in the 'FLOP factors' table of each sweep section and in the factor table of Section 6.1f.

**6.6 Environment confound — complete diff of the logged configurations of the HC and the Ant baseline** (`metadata.json`, seed 331 of each; every field that differs):


| field | HalfCheetah-v5 baseline | Ant-v5 baseline |
|---|---|---|
| `algo_config.episodic` | false | true |
| `experiment_config.env_id` | "HalfCheetah-v5" | "Ant-v5" |

- **PASS** — `algo_config.episodic` is the only logged `algo_config` field that differs between the HC and the Ant baseline (expected by the instructions); the only other differing field is `experiment_config.env_id` (['`algo_config.episodic`', '`experiment_config.env_id`'])

**Termination classifier (Ant only, `episodic` = true).** Parameters: 527873 (independent of num_q; MLP 512 → 512 → 512 → 1) = 9.56561 % of the 5518447 trainable parameters of the Ant world model (num_q 5).
- *Where it is used:* (i) **planning** — `_estimate_value` (line 467) accumulates `termination = clip(termination + (termination(z) > 0.5), max=1)` after every imagined step (line 481) and masks the later rewards and the terminal value, so it is evaluated `iterations × num_samples × horizon` times per `act()` call; (ii) **world-model update** — in `critic_update`, `termination_loss = BCE(termination(zs[1:]), terminated)` (line 614) is added to the loss with coefficient `termination_coef` = 1.0 and back-propagated; (iii) it is **not** used in `_update_pi` and **not** in the TD-target (`td_targets` bootstraps from the env's own `terminated` flag, line 583).
- *FLOPs attributable to the classifier* — separable: the episodic and non-episodic `TDMPC2Agent` can be built and measured with the same real classes at the Ant dimensions. Generator computation (not a pipeline output): `measure_flops.measure_tdmpc2` called with the Ant dimensions (105/8) and `episodic` = true and = false for each configuration, difference = classifier share; it equals the analytic term of Section 2.6 exactly (last column) and the `episodic` = true values reproduce `flops_per_call.json` exactly:


| Ant config | Δ planning FLOPs per call | Δ as share of the call [%] | Δ critic (world-model step) FLOPs per update | Δ as share of the update step [%] | Δ actor FLOPs per update | Δ over the 5000 pretrain updates | Δ over a whole run (100000 acts + 100000 updates + 5000 pretrain updates) | equals analytic term and `flops_per_call.json` |
|---|---|---|---|---|---|---|---|---|
| Ant-base | 9.67311e+09 | 17.2176 | 2.41828e+09 | 8.68509 | 0 | 1.20914e+13 | 1.22123e+15 | yes |
| Ant-numq3 | 9.67311e+09 | 19.7196 | 2.41828e+09 | 11.6736 | 0 | 1.20914e+13 | 1.22123e+15 | yes |
| Ant-numq7 | 9.67311e+09 | 15.2790 | 2.41828e+09 | 6.91485 | 0 | 1.20914e+13 | 1.22123e+15 | yes |
| Ant-hor1 | 3.22437e+09 | 9.84136 | 8.06093e+08 | 8.54463 | 0 | 4.03046e+12 | 4.07077e+14 | yes |
| Ant-hor5 | 1.61219e+10 | 20.2537 | 4.03046e+09 | 8.71374 | 0 | 2.01523e+13 | 2.03538e+15 | yes |

The classifier's contribution to the parameter count, the planning FLOPs and the update FLOPs is therefore known exactly; its **energy** is not separable (it runs fused inside the measured `rollout` and `critic_update`/`gradient_updates` tasks).

*Completed training episodes per run on Ant-v5* (`training_metrics.json`, phase `train`; HC always 100): base: 884–2201; numq3: 1376–2556; numq7: 919–1560; hor1: 440–536; hor5: 1044–1712; warmup episodes per run: [22, 27, 30, 38, 49] (Ant) / [5] (HC). Per-configuration return statistics: Section 5.3.

**6.7 Open points from the instructions (reported, not resolved):**

- *`rollout` task durations versus the 1 s polling interval:* 0 of 5000 `rollout` tasks are shorter than 1 s (min/median/max 6.61550/11.0661/16.6775 s) — unlike SAC/TD3/MBPO, the TD-MPC2 rollout tasks are long because planning dominates; 0 of 5000 `gradient_updates` tasks are < 1 s (min/median/max 8.84483/12.1990/17.8022 s). `world_model_pretrain` (one task) lasts 61.6892 ± 10.8395 s. Per configuration: Section 5.2a.
- *Host-side work counted as FLOPs:* none; only matmul FLOPs (`FlopCounterMode`). **Counted:** `rollout` = everything inside one `agent.act()` that is a matmul (encoder, π, dynamics, reward, Q-networks, termination classifier) — the MPPI bookkeeping (`randn`, `clamp`, `topk`, `exp`, score normalisation, Gumbel-softmax, `index_select`, `.cpu()`) is not; `critic_update` / `actor_update` = matmuls of the world-model step / `_update_pi` incl. backward (soft two-hot, symlog, MSE/BCE losses, SimNorm/softmax, LayerNorm, Mish, dropout, grad clipping, Adam, `RunningScale` percentile sort are not); `buffer_sample` = 0 (host Python loop + four H2D copies); `target_update` = Polyak operation count; `world_model_pretrain` = 5000 × update total. Not counted: `env.step`, `buffer.add`, `end_episode` (`np.stack`), episode resets.
- *Segments whose timers or FLOPs are produced differently from SAC:* (i) `world_model_pretrain` exists only here (own measured task; sub-timers discarded, not time-split); (ii) `critic_update` is the world-model step and `actor_update` the policy-prior step (different quantities from SAC's critic/actor); (iii) the `rollout` is dominated by planning and its FLOPs come from one measured `agent.act()` call; (iv) `buffer_sample` is a host-side per-episode Python sampler (cost grows with the number of episodes) rather than a single NumPy gather; (v) the update counter is not guarded by `len(buffer) < batch_size`; (vi) the sub-timers' coverage of the measured GU task is reported in Sections 3–4.

## 7. Discrepancies, NOT AVAILABLE items, questions

**(a) Where data or code contradict CLAUDE.md, README.md or `thesis-layout.md` (the latter is not present in the repository):**

- CLAUDE.md / README.md state that zero GPU-clock-lock / CPU-governor permission failures and zero RAPL / geolocation fallbacks were found 'by grepping every run.log'. That grep cannot detect those failures (the `gpu_control`, `thermal_gate` and CodeCarbon loggers do not write to `run.log`). For TD-MPC2 the claim is supported by other evidence: 50 of 50 runs have a sweep-log block with captured stderr (`2>&1 | tee -a`) containing none of the failure strings, plus data-side checks (varying RAPL `cpu_power`, non-fallback geolocation, `thermal_gate.reason = reached_reference`) for all runs (Section 1.3). The clock lock is verified as requested, not as applied.
- The one-row `emissions.csv` written by `tracker.stop()` has `duration` = 90.0059 s (median over the TD-MPC2 runs; it equals the duration of the last task, the 90 s idle tail) while its `energy_consumed` is the cumulative whole-run energy (equal to the sum of the per-task rows within 7.4e-06 relative). `emissions.csv` therefore cannot be used for a whole-run mean power; the 'P whole run' columns of this file use Σ energy / Σ duration of all rows of the per-task CSV (idle head, warmup, [pretrain], training tasks, idle tail; the unmeasured 30 s settle is not included).
- CLAUDE.md / README.md describe the TD-MPC2 num_q/horizon sweep's false start as 'the first 5 runs (num_q=3, HalfCheetah-v5, all 5 seeds)' crashing instantly with `FileNotFoundError: configs/overrides/tdmpc2_num_q3.json` before any run directory/tracker existed, followed by a clean relaunch. `results/tdmpc2/_numq_horizon_sweep.log` agrees on the 5 crashes (5 `FAILED (exit 1)` blocks, indices 1/40, 2/40, 3/40, 4/40, 5/40, 12:18:39–12:18:42) and additionally contains a **sixth launch** with no closing line ([6/40] Starting: algo=tdmpc2 env=Ant-v5 seed=331 num_q=3, started 12:18:42, followed directly by the relaunch's 'Priming sudo credentials' line at 12:21:00): the first attempt was stopped during that launch. No run directory was created by any of the six launches (all 50 analysed runs belong to the second attempt: `results/tdmpc2/_numq_horizon_sweep_status.json` 40/40 done, 5 + 5 + 5 + 5 + 5 runs per environment on disk), so nothing needs to be excluded, but the documents do not mention the sixth, interrupted launch. Whether it touched the GPU state (clock lock / persistence mode set by the runner before being killed) cannot be told from the log; the first run of the relaunch recorded a normal thermal-gate pass (Section 1.4).
- `thesis-layout.md` is referenced by the instructions but does not exist in the repository; the layout expectation (50 TD-MPC2 runs: 5 configurations × 2 envs × 5 seeds) was taken from the instruction file and verified against `results/`.

**(b) Quantities that could not be obtained (`NOT AVAILABLE`):**

- NOT AVAILABLE: Energy attributable to the termination classifier (inside the measured `rollout` and `gradient_updates` tasks; no run with `episodic` = false exists on Ant-v5).
- NOT AVAILABLE: GPU theoretical FP32 peak (SM count / lanes per SM are not stored in the repo and no CUDA is available on the Windows checkout) — achieved throughput is reported in GFLOP/s without a peak-utilisation fraction.
- NOT AVAILABLE: Whether FP32 matmuls ran as TF32 on the RTX 5090 under torch 2.13.0+cu130: the code sets no TF32 / matmul-precision flag and no run logs the effective setting (see Section 4.1 Part 2.3).
- NOT AVAILABLE: Positive confirmation that the GPU clock lock (2000/2000 MHz), persistence mode and the CPU `performance` governor were *applied*: `metadata.json` records only the request; failure messages would appear in the sweep-log stderr and none do.
- NOT AVAILABLE: gymnasium / mujoco versions on the Linux experiment box: not recorded in `metadata.json`; environment dimensions here were read from the local venv and agree with `flops_per_call.json`.
- NOT AVAILABLE: Simulator (MuJoCo) share of the `rollout` task duration: no per-step simulator timing is logged, so the split of `rollout` into policy/model compute and `env.step()` is not measurable from the data.

**(c) Questions for the author that the repository cannot answer:**

- Idle-baseline drift: the head idle window integrates 3.08244 s more than its recorded 90 s (RAM energy / RAM power), the tail window does not; this file reports as-recorded idle powers (definition of Section 4.1) and the window-corrected drift in Section 5.2e. Which of the two the thesis text should quote is not decided in the repository.
- The `rollout` energy includes the termination-classifier evaluations on Ant (planning) while HC has none; the FLOP share is known exactly (Section 6.6) but the energy share is not separable — whether to report Ant-vs-HC TD-MPC2 differences with or without this caveat is for the author to decide.

**Index of produced files** (`contexts/Section_4.4_data/`): `tdmpc2_segments_long.csv`, `tdmpc2_cross_seed_summary.csv`, `tdmpc2_run_inventory.csv`, `tdmpc2_per_epoch.csv`, `tdmpc2_idle_floor.csv`, `tdmpc2_ols_fits.csv`, `tdmpc2_delta_pairs.csv`, `tdmpc2_mad_flags.csv`, `tdmpc2_learning_performance.csv`; scripts `build_section_4_4.py` (here) and the shared modules in `contexts/Section_4.2_data/`.
