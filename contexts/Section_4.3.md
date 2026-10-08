# Section 4.3 MBPO — context

Data context for thesis Section 4.3 (MBPO): 80 canonical runs (HC: 7 configurations, Ant: 9 configurations, 5 seeds [331, 958, 14577, 43611, 85062] each). Facts, numbers and provenance only — no interpretation, no LaTeX. All energies are **gross** (idle floor included, nothing subtracted); mean ± sample sd (ddof = 1) over the five seeds; 6 significant digits; every table cell shows `HalfCheetah-v5 ‖ Ant-v5` (left ‖ right) unless a table says otherwise (`n/a` = configuration does not exist in that environment). Conventions of Section 4.1 apply (`measured` vs `allocated` segments; `target_update` in nJ per Polyak operation, never J/FLOP; `buffer_sample` has no FLOPs; J/FLOP = ratio of means with the sd of the five per-seed ratios — MBPO FLOPs differ between seeds, so the mean of the per-seed ratios is given as well; relative changes computed per seed then averaged; environment comparison paired by seed, Ant relative to HC). Training total = `TOTAL_MEASURED_TRAINING` = rollout + dynamics_model_update + synthetic_rollout_generation + gradient_updates; warmup and idle windows are excluded.

Checks run by the generator: 29; passed 29; failed 0. Independent recomputation vs the pipeline CSVs: 5088 values compared, 0 deviating by more than 1e-9, maximum relative deviation 4.072e-12.

**Configuration tags:** `base` = SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model ensemble 7 × (200,200,200,200), warmup 5000; HC baseline = `MBPOConfig` defaults (rollout length fixed 1), Ant baseline = defaults + `configs/overrides/mbpo_ant.json` (rollout length 1→25 between epochs 20 and 100). `utd2`/`utd4`, `w256`/`w512`, `b512`/`b1024` change exactly that one factor of the same environment's baseline; `rollout1`/`rollout15` (Ant only) change `rollout_max_length` from 25 to 1 / 15. Abbreviations: HC = HalfCheetah-v5, Ant = Ant-v5, GU = `gradient_updates`, TOTAL = `TOTAL_MEASURED_TRAINING`, dyn_model = `dynamics_model_update`, synth_rollout = `synthetic_rollout_generation`, buf_sample = `buffer_sample`, critic = `critic_update`, actor = `actor_update`, target = `target_update`.

## 0. Provenance


| Item | Value |
|---|---|
| Generated (local time) | 2026-10-07T10:38:28 |
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
**Scripts used** (all under `contexts/Section_4.2_data/` unless stated; all read-only on `results/`, `flop_analysis/` and git): `build_section_4_3.py (in Section_4.3_data/)`, `s4_data.py`, `s4_blocks.py`, `s4_audit.py (in Section_4.2_data/)`. Shared modules: `s4_data.py` (loading, derived quantities, independent FLOP recomputation), `s4_blocks.py` (common tables, cross-configuration quantities), `s4_audit.py` (this part). Re-run from the repo root with `.venv/Scripts/python.exe contexts/Section_4.3_data/build_section_4_3.py`. Definitions are taken from `contexts/section_4.1_data/build_section_4_1.py` (the generator of Section 4.1) and are listed in the docstring of `s4_data.py`; no definition of 4.1 was missing.

**Pipeline files used for the cross-check:**


| File | SHA-256 | mtime (local) | last git commit touching it |
|---|---|---|---|
| `flop_analysis/output/per_run_energy_per_flop.csv` | 6e7f8f250a797fe3cca22c85a22e0e93897a91f1e0a5af5d575a4067ecc6d748 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |
| `flop_analysis/output/cross_seed_energy_per_flop.csv` | 3ebcc97fa64628cd50115ad58dc54dee129769375e2276babadf25d5096e5ec7 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |
| `flop_analysis/flops_per_call.json` | 7a562e0b092b68b6fbaea089354298114e864db72c6309847db44d0565e957a7 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |

Freshness: newest `segment_energy.json` among the 80 MBPO runs: `results/mbpo/HalfCheetah-v5/seed_958/20260915_173600` (2026-09-23T18:04:24); oldest pipeline CSV mtime 2026-09-28T17:41:20 — **PASS** — newest segment_energy.json older than the pipeline CSVs (filesystem mtime; the content-based reconciliation is in Section 1.5)

**Code-version consistency of the 80 runs** (`metadata.json` → `git_commit`):


| git_commit | # runs | commit | configurations |
|---|---|---|---|
| `0bbbbb74a6` | 10 | 0bbbbb7 2026-09-08 baseline config for mbpo | Ant-base, HC-base |
| `7e672ef12c` | 20 | 7e672ef 2026-09-12 Setting up width sweep for SAC, TD3 and MBPO | Ant-w256, Ant-w512, HC-w256, HC-w512 |
| `ecb7135008` | 20 | ecb7135 2026-09-14 Scheduling UTD sweep for MBPO | Ant-utd2, Ant-utd4, HC-utd2, HC-utd4 |
| `9b4b44cd2f` | 10 | 9b4b44c 2026-09-15 Preparing batch size sweep for Half Cheetah | HC-b1024, HC-b512 |
| `788ffcd841` | 20 | 788ffcd 2026-09-17 Scheduling batch sweep for ant and new rollout sweep for MBPO | Ant-b1024, Ant-b512, Ant-rollout1, Ant-rollout15 |

Algorithm-side files at each run commit (git blob ids compared):


| file | status |
|---|---|
| `algorithms/mbpo.py` | identical at every run commit |
| `algorithms/dynamics_model.py` | identical at every run commit |
| `algorithms/termination_fns.py` | identical at every run commit |
| `algorithms/sac.py` | identical at every run commit |
| `algorithms/replay_buffer.py` | identical at every run commit |
| `algorithms/tracker_utils.py` | identical at every run commit |

Config dataclasses at each run commit (AST source segment compared):


| definition | status |
|---|---|
| `ExperimentConfig` | identical at every run commit |
| `MBPOConfig` | identical at every run commit |

Runner/utility files, first versus last run commit: `experiment_runner.py`: +8/−0 lines between `0bbbbb7` and `788ffcd`; `run_experiment.py`: identical; `utils/gpu_control.py`: identical; `utils/thermal_gate.py`: identical; `algorithms/tracker_utils.py`: identical.
`thermal_gate` key sets in `metadata.json` (a difference between runs that carry the same `git_commit` would show an uncommitted working tree at run time):


| thermal_gate keys | # runs | runs (if ≤ 5) |
|---|---|---|
| final_power_w, final_temp_c, gated, reason, run_end_power_w, run_end_temp_c, run_start_power_w, run_start_temp_c, waited_seconds | 80 | … |


All algorithm-side files named above are byte-identical at every run commit.

## 1. Data basis and audit

**1.1 Inventory** (one row per configuration; values from the logged `metadata.json`, identical within a configuration — checked in 1.3). `launcher` is derived from the sweep-log blocks in which the run directory appears (Section 1.4):


| config | env | description | hidden_sizes | batch_size | updates_per_env_step | rollout_min_epoch | rollout_max_epoch | rollout_min_length | rollout_max_length | warmup | train_steps | steps/epoch | architecture signature | override file | launcher | # runs | seeds present | # duplicates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | HalfCheetah-v5 | SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model (200×4)×7; HC: fixed rollout length 1; Ant: rollout length 1→25 (mbpo_ant.json) | [1024, 1024] | 256 | 1 | 20 | 150 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024_ens7_mh200x200x200x200_mb256` | `none (MBPOConfig defaults)` | scripts/run_mbpo_env_sweep.sh (log `results/mbpo/_env_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| utd2 | HalfCheetah-v5 | UTD 2 | [1024, 1024] | 256 | 2 | 20 | 150 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_utd2.json` | scripts/run_utd_sweep.sh (current MBPO UTD sweep) (log `results/_utd_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| utd4 | HalfCheetah-v5 | UTD 4 | [1024, 1024] | 256 | 4 | 20 | 150 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_utd4.json` | scripts/run_utd_sweep.sh (current MBPO UTD sweep) (log `results/_utd_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| w256 | HalfCheetah-v5 | SAC hidden (256,256) | [256, 256] | 256 | 1 | 20 | 150 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h256x256_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_width256.json` | scripts/run_width_sweep.sh (log `results/_width_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| w512 | HalfCheetah-v5 | SAC hidden (512,512) | [512, 512] | 256 | 1 | 20 | 150 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h512x512_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_width512.json` | scripts/run_width_sweep.sh (log `results/_width_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b512 | HalfCheetah-v5 | SAC batch 512 | [1024, 1024] | 512 | 1 | 20 | 150 | 1 | 1 | 5000 | 100000 | 1000 | `bs512_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_batch512.json` | scripts/run_batch_size_sweep.sh (log `results/_batch_size_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b1024 | HalfCheetah-v5 | SAC batch 1024 | [1024, 1024] | 1024 | 1 | 20 | 150 | 1 | 1 | 5000 | 100000 | 1000 | `bs1024_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_batch1024.json` | scripts/run_batch_size_sweep.sh (log `results/_batch_size_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| base | Ant-v5 | SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model (200×4)×7; HC: fixed rollout length 1; Ant: rollout length 1→25 (mbpo_ant.json) | [1024, 1024] | 256 | 1 | 20 | 100 | 1 | 25 | 5000 | 100000 | 1000 | `bs256_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant.json` | scripts/run_mbpo_env_sweep.sh (log `results/mbpo/_env_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| utd2 | Ant-v5 | UTD 2 | [1024, 1024] | 256 | 2 | 20 | 100 | 1 | 25 | 5000 | 100000 | 1000 | `bs256_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant_utd2.json` | scripts/run_utd_sweep.sh (current MBPO UTD sweep) (log `results/_utd_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| utd4 | Ant-v5 | UTD 4 | [1024, 1024] | 256 | 4 | 20 | 100 | 1 | 25 | 5000 | 100000 | 1000 | `bs256_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant_utd4.json` | scripts/run_utd_sweep.sh (current MBPO UTD sweep) (log `results/_utd_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| w256 | Ant-v5 | SAC hidden (256,256) | [256, 256] | 256 | 1 | 20 | 100 | 1 | 25 | 5000 | 100000 | 1000 | `bs256_h256x256_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant_width256.json` | scripts/run_width_sweep.sh (log `results/_width_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| w512 | Ant-v5 | SAC hidden (512,512) | [512, 512] | 256 | 1 | 20 | 100 | 1 | 25 | 5000 | 100000 | 1000 | `bs256_h512x512_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant_width512.json` | scripts/run_width_sweep.sh (log `results/_width_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b512 | Ant-v5 | SAC batch 512 | [1024, 1024] | 512 | 1 | 20 | 100 | 1 | 25 | 5000 | 100000 | 1000 | `bs512_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant_batch512.json` | scripts/run_ant_overnight_sweep.sh (= run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh) (log `results/_ant_overnight_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b1024 | Ant-v5 | SAC batch 1024 | [1024, 1024] | 1024 | 1 | 20 | 100 | 1 | 25 | 5000 | 100000 | 1000 | `bs1024_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant_batch1024.json` | scripts/run_ant_overnight_sweep.sh (= run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh) (log `results/_ant_overnight_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| rollout1 | Ant-v5 | Ant only: rollout_max_length 1 | [1024, 1024] | 256 | 1 | 20 | 100 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant_rollout1.json` | scripts/run_ant_overnight_sweep.sh (= run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh) (log `results/_ant_overnight_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| rollout15 | Ant-v5 | Ant only: rollout_max_length 15 | [1024, 1024] | 256 | 1 | 20 | 100 | 1 | 15 | 5000 | 100000 | 1000 | `bs256_h1024x1024_ens7_mh200x200x200x200_mb256` | `configs/overrides/mbpo_ant_rollout15.json` | scripts/run_ant_overnight_sweep.sh (= run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh) (log `results/_ant_overnight_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |


**1.2 Full run list** → `contexts/Section_4.3_data/mbpo_run_inventory.csv` (80 rows: algo, config, env, seed, run directory, timestamp, git commit, versions, clock lock, start/end time, full logged `algo_config`).

**1.3 Audit checks**

- **PASS** — layout-document expectation: 80 runs (HC: 7 configurations, Ant: 9 configurations × 5 seeds); found 80 (unmatched run directories at canonical seeds: 0)
- **PASS** — configurations per environment equal the layout-document counts (found {'HC': 7, 'Ant': 9})
- **PASS** — exactly 5 runs per configuration (all 16 configurations have 5)
- **PASS** — every run uses one of the 5 canonical seeds {331, 958, 14577, 43611, 85062}; each seed once per configuration
- **PASS** — no duplicate (algo, env, seed, config) runs
- **PASS** — every run has emissions.csv, segment_energy.json, training_metrics.json, metadata.json, run.log
- **PASS** — every run has exactly one per-task CodeCarbon log `emissions_*.csv`
- **PASS** — the logged configuration of every configuration differs from its environment's baseline in exactly the swept factor (algo_config), has no `experiment_config` difference (ignoring seed/env/algo and the metadata-only key `thermal_gate_reference_file`), and is identical across its 5 seeds
- **PASS** — `experiment_config` of the HC and Ant baselines are identical apart from seed/env_id/algo_name and the metadata-only key ({})
  - (train_steps, steps_per_epoch, warmup_steps, env) → #runs: {(100000, 1000, 5000, 'HC'): 35, (100000, 1000, 5000, 'Ant'): 45}
- **PASS** — per-task CSV has exactly the expected tasks: dynamics_model_update_0..99, rollout_0..99, synthetic_rollout_generation_0..99, gradient_updates_0..99, warmup_0, idle_baseline_head, idle_baseline_tail
- **PASS** — `training_metrics.json` has 100 epoch rows in every run
- **PASS** — GPU = NVIDIA GeForce RTX 5090 in every run (`cuda_device_name`)
  - per-task CSV `gpu_model`: {'1 x NVIDIA GeForce RTX 5090': 80}; `cpu_model`: {'Intel(R) Core(TM) Ultra 9 285K': 80}
- **PASS** — GPU clock lock requested at 2000/2000 MHz with lock_gpu_clocks = true in every run
  - persistence mode requested: {True: 80}; CPU `performance` governor requested: {True: 80}
- **PASS** — torch 2.13.0+cu130 and codecarbon 3.3.0 in every run ({('2.13.0+cu130', '3.3.0'): 80})
- **PASS** — thermal gate: gated = true and reason = reached_reference (no timeout) in every run
  - thermal gate: waited_seconds max 2.69413e-05 s; gate temperature 40.0000–46.0000 °C; gate power 23.1970–28.2980 W (the gate passes on its first poll because its reference is captured fresh in each process, `utils/thermal_gate.py`)
- **PASS** — `run.log` of every run contains none of the 20 failure-mode strings (list in Section 4.1, schema sample S6)
- **PASS** — sweep-log block (stdout+stderr of the launch, `2>&1 | tee -a`) contains none of the strings — available for 80 of 80 runs
- **PASS** — data-side RAPL check: per-task `cpu_power` varies across the 100 gradient_updates tasks in every run (a TDP fallback would be constant) (min distinct values per run = 100)
- **PASS** — data-side geolocation check: no run shows the Canada/Quebec fallback ({('Germany', 'baden-wurttemberg'): 80})

**1.3b Logged-configuration differences per configuration** (each row: difference of the logged `algo_config` / `experiment_config` to the baseline of the same environment; last column: difference to the dataclass defaults of `MBPOConfig`, i.e. the override file content):


| config | algo_config vs same-env baseline | experiment_config vs same-env baseline | identical across the 5 seeds | algo_config vs `MBPOConfig` defaults |
|---|---|---|---|---|
| HC-base | — | — | yes | — |
| HC-utd2 | updates_per_env_step: 1 → 2 | — | yes | updates_per_env_step: 1 → 2 |
| HC-utd4 | updates_per_env_step: 1 → 4 | — | yes | updates_per_env_step: 1 → 4 |
| HC-w256 | hidden_sizes: [1024, 1024] → [256, 256] | — | yes | hidden_sizes: [1024, 1024] → [256, 256] |
| HC-w512 | hidden_sizes: [1024, 1024] → [512, 512] | — | yes | hidden_sizes: [1024, 1024] → [512, 512] |
| HC-b512 | batch_size: 256 → 512 | — | yes | batch_size: 256 → 512 |
| HC-b1024 | batch_size: 256 → 1024 | — | yes | batch_size: 256 → 1024 |
| Ant-base | — | — | yes | rollout_max_epoch: 150 → 100; rollout_max_length: 1 → 25 |
| Ant-utd2 | updates_per_env_step: 1 → 2 | — | yes | rollout_max_epoch: 150 → 100; rollout_max_length: 1 → 25; updates_per_env_step: 1 → 2 |
| Ant-utd4 | updates_per_env_step: 1 → 4 | — | yes | rollout_max_epoch: 150 → 100; rollout_max_length: 1 → 25; updates_per_env_step: 1 → 4 |
| Ant-w256 | hidden_sizes: [1024, 1024] → [256, 256] | — | yes | hidden_sizes: [1024, 1024] → [256, 256]; rollout_max_epoch: 150 → 100; rollout_max_length: 1 → 25 |
| Ant-w512 | hidden_sizes: [1024, 1024] → [512, 512] | — | yes | hidden_sizes: [1024, 1024] → [512, 512]; rollout_max_epoch: 150 → 100; rollout_max_length: 1 → 25 |
| Ant-b512 | batch_size: 256 → 512 | — | yes | batch_size: 256 → 512; rollout_max_epoch: 150 → 100; rollout_max_length: 1 → 25 |
| Ant-b1024 | batch_size: 256 → 1024 | — | yes | batch_size: 256 → 1024; rollout_max_epoch: 150 → 100; rollout_max_length: 1 → 25 |
| Ant-rollout1 | rollout_max_length: 25 → 1 | — | yes | rollout_max_epoch: 150 → 100 |
| Ant-rollout15 | rollout_max_length: 25 → 15 | — | yes | rollout_max_epoch: 150 → 100; rollout_max_length: 1 → 15 |

The warning/grep caveat of Section 4.1 applies: `run.log` is written only by the `experiment_runner` logger; the `gpu_control`, `thermal_gate` and CodeCarbon loggers write to stderr, which the sweep scripts capture into the sweep log (`2>&1 | tee -a`). The clock lock / governor are therefore verified as *requested* (metadata) and as *not reported failed* (sweep-log blocks), never as positively applied; applied lock values are not recorded in `metadata.json`.

**1.4 Execution facts.** Times are `metadata.json` `start_time_utc`/`end_time_utc` (UTC). A *session* = maximal block of runs of **all 300 recorded runs** (all algorithms) with < 30 min between one run's end and the next run's start (same definition as Section 4.1). `contiguous` = the five runs of the configuration are consecutive among the MBPO runs. `launch logs` = sweep logs containing the run directory (each run's stdout+stderr block).


| config | first start (UTC) | last end (UTC) | session(s) | all 5 seeds in one session | contiguous | seed order (by start time) | launch log(s) | # runs with a log block |
|---|---|---|---|---|---|---|---|---|
| HC-base | 2026-09-08T17:44:55 | 2026-09-08T20:24:05 | 11 | yes | yes | 331,958,14577,43611,85062 | `results/mbpo/_env_sweep.log` | 5 |
| HC-utd2 | 2026-09-14T10:50:01 | 2026-09-14T14:07:25 | 16 | yes | yes | 331,958,14577,43611,85062 | `results/_utd_sweep.log` | 5 |
| HC-utd4 | 2026-09-14T14:07:27 | 2026-09-14T18:25:26 | 16 | yes | yes | 331,958,14577,43611,85062 | `results/_utd_sweep.log` | 5 |
| HC-w256 | 2026-09-12T20:53:22 | 2026-09-12T23:28:17 | 15 | yes | yes | 331,958,14577,43611,85062 | `results/_width_sweep.log` | 5 |
| HC-w512 | 2026-09-12T23:28:18 | 2026-09-13T02:05:03 | 15 | yes | yes | 331,958,14577,43611,85062 | `results/_width_sweep.log` | 5 |
| HC-b512 | 2026-09-15T12:29:58 | 2026-09-15T15:04:26 | 17 | yes | yes | 331,958,14577,43611,85062 | `results/_batch_size_sweep.log` | 5 |
| HC-b1024 | 2026-09-15T15:04:28 | 2026-09-15T17:39:40 | 17 | yes | yes | 331,958,14577,43611,85062 | `results/_batch_size_sweep.log` | 5 |
| Ant-base | 2026-09-08T20:24:06 | 2026-09-08T23:34:50 | 11 | yes | yes | 331,958,14577,43611,85062 | `results/mbpo/_env_sweep.log` | 5 |
| Ant-utd2 | 2026-09-14T18:25:27 | 2026-09-14T22:15:56 | 16 | yes | yes | 331,958,14577,43611,85062 | `results/_utd_sweep.log` | 5 |
| Ant-utd4 | 2026-09-14T22:15:58 | 2026-09-15T03:09:10 | 16 | yes | yes | 331,958,14577,43611,85062 | `results/_utd_sweep.log` | 5 |
| Ant-w256 | 2026-09-13T02:05:05 | 2026-09-13T05:12:52 | 15 | yes | yes | 331,958,14577,43611,85062 | `results/_width_sweep.log` | 5 |
| Ant-w512 | 2026-09-13T05:12:54 | 2026-09-13T08:17:01 | 15 | yes | yes | 331,958,14577,43611,85062 | `results/_width_sweep.log` | 5 |
| Ant-b512 | 2026-09-17T12:06:32 | 2026-09-17T15:07:00 | 18 | yes | yes | 331,958,14577,43611,85062 | `results/_ant_overnight_sweep.log` | 5 |
| Ant-b1024 | 2026-09-17T15:07:01 | 2026-09-17T18:12:14 | 18 | yes | yes | 331,958,14577,43611,85062 | `results/_ant_overnight_sweep.log` | 5 |
| Ant-rollout1 | 2026-09-17T20:09:20 | 2026-09-17T23:10:53 | 18 | yes | yes | 331,958,14577,43611,85062 | `results/_ant_overnight_sweep.log` | 5 |
| Ant-rollout15 | 2026-09-17T23:10:55 | 2026-09-18T02:15:57 | 18 | yes | yes | 331,958,14577,43611,85062 | `results/_ant_overnight_sweep.log` | 5 |


Sessions containing MBPO runs:


| session | first start | last end | # runs (all algos) | algos | # MBPO | MBPO configs in order |
|---|---|---|---|---|---|---|
| 11 | 2026-09-08T17:12 | 2026-09-08T23:34 | 11 | {'mbpo': 11} | 10 | HC-base → Ant-base |
| 15 | 2026-09-12T17:15 | 2026-09-13T10:44 | 60 | {'sac': 20, 'mbpo': 20, 'td3': 20} | 20 | HC-w256 → HC-w512 → Ant-w256 → Ant-w512 |
| 16 | 2026-09-14T10:50 | 2026-09-15T03:09 | 20 | {'mbpo': 20} | 20 | HC-utd2 → HC-utd4 → Ant-utd2 → Ant-utd4 |
| 17 | 2026-09-15T10:39 | 2026-09-15T19:32 | 35 | {'sac': 10, 'mbpo': 10, 'td3': 15} | 10 | HC-b512 → HC-b1024 |
| 18 | 2026-09-17T10:10 | 2026-09-18T02:15 | 45 | {'sac': 10, 'mbpo': 20, 'td3': 15} | 20 | Ant-b512 → Ant-b1024 → Ant-rollout1 → Ant-rollout15 |

Inside the sessions listed above, consecutive runs of **any** algorithm are separated by 1.27624–118.793 s (median 1.32697 s) between one run's `end_time_utc` and the next run's `start_time_utc`, by definition < 1800 s. Run length = `end_time_utc − start_time_utc` includes the thermal-gate reference poll, the 30 s settle, both idle windows and warmup; over the MBPO runs: median 36.6697 min (min 29.5470, max 62.1397). Whether the machine was powered off or idle between sessions is not recorded (each sweep script ends with `shutdown -h +1`).

**Baseline runs** (configuration `base`, 10 runs): launched by a sweep script — their stdout+stderr blocks are in `results/mbpo/_env_sweep.log` (10 runs) (sum 10 of 10); not started by hand, and their launch error output was saved.
Override-file names as built by the sweep scripts (pattern lines found with `echo "configs/overrides/…"` / `overrides="configs/overrides/…"`; the algorithm / env / value are substituted at run time; the launch blocks of the logs do not record the override path): `run_ant_overnight_sweep.sh:72` `echo "configs/overrides/${ALGO_OVERRIDE_PREFIX[$algo]}${batch}.json"`; `run_ant_overnight_sweep.sh:79` `echo "configs/overrides/mbpo_ant_rollout${length}.json"`; `run_batch_size_sweep.sh:57` `echo "configs/overrides/${algo}_batch${batch}.json"`; `run_batch_size_sweep_ant.sh:74` `echo "configs/overrides/${ALGO_OVERRIDE_PREFIX[$algo]}${batch}.json"`; `run_mbpo_rollout_length_sweep.sh:42` `echo "configs/overrides/mbpo_ant_rollout${length}.json"`; `run_tdmpc2_numq_horizon_sweep.sh:58` `echo "configs/overrides/tdmpc2_ant_${param_slug}${value}.json"`; `run_tdmpc2_numq_horizon_sweep.sh:60` `echo "configs/overrides/tdmpc2_${param_slug}${value}.json"`; `run_utd_sweep.sh:147` `overrides="configs/overrides/${override_prefix}_utd${utd}.json"`; `run_width_sweep.sh:73` `echo "configs/overrides/mbpo_ant_width${width}.json"`; `run_width_sweep.sh:75` `echo "configs/overrides/${algo}_width${width}.json"`.

**Launchers.** Sweep scripts that produced these runs (read from `scripts/`): `scripts/run_mbpo_env_sweep.sh` (baselines, 10 runs); `scripts/run_utd_sweep.sh` (as it exists now: MBPO UTD ∈ {2,4}, 20 runs); `scripts/run_width_sweep.sh`; `scripts/run_batch_size_sweep.sh` (HC); `scripts/run_ant_overnight_sweep.sh` (Ant batch sizes + rollout-length sweep; standalone `run_batch_size_sweep_ant.sh`, `run_mbpo_rollout_length_sweep.sh`)
Stderr capture in each script: `run_ant_overnight_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_batch_size_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_batch_size_sweep_ant.sh`: `2>&1 | tee -a $LOG_FILE`; `run_mbpo_env_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_mbpo_rollout_length_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_td3_env_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_tdmpc2_numq_horizon_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_tdmpc2_seed_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_utd_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_width_sweep.sh`: `2>&1 | tee -a $LOG_FILE`.

Sweep bookkeeping files (every `done` entry of this algorithm should point at one of the analysed run directories):


| status file | # entries | # MBPO entries | statuses | every done entry's run_dir is one of the analysed runs |
|---|---|---|---|---|
| `results/mbpo/_env_sweep_status.json` | 10 | 10 | {'done': 10} | yes |
| `results/_utd_sweep_status.json` | 20 | 20 | {'done': 20} | yes |
| `results/_width_sweep_status.json` | 60 | 20 | {'done': 60} | yes |
| `results/_batch_size_sweep_status.json` | 35 | 10 | {'done': 35} | yes |
| `results/_ant_overnight_sweep_status.json` | 45 | 20 | {'done': 45} | yes |


**Launch outcomes of MBPO in every sweep log** (each `[i/N] Starting: algo=mbpo …` block classified by its closing line: `Finished OK`, `FAILED (exit …)`, or no closing line):


| log | outcome | # launches |
|---|---|---|
| `results/_ant_overnight_sweep.log` | Finished OK | 20 |
| `results/_batch_size_sweep.log` | Finished OK | 10 |
| `results/_utd_sweep.log` | Finished OK | 20 |
| `results/_width_sweep.log` | Finished OK | 20 |
| `results/mbpo/_env_sweep.log` | Finished OK | 10 |


All 80 recorded MBPO launches finished OK; no failure, restart or duplicate launch of MBPO appears in any sweep log.
Successful launches with a run directory: 80 (analysed canonical runs: 80).

**1.5 Independent recomputation versus the pipeline CSVs.** Every (run, segment) and (configuration, segment) value recomputed here from `segment_energy.json`, the per-task CodeCarbon CSV, `training_metrics.json` and `flops_per_call.json` (FLOP formulas re-implemented in `s4_data.flops_for_run` / `mbpo_extras`, not imported) is compared with `per_run_energy_per_flop.csv` and `cross_seed_energy_per_flop.csv`. Columns compared: energy, duration, mean power, FLOPs, J/FLOP, call count, MBPO dynamics train/holdout FLOPs. The pipeline's cross-seed `mean_power_w` is the mean of per-seed powers and is compared with the same statistic.

| CSV | column | # values compared | max relative deviation |
|---|---|---|---|
| cross_seed | `mean_duration_s` | 128 | 8.825e-15 |
| cross_seed | `mean_energy_joules` | 128 | 4.278e-16 |
| cross_seed | `mean_energy_per_flop_j_per_flop` | 112 | 5.877e-16 |
| cross_seed | `mean_power_w` | 128 | 6.831e-13 |
| cross_seed | `total_flops` | 112 | 0.000e+00 |
| per_run | `call_count` | 560 | 0.000e+00 |
| per_run | `duration_s` | 880 | 1.001e-14 |
| per_run | `dynamics_holdout_flops` | 80 | 0.000e+00 |
| per_run | `dynamics_train_flops` | 80 | 0.000e+00 |
| per_run | `energy_per_flop_j_per_flop` | 560 | 4.231e-16 |
| per_run | `mean_power_w` | 880 | 4.072e-12 |
| per_run | `total_energy_joules` | 880 | 2.521e-16 |
| per_run | `total_flops` | 560 | 0.000e+00 |

- **PASS** — all 5088 compared values agree within 1e-9 relative (maximum deviation over all columns 4.072e-12)

## 2. Structural facts from code and FLOP files

**2.1 Configuration objects** (dumped from `configs/config.py` and `configs/overrides/` at HEAD):

`MBPOConfig` defaults (`dataclasses.asdict(MBPOConfig())`; the HalfCheetah baseline uses exactly these):

```json
{
  "hidden_sizes": [
    1024,
    1024
  ],
  "actor_lr": 0.0003,
  "critic_lr": 0.0003,
  "alpha_lr": 0.0003,
  "gamma": 0.99,
  "tau": 0.005,
  "target_entropy": null,
  "autotune_alpha": true,
  "init_alpha": 0.2,
  "policy_update_delay": 1,
  "updates_per_env_step": 1,
  "batch_size": 256,
  "real_ratio": 0.05,
  "buffer_capacity": 1000000,
  "ensemble_size": 7,
  "num_elites": 5,
  "model_hidden_sizes": [
    200,
    200,
    200,
    200
  ],
  "model_lr": 0.001,
  "model_weight_decays": [
    2.5e-05,
    5e-05,
    7.5e-05,
    0.0001,
    0.0001
  ],
  "model_train_batch_size": 256,
  "model_holdout_ratio": 0.2,
  "model_max_train_epochs": 200,
  "model_train_patience": 5,
  "model_train_freq": 250,
  "deterministic_model": false,
  "rollout_batch_size": 10000,
  "model_retain_epochs": 1,
  "rollout_min_epoch": 20,
  "rollout_max_epoch": 150,
  "rollout_min_length": 1,
  "rollout_max_length": 1
}
```
`configs/overrides/mbpo_ant.json` (the Ant baseline = defaults + this file):

```json
{
  "rollout_min_epoch": 20,
  "rollout_max_epoch": 100,
  "rollout_min_length": 1,
  "rollout_max_length": 25
}
```
Override files of the sweeps (content): `mbpo_ant.json` = {"rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 25}; `mbpo_ant_batch1024.json` = {"rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 25, "batch_size": 1024}; `mbpo_ant_batch512.json` = {"rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 25, "batch_size": 512}; `mbpo_ant_rollout1.json` = {"rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 1}; `mbpo_ant_rollout15.json` = {"rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 15}; `mbpo_ant_utd2.json` = {"rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 25, "updates_per_env_step": 2}; `mbpo_ant_utd4.json` = {"rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 25, "updates_per_env_step": 4}; `mbpo_ant_width256.json` = {"hidden_sizes": [256, 256], "rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 25}; `mbpo_ant_width512.json` = {"hidden_sizes": [512, 512], "rollout_min_epoch": 20, "rollout_max_epoch": 100, "rollout_min_length": 1, "rollout_max_length": 25}; `mbpo_batch1024.json` = {"batch_size": 1024}; `mbpo_batch512.json` = {"batch_size": 512}; `mbpo_utd2.json` = {"updates_per_env_step": 2}; `mbpo_utd4.json` = {"updates_per_env_step": 4}; `mbpo_width256.json` = {"hidden_sizes": [256, 256]}; `mbpo_width512.json` = {"hidden_sizes": [512, 512]}
- **PASS** — every override file listed in the inventory equals the corresponding fields of the logged `algo_config` of its runs
`ExperimentConfig` defaults (protocol; MBPO runs use them unchanged, `warmup_steps` = 5000):

```json
{
  "algo_name": "mbpo",
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

Complete `metadata.json` of `results/mbpo/HalfCheetah-v5/seed_331/20260908_194455` (HC-base, seed 331):

```json
{
  "algo_name": "mbpo",
  "env_id": "HalfCheetah-v5",
  "seed": 331,
  "git_commit": "0bbbbb74a6542fff643e5cd3ba7603af8de1fcc0",
  "torch_version": "2.13.0+cu130",
  "device": "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "start_time_utc": "2026-09-08T17:44:55.988580",
  "experiment_config": {
    "algo_name": "mbpo",
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
    "hidden_sizes": [
      1024,
      1024
    ],
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "alpha_lr": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "target_entropy": null,
    "autotune_alpha": true,
    "init_alpha": 0.2,
    "policy_update_delay": 1,
    "updates_per_env_step": 1,
    "batch_size": 256,
    "real_ratio": 0.05,
    "buffer_capacity": 1000000,
    "ensemble_size": 7,
    "num_elites": 5,
    "model_hidden_sizes": [
      200,
      200,
      200,
      200
    ],
    "model_lr": 0.001,
    "model_weight_decays": [
      2.5e-05,
      5e-05,
      7.5e-05,
      0.0001,
      0.0001
    ],
    "model_train_batch_size": 256,
    "model_holdout_ratio": 0.2,
    "model_max_train_epochs": 200,
    "model_train_patience": 5,
    "model_train_freq": 250,
    "deterministic_model": false,
    "rollout_batch_size": 10000,
    "model_retain_epochs": 1,
    "rollout_min_epoch": 20,
    "rollout_max_epoch": 150,
    "rollout_min_length": 1,
    "rollout_max_length": 1
  },
  "codecarbon_version": "3.3.0",
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 1.1444091796875e-05,
    "final_temp_c": 44.0,
    "final_power_w": 28.105,
    "run_start_temp_c": 44.0,
    "run_start_power_w": 26.569,
    "run_end_temp_c": 43.0,
    "run_end_power_w": 26.201
  },
  "end_time_utc": "2026-09-08T18:15:19.484854"
}
```

Complete `metadata.json` of `results/mbpo/Ant-v5/seed_331/20260908_222406` (Ant-base, seed 331):

```json
{
  "algo_name": "mbpo",
  "env_id": "Ant-v5",
  "seed": 331,
  "git_commit": "0bbbbb74a6542fff643e5cd3ba7603af8de1fcc0",
  "torch_version": "2.13.0+cu130",
  "device": "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "start_time_utc": "2026-09-08T20:24:06.699630",
  "experiment_config": {
    "algo_name": "mbpo",
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
    "hidden_sizes": [
      1024,
      1024
    ],
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "alpha_lr": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "target_entropy": null,
    "autotune_alpha": true,
    "init_alpha": 0.2,
    "policy_update_delay": 1,
    "updates_per_env_step": 1,
    "batch_size": 256,
    "real_ratio": 0.05,
    "buffer_capacity": 1000000,
    "ensemble_size": 7,
    "num_elites": 5,
    "model_hidden_sizes": [
      200,
      200,
      200,
      200
    ],
    "model_lr": 0.001,
    "model_weight_decays": [
      2.5e-05,
      5e-05,
      7.5e-05,
      0.0001,
      0.0001
    ],
    "model_train_batch_size": 256,
    "model_holdout_ratio": 0.2,
    "model_max_train_epochs": 200,
    "model_train_patience": 5,
    "model_train_freq": 250,
    "deterministic_model": false,
    "rollout_batch_size": 10000,
    "model_retain_epochs": 1,
    "rollout_min_epoch": 20,
    "rollout_max_epoch": 100,
    "rollout_min_length": 1,
    "rollout_max_length": 25
  },
  "codecarbon_version": "3.3.0",
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 9.775161743164062e-06,
    "final_temp_c": 43.0,
    "final_power_w": 26.017,
    "run_start_temp_c": 43.0,
    "run_start_power_w": 26.135,
    "run_end_temp_c": 42.0,
    "run_end_power_w": 26.714
  },
  "end_time_utc": "2026-09-08T21:05:33.672401"
}
```

**2.2 Verbatim code** (`algorithms/*.py` at HEAD; each file is identical at every run commit, Section 0).

`algorithms/mbpo.py` (complete):

```python
   1  """
   2  Model-Based Policy Optimization (Janner, Fu, Zhang & Levine, 2019, "When to
   3  Trust Your Model: Model-Based Policy Optimization", NeurIPS 2019).
   4  
   5  MBPO trains a probabilistic ensemble dynamics model (algorithms/dynamics_model.py)
   6  on real environment transitions, uses it to generate short "branched" rollouts
   7  starting from real states (Section 3.3), and trains SAC (Haarnoja et al. 2018)
   8  on a mix of real and model-generated transitions. This module reuses
   9  `SACAgent` from algorithms/sac.py completely unmodified as the policy
  10  optimizer -- the only MBPO-specific addition on the policy-learning side is
  11  `_MixedReplayBuffer`, which presents the same `.sample()` interface as
  12  `ReplayBuffer` so `SACAgent.update()` doesn't need to know its data is mixed.
  13  
  14  Segment structure (see train() at the bottom), matching the naming used in
  15  README.md's "Extending to PPO / MBPO / PETS" section:
  16    - "rollout": env.step() + action selection + real buffer insertion (identical to SAC)
  17    - "dynamics_model_update": retraining the ensemble on the real buffer
  18    - "synthetic_rollout_generation": branched model rollouts filling the model buffer
  19    - "gradient_updates": SAC updates on mixed real/model batches, sub-timed into
  20      "buffer_sample" / "critic_update" / "actor_update" / "target_update" exactly
  21      as in algorithms/sac.py (see that module's docstring and README.md for why
  22      this sub-split is a time-proportional allocation, not an independent
  23      hardware measurement).
  24  
  25  Unlike SAC, "dynamics_model_update" and "synthetic_rollout_generation" are
  26  directly-measured CodeCarbon tasks in their own right (not time-allocated),
  27  since they are algorithmically and often computationally distinct enough from
  28  "gradient_updates" that a shared-power approximation between them would be
  29  weaker than the one already used for the four SAC sub-segments.
  30  """
  31  from __future__ import annotations
  32  
  33  import time
  34  from typing import Dict, Optional
  35  
  36  import numpy as np
  37  import torch
  38  
  39  from algorithms.dynamics_model import EnsembleDynamicsModel
  40  from algorithms.replay_buffer import ReplayBuffer
  41  from algorithms.sac import SACAgent
  42  from algorithms.termination_fns import get_termination_fn
  43  from algorithms.tracker_utils import TrackerTask
  44  from configs.config import MBPOConfig, ExperimentConfig
  45  
  46  
  47  class _MixedReplayBuffer:
  48      """Presents a `.sample(batch_size)` / `__len__` interface identical to
  49      ReplayBuffer, drawing `real_ratio` of each minibatch from the real
  50      environment buffer and the remainder from the model (branched-rollout)
  51      buffer -- the mechanism (Janner et al. 2019, Sec 3.3) that lets
  52      algorithms/sac.py's SACAgent.update() be reused completely unmodified."""
  53  
  54      def __init__(self, real_buffer: ReplayBuffer, model_buffer: ReplayBuffer, real_ratio: float):
  55          self.real_buffer = real_buffer
  56          self.model_buffer = model_buffer
  57          self.real_ratio = real_ratio
  58  
  59      def __len__(self):
  60          return len(self.real_buffer) + len(self.model_buffer)
  61  
  62      def sample(self, batch_size: int):
  63          if len(self.model_buffer) == 0:
  64              # Before the first model rollout batch exists (start of training),
  65              # fall back to pure real data rather than starving the SAC update.
  66              return self.real_buffer.sample(batch_size)
  67  
  68          n_real = min(max(int(round(batch_size * self.real_ratio)), 0), batch_size)
  69          n_model = batch_size - n_real
  70  
  71          real = self.real_buffer.sample(n_real) if n_real > 0 else None
  72          model = self.model_buffer.sample(n_model) if n_model > 0 else None
  73  
  74          if real is None:
  75              return model
  76          if model is None:
  77              return real
  78          return tuple(np.concatenate([r, m], axis=0) for r, m in zip(real, model))
  79  
  80  
  81  def _rollout_length(epoch: int, cfg: MBPOConfig) -> int:
  82      """Linear schedule for the branched model-rollout length k, per Janner et
  83      al. 2019 Table 2 / Appendix A."""
  84      if epoch <= cfg.rollout_min_epoch:
  85          return cfg.rollout_min_length
  86      if epoch >= cfg.rollout_max_epoch:
  87          return cfg.rollout_max_length
  88      span = max(1, cfg.rollout_max_epoch - cfg.rollout_min_epoch)
  89      frac = (epoch - cfg.rollout_min_epoch) / span
  90      length = cfg.rollout_min_length + frac * (cfg.rollout_max_length - cfg.rollout_min_length)
  91      return int(round(length))
  92  
  93  
  94  def _generate_model_rollouts(
  95      agent: SACAgent,
  96      model: EnsembleDynamicsModel,
  97      real_buffer: ReplayBuffer,
  98      model_buffer: ReplayBuffer,
  99      cfg: MBPOConfig,
 100      rollout_length: int,
 101      device: torch.device,
 102      termination_fn,
 103  ) -> int:
 104      """Branches `rollout_batch_size` k-step imagined trajectories off real
 105      states sampled from `real_buffer`, using the current policy to act and
 106      the dynamics ensemble to predict transitions, and stores every generated
 107      transition in `model_buffer` (Janner et al. 2019, Algorithm 1 lines 6-9).
 108      Each partial trajectory stops early once its termination function fires.
 109      Returns the number of synthetic transitions generated."""
 110      n_start = min(cfg.rollout_batch_size, len(real_buffer))
 111      if n_start == 0:
 112          return 0
 113  
 114      idx = np.random.randint(0, len(real_buffer), size=n_start)
 115      obs = real_buffer.obs[idx].copy()
 116      n_generated = 0
 117  
 118      for _ in range(rollout_length):
 119          obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
 120          with torch.no_grad():
 121              act_t, _ = agent.actor(obs_t, deterministic=False, with_logprob=False)
 122              next_obs_t, rew_t = model.predict(obs_t, act_t)
 123  
 124          act = act_t.cpu().numpy()
 125          next_obs = next_obs_t.cpu().numpy()
 126          rew = rew_t.cpu().numpy()
 127          done = termination_fn(next_obs)
 128  
 129          model_buffer.add_batch(obs, act, rew, next_obs, done)
 130          n_generated += obs.shape[0]
 131  
 132          keep = ~done
 133          if not np.any(keep):
 134              break
 135          obs = next_obs[keep]
 136  
 137      return n_generated
 138  
 139  
 140  def train(
 141      env,
 142      mbpo_cfg: MBPOConfig,
 143      exp_cfg: ExperimentConfig,
 144      tracker,
 145      device: torch.device,
 146      logger,
 147      steps_per_epoch: int = 1000,
 148  ):
 149      """
 150      Runs MBPO warmup + measured training. Returns (agent, energy_log, metrics)
 151      with the same shapes as algorithms.sac.train() (see that function's
 152      docstring), plus MBPO-specific fields on each "epochs" row: rollout_length,
 153      model_buffer_size, model_holdout_mse, model_train_epochs.
 154  
 155      `rollout`, `dynamics_model_update`, `synthetic_rollout_generation`, and
 156      the combined `gradient_updates` bucket are real CodeCarbon measurements
 157      (one task per epoch, unique-named, per the pattern in algorithms/sac.py
 158      and README.md). `buffer_sample`, `critic_update`, `actor_update`,
 159      `target_update` are obtained by allocating `gradient_updates`' energy
 160      proportionally to wall-clock time share, exactly as in SAC.
 161      """
 162      obs_dim = env.observation_space.shape[0]
 163      act_dim = env.action_space.shape[0]
 164      act_limit = float(env.action_space.high[0])
 165  
 166      agent = SACAgent(obs_dim, act_dim, act_limit, mbpo_cfg, device)
 167      model = EnsembleDynamicsModel(obs_dim, act_dim, mbpo_cfg, device)
 168      termination_fn = get_termination_fn(exp_cfg.env_id, logger)
 169  
 170      real_buffer = ReplayBuffer(mbpo_cfg.buffer_capacity, obs_dim, act_dim)
 171      # Sized to hold `model_retain_epochs` epochs' worth of rollouts at the
 172      # longest scheduled rollout length; a plain circular buffer then
 173      # naturally approximates "retain only the most recent N epochs of model
 174      # data" (Janner et al. 2019 Sec 3.3) as older entries get overwritten.
 175      model_buffer_capacity = max(
 176          10_000, mbpo_cfg.rollout_batch_size * mbpo_cfg.rollout_max_length * mbpo_cfg.model_retain_epochs
 177      )
 178      model_buffer = ReplayBuffer(model_buffer_capacity, obs_dim, act_dim)
 179      mixed_buffer = _MixedReplayBuffer(real_buffer, model_buffer, mbpo_cfg.real_ratio)
 180  
 181      energy_log: Dict[str, float] = {}
 182      sub_time_totals = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
 183  
 184      episodes_log = []
 185      episode_idx = 0
 186      episode_return = 0.0
 187      episode_length = 0
 188  
 189      def _record_episode(phase, epoch, env_step):
 190          nonlocal episode_idx, episode_return, episode_length
 191          episodes_log.append({
 192              "episode_idx": episode_idx,
 193              "phase": phase,
 194              "epoch": epoch,
 195              "env_step": env_step,
 196              "return": episode_return,
 197              "length": episode_length,
 198          })
 199          episode_idx += 1
 200          episode_return = 0.0
 201          episode_length = 0
 202  
 203      obs, _ = env.reset(seed=exp_cfg.seed)
 204  
 205      # ---------------- warmup (random policy, fills real buffer; excluded from analysis segments) ----------------
 206      logger.info("Starting warmup: %d steps", exp_cfg.warmup_steps)
 207      with TrackerTask(tracker, "warmup", 0, energy_log):
 208          for warmup_step in range(exp_cfg.warmup_steps):
 209              action = env.action_space.sample()
 210              next_obs, reward, terminated, truncated, _ = env.step(action)
 211              real_buffer.add(obs, action, reward, next_obs, terminated)
 212              episode_return += reward
 213              episode_length += 1
 214              obs = next_obs
 215              if terminated or truncated:
 216                  _record_episode("warmup", None, warmup_step + 1)
 217                  obs, _ = env.reset()
 218  
 219      # ---------------- measured training ----------------
 220      n_epochs = max(1, exp_cfg.train_steps // steps_per_epoch)
 221      logger.info("Starting measured training: %d epochs x %d steps = %d env steps",
 222                  n_epochs, steps_per_epoch, n_epochs * steps_per_epoch)
 223  
 224      epochs_log = []
 225      cumulative_reward = 0.0
 226      global_step = 0
 227      last_model_train_step = -mbpo_cfg.model_train_freq  # forces a retrain before the first epoch's rollouts
 228  
 229      for epoch in range(n_epochs):
 230          # --- dynamics_model_update: retrain the ensemble on all real data seen so far ---
 231          model_diag = {"holdout_mse": None, "train_epochs": None}
 232          if (global_step - last_model_train_step) >= mbpo_cfg.model_train_freq:
 233              with TrackerTask(tracker, "dynamics_model_update", epoch, energy_log):
 234                  model_diag = model.fit(real_buffer)
 235              last_model_train_step = global_step
 236  
 237          # --- rollout block (real env; identical in structure to SAC) ---
 238          epoch_reward_sum = 0.0
 239          epoch_episode_returns = []
 240          with TrackerTask(tracker, "rollout", epoch, energy_log):
 241              for _ in range(steps_per_epoch):
 242                  action = agent.select_action(obs, deterministic=False)
 243                  next_obs, reward, terminated, truncated, _ = env.step(action)
 244                  real_buffer.add(obs, action, reward, next_obs, terminated)
 245                  episode_return += reward
 246                  episode_length += 1
 247                  cumulative_reward += reward
 248                  epoch_reward_sum += reward
 249                  global_step += 1
 250                  obs = next_obs
 251                  if terminated or truncated:
 252                      epoch_episode_returns.append(episode_return)
 253                      _record_episode("train", epoch, global_step)
 254                      obs, _ = env.reset()
 255  
 256          # --- synthetic_rollout_generation: branched model rollouts ---
 257          rollout_length = _rollout_length(epoch, mbpo_cfg)
 258          with TrackerTask(tracker, "synthetic_rollout_generation", epoch, energy_log):
 259              n_synthetic = _generate_model_rollouts(
 260                  agent, model, real_buffer, model_buffer, mbpo_cfg, rollout_length, device, termination_fn
 261              )
 262  
 263          # --- gradient_updates block (SAC updates on mixed real/model batches) ---
 264          epoch_sub_times = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
 265          critic_losses, actor_losses, alphas = [], [], []
 266          n_updates = steps_per_epoch * mbpo_cfg.updates_per_env_step
 267          with TrackerTask(tracker, "gradient_updates", epoch, energy_log):
 268              for _ in range(n_updates):
 269                  if len(mixed_buffer) < mbpo_cfg.batch_size:
 270                      break
 271                  info = agent.update(mixed_buffer, mbpo_cfg.batch_size)
 272                  epoch_sub_times["buffer_sample"] += info.buffer_sample_s
 273                  epoch_sub_times["critic_update"] += info.critic_update_s
 274                  epoch_sub_times["actor_update"] += info.actor_update_s
 275                  epoch_sub_times["target_update"] += info.target_update_s
 276                  if info.critic_loss is not None:
 277                      critic_losses.append(info.critic_loss)
 278                  if info.actor_loss is not None:
 279                      actor_losses.append(info.actor_loss)
 280                  if info.alpha is not None:
 281                      alphas.append(info.alpha)
 282  
 283          # Sub-segment times accumulate across all epochs and are reconciled against the
 284          # total measured "gradient_updates" energy once, after the loop (see below) --
 285          # identical scheme to algorithms/sac.py.
 286          for k in sub_time_totals:
 287              sub_time_totals[k] += epoch_sub_times[k]
 288  
 289          epochs_log.append({
 290              "epoch": epoch,
 291              "env_step": global_step,
 292              "cumulative_reward": cumulative_reward,
 293              "epoch_reward_sum": epoch_reward_sum,
 294              "epoch_reward_mean_per_step": epoch_reward_sum / steps_per_epoch,
 295              "num_episodes_completed": len(epoch_episode_returns),
 296              "mean_episode_return": (
 297                  sum(epoch_episode_returns) / len(epoch_episode_returns)
 298                  if epoch_episode_returns else None
 299              ),
 300              "critic_loss_mean": (sum(critic_losses) / len(critic_losses)) if critic_losses else None,
 301              "actor_loss_mean": (sum(actor_losses) / len(actor_losses)) if actor_losses else None,
 302              "alpha_end": alphas[-1] if alphas else None,
 303              "buffer_size": len(real_buffer),
 304              "rollout_length": rollout_length,
 305              "synthetic_transitions_generated": n_synthetic,
 306              "model_buffer_size": len(model_buffer),
 307              "model_holdout_mse": model_diag["holdout_mse"],
 308              "model_train_epochs": model_diag["train_epochs"],
 309          })
 310  
 311          if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
 312              logger.info(
 313                  "Epoch %d/%d done (real_buffer=%d, model_buffer=%d, rollout_len=%d, "
 314                  "cumulative_reward=%.2f, mean_episode_return=%s)",
 315                  epoch + 1, n_epochs, len(real_buffer), len(model_buffer), rollout_length,
 316                  cumulative_reward, epochs_log[-1]["mean_episode_return"],
 317              )
 318  
 319      # record a trailing partial episode (if training ended mid-episode) so no reward is lost
 320      if episode_length > 0:
 321          _record_episode("train_incomplete", n_epochs - 1, global_step)
 322  
 323      # ---------------- reconcile: split total gradient_updates energy by aggregate time share ----------------
 324      total_gradient_energy = energy_log.pop("gradient_updates", 0.0)
 325      total_time = sum(sub_time_totals.values())
 326      for k, t in sub_time_totals.items():
 327          share = (t / total_time) if total_time > 0 else 0.0
 328          energy_log[k] = total_gradient_energy * share
 329  
 330      energy_log["_sub_segment_wall_time_seconds"] = sub_time_totals
 331      metrics = {"episodes": episodes_log, "epochs": epochs_log}
 332      return agent, energy_log, metrics
```
`algorithms/dynamics_model.py` (complete):

```python
   1  """
   2  Probabilistic ensemble dynamics model for MBPO (Janner, Fu, Zhang & Levine,
   3  2019, "When to Trust Your Model: Model-Based Policy Optimization", NeurIPS
   4  2019), using the network design from Chua et al. (2018, "Deep Reinforcement
   5  Learning in a Handful of Trials using Probabilistic Dynamics Models" / PETS)
   6  that the MBPO reference implementation (https://github.com/JannerM/mbpo)
   7  reuses directly.
   8  
   9  Each ensemble member is an MLP predicting a diagonal Gaussian
  10  N(mean, diag(exp(logvar))) over the target [delta_obs, reward], given
  11  [obs, act] as input. The ensemble is trained by penalized Gaussian
  12  negative-log-likelihood with learned, softly-bounded log-variance limits
  13  (Chua et al. 2018, Appendix A.1), using a held-out validation split for
  14  early stopping. At rollout time, predictions are drawn from a randomly
  15  selected "elite" member per sample (the `num_elites` members with the lowest
  16  validation error) -- this bootstrap-ensemble mechanism is what MBPO relies on
  17  to keep short model rollouts from compounding a single member's bias
  18  (Janner et al. 2019, Section 3.1).
  19  """
  20  from __future__ import annotations
  21  
  22  from typing import Dict, Tuple
  23  
  24  import numpy as np
  25  import torch
  26  import torch.nn as nn
  27  import torch.nn.functional as F
  28  
  29  from algorithms.replay_buffer import ReplayBuffer
  30  
  31  
  32  class _Swish(nn.Module):
  33      def forward(self, x):
  34          return x * torch.sigmoid(x)
  35  
  36  
  37  class GaussianEnsembleMLP(nn.Module):
  38      """`ensemble_size` independent MLPs, each predicting a diagonal Gaussian
  39      over [delta_obs, reward] given [obs, act]. Kept as separate members
  40      (rather than a batched/vmapped implementation) for readability, matching
  41      the twin-Q style already used in algorithms/sac.py."""
  42  
  43      def __init__(self, obs_dim: int, act_dim: int, ensemble_size: int, hidden_sizes: Tuple[int, ...]):
  44          super().__init__()
  45          in_dim = obs_dim + act_dim
  46          out_dim = obs_dim + 1  # predicted delta_obs (obs_dim dims) + reward (1 dim)
  47          self.obs_dim = obs_dim
  48          self.out_dim = out_dim
  49          self.ensemble_size = ensemble_size
  50  
  51          self.trunks = nn.ModuleList()
  52          for _ in range(ensemble_size):
  53              layers = []
  54              prev = in_dim
  55              for h in hidden_sizes:
  56                  layers += [nn.Linear(prev, h), _Swish()]
  57                  prev = h
  58              self.trunks.append(nn.Sequential(*layers))
  59  
  60          self.mean_heads = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_dim) for _ in range(ensemble_size)])
  61          self.logvar_heads = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_dim) for _ in range(ensemble_size)])
  62  
  63          # Learned, softly-clamped log-variance bounds (Chua et al. 2018, Appendix A.1):
  64          # keeps predicted variance from collapsing to 0 or exploding early in training.
  65          self.max_logvar = nn.Parameter(torch.full((out_dim,), 0.5))
  66          self.min_logvar = nn.Parameter(torch.full((out_dim,), -10.0))
  67  
  68      def _bound_logvar(self, logvar):
  69          logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
  70          logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
  71          return logvar
  72  
  73      def forward_member(self, member_idx: int, obs_act: torch.Tensor):
  74          h = self.trunks[member_idx](obs_act)
  75          mean = self.mean_heads[member_idx](h)
  76          logvar = self._bound_logvar(self.logvar_heads[member_idx](h))
  77          return mean, logvar
  78  
  79      def forward_all(self, obs_act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  80          """Returns (means, logvars), each of shape [ensemble_size, batch, out_dim]."""
  81          means, logvars = [], []
  82          for i in range(self.ensemble_size):
  83              mean, logvar = self.forward_member(i, obs_act)
  84              means.append(mean)
  85              logvars.append(logvar)
  86          return torch.stack(means, dim=0), torch.stack(logvars, dim=0)
  87  
  88  
  89  class RunningNormalizer(nn.Module):
  90      """Standardizes [obs, act] inputs to the dynamics model using the mean/std
  91      of the current training split. Refit every time the ensemble is retrained
  92      (MBPO/PETS convention) -- keeps the model's input scale stable as the
  93      replay buffer grows and its distribution shifts over training."""
  94  
  95      def __init__(self, dim: int):
  96          super().__init__()
  97          self.register_buffer("mean", torch.zeros(dim))
  98          self.register_buffer("std", torch.ones(dim))
  99  
 100      def fit(self, x: torch.Tensor):
 101          with torch.no_grad():
 102              self.mean.copy_(x.mean(dim=0))
 103              std = x.std(dim=0)
 104              std[std < 1e-12] = 1.0
 105              self.std.copy_(std)
 106  
 107      def forward(self, x):
 108          return (x - self.mean) / self.std
 109  
 110  
 111  class EnsembleDynamicsModel:
 112      """Owns a GaussianEnsembleMLP plus its training loop (penalized Gaussian
 113      NLL with holdout early stopping, per-layer weight decay matching the
 114      PETS/MBPO reference code) and elite-based prediction for model rollouts."""
 115  
 116      def __init__(self, obs_dim: int, act_dim: int, cfg, device: torch.device):
 117          self.obs_dim = obs_dim
 118          self.act_dim = act_dim
 119          self.cfg = cfg
 120          self.device = device
 121  
 122          hidden_sizes = cfg.model_hidden_sizes
 123          decays = cfg.model_weight_decays
 124          if len(decays) != len(hidden_sizes) + 1:
 125              raise ValueError(
 126                  f"MBPOConfig.model_weight_decays must have len(model_hidden_sizes) + 1 = "
 127                  f"{len(hidden_sizes) + 1} entries (one per hidden layer, one for the output "
 128                  f"heads), got {len(decays)}."
 129              )
 130  
 131          self.net = GaussianEnsembleMLP(obs_dim, act_dim, cfg.ensemble_size, hidden_sizes).to(device)
 132          self.normalizer = RunningNormalizer(obs_dim + act_dim).to(device)
 133  
 134          param_groups = []
 135          for i in range(cfg.ensemble_size):
 136              layer_linears = [m for m in self.net.trunks[i] if isinstance(m, nn.Linear)]
 137              for linear, wd in zip(layer_linears, decays[:-1]):
 138                  param_groups.append({"params": linear.parameters(), "weight_decay": wd})
 139              param_groups.append({"params": self.net.mean_heads[i].parameters(), "weight_decay": decays[-1]})
 140              param_groups.append({"params": self.net.logvar_heads[i].parameters(), "weight_decay": decays[-1]})
 141          param_groups.append({"params": [self.net.max_logvar, self.net.min_logvar], "weight_decay": 0.0})
 142  
 143          self.optimizer = torch.optim.Adam(param_groups, lr=cfg.model_lr)
 144          # Until the first fit() call, treat the first `num_elites` members as elites
 145          # (arbitrary but harmless: warmup fills the real buffer before any model
 146          # rollout is ever generated, so fit() always runs at least once first).
 147          self.elite_indices = list(range(cfg.num_elites))
 148  
 149      def fit(self, buffer: ReplayBuffer) -> Dict[str, float]:
 150          """Trains the full ensemble from scratch on the current contents of
 151          `buffer` (MBPO retrains periodically on *all* real data seen so far,
 152          not just a minibatch -- see Janner et al. 2019 Algorithm 1, line 4).
 153          Each member is trained on its own bootstrap resample of the training
 154          split (Chua et al. 2018 Appendix A.1). Returns diagnostics for logging.
 155          """
 156          n = buffer.size
 157          obs = buffer.obs[:n]
 158          act = buffer.actions[:n]
 159          rew = buffer.rewards[:n, 0]
 160          next_obs = buffer.next_obs[:n]
 161  
 162          inputs = np.concatenate([obs, act], axis=1)
 163          targets = np.concatenate([next_obs - obs, rew[:, None]], axis=1)
 164  
 165          n_holdout = max(1, int(n * self.cfg.model_holdout_ratio))
 166          perm = np.random.permutation(n)
 167          holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
 168  
 169          inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
 170          targets_t = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
 171  
 172          self.normalizer.fit(inputs_t[train_idx])
 173  
 174          n_train = len(train_idx)
 175          # Bootstrap resample per ensemble member (with replacement), fixed for this fit() call.
 176          bootstrap_idx = np.random.randint(0, n_train, size=(self.cfg.ensemble_size, n_train))
 177          member_train_idx = train_idx[bootstrap_idx]  # [ensemble_size, n_train]
 178  
 179          best_holdout_mse = None
 180          epochs_since_improved = 0
 181          epochs_run = 0
 182          batch_size = self.cfg.model_train_batch_size
 183  
 184          for epoch in range(self.cfg.model_max_train_epochs):
 185              epochs_run = epoch + 1
 186              # shuffle each member's bootstrap sample independently
 187              for i in range(self.cfg.ensemble_size):
 188                  np.random.shuffle(member_train_idx[i])
 189  
 190              for start in range(0, n_train, batch_size):
 191                  end = start + batch_size
 192                  total_loss = 0.0
 193                  self.optimizer.zero_grad(set_to_none=True)
 194                  for i in range(self.cfg.ensemble_size):
 195                      idx = member_train_idx[i, start:end]
 196                      x = self.normalizer(inputs_t[idx])
 197                      y = targets_t[idx]
 198                      mean, logvar = self.net.forward_member(i, x)
 199                      inv_var = torch.exp(-logvar)
 200                      nll = (((mean - y) ** 2) * inv_var + logvar).mean()
 201                      total_loss = total_loss + nll
 202                  total_loss = total_loss + 0.01 * (
 203                      self.net.max_logvar.sum() - self.net.min_logvar.sum()
 204                  )
 205                  total_loss.backward()
 206                  self.optimizer.step()
 207  
 208              holdout_mse = self._holdout_mse(inputs_t[holdout_idx], targets_t[holdout_idx])
 209              mean_holdout_mse = float(holdout_mse.mean())
 210              if best_holdout_mse is None or mean_holdout_mse < best_holdout_mse - 1e-4:
 211                  best_holdout_mse = mean_holdout_mse
 212                  best_per_member = holdout_mse
 213                  epochs_since_improved = 0
 214              else:
 215                  epochs_since_improved += 1
 216                  if epochs_since_improved >= self.cfg.model_train_patience:
 217                      break
 218  
 219          self.elite_indices = list(
 220              np.argsort(best_per_member.cpu().numpy())[: self.cfg.num_elites]
 221          )
 222  
 223          return {
 224              "holdout_mse": best_holdout_mse,
 225              "train_epochs": epochs_run,
 226              "train_transitions": int(n),
 227          }
 228  
 229      @torch.no_grad()
 230      def _holdout_mse(self, inputs_t: torch.Tensor, targets_t: torch.Tensor) -> torch.Tensor:
 231          """Per-member MSE (not NLL) on the holdout split -- used for both early
 232          stopping and elite selection, following the PETS/MBPO convention of
 233          ranking ensemble members by plain prediction error rather than
 234          likelihood (which the learned logvar could otherwise game)."""
 235          x = self.normalizer(inputs_t)
 236          means, _ = self.net.forward_all(x)  # [ensemble_size, n_holdout, out_dim]
 237          return ((means - targets_t.unsqueeze(0)) ** 2).mean(dim=(1, 2))
 238  
 239      @torch.no_grad()
 240      def predict(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
 241          """For each row in the batch, samples a random *elite* ensemble member
 242          (bootstrap ensemble prediction, Chua et al. 2018 Sec 3.2 / Janner et
 243          al. 2019 Sec 3.1) and returns (next_obs, reward)."""
 244          batch_size = obs.shape[0]
 245          obs_act = torch.cat([obs, act], dim=-1)
 246          obs_act_n = self.normalizer(obs_act)
 247  
 248          means, logvars = self.net.forward_all(obs_act_n)  # [ensemble_size, batch, out_dim]
 249          elite = torch.as_tensor(self.elite_indices, device=self.device)
 250          choice = elite[torch.randint(0, len(elite), (batch_size,), device=self.device)]
 251          rows = torch.arange(batch_size, device=self.device)
 252  
 253          mean = means[choice, rows]
 254          logvar = logvars[choice, rows]
 255  
 256          if self.cfg.deterministic_model:
 257              sample = mean
 258          else:
 259              std = torch.exp(0.5 * logvar)
 260              sample = mean + std * torch.randn_like(mean)
 261  
 262          delta_obs = sample[:, : self.obs_dim]
 263          reward = sample[:, self.obs_dim]
 264          next_obs = obs + delta_obs
 265          return next_obs, reward
```
`algorithms/termination_fns.py` (complete):

```python
   1  """
   2  Environment-specific termination ("done") heuristics for model-generated
   3  transitions, matching the per-environment `termination_fn`s from the MBPO
   4  reference implementation (Janner et al., 2019,
   5  https://github.com/JannerM/mbpo/tree/master/mbpo/static), which in turn
   6  mirror each Gymnasium MuJoCo environment's built-in `terminate_when_unhealthy`
   7  early-termination checks.
   8  
   9  The learned dynamics model (algorithms/dynamics_model.py) only predicts
  10  (next_obs, reward) -- it has no notion of "the robot fell over". Without a
  11  termination function, branched model rollouts would keep "walking" through
  12  physically invalid states (e.g. a fallen-over Hopper) instead of ending the
  13  episode there, which the paper's ablations show significantly hurts
  14  performance on environments that have early termination in the real env.
  15  
  16  These assume the Gymnasium MuJoCo default observation layout
  17  (`exclude_current_positions_from_observation=True`, the default for all
  18  `gym.make(...)` calls here), under which index 0 of the observation is the
  19  torso height ("z") for the walking/running envs below.
  20  """
  21  from __future__ import annotations
  22  
  23  import numpy as np
  24  
  25  
  26  def _never_done(next_obs: np.ndarray) -> np.ndarray:
  27      return np.zeros((next_obs.shape[0],), dtype=bool)
  28  
  29  
  30  def _hopper_done(next_obs: np.ndarray) -> np.ndarray:
  31      height, angle = next_obs[:, 0], next_obs[:, 1]
  32      healthy = (
  33          np.isfinite(next_obs).all(axis=-1)
  34          & (np.abs(next_obs[:, 1:]) < 100).all(axis=-1)
  35          & (height > 0.7)
  36          & (np.abs(angle) < 0.2)
  37      )
  38      return ~healthy
  39  
  40  
  41  def _walker2d_done(next_obs: np.ndarray) -> np.ndarray:
  42      height, angle = next_obs[:, 0], next_obs[:, 1]
  43      healthy = (height > 0.8) & (height < 2.0) & (angle > -1.0) & (angle < 1.0)
  44      return ~healthy
  45  
  46  
  47  def _ant_done(next_obs: np.ndarray) -> np.ndarray:
  48      z = next_obs[:, 0]
  49      healthy = np.isfinite(next_obs).all(axis=-1) & (z >= 0.2) & (z <= 1.0)
  50      return ~healthy
  51  
  52  
  53  def _humanoid_done(next_obs: np.ndarray) -> np.ndarray:
  54      z = next_obs[:, 0]
  55      return (z < 1.0) | (z > 2.0)
  56  
  57  
  58  # Matched against env_id by lowercase substring (see get_termination_fn).
  59  # HalfCheetah/InvertedPendulum/InvertedDoublePendulum never terminate early
  60  # in Gymnasium, so `_never_done` is the *correct* function for them, not a
  61  # fallback.
  62  _REGISTRY = {
  63      "halfcheetah": _never_done,
  64      "hopper": _hopper_done,
  65      "walker2d": _walker2d_done,
  66      "ant": _ant_done,
  67      "humanoid": _humanoid_done,
  68      "invertedpendulum": _never_done,
  69      "inverteddoublependulum": _never_done,
  70  }
  71  
  72  
  73  def get_termination_fn(env_id: str, logger=None):
  74      """Returns a function `next_obs: np.ndarray[N, obs_dim] -> done: np.ndarray[N] (bool)`.
  75  
  76      Falls back to `_never_done` (with a logged warning) for unrecognized
  77      environments -- correct for non-terminating tasks, but a known-imprecise
  78      approximation for anything else (model rollouts on that env will run to
  79      the full scheduled rollout length even through what would be a terminal
  80      state in the real environment)."""
  81      key = env_id.lower()
  82      for name, fn in _REGISTRY.items():
  83          if name in key:
  84              return fn
  85      if logger is not None:
  86          logger.warning(
  87              "No known MBPO termination function for env_id=%s; model rollouts "
  88              "will never terminate early. Add one to algorithms/termination_fns.py "
  89              "if this environment has early termination in the real env.", env_id,
  90          )
  91      return _never_done
```
`algorithms/replay_buffer.py` (complete; used for the real buffer and the model buffer):

```python
   1  """Fixed-capacity circular replay buffer, numpy-backed (fast, no torch overhead
   2  until the sampled batch is converted to tensors at train-step time -- keeps
   3  the 'buffer sample' segment's energy cost isolated from tensor/GPU transfer,
   4  which you may want to sub-segment further later)."""
   5  from __future__ import annotations
   6  
   7  import numpy as np
   8  
   9  
  10  class ReplayBuffer:
  11      def __init__(self, capacity: int, obs_dim: int, act_dim: int):
  12          self.capacity = capacity
  13          self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
  14          self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
  15          self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
  16          self.rewards = np.zeros((capacity, 1), dtype=np.float32)
  17          self.dones = np.zeros((capacity, 1), dtype=np.float32)
  18          self.ptr = 0
  19          self.size = 0
  20  
  21      def add(self, obs, action, reward, next_obs, done):
  22          self.obs[self.ptr] = obs
  23          self.actions[self.ptr] = action
  24          self.rewards[self.ptr, 0] = reward
  25          self.next_obs[self.ptr] = next_obs
  26          self.dones[self.ptr, 0] = float(done)
  27          self.ptr = (self.ptr + 1) % self.capacity
  28          self.size = min(self.size + 1, self.capacity)
  29  
  30      def add_batch(self, obs, action, reward, next_obs, done):
  31          """Vectorized insert of a batch of transitions. Used by MBPO's branched
  32          model rollouts, where thousands of synthetic transitions are generated
  33          per epoch -- inserting them one at a time in a Python loop would
  34          dominate wall-clock cost. Behaves as a no-op for n == 0."""
  35          n = obs.shape[0]
  36          if n == 0:
  37              return
  38          idx = (self.ptr + np.arange(n)) % self.capacity
  39          self.obs[idx] = obs
  40          self.actions[idx] = action
  41          self.rewards[idx, 0] = reward
  42          self.next_obs[idx] = next_obs
  43          self.dones[idx, 0] = done.astype(np.float32)
  44          self.ptr = (self.ptr + n) % self.capacity
  45          self.size = min(self.size + n, self.capacity)
  46  
  47      def sample(self, batch_size: int):
  48          idx = np.random.randint(0, self.size, size=batch_size)
  49          return (
  50              self.obs[idx],
  51              self.actions[idx],
  52              self.rewards[idx],
  53              self.next_obs[idx],
  54              self.dones[idx],
  55          )
  56  
  57      def __len__(self):
  58          return self.size
```
`algorithms/sac.py` `SACAgent` (lines 98–229; the unmodified policy optimiser MBPO reuses; `train()` of sac.py is not used by MBPO):

```python
  98  class SACAgent:
  99      def __init__(self, obs_dim, act_dim, act_limit, cfg: SACConfig, device: torch.device):
 100          self.cfg = cfg
 101          self.device = device
 102          self.act_dim = act_dim
 103  
 104          self.actor = GaussianPolicy(obs_dim, act_dim, cfg.hidden_sizes, act_limit).to(device)
 105          self.q1 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
 106          self.q2 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
 107          self.q1_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
 108          self.q2_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
 109          self.q1_targ.load_state_dict(self.q1.state_dict())
 110          self.q2_targ.load_state_dict(self.q2.state_dict())
 111          for p in self.q1_targ.parameters():
 112              p.requires_grad = False
 113          for p in self.q2_targ.parameters():
 114              p.requires_grad = False
 115  
 116          self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
 117          self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
 118          self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
 119  
 120          self.autotune_alpha = cfg.autotune_alpha
 121          if cfg.autotune_alpha:
 122              self.target_entropy = (
 123                  cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
 124              )
 125              self.log_alpha = torch.tensor(
 126                  np.log(cfg.init_alpha), dtype=torch.float32, requires_grad=True, device=device
 127              )
 128              self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
 129          else:
 130              self.log_alpha = torch.tensor(np.log(cfg.init_alpha), device=device)
 131  
 132          self._update_counter = 0
 133  
 134      @property
 135      def alpha(self):
 136          return self.log_alpha.exp()
 137  
 138      @torch.no_grad()
 139      def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
 140          obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
 141          action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
 142          return action.squeeze(0).cpu().numpy()
 143  
 144      def update(self, buffer: ReplayBuffer, batch_size: int) -> UpdateInfo:
 145          info = UpdateInfo()
 146  
 147          # --- buffer_sample ---
 148          t0 = time.perf_counter()
 149          obs, act, rew, next_obs, done = buffer.sample(batch_size)
 150          obs_t = torch.as_tensor(obs, device=self.device)
 151          act_t = torch.as_tensor(act, device=self.device)
 152          rew_t = torch.as_tensor(rew, device=self.device).squeeze(-1)
 153          next_obs_t = torch.as_tensor(next_obs, device=self.device)
 154          done_t = torch.as_tensor(done, device=self.device).squeeze(-1)
 155          info.buffer_sample_s = time.perf_counter() - t0
 156  
 157          # --- critic_update ---
 158          t0 = time.perf_counter()
 159          with torch.no_grad():
 160              next_action, next_logp = self.actor(next_obs_t)
 161              q1_next = self.q1_targ(next_obs_t, next_action)
 162              q2_next = self.q2_targ(next_obs_t, next_action)
 163              q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
 164              target_q = rew_t + (1.0 - done_t) * self.cfg.gamma * q_next
 165  
 166          q1_pred = self.q1(obs_t, act_t)
 167          q2_pred = self.q2(obs_t, act_t)
 168          q1_loss = F.mse_loss(q1_pred, target_q)
 169          q2_loss = F.mse_loss(q2_pred, target_q)
 170  
 171          self.q1_opt.zero_grad(set_to_none=True)
 172          q1_loss.backward()
 173          self.q1_opt.step()
 174  
 175          self.q2_opt.zero_grad(set_to_none=True)
 176          q2_loss.backward()
 177          self.q2_opt.step()
 178          info.critic_update_s = time.perf_counter() - t0
 179          info.critic_loss = (q1_loss.item() + q2_loss.item()) / 2.0
 180  
 181          # --- actor_update (+ alpha) ---
 182          do_actor_update = (self._update_counter % self.cfg.policy_update_delay) == 0
 183          if do_actor_update:
 184              t0 = time.perf_counter()
 185              for p in self.q1.parameters():
 186                  p.requires_grad = False
 187              for p in self.q2.parameters():
 188                  p.requires_grad = False
 189  
 190              action, logp = self.actor(obs_t)
 191              q1_pi = self.q1(obs_t, action)
 192              q2_pi = self.q2(obs_t, action)
 193              q_pi = torch.min(q1_pi, q2_pi)
 194              actor_loss = (self.alpha.detach() * logp - q_pi).mean()
 195  
 196              self.actor_opt.zero_grad(set_to_none=True)
 197              actor_loss.backward()
 198              self.actor_opt.step()
 199  
 200              for p in self.q1.parameters():
 201                  p.requires_grad = True
 202              for p in self.q2.parameters():
 203                  p.requires_grad = True
 204  
 205              if self.autotune_alpha:
 206                  alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
 207                  self.alpha_opt.zero_grad(set_to_none=True)
 208                  alpha_loss.backward()
 209                  self.alpha_opt.step()
 210  
 211              info.actor_update_s = time.perf_counter() - t0
 212              info.actor_loss = actor_loss.item()
 213              info.alpha = self.alpha.item()
 214  
 215          # --- target_update ---
 216          t0 = time.perf_counter()
 217          with torch.no_grad():
 218              for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
 219                  p_targ.data.mul_(1 - self.cfg.tau)
 220                  p_targ.data.add_(self.cfg.tau * p.data)
 221              for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
 222                  p_targ.data.mul_(1 - self.cfg.tau)
 223                  p_targ.data.add_(self.cfg.tau * p.data)
 224          info.target_update_s = time.perf_counter() - t0
 225  
 226          self._update_counter += 1
 227          return info
 228  
 229
```

**2.3 Policy / critic network sizes** (real `GaussianPolicy` / `QNetwork` of `sac.py`, as used through `SACAgent`):


| env | hidden | obs/act dim | actor (incl. μ and log σ heads) | one Q-network | both critics | both target critics | all 5 networks |
|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | (256,256) | 17/6 | 73484 | 72193 | 144386 | 144386 | 362256 |
| HalfCheetah-v5 | (512,512) | 17/6 | 278028 | 275457 | 550914 | 550914 | 1379856 |
| HalfCheetah-v5 | (1024,1024) | 17/6 | 1080332 | 1075201 | 2150402 | 2150402 | 5381136 |
| Ant-v5 | (256,256) | 105/8 | 97040 | 95233 | 190466 | 190466 | 477972 |
| Ant-v5 | (512,512) | 105/8 | 325136 | 321537 | 643074 | 643074 | 1611284 |
| Ant-v5 | (1024,1024) | 105/8 | 1174544 | 1167361 | 2334722 | 2334722 | 5843988 |

**2.4 Dynamics ensemble sizes** (`GaussianEnsembleMLP(obs_dim, act_dim, ensemble_size=7, hidden_sizes=(200,200,200,200))` instantiated; input = obs+act, output = obs+1 (Δobs and reward) per head; per member: trunk + mean head + log-variance head; plus the shared learned `max_logvar`/`min_logvar` vectors (2 × (obs+1)); the running normaliser has buffers, no parameters; elites = 5 of 7 members, chosen by holdout MSE):


| env | input → output dim | trunk (one member) | mean head | logvar head | parameters per member | members | shared logvar bounds | total parameters (all 7 members + bounds) | elites | parameters of the 5 elites (members only) |
|---|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | 17+6=23 → 18 | 125400 | 3618 | 3618 | 132636 | 7 | 36 | 928488 | 5 | 663180 |
| Ant-v5 | 105+8=113 → 106 | 143400 | 21306 | 21306 | 186012 | 7 | 212 | 1302296 | 5 | 930060 |

- **PASS** — total = 7 × per-member + shared bounds

**2.5 Per-call FLOPs from `flop_analysis/flops_per_call.json`** (`measure_flops.measure_sac` for the SAC part, `measure_mbpo_dynamics` for the model; measured on `cpu`; matmul FLOPs = 2·m·n·k). Cell `HC ‖ Ant` (rollout1/rollout15 share the Ant baseline's signature, rollout length is not part of the architecture signature):


| config | actor_forward_bs1 (rollout call) | critic_fwdbwd (per update() call) | actor_fwdbwd (per update() call) | target_update_elementwise_ops (per update() call) | dynamics_member_fwdbwd (one member, batch 256) | dynamics_ensemble_forward_all_bs1 (7 members, 1 sample) | SAC batch / model batch | per synthetic sample = actor_forward_bs1 + ensemble_forward_all_bs1 |
|---|---|---|---|---|---|---|---|---|
| base | 2.15654e+06 ‖ 2.34496e+06 | 4.92359e+09 ‖ 5.25494e+09 | 3.84513e+09 ‖ 4.13244e+09 | 4.30080e+06 ‖ 4.66944e+06 | 2.00090e+08 ‖ 2.72589e+08 | 1.84520e+06 ‖ 2.59000e+06 | 256/256 ‖ 256/256 | 4.00174e+06 ‖ 4.93496e+06 |
| utd2 | 2.15654e+06 ‖ 2.34496e+06 | 4.92359e+09 ‖ 5.25494e+09 | 3.84513e+09 ‖ 4.13244e+09 | 4.30080e+06 ‖ 4.66944e+06 | 2.00090e+08 ‖ 2.72589e+08 | 1.84520e+06 ‖ 2.59000e+06 | 256/256 ‖ 256/256 | 4.00174e+06 ‖ 4.93496e+06 |
| utd4 | 2.15654e+06 ‖ 2.34496e+06 | 4.92359e+09 ‖ 5.25494e+09 | 3.84513e+09 ‖ 4.13244e+09 | 4.30080e+06 ‖ 4.66944e+06 | 2.00090e+08 ‖ 2.72589e+08 | 1.84520e+06 ‖ 2.59000e+06 | 256/256 ‖ 256/256 | 4.00174e+06 ‖ 4.93496e+06 |
| w256 | 145920 ‖ 193024 | 3.24927e+08 ‖ 4.07765e+08 | 2.56639e+08 ‖ 3.28466e+08 | 288772 ‖ 380932 | 2.00090e+08 ‖ 2.72589e+08 | 1.84520e+06 ‖ 2.59000e+06 | 256/256 ‖ 256/256 | 1.99112e+06 ‖ 2.78302e+06 |
| w512 | 553984 ‖ 648192 | 1.25383e+09 ‖ 1.41951e+09 | 9.83040e+08 ‖ 1.12669e+09 | 1.10183e+06 ‖ 1.28615e+06 | 2.00090e+08 ‖ 2.72589e+08 | 1.84520e+06 ‖ 2.59000e+06 | 256/256 ‖ 256/256 | 2.39918e+06 ‖ 3.23819e+06 |
| b512 | 2.15654e+06 ‖ 2.34496e+06 | 9.84718e+09 ‖ 1.05099e+10 | 7.69026e+09 ‖ 8.26488e+09 | 4.30080e+06 ‖ 4.66944e+06 | 2.00090e+08 ‖ 2.72589e+08 | 1.84520e+06 ‖ 2.59000e+06 | 512/256 ‖ 512/256 | 4.00174e+06 ‖ 4.93496e+06 |
| b1024 | 2.15654e+06 ‖ 2.34496e+06 | 1.96944e+10 ‖ 2.10198e+10 | 1.53805e+10 ‖ 1.65298e+10 | 4.30080e+06 ‖ 4.66944e+06 | 2.00090e+08 ‖ 2.72589e+08 | 1.84520e+06 ‖ 2.59000e+06 | 1024/256 ‖ 1024/256 | 4.00174e+06 ‖ 4.93496e+06 |
| rollout1 | n/a ‖ 2.34496e+06 | n/a ‖ 5.25494e+09 | n/a ‖ 4.13244e+09 | n/a ‖ 4.66944e+06 | n/a ‖ 2.72589e+08 | n/a ‖ 2.59000e+06 | n/a ‖ 256/256 | n/a ‖ 4.93496e+06 |
| rollout15 | n/a ‖ 2.34496e+06 | n/a ‖ 5.25494e+09 | n/a ‖ 4.13244e+09 | n/a ‖ 4.66944e+06 | n/a ‖ 2.72589e+08 | n/a ‖ 2.59000e+06 | n/a ‖ 256/256 | n/a ‖ 4.93496e+06 |


**2.6 Analytic cross-check of the dynamics constants** (derived by this generator; forward = 2·B·n_in·n_out per layer over the trunk and the two heads; member fwd+bwd = forward + weight-grad (all layers) + input-grad (all layers except the first, whose input is data); the stored `_analytic_cross_check` uses 3 × forward and is an approximation):


| env | dynamics_member_fwdbwd measured | derived | stored analytic approx (3×fwd) | ensemble_forward_all_bs1 measured | derived (7 × member fwd at B=1) | derived vs measured |
|---|---|---|---|---|---|---|
| HalfCheetah-v5 | 200089600 | 200089600 | 202444800 | 1845200 | 1845200 | exact |
| Ant-v5 | 272588800 | 272588800 | 284160000 | 2590000 | 2590000 | exact |

- **PASS** — derived dense-layer counts equal FlopCounterMode exactly for the dynamics member and the ensemble forward
- **PASS** — the SAC-part constants (rollout, critic, actor) of every MBPO signature equal the dense-layer derivation of Section 4.1 exactly

**2.7 Call counts and where they are defined** (`compute_energy_per_flop.rows_for_mbpo`, which calls `rows_for_sac_or_td3("sac", …)` for the SAC part, and `mbpo_fit_flops`):

```
n_env    = 100 * 1000 = 100000                      # rollout calls (actor forward at batch 1)
n_upd    = n_env * updates_per_env_step             # buffer_sample = critic_update = actor_update = target_update calls (policy_update_delay = 1)
dynamics_model_update: call_count = Σ over fits of epochs_run × ceil(n_train / 256)   # optimiser steps; epochs_run = logged model_train_epochs
synthetic_rollout_generation: call_count = Σ_epochs synthetic_transitions_generated    # samples; FLOPs = samples × (actor_forward_bs1 + dynamics_ensemble_forward_all_bs1)
```
`mbpo_fit_flops` (verbatim from `flop_analysis/compute_energy_per_flop.py`):

```python
 309  def mbpo_fit_flops(n_total, train_epochs, algo_config, flops_key):
 310      """Matmul FLOPs of one EnsembleDynamicsModel.fit() call on a real buffer
 311      holding n_total transitions that ran train_epochs passes. Returns
 312      (train_flops, holdout_flops, optimizer_steps).
 313  
 314      Mirrors fit() in algorithms/dynamics_model.py: each pass trains every
 315      member on its n_train-sample bootstrap (the last minibatch is partial,
 316      charged at its true size) and then scores all members on the n_holdout
 317      split via _holdout_mse() -> forward_all(). Matmul FLOPs are exactly
 318      linear in batch size, so per-sample constants are taken from the
 319      measured batch-B constants (verified by verify_mbpo_fit_flops.py)."""
 320      train_batch = algo_config["model_train_batch_size"]
 321      member_fwdbwd = flops_key["dynamics_member_fwdbwd"]
 322      if member_fwdbwd % train_batch:
 323          raise ValueError(f"dynamics_member_fwdbwd={member_fwdbwd} not divisible by "
 324                           f"model_train_batch_size={train_batch}; linear per-sample scaling doesn't hold")
 325      per_sample_member_fwdbwd = member_fwdbwd // train_batch
 326      per_sample_ensemble_fwd = flops_key["dynamics_ensemble_forward_all_bs1"]  # already all members, bs=1
 327  
 328      n_holdout = max(1, int(n_total * algo_config["model_holdout_ratio"]))  # exactly as fit()
 329      n_train = n_total - n_holdout
 330      train_flops = train_epochs * algo_config["ensemble_size"] * n_train * per_sample_member_fwdbwd
 331      holdout_flops = train_epochs * n_holdout * per_sample_ensemble_fwd
 332      optimizer_steps = train_epochs * ceil_div(n_train, train_batch)
 333      return train_flops, holdout_flops, optimizer_steps
 334  
 335
```
`rows_for_mbpo` (lines 336–381):

```python
 336  def rows_for_mbpo(flops_key, algo_config, experiment_config, epochs_log, run_dir=None):
 337      """Returns (rows, dynamics_split) -- dynamics_split holds the
 338      dynamics_model_update train/holdout FLOP breakdown for the CSV."""
 339      rows = rows_for_sac_or_td3("sac", flops_key, algo_config, experiment_config)
 340  
 341      warmup = experiment_config["warmup_steps"]
 342      steps_per_epoch = experiment_config["steps_per_epoch"]
 343      capacity = algo_config["buffer_capacity"]
 344  
 345      dyn_train = dyn_holdout = dyn_steps = 0
 346      for row in epochs_log:
 347          epochs_run = row.get("model_train_epochs")
 348          if not epochs_run:  # fit() skipped this epoch (model_train_freq) or never ran
 349              continue
 350          # algorithms/mbpo.py calls fit() at the start of each epoch, before that
 351          # epoch's rollout -- so the real buffer holds warmup + epoch * steps_per_epoch.
 352          n_total = min(warmup + row["epoch"] * steps_per_epoch, capacity)
 353          logged = row.get("buffer_size")  # len(real_buffer) at the END of the epoch
 354          if logged is not None and logged != min(n_total + steps_per_epoch, capacity):
 355              print(f"WARNING: {run_dir} epoch {row['epoch']}: logged buffer_size {logged} inconsistent "
 356                    f"with fit() size {n_total} + {steps_per_epoch} steps")
 357          train_f, holdout_f, steps = mbpo_fit_flops(n_total, epochs_run, algo_config, flops_key)
 358          dyn_train += train_f
 359          dyn_holdout += holdout_f
 360          dyn_steps += steps
 361      dyn_flops = dyn_train + dyn_holdout
 362      holdout_pct = (100.0 * dyn_holdout / dyn_flops) if dyn_flops else 0.0
 363      rows["dynamics_model_update"] = (
 364          dyn_steps, dyn_flops,
 365          f"per fit() epoch: ensemble_size x n_train x per-sample member fwd+bwd (exact partial last "
 366          f"batch) + n_holdout x per-sample full-ensemble forward (_holdout_mse). call_count = optimizer "
 367          f"steps (sum of epochs_run x ceil(n_train/batch)); epochs_run from training_metrics.json "
 368          f"(early stopping is data-dependent). train={dyn_train} holdout={dyn_holdout} "
 369          f"({holdout_pct:.2f}% holdout)",
 370      )
 371      dynamics_split = {"dynamics_train_flops": dyn_train, "dynamics_holdout_flops": dyn_holdout}
 372  
 373      per_sample = flops_key["actor_forward_bs1"] + flops_key["dynamics_ensemble_forward_all_bs1"]
 374      total_samples = sum(row.get("synthetic_transitions_generated", 0) for row in epochs_log)
 375      rows["synthetic_rollout_generation"] = (
 376          total_samples, total_samples * per_sample,
 377          "actor forward + full-ensemble forward, per synthetic sample (all members scored every predict() call)",
 378      )
 379      return rows, dynamics_split
 380  
 381
```
The generator recomputes both from the logged `training_metrics.json` epoch rows (`s4_data.mbpo_extras`): per epoch with a non-null `model_train_epochs`, n_total = min(warmup + epoch × 1000, capacity), n_holdout = max(1, int(0.2 · n_total)), n_train = n_total − n_holdout, train FLOPs = epochs_run × 7 × n_train × (dynamics_member_fwdbwd/256), holdout FLOPs = epochs_run × n_holdout × dynamics_ensemble_forward_all_bs1, steps = epochs_run × ceil(n_train/256). Call counts per configuration (mean over seeds; `(min, max)` where seeds differ):


| config | calls rollout | calls buf_sample | calls critic | calls actor | calls target | calls dyn_model | calls synth_rollout |
|---|---|---|---|---|---|---|---|
| base | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 166294 (min 160168, max 171842) ‖ 208429 (min 184615, max 234243) | 990000 ‖ 9.34276e+06 (min 9.15543e+06, max 9.57817e+06) |
| utd2 | 100000 ‖ 100000 | 200000 ‖ 200000 | 200000 ‖ 200000 | 200000 ‖ 200000 | 200000 ‖ 200000 | 173106 (min 158402, max 183326) ‖ 220701 (min 190310, max 236392) | 990000 ‖ 9.23514e+06 (min 9.06241e+06, max 9.51267e+06) |
| utd4 | 100000 ‖ 100000 | 400000 ‖ 400000 | 400000 ‖ 400000 | 400000 ‖ 400000 | 400000 ‖ 400000 | 162606 (min 147270, max 172492) ‖ 212141 (min 192218, max 239342) | 990000 ‖ 8.96171e+06 (min 8.47863e+06, max 9.30581e+06) |
| w256 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 162846 (min 154251, max 167919) ‖ 214821 (min 203214, max 232141) | 990000 ‖ 9.25392e+06 (min 9.12238e+06, max 9.37481e+06) |
| w512 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 165829 (min 156886, max 172322) ‖ 208889 (min 174143, max 230384) | 990000 ‖ 9.39758e+06 (min 9.21407e+06, max 9.66042e+06) |
| b512 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 159823 (min 152049, max 165779) ‖ 200737 (min 175986, max 234200) | 990000 ‖ 9.33493e+06 (min 9.07449e+06, max 9.67142e+06) |
| b1024 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 159605 (min 151268, max 165893) ‖ 212390 (min 192594, max 232499) | 990000 ‖ 9.21347e+06 (min 9.09501e+06, max 9.41366e+06) |
| rollout1 | n/a ‖ 100000 | n/a ‖ 100000 | n/a ‖ 100000 | n/a ‖ 100000 | n/a ‖ 100000 | n/a ‖ 207135 (min 198734, max 215805) | n/a ‖ 990000 |
| rollout15 | n/a ‖ 100000 | n/a ‖ 100000 | n/a ‖ 100000 | n/a ‖ 100000 | n/a ‖ 100000 | n/a ‖ 211290 (min 197560, max 225381) | n/a ‖ 6.11639e+06 (min 6.04659e+06, max 6.18251e+06) |


**2.8 Total FLOPs per run and segment** (`dyn_model` and `synth_rollout` vary between seeds: early-stopped fits, Ant terminations):


| config | F dyn_model | F rollout | F synth_rollout | F critic | F actor | F target [elementwise ops] | F GU | F TOTAL |
|---|---|---|---|---|---|---|---|---|
| base | 2.51804e+14 (min 2.42559e+14, max 2.60247e+14) ‖ 4.30949e+14 (min 3.81730e+14, max 4.84342e+14) | 2.15654e+11 ‖ 2.34496e+11 | 3.96173e+12 ‖ 4.61061e+13 (min 4.51817e+13, max 4.72679e+13) | 4.92359e+14 ‖ 5.25494e+14 | 3.84513e+14 ‖ 4.13244e+14 | 4.30080e+11 ‖ 4.66944e+11 | 8.76872e+14 ‖ 9.38738e+14 | 1.13285e+15 (min 1.12361e+15, max 1.14130e+15) ‖ 1.41603e+15 (min 1.36797e+15, max 1.46930e+15) |
| utd2 | 2.62168e+14 (min 2.39825e+14, max 2.77646e+14) ‖ 4.56382e+14 (min 3.93487e+14, max 4.88853e+14) | 2.15654e+11 ‖ 2.34496e+11 | 3.96173e+12 ‖ 4.55750e+13 (min 4.47226e+13, max 4.69446e+13) | 9.84718e+14 ‖ 1.05099e+15 | 7.69026e+14 ‖ 8.26488e+14 | 8.60161e+11 ‖ 9.33889e+11 | 1.75374e+15 ‖ 1.87748e+15 | 2.02009e+15 (min 1.99775e+15, max 2.03557e+15) ‖ 2.37967e+15 (min 2.31814e+15, max 2.41235e+15) |
| utd4 | 2.46235e+14 (min 2.23010e+14, max 2.61206e+14) ‖ 4.38714e+14 (min 3.97453e+14, max 4.95042e+14) | 2.15654e+11 ‖ 2.34496e+11 | 3.96173e+12 ‖ 4.42257e+13 (min 4.18417e+13, max 4.59238e+13) | 1.96944e+15 ‖ 2.10198e+15 | 1.53805e+15 ‖ 1.65298e+15 | 1.72032e+12 ‖ 1.86778e+12 | 3.50749e+15 ‖ 3.75495e+15 | 3.75790e+15 (min 3.73467e+15, max 3.77287e+15) ‖ 4.23812e+15 (min 4.19746e+15, max 4.29207e+15) |
| w256 | 2.46604e+14 (min 2.33579e+14, max 2.54253e+14) ‖ 4.44206e+14 (min 4.20149e+14, max 4.80078e+14) | 1.45920e+10 ‖ 1.93024e+10 | 1.97121e+12 ‖ 2.57539e+13 (min 2.53878e+13, max 2.60903e+13) | 3.24927e+13 ‖ 4.07765e+13 | 2.56639e+13 ‖ 3.28466e+13 | 2.88772e+10 ‖ 3.80932e+10 | 5.81566e+13 ‖ 7.36231e+13 | 3.06747e+14 (min 2.93722e+14, max 3.14395e+14) ‖ 5.43602e+14 (min 5.19438e+14, max 5.79108e+14) |
| w512 | 2.51142e+14 (min 2.37585e+14, max 2.60935e+14) ‖ 4.31938e+14 (min 3.60045e+14, max 4.76487e+14) | 5.53984e+10 ‖ 6.48192e+10 | 2.37519e+12 ‖ 3.04312e+13 (min 2.98369e+13, max 3.12823e+13) | 1.25383e+14 ‖ 1.41951e+14 | 9.83040e+13 ‖ 1.12669e+14 | 1.10183e+11 ‖ 1.28615e+11 | 2.23687e+14 ‖ 2.54620e+14 | 4.77260e+14 (min 4.63703e+14, max 4.87053e+14) ‖ 7.17054e+14 (min 6.46013e+14, max 7.61512e+14) |
| b512 | 2.42018e+14 (min 2.30233e+14, max 2.51068e+14) ‖ 4.15041e+14 (min 3.63804e+14, max 4.84264e+14) | 2.15654e+11 ‖ 2.34496e+11 | 3.96173e+12 ‖ 4.60675e+13 (min 4.47822e+13, max 4.77281e+13) | 9.84718e+14 ‖ 1.05099e+15 | 7.69026e+14 ‖ 8.26488e+14 | 4.30080e+11 ‖ 4.66944e+11 | 1.75374e+15 ‖ 1.87748e+15 | 1.99994e+15 (min 1.98815e+15, max 2.00899e+15) ‖ 2.33882e+15 (min 2.28924e+15, max 2.40680e+15) |
| b1024 | 2.41654e+14 (min 2.28985e+14, max 2.51130e+14) ‖ 4.39178e+14 (min 3.98211e+14, max 4.80791e+14) | 2.15654e+11 ‖ 2.34496e+11 | 3.96173e+12 ‖ 4.54681e+13 (min 4.48835e+13, max 4.64560e+13) | 1.96944e+15 ‖ 2.10198e+15 | 1.53805e+15 ‖ 1.65298e+15 | 4.30080e+11 ‖ 4.66944e+11 | 3.50749e+15 ‖ 3.75495e+15 | 3.75332e+15 (min 3.74065e+15, max 3.76279e+15) ‖ 4.23983e+15 (min 4.19828e+15, max 4.28141e+15) |
| rollout1 | n/a ‖ 4.28258e+14 (min 4.10804e+14, max 4.46163e+14) | n/a ‖ 2.34496e+11 | n/a ‖ 4.88561e+12 | n/a ‖ 5.25494e+14 | n/a ‖ 4.13244e+14 | n/a ‖ 4.66944e+11 | n/a ‖ 9.38738e+14 | n/a ‖ 1.37212e+15 (min 1.35466e+15, max 1.39002e+15) |
| rollout15 | n/a ‖ 4.36857e+14 (min 4.08470e+14, max 4.66053e+14) | n/a ‖ 2.34496e+11 | n/a ‖ 3.01841e+13 (min 2.98397e+13, max 3.05104e+13) | n/a ‖ 5.25494e+14 | n/a ‖ 4.13244e+14 | n/a ‖ 4.66944e+11 | n/a ‖ 9.38738e+14 | n/a ‖ 1.40601e+15 (min 1.37751e+15, max 1.43526e+15) |

Ant / HalfCheetah ratios of the (seed-mean) quantities:


| config | Ant/HC dyn_model | Ant/HC rollout | Ant/HC synth_rollout | Ant/HC critic | Ant/HC actor | Ant/HC target | Ant/HC GU | Ant/HC TOTAL |
|---|---|---|---|---|---|---|---|---|
| base | 1.71145 | 1.08737 | 11.6379 | 1.06730 | 1.07472 | 1.08571 | 1.07055 | 1.24997 |
| utd2 | 1.74080 | 1.08737 | 11.5038 | 1.06730 | 1.07472 | 1.08571 | 1.07055 | 1.17800 |
| utd4 | 1.78169 | 1.08737 | 11.1632 | 1.06730 | 1.07472 | 1.08571 | 1.07055 | 1.12779 |
| w256 | 1.80129 | 1.32281 | 13.0650 | 1.25494 | 1.27988 | 1.31914 | 1.26595 | 1.77215 |
| w512 | 1.71989 | 1.17006 | 12.8121 | 1.13213 | 1.14613 | 1.16729 | 1.13829 | 1.50244 |
| b512 | 1.71492 | 1.08737 | 11.6281 | 1.06730 | 1.07472 | 1.08571 | 1.07055 | 1.16945 |
| b1024 | 1.81738 | 1.08737 | 11.4768 | 1.06730 | 1.07472 | 1.08571 | 1.07055 | 1.12962 |


## 3. Baseline decomposition and energy per FLOP (4.3.1)

Configuration `base` (see tags above; HC and Ant baselines differ in the rollout-length schedule, Section 6.5).

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| base | 146193 ± 7439.23 ‖ 188363 ± 20295.5 | 2428.26 ± 47.7526 ‖ 4292.41 ± 29.8641 | 75.3990 ± 16.3698 ‖ 735.651 ± 37.3179 | 1677.11 ± 15.2915 ‖ 2271.09 ± 12.1950 | 24588.5 ± 267.376 ‖ 24560.0 ± 218.652 | 29882.6 ± 104.818 ‖ 30340.0 ± 62.3096 | 2550.10 ± 26.8657 ‖ 2575.51 ± 25.8022 | 58698.3 ± 211.888 ‖ 59746.5 ± 208.801 | 207395 ± 7451.41 ‖ 253138 ± 20140.0 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| base | 0.0576096 ± 0.00206984 ‖ 0.0703161 ± 0.00559445 | 18.8339 ± 2.39855 ‖ 74.0726 ± 3.03600 | 5820.88 ± 39.1697 ‖ 5763.91 ± 33.4087 | 6042.89 ± 33.9578 ‖ 6037.16 ± 35.0893 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|---|---|
| base | 70.4601 ± 1.04907 ‖ 74.2785 ± 2.09320 | 1.17147 ± 0.0245455 ‖ 1.70357 ± 0.124251 | 0.0365412 ± 0.00862227 ‖ 0.292741 ± 0.0357971 | 0.809395 ± 0.0267401 ‖ 0.901530 ± 0.0684953 | 11.8698 ± 0.495888 ‖ 9.75479 ± 0.832491 | 14.4221 ± 0.472909 ‖ 12.0462 ± 0.954617 | 1.23060 ± 0.0365860 ‖ 1.02264 ± 0.0822693 | 28.3319 ± 1.02110 ‖ 23.7252 ± 1.93477 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| base | 2.85721 ± 0.0304969 ‖ 3.80128 ± 0.0300127 | 41.8889 ± 0.334398 ‖ 41.1064 ± 0.243247 | 50.9094 ± 0.258129 ‖ 50.7815 ± 0.178929 | 4.34452 ± 0.0541846 ‖ 4.31077 ± 0.0449133 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | dyn_model (measured) [J/FLOP] | rollout (measured) [J/FLOP] | synth_rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|---|---|
| base | 5.80582e-10 ± 1.48182e-11 ‖ 4.37090e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59556e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83073e-10 ± 5.44247e-12 ‖ 1.78766e-10 ± 8.47149e-12 | 5.92935 ± 0.0624667 ‖ 5.51568 ± 0.0552575 |

MBPO FLOPs differ between seeds. **Mean of the per-seed ratios** (E_i/F_i averaged over seeds) ± sd, for comparison with the ratio-of-means table above:

| config | dyn_model [J/FLOP] | rollout [J/FLOP] | synth_rollout [J/FLOP] | critic [J/FLOP] | actor [J/FLOP] | GU [J/FLOP] | TOTAL [J/FLOP] |
|---|---|---|---|---|---|---|---|
| base | 5.80351e-10 ± 1.48182e-11 ‖ 4.37061e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59503e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83047e-10 ± 5.44247e-12 ‖ 1.78551e-10 ± 8.47149e-12 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|---|
| base | 0.752848 | 1.62566 | 0.838363 | 0.935860 | 0.944715 | 0.950778 | 0.976476 | 0.930232 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | dyn_model (measured) duration [s] | rollout (measured) duration [s] | synth_rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|---|---|
| base | 1250.07 ± 72.8336 ‖ 1605.17 ± 180.136 | 20.3785 ± 0.415238 ‖ 36.6806 ± 0.0873145 | 0.633054 ± 0.00774843 ‖ 5.07582 ± 0.0749797 | 407.584 ± 2.70475 ‖ 411.097 ± 2.42950 | 11.2917 ± 0.0973144 ‖ 15.1580 ± 0.0311855 | 165.557 ± 2.32430 ‖ 163.927 ± 1.97951 | 201.197 ± 0.836174 ‖ 202.501 ± 0.702973 | 17.1693 ± 0.155059 ‖ 17.1899 ± 0.159045 | 1678.67 ± 72.2102 ‖ 2058.02 ± 178.140 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P dyn_model [W] | P rollout [W] | P synth_rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|---|---|
| base | 116.988 ± 0.953539 ‖ 117.397 ± 0.655463 | 119.160 ± 0.550654 ‖ 117.022 ± 0.953688 | 119.137 ± 25.8677 ‖ 144.882 ± 5.45432 | 144.018 ± 0.467991 ‖ 145.336 ± 0.473385 | 123.577 ± 0.913456 ‖ 123.061 ± 0.930322 | 117.983 ± 0.615299 ‖ 118.404 ± 0.501427 | 0.827400 ± 0.00273275 ‖ 0.805180 ± 0.00538864 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | dyn_model [GFLOP/s] | rollout [GFLOP/s] | synth_rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|---|---|
| base | 201.719 ± 6.74360 ‖ 268.627 ± 3.33295 | 10.5859 ± 0.210149 ‖ 6.39295 ± 0.0152330 | 6258.88 ± 77.5462 ‖ 9083.34 ± 23.6982 | 2151.47 ± 14.2695 ‖ 2283.56 ± 13.4483 | 675.699 ± 24.6566 ‖ 690.649 ± 37.7185 | 2974.43 ± 41.5340 ‖ 3206.03 ± 38.3417 | 1911.16 ± 7.94527 ‖ 2040.72 ± 7.09689 | 96.9650 ± 0.0406469 ‖ 97.0025 ± 0.0629928 | 96.8947–96.9927 ‖ 96.9168–97.0815 |

**Hardware composition of the measured `dynamics_model_update` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of dyn_model energy [%] | GPU share of dyn_model energy [%] | RAM share of dyn_model energy [%] | dyn_model energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 21.4036 ± 0.316290 ‖ 21.4284 ± 0.217233 | 61.5003 ± 0.203244 ‖ 61.5356 ± 0.155617 | 17.0961 ± 0.139314 ‖ 17.0360 ± 0.0949920 | 0/500 ‖ 0/500 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 25.7910 ± 0.402018 ‖ 27.6475 ± 0.322486 | 57.4494 ± 0.449754 ‖ 55.2805 ± 0.439211 | 16.7596 ± 0.0798437 ‖ 17.0720 ± 0.139120 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `synthetic_rollout_generation` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of synth_rollout energy [%] | GPU share of synth_rollout energy [%] | RAM share of synth_rollout energy [%] | synth_rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 26.1498 ± 6.92079 ‖ 24.0129 ± 0.939763 | 57.2526 ± 11.2969 ‖ 62.3211 ± 1.43681 | 16.5975 ± 4.37648 ‖ 13.6660 ± 0.498430 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 18.0568 ± 0.118760 ‖ 18.2631 ± 0.0602319 | 68.0574 ± 0.145407 ‖ 67.9766 ± 0.0664136 | 13.8857 ± 0.0451822 ‖ 13.7603 ± 0.0448462 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | dyn_model [%] | rollout [%] | synth_rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|---|---|
| base | 56.3406 ± 0.609099 ‖ 55.8464 ± 0.519715 | 55.3099 ± 0.0464751 ‖ 56.0243 ± 0.227975 | 57.9021 ± 15.0568 ‖ 45.3015 ± 1.71294 | 45.7635 ± 0.167734 ‖ 45.1088 ± 0.176904 | 53.3357 ± 0.515845 ‖ 53.2775 ± 0.609384 | 65.9074 ± 0.314375 ‖ 65.5593 ± 0.279103 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| base | 1.22056 | 1.76769 | 3.81837 | dyn_model | 0.995821 | 0.973150 | 0.976476 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | dyn_model | synth_rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| base | 1.76769 (paired sd 0.0422332) | 1.28846 (paired sd 0.166389) | 9.75678 (paired sd 2.54254) | 1.35417 (paired sd 0.0179559) | 0.998842 (paired sd 0.0177614) | 1.01530 (paired sd 0.00522707) | 1.00997 (paired sd 0.0182342) | 1.01786 (paired sd 0.00643497) | 1.22056 (paired sd 0.114543) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | dyn_model | rollout | synth_rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|---|---|
| base | 3.81837 ± 2.48009 (5/5 same sign) | 0.532104 ± 0.122998 (5/5 same sign) | 0.256200 ± 0.0396612 (5/5 same sign) | 0.0921353 ± 0.0731507 (4/5 same sign) | -2.11498 ± 1.04631 (5/5 same sign) | -2.37587 ± 1.12935 (5/5 same sign) | -0.207957 ± 0.0903167 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| base | mean Δ 0.532104; 2/32 patterns; p = 0.0625000 | mean Δ 45743.3; 2/32 patterns; p = 0.0625000 |


## 4.1 Sweep: UTD — updates_per_env_step ∈ {1,2,4}

Configurations: `base` (SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model (200×4)×7; HC: fixed rollout length 1; Ant: rollout length 1→25 (mbpo_ant.json)), `utd2` (UTD 2), `utd4` (UTD 4). Baseline for the response tables: `base` of the same environment.

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| base | 146193 ± 7439.23 ‖ 188363 ± 20295.5 | 2428.26 ± 47.7526 ‖ 4292.41 ± 29.8641 | 75.3990 ± 16.3698 ‖ 735.651 ± 37.3179 | 1677.11 ± 15.2915 ‖ 2271.09 ± 12.1950 | 24588.5 ± 267.376 ‖ 24560.0 ± 218.652 | 29882.6 ± 104.818 ‖ 30340.0 ± 62.3096 | 2550.10 ± 26.8657 ‖ 2575.51 ± 25.8022 | 58698.3 ± 211.888 ‖ 59746.5 ± 208.801 | 207395 ± 7451.41 ‖ 253138 ± 20140.0 |
| utd2 | 150041 ± 6731.25 ‖ 193129 ± 18507.5 | 2409.68 ± 49.2703 ‖ 4214.55 ± 35.6918 | 72.5129 ± 11.3923 ‖ 750.253 ± 42.0048 | 3322.85 ± 32.5120 ‖ 4463.52 ± 13.6184 | 48842.7 ± 678.227 ‖ 49339.3 ± 1185.47 | 59095.0 ± 425.948 ‖ 59844.1 ± 404.340 | 5067.36 ± 50.8363 ‖ 5056.08 ± 20.2017 | 116328 ± 1048.96 ‖ 118703 ± 1584.99 | 268851 ± 6928.60 ‖ 316797 ± 17138.3 |
| utd4 | 139320 ± 8673.80 ‖ 183704 ± 16539.8 | 2381.09 ± 23.2019 ‖ 4178.96 ± 59.5280 | 74.7968 ± 12.7259 ‖ 712.228 ± 42.1119 | 6596.08 ± 28.2525 ‖ 8867.02 ± 31.6077 | 97613.2 ± 613.239 ‖ 98000.7 ± 552.597 | 117546 ± 313.899 ‖ 118862 ± 220.469 | 10075.3 ± 50.6789 ‖ 10129.2 ± 69.0324 | 231830 ± 629.988 ‖ 235859 ± 735.086 | 373606 ± 8544.94 ‖ 424454 ± 16395.0 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| base | 0.0576096 ± 0.00206984 ‖ 0.0703161 ± 0.00559445 | 18.8339 ± 2.39855 ‖ 74.0726 ± 3.03600 | 5820.88 ± 39.1697 ‖ 5763.91 ± 33.4087 | 6042.89 ± 33.9578 ‖ 6037.16 ± 35.0893 |
| utd2 | 0.0746807 ± 0.00192461 ‖ 0.0879990 ± 0.00476065 | 18.8355 ± 2.64902 ‖ 74.9788 ± 3.37524 | 5821.58 ± 213.100 ‖ 5705.21 ± 42.8543 | 5950.65 ± 66.5007 ‖ 5938.51 ± 21.4799 |
| utd4 | 0.103780 ± 0.00237359 ‖ 0.117904 ± 0.00455417 | 17.9710 ± 2.86641 ‖ 75.4283 ± 2.65531 | 5753.95 ± 130.977 ‖ 5726.17 ± 43.5735 | 6040.18 ± 49.6872 ‖ 5973.75 ± 48.2472 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|---|---|
| base | 70.4601 ± 1.04907 ‖ 74.2785 ± 2.09320 | 1.17147 ± 0.0245455 ‖ 1.70357 ± 0.124251 | 0.0365412 ± 0.00862227 ‖ 0.292741 ± 0.0357971 | 0.809395 ± 0.0267401 ‖ 0.901530 ± 0.0684953 | 11.8698 ± 0.495888 ‖ 9.75479 ± 0.832491 | 14.4221 ± 0.472909 ‖ 12.0462 ± 0.954617 | 1.23060 ± 0.0365860 ‖ 1.02264 ± 0.0822693 | 28.3319 ± 1.02110 ‖ 23.7252 ± 1.93477 |
| utd2 | 55.7864 ± 1.12301 ‖ 60.8466 ± 2.71548 | 0.896666 ± 0.0253243 ‖ 1.33410 ± 0.0866397 | 0.0269341 ± 0.00384946 ‖ 0.237930 ± 0.0264796 | 1.23666 ± 0.0363841 ‖ 1.41231 ± 0.0775139 | 18.1746 ± 0.421373 ‖ 15.6274 ± 1.24556 | 21.9928 ± 0.612521 ‖ 18.9417 ± 1.19123 | 1.88590 ± 0.0554493 ‖ 1.60001 ± 0.0930658 | 43.2900 ± 1.10294 ‖ 37.5814 ± 2.60328 |
| utd4 | 37.2635 ± 1.48973 ‖ 43.2119 ± 2.21577 | 0.637499 ± 0.00996864 ‖ 0.985538 ± 0.0341129 | 0.0200738 ± 0.00377958 ‖ 0.168004 ± 0.0120665 | 1.76632 ± 0.0445355 ‖ 2.09144 ± 0.0776815 | 26.1373 ± 0.568447 ‖ 23.1175 ± 0.944592 | 31.4773 ± 0.807492 ‖ 28.0366 ± 1.07433 | 2.69804 ± 0.0709861 ‖ 2.38903 ± 0.0855933 | 62.0790 ± 1.47765 ‖ 55.6346 ± 2.17810 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| base | 2.85721 ± 0.0304969 ‖ 3.80128 ± 0.0300127 | 41.8889 ± 0.334398 ‖ 41.1064 ± 0.243247 | 50.9094 ± 0.258129 ‖ 50.7815 ± 0.178929 | 4.34452 ± 0.0541846 ‖ 4.31077 ± 0.0449133 |
| utd2 | 2.85653 ± 0.0260464 ‖ 3.76088 ± 0.0597819 | 41.9860 ± 0.274909 ‖ 41.5607 ± 0.442828 | 50.8013 ± 0.214416 ‖ 50.4186 ± 0.337549 | 4.35621 ± 0.0402669 ‖ 4.25992 ± 0.0467816 |
| utd4 | 2.84525 ± 0.0179414 ‖ 3.75947 ± 0.0125601 | 42.1052 ± 0.181393 ‖ 41.5504 ± 0.122593 | 50.7036 ± 0.158352 ‖ 50.3956 ± 0.107614 | 4.34595 ± 0.0159772 ‖ 4.29461 ± 0.0244214 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | dyn_model (measured) [J/FLOP] | rollout (measured) [J/FLOP] | synth_rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|---|---|
| base | 5.80582e-10 ± 1.48182e-11 ‖ 4.37090e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59556e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83073e-10 ± 5.44247e-12 ‖ 1.78766e-10 ± 8.47149e-12 | 5.92935 ± 0.0624667 ‖ 5.51568 ± 0.0552575 |
| utd2 | 5.72307e-10 ± 1.16405e-11 ‖ 4.23173e-10 ± 7.68529e-12 | 1.11738e-08 ± 2.28469e-10 ‖ 1.79728e-08 ± 1.52206e-10 | 1.83033e-11 ± 2.87559e-12 ‖ 1.64619e-11 ± 7.37165e-13 | 4.96007e-11 ± 6.88753e-13 ‖ 4.69456e-11 ± 1.12796e-12 | 7.68439e-11 ± 5.53879e-13 ‖ 7.24077e-11 ± 4.89227e-13 | 6.63312e-11 ± 5.98127e-13 ‖ 6.32247e-11 ± 8.44213e-13 | 1.33089e-10 ± 2.49761e-12 ‖ 1.33126e-10 ± 5.19325e-12 | 5.89117 ± 0.0591010 ‖ 5.41400 ± 0.0216318 |
| utd4 | 5.65801e-10 ± 6.41063e-12 ‖ 4.18732e-10 ± 4.45755e-12 | 1.10412e-08 ± 1.07588e-10 ‖ 1.78210e-08 ± 2.53855e-10 | 1.88799e-11 ± 3.21221e-12 ‖ 1.61044e-11 ± 6.20996e-13 | 4.95641e-11 ± 3.11378e-13 ‖ 4.66232e-11 ± 2.62894e-13 | 7.64252e-11 ± 2.04089e-13 ‖ 7.19078e-11 ± 1.33377e-13 | 6.60959e-11 ± 1.79612e-13 ‖ 6.28128e-11 ± 1.95764e-13 | 9.94189e-11 ± 1.82537e-12 ‖ 1.00151e-10 ± 3.00310e-12 | 5.85662 ± 0.0294590 ‖ 5.42315 ± 0.0369596 |

MBPO FLOPs differ between seeds. **Mean of the per-seed ratios** (E_i/F_i averaged over seeds) ± sd, for comparison with the ratio-of-means table above:

| config | dyn_model [J/FLOP] | rollout [J/FLOP] | synth_rollout [J/FLOP] | critic [J/FLOP] | actor [J/FLOP] | GU [J/FLOP] | TOTAL [J/FLOP] |
|---|---|---|---|---|---|---|---|
| base | 5.80351e-10 ± 1.48182e-11 ‖ 4.37061e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59503e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83047e-10 ± 5.44247e-12 ‖ 1.78551e-10 ± 8.47149e-12 |
| utd2 | 5.72783e-10 ± 1.16405e-11 ‖ 4.22810e-10 ± 7.68529e-12 | 1.11738e-08 ± 2.28469e-10 ‖ 1.79728e-08 ± 1.52206e-10 | 1.83033e-11 ± 2.87559e-12 ‖ 1.64575e-11 ± 7.37165e-13 | 4.96007e-11 ± 6.88753e-13 ‖ 4.69456e-11 ± 1.12796e-12 | 7.68439e-11 ± 5.53879e-13 ‖ 7.24077e-11 ± 4.89227e-13 | 6.63312e-11 ± 5.98127e-13 ‖ 6.32247e-11 ± 8.44213e-13 | 1.33075e-10 ± 2.49761e-12 ‖ 1.33061e-10 ± 5.19325e-12 |
| utd4 | 5.66061e-10 ± 6.41063e-12 ‖ 4.18582e-10 ± 4.45755e-12 | 1.10412e-08 ± 1.07588e-10 ‖ 1.78210e-08 ± 2.53855e-10 | 1.88799e-11 ± 3.21221e-12 ‖ 1.61019e-11 ± 6.20996e-13 | 4.95641e-11 ± 3.11378e-13 ‖ 4.66232e-11 ± 2.62894e-13 | 7.64252e-11 ± 2.04089e-13 ‖ 7.19078e-11 ± 1.33377e-13 | 6.60959e-11 ± 1.79612e-13 ‖ 6.28128e-11 ± 1.95764e-13 | 9.94123e-11 ± 1.82537e-12 ‖ 1.00131e-10 ± 3.00310e-12 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|---|
| base | 0.752848 | 1.62566 | 0.838363 | 0.935860 | 0.944715 | 0.950778 | 0.976476 | 0.930232 |
| utd2 | 0.739417 | 1.60848 | 0.899395 | 0.946470 | 0.942269 | 0.953168 | 1.00028 | 0.919002 |
| utd4 | 0.740071 | 1.61404 | 0.852994 | 0.940665 | 0.940891 | 0.950328 | 1.00737 | 0.925987 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | dyn_model (measured) duration [s] | rollout (measured) duration [s] | synth_rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|---|---|
| base | 1250.07 ± 72.8336 ‖ 1605.17 ± 180.136 | 20.3785 ± 0.415238 ‖ 36.6806 ± 0.0873145 | 0.633054 ± 0.00774843 ‖ 5.07582 ± 0.0749797 | 407.584 ± 2.70475 ‖ 411.097 ± 2.42950 | 11.2917 ± 0.0973144 ‖ 15.1580 ± 0.0311855 | 165.557 ± 2.32430 ‖ 163.927 ± 1.97951 | 201.197 ± 0.836174 ‖ 202.501 ± 0.702973 | 17.1693 ± 0.155059 ‖ 17.1899 ± 0.159045 | 1678.67 ± 72.2102 ‖ 2058.02 ± 178.140 |
| utd2 | 1299.46 ± 56.8222 ‖ 1660.68 ± 170.317 | 20.6292 ± 0.236635 ‖ 36.9393 ± 0.307848 | 0.642052 ± 0.00444472 ‖ 5.05389 ± 0.0710411 | 819.784 ± 10.7804 ‖ 834.334 ± 15.3206 | 22.6977 ± 0.251504 ‖ 30.4700 ± 0.112733 | 333.651 ± 6.22029 ‖ 336.847 ± 9.81697 | 403.670 ± 4.33270 ‖ 408.537 ± 4.79519 | 34.6142 ± 0.417417 ‖ 34.5156 ± 0.270244 | 2140.52 ± 60.1159 ‖ 2537.00 ± 155.899 |
| utd4 | 1205.35 ± 76.5508 ‖ 1584.74 ± 150.937 | 20.4847 ± 0.152635 ‖ 36.6976 ± 0.256834 | 0.638555 ± 0.00100825 ‖ 4.94241 ± 0.133136 | 1642.15 ± 11.5343 ‖ 1663.35 ± 5.72558 | 45.2561 ± 0.0980753 ‖ 60.7751 ± 0.113436 | 669.758 ± 7.64756 ‖ 671.707 ± 4.14079 | 806.497 ± 3.35684 ‖ 814.690 ± 1.88932 | 69.1280 ± 0.477287 ‖ 69.4261 ± 0.357072 | 2868.62 ± 82.7085 ‖ 3289.73 ± 149.570 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P dyn_model [W] | P rollout [W] | P synth_rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|---|---|
| base | 116.988 ± 0.953539 ‖ 117.397 ± 0.655463 | 119.160 ± 0.550654 ‖ 117.022 ± 0.953688 | 119.137 ± 25.8677 ‖ 144.882 ± 5.45432 | 144.018 ± 0.467991 ‖ 145.336 ± 0.473385 | 123.577 ± 0.913456 ‖ 123.061 ± 0.930322 | 117.983 ± 0.615299 ‖ 118.404 ± 0.501427 | 0.827400 ± 0.00273275 ‖ 0.805180 ± 0.00538864 |
| utd2 | 115.460 ± 0.694350 ‖ 116.370 ± 1.09129 | 116.801 ± 1.19740 ‖ 114.096 ± 0.720371 | 112.982 ± 18.1207 ‖ 148.412 ± 7.00080 | 141.907 ± 0.684910 ‖ 142.284 ± 0.826403 | 125.608 ± 0.545402 ‖ 124.920 ± 1.12468 | 120.934 ± 0.422687 ‖ 120.912 ± 0.830522 | 0.823121 ± 0.0120235 ‖ 0.801920 ± 0.00787801 |
| utd4 | 115.592 ± 0.323504 ‖ 115.969 ± 0.814579 | 116.237 ± 0.455136 ‖ 113.873 ± 1.12498 | 117.122 ± 19.8416 ‖ 144.039 ± 5.89756 | 141.179 ± 0.784890 ‖ 141.798 ± 0.267861 | 130.257 ± 0.829538 ‖ 129.056 ± 0.929958 | 126.432 ± 0.733840 ‖ 125.724 ± 0.751827 | 0.823344 ± 0.00522251 ‖ 0.803055 ± 0.00657002 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | dyn_model [GFLOP/s] | rollout [GFLOP/s] | synth_rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|---|---|
| base | 201.719 ± 6.74360 ‖ 268.627 ± 3.33295 | 10.5859 ± 0.210149 ‖ 6.39295 ± 0.0152330 | 6258.88 ± 77.5462 ‖ 9083.34 ± 23.6982 | 2151.47 ± 14.2695 ‖ 2283.56 ± 13.4483 | 675.699 ± 24.6566 ‖ 690.649 ± 37.7185 | 2974.43 ± 41.5340 ‖ 3206.03 ± 38.3417 | 1911.16 ± 7.94527 ‖ 2040.72 ± 7.09689 | 96.9650 ± 0.0406469 ‖ 97.0025 ± 0.0629928 | 96.8947–96.9927 ‖ 96.9168–97.0815 |
| utd2 | 201.657 ± 4.97659 ‖ 275.338 ± 7.55127 | 10.4549 ± 0.118540 ‖ 6.34850 ± 0.0526406 | 6170.65 ± 42.7291 ‖ 9017.29 ± 55.1281 | 2139.57 ± 27.8369 ‖ 2250.87 ± 40.6511 | 944.193 ± 20.4657 ‖ 940.248 ± 46.1646 | 2952.16 ± 54.6114 ‖ 3122.14 ± 88.9735 | 1905.26 ± 20.2070 ‖ 2023.26 ± 23.4521 | 96.9316 ± 0.0493324 ‖ 97.1278 ± 0.0346554 | 96.8873–97.0070 ‖ 97.0772–97.1675 |
| utd4 | 204.225 ± 2.29297 ‖ 277.093 ± 4.78883 | 10.5281 ± 0.0783073 ‖ 6.39020 ± 0.0446534 | 6204.21 ± 9.79116 ‖ 8945.49 ± 129.459 | 2136.00 ± 15.0280 ‖ 2257.49 ± 7.78114 | 1310.74 ± 32.1678 ‖ 1290.01 ± 47.3420 | 2940.83 ± 33.6611 ‖ 3129.40 ± 19.3119 | 1907.10 ± 7.94911 ‖ 2028.97 ± 4.70398 | 96.8633 ± 0.0386294 ‖ 97.1895 ± 0.0582139 | 96.8012–96.9051 ‖ 97.1124–97.2443 |

**Hardware composition of the measured `dynamics_model_update` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of dyn_model energy [%] | GPU share of dyn_model energy [%] | RAM share of dyn_model energy [%] | dyn_model energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 21.4036 ± 0.316290 ‖ 21.4284 ± 0.217233 | 61.5003 ± 0.203244 ‖ 61.5356 ± 0.155617 | 17.0961 ± 0.139314 ‖ 17.0360 ± 0.0949920 | 0/500 ‖ 0/500 |
| utd2 | 21.5297 ± 0.300848 ‖ 21.8476 ± 0.419960 | 61.1484 ± 0.217964 ‖ 60.9652 ± 0.285494 | 17.3219 ± 0.103623 ‖ 17.1872 ± 0.160820 | 0/500 ‖ 0/500 |
| utd4 | 21.6600 ± 0.253136 ‖ 21.6415 ± 0.330866 | 61.0385 ± 0.250829 ‖ 61.1125 ± 0.219945 | 17.3016 ± 0.0486133 ‖ 17.2461 ± 0.120941 | 0/500 ‖ 0/500 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 25.7910 ± 0.402018 ‖ 27.6475 ± 0.322486 | 57.4494 ± 0.449754 ‖ 55.2805 ± 0.439211 | 16.7596 ± 0.0798437 ‖ 17.0720 ± 0.139120 | 500/500 ‖ 500/500 |
| utd2 | 25.6456 ± 0.506652 ‖ 27.8162 ± 0.232344 | 57.2555 ± 0.652964 ‖ 54.6755 ± 0.325149 | 17.0989 ± 0.176949 ‖ 17.5083 ± 0.110008 | 500/500 ‖ 500/500 |
| utd4 | 25.8077 ± 0.455743 ‖ 27.3432 ± 0.391587 | 57.0124 ± 0.489131 ‖ 55.1123 ± 0.501733 | 17.1798 ± 0.0681774 ‖ 17.5445 ± 0.173936 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `synthetic_rollout_generation` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of synth_rollout energy [%] | GPU share of synth_rollout energy [%] | RAM share of synth_rollout energy [%] | synth_rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 26.1498 ± 6.92079 ‖ 24.0129 ± 0.939763 | 57.2526 ± 11.2969 ‖ 62.3211 ± 1.43681 | 16.5975 ± 4.37648 ‖ 13.6660 ± 0.498430 | 500/500 ‖ 500/500 |
| utd2 | 26.7260 ± 4.00447 ‖ 23.5364 ± 1.13777 | 56.2784 ± 6.67723 ‖ 63.1308 ± 1.76608 | 16.9956 ± 2.67815 ‖ 13.3328 ± 0.628714 | 500/500 ‖ 500/500 |
| utd4 | 26.0667 ± 4.71214 ‖ 24.1385 ± 1.03715 | 57.4038 ± 7.89108 ‖ 62.1177 ± 1.62882 | 16.5296 ± 3.17937 ‖ 13.7438 ± 0.591773 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 18.0568 ± 0.118760 ‖ 18.2631 ± 0.0602319 | 68.0574 ± 0.145407 ‖ 67.9766 ± 0.0664136 | 13.8857 ± 0.0451822 ‖ 13.7603 ± 0.0448462 | 0/500 ‖ 0/500 |
| utd2 | 18.0535 ± 0.129715 ‖ 18.3446 ± 0.0955699 | 67.8533 ± 0.130658 ‖ 67.5993 ± 0.164533 | 14.0932 ± 0.0680356 ‖ 14.0561 ± 0.0819987 | 0/500 ‖ 0/500 |
| utd4 | 18.1222 ± 0.0817779 ‖ 18.1475 ± 0.0655226 | 67.7115 ± 0.0867147 ‖ 67.7482 ± 0.0854187 | 14.1663 ± 0.0786466 ‖ 14.1043 ± 0.0266340 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | dyn_model [%] | rollout [%] | synth_rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|---|---|
| base | 56.3406 ± 0.609099 ‖ 55.8464 ± 0.519715 | 55.3099 ± 0.0464751 ‖ 56.0243 ± 0.227975 | 57.9021 ± 15.0568 ‖ 45.3015 ± 1.71294 | 45.7635 ± 0.167734 ‖ 45.1088 ± 0.176904 | 53.3357 ± 0.515845 ‖ 53.2775 ± 0.609384 | 65.9074 ± 0.314375 ‖ 65.5593 ± 0.279103 |
| utd2 | 56.6452 ± 1.43274 ‖ 55.5901 ± 0.605119 | 55.9896 ± 1.00402 ‖ 56.6956 ± 0.440261 | 59.1561 ± 10.0943 ‖ 43.6689 ± 2.22999 | 46.0885 ± 1.19084 ‖ 45.4627 ± 0.152893 | 52.0678 ± 1.28731 ‖ 51.7851 ± 0.560537 | 65.3989 ± 1.50951 ‖ 64.6850 ± 0.226081 |
| utd4 | 56.6817 ± 0.722153 ‖ 56.0489 ± 0.443717 | 56.3688 ± 0.822731 ‖ 57.0818 ± 0.433261 | 57.3112 ± 10.2578 ‖ 45.1919 ± 2.06175 | 46.4077 ± 0.416075 ‖ 45.8377 ± 0.147303 | 50.2991 ± 0.436875 ‖ 50.3650 ± 0.334094 | 65.5206 ± 0.941876 ‖ 64.9972 ± 0.310650 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E dyn_model | E rollout | E synth_rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| utd2 | 2.78522 ± 5.79539 ‖ 2.77606 ± 6.82500 | -0.711797 ± 3.68970 ‖ -1.80634 ± 1.43754 | 1.64570 ± 34.1775 ‖ 1.98582 ± 2.58951 | 98.1372 ± 1.98417 ‖ 96.5398 ± 1.05339 | 98.6371 ± 1.02998 ‖ 100.879 ± 3.56205 | 97.7593 ± 1.63658 ‖ 97.2452 ± 1.29607 | 98.7286 ± 2.79422 ‖ 96.3260 ± 1.73832 | 98.1779 ± 1.38137 ‖ 98.6729 ± 2.07203 | 29.7454 ± 5.12438 ‖ 25.4294 ± 6.10301 |
| utd4 | -4.42640 ± 8.94102 ‖ -2.17303 ± 6.21243 | -1.91146 ± 2.19705 ‖ -2.64030 ± 1.45708 | 1.34180 ± 18.3785 ‖ -3.05667 ± 6.46980 | 293.332 ± 4.44207 ‖ 290.436 ± 1.86095 | 297.016 ± 3.99473 ‖ 299.041 ± 2.71300 | 293.363 ± 1.63753 ‖ 291.767 ± 0.324208 | 295.135 ± 5.25346 ‖ 293.305 ± 2.59652 | 294.954 ± 0.693386 ‖ 294.768 ± 1.16192 | 80.3638 ± 8.72466 ‖ 68.2073 ± 9.20818 |

Duration:

| config | D dyn_model | D rollout | D synth_rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|---|---|
| utd2 | 4.17959 ± 6.60795 ‖ 3.70342 ± 7.22422 | 1.26944 ± 2.64893 ‖ 0.705138 ± 0.785819 | 1.43019 ± 1.07778 ‖ -0.429221 ± 0.726658 | 101.128 ± 1.74426 ‖ 102.945 ± 2.84496 | 27.6872 ± 6.20142 ‖ 23.5836 ± 6.50518 | 101.020 ± 2.48011 ‖ 101.018 ± 1.10261 | 101.525 ± 1.38522 ‖ 105.467 ± 4.36816 | 100.635 ± 2.03175 ‖ 101.743 ± 1.97312 | 101.622 ± 3.36335 ‖ 100.801 ± 1.99747 |
| utd4 | -3.23364 ± 9.62766 ‖ -0.979463 ± 6.09206 | 0.554387 ± 2.18789 ‖ 0.0482582 ± 0.928178 | 0.882189 ± 1.35283 ‖ -2.62409 ± 2.43651 | 302.901 ± 2.00321 ‖ 304.617 ± 1.30538 | 71.1888 ± 10.0872 ‖ 60.3981 ± 8.90606 | 300.811 ± 3.15331 ‖ 300.945 ± 0.698912 | 304.589 ± 5.43087 ‖ 309.785 ± 2.81383 | 300.851 ± 0.902434 ‖ 302.316 ± 1.12270 | 302.650 ± 4.42584 ‖ 303.894 ± 2.45573 |

Mean power:

| config | P dyn_model | P rollout | P synth_rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|---|---|
| utd2 | -1.30109 ± 0.936615 ‖ -0.876332 ± 0.483094 | -1.97742 ± 1.21842 ‖ -2.49261 ± 1.33215 | 0.163378 ± 33.5094 ‖ 2.43362 ± 2.87418 | -1.46552 ± 0.330149 ‖ -2.10083 ± 0.378176 | 1.64852 ± 0.977601 ‖ 1.51057 ± 0.500740 | 2.50369 ± 0.691373 ‖ 2.11698 ± 0.381069 |
| utd4 | -1.18692 ± 0.948052 ‖ -1.21591 ± 0.451975 | -2.45238 ± 0.451895 ‖ -2.68652 ± 1.22365 | 0.342898 ± 17.1920 ‖ -0.515718 ± 4.45711 | -1.97087 ± 0.410522 ‖ -2.43390 ± 0.282995 | 5.41115 ± 1.15293 ‖ 4.87241 ± 0.294055 | 7.16392 ± 0.922388 ‖ 6.18160 ± 0.351656 |

Achieved throughput:

| config | dyn_model GFLOP/s | rollout GFLOP/s | synth_rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|---|---|
| utd2 | 0.0570964 ± 4.05924 ‖ 2.48878 ± 1.93347 | -1.19872 ± 2.62028 ‖ -0.695370 ± 0.773886 | -1.40110 ± 1.04995 ‖ -0.727387 ± 0.497545 | -0.555109 ± 0.857230 ‖ -1.43551 ± 1.37456 | 39.8901 ± 6.07953 ‖ 36.2255 ± 4.13230 |
| utd4 | 1.34252 ± 3.92178 ‖ 3.14973 ± 1.01077 | -0.513050 ± 2.19954 ‖ -0.0413695 ± 0.925065 | -0.860392 ± 1.31278 ‖ -1.51912 ± 1.26475 | -0.718075 ± 0.494333 ‖ -1.14028 ± 0.319716 | 94.2255 ± 9.38651 ‖ 86.9770 ± 5.89640 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|---|
| utd2 | -1.25296 ± 3.23983 ‖ -3.26005 ± 1.59010 | -0.711797 ± 3.68970 ‖ -1.80634 ± 1.43754 | 1.64570 ± 34.1775 ‖ 3.19081 ± 3.13584 | -0.681468 ± 0.514990 ‖ 0.439302 ± 1.78103 | -1.12035 ± 0.818290 ‖ -1.37742 ± 0.648035 | -0.911070 ± 0.690687 ‖ -0.663529 ± 1.03601 | -27.2510 ± 2.47944 ‖ -25.4358 ± 1.96782 | -0.635704 ± 1.39711 ‖ -1.83698 ± 0.869162 |
| utd4 | -2.40704 ± 2.89360 ‖ -4.22740 ± 0.699360 | -1.91146 ± 2.19705 ‖ -2.64030 ± 1.45708 | 1.34180 ± 18.3785 ‖ 1.01333 ± 4.16316 | -0.745906 ± 0.998683 ‖ -0.239747 ± 0.678251 | -1.65936 ± 0.409383 ‖ -2.05823 ± 0.0810519 | -1.26145 ± 0.173347 ‖ -1.30812 ± 0.290479 | -45.6461 ± 2.10872 ‖ -43.8694 ± 1.67932 | -1.21633 ± 1.31337 ‖ -1.67364 ± 0.649129 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|---|---|
| utd2 | 1.04092 ± 0.0488245 ‖ 1.06290 ± 0.0777621 | 1.00000 ‖ 1.00000 | 1.00000 ‖ 0.988463 ± 0.00846704 | 2.00000 ‖ 2.00000 | 2.00000 ‖ 2.00000 | 2.00000 ‖ 2.00000 | 1.78320 ± 0.0110552 ‖ 1.68141 ± 0.0388530 | 2.00000 ‖ 2.00000 |
| utd4 | 0.979513 ± 0.0886250 ‖ 1.02183 ± 0.0712141 | 1.00000 ‖ 1.00000 | 1.00000 ‖ 0.959158 ± 0.0345050 | 4.00000 ‖ 4.00000 | 4.00000 ‖ 4.00000 | 4.00000 ‖ 4.00000 | 3.31736 ± 0.0331242 ‖ 2.99495 ± 0.0788109 | 4.00000 ‖ 4.00000 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | dyn_model | rollout | synth_rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|---|---|
| utd2 | -14.6737 ± 1.17489 ‖ -13.4319 ± 1.53464 | -0.274802 ± 0.00989410 ‖ -0.369470 ± 0.0724543 | -0.00960709 ± 0.0110790 ‖ -0.0548111 ± 0.0110321 | 0.427265 ± 0.0338377 ‖ 0.510775 ± 0.0521642 | 6.30488 ± 0.468575 ‖ 5.87260 ± 0.677561 | 7.57070 ± 0.617085 ‖ 6.89544 ± 0.725317 | 0.655304 ± 0.0469329 ‖ 0.577369 ± 0.0633067 | 14.9582 ± 1.16273 ‖ 13.8562 ± 1.49064 |
| utd4 | -33.1966 ± 2.06351 ‖ -31.0666 ± 1.30622 | -0.533969 ± 0.0320501 ‖ -0.718035 ± 0.105487 | -0.0164674 ± 0.00750600 ‖ -0.124737 ± 0.0323273 | 0.956928 ± 0.0626728 ‖ 1.18991 ± 0.0467809 | 14.2675 ± 0.806485 ‖ 13.3627 ± 0.488310 | 17.0552 ± 1.08078 ‖ 15.9904 ± 0.668584 | 1.46745 ± 0.0958412 ‖ 1.36638 ± 0.0535134 | 33.7471 ± 2.02801 ‖ 31.9094 ± 1.25417 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| base | 1.22056 | 1.76769 | 3.81837 | dyn_model | 0.995821 | 0.973150 | 0.976476 |
| utd2 | 1.17834 | 1.74901 | 5.06020 | dyn_model | 0.994520 | 0.974258 | 1.00028 |
| utd4 | 1.13610 | 1.75506 | 5.94838 | dyn_model | 0.990780 | 0.975390 | 1.00737 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | dyn_model | synth_rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| base | 1.76769 (paired sd 0.0422332) | 1.28846 (paired sd 0.166389) | 9.75678 (paired sd 2.54254) | 1.35417 (paired sd 0.0179559) | 0.998842 (paired sd 0.0177614) | 1.01530 (paired sd 0.00522707) | 1.00997 (paired sd 0.0182342) | 1.01786 (paired sd 0.00643497) | 1.22056 (paired sd 0.114543) |
| utd2 | 1.74901 (paired sd 0.0409973) | 1.28718 (paired sd 0.0760936) | 10.3465 (paired sd 1.76367) | 1.34328 (paired sd 0.0122914) | 1.01017 (paired sd 0.0293008) | 1.01268 (paired sd 0.00837667) | 0.997774 (paired sd 0.0113432) | 1.02042 (paired sd 0.0161109) | 1.17834 (paired sd 0.0360779) |
| utd4 | 1.75506 (paired sd 0.0353231) | 1.31858 (paired sd 0.181748) | 9.52217 (paired sd 1.41186) | 1.34429 (paired sd 0.00743030) | 1.00397 (paired sd 0.0102973) | 1.01120 (paired sd 0.00234392) | 1.00536 (paired sd 0.00537751) | 1.01738 (paired sd 0.00464417) | 1.13610 (paired sd 0.0635668) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | dyn_model | rollout | synth_rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|---|---|
| base | 3.81837 ± 2.48009 (5/5 same sign) | 0.532104 ± 0.122998 (5/5 same sign) | 0.256200 ± 0.0396612 (5/5 same sign) | 0.0921353 ± 0.0731507 (4/5 same sign) | -2.11498 ± 1.04631 (5/5 same sign) | -2.37587 ± 1.12935 (5/5 same sign) | -0.207957 ± 0.0903167 (5/5 same sign) |
| utd2 | 5.06020 ± 1.81847 (5/5 same sign) | 0.437436 ± 0.0740107 (5/5 same sign) | 0.210996 ± 0.0272490 (5/5 same sign) | 0.175645 ± 0.0498074 (5/5 same sign) | -2.54725 ± 0.932480 (5/5 same sign) | -3.05113 ± 0.694548 (5/5 same sign) | -0.285892 ± 0.0537608 (5/5 same sign) |
| utd4 | 5.94838 ± 3.29026 (5/5 same sign) | 0.348038 ± 0.0421906 (5/5 same sign) | 0.147930 ± 0.0123383 (5/5 same sign) | 0.325112 ± 0.110489 (5/5 same sign) | -3.01978 ± 1.32683 (5/5 same sign) | -3.44067 ± 1.68189 (5/5 same sign) | -0.309017 ± 0.138752 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| base | mean Δ 0.532104; 2/32 patterns; p = 0.0625000 | mean Δ 45743.3; 2/32 patterns; p = 0.0625000 |
| utd2 | mean Δ 0.437436; 2/32 patterns; p = 0.0625000 | mean Δ 47945.9; 2/32 patterns; p = 0.0625000 |
| utd4 | mean Δ 0.348038; 2/32 patterns; p = 0.0625000 | mean Δ 50847.7; 2/32 patterns; p = 0.0625000 |


## 4.2 Sweep: width — SAC hidden_sizes ∈ {(256,256),(512,512),(1024,1024)}

Configurations: `w256` (SAC hidden (256,256)), `w512` (SAC hidden (512,512)), `base` (SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model (200×4)×7; HC: fixed rollout length 1; Ant: rollout length 1→25 (mbpo_ant.json)). Baseline for the response tables: `base` of the same environment.

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| w256 | 140997 ± 4555.36 ‖ 185042 ± 9929.96 | 2312.63 ± 25.3499 ‖ 4073.76 ± 47.0448 | 53.2193 ± 11.9230 ‖ 567.556 ± 27.9441 | 1272.69 ± 4.42470 ‖ 1716.03 ± 7.46297 | 18676.7 ± 324.941 ‖ 18648.6 ± 145.104 | 22642.1 ± 134.589 ‖ 22500.8 ± 47.8740 | 1958.56 ± 17.9772 ‖ 1942.26 ± 9.85334 | 44550.1 ± 455.190 ‖ 44807.7 ± 176.270 | 187913 ± 4849.64 ‖ 234491 ± 9869.43 |
| w512 | 143579 ± 9360.25 ‖ 179577 ± 18811.0 | 2335.78 ± 31.9842 ‖ 4108.62 ± 33.9624 | 65.7243 ± 15.2944 ‖ 642.109 ± 27.4842 | 1334.08 ± 12.5366 ‖ 1796.71 ± 8.32313 | 19544.5 ± 433.973 ‖ 19639.3 ± 333.039 | 23843.6 ± 200.818 ‖ 23816.9 ± 138.203 | 2066.58 ± 14.8405 ‖ 2057.41 ± 18.7211 | 46788.7 ± 629.312 ‖ 47310.4 ± 472.228 | 192769 ± 8814.37 ‖ 231638 ± 18563.6 |
| base | 146193 ± 7439.23 ‖ 188363 ± 20295.5 | 2428.26 ± 47.7526 ‖ 4292.41 ± 29.8641 | 75.3990 ± 16.3698 ‖ 735.651 ± 37.3179 | 1677.11 ± 15.2915 ‖ 2271.09 ± 12.1950 | 24588.5 ± 267.376 ‖ 24560.0 ± 218.652 | 29882.6 ± 104.818 ‖ 30340.0 ± 62.3096 | 2550.10 ± 26.8657 ‖ 2575.51 ± 25.8022 | 58698.3 ± 211.888 ‖ 59746.5 ± 208.801 | 207395 ± 7451.41 ‖ 253138 ± 20140.0 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| w256 | 0.0521981 ± 0.00134712 ‖ 0.0651363 ± 0.00274151 | 18.2499 ± 2.26632 ‖ 73.2635 ± 2.88784 | 5638.37 ± 99.8432 ‖ 5613.60 ± 53.7338 | 5821.53 ± 56.6730 ‖ 5787.58 ± 43.6288 |
| w512 | 0.0535470 ± 0.00244844 ‖ 0.0643439 ± 0.00515656 | 16.2833 ± 2.87380 ‖ 72.7002 ± 2.89713 | 5658.42 ± 70.6976 ‖ 5584.27 ± 49.1724 | 5859.16 ± 78.1115 ‖ 5807.12 ± 32.5766 |
| base | 0.0576096 ± 0.00206984 ‖ 0.0703161 ± 0.00559445 | 18.8339 ± 2.39855 ‖ 74.0726 ± 3.03600 | 5820.88 ± 39.1697 ‖ 5763.91 ± 33.4087 | 6042.89 ± 33.9578 ‖ 6037.16 ± 35.0893 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|---|---|
| w256 | 75.0231 ± 0.517556 ‖ 78.8818 ± 0.906080 | 1.23113 ± 0.0228696 ‖ 1.73963 ± 0.0727381 | 0.0284141 ± 0.00678889 ‖ 0.242419 ± 0.0165486 | 0.677617 ± 0.0167570 ‖ 0.732874 ± 0.0316144 | 9.94253 ± 0.227142 ‖ 7.96471 ± 0.358992 | 12.0545 ± 0.256957 ‖ 9.60906 ± 0.400460 | 1.04267 ± 0.0200888 ‖ 0.829486 ± 0.0356675 | 23.7173 ± 0.496124 ‖ 19.1361 ± 0.824148 |
| w512 | 74.4285 ± 1.48564 ‖ 77.3905 ± 2.11013 | 1.21399 ± 0.0646587 ‖ 1.78450 ± 0.169058 | 0.0343048 ± 0.00871801 ‖ 0.279379 ± 0.0353150 | 0.693380 ± 0.0364439 ‖ 0.780150 ± 0.0704630 | 10.1633 ± 0.681184 ‖ 8.53103 ± 0.832878 | 12.3925 ± 0.649010 ‖ 10.3411 ± 0.925963 | 1.07394 ± 0.0523556 ‖ 0.893373 ± 0.0814545 | 24.3232 ± 1.41617 ‖ 20.5456 ± 1.90781 |
| base | 70.4601 ± 1.04907 ‖ 74.2785 ± 2.09320 | 1.17147 ± 0.0245455 ‖ 1.70357 ± 0.124251 | 0.0365412 ± 0.00862227 ‖ 0.292741 ± 0.0357971 | 0.809395 ± 0.0267401 ‖ 0.901530 ± 0.0684953 | 11.8698 ± 0.495888 ‖ 9.75479 ± 0.832491 | 14.4221 ± 0.472909 ‖ 12.0462 ± 0.954617 | 1.23060 ± 0.0365860 ‖ 1.02264 ± 0.0822693 | 28.3319 ± 1.02110 ‖ 23.7252 ± 1.93477 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| w256 | 2.85693 ± 0.0216515 ‖ 3.82981 ± 0.0215263 | 41.9205 ± 0.323561 ‖ 41.6187 ± 0.165207 | 50.8261 ± 0.280321 ‖ 50.2167 ± 0.125178 | 4.39646 ± 0.0330796 ‖ 4.33476 ± 0.0358358 |
| w512 | 2.85146 ± 0.0214129 ‖ 3.79794 ± 0.0331508 | 41.7680 ± 0.385646 ‖ 41.5094 ± 0.309905 | 50.9633 ± 0.339629 ‖ 50.3438 ± 0.267263 | 4.41726 ± 0.0479369 ‖ 4.34881 ± 0.0207939 |
| base | 2.85721 ± 0.0304969 ‖ 3.80128 ± 0.0300127 | 41.8889 ± 0.334398 ‖ 41.1064 ± 0.243247 | 50.9094 ± 0.258129 ‖ 50.7815 ± 0.178929 | 4.34452 ± 0.0541846 ‖ 4.31077 ± 0.0449133 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | dyn_model (measured) [J/FLOP] | rollout (measured) [J/FLOP] | synth_rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|---|---|
| w256 | 5.71756e-10 ± 1.14951e-11 ‖ 4.16567e-10 ± 3.40175e-12 | 1.58486e-07 ± 1.73725e-09 ‖ 2.11049e-07 ± 2.43725e-09 | 2.69983e-11 ± 6.04855e-12 ‖ 2.20377e-11 ± 1.09363e-12 | 5.74795e-10 ± 1.00004e-11 ‖ 4.57338e-10 ± 3.55853e-12 | 8.82257e-10 ± 5.24431e-12 ‖ 6.85026e-10 ± 1.45750e-12 | 7.66036e-10 ± 7.82697e-12 ‖ 6.08610e-10 ± 2.39422e-12 | 6.12601e-10 ± 9.02267e-12 ‖ 4.31364e-10 ± 2.81789e-12 | 67.8239 ± 0.622538 ‖ 50.9871 ± 0.258664 |
| w512 | 5.71704e-10 ± 1.09539e-11 ‖ 4.15747e-10 ± 1.02489e-11 | 4.21633e-08 ± 5.77349e-10 ‖ 6.33858e-08 ± 5.23955e-10 | 2.76712e-11 ± 6.43923e-12 ‖ 2.11004e-11 ± 5.22945e-13 | 1.55877e-10 ± 3.46116e-12 ‖ 1.38353e-10 ± 2.34615e-12 | 2.42550e-10 ± 2.04282e-12 ‖ 2.11388e-10 ± 1.22663e-12 | 2.09170e-10 ± 2.81335e-12 ‖ 1.85807e-10 ± 1.85463e-12 | 4.03908e-10 ± 8.70683e-12 ‖ 3.23041e-10 ± 8.56113e-12 | 18.7559 ± 0.134690 ‖ 15.9967 ± 0.145559 |
| base | 5.80582e-10 ± 1.48182e-11 ‖ 4.37090e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59556e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83073e-10 ± 5.44247e-12 ‖ 1.78766e-10 ± 8.47149e-12 | 5.92935 ± 0.0624667 ‖ 5.51568 ± 0.0552575 |

MBPO FLOPs differ between seeds. **Mean of the per-seed ratios** (E_i/F_i averaged over seeds) ± sd, for comparison with the ratio-of-means table above:

| config | dyn_model [J/FLOP] | rollout [J/FLOP] | synth_rollout [J/FLOP] | critic [J/FLOP] | actor [J/FLOP] | GU [J/FLOP] | TOTAL [J/FLOP] |
|---|---|---|---|---|---|---|---|
| w256 | 5.71836e-10 ± 1.14951e-11 ‖ 4.16610e-10 ± 3.40175e-12 | 1.58486e-07 ± 1.73725e-09 ‖ 2.11049e-07 ± 2.43725e-09 | 2.69983e-11 ± 6.04855e-12 ‖ 2.20395e-11 ± 1.09363e-12 | 5.74795e-10 ± 1.00004e-11 ‖ 4.57338e-10 ± 3.55853e-12 | 8.82257e-10 ± 5.24431e-12 ‖ 6.85026e-10 ± 1.45750e-12 | 7.66036e-10 ± 7.82697e-12 ‖ 6.08610e-10 ± 2.39422e-12 | 6.12647e-10 ± 9.02267e-12 ‖ 4.31419e-10 ± 2.81789e-12 |
| w512 | 5.71310e-10 ± 1.09539e-11 ‖ 4.15680e-10 ± 1.02489e-11 | 4.21633e-08 ± 5.77349e-10 ‖ 6.33858e-08 ± 5.23955e-10 | 2.76712e-11 ± 6.43923e-12 ‖ 2.10931e-11 ± 5.22945e-13 | 1.55877e-10 ± 3.46116e-12 ‖ 1.38353e-10 ± 2.34615e-12 | 2.42550e-10 ± 2.04282e-12 ‖ 2.11388e-10 ± 1.22663e-12 | 2.09170e-10 ± 2.81335e-12 ‖ 1.85807e-10 ± 1.85463e-12 | 4.03741e-10 ± 8.70683e-12 ‖ 3.22739e-10 ± 8.56113e-12 |
| base | 5.80351e-10 ± 1.48182e-11 ‖ 4.37061e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59503e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83047e-10 ± 5.44247e-12 ‖ 1.78551e-10 ± 8.47149e-12 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|---|
| w256 | 0.728575 | 1.33166 | 0.816262 | 0.795654 | 0.776448 | 0.794492 | 0.704152 | 0.751757 |
| w512 | 0.727207 | 1.50334 | 0.762540 | 0.887574 | 0.871523 | 0.888308 | 0.799789 | 0.852887 |
| base | 0.752848 | 1.62566 | 0.838363 | 0.935860 | 0.944715 | 0.950778 | 0.976476 | 0.930232 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | dyn_model (measured) duration [s] | rollout (measured) duration [s] | synth_rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|---|---|
| w256 | 1211.02 ± 38.9921 ‖ 1580.59 ± 84.7862 | 19.8736 ± 0.199794 ‖ 35.9503 ± 0.177850 | 0.591295 ± 0.0126848 ‖ 4.45872 ± 0.0445189 | 400.354 ± 3.34163 ‖ 403.802 ± 1.51767 | 11.1647 ± 0.0203290 ‖ 15.0933 ± 0.0638435 | 163.839 ± 2.62203 ‖ 164.023 ± 1.29872 | 198.628 ± 0.803015 ‖ 197.905 ± 0.396742 | 17.1814 ± 0.120032 ‖ 17.0831 ± 0.0983073 | 1631.84 ± 39.9673 ‖ 2024.80 ± 83.6957 |
| w512 | 1230.02 ± 95.5958 ‖ 1532.84 ± 169.638 | 20.0232 ± 0.274176 ‖ 36.1719 ± 0.101735 | 0.606281 ± 0.0137345 ‖ 4.64312 ± 0.0603555 | 403.245 ± 4.95271 ‖ 407.009 ± 3.08714 | 11.1934 ± 0.116404 ‖ 15.0305 ± 0.0387439 | 163.982 ± 3.56951 ‖ 164.292 ± 2.53509 | 200.052 ± 1.25767 ‖ 199.242 ± 0.718037 | 17.3395 ± 0.173201 ‖ 17.2112 ± 0.105287 | 1653.89 ± 91.5725 ‖ 1980.67 ± 167.922 |
| base | 1250.07 ± 72.8336 ‖ 1605.17 ± 180.136 | 20.3785 ± 0.415238 ‖ 36.6806 ± 0.0873145 | 0.633054 ± 0.00774843 ‖ 5.07582 ± 0.0749797 | 407.584 ± 2.70475 ‖ 411.097 ± 2.42950 | 11.2917 ± 0.0973144 ‖ 15.1580 ± 0.0311855 | 165.557 ± 2.32430 ‖ 163.927 ± 1.97951 | 201.197 ± 0.836174 ‖ 202.501 ± 0.702973 | 17.1693 ± 0.155059 ‖ 17.1899 ± 0.159045 | 1678.67 ± 72.2102 ‖ 2058.02 ± 178.140 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P dyn_model [W] | P rollout [W] | P synth_rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|---|---|
| w256 | 116.433 ± 1.29227 ‖ 117.072 ± 0.325797 | 116.370 ± 1.02766 ‖ 113.314 ± 0.752407 | 90.1938 ± 21.2358 ‖ 127.299 ± 6.31220 | 111.275 ± 0.276670 ‖ 110.965 ± 0.226272 | 115.154 ± 0.975875 ‖ 115.807 ± 0.277947 | 110.037 ± 0.922288 ‖ 111.517 ± 0.345541 | 1.04578 ± 0.00863864 ‖ 1.02117 ± 0.00701471 |
| w512 | 116.824 ± 1.60167 ‖ 117.229 ± 1.43927 | 116.657 ± 0.958127 ‖ 113.586 ± 0.939156 | 108.525 ± 26.0709 ‖ 138.252 ± 4.16370 | 116.030 ± 0.578928 ‖ 116.238 ± 0.378297 | 116.607 ± 1.26914 ‖ 116.994 ± 1.15068 | 111.427 ± 0.946242 ‖ 112.492 ± 0.864684 | 1.00540 ± 0.00571119 ‖ 0.977195 ± 0.00826635 |
| base | 116.988 ± 0.953539 ‖ 117.397 ± 0.655463 | 119.160 ± 0.550654 ‖ 117.022 ± 0.953688 | 119.137 ± 25.8677 ‖ 144.882 ± 5.45432 | 144.018 ± 0.467991 ‖ 145.336 ± 0.473385 | 123.577 ± 0.913456 ‖ 123.061 ± 0.930322 | 117.983 ± 0.615299 ‖ 118.404 ± 0.501427 | 0.827400 ± 0.00273275 ‖ 0.805180 ± 0.00538864 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | dyn_model [GFLOP/s] | rollout [GFLOP/s] | synth_rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|---|---|
| w256 | 203.709 ± 6.13103 ‖ 281.031 ± 2.94575 | 0.734300 ± 0.00744140 ‖ 0.536930 ± 0.00266675 | 3334.93 ± 70.6627 ‖ 5775.99 ± 10.3258 | 145.271 ± 1.21879 ‖ 182.327 ± 0.683506 | 188.008 ± 4.13288 ‖ 268.445 ± 2.29418 | 198.362 ± 3.18541 ‖ 248.614 ± 1.96453 | 129.207 ± 0.524384 ‖ 165.972 ± 0.332739 | 97.6168 ± 0.0251155 ‖ 97.5984 ± 0.0583281 | 97.5805–97.6441 ‖ 97.5168–97.6637 |
| w512 | 204.582 ± 6.50449 ‖ 282.219 ± 10.2700 | 2.76712 ± 0.0376824 ‖ 1.79199 ± 0.00504197 | 3919.25 ± 88.5383 ‖ 6553.68 ± 35.2458 | 554.784 ± 6.73394 ‖ 625.618 ± 4.75256 | 288.969 ± 9.06190 ‖ 362.776 ± 12.9204 | 764.902 ± 16.3604 ‖ 864.183 ± 13.3426 | 491.407 ± 3.07304 ‖ 565.497 ± 2.03973 | 97.3516 ± 0.0356169 ‖ 97.2396 ± 0.0629126 | 97.3207–97.4023 ‖ 97.1777–97.3306 |
| base | 201.719 ± 6.74360 ‖ 268.627 ± 3.33295 | 10.5859 ± 0.210149 ‖ 6.39295 ± 0.0152330 | 6258.88 ± 77.5462 ‖ 9083.34 ± 23.6982 | 2151.47 ± 14.2695 ‖ 2283.56 ± 13.4483 | 675.699 ± 24.6566 ‖ 690.649 ± 37.7185 | 2974.43 ± 41.5340 ‖ 3206.03 ± 38.3417 | 1911.16 ± 7.94527 ‖ 2040.72 ± 7.09689 | 96.9650 ± 0.0406469 ‖ 97.0025 ± 0.0629928 | 96.8947–96.9927 ‖ 96.9168–97.0815 |

**Hardware composition of the measured `dynamics_model_update` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of dyn_model energy [%] | GPU share of dyn_model energy [%] | RAM share of dyn_model energy [%] | dyn_model energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| w256 | 21.4861 ± 0.466278 ‖ 21.8192 ± 0.180227 | 61.3355 ± 0.289436 ‖ 61.0978 ± 0.147340 | 17.1784 ± 0.190439 ‖ 17.0830 ± 0.0475145 | 0/500 ‖ 1/500 |
| w512 | 21.7142 ± 0.348861 ‖ 21.9286 ± 0.422484 | 61.1641 ± 0.156639 ‖ 61.0094 ± 0.232505 | 17.1217 ± 0.234647 ‖ 17.0620 ± 0.209912 | 0/500 ‖ 0/500 |
| base | 21.4036 ± 0.316290 ‖ 21.4284 ± 0.217233 | 61.5003 ± 0.203244 ‖ 61.5356 ± 0.155617 | 17.0961 ± 0.139314 ‖ 17.0360 ± 0.0949920 | 0/500 ‖ 0/500 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| w256 | 26.1516 ± 0.200584 ‖ 27.8583 ± 0.422333 | 56.6782 ± 0.156605 ‖ 54.5056 ± 0.537453 | 17.1702 ± 0.152515 ‖ 17.6362 ± 0.118385 | 500/500 ‖ 500/500 |
| w512 | 26.2085 ± 0.368262 ‖ 27.7415 ± 0.327655 | 56.6665 ± 0.453227 ‖ 54.6651 ± 0.408642 | 17.1250 ± 0.140959 ‖ 17.5934 ± 0.147149 | 500/500 ‖ 500/500 |
| base | 25.7910 ± 0.402018 ‖ 27.6475 ± 0.322486 | 57.4494 ± 0.449754 ‖ 55.2805 ± 0.439211 | 16.7596 ± 0.0798437 ‖ 17.0720 ± 0.139120 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `synthetic_rollout_generation` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of synth_rollout energy [%] | GPU share of synth_rollout energy [%] | RAM share of synth_rollout energy [%] | synth_rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| w256 | 33.9304 ± 9.40883 ‖ 27.4583 ± 1.40879 | 43.6542 ± 15.7347 ‖ 57.0052 ± 2.19973 | 22.4154 ± 6.32870 ‖ 15.5365 ± 0.791435 | 500/500 ‖ 500/500 |
| w512 | 28.5317 ± 8.46042 ‖ 25.2763 ± 0.731891 | 53.0785 ± 13.8313 ‖ 60.4228 ± 1.15221 | 18.3898 ± 5.37126 ‖ 14.3009 ± 0.421546 | 500/500 ‖ 500/500 |
| base | 26.1498 ± 6.92079 ‖ 24.0129 ± 0.939763 | 57.2526 ± 11.2969 ‖ 62.3211 ± 1.43681 | 16.5975 ± 4.37648 ‖ 13.6660 ± 0.498430 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| w256 | 23.0337 ± 0.108329 ‖ 23.3367 ± 0.0899891 | 58.9949 ± 0.124127 ‖ 58.6410 ± 0.0693716 | 17.9715 ± 0.0448313 ‖ 18.0223 ± 0.0366422 | 0/500 ‖ 0/500 |
| w512 | 22.1376 ± 0.111185 ‖ 22.3070 ± 0.0983070 | 60.6272 ± 0.119774 ‖ 60.4882 ± 0.0516557 | 17.2352 ± 0.0858443 ‖ 17.2048 ± 0.0561003 | 0/500 ‖ 0/500 |
| base | 18.0568 ± 0.118760 ‖ 18.2631 ± 0.0602319 | 68.0574 ± 0.145407 ‖ 67.9766 ± 0.0664136 | 13.8857 ± 0.0451822 ‖ 13.7603 ± 0.0448462 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | dyn_model [%] | rollout [%] | synth_rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|---|---|
| w256 | 54.6819 ± 0.515426 ‖ 54.1017 ± 0.356715 | 54.7100 ± 0.437462 ‖ 55.8985 ± 0.558466 | 74.4651 ± 20.8401 ‖ 49.8551 ± 2.53507 | 57.2124 ± 0.324638 ‖ 57.0791 ± 0.297188 | 55.2873 ± 0.399004 ‖ 54.6926 ± 0.323286 | 63.6639 ± 0.500345 ‖ 63.3376 ± 0.347530 |
| w512 | 54.7743 ± 0.510723 ‖ 53.9909 ± 0.899633 | 54.8508 ± 0.567944 ‖ 55.7166 ± 0.532098 | 62.2955 ± 17.7469 ‖ 45.8052 ± 1.32353 | 55.1445 ± 0.344601 ‖ 54.4440 ± 0.463998 | 54.8740 ± 0.381113 ‖ 54.0967 ± 0.782121 | 63.9842 ± 0.505282 ‖ 63.2833 ± 0.387484 |
| base | 56.3406 ± 0.609099 ‖ 55.8464 ± 0.519715 | 55.3099 ± 0.0464751 ‖ 56.0243 ± 0.227975 | 57.9021 ± 15.0568 ‖ 45.3015 ± 1.71294 | 45.7635 ± 0.167734 ‖ 45.1088 ± 0.176904 | 53.3357 ± 0.515845 ‖ 53.2775 ± 0.609384 | 65.9074 ± 0.314375 ‖ 65.5593 ± 0.279103 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E dyn_model | E rollout | E synth_rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| w256 | -3.30234 ± 6.66943 ‖ -1.20630 ± 7.34611 | -4.72928 ± 2.29246 ‖ -5.09199 ± 1.10572 | -25.1684 ± 31.5505 ‖ -22.6596 ± 6.02813 | -24.1074 ± 0.932229 ‖ -24.4395 ± 0.294019 | -24.0403 ± 1.25529 ‖ -24.0617 ± 1.12704 | -24.2279 ± 0.682006 ‖ -25.8375 ± 0.177307 | -23.1842 ± 1.48335 ‖ -24.5812 ± 0.866638 | -24.1025 ± 0.819602 ‖ -25.0025 ± 0.470751 | -9.26738 ± 4.76392 ‖ -7.09651 ± 4.88009 |
| w512 | -1.49749 ± 9.30274 ‖ -4.41810 ± 7.92686 | -3.78993 ± 1.63853 ‖ -4.27583 ± 1.26673 | -6.90410 ± 38.7708 ‖ -12.6646 ± 2.46556 | -20.4524 ± 0.455368 ‖ -20.8861 ± 0.568470 | -20.4978 ± 2.35801 ‖ -20.0350 ± 1.18852 | -20.2092 ± 0.596614 ‖ -21.4999 ± 0.381767 | -18.9567 ± 0.674673 ‖ -20.1120 ± 0.910605 | -20.2880 ± 1.16216 ‖ -20.8153 ± 0.690451 | -6.91643 ± 6.18003 ‖ -8.37437 ± 5.64590 |

Duration:

| config | D dyn_model | D rollout | D synth_rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|---|---|
| w256 | -2.84077 ± 6.68137 ‖ -0.896549 ± 8.01273 | -2.44954 ± 1.97301 ‖ -1.98969 ± 0.694597 | -6.58277 ± 2.41697 ‖ -12.1430 ± 1.51961 | -1.77064 ± 0.999401 ‖ -1.77164 ± 0.705566 | -2.62718 ± 5.17092 ‖ -1.24333 ± 6.01679 | -1.11857 ± 0.947567 ‖ -0.426823 ± 0.382420 | -1.02719 ± 1.75394 ‖ 0.0733551 ± 1.66262 | -1.27423 ± 0.748790 ‖ -2.26895 ± 0.288410 | 0.0811269 ± 1.53474 ‖ -0.612727 ± 1.22458 |
| w512 | -1.21149 ± 10.9954 ‖ -4.19170 ± 9.14636 | -1.72822 ± 1.25459 ‖ -1.38643 ± 0.313102 | -4.20295 ± 3.11737 ‖ -8.52147 ± 0.585354 | -1.05989 ± 1.47142 ‖ -0.993027 ± 0.765062 | -1.26959 ± 7.87920 ‖ -3.56354 ± 7.03306 | -0.868698 ± 0.894463 ‖ -0.840474 ± 0.418609 | -0.923907 ± 3.07506 ‖ 0.227238 ± 1.45742 | -0.568681 ± 0.516390 ‖ -1.60889 ± 0.377968 | 0.993737 ± 0.908852 ‖ 0.129943 ± 1.00183 |

Mean power:

| config | P dyn_model | P rollout | P synth_rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|---|---|
| w256 | -0.472816 ± 0.960569 ‖ -0.273126 ± 0.774272 | -2.34149 ± 0.765219 ‖ -3.16645 ± 0.717577 | -19.2104 ± 36.3457 ‖ -12.0043 ± 6.13423 | -22.7341 ± 0.385014 ‖ -23.6492 ± 0.224316 | -6.81544 ± 0.585793 ‖ -5.88905 ± 0.900906 | -6.73561 ± 0.560385 ‖ -5.81416 ± 0.685784 |
| w512 | -0.129953 ± 1.92150 ‖ -0.140720 ± 1.31429 | -2.09988 ± 0.849790 ‖ -2.92851 ± 1.38805 | -2.35271 ± 42.9388 ‖ -4.53029 ± 2.56749 | -19.4326 ± 0.510574 ‖ -20.0212 ± 0.293579 | -5.63428 ± 1.43018 ‖ -4.92497 ± 1.21779 | -5.55339 ± 1.08023 ‖ -4.99168 ± 0.913921 |

Achieved throughput:

| config | dyn_model GFLOP/s | rollout GFLOP/s | synth_rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|---|---|
| w256 | 1.04463 ± 3.49669 ‖ 4.63859 ± 2.22933 | -93.0614 ± 0.143011 ‖ -91.6011 ± 0.0597445 | -46.7089 ± 1.38464 ‖ -36.4110 ± 0.121120 | -93.2476 ± 0.0688434 ‖ -92.0154 ± 0.0577741 | -72.1570 ± 0.799704 ‖ -61.0267 ± 2.39998 |
| w512 | 1.57471 ± 6.21570 ‖ 5.05496 ± 3.42404 | -73.8563 ± 0.332388 ‖ -71.9692 ± 0.0889993 | -37.3640 ± 2.00666 ‖ -27.8494 ± 0.345411 | -74.2125 ± 0.381256 ‖ -72.6029 ± 0.212178 | -57.1714 ± 2.47363 ‖ -47.3648 ± 3.09820 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|---|
| w256 | -1.42741 ± 2.73323 ‖ -4.66966 ± 1.47245 | 1308.00 ± 33.8801 ‖ 1052.99 ± 13.4329 | 50.3961 ± 63.4100 ‖ 38.3759 ± 9.53709 | 1051.01 ± 19.0213 ‖ 878.630 ± 14.5243 | 1035.27 ± 10.2183 ‖ 833.039 ± 2.23069 | 1044.36 ± 12.3578 ‖ 856.261 ± 6.00235 | 234.871 ± 8.50563 ‖ 142.110 ± 12.8881 | 1044.05 ± 22.0922 ‖ 824.480 ± 10.6232 |
| w512 | -1.47477 ± 4.20088 ‖ -4.89772 ± 1.84182 | 274.526 ± 6.37847 ‖ 246.301 ± 4.58266 | 55.2803 ± 64.6682 ‖ 32.3270 ± 3.81188 | 212.191 ± 9.25951 ‖ 196.026 ± 4.39982 | 212.099 ± 2.33364 ‖ 187.919 ± 1.40023 | 212.477 ± 4.55577 ‖ 191.939 ± 2.54556 | 120.775 ± 9.68834 ‖ 81.0183 ± 8.25389 | 216.339 ± 2.63347 ‖ 190.039 ± 3.30601 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|---|---|
| w256 | 0.980666 ± 0.0570532 ‖ 1.03585 ± 0.0657892 | 0.0676638 ‖ 0.0823144 | 0.497563 ‖ 0.558682 ± 0.0103023 | 0.0659940 ‖ 0.0775965 | 0.0667439 ‖ 0.0794849 | 0.0663229 ‖ 0.0784278 | 0.270813 ± 0.00832499 ‖ 0.383817 ± 0.00859759 | 0.0671437 ‖ 0.0815797 |
| w512 | 0.997991 ± 0.0528337 ‖ 1.00417 ± 0.0658589 | 0.256885 ‖ 0.276419 | 0.599535 ‖ 0.660016 ± 0.00417022 | 0.254659 ‖ 0.270129 | 0.255659 ‖ 0.272647 | 0.255097 ‖ 0.271237 | 0.421301 ± 0.0105244 ‖ 0.506163 ± 0.0202529 | 0.256191 ‖ 0.275439 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | dyn_model | rollout | synth_rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|---|---|
| w256 | 4.56302 ± 1.26512 ‖ 4.60335 ± 1.55642 | 0.0596641 ± 0.0382447 ‖ 0.0360548 ± 0.102473 | -0.00812710 ± 0.0124923 ‖ -0.0503218 ± 0.0368753 | -0.131778 ± 0.0358786 ‖ -0.168656 ± 0.0471365 | -1.92724 ± 0.528995 ‖ -1.79008 ± 0.644140 | -2.36761 ± 0.621324 ‖ -2.43719 ± 0.679847 | -0.187928 ± 0.0466856 ‖ -0.193155 ± 0.0578838 | -4.61456 ± 1.21648 ‖ -4.58908 ± 1.42192 |
| w512 | 3.96838 ± 2.11540 ‖ 3.11202 ± 1.61364 | 0.0425243 ± 0.0670387 ‖ 0.0809298 ± 0.115036 | -0.00223641 ± 0.0153593 ‖ -0.0133623 ± 0.0134002 | -0.116015 ± 0.0488422 ‖ -0.121380 ± 0.0544375 | -1.70642 ± 1.02062 ‖ -1.22376 ± 0.653973 | -2.02958 ± 0.898994 ‖ -1.70519 ± 0.723230 | -0.156657 ± 0.0690112 ‖ -0.129269 ± 0.0684537 | -4.00867 ± 2.03486 ‖ -3.17959 ± 1.49715 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| w256 | 1.24787 | 1.76152 | 3.85870 | dyn_model | 1.00567 | 0.976461 | 0.704152 |
| w512 | 1.20163 | 1.75899 | 2.96201 | dyn_model | 1.00332 | 0.971942 | 0.799789 |
| base | 1.22056 | 1.76769 | 3.81837 | dyn_model | 0.995821 | 0.973150 | 0.976476 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | dyn_model | synth_rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| w256 | 1.76152 (paired sd 0.0381742) | 1.31238 (paired sd 0.0663659) | 10.6645 (paired sd 2.64027) | 1.34835 (paired sd 0.00674861) | 0.998499 (paired sd 0.0241912) | 0.993758 (paired sd 0.00636267) | 0.991676 (paired sd 0.00918278) | 1.00578 (paired sd 0.0133829) | 1.24787 (paired sd 0.0484656) |
| w512 | 1.75899 (paired sd 0.0250976) | 1.25072 (paired sd 0.0961457) | 9.76974 (paired sd 3.05268) | 1.34678 (paired sd 0.0164134) | 1.00485 (paired sd 0.0338867) | 0.998882 (paired sd 0.0127959) | 0.995563 (paired sd 0.0159417) | 1.01115 (paired sd 0.0208694) | 1.20163 (paired sd 0.0672867) |
| base | 1.76769 (paired sd 0.0422332) | 1.28846 (paired sd 0.166389) | 9.75678 (paired sd 2.54254) | 1.35417 (paired sd 0.0179559) | 0.998842 (paired sd 0.0177614) | 1.01530 (paired sd 0.00522707) | 1.00997 (paired sd 0.0182342) | 1.01786 (paired sd 0.00643497) | 1.22056 (paired sd 0.114543) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | dyn_model | rollout | synth_rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|---|---|
| w256 | 3.85870 ± 0.938104 (5/5 same sign) | 0.508495 ± 0.0598503 (5/5 same sign) | 0.214005 ± 0.0145017 (5/5 same sign) | 0.0552573 ± 0.0310580 (5/5 same sign) | -1.97782 ± 0.433394 (5/5 same sign) | -2.44545 ± 0.406149 (5/5 same sign) | -0.213184 ± 0.0330463 (5/5 same sign) |
| w512 | 2.96201 ± 1.66984 (5/5 same sign) | 0.570510 ± 0.139294 (5/5 same sign) | 0.245074 ± 0.0341487 (5/5 same sign) | 0.0867702 ± 0.0546923 (5/5 same sign) | -1.63232 ± 0.738028 (5/5 same sign) | -2.05148 ± 0.682306 (5/5 same sign) | -0.180568 ± 0.0666783 (5/5 same sign) |
| base | 3.81837 ± 2.48009 (5/5 same sign) | 0.532104 ± 0.122998 (5/5 same sign) | 0.256200 ± 0.0396612 (5/5 same sign) | 0.0921353 ± 0.0731507 (4/5 same sign) | -2.11498 ± 1.04631 (5/5 same sign) | -2.37587 ± 1.12935 (5/5 same sign) | -0.207957 ± 0.0903167 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| w256 | mean Δ 0.508495; 2/32 patterns; p = 0.0625000 | mean Δ 46577.3; 2/32 patterns; p = 0.0625000 |
| w512 | mean Δ 0.570510; 2/32 patterns; p = 0.0625000 | mean Δ 38868.7; 2/32 patterns; p = 0.0625000 |
| base | mean Δ 0.532104; 2/32 patterns; p = 0.0625000 | mean Δ 45743.3; 2/32 patterns; p = 0.0625000 |


## 4.3 Sweep: batch size — SAC batch_size ∈ {256,512,1024}

Configurations: `base` (SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model (200×4)×7; HC: fixed rollout length 1; Ant: rollout length 1→25 (mbpo_ant.json)), `b512` (SAC batch 512), `b1024` (SAC batch 1024). Baseline for the response tables: `base` of the same environment.

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| base | 146193 ± 7439.23 ‖ 188363 ± 20295.5 | 2428.26 ± 47.7526 ‖ 4292.41 ± 29.8641 | 75.3990 ± 16.3698 ‖ 735.651 ± 37.3179 | 1677.11 ± 15.2915 ‖ 2271.09 ± 12.1950 | 24588.5 ± 267.376 ‖ 24560.0 ± 218.652 | 29882.6 ± 104.818 ‖ 30340.0 ± 62.3096 | 2550.10 ± 26.8657 ‖ 2575.51 ± 25.8022 | 58698.3 ± 211.888 ‖ 59746.5 ± 208.801 | 207395 ± 7451.41 ‖ 253138 ± 20140.0 |
| b512 | 137282 ± 4996.11 ‖ 170819 ± 21221.5 | 2383.39 ± 14.7420 ‖ 4113.46 ± 26.6552 | 66.5633 ± 8.36530 ‖ 723.897 ± 43.7920 | 2187.14 ± 2.92658 ‖ 3526.11 ± 16.5795 | 29118.0 ± 372.208 ‖ 29700.8 ± 489.708 | 35124.4 ± 68.2198 ‖ 35554.4 ± 92.3858 | 2939.47 ± 24.7621 ‖ 2955.71 ± 37.6457 | 69369.1 ± 359.157 ‖ 71737.0 ± 397.916 | 209101 ± 4906.27 ‖ 247393 ± 21199.3 |
| b1024 | 138166 ± 4828.83 ‖ 178176 ± 15397.2 | 2362.80 ± 15.1644 ‖ 4178.11 ± 17.1951 | 80.7041 ± 21.8062 ‖ 723.343 ± 26.6858 | 2939.44 ± 14.1433 ‖ 6011.27 ± 34.1464 | 32497.7 ± 248.337 ‖ 34581.0 ± 301.119 | 39364.2 ± 64.5743 ‖ 40674.7 ± 95.8101 | 3255.96 ± 22.9396 ‖ 3350.50 ± 12.8938 | 78057.3 ± 275.017 ‖ 84617.4 ± 317.159 | 218667 ± 5009.72 ‖ 267695 ± 15140.5 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| base | 0.0576096 ± 0.00206984 ‖ 0.0703161 ± 0.00559445 | 18.8339 ± 2.39855 ‖ 74.0726 ± 3.03600 | 5820.88 ± 39.1697 ‖ 5763.91 ± 33.4087 | 6042.89 ± 33.9578 ‖ 6037.16 ± 35.0893 |
| b512 | 0.0580837 ± 0.00136285 ‖ 0.0687203 ± 0.00588869 | 16.3251 ± 2.89480 ‖ 72.0251 ± 2.66722 | 5615.17 ± 60.8780 ‖ 5553.44 ± 56.8559 | 5903.30 ± 19.8048 ‖ 5787.63 ± 70.0316 |
| b1024 | 0.0607409 ± 0.00139159 ‖ 0.0743597 ± 0.00420570 | 14.2428 ± 0.236951 ‖ 75.1844 ± 2.12606 | 5588.43 ± 88.1748 ‖ 5591.04 ± 71.5459 | 5900.72 ± 62.5926 ‖ 5890.46 ± 52.1521 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|---|---|
| base | 70.4601 ± 1.04907 ‖ 74.2785 ± 2.09320 | 1.17147 ± 0.0245455 ‖ 1.70357 ± 0.124251 | 0.0365412 ± 0.00862227 ‖ 0.292741 ± 0.0357971 | 0.809395 ± 0.0267401 ‖ 0.901530 ± 0.0684953 | 11.8698 ± 0.495888 ‖ 9.75479 ± 0.832491 | 14.4221 ± 0.472909 ‖ 12.0462 ± 0.954617 | 1.23060 ± 0.0365860 ‖ 1.02264 ± 0.0822693 | 28.3319 ± 1.02110 ‖ 23.7252 ± 1.93477 |
| b512 | 65.6378 ± 0.853501 ‖ 68.8655 ± 2.67329 | 1.14035 ± 0.0286950 ‖ 1.67205 ± 0.136916 | 0.0318817 ± 0.00443845 ‖ 0.295425 ± 0.0422322 | 1.04642 ± 0.0240806 ‖ 1.43361 ± 0.121284 | 13.9322 ± 0.405996 ‖ 12.0764 ± 1.06627 | 16.8050 ± 0.383795 ‖ 14.4549 ± 1.21603 | 1.40641 ± 0.0366246 ‖ 1.20211 ± 0.108186 | 33.1900 ± 0.825468 ‖ 29.1671 ± 2.49461 |
| b1024 | 63.1716 ± 0.776505 ‖ 66.4703 ± 1.96939 | 1.08093 ± 0.0217221 ‖ 1.56465 ± 0.0860013 | 0.0368735 ± 0.00974388 ‖ 0.270643 ± 0.0127215 | 1.34473 ± 0.0266339 ‖ 2.25097 ± 0.119905 | 14.8670 ± 0.306639 ‖ 12.9551 ± 0.820241 | 18.0093 ± 0.404642 ‖ 15.2337 ± 0.869856 | 1.48955 ± 0.0315023 ‖ 1.25463 ± 0.0666834 | 35.7106 ± 0.756968 ‖ 31.6944 ± 1.87434 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| base | 2.85721 ± 0.0304969 ‖ 3.80128 ± 0.0300127 | 41.8889 ± 0.334398 ‖ 41.1064 ± 0.243247 | 50.9094 ± 0.258129 ‖ 50.7815 ± 0.178929 | 4.34452 ± 0.0541846 ‖ 4.31077 ± 0.0449133 |
| b512 | 3.15297 ± 0.0182210 ‖ 4.91551 ± 0.0446789 | 41.9743 ± 0.327665 ‖ 41.4004 ± 0.462083 | 50.6352 ± 0.266799 ‖ 49.5636 ± 0.351995 | 4.23759 ± 0.0495749 ‖ 4.12047 ± 0.0712722 |
| b1024 | 3.76577 ± 0.0189786 ‖ 7.10423 ± 0.0627982 | 41.6327 ± 0.183816 ‖ 40.8669 ± 0.213059 | 50.4303 ± 0.139464 ‖ 48.0692 ± 0.134337 | 4.17131 ± 0.0366514 ‖ 3.95967 ± 0.0287309 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | dyn_model (measured) [J/FLOP] | rollout (measured) [J/FLOP] | synth_rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|---|---|
| base | 5.80582e-10 ± 1.48182e-11 ‖ 4.37090e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59556e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83073e-10 ± 5.44247e-12 ‖ 1.78766e-10 ± 8.47149e-12 | 5.92935 ± 0.0624667 ‖ 5.51568 ± 0.0552575 |
| b512 | 5.67242e-10 ± 9.73126e-12 ‖ 4.11571e-10 ± 1.00622e-11 | 1.10519e-08 ± 6.83593e-11 ‖ 1.75417e-08 ± 1.13670e-10 | 1.68016e-11 ± 2.11153e-12 ‖ 1.57138e-11 ± 5.55224e-13 | 2.95699e-11 ± 3.77985e-13 ‖ 2.82599e-11 ± 4.65951e-13 | 4.56739e-11 ± 8.87094e-14 ‖ 4.30186e-11 ± 1.11781e-13 | 3.95548e-11 ± 2.04794e-13 ‖ 3.82093e-11 ± 2.11942e-13 | 1.04554e-10 ± 2.07658e-12 ‖ 1.05777e-10 ± 6.70888e-12 | 6.83469 ± 0.0575755 ‖ 6.32989 ± 0.0806214 |
| b1024 | 5.71753e-10 ± 2.70198e-12 ‖ 4.05703e-10 ± 5.80026e-12 | 1.09564e-08 ± 7.03180e-11 ‖ 1.78174e-08 ± 7.33280e-11 | 2.03709e-11 ± 5.50422e-12 ‖ 1.59088e-11 ± 7.19199e-13 | 1.65010e-11 ± 1.26096e-13 ‖ 1.64517e-11 ± 1.43255e-13 | 2.55936e-11 ± 4.19845e-14 ‖ 2.46069e-11 ± 5.79622e-14 | 2.22545e-11 ± 7.84085e-14 ‖ 2.25349e-11 ± 8.44642e-14 | 5.82597e-11 ± 1.19887e-12 ‖ 6.31381e-11 ± 3.05899e-12 | 7.57058 ± 0.0533378 ‖ 7.17537 ± 0.0276132 |

MBPO FLOPs differ between seeds. **Mean of the per-seed ratios** (E_i/F_i averaged over seeds) ± sd, for comparison with the ratio-of-means table above:

| config | dyn_model [J/FLOP] | rollout [J/FLOP] | synth_rollout [J/FLOP] | critic [J/FLOP] | actor [J/FLOP] | GU [J/FLOP] | TOTAL [J/FLOP] |
|---|---|---|---|---|---|---|---|
| base | 5.80351e-10 ± 1.48182e-11 ‖ 4.37061e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59503e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83047e-10 ± 5.44247e-12 ‖ 1.78551e-10 ± 8.47149e-12 |
| b512 | 5.67267e-10 ± 9.73126e-12 ‖ 4.11801e-10 ± 1.00622e-11 | 1.10519e-08 ± 6.83593e-11 ‖ 1.75417e-08 ± 1.13670e-10 | 1.68016e-11 ± 2.11153e-12 ‖ 1.57038e-11 ± 5.55224e-13 | 2.95699e-11 ± 3.77985e-13 ‖ 2.82599e-11 ± 4.65951e-13 | 4.56739e-11 ± 8.87094e-14 ‖ 4.30186e-11 ± 1.11781e-13 | 3.95548e-11 ± 2.04794e-13 ‖ 3.82093e-11 ± 2.11942e-13 | 1.04548e-10 ± 2.07658e-12 ‖ 1.05660e-10 ± 6.70888e-12 |
| b1024 | 5.71786e-10 ± 2.70198e-12 ‖ 4.05549e-10 ± 5.80026e-12 | 1.09564e-08 ± 7.03180e-11 ‖ 1.78174e-08 ± 7.33280e-11 | 2.03709e-11 ± 5.50422e-12 ‖ 1.59145e-11 ± 7.19199e-13 | 1.65010e-11 ± 1.26096e-13 ‖ 1.64517e-11 ± 1.43255e-13 | 2.55936e-11 ± 4.19845e-14 ‖ 2.46069e-11 ± 5.79622e-14 | 2.22545e-11 ± 7.84085e-14 ‖ 2.25349e-11 ± 8.44642e-14 | 5.82574e-11 ± 1.19887e-12 ‖ 6.31184e-11 ± 3.05899e-12 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|---|
| base | 0.752848 | 1.62566 | 0.838363 | 0.935860 | 0.944715 | 0.950778 | 0.976476 | 0.930232 |
| b512 | 0.725565 | 1.58721 | 0.935259 | 0.955697 | 0.941864 | 0.965982 | 1.01170 | 0.926140 |
| b1024 | 0.709578 | 1.62620 | 0.780955 | 0.997009 | 0.961451 | 1.01260 | 1.08374 | 0.947797 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | dyn_model (measured) duration [s] | rollout (measured) duration [s] | synth_rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|---|---|
| base | 1250.07 ± 72.8336 ‖ 1605.17 ± 180.136 | 20.3785 ± 0.415238 ‖ 36.6806 ± 0.0873145 | 0.633054 ± 0.00774843 ‖ 5.07582 ± 0.0749797 | 407.584 ± 2.70475 ‖ 411.097 ± 2.42950 | 11.2917 ± 0.0973144 ‖ 15.1580 ± 0.0311855 | 165.557 ± 2.32430 ‖ 163.927 ± 1.97951 | 201.197 ± 0.836174 ‖ 202.501 ± 0.702973 | 17.1693 ± 0.155059 ‖ 17.1899 ± 0.159045 | 1678.67 ± 72.2102 ‖ 2058.02 ± 178.140 |
| b512 | 1188.74 ± 48.0099 ‖ 1471.52 ± 185.423 | 20.3308 ± 0.117561 ‖ 36.5774 ± 0.122721 | 0.636423 ± 0.00235594 ‖ 5.05308 ± 0.0922807 | 416.875 ± 3.72760 ‖ 424.627 ± 4.58935 | 12.7632 ± 0.0585246 ‖ 20.2945 ± 0.0636498 | 169.927 ± 2.84086 ‖ 170.959 ± 3.80276 | 204.971 ± 0.757392 ‖ 204.635 ± 0.824489 | 17.1532 ± 0.114284 ‖ 17.0111 ± 0.148752 | 1626.58 ± 46.0279 ‖ 1937.77 ± 184.527 |
| b1024 | 1193.50 ± 39.8849 ‖ 1508.19 ± 139.421 | 20.2730 ± 0.0494940 ‖ 36.7437 ± 0.235217 | 0.639325 ± 0.00495490 ‖ 4.97503 ± 0.0557903 | 420.908 ± 2.64711 ‖ 444.767 ± 3.63136 | 15.4002 ± 0.0805145 ‖ 30.7160 ± 0.0960614 | 170.265 ± 1.84505 ‖ 176.709 ± 2.39057 | 206.236 ± 0.782079 ‖ 207.842 ± 1.38150 | 17.0583 ± 0.0838337 ‖ 17.1203 ± 0.0591826 | 1635.32 ± 41.0892 ‖ 1994.68 ± 136.587 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P dyn_model [W] | P rollout [W] | P synth_rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|---|---|
| base | 116.988 ± 0.953539 ‖ 117.397 ± 0.655463 | 119.160 ± 0.550654 ‖ 117.022 ± 0.953688 | 119.137 ± 25.8677 ‖ 144.882 ± 5.45432 | 144.018 ± 0.467991 ‖ 145.336 ± 0.473385 | 123.577 ± 0.913456 ‖ 123.061 ± 0.930322 | 117.983 ± 0.615299 ‖ 118.404 ± 0.501427 | 0.827400 ± 0.00273275 ‖ 0.805180 ± 0.00538864 |
| b512 | 115.502 ± 0.868520 ‖ 116.117 ± 1.19097 | 117.231 ± 0.506331 ‖ 112.460 ± 0.766012 | 104.584 ± 13.0546 ‖ 143.177 ± 6.19380 | 166.407 ± 0.767381 ‖ 168.949 ± 0.959206 | 128.568 ± 0.856480 ‖ 127.767 ± 1.54311 | 122.127 ± 0.663426 ‖ 122.217 ± 1.02840 | 0.704486 ± 0.00122041 ‖ 0.665666 ± 0.00676717 |
| b1024 | 115.761 ± 0.253523 ‖ 118.190 ± 0.837502 | 116.549 ± 0.623674 ‖ 113.714 ± 0.968484 | 126.162 ± 33.6648 ‖ 145.439 ± 6.44940 | 185.453 ± 0.601500 ‖ 190.257 ± 0.892529 | 133.721 ± 0.317689 ‖ 134.291 ± 1.61125 | 126.784 ± 0.152877 ‖ 128.425 ± 1.13049 | 0.628462 ± 0.00422109 ‖ 0.597685 ± 0.00392029 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | dyn_model [GFLOP/s] | rollout [GFLOP/s] | synth_rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|---|---|
| base | 201.719 ± 6.74360 ‖ 268.627 ± 3.33295 | 10.5859 ± 0.210149 ‖ 6.39295 ± 0.0152330 | 6258.88 ± 77.5462 ‖ 9083.34 ± 23.6982 | 2151.47 ± 14.2695 ‖ 2283.56 ± 13.4483 | 675.699 ± 24.6566 ‖ 690.649 ± 37.7185 | 2974.43 ± 41.5340 ‖ 3206.03 ± 38.3417 | 1911.16 ± 7.94527 ‖ 2040.72 ± 7.09689 | 96.9650 ± 0.0406469 ‖ 97.0025 ± 0.0629928 | 96.8947–96.9927 ‖ 96.9168–97.0815 |
| b512 | 203.679 ± 4.97252 ‖ 282.161 ± 9.71263 | 10.6076 ± 0.0613209 ‖ 6.41101 ± 0.0215729 | 6225.06 ± 23.0956 ‖ 9115.49 ± 87.5398 | 4207.15 ± 37.4028 ‖ 4421.88 ± 47.6556 | 1230.22 ± 30.3368 ‖ 1213.78 ± 90.3035 | 5796.22 ± 95.7194 ‖ 6150.02 ± 136.529 | 3751.92 ± 13.8436 ‖ 4038.89 ± 16.2553 | 97.1069 ± 0.0349667 ‖ 97.2380 ± 0.0533050 | 97.0768–97.1669 ‖ 97.1501–97.2869 |
| b1024 | 202.459 ± 1.09469 ‖ 291.501 ± 5.98198 | 10.6375 ± 0.0259803 ‖ 6.38215 ± 0.0407248 | 6197.03 ± 47.7649 ‖ 9139.10 ± 21.6605 | 8333.41 ± 52.2918 ‖ 8442.95 ± 68.7204 | 2296.22 ± 52.9826 ‖ 2132.52 ± 126.978 | 11568.0 ± 124.962 ‖ 11896.9 ± 160.876 | 7457.80 ± 28.2284 ‖ 7953.32 ± 52.5175 | 97.1612 ± 0.0476162 ‖ 97.2161 ± 0.0477588 | 97.1160–97.2151 ‖ 97.1638–97.2916 |

**Hardware composition of the measured `dynamics_model_update` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of dyn_model energy [%] | GPU share of dyn_model energy [%] | RAM share of dyn_model energy [%] | dyn_model energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 21.4036 ± 0.316290 ‖ 21.4284 ± 0.217233 | 61.5003 ± 0.203244 ‖ 61.5356 ± 0.155617 | 17.0961 ± 0.139314 ‖ 17.0360 ± 0.0949920 | 0/500 ‖ 0/500 |
| b512 | 22.0041 ± 0.334668 ‖ 22.3017 ± 0.418677 | 60.6800 ± 0.238769 ‖ 60.4734 ± 0.249782 | 17.3159 ± 0.130204 ‖ 17.2249 ± 0.176031 | 0/500 ‖ 0/500 |
| b1024 | 21.9189 ± 0.119527 ‖ 22.7060 ± 0.215703 | 60.8047 ± 0.118694 ‖ 60.3721 ± 0.116997 | 17.2764 ± 0.0377757 ‖ 16.9219 ± 0.120453 | 0/500 ‖ 0/500 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 25.7910 ± 0.402018 ‖ 27.6475 ± 0.322486 | 57.4494 ± 0.449754 ‖ 55.2805 ± 0.439211 | 16.7596 ± 0.0798437 ‖ 17.0720 ± 0.139120 | 500/500 ‖ 500/500 |
| b512 | 26.4265 ± 0.260668 ‖ 28.2272 ± 0.515274 | 56.5343 ± 0.327505 ‖ 54.0068 ± 0.625446 | 17.0392 ± 0.0735847 ‖ 17.7659 ± 0.122332 | 500/500 ‖ 500/500 |
| b1024 | 26.8266 ± 0.278000 ‖ 27.8576 ± 0.343443 | 56.0332 ± 0.316219 ‖ 54.5701 ± 0.424147 | 17.1402 ± 0.0925858 ‖ 17.5722 ± 0.150849 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `synthetic_rollout_generation` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of synth_rollout energy [%] | GPU share of synth_rollout energy [%] | RAM share of synth_rollout energy [%] | synth_rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 26.1498 ± 6.92079 ‖ 24.0129 ± 0.939763 | 57.2526 ± 11.2969 ‖ 62.3211 ± 1.43681 | 16.5975 ± 4.37648 ‖ 13.6660 ± 0.498430 | 500/500 ‖ 500/500 |
| b512 | 29.0377 ± 3.61592 ‖ 24.5555 ± 1.04620 | 52.5716 ± 5.98351 ‖ 61.6178 ± 1.62778 | 18.3908 ± 2.37373 ‖ 13.8267 ± 0.584345 | 500/500 ‖ 500/500 |
| b1024 | 25.0853 ± 6.22402 ‖ 24.2713 ± 1.03672 | 59.1384 ± 10.2256 ‖ 62.1172 ± 1.65317 | 15.7764 ± 4.00249 ‖ 13.6116 ± 0.617736 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 18.0568 ± 0.118760 ‖ 18.2631 ± 0.0602319 | 68.0574 ± 0.145407 ‖ 67.9766 ± 0.0664136 | 13.8857 ± 0.0451822 ‖ 13.7603 ± 0.0448462 | 0/500 ‖ 0/500 |
| b512 | 15.8980 ± 0.106214 ‖ 15.8661 ± 0.0623918 | 72.0842 ± 0.149451 ‖ 72.2968 ± 0.0613364 | 12.0177 ± 0.0556100 ‖ 11.8371 ± 0.0674880 | 0/500 ‖ 0/500 |
| b1024 | 14.3406 ± 0.0732433 ‖ 14.2904 ± 0.0603761 | 74.8760 ± 0.101544 ‖ 75.1984 ± 0.0618120 | 10.7834 ± 0.0349297 ‖ 10.5112 ± 0.0494194 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | dyn_model [%] | rollout [%] | synth_rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|---|---|
| base | 56.3406 ± 0.609099 ‖ 55.8464 ± 0.519715 | 55.3099 ± 0.0464751 ‖ 56.0243 ± 0.227975 | 57.9021 ± 15.0568 ‖ 45.3015 ± 1.71294 | 45.7635 ± 0.167734 ‖ 45.1088 ± 0.176904 | 53.3357 ± 0.515845 ‖ 53.2775 ± 0.609384 | 65.9074 ± 0.314375 ‖ 65.5593 ± 0.279103 |
| b512 | 55.4028 ± 0.405725 ‖ 54.2629 ± 0.485920 | 54.5850 ± 0.446802 ‖ 56.0257 ± 0.448005 | 61.9859 ± 8.05697 ‖ 44.0660 ± 1.79640 | 38.4544 ± 0.339279 ‖ 37.2927 ± 0.276277 | 49.7725 ± 0.397616 ‖ 49.3163 ± 0.498653 | 63.9892 ± 0.329670 ‖ 63.0037 ± 0.159885 |
| b1024 | 55.1365 ± 0.298738 ‖ 53.9690 ± 0.541991 | 54.7645 ± 0.324549 ‖ 56.0928 ± 0.377583 | 53.4211 ± 13.2866 ‖ 43.9195 ± 1.80310 | 34.4165 ± 0.126008 ‖ 33.5252 ± 0.211722 | 47.7311 ± 0.249943 ‖ 47.5019 ± 0.645263 | 63.8263 ± 0.322921 ‖ 63.7838 ± 0.491582 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E dyn_model | E rollout | E synth_rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| b512 | -5.89690 ± 5.79375 ‖ -9.09421 ± 9.02804 | -1.82451 ± 1.53897 ‖ -4.16665 ± 0.717732 | -6.13012 ± 33.5624 ‖ -1.58595 ± 3.74722 | 30.4200 ± 1.17794 ‖ 55.2634 ± 1.00261 | 18.4239 ± 1.14713 ‖ 20.9302 ± 1.54046 | 17.5425 ± 0.477155 ‖ 17.1869 ± 0.337791 | 15.2734 ± 0.895956 ‖ 14.7743 ± 2.10062 | 18.1787 ± 0.298218 ‖ 20.0684 ± 0.383854 | 0.930915 ± 4.36167 ‖ -2.10611 ± 7.01714 |
| b1024 | -5.21680 ± 7.15628 ‖ -4.80839 ± 9.84787 | -2.66369 ± 2.12303 ‖ -2.65896 ± 0.813908 | 11.3798 ± 36.6158 ‖ -1.50562 ± 5.49794 | 75.2854 ± 2.35207 ‖ 164.696 ± 2.58220 | 32.1710 ± 0.718247 ‖ 40.8103 ± 1.68121 | 31.7308 ± 0.519676 ‖ 34.0638 ± 0.544606 | 27.6938 ± 1.87007 ‖ 30.1005 ± 1.33960 | 32.9807 ± 0.256689 ‖ 41.6287 ± 0.723475 | 5.58306 ± 5.47091 ‖ 6.13674 ± 7.98088 |

Duration:

| config | D dyn_model | D rollout | D synth_rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|---|---|
| b512 | -4.65789 ± 6.38177 ‖ -8.11389 ± 8.81195 | -0.208844 ± 1.60098 ‖ -0.281397 ± 0.145475 | 0.541154 ± 0.929473 ‖ -0.448590 ± 1.01135 | 2.27972 ± 0.624316 ‖ 3.29026 ± 0.800581 | -2.95977 ± 4.87683 ‖ -5.69325 ± 7.02844 | 13.0374 ± 0.961785 ‖ 33.8875 ± 0.564639 | 2.64397 ± 1.36477 ‖ 4.28799 ± 1.80804 | 1.87615 ± 0.212889 ‖ 1.05405 ± 0.272986 | -0.0916308 ± 0.446956 ‖ -1.03104 ± 1.45317 |
| b1024 | -4.17541 ± 7.83131 ‖ -5.42156 ± 10.0294 | -0.486621 ± 1.90910 ‖ 0.173072 ± 0.776087 | 1.00935 ± 1.96797 ‖ -1.96470 ± 2.05926 | 3.26943 ± 0.251362 ‖ 8.19518 ± 1.29999 | -2.38708 ± 5.93857 ‖ -2.69273 ± 7.92841 | 36.3956 ± 1.65287 ‖ 102.640 ± 0.703009 | 2.84897 ± 0.712820 ‖ 7.81350 ± 2.19986 | 2.50564 ± 0.398574 ‖ 2.63998 ± 0.996393 | -0.637286 ± 1.31428 ‖ -0.397591 ± 1.00791 |

Mean power:

| config | P dyn_model | P rollout | P synth_rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|---|---|
| b512 | -1.26807 ± 0.630480 ‖ -1.09087 ± 0.799605 | -1.61847 ± 0.168917 ‖ -3.89623 ± 0.704999 | -6.51807 ± 33.9810 ‖ -1.14667 ± 3.46204 | 15.5468 ± 0.490340 ‖ 16.2473 ± 0.603156 | 4.04128 ± 0.840981 ‖ 3.82190 ± 0.624843 | 3.51328 ± 0.570292 ‖ 3.21855 ± 0.501432 |
| b1024 | -1.04459 ± 0.652526 ‖ 0.676170 ± 0.496153 | -2.18850 ± 0.903779 ‖ -2.82060 ± 1.29461 | 10.0586 ± 35.0894 ‖ 0.438807 ± 4.46665 | 28.7709 ± 0.186493 ‖ 30.9102 ± 0.980480 | 8.21447 ± 1.02013 ‖ 9.12507 ± 0.937732 | 7.46222 ± 0.655607 ‖ 8.46188 ± 0.598949 |

Achieved throughput:

| config | dyn_model GFLOP/s | rollout GFLOP/s | synth_rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|---|---|
| b512 | 1.01661 ± 2.45278 ‖ 5.03669 ± 3.29131 | 0.230275 ± 1.63577 ‖ 0.282361 ± 0.146226 | -0.531491 ± 0.912828 ‖ 0.352952 ± 0.781133 | 95.5480 ± 1.19609 ‖ 93.6385 ± 1.51203 | 82.2651 ± 8.31640 ‖ 75.7448 ± 8.80850 |
| b1024 | 0.449920 ± 3.14682 ‖ 8.52800 ± 2.59176 | 0.519354 ± 1.97781 ‖ -0.168016 ± 0.767578 | -0.969804 ± 1.89031 ‖ 0.614702 ± 0.455536 | 287.338 ± 0.944861 ‖ 269.745 ± 4.40241 | 240.360 ± 19.1369 ‖ 209.160 ± 19.3759 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|---|
| b512 | -2.22786 ± 1.75752 ‖ -5.77705 ± 2.27391 | -1.82451 ± 1.53897 ‖ -4.16665 ± 0.717732 | -6.13012 ± 33.5624 ‖ -1.49757 ± 3.24648 | -40.7881 ± 0.573566 ‖ -39.5349 ± 0.770230 | -41.2288 ± 0.238578 ‖ -41.4066 ± 0.168896 | -40.9107 ± 0.149109 ‖ -39.9658 ± 0.191927 | -42.8429 ± 2.05201 ‖ -40.8144 ± 2.76285 | 15.2734 ± 0.895956 ‖ 14.7743 ± 2.10062 |
| b1024 | -1.42665 ± 2.44829 ‖ -7.20001 ± 1.83973 | -2.66369 ± 2.12303 ‖ -2.65896 ± 0.813908 | 11.3798 ± 36.6158 ‖ -0.177050 ± 4.34880 | -66.9572 ± 0.179562 ‖ -64.7974 ± 0.420302 | -67.0673 ± 0.129919 ‖ -66.4840 ± 0.136151 | -66.7548 ± 0.0641723 ‖ -64.5928 ± 0.180869 | -68.1412 ± 1.43948 ‖ -64.6125 ± 1.80063 | 27.6938 ± 1.87007 ‖ 30.1005 ± 1.33960 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|---|---|
| b512 | 0.961904 ± 0.0443293 ‖ 0.966495 ± 0.113241 | 1.00000 ‖ 1.00000 | 1.00000 ‖ 0.999073 ± 0.0166242 | 2.00000 ‖ 2.00000 | 2.00000 ‖ 2.00000 | 2.00000 ‖ 2.00000 | 1.76547 ± 0.0143579 ‖ 1.65259 ± 0.0481195 | 1.00000 ‖ 1.00000 |
| b1024 | 0.960604 ± 0.0502410 ‖ 1.02597 ± 0.108117 | 1.00000 ‖ 1.00000 | 1.00000 ‖ 0.986435 ± 0.0242273 | 4.00000 ‖ 4.00000 | 4.00000 ‖ 4.00000 | 4.00000 ‖ 4.00000 | 3.31329 ± 0.0254208 ‖ 2.99643 ± 0.0905535 | 1.00000 ‖ 1.00000 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | dyn_model | rollout | synth_rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|---|---|
| b512 | -4.82234 ± 1.36262 ‖ -5.41300 ± 1.99543 | -0.0311214 ± 0.0336010 ‖ -0.0315233 ± 0.110538 | -0.00465948 ± 0.0121705 ‖ 0.00268380 ± 0.0248186 | 0.237025 ± 0.0342805 ‖ 0.532078 ± 0.0946880 | 2.06244 ± 0.607185 ‖ 2.32165 ± 0.731230 | 2.38284 ± 0.634182 ‖ 2.40865 ± 0.961916 | 0.175811 ± 0.0557539 ‖ 0.179470 ± 0.0941498 | 4.85812 ± 1.31892 ‖ 5.44184 ± 1.86322 |
| b1024 | -7.28850 ± 1.66244 ‖ -7.80817 ± 2.28835 | -0.0905360 ± 0.0380461 ‖ -0.138922 ± 0.119632 | 3.32317e-04 ± 0.0135324 ‖ -0.0220983 ± 0.0273310 | 0.535339 ± 0.0457703 ‖ 1.34944 ± 0.109907 | 2.99725 ± 0.733573 ‖ 3.20030 ± 0.927529 | 3.58716 ± 0.788078 ‖ 3.18746 ± 1.04924 | 0.258956 ± 0.0569785 ‖ 0.231984 ± 0.0866407 | 7.37871 ± 1.62159 ‖ 7.96919 ± 2.15348 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| base | 1.22056 | 1.76769 | 3.81837 | dyn_model | 0.995821 | 0.973150 | 0.976476 |
| b512 | 1.18312 | 1.72588 | 3.22770 | dyn_model | 0.993772 | 0.944864 | 1.01170 |
| b1024 | 1.22421 | 1.76828 | 3.29870 | dyn_model | 1.00426 | 0.951041 | 1.08374 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | dyn_model | synth_rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| base | 1.76769 (paired sd 0.0422332) | 1.28846 (paired sd 0.166389) | 9.75678 (paired sd 2.54254) | 1.35417 (paired sd 0.0179559) | 0.998842 (paired sd 0.0177614) | 1.01530 (paired sd 0.00522707) | 1.00997 (paired sd 0.0182342) | 1.01786 (paired sd 0.00643497) | 1.22056 (paired sd 0.114543) |
| b512 | 1.72588 (paired sd 0.0170273) | 1.24429 (paired sd 0.119774) | 10.8753 (paired sd 0.922621) | 1.61220 (paired sd 0.00772134) | 1.02001 (paired sd 0.0270172) | 1.01224 (paired sd 0.00174013) | 1.00552 (paired sd 0.0119195) | 1.03414 (paired sd 0.0100466) | 1.18312 (paired sd 0.0787971) |
| b1024 | 1.76828 (paired sd 0.0151178) | 1.28958 (paired sd 0.107965) | 8.96290 (paired sd 2.37429) | 2.04504 (paired sd 0.0168012) | 1.06411 (paired sd 0.00943741) | 1.03329 (paired sd 0.00379221) | 1.02904 (paired sd 0.00790659) | 1.08404 (paired sd 0.00562562) | 1.22421 (paired sd 0.0668877) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | dyn_model | rollout | synth_rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|---|---|
| base | 3.81837 ± 2.48009 (5/5 same sign) | 0.532104 ± 0.122998 (5/5 same sign) | 0.256200 ± 0.0396612 (5/5 same sign) | 0.0921353 ± 0.0731507 (4/5 same sign) | -2.11498 ± 1.04631 (5/5 same sign) | -2.37587 ± 1.12935 (5/5 same sign) | -0.207957 ± 0.0903167 (5/5 same sign) |
| b512 | 3.22770 ± 2.19993 (4/5 same sign) | 0.531702 ± 0.114854 (5/5 same sign) | 0.263543 ± 0.0387087 (5/5 same sign) | 0.387188 ± 0.102096 (5/5 same sign) | -1.85577 ± 0.970884 (5/5 same sign) | -2.35007 ± 0.938040 (5/5 same sign) | -0.204298 ± 0.0805348 (5/5 same sign) |
| b1024 | 3.29870 ± 1.94733 (5/5 same sign) | 0.483718 ± 0.0865885 (5/5 same sign) | 0.233769 ± 0.0150625 (5/5 same sign) | 0.906237 ± 0.116438 (5/5 same sign) | -1.91192 ± 0.801447 (5/5 same sign) | -2.77558 ± 0.880161 (5/5 same sign) | -0.234928 ± 0.0688566 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| base | mean Δ 0.532104; 2/32 patterns; p = 0.0625000 | mean Δ 45743.3; 2/32 patterns; p = 0.0625000 |
| b512 | mean Δ 0.531702; 2/32 patterns; p = 0.0625000 | mean Δ 38291.5; 2/32 patterns; p = 0.0625000 |
| b1024 | mean Δ 0.483718; 2/32 patterns; p = 0.0625000 | mean Δ 49027.7; 2/32 patterns; p = 0.0625000 |


## 4.4 Sweep: maximum rollout length (Ant-v5 only) — rollout_max_length ∈ {1,15,25}

Configurations: `rollout1` (Ant only: rollout_max_length 1), `rollout15` (Ant only: rollout_max_length 15), `base` (SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model (200×4)×7; HC: fixed rollout length 1; Ant: rollout length 1→25 (mbpo_ant.json)). Baseline for the response tables: `base` of the same environment. Environment comparison omitted: the configurations exist only on Ant-v5 (except `base`).

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 176446 ± 3727.50 | n/a ‖ 4251.90 ± 38.9765 | n/a ‖ 121.939 ± 18.3584 | n/a ‖ 2018.36 ± 9.98716 | n/a ‖ 25237.8 ± 447.756 | n/a ‖ 30137.5 ± 174.454 | n/a ‖ 2535.41 ± 8.49494 | n/a ‖ 59929.1 ± 587.330 | n/a ‖ 240749 ± 3280.97 |
| rollout15 | n/a ‖ 181382 ± 12746.4 | n/a ‖ 4213.11 ± 27.4735 | n/a ‖ 449.641 ± 40.6506 | n/a ‖ 2241.82 ± 10.5683 | n/a ‖ 24986.9 ± 187.138 | n/a ‖ 30057.0 ± 68.7366 | n/a ‖ 2531.89 ± 20.3251 | n/a ‖ 59817.6 ± 200.807 | n/a ‖ 245862 ± 12655.9 |
| base | 146193 ± 7439.23 ‖ 188363 ± 20295.5 | 2428.26 ± 47.7526 ‖ 4292.41 ± 29.8641 | 75.3990 ± 16.3698 ‖ 735.651 ± 37.3179 | 1677.11 ± 15.2915 ‖ 2271.09 ± 12.1950 | 24588.5 ± 267.376 ‖ 24560.0 ± 218.652 | 29882.6 ± 104.818 ‖ 30340.0 ± 62.3096 | 2550.10 ± 26.8657 ‖ 2575.51 ± 25.8022 | 58698.3 ± 211.888 ‖ 59746.5 ± 208.801 | 207395 ± 7451.41 ‖ 253138 ± 20140.0 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| rollout1 | n/a ‖ 0.0668748 ± 9.11380e-04 | n/a ‖ 74.7526 ± 1.99226 | n/a ‖ 5599.98 ± 114.175 | n/a ‖ 5811.10 ± 43.0755 |
| rollout15 | n/a ‖ 0.0682951 ± 0.00351553 | n/a ‖ 74.8401 ± 2.53345 | n/a ‖ 5582.43 ± 76.5439 | n/a ‖ 5881.82 ± 77.4915 |
| base | 0.0576096 ± 0.00206984 ‖ 0.0703161 ± 0.00559445 | 18.8339 ± 2.39855 ‖ 74.0726 ± 3.03600 | 5820.88 ± 39.1697 ‖ 5763.91 ± 33.4087 | 6042.89 ± 33.9578 ‖ 6037.16 ± 35.0893 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | dyn_model (measured) | rollout (measured) | synth_rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 73.2846 ± 0.572651 | n/a ‖ 1.76652 ± 0.0387216 | n/a ‖ 0.0505958 ± 0.00712729 | n/a ‖ 0.838463 ± 0.00945973 | n/a ‖ 10.4858 ± 0.297305 | n/a ‖ 12.5207 ± 0.231758 | n/a ‖ 1.05331 ± 0.0161065 | n/a ‖ 24.8983 ± 0.539832 |
| rollout15 | n/a ‖ 73.7175 ± 1.37175 | n/a ‖ 1.71709 ± 0.0855858 | n/a ‖ 0.182786 ± 0.0121170 | n/a ‖ 0.913674 ± 0.0453253 | n/a ‖ 10.1866 ± 0.576015 | n/a ‖ 12.2505 ± 0.616982 | n/a ‖ 1.03196 ± 0.0531852 | n/a ‖ 24.3827 ± 1.28945 |
| base | 70.4601 ± 1.04907 ‖ 74.2785 ± 2.09320 | 1.17147 ± 0.0245455 ‖ 1.70357 ± 0.124251 | 0.0365412 ± 0.00862227 ‖ 0.292741 ± 0.0357971 | 0.809395 ± 0.0267401 ‖ 0.901530 ± 0.0684953 | 11.8698 ± 0.495888 ‖ 9.75479 ± 0.832491 | 14.4221 ± 0.472909 ‖ 12.0462 ± 0.954617 | 1.23060 ± 0.0365860 ‖ 1.02264 ± 0.0822693 | 28.3319 ± 1.02110 ‖ 23.7252 ± 1.93477 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| rollout1 | n/a ‖ 3.36826 ± 0.0462410 | n/a ‖ 42.1103 ± 0.352796 | n/a ‖ 50.2904 ± 0.265517 | n/a ‖ 4.23102 ± 0.0445582 |
| rollout15 | n/a ‖ 3.74778 ± 0.0178495 | n/a ‖ 41.7715 ± 0.197237 | n/a ‖ 50.2480 ± 0.152517 | n/a ‖ 4.23273 ± 0.0384821 |
| base | 2.85721 ± 0.0304969 ‖ 3.80128 ± 0.0300127 | 41.8889 ± 0.334398 ‖ 41.1064 ± 0.243247 | 50.9094 ± 0.258129 ‖ 50.7815 ± 0.178929 | 4.34452 ± 0.0541846 ‖ 4.31077 ± 0.0449133 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | dyn_model (measured) [J/FLOP] | rollout (measured) [J/FLOP] | synth_rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 4.12010e-10 ± 9.43497e-12 | n/a ‖ 1.81321e-08 ± 1.66214e-10 | n/a ‖ 2.49589e-11 ± 3.75765e-12 | n/a ‖ 4.80269e-11 ± 8.52067e-13 | n/a ‖ 7.29291e-11 ± 4.22157e-13 | n/a ‖ 6.38401e-11 ± 6.25660e-13 | n/a ‖ 1.75458e-10 ± 1.64199e-12 | n/a ‖ 5.42980 ± 0.0181926 |
| rollout15 | n/a ‖ 4.15198e-10 ± 6.68425e-12 | n/a ‖ 1.79667e-08 ± 1.17160e-10 | n/a ‖ 1.48966e-11 ± 1.38439e-12 | n/a ‖ 4.75494e-11 ± 3.56119e-13 | n/a ‖ 7.27342e-11 ± 1.66334e-13 | n/a ‖ 6.37213e-11 ± 2.13912e-13 | n/a ‖ 1.74865e-10 ± 5.75210e-12 | n/a ‖ 5.42224 ± 0.0435278 |
| base | 5.80582e-10 ± 1.48182e-11 ‖ 4.37090e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59556e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83073e-10 ± 5.44247e-12 ‖ 1.78766e-10 ± 8.47149e-12 | 5.92935 ± 0.0624667 ‖ 5.51568 ± 0.0552575 |

MBPO FLOPs differ between seeds. **Mean of the per-seed ratios** (E_i/F_i averaged over seeds) ± sd, for comparison with the ratio-of-means table above:

| config | dyn_model [J/FLOP] | rollout [J/FLOP] | synth_rollout [J/FLOP] | critic [J/FLOP] | actor [J/FLOP] | GU [J/FLOP] | TOTAL [J/FLOP] |
|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 4.12176e-10 ± 9.43497e-12 | n/a ‖ 1.81321e-08 ± 1.66214e-10 | n/a ‖ 2.49589e-11 ± 3.75765e-12 | n/a ‖ 4.80269e-11 ± 8.52067e-13 | n/a ‖ 7.29291e-11 ± 4.22157e-13 | n/a ‖ 6.38401e-11 ± 6.25660e-13 | n/a ‖ 1.75458e-10 ± 1.64199e-12 |
| rollout15 | n/a ‖ 4.15068e-10 ± 6.68425e-12 | n/a ‖ 1.79667e-08 ± 1.17160e-10 | n/a ‖ 1.49001e-11 ± 1.38439e-12 | n/a ‖ 4.75494e-11 ± 3.56119e-13 | n/a ‖ 7.27342e-11 ± 1.66334e-13 | n/a ‖ 6.37213e-11 ± 2.13912e-13 | n/a ‖ 1.74782e-10 ± 5.75210e-12 |
| base | 5.80351e-10 ± 1.48182e-11 ‖ 4.37061e-10 ± 3.90010e-12 | 1.12600e-08 ± 2.21431e-10 ‖ 1.83048e-08 ± 1.27354e-10 | 1.90319e-11 ± 4.13199e-12 ‖ 1.59503e-11 ± 5.98598e-13 | 4.99401e-11 ± 5.43051e-13 ‖ 4.67369e-11 ± 4.16089e-13 | 7.77155e-11 ± 2.72600e-13 ‖ 7.34190e-11 ± 1.50782e-13 | 6.69406e-11 ± 2.41641e-13 ‖ 6.36456e-11 ± 2.22427e-13 | 1.83047e-10 ± 5.44247e-12 ‖ 1.78551e-10 ± 8.47149e-12 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | dyn_model (measured) duration [s] | rollout (measured) duration [s] | synth_rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 1489.22 ± 38.9251 | n/a ‖ 37.0143 ± 0.370936 | n/a ‖ 0.934079 ± 0.00164873 | n/a ‖ 421.719 ± 6.82450 | n/a ‖ 13.7055 ± 0.0668302 | n/a ‖ 171.396 ± 4.22700 | n/a ‖ 204.655 ± 2.48228 | n/a ‖ 17.2169 ± 0.132738 | n/a ‖ 1948.89 ± 32.6063 |
| rollout15 | n/a ‖ 1535.37 ± 118.896 | n/a ‖ 36.6379 ± 0.291541 | n/a ‖ 3.44790 ± 0.0185424 | n/a ‖ 417.232 ± 1.28763 | n/a ‖ 15.1784 ± 0.0864635 | n/a ‖ 169.174 ± 1.10273 | n/a ‖ 203.503 ± 0.724352 | n/a ‖ 17.1422 ± 0.129375 | n/a ‖ 1992.68 ± 119.243 |
| base | 1250.07 ± 72.8336 ‖ 1605.17 ± 180.136 | 20.3785 ± 0.415238 ‖ 36.6806 ± 0.0873145 | 0.633054 ± 0.00774843 ‖ 5.07582 ± 0.0749797 | 407.584 ± 2.70475 ‖ 411.097 ± 2.42950 | 11.2917 ± 0.0973144 ‖ 15.1580 ± 0.0311855 | 165.557 ± 2.32430 ‖ 163.927 ± 1.97951 | 201.197 ± 0.836174 ‖ 202.501 ± 0.702973 | 17.1693 ± 0.155059 ‖ 17.1899 ± 0.159045 | 1678.67 ± 72.2102 ‖ 2058.02 ± 178.140 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P dyn_model [W] | P rollout [W] | P synth_rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 118.498 ± 1.11095 | n/a ‖ 114.873 ± 0.456621 | n/a ‖ 130.541 ± 19.5951 | n/a ‖ 142.119 ± 0.964173 | n/a ‖ 123.539 ± 0.803852 | n/a ‖ 118.449 ± 0.649942 | n/a ‖ 0.808306 ± 0.00339115 |
| rollout15 | n/a ‖ 118.187 ± 0.919555 | n/a ‖ 114.996 ± 0.837033 | n/a ‖ 130.448 ± 12.2584 | n/a ‖ 143.368 ± 0.375573 | n/a ‖ 123.431 ± 1.05984 | n/a ‖ 118.464 ± 0.727514 | n/a ‖ 0.802101 ± 0.00428716 |
| base | 116.988 ± 0.953539 ‖ 117.397 ± 0.655463 | 119.160 ± 0.550654 ‖ 117.022 ± 0.953688 | 119.137 ± 25.8677 ‖ 144.882 ± 5.45432 | 144.018 ± 0.467991 ‖ 145.336 ± 0.473385 | 123.577 ± 0.913456 ‖ 123.061 ± 0.930322 | 117.983 ± 0.615299 ‖ 118.404 ± 0.501427 | 0.827400 ± 0.00273275 ‖ 0.805180 ± 0.00538864 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | dyn_model [GFLOP/s] | rollout [GFLOP/s] | synth_rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 287.660 ± 9.14562 | n/a ‖ 6.33580 ± 0.0635359 | n/a ‖ 5230.42 ± 9.24802 | n/a ‖ 2226.45 ± 36.1871 | n/a ‖ 704.171 ± 10.8019 | n/a ‖ 3067.47 ± 75.8855 | n/a ‖ 2019.46 ± 24.5741 | n/a ‖ 96.5029 ± 0.0657174 | n/a ‖ 96.4058–96.5773 |
| rollout15 | n/a ‖ 284.824 ± 6.42789 | n/a ‖ 6.40068 ± 0.0505888 | n/a ‖ 8754.29 ± 38.0860 | n/a ‖ 2249.94 ± 6.94110 | n/a ‖ 706.954 ± 28.7050 | n/a ‖ 3106.33 ± 20.1371 | n/a ‖ 2030.68 ± 7.20663 | n/a ‖ 97.0677 ± 0.0295737 | n/a ‖ 97.0202–97.0911 |
| base | 201.719 ± 6.74360 ‖ 268.627 ± 3.33295 | 10.5859 ± 0.210149 ‖ 6.39295 ± 0.0152330 | 6258.88 ± 77.5462 ‖ 9083.34 ± 23.6982 | 2151.47 ± 14.2695 ‖ 2283.56 ± 13.4483 | 675.699 ± 24.6566 ‖ 690.649 ± 37.7185 | 2974.43 ± 41.5340 ‖ 3206.03 ± 38.3417 | 1911.16 ± 7.94527 ‖ 2040.72 ± 7.09689 | 96.9650 ± 0.0406469 ‖ 97.0025 ± 0.0629928 | 96.8947–96.9927 ‖ 96.9168–97.0815 |

**Hardware composition of the measured `dynamics_model_update` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of dyn_model energy [%] | GPU share of dyn_model energy [%] | RAM share of dyn_model energy [%] | dyn_model energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| rollout1 | n/a ‖ 22.3641 ± 0.446955 | n/a ‖ 60.7574 ± 0.289580 | n/a ‖ 16.8785 ± 0.158132 | n/a ‖ 0/500 |
| rollout15 | n/a ‖ 22.1455 ± 0.203922 | n/a ‖ 60.9320 ± 0.121030 | n/a ‖ 16.9225 ± 0.132354 | n/a ‖ 0/500 |
| base | 21.4036 ± 0.316290 ‖ 21.4284 ± 0.217233 | 61.5003 ± 0.203244 ‖ 61.5356 ± 0.155617 | 17.0961 ± 0.139314 ‖ 17.0360 ± 0.0949920 | 0/500 ‖ 0/500 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| rollout1 | n/a ‖ 27.7662 ± 0.255231 | n/a ‖ 54.8387 ± 0.282514 | n/a ‖ 17.3951 ± 0.0691532 | n/a ‖ 500/500 |
| rollout15 | n/a ‖ 27.5943 ± 0.295854 | n/a ‖ 55.0289 ± 0.390891 | n/a ‖ 17.3768 ± 0.126092 | n/a ‖ 500/500 |
| base | 25.7910 ± 0.402018 ‖ 27.6475 ± 0.322486 | 57.4494 ± 0.449754 ‖ 55.2805 ± 0.439211 | 16.7596 ± 0.0798437 ‖ 17.0720 ± 0.139120 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `synthetic_rollout_generation` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of synth_rollout energy [%] | GPU share of synth_rollout energy [%] | RAM share of synth_rollout energy [%] | synth_rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| rollout1 | n/a ‖ 24.5251 ± 3.64071 | n/a ‖ 60.6124 ± 5.89531 | n/a ‖ 14.8626 ± 2.25485 | n/a ‖ 500/500 |
| rollout15 | n/a ‖ 26.6484 ± 2.55390 | n/a ‖ 58.1586 ± 4.02643 | n/a ‖ 15.1930 ± 1.47281 | n/a ‖ 500/500 |
| base | 26.1498 ± 6.92079 ‖ 24.0129 ± 0.939763 | 57.2526 ± 11.2969 ‖ 62.3211 ± 1.43681 | 16.5975 ± 4.37648 ‖ 13.6660 ± 0.498430 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| rollout1 | n/a ‖ 18.6557 ± 0.118429 | n/a ‖ 67.2726 ± 0.174529 | n/a ‖ 14.0716 ± 0.0952568 | n/a ‖ 0/500 |
| rollout15 | n/a ‖ 18.4135 ± 0.0726836 | n/a ‖ 67.6377 ± 0.104659 | n/a ‖ 13.9489 ± 0.0366048 | n/a ‖ 0/500 |
| base | 18.0568 ± 0.118760 ‖ 18.2631 ± 0.0602319 | 68.0574 ± 0.145407 ‖ 67.9766 ± 0.0664136 | 13.8857 ± 0.0451822 ‖ 13.7603 ± 0.0448462 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | dyn_model [%] | rollout [%] | synth_rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 53.5039 ± 1.05326 | n/a ‖ 55.1835 ± 0.464972 | n/a ‖ 49.3873 ± 6.98200 | n/a ‖ 44.6043 ± 0.291474 | n/a ‖ 51.3178 ± 0.866097 | n/a ‖ 63.3926 ± 0.758132 |
| rollout15 | n/a ‖ 53.8897 ± 0.442790 | n/a ‖ 55.3840 ± 0.283736 | n/a ‖ 49.1891 ± 4.87734 | n/a ‖ 44.4229 ± 0.183788 | n/a ‖ 51.6007 ± 0.446576 | n/a ‖ 63.6880 ± 0.252361 |
| base | 56.3406 ± 0.609099 ‖ 55.8464 ± 0.519715 | 55.3099 ± 0.0464751 ‖ 56.0243 ± 0.227975 | 57.9021 ± 15.0568 ‖ 45.3015 ± 1.71294 | 45.7635 ± 0.167734 ‖ 45.1088 ± 0.176904 | 53.3357 ± 0.515845 ‖ 53.2775 ± 0.609384 | 65.9074 ± 0.314375 ‖ 65.5593 ± 0.279103 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E dyn_model | E rollout | E synth_rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ -5.51549 ± 9.54203 | n/a ‖ -0.941878 ± 0.915497 | n/a ‖ -83.3148 ± 3.12067 | n/a ‖ -11.1278 ± 0.265712 | n/a ‖ 2.75532 ± 1.13787 | n/a ‖ -0.666413 ± 0.712801 | n/a ‖ -1.54949 ± 0.983279 | n/a ‖ 0.305066 ± 0.848628 | n/a ‖ -4.44004 ± 7.21579 |
| rollout15 | n/a ‖ -3.35968 ± 4.61454 | n/a ‖ -1.84547 ± 0.657094 | n/a ‖ -38.6389 ± 7.55557 | n/a ‖ -1.28623 ± 0.763185 | n/a ‖ 1.74017 ± 0.469157 | n/a ‖ -0.932490 ± 0.277815 | n/a ‖ -1.69182 ± 0.432272 | n/a ‖ 0.118967 ± 0.0641124 | n/a ‖ -2.68128 ± 3.47431 |

Duration:

| config | D dyn_model | D rollout | D synth_rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ -6.39128 ± 9.44320 | n/a ‖ 0.908982 ± 0.886586 | n/a ‖ -81.5946 ± 0.248885 | n/a ‖ 2.58028 ± 1.22960 | n/a ‖ -4.78750 ± 7.53864 | n/a ‖ -9.58161 ± 0.567520 | n/a ‖ 4.54582 ± 1.54623 | n/a ‖ 1.06300 ± 1.06892 | n/a ‖ 0.164866 ± 1.29551 |
| rollout15 | n/a ‖ -4.00545 ± 4.56410 | n/a ‖ -0.115551 ± 0.858752 | n/a ‖ -32.0615 ± 0.962078 | n/a ‖ 1.49416 ± 0.459056 | n/a ‖ -2.96981 ± 3.55365 | n/a ‖ 0.135317 ± 0.641853 | n/a ‖ 3.20748 ± 0.768795 | n/a ‖ 0.495530 ± 0.487789 | n/a ‖ -0.274523 ± 0.635287 |

Mean power:

| config | P dyn_model | P rollout | P synth_rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 0.937586 ± 0.696055 | n/a ‖ -1.83323 ± 0.555473 | n/a ‖ -9.46378 ± 16.2722 | n/a ‖ -2.21461 ± 0.359936 | n/a ‖ 0.390259 ± 0.615493 | n/a ‖ 0.0376864 ± 0.436535 |
| rollout15 | n/a ‖ 0.672533 ± 0.307568 | n/a ‖ -1.72367 ± 1.32243 | n/a ‖ -9.72407 ± 10.6722 | n/a ‖ -1.35350 ± 0.402148 | n/a ‖ 0.300461 ± 0.264472 | n/a ‖ 0.0501203 ± 0.234444 |

Achieved throughput:

| config | dyn_model GFLOP/s | rollout GFLOP/s | synth_rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|---|---|
| rollout1 | n/a ‖ 7.07244 ± 2.52896 | n/a ‖ -0.894651 ± 0.874076 | n/a ‖ -42.4170 ± 0.231913 | n/a ‖ -2.50406 ± 1.18001 | n/a ‖ 2.15349 ± 4.63750 |
| rollout15 | n/a ‖ 6.03342 ± 2.25148 | n/a ‖ 0.121589 ± 0.858477 | n/a ‖ -3.62255 ± 0.366703 | n/a ‖ -1.47055 ± 0.444844 | n/a ‖ 2.44133 ± 2.63743 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ -5.69601 ± 1.84851 | n/a ‖ -0.941878 ± 0.915497 | n/a ‖ 57.2186 ± 28.2270 | n/a ‖ 2.75532 ± 1.13787 | n/a ‖ -0.666413 ± 0.712801 | n/a ‖ 0.305066 ± 0.848628 | n/a ‖ -1.58117 ± 4.01369 | n/a ‖ -1.54949 ± 0.983279 |
| rollout15 | n/a ‖ -5.02553 ± 1.78787 | n/a ‖ -1.84547 ± 0.657094 | n/a ‖ -6.35587 ± 10.7989 | n/a ‖ 1.74017 ± 0.469157 | n/a ‖ -0.932490 ± 0.277815 | n/a ‖ 0.118967 ± 0.0641124 | n/a ‖ -2.04167 ± 2.35062 | n/a ‖ -1.69182 ± 0.432272 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | dyn_model | rollout | synth_rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ 1.00341 ± 0.116193 | n/a ‖ 1.00000 | n/a ‖ 0.105986 ± 0.00170242 | n/a ‖ 1.00000 | n/a ‖ 1.00000 | n/a ‖ 1.00000 | n/a ‖ 0.969851 ± 0.0345898 | n/a ‖ 1.00000 |
| rollout15 | n/a ‖ 1.01761 ± 0.0466298 | n/a ‖ 1.00000 | n/a ‖ 0.654788 ± 0.0106891 | n/a ‖ 1.00000 | n/a ‖ 1.00000 | n/a ‖ 1.00000 | n/a ‖ 0.993262 ± 0.0132660 | n/a ‖ 1.00000 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | dyn_model | rollout | synth_rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|---|---|
| rollout1 | n/a ‖ -0.993887 ± 1.97085 | n/a ‖ 0.0629477 ± 0.123512 | n/a ‖ -0.242145 ± 0.0422804 | n/a ‖ -0.0630677 ± 0.0696637 | n/a ‖ 0.731055 ± 0.726595 | n/a ‖ 0.474432 ± 0.935228 | n/a ‖ 0.0306652 ± 0.0821142 | n/a ‖ 1.17308 ± 1.81175 |
| rollout15 | n/a ‖ -0.561009 ± 0.912776 | n/a ‖ 0.0135180 ± 0.0497227 | n/a ‖ -0.109955 ± 0.0406127 | n/a ‖ 0.0121434 ± 0.0280442 | n/a ‖ 0.431780 ± 0.358201 | n/a ‖ 0.204208 ± 0.414468 | n/a ‖ 0.00931497 ± 0.0356058 | n/a ‖ 0.657446 ± 0.831146 |


## 5. Cross-configuration quantities

**5.1a Fixed-versus-marginal fit E = a + b·F** — ordinary least squares over per-run points (one point per run: x = the run's FLOPs of the segment, y = its gross energy of the same segment; n = 5 × number of configurations of the sweep), exactly as in Section 4.1 (`ols_section`). a = intercept [J], b = slope [pJ/FLOP], residual SE = sqrt(SSR/(n−2)) [J]. GU = `gradient_updates` (FLOPs = critic + actor); TOTAL = `TOTAL_MEASURED_TRAINING`. A sweep with fewer than 3 distinct per-configuration FLOP values is skipped:

| sweep | env | configurations | segment | n | distinct FLOP values | a [J] | b [pJ/FLOP] | R² | residual SE [J] |
|---|---|---|---|---|---|---|---|---|---|
| UTD | HalfCheetah-v5 | base, utd2, utd4 | GU | 15 | 3 | 946.974 | 65.8210 | 0.999920 | 690.014 |
| UTD | HalfCheetah-v5 | base, utd2, utd4 | TOTAL | 15 | 3 | 138255 | 62.9571 | 0.990838 | 7090.56 |
| UTD | Ant-v5 | base, utd2, utd4 | GU | 15 | 3 | 1168.54 | 62.5160 | 0.999843 | 984.068 |
| UTD | Ant-v5 | base, utd2, utd4 | TOTAL | 15 | 3 | 168931 | 60.6927 | 0.962119 | 15158.7 |
| width | HalfCheetah-v5 | w256, w512, base | GU | 15 | 3 | 43233.6 | 17.5507 | 0.993553 | 536.752 |
| width | HalfCheetah-v5 | w256, w512, base | TOTAL | 15 | 3 | 180941 | 23.6080 | 0.640828 | 6762.61 |
| width | Ant-v5 | w256, w512, base | GU | 15 | 3 | 43222.9 | 17.5189 | 0.996331 | 425.425 |
| width | Ant-v5 | w256, w512, base | TOTAL | 15 | 3 | 215248 | 27.4680 | 0.341398 | 15519.3 |
| batch size | HalfCheetah-v5 | base, b512, b1024 | GU | 15 | 3 | 54354.2 | 7.01554 | 0.938186 | 2115.49 |
| batch size | HalfCheetah-v5 | base, b512, b1024 | TOTAL | 15 | 3 | 201394 | 4.49907 | 0.457865 | 5732.19 |
| batch size | Ant-v5 | base, b512, b1024 | GU | 15 | 3 | 53306.3 | 8.54978 | 0.970795 | 1865.00 |
| batch size | Ant-v5 | base, b512, b1024 | TOTAL | 15 | 3 | 238856 | 6.46159 | 0.159534 | 18740.2 |
| maximum rollout length (Ant-v5 only) | Ant-v5 | rollout1, rollout15, base | TOTAL | 15 | 3 | -276018 | 373.807 | 0.892551 | 4718.61 |

Skipped fits:
- maximum rollout length (Ant-v5 only) / HalfCheetah-v5: not all configurations of the sweep exist in this environment (['rollout1', 'rollout15'] missing) — no fit
- maximum rollout length (Ant-v5 only) / Ant-v5 / GU: only 1 distinct FLOP value(s) among the 3 configurations (rollout1=9.38738e+14, rollout15=9.38738e+14, base=9.38738e+14) — no fit

**5.1b Pairwise ΔEnergy/ΔFLOPs** — `compute_delta_pairs()` imported from `flop_dashboard.py` and called unmodified (inputs: `per_run_energy_per_flop.csv` loaded with `load_per_run`, rows of `mbpo` with `included_in_cross_seed_avg`, `PER_RUN_DIMENSIONS` so the seed is held fixed; and `cross_seed_energy_per_flop.csv` with `DIMENSIONS`). Rule: pairs differ only in the X dimension; matmul/mixed_total rows only; pairs with |ΔF| < 1% of the reference FLOPs are dropped; pooled value = ΣΔE/ΣΔF over the used pairs. Reference = smallest X value (`first`) or the next-smaller one (`previous`). Pair list: `contexts/Section_4.3_data/mbpo_delta_pairs.csv`.

| sweep | reference | segment | X pairs (x←ref) | HC ΣΔE/ΣΔF [pJ/FLOP] | Ant ΣΔE/ΣΔF [pJ/FLOP] | pairs used HC / Ant | pairs dropped HC / Ant |
|---|---|---|---|---|---|---|---|
| width | first | rollout | 1024x1024←256x256, 512x512←256x256 | 573.765 | 972.380 | 10 / 10 | 0 / 0 |
| width | first | critic | 1024x1024←256x256, 512x512←256x256 | 12.2650 | 11.7803 | 10 / 10 | 0 / 0 |
| width | first | actor | 1024x1024←256x256, 512x512←256x256 | 19.5646 | 19.8932 | 10 / 10 | 0 / 0 |
| width | first | dyn_model | 1024x1024←256x256, 512x512←256x256 | 798.629 | 153.674 | 10 / 9 | 0 / 1 |
| width | first | synth_rollout | 1024x1024←256x256, 512x512←256x256 | 14.4851 | 9.69446 | 10 / 10 | 0 / 0 |
| width | first | TOTAL | 1024x1024←256x256, 512x512←256x256 | 24.4198 | 15.1018 | 10 / 10 | 0 / 0 |
| width | previous | rollout | 1024x1024←512x512, 512x512←256x256 | 575.084 | 1016.06 | 10 / 10 | 0 / 0 |
| width | previous | critic | 1024x1024←512x512, 512x512←256x256 | 12.8555 | 12.1954 | 10 / 10 | 0 / 0 |
| width | previous | actor | 1024x1024←512x512, 512x512←256x256 | 20.1769 | 20.6078 | 10 / 10 | 0 / 0 |
| width | previous | dyn_model | 1024x1024←512x512, 512x512←256x256 | 890.431 | -250.559 | 9 / 10 | 1 / 0 |
| width | previous | synth_rollout | 1024x1024←512x512, 512x512←256x256 | 11.1427 | 8.25928 | 10 / 10 | 0 / 0 |
| width | previous | TOTAL | 1024x1024←512x512, 512x512←256x256 | 23.5820 | 21.3741 | 10 / 10 | 0 / 0 |
| batch size | first | rollout | — | dropped (ΔF≈0) | dropped (ΔF≈0) | 0 / 0 | 10 / 10 |
| batch size | first | critic | 1024←256, 512←256 | 6.31591 | 7.21313 | 10 / 10 | 0 / 0 |
| batch size | first | actor | 1024←256, 512←256 | 9.57276 | 9.40674 | 10 / 10 | 0 / 0 |
| batch size | first | dyn_model | 1024←256, 512←256 | 842.715 | 3611.63 | 8 / 10 | 2 / 0 |
| batch size | first | synth_rollout | — | dropped (ΔF≈0) | 18.9349 | 0 / 6 | 10 / 4 |
| batch size | first | TOTAL | 1024←256, 512←256 | 3.72163 | 2.35198 | 10 / 10 | 0 / 0 |
| batch size | previous | rollout | — | dropped (ΔF≈0) | dropped (ΔF≈0) | 0 / 0 | 10 / 10 |
| batch size | previous | critic | 1024←512, 512←256 | 5.35463 | 6.35656 | 10 / 10 | 0 / 0 |
| batch size | previous | actor | 1024←512, 512←256 | 8.21958 | 8.33625 | 10 / 10 | 0 / 0 |
| batch size | previous | dyn_model | 1024←512, 512←256 | 751.304 | -1169.15 | 7 / 9 | 3 / 1 |
| batch size | previous | synth_rollout | — | dropped (ΔF≈0) | -4.39644 | 0 / 7 | 10 / 3 |
| batch size | previous | TOTAL | 1024←512, 512←256 | 4.30172 | 5.15505 | 10 / 10 | 0 / 0 |
| UTD | first | rollout | — | dropped (ΔF≈0) | dropped (ΔF≈0) | 0 / 0 | 10 / 10 |
| UTD | first | critic | UTD 2←UTD 1, UTD 4←UTD 1 | 49.3944 | 46.7275 | 10 / 10 | 0 / 0 |
| UTD | first | actor | UTD 2←UTD 1, UTD 4←UTD 1 | 75.9895 | 71.4022 | 10 / 10 | 0 / 0 |
| UTD | first | dyn_model | UTD 2←UTD 1, UTD 4←UTD 1 | -278.249 | 56.9474 | 9 / 8 | 1 / 2 |
| UTD | first | synth_rollout | — | dropped (ΔF≈0) | 8.33574 | 0 / 6 | 10 / 4 |
| UTD | first | TOTAL | UTD 2←UTD 1, UTD 4←UTD 1 | 64.8204 | 62.0684 | 10 / 10 | 0 / 0 |
| UTD | previous | rollout | — | dropped (ΔF≈0) | dropped (ΔF≈0) | 0 / 0 | 10 / 10 |
| UTD | previous | critic | UTD 2←UTD 1, UTD 4←UTD 2 | 49.4387 | 46.5852 | 10 / 10 | 0 / 0 |
| UTD | previous | actor | UTD 2←UTD 1, UTD 4←UTD 2 | 75.9952 | 71.4041 | 10 / 10 | 0 / 0 |
| UTD | previous | dyn_model | UTD 2←UTD 1, UTD 4←UTD 2 | 1038.75 | -369.436 | 9 / 8 | 1 / 2 |
| UTD | previous | synth_rollout | — | dropped (ΔF≈0) | 14.7312 | 0 / 6 | 10 / 4 |
| UTD | previous | TOTAL | UTD 2←UTD 1, UTD 4←UTD 2 | 63.3176 | 60.7052 | 10 / 10 | 0 / 0 |
| max rollout length | first | rollout | — | n/a | dropped (ΔF≈0) | n/a / 0 | n/a / 10 |
| max rollout length | first | critic | — | n/a | dropped (ΔF≈0) | n/a / 0 | n/a / 10 |
| max rollout length | first | actor | — | n/a | dropped (ΔF≈0) | n/a / 0 | n/a / 10 |
| max rollout length | first | dyn_model | rl1-15_re20-100←rl1-1_re20-100, rl1-25_re20-100←rl1-1_re20-100 | n/a | 1492.74 | n/a / 10 | n/a / 0 |
| max rollout length | first | synth_rollout | rl1-15_re20-100←rl1-1_re20-100, rl1-25_re20-100←rl1-1_re20-100 | n/a | 14.1525 | n/a / 10 | n/a / 0 |
| max rollout length | first | TOTAL | rl1-15_re20-100←rl1-1_re20-100, rl1-25_re20-100←rl1-1_re20-100 | n/a | 269.407 | n/a / 6 | n/a / 4 |
| max rollout length | previous | rollout | — | n/a | dropped (ΔF≈0) | n/a / 0 | n/a / 10 |
| max rollout length | previous | critic | — | n/a | dropped (ΔF≈0) | n/a / 0 | n/a / 10 |
| max rollout length | previous | actor | — | n/a | dropped (ΔF≈0) | n/a / 0 | n/a / 10 |
| max rollout length | previous | dyn_model | rl1-15_re20-100←rl1-1_re20-100, rl1-25_re20-100←rl1-15_re20-100 | n/a | 4428.68 | n/a / 10 | n/a / 0 |
| max rollout length | previous | synth_rollout | rl1-15_re20-100←rl1-1_re20-100, rl1-25_re20-100←rl1-15_re20-100 | n/a | 14.8885 | n/a / 10 | n/a / 0 |
| max rollout length | previous | TOTAL | rl1-15_re20-100←rl1-1_re20-100, rl1-25_re20-100←rl1-15_re20-100 | n/a | 286.793 | n/a / 5 | n/a / 5 |

Per-run vs cross-seed pooled values: max relative difference 2.78e+00 over 66 comparable (sweep, reference, env, segment) cells. The cross-seed CSV has no `gradient_updates` row, so the dashboard rule yields no GU pooled value; the OLS slope b in 5.1a is the closest equivalent.

**5.2a Task durations and the 1 s polling interval** — all per-epoch measured tasks of all 5 seeds (100 epochs × 5 seeds = 500 tasks per task type; `measure_power_secs` = 1.0 s). Cell: median [min–max] of the single-task CodeCarbon `duration` [s]; and number of tasks shorter than the polling interval / total:

| config | dyn_model duration median [min–max] s | dyn_model tasks < 1 s | rollout duration median [min–max] s | rollout tasks < 1 s | synth_rollout duration median [min–max] s | synth_rollout tasks < 1 s | GU duration median [min–max] s | GU tasks < 1 s |
|---|---|---|---|---|---|---|---|---|
| base | 11.1913 [1.11163–65.7468] ‖ 15.7071 [1.03035–49.5540] | 0/500 ‖ 0/500 | 0.202506 [0.194022–0.250448] ‖ 0.366091 [0.349383–0.400519] | 500/500 ‖ 500/500 | 0.00606905 [0.00551315–0.0189979] ‖ 0.0502715 [0.00741655–0.116011] | 500/500 ‖ 500/500 | 4.02842 [3.96949–4.41997] ‖ 4.07810 [4.02774–4.42654] | 0/500 ‖ 0/500 |
| utd2 | 11.0176 [1.11316–135.478] ‖ 17.2284 [1.03650–51.1903] | 0/500 ‖ 0/500 | 0.205400 [0.197299–0.257542] ‖ 0.367967 [0.349386–0.400819] | 500/500 ‖ 500/500 | 0.00612268 [0.00549331–0.0280610] ‖ 0.0502200 [0.00752167–0.116328] | 500/500 ‖ 500/500 | 8.06169 [7.90925–8.97450] ‖ 8.18099 [8.06767–9.03571] | 0/500 ‖ 0/500 |
| utd4 | 10.9118 [1.19580–92.1921] ‖ 15.8063 [1.12007–55.3942] | 0/500 ‖ 0/500 | 0.204330 [0.195258–0.239836] ‖ 0.366229 [0.348346–0.402018] | 500/500 ‖ 500/500 | 0.00612139 [0.00550780–0.0187484] ‖ 0.0495167 [0.00745800–0.110539] | 500/500 ‖ 500/500 | 16.1673 [15.7996–17.5592] ‖ 16.3536 [16.0842–17.8972] | 0/500 ‖ 0/500 |
| w256 | 10.4651 [1.10036–90.5688] ‖ 15.5648 [0.893875–40.7894] | 0/500 ‖ 1/500 | 0.197560 [0.188283–0.236392] ‖ 0.358429 [0.341409–0.397570] | 500/500 ‖ 500/500 | 0.00555116 [0.00456115–0.0303995] ‖ 0.0441956 [0.00669379–0.100165] | 500/500 ‖ 500/500 | 3.94404 [3.84608–4.29444] ‖ 3.96306 [3.89092–4.31364] | 0/500 ‖ 0/500 |
| w512 | 10.9365 [1.17515–78.6732] ‖ 14.7441 [1.50517–43.0690] | 0/500 ‖ 0/500 | 0.198980 [0.191323–0.241522] ‖ 0.361417 [0.343296–0.392442] | 500/500 ‖ 500/500 | 0.00570408 [0.00510443–0.0178558] ‖ 0.0459479 [0.00675887–0.103290] | 500/500 ‖ 500/500 | 3.97224 [3.90494–4.35966] ‖ 4.00127 [3.93141–4.35455] | 0/500 ‖ 0/500 |
| b512 | 10.4716 [1.16708–100.512] ‖ 14.3469 [1.02706–38.0735] | 0/500 ‖ 0/500 | 0.201403 [0.195144–0.246764] ‖ 0.364172 [0.347394–0.407737] | 500/500 ‖ 500/500 | 0.00609342 [0.00552155–0.0200573] ‖ 0.0497155 [0.00786326–0.116229] | 500/500 ‖ 500/500 | 4.10948 [4.05414–4.47320] ‖ 4.17408 [4.10844–4.53274] | 0/500 ‖ 0/500 |
| b1024 | 10.8694 [1.27704–108.956] ‖ 15.1486 [1.17804–37.3316] | 0/500 ‖ 0/500 | 0.200621 [0.196227–0.248231] ‖ 0.366960 [0.347204–0.405528] | 500/500 ‖ 500/500 | 0.00611832 [0.00551527–0.0201781] ‖ 0.0483935 [0.00741284–0.111331] | 500/500 ‖ 500/500 | 4.16095 [4.09927–4.56507] ‖ 4.37362 [4.27160–4.69585] | 0/500 ‖ 0/500 |
| rollout1 | n/a ‖ 14.7444 [1.02911–39.3405] | n/a ‖ 0/500 | n/a ‖ 0.369123 [0.347411–0.406262] | n/a ‖ 500/500 | n/a ‖ 0.00837969 [0.00727289–0.0220967] | n/a ‖ 500/500 | n/a ‖ 4.12980 [4.04491–4.51714] | n/a ‖ 0/500 |
| rollout15 | n/a ‖ 15.2178 [1.02884–35.3418] | n/a ‖ 0/500 | n/a ‖ 0.365173 [0.345637–0.399555] | n/a ‖ 500/500 | n/a ‖ 0.0340090 [0.00734143–0.0715587] | n/a ‖ 500/500 | n/a ‖ 4.09323 [4.02587–4.54421] | n/a ‖ 0/500 |

Across all 80 runs: `dynamics_model_update`: 1 of 8000 tasks < 1.0 s (min/median/max 0.893875/13.5768/135.478 s); `rollout`: 8000 of 8000 tasks < 1.0 s (min/median/max 0.188283/0.353606/0.407737 s); `synthetic_rollout_generation`: 8000 of 8000 tasks < 1.0 s (min/median/max 0.00456115/0.00852251/0.116328 s); `gradient_updates`: 0 of 8000 tasks < 1.0 s (min/median/max 3.84608/4.18918/17.8972 s).
**5.2b Integration span versus task duration** — span = `ram_energy / ram_power` of the task row (CodeCarbon models RAM at constant power, so this is the time over which the task's energy was integrated); the cell is the ratio span/duration (1 = integrated over exactly the recorded duration):

| config | dyn_model integration span / duration: median [min–max] | rollout integration span / duration: median [min–max] | synth_rollout integration span / duration: median [min–max] | GU integration span / duration: median [min–max] |
|---|---|---|---|---|
| base | 0.999956 [0.999321–1.00047] ‖ 0.999965 [0.999038–1.00016] | 0.998359 [0.989868–1.00768] ‖ 0.998852 [0.994247–1.00366] | 0.943113 [0.726027–1.22350] ‖ 0.988420 [0.792183–1.10949] | 0.999889 [0.999457–1.00031] ‖ 0.999901 [0.999484–1.00033] |
| utd2 | 0.999955 [0.998909–1.00061] ‖ 0.999967 [0.999016–1.00057] | 0.998293 [0.990061–1.00728] ‖ 0.998827 [0.989440–1.00350] | 0.941707 [0.736382–1.22403] ‖ 0.986714 [0.787180–1.15078] | 0.999943 [0.999742–1.00016] ‖ 0.999943 [0.999726–1.00015] |
| utd4 | 0.999953 [0.998849–1.00021] ‖ 0.999966 [0.999205–1.00021] | 0.998336 [0.989837–1.00734] ‖ 0.998876 [0.994429–1.00403] | 0.941230 [0.725477–1.23810] ‖ 0.987500 [0.785342–1.12779] | 0.999972 [0.999865–1.00008] ‖ 0.999972 [0.999868–1.00008] |
| w256 | 0.999955 [0.999131–1.00092] ‖ 0.999964 [0.998956–1.00011] | 0.999052 [0.990131–1.00815] ‖ 0.999292 [0.994545–1.00438] | 0.954733 [0.713959–1.22971] ‖ 0.985099 [0.768514–1.13236] | 0.999890 [0.999465–1.00032] ‖ 0.999896 [0.999445–1.00034] |
| w512 | 0.999955 [0.999147–1.00049] ‖ 0.999963 [0.999352–1.00051] | 0.998912 [0.990296–1.00727] ‖ 0.999180 [0.994355–1.00380] | 0.941038 [0.714967–1.24008] ‖ 0.986614 [0.776060–1.10530] | 0.999885 [0.999445–1.00030] ‖ 0.999897 [0.999474–1.00030] |
| b512 | 0.999955 [0.999544–1.00086] ‖ 0.999962 [0.999357–1.00090] | 0.998832 [0.990314–1.00744] ‖ 0.999050 [0.994262–1.00409] | 0.943552 [0.757363–1.24231] ‖ 0.986877 [0.783184–1.13660] | 0.999899 [0.999464–1.00033] ‖ 0.999893 [0.999473–1.00031] |
| b1024 | 0.999955 [0.999363–1.00026] ‖ 0.999965 [0.999187–1.00024] | 0.998786 [0.991370–1.00765] ‖ 0.999064 [0.993933–1.00386] | 0.941723 [0.736113–1.24002] ‖ 0.986921 [0.792767–1.13462] | 0.999894 [0.999474–1.00033] ‖ 0.999882 [0.999473–1.00065] |
| rollout1 | n/a ‖ 0.999964 [0.999310–1.00039] | n/a ‖ 0.999077 [0.994665–1.00391] | n/a ‖ 0.951979 [0.783131–1.15004] | n/a ‖ 0.999886 [0.999460–1.00032] |
| rollout15 | n/a ‖ 0.999964 [0.998124–1.00034] | n/a ‖ 0.999073 [0.994309–1.00397] | n/a ‖ 0.981550 [0.795749–1.16837] | n/a ‖ 0.999891 [0.999480–1.00034] |

**5.2c Per-epoch stationarity** (per run: CV of the 100 per-epoch energies; ratio of the epoch-0 energy to the mean of epochs 1–99; relative difference of the mean duration of epochs 0–9 versus epochs 10–99; then mean ± sd over the 5 seeds; per-epoch values in the per-epoch CSV):

| config | dyn_model per-epoch energy CV within run [%] | rollout per-epoch energy CV within run [%] | synth_rollout per-epoch energy CV within run [%] | GU per-epoch energy CV within run [%] | dyn_model energy epoch 0 / mean(epochs 1–99) | rollout energy epoch 0 / mean(epochs 1–99) | synth_rollout energy epoch 0 / mean(epochs 1–99) | GU energy epoch 0 / mean(epochs 1–99) | dyn_model duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%] | rollout duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%] | synth_rollout duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%] | GU duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 72.4198 ± 5.34366 ‖ 45.2846 ± 3.19034 | 6.80536 ± 1.69008 ‖ 7.61628 ± 0.457413 | 216.862 ± 7.26022 ‖ 90.8070 ± 0.736143 | 2.35439 ± 0.161460 ‖ 1.94037 ± 0.190407 | 1.86484 ± 0.206107 ‖ 0.243924 ± 0.122481 | 1.23762 ± 0.110522 ‖ 1.07517 ± 0.0155941 | 1.26511 ± 0.350687 ‖ 0.149889 ± 0.0116674 | 0.987520 ± 0.00888944 ‖ 0.997859 ± 0.00938734 | -66.1055 ± 4.29581 ‖ -73.9847 ± 3.88158 | 2.70851 ± 1.02725 ‖ -1.56863 ± 0.467047 | 16.0826 ± 1.14626 ‖ -81.8782 ± 0.272986 | -0.688759 ± 1.51403 ‖ 0.350830 ± 1.39459 |
| utd2 | 85.2349 ± 20.5402 ‖ 47.5651 ± 2.42831 | 7.86825 ± 2.14980 ‖ 7.66392 ± 0.434462 | 217.218 ± 5.91153 ‖ 88.2887 ± 4.35699 | 2.90737 ± 0.487350 ‖ 2.81443 ± 0.748494 | 1.72724 ± 0.237663 ‖ 0.242796 ± 0.124563 | 1.08282 ± 0.142660 ‖ 1.06899 ± 0.0339893 | 7.44107 ± 5.95385 ‖ 0.147460 ± 0.00592577 | 0.998435 ± 0.0338287 ‖ 0.984808 ± 0.0171416 | -65.4043 ± 3.24026 ‖ -77.5309 ± 2.39130 | 2.05644 ± 1.52885 ‖ -1.30559 ± 0.682130 | 18.8670 ± 7.55287 ‖ -81.7036 ± 0.0423327 | -1.86610 ± 0.978597 ‖ 0.130973 ± 2.89765 |
| utd4 | 79.8943 ± 12.7108 ‖ 45.7471 ± 4.86998 | 7.59469 ± 0.983185 ‖ 7.68373 ± 0.547409 | 216.442 ± 6.36949 ‖ 89.8085 ± 7.19423 | 2.67131 ± 0.457678 ‖ 2.52571 ± 0.195852 | 1.87858 ± 0.335919 ‖ 0.243674 ± 0.114763 | 1.23460 ± 0.157650 ‖ 1.06881 ± 0.0313802 | 3.30162 ± 4.63706 ‖ 0.153926 ± 0.00668666 | 0.996671 ± 0.0140670 ‖ 1.00548 ± 0.0243906 | -63.9814 ± 3.78044 ‖ -74.3523 ± 3.63168 | 2.01854 ± 0.568697 ‖ -0.544801 ± 0.423718 | 15.8031 ± 1.75022 ‖ -81.1866 ± 0.536825 | 0.582097 ± 1.98190 ‖ 0.177404 ± 1.52830 |
| w256 | 84.7917 ± 14.6864 ‖ 48.2129 ± 3.04985 | 7.08198 ± 1.07310 ‖ 8.49572 ± 0.446509 | 217.808 ± 19.1106 ‖ 90.2692 ± 2.08982 | 3.52254 ± 0.256412 ‖ 3.89170 ± 0.455144 | 1.53891 ± 0.359897 ‖ 0.209991 ± 0.0584851 | 1.20025 ± 0.166543 ‖ 1.06490 ± 0.0351414 | 1.87365 ± 0.634293 ‖ 0.180753 ± 0.0113791 | 0.994570 ± 0.0378367 ‖ 0.960016 ± 0.00876283 | -64.2853 ± 4.10580 ‖ -78.6460 ± 2.20294 | 2.45114 ± 1.24163 ‖ -1.52968 ± 0.606483 | 23.7282 ± 17.8787 ‖ -79.8712 ± 0.492132 | 0.841716 ± 3.80530 ‖ -1.23933 ± 1.84368 |
| w512 | 78.9004 ± 9.39920 ‖ 46.2652 ± 1.07326 | 6.35537 ± 0.986331 ‖ 8.28375 ± 0.561418 | 220.022 ± 3.88170 ‖ 89.8809 ± 1.56031 | 3.08171 ± 0.508916 ‖ 3.44263 ± 0.527489 | 1.80265 ± 0.228012 ‖ 0.249146 ± 0.0851884 | 1.12893 ± 0.135341 ‖ 1.05807 ± 0.0289922 | 3.13942 ± 3.70498 ‖ 0.161885 ± 0.00429633 | 0.967654 ± 0.0113228 ‖ 0.963990 ± 0.0119704 | -64.7768 ± 3.25906 ‖ -71.5745 ± 6.88384 | 2.71553 ± 0.733942 ‖ -2.16517 ± 0.722862 | 19.3936 ± 4.96212 ‖ -80.5965 ± 0.357370 | -0.932189 ± 3.55401 ‖ -1.96745 ± 1.00833 |
| b512 | 75.2342 ± 11.7719 ‖ 44.5021 ± 1.32206 | 6.87844 ± 0.575827 ‖ 7.90791 ± 0.391266 | 219.380 ± 5.13736 ‖ 88.2723 ± 4.16525 | 1.66305 ± 0.256320 ‖ 1.84295 ± 0.334954 | 1.92523 ± 0.107813 ‖ 0.277148 ± 0.150770 | 1.09508 ± 0.131173 ‖ 1.07845 ± 0.0306140 | 3.67123 ± 5.15572 ‖ 0.148198 ± 0.0123897 | 0.996891 ± 0.0199935 ‖ 0.990310 ± 0.0122006 | -62.6061 ± 4.77930 ‖ -75.0402 ± 3.31856 | 3.10693 ± 0.378237 ‖ -1.81895 ± 0.528680 | 17.3900 ± 2.65626 ‖ -81.6997 ± 0.427351 | -1.62715 ± 1.24690 ‖ -0.778568 ± 1.70440 |
| b1024 | 79.6497 ± 10.8019 ‖ 44.3933 ± 3.36804 | 5.54188 ± 1.13960 ‖ 7.81285 ± 0.199321 | 210.092 ± 13.4023 ‖ 89.3118 ± 3.92440 | 1.57316 ± 0.278608 ‖ 1.68867 ± 0.191288 | 1.94260 ± 0.118394 ‖ 0.249604 ± 0.116325 | 1.27930 ± 0.128990 ‖ 1.06843 ± 0.0260629 | 1.22460 ± 0.330418 ‖ 0.149982 ± 0.00527850 | 1.00667 ± 0.00837451 ‖ 0.986436 ± 0.0132244 | -61.9712 ± 3.41787 ‖ -71.3499 ± 5.65493 | 2.05045 ± 0.691575 ‖ -1.24411 ± 0.827492 | 16.5837 ± 3.19063 ‖ -81.4373 ± 0.282517 | -1.51739 ± 0.985266 ‖ -2.63127 ± 0.722772 |
| rollout1 | n/a ‖ 44.1905 ± 3.05603 | n/a ‖ 7.83665 ± 0.586477 | n/a ‖ 168.389 ± 7.12375 | n/a ‖ 2.82530 ± 0.390332 | n/a ‖ 0.250061 ± 0.107532 | n/a ‖ 1.05293 ± 0.0234341 | n/a ‖ 0.913622 ± 0.134010 | n/a ‖ 0.990698 ± 0.0110001 | n/a ‖ -68.3652 ± 7.43111 | n/a ‖ -1.33485 ± 0.192840 | n/a ‖ 3.54704 ± 1.66816 | n/a ‖ -2.14054 ± 1.46179 |
| rollout15 | n/a ‖ 44.8652 ± 2.19363 | n/a ‖ 7.88300 ± 0.436852 | n/a ‖ 94.9265 ± 5.15350 | n/a ‖ 2.76130 ± 0.327141 | n/a ‖ 0.243624 ± 0.112996 | n/a ‖ 1.07047 ± 0.0133267 | n/a ‖ 0.242998 ± 0.0211278 | n/a ‖ 0.975608 ± 0.0133054 | n/a ‖ -72.3897 ± 4.65830 | n/a ‖ -1.47125 ± 0.858668 | n/a ‖ -73.3115 ± 0.190071 | n/a ‖ -1.66001 ± 0.954442 |

**5.2d Between-seed variability and allocation coverage:**

| config | CV of TOTAL energy over seeds [%] | allocation coverage min–max [%] | CV of dyn_model energy over seeds [%] | CV of rollout energy over seeds [%] | CV of synth_rollout energy over seeds [%] | CV of GU energy over seeds [%] |
|---|---|---|---|---|---|---|
| base | 3.59287 ‖ 7.95614 | 96.8947–96.9927 ‖ 96.9168–97.0815 | 5.08865 ‖ 10.7747 | 1.96654 ‖ 0.695742 | 21.7109 ‖ 5.07278 | 0.360979 ‖ 0.349477 |
| utd2 | 2.57712 ‖ 5.40989 | 96.8873–97.0070 ‖ 97.0772–97.1675 | 4.48629 ‖ 9.58297 | 2.04469 ‖ 0.846870 | 15.7108 ‖ 5.59875 | 0.901727 ‖ 1.33526 |
| utd4 | 2.28715 ‖ 3.86261 | 96.8012–96.9051 ‖ 97.1124–97.2443 | 6.22582 ‖ 9.00351 | 0.974421 ‖ 1.42447 | 17.0140 ‖ 5.91271 | 0.271745 ‖ 0.311663 |
| w256 | 2.58078 ‖ 4.20888 | 97.5805–97.6441 ‖ 97.5168–97.6637 | 3.23081 ‖ 5.36634 | 1.09615 ‖ 1.15482 | 22.4034 ‖ 4.92358 | 1.02175 ‖ 0.393391 |
| w512 | 4.57250 ‖ 8.01407 | 97.3207–97.4023 ‖ 97.1777–97.3306 | 6.51923 ‖ 10.4752 | 1.36932 ‖ 0.826613 | 23.2705 ‖ 4.28030 | 1.34501 ‖ 0.998149 |
| b512 | 2.34636 ‖ 8.56907 | 97.0768–97.1669 ‖ 97.1501–97.2869 | 3.63930 ‖ 12.4234 | 0.618530 ‖ 0.648000 | 12.5674 ‖ 6.04947 | 0.517748 ‖ 0.554688 |
| b1024 | 2.29103 ‖ 5.65589 | 97.1160–97.2151 ‖ 97.1638–97.2916 | 3.49494 ‖ 8.64159 | 0.641796 ‖ 0.411553 | 27.0199 ‖ 3.68924 | 0.352327 ‖ 0.374815 |
| rollout1 | n/a ‖ 1.36282 | n/a ‖ 96.4058–96.5773 | n/a ‖ 2.11254 | n/a ‖ 0.916684 | n/a ‖ 15.0553 | n/a ‖ 0.980042 |
| rollout15 | n/a ‖ 5.14757 | n/a ‖ 97.0202–97.0911 | n/a ‖ 7.02739 | n/a ‖ 0.652095 | n/a ‖ 9.04068 | n/a ‖ 0.335700 |

**5.2e Idle head versus tail** (as-recorded mean power = idle-task energy / its CodeCarbon `duration`):

| config | P_head [W] | P_tail [W] | P_tail − P_head [W] | \|P_tail − P_head\| / P_head [%] |
|---|---|---|---|---|
| base | 64.6743 ± 0.435338 ‖ 64.0415 ± 0.371202 | 67.1406 ± 0.377335 ‖ 67.0772 ± 0.389864 | 2.46634 ± 0.518147 ‖ 3.03567 ± 0.517665 | 3.81663 ± 0.818699 ‖ 4.74276 ± 0.827409 |
| utd2 | 64.6818 ± 2.36746 ‖ 63.3889 ± 0.476128 | 66.1159 ± 0.738939 ‖ 65.9811 ± 0.238672 | 1.43417 ± 1.78529 ‖ 2.59227 ± 0.602390 | 3.25542 ± 0.887089 ‖ 4.09493 ± 0.973787 |
| utd4 | 63.9304 ± 1.45527 ‖ 63.6218 ± 0.484180 | 67.1108 ± 0.552129 ‖ 66.3727 ± 0.536015 | 3.18035 ± 1.13876 ‖ 2.75088 ± 0.810845 | 5.00707 ± 1.87751 ‖ 4.32992 ± 1.29385 |
| w256 | 62.6464 ± 1.10931 ‖ 62.3711 ± 0.597112 | 64.6814 ± 0.629656 ‖ 64.3041 ± 0.484763 | 2.03507 ± 1.50090 ‖ 1.93300 ± 0.836641 | 3.31808 ± 2.37857 ‖ 3.10794 ± 1.37415 |
| w512 | 62.8690 ± 0.785523 ‖ 62.0453 ± 0.546324 | 65.0994 ± 0.867895 ‖ 64.5213 ± 0.361899 | 2.23035 ± 1.31123 ‖ 2.47601 ± 0.508238 | 3.56398 ± 2.11589 ‖ 3.99527 ± 0.841541 |
| b512 | 62.3888 ± 0.676432 ‖ 61.7030 ± 0.631717 | 65.5895 ± 0.220067 ‖ 64.3044 ± 0.778061 | 3.20069 ± 0.759769 ‖ 2.60140 ± 1.38081 | 5.14077 ± 1.26141 ‖ 4.23425 ± 2.28122 |
| b1024 | 62.0918 ± 0.979671 ‖ 62.1208 ± 0.794931 | 65.5608 ± 0.695378 ‖ 65.4469 ± 0.579464 | 3.46904 ± 1.57146 ‖ 3.32612 ± 0.984265 | 5.61848 ± 2.60722 ‖ 5.36814 ± 1.64330 |
| rollout1 | n/a ‖ 62.2201 ± 1.26858 | n/a ‖ 64.5651 ± 0.478664 | n/a ‖ 2.34496 ± 1.17379 | n/a ‖ 3.79875 ± 1.98131 |
| rollout15 | n/a ‖ 62.0251 ± 0.850444 | n/a ‖ 65.3509 ± 0.860956 | n/a ‖ 3.32583 ± 1.63532 | n/a ‖ 5.39065 ± 2.71793 |

Over all 80 runs of MBPO: |P_tail − P_head|/P_head (as recorded) median 4.13198 %, mean 4.29894 %, maximum 8.98642 % (run `results/mbpo/HalfCheetah-v5/seed_14577/20260915_180602`, config b1024, HalfCheetah-v5, seed 14577); tail > head in 78/80 runs. Window-corrected (energy ÷ integrated span, see `idle_power_analysis.py`): median 7.67666 %, maximum 12.7188 % (run `results/mbpo/HalfCheetah-v5/seed_14577/20260915_180602`). Median head integration span 93.0821 s vs recorded duration 90.0030 s; tail 90.0024 s vs 90.0034 s.
**5.2f Outlier flags** — rule: |x − median| > 3 × MAD (unscaled MAD, n = 5 per configuration), applied to the energy of every segment of the energy table (incl. GU and TOTAL). Flagged, **not excluded**: 101 of 720 (run, segment) cells; largest deviation 55.7640 % of the median (HalfCheetah-v5 b1024 seed 85062 synth_rollout: 108.092 J vs median 69.3950 J). Full list: `contexts/Section_4.3_data/mbpo_mad_flags.csv`.
Flag count per configuration: Ant-v5/b1024: 6, Ant-v5/b512: 2, Ant-v5/base: 5, Ant-v5/rollout1: 5, Ant-v5/rollout15: 5, Ant-v5/utd2: 14, Ant-v5/utd4: 5, Ant-v5/w512: 6, HalfCheetah-v5/b1024: 7, HalfCheetah-v5/b512: 5, HalfCheetah-v5/base: 6, HalfCheetah-v5/utd2: 9, HalfCheetah-v5/utd4: 7, HalfCheetah-v5/w256: 10, HalfCheetah-v5/w512: 9.

**5.3 Episodes and returns (sanity information only)** — definitions of `aggregate_results.py`: episodes = `training_metrics.json["episodes"]` with `phase == "train"` (warmup and the trailing partial `train_incomplete` episode excluded); final-10 %-mean return = mean of the last max(1, n_episodes // 10) of them; returns are those of the stochastic training policy (no evaluation episodes). Cell: completed training episodes per run (min–max over the 5 seeds); number of episodes in the 10 % window (min–max); mean ± sd over seeds of the window mean return:

| config | HalfCheetah-v5 | Ant-v5 |
|---|---|---|
| base | episodes 100–100; window 10–10; return 5146.45 ± 642.812 | episodes 196–282; window 19–28; return -325.478 ± 159.973 |
| utd2 | episodes 100–100; window 10–10; return 5285.43 ± 499.802 | episodes 214–285; window 21–28; return -459.138 ± 263.595 |
| utd4 | episodes 100–100; window 10–10; return 5903.65 ± 333.884 | episodes 243–431; window 24–43; return -599.210 ± 186.083 |
| w256 | episodes 100–100; window 10–10; return 4534.75 ± 1027.80 | episodes 240–303; window 24–30; return -695.464 ± 246.076 |
| w512 | episodes 100–100; window 10–10; return 5133.81 ± 451.378 | episodes 235–268; window 23–26; return -413.374 ± 167.563 |
| b512 | episodes 100–100; window 10–10; return 5055.13 ± 468.323 | episodes 230–358; window 23–35; return -244.572 ± 172.976 |
| b1024 | episodes 100–100; window 10–10; return 5181.48 ± 566.144 | episodes 219–285; window 21–28; return -348.459 ± 198.167 |
| rollout1 | n/a | episodes 154–230; window 15–23; return -756.748 ± 524.963 |
| rollout15 | n/a | episodes 211–301; window 21–30; return -407.716 ± 245.806 |


## 6. Algorithm-specific items (MBPO)

**6.1 Code facts** (`algorithms/mbpo.py`, `algorithms/dynamics_model.py`, `algorithms/termination_fns.py`, `configs/config.py`, `configs/overrides/mbpo_ant*.json`; HEAD, identical at all run commits):


**(a) Order of the blocks within an epoch and their task names** (`train()`, lines 140–332; `TrackerTask(tracker, prefix, epoch, …)` names the task `f"{prefix}_{epoch}"`, `algorithms/tracker_utils.py`):
1. `dynamics_model_update_{epoch}` — only if `(global_step − last_model_train_step) ≥ model_train_freq` (line 232; `last_model_train_step` starts at −model_train_freq, line 227, so a fit happens in epoch 0): `model.fit(real_buffer)` (task at line 233);
2. `rollout_{epoch}` (line 240): 1000 env steps with `agent.select_action(obs, deterministic=False)` (stochastic SAC policy), `env.step`, `real_buffer.add`, episode bookkeeping/reset — identical in structure to SAC;
3. `synthetic_rollout_generation_{epoch}` (line 258): `_generate_model_rollouts(...)` with `rollout_length = _rollout_length(epoch, cfg)`;
4. `gradient_updates_{epoch}` (line 267): `steps_per_epoch × updates_per_env_step` calls of `SACAgent.update(mixed_buffer, batch_size)` with the four `perf_counter` sub-timers (`buffer_sample`, `critic_update`, `actor_update`, `target_update`), allocated from the measured GU energy at the end of the run (line 323).
`warmup_0` (random actions into the real buffer) precedes epoch 0 and is excluded from every total. The per-task CSV of every run has exactly 100 tasks of each of the four per-epoch prefixes (Section 1.3 check), i.e. a fit in every epoch.

**(b) Model fit** (`EnsembleDynamicsModel.fit`, `dynamics_model.py` lines 100–228):
- `model_train_freq` = 250 real env steps; `steps_per_epoch` = 1000 ≥ `model_train_freq`, so the condition holds at the start of every epoch ⇒ **100 fits per run** (1 per epoch; verified from the logs: Section 6.2). The fit retrains the ensemble **from the current weights** on all real data in the buffer (`n = buffer.size`; the weights are not re-initialised; the Adam optimiser state persists), inputs normalised by a freshly fitted `RunningNormalizer`.
- Fit batch size `model_train_batch_size` = 256; holdout ratio `model_holdout_ratio` = 0.2 → `n_holdout = max(1, int(n · 0.2))` (line 165), `n_train = n − n_holdout`.
- Bootstrap: `bootstrap_idx = np.random.randint(0, n_train, size=(ensemble_size, n_train))` (line 176) — each member gets its own resample **with replacement** of the training split, fixed for the fit call and reshuffled every epoch; every member sees `n_train` samples per epoch.
- Per optimiser step all 7 members are evaluated on their own minibatch (`for i in range(ensemble_size)`), one summed Gaussian-NLL loss + 0.01·(Σmax_logvar − Σmin_logvar), one backward, one Adam step (`model_lr` = 0.001, per-layer weight decay [2.5e-05, 5e-05, 7.5e-05, 0.0001, 0.0001]). One *epoch* = ceil(n_train / 256) optimiser steps.
- Early stopping (lines 210–216): after each epoch the per-member holdout MSE is computed (`_holdout_mse`, one `forward_all` over the holdout set); training stops when the **mean** holdout MSE has not improved by more than 1e-4 for `model_train_patience` = 5 consecutive epochs, or after `model_max_train_epochs` = 200 epochs. After the loop the `num_elites` = 5 members with the lowest per-member holdout MSE **of the best epoch** become the elites (line 147). The weights are *not* restored to the best epoch.
- `ensemble_size` = 7, `num_elites` = 5, `model_hidden_sizes` = [200, 200, 200, 200] (Swish activations), `deterministic_model` = False. Parameters: Section 2.4.

**(c) Synthetic rollouts** (`_generate_model_rollouts`, lines 94–138):
- `rollout_batch_size` = 10000: `n_start = min(rollout_batch_size, len(real_buffer))` (line 110) start states drawn uniformly (with replacement) from the real buffer; since the real buffer holds warmup + (epoch+1)·1000 transitions at that point, n_start = 6000, 7000, 8000, 9000 for epochs 0–3 and 10000 from epoch 4 on.
- Per step (loop `for _ in range(rollout_length)`): actor forward (stochastic, `with_logprob=False`) on all live states; `model.predict` (line 122): normalise, **all 7 members' forward** (`forward_all`), pick one random elite member per row, sample next-Δobs and reward from its Gaussian (`deterministic_model` = False); `done = termination_fn(next_obs)` (line 127); `model_buffer.add_batch` of every generated transition (line 129); `keep = ~done`, the loop stops if no state is kept, otherwise continues with `next_obs[keep]` (lines 132). Terminated rows are dropped from further steps; the terminating transition itself **is** stored (with `done` = True).
- Termination function (`get_termination_fn`, line 73; `termination_fns.py`): HalfCheetah-v5 → `_never_done` (line 26; HC never terminates), Ant-v5 → `_ant_done` (line 47): `healthy = isfinite(next_obs).all & (z ≥ 0.2) & (z ≤ 1.0)` with z = `next_obs[:, 0]`, `done = ~healthy`.
- Rollout-length schedule (`_rollout_length`, line 81): `epoch ≤ rollout_min_epoch → rollout_min_length`; `epoch ≥ rollout_max_epoch → rollout_max_length`; else `int(round(min_len + (epoch − min_epoch)/max(1, max_epoch − min_epoch) · (max_len − min_len)))`. Per-epoch table below (Section 6.1e).
- Model buffer: capacity = `max(10000, rollout_batch_size · rollout_max_length · model_retain_epochs)` (line 175), `model_retain_epochs` = 1; a circular `ReplayBuffer` (`add_batch` overwrites the oldest entries). HC: 10000 · 1 · 1 = 10000; Ant L=25: 250000; Ant L=15: 150000; Ant L=1: 10000. Retained epochs of model data per regime: Section 6.1f.

**(d) `real_ratio` and the minibatch composition** (`_MixedReplayBuffer.sample`, lines 47–79; created at line 179): `real_ratio` = 0.05. `if len(model_buffer) == 0` the batch is purely real (line 63); this never happens during the measured `gradient_updates` because the model buffer is filled by the epoch's synthetic generation before the first gradient update of epoch 0. Otherwise `n_real = min(max(int(round(batch_size · real_ratio)), 0), batch_size)` (line 68), `n_model = batch_size − n_real`; two separate host-side `ReplayBuffer.sample` calls (real, model) and a NumPy `concatenate` of the five arrays; the concatenated host arrays are then moved to the GPU inside `SACAgent.update`'s `buffer_sample` timer. batch 256 → 13 real + 243 model; batch 512 → 26 + 486; batch 1024 → 51 + 973. (`len(mixed_buffer)` = real + model sizes, so the `len < batch_size` guard in the update loop never triggers.)

**(e) Is `SACAgent` used unmodified, and which quantities does each sweep touch?**
- `mbpo.py` imports `SACAgent` from `algorithms.sac` and instantiates it as `SACAgent(obs_dim, act_dim, act_limit, mbpo_cfg, device)` (line 166); there is no subclass, wrapper, monkey-patch or attribute assignment on it in `mbpo.py` (regex search for `class …(SACAgent)`: none; for `agent.<attr> = …` assignments and `setattr(`: 0 found); the only members of the agent that `mbpo.py` touches are: `agent.actor`, `agent.select_action`, `agent.update`; `algorithms/sac.py` is byte-identical at all run commits (Section 0). `SACAgent` reads these config fields (parsed from its source): `actor_lr`, `alpha_lr`, `autotune_alpha`, `critic_lr`, `gamma`, `hidden_sizes`, `init_alpha`, `policy_update_delay`, `target_entropy`, `tau`; MBPOConfig fields **not** read by `SACAgent` (consumed by `mbpo.py`/`dynamics_model.py`): `updates_per_env_step`, `batch_size`, `real_ratio`, `buffer_capacity`, `ensemble_size`, `num_elites`, `model_hidden_sizes`, `model_lr`, `model_weight_decays`, `model_train_batch_size`, `model_holdout_ratio`, `model_max_train_epochs`, `model_train_patience`, `model_train_freq`, `deterministic_model`, `rollout_batch_size`, `model_retain_epochs`, `rollout_min_epoch`, `rollout_max_epoch`, `rollout_min_length`, `rollout_max_length`.
- **UTD** (`updates_per_env_step`): used only for `n_updates = steps_per_epoch × updates_per_env_step` in the `gradient_updates` loop (mbpo.py). It does **not** change the number of model fits (decided by `model_train_freq` vs `global_step`: 100 per run in every configuration — verified in Section 6.2) nor the number of synthetic-rollout calls (one per epoch = 100); the number of synthetic *transitions* and the fit length can still differ between runs because the policy and the data differ (Section 6.2/6.4 give the measured values per configuration).
- **Width** (`hidden_sizes`): read only by `SACAgent` (actor, two critics, two targets). `model_hidden_sizes` (200×4) and the model batch (256) are unchanged: per-call dynamics FLOPs are identical across the width configurations (200089600 for the member fwd+bwd at HC); the synthetic-sample cost changes through `actor_forward_bs1` only.
- **Batch size** (`batch_size`): the SAC minibatch (`agent.update(mixed_buffer, batch_size)` and `_MixedReplayBuffer.sample`); `model_train_batch_size` stays 256, so the model-fit cost per optimiser step is unchanged; the real/model split changes (13/243 → 26/486 → 51/973).
- **Maximum rollout length** (`rollout_max_length`, Ant only): changes the schedule values (Section 6.1e), the number of synthetic transitions per epoch and per run, the model-buffer capacity (line 175), hence the content of the model buffer that GU samples; it does not change the per-call FLOP constants (`flop_keys.mbpo_rollout_regime` tracks it separately from the architecture signature).

**6.1e Rollout length used in every epoch (0–99) for each regime** — computed with `mbpo._rollout_length` from the logged configuration of each regime; `n_start` = min(10000, warmup + (epoch+1)·1000) start states; upper bound = n_start × L transitions (no early termination). The logged `rollout_length` of all 80 runs equals this schedule: **PASS** — logged rollout_length per epoch equals the schedule in every run


| epoch | L: HC fixed 1 | L: Ant 1→25 | L: Ant max 15 | L: Ant max 1 | n_start | upper bound HC (n_start·L) | upper bound Ant 25 | upper bound Ant 15 | upper bound Ant 1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 1 | 6000 | 6000 | 6000 | 6000 | 6000 |
| 1 | 1 | 1 | 1 | 1 | 7000 | 7000 | 7000 | 7000 | 7000 |
| 2 | 1 | 1 | 1 | 1 | 8000 | 8000 | 8000 | 8000 | 8000 |
| 3 | 1 | 1 | 1 | 1 | 9000 | 9000 | 9000 | 9000 | 9000 |
| 4 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 5 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 6 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 7 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 8 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 9 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 10 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 11 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 12 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 13 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 14 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 15 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 16 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 17 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 18 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 19 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 20 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 21 | 1 | 1 | 1 | 1 | 10000 | 10000 | 10000 | 10000 | 10000 |
| 22 | 1 | 2 | 1 | 1 | 10000 | 10000 | 20000 | 10000 | 10000 |
| 23 | 1 | 2 | 2 | 1 | 10000 | 10000 | 20000 | 20000 | 10000 |
| 24 | 1 | 2 | 2 | 1 | 10000 | 10000 | 20000 | 20000 | 10000 |
| 25 | 1 | 2 | 2 | 1 | 10000 | 10000 | 20000 | 20000 | 10000 |
| 26 | 1 | 3 | 2 | 1 | 10000 | 10000 | 30000 | 20000 | 10000 |
| 27 | 1 | 3 | 2 | 1 | 10000 | 10000 | 30000 | 20000 | 10000 |
| 28 | 1 | 3 | 2 | 1 | 10000 | 10000 | 30000 | 20000 | 10000 |
| 29 | 1 | 4 | 3 | 1 | 10000 | 10000 | 40000 | 30000 | 10000 |
| 30 | 1 | 4 | 3 | 1 | 10000 | 10000 | 40000 | 30000 | 10000 |
| 31 | 1 | 4 | 3 | 1 | 10000 | 10000 | 40000 | 30000 | 10000 |
| 32 | 1 | 5 | 3 | 1 | 10000 | 10000 | 50000 | 30000 | 10000 |
| 33 | 1 | 5 | 3 | 1 | 10000 | 10000 | 50000 | 30000 | 10000 |
| 34 | 1 | 5 | 3 | 1 | 10000 | 10000 | 50000 | 30000 | 10000 |
| 35 | 1 | 6 | 4 | 1 | 10000 | 10000 | 60000 | 40000 | 10000 |
| 36 | 1 | 6 | 4 | 1 | 10000 | 10000 | 60000 | 40000 | 10000 |
| 37 | 1 | 6 | 4 | 1 | 10000 | 10000 | 60000 | 40000 | 10000 |
| 38 | 1 | 6 | 4 | 1 | 10000 | 10000 | 60000 | 40000 | 10000 |
| 39 | 1 | 7 | 4 | 1 | 10000 | 10000 | 70000 | 40000 | 10000 |
| 40 | 1 | 7 | 4 | 1 | 10000 | 10000 | 70000 | 40000 | 10000 |
| 41 | 1 | 7 | 5 | 1 | 10000 | 10000 | 70000 | 50000 | 10000 |
| 42 | 1 | 8 | 5 | 1 | 10000 | 10000 | 80000 | 50000 | 10000 |
| 43 | 1 | 8 | 5 | 1 | 10000 | 10000 | 80000 | 50000 | 10000 |
| 44 | 1 | 8 | 5 | 1 | 10000 | 10000 | 80000 | 50000 | 10000 |
| 45 | 1 | 8 | 5 | 1 | 10000 | 10000 | 80000 | 50000 | 10000 |
| 46 | 1 | 9 | 6 | 1 | 10000 | 10000 | 90000 | 60000 | 10000 |
| 47 | 1 | 9 | 6 | 1 | 10000 | 10000 | 90000 | 60000 | 10000 |
| 48 | 1 | 9 | 6 | 1 | 10000 | 10000 | 90000 | 60000 | 10000 |
| 49 | 1 | 10 | 6 | 1 | 10000 | 10000 | 100000 | 60000 | 10000 |
| 50 | 1 | 10 | 6 | 1 | 10000 | 10000 | 100000 | 60000 | 10000 |
| 51 | 1 | 10 | 6 | 1 | 10000 | 10000 | 100000 | 60000 | 10000 |
| 52 | 1 | 11 | 7 | 1 | 10000 | 10000 | 110000 | 70000 | 10000 |
| 53 | 1 | 11 | 7 | 1 | 10000 | 10000 | 110000 | 70000 | 10000 |
| 54 | 1 | 11 | 7 | 1 | 10000 | 10000 | 110000 | 70000 | 10000 |
| 55 | 1 | 12 | 7 | 1 | 10000 | 10000 | 120000 | 70000 | 10000 |
| 56 | 1 | 12 | 7 | 1 | 10000 | 10000 | 120000 | 70000 | 10000 |
| 57 | 1 | 12 | 7 | 1 | 10000 | 10000 | 120000 | 70000 | 10000 |
| 58 | 1 | 12 | 8 | 1 | 10000 | 10000 | 120000 | 80000 | 10000 |
| 59 | 1 | 13 | 8 | 1 | 10000 | 10000 | 130000 | 80000 | 10000 |
| 60 | 1 | 13 | 8 | 1 | 10000 | 10000 | 130000 | 80000 | 10000 |
| 61 | 1 | 13 | 8 | 1 | 10000 | 10000 | 130000 | 80000 | 10000 |
| 62 | 1 | 14 | 8 | 1 | 10000 | 10000 | 140000 | 80000 | 10000 |
| 63 | 1 | 14 | 9 | 1 | 10000 | 10000 | 140000 | 90000 | 10000 |
| 64 | 1 | 14 | 9 | 1 | 10000 | 10000 | 140000 | 90000 | 10000 |
| 65 | 1 | 14 | 9 | 1 | 10000 | 10000 | 140000 | 90000 | 10000 |
| 66 | 1 | 15 | 9 | 1 | 10000 | 10000 | 150000 | 90000 | 10000 |
| 67 | 1 | 15 | 9 | 1 | 10000 | 10000 | 150000 | 90000 | 10000 |
| 68 | 1 | 15 | 9 | 1 | 10000 | 10000 | 150000 | 90000 | 10000 |
| 69 | 1 | 16 | 10 | 1 | 10000 | 10000 | 160000 | 100000 | 10000 |
| 70 | 1 | 16 | 10 | 1 | 10000 | 10000 | 160000 | 100000 | 10000 |
| 71 | 1 | 16 | 10 | 1 | 10000 | 10000 | 160000 | 100000 | 10000 |
| 72 | 1 | 17 | 10 | 1 | 10000 | 10000 | 170000 | 100000 | 10000 |
| 73 | 1 | 17 | 10 | 1 | 10000 | 10000 | 170000 | 100000 | 10000 |
| 74 | 1 | 17 | 10 | 1 | 10000 | 10000 | 170000 | 100000 | 10000 |
| 75 | 1 | 18 | 11 | 1 | 10000 | 10000 | 180000 | 110000 | 10000 |
| 76 | 1 | 18 | 11 | 1 | 10000 | 10000 | 180000 | 110000 | 10000 |
| 77 | 1 | 18 | 11 | 1 | 10000 | 10000 | 180000 | 110000 | 10000 |
| 78 | 1 | 18 | 11 | 1 | 10000 | 10000 | 180000 | 110000 | 10000 |
| 79 | 1 | 19 | 11 | 1 | 10000 | 10000 | 190000 | 110000 | 10000 |
| 80 | 1 | 19 | 12 | 1 | 10000 | 10000 | 190000 | 120000 | 10000 |
| 81 | 1 | 19 | 12 | 1 | 10000 | 10000 | 190000 | 120000 | 10000 |
| 82 | 1 | 20 | 12 | 1 | 10000 | 10000 | 200000 | 120000 | 10000 |
| 83 | 1 | 20 | 12 | 1 | 10000 | 10000 | 200000 | 120000 | 10000 |
| 84 | 1 | 20 | 12 | 1 | 10000 | 10000 | 200000 | 120000 | 10000 |
| 85 | 1 | 20 | 12 | 1 | 10000 | 10000 | 200000 | 120000 | 10000 |
| 86 | 1 | 21 | 13 | 1 | 10000 | 10000 | 210000 | 130000 | 10000 |
| 87 | 1 | 21 | 13 | 1 | 10000 | 10000 | 210000 | 130000 | 10000 |
| 88 | 1 | 21 | 13 | 1 | 10000 | 10000 | 210000 | 130000 | 10000 |
| 89 | 1 | 22 | 13 | 1 | 10000 | 10000 | 220000 | 130000 | 10000 |
| 90 | 1 | 22 | 13 | 1 | 10000 | 10000 | 220000 | 130000 | 10000 |
| 91 | 1 | 22 | 13 | 1 | 10000 | 10000 | 220000 | 130000 | 10000 |
| 92 | 1 | 23 | 14 | 1 | 10000 | 10000 | 230000 | 140000 | 10000 |
| 93 | 1 | 23 | 14 | 1 | 10000 | 10000 | 230000 | 140000 | 10000 |
| 94 | 1 | 23 | 14 | 1 | 10000 | 10000 | 230000 | 140000 | 10000 |
| 95 | 1 | 24 | 14 | 1 | 10000 | 10000 | 240000 | 140000 | 10000 |
| 96 | 1 | 24 | 14 | 1 | 10000 | 10000 | 240000 | 140000 | 10000 |
| 97 | 1 | 24 | 14 | 1 | 10000 | 10000 | 240000 | 140000 | 10000 |
| 98 | 1 | 24 | 15 | 1 | 10000 | 10000 | 240000 | 150000 | 10000 |
| 99 | 1 | 25 | 15 | 1 | 10000 | 10000 | 250000 | 150000 | 10000 |

Regime parameters (from the logged `algo_config`), model-buffer capacity, Σ of the schedule over the 100 epochs, and the upper bound on transitions per run:


| regime | rollout_min_epoch | rollout_max_epoch | rollout_min_length | rollout_max_length | `mbpo_rollout_regime` key | model buffer capacity | Σ_epochs L | upper bound transitions per run |
|---|---|---|---|---|---|---|---|---|
| HC fixed 1 (HC base: min_epoch 20, max_epoch 150, lengths 1→1) | 20 | 150 | 1 | 1 | rl1-1_re20-150 | 10000 | 100 | 990000 |
| Ant 1→25 (Ant base, mbpo_ant.json) | 20 | 100 | 1 | 25 | rl1-25_re20-100 | 250000 | 1048 | 10470000 |
| Ant max 15 (rollout15) | 20 | 100 | 1 | 15 | rl1-15_re20-100 | 150000 | 653 | 6520000 |
| Ant max 1 (rollout1) | 20 | 100 | 1 | 1 | rl1-1_re20-100 | 10000 | 100 | 990000 |


**6.1f Synthetic transitions generated per run and model-buffer retention** — `synthetic_transitions_generated` is logged per epoch (`training_metrics.json`); the upper bound assumes no early termination (Σ_epochs n_start·L). Explicit early-termination counts are **not logged**; the shortfall relative to the upper bound is the only (indirect) record. Retention: with capacity C and logged per-epoch counts n_e, the number of most recent epochs whose data fit completely into the circular model buffer at the end of each epoch (min / median / max / mean over all epochs and seeds); the logged `model_buffer_size` equals min(C, cumulative generated) at every epoch (column 'size mismatches' counts epochs where it does not). Per-epoch values: `contexts/Section_4.3_data/mbpo_model_per_epoch.csv`.


| config |  | transitions generated per run (mean ± sd) | upper bound (no termination) | generated / upper bound [%] | min | max |
|---|---|---|---|---|---|---|
| HC-base |  | 990000 ± 0 | 990000 | 100.000 ± 0 | 990000 | 990000 |
| HC-utd2 |  | 990000 ± 0 | 990000 | 100.000 ± 0 | 990000 | 990000 |
| HC-utd4 |  | 990000 ± 0 | 990000 | 100.000 ± 0 | 990000 | 990000 |
| HC-w256 |  | 990000 ± 0 | 990000 | 100.000 ± 0 | 990000 | 990000 |
| HC-w512 |  | 990000 ± 0 | 990000 | 100.000 ± 0 | 990000 | 990000 |
| HC-b512 |  | 990000 ± 0 | 990000 | 100.000 ± 0 | 990000 | 990000 |
| HC-b1024 |  | 990000 ± 0 | 990000 | 100.000 ± 0 | 990000 | 990000 |
| Ant-base |  | 9.34276e+06 ± 151142 | 1.04700e+07 | 89.2336 ± 1.44357 | 9.15543e+06 | 9.57817e+06 |
| Ant-utd2 |  | 9.23514e+06 ± 180486 | 1.04700e+07 | 88.2057 ± 1.72384 | 9.06241e+06 | 9.51267e+06 |
| Ant-utd4 |  | 8.96171e+06 ± 367103 | 1.04700e+07 | 85.5941 ± 3.50624 | 8.47863e+06 | 9.30581e+06 |
| Ant-w256 |  | 9.25392e+06 ± 108352 | 1.04700e+07 | 88.3851 ± 1.03488 | 9.12238e+06 | 9.37481e+06 |
| Ant-w512 |  | 9.39758e+06 ± 171633 | 1.04700e+07 | 89.7572 ± 1.63928 | 9.21407e+06 | 9.66042e+06 |
| Ant-b512 |  | 9.33493e+06 ± 257824 | 1.04700e+07 | 89.1588 ± 2.46251 | 9.07449e+06 | 9.67142e+06 |
| Ant-b1024 |  | 9.21347e+06 ± 122888 | 1.04700e+07 | 87.9988 ± 1.17372 | 9.09501e+06 | 9.41366e+06 |
| Ant-rollout1 |  | 990000 ± 0 | 990000 | 100.000 ± 0 | 990000 | 990000 |
| Ant-rollout15 |  | 6.11639e+06 ± 50541.5 | 6.52000e+06 | 93.8097 ± 0.775177 | 6.04659e+06 | 6.18251e+06 |


| config | model buffer capacity | size mismatches (epochs) | retained epochs min | median | max | mean |
|---|---|---|---|---|---|---|
| HC-base | 10000 | 0 | 1 | 1.00000 | 1 | 1.00000 |
| HC-utd2 | 10000 | 0 | 1 | 1.00000 | 1 | 1.00000 |
| HC-utd4 | 10000 | 0 | 1 | 1.00000 | 1 | 1.00000 |
| HC-w256 | 10000 | 0 | 1 | 1.00000 | 1 | 1.00000 |
| HC-w512 | 10000 | 0 | 1 | 1.00000 | 1 | 1.00000 |
| HC-b512 | 10000 | 0 | 1 | 1.00000 | 1 | 1.00000 |
| HC-b1024 | 10000 | 0 | 1 | 1.00000 | 1 | 1.00000 |
| Ant-base | 250000 | 0 | 1 | 2.00000 | 24 | 5.55000 |
| Ant-utd2 | 250000 | 0 | 1 | 2.00000 | 24 | 5.54600 |
| Ant-utd4 | 250000 | 0 | 1 | 2.00000 | 24 | 5.58000 |
| Ant-w256 | 250000 | 0 | 1 | 2.00000 | 24 | 5.54200 |
| Ant-w512 | 250000 | 0 | 1 | 2.00000 | 24 | 5.52600 |
| Ant-b512 | 250000 | 0 | 1 | 2.00000 | 24 | 5.54000 |
| Ant-b1024 | 250000 | 0 | 1 | 2.00000 | 24 | 5.53200 |
| Ant-rollout1 | 10000 | 0 | 1 | 1.00000 | 1 | 1.00000 |
| Ant-rollout15 | 150000 | 0 | 1 | 2.00000 | 16 | 4.48200 |


**6.2 FLOP definitions and the model-fit statistics** — formulas verbatim in Section 2.7. Per configuration, from the logged epoch rows of every run (`mbpo_extras`):


| config | dyn FLOPs min over seeds | mean | max | sd | holdout-forward share of dyn FLOPs [%] (mean ± sd) | train FLOPs (mean ± sd) | holdout FLOPs (mean ± sd) |
|---|---|---|---|---|---|---|---|
| HC-base | 2.42559e+14 | 2.51804e+14 | 2.60247e+14 | 7.63602e+12 | 7.77581 ± 9.93014e-16 | 2.32224e+14 ± 7.04225e+12 | 1.95798e+13 ± 5.93762e+11 |
| HC-utd2 | 2.39825e+14 | 2.62168e+14 | 2.77646e+14 | 1.58903e+13 | 7.77581 ± 9.93014e-16 | 2.41782e+14 ± 1.46547e+13 | 2.03857e+13 ± 1.23560e+12 |
| HC-utd4 | 2.23010e+14 | 2.46235e+14 | 2.61206e+14 | 1.71987e+13 | 7.77581 ± 9.93014e-16 | 2.27088e+14 ± 1.58614e+13 | 1.91468e+13 ± 1.33734e+12 |
| HC-w256 | 2.33579e+14 | 2.46604e+14 | 2.54253e+14 | 7.72595e+12 | 7.77581 ± 9.93014e-16 | 2.27429e+14 ± 7.12519e+12 | 1.91755e+13 ± 6.00755e+11 |
| HC-w512 | 2.37585e+14 | 2.51142e+14 | 2.60935e+14 | 1.17876e+13 | 7.77581 ± 9.93014e-16 | 2.31614e+14 ± 1.08710e+13 | 1.95283e+13 ± 9.16582e+11 |
| HC-b512 | 2.30233e+14 | 2.42018e+14 | 2.51068e+14 | 8.09962e+12 | 7.77581 ± 9.93014e-16 | 2.23199e+14 ± 7.46981e+12 | 1.88188e+13 ± 6.29811e+11 |
| HC-b1024 | 2.28985e+14 | 2.41654e+14 | 2.51130e+14 | 8.87449e+12 | 7.77581 ± 9.93014e-16 | 2.22863e+14 ± 8.18442e+12 | 1.87906e+13 ± 6.90063e+11 |
| Ant-base | 3.81730e+14 | 4.30949e+14 | 4.84342e+14 | 4.61055e+13 | 7.99274 ± 0 | 3.96504e+14 ± 4.24204e+13 | 3.44446e+13 ± 3.68509e+12 |
| Ant-utd2 | 3.93487e+14 | 4.56382e+14 | 4.88853e+14 | 3.81789e+13 | 7.99274 ± 0 | 4.19905e+14 ± 3.51274e+13 | 3.64775e+13 ± 3.05154e+12 |
| Ant-utd4 | 3.97453e+14 | 4.38714e+14 | 4.95042e+14 | 3.71827e+13 | 7.99274 ± 0 | 4.03649e+14 ± 3.42107e+13 | 3.50653e+13 ± 2.97191e+12 |
| Ant-w256 | 4.20149e+14 | 4.44206e+14 | 4.80078e+14 | 2.46935e+13 | 7.99274 ± 0 | 4.08702e+14 ± 2.27198e+13 | 3.55042e+13 ± 1.97369e+12 |
| Ant-w512 | 3.60045e+14 | 4.31938e+14 | 4.76487e+14 | 4.36413e+13 | 7.99274 ± 0 | 3.97414e+14 ± 4.01532e+13 | 3.45237e+13 ± 3.48814e+12 |
| Ant-b512 | 3.63804e+14 | 4.15041e+14 | 4.84264e+14 | 5.33517e+13 | 7.99274 ± 0 | 3.81868e+14 ± 4.90875e+13 | 3.31731e+13 ± 4.26426e+12 |
| Ant-b1024 | 3.98211e+14 | 4.39178e+14 | 4.80791e+14 | 3.46877e+13 | 7.99274 ± 0 | 4.04076e+14 ± 3.19152e+13 | 3.51024e+13 ± 2.77250e+12 |
| Ant-rollout1 | 4.10804e+14 | 4.28258e+14 | 4.46163e+14 | 1.30931e+13 | 7.99274 ± 0 | 3.94028e+14 ± 1.20466e+13 | 3.42295e+13 ± 1.04649e+12 |
| Ant-rollout15 | 4.08470e+14 | 4.36857e+14 | 4.66053e+14 | 2.71509e+13 | 7.99274 ± 0 | 4.01940e+14 ± 2.49808e+13 | 3.49168e+13 ± 2.17010e+12 |

The layout documents say the holdout forward is 'about 8 %' of the fit FLOPs; the actual values are 7.77581 % (HC) and 7.99274 % (Ant) in every run of every configuration (column above, sd ≈ 0). This share is a constant by construction: both terms scale linearly with epochs_run × n_total (n_holdout = 0.2·n and n_train = 0.8·n for n a multiple of 1000), so it depends only on the per-sample constants (7 × dynamics_member_fwdbwd/256 for training, dynamics_ensemble_forward_all_bs1 for the holdout forward), i.e. only on the environment's obs/act dimensions.


| config | fits per run (min–max) | fit epochs per fit, mean over fits [epochs] | first fit (epoch 0) epochs | later fits (epochs 1–99), mean epochs | optimiser steps per run | steps of the first fit | steps per later fit (mean) | longest fit of a run [epochs] |
|---|---|---|---|---|---|---|---|---|
| HC-base | 100–100 | 11.6840 ± 0.545096 | 186.000 ± 16.3707 | 9.92323 ± 0.473057 | 166294 ± 5045.78 | 2976.00 ± 261.931 | 1649.68 ± 51.5616 | 186.000 ± 16.3707 |
| HC-utd2 | 100–100 | 11.8700 ± 0.603780 | 186.000 ± 16.3707 | 10.1111 ± 0.531192 | 173106 ± 10466.4 | 2976.00 ± 261.931 | 1718.48 ± 104.348 | 186.000 ± 16.3707 |
| HC-utd4 | 100–100 | 11.4640 ± 0.427235 | 186.000 ± 16.3707 | 9.70101 ± 0.546510 | 162606 ± 11358.1 | 2976.00 ± 261.931 | 1612.43 ± 116.490 | 186.000 ± 16.3707 |
| HC-w256 | 100–100 | 11.5000 ± 0.393891 | 161.200 ± 42.9150 | 9.98788 ± 0.546081 | 162846 ± 5102.68 | 2579.20 ± 686.641 | 1618.86 ± 53.4536 | 168.000 ± 29.6564 |
| HC-w512 | 100–100 | 11.6460 ± 0.688353 | 174.400 ± 26.9128 | 10.0020 ± 0.612899 | 165829 ± 7777.63 | 2790.40 ± 430.605 | 1646.85 ± 76.3436 | 174.400 ± 26.9128 |
| HC-b512 | 100–100 | 11.2980 ± 0.513293 | 186.000 ± 16.3707 | 9.53333 ± 0.390519 | 159823 ± 5344.71 | 2976.00 ± 261.931 | 1584.31 ± 51.5571 | 186.000 ± 16.3707 |
| HC-b1024 | 100–100 | 11.4800 ± 0.638944 | 186.000 ± 16.3707 | 9.71717 ± 0.569118 | 159605 ± 5865.59 | 2976.00 ± 261.931 | 1582.11 ± 57.0816 | 186.000 ± 16.3707 |
| Ant-base | 100–100 | 14.6520 ± 1.00870 | 29.0000 ± 13.2098 | 14.5071 ± 1.04455 | 208429 ± 22282.1 | 464.000 ± 211.358 | 2100.65 ± 226.229 | 39.8000 ± 6.01664 |
| Ant-utd2 | 100–100 | 14.7820 ± 0.699979 | 29.0000 ± 13.2098 | 14.6384 ± 0.804867 | 220701 ± 18446.8 | 464.000 ± 211.358 | 2224.62 ± 188.174 | 39.4000 ± 5.12835 |
| Ant-utd4 | 100–100 | 14.2580 ± 0.967714 | 29.0000 ± 13.2098 | 14.1091 ± 1.04728 | 212141 ± 17958.2 | 464.000 ± 211.358 | 2138.15 ± 182.657 | 38.2000 ± 4.08656 |
| Ant-w256 | 100–100 | 14.5860 ± 0.546013 | 25.2000 ± 7.36206 | 14.4788 ± 0.528187 | 214821 ± 11922.8 | 403.200 ± 117.793 | 2165.84 ± 120.681 | 33.4000 ± 5.54977 |
| Ant-w512 | 100–100 | 14.7720 ± 0.958316 | 28.8000 ± 7.59605 | 14.6303 ± 1.01661 | 208889 ± 21081.6 | 460.800 ± 121.537 | 2105.34 ± 213.896 | 38.8000 ± 5.76194 |
| Ant-b512 | 100–100 | 14.1380 ± 1.22976 | 29.0000 ± 13.2098 | 13.9879 ± 1.31508 | 200737 ± 25776.0 | 464.000 ± 211.358 | 2022.96 ± 261.793 | 44.0000 ± 10.1242 |
| Ant-b1024 | 100–100 | 14.4980 ± 0.753870 | 29.0000 ± 13.2098 | 14.3515 ± 0.831338 | 212390 ± 16757.4 | 464.000 ± 211.358 | 2140.66 ± 170.318 | 37.6000 ± 4.66905 |
| Ant-rollout1 | 100–100 | 14.9840 ± 0.739953 | 29.0000 ± 13.2098 | 14.8424 ± 0.632326 | 207135 ± 6320.80 | 464.000 ± 211.358 | 2087.58 ± 63.3000 | 38.8000 ± 4.81664 |
| Ant-rollout15 | 100–100 | 14.8800 ± 0.780128 | 29.0000 ± 13.2098 | 14.7374 ± 0.766513 | 211290 ± 13112.4 | 464.000 ± 211.358 | 2129.56 ± 133.356 | 40.2000 ± 4.08656 |

Synthetic rollouts: per-sample FLOPs = `actor_forward_bs1` + `dynamics_ensemble_forward_all_bs1` — **independent of the rollout length L** (the cost of a step per sample is constant); the number of samples is Σ_epochs generated transitions (≈ n_start·L when nothing terminates):


| config | samples per run (mean ± sd) | actor_forward_bs1 | ensemble_forward_all_bs1 | per-sample FLOPs | synthetic FLOPs per run | TOTAL FLOPs per run |
|---|---|---|---|---|---|---|
| HC-base | 990000 ± 0 | 2.15654e+06 | 1.84520e+06 | 4.00174e+06 | 3.96173e+12 ± 0 | 1.13285e+15 ± 7.63602e+12 |
| HC-utd2 | 990000 ± 0 | 2.15654e+06 | 1.84520e+06 | 4.00174e+06 | 3.96173e+12 ± 0 | 2.02009e+15 ± 1.58903e+13 |
| HC-utd4 | 990000 ± 0 | 2.15654e+06 | 1.84520e+06 | 4.00174e+06 | 3.96173e+12 ± 0 | 3.75790e+15 ± 1.71987e+13 |
| HC-w256 | 990000 ± 0 | 145920 | 1.84520e+06 | 1.99112e+06 | 1.97121e+12 ± 0 | 3.06747e+14 ± 7.72595e+12 |
| HC-w512 | 990000 ± 0 | 553984 | 1.84520e+06 | 2.39918e+06 | 2.37519e+12 ± 0 | 4.77260e+14 ± 1.17876e+13 |
| HC-b512 | 990000 ± 0 | 2.15654e+06 | 1.84520e+06 | 4.00174e+06 | 3.96173e+12 ± 0 | 1.99994e+15 ± 8.09962e+12 |
| HC-b1024 | 990000 ± 0 | 2.15654e+06 | 1.84520e+06 | 4.00174e+06 | 3.96173e+12 ± 0 | 3.75332e+15 ± 8.87449e+12 |
| Ant-base | 9.34276e+06 ± 151142 | 2.34496e+06 | 2.59000e+06 | 4.93496e+06 | 4.61061e+13 ± 7.45880e+11 | 1.41603e+15 ± 4.55542e+13 |
| Ant-utd2 | 9.23514e+06 ± 180486 | 2.34496e+06 | 2.59000e+06 | 4.93496e+06 | 4.55750e+13 ± 8.90693e+11 | 2.37967e+15 ± 3.76188e+13 |
| Ant-utd4 | 8.96171e+06 ± 367103 | 2.34496e+06 | 2.59000e+06 | 4.93496e+06 | 4.42257e+13 ± 1.81164e+12 | 4.23812e+15 ± 3.64659e+13 |
| Ant-w256 | 9.25392e+06 ± 108352 | 193024 | 2.59000e+06 | 2.78302e+06 | 2.57539e+13 ± 3.01545e+11 | 5.43602e+14 ± 2.46245e+13 |
| Ant-w512 | 9.39758e+06 ± 171633 | 648192 | 2.59000e+06 | 3.23819e+06 | 3.04312e+13 ± 5.55779e+11 | 7.17054e+14 ± 4.32198e+13 |
| Ant-b512 | 9.33493e+06 ± 257824 | 2.34496e+06 | 2.59000e+06 | 4.93496e+06 | 4.60675e+13 ± 1.27235e+12 | 2.33882e+15 ± 5.25551e+13 |
| Ant-b1024 | 9.21347e+06 ± 122888 | 2.34496e+06 | 2.59000e+06 | 4.93496e+06 | 4.54681e+13 ± 6.06449e+11 | 4.23983e+15 ± 3.45975e+13 |
| Ant-rollout1 | 990000 ± 0 | 2.34496e+06 | 2.59000e+06 | 4.93496e+06 | 4.88561e+12 ± 0 | 1.37212e+15 ± 1.30931e+13 |
| Ant-rollout15 | 6.11639e+06 ± 50541.5 | 2.34496e+06 | 2.59000e+06 | 4.93496e+06 | 3.01841e+13 ± 2.49420e+11 | 1.40601e+15 ± 2.70407e+13 |

- **PASS** — every run has exactly 100 fits (a non-null `model_train_epochs` in all 100 epochs)

J/FLOP of the two MBPO-specific segments and of TOTAL, **ratio of means ‖ mean of per-seed ratios (sd of per-seed ratios)** (FLOPs differ between seeds, so the two statistics differ slightly):


| config | dynamics_model_update [J/FLOP] | synthetic_rollout_generation [J/FLOP] | TOTAL [J/FLOP] |
|---|---|---|---|
| HC-base | 5.80582e-10 ‖ 5.80351e-10 (sd 1.48182e-11) | 1.90319e-11 ‖ 1.90319e-11 (sd 4.13199e-12) | 1.83073e-10 ‖ 1.83047e-10 (sd 5.44247e-12) |
| HC-utd2 | 5.72307e-10 ‖ 5.72783e-10 (sd 1.16405e-11) | 1.83033e-11 ‖ 1.83033e-11 (sd 2.87559e-12) | 1.33089e-10 ‖ 1.33075e-10 (sd 2.49761e-12) |
| HC-utd4 | 5.65801e-10 ‖ 5.66061e-10 (sd 6.41063e-12) | 1.88799e-11 ‖ 1.88799e-11 (sd 3.21221e-12) | 9.94189e-11 ‖ 9.94123e-11 (sd 1.82537e-12) |
| HC-w256 | 5.71756e-10 ‖ 5.71836e-10 (sd 1.14951e-11) | 2.69983e-11 ‖ 2.69983e-11 (sd 6.04855e-12) | 6.12601e-10 ‖ 6.12647e-10 (sd 9.02267e-12) |
| HC-w512 | 5.71704e-10 ‖ 5.71310e-10 (sd 1.09539e-11) | 2.76712e-11 ‖ 2.76712e-11 (sd 6.43923e-12) | 4.03908e-10 ‖ 4.03741e-10 (sd 8.70683e-12) |
| HC-b512 | 5.67242e-10 ‖ 5.67267e-10 (sd 9.73126e-12) | 1.68016e-11 ‖ 1.68016e-11 (sd 2.11153e-12) | 1.04554e-10 ‖ 1.04548e-10 (sd 2.07658e-12) |
| HC-b1024 | 5.71753e-10 ‖ 5.71786e-10 (sd 2.70198e-12) | 2.03709e-11 ‖ 2.03709e-11 (sd 5.50422e-12) | 5.82597e-11 ‖ 5.82574e-11 (sd 1.19887e-12) |
| Ant-base | 4.37090e-10 ‖ 4.37061e-10 (sd 3.90010e-12) | 1.59556e-11 ‖ 1.59503e-11 (sd 5.98598e-13) | 1.78766e-10 ‖ 1.78551e-10 (sd 8.47149e-12) |
| Ant-utd2 | 4.23173e-10 ‖ 4.22810e-10 (sd 7.68529e-12) | 1.64619e-11 ‖ 1.64575e-11 (sd 7.37165e-13) | 1.33126e-10 ‖ 1.33061e-10 (sd 5.19325e-12) |
| Ant-utd4 | 4.18732e-10 ‖ 4.18582e-10 (sd 4.45755e-12) | 1.61044e-11 ‖ 1.61019e-11 (sd 6.20996e-13) | 1.00151e-10 ‖ 1.00131e-10 (sd 3.00310e-12) |
| Ant-w256 | 4.16567e-10 ‖ 4.16610e-10 (sd 3.40175e-12) | 2.20377e-11 ‖ 2.20395e-11 (sd 1.09363e-12) | 4.31364e-10 ‖ 4.31419e-10 (sd 2.81789e-12) |
| Ant-w512 | 4.15747e-10 ‖ 4.15680e-10 (sd 1.02489e-11) | 2.11004e-11 ‖ 2.10931e-11 (sd 5.22945e-13) | 3.23041e-10 ‖ 3.22739e-10 (sd 8.56113e-12) |
| Ant-b512 | 4.11571e-10 ‖ 4.11801e-10 (sd 1.00622e-11) | 1.57138e-11 ‖ 1.57038e-11 (sd 5.55224e-13) | 1.05777e-10 ‖ 1.05660e-10 (sd 6.70888e-12) |
| Ant-b1024 | 4.05703e-10 ‖ 4.05549e-10 (sd 5.80026e-12) | 1.59088e-11 ‖ 1.59145e-11 (sd 7.19199e-13) | 6.31381e-11 ‖ 6.31184e-11 (sd 3.05899e-12) |
| Ant-rollout1 | 4.12010e-10 ‖ 4.12176e-10 (sd 9.43497e-12) | 2.49589e-11 ‖ 2.49589e-11 (sd 3.75765e-12) | 1.75458e-10 ‖ 1.75458e-10 (sd 1.64199e-12) |
| Ant-rollout15 | 4.15198e-10 ‖ 4.15068e-10 (sd 6.68425e-12) | 1.48966e-11 ‖ 1.49001e-11 (sd 1.38439e-12) | 1.74865e-10 ‖ 1.74782e-10 (sd 5.75210e-12) |


**6.3 Segment tables.** The common blocks of Sections 3–4 contain all seven segments in every table: `rollout`, `dynamics_model_update`, `synthetic_rollout_generation` (measured, one CodeCarbon task per epoch), `gradient_updates` (measured) with its four allocated sub-segments `buffer_sample`, `critic_update`, `actor_update`, `target_update`; shares of TOTAL; J/FLOP of the model fit and of the synthetic rollouts (both ratio-of-means and mean-of-ratios, Section 6.2); the hardware composition (CPU/GPU/RAM) and the count of tasks shorter than 1 s of each algorithm-specific task are in the 'Hardware composition' tables (one per measured task).

**6.4 Rollout-length sweep (Ant-v5 only)** — the common block for lengths 1, 15 and 25 is Section 4.4. In addition, per configuration (Ant-v5), mean ± sd over seeds:


| config | synthetic-rollout energy [J] | model-fit energy [J] | GU energy [J] | rollout energy [J] | TOTAL energy [J] | transitions generated per run | upper bound (no termination) | shortfall vs upper bound [%] (derived; early terminations) | synthetic-rollout duration [s] | model-fit duration [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| L_max = 1 (rollout1) | 121.939 ± 18.3584 | 176446 ± 3727.50 | 59929.1 ± 587.330 | 4251.90 ± 38.9765 | 240749 ± 3280.97 | 990000 ± 0 | 990000 | 0 ± 0 | 0.934079 ± 0.00164873 | 1489.22 ± 38.9251 |
| L_max = 15 (rollout15) | 449.641 ± 40.6506 | 181382 ± 12746.4 | 59817.6 ± 200.807 | 4213.11 ± 27.4735 | 245862 ± 12655.9 | 6.11639e+06 ± 50541.5 | 6.52000e+06 | 6.19033 ± 0.775177 | 3.44790 ± 0.0185424 | 1535.37 ± 118.896 |
| L_max = 25 (base) | 735.651 ± 37.3179 | 188363 ± 20295.5 | 59746.5 ± 208.801 | 4292.41 ± 29.8641 | 253138 ± 20140.0 | 9.34276e+06 ± 151142 | 1.04700e+07 | 10.7664 ± 1.44357 | 5.07582 ± 0.0749797 | 1605.17 ± 180.136 |

Early-termination information that is logged: **none explicitly** (see Section 6.1f: NOT AVAILABLE as a count); the derived shortfall above is the fraction of the upper-bound transitions that were not generated because states terminated (Ant `_ant_done`), HC being exactly 0 (check below).
- **PASS** — HalfCheetah (never terminates, L = 1): generated transitions equal Σ n_start·1 exactly in all 35 HC runs (max |difference| = 0)

**6.5 Environment confound — complete diff of the logged configurations of the HC and the Ant baseline** (`metadata.json`, run seed 331 of each; every field that differs):


| field | HalfCheetah-v5 baseline | Ant-v5 baseline |
|---|---|---|
| `algo_config.rollout_max_epoch` | 150 | 100 |
| `algo_config.rollout_max_length` | 1 | 25 |
| `experiment_config.env_id` | "HalfCheetah-v5" | "Ant-v5" |

Other (non-config) differences that are properties of the environments: observation/action dims 17/6 vs 105/8 (Section 2.3), termination function `_never_done` vs `_ant_done` (6.1c), episodes (HC 1000-step truncation only; Ant terminates early, Section 5.3). The rollout-length schedule per epoch of both baselines is the table in 6.1e (HC: constant 1; Ant: 1 for epochs ≤ 20, then linear to 25 at epoch 100).

**6.6 MBPO versus SAC side by side** at matched settings — all 7 configurations that exist for both algorithms (`base` = hidden (1024,1024), batch 256, UTD 1; `utd2`, `utd4`, `w256`, `w512`, `b512`, `b1024`). Columns: HC MBPO, HC SAC, Ant MBPO, Ant SAC. SAC values come from the same recomputation method as Section 4.1 (`s4_data.Dataset(sac_spec())`); cross-check against `contexts/section_4.1_data/sac_cross_seed_summary.csv`. Batch composition: SAC samples 256/512/1024 transitions all from the real buffer; MBPO samples 13 real + 243 model (batch 256), 26 + 486 (512), 51 + 973 (1024) via two host-side samples and a NumPy concatenate (6.1d). The policy-optimisation code path (`SACAgent.update`) is the same object code in both.

- **PASS** — SAC energies recomputed here equal `sac_cross_seed_summary.csv` of Section 4.1 for all 7 configurations and both environments (max relative deviation 2.30e-16)

GU (`gradient_updates`, measured) energy [J]:


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 58698.3 ± 211.888 | 58415.3 ± 247.829 | 59746.5 ± 208.801 | 60050.5 ± 545.039 |
| utd2 | 116328 ± 1048.96 | 115511 ± 621.508 | 118703 ± 1584.99 | 118874 ± 666.312 |
| utd4 | 231830 ± 629.988 | 232406 ± 1075.31 | 235859 ± 735.086 | 237033 ± 1031.31 |
| w256 | 44550.1 ± 455.190 | 45129.5 ± 458.178 | 44807.7 ± 176.270 | 45884.5 ± 711.856 |
| w512 | 46788.7 ± 629.312 | 47227.2 ± 515.401 | 47310.4 ± 472.228 | 48045.6 ± 495.279 |
| b512 | 69369.1 ± 359.157 | 70377.0 ± 547.231 | 71737.0 ± 397.916 | 72131.7 ± 374.977 |
| b1024 | 78057.3 ± 275.017 | 79617.8 ± 577.494 | 84617.4 ± 317.159 | 84516.4 ± 417.067 |


`buffer_sample` (allocated) energy [J]:


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 1677.11 ± 15.2915 | 1550.17 ± 20.9057 | 2271.09 ± 12.1950 | 1922.92 ± 50.6812 |
| utd2 | 3322.85 ± 32.5120 | 3047.48 ± 32.1967 | 4463.52 ± 13.6184 | 3829.55 ± 39.3866 |
| utd4 | 6596.08 ± 28.2525 | 6072.19 ± 41.7613 | 8867.02 ± 31.6077 | 7616.53 ± 54.9489 |
| w256 | 1272.69 ± 4.42470 | 1189.42 ± 9.14672 | 1716.03 ± 7.46297 | 1457.26 ± 17.1636 |
| w512 | 1334.08 ± 12.5366 | 1244.09 ± 5.17093 | 1796.71 ± 8.32313 | 1544.62 ± 10.3502 |
| b512 | 2187.14 ± 2.92658 | 2103.17 ± 17.7266 | 3526.11 ± 16.5795 | 2887.34 ± 15.0776 |
| b1024 | 2939.44 ± 14.1433 | 2959.74 ± 28.5778 | 6011.27 ± 34.1464 | 4863.71 ± 41.1285 |


`critic_update` (allocated) energy [J]:


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 24588.5 ± 267.376 | 24575.1 ± 234.915 | 24560.0 ± 218.652 | 25133.1 ± 295.659 |
| utd2 | 48842.7 ± 678.227 | 48486.5 ± 414.609 | 49339.3 ± 1185.47 | 49256.6 ± 437.929 |
| utd4 | 97613.2 ± 613.239 | 97913.7 ± 628.512 | 98000.7 ± 552.597 | 98307.1 ± 459.369 |
| w256 | 18676.7 ± 324.941 | 18973.9 ± 277.584 | 18648.6 ± 145.104 | 19303.8 ± 405.723 |
| w512 | 19544.5 ± 433.973 | 19809.2 ± 447.308 | 19639.3 ± 333.039 | 20056.2 ± 384.348 |
| b512 | 29118.0 ± 372.208 | 29859.7 ± 660.459 | 29700.8 ± 489.708 | 30236.2 ± 363.913 |
| b1024 | 32497.7 ± 248.337 | 33302.3 ± 502.557 | 34581.0 ± 301.119 | 34806.0 ± 446.046 |


`actor_update` (allocated) energy [J]:


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 29882.6 ± 104.818 | 29743.4 ± 127.124 | 30340.0 ± 62.3096 | 30436.9 ± 252.224 |
| utd2 | 59095.0 ± 425.948 | 58953.3 ± 230.665 | 59844.1 ± 404.340 | 60663.7 ± 240.320 |
| utd4 | 117546 ± 313.899 | 118342 ± 561.913 | 118862 ± 220.469 | 120878 ± 478.788 |
| w256 | 22642.1 ± 134.589 | 22990.5 ± 176.796 | 22500.8 ± 47.8740 | 23127.4 ± 286.451 |
| w512 | 23843.6 ± 200.818 | 24101.1 ± 85.6765 | 23816.9 ± 138.203 | 24347.6 ± 144.137 |
| b512 | 35124.4 ± 68.2198 | 35466.7 ± 106.481 | 35554.4 ± 92.3858 | 36017.7 ± 39.8942 |
| b1024 | 39364.2 ± 64.5743 | 40035.2 ± 167.770 | 40674.7 ± 95.8101 | 41446.8 ± 148.180 |


`target_update` (allocated) energy [J]:


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 2550.10 ± 26.8657 | 2546.63 ± 28.8006 | 2575.51 ± 25.8022 | 2557.54 ± 32.2932 |
| utd2 | 5067.36 ± 50.8363 | 5023.68 ± 24.7458 | 5056.08 ± 20.2017 | 5124.01 ± 42.0890 |
| utd4 | 10075.3 ± 50.6789 | 10078.8 ± 72.5813 | 10129.2 ± 69.0324 | 10231.6 ± 73.6719 |
| w256 | 1958.56 ± 17.9772 | 1975.74 ± 10.6139 | 1942.26 ± 9.85334 | 1996.04 ± 22.5048 |
| w512 | 2066.58 ± 14.8405 | 2072.88 ± 11.8660 | 2057.41 ± 18.7211 | 2097.22 ± 22.0129 |
| b512 | 2939.47 ± 24.7621 | 2947.39 ± 45.1243 | 2955.71 ± 37.6457 | 2990.36 ± 20.6052 |
| b1024 | 3255.96 ± 22.9396 | 3320.52 ± 28.8662 | 3350.50 ± 12.8938 | 3399.91 ± 27.3330 |


`buffer_sample` share of GU [%] (wall-clock time share):


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 2.85721 ± 0.0304969 | 2.65368 ± 0.0310124 | 3.80128 ± 0.0300127 | 3.20200 ± 0.0702272 |
| utd2 | 2.85653 ± 0.0260464 | 2.63823 ± 0.0194719 | 3.76088 ± 0.0597819 | 3.22151 ± 0.0256823 |
| utd4 | 2.84525 ± 0.0179414 | 2.61276 ± 0.0148824 | 3.75947 ± 0.0125601 | 3.21325 ± 0.0115640 |
| w256 | 2.85693 ± 0.0216515 | 2.63566 ± 0.0167152 | 3.82981 ± 0.0215263 | 3.17611 ± 0.0195197 |
| w512 | 2.85146 ± 0.0214129 | 2.63450 ± 0.0299023 | 3.79794 ± 0.0331508 | 3.21505 ± 0.0250598 |
| b512 | 3.15297 ± 0.0182210 | 2.98866 ± 0.0424991 | 4.91551 ± 0.0446789 | 4.00298 ± 0.0332206 |
| b1024 | 3.76577 ± 0.0189786 | 3.71774 ± 0.0588210 | 7.10423 ± 0.0627982 | 5.75501 ± 0.0733368 |


`critic_update` share of GU [%] (wall-clock time share):


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 41.8889 ± 0.334398 | 42.0692 ± 0.290066 | 41.1064 ± 0.243247 | 41.8530 ± 0.244460 |
| utd2 | 41.9860 ± 0.274909 | 41.9752 ± 0.166738 | 41.5607 ± 0.442828 | 41.4355 ± 0.179234 |
| utd4 | 42.1052 ± 0.181393 | 42.1302 ± 0.121756 | 41.5504 ± 0.122593 | 41.4739 ± 0.0505889 |
| w256 | 41.9205 ± 0.323561 | 42.0417 ± 0.210031 | 41.6187 ± 0.165207 | 42.0677 ± 0.272935 |
| w512 | 41.7680 ± 0.385646 | 41.9401 ± 0.493904 | 41.5094 ± 0.309905 | 41.7413 ± 0.401730 |
| b512 | 41.9743 ± 0.327665 | 42.4245 ± 0.617453 | 41.4004 ± 0.462083 | 41.9169 ± 0.287622 |
| b1024 | 41.6327 ± 0.183816 | 41.8260 ± 0.354070 | 40.8669 ± 0.213059 | 41.1814 ± 0.346414 |


`actor_update` share of GU [%] (wall-clock time share):


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 50.9094 ± 0.258129 | 50.9176 ± 0.244273 | 50.7815 ± 0.178929 | 50.6860 ± 0.169870 |
| utd2 | 50.8013 ± 0.214416 | 51.0374 ± 0.148283 | 50.4186 ± 0.337549 | 51.0325 ± 0.146743 |
| utd4 | 50.7036 ± 0.158352 | 50.9202 ± 0.0909592 | 50.3956 ± 0.107614 | 50.9964 ± 0.0431630 |
| w256 | 50.8261 ± 0.280321 | 50.9445 ± 0.177325 | 50.2167 ± 0.125178 | 50.4057 ± 0.230517 |
| w512 | 50.9633 ± 0.339629 | 51.0356 ± 0.396922 | 50.3438 ± 0.267263 | 50.6782 ± 0.327084 |
| b512 | 50.6352 ± 0.266799 | 50.3983 ± 0.484513 | 49.5636 ± 0.351995 | 49.9342 ± 0.210098 |
| b1024 | 50.4303 ± 0.139464 | 50.2855 ± 0.265318 | 48.0692 ± 0.134337 | 49.0407 ± 0.250921 |


`target_update` share of GU [%] (wall-clock time share):


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 4.34452 ± 0.0541846 | 4.35952 ± 0.0456455 | 4.31077 ± 0.0449133 | 4.25899 ± 0.0388861 |
| utd2 | 4.35621 ± 0.0402669 | 4.34915 ± 0.0232287 | 4.25992 ± 0.0467816 | 4.31047 ± 0.0269458 |
| utd4 | 4.34595 ± 0.0159772 | 4.33685 ± 0.0462781 | 4.29461 ± 0.0244214 | 4.31649 ± 0.0166091 |
| w256 | 4.39646 ± 0.0330796 | 4.37816 ± 0.0314262 | 4.33476 ± 0.0358358 | 4.35046 ± 0.0367607 |
| w512 | 4.41726 ± 0.0479369 | 4.38976 ± 0.0694297 | 4.34881 ± 0.0207939 | 4.36542 ± 0.0633601 |
| b512 | 4.23759 ± 0.0495749 | 4.18853 ± 0.0929839 | 4.12047 ± 0.0712722 | 4.14589 ± 0.0487203 |
| b1024 | 4.17131 ± 0.0366514 | 4.17075 ± 0.0475172 | 3.95967 ± 0.0287309 | 4.02292 ± 0.0462880 |


`buffer_sample` share of TOTAL energy [%]:


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 0.809395 ± 0.0267401 | 2.54494 ± 0.0288351 | 0.901530 ± 0.0684953 | 2.98653 ± 0.0643192 |
| utd2 | 1.23666 ± 0.0363841 | 2.58205 ± 0.0193571 | 1.41231 ± 0.0775139 | 3.10884 ± 0.0246671 |
| utd4 | 1.76632 ± 0.0445355 | 2.58437 ± 0.0147005 | 2.09144 ± 0.0776815 | 3.15608 ± 0.0116322 |
| w256 | 0.677617 ± 0.0167570 | 2.50898 ± 0.0147231 | 0.732874 ± 0.0316144 | 2.91550 ± 0.0157454 |
| w512 | 0.693380 ± 0.0364439 | 2.51078 ± 0.0270139 | 0.780150 ± 0.0704630 | 2.95916 ± 0.0214503 |
| b512 | 1.04642 ± 0.0240806 | 2.88029 ± 0.0390895 | 1.43361 ± 0.121284 | 3.77446 ± 0.0316711 |
| b1024 | 1.34473 ± 0.0266339 | 3.59370 ± 0.0569057 | 2.25097 ± 0.119905 | 5.46154 ± 0.0676578 |


`buffer_sample` perf_counter time [s]:


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 11.2917 ± 0.0973144 | 10.4499 ± 0.142907 | 15.1580 ± 0.0311855 | 12.7700 ± 0.266712 |
| utd2 | 22.6977 ± 0.251504 | 20.6728 ± 0.184581 | 30.4700 ± 0.112733 | 25.5485 ± 0.316763 |
| utd4 | 45.2561 ± 0.0980753 | 41.2173 ± 0.217247 | 60.7751 ± 0.113436 | 50.9986 ± 0.418168 |
| w256 | 11.1647 ± 0.0203290 | 10.1796 ± 0.0613197 | 15.0933 ± 0.0638435 | 12.4880 ± 0.143401 |
| w512 | 11.1934 ± 0.116404 | 10.1971 ± 0.0474257 | 15.0305 ± 0.0387439 | 12.5140 ± 0.0557684 |
| b512 | 12.7632 ± 0.0585246 | 12.1521 ± 0.0840653 | 20.2945 ± 0.0636498 | 16.3681 ± 0.0952735 |
| b1024 | 15.4002 ± 0.0805145 | 15.1569 ± 0.0950767 | 30.7160 ± 0.0960614 | 24.1862 ± 0.149499 |


`critic_update` energy per FLOP [J/FLOP] (ratio of means ± sd of per-seed ratios):


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 4.99401e-11 ± 5.43051e-13 | 4.99130e-11 ± 4.77121e-13 | 4.67369e-11 ± 4.16089e-13 | 4.78277e-11 ± 5.62631e-13 |
| utd2 | 4.96007e-11 ± 6.88753e-13 | 4.92390e-11 ± 4.21044e-13 | 4.69456e-11 ± 1.12796e-12 | 4.68670e-11 ± 4.16683e-13 |
| utd4 | 4.95641e-11 ± 3.11378e-13 | 4.97166e-11 ± 3.19133e-13 | 4.66232e-11 ± 2.62894e-13 | 4.67689e-11 ± 2.18541e-13 |
| w256 | 5.74795e-10 ± 1.00004e-11 | 5.83941e-10 ± 8.54295e-12 | 4.57338e-10 ± 3.55853e-12 | 4.73405e-10 ± 9.94992e-12 |
| w512 | 1.55877e-10 ± 3.46116e-12 | 1.57989e-10 ± 3.56752e-12 | 1.38353e-10 ± 2.34615e-12 | 1.41290e-10 ± 2.70761e-12 |
| b512 | 2.95699e-11 ± 3.77985e-13 | 3.03231e-11 ± 6.70709e-13 | 2.82599e-11 ± 4.65951e-13 | 2.87694e-11 ± 3.46258e-13 |
| b1024 | 1.65010e-11 ± 1.26096e-13 | 1.69096e-11 ± 2.55178e-13 | 1.64517e-11 ± 1.43255e-13 | 1.65587e-11 ± 2.12203e-13 |


`actor_update` energy per FLOP [J/FLOP] (ratio of means ± sd of per-seed ratios):


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 7.77155e-11 ± 2.72600e-13 | 7.73536e-11 ± 3.30610e-13 | 7.34190e-11 ± 1.50782e-13 | 7.36537e-11 ± 6.10351e-13 |
| utd2 | 7.68439e-11 ± 5.53879e-13 | 7.66598e-11 ± 2.99944e-13 | 7.24077e-11 ± 4.89227e-13 | 7.33995e-11 ± 2.90773e-13 |
| utd4 | 7.64252e-11 ± 2.04089e-13 | 7.69426e-11 ± 3.65341e-13 | 7.19078e-11 ± 1.33377e-13 | 7.31277e-11 ± 2.89652e-13 |
| w256 | 8.82257e-10 ± 5.24431e-12 | 8.95829e-10 ± 6.88888e-12 | 6.85026e-10 ± 1.45750e-12 | 7.04103e-10 ± 8.72085e-12 |
| w512 | 2.42550e-10 ± 2.04282e-12 | 2.45169e-10 ± 8.71547e-13 | 2.11388e-10 ± 1.22663e-12 | 2.16097e-10 ± 1.27929e-12 |
| b512 | 4.56739e-11 ± 8.87094e-14 | 4.61190e-11 ± 1.38462e-13 | 4.30186e-11 ± 1.11781e-13 | 4.35793e-11 ± 4.82696e-14 |
| b1024 | 2.55936e-11 ± 4.19845e-14 | 2.60298e-11 ± 1.09080e-13 | 2.46069e-11 ± 5.79622e-14 | 2.50741e-11 ± 8.96444e-14 |


`target_update` energy per FLOP [nJ per Polyak op] (ratio of means ± sd of per-seed ratios):


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 5.92935 ± 0.0624667 | 5.92128 ± 0.0669656 | 5.51568 ± 0.0552575 | 5.47718 ± 0.0691586 |
| utd2 | 5.89117 ± 0.0591010 | 5.84040 ± 0.0287688 | 5.41400 ± 0.0216318 | 5.48675 ± 0.0450685 |
| utd4 | 5.85662 ± 0.0294590 | 5.85865 ± 0.0421906 | 5.42315 ± 0.0369596 | 5.47795 ± 0.0394436 |
| w256 | 67.8239 ± 0.622538 | 68.4187 ± 0.367551 | 50.9871 ± 0.258664 | 52.3989 ± 0.590783 |
| w512 | 18.7559 ± 0.134690 | 18.8131 ± 0.107694 | 15.9967 ± 0.145559 | 16.3062 ± 0.171154 |
| b512 | 6.83469 ± 0.0575755 | 6.85310 ± 0.104921 | 6.32989 ± 0.0806214 | 6.40410 ± 0.0441278 |
| b1024 | 7.57058 ± 0.0533378 | 7.72069 ± 0.0671182 | 7.17537 ± 0.0276132 | 7.28118 ± 0.0585358 |


Per-call FLOPs critic / actor (identical code path; `flops_per_call.json`):


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 4.92359e+09 / 3.84513e+09 | 4.92359e+09 / 3.84513e+09 | 5.25494e+09 / 4.13244e+09 | 5.25494e+09 / 4.13244e+09 |
| utd2 | 4.92359e+09 / 3.84513e+09 | 4.92359e+09 / 3.84513e+09 | 5.25494e+09 / 4.13244e+09 | 5.25494e+09 / 4.13244e+09 |
| utd4 | 4.92359e+09 / 3.84513e+09 | 4.92359e+09 / 3.84513e+09 | 5.25494e+09 / 4.13244e+09 | 5.25494e+09 / 4.13244e+09 |
| w256 | 3.24927e+08 / 2.56639e+08 | 3.24927e+08 / 2.56639e+08 | 4.07765e+08 / 3.28466e+08 | 4.07765e+08 / 3.28466e+08 |
| w512 | 1.25383e+09 / 9.83040e+08 | 1.25383e+09 / 9.83040e+08 | 1.41951e+09 / 1.12669e+09 | 1.41951e+09 / 1.12669e+09 |
| b512 | 9.84718e+09 / 7.69026e+09 | 9.84718e+09 / 7.69026e+09 | 1.05099e+10 / 8.26488e+09 | 1.05099e+10 / 8.26488e+09 |
| b1024 | 1.96944e+10 / 1.53805e+10 | 1.96944e+10 / 1.53805e+10 | 2.10198e+10 / 1.65298e+10 | 2.10198e+10 / 1.65298e+10 |


GU measured duration [s] and sub-timer coverage [%]:


| config | HC MBPO | HC SAC | Ant MBPO | Ant SAC |
|---|---|---|---|---|
| base | 407.584 ± 2.70475; cov 96.9650 ± 0.0406469 | 404.865 ± 2.23475; cov 97.2629 ± 0.0876070 | 411.097 ± 2.42950; cov 97.0025 ± 0.0629928 | 413.120 ± 2.85184; cov 96.5403 ± 0.146463 |
| utd2 | 819.784 ± 10.7804; cov 96.9316 ± 0.0493324 | 805.164 ± 4.92688; cov 97.3208 ± 0.0527574 | 834.334 ± 15.3206; cov 97.1278 ± 0.0346554 | 820.887 ± 6.90105; cov 96.6093 ± 0.0536830 |
| utd4 | 1642.15 ± 11.5343; cov 96.8633 ± 0.0386294 | 1620.66 ± 9.77237; cov 97.3409 ± 0.0441974 | 1663.35 ± 5.72558; cov 97.1895 ± 0.0582139 | 1641.90 ± 8.83504; cov 96.6638 ± 0.0493909 |
| w256 | 400.354 ± 3.34163; cov 97.6168 ± 0.0251155 | 395.488 ± 2.52121; cov 97.6601 ± 0.0313391 | 403.802 ± 1.51767; cov 97.5984 ± 0.0583281 | 402.446 ± 5.65309; cov 97.7029 ± 0.0378163 |
| w512 | 403.245 ± 4.95271; cov 97.3516 ± 0.0356169 | 397.242 ± 5.73848; cov 97.4493 ± 0.0846755 | 407.009 ± 3.08714; cov 97.2396 ± 0.0629126 | 402.092 ± 4.39150; cov 96.8076 ± 0.0450550 |
| b512 | 416.875 ± 3.72760; cov 97.1069 ± 0.0349667 | 417.232 ± 6.04938; cov 97.4678 ± 0.0700865 | 424.627 ± 4.58935; cov 97.2380 ± 0.0533050 | 422.920 ± 3.95414; cov 96.6889 ± 0.0338509 |
| b1024 | 420.908 ± 2.64711; cov 97.1612 ± 0.0476162 | 418.523 ± 5.64909; cov 97.4270 ± 0.0537226 | 444.767 ± 3.63136; cov 97.2161 ± 0.0477588 | 435.377 ± 4.70354; cov 96.5377 ± 0.0698847 |


**6.7 Open points from the instructions (reported, not resolved):**

- *Task durations versus the 1 s polling interval:* `rollout`: 8000 of 8000 tasks < 1 s (min/median/max 0.188283/0.353606/0.407737 s). `dynamics_model_update`: 1 of 8000 < 1 s (min/median/max 0.893875/13.5768/135.478 s); `synthetic_rollout_generation`: 8000 of 8000 < 1 s (min/median/max 0.00456115/0.00852251/0.116328 s); `gradient_updates`: 0 of 8000 < 1 s (min/median/max 3.84608/4.18918/17.8972 s). Per configuration: Section 5.2a.
- *Host-side work counted as FLOPs:* none; only matmul FLOPs (`FlopCounterMode`). **Counted:** `rollout` = actor forward at batch 1; `critic_update`/`actor_update` = as SAC (Section 4.1); `dynamics_model_update` = per fit epoch every member's fwd+bwd over its n_train bootstrap samples + the full-ensemble holdout forward (from per-sample constants; Adam, the NLL, `softplus` log-variance bounds, bootstrap resampling, `np.random.shuffle`, input normalisation, per-epoch `.cpu()` syncs are **not** counted); `synthetic_rollout_generation` = per generated sample actor forward + full 7-member ensemble forward (the random elite choice, Gaussian sampling, `termination_fn` (NumPy), `add_batch`, `.cpu().numpy()` transfers are **not** counted); `buffer_sample` = 0 (host sampling of two buffers + concatenate + H2D copies); `target_update` = Polyak operations. Counting uses all 7 members although only the 5 elites are used for the prediction (every member is scored, `predict` picks afterwards).
- *Segments whose timers or FLOPs are produced differently from SAC:* `dynamics_model_update` and `synthetic_rollout_generation` are directly measured CodeCarbon tasks (not time-allocated); the FLOPs of `dynamics_model_update` depend on the data-dependent early stopping (fit epochs per fit logged as `model_train_epochs`), and those of `synthetic_rollout_generation` on terminations; `buffer_sample` is two host samples + concatenate (SAC: one sample); everything inside `gradient_updates` is otherwise the SAC code path.

## 7. Discrepancies, NOT AVAILABLE items, questions

**(a) Where data or code contradict CLAUDE.md, README.md or `thesis-layout.md` (the latter is not present in the repository):**

- CLAUDE.md / README.md state that zero GPU-clock-lock / CPU-governor permission failures and zero RAPL / geolocation fallbacks were found 'by grepping every run.log'. That grep cannot detect those failures (the `gpu_control`, `thermal_gate` and CodeCarbon loggers do not write to `run.log`). For MBPO the claim is supported by other evidence: 80 of 80 runs have a sweep-log block with captured stderr (`2>&1 | tee -a`) containing none of the failure strings, plus data-side checks (varying RAPL `cpu_power`, non-fallback geolocation, `thermal_gate.reason = reached_reference`) for all runs (Section 1.3). The clock lock is verified as requested, not as applied.
- The one-row `emissions.csv` written by `tracker.stop()` has `duration` = 90.0056 s (median over the MBPO runs; it equals the duration of the last task, the 90 s idle tail) while its `energy_consumed` is the cumulative whole-run energy (equal to the sum of the per-task rows within 1.2e-06 relative). `emissions.csv` therefore cannot be used for a whole-run mean power; the 'P whole run' columns of this file use Σ energy / Σ duration of all rows of the per-task CSV (idle head, warmup, [pretrain], training tasks, idle tail; the unmeasured 30 s settle is not included).
- `configs/config.py` documents `model_train_freq` (250) as 'retrain the ensemble once real env steps since the last retrain reach this', and README.md says the ensemble is 'retrained periodically'. `mbpo.train()` evaluates the condition only once per epoch (1000 real steps), so the effective schedule is one fit per epoch = 100 fits per run in every configuration (Section 6.2), not one per 250 steps.
- The docstring of `EnsembleDynamicsModel.fit()` says it 'trains the full ensemble from scratch'; the code never re-initialises the network or the Adam optimiser between fits (`fit()` only refits the input normaliser and resamples the bootstrap), so every fit continues from the previous weights (warm start). Early stopping consequently ends after on average 9.85368 (HC) / 14.4759 (Ant) epochs per later fit, versus 180.800 (HC) / 28.5556 (Ant) epochs in the first fit (means over all runs; Section 6.2).
- `thesis-layout.md` is referenced by the instructions but does not exist in the repository (searched the repo and the Downloads folder); the layout expectations (80 MBPO runs: 7 + 9 configurations) were taken from the instruction file and verified against `results/`.

**(b) Quantities that could not be obtained (`NOT AVAILABLE`):**

- NOT AVAILABLE: Explicit early-termination counts of the synthetic rollouts (number of rows terminated per epoch / per step): not logged by `mbpo.py`; only the shortfall of `synthetic_transitions_generated` against n_start·L is available (Section 6.1f / 6.4).
- NOT AVAILABLE: GPU theoretical FP32 peak (SM count / lanes per SM are not stored in the repo and no CUDA is available on the Windows checkout) — achieved throughput is reported in GFLOP/s without a peak-utilisation fraction.
- NOT AVAILABLE: Whether FP32 matmuls ran as TF32 on the RTX 5090 under torch 2.13.0+cu130: the code sets no TF32 / matmul-precision flag and no run logs the effective setting (see Section 4.1 Part 2.3).
- NOT AVAILABLE: Positive confirmation that the GPU clock lock (2000/2000 MHz), persistence mode and the CPU `performance` governor were *applied*: `metadata.json` records only the request; failure messages would appear in the sweep-log stderr and none do.
- NOT AVAILABLE: gymnasium / mujoco versions on the Linux experiment box: not recorded in `metadata.json`; environment dimensions here were read from the local venv and agree with `flops_per_call.json`.
- NOT AVAILABLE: Simulator (MuJoCo) share of the `rollout` task duration: no per-step simulator timing is logged, so the split of `rollout` into policy/model compute and `env.step()` is not measurable from the data.

**(c) Questions for the author that the repository cannot answer:**

- Idle-baseline drift: the head idle window integrates 3.07903 s more than its recorded 90 s (RAM energy / RAM power), the tail window does not; this file reports as-recorded idle powers (definition of Section 4.1) and the window-corrected drift in Section 5.2e. Which of the two the thesis text should quote is not decided in the repository.
- `rollout_min_epoch`/`rollout_max_epoch` differ between the HC baseline (20/150) and the Ant baseline (20/100) in addition to `rollout_max_length` (1 vs 25). With L_max = 1 on HC the schedule is constant, so the epoch bounds have no effect there; whether the thesis should describe the HC regime as 'no scheduling' is for the author to decide.
- The model is refit in every epoch (1000 real steps ≥ `model_train_freq` = 250), i.e. 100 fits per run instead of the 400 implied by 'retrain every 250 steps'. Whether this is the intended MBPO setting for the thesis description is not determined by the repository.

**Index of produced files** (`contexts/Section_4.3_data/`): `mbpo_segments_long.csv`, `mbpo_cross_seed_summary.csv`, `mbpo_run_inventory.csv`, `mbpo_per_epoch.csv`, `mbpo_model_per_epoch.csv`, `mbpo_idle_floor.csv`, `mbpo_ols_fits.csv`, `mbpo_delta_pairs.csv`, `mbpo_mad_flags.csv`, `mbpo_learning_performance.csv`; scripts `build_section_4_3.py` (here) and the shared modules in `contexts/Section_4.2_data/`.
