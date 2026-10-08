# Section 4.2 TD3 — context

Data context for thesis Section 4.2 (TD3): 80 canonical runs (8 configurations × 2 environments × 5 seeds [331, 958, 14577, 43611, 85062]). Facts, numbers and provenance only — no interpretation, no LaTeX. All energies are **gross** (idle floor included, nothing subtracted); mean ± sample sd (ddof = 1) over the five seeds; 6 significant digits; every table cell shows `HalfCheetah-v5 ‖ Ant-v5` (left ‖ right) unless a table says otherwise. Conventions of Section 4.1 apply (`measured` vs `allocated` segments; `target_update` in nJ per Polyak operation, never J/FLOP; `buffer_sample` has no FLOPs; J/FLOP = ratio of means with the sd of the five per-seed ratios; relative changes computed per seed then averaged; environment comparison paired by seed, Ant relative to HC). Training total = `TOTAL_MEASURED_TRAINING` (rollout + gradient_updates); warmup and idle windows are excluded.

Checks run by the generator: 28; passed 28; failed 0. Independent recomputation vs the pipeline CSVs: 3808 values compared, 0 deviating by more than 1e-9, maximum relative deviation 3.533e-14.

**Configuration tags:** `base` = hidden (1024,1024), batch 100, UTD 1, policy delay 2, warmup 10000 steps (TD3 paper hyperparameters except width); `utd2`/`utd4`, `w256`/`w512`, `b256`/`b512`/`b1024` change exactly that one factor of the baseline. Abbreviations: HC = HalfCheetah-v5, Ant = Ant-v5, GU = `gradient_updates`, TOTAL = `TOTAL_MEASURED_TRAINING`, buf_sample = `buffer_sample`, critic = `critic_update`, actor = `actor_update`, target = `target_update`.

## 0. Provenance


| Item | Value |
|---|---|
| Generated (local time) | 2026-10-07T10:38:09 |
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
**Scripts used** (all under `contexts/Section_4.2_data/` unless stated; all read-only on `results/`, `flop_analysis/` and git): `build_section_4_2.py`, `s4_data.py`, `s4_blocks.py`, `s4_audit.py`. Shared modules: `s4_data.py` (loading, derived quantities, independent FLOP recomputation), `s4_blocks.py` (common tables, cross-configuration quantities), `s4_audit.py` (this part). Re-run from the repo root with `.venv/Scripts/python.exe contexts/Section_4.2_data/build_section_4_2.py`. Definitions are taken from `contexts/section_4.1_data/build_section_4_1.py` (the generator of Section 4.1) and are listed in the docstring of `s4_data.py`; no definition of 4.1 was missing.

**Pipeline files used for the cross-check:**


| File | SHA-256 | mtime (local) | last git commit touching it |
|---|---|---|---|
| `flop_analysis/output/per_run_energy_per_flop.csv` | 6e7f8f250a797fe3cca22c85a22e0e93897a91f1e0a5af5d575a4067ecc6d748 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |
| `flop_analysis/output/cross_seed_energy_per_flop.csv` | 3ebcc97fa64628cd50115ad58dc54dee129769375e2276babadf25d5096e5ec7 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |
| `flop_analysis/flops_per_call.json` | 7a562e0b092b68b6fbaea089354298114e864db72c6309847db44d0565e957a7 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |

Freshness: newest `segment_energy.json` among the 80 TD3 runs: `results/td3/HalfCheetah-v5/seed_958/20260915_210134` (2026-09-23T18:04:24); oldest pipeline CSV mtime 2026-09-28T17:41:20 — **PASS** — newest segment_energy.json older than the pipeline CSVs (filesystem mtime; the content-based reconciliation is in Section 1.5)

**Code-version consistency of the 80 runs** (`metadata.json` → `git_commit`):


| git_commit | # runs | commit | configurations |
|---|---|---|---|
| `58962d2217` | 10 | 58962d2 2026-09-09 Added and tested td3 algorithm | Ant-base, HC-base |
| `aa9186b0e5` | 20 | aa9186b 2026-09-09 Setup UTD sweep run script for SAC and TD3 across both envs and 5 seeds | Ant-utd2, Ant-utd4, HC-utd2, HC-utd4 |
| `7e672ef12c` | 20 | 7e672ef 2026-09-12 Setting up width sweep for SAC, TD3 and MBPO | Ant-w256, Ant-w512, HC-w256, HC-w512 |
| `9b4b44cd2f` | 15 | 9b4b44c 2026-09-15 Preparing batch size sweep for Half Cheetah | HC-b1024, HC-b256, HC-b512 |
| `788ffcd841` | 15 | 788ffcd 2026-09-17 Scheduling batch sweep for ant and new rollout sweep for MBPO | Ant-b1024, Ant-b256, Ant-b512 |

Algorithm-side files at each run commit (git blob ids compared):


| file | status |
|---|---|
| `algorithms/td3.py` | identical at every run commit |
| `algorithms/replay_buffer.py` | identical at every run commit |
| `algorithms/tracker_utils.py` | identical at every run commit |

Config dataclasses at each run commit (AST source segment compared):


| definition | status |
|---|---|
| `ExperimentConfig` | identical at every run commit |
| `TD3Config` | identical at every run commit |

Runner/utility files, first versus last run commit: `experiment_runner.py`: +4/−0 lines between `58962d2` and `788ffcd`; `run_experiment.py`: identical; `utils/gpu_control.py`: identical; `utils/thermal_gate.py`: identical; `algorithms/tracker_utils.py`: identical.
`thermal_gate` key sets in `metadata.json` (a difference between runs that carry the same `git_commit` would show an uncommitted working tree at run time):


| thermal_gate keys | # runs | runs (if ≤ 5) |
|---|---|---|
| final_power_w, final_temp_c, gated, reason, run_end_power_w, run_end_temp_c, run_start_power_w, run_start_temp_c, waited_seconds | 80 | … |


All algorithm-side files named above are byte-identical at every run commit.

## 1. Data basis and audit

**1.1 Inventory** (one row per configuration; values from the logged `metadata.json`, identical within a configuration — checked in 1.3). `launcher` is derived from the sweep-log blocks in which the run directory appears (Section 1.4):


| config | env | description | hidden_sizes | batch_size | updates_per_env_step | policy_update_delay | warmup | train_steps | steps/epoch | architecture signature | override file | launcher | # runs | seeds present | # duplicates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | HalfCheetah-v5 | hidden (1024,1024), batch 100 (paper default), UTD 1, policy delay 2 | [1024, 1024] | 100 | 1 | 2 | 10000 | 100000 | 1000 | `bs100_h1024x1024` | `none (TD3Config defaults)` | scripts/run_td3_env_sweep.sh (log `results/td3/_env_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| utd2 | HalfCheetah-v5 | UTD 2 | [1024, 1024] | 100 | 2 | 2 | 10000 | 100000 | 1000 | `bs100_h1024x1024` | `configs/overrides/td3_utd2.json` | scripts/run_utd_sweep.sh (at commit aa9186b / 24f41b0: SAC-Ant + TD3 UTD sweep; the file now holds the MBPO UTD sweep) (log `results/_utd_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| utd4 | HalfCheetah-v5 | UTD 4 | [1024, 1024] | 100 | 4 | 2 | 10000 | 100000 | 1000 | `bs100_h1024x1024` | `configs/overrides/td3_utd4.json` | scripts/run_utd_sweep.sh (at commit aa9186b / 24f41b0: SAC-Ant + TD3 UTD sweep; the file now holds the MBPO UTD sweep) (log `results/_utd_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| w256 | HalfCheetah-v5 | hidden (256,256) | [256, 256] | 100 | 1 | 2 | 10000 | 100000 | 1000 | `bs100_h256x256` | `configs/overrides/td3_width256.json` | scripts/run_width_sweep.sh (log `results/_width_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| w512 | HalfCheetah-v5 | hidden (512,512) | [512, 512] | 100 | 1 | 2 | 10000 | 100000 | 1000 | `bs100_h512x512` | `configs/overrides/td3_width512.json` | scripts/run_width_sweep.sh (log `results/_width_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b256 | HalfCheetah-v5 | batch 256 (SAC/MBPO default; the batch-matched comparison point) | [1024, 1024] | 256 | 1 | 2 | 10000 | 100000 | 1000 | `bs256_h1024x1024` | `configs/overrides/td3_batch256.json` | scripts/run_batch_size_sweep.sh (log `results/_batch_size_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b512 | HalfCheetah-v5 | batch 512 | [1024, 1024] | 512 | 1 | 2 | 10000 | 100000 | 1000 | `bs512_h1024x1024` | `configs/overrides/td3_batch512.json` | scripts/run_batch_size_sweep.sh (log `results/_batch_size_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b1024 | HalfCheetah-v5 | batch 1024 | [1024, 1024] | 1024 | 1 | 2 | 10000 | 100000 | 1000 | `bs1024_h1024x1024` | `configs/overrides/td3_batch1024.json` | scripts/run_batch_size_sweep.sh (log `results/_batch_size_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| base | Ant-v5 | hidden (1024,1024), batch 100 (paper default), UTD 1, policy delay 2 | [1024, 1024] | 100 | 1 | 2 | 10000 | 100000 | 1000 | `bs100_h1024x1024` | `none (TD3Config defaults)` | scripts/run_td3_env_sweep.sh (log `results/td3/_env_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| utd2 | Ant-v5 | UTD 2 | [1024, 1024] | 100 | 2 | 2 | 10000 | 100000 | 1000 | `bs100_h1024x1024` | `configs/overrides/td3_utd2.json` | scripts/run_utd_sweep.sh (at commit aa9186b / 24f41b0: SAC-Ant + TD3 UTD sweep; the file now holds the MBPO UTD sweep) (log `results/_utd_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| utd4 | Ant-v5 | UTD 4 | [1024, 1024] | 100 | 4 | 2 | 10000 | 100000 | 1000 | `bs100_h1024x1024` | `configs/overrides/td3_utd4.json` | scripts/run_utd_sweep.sh (at commit aa9186b / 24f41b0: SAC-Ant + TD3 UTD sweep; the file now holds the MBPO UTD sweep) (log `results/_utd_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| w256 | Ant-v5 | hidden (256,256) | [256, 256] | 100 | 1 | 2 | 10000 | 100000 | 1000 | `bs100_h256x256` | `configs/overrides/td3_width256.json` | scripts/run_width_sweep.sh (log `results/_width_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| w512 | Ant-v5 | hidden (512,512) | [512, 512] | 100 | 1 | 2 | 10000 | 100000 | 1000 | `bs100_h512x512` | `configs/overrides/td3_width512.json` | scripts/run_width_sweep.sh (log `results/_width_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b256 | Ant-v5 | batch 256 (SAC/MBPO default; the batch-matched comparison point) | [1024, 1024] | 256 | 1 | 2 | 10000 | 100000 | 1000 | `bs256_h1024x1024` | `configs/overrides/td3_batch256.json` | scripts/run_ant_overnight_sweep.sh (= run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh) (log `results/_ant_overnight_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b512 | Ant-v5 | batch 512 | [1024, 1024] | 512 | 1 | 2 | 10000 | 100000 | 1000 | `bs512_h1024x1024` | `configs/overrides/td3_batch512.json` | scripts/run_ant_overnight_sweep.sh (= run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh) (log `results/_ant_overnight_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |
| b1024 | Ant-v5 | batch 1024 | [1024, 1024] | 1024 | 1 | 2 | 10000 | 100000 | 1000 | `bs1024_h1024x1024` | `configs/overrides/td3_batch1024.json` | scripts/run_ant_overnight_sweep.sh (= run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh) (log `results/_ant_overnight_sweep.log`) | 5 | 331,958,14577,43611,85062 | 0 |


**1.2 Full run list** → `contexts/Section_4.2_data/td3_run_inventory.csv` (80 rows: algo, config, env, seed, run directory, timestamp, git commit, versions, clock lock, start/end time, full logged `algo_config`).

**1.3 Audit checks**

- **PASS** — layout-document expectation: 80 runs (HC: 8 configurations, Ant: 8 configurations × 5 seeds); found 80 (unmatched run directories at canonical seeds: 0)
- **PASS** — configurations per environment equal the layout-document counts (found {'HC': 8, 'Ant': 8})
- **PASS** — exactly 5 runs per configuration (all 16 configurations have 5)
- **PASS** — every run uses one of the 5 canonical seeds {331, 958, 14577, 43611, 85062}; each seed once per configuration
- **PASS** — no duplicate (algo, env, seed, config) runs
- **PASS** — every run has emissions.csv, segment_energy.json, training_metrics.json, metadata.json, run.log
- **PASS** — every run has exactly one per-task CodeCarbon log `emissions_*.csv`
- **PASS** — the logged configuration of every configuration differs from its environment's baseline in exactly the swept factor (algo_config), has no `experiment_config` difference (ignoring seed/env/algo and the metadata-only key `thermal_gate_reference_file`), and is identical across its 5 seeds
- **PASS** — `experiment_config` of the HC and Ant baselines are identical apart from seed/env_id/algo_name and the metadata-only key ({})
  - (train_steps, steps_per_epoch, warmup_steps, env) → #runs: {(100000, 1000, 10000, 'HC'): 40, (100000, 1000, 10000, 'Ant'): 40}
- **PASS** — per-task CSV has exactly the expected tasks: rollout_0..99, gradient_updates_0..99, warmup_0, idle_baseline_head, idle_baseline_tail
- **PASS** — `training_metrics.json` has 100 epoch rows in every run
- **PASS** — GPU = NVIDIA GeForce RTX 5090 in every run (`cuda_device_name`)
  - per-task CSV `gpu_model`: {'1 x NVIDIA GeForce RTX 5090': 80}; `cpu_model`: {'Intel(R) Core(TM) Ultra 9 285K': 80}
- **PASS** — GPU clock lock requested at 2000/2000 MHz with lock_gpu_clocks = true in every run
  - persistence mode requested: {True: 80}; CPU `performance` governor requested: {True: 80}
- **PASS** — torch 2.13.0+cu130 and codecarbon 3.3.0 in every run ({('2.13.0+cu130', '3.3.0'): 80})
- **PASS** — thermal gate: gated = true and reason = reached_reference (no timeout) in every run
  - thermal gate: waited_seconds max 1.64509e-05 s; gate temperature 40.0000–55.0000 °C; gate power 23.7290–41.4270 W (the gate passes on its first poll because its reference is captured fresh in each process, `utils/thermal_gate.py`)
- **PASS** — `run.log` of every run contains none of the 20 failure-mode strings (list in Section 4.1, schema sample S6)
- **PASS** — sweep-log block (stdout+stderr of the launch, `2>&1 | tee -a`) contains none of the strings — available for 80 of 80 runs
- **PASS** — data-side RAPL check: per-task `cpu_power` varies across the 100 gradient_updates tasks in every run (a TDP fallback would be constant) (min distinct values per run = 100)
- **PASS** — data-side geolocation check: no run shows the Canada/Quebec fallback ({('Germany', 'baden-wurttemberg'): 80})

**1.3b Logged-configuration differences per configuration** (each row: difference of the logged `algo_config` / `experiment_config` to the baseline of the same environment; last column: difference to the dataclass defaults of `TD3Config`, i.e. the override file content):


| config | algo_config vs same-env baseline | experiment_config vs same-env baseline | identical across the 5 seeds | algo_config vs `TD3Config` defaults |
|---|---|---|---|---|
| HC-base | — | — | yes | — |
| HC-utd2 | updates_per_env_step: 1 → 2 | — | yes | updates_per_env_step: 1 → 2 |
| HC-utd4 | updates_per_env_step: 1 → 4 | — | yes | updates_per_env_step: 1 → 4 |
| HC-w256 | hidden_sizes: [1024, 1024] → [256, 256] | — | yes | hidden_sizes: [1024, 1024] → [256, 256] |
| HC-w512 | hidden_sizes: [1024, 1024] → [512, 512] | — | yes | hidden_sizes: [1024, 1024] → [512, 512] |
| HC-b256 | batch_size: 100 → 256 | — | yes | batch_size: 100 → 256 |
| HC-b512 | batch_size: 100 → 512 | — | yes | batch_size: 100 → 512 |
| HC-b1024 | batch_size: 100 → 1024 | — | yes | batch_size: 100 → 1024 |
| Ant-base | — | — | yes | — |
| Ant-utd2 | updates_per_env_step: 1 → 2 | — | yes | updates_per_env_step: 1 → 2 |
| Ant-utd4 | updates_per_env_step: 1 → 4 | — | yes | updates_per_env_step: 1 → 4 |
| Ant-w256 | hidden_sizes: [1024, 1024] → [256, 256] | — | yes | hidden_sizes: [1024, 1024] → [256, 256] |
| Ant-w512 | hidden_sizes: [1024, 1024] → [512, 512] | — | yes | hidden_sizes: [1024, 1024] → [512, 512] |
| Ant-b256 | batch_size: 100 → 256 | — | yes | batch_size: 100 → 256 |
| Ant-b512 | batch_size: 100 → 512 | — | yes | batch_size: 100 → 512 |
| Ant-b1024 | batch_size: 100 → 1024 | — | yes | batch_size: 100 → 1024 |

The warning/grep caveat of Section 4.1 applies: `run.log` is written only by the `experiment_runner` logger; the `gpu_control`, `thermal_gate` and CodeCarbon loggers write to stderr, which the sweep scripts capture into the sweep log (`2>&1 | tee -a`). The clock lock / governor are therefore verified as *requested* (metadata) and as *not reported failed* (sweep-log blocks), never as positively applied; applied lock values are not recorded in `metadata.json`.

**1.4 Execution facts.** Times are `metadata.json` `start_time_utc`/`end_time_utc` (UTC). A *session* = maximal block of runs of **all 300 recorded runs** (all algorithms) with < 30 min between one run's end and the next run's start (same definition as Section 4.1). `contiguous` = the five runs of the configuration are consecutive among the TD3 runs. `launch logs` = sweep logs containing the run directory (each run's stdout+stderr block).


| config | first start (UTC) | last end (UTC) | session(s) | all 5 seeds in one session | contiguous | seed order (by start time) | launch log(s) | # runs with a log block |
|---|---|---|---|---|---|---|---|---|
| HC-base | 2026-09-09T10:22:04 | 2026-09-09T10:58:13 | 13 | yes | yes | 331,958,14577,43611,85062 | `results/td3/_env_sweep.log` | 5 |
| HC-utd2 | 2026-09-09T15:55:06 | 2026-09-09T16:47:48 | 13 | yes | yes | 331,958,14577,43611,85062 | `results/_utd_sweep.log` | 5 |
| HC-utd4 | 2026-09-09T16:47:49 | 2026-09-09T18:12:42 | 13 | yes | yes | 331,958,14577,43611,85062 | `results/_utd_sweep.log` | 5 |
| HC-w256 | 2026-09-13T08:17:02 | 2026-09-13T08:52:42 | 15 | yes | yes | 331,958,14577,43611,85062 | `results/_width_sweep.log` | 5 |
| HC-w512 | 2026-09-13T08:52:43 | 2026-09-13T09:29:01 | 15 | yes | yes | 331,958,14577,43611,85062 | `results/_width_sweep.log` | 5 |
| HC-b256 | 2026-09-15T17:39:42 | 2026-09-15T18:16:11 | 17 | yes | yes | 331,958,14577,43611,85062 | `results/_batch_size_sweep.log` | 5 |
| HC-b512 | 2026-09-15T18:16:12 | 2026-09-15T18:53:39 | 17 | yes | yes | 331,958,14577,43611,85062 | `results/_batch_size_sweep.log` | 5 |
| HC-b1024 | 2026-09-15T18:53:40 | 2026-09-15T19:32:05 | 17 | yes | yes | 331,958,14577,43611,85062 | `results/_batch_size_sweep.log` | 5 |
| Ant-base | 2026-09-09T10:58:15 | 2026-09-09T11:36:10 | 13 | yes | yes | 331,958,14577,43611,85062 | `results/td3/_env_sweep.log` | 5 |
| Ant-utd2 | 2026-09-09T18:12:43 | 2026-09-09T19:06:31 | 13 | yes | yes | 331,958,14577,43611,85062 | `results/_utd_sweep.log` | 5 |
| Ant-utd4 | 2026-09-09T19:06:32 | 2026-09-09T20:34:22 | 13 | yes | yes | 331,958,14577,43611,85062 | `results/_utd_sweep.log` | 5 |
| Ant-w256 | 2026-09-13T09:29:02 | 2026-09-13T10:06:24 | 15 | yes | yes | 331,958,14577,43611,85062 | `results/_width_sweep.log` | 5 |
| Ant-w512 | 2026-09-13T10:06:25 | 2026-09-13T10:44:47 | 15 | yes | yes | 331,958,14577,43611,85062 | `results/_width_sweep.log` | 5 |
| Ant-b256 | 2026-09-17T18:12:15 | 2026-09-17T18:50:22 | 18 | yes | yes | 331,958,14577,43611,85062 | `results/_ant_overnight_sweep.log` | 5 |
| Ant-b512 | 2026-09-17T18:50:23 | 2026-09-17T19:28:48 | 18 | yes | yes | 331,958,14577,43611,85062 | `results/_ant_overnight_sweep.log` | 5 |
| Ant-b1024 | 2026-09-17T19:28:49 | 2026-09-17T20:09:19 | 18 | yes | yes | 331,958,14577,43611,85062 | `results/_ant_overnight_sweep.log` | 5 |


Sessions containing TD3 runs:


| session | first start | last end | # runs (all algos) | algos | # TD3 | TD3 configs in order |
|---|---|---|---|---|---|---|
| 13 | 2026-09-09T10:22 | 2026-09-09T20:34 | 40 | {'td3': 30, 'sac': 10} | 30 | HC-base → Ant-base → HC-utd2 → HC-utd4 → Ant-utd2 → Ant-utd4 |
| 15 | 2026-09-12T17:15 | 2026-09-13T10:44 | 60 | {'sac': 20, 'mbpo': 20, 'td3': 20} | 20 | HC-w256 → HC-w512 → Ant-w256 → Ant-w512 |
| 17 | 2026-09-15T10:39 | 2026-09-15T19:32 | 35 | {'sac': 10, 'mbpo': 10, 'td3': 15} | 15 | HC-b256 → HC-b512 → HC-b1024 |
| 18 | 2026-09-17T10:10 | 2026-09-18T02:15 | 45 | {'sac': 10, 'mbpo': 20, 'td3': 15} | 15 | Ant-b256 → Ant-b512 → Ant-b1024 |

Inside the sessions listed above, consecutive runs of **any** algorithm are separated by 1.27509–574.358 s (median 1.32068 s) between one run's `end_time_utc` and the next run's `start_time_utc`, by definition < 1800 s. Run length = `end_time_utc − start_time_utc` includes the thermal-gate reference poll, the 30 s settle, both idle windows and warmup; over the TD3 runs: median 7.59506 min (min 7.07414, max 18.0064). Whether the machine was powered off or idle between sessions is not recorded (each sweep script ends with `shutdown -h +1`).

**Baseline runs** (configuration `base`, 10 runs): launched by a sweep script — their stdout+stderr blocks are in `results/td3/_env_sweep.log` (10 runs) (sum 10 of 10); not started by hand, and their launch error output was saved.
Override-file names as built by the sweep scripts (pattern lines found with `echo "configs/overrides/…"` / `overrides="configs/overrides/…"`; the algorithm / env / value are substituted at run time; the launch blocks of the logs do not record the override path): `run_ant_overnight_sweep.sh:72` `echo "configs/overrides/${ALGO_OVERRIDE_PREFIX[$algo]}${batch}.json"`; `run_ant_overnight_sweep.sh:79` `echo "configs/overrides/mbpo_ant_rollout${length}.json"`; `run_batch_size_sweep.sh:57` `echo "configs/overrides/${algo}_batch${batch}.json"`; `run_batch_size_sweep_ant.sh:74` `echo "configs/overrides/${ALGO_OVERRIDE_PREFIX[$algo]}${batch}.json"`; `run_mbpo_rollout_length_sweep.sh:42` `echo "configs/overrides/mbpo_ant_rollout${length}.json"`; `run_tdmpc2_numq_horizon_sweep.sh:58` `echo "configs/overrides/tdmpc2_ant_${param_slug}${value}.json"`; `run_tdmpc2_numq_horizon_sweep.sh:60` `echo "configs/overrides/tdmpc2_${param_slug}${value}.json"`; `run_utd_sweep.sh:147` `overrides="configs/overrides/${override_prefix}_utd${utd}.json"`; `run_width_sweep.sh:73` `echo "configs/overrides/mbpo_ant_width${width}.json"`; `run_width_sweep.sh:75` `echo "configs/overrides/${algo}_width${width}.json"`.

**Launchers.** Sweep scripts that produced these runs (read from `scripts/`): `scripts/run_td3_env_sweep.sh` (baselines, 10 runs, `--warmup-steps 10000` in the script); `scripts/run_utd_sweep.sh` (as of commit `aa9186b`: UTD ∈ {2,4}; the file has since been repointed to MBPO); `scripts/run_width_sweep.sh`; `scripts/run_batch_size_sweep.sh` (HC); `scripts/run_ant_overnight_sweep.sh` (Ant batch sizes)
Stderr capture in each script: `run_ant_overnight_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_batch_size_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_batch_size_sweep_ant.sh`: `2>&1 | tee -a $LOG_FILE`; `run_mbpo_env_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_mbpo_rollout_length_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_td3_env_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_tdmpc2_numq_horizon_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_tdmpc2_seed_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_utd_sweep.sh`: `2>&1 | tee -a $LOG_FILE`; `run_width_sweep.sh`: `2>&1 | tee -a $LOG_FILE`.

Sweep bookkeeping files (every `done` entry of this algorithm should point at one of the analysed run directories):


| status file | # entries | # TD3 entries | statuses | every done entry's run_dir is one of the analysed runs |
|---|---|---|---|---|
| `results/_utd_sweep_status.json` | 20 | 0 | {'done': 20} | n/a (no entries of this algorithm) |
| `results/_width_sweep_status.json` | 60 | 20 | {'done': 60} | yes |
| `results/_batch_size_sweep_status.json` | 35 | 15 | {'done': 35} | yes |
| `results/_ant_overnight_sweep_status.json` | 45 | 15 | {'done': 45} | yes |
| `results/td3/_env_sweep_status.json` | 10 | 10 | {'done': 10} | yes |


**Launch outcomes of TD3 in every sweep log** (each `[i/N] Starting: algo=td3 …` block classified by its closing line: `Finished OK`, `FAILED (exit …)`, or no closing line):


| log | outcome | # launches |
|---|---|---|
| `results/_ant_overnight_sweep.log` | Finished OK | 15 |
| `results/_batch_size_sweep.log` | Finished OK | 15 |
| `results/_utd_sweep.log` | Finished OK | 20 |
| `results/_width_sweep.log` | Finished OK | 20 |
| `results/td3/_env_sweep.log` | Finished OK | 10 |


All 80 recorded TD3 launches finished OK; no failure, restart or duplicate launch of TD3 appears in any sweep log.
Successful launches with a run directory: 80 (analysed canonical runs: 80).

**1.5 Independent recomputation versus the pipeline CSVs.** Every (run, segment) and (configuration, segment) value recomputed here from `segment_energy.json`, the per-task CodeCarbon CSV, `training_metrics.json` and `flops_per_call.json` (FLOP formulas re-implemented in `s4_data.flops_for_run` / `mbpo_extras`, not imported) is compared with `per_run_energy_per_flop.csv` and `cross_seed_energy_per_flop.csv`. Columns compared: energy, duration, mean power, FLOPs, J/FLOP, call count. The pipeline's cross-seed `mean_power_w` is the mean of per-seed powers and is compared with the same statistic.

| CSV | column | # values compared | max relative deviation |
|---|---|---|---|
| cross_seed | `mean_duration_s` | 96 | 4.858e-16 |
| cross_seed | `mean_energy_joules` | 96 | 3.967e-16 |
| cross_seed | `mean_energy_per_flop_j_per_flop` | 80 | 4.222e-16 |
| cross_seed | `mean_power_w` | 96 | 1.105e-14 |
| cross_seed | `total_flops` | 80 | 0.000e+00 |
| per_run | `call_count` | 400 | 0.000e+00 |
| per_run | `duration_s` | 720 | 5.117e-16 |
| per_run | `energy_per_flop_j_per_flop` | 400 | 2.998e-16 |
| per_run | `mean_power_w` | 720 | 3.533e-14 |
| per_run | `total_energy_joules` | 720 | 3.552e-16 |
| per_run | `total_flops` | 400 | 0.000e+00 |

- **PASS** — all 3808 compared values agree within 1e-9 relative (maximum deviation over all columns 3.533e-14)

## 2. Structural facts from code and FLOP files

**2.1 Configuration objects** (dumped from `configs/config.py` at HEAD):

`TD3Config` defaults (`dataclasses.asdict(TD3Config())`):

```json
{
  "hidden_sizes": [
    1024,
    1024
  ],
  "actor_lr": 0.001,
  "critic_lr": 0.001,
  "gamma": 0.99,
  "tau": 0.005,
  "batch_size": 100,
  "buffer_capacity": 1000000,
  "exploration_noise": 0.1,
  "target_policy_noise": 0.2,
  "target_noise_clip": 0.5,
  "policy_update_delay": 2,
  "updates_per_env_step": 1
}
```
Override files of the sweeps (content): `td3_batch1024.json` = {"batch_size": 1024}; `td3_batch256.json` = {"batch_size": 256}; `td3_batch512.json` = {"batch_size": 512}; `td3_utd2.json` = {"updates_per_env_step": 2}; `td3_utd4.json` = {"updates_per_env_step": 4}; `td3_width256.json` = {"hidden_sizes": [256, 256]}; `td3_width512.json` = {"hidden_sizes": [512, 512]}
- **PASS** — every override file listed in the inventory equals the corresponding fields of the logged `algo_config` of its runs
`ExperimentConfig` defaults (protocol, shared by all algorithms; `warmup_steps` is overridden to 10000 on the CLI for all TD3 runs):

```json
{
  "algo_name": "td3",
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

Complete `metadata.json` of `results/td3/HalfCheetah-v5/seed_331/20260909_122204` (HC-base, seed 331):

```json
{
  "algo_name": "td3",
  "env_id": "HalfCheetah-v5",
  "seed": 331,
  "git_commit": "58962d221751e45ed9d599f0d731eccb9cccba4c",
  "torch_version": "2.13.0+cu130",
  "device": "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "start_time_utc": "2026-09-09T10:22:04.277025",
  "experiment_config": {
    "algo_name": "td3",
    "env_id": "HalfCheetah-v5",
    "seed": 331,
    "settle_seconds": 30.0,
    "idle_baseline_seconds": 90.0,
    "idle_tail_seconds": 90.0,
    "warmup_steps": 10000,
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
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 100,
    "buffer_capacity": 1000000,
    "exploration_noise": 0.1,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "policy_update_delay": 2,
    "updates_per_env_step": 1
  },
  "codecarbon_version": "3.3.0",
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 1.0967254638671875e-05,
    "final_temp_c": 48.0,
    "final_power_w": 41.427,
    "run_start_temp_c": 48.0,
    "run_start_power_w": 28.633,
    "run_end_temp_c": 48.0,
    "run_end_power_w": 30.011
  },
  "end_time_utc": "2026-09-09T10:29:15.285136"
}
```

Complete `metadata.json` of `results/td3/Ant-v5/seed_331/20260909_125815` (Ant-base, seed 331):

```json
{
  "algo_name": "td3",
  "env_id": "Ant-v5",
  "seed": 331,
  "git_commit": "58962d221751e45ed9d599f0d731eccb9cccba4c",
  "torch_version": "2.13.0+cu130",
  "device": "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "start_time_utc": "2026-09-09T10:58:15.175632",
  "experiment_config": {
    "algo_name": "td3",
    "env_id": "Ant-v5",
    "seed": 331,
    "settle_seconds": 30.0,
    "idle_baseline_seconds": 90.0,
    "idle_tail_seconds": 90.0,
    "warmup_steps": 10000,
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
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 100,
    "buffer_capacity": 1000000,
    "exploration_noise": 0.1,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "policy_update_delay": 2,
    "updates_per_env_step": 1
  },
  "codecarbon_version": "3.3.0",
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 1.0251998901367188e-05,
    "final_temp_c": 44.0,
    "final_power_w": 25.501,
    "run_start_temp_c": 43.0,
    "run_start_power_w": 24.621,
    "run_end_temp_c": 53.0,
    "run_end_power_w": 35.037
  },
  "end_time_utc": "2026-09-09T11:05:49.455821"
}
```

**2.2 Verbatim code** — `algorithms/td3.py` at HEAD (identical at every run commit, Section 0). Networks (lines 53–71):

```python
  53  class DeterministicPolicy(nn.Module):
  54      def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
  55          super().__init__()
  56          self.net = _mlp([obs_dim, *hidden_sizes, act_dim], output_activation=nn.Tanh)
  57          self.register_buffer("act_limit", torch.as_tensor(act_limit, dtype=torch.float32))
  58  
  59      def forward(self, obs):
  60          return self.act_limit * self.net(obs)
  61  
  62  
  63  class QNetwork(nn.Module):
  64      def __init__(self, obs_dim, act_dim, hidden_sizes):
  65          super().__init__()
  66          self.net = _mlp([obs_dim + act_dim, *hidden_sizes, 1])
  67  
  68      def forward(self, obs, act):
  69          return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)
  70  
  71
```
`TD3Agent` (`__init__`, `select_action`, `update`; lines 84–204):

```python
  84      def __init__(self, obs_dim, act_dim, act_limit, cfg: TD3Config, device: torch.device):
  85          self.cfg = cfg
  86          self.device = device
  87          self.act_dim = act_dim
  88          self.act_limit = act_limit
  89  
  90          self.actor = DeterministicPolicy(obs_dim, act_dim, cfg.hidden_sizes, act_limit).to(device)
  91          self.actor_targ = DeterministicPolicy(obs_dim, act_dim, cfg.hidden_sizes, act_limit).to(device)
  92          self.actor_targ.load_state_dict(self.actor.state_dict())
  93  
  94          self.q1 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
  95          self.q2 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
  96          self.q1_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
  97          self.q2_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
  98          self.q1_targ.load_state_dict(self.q1.state_dict())
  99          self.q2_targ.load_state_dict(self.q2.state_dict())
 100  
 101          for p in self.actor_targ.parameters():
 102              p.requires_grad = False
 103          for p in self.q1_targ.parameters():
 104              p.requires_grad = False
 105          for p in self.q2_targ.parameters():
 106              p.requires_grad = False
 107  
 108          self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
 109          self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
 110          self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
 111  
 112          # Target policy smoothing noise std/clip, scaled from the paper's
 113          # normalized ([0, 1] fraction of act_limit) values -- matches the
 114          # reference implementation, which pre-scales these by max_action
 115          # before constructing the agent (main.py: `policy_noise * max_action`).
 116          self.target_policy_noise = cfg.target_policy_noise * act_limit
 117          self.target_noise_clip = cfg.target_noise_clip * act_limit
 118  
 119          self._update_counter = 0
 120  
 121      @torch.no_grad()
 122      def select_action(self, obs: np.ndarray) -> np.ndarray:
 123          """Deterministic action from the current policy (no exploration noise --
 124          the caller adds that, since it depends on the rollout phase, not the agent)."""
 125          obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
 126          action = self.actor(obs_t)
 127          return action.squeeze(0).cpu().numpy()
 128  
 129      def update(self, buffer: ReplayBuffer, batch_size: int) -> UpdateInfo:
 130          info = UpdateInfo()
 131  
 132          # --- buffer_sample ---
 133          t0 = time.perf_counter()
 134          obs, act, rew, next_obs, done = buffer.sample(batch_size)
 135          obs_t = torch.as_tensor(obs, device=self.device)
 136          act_t = torch.as_tensor(act, device=self.device)
 137          rew_t = torch.as_tensor(rew, device=self.device).squeeze(-1)
 138          next_obs_t = torch.as_tensor(next_obs, device=self.device)
 139          done_t = torch.as_tensor(done, device=self.device).squeeze(-1)
 140          info.buffer_sample_s = time.perf_counter() - t0
 141  
 142          # --- critic_update (clipped double Q-learning + target policy smoothing) ---
 143          t0 = time.perf_counter()
 144          with torch.no_grad():
 145              noise = (torch.randn_like(act_t) * self.target_policy_noise).clamp(
 146                  -self.target_noise_clip, self.target_noise_clip
 147              )
 148              next_action = (self.actor_targ(next_obs_t) + noise).clamp(-self.act_limit, self.act_limit)
 149              q1_next = self.q1_targ(next_obs_t, next_action)
 150              q2_next = self.q2_targ(next_obs_t, next_action)
 151              q_next = torch.min(q1_next, q2_next)
 152              target_q = rew_t + (1.0 - done_t) * self.cfg.gamma * q_next
 153  
 154          q1_pred = self.q1(obs_t, act_t)
 155          q2_pred = self.q2(obs_t, act_t)
 156          q1_loss = F.mse_loss(q1_pred, target_q)
 157          q2_loss = F.mse_loss(q2_pred, target_q)
 158  
 159          self.q1_opt.zero_grad(set_to_none=True)
 160          q1_loss.backward()
 161          self.q1_opt.step()
 162  
 163          self.q2_opt.zero_grad(set_to_none=True)
 164          q2_loss.backward()
 165          self.q2_opt.step()
 166          info.critic_update_s = time.perf_counter() - t0
 167          info.critic_loss = (q1_loss.item() + q2_loss.item()) / 2.0
 168  
 169          # --- actor_update + target_update (delayed, Algorithm 1: both gated on t mod d) ---
 170          do_delayed_update = (self._update_counter % self.cfg.policy_update_delay) == 0
 171          if do_delayed_update:
 172              t0 = time.perf_counter()
 173              for p in self.q1.parameters():
 174                  p.requires_grad = False
 175  
 176              actor_loss = -self.q1(obs_t, self.actor(obs_t)).mean()
 177  
 178              self.actor_opt.zero_grad(set_to_none=True)
 179              actor_loss.backward()
 180              self.actor_opt.step()
 181  
 182              for p in self.q1.parameters():
 183                  p.requires_grad = True
 184  
 185              info.actor_update_s = time.perf_counter() - t0
 186              info.actor_loss = actor_loss.item()
 187  
 188              t0 = time.perf_counter()
 189              with torch.no_grad():
 190                  for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
 191                      p_targ.data.mul_(1 - self.cfg.tau)
 192                      p_targ.data.add_(self.cfg.tau * p.data)
 193                  for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
 194                      p_targ.data.mul_(1 - self.cfg.tau)
 195                      p_targ.data.add_(self.cfg.tau * p.data)
 196                  for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
 197                      p_targ.data.mul_(1 - self.cfg.tau)
 198                      p_targ.data.add_(self.cfg.tau * p.data)
 199              info.target_update_s = time.perf_counter() - t0
 200  
 201          self._update_counter += 1
 202          return info
 203  
 204
```
`train()` (lines 205–372):

```python
 205  def train(
 206      env,
 207      td3_cfg: TD3Config,
 208      exp_cfg: ExperimentConfig,
 209      tracker,
 210      device: torch.device,
 211      logger,
 212      steps_per_epoch: int = 1000,
 213  ):
 214      """
 215      Runs TD3 warmup + measured training. Returns:
 216        - agent: the trained TD3Agent
 217        - energy_log: dict of energy_consumed (kWh) per segment: rollout,
 218          buffer_sample, critic_update, actor_update, target_update, plus
 219          'warmup' (kept separate from the measured segments).
 220        - metrics: dict with "episodes" and "epochs" lists, same schema as
 221          sac.py's train() (see that module's docstring for field details;
 222          TD3 has no alpha/entropy term so "alpha_end" is simply omitted here).
 223  
 224      Warmup uses a purely random policy to fill the buffer, matching the
 225      paper (Section 6.1): 10,000 steps for HalfCheetah-v1/Ant-v1, 1,000 for
 226      other envs -- pass the right `--warmup-steps` on the CLI (this harness's
 227      `ExperimentConfig.warmup_steps` is shared across algorithms, so it isn't
 228      baked into TD3Config). After warmup, actions are the deterministic
 229      policy plus N(0, exploration_noise * act_limit) Gaussian noise, clipped
 230      to the action bounds (paper Section 6.1; Algorithm 1).
 231      """
 232      obs_dim = env.observation_space.shape[0]
 233      act_dim = env.action_space.shape[0]
 234      act_limit = float(env.action_space.high[0])
 235  
 236      agent = TD3Agent(obs_dim, act_dim, act_limit, td3_cfg, device)
 237      buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, act_dim)
 238  
 239      energy_log: Dict[str, float] = {}
 240      sub_time_totals = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
 241  
 242      episodes_log = []
 243      episode_idx = 0
 244      episode_return = 0.0
 245      episode_length = 0
 246  
 247      def _record_episode(phase, epoch, env_step):
 248          nonlocal episode_idx, episode_return, episode_length
 249          episodes_log.append({
 250              "episode_idx": episode_idx,
 251              "phase": phase,
 252              "epoch": epoch,
 253              "env_step": env_step,
 254              "return": episode_return,
 255              "length": episode_length,
 256          })
 257          episode_idx += 1
 258          episode_return = 0.0
 259          episode_length = 0
 260  
 261      def _explore_action(obs):
 262          action = agent.select_action(obs)
 263          noise = np.random.normal(0.0, td3_cfg.exploration_noise * act_limit, size=act_dim)
 264          return np.clip(action + noise, -act_limit, act_limit)
 265  
 266      obs, _ = env.reset(seed=exp_cfg.seed)
 267  
 268      # ---------------- warmup (random policy, fills buffer; excluded from analysis segments) ----------------
 269      logger.info("Starting warmup: %d steps", exp_cfg.warmup_steps)
 270      with TrackerTask(tracker, "warmup", 0, energy_log):
 271          for warmup_step in range(exp_cfg.warmup_steps):
 272              action = env.action_space.sample()
 273              next_obs, reward, terminated, truncated, _ = env.step(action)
 274              buffer.add(obs, action, reward, next_obs, terminated)
 275              episode_return += reward
 276              episode_length += 1
 277              obs = next_obs
 278              if terminated or truncated:
 279                  _record_episode("warmup", None, warmup_step + 1)
 280                  obs, _ = env.reset()
 281  
 282      # ---------------- measured training ----------------
 283      n_epochs = max(1, exp_cfg.train_steps // steps_per_epoch)
 284      logger.info("Starting measured training: %d epochs x %d steps = %d env steps",
 285                  n_epochs, steps_per_epoch, n_epochs * steps_per_epoch)
 286  
 287      epochs_log = []
 288      cumulative_reward = 0.0
 289      global_step = 0
 290  
 291      for epoch in range(n_epochs):
 292          # --- rollout block ---
 293          epoch_reward_sum = 0.0
 294          epoch_episode_returns = []
 295          with TrackerTask(tracker, "rollout", epoch, energy_log):
 296              for _ in range(steps_per_epoch):
 297                  action = _explore_action(obs)
 298                  next_obs, reward, terminated, truncated, _ = env.step(action)
 299                  buffer.add(obs, action, reward, next_obs, terminated)
 300                  episode_return += reward
 301                  episode_length += 1
 302                  cumulative_reward += reward
 303                  epoch_reward_sum += reward
 304                  global_step += 1
 305                  obs = next_obs
 306                  if terminated or truncated:
 307                      epoch_episode_returns.append(episode_return)
 308                      _record_episode("train", epoch, global_step)
 309                      obs, _ = env.reset()
 310  
 311          # --- gradient update block ---
 312          epoch_sub_times = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
 313          critic_losses, actor_losses = [], []
 314          n_updates = steps_per_epoch * td3_cfg.updates_per_env_step
 315          with TrackerTask(tracker, "gradient_updates", epoch, energy_log):
 316              for _ in range(n_updates):
 317                  if len(buffer) < td3_cfg.batch_size:
 318                      break
 319                  info = agent.update(buffer, td3_cfg.batch_size)
 320                  epoch_sub_times["buffer_sample"] += info.buffer_sample_s
 321                  epoch_sub_times["critic_update"] += info.critic_update_s
 322                  epoch_sub_times["actor_update"] += info.actor_update_s
 323                  epoch_sub_times["target_update"] += info.target_update_s
 324                  if info.critic_loss is not None:
 325                      critic_losses.append(info.critic_loss)
 326                  if info.actor_loss is not None:
 327                      actor_losses.append(info.actor_loss)
 328  
 329          # Sub-segment times accumulate across all epochs and are reconciled against the
 330          # total measured "gradient_updates" energy once, after the loop (see below) --
 331          # simpler than a per-epoch split and equally defensible under the constant-power
 332          # approximation this scheme relies on.
 333          for k in sub_time_totals:
 334              sub_time_totals[k] += epoch_sub_times[k]
 335  
 336          epochs_log.append({
 337              "epoch": epoch,
 338              "env_step": global_step,
 339              "cumulative_reward": cumulative_reward,
 340              "epoch_reward_sum": epoch_reward_sum,
 341              "epoch_reward_mean_per_step": epoch_reward_sum / steps_per_epoch,
 342              "num_episodes_completed": len(epoch_episode_returns),
 343              "mean_episode_return": (
 344                  sum(epoch_episode_returns) / len(epoch_episode_returns)
 345                  if epoch_episode_returns else None
 346              ),
 347              "critic_loss_mean": (sum(critic_losses) / len(critic_losses)) if critic_losses else None,
 348              "actor_loss_mean": (sum(actor_losses) / len(actor_losses)) if actor_losses else None,
 349              "buffer_size": len(buffer),
 350          })
 351  
 352          if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
 353              logger.info(
 354                  "Epoch %d/%d done (buffer size=%d, cumulative_reward=%.2f, mean_episode_return=%s)",
 355                  epoch + 1, n_epochs, len(buffer), cumulative_reward,
 356                  epochs_log[-1]["mean_episode_return"],
 357              )
 358  
 359      # record a trailing partial episode (if training ended mid-episode) so no reward is lost
 360      if episode_length > 0:
 361          _record_episode("train_incomplete", n_epochs - 1, global_step)
 362  
 363      # ---------------- reconcile: split total gradient_updates energy by aggregate time share ----------------
 364      total_gradient_energy = energy_log.pop("gradient_updates", 0.0)
 365      total_time = sum(sub_time_totals.values())
 366      for k, t in sub_time_totals.items():
 367          share = (t / total_time) if total_time > 0 else 0.0
 368          energy_log[k] = total_gradient_energy * share
 369  
 370      energy_log["_sub_segment_wall_time_seconds"] = sub_time_totals
 371      metrics = {"episodes": episodes_log, "epochs": epochs_log}
 372      return agent, energy_log, metrics
```
`algorithms/replay_buffer.py` (complete):

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

**2.3 Network sizes** — real `DeterministicPolicy` / `QNetwork` classes instantiated on CPU (parameter count = `sum(p.numel())`, weights + biases). TD3 holds six networks: actor, target actor, two critics, two target critics (`TD3Agent.__init__`):


| env | hidden | obs/act dim | actor | one Q-network | both critics | target actor | both target critics | trainable (actor + 2 Q) | all six networks |
|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | (256,256) | 17/6 | 71942 | 72193 | 144386 | 71942 | 144386 | 216328 | 432656 |
| HalfCheetah-v5 | (512,512) | 17/6 | 274950 | 275457 | 550914 | 274950 | 550914 | 825864 | 1651728 |
| HalfCheetah-v5 | (1024,1024) | 17/6 | 1074182 | 1075201 | 2150402 | 1074182 | 2150402 | 3224584 | 6449168 |
| Ant-v5 | (256,256) | 105/8 | 94984 | 95233 | 190466 | 94984 | 190466 | 285450 | 570900 |
| Ant-v5 | (512,512) | 105/8 | 321032 | 321537 | 643074 | 321032 | 643074 | 964106 | 1928212 |
| Ant-v5 | (1024,1024) | 105/8 | 1166344 | 1167361 | 2334722 | 1166344 | 2334722 | 3501066 | 7002132 |

- **PASS** — local environment dims equal the dims stored in flops_per_call.json (HC 17/6, Ant 105/8)

**2.4 Per-call FLOPs from `flop_analysis/flops_per_call.json`** (`measure_flops.measure_td3`, torch `FlopCounterMode` on the real `DeterministicPolicy`/`QNetwork`; measured on `cpu`; matmul FLOPs = 2·m·n·k; Adam / Polyak / clamp / noise operations are not matmuls and count 0 FLOPs). One column per constant; cell `HC ‖ Ant`:


| config | actor_forward_bs1 (rollout call) | critic_fwdbwd (per update() call) | actor_fwdbwd (per executed actor update) | target_update_elementwise_ops (per executed target update) | batch_size used for the measurement |
|---|---|---|---|---|---|
| base | 2.14426e+06 ‖ 2.32858e+06 | 1.92205e+09 ‖ 2.05107e+09 | 1.06906e+09 ‖ 1.14319e+09 | 6.44917e+06 ‖ 7.00213e+06 | 100 ‖ 100 |
| utd2 | 2.14426e+06 ‖ 2.32858e+06 | 1.92205e+09 ‖ 2.05107e+09 | 1.06906e+09 ‖ 1.14319e+09 | 6.44917e+06 ‖ 7.00213e+06 | 100 ‖ 100 |
| utd4 | 2.14426e+06 ‖ 2.32858e+06 | 1.92205e+09 ‖ 2.05107e+09 | 1.06906e+09 ‖ 1.14319e+09 | 6.44917e+06 ‖ 7.00213e+06 | 100 ‖ 100 |
| w256 | 142848 ‖ 188928 | 1.26618e+08 ‖ 1.58874e+08 | 7.06560e+07 ‖ 8.91904e+07 | 432656 ‖ 570900 | 100 ‖ 100 |
| w512 | 547840 ‖ 640000 | 4.89165e+08 ‖ 5.53677e+08 | 2.72384e+08 ‖ 3.09453e+08 | 1.65173e+06 ‖ 1.92821e+06 | 100 ‖ 100 |
| b256 | 2.14426e+06 ‖ 2.32858e+06 | 4.92044e+09 ‖ 5.25074e+09 | 2.73678e+09 ‖ 2.92658e+09 | 6.44917e+06 ‖ 7.00213e+06 | 256 ‖ 256 |
| b512 | 2.14426e+06 ‖ 2.32858e+06 | 9.84089e+09 ‖ 1.05015e+10 | 5.47357e+09 ‖ 5.85315e+09 | 6.44917e+06 ‖ 7.00213e+06 | 512 ‖ 512 |
| b1024 | 2.14426e+06 ‖ 2.32858e+06 | 1.96818e+10 ‖ 2.10030e+10 | 1.09471e+10 ‖ 1.17063e+10 | 6.44917e+06 ‖ 7.00213e+06 | 1024 ‖ 1024 |


**2.5 Analytic cross-check** (derived by this generator, not a pipeline output): dense-layer count mirroring the code path — forward 2·B·n_in·n_out per layer; backward = weight-grad + input-grad (input-grad omitted for the first layer of a network fed by data); critic = target (target-actor fwd + 2 target-Q fwd) + 2 × (Q fwd + Q bwd); actor = actor fwd + Q1 fwd + Q1 input-grad bwd (frozen weights, all layers incl. the action input of layer 1) + actor bwd; target ops = 2 × (actor + 2 Q params). Exact integers:


| env | signature | rollout measured | rollout analytic | critic measured | critic analytic | actor measured | actor analytic | target ops measured | target ops analytic | derived vs measured |
|---|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | `bs100_h1024x1024` | 2144256 | 2144256 | 1922048000 | 1922048000 | 1069056000 | 1069056000 | 6449168 | 6449168 | exact |
| HalfCheetah-v5 | `bs100_h256x256` | 142848 | 142848 | 126617600 | 126617600 | 70656000 | 70656000 | 432656 | 432656 | exact |
| HalfCheetah-v5 | `bs100_h512x512` | 547840 | 547840 | 489164800 | 489164800 | 272384000 | 272384000 | 1651728 | 1651728 | exact |
| HalfCheetah-v5 | `bs1024_h1024x1024` | 2144256 | 2144256 | 19681771520 | 19681771520 | 10947133440 | 10947133440 | 6449168 | 6449168 | exact |
| HalfCheetah-v5 | `bs256_h1024x1024` | 2144256 | 2144256 | 4920442880 | 4920442880 | 2736783360 | 2736783360 | 6449168 | 6449168 | exact |
| HalfCheetah-v5 | `bs512_h1024x1024` | 2144256 | 2144256 | 9840885760 | 9840885760 | 5473566720 | 5473566720 | 6449168 | 6449168 | exact |
| Ant-v5 | `bs100_h1024x1024` | 2328576 | 2328576 | 2051072000 | 2051072000 | 1143193600 | 1143193600 | 7002132 | 7002132 | exact |
| Ant-v5 | `bs100_h256x256` | 188928 | 188928 | 158873600 | 158873600 | 89190400 | 89190400 | 570900 | 570900 | exact |
| Ant-v5 | `bs100_h512x512` | 640000 | 640000 | 553676800 | 553676800 | 309452800 | 309452800 | 1928212 | 1928212 | exact |
| Ant-v5 | `bs1024_h1024x1024` | 2328576 | 2328576 | 21002977280 | 21002977280 | 11706302464 | 11706302464 | 7002132 | 7002132 | exact |
| Ant-v5 | `bs256_h1024x1024` | 2328576 | 2328576 | 5250744320 | 5250744320 | 2926575616 | 2926575616 | 7002132 | 7002132 | exact |
| Ant-v5 | `bs512_h1024x1024` | 2328576 | 2328576 | 10501488640 | 10501488640 | 5853151232 | 5853151232 | 7002132 | 7002132 | exact |

- **PASS** — analytic dense-layer counts equal the FlopCounterMode counts exactly for rollout, critic, actor and target ops in every (env, signature)

**2.6 Call counts and where they are defined.** Per run (`compute_energy_per_flop.sac_like_call_counts` and `rows_for_sac_or_td3(algo="td3", …)`, called from `main()`):

```
n_epochs = max(1, train_steps // steps_per_epoch)               # = 100
n_env    = n_epochs * steps_per_epoch                           # = 100000 (warmup steps are NOT counted)
n_upd    = n_env * updates_per_env_step                         # update() calls = buffer_sample calls = critic_update calls
n_act    = ceil(n_upd / policy_update_delay)                    # executed actor updates (delay = 2)
target_update calls = n_act                                     # TD3 only (SAC: n_upd)
F_rollout = actor_forward_bs1 * n_env
F_critic  = critic_fwdbwd    * n_upd        # per update() call
F_actor   = actor_fwdbwd     * n_act        # per EXECUTED actor update
OPS_target = target_update_elementwise_ops * n_act
```
The loop that issues the calls is `train()` (`algorithms/td3.py`): `n_updates = steps_per_epoch * updates_per_env_step` calls of `agent.update()` per epoch (lines near `for _ in range(n_updates)`), breaking only if `len(buffer) < batch_size` (never the case: buffer ≥ 10000 > 1024 after the 10000 warmup steps). The delayed branch is `do_delayed_update = (self._update_counter % policy_update_delay) == 0` (line 170), `_update_counter` starts at 0 and increments once per `update()` call, so the actor and target updates run on update() calls 0, 2, 4, … (ceil(n_upd/2) times per run). Call counts per configuration (mean over seeds; identical across seeds):


| config | calls rollout | calls buf_sample | calls critic | calls actor | calls target |
|---|---|---|---|---|---|
| base | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 50000 ‖ 50000 | 50000 ‖ 50000 |
| utd2 | 100000 ‖ 100000 | 200000 ‖ 200000 | 200000 ‖ 200000 | 100000 ‖ 100000 | 100000 ‖ 100000 |
| utd4 | 100000 ‖ 100000 | 400000 ‖ 400000 | 400000 ‖ 400000 | 200000 ‖ 200000 | 200000 ‖ 200000 |
| w256 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 50000 ‖ 50000 | 50000 ‖ 50000 |
| w512 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 50000 ‖ 50000 | 50000 ‖ 50000 |
| b256 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 50000 ‖ 50000 | 50000 ‖ 50000 |
| b512 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 50000 ‖ 50000 | 50000 ‖ 50000 |
| b1024 | 100000 ‖ 100000 | 100000 ‖ 100000 | 100000 ‖ 100000 | 50000 ‖ 50000 | 50000 ‖ 50000 |

- **PASS** — call counts equal the `call_count` column of per_run_energy_per_flop.csv for all 80 runs
- The actual number of `update()` calls is **not logged**; it follows from the loop (indirect evidence: every run has 100 epochs with `buffer_size` = warmup + 1000·(epoch+1) and a non-null `critic_loss_mean` and `actor_loss_mean` in every epoch).
- **PASS** — buffer_size per epoch = warmup + 1000(e+1) and critic/actor loss logged in all 100 epochs of every run

**2.7 Total FLOPs per run and segment** (call count × per-call FLOPs; `target` = elementwise Polyak operations, not FLOPs; `GU` = critic + actor; `TOTAL` = rollout + critic + actor; `buffer_sample` = 0 FLOPs):


| config | F rollout | F critic | F actor | F target [elementwise ops] | F GU | F TOTAL |
|---|---|---|---|---|---|---|
| base | 2.14426e+11 ‖ 2.32858e+11 | 1.92205e+14 ‖ 2.05107e+14 | 5.34528e+13 ‖ 5.71597e+13 | 3.22458e+11 ‖ 3.50107e+11 | 2.45658e+14 ‖ 2.62267e+14 | 2.45872e+14 ‖ 2.62500e+14 |
| utd2 | 2.14426e+11 ‖ 2.32858e+11 | 3.84410e+14 ‖ 4.10214e+14 | 1.06906e+14 ‖ 1.14319e+14 | 6.44917e+11 ‖ 7.00213e+11 | 4.91315e+14 ‖ 5.24534e+14 | 4.91530e+14 ‖ 5.24767e+14 |
| utd4 | 2.14426e+11 ‖ 2.32858e+11 | 7.68819e+14 ‖ 8.20429e+14 | 2.13811e+14 ‖ 2.28639e+14 | 1.28983e+12 ‖ 1.40043e+12 | 9.82630e+14 ‖ 1.04907e+15 | 9.82845e+14 ‖ 1.04930e+15 |
| w256 | 1.42848e+10 ‖ 1.88928e+10 | 1.26618e+13 ‖ 1.58874e+13 | 3.53280e+12 ‖ 4.45952e+12 | 2.16328e+10 ‖ 2.85450e+10 | 1.61946e+13 ‖ 2.03469e+13 | 1.62088e+13 ‖ 2.03658e+13 |
| w512 | 5.47840e+10 ‖ 6.40000e+10 | 4.89165e+13 ‖ 5.53677e+13 | 1.36192e+13 ‖ 1.54726e+13 | 8.25864e+10 ‖ 9.64106e+10 | 6.25357e+13 ‖ 7.08403e+13 | 6.25905e+13 ‖ 7.09043e+13 |
| b256 | 2.14426e+11 ‖ 2.32858e+11 | 4.92044e+14 ‖ 5.25074e+14 | 1.36839e+14 ‖ 1.46329e+14 | 3.22458e+11 ‖ 3.50107e+11 | 6.28883e+14 ‖ 6.71403e+14 | 6.29098e+14 ‖ 6.71636e+14 |
| b512 | 2.14426e+11 ‖ 2.32858e+11 | 9.84089e+14 ‖ 1.05015e+15 | 2.73678e+14 ‖ 2.92658e+14 | 3.22458e+11 ‖ 3.50107e+11 | 1.25777e+15 ‖ 1.34281e+15 | 1.25798e+15 ‖ 1.34304e+15 |
| b1024 | 2.14426e+11 ‖ 2.32858e+11 | 1.96818e+15 ‖ 2.10030e+15 | 5.47357e+14 ‖ 5.85315e+14 | 3.22458e+11 ‖ 3.50107e+11 | 2.51553e+15 ‖ 2.68561e+15 | 2.51575e+15 ‖ 2.68585e+15 |

Ant / HalfCheetah ratios of the same quantities:


| config | Ant/HC rollout | Ant/HC critic | Ant/HC actor | Ant/HC target | Ant/HC GU | Ant/HC TOTAL |
|---|---|---|---|---|---|---|
| base | 1.08596 | 1.06713 | 1.06935 | 1.08574 | 1.06761 | 1.06763 |
| utd2 | 1.08596 | 1.06713 | 1.06935 | 1.08574 | 1.06761 | 1.06762 |
| utd4 | 1.08596 | 1.06713 | 1.06935 | 1.08574 | 1.06761 | 1.06762 |
| w256 | 1.32258 | 1.25475 | 1.26232 | 1.31952 | 1.25640 | 1.25646 |
| w512 | 1.16822 | 1.13188 | 1.13609 | 1.16739 | 1.13280 | 1.13283 |
| b256 | 1.08596 | 1.06713 | 1.06935 | 1.08574 | 1.06761 | 1.06762 |
| b512 | 1.08596 | 1.06713 | 1.06935 | 1.08574 | 1.06761 | 1.06761 |
| b1024 | 1.08596 | 1.06713 | 1.06935 | 1.08574 | 1.06761 | 1.06761 |

The factor of each configuration against its sweep baseline is given in the 'FLOP factors' table of each sweep section (Sections 3–4).

## 3. Baseline decomposition and energy per FLOP (4.2.1)

Configuration `base` (TD3 paper hyperparameters, hidden (1024,1024), batch 100, UTD 1, warmup 10000). Per-environment values, side by side.

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|
| base | 1410.91 ± 18.4254 ‖ 3071.46 ± 52.5339 | 1295.43 ± 26.2671 ‖ 1473.47 ± 26.1945 | 18530.1 ± 279.872 ‖ 18957.5 ± 337.485 | 5182.75 ± 53.5812 ‖ 5284.10 ± 73.8102 | 1692.28 ± 24.7333 ‖ 1733.25 ± 20.8448 | 26700.5 ± 301.074 ‖ 27448.4 ± 408.760 | 28111.5 ± 310.708 ‖ 30519.8 ± 431.915 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| base | 0.00780874 ± 8.63078e-05 ‖ 0.00847773 ± 1.19976e-04 | 34.8972 ± 1.32492 ‖ 153.158 ± 5.86734 | 6256.39 ± 361.457 ‖ 6392.38 ± 490.180 | 6637.29 ± 281.479 ‖ 6818.27 ± 283.203 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|
| base | 5.01916 ± 0.0588648 ‖ 10.0645 ± 0.161751 | 4.60894 ± 0.120563 ‖ 4.82836 ± 0.0887409 | 65.9143 ± 0.314294 ‖ 62.1133 ± 0.348088 | 18.4366 ± 0.0384725 ‖ 17.3137 ± 0.00998654 | 6.02101 ± 0.140385 ‖ 5.68012 ± 0.112316 | 94.9808 ± 0.0588648 ‖ 89.9355 ± 0.161751 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| base | 4.85251 ± 0.127788 ‖ 5.36883 ± 0.107654 | 69.3974 ± 0.311788 ‖ 69.0640 ± 0.268809 | 19.4108 ± 0.0479216 ‖ 19.2513 ± 0.0356577 | 6.33921 ± 0.149154 ‖ 6.31594 ± 0.134610 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|
| base | 6.57995e-09 ± 8.59292e-11 ‖ 1.31903e-08 ± 2.25605e-10 | 9.64080e-11 ± 1.45611e-12 ‖ 9.24275e-11 ± 1.64541e-12 | 9.69593e-11 ± 1.00240e-12 ‖ 9.24445e-11 ± 1.29130e-12 | 1.08690e-10 ± 1.22558e-12 ‖ 1.04658e-10 ± 1.55857e-12 | 1.14334e-10 ± 1.26370e-12 ‖ 1.16266e-10 ± 1.64539e-12 | 5.24807 ± 0.0767024 ‖ 4.95065 ± 0.0595384 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| base | 2.00462 | 0.958711 | 0.953435 | 0.962904 | 1.01690 | 0.943328 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|
| base | 11.1119 ± 0.0596187 ‖ 24.7473 ± 0.178141 | 195.987 ± 4.59564 ‖ 199.380 ± 3.54820 | 8.87991 ± 0.0441167 ‖ 9.85333 ± 0.0615825 | 127.075 ± 3.89160 ‖ 126.797 ± 2.93413 | 35.5388 ± 0.853585 ‖ 35.3410 ± 0.620329 | 11.6014 ± 0.0938117 ‖ 11.5912 ± 0.0527052 | 207.099 ± 4.64914 ‖ 224.127 ± 3.53526 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|
| base | 126.974 ± 1.58927 ‖ 124.115 ± 2.06670 | 136.273 ± 2.18738 ‖ 137.683 ± 1.75444 | 135.772 ± 2.09392 ‖ 136.183 ± 1.73524 | 105.944 ± 1.05964 ‖ 108.231 ± 1.46703 | 0.931908 ± 0.0158865 ‖ 0.901446 ± 0.00866638 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|
| base | 19.2974 ± 0.103223 ‖ 9.40980 ± 0.0676047 | 1253.98 ± 28.8906 ‖ 1315.75 ± 23.5499 | 1187.69 ± 26.2133 ‖ 1171.44 ± 18.5841 | 1513.64 ± 45.3034 ‖ 1618.30 ± 37.7247 | 1504.75 ± 35.4138 ‖ 1617.78 ± 28.5573 | 93.4174 ± 0.243312 ‖ 92.0747 ± 0.142527 | 93.1600–93.7995 ‖ 91.8827–92.2302 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 25.5248 ± 0.309549 ‖ 27.5657 ± 0.436376 | 58.7534 ± 0.500385 ‖ 56.3345 ± 0.696456 | 15.7217 ± 0.195131 ‖ 16.0998 ± 0.268080 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 20.0463 ± 0.175344 ‖ 20.1816 ± 0.123170 | 65.2771 ± 0.408611 ‖ 65.2934 ± 0.271919 | 14.6766 ± 0.237438 ‖ 14.5251 ± 0.185325 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|
| base | 56.4211 ± 1.03774 ‖ 59.1376 ± 1.43952 | 52.5691 ± 0.622384 ‖ 53.3031 ± 1.04567 | 52.7623 ± 0.614592 ‖ 53.8902 ± 1.07300 | 71.6287 ± 0.699504 ‖ 73.3896 ± 1.72627 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| base | 1.08567 | 2.17694 | 5.04534 | rollout | 1.00303 | 0.967471 | 1.01690 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|
| base | 2.17694 (paired sd 0.0466529) | 1.13744 (paired sd 0.0389360) | 1.02307 (paired sd 0.0206497) | 1.01955 (paired sd 0.0172043) | 1.02421 (paired sd 0.0204510) | 1.02801 (paired sd 0.0178265) | 1.08567 (paired sd 0.0174699) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|
| base | 5.04534 ± 0.202808 (5/5 same sign) | 0.219419 ± 0.139882 (5/5 same sign) | -3.80098 ± 0.471083 (5/5 same sign) | -1.12288 ± 0.0401157 (5/5 same sign) | -0.340890 ± 0.145388 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| base | mean Δ 5.04534; 2/32 patterns; p = 0.0625000 | mean Δ 2408.37; 2/32 patterns; p = 0.0625000 |


Excluded phases and per-seed totals of all 80 runs: `contexts/Section_4.2_data/td3_segments_long.csv` (all segments incl. warmup and idle, per run) and `td3_cross_seed_summary.csv`.


## 4.1 Sweep: UTD — updates_per_env_step ∈ {1,2,4}

Configurations: `base` (hidden (1024,1024), batch 100 (paper default), UTD 1, policy delay 2), `utd2` (UTD 2), `utd4` (UTD 4). Baseline for the response tables: `base`.

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|
| base | 1410.91 ± 18.4254 ‖ 3071.46 ± 52.5339 | 1295.43 ± 26.2671 ‖ 1473.47 ± 26.1945 | 18530.1 ± 279.872 ‖ 18957.5 ± 337.485 | 5182.75 ± 53.5812 ‖ 5284.10 ± 73.8102 | 1692.28 ± 24.7333 ‖ 1733.25 ± 20.8448 | 26700.5 ± 301.074 ‖ 27448.4 ± 408.760 | 28111.5 ± 310.708 ‖ 30519.8 ± 431.915 |
| utd2 | 1431.60 ± 25.9036 ‖ 3031.22 ± 58.5223 | 2630.09 ± 8.55513 ‖ 2949.52 ± 13.1351 | 37419.5 ± 352.301 ‖ 37577.8 ± 305.781 | 10466.8 ± 59.1597 ‖ 10505.6 ± 62.8497 | 3412.99 ± 18.8168 ‖ 3478.44 ± 33.5393 | 53929.4 ± 413.470 ‖ 54511.4 ± 353.829 | 55361.0 ± 397.478 ‖ 57542.6 ± 361.024 |
| utd4 | 1380.45 ± 11.9539 ‖ 3014.92 ± 36.8573 | 5214.36 ± 46.6765 ‖ 5838.23 ± 31.2564 | 73363.4 ± 803.512 ‖ 75164.3 ± 1338.01 | 20567.4 ± 158.075 ‖ 20961.7 ± 266.483 | 6819.28 ± 77.9142 ‖ 6920.61 ± 72.7272 | 105964 ± 1045.26 ‖ 108885 ± 1602.62 | 107345 ± 1044.85 ‖ 111900 ± 1590.48 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| base | 0.00780874 ± 8.63078e-05 ‖ 0.00847773 ± 1.19976e-04 | 34.8972 ± 1.32492 ‖ 153.158 ± 5.86734 | 6256.39 ± 361.457 ‖ 6392.38 ± 490.180 | 6637.29 ± 281.479 ‖ 6818.27 ± 283.203 |
| utd2 | 0.0153781 ± 1.10410e-04 ‖ 0.0159841 ± 1.00284e-04 | 34.3837 ± 0.769551 ‖ 153.653 ± 3.32711 | 6162.75 ± 62.0530 ‖ 5998.75 ± 45.1712 | 6641.93 ± 69.5339 ‖ 6540.26 ± 70.0748 |
| utd4 | 0.0298180 ± 2.90235e-04 ‖ 0.0310832 ± 4.41799e-04 | 34.7022 ± 0.380418 ‖ 154.195 ± 3.59729 | 6167.87 ± 107.022 ‖ 5989.39 ± 65.4730 | 6350.55 ± 67.1820 ‖ 6238.42 ± 38.9963 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|
| base | 5.01916 ± 0.0588648 ‖ 10.0645 ± 0.161751 | 4.60894 ± 0.120563 ‖ 4.82836 ± 0.0887409 | 65.9143 ± 0.314294 ‖ 62.1133 ± 0.348088 | 18.4366 ± 0.0384725 ‖ 17.3137 ± 0.00998654 | 6.02101 ± 0.140385 ‖ 5.68012 ± 0.112316 | 94.9808 ± 0.0588648 ‖ 89.9355 ± 0.161751 |
| utd2 | 2.58620 ± 0.0601155 ‖ 5.26784 ± 0.100408 | 4.75103 ± 0.0432505 ‖ 5.12603 ± 0.0491089 | 67.5910 ± 0.169153 ‖ 65.3037 ± 0.160014 | 18.9066 ± 0.0297969 ‖ 18.2572 ± 0.0372261 | 6.16514 ± 0.0426333 ‖ 6.04517 ± 0.0697347 | 97.4138 ± 0.0601155 ‖ 94.7322 ± 0.100408 |
| utd4 | 1.28610 ± 0.0169319 ‖ 2.69486 ± 0.0580217 | 4.85773 ± 0.0394820 ‖ 5.21832 ± 0.0866289 | 68.3430 ± 0.115881 ‖ 67.1685 ± 0.259064 | 19.1604 ± 0.0504578 ‖ 18.7329 ± 0.0411146 | 6.35271 ± 0.0435987 ‖ 6.18550 ± 0.0980954 | 98.7139 ± 0.0169319 ‖ 97.3051 ± 0.0580217 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| base | 4.85251 ± 0.127788 ‖ 5.36883 ± 0.107654 | 69.3974 ± 0.311788 ‖ 69.0640 ± 0.268809 | 19.4108 ± 0.0479216 ‖ 19.2513 ± 0.0356577 | 6.33921 ± 0.149154 ‖ 6.31594 ± 0.134610 |
| utd2 | 4.87718 ± 0.0471608 ‖ 5.41110 ± 0.0542544 | 69.3854 ± 0.131601 ‖ 68.9351 ± 0.128783 | 19.4086 ± 0.0415109 ‖ 19.2725 ± 0.0255887 | 6.32884 ± 0.0475959 ‖ 6.38136 ± 0.0760951 |
| utd4 | 4.92102 ± 0.0404782 ‖ 5.36288 ± 0.0919762 | 69.2334 ± 0.106782 ‖ 69.0286 ± 0.228985 | 19.4101 ± 0.0536939 ‖ 19.2517 ± 0.0470502 | 6.43548 ± 0.0448588 ‖ 6.35685 ± 0.104236 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|
| base | 6.57995e-09 ± 8.59292e-11 ‖ 1.31903e-08 ± 2.25605e-10 | 9.64080e-11 ± 1.45611e-12 ‖ 9.24275e-11 ± 1.64541e-12 | 9.69593e-11 ± 1.00240e-12 ‖ 9.24445e-11 ± 1.29130e-12 | 1.08690e-10 ± 1.22558e-12 ‖ 1.04658e-10 ± 1.55857e-12 | 1.14334e-10 ± 1.26370e-12 ‖ 1.16266e-10 ± 1.64539e-12 | 5.24807 ± 0.0767024 ‖ 4.95065 ± 0.0595384 |
| utd2 | 6.67642e-09 ± 1.20805e-10 ‖ 1.30175e-08 ± 2.51322e-10 | 9.73428e-11 ± 9.16472e-13 ‖ 9.16052e-11 ± 7.45418e-13 | 9.79070e-11 ± 5.53383e-13 ‖ 9.18974e-11 ± 5.49773e-13 | 1.09765e-10 ± 8.41557e-13 ‖ 1.03924e-10 ± 6.74558e-13 | 1.12630e-10 ± 8.08655e-13 ‖ 1.09654e-10 ± 6.87971e-13 | 5.29214 ± 0.0291771 ‖ 4.96768 ± 0.0478987 |
| utd4 | 6.43791e-09 ± 5.57484e-11 ‖ 1.29475e-08 ± 1.58283e-10 | 9.54235e-11 ± 1.04513e-12 ‖ 9.16158e-11 ± 1.63087e-12 | 9.61941e-11 ± 7.39321e-13 ‖ 9.16802e-11 ± 1.16552e-12 | 1.07838e-10 ± 1.06373e-12 ‖ 1.03792e-10 ± 1.52767e-12 | 1.09219e-10 ± 1.06308e-12 ‖ 1.06642e-10 ± 1.51575e-12 | 5.28695 ± 0.0604064 ‖ 4.94179 ± 0.0519322 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| base | 2.00462 | 0.958711 | 0.953435 | 0.962904 | 1.01690 | 0.943328 |
| utd2 | 1.94977 | 0.941058 | 0.938619 | 0.946778 | 0.973574 | 0.938691 |
| utd4 | 2.01113 | 0.960097 | 0.953075 | 0.962485 | 0.976411 | 0.934715 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|
| base | 11.1119 ± 0.0596187 ‖ 24.7473 ± 0.178141 | 195.987 ± 4.59564 ‖ 199.380 ± 3.54820 | 8.87991 ± 0.0441167 ‖ 9.85333 ± 0.0615825 | 127.075 ± 3.89160 ‖ 126.797 ± 2.93413 | 35.5388 ± 0.853585 ‖ 35.3410 ± 0.620329 | 11.6014 ± 0.0938117 ‖ 11.5912 ± 0.0527052 | 207.099 ± 4.64914 ‖ 224.127 ± 3.53526 |
| utd2 | 11.1329 ± 0.129701 ‖ 24.7527 ± 0.265030 | 392.360 ± 4.18440 ‖ 392.998 ± 3.65081 | 17.8543 ± 0.0402923 ‖ 19.5678 ± 0.0765127 | 254.031 ± 3.29469 ‖ 249.306 ± 2.72629 | 71.0555 ± 0.647853 ‖ 69.6983 ± 0.613312 | 23.1693 ± 0.162368 ‖ 23.0770 ± 0.241894 | 403.493 ± 4.07516 ‖ 417.751 ± 3.77943 |
| utd4 | 11.0858 ± 0.198639 ‖ 24.7860 ± 0.179385 | 780.635 ± 9.14925 ‖ 795.267 ± 14.9782 | 35.8054 ± 0.317285 ‖ 39.2697 ± 0.146908 | 503.780 ± 7.03570 ‖ 505.619 ± 11.4059 | 141.233 ± 1.45649 ‖ 141.003 ± 2.43210 | 46.8261 ± 0.550783 ‖ 46.5501 ± 0.464083 | 791.721 ± 9.31101 ‖ 820.053 ± 15.1310 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|
| base | 126.974 ± 1.58927 ‖ 124.115 ± 2.06670 | 136.273 ± 2.18738 ‖ 137.683 ± 1.75444 | 135.772 ± 2.09392 ‖ 136.183 ± 1.73524 | 105.944 ± 1.05964 ‖ 108.231 ± 1.46703 | 0.931908 ± 0.0158865 ‖ 0.901446 ± 0.00866638 |
| utd2 | 128.584 ± 0.932558 ‖ 122.460 ± 1.97046 | 137.453 ± 0.462909 ‖ 138.709 ± 0.395392 | 137.208 ± 0.461240 ‖ 137.747 ± 0.394001 | 116.821 ± 0.245142 ‖ 117.234 ± 0.190316 | 0.935476 ± 0.00547910 ‖ 0.882857 ± 0.0142382 |
| utd4 | 124.553 ± 2.21840 ‖ 121.642 ± 1.64726 | 135.745 ± 0.579682 ‖ 136.925 ± 0.616893 | 135.588 ± 0.587368 ‖ 136.463 ± 0.635429 | 123.350 ± 0.405669 ‖ 124.112 ± 0.373423 | 0.917540 ± 0.0148078 ‖ 0.888365 ± 0.00891308 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|
| base | 19.2974 ± 0.103223 ‖ 9.40980 ± 0.0676047 | 1253.98 ± 28.8906 ‖ 1315.75 ± 23.5499 | 1187.69 ± 26.2133 ‖ 1171.44 ± 18.5841 | 1513.64 ± 45.3034 ‖ 1618.30 ± 37.7247 | 1504.75 ± 35.4138 ‖ 1617.78 ± 28.5573 | 93.4174 ± 0.243312 ‖ 92.0747 ± 0.142527 | 93.1600–93.7995 ‖ 91.8827–92.2302 |
| utd2 | 19.2626 ± 0.223772 ‖ 9.40823 ± 0.100683 | 1252.32 ± 13.5001 ‖ 1334.79 ± 12.3514 | 1218.29 ± 12.4306 ‖ 1256.25 ± 11.3330 | 1513.45 ± 19.8903 ‖ 1645.58 ± 17.8761 | 1504.64 ± 13.8389 ‖ 1640.30 ± 14.3283 | 93.3094 ± 0.0494626 ‖ 92.0230 ± 0.0830721 | 93.2289–93.3537 ‖ 91.8752–92.0700 |
| utd4 | 19.3473 ± 0.340485 ‖ 9.39511 ± 0.0676102 | 1258.90 ± 14.6563 ‖ 1319.51 ± 24.5901 | 1241.54 ± 14.4991 ‖ 1279.90 ± 23.3680 | 1526.34 ± 21.1585 ‖ 1623.28 ± 36.2529 | 1514.02 ± 15.5868 ‖ 1621.90 ± 27.7799 | 93.2111 ± 0.0909142 ‖ 92.0994 ± 0.104882 | 93.1140–93.3500 ‖ 91.9764–92.2616 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 25.5248 ± 0.309549 ‖ 27.5657 ± 0.436376 | 58.7534 ± 0.500385 ‖ 56.3345 ± 0.696456 | 15.7217 ± 0.195131 ‖ 16.0998 ± 0.268080 | 500/500 ‖ 500/500 |
| utd2 | 25.2201 ± 0.370363 ‖ 27.3352 ± 0.438119 | 59.2653 ± 0.399160 ‖ 56.3468 ± 0.679872 | 15.5145 ± 0.115575 ‖ 16.3180 ± 0.261436 | 500/500 ‖ 500/500 |
| utd4 | 25.6489 ± 0.398024 ‖ 27.5516 ± 0.300413 | 58.3270 ± 0.646350 ‖ 56.0220 ± 0.505175 | 16.0241 ± 0.287905 ‖ 16.4264 ± 0.224612 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 20.0463 ± 0.175344 ‖ 20.1816 ± 0.123170 | 65.2771 ± 0.408611 ‖ 65.2934 ± 0.271919 | 14.6766 ± 0.237438 ‖ 14.5251 ± 0.185325 | 0/500 ‖ 0/500 |
| utd2 | 20.0786 ± 0.140589 ‖ 19.8939 ± 0.0679968 | 65.3725 ± 0.103967 ‖ 65.6889 ± 0.0643077 | 14.5489 ± 0.0488571 ‖ 14.4172 ± 0.0411044 | 0/500 ‖ 0/500 |
| utd4 | 20.2887 ± 0.0926446 ‖ 20.1232 ± 0.0999049 | 64.9784 ± 0.0839242 ‖ 65.2708 ± 0.116290 | 14.7329 ± 0.0631156 ‖ 14.6060 ± 0.0658471 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|
| base | 56.4211 ± 1.03774 ‖ 59.1376 ± 1.43952 | 52.5691 ± 0.622384 ‖ 53.3031 ± 1.04567 | 52.7623 ± 0.614592 ‖ 53.8902 ± 1.07300 | 71.6287 ± 0.699504 ‖ 73.3896 ± 1.72627 |
| utd2 | 55.3246 ± 0.669501 ‖ 56.8954 ± 1.01004 | 51.7519 ± 0.322469 ‖ 50.2192 ± 0.154340 | 51.8442 ± 0.330105 ‖ 50.5702 ± 0.181850 | 71.1340 ± 0.440332 ‖ 69.6586 ± 0.263133 |
| utd4 | 55.8449 ± 0.763064 ‖ 55.8513 ± 0.730795 | 51.2321 ± 0.433016 ‖ 49.6114 ± 0.232806 | 51.2914 ± 0.427185 ‖ 49.7793 ± 0.240348 | 69.5436 ± 0.470916 ‖ 67.9295 ± 0.291186 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|
| utd2 | 1.46098 ± 0.539624 ‖ -1.28332 ± 2.70570 | 103.093 ± 4.07236 ‖ 100.216 ± 3.00083 | 101.996 ± 4.76959 ‖ 98.2928 ± 5.10674 | 101.979 ± 3.02770 ‖ 98.8604 ± 3.95914 | 101.708 ± 2.66287 ‖ 100.711 ± 3.07015 | 102.011 ± 3.64787 ‖ 98.6465 ± 4.24583 | 96.9635 ± 3.42649 ‖ 88.5853 ± 3.85629 |
| utd4 | -2.13935 ± 1.95569 ‖ -1.82158 ± 1.83398 | 302.645 ± 8.54599 ‖ 296.337 ± 8.26139 | 295.994 ± 7.79251 ‖ 296.686 ± 14.1155 | 296.878 ± 5.15492 ‖ 296.807 ± 10.4124 | 303.029 ± 7.19558 ‖ 299.351 ± 7.73772 | 296.901 ± 5.74856 ‖ 296.827 ± 11.7059 | 281.891 ± 5.50490 ‖ 266.760 ± 10.2944 |

Duration:

| config | D rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|
| utd2 | 0.189452 ± 1.01700 ‖ 0.0210785 ± 0.676474 | 100.317 ± 6.48830 ‖ 97.1747 ± 4.80020 | 94.9378 ± 6.04206 ‖ 86.4393 ± 4.15922 | 101.068 ± 1.05711 ‖ 98.5942 ± 0.769345 | 100.105 ± 8.20411 ‖ 96.7265 ± 6.12225 | 100.059 ± 6.34720 ‖ 97.2821 ± 4.85493 | 99.7283 ± 2.75573 ‖ 99.0947 ± 2.39424 |
| utd4 | -0.230945 ± 1.98026 ‖ 0.160856 ± 1.06490 | 298.533 ± 12.5120 ‖ 299.058 ± 13.9739 | 282.490 ± 11.6484 ‖ 266.031 ± 12.0413 | 303.215 ± 2.39028 ‖ 298.559 ± 3.50711 | 296.816 ± 15.8393 ‖ 299.075 ± 17.6221 | 297.637 ± 12.4444 ‖ 299.153 ± 13.2368 | 303.651 ± 6.12073 ‖ 301.599 ± 3.60946 |

Mean power:

| config | P rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|
| utd2 | 1.27340 ± 0.559484 ‖ -1.30861 ± 2.41672 | 0.888080 ± 1.77202 ‖ 0.757368 ± 1.18333 | 1.07775 ± 1.69333 ‖ 1.15967 ± 1.18389 | 10.2761 ± 1.16633 ‖ 8.33358 ± 1.37392 |
| utd4 | -1.89124 ± 2.31611 ‖ -1.97457 ± 1.86190 | -0.363244 ± 1.92700 ‖ -0.536867 ± 1.42452 | -0.113172 ± 1.88026 ‖ 0.218938 ± 1.41900 | 16.4404 ± 1.42616 ‖ 14.6897 ± 1.53768 |

Achieved throughput:

| config | rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|
| utd2 | -0.180809 ± 1.02024 ‖ -0.0174104 ± 0.677131 | -0.0713564 ± 3.34555 ‖ 1.48032 ± 2.43647 | 2.63348 ± 3.28238 ‖ 7.26806 ± 2.36104 |
| utd4 | 0.262469 ± 1.95175 ‖ -0.151659 ± 1.05101 | 0.449255 ± 3.23391 ‖ 0.332547 ± 3.44305 | 4.58906 ± 3.26302 ‖ 9.30049 ± 3.52464 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| utd2 | 1.46098 ± 0.539624 ‖ -1.28332 ± 2.70570 | 0.998042 ± 2.38480 ‖ -0.853602 ± 2.55337 | 0.989529 ± 1.51385 ‖ -0.569815 ± 1.97957 | 1.00535 ± 1.82393 ‖ -0.676771 ± 2.12291 | -1.47528 ± 1.71399 ‖ -5.66552 ± 1.92900 | 0.854214 ± 1.33143 ‖ 0.355527 ± 1.53508 |
| utd4 | -2.13935 ± 1.95569 ‖ -1.82158 ± 1.83398 | -1.00139 ± 1.94813 ‖ -0.828500 ± 3.52887 | -0.780545 ± 1.28873 ‖ -0.798279 ± 2.60311 | -0.774784 ± 1.43714 ‖ -0.793272 ± 2.92647 | -4.46479 ± 1.37713 ‖ -8.24894 ± 2.57531 | 0.757206 ± 1.79890 ‖ -0.162339 ± 1.93443 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|
| utd2 | 1.00000 ‖ 1.00000 | 2.00000 ‖ 2.00000 | 2.00000 ‖ 2.00000 | 2.00000 ‖ 2.00000 | 1.99913 ‖ 1.99911 | 2.00000 ‖ 2.00000 |
| utd4 | 1.00000 ‖ 1.00000 | 4.00000 ‖ 4.00000 | 4.00000 ‖ 4.00000 | 4.00000 ‖ 4.00000 | 3.99738 ‖ 3.99734 | 4.00000 ‖ 4.00000 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|
| utd2 | -2.43296 ± 0.0677375 ‖ -4.79666 ± 0.197136 | 0.142093 ± 0.156114 ‖ 0.297678 ± 0.109783 | 1.67667 ± 0.439041 ‖ 3.19039 ± 0.410495 | 0.470061 ± 0.0607060 ‖ 0.943537 ± 0.0442006 | 0.144131 ± 0.172341 ‖ 0.365052 ± 0.159601 | 2.43296 ± 0.0677375 ‖ 4.79666 ± 0.197136 |
| utd4 | -3.73306 ± 0.0643340 ‖ -7.36964 ± 0.190465 | 0.248794 ± 0.157363 ‖ 0.389960 ± 0.158180 | 2.42871 ± 0.411181 ‖ 5.05511 ± 0.575858 | 0.723851 ± 0.0761092 ‖ 1.41918 ± 0.0401098 | 0.331702 ± 0.173911 ‖ 0.505382 ± 0.202549 | 3.73306 ± 0.0643340 ‖ 7.36964 ± 0.190465 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| base | 1.08567 | 2.17694 | 5.04534 | rollout | 1.00303 | 0.967471 | 1.01690 |
| utd2 | 1.03941 | 2.11737 | 2.68164 | rollout | 1.00393 | 0.943745 | 0.973574 |
| utd4 | 1.04243 | 2.18401 | 1.40876 | rollout | 1.00646 | 0.968212 | 0.976411 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|
| base | 2.17694 (paired sd 0.0466529) | 1.13744 (paired sd 0.0389360) | 1.02307 (paired sd 0.0206497) | 1.01955 (paired sd 0.0172043) | 1.02421 (paired sd 0.0204510) | 1.02801 (paired sd 0.0178265) | 1.08567 (paired sd 0.0174699) |
| utd2 | 2.11737 (paired sd 0.0768173) | 1.12145 (paired sd 0.00727693) | 1.00423 (paired sd 0.00923664) | 1.00371 (paired sd 0.00677149) | 1.01918 (paired sd 0.0104435) | 1.01079 (paired sd 0.00731424) | 1.03941 (paired sd 0.00623654) |
| utd4 | 2.18401 (paired sd 0.0409296) | 1.11965 (paired sd 0.0119522) | 1.02455 (paired sd 0.00852201) | 1.01917 (paired sd 0.00560175) | 1.01486 (paired sd 0.0152743) | 1.02756 (paired sd 0.00567016) | 1.04243 (paired sd 0.00530753) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|
| base | 5.04534 ± 0.202808 (5/5 same sign) | 0.219419 ± 0.139882 (5/5 same sign) | -3.80098 ± 0.471083 (5/5 same sign) | -1.12288 ± 0.0401157 (5/5 same sign) | -0.340890 ± 0.145388 (5/5 same sign) |
| utd2 | 2.68164 ± 0.152665 (5/5 same sign) | 0.375004 ± 0.0569585 (5/5 same sign) | -2.28727 ± 0.245817 (5/5 same sign) | -0.649407 ± 0.0168761 (5/5 same sign) | -0.119970 ± 0.0754692 (5/5 same sign) |
| utd4 | 1.40876 ± 0.0551179 (5/5 same sign) | 0.360585 ± 0.0770345 (5/5 same sign) | -1.17458 ± 0.240440 (5/5 same sign) | -0.427550 ± 0.0314126 (5/5 same sign) | -0.167210 ± 0.117360 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| base | mean Δ 5.04534; 2/32 patterns; p = 0.0625000 | mean Δ 2408.37; 2/32 patterns; p = 0.0625000 |
| utd2 | mean Δ 2.68164; 2/32 patterns; p = 0.0625000 | mean Δ 2181.60; 2/32 patterns; p = 0.0625000 |
| utd4 | mean Δ 1.40876; 2/32 patterns; p = 0.0625000 | mean Δ 4554.81; 2/32 patterns; p = 0.0625000 |


## 4.2 Sweep: width — hidden_sizes ∈ {(256,256),(512,512),(1024,1024)}

Configurations: `w256` (hidden (256,256)), `w512` (hidden (512,512)), `base` (hidden (1024,1024), batch 100 (paper default), UTD 1, policy delay 2). Baseline for the response tables: `base`.

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|
| w256 | 1240.18 ± 21.9290 ‖ 2837.25 ± 72.5056 | 1042.98 ± 18.9749 ‖ 1160.03 ± 25.3635 | 15153.1 ± 457.182 ‖ 15463.0 ± 584.149 | 4198.84 ± 112.792 ‖ 4261.64 ± 141.499 | 1401.87 ± 21.4356 ‖ 1378.52 ± 26.2389 | 21796.8 ± 602.346 ‖ 22263.2 ± 751.164 | 23037.0 ± 618.455 ‖ 25100.4 ± 811.324 |
| w512 | 1301.93 ± 21.8700 ‖ 2897.84 ± 62.2406 | 1119.71 ± 21.0988 ‖ 1244.82 ± 24.3729 | 16794.9 ± 362.842 ‖ 17133.3 ± 268.541 | 4654.20 ± 66.0151 ‖ 4704.94 ± 41.2614 | 1498.93 ± 27.0266 ‖ 1492.27 ± 32.8061 | 24067.7 ± 403.636 ‖ 24575.4 ± 275.200 | 25369.7 ± 399.056 ‖ 27473.2 ± 264.556 |
| base | 1410.91 ± 18.4254 ‖ 3071.46 ± 52.5339 | 1295.43 ± 26.2671 ‖ 1473.47 ± 26.1945 | 18530.1 ± 279.872 ‖ 18957.5 ± 337.485 | 5182.75 ± 53.5812 ‖ 5284.10 ± 73.8102 | 1692.28 ± 24.7333 ‖ 1733.25 ± 20.8448 | 26700.5 ± 301.074 ‖ 27448.4 ± 408.760 | 28111.5 ± 310.708 ‖ 30519.8 ± 431.915 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| w256 | 0.00639915 ± 1.71793e-04 ‖ 0.00697234 ± 2.25368e-04 | 34.8243 ± 1.57961 ‖ 155.831 ± 7.02581 | 6193.33 ± 502.759 ‖ 6137.15 ± 422.216 | 6621.65 ± 349.361 ‖ 6654.50 ± 244.255 |
| w512 | 0.00704713 ± 1.10849e-04 ‖ 0.00763145 ± 7.34878e-05 | 35.3844 ± 1.16357 ‖ 155.069 ± 7.44567 | 6394.89 ± 392.164 ‖ 6113.36 ± 346.204 | 6585.57 ± 381.452 ‖ 6629.18 ± 266.528 |
| base | 0.00780874 ± 8.63078e-05 ‖ 0.00847773 ± 1.19976e-04 | 34.8972 ± 1.32492 ‖ 153.158 ± 5.86734 | 6256.39 ± 361.457 ‖ 6392.38 ± 490.180 | 6637.29 ± 281.479 ‖ 6818.27 ± 283.203 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|
| w256 | 5.38503 ± 0.0986667 ‖ 11.3066 ± 0.190226 | 4.52838 ± 0.0551553 ‖ 4.62377 ± 0.117003 | 65.7727 ± 0.233904 ‖ 61.5966 ± 0.421824 | 18.2265 ± 0.0156936 ‖ 16.9780 ± 0.0342677 | 6.08733 ± 0.113217 ‖ 5.49502 ± 0.142783 | 94.6150 ± 0.0986667 ‖ 88.6934 ± 0.190226 |
| w512 | 5.13305 ± 0.128855 ‖ 10.5487 ± 0.251225 | 4.41473 ± 0.122471 ‖ 4.53150 ± 0.105095 | 66.1962 ± 0.439689 ‖ 62.3616 ± 0.493727 | 18.3459 ± 0.0420810 ‖ 17.1257 ± 0.0219620 | 5.91018 ± 0.172847 ‖ 5.43250 ± 0.147895 | 94.8670 ± 0.128855 ‖ 89.4513 ± 0.251225 |
| base | 5.01916 ± 0.0588648 ‖ 10.0645 ± 0.161751 | 4.60894 ± 0.120563 ‖ 4.82836 ± 0.0887409 | 65.9143 ± 0.314294 ‖ 62.1133 ± 0.348088 | 18.4366 ± 0.0384725 ‖ 17.3137 ± 0.00998654 | 6.02101 ± 0.140385 ‖ 5.68012 ± 0.112316 | 94.9808 ± 0.0588648 ‖ 89.9355 ± 0.161751 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| w256 | 4.78614 ± 0.0609863 ‖ 5.21344 ± 0.141867 | 69.5161 ± 0.197164 ‖ 69.4484 ± 0.332721 | 19.2639 ± 0.0238581 ‖ 19.1424 ± 0.0368749 | 6.43385 ± 0.122947 ‖ 6.19580 ± 0.172842 |
| w512 | 4.65371 ± 0.133817 ‖ 5.06615 ± 0.129974 | 69.7776 ± 0.383092 ‖ 69.7150 ± 0.368845 | 19.3386 ± 0.0686606 ‖ 19.1454 ± 0.0607421 | 6.23012 ± 0.188490 ‖ 6.07349 ± 0.179618 |
| base | 4.85251 ± 0.127788 ‖ 5.36883 ± 0.107654 | 69.3974 ± 0.311788 ‖ 69.0640 ± 0.268809 | 19.4108 ± 0.0479216 ‖ 19.2513 ± 0.0356577 | 6.33921 ± 0.149154 ‖ 6.31594 ± 0.134610 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|
| w256 | 8.68183e-08 ± 1.53513e-09 ‖ 1.50176e-07 ± 3.83774e-09 | 1.19676e-09 ± 3.61073e-11 ‖ 9.73287e-10 ± 3.67682e-11 | 1.18853e-09 ± 3.19272e-11 ‖ 9.55628e-10 ± 3.17297e-11 | 1.34593e-09 ± 3.71943e-11 ‖ 1.09418e-09 ± 3.69179e-11 | 1.42126e-09 ± 3.81554e-11 ‖ 1.23248e-09 ± 3.98376e-11 | 64.8030 ± 0.990886 ‖ 48.2930 ± 0.919210 |
| w512 | 2.37648e-08 ± 3.99204e-10 ‖ 4.52787e-08 ± 9.72510e-10 | 3.43339e-10 ± 7.41759e-12 ‖ 3.09447e-10 ± 4.85014e-12 | 3.41738e-10 ± 4.84720e-12 ‖ 3.04081e-10 ± 2.66674e-12 | 3.84864e-10 ± 6.45449e-12 ‖ 3.46912e-10 ± 3.88479e-12 | 4.05328e-10 ± 6.37567e-12 ‖ 3.87469e-10 ± 3.73117e-12 | 18.1498 ± 0.327253 ‖ 15.4783 ± 0.340274 |
| base | 6.57995e-09 ± 8.59292e-11 ‖ 1.31903e-08 ± 2.25605e-10 | 9.64080e-11 ± 1.45611e-12 ‖ 9.24275e-11 ± 1.64541e-12 | 9.69593e-11 ± 1.00240e-12 ‖ 9.24445e-11 ± 1.29130e-12 | 1.08690e-10 ± 1.22558e-12 ‖ 1.04658e-10 ± 1.55857e-12 | 1.14334e-10 ± 1.26370e-12 ‖ 1.16266e-10 ± 1.64539e-12 | 5.24807 ± 0.0767024 ‖ 4.95065 ± 0.0595384 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| w256 | 1.72978 | 0.813269 | 0.804042 | 0.812954 | 0.867175 | 0.745228 |
| w512 | 1.90528 | 0.901287 | 0.889807 | 0.901389 | 0.955939 | 0.852808 |
| base | 2.00462 | 0.958711 | 0.953435 | 0.962904 | 1.01690 | 0.943328 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|
| w256 | 10.4901 ± 0.0842059 ‖ 24.3393 ± 0.338805 | 190.570 ± 2.70440 ‖ 196.143 ± 6.49934 | 8.70301 ± 0.0306956 ‖ 9.81306 ± 0.118681 | 126.430 ± 2.26203 ‖ 130.831 ± 5.16849 | 35.0340 ± 0.501904 ‖ 36.0559 ± 1.21858 | 11.6983 ± 0.0739788 ‖ 11.6618 ± 0.126038 | 201.060 ± 2.74931 ‖ 220.482 ± 6.78034 |
| w512 | 10.6715 ± 0.0481447 ‖ 24.4217 ± 0.141512 | 198.191 ± 5.52398 ‖ 205.127 ± 5.83478 | 8.68743 ± 0.0255816 ‖ 9.74127 ± 0.0416707 | 130.367 ± 4.66279 ‖ 134.149 ± 4.77945 | 36.1231 ± 0.978021 ‖ 36.8330 ± 0.996539 | 11.6299 ± 0.0455976 ‖ 11.6772 ± 0.0455079 | 208.863 ± 5.54385 ‖ 229.549 ± 5.89236 |
| base | 11.1119 ± 0.0596187 ‖ 24.7473 ± 0.178141 | 195.987 ± 4.59564 ‖ 199.380 ± 3.54820 | 8.87991 ± 0.0441167 ‖ 9.85333 ± 0.0615825 | 127.075 ± 3.89160 ‖ 126.797 ± 2.93413 | 35.5388 ± 0.853585 ‖ 35.3410 ± 0.620329 | 11.6014 ± 0.0938117 ‖ 11.5912 ± 0.0527052 | 207.099 ± 4.64914 ‖ 224.127 ± 3.53526 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|
| w256 | 118.223 ± 1.83532 ‖ 116.566 ± 2.17607 | 114.367 ± 2.14342 ‖ 113.517 ± 2.00814 | 114.569 ± 2.11001 ‖ 113.853 ± 2.00348 | 94.0912 ± 2.86638 ‖ 94.6795 ± 2.58326 | 1.03381 ± 0.0109763 ‖ 1.02687 ± 0.00903788 |
| w512 | 122.002 ± 2.08024 ‖ 118.660 ± 2.54205 | 121.476 ± 2.13748 ‖ 119.861 ± 2.62864 | 121.503 ± 2.09557 ‖ 119.733 ± 2.60263 | 98.6448 ± 1.87821 ‖ 98.2735 ± 2.10862 | 1.00443 ± 0.0151339 ‖ 0.990009 ± 0.00730274 |
| base | 126.974 ± 1.58927 ‖ 124.115 ± 2.06670 | 136.273 ± 2.18738 ‖ 137.683 ± 1.75444 | 135.772 ± 2.09392 ‖ 136.183 ± 1.73524 | 105.944 ± 1.05964 ‖ 108.231 ± 1.46703 | 0.931908 ± 0.0158865 ‖ 0.901446 ± 0.00866638 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|
| w256 | 1.36181 ± 0.0109206 ‖ 0.776348 ± 0.0107907 | 84.9930 ± 1.18806 ‖ 103.824 ± 3.35211 | 80.6287 ± 1.08628 ‖ 92.4378 ± 2.77822 | 100.174 ± 1.75761 ‖ 121.581 ± 4.64174 | 100.855 ± 1.42513 ‖ 123.793 ± 4.06760 | 95.4312 ± 0.110905 ‖ 96.0290 ± 0.170705 | 95.2868–95.5680 ‖ 95.8454–96.3102 |
| w512 | 5.13374 ± 0.0231448 ‖ 2.62069 ± 0.0152860 | 315.726 ± 8.70854 ‖ 345.569 ± 9.66709 | 299.840 ± 7.87926 ‖ 309.046 ± 7.81740 | 375.602 ± 13.2584 ‖ 413.142 ± 14.4113 | 377.241 ± 10.1136 ‖ 420.317 ± 11.1974 | 94.2511 ± 0.236622 ‖ 93.7923 ± 0.233996 | 94.0324–94.5581 ‖ 93.4861–94.0964 |
| base | 19.2974 ± 0.103223 ‖ 9.40980 ± 0.0676047 | 1253.98 ± 28.8906 ‖ 1315.75 ± 23.5499 | 1187.69 ± 26.2133 ‖ 1171.44 ± 18.5841 | 1513.64 ± 45.3034 ‖ 1618.30 ± 37.7247 | 1504.75 ± 35.4138 ‖ 1617.78 ± 28.5573 | 93.4174 ± 0.243312 ‖ 92.0747 ± 0.142527 | 93.1600–93.7995 ‖ 91.8827–92.2302 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| w256 | 27.6343 ± 0.433449 ‖ 29.1298 ± 0.940198 | 55.4771 ± 0.674324 ‖ 53.7233 ± 1.24469 | 16.8886 ± 0.265456 ‖ 17.1468 ± 0.320294 | 500/500 ‖ 500/500 |
| w512 | 26.8763 ± 0.505987 ‖ 28.7583 ± 0.579471 | 56.7548 ± 0.760000 ‖ 54.3937 ± 0.913043 | 16.3689 ± 0.278937 ‖ 16.8480 ± 0.361148 | 500/500 ‖ 500/500 |
| base | 25.5248 ± 0.309549 ‖ 27.5657 ± 0.436376 | 58.7534 ± 0.500385 ‖ 56.3345 ± 0.696456 | 15.7217 ± 0.195131 ‖ 16.0998 ± 0.268080 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| w256 | 23.9537 ± 0.496729 ‖ 23.7914 ± 0.578969 | 58.5567 ± 0.816579 ‖ 58.5888 ± 0.824671 | 17.4896 ± 0.329452 ‖ 17.6197 ± 0.309161 | 0/500 ‖ 0/500 |
| w512 | 22.5023 ± 0.312602 ‖ 22.7686 ± 0.374441 | 61.0320 ± 0.550866 ‖ 60.5416 ± 0.727251 | 16.4658 ± 0.294390 ‖ 16.6898 ± 0.368382 | 0/500 ‖ 0/500 |
| base | 20.0463 ± 0.175344 ‖ 20.1816 ± 0.123170 | 65.2771 ± 0.408611 ‖ 65.2934 ± 0.271919 | 14.6766 ± 0.237438 ‖ 14.5251 ± 0.185325 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|
| w256 | 60.2018 ± 2.48597 ‖ 60.9376 ± 1.82337 | 62.2208 ± 2.13216 ‖ 62.5811 ± 2.17717 | 62.1117 ± 2.15069 ‖ 62.3955 ± 2.13294 | 71.1918 ± 3.64344 ‖ 71.0621 ± 3.38464 |
| w512 | 59.1015 ± 1.50691 ‖ 59.6533 ± 1.17649 | 59.3580 ± 1.53800 ‖ 59.0516 ± 0.843439 | 59.3446 ± 1.52379 ‖ 59.1149 ± 0.875039 | 72.1109 ± 2.43519 ‖ 70.7892 ± 2.24509 |
| base | 56.4211 ± 1.03774 ‖ 59.1376 ± 1.43952 | 52.5691 ± 0.622384 ‖ 53.3031 ± 1.04567 | 52.7623 ± 0.614592 ‖ 53.8902 ± 1.07300 | 71.6287 ± 0.699504 ‖ 73.3896 ± 1.72627 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|
| w256 | -12.0886 ± 1.91943 ‖ -7.60909 ± 2.60310 | -19.4561 ± 2.43766 ‖ -21.2675 ± 1.42820 | -18.2267 ± 2.02278 ‖ -18.4097 ± 3.47939 | -18.9911 ± 1.63061 ‖ -19.3309 ± 3.10003 | -17.1405 ± 2.10658 ‖ -20.4683 ± 0.980487 | -18.3715 ± 1.74972 ‖ -18.8720 ± 3.11849 | -18.0569 ± 1.69789 ‖ -17.7390 ± 3.05685 |
| w512 | -7.72249 ± 1.13783 ‖ -5.60787 ± 3.47954 | -13.5158 ± 3.17979 ‖ -15.4750 ± 3.08169 | -9.34544 ± 2.47843 ‖ -9.59932 ± 2.15406 | -10.1882 ± 1.73501 ‖ -10.9459 ± 1.50987 | -11.4032 ± 2.42438 ‖ -13.8788 ± 2.79196 | -9.84977 ± 1.91211 ‖ -10.4508 ± 1.67547 | -9.74257 ± 1.84986 ‖ -9.96711 ± 1.59735 |

Duration:

| config | D rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|
| w256 | -5.59081 ± 1.15898 ‖ -1.65255 ± 0.686268 | -2.71656 ± 2.87304 ‖ -1.61627 ± 3.04721 | -2.87220 ± 2.76291 ‖ -1.62242 ± 2.76896 | -1.99088 ± 0.424206 ‖ -0.410921 ± 0.713980 | -0.424531 ± 3.81881 ‖ 3.19757 ± 3.85543 | -1.36950 ± 2.98395 ‖ 2.03281 ± 3.32582 | 0.840301 ± 0.939296 ‖ 0.611810 ± 1.30444 |
| w512 | -3.95912 ± 0.875028 ‖ -1.30976 ± 1.15005 | 1.18262 ± 4.11548 ‖ 2.92248 ± 3.95187 | 0.905150 ± 3.93535 ‖ 2.45016 ± 3.51707 | -2.16549 ± 0.588479 ‖ -1.13524 ± 0.555773 | 2.69104 ± 5.40333 ‖ 5.86753 ± 5.16639 | 1.70766 ± 4.19610 ‖ 4.25907 ± 3.79425 | 0.250319 ± 0.824007 ‖ 0.744369 ± 0.828558 |

Mean power:

| config | P rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|
| w256 | -6.88119 ± 1.77902 ‖ -6.05730 ± 2.52999 | -16.0479 ± 2.51903 ‖ -17.5452 ± 1.53506 | -15.5920 ± 2.46085 ‖ -16.3903 ± 1.58303 | -11.1787 ± 2.91009 ‖ -12.5117 ± 2.50105 |
| w512 | -3.91293 ± 1.39892 ‖ -4.35492 ± 3.34721 | -10.8280 ± 2.66164 ‖ -12.9143 ± 2.97131 | -10.4821 ± 2.58977 ‖ -12.0494 ± 2.99547 | -6.87102 ± 2.55014 ‖ -9.16825 ± 3.07259 |

Achieved throughput:

| config | rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|
| w256 | -92.9427 ± 0.0867058 ‖ -91.7499 ± 0.0576563 | -93.2188 ± 0.202315 ‖ -92.1085 ± 0.241478 | -93.2082 ± 0.195175 ‖ -92.1087 ± 0.219617 |
| w512 | -73.3958 ± 0.243210 ‖ -72.1476 ± 0.327031 | -74.8074 ± 1.03650 ‖ -73.7258 ± 0.989463 | -74.7408 ± 0.996130 ‖ -73.6103 ± 0.892022 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| w256 | 1219.62 ± 28.8121 ‖ 1038.74 ± 32.0838 | 1141.31 ± 30.7056 ‖ 953.338 ± 44.9192 | 1125.70 ± 24.6719 ‖ 933.973 ± 39.7345 | 1138.23 ± 26.5418 ‖ 945.722 ± 40.1967 | 1143.00 ± 25.7553 ‖ 960.284 ± 39.4006 | 1135.10 ± 31.4007 ‖ 875.463 ± 12.0258 |
| w512 | 261.176 ± 4.45349 ‖ 243.436 ± 12.6600 | 256.204 ± 9.73834 ‖ 234.885 ± 7.97963 | 252.494 ± 6.80957 ‖ 228.987 ± 5.57781 | 254.135 ± 7.51129 ‖ 231.531 ± 6.20295 | 254.555 ± 7.26676 ‖ 233.317 ± 5.91366 | 245.926 ± 9.46597 ‖ 212.742 ± 10.1387 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|
| w256 | 0.0666189 ‖ 0.0811346 | 0.0658764 ‖ 0.0774588 | 0.0660920 ‖ 0.0780186 | 0.0659233 ‖ 0.0775808 | 0.0659239 ‖ 0.0775840 | 0.0670871 ‖ 0.0815323 |
| w512 | 0.255492 ‖ 0.274846 | 0.254502 ‖ 0.269945 | 0.254789 ‖ 0.270692 | 0.254564 ‖ 0.270108 | 0.254565 ‖ 0.270112 | 0.256115 ‖ 0.275375 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|
| w256 | 0.365868 ± 0.124981 ‖ 1.24212 ± 0.144168 | -0.0805601 ± 0.143495 ‖ -0.204583 ± 0.117632 | -0.141580 ± 0.407906 ‖ -0.516770 ± 0.356302 | -0.210048 ± 0.0426050 ‖ -0.335669 ± 0.0338804 | 0.0663204 ± 0.168830 ‖ -0.185097 ± 0.138828 | -0.365868 ± 0.124981 ‖ -1.24212 ± 0.144168 |
| w512 | 0.113889 ± 0.0850628 ‖ 0.484239 ± 0.358900 | -0.194212 ± 0.209782 ‖ -0.296860 ± 0.165499 | 0.281837 ± 0.572642 ‖ 0.248273 ± 0.721950 | -0.0906816 ± 0.0582259 ‖ -0.188033 ± 0.0291757 | -0.110833 ± 0.237887 ‖ -0.247619 ± 0.206240 | -0.113889 ± 0.0850628 ‖ -0.484239 ± 0.358900 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| w256 | 1.08957 | 2.28777 | 5.92159 | rollout | 0.993753 | 0.993364 | 0.867175 |
| w512 | 1.08292 | 2.22580 | 5.41569 | rollout | 0.985437 | 0.985714 | 0.955939 |
| base | 1.08567 | 2.17694 | 5.04534 | rollout | 1.00303 | 0.967471 | 1.01690 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|
| w256 | 2.28777 (paired sd 0.0868627) | 1.11223 (paired sd 0.0208642) | 1.02045 (paired sd 0.0535485) | 1.01496 (paired sd 0.0464502) | 0.983347 (paired sd 0.0162932) | 1.02140 (paired sd 0.0473837) | 1.08957 (paired sd 0.0489076) |
| w512 | 2.22580 (paired sd 0.0589866) | 1.11174 (paired sd 0.0385053) | 1.02015 (paired sd 0.0248255) | 1.01090 (paired sd 0.0119457) | 0.995561 (paired sd 0.0356184) | 1.02109 (paired sd 0.0161899) | 1.08292 (paired sd 0.0138515) |
| base | 2.17694 (paired sd 0.0466529) | 1.13744 (paired sd 0.0389360) | 1.02307 (paired sd 0.0206497) | 1.01955 (paired sd 0.0172043) | 1.02421 (paired sd 0.0204510) | 1.02801 (paired sd 0.0178265) | 1.08567 (paired sd 0.0174699) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|
| w256 | 5.92159 ± 0.199114 (5/5 same sign) | 0.0953963 ± 0.149211 (4/5 same sign) | -4.17617 ± 0.533949 (5/5 same sign) | -1.24850 ± 0.0391826 (5/5 same sign) | -0.592308 ± 0.219100 (5/5 same sign) |
| w512 | 5.41569 ± 0.293762 (5/5 same sign) | 0.116771 ± 0.199578 (3/5 same sign) | -3.83455 ± 0.777021 (5/5 same sign) | -1.22024 ± 0.0322080 (5/5 same sign) | -0.477677 ± 0.265852 (5/5 same sign) |
| base | 5.04534 ± 0.202808 (5/5 same sign) | 0.219419 ± 0.139882 (5/5 same sign) | -3.80098 ± 0.471083 (5/5 same sign) | -1.12288 ± 0.0401157 (5/5 same sign) | -0.340890 ± 0.145388 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| w256 | mean Δ 5.92159; 2/32 patterns; p = 0.0625000 | mean Δ 2063.46; 2/32 patterns; p = 0.0625000 |
| w512 | mean Δ 5.41569; 2/32 patterns; p = 0.0625000 | mean Δ 2103.54; 2/32 patterns; p = 0.0625000 |
| base | mean Δ 5.04534; 2/32 patterns; p = 0.0625000 | mean Δ 2408.37; 2/32 patterns; p = 0.0625000 |


## 4.3 Sweep: batch size — batch_size ∈ {100,256,512,1024}

Configurations: `base` (hidden (1024,1024), batch 100 (paper default), UTD 1, policy delay 2), `b256` (batch 256 (SAC/MBPO default; the batch-matched comparison point)), `b512` (batch 512), `b1024` (batch 1024). Baseline for the response tables: `base`.

**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|
| base | 1410.91 ± 18.4254 ‖ 3071.46 ± 52.5339 | 1295.43 ± 26.2671 ‖ 1473.47 ± 26.1945 | 18530.1 ± 279.872 ‖ 18957.5 ± 337.485 | 5182.75 ± 53.5812 ‖ 5284.10 ± 73.8102 | 1692.28 ± 24.7333 ‖ 1733.25 ± 20.8448 | 26700.5 ± 301.074 ‖ 27448.4 ± 408.760 | 28111.5 ± 310.708 ‖ 30519.8 ± 431.915 |
| b256 | 1516.82 ± 44.8592 ‖ 3086.58 ± 58.2258 | 1689.43 ± 12.8701 ‖ 2132.18 ± 13.2997 | 22121.8 ± 403.776 ‖ 22136.2 ± 344.166 | 6160.56 ± 66.3990 ‖ 6165.50 ± 60.8783 | 1965.08 ± 16.8894 ‖ 2000.11 ± 16.1819 | 31936.9 ± 449.997 ‖ 32434.0 ± 378.529 | 33453.7 ± 435.384 ‖ 35520.5 ± 357.746 |
| b512 | 1681.61 ± 39.5951 ‖ 3301.77 ± 33.3384 | 2431.45 ± 70.6727 ‖ 3463.12 ± 52.1586 | 27596.4 ± 829.827 ‖ 27504.1 ± 598.828 | 7582.97 ± 122.336 ‖ 7620.33 ± 97.6287 | 2390.74 ± 81.6670 ‖ 2471.88 ± 48.6378 | 40001.6 ± 807.421 ‖ 41059.4 ± 602.205 | 41683.2 ± 778.096 ‖ 44361.2 ± 591.790 |
| b1024 | 1752.39 ± 41.1506 ‖ 3390.84 ± 70.8717 | 3406.18 ± 102.421 ‖ 6012.97 ± 196.280 | 31608.6 ± 899.991 ‖ 32144.2 ± 804.987 | 8610.15 ± 119.128 ‖ 8843.11 ± 83.0931 | 2691.06 ± 100.270 ‖ 2843.88 ± 87.0553 | 46316.0 ± 819.427 ‖ 49844.2 ± 728.002 | 48068.4 ± 808.125 ‖ 53235.0 ± 707.755 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| base | 0.00780874 ± 8.63078e-05 ‖ 0.00847773 ± 1.19976e-04 | 34.8972 ± 1.32492 ‖ 153.158 ± 5.86734 | 6256.39 ± 361.457 ‖ 6392.38 ± 490.180 | 6637.29 ± 281.479 ‖ 6818.27 ± 283.203 |
| b256 | 0.00929270 ± 1.20940e-04 ‖ 0.00986682 ± 9.93739e-05 | 33.9849 ± 0.470668 ‖ 150.251 ± 3.26910 | 5711.61 ± 105.644 ‖ 5762.83 ± 101.484 | 6162.71 ± 46.5073 ‖ 6170.17 ± 28.8266 |
| b512 | 0.0115787 ± 2.16138e-04 ‖ 0.0123226 ± 1.64386e-04 | 33.9755 ± 0.491756 ‖ 149.196 ± 3.85804 | 5730.62 ± 54.6457 ‖ 5767.81 ± 18.4305 | 6328.40 ± 112.401 ‖ 6276.52 ± 53.0680 |
| b1024 | 0.0133523 ± 2.24479e-04 ‖ 0.0147875 ± 1.96599e-04 | 33.6659 ± 0.139563 ‖ 149.078 ± 4.04560 | 5632.70 ± 72.8501 ‖ 5674.55 ± 50.0936 | 6405.08 ± 44.4827 ‖ 6429.73 ± 66.5827 |

**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|
| base | 5.01916 ± 0.0588648 ‖ 10.0645 ± 0.161751 | 4.60894 ± 0.120563 ‖ 4.82836 ± 0.0887409 | 65.9143 ± 0.314294 ‖ 62.1133 ± 0.348088 | 18.4366 ± 0.0384725 ‖ 17.3137 ± 0.00998654 | 6.02101 ± 0.140385 ‖ 5.68012 ± 0.112316 | 94.9808 ± 0.0588648 ‖ 89.9355 ± 0.161751 |
| b256 | 4.53511 ± 0.161573 ‖ 8.69063 ± 0.205170 | 5.05097 ± 0.0940456 ‖ 6.00340 ± 0.0944562 | 66.1231 ± 0.376812 ‖ 62.3167 ± 0.362815 | 18.4156 ± 0.0549150 ‖ 17.3576 ± 0.0187040 | 5.87517 ± 0.116413 ‖ 5.63164 ± 0.0987609 | 95.4649 ± 0.161573 ‖ 91.3094 ± 0.205170 |
| b512 | 4.03640 ± 0.157926 ‖ 7.44421 ± 0.140978 | 5.83711 ± 0.271377 ‖ 7.80888 ± 0.216275 | 66.1941 ± 0.760518 ‖ 61.9948 ± 0.534233 | 18.1926 ± 0.0505843 ‖ 17.1780 ± 0.0211525 | 5.73984 ± 0.297152 ‖ 5.57404 ± 0.181078 | 95.9636 ± 0.157926 ‖ 92.5558 ± 0.140978 |
| b1024 | 3.64674 ± 0.117980 ‖ 6.37080 ± 0.172885 | 7.09048 ± 0.330940 ‖ 11.2980 ± 0.437363 | 65.7474 ± 0.782584 ‖ 60.3743 ± 0.716820 | 17.9130 ± 0.0567816 ‖ 16.6123 ± 0.0912859 | 5.60242 ± 0.302053 ‖ 5.34462 ± 0.233675 | 96.3533 ± 0.117980 ‖ 93.6292 ± 0.172885 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| base | 4.85251 ± 0.127788 ‖ 5.36883 ± 0.107654 | 69.3974 ± 0.311788 ‖ 69.0640 ± 0.268809 | 19.4108 ± 0.0479216 ‖ 19.2513 ± 0.0356577 | 6.33921 ± 0.149154 ‖ 6.31594 ± 0.134610 |
| b256 | 5.29104 ± 0.105886 ‖ 6.57492 ± 0.112010 | 69.2640 ± 0.296835 ‖ 68.2475 ± 0.266010 | 19.2905 ± 0.0671484 ‖ 19.0097 ± 0.0444070 | 6.15442 ± 0.130688 ‖ 6.16782 ± 0.118863 |
| b512 | 6.08297 ± 0.291450 ‖ 8.43721 ± 0.245099 | 68.9775 ± 0.687441 ‖ 66.9805 ± 0.485411 | 18.9579 ± 0.0822039 ‖ 18.5597 ± 0.0386827 | 5.98164 ± 0.318263 ‖ 6.02257 ± 0.203554 |
| b1024 | 7.35913 ± 0.351180 ‖ 12.0667 ± 0.464874 | 68.2351 ± 0.742732 ‖ 64.4819 ± 0.713606 | 18.5910 ± 0.0773751 ‖ 17.7428 ± 0.126518 | 5.81470 ± 0.319134 ‖ 5.70855 ± 0.256923 |

**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|
| base | 6.57995e-09 ± 8.59292e-11 ‖ 1.31903e-08 ± 2.25605e-10 | 9.64080e-11 ± 1.45611e-12 ‖ 9.24275e-11 ± 1.64541e-12 | 9.69593e-11 ± 1.00240e-12 ‖ 9.24445e-11 ± 1.29130e-12 | 1.08690e-10 ± 1.22558e-12 ‖ 1.04658e-10 ± 1.55857e-12 | 1.14334e-10 ± 1.26370e-12 ‖ 1.16266e-10 ± 1.64539e-12 | 5.24807 ± 0.0767024 ‖ 4.95065 ± 0.0595384 |
| b256 | 7.07388e-09 ± 2.09207e-10 ‖ 1.32552e-08 ± 2.50049e-10 | 4.49590e-11 ± 8.20608e-13 ‖ 4.21582e-11 ± 6.55462e-13 | 4.50204e-11 ± 4.85234e-13 ‖ 4.21345e-11 ± 4.16038e-13 | 5.07835e-11 ± 7.15548e-13 ‖ 4.83077e-11 ± 5.63787e-13 | 5.31773e-11 ± 6.92076e-13 ‖ 5.28866e-11 ± 5.32649e-13 | 6.09406 ± 0.0523770 ‖ 5.71287 ± 0.0462199 |
| b512 | 7.84238e-09 ± 1.84657e-10 ‖ 1.41793e-08 ± 1.43171e-10 | 2.80426e-11 ± 8.43244e-13 ‖ 2.61907e-11 ± 5.70231e-13 | 2.77076e-11 ± 4.47006e-13 ‖ 2.60384e-11 ± 3.33594e-13 | 3.18037e-11 ± 6.41948e-13 ‖ 3.05773e-11 ± 4.48467e-13 | 3.31350e-11 ± 6.18527e-13 ‖ 3.30304e-11 ± 4.40635e-13 | 7.41411 ± 0.253264 ‖ 7.06035 ± 0.138923 |
| b1024 | 8.17249e-09 ± 1.91911e-10 ‖ 1.45618e-08 ± 3.04356e-10 | 1.60599e-11 ± 4.57271e-13 ‖ 1.53046e-11 ± 3.83273e-13 | 1.57304e-11 ± 2.17642e-13 ‖ 1.51083e-11 ± 1.41963e-13 | 1.84120e-11 ± 3.25747e-13 ‖ 1.85597e-11 ± 2.71075e-13 | 1.91070e-11 ± 3.21227e-13 ‖ 1.98206e-11 ± 2.63513e-13 | 8.34546 ± 0.310955 ‖ 8.12291 ± 0.248654 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| base | 2.00462 | 0.958711 | 0.953435 | 0.962904 | 1.01690 | 0.943328 |
| b256 | 1.87382 | 0.937702 | 0.935898 | 0.951248 | 0.994533 | 0.937449 |
| b512 | 1.80804 | 0.933958 | 0.939756 | 0.961440 | 0.996845 | 0.952286 |
| b1024 | 1.78181 | 0.952971 | 0.960451 | 1.00802 | 1.03735 | 0.973333 |

**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|
| base | 11.1119 ± 0.0596187 ‖ 24.7473 ± 0.178141 | 195.987 ± 4.59564 ‖ 199.380 ± 3.54820 | 8.87991 ± 0.0441167 ‖ 9.85333 ± 0.0615825 | 127.075 ± 3.89160 ‖ 126.797 ± 2.93413 | 35.5388 ± 0.853585 ‖ 35.3410 ± 0.620329 | 11.6014 ± 0.0938117 ‖ 11.5912 ± 0.0527052 | 207.099 ± 4.64914 ‖ 224.127 ± 3.53526 |
| b256 | 10.9702 ± 0.103092 ‖ 24.5015 ± 0.197026 | 200.053 ± 4.17276 ‖ 202.982 ± 3.77624 | 9.91330 ± 0.0491517 ‖ 12.2998 ± 0.0727960 | 129.830 ± 3.50683 ‖ 127.717 ± 3.25830 | 36.1536 ± 0.710964 ‖ 35.5712 ± 0.718193 | 11.5307 ± 0.0516577 ‖ 11.5380 ± 0.0849614 | 211.023 ± 4.18610 ‖ 227.483 ± 3.81449 |
| b512 | 10.9294 ± 0.0773905 ‖ 24.4186 ± 0.316702 | 206.571 ± 9.53589 ‖ 208.754 ± 6.28943 | 11.7219 ± 0.0773220 ‖ 16.1285 ± 0.147982 | 133.242 ± 8.35711 ‖ 128.174 ± 5.63467 | 36.5988 ± 1.77199 ‖ 35.5064 ± 1.23758 | 11.5240 ± 0.0524112 ‖ 11.5111 ± 0.0700803 | 217.500 ± 9.53218 ‖ 233.173 ± 6.45638 |
| b1024 | 10.8323 ± 0.206042 ‖ 24.3646 ± 0.244790 | 217.363 ± 8.83903 ‖ 227.741 ± 6.93452 | 14.6866 ± 0.165459 ‖ 24.2977 ± 0.947397 | 136.536 ± 9.16537 ‖ 130.045 ± 8.25303 | 37.1750 ± 1.95503 ‖ 35.7589 ± 1.69667 | 11.6007 ± 0.105621 ‖ 11.4860 ± 0.0957693 | 228.195 ± 9.00097 ‖ 252.106 ± 7.13483 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|
| base | 126.974 ± 1.58927 ‖ 124.115 ± 2.06670 | 136.273 ± 2.18738 ‖ 137.683 ± 1.75444 | 135.772 ± 2.09392 ‖ 136.183 ± 1.73524 | 105.944 ± 1.05964 ‖ 108.231 ± 1.46703 | 0.931908 ± 0.0158865 ‖ 0.901446 ± 0.00866638 |
| b256 | 138.263 ± 3.68402 ‖ 125.969 ± 1.67073 | 159.662 ± 1.21620 ‖ 159.805 ± 1.18342 | 158.550 ± 1.28336 ‖ 156.160 ± 1.06155 | 115.918 ± 0.633278 ‖ 116.441 ± 0.430259 | 0.865912 ± 0.0187521 ‖ 0.788289 ± 0.0106253 |
| b512 | 153.865 ± 3.64850 ‖ 135.228 ± 1.78808 | 193.830 ± 5.01893 ‖ 196.762 ± 3.09571 | 191.815 ± 4.80554 ‖ 190.312 ± 2.79605 | 135.209 ± 1.60444 ‖ 136.447 ± 0.823072 | 0.793975 ± 0.0157173 ‖ 0.687308 ± 0.00602826 |
| b1024 | 161.806 ± 4.23361 ‖ 139.174 ± 2.78113 | 213.243 ± 4.98253 ‖ 218.947 ± 3.47920 | 210.799 ± 4.86678 ‖ 211.232 ± 3.15212 | 147.241 ± 1.23732 ‖ 151.095 ± 0.844336 | 0.758781 ± 0.00842951 ‖ 0.635686 ± 0.0108995 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|
| base | 19.2974 ± 0.103223 ‖ 9.40980 ± 0.0676047 | 1253.98 ± 28.8906 ‖ 1315.75 ± 23.5499 | 1187.69 ± 26.2133 ‖ 1171.44 ± 18.5841 | 1513.64 ± 45.3034 ‖ 1618.30 ± 37.7247 | 1504.75 ± 35.4138 ‖ 1617.78 ± 28.5573 | 93.4174 ± 0.243312 ‖ 92.0747 ± 0.142527 | 93.1600–93.7995 ‖ 91.8827–92.2302 |
| b256 | 19.5476 ± 0.184363 ‖ 9.50429 ± 0.0756488 | 3144.71 ± 66.9947 ‖ 3308.62 ± 61.3348 | 2982.14 ± 60.3792 ‖ 2953.13 ± 49.4058 | 3792.18 ± 105.228 ‖ 4113.35 ± 104.217 | 3786.13 ± 75.9774 ‖ 4115.02 ± 82.7333 | 93.6860 ± 0.194991 ‖ 92.1846 ± 0.308519 | 93.3883–93.8756 ‖ 91.8998–92.6464 |
| b512 | 19.6200 ± 0.138379 ‖ 9.53733 ± 0.122822 | 6099.02 ± 276.809 ‖ 6437.17 ± 194.669 | 5792.58 ± 249.846 ‖ 5763.39 ± 160.318 | 7408.50 ± 454.032 ‖ 8205.84 ± 361.844 | 7491.60 ± 356.565 ‖ 8250.43 ± 288.414 | 93.4505 ± 0.600126 ‖ 91.6340 ± 0.627035 | 92.8372–94.1972 ‖ 90.9899–92.3978 |
| b1024 | 19.8006 ± 0.369919 ‖ 9.55796 ± 0.0948715 | 11588.5 ± 477.003 ‖ 11801.1 ± 356.145 | 11038.4 ± 440.339 ‖ 10660.4 ± 299.084 | 14468.2 ± 989.494 ‖ 16201.6 ± 1006.13 | 14756.9 ± 786.643 ‖ 16397.3 ± 763.139 | 91.9633 ± 1.47748 ‖ 88.4683 ± 1.95388 | 89.9807–93.4098 ‖ 86.5986–90.9608 |

**Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 25.5248 ± 0.309549 ‖ 27.5657 ± 0.436376 | 58.7534 ± 0.500385 ‖ 56.3345 ± 0.696456 | 15.7217 ± 0.195131 ‖ 16.0998 ± 0.268080 | 500/500 ‖ 500/500 |
| b256 | 23.6946 ± 0.413507 ‖ 27.0431 ± 0.138298 | 61.8627 ± 0.777348 ‖ 57.0948 ± 0.306406 | 14.4427 ± 0.391543 ‖ 15.8621 ± 0.209216 | 500/500 ‖ 500/500 |
| b512 | 20.9966 ± 0.225729 ‖ 25.2888 ± 0.265115 | 66.0286 ± 0.515303 ‖ 59.9331 ± 0.412736 | 12.9748 ± 0.305543 ‖ 14.7780 ± 0.193786 | 500/500 ‖ 500/500 |
| b1024 | 20.0091 ± 0.281467 ‖ 24.7048 ± 0.493009 | 67.6473 ± 0.500012 ‖ 60.9348 ± 0.756453 | 12.3435 ± 0.313282 ‖ 14.3605 ± 0.292502 | 500/500 ‖ 500/500 |

**Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| base | 20.0463 ± 0.175344 ‖ 20.1816 ± 0.123170 | 65.2771 ± 0.408611 ‖ 65.2934 ± 0.271919 | 14.6766 ± 0.237438 ‖ 14.5251 ± 0.185325 | 0/500 ‖ 0/500 |
| b256 | 17.1861 ± 0.0684542 ‖ 17.3539 ± 0.138058 | 70.2890 ± 0.0906095 ‖ 70.1331 ± 0.165018 | 12.5249 ± 0.0947725 ‖ 12.5130 ± 0.0924099 | 0/500 ‖ 0/500 |
| b512 | 13.9874 ± 0.0898930 ‖ 14.2320 ± 0.0623047 | 75.6903 ± 0.328931 ‖ 75.6036 ± 0.136830 | 10.3223 ± 0.270070 ‖ 10.1643 ± 0.159587 | 0/500 ‖ 0/500 |
| b1024 | 12.7504 ± 0.0483434 ‖ 13.1090 ± 0.0529182 | 77.8679 ± 0.200985 ‖ 77.7566 ± 0.115732 | 9.38176 ± 0.217252 ‖ 9.13439 ± 0.146077 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|
| base | 56.4211 ± 1.03774 ‖ 59.1376 ± 1.43952 | 52.5691 ± 0.622384 ‖ 53.3031 ± 1.04567 | 52.7623 ± 0.614592 ‖ 53.8902 ± 1.07300 | 71.6287 ± 0.699504 ‖ 73.3896 ± 1.72627 |
| b256 | 47.7305 ± 1.03817 ‖ 52.6361 ± 1.09063 | 41.3179 ± 0.572853 ‖ 41.4846 ± 0.478737 | 41.6076 ± 0.560833 ‖ 42.4527 ± 0.496750 | 65.9661 ± 0.797199 ‖ 66.2920 ± 0.673162 |
| b512 | 43.5581 ± 1.12954 ‖ 49.4862 ± 0.632408 | 34.5755 ± 0.743802 ‖ 34.0113 ± 0.442575 | 34.9380 ± 0.733383 ‖ 35.1633 ± 0.428081 | 66.9922 ± 0.900202 ‖ 66.9106 ± 0.269469 |
| b1024 | 41.3543 ± 1.16694 ‖ 48.3333 ± 1.09166 | 31.3752 ± 0.788922 ‖ 30.7187 ± 0.536026 | 31.7386 ± 0.789690 ‖ 31.8399 ± 0.527228 | 66.8741 ± 0.186495 ‖ 67.2435 ± 0.324525 |

**Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|
| b256 | 7.51048 ± 3.03657 ‖ 0.529073 ± 3.16010 | 30.4604 ± 2.98087 ‖ 44.7364 ± 2.43245 | 19.3845 ± 1.38691 ‖ 16.7913 ± 2.46070 | 18.8703 ± 1.10452 ‖ 16.6938 ± 1.62720 | 16.1397 ± 1.94660 ‖ 15.4132 ± 1.94923 | 19.6119 ± 1.09637 ‖ 18.1802 ± 1.93372 | 19.0054 ± 1.07284 ‖ 16.4000 ± 1.73111 |
| b512 | 19.1967 ± 2.91163 ‖ 7.52820 ± 2.41120 | 87.8121 ± 8.35462 ‖ 135.097 ± 5.80389 | 48.9736 ± 5.66658 ‖ 45.1141 ± 3.80036 | 46.3225 ± 2.68668 ‖ 44.2362 ± 2.78626 | 41.3338 ± 6.33893 ‖ 42.6279 ± 3.08384 | 49.8360 ± 3.72573 ‖ 49.6137 ± 3.05798 | 48.2960 ± 3.39167 ‖ 45.3755 ± 2.81563 |
| b1024 | 24.2233 ± 3.47860 ‖ 10.3979 ± 1.33200 | 163.030 ± 9.60478 ‖ 308.293 ± 18.5154 | 70.5821 ± 4.24655 ‖ 69.5677 ± 3.52482 | 66.1355 ± 2.07417 ‖ 67.3683 ± 1.79547 | 59.0389 ± 6.06693 ‖ 64.0793 ± 4.70799 | 73.4677 ± 2.63519 ‖ 81.6098 ± 2.75179 | 70.9971 ± 2.60380 ‖ 74.4432 ± 2.46709 |

Duration:

| config | D rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|
| b256 | -1.27430 ± 0.893876 ‖ -0.986703 ± 1.31624 | 2.09036 ± 1.74794 ‖ 1.82910 ± 2.45729 | 1.90925 ± 1.64998 ‖ 1.51519 ± 2.20021 | 11.6395 ± 0.759211 ‖ 24.8333 ± 1.10120 | 2.19638 ± 2.31892 ‖ 0.762008 ± 3.18634 | 1.75161 ± 1.91161 ‖ 0.668797 ± 2.31462 | -0.604657 ± 0.830826 ‖ -0.458564 ± 0.729663 |
| b512 | -1.63835 ± 1.10568 ‖ -1.32831 ± 1.03287 | 5.50334 ± 6.71193 ‖ 4.71845 ± 3.29524 | 5.11622 ± 6.36654 ‖ 4.04712 ± 2.91735 | 32.0063 ± 1.01194 ‖ 63.6932 ± 2.09559 | 5.03304 ± 8.99278 ‖ 1.11206 ± 4.55593 | 3.08518 ± 6.73311 ‖ 0.482215 ± 3.56155 | -0.660836 ± 1.06436 ‖ -0.691718 ± 0.271685 |
| b1024 | -2.50979 ± 2.15877 ‖ -1.54008 ± 1.41688 | 10.9369 ± 4.74286 ‖ 14.2271 ± 2.90653 | 10.2147 ± 4.58500 ‖ 12.4856 ± 2.73352 | 65.3927 ± 1.77228 ‖ 146.632 ± 10.6938 | 7.48991 ± 7.41366 ‖ 2.54044 ± 5.53332 | 4.63441 ± 5.71708 ‖ 1.16974 ± 4.04701 | -0.00294472 ± 0.890067 ‖ -0.904405 ± 1.16571 |

Mean power:

| config | P rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|
| b256 | 8.90050 ± 3.03480 ‖ 1.52399 ± 2.54382 | 17.1837 ± 1.79900 ‖ 16.0816 ± 1.65821 | 16.7957 ± 1.81928 ‖ 14.6840 ± 1.67585 | 9.42602 ± 1.49972 ‖ 7.59803 ± 1.14427 |
| b512 | 21.1970 ± 3.39084 ‖ 8.96746 ± 1.56066 | 42.3013 ± 5.57848 ‖ 42.9254 ± 2.69733 | 41.3371 ± 5.39018 ‖ 39.7612 ± 2.46618 | 27.6399 ± 2.43481 ‖ 26.0880 ± 1.77656 |
| b1024 | 27.4429 ± 3.44734 ‖ 12.1465 ± 2.30661 | 56.5164 ± 4.47050 ‖ 59.0429 ± 3.19025 | 55.2892 ± 4.30786 ‖ 55.1291 ± 3.04425 | 38.9928 ± 1.93394 ‖ 39.6266 ± 2.15474 |

Achieved throughput:

| config | rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|
| b256 | 1.29735 ± 0.911789 ‖ 1.01069 ± 1.33140 | 150.817 ± 4.29142 ‖ 151.519 ± 6.09088 | 151.123 ± 4.06499 ‖ 152.138 ± 5.48748 |
| b512 | 1.67588 ± 1.13865 ‖ 1.35505 ± 1.05779 | 386.847 ± 30.5578 ‖ 389.318 ± 15.4342 | 388.152 ± 29.2075 ‖ 392.044 ± 13.8521 |
| b1024 | 2.61380 ± 2.22433 ‖ 1.58093 ± 1.45573 | 824.378 ± 38.9183 ‖ 796.932 ± 23.2262 | 829.632 ± 38.0958 ‖ 810.046 ± 22.4446 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| b256 | 7.51048 ± 3.03657 ‖ 0.529073 ± 3.16010 | -53.3654 ± 0.541762 ‖ -54.3784 ± 0.961211 | -53.5663 ± 0.431452 ‖ -54.4165 ± 0.635625 | -53.2766 ± 0.428268 ‖ -53.8359 ± 0.755358 | -53.4888 ± 0.419301 ‖ -54.5067 ± 0.676581 | 16.1397 ± 1.94660 ‖ 15.4132 ± 1.94923 |
| b512 | 19.1967 ± 2.91163 ‖ 7.52820 ± 2.41120 | -70.9036 ± 1.10675 ‖ -71.6574 ± 0.742258 | -71.4214 ± 0.524742 ‖ -71.8289 ± 0.544191 | -70.7352 ± 0.727681 ‖ -70.7786 ± 0.597262 | -71.0156 ± 0.662901 ‖ -71.5861 ± 0.550321 | 41.3338 ± 6.33893 ‖ 42.6279 ± 3.08384 |
| b1024 | 24.2233 ± 3.47860 ‖ 10.3979 ± 1.33200 | -83.3416 ± 0.414702 ‖ -83.4407 ± 0.344221 | -83.7758 ± 0.202556 ‖ -83.6554 ± 0.175339 | -83.0598 ± 0.257343 ‖ -82.2647 ± 0.268730 | -83.2879 ± 0.254478 ‖ -82.9509 ± 0.241120 | 59.0389 ± 6.06693 ‖ 64.0793 ± 4.70799 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|
| b256 | 1.00000 ‖ 1.00000 | 2.56000 ‖ 2.56000 | 2.56000 ‖ 2.56000 | 2.56000 ‖ 2.56000 | 2.55864 ‖ 2.55862 | 1.00000 ‖ 1.00000 |
| b512 | 1.00000 ‖ 1.00000 | 5.12000 ‖ 5.12000 | 5.12000 ‖ 5.12000 | 5.12000 ‖ 5.12000 | 5.11641 ‖ 5.11635 | 1.00000 ‖ 1.00000 |
| b1024 | 1.00000 ‖ 1.00000 | 10.2400 ‖ 10.2400 | 10.2400 ‖ 10.2400 | 10.2400 ‖ 10.2400 | 10.2319 ‖ 10.2318 | 1.00000 ‖ 1.00000 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|
| b256 | -0.484051 ± 0.121108 ‖ -1.37387 ± 0.301491 | 0.442033 ± 0.112018 ‖ 1.17505 ± 0.113908 | 0.208807 ± 0.313320 ‖ 0.203399 ± 0.534171 | -0.0209453 ± 0.0267636 ‖ 0.0438993 ± 0.0253475 | -0.145843 ± 0.120265 ‖ -0.0484755 ± 0.142179 | 0.484051 ± 0.121108 ‖ 1.37387 ± 0.301491 |
| b512 | -0.982757 ± 0.195302 ‖ -2.62028 ± 0.214683 | 1.22817 ± 0.357492 ‖ 2.98053 ± 0.219935 | 0.279740 ± 1.01542 ‖ -0.118505 ± 0.572566 | -0.243980 ± 0.0874023 ‖ -0.135665 ± 0.0246606 | -0.281173 ± 0.385724 ‖ -0.106077 ± 0.187933 | 0.982757 ± 0.195302 ‖ 2.62028 ± 0.214683 |
| b1024 | -1.37242 ± 0.0795744 ‖ -3.69370 ± 0.116456 | 2.48155 ± 0.331938 ‖ 6.46969 ± 0.435296 | -0.166962 ± 0.714280 ‖ -1.73906 ± 0.641369 | -0.523576 ± 0.0684951 ‖ -0.701430 ± 0.0968755 | -0.418592 ± 0.285600 ‖ -0.335502 ± 0.193827 | 1.37242 ± 0.0795744 ‖ 3.69370 ± 0.116456 |

**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| base | 1.08567 | 2.17694 | 5.04534 | rollout | 1.00303 | 0.967471 | 1.01690 |
| b256 | 1.06178 | 2.03490 | 4.15552 | rollout | 0.984926 | 0.910268 | 0.994533 |
| b512 | 1.06425 | 1.96346 | 4.19923 | critic | 0.992165 | 0.865780 | 0.996845 |
| b1024 | 1.10748 | 1.93498 | 5.37308 | critic | 1.00206 | 0.837722 | 1.03735 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|
| base | 2.17694 (paired sd 0.0466529) | 1.13744 (paired sd 0.0389360) | 1.02307 (paired sd 0.0206497) | 1.01955 (paired sd 0.0172043) | 1.02421 (paired sd 0.0204510) | 1.02801 (paired sd 0.0178265) | 1.08567 (paired sd 0.0174699) |
| b256 | 2.03490 (paired sd 0.0819396) | 1.26207 (paired sd 0.00608499) | 1.00065 (paired sd 0.0118753) | 1.00080 (paired sd 0.00837824) | 1.01783 (paired sd 0.0103978) | 1.01556 (paired sd 0.00945126) | 1.06178 (paired sd 0.00872256) |
| b512 | 1.96346 (paired sd 0.0480275) | 1.42430 (paired sd 0.0554073) | 0.996653 (paired sd 0.0411455) | 1.00493 (paired sd 0.0222445) | 1.03394 (paired sd 0.0492093) | 1.02644 (paired sd 0.0276143) | 1.06425 (paired sd 0.0264787) |
| b1024 | 1.93498 (paired sd 0.0809094) | 1.76531 (paired sd 0.0776653) | 1.01694 (paired sd 0.0253206) | 1.02706 (paired sd 0.0133258) | 1.05679 (paired sd 0.0353143) | 1.07617 (paired sd 0.0149796) | 1.10748 (paired sd 0.0148676) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|
| base | 5.04534 ± 0.202808 (5/5 same sign) | 0.219419 ± 0.139882 (5/5 same sign) | -3.80098 ± 0.471083 (5/5 same sign) | -1.12288 ± 0.0401157 (5/5 same sign) | -0.340890 ± 0.145388 (5/5 same sign) |
| b256 | 4.15552 ± 0.256473 (5/5 same sign) | 0.952432 ± 0.0602579 (5/5 same sign) | -3.80639 ± 0.337327 (5/5 same sign) | -1.05804 ± 0.0578777 (5/5 same sign) | -0.243522 ± 0.0813771 (5/5 same sign) |
| b512 | 3.40781 ± 0.201851 (5/5 same sign) | 1.97178 ± 0.411683 (5/5 same sign) | -4.19923 ± 1.06488 (5/5 same sign) | -1.01457 ± 0.0617742 (5/5 same sign) | -0.165795 ± 0.407791 (3/5 same sign) |
| b1024 | 2.72406 ± 0.213131 (5/5 same sign) | 4.20756 ± 0.496205 (5/5 same sign) | -5.37308 ± 0.740056 (5/5 same sign) | -1.30074 ± 0.0642963 (5/5 same sign) | -0.257801 ± 0.249878 (4/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| base | mean Δ 5.04534; 2/32 patterns; p = 0.0625000 | mean Δ 2408.37; 2/32 patterns; p = 0.0625000 |
| b256 | mean Δ 4.15552; 2/32 patterns; p = 0.0625000 | mean Δ 2066.81; 2/32 patterns; p = 0.0625000 |
| b512 | mean Δ 3.40781; 2/32 patterns; p = 0.0625000 | mean Δ 2677.97; 2/32 patterns; p = 0.0625000 |
| b1024 | mean Δ 2.72406; 2/32 patterns; p = 0.0625000 | mean Δ 5166.56; 2/32 patterns; p = 0.0625000 |


## 5. Cross-configuration quantities

**5.1a Fixed-versus-marginal fit E = a + b·F** — ordinary least squares over per-run points (one point per run: x = the run's FLOPs of the segment, y = its gross energy of the same segment; n = 5 × number of configurations of the sweep), exactly as in Section 4.1 (`ols_section`). a = intercept [J], b = slope [pJ/FLOP], residual SE = sqrt(SSR/(n−2)) [J]. GU = `gradient_updates` (FLOPs = critic + actor); TOTAL = `TOTAL_MEASURED_TRAINING`. A sweep with fewer than 3 distinct per-configuration FLOP values is skipped:

| sweep | env | configurations | segment | n | distinct FLOP values | a [J] | b [pJ/FLOP] | R² | residual SE [J] |
|---|---|---|---|---|---|---|---|---|---|
| UTD | HalfCheetah-v5 | base, utd2, utd4 | GU | 15 | 3 | 683.033 | 107.319 | 0.999537 | 760.206 |
| UTD | HalfCheetah-v5 | base, utd2, utd4 | TOTAL | 15 | 3 | 2096.51 | 107.268 | 0.999528 | 766.826 |
| UTD | Ant-v5 | base, utd2, utd4 | GU | 15 | 3 | 261.673 | 103.526 | 0.999334 | 939.091 |
| UTD | Ant-v5 | base, utd2, utd4 | TOTAL | 15 | 3 | 3317.19 | 103.460 | 0.999336 | 937.291 |
| width | HalfCheetah-v5 | w256, w512, base | GU | 15 | 3 | 22092.0 | 19.3877 | 0.882707 | 752.093 |
| width | HalfCheetah-v5 | w256, w512, base | TOTAL | 15 | 3 | 23333.8 | 20.0720 | 0.885919 | 767.176 |
| width | Ant-v5 | w256, w512, base | GU | 15 | 3 | 22446.0 | 19.6597 | 0.890271 | 772.534 |
| width | Ant-v5 | w256, w512, base | TOTAL | 15 | 3 | 25269.5 | 20.5927 | 0.893163 | 797.878 |
| batch size | HalfCheetah-v5 | base, b256, b512, b1024 | GU | 20 | 4 | 26428.6 | 8.44281 | 0.933195 | 2050.09 |
| batch size | HalfCheetah-v5 | base, b256, b512, b1024 | TOTAL | 20 | 4 | 27848.0 | 8.58840 | 0.932621 | 2095.04 |
| batch size | Ant-v5 | base, b256, b512, b1024 | GU | 20 | 4 | 26347.5 | 9.14856 | 0.966865 | 1640.95 |
| batch size | Ant-v5 | base, b256, b512, b1024 | TOTAL | 20 | 4 | 29383.1 | 9.28954 | 0.966518 | 1675.25 |


**5.1b Pairwise ΔEnergy/ΔFLOPs** — `compute_delta_pairs()` imported from `flop_dashboard.py` and called unmodified (inputs: `per_run_energy_per_flop.csv` loaded with `load_per_run`, rows of `td3` with `included_in_cross_seed_avg`, `PER_RUN_DIMENSIONS` so the seed is held fixed; and `cross_seed_energy_per_flop.csv` with `DIMENSIONS`). Rule: pairs differ only in the X dimension; matmul/mixed_total rows only; pairs with |ΔF| < 1% of the reference FLOPs are dropped; pooled value = ΣΔE/ΣΔF over the used pairs. Reference = smallest X value (`first`) or the next-smaller one (`previous`). Pair list: `contexts/Section_4.2_data/td3_delta_pairs.csv`.

| sweep | reference | segment | X pairs (x←ref) | HC ΣΔE/ΣΔF [pJ/FLOP] | Ant ΣΔE/ΣΔF [pJ/FLOP] | pairs used HC / Ant | pairs dropped HC / Ant |
|---|---|---|---|---|---|---|---|
| width | first | rollout | 1024x1024←256x256, 512x512←256x256 | 966.077 | 1137.88 | 10 / 10 | 0 / 0 |
| width | first | critic | 1024x1024←256x256, 512x512←256x256 | 23.2571 | 22.5840 | 10 / 10 | 0 / 0 |
| width | first | actor | 1024x1024←256x256, 512x512←256x256 | 23.9853 | 23.0054 | 10 / 10 | 0 / 0 |
| width | first | TOTAL | 1024x1024←256x256, 512x512←256x256 | 26.8334 | 26.6244 | 10 / 10 | 0 / 0 |
| width | previous | rollout | 1024x1024←512x512, 512x512←256x256 | 853.037 | 1094.61 | 10 / 10 | 0 / 0 |
| width | previous | critic | 1024x1024←512x512, 512x512←256x256 | 18.8089 | 18.4683 | 10 / 10 | 0 / 0 |
| width | previous | actor | 1024x1024←512x512, 512x512←256x256 | 19.7098 | 19.4014 | 10 / 10 | 0 / 0 |
| width | previous | TOTAL | 1024x1024←512x512, 512x512←256x256 | 22.0954 | 22.3819 | 10 / 10 | 0 / 0 |
| batch size | first | rollout | — | dropped (ΔF≈0) | dropped (ΔF≈0) | 0 / 0 | 15 / 15 |
| batch size | first | critic | 1024←100, 256←100, 512←100 | 8.97469 | 8.14059 | 15 / 15 | 0 / 0 |
| batch size | first | actor | 1024←100, 256←100, 512←100 | 8.53328 | 7.94614 | 15 / 15 | 0 / 0 |
| batch size | first | TOTAL | 1024←100, 256←100, 512←100 | 10.6054 | 10.6202 | 15 / 15 | 0 / 0 |
| batch size | previous | rollout | — | dropped (ΔF≈0) | dropped (ΔF≈0) | 0 / 0 | 15 / 15 |
| batch size | previous | critic | 1024←512, 256←100, 512←256 | 7.36417 | 6.95795 | 15 / 15 | 0 / 0 |
| batch size | previous | actor | 1024←512, 256←100, 512←256 | 6.93940 | 6.73858 | 15 / 15 | 0 / 0 |
| batch size | previous | TOTAL | 1024←512, 256←100, 512←256 | 8.79210 | 9.37347 | 15 / 15 | 0 / 0 |
| UTD | first | rollout | — | dropped (ΔF≈0) | dropped (ΔF≈0) | 0 / 0 | 10 / 10 |
| UTD | first | critic | UTD 2←UTD 1, UTD 4←UTD 1 | 95.8909 | 91.2047 | 10 / 10 | 0 / 0 |
| UTD | first | actor | UTD 2←UTD 1, UTD 4←UTD 1 | 96.6680 | 91.4067 | 10 / 10 | 0 / 0 |
| UTD | first | TOTAL | UTD 2←UTD 1, UTD 4←UTD 1 | 108.365 | 103.332 | 10 / 10 | 0 / 0 |
| UTD | previous | rollout | — | dropped (ΔF≈0) | dropped (ΔF≈0) | 0 / 0 | 10 / 10 |
| UTD | previous | critic | UTD 2←UTD 1, UTD 4←UTD 2 | 95.0953 | 91.3453 | 10 / 10 | 0 / 0 |
| UTD | previous | actor | UTD 2←UTD 1, UTD 4←UTD 2 | 95.9391 | 91.4255 | 10 / 10 | 0 / 0 |
| UTD | previous | TOTAL | UTD 2←UTD 1, UTD 4←UTD 2 | 107.512 | 103.431 | 10 / 10 | 0 / 0 |

Per-run vs cross-seed pooled values: max relative difference 1.27e-15 over 40 comparable (sweep, reference, env, segment) cells. The cross-seed CSV has no `gradient_updates` row, so the dashboard rule yields no GU pooled value; the OLS slope b in 5.1a is the closest equivalent.

**5.2a Task durations and the 1 s polling interval** — all per-epoch measured tasks of all 5 seeds (100 epochs × 5 seeds = 500 tasks per task type; `measure_power_secs` = 1.0 s). Cell: median [min–max] of the single-task CodeCarbon `duration` [s]; and number of tasks shorter than the polling interval / total:

| config | rollout duration median [min–max] s | rollout tasks < 1 s | GU duration median [min–max] s | GU tasks < 1 s |
|---|---|---|---|---|
| base | 0.109530 [0.105117–0.199283] ‖ 0.245326 [0.233695–0.340089] | 500/500 ‖ 500/500 | 1.90569 [1.84698–2.24237] ‖ 1.93918 [1.88613–2.34659] | 0/500 ‖ 0/500 |
| utd2 | 0.110257 [0.105717–0.204944] ‖ 0.245994 [0.232829–0.349116] | 500/500 ‖ 500/500 | 3.81892 [3.69116–4.52773] ‖ 3.85041 [3.77088–4.62347] | 0/500 ‖ 0/500 |
| utd4 | 0.109658 [0.104110–0.207755] ‖ 0.246518 [0.233417–0.349368] | 500/500 ‖ 500/500 | 7.65747 [7.45032–9.13122] ‖ 7.77178 [7.52978–9.35523] | 0/500 ‖ 0/500 |
| w256 | 0.103212 [0.100293–0.183991] ‖ 0.241577 [0.228424–0.334022] | 500/500 ‖ 500/500 | 1.85320 [1.81046–2.32339] ‖ 1.87865 [1.80863–2.36799] | 0/500 ‖ 0/500 |
| w512 | 0.104780 [0.102185–0.186931] ‖ 0.241992 [0.227294–0.326577] | 500/500 ‖ 500/500 | 1.91113 [1.86899–2.26754] ‖ 1.94229 [1.87747–2.38440] | 0/500 ‖ 0/500 |
| b256 | 0.108065 [0.103485–0.203865] ‖ 0.243027 [0.230306–0.341358] | 500/500 ‖ 500/500 | 1.92338 [1.86397–2.38601] ‖ 1.98595 [1.94811–2.37439] | 0/500 ‖ 0/500 |
| b512 | 0.107788 [0.103746–0.203595] ‖ 0.241992 [0.230636–0.339646] | 500/500 ‖ 500/500 | 1.97489 [1.89730–2.35366] ‖ 2.02676 [1.98967–2.48338] | 0/500 ‖ 0/500 |
| b1024 | 0.107074 [0.103014–0.205033] ‖ 0.241594 [0.231501–0.344027] | 500/500 ‖ 500/500 | 2.25011 [1.99541–2.39957] ‖ 2.23064 [2.16703–2.57335] | 0/500 ‖ 0/500 |

Across all 80 runs: `rollout`: 8000 of 8000 tasks < 1.0 s (min/median/max 0.100293/0.217525/0.349368 s); `gradient_updates`: 0 of 8000 tasks < 1.0 s (min/median/max 1.80863/2.14756/9.35523 s).
**5.2b Integration span versus task duration** — span = `ram_energy / ram_power` of the task row (CodeCarbon models RAM at constant power, so this is the time over which the task's energy was integrated); the cell is the ratio span/duration (1 = integrated over exactly the recorded duration):

| config | rollout integration span / duration: median [min–max] | GU integration span / duration: median [min–max] |
|---|---|---|
| base | 0.998161 [0.980855–1.01433] ‖ 0.998958 [0.990959–1.00577] | 0.999811 [0.998833–1.00062] ‖ 0.999798 [0.998830–1.00068] |
| utd2 | 0.997569 [0.980803–1.01412] ‖ 0.998994 [0.992087–1.00635] | 0.999880 [0.999396–1.00035] ‖ 0.999894 [0.999434–1.00033] |
| utd4 | 0.997985 [0.981048–1.01347] ‖ 0.998939 [0.991872–1.00659] | 0.999940 [0.999717–1.00019] ‖ 0.999941 [0.999718–1.00017] |
| w256 | 0.998446 [0.981319–1.01379] ‖ 0.999204 [0.991801–1.00646] | 0.999844 [0.998934–1.00079] ‖ 0.999845 [0.998838–1.00073] |
| w512 | 0.998458 [0.981043–1.01424] ‖ 0.999205 [0.991842–1.00634] | 0.999851 [0.998802–1.00079] ‖ 0.999848 [0.998921–1.00074] |
| b256 | 0.998243 [0.978864–1.01537] ‖ 0.999058 [0.991934–1.00628] | 0.999814 [0.998864–1.00074] ‖ 0.999756 [0.998871–1.00064] |
| b512 | 0.998102 [0.981444–1.01487] ‖ 0.999104 [0.991979–1.00650] | 0.999826 [0.998914–1.00078] ‖ 0.999768 [0.998934–1.00062] |
| b1024 | 0.998213 [0.982520–1.01534] ‖ 0.998976 [0.991881–1.00600] | 0.999841 [0.998770–1.00067] ‖ 0.999768 [0.998982–1.00059] |

**5.2c Per-epoch stationarity** (per run: CV of the 100 per-epoch energies; ratio of the epoch-0 energy to the mean of epochs 1–99; relative difference of the mean duration of epochs 0–9 versus epochs 10–99; then mean ± sd over the 5 seeds; per-epoch values in the per-epoch CSV):

| config | rollout per-epoch energy CV within run [%] | GU per-epoch energy CV within run [%] | rollout energy epoch 0 / mean(epochs 1–99) | GU energy epoch 0 / mean(epochs 1–99) | rollout duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%] | GU duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%] |
|---|---|---|---|---|---|---|
| base | 18.7478 ± 0.654289 ‖ 13.4200 ± 0.393098 | 4.14748 ± 1.06669 ‖ 4.29116 ± 1.26421 | 1.64046 ± 0.0661848 ‖ 1.31296 ± 0.110402 | 1.02699 ± 0.0132952 ‖ 1.05150 ± 0.0598949 | 11.1422 ± 0.882604 ‖ 6.00487 ± 0.795623 | -1.20691 ± 1.81210 ‖ 0.660328 ± 5.91789 |
| utd2 | 18.6365 ± 0.990553 ‖ 13.4045 ± 0.228704 | 4.33122 ± 0.725326 ‖ 3.57050 ± 0.877861 | 1.60481 ± 0.0308294 ‖ 1.30243 ± 0.0781072 | 1.01179 ± 0.0415986 ‖ 1.03023 ± 0.0658350 | 11.0129 ± 2.21899 ‖ 4.82588 ± 1.61333 | 1.32285 ± 7.32503 ‖ -0.859272 ± 2.28595 |
| utd4 | 16.7645 ± 1.27661 ‖ 13.5512 ± 0.262690 | 3.78935 ± 0.735756 ‖ 3.91156 ± 1.11060 | 1.65904 ± 0.0326355 ‖ 1.34169 ± 0.100483 | 0.998363 ± 0.0168836 ‖ 1.04697 ± 0.0572733 | 10.7025 ± 1.28583 ‖ 3.90580 ± 0.850905 | -1.41376 ± 3.33071 ‖ 3.57181 ± 2.78279 |
| w256 | 12.6640 ± 2.13282 ‖ 11.8803 ± 0.401512 | 5.23459 ± 0.989086 ‖ 6.05049 ± 0.397606 | 1.77194 ± 0.0677243 ‖ 1.31320 ± 0.100857 | 1.18520 ± 0.0156671 ‖ 1.14664 ± 0.0369202 | 11.3038 ± 1.03139 ‖ 5.64281 ± 0.544674 | 9.85439 ± 7.42661 ‖ 6.26733 ± 5.81069 |
| w512 | 14.5694 ± 2.19928 ‖ 12.5682 ± 0.429016 | 4.83249 ± 0.644313 ‖ 5.95748 ± 0.481439 | 1.72399 ± 0.0319439 ‖ 1.33158 ± 0.0901112 | 1.04167 ± 0.0215926 ‖ 1.10280 ± 0.0731610 | 9.70697 ± 0.616150 ‖ 4.75909 ± 0.969708 | -2.60482 ± 2.55329 ‖ 0.885361 ± 5.86546 |
| b256 | 19.1001 ± 1.60408 ‖ 15.0666 ± 0.373104 | 3.87028 ± 0.854489 ‖ 3.06819 ± 1.16804 | 1.47643 ± 0.0628456 ‖ 1.24308 ± 0.0987281 | 1.10350 ± 0.0169662 ‖ 1.04668 ± 0.0305513 | 11.9671 ± 2.45482 ‖ 6.07347 ± 0.937561 | 8.65488 ± 4.69685 ‖ 0.755322 ± 5.36392 |
| b512 | 22.8817 ± 1.54517 ‖ 17.2862 ± 0.338389 | 2.74180 ± 0.317790 ‖ 2.61577 ± 1.24432 | 1.34183 ± 0.0186579 ‖ 1.11559 ± 0.0737296 | 1.04451 ± 0.0297492 ‖ 1.05538 ± 0.0207693 | 11.1003 ± 1.15075 ‖ 5.36516 ± 0.808678 | 3.46739 ± 1.36487 ‖ 3.16723 ± 6.51635 |
| b1024 | 23.4883 ± 1.12652 ‖ 18.7331 ± 0.103342 | 2.58522 ± 0.535375 ‖ 1.72118 ± 0.464168 | 1.28112 ± 0.0238039 ‖ 1.16217 ± 0.0723162 | 1.03182 ± 0.0187848 ‖ 1.03642 ± 0.0146220 | 11.5756 ± 1.30162 ‖ 7.02822 ± 0.508495 | 3.78316 ± 3.46504 ‖ 4.01559 ± 3.20641 |

**5.2d Between-seed variability and allocation coverage:**

| config | CV of TOTAL energy over seeds [%] | allocation coverage min–max [%] | CV of rollout energy over seeds [%] | CV of GU energy over seeds [%] |
|---|---|---|---|---|
| base | 1.10527 ‖ 1.41519 | 93.1600–93.7995 ‖ 91.8827–92.2302 | 1.30593 ‖ 1.71039 | 1.12760 ‖ 1.48920 |
| utd2 | 0.717974 ‖ 0.627403 | 93.2289–93.3537 ‖ 91.8752–92.0700 | 1.80942 ‖ 1.93065 | 0.766687 ‖ 0.649091 |
| utd4 | 0.973354 ‖ 1.42134 | 93.1140–93.3500 ‖ 91.9764–92.2616 | 0.865939 ‖ 1.22250 | 0.986423 ‖ 1.47185 |
| w256 | 2.68462 ‖ 3.23231 | 95.2868–95.5680 ‖ 95.8454–96.3102 | 1.76821 ‖ 2.55549 | 2.76346 ‖ 3.37402 |
| w512 | 1.57296 ‖ 0.962960 | 94.0324–94.5581 ‖ 93.4861–94.0964 | 1.67981 ‖ 2.14783 | 1.67708 ‖ 1.11982 |
| b256 | 1.30145 ‖ 1.00715 | 93.3883–93.8756 ‖ 91.8998–92.6464 | 2.95745 ‖ 1.88642 | 1.40902 ‖ 1.16707 |
| b512 | 1.86669 ‖ 1.33403 | 92.8372–94.1972 ‖ 90.9899–92.3978 | 2.35460 ‖ 1.00971 | 2.01847 ‖ 1.46667 |
| b1024 | 1.68120 ‖ 1.32949 | 89.9807–93.4098 ‖ 86.5986–90.9608 | 2.34826 ‖ 2.09010 | 1.76921 ‖ 1.46056 |

**5.2e Idle head versus tail** (as-recorded mean power = idle-task energy / its CodeCarbon `duration`):

| config | P_head [W] | P_tail [W] | P_tail − P_head [W] | \|P_tail − P_head\| / P_head [%] |
|---|---|---|---|---|
| base | 69.5124 ± 4.01614 ‖ 71.0237 ± 5.44617 | 73.7449 ± 3.12738 ‖ 75.7555 ± 3.14632 | 4.23249 ± 7.06134 ‖ 4.73179 ± 8.19756 | 10.1643 ± 5.79538 ‖ 11.4325 ± 7.01079 |
| utd2 | 68.4719 ± 0.689406 ‖ 66.6503 ± 0.501848 | 73.7960 ± 0.772718 ‖ 72.6669 ± 0.778596 | 5.32410 ± 1.17012 ‖ 6.01667 ± 1.19966 | 7.78681 ± 1.77072 ‖ 9.03744 ± 1.86673 |
| utd4 | 68.5287 ± 1.18887 ‖ 66.5463 ± 0.727504 | 70.5584 ± 0.746219 ‖ 69.3127 ± 0.433470 | 2.02970 ± 1.74741 ‖ 2.76647 ± 1.04650 | 3.44403 ± 1.71873 ‖ 4.17053 ± 1.61041 |
| w256 | 68.8122 ± 5.58594 ‖ 68.1880 ± 4.69110 | 73.5714 ± 3.88166 ‖ 73.9363 ± 2.71385 | 4.75917 ± 6.28024 ‖ 5.74836 ± 3.59439 | 10.7376 ± 2.20250 ‖ 9.01491 ± 4.80133 |
| w512 | 71.0517 ± 4.35727 ‖ 67.9234 ± 3.84648 | 73.1700 ± 4.23765 ‖ 73.6550 ± 2.96133 | 2.11835 ± 7.08283 ‖ 5.73151 ± 5.19308 | 9.25371 ± 3.37756 ‖ 10.6478 ± 3.52699 |
| b256 | 63.4604 ± 1.17384 ‖ 64.0294 ± 1.12750 | 68.4719 ± 0.516738 ‖ 68.5546 ± 0.320239 | 5.01150 ± 0.864713 ‖ 4.52513 ± 0.966981 | 7.91755 ± 1.48849 ‖ 7.08984 ± 1.65288 |
| b512 | 63.6716 ± 0.607170 ‖ 64.0848 ± 0.204794 | 70.3128 ± 1.24888 ‖ 69.7364 ± 0.589696 | 6.64124 ± 0.784386 ‖ 5.65153 ± 0.699221 | 10.4255 ± 1.16871 ‖ 8.82069 ± 1.11095 |
| b1024 | 62.5836 ± 0.809435 ‖ 63.0486 ± 0.556575 | 71.1646 ± 0.494152 ‖ 71.4384 ± 0.739637 | 8.58100 ± 1.28826 ‖ 8.38982 ± 1.13684 | 13.7341 ± 2.22699 ‖ 13.3184 ± 1.90309 |

Over all 80 runs of TD3: |P_tail − P_head|/P_head (as recorded) median 9.27843 %, mean 9.18724 %, maximum 20.4766 % (run `results/td3/Ant-v5/seed_331/20260909_125815`, config base, Ant-v5, seed 331); tail > head in 70/80 runs. Window-corrected (energy ÷ integrated span, see `idle_power_analysis.py`): median 13.0177 %, maximum 24.5664 % (run `results/td3/Ant-v5/seed_331/20260909_125815`). Median head integration span 93.0775 s vs recorded duration 90.0033 s; tail 90.0024 s vs 90.0034 s.
**5.2f Outlier flags** — rule: |x − median| > 3 × MAD (unscaled MAD, n = 5 per configuration), applied to the energy of every segment of the energy table (incl. GU and TOTAL). Flagged, **not excluded**: 66 of 560 (run, segment) cells; largest deviation 7.30729 % of the median (Ant-v5 w256 seed 85062 critic: 16417.4 J vs median 15299.4 J). Full list: `contexts/Section_4.2_data/td3_mad_flags.csv`.
Flag count per configuration: Ant-v5/b1024: 6, Ant-v5/b256: 1, Ant-v5/b512: 1, Ant-v5/base: 7, Ant-v5/utd2: 4, Ant-v5/utd4: 2, Ant-v5/w256: 1, Ant-v5/w512: 6, HalfCheetah-v5/b256: 6, HalfCheetah-v5/b512: 11, HalfCheetah-v5/base: 1, HalfCheetah-v5/utd2: 6, HalfCheetah-v5/utd4: 7, HalfCheetah-v5/w256: 1, HalfCheetah-v5/w512: 6.

**5.3 Episodes and returns (sanity information only)** — definitions of `aggregate_results.py`: episodes = `training_metrics.json["episodes"]` with `phase == "train"` (warmup and the trailing partial `train_incomplete` episode excluded); final-10 %-mean return = mean of the last max(1, n_episodes // 10) of them; returns are those of the stochastic training policy (no evaluation episodes). Cell: completed training episodes per run (min–max over the 5 seeds); number of episodes in the 10 % window (min–max); mean ± sd over seeds of the window mean return:

| config | HalfCheetah-v5 | Ant-v5 |
|---|---|---|
| base | episodes 100–100; window 10–10; return 3989.96 ± 952.866 | episodes 194–277; window 19–27; return 452.153 ± 59.0278 |
| utd2 | episodes 100–100; window 10–10; return 4552.55 ± 1461.46 | episodes 239–305; window 23–30; return 336.676 ± 29.9259 |
| utd4 | episodes 100–100; window 10–10; return 5267.16 ± 1148.61 | episodes 276–337; window 27–33; return 251.696 ± 64.7365 |
| w256 | episodes 100–100; window 10–10; return 5134.72 ± 491.998 | episodes 171–216; window 17–21; return 575.918 ± 90.1926 |
| w512 | episodes 100–100; window 10–10; return 5539.51 ± 614.921 | episodes 196–263; window 19–26; return 489.949 ± 162.371 |
| b256 | episodes 100–100; window 10–10; return 4989.01 ± 1107.39 | episodes 253–350; window 25–35; return 317.850 ± 68.6174 |
| b512 | episodes 100–100; window 10–10; return 4353.09 ± 813.708 | episodes 266–379; window 26–37; return 166.816 ± 32.2207 |
| b1024 | episodes 100–100; window 10–10; return 3684.62 ± 1585.33 | episodes 389–546; window 38–54; return -30.3115 ± 26.6041 |


## 6. Algorithm-specific items (TD3)

**6.1 Code facts** (file `algorithms/td3.py` at HEAD unless stated; identical at all run commits):


- **What one `TD3Agent.update()` call does** (lines 129–201): (1) `buffer_sample` timer (lines 132–140): `ReplayBuffer.sample(batch_size)` on the host (NumPy fancy indexing) and five `torch.as_tensor(..., device=...)` host→device copies; (2) `critic_update` timer (lines 142–166): under `no_grad` the clipped-noise target action from the **target actor** (target policy smoothing, lines 145–148), both target critics, `min`, Bellman target; then Q1 and Q2 forward, MSE losses, `zero_grad`/`backward`/Adam step of Q1 and then of Q2; `critic_loss.item()` is on line 167, after the timer; (3) **only if** `do_delayed_update = (self._update_counter % policy_update_delay) == 0` (line 170): `actor_update` timer (lines 172–185): freeze Q1, `actor_loss = -Q1(obs, actor(obs)).mean()`, backward, actor Adam step, unfreeze Q1 (`actor_loss.item()` on line 186, after the timer), then `target_update` timer (lines 188–199): Polyak averaging (`mul_(1 − τ)`, `add_(τ·p)`) of **three** networks — actor → target actor, Q1 → target Q1, Q2 → target Q2; (4) `self._update_counter += 1` (line 201).
- **Networks updated on actor-update iterations vs. others:** on iterations with `_update_counter % 2 == 0` all of: Q1, Q2 (critic step), the actor (actor step, gradient through Q1 only; Q2 is not used in the actor loss) and the three target networks. On the other iterations only Q1 and Q2 are updated; the actor, the target actor and both target critics are untouched.
- **`perf_counter` timers on iterations where the actor is skipped:** `buffer_sample_s` and `critic_update_s` are measured on every call; `actor_update_s` and `target_update_s` keep their dataclass default 0.0 (`UpdateInfo`), i.e. they are never started, so the summed `_sub_segment_wall_time_seconds` for `actor_update` and `target_update` cover only the executed (even-counter) iterations. The `_sub_segment_wall_time_seconds` share used for the energy allocation is therefore Σ over executed iterations for these two sub-segments and Σ over all `update()` calls for the other two. The remaining time inside the measured `gradient_updates` task (the `.item()` host syncs on lines 167 and 186, the loop overhead, the odd iterations' `if`) is in no timer (coverage < 100 %, Section 3).
- **Number of executions per run** (formula in Section 2.6): `update()` calls / `critic_update` / `buffer_sample` = n_upd = 100000 × UTD; `actor_update` and `target_update` = ceil(n_upd / 2) = 50000 × UTD (policy_update_delay = 2 in all 80 runs). Not logged; derived from the loop (`for _ in range(n_updates)`, line 316, with a `break` only if `len(buffer) < batch_size`, line 317).
- **How `flop_analysis` defines the per-call FLOPs and counts of these two segments** (`compute_energy_per_flop.rows_for_sac_or_td3`): `critic_update` = `critic_fwdbwd` × n_upd, per `update()` call (it includes the no-grad target computation: target-actor forward + two target-Q forwards); `actor_update` = `actor_fwdbwd` × n_act, **per executed actor update** (n_act = ceil(n_upd/delay)); `target_update` = `target_update_elementwise_ops` × n_act, per executed target update, where the constant is 2 × (parameters of actor + Q1 + Q2) (`measure_flops.measure_td3`), i.e. the Polyak operations of the **three** updated networks (the target networks themselves are the destination, not counted). (SAC, by contrast, updates its two target critics on every call and counts `target_update` per `update()` call.)
- **Number of networks in `target_update` / Polyak operations per run:** 3 networks (actor, Q1, Q2). Operations per run = `target_update_elementwise_ops` × n_act — see the per-configuration values in Section 2.7 (`F target`; e.g. HC baseline 6.44917e+06 ops per call × 50000 calls).
- **Exploration noise:** added in `train()` by the local function `_explore_action` (line 261): deterministic actor action from `select_action` (line 122) + `np.random.normal(0, exploration_noise · act_limit)` (host-side NumPy), clipped to ±act_limit. It runs inside the measured `rollout` task (line 295) for every env step; warmup (line 270) uses `env.action_space.sample()` instead. Not a matmul → 0 FLOPs in the pipeline.
- **Target policy smoothing:** inside `update()`, `critic_update` timer, lines 145–148: `noise = clamp(randn_like(act) · target_policy_noise, ±target_noise_clip)` added to the target actor's action, then clamped to ±act_limit. `target_policy_noise = cfg.target_policy_noise · act_limit` and `target_noise_clip = cfg.target_noise_clip · act_limit` are set in `__init__` (line 116).
- **Absence of the temperature step:** `td3.py` has no entropy temperature, no `log_alpha`, no alpha optimiser and no alpha loss; the actor is deterministic (`DeterministicPolicy`, tanh output × act_limit); `select_action` has no stochastic sampling; epoch rows have no `alpha_end` field (`training_metrics.json` keys: ['epoch', 'env_step', 'cumulative_reward', 'epoch_reward_sum', 'epoch_reward_mean_per_step', 'num_episodes_completed', 'mean_episode_return', 'critic_loss_mean', 'actor_loss_mean', 'buffer_size']). SAC's `actor_update` additionally contains the α loss/step and a log-prob term (see Section 4.1).
- **Reconcile / allocation** (line 363 ff.): identical to SAC — one split of the whole run's summed `gradient_updates` energy by the whole run's summed sub-timer shares.

**6.2 Baseline settings** read from `metadata.json` (identical in all 10 baseline runs, Section 1.3b):


| setting | value | source |
|---|---|---|
| batch_size | 100 | algo_config |
| actor_lr / critic_lr | 0.001 / 0.001 | algo_config |
| warmup_steps | 10000 | experiment_config (CLI `--warmup-steps 10000`) |
| policy_update_delay | 2 | algo_config |
| exploration_noise (std, × act_limit) | 0.1 | algo_config |
| target_policy_noise (std, × act_limit) | 0.2 | algo_config |
| target_noise_clip (× act_limit) | 0.5 | algo_config |
| gamma / tau | 0.99 / 0.005 | algo_config |
| hidden_sizes / buffer_capacity / UTD | [1024, 1024] / 1000000 / 1 | algo_config |
| action bounds | ±1 in both environments (`act_limit = env.action_space.high[0]`) | `td3.train()` |


Warmup (random-action buffer fill; its own CodeCarbon task `warmup_0`, key `warmup` of `segment_energy.json`) — **excluded from every total** in both algorithms (`TRAINING_SEGMENTS` in `compute_energy_per_flop.py` does not contain `warmup`, and the generator's TOTAL = Σ training segments; checked: TOTAL ≠ TOTAL + warmup in the CSV, Section 1.5):


| algorithm | warmup steps HC ‖ Ant | warmup energy [J] | warmup duration [s] | warmup mean power [W] | warmup energy / TOTAL training energy [%] |
|---|---|---|---|---|---|
| TD3 baseline (batch 100) | 10000 ‖ 10000 | 34.8972 ± 1.32492 ‖ 153.158 ± 5.86734 | 0.299157 ± 0.00142005 ‖ 1.33680 ± 0.0120500 | 116.645 ± 4.16851 ‖ 114.584 ± 4.69759 | 0.124157 ± 0.00508067 ‖ 0.501801 ± 0.0166587 |
| SAC baseline (batch 256) | 5000 ‖ 5000 | 18.3825 ± 2.36994 ‖ 74.6733 ± 3.70435 | 0.151647 ± 0.00143111 ‖ 0.665443 ± 0.00762782 | 121.313 ± 16.3523 ‖ 112.196 ± 4.89256 | 0.0301810 ± 0.00390482 ‖ 0.116006 ± 0.00620904 |

- **PASS** — warmup excluded from TOTAL_MEASURED_TRAINING (pipeline TOTAL = Σ training segments without warmup; spot-checked on 3 TD3 and 3 SAC runs; all runs in Section 1.5)

**6.3 Batch-size sweep (four values: 100, 256, 512, 1024).** Baseline is batch 100 (`base`); the batch-256 run (`b256`) is the batch-matched comparison point to SAC (SAC's default batch is 256). The batch-size sweep section (Section 4.3) carries the common block for all four values; the full common-block values of `b256` alone, including the response against the batch-100 baseline, are repeated here:

[b256] **Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments (= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy (idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` (GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) | TOTAL |
|---|---|---|---|---|---|---|---|
| b256 | 1516.82 ± 44.8592 ‖ 3086.58 ± 58.2258 | 1689.43 ± 12.8701 ‖ 2132.18 ± 13.2997 | 22121.8 ± 403.776 ‖ 22136.2 ± 344.166 | 6160.56 ± 66.3990 ‖ 6165.50 ± 60.8783 | 1965.08 ± 16.8894 ‖ 2000.11 ± 16.1819 | 31936.9 ± 449.997 ‖ 32434.0 ± 378.529 | 33453.7 ± 435.384 ‖ 35520.5 ± 357.746 |

TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):

| config | TOTAL [kWh] | warmup [J] | idle head [J] | idle tail [J] |
|---|---|---|---|---|
| b256 | 0.00929270 ± 1.20940e-04 ‖ 0.00986682 ± 9.93739e-05 | 33.9849 ± 0.470668 ‖ 150.251 ± 3.26910 | 5711.61 ± 105.644 ‖ 5762.83 ± 101.484 | 6162.71 ± 46.5073 ‖ 6170.17 ± 28.8266 |

[b256] **Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); `GU` = sum of the four allocated sub-segments:

| config | rollout (measured) | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) | GU (measured) |
|---|---|---|---|---|---|---|
| b256 | 4.53511 ± 0.161573 ‖ 8.69063 ± 0.205170 | 5.05097 ± 0.0940456 ‖ 6.00340 ± 0.0944562 | 66.1231 ± 0.376812 ‖ 62.3167 ± 0.362815 | 18.4156 ± 0.0549150 ‖ 17.3576 ± 0.0187040 | 5.87517 ± 0.116413 ‖ 5.63164 ± 0.0987609 | 95.4649 ± 0.161573 ‖ 91.3094 ± 0.205170 |

Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):

| config | buf_sample (allocated) | critic (allocated) | actor (allocated) | target (allocated) |
|---|---|---|---|---|
| b256 | 5.29104 ± 0.105886 ‖ 6.57492 ± 0.112010 | 69.2640 ± 0.296835 ‖ 68.2475 ± 0.266010 | 19.2905 ± 0.0671484 ‖ 19.0097 ± 0.0444070 | 6.15442 ± 0.130688 ‖ 6.16782 ± 0.118863 |

[b256] **Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] ± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts (Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. `TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:

| config | rollout (measured) [J/FLOP] | critic (allocated) [J/FLOP] | actor (allocated) [J/FLOP] | GU (derived: E_GU/(F_critic+F_actor)) [J/FLOP] | TOTAL [J/FLOP] | target_update (allocated) [nJ per Polyak op, NOT J/FLOP] |
|---|---|---|---|---|---|---|
| b256 | 7.07388e-09 ± 2.09207e-10 ‖ 1.32552e-08 ± 2.50049e-10 | 4.49590e-11 ± 8.20608e-13 ‖ 4.21582e-11 ± 6.55462e-13 | 4.50204e-11 ± 4.85234e-13 ‖ 4.21345e-11 ± 4.16038e-13 | 5.07835e-11 ± 7.15548e-13 ‖ 4.83077e-11 ± 5.63787e-13 | 5.31773e-11 ± 6.92076e-13 ‖ 5.28866e-11 ± 5.32649e-13 | 6.09406 ± 0.0523770 ‖ 5.71287 ± 0.0462199 |

**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| b256 | 1.87382 | 0.937702 | 0.935898 | 0.951248 | 0.994533 | 0.937449 |

[b256] **Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of the measured training tasks:

| config | rollout (measured) duration [s] | GU (measured) duration [s] | T buf_sample (perf_counter) [s] | T critic (perf_counter) [s] | T actor (perf_counter) [s] | T target (perf_counter) [s] | TOTAL duration (Σ measured tasks) [s] |
|---|---|---|---|---|---|---|---|
| b256 | 10.9702 ± 0.103092 ‖ 24.5015 ± 0.197026 | 200.053 ± 4.17276 ‖ 202.982 ± 3.77624 | 9.91330 ± 0.0491517 ‖ 12.2998 ± 0.0727960 | 129.830 ± 3.50683 ‖ 127.717 ± 3.25830 | 36.1536 ± 0.710964 ‖ 35.5712 ± 0.718193 | 11.5307 ± 0.0516577 ‖ 11.5380 ± 0.0849614 | 211.023 ± 4.18610 ‖ 227.483 ± 3.81449 |

**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the measured training tasks), then mean ± sd:

| config | P rollout [W] | P GU [W] | P TOTAL [W] | P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W] | P_rollout / P_GU |
|---|---|---|---|---|---|
| b256 | 138.263 ± 3.68402 ‖ 125.969 ± 1.67073 | 159.662 ± 1.21620 ‖ 159.805 ± 1.18342 | 158.550 ± 1.28336 ‖ 156.160 ± 1.06155 | 115.918 ± 0.633278 ‖ 116.441 ± 0.430259 | 0.865912 ± 0.0187521 ‖ 0.788289 ± 0.0106253 |

**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter times) / summed measured GU task duration:

| config | rollout [GFLOP/s] | GU [GFLOP/s] | TOTAL [GFLOP/s] | critic (FLOPs / T_perf) [GFLOP/s] | actor (FLOPs / T_perf) [GFLOP/s] | sub-timer coverage ΣT / D_GU [%] mean ± sd | coverage min–max [%] |
|---|---|---|---|---|---|---|---|
| b256 | 19.5476 ± 0.184363 ‖ 9.50429 ± 0.0756488 | 3144.71 ± 66.9947 ‖ 3308.62 ± 61.3348 | 2982.14 ± 60.3792 ‖ 2953.13 ± 49.4058 | 3792.18 ± 105.228 ‖ 4113.35 ± 104.217 | 3786.13 ± 75.9774 ‖ 4115.02 ± 82.7333 | 93.6860 ± 0.194991 ‖ 92.1846 ± 0.308519 | 93.3883–93.8756 ‖ 91.8998–92.6464 |

[b256] **Hardware composition of the measured `rollout` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of rollout energy [%] | GPU share of rollout energy [%] | RAM share of rollout energy [%] | rollout energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| b256 | 23.6946 ± 0.413507 ‖ 27.0431 ± 0.138298 | 61.8627 ± 0.777348 ‖ 57.0948 ± 0.306406 | 14.4427 ± 0.391543 ‖ 15.8621 ± 0.209216 | 500/500 ‖ 500/500 |

[b256] **Hardware composition of the measured `gradient_updates` task** (per-task CSV `cpu_energy`, `gpu_energy`, `ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):

| config | CPU share of GU energy [%] | GPU share of GU energy [%] | RAM share of GU energy [%] | GU energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks |
|---|---|---|---|---|
| b256 | 17.1861 ± 0.0684542 ‖ 17.3539 ± 0.138058 | 70.2890 ± 0.0906095 ‖ 70.1331 ± 0.165018 | 12.5249 ± 0.0947725 ‖ 12.5130 ± 0.0924099 | 0/500 ‖ 0/500 |

**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:

| config | rollout [%] | GU [%] | TOTAL [%] | P_idle (mean of head/tail, as recorded) [W] |
|---|---|---|---|---|
| b256 | 47.7305 ± 1.03817 ‖ 52.6361 ± 1.09063 | 41.3179 ± 0.572853 ‖ 41.4846 ± 0.478737 | 41.6076 ± 0.560833 ‖ 42.4527 ± 0.496750 | 65.9661 ± 0.797199 ‖ 66.2920 ± 0.673162 |

[b256] **Response to the sweep — paired relative change vs the same environment's baseline `base` [%]** (per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):
Energy:

| config | E rollout | E buf_sample | E critic | E actor | E target | E GU | E TOTAL |
|---|---|---|---|---|---|---|---|
| b256 | 7.51048 ± 3.03657 ‖ 0.529073 ± 3.16010 | 30.4604 ± 2.98087 ‖ 44.7364 ± 2.43245 | 19.3845 ± 1.38691 ‖ 16.7913 ± 2.46070 | 18.8703 ± 1.10452 ‖ 16.6938 ± 1.62720 | 16.1397 ± 1.94660 ‖ 15.4132 ± 1.94923 | 19.6119 ± 1.09637 ‖ 18.1802 ± 1.93372 | 19.0054 ± 1.07284 ‖ 16.4000 ± 1.73111 |

Duration:

| config | D rollout | D GU | D TOTAL | T buf_sample (perf_counter) | T critic (perf_counter) | T actor (perf_counter) | T target (perf_counter) |
|---|---|---|---|---|---|---|---|
| b256 | -1.27430 ± 0.893876 ‖ -0.986703 ± 1.31624 | 2.09036 ± 1.74794 ‖ 1.82910 ± 2.45729 | 1.90925 ± 1.64998 ‖ 1.51519 ± 2.20021 | 11.6395 ± 0.759211 ‖ 24.8333 ± 1.10120 | 2.19638 ± 2.31892 ‖ 0.762008 ± 3.18634 | 1.75161 ± 1.91161 ‖ 0.668797 ± 2.31462 | -0.604657 ± 0.830826 ‖ -0.458564 ± 0.729663 |

Mean power:

| config | P rollout | P GU | P TOTAL | P whole run |
|---|---|---|---|---|
| b256 | 8.90050 ± 3.03480 ‖ 1.52399 ± 2.54382 | 17.1837 ± 1.79900 ‖ 16.0816 ± 1.65821 | 16.7957 ± 1.81928 ‖ 14.6840 ± 1.67585 | 9.42602 ± 1.49972 ‖ 7.59803 ± 1.14427 |

Achieved throughput:

| config | rollout GFLOP/s | GU GFLOP/s | TOTAL GFLOP/s |
|---|---|---|---|
| b256 | 1.29735 ± 0.911789 ‖ 1.01069 ± 1.33140 | 150.817 ± 4.29142 ‖ 151.519 ± 6.09088 | 151.123 ± 4.06499 ‖ 152.138 ± 5.48748 |

Energy per FLOP (J/FLOP, per-seed ratio):

| config | rollout | critic | actor | GU | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| b256 | 7.51048 ± 3.03657 ‖ 0.529073 ± 3.16010 | -53.3654 ± 0.541762 ‖ -54.3784 ± 0.961211 | -53.5663 ± 0.431452 ‖ -54.4165 ± 0.635625 | -53.2766 ± 0.428268 ‖ -53.8359 ± 0.755358 | -53.4888 ± 0.419301 ‖ -54.5067 ± 0.676581 | 16.1397 ± 1.94660 ‖ 15.4132 ± 1.94923 |

FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; `target` = factor of the Polyak operation count):

| config | rollout | critic | actor | GU | TOTAL | target |
|---|---|---|---|---|---|---|
| b256 | 1.00000 ‖ 1.00000 | 2.56000 ‖ 2.56000 | 2.56000 ‖ 2.56000 | 2.56000 ‖ 2.56000 | 2.55864 ‖ 2.55862 | 1.00000 ‖ 1.00000 |

Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:

| config | rollout | buf_sample | critic | actor | target | GU |
|---|---|---|---|---|---|---|
| b256 | -0.484051 ± 0.121108 ‖ -1.37387 ± 0.301491 | 0.442033 ± 0.112018 ‖ 1.17505 ± 0.113908 | 0.208807 ± 0.313320 ‖ 0.203399 ± 0.534171 | -0.0209453 ± 0.0267636 ‖ 0.0438993 ± 0.0253475 | -0.145843 ± 0.120265 ‖ -0.0484755 ± 0.142179 | 0.484051 ± 0.121108 ‖ 1.37387 ± 0.301491 |

[b256] **Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):

| config | E_TOTAL Ant/HC | E_rollout Ant/HC | max \|Δshare\| [pp] | attained by | P_TOTAL Ant/HC | (P_roll/P_GU) Ant/HC | J/FLOP TOTAL Ant/HC |
|---|---|---|---|---|---|---|---|
| b256 | 1.06178 | 2.03490 | 4.15552 | rollout | 0.984926 | 0.910268 | 0.994533 |

Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):

| config | rollout | buf_sample | critic | actor | target | GU | TOTAL |
|---|---|---|---|---|---|---|---|
| b256 | 2.03490 (paired sd 0.0819396) | 1.26207 (paired sd 0.00608499) | 1.00065 (paired sd 0.0118753) | 1.00080 (paired sd 0.00837824) | 1.01783 (paired sd 0.0103978) | 1.01556 (paired sd 0.00945126) | 1.06178 (paired sd 0.00872256) |

Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, and the number of seeds (of 5) whose difference has the same sign as the mean:

| config | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|
| b256 | 4.15552 ± 0.256473 (5/5 same sign) | 0.952432 ± 0.0602579 (5/5 same sign) | -3.80639 ± 0.337327 (5/5 same sign) | -1.05804 ± 0.0578777 (5/5 same sign) | -0.243522 ± 0.0813771 (5/5 same sign) |

Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):

| config | rollout share | TOTAL energy |
|---|---|---|
| b256 | mean Δ 4.15552; 2/32 patterns; p = 0.0625000 | mean Δ 2066.81; 2/32 patterns; p = 0.0625000 |


**6.4 TD3 versus SAC side by side** at hidden (1024,1024), UTD 1, batch 256, both environments. TD3 = `b256` (TD3Config otherwise: lr 1e-3, policy delay 2, warmup 10000); SAC = `base` (SACConfig: lr 3e-4, α tuning, warmup 5000). SAC values come from the same recomputation method as Section 4.1 (`s4_data.Dataset(sac_spec())`, same definitions as `build_section_4_1.py`); cross-check against `contexts/section_4.1_data/sac_cross_seed_summary.csv` below. Columns: HC TD3, HC SAC, Ant TD3, Ant SAC.

- **PASS** — SAC baseline energies and J/FLOP recomputed here equal `sac_cross_seed_summary.csv` of Section 4.1 (max relative deviation energy 1.82e-16, J/FLOP 1.88e-16)

Energy [J] (measured: rollout, GU; allocated: the four sub-segments):


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 1516.82 ± 44.8592 | 2495.70 ± 33.3514 | 3086.58 ± 58.2258 | 4332.11 ± 50.1937 |
| buf_sample | 1689.43 ± 12.8701 | 1550.17 ± 20.9057 | 2132.18 ± 13.2997 | 1922.92 ± 50.6812 |
| critic | 22121.8 ± 403.776 | 24575.1 ± 234.915 | 22136.2 ± 344.166 | 25133.1 ± 295.659 |
| actor | 6160.56 ± 66.3990 | 29743.4 ± 127.124 | 6165.50 ± 60.8783 | 30436.9 ± 252.224 |
| target | 1965.08 ± 16.8894 | 2546.63 ± 28.8006 | 2000.11 ± 16.1819 | 2557.54 ± 32.2932 |
| GU | 31936.9 ± 449.997 | 58415.3 ± 247.829 | 32434.0 ± 378.529 | 60050.5 ± 545.039 |
| TOTAL | 33453.7 ± 435.384 | 60911.0 ± 254.111 | 35520.5 ± 357.746 | 64382.6 ± 588.270 |


Share of TOTAL energy [%]:


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 4.53511 ± 0.161573 | 4.09730 ± 0.0530434 | 8.69063 ± 0.205170 | 6.72865 ± 0.0383970 |
| buf_sample | 5.05097 ± 0.0940456 | 2.54494 ± 0.0288351 | 6.00340 ± 0.0944562 | 2.98653 ± 0.0643192 |
| critic | 66.1231 ± 0.376812 | 40.3456 ± 0.298375 | 62.3167 ± 0.362815 | 39.0369 ± 0.242706 |
| actor | 18.4156 ± 0.0549150 | 48.8313 ± 0.210050 | 17.3576 ± 0.0187040 | 47.2755 ± 0.142438 |
| target | 5.87517 ± 0.116413 | 4.18089 ± 0.0427508 | 5.63164 ± 0.0987609 | 3.97241 ± 0.0358136 |
| GU | 95.4649 ± 0.161573 | 95.9027 ± 0.0530434 | 91.3094 ± 0.205170 | 93.2714 ± 0.0383970 |


Share of GU energy [%] (= wall-clock time share of each sub-phase):


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| buf_sample | 5.29104 ± 0.105886 | 2.65368 ± 0.0310124 | 6.57492 ± 0.112010 | 3.20200 ± 0.0702272 |
| critic | 69.2640 ± 0.296835 | 42.0692 ± 0.290066 | 68.2475 ± 0.266010 | 41.8530 ± 0.244460 |
| actor | 19.2905 ± 0.0671484 | 50.9176 ± 0.244273 | 19.0097 ± 0.0444070 | 50.6860 ± 0.169870 |
| target | 6.15442 ± 0.130688 | 4.35952 ± 0.0456455 | 6.16782 ± 0.118863 | 4.25899 ± 0.0388861 |


J/FLOP (ratio of means ± sd of per-seed ratios; GU derived; `target_update` in nJ per Polyak op):


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 7.07388e-09 ± 2.09207e-10 | 1.15727e-08 ± 1.54652e-10 | 1.32552e-08 ± 2.50049e-10 | 1.84741e-08 ± 2.14049e-10 |
| buf_sample | no FLOPs | no FLOPs | no FLOPs | no FLOPs |
| critic | 4.49590e-11 ± 8.20608e-13 | 4.99130e-11 ± 4.77121e-13 | 4.21582e-11 ± 6.55462e-13 | 4.78277e-11 ± 5.62631e-13 |
| actor | 4.50204e-11 ± 4.85234e-13 | 7.73536e-11 ± 3.30610e-13 | 4.21345e-11 ± 4.16038e-13 | 7.36537e-11 ± 6.10351e-13 |
| target | 6.09406 ± 0.0523770 | 5.92128 ± 0.0669656 | 5.71287 ± 0.0462199 | 5.47718 ± 0.0691586 |
| GU | 5.07835e-11 ± 7.15548e-13 | 6.66179e-11 ± 2.82629e-13 | 4.83077e-11 ± 5.63787e-13 | 6.39694e-11 ± 5.80609e-13 |
| TOTAL | 5.31773e-11 ± 6.92076e-13 | 6.94469e-11 ± 2.89722e-13 | 5.28866e-11 ± 5.32649e-13 | 6.85671e-11 ± 6.26504e-13 |


Per-call FLOPs (`flops_per_call.json`; critic per update() call, actor/target per executed update; target in elementwise ops):


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 2.14426e+06 | 2.15654e+06 | 2.32858e+06 | 2.34496e+06 |
| buf_sample | 0 | 0 | 0 | 0 |
| critic | 4.92044e+09 | 4.92359e+09 | 5.25074e+09 | 5.25494e+09 |
| actor | 2.73678e+09 | 3.84513e+09 | 2.92658e+09 | 4.13244e+09 |
| target | 6.44917e+06 | 4.30080e+06 | 7.00213e+06 | 4.66944e+06 |


Call counts per run:


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 100000 | 100000 | 100000 | 100000 |
| buf_sample | 100000 | 100000 | 100000 | 100000 |
| critic | 100000 | 100000 | 100000 | 100000 |
| actor | 50000 | 100000 | 50000 | 100000 |
| target | 50000 | 100000 | 50000 | 100000 |


Total FLOPs per run (`target`: ops):


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 2.14426e+11 | 2.15654e+11 | 2.32858e+11 | 2.34496e+11 |
| critic | 4.92044e+14 | 4.92359e+14 | 5.25074e+14 | 5.25494e+14 |
| actor | 1.36839e+14 | 3.84513e+14 | 1.46329e+14 | 4.13244e+14 |
| target | 3.22458e+11 | 4.30080e+11 | 3.50107e+11 | 4.66944e+11 |
| GU | 6.28883e+14 | 8.76872e+14 | 6.71403e+14 | 9.38738e+14 |
| TOTAL | 6.29098e+14 | 8.77087e+14 | 6.71636e+14 | 9.38972e+14 |


Duration [s] (measured: rollout, GU; sub-segments = perf_counter time):


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 10.9702 ± 0.103092 | 20.4774 ± 0.265263 | 24.5015 ± 0.197026 | 35.9898 ± 0.288031 |
| buf_sample | 9.91330 ± 0.0491517 | 10.4499 ± 0.142907 | 12.2998 ± 0.0727960 | 12.7700 ± 0.266712 |
| critic | 129.830 ± 3.50683 | 165.664 ± 1.83060 | 127.717 ± 3.25830 | 166.925 ± 2.03291 |
| actor | 36.1536 ± 0.710964 | 200.502 ± 0.822675 | 35.5712 ± 0.718193 | 202.146 ± 0.956358 |
| target | 11.5307 ± 0.0516577 | 17.1670 ± 0.192437 | 11.5380 ± 0.0849614 | 16.9859 ± 0.191168 |
| GU | 200.053 ± 4.17276 | 404.865 ± 2.23475 | 202.982 ± 3.77624 | 413.120 ± 2.85184 |
| TOTAL | 211.023 ± 4.18610 | 425.343 ± 2.26565 | 227.483 ± 3.81449 | 449.109 ± 2.95475 |


Mean power [W] (energy / duration; allocated sub-segments: E_k/T_k):


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 138.263 ± 3.68402 | 121.876 ± 0.488145 | 125.969 ± 1.67073 | 120.369 ± 0.712466 |
| buf_sample | 170.425 ± 1.63471 | 148.345 ± 0.229807 | 173.357 ± 1.82418 | 150.568 ± 0.936315 |
| critic | 170.425 ± 1.63471 | 148.345 ± 0.229807 | 173.357 ± 1.82418 | 150.568 ± 0.936315 |
| actor | 170.425 ± 1.63471 | 148.345 ± 0.229807 | 173.357 ± 1.82418 | 150.568 ± 0.936315 |
| target | 170.425 ± 1.63471 | 148.345 ± 0.229807 | 173.357 ± 1.82418 | 150.568 ± 0.936315 |
| GU | 159.662 ± 1.21620 | 144.284 ± 0.219340 | 159.805 ± 1.18342 | 145.358 ± 0.727140 |
| TOTAL | 158.550 ± 1.28336 | 143.205 ± 0.194907 | 156.160 ± 1.06155 | 143.355 ± 0.710082 |


Hardware composition of the measured tasks — CPU / GPU / RAM share [%]:


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| rollout | 23.6946 / 61.8627 / 14.4427 | 25.2380 / 58.3684 / 16.3936 | 27.0431 / 57.0948 / 15.8621 | 26.4735 / 56.9261 / 16.6004 |
| GU | 17.1861 / 70.2890 / 12.5249 | 17.5806 / 68.5591 / 13.8603 | 17.3539 / 70.1331 / 12.5130 | 17.8235 / 68.4185 / 13.7580 |


Sub-timer coverage ΣT / D_GU [%] and idle-floor fraction of GU / TOTAL [%]:


| segment | HC TD3 | HC SAC | Ant TD3 | Ant SAC |
|---|---|---|---|---|
| cov | 93.6860 ± 0.194991 | 97.2629 ± 0.0876070 | 92.1846 ± 0.308519 | 96.5403 ± 0.146463 |
| idleGU | 41.3179 ± 0.572853 | 47.3193 ± 0.274503 | 41.4846 ± 0.478737 | 47.7078 ± 0.885576 |
| idleTOT | 41.6076 ± 0.560833 | 47.6758 ± 0.285177 | 42.4527 ± 0.496750 | 48.3743 ± 0.897579 |


**6.5 Actor update** — for every TD3 configuration: energy, FLOPs, J/FLOP and share of GU of `actor_update` and `critic_update` (allocated values), with the SAC counterpart of the same swept factor. **Matched settings exist exactly** for `b256`↔SAC `base`, `b512`↔SAC `b512`, `b1024`↔SAC `b1024` (hidden (1024,1024), UTD 1, same batch). For the TD3 configurations `base`, `utd2`, `utd4`, `w256`, `w512` the batch size is 100 whereas the SAC counterpart (same UTD / width) has batch 256, so those SAC values are **not** batch-matched (marked ≠batch). Cell `HC ‖ Ant`; `SAC counterpart` is the SAC configuration named in the second column:


| config | matching | E actor [J] | F actor | J/FLOP actor | actor share of GU [%] | E critic [J] | F critic | J/FLOP critic | critic share of GU [%] |
|---|---|---|---|---|---|---|---|---|---|
| TD3 base |  | 5182.75 ± 53.5812 ‖ 5284.10 ± 73.8102 | 5.34528e+13 ‖ 5.71597e+13 | 9.69593e-11 ± 1.00240e-12 ‖ 9.24445e-11 ± 1.29130e-12 | 19.4108 ± 0.0479216 ‖ 19.2513 ± 0.0356577 | 18530.1 ± 279.872 ‖ 18957.5 ± 337.485 | 1.92205e+14 ‖ 2.05107e+14 | 9.64080e-11 ± 1.45611e-12 ‖ 9.24275e-11 ± 1.64541e-12 | 69.3974 ± 0.311788 ‖ 69.0640 ± 0.268809 |
| SAC base | ≠batch (100 vs 256) | 29743.4 ± 127.124 ‖ 30436.9 ± 252.224 | 3.84513e+14 ‖ 4.13244e+14 | 7.73536e-11 ± 3.30610e-13 ‖ 7.36537e-11 ± 6.10351e-13 | 50.9176 ± 0.244273 ‖ 50.6860 ± 0.169870 | 24575.1 ± 234.915 ‖ 25133.1 ± 295.659 | 4.92359e+14 ‖ 5.25494e+14 | 4.99130e-11 ± 4.77121e-13 ‖ 4.78277e-11 ± 5.62631e-13 | 42.0692 ± 0.290066 ‖ 41.8530 ± 0.244460 |
| TD3 utd2 |  | 10466.8 ± 59.1597 ‖ 10505.6 ± 62.8497 | 1.06906e+14 ‖ 1.14319e+14 | 9.79070e-11 ± 5.53383e-13 ‖ 9.18974e-11 ± 5.49773e-13 | 19.4086 ± 0.0415109 ‖ 19.2725 ± 0.0255887 | 37419.5 ± 352.301 ‖ 37577.8 ± 305.781 | 3.84410e+14 ‖ 4.10214e+14 | 9.73428e-11 ± 9.16472e-13 ‖ 9.16052e-11 ± 7.45418e-13 | 69.3854 ± 0.131601 ‖ 68.9351 ± 0.128783 |
| SAC utd2 | ≠batch | 58953.3 ± 230.665 ‖ 60663.7 ± 240.320 | 7.69026e+14 ‖ 8.26488e+14 | 7.66598e-11 ± 2.99944e-13 ‖ 7.33995e-11 ± 2.90773e-13 | 51.0374 ± 0.148283 ‖ 51.0325 ± 0.146743 | 48486.5 ± 414.609 ‖ 49256.6 ± 437.929 | 9.84718e+14 ‖ 1.05099e+15 | 4.92390e-11 ± 4.21044e-13 ‖ 4.68670e-11 ± 4.16683e-13 | 41.9752 ± 0.166738 ‖ 41.4355 ± 0.179234 |
| TD3 utd4 |  | 20567.4 ± 158.075 ‖ 20961.7 ± 266.483 | 2.13811e+14 ‖ 2.28639e+14 | 9.61941e-11 ± 7.39321e-13 ‖ 9.16802e-11 ± 1.16552e-12 | 19.4101 ± 0.0536939 ‖ 19.2517 ± 0.0470502 | 73363.4 ± 803.512 ‖ 75164.3 ± 1338.01 | 7.68819e+14 ‖ 8.20429e+14 | 9.54235e-11 ± 1.04513e-12 ‖ 9.16158e-11 ± 1.63087e-12 | 69.2334 ± 0.106782 ‖ 69.0286 ± 0.228985 |
| SAC utd4 | ≠batch | 118342 ± 561.913 ‖ 120878 ± 478.788 | 1.53805e+15 ‖ 1.65298e+15 | 7.69426e-11 ± 3.65341e-13 ‖ 7.31277e-11 ± 2.89652e-13 | 50.9202 ± 0.0909592 ‖ 50.9964 ± 0.0431630 | 97913.7 ± 628.512 ‖ 98307.1 ± 459.369 | 1.96944e+15 ‖ 2.10198e+15 | 4.97166e-11 ± 3.19133e-13 ‖ 4.67689e-11 ± 2.18541e-13 | 42.1302 ± 0.121756 ‖ 41.4739 ± 0.0505889 |
| TD3 w256 |  | 4198.84 ± 112.792 ‖ 4261.64 ± 141.499 | 3.53280e+12 ‖ 4.45952e+12 | 1.18853e-09 ± 3.19272e-11 ‖ 9.55628e-10 ± 3.17297e-11 | 19.2639 ± 0.0238581 ‖ 19.1424 ± 0.0368749 | 15153.1 ± 457.182 ‖ 15463.0 ± 584.149 | 1.26618e+13 ‖ 1.58874e+13 | 1.19676e-09 ± 3.61073e-11 ‖ 9.73287e-10 ± 3.67682e-11 | 69.5161 ± 0.197164 ‖ 69.4484 ± 0.332721 |
| SAC w256 | ≠batch | 22990.5 ± 176.796 ‖ 23127.4 ± 286.451 | 2.56639e+13 ‖ 3.28466e+13 | 8.95829e-10 ± 6.88888e-12 ‖ 7.04103e-10 ± 8.72085e-12 | 50.9445 ± 0.177325 ‖ 50.4057 ± 0.230517 | 18973.9 ± 277.584 ‖ 19303.8 ± 405.723 | 3.24927e+13 ‖ 4.07765e+13 | 5.83941e-10 ± 8.54295e-12 ‖ 4.73405e-10 ± 9.94992e-12 | 42.0417 ± 0.210031 ‖ 42.0677 ± 0.272935 |
| TD3 w512 |  | 4654.20 ± 66.0151 ‖ 4704.94 ± 41.2614 | 1.36192e+13 ‖ 1.54726e+13 | 3.41738e-10 ± 4.84720e-12 ‖ 3.04081e-10 ± 2.66674e-12 | 19.3386 ± 0.0686606 ‖ 19.1454 ± 0.0607421 | 16794.9 ± 362.842 ‖ 17133.3 ± 268.541 | 4.89165e+13 ‖ 5.53677e+13 | 3.43339e-10 ± 7.41759e-12 ‖ 3.09447e-10 ± 4.85014e-12 | 69.7776 ± 0.383092 ‖ 69.7150 ± 0.368845 |
| SAC w512 | ≠batch | 24101.1 ± 85.6765 ‖ 24347.6 ± 144.137 | 9.83040e+13 ‖ 1.12669e+14 | 2.45169e-10 ± 8.71547e-13 ‖ 2.16097e-10 ± 1.27929e-12 | 51.0356 ± 0.396922 ‖ 50.6782 ± 0.327084 | 19809.2 ± 447.308 ‖ 20056.2 ± 384.348 | 1.25383e+14 ‖ 1.41951e+14 | 1.57989e-10 ± 3.56752e-12 ‖ 1.41290e-10 ± 2.70761e-12 | 41.9401 ± 0.493904 ‖ 41.7413 ± 0.401730 |
| TD3 b256 |  | 6160.56 ± 66.3990 ‖ 6165.50 ± 60.8783 | 1.36839e+14 ‖ 1.46329e+14 | 4.50204e-11 ± 4.85234e-13 ‖ 4.21345e-11 ± 4.16038e-13 | 19.2905 ± 0.0671484 ‖ 19.0097 ± 0.0444070 | 22121.8 ± 403.776 ‖ 22136.2 ± 344.166 | 4.92044e+14 ‖ 5.25074e+14 | 4.49590e-11 ± 8.20608e-13 ‖ 4.21582e-11 ± 6.55462e-13 | 69.2640 ± 0.296835 ‖ 68.2475 ± 0.266010 |
| SAC base | matched | 29743.4 ± 127.124 ‖ 30436.9 ± 252.224 | 3.84513e+14 ‖ 4.13244e+14 | 7.73536e-11 ± 3.30610e-13 ‖ 7.36537e-11 ± 6.10351e-13 | 50.9176 ± 0.244273 ‖ 50.6860 ± 0.169870 | 24575.1 ± 234.915 ‖ 25133.1 ± 295.659 | 4.92359e+14 ‖ 5.25494e+14 | 4.99130e-11 ± 4.77121e-13 ‖ 4.78277e-11 ± 5.62631e-13 | 42.0692 ± 0.290066 ‖ 41.8530 ± 0.244460 |
| TD3 b512 |  | 7582.97 ± 122.336 ‖ 7620.33 ± 97.6287 | 2.73678e+14 ‖ 2.92658e+14 | 2.77076e-11 ± 4.47006e-13 ‖ 2.60384e-11 ± 3.33594e-13 | 18.9579 ± 0.0822039 ‖ 18.5597 ± 0.0386827 | 27596.4 ± 829.827 ‖ 27504.1 ± 598.828 | 9.84089e+14 ‖ 1.05015e+15 | 2.80426e-11 ± 8.43244e-13 ‖ 2.61907e-11 ± 5.70231e-13 | 68.9775 ± 0.687441 ‖ 66.9805 ± 0.485411 |
| SAC b512 | matched | 35466.7 ± 106.481 ‖ 36017.7 ± 39.8942 | 7.69026e+14 ‖ 8.26488e+14 | 4.61190e-11 ± 1.38462e-13 ‖ 4.35793e-11 ± 4.82696e-14 | 50.3983 ± 0.484513 ‖ 49.9342 ± 0.210098 | 29859.7 ± 660.459 ‖ 30236.2 ± 363.913 | 9.84718e+14 ‖ 1.05099e+15 | 3.03231e-11 ± 6.70709e-13 ‖ 2.87694e-11 ± 3.46258e-13 | 42.4245 ± 0.617453 ‖ 41.9169 ± 0.287622 |
| TD3 b1024 |  | 8610.15 ± 119.128 ‖ 8843.11 ± 83.0931 | 5.47357e+14 ‖ 5.85315e+14 | 1.57304e-11 ± 2.17642e-13 ‖ 1.51083e-11 ± 1.41963e-13 | 18.5910 ± 0.0773751 ‖ 17.7428 ± 0.126518 | 31608.6 ± 899.991 ‖ 32144.2 ± 804.987 | 1.96818e+15 ‖ 2.10030e+15 | 1.60599e-11 ± 4.57271e-13 ‖ 1.53046e-11 ± 3.83273e-13 | 68.2351 ± 0.742732 ‖ 64.4819 ± 0.713606 |
| SAC b1024 | matched | 40035.2 ± 167.770 ‖ 41446.8 ± 148.180 | 1.53805e+15 ‖ 1.65298e+15 | 2.60298e-11 ± 1.09080e-13 ‖ 2.50741e-11 ± 8.96444e-14 | 50.2855 ± 0.265318 ‖ 49.0407 ± 0.250921 | 33302.3 ± 502.557 ‖ 34806.0 ± 446.046 | 1.96944e+15 ‖ 2.10198e+15 | 1.69096e-11 ± 2.55178e-13 ‖ 1.65587e-11 ± 2.12203e-13 | 41.8260 ± 0.354070 ‖ 41.1814 ± 0.346414 |

Per-call constants for the same pairs are in Section 2.4 (TD3) and Section 4.1 Part 3 (SAC); the actor-update FLOPs per executed update differ between TD3 and SAC by construction of the two actor losses (TD3: actor fwd + Q1 only; SAC: actor fwd + Q1 + Q2 + log-prob/α), and the number of executed updates differs (TD3 ceil(n_upd/2), SAC n_upd).

**6.6 Open points from the instructions (reported, not resolved):**

- *`rollout` task durations versus the 1 s polling interval:* across all 80 TD3 runs 8000 of 8000 `rollout` tasks are shorter than 1 s (min/median/max 0.100293/0.217525/0.349368 s); 0 of 8000 `gradient_updates` tasks are shorter than 1 s (min/median/max 1.80863/2.14756/9.35523 s). Per configuration: Section 5.2a. (The same situation was reported for SAC in Section 4.1.)
- *Host-side work counted as FLOPs:* none. `FlopCounterMode` counts matmul-type operators only (`aten.mm/addmm/bmm`…). **Not counted (0 FLOPs):** replay-buffer sampling and the host→device copies (`buffer_sample`: 0 by definition, `ZERO_FLOP_SEGMENTS`), exploration noise / `np.clip`, `env.step()` (MuJoCo), `buffer.add`, target-smoothing noise and clamps, the Q-value `min`, MSE loss, Adam updates, ReLU/tanh, and the Polyak averaging (counted separately as elementwise operations). **Counted:** `rollout` = one actor forward at batch 1 per env step (env step and noise excluded); `critic_update` = target-actor fwd + 2 target-Q fwd + 2 × (Q fwd + bwd) at the configured batch; `actor_update` = actor fwd + Q1 fwd + Q1 input-grad bwd + actor bwd; `target_update` = Polyak operation count; `gradient_updates` (derived) = critic + actor.
- *Segments whose timers or FLOPs are produced differently from SAC:* (i) `actor_update` and `target_update` are timed and counted only on executed delayed iterations (every second `update()` call), `target_update` covers three networks including the target actor; (ii) the critic FLOPs include a target-**actor** forward (deterministic, no log-prob) instead of SAC's stochastic next-action sampling; (iii) the actor loss uses Q1 only (SAC: min of both critics); (iv) no α step. (v) warmup is 10000 steps (SAC 5000), i.e. the replay buffer holds 10000 more transitions throughout training.

## 7. Discrepancies, NOT AVAILABLE items, questions

**(a) Where data or code contradict CLAUDE.md, README.md or `thesis-layout.md`:**

- CLAUDE.md / README.md state that zero GPU-clock-lock / CPU-governor permission failures and zero RAPL / geolocation fallbacks were found 'by grepping every run.log'. That grep cannot detect those failures (the `gpu_control`, `thermal_gate` and CodeCarbon loggers do not write to `run.log`). For TD3 the claim is supported by other evidence: 80 of 80 runs have a sweep-log block with captured stderr (`2>&1 | tee -a`) containing none of the failure strings, plus data-side checks (varying RAPL `cpu_power`, non-fallback geolocation, `thermal_gate.reason = reached_reference`) for all runs (Section 1.3). The clock lock is verified as requested, not as applied.
- The one-row `emissions.csv` written by `tracker.stop()` has `duration` = 90.0058 s (median over the TD3 runs; it equals the duration of the last task, the 90 s idle tail) while its `energy_consumed` is the cumulative whole-run energy (equal to the sum of the per-task rows within 1.2e-04 relative). `emissions.csv` therefore cannot be used for a whole-run mean power; the 'P whole run' columns of this file use Σ energy / Σ duration of all rows of the per-task CSV (idle head, warmup, [pretrain], training tasks, idle tail; the unmeasured 30 s settle is not included).
- README.md ('Hyperparameter sensitivity sweeps') and CLAUDE.md say the original SAC-on-Ant/TD3 UTD sweep's top-level status/log bookkeeping was overwritten when `scripts/run_utd_sweep.sh` was repointed to MBPO. For TD3: `results/_utd_sweep_status.json` indeed holds only the MBPO entries (0 TD3 entries, Section 1.4), but `results/_utd_sweep.log` (opened with `tee -a`) still contains the complete launch blocks (stdout+stderr) of all 20 TD3 UTD runs (`utd2`/`utd4`, both envs), so every TD3 UTD run has a log block; only the status file is lost.

**(b) Quantities that could not be obtained (`NOT AVAILABLE`):**

- NOT AVAILABLE: GPU theoretical FP32 peak (SM count / lanes per SM are not stored in the repo and no CUDA is available on the Windows checkout) — achieved throughput is reported in GFLOP/s without a peak-utilisation fraction.
- NOT AVAILABLE: Whether FP32 matmuls ran as TF32 on the RTX 5090 under torch 2.13.0+cu130: the code sets no TF32 / matmul-precision flag and no run logs the effective setting (see Section 4.1 Part 2.3).
- NOT AVAILABLE: Positive confirmation that the GPU clock lock (2000/2000 MHz), persistence mode and the CPU `performance` governor were *applied*: `metadata.json` records only the request; failure messages would appear in the sweep-log stderr and none do.
- NOT AVAILABLE: gymnasium / mujoco versions on the Linux experiment box: not recorded in `metadata.json`; environment dimensions here were read from the local venv and agree with `flops_per_call.json`.
- NOT AVAILABLE: Simulator (MuJoCo) share of the `rollout` task duration: no per-step simulator timing is logged, so the split of `rollout` into policy/model compute and `env.step()` is not measurable from the data.

**(c) Questions for the author that the repository cannot answer:**

- Idle-baseline drift: the head idle window integrates 3.07424 s more than its recorded 90 s (RAM energy / RAM power), the tail window does not; this file reports as-recorded idle powers (definition of Section 4.1) and the window-corrected drift in Section 5.2e. Which of the two the thesis text should quote is not decided in the repository.
- TD3 baseline = batch 100 (the TD3 paper default) is the thesis baseline per the layout instructions; the batch-matched comparison to SAC/MBPO is `b256`. Whether the SAC-vs-TD3 comparison of the thesis should use `base` or `b256` for TD3 is a decision for the author (both are reported in full).

**Index of produced files** (`contexts/Section_4.2_data/`): `td3_segments_long.csv`, `td3_cross_seed_summary.csv`, `td3_run_inventory.csv`, `td3_per_epoch.csv`, `td3_idle_floor.csv`, `td3_ols_fits.csv`, `td3_delta_pairs.csv`, `td3_mad_flags.csv`, `td3_learning_performance.csv`; scripts `build_section_4_2.py`, `s4_data.py`, `s4_blocks.py`, `s4_audit.py`.
