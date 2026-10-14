# Step 1 summary

All outputs are in `contexts/step1/`, scripts in `contexts/step1/scripts/`. Nothing under `results/`, `configs/` or existing scripts was modified, nothing was committed. Canonical list: 280 runs = 56 configurations × 5 seeds (SAC 7/7, TD3 8/8, MBPO 7 HalfCheetah + 9 Ant, TD-MPC2 5/5 per seed; `04_canonical_run_list.csv`).

## Task 1: idle drift (`01_idle_drift.md`)

- "Window-corrected" **is defined in the repo** (`idle_power_analysis.py` L96-108, `README.md` L646-649): power = energy ÷ (`ram_energy`/`ram_power`) instead of ÷ recorded duration; the head window integrates 3.08 s (median 93.080 s) more than its recorded 90.003 s, the tail does not (90.002 s).
- Total-power drift (tail − head)/head over 280 runs: **(a) as recorded** median +6.34 %, median |drift| 7.04 %, mean +6.69 %, p5/p95 −6.13/+16.36 %, max 23.19 %, tail > head in 256/280 (91.4 %). **(b) window-corrected** median +9.98 %, median |drift| 10.02 %, mean +10.34 %, p5/p95 −2.96/+20.32 %, max 27.39 %, tail > head in 261/280 (93.2 %). In W: (a) median +3.99 W, (b) +6.14 W.
- Largest drift is in SAC runs (e.g. `results/sac/Ant-v5/seed_43611/20260912_223103`: 63.57 → 78.32 W as recorded, 61.48 → 78.32 W corrected); lowest in MBPO (median 4.1 % / 7.7 %); TD-MPC2 and TD3 highest medians (10 % / 13.8 % and 9-9.6 % / 12.8-13.3 %).
- Surprise: head idle power is not a constant: daily medians 60.8-65.6 W across 11 dates and r = 0.88 with pre-head GPU temperature; GPU drift dominates (as-recorded median +20 %, corrected +24 %), CPU is slightly negative.

## Task 2: sub-interval tasks (`02_subinterval_tasks.md`)

- Rollout tasks shorter than `measure_power_secs` (1.0 s, all 280 runs): **SAC 7000/7000, TD3 8000/8000, MBPO 8000/8000, TD-MPC2 0/5000** (TD-MPC2 rollout 6.6-16.7 s). The 8000 = 16 MBPO configs × 5 seeds × 100 epochs; 720/1000 baseline synthetic-rollout tasks have zero GPU energy (confirmed; 94 % on HalfCheetah, 57 % on Ant over all configs).
- **Mismatch to flag:** the quoted medians 0.3409 s (SAC rollout) and 0.2175 s (TD3 rollout) reproduce only when **all configurations and both envs are pooled** (baseline-only pooled: 0.3436 s and 0.2165 s; per env baseline SAC 0.203/0.357 s, TD3 0.110/0.245 s for HalfCheetah/Ant).
- CodeCarbon 3.3.0 (source read): **task energy is a counter difference between `start_task` and `stop_task`** (NVML energy counter for GPU, RAPL for CPU, constant 20 W × elapsed time for RAM); the periodic 1 s scheduler is **stopped for the duration of a task** (`emissions_tracker.py` L754-755), `stop_task` takes one fresh measurement. So there is no "no sample in the window" case; the 1 s interval does not bound resolution.
- Established from data, mechanism **inferred**: GPU energy of ms-scale tasks is quantised (HalfCheetah synthetic rollout, 6 ms: 3283/3500 tasks 0 J, the rest 5.9-6.9 J ≈ one lump of an ≈100 ms-update counter; implausible per-task GPU power up to 1215 W); 0.1-0.4 s rollouts are likely ± one lump (≈ ±25-40 % per task). Whole-run energy equals Σ task energies to 0.0125 % (identity of the accounting, not a validation). 2C scripts (`sync_timing_run.py`, `nvml_poll.py`) written, **not run**.

## Task 3: machine fact sheet (`03_machine_fact_sheet.md`)

- **Not run** — session was on the Windows dev checkout. Script `scripts/machine_fact_sheet.sh` covers all 10 items. From the repo/run data: GPU RTX 5090, CPU Ultra 9 285K (24 threads), 60.8 GB RAM, kernel 7.0.0-31-generic, Python 3.14.4, torch 2.13.0+cu130, codecarbon 3.3.0 (all 280 runs); `gym.make(env_id)` with no kwargs.
- **TF32:** no code path sets it (no hits for `allow_tf32`, `set_float32_matmul_precision`, `autocast`, `bfloat16`, `float16`, `half(`); the runs used PyTorch's defaults (documented default: full FP32), which are not recorded in the run data, so "FP32 vs TF32" is **not verified**. Peak FP32 value: **not computed** (needs SM count, compute capability, boost clock).
- Git: no LICENSE, no tags, remote `github.com/SamBro2901/rl-dissections`; `2b4882f` (2026-10-08 08:57 +02:00) and `050b3c3` (2026-10-08 16:48 +02:00) exist.

## Task 4: canonical data quality (`04_canonical_data_quality.md`)

- **0 violations on the 280 canonical runs** for `check_totals.py` checks 1-3 (max rel. error 0 / exact / 4.8e-13), 0 missing files, 0 duplicate (config, seed); `train_steps` 100 000, `steps_per_epoch` 1000, 100 epochs, warmup 5000 (TD3 10 000) in all; d = 1 in all 8 baselines.
- **The `run.log` scans for RAPL/geolocation/clock-lock failures cannot detect anything** (`gpu_control`, `thermal_gate` and CodeCarbon loggers have no handler writing to `run.log`; verified in code); the valid scan is the sweep logs (stderr captured): 0 hits for 270 of 280 runs, **none for the 10 SAC baseline runs**. Evidence level: thermal gate = applied (280); clock lock / persistence / governor = requested only (280), no failure warning for 270, **no positive record of success for any run**; RAPL = indirect only.
- Execution order: **configuration-major, fixed seed order (331, 958, 14577, 43611, 85062), not randomised** — 56/56 blocks; campaign 2026-09-07 10:13:36 UTC → 2026-09-19 16:10:36 UTC in **10 sessions** (gap threshold 2 h; same count for 0.5-2 h).
- Seeds: first appearance of `14577` in git is `92dcee0` (2026-09-07 14:13 +02:00, results only), **after** the first canonical run (12:13:35 +02:00); first appearance in code/scripts `d1cebb4` (2026-09-08 18:58). UTD: 60 runs complete, 20 TD3 launch blocks in the (appended, not overwritten) log. TD-MPC2 false start: 5 failed launches + a 6th with no closing line, none created a run directory; no clock-lock evidence for the 6th is obtainable from the log.
- Code: 12 distinct commits; between the MBPO HalfCheetah baseline (`0bbbbb7`) and Ant `L_max=1` (`788ffcd`) only new algorithms/configs/dispatch were added, `sac.py`/`mbpo.py`/`dynamics_model.py` unchanged. The 8 earliest SAC runs ran with uncommitted edits relative to their recorded commit (4 lack `run_start_temp_c`).

## NOT DETERMINABLE (and what resolves each)

| item | reason | needed |
|---|---|---|
| All of Task 3 (OS/kernel, GPU/driver/VBIOS/clocks, board, RAM type/speed, governor, RAPL access, package versions, TF32 runtime flags, SM count, peak FP32, env dimensions, research-PC git state) | session not on the research PC | run `scripts/machine_fact_sheet.sh` there and paste the output; `sudo dmidecode -t memory -t baseboard` by hand |
| Whether clock lock / persistence mode / governor were actually applied in any run | INFO messages never logged; no readback in metadata; no sweep-log for SAC baselines | nothing in the repo; only an independent record (e.g. old terminal output, nvidia-smi logs) or a decision to word it as "requested and no failure reported" |
| Whether RAPL (not TDP fallback) was used in the 10 SAC baseline runs | no stderr capture for them | same as above (indirect evidence only: CPU idle 14.5-18.3 W) |
| NVML counter update interval and the per-task error of the GPU energy for short tasks | inferred from data only | run `scripts/nvml_poll.py` (2C) on the research PC |
| Sub-phase timing without CUDA synchronisation (ratio of Σ sub-phase times to measured `gradient_updates`) | needs new runs | `scripts/sync_timing_run.py` (2C, awaiting your go-ahead) |
| Cause of the 3.08 s head-window excess | attributed by reading source order (lazy geolocation in first `start_task`), not measured | a timing print in a throw-away run (optional) |
| When the canonical seeds were chosen | repo only shows first appearance after run 1 | your own records |
| Whether the 6th launch of the TD-MPC2 false start touched the GPU clock lock | `gpu_control` INFO not logged; process killed before the run header | not recoverable |
| Package versions on the Linux box vs. recorded | `rl-exp` venv not accessible | script §5 on the research PC |
