# rl-dissections — project context for Claude

Thesis project: a harness that measures **per-segment energy consumption**
(via CodeCarbon/NVML/RAPL) of training four different RL algorithms, so their
energy cost can be "dissected" into rollout vs. gradient-update sub-phases and
ultimately compared on an **energy-per-FLOP** basis. This file exists so a
fresh Claude session has full context without re-deriving it from source —
`README.md` has the deepest technical/methodological detail and should be
read alongside this file for anything protocol-related.

## Environment / where things actually run

- This checkout (`C:\Users\saman\Desktop\Thesis\Project Repository\rl-dissections`)
  is the **Windows dev machine** — has its own `.venv/` here, used for
  editing/light testing (e.g. CPU-only smoke tests on `Pendulum-v1`).
- **Real experiment runs happen on a separate Linux box**, checked out at
  `/home/samanth/rl-dissection-linux` with a venv named `rl-exp/` (see
  `cli_commands.txt`, `run_*_sweep.sh`). GPU clock locking and CPU governor
  pinning (the confound controls) require root, so real runs are launched
  with `sudo rl-exp/bin/python run_experiment.py ...` — without `sudo` both
  controls fail soft and the run proceeds "unlocked" (still valid, just
  noted in `metadata.json`/logs).
- The project was originally developed on Windows, then ported to Linux
  (git commit `d05cbe6`) once real MuJoCo/GPU runs started — Linux is now
  the canonical environment for anything producing thesis numbers.
- `requirements.txt`: torch, gymnasium(+mujoco), codecarbon>=3.0,
  nvidia-ml-py, pandas, plotly, dash.

## Repo layout

```
configs/config.py         ExperimentConfig (protocol/timing) + SACConfig/MBPOConfig/TD3Config/TDMPC2Config
configs/overrides/*.json  Per-env/per-sweep hyperparameter overrides (see below)
utils/gpu_control.py      GPU clock locking, persistence mode, CPU governor (GpuCpuGuard)
utils/thermal_gate.py     Waits between runs until GPU temp/power return near a reference cold state
algorithms/replay_buffer.py   Numpy circular replay buffer
algorithms/tracker_utils.py   Shared CodeCarbon start_task/stop_task context manager (TrackerTask)
algorithms/sac.py         SAC agent + segment-instrumented train() loop
algorithms/td3.py         TD3 agent + segment-instrumented train() loop
algorithms/dynamics_model.py  MBPO's probabilistic ensemble dynamics model (PETS-style)
algorithms/termination_fns.py Per-env early-termination heuristics for MBPO model rollouts
algorithms/mbpo.py        MBPO train() loop — reuses SACAgent unmodified, adds model training + branched rollouts
algorithms/tdmpc2.py      TD-MPC2 latent world model + MPPI planner + train() loop
experiment_runner.py      Orchestrates settle -> idle baseline -> warmup+train -> idle tail -> teardown
run_experiment.py         CLI entry point (--algo, --env, --seed, --algo-config-overrides, ...)
aggregate_results.py      Combines many runs' segment_energy.json/metadata.json/training_metrics.json into one CSV
dashboard.py              Interactive Plotly Dash app for browsing individual runs under results/
flop_analysis/            Energy-per-FLOP methodology (measure_flops.py, compute_energy_per_flop.py, flop_keys.py)
flop_dashboard.py         Interactive Plotly Dash app for browsing flop_analysis/output/*.csv (cross-seed + per-run tabs)
scripts/                  All sweep-runner shell scripts (moved out of repo root, commit 88170be)
scripts/run_mbpo_env_sweep.sh     MBPO across HalfCheetah-v5/Ant-v5, 5 seeds — 10 runs
scripts/run_td3_env_sweep.sh      TD3 across HalfCheetah-v5/Ant-v5, 5 seeds, warmup=10000 — 10 runs
scripts/run_tdmpc2_seed_sweep.sh  TD-MPC2 across HalfCheetah-v5/Ant-v5, 5 seeds — 10 runs
scripts/run_utd_sweep.sh          MBPO UTD∈{2,4} sweep, both envs, 5 seeds — 20 runs (reuses the filename that
                                   originally ran SAC-Ant/TD3-both-envs' UTD sweep, aa9186b/24f41b0 — that data
                                   is still under results/sac/Ant-v5, results/td3/, its own status file was
                                   overwritten when the script was repointed at MBPO, see "Runs recorded" below)
scripts/run_width_sweep.sh        hidden_sizes∈{256,512} (vs. canonical 1024) for sac/mbpo/td3, both envs, 5 seeds — 60 runs
scripts/run_batch_size_sweep.sh       batch_size∈{256,512,1024} for sac/mbpo/td3, HalfCheetah-v5, 5 seeds — 35 runs
scripts/run_batch_size_sweep_ant.sh   batch_size∈{256,512,1024} for sac/mbpo/td3, Ant-v5, 5 seeds — 35 runs
scripts/run_mbpo_rollout_length_sweep.sh  MBPO rollout_max_length∈{1,15}, Ant-v5, 5 seeds — 10 runs
scripts/run_tdmpc2_numq_horizon_sweep.sh  TD-MPC2 num_q∈{3,7} and horizon∈{1,5} (2 separate ablations), both envs, 5 seeds — 40 runs
scripts/run_ant_overnight_sweep.sh    combines run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh into one
                                       unattended overnight session (single sudo priming/shutdown) — 45 runs
cli_commands.txt          Reference invocations for the Linux box (sudo + rl-exp venv)
results/                  All run output, organized results/{algo}/{env}/seed_{n}/{timestamp}/
```

## Experiment protocol (see README.md for full rationale)

```
[settle, unmeasured] -> [idle baseline head] -> [warmup] -> [epochs: rollout, gradient_updates (+ MBPO/TDMPC2 extras)] -> [idle baseline tail] -> [teardown]
```

Each run writes `results/{algo}/{env}/seed_{n}/{timestamp}/`:
`emissions.csv` (CodeCarbon native log), `segment_energy.json` (reconciled
per-segment kWh), `training_metrics.json` (per-episode/per-epoch RL
performance), `metadata.json` (git commit, versions, full config, thermal
gate info), `run.log`.

Key design decisions (full justification in README.md):
- CodeCarbon tasks are tagged **once per epoch** (`rollout_{i}`,
  `gradient_updates_{i}`), not per gradient step — reusing a task name across
  start/stop cycles corrupts CodeCarbon's internal state, and per-step
  tagging would be both overkill (thousands of calls) and finer than
  CodeCarbon's own polling resolution (`measure_power_secs=1.0`).
- `rollout` and `gradient_updates` are **direct hardware measurements**.
  Sub-segments (`buffer_sample`, `critic_update`, `actor_update`,
  `target_update` — and MBPO's `dynamics_model_update`/
  `synthetic_rollout_generation`) are **allocated** from the fused
  `gradient_updates` energy by wall-clock time share (`time.perf_counter()`
  inside each agent's `update()`), justified by GPU clock locking
  suppressing power variance across those sub-phases. This is documented
  explicitly so it's reported correctly in the thesis (measured vs.
  allocated).
- Confound controls: GPU clock lock + persistence mode + CPU `performance`
  governor (`utils/gpu_control.py`, needs root), thermal gating between runs
  (`utils/thermal_gate.py`, polls NVML temp/power until near a reference
  cold state, falls back to a flat 30s sleep without NVML), fresh process
  per run (no looping inside one long-lived Python process).
- Known measurement caveats: CodeCarbon's CPU energy degrades to a generic
  per-thread TDP estimate if RAPL (`/sys/class/powercap/intel-rapl`) isn't
  readable — check `run.log` for "default power consumption of 4 W per
  thread"; CO2e conversion can silently fall back to a default country if
  geolocation fails (doesn't affect kWh numbers); an idle-period overhead
  control run is recommended before trusting the smallest segment
  (`target_update`); ≥5 seeds per (algo, env) pair is the stated minimum
  given measurement noise.

## Algorithms implemented

All four share `ExperimentConfig` (protocol/timing, algorithm-agnostic) and
get their own hyperparameter dataclass in `configs/config.py`, registered in
`ALGO_CONFIGS = {"sac": SACConfig, "mbpo": MBPOConfig, "td3": TD3Config,
"tdmpc2": TDMPC2Config}`. `run_experiment.py --algo-config-overrides <json>`
applies a JSON field-override file on top of a config's dataclass defaults;
overrides live in `configs/overrides/`.

### SAC (`algorithms/sac.py`) — implemented first
Haarnoja et al. 2018, automatic entropy tuning variant. `hidden_sizes=(1024,1024)`
(settled on after earlier dev runs at (256,256)/(512,512)), twin critics,
Polyak target averaging (`tau=0.005`), `updates_per_env_step` (UTD) swept at
1 (default)/2/4. Segments: `rollout`, `buffer_sample`, `critic_update`,
`actor_update`, `target_update`.

### MBPO (`algorithms/mbpo.py`) — second
Janner, Fu, Zhang & Levine, NeurIPS 2019 (arxiv.org/abs/1906.08253), following
github.com/JannerM/mbpo closely enough to reuse its published HalfCheetah
hyperparameters as `MBPOConfig`'s defaults. Reuses `SACAgent` **completely
unmodified** as the inner policy optimizer via `_MixedReplayBuffer` (mixes
`real_ratio` real transitions with model-generated ones behind the same
`.sample()` interface) — this is what makes the SAC-vs-MBPO energy comparison
clean, since the policy-learning code path is byte-for-byte identical.
Adds: `algorithms/dynamics_model.py` (7-network probabilistic ensemble,
PETS-style, 5 elites by holdout error), `algorithms/termination_fns.py`
(per-env early-termination heuristics for branched rollouts), and two extra
directly-measured segments: `dynamics_model_update`, `synthetic_rollout_generation`.
HalfCheetah uses defaults (fixed rollout length 1); Hopper/Walker2d/Ant/Humanoid
need `configs/overrides/mbpo_{env}.json` for a longer scheduled rollout length
(Humanoid also needs a wider model, `model_hidden_sizes=(400,400,400,400)`).

### TD3 (`algorithms/td3.py`) — third
Fujimoto, van Hoof & Meger, ICML 2018 (arxiv.org/abs/1802.09477), paper's own
hyperparameters (Table 3) applied unmodified across all MuJoCo tasks — no
per-env override file needed. Deterministic actor + clipped double
Q-learning + delayed policy updates (`policy_update_delay=2`) + target policy
smoothing. `hidden_sizes` deliberately kept at (1024,1024) (paper uses
(400,300)) to hold network capacity constant across algorithms for the energy
comparison. The one env-dependent paper setting is `--warmup-steps 10000` for
HalfCheetah-v5/Ant-v5 ("stable length" envs, vs. 1000 default elsewhere) — not
part of `TD3Config` since `ExperimentConfig.warmup_steps` is shared across
algos, so it must be passed explicitly on the CLI.

### TD-MPC2 (`algorithms/tdmpc2.py`) — fourth, most recently added
Hansen, Su & Wang, ICLR 2024 (arxiv.org/abs/2310.16828), single-task
state-observation port of github.com/nicklashansen/tdmpc2, reusing the
paper's own hyperparameters unmodified. Not model-free actor-critic — learns
a decoder-free latent world model (encoder w/ SimNorm, latent dynamics,
reward model, 5-network Q-ensemble via soft two-hot discrete regression,
Gaussian policy prior) and acts via **MPPI planning in latent space**
(512 samples, 24 seeded from the policy prior, horizon=3, 6 iterations,
64 elites). `rollout` here is expensive (dominated by planning) — the
opposite of the other three algorithms where `rollout` is cheap relative to
`gradient_updates`. Segment mapping: `critic_update` = world-model step
(consistency+reward+value losses), `actor_update` = policy-prior step,
`target_update` = Q-ensemble Polyak average. Has one segment with no
analogue elsewhere: `world_model_pretrain` (a one-off burst of `warmup_steps`
gradient updates right after warmup ends, matching the reference
implementation's "pretraining on seed data"). `configs/overrides/tdmpc2_ant.json`
sets `episodic: true` for Ant-v5 (adds a termination classifier for imagined
MPPI rollouts, since Ant can terminate early unlike HalfCheetah).

### Extending to PPO / PETS (not yet done)
Add a `{Algo}Config` dataclass + `ALGO_CONFIGS` entry, a new
`algorithms/{algo}.py` with a `train(env, algo_cfg, exp_cfg, tracker, device,
logger, steps_per_epoch=...)` returning `(agent, energy_log, metrics)` using
the same `TrackerTask` pattern, and a dispatch branch in
`experiment_runner._dispatch_train`.

## Hyperparameter sensitivity sweeps (`scripts/run_*.sh`, most recent work)

Once the canonical 5-seed, `hidden_sizes=(1024,1024)` energy comparisons
existed for all four algorithms (see "Runs recorded so far"), the project
moved to testing whether those energy/energy-per-FLOP comparisons are
robust to architecture/hyperparameter choices, rather than an artifact of
the one settled configuration — i.e. does the *shape* of the SAC-vs-MBPO-vs-
TD3-vs-TD-MPC2 energy comparison hold if you change network width, batch
size, update-to-data ratio, MBPO's model-rollout length, or TD-MPC2's
Q-ensemble size/planning horizon? Every sweep below reuses the canonical
5-seed set `{331, 958, 14577, 43611, 85062}` so results are directly
comparable to the existing baseline runs, and (per `flop_analysis/flop_keys.py`)
each distinct architecture gets its own FLOP signature so
`compute_energy_per_flop.py` never averages across incompatible architectures.
All sweep scripts share one shape: prime+keepalive a sudo credential cache
(needed for GPU clock lock/CPU governor per run), write progress to a
`results/..._status.json` + `.log` pair, and power the machine off when done
(meant to be launched inside `screen`/`nohup` and left running unattended,
often overnight) — see each script's own header comment for exact combos.

- **Width sweep** (`run_width_sweep.sh`, commit `96bab02`/`7e672ef`, values
  updated `5f6c074`): `hidden_sizes` ∈ {(256,256), (512,512)} — revisits the
  same widths as the pre-canonical dev exploration, but now at the settled
  protocol/hyperparameters and the full 5-seed set, both envs, for sac/mbpo/td3.
  TD-MPC2 excluded (no `hidden_sizes` field — see script header). 60 runs.
- **MBPO UTD sweep** (`run_utd_sweep.sh`, commit `ecb7135`/`e53e805`):
  `updates_per_env_step` ∈ {2, 4} for MBPO on both envs — the SAC/TD3 UTD
  sweep already existed (see below); this fills in the MBPO leg by reusing
  (and repointing) the same script file. 20 runs.
- **Batch-size sweeps** (`run_batch_size_sweep.sh`/`_ant.sh`, commit
  `9b4b44c`…`ccc3ee5`): `batch_size` ∈ {256, 512, 1024} for sac/mbpo/td3
  (256 skipped for sac/mbpo since it's already their default, covered by the
  canonical baseline runs), HalfCheetah-v5 and Ant-v5 separately. 35 runs each.
- **MBPO model-rollout-length sweep** (`run_mbpo_rollout_length_sweep.sh`
  and its override files, all created correctly in one shot at `788ffcd`;
  `9d989a3` only recomputed `flop_analysis/output/*` and `flop_dashboard.py`
  after `mbpo_rollout_regime()` was added to `flop_keys.py`, it did **not**
  change any override hyperparameters): `rollout_max_length`
  ∈ {1, 15} on Ant-v5 only (`rollout_min_length` fixed at 1 in both) —
  brackets the established Ant-v5 schedule (`mbpo_ant.json`, max length 25)
  to see how model-rollout length trades off against dynamics-model/synthetic-
  rollout energy. Deliberately **not** folded into the FLOP architecture
  signature (`flop_keys.mbpo_rollout_regime` tracks it separately instead,
  since rollout length changes call *counts* not per-call FLOP costs). 10 runs.
- **TD-MPC2 num_q / horizon sweep** (`run_tdmpc2_numq_horizon_sweep.sh`,
  commit `a3582b5`/`88170be`): two independent single-parameter ablations —
  `num_q` ∈ {3, 7} (Q-ensemble size) and `horizon` ∈ {1, 5} (MPPI planning
  horizon) — each across both envs (defaults 5/3 already covered by the
  canonical seed sweep, so skipped here). 40 runs.
- **`run_ant_overnight_sweep.sh`** (commit `5a21db9`) is the Ant-v5 batch-size
  sweep and the MBPO rollout-length sweep combined into a single unattended
  session (one sudo priming, one status/log file, one shutdown at the end);
  `run_batch_size_sweep_ant.sh`/`run_mbpo_rollout_length_sweep.sh` still exist
  standalone if you want to rerun just one phase.
- `dashboard.py`/`flop_dashboard.py` were updated alongside the TD-MPC2 sweep
  to display the new data correctly (commit `88170be`, y-axis label fix `f09a861`).

All five sweeps completed 100% of their scheduled runs (see each
`results/..._status.json`, all-`"done"`) as of `f09a861` (current HEAD).

## Runs recorded so far (chronological, per git history)

Git history order: SAC first (`fd50077` first HalfCheetah run) → ported to
Linux (`d05cbe6`) → architecture/seed conventions settled at
`hidden_sizes=(1024,1024)` after dev exploration at (256,256)/(512,512)/on
Humanoid → per-algo config dataclasses replaced CLI hyperparameter flags
(`2aa4ccd`) → **canonical 5-seed set fixed**: `{331, 958, 14577, 43611,
85062}` (this is the seed set that recurs across every algo/env/sweep;
earlier seeds `0`, `42`, `56` are leftover dev/exploratory runs at older
architectures/configs) → MBPO added and its full env sweep recorded
(`f109bd4`…`d9cf6a7`) → TD3 added and recorded (`58962d2`, `1f39982`) → UTD
sweep (`updates_per_env_step` ∈ {2, 4}) run for SAC-on-Ant-v5 and TD3-on-both-envs
(`aa9186b`, `24f41b0`) — SAC-on-HalfCheetah's UTD sweep had already been done
earlier (`d1cebb4`) → TD-MPC2 added and its seed sweep recorded across both
envs (`b5b7e54`, `2378fe7`) → FLOP-per-segment analysis pipeline built
(`6a0f790`) → **hyperparameter sensitivity sweeps** (see section above):
width sweep (`7e672ef`…`5f6c074`) → MBPO UTD sweep (`ecb7135`…`96aba25`) →
batch-size sweeps, HalfCheetah then Ant+MBPO-rollout-length combined
(`9b4b44c`…`9f58907`) → TD-MPC2 num_q/horizon sweep + dashboard updates
(`a3582b5`, `88170be`, `f09a861`, current HEAD).

Concretely, under `results/` (see `dashboard.py`/`flop_dashboard.py`/
`aggregate_results.py` to browse/summarize; this is a snapshot as of
`f09a861` — re-check `results/` for ground truth). Run counts below are
per-seed-folder timestamp-directory counts for the 5 canonical seeds
(verified via `ls results/*/*/seed_*`):

- **SAC**: `HalfCheetah-v5` (heaviest history — dev runs at seed 0/42/56
  across architectures, plus 7 runs/canonical seed: baseline + UTD2 + UTD4 +
  width256 + width512 + batch512 + batch1024), `Ant-v5` (same 7 runs/seed
  pattern), `Humanoid-v5` (seed 56 only, exploratory — no full sweep, not
  part of any of the sweeps above).
- **MBPO**: `HalfCheetah-v5` (seed 0 dev runs + 7 runs/canonical seed:
  baseline + UTD2 + UTD4 + width256 + width512 + batch512 + batch1024, using
  `MBPOConfig` defaults — no override needed), `Ant-v5` (9 runs/canonical
  seed: the same 7 plus rollout1 + rollout15, all needing
  `configs/overrides/mbpo_ant*.json` for the rollout schedule). Hopper/
  Walker2d/Humanoid have override files prepared but **no recorded runs**.
- **TD3**: `HalfCheetah-v5` (seed 0 dev run + 8 runs/canonical seed: baseline
  + UTD2 + UTD4 + width256 + width512 + batch256 + batch512 + batch1024, all
  `--warmup-steps 10000`), `Ant-v5` (same 8 runs/seed pattern, same warmup
  override). Note TD3 gets a batch256 run (unlike SAC/MBPO) since its own
  paper default is 100, not 256.
- **TD-MPC2**: `HalfCheetah-v5` (seed 0 dev run + 5 runs/canonical seed:
  baseline + numq3 + numq7 + horizon1 + horizon5, plain defaults), `Ant-v5`
  (same 5 runs/seed pattern, all with `configs/overrides/tdmpc2_ant*.json`
  for `episodic=true`).
- Top-level sweep bookkeeping files: `results/_width_sweep_status.json`,
  `_utd_sweep_status.json` (now the MBPO UTD sweep — see "Hyperparameter
  sensitivity sweeps" above for why this overwrote the earlier SAC/TD3 UTD
  sweep's bookkeeping file; the underlying run data under
  `results/sac/Ant-v5`/`results/td3/` is unaffected), `_batch_size_sweep_status.json`
  (HalfCheetah-v5), `_ant_overnight_sweep_status.json` (Ant-v5 batch sizes +
  MBPO rollout-length), plus per-algo `_env_sweep.log`/`_seed_sweep_status.json`
  files under `results/mbpo/`, `results/td3/`, `results/tdmpc2/`, and
  `results/tdmpc2/_numq_horizon_sweep_status.json`. `results/sac/HalfCheetah-v5/
  _utd_sweep_status.json` is the original SAC-HalfCheetah-only UTD sweep
  (`d1cebb4`, pre-canonical-seed) and is untouched by the later overwrite above.
- `results/_thermal_reference.json` — audit-only snapshot from the most
  recent run's thermal gate (overwritten every run, never read back).

**Net picture**: full 5-seed energy comparisons exist for SAC/MBPO/TD3/TD-MPC2
on HalfCheetah-v5 and Ant-v5, each now with a UTD∈{1,2,4} sweep, a network-width
sweep ((256,256)/(512,512), sac/mbpo/td3 only), a batch-size sweep
({256,512,1024}), and an algorithm-specific structural sweep on top (MBPO
model-rollout-length on Ant-v5; TD-MPC2 num_q/horizon on both envs) — all run
through the same `flop_analysis/` pipeline so energy-per-FLOP comparisons can
be checked for robustness across architectures. Humanoid-v5 and Hopper/
Walker2d (MBPO) have config support but no completed sweep — natural
next-step candidates if more comparisons are needed.

## Data-quality exceptions and notes (verified against `results/`, for thesis reporting)

Everything below was checked directly against the 300 run directories and
their `metadata.json`/`run.log` files on disk, not just script intent — worth
citing/flagging explicitly in the thesis write-up.

**Positive validity findings** (i.e. things that did *not* go wrong, checked
because they easily could have):
- All 300 recorded runs ran on the **same physical GPU**
  (`NVIDIA GeForce RTX 5090`) — no hardware-swap confound anywhere in the
  dataset.
- **Zero** RAPL-fallback warnings ("default power consumption of 4 W per
  thread"), **zero** geolocation-fallback warnings, **zero** thermal-gate
  timeouts, and **zero** GPU-clock-lock/CPU-governor permission failures
  across all 300 `run.log` files — every run got real RAPL+NVML hardware
  energy readings and had its confound controls actually applied, not just
  requested. (Checked by grepping every `run.log` for the known failure-mode
  strings documented in "Known measurement caveats" above.)
- All five hyperparameter sweeps plus every seed/env sweep completed
  **100% of their scheduled runs** (`results/**/*_status.json` all show
  every entry `"done"`, zero `"failed"`) — the numbers below don't include
  any runs still pending.

**Exceptions worth flagging**:
- **TD-MPC2 num_q/horizon sweep had a false start** (`results/tdmpc2/
  _numq_horizon_sweep.log`, 2026-09-18 12:18): the first launch of
  `run_tdmpc2_numq_horizon_sweep.sh` crashed instantly on its first 5 runs
  (`num_q=3`, HalfCheetah-v5, all 5 canonical seeds) with
  `FileNotFoundError: configs/overrides/tdmpc2_num_q3.json` — the script's
  `override_file()` originally generated the override filename with an
  underscore (`num_q3`) where the actual file is `numq3.json`. Each failure
  happened in `run_experiment.py`'s `build_algo_config()` *before* any
  process/GPU/tracker work started, so **no orphaned run directories, no
  wasted GPU/energy time, and no thermal-state disturbance** resulted — the
  script was stopped, the filename bug fixed (the fix is documented inline
  in the script's own comment), and the full 40-run sweep relaunched
  cleanly from run 1 at 12:21. The final `results/tdmpc2/
  _numq_horizon_sweep_status.json` (40/40 done) and the 5 TD-MPC2 run
  directories per canonical seed on disk both reflect only the successful
  second attempt — there is nothing to filter out or exclude.
- **`scripts/run_utd_sweep.sh` was repurposed mid-project**: it originally
  ran the SAC-on-Ant-v5/TD3-on-both-envs UTD∈{2,4} sweep (`aa9186b`,
  `24f41b0`) and was later edited in place to run MBPO's UTD sweep instead
  (`ecb7135`, `e53e805`), reusing the exact same status-file path
  (`results/_utd_sweep_status.json`). The **run data itself is untouched**
  (still under `results/sac/Ant-v5/`, `results/td3/*/*/` with full
  `metadata.json` provenance either way), but the *top-level bookkeeping
  file* for the original SAC/TD3 UTD sweep was silently overwritten by the
  later MBPO sweep's status — don't use `results/_utd_sweep_status.json` as
  evidence of the SAC/TD3 UTD sweep's completion; use the actual run
  directories/`metadata.json` instead (or `results/sac/HalfCheetah-v5/
  _utd_sweep_status.json`, which is a *different*, untouched sweep — SAC-
  HalfCheetah-only, `d1cebb4`, predating the canonical seed set).
- **Software-stack version drift, isolated to SAC's earliest dev history**:
  `results/sac/HalfCheetah-v5/seed_0/*` and `seed_42/*` (7 runs total,
  2026-08-13 to 2026-08-26) ran under **torch 2.11.0+cu128 / codecarbon
  3.2.9–3.3.0**; literally every other recorded run in the entire
  project — every canonical-seed run, every sweep run, every other
  algorithm/environment/seed — ran under **torch 2.13.0+cu130 / codecarbon
  3.3.0** (verified across all 300 `metadata.json` files, grouped by
  `(algo, env)`: SAC-HalfCheetah is the *only* pair with more than one
  version combination). This compounds the already-known reason seed 0/42
  must be excluded from any comparison (they also predate the
  `hidden_sizes=(1024,1024)` architecture) with a second, independent
  reason: they also predate the current PyTorch/CUDA/CodeCarbon stack, so
  even their `dynamics_model_update`/etc. energy numbers at matching
  architecture would not be a clean comparison against later runs.
- **One-off GPU clock-lock anomaly**: exactly one run in the whole dataset —
  `results/mbpo/HalfCheetah-v5/seed_0/20260904_130915` (an already-excluded
  seed-0 dev run) — has `gpu_min_clock_mhz`/`gpu_max_clock_mhz` = **200/200**
  instead of the standard 2000/2000. Doesn't touch canonical or sweep data,
  but flag it explicitly if this specific run is ever cited.
- **`gpu_min/max_clock_mhz: null` on 15 of the very earliest SAC dev runs**
  (seed 0/42/56, 2026-08-13 to 08-31): predates the CLI exposing an explicit
  clock-lock default: `utils.gpu_control.GpuCpuGuard` falls back to its own
  `DEFAULT_LOCK_MHZ=2000` in this case, so these runs are **functionally
  identical** to the later explicit-2000/2000 runs, not a real confound —
  just explains why the field reads `null` in their `metadata.json`.
- **`training_metrics.json` missing on 4 of the earliest SAC seed_0 runs**
  (2026-08-13/08-19, before per-episode/per-epoch return logging was added
  in `4b9f11a`): all other 296 run directories in `results/` (every
  canonical-seed run, every sweep run) have the complete file set
  (`metadata.json`, `segment_energy.json`, `training_metrics.json`,
  `emissions.csv`, `run.log`) — verified directly, not just assumed.
- The `flop_calculation_methodology.md` referenced by both
  `flop_analysis/measure_flops.py` and `compute_energy_per_flop.py`'s
  docstrings is **not checked into the repo** (see FLOP section below) —
  confirm with the user whether it exists in the thesis document itself
  before citing it as a repo file.

## FLOP / energy-per-FLOP analysis (`flop_analysis/`)

Goal: express each measured segment's energy as **Joules per FLOP**, not just
kWh, to compare algorithms on a compute-normalized basis.

1. **`measure_flops.py`** — for every distinct `(algo, env_id,
   architecture_signature)` combo actually found under `results/`
   (`flop_keys.signature()` fingerprints hidden sizes / batch size / MBPO
   ensemble config / TD-MPC2 planner config — several architectures coexist
   in `results/` from the project's evolution, e.g. SAC HalfCheetah at
   (256,256)/(512,512)/(1024,1024)), runs `torch.utils.flop_counter.FlopCounterMode`
   against the **real** network classes / `TDMPC2Agent.act()`/`update()` to
   get exact per-call FLOP counts (rollout forward, critic fwd+bwd, actor
   fwd+bwd, target-update elementwise ops, MBPO dynamics ensemble, TD-MPC2's
   3-way critic/actor/target breakdown cross-checked against a black-box
   `agent.update()` call). Includes an analytic dense-layer cross-check.
   Output: `flop_analysis/flops_per_call.json`.
2. **`compute_energy_per_flop.py`** — joins each run's `metadata.json`
   (hyperparameters → call counts, since this repo logs per-epoch not
   per-gradient-step CodeCarbon tasks) and `segment_energy.json` (the
   authoritative per-segment kWh, already reconciled from measured
   `gradient_updates`) against `flops_per_call.json`, per architecture
   signature. Cross-seed averaging **defaults to the canonical 5-seed set**
   `{331, 958, 14577, 43611, 85062}` (`--all-seeds` to include everything, or
   `--seeds` for a custom list) — dev/exploratory runs at other seeds/older
   architectures are still written to the per-run CSV
   (`included_in_cross_seed_avg=False`) but excluded from the cross-seed
   average so they don't dilute it.
   Outputs: `flop_analysis/output/per_run_energy_per_flop.csv` (one row per
   run×segment, plus a `TOTAL_MEASURED_TRAINING` rollup row per run summing
   matmul-FLOP-accounted segments only — excludes idle baselines, `warmup`,
   `buffer_sample` (CPU-side, 0 FLOPs), and `target_update` (elementwise
   Polyak ops, not matmul FLOPs — expect a meaningless-looking Energy/FLOP
   there by design)) and `cross_seed_energy_per_flop.csv` (averaged by
   `(algo, env, architecture, UTD, segment)`).
3. `flop_keys.py` — shared architecture-signature functions
   (`sig_sac_td3`/`sig_mbpo`/`sig_tdmpc2`) used by both scripts so a run is
   always joined against FLOP constants for its *actual* architecture, plus
   `mbpo_rollout_regime()` which fingerprints MBPO's model-rollout-length
   schedule (`rollout_min/max_length`, `rollout_min/max_epoch`) *separately*
   from `sig_mbpo` — rollout length changes how many `synthetic_rollout_generation`
   calls a run makes, not the per-call FLOP cost, so it must not be conflated
   with the architecture signature (added for the MBPO rollout-length sweep,
   see "Hyperparameter sensitivity sweeps" above).

Note: both scripts' docstrings reference a `flop_calculation_methodology.md`
for the full step-by-step methodology — **this file does not currently exist
in the repo** (not found under any tracked path); it may live only in the
thesis write-up, or still needs to be created/committed. Worth checking with
the user before assuming it's missing/lost.

## Aggregation & visualization tooling

- **`aggregate_results.py`** — flat CSV across all runs (energy per segment +
  key training-performance summary stats: final cumulative reward, episode
  count, last-10%-mean return, max return). `python aggregate_results.py
  --results-dir results --out summary.csv`.
- **`dashboard.py`** — Plotly Dash app, cascading Algorithm→Environment→
  Seed→Folder dropdowns, per-run phase duration/power/energy breakdown bars
  plus raw CodeCarbon per-task metrics and training episode/epoch curves
  (toggle any numeric column). `python dashboard.py [--results-dir results]
  [--port 8050]`.
- **`flop_dashboard.py`** — separate Plotly Dash app over
  `flop_analysis/output/*.csv` (not `results/` directly): a "cross-seed
  comparison" tab (one row per `(algo, env, architecture, UTD, segment)`,
  averaged over the canonical 5 seeds) and a "per-run detail" tab (one row
  per run×segment, including dev/exploratory runs flagged via
  `included_in_cross_seed_avg`), each with crossfiltering dropdowns pivoting
  into a grouped bar chart or per-seed box plot, plus a sortable data table.
  `python flop_dashboard.py [--flop-dir flop_analysis/output] [--port 8051]`.

## Quick reference: running an experiment

```bash
# Fast CPU smoke test (any algo), no MuJoCo/GPU/thermal gate needed:
python run_experiment.py --algo sac --env Pendulum-v1 --seed 0 --device cpu \
    --train-steps 2000 --warmup-steps 500 --steps-per-epoch 500 \
    --settle-seconds 2 --idle-baseline-seconds 3 --idle-tail-seconds 3 \
    --no-gpu-lock --no-thermal-gate --output-dir results_smoketest

# Real run (Linux box, sudo for confound controls):
sudo rl-exp/bin/python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 0
sudo rl-exp/bin/python run_experiment.py --algo mbpo --env Hopper-v5 --seed 0 \
    --algo-config-overrides configs/overrides/mbpo_hopper.json
sudo rl-exp/bin/python run_experiment.py --algo td3 --env HalfCheetah-v5 --seed 0 --warmup-steps 10000
sudo rl-exp/bin/python run_experiment.py --algo tdmpc2 --env Ant-v5 --seed 0 \
    --algo-config-overrides configs/overrides/tdmpc2_ant.json

# After collecting runs:
python aggregate_results.py --results-dir results --out summary.csv
python flop_analysis/measure_flops.py            # once per new architecture
python flop_analysis/compute_energy_per_flop.py  # joins results/ against flops_per_call.json

# Launching a full unattended sweep (Linux box only — needs sudo, screen/nohup
# recommended since these run for hours and shut the machine down when done):
screen -S width_sweep
scripts/run_width_sweep.sh
# Ctrl-A D to detach; reattach with: screen -r width_sweep
```
