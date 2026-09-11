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
run_utd_sweep.sh          SAC(Ant)/TD3(both envs) UTD∈{2,4} sweep, 5 seeds — 30 runs
run_mbpo_env_sweep.sh     MBPO across HalfCheetah-v5/Ant-v5, 5 seeds — 10 runs
run_td3_env_sweep.sh      TD3 across HalfCheetah-v5/Ant-v5, 5 seeds, warmup=10000 — 10 runs
run_tdmpc2_seed_sweep.sh  TD-MPC2 across HalfCheetah-v5/Ant-v5, 5 seeds — 10 runs
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

## Runs recorded so far (chronological, per git history)

Git history order: SAC first (`fd50077` first HalfCheetah run) → ported to
Linux (`d05cbe6`) → architecture/seed conventions settled at
`hidden_sizes=(1024,1024)` after dev exploration at (256,256)/(512,512)/on
Humanoid → per-algo config dataclasses replaced CLI hyperparameter flags
(`2aa4ccd`) → **canonical 5-seed set fixed**: `{331, 958, 14577, 43611,
85062}` (this is the seed set that recurs across every algo/env at the final
architecture; earlier seeds `0`, `42`, `56` are leftover dev/exploratory runs
at older architectures/configs) → MBPO added and its full env sweep recorded
(`f109bd4`…`d9cf6a7`) → TD3 added and recorded (`58962d2`, `1f39982`) → UTD
sweep (`updates_per_env_step` ∈ {2, 4}) run for SAC-on-Ant-v5 and TD3-on-both-envs
(`aa9186b`, `24f41b0`) — SAC-on-HalfCheetah's UTD sweep had already been done
earlier (`d1cebb4`) → TD-MPC2 added and its seed sweep recorded across both
envs (`b5b7e54`, `2378fe7`) → FLOP-per-segment analysis pipeline built
(`6a0f790`, current HEAD).

Concretely, under `results/` (see `dashboard.py`/`aggregate_results.py` to
browse/summarize; this is a snapshot, re-check `results/` for ground truth):

- **SAC**: `HalfCheetah-v5` (heaviest history — many dev runs at seed 0/42/56
  across architectures, plus the canonical 5-seed sweep, plus its own
  `_utd_sweep.log`/`_utd_sweep_status.json` for the UTD2/4 sweep),
  `Ant-v5` (canonical 5 seeds × 3 runs each = baseline + UTD2 + UTD4),
  `Humanoid-v5` (seed 56 only, exploratory — no full sweep).
- **MBPO**: `HalfCheetah-v5` (seed 0 dev runs + canonical 5 seeds, using
  `MBPOConfig` defaults — no override needed), `Ant-v5` (canonical 5 seeds,
  using `configs/overrides/mbpo_ant.json`). Hopper/Walker2d/Humanoid have
  override files prepared but **no recorded runs**.
- **TD3**: `HalfCheetah-v5` (seed 0 dev run + canonical 5 seeds × 3 runs =
  baseline + UTD2 + UTD4, `--warmup-steps 10000`), `Ant-v5` (canonical 5
  seeds × 3 runs, same warmup override).
- **TD-MPC2**: `HalfCheetah-v5` (seed 0 dev run + canonical 5 seeds, plain
  defaults), `Ant-v5` (canonical 5 seeds, `configs/overrides/tdmpc2_ant.json`
  for `episodic=true`).
- Top-level sweep bookkeeping files: `results/_utd_sweep.log` +
  `_utd_sweep_status.json` (the SAC-Ant/TD3-both UTD sweep), plus per-algo
  `_env_sweep.log`/`_seed_sweep_status.json` files under `results/mbpo/`,
  `results/td3/`, `results/tdmpc2/`.
- `results/_thermal_reference.json` — audit-only snapshot from the most
  recent run's thermal gate (overwritten every run, never read back).

**Net picture**: full 5-seed energy comparisons exist for SAC/MBPO/TD3/TD-MPC2
on HalfCheetah-v5 and Ant-v5, plus a UTD∈{1,2,4} sweep for SAC and TD3 on
both envs. Humanoid-v5 and Hopper/Walker2d (MBPO) have config support but no
completed sweep — natural next-step candidates if more comparisons are
needed.

## FLOP / energy-per-FLOP analysis (`flop_analysis/`, most recent work)

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
   always joined against FLOP constants for its *actual* architecture.

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
```
