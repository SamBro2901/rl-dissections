# RL Energy Profiling Harness

Measures per-segment energy consumption of RL training (rollout vs. buffer
sampling vs. critic update vs. actor update vs. target update, plus idle
baseline) using CodeCarbon (NVML + RAPL under the hood).

## Layout

```
configs/config.py        ExperimentConfig (protocol/timing) + SACConfig/MBPOConfig/TD3Config/TDMPC2Config
configs/overrides/        --algo-config-overrides JSON files -- per-env hyperparameters (e.g. MBPO rollout
                           schedules) and per-sweep hyperparameters (width/batch-size/UTD/rollout-length/
                           num_q/horizon -- see "Hyperparameter sensitivity sweeps" below)
utils/gpu_control.py      GPU clock locking, persistence mode, CPU governor (GpuCpuGuard context manager)
utils/thermal_gate.py     Waits between runs until GPU temp/power return near a reference cold state
algorithms/replay_buffer.py  Numpy circular replay buffer
algorithms/tracker_utils.py  Shared CodeCarbon start_task/stop_task context manager (TrackerTask)
algorithms/sac.py         SAC agent + segment-instrumented train() loop
algorithms/td3.py         TD3 agent + segment-instrumented train() loop
algorithms/dynamics_model.py  MBPO's probabilistic ensemble dynamics model (PETS-style)
algorithms/termination_fns.py  Per-env early-termination heuristics for model rollouts
algorithms/mbpo.py        MBPO train() loop -- reuses SACAgent, adds model training + branched rollouts
algorithms/tdmpc2.py      TD-MPC2 latent world model + MPPI planner + train() loop
experiment_runner.py      Orchestrates settle -> idle baseline -> warmup+train -> idle tail -> teardown
run_experiment.py         CLI entry point
aggregate_results.py      Combines many runs' segment_energy.json/metadata.json/training_metrics.json into one CSV
dashboard.py              Plotly Dash app for browsing individual runs under results/
flop_analysis/            Energy-per-FLOP methodology: measure_flops.py, compute_energy_per_flop.py, flop_keys.py
flop_dashboard.py         Plotly Dash app for browsing flop_analysis/output/*.csv
scripts/                  Sweep-runner shell scripts (seed sweeps, UTD/width/batch-size/rollout-length/
                           num_q/horizon sweeps -- see "Hyperparameter sensitivity sweeps" below)
cli_commands.txt          Reference invocations for the Linux experiment box (sudo + rl-exp venv)
```

## Quickstart

```bash
pip install -r requirements.txt

# Fast smoke test (no MuJoCo needed, no GPU lock/thermal gate, short run):
python run_experiment.py --algo sac --env Pendulum-v1 --seed 0 --device cpu \
    --train-steps 2000 --warmup-steps 500 --steps-per-epoch 500 \
    --settle-seconds 2 --idle-baseline-seconds 3 --idle-tail-seconds 3 \
    --no-gpu-lock --no-thermal-gate --output-dir results_smoketest

# Real run on your RL box:
python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 0

# MBPO (model-based; same env/CLI surface, its own hyperparameters live in MBPOConfig):
python run_experiment.py --algo mbpo --env HalfCheetah-v5 --seed 0

# MBPO on an env with early termination needs its own rollout-length schedule
# (see configs/overrides/mbpo_*.json and MBPOConfig's docstring):
python run_experiment.py --algo mbpo --env Hopper-v5 --seed 0 \
    --algo-config-overrides configs/overrides/mbpo_hopper.json

# TD3 (model-free; the paper's warmup length differs by env -- see TD3Config's
# docstring -- so pass it explicitly on HalfCheetah/Ant):
python run_experiment.py --algo td3 --env HalfCheetah-v5 --seed 0 --warmup-steps 10000
python run_experiment.py --algo td3 --env Ant-v5 --seed 0 --warmup-steps 10000

# TD-MPC2 (planning-based; HalfCheetah-v5 uses defaults as-is, Ant-v5 needs the
# episodic-termination override since it can terminate early -- see TDMPC2Config):
python run_experiment.py --algo tdmpc2 --env HalfCheetah-v5 --seed 0
python run_experiment.py --algo tdmpc2 --env Ant-v5 --seed 0 \
    --algo-config-overrides configs/overrides/tdmpc2_ant.json

# After several runs across seeds/envs:
python aggregate_results.py --results-dir results --out summary.csv
```

Each run writes to `results/{algo}/{env}/seed_{n}/{timestamp}/`:
- `emissions.csv` — CodeCarbon's native per-task log (one row per task, including
  every per-epoch `rollout_{i}` / `gradient_updates_{i}` task)
- `segment_energy.json` — reconciled per-segment energy in kWh (see below)
- `metadata.json` — full provenance: git commit, torch/codecarbon versions,
  GPU name, full config, thermal-gate wait info
- `run.log` — this run's log output

## Protocol phases

```
[settle, unmeasured] -> [idle baseline head] -> [warmup] -> [epochs: rollout, gradient_updates] -> [idle baseline tail] -> [teardown]
```

- **Settle** (default 30s, unmeasured): lets OS/driver background activity and
  the previous process's GPU clocks/thermal state die down before any
  tracker starts.
- **Idle baseline head/tail** (default 90s each): CodeCarbon-tracked idle
  periods with no workload. Measuring *both* ends (not just one) lets you
  check for power-floor drift across a long session — if head and tail
  disagree substantially, treat that run's segment numbers with more caution.
- **Warmup** (default 5,000 steps): random-policy rollout to fill the replay
  buffer before any gradient step. Tracked as its own segment but deliberately
  kept separate from the "measured training" segments below, since it's not
  representative once the buffer is warm.
- **Epochs**: training is chunked into epochs of `steps_per_epoch` (default
  1,000) environment steps. Each epoch = one `rollout` task (env stepping +
  action selection + buffer insertion) + one `gradient_updates` task
  (`steps_per_epoch × updates_per_env_step` SAC gradient steps).

## Why epoch-level tagging instead of per-gradient-step tagging

We tested CodeCarbon's `start_task`/`stop_task` directly (codecarbon 3.2.9)
and found two things that shaped this design:

1. **Reusing the same task name across multiple start/stop cycles corrupts
   internal state** (confirmed: the second reuse produced an accumulated
   duration and codecarbon logged `_active_task_emissions_at_start was None`).
   The fix is to give every `start_task` call a **unique name** — we do this
   with `{prefix}_{counter}` — and aggregate by prefix afterward.
2. Calling `start_task`/`stop_task` once per environment step (never mind once
   per gradient step) would mean hundreds of thousands of calls for a
   100k-step run. Each call does real work (hardware queries, bookkeeping),
   and CodeCarbon's polling granularity (`measure_power_secs`, we use 1s) is
   coarser than a single gradient step anyway — so tagging at that resolution
   buys you nothing but overhead.

So the harness only creates **two directly-measured CodeCarbon tasks per
epoch**: `rollout_{epoch}` and `gradient_updates_{epoch}`. Within
`gradient_updates`, `SACAgent.update()` times its own four sub-phases
(`buffer_sample`, `critic_update`, `actor_update`, `target_update`) with
`time.perf_counter()`. At the end of the run, the **total** measured
`gradient_updates` energy is split across these four sub-phases in proportion
to their aggregate wall-clock time share across the whole run.

**This means `rollout` and `gradient_updates` (before the split) are direct
hardware measurements. `buffer_sample`, `critic_update`, `actor_update`, and
`target_update` are an *allocation*, not an independent measurement** — they
rest on the assumption that instantaneous power draw doesn't vary wildly
between these four sub-phases. This is a reasonable approximation here
because (a) GPU clocks are locked (see below), which suppresses the biggest
source of power variance, and (b) all four sub-phases are similar in
character (small-batch tensor ops on the same networks). It is **not** a
reasonable approximation between `rollout` and `gradient_updates` themselves
(env stepping is often CPU-bound / MuJoCo-simulation-bound, while gradient
updates are GPU-bound) — which is exactly why those two stay as direct,
independent measurements rather than being split by time-share too.

**For your thesis write-up**: report `rollout` and `gradient_updates` as
directly measured, and clearly flag `buffer_sample` / `critic_update` /
`actor_update` / `target_update` as a time-proportional allocation within
`gradient_updates`, not an independent per-component hardware measurement.
If a reviewer pushes on this, the honest answer is "CodeCarbon's temporal
resolution doesn't support sub-second hardware attribution, so we allocated
by wall-clock share under a constant-power approximation, justified by GPU
clock locking."

If you later want a *directly measured* critic-vs-actor split instead of an
allocated one, the only way to get it with CodeCarbon is to restructure the
update loop so it does all critic updates for a chunk of steps, then all actor
updates for that chunk, each as its own task — but that changes the SAC
update order (actor updates then see a critic that's had many more updates
than in standard interleaved SAC), so it's a real algorithmic tradeoff, not
just an instrumentation change. Worth a footnote if you go that route.

## Confound controls

- **GPU clock locking** (`utils/gpu_control.py`, `GpuCpuGuard`): calls
  `nvidia-smi -pm 1` (persistence mode) and `nvidia-smi -lgc <min>,<max>` to
  lock the graphics clock to a fixed low-variance value, resetting with
  `-rgc` on exit (even on exception, via try/finally in the context
  manager). Both clock-lock calls require root -- without it they fail soft
  and log a permission warning, but the clock lock silently isn't applied.
  (Persistence mode itself doesn't need root if `nvidia-persistenced` is
  running, since the daemon applies it on your behalf.) Disable clock
  locking with `--no-gpu-lock` (e.g. if you deliberately want to study
  boost-clock behavior later, or you're on a machine without an NVIDIA GPU).
- **CPU governor**: `gpu_control.py` pins every core to the `performance`
  governor via `cpupower frequency-set -g performance` for the duration of
  the run, restoring whatever governor was active before on exit. Writing
  the governor sysfs node also requires root, so like GPU clock locking this
  fails soft with a warning if you don't run under `sudo`.
- **Run under `sudo` to get both confound controls actually applied**:
  `sudo rl-exp/bin/python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 0`
  (see `cli_commands.txt`). Without root, GPU clock locking and CPU governor
  pinning both no-op with a logged warning and the run proceeds unlocked --
  still valid to run, just flag it as an unlocked-clocks run in your notes.
- **Thermal gating** (`utils/thermal_gate.py`): before starting a run,
  polls GPU temp/power via NVML and blocks until they're within tolerance
  of a reference "cold" state. The reference is captured fresh at the
  start of every process (not reused from a previous script invocation),
  after a short stabilization poll so it doesn't lock in a still-hot
  reading left over from the prior run. It's written to
  `results/_thermal_reference.json` for audit purposes only (each run
  overwrites it; it's never read back). This adapts to thermal drift over
  a multi-hour session better than a fixed cooldown sleep would. Falls
  back to a flat 30s sleep if NVML is unavailable. Disable with
  `--no-thermal-gate`.
- **Fresh process per run**: `run_experiment.py` is meant to be invoked once
  per run (e.g. from a shell loop over seeds/algorithms), not looped inside
  one long-lived Python process, to avoid memory/cache/CUDA-context carryover
  between runs.

## MBPO

`algorithms/mbpo.py` implements Model-Based Policy Optimization (Janner, Fu,
Zhang & Levine, NeurIPS 2019, https://arxiv.org/abs/1906.08253), following
the paper's reference implementation (https://github.com/JannerM/mbpo)
closely enough to reuse its published HalfCheetah hyperparameters as
`MBPOConfig`'s defaults (see that dataclass's docstring in `configs/config.py`
for the full list and citations).

It reuses `SACAgent` from `sac.py` completely unmodified as its inner policy
optimizer — the only reason this works is `_MixedReplayBuffer` in `mbpo.py`,
which presents the same `.sample()` interface as `ReplayBuffer` while drawing
`real_ratio` of each minibatch from real env transitions and the rest from
model-generated ones, so `SACAgent.update()` never needs to know the data is
mixed. On top of that, MBPO adds:

- **`algorithms/dynamics_model.py`** — a 7-network probabilistic ensemble
  (PETS-style, Chua et al. 2018) predicting a diagonal Gaussian over
  `[delta_obs, reward]`, retrained periodically on all real data seen so far
  with holdout early stopping; the 5 lowest-holdout-error members ("elites")
  are used for prediction.
- **`algorithms/termination_fns.py`** — per-environment early-termination
  heuristics (matching Gymnasium's `healthy_z_range`/`healthy_angle_range`
  checks) so branched model rollouts stop at a fallen-over Hopper/Walker2d/
  Ant/Humanoid instead of continuing through physically invalid states.
- Two extra directly-measured CodeCarbon tasks per epoch, alongside `rollout`
  and `gradient_updates`: **`dynamics_model_update`** (retraining the
  ensemble) and **`synthetic_rollout_generation`** (branched model rollouts
  filling the model buffer). `gradient_updates` is sub-split into
  `buffer_sample`/`critic_update`/`actor_update`/`target_update` by the exact
  same time-proportional allocation scheme as SAC (see above) — reusing
  `SACAgent.update()` means that instrumentation comes along for free.

This gives the cleanest possible model-free-vs-model-based energy comparison
available under this harness's protocol, since the policy-learning code path
(`SACAgent`) is byte-for-byte identical between the two `--algo` runs; only
what feeds its replay buffer differs.

MBPO defaults to HalfCheetah's published rollout schedule (fixed length 1 —
the paper finds HalfCheetah gets no benefit from longer imagined rollouts).
Hopper/Walker2d/Ant/Humanoid need a longer, scheduled rollout length (and
Humanoid needs a wider dynamics model); `configs/overrides/mbpo_*.json` has
the paper's settings for each, passed via `--algo-config-overrides`.

## TD3

`algorithms/td3.py` implements Twin Delayed Deep Deterministic Policy
Gradient (Fujimoto, van Hoof & Meger, ICML 2018,
https://arxiv.org/abs/1802.09477), reusing the paper's own hyperparameters
(Section 6.1, Table 3) as `TD3Config`'s defaults. Unlike MBPO, the paper
applies the same hyperparameters across every MuJoCo-v1 task it evaluates
-- including HalfCheetah-v1 and Ant-v1 -- so there's no per-env
`--algo-config-overrides` file for TD3; `TD3Config`'s docstring in
`configs/config.py` has the full list and citations.

Structurally it mirrors `sac.py` (same `TrackerTask` segment names, same
`train()` return shape) but is model-free and off-policy with a
*deterministic* actor rather than SAC's stochastic Gaussian policy, so the
update step differs in three paper-specified ways:

- **Clipped double Q-learning** — both critics are trained against a shared
  target that takes the min over `Q1_targ`/`Q2_targ`, capping the
  overestimation bias a single critic (or DDPG's single critic) is prone to.
- **Delayed policy updates** — the actor and *both* target networks
  (`policy_update_delay`, `d=2` by default) are only updated once every `d`
  critic updates, so the actor is trained against a lower-variance,
  more-converged critic.
- **Target policy smoothing** — clipped `N(0, target_policy_noise)` noise is
  added to the target action before it's passed to the target critics,
  regularizing the target against sharp Q-value peaks a deterministic policy
  could otherwise exploit.

Exploration is handled differently from SAC too: since the actor is
deterministic, `train()` adds `N(0, exploration_noise * act_limit)` Gaussian
noise to the selected action during rollout (clipped to the action bounds)
rather than relying on the policy's own stochasticity.

The paper's warmup length is env-dependent (Section 6.1) — 10,000 purely
random steps for HalfCheetah-v1/Ant-v1, 1,000 for the rest. Since
`ExperimentConfig.warmup_steps` is shared across all algorithms in this
harness (not part of `TD3Config`), pass `--warmup-steps 10000` explicitly
when running TD3 on HalfCheetah-v5/Ant-v5 (see Quickstart above).

## TD-MPC2

`algorithms/tdmpc2.py` implements TD-MPC2 (Hansen, Su & Wang, ICLR 2024,
"TD-MPC2: Scalable, Robust World Models for Continuous Control",
https://arxiv.org/abs/2310.16828), a single-task, state-observation port of
the authors' reference implementation
(https://github.com/nicklashansen/tdmpc2) closely enough to reuse the
paper's own hyperparameters as `TDMPC2Config`'s defaults (see that
dataclass's docstring in `configs/config.py`).

Unlike every other algorithm in this repo, TD-MPC2 is **not** a model-free
actor-critic method wrapped around direct policy queries: it learns a
decoder-free latent world model --

- an **encoder** mapping raw observations to a `SimNorm`-normalized latent
  state (https://arxiv.org/abs/2204.00616 -- softmax over small groups of
  latent units, used to keep the latent space bounded and stable),
- a **latent dynamics model** predicting the next latent state,
- a **reward model** and a **5-network Q-ensemble**, both trained as
  *discrete regression* (soft two-hot cross-entropy in a symlog-transformed
  space, `num_bins=101`) rather than direct MSE/Huber regression,
- a **Gaussian policy prior** trained to maximize Q-value plus an entropy
  bonus --

and at every environment step, selects actions by **MPPI trajectory
optimization in latent space** (`TDMPC2Agent.act`/`_estimate_value`):
`num_samples=512` imagined action sequences (`num_pi_trajs=24` seeded from
the policy prior, the rest from a Gaussian search distribution) are rolled
out through the *learned* dynamics/reward models over a short
`horizon=3` and scored, the top `num_elites=64` re-fit the search
distribution's mean/std, this repeats `iterations=6` times, and one action
is sampled from the final distribution. The policy prior itself is mostly
there to seed and regularize this search -- it is never used to act
directly.

The world model is trained on random-length-3 subsequences (not single
transitions) drawn uniformly from a replay buffer of whole episodes
(`_EpisodeSequenceBuffer` in `algorithms/tdmpc2.py` -- a plain-numpy
adaptation of the reference's torchrl `SliceSampler`-based buffer). Its loss
combines, at every step of the sampled horizon (weighted by `rho**t`,
`rho=0.5`):
- a **consistency loss**: only the first latent state in the sequence comes
  from encoding a real observation; every later one is produced by unrolling
  the dynamics model, and is pulled toward the encoding of the corresponding
  real next-observation (stop-gradient target). This is what makes
  multi-step *imagined* rollouts (used both in training and in MPPI
  planning) trustworthy.
- **reward loss** and **value loss** (soft two-hot cross-entropy against a
  TD-target bootstrapped off a target Q-ensemble, exactly as SAC/TD3 use
  target critics in this repo).

Mapping onto this repo's segment names: `critic_update` is the world-model
step (consistency + reward + value losses, since the Q-ensemble lives
inside the world model here) and `actor_update` is the policy-prior step,
by analogy with SAC/TD3's critic/actor split; `target_update` is the
Q-ensemble's Polyak average. `rollout` includes the MPPI planner, which
dominates TD-MPC2's per-step compute -- the opposite of SAC/TD3/MBPO, where
`rollout` is cheap relative to `gradient_updates`. Two segments have no
analogue elsewhere in this repo:
- **`world_model_pretrain`**: right when warmup ends, the reference
  algorithm runs a one-off burst of `warmup_steps` gradient updates on the
  seed data ("pretraining on seed data") before ever using the planner for
  real actions. This module reproduces that burst as its own
  directly-measured CodeCarbon task.
- Every other algorithm here interleaves rollout and gradient updates
  strictly per-epoch (all of an epoch's rollout, then all of its updates);
  TD-MPC2's reference implementation instead interleaves them one env-step
  at a time. This module keeps the per-epoch separation for consistency
  with the rest of this repo's CodeCarbon tagging scheme (see "Why
  epoch-level tagging" above) -- a deliberate deviation from the reference
  training loop's exact interleaving, not from its hyperparameters or
  losses.

`configs/overrides/tdmpc2_ant.json` sets `episodic: true` for Ant-v5, which
turns on an auxiliary termination classifier used only to truncate
*imagined* MPPI rollouts (Gymnasium's Ant-v5 can terminate early on an
unhealthy state, unlike HalfCheetah-v5, which never does) -- see
`TDMPC2Config`'s docstring for why the real TD-target bootstraps correctly
off the environment's own termination signal either way.

## Energy-per-FLOP analysis (`flop_analysis/`)

Raw per-segment kWh numbers only tell you how much energy an algorithm used
at *one particular architecture and batch size*; they don't tell you whether
that's because the algorithm is inherently more expensive per unit of useful
compute, or just because someone gave it a bigger network. `flop_analysis/`
adds a second, compute-normalized metric -- **Joules per FLOP** -- so
algorithms (and hyperparameter choices) can be compared on that basis too.
It's a two-step pipeline:

1. **`flop_analysis/measure_flops.py`** -- for every distinct `(algo, env_id,
   architecture_signature)` combination actually present under `results/`
   (see `flop_keys.py` below), runs `torch.utils.flop_counter.FlopCounterMode`
   against the **real** network classes (`SACAgent`, `TD3Agent`, MBPO's
   dynamics ensemble, `TDMPC2Agent.act()`/`.update()`) to get exact per-call
   FLOP counts: rollout forward pass, critic forward+backward, actor
   forward+backward, target-update elementwise ops, MBPO's dynamics-ensemble
   forward/backward, and TD-MPC2's three-way critic/actor/target breakdown
   (cross-checked against a black-box `agent.update()` call, since TD-MPC2's
   world-model/policy-prior/Q-ensemble losses aren't as cleanly separable as
   SAC/TD3's critic/actor split). Includes an analytic dense-layer FLOP
   cross-check as a sanity check on the profiler's own numbers. Output:
   `flop_analysis/flops_per_call.json`.
2. **`flop_analysis/compute_energy_per_flop.py`** -- joins each run's
   `metadata.json` (hyperparameters, used to derive call *counts* -- this
   harness logs per-epoch, not per-gradient-step, so a call count has to be
   reconstructed from `steps_per_epoch × updates_per_env_step` etc., not read
   directly off a CodeCarbon task) and `segment_energy.json` (the
   authoritative per-segment kWh, already reconciled from the directly
   measured `rollout`/`gradient_updates` tasks -- see "Why epoch-level
   tagging" above) against `flops_per_call.json`, per architecture signature.
   Cross-seed averaging defaults to the canonical 5-seed set `{331, 958,
   14577, 43611, 85062}` (`--all-seeds` to include every run found instead,
   or `--seeds 1,2,3` for a custom list); dev/exploratory runs at other seeds
   or older architectures are still written to the per-run CSV
   (`included_in_cross_seed_avg=False`) but excluded from the cross-seed
   average so they don't dilute it. Outputs:
   - `flop_analysis/output/per_run_energy_per_flop.csv` -- one row per
     run×segment, plus a `TOTAL_MEASURED_TRAINING` rollup row per run summing
     only matmul-FLOP-accounted segments (excludes idle baselines, `warmup`,
     `buffer_sample` (CPU-side, 0 FLOPs), and `target_update` (elementwise
     Polyak-average ops, not matmul FLOPs -- expect a near-meaningless
     Energy/FLOP figure there by design, not a bug).
   - `flop_analysis/output/cross_seed_energy_per_flop.csv` -- averaged by
     `(algo, env, architecture, UTD, segment)`.

`flop_analysis/flop_keys.py` supplies the shared architecture-signature
functions (`sig_sac_td3`, `sig_mbpo`, `sig_tdmpc2`) both scripts use, so a
run is always joined against the FLOP constants for its *actual*
architecture -- `results/` holds more than one architecture per (algo, env)
pair (e.g. SAC on HalfCheetah-v5 has runs at (256,256), (512,512), and
(1024,1024) from the project's evolution, plus the hyperparameter sweeps
below), and averaging across architectures would silently corrupt the
metric. `sig_sac_td3`/`sig_mbpo` key on `hidden_sizes`/`batch_size` (plus
MBPO's ensemble config); `sig_tdmpc2` keys on the planner config
(`num_q`, `horizon`, `num_samples`, etc.) and `episodic`. A separate
`mbpo_rollout_regime()` function fingerprints MBPO's model-rollout-length
schedule independently of the architecture signature -- rollout length
changes how many `synthetic_rollout_generation` calls a run makes, not the
per-call FLOP cost, so conflating the two would be wrong.

Browse the output interactively with **`flop_dashboard.py`** (`python
flop_dashboard.py [--flop-dir flop_analysis/output] [--port 8051]`) -- a
cross-seed-comparison tab and a per-run-detail tab, each with crossfiltering
dropdowns that pivot the filtered rows into a grouped bar chart (or, on the
per-run tab, a per-seed box plot), plus a sortable/filterable data table.

Both `measure_flops.py`/`compute_energy_per_flop.py` reference a
`flop_calculation_methodology.md` for the full step-by-step methodology --
**this file is not currently checked into the repo**; it may exist only in
the thesis write-up, or may still need to be created. Worth confirming
before assuming the methodology documentation is lost.

## Hyperparameter sensitivity sweeps (`scripts/`)

Once 5-seed energy comparisons existed for all four algorithms at the
settled `hidden_sizes=(1024,1024)` architecture (see runs recorded in
`CLAUDE.md`), the natural next question is whether those comparisons --
energy in kWh *and* energy-per-FLOP -- are robust to the specific
architecture/hyperparameter choice, or an artifact of that one point. A set
of sweep scripts under `scripts/` answer this by revisiting the same
canonical 5-seed set `{331, 958, 14577, 43611, 85062}` at different
hyperparameter values, on both `HalfCheetah-v5` and `Ant-v5` unless noted:

- **Network width** (`scripts/run_width_sweep.sh`): `hidden_sizes` ∈
  {(256,256), (512,512)} for SAC/MBPO/TD3 (TD-MPC2 excluded -- its
  architecture is the paper's own fixed spec with no equivalent width knob).
  60 runs.
- **Update-to-data ratio (UTD)**: `updates_per_env_step` ∈ {2, 4}. SAC-on-
  Ant-v5 and TD3-on-both-envs were swept first (`scripts/run_utd_sweep.sh`
  originally); MBPO's UTD sweep (`scripts/run_utd_sweep.sh` as it exists now
  -- the file was later repointed at MBPO, see `CLAUDE.md` for the exact
  history) fills in the fourth algorithm, on both envs. SAC-on-HalfCheetah's
  UTD sweep predates the canonical seed set and was run separately earlier.
- **Batch size** (`scripts/run_batch_size_sweep.sh` for HalfCheetah-v5,
  `scripts/run_batch_size_sweep_ant.sh` for Ant-v5): `batch_size` ∈ {256,
  512, 1024} for SAC/MBPO/TD3 (256 skipped for SAC/MBPO where it's already
  each algorithm's own default, already covered by the baseline runs; TD3's
  own paper default is 100, so all three sizes are new for TD3). 35 runs per env.
- **MBPO model-rollout length** (`scripts/run_mbpo_rollout_length_sweep.sh`,
  Ant-v5 only): `rollout_max_length` ∈ {1, 15} (with `rollout_min_length`
  held at 1), bracketing the established Ant-v5 schedule
  (`configs/overrides/mbpo_ant.json`, max length 25) to see how the length of
  MBPO's branched imagined rollouts trades off against
  `dynamics_model_update`/`synthetic_rollout_generation` energy. 10 runs.
- **TD-MPC2 num_q / horizon** (`scripts/run_tdmpc2_numq_horizon_sweep.sh`):
  two independent single-parameter ablations, `num_q` (Q-ensemble size) ∈
  {3, 7} and `horizon` (MPPI planning horizon) ∈ {1, 5} -- the paper
  defaults (5 and 3 respectively) are already covered by the canonical seed
  sweep, so they're skipped here. 40 runs.
- **`scripts/run_ant_overnight_sweep.sh`** combines the Ant-v5 batch-size
  sweep and the MBPO rollout-length sweep into a single unattended overnight
  session (one sudo priming, one status/log file, one shutdown at the end);
  the two component scripts still exist standalone if you want to rerun just
  one phase.

Every sweep script follows the same shape: prime a sudo credential cache up
front (needed once per run for GPU clock locking + CPU governor pinning),
keep it alive in the background for the sweep's duration, write live
progress to a `results/..._status.json` + `.log` pair (one JSON row per
planned run: `pending` → `running` → `done`/`failed`), and power the
machine off automatically once every run has been attempted -- meant to be
launched inside `screen`/`nohup` and left running unattended (often
overnight, given some sweeps take dozens of runs). See each script's own
header comment for the exact algo/env/value combinations and run counts;
run one with, e.g.:

```bash
screen -S width_sweep
scripts/run_width_sweep.sh
# Ctrl-A D to detach; reattach later with: screen -r width_sweep
```

All hyperparameter-sweep runs go through the same `configs/overrides/*.json`
mechanism as any other run (`--algo-config-overrides`) and land in the same
`results/{algo}/{env}/seed_{n}/{timestamp}/` layout as the baseline runs --
they're distinguished from each other only by their `metadata.json` config,
which is exactly what `flop_analysis/flop_keys.py`'s architecture signatures
(and, for MBPO rollout length, `mbpo_rollout_regime()`) key on. As of the
most recent commit, all five sweeps above have completed every one of their
scheduled runs.

## Extending to PPO / PETS

- Add a new `{Algo}Config` dataclass to `configs/config.py` and register it
  in `ALGO_CONFIGS`.
- Add a new `algorithms/{algo}.py` with a `train(env, algo_cfg, exp_cfg,
  tracker, device, logger, steps_per_epoch=...)` function returning
  `(agent, energy_log, metrics)`, following the same `TrackerTask` pattern
  (`algorithms/tracker_utils.py`, shared by `sac.py` and `mbpo.py`: unique
  task names per epoch, sub-segment timing via `time.perf_counter()` where
  finer breakdown than CodeCarbon's resolution allows).
- Add a dispatch branch in `experiment_runner._dispatch_train`.

## Known exceptions in the recorded data

The sections above describe the harness as designed; this section documents
what was actually found on disk when auditing all 300 recorded runs under
`results/` (as opposed to generic caveats about the methodology, which are
in "Known limitations" below). Cite these directly in the thesis rather than
re-deriving them, since some are easy to miss by just looking at summary CSVs.

**Confirmed clean** (checked because these are exactly the kind of thing that
would silently invalidate a comparison if they went wrong):
- Every one of the 300 runs used the same physical GPU (`NVIDIA GeForce RTX
  5090`) -- no hardware substitution anywhere in the dataset.
- No run's log shows a RAPL fallback, a geolocation fallback, a thermal-gate
  timeout, or a GPU-clock-lock/CPU-governor permission failure -- every run's
  confound controls were actually applied (not merely requested), and every
  run's CPU/GPU energy is a real hardware reading, not an estimate.
- Every hyperparameter sweep (width, UTD, batch-size ×2, MBPO rollout-length,
  TD-MPC2 num_q/horizon) and every per-algorithm seed/env sweep completed
  100% of its scheduled runs -- no partial sweeps to account for.

**Exceptions to know about**:
- **The TD-MPC2 num_q/horizon sweep had a false start.** Its first launch
  (2026-09-18, `results/tdmpc2/_numq_horizon_sweep.log`) crashed instantly on
  its first 5 runs with a `FileNotFoundError` from a filename mismatch in the
  sweep script (`tdmpc2_num_q3.json` generated vs. the actual
  `tdmpc2_numq3.json`). The crash happened before `run_experiment.py` ever
  started a process/tracker, so nothing was written to `results/` and no
  energy/GPU time was spent on the failed attempts -- the script was fixed and
  the full 40-run sweep relaunched cleanly from the beginning a few minutes
  later. The recorded data (5 runs × 2 envs × 2 params × 5 seeds = 40 runs)
  is exclusively from the successful second attempt; there's nothing to
  exclude or filter out.
- **`scripts/run_utd_sweep.sh` (and its status-file path,
  `results/_utd_sweep_status.json`) was reused for two different sweeps.**
  It originally ran the SAC-on-Ant-v5/TD3-on-both-envs UTD∈{2,4} sweep, then
  was edited in place to run MBPO's UTD sweep instead, overwriting the
  original sweep's top-level status/log bookkeeping (the actual run data for
  both sweeps is intact under `results/sac/Ant-v5/`, `results/td3/*/*/`, and
  `results/mbpo/*/*/`, distinguishable via each run's own `metadata.json`).
  Don't treat `results/_utd_sweep_status.json` as a record of the SAC/TD3 UTD
  sweep -- it now reflects the MBPO one.
- **A software-stack version change happened partway through the project,
  but only affects already-excluded dev runs.** `results/sac/HalfCheetah-v5/
  seed_0/*` and `seed_42/*` (7 runs, Aug 2026) ran under torch 2.11.0+cu128 /
  codecarbon 3.2.9-3.3.0; every other run in the entire dataset -- all
  canonical-seed runs, all sweep runs, every other algorithm/environment --
  ran under torch 2.13.0+cu130 / codecarbon 3.3.0. This is a second,
  independent reason (beyond their non-canonical architectures) that seed
  0/42 must never be mixed into a canonical or sweep-based comparison.
- **One run was locked at the wrong GPU clock.**
  `results/mbpo/HalfCheetah-v5/seed_0/20260904_130915` (an already-excluded
  seed-0 dev run) has GPU clocks locked at 200/200 MHz instead of the
  standard 2000/2000 MHz -- isolated to that single dev run, doesn't touch
  any canonical or sweep data, but flag it if that specific run is ever cited.
- **15 of the earliest SAC dev runs** (seed 0/42/56, Aug 2026) have
  `gpu_min_clock_mhz`/`gpu_max_clock_mhz: null` in their metadata -- this
  predates the CLI exposing an explicit default and resolves to the same
  2000 MHz via `GpuCpuGuard`'s own internal default, so it's not a real
  confound, just an artifact of older metadata.
- **`training_metrics.json` is missing on 4 of the earliest SAC seed_0 runs**
  (Aug 13/19, before per-episode/per-epoch return logging existed). All 296
  other run directories -- every canonical-seed and sweep run included --
  have the complete output set (`metadata.json`, `segment_energy.json`,
  `training_metrics.json`, `emissions.csv`, `run.log`).
- The `flop_calculation_methodology.md` referenced by both FLOP-analysis
  scripts' docstrings is not checked into this repo -- confirm whether it
  exists as part of the thesis document itself before citing it as a repo file.

## Known limitations / things to sanity-check before trusting the numbers

- **CodeCarbon's CPU energy model degrades to a generic per-thread TDP
  estimate** if it can't read `/sys/class/powercap/intel-rapl/subsystem`
  (permissions, non-Intel CPU, or a kernel built without the `powercap`
  subsystem). On a normal Linux desktop install this file is world-readable
  and RAPL "just works" with no root needed -- confirm with
  `cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj` (should print a
  number, not a permission error). `configs/config.py`'s
  `force_cpu_power_w` is `None` by default so CodeCarbon uses this real RAPL
  reading; only set it to a fixed watts figure if RAPL turns out to be
  unreadable on your box. Check your `run.log` for a warning like *"We will
  use the default power consumption of 4 W per thread"* — if you see this,
  RAPL wasn't actually read, your CPU-side numbers are a rough estimate
  rather than a direct hardware reading, and you should say so in the
  thesis. GPU (NVML) readings are unaffected by this.
- **Geolocation for CO2 conversion** may silently fall back to a default
  country if CodeCarbon can't reach its geolocation API (offline machine,
  firewall). This only affects the CO2e (kg) conversion, not the energy (kWh)
  numbers your segment analysis actually depends on — but don't report the
  CO2e figures without checking `run.log` for a geolocation fallback warning.
- Run the **overhead control** mentioned earlier: run the tracker across a
  no-op idle period of the same duration as a typical epoch, several times,
  and confirm the variance is small relative to your smallest measured
  segment (`target_update` will be your smallest — check this one first).
- **≥5 seeds per (algorithm, env) pair** — energy measurements are noisy
  (thermal state, background OS jitter); don't trust a single-seed number.
