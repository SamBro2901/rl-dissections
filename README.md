# RL Energy Profiling Harness

Measures per-segment energy consumption of RL training (rollout vs. buffer
sampling vs. critic update vs. actor update vs. target update, plus idle
baseline) using CodeCarbon (NVML + RAPL under the hood).

## Layout

```
configs/config.py        ExperimentConfig (protocol/timing) + SACConfig (hyperparameters)
utils/gpu_control.py      GPU clock locking, persistence mode, CPU governor (GpuCpuGuard context manager)
utils/thermal_gate.py     Waits between runs until GPU temp/power return near a reference cold state
algorithms/replay_buffer.py  Numpy circular replay buffer
algorithms/sac.py         SAC agent + segment-instrumented train() loop
experiment_runner.py      Orchestrates settle -> idle baseline -> warmup+train -> idle tail -> teardown
run_experiment.py         CLI entry point
aggregate_results.py      Combines many runs' segment_energy.json/metadata.json into one CSV
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
  `nvidia-smi -pm 1` (persistence mode -- reports `N/A`/no-op on Windows,
  where the driver is always resident regardless) and
  `nvidia-smi -lgc <min>,<max>` to lock the graphics clock to a fixed
  low-variance value, resetting with `-rgc` on exit (even on exception, via
  try/finally in the context manager). On Windows, `-lgc`/`-rgc` require an
  elevated (Administrator) shell -- without it they fail soft and log a
  permission warning, but the clock lock silently isn't applied. Disable
  clock locking with `--no-gpu-lock` (e.g. if you deliberately want to study
  boost-clock behavior later, or you're on a machine without an NVIDIA GPU).
- **CPU governor**: Windows has no per-core governor like Linux's
  `cpupower`; the closest equivalent is the active power plan, so
  `gpu_control.py` switches to the built-in "High performance" plan via
  `powercfg /setactive` for the duration of the run.
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

## Extending to PPO / MBPO / PETS

- Add a new `{Algo}Config` dataclass to `configs/config.py`.
- Add a new `algorithms/{algo}.py` with a `train(env, algo_cfg, exp_cfg,
  tracker, device, logger, steps_per_epoch=...)` function returning
  `(agent, energy_log)`, following the same `_TrackerTask` pattern used in
  `sac.py` (unique task names per epoch, sub-segment timing via
  `time.perf_counter()` where finer breakdown than CodeCarbon's resolution
  allows).
- Add a dispatch branch in `experiment_runner._dispatch_train`.
- For MBPO specifically: since it wraps SAC as its inner policy optimizer,
  you can literally reuse `SACAgent` and just add two more epoch-level tasks
  (`dynamics_model_update`, `synthetic_rollout_generation`) around it — this
  gives you the cleanest possible model-free-vs-model-based comparison, since
  the policy-learning code path is identical.

## Known limitations / things to sanity-check before trusting the numbers

- **CodeCarbon's CPU energy model degrades to a generic per-thread TDP
  estimate** if it doesn't recognize your CPU model or can't read
  `/sys/class/powercap/intel-rapl/subsystem` (permissions or non-Intel CPU).
  That path is Linux-only (part of the kernel's `powercap` sysfs subsystem),
  so **on Windows this fallback is essentially guaranteed** regardless of
  CPU model -- CodeCarbon's Windows alternative is Intel Power Gadget, which
  Intel has since discontinued and doesn't reliably support recent hybrid-
  core CPUs. Check your `run.log` for a warning like *"We will use the
  default power consumption of 4 W per thread"* — if you see this, your
  CPU-side numbers are a rough estimate, not a direct RAPL reading, and you
  should say so in the thesis. GPU (NVML) readings are unaffected by this.
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
