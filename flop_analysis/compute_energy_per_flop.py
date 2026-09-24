"""
Step 2+3 of the FLOP-per-segment methodology (documented in README.md,
section "Energy-per-FLOP analysis (`flop_analysis/`)").

For every run under results/<algo>/<env_id>/seed_*/<timestamp>/, this:
  1. Reads metadata.json for the exact hyperparameters used (batch size, UTD,
     policy_update_delay, warmup/train steps, ...) -- call counts are derived
     from these numbers directly, NOT by counting CodeCarbon task_name rows,
     because this codebase logs one CodeCarbon task per EPOCH ("rollout_N",
     "gradient_updates_N"), not one per individual gradient step. Counting
     rows would only recover the number of epochs, not the number of actual
     critic/actor update calls inside each epoch's fused "gradient_updates"
     task -- see the note in run_experiment.py-generated segment_energy.json
     below for the finer-grained energy split this script joins against.
  2. Reads segment_energy.json, which experiment_runner.py already writes
     per run -- it reconciles each epoch's *hardware-measured* CodeCarbon
     energy for the fused "gradient_updates" task into buffer_sample /
     critic_update / actor_update / target_update buckets by wall-clock time
     share (see algorithms/sac.py's module docstring). This is the
     authoritative per-segment Energy(segment, run) in kWh.
  3. Reads training_metrics.json for the few call counts that are genuinely
     data-dependent rather than derivable from config alone: MBPO's
     model_train_epochs (early-stopping is holdout-MSE-dependent) and
     synthetic_transitions_generated (termination-dependent rollout length).
     MBPO's dynamics_model_update FLOPs follow EnsembleDynamicsModel.fit()
     exactly (see mbpo_fit_flops()): per fit epoch, every member's fwd+bwd
     over the n_train bootstrap (true partial last batch) PLUS the
     full-ensemble holdout forward pass (_holdout_mse), from per-sample
     constants -- verified against FlopCounterMode on the real fit() by
     verify_mbpo_fit_flops.py.
  4. Joins against flops_per_call.json (produced by measure_flops.py) to get
     Total_FLOPs(segment, run), then Energy_per_FLOP = Energy_J / Total_FLOPs.
  5. Reads each run's per-task CodeCarbon log (run_dir/emissions_<experiment>_
     <run_id>.csv -- NOT the top-level emissions.csv, which holds only the
     one whole-run row written by tracker.stop(); see codecarbon's
     FileOutput.task_out()) to get duration_s and a CPU/GPU/RAM mean-power
     breakdown (energy / duration) per segment. buffer_sample/critic_update/
     actor_update/target_update aren't their own CodeCarbon task, so their
     CPU/GPU/RAM energy is split out of the measured "gradient_updates" task
     using the same wall-clock time-share ratio segment_energy.json already
     uses for the *total* energy allocation (see segment_hw_raw()).

Output:
  flop_analysis/output/per_run_energy_per_flop.csv       -- one row per (run, segment); every run,
                                                              tagged with included_in_cross_seed_avg
  flop_analysis/output/cross_seed_energy_per_flop.csv    -- averaged across seeds per (algo, env,
                                                              architecture, UTD, rollout regime, segment)

Each row carries flop_type: "matmul", "elementwise" (target_update -- its
energy_per_flop is J per elementwise Polyak op, NOT J/FLOP), "none"
(buffer_sample, warmup, idle baselines) or "mixed_total". Each run also gets
a TOTAL_MEASURED_TRAINING row = all training energy / matmul FLOPs only (see
TRAINING_SEGMENTS below); check_totals.py asserts its consistency.

By default, cross-seed averaging is restricted to the canonical 5-seed sweep
{331, 958, 14577, 43611, 85062} -- inspecting results/, this is the exact seed
set that recurs across every algo/env at the final (1024x1024, etc.)
architecture. A handful of older runs (seeds 0/42/56, some at earlier
architectures like hidden_sizes=(256,256)/(512,512)) are leftover
dev/exploratory runs from before that sweep was fixed and are excluded by
default so they don't silently dilute the average -- they still appear in
per_run_energy_per_flop.csv (with included_in_cross_seed_avg=False), just not
in the cross-seed aggregate.

Usage (from repo root):
    python3 flop_analysis/compute_energy_per_flop.py
    python3 flop_analysis/compute_energy_per_flop.py --seeds 331,958,14577,43611,85062  # equivalent to the default
    python3 flop_analysis/compute_energy_per_flop.py --all-seeds                        # average over every run found
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

from flop_keys import mbpo_rollout_regime, signature

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FLOPS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flops_per_call.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

KWH_TO_J = 3.6e6

# CodeCarbon's per-task log (see algorithms/tracker_utils.py's TrackerTask --
# every start_task/stop_task call gets a unique "{prefix}_{counter}" name).
# experiment_runner.py's tracker is created with save_to_file=True, which
# makes CodeCarbon write this to run_dir/emissions_<experiment_name>_<run_id>.csv
# (NOT the top-level emissions.csv, which only holds the one whole-run row
# written by tracker.stop()) -- see codecarbon's FileOutput.task_out().
SEGMENT_TASK_RE = re.compile(r"^(.*)_(\d+)$")

# buffer_sample/critic_update/actor_update/target_update are never their own
# CodeCarbon task -- they're allocated out of the fused "gradient_updates"
# task by wall-clock time share (see algorithms/sac.py's "reconcile" step,
# whose exact same ratios are replicated below for the CPU/GPU/RAM split).
ALLOCATED_SUB_SEGMENTS = ("buffer_sample", "critic_update", "actor_update", "target_update")

# The 5-seed sweep that recurs across every algo/env in results/ (see docstring above).
CANONICAL_SEEDS = {331, 958, 14577, 43611, 85062}

# Segments excluded from the primary energy accounting (matches dashboard.py's
# NON_TRAINING_PHASES): idle calibration windows aren't algorithmic compute.
NON_TRAINING_SEGMENTS = {"idle_baseline_head", "idle_baseline_tail"}

# TOTAL_MEASURED_TRAINING = all training energy / matmul FLOPs only.
# Membership is fixed here explicitly -- never inferred from whether a
# segment's FLOP count happens to be non-zero.
#   numerator:   energy of every TRAINING_SEGMENTS key present in segment_energy.json
#                (buffer_sample+critic_update+actor_update+target_update together
#                are the whole measured gradient_updates task)
#   denominator: FLOPs of the same segments minus ELEMENTWISE_OP_SEGMENTS
#                (target_update's count is elementwise Polyak ops, not matmul
#                FLOPs, and must never be mixed with them); buffer_sample adds 0.
# "warmup" (random-action buffer fill) is kept in the per-segment table for
# completeness but is not a training segment, and neither are the idle
# baselines; keys starting with "_" are bookkeeping, not segments.
TRAINING_SEGMENTS = {
    "rollout",
    "buffer_sample", "critic_update", "actor_update", "target_update",
    "dynamics_model_update", "synthetic_rollout_generation",  # MBPO
    "world_model_pretrain",                                   # TD-MPC2
}
EXCLUDED_FROM_TOTAL_SEGMENTS = NON_TRAINING_SEGMENTS | {"warmup"}
ELEMENTWISE_OP_SEGMENTS = {"target_update"}
ZERO_FLOP_SEGMENTS = {"buffer_sample"}  # CPU-side sampling, 0 FLOPs by definition

TOTAL_SEGMENT = "TOTAL_MEASURED_TRAINING"
TOTAL_NOTE = (
    "all training energy / matmul FLOPs only. Energy = sum over rollout, buffer_sample, "
    "critic_update, actor_update, target_update (= all of gradient_updates), "
    "dynamics_model_update, synthetic_rollout_generation (MBPO), world_model_pretrain "
    "(TD-MPC2), whichever are present; excludes idle_baseline_head/tail and warmup. "
    "FLOPs = matmul FLOPs of the same segments except target_update (elementwise op "
    "count, never mixed in); buffer_sample contributes 0. Duration/power fields use "
    "the directly measured gradient_updates task, not the allocated sub-segments."
)


def flop_type(segment):
    """What a segment's total_flops number counts: "matmul" (FlopCounterMode
    matmul FLOPs), "elementwise" (target_update's Polyak op count -- its
    energy_per_flop is J/op, NOT J/FLOP), "none" (no FLOP accounting:
    buffer_sample, warmup, idle baselines) or "mixed_total" (the
    TOTAL_MEASURED_TRAINING row: all training energy / matmul FLOPs only)."""
    if segment == TOTAL_SEGMENT:
        return "mixed_total"
    if segment in ELEMENTWISE_OP_SEGMENTS:
        return "elementwise"
    if segment in ZERO_FLOP_SEGMENTS or segment not in TRAINING_SEGMENTS:
        return "none"
    return "matmul"


def ceil_div(a: int, b: int) -> int:
    return -(-a // b) if b else 0


def load_flops():
    with open(FLOPS_JSON) as f:
        return json.load(f)


def find_runs(results_dir):
    for meta_path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "seed_*", "*", "metadata.json"))):
        run_dir = os.path.dirname(meta_path)
        seg_path = os.path.join(run_dir, "segment_energy.json")
        tm_path = os.path.join(run_dir, "training_metrics.json")
        if not os.path.exists(seg_path):
            continue
        yield run_dir, meta_path, seg_path, tm_path


def find_task_emissions_csv(run_dir):
    matches = sorted(glob.glob(os.path.join(run_dir, "emissions_*.csv")))
    return matches[0] if matches else None


def segment_from_task_name(task_name):
    m = SEGMENT_TASK_RE.match(task_name)
    return m.group(1) if m else task_name  # idle_baseline_head/tail carry no _N counter


def load_segment_hw_energy(run_dir):
    """Aggregates the run's per-task CodeCarbon CSV by segment (stripping the
    trailing _N call counter), returning summed wall-clock duration plus the
    CPU/GPU/RAM energy split CodeCarbon already measures per task (these three
    sum to exactly the task's energy_consumed, which is what TrackerTask sums
    into segment_energy.json)."""
    csv_path = find_task_emissions_csv(run_dir)
    if not csv_path:
        return {}
    agg = defaultdict(lambda: {"duration_s": 0.0, "cpu_energy_kwh": 0.0, "gpu_energy_kwh": 0.0, "ram_energy_kwh": 0.0,
                               "energy_consumed_kwh": 0.0})
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            task_name = row.get("task_name")
            if not task_name:
                continue
            a = agg[segment_from_task_name(task_name)]
            a["duration_s"] += float(row.get("duration") or 0.0)
            a["cpu_energy_kwh"] += float(row.get("cpu_energy") or 0.0)
            a["gpu_energy_kwh"] += float(row.get("gpu_energy") or 0.0)
            a["ram_energy_kwh"] += float(row.get("ram_energy") or 0.0)
            a["energy_consumed_kwh"] += float(row.get("energy_consumed") or 0.0)
    return agg


def segment_hw_raw(run_dir, seg_energy):
    """Per-segment {duration_s, cpu_energy_kwh, gpu_energy_kwh, ram_energy_kwh}
    for every segment_energy.json key that has hardware data available.
    Directly-measured segments (rollout/gradient_updates/dynamics_model_update/
    synthetic_rollout_generation/world_model_pretrain/warmup/idle_baseline_*)
    come straight from the per-task CSV. The four allocated sub-segments
    (ALLOCATED_SUB_SEGMENTS) are never their own CodeCarbon task, so their
    CPU/GPU/RAM energy is split out of gradient_updates' measured hardware
    energy using the same wall-clock time-share ratio segment_energy.json
    already used to allocate gradient_updates' *total* energy; their duration
    is that sub-segment's own summed time.perf_counter() wall time (an exact
    measurement, not itself an allocation) -- see
    segment_energy.json["_sub_segment_wall_time_seconds"] and the "reconcile"
    step in algorithms/sac.py (shared by td3.py/mbpo.py/tdmpc2.py)."""
    return _segment_hw_raw(load_segment_hw_energy(run_dir), seg_energy)


def _segment_hw_raw(hw, seg_energy):
    out = {segment: dict(vals) for segment, vals in hw.items()}

    sub_times = seg_energy.get("_sub_segment_wall_time_seconds")
    grad = hw.get("gradient_updates")
    if sub_times and grad:
        total_sub_time = sum(sub_times.values())
        for sub_seg in ALLOCATED_SUB_SEGMENTS:
            t = sub_times.get(sub_seg, 0.0)
            share = (t / total_sub_time) if total_sub_time > 0 else 0.0
            out[sub_seg] = {
                "duration_s": t,
                "cpu_energy_kwh": grad["cpu_energy_kwh"] * share,
                "gpu_energy_kwh": grad["gpu_energy_kwh"] * share,
                "ram_energy_kwh": grad["ram_energy_kwh"] * share,
            }
    return out


def mean_power_w(energy_kwh, duration_s):
    return (energy_kwh * KWH_TO_J / duration_s) if duration_s else None


def power_fields(raw):
    """raw: {duration_s, cpu_energy_kwh, gpu_energy_kwh, ram_energy_kwh} -> the
    duration/mean-power fields carried in the output CSVs."""
    d = raw["duration_s"]
    total_e = raw["cpu_energy_kwh"] + raw["gpu_energy_kwh"] + raw["ram_energy_kwh"]
    return {
        "duration_s": d,
        "mean_power_w": mean_power_w(total_e, d),
        "mean_cpu_power_w": mean_power_w(raw["cpu_energy_kwh"], d),
        "mean_gpu_power_w": mean_power_w(raw["gpu_energy_kwh"], d),
        "mean_ram_power_w": mean_power_w(raw["ram_energy_kwh"], d),
    }


EMPTY_POWER_FIELDS = {"duration_s": None, "mean_power_w": None, "mean_cpu_power_w": None,
                       "mean_gpu_power_w": None, "mean_ram_power_w": None}


def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def sac_like_call_counts(algo_config, experiment_config):
    steps_per_epoch = experiment_config["steps_per_epoch"]
    n_epochs = max(1, experiment_config["train_steps"] // steps_per_epoch)
    total_env_steps = n_epochs * steps_per_epoch
    n_updates = total_env_steps * algo_config["updates_per_env_step"]
    n_actor = ceil_div(n_updates, algo_config["policy_update_delay"])
    return {
        "rollout": total_env_steps,
        "buffer_sample": n_updates,
        "critic_update": n_updates,
        "actor_update": n_actor,
        "n_updates": n_updates,
        "n_actor": n_actor,
    }


def rows_for_sac_or_td3(algo, flops_key, algo_config, experiment_config):
    fc = sac_like_call_counts(algo_config, experiment_config)
    target_update_count = fc["n_updates"] if algo == "sac" else fc["n_actor"]  # SAC: unconditional; TD3: delay-gated
    rows = {
        "rollout": (fc["rollout"], flops_key["actor_forward_bs1"] * fc["rollout"], "actor forward, bs=1"),
        "buffer_sample": (fc["buffer_sample"], 0, "CPU-side gather, excluded from GPU FLOP accounting"),
        "critic_update": (fc["critic_update"], flops_key["critic_fwdbwd"] * fc["critic_update"], "twin-Q fwd+bwd"),
        "actor_update": (fc["actor_update"], flops_key["actor_fwdbwd"] * fc["actor_update"], "policy fwd+bwd (+alpha)"),
        "target_update": (
            target_update_count,
            flops_key["target_update_elementwise_ops"] * target_update_count,
            "Polyak averaging, elementwise ops (not matmul FLOPs -- expect a huge, meaningless Energy/FLOP here)",
        ),
    }
    return rows


def mbpo_fit_flops(n_total, train_epochs, algo_config, flops_key):
    """Matmul FLOPs of one EnsembleDynamicsModel.fit() call on a real buffer
    holding n_total transitions that ran train_epochs passes. Returns
    (train_flops, holdout_flops, optimizer_steps).

    Mirrors fit() in algorithms/dynamics_model.py: each pass trains every
    member on its n_train-sample bootstrap (the last minibatch is partial,
    charged at its true size) and then scores all members on the n_holdout
    split via _holdout_mse() -> forward_all(). Matmul FLOPs are exactly
    linear in batch size, so per-sample constants are taken from the
    measured batch-B constants (verified by verify_mbpo_fit_flops.py)."""
    train_batch = algo_config["model_train_batch_size"]
    member_fwdbwd = flops_key["dynamics_member_fwdbwd"]
    if member_fwdbwd % train_batch:
        raise ValueError(f"dynamics_member_fwdbwd={member_fwdbwd} not divisible by "
                         f"model_train_batch_size={train_batch}; linear per-sample scaling doesn't hold")
    per_sample_member_fwdbwd = member_fwdbwd // train_batch
    per_sample_ensemble_fwd = flops_key["dynamics_ensemble_forward_all_bs1"]  # already all members, bs=1

    n_holdout = max(1, int(n_total * algo_config["model_holdout_ratio"]))  # exactly as fit()
    n_train = n_total - n_holdout
    train_flops = train_epochs * algo_config["ensemble_size"] * n_train * per_sample_member_fwdbwd
    holdout_flops = train_epochs * n_holdout * per_sample_ensemble_fwd
    optimizer_steps = train_epochs * ceil_div(n_train, train_batch)
    return train_flops, holdout_flops, optimizer_steps


def rows_for_mbpo(flops_key, algo_config, experiment_config, epochs_log, run_dir=None):
    """Returns (rows, dynamics_split) -- dynamics_split holds the
    dynamics_model_update train/holdout FLOP breakdown for the CSV."""
    rows = rows_for_sac_or_td3("sac", flops_key, algo_config, experiment_config)

    warmup = experiment_config["warmup_steps"]
    steps_per_epoch = experiment_config["steps_per_epoch"]
    capacity = algo_config["buffer_capacity"]

    dyn_train = dyn_holdout = dyn_steps = 0
    for row in epochs_log:
        epochs_run = row.get("model_train_epochs")
        if not epochs_run:  # fit() skipped this epoch (model_train_freq) or never ran
            continue
        # algorithms/mbpo.py calls fit() at the start of each epoch, before that
        # epoch's rollout -- so the real buffer holds warmup + epoch * steps_per_epoch.
        n_total = min(warmup + row["epoch"] * steps_per_epoch, capacity)
        logged = row.get("buffer_size")  # len(real_buffer) at the END of the epoch
        if logged is not None and logged != min(n_total + steps_per_epoch, capacity):
            print(f"WARNING: {run_dir} epoch {row['epoch']}: logged buffer_size {logged} inconsistent "
                  f"with fit() size {n_total} + {steps_per_epoch} steps")
        train_f, holdout_f, steps = mbpo_fit_flops(n_total, epochs_run, algo_config, flops_key)
        dyn_train += train_f
        dyn_holdout += holdout_f
        dyn_steps += steps
    dyn_flops = dyn_train + dyn_holdout
    holdout_pct = (100.0 * dyn_holdout / dyn_flops) if dyn_flops else 0.0
    rows["dynamics_model_update"] = (
        dyn_steps, dyn_flops,
        f"per fit() epoch: ensemble_size x n_train x per-sample member fwd+bwd (exact partial last "
        f"batch) + n_holdout x per-sample full-ensemble forward (_holdout_mse). call_count = optimizer "
        f"steps (sum of epochs_run x ceil(n_train/batch)); epochs_run from training_metrics.json "
        f"(early stopping is data-dependent). train={dyn_train} holdout={dyn_holdout} "
        f"({holdout_pct:.2f}% holdout)",
    )
    dynamics_split = {"dynamics_train_flops": dyn_train, "dynamics_holdout_flops": dyn_holdout}

    per_sample = flops_key["actor_forward_bs1"] + flops_key["dynamics_ensemble_forward_all_bs1"]
    total_samples = sum(row.get("synthetic_transitions_generated", 0) for row in epochs_log)
    rows["synthetic_rollout_generation"] = (
        total_samples, total_samples * per_sample,
        "actor forward + full-ensemble forward, per synthetic sample (all members scored every predict() call)",
    )
    return rows, dynamics_split


def rows_for_tdmpc2(flops_key, algo_config, experiment_config):
    steps_per_epoch = experiment_config["steps_per_epoch"]
    n_epochs = max(1, experiment_config["train_steps"] // steps_per_epoch)
    total_env_steps = n_epochs * steps_per_epoch
    n_updates = total_env_steps * algo_config["updates_per_env_step"]
    warmup = experiment_config["warmup_steps"]

    return {
        "rollout": (total_env_steps, flops_key["rollout_per_env_step"] * total_env_steps, "MPPI planning call (encoder + planner + pi/Q), per env step"),
        "world_model_pretrain": (
            warmup, flops_key["gradient_update_total"] * warmup,
            "one-off burst of warmup_steps full agent.update() calls; logged as a single fused CodeCarbon task, not time-split",
        ),
        "buffer_sample": (n_updates, 0, "CPU-side sequence sampling, excluded from GPU FLOP accounting"),
        "critic_update": (n_updates, flops_key["gradient_update_critic"] * n_updates, "world-model step (consistency+reward+value losses), fwd+bwd"),
        "actor_update": (n_updates, flops_key["gradient_update_actor"] * n_updates, "policy-prior step (_update_pi), fwd+bwd"),
        "target_update": (
            n_updates, flops_key["gradient_update_target_elementwise_ops"] * n_updates,
            "Polyak averaging of Q-ensemble, elementwise ops (not matmul FLOPs)",
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed list to include in cross-seed averaging "
             f"(default: the canonical sweep {sorted(CANONICAL_SEEDS)}).",
    )
    p.add_argument(
        "--all-seeds", action="store_true",
        help="Include every run found in the cross-seed average, ignoring the canonical seed list.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.all_seeds:
        canonical_seeds = None  # None => no filtering, everything is "canonical"
    elif args.seeds:
        canonical_seeds = {int(s.strip()) for s in args.seeds.split(",") if s.strip()}
    else:
        canonical_seeds = CANONICAL_SEEDS
    if canonical_seeds is not None:
        print(f"Restricting cross-seed averaging to seeds: {sorted(canonical_seeds)} "
              f"(pass --all-seeds to include every run instead)")
    else:
        print("Including every run found in cross-seed averaging (--all-seeds)")

    os.makedirs(OUT_DIR, exist_ok=True)
    flops = load_flops()

    per_run_rows = []
    # cross-seed accumulator: (algo, env, architecture_signature, utd, rollout_regime, segment, flop_type)
    #   -> [(energy_kwh, total_flops, power_fields, extra_cols), ...]
    agg = defaultdict(list)

    for run_dir, meta_path, seg_path, tm_path in find_runs(RESULTS_DIR):
        with open(meta_path) as f:
            meta = json.load(f)
        with open(seg_path) as f:
            seg_energy = json.load(f)
        hw = load_segment_hw_energy(run_dir)
        seg_hw_raw = _segment_hw_raw(hw, seg_energy)

        algo = meta["algo_name"]
        env_id = meta["env_id"]
        seed = meta["seed"]
        algo_config = meta["algo_config"]
        experiment_config = meta["experiment_config"]
        include_in_avg = (canonical_seeds is None) or (seed in canonical_seeds)

        sig = signature(algo, algo_config)
        if algo not in flops or env_id not in flops[algo] or sig not in flops[algo][env_id]:
            print(f"WARNING: no flops_per_call.json entry for {algo}/{env_id}/{sig}; "
                  f"run measure_flops.py first (or it hasn't seen this architecture). Skipping {run_dir}")
            continue
        flops_key = flops[algo][env_id][sig]

        epochs_log = []
        extra_cols = {}  # segment -> extra CSV columns (MBPO's dynamics train/holdout FLOP split)
        if os.path.exists(tm_path):
            with open(tm_path) as f:
                epochs_log = json.load(f).get("epochs", [])

        if algo == "sac":
            rows = rows_for_sac_or_td3("sac", flops_key, algo_config, experiment_config)
        elif algo == "td3":
            rows = rows_for_sac_or_td3("td3", flops_key, algo_config, experiment_config)
        elif algo == "mbpo":
            rows, dynamics_split = rows_for_mbpo(flops_key, algo_config, experiment_config, epochs_log, run_dir)
            extra_cols["dynamics_model_update"] = dynamics_split
        elif algo == "tdmpc2":
            rows = rows_for_tdmpc2(flops_key, algo_config, experiment_config)
        else:
            print(f"WARNING: unknown algo {algo!r}, skipping {run_dir}")
            continue

        # also carry through the always-present bookkeeping segments if present
        for extra in ("warmup", "idle_baseline_head", "idle_baseline_tail"):
            if extra in seg_energy:
                rows.setdefault(extra, (None, None, "not part of FLOP accounting (see README 'Energy-per-FLOP analysis')"))

        # Any other segment with energy but no FLOP row still gets a row (and, if it
        # is a training segment, still counts toward the TOTAL numerator).
        for segment in seg_energy:
            if segment.startswith("_") or segment in rows:
                continue
            if segment not in TRAINING_SEGMENTS and segment not in EXCLUDED_FROM_TOTAL_SEGMENTS:
                print(f"WARNING: unexpected segment {segment!r} in {seg_path} (neither a training "
                      f"segment nor an excluded one) -- carried through, left out of the TOTAL")
            elif segment in TRAINING_SEGMENTS:
                print(f"WARNING: training segment {segment!r} in {seg_path} has no FLOP row for "
                      f"{algo} -- its energy counts toward the TOTAL, its FLOPs do not")
            rows[segment] = (None, None, "no FLOP accounting for this segment")

        # UTD (updates_per_env_step) changes call counts (hence Total_FLOPs) for every
        # gradient-related segment even at fixed architecture -- cross-seed averaging
        # must not mix runs at different UTD together, so it's part of the group key.
        utd = algo_config["updates_per_env_step"]

        # Same reasoning for MBPO's model-rollout-length schedule: it doesn't change
        # the architecture signature (per-call FLOP constants are rollout-length
        # invariant, see flop_keys.mbpo_rollout_regime), but it does change how many
        # synthetic_rollout_generation calls a run makes -- must not average runs
        # from different rollout-length regimes (e.g. the rollout1/rollout15 sweep)
        # together with each other or with the canonical Ant-v5 sweep (length up to 25).
        rollout_regime = mbpo_rollout_regime(algo_config) if algo == "mbpo" else None

        # run_dir is written with "/" regardless of OS so Windows/Linux regenerate identical CSVs
        run_rel = os.path.relpath(run_dir, REPO_ROOT).replace(os.sep, "/")
        row_ids = {
            "algo": algo, "env_id": env_id, "architecture_signature": sig, "updates_per_env_step": utd,
            "mbpo_rollout_regime": rollout_regime,
            "seed": seed, "run_dir": run_rel,
            "included_in_cross_seed_avg": include_in_avg,
        }

        total_flops = 0
        for segment, (call_count, total_flops_seg, note) in rows.items():
            energy_kwh = seg_energy.get(segment)
            if energy_kwh is None:
                continue
            energy_j = energy_kwh * KWH_TO_J
            if total_flops_seg:
                energy_per_flop = energy_j / total_flops_seg
            else:
                energy_per_flop = None
            seg_flop_type = flop_type(segment) if total_flops_seg is not None else "none"

            seg_raw = seg_hw_raw.get(segment)
            pw = power_fields(seg_raw) if seg_raw else dict(EMPTY_POWER_FIELDS)

            per_run_rows.append({
                **row_ids,
                "segment": segment, "flop_type": seg_flop_type,
                "call_count": call_count, "total_flops": total_flops_seg,
                "total_energy_kwh": energy_kwh, "total_energy_joules": energy_j,
                **pw,
                "energy_per_flop_j_per_flop": energy_per_flop, "note": note,
                **extra_cols.get(segment, {}),
            })

            if seg_flop_type == "matmul":
                total_flops += total_flops_seg

            if total_flops_seg is not None and include_in_avg:
                agg[(algo, env_id, sig, utd, rollout_regime, segment, seg_flop_type)].append(
                    (energy_kwh, total_flops_seg, pw, extra_cols.get(segment, {}))
                )

        # Numerator: iterate the fixed training-segment set against segment_energy.json
        # itself (not `rows`), so a training segment with energy but no FLOP row still counts.
        training_present = sorted(TRAINING_SEGMENTS & seg_energy.keys())
        if training_present:
            total_energy_kwh = sum(seg_energy[s] for s in training_present)
            total_energy_j = total_energy_kwh * KWH_TO_J
            # Hardware/power fields: the directly measured CodeCarbon tasks. The four
            # allocated sub-segments are represented by the measured gradient_updates
            # task as a whole -- their perf_counter durations don't cover the full task
            # duration, so summing them would bias mean power upward.
            total_hw_raw = {"duration_s": 0.0, "cpu_energy_kwh": 0.0, "gpu_energy_kwh": 0.0, "ram_energy_kwh": 0.0}
            hw_sources = {("gradient_updates" if s in ALLOCATED_SUB_SEGMENTS else s) for s in training_present}
            missing_hw = sorted(src for src in hw_sources if src not in hw)
            for src in sorted(hw_sources - set(missing_hw)):  # fixed order -> bit-identical reruns
                for k in total_hw_raw:
                    total_hw_raw[k] += hw[src][k]
            if missing_hw:
                print(f"WARNING: no per-task CodeCarbon rows for {missing_hw} in {run_dir}; "
                      f"TOTAL duration/power fields left empty")
                total_pw = dict(EMPTY_POWER_FIELDS)
            else:
                total_pw = power_fields(total_hw_raw)
            per_run_rows.append({
                **row_ids,
                "segment": TOTAL_SEGMENT, "flop_type": "mixed_total",
                "call_count": None, "total_flops": total_flops,
                "total_energy_kwh": total_energy_kwh, "total_energy_joules": total_energy_j,
                **total_pw,
                "energy_per_flop_j_per_flop": (total_energy_j / total_flops) if total_flops else None,
                "note": TOTAL_NOTE,
            })
            if include_in_avg:
                agg[(algo, env_id, sig, utd, rollout_regime, TOTAL_SEGMENT, "mixed_total")].append(
                    (total_energy_kwh, total_flops, total_pw, {})
                )

    per_run_csv = os.path.join(OUT_DIR, "per_run_energy_per_flop.csv")
    fieldnames = ["algo", "env_id", "architecture_signature", "updates_per_env_step", "mbpo_rollout_regime",
                  "seed", "run_dir",
                  "included_in_cross_seed_avg", "segment", "flop_type", "call_count", "total_flops",
                  "total_energy_kwh", "total_energy_joules",
                  "duration_s", "mean_power_w", "mean_cpu_power_w", "mean_gpu_power_w", "mean_ram_power_w",
                  "energy_per_flop_j_per_flop", "note",
                  "dynamics_train_flops", "dynamics_holdout_flops"]
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_run_rows)
    print(f"Wrote {per_run_csv} ({len(per_run_rows)} rows)")

    cross_seed_csv = os.path.join(OUT_DIR, "cross_seed_energy_per_flop.csv")
    with open(cross_seed_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algo", "env_id", "architecture_signature", "updates_per_env_step", "mbpo_rollout_regime",
            "segment", "flop_type", "n_seeds",
            "mean_energy_kwh", "mean_energy_joules",
            "mean_duration_s", "mean_power_w", "mean_cpu_power_w", "mean_gpu_power_w", "mean_ram_power_w",
            "total_flops", "mean_energy_per_flop_j_per_flop",
            "dynamics_train_flops", "dynamics_holdout_flops",
        ])
        w.writeheader()
        for (algo, env_id, sig, utd, rollout_regime, segment, seg_flop_type), vals in sorted(agg.items()):
            energies = [e for e, _, _, _ in vals]
            flop_vals = [fl for _, fl, _, _ in vals if fl]
            mean_energy_kwh = sum(energies) / len(energies)
            mean_flops = sum(flop_vals) / len(flop_vals) if flop_vals else 0
            mean_energy_j = mean_energy_kwh * KWH_TO_J
            w.writerow({
                "algo": algo, "env_id": env_id, "architecture_signature": sig, "updates_per_env_step": utd,
                "mbpo_rollout_regime": rollout_regime,
                "segment": segment, "flop_type": seg_flop_type, "n_seeds": len(vals),
                "mean_energy_kwh": mean_energy_kwh, "mean_energy_joules": mean_energy_j,
                "mean_duration_s": safe_mean([pw["duration_s"] for _, _, pw, _ in vals]),
                "mean_power_w": safe_mean([pw["mean_power_w"] for _, _, pw, _ in vals]),
                "mean_cpu_power_w": safe_mean([pw["mean_cpu_power_w"] for _, _, pw, _ in vals]),
                "mean_gpu_power_w": safe_mean([pw["mean_gpu_power_w"] for _, _, pw, _ in vals]),
                "mean_ram_power_w": safe_mean([pw["mean_ram_power_w"] for _, _, pw, _ in vals]),
                "total_flops": mean_flops,
                "mean_energy_per_flop_j_per_flop": (mean_energy_j / mean_flops) if mean_flops else None,
                "dynamics_train_flops": safe_mean([ex.get("dynamics_train_flops") for _, _, _, ex in vals]),
                "dynamics_holdout_flops": safe_mean([ex.get("dynamics_holdout_flops") for _, _, _, ex in vals]),
            })
    print(f"Wrote {cross_seed_csv} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
