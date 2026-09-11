"""
Step 2+3 of the FLOP-per-segment methodology (see flop_calculation_methodology.md).

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
     authoritative per-segment Energy(segment, run) in kWh -- it is already
     exactly what the methodology doc's Step 2/3 call "Energy(segment, run)".
  3. Reads training_metrics.json for the few call counts that are genuinely
     data-dependent rather than derivable from config alone: MBPO's
     model_train_epochs (early-stopping is holdout-MSE-dependent) and
     synthetic_transitions_generated (termination-dependent rollout length).
  4. Joins against flops_per_call.json (produced by measure_flops.py) to get
     Total_FLOPs(segment, run), then Energy_per_FLOP = Energy_J / Total_FLOPs.

Output:
  flop_analysis/output/per_run_energy_per_flop.csv       -- one row per (run, segment); every run,
                                                              tagged with included_in_cross_seed_avg
  flop_analysis/output/cross_seed_energy_per_flop.csv    -- averaged across seeds per (algo, env,
                                                              architecture, UTD, segment)

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
from collections import defaultdict

from flop_keys import signature

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FLOPS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flops_per_call.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

KWH_TO_J = 3.6e6

# The 5-seed sweep that recurs across every algo/env in results/ (see docstring above).
CANONICAL_SEEDS = {331, 958, 14577, 43611, 85062}

# Segments excluded from the primary energy accounting (matches dashboard.py's
# NON_TRAINING_PHASES): idle calibration windows aren't algorithmic compute.
# "warmup" (random-action buffer fill) is kept in the per-segment table for
# completeness but excluded from the TOTAL_MEASURED_TRAINING aggregate row.
NON_TRAINING_SEGMENTS = {"idle_baseline_head", "idle_baseline_tail"}


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


def rows_for_mbpo(flops_key, algo_config, experiment_config, epochs_log):
    fc = sac_like_call_counts(algo_config, experiment_config)
    rows = rows_for_sac_or_td3("sac", flops_key, algo_config, experiment_config)

    warmup = experiment_config["warmup_steps"]
    steps_per_epoch = experiment_config["steps_per_epoch"]
    holdout_ratio = algo_config["model_holdout_ratio"]
    train_batch = algo_config["model_train_batch_size"]
    ensemble_size = algo_config["ensemble_size"]
    member_fwdbwd = flops_key["dynamics_member_fwdbwd"]

    dyn_flops = 0
    dyn_calls = 0
    for row in epochs_log:
        epoch = row["epoch"]
        n_total = warmup + epoch * steps_per_epoch  # real buffer size just before this epoch's fit() call
        n_holdout = max(1, int(n_total * holdout_ratio))
        n_train = n_total - n_holdout
        n_batches = ceil_div(n_train, train_batch)
        epochs_run = row.get("model_train_epochs") or 0
        dyn_calls += n_batches * epochs_run
        dyn_flops += ensemble_size * member_fwdbwd * n_batches * epochs_run
    rows["dynamics_model_update"] = (dyn_calls, dyn_flops, "ensemble_size x member fwd+bwd x batches x epochs_run (epochs_run from training_metrics.json, early-stopping is data-dependent)")

    per_sample = flops_key["actor_forward_bs1"] + flops_key["dynamics_ensemble_forward_all_bs1"]
    total_samples = sum(row.get("synthetic_transitions_generated", 0) for row in epochs_log)
    rows["synthetic_rollout_generation"] = (
        total_samples, total_samples * per_sample,
        "actor forward + full-ensemble forward, per synthetic sample (all members scored every predict() call)",
    )
    return rows


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
    # cross-seed accumulator: (algo, env, architecture_signature, utd, segment) -> [(energy_kwh, total_flops), ...]
    agg = defaultdict(list)

    for run_dir, meta_path, seg_path, tm_path in find_runs(RESULTS_DIR):
        with open(meta_path) as f:
            meta = json.load(f)
        with open(seg_path) as f:
            seg_energy = json.load(f)

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
        if os.path.exists(tm_path):
            with open(tm_path) as f:
                epochs_log = json.load(f).get("epochs", [])

        if algo == "sac":
            rows = rows_for_sac_or_td3("sac", flops_key, algo_config, experiment_config)
        elif algo == "td3":
            rows = rows_for_sac_or_td3("td3", flops_key, algo_config, experiment_config)
        elif algo == "mbpo":
            rows = rows_for_mbpo(flops_key, algo_config, experiment_config, epochs_log)
        elif algo == "tdmpc2":
            rows = rows_for_tdmpc2(flops_key, algo_config, experiment_config)
        else:
            print(f"WARNING: unknown algo {algo!r}, skipping {run_dir}")
            continue

        # also carry through the always-present bookkeeping segments if present
        for extra in ("warmup", "idle_baseline_head", "idle_baseline_tail"):
            if extra in seg_energy:
                rows.setdefault(extra, (None, None, "not part of FLOP accounting (see methodology doc caveats)"))

        # UTD (updates_per_env_step) changes call counts (hence Total_FLOPs) for every
        # gradient-related segment even at fixed architecture -- cross-seed averaging
        # must not mix runs at different UTD together, so it's part of the group key.
        utd = algo_config["updates_per_env_step"]

        total_energy_kwh = 0.0
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

            per_run_rows.append({
                "algo": algo, "env_id": env_id, "architecture_signature": sig, "updates_per_env_step": utd,
                "seed": seed, "run_dir": os.path.relpath(run_dir, REPO_ROOT),
                "included_in_cross_seed_avg": include_in_avg,
                "segment": segment, "call_count": call_count, "total_flops": total_flops_seg,
                "total_energy_kwh": energy_kwh, "total_energy_joules": energy_j,
                "energy_per_flop_j_per_flop": energy_per_flop, "note": note,
            })

            if segment not in NON_TRAINING_SEGMENTS and segment != "warmup" and total_flops_seg:
                total_energy_kwh += energy_kwh
                total_flops += total_flops_seg

            if total_flops_seg is not None and include_in_avg:
                agg[(algo, env_id, sig, utd, segment)].append((energy_kwh, total_flops_seg))

        if total_flops:
            total_energy_j = total_energy_kwh * KWH_TO_J
            per_run_rows.append({
                "algo": algo, "env_id": env_id, "architecture_signature": sig, "updates_per_env_step": utd,
                "seed": seed, "run_dir": os.path.relpath(run_dir, REPO_ROOT),
                "included_in_cross_seed_avg": include_in_avg,
                "segment": "TOTAL_MEASURED_TRAINING", "call_count": None, "total_flops": total_flops,
                "total_energy_kwh": total_energy_kwh, "total_energy_joules": total_energy_j,
                "energy_per_flop_j_per_flop": total_energy_j / total_flops,
                "note": "sum over matmul-FLOP-accounted segments only (excludes idle baselines, warmup, buffer_sample, target_update)",
            })
            if include_in_avg:
                agg[(algo, env_id, sig, utd, "TOTAL_MEASURED_TRAINING")].append((total_energy_kwh, total_flops))

    per_run_csv = os.path.join(OUT_DIR, "per_run_energy_per_flop.csv")
    fieldnames = ["algo", "env_id", "architecture_signature", "updates_per_env_step", "seed", "run_dir",
                  "included_in_cross_seed_avg", "segment", "call_count", "total_flops",
                  "total_energy_kwh", "total_energy_joules", "energy_per_flop_j_per_flop", "note"]
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_run_rows)
    print(f"Wrote {per_run_csv} ({len(per_run_rows)} rows)")

    cross_seed_csv = os.path.join(OUT_DIR, "cross_seed_energy_per_flop.csv")
    with open(cross_seed_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algo", "env_id", "architecture_signature", "updates_per_env_step", "segment", "n_seeds",
            "mean_energy_kwh", "mean_energy_joules", "total_flops", "mean_energy_per_flop_j_per_flop",
        ])
        w.writeheader()
        for (algo, env_id, sig, utd, segment), vals in sorted(agg.items()):
            energies = [e for e, _ in vals]
            flop_vals = [fl for _, fl in vals if fl]
            mean_energy_kwh = sum(energies) / len(energies)
            mean_flops = sum(flop_vals) / len(flop_vals) if flop_vals else 0
            mean_energy_j = mean_energy_kwh * KWH_TO_J
            w.writerow({
                "algo": algo, "env_id": env_id, "architecture_signature": sig, "updates_per_env_step": utd,
                "segment": segment, "n_seeds": len(vals),
                "mean_energy_kwh": mean_energy_kwh, "mean_energy_joules": mean_energy_j,
                "total_flops": mean_flops,
                "mean_energy_per_flop_j_per_flop": (mean_energy_j / mean_flops) if mean_flops else None,
            })
    print(f"Wrote {cross_seed_csv} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
