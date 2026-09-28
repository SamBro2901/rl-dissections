"""
Consistency checks on flop_analysis/output/per_run_energy_per_flop.csv's
TOTAL_MEASURED_TRAINING rows, against the raw run data under results/.
Exits non-zero on any failure.

  1. TOTAL total_energy_kwh == sum of the training-segment energies read
     directly from segment_energy.json (relative error <= 1e-12).
  2. TOTAL total_flops == sum of total_flops over that run's per-segment rows
     whose flop_type == "matmul" (exact integer equality).
  3. buffer_sample + critic_update + actor_update + target_update energy in
     segment_energy.json == the directly measured gradient_updates energy in
     the run's per-task CodeCarbon CSV (a consistency check on the time-share
     allocation, not new logic).

Plus coverage: every run under results/ with a segment_energy.json has
exactly one TOTAL row.

The training-segment definition below is deliberately restated here rather
than imported from compute_energy_per_flop.py, so the check is independent
of the code it checks (and then asserted equal to it, so the two can't drift).

Usage (from repo root):
    python flop_analysis/check_totals.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_energy_per_flop as cepf  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PER_RUN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "per_run_energy_per_flop.csv")

TRAINING_SEGMENTS = {
    "rollout", "buffer_sample", "critic_update", "actor_update", "target_update",
    "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain",
}
EXCLUDED_SEGMENTS = {"idle_baseline_head", "idle_baseline_tail", "warmup"}
SUB_SEGMENTS = ("buffer_sample", "critic_update", "actor_update", "target_update")

ENERGY_RTOL = 1e-12
ALLOC_RTOL = 1e-9  # check 3: CSV-parsed floats summed over ~100 epoch rows vs. TrackerTask's in-process sum


def rel_err(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)


def gradient_updates_energy_from_task_csv(run_dir):
    csvs = sorted(glob.glob(os.path.join(run_dir, "emissions_*.csv")))
    if not csvs:
        return None
    total, found = 0.0, False
    with open(csvs[0], newline="") as f:
        for row in csv.DictReader(f):
            if re.fullmatch(r"gradient_updates_\d+", row.get("task_name") or ""):
                total += float(row["energy_consumed"])
                found = True
    return total if found else None


def main():
    failures = []
    stats = defaultdict(int)
    max_err = {"check1": 0.0, "check3": 0.0}

    if cepf.TRAINING_SEGMENTS != TRAINING_SEGMENTS or cepf.EXCLUDED_FROM_TOTAL_SEGMENTS != EXCLUDED_SEGMENTS \
            or cepf.ELEMENTWISE_OP_SEGMENTS != {"target_update"}:
        failures.append("compute_energy_per_flop.py's segment sets differ from the definition restated here")

    rows_by_run = defaultdict(list)
    with open(PER_RUN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows_by_run[row["run_dir"]].append(row)

    run_dirs = sorted(os.path.dirname(p) for p in glob.glob(
        os.path.join(RESULTS_DIR, "*", "*", "seed_*", "*", "segment_energy.json")))

    for run_dir in run_dirs:
        rel = os.path.relpath(run_dir, REPO_ROOT).replace(os.sep, "/")
        stats["runs"] += 1
        with open(os.path.join(run_dir, "segment_energy.json")) as f:
            seg_energy = json.load(f)

        # ---- check 3 (independent of the CSV) ----
        if all(s in seg_energy for s in SUB_SEGMENTS):
            measured = gradient_updates_energy_from_task_csv(run_dir)
            if measured is None:
                failures.append(f"[check3] {rel}: no gradient_updates_N rows in per-task CSV")
            else:
                allocated = sum(seg_energy[s] for s in SUB_SEGMENTS)
                e = rel_err(allocated, measured)
                max_err["check3"] = max(max_err["check3"], e)
                stats["check3_runs"] += 1
                if e > ALLOC_RTOL:
                    failures.append(f"[check3] {rel}: sub-segments sum {allocated!r} != measured "
                                    f"gradient_updates {measured!r} (rel err {e:.3e})")

        unexpected = [k for k in seg_energy if not k.startswith("_")
                      and k not in TRAINING_SEGMENTS and k not in EXCLUDED_SEGMENTS]
        if unexpected:
            failures.append(f"[coverage] {rel}: unexpected segment(s) {unexpected} in segment_energy.json")

        rows = rows_by_run.get(rel)
        if not rows:
            failures.append(f"[coverage] {rel}: no rows in per_run_energy_per_flop.csv")
            continue
        totals = [r for r in rows if r["segment"] == "TOTAL_MEASURED_TRAINING"]
        if len(totals) != 1:
            failures.append(f"[coverage] {rel}: expected 1 TOTAL row, found {len(totals)}")
            continue
        total = totals[0]
        if total["flop_type"] != "mixed_total":
            failures.append(f"[coverage] {rel}: TOTAL row flop_type={total['flop_type']!r}")

        # ---- check 1 ----
        expected_e = sum(v for k, v in seg_energy.items() if k in TRAINING_SEGMENTS)
        got_e = float(total["total_energy_kwh"])
        e = rel_err(got_e, expected_e)
        max_err["check1"] = max(max_err["check1"], e)
        stats["check1_runs"] += 1
        if e > ENERGY_RTOL:
            failures.append(f"[check1] {rel}: TOTAL energy {got_e!r} != {expected_e!r} (rel err {e:.3e})")

        # ---- check 2 (exact integers, parsed with int() -- FLOP totals exceed 2**53) ----
        matmul_sum = sum(int(r["total_flops"]) for r in rows if r["flop_type"] == "matmul")
        got_f = int(total["total_flops"])
        stats["check2_runs"] += 1
        if got_f != matmul_sum:
            failures.append(f"[check2] {rel}: TOTAL flops {got_f} != sum of matmul rows {matmul_sum}")
        # and no non-matmul segment may carry FLOPs into the total by another route
        for r in rows:
            if r["segment"] == "target_update" and r["flop_type"] != "elementwise":
                failures.append(f"[check2] {rel}: target_update flop_type={r['flop_type']!r}")

    orphan = set(rows_by_run) - {os.path.relpath(d, REPO_ROOT).replace(os.sep, "/") for d in run_dirs}
    if orphan:
        failures.append(f"[coverage] {len(orphan)} CSV run_dir(s) not found under results/: {sorted(orphan)[:3]} ...")

    print(f"Runs under results/ with segment_energy.json: {stats['runs']}")
    print(f"Check 1 (TOTAL energy == sum of training segments, rtol {ENERGY_RTOL:g}): "
          f"{stats['check1_runs']} runs, max rel err {max_err['check1']:.3e}")
    print(f"Check 2 (TOTAL flops == sum of matmul rows, exact): {stats['check2_runs']} runs")
    print(f"Check 3 (sub-segments == measured gradient_updates, rtol {ALLOC_RTOL:g}): "
          f"{stats['check3_runs']} runs, max rel err {max_err['check3']:.3e}")
    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        for msg in failures[:50]:
            print("  " + msg)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
