"""B5 (NOT RUN): CodeCarbon task overhead control, tracker built exactly as in experiment_runner.py.

Runs N repetitions of start_task / sleep(d) / stop_task for two idle durations and reports the
energy per task (J), plus the time-aligned head-window comparison. Idle only: no training, no GPU load.
Must be run with sudo so RAPL is readable (same as the canonical runs). Writes only to --output-dir.
Usage: sudo rl-exp/bin/python contexts/step9/scratch/overhead_control.py --output-dir /tmp/overhead_check
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from configs.config import ExperimentConfig
from codecarbon import EmissionsTracker

ap = argparse.ArgumentParser()
ap.add_argument("--output-dir", required=True)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--durations", type=float, nargs="+", default=[4.0, 0.2])
ap.add_argument("--settle-seconds", type=float, default=30.0)
a = ap.parse_args()
os.makedirs(a.output_dir, exist_ok=True)
cfg = ExperimentConfig(algo_name="overhead_control", env_id="none")   # required positional fields, unused here; tracker fields keep the harness defaults
tracker = EmissionsTracker(
    project_name="overhead_control", measure_power_secs=cfg.measure_power_secs, tracking_mode=cfg.tracking_mode,
    output_dir=a.output_dir, output_file="emissions.csv", save_to_file=True, log_level="warning",
    allow_multiple_runs=True, force_cpu_power=cfg.force_cpu_power_w)
tracker.start()
time.sleep(a.settle_seconds)
res = {}
for d in a.durations:
    e_j, wall = [], []
    for i in range(a.reps):
        name = f"empty_{d}_{i}"
        t0 = time.perf_counter()
        tracker.start_task(name)
        time.sleep(d)
        data = tracker.stop_task(name)
        wall.append(time.perf_counter() - t0)
        e_j.append(data.energy_consumed * 3.6e6)   # kWh -> J
    res[str(d)] = dict(mean_J=float(np.mean(e_j)), sd_J=float(np.std(e_j, ddof=1)), mean_wall_s=float(np.mean(wall)),
                       mean_power_W=float(np.mean(e_j) / np.mean(wall)), n=len(e_j), energies_J=e_j)
    print(f"duration {d}s: n={len(e_j)} energy per task mean {np.mean(e_j):.3f} J sd {np.std(e_j, ddof=1):.3f} J; mean wall {np.mean(wall):.3f} s")
tracker.stop()
json.dump(res, open(os.path.join(a.output_dir, "overhead_control.json"), "w"), indent=1)
print("wrote", os.path.join(a.output_dir, "overhead_control.json"))
