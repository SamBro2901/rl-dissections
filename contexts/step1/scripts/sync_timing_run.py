"""Task 2C (NOT RUN): baseline run with torch.cuda.synchronize() before every perf_counter()
call inside the algorithm modules, so the sub-phase timers measure finished GPU work rather than
kernel-launch time. Existing repo code is NOT modified: the `time` name inside algorithms.* is
replaced by a shim at import time, then run_experiment.main() is called unchanged.

Needs the research PC (GPU, MuJoCo), a quiet machine, and the same protocol as the canonical runs
(sudo for clock lock / governor). One baseline run per algorithm, e.g.:

  sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo sac --env HalfCheetah-v5 \
       --seed 331 --output-dir results_sync_check
  sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo td3 --env HalfCheetah-v5 \
       --seed 331 --warmup-steps 10000 --output-dir results_sync_check
  sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo mbpo --env HalfCheetah-v5 \
       --seed 331 --output-dir results_sync_check
  sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo tdmpc2 --env HalfCheetah-v5 \
       --seed 331 --output-dir results_sync_check

(Use --output-dir results_sync_check so the canonical results/ directory is untouched.)
After the run it prints  ratio = sum(sub-phase wall times) / sum(measured gradient_updates durations)
and the same ratio for the matching canonical baseline run (pass --compare-dir to a canonical run dir).
"""
import glob, json, os, sys, time as _real_time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch


class _SyncTime:
    """Drop-in for the `time` module as used by algorithms/*.py (only perf_counter/sleep/time used)."""
    def __getattr__(self, name):
        return getattr(_real_time, name)

    @staticmethod
    def perf_counter():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return _real_time.perf_counter()


import importlib
for mod in ("algorithms.sac", "algorithms.td3", "algorithms.mbpo", "algorithms.tdmpc2"):
    m = importlib.import_module(mod)
    if hasattr(m, "time"):
        m.time = _SyncTime()


def ratio(run_dir):
    import pandas as pd
    seg = json.load(open(os.path.join(run_dir, "segment_energy.json")))
    sub = seg.get("_sub_segment_wall_time_seconds", {})
    base = pd.read_csv(glob.glob(os.path.join(run_dir, "emissions_base_*.csv"))[0])
    gu = base[base.task_name.str.startswith("gradient_updates_")].duration.sum()
    # sub-segments belonging to gradient_updates only (exclude MBPO dynamics/synthetic extras)
    s = sum(v for k, v in sub.items() if k in ("buffer_sample", "critic_update", "actor_update", "target_update"))
    return dict(sum_sub_phase_s=s, measured_gradient_updates_s=gu, ratio=s / gu if gu else None, sub=sub)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--compare-dir", default=None)
    known, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    import run_experiment
    outdir = "results"
    for i, a in enumerate(rest):
        if a == "--output-dir":
            outdir = rest[i + 1]
    before = set(glob.glob(os.path.join(outdir, "*", "*", "seed_*", "*")))
    run_experiment.main()
    new = sorted(set(glob.glob(os.path.join(outdir, "*", "*", "seed_*", "*"))) - before)
    for d in new:
        print("SYNC RUN", d, json.dumps(ratio(d), indent=1))
        if known.compare_dir:
            print("CANONICAL", known.compare_dir, json.dumps(ratio(known.compare_dir), indent=1))
