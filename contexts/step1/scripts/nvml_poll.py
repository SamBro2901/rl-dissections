"""Task 2C (NOT RUN): standalone NVML energy-counter poller.

Samples nvmlDeviceGetTotalEnergyConsumption (mJ) at ~1 kHz and logs every *change* of the counter
with its timestamp, to measure how often the counter actually updates (the data suggest ~100 ms
lumps of ~6 J, see 02_subinterval_tasks.md). Run it in a separate process while a SAC baseline run
executes a few epochs on the same GPU, e.g.

  # terminal 1 (research PC, quiet machine, clocks locked as in the protocol)
  python contexts/step1/scripts/nvml_poll.py --seconds 120 --out nvml_poll_sac.csv
  # terminal 2, right after starting the poller
  sudo rl-exp/bin/python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 331 \
       --train-steps 5000 --warmup-steps 1000 --steps-per-epoch 1000 --output-dir results_poll_check

Output: CSV (t_s, counter_mJ, delta_mJ, dt_since_last_change_s) plus a printed histogram of the
spacing between counter updates. Use -I when running on untrusted input; this script reads none.
"""
import argparse, time
import numpy as np
import pynvml

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", default="nvml_poll.csv")
ap.add_argument("--poll-hz", type=float, default=1000)
a = ap.parse_args()

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(a.gpu)
period = 1.0 / a.poll_hz
t0 = time.perf_counter()
last = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
last_t = 0.0
rows = []
n_polls = 0
while True:
    now = time.perf_counter() - t0
    if now > a.seconds:
        break
    e = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    n_polls += 1
    if e != last:
        rows.append((now, e, e - last, now - last_t))
        last, last_t = e, now
    nxt = t0 + now + period
    while time.perf_counter() < nxt:
        pass
pynvml.nvmlShutdown()

with open(a.out, "w") as f:
    f.write("t_s,counter_mJ,delta_mJ,dt_since_last_change_s\n")
    for r in rows:
        f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")
dt = np.array([r[3] for r in rows[1:]])
print(f"polls={n_polls}, counter changes={len(rows)}")
if len(dt):
    print("spacing between counter updates (s): min %.4f p5 %.4f median %.4f p95 %.4f max %.4f" %
          (dt.min(), np.percentile(dt, 5), np.median(dt), np.percentile(dt, 95), dt.max()))
    hist, edges = np.histogram(dt, bins=[0, .005, .01, .02, .05, .08, .09, .1, .11, .12, .2, .5, 1, 10])
    for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"  [{lo:.3f}, {hi:.3f}) s: {c}")
