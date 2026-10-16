"""B4: time _EpisodeSequenceBuffer.sample() vs number of stored episodes. CPU only, no tracker."""
import sys, json, time, glob
import numpy as np
sys.path.insert(0, "/home/samanth/rl-dissection-linux")
from algorithms.tdmpc2 import _EpisodeSequenceBuffer

BATCH, HORIZON, CALLS, WARM = 256, 3, 200, 10   # batch/horizon = TDMPC2Config baseline (checked in metadata.json)
rng = np.random.default_rng(0)
np.random.seed(0)

def fill(lengths, obs_dim, act_dim):
    buf = _EpisodeSequenceBuffer(10**9, obs_dim, act_dim, HORIZON)
    for L in lengths:
        buf.start_episode(rng.standard_normal(obs_dim))
        # build episode directly (equivalent to add() L times; avoids slow python loop)
        ep = {"obs": rng.standard_normal((L + 1, obs_dim)).astype(np.float32),
              "actions": rng.standard_normal((L, act_dim)).astype(np.float32),
              "rewards": rng.standard_normal((L, 1)).astype(np.float32),
              "terminateds": np.zeros((L, 1), np.float32), "length": L}
        buf._episodes.append(ep); buf._total_transitions += L
    return buf

def profile(name, lengths, obs_dim, act_dim):
    buf = fill(lengths, obs_dim, act_dim)
    for _ in range(WARM): buf.sample(BATCH, HORIZON)
    t = []
    for _ in range(CALLS):
        t0 = time.perf_counter(); buf.sample(BATCH, HORIZON); t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    row = dict(name=name, episodes=len(lengths), transitions=int(sum(lengths)), mean_ms=t.mean(), sd_ms=t.std(ddof=1))
    print("%-28s episodes=%5d transitions=%8d  mean=%.4f ms  sd=%.4f ms (n=%d calls)" % (name, row["episodes"], row["transitions"], row["mean_ms"], row["sd_ms"], CALLS), flush=True)
    return row

# real Ant-v5 episode-length pool from the canonical TD-MPC2 baseline run (seed 331)
p = sorted(glob.glob("results/tdmpc2/Ant-v5/seed_331/*/training_metrics.json"))[0]
pool = [e["length"] for e in json.load(open(p))["episodes"]]
print("length pool:", p, "n=%d" % len(pool), "mean=%.1f median=%.1f max=%d" % (np.mean(pool), np.median(pool), max(pool)))
rows = []
for n in (105, 478, 923, 2587):                      # Ant-like (obs 105, act 8), lengths resampled from the real pool
    rows.append(profile("Ant-like pool lengths", rng.choice(pool, n).tolist(), 105, 8))
for n in (105, 478, 2587):                           # controlled: fixed length 100, only episode count varies
    rows.append(profile("Ant-like fixed length 100", [100] * n, 105, 8))
rows.append(profile("HalfCheetah-like 105x1000", [1000] * 105, 17, 6))
# linear fits (time vs episode count)
for name in ("Ant-like pool lengths", "Ant-like fixed length 100"):
    r = [x for x in rows if x["name"] == name]
    x = np.array([a["episodes"] for a in r], float); y = np.array([a["mean_ms"] for a in r])
    b, a = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
    print("linear fit %-28s time_ms = %.4f + %.6f * episodes  (R^2=%.4f, points=%d)" % (name, a, b, r2, len(x)))
