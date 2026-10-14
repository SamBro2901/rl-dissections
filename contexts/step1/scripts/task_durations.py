"""Task 2A: per-task duration / energy statistics from emissions_base_*.csv (read-only).
Run: .venv/Scripts/python.exe contexts/step1/scripts/task_durations.py
Writes 02_task_duration_stats.csv (+ 02_task_empirical_checks.json used by 02_subinterval_tasks.md).
Span = ram_energy[J]/ram_power[W] (same proxy as Task 1)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import canonical_runs, read_base_csv, OUT

J = 3.6e6
runs = canonical_runs()
frames = []
for _, r in runs.iterrows():
    b = read_base_csv(r.run_dir).copy()
    b["prefix"] = b.task_name.str.replace(r"_\d+$", "", regex=True)
    b["algo"], b["env"], b["seed"], b["config"], b["run_dir"] = r.algo, r.env, r.seed, r.config, r.run_dir
    frames.append(b[["task_name", "prefix", "algo", "env", "seed", "config", "run_dir", "timestamp", "duration",
                     "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed", "ram_power"]])
T = pd.concat(frames, ignore_index=True)
T["span"] = T.ram_energy * J / T.ram_power
T["span_rel_dev"] = (T.span - T.duration).abs() / T.duration
T["cpu_j"], T["gpu_j"], T["ram_j"], T["tot_j"] = (T.cpu_energy * J, T.gpu_energy * J, T.ram_energy * J, T.energy_consumed * J)
T["power_w"] = T.tot_j / T.duration
T["gpu_w"] = T.gpu_j / T.duration
T["cpu_w"] = T.cpu_j / T.duration
T = T[~T.prefix.isin(["idle_baseline_head", "idle_baseline_tail"])]


def summ(g):
    d = g.duration
    return pd.Series(dict(
        n_tasks=len(g), dur_min=d.min(), dur_p5=d.quantile(.05), dur_median=d.median(), dur_p95=d.quantile(.95), dur_max=d.max(),
        frac_lt_1s=(d < 1).mean(), frac_lt_2s=(d < 2).mean(), n_lt_1s=int((d < 1).sum()),
        span_dev_median=g.span_rel_dev.median(), span_dev_max=g.span_rel_dev.max(), frac_span_dev_gt_10pct=(g.span_rel_dev > .10).mean(),
        frac_zero_gpu=(g.gpu_energy == 0).mean(), frac_zero_cpu=(g.cpu_energy == 0).mean(), frac_zero_total=(g.energy_consumed == 0).mean(),
        n_zero_gpu=int((g.gpu_energy == 0).sum()),
        median_total_w=g.power_w.median(), median_gpu_w=g.gpu_w.median(), median_cpu_w=g.cpu_w.median(),
        min_total_w=g.power_w.min(), max_total_w=g.power_w.max(), max_gpu_w=g.gpu_w.max()))


rows = []
for scope, sub in (("all_configs", T), ("baseline_only", T[T.config == "baseline"])):
    for (a, e, p), g in sub.groupby(["algo", "env", "prefix"]):
        rows.append({"scope": scope, "algo": a, "env": e, "segment": p, **summ(g).to_dict()})
    for (a, p), g in sub.groupby(["algo", "prefix"]):
        rows.append({"scope": scope, "algo": a, "env": "POOLED", "segment": p, **summ(g).to_dict()})
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "02_task_duration_stats.csv"), index=False)

# --- cross-checks ---
chk = {}
base = T[T.config == "baseline"]
for a in ("sac", "td3"):
    g = base[(base.algo == a) & (base.prefix == "rollout")]
    chk[f"{a}_rollout_baseline_median_pooled_envs"] = g.duration.median()
    for e in ("HalfCheetah-v5", "Ant-v5"):
        chk[f"{a}_rollout_baseline_median_{e}"] = g[g.env == e].duration.median()
    gall = T[(T.algo == a) & (T.prefix == "rollout")]
    chk[f"{a}_rollout_allconfigs_median_pooled"] = gall.duration.median()
for a in ("sac", "mbpo", "td3", "tdmpc2"):
    g = T[(T.algo == a) & (T.prefix == "rollout")]
    chk[f"{a}_rollout_all_configs_n"] = len(g)
    chk[f"{a}_rollout_all_configs_n_lt_1s"] = int((g.duration < 1).sum())
    chk[f"{a}_rollout_all_configs_max_dur"] = g.duration.max()
m = T[T.algo == "mbpo"]
for p in ("rollout", "synthetic_rollout_generation", "dynamics_model_update", "gradient_updates"):
    g = m[m.prefix == p]
    chk[f"mbpo_{p}_n"] = len(g); chk[f"mbpo_{p}_n_lt_1s"] = int((g.duration < 1).sum())
    gb = g[g.config == "baseline"]
    chk[f"mbpo_{p}_baseline_n"] = len(gb); chk[f"mbpo_{p}_baseline_n_zero_gpu"] = int((gb.gpu_energy == 0).sum())
    chk[f"mbpo_{p}_baseline_n_lt_1s"] = int((gb.duration < 1).sum())
chk["mbpo_n_configs"] = int(runs[runs.algo == "mbpo"].groupby(["env", "config"]).ngroups)
chk["mbpo_runs"] = int((runs.algo == "mbpo").sum())

# quantisation of GPU energy on sub-second tasks
sub1 = T[(T.duration < 1) & (T.gpu_energy > 0)]
gj = (sub1.gpu_j * 1000).round(3)  # mJ
chk["subsecond_gpu_tasks_nonzero"] = len(sub1)
chk["subsecond_gpu_mJ_distinct_values"] = int(gj.nunique())
chk["subsecond_gpu_mJ_smallest_10_distinct"] = sorted(gj.unique().tolist())[:10]
diffs = np.diff(np.sort(gj.unique()))
chk["subsecond_gpu_mJ_min_gap_between_distinct_values"] = float(diffs.min()) if len(diffs) else None
# is gpu energy a multiple of a base step (mJ granularity of NVML counter)?
chk["subsecond_gpu_mJ_fraction_integer"] = float((np.abs(gj - gj.round()) < 1e-3).mean())
# short-task power plausibility
short = T[(T.duration < 1)]
chk["subsecond_tasks_n"] = len(short)
chk["subsecond_total_w_quantiles_p1_p5_p50_p95_p99"] = short.power_w.quantile([.01, .05, .5, .95, .99]).tolist()
chk["subsecond_gpu_w_quantiles_p1_p5_p50_p95_p99_max"] = short.gpu_w.quantile([.01, .05, .5, .95, .99, 1.0]).tolist()
chk["long_tasks_ge2s_gpu_w_quantiles_p1_p5_p50_p95_p99_max"] = T[T.duration >= 2].gpu_w.quantile([.01, .05, .5, .95, .99, 1.0]).tolist()
# span-dev as function of task length
for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 10), (10, 1e9)):
    g = T[(T.duration >= lo) & (T.duration < hi)]
    chk[f"span_minus_duration_s_median_dur_{lo}_{hi}"] = float((g.span - g.duration).median()) if len(g) else None
    chk[f"n_tasks_dur_{lo}_{hi}"] = len(g)

# sum of task energies vs whole-run cumulative energy (emissions.csv)
cmp_rows = []
for _, r in runs.iterrows():
    b = read_base_csv(r.run_dir)
    whole = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", r.run_dir, "emissions.csv")).iloc[-1]
    cmp_rows.append(dict(algo=r.algo, env=r.env, config=r.config, run_dir=r.run_dir,
                         sum_tasks_kwh=b.energy_consumed.sum(), whole_run_kwh=float(whole.energy_consumed),
                         sum_gpu=b.gpu_energy.sum(), whole_gpu=float(whole.gpu_energy),
                         sum_cpu=b.cpu_energy.sum(), whole_cpu=float(whole.cpu_energy),
                         sum_ram=b.ram_energy.sum(), whole_ram=float(whole.ram_energy),
                         n_tasks=len(b)))
C = pd.DataFrame(cmp_rows)
C["diff_kwh"] = C.whole_run_kwh - C.sum_tasks_kwh
C["diff_pct"] = 100 * C.diff_kwh / C.whole_run_kwh
for c in ("gpu", "cpu", "ram"):
    C[f"diff_{c}_pct"] = 100 * (C[f"whole_{c}"] - C[f"sum_{c}"]) / C[f"whole_{c}"]
C.to_csv(os.path.join(OUT, "02_sum_tasks_vs_whole_run.csv"), index=False)
chk["whole_vs_sum_by_algo"] = {f"{a}|{k[0]}|{k[1]}": v for k, d in C.groupby("algo")[["diff_pct", "diff_gpu_pct", "diff_cpu_pct", "diff_ram_pct"]].agg(["median", "min", "max"]).round(4).items() for a, v in d.items()}
chk["whole_vs_sum_overall_diff_pct_median_min_max"] = [C.diff_pct.median(), C.diff_pct.min(), C.diff_pct.max()]
chk["whole_vs_sum_per_task_gap_s_equiv_median"] = None
json.dump(chk, open(os.path.join(OUT, "02_task_empirical_checks.json"), "w"), indent=1, default=str)
print(json.dumps(chk, indent=1, default=str))
