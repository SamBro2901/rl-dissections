"""Read-only checks for Chapter 7 (Step 7, Evaluation). Prints everything to stdout, writes no file.

Reads the run files of results/*/*/seed_*/<timestamp>/ (metadata.json, emissions_base_*.csv, segment_energy.json,
training_metrics.json) and flop_analysis/output/per_run_energy_per_flop.csv (run -> configuration map and FLOP counts).
Usage: python flop_analysis/section_7_checks.py > out.txt
Sections (named in the output): 0 environment, 1 task rows / zero GPU energy / RAM, 2 rollout tasks, 3 stationarity and
MBPO growth, 4 idle drift, 5 tab:thr-gross-net, 6 net energy of all 280 runs, 7 order and time confounding / sessions,
8 learning check, 9 provenance, 10 sessions (in 7), 11 development runs, 12 numbers for the design limits.
"""
import glob
import json
import math
import re
import subprocess
import sys
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "flop_analysis" / "output"
J = 3.6e6
CANON = [331, 958, 14577, 43611, 85062]
ENVS = ["HalfCheetah-v5", "Ant-v5"]
HC, ANT = ENVS
ALGOS = ["sac", "td3", "mbpo", "tdmpc2"]
MEAS = ["rollout", "gradient_updates", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain"]
IDLE = ["idle_baseline_head", "idle_baseline_tail"]
BASE_BATCH = {"sac": 256, "mbpo": 256, "td3": 100}
VARIANTS = ["gross", "net_mean", "net_head", "net_tail"]

warnings.simplefilter("ignore", FutureWarning)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)


def head(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


def parse(row):
    """Configuration label of a run (copy of flop_analysis/section_5_2_analysis.parse)."""
    sig = row.architecture_signature
    algo = row.algo
    if algo == "tdmpc2":
        h = int(re.search(r"_h(\d+)_ns", sig).group(1))
        nq = int(re.search(r"_nq(\d+)", sig).group(1))
        return "base" if (h == 3 and nq == 5) else (f"numq{nq}" if nq != 5 else f"horizon{h}")
    batch = int(re.match(r"bs(\d+)", sig).group(1))
    width = int(re.search(r"_h(\d+)x", sig).group(1))
    utd = int(row.updates_per_env_step)
    parts = []
    if width != 1024:
        parts.append(f"w{width}")
    if batch != BASE_BATCH[algo]:
        parts.append(f"b{batch}")
    if utd != 1:
        parts.append(f"utd{utd}")
    if algo == "mbpo":
        base_reg = "rl1-1_re20-150" if row.env_id == HC else "rl1-25_re20-100"
        if row.mbpo_rollout_regime != base_reg:
            parts.append("rollout" + row.mbpo_rollout_regime.split("_")[0][2:].split("-")[1])
    return "+".join(parts) or "base"


def sc(x, d=3):
    return f"{x:.{d}f}"


# ---------------------------------------------------------------------------------------------
head("0. Environment")
print("python", sys.version.split()[0], "| scipy", scipy.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)
print("root:", ROOT)

pr = pd.read_csv(OUT / "per_run_energy_per_flop.csv")
pr["run_dir"] = pr["run_dir"].str.replace("\\", "/", regex=False)
prc = pr[pr.seed.isin(CANON) & pr.env_id.isin(ENVS)].copy()
assert prc.run_dir.nunique() == 280
runs = {}
for rd, g in prc.groupby("run_dir"):
    r0 = g.iloc[0]
    runs[rd] = dict(algo=r0.algo, env=r0.env_id, seed=int(r0.seed), config=parse(r0))
RUNS = pd.DataFrame.from_dict(runs, orient="index")
RUNS.index.name = "run_dir"
tot = prc[prc.segment == "TOTAL_MEASURED_TRAINING"].set_index("run_dir")
RUNS["F"] = tot.total_flops
RUNS["E_pipeline"] = tot.total_energy_joules
print("canonical runs:", len(RUNS), "| configurations:", RUNS.groupby(["algo", "env", "config"]).ngroups)

META = {rd: json.load(open(ROOT / rd / "metadata.json")) for rd in RUNS.index}

cols = ["task_name", "timestamp", "duration", "cpu_power", "gpu_power", "ram_power", "cpu_energy", "gpu_energy", "ram_energy",
        "energy_consumed", "os", "python_version", "codecarbon_version", "cpu_count", "cpu_model", "gpu_count", "gpu_model",
        "ram_total_size", "tracking_mode"]
parts = []
for rd in RUNS.index:
    f = glob.glob(str(ROOT / rd / "emissions_base_*.csv"))
    assert len(f) == 1, (rd, f)
    d = pd.read_csv(f[0])[cols]
    d["run_dir"] = rd
    parts.append(d)
T = pd.concat(parts, ignore_index=True)
T["seg"] = T.task_name.str.replace(r"_\d+$", "", regex=True)
T["epoch"] = T.task_name.str.extract(r"_(\d+)$")[0].astype(float)
T["E_J"] = T.energy_consumed * J
for c in ("cpu", "gpu", "ram"):
    T[c + "_J"] = T[c + "_energy"] * J
T = T.join(RUNS[["algo", "env", "config", "seed"]], on="run_dir")
print("task rows:", len(T), "| segments:", sorted(T.seg.unique()))
print(T.groupby(["algo", "seg"]).size().unstack(0).to_string())
NI = T[~T.seg.isin(IDLE)]

# ---------------------------------------------------------------------------------------------
head("1. Task rows, RAM power, zero GPU energy, short tasks")
print("rows in all emissions_base files:", len(T), "| idle baseline rows:", int(T.seg.isin(IDLE).sum()), "| other rows:", len(NI),
      "| rows - 2*280 =", len(T) - 560)
print("ram_power: min %.6f max %.6f unique(rounded 9) %s" % (T.ram_power.min(), T.ram_power.max(), sorted(T.ram_power.round(9).unique())[:5]))
bins = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 5, 1e9]
lab = ["<=0.01", "(0.01,0.025]", "(0.025,0.05]", "(0.05,0.1]", "(0.1,0.2]", "(0.2,0.5]", "(0.5,1]", "(1,5]", ">5"]
NI = NI.assign(b=pd.cut(NI.duration, bins, labels=lab))
tb = NI.groupby("b", observed=True).agg(tasks=("duration", "size"), gpu_eq0=("gpu_energy", lambda x: int((x == 0).sum())),
                                        gpu_lt0=("gpu_energy", lambda x: int((x < 0).sum())), gpu_le0=("gpu_energy", lambda x: int((x <= 0).sum())))
print("tab:thr-gpu-zero (all 280 runs, all rows except idle baselines):")
print(tb.to_string())
print("sum of tasks:", int(tb.tasks.sum()), "| zero total:", int(tb.gpu_eq0.sum()), "| min duration of any task with zero GPU energy: %.4f s"
      % NI[NI.gpu_energy == 0].duration.max())
print("max duration among tasks with gpu_energy==0: %.4f s; min duration among tasks with gpu_energy>0: %.5f s" %
      (NI[NI.gpu_energy == 0].duration.max(), NI[NI.gpu_energy > 0].duration.min()))
print("tasks >= 0.1 s with zero GPU energy:", int(((NI.duration >= 0.1) & (NI.gpu_energy == 0)).sum()))
S = NI[(NI.algo == "mbpo") & (NI.env == HC) & (NI.seg == "synthetic_rollout_generation")]
nz = S[S.gpu_J > 0]
print(f"MBPO HalfCheetah synthetic tasks (all 7 configs): {len(S)}; non-zero GPU: {len(nz)} ({len(nz) / len(S) * 100:.2f} %); "
      f"GPU J min {nz.gpu_J.min():.3f} max {nz.gpu_J.max():.3f} median {nz.gpu_J.median():.3f}; "
      f"duration median all {S.duration.median() * 1e3:.2f} ms, of non-zero {nz.duration.median() * 1e3:.2f} ms; "
      f"mean duration {S.duration.mean() * 1e3:.2f} ms; fraction nz / (mean duration/0.1 s) = {len(nz) / len(S):.4f} / {S.duration.mean() / 0.1:.4f}")
NI = NI.assign(Pg=NI.gpu_J / NI.duration)
sh = NI[NI.duration < 1]
print(f"GPU power (gpu_J/duration) tasks <1 s: n={len(sh)} p99={sh.Pg.quantile(.99):.1f} W max={sh.Pg.max():.1f} W; "
      f"segments of the 20 largest: {sh.nlargest(20, 'Pg').seg.value_counts().to_dict()}; "
      f"algo/env of those: {sh.nlargest(20, 'Pg').groupby(['algo', 'env']).size().to_dict()}")
lg = NI[NI.duration >= 2]
print(f"GPU power tasks >= 2 s: n={len(lg)} min {lg.Pg.min():.1f} W max {lg.Pg.max():.1f} W")
print("tasks < 1 s per segment:", sh.groupby(["algo", "seg"]).size().to_dict())
print("max GPU power of tasks < 1 s excluding synthetic_rollout_generation: %.1f W" % sh[sh.seg != "synthetic_rollout_generation"].Pg.max())

# ---------------------------------------------------------------------------------------------
head("2. Rollout tasks: durations, median power, RAM share, per-task GPU energy, CV, rollout share")
RO = NI[NI.seg == "rollout"].assign(P=lambda d: d.E_J / d.duration)
for scope, df in [("baselines", RO[RO.config == "base"]), ("all configurations", RO)]:
    g = df.groupby(["algo", "env"]).agg(n=("P", "size"), dur_min=("duration", "min"), dur_med=("duration", "median"), dur_max=("duration", "max"),
                                         P_med=("P", "median"), P_mean_of_tasks=("P", "mean"), gpuJ_mean=("gpu_J", "mean"), gpuJ_med=("gpu_J", "median"),
                                         gpuJ_min=("gpu_J", "min"), gpuJ_max=("gpu_J", "max"))
    g["ram_share_of_median_power_%"] = 20.0 / g.P_med * 100
    g["sum_E/sum_T_W"] = df.groupby(["algo", "env"]).apply(lambda x: x.E_J.sum() / x.duration.sum())
    print(f"-- rollout tasks, {scope}")
    print(g.round(4).to_string())
    print("   duration min/max over algorithms (algo!=tdmpc2): %.4f / %.4f s; tdmpc2: %.3f / %.3f s, median %.3f s" %
          (df[df.algo != "tdmpc2"].duration.min(), df[df.algo != "tdmpc2"].duration.max(), df[df.algo == "tdmpc2"].duration.min(),
           df[df.algo == "tdmpc2"].duration.max(), df[df.algo == "tdmpc2"].duration.median()))
    nonT = g[g.index.get_level_values(0) != "tdmpc2"]
    print("   median task power range over sac/td3/mbpo x env: %.1f - %.1f W; incl. tdmpc2: %.1f - %.1f W" % (nonT.P_med.min(), nonT.P_med.max(), g.P_med.min(), g.P_med.max()))
    print("   RAM share of median power: sac/td3/mbpo %.1f - %.1f %%; tdmpc2 %.1f - %.1f %%" % (
        nonT["ram_share_of_median_power_%"].min(), nonT["ram_share_of_median_power_%"].max(),
        g.loc["tdmpc2"]["ram_share_of_median_power_%"].min(), g.loc["tdmpc2"]["ram_share_of_median_power_%"].max()))
    cvp = df.assign(Pg=df.gpu_J / df.duration)
    cv_pool = cvp.groupby(["algo", "env"]).Pg.apply(lambda x: x.std(ddof=1) / x.mean() * 100)
    cv_run = cvp.groupby(["algo", "env", "run_dir"]).Pg.apply(lambda x: x.std(ddof=1) / x.mean() * 100).groupby(["algo", "env"]).mean()
    print("   CV of per-task GPU power [%]: pooled over tasks / mean of per-run CV")
    print(pd.concat([cv_pool.rename("pooled"), cv_run.rename("mean_per_run")], axis=1).round(1).to_string())
# MBPO synthetic zero tasks at baseline
sb = NI[(NI.algo == "mbpo") & (NI.config == "base") & (NI.seg == "synthetic_rollout_generation")]
print("MBPO baseline synthetic tasks:", len(sb), "zero GPU energy:", int((sb.gpu_energy == 0).sum()), sb.groupby("env").apply(
    lambda x: f"{int((x.gpu_energy == 0).sum())}/{len(x)} zero, median duration {x.duration.median() * 1e3:.1f} ms").to_dict())
mt = NI[(NI.algo == "mbpo") & (NI.config == "base") & NI.seg.isin(MEAS)]
print("synthetic share of MBPO baseline training total [%] (ratio of sums per env):",
      (mt[mt.seg == "synthetic_rollout_generation"].groupby("env").E_J.sum() / mt.groupby("env").E_J.sum() * 100).round(3).to_dict())
# rollout share of training energy
MT = NI[NI.seg.isin(MEAS)]
rs = MT.groupby(["run_dir", "seg"]).E_J.sum().unstack().fillna(0)
rs["total"] = rs.sum(axis=1)
rs = rs.join(RUNS[["algo", "env", "config", "seed"]])
rs["roll_share"] = rs.rollout / rs.total * 100
base_sh = rs[rs.config == "base"].groupby(["algo", "env"]).roll_share.mean()
print("rollout share of the training total [%], mean of per-run shares, baselines:")
print(base_sh.round(3).to_string())
sel = rs[rs.algo.isin(["sac", "td3", "mbpo"])].groupby(["algo", "env", "config"]).roll_share.mean()
print("(a) baselines of sac/td3/mbpo: min %.3f max %.3f | (b) all %d configurations of sac/td3/mbpo: min %.3f (%s) max %.3f (%s)" % (
    base_sh.drop("tdmpc2", level=0).min(), base_sh.drop("tdmpc2", level=0).max(), len(sel), sel.min(), sel.idxmin(), sel.max(), sel.idxmax()))
selr = rs[rs.algo.isin(["sac", "td3", "mbpo"])].groupby(["algo", "env", "config"]).apply(lambda x: x.rollout.sum() / x.total.sum() * 100)
print("   same as ratio of sums: min %.3f max %.3f" % (selr.min(), selr.max()))
print("   MBPO UTD 4 HalfCheetah: rollout J %.0f of total %.0f -> %.3f %%" % (
    rs[(rs.algo == "mbpo") & (rs.env == HC) & (rs.config == "utd4")].rollout.mean(),
    rs[(rs.algo == "mbpo") & (rs.env == HC) & (rs.config == "utd4")].total.mean(),
    rs[(rs.algo == "mbpo") & (rs.env == HC) & (rs.config == "utd4")].roll_share.mean()))
print(sel.sort_values().head(5).round(3).to_string())
print(sel.sort_values().tail(5).round(3).to_string())
chk = rs.total.sub(RUNS.E_pipeline).abs() / RUNS.E_pipeline
print("max rel. diff task-sum of measured segments vs pipeline TOTAL_MEASURED_TRAINING: %.2e" % chk.max())

# ---------------------------------------------------------------------------------------------
head("3. Stationarity (baselines), within-run CV, trends, MBPO growth decomposition")
EPS = ["rollout", "gradient_updates", "dynamics_model_update", "synthetic_rollout_generation"]
B = NI[(NI.config == "base") & NI.epoch.notna() & NI.seg.isin(EPS)]
pe = B.pivot_table(index=["algo", "env", "seed", "epoch"], columns="seg", values="E_J", aggfunc="sum").fillna(0)
pe["TOTAL"] = pe[[c for c in EPS if c in pe.columns]].sum(axis=1)


def trend(x):
    n = len(x)
    return np.polyfit(np.arange(n), x, 1)[0] * 100 / x.mean() * 100


rows = []
for (a, e), g in pe.groupby(level=[0, 1]):
    for seg in EPS + ["TOTAL"]:
        if seg not in g.columns or g[seg].sum() == 0:
            continue
        rat, cv, tr, r2 = [], [], [], []
        for s, x in g[seg].groupby(level="seed"):
            x = x.values
            assert len(x) == 100
            rat.append(x[:10].mean() / x[10:].mean())
            cv.append(x.std(ddof=1) / x.mean() * 100)
            tr.append(trend(x))
            if seg == "TOTAL":
                r2.append(np.corrcoef(1000 * (np.arange(100) + 1), np.cumsum(x))[0, 1] ** 2)
        m = g[seg].groupby(level="epoch").mean().values
        rows.append(dict(algo=a, env=e, seg=seg, ratio_seedmean=np.mean(rat), ratio_of_seedmeans=m[:10].mean() / m[10:].mean(), cv_pct=np.mean(cv),
                         trend_min=min(tr), trend_max=max(tr), trend_mean=np.mean(tr), r2=np.mean(r2) if r2 else np.nan))
ST = pd.DataFrame(rows)
print(ST.round(4).to_string())
print("tab:thr-stationarity (TOTAL):")
print(ST[ST.seg == "TOTAL"].pivot(index="algo", columns="env", values=["ratio_seedmean", "r2"]).round(3).to_string())
tt = ST[(ST.seg == "TOTAL")]
print("ratio range SAC/TD3/TD-MPC2:", tt[tt.algo != "mbpo"].ratio_seedmean.min().round(4), tt[tt.algo != "mbpo"].ratio_seedmean.max().round(4))
print("CV of total, sac/td3: %.2f - %.2f %%; tdmpc2: %s %%" % (tt[tt.algo.isin(["sac", "td3"])].cv_pct.min(), tt[tt.algo.isin(["sac", "td3"])].cv_pct.max(),
                                                         tt[tt.algo == "tdmpc2"].cv_pct.round(2).tolist()))
gt = ST[(ST.algo == "td3") & (ST.seg == "gradient_updates")]
print("TD3 gradient_updates per-seed trend (% of mean per 100 epochs), min/max over seeds:", gt.trend_min.min().round(2), gt.trend_max.max().round(2))
mb = ST[ST.algo == "mbpo"]
print("MBPO flat segments (rollout, gradient_updates) ratios:", mb[mb.seg.isin(["rollout", "gradient_updates"])].ratio_seedmean.round(3).tolist(),
      "(of seed means:", mb[mb.seg.isin(["rollout", "gradient_updates"])].ratio_of_seedmeans.round(3).tolist(), ")")
print(mb[mb.seg.isin(["dynamics_model_update", "synthetic_rollout_generation"])][["env", "seg", "ratio_seedmean", "ratio_of_seedmeans", "trend_mean"]].round(3).to_string())

# MBPO growth decomposition
print("\n-- MBPO fit-energy growth decomposition (baselines)")
ceil = math.ceil
rec = []
for rd in RUNS[(RUNS.algo == "mbpo") & (RUNS.config == "base")].index:
    tm = json.load(open(ROOT / rd / "training_metrics.json"))
    ep = tm["epochs"]
    fe = B[(B.run_dir == rd) & (B.seg == "dynamics_model_update")].set_index("epoch").E_J.sort_index()
    for e in range(100):
        n = 5000 + 1000 * e
        ntr = n - max(1, int(n * 0.2))
        steps = ceil(ntr / 256)
        mte = ep[e].get("model_train_epochs")
        rec.append(dict(run_dir=rd, env=RUNS.env[rd], epoch=e, n=n, steps_per_fit_epoch=steps, fit_epochs=mte, fit_J=fe.get(e, np.nan),
                        buffer_end=ep[e]["buffer_size"]))
D = pd.DataFrame(rec)
D["opt_steps"] = D.fit_epochs * D.steps_per_fit_epoch
print("buffer at end of epoch e equals 5000+1000(e+1) in all rows:", bool((D.buffer_end == D.n + 1000).all()), "| rows:", len(D), "| missing fit energy:", int(D.fit_J.isna().sum()))
print("optimiser steps per run (mean over seeds):", D.groupby(["env", "run_dir"]).opt_steps.sum().groupby("env").mean().round(0).to_dict(),
      "| total fit epochs per run:", D.groupby(["env", "run_dir"]).fit_epochs.sum().groupby("env").mean().round(1).to_dict())
for env, g in D.groupby("env"):
    early = g[(g.epoch >= 1) & (g.epoch <= 10)]
    late = g[(g.epoch >= 90) & (g.epoch <= 99)]
    ep_ = dict(fit_J=(early.fit_J.mean(), late.fit_J.mean()), steps_per_fit_epoch=(early.steps_per_fit_epoch.mean(), late.steps_per_fit_epoch.mean()),
               fit_epochs=(early.fit_epochs.mean(), late.fit_epochs.mean()), opt_steps=(early.opt_steps.mean(), late.opt_steps.mean()))
    ej_e = (early.fit_J / early.opt_steps).mean()
    ej_l = (late.fit_J / late.opt_steps).mean()
    print(f"{env}: epochs 1-10 vs 90-99")
    for k, (a, b) in ep_.items():
        print(f"   {k:20s} {a:10.3f} {b:10.3f}  factor {b / a:.3f}")
    print(f"   energy per optimiser step: ratio of group means {early.fit_J.mean() / early.opt_steps.mean():.3f} -> {late.fit_J.mean() / late.opt_steps.mean():.3f} "
          f"(factor {(late.fit_J.mean() / late.opt_steps.mean()) / (early.fit_J.mean() / early.opt_steps.mean()):.4f}); "
          f"mean of per-epoch ratios {ej_e:.3f} -> {ej_l:.3f} (factor {ej_l / ej_e:.3f})")
    print("   log decomposition (steps/epoch, fit epochs, J/step) = %.2f %+.2f %+.2f ; ln(energy factor) = %.2f" % (
        math.log(ep_["steps_per_fit_epoch"][1] / ep_["steps_per_fit_epoch"][0]), math.log(ep_["fit_epochs"][1] / ep_["fit_epochs"][0]),
        math.log((late.fit_J.mean() / late.opt_steps.mean()) / (early.fit_J.mean() / early.opt_steps.mean())),
        math.log(ep_["fit_J"][1] / ep_["fit_J"][0])))
src = (ROOT / "algorithms" / "dynamics_model.py").read_text()
print("code check (algorithms/dynamics_model.py): GaussianEnsembleMLP( constructed %d time(s); optimizer created %d time(s) (both in __init__, not in fit => warm start); "
      "'reset_parameters' occurrences: %d; patience break present: %s; holdout int(n*ratio): %s; batch loop 'range(0, n_train, batch_size)': %s" % (
          src.count("GaussianEnsembleMLP("), src.count("torch.optim.Adam("), src.count("reset_parameters"), "epochs_since_improved >= self.cfg.model_train_patience" in src,
          "n_holdout = max(1, int(n * self.cfg.model_holdout_ratio))" in src, "for start in range(0, n_train, batch_size)" in src))

# ---------------------------------------------------------------------------------------------
head("4. Idle drift (window-corrected and as recorded), correlation with temperature, tab:idle-by-date")
ID = {}
for ph in ("head", "tail"):
    r = T[T.seg == f"idle_baseline_{ph}"].set_index("run_dir")
    win = r.ram_J / r.ram_power
    ID[ph + "_corr"] = r.E_J / win
    ID[ph + "_rec"] = r.E_J / r.duration
    ID[ph + "_win_s"] = win
    ID[ph + "_dur_s"] = r.duration
    ID[ph + "_cpu_corr"] = r.cpu_J / win
    ID[ph + "_gpu_corr"] = r.gpu_J / win
ID = pd.DataFrame(ID).join(RUNS[["algo", "env", "config", "seed"]])
ID["start"] = pd.to_datetime(pd.Series({rd: META[rd]["start_time_utc"] for rd in RUNS.index}))
ID["date"] = ID.start.dt.date
ID["temp0"] = [META[rd].get("thermal_gate", {}).get("run_start_temp_c") for rd in ID.index]
ID["temp_gate"] = [META[rd].get("thermal_gate", {}).get("final_temp_c") for rd in ID.index]
for conv in ("rec", "corr"):
    h, t = ID[f"head_{conv}"], ID[f"tail_{conv}"]
    d = (t - h).abs()
    rel = d / h * 100
    print(f"[{conv}] tail>head: {int((t > h).sum())}/280 | median |diff| {d.median():.2f} W ({rel.median():.2f} %) | IQR of % {rel.quantile(.25):.2f}-{rel.quantile(.75):.2f} | "
          f"max {rel.max():.2f} % | mean head {h.mean():.2f} W tail {t.mean():.2f} W | mean of the two windows {((h + t) / 2).mean():.2f} W")
gd = (ID.tail_gpu_corr - ID.head_gpu_corr) / ID.head_gpu_corr * 100
cd = (ID.tail_cpu_corr - ID.head_cpu_corr) / ID.head_cpu_corr * 100
print(f"[corr] GPU term change median {gd.median():.2f} %, CPU term median {cd.median():.2f} %")
print("integrated span head: median %.3f s (min %.3f max %.3f), tail median %.3f s" % (ID.head_win_s.median(), ID.head_win_s.min(), ID.head_win_s.max(), ID.tail_win_s.median()))
for tc in ("temp0", "temp_gate"):
    x = ID[["head_corr", tc]].dropna()
    print(f"r(head idle power corrected, {tc}) = {x.corr().iloc[0, 1]:.3f} (n={len(x)}; temperature available for {ID[tc].notna().sum()} runs)")
bd = ID.groupby("date").agg(runs=("head_corr", "size"), median=("head_corr", "median"), min=("head_corr", "min"), max=("head_corr", "max"),
                            temp_med=("temp0", "median"), temp_gate_med=("temp_gate", "median"))
print("tab:idle-by-date:")
print(bd.round(2).to_string())
print("runs per day sum:", int(bd.runs.sum()), "| daily medians min/max: %.2f / %.2f W" % (bd["median"].min(), bd["median"].max()))

# ---------------------------------------------------------------------------------------------
head("5/6. Net energy: tab:thr-gross-net reproduction (40 baselines) and all 280 runs")
IDm = ID[["head_corr", "tail_corr"]].copy()
IDm["mean"] = (IDm.head_corr + IDm.tail_corr) / 2
M = NI[NI.seg.isin(MEAS)].join(IDm, on="run_dir")
M["gross"] = M.E_J
M["net_mean"] = M.E_J - M["mean"] * M.duration
M["net_head"] = M.E_J - M.head_corr * M.duration
M["net_tail"] = M.E_J - M.tail_corr * M.duration
print("measured task rows:", len(M), "| tasks with net_mean < 0:", int((M.net_mean < 0).sum()), "(", M[M.net_mean < 0].groupby(["algo", "seg"]).size().to_dict(),
      ") | head:", int((M.net_head < 0).sum()), "| tail:", int((M.net_tail < 0).sum()))
RS = M.groupby(["run_dir", "seg"])[VARIANTS].sum()
print("run-segments with negative net (mean):", int((RS.net_mean < 0).sum()), "of", len(RS))
RT = RS.groupby("run_dir").sum().join(RUNS)
for v in VARIANTS:
    RT[v + "_roll"] = RS.xs("rollout", level="seg")[v]
    RT[v + "_dyn"] = RS.xs("dynamics_model_update", level="seg")[v].reindex(RT.index)
    RT[v + "_gu"] = RS.xs("gradient_updates", level="seg")[v]
print("run-segments with negative net (mean):", RS[RS.net_mean < 0].reset_index()[["run_dir", "seg", "net_mean", "gross"]].values.tolist())
print("negative run totals:", int((RT[VARIANTS] < 0).sum().sum()))
print("check gross total vs pipeline: max rel diff %.2e" % ((RT.gross - RT.E_pipeline).abs() / RT.E_pipeline).max())
RT["idle_mean_W"] = IDm["mean"]
CT = RT.groupby(["env", "algo", "config"]).agg(**{v: (v, "mean") for v in VARIANTS}, F=("F", "mean"), n=("seed", "size"), idle=("idle_mean_W", "mean"))
for v in VARIANTS:
    CT["jpf_" + v] = CT[v] / CT.F * 1e12  # pJ/FLOP
print("configurations:", len(CT), "| seeds per configuration:", sorted(CT.n.unique()))

# tab:thr-gross-net
print("\n-- tab:thr-gross-net reproduction (mean over 5 seeds; net = mean of head and tail)")
rows = []
for a in ALGOS:
    for e in ENVS:
        c = CT.loc[(e, a, "base")]
        rows.append(dict(algo=a, env=e, gross_kJ=c.gross / 1e3, net_kJ=c.net_mean / 1e3, ratio=c.net_mean / c.gross, jpf_gross=c.jpf_gross, jpf_net=c.jpf_net_mean,
                         net_head_kJ=c.net_head / 1e3, net_tail_kJ=c.net_tail / 1e3, jpf_net_head=c.jpf_net_head, jpf_net_tail=c.jpf_net_tail))
GN = pd.DataFrame(rows)
print(GN.round(3).to_string())
print("net/gross range: %.3f - %.3f -> removed %.1f - %.1f %%" % (GN.ratio.min(), GN.ratio.max(), (1 - GN.ratio.max()) * 100, (1 - GN.ratio.min()) * 100))
for e in ENVS:
    for v in VARIANTS:
        g = CT.loc[e].xs("base", level="config")
        eo = " < ".join(g[v].sort_values().index)
        jo = " < ".join(g["jpf_" + v].sort_values().index)
        print(f"baseline orderings {e:15s} {v:9s} energy: {eo:28s} J/FLOP: {jo}")
g = {v: {e: CT.loc[e].xs("base", level="config") for e in ENVS} for v in VARIANTS}
for e in ENVS:
    for v in VARIANTS:
        d = g[v][e]
        print(f"  ratios {e:15s} {v:9s} tdmpc2/mbpo {d.loc['tdmpc2', v] / d.loc['mbpo', v]:.3f} | sac/td3 {d.loc['sac', v] / d.loc['td3', v]:.3f} | mbpo/sac {d.loc['mbpo', v] / d.loc['sac', v]:.3f}"
              f" | J/FLOP tdmpc2/sac {d.loc['tdmpc2', 'jpf_' + v] / d.loc['sac', 'jpf_' + v]:.3f}")
print("\nhead-only vs tail-only net total, relative to the smaller (baselines):")
for _, r in GN.iterrows():
    hi, lo = max(r.net_head_kJ, r.net_tail_kJ), min(r.net_head_kJ, r.net_tail_kJ)
    print(f"  {r.algo:7s} {r.env:15s} head {r.net_head_kJ:.2f} tail {r.net_tail_kJ:.2f} kJ  difference {(hi / lo - 1) * 100:.2f} % of the smaller, {(hi - lo) / ((hi + lo) / 2) * 100:.2f} % of their mean")
print("net/gross at 4 decimals:", GN.ratio.round(4).tolist())
# shares
RT["tot"] = RT[VARIANTS].values[:, 0]
print("\n-- segment shares of the baselines (mean of per-run shares, %), gross vs net(mean)")
sh_rows = []
for (a, e), gr in RT[RT.config == "base"].groupby(["algo", "env"]):
    seg_df = RS.loc[gr.index].reset_index()
    for seg, s in seg_df.groupby("seg"):
        s = s.set_index("run_dir")
        gsh = (s.gross / RT.loc[s.index, "gross"]).mean() * 100
        nsh = (s.net_mean / RT.loc[s.index, "net_mean"]).mean() * 100
        hsh = (s.net_head / RT.loc[s.index, "net_head"]).mean() * 100
        tsh = (s.net_tail / RT.loc[s.index, "net_tail"]).mean() * 100
        sh_rows.append(dict(algo=a, env=e, seg=seg, gross=gsh, net_mean=nsh, net_head=hsh, net_tail=tsh, shift=nsh - gsh))
SH = pd.DataFrame(sh_rows)
print(SH.round(2).to_string())
print("largest |shift| gross->net(mean): %.2f pp (%s)" % (SH["shift"].abs().max(), SH.loc[SH["shift"].abs().idxmax(), ["algo", "env", "seg"]].tolist()))
print("largest |shift| gross->any net variant: %.2f pp" % max((SH[v] - SH.gross).abs().max() for v in ["net_mean", "net_head", "net_tail"]))

# orderings of the sweeps
print("\n-- orderings of tab:rob-energy / tab:rob-jpf, per variant")
SW = [("Baseline", dict(sac="base", td3="base", mbpo="base")), ("UTD 2", dict(sac="utd2", td3="utd2", mbpo="utd2")),
      ("UTD 4", dict(sac="utd4", td3="utd4", mbpo="utd4")), ("Width 256", dict(sac="w256", td3="w256", mbpo="w256")),
      ("Width 512", dict(sac="w512", td3="w512", mbpo="w512")), ("Batch 256", dict(sac="base", td3="b256", mbpo="base")),
      ("Batch 512", dict(sac="b512", td3="b512", mbpo="b512")), ("Batch 1024", dict(sac="b1024", td3="b1024", mbpo="b1024"))]
ABB = dict(sac="S", td3="T", mbpo="M")
RTi = RT.reset_index()
PAIRS = [("td3", "sac"), ("sac", "mbpo"), ("td3", "mbpo")]


def seed_vals(env, algo, cfg, v, metric):
    d = RTi[(RTi.env == env) & (RTi.algo == algo) & (RTi.config == cfg)].set_index("seed")
    return (d[v] if metric == "energy" else d[v] / d.F * 1e12)


summary = {}
for v in VARIANTS:
    for metric in ("energy", "jpf"):
        base_order = {}
        for e in ENVS:
            for name, sel in SW:
                if metric == "energy":
                    vals = {a: CT.loc[(e, a, c), v] for a, c in sel.items()}
                else:
                    vals = {a: CT.loc[(e, a, c), "jpf_" + v] for a, c in sel.items()}
                order = sorted(vals, key=vals.get)
                if name == "Baseline":
                    base_order[e] = order
                rank = {a: i for i, a in enumerate(order)}
                brank = {a: i for i, a in enumerate(base_order[e])}
                swaps = [f"{a}-{b}" for a, b in itertools.combinations(["sac", "td3", "mbpo"], 2) if (rank[a] < rank[b]) != (brank[a] < brank[b])]
                sv = {a: seed_vals(e, a, c, v, metric) for a, c in sel.items()}
                n_eq = sum(1 for s in CANON if sorted(sv, key=lambda a: sv[a][s]) == order)
                n_base = sum(1 for s in CANON if sorted(sv, key=lambda a: sv[a][s]) == base_order[e])
                ratios = [vals[order[i + 1]] / vals[order[i]] for i in range(2)]
                summary[(v, metric, e, name)] = dict(order="".join(ABB[a] for a in order), swaps=swaps, n_eq=n_eq, n_base=n_base, min_ratio=min(ratios),
                                                     vals=vals, pair={(a, b): (vals[b] / vals[a], sum(1 for s in CANON if sv[b][s] > sv[a][s])) for a, b in PAIRS})
for metric in ("energy", "jpf"):
    print(f"\n### {metric}")
    for e in ENVS:
        print(f"-- {e}")
        for name, _ in SW:
            line = f"{name:11s}"
            for v in VARIANTS:
                r = summary[(v, metric, e, name)]
                line += f" | {v:8s} {r['order']} seeds {r['n_eq']}/5 base-ord {r['n_base']}/5 min-ratio {r['min_ratio']:.3f} swaps {','.join(r['swaps']) or '-'}"
            print(line)
print("\n### gross values (check against tab:rob-energy [J] and tab:rob-jpf [pJ])")
for metric in ("energy", "jpf"):
    for e in ENVS:
        for name, _ in SW:
            vals = summary[("gross", metric, e, name)]["vals"]
            print(f"{metric:6s} {e:15s} {name:11s} " + "  ".join(f"{a}={vals[a]:.4g}" for a in ("sac", "td3", "mbpo")))
print("\n### do the net orderings, swaps and seed counts equal the gross ones?")
for metric in ("energy", "jpf"):
    for v in VARIANTS[1:]:
        diffs = []
        for e in ENVS:
            for name, _ in SW:
                a, b = summary[("gross", metric, e, name)], summary[(v, metric, e, name)]
                if a["order"] != b["order"] or a["swaps"] != b["swaps"] or a["n_eq"] != b["n_eq"] or a["n_base"] != b["n_base"]:
                    diffs.append((e, name, a["order"], b["order"], a["swaps"], b["swaps"], a["n_eq"], b["n_eq"], a["n_base"], b["n_base"]))
        print(f"{metric:6s} {v:9s}: {len(diffs)} of 16 combinations differ in ordering / swaps / seed counts")
        for d in diffs:
            print("    ", d)
print("\n### highlighted pairs (ratio of second to first / seeds of 5 in which second > first)")
for (e, name, a, b) in [(HC, "Width 512", "td3", "mbpo"), (HC, "UTD 4", "td3", "sac"), (HC, "UTD 4", "td3", "mbpo"), (ANT, "UTD 4", "td3", "sac"),
                        (ANT, "UTD 4", "td3", "mbpo"), (HC, "Width 256", "sac", "mbpo"), (ANT, "Width 256", "sac", "mbpo"), (ANT, "Width 512", "td3", "mbpo")]:
    for metric in ("energy", "jpf"):
        line = f"{metric:6s} {e:15s} {name:10s} {b}/{a}: "
        for v in VARIANTS:
            ratio, n = summary[(v, metric, e, name)]["pair"][(a, b)]
            line += f"{v}={ratio:.4f} ({n}/5)  "
        print(line)
# ranges
print("\n-- tab:rob-ranges (range over configurations; overlap with TD-MPC2)")
for v in VARIANTS:
    for metric, col in (("energy", v), ("jpf", "jpf_" + v)):
        for e in ENVS:
            c = CT.loc[e]
            rg = c.groupby(level="algo")[col].agg(["min", "max", "size"])
            t = rg.loc["tdmpc2"]
            ov = {a: bool(rg.loc[a, "min"] <= t["max"] and t["min"] <= rg.loc[a, "max"]) for a in ["sac", "td3", "mbpo"]}
            print(f"{v:9s} {metric:6s} {e:15s} " + "  ".join(f"{a}:[{rg.loc[a, 'min']:.4g},{rg.loc[a, 'max']:.4g}] overlap={ov[a]}" for a in ["sac", "td3", "mbpo"])
                  + f"  tdmpc2:[{t['min']:.4g},{t['max']:.4g}]")
# pooled correlations and spreads
print("\n-- pooled over all configurations of an environment: Spearman rho and log-log slope of energy on FLOPs, J/FLOP spread factor")
for v in VARIANTS:
    for e in ENVS:
        c = CT.loc[e]
        rho = stats.spearmanr(c.F, c[v])[0]
        sl = stats.linregress(np.log10(c.F), np.log10(c[v])).slope
        sp = c["jpf_" + v].max() / c["jpf_" + v].min()
        print(f"{v:9s} {e:15s} n={len(c)} rho={rho:.4f} slope={sl:.4f} J/FLOP spread {sp:.2f} (min {c['jpf_' + v].min():.2f} pJ, max {c['jpf_' + v].max():.2f} pJ)")
# environment effect
print("\n-- environment effect: Ant over HalfCheetah, 27 pairs x 5 seeds")
for v in VARIANTS:
    rr = RTi[["env", "algo", "config", "seed", v, v + "_roll"]]
    h = rr[rr.env == HC].set_index(["algo", "config", "seed"])
    a = rr[rr.env == ANT].set_index(["algo", "config", "seed"])
    j = h.join(a, lsuffix="_hc", rsuffix="_ant", how="inner")
    cfg_ratio = (j[v + "_ant"].groupby(level=[0, 1]).mean() / j[v + "_hc"].groupby(level=[0, 1]).mean())
    seed_ratio = j[v + "_ant"] / j[v + "_hc"]
    rsh = j[v + "_roll_ant"].groupby(level=[0, 1]).mean() / j[v + "_ant"].groupby(level=[0, 1]).mean() - j[v + "_roll_hc"].groupby(level=[0, 1]).mean() / j[v + "_hc"].groupby(level=[0, 1]).mean()
    print(f"{v:9s} pairs {len(cfg_ratio)} seed-pairs {len(seed_ratio)} | ratio min {cfg_ratio.min():.3f} max {cfg_ratio.max():.3f} | all configs > 1: {bool((cfg_ratio > 1).all())} | "
          f"seed pairs > 1: {int((seed_ratio > 1).sum())}/{len(seed_ratio)} | rollout share shift (Ant-HC) min {rsh.min() * 100:.2f} max {rsh.max() * 100:.2f} pp, positive in {int((rsh > 0).sum())}/{len(rsh)}")
    print("     configurations with ratio <= 1.05:", {k: round(x, 4) for k, x in cfg_ratio[cfg_ratio <= 1.05].items()}, "| seed pairs < 1 by configuration:",
          {k: int(x) for k, x in (seed_ratio < 1).groupby(level=[0, 1]).sum().items() if x > 0})

# ---------------------------------------------------------------------------------------------
head("7. Order and time confounding; sessions (30 min and 2 h)")
rows = []
for p in (RES).rglob("metadata.json"):
    m = json.load(open(p))
    rows.append(dict(run_dir=p.parent.relative_to(ROOT).as_posix(), algo=m["algo_name"], env=m["env_id"], seed=m["seed"],
                     start=pd.Timestamp(m["start_time_utc"]), end=pd.Timestamp(m["end_time_utc"]), commit=m["git_commit"][:7]))
A = pd.DataFrame(rows).join(RUNS[["config"]], on="run_dir")
A["canonical"] = A.seed.isin(CANON) & A.env.isin(ENVS) & A.run_dir.isin(RUNS.index)
A["config"] = A.config.fillna("DEV")
Ac = A[A.canonical].sort_values("start").reset_index(drop=True)
Ac["dur_h"] = (Ac.end - Ac.start).dt.total_seconds() / 3600
Ac["gap_min"] = (Ac.start - Ac.end.shift()).dt.total_seconds() / 60
print(f"canonical runs {len(Ac)}; first start {Ac.start.min()}, last end {Ac.end.max()}; overlaps (gap<0): {int((Ac.gap_min < 0).sum())}; min gap {Ac.gap_min.min():.2f} min; "
      f"sum of run durations {Ac.dur_h.sum():.2f} h; wall-clock span first start to last end {(Ac.end.max() - Ac.start.min()).total_seconds() / 3600:.2f} h")
print("gaps between consecutive runs: median %.2f min, 99th percentile %.2f, max %.2f min; gaps > 30 min: %d; > 2 h: %d" % (
    Ac.gap_min.median(), Ac.gap_min.quantile(.99), Ac.gap_min.max(), int((Ac.gap_min > 30).sum()), int((Ac.gap_min > 120).sum())))
print("gaps between 10 and 60 min:", Ac[(Ac.gap_min > 10) & (Ac.gap_min <= 60)].gap_min.round(1).tolist())
SESS = {}
for thr in (30, 120):
    Ac[f"s{thr}"] = (Ac.gap_min.isna() | (Ac.gap_min > thr)).cumsum()
    g = Ac.groupby(f"s{thr}").agg(start=("start", "min"), end=("end", "max"), runs=("run_dir", "size"), algos=("algo", lambda x: ",".join(dict.fromkeys(x))),
                                  envs=("env", lambda x: ",".join(dict.fromkeys(e[:2] for e in x))), cfgs=("config", lambda x: ",".join(dict.fromkeys(x))))
    g["span_h"] = (g.end - g.start).dt.total_seconds() / 3600
    SESS[thr] = g
    print(f"\nsessions with a gap threshold of {thr} min (canonical runs only): {len(g)}; sum of session spans {g.span_h.sum():.2f} h")
    print(g[["start", "end", "span_h", "runs", "algos", "envs"]].round(2).to_string())
# the pair statistics
Cf = Ac.groupby(["algo", "env", "config"]).agg(first=("start", "min"), last=("end", "max"), s30=("s30", "first"), s120=("s120", "first"),
                                              n30=("s30", "nunique"), n120=("s120", "nunique"))
Cf["day"] = Cf["first"].dt.normalize()
pairs = []
for (a, e), gr in Cf.groupby(level=[0, 1]):
    b = gr.xs("base", level="config").iloc[0]
    for cfg, r in gr.iterrows():
        if cfg[2] == "base":
            continue
        pairs.append(dict(kind="sweep-baseline", algo=a, env=e, cfg=cfg[2], days=(r.day - b.day).days, hours=(r["first"] - b["first"]).total_seconds() / 3600,
                          same30=r.s30 == b.s30, same120=r.s120 == b.s120))
for a in ALGOS:
    for c in set(Cf.loc[(a, HC)].index) & set(Cf.loc[(a, ANT)].index):
        h, n = Cf.loc[(a, HC, c)], Cf.loc[(a, ANT, c)]
        pairs.append(dict(kind="HC-Ant", algo=a, env="HC|Ant", cfg=c, days=(n.day - h.day).days, hours=(n["first"] - h["first"]).total_seconds() / 3600,
                          same30=h.s30 == n.s30, same120=h.s120 == n.s120))
P = pd.DataFrame(pairs)
P["diffday"] = P.days != 0
for k, g in P.groupby("kind"):
    print(f"{k}: pairs {len(g)}; different calendar day (UTC): {int(g.diffday.sum())}; different session (30 min): {int((~g.same30).sum())}; "
          f"different session (2 h): {int((~g.same120).sum())}; same session (30 min): {int(g.same30.sum())}; "
          f"hours apart among different-day pairs: min {g[g.diffday].hours.abs().min():.1f} max {g[g.diffday].hours.abs().max():.1f}; all pairs min {g.hours.abs().min():.1f} max {g.hours.abs().max():.1f}")
print("same-session sweep-baseline pairs (30 min):", P[(P.kind == "sweep-baseline") & P.same30][["algo", "env", "cfg"]].values.tolist())
print("same-session sweep-baseline pairs (2 h):", P[(P.kind == "sweep-baseline") & P.same120][["algo", "env", "cfg"]].values.tolist())
print("different-day sweep-baseline pairs per calendar day: hours-apart quantiles", P[(P.kind == "sweep-baseline") & P.diffday].hours.abs().describe().round(1).to_dict())
so = []
for (a, e, c), gr in Ac.groupby(["algo", "env", "config"]):
    gr = gr.sort_values("start")
    idx = gr.index.tolist()
    so.append(dict(order=list(gr.seed) == CANON, consecutive=all(idx[i + 1] - idx[i] == 1 for i in range(len(idx) - 1))))
so = pd.DataFrame(so)
print(f"seed order 331,958,14577,43611,85062: {int(so.order.sum())}/{len(so)} configurations; consecutive in the global order: {int(so.consecutive.sum())}/{len(so)}")
print("distinct git commits (canonical):", Ac.commit.nunique(), "| per commit:", Ac.commit.value_counts().to_dict())
print("calendar days (UTC) of run starts:", sorted({d.strftime('%Y-%m-%d') for d in Ac.start}), "| runs per day sum:", len(Ac))
print("first and last run: ", Ac.start.min().strftime("%Y-%m-%d %H:%M"), "-", Ac.end.max().strftime("%Y-%m-%d %H:%M"))
print("\nsessions in text (30 min): algo/env/config content per session:")
for thr in (30, 120):
    print(f"-- threshold {thr} min")
    for s, g in Ac.groupby(f"s{thr}"):
        print(f"  S{s}: {g.start.min():%m-%d %H:%M} .. {g.end.max():%m-%d %H:%M} runs {len(g)} | " + "; ".join(f"{a}/{e[:2]}:{','.join(dict.fromkeys(x.config))}" for (a, e), x in g.groupby(["algo", "env"], sort=False)))

# ---------------------------------------------------------------------------------------------
head("8. Learning check")
LR = []
for rd, r in RUNS.iterrows():
    tm = json.load(open(ROOT / rd / "training_metrics.json"))
    ep = [e for e in tm["episodes"] if e["phase"] == "train"]
    k = max(1, len(ep) // 10)
    f, l = ep[:k], ep[-k:]
    LR.append(dict(run_dir=rd, algo=r.algo, env=r.env, config=r.config, seed=r.seed, n_ep=len(ep), k=k,
                   first_ret=np.mean([e["return"] for e in f]), last_ret=np.mean([e["return"] for e in l]),
                   first_len=np.mean([e["length"] for e in f]), last_len=np.mean([e["length"] for e in l]),
                   first_ps=sum(e["return"] for e in f) / sum(e["length"] for e in f), last_ps=sum(e["return"] for e in l) / sum(e["length"] for e in l),
                   last_early=np.mean([e["length"] < 1000 for e in l]), any_early=any(e["length"] < 1000 for e in ep),
                   all_1000=all(e["length"] == 1000 for e in ep), n_warm=sum(1 for e in tm["episodes"] if e["phase"] != "train")))
LR = pd.DataFrame(LR)
LC = LR.groupby(["env", "algo", "config"]).agg(first_ret=("first_ret", "mean"), last_ret=("last_ret", "mean"), first_len=("first_len", "mean"),
                                              last_len=("last_len", "mean"), first_ps=("first_ps", "mean"), last_ps=("last_ps", "mean"), last_early=("last_early", "mean"),
                                              n_ep_mean=("n_ep", "mean"), n_ep_min=("n_ep", "min"), n_ep_max=("n_ep", "max"),
                                              seeds_improve=("last_ret", lambda x: 0), n=("seed", "size"))
LC["seeds_improve"] = LR.assign(imp=LR.last_ret > LR.first_ret).groupby(["env", "algo", "config"]).imp.sum()
LC["flag"] = LC.last_ret <= LC.first_ret
LC["flag_ps"] = LC.last_ps <= LC.first_ps
hc = LC.loc[HC]
ant = LC.loc[ANT]
print(f"HalfCheetah-v5: configurations {len(hc)}; improving (mean) {int((~hc.flag).sum())}; improving in all 5 seeds: {int((hc.seeds_improve == 5).sum())}; all episodes 1000 steps: {bool(LR[LR.env == HC].all_1000.all())}")
print(f"Ant-v5: configurations {len(ant)}; flagged (last <= first) {int(ant.flag.sum())}; by algo: {ant[ant.flag].groupby(level='algo').size().to_dict()}")
print(f"Ant-v5 runs with at least one early-terminated training episode: {int(LR[LR.env == ANT].any_early.sum())}/{int((LR.env == ANT).sum())}")
fl = ant[ant.flag]
print("\nflagged Ant-v5 configurations (tab:app-learning):")
print(fl[["first_ret", "last_ret", "first_len", "last_len", "first_ps", "last_ps", "flag_ps", "last_early", "seeds_improve"]].round(3).to_string())
print(f"episodes longer in last tenth: {int((fl.last_len > fl.first_len).sum())}/{len(fl)}; shorter: {fl[fl.last_len < fl.first_len].index.tolist()}")
print(f"per-step return rises: {int((~fl.flag_ps).sum())}/{len(fl)}; still not improving per step: {fl[fl.flag_ps].index.tolist()}")
print("frac terminated late episodes: flagged sac/td3:", fl[fl.index.get_level_values(0).isin(["sac", "td3"])].last_early.round(3).to_dict(),
      "| all sac/td3 Ant configs range %.3f - %.3f" % (ant.loc[["sac", "td3"]].last_early.min(), ant.loc[["sac", "td3"]].last_early.max()))
print("frac terminated (4 decimals) sac/td3 flagged:", fl[fl.index.get_level_values(0).isin(["sac", "td3"])].last_early.round(4).to_dict())
print("MBPO flagged configs: late episodes terminated fraction:", fl.loc["mbpo"].last_early.round(3).tolist() if "mbpo" in fl.index.get_level_values(0) else None)
print("episode counts on Ant-v5 (run level): sac all configs", int(LR[(LR.env == ANT) & (LR.algo == "sac")].n_ep.min()), "-", int(LR[(LR.env == ANT) & (LR.algo == "sac")].n_ep.max()),
      "| sac baseline", int(LR[(LR.env == ANT) & (LR.algo == "sac") & (LR.config == "base")].n_ep.min()), "-", int(LR[(LR.env == ANT) & (LR.algo == "sac") & (LR.config == "base")].n_ep.max()),
      "| tdmpc2 baseline", int(LR[(LR.env == ANT) & (LR.algo == "tdmpc2") & (LR.config == "base")].n_ep.min()), "-", int(LR[(LR.env == ANT) & (LR.algo == "tdmpc2") & (LR.config == "base")].n_ep.max()),
      "| tdmpc2 all five", int(LR[(LR.env == ANT) & (LR.algo == "tdmpc2")].n_ep.min()), "-", int(LR[(LR.env == ANT) & (LR.algo == "tdmpc2")].n_ep.max()),
      "| all algorithms", int(LR[LR.env == ANT].n_ep.min()), "-", int(LR[LR.env == ANT].n_ep.max()))
print("episode counts per algo (min/max over runs on Ant-v5):", LR[LR.env == ANT].groupby("algo").n_ep.agg(["min", "max"]).to_dict("index"))
print("TD-MPC2 Ant-v5 configs: n episodes, late/early length, return:")
print(ant.loc["tdmpc2"][["n_ep_min", "n_ep_max", "first_len", "last_len", "first_ret", "last_ret"]].round(1).to_string())
print("TD-MPC2 mean length of late episodes over configs (config means): %.1f - %.1f ; over runs: %.1f - %.1f" % (
    ant.loc["tdmpc2"].last_len.min(), ant.loc["tdmpc2"].last_len.max(), LR[(LR.env == ANT) & (LR.algo == "tdmpc2")].last_len.min(), LR[(LR.env == ANT) & (LR.algo == "tdmpc2")].last_len.max()))
print("MBPO Ant-v5 last-10% return, all 9 configs negative:", bool((ant.loc["mbpo"].last_ret < 0).all()), ant.loc["mbpo"].last_ret.round(1).tolist())
print("Ant-v5 return range TD-MPC2 (config means of last tenth): %.1f - %.1f" % (ant.loc["tdmpc2"].last_ret.min(), ant.loc["tdmpc2"].last_ret.max()))
# Spearman return vs FLOPs
for e in ENVS:
    x = CT.loc[e].join(LC.loc[e][["last_ret"]]).reset_index()
    r = stats.spearmanr(x.F, x.last_ret)
    print(f"Spearman(return of last tenth, FLOPs) {e}: n={len(x)} rho={r[0]:.3f} p={r[1]:.4f}")
# config means for tab:app-config-means (energy kJ, FLOPs, return)
print("\nconfig means (env, algo, config): FLOPs, gross energy [kJ], last-10% return")
cm = CT[["F", "gross"]].join(LC[["last_ret"]])
for k, r in cm.iterrows():
    print("CM", k[0], k[1], k[2], f"{r.F:.3e}", f"{r.gross / 1e3:.1f}", f"{r.last_ret:.1f}")

# ---------------------------------------------------------------------------------------------
head("9. Provenance uniformity across the 280 canonical runs")
V = pd.DataFrame({rd: dict(torch=META[rd]["torch_version"], codecarbon_meta=META[rd]["codecarbon_version"], device=META[rd]["device"], cuda_name=META[rd]["cuda_device_name"],
                           lock=META[rd]["experiment_config"].get("lock_gpu_clocks"), min_clk=META[rd]["experiment_config"].get("gpu_min_clock_mhz"),
                           max_clk=META[rd]["experiment_config"].get("gpu_max_clock_mhz"), persist=META[rd]["experiment_config"].get("set_persistence_mode"),
                           governor=META[rd]["experiment_config"].get("set_cpu_performance_governor"), trk=META[rd]["experiment_config"].get("tracking_mode"),
                           steps=META[rd]["experiment_config"].get("train_steps"), warm=META[rd]["experiment_config"].get("warmup_steps"),
                           spe=META[rd]["experiment_config"].get("steps_per_epoch"), settle=META[rd]["experiment_config"].get("settle_seconds"),
                           idle=META[rd]["experiment_config"].get("idle_baseline_seconds"), tail=META[rd]["experiment_config"].get("idle_tail_seconds"),
                           measure_secs=META[rd]["experiment_config"].get("measure_power_secs"), country=META[rd]["experiment_config"].get("country_iso_code"))
                  for rd in RUNS.index}).T
for c in V.columns:
    print(f"metadata {c:16s}: {V[c].astype(str).value_counts().to_dict()}")
for c in ["os", "python_version", "codecarbon_version", "cpu_count", "cpu_model", "gpu_count", "gpu_model", "tracking_mode"]:
    u = T.groupby("run_dir")[c].nunique()
    print(f"CSV {c:20s}: distinct values per run (max) {int(u.max())}; across all runs {T[c].astype(str).value_counts().to_dict()}")
print("ram_total_size distinct values (rounded 3):", T.ram_total_size.round(3).value_counts().to_dict())
print("glibc in os string:", T.os.str.extract(r"glibc([\d.]+)")[0].value_counts().to_dict())
mk = pd.Series({rd: tuple(sorted(META[rd].keys())) for rd in RUNS.index})
print("metadata key sets:", mk.map(lambda x: len(x)).value_counts().to_dict())
tgk = pd.Series({rd: tuple(sorted(META[rd].get("thermal_gate", {}).keys())) for rd in RUNS.index})
print("thermal_gate key sets (n keys):", tgk.map(len).value_counts().to_dict(), "| runs with run_start_temp_c:", int(sum("run_start_temp_c" in k for k in tgk)))
# commits
CM_ = A[A.canonical].sort_values("start")
cc = CM_.groupby("commit").agg(first=("start", "min"), last=("start", "max"), runs=("run_dir", "size"), algos=("algo", lambda x: ",".join(sorted(set(x)))))
print("commits (by first start):")
print(cc.sort_values("first").to_string())
print("distinct commits per algorithm (canonical, all configs):", CM_.groupby("algo").commit.nunique().to_dict())
print("distinct commits per algorithm (baselines only):", CM_[CM_.config == "base"].groupby("algo").commit.nunique().to_dict())
print("distinct commits per algorithm/env (all configs):", CM_.groupby(["algo", "env"]).commit.nunique().to_dict())
sacb = CM_[(CM_.algo == "sac") & (CM_.config == "base")].sort_values("start")
sacb = sacb.assign(has_start_temp=[("run_start_temp_c" in META[rd].get("thermal_gate", {})) for rd in sacb.run_dir])
print("SAC baseline runs in start order (env, seed, commit, thermal_gate has run_start_temp_c):")
print(sacb[["env", "seed", "commit", "start", "has_start_temp"]].to_string())
print("runs recording the earliest commit:", CM_[CM_.commit == cc.sort_values('first').index[0]].groupby(["algo", "env", "config"]).size().to_dict())
mh = pd.DataFrame([dict(run_dir=rd, commit=META[rd]["git_commit"][:7], config=RUNS.config[rd], env=RUNS.env[rd], hid=tuple(META[rd]["algo_config"].get("hidden_sizes", [])),
                        start=META[rd]["start_time_utc"]) for rd in RUNS[RUNS.algo == "mbpo"].index]).sort_values("start")
print("MBPO hidden sizes of baselines:", mh[mh.config == "base"].hid.value_counts().to_dict(), "| other hidden sizes by config:", mh[mh.hid != (1024, 1024)].groupby("config").hid.first().to_dict())
print("MBPO runs by commit and hidden size:", mh.groupby(["commit", "hid"]).size().to_dict())
# git diffs between consecutive recorded commits
try:
    order = list(cc.sort_values("first").index)
    full = {}
    for c in order:
        full[c] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", c], capture_output=True, text=True).stdout.strip()
    print("git commits resolvable:", {c: bool(v) for c, v in full.items()})
    watch = ["algorithms/sac.py", "algorithms/mbpo.py", "algorithms/dynamics_model.py", "algorithms/replay_buffer.py", "utils/gpu_control.py"]
    for a, b in zip(order[:-1], order[1:]):
        if not (full[a] and full[b]):
            continue
        names = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", full[a], full[b]], capture_output=True, text=True).stdout.split()
        print(f"  {a} -> {b}: changed files: {names}")
        d = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", full[a], full[b], "--"] + watch, capture_output=True, text=True).stdout.strip()
        print(f"     changes in watched files (sac, mbpo, dynamics_model, replay_buffer, gpu_control): {d or 'none'}")
except Exception as exc:  # git not available
    print("git check not possible:", exc)

# ---------------------------------------------------------------------------------------------
head("11. Development runs")
print("run directories with metadata.json:", len(A), "| canonical:", int(A.canonical.sum()), "| non-canonical:", int((~A.canonical).sum()))
dev = A[~A.canonical]
print(dev.groupby(["algo", "env"]).size().to_string())
print("seeds of non-canonical runs:", sorted(dev.seed.unique()))
print("MBPO seed 0 HalfCheetah dev runs:", dev[(dev.algo == "mbpo") & (dev.seed == 0)][["run_dir", "start"]].values.tolist())
print("dev runs with canonical seeds in other environments:", dev[dev.seed.isin(CANON)].groupby(["algo", "env"]).size().to_dict())

# ---------------------------------------------------------------------------------------------
head("12. Numbers for the design and scope limits")
cov = []
for rd, r in RUNS.iterrows():
    se = json.load(open(ROOT / rd / "segment_energy.json"))
    sub = se.get("_sub_segment_wall_time_seconds")
    gu = T[(T.run_dir == rd) & (T.seg == "gradient_updates")].duration.sum()
    cov.append(dict(run_dir=rd, algo=r.algo, env=r.env, config=r.config, sub=sum(sub.values()) if sub else np.nan, gu=gu))
cv = pd.DataFrame(cov)
cv["coverage"] = cv["sub"] / cv.gu * 100
print("allocation coverage (sum of four timers / measured GU duration), per run:")
print(cv.groupby("algo").coverage.agg(["min", "max", "count"]).round(2).to_string())
print("overall min %.2f (%s) max %.2f (%s) over %d runs" % (cv.coverage.min(), cv.loc[cv.coverage.idxmin(), ["algo", "env", "config"]].tolist(), cv.coverage.max(),
                                                         cv.loc[cv.coverage.idxmax(), ["algo", "env", "config"]].tolist(), int(cv.coverage.notna().sum())))
# clock study: R2000 against the five canonical MBPO width-256 runs on HalfCheetah-v5
BASE0 = RES / "mbpo" / "HalfCheetah-v5" / "seed_0"
R2000 = BASE0 / "20260904_122958"
rows = []
for name, d in [("R2000", R2000)] + [(f"canon {RUNS.seed[rd]}", ROOT / rd) for rd in RUNS[(RUNS.algo == "mbpo") & (RUNS.env == HC) & (RUNS.config == "w256")].index]:
    tm = json.load(open(d / "training_metrics.json"))
    ep = tm["epochs"]
    fits = [e["model_train_epochs"] for e in ep]
    steps = 0
    for e, x in enumerate(ep):
        n = x["buffer_size"] - 1000
        steps += x["model_train_epochs"] * math.ceil((n - max(1, int(n * 0.2))) / 256)
    tr = [e["return"] for e in tm["episodes"] if e["phase"] == "train"]
    k = max(1, len(tr) // 10)
    rel = str(d.relative_to(ROOT)).replace("\\", "/")
    flops = pr[(pr.run_dir == rel) & (pr.segment == "TOTAL_MEASURED_TRAINING")].total_flops
    rows.append(dict(run=name, total_flops=float(flops.iloc[0]) if len(flops) else np.nan, fit_epochs=sum(fits), optimiser_steps=steps, last10_return=float(np.mean(tr[-k:]))))
CL = pd.DataFrame(rows)
print("clock study: R2000 against the five canonical width-256 runs")
print(CL.to_string())
can = CL[CL.run != "R2000"]
print("canonical range: FLOPs %.3e - %.3e | fit epochs %d - %d | optimiser steps %d - %d | last-10%% return %.1f - %.1f" % (
    can.total_flops.min(), can.total_flops.max(), can.fit_epochs.min(), can.fit_epochs.max(), can.optimiser_steps.min(), can.optimiser_steps.max(),
    can.last10_return.min(), can.last10_return.max()))

head("13. Summary of the sweep orderings per convention")
for v in VARIANTS:
    for metric in ("energy", "jpf"):
        items = [(k[2], k[3], r) for k, r in summary.items() if k[0] == v and k[1] == metric]
        n_same = 0
        for e in ENVS:
            base_o = summary[(v, metric, e, "Baseline")]["order"]
            n_same += sum(1 for k, r in summary.items() if k[0] == v and k[1] == metric and k[2] == e and r["order"] == base_o)
        print(f"{v:9s} {metric:6s}: min neighbour ratio {min(r['min_ratio'] for _, _, r in items):.3f} | combinations with all 5 seeds equal to the order of the means "
              f"{sum(1 for _, _, r in items if r['n_eq'] == 5)}/16 | sweep values (of 16) with the baseline ordering {n_same} | orderings of the means equal to gross: "
              f"{sum(1 for e, n, r in items if r['order'] == summary[('gross', metric, e, n)]['order'])}/16")
head("14. Shares: mean of per-run shares against ratio of means (gross)")
dd = RS.reset_index().merge(RUNS[["algo", "env", "config", "seed"]].reset_index(), on="run_dir")
dd["tot"] = dd.run_dir.map(RT.gross)
dd["share"] = dd.gross / dd.tot * 100
res = []
for (a, e, c, s), g in dd.groupby(["algo", "env", "config", "seg"]):
    res.append(dict(algo=a, env=e, config=c, seg=s, mean_of_shares=g.share.mean(), ratio_of_means=g.gross.mean() / g.tot.mean() * 100))
res = pd.DataFrame(res)
res["diff"] = (res.mean_of_shares - res.ratio_of_means).abs()
for scope, x in [("baselines", res[res.config == "base"]), ("all 56 configurations", res)]:
    mx = x.loc[x["diff"].idxmax()]
    print(f"{scope}: max |difference| {mx['diff']:.4f} percentage points ({mx.algo} {mx.env} {mx.config} {mx.seg}); segments compared: {len(x)}")

print("done")
