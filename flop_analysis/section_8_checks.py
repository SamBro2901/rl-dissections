"""Read-only checks for Chapter 8 (Step 8, Conclusion and Outlook). Prints everything to stdout, writes no file.

Reads the run files of results/*/*/seed_*/<timestamp>/ (metadata.json, emissions_base_*.csv, training_metrics.json),
flop_analysis/output/per_run_energy_per_flop.csv (run -> configuration map and FLOP counts) and the repository tree.
Self-contained: the functions of section_7_checks.py (no __main__ guard) are copied, not imported.
Usage: python flop_analysis/section_8_checks.py > out.txt
Sections (named in the output): 0 environment, 1 segment shares (RQ1), 2 RQ2 statistics, 3 net analysis and RQ3 statistics,
4 RQ4 (the two clock runs), 5 repository facts, 6 counts for the ripple edits.
"""
import glob
import itertools
import json
import math
import re
import subprocess
import sys
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
MEASURED4 = ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "gradient_updates"]

warnings.simplefilter("ignore", FutureWarning)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)


def head(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


def parse(row):
    """Configuration label of a run (copy of section_7_checks.parse)."""
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


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True).stdout.strip()
    except Exception as exc:  # git not available
        return f"<git failed: {exc}>"


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
print("canonical runs:", len(RUNS), "| configurations:", RUNS.groupby(["algo", "env", "config"]).ngroups,
      "| per environment:", RUNS.groupby("env").apply(lambda x: x.groupby(["algo", "config"]).ngroups).to_dict())
META = {rd: json.load(open(ROOT / rd / "metadata.json")) for rd in RUNS.index}

cols = ["task_name", "duration", "ram_power", "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed"]
parts = []
for rd in RUNS.index:
    f = glob.glob(str(ROOT / rd / "emissions_base_*.csv"))
    assert len(f) == 1, (rd, f)
    d = pd.read_csv(f[0])[cols]
    d["run_dir"] = rd
    parts.append(d)
T = pd.concat(parts, ignore_index=True)
T["seg"] = T.task_name.str.replace(r"_\d+$", "", regex=True)
T["E_J"] = T.energy_consumed * J
T["ram_J"] = T.ram_energy * J
T = T.join(RUNS[["algo", "env", "config", "seed"]], on="run_dir")
NI = T[~T.seg.isin(IDLE)]

# ---------------------------------------------------------------------------------------------
head("1. RQ1: segment shares (mean of per-run shares), seed spread, rollout share")
MT = NI[NI.seg.isin(MEAS)]
rs = MT.groupby(["run_dir", "seg"]).E_J.sum().unstack().fillna(0)
rs["total"] = rs.sum(axis=1)
rs = rs.join(RUNS[["algo", "env", "config", "seed"]])
print("max rel. diff task-sum vs pipeline TOTAL_MEASURED_TRAINING: %.2e" % ((rs.total - RUNS.E_pipeline).abs() / RUNS.E_pipeline).max())
SEGS = [s for s in MEAS if s in rs.columns]
for s in SEGS:
    rs["sh_" + s] = rs[s] / rs.total * 100
bl = rs[rs.config == "base"]
print("-- baselines: mean of per-run shares [%] / sample sd (ddof=1) [pp] / population sd (ddof=0) [pp]")
sdmax = {}
for (a, e), g in bl.groupby(["algo", "env"]):
    line = f"{a:7s} {e:15s}"
    for s in SEGS:
        x = g["sh_" + s]
        if x.sum() == 0:
            continue
        line += f" | {s[:14]} {x.mean():7.3f} sd {x.std(ddof=1):.3f}/{x.std(ddof=0):.3f}"
        sdmax[(a, e, s)] = x.std(ddof=1)
    print(line)
q = {("sac", "gradient_updates"), ("td3", "gradient_updates"), ("tdmpc2", "gradient_updates"), ("tdmpc2", "rollout")}
print("max sample sd of the quoted shares, SAC/TD3/TD-MPC2 (GU; TD-MPC2 GU and rollout): %.4f pp" % max(v for (a, e, s), v in sdmax.items() if (a, s) in q))
print("max sample sd of the quoted shares, MBPO (fit and GU): %.4f pp" % max(v for (a, e, s), v in sdmax.items() if a == "mbpo" and s in ("dynamics_model_update", "gradient_updates")))
print("max sample sd over ALL segments: SAC/TD3/TD-MPC2 %.4f pp ; MBPO %.4f pp" % (
    max(v for (a, e, s), v in sdmax.items() if a != "mbpo"), max(v for (a, e, s), v in sdmax.items() if a == "mbpo")))
print("max population sd of the quoted shares: SAC/TD3/TD-MPC2 %.4f, MBPO %.4f" % (
    max(g["sh_" + s].std(ddof=0) for (a, e), g in bl.groupby(["algo", "env"]) for s in ("gradient_updates", "rollout") if a != "mbpo"),
    max(g["sh_" + s].std(ddof=0) for (a, e), g in bl.groupby(["algo", "env"]) for s in ("gradient_updates", "dynamics_model_update") if a == "mbpo")))

roll_b = bl.groupby(["algo", "env"]).sh_rollout.mean()
rb3 = roll_b.drop("tdmpc2", level=0)
print("\nrollout share at the baselines [%]:", roll_b.round(3).to_dict())
print("(a) baselines of SAC/TD3/MBPO: min %.3f (%s) max %.3f (%s)" % (rb3.min(), rb3.idxmin(), rb3.max(), rb3.idxmax()))
sel = rs[rs.algo.isin(["sac", "td3", "mbpo"])].groupby(["algo", "env", "config"]).sh_rollout.mean()
print("(b) all %d configurations of SAC/TD3/MBPO: min %.3f (%s) max %.3f (%s)" % (len(sel), sel.min(), sel.idxmin(), sel.max(), sel.idxmax()))
selr = rs[rs.algo.isin(["sac", "td3", "mbpo"])].groupby(["algo", "env", "config"]).apply(lambda x: x.rollout.sum() / x.total.sum() * 100)
print("    same as ratio of sums: min %.3f max %.3f" % (selr.min(), selr.max()))
print("    all configs (config means) with rollout share > 10 %:", sel[sel > 10].round(3).to_dict())

print("\nTD-MPC2, all configurations: shares [%] (mean of per-run shares)")
tm = rs[rs.algo == "tdmpc2"].groupby(["env", "config"])[["sh_" + s for s in SEGS]].mean()
tm["roll_gt_GU"] = tm.sh_rollout > tm.sh_gradient_updates
tm["GU_gt_roll"] = tm.sh_gradient_updates > tm.sh_rollout
print(tm.round(3).to_string())
print("TD-MPC2: configurations", len(tm), "| rollout is the largest single segment:",
      int((tm[["sh_" + s for s in SEGS if s != "synthetic_rollout_generation" and s != "dynamics_model_update"]].idxmax(axis=1) == "sh_rollout").sum()),
      "| gradient_updates share larger than rollout share in:", int(tm.GU_gt_roll.sum()), "of", len(tm))

print("\nTD-MPC2: is the rollout larger than every single allocated sub-segment of the gradient updates (segment_energy.json, per run)?")
sub = ["buffer_sample", "critic_update", "actor_update", "target_update"]
big = []
for rd in RUNS[RUNS.algo == "tdmpc2"].index:
    se = json.load(open(ROOT / rd / "segment_energy.json"))
    big.append(dict(env=RUNS.env[rd], config=RUNS.config[rd], roll=se["rollout"], maxsub=max(se[s] for s in sub), arg=max(sub, key=lambda s: se[s])))
big = pd.DataFrame(big)
big["roll_larger"] = big.roll > big.maxsub
print(big.groupby(["env", "config"]).agg(runs=("roll", "size"), roll_larger=("roll_larger", "sum"), largest_sub=("arg", lambda x: ",".join(sorted(set(x))))).to_string())
print("runs in which the rollout exceeds every allocated sub-segment:", int(big.roll_larger.sum()), "of", len(big),
      "| configurations in which it does so in all five seeds:", int((big.groupby(["env", "config"]).roll_larger.all()).sum()), "of 10")

# ---------------------------------------------------------------------------------------------
head("2. RQ2: configuration means, pooled statistics, discordant pairs")
M = NI[NI.seg.isin(MEAS)].copy()
RS0 = M.groupby(["run_dir", "seg"]).E_J.sum()
RT0 = RS0.groupby("run_dir").sum().rename("gross").to_frame().join(RUNS)
CT0 = RT0.groupby(["env", "algo", "config"]).agg(E=("gross", "mean"), F=("F", "mean"), n=("seed", "size"))
CT0["jpf"] = CT0.E / CT0.F * 1e12
d1 = pd.read_csv(ROOT / "contexts" / "section_5.2_data" / "d1_config_table.csv").set_index(["env", "algo", "config"])
cmp_ = CT0.join(d1[["total_E_J", "total_F"]])
print("configurations:", len(CT0), "| max rel. diff energy vs d1_config_table %.2e, FLOPs %.2e" % (
    ((cmp_.E - cmp_.total_E_J).abs() / cmp_.total_E_J).max(), ((cmp_.F - cmp_.total_F).abs() / cmp_.total_F).max()))
for e in ENVS:
    c = CT0.loc[e]
    rho = stats.spearmanr(c.F, c.E)
    lr = stats.linregress(np.log10(c.F), np.log10(c.E))
    imin, imax = c.jpf.idxmin(), c.jpf.idxmax()
    print(f"{e:15s} n={len(c)} rho={rho[0]:.4f} slope={lr.slope:.4f} R2={lr.rvalue ** 2:.4f} | J/FLOP min {c.jpf.min():.2f} pJ {imin} max {c.jpf.max():.2f} pJ {imax} factor {c.jpf.max() / c.jpf.min():.2f}")
    print("   baselines J/FLOP [pJ]:", {a: round(c.loc[(a, 'base'), 'jpf'], 2) for a in ALGOS})
print("\nDiscordant pairs (pooled, ties in FLOPs excluded; gross config means)")
for e in ENVS:
    d = CT0.loc[e].reset_index()
    n_pairs = n_disc = n_tie = 0
    for i, j in itertools.combinations(range(len(d)), 2):
        fi, fj = d.F[i], d.F[j]
        if abs(fi - fj) / max(fi, fj) < 1e-6:
            n_tie += 1
            continue
        hi, lo = (i, j) if fi > fj else (j, i)
        n_pairs += 1
        n_disc += d.E[hi] < d.E[lo]
    print(f"{e:15s} pairs {n_pairs} discordant {n_disc} ({n_disc / n_pairs * 100:.2f} %) concordant {n_pairs - n_disc} ({(n_pairs - n_disc) / n_pairs * 100:.2f} %) FLOP ties {n_tie}")

# ---------------------------------------------------------------------------------------------
head("3. Net analysis (definitions of 7.1.3 / section_7_checks.py) and RQ3 statistics")
ID = {}
for ph in ("head", "tail"):
    r = T[T.seg == f"idle_baseline_{ph}"].set_index("run_dir")
    ID[ph + "_corr"] = r.E_J / (r.ram_J / r.ram_power)
ID = pd.DataFrame(ID)
ID["mean"] = (ID.head_corr + ID.tail_corr) / 2
M = M.join(ID, on="run_dir")
M["gross"] = M.E_J
M["net_mean"] = M.E_J - M["mean"] * M.duration
M["net_head"] = M.E_J - M.head_corr * M.duration
M["net_tail"] = M.E_J - M.tail_corr * M.duration
RS = M.groupby(["run_dir", "seg"])[VARIANTS].sum()
RT = RS.groupby("run_dir").sum().join(RUNS)
for v in VARIANTS:
    RT[v + "_roll"] = RS.xs("rollout", level="seg")[v]
print("check gross total vs pipeline: max rel diff %.2e" % ((RT.gross - RT.E_pipeline).abs() / RT.E_pipeline).max())
print("run-segments with negative net (mean):", int((RS.net_mean < 0).sum()), "of", len(RS), "| negative run totals:", int((RT[VARIANTS] < 0).sum().sum()))
CT = RT.groupby(["env", "algo", "config"]).agg(**{v: (v, "mean") for v in VARIANTS}, F=("F", "mean"), n=("seed", "size"))
for v in VARIANTS:
    CT["jpf_" + v] = CT[v] / CT.F * 1e12

print("\n-- reproduction of tab:thr-gross-net (baselines; expected from Step 7, not evidence)")
rows = []
for a in ALGOS:
    for e in ENVS:
        c = CT.loc[(e, a, "base")]
        rows.append(dict(algo=a, env=e, gross_kJ=c.gross / 1e3, net_kJ=c.net_mean / 1e3, ratio=c.net_mean / c.gross, jpf_gross=c.jpf_gross, jpf_net=c.jpf_net_mean))
GN = pd.DataFrame(rows)
print(GN.round(3).to_string())
print("net/gross range %.3f - %.3f" % (GN.ratio.min(), GN.ratio.max()))

SW = [("Baseline", dict(sac="base", td3="base", mbpo="base")), ("UTD 2", dict(sac="utd2", td3="utd2", mbpo="utd2")),
      ("UTD 4", dict(sac="utd4", td3="utd4", mbpo="utd4")), ("Width 256", dict(sac="w256", td3="w256", mbpo="w256")),
      ("Width 512", dict(sac="w512", td3="w512", mbpo="w512")), ("Batch 256", dict(sac="base", td3="b256", mbpo="base")),
      ("Batch 512", dict(sac="b512", td3="b512", mbpo="b512")), ("Batch 1024", dict(sac="b1024", td3="b1024", mbpo="b1024"))]
ABB = dict(sac="S", td3="T", mbpo="M")
RTi = RT.reset_index()
PAIRS = [("td3", "sac"), ("sac", "mbpo"), ("td3", "mbpo")]


def seed_vals(env, algo, cfg, v, metric):
    d = RTi[(RTi.env == env) & (RTi.algo == algo) & (RTi.config == cfg)].set_index("seed")
    return d[v] if metric == "energy" else d[v] / d.F * 1e12


summary = {}
for v in VARIANTS:
    for metric in ("energy", "jpf"):
        base_order = {}
        for e in ENVS:
            for name, sel_ in SW:
                vals = {a: CT.loc[(e, a, c), v if metric == "energy" else "jpf_" + v] for a, c in sel_.items()}
                order = sorted(vals, key=vals.get)
                if name == "Baseline":
                    base_order[e] = order
                sv = {a: seed_vals(e, a, c, v, metric) for a, c in sel_.items()}
                n_eq = sum(1 for s in CANON if sorted(sv, key=lambda a: sv[a][s]) == order)
                ratios = [vals[order[i + 1]] / vals[order[i]] for i in range(2)]
                summary[(v, metric, e, name)] = dict(order="".join(ABB[a] for a in order), same_as_base=order == base_order[e], n_eq=n_eq, min_ratio=min(ratios),
                                                     vals=vals, pair={(a, b): (vals[b] / vals[a], sum(1 for s in CANON if sv[b][s] > sv[a][s])) for a, b in PAIRS})
print("\n-- RQ3 orderings per convention (tab:rob-energy, tab:rob-jpf)")
for v in VARIANTS:
    for metric in ("energy", "jpf"):
        items = {k[2:]: r for k, r in summary.items() if k[0] == v and k[1] == metric}
        occur = sorted({r["order"] for r in items.values()})
        per_env = {e: sum(1 for (ee, n), r in items.items() if ee == e and n != "Baseline" and not r["same_as_base"]) for e in ENVS}
        keep = {e: [n for (ee, n), r in items.items() if ee == e and n != "Baseline" and r["same_as_base"]] for e in ENVS}
        print(f"{v:9s} {metric:6s}: distinct orderings {len(occur)} {occur} | non-baseline values differing from the baseline ordering {per_env} (of 7 per environment), keeping it: {keep} "
              f"| seed-level orderings equal to the order of the means: {sum(r['n_eq'] for r in items.values())} of {5 * len(items)} | smallest neighbour ratio {min(r['min_ratio'] for r in items.values()):.4f}")
print("\n-- detail per row (order, seeds equal, min ratio)")
for metric in ("energy", "jpf"):
    for e in ENVS:
        for name, _ in SW:
            print(f"{metric:6s} {e:15s} {name:11s} " + " | ".join(
                f"{v}: {summary[(v, metric, e, name)]['order']} {summary[(v, metric, e, name)]['n_eq']}/5 {summary[(v, metric, e, name)]['min_ratio']:.4f}" for v in VARIANTS))
print("\n-- gross values")
for metric in ("energy", "jpf"):
    for e in ENVS:
        for name, _ in SW:
            vals = summary[("gross", metric, e, name)]["vals"]
            print(f"{metric:6s} {e:15s} {name:11s} " + "  ".join(f"{a}={vals[a]:.5g}" for a in ("sac", "td3", "mbpo")))
print("\n-- pair TD3--MBPO (ratio MBPO/TD3 of the J/FLOP, seeds of 5 with MBPO > TD3) and SAC--TD3 at the batch-256 row")
for (e, name, a, b) in [(HC, "Width 512", "td3", "mbpo"), (ANT, "Width 512", "td3", "mbpo"), (HC, "Batch 256", "td3", "sac"), (ANT, "Batch 256", "td3", "sac"),
                        (HC, "Baseline", "td3", "sac"), (ANT, "Baseline", "td3", "sac")]:
    line = f"jpf {e:15s} {name:10s} {b}/{a}: "
    for v in VARIANTS:
        ratio, n = summary[(v, "jpf", e, name)]["pair"][(a, b)]
        line += f"{v}={ratio:.4f} ({n}/5)  "
    print(line)
print("\n-- ranges and overlap with TD-MPC2 (J/FLOP)")
for v in VARIANTS:
    for e in ENVS:
        c = CT.loc[e]
        rg = c.groupby(level="algo")["jpf_" + v].agg(["min", "max", "size"])
        t = rg.loc["tdmpc2"]
        ov = {a: bool(rg.loc[a, "min"] <= t["max"] and t["min"] <= rg.loc[a, "max"]) for a in ["sac", "td3", "mbpo"]}
        print(f"{v:9s} {e:15s} " + "  ".join(f"{a}:[{rg.loc[a, 'min']:.2f},{rg.loc[a, 'max']:.2f}] overlap={ov[a]}" for a in ["sac", "td3", "mbpo"]) + f"  tdmpc2:[{t['min']:.2f},{t['max']:.2f}]")
print("\n-- pooled over all configurations of an environment")
for v in VARIANTS:
    for e in ENVS:
        c = CT.loc[e]
        rho = stats.spearmanr(c.F, c[v])[0]
        lr = stats.linregress(np.log10(c.F), np.log10(c[v]))
        sp = c["jpf_" + v].max() / c["jpf_" + v].min()
        print(f"{v:9s} {e:15s} n={len(c)} rho={rho:.4f} slope={lr.slope:.4f} R2={lr.rvalue ** 2:.4f} J/FLOP spread {sp:.3f} (min {c['jpf_' + v].min():.2f} pJ {c['jpf_' + v].idxmin()}, max {c['jpf_' + v].max():.2f} pJ {c['jpf_' + v].idxmax()})")
print("\n-- TD-MPC2: energy factor horizon 5 / horizon 1 and rollout-share deviation along its sweeps")
rsc = rs[rs.algo == "tdmpc2"].groupby(["env", "config"]).sh_rollout.mean()
for e in ENVS:
    line = f"{e:15s}"
    for v in VARIANTS:
        line += f" | {v} {CT.loc[(e, 'tdmpc2', 'horizon5'), v] / CT.loc[(e, 'tdmpc2', 'horizon1'), v]:.4f}"
    dev = (rsc.loc[e] - rsc.loc[e, "base"]).abs()
    print(line + f" | max |rollout share - baseline| = {dev.max():.3f} pp ({dev.idxmax()}); all: {dev.round(3).to_dict()}")
print("\n-- environment effect: Ant over HalfCheetah")
for v in VARIANTS:
    rr = RTi[["env", "algo", "config", "seed", v, v + "_roll"]]
    h = rr[rr.env == HC].set_index(["algo", "config", "seed"])
    a = rr[rr.env == ANT].set_index(["algo", "config", "seed"])
    j = h.join(a, lsuffix="_hc", rsuffix="_ant", how="inner")
    cfg_ratio = j[v + "_ant"].groupby(level=[0, 1]).mean() / j[v + "_hc"].groupby(level=[0, 1]).mean()
    roll_ratio = j[v + "_roll_ant"].groupby(level=[0, 1]).mean() / j[v + "_roll_hc"].groupby(level=[0, 1]).mean()
    seed_ratio = j[v + "_ant"] / j[v + "_hc"]
    rsh = (j[v + "_roll_ant"].groupby(level=[0, 1]).mean() / j[v + "_ant"].groupby(level=[0, 1]).mean()
           - j[v + "_roll_hc"].groupby(level=[0, 1]).mean() / j[v + "_hc"].groupby(level=[0, 1]).mean())
    print(f"{v:9s} pairs {len(cfg_ratio)} seed-pairs {len(seed_ratio)} | total ratio {cfg_ratio.min():.4f} - {cfg_ratio.max():.4f} | configs > 1: {int((cfg_ratio > 1).sum())}/{len(cfg_ratio)} "
          f"| seed pairs > 1: {int((seed_ratio > 1).sum())}/{len(seed_ratio)} | rollout ratio {roll_ratio.min():.4f} - {roll_ratio.max():.4f} (>1 in {int((roll_ratio > 1).sum())}) "
          f"| rollout-share rise positive in {int((rsh > 0).sum())}/{len(rsh)} (min {rsh.min() * 100:.3f} pp)")
    print("     configurations with ratio <= 1.05:", {k: round(x, 4) for k, x in cfg_ratio[cfg_ratio <= 1.05].items()},
          "| seed pairs < 1 by configuration:", {k: int(x) for k, x in (seed_ratio < 1).groupby(level=[0, 1]).sum().items() if x > 0})

# ---------------------------------------------------------------------------------------------
head("4. RQ4: the two clock runs and the five canonical width-256 runs")
BASE0 = RES / "mbpo" / "HalfCheetah-v5" / "seed_0"
RUN2 = {"R2000": BASE0 / "20260904_122958", "R200": BASE0 / "20260904_130915"}


def load(run_dir):
    em = pd.read_csv(glob.glob(str(run_dir / "emissions_base_*.csv"))[0])
    em["prefix"] = em["task_name"].str.replace(r"_\d+$", "", regex=True)
    em["energy_j"] = em["energy_consumed"] * J
    return dict(dir=run_dir, meta=json.load(open(run_dir / "metadata.json")), tm=json.load(open(run_dir / "training_metrics.json")), em=em)


def totals(r):
    d = r["em"][r["em"]["prefix"].isin(MEASURED4)]
    return dict(energy=d.energy_j.sum(), duration=d.duration.sum())


def idle_floor(r):
    out = {}
    for ph in ("head", "tail"):
        row = r["em"][r["em"]["task_name"] == f"idle_baseline_{ph}"].iloc[0]
        win = float(row["ram_energy"]) * J / float(row["ram_power"])
        out[ph] = float(row["energy_consumed"]) * J / win
    out["mean"] = (out["head"] + out["tail"]) / 2
    return out


def work(r):
    ep, eps = r["tm"]["epochs"], r["tm"]["episodes"]
    trn = [e["return"] for e in eps if e["phase"] == "train"]
    n10 = max(1, len(trn) // 10)
    fits = [e["model_train_epochs"] for e in ep if e.get("model_train_epochs") is not None]
    steps = 0
    for x in ep:
        n = x["buffer_size"] - 1000
        steps += x["model_train_epochs"] * math.ceil((n - max(1, int(n * 0.2))) / 256)
    return dict(fit_epochs=sum(fits), opt_steps=steps, last10=float(np.mean(trn[-n10:])))


R = {k: load(v) for k, v in RUN2.items()}
Tt = {k: totals(r) for k, r in R.items()}
Fl = {k: idle_floor(r) for k, r in R.items()}
Wk = {k: work(r) for k, r in R.items()}
for k in R:
    rel = str(RUN2[k].relative_to(ROOT)).replace("\\", "/")
    flops = pr[(pr.run_dir == rel) & (pr.segment == "TOTAL_MEASURED_TRAINING")].total_flops.iloc[0]
    print(f"{k}: energy {Tt[k]['energy'] / 1e3:.4f} kJ | duration {Tt[k]['duration']:.3f} s | power {Tt[k]['energy'] / Tt[k]['duration']:.4f} W | FLOPs {flops:.4e} | {Wk[k]}"
          f" | floor head/tail/mean {Fl[k]['head']:.3f}/{Fl[k]['tail']:.3f}/{Fl[k]['mean']:.3f} W | hidden {R[k]['meta']['algo_config']['hidden_sizes']} | commit {R[k]['meta']['git_commit'][:7]}")
    Wk[k]["flops"] = flops
rE = Tt["R200"]["energy"] / Tt["R2000"]["energy"]
rD = Tt["R200"]["duration"] / Tt["R2000"]["duration"]
rP = (Tt["R200"]["energy"] / Tt["R200"]["duration"]) / (Tt["R2000"]["energy"] / Tt["R2000"]["duration"])
print(f"ratios R200/R2000: energy {rE:.5f} | duration {rD:.5f} | power {rP:.5f}")
fac = {}
for lab in ("mean", "head", "tail"):
    ab = {k: Tt[k]["energy"] - Fl[k][lab] * Tt[k]["duration"] for k in R}
    fac[lab] = ab["R200"] / ab["R2000"]
    print(f"above-idle factor with {lab} floor: {fac[lab]:.4f}")
print("above-idle factor range over the three idle choices: %.3f - %.3f" % (min(fac.values()), max(fac.values())))
can = RUNS[(RUNS.algo == "mbpo") & (RUNS.env == HC) & (RUNS.config == "w256")]
cr = []
for rd in can.index:
    r = load(ROOT / rd)
    w = work(r)
    cr.append(dict(run=rd, seed=can.seed[rd], energy=totals(r)["energy"], flops=can.F[rd], fit_epochs=w["fit_epochs"], opt_steps=w["opt_steps"], last10=w["last10"]))
cr = pd.DataFrame(cr)
print(cr.to_string())
print(f"canonical width-256 runs: n={len(cr)} energy mean {cr.energy.mean() / 1e3:.4f} kJ sample sd {cr.energy.std(ddof=1) / 1e3:.4f} kJ population sd {cr.energy.std(ddof=0) / 1e3:.4f} | "
      f"FLOPs {cr.flops.min():.4e} - {cr.flops.max():.4e} | fit epochs {cr.fit_epochs.min()} - {cr.fit_epochs.max()} | optimiser steps {cr.opt_steps.min()} - {cr.opt_steps.max()} | return {cr.last10.min():.1f} - {cr.last10.max():.1f}")
print(f"R2000: FLOPs {Wk['R2000']['flops']:.4e} fit epochs {Wk['R2000']['fit_epochs']} optimiser steps {Wk['R2000']['opt_steps']} return {Wk['R2000']['last10']:.1f}")
print("R2000 seeds: R2000/R200 seed", R["R2000"]["meta"]["seed"], R["R200"]["meta"]["seed"], "| canonical seeds", sorted(cr.seed.tolist()))

# ---------------------------------------------------------------------------------------------
head("5. Repository facts (read-only)")
for pat in ["algorithms/*.py", "configs/overrides/*.json", "flop_analysis/*.py"]:
    print(pat, "->", sorted(p.name for p in ROOT.glob(pat)))
print("top-level files:", sorted(p.name for p in ROOT.iterdir() if p.is_file()))
print("LICENSE* in repository root:", sorted(p.name for p in ROOT.glob("LICENSE*")))
print("run_experiment.py exists:", (ROOT / "run_experiment.py").exists(), "| experiment_runner.py exists:", (ROOT / "experiment_runner.py").exists(),
      "| flop_analysis/flop_keys.py exists:", (ROOT / "flop_analysis" / "flop_keys.py").exists(), "| verify_analytic_flops.py exists:", (ROOT / "flop_analysis" / "verify_analytic_flops.py").exists())
for fn in ["ALGO_CONFIGS", "_dispatch_train"]:
    hits = [(str(p.relative_to(ROOT)), i + 1) for p in ROOT.glob("*.py") for i, ln in enumerate(p.read_text(errors="ignore").splitlines()) if fn in ln]
    print(f"occurrences of {fn} in top-level *.py:", hits[:8])
print("git remote:", git("config", "--get", "remote.origin.url"), "| tags:", repr(git("tag")), "| HEAD:", git("rev-parse", "--short", "HEAD"))
print("git status --short:", git("status", "--short").splitlines())
print("git log -1:", git("log", "-1", "--format=%h %ad %s", "--date=short"))
com = pd.Series({rd: META[rd]["git_commit"] for rd in RUNS.index})
print("distinct git_commit values over the 280 canonical metadata.json:", com.nunique(), sorted(c[:7] for c in com.unique()))
allm = [json.load(open(p)) for p in RES.rglob("metadata.json")]
print("metadata.json files in results/:", len(allm), "| distinct commits over all:", len({m["git_commit"] for m in allm}))
print("PPO or PETS in algorithms/ (file names):", [p.name for p in (ROOT / "algorithms").glob("*") if re.search(r"ppo|pets", p.name, re.I)])
hits = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py") if "results" not in p.parts and "__pycache__" not in p.parts and re.search(r"\b(PPO|PETS)\b", p.read_text(errors="ignore"))]
print("python files mentioning PPO/PETS:", hits)
print("\nparameter counts (analytic, dense layers with bias; SAC policy outputs mean and log-std):")
DIMS = {HC: (17, 6), ANT: (105, 8)}
for e, (o, a) in DIMS.items():
    for w in (1024, 512, 256):
        pol = o * w + w + w * w + w + w * 2 * a + 2 * a
        crit = (o + a) * w + w + w * w + w + w + 1
        print(f"  {e:15s} width {w:4d}: SAC policy {pol:8d}  critic {crit:8d}")
print("hidden sizes recorded (canonical): ", RUNS.assign(h=[tuple(META[rd]["algo_config"].get("hidden_sizes", [])) for rd in RUNS.index]).groupby(["algo", "h"]).size().to_dict())

readme = (ROOT / "README.md").read_text(errors="ignore")
print("README.md names:", {t: (t in readme) for t in ["run_experiment.py", "experiment_runner.py", "configs/overrides", "algorithms/sac.py", "algorithms/td3.py",
                                                       "algorithms/mbpo.py", "algorithms/tdmpc2.py", "flop_keys.py", "Extending to PPO"]})
cfgtxt = (ROOT / "configs" / "config.py").read_text(errors="ignore")
print("ALGO_CONFIGS defined in configs/config.py:", "ALGO_CONFIGS = {" in cfgtxt, "| keys:", re.findall(r'"(\w+)": \w+Config', cfgtxt[cfgtxt.index("ALGO_CONFIGS = {"):][:200]))

# ---------------------------------------------------------------------------------------------
head("5b. Further checks for the limitation and Outlook statements")
syn = rs[rs.algo == "mbpo"].groupby(["env", "config"]).sh_synthetic_rollout_generation.mean()
print("MBPO synthetic_rollout_generation share of the training total [%%], config means: min %.4f max %.4f (%s) over %d configurations" % (
    syn.min(), syn.max(), syn.idxmax(), len(syn)))
print("   per-run maximum: %.4f" % rs[rs.algo == "mbpo"].sh_synthetic_rollout_generation.max())
LR = []
for rd, r in RUNS.iterrows():
    tmj = json.load(open(ROOT / rd / "training_metrics.json"))
    ep = [e for e in tmj["episodes"] if e["phase"] == "train"]
    k = max(1, len(ep) // 10)
    LR.append(dict(algo=r.algo, env=r.env, config=r.config, seed=r.seed, first=np.mean([e["return"] for e in ep[:k]]), last=np.mean([e["return"] for e in ep[-k:]])))
LR = pd.DataFrame(LR)
LC = LR.groupby(["env", "algo", "config"]).agg(first=("first", "mean"), last=("last", "mean"))
LC["flag"] = LC["last"] <= LC["first"]
print("Ant-v5 configurations with last-tenth return <= first-tenth return:", int(LC.loc[ANT].flag.sum()), "of", len(LC.loc[ANT]),
      "| by algorithm:", LC.loc[ANT][LC.loc[ANT].flag].groupby(level="algo").size().to_dict(), "| HalfCheetah-v5 flagged:", int(LC.loc[HC].flag.sum()))
print("MBPO Ant-v5 last-tenth return negative in", int((LC.loc[(ANT, "mbpo")]["last"] < 0).sum()), "of", len(LC.loc[(ANT, "mbpo")]),
      "| TD-MPC2 Ant-v5 last-tenth return range %.1f - %.1f" % (LC.loc[(ANT, "tdmpc2")]["last"].min(), LC.loc[(ANT, "tdmpc2")]["last"].max()))


def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, p + k + "."))
        else:
            o[p + k] = v
    return o


fa = flat(R["R2000"]["meta"])
print("R2000 against each canonical width-256 MBPO run: differing keys of metadata.algo_config / experiment_config (the two blocks only):")
for rd in can.index:
    fc = flat(META[rd])
    keys = [k for k in sorted(set(fa) | set(fc)) if (k.startswith("algo_config.") or k.startswith("experiment_config.")) and fa.get(k, "<absent>") != fc.get(k, "<absent>")]
    print(f"   seed {can.seed[rd]}: {keys}")
print("commit of R2000", R["R2000"]["meta"]["git_commit"][:7], "| in the 12 canonical commits:", R["R2000"]["meta"]["git_commit"] in set(com.unique()))
print("canonical width-256 commits:", sorted({META[rd]['git_commit'][:7] for rd in can.index}))

# ---------------------------------------------------------------------------------------------
head("6. Counts for the ripple edits")
t0 = pd.Series({rd: META[rd].get("thermal_gate", {}).get("run_start_temp_c") for rd in RUNS.index})
print("runs with run_start_temp_c (temperature before the head window):", int(t0.notna().sum()), "of", len(t0))
x = pd.concat([ID.head_corr, t0.rename("t0")], axis=1).dropna()
print("r(head idle power corrected, run_start_temp_c) = %.4f (n=%d)" % (x.corr().iloc[0, 1], len(x)))
print("done")
