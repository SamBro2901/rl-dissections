"""Read-only checks for Chapter 5 (Step 5). Prints everything to stdout, writes no file.

Reads the context CSVs (contexts/section_5.*_data), flop_analysis/output and results/*.
Usage: python flop_analysis/section_5_checks.py > out.txt
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
C52 = ROOT / "contexts" / "section_5.2_data"
C53 = ROOT / "contexts" / "section_5.3_data"
C54 = ROOT / "contexts" / "section_5.4_data"
OUT = ROOT / "flop_analysis" / "output"
RES = ROOT / "results"
ENVS = ["HalfCheetah-v5", "Ant-v5"]
ALGOS = ["sac", "td3", "mbpo", "tdmpc2"]
CANON = {331, 958, 14577, 43611, 85062}

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


def head(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


d1 = pd.read_csv(C52 / "d1_config_table.csv")
d1["jpf"] = d1["total_E_J"] / d1["total_F"]
cfg = d1.set_index(["env", "algo", "config"])


def E(env, algo, c, seg="total"):
    return cfg.loc[(env, algo, c), f"{seg}_E_J"]


def F(env, algo, c, seg="total"):
    return cfg.loc[(env, algo, c), f"{seg}_F"]


def J(env, algo, c):
    return E(env, algo, c) / F(env, algo, c)


# sweeps: configurations ordered by swept value (the baseline is part of every sweep)
ORDER = {
    ("sac", "UTD"): ["base", "utd2", "utd4"], ("td3", "UTD"): ["base", "utd2", "utd4"], ("mbpo", "UTD"): ["base", "utd2", "utd4"],
    ("sac", "Width"): ["w256", "w512", "base"], ("td3", "Width"): ["w256", "w512", "base"], ("mbpo", "Width"): ["w256", "w512", "base"],
    ("sac", "Batch size"): ["base", "b512", "b1024"], ("td3", "Batch size"): ["base", "b256", "b512", "b1024"],
    ("mbpo", "Batch size"): ["base", "b512", "b1024"],
    ("tdmpc2", "num_q"): ["numq3", "base", "numq7"], ("tdmpc2", "Horizon"): ["horizon1", "base", "horizon5"],
}


def fit_line(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    r = stats.linregress(x, y)
    pred = r.intercept + r.slope * x
    ssr = float(((y - pred) ** 2).sum())
    rse = np.sqrt(ssr / (n - 2)) if n > 2 else float("nan")
    return r.intercept, r.slope, r.rvalue ** 2, rse


def loglog(x, y):
    r = stats.linregress(np.log10(x), np.log10(y))
    return r.slope, r.stderr, r.rvalue ** 2


# ---------------------------------------------------------------------------------------------
head("0. Environment")
print("python", sys.version.split()[0], "| scipy", scipy.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)
print("configs per env:", d1.groupby("env").size().to_dict(), "| per algo/env:", d1.groupby(["env", "algo"]).size().to_dict())

# ---------------------------------------------------------------------------------------------
head("1. Predictability: pooled Spearman and log-log fits (with and without TD-MPC2), discordant pairs")
for env in ENVS:
    for label, sel in [("with TD-MPC2", ALGOS), ("without TD-MPC2", ["sac", "td3", "mbpo"])]:
        d = d1[(d1.env == env) & d1.algo.isin(sel)]
        for seg, (ec, fc) in {"total": ("total_E_J", "total_F"), "rollout": ("rollout_E_J", "rollout_F"),
                              "GU": ("gradient_updates_E_J", "gradient_updates_F")}.items():
            dd = d.dropna(subset=[ec, fc])
            rho, p = stats.spearmanr(dd[fc], dd[ec])
            sl, se, r2 = loglog(dd[fc], dd[ec])
            print(f"{env:15s} {label:16s} {seg:8s} n={len(dd):2d} rho={rho:.4f} p={p:.3g} slope={sl:.4f}+-{se:.4f} R2={r2:.4f} "
                  f"Fspan={np.log10(dd[fc].max() / dd[fc].min()):.2f} Efactor={dd[ec].max() / dd[ec].min():.1f} Ffactor={dd[fc].max() / dd[fc].min():.1f}")
print("\nper algorithm (total):")
for env in ENVS:
    for a in ALGOS:
        dd = d1[(d1.env == env) & (d1.algo == a)]
        rho, p = stats.spearmanr(dd.total_F, dd.total_E_J)
        sl, se, r2 = loglog(dd.total_F, dd.total_E_J)
        print(f"{env:15s} {a:7s} n={len(dd)} rho={rho:.4f} slope={sl:.4f} R2={r2:.4f} Fspan={np.log10(dd.total_F.max() / dd.total_F.min()):.3f}")

print("\nExtreme pair (FLOPs min/max) and energy factors:")
for env in ENVS:
    d = d1[d1.env == env]
    imin, imax = d.total_F.idxmin(), d.total_F.idxmax()
    print(env, "Fmin", d.loc[imin, ["algo", "config"]].tolist(), "Fmax", d.loc[imax, ["algo", "config"]].tolist(),
          "F factor %.1f" % (d.loc[imax, "total_F"] / d.loc[imin, "total_F"]), "E factor %.2f" % (d.loc[imax, "total_E_J"] / d.loc[imin, "total_E_J"]))
    for b in (0.4715, 0.5071):
        pass

print("\nDiscordant pairs (more FLOPs, lower energy). Ties in FLOPs (rel. diff < 1e-6) are excluded and counted.")
disc_rows = []
for env in ENVS:
    d = d1[d1.env == env].reset_index(drop=True)
    for scope in ["pooled"] + ALGOS:
        dd = d if scope == "pooled" else d[d.algo == scope]
        n_pairs = n_disc = n_tie = 0
        worst = (0.0, None)
        for i, j in itertools.combinations(range(len(dd)), 2):
            fi, fj = dd.total_F.iloc[i], dd.total_F.iloc[j]
            if abs(fi - fj) / max(fi, fj) < 1e-6:
                n_tie += 1
                continue
            hi, lo = (i, j) if fi > fj else (j, i)
            n_pairs += 1
            if dd.total_E_J.iloc[hi] < dd.total_E_J.iloc[lo]:
                n_disc += 1
                ratio = dd.total_E_J.iloc[lo] / dd.total_E_J.iloc[hi]
                if ratio > worst[0]:
                    worst = (ratio, (dd.algo.iloc[lo], dd.config.iloc[lo], dd.algo.iloc[hi], dd.config.iloc[hi]))
        disc_rows.append((env, scope, n_pairs, n_disc, n_tie))
        print(f"{env:15s} {scope:7s} pairs={n_pairs:4d} discordant={n_disc:3d} frac={n_disc / max(n_pairs, 1):.4f} ties={n_tie} "
              f"max E(lowF)/E(highF)={worst[0]:.3f} {worst[1]}")
        if scope == "pooled" and n_disc:
            pass
print("\nDiscordant pairs listed (pooled):")
for env in ENVS:
    d = d1[d1.env == env].reset_index(drop=True)
    for i, j in itertools.combinations(range(len(d)), 2):
        fi, fj = d.total_F.iloc[i], d.total_F.iloc[j]
        if abs(fi - fj) / max(fi, fj) < 1e-6:
            continue
        hi, lo = (i, j) if fi > fj else (j, i)
        if d.total_E_J.iloc[hi] < d.total_E_J.iloc[lo]:
            print(f"  {env} more FLOPs: {d.algo.iloc[hi]}:{d.config.iloc[hi]} (F={d.total_F.iloc[hi]:.3e}, E={d.total_E_J.iloc[hi]:.0f}) "
                  f"< {d.algo.iloc[lo]}:{d.config.iloc[lo]} (F={d.total_F.iloc[lo]:.3e}, E={d.total_E_J.iloc[lo]:.0f}) ratio={d.total_E_J.iloc[lo] / d.total_E_J.iloc[hi]:.3f}")

# ---------------------------------------------------------------------------------------------
head("2. Rollout groups: FLOPs and energy of the rollout segment per algorithm")
for env in ENVS:
    for a in ALGOS:
        dd = d1[(d1.env == env) & (d1.algo == a)]
        print(f"{env:15s} {a:7s} rollout F {dd.rollout_F.min():.3e}..{dd.rollout_F.max():.3e}  E {dd.rollout_E_J.min():.0f}..{dd.rollout_E_J.max():.0f} J")

# ---------------------------------------------------------------------------------------------
head("3. Baseline-calibrated estimate (SAC): J/FLOP(base) / J/FLOP(config)")
for env in ENVS:
    jb = J(env, "sac", "base")
    print(env, {c: round(jb / J(env, "sac", c), 4) for c in ["utd2", "utd4", "w256", "w512", "b512", "b1024"]})

# ---------------------------------------------------------------------------------------------
head("4. Fits E = a + bF along each sweep (configuration means), intercept share of baseline energy")
fits = {}
for env in ENVS:
    for a in ALGOS:
        for sw, cs in [(k[1], v) for k, v in ORDER.items() if k[0] == a]:
            x = [F(env, a, c) for c in cs]
            y = [E(env, a, c) for c in cs]
            ic, sl, r2, rse = fit_line(x, y)
            base = E(env, a, "base")
            fits[(env, a, sw, "total")] = (ic, sl, r2, rse)
            print(f"{env:15s} {a:7s} {sw:11s} n={len(cs)} a={ic / 1e3:8.2f} kJ b={sl * 1e12:8.3f} pJ/F R2={r2:.4f} rse={rse / 1e3:.2f} kJ  a/E_base={ic / base * 100:5.1f}%  "
                  f"J/F_base/b={J(env, a, 'base') / sl:.2f}")

print("\nMax/min ratio of the three slopes per algorithm/env, and baseline J/FLOP over slope (E-5.7):")
for env in ENVS:
    for a in ["sac", "td3", "mbpo"]:
        s = {sw: fits[(env, a, sw, "total")][1] for sw in ["UTD", "Width", "Batch size"]}
        jb = J(env, a, "base")
        print(f"{env:15s} {a:5s} slopes pJ: " + ", ".join(f"{k}={v * 1e12:.2f}" for k, v in s.items()) +
              f" | Jbase/slope: " + ", ".join(f"{k}={jb / v:.2f}" for k, v in s.items()) +
              f" | UTD/width={s['UTD'] / s['Width']:.2f} UTD/batch={s['UTD'] / s['Batch size']:.2f} max/min={max(s.values()) / min(s.values()):.2f}")

# ---------------------------------------------------------------------------------------------
head("5. Fits quoted in the prose of 5.2 that are not in tab:ec-fits")
print("5a GU fits (SAC width; MBPO UTD/width/batch); fit of dynamics_model_update (MBPO), rollout fits")
seg_cols = {"GU": ("gradient_updates_E_J", "gradient_updates_F"), "dyn": ("dynamics_model_update_E_J", "dynamics_model_update_F"),
            "roll": ("rollout_E_J", "rollout_F")}
for seg, (ec, fc) in seg_cols.items():
    for env in ENVS:
        for a in ["sac", "td3", "mbpo", "tdmpc2"]:
            if seg == "dyn" and a != "mbpo":
                continue
            if seg == "GU" and a == "tdmpc2":
                continue
            for (aa, sw), cs in ORDER.items():
                if aa != a:
                    continue
                if seg == "roll" and sw in ("UTD", "Batch size") and a != "tdmpc2":
                    continue  # constant regressor
                x = [F(env, a, c, "total") if False else cfg.loc[(env, a, c), fc] for c in cs]
                y = [cfg.loc[(env, a, c), ec] for c in cs]
                if np.isnan(x).any():
                    continue
                if np.ptp(x) == 0:
                    continue
                ic, sl, r2, rse = fit_line(x, y)
                base = cfg.loc[(env, a, "base"), ec]
                print(f"{seg:5s} {env:15s} {a:7s} {sw:11s} a={ic / 1e3:9.2f} kJ b={sl * 1e12:9.3f} pJ/F R2={r2:7.4f} rse={rse / 1e3:7.2f} kJ a/E_base={ic / base * 100:6.1f}%")

print("\n5b Per-seed versus mean fits for MBPO (slope difference), from per_run_segments_5_2.csv")
pr = pd.read_csv(C52 / "per_run_segments_5_2.csv")
seg_run = {"total": ("total_E", "total_F"), "GU": ("gradient_updates_E", "gradient_updates_F"),
           "dyn": ("dynamics_model_update_E", "dynamics_model_update_F"), "roll": ("rollout_E", "rollout_F")}
worst = (0, None)
for env in ENVS:
    for sw in ["UTD", "Width", "Batch size"]:
        cs = ORDER[("mbpo", sw)]
        for seg, (ec, fc) in seg_run.items():
            sub = pr[(pr.algo == "mbpo") & (pr.env == env) & pr.config.isin(cs)]
            if seg == "roll" and sw != "Width":
                continue
            _, sl_run, _, _ = fit_line(sub[fc], sub[ec])
            ccol = {"total": ("total_E_J", "total_F"), "GU": seg_cols["GU"], "dyn": seg_cols["dyn"], "roll": seg_cols["roll"]}[seg]
            _, sl_mean, _, _ = fit_line([cfg.loc[(env, "mbpo", c), ccol[1]] for c in cs], [cfg.loc[(env, "mbpo", c), ccol[0]] for c in cs])
            rel = abs(sl_run - sl_mean) / abs(sl_mean)
            print(f"  {env:15s} {sw:11s} {seg:5s} mean-fit {sl_mean * 1e12:8.3f} per-run {sl_run * 1e12:8.3f} rel.diff {rel * 100:6.1f}%")
            if rel > worst[0]:
                worst = (rel, (env, sw, seg))
print("  max relative slope difference:", f"{worst[0] * 100:.1f}%", worst[1])

print("\n5c Adjacent-value marginal costs (total), pairs with |dF| < 1% of baseline FLOPs dropped")
for env in ENVS:
    for a in ALGOS:
        for (aa, sw), cs in ORDER.items():
            if aa != a:
                continue
            for c0, c1 in zip(cs[:-1], cs[1:]):
                dF = F(env, a, c1) - F(env, a, c0)
                dE = E(env, a, c1) - E(env, a, c0)
                keep = abs(dF) >= 0.01 * F(env, a, "base")
                print(f"  {env:15s} {a:7s} {sw:11s} {c0:>7s}->{c1:<7s} dE/dF={dE / dF * 1e12:9.3f} pJ  {'ok' if keep else 'dropped'}")

# ---------------------------------------------------------------------------------------------
head("6. Ranges per algorithm and environment (energy, J/FLOP), TD-MPC2 overlap")
rng = {}
for env in ENVS:
    for a in ALGOS:
        dd = d1[(d1.env == env) & (d1.algo == a)]
        rng[(env, a)] = (dd.total_E_J.min(), dd.total_E_J.max(), dd.jpf.min(), dd.jpf.max())
        imin, imax = dd.total_E_J.idxmin(), dd.total_E_J.idxmax()
        jmin, jmax = dd.jpf.idxmin(), dd.jpf.idxmax()
        print(f"{env:15s} {a:7s} n={len(dd)} E {dd.total_E_J.min():.0f}({dd.config[imin]})..{dd.total_E_J.max():.0f}({dd.config[imax]}) ratio {dd.total_E_J.max() / dd.total_E_J.min():.3f}"
              f" | J/F {dd.jpf.min() * 1e12:.2f}({dd.config[jmin]})..{dd.jpf.max() * 1e12:.2f}({dd.config[jmax]}) ratio {dd.jpf.max() / dd.jpf.min():.3f}")
    for a in ["sac", "td3", "mbpo"]:
        e0, e1, j0, j1 = rng[(env, a)]
        t0, t1, tj0, tj1 = rng[(env, "tdmpc2")]
        print(f"   overlap {a} vs tdmpc2 ({env}): energy {not (e1 < t0 or t1 < e0)}  jpf {not (j1 < tj0 or tj1 < j0)}")

print("\nMaximum factor of constant J/FLOP: max over configurations of max(Jbase/Jcfg, Jcfg/Jbase) with the algorithm's own baseline")
for env in ENVS:
    for a in ["sac", "td3", "mbpo", "tdmpc2"]:
        dd = d1[(d1.env == env) & (d1.algo == a)]
        jb = J(env, a, "base")
        f = np.maximum(jb / dd.jpf, dd.jpf / jb)
        i = f.idxmax()
        print(f"{env:15s} {a:7s} max factor {f.max():.2f} at {dd.config[i]} (J/F {dd.jpf[i] * 1e12:.2f} vs base {jb * 1e12:.2f})")

# ---------------------------------------------------------------------------------------------
head("7. Ratios of 5.1 and 5.3 from full-precision means")
for env in ENVS:
    eb = {a: E(env, a, "base") for a in ALGOS}
    print(env, "totals kJ:", {a: round(v / 1e3, 3) for a, v in eb.items()},
          "ratio to SAC:", {a: round(eb[a] / eb["sac"], 4) for a in ALGOS},
          "TD-MPC2 rollout/total:", round(cfg.loc[(env, 'tdmpc2', 'base'), 'rollout_E_J'] / eb['tdmpc2'] * 100, 3))

print("\nEnergy distances and J/FLOP ratios along the sweep rows (batch rows use TD3 b256/b512/b1024):")
ROWS = [("Baseline", {"sac": "base", "td3": "base", "mbpo": "base"}), ("UTD 2", {"sac": "utd2", "td3": "utd2", "mbpo": "utd2"}),
        ("UTD 4", {"sac": "utd4", "td3": "utd4", "mbpo": "utd4"}), ("Width 256", {"sac": "w256", "td3": "w256", "mbpo": "w256"}),
        ("Width 512", {"sac": "w512", "td3": "w512", "mbpo": "w512"}), ("Batch 256", {"sac": "base", "td3": "b256", "mbpo": "base"}),
        ("Batch 512", {"sac": "b512", "td3": "b512", "mbpo": "b512"}), ("Batch 1024", {"sac": "b1024", "td3": "b1024", "mbpo": "b1024"})]
for env in ENVS:
    print(env)
    for name, m in ROWS:
        e = {a: E(env, a, m[a]) for a in m}
        j = {a: J(env, a, m[a]) * 1e12 for a in m}
        order_e = " < ".join(sorted(e, key=e.get))
        order_j = " < ".join(sorted(j, key=j.get))
        print(f"  {name:10s} E: {e['td3']:.0f}/{e['sac']:.0f}/{e['mbpo']:.0f}  SAC/TD3={e['sac'] / e['td3']:.3f} MBPO/SAC={e['mbpo'] / e['sac']:.3f} ({order_e}) | "
              f"J/F: S={j['sac']:.2f} T={j['td3']:.2f} M={j['mbpo']:.2f} ({order_j}) TD3/MBPO={j['td3'] / j['mbpo']:.3f} MBPO/TD3={j['mbpo'] / j['td3']:.3f} "
              f"SAC/MBPO={j['sac'] / j['mbpo']:.3f} TD3/SAC={j['td3'] / j['sac']:.3f} SAC/TD3={j['sac'] / j['td3']:.3f} MBPO/SAC={j['mbpo'] / j['sac']:.3f}")
    # J/FLOP rises from width 1024 to 256
    print("   J/F rise width 1024->256:", {a: round(J(env, a, "w256") / J(env, a, "base"), 3) for a in ["sac", "td3", "mbpo"]})
    print("   TD3 batch 100->256: E x%.3f, F x%.3f, J/F x%.3f (J/F falls by %.3f)" % (E(env, "td3", "b256") / E(env, "td3", "base"),
          F(env, "td3", "b256") / F(env, "td3", "base"), J(env, "td3", "b256") / J(env, "td3", "base"), J(env, "td3", "base") / J(env, "td3", "b256")))
    print("   MBPO J/F UTD1->UTD4: %.1f%% ; SAC %.1f%% ; TD3 %.1f%%" % tuple((J(env, a, "utd4") / J(env, a, "base") - 1) * 100 for a in ["mbpo", "sac", "td3"]))
    print("   MBPO J/F ratio to TD3 at base, utd2:", J(env, "mbpo", "base") / J(env, "td3", "base"), J(env, "mbpo", "utd2") / J(env, "td3", "utd2"))
    print("   MBPO: minimal MBPO/SAC distance:", min(E(env, "mbpo", m["mbpo"]) / E(env, "sac", m["sac"]) for _, m in ROWS))
    print("   max over rows of MBPO/SAC:", max(E(env, "mbpo", m["mbpo"]) / E(env, "sac", m["sac"]) for _, m in ROWS),
          " SAC/TD3 min/max:", min(E(env, 'sac', m['sac']) / E(env, 'td3', m['td3']) for _, m in ROWS), max(E(env, 'sac', m['sac']) / E(env, 'td3', m['td3']) for _, m in ROWS))

print("\nSeed-level orderings recomputed from per_run_segments_5_2.csv (energy and J/FLOP by seed):")
cnt = {}
ord_mean_all = {}
for env in ENVS:
    for name, m in ROWS:
        for metric in ["energy", "jpf"]:
            if metric == "energy":
                vm = {a: E(env, a, m[a]) for a in m}
            else:
                vm = {a: J(env, a, m[a]) for a in m}
            om = tuple(sorted(vm, key=vm.get))
            ord_mean_all[(env, name, metric)] = om
            seeds_same = 0
            seed_orders = []
            for s in sorted(CANON):
                vs = {}
                for a in m:
                    r = pr[(pr.algo == a) & (pr.env == env) & (pr.config == m[a]) & (pr.seed == s)]
                    assert len(r) == 1, (env, a, m[a], s, len(r))
                    vs[a] = r.total_E.iloc[0] if metric == "energy" else r.total_E.iloc[0] / r.total_F.iloc[0]
                o = tuple(sorted(vs, key=vs.get))
                seed_orders.append(o)
                seeds_same += o == om
            cnt[(env, name, metric)] = (seeds_same, seed_orders)
tot = {"energy": 0, "jpf": 0}
for (env, name, metric), (k, _) in cnt.items():
    tot[metric] += k
print("seeds whose ordering equals the ordering of the means (of 80):", tot)
for (env, name, metric), (k, so) in cnt.items():
    if k != 5:
        print("  not 5/5:", env, name, metric, k)
# against baseline ordering
for env in ENVS:
    for name, m in ROWS:
        ob = ord_mean_all[(env, "Baseline", "jpf")]
        k = sum(o == ob for o in cnt[(env, name, "jpf")][1])
        print(f"  {env} {name:10s} jpf seeds with baseline ordering: {k}/5 (mean ordering {ord_mean_all[(env, name, 'jpf')]})")
# per-seed ratio TD3/MBPO at width 512 HC
vals = []
for s in sorted(CANON):
    t = pr[(pr.algo == "td3") & (pr.env == "HalfCheetah-v5") & (pr.config == "w512") & (pr.seed == s)].iloc[0]
    mm = pr[(pr.algo == "mbpo") & (pr.env == "HalfCheetah-v5") & (pr.config == "w512") & (pr.seed == s)].iloc[0]
    vals.append((t.total_E / t.total_F) / (mm.total_E / mm.total_F))
print("per-seed TD3/MBPO J/F ratio, width 512 HC:", [round(v, 4) for v in vals], "min/max", min(vals), max(vals),
      " mean-ratio", J("HalfCheetah-v5", "td3", "w512") / J("HalfCheetah-v5", "mbpo", "w512"))

print("\nMBPO rollout-length table (Ant-v5):")
env = "Ant-v5"
for c in ["rollout1", "rollout15", "base"]:
    e, f = E(env, "mbpo", c), F(env, "mbpo", c)
    print(f"  {c:9s} E={e:.0f} F={f:.4e} J/F={e / f * 1e12:.2f} x SAC E {e / E(env, 'sac', 'base'):.3f} x TD3 E {e / E(env, 'td3', 'base'):.3f} "
          f"x SAC J/F {e / f / J(env, 'sac', 'base'):.3f} x TD3 J/F {e / f / J(env, 'td3', 'base'):.3f}")
print("  L1->L25 energy +%.2f%% FLOPs +%.2f%% J/F %+.2f%%" % ((E(env, 'mbpo', 'base') / E(env, 'mbpo', 'rollout1') - 1) * 100,
      (F(env, 'mbpo', 'base') / F(env, 'mbpo', 'rollout1') - 1) * 100, (J(env, 'mbpo', 'base') / J(env, 'mbpo', 'rollout1') - 1) * 100))
print("  MBPO/SAC range", min(E(env, 'mbpo', c) / E(env, 'sac', 'base') for c in ['rollout1', 'rollout15', 'base']), max(E(env, 'mbpo', c) / E(env, 'sac', 'base') for c in ['rollout1', 'rollout15', 'base']))

print("\nTD-MPC2 ranges (tab:rob-tdmpc2):")
for env in ENVS:
    dd = d1[(d1.env == env) & (d1.algo == "tdmpc2")]
    print(env, "E factor %.3f" % (dd.total_E_J.max() / dd.total_E_J.min()), "J/F factor %.3f" % (dd.jpf.max() / dd.jpf.min()))
    for _, r in dd.iterrows():
        print(f"   {r.config:9s} E={r.total_E_J:.0f}+-{r.total_E_sd_J:.0f} J/F={r.jpf * 1e12:.3f} rollout share={r.rollout_E_J / r.total_E_J * 100:.2f}%")

# ---------------------------------------------------------------------------------------------
head("8. FLOP and energy factors along the sweeps (vs baseline of the algorithm; MBPO fit shares)")
for env in ENVS:
    for a in ["sac", "td3", "mbpo"]:
        out = []
        for c in ["w256", "w512", "utd2", "utd4", "b256", "b512", "b1024"]:
            if (env, a, c) not in cfg.index:
                continue
            out.append(f"{c}: F x{F(env, a, c) / F(env, a, 'base'):.4f} E x{E(env, a, c) / E(env, a, 'base'):.4f}")
        print(f"{env:15s} {a:5s} " + " | ".join(out))
    for c in ["w256", "w512", "utd4", "base"]:
        r = cfg.loc[(env, "mbpo", c)]
        print(f"   MBPO {c}: fit FLOP share {r.dynamics_model_update_F / r.total_F * 100:.1f}%  GU FLOP share {r.gradient_updates_F / r.total_F * 100:.1f}% "
              f"GU pJ/F {r.gradient_updates_E_J / r.gradient_updates_F * 1e12:.1f} dyn E {r.dynamics_model_update_E_J / 1e3:.1f} kJ  dyn pJ/F {r.dynamics_model_update_E_J / r.dynamics_model_update_F * 1e12:.1f}")
    print(f"   TD3 batch 100->256: F x{F(env, 'td3', 'b256') / F(env, 'td3', 'base'):.3f}")

# ---------------------------------------------------------------------------------------------
head("9. Section 5.4: paired environment table and derived quantities")
ec = pd.read_csv(C54 / "env_comparison_paired.csv")
print(ec[["algo", "config", "E_total_ratio_of_means", "E_rollout_ratio_of_means", "max_abs_dshare_pp", "max_abs_dshare_segment",
          "P_total_ratio_of_means", "jpf_total_ratio_Ant_over_HC", "flops_total_ratio_Ant_over_HC", "dshare_rollout_mean_pp",
          "dshare_dynamics_model_update_mean_pp", "dshare_gradient_updates_mean_pp", "dshare_world_model_pretrain_mean_pp"]].to_string())
for a in ALGOS:
    s = ec[ec.algo == a]
    print(f"{a:7s} n={len(s)} Etot {s.E_total_ratio_of_means.min():.4f}-{s.E_total_ratio_of_means.max():.4f} Eroll {s.E_rollout_ratio_of_means.min():.4f}-{s.E_rollout_ratio_of_means.max():.4f} "
          f"maxshare {s.max_abs_dshare_pp.min():.4f}-{s.max_abs_dshare_pp.max():.4f} P {s.P_total_ratio_of_means.min():.4f}-{s.P_total_ratio_of_means.max():.4f} "
          f"J/F {s.jpf_total_ratio_Ant_over_HC.min():.4f}-{s.jpf_total_ratio_Ant_over_HC.max():.4f} rollout-share shift {s.dshare_rollout_mean_pp.min():.4f}-{s.dshare_rollout_mean_pp.max():.4f} "
          f"GU shift {s.dshare_gradient_updates_mean_pp.min():.3f}..{s.dshare_gradient_updates_mean_pp.max():.3f}")
print("pairs:", len(ec), ec.groupby("algo").size().to_dict())
print("All: Etot", ec.E_total_ratio_of_means.min(), ec.E_total_ratio_of_means.max(), "Eroll", ec.E_rollout_ratio_of_means.min(), ec.E_rollout_ratio_of_means.max(),
      "P", ec.P_total_ratio_of_means.min(), ec.P_total_ratio_of_means.max(), "J/F", ec.jpf_total_ratio_Ant_over_HC.min(), ec.jpf_total_ratio_Ant_over_HC.max())
print("\nconfigs with J/FLOP ratio > 1:")
print(ec[ec.jpf_total_ratio_Ant_over_HC > 1][["algo", "config", "jpf_total_ratio_Ant_over_HC"]].to_string())
print("count:", int((ec.jpf_total_ratio_Ant_over_HC > 1).sum()), "of", len(ec), "| closest to 1:",
      ec.assign(d=(ec.jpf_total_ratio_Ant_over_HC - 1).abs()).nsmallest(4, "d")[["algo", "config", "jpf_total_ratio_Ant_over_HC"]].to_dict("records"))
print("\nsign-flip p total energy / rollout share all equal 0.0625?", (ec.signflip_p_total_energy == 0.0625).all(), (ec.signflip_p_rollout_share == 0.0625).all())
print("rollout-share shift positive in all pairs:", (ec.dshare_rollout_mean_pp > 0).all(), " same sign in 5/5:", (ec.dshare_rollout_same_sign_k_of_5 == 5).all())
print("\nrollout share shift (rollout) per algorithm: min/max", ec.groupby("algo").dshare_rollout_mean_pp.agg(["min", "max"]).to_dict("index"))
print("tdmpc2 world_model_pretrain share shifts:", ec[ec.algo == "tdmpc2"].dshare_world_model_pretrain_mean_pp.round(4).tolist())
print("mbpo GU shifts:", ec[ec.algo == "mbpo"].dshare_gradient_updates_mean_pp.round(3).tolist(), " fit shifts:", ec[ec.algo == "mbpo"].dshare_dynamics_model_update_mean_pp.round(3).tolist())
print("\nrollout energies base (J): HC/Ant and difference")
for a in ALGOS:
    h, an = E("HalfCheetah-v5", a, "base", "rollout"), E("Ant-v5", a, "base", "rollout")
    print(f"  {a:7s} HC {h / 1e3:.3f} kJ  Ant {an / 1e3:.3f} kJ  diff {(an - h) / 1e3:.3f} kJ  ratio {an / h:.4f}")
d_sac = E("Ant-v5", "sac", "base", "rollout") - E("HalfCheetah-v5", "sac", "base", "rollout")
d_tm = E("Ant-v5", "tdmpc2", "base", "rollout") - E("HalfCheetah-v5", "tdmpc2", "base", "rollout")
print("  TD-MPC2 diff / SAC diff = %.1f" % (d_tm / d_sac))
print("  GU energy ratios per pair would be in the CSV via E_total; GU ratio (base): ",
      {a: round(E("Ant-v5", a, "base", "gradient_updates") / E("HalfCheetah-v5", a, "base", "gradient_updates"), 4) for a in ALGOS})
gur = {}
for a in ALGOS:
    v = []
    for c in sorted(set(d1[(d1.algo == a) & (d1.env == "HalfCheetah-v5")].config)):
        if (("Ant-v5", a, c) in cfg.index):
            v.append(E("Ant-v5", a, c, "gradient_updates") / E("HalfCheetah-v5", a, c, "gradient_updates"))
    gur[a] = (round(min(v), 4), round(max(v), 4))
print("  GU energy ratio ranges over all configs:", gur)
print("  MBPO fit energy/FLOP ratio (base): E %.3f F %.3f" % (E("Ant-v5", "mbpo", "base", "dynamics_model_update") / E("HalfCheetah-v5", "mbpo", "base", "dynamics_model_update"),
      F("Ant-v5", "mbpo", "base", "dynamics_model_update") / F("HalfCheetah-v5", "mbpo", "base", "dynamics_model_update")))
for a in ALGOS:
    print(f"  {a} FLOP ratio GU (base) {F('Ant-v5', a, 'base', 'gradient_updates') / F('HalfCheetah-v5', a, 'base', 'gradient_updates'):.4f}")

print("\nTime ratios per paired configuration (c5_rollout_wallclock_per_1000_steps.csv):")
c5 = pd.read_csv(C54 / "c5_rollout_wallclock_per_1000_steps.csv")
print(c5.columns.tolist())
print(c5.head(3).to_string())

# ---------------------------------------------------------------------------------------------
head("10. Matched MBPO pairing: Ant-v5 rollout1 vs HalfCheetah-v5 base (paired by seed)")
a_ = pr[(pr.algo == "mbpo") & (pr.env == "Ant-v5") & (pr.config == "rollout1")].set_index("seed")
h_ = pr[(pr.algo == "mbpo") & (pr.env == "HalfCheetah-v5") & (pr.config == "base")].set_index("seed")
seeds = sorted(set(a_.index) & set(h_.index))
print("seeds paired:", seeds)
ratios = np.array([a_.loc[s, "total_E"] / h_.loc[s, "total_E"] for s in seeds])
diff = np.array([a_.loc[s, "total_E"] - h_.loc[s, "total_E"] for s in seeds])
print("per-seed total ratios", ratios.round(4), "mean %.4f sd %.5f" % (ratios.mean(), ratios.std(ddof=1)))
print("ratio of means %.4f" % (a_.loc[seeds, "total_E"].mean() / h_.loc[seeds, "total_E"].mean()))
pat = 0
obs = abs(diff.mean())
cnt_ge = 0
for signs in itertools.product([1, -1], repeat=len(diff)):
    cnt_ge += abs((np.array(signs) * np.abs(diff)).mean()) >= obs - 1e-9
print("sign-flip p (two-sided, mean difference) = %.4f over %d patterns" % (cnt_ge / 2 ** len(diff), 2 ** len(diff)))
ab = pr[(pr.algo == "mbpo") & (pr.env == "Ant-v5") & (pr.config == "base")].set_index("seed")
r25 = np.array([ab.loc[s, "total_E"] / h_.loc[s, "total_E"] for s in seeds])
print("baseline pairing (L25): per-seed ratios", r25.round(4), "mean %.4f sd %.5f (CSV E_total_paired_sd = %.5f)" % (r25.mean(), r25.std(ddof=1),
      ec[(ec.algo == "mbpo") & (ec.config == "baseline")].E_total_paired_sd.iloc[0]))

# ---------------------------------------------------------------------------------------------
head("11. Run counts (B3) and metadata for the commit claims (B4)")
meta = []
for p in RES.glob("*/*/seed_*/*/metadata.json"):
    m = json.load(open(p))
    meta.append(dict(algo=p.parts[-5], env=p.parts[-4], seed=m["seed"], dir=Path(*p.parts[-5:-1]).as_posix(), commit=m["git_commit"],
                     start=m["start_time_utc"], episodic=m.get("algo_config", {}).get("episodic"), algo_config=m.get("algo_config", {})))
meta = pd.DataFrame(meta)
meta["canon"] = meta.seed.isin(CANON)
print(meta.groupby(["algo", "env", "canon"]).size().to_string())
print("canonical total:", int(meta.canon.sum()), " all:", len(meta))
print("non-canonical runs:", meta[~meta.canon][["dir", "seed", "start"]].to_string())
print(meta[(meta.algo == "tdmpc2")].groupby(["env", "canon", "episodic"]).size().to_string())
print("\nMBPO HalfCheetah-v5 baseline runs (canonical): commits and start dates")
pr_runs = pd.read_csv(OUT / "per_run_energy_per_flop.csv")
tot = pr_runs[(pr_runs.segment == "TOTAL_MEASURED_TRAINING") & pr_runs.included_in_cross_seed_avg].copy()
tot["dir"] = tot.run_dir.str.replace("\\", "/", regex=False).str.replace("results/", "", n=1, regex=False)
j = tot.merge(meta[["dir", "commit", "start"]], on="dir", how="left")
sel = j[(j.algo == "mbpo") & (((j.env_id == "HalfCheetah-v5") & (j.mbpo_rollout_regime == "rl1-1_re20-150") & (j.updates_per_env_step == 1) & j.architecture_signature.str.startswith("bs256_h1024")) |
                              ((j.env_id == "Ant-v5") & (j.mbpo_rollout_regime == "rl1-1_re20-100")))]
print(sel[["env_id", "seed", "mbpo_rollout_regime", "commit", "start"]].sort_values(["env_id", "seed"]).to_string())

# ---------------------------------------------------------------------------------------------
head("12. Episode statistics at the baselines (B5)")


def baseline_dirs(env, algo):
    s = tot[(tot.algo == algo) & (tot.env_id == env) & (tot.updates_per_env_step == 1)]
    if algo == "sac":
        s = s[s.architecture_signature == "bs256_h1024x1024"]
    elif algo == "td3":
        s = s[s.architecture_signature == "bs100_h1024x1024"]
    elif algo == "mbpo":
        s = s[s.architecture_signature == "bs256_h1024x1024_ens7_mh200x200x200x200_mb256"]
        s = s[s.mbpo_rollout_regime == ("rl1-1_re20-150" if env == "HalfCheetah-v5" else "rl1-25_re20-100")]
    else:
        s = s[s.architecture_signature.str.contains("_h3_") & s.architecture_signature.str.contains("_nq5_")]
    return s


epstats = {}
for env in ENVS:
    for a in ALGOS:
        s = baseline_dirs(env, a)
        n_ep, lens_all, med_per_seed, mean_per_seed = [], [], [], []
        for d in s.run_dir:
            tm = json.load(open(ROOT / d / "training_metrics.json"))
            eps = [e for e in tm["episodes"] if e["phase"] == "train"]
            n_ep.append(len(eps))
            L = np.array([e["length"] for e in eps])
            lens_all.extend(L.tolist())
            med_per_seed.append(np.median(L))
            mean_per_seed.append(L.mean())
        epstats[(env, a)] = (n_ep, lens_all, med_per_seed, mean_per_seed)
        la = np.array(lens_all)
        print(f"{env:15s} {a:7s} runs={len(s)} train episodes min/median/max={min(n_ep)}/{np.median(n_ep):.0f}/{max(n_ep)} "
              f"| length pooled median={np.median(la):.1f} mean={la.mean():.1f} | median over seeds of per-seed median={np.median(med_per_seed):.1f} "
              f"(range {min(med_per_seed):.0f}-{max(med_per_seed):.0f}) of per-seed mean={np.median(mean_per_seed):.1f}")

# ---------------------------------------------------------------------------------------------
head("13. Returns (B6): mean return of the last 10% of the training episodes per configuration")
cm = pd.read_csv(OUT / "correlation" / "config_means.csv")
for env in ENVS:
    for a in ALGOS:
        s = cm[(cm.env == env) & (cm.algo == a)]
        lo, hi = s.loc[s.return_mean.idxmin()], s.loc[s.return_mean.idxmax()]
        neg = s[s.return_mean < 0]
        print(f"{env:15s} {a:7s} n={len(s)} min {lo.return_mean:9.1f} ({lo.config_id.split('|', 1)[1]}) max {hi.return_mean:9.1f} ({hi.config_id.split('|', 1)[1]}) negative: {len(neg)}")
        if len(neg):
            print("      negative configs:", [c.split("|", 1)[1] for c in neg.config_id])
print("negative on Ant-v5 total:", int(((cm.env == "Ant-v5") & (cm.return_mean < 0)).sum()), "of", int((cm.env == "Ant-v5").sum()))
print("per algo:", cm[(cm.env == "Ant-v5") & (cm.return_mean < 0)].groupby("algo").size().to_dict())

# ---------------------------------------------------------------------------------------------
head("14. Further point checks (rounding, TD-MPC2 segments, time ratios, per-call FLOP ratios)")
print("SAC Ant UTD 4 J/FLOP [pJ] %.5f ; TD3 width 256 / baseline J/FLOP: HC %.5f Ant %.5f ; TD3 batch 100->256 J/FLOP fall: HC %.5f Ant %.5f" % (
    J("Ant-v5", "sac", "utd4") * 1e12, J("HalfCheetah-v5", "td3", "w256") / J("HalfCheetah-v5", "td3", "base"),
    J("Ant-v5", "td3", "w256") / J("Ant-v5", "td3", "base"), J("HalfCheetah-v5", "td3", "base") / J("HalfCheetah-v5", "td3", "b256"),
    J("Ant-v5", "td3", "base") / J("Ant-v5", "td3", "b256")))
print("MBPO HalfCheetah-v5 batch 512 FLOPs %.6e, UTD 2 FLOPs %.6e" % (F("HalfCheetah-v5", "mbpo", "b512"), F("HalfCheetah-v5", "mbpo", "utd2")))
print("MBPO Ant-v5 J/FLOP change between rollout lengths: 15->25 %.3f%%, 1->25 %.3f%%" % (
    (J("Ant-v5", "mbpo", "base") / J("Ant-v5", "mbpo", "rollout15") - 1) * 100, (J("Ant-v5", "mbpo", "base") / J("Ant-v5", "mbpo", "rollout1") - 1) * 100))
mb = ec[(ec.algo == "mbpo") & (ec.config == "baseline")].iloc[0]
print("MBPO baseline Pr/PGU ratio of ratios %.6f (paired mean %.6f)" % (mb.Proll_over_PGU_double_ratio_Ant_over_HC, mb.Proll_over_PGU_double_ratio_paired_mean))
print("Pr/PGU ratio Ant/HC below 1 in", int((ec.Proll_over_PGU_double_ratio_Ant_over_HC < 1).sum()), "of", len(ec))
print("TD-MPC2: gradient-update block vs rollout (energy)")
for env in ENVS:
    for c in ["numq3", "base", "numq7", "horizon1", "horizon5"]:
        r = cfg.loc[(env, "tdmpc2", c)]
        print(f"  {env:15s} {c:9s} rollout {r.rollout_E_J / r.total_E_J * 100:5.2f}% GU {r.gradient_updates_E_J / r.total_E_J * 100:5.2f}% GU larger: {r.gradient_updates_E_J > r.rollout_E_J}")
print("GU FLOP ratio (Ant/HC) ranges over paired configurations:")
for a in ALGOS:
    v = [cfg.loc[("Ant-v5", a, c), "gradient_updates_F"] / cfg.loc[("HalfCheetah-v5", a, c), "gradient_updates_F"]
         for c in sorted(set(d1[(d1.algo == a) & (d1.env == "HalfCheetah-v5")].config)) if ("Ant-v5", a, c) in cfg.index]
    print(f"  {a:7s} {min(v):.4f}-{max(v):.4f}")
c5 = pd.read_csv(C54 / "c5_rollout_wallclock_per_1000_steps.csv").set_index(["algo", "env", "config"])
tr = [(a, c, r.rollout_s_per_1000_steps_mean / c5.loc[(a, "HalfCheetah-v5", c), "rollout_s_per_1000_steps_mean"])
      for (a, en, c), r in c5.iterrows() if en == "Ant-v5" and (a, "HalfCheetah-v5", c) in c5.index]
print("rollout time ratios (Ant/HC) per algorithm:", pd.DataFrame(tr, columns=["algo", "config", "ratio"]).groupby("algo").ratio.agg(["min", "max", "count"]).round(4).to_dict("index"))
c3 = pd.read_csv(C54 / "c3_per_call_flop_ratios.csv")
print(c3[c3.algo.isin(["sac", "mbpo", "tdmpc2"])][["algo", "quantity", "ratio_Ant_over_HC"]].round(4).to_string())
