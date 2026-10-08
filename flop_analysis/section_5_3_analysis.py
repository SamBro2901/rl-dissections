"""
Section 5.3 analysis: robustness of the SAC / TD3 / MBPO ordering across sweeps; MBPO rollout-length sweep;
TD-MPC2 sweep range.  (read-only on results/ and flop_analysis/output/)

Reuses load() / parse() of section_5_2_analysis.py (canonical seeds, 280 runs, TOTAL_MEASURED_TRAINING).
Output: contexts/section_5.3_data/*.csv  (rendered to contexts/section_5.3.md by section_5_3_render.py)

Definitions
  * per-seed energy E = TOTAL_MEASURED_TRAINING gross joules; per-seed J/FLOP = E / F (same run).
  * config value: mean E over 5 seeds (sd ddof=1); J/FLOP = mean E / mean F (ratio of means), sd = sd of per-seed ratios.
  * ordering = algorithms sorted lowest -> highest on the config value; neighbour ratio = higher / lower.
  * seed-level: per seed, sort the algorithms on that seed's own value (paired seeds, same seed number for all algos).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import section_5_2_analysis as s52  # noqa: E402

REPO = s52.REPO
OUT = REPO / "contexts" / "section_5.3_data"
OUT.mkdir(parents=True, exist_ok=True)
ENVS = s52.ENVS
CANON = s52.CANON
ORD_ALGOS = ["sac", "td3", "mbpo"]
METRICS = {"energy": "E_J", "jpf": "J_per_F"}

# sweep value -> config label of each algorithm in the ordering
SWEEPS = [
    ("baseline", "base", {"sac": "base", "td3": "base", "mbpo": "base"}),
    ("utd2", "UTD 2", {a: "utd2" for a in ORD_ALGOS}),
    ("utd4", "UTD 4", {a: "utd4" for a in ORD_ALGOS}),
    ("w256", "width 256", {a: "w256" for a in ORD_ALGOS}),
    ("w512", "width 512", {a: "w512" for a in ORD_ALGOS}),
    ("b256", "batch 256", {"sac": "base", "td3": "b256", "mbpo": "base"}),   # SAC/MBPO baseline batch is 256
    ("b512", "batch 512", {a: "b512" for a in ORD_ALGOS}),
    ("b1024", "batch 1024", {a: "b1024" for a in ORD_ALGOS}),
]


def per_run_values(runs):
    r = runs.copy()
    r["E_J"] = r.total_E
    r["J_per_F"] = r.total_E / r.total_F
    return r


def cfg_value(g, metric):
    """(mean, sd) of config for metric over the seeds of g."""
    if metric == "E_J":
        return g.total_E.mean(), g.total_E.std(ddof=1)
    return g.total_E.mean() / g.total_F.mean(), (g.total_E / g.total_F).std(ddof=1)


def main():
    runs = per_run_values(s52.load())
    assert runs.groupby(["algo", "env", "config"]).seed.nunique().eq(5).all()
    runs["algo"] = runs.algo.astype(str)

    def grp(algo, env, cfg):
        g = runs[(runs.algo == algo) & (runs.env == env) & (runs.config == cfg)]
        assert len(g) == 5, (algo, env, cfg, len(g))
        return g

    # ---- 1/2: orderings ----
    ord_rows, nb_rows, seed_rows, swap_rows, pair_rows = [], [], [], [], []
    ordering = {}   # (env, sweep, metric) -> tuple of algos low->high
    for env in ENVS:
        for key, label, cfgmap in SWEEPS:
            vals = {}
            for a in ORD_ALGOS:
                g = grp(a, env, cfgmap[a])
                for m, col in METRICS.items():
                    vals[(a, m)] = cfg_value(g, col)
            for m, col in METRICS.items():
                order = sorted(ORD_ALGOS, key=lambda a: vals[(a, m)][0])
                ordering[(env, key, m)] = tuple(order)
                for rank, a in enumerate(order, 1):
                    mean, sd = vals[(a, m)]
                    prev = vals[(order[rank - 2], m)][0] if rank > 1 else np.nan
                    ord_rows.append(dict(env=env, sweep=key, sweep_label=label, metric=m, rank=rank, algo=a,
                                         config=cfgmap[a], mean=mean, sd=sd, cv_pct=100 * sd / mean,
                                         ratio_to_previous=mean / prev if rank > 1 else np.nan,
                                         ordering=" < ".join(order)))
    # reference row: TD3 batch-100 baseline, in the batch sweeps
    ref_rows = []
    for env in ENVS:
        g = grp("td3", env, "base")
        for m, col in METRICS.items():
            mean, sd = cfg_value(g, col)
            ref_rows.append(dict(env=env, metric=m, algo="td3", config="base (batch 100, warmup 10000)", mean=mean, sd=sd))
    # TD-MPC2 baseline next to baseline ordering (not part of it)
    tm_rows = []
    for env in ENVS:
        g = grp("tdmpc2", env, "base")
        for m, col in METRICS.items():
            mean, sd = cfg_value(g, col)
            tm_rows.append(dict(env=env, metric=m, algo="tdmpc2", config="base", mean=mean, sd=sd))
    pd.DataFrame(ord_rows).to_csv(OUT / "o1_orderings.csv", index=False)
    pd.DataFrame(ref_rows + tm_rows).to_csv(OUT / "o1b_reference_rows.csv", index=False)

    # ---- 3: swaps relative to the baseline ordering ----
    def pairs_order(order):  # set of (a,b) with a below b
        return {(order[i], order[j]) for i in range(3) for j in range(i + 1, 3)}
    for env in ENVS:
        for key, label, _ in SWEEPS:
            for m in METRICS:
                base = ordering[(env, "baseline", m)]
                cur = ordering[(env, key, m)]
                bp, cp = pairs_order(base), pairs_order(cur)
                swapped = sorted(bp - cp)
                swap_rows.append(dict(env=env, sweep=key, metric=m, baseline_ordering=" < ".join(base),
                                      ordering=" < ".join(cur), same_as_baseline=(base == cur),
                                      swapped_pairs="; ".join(f"{a}<{b} -> {b}<{a}" for a, b in swapped) or "none"))
    pd.DataFrame(swap_rows).to_csv(OUT / "o2_swaps_vs_baseline.csv", index=False)

    # ---- 4: seed-level ----
    for env in ENVS:
        for key, label, cfgmap in SWEEPS:
            per = {}
            for a in ORD_ALGOS:
                g = grp(a, env, cfgmap[a]).set_index("seed").loc[CANON]
                per[a] = g
            for m, col in METRICS.items():
                v = {a: per[a][col] for a in ORD_ALGOS}
                own = ordering[(env, key, m)]
                base = ordering[(env, "baseline", m)]
                n_own = n_base = 0
                seed_orders = []
                for s in CANON:
                    o = tuple(sorted(ORD_ALGOS, key=lambda a: v[a][s]))
                    seed_orders.append(" < ".join(o))
                    n_own += o == own
                    n_base += o == base
                rec = dict(env=env, sweep=key, metric=m, mean_ordering=" < ".join(own),
                           seeds_same_as_mean_ordering=n_own, baseline_mean_ordering=" < ".join(base),
                           seeds_same_as_baseline_ordering=n_base)
                for s, so in zip(CANON, seed_orders):
                    rec[f"seed_{s}"] = so
                seed_rows.append(rec)
                for a, b in itertools.combinations(ORD_ALGOS, 2):
                    lo, hi = (a, b) if own.index(a) < own.index(b) else (b, a)
                    pair_rows.append(dict(env=env, sweep=key, metric=m, lower_algo=lo, higher_algo=hi,
                                          seeds_lower_below_higher=int(sum(v[lo][s] < v[hi][s] for s in CANON)),
                                          min_seed_ratio=float((v[hi] / v[lo]).min()), max_seed_ratio=float((v[hi] / v[lo]).max())))
    pd.DataFrame(seed_rows).to_csv(OUT / "o3_seed_level.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "o3b_pairwise_seed_counts.csv", index=False)

    # ---- 5: MBPO rollout-length sweep (Ant only) ----
    roll = []
    env = "Ant-v5"
    comp = {m: {a: cfg_value(grp(a, env, "base"), col) for a in ["sac", "td3"]} for m, col in METRICS.items()}
    for cfg, lmax in [("rollout1", 1), ("rollout15", 15), ("base", 25)]:
        g = grp("mbpo", env, cfg)
        r = dict(env=env, config=cfg, rollout_max_length=lmax, n_seeds=5, F_mean=g.total_F.mean(), F_min=g.total_F.min(), F_max=g.total_F.max())
        for m, col in METRICS.items():
            mean, sd = cfg_value(g, col)
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            for a in ["sac", "td3"]:
                r[f"{m}_ratio_to_{a}_base"] = mean / comp[m][a][0]
            allv = {"sac": comp[m]["sac"][0], "td3": comp[m]["td3"][0], "mbpo_this": mean}
            r[f"{m}_ordering_with_sac_td3_base"] = " < ".join(sorted(allv, key=allv.get)).replace("mbpo_this", f"mbpo(L{lmax})")
            # seed-level vs SAC/TD3 baselines (paired seeds)
            for a in ["sac", "td3"]:
                b = grp(a, env, "base").set_index("seed").loc[CANON][col].to_numpy()
                x = g.set_index("seed").loc[CANON][col].to_numpy()
                r[f"{m}_seeds_mbpo_above_{a}"] = int((x > b).sum())
        roll.append(r)
    pd.DataFrame(roll).to_csv(OUT / "r1_mbpo_rollout_sweep.csv", index=False)

    # ---- 6: TD-MPC2 sweep + other algos' ranges ----
    tm = []
    for env in ENVS:
        for cfg, nq, hz in [("base", 5, 3), ("numq3", 3, 3), ("numq7", 7, 3), ("horizon1", 5, 1), ("horizon5", 5, 5)]:
            g = grp("tdmpc2", env, cfg)
            share = g.rollout_E / g.total_E
            r = dict(env=env, config=cfg, num_q=nq, horizon=hz, E_mean=g.total_E.mean(), E_sd=g.total_E.std(ddof=1),
                     F_mean=g.total_F.mean(), jpf_mean=g.total_E.mean() / g.total_F.mean(),
                     jpf_sd=(g.total_E / g.total_F).std(ddof=1),
                     rollout_E_mean=g.rollout_E.mean(), rollout_share_of_total_mean=share.mean(),
                     rollout_share_of_total_sd=share.std(ddof=1),
                     rollout_share_ratio_of_means=g.rollout_E.mean() / g.total_E.mean())
            tm.append(r)
    tm = pd.DataFrame(tm)
    tm.to_csv(OUT / "t1_tdmpc2_sweep.csv", index=False)

    ctx = []
    cfgvals = []
    for (a, env, cfg), g in runs.groupby(["algo", "env", "config"]):
        cfgvals.append(dict(algo=a, env=env, config=cfg, E_mean=g.total_E.mean(), jpf=g.total_E.mean() / g.total_F.mean()))
    cfgvals = pd.DataFrame(cfgvals)
    cfgvals.to_csv(OUT / "t2_all_config_values.csv", index=False)
    for env in ENVS:
        t = tm[tm.env == env]
        tmr = dict(E_min=t.E_mean.min(), E_max=t.E_mean.max(), J_min=t.jpf_mean.min(), J_max=t.jpf_mean.max())
        for a in ["sac", "td3", "mbpo", "tdmpc2"]:
            c = cfgvals[(cfgvals.algo == a) & (cfgvals.env == env)]
            r = dict(env=env, algo=a, n_configs=len(c),
                     E_min=c.E_mean.min(), E_argmin=c.loc[c.E_mean.idxmin(), "config"],
                     E_max=c.E_mean.max(), E_argmax=c.loc[c.E_mean.idxmax(), "config"],
                     jpf_min=c.jpf.min(), jpf_argmin=c.loc[c.jpf.idxmin(), "config"],
                     jpf_max=c.jpf.max(), jpf_argmax=c.loc[c.jpf.idxmax(), "config"])
            if a != "tdmpc2":
                r["E_range_overlaps_tdmpc2"] = bool(r["E_min"] <= tmr["E_max"] and tmr["E_min"] <= r["E_max"])
                r["jpf_range_overlaps_tdmpc2"] = bool(r["jpf_min"] <= tmr["J_max"] and tmr["J_min"] <= r["jpf_max"])
            ctx.append(r)
    pd.DataFrame(ctx).to_csv(OUT / "t3_algo_ranges_vs_tdmpc2.csv", index=False)

    # ---- X: consistency checks ----
    chk = []
    d1 = pd.read_csv(REPO / "contexts" / "section_5.2_data" / "d1_config_table.csv")  # itself validated against Ch.4 in 5.2
    for _, r in cfgvals.iterrows():
        d = d1[(d1.algo == r.algo) & (d1.env == r.env) & (d1.config == r.config)].iloc[0]
        chk.append(dict(source="section_5.2 d1_config_table", algo=r.algo, env=r.env, config=r.config,
                        rel_dev_E=abs(r.E_mean - d.total_E_J) / d.total_E_J, rel_dev_jpf=abs(r.jpf - d.total_J_per_F) / d.total_J_per_F))
    cs = pd.read_csv(REPO / "flop_analysis" / "output" / "cross_seed_energy_per_flop.csv")
    # Ch.4.4 TD-MPC2 rollout share
    c44 = pd.read_csv(REPO / "contexts" / "Section_4.4_data" / "tdmpc2_cross_seed_summary.csv")
    for _, r in tm.iterrows():
        h = {"horizon1": "hor1", "horizon5": "hor5"}.get(r.config, r.config)
        m = c44[(c44.config == h) & (c44.env == r.env) & (c44.segment == "rollout")]
        if len(m):
            chk.append(dict(source="section_4.4 tdmpc2_cross_seed_summary rollout share", algo="tdmpc2", env=r.env, config=r.config,
                            rel_dev_E=abs(r.rollout_E_mean - m.iloc[0].energy_j_mean) / m.iloc[0].energy_j_mean,
                            rel_dev_jpf=abs(r.rollout_share_of_total_mean * 100 - m.iloc[0].share_of_total_pct_mean) / m.iloc[0].share_of_total_pct_mean))
        else:
            chk.append(dict(source="section_4.4 (no matching rollout row)", algo="tdmpc2", env=r.env, config=r.config))
    # Ch.4 TOTAL via pipeline cross-seed CSV
    tot = cs[cs.segment == "TOTAL_MEASURED_TRAINING"]
    chk_df = pd.DataFrame(chk)
    chk_df.to_csv(OUT / "x1_consistency_checks.csv", index=False)
    print("max rel dev vs 5.2 d1:", chk_df[chk_df.source.str.startswith("section_5.2")][["rel_dev_E", "rel_dev_jpf"]].max().to_dict())
    print("4.4 rows:\n", chk_df[chk_df.source.str.startswith("section_4.4")])
    print("cross_seed total rows:", len(tot))


if __name__ == "__main__":
    main()
