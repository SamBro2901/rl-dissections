"""
Section 5.2 analysis: how well do FLOPs predict energy?  (read-only on results/ and flop_analysis/output/)

Input : flop_analysis/output/per_run_energy_per_flop.csv (canonical seeds only, 280 runs, 56 configs)
Output: contexts/section_5.2_data/*.csv  (+ console summary used by contexts/section_5.2.md)

Definitions
  * energy = gross joules (no idle subtraction); config value = mean over the 5 canonical seeds.
  * segments with matmul FLOPs: rollout, gradient_updates (GU), dynamics_model_update, synthetic_rollout_generation,
    world_model_pretrain; total = TOTAL_MEASURED_TRAINING.
  * GU energy = buffer_sample + critic_update + actor_update + target_update energy (the directly measured
    gradient_updates task, which the pipeline splits into those four allocated parts; no separate GU row exists);
    GU FLOPs = critic_update + actor_update matmul FLOPs.
  * J/FLOP = mean energy / mean FLOPs (ratio of means); sd = sample sd (ddof=1) of the five per-seed ratios.
  * Spearman: scipy.stats.spearmanr; log-log OLS: scipy.stats.linregress(log10 F, log10 E).
  * Linear fit E = a + b F: ordinary least squares on config-mean points (one point per sweep value); also repeated
    on per-run points (5 per config) for the comparison with the Chapter 4 fits.
  * delta pairs: adjacent sweep values, dE/dF between config means (as flop_dashboard.compute_delta_pairs, 'prev' mode,
    pairs with |dF| < 1% of reference FLOPs dropped).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
PER_RUN = REPO / "flop_analysis" / "output" / "per_run_energy_per_flop.csv"
CROSS = REPO / "flop_analysis" / "output" / "cross_seed_energy_per_flop.csv"
OUT = REPO / "contexts" / "section_5.2_data"
CANON = [331, 958, 14577, 43611, 85062]
ENVS = ["HalfCheetah-v5", "Ant-v5"]
SEGS = ["rollout", "gradient_updates", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain"]
BASE_BATCH = {"sac": 256, "mbpo": 256, "td3": 100}


def parse(row) -> dict:
    sig = row.architecture_signature
    algo = row.algo
    d = dict(algo=algo, env=row.env_id)
    if algo == "tdmpc2":
        h = int(re.search(r"_h(\d+)_ns", sig).group(1))
        nq = int(re.search(r"_nq(\d+)", sig).group(1))
        label = "base" if (h == 3 and nq == 5) else (f"numq{nq}" if nq != 5 else f"horizon{h}")
        d.update(width=None, batch=int(re.match(r"bs(\d+)", sig).group(1)), utd=int(row.updates_per_env_step),
                 horizon=h, num_q=nq, rollout_regime=None)
    else:
        batch = int(re.match(r"bs(\d+)", sig).group(1))
        width = int(re.search(r"_h(\d+)x", sig).group(1))
        utd = int(row.updates_per_env_step)
        reg = row.mbpo_rollout_regime if algo == "mbpo" else None
        parts = []
        if width != 1024: parts.append(f"w{width}")
        if batch != BASE_BATCH[algo]: parts.append(f"b{batch}")
        if utd != 1: parts.append(f"utd{utd}")
        if algo == "mbpo":
            base_reg = "rl1-1_re20-150" if row.env_id == "HalfCheetah-v5" else "rl1-25_re20-100"
            if reg != base_reg:
                parts.append("rollout" + reg.split("_")[0][2:].split("-")[1])
        label = "+".join(parts) or "base"
        d.update(width=width, batch=batch, utd=utd, horizon=None, num_q=None, rollout_regime=reg)
    d["config"] = label
    return d


def load() -> pd.DataFrame:
    df = pd.read_csv(PER_RUN)
    df = df[df.seed.isin(CANON) & df.env_id.isin(ENVS)].copy()
    assert df.run_dir.nunique() == 280
    meta = {k: parse(g.iloc[0]) for k, g in df.groupby(["algo", "env_id", "architecture_signature", "updates_per_env_step", "mbpo_rollout_regime"], dropna=False)}
    df["_k"] = list(zip(df.algo, df.env_id, df.architecture_signature, df.updates_per_env_step, df.mbpo_rollout_regime.where(df.mbpo_rollout_regime.notna(), np.nan)))
    # build per-run segment table
    recs = []
    for run, g in df.groupby("run_dir"):
        r0 = g.iloc[0]
        d = parse(r0)
        d.update(seed=int(r0.seed), run_dir=run)
        s = g.set_index("segment")
        tot = s.loc["TOTAL_MEASURED_TRAINING"]
        d["total_E"], d["total_F"] = tot.total_energy_joules, tot.total_flops
        gu = ["buffer_sample", "critic_update", "actor_update", "target_update"]
        d["gradient_updates_E"] = sum(s.loc[x, "total_energy_joules"] for x in gu)
        d["gradient_updates_F"] = s.loc["critic_update", "total_flops"] + s.loc["actor_update", "total_flops"]
        for seg in ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain"]:
            if seg in s.index:
                d[f"{seg}_E"], d[f"{seg}_F"] = s.loc[seg, "total_energy_joules"], s.loc[seg, "total_flops"]
        recs.append(d)
    runs = pd.DataFrame(recs)
    # reconciliation: TOTAL energy = sum of measured segments
    parts = runs[[f"{s}_E" for s in SEGS]].sum(axis=1, min_count=1)
    runs["recon_err"] = (parts - runs.total_E).abs() / runs.total_E
    return runs


def ratio_stats(g: pd.DataFrame, seg: str):
    e, f = g[f"{seg}_E"], g[f"{seg}_F"]
    jpf = e.mean() / f.mean()
    sd = (e / f).std(ddof=1)
    return e.mean(), e.std(ddof=1), f.mean(), jpf, sd


def config_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["algo", "env", "config", "width", "batch", "utd", "horizon", "num_q", "rollout_regime"]
    for k, g in runs.groupby(keys, dropna=False, sort=False):
        d = dict(zip(keys, k))
        d["n_seeds"] = len(g)
        for seg in ["total"] + SEGS:
            if f"{seg}_E" not in g or g[f"{seg}_E"].isna().all():
                continue
            e, esd, f, jpf, sd = ratio_stats(g, seg)
            d[f"{seg}_E_J"], d[f"{seg}_E_sd_J"], d[f"{seg}_F"], d[f"{seg}_J_per_F"], d[f"{seg}_J_per_F_sd"] = e, esd, f, jpf, sd
            d[f"{seg}_F_min"], d[f"{seg}_F_max"] = g[f"{seg}_F"].min(), g[f"{seg}_F"].max()
        rows.append(d)
    ct = pd.DataFrame(rows)
    ct["env"] = pd.Categorical(ct.env, ENVS, ordered=True)
    ct["algo"] = pd.Categorical(ct.algo, ["sac", "td3", "mbpo", "tdmpc2"], ordered=True)
    return ct.sort_values(["env", "algo", "config"]).reset_index(drop=True)


def exact_perm_p(x, y) -> float:
    """Two-sided exact permutation p of Spearman rho (all n! orderings; n <= 9). Ties use mid-ranks."""
    import itertools
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    obs = abs(np.corrcoef(rx, ry)[0, 1])
    cnt = tot = 0
    for perm in itertools.permutations(range(len(ry))):
        tot += 1
        if abs(np.corrcoef(rx, ry[list(perm)])[0, 1]) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def corr(sub: pd.DataFrame, seg: str, label: dict) -> dict:
    e, f = sub[f"{seg}_E_J"].to_numpy(), sub[f"{seg}_F"].to_numpy()
    d = dict(label, segment=seg, n=len(sub))
    if len(sub) < 3:
        return d
    sp = stats.spearmanr(f, e)
    fit = stats.linregress(np.log10(f), np.log10(e))
    d.update(spearman_rho=sp.statistic, spearman_p=sp.pvalue,
             spearman_p_exact_perm=exact_perm_p(f, e) if len(sub) <= 9 else np.nan, loglog_slope=fit.slope, loglog_slope_se=fit.stderr,
             loglog_intercept=fit.intercept, loglog_r2=fit.rvalue ** 2,
             flops_min=f.min(), flops_max=f.max(), flops_range_orders=np.log10(f.max() / f.min()))
    return d


def linfit(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    fit = stats.linregress(x, y)
    pred = fit.intercept + fit.slope * x
    ssr = ((y - pred) ** 2).sum()
    return dict(n=n, resid_df=n - 2, a_J=fit.intercept, b_J_per_FLOP=fit.slope, b_pJ_per_FLOP=fit.slope * 1e12,
                r2=fit.rvalue ** 2, rse_J=np.sqrt(ssr / (n - 2)) if n > 2 else np.nan)


# sweeps: (sweep name, factor column, fixed-at-base filter builder)
def sweep_members(ct: pd.DataFrame, algo: str, env: str, sweep: str) -> pd.DataFrame:
    s = ct[(ct.algo == algo) & (ct.env == env)]
    if algo == "tdmpc2":
        if sweep == "num_q":
            m = s[(s.horizon == 3)].sort_values("num_q")
        else:
            m = s[(s.num_q == 5)].sort_values("horizon")
        return m
    base_b = BASE_BATCH[algo]
    if algo == "mbpo":
        s = s[s.rollout_regime == ("rl1-1_re20-150" if env == "HalfCheetah-v5" else "rl1-25_re20-100")]
    if sweep == "UTD":
        return s[(s.width == 1024) & (s.batch == base_b)].sort_values("utd")
    if sweep == "width":
        return s[(s.utd == 1) & (s.batch == base_b)].sort_values("width")
    if sweep == "batch":
        return s[(s.utd == 1) & (s.width == 1024)].sort_values("batch")
    raise ValueError(sweep)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load()
    print("max TOTAL reconciliation error:", runs.recon_err.max())
    runs.drop(columns=["run_dir"]).to_csv(OUT / "per_run_segments_5_2.csv", index=False)
    ct = config_table(runs)
    ct.to_csv(OUT / "d1_config_table.csv", index=False)
    print(ct.groupby("env", observed=True).size())

    # ---- (d) total-level, per env
    rows = []
    for env in ENVS:
        rows.append(corr(ct[ct.env == env], "total", dict(level="pooled", env=env, algo="all")))
    # segment-level pooled
    seg_rows = []
    for env in ENVS:
        for seg in SEGS:
            sub = ct[(ct.env == env) & ct[f"{seg}_E_J"].notna()]
            seg_rows.append(corr(sub, seg, dict(level="pooled", env=env, algo="all")))
    # per algo total
    algo_rows = []
    for env in ENVS:
        for algo in ["sac", "td3", "mbpo", "tdmpc2"]:
            sub = ct[(ct.env == env) & (ct.algo == algo)]
            algo_rows.append(corr(sub, "total", dict(level="per_algo", env=env, algo=algo)))
    pd.DataFrame(rows).to_csv(OUT / "d2_total_correlation.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(OUT / "d3_segment_correlation.csv", index=False)
    pd.DataFrame(algo_rows).to_csv(OUT / "d4_per_algo_total_correlation.csv", index=False)
    # per-algo segment correlations (supplement, small n)
    sup = []
    for env in ENVS:
        for algo in ["sac", "td3", "mbpo", "tdmpc2"]:
            for seg in SEGS:
                sub = ct[(ct.env == env) & (ct.algo == algo) & ct[f"{seg}_E_J"].notna()]
                if len(sub):
                    sup.append(corr(sub, seg, dict(level="per_algo", env=env, algo=algo)))
    pd.DataFrame(sup).to_csv(OUT / "d5_per_algo_segment_correlation.csv", index=False)

    # ---- (a) linear fits + delta pairs
    fits, pairs = [], []
    for env in ENVS:
        for algo, sweeps in [("sac", ["UTD", "width", "batch"]), ("td3", ["UTD", "width", "batch"]),
                             ("mbpo", ["UTD", "width", "batch"]), ("tdmpc2", ["num_q", "horizon"])]:
            for sw in sweeps:
                m = sweep_members(ct, algo, env, sw)
                if len(m) < 3:
                    continue
                factor = {"UTD": "utd", "width": "width", "batch": "batch", "num_q": "num_q", "horizon": "horizon"}[sw]
                segs = ["total", "gradient_updates", "rollout"] + (["dynamics_model_update"] if algo == "mbpo" else [])
                for seg in segs:
                    cfgs = "+".join(m.config)
                    d = dict(algo=algo, env=env, sweep=sw, segment=seg, configs=cfgs,
                             values="/".join(str(int(v)) for v in m[factor]))
                    d.update(linfit(m[f"{seg}_F"], m[f"{seg}_E_J"]))
                    d["fit_basis"] = "config_means"
                    fits.append(d)
                    # per-run points
                    rr = runs[(runs.algo == algo) & (runs.env == env) & runs.config.isin(m.config)]
                    d2 = dict(algo=algo, env=env, sweep=sw, segment=seg, configs=cfgs, values=d["values"])
                    d2.update(linfit(rr[f"{seg}_F"], rr[f"{seg}_E"]))
                    d2["fit_basis"] = "per_run"
                    fits.append(d2)
                    # adjacent pairs
                    mm = m.reset_index(drop=True)
                    for i in range(1, len(mm)):
                        dE = mm.loc[i, f"{seg}_E_J"] - mm.loc[i - 1, f"{seg}_E_J"]
                        dF = mm.loc[i, f"{seg}_F"] - mm.loc[i - 1, f"{seg}_F"]
                        ok = abs(dF) >= 0.01 * mm.loc[i - 1, f"{seg}_F"]
                        pairs.append(dict(algo=algo, env=env, sweep=sw, segment=seg, from_value=mm.loc[i - 1, factor],
                                          to_value=mm.loc[i, factor], from_config=mm.loc[i - 1, "config"], to_config=mm.loc[i, "config"],
                                          E_from_J=mm.loc[i - 1, f"{seg}_E_J"], E_to_J=mm.loc[i, f"{seg}_E_J"], dE_J=dE,
                                          F_from=mm.loc[i - 1, f"{seg}_F"], F_to=mm.loc[i, f"{seg}_F"], dF=dF,
                                          dE_per_dF_pJ=dE / dF * 1e12 if ok else np.nan, status="ok" if ok else "dropped (|dF|<1%)"))
    pd.DataFrame(fits).to_csv(OUT / "a1_linear_fits.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "a2_delta_pairs.csv", index=False)

    # ---- checks against the Chapter 4 cross-seed summaries and the pipeline cross-seed CSV
    ch4 = {"sac": ("section_4.1_data/sac_cross_seed_summary.csv", "tag"), "td3": ("Section_4.2_data/td3_cross_seed_summary.csv", "config"),
           "mbpo": ("Section_4.3_data/mbpo_cross_seed_summary.csv", "config"), "tdmpc2": ("Section_4.4_data/tdmpc2_cross_seed_summary.csv", "config")}
    segmap = {"total": "TOTAL_MEASURED_TRAINING"}
    rows = []
    for algo, (f, col) in ch4.items():
        c4 = pd.read_csv(REPO / "contexts" / f)
        c4["_c"] = c4[col].replace({"hor1": "horizon1", "hor5": "horizon5"})
        for r in ct[ct.algo == algo].itertuples():
            for seg in ["total"] + SEGS:
                if pd.isna(getattr(r, f"{seg}_E_J", np.nan)):
                    continue
                m = c4[(c4._c == r.config) & (c4.env == r.env) & (c4.segment == segmap.get(seg, seg))]
                if len(m) != 1:
                    rows.append(dict(algo=algo, env=r.env, config=r.config, segment=seg, status="no ch4 row")); continue
                m = m.iloc[0]
                rows.append(dict(algo=algo, env=r.env, config=r.config, segment=seg, status="compared",
                                 rel_dev_energy=abs(getattr(r, f"{seg}_E_J") / m.energy_j_mean - 1),
                                 rel_dev_flops=abs(getattr(r, f"{seg}_F") / m.flops_mean - 1),
                                 rel_dev_jpf=abs(getattr(r, f"{seg}_J_per_F") / m.j_per_flop_ratio_of_means - 1),
                                 rel_dev_jpf_sd=abs(getattr(r, f"{seg}_J_per_F_sd") / m.j_per_flop_per_seed_sd - 1)))
    chk = pd.DataFrame(rows)
    chk.to_csv(OUT / "x1_check_vs_chapter4.csv", index=False)
    cc = chk[chk.status == "compared"]
    print("ch4 comparison: n=", len(cc), "no-row:", (chk.status != "compared").sum(), cc[[c for c in cc if c.startswith("rel")]].max().to_dict())
    cs = pd.read_csv(CROSS)
    cs = cs[cs.segment == "TOTAL_MEASURED_TRAINING"].copy()
    dev = []
    for r in ct.itertuples():
        m = cs[(cs.algo == r.algo) & (cs.env_id == r.env)]
        m = m[m.mean_energy_joules.sub(r.total_E_J).abs() / r.total_E_J < 1e-9]
        dev.append(len(m))
    print("configs with a pipeline cross-seed TOTAL row matching energy to 1e-9:", sum(d >= 1 for d in dev), "of", len(ct))


if __name__ == "__main__":
    main()
