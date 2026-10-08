"""
Section 5.4 data builder: environment effect (Ant-v5 vs HalfCheetah-v5), all four algorithms.
Read-only on results/ and flop_analysis/ and the thesis .tex; writes only contexts/section_5.4_data/*.csv and findings.json.
Rendered to contexts/section_5.4.md by render_section_5_4.py.

Re-run from the repo root:   python -I -W ignore contexts/section_5.4_data/build_section_5_4.py

Reuse (nothing re-invented where a repo definition exists):
  * run table + config labels: flop_analysis/section_5_2_analysis.parse()  (canonical seeds, labels base/utd2/w256/b512/...)
  * per-run energy/FLOPs/power/duration: flop_analysis/output/per_run_energy_per_flop.csv (the pipeline output)
  * env-comparison definitions copied from contexts/section_4.1_data/build_section_4_1.py part5 (SAC; tab:sac-env) and
    contexts/Section_4.2_data/s4_blocks.py env_comparison_block (TD3/MBPO/TD-MPC2; tab:mbpo-env):
      - E ratio = ratio of means (mean E Ant / mean E HC); paired per-seed ratios give the sd
      - share difference = per-seed (share_Ant - share_HC) in pp, then mean/sd; "same sign" = seeds whose sign equals the mean's
      - max|dshare| = largest |difference of mean shares| over the measured non-GU segments + the 4 allocated GU sub-segments
        (gradient_updates itself excluded: it is the sum of its sub-segments)
      - P ratio = ratio of means; (P_rollout/P_GU) double ratio from env means; J/FLOP ratio = ratio of the two ratio-of-means J/FLOP
      - exact paired two-sided sign-flip test over 2^5 patterns, statistic |mean of paired differences| (signflip_p copied verbatim)
  * last-10% return: flop_analysis/correlation_analysis.last_10pct_return() definition (train-phase episodes, last len//10)
Everything that is computed here and not by one of those repo scripts is tagged DERIVED (not a repo script) in the md.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "contexts" / "section_5.4_data"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "flop_analysis"))
import section_5_2_analysis as s52  # noqa: E402

CANON = s52.CANON
HC, ANT = "HalfCheetah-v5", "Ant-v5"
KWH = 3.6e6
SUBS = ["buffer_sample", "critic_update", "actor_update", "target_update"]
MEASURED = {
    "sac": ["rollout", "gradient_updates"],
    "td3": ["rollout", "gradient_updates"],
    "mbpo": ["rollout", "gradient_updates", "dynamics_model_update", "synthetic_rollout_generation"],
    "tdmpc2": ["rollout", "gradient_updates", "world_model_pretrain"],
}
ALGOS = ["sac", "td3", "mbpo", "tdmpc2"]
CFG_ORDER = {
    "sac": ["base", "utd2", "utd4", "w256", "w512", "b512", "b1024"],
    "td3": ["base", "utd2", "utd4", "w256", "w512", "b256", "b512", "b1024"],
    "mbpo": ["base", "utd2", "utd4", "w256", "w512", "b512", "b1024"],
    "tdmpc2": ["base", "numq3", "numq7", "horizon1", "horizon5"],
}
CFG_LABEL = {"base": "baseline", "utd2": "UTD 2", "utd4": "UTD 4", "w256": "width 256", "w512": "width 512",
             "b256": "batch 256", "b512": "batch 512", "b1024": "batch 1024", "numq3": "num_q 3", "numq7": "num_q 7",
             "horizon1": "horizon 1", "horizon5": "horizon 5", "rollout1": "rollout L_max 1", "rollout15": "rollout L_max 15"}
findings: dict = {}


def signflip_p(d):  # verbatim from contexts/Section_4.2_data/s4_data.py / build_section_4_1.py
    d = np.asarray(d, float)
    obs = abs(d.mean())
    cnt = 0
    for signs in itertools.product([1, -1], repeat=len(d)):
        if abs((np.array(signs) * d).mean()) >= obs - 1e-12 * max(1.0, obs):
            cnt += 1
    return cnt / 2 ** len(d), cnt


# --------------------------------------------------------------------------------------------- load runs
def load_runs():
    df = pd.read_csv(s52.PER_RUN)
    df = df[df.seed.isin(CANON) & df.env_id.isin([HC, ANT])].copy()
    assert df.run_dir.nunique() == 280, df.run_dir.nunique()
    recs = []
    for run, g in df.groupby("run_dir"):
        r0 = g.iloc[0]
        d = s52.parse(r0)
        d.update(seed=int(r0.seed), run_dir=run)
        s = g.set_index("segment")
        tot = s.loc["TOTAL_MEASURED_TRAINING"]
        E = {k: s.loc[k, "total_energy_joules"] for k in s.index if k not in ("TOTAL_MEASURED_TRAINING",)}
        E["gradient_updates"] = sum(E[k] for k in SUBS)
        E["TOTAL"] = tot.total_energy_joules
        F = {k: s.loc[k, "total_flops"] for k in s.index}
        F["gradient_updates"] = F["critic_update"] + F["actor_update"]
        F["TOTAL"] = tot.total_flops
        # GU task duration: the pipeline CSV's sub-segment rows carry the allocated perf_counter times (their mean_power_w is
        # E_sub / perf_counter-time, NOT the power of the measured gradient_updates task), so the GU duration is read from the
        # per-task CSV exactly as contexts/Section_4.2_data/s4_data.py does (sum of the gradient_updates_<i> task durations).
        tc = pd.read_csv(sorted((REPO / run).glob("emissions_*.csv"))[0])
        d_gu = tc[tc.task_name.str.fullmatch(r"gradient_updates_\d+")].duration.sum()
        d.update(E=E, F=F, P_total=tot.mean_power_w, P_roll=s.loc["rollout", "mean_power_w"],
                 P_gu=E["gradient_updates"] / d_gu, D_gu=d_gu,
                 P_gu_pipeline_csv_subrow=s.loc["critic_update", "mean_power_w"],
                 D_roll=s.loc["rollout", "duration_s"], roll_steps=s.loc["rollout", "call_count"],
                 D_total=tot.duration_s)
        tot_e = E["TOTAL"]
        d["share"] = {k: 100 * E[k] / tot_e for k in E if k != "TOTAL"}
        recs.append(d)
    return recs


RUNS = load_runs()
for r in RUNS:
    r["env_s"] = "HC" if r["env"] == HC else "Ant"


def grp(algo, env, cfg):
    g = sorted([r for r in RUNS if r["algo"] == algo and r["env"] == env and r["config"] == cfg], key=lambda r: r["seed"])
    return g


def vals(g, key, sub=None):
    return np.array([r[key][sub] if sub else r[key] for r in g], float)


# --------------------------------------------------------------------------------------------- metadata
META = {}
for r in RUNS:
    META[r["run_dir"]] = json.loads((REPO / r["run_dir"] / "metadata.json").read_text(encoding="utf-8"))


def algo_cfg_of(g):
    cfgs = {json.dumps(META[r["run_dir"]]["algo_config"], sort_keys=True) for r in g}
    assert len(cfgs) == 1, "algo_config not identical within a configuration"
    return json.loads(next(iter(cfgs)))


def exp_cfg_of(g):
    skip = {"env_id", "seed", "output_dir"}
    keys = sorted({k for r in g for k in META[r["run_dir"]]["experiment_config"] if k not in skip})
    out = {}
    for k in keys:
        vs = {json.dumps(META[r["run_dir"]]["experiment_config"].get(k)) for r in g}
        out[k] = json.loads(next(iter(vs))) if len(vs) == 1 else "MIXED:" + "|".join(sorted(vs))
    return out


def cfg_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def commits(g):
    return sorted({META[r["run_dir"]]["git_commit"][:7] for r in g})


# --------------------------------------------------------------------------------------------- 2. paired comparison
def paired(algo, cfg, hc_cfg=None, ant_cfg=None):
    hc, ant = grp(algo, HC, hc_cfg or cfg), grp(algo, ANT, ant_cfg or cfg)
    assert len(hc) == len(ant) == 5 and [r["seed"] for r in hc] == [r["seed"] for r in ant] == sorted(CANON)
    return hc, ant


def env_row(algo, cfg, hc_cfg=None, ant_cfg=None):
    hc, ant = paired(algo, cfg, hc_cfg, ant_cfg)
    row = dict(algo=algo, config=cfg_label_for(cfg), config_tag=cfg)
    eT = vals(ant, "E", "TOTAL") / vals(hc, "E", "TOTAL")
    row["E_total_ratio_of_means"] = vals(ant, "E", "TOTAL").mean() / vals(hc, "E", "TOTAL").mean()
    row["E_total_paired_mean"], row["E_total_paired_sd"] = eT.mean(), eT.std(ddof=1)
    eR = vals(ant, "E", "rollout") / vals(hc, "E", "rollout")
    row["E_rollout_ratio_of_means"] = vals(ant, "E", "rollout").mean() / vals(hc, "E", "rollout").mean()
    row["E_rollout_paired_mean"], row["E_rollout_paired_sd"] = eR.mean(), eR.std(ddof=1)
    seg_list = MEASURED[algo] + [s for s in SUBS]
    diffs_of_means = {}
    for s in seg_list:
        d = vals(ant, "share", s) - vals(hc, "share", s)
        row[f"dshare_{s}_mean_pp"], row[f"dshare_{s}_sd_pp"] = d.mean(), d.std(ddof=1)
        row[f"dshare_{s}_same_sign_k_of_5"] = int((np.sign(d) == np.sign(d.mean())).sum())
        diffs_of_means[s] = d.mean()
    cand = [s for s in seg_list if s != "gradient_updates"]
    smax = max(cand, key=lambda s: abs(diffs_of_means[s]))
    row["max_abs_dshare_pp"], row["max_abs_dshare_segment"] = abs(diffs_of_means[smax]), smax
    for name, f in (("rollout_share", lambda r: r["share"]["rollout"]), ("total_energy", lambda r: r["E"]["TOTAL"])):
        d = np.array([f(a) - f(h) for h, a in zip(hc, ant)])
        p, cnt = signflip_p(d)
        row[f"signflip_p_{name}"], row[f"signflip_patterns_{name}"] = p, cnt
        row[f"signflip_n_positive_{name}"] = int((d > 0).sum())
    row["P_total_ratio_of_means"] = vals(ant, "P_total").mean() / vals(hc, "P_total").mean()
    pt = vals(ant, "P_total") / vals(hc, "P_total")
    row["P_total_paired_mean"], row["P_total_paired_sd"] = pt.mean(), pt.std(ddof=1)
    rra = vals(ant, "P_roll").mean() / vals(ant, "P_gu").mean()
    rrh = vals(hc, "P_roll").mean() / vals(hc, "P_gu").mean()
    row["Proll_over_PGU_HC"], row["Proll_over_PGU_Ant"], row["Proll_over_PGU_double_ratio_Ant_over_HC"] = rrh, rra, rra / rrh
    rrp = (vals(ant, "P_roll") / vals(ant, "P_gu")) / (vals(hc, "P_roll") / vals(hc, "P_gu"))
    row["Proll_over_PGU_double_ratio_paired_mean"], row["Proll_over_PGU_double_ratio_paired_sd"] = rrp.mean(), rrp.std(ddof=1)
    jpf_a = vals(ant, "E", "TOTAL").mean() / vals(ant, "F", "TOTAL").mean()
    jpf_h = vals(hc, "E", "TOTAL").mean() / vals(hc, "F", "TOTAL").mean()
    row["jpf_total_HC_J_per_FLOP"], row["jpf_total_Ant_J_per_FLOP"], row["jpf_total_ratio_Ant_over_HC"] = jpf_h, jpf_a, jpf_a / jpf_h
    row["flops_total_ratio_Ant_over_HC"] = vals(ant, "F", "TOTAL").mean() / vals(hc, "F", "TOTAL").mean()
    row["flops_total_HC"], row["flops_total_Ant"] = vals(hc, "F", "TOTAL").mean(), vals(ant, "F", "TOTAL").mean()
    return row


def cfg_label_for(tag):
    return CFG_LABEL.get(tag, tag)


rows = []
for algo in ALGOS:
    for cfg in CFG_ORDER[algo]:
        row = env_row(algo, cfg)
        # matching flag vs SAC/MBPO (batch 256): TD3 UTD/width/baseline rows run at TD3's own batch 100
        row["tag_td3_batch_matched_to_sac_mbpo"] = ("unmatched (TD3 batch 100)" if algo == "td3" and cfg in
                                                    ("base", "utd2", "utd4", "w256", "w512") else
                                                    "matched (batch 256/512/1024)" if algo == "td3" else "n/a")
        hc, ant = paired(algo, cfg)
        ad = cfg_diff(algo_cfg_of(hc), algo_cfg_of(ant))
        row["algo_config_diff_HC_vs_Ant"] = json.dumps(ad, sort_keys=True) if ad else "none"
        row["experiment_config_diff_HC_vs_Ant"] = json.dumps(cfg_diff(exp_cfg_of(hc), exp_cfg_of(ant)), sort_keys=True) or "none"
        row["env_vs_config_confounded"] = bool(ad)
        row["git_commits_HC"], row["git_commits_Ant"] = ",".join(commits(hc)), ",".join(commits(ant))
        rows.append(row)
paired_df = pd.DataFrame(rows)
assert len(paired_df) == 27, len(paired_df)
paired_df.to_csv(OUT / "env_comparison_paired.csv", index=False)
findings["pair_counts"] = paired_df.groupby("algo").size().to_dict()

# --------------------------------------------------------------------------------------------- cross-check vs thesis
TEX = Path(r"C:\Users\saman\Desktop\Thesis\Thesis Report\thesis-main.tex")
tex = TEX.read_text(encoding="utf-8") if TEX.exists() else ""


def parse_tex_table(label):
    i = tex.find(rf"\label{{{label}}}")
    if i < 0:
        return None
    j = tex.find(r"\end{tabular}", i)
    body = tex[i:j]
    out = {}
    names = {"Baseline": "base", "UTD 2": "utd2", "UTD 4": "utd4", "Width 256": "w256", "Width 512": "w512",
             "Batch 512": "b512", "Batch 1024": "b1024"}
    for ln in body.splitlines():
        m = re.match(r"\s*(Baseline|UTD 2|UTD 4|Width 256|Width 512|Batch 512|Batch 1024)\s*&", ln)
        if m:
            nums = re.findall(r"\\num\{([^}]*)\}", ln)
            seg = re.search(r"\(([a-z]+)\)", ln)
            out[names[m.group(1)]] = dict(vals=[float(x) for x in nums], seg=seg.group(1) if seg else None)
    return out


def r4(x):
    return float(f"{x:.4g}")


def r4_via6_half_up(x):
    """round to 6 significant digits (what the repo context files print), then half-up to 4 (hand/Excel-style rounding)."""
    from decimal import Decimal, ROUND_HALF_UP
    d6 = Decimal(f"{x:.6g}")
    exp = d6.adjusted() - 3
    return float(d6.quantize(Decimal(1).scaleb(exp), rounding=ROUND_HALF_UP))


checks = []
for algo, label in (("sac", "tab:sac-env"), ("mbpo", "tab:mbpo-env")):
    t = parse_tex_table(label)
    if t is None:
        checks.append(dict(algo=algo, config="*", item="table found in tex", thesis=None, computed=None, ok=False, note=f"{label} NOT FOUND"))
        continue
    for cfg, rec in t.items():
        r = paired_df[(paired_df.algo == algo) & (paired_df.config_tag == cfg)].iloc[0]
        comp = [r.E_total_ratio_of_means, r.E_rollout_ratio_of_means, r.max_abs_dshare_pp, r.P_total_ratio_of_means,
                r.Proll_over_PGU_double_ratio_Ant_over_HC, r.jpf_total_ratio_Ant_over_HC]
        names = ["E_total ratio", "E_rollout ratio", "max|dshare| pp", "P_total ratio", "Proll/PGU ratio", "JPF ratio"]
        for n, tv, cv in zip(names, rec["vals"], comp):
            direct = r4(cv) == tv
            via6 = r4_via6_half_up(cv) == tv   # repo context files print 6 digits; the thesis rounded those to 4
            checks.append(dict(algo=algo, config=cfg, item=n, thesis=tv, computed=cv, ok=(direct or via6),
                               note="" if direct else ("matches only via double rounding (6-digit repo value -> 4 digits)" if via6 else "MISMATCH")))
        seg_map = {"fit": "dynamics_model_update", "rollout": "rollout", "critic": "critic_update", "actor": "actor_update"}
        checks.append(dict(algo=algo, config=cfg, item="max|dshare| segment", thesis=rec["seg"], computed=r.max_abs_dshare_segment,
                           ok=(seg_map.get(rec["seg"], rec["seg"]) == r.max_abs_dshare_segment), note=""))
# text numbers (paired sd of the total ratio, baseline)
for algo, thesis_sd, thesis_ratio in (("sac", 0.01179, 1.057), ("td3", 0.01747, 1.086), ("mbpo", 0.1145, 1.221), ("tdmpc2", 0.006616, 1.103)):
    r = paired_df[(paired_df.algo == algo) & (paired_df.config_tag == "base")].iloc[0]
    checks.append(dict(algo=algo, config="base", item="E_total ratio (text)", thesis=thesis_ratio, computed=r.E_total_ratio_of_means,
                       ok=r4(r.E_total_ratio_of_means) == thesis_ratio, note="thesis per-algorithm section text"))
    checks.append(dict(algo=algo, config="base", item="E_total paired sd (text)", thesis=thesis_sd, computed=r.E_total_paired_sd,
                       ok=r4(r.E_total_paired_sd) == thesis_sd, note="thesis per-algorithm section text"))
r = paired_df[(paired_df.algo == "td3") & (paired_df.config_tag == "base")].iloc[0]
checks.append(dict(algo="td3", config="base", item="max|dshare| pp (instruction: 5.045 rollout)", thesis=5.045, computed=r.max_abs_dshare_pp,
                   ok=(r4(r.max_abs_dshare_pp) == 5.045 and r.max_abs_dshare_segment == "rollout"), note=r.max_abs_dshare_segment))
chk_df = pd.DataFrame(checks)
chk_df.to_csv(OUT / "x1_crosscheck_vs_thesis.csv", index=False)

# --------------------------------------------------------------------------------------------- cross-check vs contexts/Section_4.{2,3,4}.md (6 digits)
SEG_MD = {"dyn_model": "dynamics_model_update", "critic": "critic_update", "actor": "actor_update", "rollout": "rollout",
          "synth_rollout": "synthetic_rollout_generation", "buf_sample": "buffer_sample", "target": "target_update"}
x2 = []
for algo, fn, tmap in (("td3", "Section_4.2.md", {}), ("mbpo", "Section_4.3.md", {}), ("tdmpc2", "Section_4.4.md", {"hor1": "horizon1", "hor5": "horizon5"})):
    lines = (REPO / "contexts" / fn).read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("| config | E_TOTAL Ant/HC |"):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                c = [x.strip() for x in lines[j].strip("|").split("|")]
                tag = tmap.get(c[0], c[0])
                r = paired_df[(paired_df.algo == algo) & (paired_df.config_tag == tag)]
                if len(r):
                    r = r.iloc[0]
                    for name, md_v, my_v in (("E_total ratio", float(c[1]), r.E_total_ratio_of_means), ("E_rollout ratio", float(c[2]), r.E_rollout_ratio_of_means),
                                             ("max|dshare| pp", float(c[3]), r.max_abs_dshare_pp), ("P_total ratio", float(c[5]), r.P_total_ratio_of_means),
                                             ("Proll/PGU ratio", float(c[6]), r.Proll_over_PGU_double_ratio_Ant_over_HC), ("JPF ratio", float(c[7]), r.jpf_total_ratio_Ant_over_HC)):
                        x2.append(dict(algo=algo, config=tag, item=name, md_file=fn, md_value=md_v, computed=my_v, rel_dev=abs(my_v - md_v) / abs(md_v),
                                       ok=abs(my_v - md_v) / abs(md_v) < 1e-5))
                    x2.append(dict(algo=algo, config=tag, item="max|dshare| segment", md_file=fn, md_value=c[4], computed=r.max_abs_dshare_segment,
                                   rel_dev=None, ok=(SEG_MD.get(c[4], c[4]) == r.max_abs_dshare_segment)))
                j += 1
x2_df = pd.DataFrame(x2)
x2_df.to_csv(OUT / "x2_crosscheck_vs_contexts_4x.csv", index=False)
findings["x2_n_compared"], findings["x2_all_ok"] = int(len(x2_df)), bool(x2_df.ok.all())
findings["x2_configs_covered"] = sorted({(a, c) for a, c in zip(x2_df.algo, x2_df.config)})

# --------------------------------------------------------------------------------------------- 3. pooled summaries
pool = []


def summarize(df, name):
    d = dict(group=name, n_configs=len(df))
    for col, key in (("E_total_ratio_of_means", "E_total"), ("E_rollout_ratio_of_means", "E_rollout"), ("max_abs_dshare_pp", "max_abs_dshare"),
                     ("P_total_ratio_of_means", "P_total"), ("jpf_total_ratio_Ant_over_HC", "jpf_total")):
        d[f"{key}_min"], d[f"{key}_max"] = df[col].min(), df[col].max()
    d["max_abs_dshare_segment_at_min"] = df.loc[df.max_abs_dshare_pp.idxmin(), "max_abs_dshare_segment"]
    d["max_abs_dshare_segment_at_max"] = df.loc[df.max_abs_dshare_pp.idxmax(), "max_abs_dshare_segment"]
    d["max_abs_dshare_segments_all"] = ";".join(sorted(df.max_abs_dshare_segment.unique()))
    d["rollout_share_same_sign_all5_n_configs"] = int((df["dshare_rollout_same_sign_k_of_5"] == 5).sum())
    d["rollout_share_dshare_positive_n_configs"] = int((df["dshare_rollout_mean_pp"] > 0).sum())
    d["signflip_p_rollout_values"] = ";".join(f"{x:.4f}" for x in sorted(df.signflip_p_rollout_share.unique()))
    d["signflip_p_total_values"] = ";".join(f"{x:.4f}" for x in sorted(df.signflip_p_total_energy.unique()))
    d["E_total_ratio_gt1_n_configs"] = int((df.E_total_ratio_of_means > 1).sum())
    d["E_total_all5_seeds_ant_higher_n_configs"] = int((df.signflip_n_positive_total_energy == 5).sum())
    return d


for algo in ALGOS:
    pool.append(summarize(paired_df[paired_df.algo == algo], algo))
pool.append(summarize(paired_df, "ALL (27 pairs)"))
pool_df = pd.DataFrame(pool)
pool_df.to_csv(OUT / "pooled_summary.csv", index=False)

# --------------------------------------------------------------------------------------------- 4.1 MBPO schedule confound
mb_hc_base, mb_ant_base = grp("mbpo", HC, "base"), grp("mbpo", ANT, "base")
mb_ant_r1, mb_ant_r15 = grp("mbpo", ANT, "rollout1"), grp("mbpo", ANT, "rollout15")
assert len(mb_ant_r1) == 5 and len(mb_ant_r15) == 5
conf = dict(
    algo_config_diff_HC_base_vs_Ant_base=cfg_diff(algo_cfg_of(mb_hc_base), algo_cfg_of(mb_ant_base)),
    algo_config_diff_HC_base_vs_Ant_rollout1=cfg_diff(algo_cfg_of(mb_hc_base), algo_cfg_of(mb_ant_r1)),
    algo_config_diff_Ant_base_vs_Ant_rollout1=cfg_diff(algo_cfg_of(mb_ant_base), algo_cfg_of(mb_ant_r1)),
    algo_config_diff_Ant_base_vs_Ant_rollout15=cfg_diff(algo_cfg_of(mb_ant_base), algo_cfg_of(mb_ant_r15)),
    experiment_config_diff_HC_base_vs_Ant_base=cfg_diff(exp_cfg_of(mb_hc_base), exp_cfg_of(mb_ant_base)),
    experiment_config_diff_HC_base_vs_Ant_rollout1=cfg_diff(exp_cfg_of(mb_hc_base), exp_cfg_of(mb_ant_r1)),
    git_commits=dict(HC_base=commits(mb_hc_base), Ant_base=commits(mb_ant_base), Ant_rollout1=commits(mb_ant_r1), Ant_rollout15=commits(mb_ant_r15)),
    regimes=dict(HC_base=sorted({r["rollout_regime"] for r in mb_hc_base}), Ant_base=sorted({r["rollout_regime"] for r in mb_ant_base}),
                 Ant_rollout1=sorted({r["rollout_regime"] for r in mb_ant_r1}), Ant_rollout15=sorted({r["rollout_regime"] for r in mb_ant_r15})),
    start_utc_range=dict(HC_base=[min(META[r["run_dir"]]["start_time_utc"] for r in mb_hc_base), max(META[r["run_dir"]]["start_time_utc"] for r in mb_hc_base)],
                         Ant_base=[min(META[r["run_dir"]]["start_time_utc"] for r in mb_ant_base), max(META[r["run_dir"]]["start_time_utc"] for r in mb_ant_base)],
                         Ant_rollout1=[min(META[r["run_dir"]]["start_time_utc"] for r in mb_ant_r1), max(META[r["run_dir"]]["start_time_utc"] for r in mb_ant_r1)]),
)
findings["mbpo_confound"] = conf

# like-for-like: HC baseline vs Ant rollout1, every segment
hc, ant = mb_hc_base, mb_ant_r1
seg_rows = []
for s in ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "gradient_updates"] + SUBS + ["TOTAL"]:
    ratio_p = vals(ant, "E", s) / vals(hc, "E", s)
    r_ = dict(comparison="HC baseline vs Ant rollout_max_length=1", segment=s,
              E_HC_mean_J=vals(hc, "E", s).mean(), E_Ant_mean_J=vals(ant, "E", s).mean(),
              E_ratio_of_means=vals(ant, "E", s).mean() / vals(hc, "E", s).mean(),
              E_ratio_paired_mean=ratio_p.mean(), E_ratio_paired_sd=ratio_p.std(ddof=1),
              E_ratio_seeds_gt1=int((ratio_p > 1).sum()))
    if s != "TOTAL":
        d = vals(ant, "share", s) - vals(hc, "share", s)
        r_.update(share_HC_mean_pct=vals(hc, "share", s).mean(), share_Ant_mean_pct=vals(ant, "share", s).mean(),
                  dshare_mean_pp=d.mean(), dshare_sd_pp=d.std(ddof=1), dshare_same_sign_k_of_5=int((np.sign(d) == np.sign(d.mean())).sum()))
        sp, _ = signflip_p(d)
        r_["signflip_p_share"] = sp
    de = np.array([a["E"][s] - h["E"][s] for h, a in zip(hc, ant)])
    r_["signflip_p_energy"] = signflip_p(de)[0]
    seg_rows.append(r_)
mb_l1 = pd.DataFrame(seg_rows)
# same table for the baseline pairing (for side-by-side)
hc, ant = mb_hc_base, mb_ant_base
base_rows = []
for s in ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "gradient_updates"] + SUBS + ["TOTAL"]:
    base_rows.append(dict(segment=s, E_ratio_of_means_baseline_pairing=vals(ant, "E", s).mean() / vals(hc, "E", s).mean()))
mb_l1 = mb_l1.merge(pd.DataFrame(base_rows), on="segment")
mb_l1.to_csv(OUT / "c1_mbpo_hc_base_vs_ant_rollout1.csv", index=False)
row_l1 = env_row("mbpo", "rollout1", hc_cfg="base", ant_cfg="rollout1")
pd.DataFrame([row_l1]).to_csv(OUT / "c1b_mbpo_hc_base_vs_ant_rollout1_envrow.csv", index=False)
findings["mbpo_l1_envrow"] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in row_l1.items()}
# synthetic sample counts
tm_syn = {}
for name, g in (("HC_base", mb_hc_base), ("Ant_base", mb_ant_base), ("Ant_rollout1", mb_ant_r1), ("Ant_rollout15", mb_ant_r15)):
    tot = []
    for r in g:
        tm = json.loads((REPO / r["run_dir"] / "training_metrics.json").read_text(encoding="utf-8"))
        tot.append(sum(e.get("synthetic_transitions_generated", 0) for e in tm["epochs"]))
    tm_syn[name] = dict(min=min(tot), median=float(np.median(tot)), max=max(tot))
findings["mbpo_synthetic_samples"] = tm_syn

# --------------------------------------------------------------------------------------------- 4.2 TD-MPC2 episodic
epi = {}
for m in sorted((REPO / "results").glob("tdmpc2/*/seed_*/*/metadata.json")):
    md = json.loads(m.read_text(encoding="utf-8"))
    key = (md["env_id"], bool(md["algo_config"]["episodic"]), "canonical" if md["seed"] in CANON else "non-canonical")
    epi[key] = epi.get(key, 0) + 1
findings["tdmpc2_episodic_counts"] = {f"{k[0]}|episodic={k[1]}|{k[2]}": v for k, v in sorted(epi.items())}
fj = json.loads((REPO / "flop_analysis" / "flops_per_call.json").read_text(encoding="utf-8"))
ep_sigs = {env: sorted({(sig, v["episodic"]) for sig, v in fj["tdmpc2"][env].items()}) for env in (HC, ANT)}
findings["tdmpc2_flop_sigs_episodic"] = {env: [(s, e) for s, e in ep_sigs[env]] for env in ep_sigs}
# any flops entry where the same env appears with both episodic values?
findings["tdmpc2_flop_both_episodic_values_in_one_env"] = {env: len({e for _, e in ep_sigs[env]}) > 1 for env in ep_sigs}
# metadata consistency: episodic equals the env-specific expectation across ALL tdmpc2 runs
# (also: does measure_flops/compute_energy_per_flop isolate termination-classifier FLOPs?  grep below)
ms = (REPO / "flop_analysis" / "measure_flops.py").read_text(encoding="utf-8")
findings["measure_flops_mentions_termination_isolation"] = bool(re.search(r"termination.*(only|isolat|classifier flops)", ms, re.I))

# --------------------------------------------------------------------------------------------- 4.3 per-call FLOP ratios
BASE_SIG = {
    "sac": "bs256_h1024x1024",
    "td3": "bs100_h1024x1024",
    "mbpo": "bs256_h1024x1024_ens7_mh200x200x200x200_mb256",
}
fl_rows = []


def add_fl(algo, hc_sig, ant_sig, keys, extra=None):
    h, a = fj[algo][HC][hc_sig], fj[algo][ANT][ant_sig]
    for k in keys:
        fl_rows.append(dict(algo=algo, quantity=k, HC=h[k], Ant=a[k], ratio_Ant_over_HC=(a[k] / h[k]) if h[k] else None,
                            HC_signature=hc_sig, Ant_signature=ant_sig, source="flops_per_call.json"))
    for name, f in (extra or {}).items():
        hv, av = f(h), f(a)
        fl_rows.append(dict(algo=algo, quantity=name, HC=hv, Ant=av, ratio_Ant_over_HC=av / hv, HC_signature=hc_sig,
                            Ant_signature=ant_sig, source="DERIVED (not a repo script): sum of two flops_per_call.json entries "
                                                          "(as in s4_data.mbpo_extras)"))


for algo in ("sac", "td3"):
    add_fl(algo, BASE_SIG[algo], BASE_SIG[algo], ["actor_forward_bs1", "critic_fwdbwd", "actor_fwdbwd", "target_update_elementwise_ops"])
add_fl("mbpo", BASE_SIG["mbpo"], BASE_SIG["mbpo"],
       ["actor_forward_bs1", "critic_fwdbwd", "actor_fwdbwd", "target_update_elementwise_ops", "dynamics_member_fwdbwd",
        "dynamics_ensemble_forward_all_bs1"],
       extra={"synthetic_per_sample (actor_forward_bs1 + dynamics_ensemble_forward_all_bs1)":
              lambda x: x["actor_forward_bs1"] + x["dynamics_ensemble_forward_all_bs1"],
              "dynamics_fit_per_sample_per_member (dynamics_member_fwdbwd / model_train_batch_size)":
              lambda x: x["dynamics_member_fwdbwd"] / x["model_train_batch_size"]})
tm_hc = "bs256_h3_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epFalse"
tm_ant = "bs256_h3_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue"
add_fl("tdmpc2", tm_hc, tm_ant, ["rollout_per_env_step", "gradient_update_critic", "gradient_update_actor", "gradient_update_total",
                                  "gradient_update_target_elementwise_ops"])
flop_ratio_df = pd.DataFrame(fl_rows)
flop_ratio_df.to_csv(OUT / "c3_per_call_flop_ratios.csv", index=False)
dims = []
for algo in ALGOS:
    for env in (HC, ANT):
        for sig, v in fj[algo][env].items():
            dims.append(dict(algo=algo, env=env, signature=sig, obs_dim=v["obs_dim"], act_dim=v["act_dim"]))
dims_df = pd.DataFrame(dims)
dims_df.to_csv(OUT / "c3b_obs_act_dims.csv", index=False)
findings["dims"] = {env: sorted({(int(a), int(b)) for a, b in dims_df[dims_df.env == env][["obs_dim", "act_dim"]].values}) for env in (HC, ANT)}
# metadata.json: any obs/act dim field?
md0 = next(iter(META.values()))
findings["metadata_has_obs_dim"] = any("obs" in k.lower() or "dim" in k.lower() for k in md0)
try:
    import gymnasium as gym
    live = {}
    for env in (HC, ANT):
        e = gym.make(env)
        live[env] = (e.observation_space.shape[0], e.action_space.shape[0])
        e.close()
    findings["live_env_dims_this_machine"] = live
    import importlib.metadata as im
    findings["gymnasium_version_this_machine"] = im.version("gymnasium")
except Exception as ex:  # noqa: BLE001
    findings["live_env_dims_this_machine"] = f"NOT RUN: {type(ex).__name__}: {ex}"

# --------------------------------------------------------------------------------------------- 4.4 episode counts / 4.5 rollout wall-clock
ep_rows = []
for algo in ALGOS:
    for env in (HC, ANT):
        for cfg in sorted({r["config"] for r in RUNS if r["algo"] == algo and r["env"] == env}, key=lambda c: (CFG_ORDER[algo] + ["rollout1", "rollout15"]).index(c)):
            g = grp(algo, env, cfg)
            n_tr, n_wu, lens = [], [], []
            for r in g:
                tm = json.loads((REPO / r["run_dir"] / "training_metrics.json").read_text(encoding="utf-8"))
                ep = tm["episodes"]
                n_tr.append(sum(e["phase"] == "train" for e in ep))
                n_wu.append(sum(e["phase"] == "warmup" for e in ep))
                lens += [e["length"] for e in ep if e["phase"] == "train"]
            ep_rows.append(dict(algo=algo, env=env, config=cfg, n_runs=len(g), train_episodes_min=min(n_tr), train_episodes_median=float(np.median(n_tr)),
                                train_episodes_max=max(n_tr), warmup_episodes_min=min(n_wu), warmup_episodes_median=float(np.median(n_wu)),
                                warmup_episodes_max=max(n_wu), train_episode_length_min=min(lens), train_episode_length_median=float(np.median(lens)),
                                train_episode_length_max=max(lens)))
ep_df = pd.DataFrame(ep_rows)
ep_df.to_csv(OUT / "c4_episode_counts.csv", index=False)

wc_rows = []
dev_max = 0.0
for algo in ALGOS:
    for env in (HC, ANT):
        for cfg in sorted({r["config"] for r in RUNS if r["algo"] == algo and r["env"] == env}, key=lambda c: (CFG_ORDER[algo] + ["rollout1", "rollout15"]).index(c)):
            g = grp(algo, env, cfg)
            per1000, per1000_tc = [], []
            for r in g:
                per1000.append(r["D_roll"] / r["roll_steps"] * 1000)
                tc = sorted((REPO / r["run_dir"]).glob("emissions_*.csv"))[0]
                t = pd.read_csv(tc)
                tr = t[t.task_name.str.fullmatch(r"rollout_\d+")]
                spe = META[r["run_dir"]]["experiment_config"]["steps_per_epoch"]
                per1000_tc.append(tr.duration.sum() / (len(tr) * spe) * 1000)
                dev_max = max(dev_max, abs(per1000[-1] - per1000_tc[-1]) / per1000_tc[-1])
            wc_rows.append(dict(algo=algo, env=env, config=cfg, n_runs=len(g), rollout_steps=int(g[0]["roll_steps"]),
                                rollout_s_per_1000_steps_mean=np.mean(per1000_tc), rollout_s_per_1000_steps_sd=np.std(per1000_tc, ddof=1),
                                rollout_s_per_1000_steps_pipeline_csv_mean=np.mean(per1000)))
wc_df = pd.DataFrame(wc_rows)
wc_df.to_csv(OUT / "c5_rollout_wallclock_per_1000_steps.csv", index=False)
findings["wallclock_pipeline_vs_taskcsv_max_rel_dev"] = dev_max

# --------------------------------------------------------------------------------------------- 4.6 simulator timing search
sim = {}
tm_keys, seg_keys = set(), set()
for algo in ALGOS:
    for env in (HC, ANT):
        r = grp(algo, env, "base")[0]
        tm = json.loads((REPO / r["run_dir"] / "training_metrics.json").read_text(encoding="utf-8"))
        tm_keys |= set(tm["epochs"][0].keys())
        seg_keys |= set(json.loads((REPO / r["run_dir"] / "segment_energy.json").read_text(encoding="utf-8")).keys())
findings["training_metrics_epoch_keys"] = sorted(tm_keys)
findings["segment_energy_keys"] = sorted(seg_keys)
hits = []
for p in (REPO / "algorithms").glob("*.py"):
    for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if re.search(r"perf_counter|time\.time", ln):
            hits.append(f"{p.name}:{i}")
findings["algorithms_files_with_timers"] = hits
sim_hits = []
for p in [REPO / "README.md", REPO / "CLAUDE.md"] + sorted((REPO / "contexts").glob("*.md")):
    for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if re.search(r"simulator|mujoco.*step|env\.step.*(time|dur)|NOT MEASURED", ln, re.I):
            sim_hits.append(f"{p.name}:{i}: {ln.strip()[:200]}")
findings["simulator_mentions"] = sim_hits

# --------------------------------------------------------------------------------------------- 4.7 MBPO return
def last10(run_dir):
    ep = json.loads((REPO / run_dir / "training_metrics.json").read_text(encoding="utf-8"))["episodes"]
    rets = [e["return"] for e in ep if e["phase"] == "train"]
    return float(np.mean(rets[-max(1, len(rets) // 10):]))


ret_rows = []
for env in (HC, ANT):
    for cfg in sorted({r["config"] for r in RUNS if r["algo"] == "mbpo" and r["env"] == env}, key=lambda c: (CFG_ORDER["mbpo"] + ["rollout1", "rollout15"]).index(c)):
        g = grp("mbpo", env, cfg)
        v = [last10(r["run_dir"]) for r in g]
        ret_rows.append(dict(env=env, config=cfg, n_runs=5, last10pct_return_mean=np.mean(v), last10pct_return_sd=np.std(v, ddof=1),
                             last10pct_return_min=min(v), last10pct_return_max=max(v)))
ret_df = pd.DataFrame(ret_rows)
ret_df.to_csv(OUT / "c7_mbpo_last10pct_return.csv", index=False)
a = ret_df[ret_df.env == ANT]
findings["mbpo_ant_returns"] = dict(n_configs=len(a), n_negative_mean=int((a.last10pct_return_mean < 0).sum()),
                                     n_all_seeds_negative=int((a.last10pct_return_max < 0).sum()))
h = ret_df[ret_df.env == HC]
findings["mbpo_hc_returns"] = dict(n_configs=len(h), n_negative_mean=int((h.last10pct_return_mean < 0).sum()),
                                    min=float(h.last10pct_return_mean.min()), max=float(h.last10pct_return_mean.max()))
# cross-check vs summary.csv (aggregate_results.py output)
try:
    sm = pd.read_csv(REPO / "summary.csv")
    sm = sm[(sm.algo == "mbpo") & sm.seed.isin(CANON) & sm.env.isin([HC, ANT])]
    mine = {r["run_dir"].replace("/", "\\"): last10(r["run_dir"]) for r in RUNS if r["algo"] == "mbpo"}
    m2 = {k.replace("/", "\\"): v for k, v in mine.items()}
    dev = [abs(row["reward__last_10pct_mean_episode_return"] - m2[row["run_dir"]]) for _, row in sm.iterrows() if row["run_dir"] in m2]
    findings["return_check_vs_summary_csv"] = dict(n_matched=len(dev), max_abs_dev=max(dev) if dev else None)
except Exception as ex:  # noqa: BLE001
    findings["return_check_vs_summary_csv"] = f"NOT RUN: {ex}"

# --------------------------------------------------------------------------------------------- misc facts for the md
findings["n_runs"] = len(RUNS)
findings["cross_check_all_ok"] = bool(chk_df.ok.all())
findings["cross_check_mismatches"] = chk_df[~chk_df.ok].to_dict("records")
(OUT / "findings.json").write_text(json.dumps(findings, indent=1, default=str), encoding="utf-8")
print("pairs:", findings["pair_counts"])
print("crosscheck all ok:", findings["cross_check_all_ok"], "mismatches:", len(findings["cross_check_mismatches"]))
for m in findings["cross_check_mismatches"][:20]:
    print("  MISMATCH", m)
print("wallclock pipeline-vs-taskcsv max rel dev:", dev_max)
