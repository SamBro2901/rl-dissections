"""
Builds the companion CSVs and table fragments for contexts/Section_3.1_3.2_3.3.md.

Read-only with respect to the repo: it only reads results/ and flop_analysis/output/ and writes into its own
folder (contexts/Section_3.1_3.2_3.3_data/).  Canonical seeds only (the dev seed 0 is touched only by
d4_rq4.py, not here).

Run (from the repo root):
    python contexts/Section_3.1_3.2_3.3_data/build_section_3.py

Inputs : results/**/metadata.json, segment_energy.json, training_metrics.json, emissions_*.csv,
         flop_analysis/output/per_run_energy_per_flop.csv, flop_analysis/output/cross_seed_energy_per_flop.csv,
         configs/config.py + configs/overrides/*.json
Outputs: E1_canonical_groups.csv, E2_run_settings.csv, B1_baseline_decomposition.csv, B2_sweep_scaling.csv,
         B2_ratio_largest_over_smallest.csv, C4_per_run_table.csv, D2_segment_flops_energy.csv,
         D3_group_ranking_candidates.csv, checks.json, frag_*.md (table fragments used by the main file)
Std convention everywhere: sample std, ddof = 1 (same as contexts/section_4.1_data/build_section_4_1.py).
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flop_analysis"))

from configs.config import ALGO_CONFIGS  # noqa: E402
from flop_keys import hidden_sizes as fk_hidden, mbpo_rollout_regime, signature  # noqa: E402
import compute_energy_per_flop as cepf  # noqa: E402

CANON = [331, 958, 14577, 43611, 85062]
ALGOS = ["sac", "mbpo", "td3", "tdmpc2"]
ENVS = ["HalfCheetah-v5", "Ant-v5"]
KWH_TO_J = 3.6e6
PER_RUN = REPO / "flop_analysis" / "output" / "per_run_energy_per_flop.csv"
CROSS = REPO / "flop_analysis" / "output" / "cross_seed_energy_per_flop.csv"

SUBS = ["buffer_sample", "critic_update", "actor_update", "target_update"]
MEASURED = {"rollout", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain",
            "gradient_updates", "warmup", "idle_baseline_head", "idle_baseline_tail"}
SEG_ORDER = ["rollout", "gradient_updates", "buffer_sample", "critic_update", "actor_update", "target_update",
             "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain",
             "TOTAL_MEASURED_TRAINING", "warmup", "idle_baseline_head", "idle_baseline_tail"]
SWEEP_FIELD = {"hidden_sizes": "width", "batch_size": "batch_size", "updates_per_env_step": "utd",
               "rollout_max_length": "rollout_max_length", "num_q": "num_q", "horizon": "horizon"}
SWEEP_ORDER = ["baseline", "utd", "width", "batch_size", "rollout_max_length", "num_q", "horizon"]

checks = OrderedDict()


def msd(a):
    a = np.asarray(a, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else float("nan"))


# --------------------------------------------------------------------------------------------- load runs
def baseline_cfg(algo, env):
    base = json.loads(json.dumps(dataclasses.asdict(ALGO_CONFIGS[algo]())))
    ov = None
    if env == "Ant-v5" and algo in ("mbpo", "tdmpc2"):
        ov = REPO / "configs" / "overrides" / f"{algo}_ant.json"
        base.update(json.loads(ov.read_text()))
    return base, (str(ov.relative_to(REPO)).replace("\\", "/") if ov else None)


runs = []
for p in sorted((REPO / "results").glob("*/*/seed_*/*/metadata.json")):
    m = json.loads(p.read_text())
    if m["seed"] not in CANON:
        continue
    d = p.parent
    runs.append(dict(dir=d, rel=str(d.relative_to(REPO)).replace("\\", "/"), meta=m))
checks["n_canonical_runs"] = len(runs)
checks["n_all_metadata"] = len(list((REPO / "results").glob("*/*/seed_*/*/metadata.json")))

bases = {}
for r in runs:
    m = r["meta"]
    algo, env = m["algo_name"], m["env_id"]
    if (algo, env) not in bases:
        bases[(algo, env)] = baseline_cfg(algo, env)
    base, _ = bases[(algo, env)]
    ac = json.loads(json.dumps(m["algo_config"]))
    assert set(ac) == set(base), (r["rel"], set(ac) ^ set(base))
    diff = [k for k in ac if ac[k] != base[k]]
    r["diff"] = diff
    if not diff:
        r["sweep_name"], r["sweep_value"], r["is_baseline"] = "baseline", "", True
    elif len(diff) == 1 and diff[0] in SWEEP_FIELD:
        k = diff[0]
        r["sweep_name"] = SWEEP_FIELD[k]
        r["sweep_value"] = ac[k][0] if k == "hidden_sizes" else ac[k]
        r["is_baseline"] = False
    else:
        raise SystemExit(f"unclassifiable config diff {diff} in {r['rel']}")
    r["algo"], r["env"], r["seed"] = algo, env, m["seed"]
    r["sig"] = signature(algo, ac)
    r["regime"] = mbpo_rollout_regime(ac) if algo == "mbpo" else None
    r["utd"] = ac["updates_per_env_step"]

# group ids
def gkey(r):
    return (r["algo"], r["env"], r["sweep_name"], r["sweep_value"])


def sort_key(k):
    algo, env, sn, sv = k
    return (ALGOS.index(algo), ENVS.index(env), SWEEP_ORDER.index(sn), sv if sv != "" else -1)


gkeys = sorted({gkey(r) for r in runs}, key=sort_key)
gid = {k: f"G{i + 1:02d}" for i, k in enumerate(gkeys)}
for r in runs:
    r["gid"] = gid[gkey(r)]
checks["n_groups"] = len(gkeys)
checks["groups_with_fewer_than_5_seeds"] = [
    (gid[k], k) for k in gkeys if len({r["seed"] for r in runs if gkey(r) == k}) != 5]

# compare my group definition against the pipeline's cross-seed key (algo, env, sig, utd, regime)
cross = pd.read_csv(CROSS)
tot = cross[cross.segment == "TOTAL_MEASURED_TRAINING"]
pk = set(zip(tot.algo, tot.env_id, tot.architecture_signature, tot.updates_per_env_step, tot.mbpo_rollout_regime.fillna("")))
mk = {(r["algo"], r["env"], r["sig"], r["utd"], r["regime"] or "") for r in runs}
checks["my_group_keys_equal_pipeline_cross_seed_keys"] = (pk == mk)
checks["n_pipeline_cross_seed_groups"] = len(pk)
checks["groups_distinct_by_pipeline_key"] = len(mk)

# --------------------------------------------------------------------------------------------- E1
e1 = []
for k in gkeys:
    rs = [r for r in runs if gkey(r) == k]
    r0 = rs[0]
    ac, ec = r0["meta"]["algo_config"], r0["meta"]["experiment_config"]
    e1.append(OrderedDict(
        group_id=gid[k], algo=k[0], env=k[1], sweep_name=k[2], sweep_value=k[3], is_baseline=r0["is_baseline"],
        hidden_sizes=fk_hidden(k[0], ac), utd=ac["updates_per_env_step"], batch_size=ac["batch_size"],
        rollout_max_length=ac.get("rollout_max_length", ""), num_q=ac.get("num_q", ""), horizon=ac.get("horizon", ""),
        episodic=ac.get("episodic", ""), warmup_steps=ec["warmup_steps"], n_seeds=len({r["seed"] for r in rs}),
        n_runs=len(rs), seeds=" ".join(str(s) for s in sorted({r["seed"] for r in rs})),
        architecture_signature=r0["sig"], mbpo_rollout_regime=r0["regime"] or "",
        baseline_override_file=bases[(k[0], k[1])][1] or "", config_diff_vs_baseline=",".join(r0["diff"])))
e1 = pd.DataFrame(e1)
e1.to_csv(HERE / "E1_canonical_groups.csv", index=False)

# --------------------------------------------------------------------------------------------- per-run segment table
pr = pd.read_csv(PER_RUN)
pr = pr[pr.included_in_cross_seed_avg]
pr["run_dir"] = pr["run_dir"].astype(str)
checks["per_run_csv_canonical_runs"] = int(pr.run_dir.nunique())
byrun = {rel: g for rel, g in pr.groupby("run_dir")}
assert all(r["rel"] in byrun for r in runs), "run missing from per_run_energy_per_flop.csv"


def per_run_segments(r):
    """segment -> dict(E_j, E_kwh, F, flop_type, dur) for one run, incl. derived gradient_updates."""
    g = byrun[r["rel"]].set_index("segment")
    out = {}
    for seg, row in g.iterrows():
        out[seg] = dict(E_j=float(row.total_energy_joules), E_kwh=float(row.total_energy_kwh),
                        F=(float(row.total_flops) if pd.notna(row.total_flops) else float("nan")),
                        flop_type=row.flop_type, dur=(float(row.duration_s) if pd.notna(row.duration_s) else float("nan")),
                        pw=(float(row.mean_power_w) if pd.notna(row.mean_power_w) else float("nan")))
    if all(s in out for s in SUBS):
        out["gradient_updates"] = dict(
            E_j=sum(out[s]["E_j"] for s in SUBS), E_kwh=sum(out[s]["E_kwh"] for s in SUBS),
            F=out["critic_update"]["F"] + out["actor_update"]["F"], flop_type="matmul_derived",
            dur=float("nan"), pw=float("nan"))
    return out


for r in runs:
    r["seg"] = per_run_segments(r)

# check: per_run csv energies == segment_energy.json values (kWh) for every canonical (run, segment)
mx_json = 0.0
n_json = 0
for r in runs:
    sj = json.loads((r["dir"] / "segment_energy.json").read_text())
    for sg, v in r["seg"].items():
        if sg in sj and not sg.startswith("_"):
            mx_json = max(mx_json, abs(v["E_kwh"] - sj[sg]) / sj[sg])
            n_json += 1
checks["per_run_csv_vs_segment_energy_json"] = dict(n_pairs=n_json, max_rel_diff=mx_json)
# check: derived gradient_updates energy == measured gradient_updates task energy (per-task CSV) ; TOTAL == sum
mx_rel, mx_tot = 0.0, 0.0
for r in runs:
    hw = cepf.load_segment_hw_energy(str(r["dir"]))
    gu = hw["gradient_updates"]["energy_consumed_kwh"]
    mx_rel = max(mx_rel, abs(r["seg"]["gradient_updates"]["E_kwh"] - gu) / gu)
    s = sum(v["E_kwh"] for k, v in r["seg"].items()
            if k in cepf.TRAINING_SEGMENTS and k != "gradient_updates")
    mx_tot = max(mx_tot, abs(s - r["seg"]["TOTAL_MEASURED_TRAINING"]["E_kwh"]) / s)
checks["max_rel_diff_sum_of_4_allocated_vs_measured_gradient_updates_task"] = mx_rel
checks["max_rel_diff_TOTAL_vs_sum_of_training_segments"] = mx_tot

# --------------------------------------------------------------------------------------------- B1 / B2 / D2 aggregation
def agg_group(rs, seg):
    """cross-seed statistics of one segment over the runs rs (each having it)."""
    have = [r for r in rs if seg in r["seg"]]
    if not have:
        return None
    E = np.array([r["seg"][seg]["E_j"] for r in have])
    Ek = np.array([r["seg"][seg]["E_kwh"] for r in have])
    F = np.array([r["seg"][seg]["F"] for r in have], float)
    ft = have[0]["seg"][seg]["flop_type"]
    tot = np.array([r["seg"]["TOTAL_MEASURED_TRAINING"]["E_j"] for r in have])
    share = 100 * E / tot
    em, es = msd(E)
    km, ks = msd(Ek)
    shm, shs = msd(share)
    fm = float(np.nanmean(F)) if not np.all(np.isnan(F)) else float("nan")
    fsd = float(np.nanstd(F, ddof=1)) if (len(F) > 1 and not np.all(np.isnan(F))) else float("nan")
    fnu = int(len(set(F[~np.isnan(F)].tolist()))) if not np.all(np.isnan(F)) else 0
    return dict(n=len(have), E_mean=em, E_sd=es, Ekwh_mean=km, Ekwh_sd=ks, share_mean=shm, share_sd=shs,
                share_from_means=100 * em / float(tot.mean()), F_mean=fm, F_sd=fsd, F_nuniq=fnu, flop_type=ft)


def kind_of(seg):
    if seg == "TOTAL_MEASURED_TRAINING":
        return "total"
    if seg in SUBS:
        return "allocated"
    return "measured"


def jpf(seg, a):
    """cross-seed J/FLOP (ratio of means, as the pipeline's cross_seed csv) where the repo defines it."""
    if a["flop_type"] in ("matmul", "mixed_total", "matmul_derived") and a["F_mean"] and not math.isnan(a["F_mean"]) and a["F_mean"] > 0:
        return a["E_mean"] / a["F_mean"]
    return float("nan")


def segs_for(rs):
    present = set().union(*[set(r["seg"]) for r in rs])
    return [s for s in SEG_ORDER if s in present]


group_runs = {k: [r for r in runs if gkey(r) == k] for k in gkeys}

# validation of my aggregation against the pipeline cross-seed CSV (energy mean + flops mean + J/FLOP) for every (group, segment)
cs_idx = cross.set_index(["algo", "env_id", "architecture_signature", "updates_per_env_step",
                          cross.mbpo_rollout_regime.fillna(""), "segment"])
mx_e = mx_f = mx_j = 0.0
n_cmp = 0
n_missing = []
for k in gkeys:
    rs = group_runs[k]
    for seg in segs_for(rs):
        if seg == "gradient_updates":
            continue
        a = agg_group(rs, seg)
        key = (k[0], k[1], rs[0]["sig"], rs[0]["utd"], rs[0]["regime"] or "", seg)
        if key not in cs_idx.index:  # warmup / idle_baseline_* have no FLOP row -> not written to the cross-seed csv
            n_missing.append(seg)
            continue
        row = cs_idx.loc[key]
        mx_e = max(mx_e, abs(a["E_mean"] - row.mean_energy_joules) / row.mean_energy_joules)
        if a["F_mean"] and not math.isnan(a["F_mean"]) and row.total_flops:
            mx_f = max(mx_f, abs(a["F_mean"] - row.total_flops) / row.total_flops)
        if pd.notna(row.mean_energy_per_flop_j_per_flop) and not math.isnan(jpf(seg, a)):
            mx_j = max(mx_j, abs(jpf(seg, a) - row.mean_energy_per_flop_j_per_flop) / row.mean_energy_per_flop_j_per_flop)
        n_cmp += 1
checks["recomputed_vs_cross_seed_csv"] = dict(n_rows_compared=n_cmp, max_rel_diff_energy=mx_e, max_rel_diff_flops=mx_f,
                                              max_rel_diff_j_per_flop_for_defined_rows=mx_j,
                                              segments_absent_from_cross_seed_csv=dict(Counter(n_missing)))

# ---- B1: baseline decomposition
b1 = []
for k in gkeys:
    if k[2] != "baseline":
        continue
    rs = group_runs[k]
    for seg in segs_for(rs):
        a = agg_group(rs, seg)
        in_total = seg in cepf.TRAINING_SEGMENTS or seg == "gradient_updates"
        b1.append(OrderedDict(
            group_id=gid[k], algo=k[0], env=k[1], segment=seg, kind=kind_of(seg) if seg != "gradient_updates" else "measured (task = sum of the 4 allocated sub-segments)",
            counted_in_TOTAL_MEASURED_TRAINING=("derived_composite_of_4_subsegments" if seg == "gradient_updates" else (seg in cepf.TRAINING_SEGMENTS)),
            n_seeds=a["n"], energy_j_mean=a["E_mean"], energy_j_std_ddof1=a["E_sd"],
            energy_kwh_mean=a["Ekwh_mean"], energy_kwh_std_ddof1=a["Ekwh_sd"],
            share_of_TOTAL_pct_mean_of_per_seed_shares=a["share_mean"], share_of_TOTAL_pct_std_of_per_seed_shares_ddof1=a["share_sd"],
            share_of_TOTAL_pct_from_mean_energies=a["share_from_means"],
            matmul_flops_mean=a["F_mean"], flop_type=a["flop_type"], j_per_flop_ratio_of_means=jpf(seg, a)))
pd.DataFrame(b1).to_csv(HERE / "B1_baseline_decomposition.csv", index=False)
b1df = pd.DataFrame(b1)

# max difference between mean-of-per-seed-shares and share-of-mean-energies
checks["share_mean_of_ratios_vs_ratio_of_means_max_abs_diff_pp_over_B1"] = float(
    (b1df.share_of_TOTAL_pct_mean_of_per_seed_shares - b1df.share_of_TOTAL_pct_from_mean_energies).abs().max())
# do non-overlapping shares sum to 100?
sums = []
for k in gkeys:
    rs = group_runs[k]
    for r in rs:
        s = sum(v["E_j"] for sg, v in r["seg"].items() if sg in cepf.TRAINING_SEGMENTS)
        sums.append(abs(100 * s / r["seg"]["TOTAL_MEASURED_TRAINING"]["E_j"] - 100))
checks["max_abs_deviation_of_nonoverlapping_shares_sum_from_100_pp"] = float(max(sums))

# ---- B2: sweep scaling
sweeps_of = {}
for k in gkeys:
    if k[2] != "baseline":
        sweeps_of.setdefault((k[0], k[1]), set()).add(k[2])
b2 = []
for (algo, env), sset in sorted(sweeps_of.items(), key=lambda t: (ALGOS.index(t[0][0]), ENVS.index(t[0][1]))):
    base_k = next(k for k in gkeys if k[0] == algo and k[1] == env and k[2] == "baseline")
    base_ac, _ = bases[(algo, env)]
    for sn in sorted(sset, key=SWEEP_ORDER.index):
        field = next(f for f, n in SWEEP_FIELD.items() if n == sn)
        base_val = base_ac[field][0] if field == "hidden_sizes" else base_ac[field]
        members = [(base_k, base_val, True)] + [(k, k[3], False) for k in gkeys if k[0] == algo and k[1] == env and k[2] == sn]
        for k, val, isb in members:
            rs = group_runs[k]
            for seg in segs_for(rs):
                a = agg_group(rs, seg)
                b2.append(OrderedDict(
                    group_id=gid[k], algo=algo, env=env, sweep_name=sn, sweep_value=val, is_baseline=isb, segment=seg,
                    kind=kind_of(seg) if seg != "gradient_updates" else "measured_composite", flop_type=a["flop_type"],
                    n_seeds=a["n"], energy_j_mean=a["E_mean"], energy_j_std_ddof1=a["E_sd"],
                    matmul_flops_mean=a["F_mean"], matmul_flops_std_ddof1=a["F_sd"], j_per_flop_ratio_of_means=jpf(seg, a)))
b2df = pd.DataFrame(b2)
b2df.to_csv(HERE / "B2_sweep_scaling.csv", index=False)

# ratio table for rollout and gradient_updates: largest sweep value / smallest sweep value (numeric order of sweep_value)
rat = []
for (algo, env, sn), g in b2df.groupby(["algo", "env", "sweep_name"], sort=False):
    vals = sorted(g.sweep_value.unique(), key=float)
    lo, hi = vals[0], vals[-1]
    row = OrderedDict(algo=algo, env=env, sweep=sn, values_numeric_order=" < ".join(str(v) for v in vals),
                      smallest=lo, largest=hi)
    for seg in ("rollout", "gradient_updates"):
        a = g[(g.segment == seg) & (g.sweep_value == lo)].iloc[0]
        b = g[(g.segment == seg) & (g.sweep_value == hi)].iloc[0]
        row[f"{seg}_energy_ratio"] = b.energy_j_mean / a.energy_j_mean
        row[f"{seg}_flops_ratio"] = b.matmul_flops_mean / a.matmul_flops_mean
    rat.append(row)
ratdf = pd.DataFrame(rat)
ratdf.to_csv(HERE / "B2_ratio_largest_over_smallest.csv", index=False)

# ---- D2: per (group, segment)
d2 = []
for k in gkeys:
    rs = group_runs[k]
    for seg in segs_for(rs):
        a = agg_group(rs, seg)
        d2.append(OrderedDict(group_id=gid[k], algo=k[0], env=k[1], sweep_name=k[2], sweep_value=k[3], segment=seg,
                              kind=kind_of(seg) if seg != "gradient_updates" else "measured_composite",
                              flop_type=a["flop_type"], n_seeds=a["n"], energy_j_mean=a["E_mean"],
                              energy_j_std_ddof1=a["E_sd"], matmul_flops_mean=a["F_mean"],
                              matmul_flops_std_ddof1=a["F_sd"], n_distinct_flop_values_over_seeds=a["F_nuniq"],
                              j_per_flop_ratio_of_means=jpf(seg, a)))
d2df = pd.DataFrame(d2)
d2df.to_csv(HERE / "D2_segment_flops_energy.csv", index=False)

# ---- D3 ranking candidates per group
d3 = []
for k in gkeys:
    rs = group_runs[k]
    a = agg_group(rs, "TOTAL_MEASURED_TRAINING")
    dur = np.mean([r["seg"]["TOTAL_MEASURED_TRAINING"]["dur"] for r in rs])
    d3.append(OrderedDict(group_id=gid[k], algo=k[0], env=k[1], sweep_name=k[2], sweep_value=k[3],
                          total_energy_j_mean=a["E_mean"], total_energy_j_std_ddof1=a["E_sd"],
                          total_matmul_flops_mean=a["F_mean"], total_j_per_flop_ratio_of_means=jpf("TOTAL_MEASURED_TRAINING", a),
                          total_duration_s_mean=float(dur)))
pd.DataFrame(d3).to_csv(HERE / "D3_group_ranking_candidates.csv", index=False)

# --------------------------------------------------------------------------------------------- C4 per-run return table
c4 = []
for r in runs:
    tm = json.loads((r["dir"] / "training_metrics.json").read_text())
    eps, epochs = tm["episodes"], tm["epochs"]
    rets = [e["return"] for e in eps if e["phase"] == "train"]
    n10 = max(1, len(rets) // 10)
    lens = [e["length"] for e in eps if e["phase"] == "train"]
    T = r["seg"]["TOTAL_MEASURED_TRAINING"]
    c4.append(OrderedDict(
        algo=r["algo"], env=r["env"], group_id=r["gid"], sweep_name=r["sweep_name"], sweep_value=r["sweep_value"], seed=r["seed"],
        run_dir=r["rel"],
        return_last10pct_mean_train_episodes=float(np.mean(rets[-n10:])),   # repo definition (aggregate_results.py, correlation_analysis.py)
        return_max_train_episode=float(max(rets)),
        return_final_epoch_mean_episode_return=epochs[-1]["mean_episode_return"],
        return_final_cumulative_reward=epochs[-1]["cumulative_reward"],
        n_train_episodes=len(rets), n_last10pct_episodes=n10,
        train_episode_length_min=min(lens), train_episode_length_median=float(np.median(lens)), train_episode_length_max=max(lens),
        n_train_incomplete_episodes=sum(e["phase"] == "train_incomplete" for e in eps),
        total_matmul_flops_TOTAL_MEASURED_TRAINING=T["F"], total_energy_j_TOTAL_MEASURED_TRAINING=T["E_j"]))
c4df = pd.DataFrame(c4)
c4df.to_csv(HERE / "C4_per_run_table.csv", index=False)
checks["C4_rows"] = len(c4df)
checks["C4_distinct_seeds"] = sorted(int(s) for s in c4df.seed.unique())

# --------------------------------------------------------------------------------------------- E2 settings
def tally(extract):
    return Counter(extract(r["meta"]) for r in runs)


settings = OrderedDict([
    ("train_steps", lambda m: m["experiment_config"]["train_steps"]),
    ("steps_per_epoch", lambda m: m["experiment_config"]["steps_per_epoch"]),
    ("warmup_steps", lambda m: m["experiment_config"]["warmup_steps"]),
    ("settle_seconds", lambda m: m["experiment_config"]["settle_seconds"]),
    ("idle_baseline_seconds(head)", lambda m: m["experiment_config"]["idle_baseline_seconds"]),
    ("idle_tail_seconds", lambda m: m["experiment_config"]["idle_tail_seconds"]),
    ("measure_power_secs", lambda m: m["experiment_config"]["measure_power_secs"]),
    ("tracking_mode", lambda m: m["experiment_config"]["tracking_mode"]),
    ("gpu_min_clock_mhz", lambda m: m["experiment_config"]["gpu_min_clock_mhz"]),
    ("gpu_max_clock_mhz", lambda m: m["experiment_config"]["gpu_max_clock_mhz"]),
    ("lock_gpu_clocks", lambda m: m["experiment_config"]["lock_gpu_clocks"]),
    ("set_persistence_mode", lambda m: m["experiment_config"]["set_persistence_mode"]),
    ("set_cpu_performance_governor", lambda m: m["experiment_config"]["set_cpu_performance_governor"]),
    ("thermal_gate_enabled", lambda m: m["experiment_config"]["thermal_gate_enabled"]),
    ("torch_version", lambda m: m["torch_version"]),
    ("codecarbon_version", lambda m: m["codecarbon_version"]),
    ("cuda_device_name", lambda m: m["cuda_device_name"]),
    ("device", lambda m: m["device"]),
    ("country_iso_code", lambda m: m["experiment_config"]["country_iso_code"]),
    ("force_cpu_power_w", lambda m: m["experiment_config"]["force_cpu_power_w"]),
    ("git_commit", lambda m: m["git_commit"][:7]),
])
e2 = []
for name, fn in settings.items():
    c = tally(fn)
    for val, n in sorted(c.items(), key=lambda t: -t[1]):
        e2.append(OrderedDict(setting=name, value=val, n_runs=n))
# warmup by algo/env
for (algo, env), c in sorted({(r["algo"], r["env"]): None for r in runs}.items()):
    pass
wu = Counter((r["algo"], r["env"], r["meta"]["experiment_config"]["warmup_steps"]) for r in runs)
for (algo, env, w), n in sorted(wu.items(), key=lambda t: (ALGOS.index(t[0][0]), ENVS.index(t[0][1]))):
    e2.append(OrderedDict(setting=f"warmup_steps[{algo}|{env}]", value=w, n_runs=n))
pd.DataFrame(e2).to_csv(HERE / "E2_run_settings.csv", index=False)
checks["e2_deviating_runs_train_steps_not_100000"] = [r["rel"] for r in runs if r["meta"]["experiment_config"]["train_steps"] != 100000]
checks["e2_runs_with_epochs_not_100"] = [r["rel"] for r in runs
                                         if len(json.loads((r["dir"] / "training_metrics.json").read_text())["epochs"]) != 100]
# git commits / time span of canonical runs
checks["e2_start_time_min_max_utc"] = [min(r["meta"]["start_time_utc"] for r in runs), max(r["meta"]["start_time_utc"] for r in runs)]
checks["e2_git_commit_count"] = len(tally(lambda m: m["git_commit"]))

# --------------------------------------------------------------------------------------------- fragments for the main file
def fmt(x, nd=6):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def md_table(df, cols=None, nd=6):
    cols = cols or list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


frag = {}
# E1 compact
e1c = e1[["group_id", "algo", "env", "sweep_name", "sweep_value", "is_baseline", "hidden_sizes", "utd", "batch_size",
          "rollout_max_length", "num_q", "horizon", "episodic", "n_seeds"]]
frag["e1_compact"] = md_table(e1c)
# sweep presence matrix (D3): per sweep -> algo/env -> values (incl baseline value)
lines = []
for sn in SWEEP_ORDER[1:]:
    for algo in ALGOS:
        row = []
        for env in ENVS:
            g = b2df[(b2df.algo == algo) & (b2df.env == env) & (b2df.sweep_name == sn)]
            vals = sorted(g.sweep_value.unique(), key=float)
            if vals:
                base_v = g[g.is_baseline].sweep_value.iloc[0]
                row.append(", ".join(f"{v}{'*' if v == base_v else ''}" for v in vals) + f" ({','.join(sorted(g.group_id.unique()))})")
            else:
                row.append("— absent")
        lines.append([sn, algo, row[0], row[1]])
frag["sweep_presence"] = md_table(pd.DataFrame(lines, columns=["sweep", "algo", "HalfCheetah-v5 (values; *=baseline value; group_ids)", "Ant-v5"]))
# B1 table (energy J mean ± sd and share)
t = b1df.copy()
t["E_J mean ± sd"] = [f"{a:.6g} ± {b:.4g}" for a, b in zip(t.energy_j_mean, t.energy_j_std_ddof1)]
t["share % (mean of per-seed ± sd)"] = [f"{a:.5g} ± {b:.3g}" for a, b in zip(t.share_of_TOTAL_pct_mean_of_per_seed_shares,
                                                                      t.share_of_TOTAL_pct_std_of_per_seed_shares_ddof1)]
t["kind"] = [k.split(" ")[0] for k in t.kind]
frag["b1_main"] = md_table(t[["algo", "env", "segment", "kind", "E_J mean ± sd", "share % (mean of per-seed ± sd)"]])
frag["b1_totals"] = md_table(b1df[b1df.segment == "TOTAL_MEASURED_TRAINING"][
    ["group_id", "algo", "env", "n_seeds", "energy_j_mean", "energy_j_std_ddof1", "energy_kwh_mean", "energy_kwh_std_ddof1",
     "matmul_flops_mean", "j_per_flop_ratio_of_means"]], nd=10)
frag["ratio_table"] = md_table(ratdf[["algo", "env", "sweep", "values_numeric_order", "rollout_energy_ratio", "rollout_flops_ratio",
                                       "gradient_updates_energy_ratio", "gradient_updates_flops_ratio"]], nd=6)
# D2 counts
mm = d2df[d2df.flop_type.isin(["matmul"]) & d2df.j_per_flop_ratio_of_means.notna()]
cnt = (mm.groupby(["segment"]).size().rename("n_group_segment_pairs_with_defined_matmul_J_per_FLOP")).reset_index()
cnt_env = mm.groupby(["segment", "env"]).size().unstack(fill_value=0).reset_index()
frag["d2_counts"] = md_table(cnt.merge(cnt_env, on="segment"))
nd = d2df[~d2df.flop_type.isin(["matmul", "mixed_total", "matmul_derived"])]
frag["d2_undefined"] = md_table(nd.groupby(["segment", "flop_type"]).size().rename("n_pairs").reset_index())
frag["d2_derived"] = md_table(d2df[d2df.flop_type.isin(["matmul_derived"])].groupby(["segment", "flop_type"]).size().rename("n_pairs").reset_index())
frag["d2_total"] = md_table(d2df[d2df.flop_type.isin(["mixed_total"])].groupby(["segment", "flop_type"]).size().rename("n_pairs").reset_index())
# E2 settings (compact)
e2df = pd.DataFrame(e2)
frag["e2"] = md_table(e2df)
for k, v in frag.items():
    (HERE / f"frag_{k}.md").write_text(v + "\n", encoding="utf-8")

(HERE / "checks.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
print(json.dumps(checks, indent=2, default=str))
print("groups:", len(gkeys), "runs:", len(runs))
