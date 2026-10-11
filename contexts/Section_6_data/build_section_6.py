"""Collects the facts for thesis Section 6 (GPU clock comparison, MBPO / HalfCheetah-v5 / seed 0).
Reads results/ and flop_analysis/output/; writes only CSVs + findings.json next to this file.
Run from the repo root:  .venv/Scripts/python contexts/Section_6_data/build_section_6.py
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd

ROOT = os.getcwd()
OUT = os.path.join(ROOT, "contexts", "Section_6_data")
sys.path.insert(0, ROOT)
import idle_power_analysis as ipa

J = 3.6e6
BASE = "results/mbpo/HalfCheetah-v5/seed_0/"
RUNS = {"2000MHz": BASE + "20260904_122958", "200MHz": BASE + "20260904_130915"}
CANON = {331, 958, 14577, 43611, 85062}
MEASURED = ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "gradient_updates"]
ALLOC = ["buffer_sample", "critic_update", "actor_update", "target_update"]
F = {}  # findings


def prefix(t):
    return re.sub(r"_\d+$", "", t)


def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, p + k + "."))
        else:
            o[p + k] = v
    return o


meta = {k: json.load(open(v + "/metadata.json")) for k, v in RUNS.items()}
seg = {k: json.load(open(v + "/segment_energy.json")) for k, v in RUNS.items()}
tm = {k: json.load(open(v + "/training_metrics.json")) for k, v in RUNS.items()}
em = {k: pd.read_csv(glob.glob(v + "/emissions_base_*.csv")[0]) for k, v in RUNS.items()}
emr = {k: pd.read_csv(v + "/emissions.csv").iloc[-1] for k, v in RUNS.items()}
for k in em:
    em[k]["prefix"] = em[k]["task_name"].map(prefix)
    em[k]["run"] = k
    em[k]["energy_j"] = em[k]["energy_consumed"] * J
perrun = pd.read_csv("flop_analysis/output/per_run_energy_per_flop.csv")

# ---------- A: runs with non-2000/2000 lock; all MBPO/HC/seed0 runs ----------
non, hc = [], []
for p in sorted(glob.glob("results/**/metadata.json", recursive=True)):
    m = json.load(open(p)); c = m["experiment_config"]; a = m["algo_config"]
    d = os.path.dirname(p).replace("\\", "/")
    if (c.get("gpu_min_clock_mhz"), c.get("gpu_max_clock_mhz")) != (2000, 2000):
        non.append(dict(run_dir=d, algo=m["algo_name"], env=m["env_id"], seed=m["seed"], gpu_min_clock_mhz=c.get("gpu_min_clock_mhz"),
                        gpu_max_clock_mhz=c.get("gpu_max_clock_mhz"), lock_gpu_clocks=c.get("lock_gpu_clocks"), torch=m["torch_version"], codecarbon=m["codecarbon_version"]))
    if m["algo_name"] == "mbpo" and m["env_id"] == "HalfCheetah-v5" and m["seed"] == 0:
        hc.append(dict(run_dir=d, timestamp=os.path.basename(d), git_commit=m["git_commit"], gpu_min_clock_mhz=c["gpu_min_clock_mhz"], gpu_max_clock_mhz=c["gpu_max_clock_mhz"],
                       hidden_sizes=a["hidden_sizes"], batch_size=a["batch_size"], updates_per_env_step=a["updates_per_env_step"],
                       rollout_min_length=a["rollout_min_length"], rollout_max_length=a["rollout_max_length"], rollout_min_epoch=a["rollout_min_epoch"], rollout_max_epoch=a["rollout_max_epoch"],
                       warmup_steps=c["warmup_steps"], train_steps=c["train_steps"], steps_per_epoch=c["steps_per_epoch"], model_hidden_sizes=a["model_hidden_sizes"],
                       torch=m["torch_version"], codecarbon=m["codecarbon_version"], gpu=m["cuda_device_name"], training_metrics_exists=os.path.exists(d + "/training_metrics.json")))
pd.DataFrame(non).to_csv(f"{OUT}/A1_non_2000_lock_runs.csv", index=False)
pd.DataFrame(hc).to_csv(f"{OUT}/A2_mbpo_hc_seed0_runs.csv", index=False)
F["A1_count"] = len(non)

# A4 diff
fa, fb = flat(meta["2000MHz"]), flat(meta["200MHz"])
rows = [dict(field=k, v_2000MHz_run=fa.get(k, "<absent>"), v_200MHz_run=fb.get(k, "<absent>"), differs=fa.get(k, "<absent>") != fb.get(k, "<absent>")) for k in sorted(set(fa) | set(fb))]
pd.DataFrame(rows).to_csv(f"{OUT}/A4_metadata_diff.csv", index=False)

# A6 completeness
comp = []
for k in RUNS:
    vc = em[k]["prefix"].value_counts().to_dict()
    log = open(RUNS[k] + "/run.log").read()
    comp.append(dict(run=k, **{f"n_{p}": vc.get(p, 0) for p in MEASURED + ["warmup", "idle_baseline_head", "idle_baseline_tail"]},
                     n_rows_emissions_base=len(em[k]), n_log_lines=len(log.splitlines()),
                     n_WARNING_or_ERROR=len(re.findall(r"\[(WARNING|ERROR)\]", log)),
                     rapl_fallback=("default power consumption of 4 W per thread" in log), geoloc_fallback=bool(re.search("geoloc|default country", log, re.I)),
                     thermal_timeout=bool(re.search("timed out|timeout", log, re.I)), permission_failure=bool(re.search("permission|denied|not permitted|root", log, re.I)),
                     gpu_control_lines=len(re.findall("gpu_control|Locked GPU|Reset GPU|governor", log, re.I))))
pd.DataFrame(comp).to_csv(f"{OUT}/A6_completeness_and_log.csv", index=False)

# log diff (timestamps stripped)
def strip(l): return re.sub(r"^\S+ \S+ ", "", l)
la = [strip(l) for l in open(RUNS["2000MHz"] + "/run.log").read().splitlines()]
lb = [strip(l) for l in open(RUNS["200MHz"] + "/run.log").read().splitlines()]
ld = [(i + 1, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
pd.DataFrame(ld, columns=["line", "log_2000MHz", "log_200MHz"]).to_csv(f"{OUT}/A6_runlog_diff_timestamps_stripped.csv", index=False)

# ---------- C ----------
pr = perrun[perrun.run_dir.isin(RUNS.values())].copy()
pr["clock"] = pr.run_dir.map({v: k for k, v in RUNS.items()})
pr.to_csv(f"{OUT}/per_run_rows.csv", index=False)
pt = pd.concat([em[k] for k in RUNS], ignore_index=True)
pt.drop(columns=[c for c in ["project_name", "run_id", "country_name", "country_iso_code", "region", "cloud_provider", "cloud_region", "os", "python_version", "cpu_model", "gpu_model", "longitude", "latitude"] if c in pt]).to_csv(f"{OUT}/per_task_rows.csv", index=False)

def csvrow(k, s):
    r = pr[(pr.clock == k) & (pr.segment == s)]
    return r.iloc[0] if len(r) else None

summ = []
for k in RUNS:
    tot = csvrow(k, "TOTAL_MEASURED_TRAINING")
    for s in MEASURED + ALLOC + ["TOTAL_MEASURED_TRAINING", "warmup", "idle_baseline_head", "idle_baseline_tail"]:
        e = dur = None
        if s in MEASURED:
            d = em[k][em[k].prefix == s]; e = d.energy_j.sum(); dur = d.duration.sum(); p_ = e / dur; kind = "measured"
            r = None
        elif s in ALLOC:
            r = csvrow(k, s); e, dur, p_ = r.total_energy_joules, r.duration_s, r.mean_power_w; kind = "allocated (within gradient_updates)"
        else:
            r = csvrow(k, s); e, dur, p_ = r.total_energy_joules, r.duration_s, r.mean_power_w; kind = "total" if s.startswith("TOTAL") else "other"
        r = csvrow(k, s)
        summ.append(dict(clock=k, segment=s, kind=kind, energy_j=e, energy_kwh=e / J, share_of_training_total=(e / tot.total_energy_joules) if (s in MEASURED + ALLOC + ["TOTAL_MEASURED_TRAINING"]) else None,
                         duration_s=dur, mean_power_w=p_, total_flops=(r.total_flops if r is not None else None), flop_type=(r.flop_type if r is not None else None),
                         energy_per_flop_j_per_flop=(r.energy_per_flop_j_per_flop if r is not None else None), call_count=(r.call_count if r is not None else None)))
summ = pd.DataFrame(summ)
summ.to_csv(f"{OUT}/C_segment_summary.csv", index=False)

# consistency: CSV total vs. sum of measured
chk = []
for k in RUNS:
    tot = csvrow(k, "TOTAL_MEASURED_TRAINING")
    sm = summ[(summ.clock == k) & summ.segment.isin(MEASURED)]
    sa = summ[(summ.clock == k) & summ.segment.isin(ALLOC)]
    chk.append(dict(clock=k, csv_total_energy_j=tot.total_energy_joules, sum_measured_energy_j=sm.energy_j.sum(), csv_total_duration_s=tot.duration_s, sum_measured_duration_s=sm.duration_s.sum(),
                    sum_alloc_energy_j=sa.energy_j.sum(), gradient_updates_measured_energy_j=float(sm[sm.segment == "gradient_updates"].energy_j.iloc[0]),
                    sum_alloc_wall_time_s=sa.duration_s.sum(), gradient_updates_measured_duration_s=float(sm[sm.segment == "gradient_updates"].duration_s.iloc[0])))
pd.DataFrame(chk).to_csv(f"{OUT}/C_consistency_checks.csv", index=False)

# segment_energy.json all keys
rows = []
for k in RUNS:
    for key, v in seg[k].items():
        if isinstance(v, dict):
            for kk, vv in v.items(): rows.append(dict(clock=k, key=f"{key}.{kk}", value=vv))
        else:
            rows.append(dict(clock=k, key=key, value=v))
pd.DataFrame(rows).to_csv(f"{OUT}/C3_segment_energy_json_all_keys.csv", index=False)

# C4 cpu/gpu/ram split per measured task and training total
sp = []
for k in RUNS:
    tot = {c: 0.0 for c in ["cpu", "gpu", "ram", "total"]}
    for s in MEASURED + ["warmup", "idle_baseline_head", "idle_baseline_tail"]:
        d = em[k][em[k].prefix == s]
        e = {"cpu": d.cpu_energy.sum() * J, "gpu": d.gpu_energy.sum() * J, "ram": d.ram_energy.sum() * J, "total": d.energy_j.sum()}
        if s in MEASURED:
            for c in tot: tot[c] += e[c]
        sp.append(dict(clock=k, task=s, cpu_j=e["cpu"], gpu_j=e["gpu"], ram_j=e["ram"], sum_components_j=e["cpu"] + e["gpu"] + e["ram"], total_energy_consumed_j=e["total"],
                       cpu_pct=100 * e["cpu"] / e["total"], gpu_pct=100 * e["gpu"] / e["total"], ram_pct=100 * e["ram"] / e["total"]))
    sp.append(dict(clock=k, task="TRAINING_TOTAL (sum of 4 measured)", cpu_j=tot["cpu"], gpu_j=tot["gpu"], ram_j=tot["ram"], sum_components_j=tot["cpu"] + tot["gpu"] + tot["ram"], total_energy_consumed_j=tot["total"],
                   cpu_pct=100 * tot["cpu"] / tot["total"], gpu_pct=100 * tot["gpu"] / tot["total"], ram_pct=100 * tot["ram"] / tot["total"]))
pd.DataFrame(sp).to_csv(f"{OUT}/C4_cpu_gpu_ram_split.csv", index=False)

# C5 whole-run row
w = []
for k in RUNS:
    r = emr[k]
    s_all = em[k]
    w.append(dict(clock=k, run_duration_s=r["duration"], run_energy_kwh=r["energy_consumed"], run_energy_j=r["energy_consumed"] * J, cpu_energy_j=r["cpu_energy"] * J, gpu_energy_j=r["gpu_energy"] * J, ram_energy_j=r["ram_energy"] * J,
                  sum_all_task_rows_energy_j=s_all.energy_j.sum(), diff_run_minus_sum_tasks_j=r["energy_consumed"] * J - s_all.energy_j.sum(),
                  sum_all_task_rows_duration_s=s_all.duration.sum(), diff_run_duration_minus_sum_tasks_s=r["duration"] - s_all.duration.sum(),
                  cc_run_avg_cpu_power_w=r["cpu_power"], cc_run_avg_gpu_power_w=r["gpu_power"], cc_run_avg_ram_power_w=r["ram_power"]))
pd.DataFrame(w).to_csv(f"{OUT}/C5_whole_run_row.csv", index=False)

# C6 per task stats
st = []
for k in RUNS:
    for s in MEASURED:
        d = em[k][em[k].prefix == s].duration
        st.append(dict(clock=k, task=s, count=len(d), min_s=d.min(), median_s=d.median(), max_s=d.max(), sum_s=d.sum(), n_shorter_than_1s=int((d < 1.0).sum())))
pd.DataFrame(st).to_csv(f"{OUT}/C6_task_duration_stats.csv", index=False)

# C7 work done
wd = []
for k in RUNS:
    ep, eps = tm[k]["epochs"], tm[k]["episodes"]
    tr = [e["return"] for e in eps if e["phase"] == "train"]
    n10 = max(1, len(tr) // 10)
    fits = [e["model_train_epochs"] for e in ep if e.get("model_train_epochs") is not None]
    syn = [e.get("synthetic_transitions_generated") for e in ep]
    dm = csvrow(k, "dynamics_model_update"); sr = csvrow(k, "synthetic_rollout_generation")
    wd.append(dict(clock=k, env_steps_train=meta[k]["experiment_config"]["train_steps"], n_epochs=len(ep), n_train_episodes=len(tr), n_warmup_episodes=sum(e["phase"] == "warmup" for e in eps),
                   n_model_fits=len(fits), fit_epochs_total=sum(fits), fit_epochs_first=fits[0], fit_epochs_min=min(fits), fit_epochs_max=max(fits),
                   optimiser_steps_dynamics_call_count=dm.call_count, dynamics_total_flops=dm.total_flops,
                   synthetic_transitions_generated_last_epoch=syn[-1], synthetic_transitions_sum_over_epochs=sum(s for s in syn if s is not None), synth_rollout_call_count=sr.call_count,
                   n_epochs_with_model_train_epochs_key=len(fits), last10pct_n_episodes=n10, last10pct_mean_return=float(np.mean(tr[-n10:])), max_train_return=max(tr),
                   final_cumulative_reward=ep[-1]["cumulative_reward"]))
pd.DataFrame(wd).to_csv(f"{OUT}/C7_work_done.csv", index=False)
# identical-per-epoch check
a = [(e["model_train_epochs"], e["synthetic_transitions_generated"], e["mean_episode_return"], e.get("model_holdout_mse")) for e in tm["2000MHz"]["epochs"]]
b = [(e["model_train_epochs"], e["synthetic_transitions_generated"], e["mean_episode_return"], e.get("model_holdout_mse")) for e in tm["200MHz"]["epochs"]]
F["training_metrics_epochs_identical"] = a == b
F["n_epoch_rows_differing"] = sum(x != y for x, y in zip(a, b))
ea = [e["return"] for e in tm["2000MHz"]["episodes"]]; eb = [e["return"] for e in tm["200MHz"]["episodes"]]
F["episode_returns_identical"] = ea == eb

# C8 idle
idle = ipa.load_runs("results", seeds={0})
idle = idle[idle.run_dir.isin([v.replace("results/", "") for v in RUNS.values()])].copy()
idle["clock"] = idle.run_dir.map({v.replace("results/", ""): k for k, v in RUNS.items()})
idle.drop(columns=["variant"]).to_csv(f"{OUT}/C8_idle_power_two_runs.csv", index=False)

# C9 idle-floor diagnostic
fl = []
for k in RUNS:
    ir = idle[idle.clock == k].iloc[0]
    floor = (ir.head_total_w_corr + ir.tail_total_w_corr) / 2
    tot_e = tot_d = 0
    for s in MEASURED:
        d = em[k][em[k].prefix == s]; e = d.energy_j.sum(); du = d.duration.sum(); tot_e += e; tot_d += du
        fl.append(dict(clock=k, task=s, idle_floor_w=floor, duration_s=du, gross_energy_j=e, idle_floor_energy_j=floor * du, idle_floor_fraction=floor * du / e))
    fl.append(dict(clock=k, task="TRAINING_TOTAL", idle_floor_w=floor, duration_s=tot_d, gross_energy_j=tot_e, idle_floor_energy_j=floor * tot_d, idle_floor_fraction=floor * tot_d / tot_e))
pd.DataFrame(fl).to_csv(f"{OUT}/C9_idle_floor_diagnostic.csv", index=False)

# ---------- D ----------
dr = []
for s in MEASURED + ["TOTAL_MEASURED_TRAINING"] + ALLOC:
    a_ = summ[(summ.clock == "2000MHz") & (summ.segment == s)].iloc[0]; b_ = summ[(summ.clock == "200MHz") & (summ.segment == s)].iloc[0]
    dr.append(dict(segment=s, kind=a_.kind, energy_j_2000=a_.energy_j, energy_j_200=b_.energy_j, energy_ratio_200_over_2000=b_.energy_j / a_.energy_j, energy_diff_j_200_minus_2000=b_.energy_j - a_.energy_j,
                   duration_s_2000=a_.duration_s, duration_s_200=b_.duration_s, duration_ratio=b_.duration_s / a_.duration_s, duration_diff_s=b_.duration_s - a_.duration_s,
                   power_w_2000=a_.mean_power_w, power_w_200=b_.mean_power_w, power_ratio=b_.mean_power_w / a_.mean_power_w, power_diff_w=b_.mean_power_w - a_.mean_power_w))
pd.DataFrame(dr).to_csv(f"{OUT}/D_ratios_200_over_2000.csv", index=False)
ia, ib = idle[idle.clock == "2000MHz"].iloc[0], idle[idle.clock == "200MHz"].iloc[0]
di = []
for ph in ("head", "tail"):
    for c in ipa.COMPONENTS:
        for suf, lab in (("", "as_recorded"), ("_corr", "window_corrected")):
            x, y = ia[f"{ph}_{c}_w{suf}"], ib[f"{ph}_{c}_w{suf}"]
            di.append(dict(window=ph, component=c, basis=lab, power_w_2000=x, power_w_200=y, ratio_200_over_2000=y / x, diff_w_200_minus_2000=y - x))
pd.DataFrame(di).to_csv(f"{OUT}/D_idle_ratios.csv", index=False)

# ---------- E ----------
cmp_ = csvrow("2000MHz", "TOTAL_MEASURED_TRAINING")
sel = perrun[(perrun.algo == "mbpo") & (perrun.env_id == "HalfCheetah-v5") & (perrun.architecture_signature == cmp_.architecture_signature) & (perrun.updates_per_env_step == cmp_.updates_per_env_step)
             & (perrun.mbpo_rollout_regime == cmp_.mbpo_rollout_regime) & (perrun.segment == "TOTAL_MEASURED_TRAINING")]
F["E_candidate_runs_all_seeds"] = sel[["seed", "run_dir", "included_in_cross_seed_avg"]].values.tolist()
can = sel[sel.seed.isin(CANON)].copy()
cfgdiff = []
ref = flat(meta["2000MHz"])
IGN = {"seed", "experiment_config.seed", "start_time_utc", "end_time_utc", "git_commit"}
for rd in can.run_dir:
    m = flat(json.load(open(rd + "/metadata.json")))
    for key in sorted(set(ref) | set(m)):
        if key in IGN or key.startswith("thermal_gate."): continue
        if ref.get(key, "<absent>") != m.get(key, "<absent>"):
            cfgdiff.append(dict(canonical_run=rd, field=key, comparator=ref.get(key, "<absent>"), canonical=m.get(key, "<absent>")))
pd.DataFrame(cfgdiff, columns=["canonical_run", "field", "comparator", "canonical"]).to_csv(f"{OUT}/E_config_diff_vs_canonical.csv", index=False)
F["E_n_config_field_diffs"] = len(cfgdiff)
F["E_canonical_commits"] = sorted({json.load(open(rd + "/metadata.json"))["git_commit"] for rd in can.run_dir})
F["E_canonical_versions"] = sorted({(json.load(open(rd + "/metadata.json"))["torch_version"], json.load(open(rd + "/metadata.json"))["codecarbon_version"]) for rd in can.run_dir})
can = can[["seed", "run_dir", "total_energy_joules", "total_energy_kwh", "duration_s", "mean_power_w", "total_flops", "energy_per_flop_j_per_flop"]]
can.to_csv(f"{OUT}/E_canonical_per_seed.csv", index=False)
est = []
for col in ["total_energy_joules", "duration_s", "mean_power_w"]:
    v = can[col]; c200 = csvrow("200MHz", "TOTAL_MEASURED_TRAINING")[col]; c2000 = cmp_[col]
    est.append(dict(metric=col, n=len(v), canonical_mean=v.mean(), canonical_sd_ddof1=v.std(ddof=1), canonical_min=v.min(), canonical_max=v.max(),
                    comparator_2000=c2000, comparator_minus_mean=c2000 - v.mean(), comparator_minus_mean_over_sd=(c2000 - v.mean()) / v.std(ddof=1),
                    run_200_value=c200, run_200_over_canonical_mean=c200 / v.mean()))
pd.DataFrame(est).to_csv(f"{OUT}/E_canonical_stats.csv", index=False)
cs = pd.read_csv("flop_analysis/output/cross_seed_energy_per_flop.csv")
cs = cs[(cs.algo == "mbpo") & (cs.env_id == "HalfCheetah-v5") & (cs.architecture_signature == cmp_.architecture_signature) & (cs.updates_per_env_step == cmp_.updates_per_env_step)
        & (cs.mbpo_rollout_regime == cmp_.mbpo_rollout_regime) & (cs.segment == "TOTAL_MEASURED_TRAINING")]
cs.to_csv(f"{OUT}/E_cross_seed_csv_row.csv", index=False)

# ---------- F ----------
dev = []
for p in sorted(glob.glob("results/**/metadata.json", recursive=True)):
    m = json.load(open(p))
    if m["seed"] not in CANON: dev.append((m["algo_name"], m["env_id"], m["seed"]))
dv = pd.Series(dev).value_counts().rename_axis("algo_env_seed").reset_index(name="n_runs") if dev else pd.DataFrame()
dv = pd.DataFrame([dict(algo=a, env=e, seed=s, n_runs=n) for (a, e, s), n in pd.Series(dev).value_counts().items()]).sort_values(["algo", "env", "seed"])
dv.to_csv(f"{OUT}/F3_dev_runs_by_algo_env_seed.csv", index=False)
F["F3_total_dev_runs"] = len(dev)
F["F3_total_runs"] = len(glob.glob("results/**/metadata.json", recursive=True))
bl = perrun[(perrun.algo == "mbpo") & (perrun.env_id == "HalfCheetah-v5") & perrun.seed.isin(CANON) & (perrun.segment == "TOTAL_MEASURED_TRAINING")]
F["F2_canonical_mbpo_hc_batch_sizes_by_signature"] = bl.groupby("architecture_signature").size().to_dict()
bsz = {json.load(open(rd + "/metadata.json"))["algo_config"]["batch_size"] for rd in bl.run_dir}
F["F2_canonical_mbpo_hc_batch_sizes_set"] = sorted(bsz)
mb = {}
for rd in bl.run_dir:
    a = json.load(open(rd + "/metadata.json"))["algo_config"]
    mb.setdefault((tuple(a["hidden_sizes"]), a["batch_size"], a["updates_per_env_step"]), 0); mb[(tuple(a["hidden_sizes"]), a["batch_size"], a["updates_per_env_step"])] += 1
F["F2_canonical_mbpo_hc_configs(hidden,batch,utd):n"] = {str(k): v for k, v in mb.items()}

# G: git diff between commits -> identical commit
F["same_commit"] = meta["2000MHz"]["git_commit"] == meta["200MHz"]["git_commit"]
F["thermal_gate"] = {k: meta[k]["thermal_gate"] for k in RUNS}
F["metadata_keys_run_start_temp_present"] = {k: "run_start_temp_c" in meta[k]["thermal_gate"] for k in RUNS}
json.dump(F, open(f"{OUT}/findings.json", "w"), indent=1, default=str)
print(json.dumps(F, indent=1, default=str))
