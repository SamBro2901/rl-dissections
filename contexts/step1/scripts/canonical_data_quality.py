"""Task 4: canonical-only data quality + execution order (read-only on results/, configs/, scripts/).
Run: .venv/Scripts/python.exe contexts/step1/scripts/canonical_data_quality.py
Writes 04_canonical_run_list.csv, 04_execution_timeline.csv, 04_results.json (consumed by make_04_md.py)."""
import csv, glob, json, os, re, subprocess, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import canonical_runs, load_all_meta, OUT, REPO, RESULTS, CANONICAL_SEEDS, read_base_csv

R = {}
runs = canonical_runs()
runs["run_path"] = runs.run_dir.map(lambda r: os.path.join(RESULTS, r))
S_ORDER = [331, 958, 14577, 43611, 85062]

# ------------------------------------------------------------------ 1. completeness
R["n_runs"] = len(runs)
R["per_algo_env_configs_per_seed"] = {f"{a}/{e}": int(g.groupby("seed").size().iloc[0]) for (a, e), g in runs.groupby(["algo", "env"])}
R["per_algo_env_seed_counts_all_equal"] = {f"{a}/{e}": g.groupby("seed").size().nunique() == 1 for (a, e), g in runs.groupby(["algo", "env"])}
grid = runs.groupby(["algo", "env", "config", "seed"]).size()
R["duplicates_same_algo_env_config_seed"] = [list(k) for k, v in grid.items() if v > 1]
cfgs = runs.groupby(["algo", "env", "config"]).seed.apply(lambda s: sorted(s)).to_dict()
R["configs_without_all_5_seeds"] = [list(k) for k, v in cfgs.items() if v != sorted(CANONICAL_SEEDS)]
R["n_configurations"] = len(cfgs)
# other (non-canonical) dirs
allm = load_all_meta()
R["non_canonical_run_dirs"] = Counter(f"{m['algo_name']}/{m['env_id']}/seed_{m['seed']}" for m in allm if m["seed"] not in CANONICAL_SEEDS)
R["total_run_dirs"] = len(allm)
# file completeness
need = ["metadata.json", "segment_energy.json", "training_metrics.json", "emissions.csv", "run.log"]
missing = []
for _, r in runs.iterrows():
    for f in need:
        if not os.path.exists(os.path.join(r.run_path, f)):
            missing.append((r.run_dir, f))
    if not glob.glob(os.path.join(r.run_path, "emissions_base_*.csv")):
        missing.append((r.run_dir, "emissions_base_*.csv"))
    if not r.meta.get("end_time_utc"):
        missing.append((r.run_dir, "metadata.end_time_utc"))
R["missing_files"] = missing
# protocol values
def cnt(fn):
    return {str(k): v for k, v in Counter(fn(r) for _, r in runs.iterrows()).items()}
R["train_steps"] = cnt(lambda r: r.meta["experiment_config"]["train_steps"])
R["steps_per_epoch"] = cnt(lambda r: r.meta["experiment_config"]["steps_per_epoch"])
R["warmup_steps_by_algo"] = {a: {str(k): v for k, v in Counter(r.meta["experiment_config"]["warmup_steps"] for _, r in g.iterrows()).items()} for a, g in runs.groupby("algo")}
R["idle_secs"] = cnt(lambda r: (r.meta["experiment_config"]["idle_baseline_seconds"], r.meta["experiment_config"]["idle_tail_seconds"], r.meta["experiment_config"]["settle_seconds"]))
tm_epochs, rollout_tasks = [], []
for _, r in runs.iterrows():
    tm = json.load(open(os.path.join(r.run_path, "training_metrics.json")))
    tm_epochs.append(len(tm["epochs"]))
    b = read_base_csv(r.run_dir)
    rollout_tasks.append(int(b.task_name.str.fullmatch(r"rollout_\d+").sum()))
    # last epoch env_step
runs["n_epochs_tm"], runs["n_rollout_tasks"] = tm_epochs, rollout_tasks
R["epochs_in_training_metrics"] = dict(Counter(tm_epochs))
R["rollout_task_rows_per_run"] = dict(Counter(rollout_tasks))
last_env_step = [json.load(open(os.path.join(r.run_path, "training_metrics.json")))["epochs"][-1]["env_step"] for _, r in runs.iterrows()]
R["last_epoch_env_step_distinct"] = dict(Counter(last_env_step))
R["N_env_equals_train_steps_all"] = bool(all(r.meta["experiment_config"]["train_steps"] == 100000 for _, r in runs.iterrows()))
upd = defaultdict(Counter)
for _, r in runs.iterrows():
    upd[f"{r.algo}"][str(r.meta["algo_config"].get("updates_per_env_step", "n/a (no field)"))] += 1
R["updates_per_env_step_by_algo_all_configs"] = {k: dict(v) for k, v in upd.items()}
b_upd = defaultdict(Counter)
for _, r in runs[runs.config == "baseline"].iterrows():
    b_upd[f"{r.algo}/{r.env}"][str(r.meta["algo_config"].get("updates_per_env_step", "n/a (no field)"))] += 1
R["updates_per_env_step_baseline"] = {k: dict(v) for k, v in b_upd.items()}
R["baseline_batch_size"] = {f"{a}/{e}": dict(Counter(str(r.meta['algo_config'].get('batch_size')) for _, r in g.iterrows())) for (a, e), g in runs[runs.config == "baseline"].groupby(["algo", "env"])}
R["baseline_hidden"] = {f"{a}/{e}": dict(Counter(str(r.meta['algo_config'].get('hidden_sizes')) for _, r in g.iterrows())) for (a, e), g in runs[runs.config == "baseline"].groupby(["algo", "env"])}
R["tdmpc2_config_keys"] = sorted(runs[runs.algo == "tdmpc2"].iloc[0].meta["algo_config"].keys())
R["configs_by_algo_env"] = {f"{a}/{e}": sorted(g.config.unique()) for (a, e), g in runs.groupby(["algo", "env"])}

# ------------------------------------------------------------------ 2. checks restricted to canonical runs
# (a) check_totals.py checks 1-3, restated (definitions copied from flop_analysis/check_totals.py)
TRAIN = {"rollout", "buffer_sample", "critic_update", "actor_update", "target_update", "dynamics_model_update",
         "synthetic_rollout_generation", "world_model_pretrain"}
EXCL = {"idle_baseline_head", "idle_baseline_tail", "warmup"}
SUB = ("buffer_sample", "critic_update", "actor_update", "target_update")
rows_by_run = defaultdict(list)
with open(os.path.join(REPO, "flop_analysis", "output", "per_run_energy_per_flop.csv"), newline="") as f:
    for row in csv.DictReader(f):
        rows_by_run[row["run_dir"]].append(row)
fails, mx = [], {"c1": 0.0, "c3": 0.0}
n1 = n2 = n3 = ncov = 0
for _, r in runs.iterrows():
    rel = "results/" + r.run_dir
    seg = json.load(open(os.path.join(r.run_path, "segment_energy.json")))
    unexpected = [k for k in seg if not k.startswith("_") and k not in TRAIN and k not in EXCL]
    if unexpected: fails.append(f"[coverage] {rel}: unexpected segments {unexpected}")
    if all(s in seg for s in SUB):
        b = read_base_csv(r.run_dir)
        meas = b[b.task_name.str.fullmatch(r"gradient_updates_\d+")].energy_consumed.sum()
        e = abs(sum(seg[s] for s in SUB) - meas) / abs(meas)
        mx["c3"] = max(mx["c3"], e); n3 += 1
        if e > 1e-9: fails.append(f"[check3] {rel}: rel err {e:.3e}")
    rows = rows_by_run.get(rel)
    if not rows: fails.append(f"[coverage] {rel}: no rows in per_run csv"); continue
    ncov += 1
    tot = [x for x in rows if x["segment"] == "TOTAL_MEASURED_TRAINING"]
    if len(tot) != 1: fails.append(f"[coverage] {rel}: {len(tot)} TOTAL rows"); continue
    exp_e = sum(v for k, v in seg.items() if k in TRAIN)
    e = abs(float(tot[0]["total_energy_kwh"]) - exp_e) / abs(exp_e)
    mx["c1"] = max(mx["c1"], e); n1 += 1
    if e > 1e-12: fails.append(f"[check1] {rel}: rel err {e:.3e}")
    msum = sum(int(x["total_flops"]) for x in rows if x["flop_type"] == "matmul")
    n2 += 1
    if int(tot[0]["total_flops"]) != msum: fails.append(f"[check2] {rel}: flops mismatch")
R["check_totals_canonical"] = dict(runs_with_csv_rows=ncov, check1_runs=n1, check1_max_rel_err=mx["c1"], check2_runs=n2, check3_runs=n3,
                                   check3_max_rel_err=mx["c3"], failures=fails)

# (b) run.log scans over canonical runs
LOG_PATTERNS = {
    "rapl_fallback_4W_per_thread": r"default power consumption of 4 W per thread",
    "geolocation_fallback": r"(?i)geoloc|fall(ing)? back to default country|Unable to retrieve|fallback.*country",
    "thermal_gate_timeout": r"(?i)reason': 'timeout|timed out waiting|max_wait",
    "thermal_gate_nvml_unavailable": r"nvml_unavailable|No NVML reference",
    "gpu_control_failure": r"Command failed|Command not found|Command timed out|Could not query GPU clock range|Could not read current CPU governor",
    "gpu_control_success_info": r"Locked GPU|Reset GPU|Set CPU governor",
    "codecarbon_any_warning": r"codecarbon \w+ @",
    "any_WARNING_level_line": r"\[WARNING\]",
    "any_ERROR_level_line": r"\[ERROR\]",
}
hits = {k: [] for k in LOG_PATTERNS}
reasons = Counter(); lines_types = Counter()
for _, r in runs.iterrows():
    txt = open(os.path.join(r.run_path, "run.log"), encoding="utf-8", errors="replace").read()
    for k, p in LOG_PATTERNS.items():
        if re.search(p, txt): hits[k].append(r.run_dir)
    m = re.search(r"Thermal gate result: (\{.*\})", txt)
    reasons[re.search(r"'reason': '([^']*)'", m.group(1)).group(1) if m else "<no thermal line>"] += 1
R["run_log_scan_patterns"] = LOG_PATTERNS
R["run_log_scan_hits"] = {k: dict(n_runs=len(v), runs=v[:10]) for k, v in hits.items()}
R["run_log_thermal_reason_counts"] = dict(reasons)
tg = [r.meta.get("thermal_gate", {}) for _, r in runs.iterrows()]
R["metadata_thermal_gate_reason"] = dict(Counter(str(t.get("reason")) for t in tg))
R["metadata_thermal_gate_gated"] = dict(Counter(str(t.get("gated")) for t in tg))
R["metadata_thermal_waited_s_max"] = max(t.get("waited_seconds", float("nan")) for t in tg)
R["metadata_thermal_missing_fields"] = {k: sum(1 for t in tg if k not in t) for k in ("final_temp_c", "run_start_temp_c", "run_end_temp_c")}

# ------------------------------------------------------------------ 3. evidence table
ec = [r.meta["experiment_config"] for _, r in runs.iterrows()]
R["meta_requested"] = {k: dict(Counter(str(e.get(k, "<missing>")) for e in ec)) for k in
                       ("lock_gpu_clocks", "gpu_min_clock_mhz", "gpu_max_clock_mhz", "set_persistence_mode", "set_cpu_performance_governor",
                        "thermal_gate_enabled", "tracking_mode", "measure_power_secs")}
# sweep-log coverage (stdout+stderr captured by tee in the sweep scripts)
def parse_log(path):
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    blocks, cur = [], None
    for i, ln in enumerate(lines):
        m = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \[(\d+)/(\d+)\] Starting: (.*)$", ln)
        if m:
            cur = dict(log=os.path.relpath(path, REPO).replace("\\", "/"), start_line=i + 1, ts=m.group(1), idx=int(m.group(2)), total=int(m.group(3)),
                       desc=m.group(4), lines=[], run_dir=None)
            blocks.append(cur)
        elif re.match(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] (Priming|sudo credential)", ln):
            cur = None
            blocks.append(dict(log=os.path.relpath(path, REPO).replace("\\", "/"), start_line=i + 1, ts=ln[1:20], kind="session_start", lines=[ln]))
        elif cur is not None:
            cur["lines"].append(ln)
            m2 = re.search(r"Run directory: (\S+)", ln)
            if m2: cur["run_dir"] = m2.group(1)
    return blocks
logs = sorted(set(glob.glob(os.path.join(RESULTS, "*.log")) + glob.glob(os.path.join(RESULTS, "*", "*.log")) + glob.glob(os.path.join(RESULTS, "*", "*", "*.log"))))
R["sweep_logs"] = [os.path.relpath(p, REPO).replace("\\", "/") for p in logs]
all_blocks = []
for p in logs:
    all_blocks += parse_log(p)
STD = [r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ \[INFO\] ", r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] ", r"^\[codecarbon WARNING @ \d\d:\d\d:\d\d\] Multiple instances of codecarbon are allowed", r"^$"]
nonstd = Counter(); nonstd_where = {}
blocks_by_run = {}
for b in all_blocks:
    if b.get("kind") == "session_start": continue
    for ln in b["lines"]:
        if not any(re.match(p, ln) for p in STD):
            key = re.sub(r"\d+", "#", ln.strip())[:160]
            nonstd[key] += 1; nonstd_where.setdefault(key, (b["log"], b["start_line"]))
    if b["run_dir"]:
        blocks_by_run.setdefault(b["run_dir"].replace("results/", "", 1), []).append(b)
covered = {r.run_dir: (r.run_dir in blocks_by_run) for _, r in runs.iterrows()}
runs["in_sweep_log"] = runs.run_dir.map(covered)
runs["sweep_log"] = runs.run_dir.map(lambda d: blocks_by_run[d][0]["log"] if d in blocks_by_run else "")
R["sweep_log_coverage_by_group"] = {f"{a}/{e}/{c}": f"{int(g.in_sweep_log.sum())}/{len(g)}" for (a, e, c), g in runs.groupby(["algo", "env", "config"])}
R["sweep_log_coverage_total"] = int(runs.in_sweep_log.sum())
R["sweep_log_nonstandard_lines"] = [dict(pattern=k, count=v, first_seen=nonstd_where[k]) for k, v in nonstd.most_common(40)]
# warnings of interest inside covered blocks
SW = {"gpu_control_failure": LOG_PATTERNS["gpu_control_failure"], "rapl_fallback": LOG_PATTERNS["rapl_fallback_4W_per_thread"],
      "geolocation": r"(?i)geoloc|country", "thermal_timeout_or_nvml": r"reason': '(?!reached_reference)|nvml_unavailable|No NVML"}
sw_hits = {k: 0 for k in SW}
sw_hit_runs = defaultdict(list)
for b in all_blocks:
    if b.get("kind") == "session_start": continue
    txt = "\n".join(b["lines"])
    for k, p in SW.items():
        if re.search(p, txt):
            sw_hits[k] += 1; sw_hit_runs[k].append((b["log"], b["start_line"], b["desc"]))
R["sweep_log_block_hits"] = {k: dict(n_blocks=v, examples=sw_hit_runs[k][:8]) for k, v in sw_hits.items()}
# the same scans over whole sweep logs (including preamble lines)
R["sweep_log_whole_file_counts"] = {os.path.relpath(p, REPO).replace("\\", "/"): {k: len(re.findall(pt, open(p, encoding="utf-8", errors="replace").read())) for k, pt in
                                    {**SW, "Locked/Reset/governor INFO": LOG_PATTERNS["gpu_control_success_info"]}.items()} for p in logs}
# group coverage by sweep family
def family(r):
    c = r.config
    if c == "baseline": return "baseline"
    if "updates_per_env_step" in c: return "utd"
    if "hidden_sizes" in c: return "width"
    if "batch_size" in c: return "batch"
    if "rollout_max_length" in c: return "mbpo_rollout_length"
    if "num_q" in c or "horizon" in c: return "tdmpc2_numq_horizon"
    return "other"
runs["family"] = runs.apply(family, axis=1)
R["evidence_coverage_by_family"] = {f: dict(n=len(g), in_sweep_log=int(g.in_sweep_log.sum())) for f, g in runs.groupby("family")}
R["evidence_coverage_baseline_by_algo_env"] = {f"{a}/{e}": dict(n=len(g), in_sweep_log=int(g.in_sweep_log.sum())) for (a, e), g in runs[runs.family == "baseline"].groupby(["algo", "env"])}
# indirect RAPL evidence: cpu idle power vs generic TDP fallback
idle = pd.read_csv(os.path.join(OUT, "01_idle_drift_per_run.csv"))
R["cpu_idle_head_w_corr_min_median_max"] = [idle.head_cpu_w_corr.min(), idle.head_cpu_w_corr.median(), idle.head_cpu_w_corr.max()]
R["cpu_model"] = Counter(read_base_csv(r.run_dir).cpu_model.iloc[0] for _, r in runs.iterrows())
R["cpu_count"] = Counter(int(read_base_csv(r.run_dir).cpu_count.iloc[0]) for _, r in runs.iterrows())
R["emissions_os"] = Counter(read_base_csv(r.run_dir).os.iloc[0] for _, r in runs.iterrows())
R["emissions_python"] = Counter(read_base_csv(r.run_dir).python_version.iloc[0] for _, r in runs.iterrows())
R["emissions_gpu_model"] = Counter(read_base_csv(r.run_dir).gpu_model.iloc[0] for _, r in runs.iterrows())
R["emissions_tracking_mode"] = Counter(read_base_csv(r.run_dir).tracking_mode.iloc[0] for _, r in runs.iterrows())
R["emissions_ram_total_gb"] = Counter(round(float(read_base_csv(r.run_dir).ram_total_size.iloc[0]), 3) for _, r in runs.iterrows())
R["emissions_country"] = Counter(read_base_csv(r.run_dir).country_iso_code.iloc[0] for _, r in runs.iterrows())

# ------------------------------------------------------------------ 4. execution order
runs["t0"] = pd.to_datetime(runs.start_time_utc)
runs["t1"] = pd.to_datetime(runs.end_time_utc)
runs = runs.sort_values("t0").reset_index(drop=True)
gap = (runs.t0.shift(-1) - runs.t1).dt.total_seconds()
R["gap_seconds_quantiles_p0_50_90_99_100"] = gap.dropna().quantile([0, .5, .9, .99, 1]).tolist()
R["gap_negative_overlap_count"] = int((gap < 0).sum())
THR = 2 * 3600
sess = [0]
for g_ in gap.iloc[:-1]:
    sess.append(sess[-1] + (1 if g_ > THR else 0))
runs["session"] = sess
R["campaign_first_start_utc"] = str(runs.t0.min()); R["campaign_last_end_utc"] = str(runs.t1.max())
R["n_sessions_gap_gt_2h"] = int(runs.session.nunique())
R["n_sessions_by_threshold_hours"] = {str(h): int(1 + (gap.iloc[:-1] > h * 3600).sum()) for h in (0.5, 1, 2, 4, 8, 12)}
srows = []
for s, g in runs.groupby("session"):
    srows.append(dict(session=int(s), start_utc=str(g.t0.min()), end_utc=str(g.t1.max()), n_runs=len(g), hours=round((g.t1.max() - g.t0.min()).total_seconds() / 3600, 2),
                      groups={f"{a}/{e.split('-')[0]}/{c}": int(n) for (a, e, c), n in g.groupby(["algo", "env", "config"]).size().items()}))
R["sessions"] = srows
tl = runs[["run_dir", "algo", "env", "config", "seed", "start_time_utc", "end_time_utc", "session", "family", "in_sweep_log", "sweep_log", "git_commit"]].copy()
tl.columns = ["run_directory", "algo", "env", "config", "seed", "start_time_utc", "end_time_utc", "session_id", "family", "in_sweep_log", "sweep_log", "git_commit"]
tl["run_directory"] = "results/" + tl.run_directory
tl.to_csv(os.path.join(OUT, "04_execution_timeline.csv"), index=False)
# block analysis in time order
blocks = []
for k, g in runs.groupby((runs[["algo", "env", "config"]].ne(runs[["algo", "env", "config"]].shift()).any(axis=1)).cumsum()):
    blocks.append(dict(algo=g.algo.iloc[0], env=g.env.iloc[0], config=g.config.iloc[0], seeds=g.seed.tolist(), start=str(g.t0.iloc[0]), session=int(g.session.iloc[0])))
R["n_time_ordered_blocks"] = len(blocks)
R["n_distinct_configs"] = len({(b["algo"], b["env"], b["config"]) for b in blocks})
R["blocks_of_5_in_canonical_seed_order"] = sum(1 for b in blocks if b["seeds"] == S_ORDER)
R["blocks_not_of_5_or_order"] = [b for b in blocks if b["seeds"] != S_ORDER][:20]
R["config_noncontiguous"] = [k for k, v in Counter((b["algo"], b["env"], b["config"]) for b in blocks).items() if v > 1]
# per sweep: from the logs, nesting of loops
seqs = defaultdict(list)
for b in all_blocks:
    if b.get("kind") == "session_start" or not b["run_dir"]: continue
    d = dict(re.findall(r"(\w+)=(\S+)", b["desc"]))
    seqs[b["log"]].append((d.get("algo"), d.get("env"), int(d["seed"]), b["desc"]))
order_class = {}
for lg, s in seqs.items():
    seeds_ = [x[2] for x in s]
    changes = sum(1 for i in range(1, len(seeds_)) if seeds_[i] != seeds_[i - 1])
    nested = [(x[0], x[1], re.sub(r"seed=\d+", "", x[3]).strip()) for x in s]
    cfg_blocks = [list(g) for _, g in __import__("itertools").groupby(range(len(s)), key=lambda i: nested[i])]
    seed_order_ok = all([s[i][2] for i in blk] == S_ORDER for blk in cfg_blocks if len(blk) == 5)
    order_class[lg] = dict(n_runs=len(s), seed_changes_between_consecutive_runs=changes, n_config_blocks=len(cfg_blocks),
                           block_sizes=dict(Counter(len(b_) for b_ in cfg_blocks)), all_5blocks_in_canonical_seed_order=seed_order_ok,
                           first=s[0][3], last=s[-1][3])
R["log_order_classification"] = order_class

# ------------------------------------------------------------------ 5. seeds fixed before the runs? (git history)
def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, encoding="utf-8").stdout.strip()
res = {}
for term in ("14577", "43611", "85062", "331 958", "331, 958"):
    out = git("log", "--all", "--reverse", f"-S{term}", "--format=%H|%aI|%cI|%s", "--", "*.py", "*.sh", "*.md", "*.txt", "*.json")
    res[term] = out.split("\n")[:6]
R["git_first_commits_mentioning_seeds"] = res
R["earliest_canonical_run_start_utc"] = str(runs.t0.min()); R["earliest_canonical_run_dir"] = runs.run_dir.iloc[0]
R["git_total_commits"] = int(git("rev-list", "--count", "HEAD") or 0)
R["git_first_commits"] = git("log", "--reverse", "--format=%h|%aI|%s").split("\n")[:12]

# ------------------------------------------------------------------ 6. UTD sweep completeness
utd = runs[runs.config.str.contains("updates_per_env_step")]
R["utd_counts_by_algo_env_config"] = {f"{a}/{e}/{c}": int(n) for (a, e, c), n in utd.groupby(["algo", "env", "config"]).size().items()}
R["utd_sweep_log_td3_launch_blocks"] = None
ub = [b for b in all_blocks if b.get("log") == "results/_utd_sweep.log" and b.get("kind") != "session_start"]
R["utd_log_blocks_total"] = len(ub)
R["utd_log_blocks_by_algo_env"] = dict(Counter(re.search(r"algo=(\S+) env=(\S+)", b["desc"]).group(1) + "/" + re.search(r"algo=(\S+) env=(\S+)", b["desc"]).group(2) for b in ub))
R["utd_log_td3_blocks"] = sum(1 for b in ub if "algo=td3" in b["desc"])
R["utd_log_td3_blocks_with_run_dir_and_complete"] = sum(1 for b in ub if "algo=td3" in b["desc"] and b["run_dir"] and any("Run complete" in l for l in b["lines"]))
R["utd_log_td3_blocks_ok_in_run_dirs"] = sum(1 for b in ub if "algo=td3" in b["desc"] and b["run_dir"] and os.path.isdir(os.path.join(REPO, b["run_dir"])))
R["utd_log_sessions"] = [(b["ts"], b["start_line"]) for b in all_blocks if b.get("kind") == "session_start" and b["log"] == "results/_utd_sweep.log"]
R["utd_status_file_mixed"] = json.load(open(os.path.join(RESULTS, "_utd_sweep_status.json")))["total_runs"]

# ------------------------------------------------------------------ 7. TD-MPC2 false start
tb = [b for b in all_blocks if b["log"] == "results/tdmpc2/_numq_horizon_sweep.log"]
fs = []
for i, b in enumerate(tb):
    if b.get("kind") == "session_start":
        fs.append(dict(kind="session_start", ts=b["ts"], line=b["start_line"])); continue
    txt = "\n".join(b["lines"])
    err = re.findall(r"^(\w*Error[^\n]*)$", txt, re.M)
    fs.append(dict(kind="launch", idx=b["idx"], desc=b["desc"], ts=b["ts"], line=b["start_line"], run_dir=b["run_dir"],
                   failed_line=next((l for l in b["lines"] if "FAILED" in l), None), error=err[:1], n_lines=len(b["lines"]), closing_line=(b["lines"][-1] if b["lines"] else None)))
R["tdmpc2_false_start_blocks"] = fs[:12]
R["tdmpc2_run_dirs_created_2026_09_18_12h"] = sorted(glob.glob(os.path.join(RESULTS, "tdmpc2", "*", "seed_*", "20260918_12*")) and [os.path.relpath(p, RESULTS) for p in glob.glob(os.path.join(RESULTS, "tdmpc2", "*", "seed_*", "20260918_12*"))])
R["tdmpc2_run_dirs_20260918_1218_to_1220"] = [os.path.relpath(p, RESULTS).replace("\\", "/") for p in glob.glob(os.path.join(RESULTS, "tdmpc2", "*", "seed_*", "20260918_121[89]*")) + glob.glob(os.path.join(RESULTS, "tdmpc2", "*", "seed_*", "20260918_1220*"))]
lines_all = open(os.path.join(RESULTS, "tdmpc2", "_numq_horizon_sweep.log"), encoding="utf-8").read().split("\n")
R["tdmpc2_log_lines_62_to_68"] = lines_all[61:68]
R["tdmpc2_log_clock_lock_mentions_before_second_attempt"] = sum(1 for l in lines_all[:65] if re.search(r"Locked|Reset|governor|nvidia-smi|lgc|rgc|Command failed", l))
R["tdmpc2_status_json_done_counts"] = dict(Counter(x["status"] for x in json.load(open(os.path.join(RESULTS, "tdmpc2", "_numq_horizon_sweep_status.json")))["runs"]))

# ------------------------------------------------------------------ 8. code consistency
runs["fam_label"] = runs.family
cm = runs.groupby(["algo", "env", "family", "config"]).git_commit.apply(lambda s: dict(Counter(x[:7] for x in s))).to_dict()
R["commits_by_group"] = {"/".join(k): v for k, v in cm.items()}
R["distinct_commits"] = {c[:7]: int(n) for c, n in runs.git_commit.value_counts().items()}
commit_info = {}
for c in runs.git_commit.unique():
    ex = subprocess.run(["git", "-C", REPO, "cat-file", "-e", c + "^{commit}"], capture_output=True).returncode == 0
    commit_info[c[:7]] = dict(exists=ex, date=git("log", "-1", "--format=%aI", c) if ex else None, subject=git("log", "-1", "--format=%s", c) if ex else None)
R["commit_info"] = commit_info
PATHS = ["algorithms", "experiment_runner.py", "run_experiment.py", "configs/config.py", "utils"]
def first_commit(algo, env, cfg):
    g = runs[(runs.algo == algo) & (runs.env == env) & (runs.config == cfg)]
    return g.git_commit.iloc[0], g.git_commit.nunique()
c_ant1, n_ant1 = first_commit("mbpo", "Ant-v5", "rollout_max_length=1")
c_hcb, n_hcb = first_commit("mbpo", "HalfCheetah-v5", "baseline")
R["mbpo_ant_rollout1_commit"] = c_ant1[:7]; R["mbpo_hc_baseline_commit"] = c_hcb[:7]
R["mbpo_ant_rollout1_n_commits"] = n_ant1; R["mbpo_hc_baseline_n_commits"] = n_hcb
R["diff_stat_hc_baseline_to_ant_rollout1_training_code"] = git("diff", "--stat", c_hcb, c_ant1, "--", *PATHS)
R["diff_stat_full_hc_baseline_to_ant_rollout1"] = git("diff", "--stat", c_hcb, c_ant1)
R["commits_between_hc_baseline_and_ant_rollout1_training_code"] = git("log", "--format=%h|%aI|%s", f"{c_hcb}..{c_ant1}", "--", *PATHS).split("\n")
# all pairwise: distinct commit sequence and training-code diffs between consecutive commits (by date)
dc = sorted({c for c in runs.git_commit.unique()}, key=lambda c: git("log", "-1", "--format=%at", c))
R["distinct_commits_in_time_order"] = [c[:7] for c in dc]
R["training_code_diff_between_consecutive_distinct_commits"] = {f"{a[:7]}..{b[:7]}": git("diff", "--stat", a, b, "--", *PATHS) for a, b in zip(dc, dc[1:])}
R["training_code_diff_first_to_last_commit"] = git("diff", "--stat", dc[0], dc[-1], "--", *PATHS)

runs.drop(columns=["meta", "run_path"]).to_csv(os.path.join(OUT, "04_canonical_run_list.csv"), index=False)
json.dump(R, open(os.path.join(OUT, "04_results.json"), "w"), indent=1, default=str)
print("ok", len(runs))
