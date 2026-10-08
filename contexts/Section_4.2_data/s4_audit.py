"""
Provenance (Section 0), data basis and audit (Section 1), generic structural tables (Section 2) shared by the three
section generators.
"""
from __future__ import annotations

import ast
import platform

from s4_blocks import *  # noqa: F401,F403
from s4_blocks import srt  # noqa: F401


# =============================================================================== Section 0
def provenance(ds: Dataset, scripts: list, code_files: list):
    import torch
    import gymnasium
    import mujoco
    spec = ds.spec
    out = []
    head = sh(["git", "rev-parse", "HEAD"]).strip()
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    porcelain = sh(["git", "status", "--porcelain"]).rstrip()
    outside = [ln for ln in porcelain.splitlines() if "contexts/" not in ln]
    log1 = sh(["git", "log", "-1", "--format=%h %ad %s", "--date=iso"]).strip()
    out.append(table(["Item", "Value"], [
        ["Generated (local time)", dt.datetime.now().isoformat(timespec="seconds")],
        ["Machine / OS", f"{platform.system()} {platform.release()} ({platform.version()}), {platform.machine()} — the Windows dev "
                         "checkout (synced `results/`), not the Linux experiment box"],
        ["Repo path", str(REPO)],
        ["Python / pandas / numpy", f"{platform.python_version()} / {pd.__version__} / {np.__version__}"],
        ["torch / gymnasium / mujoco (local venv; used only to instantiate networks and read env dims)",
         f"{torch.__version__} / {gymnasium.__version__} / {mujoco.__version__}"],
        ["git HEAD", f"{head} ({log1})"], ["branch", branch],
        ["working tree (`git status --porcelain`)",
         "clean" if not porcelain else ("clean apart from `contexts/` (this generator's own output)" if not outside
                                        else "DIRTY outside contexts/: " + "; ".join(outside))],
    ]))
    if porcelain:
        out.append(code(porcelain))
    out.append("**Scripts used** (all under `contexts/Section_4.2_data/` unless stated; all read-only on `results/`, "
               "`flop_analysis/` and git): " + ", ".join(f"`{s}`" for s in scripts) + ". Shared modules: `s4_data.py` (loading, "
               "derived quantities, independent FLOP recomputation), `s4_blocks.py` (common tables, cross-configuration "
               "quantities), `s4_audit.py` (this part). Re-run from the repo root with `.venv/Scripts/python.exe "
               f"contexts/Section_4.{spec.section.split('.')[1]}_data/build_section_{spec.section.replace('.', '_')}.py`. "
               "Definitions are taken from `contexts/section_4.1_data/build_section_4_1.py` (the generator of Section 4.1) and "
               "are listed in the docstring of `s4_data.py`; no definition of 4.1 was missing.")
    files = ["flop_analysis/output/per_run_energy_per_flop.csv", "flop_analysis/output/cross_seed_energy_per_flop.csv",
             "flop_analysis/flops_per_call.json"]
    rows = []
    for fpath in files:
        last = sh(["git", "log", "-1", "--format=%h %ad", "--date=iso", "--", fpath]).strip()
        rows.append([f"`{fpath}`", sha256(fpath), dt.datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(timespec="seconds"), last])
    out.append("\n**Pipeline files used for the cross-check:**\n")
    out.append(table(["File", "SHA-256", "mtime (local)", "last git commit touching it"], rows))
    # freshness
    newest = max((os.path.getmtime(Path(r["run_dir"]) / "segment_energy.json"), r["run_dir"]) for r in ds.runs)
    csv_mt = min(os.path.getmtime(f) for f in files[:2])
    out.append(f"Freshness: newest `segment_energy.json` among the {len(ds.runs)} {spec.name} runs: `{newest[1]}` "
               f"({dt.datetime.fromtimestamp(newest[0]).isoformat(timespec='seconds')}); oldest pipeline CSV mtime "
               f"{dt.datetime.fromtimestamp(csv_mt).isoformat(timespec='seconds')} — "
               + check("newest segment_energy.json older than the pipeline CSVs (filesystem mtime; the content-based "
                       "reconciliation is in Section 1.5)", newest[0] <= csv_mt))
    # code-version consistency
    commits = Counter(r["meta"]["git_commit"] for r in ds.runs)
    order = sorted(commits, key=lambda c: sh(["git", "log", "-1", "--format=%ct", c]).strip() or "0")
    rows = []
    for c in order:
        known = subprocess.run(["git", "cat-file", "-e", c + "^{commit}"], cwd=REPO, capture_output=True).returncode == 0
        subj = sh(["git", "log", "-1", "--format=%h %ad %s", "--date=short", c]).strip() if known else "UNKNOWN TO GIT"
        tg = sorted({label(r["es"], r["tag"]) for r in ds.runs if r["meta"]["git_commit"] == c})
        rows.append([f"`{c[:10]}`", commits[c], subj, ", ".join(tg)])
    out.append(f"\n**Code-version consistency of the {len(ds.runs)} runs** (`metadata.json` → `git_commit`):\n")
    out.append(table(["git_commit", "# runs", "commit", "configurations"], rows))
    rows = []
    any_diff = []
    for fpath in code_files:
        blobs = {}
        for c in order:
            b = sh(["git", "rev-parse", f"{c}:{fpath}"]).strip()
            blobs[c[:7]] = b[:10] if b and "fatal" not in b else "ABSENT"
        same = len(set(blobs.values())) == 1
        if not same:
            any_diff.append(fpath)
        rows.append([f"`{fpath}`", "identical at every run commit" if same else "DIFFERS: " + ", ".join(f"{k}={v}" for k, v in blobs.items())])
    out.append("Algorithm-side files at each run commit (git blob ids compared):\n")
    out.append(table(["file", "status"], rows))
    # config classes
    def classes_at(commit):
        src_ = sh(["git", "show", f"{commit}:configs/config.py"])
        tree = ast.parse(src_)
        return {n.name: ast.get_source_segment(src_, n) for n in tree.body if isinstance(n, ast.ClassDef)}
    cfg_name = spec.config_cls.__name__
    segs = {c[:7]: classes_at(c) for c in order}
    rows = []
    for name in ("ExperimentConfig", cfg_name):
        vals_ = {k: v.get(name) for k, v in segs.items()}
        same = len({v for v in vals_.values()}) == 1
        rows.append([f"`{name}`", "identical at every run commit" if same else "CHANGED between run commits"])
        if not same:
            any_diff.append(name)
    out.append("Config dataclasses at each run commit (AST source segment compared):\n")
    out.append(table(["definition", "status"], rows))
    first, last = order[0], order[-1]
    other = []
    for fpath in ["experiment_runner.py", "run_experiment.py", "utils/gpu_control.py", "utils/thermal_gate.py", "algorithms/tracker_utils.py"]:
        d = sh(["git", "diff", first, last, "--", fpath])
        added = [ln for ln in d.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        removed = [ln for ln in d.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        other.append(f"`{fpath}`: " + (f"+{len(added)}/−{len(removed)} lines between `{first[:7]}` and `{last[:7]}`" if d.strip() else "identical"))
    out.append("Runner/utility files, first versus last run commit: " + "; ".join(other) + ".")
    keysets = Counter()
    by_ks = defaultdict(list)
    for r in ds.runs:
        ks = tuple(sorted(r["meta"].get("thermal_gate", {}).keys()))
        keysets[ks] += 1
        by_ks[ks].append(f"{label(r['es'], r['tag'])}/s{r['seed']} (commit {r['meta']['git_commit'][:7]})")
    out.append("`thermal_gate` key sets in `metadata.json` (a difference between runs that carry the same `git_commit` would "
               "show an uncommitted working tree at run time):\n")
    out.append(table(["thermal_gate keys", "# runs", "runs (if ≤ 5)"],
                     [[", ".join(k), n, "; ".join(by_ks[k]) if n <= 5 else "…"] for k, n in keysets.most_common()]))
    # uncommitted-tree detection: same commit but different key sets
    by_commit = defaultdict(set)
    for r in ds.runs:
        by_commit[r["meta"]["git_commit"]].add(tuple(sorted(r["meta"].get("thermal_gate", {}).keys())))
    mixed = {c[:7]: [list(k) for k in ks] for c, ks in by_commit.items() if len(ks) > 1}
    if mixed:
        out.append(f"Commits whose runs carry different `thermal_gate` key sets (⇒ working tree differed from the commit): {mixed}")
        disc(f"{spec.name}: runs recorded under the same `git_commit` carry different `metadata.thermal_gate` key sets "
             f"({list(mixed)}), i.e. some runs executed with an uncommitted working tree; the visible differences are "
             "metadata-logging only.")
    out.append("\n" + ("All algorithm-side files named above are byte-identical at every run commit." if not any_diff else
                       f"**Differences found in: {any_diff}** — see tables above."))
    if any_diff:
        disc(f"{spec.name}: algorithm-side code/config differs between run commits: {any_diff}")
    return "\n".join(out)


# =============================================================================== Section 1
def inventory_table(ds: Dataset, inv_keys: list):
    spec = ds.spec
    rows = []
    for es, t in ds.cfgs():
        rs = srt(ds.runs_of(es, t))
        ac = rs[0]["meta"]["algo_config"] if rs else {}
        ec = rs[0]["meta"]["experiment_config"] if rs else {}
        seeds = sorted(r["seed"] for r in rs)
        rows.append([t, ENVS[[e[0] for e in ENVS].index(es)][1], spec.tag_desc[t]]
                    + [json.dumps(ac.get(k)) if not isinstance(ac.get(k), (int, float)) else ac.get(k) for k in inv_keys]
                    + [ec.get("warmup_steps"), ec.get("train_steps"), ec.get("steps_per_epoch"), f"`{rs[0]['sig']}`",
                       f"`{spec.tag_override_file(es, t)}`", spec.tag_script(es, t), len(rs), ",".join(map(str, seeds)),
                       len(rs) - len(set(seeds))])
    return table(["config", "env", "description"] + inv_keys + ["warmup", "train_steps", "steps/epoch", "architecture signature",
                 "override file", "launcher", "# runs", "seeds present", "# duplicates"], rows)


def config_diff_table(ds: Dataset):
    spec = ds.spec
    ign = {"seed", "env_id", "algo_name"}
    META_ONLY = "thermal_gate_reference_file"
    rows = []
    bad = []
    for es, t in ds.cfgs():
        rs = srt(ds.runs_of(es, t))
        base = srt(ds.runs_of(es, "base"))[0]["meta"]
        ac0, ec0 = rs[0]["meta"]["algo_config"], rs[0]["meta"]["experiment_config"]
        dac = {k: (base["algo_config"].get(k), ac0.get(k)) for k in set(ac0) | set(base["algo_config"]) if ac0.get(k) != base["algo_config"].get(k)}
        dec = {k: (base["experiment_config"].get(k), ec0.get(k)) for k in (set(ec0) | set(base["experiment_config"])) - ign - {META_ONLY}
               if ec0.get(k) != base["experiment_config"].get(k)}
        within = all(r["meta"]["algo_config"] == ac0 and {k: v for k, v in r["meta"]["experiment_config"].items() if k not in ign | {META_ONLY}}
                     == {k: v for k, v in ec0.items() if k not in ign | {META_ONLY}} for r in rs)
        dd = {k: (ds.defaults.get(k), ac0.get(k)) for k in set(ac0) | set(ds.defaults) if ac0.get(k) != ds.defaults.get(k)}
        exp_keys = set(spec.tag_factor[t])
        if t != "base" and set(dac) != exp_keys:
            bad.append((label(es, t), dac, exp_keys))
        if t == "base" and dac:
            bad.append((label(es, t), dac, "baseline of own env"))
        if dec:
            bad.append((label(es, t), "experiment_config", dec))
        if not within:
            bad.append((label(es, t), "seeds differ"))
        fmt = lambda d_: "—" if not d_ else "; ".join(f"{k}: {a} → {b}" for k, (a, b) in sorted(d_.items()))  # noqa: E731
        rows.append([label(es, t), fmt(dac), fmt(dec), "yes" if within else "NO", fmt(dd)])
    return rows, bad


def audit(ds: Dataset, inv_keys, extra_checks=None):
    spec = ds.spec
    out = []
    out.append("**1.1 Inventory** (one row per configuration; values from the logged `metadata.json`, identical within a "
               "configuration — checked in 1.3). `launcher` is derived from the sweep-log blocks in which the run directory "
               "appears (Section 1.4):\n")
    out.append(inventory_table(ds, inv_keys))
    out.append(f"\n**1.2 Full run list** → `contexts/Section_{spec.section}_data/{spec.algo}_run_inventory.csv` "
               f"({len(ds.runs)} rows: algo, config, env, seed, run directory, timestamp, git commit, versions, clock lock, "
               "start/end time, full logged `algo_config`).\n")
    out.append("**1.3 Audit checks**\n")
    L = []
    exp_total = sum(spec.expected_runs.values())
    per_cfg = {label(es, t): len(ds.runs_of(es, t)) for es, t in ds.cfgs()}
    L.append(check(f"layout-document expectation: {exp_total} runs ({', '.join(f'{k}: {v // 5} configurations' for k, v in spec.expected_runs.items())}"
                   f" × 5 seeds); found {len(ds.runs)}", len(ds.runs) == exp_total,
                   f"unmatched run directories at canonical seeds: {len(ds.unmatched)}"))
    cfg_per_env = {es: len([t for t in spec.tags if ds.has(es, t)]) for es, _ in ENVS}
    L.append(check("configurations per environment equal the layout-document counts", all(
        cfg_per_env[es] * 5 == n for es, n in spec.expected_runs.items()), f"found {cfg_per_env}"))
    bad = {k: v for k, v in per_cfg.items() if v != 5}
    L.append(check("exactly 5 runs per configuration", not bad, f"offenders: {bad}" if bad else f"all {len(per_cfg)} configurations have 5"))
    L.append(check("every run uses one of the 5 canonical seeds {331, 958, 14577, 43611, 85062}; each seed once per configuration",
                   all(sorted(r["seed"] for r in ds.runs_of(es, t)) == sorted(CANON) for es, t in ds.cfgs())))
    dups = [k for k, v in Counter((r["es"], r["tag"], r["seed"]) for r in ds.runs).items() if v > 1]
    L.append(check("no duplicate (algo, env, seed, config) runs", not dups, str(dups) if dups else ""))
    miss = [(r["run_dir"], [f for f, ok in r["files"].items() if not ok]) for r in ds.runs if not all(r["files"].values())]
    L.append(check("every run has emissions.csv, segment_energy.json, training_metrics.json, metadata.json, run.log", not miss, str(miss) if miss else ""))
    L.append(check("every run has exactly one per-task CodeCarbon log `emissions_*.csv`", all(r["n_task_csvs"] == 1 for r in ds.runs)))
    rows, bad = config_diff_table(ds)
    L.append(check("the logged configuration of every configuration differs from its environment's baseline in exactly the "
                   "swept factor (algo_config), has no `experiment_config` difference (ignoring seed/env/algo and the metadata-only "
                   "key `thermal_gate_reference_file`), and is identical across its 5 seeds", not bad, str(bad) if bad else ""))
    ec0 = ds.runs[0]["meta"]["experiment_config"]
    n_hc = ds.runs_of("HC", "base")[0]["meta"]["experiment_config"]
    n_ant = ds.runs_of("Ant", "base")[0]["meta"]["experiment_config"]
    dd = {k: (n_hc.get(k), n_ant.get(k)) for k in set(n_hc) | set(n_ant) if n_hc.get(k) != n_ant.get(k) and k not in ("seed", "env_id", "algo_name")}
    L.append(check("`experiment_config` of the HC and Ant baselines are identical apart from seed/env_id/algo_name"
                   " and the metadata-only key", set(dd) <= {"thermal_gate_reference_file"}, str(dd)))
    ws = Counter((r["meta"]["experiment_config"]["train_steps"], r["meta"]["experiment_config"]["steps_per_epoch"],
                  r["meta"]["experiment_config"]["warmup_steps"], r["es"]) for r in ds.runs)
    L.append(f"  - (train_steps, steps_per_epoch, warmup_steps, env) → #runs: {dict(ws)}")
    bad = []
    for r in ds.runs:
        names = set(r["tasks"]["task_name"])
        exp = set()
        for k in spec.per_epoch_tasks:
            exp |= {f"{k}_{i}" for i in range(100)}
        single = {"idle_baseline_head", "idle_baseline_tail", "warmup_0"} | {f"{m}_0" for m in spec.measured_order if m not in spec.per_epoch_tasks}
        other = names - exp - single
        n_exp = 100 * len(spec.per_epoch_tasks) + len(single)
        if not exp <= names or other or len(r["tasks"]) != n_exp:
            bad.append((r["run_dir"], len(r["tasks"]), sorted(other)[:3]))
    L.append(check("per-task CSV has exactly the expected tasks: " + ", ".join(f"{k}_0..99" for k in spec.per_epoch_tasks)
                   + ", warmup_0, idle_baseline_head, idle_baseline_tail" + "".join(f", {m}_0" for m in spec.measured_order if m not in spec.per_epoch_tasks), not bad, str(bad) if bad else ""))
    bad = []
    for r in ds.runs:
        ep = r["tm"]["epochs"]
        if len(ep) != 100:
            bad.append((r["run_dir"], "epochs", len(ep)))
    L.append(check("`training_metrics.json` has 100 epoch rows in every run", not bad, str(bad) if bad else ""))
    bad = [(r["run_dir"], r["meta"].get("cuda_device_name")) for r in ds.runs if r["meta"].get("cuda_device_name") != "NVIDIA GeForce RTX 5090"]
    L.append(check("GPU = NVIDIA GeForce RTX 5090 in every run (`cuda_device_name`)", not bad, str(bad) if bad else ""))
    gm = Counter(r["tasks"]["gpu_model"].iloc[0] for r in ds.runs)
    cm = Counter(r["tasks"]["cpu_model"].iloc[0] for r in ds.runs)
    L.append(f"  - per-task CSV `gpu_model`: {dict(gm)}; `cpu_model`: {dict(cm)}")
    bad = [(r["run_dir"],) for r in ds.runs if (r["meta"]["experiment_config"].get("gpu_min_clock_mhz"), r["meta"]["experiment_config"].get("gpu_max_clock_mhz")) != (2000, 2000)
           or not r["meta"]["experiment_config"].get("lock_gpu_clocks")]
    L.append(check("GPU clock lock requested at 2000/2000 MHz with lock_gpu_clocks = true in every run", not bad, str(bad) if bad else ""))
    L.append(f"  - persistence mode requested: {dict(Counter(r['meta']['experiment_config']['set_persistence_mode'] for r in ds.runs))}; "
             f"CPU `performance` governor requested: {dict(Counter(r['meta']['experiment_config']['set_cpu_performance_governor'] for r in ds.runs))}")
    vers = Counter((r["meta"]["torch_version"], r["meta"].get("codecarbon_version")) for r in ds.runs)
    L.append(check("torch 2.13.0+cu130 and codecarbon 3.3.0 in every run", set(vers) == {("2.13.0+cu130", "3.3.0")}, str(dict(vers))))
    tg = [(r["run_dir"], r["meta"].get("thermal_gate", {}).get("reason")) for r in ds.runs
          if not r["meta"].get("thermal_gate", {}).get("gated") or r["meta"]["thermal_gate"].get("reason") != "reached_reference"]
    L.append(check("thermal gate: gated = true and reason = reached_reference (no timeout) in every run", not tg, str(tg) if tg else ""))
    wmax = max(r["meta"]["thermal_gate"]["waited_seconds"] for r in ds.runs)
    L.append(f"  - thermal gate: waited_seconds max {g6(wmax)} s; gate temperature {g6(min(r['meta']['thermal_gate']['final_temp_c'] for r in ds.runs))}"
             f"–{g6(max(r['meta']['thermal_gate']['final_temp_c'] for r in ds.runs))} °C; gate power "
             f"{g6(min(r['meta']['thermal_gate']['final_power_w'] for r in ds.runs))}–{g6(max(r['meta']['thermal_gate']['final_power_w'] for r in ds.runs))} W "
             "(the gate passes on its first poll because its reference is captured fresh in each process, `utils/thermal_gate.py`)")
    hits_log, hits_sweep, covered = defaultdict(list), defaultdict(list), 0
    for r in ds.runs:
        for s in GREP_STRINGS:
            if s.lower() in r["log"].lower():
                hits_log[s].append(r["run_dir"])
        blks = SWEEP_BLOCKS.get(r["run_dir"])
        if blks:
            covered += 1
            for _, blk in blks:
                for s in GREP_STRINGS:
                    if s.lower() in blk.lower():
                        hits_sweep[s].append(r["run_dir"])
    L.append(check(f"`run.log` of every run contains none of the {len(GREP_STRINGS)} failure-mode strings (list in Section 4.1, "
                   "schema sample S6)", not hits_log, str(dict(hits_log)) if hits_log else ""))
    L.append(check(f"sweep-log block (stdout+stderr of the launch, `2>&1 | tee -a`) contains none of the strings — available for "
                   f"{covered} of {len(ds.runs)} runs", not hits_sweep, str(dict(hits_sweep)) if hits_sweep else ""))
    cpu_p = [r["tasks"].loc[r["tasks"]["task_name"].str.startswith("gradient_updates_"), "cpu_power"].nunique() for r in ds.runs]
    L.append(check("data-side RAPL check: per-task `cpu_power` varies across the 100 gradient_updates tasks in every run (a TDP "
                   "fallback would be constant)", all(n > 2 for n in cpu_p), f"min distinct values per run = {min(cpu_p)}"))
    geo = Counter((r["tasks"]["country_name"].iloc[0], r["tasks"]["region"].iloc[0]) for r in ds.runs)
    L.append(check("data-side geolocation check: no run shows the Canada/Quebec fallback", not any(c == "Canada" for c, _ in geo), str(dict(geo))))
    for line in (extra_checks or []):
        L.append(line)
    out.append("\n".join("- " + ln if not ln.startswith("  -") else ln for ln in L))
    out.append("\n**1.3b Logged-configuration differences per configuration** (each row: difference of the logged `algo_config` / "
               "`experiment_config` to the baseline of the same environment; last column: difference to the dataclass defaults "
               f"of `{spec.config_cls.__name__}`, i.e. the override file content):\n")
    out.append(table(["config", "algo_config vs same-env baseline", "experiment_config vs same-env baseline",
                      "identical across the 5 seeds", f"algo_config vs `{spec.config_cls.__name__}` defaults"], rows))
    out.append("The warning/grep caveat of Section 4.1 applies: `run.log` is written only by the `experiment_runner` logger; the "
               "`gpu_control`, `thermal_gate` and CodeCarbon loggers write to stderr, which the sweep scripts capture into the "
               "sweep log (`2>&1 | tee -a`). The clock lock / governor are therefore verified as *requested* (metadata) and as "
               "*not reported failed* (sweep-log blocks), never as positively applied; applied lock values are not recorded in "
               "`metadata.json`.")
    return "\n".join(out)


# =============================================================================== Section 1.4 execution facts
def exec_facts(ds: Dataset, status_files: list, sweep_scripts: list):
    spec = ds.spec
    out = []
    allm = []
    for d, md in ALL_META:
        s = dt.datetime.fromisoformat(md["start_time_utc"])
        e = dt.datetime.fromisoformat(md["end_time_utc"])
        allm.append((s, e, md["algo_name"], md["env_id"], md["seed"], d.as_posix()))
    allm.sort()
    sessions, cur = [], [allm[0]]
    for prev, nxt in zip(allm, allm[1:]):
        if (nxt[0] - prev[1]).total_seconds() > 1800:
            sessions.append(cur)
            cur = []
        cur.append(nxt)
    sessions.append(cur)
    sess_of = {x[5]: i for i, ses in enumerate(sessions) for x in ses}
    rs_sorted = sorted(ds.runs, key=lambda r: r["meta"]["start_time_utc"])
    rows = []
    for es, t in ds.cfgs():
        rs = srt(ds.runs_of(es, t))
        st = [dt.datetime.fromisoformat(r["meta"]["start_time_utc"]) for r in rs]
        en = [dt.datetime.fromisoformat(r["meta"]["end_time_utc"]) for r in rs]
        ses = sorted({sess_of[r["run_dir"]] for r in rs})
        logs = sorted({lf for r in rs for lf, _ in SWEEP_BLOCKS.get(r["run_dir"], [])})
        order = sorted(rs, key=lambda r: r["meta"]["start_time_utc"])
        seed_order = [r["seed"] for r in order]
        idx = [i for i, r in enumerate(rs_sorted) if r["es"] == es and r["tag"] == t]
        contig = idx == list(range(idx[0], idx[0] + len(idx)))
        rows.append([label(es, t), min(st).isoformat(timespec="seconds"), max(en).isoformat(timespec="seconds"),
                     ", ".join(map(str, ses)), "yes" if len(ses) == 1 else "NO",
                     "yes" if contig else "no", ",".join(map(str, seed_order)),
                     ", ".join(f"`{l}`" for l in logs) or "none",
                     sum(1 for r in rs if r["run_dir"] in SWEEP_BLOCKS)])
    out.append("**1.4 Execution facts.** Times are `metadata.json` `start_time_utc`/`end_time_utc` (UTC). A *session* = maximal "
               "block of runs of **all 300 recorded runs** (all algorithms) with < 30 min between one run's end and the next run's "
               "start (same definition as Section 4.1). `contiguous` = the five runs of the configuration are consecutive among "
               f"the {spec.name} runs. `launch logs` = sweep logs containing the run directory (each run's stdout+stderr block).\n")
    out.append(table(["config", "first start (UTC)", "last end (UTC)", "session(s)", "all 5 seeds in one session", "contiguous",
                      "seed order (by start time)", "launch log(s)", "# runs with a log block"], rows))
    srows = []
    for i in sorted({sess_of[r["run_dir"]] for r in ds.runs}):
        in_s = [r for r in rs_sorted if sess_of[r["run_dir"]] == i]
        ses = sessions[i]
        srows.append([i, ses[0][0].isoformat(timespec="minutes"), ses[-1][1].isoformat(timespec="minutes"), len(ses),
                      dict(Counter(x[2] for x in ses)), len(in_s), " → ".join(dict.fromkeys(label(r["es"], r["tag"]) for r in in_s))])
    out.append("\nSessions containing " + spec.name + " runs:\n")
    out.append(table(["session", "first start", "last end", "# runs (all algos)", "algos", f"# {spec.name}", f"{spec.name} configs in order"], srows))
    my_sessions = {sess_of[r["run_dir"]] for r in ds.runs}
    gaps = []
    for i in my_sessions:
        ses = sessions[i]
        gaps += [(b[0] - a[1]).total_seconds() for a, b in zip(ses, ses[1:])]
    run_len = [(dt.datetime.fromisoformat(r["meta"]["end_time_utc"]) - dt.datetime.fromisoformat(r["meta"]["start_time_utc"])).total_seconds() / 60 for r in ds.runs]
    out.append(f"Inside the sessions listed above, consecutive runs of **any** algorithm are separated by {g6(min(gaps))}–{g6(max(gaps))} s "
               f"(median {g6(np.median(gaps))} s) between one run's `end_time_utc` and the next run's `start_time_utc`, by definition < 1800 s. "
               "Run length = `end_time_utc − start_time_utc` includes the thermal-gate reference poll, the 30 s settle, both idle windows and warmup; "
               f"over the {spec.name} runs: median {g6(np.median(run_len))} min (min {g6(min(run_len))}, max {g6(max(run_len))}). Whether the machine was powered off or idle "
               "between sessions is not recorded (each sweep script ends with `shutdown -h +1`).")
    # baseline launch + override naming in scripts
    base_logs = Counter(lf for r in ds.runs if r["tag"] == "base" for lf, _ in SWEEP_BLOCKS.get(r["run_dir"], []))
    n_base = sum(1 for r in ds.runs if r["tag"] == "base")
    out.append(f"\n**Baseline runs** (configuration `base`, {n_base} runs): " + (
        "launched by a sweep script — their stdout+stderr blocks are in " + ", ".join(f"`{k}` ({v} runs)" for k, v in base_logs.items())
        + f" (sum {sum(base_logs.values())} of {n_base}); not started by hand, and their launch error output was saved."
        if sum(base_logs.values()) == n_base else
        f"only {sum(base_logs.values())} of {n_base} have a sweep-log block; the rest were launched by hand and their launch error output was not saved."))
    pats = []
    for sc in sorted(glob.glob("scripts/*.sh")):
        for i, ln in enumerate(Path(sc).read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r'(echo|overrides=)\s*"?configs/overrides/', ln) or re.search(r'echo "configs/overrides/', ln):
                pats.append(f"`{Path(sc).name}:{i}` `{ln.strip()}`")
    out.append("Override-file names as built by the sweep scripts (pattern lines found with `echo \"configs/overrides/…\"` / `overrides=\"configs/overrides/…\"`; the algorithm / env / value are substituted at run time; "
               "the launch blocks of the logs do not record the override path): " + "; ".join(pats) + ".")
    # launch scripts
    out.append("\n**Launchers.** Sweep scripts that produced these runs (read from `scripts/`): " + "; ".join(sweep_scripts))
    stderr_ok = []
    for sc in sorted({Path(p).name for p in glob.glob("scripts/*.sh")}):
        txt = Path("scripts", sc).read_text(encoding="utf-8")
        stderr_ok.append(f"`{sc}`: " + ("`2>&1 | tee -a $LOG_FILE`" if "2>&1 | tee" in txt else "no stderr capture found"))
    out.append("Stderr capture in each script: " + "; ".join(stderr_ok) + ".")
    # status files
    rows = []
    run_dirs = {r["run_dir"] for r in ds.runs}
    for sf in status_files:
        d = json.loads(Path(sf).read_text(encoding="utf-8"))
        ents = [x for x in d["runs"] if x.get("algo", spec.algo) == spec.algo]
        rows.append([f"`{sf}`", len(d["runs"]), len(ents), dict(Counter(x["status"] for x in d["runs"])),
                     "n/a (no entries of this algorithm)" if not ents else
                     "yes" if all((x.get("run_dir") or "").rstrip("/") in run_dirs for x in ents if x["status"] == "done") else
                     "NO: " + str([x.get("run_dir") for x in ents if (x.get("run_dir") or "").rstrip("/") not in run_dirs][:5])])
    out.append("\nSweep bookkeeping files (every `done` entry of this algorithm should point at one of the analysed run directories):\n")
    out.append(table(["status file", "# entries", f"# {spec.name} entries", "statuses", "every done entry's run_dir is one of the analysed runs"], rows))
    # failures / restarts in the logs
    outc = block_outcomes(spec.algo)
    cnt = Counter((o["log"], o["outcome"]) for o in outc)
    rows = [[f"`{lf}`", oc, n] for (lf, oc), n in sorted(cnt.items())]
    out.append(f"\n**Launch outcomes of {spec.name} in every sweep log** (each `[i/N] Starting: algo={spec.algo} …` block classified by its closing line: "
               "`Finished OK`, `FAILED (exit …)`, or no closing line):\n")
    out.append(table(["log", "outcome", "# launches"], rows))
    bad = [o for o in outc if o["outcome"] != "Finished OK"]
    if bad:
        out.append(f"\nNon-successful {spec.name} launches (first lines of the block):\n")
        out.append(table(["log", "time", "index", "Starting line", "outcome", "first error line"],
                         [[f"`{o['log']}`", o["time"], o["index"], o["start_line"][:120], o["outcome"], o["error_line"][:150]] for o in bad]))
        out.append(f"{len(bad)} launch(es) of {spec.name} did not finish OK; run directories created by them: "
                   f"{sorted({o['run_dir'] for o in bad if o['run_dir']}) or 'none'}.")
    else:
        out.append(f"\nAll {len(outc)} recorded {spec.name} launches finished OK; no failure, restart or duplicate launch of {spec.name} appears in any sweep log.")
    n_dir = len({o["run_dir"] for o in outc if o["run_dir"] and o["outcome"] == "Finished OK"})
    out.append(f"Successful launches with a run directory: {n_dir} (analysed canonical runs: {len(ds.runs)}).")
    return "\n".join(out), sessions


def recon(ds: Dataset):
    """Compare every recomputed (run, segment) and (configuration, segment) value with the two pipeline CSVs."""
    spec = ds.spec
    pr = pd.read_csv("flop_analysis/output/per_run_energy_per_flop.csv")
    cs = pd.read_csv("flop_analysis/output/cross_seed_energy_per_flop.csv")
    maxdev = defaultdict(float)
    ncmp = defaultdict(int)
    fails = []

    def rel(a, b):
        return abs(a - b) / abs(b) if b else abs(a - b)

    segs = spec.train_segments + ["TOTAL_MEASURED_TRAINING", "warmup", "idle_baseline_head", "idle_baseline_tail"]
    for r in ds.runs:
        sub = pr[pr.run_dir == r["run_dir"]].set_index("segment")
        for s in segs:
            row = sub.loc[s]
            for col, mine in (("total_energy_joules", r["E"][s]), ("duration_s", r["D"][s]), ("mean_power_w", r["P"][s])):
                d = rel(float(row[col]), mine)
                maxdev[("per_run", col)] = max(maxdev[("per_run", col)], d)
                ncmp[("per_run", col)] += 1
                if d > 1e-9:
                    fails.append(("per_run", r["run_dir"], s, col, d))
            if s in r["F"] and s != "buffer_sample":
                for col, mine in (("total_flops", r["F"][s]), ("energy_per_flop_j_per_flop", r["E"][s] / r["F"][s] if r["F"][s] else float("nan"))):
                    d = rel(float(row[col]), mine)
                    maxdev[("per_run", col)] = max(maxdev[("per_run", col)], d)
                    ncmp[("per_run", col)] += 1
                    if d > 1e-9:
                        fails.append(("per_run", r["run_dir"], s, col, d))
            if s in r["calls"] and not pd.isna(row["call_count"]):
                d = rel(float(row["call_count"]), r["calls"][s])
                maxdev[("per_run", "call_count")] = max(maxdev[("per_run", "call_count")], d)
                ncmp[("per_run", "call_count")] += 1
                if d > 1e-9:
                    fails.append(("per_run", r["run_dir"], s, "call_count", d))
        if spec.algo == "mbpo":
            row = sub.loc["dynamics_model_update"]
            for col, mine in (("dynamics_train_flops", r["mbpo"]["dyn_train"]), ("dynamics_holdout_flops", r["mbpo"]["dyn_hold"])):
                d = rel(float(row[col]), mine)
                maxdev[("per_run", col)] = max(maxdev[("per_run", col)], d)
                ncmp[("per_run", col)] += 1
                if d > 1e-9:
                    fails.append(("per_run", r["run_dir"], "dynamics_model_update", col, d))
    cs["mbpo_rollout_regime"] = cs["mbpo_rollout_regime"].fillna("")
    for es, t in ds.cfgs():
        rs = srt(ds.runs_of(es, t))
        r0 = rs[0]
        regime = r0["regime"] if spec.algo == "mbpo" else ""
        sub = cs[(cs.algo == spec.algo) & (cs.env_id == r0["env"]) & (cs.architecture_signature == r0["sig"])
                 & (cs.updates_per_env_step == r0["meta"]["algo_config"]["updates_per_env_step"])
                 & (cs.mbpo_rollout_regime == regime)].set_index("segment")
        for s in spec.train_segments + ["TOTAL_MEASURED_TRAINING"]:
            row = sub.loc[s]
            Em = np.mean(vals(rs, "E", s))
            comps = [("mean_energy_joules", Em), ("mean_duration_s", np.mean([r["D"][s] for r in rs])),
                     ("mean_power_w", np.mean([r["P"][s] for r in rs]))]
            if s in r0["F"] and s != "buffer_sample":
                Fm = np.mean([r["F"][s] for r in rs])
                comps += [("total_flops", Fm), ("mean_energy_per_flop_j_per_flop", Em / Fm)]
            for col, mine in comps:
                d = rel(float(row[col]), mine)
                maxdev[("cross_seed", col)] = max(maxdev[("cross_seed", col)], d)
                ncmp[("cross_seed", col)] += 1
                if d > 1e-9:
                    fails.append(("cross_seed", label(es, t), s, col, d))
    rows = [[k[0], f"`{k[1]}`", ncmp[k], f"{v:.3e}"] for k, v in sorted(maxdev.items())]
    n_all = sum(ncmp.values())
    n_fail = len(fails)
    txt = ("**1.5 Independent recomputation versus the pipeline CSVs.** Every (run, segment) and (configuration, segment) value "
           "recomputed here from `segment_energy.json`, the per-task CodeCarbon CSV, `training_metrics.json` and "
           "`flops_per_call.json` (FLOP formulas re-implemented in `s4_data.flops_for_run` / `mbpo_extras`, not imported) is compared with "
           "`per_run_energy_per_flop.csv` and `cross_seed_energy_per_flop.csv`. Columns compared: energy, duration, mean power, FLOPs, "
           "J/FLOP, call count" + (", MBPO dynamics train/holdout FLOPs" if spec.algo == "mbpo" else "") + ". The pipeline's cross-seed "
           "`mean_power_w` is the mean of per-seed powers and is compared with the same statistic.\n")
    txt += table(["CSV", "column", "# values compared", "max relative deviation"], rows)
    txt += "\n- " + check(f"all {n_all} compared values agree within 1e-9 relative", not fails,
                         f"{n_fail} offending cells: {fails[:5]}" if fails else f"maximum deviation over all columns {max(maxdev.values()):.3e}")
    return txt, dict(n_compared=n_all, n_fail=n_fail, max_dev=max(maxdev.values()))


# =============================================================================== Section 2 generic tables
def flop_constant_table(ds: Dataset, keys: list):
    """keys: list of (header, fk -> value). Rows: configurations; cell HC ‖ Ant."""
    rows = []
    for t in ds.spec.tags:
        cells = [t]
        for _, fn in keys:
            parts = []
            for es, env in ENVS:
                rs = ds.runs_of(es, t)
                parts.append(g6(fn(rs[0]["fk"])) if rs else "n/a")
            cells.append(SEP.join(parts))
        rows.append(cells)
    return table(["config"] + [h for h, _ in keys], rows)


def call_count_table(ds: Dataset):
    keys = [s for s in (ds.spec.measured_order if False else []) ]
    seglist = [s for s in ds.runs[0]["calls"]]
    rows = []
    for t in ds.spec.tags:
        cells = [t]
        for s in seglist:
            parts = []
            for es, _ in ENVS:
                rs = ds.runs_of(es, t)
                if not rs:
                    parts.append("n/a")
                else:
                    v = [r["calls"][s] for r in rs]
                    parts.append((str(int(v[0])) if float(v[0]).is_integer() else g6(v[0])) if len(set(v)) == 1 else f"{g6(np.mean(v))} (min {g6(min(v))}, max {g6(max(v))})")
            cells.append(SEP.join(parts))
        rows.append(cells)
    return table(["config"] + [f"calls {SEG_SHORT[s]}" for s in seglist], rows)


def total_flops_table(ds: Dataset):
    spec = ds.spec
    segs = [s for s in spec.seg_order if s != "buffer_sample"]
    rows = []
    for t in spec.tags:
        cells = [t]
        for s in segs:
            parts = []
            for es, _ in ENVS:
                rs = ds.runs_of(es, t)
                if not rs:
                    parts.append("n/a")
                else:
                    v = [r["F"][s] for r in rs]
                    parts.append(g6(np.mean(v)) if len(set(v)) == 1 else f"{g6(np.mean(v))} (min {g6(min(v))}, max {g6(max(v))})")
            cells.append(SEP.join(parts))
        rows.append(cells)
    hdr = [f"F {SEG_SHORT[s]}" + (" [elementwise ops]" if s == "target_update" else "") for s in segs]
    t1 = table(["config"] + hdr, rows)
    rows = []
    segs2 = [s for s in segs if s != "target_update"] + ["target_update"]
    for t in spec.tags:
        if not (ds.has("HC", t) and ds.has("Ant", t)):
            continue
        cells = [t]
        for s in segs:
            a = np.mean([r["F"][s] for r in ds.runs_of("Ant", t)])
            h = np.mean([r["F"][s] for r in ds.runs_of("HC", t)])
            cells.append(g6(a / h))
        rows.append(cells)
    t2 = table(["config"] + [f"Ant/HC {SEG_SHORT[s]}" for s in segs], rows)
    return t1, t2


# =============================================================================== source-code helpers
class Src:
    """Line-numbered access to a source file at HEAD."""

    def __init__(self, path):
        self.path = path
        self.lines = Path(path).read_text(encoding="utf-8").splitlines()

    def find(self, pat, start=1):
        for i in range(start - 1, len(self.lines)):
            if re.search(pat, self.lines[i]):
                return i + 1
        raise ValueError(f"{pat!r} not found in {self.path}")

    def excerpt(self, a, b):
        return "\n".join(f"{i:4d}  {self.lines[i - 1]}" for i in range(a, b + 1))

    def block(self, start_pat, end_pat=None, start_from=1, include_end=False):
        a = self.find(start_pat, start_from)
        if end_pat is None:
            b = len(self.lines)
        else:
            b = self.find(end_pat, a + 1) - (0 if include_end else 1)
        return a, b, code(self.excerpt(a, b), "python")


def versus_rows(dsA, tagA, dsB, tagB, getter, segs, fmt=msd):
    """rows = segments; columns = HC A, HC B, Ant A, Ant B."""
    rows = []
    for s in segs:
        cells = [SEG_SHORT.get(s, s)]
        for es, _ in ENVS:
            for dsx, tg in ((dsA, tagA), (dsB, tagB)):
                rs = srt(dsx.runs_of(es, tg))
                try:
                    v = getter(rs, s)
                    cells.append(fmt(v) if not isinstance(v, str) else v)
                except KeyError:
                    cells.append("n/a")
        rows.append(cells)
    return rows


def block_outcomes(algo):
    """Classify every '[i/N] Starting: algo=<algo>' block of every sweep log."""
    rows = []
    for lf in SWEEP_LOGS:
        txt = Path(lf).read_text(encoding="utf-8", errors="replace").splitlines()
        starts = [i for i, ln in enumerate(txt) if re.search(r"\] \[\d+/\d+\] Starting:", ln)]
        for j, st in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(txt)
            blk = txt[st:e]
            if not re.search(rf"algo={algo}\b", blk[0]):
                continue
            joined = "\n".join(blk)
            m = re.search(r"Run directory: (\S+)", joined)
            if "Finished OK" in joined:
                oc = "Finished OK"
            elif re.search(r"FAILED \(exit", joined):
                oc = re.search(r"FAILED \(exit \d+\)", joined).group(0)
            else:
                oc = "no closing line"
            err = next((ln for ln in blk if re.search(r"Error|Traceback|FAILED", ln)), "")
            idx = re.search(r"\[(\d+/\d+)\] Starting", blk[0]).group(1)
            rows.append(dict(log=Path(lf).as_posix(), time=blk[0][:21], index=idx, start_line=blk[0][22:], outcome=oc,
                             run_dir=m.group(1) if m else None, error_line=err))
    return rows


def common_section7_items(ds: Dataset, IDLE):
    """Items shared by the three files for Section 7 (appended to the global lists)."""
    spec = ds.spec
    n_blk = sum(1 for r in ds.runs if r["run_dir"] in SWEEP_BLOCKS)
    disc("CLAUDE.md / README.md state that zero GPU-clock-lock / CPU-governor permission failures and zero RAPL / geolocation fallbacks were found "
         "'by grepping every run.log'. That grep cannot detect those failures (the `gpu_control`, `thermal_gate` and CodeCarbon loggers do not "
         f"write to `run.log`). For {spec.name} the claim is supported by other evidence: {n_blk} of {len(ds.runs)} runs have a sweep-log block "
         "with captured stderr (`2>&1 | tee -a`) containing none of the failure strings, plus data-side checks (varying RAPL `cpu_power`, "
         "non-fallback geolocation, `thermal_gate.reason = reached_reference`) for all runs (Section 1.3). The clock lock is verified as requested, "
         "not as applied.")
    top_dev = max(r["E_run_top_dev"] for r in ds.runs)
    disc(f"The one-row `emissions.csv` written by `tracker.stop()` has `duration` = {g6(np.median([r['top_duration'] for r in ds.runs]))} s (median over the "
         f"{spec.name} runs; it equals the duration of the last task, the 90 s idle tail) while its `energy_consumed` is the cumulative whole-run energy "
         f"(equal to the sum of the per-task rows within {top_dev:.1e} relative). `emissions.csv` therefore cannot be used for a whole-run mean power; "
         "the 'P whole run' columns of this file use Σ energy / Σ duration of all rows of the per-task CSV (idle head, warmup, [pretrain], training tasks, idle tail; "
         "the unmeasured 30 s settle is not included).")
    ratio_head = IDLE.head_integration_window_s.median() - IDLE.head_duration_s.median()
    question(f"Idle-baseline drift: the head idle window integrates {g6(ratio_head)} s more than its recorded 90 s (RAM energy / RAM power), the tail window does not; "
             "this file reports as-recorded idle powers (definition of Section 4.1) and the window-corrected drift in Section 5.2e. Which of the two the thesis text should quote is not decided in the repository.")
    na("GPU theoretical FP32 peak (SM count / lanes per SM are not stored in the repo and no CUDA is available on the Windows checkout) — achieved throughput is reported in GFLOP/s without a peak-utilisation fraction.")
    na("Whether FP32 matmuls ran as TF32 on the RTX 5090 under torch 2.13.0+cu130: the code sets no TF32 / matmul-precision flag and no run logs the effective setting (see Section 4.1 Part 2.3).")
    na("Positive confirmation that the GPU clock lock (2000/2000 MHz), persistence mode and the CPU `performance` governor were *applied*: `metadata.json` records only the request; failure messages would appear in the sweep-log stderr and none do.")
    na("gymnasium / mujoco versions on the Linux experiment box: not recorded in `metadata.json`; environment dimensions here were read from the local venv and agree with `flops_per_call.json`.")
    na("Simulator (MuJoCo) share of the `rollout` task duration: no per-step simulator timing is logged, so the split of `rollout` into policy/model compute and `env.step()` is not measurable from the data.")
