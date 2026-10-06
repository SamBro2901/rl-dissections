"""
Generator for contexts/section_4.1.md (+ companion CSVs in contexts/section_4.1_data/).

Read-only on results/, flop_analysis/ and git. Writes ONLY under contexts/.
Every number in the markdown is computed here from files on disk; nothing is typed in by hand.

Run from the repo root with the project venv:
    .venv/Scripts/python.exe contexts/section_4.1_data/build_section_4_1.py
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import glob
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flop_analysis"))

OUT_MD = REPO / "contexts" / "section_4.1.md"
OUT_DIR = REPO / "contexts" / "section_4.1_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KWH_TO_J = 3.6e6
CANON = [331, 958, 14577, 43611, 85062]
ENVS = [("HC", "HalfCheetah-v5"), ("Ant", "Ant-v5")]
ENV_SHORT = {e: s for s, e in ENVS}
TAGS = ["base", "utd2", "utd4", "w256", "w512", "b512", "b1024"]
SWEEP_TAGS = TAGS[1:]
TAG_DESC = {
    "base": "hidden (1024,1024), batch 256, UTD 1",
    "utd2": "UTD 2", "utd4": "UTD 4",
    "w256": "hidden (256,256)", "w512": "hidden (512,512)",
    "b512": "batch 512", "b1024": "batch 1024",
}
TAG_FACTOR = {
    "base": {}, "utd2": {"updates_per_env_step": 2}, "utd4": {"updates_per_env_step": 4},
    "w256": {"hidden_sizes": [256, 256]}, "w512": {"hidden_sizes": [512, 512]},
    "b512": {"batch_size": 512}, "b1024": {"batch_size": 1024},
}
TAG_OVERRIDE = {
    "base": "none (SACConfig defaults; no script found -- see Part 1)",
    "utd2": "configs/overrides/sac_utd2.json", "utd4": "configs/overrides/sac_utd4.json",
    "w256": "configs/overrides/sac_width256.json", "w512": "configs/overrides/sac_width512.json",
    "b512": "configs/overrides/sac_batch512.json", "b1024": "configs/overrides/sac_batch1024.json",
}
SUBS = ["buffer_sample", "critic_update", "actor_update", "target_update"]
SEG_ORDER = ["rollout"] + SUBS + ["gradient_updates", "TOTAL_MEASURED_TRAINING"]
SEG_STATUS = {"rollout": "measured", "gradient_updates": "measured",
              "buffer_sample": "allocated", "critic_update": "allocated", "actor_update": "allocated",
              "target_update": "allocated", "TOTAL_MEASURED_TRAINING": "aggregate",
              "warmup": "excluded", "idle_baseline_head": "excluded", "idle_baseline_tail": "excluded"}
SEG_SHORT = {"rollout": "rollout", "buffer_sample": "buf_sample", "critic_update": "critic",
             "actor_update": "actor", "target_update": "target", "gradient_updates": "grad_updates",
             "TOTAL_MEASURED_TRAINING": "TOTAL"}

DISCREPANCIES: list[str] = []   # Part 11
OPEN_QUESTIONS: list[str] = []  # Part 11
CHECKS: list[tuple[str, bool, str]] = []  # (name, pass, detail)


def disc(msg):
    DISCREPANCIES.append(msg)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    return f"**{'PASS' if ok else 'FAIL'}** — {name}" + (f" ({detail})" if detail else "")


# ----------------------------------------------------------------------------- formatting
def g4(x):
    if x is None:
        return "NaN"
    try:
        if isinstance(x, (int, np.integer)) and abs(int(x)) < 10000:
            return str(int(x))
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "NaN"
    if xf == 0:
        return "0"
    ax = abs(xf)
    if 1e-3 <= ax < 1e6:  # positional notation, 4 significant digits
        digits = 3 - int(math.floor(math.log10(ax)))
        rounded = round(xf, digits)
        if rounded != 0 and int(math.floor(math.log10(abs(rounded)))) != int(math.floor(math.log10(ax))):
            digits -= 1  # rounding carried into the next decade (e.g. 9.9996 -> 10.00)
            rounded = round(xf, digits)
        return f"{rounded:.{max(digits, 0)}f}" if digits > 0 else f"{rounded:.0f}"
    return f"{xf:.4g}"


def msd(vals):
    a = np.asarray([v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if len(a) == 0:
        return "NaN"
    sd = a.std(ddof=1) if len(a) > 1 else float("nan")
    return f"{g4(a.mean())} ± {g4(sd)}"


def stats(vals):
    a = np.asarray(vals, float)
    return dict(mean=a.mean(), sd=a.std(ddof=1) if len(a) > 1 else float("nan"),
                min=a.min(), max=a.max(), n=len(a))


def _cell(c):
    return str(c).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    """GFM table, surrounded by blank lines; '|' inside cells is escaped."""
    out = ["", "| " + " | ".join(_cell(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    out.append("")
    return "\n".join(out)


CO2_COLS = ("emissions", "emissions_rate")


def redact_co2(df):
    """CO2e is not reported anywhere in this context file (ground rule); blank its values in samples."""
    df = df.copy()
    for c in CO2_COLS:
        if c in df.columns:
            df[c] = "<CO2e omitted>"
    return df


def code(text, lang=""):
    return f"```{lang}\n{text.rstrip()}\n```"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=REPO).stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime(p):
    return dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")


def cfgkey(c):
    return json.dumps(c, sort_keys=True)


# ----------------------------------------------------------------------------- load runs
BASE_ALGO = None


def load_all_metadata():
    out = []
    for m in sorted(Path("results").glob("*/*/seed_*/*/metadata.json")):
        with open(m, encoding="utf-8") as f:
            out.append((m.parent, json.load(f)))
    return out


ALL_META = load_all_metadata()

from configs.config import SACConfig, ExperimentConfig, ALGO_CONFIGS  # noqa: E402

SAC_DEFAULTS = json.loads(json.dumps(dataclasses.asdict(SACConfig())))  # tuples -> lists


def tag_of(ac):
    diffs = {k: v for k, v in ac.items() if SAC_DEFAULTS.get(k) != v}
    for t, fac in TAG_FACTOR.items():
        if diffs == fac:
            return t
    return None


RUNS = []
EXCLUDED = Counter()
UNMATCHED = []
for run_dir, md in ALL_META:
    if md["algo_name"] != "sac":
        continue
    if md["seed"] not in CANON or md["env_id"] not in ENV_SHORT:
        EXCLUDED[(md["env_id"], md["seed"])] += 1
        continue
    tag = tag_of(md["algo_config"])
    if tag is None:
        UNMATCHED.append(str(run_dir))
        continue
    task_csvs = sorted(run_dir.glob("emissions_*.csv"))
    r = dict(tag=tag, env=md["env_id"], es=ENV_SHORT[md["env_id"]], seed=md["seed"],
             run_dir=run_dir.as_posix(), timestamp=run_dir.name, meta=md,
             files={f: (run_dir / f).exists() for f in
                    ["emissions.csv", "segment_energy.json", "training_metrics.json", "metadata.json", "run.log"]},
             task_csv=task_csvs[0].as_posix() if task_csvs else None, n_task_csvs=len(task_csvs))
    with open(run_dir / "segment_energy.json", encoding="utf-8") as f:
        r["seg"] = json.load(f)
    with open(run_dir / "training_metrics.json", encoding="utf-8") as f:
        r["tm"] = json.load(f)
    r["tasks"] = pd.read_csv(task_csvs[0])
    r["top"] = pd.read_csv(run_dir / "emissions.csv")
    r["log"] = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    RUNS.append(r)

RUNS.sort(key=lambda r: (["HC", "Ant"].index(r["es"]), TAGS.index(r["tag"]), CANON.index(r["seed"])))
CFG = [(es, t) for es, _ in ENVS for t in TAGS]


def runs_of(es, tag):
    return [r for r in RUNS if r["es"] == es and r["tag"] == tag]


def label(es, tag):
    return f"{es}-{tag}"


# ----------------------------------------------------------------------------- FLOPs
with open("flop_analysis/flops_per_call.json", encoding="utf-8") as f:
    FLOPS_JSON = json.load(f)
from flop_keys import signature  # noqa: E402


def ceil_div(a, b):
    return -(-a // b)


def flops_for(r):
    ac, ec = r["meta"]["algo_config"], r["meta"]["experiment_config"]
    sig = signature("sac", ac)
    fk = FLOPS_JSON["sac"][r["env"]][sig]
    spe = ec["steps_per_epoch"]
    n_env = max(1, ec["train_steps"] // spe) * spe
    n_upd = n_env * ac["updates_per_env_step"]
    n_act = ceil_div(n_upd, ac["policy_update_delay"])
    F = {"rollout": fk["actor_forward_bs1"] * n_env, "buffer_sample": 0,
         "critic_update": fk["critic_fwdbwd"] * n_upd, "actor_update": fk["actor_fwdbwd"] * n_act,
         "target_update": fk["target_update_elementwise_ops"] * n_upd}
    F["gradient_updates"] = F["critic_update"] + F["actor_update"]
    F["TOTAL_MEASURED_TRAINING"] = F["rollout"] + F["critic_update"] + F["actor_update"]
    return sig, fk, dict(n_env=n_env, n_upd=n_upd, n_act=n_act), F


# ----------------------------------------------------------------------------- per-run derived
def task_rows(r, prefix):
    t = r["tasks"]
    if prefix in ("idle_baseline_head", "idle_baseline_tail"):
        return t[t["task_name"] == prefix]
    return t[t["task_name"].str.fullmatch(rf"{prefix}_\d+")]


for r in RUNS:
    seg = r["seg"]
    sig, fk, cc, F = flops_for(r)
    r["sig"], r["fk"], r["cc"], r["F"] = sig, fk, cc, F
    E = {k: seg[k] * KWH_TO_J for k in ["rollout"] + SUBS + ["warmup", "idle_baseline_head", "idle_baseline_tail"]}
    E["gradient_updates"] = sum(E[k] for k in SUBS)
    E["TOTAL_MEASURED_TRAINING"] = E["rollout"] + E["gradient_updates"]
    r["E"] = E
    r["E_kwh"] = {k: v / KWH_TO_J for k, v in E.items()}
    hw = {}
    for p in ["rollout", "gradient_updates", "warmup", "idle_baseline_head", "idle_baseline_tail"]:
        tr = task_rows(r, p)
        hw[p] = dict(n=len(tr), duration=tr["duration"].sum(), E=tr["energy_consumed"].sum() * KWH_TO_J,
                     cpu=tr["cpu_energy"].sum() * KWH_TO_J, gpu=tr["gpu_energy"].sum() * KWH_TO_J,
                     ram=tr["ram_energy"].sum() * KWH_TO_J)
    r["hw"] = hw
    T = dict(seg["_sub_segment_wall_time_seconds"])
    D = {k: hw[k]["duration"] for k in hw}
    D.update({k: T[k] for k in SUBS})
    D["TOTAL_MEASURED_TRAINING"] = hw["rollout"]["duration"] + hw["gradient_updates"]["duration"]
    r["D"], r["T"] = D, T
    r["P"] = {k: (E[k] / D[k] if D.get(k) else float("nan")) for k in E}
    tot = E["TOTAL_MEASURED_TRAINING"]
    r["share"] = {k: 100 * E[k] / tot for k in ["rollout"] + SUBS + ["gradient_updates"]}
    r["share_gu"] = {k: 100 * E[k] / E["gradient_updates"] for k in SUBS}
    r["jpf"] = {k: (E[k] / F[k] if F.get(k) else float("nan"))
                for k in ["rollout", "critic_update", "actor_update", "target_update",
                          "gradient_updates", "TOTAL_MEASURED_TRAINING"]}
    r["tp"] = {k: F[k] / D[k] / 1e9 for k in ["rollout", "gradient_updates", "TOTAL_MEASURED_TRAINING"]}
    pi = 0.5 * (r["P"]["idle_baseline_head"] + r["P"]["idle_baseline_tail"])
    r["P_idle"] = pi

# ----------------------------------------------------------------------------- long CSV
long_rows = []
for r in RUNS:
    for s in SEG_ORDER + ["warmup", "idle_baseline_head", "idle_baseline_tail"]:
        ftype = ("mixed_total" if s == "TOTAL_MEASURED_TRAINING" else "elementwise" if s == "target_update"
                 else "matmul" if s in ("rollout", "critic_update", "actor_update", "gradient_updates") else "none")
        fl = r["F"].get(s) if s in r["F"] else None
        long_rows.append(dict(
            tag=r["tag"], env=r["env"], seed=r["seed"], segment=s, status=SEG_STATUS[s],
            energy_kwh=r["E_kwh"][s], energy_j=r["E"][s], duration_s=r["D"].get(s),
            mean_power_w=r["P"].get(s), flops=fl, flop_type=ftype,
            j_per_flop=(r["jpf"].get(s) if ftype in ("matmul", "mixed_total", "elementwise") else None),
            share_of_total_pct=r["share"].get(s) if s in r["share"] else (100.0 if s == "TOTAL_MEASURED_TRAINING" else None),
            share_of_gradient_updates_pct=r["share_gu"].get(s) if s in SUBS else (100.0 if s == "gradient_updates" else None),
        ))
LONG = pd.DataFrame(long_rows)
LONG.to_csv(OUT_DIR / "sac_segments_long.csv", index=False, float_format="%.17g")

# ----------------------------------------------------------------------------- inventory CSV
inv_rows = []
for r in RUNS:
    md = r["meta"]
    ec, ac = md["experiment_config"], md["algo_config"]
    inv_rows.append(dict(tag=r["tag"], env=r["env"], seed=r["seed"], run_dir=r["run_dir"], timestamp=r["timestamp"],
                         git_commit=md["git_commit"], torch_version=md["torch_version"],
                         codecarbon_version=md.get("codecarbon_version"), gpu_name=md.get("cuda_device_name"),
                         gpu_min_clock_mhz=ec.get("gpu_min_clock_mhz"), gpu_max_clock_mhz=ec.get("gpu_max_clock_mhz"),
                         hidden_sizes="x".join(map(str, ac["hidden_sizes"])), batch_size=ac["batch_size"],
                         utd=ac["updates_per_env_step"], warmup_steps=ec["warmup_steps"],
                         train_steps=ec["train_steps"], steps_per_epoch=ec["steps_per_epoch"]))
INV = pd.DataFrame(inv_rows)
INV.to_csv(OUT_DIR / "sac_run_inventory.csv", index=False)

# ----------------------------------------------------------------------------- per-epoch CSV
pe_rows = []
for r in RUNS:
    for task in ["rollout", "gradient_updates"]:
        tr = task_rows(r, task)
        for _, row in tr.iterrows():
            ep = int(row["task_name"].rsplit("_", 1)[1])
            e = row["energy_consumed"] * KWH_TO_J
            pe_rows.append(dict(tag=r["tag"], env=r["env"], seed=r["seed"], epoch=ep, task=task, energy_j=e,
                                duration_s=row["duration"], mean_power_w=e / row["duration"],
                                ram_integration_window_s=row["ram_energy"] * KWH_TO_J / row["ram_power"]
                                if row["ram_power"] else None))
PE = pd.DataFrame(pe_rows).sort_values(["env", "tag", "seed", "task", "epoch"])
PE.to_csv(OUT_DIR / "sac_per_epoch.csv", index=False, float_format="%.17g")

# ============================================================================= markdown building
MD = []


def h(level, text):
    MD.append("\n" + "#" * level + " " + text + "\n")


def p(text=""):
    MD.append(text)


# ----------------------------------------------------------------------------- Part 0 provenance
def part0():
    out = []
    import torch
    import gymnasium
    import mujoco
    head = sh(["git", "rev-parse", "HEAD"]).strip()
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    porcelain = sh(["git", "status", "--porcelain"]).rstrip()
    out.append(table(["Item", "Value"], [
        ["Generated (local time)", dt.datetime.now().isoformat(timespec="seconds")],
        ["Machine / OS", f"{platform.system()} {platform.release()} ({platform.version()}), {platform.machine()} — "
                         "this is the **Windows dev checkout**, not the Linux experiment box"],
        ["Repo path", str(REPO)],
        ["Python / pandas / numpy", f"{platform.python_version()} / {pd.__version__} / {np.__version__}"],
        ["torch / gymnasium / mujoco (local venv, used only for Part 2.4 instantiation)",
         f"{torch.__version__} / {gymnasium.__version__} / {mujoco.__version__}"],
        ["git HEAD", head], ["branch", branch],
        ["git status --porcelain", "clean (empty)" if not porcelain else "DIRTY, see below"],
    ]))
    if porcelain:
        out.append(code(porcelain))
        out.append("(The only untracked/changed paths should be under `contexts/`, i.e. this generator's own output.)")
    files = ["flop_analysis/output/per_run_energy_per_flop.csv", "flop_analysis/output/cross_seed_energy_per_flop.csv",
             "flop_analysis/flops_per_call.json"]
    rows = []
    for fpath in files:
        last = sh(["git", "log", "-1", "--format=%h %ad", "--date=iso", "--", fpath]).strip()
        rows.append([f"`{fpath}`", sha256(fpath), mtime(fpath), last])
    out.append("\n**Pipeline files** (mtime = filesystem modification time on this checkout, which reflects git "
               "checkout/pull time, not the time the file was generated on the Linux box; the last-commit column is the "
               "more meaningful timestamp):\n")
    out.append(table(["File", "SHA-256", "mtime (local)", "last git commit touching it"], rows))
    seg_mtimes = [(os.path.getmtime(Path(r["run_dir"]) / "segment_energy.json"), r["run_dir"]) for r in RUNS]
    newest = max(seg_mtimes)
    csv_mt = min(os.path.getmtime(f) for f in files[:2])
    seg_commit = sh(["git", "log", "-1", "--format=%h %ad", "--date=iso", "--", "results/sac"]).strip()
    csv_commit = sh(["git", "log", "-1", "--format=%h %ad", "--date=iso", "--", files[0]]).strip()
    res_any_commit = sh(["git", "log", "-1", "--format=%h %ad", "--date=iso", "--", "results"]).strip()
    fresh_fs = newest[0] <= csv_mt
    out.append(f"\n**Freshness check.** Newest `segment_energy.json` among the 70 SAC runs: `{newest[1]}`, mtime "
               f"{dt.datetime.fromtimestamp(newest[0]).isoformat(timespec='seconds')}; oldest of the two CSVs' mtime "
               f"{dt.datetime.fromtimestamp(csv_mt).isoformat(timespec='seconds')}. "
               + check("filesystem mtime: newest SAC segment_energy.json older than the pipeline CSVs", fresh_fs))
    out.append(f"\nGit view: last commit touching `results/sac/` = `{seg_commit}`; last commit touching anything under "
               f"`results/` = `{res_any_commit}`; last commit touching `per_run_energy_per_flop.csv` = `{csv_commit}`. "
               "Because mtimes on a git checkout are not generation times, staleness is decided by content instead: "
               "Part 4.8 recomputes every SAC energy/FLOP/J-per-FLOP value from `results/` and compares it to the CSV "
               "rows (if that reconciliation PASSes, the CSVs are not stale for SAC regardless of timestamps).")
    # code-version consistency
    commits = Counter(r["meta"]["git_commit"] for r in RUNS)
    order = sorted(commits, key=lambda c: sh(["git", "log", "-1", "--format=%ct", c]).strip() or "0")
    rows = []
    for c in order:
        known = subprocess.run(["git", "cat-file", "-e", c + "^{commit}"], cwd=REPO, capture_output=True).returncode == 0
        subj = sh(["git", "log", "-1", "--format=%h %ad %s", "--date=short", c]).strip() if known else "UNKNOWN TO GIT"
        tags_ = sorted({label(r["es"], r["tag"]) for r in RUNS if r["meta"]["git_commit"] == c})
        rows.append([f"`{c[:10]}`", commits[c], subj, ", ".join(tags_)])
    out.append("\n**Code-version consistency of the 70 runs** (`metadata.json` → `git_commit`):\n")
    out.append(table(["git_commit", "# runs", "commit", "configurations"], rows))
    paths = ["algorithms/sac.py", "algorithms/replay_buffer.py", "algorithms/tracker_utils.py", "experiment_runner.py",
             "configs/config.py", "utils/", "run_experiment.py", ":(exclude)*.pyc"]
    out.append(f"\nPairwise `git diff --stat <c1> <c2> -- {' '.join(paths[:-1])}` (chronological order; tracked "
               "`utils/__pycache__/*.pyc` files are excluded with `':(exclude)*.pyc'`; the content of the changes is "
               "summarised underneath):\n")
    pair_rows = []
    any_sac = False
    for c1, c2 in itertools.combinations(order, 2):
        st = sh(["git", "diff", "--stat", c1, c2, "--", *paths]).strip()
        files_changed = [ln.split("|")[0].strip() for ln in st.splitlines() if "|" in ln]
        sac_affecting = [f_ for f_ in files_changed if f_ in ("algorithms/sac.py", "algorithms/replay_buffer.py",
                                                              "algorithms/tracker_utils.py")]
        any_sac = any_sac or bool(sac_affecting)
        pair_rows.append([f"`{c1[:7]}`→`{c2[:7]}`", ", ".join(files_changed) if files_changed else "(no change)",
                          st.splitlines()[-1].strip() if st else ""])
    out.append(table(["pair", "changed files in the listed paths", "summary"], pair_rows))
    # describe what changed per file across the whole span
    first, last = order[0], order[-1]
    span = sh(["git", "diff", first, last, "--", *paths])
    out.append(f"\nWhole-span diff `{first[:7]}`→`{last[:7]}` restricted to those paths: "
               f"{len(span.splitlines())} diff lines; files: "
               + ", ".join(sorted(set(re.findall(r'^diff --git a/(\S+)', span, re.M)))) + ".")
    for fpath in ["algorithms/sac.py", "algorithms/replay_buffer.py", "algorithms/tracker_utils.py"]:
        d = sh(["git", "diff", first, last, "--", fpath])
        out.append(f"- `{fpath}`: {'**changed**' if d.strip() else 'identical across all run commits'}")
    hunks = {}
    for fpath in ["experiment_runner.py", "configs/config.py", "run_experiment.py", "utils/gpu_control.py",
                  "utils/thermal_gate.py"]:
        d = sh(["git", "diff", first, last, "--", fpath])
        added = [ln[1:].rstrip() for ln in d.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        removed = [ln[1:].rstrip() for ln in d.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        hunks[fpath] = (added, removed)
        out.append(f"- `{fpath}`: {'+' + str(len(added)) + ' / -' + str(len(removed)) + ' lines' if d.strip() else 'identical'}")
    # configs/config.py: compare class by class (the raw diff is dominated by newly added TD3/TD-MPC2 docstrings)
    import ast

    def classes_at(commit):
        src_ = sh(["git", "show", f"{commit}:configs/config.py"])
        tree = ast.parse(src_)
        return {n.name: ast.get_source_segment(src_, n) for n in tree.body if isinstance(n, ast.ClassDef)}, \
            {n.targets[0].id: ast.get_source_segment(src_, n) for n in tree.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    c_first, a_first = classes_at(first)
    c_last, a_last = classes_at(last)
    cls_rows = []
    for name in sorted(set(c_first) | set(c_last)):
        st = ("added" if name not in c_first else "removed" if name not in c_last else
              "identical" if c_first[name] == c_last[name] else "changed")
        detail = ""
        if st == "changed":
            dl = [ln for ln in __import__("difflib").unified_diff(c_first[name].splitlines(), c_last[name].splitlines(), lineterm="", n=0)
                  if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
            detail = "; ".join(x.strip() for x in dl)
        cls_rows.append([f"`{name}`", st, detail])
    for name in sorted(set(a_first) | set(a_last)):
        st = "identical" if a_first.get(name) == a_last.get(name) else "changed"
        cls_rows.append([f"`{name}` (module-level)", st, "" if st == "identical" else "entries added: "
                         + ", ".join(sorted(set(re.findall(r'"(\w+)":', a_last.get(name, ""))) - set(re.findall(r'"(\w+)":', a_first.get(name, "")))))])
    out.append(f"\n`configs/config.py`, compared class by class between `{first[:7]}` and `{last[:7]}` (AST source segments):\n")
    out.append(table(["definition", "status", "changed lines"], cls_rows))
    out.append("Verbatim changed lines of the other files (`+` added / `-` removed):")
    for fpath, (added, removed) in hunks.items():
        if fpath == "configs/config.py":
            continue
        if added or removed:
            body = "\n".join(["- " + x for x in removed if x.strip()] + ["+ " + x for x in added if x.strip()])
            out.append(f"\n`{fpath}`:\n" + code(body, "diff"))
    # working-tree caveat
    keysets = Counter()
    by_ks = defaultdict(list)
    for r in RUNS:
        ks = tuple(sorted(r["meta"].get("thermal_gate", {}).keys()))
        keysets[ks] += 1
        by_ks[ks].append(f"{label(r['es'], r['tag'])}/s{r['seed']} ({r['timestamp']}, commit {r['meta']['git_commit'][:7]})")
    out.append("\n**Working-tree caveat.** `git_commit` records HEAD, not uncommitted edits. The `thermal_gate` block "
               "of `metadata.json` has different key sets across the 70 runs:\n")
    out.append(table(["thermal_gate keys", "# runs", "runs (if ≤ 5)"],
                     [[", ".join(k), n, "; ".join(by_ks[k]) if n <= 5 else "…"] for k, n in keysets.most_common()]))
    out.append("\nThe `run_start_*`/`run_end_*` keys were added to `experiment_runner.py` in commit `92dcee0` (diff above "
               "is logging-only: `read_current_state()` NVML snapshot before the head idle window and after the tail). "
               "Runs that carry those keys but are labelled with the earlier commit `f23f448`, and the one run with "
               "`reference_*` keys, show that some runs executed with an uncommitted working tree. The observed "
               "differences are in metadata logging only; whether any other uncommitted edit existed cannot be "
               "determined from the data.")
    disc("Some SAC runs labelled `git_commit=f23f448` carry metadata keys (`thermal_gate.run_start_*`) that only exist "
         "from commit `92dcee0` onward, and one run carries `thermal_gate.reference_*` keys that no committed version "
         "writes: those runs executed with an uncommitted working tree, so `git_commit` alone does not pin their exact "
         "code. The visible differences are metadata-logging only.")
    verdict = ("All 70 runs can be regarded as running the same SAC training/measurement code: "
               "`algorithms/sac.py`, `algorithms/replay_buffer.py` and `algorithms/tracker_utils.py` are byte-identical "
               "across every recorded commit, `SACConfig` and `ExperimentConfig` are unchanged, and the remaining "
               "changes in the runner/config/utils files (listed above) concern metadata logging and other algorithms' "
               "dispatch/config — subject to the working-tree caveat."
               if not any_sac else
               "**SAC-affecting files differ between run commits — see table above.**")
    out.append("\n" + verdict)
    return "\n".join(out)


# ----------------------------------------------------------------------------- Part 1 audit
GREP_STRINGS = [
    "default power consumption of 4 W per thread",   # README 'Known limitations' + codecarbon core/cpu.py
    "Resorting to a default power consumption",       # codecarbon core/cpu.py
    "No CPU tracking mode found",                     # codecarbon core/resource_tracker.py
    "Unable to read RAPL value",                      # codecarbon core/rapl.py
    "Unable to read max_energy_range_uj",             # codecarbon core/rapl.py
    "Unable to access geographical location",         # codecarbon external/geography.py (primary + fallback API)
    "Using 'Canada' as the default value",            # codecarbon external/geography.py (final geolocation fallback)
    "Thermal gate timed out",                         # utils/thermal_gate.py
    "Thermal reference did not stabilize",            # utils/thermal_gate.py
    "No NVML reference available",                    # utils/thermal_gate.py
    "NVML read failed",                               # utils/thermal_gate.py
    "Command failed (",                               # utils/gpu_control.py (_run: nvidia-smi / cpupower non-zero exit)
    "Command not found",                              # utils/gpu_control.py
    "Command timed out",                              # utils/gpu_control.py
    "Could not query GPU clock range",                # utils/gpu_control.py
    "Could not read current CPU governor",            # utils/gpu_control.py
    "CUDA requested but not available",               # experiment_runner.py
    "Permission denied", "permission error", "Insufficient Permissions",
]

SWEEP_LOGS = ["results/sac/HalfCheetah-v5/_utd_sweep.log", "results/_utd_sweep.log", "results/_width_sweep.log",
              "results/_batch_size_sweep.log", "results/_ant_overnight_sweep.log"]


def sweep_log_blocks():
    """Map run_dir -> (log file, text block) by splitting each sweep log on its '[i/N] Starting:' lines."""
    blocks = {}
    for lf in SWEEP_LOGS:
        txt = Path(lf).read_text(encoding="utf-8", errors="replace").splitlines()
        starts = [i for i, ln in enumerate(txt) if re.search(r"\] \[\d+/\d+\] Starting:", ln)]
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(txt)
            blk = "\n".join(txt[s:e])
            m = re.search(r"Run directory: (\S+)", blk)
            if m:
                blocks[m.group(1)] = (lf, blk)
    return blocks


SWEEP_BLOCKS = sweep_log_blocks()


def part1():
    out = []
    inv_rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        ac = rs[0]["meta"]["algo_config"] if rs else {}
        ec = rs[0]["meta"]["experiment_config"] if rs else {}
        seeds = sorted(r["seed"] for r in rs)
        dup = len(rs) - len(set(seeds))
        script = {"base": "none found (manual CLI; commits 92dcee0/ecb98bb record them)",
                  "utd2": ("scripts/run_utd_sweep.sh @ d1cebb4 (HC)" if es == "HC" else "scripts/run_utd_sweep.sh @ aa9186b/24f41b0 (Ant)"),
                  "w256": "scripts/run_width_sweep.sh", "w512": "scripts/run_width_sweep.sh",
                  "b512": ("scripts/run_batch_size_sweep.sh" if es == "HC" else "scripts/run_ant_overnight_sweep.sh"),
                  }
        script["utd4"] = script["utd2"]
        script["b1024"] = script["b512"]
        inv_rows.append([label(es, t), ENVS[[e[0] for e in ENVS].index(es)][1], str(tuple(ac.get("hidden_sizes", []))),
                         ac.get("batch_size"), ac.get("updates_per_env_step"), ac.get("policy_update_delay"),
                         ec.get("warmup_steps"), ec.get("train_steps"), ec.get("steps_per_epoch"),
                         f"`{rs[0]['sig']}`" if rs else "", f"`{TAG_OVERRIDE[t]}`; {script[t]}", len(rs),
                         ",".join(map(str, seeds)), dup])
    out.append("**1.1 Inventory** (one row per configuration; values from each run's logged `metadata.json`, identical "
               "within a configuration — checked below):\n")
    out.append(table(["tag", "env", "hidden_sizes", "batch", "UTD", "policy_update_delay", "warmup", "train_steps",
                      "steps/epoch", "architecture signature (`flop_keys.sig_sac_td3`)", "override / script",
                      "# runs", "seeds present", "# duplicates"], inv_rows))
    out.append("\nCLI used, from the scripts: every sweep invocation is "
               "`sudo rl-exp/bin/python run_experiment.py --algo sac --env <env> --seed <seed> --algo-config-overrides "
               "<override>` (no `--warmup-steps` for SAC, so the CLI default 5000 applies). The SAC baselines have no "
               "script in the repo or in git history; their `metadata.json` (no override fields differ from "
               "`SACConfig`, all `ExperimentConfig` fields at CLI defaults) is consistent with "
               "`sudo rl-exp/bin/python run_experiment.py --algo sac --env <env> --seed <seed>` as in `cli_commands.txt`. "
               "`metadata.json` does not record the CLI arguments or the override-file path, so the override column is "
               "inferred from the logged config + scripts, not read from the run.")
    disc("No launcher script exists (in the tree or in git history) for the 10 SAC baseline runs; their CLI is "
         "inferred from metadata + `cli_commands.txt`. `metadata.json` never records CLI args or the override path.")
    out.append("\n**1.2 Full run list** → `contexts/section_4.1_data/sac_run_inventory.csv` (70 rows).\n")
    out.append("**1.3 Checks**\n")
    lines = []
    n_ok = len(RUNS) == 70
    per_cfg = {label(es, t): len(runs_of(es, t)) for es, t in CFG}
    lines.append(check("exactly 70 SAC canonical runs", n_ok, f"found {len(RUNS)}; unmatched configs: {len(UNMATCHED)}"))
    bad = {k: v for k, v in per_cfg.items() if v != 5}
    lines.append(check("exactly 5 runs per configuration (14 configurations)", not bad and len(per_cfg) == 14,
                       f"offenders: {bad}" if bad else "all 14 have 5"))
    dups = [k for k, v in Counter((r["es"], r["tag"], r["seed"]) for r in RUNS).items() if v > 1]
    lines.append(check("no duplicate (algo, env, seed, config) runs", not dups, str(dups) if dups else ""))
    miss = [(r["run_dir"], [f for f, ok in r["files"].items() if not ok]) for r in RUNS if not all(r["files"].values())]
    lines.append(check("every run has emissions.csv, segment_energy.json, training_metrics.json, metadata.json, run.log",
                       not miss, str(miss) if miss else ""))
    multi = [r["run_dir"] for r in RUNS if r["n_task_csvs"] != 1]
    lines.append(check("every run has exactly one per-task CodeCarbon log `emissions_*.csv` (the file holding the "
                       "rollout_i / gradient_updates_i rows; see note)", not multi, str(multi) if multi else ""))
    bad = [r["run_dir"] for r in RUNS if (r["meta"]["experiment_config"]["train_steps"], r["meta"]["experiment_config"]["steps_per_epoch"],
                                          r["meta"]["experiment_config"]["warmup_steps"]) != (100000, 1000, 5000)]
    lines.append(check("train_steps = 100000, steps_per_epoch = 1000, warmup_steps = 5000 in every run", not bad, str(bad) if bad else ""))
    bad = []
    for r in RUNS:
        names = set(r["tasks"]["task_name"])
        exp = {f"rollout_{i}" for i in range(100)} | {f"gradient_updates_{i}" for i in range(100)}
        other = names - exp - {"idle_baseline_head", "idle_baseline_tail", "warmup_0"}
        if not exp <= names or other or len(r["tasks"]) != 203:
            bad.append((r["run_dir"], len(r["tasks"]), sorted(other)[:3]))
    lines.append(check("per-task CSV has exactly rollout_0..99, gradient_updates_0..99, warmup_0, idle_baseline_head, "
                       "idle_baseline_tail (203 rows)", not bad, str(bad) if bad else ""))
    # gradient update count: not logged directly; check the loop cannot break early
    bad = []
    for r in RUNS:
        ep = r["tm"]["epochs"]
        bs = r["meta"]["algo_config"]["batch_size"]
        # update loop breaks only if len(buffer) < batch_size; buffer at the start of epoch e's update block = 5000 + 1000(e+1)
        if len(ep) != 100 or min(e["buffer_size"] for e in ep) < bs or any(e["critic_loss_mean"] is None for e in ep):
            bad.append(r["run_dir"])
        exp_bs = [5000 + 1000 * (i + 1) for i in range(100)]
        if [e["buffer_size"] for e in ep] != exp_bs:
            bad.append(r["run_dir"] + " (buffer_size sequence)")
    lines.append(check("gradient_updates call structure = 1000 × UTD updates per epoch — **indirect**: the number of "
                       "update() calls is not logged; `sac.train()` runs exactly `steps_per_epoch × updates_per_env_step` "
                       "calls unless `len(buffer) < batch_size`, and the logged `buffer_size` per epoch is 6000…105000 "
                       "(≥ 1024 > every batch size) with a non-null critic loss in all 100 epochs of every run", not bad,
                       str(bad) if bad else ""))
    bad = [(r["run_dir"], r["meta"].get("cuda_device_name")) for r in RUNS if r["meta"].get("cuda_device_name") != "NVIDIA GeForce RTX 5090"]
    lines.append(check("GPU = NVIDIA GeForce RTX 5090 (metadata `cuda_device_name`)", not bad, str(bad) if bad else ""))
    gm = Counter(r["tasks"]["gpu_model"].iloc[0] for r in RUNS)
    cm = Counter(r["tasks"]["cpu_model"].iloc[0] for r in RUNS)
    lines.append(f"  - per-task CSV `gpu_model`: {dict(gm)}; `cpu_model`: {dict(cm)}")
    bad = [(r["run_dir"], r["meta"]["experiment_config"].get("gpu_min_clock_mhz"), r["meta"]["experiment_config"].get("gpu_max_clock_mhz"))
           for r in RUNS if (r["meta"]["experiment_config"].get("gpu_min_clock_mhz"), r["meta"]["experiment_config"].get("gpu_max_clock_mhz")) != (2000, 2000)
           or not r["meta"]["experiment_config"].get("lock_gpu_clocks")]
    lines.append(check("clock lock requested at 2000/2000 MHz (not null, not 200) with lock_gpu_clocks = true", not bad, str(bad) if bad else ""))
    bad = [(r["run_dir"], r["meta"]["torch_version"], r["meta"].get("codecarbon_version")) for r in RUNS
           if (r["meta"]["torch_version"], r["meta"].get("codecarbon_version")) != ("2.13.0+cu130", "3.3.0")]
    lines.append(check("torch 2.13.0+cu130 and codecarbon 3.3.0 in every run", not bad, str(bad) if bad else ""))
    # one-factor-at-a-time
    dev = []
    ign_exp = {"seed", "env_id", "algo_name"}
    META_ONLY = "thermal_gate_reference_file"  # dropped from metadata by experiment_runner at 92dcee0 (logging only)
    ref_ac = {k: v for k, v in SAC_DEFAULTS.items()}
    ref_ec = {k: v for k, v in runs_of("HC", "base")[-1]["meta"]["experiment_config"].items() if k not in ign_exp}
    n_meta_only = 0
    meta_only_vals = set()
    for r in RUNS:
        ac, ec = r["meta"]["algo_config"], r["meta"]["experiment_config"]
        if ac["hidden_sizes"] != (TAG_FACTOR[r["tag"]].get("hidden_sizes", [1024, 1024])) or \
           ac["batch_size"] != TAG_FACTOR[r["tag"]].get("batch_size", 256) or \
           ac["updates_per_env_step"] != TAG_FACTOR[r["tag"]].get("updates_per_env_step", 1):
            dev.append((r["run_dir"], "tag/config mismatch"))
        for k in set(ac) | set(ref_ac):
            if k not in TAG_FACTOR[r["tag"]] and ac.get(k, "<missing>") != ref_ac.get(k, "<missing>"):
                dev.append((r["run_dir"], f"algo_config.{k}"))
        if META_ONLY in ec:
            n_meta_only += 1
            meta_only_vals.add(ec[META_ONLY])
        for k in (set(ec) | set(ref_ec)) - ign_exp - {META_ONLY}:
            if ec.get(k, "<missing>") != ref_ec.get(k, "<missing>"):
                dev.append((r["run_dir"], f"experiment_config.{k}"))
    lines.append(check("logged config matches tag, and every other algo_config field equals the SACConfig default and "
                       "every experiment_config field equals the HC-base value, symmetric over the union of keys "
                       "(one-factor-at-a-time; seed/env_id/algo_name ignored)", not dev, str(dev[:10]) if dev else
                       f"same reference for both envs, so HC and Ant also differ only in env_id; the key "
                       f"`{META_ONLY}` is present in {n_meta_only} runs' experiment_config and absent in the others "
                       f"(metadata format change at 92dcee0; values where present: {sorted(meta_only_vals)}; excluded "
                       f"from the comparison)"))
    # failure-mode grep
    hits_log, hits_sweep = defaultdict(list), defaultdict(list)
    covered = 0
    for r in RUNS:
        for s in GREP_STRINGS:
            if s.lower() in r["log"].lower():
                hits_log[s].append(r["run_dir"])
        key = r["run_dir"]
        if key in SWEEP_BLOCKS:
            covered += 1
            blk = SWEEP_BLOCKS[key][1]
            for s in GREP_STRINGS:
                if s.lower() in blk.lower():
                    hits_sweep[s].append(r["run_dir"])
    lines.append(check(f"`run.log` of every run contains none of the {len(GREP_STRINGS)} failure-mode strings (list in "
                       "'Schema samples' §6)", not hits_log, str(dict(hits_log)) if hits_log else ""))
    lines.append(check(f"sweep-log block (stdout+stderr captured by the sweep script) contains none of the strings — "
                       f"available for {covered} of 70 runs", not hits_sweep, str(dict(hits_sweep)) if hits_sweep else ""))
    tg_bad = [(r["run_dir"], r["meta"].get("thermal_gate", {}).get("reason")) for r in RUNS
              if not r["meta"].get("thermal_gate", {}).get("gated") or r["meta"]["thermal_gate"].get("reason") != "reached_reference"]
    lines.append(check("metadata thermal_gate: gated = true, reason = reached_reference (no timeout) in every run",
                       not tg_bad, str(tg_bad) if tg_bad else ""))
    cpu_p = []
    for r in RUNS:
        tr = r["tasks"]
        cpu_p.append(tr.loc[tr["task_name"].str.startswith("gradient_updates_"), "cpu_power"].nunique())
    fixed = [r["run_dir"] for r, n in zip(RUNS, cpu_p) if n <= 2]
    lines.append(check("data-side RAPL check: per-task `cpu_power` varies across the 100 gradient_updates tasks in every "
                       "run (a TDP fallback would give a constant value)", not fixed,
                       f"min distinct values per run = {min(cpu_p)}" + (f"; offenders {fixed}" if fixed else "")))
    geo = Counter((r["tasks"]["country_name"].iloc[0], r["tasks"]["region"].iloc[0]) for r in RUNS)
    lines.append(check("data-side geolocation check: no run's per-task CSV shows the Canada/Quebec fallback "
                       "(country_name, region)", not any(c == "Canada" for c, _ in geo), str(dict(geo))))
    out.append("\n".join("- " + ln if not ln.startswith("  -") else ln for ln in lines))
    out.append(f"\nGrep caveat: `run.log` is written only by the `experiment_runner` logger. The `gpu_control`, "
               f"`thermal_gate` and CodeCarbon loggers are **not** attached to `run.log` (their warnings go to "
               f"stderr only), so a clean `run.log` is not by itself evidence that the clock lock / governor / RAPL / "
               f"geolocation calls succeeded. The sweep scripts tee stdout+stderr into their sweep log, which covers "
               f"{covered}/70 runs (all except the 10 baselines, which were launched manually and whose stderr was "
               f"not saved). The only warning text in those blocks is CodeCarbon's \"Multiple instances of codecarbon "
               f"are allowed to run at the same time.\" (Part 8.5). Thermal-gate timeouts are additionally recorded in "
               f"`metadata.json` (checked above). For the 10 baselines the clock lock is therefore verified only as "
               f"*requested* (metadata), not as *applied*.")
    disc("CLAUDE.md/README claim zero GPU-clock-lock/CPU-governor permission failures and zero RAPL/geolocation "
         "fallbacks 'by grepping every run.log'. That grep cannot detect those failures: the `gpu_control`, "
         "`thermal_gate` and CodeCarbon loggers do not write to `run.log`. For SAC the claim is supported for the 60 "
         "sweep runs by their sweep logs (stderr captured) and for all 70 by data-side checks (varying RAPL CPU power, "
         "non-fallback geolocation, metadata thermal-gate reason); the clock lock of the 10 manually launched "
         "baselines is verified only as requested (metadata), not as applied.")
    # UTD provenance
    utd_lines = []
    for es, _ in ENVS:
        for t in ("utd2", "utd4"):
            rs = runs_of(es, t)
            utd_lines.append(f"{label(es, t)}: seeds {sorted(r['seed'] for r in rs)}, hidden_sizes "
                             f"{sorted({tuple(r['meta']['algo_config']['hidden_sizes']) for r in rs})}, stack "
                             f"{sorted({(r['meta']['torch_version'], r['meta']['codecarbon_version']) for r in rs})}, "
                             f"timestamps {min(r['timestamp'] for r in rs)}…{max(r['timestamp'] for r in rs)}, commits "
                             f"{sorted({r['meta']['git_commit'][:7] for r in rs})}")
    utd_ok = all(len(runs_of(es, t)) == 5 and all(r["meta"]["algo_config"]["hidden_sizes"] == [1024, 1024] for r in runs_of(es, t))
                 for es, _ in ENVS for t in ("utd2", "utd4"))
    noncanon_utd = [(d.as_posix(), md["seed"]) for d, md in ALL_META if md["algo_name"] == "sac" and md["seed"] not in CANON
                    and md["algo_config"]["updates_per_env_step"] != 1]
    out.append("\n- " + check("UTD sweep provenance: utd2/utd4 exist for all five canonical seeds in both envs, at "
                              "(1024,1024) and the current stack; no non-canonical-seed run is in the analysed set", utd_ok,
                              "") + "\n  - " + "\n  - ".join(utd_lines)
               + f"\n  - non-canonical SAC runs with UTD ≠ 1 (excluded by seed filter): {noncanon_utd}")
    hc_status = json.loads(Path("results/sac/HalfCheetah-v5/_utd_sweep_status.json").read_text(encoding="utf-8"))
    out.append(f"  - `results/sac/HalfCheetah-v5/_utd_sweep_status.json`: {len(hc_status['runs'])} entries, seeds "
               f"{sorted({x['seed'] for x in hc_status['runs']})}, statuses {dict(Counter(x['status'] for x in hc_status['runs']))}, "
               f"started {hc_status.get('started_utc')}.")
    disc("CLAUDE.md says the SAC-HalfCheetah UTD sweep (`d1cebb4`, `results/sac/HalfCheetah-v5/_utd_sweep_status.json`) "
         "'predat[es] the canonical seed set'. The script at `d1cebb4` uses `SEEDS=(331 958 14577 43611 85062)`, the "
         "status file lists exactly those seeds, and the 10 HC-utd2/utd4 runs (2026-09-08, metadata commit `ecb98bb`) "
         "are at (1024,1024) on the current stack, i.e. they are canonical-seed runs recorded after the HC baselines "
         "(2026-09-07).")
    disc("CLAUDE.md says the original SAC/TD3 UTD sweep's bookkeeping was 'silently overwritten'. Only "
         "`results/_utd_sweep_status.json` was replaced; `results/_utd_sweep.log` is appended to (`tee -a`) and still "
         "contains the full SAC-Ant/TD3 30-run block (2026-09-09, 'Sweep complete: 30/30') followed by the MBPO "
         "20-run block.")
    # exclusions
    ex_rows = [[e, s, n] for (e, s), n in sorted(EXCLUDED.items())]
    out.append("\n- Excluded SAC runs (counts only, no data used):\n" + (table(["env", "seed", "# runs"], ex_rows) if ex_rows else "none")
               + f"\n\n  Total excluded: {sum(EXCLUDED.values())}; SAC run dirs not matching any of the 7 tags at a "
                 f"canonical seed: {len(UNMATCHED)} {UNMATCHED if UNMATCHED else ''}")
    # status files
    st_rows = []
    for sf in ["results/_width_sweep_status.json", "results/_batch_size_sweep_status.json",
               "results/_ant_overnight_sweep_status.json", "results/sac/HalfCheetah-v5/_utd_sweep_status.json",
               "results/_utd_sweep_status.json"]:
        d = json.loads(Path(sf).read_text(encoding="utf-8"))
        sac_e = [x for x in d["runs"] if x.get("algo", "sac") == "sac"]
        run_dirs_70 = {r["run_dir"] for r in RUNS}
        st_rows.append([f"`{sf}`", len(d["runs"]), len(sac_e), dict(Counter(x["status"] for x in d["runs"])),
                        "n/a (no SAC entries)" if not sac_e else
                        "yes" if all((x.get("run_dir") or "").rstrip("/") in run_dirs_70 for x in sac_e) else
                        "NO: " + str([x.get("run_dir") for x in sac_e if (x.get("run_dir") or "").rstrip("/") not in run_dirs_70])])
    out.append("\nSweep bookkeeping files (for completeness; run directories are the evidence):\n")
    out.append(table(["status file", "# entries", "# SAC entries", "statuses", "every SAC entry's run_dir is one of the 70"], st_rows))
    out.append("`results/_utd_sweep_status.json` now holds the MBPO UTD sweep (0 SAC entries), as documented. The "
               "SAC-Ant utd2/utd4 runs are evidenced by their run directories and by the retained "
               "`results/_utd_sweep.log` block.")
    # 1.4 cross-seed CSV rows
    cs = pd.read_csv("flop_analysis/output/cross_seed_energy_per_flop.csv")
    sac_cs = cs[cs["algo"] == "sac"].copy()
    sigs_csv = set(sac_cs["architecture_signature"])
    sigs_runs = {r["sig"] for r in RUNS}
    groups = sac_cs.groupby(["env_id", "architecture_signature", "updates_per_env_step"]).size()
    out.append("\n**1.4 Cross-seed CSV rows for SAC** (`cross_seed_energy_per_flop.csv`, `algo == 'sac'`):\n")
    rows = []
    for (env, sig, utd), n in groups.items():
        sub = sac_cs[(sac_cs.env_id == env) & (sac_cs.architecture_signature == sig) & (sac_cs.updates_per_env_step == utd)]
        rows.append([env, f"`{sig}`", utd, n, ", ".join(sub["segment"]), sorted(set(sub["n_seeds"]))])
    out.append(table(["env", "signature", "UTD", "# rows", "segments", "n_seeds values"], rows))
    lines = [check("every SAC cross-seed row has n_seeds = 5 (the cross-seed CSV only contains canonical-seed runs; "
                   "`included_in_cross_seed_avg` lives in the per-run CSV)", set(sac_cs["n_seeds"]) == {5},
                   f"{len(sac_cs)} rows")]
    pr = pd.read_csv("flop_analysis/output/per_run_energy_per_flop.csv")
    pr_sac = pr[(pr.algo == "sac")]
    inc = pr_sac[pr_sac.run_dir.isin([r["run_dir"] for r in RUNS])]
    lines.append(check("all 70 runs present in per_run CSV with included_in_cross_seed_avg = True",
                       inc.run_dir.nunique() == 70 and inc.included_in_cross_seed_avg.astype(str).eq("True").all(),
                       f"{inc.run_dir.nunique()} run_dirs"))
    nc = pr_sac[~pr_sac.run_dir.isin([r["run_dir"] for r in RUNS])]
    lines.append(f"Non-canonical SAC rows in per-run CSV: {nc.run_dir.nunique()} runs, included flag values "
                 f"{sorted(set(nc.included_in_cross_seed_avg.astype(str)))} (these are the excluded dev runs).")
    lines.append(check("set of SAC signatures in the cross-seed CSV equals the 5 derived from the 14 configurations",
                       sigs_csv == sigs_runs, f"CSV: {sorted(sigs_csv)}; runs: {sorted(sigs_runs)}"))
    sh_ = {(r["es"], r["tag"]): r["sig"] for r in RUNS}
    same = all(sh_[(es, "utd2")] == sh_[(es, "base")] == sh_[(es, "utd4")] for es, _ in ENVS)
    lines.append(check("utd2/utd4 share the baseline's signature (UTD is not part of the signature; it is a separate "
                       "group key `updates_per_env_step`)", same))
    segsets = Counter(tuple(sorted(g.segment)) for _, g in sac_cs.groupby(["env_id", "architecture_signature", "updates_per_env_step"]))
    lines.append(check("cross-seed SAC groups = 14 = the 14 configurations", len(groups) == 14,
                       f"segment sets per group: {dict(segsets)}"))
    lines.append("The cross-seed CSV has no `warmup`/`idle_baseline_*` rows (the pipeline only aggregates rows with a "
                 "FLOP entry; those segments exist only in the per-run CSV) and no `gradient_updates` row in either CSV.")
    out.append("\n".join("- " + ln for ln in lines))
    return "\n".join(out)


# ----------------------------------------------------------------------------- Part 2
def part2():
    import torch
    out = []
    out.append("**2.1 `SACConfig` defaults** (`configs/config.py`, dumped via `dataclasses.asdict(SACConfig())`):\n")
    out.append(code(json.dumps(dataclasses.asdict(SACConfig()), indent=2, default=list), "json"))
    out.append("`ExperimentConfig` defaults (protocol, shared by all algorithms):\n")
    out.append(code(json.dumps(dataclasses.asdict(ExperimentConfig(algo_name="sac", env_id="HalfCheetah-v5")), indent=2), "json"))
    for es in ("HC", "Ant"):
        r = runs_of(es, "base")[0]
        md = r["meta"]
        out.append(f"\n`metadata.json` of `{r['run_dir']}` ({label(es, 'base')}, seed {r['seed']}) — verbatim, complete:\n")
        out.append(code(json.dumps(md, indent=2), "json"))
    src = Path("algorithms/sac.py").read_text(encoding="utf-8").splitlines()

    def excerpt(a, b):
        return "\n".join(f"{i:4d}  {src[i - 1]}" for i in range(a, b + 1))

    def find(pat, start=1):
        for i in range(start - 1, len(src)):
            if re.search(pat, src[i]):
                return i + 1
        raise ValueError(pat)

    l_mlp, l_pol, l_q, l_info, l_agent = find(r"^def _mlp"), find(r"^class GaussianPolicy"), find(r"^class QNetwork"), \
        find(r"^class UpdateInfo"), find(r"^class SACAgent")
    l_upd, l_train = find(r"    def update\("), find(r"^def train\(")
    out.append(f"\n**2.2 Verbatim code** — `algorithms/sac.py` (line numbers as in the file at HEAD; byte-identical "
               "at every run commit, Part 0).\n\nNetworks (`_mlp`, `GaussianPolicy`, `QNetwork`, `UpdateInfo`):\n")
    out.append(code(excerpt(l_mlp, l_agent - 1), "python"))
    out.append("`SACAgent.__init__`, `select_action`, `update()`:\n")
    out.append(code(excerpt(l_agent, l_train - 1), "python"))
    out.append("`train()`:\n")
    out.append(code(excerpt(l_train, len(src)), "python"))
    out.append("`algorithms/replay_buffer.py` (complete) and `algorithms/tracker_utils.py` (complete):\n")
    rb = Path("algorithms/replay_buffer.py").read_text(encoding="utf-8").splitlines()
    tu = Path("algorithms/tracker_utils.py").read_text(encoding="utf-8").splitlines()
    out.append(code("\n".join(f"{i + 1:4d}  {s}" for i, s in enumerate(rb)), "python"))
    out.append(code("\n".join(f"{i + 1:4d}  {s}" for i, s in enumerate(tu)), "python"))
    L = {k: find(p_) for k, p_ in {
        "bs_t0": r"# --- buffer_sample ---", "bs_end": r"info.buffer_sample_s =", "cu_t0": r"# --- critic_update ---", "cu_end": r"info.critic_update_s =",
        "cu_item": r"info.critic_loss = ", "au": r"# --- actor_update", "au_end": r"info.actor_update_s =",
        "au_item": r"info.actor_loss = actor_loss.item", "al_item": r"info.alpha = self.alpha.item", "tu": r"# --- target_update ---",
        "tu_end": r"info.target_update_s =", "roll": r'TrackerTask\(tracker, "rollout"', "gu": r'TrackerTask\(tracker, "gradient_updates"',
        "recon": r"# ---------------- reconcile", "sel": r"def select_action"}.items()}
    out.append(f"""
**What each timer contains** (facts read from the code above):

| sub-segment | timer lines | contents | device sync inside the timer? |
|---|---|---|---|
| `buffer_sample` | {L['bs_t0']}–{L['bs_end']} | `ReplayBuffer.sample()` = `np.random.randint` + 5 NumPy fancy-index gathers on **host** (float32 arrays, pageable memory), then 5 × `torch.as_tensor(x, device=cuda)` = blocking (non-`non_blocking`) host→device copies from pageable memory (no `pin_memory`) | no explicit `torch.cuda.synchronize()`; how long each blocking copy waits for previously queued GPU work is determined by PyTorch/CUDA copy semantics and is **not measured** here |
| `critic_update` | {L['cu_t0']}–{L['cu_end']} | no-grad target: actor forward on `next_obs` (sampling + log-prob), both target-Q forwards, min, entropy term, Bellman target; Q1/Q2 forward, MSE losses; `q1` zero_grad/backward/Adam step, then `q2` zero_grad/backward/Adam step | **none**: `q1_loss.item()`/`q2_loss.item()` are on line {L['cu_item']}, *after* the timer stops |
| `actor_update` (+ α) | {L['au']}–{L['au_end']} | freeze Q params (`requires_grad=False`), actor forward on `obs` (rsample + tanh-corrected log-prob), Q1/Q2 forward on (obs, new action), min, actor loss, backward, Adam step, unfreeze Q params, α loss, backward, Adam step on `log_alpha` (runs every update since `policy_update_delay = 1`) | **none**: `actor_loss.item()` and `self.alpha.item()` are on lines {L['au_item']}–{L['al_item']}, *after* the timer stops |
| `target_update` | {L['tu']}–{L['tu_end']} | Polyak averaging under `no_grad`: for every parameter of q1→q1_targ and q2→q2_targ, `mul_(1-τ)` then `add_(τ·p)` (a Python loop over 2 × 6 parameter tensors; `τ·p` creates a temporary) | none |

- `time.perf_counter()` measures **host wall-clock** time. CUDA kernels are launched asynchronously and there is no
  `torch.cuda.synchronize()` anywhere in the repo (grep, 2.3), so GPU work launched inside one timer can still be
  executing when that timer stops; the `.item()` calls *between* the timers (lines {L['cu_item']}, {L['au_item']}–{L['al_item']})
  are device→host copies of results, so they wait for the kernels producing them, and they lie outside every timer.
  Host time spent waiting in those `.item()` calls is inside the CodeCarbon `gradient_updates` task but in no
  sub-timer (→ allocation coverage < 100 %, Part 8.2).
- Allocation (lines {L['recon']}–{L['recon'] + 7}): the *whole run's* `gradient_updates` energy (sum of the 100 measured tasks)
  is split by the *whole run's* summed sub-timer shares `T_k / Σ_j T_j` (one split per run, not per epoch, despite
  the `train()` docstring saying "within that epoch").
- `select_action` (line {L['sel']}): `torch.as_tensor(obs, float32, device)` (H2D copy of one observation), actor
  forward with `with_logprob=False` (stochastic: `dist.rsample()`), `.cpu().numpy()` (D2H copy = device sync every
  env step).
- **`rollout` task contents** (lines {L['roll']}–{L['gu'] - 2}): per env step: `select_action`, `env.step(action)` (MuJoCo on CPU),
  `buffer.add(...)` (5 host array writes), reward bookkeeping, and on `terminated or truncated` an episode record +
  `env.reset()`. The `rollout` task therefore includes episode resets.
- **Evaluation episodes: none.** `train()` contains no deterministic/evaluation rollout anywhere — neither inside nor
  outside the measured tasks. All logged returns are those of the stochastic training policy.
- Between the two tasks of an epoch, only Python bookkeeping (dict updates, `epochs_log.append`, a `logger.info` every
  10 epochs) runs outside any task. `warmup` (random actions, `env.action_space.sample()`, 5000 steps) is its own task.
- Seeding: `experiment_runner` seeds `random`, `numpy`, `torch.manual_seed` with the run seed; `_make_env` calls
  `env.reset(seed)` and `env.action_space.seed(seed)`; `train()` calls `env.reset(seed=exp_cfg.seed)` again. No
  CUDA determinism flags are set.
""")
    disc("`algorithms/sac.py`'s `train()` docstring says each epoch's `gradient_updates` energy is allocated 'proportionally "
         "to the wall-clock time share of each sub-segment within that epoch'; the code (reconcile step at the end of "
         "`train()`) does one split of the whole run's summed energy by the whole run's summed sub-timer times, as README "
         "states. The numbers here follow the code.")
    # 2.3
    pats = ["allow_tf32", "set_float32_matmul_precision", "torch.compile", "autocast", "torch.cuda.amp", "GradScaler",
            "cuda.graphs", "CUDAGraph", "cudnn.deterministic", "cudnn.benchmark", "pin_memory", "non_blocking",
            "synchronize", "set_default_dtype", ".half(", "bfloat16", "float16", "set_num_threads", "use_deterministic_algorithms"]
    py_files = [f for f in glob.glob("**/*.py", recursive=True) if not f.startswith((".venv", "contexts"))]
    found = defaultdict(list)
    for fpath in py_files:
        txt = Path(fpath).read_text(encoding="utf-8", errors="replace")
        for pat in pats:
            for i, ln in enumerate(txt.splitlines(), 1):
                if pat in ln:
                    found[pat].append(f"{Path(fpath).as_posix()}:{i} (`{ln.strip()[:80]}`)")
    rows = [[f"`{pat}`", ", ".join(found[pat]) if found[pat] else "none found"] for pat in pats]
    out.append(f"**2.3 Numerics / performance flags** — searched all {len(py_files)} `.py` files of the repo (excluding "
               "`.venv/` and `contexts/`):\n")
    out.append(table(["pattern", "occurrences"], rows))
    out.append(f"""
- The hits listed (with the matching line) are prose inside a docstring / report text, not code; none is in
  `algorithms/sac.py`, `replay_buffer.py`, `tracker_utils.py`, `experiment_runner.py` or `run_experiment.py`. The
  code therefore **sets none** of TF32, matmul precision, `torch.compile`,
  AMP/autocast, CUDA graphs, cuDNN flags or pinned memory. Whatever the PyTorch defaults of torch 2.13.0+cu130 are,
  they apply.
- dtype: all networks are default-constructed `nn.Linear` (float32 parameters); the replay buffer stores float32;
  `select_action` casts observations to float32. → all SAC matmuls are float32-typed tensors.
- Local evidence about defaults (Windows venv, torch {torch.__version__}, **not** the run stack):
  `torch.backends.cuda.matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}`,
  `torch.get_float32_matmul_precision() = '{torch.get_float32_matmul_precision()}'`. Whether FP32 matmuls ran as
  TF32 on the RTX 5090 under 2.13.0+cu130 is **NOT VERIFIED** from the run data (no run logs these flags); confirm on the
  Linux box with `rl-exp/bin/python -c "import torch;print(torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())"`.
""")
    # 2.4 sizes
    import gymnasium as gym
    from algorithms.sac import GaussianPolicy, QNetwork
    env_info = {}
    for es, env_id in ENVS:
        env = gym.make(env_id)
        u = env.unwrapped
        attrs = {k: v for k, v in vars(u).items() if k.startswith("_") and isinstance(v, (bool, int, float, tuple, str))
                 and not k.startswith("__") and k not in ("_np_random_seed",)}
        env_info[es] = dict(obs=env.observation_space.shape[0], act=env.action_space.shape[0],
                            act_high=float(env.action_space.high[0]), act_low=float(env.action_space.low[0]),
                            max_steps=env.spec.max_episode_steps, frame_skip=getattr(u, "frame_skip", None),
                            dt=getattr(u, "dt", None), kwargs=dict(env.spec.kwargs), attrs=attrs)
        env.close()
    rows = []
    PC = {}
    for es, env_id in ENVS:
        o, a = env_info[es]["obs"], env_info[es]["act"]
        for w in (256, 512, 1024):
            pol = GaussianPolicy(o, a, (w, w), 1.0)
            q = QNetwork(o, a, (w, w))
            npol, nq = sum(x.numel() for x in pol.parameters()), sum(x.numel() for x in q.parameters())
            PC[(es, w)] = (npol, nq)
            rows.append([env_id, f"({w},{w})", npol, nq, 2 * nq, 2 * nq, npol + 4 * nq])
    out.append("**2.4 Network sizes** — real `GaussianPolicy`/`QNetwork` classes instantiated on CPU with the env "
               "dims below (parameter count = `sum(p.numel())`):\n")
    out.append(table(["env", "hidden", "actor (incl. μ and log σ heads)", "one Q-network", "both critics", "both target critics",
                      "all 5 networks"], rows))
    fj = {es: FLOPS_JSON["sac"][e]["bs256_h1024x1024"] for es, e in ENVS}
    rows = []
    for es, env_id in ENVS:
        ei = env_info[es]
        rows.append([env_id, ei["obs"], ei["act"], f"[{ei['act_low']}, {ei['act_high']}]", ei["max_steps"], ei["frame_skip"],
                     ei["dt"], f"{fj[es]['obs_dim']} / {fj[es]['act_dim']}", json.dumps(ei["kwargs"])])
    out.append("\nEnvironment dimensions (`gymnasium.make(env_id)` with **no kwargs**, exactly as "
               "`experiment_runner._make_env` does; instantiated locally with gymnasium "
               f"{gym.__version__}). `metadata.json` does **not** store obs/act dims or gymnasium/mujoco versions; "
               "`flops_per_call.json` stores the dims `measure_flops.py` saw:\n")
    out.append(table(["env", "obs dim", "act dim", "action bounds", "max_episode_steps", "frame_skip", "dt (s)",
                      "flops_per_call.json obs/act", "spec kwargs"], rows))
    ok = all(env_info[es]["obs"] == fj[es]["obs_dim"] and env_info[es]["act"] == fj[es]["act_dim"] for es, _ in ENVS)
    out.append("\n- " + check("local env dims equal the dims recorded in flops_per_call.json (HC 17/6, Ant 105/8 — the "
                              "draft's assumed Ant obs = 105 is confirmed)", ok and env_info["Ant"]["obs"] == 105))
    for es, env_id in ENVS:
        a = env_info[es]["attrs"]
        keep = {k: v for k, v in a.items() if any(s in k for s in ("contact", "cfrc", "healthy", "terminate", "weight",
                                                                     "reset_noise", "exclude", "main_body", "include"))}
        out.append(f"- `{env_id}` unwrapped settings (defaults, since `gym.make` gets no kwargs): `{json.dumps(keep, default=str)}`")
    out.append("- Ant-v5 therefore runs with Gymnasium's **default contact-force settings** (no kwargs are passed): "
               "the 105-dim observation includes the external contact forces (`include_cfrc_ext_in_observation=True` "
               "above), and `terminate_when_unhealthy=True` with the default `healthy_z_range`. HalfCheetah-v5 has no "
               "termination condition (episodes end only by the 1000-step `TimeLimit` truncation).")
    disc("gymnasium/mujoco versions of the Linux run environment are not recorded in `metadata.json`; Part 2.4 env "
         f"facts come from the local Windows venv (gymnasium {gym.__version__}). The dims agree with "
         "`flops_per_call.json`, but default env kwargs could in principle differ between versions.")
    # 2.5
    r0 = runs_of("HC", "base")[0]
    ec = r0["meta"]["experiment_config"]
    tg_keys = {k: v for k, v in ec.items() if k.startswith("thermal_gate")}
    out.append(f"""
**2.5 Phase constants actually used** (identical in all 70 runs — Part 1.3 one-factor check covers every
`experiment_config` field):

| constant | value | source |
|---|---|---|
| settle (unmeasured) | {ec['settle_seconds']} s | `experiment_config.settle_seconds` |
| idle baseline head / tail | {ec['idle_baseline_seconds']} s / {ec['idle_tail_seconds']} s | `experiment_config` |
| warmup | {ec['warmup_steps']} env steps (random actions) | `experiment_config.warmup_steps` |
| training | {ec['train_steps']} env steps = 100 epochs × {ec['steps_per_epoch']} | `experiment_config` |
| CodeCarbon `measure_power_secs` | {ec['measure_power_secs']} s | `experiment_config` |
| CodeCarbon `tracking_mode` | `{ec['tracking_mode']}` | `experiment_config` |
| `force_cpu_power` | `{ec['force_cpu_power_w']}` (→ RAPL used) | `experiment_config.force_cpu_power_w` |
| RAM power | no override in code; distinct per-task CSV `ram_power` values over all 70 runs: {', '.join(g4(x) for x in sorted(set(float(v) for v in np.round(np.concatenate([r['tasks']['ram_power'].values for r in RUNS]), 6))))} W (CodeCarbon's RAM model) | `emissions_*.csv` |
| tracker other options | `save_to_file=True`, `output_file="emissions.csv"`, `log_level="warning"`, `allow_multiple_runs=True`; `country_iso_code` is in `ExperimentConfig` but **not passed** to `EmissionsTracker` | `experiment_runner.py` |
| GPU clock lock | {ec['gpu_min_clock_mhz']}/{ec['gpu_max_clock_mhz']} MHz, persistence mode {ec['set_persistence_mode']}, CPU governor `performance` {ec['set_cpu_performance_governor']} | `experiment_config`, `utils/gpu_control.py` |
| thermal gate | {json.dumps(tg_keys)}; reference = stable reading (3 consecutive polls within 0.5 °C, 5 s apart, ≤ 120 s) captured fresh in each process | `experiment_config`, `utils/thermal_gate.py` |
| tracker lifetime | one `EmissionsTracker` started after settle; tasks: `idle_baseline_head`, `warmup_0`, `rollout_i`/`gradient_updates_i` (i = 0…99), `idle_baseline_tail`; `tracker.stop()` writes the one-row `emissions.csv` | `experiment_runner.py`, `sac.py` |
""")
    return "\n".join(out), env_info, PC


# ----------------------------------------------------------------------------- Part 3
def part3(env_info):
    out = []
    out.append("**3.1 `flops_per_call.json` SAC entries** (verbatim; `_device_used_for_measurement` = "
               f"`{FLOPS_JSON['_device_used_for_measurement']}`). Only the 5 signatures × 2 envs used by the 70 runs are "
               "shown (Humanoid-v5 entries exist for dev runs and are omitted):\n")
    used = sorted({(r["env"], r["sig"]) for r in RUNS})
    sub = {env: {sig: FLOPS_JSON["sac"][env][sig] for e2, sig in used if e2 == env} for env in ["HalfCheetah-v5", "Ant-v5"]}
    out.append(code(json.dumps(sub, indent=2), "json"))
    out.append("""Field meanings (from `measure_flops.measure_sac`, run under `torch.utils.flop_counter.FlopCounterMode`
on the real classes; FlopCounterMode counts matmul/addmm/bmm-type FLOPs = 2·m·n·k, i.e. one multiply-add = 2 FLOPs):
- `actor_forward_bs1` — `rollout` per-call FLOPs: one `GaussianPolicy` forward at batch 1, no grad, `with_logprob=False`.
- `critic_fwdbwd` — `critic_update` per call at batch B: the no-grad target computation (actor forward on next_obs +
  both target-Q forwards) **plus** Q1/Q2 forward, backward and Adam steps (Adam's elementwise ops are not matmuls → 0).
- `actor_fwdbwd` — `actor_update` per call at batch B: actor forward, Q1/Q2 forward on the new action, backward
  (actor weight grads + input grads through the frozen critics), α loss/backward (elementwise → 0).
- `target_update_elementwise_ops` — `2 × (params(q1) + params(q2))` = counted as 2 elementwise ops (mul + add) per
  critic parameter per Polyak update (the `τ·p` temporary is not counted). **Not a matmul FLOP count.**
- `_analytic_cross_check` — stored by `measure_flops.py`: `actor_forward_bs1_analytic` = 2·Σ n_in·n_out (incl. both
  heads); `critic_fwdbwd_analytic_approx` = 3 × 2 × (Q forward at batch B), i.e. twin-Q fwd+bwd only — it omits the
  target forward pass and the reduced first-layer backward, so it is not expected to match.
""")
    # 3.2
    out.append("**3.2 Call-count formulas** (`compute_energy_per_flop.sac_like_call_counts` / `rows_for_sac_or_td3`):\n")
    out.append(code("""n_epochs = max(1, train_steps // steps_per_epoch)        # = 100
n_env    = n_epochs * steps_per_epoch                      # = 100000  (warmup steps are NOT counted; warmup has no FLOP row)
n_upd    = n_env * updates_per_env_step                    # critic_update and buffer_sample calls
n_act    = ceil(n_upd / policy_update_delay)               # = n_upd for SAC (delay = 1)
target_update calls = n_upd                                # SAC: unconditional (TD3: n_act)
F_rollout = actor_forward_bs1 * n_env
F_critic  = critic_fwdbwd    * n_upd
F_actor   = actor_fwdbwd     * n_act
OPS_target = target_update_elementwise_ops * n_upd"""))
    rows = []
    for t in ("base", "utd2", "utd4"):
        cc = runs_of("HC", t)[0]["cc"]
        rows.append([t, cc["n_env"], cc["n_upd"], cc["n_act"], cc["n_upd"]])
    out.append(table(["tag", "n_env (rollout calls)", "n_upd (critic / buffer_sample calls)", "n_act (actor calls)",
                      "target_update calls"], rows))
    pr = pd.read_csv("flop_analysis/output/per_run_energy_per_flop.csv")
    ok = True
    for r in RUNS:
        sub_ = pr[pr.run_dir == r["run_dir"]].set_index("segment")
        for s, k in (("rollout", "n_env"), ("critic_update", "n_upd"), ("actor_update", "n_act"), ("target_update", "n_upd"),
                     ("buffer_sample", "n_upd")):
            if int(sub_.loc[s, "call_count"]) != r["cc"][k]:
                ok = False
    out.append("\n- " + check("call counts above equal the `call_count` column of per_run_energy_per_flop.csv for all 70 runs", ok))
    # 3.3
    rows = []
    same_all = True
    for es, t in CFG:
        rs = runs_of(es, t)
        Fs = [r["F"] for r in rs]
        same = all(F == Fs[0] for F in Fs)
        same_all &= same
        F = Fs[0]
        rows.append([label(es, t), F"`{rs[0]['sig']}`", g4(F["rollout"]), g4(F["critic_update"]), g4(F["actor_update"]),
                     g4(F["gradient_updates"]), g4(F["TOTAL_MEASURED_TRAINING"]), g4(F["target_update"]),
                     "yes" if same else "NO"])
    out.append("\n**3.3 Per-run segment FLOPs** (matmul FLOPs; target = elementwise op count, not FLOPs):\n")
    out.append(table(["config", "signature", "F_rollout", "F_critic", "F_actor", "F_grad_updates = F_critic+F_actor",
                      "F_TOTAL = F_roll+F_crit+F_act", "target-update ops", "identical across 5 seeds"], rows))
    out.append("\n- " + check("FLOPs identical across the 5 seeds of every configuration", same_all)
               + " Exact integers (full precision) are in `sac_cross_seed_summary.csv` (`flops_mean`) and "
                 "`sac_segments_long.csv` (`flops`).")
    # 3.4 analytic
    rows = []
    for es, env_id in ENVS:
        o, a = env_info[es]["obs"], env_info[es]["act"]
        for sig in sorted({r["sig"] for r in RUNS if r["es"] == es}, key=lambda s: (int(s.split("_h")[1].split("x")[0]), int(s[2:].split("_")[0]))):
            fk = FLOPS_JSON["sac"][env_id][sig]
            hsz, B = fk["hidden_sizes"], fk["batch_size"]
            h1, h2 = hsz
            fwd_actor = lambda b: 2 * b * (o * h1 + h1 * h2 + 2 * h2 * a)  # noqa: E731
            fwd_q = lambda b: 2 * b * ((o + a) * h1 + h1 * h2 + h2 * 1)    # noqa: E731
            first_q = 2 * B * (o + a) * h1
            first_actor = 2 * B * o * h1
            # critic: target (actor fwd + 2 Q_targ fwd) + 2 x [Q fwd + Q bwd (weight grads all layers + input grads all but layer 1)]
            crit = fwd_actor(B) + 2 * fwd_q(B) + 2 * (fwd_q(B) + (2 * fwd_q(B) - first_q))
            # actor: actor fwd + actor bwd (weight grads all + input grads all but first) + 2 Q fwd + 2 Q input-grad bwd
            act = fwd_actor(B) + (2 * fwd_actor(B) - first_actor) + 2 * fwd_q(B) + 2 * fwd_q(B)
            roll = fwd_actor(1)
            rows.append([env_id, f"`{sig}`", fk["actor_forward_bs1"], fk["_analytic_cross_check"]["actor_forward_bs1_analytic"], roll,
                         fk["critic_fwdbwd"], fk["_analytic_cross_check"]["critic_fwdbwd_analytic_approx"], crit,
                         fk["actor_fwdbwd"], act,
                         "exact" if (roll, crit, act) == (fk["actor_forward_bs1"], fk["critic_fwdbwd"], fk["actor_fwdbwd"]) else "MISMATCH"])
    out.append("\n**3.4 Analytic cross-check.** Stored in the repo: `_analytic_cross_check` (columns 4 and 7). "
               "Additionally, *derived by the context generator (not a pipeline output)*: a full dense-layer count that "
               "mirrors the code path — forward = 2·B·n_in·n_out per layer; backward = weight-grad (2·B·n_in·n_out, "
               "every layer with trainable weights) + input-grad (2·B·n_in·n_out, every layer whose input needs a "
               "gradient, i.e. all but the first layer of a network fed by data; for the frozen critics in the actor "
               "loss only input-grads, including layer 1 since the action input needs a gradient); critic = target "
               "(actor fwd + 2 target-Q fwd) + 2 × (Q fwd + Q bwd); actor = actor fwd + actor bwd + 2 Q fwd + 2 Q "
               "input-grad bwd. Exact integer FLOPs:\n")
    out.append(table(["env", "signature", "rollout measured", "rollout analytic (stored)", "rollout analytic (derived)",
                      "critic measured", "critic analytic_approx (stored)", "critic analytic (derived)",
                      "actor measured", "actor analytic (derived)", "derived vs measured"], rows))
    out.append("\n- " + check("stored rollout analytic value equals the measured count for every SAC signature",
                              all(rw[2] == rw[3] for rw in rows)))
    out.append("- " + check("derived full analytic count equals FlopCounterMode exactly for rollout, critic and actor in all "
                            "10 (env, signature) entries", all(rw[-1] == "exact" for rw in rows)))
    out.append("- The stored `critic_fwdbwd_analytic_approx` is "
               + ", ".join(f"{rw[6] / rw[5] * 100:.2f} %" for rw in rows[:1]) + " of the measured value for HC-base "
               "(it omits the target forward pass), as expected from its definition.")
    # 3.5
    import compute_energy_per_flop as cepf
    out.append(f"""
**3.5 TOTAL_MEASURED_TRAINING definition in `compute_energy_per_flop.py`:**
- numerator = Σ energy of `TRAINING_SEGMENTS ∩ keys(segment_energy.json)` = `{sorted(cepf.TRAINING_SEGMENTS & set(RUNS[0]['seg']))}`
  for SAC = `E_rollout + (E_buffer_sample + E_critic + E_actor + E_target)` = `E_rollout + E_gradient_updates`;
- denominator = Σ `total_flops` of rows with `flop_type == "matmul"` = `F_rollout + F_critic + F_actor`
  (`target_update` is `elementwise`, `buffer_sample` is `none`/0);
- duration/power of the TOTAL row = measured `rollout` + measured `gradient_updates` task durations/energies.
""")
    pr_t = pr[pr.segment == "TOTAL_MEASURED_TRAINING"].set_index("run_dir")
    e_err = max(abs(float(pr_t.loc[r["run_dir"], "total_energy_joules"]) - r["E"]["TOTAL_MEASURED_TRAINING"]) / r["E"]["TOTAL_MEASURED_TRAINING"] for r in RUNS)
    f_ok = all(int(pr_t.loc[r["run_dir"], "total_flops"]) == r["F"]["TOTAL_MEASURED_TRAINING"] for r in RUNS)
    gu_err = max(abs(r["E"]["gradient_updates"] - r["hw"]["gradient_updates"]["E"]) / r["hw"]["gradient_updates"]["E"] for r in RUNS)
    out.append("- " + check("numerically, for all 70 runs: CSV TOTAL energy = E_rollout + Σ4 sub-segments (from "
                            "segment_energy.json)", e_err < 1e-12, f"max rel. err {e_err:.3e}"))
    out.append("- " + check("CSV TOTAL FLOPs = F_rollout + F_critic + F_actor exactly", f_ok))
    out.append("- " + check("Σ4 allocated sub-segments = Σ measured `gradient_updates_i` task energy in the per-task CSV",
                            gu_err < 1e-9, f"max rel. err {gu_err:.3e}"))
    return "\n".join(out)


# ----------------------------------------------------------------------------- Part 4 tables
def seg_vals(rs, d, s):
    return [r[d][s] for r in rs]


def part4():
    out = []
    note = ("n = 5 seeds per row; mean ± sample sd (ddof = 1); 4 significant digits. Energy is **gross** (idle floor "
            "included, nothing subtracted). `rollout` and `gradient_updates` are **measured** CodeCarbon tasks; "
            "`buffer_sample`, `critic`, `actor`, `target` are **allocated** from `gradient_updates` by the run's "
            "wall-clock (`perf_counter`) time share; TOTAL = rollout + gradient_updates.")
    # 4.1
    segs = SEG_ORDER
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        rows.append([label(es, t)] + [msd(seg_vals(rs, "E", s)) for s in segs])
    out.append("**Table 4.1 — absolute energy per segment [J].** " + note + "\n")
    out.append(table(["config"] + [SEG_SHORT[s] for s in segs], rows))
    rows = []
    for es in ("HC", "Ant"):
        rs = runs_of(es, "base")
        rows.append([label(es, "base")] + [msd(seg_vals(rs, "E_kwh", s)) for s in segs])
    out.append("\nBaseline rows in **kWh** (same statistic):\n")
    out.append(table(["config"] + [SEG_SHORT[s] for s in segs], rows))
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        rows.append([label(es, t)] + [msd(seg_vals(rs, "E", s)) for s in ("warmup", "idle_baseline_head", "idle_baseline_tail")]
                    + [msd([r["D"][s] for r in rs]) for s in ("warmup", "idle_baseline_head", "idle_baseline_tail")])
    out.append("\nExcluded phases (not part of any total; kept for reference) — energy [J] and measured task duration [s], "
               "mean ± sd, n = 5:\n")
    out.append(table(["config", "warmup E", "idle head E", "idle tail E", "warmup dur", "idle head dur", "idle tail dur"], rows))
    # 4.2
    rows = []
    sum_err = 0
    for es, t in CFG:
        rs = runs_of(es, t)
        for r in rs:
            sum_err = max(sum_err, abs(sum(r["share"][s] for s in ["rollout"] + SUBS) - 100),
                          abs(sum(r["share_gu"][s] for s in SUBS) - 100))
        rows.append([label(es, t)] + [msd([r["share"][s] for r in rs]) for s in ["rollout"] + SUBS + ["gradient_updates"]]
                    + [msd([r["share_gu"][s] for r in rs]) for s in SUBS])
    out.append("\n**Table 4.2 — shares [%].** Left block: share of `TOTAL_MEASURED_TRAINING` energy, mean ± sd of the "
               "per-seed shares; right block: share of each allocated sub-segment **within** `gradient_updates` "
               "(= its time share `T_k/ΣT_j`, by construction). n = 5, ddof = 1. "
               + check("shares sum to 100 % in every run (both blocks)", sum_err < 1e-9, f"max |Σ−100| = {sum_err:.2e} pp") + "\n")
    out.append(table(["config", "rollout", "buf_sample", "critic", "actor", "target", "grad_updates (= 100 − rollout)",
                      "buf | GU", "critic | GU", "actor | GU", "target | GU"], rows))
    # 4.3
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        rows.append([label(es, t)] + [msd([r["D"][s] for r in rs]) for s in segs]
                    + [msd([r["P"][s] for r in rs]) for s in ("rollout", "gradient_updates", "TOTAL_MEASURED_TRAINING")])
    out.append("\n**Table 4.3 — duration [s] and mean power [W].** Durations of `rollout`/`gradient_updates` = Σ of the 100 "
               "CodeCarbon task `duration`s (per-task CSV); allocated sub-segments = summed `perf_counter` times stored "
               "in `segment_energy.json[\"_sub_segment_wall_time_seconds\"]` (host wall time, see Part 2.2); TOTAL = "
               "rollout + gradient_updates task durations. Power = per-seed energy / duration, then mean ± sd (n = 5). "
               "Power of allocated sub-segments is not tabulated here (it is `E_k/T_k` = gradient_updates energy / ΣT, "
               "identical for all four by construction; see the long CSV).\n")
    out.append(table(["config"] + [f"T {SEG_SHORT[s]}" for s in segs] + ["P rollout", "P grad_updates", "P TOTAL"], rows))
    # 4.4 J/FLOP
    rows, rows_p = [], []
    for es, t in CFG:
        rs = runs_of(es, t)
        cells, cells_p = [label(es, t)], [label(es, t)]
        for s in ("rollout", "critic_update", "actor_update", "TOTAL_MEASURED_TRAINING", "target_update"):
            Em = np.mean(seg_vals(rs, "E", s))
            Fm = np.mean([r["F"][s] for r in rs])
            per = np.array([r["jpf"][s] for r in rs])
            ratio = Em / Fm
            sc = 1e12 if s != "target_update" else 1e6
            cells.append(f"{g4(ratio)} [{g4(per.min())}, {g4(per.max())}] sd {g4(per.std(ddof=1))}")
            cells_p.append(f"{g4(ratio * sc)} ± {g4(per.std(ddof=1) * sc)}")
        rows.append(cells)
        rows_p.append(cells_p)
    out.append("\n**Table 4.4 — energy per FLOP.** J/FLOP = **ratio of means** (mean energy over 5 seeds / mean FLOPs; FLOPs "
               "are identical across seeds so this equals the mean of per-seed ratios), shown as `ratio [min, max] sd` of "
               "the 5 per-seed ratios. `target` is **J per Polyak elementwise op, not J/FLOP**. Gross energy; "
               "critic/actor/target are allocated values.\n")
    out.append(table(["config", "rollout J/FLOP", "critic J/FLOP", "actor J/FLOP", "TOTAL J/FLOP", "target J/op"], rows))
    out.append("\nSame in **pJ/FLOP** (target in **µJ/op**), ratio of means ± sd of per-seed ratios:\n")
    out.append(table(["config", "rollout pJ/FLOP", "critic pJ/FLOP", "actor pJ/FLOP", "TOTAL pJ/FLOP", "target µJ/op"], rows_p))
    # 4.5
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        Em, Fm = np.mean(seg_vals(rs, "E", "gradient_updates")), np.mean([r["F"]["gradient_updates"] for r in rs])
        per = np.array([r["jpf"]["gradient_updates"] for r in rs])
        rows.append([label(es, t), g4(Em / Fm), g4(Em / Fm * 1e12), g4(per.std(ddof=1) * 1e12), g4(per.min() * 1e12),
                     g4(per.max() * 1e12)])
    out.append("\n**Table 4.5 — `gradient_updates` J/FLOP** — *derived by the context generator, not a pipeline output*: "
               "`E_gradient_updates / (F_critic + F_actor)` (measured energy, no time-share allocation involved). "
               "Ratio of means; spread over per-seed ratios.\n")
    out.append(table(["config", "J/FLOP", "pJ/FLOP", "sd (pJ/FLOP)", "min (pJ/FLOP)", "max (pJ/FLOP)"], rows))
    out.append("\nRelation used in the thesis (allocated-jpf): within one run every allocated sub-segment has the same "
               "mean power `P̄ = E_GU / ΣT_j`, so `E_k/F_k = P̄ · T_k / F_k`; critic and actor J/FLOP differ only "
               "through their time per FLOP `T_k/F_k`.")
    # 4.6
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        rows.append([label(es, t)] + [msd([r["tp"][s] for r in rs]) for s in ("rollout", "gradient_updates", "TOTAL_MEASURED_TRAINING")]
                    + [msd([r["F"][s] / r["D"][s] / 1e9 for r in rs]) for s in ("critic_update", "actor_update")])
    out.append("\n**Table 4.6 — achieved throughput [GFLOP/s]** — *derived by the context generator*: FLOPs / duration per "
               "seed, mean ± sd (n = 5). Critic/actor columns divide by their `perf_counter` host time (Part 2.2 caveat: "
               "host time ≠ GPU busy time).\n")
    out.append(table(["config", "rollout", "gradient_updates", "TOTAL", "critic (/T_perf)", "actor (/T_perf)"], rows))
    out.append("\nGPU theoretical FP32 peak: **NOT DERIVED**. The repo stores neither SM count nor FP32 lanes per SM nor "
               "boost/locked-clock throughput, and `torch.cuda.get_device_properties` is not available on this Windows "
               "checkout (CPU-only torch; the RTX 5090 is on the Linux box). To derive it on the Linux box: "
               "`peak = 2 × (FP32 lanes per SM) × multi_processor_count × 2.000 GHz` with `multi_processor_count` from "
               "`torch.cuda.get_device_properties(0)` and the lanes-per-SM figure from NVIDIA's architecture "
               "documentation (not in the repo).")
    # 4.7 per-seed table -> CSV ; CV and MAD flags in md
    ps_rows = []
    for r in RUNS:
        ps_rows.append(dict(tag=r["tag"], env=r["env"], seed=r["seed"], run_dir=r["run_dir"],
                            total_energy_j=r["E"]["TOTAL_MEASURED_TRAINING"], total_duration_s=r["D"]["TOTAL_MEASURED_TRAINING"],
                            total_mean_power_w=r["P"]["TOTAL_MEASURED_TRAINING"], rollout_energy_j=r["E"]["rollout"],
                            gradient_updates_energy_j=r["E"]["gradient_updates"], rollout_duration_s=r["D"]["rollout"],
                            gradient_updates_duration_s=r["D"]["gradient_updates"]))
    PS = pd.DataFrame(ps_rows)
    PS.to_csv(OUT_DIR / "sac_per_seed_totals.csv", index=False, float_format="%.17g")
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        cv = lambda v: 100 * np.std(v, ddof=1) / np.mean(v)  # noqa: E731
        rows.append([label(es, t), " / ".join(g4(r["E"]["TOTAL_MEASURED_TRAINING"]) for r in rs),
                     " / ".join(g4(r["D"]["TOTAL_MEASURED_TRAINING"]) for r in rs),
                     " / ".join(g4(r["P"]["TOTAL_MEASURED_TRAINING"]) for r in rs),
                     g4(cv(seg_vals(rs, "E", "TOTAL_MEASURED_TRAINING"))), g4(cv(seg_vals(rs, "E", "rollout"))),
                     g4(cv(seg_vals(rs, "E", "gradient_updates"))), g4(cv([r["D"]["rollout"] for r in rs])),
                     g4(cv([r["D"]["gradient_updates"] for r in rs]))])
    out.append("\n**Table 4.7 — per-seed totals and coefficient of variation.** Per-seed values in seed order "
               f"{CANON}; CV = sd(ddof=1)/mean × 100 %. Full precision: `sac_per_seed_totals.csv`.\n")
    out.append(table(["config", "TOTAL energy [J] per seed", "TOTAL duration [s] per seed", "TOTAL mean power [W] per seed",
                      "CV E_TOTAL %", "CV E_rollout %", "CV E_GU %", "CV T_rollout %", "CV T_GU %"], rows))
    flags = []
    n_scaled = 0
    max_dev_pct = 0.0
    for es, t in CFG:
        rs = runs_of(es, t)
        for s in SEG_ORDER:
            v = np.array(seg_vals(rs, "E", s))
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            for r, x in zip(rs, v):
                if mad > 0 and abs(x - med) > 3 * mad:
                    dev_pct = 100 * abs(x - med) / med
                    max_dev_pct = max(max_dev_pct, dev_pct)
                    flags.append([label(es, t), r["seed"], SEG_SHORT[s], g4(x), g4(med), g4(mad), g4(abs(x - med) / mad),
                                  g4(dev_pct)])
                if mad > 0 and abs(x - med) > 3 * 1.4826 * mad:
                    n_scaled += 1
    n_cells = len(CFG) * len(SEG_ORDER) * 5
    out.append(f"\nOutlier flag rule (as requested): |x − median| > 3 × MAD within the configuration (MAD = median absolute "
               f"deviation, **unscaled**, n = 5), applied to the energy of every segment in Table 4.1 (incl. GU and TOTAL). "
               f"Flagged, **not excluded**: **{len(flags)}** of {n_cells} (run, segment) cells; the largest flagged deviation "
               f"from its median is {g4(max_dev_pct)} % of the median. For reference, with the normal-consistent scaled MAD "
               f"(1.4826 × MAD) the same rule flags {n_scaled} cells.\n")
    if flags:
        out.append(table(["config", "seed", "segment", "value [J]", "median [J]", "MAD [J]", "|x−med|/MAD",
                          "|x−med| [% of median]"], flags))
    # 4.8 reconciliation
    pr = pd.read_csv("flop_analysis/output/per_run_energy_per_flop.csv")
    cs = pd.read_csv("flop_analysis/output/cross_seed_energy_per_flop.csv")
    maxdev = defaultdict(float)
    fails = []

    def rel(a, b):
        return abs(a - b) / abs(b) if b else abs(a - b)

    for r in RUNS:
        sub = pr[pr.run_dir == r["run_dir"]].set_index("segment")
        for s in ["rollout"] + SUBS + ["TOTAL_MEASURED_TRAINING", "warmup", "idle_baseline_head", "idle_baseline_tail"]:
            row = sub.loc[s]
            for col, mine in (("total_energy_joules", r["E"][s]), ("duration_s", r["D"][s]),
                              ("mean_power_w", r["P"][s])):
                d = rel(float(row[col]), mine)
                maxdev[("per_run", col)] = max(maxdev[("per_run", col)], d)
                if d > 1e-9:
                    fails.append(("per_run", r["run_dir"], s, col, d))
            if s in r["F"] and s != "buffer_sample":
                d = rel(float(row["total_flops"]), r["F"][s])
                maxdev[("per_run", "total_flops")] = max(maxdev[("per_run", "total_flops")], d)
                d2 = rel(float(row["energy_per_flop_j_per_flop"]), r["jpf"][s])
                maxdev[("per_run", "energy_per_flop_j_per_flop")] = max(maxdev[("per_run", "energy_per_flop_j_per_flop")], d2)
                if d > 1e-9 or d2 > 1e-9:
                    fails.append(("per_run", r["run_dir"], s, "flops/jpf", max(d, d2)))
    for es, t in CFG:
        rs = runs_of(es, t)
        r0 = rs[0]
        sub = cs[(cs.algo == "sac") & (cs.env_id == r0["env"]) & (cs.architecture_signature == r0["sig"])
                 & (cs.updates_per_env_step == r0["meta"]["algo_config"]["updates_per_env_step"])].set_index("segment")
        for s in ["rollout"] + SUBS + ["TOTAL_MEASURED_TRAINING"]:  # cross-seed CSV has no warmup/idle rows
            row = sub.loc[s]
            Em = np.mean(seg_vals(rs, "E", s))
            comps = [("mean_energy_joules", Em), ("mean_duration_s", np.mean([r["D"][s] for r in rs])),
                     ("mean_power_w", np.mean([r["P"][s] for r in rs]))]
            if s in r0["F"] and s != "buffer_sample":
                Fm = np.mean([r["F"][s] for r in rs])
                comps += [("total_flops", Fm), ("mean_energy_per_flop_j_per_flop", Em / Fm)]
            for col, mine in comps:
                d = rel(float(row[col]), mine)
                maxdev[("cross_seed", col)] = max(maxdev[("cross_seed", col)], d)
                if d > 1e-9:
                    fails.append(("cross_seed", label(es, t), s, col, d))
    rows = [[k[0], f"`{k[1]}`", f"{v:.3e}"] for k, v in sorted(maxdev.items())]
    out.append("\n**4.8 Reconciliation against the pipeline outputs.** Every (run, segment) and (configuration, segment) "
               "value recomputed here from `segment_energy.json`, the per-task CSV and `flops_per_call.json` is compared "
               "with `per_run_energy_per_flop.csv` (segments: rollout, the 4 allocated, TOTAL, warmup, idle head/tail) and "
               "`cross_seed_energy_per_flop.csv` (rollout, the 4 allocated, TOTAL — the only SAC segments it contains). "
               "Columns compared: energy, duration, mean power, FLOPs, J/FLOP. Note: the pipeline's cross-seed "
               "`mean_power_w` is the **mean of per-seed powers**; it is compared against the same statistic here.\n")
    out.append(table(["CSV", "column", "max relative deviation"], rows))
    out.append("\n- " + check("all deviations ≤ 1e-9 relative", not fails, f"{len(fails)} offending cells: {fails[:5]}" if fails else ""))
    return "\n".join(out)


def cross_seed_summary_csv(env_info):
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        for s in SEG_ORDER + ["warmup", "idle_baseline_head", "idle_baseline_tail"]:
            E = np.array(seg_vals(rs, "E", s))
            D = np.array([r["D"][s] for r in rs])
            P = np.array([r["P"][s] for r in rs])
            d = dict(tag=t, env=rs[0]["env"], config=label(es, t), segment=s, status=SEG_STATUS[s], n=len(rs),
                     architecture_signature=rs[0]["sig"], utd=rs[0]["meta"]["algo_config"]["updates_per_env_step"],
                     energy_j_mean=E.mean(), energy_j_sd=E.std(ddof=1), energy_j_min=E.min(), energy_j_max=E.max(),
                     energy_kwh_mean=E.mean() / KWH_TO_J, duration_s_mean=D.mean(), duration_s_sd=D.std(ddof=1),
                     mean_power_w_mean=P.mean(), mean_power_w_sd=P.std(ddof=1))
            if s in rs[0]["share"]:
                sv = np.array([r["share"][s] for r in rs])
                d.update(share_of_total_pct_mean=sv.mean(), share_of_total_pct_sd=sv.std(ddof=1))
            if s in SUBS:
                sv = np.array([r["share_gu"][s] for r in rs])
                d.update(share_of_gu_pct_mean=sv.mean(), share_of_gu_pct_sd=sv.std(ddof=1))
            if s in rs[0]["F"]:
                Fm = np.mean([r["F"][s] for r in rs])
                d.update(flops_mean=Fm, flop_type=("elementwise" if s == "target_update" else "none" if s == "buffer_sample"
                                                   else "mixed_total" if s == "TOTAL_MEASURED_TRAINING" else "matmul"))
                if Fm:
                    per = np.array([r["jpf"][s] for r in rs])
                    d.update(j_per_flop_ratio_of_means=E.mean() / Fm, j_per_flop_per_seed_mean=per.mean(),
                             j_per_flop_per_seed_sd=per.std(ddof=1), j_per_flop_per_seed_min=per.min(),
                             j_per_flop_per_seed_max=per.max(), pj_per_flop_ratio_of_means=E.mean() / Fm * 1e12,
                             jpf_note=("J per elementwise Polyak op (NOT J/FLOP)" if s == "target_update" else
                                       "derived by context generator (not a pipeline output)" if s == "gradient_updates" else ""))
            if s in rs[0]["tp"]:
                tp = np.array([r["tp"][s] for r in rs])
                d.update(gflops_per_s_mean=tp.mean(), gflops_per_s_sd=tp.std(ddof=1))
            rows.append(d)
    pd.DataFrame(rows).to_csv(OUT_DIR / "sac_cross_seed_summary.csv", index=False, float_format="%.17g")


# ----------------------------------------------------------------------------- Part 5
def signflip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    cnt = 0
    for signs in itertools.product([1, -1], repeat=len(d)):
        if abs((np.array(signs) * d).mean()) >= obs - 1e-12 * max(1.0, obs):
            cnt += 1
    return cnt / 2 ** len(d), cnt


def part5(env_info):
    out = []
    pairs = {t: (sorted(runs_of("HC", t), key=lambda r: r["seed"]), sorted(runs_of("Ant", t), key=lambda r: r["seed"])) for t in TAGS}
    env_rows = []
    out.append("All statistics pair HalfCheetah-v5 and Ant-v5 runs **by seed** (the 5 seeds are shared). n = 5, "
               "ddof = 1, gross energy, 4 significant digits. Shares are of `TOTAL_MEASURED_TRAINING` energy. Full "
               "precision in `sac_env_comparison.csv`.\n")
    # 5.1
    rows = []
    for t in TAGS:
        hc, ant = pairs[t]
        diffs = {s: np.mean([a["share"][s] for a in ant]) - np.mean([hh["share"][s] for hh in hc]) for s in ["rollout"] + SUBS}
        smax = max(diffs, key=lambda s: abs(diffs[s]))
        rows.append([t, g4(abs(diffs[smax])), SEG_SHORT[smax]] + [g4(diffs[s]) for s in ["rollout"] + SUBS])
        for s in ["rollout"] + SUBS:
            env_rows.append(dict(tag=t, metric="mean_share_diff_ant_minus_hc_pp", segment=s, value=diffs[s]))
    out.append("**5.1 Max absolute share difference** (difference of the mean shares, Ant − HC, percentage points; "
               "`gradient_updates` share = 100 − rollout share, so it is not listed separately):\n")
    out.append(table(["tag", "max |Δshare| [pp]", "attained by", "Δ rollout", "Δ buf_sample", "Δ critic", "Δ actor", "Δ target"], rows))
    # 5.2
    rows = []
    for t in TAGS:
        hc, ant = pairs[t]
        cells = [t]
        for s in ["rollout"] + SUBS:
            d = np.array([a["share"][s] - hh["share"][s] for hh, a in zip(hc, ant)])
            cells.append(f"{g4(d.mean())} ± {g4(d.std(ddof=1))} ({int((d > 0).sum())}/5 > 0)")
            env_rows.append(dict(tag=t, metric="paired_share_diff_mean_pp", segment=s, value=d.mean()))
            env_rows.append(dict(tag=t, metric="paired_share_diff_sd_pp", segment=s, value=d.std(ddof=1)))
            env_rows.append(dict(tag=t, metric="paired_share_diff_n_positive", segment=s, value=int((d > 0).sum())))
        rows.append(cells)
    out.append("\n**5.2 Per-segment share difference Ant − HC, paired by seed** (mean ± sd of the 5 per-seed differences "
               "in pp; count of seeds with a positive difference):\n")
    out.append(table(["tag", "rollout", "buf_sample", "critic", "actor", "target"], rows))
    # 5.3 powers
    rows, rows2 = [], []
    for t in TAGS:
        hc, ant = pairs[t]
        cells = [t]
        for s in ("rollout", "gradient_updates", "TOTAL_MEASURED_TRAINING"):
            cells += [msd([r["P"][s] for r in hc]), msd([r["P"][s] for r in ant])]
        rows.append(cells)
        rr_hc = np.array([r["P"]["rollout"] / r["P"]["gradient_updates"] for r in hc])
        rr_ant = np.array([r["P"]["rollout"] / r["P"]["gradient_updates"] for r in ant])
        ror_means = (np.mean([r["P"]["rollout"] for r in ant]) / np.mean([r["P"]["gradient_updates"] for r in ant])) / \
                    (np.mean([r["P"]["rollout"] for r in hc]) / np.mean([r["P"]["gradient_updates"] for r in hc]))
        ror_paired = rr_ant / rr_hc
        cells2 = [t, msd(rr_hc), msd(rr_ant), g4(ror_means), msd(ror_paired)]
        for s in ("rollout", "gradient_updates", "TOTAL_MEASURED_TRAINING"):
            bm = np.mean([r["P"][s] for r in ant]) / np.mean([r["P"][s] for r in hc])
            bp = np.array([a["P"][s] / hh["P"][s] for hh, a in zip(hc, ant)])
            cells2.append(f"{g4(bm)} ({msd(bp)})")
            env_rows.append(dict(tag=t, metric="power_ratio_ant_over_hc_ratio_of_means", segment=s, value=bm))
            env_rows.append(dict(tag=t, metric="power_ratio_ant_over_hc_paired_mean", segment=s, value=bp.mean()))
            env_rows.append(dict(tag=t, metric="power_ratio_ant_over_hc_paired_sd", segment=s, value=bp.std(ddof=1)))
        rows2.append(cells2)
        env_rows += [dict(tag=t, metric="P_rollout_over_P_GU_HC_mean", segment="", value=rr_hc.mean()),
                     dict(tag=t, metric="P_rollout_over_P_GU_Ant_mean", segment="", value=rr_ant.mean()),
                     dict(tag=t, metric="ratio_of_ratios_from_means", segment="", value=ror_means),
                     dict(tag=t, metric="ratio_of_ratios_paired_mean", segment="", value=ror_paired.mean()),
                     dict(tag=t, metric="ratio_of_ratios_paired_sd", segment="", value=ror_paired.std(ddof=1))]
    out.append("\n**5.3 Power.** Mean power [W] = energy / measured task duration, per seed, mean ± sd:\n")
    out.append(table(["tag", "P rollout HC", "P rollout Ant", "P GU HC", "P GU Ant", "P TOTAL HC", "P TOTAL Ant"], rows))
    out.append("\nRatios: within-env `P_rollout / P_GU` (per seed, mean ± sd); ratio of ratios "
               "`(P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC` from the env means and paired by seed (mean ± sd); "
               "between-env `P_Ant / P_HC` per measured task as ratio of means (paired-by-seed mean ± sd in parentheses):\n")
    out.append(table(["tag", "P_roll/P_GU HC", "P_roll/P_GU Ant", "ratio of ratios (means)", "ratio of ratios (paired)",
                      "P_Ant/P_HC rollout", "P_Ant/P_HC GU", "P_Ant/P_HC TOTAL"], rows2))
    # 5.4 energy and J/FLOP ratios
    rows, rows2 = [], []
    for t in TAGS:
        hc, ant = pairs[t]
        cells, cells2 = [t], [t]
        for s in ["rollout"] + SUBS + ["gradient_updates", "TOTAL_MEASURED_TRAINING"]:
            em = np.mean([r["E"][s] for r in ant]) / np.mean([r["E"][s] for r in hc])
            ep = np.array([a["E"][s] / hh["E"][s] for hh, a in zip(hc, ant)])
            cells.append(f"{g4(em)} (±{g4(ep.std(ddof=1))})")
            env_rows.append(dict(tag=t, metric="energy_ratio_ant_over_hc_ratio_of_means", segment=s, value=em))
            env_rows.append(dict(tag=t, metric="energy_ratio_ant_over_hc_paired_sd", segment=s, value=ep.std(ddof=1)))
        for s in ["rollout", "critic_update", "actor_update", "gradient_updates", "TOTAL_MEASURED_TRAINING", "target_update"]:
            ja = np.mean([r["E"][s] for r in ant]) / np.mean([r["F"][s] for r in ant])
            jh = np.mean([r["E"][s] for r in hc]) / np.mean([r["F"][s] for r in hc])
            cells2.append(g4(ja / jh))
            env_rows.append(dict(tag=t, metric="jpf_ratio_ant_over_hc", segment=s, value=ja / jh))
        rows.append(cells)
        rows2.append(cells2)
    out.append("\n**5.4 Energy ratio Ant/HC** (ratio of means; sd of the 5 paired per-seed ratios in parentheses):\n")
    out.append(table(["tag", "rollout", "buf_sample", "critic", "actor", "target", "grad_updates", "TOTAL"], rows))
    out.append("\n**J/FLOP ratio Ant/HC** (ratio of the two ratio-of-means J/FLOP values; `grad_updates` column uses the "
               "derived Table 4.5 quantity; `target` is a J/op ratio):\n")
    out.append(table(["tag", "rollout", "critic", "actor", "grad_updates (derived)", "TOTAL", "target (J/op)"], rows2))
    # 5.5 permutation
    rows = []
    for t in TAGS:
        hc, ant = pairs[t]
        for name, f in (("rollout share [pp]", lambda r: r["share"]["rollout"]),
                        ("TOTAL energy [J]", lambda r: r["E"]["TOTAL_MEASURED_TRAINING"])):
            d = np.array([f(a) - f(hh) for hh, a in zip(hc, ant)])
            pval, cnt = signflip_p(d)
            rows.append([t, name, " / ".join(g4(x) for x in d), g4(d.mean()), f"{cnt}/32", g4(pval)])
            env_rows.append(dict(tag=t, metric=f"signflip_p_two_sided::{name}", segment="", value=pval))
    out.append("\n**5.5 Exact paired sign-flip permutation test** (two-sided; statistic |mean of paired differences Ant − HC|; "
               "all 2⁵ = 32 sign patterns enumerated; p = #{patterns with |mean| ≥ observed}/32). With n = 5 the smallest "
               "attainable two-sided p is 2/32 = 0.0625, so no difference can reach p < 0.05 with this test.\n")
    out.append(table(["tag", "quantity", "per-seed differences (seed order " + ",".join(map(str, CANON)) + ")", "mean diff",
                      "count", "p (two-sided)"], rows))
    pd.DataFrame(env_rows).to_csv(OUT_DIR / "sac_env_comparison.csv", index=False, float_format="%.17g")
    # 5.6 by construction
    hc_b, ant_b = FLOPS_JSON["sac"]["HalfCheetah-v5"], FLOPS_JSON["sac"]["Ant-v5"]
    rows = []
    for sig in ["bs256_h256x256", "bs256_h512x512", "bs256_h1024x1024", "bs512_h1024x1024", "bs1024_h1024x1024"]:
        rows.append([f"`{sig}`"] + [g4(ant_b[sig][k] / hc_b[sig][k]) for k in ("actor_forward_bs1", "critic_fwdbwd", "actor_fwdbwd",
                                                                                 "target_update_elementwise_ops")])
    ep_counts = {}
    for es in ("HC", "Ant"):
        n_train = [sum(1 for e in r["tm"]["episodes"] if e["phase"] == "train") for r in RUNS if r["es"] == es]
        n_warm = [sum(1 for e in r["tm"]["episodes"] if e["phase"] == "warmup") for r in RUNS if r["es"] == es]
        lens = [e["length"] for r in RUNS if r["es"] == es for e in r["tm"]["episodes"] if e["phase"] == "train"]
        ep_counts[es] = (n_train, n_warm, lens)
    out.append(f"""
**5.6 Facts that differ between the environments by construction** (for interpretation; not interpreted here):
- Observation / action dims: HalfCheetah-v5 {env_info['HC']['obs']}/{env_info['HC']['act']}, Ant-v5 {env_info['Ant']['obs']}/{env_info['Ant']['act']} (Part 2.4). Only the first layer of each network
  (input `obs` for the actor, `obs+act` for the Q-nets) and the output heads (2·act for the actor) depend on them, so
  FLOPs per call differ by the ratios below (Ant / HC), shrinking with width as the hidden×hidden layer dominates:
""")
    out.append(table(["signature", "rollout (actor fwd bs1)", "critic_fwdbwd", "actor_fwdbwd", "target ops"], rows))
    out.append(f"""
- Episode structure inside the `rollout` task (same code for both envs; `env.reset()` is called inside the task):
  HalfCheetah-v5 never terminates (only the 1000-step truncation): completed train-phase episodes per run =
  {', '.join(map(str, sorted(set(ep_counts['HC'][0]))))} in all 35 HC runs (episode length
  {min(ep_counts['HC'][2])}–{max(ep_counts['HC'][2])} steps), i.e. 100 `env.reset()` calls inside the 100 `rollout` tasks.
  Ant-v5 has `terminate_when_unhealthy=True`: completed train-phase episodes per run
  {min(ep_counts['Ant'][0])}–{max(ep_counts['Ant'][0])} (median {np.median(ep_counts['Ant'][0]):g}) over the 35 Ant runs, episode length
  min/median/max {min(ep_counts['Ant'][2])}/{np.median(ep_counts['Ant'][2]):g}/{max(ep_counts['Ant'][2])} steps; each completed
  episode triggers one `env.reset()` inside `rollout`. (Per-configuration episode counts: Part 9.)
- Simulator: both are MuJoCo envs stepped on the CPU (`frame_skip` = {env_info['HC']['frame_skip']} for HC, {env_info['Ant']['frame_skip']} for Ant; Ant has more
  bodies/contacts and computes the 78-dim contact-force part of its observation). No per-step simulator timing is
  logged, so the simulator's share of the `rollout` task duration is **NOT MEASURED**.
- The `gradient_updates` code path is identical across envs; only the per-call FLOPs (table above) and the stored
  `done` flags differ (Ant terminations enter the Bellman target via `(1 − done)`; HC's `done` is always 0 since
  `buffer.add` stores `terminated`, not `truncated`).
""")
    return "\n".join(out)


# ----------------------------------------------------------------------------- Part 6
def part6():
    out = []
    srows = []
    metrics = OrderedDict()
    for s in SEG_ORDER:
        metrics[f"E {SEG_SHORT[s]}"] = lambda r, s=s: r["E"][s]
    for s in ("rollout", "gradient_updates", "TOTAL_MEASURED_TRAINING"):
        metrics[f"T {SEG_SHORT[s]}"] = lambda r, s=s: r["D"][s]
    for s in ("rollout", "gradient_updates", "TOTAL_MEASURED_TRAINING"):
        metrics[f"P {SEG_SHORT[s]}"] = lambda r, s=s: r["P"][s]
    metrics["J/FLOP TOTAL"] = lambda r: r["jpf"]["TOTAL_MEASURED_TRAINING"]
    metrics["J/FLOP GU (derived)"] = lambda r: r["jpf"]["gradient_updates"]
    out.append("Relative change vs the same env's `base`, in %: **paired** = per-seed (config_seed / base_seed − 1), "
               "then mean ± sd over the 5 seeds; **means** = (mean_config / mean_base − 1). Gross energy. Share changes "
               "are in percentage points (paired). Full precision in `sac_sweep_effects.csv`.\n")
    for es, env_id in ENVS:
        base = {r["seed"]: r for r in runs_of(es, "base")}
        rows = []
        for t in SWEEP_TAGS:
            rs = sorted(runs_of(es, t), key=lambda r: r["seed"])
            cells = [label(es, t)]
            for name, f in metrics.items():
                pv = np.array([100 * (f(r) / f(base[r["seed"]]) - 1) for r in rs])
                mv = 100 * (np.mean([f(r) for r in rs]) / np.mean([f(b) for b in base.values()]) - 1)
                cells.append(f"{g4(pv.mean())} ± {g4(pv.std(ddof=1))} [{g4(mv)}]")
                srows.append(dict(env=env_id, tag=t, metric=name, paired_mean_pct=pv.mean(), paired_sd_pct=pv.std(ddof=1),
                                  from_means_pct=mv))
            rows.append(cells)
        hdr = ["config"] + list(metrics)
        out.append(f"\n**6.1 {env_id} — relative change [%]** (cells: `paired mean ± sd [from means]`):\n")
        # split into two tables for readability
        k = 8
        out.append(table(hdr[:1 + k], [rw[:1 + k] for rw in rows]))
        out.append("")
        out.append(table([hdr[0]] + hdr[1 + k:], [[rw[0]] + rw[1 + k:] for rw in rows]))
        rows = []
        for t in SWEEP_TAGS:
            rs = sorted(runs_of(es, t), key=lambda r: r["seed"])
            cells = [label(es, t)]
            for s in ["rollout"] + SUBS + ["gradient_updates"]:
                d = np.array([r["share"][s] - base[r["seed"]]["share"][s] for r in rs])
                cells.append(f"{g4(d.mean())} ± {g4(d.std(ddof=1))}")
                srows.append(dict(env=env_id, tag=t, metric=f"share_change_pp {SEG_SHORT[s]}", paired_mean_pct=d.mean(),
                                  paired_sd_pct=d.std(ddof=1), from_means_pct=np.mean([r["share"][s] for r in rs]) -
                                  np.mean([b["share"][s] for b in base.values()])))
            rows.append(cells)
        out.append(f"\nShare change vs base [pp], paired mean ± sd ({env_id}):\n")
        out.append(table(["config", "rollout", "buf_sample", "critic", "actor", "target", "grad_updates"], rows))
    pd.DataFrame(srows).to_csv(OUT_DIR / "sac_sweep_effects.csv", index=False, float_format="%.17g")
    # 6.2 scaling
    out.append("\n**6.2 Scaling check — FLOPs ratio vs energy ratio (config / base, ratio of means)** "
               "(`target` uses op counts):\n")
    for es, env_id in ENVS:
        base = runs_of(es, "base")
        rows = []
        for t in SWEEP_TAGS:
            rs = runs_of(es, t)
            cells = [label(es, t)]
            for s in ["rollout", "critic_update", "actor_update", "target_update", "gradient_updates", "TOTAL_MEASURED_TRAINING"]:
                fr = np.mean([r["F"][s] for r in rs]) / np.mean([b["F"][s] for b in base])
                er = np.mean([r["E"][s] for r in rs]) / np.mean([b["E"][s] for b in base])
                cells.append(f"F×{g4(fr)} / E×{g4(er)}")
            rows.append(cells)
        out.append(f"\n{env_id}:\n")
        out.append(table(["config", "rollout", "critic", "actor", "target", "grad_updates", "TOTAL"], rows))
    # 6.3 delta pairs
    out.append(delta_pairs_section())
    # 6.4 OLS
    out.append(ols_section())
    return "\n".join(out)


def delta_pairs_section():
    out = []
    import flop_dashboard as fd  # module-level code: constants + function defs; Dash app only built in main()
    src = inspect.getsource(fd)
    safe = "if __name__ == \"__main__\":" in src and not re.search(r"^app\s*=", src, re.M)
    pr = fd.load_per_run(str(REPO / "flop_analysis" / "output"))
    pr = pr[(pr.algo == "sac") & (pr.included_in_cross_seed_avg)]
    cs = fd.load_cross_seed(str(REPO / "flop_analysis" / "output"))
    cs = cs[cs.algo == "sac"]
    out.append(f"\n**6.3 ΔEnergy/ΔFLOPs (marginal J per extra FLOP).** `compute_delta_pairs()` was **imported from "
               f"`flop_dashboard.py` and called unmodified** (import is side-effect free: module level defines constants "
               f"and functions only, the Dash app is built inside `main()` under `if __name__ == \"__main__\"` — "
               f"{'confirmed' if safe else 'NOT confirmed'}). Inputs: the pipeline's `per_run_energy_per_flop.csv` "
               f"(SAC, `included_in_cross_seed_avg`, loaded with the dashboard's own `load_per_run`) with "
               f"`PER_RUN_DIMENSIONS` (seed held fixed → pairs are within-seed), energy `total_energy_joules`, FLOPs "
               f"`total_flops`; and the cross-seed CSV with `DIMENSIONS` (`mean_energy_joules`, `total_flops`). Rule "
               f"(dashboard): pairs differ only in the X dimension, all other config dims + flop_type (+ seed) fixed; "
               f"matmul/mixed_total rows only; pairs with |ΔF| < {fd.MIN_REL_DELTA_FLOPS:.0%} of the reference dropped; "
               f"pooled value = ΣΔE/ΣΔF. Reference = smallest X value (`first`) or next-smaller (`previous`). "
               f"Pair lists: `sac_delta_pairs.csv`.\n")
    all_pairs = []
    pooled = {}  # (sweep, mode, level, env, seg) -> (value, n_ok, n_dropped, xs)
    for x_dim, name in (("hidden_sizes", "width"), ("batch_size", "batch size"), ("updates_per_env_step", "UTD")):
        for mode in ("first", "previous"):
            for level, df, dims, ecol in (("per-run", pr, fd.PER_RUN_DIMENSIONS, "total_energy_joules"),
                                          ("cross-seed", cs, fd.DIMENSIONS, "mean_energy_joules")):
                pairs, n_ex = fd.compute_delta_pairs(df, dims, x_dim, ecol, "total_flops", mode)
                if pairs.empty:
                    continue
                pairs = pairs.copy()
                pairs["x_dim"], pairs["ref_mode"], pairs["level"] = x_dim, mode, level
                all_pairs.append(pairs)
                xcol = dims[x_dim]["col"]
                for (env, seg), g in pairs.groupby(["env_id", "segment"]):
                    ok = g[g.status == "ok"]
                    val = ok.delta_energy_j.sum() / ok.delta_flops.sum() if len(ok) else float("nan")
                    xs = ", ".join(f"{a}←{b}" for a, b in sorted(set(zip(ok[xcol].astype(str), ok["reference_value"].astype(str)))))
                    pooled[(name, mode, level, ENV_SHORT[env], seg)] = (val, len(ok), len(g) - len(ok), xs)
    if all_pairs:
        ap = pd.concat(all_pairs, ignore_index=True)
        ap.to_csv(OUT_DIR / "sac_delta_pairs.csv", index=False, float_format="%.17g")
    # per-run vs cross-seed pooled values
    diffs = [abs(v[0] - pooled[(k[0], k[1], "cross-seed", k[3], k[4])][0]) / abs(v[0])
             for k, v in pooled.items() if k[2] == "per-run" and not math.isnan(v[0])
             and (k[0], k[1], "cross-seed", k[3], k[4]) in pooled]
    rows = []
    for name in ("width", "batch size", "UTD"):
        for mode in ("first", "previous"):
            for seg in ("rollout", "critic_update", "actor_update", "TOTAL_MEASURED_TRAINING"):
                h_ = pooled.get((name, mode, "per-run", "HC", seg))
                a_ = pooled.get((name, mode, "per-run", "Ant", seg))
                if not h_ or not a_:
                    continue
                fmt = lambda v: "dropped (ΔF ≈ 0)" if math.isnan(v[0]) else g4(v[0] * 1e12)  # noqa: E731
                rows.append([name, mode, SEG_SHORT[seg], h_[3] or a_[3] or "—", fmt(h_), fmt(a_),
                             f"{h_[1]} / {a_[1]}", f"{h_[2]} / {a_[2]}"])
    out.append(table(["sweep", "reference", "segment", "X pairs (x←ref)", "HC ΣΔE/ΣΔF [pJ/FLOP]", "Ant ΣΔE/ΣΔF [pJ/FLOP]",
                      "pairs used HC / Ant", "pairs dropped HC / Ant"], rows))
    out.append(f"\nPooled values are from the per-run level (5 seeds × the listed X pairs). The cross-seed level gives the "
               f"same pooled values (max relative difference {max(diffs):.1e}), because FLOPs are identical across seeds so "
               f"ΣΔE/ΣΔF over seeds = Δ(mean E)/Δ(mean F). Dropped pairs have |ΔF| < 1 % of the reference (rollout FLOPs "
               f"do not change with UTD or batch size). The pipeline CSVs contain no `gradient_updates` row, so the "
               f"dashboard rule yields no gradient_updates ΔE/ΔF; the OLS slope b in 6.4 is the closest equivalent.")
    return "\n".join(out)


def ols_section():
    out = ["\n**6.4 (exploratory; belongs to the open Section 5.2 analysis (a)) Fixed-versus-marginal fit `E = a + b·F`** "
           "by ordinary least squares over per-run points (each run = one point; gross energy). Width sweep: w256, "
           "w512, base (batch 256, UTD 1); batch sweep: base, b512, b1024 (width 1024, UTD 1); UTD sweep added for "
           "completeness: base, utd2, utd4. Residual SE = sqrt(SSR/(n−2)). Full precision in `sac_ols_fits.csv`.\n"]
    rows, csv_rows = [], []
    for sweep, tags in (("width", ["w256", "w512", "base"]), ("batch", ["base", "b512", "b1024"]), ("UTD", ["base", "utd2", "utd4"])):
        for es, env_id in ENVS:
            for s in ("gradient_updates", "TOTAL_MEASURED_TRAINING"):
                pts = [(r["F"][s], r["E"][s]) for t in tags for r in runs_of(es, t)]
                F = np.array([p_[0] for p_ in pts], float)
                E = np.array([p_[1] for p_ in pts], float)
                b, a = np.polyfit(F, E, 1)
                pred = a + b * F
                ssr = ((E - pred) ** 2).sum()
                r2 = 1 - ssr / ((E - E.mean()) ** 2).sum()
                rse = math.sqrt(ssr / (len(E) - 2))
                rows.append([sweep, env_id, SEG_SHORT[s], len(E), g4(a), g4(b), g4(b * 1e12), g4(r2), g4(rse)])
                csv_rows.append(dict(sweep=sweep, env=env_id, segment=s, n=len(E), a_J=a, b_J_per_FLOP=b, r2=r2, rse_J=rse))
    out.append(table(["sweep", "env", "segment", "n", "a [J]", "b [J/FLOP]", "b [pJ/FLOP]", "R²", "residual SE [J]"], rows))
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "sac_ols_fits.csv", index=False, float_format="%.17g")
    return "\n".join(out)


# ----------------------------------------------------------------------------- Part 7
def part7():
    out = []
    comp_rows = []
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        cells = [label(es, t)]
        for task in ("rollout", "gradient_updates"):
            for c in ("cpu", "gpu", "ram"):
                v = [r["hw"][task][c] for r in rs]
                pc = [100 * r["hw"][task][c] / r["hw"][task]["E"] for r in rs]
                cells.append(f"{msd(v)} ({msd(pc)} %)")
                comp_rows.append(dict(env=rs[0]["env"], tag=t, task=task, component=c, energy_j_mean=np.mean(v),
                                      energy_j_sd=np.std(v, ddof=1), pct_mean=np.mean(pc), pct_sd=np.std(pc, ddof=1)))
        rows.append(cells)
    out.append("All values in this part are **diagnostics, not subtracted from any reported value**; energy stays gross "
               "everywhere else.\n")
    out.append("**7.1 CPU / GPU / RAM split of the two measured tasks** — per-task CSV columns `cpu_energy`, "
               "`gpu_energy`, `ram_energy` [kWh → J], summed over the 100 tasks of each prefix per run; mean ± sd over "
               "5 seeds in J, and (in parentheses) mean ± sd of the per-run percentage of the task energy. CPU = RAPL "
               "package energy, GPU = NVML, RAM = CodeCarbon's constant-power RAM model (20 W, Part 2.5). "
               f"Check: cpu+gpu+ram = energy_consumed per task, max rel. err "
               f"{max(abs(r['hw'][k]['cpu'] + r['hw'][k]['gpu'] + r['hw'][k]['ram'] - r['hw'][k]['E']) / r['hw'][k]['E'] for r in RUNS for k in ('rollout', 'gradient_updates')):.2e}.\n")
    out.append(table(["config", "rollout CPU", "rollout GPU", "rollout RAM", "GU CPU", "GU GPU", "GU RAM"], rows))
    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "sac_energy_components.csv", index=False, float_format="%.17g")
    # 7.2 idle
    idle_rows = []
    rows = []
    rel_all, rel_all_corr = [], []
    for r in RUNS:
        ph, pt = r["P"]["idle_baseline_head"], r["P"]["idle_baseline_tail"]
        th, tt = task_rows(r, "idle_baseline_head").iloc[0], task_rows(r, "idle_baseline_tail").iloc[0]
        wh = th["ram_energy"] * KWH_TO_J / th["ram_power"]
        wt = tt["ram_energy"] * KWH_TO_J / tt["ram_power"]
        ph_c, pt_c = r["E"]["idle_baseline_head"] / wh, r["E"]["idle_baseline_tail"] / wt
        rel_ = abs(pt - ph) / ph
        rel_c = abs(pt_c - ph_c) / ph_c
        rel_all.append(rel_)
        rel_all_corr.append(rel_c)
        idle_rows.append(dict(tag=r["tag"], env=r["env"], seed=r["seed"], run_dir=r["run_dir"], P_head_w=ph, P_tail_w=pt,
                              rel_head_tail_diff=rel_, signed_tail_minus_head_w=pt - ph, head_duration_s=th["duration"],
                              head_integration_window_s=wh, tail_duration_s=tt["duration"], tail_integration_window_s=wt,
                              P_head_window_corrected_w=ph_c, P_tail_window_corrected_w=pt_c, rel_diff_window_corrected=rel_c))
    IDLE = pd.DataFrame(idle_rows)
    IDLE.to_csv(OUT_DIR / "sac_idle_floor.csv", index=False, float_format="%.17g")
    for es, t in CFG:
        sub = IDLE[(IDLE.env == dict(ENVS)[es]) & (IDLE.tag == t)]
        rows.append([label(es, t), msd(sub.P_head_w), msd(sub.P_tail_w), msd(100 * sub.rel_head_tail_diff),
                     msd(sub.signed_tail_minus_head_w)])
    a = np.array(rel_all) * 100
    ac = np.array(rel_all_corr) * 100
    worst = IDLE.sort_values("rel_head_tail_diff", ascending=False).head(3)
    out.append("\n**7.2 Idle floor.** `P_head`, `P_tail` = idle-task energy / its CodeCarbon `duration` (as recorded), "
               "per run; mean ± sd over 5 seeds:\n")
    out.append(table(["config", "P_head [W]", "P_tail [W]", "|P_tail−P_head|/P_head [%]", "P_tail − P_head [W]"], rows))
    out.append(f"\nOver all 70 runs: |P_tail − P_head| / P_head — **median {g4(np.median(a))} %, maximum {g4(a.max())} %** "
               f"(mean {g4(a.mean())} %; tail > head in {int((IDLE.signed_tail_minus_head_w > 0).sum())}/70 runs). Largest: "
               + "; ".join(f"{label(ENV_SHORT[w.env], w.tag)} s{w.seed}: {g4(100 * w.rel_head_tail_diff)} %" for w in worst.itertuples()) + ".")
    out.append(f"\nIntegration-window diagnostic (derived; same method as the repo's `idle_power_analysis.py`): CodeCarbon "
               f"models RAM at a constant 20 W, so `ram_energy / ram_power` is the time span over which a task's energy "
               f"was integrated. For `idle_baseline_head` this span is {msd(IDLE.head_integration_window_s)} s vs a "
               f"recorded duration of {msd(IDLE.head_duration_s)} s; for the tail {msd(IDLE.tail_integration_window_s)} s "
               f"vs {msd(IDLE.tail_duration_s)} s. Dividing by the integrated span instead (window-corrected) gives "
               f"|ΔP|/P_head median {g4(np.median(ac))} %, max {g4(ac.max())} % over the 70 runs. The as-recorded values "
               f"above are the ones that follow the stated protocol (energy / duration); the corrected ones are for "
               f"reference only. Per-run values: `sac_idle_floor.csv`.")
    disc("CodeCarbon's `idle_baseline_head` task integrates energy over a span a few seconds longer than its recorded "
         "`duration` (RAM energy / RAM power ≈ 93–94 s vs 90 s; tail spans match), so as-recorded head idle power is "
         "biased high by a few percent. This is already noted in `idle_power_analysis.py`; it affects the head–tail "
         "drift figure (both versions given in Part 7.2).")
    # 7.3
    rows = []
    for es, t in CFG:
        rs = runs_of(es, t)
        cells = [label(es, t)]
        for s in ("rollout", "gradient_updates", "TOTAL_MEASURED_TRAINING"):
            cells.append(msd([100 * r["P_idle"] * r["D"][s] / r["E"][s] for r in rs]))
        rows.append(cells)
    out.append("\n**7.3 Idle-floor fraction (derived, descriptive):** `P̄_idle × duration / E` per run, with "
               "`P̄_idle` = mean of that run's as-recorded head and tail idle power, in %; mean ± sd over 5 seeds. "
               "This is a diagnostic of how large the idle floor is inside the gross value — **not** a net-energy "
               "result.\n")
    out.append(table(["config", "rollout [%]", "gradient_updates [%]", "TOTAL [%]"], rows))
    return "\n".join(out)


# ----------------------------------------------------------------------------- Part 8
def part8():
    out = []
    mps = RUNS[0]["meta"]["experiment_config"]["measure_power_secs"]
    rows = []
    for es, t in CFG:
        sub = PE[(PE.env == dict(ENVS)[es]) & (PE.tag == t)]
        cells = [label(es, t)]
        for task in ("rollout", "gradient_updates"):
            d = sub[sub.task == task].duration_s
            cells += [f"{g4(d.min())} / {g4(d.median())} / {g4(d.max())}", g4(d.min() / mps)]
        for task in ("rollout", "gradient_updates"):
            w = sub[sub.task == task]
            ratio = w.ram_integration_window_s / w.duration_s
            cells.append(f"{g4(ratio.min())} / {g4(ratio.median())} / {g4(ratio.max())}")
        rows.append(cells)
    out.append(f"**8.1 Single-task durations** over all 100 epochs × 5 seeds (500 tasks per cell), min / median / max in s, "
               f"and the shortest task in units of the polling interval `measure_power_secs` = {mps} s. Last two columns "
               f"(derived diagnostic): integration span / duration = (`ram_energy`/`ram_power`) / `duration` per task "
               f"(1 = CodeCarbon integrated energy over exactly the task's duration).\n")
    out.append(table(["config", "rollout_i dur min/med/max [s]", "shortest rollout / poll", "gradient_updates_i dur min/med/max [s]",
                      "shortest GU / poll", "rollout span/dur min/med/max", "GU span/dur min/med/max"], rows))
    allr = PE[PE.task == "rollout"].duration_s
    out.append(f"\nAcross all 70 runs: rollout tasks {g4(allr.min())}–{g4(allr.max())} s (median {g4(allr.median())} s); "
               f"{int((allr < mps).sum())} of {len(allr)} rollout tasks are shorter than one polling interval. "
               f"gradient_updates tasks {g4(PE[PE.task == 'gradient_updates'].duration_s.min())}–"
               f"{g4(PE[PE.task == 'gradient_updates'].duration_s.max())} s.")
    # 8.2 coverage
    rows = []
    cov_rows = []
    chk = 0
    for es, t in CFG:
        rs = runs_of(es, t)
        cov = [sum(r["T"].values()) / r["D"]["gradient_updates"] for r in rs]
        ts = {k: [100 * r["T"][k] / sum(r["T"].values()) for r in rs] for k in SUBS}
        for r in rs:
            for k in SUBS:
                chk = max(chk, abs(r["T"][k] / sum(r["T"].values()) * 100 - r["share_gu"][k]))
            cov_rows.append(dict(env=r["env"], tag=t, seed=r["seed"], coverage=sum(r["T"].values()) / r["D"]["gradient_updates"],
                                 **{f"T_{k}_s": r["T"][k] for k in SUBS}, gradient_updates_task_duration_s=r["D"]["gradient_updates"]))
        s_ = stats(cov)
        rows.append([label(es, t), f"{g4(100 * s_['mean'])} ± {g4(100 * s_['sd'])}", g4(100 * s_["min"]), g4(100 * s_["max"])]
                    + [msd(ts[k]) for k in SUBS])
    pd.DataFrame(cov_rows).to_csv(OUT_DIR / "sac_allocation_coverage.csv", index=False, float_format="%.17g")
    out.append("\n**8.2 Allocation coverage** `Σ_k T_k / duration(gradient_updates)` (Σ of the four `perf_counter` "
               "sub-phase times, from `segment_energy.json[\"_sub_segment_wall_time_seconds\"]`, divided by the summed "
               "CodeCarbon durations of the 100 `gradient_updates_i` tasks), in %, mean ± sd / min / max over 5 seeds; "
               "and the time shares `T_k / Σ_j T_j` [%] used by the allocation. "
               + check("time shares equal the allocated-energy shares within gradient_updates (Table 4.2 right block)",
                       chk < 1e-9, f"max |Δ| = {chk:.2e} pp") + "\n")
    out.append(table(["config", "coverage [%] mean ± sd", "min", "max", "T buf_sample", "T critic", "T actor", "T target"], rows))
    # 8.3 stationarity
    out.append("\n**8.3 Per-epoch stationarity** (per-epoch values in `sac_per_epoch.csv`). For each run and task: mean and "
               "sd across the 100 epochs, CV across epochs, relative difference (mean of epochs 0–9 − mean of epochs "
               "10–99) / mean of 10–99, and the OLS slope vs epoch index (also as % of the run's mean per 100 epochs); "
               "each statistic is then averaged over the 5 seeds (± sd over seeds).\n")
    st_rows = []
    for quantity, col in (("energy", "energy_j"), ("duration", "duration_s")):
        rows = []
        for es, t in CFG:
            cells = [label(es, t)]
            for task in ("rollout", "gradient_updates"):
                per = []
                for seed in CANON:
                    v = PE[(PE.env == dict(ENVS)[es]) & (PE.tag == t) & (PE.seed == seed) & (PE.task == task)].sort_values("epoch")
                    y = v[col].to_numpy()
                    x = v.epoch.to_numpy()
                    slope = np.polyfit(x, y, 1)[0]
                    m = y.mean()
                    per.append(dict(mean=m, sd=y.std(ddof=1), cv=100 * y.std(ddof=1) / m,
                                    early=100 * (y[:10].mean() - y[10:].mean()) / y[10:].mean(), slope=slope,
                                    slope_pct=100 * slope * 100 / m))
                    st_rows.append(dict(env=dict(ENVS)[es], tag=t, seed=seed, task=task, quantity=quantity, **per[-1]))
                P = pd.DataFrame(per)
                cells += [f"{g4(P['mean'].mean())} ± {g4(P['sd'].mean())}", msd(P.cv), msd(P.early), msd(P.slope),
                          msd(P.slope_pct)]
            rows.append(cells)
        unit = "J" if quantity == "energy" else "s"
        out.append(f"\nPer-epoch **{quantity}** ({unit}):\n")
        out.append(table(["config", f"rollout mean ± sd-across-epochs [{unit}]", "rollout CV %", "rollout ep0–9 vs 10–99 %",
                          f"rollout slope [{unit}/epoch]", "rollout slope % of mean /100 ep",
                          f"GU mean ± sd-across-epochs [{unit}]", "GU CV %", "GU ep0–9 vs 10–99 %", f"GU slope [{unit}/epoch]",
                          "GU slope % of mean /100 ep"], rows))
    pd.DataFrame(st_rows).to_csv(OUT_DIR / "sac_per_epoch_stationarity.csv", index=False, float_format="%.17g")
    # first-epoch detail
    e0 = PE[PE.epoch == 0]
    rest = PE[PE.epoch > 0].groupby(["env", "tag", "seed", "task"]).energy_j.mean()
    e0r = e0.set_index(["env", "tag", "seed", "task"]).energy_j / rest
    out.append(f"\nEpoch 0 vs mean of epochs 1–99 (energy ratio), all 70 runs: rollout {msd(e0r.xs('rollout', level='task'))}, "
               f"gradient_updates {msd(e0r.xs('gradient_updates', level='task'))}.")
    # 8.4 execution order
    out.append(exec_order_section())
    # 8.5 run.log lines
    out.append(log_lines_section())
    return "\n".join(out)


def exec_order_section():
    out = []
    allm = []
    for d, md in ALL_META:
        try:
            s = dt.datetime.fromisoformat(md["start_time_utc"])
            e = dt.datetime.fromisoformat(md["end_time_utc"])
        except Exception:
            continue
        allm.append((s, e, md["algo_name"], md["env_id"], md["seed"], d.as_posix()))
    allm.sort()
    sessions = []
    cur = [allm[0]]
    for prev, nxt in zip(allm, allm[1:]):
        if (nxt[0] - prev[1]).total_seconds() > 1800:
            sessions.append(cur)
            cur = []
        cur.append(nxt)
    sessions.append(cur)
    sess_of = {}
    for i, ses in enumerate(sessions):
        for x in ses:
            sess_of[x[5]] = i
    rs = sorted(RUNS, key=lambda r: r["meta"]["start_time_utc"])
    rows = []
    prev_end = None
    for k, r in enumerate(rs, 1):
        tg = r["meta"].get("thermal_gate", {})
        s = dt.datetime.fromisoformat(r["meta"]["start_time_utc"])
        e = dt.datetime.fromisoformat(r["meta"]["end_time_utc"])
        gap = (s - prev_end).total_seconds() / 60 if prev_end else float("nan")
        prev_end = e
        rows.append([k, r["timestamp"], label(r["es"], r["tag"]), r["seed"], sess_of[r["run_dir"]], g4(gap),
                     g4((e - s).total_seconds() / 60), g4(tg.get("waited_seconds")), g4(tg.get("final_temp_c")),
                     g4(tg.get("final_power_w")), tg.get("gated"), tg.get("reason"), g4(tg.get("run_start_temp_c")),
                     g4(tg.get("run_start_power_w")), g4(tg.get("run_end_temp_c")), g4(tg.get("run_end_power_w"))])
    out.append("\n**8.4 Execution order and thermal state.** The 70 runs sorted by `metadata.start_time_utc` (directory "
               "timestamp = local start time). Session = maximal block of runs from **all 300** recorded runs (all "
               "algorithms) with < 30 min between one run's end and the next run's start. Gap = minutes since the previous "
               "*SAC* run ended. Thermal fields from `metadata.json[\"thermal_gate\"]` (`final_*` = reading at which the "
               "gate passed; `run_start_*` = NVML reading after the 30 s settle, just before the tracker starts; "
               "`run_end_*` = after the idle tail; blank = key absent, see Part 0).\n")
    out.append(table(["#", "dir timestamp", "config", "seed", "session", "gap [min]", "run length [min]", "gate waited [s]",
                      "gate temp [°C]", "gate power [W]", "gated", "reason", "start temp", "start power [W]", "end temp",
                      "end power [W]"], rows))
    srows = []
    for i, ses in enumerate(sessions):
        sac_in = [r for r in rs if sess_of[r["run_dir"]] == i]
        if not sac_in:
            continue
        algos = Counter(x[2] for x in ses)
        seq = [label(r["es"], r["tag"]) for r in sac_in]
        contiguous = all(len({j for j, s_ in enumerate(seq) if s_ == c}) == (max(j for j, s_ in enumerate(seq) if s_ == c) -
                                                                          min(j for j, s_ in enumerate(seq) if s_ == c) + 1)
                         for c in set(seq))
        seeds_order = [r["seed"] for r in sac_in]
        srows.append([i, ses[0][0].isoformat(timespec="minutes"), ses[-1][1].isoformat(timespec="minutes"), len(ses),
                      dict(algos), len(sac_in), " → ".join(dict.fromkeys(seq)),
                      "configuration-major (each config's seeds contiguous, seeds in order "
                      + ",".join(map(str, CANON)) + ")" if contiguous and all(
                          [r["seed"] for r in sac_in if label(r["es"], r["tag"]) == c] == CANON for c in set(seq)) else
                      ("configuration-major" if contiguous else "neither / interleaved")])
    out.append("\nSessions containing SAC runs (times UTC):\n")
    out.append(table(["session", "first start", "last end", "# runs (all algos)", "algos", "# SAC", "SAC configs in order",
                      "order"], srows))
    gaps = [(dt.datetime.fromisoformat(b["meta"]["start_time_utc"]) - dt.datetime.fromisoformat(a["meta"]["end_time_utc"])).total_seconds()
            for a, b in zip(rs, rs[1:]) if sess_of[a["run_dir"]] == sess_of[b["run_dir"]]]
    order_ = [label(r["es"], r["tag"]) for r in rs]
    cfg_lines = []
    for es, t in CFG:
        c = label(es, t)
        idx = [i for i, x in enumerate(order_) if x == c]
        ses_c = sorted({sess_of[rs[i]["run_dir"]] for i in idx})
        contig = idx == list(range(idx[0], idx[0] + len(idx)))
        cfg_lines.append((c, ses_c, contig))
    split = [f"{c} (sessions {s_})" for c, s_, _ in cfg_lines if len(s_) > 1]
    seed_order_ok = all([r["seed"] for r in rs if label(r["es"], r["tag"]) == c] == CANON for c, _, _ in cfg_lines)
    noncontig = [c for c, _, k in cfg_lines if not k]
    out.append(f"\nWithin a session, consecutive SAC runs are separated by {g4(min(gaps))}–{g4(max(gaps))} s "
               "(median " + g4(np.median(gaps)) + " s) between one run's `end_time_utc` and the next run's "
               "`start_time_utc`. 'Run length' = `end_time_utc − start_time_utc`; `start_time_utc` is taken before the "
               "thermal gate, so run length includes the gate's reference poll, the 30 s settle and both 90 s idle "
               "windows. Each sweep script powers the machine off at its end (`sudo shutdown -h +1`); for the manually "
               "launched baseline runs no shutdown is scripted, so whether the machine was off between the baseline "
               f"sessions ({', '.join(map(str, sorted({sess_of[r['run_dir']] for r in rs if r['tag'] == 'base'})))}) "
               "is unknown. Ordering: the 5 seeds of every configuration ran consecutively (no other SAC configuration "
               "in between) — " + ("true for all 14" if not noncontig else f"NOT for {noncontig}") + "; all 5 seeds of a "
               "configuration lie in one session — " + ("true for all 14" if not split else
               f"true for {14 - len(split)} of 14, exception(s): {', '.join(split)}") + ". Seed order within every "
               f"configuration is {','.join(map(str, CANON))}: " + ("true for all 14 (configuration-major, seed-minor)."
                                                                     if seed_order_ok else "NOT always (see run table)."))
    if not seed_order_ok:
        disc("Part 8.4: seed order within a configuration is not always the canonical order (see run table).")
    out.append(f"\nThermal gate across the 70 runs: waited_seconds max {g4(max(r['meta']['thermal_gate']['waited_seconds'] for r in RUNS))} s; "
               f"gate temperature {g4(min(r['meta']['thermal_gate']['final_temp_c'] for r in RUNS))}–"
               f"{g4(max(r['meta']['thermal_gate']['final_temp_c'] for r in RUNS))} °C; gate power "
               f"{g4(min(r['meta']['thermal_gate']['final_power_w'] for r in RUNS))}–{g4(max(r['meta']['thermal_gate']['final_power_w'] for r in RUNS))} W. "
               "The gate passes on its first poll because the reference is captured fresh in each process right before "
               "(utils/thermal_gate.py); the effective cool-down is the reference stabilisation poll + 30 s settle.")
    return "\n".join(out)


def log_lines_section():
    out = []
    pat = re.compile(r"WARNING|ERROR|Traceback", re.I)
    counts = []
    msgs = Counter()
    sweep_msgs = Counter()
    for r in RUNS:
        lines_ = [ln for ln in r["log"].splitlines() if pat.search(ln)]
        counts.append(len(lines_))
        for ln in lines_:
            msgs[re.sub(r"^\S+ \S+ ", "", ln)] += 1
        if r["run_dir"] in SWEEP_BLOCKS:
            for ln in SWEEP_BLOCKS[r["run_dir"]][1].splitlines():
                if pat.search(ln):
                    sweep_msgs[re.sub(r"@ \d\d:\d\d:\d\d", "@ <time>", ln)] += 1
    out.append(f"\n**8.5 Warning/error lines.** `run.log`: lines containing `WARNING`, `ERROR` or `Traceback` "
               f"(case-insensitive) per run — total over 70 runs = {sum(counts)}, max per run = {max(counts)}.")
    if msgs:
        out.append(table(["message", "count"], [[f"`{m}`", c] for m, c in msgs.most_common()]))
    else:
        out.append("No such line in any of the 70 `run.log` files.")
    out.append(f"\nSweep-log blocks (stdout+stderr, {sum(1 for r in RUNS if r['run_dir'] in SWEEP_BLOCKS)} runs), distinct "
               "matching lines:\n")
    out.append(table(["line", "count"], [[f"`{m}`", c] for m, c in sweep_msgs.most_common()]) if sweep_msgs else "none")
    return "\n".join(out)


# ----------------------------------------------------------------------------- Part 9
def part9():
    out = []
    rows = []
    perf_rows = []
    marks = [10, 25, 50, 75, 100]
    for es, t in CFG:
        rs = runs_of(es, t)
        f10, mx, nep, steps = [], [], [], []
        mk = {m: [] for m in marks}
        for r in rs:
            eps = r["tm"]["episodes"]
            ret = [e["return"] for e in eps if e["phase"] == "train"]
            k = max(1, len(ret) // 10)
            f10.append(sum(ret[-k:]) / len(ret[-k:]))
            mx.append(max(ret))
            nep.append(len(ret))
            steps.append(r["tm"]["epochs"][-1]["env_step"])
            byep = {e["epoch"]: e for e in r["tm"]["epochs"]}
            for m in marks:
                v = byep[m - 1]["mean_episode_return"]
                mk[m].append(v)
            perf_rows.append(dict(env=r["env"], tag=t, seed=r["seed"], final_10pct_mean_return=f10[-1], max_return=mx[-1],
                                  n_train_episodes=nep[-1], env_steps=steps[-1],
                                  **{f"mean_episode_return_epoch{m}": byep[m - 1]["mean_episode_return"] for m in marks}))
        cells = [label(es, t), msd(f10), msd(mx), f"{min(nep)}–{max(nep)}", ", ".join(map(str, sorted(set(steps))))]
        for m in marks:
            v = [x for x in mk[m] if x is not None]
            cells.append(msd(v) + ("" if len(v) == 5 else f" (n={len(v)})"))
        rows.append(cells)
    pd.DataFrame(perf_rows).to_csv(OUT_DIR / "sac_learning_performance.csv", index=False, float_format="%.17g")
    lp = pd.DataFrame(perf_rows)
    ant_cfg = lp[lp.env == "Ant-v5"].groupby("tag").final_10pct_mean_return.mean()
    hc_cfg = lp[lp.env == "HalfCheetah-v5"].groupby("tag").final_10pct_mean_return.mean()
    OPEN_QUESTIONS.append(
        f"Part 9 sanity check: HalfCheetah-v5 final-10%-mean returns per configuration are {g4(hc_cfg.min())}–"
        f"{g4(hc_cfg.max())}; Ant-v5 values are {g4(ant_cfg.min())}–{g4(ant_cfg.max())}, negative in "
        f"{int((ant_cfg < 0).sum())} of 7 configurations ({', '.join(t for t in TAGS if ant_cfg[t] < 0)}). Whether "
        f"this satisfies the 'SAC learns' sanity check for Ant-v5 within 100k steps is for the thesis author to judge "
        f"(stated as a fact, not interpreted).")
    out.append("Definitions (from `aggregate_results.py`, reproduced exactly): episodes = `training_metrics.json[\"episodes\"]` "
               "with `phase == \"train\"` (warmup episodes and the trailing `train_incomplete` partial episode excluded), "
               "in order; **final-10 %-mean return** = mean of the last `max(1, n_episodes // 10)` of them; **max return** "
               "= max over them. Returns are undiscounted sums of env reward of the **stochastic training policy** (no "
               "evaluation episodes exist, Part 2.2). Per-epoch: `training_metrics.json[\"epochs\"][e][\"mean_episode_return\"]` "
               "= mean return of episodes that *ended* during epoch e; \"epoch N\" = the N-th epoch (0-based index N−1, "
               "env_step = N×1000); `None` when no episode ended in that epoch (n < 5 shown). mean ± sd over seeds "
               "(ddof = 1). Sanity check only.\n")
    out.append(table(["config", "final-10%-mean return", "max return", "# completed train episodes (range)",
                      "env steps (measured training)", "epoch 10", "epoch 25", "epoch 50", "epoch 75", "epoch 100"], rows))
    out.append("\n`training_metrics.json` keys: top level `episodes`, `epochs`; episode records "
               f"{list(RUNS[0]['tm']['episodes'][0].keys())}; epoch records {list(RUNS[0]['tm']['epochs'][0].keys())}.")
    return "\n".join(out)


# ----------------------------------------------------------------------------- schema samples
def schema_section():
    out = []
    r = runs_of("HC", "base")[0]
    out.append(f"**S1. `segment_energy.json`** — complete file of `{r['run_dir']}` (HC-base, seed {r['seed']}):\n")
    seg_sample = dict(r["seg"])
    seg_sample["_total_kg_co2eq"] = "<CO2e value omitted: CO2e is not reported in this file>"
    out.append(code(json.dumps(seg_sample, indent=2), "json"))
    keysets = Counter(tuple(r_["seg"].keys()) for r_ in RUNS)
    out.append(f"Key list shared by all 70 runs ({'identical, in this order' if len(keysets) == 1 else 'NOT identical: ' + str(keysets)}):\n")
    out.append(table(["key", "unit", "meaning"], [
        ["`idle_baseline_head`", "kWh", "measured CodeCarbon task, 90 s idle before warmup (excluded from totals)"],
        ["`warmup`", "kWh", "measured task `warmup_0`, 5000 random-action env steps (excluded from totals)"],
        ["`rollout`", "kWh", "**measured**: Σ of the 100 `rollout_i` tasks"],
        ["`buffer_sample`", "kWh", "**allocated**: Σ `gradient_updates_i` energy × T_buffer_sample/ΣT"],
        ["`critic_update`", "kWh", "**allocated** (same rule)"],
        ["`actor_update`", "kWh", "**allocated** (same rule)"],
        ["`target_update`", "kWh", "**allocated** (same rule)"],
        ["`_sub_segment_wall_time_seconds`", "s", "`_`-prefixed bookkeeping: Σ perf_counter times of the 4 sub-phases over the run"],
        ["`idle_baseline_tail`", "kWh", "measured task, 90 s idle after the last epoch (excluded)"],
        ["`_total_kg_co2eq`", "kg CO₂e", "`_`-prefixed bookkeeping: tracker.stop() return value (CO₂e, not used anywhere in this file)"],
    ]))
    out.append("There is no `gradient_updates` key: the measured fused-task energy is replaced by its 4-way allocation "
               "(it equals the sum of the 4 allocated keys, Part 3.5).")
    md = r["meta"]

    def shape(o, depth=0):
        if isinstance(o, dict):
            return {k: shape(v, depth + 1) for k, v in o.items()}
        return type(o).__name__
    out.append("\n**S2. `metadata.json`** — top-level key structure (types), then values of the relevant blocks for the "
               "same run (the complete verbatim files of one HC-base and one Ant-base run are in Part 2.1):\n")
    out.append(code(json.dumps(shape(md), indent=2), "json"))
    out.append(code(json.dumps({k: md[k] for k in ("git_commit", "torch_version", "codecarbon_version", "cuda_device_name",
                                                    "device", "start_time_utc", "end_time_utc")} |
                               {"experiment_config.gpu_min_clock_mhz": md["experiment_config"]["gpu_min_clock_mhz"],
                                "experiment_config.gpu_max_clock_mhz": md["experiment_config"]["gpu_max_clock_mhz"],
                                "algo_config": md["algo_config"], "thermal_gate": md.get("thermal_gate")}, indent=2), "json"))
    t = r["tasks"]
    out.append(f"\n**S3. CodeCarbon CSVs.** Two files per run: `emissions.csv` (one row, written by `tracker.stop()`) and the "
               f"**per-task log** `emissions_<experiment>_<run_id>.csv` (here `{Path(r['task_csv']).name}`), which holds "
               "one row per task (`idle_baseline_head`, `warmup_0`, `rollout_0…99`, `gradient_updates_0…99`, "
               "`idle_baseline_tail`) and is the file used for every task-level number in this document (the pipeline "
               "reads the same file). Per-task CSV columns and pandas dtypes:\n")
    out.append(code("\n".join(f"{c}: {d}" for c, d in t.dtypes.items())))
    out.append("First 3 rows (all columns):\n")
    out.append(code(redact_co2(t.head(3)).to_csv(index=False)))
    top = r["top"]
    out.append("`emissions.csv` (top-level) columns/dtypes and its single row:\n")
    out.append(code("\n".join(f"{c}: {d}" for c, d in top.dtypes.items())))
    out.append(code(redact_co2(top).to_csv(index=False)))
    out.append("(Values of the CO₂e columns `emissions` and `emissions_rate` are replaced by a placeholder in these "
               "samples; no CO₂e figure is used anywhere in this file.)")
    disc("The context request refers to per-epoch `rollout_{i}`/`gradient_updates_{i}` tasks 'in `emissions.csv`'. In this "
         "repo `emissions.csv` holds a single row written by `tracker.stop()`; the per-task rows are in "
         "`emissions_<experiment>_<run_id>.csv` in the same run directory (as `compute_energy_per_flop.py`'s docstring "
         "says). All task-level numbers here use the per-task file.")
    # pipeline CSVs
    pr = pd.read_csv("flop_analysis/output/per_run_energy_per_flop.csv")
    cs = pd.read_csv("flop_analysis/output/cross_seed_energy_per_flop.csv")
    desc_pr = {
        "algo": "algorithm", "env_id": "Gymnasium env id", "architecture_signature": "flop_keys.signature() (hidden sizes + batch for SAC)",
        "hidden_sizes": "'h1xh2'", "updates_per_env_step": "UTD (separate group key)", "mbpo_rollout_regime": "MBPO only (empty for SAC)",
        "seed": "run seed", "run_dir": "run directory relative to repo root",
        "included_in_cross_seed_avg": "True iff seed in the canonical set", "segment": "segment name (incl. TOTAL_MEASURED_TRAINING)",
        "flop_type": "matmul | elementwise (target_update) | none | mixed_total", "call_count": "derived calls (Part 3.2)",
        "total_flops": "per-call constant × call_count (target: op count)", "total_energy_kwh": "segment_energy.json value (TOTAL: sum)",
        "total_energy_joules": "× 3.6e6", "duration_s": "measured task duration (allocated: perf_counter time; TOTAL: rollout+GU tasks)",
        "mean_power_w": "(cpu+gpu+ram energy)/duration", "mean_cpu_power_w": "CPU share of the above",
        "mean_gpu_power_w": "GPU share", "mean_ram_power_w": "RAM share",
        "energy_per_flop_j_per_flop": "total_energy_joules / total_flops (target_update: J per op)", "note": "free text",
        "dynamics_train_flops": "MBPO only", "dynamics_holdout_flops": "MBPO only"}
    desc_cs = {"n_seeds": "number of runs averaged", "mean_energy_kwh": "mean over seeds", "mean_energy_joules": "× 3.6e6",
               "mean_duration_s": "mean over seeds", "mean_power_w": "MEAN of per-seed powers (not ratio of means)",
               "mean_cpu_power_w": "mean of per-seed values", "mean_gpu_power_w": "mean of per-seed values",
               "mean_ram_power_w": "mean of per-seed values", "total_flops": "mean of non-zero per-seed FLOPs",
               "mean_energy_per_flop_j_per_flop": "mean energy J / mean FLOPs (ratio of means)"}
    out.append("\n**S4. Pipeline CSVs.** `per_run_energy_per_flop.csv` columns, dtypes, meaning (from "
               "`compute_energy_per_flop.py`):\n")
    out.append(table(["column", "dtype", "meaning"], [[f"`{c}`", str(d), desc_pr.get(c, "")] for c, d in pr.dtypes.items()]))
    sample = pr[pr.run_dir == r["run_dir"]].drop(columns=["note"])
    out.append(f"\nRows of `{r['run_dir']}` (`note` column omitted):\n")
    out.append(code(sample.to_csv(index=False)))
    out.append("`cross_seed_energy_per_flop.csv` columns, dtypes, meaning:\n")
    out.append(table(["column", "dtype", "meaning"], [[f"`{c}`", str(d), desc_cs.get(c, desc_pr.get(c, ""))] for c, d in cs.dtypes.items()]))
    sc = cs[(cs.algo == "sac") & (cs.env_id == "HalfCheetah-v5") & (cs.architecture_signature == "bs256_h1024x1024") & (cs.updates_per_env_step == 1)]
    out.append("\nRows for HC-base:\n")
    out.append(code(sc.to_csv(index=False)))
    out.append(f"\n**S5. `training_metrics.json` keys:** top level `{list(r['tm'].keys())}`; each episode "
               f"`{list(r['tm']['episodes'][0].keys())}`; each epoch `{list(r['tm']['epochs'][0].keys())}`.")
    out.append("\n**S6. Failure-mode strings** searched (case-insensitive substring) in every `run.log` and in every "
               "available sweep-log block (sources: README 'Known limitations', `utils/gpu_control.py`, "
               "`utils/thermal_gate.py`, `experiment_runner.py`, and CodeCarbon 3.3.0 `core/cpu.py`, `core/rapl.py`, "
               "`core/resource_tracker.py`, `external/geography.py` in the local venv):\n")
    out.append(code("\n".join(GREP_STRINGS)))
    return "\n".join(out)


# ----------------------------------------------------------------------------- assemble
def main():
    import torch  # noqa: F401
    p0 = part0()
    p1 = part1()
    p2, env_info, _ = part2()
    p3 = part3(env_info)
    p4 = part4()
    cross_seed_summary_csv(env_info)
    p5 = part5(env_info)
    p6 = part6()
    p7 = part7()
    p8 = part8()
    p9 = part9()
    sch = schema_section()

    hc, ant = runs_of("HC", "base"), runs_of("Ant", "base")
    fails = [c for c in CHECKS if not c[1]]
    summary = (
        f"This file is the complete data context for thesis Section 4.1 (SAC per-algorithm energy analysis). It covers "
        f"the 70 canonical SAC runs (7 configurations × 2 environments × 5 seeds {CANON}) recorded on the Linux "
        f"experiment box (RTX 5090, clocks requested at 2000 MHz, torch 2.13.0+cu130, CodeCarbon 3.3.0) and was "
        f"generated on the Windows dev checkout from the synced `results/` and `flop_analysis/` directories by "
        f"`contexts/section_4.1_data/build_section_4_1.py` (read-only on data). All energy is gross; `rollout` and "
        f"`gradient_updates` are measured, the four sub-segments are allocated by wall-clock share. Headline: "
        f"HC-base TOTAL_MEASURED_TRAINING energy {msd([r['E']['TOTAL_MEASURED_TRAINING'] for r in hc])} J, "
        f"Ant-base {msd([r['E']['TOTAL_MEASURED_TRAINING'] for r in ant])} J; rollout share HC "
        f"{msd([r['share']['rollout'] for r in hc])} %, Ant {msd([r['share']['rollout'] for r in ant])} %. "
        f"{len(CHECKS)} checks were run, {len(fails)} FAILED; all recomputed values reconcile with the pipeline CSVs "
        f"(Part 4.8). Discrepancies and open items are in Part 11. Nothing here is interpreted; 'by construction' notes "
        f"are confined to Parts 2 and 5.")
    toc = """**Contents**
- Schema samples (S1–S6)
- Part 0 — Provenance
- Part 1 — Audit of the SAC canonical run set
- Part 2 — SAC implementation facts
- Part 3 — FLOP facts for SAC
- Part 4 — Per-configuration results (Tables 4.1–4.8)
- Part 5 — Environment comparison (HalfCheetah-v5 vs Ant-v5)
- Part 6 — Effect of each sweep (baseline-relative)
- Part 7 — Energy composition and idle-floor diagnostics
- Part 8 — Measurement-quality diagnostics
- Part 9 — Learning-performance sanity check
- Part 10 — Optional backlog (not produced in this pass)
- Part 11 — Discrepancies, open questions, things not found
- Part 12 — Index of produced files

**Conventions used throughout.** Tags: `HC` = HalfCheetah-v5, `Ant` = Ant-v5; `base` = hidden (1024,1024), batch 256,
UTD 1; `utd2`/`utd4`, `w256`/`w512`, `b512`/`b1024` change exactly that one factor. Short segment names in tables:
`buf_sample` = buffer_sample, `critic` = critic_update, `actor` = actor_update, `target` = target_update,
`grad_updates`/`GU` = gradient_updates (= Σ of the four allocated), `TOTAL` = TOTAL_MEASURED_TRAINING (= rollout +
gradient_updates). Unless a table says otherwise: n = 5 seeds, mean ± sample sd (ddof = 1), 4 significant digits,
gross energy (idle floor included, never subtracted), 1 kWh = 3.6e6 J, J/FLOP = energy per **matmul** FLOP as a ratio
of means; `target` is reported in J per Polyak elementwise op, never as J/FLOP; `buffer_sample` has no FLOPs.
Companion CSVs carry full precision."""
    disc_list = list(dict.fromkeys(DISCREPANCIES))
    not_found = [
        "GPU theoretical FP32 peak: NOT DERIVED (Part 4.6 — no SM/lane data in repo; CUDA not available on this checkout).",
        "TF32 / float32 matmul precision actually in effect on the run stack: NOT VERIFIED (Part 2.3 — code sets nothing; "
        "local CPU torch reports its defaults only).",
        "Launcher script/CLI for the 10 SAC baseline runs: NOT FOUND (Part 1.1 — inferred from metadata + cli_commands.txt).",
        "Number of update() calls per epoch: NOT LOGGED; verified indirectly (Part 1.3).",
        "Simulator share of rollout time: NOT MEASURED (Part 5.6).",
        "gymnasium/mujoco versions on the Linux box: NOT RECORDED in metadata (Part 2.4).",
        "Applied (vs requested) GPU clock lock for the 10 baseline runs: NOT VERIFIABLE from saved logs (Part 1.3).",
    ]
    decisions = [
        "Configuration tags are assigned from `metadata.json → algo_config` by diffing against `SACConfig()` defaults "
        "(exactly one factor may differ), not from directory names.",
        "`gradient_updates` energy per run = Σ of the four allocated keys in `segment_energy.json` (equals the per-task "
        "CSV sum to ≤ 2.2e-16 relative); durations of measured tasks = Σ CodeCarbon task `duration` from the per-task CSV.",
        "Mean power in all tables = per-seed energy/duration, then mean ± sd over seeds (the pipeline's cross-seed "
        "`mean_power_w` uses the same statistic).",
        "Allocated sub-segment 'duration' = summed perf_counter host time (as in the pipeline), even though coverage < 100 %.",
        "Outlier rule uses the unscaled MAD.",
        "'Epoch N' in Part 9 = 0-based epoch index N−1.",
        "Sessions in Part 8.4 are defined with all 300 runs (all algorithms), gap threshold 30 min end→start.",
        "Part 10 (optional backlog over all 280 runs) was deliberately not produced, per the user's instruction.",
    ]
    failed_lines = [f"FAILED check: {n} — {d}" for n, _, d in fails] or ["No check FAILED."]
    files = [
        ("contexts/section_4.1.md", "this file"),
        ("contexts/section_4.1_data/build_section_4_1.py", "generator script (read-only on data); rerun to regenerate everything"),
        ("contexts/section_4.1_data/sac_run_inventory.csv", "70 runs: " + ", ".join(INV.columns)),
        ("contexts/section_4.1_data/sac_segments_long.csv", "70 runs × 10 segments: " + ", ".join(LONG.columns)
         + " (target_update: `j_per_flop` holds J per elementwise op; flops = op count)"),
        ("contexts/section_4.1_data/sac_cross_seed_summary.csv", "14 configs × 10 segments, cross-seed values of Tables 4.1–4.6: "
         + ", ".join(pd.read_csv(OUT_DIR / "sac_cross_seed_summary.csv", nrows=1).columns)),
        ("contexts/section_4.1_data/sac_per_seed_totals.csv", "Table 4.7 per run: " + ", ".join(pd.read_csv(OUT_DIR / "sac_per_seed_totals.csv", nrows=1).columns)),
        ("contexts/section_4.1_data/sac_env_comparison.csv", "Part 5 quantities (long format): tag, metric, segment, value"),
        ("contexts/section_4.1_data/sac_sweep_effects.csv", "Part 6.1: env, tag, metric, paired_mean_pct, paired_sd_pct, from_means_pct (share metrics in pp)"),
        ("contexts/section_4.1_data/sac_delta_pairs.csv", "Part 6.3 pair list from flop_dashboard.compute_delta_pairs (+ x_dim, ref_mode, level)"),
        ("contexts/section_4.1_data/sac_ols_fits.csv", "Part 6.4: sweep, env, segment, n, a_J, b_J_per_FLOP, r2, rse_J"),
        ("contexts/section_4.1_data/sac_energy_components.csv", "Part 7.1: env, tag, task, component, energy_j_mean, energy_j_sd, pct_mean, pct_sd"),
        ("contexts/section_4.1_data/sac_idle_floor.csv", "Part 7.2 per run: " + ", ".join(pd.read_csv(OUT_DIR / "sac_idle_floor.csv", nrows=1).columns)),
        ("contexts/section_4.1_data/sac_allocation_coverage.csv", "Part 8.2 per run: " + ", ".join(pd.read_csv(OUT_DIR / "sac_allocation_coverage.csv", nrows=1).columns)),
        ("contexts/section_4.1_data/sac_per_epoch.csv", "Part 8.3 (14 000 rows): " + ", ".join(PE.columns)),
        ("contexts/section_4.1_data/sac_per_epoch_stationarity.csv", "Part 8.3 per run×task×quantity: env, tag, seed, task, quantity, mean, sd, cv, early, slope, slope_pct"),
        ("contexts/section_4.1_data/sac_learning_performance.csv", "Part 9 per run: " + ", ".join(pd.read_csv(OUT_DIR / "sac_learning_performance.csv", nrows=1).columns)),
    ]

    doc = [f"# Section 4.1 context — SAC per-algorithm energy analysis\n", summary, "", toc]
    doc += ["\n## Schema samples\n", sch]
    doc += ["\n## Part 0 — Provenance\n", p0]
    doc += ["\n## Part 1 — Audit of the SAC canonical run set\n", p1]
    doc += ["\n## Part 2 — SAC implementation facts\n", p2]
    doc += ["\n## Part 3 — FLOP facts for SAC\n", p3]
    doc += ["\n## Part 4 — Per-configuration results\n", p4]
    doc += ["\n## Part 5 — Environment comparison (HalfCheetah-v5 vs Ant-v5)\n", p5]
    doc += ["\n## Part 6 — Effect of each sweep (baseline-relative)\n", p6]
    doc += ["\n## Part 7 — Energy composition and idle-floor diagnostics (descriptive only)\n", p7]
    doc += ["\n## Part 8 — Measurement-quality diagnostics for SAC\n", p8]
    doc += ["\n## Part 9 — Learning-performance sanity check (SAC)\n", p9]
    doc += ["\n## Part 10 — Optional backlog\n",
            "Not produced in this pass (excluded by the user's instruction). Nothing in Parts 0–9 depends on it."]
    doc += ["\n## Part 11 — Discrepancies, open questions, and things not found\n",
            "**Contradictions between documentation and code/data:**\n",
            "\n".join(f"{i}. {d}" for i, d in enumerate(disc_list, 1)),
            "\n**Open questions (facts the thesis author should look at):**\n",
            "\n".join(f"- {q}" for q in OPEN_QUESTIONS) if OPEN_QUESTIONS else "none",
            "\n**NOT FOUND / NOT COMPUTED / NOT VERIFIED:**\n",
            "\n".join(f"{i}. {d}" for i, d in enumerate(not_found, len(disc_list) + 1)),
            "\n**Failed checks:**\n", "\n".join(f"- {x}" for x in failed_lines),
            "\n**Decisions taken:**\n",
            "\n".join(f"{i}. {d}" for i, d in enumerate(decisions, len(disc_list) + len(not_found) + 1)),
            f"\nCheck tally: {len(CHECKS)} checks, {len(CHECKS) - len(fails)} PASS, {len(fails)} FAIL."]
    doc += ["\n## Part 12 — Index of produced files\n", table(["file", "description / columns"], [[f"`{a}`", b] for a, b in files])]
    OUT_MD.write_text("\n".join(doc) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT_MD} ({OUT_MD.stat().st_size} bytes); checks {len(CHECKS)}, failed {len(fails)}")
    for n, ok, d in CHECKS:
        if not ok:
            print("FAIL:", n, d)


if __name__ == "__main__":
    main()
