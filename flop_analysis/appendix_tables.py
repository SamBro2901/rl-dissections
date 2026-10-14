"""
Appendix B (full result tables): reproducible generator of the LaTeX tables.

Reads the raw run files of the 280 canonical runs (metadata.json, segment_energy.json, per-task CodeCarbon CSV),
flop_analysis/output/per_run_energy_per_flop.csv (FLOP counts) and the Section 5 context CSVs, and writes LaTeX
table fragments (longtable bodies and complete tables) plus CSV copies of every computed table.

Conventions (as in Chapter 4 of the thesis): energies are gross; a configuration value is the mean over the five
canonical seeds with the sample standard deviation (n-1); J/FLOP is the ratio of the mean energy to the mean FLOPs,
its spread the sample sd of the five per-seed ratios; shares and fractions are computed per run and then averaged.

Usage (from the repository root, with pandas and SciPy):
    python flop_analysis/appendix_tables.py [--out contexts/appendix/appendix_data/B]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = os.path.join(REPO, "contexts")
BS = chr(92)
KWH = 3.6e6
SEEDS = [331, 958, 14577, 43611, 85062]
ENVS = ["HalfCheetah-v5", "Ant-v5"]
ALGOS = ["sac", "td3", "mbpo", "tdmpc2"]
NAME = {"sac": "SAC", "td3": "TD3", "mbpo": "MBPO", "tdmpc2": "TD-MPC2"}
ENVSHORT = {"HalfCheetah-v5": "HC", "Ant-v5": "Ant"}
SUBS = ["buffer_sample", "critic_update", "actor_update", "target_update"]
ORDER = {
    "sac": ["baseline", "UTD 2", "UTD 4", "width 256", "width 512", "batch 512", "batch 1024"],
    "td3": ["baseline", "UTD 2", "UTD 4", "width 256", "width 512", "batch 256", "batch 512", "batch 1024"],
    "mbpo": ["baseline", "UTD 2", "UTD 4", "width 256", "width 512", "batch 512", "batch 1024", "rollout 1", "rollout 15"],
    "tdmpc2": ["baseline", "num_q 3", "num_q 7", "horizon 1", "horizon 5"],
}
SEG_ORDER = {
    "sac": ["rollout"] + SUBS + ["gradient_updates", "TOTAL"],
    "td3": ["rollout"] + SUBS + ["gradient_updates", "TOTAL"],
    "mbpo": ["rollout", "dynamics_model_update", "synthetic_rollout_generation"] + SUBS + ["gradient_updates", "TOTAL"],
    "tdmpc2": ["rollout", "world_model_pretrain"] + SUBS + ["gradient_updates", "TOTAL"],
}
MEASURED = {"rollout", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain", "gradient_updates", "TOTAL"}
SEGLABEL = {"TOTAL": "Total", "gradient_updates": "gradient_updates"}


def short(cfg: str) -> str:
    if cfg == "baseline":
        return "baseline"
    k, v = cfg.split("=")
    v = v.replace("[", "").replace("]", "").replace(" ", "").split(",")[0]
    return {"updates_per_env_step": "UTD ", "hidden_sizes": "width ", "batch_size": "batch ", "rollout_max_length": "rollout ",
            "num_q": "num_q ", "horizon": "horizon "}[k] + v


def config_name(algo_cfg: dict, base: dict) -> str:
    """Name of a run's configuration: the field that differs from the baseline of its algorithm and environment."""
    diffs = {k: v for k, v in algo_cfg.items() if base.get(k) != v}
    if not diffs:
        return "baseline"
    (k, v), = diffs.items()
    return short(f"{k}={v}")


# ---------------------------------------------------------------------------------------------- LaTeX helpers
def tt(s: str) -> str:
    return BS + "texttt{" + s.replace("_", BS + "_" + BS + "allowbreak ") + "}"


def sig(x: float, n: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "---"
    if x == 0:
        return "0"
    out = f"{x:.{n}g}"
    return re.sub(r"e([+-])0*(\d)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), out)


def num(x: float, n: int = 6) -> str:
    s = sig(x, n)
    return s if s == "---" else BS + "num{" + s + "}"


def ms(m: float, s: float, n: int = 6) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "---"
    if s is None or (isinstance(s, float) and math.isnan(s)) or abs(s) < 1e-10 * abs(m):
        return num(m, n)    # no spread (or only floating-point noise, e.g. the share of the total in the total row)
    # siunitx does not accept an exponent in front of "+-": large values are written as plain integers
    return BS + "num{" + plain(sig(m, n)) + "+-" + plain(sig(s, n - 2 if n > 3 else n)) + "}"


def plain(s: str) -> str:
    """Convert a value with an exponent (e.g. 9.34276e6) to plain decimal notation."""
    if "e" not in s:
        return s
    v = float(s)
    if abs(v) >= 1:
        return f"{v:.0f}"
    return f"{v:.{int(-math.floor(math.log10(abs(v)))) + 5}f}".rstrip("0")


def segname(seg: str) -> str:
    return "Total" if seg == "TOTAL" else tt(seg)


# ---------------------------------------------------------------------------------------------- data
def load_runs() -> pd.DataFrame:
    """One row per canonical run with all per-run quantities used by the tables."""
    per_run = pd.read_csv(os.path.join(REPO, "flop_analysis/output/per_run_energy_per_flop.csv"),
                          usecols=["algo", "env_id", "seed", "run_dir", "segment", "flop_type", "call_count", "total_flops", "total_energy_joules",
                                   "duration_s", "dynamics_train_flops", "dynamics_holdout_flops"], dtype={"run_dir": str, "segment": str})
    per_run["rd"] = per_run.run_dir.str.replace("results/", "", regex=False)
    paths = sorted(glob.glob(os.path.join(REPO, "results/*/*/seed_*/*/metadata.json")))
    meta = {}
    for p in paths:
        m = json.load(open(p))
        if m["seed"] in SEEDS:
            rd = os.path.relpath(os.path.dirname(p), os.path.join(REPO, "results")).replace(os.sep, "/")
            meta[rd] = m
    assert len(meta) == 280, len(meta)
    base = {}
    for rd, m in meta.items():  # the baseline is the config that equals the default of the algorithm: take the most common config
        pass
    # baseline per (algo, env) = the config with the modal field values: determine via the sweep structure
    groups = {}
    for rd, m in meta.items():
        groups.setdefault((m["algo_name"], m["env_id"]), []).append(m["algo_config"])
    for key, cfgs in groups.items():
        base[key] = {}
        for k in cfgs[0]:
            vals = [json.dumps(c[k]) for c in cfgs]
            base[key][k] = json.loads(max(set(vals), key=vals.count))
    rows = []
    for rd, m in meta.items():
        algo, env, seed = m["algo_name"], m["env_id"], m["seed"]
        d = os.path.join(REPO, "results", rd)
        seg = json.load(open(os.path.join(d, "segment_energy.json")))
        task = pd.read_csv(glob.glob(os.path.join(d, "emissions_base_*.csv"))[0], float_precision="round_trip")
        task["kind"] = task.task_name.str.replace(r"_\d+$", "", regex=True)
        pr = per_run[per_run.rd == rd].set_index("segment")
        r = dict(rd=rd, algo=algo, env=env, seed=seed, cfg=config_name(m["algo_config"], base[(algo, env)]),
                 start=m["start_time_utc"], commit=m["git_commit"][:7])
        # energies [J] from segment_energy.json (as in the thesis tables)
        E = {k: seg[k] * KWH for k in ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain"] + SUBS if k in seg}
        E["gradient_updates"] = sum(E[s] for s in SUBS)
        train = [k for k in ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain", "gradient_updates"] if k in E]
        E["TOTAL"] = sum(E[k] for k in train)
        timers = seg["_sub_segment_wall_time_seconds"]
        # task durations and component energies [J]
        T, CPU, GPU, RAM, TASKS = {}, {}, {}, {}, {}
        for kind in ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain", "gradient_updates"]:
            t = task[task.kind == kind]
            if len(t):
                T[kind] = t.duration.sum()
                CPU[kind] = t.cpu_energy.sum() * KWH
                GPU[kind] = t.gpu_energy.sum() * KWH
                RAM[kind] = t.ram_energy.sum() * KWH
                TASKS[kind] = t.duration.to_numpy()
        for s in SUBS:
            T[s] = timers[s]
        T["TOTAL"] = sum(T[k] for k in train)
        F = {}
        for s in ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain", "buffer_sample", "critic_update", "actor_update", "target_update"]:
            if s in pr.index and not (isinstance(pr.loc[s, "total_flops"], float) and math.isnan(pr.loc[s, "total_flops"])):
                F[s] = float(pr.loc[s, "total_flops"])
        F["gradient_updates"] = F["critic_update"] + F["actor_update"]
        F["TOTAL"] = float(pr.loc["TOTAL_MEASURED_TRAINING", "total_flops"])
        # idle floor
        P_head = task[task.task_name == "idle_baseline_head"].energy_consumed.iloc[0] * KWH / task[task.task_name == "idle_baseline_head"].duration.iloc[0]
        P_tail = task[task.task_name == "idle_baseline_tail"].energy_consumed.iloc[0] * KWH / task[task.task_name == "idle_baseline_tail"].duration.iloc[0]
        r.update(E=E, T=T, F=F, cpu=CPU, gpu=GPU, ram=RAM, tasks=TASKS, P_head=P_head, P_tail=P_tail, P_idle=0.5 * (P_head + P_tail),
                 timers=timers, mdl=m["algo_config"])
        if algo == "mbpo":
            tm = json.load(open(os.path.join(d, "training_metrics.json")))
            fe = [e["model_train_epochs"] for e in tm["epochs"] if e["model_train_epochs"]]
            ft = task[task.kind == "dynamics_model_update"].copy()
            ft["i"] = ft.task_name.str.extract(r"_(\d+)$").astype(int)
            ft = ft.sort_values("i")
            r.update(fit_epochs=fe, fit_E_epochs=(ft.energy_consumed.to_numpy() * KWH),
                     opt_steps=float(pr.loc["dynamics_model_update", "call_count"]),
                     synth_samples=float(pr.loc["synthetic_rollout_generation", "call_count"]),
                     dyn_train=float(pr.loc["dynamics_model_update", "dynamics_train_flops"]),
                     dyn_hold=float(pr.loc["dynamics_model_update", "dynamics_holdout_flops"]))
        # consistency with the per-run table
        r["_chk_E"] = abs(E["TOTAL"] / KWH - float(pr.loc["TOTAL_MEASURED_TRAINING", "total_energy_joules"]) / KWH) / (E["TOTAL"] / KWH)
        rows.append(r)
    return pd.DataFrame(rows)


def agg(runs: pd.DataFrame) -> dict:
    """Aggregates per (algo, env, cfg, segment): dict of arrays over the five seeds."""
    out = OrderedDict()
    for (algo, env, cfg), g in runs.groupby(["algo", "env", "cfg"], sort=False):
        assert len(g) == 5, (algo, env, cfg, len(g))
        g = g.sort_values("seed")
        segs = SEG_ORDER[algo]
        d = {}
        for s in segs:
            E = np.array([r["E"][s] for _, r in g.iterrows()])
            Et = np.array([r["E"]["TOTAL"] for _, r in g.iterrows()])
            Egu = np.array([r["E"]["gradient_updates"] for _, r in g.iterrows()])
            T = np.array([r["T"][s] for _, r in g.iterrows()])
            F = np.array([r["F"].get(s, np.nan) for _, r in g.iterrows()])
            rec = dict(E=E, T=T, F=F, share_total=100 * E / Et, share_gu=(100 * E / Egu if (s in SUBS) else None))
            with np.errstate(divide="ignore", invalid="ignore"):
                rec["jpf_runs"] = E / F
                rec["P"] = E / T
                rec["gflops"] = F / T / 1e9
            d[s] = rec
        out[(algo, env, cfg)] = d
    return out


def msd(a):
    a = np.asarray(a, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1))


# ---------------------------------------------------------------------------------------------- table scaffolding
ABBR = {"rollout": "rollout", "buffer_sample": "buffer", "critic_update": "critic", "actor_update": "actor", "target_update": "target",
        "gradient_updates": "GU", "TOTAL": "total", "dynamics_model_update": "fit", "synthetic_rollout_generation": "synth",
        "world_model_pretrain": "pretrain"}
ABBR_NOTE = ("Segment abbreviations: buffer = \\texttt{buffer\\_sample}, critic = \\texttt{critic\\_update} (TD-MPC2: world-model step), "
             "actor = \\texttt{actor\\_update} (TD-MPC2: policy-prior step), target = \\texttt{target\\_update}, GU = \\texttt{gradient\\_updates}, "
             "fit = \\texttt{dynamics\\_model\\_update}, synth = \\texttt{synthetic\\_rollout\\_generation}, pretrain = \\texttt{world\\_model\\_pretrain}. "
             "Buffer, critic, actor and target are allocated from the GU energy by timed share; the other segments are measured. HC is HalfCheetah-v5, Ant is Ant-v5.")


def longtable(label: str, caption: str, colspec: str, header: str, body: list, ncols: int, size: str = "footnotesize") -> str:
    """A longtable inside a size group; header is the LaTeX of the header rows (ending with a line break)."""
    L = []
    L.append("{" + BS + size + BS + "setlength{" + BS + "tabcolsep}{4pt}")
    L.append("  " + BS + "begin{longtable}{" + colspec + "}")
    # short caption for the list of tables: the first sentence (up to the first full stop followed by a space), without a final period
    short = caption.split(". ")[0].rstrip(".")
    if short.count("{") != short.count("}") or short.count("$") % 2:
        short = caption.split(":")[0]
    L.append("    " + BS + "caption[" + short + "]{" + caption + "}")
    L.append("    " + BS + "label{" + label + "}" + " " + BS + BS)
    L.append("    " + BS + "toprule")
    L.append("    " + header)
    L.append("    " + BS + "midrule")
    L.append("    " + BS + "endfirsthead")
    L.append("    " + BS + "toprule")
    L.append("    " + header)
    L.append("    " + BS + "midrule")
    L.append("    " + BS + "endhead")
    L.append("    " + BS + "midrule")
    L.append("    " + BS + "multicolumn{%d}{r@{}}{" % ncols + BS + "emph{continued on the next page}} " + BS + BS)
    L.append("    " + BS + "endfoot")
    L.append("    " + BS + "bottomrule")
    L.append("    " + BS + "endlastfoot")
    L.extend(body)
    L.append("  " + BS + "end{longtable}")
    L.append("}")
    return "\n".join(L) + "\n"


def cfg_label(c: str) -> str:
    return c.replace("num_q", BS + "texttt{num" + BS + "_q}")


def iter_rows(A, algo):
    """Yield (env, cfg, segs) in the display order."""
    for env in ENVS:
        for cfg in ORDER[algo]:
            if (algo, env, cfg) in A:
                yield env, cfg, A[(algo, env, cfg)]


def build_rows(A, algo, cells_fn, segs=None):
    """Body rows: env | config | segment | cells...  cells_fn(seg, rec) -> list of cell strings."""
    body = []
    prev_env = None
    for env, cfg, d in iter_rows(A, algo):
        if prev_env is not None and env != prev_env:
            body.append("    " + BS + "midrule")
        first_seg = True
        for s in (segs or SEG_ORDER[algo]):
            if s not in d:
                continue
            cells = cells_fn(s, d[s])
            lead = [ENVSHORT[env] if (env != prev_env and first_seg) else "", cfg_label(cfg) if first_seg else "", ABBR[s]]
            body.append("    " + " & ".join(lead + cells) + " " + BS + BS)
            first_seg = False
        body.append("    " + BS + "addlinespace[2pt]")
        prev_env = env
    return body


# ---------------------------------------------------------------------------------------------- Chapter 4 tables
def table_energy(A, algo):
    def cells(s, r):
        return [ms(*msd(r["E"])), ms(*msd(r["share_total"])), (ms(*msd(r["share_gu"])) if r["share_gu"] is not None else ("100" if s == "gradient_updates" else "---"))]
    cap = f"{NAME[algo]}: energy per segment, share of the total and share of the gradient updates (mean $\\pm$ sd over the five seeds, gross energy). " + ABBR_NOTE
    hdr = "Env. & Config. & Segment & Energy [J] & Share of total [\\%] & Share of GU [\\%] " + BS + BS
    return longtable(f"tab:app-{algo}-energy", cap, "@{}lllrrr@{}", hdr, build_rows(A, algo, cells), 6)


def table_jpf(A, algo):
    def cells(s, r):
        F = r["F"]
        if s == "buffer_sample":
            return [num(0), "---"]
        if s == "target_update":
            m = np.mean(r["E"]) / np.mean(F) * 1e9
            sd = np.std(r["jpf_runs"] * 1e9, ddof=1)
            return [num(float(np.mean(F))), ms(m, sd) + BS + ",nJ/op"]
        m = np.mean(r["E"]) / np.mean(F) * 1e12
        sd = np.std(r["jpf_runs"] * 1e12, ddof=1)
        return [num(float(np.mean(F))), ms(m, sd)]
    cap = (f"{NAME[algo]}: matmul FLOPs per run and energy per FLOP (ratio of the mean energy to the mean FLOPs, with the sd of the five per-seed ratios; gross energy). "
           "The FLOPs of \\texttt{target\\_update} are Polyak operations and its ratio is in nJ per operation. " + ABBR_NOTE)
    hdr = "Env. & Config. & Segment & FLOPs & J/FLOP [pJ] " + BS + BS
    return longtable(f"tab:app-{algo}-jpf", cap, "@{}lllrr@{}", hdr, build_rows(A, algo, cells), 5)


def table_time(A, algo):
    def cells(s, r):
        T = r["T"]
        P = (ms(*msd(r["P"])) if s in MEASURED else "---")
        g = (ms(*msd(r["gflops"])) if (s in MEASURED or s in ("critic_update", "actor_update")) else "---")
        return [ms(*msd(T)), P, g]
    cap = (f"{NAME[algo]}: duration $T$, mean power $\\bar P$ and achieved throughput (mean $\\pm$ sd over the five seeds). "
           "The duration of a measured segment is the sum of the CodeCarbon task durations, that of an allocated sub-segment the summed host time of its timers, "
           "and the throughput of a sub-segment is its FLOPs divided by this host time. " + ABBR_NOTE)
    hdr = "Env. & Config. & Segment & $T$ [s] & $\\bar P$ [W] & GFLOP/s " + BS + BS
    return longtable(f"tab:app-{algo}-time", cap, "@{}lllrrr@{}", hdr, build_rows(A, algo, cells), 6)


def comp_arrays(runs, algo, env, cfg, task):
    g = runs[(runs.algo == algo) & (runs.env == env) & (runs.cfg == cfg)].sort_values("seed")
    cpu = np.array([r["cpu"][task] for _, r in g.iterrows()])
    gpu = np.array([r["gpu"][task] for _, r in g.iterrows()])
    ram = np.array([r["ram"][task] for _, r in g.iterrows()])
    E = np.array([r["E"][task] for _, r in g.iterrows()])
    T = np.array([r["T"][task] for _, r in g.iterrows()])
    Pi = np.array([r["P_idle"] for _, r in g.iterrows()])
    tot = cpu + gpu + ram
    return 100 * cpu / tot, 100 * gpu / tot, 100 * ram / tot, 100 * Pi * T / E, float(np.max(np.abs(tot - E) / E))


def table_comp(runs, A, algo):
    tasks = {"sac": ["rollout", "gradient_updates"], "td3": ["rollout", "gradient_updates"],
             "mbpo": ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "gradient_updates"],
             "tdmpc2": ["rollout", "world_model_pretrain", "gradient_updates"]}[algo]
    body = []
    prev_env = None
    maxdev = 0.0
    for env, cfg, d in iter_rows(A, algo):
        if prev_env is not None and env != prev_env:
            body.append("    " + BS + "midrule")
        first = True
        for t in tasks:
            c, g, r, idle, dev = comp_arrays(runs, algo, env, cfg, t)
            maxdev = max(maxdev, dev)
            body.append("    " + " & ".join([ENVSHORT[env] if (env != prev_env and first) else "", cfg_label(cfg) if first else "", ABBR[t],
                                              ms(*msd(c)), ms(*msd(g)), ms(*msd(r)), ms(*msd(idle))]) + " " + BS + BS)
            first = False
        body.append("    " + BS + "addlinespace[2pt]")
        prev_env = env
    cap = (f"{NAME[algo]}: composition of the energy of the measured tasks by hardware term (CPU, GPU, RAM) and idle-floor fraction (mean $\\pm$ sd of the per-run percentages over the five seeds). "
           "The RAM term is the constant capacity heuristic of CodeCarbon. The idle-floor fraction is a diagnostic: the mean of the head and tail idle power of the run (as recorded) times the duration of the task, divided by the gross energy of the task; it is not subtracted anywhere. "
           + ABBR_NOTE)
    hdr = "Env. & Config. & Task & CPU [\\%] & GPU [\\%] & RAM [\\%] & Idle floor [\\%] " + BS + BS
    return longtable(f"tab:app-{algo}-comp", cap, "@{}lllrrrr@{}", hdr, body, 7, size="scriptsize"), maxdev


def table_quality(runs, A, algo):
    cols = {"sac": ["rollout", "gradient_updates"], "td3": ["rollout", "gradient_updates"],
            "mbpo": ["dynamics_model_update", "rollout", "synthetic_rollout_generation", "gradient_updates"], "tdmpc2": ["rollout", "gradient_updates"]}[algo]
    head = {"rollout": "Rollout task [s]", "gradient_updates": "GU task [s]", "dynamics_model_update": "Fit task [s]", "synthetic_rollout_generation": "Synth task [s]"}
    body = []
    prev_env = None
    for env, cfg, d in iter_rows(A, algo):
        g = runs[(runs.algo == algo) & (runs.env == env) & (runs.cfg == cfg)].sort_values("seed")
        if prev_env is not None and env != prev_env:
            body.append("    " + BS + "midrule")
        cells = []
        for t in cols:
            allt = np.concatenate([r["tasks"][t] for _, r in g.iterrows()])
            cells.append(num(float(np.median(allt))))
        cov = np.array([sum(r["timers"][s] for s in SUBS) / r["T"]["gradient_updates"] for _, r in g.iterrows()]) * 100
        Et = np.array([r["E"]["TOTAL"] for _, r in g.iterrows()])
        cv = 100 * np.std(Et, ddof=1) / np.mean(Et)
        idle = np.array([abs(r["P_tail"] - r["P_head"]) / r["P_head"] for _, r in g.iterrows()]) * 100
        if algo != "mbpo":
            cells.append(ms(float(np.mean(cov)), float(np.std(cov, ddof=1))))
        cells += [num(cv), ms(*msd(idle))]
        body.append("    " + " & ".join([ENVSHORT[env] if env != prev_env else "", cfg_label(cfg)] + cells) + " " + BS + BS)
        prev_env = env
    ncov = 0 if algo == "mbpo" else 1
    hdr = "Env. & Config. & " + " & ".join(head[t] for t in cols) + (" & Coverage [\\%]" if ncov else "") + " & CV total [\\%] & Idle diff. [\\%] " + BS + BS
    cs = "@{}ll" + "r" * (len(cols) + ncov + 2) + "@{}"
    cap = (f"{NAME[algo]}: measurement diagnostics of all configurations. Task durations are medians over the 500 tasks of a configuration (five seeds of 100 epochs). "
           "Coverage is the mean over the seeds of the sum of the four sub-segment timers over the measured GU duration (mean $\\pm$ sd). CV is the coefficient of variation of the total energy over the seeds. "
           "Idle difference is $|P_{\\mathrm{tail}}-P_{\\mathrm{head}}|/P_{\\mathrm{head}}$ (mean $\\pm$ sd over seeds, as recorded). HC is HalfCheetah-v5, Ant is Ant-v5.")
    return longtable(f"tab:app-{algo}-quality", cap, cs, hdr, body, len(cols) + ncov + 4, size="scriptsize")


def table_mbpo_fit(runs):
    """Two tables: (a) fit epochs, optimiser steps and synthetic samples, (b) FLOPs and energy statistics of the fit."""
    body_a, body_b = [], []
    prev_env = None
    for env in ENVS:
        for cfg in ORDER["mbpo"]:
            g = runs[(runs.algo == "mbpo") & (runs.env == env) & (runs.cfg == cfg)].sort_values("seed")
            if g.empty:
                continue
            if prev_env is not None and env != prev_env:
                body_a.append("    " + BS + "midrule")
                body_b.append("    " + BS + "midrule")
            first = [r["fit_epochs"][0] for _, r in g.iterrows()]
            later = [np.mean(r["fit_epochs"][1:]) for _, r in g.iterrows()]
            steps = [r["opt_steps"] for _, r in g.iterrows()]
            fl = np.array([r["dyn_train"] + r["dyn_hold"] for _, r in g.iterrows()])
            hs = np.array([100 * r["dyn_hold"] / (r["dyn_train"] + r["dyn_hold"]) for _, r in g.iterrows()])
            cvw = [100 * np.std(r["fit_E_epochs"], ddof=1) / np.mean(r["fit_E_epochs"]) for _, r in g.iterrows()]
            e0 = [r["fit_E_epochs"][0] / np.mean(r["fit_E_epochs"][1:]) for _, r in g.iterrows()]
            Ef = np.array([r["E"]["dynamics_model_update"] for _, r in g.iterrows()])
            steps_first = [e0_ * math.ceil((5000 - max(1, int(5000 * 0.2))) / 256) for e0_ in first]
            synth = [r["synth_samples"] for _, r in g.iterrows()]
            lead = [ENVSHORT[env] if env != prev_env else "", cfg_label(cfg)]
            body_a.append("    " + " & ".join(lead + [ms(*msd(first)), ms(*msd(later)), ms(*msd(steps)), ms(*msd(steps_first)), ms(*msd(synth))]) + " " + BS + BS)
            body_b.append("    " + " & ".join(lead + [num(float(fl.mean()), 5), num(float(fl.min()), 5) + "--" + num(float(fl.max()), 5), num(float(hs.mean()), 5),
                                                      ms(*msd(cvw), 5), ms(*msd(e0), 5), num(100 * float(np.std(Ef, ddof=1) / np.mean(Ef)), 5)]) + " " + BS + BS)
            prev_env = env
    cap_a = ("MBPO: fit epochs, optimiser steps and synthetic samples of all configurations (mean $\\pm$ sd over the five seeds). Every run has exactly 100 fits. "
             "Fit epochs are the passes over the training split of one fit; the first fit is the one of epoch 0 and the later fits are those of epochs 1 to 99. "
             "Steps of the first fit are its fit epochs times 16 optimiser steps per fit epoch (buffer of 5000 transitions, 4000 of them in the training split). "
             "Synthetic samples are the synthetic transitions generated per run. HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr_a = "Env. & Config. & Epochs, first fit & Epochs, later fits & Optimiser steps & Steps, first fit & Synthetic samples " + BS + BS
    cap_b = ("MBPO: FLOPs and energy of the model fit of all configurations (mean $\\pm$ sd over the five seeds; for the FLOPs the mean and the minimum and maximum over the seeds). "
             "The holdout share is the share of the holdout evaluation in the FLOPs of the fit. CV (within) is the coefficient of variation of the per-epoch fit energy within a run, "
             "epoch 0 / rest the energy of epoch 0 divided by the mean of epochs 1 to 99, and CV (seeds) the coefficient of variation of the fit energy of the run over the seeds. HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr_b = "Env. & Config. & FLOPs (mean) & FLOPs (min--max) & Holdout share [\\%] & CV (within) [\\%] & Epoch 0 / rest & CV (seeds) [\\%] " + BS + BS
    return (regular_table("tab:app-mbpo-fit", cap_a, "@{}llrrrrr@{}", hdr_a, body_a)
            + "\n" + regular_table("tab:app-mbpo-fit-flops", cap_b, "@{}llrrrrrr@{}", hdr_b, body_b))


def regular_table(label: str, caption: str, colspec: str, header: str, body: list, size: str = "scriptsize") -> str:
    """A table of one page, scaled to the text width."""
    short = caption.split(". ")[0].rstrip(".")
    if short.count("{") != short.count("}") or short.count("$") % 2:
        short = caption.split(":")[0]
    L = [BS + "begin{table}[tbp]", "  " + BS + "centering", "  " + BS + size, "  " + BS + "caption[" + short + "]{" + caption + "}", "  " + BS + "label{" + label + "}",
         "  " + BS + "resizebox{" + BS + "textwidth}{!}{%", "    " + BS + "begin{tabular}{" + colspec + "}", "      " + BS + "toprule", "      " + header, "      " + BS + "midrule"]
    L.extend("    " + b for b in body)
    L += ["      " + BS + "bottomrule", "    " + BS + "end{tabular}}", BS + "end{table}"]
    return "\n".join(L) + "\n"


def table_mbpo_vs_sac(A):
    body = []
    for cfg in ORDER["sac"]:
        cells = []
        for env in ENVS:
            m, s = A[("mbpo", env, cfg)], A[("sac", env, cfg)]
            dgu = 100 * (np.mean(m["gradient_updates"]["E"]) / np.mean(s["gradient_updates"]["E"]) - 1)
            dbuf = 100 * (np.mean(m["buffer_sample"]["E"]) / np.mean(s["buffer_sample"]["E"]) - 1)
            tot = np.mean(m["TOTAL"]["E"]) / np.mean(s["TOTAL"]["E"])
            cells += [num(dgu), num(dbuf), num(float(tot))]
        body.append("    " + " & ".join([cfg_label(cfg)] + cells) + " " + BS + BS)
    cap = ("MBPO compared with SAC at the same settings (full precision). $\\Delta$GU and $\\Delta$buffer are the relative differences of the mean energy of the gradient updates and of the allocated buffer sample (MBPO relative to SAC, in percent). "
           "The total ratio is the mean total energy of MBPO divided by that of SAC.")
    hdr = ("& \\multicolumn{3}{c}{HalfCheetah-v5} & \\multicolumn{3}{c}{Ant-v5} " + BS + BS + "\n    " + BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}\n    "
           "Config. & $\\Delta$GU [\\%] & $\\Delta$buffer [\\%] & Total ratio & $\\Delta$GU [\\%] & $\\Delta$buffer [\\%] & Total ratio " + BS + BS)
    return longtable("tab:app-mbpo-sac", cap, "@{}lrrrrrr@{}", hdr, body, 7)


# ---------------------------------------------------------------------------------------------- environment comparison (context 5.4)
def table_env(algo):
    d = pd.read_csv(os.path.join(CTX, "section_5.4_data/env_comparison_paired.csv"))
    d = d[d.algo == algo]
    key = {c: i for i, c in enumerate(ORDER[algo])}
    tag = {"base": "baseline", "utd2": "UTD 2", "utd4": "UTD 4", "w256": "width 256", "w512": "width 512", "b256": "batch 256", "b512": "batch 512",
           "b1024": "batch 1024", "base256": "batch 256", "numq3": "num_q 3", "numq7": "num_q 7", "h1": "horizon 1", "h5": "horizon 5"}
    rows = []
    for _, r in d.iterrows():
        rows.append((r.config, r))
    body = []
    for cname, r in rows:
        seg = str(r.max_abs_dshare_segment).replace("gradient_updates", "GU").replace("dynamics_model_update", "fit").replace("world_model_pretrain", "pretrain") \
            .replace("synthetic_rollout_generation", "synth").replace("_update", "").replace("buffer_sample", "buffer")
        body.append("    " + " & ".join([cfg_label(str(cname)),
                                          num(r.E_total_ratio_of_means), num(r.E_rollout_ratio_of_means), num(r.max_abs_dshare_pp) + f" ({seg})",
                                          num(r.P_total_ratio_of_means), num(r.Proll_over_PGU_double_ratio_Ant_over_HC), num(r.jpf_total_ratio_Ant_over_HC)]) + " " + BS + BS)
    return body, d


def table_env_tex(algo, label, caption_extra=""):
    body, d = table_env(algo)
    cap = (f"Environment comparison of all {NAME[algo]} configurations (Ant-v5 relative to HalfCheetah-v5, runs paired by seed; full precision). "
           "Energy ratios are ratios of means. $\\Delta$share is the difference of the mean shares in percentage points; the segment that attains the maximum absolute difference is given in brackets. "
           "$\\bar P_r/\\bar P_{\\mathrm{GU}}$ is the ratio of the mean power of the rollout to that of the GU; the column gives the ratio of this quantity on Ant-v5 to that on HalfCheetah-v5. " + caption_extra)
    hdr = ("Config. & $E_{\\mathrm{total}}$ ratio & $E_{\\mathrm{rollout}}$ ratio & max $|\\Delta$share$|$ [pp] & $\\bar P_{\\mathrm{total}}$ ratio & "
           "$\\bar P_r/\\bar P_{\\mathrm{GU}}$ ratio & J/FLOP total ratio " + BS + BS)
    return longtable(label, cap, "@{}lrrrrrr@{}", hdr, body, 7)


# ---------------------------------------------------------------------------------------------- Chapter 5 tables (context CSVs)
def read52(name):
    return pd.read_csv(os.path.join(CTX, "section_5.2_data", name))


def read53(name):
    return pd.read_csv(os.path.join(CTX, "section_5.3_data", name))


SEGSHORT = {"total": "total", "TOTAL": "total", "gradient_updates": "GU", "rollout": "rollout", "dynamics_model_update": "fit",
            "synthetic_rollout_generation": "synth", "world_model_pretrain": "pretrain"}
SEGNOTE = ("Segments: GU = \\texttt{gradient\\_updates}, fit = \\texttt{dynamics\\_model\\_update}, synth = \\texttt{synthetic\\_rollout\\_generation}, "
           "pretrain = \\texttt{world\\_model\\_pretrain}.")


def table_segment_stats():
    d = pd.concat([read52("d2_total_correlation.csv"), read52("d3_segment_correlation.csv"), read52("d4_per_algo_total_correlation.csv"),
                   read52("d5_per_algo_segment_correlation.csv")], ignore_index=True)
    d["lvl"] = d["level"].map({"pooled": 0, "per_algo": 1})
    d["aord"] = d.algo.map({"all": 0, "sac": 1, "td3": 2, "mbpo": 3, "tdmpc2": 4})
    sord = {"total": 0, "rollout": 1, "gradient_updates": 2, "dynamics_model_update": 3, "synthetic_rollout_generation": 4, "world_model_pretrain": 5}
    d["sord"] = d.segment.map(sord)
    d["eord"] = d.env.map({e: i for i, e in enumerate(ENVS)})
    d = d.sort_values(["eord", "lvl", "aord", "sord"])
    body, prev = [], None
    for _, r in d.iterrows():
        key = (r.env, r.level)
        if prev is not None and key != prev:
            body.append("    " + BS + "midrule")
        pe = "" if pd.isna(r.spearman_p_exact_perm) else num(r.spearman_p_exact_perm, 3)
        body.append("    " + " & ".join([ENVSHORT[r.env], "pooled" if r.level == "pooled" else NAME[r.algo], SEGSHORT[r.segment], BS + "num{%d}" % r.n,
                                          num(r.spearman_rho, 5), num(r.spearman_p, 3), pe or "---", ms(r.loglog_slope, r.loglog_slope_se, 5), num(r.loglog_r2, 5),
                                          num(r.flops_range_orders, 4)]) + " " + BS + BS)
        prev = key
    cap = ("Rank correlation and log-log fit of energy on FLOPs for every algorithm--segment combination (one point per configuration: the seed mean). "
           "Pooled rows combine the algorithms of an environment, the other rows use the configurations of one algorithm. $p$ is the two-sided asymptotic $p$-value of the rank correlation, "
           "$p_{\\mathrm{exact}}$ the exact permutation $p$-value (given for the per-algorithm rows). The slope is that of $\\log_{10}E$ on $\\log_{10}F$ (with its standard error), "
           "span the range of the FLOPs in orders of magnitude. " + SEGNOTE + " HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr = "Env. & Scope & Segment & $n$ & $\\rho$ & $p$ & $p_{\\mathrm{exact}}$ & Slope & $R^2$ & Span " + BS + BS
    return longtable("tab:app-segment-stats", cap, "@{}lllrrrrrrr@{}", hdr, body, 10, size="scriptsize")


def table_fits(basis):
    f = read52("a1_linear_fits.csv")
    f = f[f.fit_basis == basis].copy()
    f["aord"] = f.algo.map({"sac": 0, "td3": 1, "mbpo": 2, "tdmpc2": 3})
    f["eord"] = f.env.map({e: i for i, e in enumerate(ENVS)})
    f["wo"] = f.sweep.map({"UTD": 0, "width": 1, "batch": 2, "num_q": 3, "horizon": 4})
    f["so"] = f.segment.map({"total": 0, "rollout": 1, "gradient_updates": 2, "dynamics_model_update": 3})
    f = f.sort_values(["aord", "eord", "wo", "so"])
    body, prev = [], None
    for _, r in f.iterrows():
        key = (r.algo, r.env)
        if prev is not None and key[0] != prev[0]:
            body.append("    " + BS + "midrule")
        nflag = BS + "num{%d}" % r.n + ("$^{\\dagger}$" if r.n == 3 else "")
        body.append("    " + " & ".join([NAME[r.algo] if key != prev else "", ENVSHORT[r.env] if key != prev else "", cfg_label(r.sweep.replace("_q", "_q")) if r.sweep == "num_q" else r.sweep,
                                          SEGSHORT[r.segment], nflag, num(r.a_J / 1e3, 6), num(r.b_pJ_per_FLOP, 6), num(r.r2, 6), num(r.rse_J / 1e3, 6)]) + " " + BS + BS)
        prev = key
    what = "configuration means (one point per sweep value)" if basis == "config_means" else "the individual runs (five points per sweep value)"
    cap = (f"Fixed and marginal cost of every segment along every sweep: least-squares fit $E=a+bF$ to {what}. $a$ is the intercept in kJ, $b$ the slope in pJ/FLOP, "
           "RSE the residual standard error in kJ. " + ("A sweep has $n=3$ points (TD3 batch size: $n=4$), so the fit has one (two) residual degrees of freedom; fits with $n=3$ are marked with $^{\\dagger}$ and $R^2$ is high by construction. "
                                                         if basis == "config_means" else "") + SEGNOTE + " HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr = "Algorithm & Env. & Sweep & Segment & $n$ & $a$ [kJ] & $b$ [pJ/FLOP] & $R^2$ & RSE [kJ] " + BS + BS
    lab = "tab:app-fits-means" if basis == "config_means" else "tab:app-fits-runs"
    return longtable(lab, cap, "@{}llllrrrrr@{}", hdr, body, 9, size="scriptsize")


def table_pairs():
    f = read52("a2_delta_pairs.csv").copy()
    f["aord"] = f.algo.map({"sac": 0, "td3": 1, "mbpo": 2, "tdmpc2": 3})
    f["eord"] = f.env.map({e: i for i, e in enumerate(ENVS)})
    f["wo"] = f.sweep.map({"UTD": 0, "width": 1, "batch": 2, "num_q": 3, "horizon": 4})
    f["so"] = f.segment.map({"total": 0, "rollout": 1, "gradient_updates": 2, "dynamics_model_update": 3})
    f = f.sort_values(["aord", "wo", "so", "eord"], kind="stable")
    body, prev = [], None
    for _, r in f.iterrows():
        key = (r.algo, r.sweep, r.segment)
        if prev is not None and key[0] != prev[0]:
            body.append("    " + BS + "midrule")
        fv = lambda x: ("%g" % x)
        interval = "$%s \\to %s$" % (fv(r.from_value), fv(r.to_value))
        dropped = str(r.status).startswith("dropped")
        unrel = (r.algo == "mbpo" and r.segment == "total")
        body.append("    " + " & ".join([NAME[r.algo] if key[0] != (prev or [None])[0] else "", cfg_label(r.sweep) if r.sweep == "num_q" else r.sweep, SEGSHORT[r.segment],
                                          ENVSHORT[r.env], interval, num(r.dE_J, 6), num(r.dF, 6),
                                          ("dropped" if dropped else num(r.dE_per_dF_pJ, 6) + ("$^{\\dagger}$" if unrel else ""))]) + " " + BS + BS)
        prev = key
    cap = ("Marginal cost $\\Delta E/\\Delta F$ between adjacent sweep values for every algorithm, sweep and segment (configuration means, in pJ/FLOP). "
           "Pairs whose FLOPs differ by less than \\SI{1}{\\percent} of the reference FLOPs are dropped. $^{\\dagger}$: the pairs of the MBPO totals are not reliable, "
           "because they include the energy of the dynamics model, which does not follow the swept factor; some are negative on Ant-v5. " + SEGNOTE + " HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr = "Algorithm & Sweep & Segment & Env. & Interval & $\\Delta E$ [J] & $\\Delta F$ & $\\Delta E/\\Delta F$ [pJ/FLOP] " + BS + BS
    return longtable("tab:app-pairs", cap, "@{}lllllrrr@{}", hdr, body, 8, size="scriptsize")


def table_rob_sd():
    o = read53("o1_orderings.csv").copy()
    sord = {s: i for i, s in enumerate(["baseline", "utd2", "utd4", "w256", "w512", "b256", "b512", "b1024"])}
    o["so"] = o.sweep.map(sord)
    o["eo"] = o.env.map({e: i for i, e in enumerate(ENVS)})
    o["mo"] = o.metric.map({"energy": 0, "jpf": 1})
    o = o.sort_values(["mo", "eo", "so", "rank"])
    body, prev = [], None
    for _, r in o.iterrows():
        key = (r.metric, r.env, r.sweep)
        if prev is not None and key[:2] != prev[:2]:
            body.append("    " + BS + "midrule")
        first = key != prev
        f = 1.0 if r.metric == "energy" else 1e12
        body.append("    " + " & ".join([ENVSHORT[r.env] if first else "", r.sweep_label if first else "", BS + "num{%d}" % r["rank"], NAME[r.algo],
                                          ms(r["mean"] * f, r["sd"] * f, 6), num(r.cv_pct, 6), (num(r.ratio_to_previous, 6) if not pd.isna(r.ratio_to_previous) else "---"),
                                          (abbrev_order(r.ordering) if first else "")]) + " " + BS + BS)
        prev = key
    def sub(metric, unit):
        pass
    cap = ("Algorithms ordered by total energy (J) and by total J/FLOP (pJ) at every sweep value that the algorithms share, with the standard deviation over the five seeds and the coefficient of variation (CV, in percent). "
           "TD3 is at its baseline batch size \\num{100} in the baseline, update-to-data and width rows (batch-size rows: TD3 at the swept size). Ratio is the value divided by that of the previous rank. "
           "The ordering is given with S = SAC, T = TD3, M = MBPO, from the lowest to the highest value. "
           "The first block (energy) and the second block (J/FLOP) belong to \\texttt{tab:rob-energy} and \\texttt{tab:rob-jpf}. HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr = "Env. & Sweep value & Rank & Algorithm & Mean $\\pm$ sd & CV [\\%] & Ratio to previous & Ordering " + BS + BS
    return longtable("tab:app-rob-sd", cap, "@{}llrlrrrl@{}", hdr, body, 8, size="scriptsize")


def abbrev_order(s):
    return str(s).replace("td3", "T").replace("sac", "S").replace("mbpo", "M").replace(" < ", "<")


def table_seed_orderings():
    o = read53("o3_seed_level.csv").copy()
    sord = {s: i for i, s in enumerate(["baseline", "utd2", "utd4", "w256", "w512", "b256", "b512", "b1024"])}
    o["so"] = o.sweep.map(sord)
    o["eo"] = o.env.map({e: i for i, e in enumerate(ENVS)})
    o["mo"] = o.metric.map({"energy": 0, "jpf": 1})
    o = o.sort_values(["mo", "eo", "so"])
    lbl = {"baseline": "baseline", "utd2": "UTD 2", "utd4": "UTD 4", "w256": "width 256", "w512": "width 512", "b256": "batch 256", "b512": "batch 512", "b1024": "batch 1024"}
    body, prev = [], None
    for _, r in o.iterrows():
        key = (r.metric, r.env)
        if prev is not None and key != prev:
            body.append("    " + BS + "midrule")
        body.append("    " + " & ".join([("energy" if r.metric == "energy" else "J/FLOP") if key != prev else "", ENVSHORT[r.env] if key != prev else "", lbl[r.sweep],
                                          abbrev_order(r.mean_ordering), BS + "num{%d}" % r.seeds_same_as_mean_ordering, abbrev_order(r.baseline_mean_ordering),
                                          BS + "num{%d}" % r.seeds_same_as_baseline_ordering] + [abbrev_order(r[f"seed_{s}"]) for s in SEEDS]) + " " + BS + BS)
        prev = key
    cap = ("Orderings of SAC (S), TD3 (T) and MBPO (M) by total energy and by total J/FLOP, from lowest to highest, at every sweep value: the ordering of the seed means, "
           "the number of seeds (of five) whose own ordering is the same, the ordering at the baseline and the number of seeds whose ordering equals it, and the ordering of each seed. "
           "TD3 is at its baseline batch size \\num{100} in the baseline, update-to-data and width rows. HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr = ("Metric & Env. & Sweep value & Mean order & $n$ & Base order & $n$ & "
           + " & ".join(f"{s}" for s in SEEDS) + " " + BS + BS)
    return longtable("tab:app-seed-orderings", cap, "@{}lllllllccccc@{}", hdr, body, 12, size="scriptsize")


def table_pairwise():
    o = read53("o3b_pairwise_seed_counts.csv").copy()
    sord = {s: i for i, s in enumerate(["baseline", "utd2", "utd4", "w256", "w512", "b256", "b512", "b1024"])}
    o["so"] = o.sweep.map(sord)
    o["eo"] = o.env.map({e: i for i, e in enumerate(ENVS)})
    o["mo"] = o.metric.map({"energy": 0, "jpf": 1})
    o = o.sort_values(["mo", "eo", "so", "lower_algo", "higher_algo"])
    lbl = {"baseline": "baseline", "utd2": "UTD 2", "utd4": "UTD 4", "w256": "width 256", "w512": "width 512", "b256": "batch 256", "b512": "batch 512", "b1024": "batch 1024"}
    body, prev = [], None
    for _, r in o.iterrows():
        key = (r.metric, r.env)
        if prev is not None and key != prev:
            body.append("    " + BS + "midrule")
        body.append("    " + " & ".join([("energy" if r.metric == "energy" else "J/FLOP") if key != prev else "", ENVSHORT[r.env] if key != prev else "", lbl[r.sweep],
                                          NAME[r.lower_algo] + " $<$ " + NAME[r.higher_algo], BS + "num{%d}" % r.seeds_lower_below_higher, num(r.min_seed_ratio, 6), num(r.max_seed_ratio, 6)])
                    + " " + BS + BS)
        prev = key
    cap = ("Pairwise seed counts: for every pair of algorithms, the algorithm with the lower seed mean is named first, and the table gives the number of seeds (of five) in which it also has the lower value, "
           "and the smallest and largest seed-wise ratio of the higher to the lower mean-ordered algorithm (ratios of values of the same seed). A ratio below one means that the pair swaps in that seed. "
           "TD3 is at its baseline batch size \\num{100} in the baseline, update-to-data and width rows. HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr = "Metric & Env. & Sweep value & Pair & Seeds & Min.\\ ratio & Max.\\ ratio " + BS + BS
    return longtable("tab:app-pairwise", cap, "@{}lllrrrr@{}", hdr, body, 7, size="scriptsize")


def table_cross_seed(A):
    body, prev = [], None
    for algo in ALGOS:
        for env in ENVS:
            for cfg in ORDER[algo]:
                if (algo, env, cfg) not in A:
                    continue
                t = A[(algo, env, cfg)]["TOTAL"]
                if prev is not None and algo != prev:
                    body.append("    " + BS + "midrule")
                body.append("    " + " & ".join([ENVSHORT[env], NAME[algo], cfg_label(cfg), ms(*msd(t["E"]), 5), ms(*msd(t["T"]), 5), ms(*msd(t["P"]), 5),
                                                  num(float(np.mean(t["F"])), 5), ms(float(np.mean(t["E"]) / np.mean(t["F"]) * 1e12), float(np.std(t["jpf_runs"] * 1e12, ddof=1)), 5)]) + " " + BS + BS)
                prev = algo
    cap = ("The cross-seed table of all 56 configurations: training total (\\texttt{TOTAL\\_MEASURED\\_TRAINING}) over the five canonical seeds (mean $\\pm$ sd; J/FLOP is the ratio of the mean energy "
           "to the mean FLOPs with the sd of the per-seed ratios; gross energy). HC is HalfCheetah-v5, Ant is Ant-v5.")
    hdr = "Env. & Algorithm & Config. & Energy [J] & Duration [s] & $\\bar P$ [W] & FLOPs & J/FLOP [pJ] " + BS + BS
    return longtable("tab:app-cross-seed", cap, "@{}lllrrrrr@{}", hdr, body, 8, size="scriptsize")


# ---------------------------------------------------------------------------------------------- optional variants
def table_rob_tdmpc2():
    t = read53("t1_tdmpc2_sweep.csv")
    cfgs = ["numq3", "base", "numq7", "horizon1", "horizon5"]
    body = []
    for env in ENVS:
        for c in cfgs:
            r = t[(t.env == env) & (t.config == c)].iloc[0]
            body.append("    " + " & ".join([env if c == cfgs[0] else "", tt(c), BS + "num{%d}" % r.num_q, BS + "num{%d}" % r.horizon, ms(r.E_mean, r.E_sd, 6),
                                              ms(r.jpf_mean * 1e12, r.jpf_sd * 1e12, 6), ms(r.rollout_share_of_total_mean * 100, r.rollout_share_of_total_sd * 100, 6)]) + " " + BS + BS)
        if env != ENVS[-1]:
            body.append("    " + BS + "midrule")
    s = []
    s.append(BS + "begin{table}[tbp]")
    s.append("  " + BS + "centering")
    s.append("  " + BS + "footnotesize")
    s.append("  " + BS + "caption{TD-MPC2: total energy, J/FLOP of the training total and share of the " + tt("rollout") + " in the training energy for the five configurations (mean $\\pm$ sd over five seeds, gross; full precision). "
             "The baseline is " + tt("base") + "; the " + tt("num_q") + " sweep varies the first, the horizon sweep the second parameter.}")
    s.append("  " + BS + "label{tab:app-rob-tdmpc2}")
    s.append("  " + BS + "begin{tabular}{@{}llrrrrr@{}}")
    s.append("    " + BS + "toprule")
    s.append("    Environment & Config. & " + tt("num_q") + " & Horizon & Energy [J] & J/FLOP [pJ] & Rollout share [" + BS + "%] " + BS + BS)
    s.append("    " + BS + "midrule")
    s.extend(body)
    s.append("    " + BS + "bottomrule")
    s.append("  " + BS + "end{tabular}")
    s.append(BS + "end{table}")
    return "\n".join(s) + "\n"


def table_rob_ranges():
    t = read53("t3_algo_ranges_vs_tdmpc2.csv")
    body = []
    for env in ENVS:
        for algo in ["sac", "td3", "mbpo", "tdmpc2"]:
            r = t[(t.env == env) & (t.algo == algo)]
            if r.empty:
                continue
            r = r.iloc[0]
            yn = lambda v: "--" if algo == "tdmpc2" else ("yes" if bool(v) else "no")
            body.append("    " + " & ".join([env if algo == "sac" else "", NAME[algo], BS + "num{%d}" % r.n_configs, num(r.E_min, 6) + "--" + num(r.E_max, 6), yn(r.E_range_overlaps_tdmpc2),
                                              num(r.jpf_min * 1e12, 6) + "--" + num(r.jpf_max * 1e12, 6), yn(r.jpf_range_overlaps_tdmpc2)]) + " " + BS + BS)
        if env != ENVS[-1]:
            body.append("    " + BS + "midrule")
    s = []
    s.append(BS + "begin{table}[tbp]")
    s.append("  " + BS + "centering")
    s.append("  " + BS + "footnotesize")
    s.append("  " + BS + "caption{Range of the configuration values (seed means) of each algorithm over all its configurations, and whether it overlaps the range of TD-MPC2 over its five configurations (full precision). "
             "\\#cfg is the number of configurations of the algorithm on the environment.}")
    s.append("  " + BS + "label{tab:app-rob-ranges}")
    s.append("  " + BS + "resizebox{" + BS + "textwidth}{!}{%")
    s.append("    " + BS + "begin{tabular}{@{}llrrcrc@{}}")
    s.append("      " + BS + "toprule")
    s.append("      Environment & Algorithm & " + BS + "#cfg & Energy range [J] & Overlap with TD-MPC2 & J/FLOP range [pJ] & Overlap with TD-MPC2 " + BS + BS)
    s.append("      " + BS + "midrule")
    s.extend("  " + b for b in body)
    s.append("      " + BS + "bottomrule")
    s.append("    " + BS + "end{tabular}}")
    s.append(BS + "end{table}")
    return "\n".join(s) + "\n"


# ---------------------------------------------------------------------------------------------- verification against thesis-main.tex
NUM_RE = re.compile(r"\\num\{([^{}]*)\}")


def sigdigits(tok: str):
    """Possible numbers of significant digits of a printed token (trailing zeros of an integer are ambiguous)."""
    t = tok.strip().lstrip("+-")
    t = re.sub(r"[eE][+-]?\d+$", "", t)
    d = t.replace(".", "").lstrip("0")
    n = max(1, len(d))
    if "." not in t and tok.strip().lstrip("+-").isdigit():
        return list(range(max(1, len(d.rstrip("0"))), n + 1))
    return [n]


def strip_caption(block: str) -> str:
    i = block.find("\\caption{")
    if i < 0:
        return block
    j, depth = i + len("\\caption{"), 1
    while j < len(block) and depth:
        depth += {"{": 1, "}": -1}.get(block[j], 0)
        j += 1
    return block[:i] + block[j:]


def tokens_of(text: str):
    """All numeric tokens of \\num{...}; 'a+-b' gives two tokens."""
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group(1).replace("{,}", "")
        for part in s.split("+-"):
            try:
                float(part)
            except ValueError:
                continue
            out.append(part)
    return out


def pool_from(texts):
    pool = {}
    for tx in texts:
        for tok in tokens_of(tx):
            v = float(tok)
            # a printed value is the true value rounded to its own digits: allow for that rounding (half a unit of the last digit)
            nd = min(max(sigdigits(tok)), 6)
            half = 0.5 * 10 ** (math.floor(math.log10(abs(v))) - nd + 1) if v != 0 else 0.0
            for vv in (abs(v), abs(v) - half, abs(v) + half):
                for n in (1, 2, 3, 4, 5, 6):
                    key = (n, float(f"{vv:.{n}g}"))
                    pool.setdefault(key, []).append(abs(v))
    return pool


def table_block(tex: str, label: str):
    i = tex.index("\\label{" + label + "}")
    j0 = max(tex.rfind("\\begin{table}", 0, i), tex.rfind("\\begin{longtable}", 0, i), tex.rfind("\\begin{table*}", 0, i))
    ends = [e for e in (tex.find("\\end{table}", i), tex.find("\\end{longtable}", i)) if e > 0]
    return tex[j0:min(ends)]


def verify(tex: str, pairs: dict) -> list:
    """pairs: label -> list of generated LaTeX strings forming the candidate pool."""
    rows = []
    for label, texts in pairs.items():
        try:
            block = table_block(tex, label)
        except ValueError:
            rows.append(dict(label=label, tokens=0, matched=0, unmatched=0, max_rel_dev=float("nan"), note="label not found"))
            continue
        pool = pool_from(texts)
        toks = tokens_of(strip_caption(block))
        matched, unmatched, maxdev = 0, [], 0.0
        for tok in toks:
            v = abs(float(tok))
            if v == 0:
                matched += 1
                continue
            cands = []
            ns = [min(n, 6) for n in sigdigits(tok)]
            for n in ns:
                cands += pool.get((n, v), [])
            if cands:
                matched += 1
                if min(ns) >= 4:
                    maxdev = max(maxdev, min(abs(v - c) / c for c in cands))
            else:
                unmatched.append(tok)
        rows.append(dict(label=label, tokens=len(toks), matched=matched, unmatched=len(unmatched), max_rel_dev=maxdev, note=" ".join(unmatched[:12])))
    return rows


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CTX, "appendix", "appendix_data", "B"))
    _t = os.path.join(os.path.dirname(os.path.dirname(REPO)), "Thesis Report", "thesis-main.tex")
    ap.add_argument("--thesis", default=(_t + ".bak-appendix") if os.path.exists(_t + ".bak-appendix") else _t)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs()
    A = agg(runs)
    assert runs._chk_E.max() < 1e-12
    frags = OrderedDict()
    comp_dev = {}
    A_td3_base = {k: v for k, v in A.items() if not (k[0] == "td3" and k[2] != "baseline")}
    A_td3_sweep = {k: v for k, v in A.items() if k[0] == "td3" and k[2] != "baseline"}
    for algo in ALGOS:
        Ause = A_td3_base if algo == "td3" else A
        frags[f"{algo}_energy"] = table_energy(Ause, algo)
        frags[f"{algo}_jpf"] = table_jpf(Ause, algo)
        frags[f"{algo}_time"] = table_time(Ause, algo)
        frags[f"{algo}_comp"], comp_dev[algo] = table_comp(runs, Ause, algo)
        frags[f"{algo}_quality"] = table_quality(runs, A, algo)
    frags["mbpo_fit"] = table_mbpo_fit(runs)
    frags["mbpo_sac"] = table_mbpo_vs_sac(A)
    frags["sac_env"] = table_env_tex("sac", "tab:app-sac-env")
    frags["mbpo_env"] = table_env_tex("mbpo", "tab:app-mbpo-env", "MBPO and the environment are confounded through the rollout schedule (fixed length 1 on HalfCheetah-v5, scheduled maximum 25 on Ant-v5).")
    frags["segment_stats"] = table_segment_stats()
    frags["fits_means"] = table_fits("config_means")
    frags["fits_runs"] = table_fits("per_run")
    frags["pairs"] = table_pairs()
    frags["rob_sd"] = table_rob_sd()
    frags["seed_orderings"] = table_seed_orderings()
    frags["pairwise"] = table_pairwise()
    frags["cross_seed"] = table_cross_seed(A)
    # optional variants
    frags["opt_rob_tdmpc2"] = table_rob_tdmpc2()
    frags["opt_rob_ranges"] = table_rob_ranges()
    frags["opt_td3_env"] = table_env_tex("td3", "tab:app-td3-env", "The TD3 baseline, update-to-data and width rows use TD3 at its baseline batch size of 100, which is not matched to SAC and MBPO; the batch-size rows are matched.")
    frags["opt_tdmpc2_env"] = table_env_tex("tdmpc2", "tab:app-tdmpc2-env", "TD-MPC2 and the environment are confounded through the termination classifier (active on Ant-v5 only).")
    frags["opt_td3_sweep_energy"] = table_energy(A_td3_sweep, "td3").replace("tab:app-td3-energy", "tab:app-td3-sweep-energy")
    frags["opt_td3_sweep_jpf"] = table_jpf(A_td3_sweep, "td3").replace("tab:app-td3-jpf", "tab:app-td3-sweep-jpf")
    for k, v in frags.items():
        with open(os.path.join(args.out, f"B_{k}.tex"), "w", encoding="utf-8") as f:
            f.write(v if isinstance(v, str) else v[0])
    # long CSV of the aggregates
    recs = []
    for (algo, env, cfg), d in A.items():
        for s, r in d.items():
            recs.append(dict(algo=algo, env=env, cfg=cfg, segment=s, E_mean=np.mean(r["E"]), E_sd=np.std(r["E"], ddof=1), share_total_mean=np.mean(r["share_total"]),
                             share_total_sd=np.std(r["share_total"], ddof=1), share_gu_mean=(np.mean(r["share_gu"]) if r["share_gu"] is not None else np.nan),
                             F_mean=np.nanmean(r["F"]) if not np.all(np.isnan(r["F"])) else np.nan, jpf_ratio_of_means=(np.mean(r["E"]) / np.nanmean(r["F"]) if not np.all(np.isnan(r["F"])) else np.nan),
                             jpf_sd=(np.std(r["jpf_runs"], ddof=1) if not np.all(np.isnan(r["F"])) else np.nan), T_mean=np.mean(r["T"]), T_sd=np.std(r["T"], ddof=1),
                             P_mean=np.mean(r["P"]), gflops_mean=np.nanmean(r["gflops"]) if not np.all(np.isnan(r["gflops"])) else np.nan))
    pd.DataFrame(recs).to_csv(os.path.join(args.out, "B_config_segment_stats.csv"), index=False)
    # verification against the tables printed in the thesis
    if os.path.exists(args.thesis):
        tex = open(args.thesis, encoding="utf-8").read()
        pools = {}
        for algo in ALGOS:
            pool_texts = [frags[f"{algo}_{k}"] for k in ("energy", "jpf", "time", "comp", "quality")]
            if algo == "td3":
                pool_texts += [frags["opt_td3_sweep_energy"], frags["opt_td3_sweep_jpf"]]
            if algo == "mbpo":
                pool_texts += [frags["mbpo_fit"], frags["mbpo_sac"], frags["mbpo_env"]]
            if algo == "sac":
                pool_texts += [frags["sac_env"]]
            if algo == "tdmpc2":
                pool_texts += [frags["opt_tdmpc2_env"]]
            for suffix in ("base-energy", "base-jpf", "base-time", "base-comp", "quality", "utd-energy", "utd-jpf", "width-energy", "width-jpf", "batch-energy", "batch-jpf",
                           "fit", "rollout-energy", "rollout-jpf", "numq-energy", "numq-jpf", "hor-energy", "hor-jpf", "pretrain", "env"):
                lab = f"tab:{algo}-{suffix}"
                if ("\\label{" + lab + "}") in tex:
                    pools[lab] = pool_texts
        pools["tab:mbpo-sac"] = [frags["mbpo_sac"]]
        for lab, key in (("tab:ec-pooled", "segment_stats"), ("tab:ec-algo", "segment_stats"), ("tab:ec-fits", "fits_means"), ("tab:ec-delta", "pairs"),
                         ("tab:rob-energy", "rob_sd"), ("tab:rob-jpf", "rob_sd"), ("tab:rob-tdmpc2", "opt_rob_tdmpc2"), ("tab:rob-ranges", "opt_rob_ranges"),
                         ("tab:spearman", None)):
            if key:
                pools[lab] = [frags[key]]
        rows = verify(tex, pools)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "B_verification.csv"), index=False)
        tot = sum(r["tokens"] for r in rows)
        mt = sum(r["matched"] for r in rows)
        print(f"verification: {len(rows)} thesis tables, {tot} numeric tokens, {mt} matched, max relative deviation over matched tokens "
              f"{max([r['max_rel_dev'] for r in rows if not math.isnan(r['max_rel_dev'])] or [float('nan')]):.3g}")
    print("component-sum deviation (CPU+GPU+RAM vs task energy):", {k: f"{v:.2e}" for k, v in comp_dev.items()})
    print("written", len(frags), "fragments to", args.out)


if __name__ == "__main__":
    main()
