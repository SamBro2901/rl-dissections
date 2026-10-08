"""
Shared data layer for the generators of contexts/Section_4.2.md (TD3), Section_4.3.md (MBPO) and
Section_4.4.md (TD-MPC2).  (Lives in Section_4.2_data/ because it was written first; 4.3 / 4.4 import it.)

Definitions are copied from contexts/section_4.1_data/build_section_4_1.py (the generator of Section 4.1, SAC) so that
the numbers of 4.2-4.4 are computed exactly like those of 4.1:
  * energy = gross, J = kWh * 3.6e6, taken from segment_energy.json; durations of measured tasks = sum of the CodeCarbon
    per-task `duration` of the per-task CSV (emissions_<experiment>_<run_id>.csv); allocated sub-segment durations =
    segment_energy.json["_sub_segment_wall_time_seconds"];
  * mean power = energy / duration (per seed, then mean +- sd);
  * idle-floor fraction = mean(P_idle_head, P_idle_tail) * duration / energy (as-recorded idle powers);
  * allocation coverage = sum of the four perf_counter times / summed gradient_updates task duration;
  * J/FLOP = ratio of means (mean energy / mean FLOPs), plus the mean/sd of the per-seed ratios;
  * outlier flag = |x - median| > 3 x MAD (unscaled), n = 5 per configuration;
  * exact paired sign-flip test over the 2^5 sign patterns;
  * marginal-cost fit = OLS of per-run (FLOPs, energy) points of a sweep.

Read-only on results/, flop_analysis/ and git.  Writes nothing by itself.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import glob
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flop_analysis"))

KWH_TO_J = 3.6e6
CANON = [331, 958, 14577, 43611, 85062]
ENVS = [("HC", "HalfCheetah-v5"), ("Ant", "Ant-v5")]
ENV_SHORT = {e: s for s, e in ENVS}
SUBS = ["buffer_sample", "critic_update", "actor_update", "target_update"]
SEG_STATUS = {"rollout": "measured", "gradient_updates": "measured", "buffer_sample": "allocated",
              "critic_update": "allocated", "actor_update": "allocated", "target_update": "allocated",
              "dynamics_model_update": "measured", "synthetic_rollout_generation": "measured",
              "world_model_pretrain": "measured", "TOTAL_MEASURED_TRAINING": "aggregate",
              "warmup": "excluded", "idle_baseline_head": "excluded", "idle_baseline_tail": "excluded"}
SEG_SHORT = {"rollout": "rollout", "buffer_sample": "buf_sample", "critic_update": "critic", "actor_update": "actor",
             "target_update": "target", "gradient_updates": "GU", "TOTAL_MEASURED_TRAINING": "TOTAL",
             "dynamics_model_update": "dyn_model", "synthetic_rollout_generation": "synth_rollout",
             "world_model_pretrain": "wm_pretrain", "warmup": "warmup", "idle_baseline_head": "idle_head",
             "idle_baseline_tail": "idle_tail"}
SEP = " ‖ "   # separates the HalfCheetah-v5 value (left) from the Ant-v5 value (right) inside one cell

CHECKS: List[Tuple[str, bool, str]] = []
DISCREPANCIES: List[str] = []
NOT_AVAILABLE: List[str] = []
QUESTIONS: List[str] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    return f"**{'PASS' if ok else 'FAIL'}** — {name}" + (f" ({detail})" if detail else "")


def disc(msg):
    DISCREPANCIES.append(msg)


def na(msg):
    NOT_AVAILABLE.append(msg)


def question(msg):
    QUESTIONS.append(msg)


# ----------------------------------------------------------------------------- formatting
def gN(x, n=6):
    """n significant digits; positional for 1e-3 <= |x| < 1e6, scientific otherwise."""
    if x is None:
        return "NaN"
    try:
        if isinstance(x, (bool, np.bool_)):
            return str(bool(x))
        if isinstance(x, (int, np.integer)):
            return str(int(x)) if abs(int(x)) < 10 ** n else f"{float(x):.{n - 1}e}"
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "NaN"
    if math.isinf(xf):
        return "inf" if xf > 0 else "-inf"
    if xf == 0:
        return "0"
    ax = abs(xf)
    if 1e-3 <= ax < 1e6:
        digits = n - 1 - int(math.floor(math.log10(ax)))
        rounded = round(xf, digits)
        if rounded != 0 and int(math.floor(math.log10(abs(rounded)))) != int(math.floor(math.log10(ax))):
            digits -= 1
            rounded = round(xf, digits)
        return f"{rounded:.{max(digits, 0)}f}"
    return f"{xf:.{n - 1}e}"


def g6(x):
    return gN(x, 6)


def _clean(vals):
    return np.asarray([v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))], float)


def msd(vals):
    a = _clean(vals)
    if len(a) == 0:
        return "NaN"
    sd = a.std(ddof=1) if len(a) > 1 else float("nan")
    return f"{g6(a.mean())} ± {g6(sd)}"


def sd1(vals):
    a = _clean(vals)
    return a.std(ddof=1) if len(a) > 1 else float("nan")


def _cell(c):
    return str(c).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    out = ["", "| " + " | ".join(_cell(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    out.append("")
    return "\n".join(out)


def code(text, lang=""):
    return f"```{lang}\n{text.rstrip()}\n```"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=REPO).stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ceil_div(a, b):
    return -(-a // b)


def signflip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    cnt = 0
    for signs in itertools.product([1, -1], repeat=len(d)):
        if abs((np.array(signs) * d).mean()) >= obs - 1e-12 * max(1.0, obs):
            cnt += 1
    return cnt / 2 ** len(d), cnt


# ----------------------------------------------------------------------------- spec
@dataclass
class Spec:
    algo: str
    section: str                       # "4.2"
    name: str                          # "TD3"
    config_cls: type
    tags: List[str]
    tag_desc: Dict[str, str]
    tag_factor: Dict[str, dict]        # tag -> {algo_config field: value} relative to the env reference config
    env_ref_override: Dict[str, Optional[str]]   # es -> override json path (the env baseline) or None
    extra_measured: List[str]          # algorithm-specific measured tasks (epoch order is given by measured_order)
    measured_order: List[str]          # all measured task names in epoch order, 'gradient_updates' included
    sweeps: List[Tuple[str, str, List[str], Optional[str]]]   # (name, factor text, tags in order, flop_dashboard x_dim)
    tag_envs: Dict[str, List[str]] = field(default_factory=dict)   # tags that exist only in some envs
    tag_script: Callable = None        # (es, tag) -> launcher script text
    tag_override_file: Callable = None  # (es, tag) -> override file text
    expected_runs: Dict[str, int] = field(default_factory=dict)
    per_epoch_tasks: List[str] = field(default_factory=list)   # measured tasks that have one task per epoch

    @property
    def seg_order(self):
        """Energy-table column order: measured tasks in epoch order with gradient_updates expanded into its 4 allocated
        sub-segments + the measured GU row, then TOTAL."""
        out = []
        for t in self.measured_order:
            if t == "gradient_updates":
                out += SUBS + ["gradient_updates"]
            else:
                out.append(t)
        return out + ["TOTAL_MEASURED_TRAINING"]

    @property
    def train_segments(self):
        return [t for t in self.measured_order if t != "gradient_updates"] + SUBS

    def cfg_envs(self, tag):
        return self.tag_envs.get(tag, ["HC", "Ant"])


def label(es, tag):
    return f"{es}-{tag}"


# ----------------------------------------------------------------------------- FLOP JSON
with open("flop_analysis/flops_per_call.json", encoding="utf-8") as _f:
    FLOPS_JSON = json.load(_f)
from flop_keys import signature, mbpo_rollout_regime  # noqa: E402
from configs.config import ALGO_CONFIGS, ExperimentConfig  # noqa: E402


def defaults_of(algo):
    return json.loads(json.dumps(dataclasses.asdict(ALGO_CONFIGS[algo]())))


def load_all_metadata():
    out = []
    for m in sorted(Path("results").glob("*/*/seed_*/*/metadata.json")):
        with open(m, encoding="utf-8") as f:
            out.append((m.parent, json.load(f)))
    return out


ALL_META = load_all_metadata()

SWEEP_LOGS = sorted(glob.glob("results/*.log") + glob.glob("results/*/*.log") + glob.glob("results/*/*/*.log"))


def sweep_log_blocks():
    """run_dir -> (log file, text block) from every sweep log, split on '[i/N] Starting:' lines."""
    blocks = {}
    for lf in SWEEP_LOGS:
        txt = Path(lf).read_text(encoding="utf-8", errors="replace").splitlines()
        starts = [i for i, ln in enumerate(txt) if re.search(r"\] \[\d+/\d+\] Starting:", ln)]
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(txt)
            blk = "\n".join(txt[s:e])
            m = re.search(r"Run directory: (\S+)", blk)
            if m:
                blocks.setdefault(m.group(1), []).append((Path(lf).as_posix(), blk))
    return blocks


SWEEP_BLOCKS = sweep_log_blocks()

GREP_STRINGS = [
    "default power consumption of 4 W per thread", "Resorting to a default power consumption", "No CPU tracking mode found",
    "Unable to read RAPL value", "Unable to read max_energy_range_uj", "Unable to access geographical location",
    "Using 'Canada' as the default value", "Thermal gate timed out", "Thermal reference did not stabilize",
    "No NVML reference available", "NVML read failed", "Command failed (", "Command not found", "Command timed out",
    "Could not query GPU clock range", "Could not read current CPU governor", "CUDA requested but not available",
    "Permission denied", "permission error", "Insufficient Permissions",
]


def task_rows(tasks, prefix):
    if prefix in ("idle_baseline_head", "idle_baseline_tail"):
        return tasks[tasks["task_name"] == prefix]
    return tasks[tasks["task_name"].str.fullmatch(rf"{re.escape(prefix)}_\d+")]


# ----------------------------------------------------------------------------- FLOPs (independent recomputation)
def flops_for_run(algo, meta, fk, env):
    """Independent re-implementation of flop_analysis/compute_energy_per_flop.py (formulas read from the code;
    FLOP constants from flops_per_call.json). Returns (calls, F, extra)."""
    ac, ec = meta["algo_config"], meta["experiment_config"]
    spe = ec["steps_per_epoch"]
    n_env = max(1, ec["train_steps"] // spe) * spe
    n_upd = n_env * ac["updates_per_env_step"]
    extra = {}
    if algo in ("sac", "td3", "mbpo"):
        delay = ac["policy_update_delay"]
        n_act = ceil_div(n_upd, delay)
        n_tgt = n_act if algo == "td3" else n_upd
        calls = {"rollout": n_env, "buffer_sample": n_upd, "critic_update": n_upd, "actor_update": n_act,
                 "target_update": n_tgt}
        F = {"rollout": fk["actor_forward_bs1"] * n_env, "buffer_sample": 0,
             "critic_update": fk["critic_fwdbwd"] * n_upd, "actor_update": fk["actor_fwdbwd"] * n_act,
             "target_update": fk["target_update_elementwise_ops"] * n_tgt}
    elif algo == "tdmpc2":
        warm = ec["warmup_steps"]
        calls = {"rollout": n_env, "world_model_pretrain": warm, "buffer_sample": n_upd, "critic_update": n_upd,
                 "actor_update": n_upd, "target_update": n_upd}
        F = {"rollout": fk["rollout_per_env_step"] * n_env, "world_model_pretrain": fk["gradient_update_total"] * warm,
             "buffer_sample": 0, "critic_update": fk["gradient_update_critic"] * n_upd,
             "actor_update": fk["gradient_update_actor"] * n_upd,
             "target_update": fk["gradient_update_target_elementwise_ops"] * n_upd}
    else:
        raise ValueError(algo)
    F["gradient_updates"] = F["critic_update"] + F["actor_update"]
    return calls, F, extra


def mbpo_extras(meta, fk, tm):
    """MBPO dynamics-model / synthetic-rollout FLOPs and counts, recomputed from training_metrics.json epoch rows
    following EnsembleDynamicsModel.fit() / _generate_model_rollouts (see Section_4.3.md Part 6.2)."""
    ac, ec = meta["algo_config"], meta["experiment_config"]
    spe, warm, cap = ec["steps_per_epoch"], ec["warmup_steps"], ac["buffer_capacity"]
    mb = ac["model_train_batch_size"]
    per_member = fk["dynamics_member_fwdbwd"] // mb
    assert fk["dynamics_member_fwdbwd"] % mb == 0
    per_ens = fk["dynamics_ensemble_forward_all_bs1"]
    fits = []
    dyn_train = dyn_hold = steps_total = 0
    for row in tm["epochs"]:
        ep_run = row.get("model_train_epochs")
        if not ep_run:
            continue
        n_total = min(warm + row["epoch"] * spe, cap)
        n_hold = max(1, int(n_total * ac["model_holdout_ratio"]))
        n_train = n_total - n_hold
        tr = ep_run * ac["ensemble_size"] * n_train * per_member
        ho = ep_run * n_hold * per_ens
        st = ep_run * ceil_div(n_train, mb)
        dyn_train += tr
        dyn_hold += ho
        steps_total += st
        fits.append(dict(epoch=row["epoch"], fit_epochs=ep_run, n_total=n_total, n_train=n_train, n_hold=n_hold,
                         steps=st, train_flops=tr, hold_flops=ho))
    synth = sum(r.get("synthetic_transitions_generated", 0) for r in tm["epochs"])
    per_sample = fk["actor_forward_bs1"] + fk["dynamics_ensemble_forward_all_bs1"]
    return dict(fits=fits, dyn_train=dyn_train, dyn_hold=dyn_hold, dyn_steps=steps_total,
                F_dyn=dyn_train + dyn_hold, synth_samples=synth, F_synth=synth * per_sample)


# ----------------------------------------------------------------------------- dataset
class Dataset:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.algo = spec.algo
        self.defaults = defaults_of(spec.algo)
        self.ref_ac = {}
        for es, ov in spec.env_ref_override.items():
            d = dict(self.defaults)
            if ov:
                with open(ov, encoding="utf-8") as f:
                    d.update(json.load(f))
            self.ref_ac[es] = d
        self.runs: List[dict] = []
        self.excluded = Counter()
        self.unmatched = []
        self._load()
        self.runs.sort(key=lambda r: (["HC", "Ant"].index(r["es"]), spec.tags.index(r["tag"]), CANON.index(r["seed"])))

    # -- tag classification
    def tag_of(self, es, ac):
        diff = {k: v for k, v in ac.items() if self.ref_ac[es].get(k) != v}
        for t, fac in self.spec.tag_factor.items():
            if diff == fac:
                return t
        return None

    def _load(self):
        spec = self.spec
        for run_dir, md in ALL_META:
            if md["algo_name"] != spec.algo:
                continue
            if md["seed"] not in CANON or md["env_id"] not in ENV_SHORT:
                self.excluded[(md["env_id"], md["seed"])] += 1
                continue
            es = ENV_SHORT[md["env_id"]]
            tag = self.tag_of(es, md["algo_config"])
            if tag is None:
                self.unmatched.append(str(run_dir))
                continue
            task_csvs = sorted(run_dir.glob("emissions_*.csv"))
            r = dict(tag=tag, env=md["env_id"], es=es, seed=md["seed"], run_dir=run_dir.as_posix(),
                     timestamp=run_dir.name, meta=md,
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
            self._derive(r)
            self.runs.append(r)

    def _derive(self, r):
        spec = self.spec
        md, seg = r["meta"], r["seg"]
        r["sig"] = signature(spec.algo, md["algo_config"])
        r["fk"] = FLOPS_JSON[spec.algo][r["env"]][r["sig"]]
        calls, F, _ = flops_for_run(spec.algo, md, r["fk"], r["env"])
        if spec.algo == "mbpo":
            ex = mbpo_extras(md, r["fk"], r["tm"])
            r["mbpo"] = ex
            calls["dynamics_model_update"] = ex["dyn_steps"]
            calls["synthetic_rollout_generation"] = ex["synth_samples"]
            F["dynamics_model_update"] = ex["F_dyn"]
            F["synthetic_rollout_generation"] = ex["F_synth"]
            r["regime"] = mbpo_rollout_regime(md["algo_config"])
        # TOTAL FLOPs = matmul FLOPs of every training segment except target_update (elementwise) and buffer_sample (0);
        # gradient_updates is excluded here because it is already the sum critic + actor.
        F["TOTAL_MEASURED_TRAINING"] = sum(v for k, v in F.items() if k not in ("buffer_sample", "target_update", "gradient_updates"))
        r["calls"], r["F"] = calls, F
        meas = spec.measured_order
        hw = {}
        for p in meas + ["warmup", "idle_baseline_head", "idle_baseline_tail"]:
            tr = task_rows(r["tasks"], p)
            hw[p] = dict(n=len(tr), duration=tr["duration"].sum(), E=tr["energy_consumed"].sum() * KWH_TO_J,
                         cpu=tr["cpu_energy"].sum() * KWH_TO_J, gpu=tr["gpu_energy"].sum() * KWH_TO_J,
                         ram=tr["ram_energy"].sum() * KWH_TO_J)
        r["hw"] = hw
        E = {k: seg[k] * KWH_TO_J for k in spec.train_segments + ["warmup", "idle_baseline_head", "idle_baseline_tail"]}
        E["gradient_updates"] = sum(E[k] for k in SUBS)
        E["TOTAL_MEASURED_TRAINING"] = sum(E[k] for k in spec.train_segments)
        r["E"] = E
        r["E_kwh"] = {k: v / KWH_TO_J for k, v in E.items()}
        T = dict(seg["_sub_segment_wall_time_seconds"])
        D = {k: hw[k]["duration"] for k in hw}
        D.update({k: T[k] for k in SUBS})
        D["TOTAL_MEASURED_TRAINING"] = sum(hw[k]["duration"] for k in meas)
        r["D"], r["T"] = D, T
        r["P"] = {k: (E[k] / D[k] if D.get(k) else float("nan")) for k in E}
        tot = E["TOTAL_MEASURED_TRAINING"]
        shares = [t for t in spec.train_segments] + ["gradient_updates"]
        r["share"] = {k: 100 * E[k] / tot for k in shares}
        r["share_gu"] = {k: 100 * E[k] / E["gradient_updates"] for k in SUBS}
        jk = [k for k in F if k not in ("buffer_sample",)]
        r["jpf"] = {k: (E[k] / F[k] if F.get(k) else float("nan")) for k in jk if k in E}
        r["tp"] = {k: F[k] / D[k] / 1e9 for k in F if k not in ("buffer_sample", "target_update") and D.get(k)}
        r["P_idle"] = 0.5 * (r["P"]["idle_baseline_head"] + r["P"]["idle_baseline_tail"])
        r["idle_frac"] = {k: 100 * r["P_idle"] * D[k] / E[k] for k in meas + ["TOTAL_MEASURED_TRAINING"]}
        r["coverage"] = sum(T.values()) / D["gradient_updates"]
        # whole run = every row of the per-task CSV (idle head, warmup, [pretrain], training tasks, idle tail).
        # NOTE: the top-level emissions.csv `duration` is the duration of the LAST task (the 90 s idle tail), not of the run,
        # while its energy_consumed is cumulative; therefore the run power is built from the per-task rows.
        top = r["top"].iloc[0]
        r["E_run"] = r["tasks"]["energy_consumed"].sum() * KWH_TO_J
        r["D_run"] = r["tasks"]["duration"].sum()
        r["P_run"] = r["E_run"] / r["D_run"]
        r["E_run_top_dev"] = abs(top["energy_consumed"] * KWH_TO_J - r["E_run"]) / r["E_run"]
        r["top_duration"] = top["duration"]

    # -- accessors
    def runs_of(self, es, tag):
        return [r for r in self.runs if r["es"] == es and r["tag"] == tag]

    def cfgs(self):
        return [(es, t) for es, _ in ENVS for t in self.spec.tags if es in self.spec.cfg_envs(t)]

    def has(self, es, tag):
        return bool(self.runs_of(es, tag))


# ----------------------------------------------------------------------------- per-epoch table
def per_epoch_frame(ds: Dataset):
    rows = []
    for r in ds.runs:
        for task in ds.spec.per_epoch_tasks:
            tr = task_rows(r["tasks"], task)
            for _, row in tr.iterrows():
                ep = int(row["task_name"].rsplit("_", 1)[1])
                e = row["energy_consumed"] * KWH_TO_J
                rows.append(dict(tag=r["tag"], env=r["env"], seed=r["seed"], epoch=ep, task=task, energy_j=e,
                                 duration_s=row["duration"], mean_power_w=e / row["duration"],
                                 ram_integration_window_s=row["ram_energy"] * KWH_TO_J / row["ram_power"]
                                 if row["ram_power"] else None))
    return pd.DataFrame(rows).sort_values(["env", "tag", "seed", "task", "epoch"])


# ----------------------------------------------------------------------------- reference algorithm: SAC (for comparisons)
def sac_spec():
    from configs.config import SACConfig
    tags = ["base", "utd2", "utd4", "w256", "w512", "b512", "b1024"]
    return Spec(
        algo="sac", section="4.1", name="SAC", config_cls=SACConfig, tags=tags,
        tag_desc={"base": "hidden (1024,1024), batch 256, UTD 1", "utd2": "UTD 2", "utd4": "UTD 4",
                  "w256": "hidden (256,256)", "w512": "hidden (512,512)", "b512": "batch 512", "b1024": "batch 1024"},
        tag_factor={"base": {}, "utd2": {"updates_per_env_step": 2}, "utd4": {"updates_per_env_step": 4},
                    "w256": {"hidden_sizes": [256, 256]}, "w512": {"hidden_sizes": [512, 512]},
                    "b512": {"batch_size": 512}, "b1024": {"batch_size": 1024}},
        env_ref_override={"HC": None, "Ant": None}, extra_measured=[], measured_order=["rollout", "gradient_updates"],
        sweeps=[], per_epoch_tasks=["rollout", "gradient_updates"])
