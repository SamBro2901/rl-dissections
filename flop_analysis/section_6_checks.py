"""Read-only checks for Chapter 6 (Step 6, GPU clock frequency). Prints everything to stdout, writes no file.

Reads the two development runs (R2000, R200) and the five canonical MBPO width-256 runs on HalfCheetah-v5
directly from results/*, plus flop_analysis/output/per_run_energy_per_flop.csv for the FLOP counts.
Usage: python flop_analysis/section_6_checks.py > out.txt
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "flop_analysis" / "output"
J = 3.6e6
BASE = RES / "mbpo" / "HalfCheetah-v5" / "seed_0"
RUNS = {"R2000": BASE / "20260904_122958", "R200": BASE / "20260904_130915"}
CANON = {331, 958, 14577, 43611, 85062}
MEASURED = ["rollout", "dynamics_model_update", "synthetic_rollout_generation", "gradient_updates"]
ALLOC = ["buffer_sample", "critic_update", "actor_update", "target_update"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def head(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


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


def load(run_dir):
    em = pd.read_csv(glob.glob(str(run_dir / "emissions_base_*.csv"))[0])
    em["prefix"] = em["task_name"].map(prefix)
    em["energy_j"] = em["energy_consumed"] * J
    return dict(
        dir=run_dir,
        meta=json.load(open(run_dir / "metadata.json")),
        seg=json.load(open(run_dir / "segment_energy.json")),
        tm=json.load(open(run_dir / "training_metrics.json")),
        em=em,
        whole=pd.read_csv(run_dir / "emissions.csv").iloc[-1],
        log=open(run_dir / "run.log").read(),
    )


def training_totals(r):
    """Sum of the four measured task groups: energy (J), duration (s), and per-component energy (J)."""
    d = r["em"][r["em"]["prefix"].isin(MEASURED)]
    return dict(energy=d.energy_j.sum(), duration=d.duration.sum(), cpu=d.cpu_energy.sum() * J,
                gpu=d.gpu_energy.sum() * J, ram=d.ram_energy.sum() * J)


def idle(r):
    """Head and tail idle power, as recorded (energy / reported duration) and window-corrected (energy / RAM-integrated window)."""
    out = {}
    for ph in ("head", "tail"):
        row = r["em"][r["em"]["task_name"] == f"idle_baseline_{ph}"].iloc[0]
        dur = float(row["duration"])
        win = float(row["ram_energy"]) * J / float(row["ram_power"])
        e = float(row["energy_consumed"]) * J
        out[ph] = dict(rec=e / dur, corr=e / win, dur=dur, win=win,
                       cpu=float(row["cpu_energy"]) * J / win, gpu=float(row["gpu_energy"]) * J / win, ram=float(row["ram_energy"]) * J / win)
    out["floor_mean"] = (out["head"]["corr"] + out["tail"]["corr"]) / 2
    out["floor_head"] = out["head"]["corr"]
    out["floor_tail"] = out["tail"]["corr"]
    return out


def work(r):
    ep, eps = r["tm"]["epochs"], r["tm"]["episodes"]
    tr = [e["return"] for e in eps if e["phase"] == "train"]
    n10 = max(1, len(tr) // 10)
    fits = [e["model_train_epochs"] for e in ep if e.get("model_train_epochs") is not None]
    return dict(n_epochs=len(ep), n_train_episodes=len(tr), fit_epochs=sum(fits), n_fits=len(fits),
                synth=sum(e["synthetic_transitions_generated"] for e in ep if e.get("synthetic_transitions_generated") is not None),
                last10=float(np.mean(tr[-n10:])), n10=n10)


# ---------------------------------------------------------------------------------------------
head("0. Environment")
print("python", sys.version.split()[0], "| scipy", scipy.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)
print("root:", ROOT)
R = {k: load(v) for k, v in RUNS.items()}
a, b = R["R2000"], R["R200"]
pr = pd.read_csv(OUT / "per_run_energy_per_flop.csv")
pr["run_dir"] = pr["run_dir"].str.replace("\\", "/", regex=False)
print("run order by start_time_utc: R200 started after R2000 =",
      pd.Timestamp(b["meta"]["start_time_utc"]) > pd.Timestamp(a["meta"]["start_time_utc"]))

# ---------------------------------------------------------------------------------------------
head("1. Metadata diff (all keys, nested) between R2000 and R200")
fa, fb = flat(a["meta"]), flat(b["meta"])
n_diff = 0
for k in sorted(set(fa) | set(fb)):
    if fa.get(k, "<absent>") != fb.get(k, "<absent>"):
        n_diff += 1
        print(f"  DIFF {k}: R2000={fa.get(k, '<absent>')!r}  R200={fb.get(k, '<absent>')!r}")
print(f"differing keys: {n_diff} of {len(set(fa) | set(fb))}")
for k in ["algo_name", "env_id", "seed", "git_commit", "torch_version", "codecarbon_version", "algo_config.hidden_sizes", "algo_config.batch_size",
          "algo_config.updates_per_env_step", "algo_config.model_hidden_sizes", "algo_config.rollout_min_length", "algo_config.rollout_max_length",
          "experiment_config.train_steps", "experiment_config.warmup_steps", "experiment_config.steps_per_epoch"]:
    print(f"  {k}: R2000={fa.get(k)!r} R200={fb.get(k)!r} identical={fa.get(k) == fb.get(k)}")

# ---------------------------------------------------------------------------------------------
head("2. Completeness and run.log")
for k, r in R.items():
    vc = r["em"]["prefix"].value_counts()
    print(k, "rows:", len(r["em"]), {p: int(vc.get(p, 0)) for p in MEASURED + ["warmup", "idle_baseline_head", "idle_baseline_tail"]})
    print("   run.log lines:", len(r["log"].splitlines()), "| WARNING/ERROR:", len(re.findall(r"\[(WARNING|ERROR)\]", r["log"])),
          "| thermal gate:", r["meta"]["thermal_gate"]["final_temp_c"], "C, reason", r["meta"]["thermal_gate"]["reason"])
    print("   run.log mentions of clock/lock/RAPL/geoloc:", len(re.findall(r"lgc|Locked|GPU clock|clock|RAPL|geoloc", r["log"], re.I)))

# ---------------------------------------------------------------------------------------------
head("3. Training totals (full precision)")
T = {k: training_totals(r) for k, r in R.items()}
for k in R:
    t = T[k]
    csvrow = pr[(pr.run_dir == str(RUNS[k].relative_to(ROOT)).replace("\\", "/")) & (pr.segment == "TOTAL_MEASURED_TRAINING")]
    print(f"{k}: energy {t['energy']:.4f} J = {t['energy'] / 1e3:.3f} kJ = {t['energy'] / J:.5f} kWh | duration {t['duration']:.3f} s = {t['duration'] / 60:.3f} min "
          f"| mean power {t['energy'] / t['duration']:.4f} W")
    if len(csvrow):
        c = csvrow.iloc[0]
        print(f"     per_run CSV: energy {c.total_energy_joules:.4f} J | duration {c.duration_s:.3f} s | power {c.mean_power_w:.4f} W | FLOPs {c.total_flops:.6e} "
              f"| J/FLOP {c.energy_per_flop_j_per_flop:.4e}")
    else:
        print("     (no TOTAL_MEASURED_TRAINING row in per_run CSV for this run_dir)")
rE, rD = T["R200"]["energy"] / T["R2000"]["energy"], T["R200"]["duration"] / T["R2000"]["duration"]
rP = (T["R200"]["energy"] / T["R200"]["duration"]) / (T["R2000"]["energy"] / T["R2000"]["duration"])
print(f"ratios R200/R2000: energy {rE:.5f} | duration {rD:.5f} | power {rP:.5f} | rP*rD = {rP * rD:.5f} | 1/rD = {1 / rD:.5f} | power fall = {100 * (1 - rP):.2f}%")
print(f"duration growth = {100 * (rD - 1):.1f}%; clock ratio 2000/200 = {2000 / 200:.1f}")
for k, r in R.items():
    d = r["em"]
    s4 = d[d["prefix"].isin(MEASURED)].duration.sum()
    print(f"{k}: sum of the four task durations {s4:.4f} s")

# ---------------------------------------------------------------------------------------------
head("4. Segments (tab:clock-segments)")
seg_rows = []
for s in MEASURED:
    row = {"segment": s}
    for k, r in R.items():
        d = r["em"][r["em"]["prefix"] == s]
        row[f"E_{k}"], row[f"t_{k}"] = d.energy_j.sum(), d.duration.sum()
        row[f"P_{k}"] = row[f"E_{k}"] / row[f"t_{k}"]
        row[f"share_{k}"] = 100 * row[f"E_{k}"] / T[k]["energy"]
    row["E_ratio"], row["t_ratio"], row["P_ratio"] = row["E_R200"] / row["E_R2000"], row["t_R200"] / row["t_R2000"], row["P_R200"] / row["P_R2000"]
    seg_rows.append(row)
seg_df = pd.DataFrame(seg_rows).set_index("segment")
print(seg_df.round(4).T)
print("sum of shares R2000 / R200:", seg_df.share_R2000.sum(), seg_df.share_R200.sum())
print("rollout + synthetic rollout gen. share of training energy:", {k: round(seg_df.loc[["rollout", "synthetic_rollout_generation"], f"share_{k}"].sum(), 3) for k in R})

head("5. Per-epoch task durations of rollout and synthetic rollout generation (polling interval 1 s)")
for s in ["rollout", "synthetic_rollout_generation"]:
    for k, r in R.items():
        d = r["em"][r["em"]["prefix"] == s].duration
        print(f"{s:30s} {k}: n={len(d)} max={d.max():.4f} s median={d.median():.4f} s n(<1 s)={int((d < 1.0).sum())}")

head("6. Allocation coverage (sum of timed sub-segments / measured gradient-update duration)")
for k, r in R.items():
    sw = r["seg"]["_sub_segment_wall_time_seconds"]
    gu = r["em"][r["em"]["prefix"] == "gradient_updates"]
    cov = sum(sw.values()) / gu.duration.sum()
    print(f"{k}: sum of sub-segment timers {sum(sw.values()):.2f} s of {gu.duration.sum():.2f} s -> {100 * cov:.2f}%")
    en = {s: r["seg"][s] * J for s in ALLOC}
    print("    allocated energies (kJ):", {s: round(v / 1e3, 3) for s, v in en.items()}, "sum", round(sum(en.values()) / 1e3, 3),
          "| measured GU energy (kJ):", round(gu.energy_j.sum() / 1e3, 3))
    print("    allocated powers (W) = energy / timed sub-segment duration:", {s: round(en[s] / sw[s], 1) for s in ALLOC},
          "| pooled:", round(sum(en.values()) / sum(sw.values()), 1), "| measured GU power:", round(gu.energy_j.sum() / gu.duration.sum(), 1))

# ---------------------------------------------------------------------------------------------
head("7. Idle windows and above-idle energy")
I = {k: idle(r) for k, r in R.items()}
for k in R:
    i = I[k]
    print(f"{k}: head corr {i['head']['corr']:.4f} W (as recorded {i['head']['rec']:.4f}; window {i['head']['win']:.2f} s of {i['head']['dur']:.2f} s) | "
          f"tail corr {i['tail']['corr']:.4f} W (as recorded {i['tail']['rec']:.4f}; window {i['tail']['win']:.2f} s) | floor mean {i['floor_mean']:.4f} W "
          f"| tail-head {i['tail']['corr'] - i['head']['corr']:.3f} W ({100 * (i['tail']['corr'] / i['head']['corr'] - 1):.2f}%)")
print(f"floor ratio R200/R2000 (mean): {I['R200']['floor_mean'] / I['R2000']['floor_mean']:.4f} -> fall {100 * (1 - I['R200']['floor_mean'] / I['R2000']['floor_mean']):.2f}%")
above = {}
for k in R:
    t, i = T[k], I[k]
    row = {}
    for lab, fl in [("mean", i["floor_mean"]), ("head", i["floor_head"]), ("tail", i["floor_tail"])]:
        row[lab] = dict(floor_e=fl * t["duration"], above_e=t["energy"] - fl * t["duration"], above_p=(t["energy"] - fl * t["duration"]) / t["duration"],
                        share=100 * fl * t["duration"] / t["energy"])
    above[k] = row
    print(f"{k}: floor share (mean) {row['mean']['share']:.2f}% | above-idle energy (mean floor) {row['mean']['above_e'] / 1e3:.3f} kJ, power {row['mean']['above_p']:.3f} W "
          f"| floor energy {row['mean']['floor_e'] / 1e3:.3f} kJ")
for lab in ("mean", "head", "tail"):
    print(f"above-idle ratio R200/R2000 with {lab} floor: {above['R200'][lab]['above_e'] / above['R2000'][lab]['above_e']:.4f}")
print(f"S-6.1: idle floor alone accumulated over the R200 duration: {I['R200']['floor_mean']:.2f} W x {T['R200']['duration']:.1f} s = "
      f"{I['R200']['floor_mean'] * T['R200']['duration'] / 1e3:.2f} kJ vs. R2000 training energy {T['R2000']['energy'] / 1e3:.2f} kJ")

# ---------------------------------------------------------------------------------------------
head("8. Component split (tab:clock-components)")
for k in R:
    t = T[k]
    print(f"{k}: CPU {t['cpu'] / 1e3:.3f} kJ ({t['cpu'] / t['duration']:.3f} W, {100 * t['cpu'] / t['energy']:.2f}%) | GPU {t['gpu'] / 1e3:.3f} kJ ({t['gpu'] / t['duration']:.3f} W, "
          f"{100 * t['gpu'] / t['energy']:.2f}%) | RAM {t['ram'] / 1e3:.3f} kJ ({t['ram'] / t['duration']:.3f} W, {100 * t['ram'] / t['energy']:.2f}%) "
          f"| sum {(t['cpu'] + t['gpu'] + t['ram']) / 1e3:.3f} kJ vs total {t['energy'] / 1e3:.3f} kJ")
print("component ratios R200/R2000:", {c: round(T["R200"][c] / T["R2000"][c], 4) for c in ("cpu", "gpu", "ram")})
for k, r in R.items():
    d = r["em"]
    rp = (d.ram_energy * J / d.duration)
    print(f"{k}: ram_power column unique (rounded 6): {sorted(set(np.round(d.ram_power, 6)))[:5]} | RAM energy / duration over measured tasks: "
          f"{T[k]['ram'] / T[k]['duration']:.5f} W | RAM total size {d.ram_total_size.iloc[0]:.2f} GB")

# ---------------------------------------------------------------------------------------------
head("9. Work done (identical-work claim)")
W = {k: work(r) for k, r in R.items()}
for k in R:
    c = pr[(pr.run_dir == str(RUNS[k].relative_to(ROOT)).replace("\\", "/")) & (pr.segment == "dynamics_model_update")]
    steps = int(c.iloc[0].call_count) if len(c) else None
    print(k, W[k], "| optimiser steps (dynamics call_count):", steps)
    tot = pr[(pr.run_dir == str(RUNS[k].relative_to(ROOT)).replace("\\", "/")) & (pr.segment == "TOTAL_MEASURED_TRAINING")]
    print("    total FLOPs:", f"{tot.iloc[0].total_flops:.4e}" if len(tot) else None)
ka = [(e["model_train_epochs"], e["synthetic_transitions_generated"], e["mean_episode_return"], e.get("model_holdout_mse")) for e in a["tm"]["epochs"]]
kb = [(e["model_train_epochs"], e["synthetic_transitions_generated"], e["mean_episode_return"], e.get("model_holdout_mse")) for e in b["tm"]["epochs"]]
print("per-epoch (fit epochs, synthetic transitions, mean return, holdout MSE) identical:", ka == kb)
print("episode returns identical:", [e["return"] for e in a["tm"]["episodes"]] == [e["return"] for e in b["tm"]["episodes"]])
print("epoch rows total identical (full dict):", a["tm"]["epochs"] == b["tm"]["epochs"])

# ---------------------------------------------------------------------------------------------
head("10. Canonical comparator: MBPO / HalfCheetah-v5 / hidden [256,256]")
cands = []
for p in sorted(glob.glob(str(RES / "mbpo" / "HalfCheetah-v5" / "seed_*" / "*" / "metadata.json"))):
    m = json.load(open(p))
    if m["seed"] in CANON and m["algo_config"]["hidden_sizes"] == [256, 256] and m["algo_config"]["batch_size"] == 256 and m["algo_config"]["updates_per_env_step"] == 1:
        cands.append(Path(p).parent)
print("candidates (metadata-based):", len(cands))
tot_rows = pr[(pr.segment == "TOTAL_MEASURED_TRAINING") & (pr.algo == "mbpo") & (pr.env_id == "HalfCheetah-v5") & pr.seed.isin(CANON) & (pr.hidden_sizes == "256x256")]
print("per_run CSV rows (hidden 256x256, canonical seeds):", len(tot_rows), "| included_in_cross_seed_avg:", tot_rows.included_in_cross_seed_avg.tolist())
can_dirs = [ROOT / d for d in tot_rows[tot_rows.included_in_cross_seed_avg].run_dir]
print("canonical runs used:", [str(d.relative_to(ROOT)).replace("\\", "/") for d in can_dirs])
print("same set as metadata-based candidates:", sorted(map(str, can_dirs)) == sorted(map(str, cands)))
C = {}
for d in can_dirs:
    r = load(d)
    t = training_totals(r)
    w = work(r)
    row = tot_rows[tot_rows.run_dir.str.contains(d.name)].iloc[0]
    steps = pr[(pr.run_dir == str(d.relative_to(ROOT)).replace("\\", "/")) & (pr.segment == "dynamics_model_update")].iloc[0].call_count
    C[r["meta"]["seed"]] = dict(meta=r["meta"], energy=t["energy"], duration=t["duration"], power=t["energy"] / t["duration"], flops=row.total_flops, fit=w["fit_epochs"],
                                steps=int(steps), last10=w["last10"], synth=w["synth"], csv_energy=row.total_energy_joules)
cd = pd.DataFrame({s: {k: v for k, v in x.items() if k != "meta"} for s, x in C.items()}).T
print(cd)
E = cd.energy.astype(float)
mean, sd = E.mean(), E.std(ddof=1)
print(f"canonical energy: mean {mean / 1e3:.4f} kJ | sample sd (ddof=1) {sd / 1e3:.4f} kJ | population sd {E.std(ddof=0) / 1e3:.4f} kJ | min {E.min() / 1e3:.3f} max {E.max() / 1e3:.3f}")
print(f"R2000 z = {(T['R2000']['energy'] - mean) / sd:.4f} | R200 / canonical mean = {T['R200']['energy'] / mean:.4f} | R2000 / mean = {T['R2000']['energy'] / mean:.4f}")
for col in ["duration", "power"]:
    v = cd[col].astype(float)
    print(f"canonical {col}: mean {v.mean():.4f} sd {v.std(ddof=1):.4f} min {v.min():.4f} max {v.max():.4f} | R2000 {T['R2000']['duration'] if col == 'duration' else T['R2000']['energy'] / T['R2000']['duration']:.4f}")
csv_tot = {k: pr[(pr.run_dir == str(RUNS[k].relative_to(ROOT)).replace("\\", "/")) & (pr.segment == "TOTAL_MEASURED_TRAINING")].iloc[0].total_flops for k in R}
print(f"R2000 vs canonical: FLOPs {csv_tot['R2000']:.4e} (canonical min {cd.flops.astype(float).min():.4e}, max {cd.flops.astype(float).max():.4e})")
print(f"fit epochs R2000 {W['R2000']['fit_epochs']} (canonical {cd.fit.astype(int).tolist()}) | optimiser steps R2000 "
      f"{int(pr[(pr.run_dir == str(RUNS['R2000'].relative_to(ROOT)).replace(chr(92), '/')) & (pr.segment == 'dynamics_model_update')].iloc[0].call_count)} (canonical {cd.steps.astype(int).tolist()})")
print(f"last-10% return R2000 {W['R2000']['last10']:.2f} | canonical {[round(x, 1) for x in cd.last10.astype(float)]} min {cd.last10.astype(float).min():.1f} max {cd.last10.astype(float).max():.1f}")
print("R2000 outside canonical range? FLOPs:", not (cd.flops.astype(float).min() <= csv_tot["R2000"] <= cd.flops.astype(float).max()),
      "| fit epochs:", not (cd.fit.astype(int).min() <= W["R2000"]["fit_epochs"] <= cd.fit.astype(int).max()),
      "| return:", not (cd.last10.astype(float).min() <= W["R2000"]["last10"] <= cd.last10.astype(float).max()))
print("\nmetadata diff of R2000 against each canonical run (keys ignored: start/end time, thermal_gate.*):")
IGN = {"start_time_utc", "end_time_utc"}
for s, x in C.items():
    fc = flat(x["meta"])
    keys = [k for k in sorted(set(fa) | set(fc)) if k not in IGN and not k.startswith("thermal_gate.") and fa.get(k, "<absent>") != fc.get(k, "<absent>")]
    print(f"  seed {s}: " + "; ".join(f"{k}: R2000={fa.get(k, '<absent>')!r} canon={fc.get(k, '<absent>')!r}" for k in keys))
print("canonical clock request (min/max):", {s: (x["meta"]["experiment_config"]["gpu_min_clock_mhz"], x["meta"]["experiment_config"]["gpu_max_clock_mhz"]) for s, x in C.items()})
print("canonical lock_gpu_clocks:", {s: x["meta"]["experiment_config"]["lock_gpu_clocks"] for s, x in C.items()})

# ---------------------------------------------------------------------------------------------
head("11. CodeCarbon utilisation fields (is a CPU or GPU utilisation record usable?)")
for k, r in R.items():
    d = r["em"]
    print(f"{k}: per-task rows: cpu_utilization_percent max {d.cpu_utilization_percent.max():.3f}, gpu_utilization_percent max {d.gpu_utilization_percent.max():.3f}, "
          f"ram_used_gb max {d.ram_used_gb.max():.3f} | whole-run row: cpu {r['whole']['cpu_utilization_percent']:.3f} %, gpu {r['whole']['gpu_utilization_percent']:.3f} %")
