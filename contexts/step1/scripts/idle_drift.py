"""Task 1: idle-baseline head/tail drift, as recorded vs window-corrected (read-only).
Run: .venv/Scripts/python.exe contexts/step1/scripts/idle_drift.py
Window-corrected definition = the one in idle_power_analysis.py:101 (integration span
= ram_energy[J] / ram_power[W]; power = component energy / span)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import canonical_runs, read_base_csv, OUT

J = 3.6e6
COMPS = ["total", "cpu", "gpu", "ram"]
runs = canonical_runs()
rec = []
for _, r in runs.iterrows():
    b = read_base_csv(r.run_dir)
    d = dict(run_dir=r.run_dir, algo=r.algo, env=r.env, seed=r.seed, config=r.config,
             start_time_utc=r.start_time_utc, date_utc=r.start_time_utc[:10])
    for ph in ("head", "tail"):
        row = b[b.task_name == f"idle_baseline_{ph}"].iloc[0]
        dur = float(row.duration)
        span = float(row.ram_energy) * J / float(row.ram_power)
        d[f"{ph}_duration_s"] = dur
        d[f"{ph}_span_s"] = span
        d[f"{ph}_timestamp"] = row.timestamp
        for c in COMPS:
            col = "energy_consumed" if c == "total" else f"{c}_energy"
            e = float(row[col])
            d[f"{ph}_{c}_kwh"] = e
            d[f"{ph}_{c}_w_rec"] = e * J / dur
            d[f"{ph}_{c}_w_corr"] = e * J / span
    for sfx in ("rec", "corr"):
        for c in COMPS:
            h, t = d[f"head_{c}_w_{sfx}"], d[f"tail_{c}_w_{sfx}"]
            d[f"drift_{c}_w_{sfx}"] = t - h
            d[f"drift_{c}_pct_{sfx}"] = 100 * (t - h) / h
    d["pre_head_gpu_temp_c"] = r.meta.get("thermal_gate", {}).get("run_start_temp_c")
    rec.append(d)
df = pd.DataFrame(rec)
df.to_csv(os.path.join(OUT, "01_idle_drift_per_run.csv"), index=False)
assert len(df) == 280


def stats(s):
    a = s.abs()
    return dict(n=len(s), median=s.median(), mean=s.mean(), min=s.min(), max=s.max(),
                p5=s.quantile(.05), p95=s.quantile(.95), median_abs=a.median(),
                frac_tail_gt_head=(s > 0).mean())


def fmt(x, p=2):
    return f"{x:.{p}f}"


lines = []
P = lines.append
P("# Task 1: Idle-baseline drift\n")
P("## Question\n")
P("Which numbers describe head-vs-tail idle drift under (a) *as recorded* and (b) *window-corrected*, and how exactly is \"window-corrected\" defined? Basis: the 280 canonical runs.\n")
P("## Method\n")
P("- Script: `contexts/step1/scripts/idle_drift.py` (run with the repo `.venv` Python; uses `contexts/step1/scripts/common.py` for the canonical run list, 280 rows, grid verified in Task 4).")
P("- Data: rows `idle_baseline_head` / `idle_baseline_tail` of each run's `emissions_base_*.csv` (columns `duration`, `*_energy`, `ram_power`, `timestamp`), 280 runs. Full-precision per-run output: `01_idle_drift_per_run.csv`.")
P("- Drift = (P_tail − P_head) / P_head, signed, in percent. Absolute drift in W is also in the CSV (`drift_*_w_*`).\n")
P("### Definition found in the repo (verbatim)\n")
P("`idle_power_analysis.py` L96-101 (code comment + code):")
P("```\n# CodeCarbon models RAM at constant power, so RAM energy / RAM\n# power is the time span the task's energy was actually integrated\n# over. For the head task this comes out ~3 s longer than its\n# reported duration ...\neff_window = float(row[\"ram_energy\"]) * J_PER_KWH / float(row[\"ram_power\"])\n```")
P("`idle_power_analysis.py` L106-108: `rec[f\"{phase}_{comp}_w\"] = energy_j / dur` (as recorded) and `rec[f\"{phase}_{comp}_w_corr\"] = energy_j / eff_window` (window-corrected).")
P("`idle_power_analysis.py` L190-191: \"*Window-corrected* divides by the integrated span instead (assumes power during the extra ~3 s matches the rest of the window).\"")
P("`README.md` L646-649: \"The analysis reports both *as-recorded* (÷ reported duration) and *window-corrected* (÷ integrated span; assumes power during the extra seconds matches the rest) values; the `_corr` columns are the ones to use. The correction is **not** applied to `segment_energy.json`.\"")
P("`CLAUDE.md` L506-508: \"the CSV/report give both *as-recorded* and *window-corrected* (energy ÷ integrated span) values — use corrected.\"")
P("\nSo the definition **exists** in the repo and is used here unchanged: window-corrected power = component energy / (`ram_energy`/`ram_power`), per window. The \"integration span\" is a *proxy* (CodeCarbon's RAM power is a constant, so RAM energy/RAM power = integrated seconds); `emissions_base_*.csv` has no per-task start/stop timestamps (only one `timestamp` column, second resolution, at task stop), so the span cannot be cross-checked from timestamps there. In the example run `run.log` shows \"Recording idle baseline for 90s...\" at 13:30:13 and \"Starting warmup\" at 13:31:47 (≈94 s wall clock between the two log lines, consistent with a span of ≈93 s).\n")

P("## Results\n")
P("### 1. Durations of the idle windows (nominal 90 s)\n")
P("| window | nominal (config) | recorded duration: min / median / max (s) | integration span: min / median / max (s) | span − duration: median (s) |")
P("|---|---|---|---|---|")
for ph in ("head", "tail"):
    du, sp = df[f"{ph}_duration_s"], df[f"{ph}_span_s"]
    P(f"| {ph} | 90 | {fmt(du.min(),4)} / {fmt(du.median(),4)} / {fmt(du.max(),4)} | {fmt(sp.min(),3)} / {fmt(sp.median(),3)} / {fmt(sp.max(),3)} | {fmt((sp-du).median(),3)} |")
P("\nThe recorded durations confirm 90 s; the head window integrates ≈3.1 s more than that, the tail does not.\n")

P("### 2. Drift summary, all 280 runs (percent of head power)\n")
P("| statistic | component | median | mean | min | max | p5 | p95 | median abs | frac tail>head |")
P("|---|---|---|---|---|---|---|---|---|---|")
allrows = []
for sfx, name in (("rec", "(a) as recorded"), ("corr", "(b) window-corrected")):
    for c in COMPS:
        s = stats(df[f"drift_{c}_pct_{sfx}"])
        allrows.append(dict(group="ALL", statistic=sfx, component=c, **s))
        P(f"| {name} | {c} | {fmt(s['median'])} | {fmt(s['mean'])} | {fmt(s['min'])} | {fmt(s['max'])} | {fmt(s['p5'])} | {fmt(s['p95'])} | {fmt(s['median_abs'])} | {fmt(s['frac_tail_gt_head'],3)} |")
P("\nSame in W (tail − head), total only:\n")
P("| statistic | median | mean | min | max | p5 | p95 | median abs |")
P("|---|---|---|---|---|---|---|---|")
for sfx, name in (("rec", "(a)"), ("corr", "(b)")):
    s = stats(df[f"drift_total_w_{sfx}"])
    P(f"| {name} | {fmt(s['median'])} | {fmt(s['mean'])} | {fmt(s['min'])} | {fmt(s['max'])} | {fmt(s['p5'])} | {fmt(s['p95'])} | {fmt(s['median_abs'])} |")
P("\nNote on RAM (a): CodeCarbon's RAM power is constant, so any non-zero RAM drift under (a) is the 3 s window artefact (head energy is divided by 90 s but integrated over ≈93 s); under (b) RAM drift is ≈0 by construction.\n")

P("### 3. Drift by algorithm × environment (total power, % of head)\n")
P("| group | n | (a) median | (a) mean | (a) min | (a) max | (a) p5 | (a) p95 | (a) median abs | (a) frac tail>head | (b) median | (b) mean | (b) min | (b) max | (b) p5 | (b) p95 | (b) median abs | (b) frac tail>head |")
P("|" + "---|" * 18)
grp_rows = []
KEYS = ("median", "mean", "min", "max", "p5", "p95", "median_abs")
for (a_, e_), g in df.groupby(["algo", "env"]):
    sa, sb = stats(g.drift_total_pct_rec), stats(g.drift_total_pct_corr)
    for sfx in ("rec", "corr"):
        for c in COMPS:
            grp_rows.append(dict(group=f"{a_}/{e_}", statistic=sfx, component=c, **stats(g[f"drift_{c}_pct_{sfx}"])))
    P(f"| {a_} / {e_} | {sa['n']} | " + " | ".join(fmt(sa[k]) for k in KEYS) + f" | {fmt(sa['frac_tail_gt_head'],3)} | " +
      " | ".join(fmt(sb[k]) for k in KEYS) + f" | {fmt(sb['frac_tail_gt_head'],3)} |")
pd.DataFrame(allrows + grp_rows).to_csv(os.path.join(OUT, "01_idle_drift_summary.csv"), index=False)
P("\n(Per-component tables for GPU/CPU/RAM for all groups: `01_idle_drift_summary.csv`, full precision.)\n")

P("### 4. Top 5 runs by absolute drift (total power)\n")
for sfx, name in (("rec", "(a) as recorded"), ("corr", "(b) window-corrected")):
    P(f"**{name}**\n")
    P("| run directory | head W | tail W | drift % |")
    P("|---|---|---|---|")
    t = df.reindex(df[f"drift_total_pct_{sfx}"].abs().sort_values(ascending=False).index).head(5)
    for _, r in t.iterrows():
        P(f"| `results/{r.run_dir}` | {fmt(r[f'head_total_w_{sfx}'])} | {fmt(r[f'tail_total_w_{sfx}'])} | {fmt(r[f'drift_total_pct_{sfx}'])} |")
    P("")

P("### 5. Absolute head idle power by algorithm × environment (total W)\n")
P("| group | n | (a) median | (a) range | (b) median | (b) range |")
P("|---|---|---|---|---|---|")
for (a_, e_), g in df.groupby(["algo", "env"]):
    P(f"| {a_} / {e_} | {len(g)} | {fmt(g.head_total_w_rec.median())} | {fmt(g.head_total_w_rec.min())}–{fmt(g.head_total_w_rec.max())} | {fmt(g.head_total_w_corr.median())} | {fmt(g.head_total_w_corr.min())}–{fmt(g.head_total_w_corr.max())} |")
P("\n### 6. Head idle power by calendar date of run start (UTC), window-corrected total W\n")
P("| date (UTC) | n runs | median | min | max | median pre-head GPU temp (°C) | algos/envs that day |")
P("|---|---|---|---|---|---|---|")
for dt, g in df.groupby("date_utc"):
    ae = ", ".join(sorted({f'{a}/{e.split("-")[0]}' for a, e in zip(g.algo, g.env)}))
    P(f"| {dt} | {len(g)} | {fmt(g.head_total_w_corr.median())} | {fmt(g.head_total_w_corr.min())} | {fmt(g.head_total_w_corr.max())} | {fmt(g.pre_head_gpu_temp_c.median(),1)} | {ae} |")
dd = df.groupby("date_utc").head_total_w_corr.median()
r_temp = df[["head_total_w_corr", "pre_head_gpu_temp_c"]].dropna().corr().iloc[0, 1]
P(f"\nDaily medians range {fmt(dd.min())}–{fmt(dd.max())} W (spread {fmt(dd.max()-dd.min())} W) across {len(dd)} calendar dates; correlation of head total power (corrected) with pre-head GPU temperature: r = {fmt(r_temp,3)}.\n")

P("## Caveats / things that look odd\n")
P("- \"Window-corrected\" rests on an assumption (power during the extra ≈3 s equals power during the rest of the window) and on RAM power being constant; it is a proxy for the integrated span, not timestamp-based. Not checked against independent timestamps (none in `emissions_base_*.csv`).")
P("- The correction raises drift because head power falls (÷ ≈93 s instead of 90 s) while tail is unchanged; it is not applied to `segment_energy.json` (README L648-649), so `idle_baseline_head` kWh there is the uncorrected, ≈3.4 % high value.")
P("- Head idle power depends on session/day and pre-head GPU temperature (sections 5-6), i.e. drift is not only within-run heat carry-over.")
open(os.path.join(OUT, "01_idle_drift.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("done")
