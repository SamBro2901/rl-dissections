"""
Mean idle power, head vs. tail, for every recorded run (data-quality check).

Each run brackets training with two idle windows of equal length
(``idle_baseline_head`` before warmup, ``idle_baseline_tail`` after the last
epoch; see experiment_runner.py). If the thermal gate + settle period work as
intended, the machine should draw the same power in both windows; a
systematically higher tail means residual heat/activity from training leaks
into the "idle" measurement, which matters for any idle-subtracted energy
figure and for the smallest segments (e.g. ``target_update``).

Mean power is derived as energy / duration per component (CPU, GPU, RAM,
total) from the per-task rows in ``emissions_base_*.csv``. CodeCarbon's own
``*_power`` columns on those rows are *not* used for the analysis: they are a
point sample, not a window mean, and on the tail row they are overwritten
with the whole-run average (identical to ``emissions.csv``). They are kept in
the CSV (``cc_*_power_w``) so that discrepancy can be shown.

By default only the canonical 5-seed set is included (same convention as
flop_analysis/compute_energy_per_flop.py); ``--all-seeds`` includes dev runs.

Usage:
    python idle_power_analysis.py [--results-dir results]
        [--out-html idle_power_report.html] [--out-csv idle_power.csv]
        [--all-seeds]
"""

import argparse
import glob
import json
import os
import subprocess
from collections import Counter

import numpy as np
import pandas as pd

CANONICAL_SEEDS = {331, 958, 14577, 43611, 85062}
COMPONENTS = ("cpu", "gpu", "ram", "total")
J_PER_KWH = 3.6e6

# Fields that legitimately differ between runs and must not count as a
# "config variant" when labelling sweep runs.
_IGNORED_EXP_FIELDS = {"seed", "output_dir", "algo_name", "env_id"}


def _read_idle_rows(run_dir):
    paths = glob.glob(os.path.join(run_dir, "emissions_base_*.csv"))
    if not paths:
        return None
    df = pd.read_csv(paths[0])
    rows = df[df["task_name"].isin(["idle_baseline_head", "idle_baseline_tail"])]
    if set(rows["task_name"]) != {"idle_baseline_head", "idle_baseline_tail"}:
        return None
    return rows.set_index("task_name")


def load_runs(results_dir, seeds=None):
    records = []
    for meta_path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "seed_*", "*", "metadata.json"))):
        run_dir = os.path.dirname(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        if seeds is not None and meta["seed"] not in seeds:
            continue
        idle = _read_idle_rows(run_dir)
        if idle is None:
            continue

        gate = meta.get("thermal_gate") or {}
        rec = {
            "run_dir": os.path.relpath(run_dir, results_dir).replace("\\", "/"),
            "algo": meta["algo_name"],
            "env": meta["env_id"],
            "seed": meta["seed"],
            "start_time_utc": pd.Timestamp(meta["start_time_utc"]),
            "settle_seconds": meta.get("experiment_config", {}).get("settle_seconds"),
            "thermal_gate_temp_c": gate.get("final_temp_c"),
            "thermal_gate_power_w": gate.get("final_power_w"),
            "thermal_gate_waited_s": gate.get("waited_seconds"),
            # NVML readings just before tracker.start() (i.e. just before the
            # head window) and just after the tail window ends.
            "run_start_temp_c": gate.get("run_start_temp_c"),
            "run_start_gpu_nvml_w": gate.get("run_start_power_w"),
            "run_end_temp_c": gate.get("run_end_temp_c"),
            "run_end_gpu_nvml_w": gate.get("run_end_power_w"),
            "_algo_config": meta.get("algo_config", {}),
            "_exp_config": meta.get("experiment_config", {}),
        }
        run_avg = pd.read_csv(os.path.join(run_dir, "emissions.csv")).iloc[-1]
        rec["run_avg_cpu_power_w"] = float(run_avg["cpu_power"])
        rec["run_avg_gpu_power_w"] = float(run_avg["gpu_power"])
        for phase in ("head", "tail"):
            row = idle.loc[f"idle_baseline_{phase}"]
            dur = float(row["duration"])
            # CodeCarbon models RAM at a constant power, so RAM energy / RAM
            # power is the time span the task's energy was actually integrated
            # over. For the head task this comes out ~3 s longer than its
            # reported duration (it absorbs energy accumulated between
            # tracker.start() and start_task()); for the tail it matches.
            eff_window = float(row["ram_energy"]) * J_PER_KWH / float(row["ram_power"])
            rec[f"{phase}_duration_s"] = dur
            rec[f"{phase}_integrated_window_s"] = eff_window
            for comp in COMPONENTS:
                col = "energy_consumed" if comp == "total" else f"{comp}_energy"
                energy_j = float(row[col]) * J_PER_KWH
                rec[f"{phase}_{comp}_w"] = energy_j / dur
                rec[f"{phase}_{comp}_w_corr"] = energy_j / eff_window
            for comp in ("cpu", "gpu", "ram"):
                rec[f"cc_{phase}_{comp}_power_w"] = float(row[f"{comp}_power"])
        for suffix in ("", "_corr"):
            for comp in COMPONENTS:
                d = rec[f"tail_{comp}_w{suffix}"] - rec[f"head_{comp}_w{suffix}"]
                rec[f"delta_{comp}_w{suffix}"] = d
                rec[f"delta_{comp}_pct{suffix}"] = 100.0 * d / rec[f"head_{comp}_w{suffix}"]
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["variant"] = _variant_labels(df)
    df = df.drop(columns=["_algo_config", "_exp_config"])
    df = df.sort_values("start_time_utc").reset_index(drop=True)
    return df


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        out[prefix + k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return out


def _variant_labels(df):
    """Label each run by how its config differs from the modal config of its
    (algo, env) group, e.g. "baseline", "batch_size=512", "num_q=7".

    Each sweep varies exactly one field, so the modal value of every field
    within a group is the canonical baseline value."""
    labels = pd.Series("baseline", index=df.index)
    for _, grp in df.groupby(["algo", "env"]):
        flat = {
            i: {**_flatten(r["_algo_config"]),
                **{k: v for k, v in _flatten(r["_exp_config"], "exp.").items()
                   if k[4:] not in _IGNORED_EXP_FIELDS}}
            for i, r in grp.iterrows()
        }
        # A key missing from a run means the field didn't exist yet when it
        # was recorded (older code), not that the run deviated from baseline.
        keys = set().union(*(f.keys() for f in flat.values()))
        modal = {k: Counter(f[k] for f in flat.values() if k in f).most_common(1)[0][0] for k in keys}
        for i, f in flat.items():
            diffs = sorted(f"{k.removeprefix('exp.')}={f[k]}" for k in keys if k in f and f[k] != modal[k])
            if diffs:
                labels[i] = ", ".join(diffs)
    return labels


def build_notes(df):
    """Mode-independent data-quality findings, as HTML list items."""
    n = len(df)
    hw, tw = df["head_integrated_window_s"], df["tail_integrated_window_s"]
    excess_pct = 100.0 * (hw / df["head_duration_s"] - 1.0)
    tail_is_run_avg = (
        np.isclose(df["cc_tail_gpu_power_w"], df["run_avg_gpu_power_w"])
        & np.isclose(df["cc_tail_cpu_power_w"], df["run_avg_cpu_power_w"])
    ).sum()
    gate_zero = (df["thermal_gate_waited_s"].fillna(np.inf) < 1.0).sum()
    temps = df[["run_start_temp_c", "head_total_w"]].dropna()
    r_head_temp = temps.corr().iloc[0, 1]
    d_tot, d_gpu, d_cpu = (df[f"delta_{c}_w_corr"].mean() for c in ("total", "gpu", "cpu"))
    by_algo = df.groupby("algo")["delta_total_w_corr"].mean().sort_values()
    algo_name = {"sac": "SAC", "mbpo": "MBPO", "td3": "TD3", "tdmpc2": "TD-MPC2"}
    # Does the GPU's temperature explain the tail excess? Fit head power vs.
    # pre-head temperature, then ask how far tail power sits above that line
    # at the tail's own (post-tail) temperature.
    fit = df[["run_start_temp_c", "head_total_w_corr", "run_end_temp_c", "tail_total_w_corr"]].dropna()
    slope, intercept = np.polyfit(fit["run_start_temp_c"], fit["head_total_w_corr"], 1)
    tail_resid = (fit["tail_total_w_corr"] - (slope * fit["run_end_temp_c"] + intercept)).mean()

    items = [
        f"""<b>Head window integrates ~{(hw - df['head_duration_s']).mean():.2f}&nbsp;s more energy than
        its reported duration.</b> CodeCarbon models RAM at a constant {df['cc_head_ram_power_w'].iloc[0]:.0f}&nbsp;W,
        so RAM energy ÷ RAM power gives the span the energy was actually summed over: head
        {hw.mean():.2f}&nbsp;s (sd {hw.std():.2f}) vs. a reported {df['head_duration_s'].mean():.2f}&nbsp;s; tail
        {tw.mean():.2f}&nbsp;s, matching its reported duration. The head task most likely absorbs energy
        accumulated between <code>tracker.start()</code> and <code>start_task()</code>. Effect: as-recorded
        head power, and <code>idle_baseline_head</code> kWh in <code>segment_energy.json</code>, are
        ~{excess_pct.mean():.1f}% high, which is why as-recorded RAM shows a head &gt; tail gap that cannot
        be physical. <i>Window-corrected</i> divides by the integrated span instead (assumes power during
        the extra ~3&nbsp;s matches the rest of the window).""",
        f"""<b>CodeCarbon's per-task <code>*_power</code> columns are not window means and are not used
        here.</b> On the head row, reported GPU power has a median of {df['cc_head_gpu_power_w'].median():.1f}&nbsp;W
        vs. {df['head_gpu_w_corr'].median():.1f}&nbsp;W energy-derived (max reported
        {df['cc_head_gpu_power_w'].max():.0f}&nbsp;W). On the tail row they equal the whole-run average in
        <code>emissions.csv</code> for {tail_is_run_avg} of {n} runs. All values on this page are energy ÷ time.""",
        f"""<b>Thermal gate passed immediately in {gate_zero} of {n} runs</b> (<code>waited_seconds</code> ≈ 0).
        This is how it is built: the reference is captured fresh in each process after a stabilisation poll,
        so the poll plus the {df['settle_seconds'].median():.0f}&nbsp;s settle are the real cool-down.
        Even so, head idle power is not a machine constant. It tracks GPU temperature just before the
        head window (r&nbsp;=&nbsp;{r_head_temp:.2f}, {temps['run_start_temp_c'].min():.0f}–{temps['run_start_temp_c'].max():.0f}&nbsp;°C).""",
        f"""<b>Tail &gt; head is driven by the GPU.</b> Window-corrected, the mean tail − head difference is
        {d_tot:+.1f}&nbsp;W in total: GPU {d_gpu:+.1f}&nbsp;W, CPU {d_cpu:+.1f}&nbsp;W, RAM ≈ 0 (constant model).
        The GPU has not returned to its pre-run idle 90&nbsp;s after training stops. The size of the gap depends
        on the algorithm: smallest for {algo_name.get(by_algo.index[0], by_algo.index[0])} ({by_algo.iloc[0]:+.1f}&nbsp;W),
        largest for {algo_name.get(by_algo.index[-1], by_algo.index[-1])} ({by_algo.iloc[-1]:+.1f}&nbsp;W).
        Temperature alone does not account for it: fitting head power against pre-head GPU temperature
        ({slope:.2f}&nbsp;W/°C), tail power sits {tail_resid:+.1f}&nbsp;W above that line at its own temperature.""",
    ]
    return "\n".join(f"<li>{s}</li>" for s in items)


def _git_short_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def render_html(df, out_path, scope_label):
    from plotly.offline import get_plotlyjs

    keep = ["run_dir", "algo", "env", "seed", "variant", "start_time_utc",
            "run_start_temp_c", "run_end_temp_c"]
    keep += [c for c in df.columns if c.startswith(("head_", "tail_", "delta_"))]
    data_json = df[keep].to_json(orient="records", date_format="iso", double_precision=4)
    html = (_TEMPLATE
            .replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__DATA__", data_json)
            .replace("__NOTES__", build_notes(df))
            .replace("__SCOPE__", scope_label)
            .replace("__N__", str(len(df)))
            .replace("__COMMIT__", _git_short_head())
            .replace("__GENERATED__", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-html", default="idle_power_report.html")
    parser.add_argument("--out-csv", default="idle_power.csv")
    parser.add_argument("--all-seeds", action="store_true", help="include non-canonical dev/exploratory seeds")
    args = parser.parse_args()

    df = load_runs(args.results_dir, seeds=None if args.all_seeds else CANONICAL_SEEDS)
    if df.empty:
        raise SystemExit(f"No runs with idle head/tail rows found under {args.results_dir}")
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} runs to {args.out_csv}")
    scope = "all seeds" if args.all_seeds else "canonical seeds {" + ", ".join(map(str, sorted(CANONICAL_SEEDS))) + "}"
    render_html(df, args.out_html, scope)
    print(f"Wrote {args.out_html}")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Idle Power Check</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --head: #2a78d6; --tail: #eb6834; --delta: #52514e; --accent-wash: rgba(42,120,214,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --head: #3987e5; --tail: #d95926; --delta: #c3c2b7; --accent-wash: rgba(57,135,229,0.16);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --head: #3987e5; --tail: #d95926; --delta: #c3c2b7; --accent-wash: rgba(57,135,229,0.16);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; font-weight: 650; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 0 0 2px; font-weight: 600; }
.sub { color: var(--ink-2); margin: 0 0 24px; max-width: 760px; }
.cap { color: var(--ink-2); margin: 0 0 12px; font-size: 13px; max-width: 820px; }
.controls { display: flex; flex-wrap: wrap; gap: 16px 28px; align-items: center;
  position: sticky; top: 0; z-index: 5; background: var(--page); padding: 12px 0; margin-bottom: 8px;
  border-bottom: 1px solid var(--border); }
.ctl { display: flex; align-items: center; gap: 8px; }
.ctl > span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { border: 0; background: transparent; color: var(--ink-2); font: inherit; font-size: 13px;
  padding: 5px 12px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--accent-wash); color: var(--ink); font-weight: 600; }
.seg button:focus-visible { outline: 2px solid var(--head); outline-offset: -2px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0 28px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .k { color: var(--ink-2); font-size: 12px; }
.tile .v { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile .s { color: var(--muted); font-size: 12px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 16px 8px; margin-bottom: 20px; }
.plot { width: 100%; height: 420px; }
.notes { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; }
.notes ol { margin: 8px 0 0; padding-left: 20px; }
.notes li { margin: 0 0 10px; color: var(--ink-2); max-width: 900px; }
.notes li b { color: var(--ink); }
code { font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; background: var(--accent-wash); padding: 1px 4px; border-radius: 4px; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
th:first-child, td:first-child { text-align: left; }
tr.all td { font-weight: 600; border-top: 1px solid var(--axis); }
footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
@media (max-width: 640px) { .plot { height: 360px; } h1 { font-size: 20px; } .tile .v { font-size: 22px; } }
</style>
<script>__PLOTLYJS__</script>
</head>
<body>
<main>
  <h1>Idle power: head vs. tail window</h1>
  <p class="sub">Mean power in the two 90&nbsp;s idle windows that bracket each training run
    (<code>idle_baseline_head</code> before warmup, <code>idle_baseline_tail</code> after the last epoch),
    computed as energy&nbsp;÷&nbsp;time from each run's per-task CodeCarbon log. __N__ runs, __SCOPE__.
    If the two windows measure the same "idle machine", tail − head should be centred on zero.</p>

  <div class="controls" role="toolbar" aria-label="Chart controls">
    <div class="ctl"><span>Component</span>
      <div class="seg" data-key="comp">
        <button data-v="total">Total</button><button data-v="gpu">GPU</button>
        <button data-v="cpu">CPU</button><button data-v="ram">RAM</button>
      </div></div>
    <div class="ctl"><span>Head window</span>
      <div class="seg" data-key="corr">
        <button data-v="raw">As recorded</button><button data-v="corr">Window-corrected</button>
      </div></div>
    <div class="ctl"><span>Runs</span>
      <div class="seg" data-key="scope">
        <button data-v="all">All configs</button><button data-v="baseline">Baseline config only</button>
      </div></div>
  </div>

  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2>Idle power per run, by algorithm and environment</h2>
    <p class="cap">One point per run, boxes show the quartiles. Hover a point for the run directory and its sweep variant.</p>
    <div class="plot" id="p-levels"></div>
  </div>

  <div class="card">
    <h2>Paired difference, tail − head</h2>
    <p class="cap">Within-run difference, so it is unaffected by drift across runs. The dark marker is the mean with a 95% t-interval. Points above zero mean the machine drew more power after training than before it.</p>
    <div class="plot" id="p-delta"></div>
  </div>

  <div class="card">
    <h2>Across the measurement campaign</h2>
    <p class="cap">Each run's head and tail window by start time, joined by a thin line. Shows whether idle power drifted across days or sweeps.</p>
    <div class="plot" id="p-time"></div>
  </div>

  <div class="card">
    <h2>Idle power vs. GPU temperature</h2>
    <p class="cap">NVML GPU temperature read just before the head window (head) and just after the tail window (tail). r is the Pearson correlation within each window.</p>
    <div class="plot" id="p-temp"></div>
  </div>

  <div class="card">
    <h2>Summary by algorithm and environment</h2>
    <p class="cap">Mean ± sd across runs. Δ = tail − head, paired per run; the interval is a 95% t-interval on the mean Δ.</p>
    <div class="tablewrap"><table id="tbl"></table></div>
  </div>

  <div class="notes">
    <h2>Data-quality notes</h2>
    <ol>__NOTES__</ol>
  </div>

  <footer>Generated __GENERATED__ by <code>idle_power_analysis.py</code> at commit __COMMIT__.
    Use the camera icon on any chart to save it as SVG.</footer>
</main>

<script>
const ROWS = __DATA__;
const ALGOS = ["sac", "mbpo", "td3", "tdmpc2"];
const ALGO_LABEL = {sac: "SAC", mbpo: "MBPO", td3: "TD3", tdmpc2: "TD-MPC2"};
const ENVS = ["HalfCheetah-v5", "Ant-v5"];
const COMP_LABEL = {total: "Total", gpu: "GPU", cpu: "CPU", ram: "RAM"};
const state = {comp: "total", corr: "raw", scope: "all"};

const groupOf = r => `${ALGO_LABEL[r.algo] || r.algo} · ${r.env.replace(/-v\d+$/, "")}`;
const GROUPS = [];
for (const a of ALGOS) for (const e of ENVS) {
  const g = `${ALGO_LABEL[a]} · ${e.replace(/-v\d+$/, "")}`;
  if (ROWS.some(r => groupOf(r) === g)) GROUPS.push(g);
}
for (const r of ROWS) { const g = groupOf(r); if (!GROUPS.includes(g)) GROUPS.push(g); }

const suf = () => state.corr === "corr" ? "_corr" : "";
const val = (r, phase) => r[`${phase}_${state.comp}_w${suf()}`];
const rows = () => ROWS.filter(r => state.scope === "all" || r.variant === "baseline");

const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
function tcrit(df) {  // two-sided 95%, Cornish-Fisher expansion around z
  const z = 1.959964;
  return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2);
}
const ci95 = a => a.length < 2 ? NaN : tcrit(a.length - 1) * sd(a) / Math.sqrt(a.length);
function pearson(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) { const dx = x[i] - mx, dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  return sxy / Math.sqrt(sxx * syy);
}
const f1 = x => Number.isFinite(x) ? x.toFixed(1) : "–";
const f2 = x => Number.isFinite(x) ? x.toFixed(2) : "–";
const sgn = (x, f = f2) => (x > 0 ? "+" : x < 0 ? "−" : "") + f(Math.abs(x));

function theme() {
  const cs = getComputedStyle(document.documentElement);
  const v = n => cs.getPropertyValue(n).trim();
  return {surface: v("--surface"), ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"),
          grid: v("--grid"), axis: v("--axis"), head: v("--head"), tail: v("--tail"), delta: v("--delta")};
}
function axis(t, extra) {
  return Object.assign({gridcolor: t.grid, linecolor: t.axis, zeroline: false, tickfont: {color: t.muted},
    title: {font: {color: t.ink2, size: 12}}, automargin: true}, extra);
}
function layout(t, extra) {
  return Object.assign({
    paper_bgcolor: t.surface, plot_bgcolor: t.surface,
    font: {family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: t.ink2, size: 12},
    margin: {l: 8, r: 8, t: 28, b: 8},
    legend: {orientation: "h", x: 0, y: 1.06, yanchor: "bottom", font: {color: t.ink2}},
    hoverlabel: {bgcolor: t.surface, bordercolor: t.axis, font: {color: t.ink, size: 12}},
  }, extra);
}
const CONFIG = {displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  toImageButtonOptions: {format: "svg", filename: "idle_power", width: 1000, height: 460}};
const unit = () => `${COMP_LABEL[state.comp]} power (W)`;
const runHover = "<b>%{customdata[0]}</b><br>%{customdata[1]} · seed %{customdata[2]}<br>";
const cd = rs => rs.map(r => [r.run_dir, r.variant, r.seed]);

function drawLevels(t, rs) {
  const tr = ["head", "tail"].map(ph => ({
    type: "box", name: ph === "head" ? "Head window" : "Tail window",
    x: rs.map(groupOf), y: rs.map(r => val(r, ph)), customdata: cd(rs),
    marker: {color: t[ph], size: 5, opacity: 0.6}, line: {color: t[ph], width: 1.5},
    fillcolor: "rgba(0,0,0,0)", boxpoints: "all", jitter: 0.45, pointpos: 0,
    hoveron: "points", hovertemplate: runHover + `${ph}: %{y:.2f} W<extra></extra>`,
  }));
  Plotly.react("p-levels", tr, layout(t, {boxmode: "group", boxgap: 0.25, boxgroupgap: 0.15,
    xaxis: axis(t, {categoryorder: "array", categoryarray: GROUPS, showgrid: false}),
    yaxis: axis(t, {title: {text: unit()}})}), CONFIG);
}

function drawDelta(t, rs) {
  const key = r => r[`delta_${state.comp}_w${suf()}`];
  const pts = {type: "box", name: "Run", x: rs.map(groupOf), y: rs.map(key), customdata: cd(rs),
    marker: {color: t.muted, size: 5, opacity: 0.65}, line: {color: t.axis, width: 1},
    fillcolor: "rgba(0,0,0,0)", boxpoints: "all", jitter: 0.5, pointpos: 0, hoveron: "points",
    hovertemplate: runHover + "tail − head: %{y:+.2f} W<extra></extra>", showlegend: false};
  const gs = GROUPS.filter(g => rs.some(r => groupOf(r) === g));
  const stats = gs.map(g => { const a = rs.filter(r => groupOf(r) === g).map(key); return [mean(a), ci95(a), a.length]; });
  const mk = {type: "scatter", mode: "markers", name: "Mean ± 95% CI", x: gs, y: stats.map(s => s[0]),
    customdata: stats.map(s => [s[1], s[2]]),
    marker: {color: t.ink, size: 10, symbol: "diamond", line: {color: t.surface, width: 2}},
    error_y: {type: "data", array: stats.map(s => s[1]), color: t.ink, thickness: 2, width: 6},
    hovertemplate: "<b>%{x}</b><br>mean Δ %{y:+.2f} W ± %{customdata[0]:.2f}<br>n = %{customdata[1]}<extra></extra>"};
  Plotly.react("p-delta", [pts, mk], layout(t, {showlegend: false,
    xaxis: axis(t, {categoryorder: "array", categoryarray: GROUPS, showgrid: false}),
    yaxis: axis(t, {title: {text: `Tail − head, ${COMP_LABEL[state.comp]} (W)`}, zeroline: true, zerolinecolor: t.axis, zerolinewidth: 1.5}),
  }), CONFIG);
}

function drawTime(t, rs) {
  const s = [...rs].sort((a, b) => a.start_time_utc < b.start_time_utc ? -1 : 1);
  const lx = [], ly = [];
  for (const r of s) { lx.push(r.start_time_utc, r.start_time_utc, null); ly.push(val(r, "head"), val(r, "tail"), null); }
  const link = {type: "scatter", mode: "lines", x: lx, y: ly, line: {color: t.grid, width: 1.5},
    hoverinfo: "skip", showlegend: false};
  const tr = ["head", "tail"].map(ph => ({type: "scatter", mode: "markers",
    name: ph === "head" ? "Head window" : "Tail window",
    x: s.map(r => r.start_time_utc), y: s.map(r => val(r, ph)),
    customdata: s.map(r => [r.run_dir, r.variant, r.seed]),
    marker: {color: t[ph], size: 7, line: {color: t.surface, width: 1}},
    hovertemplate: runHover + `${ph}: %{y:.2f} W<extra></extra>`}));
  Plotly.react("p-time", [link, ...tr], layout(t, {hovermode: "closest",
    xaxis: axis(t, {title: {text: "Run start (UTC)"}, showgrid: false}),
    yaxis: axis(t, {title: {text: unit()}})}), CONFIG);
}

function drawTemp(t, rs) {
  const tr = [["head", "run_start_temp_c", "before head"], ["tail", "run_end_temp_c", "after tail"]].map(([ph, tk, lbl]) => {
    const s = rs.filter(r => r[tk] != null);
    const x = s.map(r => r[tk]), y = s.map(r => val(r, ph));
    const r = x.length > 2 ? pearson(x, y) : NaN;
    return {type: "scatter", mode: "markers",
      name: `${ph === "head" ? "Head" : "Tail"} window, GPU temp ${lbl} (r = ${f2(r)})`,
      x: x.map((v, i) => v + (((i * 0.618034) % 1) - 0.5) * 0.5), y, customdata: s.map(q => [q.run_dir, q.variant, q.seed, q[tk]]),
      marker: {color: t[ph], size: 7, opacity: 0.75, line: {color: t.surface, width: 1}},
      hovertemplate: runHover + `GPU %{customdata[3]} °C<br>${ph}: %{y:.2f} W<extra></extra>`};
  });
  Plotly.react("p-temp", tr, layout(t, {hovermode: "closest",
    xaxis: axis(t, {title: {text: "GPU temperature (°C, integer NVML reading, jittered ±0.25)"}}),
    yaxis: axis(t, {title: {text: unit()}})}), CONFIG);
}

function drawTiles(rs) {
  const h = rs.map(r => val(r, "head")), tl = rs.map(r => val(r, "tail"));
  const d = rs.map(r => r[`delta_${state.comp}_w${suf()}`]), dp = rs.map(r => r[`delta_${state.comp}_pct${suf()}`]);
  const up = d.filter(x => x > 0).length;
  const t = (k, v, s, sw) => `<div class="tile"><div class="k">${sw ? `<span class="sw" style="background:var(--${sw})"></span>` : ""}${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  document.getElementById("tiles").innerHTML =
    t(`Head window, ${COMP_LABEL[state.comp]}`, `${f1(mean(h))} W`, `sd ${f2(sd(h))} W · n = ${rs.length}`, "head") +
    t(`Tail window, ${COMP_LABEL[state.comp]}`, `${f1(mean(tl))} W`, `sd ${f2(sd(tl))} W · n = ${rs.length}`, "tail") +
    t("Mean tail − head", `${sgn(mean(d))} W`, `95% CI ±${f2(ci95(d))} W · ${sgn(mean(dp), f1)}% of head`) +
    t("Runs with tail > head", `${up} of ${rs.length}`, `${f1(100 * up / rs.length)}% of runs`);
}

function drawTable(rs) {
  const line = (label, a, cls = "") => {
    const h = a.map(r => val(r, "head")), tl = a.map(r => val(r, "tail"));
    const d = a.map(r => r[`delta_${state.comp}_w${suf()}`]), dp = a.map(r => r[`delta_${state.comp}_pct${suf()}`]);
    const up = d.filter(x => x > 0).length;
    return `<tr class="${cls}"><td>${label}</td><td>${a.length}</td>
      <td>${f2(mean(h))} ± ${f2(sd(h))}</td><td>${f2(mean(tl))} ± ${f2(sd(tl))}</td>
      <td>${sgn(mean(d))} [${sgn(mean(d) - ci95(d))}, ${sgn(mean(d) + ci95(d))}]</td>
      <td>${sgn(mean(dp), f1)}%</td><td>${up}/${a.length}</td></tr>`;
  };
  let html = `<thead><tr><th>Algorithm · env</th><th>n</th><th>Head (W)</th><th>Tail (W)</th>
    <th>Δ mean (W) [95% CI]</th><th>Δ % of head</th><th>tail &gt; head</th></tr></thead><tbody>`;
  for (const g of GROUPS) { const a = rs.filter(r => groupOf(r) === g); if (a.length) html += line(g, a); }
  html += line("All runs", rs, "all") + "</tbody>";
  document.getElementById("tbl").innerHTML = html;
}

function render() {
  for (const seg of document.querySelectorAll(".seg"))
    for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(state[seg.dataset.key] === b.dataset.v));
  const t = theme(), rs = rows();
  drawTiles(rs); drawLevels(t, rs); drawDelta(t, rs); drawTime(t, rs); drawTemp(t, rs); drawTable(rs);
}
for (const seg of document.querySelectorAll(".seg"))
  seg.addEventListener("click", e => { const b = e.target.closest("button"); if (b) { state[seg.dataset.key] = b.dataset.v; render(); } });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
