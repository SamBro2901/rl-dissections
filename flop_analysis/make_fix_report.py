"""
Builds flop_analysis/output/FIX_REPORT.md: before/after comparison of the
TOTAL_MEASURED_TRAINING and MBPO dynamics_model_update fixes, from
output/_before_fix/*.csv (generated at 0c360ad) vs. the current output/*.csv,
plus the saved output of verify_mbpo_fit_flops.py and check_totals.py.

Usage (from repo root, after compute_energy_per_flop.py has been re-run):
    python flop_analysis/verify_mbpo_fit_flops.py > flop_analysis/output/verify_mbpo_fit_flops.log
    python flop_analysis/check_totals.py          > flop_analysis/output/check_totals.log
    python flop_analysis/make_fix_report.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "output")
BEFORE = os.path.join(OUT, "_before_fix")
GROUP = ["algo", "env_id", "architecture_signature", "updates_per_env_step", "mbpo_rollout_regime"]
TOTAL = "TOTAL_MEASURED_TRAINING"
CANONICAL_SEEDS = {331, 958, 14577, 43611, 85062}


def pct(new, old):
    return 100.0 * (new / old - 1.0) if old else float("nan")


def read_log(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return f"({name} not found -- run the script and save its output there)"
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def group_label(row):
    parts = [row.algo, row.env_id, row.architecture_signature, f"UTD{row.updates_per_env_step}"]
    if isinstance(row.mbpo_rollout_regime, str):
        parts.append(row.mbpo_rollout_regime)
    return " / ".join(parts)


def merged(segment, old_cs, new_cs):
    o = old_cs[old_cs.segment == segment]
    n = new_cs[new_cs.segment == segment]
    m = o.merge(n, on=GROUP, suffixes=("_old", "_new"), how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), f"group mismatch for {segment}"
    return m


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def sub_timer_coverage():
    """sum of perf_counter sub-segment time / measured gradient_updates task
    duration, per algo, canonical seeds -- context for why the TOTAL row's power
    fields now use the measured task duration."""
    import csv
    cov = {}
    for seg_path in glob.glob(os.path.join(REPO_ROOT, "results", "*", "*", "seed_*", "*", "segment_energy.json")):
        run_dir = os.path.dirname(seg_path)
        with open(os.path.join(run_dir, "metadata.json")) as f:
            meta = json.load(f)
        if meta["seed"] not in CANONICAL_SEEDS:
            continue
        with open(seg_path) as f:
            sub = json.load(f).get("_sub_segment_wall_time_seconds")
        csvs = glob.glob(os.path.join(run_dir, "emissions_*.csv"))
        if not sub or not csvs:
            continue
        with open(csvs[0], newline="") as f:
            dur = sum(float(r["duration"]) for r in csv.DictReader(f)
                      if re.fullmatch(r"gradient_updates_\d+", r.get("task_name") or ""))
        if dur:
            cov.setdefault(meta["algo_name"], []).append(sum(sub.values()) / dur)
    return {a: (min(v), sum(v) / len(v), max(v)) for a, v in sorted(cov.items())}


def main():
    old_cs = pd.read_csv(os.path.join(BEFORE, "cross_seed_energy_per_flop.csv"))
    new_cs = pd.read_csv(os.path.join(OUT, "cross_seed_energy_per_flop.csv"))

    # ---- TOTAL ----
    t = merged(TOTAL, old_cs, new_cs)
    t["e_pct"] = [pct(n, o) for n, o in zip(t.mean_energy_kwh_new, t.mean_energy_kwh_old)]
    t["f_pct"] = [pct(n, o) for n, o in zip(t.total_flops_new, t.total_flops_old)]
    t["j_pct"] = [pct(n, o) for n, o in zip(t.mean_energy_per_flop_j_per_flop_new, t.mean_energy_per_flop_j_per_flop_old)]
    t = t.sort_values(GROUP, na_position="first")
    total_rows = [[
        group_label(r),
        f"{r.mean_energy_kwh_old:.5f}", f"{r.mean_energy_kwh_new:.5f}", f"{r.e_pct:+.2f}%",
        f"{r.total_flops_old:.4e}", f"{r.total_flops_new:.4e}", f"{r.f_pct:+.2f}%",
        f"{r.mean_energy_per_flop_j_per_flop_old:.4e}", f"{r.mean_energy_per_flop_j_per_flop_new:.4e}", f"{r.j_pct:+.2f}%",
    ] for r in t.itertuples()]
    t_max = t.loc[t.j_pct.abs().idxmax()]
    t_min = t.loc[t.j_pct.abs().idxmin()]

    # ---- MBPO dynamics ----
    d = merged("dynamics_model_update", old_cs, new_cs)
    d["f_pct"] = [pct(n, o) for n, o in zip(d.total_flops_new, d.total_flops_old)]
    d["j_pct"] = [pct(n, o) for n, o in zip(d.mean_energy_per_flop_j_per_flop_new, d.mean_energy_per_flop_j_per_flop_old)]
    d["holdout_share"] = 100.0 * d.dynamics_holdout_flops / d.total_flops_new  # column is new-only, no suffix
    d = d.sort_values(GROUP)
    dyn_rows = [[
        group_label(r),
        f"{r.total_flops_old:.4e}", f"{r.total_flops_new:.4e}", f"{r.f_pct:+.2f}%",
        f"{r.mean_energy_per_flop_j_per_flop_old:.4e}", f"{r.mean_energy_per_flop_j_per_flop_new:.4e}", f"{r.j_pct:+.2f}%",
        f"{r.holdout_share:.2f}%",
    ] for r in d.itertuples()]
    d_max = d.loc[d.f_pct.abs().idxmax()]

    # per-run MBPO dynamics range (all runs, not just canonical averages)
    old_pr = pd.read_csv(os.path.join(BEFORE, "per_run_energy_per_flop.csv"))
    new_pr = pd.read_csv(os.path.join(OUT, "per_run_energy_per_flop.csv"))
    pr = old_pr[old_pr.segment == "dynamics_model_update"].merge(
        new_pr[new_pr.segment == "dynamics_model_update"], on=["run_dir", "segment"], suffixes=("_old", "_new"))
    pr_pct = 100.0 * (pr.total_flops_new / pr.total_flops_old - 1.0)
    calls_unchanged = bool((pr.call_count_new == pr.call_count_old).all())
    total_direction = "every group goes up" if (t.j_pct > 0).all() else "direction varies by group"
    prt = old_pr[old_pr.segment == TOTAL].merge(new_pr[new_pr.segment == TOTAL], on=["run_dir", "segment"],
                                                  suffixes=("_old", "_new"))
    prt_pct = 100.0 * (prt.energy_per_flop_j_per_flop_new / prt.energy_per_flop_j_per_flop_old - 1.0)

    verify_log = read_log("verify_mbpo_fit_flops.log")
    checks_log = read_log("check_totals.log")
    verify_status = "**ALL PASSED**" if "ALL PASSED" in verify_log else "**FAILED / not run**"
    checks_status = "**ALL CHECKS PASSED**" if "ALL CHECKS PASSED" in checks_log else "**FAILED / not run**"
    mbpo_t = t[t.algo == "mbpo"]
    other_t = t[t.algo != "mbpo"]

    cov = sub_timer_coverage()
    cov_lines = "\n".join(f"  - {a}: min {lo:.3f}, mean {mean:.3f}, max {hi:.3f}" for a, (lo, mean, hi) in cov.items())

    report = f"""# FIX_REPORT -- energy-per-FLOP aggregation fixes

Branch `fix/flop-aggregation`. "Old" = `flop_analysis/output/_before_fix/*.csv` (output of
`compute_energy_per_flop.py` at `0c360ad`); "new" = `flop_analysis/output/*.csv` after both fixes.
Cross-seed numbers are over the canonical seeds {{331, 958, 14577, 43611, 85062}}; J/FLOP is
mean energy / mean FLOPs per group, as in the cross-seed CSV. Per-call FLOP constants
(`flops_per_call.json`) are unchanged; only aggregation and call counts changed.
Regenerate this file with `python flop_analysis/make_fix_report.py`.

## Headline numbers

- **TOTAL_MEASURED_TRAINING J/FLOP**: {total_direction}. Largest change
  **{t_max.j_pct:+.2f}%** ({group_label(t_max)}), smallest {t_min.j_pct:+.2f}%
  ({group_label(t_min)}). Per run (all 300 runs, including non-canonical seeds): {prt_pct.min():+.2f}% to {prt_pct.max():+.2f}%.
  Non-MBPO groups: {other_t.j_pct.min():+.2f}% to {other_t.j_pct.max():+.2f}% (Fix 1 only: `buffer_sample` energy
  enters the numerator, `target_update` ops leave the denominator). MBPO groups:
  {mbpo_t.j_pct.min():+.2f}% to {mbpo_t.j_pct.max():+.2f}%, because Fix 2's ~8% more dynamics FLOPs in the denominator
  offset the added `buffer_sample` energy.
- **MBPO dynamics_model_update FLOPs**: up **{d.f_pct.min():+.2f}% to {d.f_pct.max():+.2f}%** across groups
  (largest: {group_label(d_max)}), so J/FLOP goes down by the same factor
  ({d.j_pct.min():+.2f}% to {d.j_pct.max():+.2f}%). Holdout evaluation makes up
  {d.holdout_share.min():.2f}%-{d.holdout_share.max():.2f}% of the corrected count. Per run (all {len(pr)} MBPO runs):
  {pr_pct.min():+.2f}% to {pr_pct.max():+.2f}%. `call_count` (optimizer steps) {"is unchanged in every run" if calls_unchanged else "CHANGED in some runs"}.
- `verify_mbpo_fit_flops.py`: {verify_status} (see output below for per-case relative differences).
  `check_totals.py`: {checks_status}.

## Definitions: old vs. new

### TOTAL_MEASURED_TRAINING

**Old.** A segment counted toward both numerator and denominator iff it was not an idle
baseline, not `warmup`, and had non-zero `total_flops`. So `target_update` was *included*: its
energy in the numerator, its elementwise Polyak op count in the denominator, mixed with matmul
FLOPs. `buffer_sample` (0 FLOPs) was *excluded*, so part of the measured `gradient_updates`
energy dropped out of the total. Duration/power fields summed the allocated sub-segments'
`perf_counter` durations. The row's note claimed `target_update` was excluded, but it wasn't.

**New: all training energy / matmul FLOPs only.**

```
numerator   = sum of segment_energy.json energy over TRAINING_SEGMENTS present:
                rollout,
                buffer_sample, critic_update, actor_update, target_update   (= all of gradient_updates),
                dynamics_model_update, synthetic_rollout_generation          (MBPO),
                world_model_pretrain                                         (TD-MPC2)
              excluding idle_baseline_head/tail, warmup and "_"-prefixed keys
denominator = sum of total_flops over the same segments with flop_type == "matmul"
              (i.e. all except target_update; buffer_sample contributes 0)
```

Segment membership is fixed by module-level constants (`TRAINING_SEGMENTS`,
`ELEMENTWISE_OP_SEGMENTS`, `ZERO_FLOP_SEGMENTS`), not inferred from FLOP counts. Duration/power
fields of the TOTAL row now come from the directly measured CodeCarbon tasks: the whole
`gradient_updates` task stands in for the four allocated sub-segments.

New `flop_type` column in both CSVs: `matmul` | `elementwise` (`target_update`: its
`energy_per_flop_j_per_flop` is **J per elementwise op**, not J/FLOP) | `none` (`buffer_sample`,
`warmup`, idle baselines) | `mixed_total` (the TOTAL row).

### MBPO dynamics_model_update FLOPs

**Old.** Per logged fit() call: `ceil(n_train / B) x epochs_run x ensemble_size x member_fwdbwd(B)`.
This leaves out `_holdout_mse()`, which scores all members on the holdout split after every fit
epoch, and charges the partial last minibatch as a full batch of B.

**New** (`mbpo_fit_flops()`, mirrors `EnsembleDynamicsModel.fit()` line by line):

```
per_sample_member_fwdbwd = dynamics_member_fwdbwd / model_train_batch_size   (exact integer division, asserted)
per_sample_ensemble_fwd  = dynamics_ensemble_forward_all_bs1
for each epochs_log row with model_train_epochs = E (skipped if None/0):
    n_total   = min(warmup_steps + epoch * steps_per_epoch, buffer_capacity)
    n_holdout = max(1, int(n_total * model_holdout_ratio))
    n_train   = n_total - n_holdout
    train     = E * ensemble_size * n_train * per_sample_member_fwdbwd
    holdout   = E * n_holdout * per_sample_ensemble_fwd
call_count  = sum of E * ceil(n_train / B)    (optimizer steps)
```

Checks behind this formula: `algorithms/mbpo.py` calls `fit()` at the start of each epoch,
before that epoch's rollout; the real buffer is `ReplayBuffer(buffer_capacity=1,000,000)`, never
reached in these runs; and the `n_total` formula matches every logged end-of-epoch
`buffer_size` (= n_total + steps_per_epoch) in all MBPO runs (the script warns on mismatch,
and no warnings were raised). New columns `dynamics_train_flops` / `dynamics_holdout_flops` hold
the split.

## TOTAL_MEASURED_TRAINING, per group (old -> new)

{md_table(["group", "E old (kWh)", "E new (kWh)", "dE", "FLOPs old", "FLOPs new", "dF", "J/FLOP old", "J/FLOP new", "dJ/FLOP"], total_rows)}

## MBPO dynamics_model_update, per group (old -> new)

{md_table(["group", "FLOPs old", "FLOPs new", "dF", "J/FLOP old", "J/FLOP new", "dJ/FLOP", "holdout share"], dyn_rows)}

## verify_mbpo_fit_flops.py output

```
{verify_log}
```

## check_totals.py output

```
{checks_log}
```

## Observations (out of scope, not changed)

- **Sub-segment timer coverage.** `sum(_sub_segment_wall_time_seconds) / measured
  gradient_updates task duration` over the canonical seeds:
{cov_lines}
  The `perf_counter` sub-timers cover only this fraction of the measured task. That's why
  the TOTAL row's power fields now use the task duration: summing sub-segment durations
  would inflate mean power by 1/coverage. The allocated sub-segment rows still carry their
  `perf_counter` durations and the power derived from them (unchanged by this fix).
- **`.item()` sync placement differs across algorithms.** SAC/TD3 call
  `q1_loss.item()`/`actor_loss.item()` just *after* their timers stop
  (`algorithms/sac.py:179,212`, `algorithms/td3.py:167,186`). TD-MPC2's critic `.item()`s are
  also outside the timer (`algorithms/tdmpc2.py:630-633`), but `_update_pi()` returns
  `pi_loss.item()` (`tdmpc2.py:559`), so the actor timer (`tdmpc2.py:636-638`) *includes* a
  device sync. Its GPU drain then shows up in `actor_update` time rather than in whichever
  timer ran next. Together with the missing `torch.cuda.synchronize()`, this shifts
  time-share (and so allocated energy) between sub-segments differently per algorithm. It
  does not affect the TOTAL row, which now uses only the measured `gradient_updates` energy.
- MBPO `dynamics_model_update` is roughly an order of magnitude less energy-efficient per
  FLOP than the SAC critic/actor updates in the same runs (see the table above against
  `critic_update` in the cross-seed CSV). This is plausibly due to small 200-wide layers,
  a Python loop over 7 members per minibatch, and host-side numpy indexing per batch. It's
  worth keeping in mind when interpreting its J/FLOP; it is unaffected by this fix beyond
  the ~8% FLOP correction.
"""
    path = os.path.join(OUT, "FIX_REPORT.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
