"""
RQ4 (item D4): GPU clock-lock comparison, dev seed 0 on purpose (the only non-canonical-seed item).
Read-only. Run from the repo root:  python contexts/Section_3.1_3.2_3.3_data/d4_rq4.py
Writes D4_seed0_mbpo_hc_runs.csv, D4_field_diff.csv, D4_rq4_energy_power_duration.csv, d4_report.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "flop_analysis"))
import compute_energy_per_flop as cepf  # noqa: E402

PRIMARY = "20260904_130915"
ROOT = REPO / "results" / "mbpo" / "HalfCheetah-v5" / "seed_0"
dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
        else:
            out[pre + k] = json.dumps(v) if isinstance(v, list) else v
    return out


metas = {p.name: json.loads((p / "metadata.json").read_text()) for p in dirs}
flats = {k: flat(m) for k, m in metas.items()}

# 1) inventory
inv = []
for k, m in metas.items():
    ec, ac = m["experiment_config"], m["algo_config"]
    inv.append(dict(timestamp_dir=k, gpu_min_clock_mhz=ec["gpu_min_clock_mhz"], gpu_max_clock_mhz=ec["gpu_max_clock_mhz"],
                    torch_version=m["torch_version"], codecarbon_version=m["codecarbon_version"], git_commit=m["git_commit"],
                    train_steps=ec["train_steps"], warmup_steps=ec["warmup_steps"], steps_per_epoch=ec["steps_per_epoch"],
                    hidden_sizes=json.dumps(ac["hidden_sizes"]), updates_per_env_step=ac["updates_per_env_step"],
                    batch_size=ac["batch_size"], rollout_max_length=ac["rollout_max_length"],
                    rollout_max_epoch=ac["rollout_max_epoch"], start_time_utc=m["start_time_utc"], end_time_utc=m["end_time_utc"],
                    cuda_device_name=m["cuda_device_name"]))
pd.DataFrame(inv).to_csv(HERE / "D4_seed0_mbpo_hc_runs.csv", index=False)

# 2) field-by-field diff of every other run against the primary
rows = []
for k in metas:
    if k == PRIMARY:
        continue
    keys = sorted(set(flats[k]) | set(flats[PRIMARY]))
    for f in keys:
        a, b = flats[PRIMARY].get(f, "<absent>"), flats[k].get(f, "<absent>")
        rows.append(dict(candidate=k, field=f, primary=a, candidate_value=b, differs=(a != b)))
dd = pd.DataFrame(rows)
dd.to_csv(HERE / "D4_field_diff.csv", index=False)

# 3) energy / duration / power of the training phase for every seed-0 run
out = []
for p in dirs:
    seg = json.loads((p / "segment_energy.json").read_text())
    hw = cepf.load_segment_hw_energy(str(p))
    train = [s for s in cepf.TRAINING_SEGMENTS if s in seg]
    e_kwh = sum(seg[s] for s in train)
    hw_src = sorted({("gradient_updates" if s in cepf.ALLOCATED_SUB_SEGMENTS else s) for s in train})
    dur = sum(hw[s]["duration_s"] for s in hw_src)
    gpu = sum(hw[s]["gpu_energy_kwh"] for s in hw_src)
    cpu = sum(hw[s]["cpu_energy_kwh"] for s in hw_src)
    ram = sum(hw[s]["ram_energy_kwh"] for s in hw_src)
    csv_total = sum(hw[s]["energy_consumed_kwh"] for s in hw_src)
    out.append(dict(
        timestamp_dir=p.name, gpu_clock_mhz=metas[p.name]["experiment_config"]["gpu_min_clock_mhz"],
        total_measured_training_energy_kwh=e_kwh, total_measured_training_energy_j=e_kwh * 3.6e6,
        sum_of_task_energy_consumed_kwh_from_per_task_csv=csv_total,
        training_duration_s_sum_of_measured_tasks=dur,
        mean_power_total_w=e_kwh * 3.6e6 / dur,
        mean_power_gpu_w=gpu * 3.6e6 / dur, mean_power_cpu_w=cpu * 3.6e6 / dur, mean_power_ram_w=ram * 3.6e6 / dur,
        gpu_energy_j=gpu * 3.6e6, cpu_energy_j=cpu * 3.6e6, ram_energy_j=ram * 3.6e6,
        rollout_kwh=seg["rollout"], gradient_updates_kwh=sum(seg[s] for s in cepf.ALLOCATED_SUB_SEGMENTS),
        dynamics_model_update_kwh=seg["dynamics_model_update"], synthetic_rollout_generation_kwh=seg["synthetic_rollout_generation"],
        warmup_kwh=seg.get("warmup"), idle_baseline_head_kwh=seg.get("idle_baseline_head"),
        idle_baseline_tail_kwh=seg.get("idle_baseline_tail"),
        rollout_duration_s=hw["rollout"]["duration_s"], gradient_updates_duration_s=hw["gradient_updates"]["duration_s"],
        dynamics_model_update_duration_s=hw["dynamics_model_update"]["duration_s"],
        synthetic_rollout_generation_duration_s=hw["synthetic_rollout_generation"]["duration_s"],
        idle_head_duration_s=hw["idle_baseline_head"]["duration_s"], idle_tail_duration_s=hw["idle_baseline_tail"]["duration_s"],
        n_task_rows=int(sum(1 for _ in open(next(p.glob("emissions_*.csv")), newline="")) - 1)))
od = pd.DataFrame(out)
od.to_csv(HERE / "D4_rq4_energy_power_duration.csv", index=False)

# 4) is it in the cross-seed / correlation inputs?
pr = pd.read_csv(REPO / "flop_analysis" / "output" / "per_run_energy_per_flop.csv")
cs = pd.read_csv(REPO / "flop_analysis" / "output" / "cross_seed_energy_per_flop.csv")
seed0 = pr[pr.run_dir.str.contains("mbpo/HalfCheetah-v5/seed_0/")]
rep = []
rep.append("## d4 report (generated)")
rep.append(f"per_run_energy_per_flop.csv rows for mbpo/HalfCheetah-v5/seed_0: {len(seed0)}; "
           f"included_in_cross_seed_avg values: {sorted(set(seed0.included_in_cross_seed_avg))}; "
           f"run_dirs: {sorted(set(seed0.run_dir))}")
rep.append(f"cross_seed_energy_per_flop.csv: n_seeds values for mbpo HalfCheetah-v5 rows: {sorted(set(cs[(cs.algo == 'mbpo') & (cs.env_id == 'HalfCheetah-v5')].n_seeds))}; "
           f"architecture signatures: {sorted(set(cs[(cs.algo == 'mbpo') & (cs.env_id == 'HalfCheetah-v5')].architecture_signature))}")
tm = {p.name: json.loads((p / 'training_metrics.json').read_text()) for p in dirs}
for k, t in tm.items():
    eps = [e for e in t["episodes"] if e["phase"] == "train"]
    rets = [e["return"] for e in eps]
    rep.append(f"{k}: n_epochs={len(t['epochs'])}, n_train_episodes={len(eps)}, last10pct_mean_return={np.mean(rets[-max(1, len(rets)//10):])!r}, "
               f"final_cumulative_reward={t['epochs'][-1]['cumulative_reward']!r}, "
               f"sum(model_train_epochs)={sum(e.get('model_train_epochs') or 0 for e in t['epochs'])}, "
               f"sum(synthetic_transitions_generated)={sum(e.get('synthetic_transitions_generated', 0) for e in t['epochs'])}")
n_diff = {k: int(g.differs.sum()) for k, g in dd.groupby("candidate")}
rep.append(f"n differing metadata fields vs primary: {n_diff}")
for k, g in dd[dd.differs].groupby("candidate"):
    rep.append(f"--- differing fields, primary {PRIMARY} vs {k}:")
    for _, r in g.iterrows():
        rep.append(f"  {r.field}: primary={r.primary!r} | {k}={r.candidate_value!r}")
(HERE / "d4_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
print("\n".join(rep))
print(od.T.to_string())
