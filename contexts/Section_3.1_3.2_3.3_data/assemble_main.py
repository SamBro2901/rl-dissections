"""
Assembles contexts/Section_3.1_3.2_3.3.md from main_template.md + generated fragments.
Run (repo root) AFTER build_section_3.py, d4_rq4.py and c4_crosscheck.py:
    python contexts/Section_3.1_3.2_3.3_data/assemble_main.py
Also copies (read-only sources -> new files here) flop_analysis/output/correlation/{config_means,correlation_summary}.csv
as C2_config_means.csv / C2_correlation_summary.csv, and cross-checks the SAC rows of B1/B2 against contexts/section_4.1_data.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "contexts" / "Section_3.1_3.2_3.3.md"

shutil.copyfile(REPO / "flop_analysis/output/correlation/config_means.csv", HERE / "C2_config_means.csv")
shutil.copyfile(REPO / "flop_analysis/output/correlation/correlation_summary.csv", HERE / "C2_correlation_summary.csv")

frag = {p.stem[5:]: p.read_text(encoding="utf-8") for p in HERE.glob("frag_*.md")}
checks = json.loads((HERE / "checks.json").read_text())

# ---- SAC cross-check against the Section 4.1 data (same pipeline inputs, independent build script)
s41 = pd.read_csv(REPO / "contexts/section_4.1_data/sac_cross_seed_summary.csv")
b2 = pd.read_csv(HERE / "B2_sweep_scaling.csv")
b1 = pd.read_csv(HERE / "B1_baseline_decomposition.csv")
tag_of = {("utd", 2): "utd2", ("utd", 4): "utd4", ("width", 256): "w256", ("width", 512): "w512", ("batch_size", 512): "b512",
          ("batch_size", 1024): "b1024"}
mx = dict(energy_mean=0.0, energy_sd=0.0, share_mean=0.0, share_sd=0.0, jpf=0.0, n=0)
for _, r in b1[b1.algo == "sac"].iterrows():
    q = s41[(s41.tag == "base") & (s41.env == r.env) & (s41.segment == r.segment)]
    if q.empty:
        continue
    q = q.iloc[0]
    mx["n"] += 1
    mx["energy_mean"] = max(mx["energy_mean"], abs(r.energy_j_mean - q.energy_j_mean) / q.energy_j_mean)
    mx["energy_sd"] = max(mx["energy_sd"], abs(r.energy_j_std_ddof1 - q.energy_j_sd) / q.energy_j_sd)
    if pd.notna(q.share_of_total_pct_mean):
        mx["share_mean"] = max(mx["share_mean"], abs(r.share_of_TOTAL_pct_mean_of_per_seed_shares - q.share_of_total_pct_mean))
        mx["share_sd"] = max(mx["share_sd"], abs(r.share_of_TOTAL_pct_std_of_per_seed_shares_ddof1 - q.share_of_total_pct_sd))
sac_cc = dict(compared_rows_baseline=mx["n"], max_rel_diff_energy_mean=mx["energy_mean"], max_rel_diff_energy_sd=mx["energy_sd"],
              max_abs_diff_share_mean_pp=mx["share_mean"], max_abs_diff_share_sd_pp=mx["share_sd"])
# sweep rows
mx2 = dict(n=0, e=0.0, f=0.0)
for _, r in b2[(b2.algo == "sac") & (~b2.is_baseline)].iterrows():
    tag = tag_of[(r.sweep_name, int(r.sweep_value))]
    q = s41[(s41.tag == tag) & (s41.env == r.env) & (s41.segment == r.segment)]
    if q.empty:
        continue
    q = q.iloc[0]
    mx2["n"] += 1
    mx2["e"] = max(mx2["e"], abs(r.energy_j_mean - q.energy_j_mean) / q.energy_j_mean)
    if q.flops_mean and r.matmul_flops_mean and q.flops_mean > 0:
        mx2["f"] = max(mx2["f"], abs(r.matmul_flops_mean - q.flops_mean) / q.flops_mean)
sac_cc["sweep_rows_compared"] = mx2["n"]
sac_cc["sweep_max_rel_diff_energy_mean"] = mx2["e"]
sac_cc["sweep_max_rel_diff_flops_mean"] = mx2["f"]
checks["sac_rows_vs_section_4_1_data"] = sac_cc

# ---- D4 table (full repr precision)
od = pd.read_csv(HERE / "D4_rq4_energy_power_duration.csv", dtype=str)
t = od.set_index("timestamp_dir").T
lines = ["| quantity | 20260904_122958 (2000 MHz) | 20260904_130915 (200 MHz, primary) |", "|---|---|---|"]
for q in t.index:
    if q in ("gpu_clock_mhz",):
        continue
    lines.append(f"| {q} | {t.loc[q, '20260904_122958']} | {t.loc[q, '20260904_130915']} |")
frag["d4_table"] = "\n".join(lines)
rep = (HERE / "d4_report.md").read_text(encoding="utf-8")
frag["d4_report"] = rep.replace("np.float64(", "").replace(")", ")") if False else rep
pr = pd.read_csv(REPO / "flop_analysis/output/per_run_energy_per_flop.csv")
pr = pr[pr.run_dir.str.contains("mbpo/HalfCheetah-v5/seed_0/") & (pr.segment == "TOTAL_MEASURED_TRAINING")]
frag["d4_pipeline_total_rows"] = "\n".join(
    ["| run_dir | included_in_cross_seed_avg | total_energy_joules | duration_s | mean_power_w | mean_gpu_power_w | mean_cpu_power_w | mean_ram_power_w | total_flops | energy_per_flop_j_per_flop |", "|---|---|---|---|---|---|---|---|---|---|"] +
    [f"| {r.run_dir} | {r.included_in_cross_seed_avg} | {r.total_energy_joules!r} | {r.duration_s!r} | {r.mean_power_w!r} | {r.mean_gpu_power_w!r} | {r.mean_cpu_power_w!r} | {r.mean_ram_power_w!r} | {r.total_flops!r} | {r.energy_per_flop_j_per_flop!r} |"
     for r in pr.itertuples()])

# ---- C3 stats from C4 table
c4 = pd.read_csv(HERE / "C4_per_run_table.csv")
g = c4.groupby(["algo", "env"]).agg(
    runs=("seed", "size"), train_episodes_min=("n_train_episodes", "min"), train_episodes_median=("n_train_episodes", "median"),
    train_episodes_max=("n_train_episodes", "max"), ep_len_min=("train_episode_length_min", "min"),
    ep_len_max=("train_episode_length_max", "max"), runs_with_a_train_incomplete_episode=("n_train_incomplete_episodes", lambda s: int((s > 0).sum())),
    last10pct_window_episodes_min=("n_last10pct_episodes", "min"), last10pct_window_episodes_max=("n_last10pct_episodes", "max"),
    runs_with_null_final_epoch_mean_episode_return=("return_final_epoch_mean_episode_return", lambda s: int(s.isna().sum())),
    last10pct_return_min=("return_last10pct_mean_train_episodes", "min"), last10pct_return_max=("return_last10pct_mean_train_episodes", "max")).reset_index()
cols = list(g.columns)
frag["c3_stats"] = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                              ["| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + " |" for row in g.itertuples(index=False)])

# ---- checks block
frag["checks_json"] = json.dumps(checks, indent=2, default=str)
(HERE / "checks.json").write_text(frag["checks_json"], encoding="utf-8")
cc = pd.read_csv(HERE / "C4_spearman_crosscheck.csv")
c5 = pd.read_csv(HERE / "C5_per_epoch_task_resolution.csv")
c5 = c5[c5.segment.isin(["rollout", "gradient_updates", "dynamics_model_update", "synthetic_rollout_generation", "world_model_pretrain"])]
c5cols = ["algo", "env", "segment", "tasks_per_run", "task_duration_s_min", "task_duration_s_median", "task_duration_s_max", "frac_tasks_shorter_than_1s"]
frag["c5_res"] = "\n".join(["| " + " | ".join(c5cols) + " |", "|" + "---|" * len(c5cols)] +
                           ["| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + " |"
                            for row in c5[c5cols].itertuples(index=False)])
frag["c4_crosscheck"] = "\n".join(["| env | analysis | n_groups | rho (newly computed) | p (newly computed) | rho (existing csv) | p (existing csv) |", "|---|---|---|---|---|---|---|"] +
                                  [f"| {r.env} | {r.analysis} | {r.n_groups} | {r.rho!r} | {r.p_value!r} | {r.existing_rho!r} | {r.existing_p!r} |" for r in cc.itertuples()])
frag["c2_summary_raw"] = (HERE / "C2_correlation_summary.csv").read_text(encoding="utf-8").strip()

tpl = (HERE / "main_template.md").read_text(encoding="utf-8")
missing = []


def sub(m):
    k = m.group(1)
    if k not in frag:
        missing.append(k)
        return m.group(0)
    return frag[k].rstrip("\n")


txt = re.sub(r"\{\{([a-z0-9_]+)\}\}", sub, tpl)
OUT.write_text(txt, encoding="utf-8")
print("wrote", OUT, len(txt), "chars; unresolved placeholders:", missing)
print(json.dumps(sac_cc, indent=2))
