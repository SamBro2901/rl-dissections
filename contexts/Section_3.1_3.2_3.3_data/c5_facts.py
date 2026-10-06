"""
Item C5 facts (no metric is computed): resolution of per-epoch task energy/duration in the per-task CodeCarbon csv
(results/**/emissions_base_*.csv, canonical runs) and of the return logging in training_metrics.json.
Run from repo root: python contexts/Section_3.1_3.2_3.3_data/c5_facts.py   ->  C5_per_epoch_task_resolution.csv
"""
import csv, json, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CANON = {331, 958, 14577, 43611, 85062}
rows = []
for meta in sorted((REPO / "results").glob("*/*/seed_*/*/metadata.json")):
    m = json.loads(meta.read_text())
    if m["seed"] not in CANON:
        continue
    d = meta.parent
    f = next(d.glob("emissions_*.csv"))
    for r in csv.DictReader(open(f, newline="")):
        mm = re.match(r"^(.*)_(\d+)$", r["task_name"])
        rows.append(dict(algo=m["algo_name"], env=m["env_id"], seed=m["seed"], run=f"{m['seed']}/{d.name}", segment=mm.group(1) if mm else r["task_name"],
                         epoch=int(mm.group(2)) if mm else -1, duration_s=float(r["duration"]), energy_kwh=float(r["energy_consumed"])))
df = pd.DataFrame(rows)
g = (df.groupby(["algo", "env", "segment"]).agg(n_task_rows_total=("duration_s", "size"), n_runs=("run", "nunique"),
        task_duration_s_min=("duration_s", "min"), task_duration_s_median=("duration_s", "median"), task_duration_s_max=("duration_s", "max"),
        task_energy_kwh_min=("energy_kwh", "min"), task_energy_kwh_median=("energy_kwh", "median"), task_energy_kwh_max=("energy_kwh", "max"),
        frac_tasks_shorter_than_1s=("duration_s", lambda s: float((s < 1.0).mean()))).reset_index())
g["tasks_per_run"] = g.n_task_rows_total / g.n_runs
g.to_csv(HERE / "C5_per_epoch_task_resolution.csv", index=False)
pd.set_option("display.width", 250)
print(g[g.segment.isin(["rollout", "gradient_updates", "dynamics_model_update", "synthetic_rollout_generation"])].round(4).to_string(index=False))
# episodes/epoch record resolution
tm = json.loads(next((REPO / "results/sac/HalfCheetah-v5/seed_331").glob("*/training_metrics.json")).read_text())
print("epochs[i].env_step first/last:", tm["epochs"][0]["env_step"], tm["epochs"][-1]["env_step"], "episodes[0..2] env_step:", [e["env_step"] for e in tm["episodes"][:3]])
