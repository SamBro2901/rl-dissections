"""
NEWLY COMPUTED cross-check (item C4). Reads ONLY C4_per_run_table.csv (canonical rows) and recomputes the
config-level Spearman correlations of flop_analysis/correlation_analysis.py (unit of observation = one group = mean over
its 5 seeds; per env; return = last-10% mean of train episodes; FLOPs = TOTAL_MEASURED_TRAINING matmul FLOPs) and compares
them with flop_analysis/output/correlation/correlation_summary.csv.  Also writes C4_spearman_crosscheck.csv.
Run from repo root: python contexts/Section_3.1_3.2_3.3_data/c4_crosscheck.py
"""
from pathlib import Path
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
c4 = pd.read_csv(HERE / "C4_per_run_table.csv")
ex = pd.read_csv(REPO / "flop_analysis" / "output" / "correlation" / "correlation_summary.csv").set_index(["env", "analysis"])
rows = []
gm = (c4.groupby(["env", "group_id"])
        .agg(flops=("total_matmul_flops_TOTAL_MEASURED_TRAINING", "mean"), ret=("return_last10pct_mean_train_episodes", "mean"),
             energy=("total_energy_j_TOTAL_MEASURED_TRAINING", "mean"), n=("seed", "size")).reset_index())
for env, g in gm.groupby("env"):
    for name, col in (("return_vs_flops", "ret"), ("energy_vs_flops", "energy")):
        r = stats.spearmanr(g.flops, g[col])
        e = ex.loc[(env, name)]
        rows.append(dict(label="NEWLY COMPUTED", env=env, analysis=name, grouping="per env, one point per group (mean of 5 seeds)",
                         n_groups=len(g), rho=r.statistic, p_value=r.pvalue, existing_rho=e.spearman_rho, existing_p=e.p_value,
                         abs_diff_rho=abs(r.statistic - e.spearman_rho), abs_diff_p=abs(r.pvalue - e.p_value)))
# pooled over both envs (not in the existing analysis; shown only as the count/size of alternative groupings)
out = pd.DataFrame(rows)
out.to_csv(HERE / "C4_spearman_crosscheck.csv", index=False)
print(out.to_string(index=False))
# group counts for alternative groupings (sizes only)
print(gm.groupby("env").size().to_dict(), "per-algo-env group counts:",
      c4.groupby(["algo", "env"]).group_id.nunique().to_dict())
