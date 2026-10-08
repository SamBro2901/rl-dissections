"""Renders contexts/section_5.2.md from the CSVs written by section_5_2_analysis.py (run that first)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
D = REPO / "contexts" / "section_5.2_data"
ENVS = ["HalfCheetah-v5", "Ant-v5"]
ALGOS = ["sac", "td3", "mbpo", "tdmpc2"]


def g(x, n=6):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{n}g}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


ct = pd.read_csv(D / "d1_config_table.csv")
d2 = pd.read_csv(D / "d2_total_correlation.csv")
d3 = pd.read_csv(D / "d3_segment_correlation.csv")
d4 = pd.read_csv(D / "d4_per_algo_total_correlation.csv")
d5 = pd.read_csv(D / "d5_per_algo_segment_correlation.csv")
fits = pd.read_csv(D / "a1_linear_fits.csv")
pairs = pd.read_csv(D / "a2_delta_pairs.csv")
chk = pd.read_csv(D / "x1_check_vs_chapter4.csv")
cm = fits[fits.fit_basis == "config_means"]
pr = fits[fits.fit_basis == "per_run"]


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


L = []
w = L.append

w("# Section 5.2 — Energy versus compute: are FLOP-based energy estimates trustworthy? — context\n")
w("Data context for thesis Section 5.2. Facts, numbers and provenance only — no interpretation, no LaTeX. "
  "280 canonical runs (56 configurations × 5 seeds [331, 958, 14577, 43611, 85062]); dev runs ignored. "
  "All energies are **gross** (idle floor included, nothing subtracted), joules, seed-mean over the five seeds. "
  "HalfCheetah-v5 (HC): 27 configurations; Ant-v5: 29 (Ant has the two extra MBPO rollout-length configurations `rollout1`, `rollout15`). "
  "The two environments are always reported separately. Returns are not used anywhere in this section.\n")
w(f"Generated 2026-10-08 from git HEAD `{git('rev-parse', '--short', 'HEAD')}` (branch master); Windows dev checkout with synced `results/`. "
  "Python 3.13 / pandas 3.0.5 / scipy 1.18.1.\n")

# ---------------- methods
w("## 0. Methods note\n")
w("**Scripts** (both new, under `flop_analysis/`, read-only on `results/` and `flop_analysis/output/`): "
  "`flop_analysis/section_5_2_analysis.py` (computes everything, writes `contexts/section_5.2_data/*.csv`) and "
  "`flop_analysis/section_5_2_render.py` (renders this file from those CSVs). Re-run from the repo root: "
  "`.venv/Scripts/python -I -W ignore flop_analysis/section_5_2_analysis.py` then `.venv/Scripts/python flop_analysis/section_5_2_render.py`. "
  "Input is `flop_analysis/output/per_run_energy_per_flop.csv` (canonical seeds, HC and Ant), the pipeline's J/FLOP definitions unchanged.\n")
w("**Definitions**")
w("- *Configuration* = one (algorithm, environment, hyper-parameter setting); label tags: `base`; `utdK` = updates per env step K; `wN` = hidden width N (baseline 1024); `bN` = batch size N (baseline 256 for SAC/MBPO, 100 for TD3); `rolloutN` = MBPO max model-rollout length N (Ant only; Ant baseline is 25, HC baseline is the fixed length 1); `numqN`, `horizonN` = TD-MPC2 Q-ensemble size (baseline 5) and MPPI horizon (baseline 3).")
w("- *Total* = `TOTAL_MEASURED_TRAINING` energy and matmul FLOPs (rollout + gradient_updates + MBPO dynamics/synthetic-rollout + TD-MPC2 world_model_pretrain; warmup and idle excluded; `target_update` elementwise ops excluded from FLOPs). Per-run total energy equals the sum of its measured segments to a relative error of 3.6e-16.")
w("- *gradient_updates (GU)* has no row of its own in the pipeline CSV. GU energy = `buffer_sample` + `critic_update` + `actor_update` + `target_update` energy (the directly measured `gradient_updates` task, which the pipeline splits into those four allocated parts); GU FLOPs = `critic_update` + `actor_update` matmul FLOPs (`buffer_sample` has none; `target_update` is elementwise and excluded). The allocated sub-segments `critic_update`/`actor_update` are not analysed separately here, per the task.")
w("- *Seed-mean* energy and FLOPs per configuration; *J/FLOP* = mean energy / mean FLOPs (ratio of means), *sd* = sample sd (ddof = 1) of the five per-seed ratios. Tables show pJ/FLOP (1 pJ = 1e-12 J). MBPO FLOPs differ slightly between seeds (dynamics-model early stopping, variable synthetic-rollout termination); the FLOPs shown for MBPO are seed-means (min/max in the CSV). For all other algorithms FLOPs are identical across seeds.")
w("- *Spearman*: `scipy.stats.spearmanr` (mid-ranks for ties; two-sided asymptotic p), FLOPs vs energy over configuration means. For n ≤ 9 the CSVs also give an exact two-sided permutation p (all n! orderings). *Log-log fit*: `scipy.stats.linregress(log10 FLOPs, log10 energy[J])` on configuration means: slope, intercept (log10 J at 1 FLOP), R².")
w("- *Linear fit* E = a + b·F: ordinary least squares on **configuration-mean points** (one point per sweep value, baseline included). a [J], b [pJ/FLOP], R², residual SE = sqrt(SSR/(n−2)) [J]. The same fits on per-run points (5 per configuration, n = 15 or 20) are in `a1_linear_fits.csv` (`fit_basis = per_run`); for SAC, TD3 and TD-MPC2 they are numerically identical to the configuration-mean fits (FLOPs are equal across seeds; maximum relative difference of the slope 2.3e-15), for MBPO they differ (see Section 6).")
w("- *Sweeps and fit rows*: UTD {1,2,4}, width {256,512,1024} and batch size ({256,512,1024}; TD3 {100,256,512,1024}) each holding the other factors at the baseline; for MBPO the rollout schedule is held at the environment baseline. TD-MPC2: `num_q` {3,5,7} at horizon 3, and `horizon` {1,3,5} at num_q 5. The baseline configuration is a point of every sweep of its (algorithm, environment). The MBPO rollout-length sweep (Ant) was not part of the request and is not fitted.")
w("- *Delta pairs*: ΔE/ΔF between **adjacent** sweep values on configuration means, as in `flop_dashboard.py` (`compute_delta_pairs`, previous-value mode); pairs with |ΔF| < 1 % of the reference FLOPs are dropped (shown as n/a). With one pair per interval and one sweep per (algorithm, environment) there is no pooling.")
w("- **Independence caveat.** The configurations are not independent observations: they come from one-factor-at-a-time sweeps around shared baselines (each baseline appears in several sweeps and in the pooled sets, and many configurations share the same algorithm, environment and hardware). All p-values in this file are descriptive, not inferential. Configurations with exactly equal FLOPs (SAC `utd2`/`b512` and `utd4`/`b1024`, in both environments) are ties in the rank correlation.")
w("- **Small-n caveat.** A 3-point linear fit has one residual degree of freedom (n − 2 = 1); a 4-point fit has two. R² is then high almost by construction and the residual SE is computed with n − 2 = 1 in the denominator (not corrected).\n")

# ---------------- (d) 1
w("## 1. Part (d) — per-configuration table (all 56 configurations)\n")
w("`d1_config_table.csv` holds every column (energy mean/sd, FLOPs mean/min/max, J/FLOP and its sd, for total and each segment). Below, per environment: E [J] gross seed-mean, F = matmul FLOPs seed-mean, pJ/F = J/FLOP ×1e12 (ratio of means; ±sd of per-seed ratios for the total). "
  "`buffer_sample` and `target_update` have no matmul FLOPs and are not listed as segments (their energy is inside GU).\n")
for env in ENVS:
    w(f"### {env}\n")
    rows = []
    for r in ct[ct.env == env].itertuples():
        rows.append([r.algo, r.config, g(r.total_E_J), g(r.total_F), f"{g(r.total_J_per_F*1e12)} ± {g(r.total_J_per_F_sd*1e12,3)}",
                     g(r.rollout_E_J), g(r.rollout_F), g(r.rollout_J_per_F*1e12),
                     g(r.gradient_updates_E_J), g(r.gradient_updates_F), g(r.gradient_updates_J_per_F*1e12)])
    w(table(["algo", "config", "total E [J]", "total F", "total pJ/F", "rollout E [J]", "rollout F", "rollout pJ/F", "GU E [J]", "GU F", "GU pJ/F"], rows))
    w("")
    ex = ct[(ct.env == env) & (ct.algo.isin(["mbpo", "tdmpc2"]))]
    rows = []
    for r in ex.itertuples():
        if r.algo == "mbpo":
            rows.append([r.algo, r.config, g(r.dynamics_model_update_E_J), g(r.dynamics_model_update_F), g(r.dynamics_model_update_J_per_F * 1e12),
                         g(r.synthetic_rollout_generation_E_J), g(r.synthetic_rollout_generation_F), g(r.synthetic_rollout_generation_J_per_F * 1e12), "", "", ""])
        else:
            rows.append([r.algo, r.config, "", "", "", "", "", "", g(r.world_model_pretrain_E_J), g(r.world_model_pretrain_F), g(r.world_model_pretrain_J_per_F * 1e12)])
    w(table(["algo", "config", "dynamics_model_update E [J]", "F", "pJ/F", "synthetic_rollout_generation E [J]", "F", "pJ/F", "world_model_pretrain E [J]", "F", "pJ/F"], rows))
    w("")

# ---------------- (d) 2
w("## 2. Part (d) — training total, FLOPs vs energy, pooled over algorithms (per environment)\n")
rows = []
for r in d2.itertuples():
    rows.append([r.env, r.n, g(r.spearman_rho, 4), f"{r.spearman_p:.3g}", g(r.loglog_slope, 4), g(r.loglog_slope_se, 3), g(r.loglog_intercept, 4), g(r.loglog_r2, 4), f"{g(r.flops_min, 3)} – {g(r.flops_max, 3)}"])
w(table(["env", "n configs", "Spearman ρ", "p (asymptotic)", "log-log slope", "slope SE", "intercept (log10 J)", "R²", "FLOPs range"], rows))
w("\n**Reproduction of the thesis values** (ρ 0.836 / 0.846; slope 0.47 / 0.51; R² 0.61 / 0.63): reproduced (ρ 0.8357 / 0.8463, slope 0.4715 / 0.5071, R² 0.6110 / 0.6253), identical to `flop_analysis/output/correlation/correlation_summary.csv` (same 27 / 29 configuration points).\n")

# ---------------- (d) 3
w("## 3. Part (d) — per measured segment, pooled over algorithms (per environment)\n")
rows = []
for r in d3.itertuples():
    ex = "" if np.isnan(r.spearman_p_exact_perm) else f"{r.spearman_p_exact_perm:.3g}"
    rows.append([r.env, r.segment, r.n, g(r.spearman_rho, 4), f"{r.spearman_p:.3g}", ex, g(r.loglog_slope, 4), g(r.loglog_intercept, 4), g(r.loglog_r2, 4), g(r.flops_range_orders, 3)])
w(table(["env", "segment", "n", "Spearman ρ", "p (asymptotic)", "p (exact perm., n≤9)", "log-log slope", "intercept", "R²", "FLOPs range [orders of mag.]"], rows))
w("\nPooled `rollout` and `gradient_updates` use all configurations of the environment (n = 27 / 29). `dynamics_model_update` and `synthetic_rollout_generation` exist for MBPO only (n = 7 on HC, 9 on Ant); `world_model_pretrain` for TD-MPC2 only (n = 5). Those four rows are **small-n, single-algorithm** results. Asymptotic p at ρ = 1 with n = 5 is a numerical artefact of the t approximation (1.4e-24); the exact permutation p is 2/120 = 0.0167. Per-algorithm segment statistics are in `d5_per_algo_segment_correlation.csv`.\n")

# ---------------- (d) 4
w("## 4. Part (d) — training total per algorithm and environment (small n)\n")
rows = []
for r in d4.itertuples():
    rows.append([r.env, r.algo, r.n, g(r.spearman_rho, 4), f"{r.spearman_p:.3g}", f"{r.spearman_p_exact_perm:.3g}", g(r.loglog_slope, 4), g(r.loglog_intercept, 4), g(r.loglog_r2, 4), g(r.flops_range_orders, 3)])
w(table(["env", "algo", "n", "Spearman ρ", "p (asymptotic)", "p (exact perm.)", "log-log slope", "intercept", "R²", "FLOPs range [orders]"], rows))
w("\n**Small-n warning**: n = 5–9 configurations per row, all from one-factor-at-a-time sweeps (non-independent), so these are descriptive. SAC's HC and Ant rows have identical ρ because the FLOPs ranking of SAC configurations is the same in both environments and the energy ranking coincides. For TD-MPC2 the ranking by FLOPs and by energy is identical in both environments (ρ = 1).\n")

# ---------------- (a) fits
w("## 5. Part (a) — fixed versus marginal cost: E = a + b·F per sweep\n")
w("Points = seed-mean (F, E) of the sweep's configurations (n = 3; TD3 batch sweep n = 4). **n = 3 fits have one residual degree of freedom; n = 4, two.** "
  "`rollout` fits for the UTD and batch sweeps are undefined: the rollout FLOPs (batch-size-1 actor forward × env steps) are identical across the sweep values, so the regressor is constant (listed n/a). Rollout is not affected by those factors in FLOP terms. "
  "Per-run versions: `a1_linear_fits.csv`.\n")
ORDER = {"UTD": 0, "width": 1, "batch": 2, "num_q": 3, "horizon": 4}
for env in ENVS:
    w(f"### {env}\n")
    rows = []
    sub = cm[cm.env == env].copy()
    sub["algo"] = pd.Categorical(sub.algo, ALGOS, ordered=True)
    sub["so"] = sub.sweep.map(ORDER)
    sub["sg"] = sub.segment.map({"total": 0, "gradient_updates": 1, "rollout": 2, "dynamics_model_update": 3})
    for r in sub.sort_values(["algo", "so", "sg"]).itertuples():
        note = "architecture change" if r.algo == "tdmpc2" else ""
        rows.append([r.algo, r.sweep, r.values, r.segment, r.n, g(r.a_J), g(r.b_pJ_per_FLOP, 5), g(r.r2, 5), g(r.rse_J), note])
    w(table(["algo", "sweep", "values", "segment", "n pts", "a [J]", "b [pJ/FLOP]", "R²", "residual SE [J]", "note"], rows))
    w("")
w("**TD-MPC2 note.** `num_q` changes the size of the critic ensemble (the Q-network count in the world model update, in the target update and in planning value evaluation) and `horizon` changes the length of the MPPI planning rollout in latent space and the unrolled world-model loss; both change the *architecture of the planner or world model*, not only the per-call batch/size of an unchanged computation. The slope b of these fits is therefore the marginal cost along a path that alters what is computed, not along a fixed workload.\n")

# ---------------- (a) delta
w("## 6. Part (a) — adjacent-value ΔE/ΔF (marginal J per extra FLOP)\n")
w("pJ/FLOP = ΔE/ΔF × 1e12 between configuration means of adjacent sweep values (same segments as Section 5; n/a = dropped, |ΔF| < 1 % of reference). Full columns in `a2_delta_pairs.csv`.\n")
for env in ENVS:
    w(f"### {env}\n")
    sub = pairs[pairs.env == env].copy()
    sub["algo"] = pd.Categorical(sub.algo, ALGOS, ordered=True)
    sub["so"] = sub.sweep.map(ORDER)
    sub["sg"] = sub.segment.map({"total": 0, "gradient_updates": 1, "rollout": 2, "dynamics_model_update": 3})
    rows = []
    for r in sub.sort_values(["algo", "so", "sg", "to_value"]).itertuples():
        rows.append([r.algo, r.sweep, r.segment, f"{int(r.from_value)}→{int(r.to_value)}", g(r.dE_J), g(r.dF), g(r.dE_per_dF_pJ, 5)])
    w(table(["algo", "sweep", "segment", "from→to", "ΔE [J]", "ΔF", "ΔE/ΔF [pJ/FLOP]"], rows))
    w("")

# ---------------- (a) SAC check
w("## 7. Check of the SAC values quoted in the thesis\n")
sac = pr[(pr.algo == "sac") & pr.segment.isin(["total", "gradient_updates"])]
rows = []
for r in sac.sort_values(["sweep", "env", "segment"], key=lambda s: s.map(ORDER) if s.name == "sweep" else s).itertuples():
    rows.append([r.sweep, r.env, r.segment, r.n, g(r.a_J), g(r.b_pJ_per_FLOP, 4), g(r.r2, 4)])
w("Per-run fits (n = 15, as in Section 4.1) — identical to the configuration-mean fits of Section 5 for SAC:\n")
w(table(["sweep", "env", "segment", "n", "a [J]", "b [pJ/FLOP]", "R²"], rows))
s = sac
def rng(sw, col):
    x = s[s.sweep == sw][col]
    return x.min(), x.max()
w("")
w(f"- Width marginal slope: {rng('width','b_pJ_per_FLOP')[0]:.2f}–{rng('width','b_pJ_per_FLOP')[1]:.2f} pJ/FLOP (thesis: ≈16.5–17.0) — **reproduces**.")
w(f"- Batch-size marginal slope: {rng('batch','b_pJ_per_FLOP')[0]:.2f}–{rng('batch','b_pJ_per_FLOP')[1]:.2f} pJ/FLOP (thesis: 7.7–8.5) — **reproduces**.")
w(f"- UTD marginal slope: {rng('UTD','b_pJ_per_FLOP')[0]:.2f}–{rng('UTD','b_pJ_per_FLOP')[1]:.2f} pJ/FLOP (thesis: 62.8–66.2) — **reproduces**.")
wb = s[s.sweep.isin(["width", "batch"])]
w(f"- Width and batch intercepts: {wb.a_J.min()/1e3:.1f}–{wb.a_J.max()/1e3:.1f} kJ over total and GU, both environments (thesis: ≈44–58 kJ) — **reproduces** (width GU/total 43.9–48.4 kJ, batch 53.8–58.1 kJ). UTD intercepts are small (−0.03 to 5.3 kJ).\n")

# ---------------- surprises
w("## 8. Inconsistencies, surprises and things not computed\n")
w("**Consistency with Chapter 4 — none found.** For all 56 configurations and every segment present (210 comparisons of seed-mean energy, FLOPs, J/FLOP and sd of per-seed J/FLOP against `contexts/section_4.1_data/sac_cross_seed_summary.csv`, `Section_4.2_data/td3_…`, `Section_4.3_data/mbpo_…`, `Section_4.4_data/tdmpc2_…`) the maximum relative deviation is 4.4e-16 (energy), 0 (FLOPs), 4.4e-16 (J/FLOP), 4.5e-14 (sd) — `x1_check_vs_chapter4.csv`. All 56 configuration total energies also match `flop_analysis/output/cross_seed_energy_per_flop.csv` to 1e-9. The pooled correlation values match `correlation_summary.csv`.\n")
w("**Surprising or noteworthy**")
items = [
    "**Config-mean vs per-run fits differ for MBPO only.** Because MBPO FLOPs vary per seed, its per-run and configuration-mean fits are not the same regression (maximum relative slope difference 0.89 over the MBPO fit rows); SAC, TD3, TD-MPC2 fits are identical (difference ≤ 2.3e-15). The tables here use configuration means.",
    "**MBPO `dynamics_model_update` does not scale with the swept factor.** The dynamics ensemble is a fixed 7×(200×4) network trained per epoch with early stopping; its FLOPs per configuration range only 2.2e14–2.8e14 (HC) and 3.6e14–5.0e14 (Ant) across seeds and sweeps with no monotone relation to UTD/width/batch (e.g. HC energy 137–150 kJ). The dynamics linear fits are therefore dominated by early-stopping noise: HC UTD a = −15.6 kJ, b = 635 pJ/FLOP; HC width a = −66.7 kJ; Ant width R² = 0.0053 (b = 43.7 pJ/FLOP); Ant batch R² = 0.33; Ant UTD R² = 0.47. Negative intercepts appear in these fits.",
    "**MBPO total fits carry the dynamics-model block in the intercept** (a ≈ 138–240 kJ vs 1–54 kJ for the MBPO GU fits), and the MBPO total slope differs from its GU slope (e.g. HC width total 23.2 vs GU 17.6 pJ/FLOP; HC batch 4.47 vs 7.02; Ant batch total R² = 0.68).",
    "**Pooled vs within-algorithm log-log slopes.** Pooled total slope 0.47 / 0.51; within algorithm 0.17–0.28 for SAC/TD3/MBPO and 0.71 for TD-MPC2 (both environments); within-algorithm R² 0.47–0.57 (SAC/TD3/MBPO) and 0.99 (TD-MPC2).",
    "**Rank correlation is high even where log-log R² is moderate** (pooled ρ 0.84–0.85, R² 0.61–0.63). Rollout alone: ρ 0.88, R² 0.95–0.96 (slope 0.38–0.42); pooled GU: ρ 0.84–0.87, R² 0.62–0.64.",
    "**Marginal cost differs across sweeps.** Marginal costs b of SAC (width ≈ 16.5–17.0, batch ≈ 7.7–8.5, UTD ≈ 62.8–66.2 pJ/FLOP) differ by a factor of ≈ 8 between the batch and UTD sweeps of the same algorithm. Rollout slopes for the width sweep: SAC ≈ 1.07, TD3 0.80–1.08, MBPO 0.58–1.03 nJ/FLOP (batch-size-1 matmuls).",
    "**TD3 UTD marginal cost (103–107 pJ/FLOP) is higher than SAC's (≈ 63–66) and MBPO's GU (≈ 62–66).**",
    "**TD-MPC2 `horizon` fits have large residuals in absolute terms** (total residual SE 13.4–17.2 kJ; GU R² 0.976–0.988), larger than the `num_q` total fits (0.8–3.6 kJ).",
    "**Equal FLOPs, different energy.** SAC `utd2` and `b512` have identical total FLOPs (HC 1.753959e15; energy 118,024 J vs 73,024 J), as do `utd4` and `b1024` (3.507702e15; 234,959 J vs 82,366 J); same pattern on Ant (123,182 vs 76,499 J; 241,327 vs 89,057 J). MBPO `utd2`/`b512` are within ≈ 1 % of each other in FLOPs (HC 2.02e15 vs 2.00e15) with energy 268,851 vs 209,101 J. TD3 has no such tie.",
]
for it in items:
    w("- " + it)
w("")
w("**Could not compute / not done**")
for it in [
    "Rollout-segment linear fits and ΔE/ΔF for the UTD and batch sweeps of SAC, TD3 and MBPO: undefined (rollout FLOPs constant across those sweeps).",
    "A GU row is not in `cross_seed_energy_per_flop.csv`; GU was reconstructed as described in Section 0 (reconciled to the Chapter 4 `gradient_updates` rows to 4.4e-16, so this is not a source of difference).",
    "MBPO rollout-length sweep (Ant, `rollout1`/`rollout15`) is in the config table and the pooled statistics but has no linear fit or delta pairs (not requested).",
    "No inferential p-values for the sweeps: n = 3–4 per fit and non-independent configurations (descriptive only). No confidence intervals on the linear-fit parameters are given (one residual df for n = 3).",
    "Per-segment statistics for `target_update` (elementwise) and `buffer_sample` (no FLOPs) and the allocated sub-segments `critic_update`/`actor_update` were excluded as instructed.",
]:
    w("- " + it)
w("")
w("## 9. Index of produced files\n")
w("`contexts/section_5.2_data/`: `d1_config_table.csv` (56 rows), `d2_total_correlation.csv`, `d3_segment_correlation.csv`, `d4_per_algo_total_correlation.csv`, `d5_per_algo_segment_correlation.csv` (supplement), `a1_linear_fits.csv` (config-mean and per-run fits), `a2_delta_pairs.csv`, `x1_check_vs_chapter4.csv`, `per_run_segments_5_2.csv` (280 rows: per-run total, GU and segment energies/FLOPs). Scripts: `flop_analysis/section_5_2_analysis.py`, `flop_analysis/section_5_2_render.py`.")

(REPO / "contexts" / "section_5.2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("written", sum(len(x) for x in L), "chars")
