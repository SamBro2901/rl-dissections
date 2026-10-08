"""Render contexts/section_5.3.md from contexts/section_5.3_data/*.csv (4 significant digits). No interpretation."""
from pathlib import Path
import subprocess

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
D = REPO / "contexts" / "section_5.3_data"
ENVS = ["HalfCheetah-v5", "Ant-v5"]
NAME = {"sac": "SAC", "td3": "TD3", "mbpo": "MBPO", "tdmpc2": "TD-MPC2"}
SW = ["baseline", "utd2", "utd4", "w256", "w512", "b256", "b512", "b1024"]
SWL = {"baseline": "Baseline (UTD 1, width 1024)", "utd2": "UTD 2", "utd4": "UTD 4", "w256": "Width 256", "w512": "Width 512",
       "b256": "Batch 256", "b512": "Batch 512", "b1024": "Batch 1024"}
o1 = pd.read_csv(D / "o1_orderings.csv")
ref = pd.read_csv(D / "o1b_reference_rows.csv")
sw = pd.read_csv(D / "o2_swaps_vs_baseline.csv")
sl = pd.read_csv(D / "o3_seed_level.csv")
pw = pd.read_csv(D / "o3b_pairwise_seed_counts.csv")
ro = pd.read_csv(D / "r1_mbpo_rollout_sweep.csv")
tm = pd.read_csv(D / "t1_tdmpc2_sweep.csv")
rg = pd.read_csv(D / "t3_algo_ranges_vs_tdmpc2.csv")
x1 = pd.read_csv(D / "x1_consistency_checks.csv")

def g4(v):
    t = f"{v:.4g}"
    if "e" in t and 1e-3 <= abs(v) < 1e7:
        t = f"{float(t):.0f}"
    return t
pj = lambda v: g4(v * 1e12)          # J/FLOP -> pJ/FLOP
fmt = {"energy": (g4, "J"), "jpf": (pj, "pJ/FLOP")}
L = []
w = L.append


def head():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    w("# Section 5.3 — Robustness of the cross-algorithm comparison across sweeps — context\n")
    w("Data context for thesis Section 5.3. Facts, numbers and provenance only — no interpretation, no LaTeX. Canonical seeds [331, 958, 14577, 43611, 85062] only (dev runs ignored); 280 runs. All energies are **gross** (idle floor included, nothing subtracted), joules, training total (`TOTAL_MEASURED_TRAINING`). J/FLOP is the pipeline's training-total J/FLOP, shown as pJ/FLOP (1 pJ = 1e-12 J). Four significant digits here, full precision in the CSVs. Returns are not used anywhere; Kendall's tau is not computed. HalfCheetah-v5 (HC) and Ant-v5 (Ant) are always reported separately.\n")
    w(f"Generated 2026-10-08 from git HEAD `{commit}` (branch master); Windows dev checkout with synced `results/`.\n")
    w("## 0. Methods note\n")
    w("**Scripts** (both new, under `flop_analysis/`, read-only on `results/` and `flop_analysis/output/`): `flop_analysis/section_5_3_analysis.py` (computes everything, writes `contexts/section_5.3_data/*.csv`; it imports `load()` — the canonical-seed run table — from `flop_analysis/section_5_2_analysis.py`) and `flop_analysis/section_5_3_render.py` (renders this file). Re-run from the repo root: `.venv/Scripts/python -I -W ignore flop_analysis/section_5_3_analysis.py` then `.venv/Scripts/python flop_analysis/section_5_3_render.py`. Input: `flop_analysis/output/per_run_energy_per_flop.csv`.\n")
    w("**Definitions**")
    w("- *Per-seed values*: energy E = the run's `TOTAL_MEASURED_TRAINING` energy; J/FLOP = E / F of the same run (F = its total matmul FLOPs).")
    w("- *Configuration value* (the mean in every table): energy = mean of the 5 per-seed E, sd = sample sd (ddof = 1) over the 5 seeds. J/FLOP = mean E / mean F (ratio of means, the 5.2 convention); sd = sample sd of the 5 per-seed ratios. MBPO's F differs slightly between seeds; SAC, TD3 and TD-MPC2 have identical F across seeds.")
    w("- *Ordering* = SAC, TD3, MBPO sorted lowest → highest on the configuration value. *Ratio to previous* = value of this algorithm / value of the next-lower algorithm in the same ordering. TD-MPC2 is in no ordering.")
    w("- *Swap* relative to the baseline: a pair of algorithms whose relative order differs from the baseline ordering (baseline ordering of the same environment and metric).")
    w("- *Seed-level check*: paired seeds (the same seed number for all three algorithms). Per seed the three algorithms are sorted on that seed's own value of the metric; the count is the number of seeds (of 5) whose full ordering equals the ordering of the configuration means (and, additionally, the baseline ordering). Pairwise counts (seeds in which the mean-lower algorithm is below the mean-higher one) are in `o3b_pairwise_seed_counts.csv`.")
    w("- *Runs entering each ordering row* (3 algorithms × 5 seeds = 15 runs per environment per row): baseline = `base` of SAC, TD3, MBPO (SAC/MBPO batch 256; **TD3 batch 100, warmup 10000**; MBPO HC rollout length fixed 1, Ant 1→25). UTD 2 / UTD 4 = `utd2` / `utd4` of each (TD3 stays batch 100, MBPO keeps its baseline rollout schedule). Width 256 / 512 = `w256` / `w512` of each (TD3 batch 100). Batch 256 = SAC `base` and MBPO `base` (their baseline batch is already 256, so those runs are the same as in the baseline row) with TD3 `b256` (its batch-sweep run, warmup 10000 as all TD3 runs); batch 512 / 1024 = `b512` / `b1024` of all three. The TD3 batch-100 baseline is listed as a reference row beneath the batch tables.")
    w("- *Rollout share* (TD-MPC2): per-seed `rollout` energy / per-seed total energy; table gives the seed-mean and sample sd of that share (same convention as the Chapter 4.4 `share_of_total_pct`). The ratio of means is in the CSV.")
    w("- Configuration tags as in Sections 5.2/4.x: `base`, `utdK`, `wN`, `bN`, `rolloutN` (MBPO max model-rollout length, Ant only; baseline L_max = 25, `rollout_min_length` = 1 throughout), `numqN`, `horizonN`.\n")


def ordering_tables():
    w("## 1. Orderings per sweep value (SAC, TD3, MBPO)\n")
    w("Each table: algorithms lowest → highest. `mean` = seed-mean, `sd` = sample sd over seeds, `×prev` = ratio to the next-lower algorithm. Source: `o1_orderings.csv`.\n")
    for env in ENVS:
        w(f"### 1.{ENVS.index(env) + 1} {env}\n")
        for key in SW:
            w(f"**{SWL[key]}**\n")
            sub = o1[(o1.env == env) & (o1.sweep == key)]
            w("| rank | energy: algorithm (config) | mean [J] | sd [J] | ×prev | J/FLOP: algorithm (config) | mean [pJ/FLOP] | sd [pJ/FLOP] | ×prev |")
            w("|---|---|---|---|---|---|---|---|---|")
            e = sub[sub.metric == "energy"].sort_values("rank")
            j = sub[sub.metric == "jpf"].sort_values("rank")
            for (_, a), (_, b) in zip(e.iterrows(), j.iterrows()):
                pa = "—" if pd.isna(a.ratio_to_previous) else g4(a.ratio_to_previous)
                pb = "—" if pd.isna(b.ratio_to_previous) else g4(b.ratio_to_previous)
                w(f"| {a['rank']} | {NAME[a.algo]} (`{a.config}`) | {g4(a['mean'])} | {g4(a.sd)} | {pa} | {NAME[b.algo]} (`{b.config}`) | {pj(b['mean'])} | {pj(b.sd)} | {pb} |")
            w("")
            if key == "baseline":
                t = ref[(ref.env == env) & (ref.algo == "tdmpc2")].set_index("metric")
                w(f"TD-MPC2 baseline (`base`; **not part of the ordering**): energy {g4(t.loc['energy', 'mean'])} J (sd {g4(t.loc['energy', 'sd'])}); J/FLOP {pj(t.loc['jpf', 'mean'])} pJ/FLOP (sd {pj(t.loc['jpf', 'sd'])}).\n")
            if key in ("b256", "b512", "b1024") and key == "b1024":
                t = ref[(ref.env == env) & (ref.algo == "td3")].set_index("metric")
                w(f"Reference row for the batch tables — TD3 batch-100 baseline (`base`, warmup 10000; this is TD3's entry in the baseline ordering above): energy {g4(t.loc['energy', 'mean'])} J (sd {g4(t.loc['energy', 'sd'])}); J/FLOP {pj(t.loc['jpf', 'mean'])} pJ/FLOP (sd {pj(t.loc['jpf', 'sd'])}).\n")


def swaps():
    w("## 2. Pairs that swap order relative to the baseline ordering\n")
    w("Source: `o2_swaps_vs_baseline.csv`. `a<b -> b<a` means: a is below b in the baseline ordering and above b in this sweep value.\n")
    w("| env | sweep value | energy ordering | energy swaps | J/FLOP ordering | J/FLOP swaps |")
    w("|---|---|---|---|---|---|")
    for env in ENVS:
        for key in SW:
            e = sw[(sw.env == env) & (sw.sweep == key) & (sw.metric == "energy")].iloc[0]
            j = sw[(sw.env == env) & (sw.sweep == key) & (sw.metric == "jpf")].iloc[0]
            f = lambda s: s.replace("sac", "SAC").replace("td3", "TD3").replace("mbpo", "MBPO")
            w(f"| {env} | {SWL[key]} | {f(e.ordering)} | {f(e.swapped_pairs)} | {f(j.ordering)} | {f(j.swapped_pairs)} |")
    w("")
    for m, nm in [("energy", "energy"), ("jpf", "J/FLOP")]:
        for env in ENVS:
            s = sw[(sw.env == env) & (sw.metric == m) & (sw.sweep != "baseline")]
            bad = s[~s.same_as_baseline]
            w(f"- {env}, {nm}: baseline ordering {s.iloc[0].baseline_ordering.replace('sac','SAC').replace('td3','TD3').replace('mbpo','MBPO')}; " + (f"{len(bad)} of {len(s)} sweep values differ from it ({', '.join(SWL[k] for k in bad.sweep)})." if len(bad) else f"no pair swaps in any of the {len(s)} sweep values."))
    w("")


def seeds():
    w("## 3. Seed-level check (paired seeds)\n")
    w("`n_own` = seeds (of 5) whose full SAC/TD3/MBPO ordering equals the ordering of the configuration means in that row; `n_base` = seeds whose ordering equals the baseline mean ordering. Source: `o3_seed_level.csv` (per-seed orderings in columns `seed_*`), `o3b_pairwise_seed_counts.csv`.\n")
    for env in ENVS:
        w(f"### 3.{ENVS.index(env) + 1} {env}\n")
        w("| sweep value | energy: mean ordering | n_own | n_base | J/FLOP: mean ordering | n_own | n_base |")
        w("|---|---|---|---|---|---|---|")
        f = lambda s: s.replace("sac", "SAC").replace("td3", "TD3").replace("mbpo", "MBPO")
        for key in SW:
            e = sl[(sl.env == env) & (sl.sweep == key) & (sl.metric == "energy")].iloc[0]
            j = sl[(sl.env == env) & (sl.sweep == key) & (sl.metric == "jpf")].iloc[0]
            w(f"| {SWL[key]} | {f(e.mean_ordering)} | {e.seeds_same_as_mean_ordering}/5 | {e.seeds_same_as_baseline_ordering}/5 | {f(j.mean_ordering)} | {j.seeds_same_as_mean_ordering}/5 | {j.seeds_same_as_baseline_ordering}/5 |")
        w("")
        part = pw[(pw.env == env) & (pw.seeds_lower_below_higher < 5)]
        if len(part):
            w("Pairs that are **not** in the mean-ordering direction in all 5 seeds:\n")
            w("| sweep value | metric | pair (mean-lower < mean-higher) | seeds in that direction | per-seed ratio higher/lower, min – max |")
            w("|---|---|---|---|---|")
            for _, r in part.iterrows():
                w(f"| {SWL[r.sweep]} | {'energy' if r.metric == 'energy' else 'J/FLOP'} | {NAME[r.lower_algo]} < {NAME[r.higher_algo]} | {r.seeds_lower_below_higher}/5 | {g4(r.min_seed_ratio)} – {g4(r.max_seed_ratio)} |")
            w("")
        else:
            w("Every pair is in the mean-ordering direction in all 5 seeds, for all sweep values and both metrics.\n")


def rollout():
    w("## 4. MBPO rollout-length sweep (Ant-v5 only; not an ordering sweep)\n")
    w("`L_max` = `rollout_max_length` (min length 1, schedule epochs 20–100 as in `configs/overrides/mbpo_ant*.json`); L_max = 25 is the Ant MBPO baseline. SAC / TD3 comparison values are their Ant baselines (SAC `base`, TD3 `base` = batch 100, warmup 10000). `×SAC`, `×TD3` = MBPO value / that value; `seeds >` = number of paired seeds (of 5) in which MBPO is above it. Source: `r1_mbpo_rollout_sweep.csv`.\n")
    from_o = o1[(o1.env == "Ant-v5") & (o1.sweep == "baseline")]
    sac = from_o[(from_o.algo == "sac")].set_index("metric")
    td3 = from_o[(from_o.algo == "td3")].set_index("metric")
    w(f"Reference baselines (Ant): SAC energy {g4(sac.loc['energy','mean'])} J, J/FLOP {pj(sac.loc['jpf','mean'])} pJ/FLOP; TD3 energy {g4(td3.loc['energy','mean'])} J, J/FLOP {pj(td3.loc['jpf','mean'])} pJ/FLOP.\n")
    w("| L_max | config | MBPO energy mean ± sd [J] | ×SAC | ×TD3 | seeds > SAC / TD3 | energy ordering (SAC, TD3 baselines + this MBPO) | MBPO J/FLOP mean ± sd [pJ/FLOP] | ×SAC | ×TD3 | seeds > SAC / TD3 | J/FLOP ordering | MBPO F mean [matmul FLOPs] (min – max) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda s: s.replace("sac", "SAC").replace("td3", "TD3")
    for _, r in ro.iterrows():
        w(f"| {r.rollout_max_length} | `{r.config}` | {g4(r.energy_mean)} ± {g4(r.energy_sd)} | {g4(r.energy_ratio_to_sac_base)} | {g4(r.energy_ratio_to_td3_base)} | {r.energy_seeds_mbpo_above_sac}/5 / {r.energy_seeds_mbpo_above_td3}/5 | {f(r.energy_ordering_with_sac_td3_base)} | {pj(r.jpf_mean)} ± {pj(r.jpf_sd)} | {g4(r.jpf_ratio_to_sac_base)} | {g4(r.jpf_ratio_to_td3_base)} | {r.jpf_seeds_mbpo_above_sac}/5 / {r.jpf_seeds_mbpo_above_td3}/5 | {f(r.jpf_ordering_with_sac_td3_base)} | {g4(r.F_mean)} ({g4(r.F_min)} – {g4(r.F_max)}) |")
    w("")


def tdmpc2():
    w("## 5. TD-MPC2 sweep (baseline, num_q ∈ {3,5,7}, horizon ∈ {1,3,5})\n")
    w("`base` = num_q 5, horizon 3 (shared by both sweeps). Rollout share = rollout energy / total energy (seed-mean ± sd of the per-seed share). Ant runs use `episodic: true`. Source: `t1_tdmpc2_sweep.csv`.\n")
    for env in ENVS:
        w(f"**{env}**\n")
        w("| config | num_q | horizon | total energy mean ± sd [J] | total J/FLOP mean ± sd [pJ/FLOP] | rollout share [%] |")
        w("|---|---|---|---|---|---|")
        t = tm[tm.env == env].copy()
        order = {"numq3": 0, "numq7": 1, "horizon1": 2, "base": 3, "horizon5": 4}
        t = t.sort_values(["num_q", "horizon"])
        for _, r in t.iterrows():
            w(f"| `{r.config}` | {r.num_q} | {r.horizon} | {g4(r.E_mean)} ± {g4(r.E_sd)} | {pj(r.jpf_mean)} ± {pj(r.jpf_sd)} | {g4(100 * r.rollout_share_of_total_mean)} ± {g4(100 * r.rollout_share_of_total_sd)} |")
        w("")
    w("### Range of the other three algorithms over all their configurations (context for TD-MPC2's sweep range)\n")
    w("Min / max of the configuration value (seed-mean energy; J/FLOP = ratio of means) over *all* configurations of the algorithm in that environment, including every sweep (SAC, TD3: 7 and 8 configurations; MBPO: 7 on HC, 9 on Ant; TD-MPC2: 5). `overlap` = the [min, max] interval intersects TD-MPC2's [min, max] over its 5 configurations. Source: `t3_algo_ranges_vs_tdmpc2.csv`, all configuration values in `t2_all_config_values.csv`.\n")
    for env in ENVS:
        w(f"**{env}**\n")
        w("| algorithm | #cfg | energy min [J] (config) | energy max [J] (config) | overlap TD-MPC2 (energy) | J/FLOP min [pJ/FLOP] (config) | J/FLOP max [pJ/FLOP] (config) | overlap TD-MPC2 (J/FLOP) |")
        w("|---|---|---|---|---|---|---|---|")
        for _, r in rg[rg.env == env].iterrows():
            oe = "—" if r.algo == "tdmpc2" else ("yes" if r.E_range_overlaps_tdmpc2 else "no")
            oj = "—" if r.algo == "tdmpc2" else ("yes" if r.jpf_range_overlaps_tdmpc2 else "no")
            w(f"| {NAME[r.algo]}{' (reference)' if r.algo == 'tdmpc2' else ''} | {r.n_configs} | {g4(r.E_min)} (`{r.E_argmin}`) | {g4(r.E_max)} (`{r.E_argmax}`) | {oe} | {pj(r.jpf_min)} (`{r.jpf_argmin}`) | {pj(r.jpf_max)} (`{r.jpf_argmax}`) | {oj} |")
        w("")


def checks():
    w("## 6. Consistency with Chapter 4 and items not computed\n")
    a = x1[x1.source.str.startswith("section_5.2")]
    b = x1[x1.source.str.startswith("section_4.4") & x1.rel_dev_E.notna()]
    w(f"**Consistency with Chapter 4 — none found.** All {len(a)} configuration totals (energy, J/FLOP) used here equal the 5.2 configuration table (itself checked against the Chapter 4.1–4.4 `*_cross_seed_summary.csv` files in 5.2) to a maximum relative deviation of {a.rel_dev_E.max():.2g} (energy) and {a.rel_dev_jpf.max():.2g} (J/FLOP). The {len(b)} TD-MPC2 rollout energies and rollout shares equal `contexts/Section_4.4_data/tdmpc2_cross_seed_summary.csv` (segment `rollout`: `energy_j_mean`, `share_of_total_pct_mean`) to {b.rel_dev_E.max():.2g} / {b.rel_dev_jpf.max():.2g}. `x1_consistency_checks.csv`.\n")
    w("**Notes / caveats (no numbers are inconsistent):**")
    w("- The batch-256 row uses SAC `base` and MBPO `base` (batch 256 is their baseline); only TD3 contributes a separate run set (`b256`). Rows `baseline` and `b256` therefore differ only in TD3 (batch 100 vs 256).")
    w("- The baseline orderings compare TD3 at batch 100 / warmup 10000 with SAC and MBPO at batch 256; the batch rows hold batch size equal across the three algorithms. Warmup is not equalised: TD3 uses 10000 in all its runs, MBPO 5000 (Section 4.3); warmup energy is excluded from the training total.")
    w("- MBPO's HC baseline has a fixed model-rollout length 1, Ant baseline a 1→25 schedule; the environment baselines are not the same MBPO configuration (Section 4.3 convention).")
    w("- Chapter 4 does not publish SAC/TD3/MBPO orderings, so there is no Chapter 4 ordering to compare with; only the underlying means, sds, J/FLOP and TD-MPC2 shares were cross-checked.")
    w("- Ratios between neighbours are ratios of configuration means (energy) or of ratio-of-means J/FLOP; the per-seed ratio range for each pair is in `o3b_pairwise_seed_counts.csv`.")
    w("- Not computed: no significance tests (n = 5 seeds; counts only); no returns; no Kendall's tau; TD-MPC2 is not placed in any ordering; the MBPO rollout sweep and TD-MPC2 sweep are not compared to SAC/TD3 in any ordering across other sweep values (only baselines, as specified).")
    w("- Environment: Windows dev checkout of the data, not the Linux experiment box; software versions as in `metadata.json` are unchanged by this analysis.")


head(); ordering_tables(); swaps(); seeds(); rollout(); tdmpc2(); checks()
(REPO / "contexts" / "section_5.3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("written", len(L), "lines")
