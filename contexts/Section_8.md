# Section 8 context: Conclusion and Outlook

Date: 2026-10-11 · Repo `HEAD`: `050b3c3d2618b7536f0cf5c5069e312fbebd4526` (committed 2026-10-08T16:48:12+02:00, branch `master`)

Existing scripts re-run: **none.**

Notes on provenance: `flop_analysis/output/*.csv` (2026-09-28), `flop_analysis/output/correlation/*` (2026-09-28) and `idle_power.csv` (2026-10-02) are newer than the newest run directory (`results/tdmpc2/HalfCheetah-v5/seed_958/20260919_094502`, run date 2026-09-19), so they were read as they are. Read-only calls used: `idle_power_analysis.load_runs()` was imported (no file written) to read the idle windows of the two seed-0 clock-study runs, which are not in `idle_power.csv` (280 canonical runs only); all other numbers are one-off pandas/scipy calculations on the existing CSVs/JSONs (scratch directory outside the repo).

Conventions in this file: all energies are gross (idle floor included), 1 kWh = 3.6e6 J, four significant digits, sd = sample sd (n−1) over the five seeds, J/FLOP = energy per matmul FLOP of `TOTAL_MEASURED_TRAINING`, canonical seeds {331, 958, 14577, 43611, 85062}. "HC" = HalfCheetah-v5, "Ant" = Ant-v5. Thesis line numbers refer to `thesis-main.tex` (compiled 2026-10-11; chapter/section numbers as in `thesis-main.toc`). "Recomputed" means recomputed in this extraction from `results/` or `flop_analysis/output/per_run_energy_per_flop.csv` (canonical rows only: `included_in_cross_seed_avg == True`, 280 run directories, 56 configurations × 5 seeds).

## Files read

- `CLAUDE.md`, `README.md` — `C:\Users\saman\Desktop\Thesis\Project Repository\rl-dissections\` (repo root).
- `thesis-layout.md` — **not at `contexts/thesis-layout.md`** (that path does not exist). Found and read at `C:\Users\saman\Desktop\Thesis\Thesis Report\thesis-layout.md` (last updated 2026-10-08 per its header).
- Thesis sources, `C:\Users\saman\Desktop\Thesis\Thesis Report\`:
  - `thesis-main.tex` (605,535 bytes, modified 2026-10-11 10:53) — the file with the written Chapters 3–7 (lines 2089–7003; lines 1–2088 and 7004–7096 are template sample text / Introduction placeholders / LaTeX hints / a commented-out clock appendix).
  - `current-status.txt` (modified 2026-10-11 10:57) — **content identical** to `thesis-main.tex` lines 2089–7003 (compared after removing blank and comment lines: 4,266 vs 4,266 lines, similarity ratio 1.0, same 180 `\label`s).
  - `main-english.tex` — exists (modified 2026-09-28) but is the unmodified template (no thesis chapter labels: 0 matches for `ch:method`/`ch:per-algorithm`); not used.
  - `thesis-main.toc` — read for chapter/section numbers.
- Data inputs: `flop_analysis/output/per_run_energy_per_flop.csv`, `flop_analysis/output/cross_seed_energy_per_flop.csv` (header/columns read; all numbers recomputed from the per-run file), `flop_analysis/output/correlation/correlation_summary.csv`, `…/config_means.csv`, `idle_power.csv`, `flop_analysis/output/check_totals.log`, `…/verify_mbpo_fit_flops.log`, `flop_analysis/flops_per_call.json`, all 300 `results/*/*/seed_*/*/metadata.json`, all 280 canonical `emissions_base_*.csv` and `run.log`, the 280 canonical `training_metrics.json`, `segment_energy.json` of the two clock-study runs.
- Code read: `configs/config.py`, `experiment_runner.py`, `run_experiment.py`, `utils/gpu_control.py`, `utils/thermal_gate.py`, `algorithms/sac.py`, `td3.py`, `tdmpc2.py`, `mbpo.py`, `tracker_utils.py` (symbols/lines only), `flop_analysis/correlation_analysis.py` (seed filter), `idle_power_analysis.py` (function `load_runs`), `requirements.txt`, `cli_commands.txt`, `.gitignore`; CodeCarbon 3.3.0 `codecarbon/external/ram.py` from the Windows dev venv `.venv/`.
- Existing earlier files in `contexts/` (names only, **not read**): `Section_3.1_3.2_3.3.md`, `section_4.1.md`, `Section_4.2.md`, `Section_4.3.md`, `Section_4.4.md`, `section_5.2.md`, `section_5.3.md`, `section_5.4.md`, `Section_6.md`, `section_7.md` (and their `*_data/` folders). No `section_5.1.md` exists. (One `grep` over `contexts/` incidentally printed two lines of `contexts/section_7.md`; they were not used as a source.)

## Task 1: Answers to RQ1–RQ4

RQ definitions: `tab:rqs` and `sec:rqs` (thesis 3.3, `thesis-main.tex` L2192–2266). Note: `tab:rqs` (L2221) and the RQ3 paragraph (L2252) still name Kendall's τ, although Section 5.3.1 (L5949–5953) says τ is not computed and orders/swaps are reported instead (see Consistency flags).

### RQ1 Decomposition

**(a) Statements in the thesis** (≤3):

1. [`sec:mbpo-interpretation` 4.3.6, L4681] "For RQ1, the model fit is therefore the segment that dominates at the baseline, and which of the two large segments dominates depends on the learner's configuration."
2. [`sec:tdmpc2-interpretation` 4.4.5, L5314] "For RQ1, TD-MPC2 is a two-block algorithm in which planning costs about as much as learning."
3. [`sec:sac-interpretation` 4.1.5, L3656] "SAC is therefore a workload dominated by its update phase on this machine."
- TD3: **NO ANSWER STATEMENT IN THESIS TEXT** — Section 4.2 (`sec:td3`) contains only the baseline subsection `sec:td3-baseline`; its announced subsections (`sec:td3-utd`, `sec:td3-width`, `sec:td3-batch`, `sec:td3-interpretation`, L3730–3737) do not exist in the file.
- Cross-algorithm statement (not counted above): [`sec:segment-shares` 5.1, L5513] "TD-MPC2 is the only algorithm of the four in which the rollout is a major part of the training energy." Section 5.1 has no summary paragraph and no explicit "answer to RQ1" sentence.

**(b) Numbers behind it** (baseline per algorithm × environment; TD3 at its own baseline, batch 100). Recomputed from `per_run_energy_per_flop.csv`; per-run share = segment energy ÷ `TOTAL_MEASURED_TRAINING`; measured partition = rollout, gradient_updates (= sum of the 4 allocated sub-segments), dynamics_model_update, synthetic_rollout_generation, world_model_pretrain (sum of parts = total, max relative deviation 3.6e-16):

| algo | env | total training energy [J], mean ± sd | dominant measured segment | share [%], mean ± sd | 2nd segment | share [%]  | vs thesis (tab:cmp-energy, tab:cmp-shares) |
|---|---|---|---|---|---|---|---|
| SAC | HC | 60910 ± 254.1 | gradient_updates | 95.90 ± 0.05304 | rollout | 4.097 ± 0.05304 | matches |
| SAC | Ant | 64380 ± 588.3 | gradient_updates | 93.27 ± 0.03840 | rollout | 6.729 ± 0.03840 | matches |
| TD3 | HC | 28110 ± 310.7 | gradient_updates | 94.98 ± 0.05886 | rollout | 5.019 ± 0.05886 | matches |
| TD3 | Ant | 30520 ± 431.9 | gradient_updates | 89.94 ± 0.1618 | rollout | 10.06 ± 0.1618 | matches |
| MBPO | HC | 207400 ± 7451 | dynamics_model_update | 70.46 ± 1.049 | gradient_updates | 28.33 ± 1.021 | matches |
| MBPO | Ant | 253100 ± 20140 | dynamics_model_update | 74.28 ± 2.093 | gradient_updates | 23.73 ± 1.935 | matches |
| TD-MPC2 | HC | 447100 ± 3184 | gradient_updates | 49.75 ± 0.2695 | rollout | 47.83 ± 0.3413 | matches |
| TD-MPC2 | Ant | 493200 ± 1782 | rollout | 51.62 ± 0.2272 | gradient_updates | 46.18 ± 0.2214 | matches |

- For TD-MPC2 on HC the measured block `gradient_updates` (49.75 %) is slightly above `rollout` (47.83 %); the thesis states the same (L5511, L5514, L4863 `\todo`), and says `rollout` is the largest *single* segment (compared with the allocated sub-segments, e.g. world-model step 32.74 % on HC; `tab:cmp-shares`). On Ant the rollout (51.62 %) is above the gradient_updates block (46.18 %).
- Total energies match `tab:cmp-energy` (60.91, 64.38, 28.11, 30.52, 207.4, 253.1, 447.1, 493.2 kJ). The sd's stated in L5443 (≤0.35 pp for SAC/TD3/TD-MPC2, ≤2.1 pp for MBPO) are reproduced (max 0.341 pp; 2.093 pp). Verdict: **matches thesis text** in all 8 rows.

**(c) Open points:** `\todo` L5447 (TD3 columns of `tab:cmp-shares`/`tab:cmp-energy` to be replaced by the batch-256 run once its context file exists; currently batch-100 values); `\todo` L5514 and L4863 (wording of TD-MPC2 "dominance" to be aligned between 3.1, Introduction and 5.1); `\todo` L5535 (ratios from rounded values); Section 4.2 TD3 subsections unwritten; interpretation of Chapter 5 deferred to `ch:evaluation` (L5427, L5428 `\todo`) but not present there; `\todo` L4053 (MBPO refit schedule stated as every 250 steps in 3.4/3.6 vs once per epoch in code, 100 fits per run; "decide whether this is the intended MBPO setting"); allocated sub-segments are time-share allocations (hedged throughout, `sec:measured-allocated`); interpretations are worded as hypotheses (L3652, L4673, L5306).

### RQ2 Energy vs. compute

**(a) Statements in the thesis** (≤3), all in `sec:ec-interpretation` 5.2.4:

1. [L5878] "FLOP counts are trustworthy as an ordering signal in a narrow sense: in this data set, configurations with more FLOPs almost never consume less energy, the ordering within TD-MPC2 is exact, and the update-to-data sweep behaves nearly proportionally."
2. [L5879] "They are not trustworthy as an estimate of the energy of an unmeasured configuration."
3. [L5881] "For RQ2, the answer is thus that compute explains the ranking of the configurations well and their energy only partly, and that the energy per FLOP is a property of the operating point and not of the algorithm."

**(b) Numbers behind it.**

Baseline J/FLOP of the total (ratio of mean energy to mean FLOPs; sd of the five per-seed ratios), recomputed:

| algo | env | matmul FLOPs (mean) | J/FLOP of total, ratio of means [pJ] | sd of 5 per-seed ratios [pJ] |
|---|---|---|---|---|
| SAC | HC | 8.771e+14 | 69.45 | 0.2897 |
| SAC | Ant | 9.390e+14 | 68.57 | 0.6265 |
| TD3 | HC | 2.459e+14 | 114.3 | 1.264 |
| TD3 | Ant | 2.625e+14 | 116.3 | 1.645 |
| MBPO | HC | 1.133e+15 | 183.1 | 5.442 |
| MBPO | Ant | 1.416e+15 | 178.8 | 8.471 |
| TD-MPC2 | HC | 8.833e+15 | 50.62 | 0.3605 |
| TD-MPC2 | Ant | 1.008e+16 | 48.95 | 0.1768 |

Match: **matches** `tab:rob-jpf` baseline row (SAC 69.45/68.57, TD3 114.3/116.3, MBPO 183.1/178.8 pJ) and `tab:rob-tdmpc2` (TD-MPC2 50.62/48.95 pJ). Layout/CLAUDE.md quote SAC 69.5/68.6, TD3 114.3/116.3, MBPO 183.1/178.8 (rounding differences only). The J/FLOP sd's of the thesis tables (not re-read) are stated by the thesis to be sd of per-seed ratios (`sec:rqs` L2234; `\todo` L2235 still asks to confirm).

Minimum and maximum J/FLOP of the total over all configurations per environment:

| env | n configs | min J/FLOP [pJ] | config of min | max J/FLOP [pJ] | config of max | max/min |
|---|---|---|---|---|---|---|
| HC | 27 | 19.11 | td3 / batch1024 | 1421 | td3 / width256 | 74.38 |
| Ant | 29 | 19.82 | td3 / batch1024 | 1232 | td3 / width256 | 62.18 |

Match: **matches** L5651–5652 (19.11 pJ → 1421 pJ, factor 74.4 on HC; 19.82 → 1232 pJ, factor 62.2 on Ant, "same two configurations at the ends").

Rank correlation and log–log fit of total energy on total FLOPs, pooled over algorithms and sweeps, one point per configuration (seed means):

| env | n | Spearman ρ | p (two-sided) | log₁₀–log₁₀ slope (± se) | R² | source |
|---|---|---|---|---|---|---|
| HC | 27 | 0.8357 | 5.782e-08 | 0.4715 (± 0.0752) | 0.6110 | `correlation_summary.csv` rows `energy_vs_flops`; recomputed independently from the per-run CSV: identical (ρ 0.835725, p 5.782e-08, slope 0.471534, se 0.075247, R² 0.611010) |
| Ant | 29 | 0.8463 | 7.333e-09 | 0.5071 (± 0.0755) | 0.6253 | same; recomputed: ρ 0.846268, p 7.333e-09, slope 0.507088, se 0.075540, R² 0.625324 |

- Canonical-seed confirmation: `flop_analysis/correlation_analysis.py` line 45 defines `CANONICAL_SEEDS = {331, 958, 14577, 43611, 85062}` and line 71 filters `df.seed.isin(CANONICAL_SEEDS)`; `config_means.csv` has 56 rows with `n_seeds`=5.
- Match: **matches** `tab:spearman` (L2160–2161), `tab:ec-pooled` (L5631, L5635) and L5617–5618. (The return-vs-FLOPs rows of `correlation_summary.csv`, ρ = +0.4174 (p = 0.03029, n = 27) and −0.3262 (p = 0.08418, n = 29), are also cited in 3.2 and 7.1.5.)
- Not computed in the thesis/repo output: per-algorithm correlations are in `tab:ec-algo` (5.2.2, n = 5–9) but are from `flop_analysis/section_5_2_analysis.py` (not re-run here).

**(c) Open points:** `\todo` L2186 (per-algorithm correlations "not computed" at that point; bootstrap intervals not shown; Appendix D); `\todo` L2249 (confirm Spearman as RQ2 statistic; per-segment version "not yet decided"); `\todo` L5855 (rollout fit and total fit without TD-MPC2 "neither was computed" — the statement on a gap between two groups is a hypothesis); `\todo` L5723 (baseline-calibrated ratios computed by hand); `\todo` L5769 (intercept shares by hand); `\todo` L5882 (link to related work 2.5.4 missing; Section 2.4 and 2.5 not written); `\todo` L5894 (analysis (b) peak-throughput fraction and (c) GPU-only J/FLOP not part of 5.2; open point to move to discussion in `ch:evaluation`); p-values and fits described as descriptive (configurations not independent, L5599–5601, L5885); fits with 3 points per sweep (L5373); hypotheses on roofline/kernel-launch cost are labelled as not separable by the data (L5861–5865).

### RQ3 Robustness

**(a) Statements in the thesis** (≤3):

1. [`sec:rob-summary` 5.3.6, L6176] "First, the ordering by energy is stable: TD3 $<$ SAC $<$ MBPO holds at all eight sweep values on both environments and in all 80 seed-level orderings, with no pair closer than a factor of 1.59."
2. [`sec:rob-summary`, L6178] "Second, the ordering by J/FLOP is not stable."
3. [`sec:env-summary` 5.4.7, L6421–6422] "The answer to RQ3 for the two environments has four parts. First, the total energy is higher on Ant-v5 in all 27 configurations and all seeds, by a factor of 1.027 to 1.248, and the rollout energy by 1.106 to 2.288."
- Additional locations (not quoted): [`sec:tdmpc2-interpretation`, L5367] "For RQ3, the decomposition of TD-MPC2 is robust to these two sweeps, but its level of energy is not: the horizon changes it by up to a factor of two."; [`sec:mbpo-interpretation`, L4687–4688] RQ3 sentence on the MBPO total as the learner's sensitivity "diluted by a fixed term". Section 5.3 and 5.4 only describe; "the interpretation follows in `ch:evaluation`" (L5913, L6427) — no such interpretation exists in Chapter 7.

**(b) Numbers behind it.**

(i) Environment ratio Ant/HC over the 27 configurations that exist on both environments (ratio of means; definition of L6211):

| quantity | min | configuration of min | max | configuration of max | thesis |
|---|---|---|---|---|---|
| total energy | 1.027 | SAC, UTD 4 | 1.248 | MBPO, width 256 | matches L6263–6265 (1.027–1.248; SAC UTD 4; MBPO width 256) |
| rollout energy | 1.106 | TD-MPC2, horizon 1 | 2.288 | TD3, width 256 | matches L6264 (1.106–2.288) |

The only configurations without a counterpart are the two Ant-only MBPO rollout-length runs (`rollout1`, `rollout15`; Ant has 29 configurations, HC 27). Total energy is higher on Ant in all 27 shared configurations: **recomputed** (smallest ratio 1.027 > 1).

(ii) Order of SAC, TD3, MBPO at each of the 8 shared sweep values (seed means; J/FLOP = ratio of mean energy to mean FLOPs). TD3 enters at its **own baseline batch size 100** in the baseline, UTD and width rows (SAC and MBPO: 256); only the three batch rows use TD3 at the matched batch size (256/512/1024); the "batch 256" row uses SAC/MBPO baseline runs with the TD3 batch-256 run. "Swapped" = pair whose relative order differs from the baseline ordering of the same environment and metric.

**Order by total energy**

| sweep value | TD3 batch size in row | HC ordering (low → high) | HC pairs swapped vs baseline | HC values [J] SAC / TD3 / MBPO | Ant ordering | Ant pairs swapped | Ant values |
|---|---|---|---|---|---|---|---|
| baseline | 100 | TD3 < SAC < MBPO | none | 60910 / 28110 / 207400 | TD3 < SAC < MBPO | none | 64380 / 30520 / 253100 |
| UTD 2 | 100 | TD3 < SAC < MBPO | none | 118000 / 55360 / 268900 | TD3 < SAC < MBPO | none | 123200 / 57540 / 316800 |
| UTD 4 | 100 | TD3 < SAC < MBPO | none | 235000 / 107300 / 373600 | TD3 < SAC < MBPO | none | 241300 / 111900 / 424500 |
| width 256 | 100 | TD3 < SAC < MBPO | none | 47410 / 23040 / 187900 | TD3 < SAC < MBPO | none | 49990 / 25100 / 234500 |
| width 512 | 100 | TD3 < SAC < MBPO | none | 49550 / 25370 / 192800 | TD3 < SAC < MBPO | none | 52200 / 27470 / 231600 |
| batch 256 | 256 | TD3 < SAC < MBPO | none | 60910 / 33450 / 207400 | TD3 < SAC < MBPO | none | 64380 / 35520 / 253100 |
| batch 512 | 512 | TD3 < SAC < MBPO | none | 73020 / 41680 / 209100 | TD3 < SAC < MBPO | none | 76500 / 44360 / 247400 |
| batch 1024 | 1024 | TD3 < SAC < MBPO | none | 82370 / 48070 / 218700 | TD3 < SAC < MBPO | none | 89060 / 53230 / 267700 |

**Order by J/FLOP of the total**

| sweep value | TD3 batch size in row | HC ordering (low → high) | HC pairs swapped vs baseline | HC values [pJ/FLOP] SAC / TD3 / MBPO | Ant ordering | Ant pairs swapped | Ant values |
|---|---|---|---|---|---|---|---|
| baseline | 100 | SAC < TD3 < MBPO | none | 69.45 / 114.3 / 183.1 | SAC < TD3 < MBPO | none | 68.57 / 116.3 / 178.8 |
| UTD 2 | 100 | SAC < TD3 < MBPO | none | 67.29 / 112.6 / 133.1 | SAC < TD3 < MBPO | none | 65.60 / 109.7 / 133.1 |
| UTD 4 | 100 | SAC < MBPO < TD3 | TD3-MBPO | 66.98 / 109.2 / 99.42 | SAC < MBPO < TD3 | TD3-MBPO | 64.26 / 106.6 / 100.2 |
| width 256 | 100 | MBPO < SAC < TD3 | SAC-MBPO, TD3-MBPO | 815.0 / 1421 / 612.6 | MBPO < SAC < TD3 | SAC-MBPO, TD3-MBPO | 678.8 / 1232 / 431.4 |
| width 512 | 100 | SAC < MBPO < TD3 | TD3-MBPO | 221.5 / 405.3 / 403.9 | SAC < MBPO < TD3 | TD3-MBPO | 205.0 / 387.5 / 323.0 |
| batch 256 | 256 | TD3 < SAC < MBPO | SAC-TD3 | 69.45 / 53.18 / 183.1 | TD3 < SAC < MBPO | SAC-TD3 | 68.57 / 52.89 / 178.8 |
| batch 512 | 512 | TD3 < SAC < MBPO | SAC-TD3 | 41.63 / 33.14 / 104.6 | TD3 < SAC < MBPO | SAC-TD3 | 40.74 / 33.03 / 105.8 |
| batch 1024 | 1024 | TD3 < SAC < MBPO | SAC-TD3 | 23.48 / 19.11 / 58.26 | TD3 < SAC < MBPO | SAC-TD3 | 23.72 / 19.82 / 63.14 |

- Order by energy: TD3 < SAC < MBPO at all 16 environment × sweep-value combinations; no pair swaps. Order by J/FLOP: four of the six possible orderings occur (S<T<M; S<M<T; M<S<T; T<S<M); 6 of 7 non-baseline sweep values differ from the baseline ordering on both environments; only UTD 2 keeps it.
- Match: **matches** `tab:rob-energy` (L5972–5979), `tab:rob-jpf` (L6018–6025) and L5984–5985, L6030–6033 on every value and every swap listed. The value 405.3 vs 403.9 pJ at width 512 on HC (TD3 vs MBPO, ratio 1.004) is as in L6045. Seed-level counts (80 of 80 for energy; 78 of 80 for J/FLOP) are from the thesis (`o3_seed_level.csv`, not part of the repo) and were **not** re-derived here.

(iii) TD-MPC2 along `num_q` ∈ {3, 5, 7} (horizon 3) and `horizon` ∈ {1, 3, 5} (num_q 5); baseline = num_q 5, horizon 3. Recomputed; rollout share = per-run rollout energy ÷ total, mean ± sd over 5 seeds:

| env | sweep | value | total energy [J] | ratio to baseline | rollout share [%] mean ± sd | Δ share vs baseline [pp] |
|---|---|---|---|---|---|---|
| HC | num_q | 3 | 358800 | 0.8025 | 50.72 ± 0.2962 | 2.899 |
| HC | num_q | 5 | 447100 | 1.000 | 47.83 ± 0.3413 | 0 (baseline) |
| HC | num_q | 7 | 526600 | 1.178 | 46.00 ± 0.2268 | -1.829 |
| HC | horizon | 1 | 280700 | 0.6278 | 48.16 ± 0.3071 | 0.3361 |
| HC | horizon | 3 | 447100 | 1.000 | 47.83 ± 0.3413 | 0 (baseline) |
| HC | horizon | 5 | 571300 | 1.278 | 50.46 ± 0.3819 | 2.637 |
| Ant | num_q | 3 | 411400 | 0.8341 | 54.22 ± 0.2805 | 2.597 |
| Ant | num_q | 5 | 493200 | 1.000 | 51.62 ± 0.2272 | 0 (baseline) |
| Ant | num_q | 7 | 576900 | 1.170 | 48.69 ± 0.1874 | -2.930 |
| Ant | horizon | 1 | 303200 | 0.6148 | 49.29 ± 0.3119 | -2.334 |
| Ant | horizon | 3 | 493200 | 1.000 | 51.62 ± 0.2272 | 0 (baseline) |
| Ant | horizon | 5 | 650300 | 1.318 | 54.48 ± 0.3243 | 2.857 |

- Match: **matches** `tab:rob-tdmpc2` (L6114–6124) and L6129–6133 (energy factor horizon 1→5 = 571 300 / 280 700 = 2.035 on HC, 2.145 on Ant; share monotone with `num_q` on both, monotone with horizon on Ant only; on HC 48.16 / 47.83 / 50.46 %). Reported differences from baseline in the TD-MPC2 interpretation (L5362: +2.899 ± 0.4733 pp for num_q 3 and +2.637 ± 0.6224 pp for horizon 5 on HC) match the Δ column.

**(c) Open points:** `\todo` L5914 (where the interpretation of Chapter 5 sits: 5.2 has its own interpretation; chapter introduction says Section 7.5); `\todo` L5953 (edit Section 3.3 — Kendall's τ — still pending; `tab:rqs` still names it); `\todo` L6184 (items (i)–(iii) to be carried into an interpretation in `ch:evaluation`: FLOP factors change more than energy along width/batch; SAC–TD3 swap depends on TD3 batch size not on the sweep; design limits); `\todo` L6226 (tag TD3 batch-100 rows); `\todo` L6309 (no `tab:td3-env`/`tab:tdmpc2-env`; per-configuration values only in prose of 4.2/4.4, and 4.2 has no such prose); `\todo` L6370 and L4667 (MBPO paired sd and permutation test; comparator = Ant run with L_max = 1; commits `0bbbbb7` vs `788ffcd`, sessions differ); `\todo` L6376 (a pairing that separates the TD-MPC2 classifier was not run; "state this as a limit in Section 7"); `\todo` L6417; `\todo` L6064 (figure for `tab:rob-jpf` missing); sign-flip test cannot go below p = 0.0625 with five pairs, so only direction is stated (L5947, L6216, L6426); MBPO and TD-MPC2 environment pairs are confounded with configuration (L6224).

### RQ4 GPU operating point (exploratory; seed 0, n = 1 per clock — the labelled exception to rule 3)

**(a) Statements in the thesis** (≤3):

1. [`sec:clock-results` 6.3.1, L6491] "Lowering the clock tenfold increases the energy to completion by a factor of 3.00 and the duration by 3.50, while the mean power falls only by a factor of 0.857."
2. [`sec:clock-discussion` 6.4, L6587] "The central observation is that lowering the clock reduced the power but increased the energy to complete the same work by a factor of three."
3. [`sec:clock-summary` 6.5, L6614] "Within the limits listed above (n=1, run order, unlogged lock), a lower clock in this setting does not save energy, which is the answer to RQ4 supported by the data."

**(b) Numbers behind it** (EXCEPTION: the only analysis that uses non-canonical runs; both are seed-0 dev runs, in no aggregate). Runs: 200/200 MHz = `results/mbpo/HalfCheetah-v5/seed_0/20260904_130915` (start 2026-09-04T11:09:15 UTC, end 12:48:34); matched 2000/2000 MHz = `results/mbpo/HalfCheetah-v5/seed_0/20260904_122958` (start 10:29:59 UTC, end 11:01:08). Source: `per_run_energy_per_flop.csv` rows `TOTAL_MEASURED_TRAINING` of the two `run_dir`s (`included_in_cross_seed_avg == False`); duration = sum of the durations of the four measured tasks (rollout, gradient_updates, dynamics_model_update, synthetic_rollout_generation); mean power = energy ÷ duration.

| quantity | 2000 MHz | 200 MHz | factor 200/2000 | thesis (`tab:clock-headline`) |
|---|---|---|---|---|
| energy to completion [J] | 189 500 (189 511.2) | 568 200 (568 210.0) | 2.998 | 189.5 kJ / 568.2 kJ / 3.00 — matches |
| duration [s] | 1 638 (1 638.11) | 5 732 (5 731.98) | 3.499 | 1 638 / 5 732 / 3.50 — matches |
| mean power [W] | 115.7 (115.69) | 99.13 | 0.8569 | 115.7 / 99.1 / 0.857 — matches |
| J/FLOP (3.153e14 FLOPs, same in both) | 6.010e-10 | 1.802e-9 | 2.998 | 6.01e-10 / 1.80e-9 / 3.00 — matches |

Above-idle energy, with the idle-window definition stated in the thesis (L6549: "mean of the two window-corrected windows" of the run's own head and tail idle windows, 90 s each; above-idle = E − floor × duration of the training total), recomputed with `idle_power_analysis.load_runs` (window-corrected powers: 2000 MHz head 63.40 W / tail 66.27 W; 200 MHz head 57.33 W / tail 63.09 W):

| quantity | 2000 MHz | 200 MHz | factor | thesis |
|---|---|---|---|---|
| floor (mean of head and tail, corrected) [W] | 64.84 | 60.21 | 0.9286 | 64.8 / 60.2 — matches |
| above-idle energy [kJ] | 83.30 | 223.1 | **2.678** | 83.3 / 223.1 / 2.68 — matches |
| floor share of gross energy | 56.04 % | 60.73 % | | 56 % / 61 % — matches |
| above-idle mean power [W] | 50.85 | 38.92 | | 50.9 / 38.9 — matches |
| bracket: head window only / tail window only as floor | factor 2.797 / 2.552 | | | 2.55–2.80 — matches |
| window-corrected tail − head [W] | +2.87 | +5.76 | | 2.9 / 5.8 — matches |

Field-by-field comparison of the two `metadata.json` files (all leaf fields of `algo_config`, `experiment_config`, top level):
- **Differ (6):** `experiment_config.gpu_min_clock_mhz` (2000 vs 200), `experiment_config.gpu_max_clock_mhz` (2000 vs 200), `start_time_utc` (2026-09-04T10:29:59.128368 vs 11:09:15.306557), `end_time_utc` (11:01:08.461191 vs 12:48:34.409574), `thermal_gate.final_power_w` (26.467 vs 21.006), `thermal_gate.final_temp_c` (45.0 vs 44.0) — plus `thermal_gate.waited_seconds` (1.17e-05 vs 1.07e-05, both ≈ 0).
- **Identical:** every other field, including `git_commit` (`ae3880860ebcce0c1e729714cdf48a08039adc46`), `torch_version` 2.13.0+cu130, `codecarbon_version` 3.3.0, `cuda_device_name`, seed 0, `algo_config` (hidden sizes [256, 256], batch size 256, `updates_per_env_step` 1, model hidden sizes [200,200,200,200], rollout length 1, rollout epochs 20–150, `model_train_freq` 250), `warmup_steps` 5000, `train_steps` 100000, `steps_per_epoch` 1000, `idle_baseline_seconds` 90.
- Same work (recomputed): 168 525 optimiser steps, 990 000 synthetic transitions, 3.153e14 total FLOPs in both runs; 403 rows in each `emissions_base_*.csv`; last-10 % mean training return 5936.56 in both (training_metrics.json). Not re-counted: "1 196 dynamics-model fit epochs" (L6474).
- Orientation: canonical MBPO HC width-256 runs (the matched architecture: hidden 256, batch 256, UTD 1): total energy mean 187 900 J, sd 4 850 J (n = 5) → the 2000 MHz run is +0.329 sd above the mean; the 200 MHz run is 3.024 × that mean (thesis L6511–6512: 187.9 kJ, sd 4.8 kJ, 0.33 sd, 3.02 ×) — matches (thesis writes the sd as 4.8 kJ; computed 4.850 kJ).
- Idle floor: the thesis states (`\todo` L6556) that it still has to be said which window the other chapters use as floor; Chapter 7 (`sec:thr-idle`) uses "mean of head and tail power" for its net-energy sensitivity (L6776).

**(c) Open points:** `\todo` L6439 (align RQ4 wording with the research-question section; the label it cites, `sec:research-questions`, does not exist); `\todo` L6482 (decide whether to re-run both configurations from a committed state; recorded commit `ae38808…` predates the MBPO implementation per L6480); `\todo` L6544 (allocation validation, `sec:allocation` does not exist; sub-segment timers cover 61 % of gradient-update duration at 200 MHz (524 s of 853 s) vs 98 % at 2000 MHz — recomputed from `segment_energy.json`: 524.0 s of 853.4 s = 61.4 %; 2000 MHz 396.1 s of 405.7 s = 97.6 %); `\todo` L6556; `\todo` L6564 (CPU power increase without a utilisation measurement); `\todo` L6590 (DVFS references to be verified); `\todo` L6601 (lock verification with `nvidia-smi` — actual clock not logged); `\todo` L6603 (supported clock range, power limit, driver version not recorded); `\todo` L6615 (rephrase last sentence once RQ4 wording is fixed); n = 1, run order (R200 after R2000, 8 min), single hidden size 256, hedged as "plausible reading, not a result" (L6592–6593); result stated to apply to "this MBPO configuration, which is a low-FLOP workload" (L6594). Commented-out appendix (`app:clock-*`, L7004–7080) is not part of the compiled thesis.

## Task 2: Contributions

The five contributions are those of `thesis-layout.md` Chapter 8 (L271). The written thesis text contains **no** sentence that lists or labels contributions (searched Chapters 3–7 for "contribut", "novel", "we propose", "open-source", "public", "release": only unrelated hits); Chapter 1 and Section 1.5 are not written.

| # | Contribution (layout wording) | Delivered in (labels) | Evidence count |
|---|---|---|---|
| 1 | per-segment decomposition across 4 algorithm families | `sec:measured-allocated`, `sec:alg-segments`, `tab:segment-meaning` (method); `sec:sac`, `sec:td3`, `sec:mbpo`, `sec:tdmpc2` (`ch:per-algorithm`); `sec:segment-shares` (`ch:comparison`) | 4 algorithms × 2 environments; 56 configurations (27 HC, 29 Ant) × 5 seeds = 280 runs; measured segment types per algorithm: SAC 2 and TD3 2 (rollout, gradient_updates), MBPO 4 (+ dynamics_model_update, synthetic_rollout_generation), TD-MPC2 3 (+ world_model_pretrain); 4 allocated sub-segments in every algorithm; task rows (280 runs, from `emissions_base_*.csv`): 72,890 total = 72,330 non-idle (rollout 28,000; gradient_updates 28,000; dynamics_model_update 8,000; synthetic_rollout_generation 8,000; world_model_pretrain 50; warmup 280) + 560 idle (head/tail) |
| 2 | compute-normalised (J/FLOP) comparison | `sec:flop-accounting` and subsections 3.5.1–3.5.7; J/FLOP tables in 4.1–4.4 (`tab:*-base-jpf`); `sec:energy-vs-compute` (5.2); `sec:rob-jpf` (5.3.3) | J/FLOP of the total for 56 configurations; ranges ×74.38 (HC) and ×62.18 (Ant); ρ 0.8357 / 0.8463 (n 27 / 29), log–log slope 0.4715 / 0.5071, R² 0.6110 / 0.6253; `flops_per_call.json` entries for 4 algorithms × 2–3 environments (SAC also Humanoid-v5 = dev); `check_totals.log`: "ALL CHECKS PASSED" over 300 run directories (includes the 20 dev runs); `verify_mbpo_fit_flops.log`: "ALL PASSED" (relative difference 0 for N = 5000, 12345, 30000) |
| 3 | robustness analysis across sweeps | `sec:sweeps` (3.6.4); `sec:sac-utd/-width/-batch`, `sec:mbpo-utd/-width/-batch/-rollout`, `sec:tdmpc2-numq`, `sec:tdmpc2-horizon`; `sec:robustness` (5.3); `sec:env-effect` (5.4); TD3 sweeps appear only in 5.2–5.4 tables (4.2 has only the baseline subsection) | 56 configurations = baselines 8 + UTD 12 + width 12 + batch 14 + MBPO rollout length 2 + TD-MPC2 `num_q`/`horizon` 8 (recomputed from `per_run_energy_per_flop.csv`); 8 shared sweep values × 2 environments × 2 metrics = 32 orderings; 80 seed-level orderings (thesis, L5986; not re-derived); 27 paired environment configurations |
| 4 | the clock-frequency observation | `ch:clock` (`sec:clock-scope` … `sec:clock-summary`, 6.1–6.5) | 2 runs (n = 1 per clock), seed 0, 0 canonical runs; 403 task rows each; identical work (168,525 optimiser steps; 990,000 synthetic transitions; 3.153e14 FLOPs); energy factor 2.998, duration 3.499, power 0.857, above-idle energy 2.678 |
| 5 | open measurement harness and FLOP pipeline | `sec:harness` (3.4), `sec:flop-accounting` (3.5), `sec:flop-verification` (3.5.7), `sec:setup`/`sec:launch` (3.7), `app:reproducibility` (7.5) | repository facts in Task 3: 4 algorithm modules, 10 sweep scripts, all 10 named entry points exist, 12 distinct git commits across the 280 runs, 300 run directories tracked in git (77 MB); public availability of the repository: **NOT VERIFIED** (stated by the user; remote is `https://github.com/SamBro2901/rl-dissections.git`; no `LICENSE` file); the analytic dense-layer cross-check is not a repository script (`\todo` L2827, L3156, L3747, L4009, L4802) |

**Not in layout:** nothing found presented as a contribution in the thesis text.

## Task 3: Repository and release facts

- `git remote -v`: `origin  https://github.com/SamBro2901/rl-dissections.git (fetch)` and `(push)`.
- Current branch: `master`. `HEAD`: `050b3c3d2618b7536f0cf5c5069e312fbebd4526`, commit date 2026-10-08T16:48:12+02:00, subject "Section 5 context and data". (`CLAUDE.md` still names `f09a861` as "current HEAD".)
- Tags: none (`git tag` prints nothing).
- Working tree: **not clean** — no tracked file is modified; untracked: `contexts/Section_6.md`, `contexts/Section_6_data/`, `contexts/section_7.md`, `contexts/section_7_data/` (and `contexts/Section_8.md` once this file is written).
- License: **NO LICENSE FILE** in the repository root (`LICENSE*` does not match any file). A `LICENSE` file exists in `C:\Users\saman\Desktop\Thesis\Thesis Report\` (the thesis-template directory), which is not this repository.
- Public visibility on GitHub: **NOT VERIFIED** (the user states it will be public; no network call made).
- `results/`: **tracked by git** (1,815 tracked files under `results/`, including all 300 `metadata.json`); **not** matched by `.gitignore` (patterns: `rl-exp/`, `__pycache__/`, `*.pyc`, `*.pyo`, `.env`, `.env.*`, `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`, `*.pt`, `*.pth`, `*.ckpt`, `configs/__pycache__/`, `utils/__pycache__/`; `git check-ignore results` prints nothing). Size on disk: 77 MB (`du -sh results`). Run directories (`results/{algo}/{env}/seed_{n}/{timestamp}`): 300 = 280 canonical + 20 dev (seeds 0/42/56; SAC Humanoid-v5 seed 56).
- Git commit recorded in `metadata.json` of the **280 canonical runs**: 12 distinct values (all 280 accounted for); commit dates/subjects looked up in the repo history:

| commit (recorded) | runs | commit date | subject |
|---|---|---|---|
| `7e672ef12c53bf4c201241c549527a7ef8bd4bff` | 60 | 2026-09-12 | Setting up width sweep for SAC, TD3 and MBPO |
| `788ffcd841101338cccd395b9afe6773eb381bfb` | 45 | 2026-09-17 | Scheduling batch sweep for ant and new rollout sweep for MBPO |
| `a3582b52d5d3a0c6617038fd647c67eee6a92a78` | 40 | 2026-09-18 | TDMPC2 sweep scripts |
| `9b4b44cd2f53f78518b07ae6be3ff44eebfdfa10` | 35 | 2026-09-15 | Preparing batch size sweep for Half Cheetah |
| `aa9186b0e59e60ecd5287c1e1b52b572591a7b7b` | 30 | 2026-09-09 | Setup UTD sweep run script for SAC and TD3 across both envs and 5 seeds |
| `ecb71350088e84d75a649cee8289d489c59a7503` | 20 | 2026-09-14 | Scheduling UTD sweep for MBPO |
| `0bbbbb74a6542fff643e5cd3ba7603af8de1fcc0` | 10 | 2026-09-08 | baseline config for mbpo |
| `ecb98bbba4e2c648a08d6e7aada30b9ba0d168c2` | 10 | 2026-09-08 | Finished remaining baseline runs for Ant/v5 |
| `58962d221751e45ed9d599f0d731eccb9cccba4c` | 10 | 2026-09-09 | Added and tested td3 algorithm |
| `b5b7e54502412299ce94a5b46e9aa18c9a877308` | 10 | 2026-09-10 | Setup new  algorithm tdmpc2 and tested recording |
| `f23f448480ecdf1dd6dc552b45f9194b2602bc11` | 8 | 2026-09-07 | Setting baseline config for all experiments |
| `92dcee0763ed54c13fa816cc364055fa68c958af` | 2 | 2026-09-07 | Recorded runs for seeds - 331, 958, 14577, 43611, 85062 |

  (matches the "12 distinct git commits" of `sec:thr-order`/`sec:eval-repro`. The recorded commit is the repo `HEAD` at launch.) Clock-lock fields across all 300 runs: 284 runs 2000/2000 MHz; 16 runs not 2000/2000 (listed in 5.6); none of them canonical.
- Entry points named in `README.md` — existence check: `run_experiment.py` OK; `aggregate_results.py` OK; `dashboard.py` OK; `flop_dashboard.py` OK; `flop_analysis/measure_flops.py` OK; `flop_analysis/compute_energy_per_flop.py` OK; `flop_analysis/check_totals.py` OK; `flop_analysis/verify_mbpo_fit_flops.py` OK; `flop_analysis/correlation_analysis.py` OK; `idle_power_analysis.py` OK; `scripts/*.sh` = 10 files: `run_ant_overnight_sweep.sh`, `run_batch_size_sweep.sh`, `run_batch_size_sweep_ant.sh`, `run_mbpo_env_sweep.sh`, `run_mbpo_rollout_length_sweep.sh`, `run_td3_env_sweep.sh`, `run_tdmpc2_numq_horizon_sweep.sh`, `run_tdmpc2_seed_sweep.sh`, `run_utd_sweep.sh`, `run_width_sweep.sh`. **Missing: none.** Other file names that appear in the README but do not exist as files: `flop_calculation_methodology.md` (README/CLAUDE.md state it was never checked in), `tdmpc2_num_q3.json` (the documented wrong name of the false start; actual file `configs/overrides/tdmpc2_numq3.json`), and `results/..._status.json` (a glob pattern). `flop_analysis/` also contains `section_5_2_analysis.py`, `section_5_2_render.py`, `section_5_3_analysis.py`, `section_5_3_render.py`, `make_fix_report.py`, `flop_keys.py`, which the thesis cites as scripts of Sections 5.2/5.3 (`\todo` L5611, L6184) but are not named in `README.md`; the Section 5.4 script cited via `env_comparison_paired.csv` (`\todo` L6309) is not in the repo (`flop_analysis/section_5_4_*` does not exist).
- Algorithm modules: `algorithms/sac.py`, `algorithms/td3.py`, `algorithms/mbpo.py` (reuses `SACAgent` via `from algorithms.sac import SACAgent`, `mbpo.py:41`), `algorithms/tdmpc2.py`; supporting: `algorithms/dynamics_model.py`, `algorithms/termination_fns.py`, `algorithms/replay_buffer.py`, `algorithms/tracker_utils.py`. `ALGO_CONFIGS` (`configs/config.py` L296–301): `"sac": SACConfig`, `"mbpo": MBPOConfig`, `"td3": TD3Config`, `"tdmpc2": TDMPC2Config`; the CLI choice list `--algo` is `sorted(ALGO_CONFIGS)` (`run_experiment.py` L32).
- `requirements.txt` (verbatim):

```
torch
gymnasium
codecarbon>=3.0
nvidia-ml-py  # replaces the deprecated pynvml; codecarbon and our thermal_gate.py both use the pynvml import name, which this package still provides
pandas
plotly
dash

# For actual MuJoCo continuous-control tasks (HalfCheetah, Hopper, Walker2d, Ant):
gymnasium[mujoco]
```
  (No version pins; `scipy` is not listed although `flop_analysis/correlation_analysis.py` and the 5.x analyses use it — not checked whether the scripts import it at top level.)

## Task 4: Limitations

### Limitations table

Notes: (i) "RQ(s) affected" gives the research question whose reported quantity the row concerns (by the quantity named in the cited text); where the thesis states an effect itself, it is quoted. (ii) "Recomputed" = recomputed in this extraction; "thesis only" = number taken from the thesis, not re-derived. (iii) The last column is used only for group C.

**Group A — Chapter 7 (`sec:threats` subsections, `tab:thr-summary`), in order**

| # | Limitation | Where stated | Number/evidence given there | RQ(s) | Source / check | Layout flag |
|---|---|---|---|---|---|---|
| A1 | Coarse GPU energy counter on very short tasks; the GPU term of MBPO `synthetic_rollout_generation` is largely missing | `sec:thr-resolution` 7.1.1 (`tab:thr-gpu-zero` L6676–6696), `tab:thr-summary` | zero GPU energy for 4,140 of 4,441 tasks ≤ 10 ms; none above 0.1 s; 720 of 1,000 baseline synthetic tasks (466/500 HC, 254/500 Ant); segment = 0.04 % (HC) / 0.29 % (Ant) of MBPO baseline energy; thesis effect: "MBPO `synthetic_rollout_generation` unreliable (≤ 0.29 % of total); other segments unaffected" | RQ1, RQ2 (that segment) | **Recomputed: match** — all 9 duration bins (4,441/4,140; 881/745; 836/521; 1,673/448; 7,299/0; 16,004/0; 107/0; 19,333/0; 21,756/0), 466/500 and 254/500 (median durations 6.1 ms and 50.3 ms), baseline shares 0.03654 % and 0.2927 %; all 5,854 zero-GPU tasks are `synthetic_rollout_generation` tasks | |
| A2 | Rollout tasks of SAC/TD3/MBPO shorter than the 1 s polling interval; individual task energies coarse; a bias test "was not performed" | `sec:thr-resolution` | rollout 0.10–0.53 s (smallest TD3 HC, median 0.110 s); median task power 117–122 W; TD-MPC2 rollouts 6.6–16.7 s (median 11.1 s) | RQ1 (rollout shares), RQ2 (rollout J/FLOP) | **Recomputed: match** — see "Counts of tasks shorter than 1 s" below; TD3 HC baseline rollout median 0.1095 s (all TD3 HC configurations 0.1079 s); median rollout-task power: SAC 121.4 W, TD3 121.6 W, MBPO 116.7 W, TD-MPC2 211.3 W | |
| A3 | RAM energy is not measured (constant power × duration) | `sec:thr-resolution`, `tab:thr-summary` | 20.0 W in all 72,890 task rows (CodeCarbon 3.3.0, `machine` mode, 60.8 GB); ≈ 9–17 % of median task power | RQ1–RQ3 (all absolute energies); thesis effect: "RAM energy proportional to duration; no independent information" | **Recomputed: match** — 72,890 rows (280 runs, incl. 560 idle rows), `ram_power` = 20.0 W in every row, 60.82 GB; 20 W ÷ median rollout power = 16.5 % (SAC), 16.4 % (TD3), 17.1 % (MBPO), 9.5 % (TD-MPC2) | |
| A4 | RAPL mode not confirmed on the research PC | `sec:thr-resolution` (`\todo` L6654) | "the run logs contain no RAPL fallback warnings" | RQ1–RQ3 (CPU energy) | **Recomputed:** 0 of 280 canonical `run.log` files contain "default power consumption of 4 W per thread", a geolocation/fallback line, a timeout, a permission/sudo line, a `[WARNING]`, `[ERROR]` or `Traceback`. RAPL mode itself: NOT VERIFIED (needs the experiment machine) | |
| A5 | Non-stationary MBPO per-epoch energy | `sec:thr-stationarity` 7.1.2 (`tab:thr-stationarity`), `tab:thr-summary` | epoch ratio 0–9 / 10–99: MBPO 0.520 (HC), 0.437 (Ant); R² of cumulative energy 0.984 / 0.993; SAC/TD3/TD-MPC2 ratios 0.978–0.997; dynamics_model_update ratio 0.34 / 0.26; thesis effect: "No linear extrapolation beyond 100 epochs for MBPO; comparisons at equal steps valid" | RQ1, RQ2 (MBPO) | thesis only — NOT VERIFIED here | |
| A6 | Idle-power drift (tail above head) and the gross convention | `sec:thr-idle` 7.1.3, `tab:thr-gross-net`, `tab:thr-summary` | tail > head in 256 of 280 runs; median abs(Δ) 4.5 W (7.0 % of head), IQR 4.5–10.9 %, max 23.2 % (as recorded); window-corrected median 6.2 W (10.0 %); mean idle 65.3 W head, 69.5 W tail; net removes 35–53 % of total; orderings unchanged gross vs. net | RQ1–RQ3 (absolute ratios between algorithms; small segments); thesis effect: "Absolute ratios shift; orderings unchanged gross vs. net" | **Recomputed from `idle_power.csv` (280 rows): match.** As recorded: tail > head 256/280; median abs(Δ) 4.484 W; median 7.036 %; IQR 4.454–10.92 %; max 23.19 %; mean head 65.27 W, tail 69.51 W. Window-corrected: tail > head 261/280; median abs(Δ) 6.229 W; median 10.02 %; IQR 7.614–14.36 %; max 27.39 %; mean head 63.11 W, tail 69.51 W. `tab:thr-gross-net` (net/gross 0.47–0.65) thesis only — NOT VERIFIED | |
| A7 | Fixed (non-randomised) run order; configuration confounded with time | `sec:thr-order` 7.1.4, `tab:thr-summary` | 280 runs 2026-09-07 → 2026-09-19, 117.7 h, no overlap; seeds of each configuration consecutive in the order 331, 958, 14577, 43611, 85062 (56 of 56); 44 of 48 sweep–baseline pairs on different days (5.6–239.6 h); 11 of 27 HC/Ant pairs on different days; 12 git commits; time effect "not quantified" | RQ3 (sweep and environment comparisons) | partly recomputed: first start 2026-09-07T10:13:36 UTC, last end 2026-09-19T16:10:36 UTC; summed (end − start) = 117.72 h; minimum gap between runs 1.27 s, 0 overlaps; 12 commits: match. 56/56, 44/48, 11/27, 5.6–239.6 h thesis only — NOT VERIFIED | |
| A8 | Learning behaviour: early termination on Ant; 14 of 29 Ant configurations do not improve in episode return | `sec:thr-learning` 7.1.5, `tab:app-learning`, `tab:thr-summary` | HC: all 27 configurations improve in all 5 seeds; Ant: 14 of 29 do not (9 MBPO, 4 SAC, 1 TD3); thesis effect: "None (no return-based conclusions)" | none stated by the thesis (only the Spearman return-vs-FLOPs control of 3.2 is return-based) | **Recomputed: match** (see learning table: 14 non-improving Ant configurations = MBPO 9, SAC 4, TD3 1; HC 27/27) | |
| A9 | Incomplete provenance of the research PC | `sec:eval-repro` 7.2, `tab:thr-provenance`, `tab:thr-summary` | "Seven items not recorded"; named: distribution/release, driver, kernel modules, power limits, supported clocks, SM count, peak FP32, RAM type/speed, mainboard, storage, Gymnasium/MuJoCo/`nvidia-ml-py` versions | reproducibility; RQ2 (peak FP32 / throughput fraction) | recorded values **recomputed: match** — in all 280 runs: `Linux-7.0.0-31-generic-x86_64-with-glibc2.43`, Python 3.14.4, CodeCarbon 3.3.0, Intel(R) Core(TM) Ultra 9 285K (24), 1 × NVIDIA GeForce RTX 5090, RAM 60.82 GB, `machine` mode, torch 2.13.0+cu130. `tab:thr-provenance` contains 6 rows with `\todo`; "seven" NOT VERIFIED | |
| A10 | Float32 matmul precision (TF32) not verified in the experiment environment | `sec:eval-repro` (L6881–6884, `\todo`) | "No module of the code base sets `allow_tf32` or the float32 matmul precision"; confirmed "only on a different build" | RQ2 (throughput interpretation) — not stated | **Recomputed:** no occurrence of `tf32`/`TF32`/`allow_tf32`/`float32_matmul`/`set_float32` in the repo's `.py` files; the runtime default on the experiment machine NOT VERIFIED | |

**Group B — Chapters 3 (limits of the metric), 4, 5, 6**

| # | Limitation | Where stated | Number/evidence given there | RQ(s) | Source / check | Layout flag |
|---|---|---|---|---|---|---|
| B1 | Sub-segments are allocated by host-clock time share (no device synchronisation); assumption of equal power across sub-phases | `sec:measured-allocated` 3.4.3 ("Assumption", "Timing caveat"); repeated in all 4.x / 5.1 limits | timer coverage of the measured gradient-update duration: 97.64–98.70 % (TD-MPC2, L5387); 61 % at 200 MHz vs 98 % at 2000 MHz (clock study) | RQ1 (allocated rows), RQ2 (allocated J/FLOP) | TD-MPC2 and clock-study coverage thesis/recomputed (clock study: 61.4 % and 97.6 %, recomputed); layout's "86–98 %" NOT VERIFIED here; code: no `synchronize` call in `algorithms/`, `utils/`, `experiment_runner.py` (recomputed by grep) | |
| B2 | Rollout energy and rollout FLOPs measure different things (CPU/simulator energy, no FLOPs) for SAC/TD3/MBPO | `sec:flop-caveats` 3.5.6, 4.x limits | qualitative; J/FLOP of TD-MPC2 rollout 44.37–46.30 pJ (L5324) vs rollouts of the other three "dominated by energy that has no FLOPs counted against it" | RQ2 | text only | |
| B3 | Only matmul FLOPs are counted (activations, optimiser, sampling, simulator, buffer excluded); `target_update` reported per Polyak op | `sec:flop-caveats`, `sec:ec-interpretation` limit 2 (L5886) | none | RQ2 | text only | |
| B4 | Gross energy: idle power is part of every J/FLOP and intercept | `sec:flop-caveats`, `sec:ec-interpretation` limit 3, `sec:sac-interpretation` | idle floor = 35.06–62.59 % of SAC gross total (L3682) | RQ2 | thesis only | |
| B5 | Same segment names stand for different computations in TD-MPC2 | `sec:flop-caveats`, `tab:segment-meaning`, 5.1 limit 2 (L5539) | `world_model_pretrain` exists only in TD-MPC2 and is not split | RQ1 | text only | |
| B6 | SAC: rollout power follows the preceding update phase | `sec:sac-interpretation` Limits (L3709) | 121.9 / 129.6 / 134.2 W at batch 256 / 512 / 1024 (HC) | RQ1, RQ2 (rollout) | thesis only | |
| B7 | SAC: Ant rollout depends on the learned policy (episode resets inside the task) | L3711 | 178–216 episodes (width 256), 379–511 (batch 1024) | RQ3 (environment) | thesis only | |
| B8 | SAC: buffer-sample allocation is the weakest case (priced at mean GU power) | L3693–3696 | +90.95 % energy vs +45 % time at batch 1024 (HC) | RQ1 | thesis only | |
| B9 | SAC: idle drift and fixed order | L3713 | head–tail idle power differs by up to 23.19 % | RQ3 | **Recomputed: match** (max 23.19 % as recorded, SAC) | |
| B10 | TD3: no interpretation / limits subsection written; rollout tasks shorter than the polling interval | `sec:td3` (only `sec:td3-baseline` exists); `\todo` L3981 | TD3 rollout median 0.2175 s vs SAC 0.3409 s | RQ1, RQ2 | **Recomputed: match** (0.2175 s and 0.3409 s, all configurations) | |
| B11 | MBPO: 8,000 rollout and 8,000 synthetic-rollout tasks shorter than the polling interval; synthetic integration span departs from task duration by up to 28 % | `sec:mbpo-interpretation` Limits 1 (L4755–4756), `\todo` L4283 | 8,000 and 8,000 tasks | RQ1, RQ2 | **Recomputed:** 8,000 / 8,000 tasks < 1 s (match); the "28 %" integration-span figure thesis only | |
| B12 | MBPO: rollout power follows the model fit, not the learner | L4758–4761 | 116.5–119.2 W (HC) | RQ1/RQ2 (rollout) | thesis only | |
| B13 | MBPO: buffer-sample allocation weakest | L4762 | +75.29 % (HC) / +164.7 % (Ant) energy vs +36.40 % / +102.6 % time at batch 1024 | RQ1 | thesis only | |
| B14 | MBPO: idle drift, fixed order; unavailable quantities | L4763–4764 | up to 8.986 % as recorded (12.72 % corrected); unavailable: terminated synthetic rollouts, simulator share, peak throughput, precision, application of the clock lock | RQ3 | **Recomputed: match** (MBPO max 8.986 % / 12.72 %) | |
| B15 | MBPO: early stopping makes the seed spread of the model fit large; 101 of 720 (run, segment) energies flagged as outliers, not excluded | L4765–4766, L4708–4710 | CV of total 3.593 % (HC) / 7.956 % (Ant) vs SAC 0.4172 % / 0.9137 % | RQ1, RQ3 | thesis only | |
| B16 | MBPO: refit schedule implemented once per epoch (100 fits/run) while 3.4/3.6 state every 250 steps | `\todo` L4053, L4714–4715 | 100 fits per run instead of 400 | RQ1, RQ3 | code: `model_train_freq: 250` in both clock-study metadata files; the once-per-epoch evaluation is stated by the thesis, not re-derived here | |
| B17 | TD-MPC2: allocation prices Polyak average and buffer sample at mean GU power | `sec:tdmpc2-interpretation` Limits 1 (L5387–5391) | Polyak time 56.53–57.05 s unchanged, allocated energy −18.33 % at horizon 1 (power −18.96 %) | RQ1 | thesis only | |
| B18 | TD-MPC2: matmul-only FLOPs; simulator share, peak throughput, precision, classifier energy not available | Limits 2 (L5392–5394) | none | RQ2 | text only | |
| B19 | TD-MPC2: first/last second attribution of tasks unquantified | Limits 3 (L5395–5396) | all measured tasks ≥ 6.6 s | RQ1, RQ2 | **Recomputed:** TD-MPC2 measured-task minima: rollout 6.616 s, gradient_updates 8.845 s, world_model_pretrain 44.63 s (warmup tasks, 0.15–0.69 s, are not measured segments) — match | |
| B20 | TD-MPC2: idle drift and fixed order (baselines a week before the sweeps); 55 of 400 outliers | Limits 4–5 (L5397–5400) | median 10.04 % as recorded (13.83 % corrected), max 21.05 %; head idle 63.41–65.21 W | RQ3 | **Recomputed: match** (TD-MPC2 median 10.04 % / 13.83 %, corrected max 21.05 %) | |
| B21 | TD-MPC2: only `num_q` and `horizon` varied; schedule is blocks, not interleaved (reference interleaves); not in UTD/width/batch sweeps | Limits 6 (L5401–5404), `sec:rob-tdmpc2` | energy of an interleaved schedule "was not measured" | RQ3 | text only | |
| B22 | TD-MPC2: termination classifier (Ant, `episodic: true`) confounds the environment comparison; no run without it on Ant | L5380–5383, L4849, `sec:env-confounds` (L6372–6376) | classifier = 9.841–20.25 % of the planning call; without it Ant planning call = 1.002 × HC | RQ3 (environment) | **Recomputed:** `episodic` True in all 25 Ant TD-MPC2 runs, False in all 25 HC runs (from architecture signatures); percentages thesis only | |
| B23 | Chapter 5: TD-MPC2 only in its own sweeps; TD3 baseline batch 100 not matched; same-named segments not alike | Chapter 5 introduction (L5423–5426) | none | RQ1, RQ3 | text only | |
| B24 | 5.1: four limits (TD3 unmatched; TD-MPC2 names and unsplit pretraining; allocated rows less certain; MBPO/TD-MPC2 baselines differ between environments) | `sec:segment-shares` (L5537–5543) | sd ≤ 0.35 pp (SAC, TD3, TD-MPC2) and ≤ 2.1 pp (MBPO) | RQ1 | **Recomputed: match** (max sd 0.341 pp and 2.093 pp) | |
| B25 | 5.2: six limits — configurations not independent, p-values descriptive, 3–4 points per sweep; matmul-only; gross; allocation; MBPO FLOPs differ per seed and TD-MPC2 sweeps change the architecture; one GPU, two environments, networks ≤ ~1 M parameters | `sec:ec-interpretation` Limits (L5884–5892) | pooled slope 0.4715 / 0.5071, R² 0.611 / 0.625 | RQ2 | **Recomputed: match** (Task 1 RQ2) | |
| B26 | 5.3: unmatched settings (TD3 batch 100 in baseline/UTD/width rows; warmup 10,000 vs 5,000, excluded from totals; MBPO baseline differs between environments); one-at-a-time sweeps, interactions not run | `sec:rob-basis` (L5932–5937), `sec:sweeps` (L2941) | | RQ3 | text only | |
| B27 | Five paired seeds: exact sign-flip test cannot go below p = 0.0625, only direction is stated | `sec:rqs`, `sec:rob-basis`, `sec:env-basis`, `sec:env-summary` | p_min = 2/32 = 0.0625 | RQ3 | arithmetic | |
| B28 | MBPO L_max = 1 comparator for the environment effect has no same-commit/same-session pair | `sec:env-confounds` (L6366–6369) | HC baseline launched 2026-09-08 at `0bbbbb7`, Ant L_max = 1 runs 2026-09-17 at `788ffcd` | RQ3 | **Recomputed: match** (`metadata.json`: HC baseline L_max 1 at `0bbbbb7`, 2026-09-08; Ant L_max 1 and 15 at `788ffcd`, 2026-09-17) | |
| B29 | Simulator share of rollout time not measured; MBPO returns negative on Ant in all 9 configurations | `sec:env-episodes` (L6411, L6414–6416) | MBPO Ant last-10 % returns −756.7 … −244.6; HC 4535 … 5904 | RQ3 | **Recomputed: match** (see learning table) | |
| B30 | Clock study: single pair, single seed, no replication | `sec:clock-discussion` Limitations (L6599) | n = 1 per condition | RQ4 | text | |
| B31 | Clock study: R200 run after R2000 on the same machine; order/drift not separable | L6600 | 8 min gap; thermal gate 45 °C / 44 °C; head-to-tail drift +2.9 W / +5.8 W | RQ4 | **Recomputed: match** (start 10:29:59 and 11:09:15 UTC; end of first 11:01:08; final gate temp 45.0 / 44.0 °C) | |
| B32 | Clock study: lock recorded as requested; actual clock not logged | L6601 (`\todo`) | duration and power change as indirect evidence | RQ4 | text; actual clock NOT VERIFIED | |
| B33 | Clock study: provenance (command, configuration edit, executed code state not recorded; recorded commit predates MBPO) | L6479–6480, L6602 | commit `ae38808…` | RQ4 | **Recomputed:** both `metadata.json` record `ae3880860ebcce0c1e729714cdf48a08039adc46`; this hash is not among the 12 commits of the canonical runs | |
| B34 | Clock study: hardware parameters not recorded (supported clocks, power limit, driver) | L6603 | none | RQ4 | NOT FOUND (see list) | |
| B35 | Clock study: hidden size 256, not the canonical 1024; low-FLOP workload; larger networks "might be more clock-sensitive"; no utilisation data | L6604, L6592–6594 | R2000 is 0.33 sd above the canonical width-256 mean | RQ4 | **Recomputed: match** | |
| B36 | Clock study: rollout and synthetic rollout below the polling interval in every one of the 100 epochs at both clocks | L6541, L6605 | together < 1.3 % of training energy | RQ4 | **Recomputed:** shares of the clock-study runs from `segment_energy.json`/per-run CSV (rollout 1.2 %, synthetic 0.02 %) — match | |

**Group C — layout items not already covered by group A or B** (layout `thesis-layout.md` §5 table and §4 "7.6" list)

| # | Limitation | Where stated | Number/evidence | RQ(s) | Source / check | Layout flag |
|---|---|---|---|---|---|---|
| C1 | Overhead-validation run: data unavailable; reliability of `target_update` unverified | layout §5 #15, §4 (7.6 list: "`target_update` reliability unverified"); README "Known limitations" (L789–792) and CLAUDE.md recommend the control | none | RQ1 (smallest sub-segment `target_update`) | thesis text: **no mention at all** (no occurrence of an overhead/validation run in Chapters 3–7); data/script NOT FOUND (Task 5.4) | **in layout only, not in written Chapter 7** |
| C2 | Single GPU model (RTX 5090); two environments only | layout §5 #16, §4 (7.6 list) | — | RQ1–RQ4 | thesis states it in 5.2 limit 6 (L5891), 5.3.6 (L6183) and the hardware table, not in Chapter 7 | **in layout only, not in written Chapter 7** (stated in 5.2/5.3) |
| C3 | TD3 baseline not batch-matched (batch 100 vs 256) | layout §5 #6, §4 (7.6 list) | TD3 batch-256 run used as matched point | RQ1, RQ3 | stated in Chapter 5 (L5425, L5932), not in Chapter 7 | **in layout only, not in written Chapter 7** |
| C4 | Environment vs. configuration confounds (MBPO schedule, TD-MPC2 classifier) | layout §5 #8/#9, §4 (7.6 list) | MBPO synthetic samples 9.44× more on Ant; ratio 1.221 → 1.161 at equal L_max | RQ3 | stated in 5.4.1/5.4.5; `\todo` L6376 and L5406 ask to state it in Section 7 | **in layout only, not in written Chapter 7** |
| C5 | MBPO model-fit energy depends on early stopping and on the policy's data | layout §4 (7.6 list) | see B15 | RQ1, RQ3 | stated in 4.3.6 limit 5; `\todo` L4768 asks to state it in the threats | **in layout only, not in written Chapter 7** |
| C6 | Clock-frequency finding is n = 1 on a dev seed at hidden size 256 | layout §5 #17, §4 (7.6 list) | see B30–B35 | RQ4 | stated in Chapter 6 only | **in layout only, not in written Chapter 7** |
| C7 | Clock lock verified "as requested", not "as applied" (hand-started SAC baselines) | layout §5 #23 | 10 SAC baseline runs started by hand, error output not saved (L3163); all 80 TD3, 80 MBPO, 50 TD-MPC2 runs "verified as requested and as not reported failed, but not as applied" (L3754, L4017, L4811) | RQ1–RQ3 (all) | stated in 4.1–4.4 and 6.4.1; **Chapter 7.2 says** "The GPU graphics clock was locked to 2000 MHz in all canonical runs" (L6868–6869, and `tab:thr-provenance`) | **in layout only, not in written Chapter 7** (Chapter 7 states the lock as applied; see Consistency flags) |
| C8 | Allocated sub-segments not validated at sweep extremes; buffer-sample allocation the weakest case | layout §4 (7.6 list) | see B1, B8, B13 | RQ1 | stated in Chapters 3–4; not in Chapter 7 | **in layout only, not in written Chapter 7** |
| C9 | UTD status-file overwrite; TD-MPC2 false start (Appendix E only) | layout §5 #19 | `\todo` L3094 and L4812 describe them; Appendix E is not written | none (bookkeeping) | repository documents (CLAUDE.md "Data-quality exceptions"); not part of any result | **in layout only, not in written Chapter 7** |
| C10 | Non-randomised order / idle drift up to 23 % | layout §5 #22 | — | RQ3 | covered by A6, A7 | (covered; not a group-C item) |

Layout §5 items #1–#5, #10–#14 and #18 are conventions or reporting rules handled in Chapters 3–4 (allocation, `target_update`/`buffer_sample` units, TD-MPC2 names, TD3 warmup, sweep scope, signatures, kWh/J, dev runs); the "Dev-run-only issues are not discussed" rule (layout §5 footnote) is respected: searching `thesis-main.tex` L2089–7003 for "Humanoid" and "version drift" gives no hit (the dev runs appear only in `sec:dev-runs`, which says 20 development runs "are not part of any total, table, figure or aggregate", and in Chapter 6).

**Counts of tasks shorter than the 1 s polling interval, per algorithm** (recomputed from the 280 canonical `emissions_base_*.csv`; non-idle tasks; the thesis `\todo` L6874 says "all rollout tasks are shorter than 1 s" holds for SAC, TD3 and MBPO but not TD-MPC2):

| algorithm | rollout tasks | rollout tasks < 1 s | rollout min / median / max [s] | other tasks < 1 s | all non-idle tasks < 1 s |
|---|---|---|---|---|---|
| SAC | 7,000 | **7,000** | 0.1896 / 0.3409 / 0.5322 | warmup 70 | 7,070 |
| TD3 | 8,000 | **8,000** | 0.1003 / 0.2175 / 0.3494 | warmup 40 | 8,040 |
| MBPO | 8,000 | **8,000** | 0.1883 / 0.3536 / 0.4077 | synthetic_rollout_generation 8,000 (of 8,000), dynamics_model_update 1 (of 8,000), warmup 80 | 16,081 |
| TD-MPC2 | 5,000 | **0** | 6.616 / 11.07 / 16.68 | warmup 50 | 50 |

Verdict: **matches the `\todo`**. By environment the rollout minimum/maximum are: SAC HC 0.1896–0.3698 s, Ant 0.3385–0.5322 s; TD3 HC 0.1003–0.2078 s, Ant 0.2273–0.3494 s; MBPO HC 0.1883–0.2575 s, Ant 0.3414–0.4077 s; TD-MPC2 HC 6.616–13.64 s, Ant 7.312–16.68 s.

**Zero-GPU-energy tasks of short duration:** 5,854 tasks with `gpu_energy == 0` in total (4,140 at ≤ 10 ms; 745 at 10–25 ms; 521 at 25–50 ms; 448 at 50–100 ms; 0 at > 0.1 s); 5,854 of them are `synthetic_rollout_generation` (5,854 of 8,000 such tasks); CPU energy is positive in all 72,330 non-idle tasks. Matches `tab:thr-gpu-zero`.

### Learning sanity check

Per algorithm × environment over the canonical configurations. Return of a run = mean of the last `max(1, n//10)` train-phase episode returns (definition of `aggregate_results.py` L62–65, `reward__last_10pct_mean_episode_return`); "first 10 %" uses the first `max(1, n//10)` train-phase episodes (warm-up episodes excluded); a configuration "improves" if the seed mean of the last-10 % return exceeds the seed mean of the first-10 % return. Reported only; no conclusion.

| algo | env | n configs | n configs with last-10% mean > first-10% mean | min last-10% mean return | max last-10% mean return | n configs improving in all 5 seeds |
|---|---|---|---|---|---|---|
| MBPO | Ant | 9 | 0 | -756.7 | -244.6 | 0 |
| MBPO | HC | 7 | 7 | 4535 | 5904 | 7 |
| SAC | Ant | 7 | 3 | -220.1 | 408.6 | 3 |
| SAC | HC | 7 | 7 | 2921 | 4885 | 7 |
| TD3 | Ant | 8 | 7 | -30.31 | 575.9 | 7 |
| TD3 | HC | 8 | 8 | 3685 | 5540 | 8 |
| TD-MPC2 | Ant | 5 | 5 | 125.8 | 438.5 | 2 |
| TD-MPC2 | HC | 5 | 5 | 5791 | 7756 | 5 |

Non-improving configurations (seed-mean first → last 10 %): MBPO Ant — baseline −108.2 → −325.5, batch 512 −77.01 → −244.6, batch 1024 −89.43 → −348.5, rollout 1 −113.4 → −756.7, rollout 15 −98.78 → −407.7, UTD 2 −115.1 → −459.1, UTD 4 −104.4 → −599.2, width 256 −92.77 → −695.5, width 512 −74.28 → −413.4; SAC Ant — batch 512 −52.19 → −54.02, batch 1024 −56.55 → −104.5, UTD 2 −102.1 → −122.4, UTD 4 −152.7 → −220.1; TD3 Ant — batch 1024 31.36 → −30.31. All match `tab:app-learning`. Total: 56 configurations, 42 improving, 14 not (all on Ant).

## Task 5: Future work evidence

### 5.1 PPO and PETS

- Documented extension path: `README.md`, heading "## Extending to PPO / PETS" (L688–698); `CLAUDE.md`, heading "### Extending to PPO / PETS (not yet done)" (L175–181). Steps as written: (1) add a `{Algo}Config` dataclass to `configs/config.py` and register it in `ALGO_CONFIGS`; (2) add `algorithms/{algo}.py` with `train(env, algo_cfg, exp_cfg, tracker, device, logger, steps_per_epoch=...)` returning `(agent, energy_log, metrics)`, following the `TrackerTask` pattern (`algorithms/tracker_utils.py`; unique task names per epoch; sub-segment timing with `time.perf_counter()` where a finer breakdown than CodeCarbon's resolution is needed); (3) add a dispatch branch in `experiment_runner._dispatch_train`.
- Symbols verified to exist: `configs/config.py` (`ALGO_CONFIGS` L296–301, four entries; comment L295: "(PPO, PETS, ...); no changes to run_experiment.py are needed"); `algorithms/tracker_utils.py` (`class TrackerTask`, L19); `experiment_runner._dispatch_train` (L245; branches `sac`, `mbpo`, `td3`, `tdmpc2`; `raise ValueError` at L264; docstring L246–247 "each train() must return this 3-tuple"). `run_experiment.py` L32 builds `--algo` choices from `sorted(ALGO_CONFIGS)`, so no CLI change is needed.
- No PPO or PETS implementation exists in the repository (case-insensitive search of all `.py` files outside `.venv/`, `contexts/`: PETS appears only in comments describing MBPO's dynamics-model ensemble (`algorithms/dynamics_model.py`, `configs/config.py` L115, L121); PPO only in comments (`configs/config.py` L6, `experiment_runner.py` L83)). Stale wording noted: README says `tracker_utils.py` is "shared by `sac.py` and `mbpo.py`"; `td3.py` (L41) and `tdmpc2.py` (L76) also import it. The README's three steps do not mention the FLOP pipeline (`flop_analysis/flop_keys.py` has per-algorithm signature functions `sig_sac_td3`, `sig_mbpo`, `sig_tdmpc2`, per `CLAUDE.md`).

### 5.2 More environments

| file | algorithm | environment (per file name) | content |
|---|---|---|---|
| `configs/overrides/mbpo_hopper.json` | MBPO | Hopper | `updates_per_env_step` 40; rollout epochs 20–100; rollout length 1–15 |
| `configs/overrides/mbpo_walker2d.json` | MBPO | Walker2d | rollout epochs 20–150; length 1–15 |
| `configs/overrides/mbpo_humanoid.json` | MBPO | Humanoid | `model_hidden_sizes` [400,400,400,400]; model weight decays; rollout epochs 20–300; length 1–25 |

- No override files exist for Hopper, Walker2d or Humanoid for SAC, TD3 or TD-MPC2 (`configs/overrides/` listing: only the MBPO files above carry those names).
- Run directories: `results/` contains only `HalfCheetah-v5` and `Ant-v5` for MBPO, TD3, TD-MPC2; for SAC additionally `results/sac/Humanoid-v5/seed_56/` (2 run directories, 20260902_131059 and 20260902_135248 — dev data named in the task); no run directory for Hopper or Walker2d under any algorithm. No result from the dev runs is reported here.
- Documentation: `CLAUDE.md` ("Hopper/Walker2d/Humanoid have override files prepared but no recorded runs"; "natural next-step candidates"); `configs/config.py` L86–87 and L138–140 (MBPO docstring: Hopper `(20, 100, 1, 15)`, Walker2d `(20, 150, 1, 15)`, Ant `(20, 100, 1, 25)`, Humanoid `(20, 300, 1, 25)`; model size for Humanoid).

### 5.3 Other GPUs / hardware — assumptions in code and docs

| assumption | file : function / location | what must be provided on another machine, as stated by code/docs |
|---|---|---|
| RAPL (CPU energy, bare-metal Linux) | `configs/config.py` L46 `tracking_mode = "machine"`, L49–50 `force_cpu_power_w = None` (comment: "None => let CodeCarbon read real RAPL energy (Linux); only set this to a fixed watts figure as a fallback if RAPL is unreadable"); README "Known limitations" (L768ff) | readable `/sys/class/powercap/intel-rapl` (check `cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`); otherwise CodeCarbon falls back to "default power consumption of 4 W per thread" and `force_cpu_power_w` should be set; README: world-readable on a normal Linux desktop, "no root needed" |
| NVML (GPU energy, temperature, power) | `utils/thermal_gate.py` L33–36 (`import pynvml`, flag `_NVML_AVAILABLE`), L53–66 (read temperature/power; `nvmlInit`/`nvmlShutdown`), L180–182 (fallback fixed 30 s sleep if no NVML); CodeCarbon `core/gpu_nvidia.py` L54 (`nvmlDeviceGetTotalEnergyConsumption`); `requirements.txt` `nvidia-ml-py` | an NVIDIA GPU with NVML; without NVML the thermal gate degrades to a 30 s sleep (`"reason": "nvml_unavailable"`) and CodeCarbon has no GPU term |
| `nvidia-smi -pm` (persistence mode) | `utils/gpu_control.py` L57–60 `set_persistence_mode` | root, or `nvidia-persistenced` running (README) |
| `nvidia-smi -lgc` / `-rgc` (clock lock/reset) | `utils/gpu_control.py` L62–82 `lock_gpu_clocks`, L84–88 `reset_gpu_clocks`, L42–55 `query_gpu_clock_range` (`--query-supported-clocks=graphics`); values from `configs/config.py` L31–32 `gpu_min_clock_mhz = gpu_max_clock_mhz = 2000` (no CLI flag for the values; `--no-gpu-lock` only disables; `run_experiment.py` L46) | root; a GPU supporting a 2000 MHz graphics clock (the value is clamped to the supported range when `None` is passed, L66–78); on failure the code logs a warning and the run proceeds unlocked ("fails soft") |
| `cpupower frequency-set -g performance` (CPU governor) | `utils/gpu_control.py` L91–103 `set_cpu_governor`, L106 `get_cpu_governor`, `GpuCpuGuard` L118ff (restores the previous governor on exit) | `cpupower` installed, root; otherwise warning and unpinned governor |
| default lock value | `utils/gpu_control.py` L23 `DEFAULT_LOCK_MHZ = 2000` (comment: "representative of a real compute workload, not the idle floor") | used when min/max are `None` (explains `null` clock fields in 15 early SAC dev runs, per `CLAUDE.md`) |
| root requirement | `utils/gpu_control.py` docstring L4–8; README "Confound controls"; `cli_commands.txt` (`sudo /home/samanth/rl-dissection-linux/rl-exp/bin/python …`) | run under `sudo` for the confound controls to be applied |
| RAM heuristic "3 W per 8 GB" | **not set in this repository.** CodeCarbon 3.3.0 `codecarbon/external/ram.py`: L21–22 docstring "Before V3 heuristic: 3 watts of power for every 8GB"; active model L14 `RAM_SLOT_POWER_X86 = 5` W, `_estimate_dimm_count` (≤ 64 GB → 4 DIMMs), `_calculate_ram_power` L141–193 (≤ 4 DIMMs: 5 W × DIMM count, minimum 10 W); the recorded RAM power is 20.0 W for 60.8 GB (= 4 × 5 W) | an override only via CodeCarbon's `force_ram_power`; not exposed in `ExperimentConfig`. Read from the Windows dev venv `.venv/` (`codecarbon-3.3.0.dist-info`); the Linux `rl-exp` install is not accessible here (recorded `codecarbon_version` 3.3.0 in all runs) |
| FLOP counts measured on CPU | `flop_analysis/flops_per_call.json` key `_device_used_for_measurement` = `"cpu"` | none (FLOP counts are a property of the network classes) |
| thermal-gate tolerances | `metadata.json` `experiment_config`: temperature tolerance 2.0 °C, power tolerance 5.0 W, poll 5 s, maximum wait 600 s, reference file `results/_thermal_reference.json` | a cold reference is captured at process start on the new machine |

### 5.4 Overhead-validation run

- `README.md`, heading "Known limitations / things to sanity-check before trusting the numbers" (L768ff), bullet at L789–792: "Run the **overhead control** mentioned earlier: run the tracker across a no-op idle period of the same duration as a typical epoch, several times, and confirm the variance is small relative to your smallest measured segment (`target_update` will be your smallest — check this one first)." ("mentioned earlier": the README has no earlier passage that introduces it.) `CLAUDE.md` (Experiment protocol, "Known measurement caveats", L107–109): "an idle-period overhead control run is recommended before trusting the smallest segment (`target_update`)". Meant to confirm: tracker overhead variance small relative to the smallest segment; segment: `target_update`.
- Search for data or script: **NOT FOUND.** No file name containing "overhead" or "noop" anywhere outside `.venv/`; no `results/` directory for it (`results/` contains `mbpo`, `sac`, `td3`, `tdmpc2` and bookkeeping files only); `git log --all -i --grep=overhead` returns no commit; the only "overhead" hits in `.py`/`.sh`/`.md`/`.json`/`.csv` files are unrelated (`algorithms/replay_buffer.py` L1 "no torch overhead", `dashboard.py` L43 "calibration/exploration overhead", README L119 "nothing but overhead"). Layout: "a validation run was performed but its data is unavailable". The written thesis never mentions an overhead-validation run (see C1).

### 5.5 Directly measured critic/actor split — current `update()` structure

Each `agent.update()` call performs, in this order, per call: buffer sample → critic → actor → target; the training loop calls `update()` n times inside one `gradient_updates` CodeCarbon task per epoch (`sac.py` L344–349, `td3.py` L315, `tdmpc2.py` L790, `mbpo.py` L267), so critic, actor and target updates are interleaved per update call; the sub-phase timers are `time.perf_counter()` calls (host clock) inside `update()`; no `torch.cuda.synchronize()` exists in the code base (grep: 0 hits).

| file : `update()` | buffer_sample timer | critic_update timer | actor_update timer | target_update timer | notes |
|---|---|---|---|---|---|
| `algorithms/sac.py` `SACAgent.update` (L144) | L148 → L155 | L158 → L178 | L184 → L211, inside `if do_actor_update` (L183; `_update_counter % policy_update_delay == 0`; SAC default delay 1) incl. temperature (alpha) step | L216 → L224 (Polyak average of both critics) | `MBPO` reuses this class unmodified (`mbpo.py` L41, L166) |
| `algorithms/td3.py` `update` (L129) | L133 → L140 | L143 → L166 | L172 → L185, inside `if do_delayed_update` (condition L170–171; every `policy_update_delay` = 2 calls) | L188 → L199, same `if` block (actor + both critics Polyak) | actor and target on every second call |
| `algorithms/tdmpc2.py` `update` (L561) | L565 → L574 | L577 → L629 (world-model step: consistency + reward + value losses) | L636 → L638 (`_update_pi`, policy-prior step) | L642 → L647 (Q-ensemble Polyak average) | critic = world model, actor = policy prior (`tab:segment-meaning`) |

README passage on the trade-off (heading "Why epoch-level tagging instead of per-gradient-step tagging", L152–157): "If you later want a *directly measured* critic-vs-actor split instead of an allocated one, the only way to get it with CodeCarbon is to restructure the update loop so it does all critic updates for a chunk of steps, then all actor updates for that chunk, each as its own task — but that changes the SAC update order (actor updates then see a critic that's had many more updates than in standard interleaved SAC), so it's a real algorithmic tradeoff, not just an instrumentation change. Worth a footnote if you go that route."

### 5.6 Canonical-seed replication of the clock study

- Clock-study configuration: MBPO, HalfCheetah-v5, `hidden_sizes` 256, batch size 256, UTD 1 (`rollout_max_length` 1). Canonical runs matching it at 2000/2000 MHz (configuration `width256` of MBPO on HC; `gpu_min/max_clock_mhz` = 2000/2000 in their `metadata.json`):

| seed | run directory | total training energy [J] | duration [s] |
|---|---|---|---|
| 331 | `results/mbpo/HalfCheetah-v5/seed_331/20260912_225321` | 186 300 | 1 604 |
| 958 | `results/mbpo/HalfCheetah-v5/seed_958/20260912_232353` | 180 200 | 1 581 |
| 14577 | `results/mbpo/HalfCheetah-v5/seed_14577/20260912_235401` | 190 300 | 1 639 |
| 43611 | `results/mbpo/HalfCheetah-v5/seed_43611/20260913_002507` | 190 400 | 1 655 |
| 85062 | `results/mbpo/HalfCheetah-v5/seed_85062/20260913_005629` | 192 300 | 1 681 |

  mean 187 900 J, sd 4 850 J. (Matched seed-0 comparator: `results/mbpo/HalfCheetah-v5/seed_0/20260904_122958`, 189 500 J, 1 638 s — dev seed.)
- Canonical-seed runs at 200 MHz: **none.** Runs whose `experiment_config.gpu_min_clock_mhz`/`gpu_max_clock_mhz` are not 2000/2000 (16 of 300; none canonical):
  - 200/200: `results/mbpo/HalfCheetah-v5/seed_0/20260904_130915` (the clock-study 200 MHz run).
  - `null`/`null` (15 early SAC dev runs): `results/sac/HalfCheetah-v5/seed_0/` {20260813_112621, 20260813_121500, 20260819_104727, 20260819_115112, 20260821_105506, 20260824_110200, 20260824_115117}; `…/seed_42/` {20260824_122611, 20260824_130005, 20260826_111906}; `…/seed_56/` {20260831_131353, 20260831_135517, 20260831_143605}; `results/sac/Humanoid-v5/seed_56/` {20260902_131059, 20260902_135248}.
- 200 MHz run: energy to completion 568 200 J (568 209.95), duration 5 732 s (5 731.98 s), mean power 99.13 W. (Note, from the commented thesis appendix `\todo` L7017: the repository documents describe the 15 `null`-clock runs as equivalent to 2000 MHz, "which the commit history does not support"; `CLAUDE.md` indeed states `DEFAULT_LOCK_MHZ=2000` fallback; the README/CLAUDE.md correction is flagged by that `\todo`, not made.)

### 5.7 Other open items stated in the thesis text

Search over the written content (`thesis-main.tex` L2089–7003; terms `not measured`, `not tested`, `left open`, `remains`, `future`, `\todo`, `ch:conclusion`, `Chapter~8`, `Chapter 8`, plus near-synonyms `not quantified`, `not examined`, `not investigated`, `not separated`, `not run`, `not computed`, `cannot be separated`, `not recorded`, `not available`). The thesis itself assigns to Chapter 8 only: (i) the J/FLOP-vs-useful-computation question (L2185, L2186(e)) and (ii) a TD-MPC2 Ant-v5 run without the termination classifier (L5018). Counts (matching lines): 22 for the exact terms, 48 with the near-synonyms, 122 uncommented `\todo` (listed below). `Chapter~8` / `Chapter 8` / `ch:conclusion` appear in L2185, L2186, L5018 only.

Statements (not chosen for the Outlook; "Ch. 8?" = whether the thesis itself says it belongs to Chapter 8):

| statement (short quote) | section label | Ch. 8? |
|---|---|---|
| "This question is left open and is mentioned in \zcref{ch:conclusion}." (J/FLOP says nothing about whether the computation was useful) | `sec:metric` L2185 | **yes** |
| "add one such configuration to the future work of Chapter~8, since it would separate the energy of the classifier from the environment" (no Ant-v5 TD-MPC2 run without the classifier) | `sec:tdmpc2-baseline` L5018 `\todo` | **yes** |
| "The second number is not measured on the experiment machine." (achieved FLOP/s fraction of peak) | `sec:flop-principle` L2581 | no |
| "The share of the simulator in the rollout time was not measured." | `sec:sac-baseline` L3333; `sec:td3-baseline` L3928; `sec:tdmpc2-baseline` L5015; `sec:sac-interpretation` L3702; `sec:mbpo-interpretation` L4747; `sec:env-episodes` L6411 | no |
| "Whether they explain the difference was not tested." (actor-step operations without counted FLOPs) | `sec:sac-interpretation` L3691 | no |
| "this was not tested" (smaller model buffer cheaper to sample) | `sec:mbpo-interpretation` L4730 | no |
| "A direct test would need a profile of the fit with device synchronisation, which this study did not run." | `sec:mbpo-interpretation` L4705 | no |
| "This is consistent with its higher J/FLOP but was not tested." (world-model step) | `sec:tdmpc2-interpretation` L5348 | no |
| "this was not tested" (sampler loops over stored episodes on Ant) | `sec:tdmpc2-interpretation` L5384, L5406(4) | no |
| "the energy of an interleaved schedule was not measured" | `sec:tdmpc2-interpretation` L5403 | no |
| "the effect of this was not quantified" (first/last second of a task) | `sec:tdmpc2-interpretation` L5396 | no |
| "A count that includes them could behave differently, and it was not tried." (non-matmul ops) | `sec:ec-interpretation` L5887 | no |
| "The present data cannot separate these hypotheses, because the achieved fraction of the peak throughput of the GPU was not derived" | `sec:ec-interpretation` L5865 | no |
| "Neither was computed." (pooled fits without TD-MPC2) | `sec:ec-interpretation` L5855 `\todo` | no |
| "Interactions between swept parameters, for example width and batch size, were not run and are not studied." | `sec:sweeps` L2941 | no |
| "A pairing that separates the classifier ... was not run; state this as a limit in Section 7." | `sec:env-confounds` L6376 `\todo` | no (assigned to Section 7) |
| "Rollout and synthetic rollout generation are below the polling interval" / lock "was not logged" / hardware parameters "not recorded" | `sec:clock-discussion` L6601–6605 | no |
| "the effect of time-of-run on the sweep and environment comparisons was not quantified" | `sec:thr-order` L6831 | no |
| "The cause of the upward drift has not been investigated" | `sec:thr-idle` L6771–6772 | no |
| "the update period itself was not measured" (GPU energy counter); "a direct test of this was not performed" | `sec:thr-resolution` L6670–6671, L6703 | no |
| "Identify the cause of the growth from the MBPO implementation ... before stating it." (MBPO `dynamics_model_update` growth) | `sec:thr-stationarity` L6753 `\todo` | no |
| analysis (b) hardware utilisation (peak throughput fraction) and (c) GPU-only J/FLOP: "either remove them or move the open point to the discussion in `ch:evaluation`" | `sec:ec-interpretation` L5894 `\todo` | no (assigned to Chapter 7) |
| per-algorithm correlations "not computed" / bootstrap intervals / Appendix D | `sec:metric` L2186 `\todo` | partly: item (e) is the Chapter 8 label |
| Section 7 discussion expected to contain the interpretation of Chapters 5 (L5427–5428, L5913, L6184, L6328, L6427–6428) and of RQ1–RQ4 (L2265) | `ch:comparison`, `sec:robustness`, `sec:rob-summary`, `sec:env-power`, `sec:env-summary`, `sec:rqs` | no (assigned to Chapter 7) |

All 122 uncommented `\todo` (line, section label, first words), as an inventory:

- L2123 `sec:context`: Check consistency with Sections 1.2 and 2.5 once written: the claim that energy is usually reported as on…
- L2124 `sec:context`: The numbers in the second paragraph are taken from the baseline and sweep results of ch:per-algorithm (en…
- L2179 `sec:metric`: Support the two reasons. (1) State the training budgets of the original publications for comparison and c…
- L2186 `sec:metric`: (a) The analysis is pooled over algorithms and sweeps. Decide whether per-algorithm correlations ($n$ bet…
- L2235 `sec:rqs`: Confirm that the J/FLOP spreads reported in Chapter~4 are standard deviations of per-seed ratios. This al…
- L2249 `sec:rqs`: Confirm Spearman's rho as the rank correlation of RQ2 (chosen for consistency with sec:metric). Only the …
- L2258 `sec:rqs`: Kendall's tau is a proposal and no code for it exists yet. With three algorithms an ordering has six poss…
- L2266 `sec:rqs`: Placeholder labels to replace: ch:clock (Chapter 6), ch:evaluation (Chapter 7). Check that sec:dev-runs s…
- L2358 `sec:epoch-tasks`: Insert minimum and median task duration of rollout and gradient updates per algorithm from mean duration …
- L2409 `sec:measured-allocated`: Quantify as before: sum of sub-phase times divided by the measured duration, and one run with torch.cuda.…
- L2596 `sec:flop-principle`: Check that Section 2.4 states the matmul-only scope and the factor of two per multiply--add; remove the r…
- L2634 `sec:flop-counts`: Confirm $N_{\mathrm{env = \num{100000$ (100 epochs of \num{1000 steps) for all canonical runs, as in the …
- L2663 `sec:flop-signatures`: Check that the number of groups still matches the sweep matrix in the experimental design section.…
- L2758 `sec:flop-jpf`: State, once the metric of RQ3 is fixed (Section on detailed research questions), whether the spread is re…
- L2771 `sec:flop-caveats`: Decide whether to quote the measured J/FLOP magnitudes of the two kinds of rollout here or only in the re…
- L2816 `sec:flop-verification`: Rule 4: these checks ran over every recorded run directory, including the development runs. Reconcile the…
- L2827 `sec:flop-verification`: Add this reconciliation as a script to the repository and extend it to all signatures before citing it as…
- L2829 `sec:flop-verification`: Placeholder labels to replace: sec:compute-accounting (Section 2.4), sec:design (Section 3.6), sec:energy…
- L2869 `sec:environments`: Check the observation dimension of Ant-v5 under the Gymnasium options used in the runs (default contact-f…
- L2885 `sec:seeds`: Confirm that the seeds were fixed before any canonical run, and state the order in which the runs of a sw…
- L2935 `sec:baselines`: (1) State why the width \num{1024 was chosen, if it was a deliberate choice beyond ``settled during devel…
- L3006 `sec:dev-runs`: Rule~4 exception: the GPU clock study of the chapter on GPU clock frequency uses one dev seed. If it is k…
- L3034 `sec:setup`: RAM | capacity, type, speed…
- L3035 `sec:setup`: Mainboard | model…
- L3036 `sec:setup`: Storage | drive type and model…
- L3037 `sec:setup`: Operating system | Debian version…
- L3038 `sec:setup`: Linux kernel | version…
- L3039 `sec:setup`: NVIDIA driver | version, open kernel modules…
- L3052 `sec:setup`: Python | version…
- L3056 `sec:setup`: Gymnasium~\cite{towers2024gymnasium | version…
- L3057 `sec:setup`: MuJoCo~\cite{todorov2012mujoco | version…
- L3058 `sec:setup`: NVML bindings (nvidia-ml-py) | version…
- L3062 `sec:setup`: Fill the open entries from pip freeze of the experiment virtual environment, and from nvidia-smi, uname -…
- L3094 `sec:launch`: The status files of the UTD sweep were overwritten when its script was reused, so the completion of the S…
- L3116 `sec:data-quality`: Quantify the idle-baseline drift promised in sec:protocol: for each run, the relative difference between …
- L3117 `sec:data-quality`: The log checks were run over all recorded run directories, which include the development runs. A property…
- L3156 `sec:sac`: This analytic count was derived by the context generator and is not a script of the repository. Add it to…
- L3164 `sec:sac`: Reconcile with sec:data-quality: the check of run.log cannot detect failures of the clock lock, the gover…
- L3176 `sec:sac`: Returns of the training policy are reported in app:learning only as a sanity check and are not used in an…
- L3297 `sec:sac-baseline`: The peak FP32 throughput of the GPU is not derived in the repository, so the achieved throughput cannot y…
- L3384 `sec:sac-baseline`: The rollout tasks are shorter than the polling interval, which contradicts the intent of the TODO in sec:…
- L3675 `sec:sac-interpretation`: This fixed-versus-marginal fit and the pairwise delta E/delta F of the dashboard (\num{65.91 and \num{62.…
- L3715 `sec:sac-interpretation`: Check the dependence of the Ant-v5 rollout energy on the number of episodes and resets per run (per-run e…
- L3747 `sec:td3`: As for SAC, the analytic count was derived by the context generator and is not a script of the repository…
- L3756 `sec:td3`: Reconcile with sec:data-quality as requested in sec:sac: the check of run.log cannot detect these failure…
- L3765 `sec:td3`: Returns of the training policy are reported in app:learning only as a sanity check. On HalfCheetah-v5 the…
- L3892 `sec:td3-baseline`: The peak FP32 throughput of the GPU is not derived in the repository, so the achieved throughput cannot y…
- L3981 `sec:td3-baseline`: As for SAC, the rollout tasks are shorter than the polling interval. Rewrite the paragraph on task durati…
- L4009 `sec:mbpo`: This analytic count was derived by the context generator and is not a script of the repository. Add it to…
- L4030 `sec:mbpo`: Returns of the training policy are reported in app:learning only as a sanity check. On HalfCheetah-v5 the…
- L4053 `sec:mbpo-baseline`: The sections on the algorithm-specific segments (sec:alg-segments) and on the baseline configurations (se…
- L4172 `sec:mbpo-baseline`: The peak FP32 throughput of the GPU is not derived in the repository, so the achieved throughput cannot y…
- L4179 `sec:mbpo-baseline`: The CPU, GPU and RAM composition of the four measured tasks and the idle-floor fractions for all \num{16 …
- L4215 `sec:mbpo-baseline`: This answers the open point of sec:metric (energy per epoch roughly constant after the first epochs) for …
- L4283 `sec:mbpo-baseline`: All \num{8000 rollout tasks and all \num{8000 synthetic-rollout tasks are shorter than the polling interv…
- L4593 `sec:mbpo-batch`: The SAC totals in tab:mbpo-sac are the rounded values of the SAC section. Recompute the total ratios from…
- L4667 `sec:mbpo-rollout`: These ratios are ratios of means. Pair them by seed in the pipeline to obtain the paired sd and the permu…
- L4721 `sec:mbpo-interpretation`: This fit and the pairwise Delta E/Delta F of the dashboard (UTD \num{64.82 and \num{62.07\,pJ/FLOP, width…
- L4768 `sec:mbpo-interpretation`: (1) State in sec:threats that the model-fit energy depends on early stopping and on the data of the polic…
- L4802 `sec:tdmpc2`: This analytic count was derived by the context generator and is not a script of the repository. Add it to…
- L4812 `sec:tdmpc2`: (1) Appendix~E: the false start of the num_q/horizon sweep consists of five failed launches (override fil…
- L4829 `sec:tdmpc2`: Returns of the training policy are reported in app:learning only as a sanity check. On HalfCheetah-v5 the…
- L4863 `sec:tdmpc2-baseline`: The layout and sec:context describe the rollout as the dominant segment of TD-MPC2. This holds for the si…
- L4974 `sec:tdmpc2-baseline`: The peak FP32 throughput of the GPU is not derived in the repository, so the achieved throughput cannot y…
- L5018 `sec:tdmpc2-baseline`: State in sec:alg-segments or sec:baselines whether the termination classifier follows an episodic mode of…
- L5036 `sec:tdmpc2-baseline`: Decide which idle-drift statistic the thesis quotes, the as-recorded one or the window-corrected one, and…
- L5374 `sec:tdmpc2-interpretation`: These fits and the pairwise Delta E/ Delta F of the dashboard (rollout \num{45.59 and \num{44.11\,pJ/FLOP…
- L5406 `sec:tdmpc2-interpretation`: (1) State in sec:threats that the allocation prices the Polyak average and the buffer sample at the mean …
- L5428 `ch:comparison`: Confirm that the interpretation of Chapter~5 is collected in Section 7.5 and not given in the chapter its…
- L5447 `sec:segment-shares`: TD3 is shown at its own baseline (batch size \num{100), taken from sec:td3-baseline. The batch-matched co…
- L5514 `sec:segment-shares`: Word the dominance of the TD-MPC2 rollout the same way in sec:context, in the Introduction and here: the …
- L5535 `sec:segment-shares`: Recompute the total ratios from the full-precision means of Appendix~B. The ratios of MBPO are those of t…
- L5544 `sec:segment-shares`: Consider a stacked bar chart of tab:cmp-shares (one bar per algorithm and environment, measured segments …
- L5611 `sec:ec-data`: The analysis scripts are flop_analysis/section_5_2_analysis.py and section_5_2_render.py (Git commit 2b48…
- L5649 `sec:ec-predictor`: Create fig:ec-scatter from d1_config_table.csv: two panels (one per environment), log--log axes, colour b…
- L5723 `sec:ec-predictor`: These ratios were computed by hand from the J/FLOP values of ch:per-algorithm and are not output of the a…
- L5769 `sec:ec-fixed-marginal`: The intercept shares of the baseline energy (in this paragraph and the next) are derived by hand from tab…
- L5830 `sec:ec-fixed-marginal`: sec:sac-interpretation quotes pooled marginal values from the dashboard (\num{65.91 and \num{62.78\,pJ/FL…
- L5855 `sec:ec-interpretation`: Check the grouping with fig:ec-scatter and compute the pooled fit of rollout without TD-MPC2, and the poo…
- L5882 `sec:ec-interpretation`: Connect to sec:related-flop (Section 2.5.4, placeholder label). State which FLOP-based estimators of \cit…
- L5894 `sec:ec-interpretation`: State in sec:threats the limits that concern this section, and add the point on the matmul-only count if …
- L5914 `sec:robustness`: Check that this matches the decision on where the interpretation of Chapter~5 goes. Section~5.2 currently…
- L5953 `sec:rob-basis`: Update Section~3.3 accordingly: the RQ3 row of tab:rqs and part (ii) of the paragraph on RQ3 still name K…
- L6059 `sec:rob-jpf`: The per-seed orderings and the pairwise seed counts are in o3_seed_level.csv and o3b_pairwise_seed_counts…
- L6064 `sec:rob-jpf`: Create a figure for tab:rob-jpf: one panel per environment, the sweep values on the horizontal axis, the …
- L6135 `sec:rob-tdmpc2`: The numbers agree with the tables of sec:tdmpc2. Keep them in sync with the final version of that section…
- L6170 `sec:rob-tdmpc2`: Decide whether tab:rob-tdmpc2 and tab:rob-ranges stay in the main text or move to app:full-results, with …
- L6184 `sec:rob-summary`: Carry the following into the interpretation in ch:evaluation: (i) the FLOPs of an algorithm change by muc…
- L6226 `sec:env-basis`: Tag the TD3 batch-100 rows as in Section 5.3. If the per-configuration tables of the appendix are added, …
- L6309 `sec:env-shares`: The thesis has no tab:td3-env or tab:tdmpc2-env; their per-configuration values are only in the prose of …
- L6328 `sec:env-power`: Interpretation of the J/FLOP ratio (fixed cost of the loop versus marginal cost of the FLOPs, sec:energy-…
- L6370 `sec:env-confounds`: Replace the earlier todo of the MBPO rollout-length subsection (thesis-main.tex, line 4667) by a referenc…
- L6376 `sec:env-confounds`: Check the formulation against the open todo at the end of the TD-MPC2 chapter (thesis-main.tex, line 5406…
- L6412 `sec:env-episodes`: No reference on the environment dependence of the energy of the simulator or of the energy of the RL loop…
- L6417 `sec:env-episodes`: Check whether this limit is stated in Section 3 (scope) and Section 7 (threats, sec:threats); if so, repl…
- L6428 `sec:env-summary`: Check that the numbers in this summary are repeated in the RQ summary of Chapter 7. Resolve the reference…
- L6439 `ch:clock`: Align the wording of RQ4 with the formulation in Section~\ref{sec:research-questions.…
- L6482 `sec:clock-setup`: Decide whether to re-run both configurations from a committed state before submission; if not, keep the c…
- L6544 `Where the Change Occurs`: Check this against the allocation-validation discussion in Section~\ref{sec:allocation and cross-referenc…
- L6556 `Idle Floor and Above-Idle Energy`: State in one sentence which of the head/tail windows the other chapters use as floor, to keep the idle-fl…
- L6564 `Component Split`: The context file only reports the CPU power increase; add an interpretation only if it is supported by a …
- L6590 `sec:clock-discussion`: Verify that these three references actually support the wording above before submission, and soften the s…
- L6601 `Limitations`: item \textbf{Lock verification. The clock is recorded as requested in metadata.json; the clock the GPU ac…
- L6603 `Limitations`: item \textbf{Hardware parameters. Supported clock range, power limit, and driver version were not recorde…
- L6615 `sec:clock-summary`: Rephrase the last sentence once the RQ4 formulation is fixed.…
- L6654 `sec:thr-resolution`: confirm RAPL mode via CodeCarbon's start-up log line or ls /sys/class/powercap/intel-rapl* on the researc…
- L6664 `sec:thr-resolution`: Align this paragraph with the RAM description in sec:measured-allocated.…
- L6671 `sec:thr-resolution`: measured poll nvmlDeviceGetTotalEnergyConsumption at about 1\,kHz under load and histogram the step spaci…
- L6712 `sec:thr-resolution`: interpreted. State in sec:mbpo-baseline that this segment's energy is unreliable.…
- L6753 `sec:thr-stationarity`: Identify the cause of the growth from the MBPO implementation (e.g.\ model training on a growing replay b…
- L6772 `sec:thr-idle`: not been investigated optional: compare head idle power per session.…
- L6832 `sec:thr-order`: optional: compare head idle power and baseline-configuration energy across sessions.…
- L6874 `sec:eval-repro`: Fix the claim ``all rollout tasks are shorter than 1\,s'' in earlier chapters: it holds for SAC, TD3 and …
- L6884 `sec:eval-repro`: run torch.get_float32_matmul_precision() and torch.backends.cuda.matmul.allow_tf32 in rl-exp.…
- L6919 `sec:thr-summary`: Reproducibility only fill from research PC…
- L6979 `app:reproducibility`: Distribution | cat /etc/os-release (draft table says Debian)…
- L6986 `app:reproducibility`: Gymnasium, MuJoCo, nvidia-ml-py | pip freeze in rl-exp…
- L6987 `app:reproducibility`: Driver, kernel modules, VBIOS | nvidia-smi…
- L6988 `app:reproducibility`: Power limits, supported clocks | nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS…
- L6989 `app:reproducibility`: SM count, compute capability, peak FP32 | torch.cuda.get_device_properties(0)…
- L6990 `app:reproducibility`: RAM type/speed, mainboard, storage | lscpu, lsblk, dmidecode…
- L7001 `app:reproducibility`: sweeps (09-18/19). Optionally replace this list by the session table from execution_timeline.csv.…

## Task 6: Cross-reference labels

Defined `ch:*`, `sec:*`, `app:*` labels in the written thesis content (`thesis-main.tex` L2089–7003; numbers as in `thesis-main.toc`). `ch:conclusion` is **PLACEHOLDER**: referenced once (L2185, `\zcref{ch:conclusion}` in `sec:metric`; also named in the `\todo` L2186(e)) and not defined anywhere (a commented-out `\chapter{Conclusion and Outlook}\label{sec:conclusion}` exists at L2073, not active). Chapter 8 will be chapter 8 (after Chapter 7 "Evaluation").

| No. | Label | Heading | Line (thesis-main.tex) |
|---|---|---|---|
| 3 | `ch:method` | Problem Exposition and Method | 2089 |
| 3.1 | `sec:context` | Context | 2097 |
| 3.2 | `sec:metric` | Choice of Metric | 2130 |
| 3.3 | `sec:rqs` | Detailed Research Questions | 2192 |
| 3.4 | `sec:harness` | Measurement Harness and Protocol | 2273 |
| 3.4.1 | `sec:protocol` | Protocol Phases | 2281 |
| 3.4.2 | `sec:epoch-tasks` | Per-Epoch Task Design | 2347 |
| 3.4.3 | `sec:measured-allocated` | Measured versus Allocated Segments | 2366 |
| 3.4.4 | `sec:alg-segments` | Algorithm-Specific Segments | 2434 |
| 3.4.5 | `sec:confounds` | Confound Controls | 2513 |
| 3.4.6 | `sec:units` | Energy Sources, Units and Accounting Convention | 2543 |
| 3.5 | `sec:flop-accounting` | FLOP Accounting | 2564 |
| 3.5.1 | `sec:flop-principle` | Principle and Counting Convention | 2577 |
| 3.5.2 | `sec:flop-per-call` | Per-Call FLOP Measurement | 2598 |
| 3.5.3 | `sec:flop-counts` | Reconstruction of Call Counts | 2622 |
| 3.5.4 | `sec:flop-signatures` | Architecture Signatures | 2649 |
| 3.5.5 | `sec:flop-jpf` | Energy per FLOP | 2665 |
| 3.5.6 | `sec:flop-caveats` | Limits of the Metric | 2760 |
| 3.5.7 | `sec:flop-verification` | Pipeline and Verification | 2784 |
| 3.6 | `sec:design` | Experimental Design | 2836 |
| 3.6.1 | `sec:environments` | Environments | 2843 |
| 3.6.2 | `sec:seeds` | Seeds and Paired Design | 2874 |
| 3.6.3 | `sec:baselines` | Baseline Configurations | 2887 |
| 3.6.4 | `sec:sweeps` | Sweeps | 2937 |
| 3.6.5 | `sec:dev-runs` | Development Runs | 3001 |
| 3.7 | `sec:execution` | Execution and Data Quality | 3011 |
| 3.7.1 | `sec:setup` | Hardware and Software | 3016 |
| 3.7.2 | `sec:launch` | Launch and Execution | 3064 |
| 3.7.3 | `sec:data-quality` | Data Quality | 3096 |
| 4 | `ch:per-algorithm` | Per-Algorithm Energy Analysis | 3123 |
| 4.1 | `sec:sac` | SAC | 3141 |
| 4.1.1 | `sec:sac-baseline` | Baseline Decomposition and Energy per FLOP | 3179 |
| 4.1.2 | `sec:sac-utd` | Update-to-Data Ratio | 3387 |
| 4.1.3 | `sec:sac-width` | Network Width | 3461 |
| 4.1.4 | `sec:sac-batch` | Batch Size | 3542 |
| 4.1.5 | `sec:sac-interpretation` | Interpretation | 3649 |
| 4.2 | `sec:td3` | TD3 | 3722 |
| 4.2.1 | `sec:td3-baseline` | Baseline Decomposition and Energy per FLOP | 3768 |
| 4.3 | `sec:mbpo` | MBPO | 3988 |
| 4.3.1 | `sec:mbpo-baseline` | Baseline Decomposition and Energy per FLOP | 4033 |
| 4.3.2 | `sec:mbpo-utd` | Update-to-Data Ratio | 4286 |
| 4.3.3 | `sec:mbpo-width` | Network Width | 4367 |
| 4.3.4 | `sec:mbpo-batch` | Batch Size | 4450 |
| 4.3.5 | `sec:mbpo-rollout` | Maximum Rollout Length | 4596 |
| 4.3.6 | `sec:mbpo-interpretation` | Interpretation | 4670 |
| 4.4 | `sec:tdmpc2` | TD-MPC2 | 4774 |
| 4.4.1 | `sec:tdmpc2-baseline` | Baseline Decomposition and Energy per FLOP | 4832 |
| 4.4.2 | `sec:tdmpc2-pretrain` | The Pretraining Segment | 5067 |
| 4.4.3 | `sec:tdmpc2-numq` | Size of the Q-Ensemble | 5118 |
| 4.4.4 | `sec:tdmpc2-horizon` | Planning Horizon | 5209 |
| 4.4.5 | `sec:tdmpc2-interpretation` | Interpretation | 5303 |
| 5 | `ch:comparison` | Cross-Algorithm Comparison | 5412 |
| 5.1 | `sec:segment-shares` | Segment Shares Across Algorithms | 5434 |
| 5.2 | `sec:energy-vs-compute` | Energy versus Compute: Are FLOP-Based Energy Estimates Trustworthy? | 5551 |
| 5.2.1 | `sec:ec-data` | Analyses and Data | 5566 |
| 5.2.2 | `sec:ec-predictor` | FLOPs as a Predictor of Energy | 5614 |
| 5.2.3 | `sec:ec-fixed-marginal` | Fixed and Marginal Cost | 5726 |
| 5.2.4 | `sec:ec-interpretation` | Interpretation | 5833 |
| 5.3 | `sec:robustness` | Robustness of the Comparison Across Sweeps | 5901 |
| 5.3.1 | `sec:rob-basis` | Comparison Basis | 5917 |
| 5.3.2 | `sec:rob-energy` | Ordering by Energy | 5956 |
| 5.3.3 | `sec:rob-jpf` | Ordering by Energy per FLOP | 6002 |
| 5.3.4 | `sec:rob-rollout` | MBPO Rollout Length | 6067 |
| 5.3.5 | `sec:rob-tdmpc2` | TD-MPC2 and the Range of the Other Algorithms | 6098 |
| 5.3.6 | `sec:rob-summary` | Summary for RQ3 | 6173 |
| 5.4 | `sec:env-effect` | Environment Effect, Quantified | 6192 |
| 5.4.1 | `sec:env-basis` | Pairing, Statistics and Confounds | 6207 |
| 5.4.2 | `sec:env-energy` | Energy Ratios | 6228 |
| 5.4.3 | `sec:env-shares` | Shifts in the Segment Shares | 6295 |
| 5.4.4 | `sec:env-power` | Power and Energy per FLOP | 6311 |
| 5.4.5 | `sec:env-confounds` | Pairs That Differ in More Than the Environment | 6330 |
| 5.4.6 | `sec:env-episodes` | Episodes and Wall-Clock Time per Rollout Step | 6378 |
| 5.4.7 | `sec:env-summary` | Summary | 6419 |
| 6 | `ch:clock` | GPU Clock Frequency | 6436 |
| 6.1 | `sec:clock-scope` | Scope and Status of the Evidence | 6445 |
| 6.2 | `sec:clock-setup` | Experimental Setup | 6458 |
| 6.3 | `sec:clock-results` | Results | 6485 |
| 6.4 | `sec:clock-discussion` | Discussion | 6584 |
| 6.5 | `sec:clock-summary` | Summary | 6609 |
| 7 | `ch:evaluation` | Evaluation | 6623 |
| 7.1 | `sec:threats` | Threats to validity | 6638 |
| 7.1.1 | `sec:thr-resolution` | Resolution of task-level energy measurements | 6641 |
| 7.1.2 | `sec:thr-stationarity` | Stationarity of per-epoch energy | 6714 |
| 7.1.3 | `sec:thr-idle` | Idle-power drift and the gross convention | 6762 |
| 7.1.4 | `sec:thr-order` | Execution order and time confounding | 6815 |
| 7.1.5 | `sec:thr-learning` | Learning behaviour of the trained agents | 6834 |
| 7.2 | `sec:eval-repro` | Provenance and reproducibility | 6862 |
| 7.3 | `sec:thr-summary` | Summary of threats | 6887 |
| 7.4 | `app:learning` | Learning check details | 6927 |
| 7.5 | `app:reproducibility` | Reproducibility details | 6966 |

- `app:learning` (7.4) and `app:reproducibility` (7.5) are defined as `\section`s inside Chapter 7 ("APPENDIX BLOCKS (place in the appendix chapters)", L6925); `app:clock-details`, `app:clock-provenance`, `app:clock-suballoc`, `app:clock-idle`, `app:clock-full` exist only inside a commented block (L7004–7080) and are not defined in the compiled thesis.
- Template-sample labels before L2089 (Introduction/Background/Problem-exposition/Evaluation sample text; Chapters 1–2 are not written): `sec:introduction`, `sec:introduction:topic`, `sec:introduction:state-of-art`, `sec:introduction:research-question`, `sec:introduction:method`, `sec:introduction:results`, `sec:background`, `sec:problem-exposition` (+ `:context-understanding`, `:data-understanding`, `:research-problems`, `:research-method`), `sec:problem1` (+ `:subproblem1`, `:subproblem2`), `sec:sub-problem2`, `sec:evaluation` (+ `:objective`, `:setup`, `:execution`, `:results`, `:discussion`), `sec:latexhints`, `sec:ex:cref`, `sec:ex:vref`, and `sec:appendix1` (L7088, "My first appendix"). Chapter 1 = "Introduction" (template), Chapter 2 = "LaTeX Hints" (to be removed), per `thesis-main.toc`.
- **Referenced but not defined** (`ch`/`sec`/`app`, uncommented references): `app:full-results` (12 references), `app:hyperparameters` (2), `app:flop-details` (2), `app:clock-details`, `app:clock-provenance` (2), `ch:conclusion`, `sec:compute-accounting` (2; Section 2.4), `sec:related` (2; Section 2.5.5), `sec:related-flop` (Section 2.5.4), `sec:td3-utd`, `sec:td3-width`, `sec:td3-batch`, `sec:td3-interpretation` (Section 4.2 subsections announced but not written), `sec:allocation`, `sec:energy-definitions`, `sec:hardware`, `sec:ram-heuristic`, `sec:research-questions` (cited from Chapter 6 only). No undefined `tab:`/`fig:`/`eq:` references.

## Consistency flags

- **Chapter 7 contains no RQ-by-RQ discussion.** The layout (`thesis-layout.md` §4) plans 7.1 Objective, 7.2 Setup, 7.3 Execution, 7.4 Results (answer to each RQ), 7.5 Discussion, 7.6 Threats. The written Chapter 7 "Evaluation" (`ch:evaluation`, `thesis-main.tex` L6623) is, per its own introduction (L6626–6629), "does not add new algorithm comparisons; it assesses the study itself"; its sections are 7.1 Threats to validity (with 7.1.1–7.1.5), 7.2 Provenance and reproducibility, 7.3 Summary of threats, 7.4 Learning check details, 7.5 Reproducibility details. The interpretation of RQ1–RQ3 currently sits in Chapter 4 (`sec:sac-interpretation`, `sec:mbpo-interpretation`, `sec:tdmpc2-interpretation`; **none for TD3**), in 5.2.4 (`sec:ec-interpretation`, RQ2) and in the descriptive summaries 5.3.6 and 5.4.7 (RQ3); RQ4 is answered in Chapter 6 (6.4 Discussion, 6.5 Summary). RQ1 across algorithms has no summary paragraph in 5.1.
- **Forward references to a Chapter 7 interpretation that does not exist:** L2265 ("`ch:evaluation` brings the answers to all four questions together"), L5427 and `\todo` L5428 (interpretation of Chapter 5 "collected in Section 7.5"), L5913, `\todo` L6184 ("Carry the following into the interpretation in `ch:evaluation`"), `\todo` L6328, L6427 ("The interpretation follows in `ch:evaluation`"), `\todo` L6428 ("repeated in the RQ summary of Chapter 7"). Also section-number references in `\todo` text ("Section 7.5" = discussion, "Section 7.6" = threats, `thesis-layout.md` numbering) do not match the compiled numbering (7.1 = threats; 7.5 = "Reproducibility details").
- **Section 4.2 (TD3) is incomplete:** `sec:td3` announces `sec:td3-utd`, `sec:td3-width`, `sec:td3-batch`, `sec:td3-interpretation` (L3730–3737) that are not in the file; the TD3 sweep results appear only in Chapter 5 tables; `tab:td3-env` does not exist; `thesis-layout.md` §4 describes 4.2 as drafted, but `current-status.txt` (same content as `thesis-main.tex`) contains only `sec:td3-baseline`.
- **Development runs and the "280 runs" statements:** `sec:dev-runs` (L3001–3005) says the 20 development runs are in no total, table, figure or aggregate and does not yet name the clock-study exception (`\todo` L3006); Chapter 6 uses two dev runs (L6450); `ch:evaluation` introduction says "the 20 development runs are never used" (L6635); `sec:env-confounds` L6373 states "all 26 HalfCheetah-v5 runs" of TD-MPC2, but `results/tdmpc2/HalfCheetah-v5/` has 26 run directories only because it includes `seed_0` (one development run) — the canonical count is 25 (recomputed). L6373 therefore counts a development run (the sentence's conclusion — `episodic` false on HC — is unaffected: 25 of 25 canonical HC runs have `epFalse`, 25 of 25 Ant runs `epTrue`).
- **`tab:rqs` / Section 3.3 still name Kendall's τ** (L2221, L2252, `\todo` L2258) whereas Section 5.3.1 (L5949–5953) states that τ is not computed and that orderings and swapped pairs are reported (the layout §2 note asks for this edit).
- **Clock lock "as applied":** Chapters 4.1–4.4 and 6.4.1 say the 2000 MHz lock is verified as requested and as not reported failed, not as applied (L3163, L3754, L4017, L4811, L6601; the 200 MHz run's actual clock was not logged), whereas Chapter 7.2 states "The GPU graphics clock was locked to 2000 MHz in all canonical runs" (L6868–6869) and `tab:thr-provenance` lists "clocks locked 2000/2000 MHz".
- **`sec:thr-resolution` vs. earlier `\todo`s:** 7.1.1 states that tasks shorter than the polling interval are valid counter differences ("The polling interval ... therefore does not quantise task energy"), while `\todo` L3384, L3981, L4283 and the RQ text in 4.x still say "check how CodeCarbon accounts the energy of a task shorter than one polling interval" and that rollout energies "carry a resolution uncertainty that these data do not quantify" (L3708, L4755–4756). `\todo` L6874 notes that "all rollout tasks are shorter than 1 s" is false for TD-MPC2 (recomputed: 0 of 5,000).
- **RAM model:** `thesis-layout.md` §3.3 and the task text say RAM energy is "3 W per 8 GB"; the code in CodeCarbon 3.3.0 uses a DIMM-count model (5 W per DIMM; 4 DIMMs for 60.8 GB → 20.0 W, the value recorded in all 72,890 rows); `ram.py` mentions 3 W / 8 GB only as the pre-V3 heuristic. `sec:units` (L2548) says "a constant heuristic based on the machine's RAM capacity"; `sec:thr-resolution` states 20.0 W.
- **Idle drift per algorithm:** layout §3.3 states "median ~4–9 % per algorithm, max 23 %" (as recorded) and "~7–13 % median" (window-corrected). Recomputed per algorithm as recorded: MBPO 4.13 %, SAC 8.37 %, TD3 9.28 %, TD-MPC2 10.04 % (range 4.1–10.0 %); window-corrected: MBPO 7.68 %, SAC 9.53 %, TD3 13.02 %, TD-MPC2 13.83 % (range 7.7–13.8 %). `CLAUDE.md` quotes 261/280 (window-corrected) while Chapter 7 quotes 256/280 (as recorded); both reproduce. `CLAUDE.md` mean head 63.1 W / tail 69.5 W is the window-corrected pair; Chapter 7's 65.3 / 69.5 W is the as-recorded pair.
- **Count "seven items not recorded"** (`tab:thr-summary`, L6918) vs. 6 `\todo` rows in `tab:thr-provenance` and the longer list in L6876–6881.
- **Idle floor in Chapter 6 vs. Chapter 7:** Chapter 6 uses the mean of the two window-corrected windows of the run itself as floor (L6549); Chapter 7's net-energy sensitivity uses "the mean of head and tail power" for the 40 baseline runs (L6776) without saying corrected or as recorded; `\todo` L6556 asks for this to be stated.
- **`CLAUDE.md` / `README.md` stale items:** `CLAUDE.md` names `f09a861` as current HEAD (now `050b3c3`); README says `tracker_utils.py` is shared by `sac.py` and `mbpo.py` only; README "Run the overhead control mentioned earlier" has no earlier mention; README/CLAUDE.md describe the 15 `null`-clock dev runs as equivalent to 2000 MHz (`\todo` L7017 in the commented clock appendix disputes this).
- **`contexts/thesis-layout.md`** is referenced by the task but exists only at `Thesis Report/thesis-layout.md`.
- **Figures/tables of Chapter 5 not yet produced:** `fig:ec-scatter` (placeholder box, L5642–5648), figure for `tab:rob-jpf` (`\todo` L6064), stacked bar chart for `tab:cmp-shares` (`\todo` L5544).

## NOT FOUND / NOT VERIFIED (collected list)

1. `contexts/thesis-layout.md` — NOT FOUND at that path (read from `Thesis Report/thesis-layout.md`).
2. LICENSE file in the repository — NO LICENSE FILE.
3. Public visibility of `https://github.com/SamBro2901/rl-dissections` — NOT VERIFIED (no network check; stated by the user).
4. Overhead-validation run (data or script) — NOT FOUND in the repository, `results/` or the git history; the written thesis does not mention it.
5. RQ1 answer statement for TD3 — NO ANSWER STATEMENT IN THESIS TEXT (Section 4.2 has no interpretation subsection).
6. Experiment-machine details (distribution and release, kernel modules, NVIDIA driver, Gymnasium/MuJoCo/`nvidia-ml-py` versions, power limits, supported clocks, SM count, peak FP32, RAM type/speed, mainboard, storage) — NOT FOUND in `metadata.json`, `run.log`, `results/` or the repository (thesis `\todo` L3034–3062, L6979–6990).
7. Clock frequency actually applied during any run — NOT VERIFIED (only the requested value is recorded; thesis L6601).
8. RAPL mode on the research PC — NOT VERIFIED (no start-up log line stored; thesis `\todo` L6654); the "no RAPL fallback warning" check itself is verified (0/280 logs).
9. Equality of the Linux `rl-exp` CodeCarbon install with the Windows `.venv` copy from which `ram.py` was read — NOT VERIFIED (only the version 3.3.0 is identical).
10. Stationarity values of `tab:thr-stationarity` (ratios 0.520/0.437, R² 0.984/0.993) — NOT VERIFIED (thesis only).
11. `tab:thr-gross-net` (gross vs net energy and J/FLOP, net/gross 0.47–0.65) — NOT VERIFIED (thesis only).
12. Execution-order counts of `sec:thr-order` (56 of 56 consecutive seed blocks; 44 of 48 sweep–baseline pairs on different days, 5.6–239.6 h; 11 of 27 HC/Ant pairs on different days) — NOT VERIFIED (117.7 h, no overlap, date span and 12 commits are verified).
13. Sub-segment timer coverage "86–98 %" of `thesis-layout.md` §3.2 — NOT VERIFIED here (only TD-MPC2 97.64–98.70 % as stated by the thesis, and the clock-study values 61.4 % / 97.6 %, were checked).
14. "Seven items not recorded" (`tab:thr-summary`) — NOT VERIFIED (6 `\todo` rows in `tab:thr-provenance`).
15. Seed-level ordering counts (80 of 80 for energy, 78 of 80 for J/FLOP, per-seed pair counts) and the outputs of `section_5_2/5_3_analysis.py` (`tab:ec-algo`, `tab:ec-fixed-marginal`, `tab:ec-delta`) — NOT VERIFIED (thesis only; those scripts and `o3_seed_level.csv` were not re-run).
16. Clock-study "1,196 dynamics-model fit epochs" (thesis L6474) — NOT VERIFIED (optimiser steps 168,525, synthetic samples 990,000 and FLOPs 3.153e14 are verified).
17. Definition of the idle floor used by Chapter 7 for the net-energy sensitivity (corrected or as recorded) — NOT FOUND in the thesis text (`\todo` L6556).
18. Chapter 1 (Introduction), Chapter 2 (Background / Related work) and Appendices A–E of the layout — NOT FOUND (not written; template placeholders only); labels `ch:conclusion`, `app:full-results`, `app:hyperparameters`, `app:flop-details`, `sec:compute-accounting`, `sec:related`, `sec:related-flop` are referenced but undefined.
19. Section 4.2 subsections `sec:td3-utd`, `sec:td3-width`, `sec:td3-batch`, `sec:td3-interpretation` — NOT FOUND (referenced, not written).
20. Runtime float32 matmul precision (TF32) in the experiment environment — NOT VERIFIED (not set anywhere in the repository code; runtime default not recorded; thesis `\todo` L6884).
21. A Section 5.4 analysis script and `env_comparison_paired.csv` (cited by `\todo` L6309) — NOT FOUND in the repository.
