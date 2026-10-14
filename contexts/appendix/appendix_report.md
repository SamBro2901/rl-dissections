# Appendix report (2026-10-14)

Session type: Windows analysis machine (Python 3.13.9, SciPy 1.16.2, pandas 2.3.3, NumPy 2.3.4; no PyTorch, no GPU). Nothing that needs the research PC was run; nothing was committed or pushed.
Data: `Project Repository/rl-dissections` (HEAD `94e1de288e7aa8e3709568c66f3b136048961c90`, no tags). Thesis: `Thesis Report/thesis-main.tex`.
Locations (split by repository, as agreed): deliverables, scripts, data and logs in `rl-dissections/contexts/appendix/`; the generator `rl-dissections/flop_analysis/appendix_tables.py` (+ `appendix_tables_verify.py`); the Appendix D figures in `Thesis Report/figures/`; the edited `thesis-main.tex` and its backup `thesis-main.tex.bak-appendix` in `Thesis Report/`.
Build: `python -I contexts/appendix/build_all.py` (from the repository root) regenerates every table and appendix file and re-inserts them into `thesis-main.tex` (always from the backup).
The compile you permitted ("compile as needed, fall back to static check") was done on a copy in the scratchpad (lualatex, biber, lualatex twice); the files in `Thesis Report/` were not touched by it. Result: see "Build and static checks" at the end.

## 0. What is where

| Item | Path |
|---|---|
| Appendix chapters | `contexts/appendix/appendix_A.tex` ... `appendix_F.tex`, `appendices.tex` (inputs them), `appendices_inline.tex` (what was pasted into the thesis) |
| Templates and generators | `appendix_?.tpl.tex`, `gen_A.py`, `gen_A_sweep_table.py`, `gen_B.py`, `gen_C_tables.py`, `gen_D.py`, `gen_D_figures.py`, `gen_E.py`, `gen_F.py`, `insert_appendix.py`, `build_all.py` |
| Verification scripts | `A_verify.py`, `build_A_overrides.py`, `B_independent_check.py`, `C_totals_check.py`, `C_flop_tables.py`, `D_learning.py`, `D_verify.py`, `E_sessions.py`, `F_clock_verify.py`, `labelcheck.py`, `texlint.py`, `diff_check.py`, `overfull_report.py`, `todo_count.py` |
| Generated data | `contexts/appendix/appendix_data/` (A_, B/, C_, D_, E_, F_ prefixes; `B/B_*.tex` are the table fragments of Appendix B) |
| Logs | `edit_log.md`, `research_pc_checklist.md`, this report |
| Figures (Appendix D) | `Thesis Report/figures/app-return-{sac,td3,mbpo,tdmpc2}-{hc,ant}.tex` and `.csv` (16 files, `\input{figures/...}` paths match) |
| Backup | `Thesis Report/thesis-main.tex.bak-appendix` |

Order of the chapters after `\appendix`: A `app:hyperparameters`, B `app:full-results`, C `app:flop-details`, D `app:learning`, E `app:reproducibility`, F `app:clock-details` (F is its own chapter, as instructed; you may fold it into D or E, the labels `app:clock-*` can then stay).
Size (lines in `thesis-main.tex` / pages in the scratch build): see the table in "Build and static checks".

## 1. Appendix A: Hyperparameters and Configuration (`app:hyperparameters`)

**Produced.** Four parameter tables (SAC, TD3, MBPO, TD-MPC2; every field of the configuration class, `†` implementation setting not from the publication, `‡` deliberate deviation), a table of the protocol settings common to all runs, a table of the 37 override files with parameter, value, algorithm, environment and number of canonical runs (the 6 baseline configurations without a file account for 30 runs, the files for 250), the verification statement, and the MBPO refit paragraph (`model_train_freq` 250, condition evaluated once per epoch, 100 fits per run, warm start, deviation from the publication).
**Sources.** `configs/config.py`, `configs/overrides/*.json`, `metadata.json` of the 280 runs, `algorithms/mbpo.py` (`train()`, `_rollout_length`), `algorithms/dynamics_model.py` (`fit`), `utils/gpu_control.py`, `utils/thermal_gate.py`, `run_experiment.py`.
**Verification log.**
- Logged `algo_config` = dataclass default with the override file of `tab:app-hp-sweeps` applied: 280 of 280 runs, 0 deviations (`build_A_overrides.py`).
- Each of the 56 configurations differs from the baseline of its algorithm and environment in exactly the swept field (0 exceptions); the five seeds of a configuration have identical `algo_config` and `experiment_config` apart from the seed and one logging field (`A_verify.py`, `A_config_diffs.csv`).
- `experiment_config` is identical in all 280 runs except `warmup_steps` (10000 in the 80 TD3 runs) and the logging field `thermal_gate_reference_file`, which is stored in 3 runs only (SAC HalfCheetah-v5 baseline, seeds 331, 958, 14577) because the runner stopped writing it to the logged configuration from the fourth canonical run on (diff of `92dcee0`). The thermal gate reached its reference in 280 of 280 runs.
- Each of the 80 MBPO runs has a model fit in all 100 epochs (100 fits per run); the code line `if (global_step - last_model_train_step) >= mbpo_cfg.model_train_freq` is evaluated once per epoch (L232 of `mbpo.py`).
- Three override files exist that no canonical run uses (`mbpo_hopper.json`, `mbpo_walker2d.json`, `mbpo_humanoid.json`); they are not listed in the appendix.
**Disagreements with the thesis text.** None. Note: the claim of Chapter 4 "the logged configuration of every run differs from the baseline in exactly the swept factor" holds for `algo_config`; for `experiment_config` it holds apart from the logging field above.
**Todos.** The todo "Appendix~A must state that `model_train_freq` is 250 ..." (old L3825) is deleted (resolved). No new todo.

## 2. Appendix B: Full Result Tables (`app:full-results`)

**Produced.** `flop_analysis/appendix_tables.py` writes 38 table fragments (`appendix_data/B/B_*.tex`) from the raw run files and the Section 5 context CSVs; `gen_B.py` assembles the chapter. Contents: for each algorithm the tables of its section of Chapter 4 for all configurations (energy and shares, FLOPs and J/FLOP, duration/power/throughput, composition and idle-floor fractions, measurement diagnostics), MBPO fit statistics (two tables), MBPO against SAC (14 pairs), full-precision environment tables of SAC and MBPO; for Chapter 5 the statistics of all algorithm--segment combinations (L5435), the fits `E = a + bF` of all segments on configuration means and on single runs, with `n = 3` flagged (L5472), all adjacent-pair costs with the MBPO totals marked unreliable and dropped pairs listed (L5549), the values with standard deviations behind `tab:rob-energy` and `tab:rob-jpf` (L5711, L5761), the per-seed orderings and pairwise seed counts (L5811), the cross-seed table of all 56 configurations, and a verification table. TD3 enters Chapter 4 with its baseline (as Section 4.2 does) and with all 16 configurations in the diagnostics table. The composition and idle-floor fractions of all 16 MBPO configurations (the old todo L3941) are in `tab:app-mbpo-comp`.
**Verification log.**
- Per-run recomputation from `segment_energy.json` and the per-task CodeCarbon files: total energy equals the per-run table to 4.3e-16 (relative); CPU+GPU+RAM energies of the measured tasks sum to the task energy within 3.8e-14 (SAC), 1.3e-14 (TD3), 4.1e-12 (MBPO), 2.3e-15 (TD-MPC2).
- Row-wise check of the 15 baseline tables of Chapter 4 against the recomputed values (`appendix_tables_verify.py`): 846 cells, 844 equal the recomputed value rounded to the printed digits; the other two differ in the last printed digit: `tab:tdmpc2-base-energy`, `target_update`, sd of the energy on HalfCheetah-v5, printed 109.8, recomputed 109.7498 (the context file `tdmpc2_cross_seed_summary.csv` has 109.749763 too, so the thesis printed a double-rounded value), and `tab:tdmpc2-base-comp`, sd of the RAM share in the gradient updates on HalfCheetah-v5, printed 0.1613, recomputed 0.161250 (rounding tie). Largest relative difference of the other cells: 4.5e-4 (rounding to four digits).
- Look-up of every printed number of 50 tables (all tables of Sections 4.1 to 4.4 with numbers of the types above, `tab:ec-pooled`, `tab:ec-algo`, `tab:ec-fits`, `tab:ec-delta`, `tab:rob-energy`, `tab:rob-jpf`, `tab:rob-tdmpc2`, `tab:rob-ranges`): 3293 numbers, 3256 found at the printed precision; the 37 not found are derived columns that are not tabulated (Ant/HC ratios in the J/FLOP tables, `Samples` and first-fit columns, sweep values 256/512/1024). `B_verification.csv` lists them.
- Independent recomputation from the raw runs (`B_independent_check.py`): the 22 fits `E = a + bF` of the training total (intercept, slope, R2: max relative deviation 2.4e-14, 6.0e-15, 4.4e-16 absolute), the 46 adjacent pair costs of the totals (1.7e-14), and the 56 rows of `cross_seed_energy_per_flop.csv` (energy 2.9e-16, J/FLOP 4.1e-16) agree with the context CSVs the Section 5 tables are generated from.
**Disagreements.** (1) The instructions put `env_comparison_paired.csv`, `per_run_segments_5_2.csv`, `t2_all_config_values.csv`, `t3_...`, `o3_...`, `o3b_...` and `config_means.csv` in `flop_analysis/output/`; they are in `contexts/section_5.{2,3,4}_data/` and `flop_analysis/output/correlation/`. No CSV named in the text is missing. (2) The two rounding cases above.
**Length.** About 3200 lines of LaTeX and about 60 pages in the scratch build (section 8), the longest chapter of the thesis. If it is too long for print, the tables that are subsets of main-text tables can be dropped by deleting their placeholder in `appendix_B.tpl.tex`; nothing else refers to them except `sec:app-full-*` sentences.
**Todos.** Deleted: old L5346 (scripts of Section 5.2 and per-segment statistics), L5811 (per-seed orderings), L5952 (scripts of Section 5.3), L3941. Kept (author decisions, see section "Decisions delivered as variants"): the todos at old L5937 and L6099 in Chapter 5, plus three `AUTHOR` todos at the head of the optional blocks.

## 3. Appendix C: FLOP-Counting Details (`app:flop-details`)

**Produced.** Counting rules (forward and backward cost, first-layer input gradient, frozen critics, Polyak operations, TD3 delay) with the equations of the per-call FLOPs of SAC, TD3, the MBPO ensemble and TD-MPC2 (planning call, world-model step, policy-prior step, termination classifier), the `fit()` formula with its derivation and a worked example for one baseline run per environment, the reconciliation of the textbook estimate with the counted critic count (5254938624 = 3 x 2 F(Q) + target computation - first-layer input gradient), the per-call table of all 42 canonical signatures (three tables), the analytic cross-check summary, and the consistency checks of the totals.
**Sources.** `flop_analysis/` (`measure_flops.py`, `flops_per_call.json`, `flop_keys.py`, `compute_energy_per_flop.py`, `check_totals.py`, `verify_mbpo_fit_flops.py` and its log, `verify_analytic_flops.py`), `algorithms/dynamics_model.py`, `algorithms/sac.py`, `td3.py`, `tdmpc2.py` (code paths named in `verify_analytic_flops.py`).
**Verification log.**
- `verify_analytic_flops.py` rerun (torch-free): 196 comparisons, 0 differ. Of these, 188 belong to the 42 canonical signatures (SAC 10 signatures / 40 comparisons, TD3 12 / 48, MBPO 10 / 60, TD-MPC2 10 / 40); the other 8 are for a development environment (two SAC signatures) and are not reported in the appendix. All 188 are exact (`C_crosscheck_summary.csv`, `C_verify_analytic_output.txt`).
- `check_totals.py` rerun: passes (it covers all run directories). Restricted to the 280 canonical runs (`C_totals_check.py`): total energy = sum of training segments, max relative error 0; total FLOPs = sum of the matmul rows, exact for 280/280; exactly one total row per run (280/280); no unexpected segment key; sub-segment sum = measured gradient-update energy: max relative deviation 2.2e-16 with exact parsing of the per-task CSV and 4.8e-13 with the default pandas float parser (3.6e-16 with `float_precision='round_trip'`).
- MBPO fit formula: for one baseline run (seed 331) per environment the formula over the 100 fits gives the logged FLOPs (HalfCheetah-v5 245164122000000, Ant-v5 484342085920000), the logged optimiser steps (161913, 234243) and the logged FLOPs of the synthetic rollouts (3961726560000, 45987298247920) exactly. Holdout share 7.776 % (HalfCheetah-v5) and 7.993 % (Ant-v5) in all 35 and 45 runs. Exact match of the formula with `FlopCounterMode` at three buffer sizes and both environments and exact linearity in the batch size: taken from `verify_mbpo_fit_flops.log` (not rerun, no PyTorch here; `\todo{NOT RUN}` in the appendix).
- TD-MPC2: sum of the three counted regions equals the unmodified `update()` for all 10 signatures (from `flops_per_call.json`, recomputed); classifier FLOPs 9673113600 per planning call and 2418278400 per world-model step on the Ant-v5 baseline, 1.221e15 per run (12.12 % of the run total); the five HalfCheetah-v5 signatures have none.
**Disagreements.** (1) `todos.md` quotes "196 of 196" exact comparisons; 8 of them belong to a development environment, the canonical count is 188 of 188. (2) Section 3.5.2 says the per-call counts were measured "on the GPU of the experiment machine"; `flops_per_call.json` records `_device_used_for_measurement: cpu` (proposal in `edit_log.md`). (3) Section 3.5.6 gives 4.8e-13 for the sub-segment sum; this is the pandas-parsing figure, the exact figure is 2.2e-16 (both stated in the appendix; proposal in `edit_log.md`).
**Todos.** Deleted: old L2575 (reconciliation script) and L2577 (placeholder labels). New: `AUTHOR` (commit and cite `verify_analytic_flops.py`), `NOT RUN` (rerun `verify_mbpo_fit_flops.py`). The four existing "Commit ... and cite the commit" todos in Chapter 4 are kept.

## 4. Appendix D: Learning check details (`app:learning`)

**Produced.** Definitions (first/last 10 %, return, return per step, which episodes), the return curves (8 pgfplots figures: one per algorithm and environment, panels per sweep, ten windows of 10000 steps, mean and sd over five seeds), the HalfCheetah-v5 table of all 27 configurations, the existing `tab:app-learning` (verbatim) plus a new per-step table of the 14 flagged Ant-v5 configurations, the existing `tab:app-config-means` (verbatim), an optional Spearman block, and a random-policy table skeleton with the `NOT RUN` todo.
**Verification log.** Everything recomputed from the 280 `training_metrics.json` (`D_learning.py`).
- `tab:app-learning` (14 rows) and `tab:app-config-means` (56 rows) agree with the recomputation: maximum absolute difference 0.05 (display rounding to one decimal) for returns and lengths, maximum relative difference 3.6e-4 for FLOPs (four digits), 1.6e-3 for energies in kJ (three digits). No difference.
- HalfCheetah-v5: the last-10 % mean is above the first-10 % mean in all 27 configurations and in all 5 seeds of each (135 of 135 runs); range of the last-10 % means 2920.6 (SAC batch 1024) to 7755.7 (TD-MPC2 `num_q` 7), i.e. 2921 to 7756 as quoted in Chapter 7. 100 episodes per run, k = 10 in all runs.
- Ant-v5: 14 configurations not above; 13 of 14 have longer late episodes; the return per step rises in 10 of 14 and in 7 of the 9 MBPO configurations; the four configurations whose return per step does not rise are MBPO rollout 1, MBPO width 256, SAC batch 1024, TD3 batch 1024 (the four marked with an asterisk in `tab:app-learning`). Episodes per run 154 to 2556, k from 15 to 255.
- Definition read from the analysis script (`contexts/section_7_data/item2_learning_check.py`): k = max(1, floor(n/10)) of the training episodes (phase `train`, warm-up and the incomplete last episode excluded); return = unweighted mean of episode returns; return per step = sum of returns / sum of lengths (ratio of sums).
- Pooled Spearman values reproduce `tab:spearman` (0.417, 0.0303; -0.326, 0.0842; 0.836, 5.78e-8; 0.846, 7.33e-9).
**Disagreements.** None. The instruction asked for "the four marked with a dagger"; the existing table marks them with an asterisk, which is kept.
**Todos.** `NOT RUN` random-policy return (command F3, see the Linux list); `AUTHOR` for the optional Spearman block. The banner comment "APPENDIX BLOCKS" disappeared with the insertion (it was in the replaced region). Chapter title kept.
**Recommendations (not decisions).** `tab:app-config-means` (56 rows) stays in D (Chapter 3 cites it and it holds the returns). Appendix B now also holds a 56-row cross-seed table (`tab:app-cross-seed`) with energy, duration, power, FLOPs and J/FLOP including sds, so energy and FLOPs appear in both; B has no return column, so I would keep both. The rename of the chapter is not needed: it still holds a results table, but it is about returns.

## 5. Appendix E: Reproducibility details (`app:reproducibility`)

**Produced.** The existing `tab:thr-provenance` (verbatim, five research-PC todos kept), `tab:sessions` (ten sessions: first start, last end, span, gap, runs per algorithm/environment, content, commits) with the threshold statement, the E24 paragraphs (false start of the TD-MPC2 sweep, status files of the UTD sweep), the code-provenance table of the 12 commits with the changes of the training and measurement code, the note on the first SAC group, the table of analysis scripts with version-control status, the package versions of the analysis environment, a "How to run" section (environment, `sudo`, CLI options, thermal gate, output layout, sweep scripts with status files), and the repository-state todo.
**Verification log.**
- Sessions (`E_sessions.py`, from the start/end times of the 280 runs): ten sessions for 30, 60 and 120 minutes; largest gap inside a session 9.6 minutes, smallest gap between sessions 3.3 hours; 280 runs; sum of run durations 117.7 h; sum of session spans 118.6 h. The values of the E24 table (start, hours, runs per session) are reproduced.
- False start (log `results/tdmpc2/_numq_horizon_sweep.log`): five launches 12:18:39 to 12:18:42 (local) fail with `FileNotFoundError` for `tdmpc2_num_q3.json` raised in `build_algo_config` before the runner; sixth launch (Ant-v5, seed 331) has a start line and nothing else; relaunch at 12:21:00; the first run directory of the sweep is `20260918_122100`; no run directory has a timestamp between 12:18 and 12:21; the sweep log has 46 launch blocks (40 + 6). The E24 text said the failed launches "ended within about four seconds"; each ended after about one second (all five between 12:18:39 and 12:18:42), the text of the appendix says so.
- UTD status files: `results/_utd_sweep.log` has 50 launch blocks (SAC Ant-v5 10, TD3 20, MBPO 20) with completion lines "30/30" and "20/20"; the status file describes the 20 MBPO runs (all done); `results/sac/HalfCheetah-v5/_utd_sweep.log` has 10 blocks ("10/10"); `run_utd_sweep.sh` itself now only contains MBPO.
- Commits: 12 distinct commits, 280 runs (8+2+10+10+10+30+10+60+20+35+45+40). Diffs between consecutive recorded commits in `algorithms/`, `configs/config.py`, `experiment_runner.py`, `run_experiment.py`, `utils/`: `92dcee0` adds start/end temperature and power logging and removes the thermal-reference file name from the logged configuration (logging only), `0bbbbb7` sets the MBPO default hidden size to (1024,1024) (one line), `58962d2` adds TD3 (module, config class, 4 dispatch lines), `b5b7e54` adds TD-MPC2 (module, config class, 4 dispatch lines); all other consecutive pairs have no difference; SAC, MBPO, dynamics model, buffers and clock-control utilities are identical between the first and the last recorded commit.
- First SAC group (commit `f23f448`, 8 runs): metadata of runs 1 to 3 has neither the thermal reference nor start/end values, run 4 has `reference_temp_c`/`reference_power_w`, runs 5 to 8 have start/end values; so the working tree changed twice (before run 4 and before run 5). Step 7 and E21 mention only the change before the fifth run (four earliest lack the start temperature); the appendix states both.
- Override files `sac_utd2.json` and `sac_utd4.json` first appear in `0bbbbb7`, after the 10 SAC UTD runs that record `ecb98bb`; their logged configuration still equals default plus file.
- Scripts: `050b3c3` (2026-10-08) is the first commit of `section_5_2_analysis.py`, `section_5_2_render.py`, `section_5_3_analysis.py`, `section_5_3_render.py` (`git log --follow --diff-filter=A`), identical at HEAD; `section_{5,6,7,8}_checks.py`, `verify_analytic_flops.py` and the new `appendix_tables.py`, `appendix_tables_verify.py` are not under version control; `contexts/step1/scripts/` is untracked.
- Versions of the analysis environment: Python 3.13.9, SciPy 1.16.2, pandas 2.3.3, NumPy 2.3.4 (checked on this machine).
**Disagreements.** (1) "about four seconds" (E24) vs one second per failed launch (log). (2) Step 7 and E21 describe one edit of the working tree in the first SAC group, the metadata show two (see above). (3) `todo-appendix.md` and the spec list `execution_timeline.csv` as the source; the table is regenerated from the metadata and agrees.
**Todos.** Kept: the five research-PC rows of `tab:thr-provenance`; new `NOT RUN` (`pip freeze` in `rl-exp`), `AUTHOR` (contribution 5: LICENSE, commit of untracked scripts, tag). Deleted with the replaced region: "Optionally replace this list by the session table".
**Manual items (not done).** Contribution 5: `LICENSE` file, commit of `verify_analytic_flops.py`, `section_{5,6,7,8}_checks.py`, `appendix_tables.py`, `appendix_tables_verify.py` and `contexts/` files, tag of the cited commit (`git tag` is empty; HEAD is `94e1de288e7aa8e3709568c66f3b136048961c90` and changes with your commit).

## 6. Appendix F: Clock Experiment Details (`app:clock-details`)

**Produced.** Active version of the commented block: `\chapter` (numbered), `app:clock-details`, section labels `app:clock-provenance`, `-suballoc`, `-idle`, `-full`, tables `tab:app-clock-suballoc`, `tab:app-clock-idle`, `tab:app-clock-full`, the allocated-power sentence as a `\paragraph`, the provenance list, and a paragraph on the unset clock fields of the development runs.
**Verification log** (`F_clock_verify.py` from the two run directories; all reproduce the old block): sub-segment energies R2000 / R200 in kJ 1.326/3.677, 18.957/34.663, 22.756/40.765 (printed 40.77), 1.961/2.660, gradient updates 44.999/81.765 (printed 81.76); ratios 2.77, 1.83, 1.79, 1.36, 1.82; allocated power 113.6 W and 156.0 W; measured gradient-update power 110.9 W and 95.8 W; window-corrected idle power 63.40/66.27/64.84 W and 57.33/63.09/60.21 W (as recorded 65.62/66.27 and 59.26/63.08); both runs record commit `ae38808` (2026-09-02 14:53 +02:00; `algorithms/mbpo.py` is not in that tree; first commit of the file and of both run directories `f109bd4`, 2026-09-04 14:51 +02:00), torch 2.13.0+cu130, CodeCarbon 3.3.0, hidden sizes (256,256), clock request 2000/2000 and 200/200 MHz; `f23f448` (2026-09-07) sets the defaults to 2000/2000. The full segment table (12 rows per run) is built from `contexts/Section_6_data/C_segment_summary.csv`; its sub-segment, GU and idle rows agree with the recomputation.
**Open item: what `None` meant** (`git show <commit>:utils/gpu_control.py`):
- Commits recorded by the 15 development runs with null clock fields (all SAC, 13 August to 2 September 2026: `97eb7a7`, `64fb0cd`, `20b4d4f`, `4b9f11a`, `2aa4ccd`, `cf4d327`, `d05cbe6`, `d5880ed`) and `ae38808`: `lock_gpu_clocks(None, None)` queries the supported graphics clocks and sets each unset bound to the **lowest** supported clock (code comment: "lock to a single fixed low clock by default"; the config comment says "None => query device and use its min / a fixed low-variance value"). The value of that clock was not recorded. (The early commits contain the Windows version of `gpu_control.py`, `d05cbe6` the Linux one; the rule is the same.)
- `f109bd4` (2026-09-04): `DEFAULT_LOCK_MHZ = 200` (clamped to the supported range) for unset bounds. `f23f448` (2026-09-07): `DEFAULT_LOCK_MHZ = 2000` and explicit 2000/2000 defaults.
- So the statement of `README.md`/`CLAUDE.md` that the 15 runs are equivalent to 2000 MHz is not supported by the commit history (and, if the lock was applied at all, they ran at the lowest supported clock; whether the lock took effect for these hand-started runs is unknown).
**Development-run counts.** The results directory has 300 run directories: 280 canonical and 20 development runs. The 20: SAC HalfCheetah-v5 13 (seeds 0, 42, 56), SAC Humanoid-v5 2, MBPO HalfCheetah-v5 3 (R2000, R200, one seed-42 run), TD3 1, TD-MPC2 1. The 15 null-clock runs are exactly the 15 SAC runs; the other 5 record explicit values (2000/2000, one 200/200). The word "Sixteen" no longer occurs in `thesis-main.tex`; `README.md` says "15 unset runs" (consistent), and says 7 runs ran under torch 2.11.0, where the run files show 10 (seed 0: 7, seed 42: 3). One `\todo{AUTHOR: ...}` is kept in the appendix.
**Todos.** Deleted: the two Chapter 6 todos (old L6305, L6370). Kept in Chapter 6: the `NOT RUN` lock check and the `NOT RUN` hardware parameters.

## 7. Decisions delivered as variants (Appendix B, section "Optional Blocks"; each block has `\todo{AUTHOR: keep or delete this block}` and `% --- OPTIONAL BLOCK n ... ---` markers; Chapters 4 and 5 are unchanged)

1. **`tab:rob-tdmpc2` and `tab:rob-ranges`** (todo at the end of `sec:rob-tdmpc2`): Block 1 has `tab:app-rob-tdmpc2` and `tab:app-rob-ranges` (generated from `t1_tdmpc2_sweep.csv` and `t3_algo_ranges_vs_tdmpc2.csv`). If you move the tables, these places in `thesis-main.tex` change: the two table environments (L5862 to L5885 and L5900 to L5921 in the current file), the sentences `\Zcref{tab:rob-tdmpc2}` (L5861) and `\Zcref{tab:rob-ranges}` (L5899), the `\zcref{tab:rob-ranges}` in the summary (L5819), and the author todo (L5931); the label `tab:rob-tdmpc2`/`tab:rob-ranges` in the optional block's own text. Line numbers are those of the current file.
2. **`tab:td3-env` and `tab:tdmpc2-env`** (todo at the end of `sec:env-effect`): Block 2 has `tab:app-td3-env` (8 TD3 configurations) and `tab:app-tdmpc2-env` (5 TD-MPC2 configurations), format of `tab:sac-env`/`tab:mbpo-env`, from `env_comparison_paired.csv`; the SAC and MBPO tables at full precision are `tab:app-sac-env`, `tab:app-mbpo-env` in Section "Tables of the Per-Algorithm Chapter". All 27 pairs are covered (7 + 8 + 7 + 5). The todo at L6092 would be replaced by a reference.
3. **`tab:app-config-means` in D and B**: both locations now hold energy and FLOPs of the 56 configurations (D adds returns, B adds sd, durations, power and J/FLOP). Recommendation: keep it in D (cited by Chapter 3, holds the returns).
4. **TD3 sweep segment shares**: Block 3 has per-segment energy, share of total, share of GU (`tab:app-td3-sweep-energy`) and J/FLOP (`tab:app-td3-sweep-jpf`) for the 14 non-baseline TD3 configurations (7 per environment). If kept, the statements "not tabulated" in `sec:td3` and the corresponding sentence of Chapter 8 become false and need an edit (not done).

## 8. Build and static checks

Compile (scratch copy, `lualatex --shell-escape`, `biber`, `lualatex` twice; the files in `Thesis Report/` were not touched by it): **323 pages, 0 errors, 0 undefined references or citations in the last pass**, 1 LaTeX warnings (all pre-existing kinds such as `\@parboxrestore`; none from the appendices). The first pass of the first build stopped at 8 `siunitx` errors (exponent in front of `+-`), which are fixed in the generator.
Pages of the appendices in that build (first page of each chapter; total 323 pages): A p. 219, B p. 227, C p. 287, D p. 297, E p. 311, F p. 319; the appendix part has 104 pages up to the declaration page(s). Appendix B is the longest chapter (60 pages).

Overfull boxes in the appendix region (lines of the current file, 12 pt or more; the `10.9 pt` entries are the margin notes of `\todo`):
```
total labels with overfull boxes >= 12.0 pt: 0
errors: 0 []
undefined references/citations in log: 0 []
```

Static check of the file (comments stripped): `labels 365 duplicates []`; `reference targets 1023 undefined 1`; `unbalanced environments {} brace depth 0 negative 0`; undefined targets that were already undefined before: only the macro argument `#3` (false positive of the check).
All `app:`/`tab:app-`/`sec:app-`/`fig:app-` references are defined. The five labels cited before but undefined (`app:full-results`, `app:hyperparameters`, `app:flop-details`, `app:clock-details`, `app:clock-provenance`) are defined now; `app:learning`, `app:reproducibility`, `app:clock-suballoc`, `app:clock-idle`, `app:clock-full` as well.
Diff against the backup (`diff_check.py`): only the appendix region (old L7022 to L7247) is replaced, and exactly nine todo lines are deleted:
```
delete   old lines 2575-2575  ->  new lines 2575-2574   (1 old, 0 new)
     deleted: \todo{Add this reconciliation as a script to the repository and extend it to all signatures before c
delete   old lines 2577-2577  ->  new lines 2576-2575   (1 old, 0 new)
     deleted: \todo{Placeholder labels still undefined: \texttt{sec:compute-accounting} (Section 2.4) and \texttt{
delete   old lines 3825-3825  ->  new lines 3823-3822   (1 old, 0 new)
     deleted: \todo{Appendix~A must state that \texttt{model\_train\_freq} is \num{250} but the condition is evalu
delete   old lines 3941-3941  ->  new lines 3938-3937   (1 old, 0 new)
     deleted: \todo{Appendix~B should hold the composition and the idle-floor fractions of all \num{16} configurat
delete   old lines 5346-5346  ->  new lines 5342-5341   (1 old, 0 new)
     deleted: \todo{Scripts \texttt{flop\_analysis/section\_5\_2\_analysis.py} and \texttt{section\_5\_2\_render.p
delete   old lines 5811-5811  ->  new lines 5806-5805   (1 old, 0 new)
     deleted: \todo{Per-seed orderings and pairwise seed counts (\texttt{o3\_seed\_level.csv}, \texttt{o3b\_pairwi
delete   old lines 5952-5952  ->  new lines 5946-5945   (1 old, 0 new)
     deleted: \todo{Add the scripts \texttt{flop\_analysis/section\_5\_3\_analysis.py} and \texttt{section\_5\_3\_
delete   old lines 6305-6305  ->  new lines 6298-6297   (1 old, 0 new)
     deleted: \todo{Appendix \zcref{app:clock-provenance} (Provenance of the Two Runs) is a commented block at the
delete   old lines 6370-6370  ->  new lines 6362-6361   (1 old, 0 new)
     deleted: \todo{Appendix \zcref{app:clock-details} (Allocated Sub-Segments and Idle Windows) is a commented bl
replace  old lines 7022-7024  ->  new lines 7013-7860   (3 old, 848 new)
replace  old lines 7027-7028  ->  new lines 7863-7863   (2 old, 1 new)
replace  old lines 7030-7031  ->  new lines 7865-7866   (2 old, 2 new)
replace  old lines 7033-7040  ->  new lines 7868-7903   (8 old, 36 new)
... (further replaced fragments of the appendix region)
deleted lines: 9  lines in replaced/inserted regions: 4467  old lines 7255  new lines 11648
```
Todo counts of the current file: `{'other': 38, 'NOT RUN': 9, 'AUTHOR': 7} total todos 54`; the 9 `NOT RUN` and 7 `AUTHOR` todos are all listed in sections 10 and 11.


## 9. Repository state

`git rev-parse HEAD` (rl-dissections): `94e1de288e7aa8e3709568c66f3b136048961c90`; `git tag`: (none).
`git status --short` in `rl-dissections` (untracked scripts and files; the author commits them):
```
?? contexts/appendix/
?? contexts/section_2.md
?? contexts/section_2_data/
?? contexts/section_2_draft.tex
?? contexts/step1/
?? flop_analysis/appendix_tables.py
?? flop_analysis/appendix_tables_verify.py
?? flop_analysis/section_5_checks.py
?? flop_analysis/section_6_checks.py
?? flop_analysis/section_7_checks.py
?? flop_analysis/section_8_checks.py
?? flop_analysis/verify_analytic_flops.py
```
`git status --short` in `Thesis Report` (`thesis-main.tex` modified; `thesis-main.tex.bak-appendix` and the 16 figure files new):
```
D main-english-university-of-hamburg.tex
 M thesis-main.tex
?? figures/app-return-mbpo-ant.csv
?? figures/app-return-mbpo-ant.tex
?? figures/app-return-mbpo-hc.csv
?? figures/app-return-mbpo-hc.tex
?? figures/app-return-sac-ant.csv
?? figures/app-return-sac-ant.tex
?? figures/app-return-sac-hc.csv
?? figures/app-return-sac-hc.tex
?? figures/app-return-td3-ant.csv
?? figures/app-return-td3-ant.tex
?? figures/app-return-td3-hc.csv
?? figures/app-return-td3-hc.tex
?? figures/app-return-tdmpc2-ant.csv
?? figures/app-return-tdmpc2-ant.tex
?? figures/app-return-tdmpc2-hc.csv
?? figures/app-return-tdmpc2-hc.tex
?? thesis-main.tex.bak-appendix
?? todo-appendix.md
```

## 10. Linux machine list (to run later by the author)

All commands are collected, with the expected output and the place it fills, in `contexts/appendix/research_pc_checklist.md` (line numbers of `thesis-main.tex` after the insertion). Summary (15 items): pip freeze and package versions (`tab:thr-provenance` row 1, `tab:software`); `nvidia-smi`, driver, VBIOS, open kernel modules (row 2); `nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS` (row 3, Chapter 6 `NOT RUN`); `torch.cuda.get_device_properties(0)` (row 4); `lscpu`, `lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN`, `sudo dmidecode -t memory -t baseboard`, mainboard (row 5); `cat /etc/os-release` (distribution row); `ls -d /sys/class/powercap/intel-rapl*` and the CodeCarbon start-up line (`NOT RUN` in 7.1.1 and Section 3.4.6); TF32 check (`NOT RUN` in 7.2, Section 4.1); clock lock check (`NOT RUN`, needs `sudo`); random-policy returns, command F3 (Appendix D `NOT RUN`, Sections 4.1 and 4.4); `nvml_poll.py`; `sync_timing_run.py`; optional rerun of `verify_mbpo_fit_flops.py`; the "fill from the research PC" cell of `tab:thr-summary`.

| `NOT RUN` todo | Location (`thesis-main.tex`, current) | Command (checklist item) |
|---|---|---|
| lock check with `nvidia-smi` (Chapter 6 limitations) | L6426 | item 9 (needs sudo) |
| clock range, power limit and driver version (Chapter 6) | L6429 | items 2, 3 |
| confirm the RAPL mode (7.1.1) | L6475 | item 7 |
| poll the NVML counter at about 1 kHz (7.1.1) | L6495 | item 11 |
| `torch.get_float32_matmul_precision()`, `allow_tf32` (7.2) | L6759 | item 8 |
| fill from the research PC (`tab:thr-summary`) | L6809 | item 13 |
| rerun `verify_mbpo_fit_flops.py` (Appendix C) | L10868 | item 14 (any machine with PyTorch) |
| random-policy return, command F3 (Appendix D) | L11247 | item 10 |
| `pip freeze` in `rl-exp` (Appendix E) | L11447 | items 1, 15 |

`grep -c "NOT RUN"` of the todos gives 9 (list above); the 5 research-PC rows of `tab:thr-provenance` and 7 further research-PC todos of Chapters 3, 4 and 7 (no `NOT RUN` label) are covered by checklist items 1 to 9 and 12.

## 11. Author-decision and manual list

- Decisions of section 7 (four optional blocks); whether Appendix F stays a chapter; whether the optional Spearman block of D is wanted (it contradicts the sentence "not computed per algorithm" in `sec:metric` if kept); title case of `Learning check details`/`Reproducibility details`; length of Appendix B.
- Repository: `LICENSE`, commit of the untracked scripts (list in `tab:app-repro-scripts`), tag of the cited commit, cite the tag in Appendix E and the contribution list; commit `verify_analytic_flops.py` and cite it in `sec:flop-verification`, Sections 4.1 to 4.4 and 8.2 (the five existing todos stay); `README.md`/`CLAUDE.md` statement on the 15 unset clock fields (and the "7 runs" count).
- Proposed edits not applied (outside the insertion points): see `edit_log.md`, "Proposed, not applied" (device of the FLOP measurement, 4.8e-13 figure, one rounding case in `tab:tdmpc2-base-energy`, title case).
- `AUTHOR` todos in the file: 7 (list below).

- L10822 (Appendix C): commit `verify_analytic_flops.py` and cite the commit hash.
- L10133, L10188, L10249 (Appendix B): the three optional blocks (section 7).
- L11209 (Appendix D): optional Spearman block.
- L11529 (Appendix E): contribution 5 (LICENSE, commit of the untracked scripts, tag).
- L11560 (Appendix F): description of the unset clock fields of the fifteen development runs; correct README.md/CLAUDE.md.
