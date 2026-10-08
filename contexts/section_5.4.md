# Section 5.4 — Environment effect, quantified — context

Data context for thesis Section 5.4 (HalfCheetah-v5 vs. Ant-v5, all four algorithms). Facts, numbers and provenance only — no interpretation, no LaTeX prose. Canonical seeds [331, 958, 14577, 43611, 85062] only (dev runs ignored); 280 runs; **gross** energies (idle floor included); joules. HC = HalfCheetah-v5, Ant = Ant-v5; every ratio/difference is **Ant relative to HC**, runs paired by seed number. Labelling rule: anything not found/computed/verified is written `NOT FOUND: <what, where looked>`; anything computed here and not by a repo script is tagged `DERIVED (not a repo script)`. Four significant digits here; full precision in the CSVs.

Generated 2026-10-08 from git HEAD `2b4882f` (branch master); Windows dev checkout with synced `results/`. Nothing under `results/`, `flop_analysis/output/`, or the thesis `.tex`/`.bib` was modified; the thesis files were only read.

**Files written** (all under `contexts/`): this file; `contexts/section_5.4_data/build_section_5_4.py` (computes everything → CSVs + `findings.json`; **new script**, `DERIVED (not a repo script)` in the sense that it is not one of the Chapter-4 generators, but its definitions are copied from them — see §2 of this file and the two cross-checks); `contexts/section_5.4_data/render_section_5_4.py` (renders this file). Re-run: `.venv/Scripts/python -I -W ignore contexts/section_5.4_data/build_section_5_4.py` then `.venv/Scripts/python -I -W ignore contexts/section_5.4_data/render_section_5_4.py`.

## 1. Thesis state

**Source files.** The thesis lives in one file: `C:\Users\saman\Desktop\Thesis\Thesis Report\thesis-main.tex` (6221 lines; chapters are inline, no `\input`/`\include` of chapter files). `C:\Users\saman\Desktop\Thesis\Thesis Report\bibliography.bib` is the bibliography; `commands.tex`, `abbreviations.tex` are small helper inputs. `main-english.tex` and `main-english-university-of-hamburg.tex` in the same folder are other (shorter, ~2100-line) documents; a search for `ch:comparison` / `sec:env-effect` in `main-english.tex` finds nothing, so they hold no Chapter 5.

**Chapter 5**: `\chapter{Cross-Algorithm Comparison}\label{ch:comparison}` at `thesis-main.tex:5412`; the chapter runs to line 6212 (followed by `\chapter{My first appendix}` at line 6213). Chapter 5 intro is written (lines 5412–5429) and already announces Section 5.4 ("\Zcref{sec:env-effect} compares the two environments with the definitions of \zcref{sec:rqs} for all four algorithms (RQ3)", line 5419).

**Sections 5.1–5.3 and 5.4**

| label | title | lines | non-comment non-blank lines | # subsections | state |
|---|---|---|---|---|---|
| sec:segment-shares | Segment Shares Across Algorithms | 5434–5550 | 102 | 0 | written (non-empty) |
| sec:energy-vs-compute | Energy versus Compute: Are FLOP-Based Energy Estimates Trustworthy? | 5551–5900 | 302 | 4 | written (non-empty) |
| sec:robustness | Robustness of the Comparison Across Sweeps | 5901–6212 | 246 | 6 | written (non-empty) |
| `sec:env-effect` | (Section 5.4 — does not exist) | — | 0 | 0 | NOT FOUND: no `\section` / `\label{sec:env-effect}` definition in thesis-main.tex (label only *referenced*) |

**Labels** (definition line in `thesis-main.tex`):

| label | definition | references |
|---|---|---|
| `ch:comparison` | line 5412 | 4 reference(s), e.g. lines 2242, 3135, 4863, 5356 |
| `sec:segment-shares` | line 5434 | 3 reference(s), e.g. lines 5416, 5428, 5903 |
| `sec:energy-vs-compute` | line 5551 | 18 reference(s), e.g. lines 2186, 2248, 2249, 2782, 2829 |
| `sec:robustness` | line 5901 | 3 reference(s), e.g. lines 5418, 5428, 5438 |
| `sec:rob-basis` | line 5917 | 2 reference(s), e.g. lines 5909, 6183 |
| `sec:rob-energy` | line 5956 | 2 reference(s), e.g. lines 5910, 6070 |
| `sec:rob-jpf` | line 6002 | 2 reference(s), e.g. lines 5910, 6070 |
| `sec:rob-rollout` | line 6067 | 1 reference(s), e.g. lines 5911 |
| `sec:rob-tdmpc2` | line 6098 | 2 reference(s), e.g. lines 5911, 5920 |
| `sec:rob-summary` | line 6173 | 1 reference(s), e.g. lines 5912 |
| `sec:env-effect` | NOT FOUND (no `\label` definition) | 5 reference(s), e.g. lines 4667, 5419, 5428, 5543, 5937 |
| `sec:environments` | line 2843 | 2 reference(s), e.g. lines 2839, 4812 |
| `tab:envs` | line 2859 | 1 reference(s), e.g. lines 2845 |
| `tab:sac-env` | line 3631 | 1 reference(s), e.g. lines 3617 |
| `tab:mbpo-env` | line 4546 | 1 reference(s), e.g. lines 4530 |
| `tab:td3-env` | NOT FOUND (no `\label` definition) | 0 reference(s) |
| `tab:tdmpc2-env` | NOT FOUND (no `\label` definition) | 0 reference(s) |
| `sec:rqs` | line 2192 | 6 reference(s), e.g. lines 2093, 5419, 5440, 5906, 5947 |
| `sec:seeds` | line 2874 | 8 reference(s), e.g. lines 2196, 2252, 2839, 2942, 3164 |
| `sec:sweeps` | line 2937 | 16 reference(s), e.g. lines 2138, 2251, 2841, 3068, 3125 |
| `sec:baselines` | line 2887 | 8 reference(s), e.g. lines 2257, 2840, 3734, 4053, 5018 |
| `sec:threats` | NOT FOUND (no `\label` definition) | 22 reference(s), e.g. lines 2196, 2266, 2782, 2829, 2872 |
| `ch:evaluation` | NOT FOUND (no `\label` definition) | 7 reference(s), e.g. lines 2265, 2266, 5427, 5428, 5894 |

Any `sec:env-*` label **defined**: only `sec:environments` (the "Environments" subsection of the experimental design, line 2843 — about the environments, not the comparison). `sec:env-effect` is referenced at lines 4667, 5419, 5428, 5543, 5937 but never defined (placeholder, as the todo at the end of the MBPO rollout subsection says). `sec:threats` and `ch:evaluation` are likewise referenced but not defined (see the table).

**Bib keys** (`bibliography.bib`):

| topic | key | line(s) of definition |
|---|---|---|
| MuJoCo | `todorov2012mujoco` | 159 |
| Gymnasium | `towers2024gymnasium` | 152 |
| CodeCarbon | `codecarbon` | 78 |
| SAC | `haarnoja2018sac` | 99, 211, 279, 344, 394, 554 |
| TD3 | `fujimoto2018td3` | 107, 222, 265, 405, 566 |
| MBPO | `janner2019mbpo` | 116, 233, 323, 416, 577 |
| TD-MPC2 | `hansen2024tdmpc2` | 130, 249, 356, 425, 585 |

Other keys present that 5.4 may reuse (key present = yes): `chua2018pets`, `williams2017mppi`, `hansen2022tdmpc`, `schulman2016gae`, `wawrzynski2009cat`, `henderson2018matters`, `colas2018seeds`, `agarwal2021precipice`, `henderson2020systematic`, `strubell2019energy`, `schwartz2020green`, `patterson2021carbon`, `desislavov2023trends`, `getzner2023accuracy`, `khan2018rapl`, `nvml`, `pytorchflopcounter`, `paszke2019pytorch`, `virtanen2020scipy`.

**Duplicate keys in `bibliography.bib`**: `chua2018pets` (lines 123, 241, 333), `fujimoto2018td3` (lines 107, 222, 265, 405, 566), `haarnoja2018sac` (lines 99, 211, 279, 344, 394, 554), `hansen2024tdmpc2` (lines 130, 249, 356, 425, 585), `henderson2018matters` (lines 176, 533), `janner2019mbpo` (lines 116, 233, 323, 416, 577), `williams2017mppi` (lines 136, 256, 378, 432). (Observation only; `bibliography.bib` was not modified.)

**Citation about environment-dependent energy or simulator cost**: NOT FOUND — no key in `bibliography.bib` whose title/fields concern simulator cost or environment-dependent energy (searched the file for `simulat`, `energy`: only `strubell2019energy`, `desislavov2023trends`, `getzner2023accuracy`, `henderson2020systematic`, which are general ML-energy references).

**Macros / packages in use** (counts over `thesis-main.tex`):

| macro / package | count | second count |
|---|---|---|
| `\zcref{...}` / `\Zcref{...}` (zref-clever; `\Zcref` at sentence start) | 504 | 114 |
| `\todo{...}` (defined line 584: `TODO!` + `\sidecomment`/`pdfcomment`) | 94 | — |
| siunitx: `\num{}` (also `\num{x+-y}`), `\SI{}{\percent}`, `\SI{}{\joule}` etc. (loaded line 611 with `group-minimum-digits=4,per-mode=fraction`; the thesis writes `\SI{2.084}{\percent}-points`) | 3841 | 1776 |
| booktabs: `\toprule \midrule \bottomrule`; `\cmidrule(lr){a-b}` | 70 | cmidrule 40 |
| `\resizebox{\textwidth}{!}{%` around `tabular` (tables `[tbp]`, `\centering`, `\footnotesize`, `@{}` column ends) | 41 | — |
| `\paragraph{Name.}` blocks for sub-results; math `$\bar P_r/\bar P_{\mathrm{GU}}$`, `$|\Delta$share$|$` | 176 | — |
| `multirow` package is loaded (line 494) but `\multirow{` is used 0 times; `makecell`, `tabularx`, `cellcolor`, `rowcolors`: not used | 0 | — |

**Exact LaTeX of `tab:sac-env`** (`thesis-main.tex:3621–3646`; the table that 5.4 should match; `tab:mbpo-env` has the identical column structure at the line given above):

```latex
\begin{table}[tbp]
  \centering
  \footnotesize
  \caption{Environment comparison of all SAC configurations (Ant-v5 relative to
    HalfCheetah-v5, runs paired by seed). Energy ratios are ratios of means.
    $\Delta$share is the difference of the mean shares in percentage points; the
    segment that attains the maximum absolute difference is given in brackets.
    $\bar P_r/\bar P_{\mathrm{GU}}$ is the ratio of the mean power of the rollout
    to that of the GU; the column gives the ratio of this quantity on Ant-v5 to
    that on HalfCheetah-v5.}
  \label{tab:sac-env}
  \resizebox{\textwidth}{!}{%
    \begin{tabular}{@{}lrrrrrr@{}}
      \toprule
      Config.    & $E_{\mathrm{total}}$ ratio & $E_{\mathrm{rollout}}$ ratio & max $|\Delta$share$|$ [pp] & $\bar P_{\mathrm{total}}$ ratio & $\bar P_r/\bar P_{\mathrm{GU}}$ ratio & J/FLOP total ratio \\
      \midrule
      Baseline   & \num{1.057}                & \num{1.736}                  & \num{2.631} (rollout)      & \num{1.001}                     & \num{0.9803}                          & \num{0.9873}       \\
      UTD 2      & \num{1.044}                & \num{1.714}                  & \num{1.368} (rollout)      & \num{1.005}                     & \num{0.9557}                          & \num{0.9749}       \\
      UTD 4      & \num{1.027}                & \num{1.682}                  & \num{0.9364} (critic)      & \num{1.004}                     & \num{0.9388}                          & \num{0.9594}       \\
      Width 256  & \num{1.054}                & \num{1.800}                  & \num{3.399} (rollout)      & \num{0.9999}                    & \num{1.004}                           & \num{0.8329}       \\
      Width 512  & \num{1.053}                & \num{1.785}                  & \num{3.263} (rollout)      & \num{1.004}                     & \num{0.9931}                          & \num{0.9254}       \\
      Batch 512  & \num{1.048}                & \num{1.650}                  & \num{2.084} (rollout)      & \num{1.000}                     & \num{0.9376}                          & \num{0.9785}       \\
      Batch 1024 & \num{1.081}                & \num{1.652}                  & \num{2.068} (actor)        & \num{1.008}                     & \num{0.9279}                          & \num{1.010}        \\
      \bottomrule
    \end{tabular}}
\end{table}
```

**Chapter 5 opening** (`thesis-main.tex:5412–5429`):

```latex
\chapter{Cross-Algorithm Comparison}\label{ch:comparison}

\Zcref{ch:per-algorithm} reports each algorithm on its own.
This chapter puts SAC~\cite{haarnoja2018sac}, TD3~\cite{fujimoto2018td3}, MBPO~\cite{janner2019mbpo} and TD-MPC2~\cite{hansen2024tdmpc2} side by side and answers the first three research questions across algorithms (\zcref{tab:rqs}).
\Zcref{sec:segment-shares} compares how the training energy is distributed over the segments at the baselines (RQ1).
\Zcref{sec:energy-vs-compute} asks how well the computation performed predicts the energy measured, by a rank correlation and a log--log fit over the configurations and by a split of the energy into a fixed and a marginal part (RQ2).
\Zcref{sec:robustness} reports the ordering of the algorithms along the sweeps and the pairs that change places (RQ3).
\Zcref{sec:env-effect} compares the two environments with the definitions of \zcref{sec:rqs} for all four algorithms (RQ3).

All values come from the \num{280} canonical runs with the five seeds of \zcref{sec:seeds}, and all energies are gross (\zcref{sec:units}).
Segments are marked as measured or allocated according to \zcref{sec:measured-allocated}.
Three limits of the design apply throughout the chapter.
First, TD-MPC2 takes part only in the sweeps of its own structure, so comparisons along the update-to-data, width and batch-size sweeps contain SAC, TD3 and MBPO only (\zcref{sec:sweeps}).
Second, the TD3 baseline uses batch size \num{100} and not \num{256}, so the batch-matched comparison with SAC and MBPO uses the TD3 runs with batch size \num{256} (\zcref{sec:baselines}).
Third, segments with the same name do not stand for the same computation in all algorithms (\zcref{tab:segment-meaning}).
In line with the argument structure of the thesis, this chapter describes the comparison, and its interpretation follows in \zcref{ch:evaluation}.
\todo{Confirm that the interpretation of Chapter~5 is collected in Section 7.5 and not given in the chapter itself. Placeholder label: \texttt{ch:evaluation} (Chapter 7). Labels \texttt{sec:segment-shares}, \texttt{sec:robustness} and \texttt{sec:env-effect} are the labels of Sections 5.1, 5.3 and 5.4; \texttt{sec:energy-vs-compute} is the label already used for Section 5.2.}

```

**Section 5.3 opening** (`thesis-main.tex:5901–5912`, a section-level opening for style):

```latex
\section{Robustness of the Comparison Across Sweeps}\label{sec:robustness}

This section answers the third research question for the comparison of the algorithms: does the order of the algorithms found at the baselines (\zcref{sec:segment-shares}) survive a change of the update-to-data ratio, the network width or the batch size?
The question matters because rankings of reinforcement-learning algorithms are known to depend on implementation details, on hyperparameters and on the number of seeds~\cite{henderson2018matters,agarwal2021precipice}.
A comparison that holds at one setting only would say little about the algorithms themselves.
Two quantities are ordered, as defined in \zcref{sec:rqs}: the energy of \texttt{TOTAL\_MEASURED\_TRAINING}, and the energy of the same total per matmul FLOP (J/FLOP, \zcref{eq:jpf-total}).
For each value of each sweep, the algorithms are ordered by both quantities, the pairs that change places relative to the baseline are listed, and the order of the configuration means is compared with the order found in the individual seeds.

\Zcref{sec:rob-basis} states how the orderings are formed and which runs enter them.
\Zcref{sec:rob-energy} and \zcref{sec:rob-jpf} report the orderings by energy and by J/FLOP.
\Zcref{sec:rob-rollout} reports the sweep of the maximum rollout length of MBPO, and \zcref{sec:rob-tdmpc2} the sweeps of TD-MPC2 and the comparison of its range with that of the other algorithms.
\Zcref{sec:rob-summary} summarises the answer to RQ3 for the comparison of the algorithms.
```

Related thesis facts 5.4 must be consistent with (verbatim from `thesis-main.tex`): RQ3 part (iii) at line None ("The two environments are compared by the difference of the shares of each segment and by the ratio of their energies, with the per-seed pairing of \zcref{sec:seeds}. The paired differences are assessed with an exact sign-flip test, which cannot go below $p = \num{0.0625}$ with five pairs, so the sign of a difference is stated but no stronger claim is made."); environments table `tab:envs` (line 2859) lists Ant-v5 observation dimension **105**, action dimension **8**, HalfCheetah-v5 **17**/**6** and carries an open `\todo` to check the Ant-v5 observation dimension (resolved in §4.3 below); the todo at the end of the TD-MPC2 chapter (line 5406) says the termination classifier confounds the environment comparison; the todo at the MBPO rollout-length subsection (line 4667) asks whether Ant `L_max = 1` is the comparator (§4.1 below). The SAC/TD3/MBPO/TD-MPC2 baseline sentences already in the thesis: `Paired by seed, the total energy of Ant-v5 is \num{1.057} / \num{1.086} / \num{1.221} / \num{1.103} times that of HalfCheetah-v5 (sd of the paired ratios \num{0.01179} / \num{0.01747} / \num{0.1145} / \num{0.006616})`. Tables with this content exist for SAC (`tab:sac-env`) and MBPO (`tab:mbpo-env`) only; TD3 and TD-MPC2: NOT FOUND (no `tab:td3-env`/`tab:tdmpc2-env`; their environment comparison is in the prose paragraphs).

## 2. Method and conventions in use

**Which statistic is "E_total ratio" in the existing tables?** **Ratio of means** — `mean(E_Ant over 5 seeds) / mean(E_HC over 5 seeds)`; the captions of `tab:sac-env` and `tab:mbpo-env` say "Energy ratios are ratios of means", and the code (`contexts/section_4.1_data/build_section_4_1.py` part5, 5.4 block; `contexts/Section_4.2_data/s4_blocks.py` `env_comparison_block`) computes `np.mean(E_ant)/np.mean(E_hc)`. The "(sd of the paired ratios …)" in the per-algorithm text is the sample sd (ddof = 1) of the five **per-seed** ratios `E_Ant,s / E_HC,s`; the headline number in the sentence ("is 1.057 times") is the ratio of means, **not** the mean of the paired ratios. Side by side (baselines):

| algo | ratio of means (thesis number) | mean of 5 paired ratios | sd of paired ratios (thesis `sd`) | number printed in the thesis |
|---|---|---|---|---|
| SAC | 1.057 | 1.057 | 0.01179 | 1.057 |
| TD3 | 1.086 | 1.086 | 0.01747 | 1.086 |
| MBPO | 1.221 | 1.222 | 0.1145 | 1.221 |
| TD-MPC2 | 1.103 | 1.103 | 0.006616 | 1.103 |

Source: `env_comparison_paired.csv` columns `E_total_ratio_of_means`, `E_total_paired_mean`, `E_total_paired_sd`. For the power ratio and the J/FLOP ratio the convention is also ratio of means (J/FLOP ratio = (mean E_Ant / mean F_Ant) / (mean E_HC / mean F_HC); the (P_rollout/P_GU) column is the double ratio built from the env means; the paired versions are in the CSV).

**Share differences — per seed first, then averaged?** Yes. For each seed the share is `100 · E_segment / E_TOTAL_MEASURED_TRAINING` of that run, the paired difference is `share_Ant,s − share_HC,s` (pp), then mean and sd (ddof = 1) over the 5 differences; "same sign" = number of the 5 differences whose sign equals the sign of their mean (build_section_4_1.py part5 5.2 counts `> 0`, s4_blocks.py counts `sign == sign(mean)`; identical outcome whenever all 5 agree). `max |Δshare|` is `|mean of paired differences|` (equal to the difference of the mean shares) maximised over the measured non-GU segments **and the four allocated GU sub-segments**; `gradient_updates` itself is excluded because it is the sum of its sub-segments (this is why "critic"/"actor" can attain the maximum in `tab:sac-env`). Allocated sub-segments are included in the CSV (`dshare_<segment>_*`).

**Sign-flip p-values — script/output.** Test: exact two-sided paired sign-flip permutation test, statistic |mean of the 5 paired differences (Ant − HC)|, all 2⁵ = 32 sign patterns, p = #{|mean| ≥ observed}/32 (tolerance 1e-12). Implemented as `signflip_p()` in `contexts/section_4.1_data/build_section_4_1.py:1416` (SAC; output `contexts/section_4.1_data/sac_env_comparison.csv`, metrics `signflip_p_two_sided::rollout share [pp]` and `::TOTAL energy [J]`) and verbatim in `contexts/Section_4.2_data/s4_data.py:169` (used by `s4_blocks.env_comparison_block`, line ~342, for TD3 / MBPO / TD-MPC2; output only inside `contexts/Section_4.2.md`, `Section_4.3.md`, `Section_4.4.md` — no CSV export of those env comparisons: NOT FOUND). Copied verbatim into `contexts/section_5.4_data/build_section_5_4.py` for this section. Tested quantities: (a) the paired difference of the rollout **share** (pp), (b) the paired difference of the **total energy** (J).

**Which script produced `tab:sac-env` / `tab:mbpo-env`?** SAC: `contexts/section_4.1_data/build_section_4_1.py` (part5) → `contexts/section_4.1.md` / `sac_env_comparison.csv`. MBPO (and TD3, TD-MPC2): `contexts/Section_4.2_data/s4_blocks.py::env_comparison_block`, invoked by `build_section_4_2.py` / `build_section_4_3.py` / `build_section_4_4.py` → `Section_4.2.md` / `Section_4.3.md` / `Section_4.4.md`. Those scripts load `results/` themselves; the 27-row table below is produced by the **new** `build_section_5_4.py`, which re-implements the same definitions on top of `flop_analysis/output/per_run_energy_per_flop.csv` and `flop_analysis/section_5_2_analysis.parse()` (the config labelling used in 5.2/5.3). Two independent cross-checks (below) show it reproduces the earlier numbers.

**Cross-checks (all passed)**

- **vs. thesis draft tables** (`x1_crosscheck_vs_thesis.csv`): every cell of `tab:sac-env` and `tab:mbpo-env` (7 configs × 6 columns + segment label each, parsed directly from `thesis-main.tex`), plus the baseline E_total ratio and paired sd of SAC/TD3/MBPO/TD-MPC2 from the thesis text, plus the instruction's TD3 numbers: **107/107 reproduced** at the printed 4 digits. Specifically: baseline E_total ratio SAC 1.057 ✓, MBPO 1.221 ✓, TD3 1.086 ✓; max|Δshare| SAC 2.631 pp (rollout) ✓, MBPO 3.818 pp (fit) ✓, TD3 5.045 pp (rollout) ✓. **One cell is a rounding artefact, not a mismatch**: MBPO baseline (P_rollout/P_GU) ratio — computed 0.9731496, the repo context file prints 0.973150 (6 digits) and the thesis prints 0.9732 (the 6-digit value rounded half-up); a direct 4-digit rounding would give 0.9731.
- **vs. the repo's own Chapter-4 context files** (`x2_crosscheck_vs_contexts_4x.csv`): the env-comparison tables of `Section_4.2.md` (TD3, 8 configs), `Section_4.3.md` (MBPO, 7), `Section_4.4.md` (TD-MPC2, 5) — 6 columns + segment label each: **203/203 reproduced** (max relative deviation 4.9e-06, i.e. 6-digit rounding). TD3 and TD-MPC2 numbers in this file are therefore identical to the Chapter-4 contexts.

**A pitfall found while reproducing the power column** (`DERIVED (not a repo script)` finding): in `flop_analysis/output/per_run_energy_per_flop.csv` the four GU sub-segment rows carry `duration_s` = the allocated `perf_counter` time and `mean_power_w` = E_sub/that time (e.g. 150.4 W), **not** the power of the measured `gradient_updates` task (145.4 W for the same SAC-Ant run). The thesis' (P_rollout/P_GU) uses E_GU / (sum of the `gradient_updates_<i>` task durations), so `build_section_5_4.py` reads the task durations from `emissions_*.csv`. Using the sub-row power instead shifts the double ratio by up to ~0.9 % (SAC batch 1024: 0.9279 vs 0.9194).

**Existing figures that could be reused** (paths only; none is a static plot of per-env segment shares → NOT FOUND for a ready-made per-env segment-share figure):

- `dashboard.py` (Dash app; per-run phase duration/power/energy bars — one run at a time)
- `flop_dashboard.py` (Dash app over `flop_analysis/output/*.csv`; `env_id` is a pivot dimension, but the y-axes are energy / J-per-FLOP / ΔE/ΔF, not shares)
- `flop_analysis/output/correlation/correlation_plots.html` (return/energy vs FLOPs per env)
- `idle_power_report.html` (idle-baseline drift)
- no `.png`/`.svg`/`.pdf` figure files exist in the repo outside `.venv`.

## 3. Paired comparison table references and per-algorithm summaries

**CSV (the §2 deliverable): `contexts/section_5.4_data/env_comparison_paired.csv`** — 27 rows = 27 (algorithm, configuration) pairs present on both environments: SAC 7, TD3 8, MBPO 7, TD-MPC2 5 (count check ✓; Ant-only MBPO `rollout_max_length` 1 / 15 are not among them, see §4.1). Columns: `algo, config, config_tag`; `E_total_ratio_of_means`, `E_total_paired_mean`, `E_total_paired_sd`; same three for `E_rollout`; for every measured segment of the algorithm and the four allocated GU sub-segments `dshare_<segment>_mean_pp / _sd_pp / _same_sign_k_of_5`; `max_abs_dshare_pp, max_abs_dshare_segment`; `signflip_p_rollout_share, signflip_patterns_rollout_share, signflip_n_positive_rollout_share`, same for `total_energy`; `P_total_ratio_of_means` (+ paired mean/sd); `Proll_over_PGU_HC / _Ant / double_ratio_Ant_over_HC` (+ paired mean/sd); `jpf_total_HC/Ant_J_per_FLOP, jpf_total_ratio_Ant_over_HC`; `flops_total_ratio_Ant_over_HC` (+ the two FLOP totals); `tag_td3_batch_matched_to_sac_mbpo`; `algo_config_diff_HC_vs_Ant`, `experiment_config_diff_HC_vs_Ant`, `env_vs_config_confounded`, `git_commits_HC/Ant`.

**TD3 matching flag**: TD3 rows `baseline, UTD 2, UTD 4, width 256, width 512` are TD3's own batch-100 runs (warmup 10000) — flagged *unmatched* to SAC/MBPO (batch 256); the TD3 batch rows use `b256 / b512 / b1024` and are the matched points (column `tag_td3_batch_matched_to_sac_mbpo`). HC vs. Ant within TD3 is always the same configuration on both sides.

**Env-vs-config confound flag** (column `env_vs_config_confounded`, from the logged `algo_config` of the HC and Ant run groups — each group has an identical `algo_config` over its 5 seeds, asserted): SAC 0/7 configs differ between HC and Ant, TD3 0/8, **MBPO 7/7** (`rollout_max_length` 1 vs 25 and `rollout_max_epoch` 150 vs 100), **TD-MPC2 5/5** (`episodic` false vs true). `experiment_config` (protocol: warmup, steps, clocks, …) is identical between HC and Ant in all pairs except the SAC baseline, where the only difference is the logged `thermal_gate_reference_file` field (`null` on some runs vs `results/_thermal_reference.json`; irrelevant to the measurement). Git commits of the compared groups differ in several pairs (`git_commits_HC` vs `git_commits_Ant`: SAC utd2/utd4 `ecb98bb` vs `aa9186b`, SAC/TD3/MBPO batch rows `9b4b44c` vs `788ffcd`, SAC baseline Ant has two commits `92dcee0,f23f448`) — i.e. HC and Ant runs of the same configuration were launched by different sweep scripts at different times for those pairs.

**Sign-flip result in one line**: in all 27 pairs the exact two-sided p returned is **0.0625** for the rollout share **and** for the total energy (the minimum possible with 5 pairs); the rollout-share difference is positive (Ant > HC) in all 5 seeds in all 27 pairs; the total energy is higher on Ant in all 5 seeds in all 27 pairs.

### 3.1 SAC — ratios, max |Δshare|, p, power, J/FLOP

| config | E_total ratio (of means) | E_total paired mean ± sd | E_rollout ratio (of means) | E_rollout paired mean ± sd | max abs Δshare pp (segment) | p rollout share / p total | P_total ratio | (P_roll/P_GU) Ant/HC | J/FLOP total ratio | FLOPs total ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.057 | 1.057 ± 0.01179 | 1.736 | 1.736 ± 0.01307 | 2.631 (rollout) | 0.0625 / 0.0625 | 1.001 | 0.9803 | 0.9873 | 1.071 |
| UTD 2 | 1.044 | 1.044 ± 0.006636 | 1.714 | 1.714 ± 0.02001 | 1.368 (rollout) | 0.0625 / 0.0625 | 1.005 | 0.9557 | 0.9749 | 1.071 |
| UTD 4 | 1.027 | 1.027 ± 0.004492 | 1.682 | 1.682 ± 0.0215 | 0.9364 (critic) | 0.0625 / 0.0625 | 1.004 | 0.9388 | 0.9594 | 1.071 |
| width 256 | 1.054 | 1.054 ± 0.00694 | 1.8 | 1.8 ± 0.02392 | 3.399 (rollout) | 0.0625 / 0.0625 | 0.9999 | 1.004 | 0.8329 | 1.266 |
| width 512 | 1.053 | 1.054 ± 0.01735 | 1.785 | 1.785 ± 0.02044 | 3.263 (rollout) | 0.0625 / 0.0625 | 1.004 | 0.9931 | 0.9254 | 1.138 |
| batch 512 | 1.048 | 1.048 ± 0.01077 | 1.65 | 1.65 ± 0.0254 | 2.084 (rollout) | 0.0625 / 0.0625 | 1 | 0.9376 | 0.9785 | 1.071 |
| batch 1024 | 1.081 | 1.081 ± 0.004123 | 1.652 | 1.653 ± 0.02809 | 2.068 (actor) | 0.0625 / 0.0625 | 1.008 | 0.9279 | 1.01 | 1.071 |

Per-segment share difference Ant − HC [pp], mean ± sd of the 5 paired differences, (k of 5 seeds with the same sign as the mean). Measured segments: `rollout`, `gradient_updates`; allocated GU sub-segments are in the CSV.

| config | rollout | GU |
|---|---|---|
| baseline | 2.631 ± 0.02911 (5/5) | -2.631 ± 0.02911 (5/5) |
| UTD 2 | 1.368 ± 0.03395 (5/5) | -1.368 ± 0.03395 (5/5) |
| UTD 4 | 0.6926 ± 0.01741 (5/5) | -0.6926 ± 0.01741 (5/5) |
| width 256 | 3.399 ± 0.09963 (5/5) | -3.399 ± 0.09963 (5/5) |
| width 512 | 3.263 ± 0.1608 (5/5) | -3.263 ± 0.1608 (5/5) |
| batch 512 | 2.084 ± 0.09889 (5/5) | -2.084 ± 0.09889 (5/5) |
| batch 1024 | 1.762 ± 0.0644 (5/5) | -1.762 ± 0.0644 (5/5) |

### 3.2 TD3 — ratios, max |Δshare|, p, power, J/FLOP

| config | E_total ratio (of means) | E_total paired mean ± sd | E_rollout ratio (of means) | E_rollout paired mean ± sd | max abs Δshare pp (segment) | p rollout share / p total | P_total ratio | (P_roll/P_GU) Ant/HC | J/FLOP total ratio | FLOPs total ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline* | 1.086 | 1.086 ± 0.01747 | 2.177 | 2.177 ± 0.04665 | 5.045 (rollout) | 0.0625 / 0.0625 | 1.003 | 0.9675 | 1.017 | 1.068 |
| UTD 2* | 1.039 | 1.039 ± 0.006237 | 2.117 | 2.118 ± 0.07682 | 2.682 (rollout) | 0.0625 / 0.0625 | 1.004 | 0.9437 | 0.9736 | 1.068 |
| UTD 4* | 1.042 | 1.042 ± 0.005308 | 2.184 | 2.184 ± 0.04093 | 1.409 (rollout) | 0.0625 / 0.0625 | 1.006 | 0.9682 | 0.9764 | 1.068 |
| width 256* | 1.09 | 1.09 ± 0.04891 | 2.288 | 2.289 ± 0.08686 | 5.922 (rollout) | 0.0625 / 0.0625 | 0.9938 | 0.9934 | 0.8672 | 1.256 |
| width 512* | 1.083 | 1.083 ± 0.01385 | 2.226 | 2.226 ± 0.05899 | 5.416 (rollout) | 0.0625 / 0.0625 | 0.9854 | 0.9857 | 0.9559 | 1.133 |
| batch 256 | 1.062 | 1.062 ± 0.008723 | 2.035 | 2.037 ± 0.08194 | 4.156 (rollout) | 0.0625 / 0.0625 | 0.9849 | 0.9103 | 0.9945 | 1.068 |
| batch 512 | 1.064 | 1.065 ± 0.02648 | 1.963 | 1.964 ± 0.04803 | 4.199 (critic) | 0.0625 / 0.0625 | 0.9922 | 0.8658 | 0.9968 | 1.068 |
| batch 1024 | 1.107 | 1.108 ± 0.01487 | 1.935 | 1.936 ± 0.08091 | 5.373 (critic) | 0.0625 / 0.0625 | 1.002 | 0.8377 | 1.037 | 1.068 |

`*` = TD3 batch-100 run (unmatched to SAC/MBPO batch 256).

Per-segment share difference Ant − HC [pp], mean ± sd of the 5 paired differences, (k of 5 seeds with the same sign as the mean). Measured segments: `rollout`, `gradient_updates`; allocated GU sub-segments are in the CSV.

| config | rollout | GU |
|---|---|---|
| baseline | 5.045 ± 0.2028 (5/5) | -5.045 ± 0.2028 (5/5) |
| UTD 2 | 2.682 ± 0.1527 (5/5) | -2.682 ± 0.1527 (5/5) |
| UTD 4 | 1.409 ± 0.05512 (5/5) | -1.409 ± 0.05512 (5/5) |
| width 256 | 5.922 ± 0.1991 (5/5) | -5.922 ± 0.1991 (5/5) |
| width 512 | 5.416 ± 0.2938 (5/5) | -5.416 ± 0.2938 (5/5) |
| batch 256 | 4.156 ± 0.2565 (5/5) | -4.156 ± 0.2565 (5/5) |
| batch 512 | 3.408 ± 0.2019 (5/5) | -3.408 ± 0.2019 (5/5) |
| batch 1024 | 2.724 ± 0.2131 (5/5) | -2.724 ± 0.2131 (5/5) |

### 3.3 MBPO — ratios, max |Δshare|, p, power, J/FLOP

| config | E_total ratio (of means) | E_total paired mean ± sd | E_rollout ratio (of means) | E_rollout paired mean ± sd | max abs Δshare pp (segment) | p rollout share / p total | P_total ratio | (P_roll/P_GU) Ant/HC | J/FLOP total ratio | FLOPs total ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.221 | 1.222 ± 0.1145 | 1.768 | 1.768 ± 0.04223 | 3.818 (fit) | 0.0625 / 0.0625 | 0.9958 | 0.9731 | 0.9765 | 1.25 |
| UTD 2 | 1.178 | 1.178 ± 0.03608 | 1.749 | 1.75 ± 0.041 | 5.06 (fit) | 0.0625 / 0.0625 | 0.9945 | 0.9743 | 1 | 1.178 |
| UTD 4 | 1.136 | 1.137 ± 0.06357 | 1.755 | 1.755 ± 0.03532 | 5.948 (fit) | 0.0625 / 0.0625 | 0.9908 | 0.9754 | 1.007 | 1.128 |
| width 256 | 1.248 | 1.248 ± 0.04847 | 1.762 | 1.762 ± 0.03817 | 3.859 (fit) | 0.0625 / 0.0625 | 1.006 | 0.9765 | 0.7042 | 1.772 |
| width 512 | 1.202 | 1.201 ± 0.06729 | 1.759 | 1.759 ± 0.0251 | 2.962 (fit) | 0.0625 / 0.0625 | 1.003 | 0.9719 | 0.7998 | 1.502 |
| batch 512 | 1.183 | 1.182 ± 0.0788 | 1.726 | 1.726 ± 0.01703 | 3.228 (fit) | 0.0625 / 0.0625 | 0.9938 | 0.9449 | 1.012 | 1.169 |
| batch 1024 | 1.224 | 1.224 ± 0.06689 | 1.768 | 1.768 ± 0.01512 | 3.299 (fit) | 0.0625 / 0.0625 | 1.004 | 0.951 | 1.084 | 1.13 |

Per-segment share difference Ant − HC [pp], mean ± sd of the 5 paired differences, (k of 5 seeds with the same sign as the mean). Measured segments: `rollout`, `gradient_updates`, `dynamics_model_update`, `synthetic_rollout_generation`; allocated GU sub-segments are in the CSV.

| config | rollout | GU | fit | synth |
|---|---|---|---|---|
| baseline | 0.5321 ± 0.123 (5/5) | -4.607 ± 2.326 (5/5) | 3.818 ± 2.48 (5/5) | 0.2562 ± 0.03966 (5/5) |
| UTD 2 | 0.4374 ± 0.07401 (5/5) | -5.709 ± 1.719 (5/5) | 5.06 ± 1.818 (5/5) | 0.211 ± 0.02725 (5/5) |
| UTD 4 | 0.348 ± 0.04219 (5/5) | -6.444 ± 3.243 (5/5) | 5.948 ± 3.29 (5/5) | 0.1479 ± 0.01234 (5/5) |
| width 256 | 0.5085 ± 0.05985 (5/5) | -4.581 ± 0.8849 (5/5) | 3.859 ± 0.9381 (5/5) | 0.214 ± 0.0145 (5/5) |
| width 512 | 0.5705 ± 0.1393 (5/5) | -3.778 ± 1.526 (5/5) | 2.962 ± 1.67 (5/5) | 0.2451 ± 0.03415 (5/5) |
| batch 512 | 0.5317 ± 0.1149 (5/5) | -4.023 ± 2.05 (5/5) | 3.228 ± 2.2 (4/5) | 0.2635 ± 0.03871 (5/5) |
| batch 1024 | 0.4837 ± 0.08659 (5/5) | -4.016 ± 1.858 (5/5) | 3.299 ± 1.947 (5/5) | 0.2338 ± 0.01506 (5/5) |

### 3.4 TD-MPC2 — ratios, max |Δshare|, p, power, J/FLOP

| config | E_total ratio (of means) | E_total paired mean ± sd | E_rollout ratio (of means) | E_rollout paired mean ± sd | max abs Δshare pp (segment) | p rollout share / p total | P_total ratio | (P_roll/P_GU) Ant/HC | J/FLOP total ratio | FLOPs total ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.103 | 1.103 ± 0.006616 | 1.191 | 1.191 ± 0.002569 | 3.794 (rollout) | 0.0625 / 0.0625 | 1.035 | 0.9706 | 0.967 | 1.141 |
| num_q 3 | 1.147 | 1.147 ± 0.01206 | 1.225 | 1.225 ± 0.002517 | 3.491 (rollout) | 0.0625 / 0.0625 | 1.023 | 1 | 0.9713 | 1.18 |
| num_q 7 | 1.096 | 1.096 ± 0.008438 | 1.16 | 1.16 ± 0.001741 | 2.693 (rollout) | 0.0625 / 0.0625 | 0.9957 | 1.012 | 0.9821 | 1.116 |
| horizon 1 | 1.08 | 1.08 ± 0.01157 | 1.106 | 1.106 ± 0.002747 | 1.151 (actor) | 0.0625 / 0.0625 | 1.006 | 0.9924 | 0.9903 | 1.091 |
| horizon 5 | 1.138 | 1.138 ± 0.0113 | 1.229 | 1.229 ± 0.00372 | 4.014 (rollout) | 0.0625 / 0.0625 | 1.016 | 0.9972 | 0.9826 | 1.158 |

Per-segment share difference Ant − HC [pp], mean ± sd of the 5 paired differences, (k of 5 seeds with the same sign as the mean). Measured segments: `rollout`, `gradient_updates`, `world_model_pretrain`; allocated GU sub-segments are in the CSV.

| config | rollout | GU | pretrain |
|---|---|---|---|
| baseline | 3.794 ± 0.2919 (5/5) | -3.567 ± 0.2139 (5/5) | -0.2269 ± 0.08348 (5/5) |
| num_q 3 | 3.491 ± 0.5661 (5/5) | -3.224 ± 0.6111 (5/5) | -0.2668 ± 0.09416 (5/5) |
| num_q 7 | 2.693 ± 0.3326 (5/5) | -2.582 ± 0.3044 (5/5) | -0.1101 ± 0.06732 (4/5) |
| horizon 1 | 1.124 ± 0.4617 (5/5) | -1.11 ± 0.4835 (5/5) | -0.01383 ± 0.06512 (3/5) |
| horizon 5 | 4.014 ± 0.4823 (5/5) | -3.818 ± 0.4477 (5/5) | -0.1958 ± 0.04455 (5/5) |

### 3.5 Pooled summary across each algorithm's configurations (and all 27 pairs)

min–max over the configurations of the E ratios (ratio of means), max|Δshare| (with the segment that attains it at the minimum and at the maximum, and the set of segments that attain it in any configuration), total-power ratio, J/FLOP ratio; and in how many configurations the rollout-share difference has the same sign in all 5 seeds. Source: `pooled_summary.csv`.

| algorithm | # configs | E_total ratio | E_rollout ratio | max abs Δshare pp (segment at min – at max) | segments attaining max in any config | P_total ratio | J/FLOP total ratio | configs with rollout-share Δ same sign in all 5 seeds |
|---|---|---|---|---|---|---|---|---|
| SAC | 7 | 1.027 – 1.081 | 1.65 – 1.8 | 0.9364 (critic) – 3.399 (rollout) | actor; critic; rollout | 0.9999 – 1.008 | 0.8329 – 1.01 | 7 of 7 |
| TD3 | 8 | 1.039 – 1.107 | 1.935 – 2.288 | 1.409 (rollout) – 5.922 (rollout) | critic; rollout | 0.9849 – 1.006 | 0.8672 – 1.037 | 8 of 8 |
| MBPO | 7 | 1.136 – 1.248 | 1.726 – 1.768 | 2.962 (fit) – 5.948 (fit) | fit | 0.9908 – 1.006 | 0.7042 – 1.084 | 7 of 7 |
| TD-MPC2 | 5 | 1.08 – 1.147 | 1.106 – 1.229 | 1.151 (actor) – 4.014 (rollout) | actor; rollout | 0.9957 – 1.035 | 0.967 – 0.9903 | 5 of 5 |
| ALL (27 pairs) | 27 | 1.027 – 1.248 | 1.106 – 2.288 | 0.9364 (critic) – 5.948 (fit) | actor; critic; fit; rollout | 0.9849 – 1.035 | 0.7042 – 1.084 | 27 of 27 |

Additional pooled counts (all 27 pairs): rollout-share Δ positive (Ant higher) in 27/27 configurations, all 5 seeds each; total energy higher on Ant in 27/27, all 5 seeds each; every returned two-sided sign-flip p = 0.0625 (rollout share and total energy).

## 4. Confound checks

### 4.1 MBPO schedule confound

**Logged config differences** (from each run group's `metadata.json` → `algo_config` / `experiment_config`; identical across the 5 seeds of a group — asserted):

| comparison | algo_config differences (key: [first, second]) | experiment_config differences |
|---|---|---|
| HC baseline vs Ant baseline | {"rollout_max_epoch": [150, 100], "rollout_max_length": [1, 25]} | none |
| HC baseline vs Ant `rollout_max_length = 1` | {"rollout_max_epoch": [150, 100]} | none |
| Ant baseline vs Ant `rollout_max_length = 1` | {"rollout_max_length": [25, 1]} | — |
| Ant baseline vs Ant `rollout_max_length = 15` | {"rollout_max_length": [25, 15]} | — |

Differences are `rollout_max_length` 1 (HC) vs 25 (Ant) and `rollout_max_epoch` 150 (HC) vs 100 (Ant) — nothing else (hidden sizes, batch, UTD, ensemble, `real_ratio`, `rollout_min_epoch` = 20, `rollout_min_length` = 1, model training settings are equal; checked by the full dict diff). Rollout regimes (`flop_keys.mbpo_rollout_regime`): HC baseline `rl1-1_re20-150`, Ant baseline `rl1-25_re20-100`, Ant L1 `rl1-1_re20-100`, Ant L15 `rl1-15_re20-100`. Git commits: HC baseline `0bbbbb7`, Ant baseline `0bbbbb7`, **Ant L1 / L15 `788ffcd`** (a later sweep). Run start times (UTC): HC baseline 2026-09-08T17:44 – 2026-09-08T19:52; Ant baseline 2026-09-08T20:24 – 2026-09-08T22:57; Ant L1 2026-09-17T20:09 – 2026-09-17T22:34.

**Is any other configuration difference left for HC baseline vs Ant `rollout_max_length = 1`?** Logged: only `rollout_max_epoch` 150 vs 100. Verified that this difference has no effect on the executed schedule: `algorithms/mbpo.py::_rollout_length` interpolates between `rollout_min_length` and `rollout_max_length`, which are both 1, so it returns 1 at every epoch whatever the epoch bounds; the logged `rollout_length` per epoch is `{1}` in all 10 runs (HC baseline and Ant L1; read from `training_metrics.json`), the model-buffer capacity `max(10000, rollout_batch_size · rollout_max_length · model_retain_epochs)` depends on `rollout_max_length` only, and the synthetic samples per run are 990,000 in every HC-baseline and every Ant-L1 run (vs. Ant baseline 9,155,434–9,578,166). What **does** remain: (i) the environment itself; (ii) the Ant L1 runs were produced at a later commit (`788ffcd`, UTC start times 2026-09-17) than the HC baseline (`0bbbbb7`, 2026-09-08) — `git diff --name-status 0bbbbb7 788ffcd -- algorithms utils configs experiment_runner.py run_experiment.py` shows `algorithms/td3.py`, `algorithms/tdmpc2.py`, the `configs/overrides/*.json` files added, `configs/config.py` (+147 lines, insertions only per `git diff --stat`) and `experiment_runner.py` (+8 lines: the `td3` and `tdmpc2` dispatch branches added after the unchanged `mbpo` branch of `_dispatch_train`, verified with `git diff 0bbbbb7 788ffcd -- experiment_runner.py`) modified, and **no change to `algorithms/mbpo.py`, `sac.py`, `dynamics_model.py`, `replay_buffer.py`, `termination_fns.py` or `utils/`** (consistent with the thesis todo); (iii) the HC baseline and Ant L1 runs were launched in different sessions (different days). A same-commit, same-session HC-vs-Ant L1 pair does not exist: NOT FOUND (see section 5 below).

**Paired comparison HC baseline vs Ant `rollout_max_length = 1`** (new, `DERIVED (not a repo script)`: same definitions as §2; `c1_mbpo_hc_base_vs_ant_rollout1.csv`, one-row env summary `c1b_mbpo_hc_base_vs_ant_rollout1_envrow.csv`). Energy ratio = ratio of means [paired mean ± sd; seeds with ratio > 1]; the last column repeats the ratio of means of the **baseline** pairing (HC baseline vs Ant baseline, L_max = 25) for comparison. Share Δ in pp, mean ± sd (k/5 same sign); p = exact two-sided sign-flip p of the paired share difference / of the paired energy difference.

| segment | E HC base [J] | E Ant L1 [J] | E ratio of means [paired mean ± sd; seeds>1] | energy sign-flip | share HC → Ant (%); Δ pp; p | ratio of means, baseline pairing |
|---|---|---|---|---|---|---|
| rollout | 2428 | 4252 | 1.751 [1.752 ± 0.04146; 5/5] | p 0.0625 | 1.171 → 1.767; 0.5951 ± 0.04322 (5/5); p 0.0625 | 1.768 |
| fit | 1.462e+05 | 1.764e+05 | 1.207 [1.209 ± 0.06485; 5/5] | p 0.0625 | 70.46 → 73.28; 2.824 ± 1.208 (5/5); p 0.0625 | 1.288 |
| synth | 75.4 | 121.9 | 1.617 [1.682 ± 0.4605; 5/5] | p 0.0625 | 0.03654 → 0.0506; 0.01405 ± 0.009114 (5/5); p 0.0625 | 9.757 |
| GU | 5.87e+04 | 5.993e+04 | 1.021 [1.021 ± 0.01351; 5/5] | p 0.0625 | 28.33 → 24.9; -3.434 ± 1.175 (5/5); p 0.0625 | 1.018 |
| buf | 1677 | 2018 | 1.203 [1.204 ± 0.01541; 5/5] | p 0.0625 | 0.8094 → 0.8385; 0.02907 ± 0.02742 (4/5); p 0.1250 | 1.354 |
| critic | 2.459e+04 | 2.524e+04 | 1.026 [1.027 ± 0.02862; 4/5] | p 0.1875 | 11.87 → 10.49; -1.384 ± 0.6355 (5/5); p 0.0625 | 0.9988 |
| actor | 2.988e+04 | 3.014e+04 | 1.009 [1.009 ± 0.006989; 5/5] | p 0.0625 | 14.42 → 12.52; -1.901 ± 0.5008 (5/5); p 0.0625 | 1.015 |
| target | 2550 | 2535 | 0.9942 [0.9943 ± 0.0126; 3/5] | p 0.5000 | 1.231 → 1.053; -0.1773 ± 0.03782 (5/5); p 0.0625 | 1.01 |
| TOTAL | 2.074e+05 | 2.407e+05 | 1.161 [1.162 ± 0.04271; 5/5] | p 0.0625 | — | 1.221 |

Other columns for this pairing (from `c1b…envrow.csv`): max abs Δshare 2.824 pp attained by `dynamics_model_update`; P_total ratio 0.9997; (P_rollout/P_GU) double ratio 0.9769; J/FLOP total ratio 0.9584; FLOPs total ratio 1.211; sign-flip p rollout share 0.0625, total energy 0.0625. The ratios of means for the total (1.161), fit (1.207), rollout (1.751), synthetic (1.617) and GU (1.021) reproduce the numbers in the thesis paragraph "Regime-matched environment comparison" (line 4664); the paired sd and the sign-flip tests that the thesis todo asks for are the new columns above.

### 4.2 TD-MPC2 episodic confound

`algo_config.episodic` in `metadata.json` over **all** TD-MPC2 runs in `results/` (canonical and non-canonical): `Ant-v5|episodic=True|canonical`: 25 run(s); `HalfCheetah-v5|episodic=False|canonical`: 25 run(s); `HalfCheetah-v5|episodic=False|non-canonical`: 1 run(s). → Ant uses `episodic: true` in all 25 Ant runs (`configs/overrides/tdmpc2_ant*.json`), HC uses `episodic: false` in all 26 HC runs. Any Ant run with `episodic: false`: NOT FOUND (0 runs; searched `results/tdmpc2/Ant-v5/*/*/metadata.json`). Any HC run with `episodic: true`: NOT FOUND (0 runs). `flops_per_call.json` likewise has `episodic` = false for every HC signature (5) and true for every Ant signature (5); no environment has both values.

FLOP-per-call difference attributable to the termination classifier: NOT FOUND — `flop_analysis/measure_flops.py` / `flops_per_call.json` / `compute_energy_per_flop.py` contain no isolated termination-head or classifier FLOP count (the `episodic` flag only switches the termination loss in the world-model step, `measure_flops.py:381-385`, and the imagined-trajectory termination in `algorithms/tdmpc2.py`), and there is no run with the classifier on HC or off on Ant. The per-call ratios in §4.3 (TD-MPC2) are therefore *environment + classifier combined*.

### 4.3 Per-call FLOP ratios Ant/HC at the baselines, and logged observation/action dimensions

Source: `flop_analysis/flops_per_call.json` (matmul FLOPs per call, measured with `torch.utils.flop_counter` on the real classes by `measure_flops.py`). Baseline signatures: SAC/MBPO `bs256_h1024x1024`, TD3 `bs100_h1024x1024`, TD-MPC2 `bs256_h3_…_nq5_…_epFalse` (HC) vs `…_epTrue` (Ant). Full table: `c3_per_call_flop_ratios.csv`.

| algo | quantity (per call) | HC | Ant | Ant/HC | source |
|---|---|---|---|---|---|
| SAC | actor_forward_bs1 | 2.15654e+06 | 2.34496e+06 | 1.087 | flops_per_call.json |
| SAC | critic_fwdbwd | 4.92359e+09 | 5.25494e+09 | 1.067 | flops_per_call.json |
| SAC | actor_fwdbwd | 3.84513e+09 | 4.13244e+09 | 1.075 | flops_per_call.json |
| SAC | target_update_elementwise_ops | 4.3008e+06 | 4.66944e+06 | 1.086 | flops_per_call.json |
| TD3 | actor_forward_bs1 | 2.14426e+06 | 2.32858e+06 | 1.086 | flops_per_call.json |
| TD3 | critic_fwdbwd | 1.92205e+09 | 2.05107e+09 | 1.067 | flops_per_call.json |
| TD3 | actor_fwdbwd | 1.06906e+09 | 1.14319e+09 | 1.069 | flops_per_call.json |
| TD3 | target_update_elementwise_ops | 6.44917e+06 | 7.00213e+06 | 1.086 | flops_per_call.json |
| MBPO | actor_forward_bs1 | 2.15654e+06 | 2.34496e+06 | 1.087 | flops_per_call.json |
| MBPO | critic_fwdbwd | 4.92359e+09 | 5.25494e+09 | 1.067 | flops_per_call.json |
| MBPO | actor_fwdbwd | 3.84513e+09 | 4.13244e+09 | 1.075 | flops_per_call.json |
| MBPO | target_update_elementwise_ops | 4.3008e+06 | 4.66944e+06 | 1.086 | flops_per_call.json |
| MBPO | dynamics_member_fwdbwd | 2.0009e+08 | 2.72589e+08 | 1.362 | flops_per_call.json |
| MBPO | dynamics_ensemble_forward_all_bs1 | 1.8452e+06 | 2.59e+06 | 1.404 | flops_per_call.json |
| MBPO | synthetic_per_sample (actor_forward_bs1 + dynamics_ensemble_forward_all_bs1) | 4.00174e+06 | 4.93496e+06 | 1.233 | DERIVED (not a repo script) |
| MBPO | dynamics_fit_per_sample_per_member (dynamics_member_fwdbwd / model_train_batch_size) | 781600 | 1.0648e+06 | 1.362 | DERIVED (not a repo script) |
| TD-MPC2 | rollout_per_env_step | 4.64261e+10 | 5.61814e+10 | 1.21 | flops_per_call.json |
| TD-MPC2 | gradient_update_critic | 2.5324e+10 | 2.7844e+10 | 1.1 | flops_per_call.json |
| TD-MPC2 | gradient_update_actor | 1.45815e+10 | 1.46151e+10 | 1.002 | flops_per_call.json |
| TD-MPC2 | gradient_update_total | 3.99055e+10 | 4.24591e+10 | 1.064 | flops_per_call.json |
| TD-MPC2 | gradient_update_target_elementwise_ops | 5.82245e+06 | 5.83269e+06 | 1.002 | flops_per_call.json |

Mapping to the segments: rollout = `actor_forward_bs1` (SAC/TD3/MBPO) or `rollout_per_env_step` (TD-MPC2, MPPI planning per env step); critic = `critic_fwdbwd`; actor = `actor_fwdbwd`; TD-MPC2 world-model step = `gradient_update_critic`, policy-prior step = `gradient_update_actor`; MBPO fit = `dynamics_member_fwdbwd` (per 256-sample batch per member) / per sample `dynamics_member_fwdbwd / model_train_batch_size`; MBPO synthetic per sample = `actor_forward_bs1 + dynamics_ensemble_forward_all_bs1` (as in `Section_4.2_data/s4_data.py::mbpo_extras`). The target update is elementwise (ops, not FLOPs).

**Observation / action dimensions actually logged**: `flops_per_call.json` (every algorithm × every signature; `measure_flops.py:84-85` reads `env.observation_space.shape[0]` / `env.action_space.shape[0]` from the instantiated Gymnasium env) gives HalfCheetah-v5 obs **17** / act **6** and Ant-v5 obs **105** / act **8** — consistent across all 42 (algo, env, signature) entries (`c3b_obs_act_dims.csv`). This resolves the open Ant-v5 observation-dimension todo in `tab:envs`: **105 is confirmed** (act 8). Cross-check by instantiating the env on this Windows machine (`gym.make`; Gymnasium 1.3.0): {'HalfCheetah-v5': [17, 6], 'Ant-v5': [105, 8]} — note this is *this* machine's Gymnasium, not the Linux box's. `metadata.json` logs **neither** the dimensions nor Gymnasium/MuJoCo versions or the Ant-v5 kwargs (NOT FOUND: obs dim in metadata.json; Gymnasium/MuJoCo version of the Linux experiment box — `contexts/Section_3.1_3.2_3.3.md` records the same gap); default `gym.make("Ant-v5")` settings are assumed by the code (no kwargs in `run_experiment.py`).

### 4.4 Episode counts per run (completed training episodes)

Source: `training_metrics.json` → `episodes` with `phase == "train"` (the same set `aggregate_results.py`/`correlation_analysis.py` use for returns); min / median / max over the 5 seeds. Baselines below; all 56 configurations in `c4_episode_counts.csv` (also warmup-phase episodes and episode-length min/median/max). HC: every episode lasts 1000 steps (never terminates) → exactly 100 train episodes per run, in every config. Ant episodes end early, so resets happen inside the `rollout` task.

| algorithm | HC baseline: min / median / max | Ant baseline: min / median / max | Ant baseline train-episode length min / median / max [steps] | Ant, all configs: lowest min – highest max |
|---|---|---|---|---|
| SAC | 100 / 100 / 100 | 234 / 269 / 271 | 9 / 192 / 1000 | 178 – 511 (7 configs) |
| TD3 | 100 / 100 / 100 | 194 / 248 / 277 | 6 / 188 / 1000 | 171 – 546 (8 configs) |
| MBPO | 100 / 100 / 100 | 196 / 225 / 282 | 9 / 142 / 1000 | 154 – 431 (9 configs) |
| TD-MPC2 | 100 / 100 / 100 | 884 / 1833 / 2201 | 6 / 22 / 1000 | 440 – 2556 (5 configs) |

Per-config table for Ant (train episodes min / median / max over the 5 seeds):

| algorithm | config | Ant train episodes |
|---|---|---|
| SAC | base | 234 / 269 / 271 |
| SAC | utd2 | 288 / 319 / 327 |
| SAC | utd4 | 375 / 409 / 433 |
| SAC | w256 | 178 / 189 / 216 |
| SAC | w512 | 255 / 269 / 325 |
| SAC | b512 | 328 / 355 / 355 |
| SAC | b1024 | 379 / 430 / 511 |
| TD3 | base | 194 / 248 / 277 |
| TD3 | utd2 | 239 / 249 / 305 |
| TD3 | utd4 | 276 / 316 / 337 |
| TD3 | w256 | 171 / 189 / 216 |
| TD3 | w512 | 196 / 219 / 263 |
| TD3 | b256 | 253 / 299 / 350 |
| TD3 | b512 | 266 / 339 / 379 |
| TD3 | b1024 | 389 / 504 / 546 |
| MBPO | base | 196 / 225 / 282 |
| MBPO | utd2 | 214 / 276 / 285 |
| MBPO | utd4 | 243 / 324 / 431 |
| MBPO | w256 | 240 / 266 / 303 |
| MBPO | w512 | 235 / 243 / 268 |
| MBPO | b512 | 230 / 263 / 358 |
| MBPO | b1024 | 219 / 259 / 285 |
| MBPO | rollout1 | 154 / 160 / 230 |
| MBPO | rollout15 | 211 / 267 / 301 |
| TD-MPC2 | base | 884 / 1833 / 2201 |
| TD-MPC2 | numq3 | 1376 / 1990 / 2556 |
| TD-MPC2 | numq7 | 919 / 1216 / 1560 |
| TD-MPC2 | horizon1 | 440 / 454 / 536 |
| TD-MPC2 | horizon5 | 1044 / 1631 / 1712 |

HC: all 27 (algo, config) groups have min = median = max = 100 train episodes (verified).

### 4.5 Wall-clock per rollout step

`rollout` task duration per 1000 env steps = sum of the 100 per-epoch CodeCarbon `rollout_<i>` task `duration`s ÷ (100 epochs × 1000 steps) × 1000, from each run's per-task CSV `emissions_*.csv` (equal to the pipeline's `duration_s` of the `rollout` row ÷ 100000 × 1000; max relative deviation between the two routes 5.3e-16). Mean ± sd over 5 seeds. The rollout task contains action selection, `env.step`, buffer insertion and any `env.reset` (for TD-MPC2: the MPPI planning). Baselines; all configs in `c5_rollout_wallclock_per_1000_steps.csv`.

| algorithm | HC [s / 1000 steps] | Ant [s / 1000 steps] | Ant/HC (baseline) | Ant/HC range over all paired configs |
|---|---|---|---|---|
| SAC | 0.2048 ± 0.002653 | 0.3599 ± 0.00288 | 1.758 | 1.74 – 1.794 |
| TD3 | 0.1111 ± 0.0005962 | 0.2475 ± 0.001781 | 2.227 | 2.223 – 2.32 |
| MBPO | 0.2038 ± 0.004152 | 0.3668 ± 0.0008731 | 1.8 | 1.791 – 1.812 |
| TD-MPC2 | 10.14 ± 0.02517 | 11.95 ± 0.00509 | 1.178 | 1.108 – 1.22 |

(Rollout-energy ratio Ant/HC at the baselines for comparison: SAC 1.736, TD3 2.177, MBPO 1.768, TD-MPC2 1.191 — §3.)

### 4.6 Simulator-only cost

NOT FOUND: no measurement of MuJoCo step time, or of a simulator share of the `rollout` task, for either environment. Looked at: (a) `algorithms/{sac,td3,mbpo,tdmpc2}.py` — `perf_counter` timers exist only inside `update()` (e.g. sac.py:12, sac.py:148, sac.py:155, sac.py:158, sac.py:178, sac.py:184, …) for the allocated GU sub-segments; none wraps `env.step`; (b) `training_metrics.json` per-epoch keys: `actor_loss_mean`, `alpha_end`, `buffer_size`, `consistency_loss_mean`, `critic_loss_mean`, `cumulative_reward`, `env_step`, `epoch`, `epoch_reward_mean_per_step`, `epoch_reward_sum`, `mean_episode_return`, `model_buffer_size`, `model_holdout_mse`, `model_train_epochs`, `num_episodes_completed`, `pi_loss_mean`, `reward_loss_mean`, `rollout_length`, `synthetic_transitions_generated`, `total_loss_mean`, `value_loss_mean` — no env-step time; (c) `segment_energy.json` keys: `_sub_segment_wall_time_seconds`, `_total_kg_co2eq`, `actor_update`, `buffer_sample`, `critic_update`, `dynamics_model_update`, `idle_baseline_head`, `idle_baseline_tail`, `rollout`, `synthetic_rollout_generation`, `target_update`, `warmup`, `world_model_pretrain`; (d) the earlier context files already record it as not measured (`contexts/section_4.1.md:2143-2144` "No per-step simulator timing is logged, so the simulator's share of the `rollout` task duration is **NOT MEASURED**"; `section_4.1.md:2632` open item 14 "Simulator share of rollout time"). README.md line 138 only states that env stepping is "often CPU-bound / MuJoCo-simulation-bound" — a statement, not a measurement. No new experiment was run.

### 4.7 MBPO learning on Ant

Mean return of the last 10 % of train-phase episodes (`len//10` episodes; `flop_analysis/correlation_analysis.last_10pct_return` definition = `aggregate_results.py`'s; recomputed here and equal to `summary.csv` for all 80 MBPO canonical runs, max abs deviation 9.1e-13): **Ant: negative in all 9 of 9 MBPO configurations** (config means -756.7 … -244.6; 8 of 9 configs negative in every seed — the exception is batch 512 with a seed max of 2.283). HC: positive in all 7 configurations (4535 … 5904). Table (mean ± sd over 5 seeds; `c7_mbpo_last10pct_return.csv`):

| env | config | last-10 % return mean ± sd | seed min … max |
|---|---|---|---|
| HalfCheetah-v5 | base | 5146 ± 642.8 | 4371 … 5880 |
| HalfCheetah-v5 | utd2 | 5285 ± 499.8 | 4680 … 5946 |
| HalfCheetah-v5 | utd4 | 5904 ± 333.9 | 5433 … 6248 |
| HalfCheetah-v5 | w256 | 4535 ± 1028 | 3255 … 5653 |
| HalfCheetah-v5 | w512 | 5134 ± 451.4 | 4491 … 5547 |
| HalfCheetah-v5 | b512 | 5055 ± 468.3 | 4607 … 5763 |
| HalfCheetah-v5 | b1024 | 5181 ± 566.1 | 4361 … 5877 |
| Ant-v5 | base | -325.5 ± 160 | -523.2 … -157.6 |
| Ant-v5 | utd2 | -459.1 ± 263.6 | -883.1 … -188.1 |
| Ant-v5 | utd4 | -599.2 ± 186.1 | -855.2 … -336.5 |
| Ant-v5 | w256 | -695.5 ± 246.1 | -903.8 … -342.1 |
| Ant-v5 | w512 | -413.4 ± 167.6 | -628.3 … -250.2 |
| Ant-v5 | b512 | -244.6 ± 173 | -436.3 … 2.283 |
| Ant-v5 | b1024 | -348.5 ± 198.2 | -698.4 … -209.1 |
| Ant-v5 | rollout1 | -756.7 ± 525 | -1341 … -357.2 |
| Ant-v5 | rollout15 | -407.7 ± 245.8 | -802.1 … -138.5 |

## 5. NOT FOUND list

- NOT FOUND: `sec:env-effect` — no `\section`/`\label` for Section 5.4 in `thesis-main.tex` (only referenced at lines 4667, 5419, 5428, 5543, 5937).
- NOT FOUND: `tab:td3-env`, `tab:tdmpc2-env` — no TD3 / TD-MPC2 environment-comparison tables in the thesis (only prose paragraphs); `tab:sac-env` and `tab:mbpo-env` exist.
- NOT FOUND: Bib key about environment-dependent energy or simulator cost — none in `bibliography.bib` (searched `simulat`, `energy`).
- NOT FOUND: CSV export of the TD3 / MBPO / TD-MPC2 environment comparisons from the Chapter-4 scripts — only in `Section_4.{2,3,4}.md` (a SAC CSV, `sac_env_comparison.csv`, does exist). Now covered by `env_comparison_paired.csv`.
- NOT FOUND: Observation/action dimensions and Gymnasium/MuJoCo versions in `metadata.json` — not logged (dims are in `flops_per_call.json`: HC 17/6, Ant 105/8; Gymnasium/MuJoCo versions of the Linux box appear in no `metadata.json` or `run.log` under `results/`, and `requirements.txt` does not pin them).
- NOT FOUND: Any Ant TD-MPC2 run with `episodic: false`, or HC TD-MPC2 run with `episodic: true` — 0 runs.
- NOT FOUND: FLOP-per-call difference attributable to the TD-MPC2 termination classifier — no script/output isolates it.
- NOT FOUND: Simulator-only cost (MuJoCo step time, simulator share of `rollout`) for either environment — never measured.
- NOT FOUND: A same-commit, same-session HC-vs-Ant `rollout_max_length = 1` pair for MBPO — the Ant L1 runs were produced at `788ffcd` (UTC start 2026-09-17), the HC baseline at `0bbbbb7` (2026-09-08); the code difference between those commits does not touch MBPO (§4.1), but session effects cannot be separated.
- NOT FOUND: A ready-made static figure of per-environment segment shares — none in the repo (only the interactive dashboards listed in §2).
- NOT FOUND: Environment-matched comparisons of TD-MPC2 (episodic classifier) and of MBPO beyond `rollout_max_length = 1` — no runs exist (e.g. HC with L_max = 25, or HC TD-MPC2 with `episodic: true`).
