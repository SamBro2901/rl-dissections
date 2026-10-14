# Edit log: insertion of the appendices into `thesis-main.tex`

Backup: `thesis-main.tex.bak-appendix` (copy of the file before the first edit). Lines before: 7255, after: 11648.

## Applied

1. **Appendix chapters.** old lines 7022..7248 ("APPENDIX BLOCKS" banner, commented clock block, \chapter{Learning check details}, \chapter{Reproducibility details}) -> replaced by appendix_A.tex .. appendix_F.tex (inline), 4628 lines.
- **Todo deleted** (FLOP details: reconciliation script (Appendix C has it; the commit todo is an AUTHOR todo in Appendix C)). Old: `old line 2575: \todo{Add this reconciliation as a script to the repository and extend it to all signatures before citing it a...`. Deleted.
- **Todo deleted** (placeholder labels sec:compute-accounting and app:flop-details (both defined now)). Old: `old line 2577: \todo{Placeholder labels still undefined: \texttt{sec:compute-accounting} (Section 2.4) and \texttt{app:flop-d...`. Deleted.
- **Todo deleted** (MBPO refit correction (Appendix A, Section "Refit Interval of the MBPO Dynamics Model")). Old: `old line 3825: \todo{Appendix~A must state that \texttt{model\_train\_freq} is \num{250} but the condition is evaluated once ...`. Deleted.
- **Todo deleted** (MBPO composition and idle-floor fractions (Appendix B, tab:app-mbpo-comp)). Old: `old line 3941: \todo{Appendix~B should hold the composition and the idle-floor fractions of all \num{16} configurations (the ...`. Deleted.
- **Todo deleted** (scripts of Section 5.2 and per-segment statistics and fits (Appendix E scripts table, Appendix B)). Old: `old line 5346: \todo{Scripts \texttt{flop\_analysis/section\_5\_2\_analysis.py} and \texttt{section\_5\_2\_render.py}, first ...`. Deleted.
- **Todo deleted** (per-seed orderings and pairwise seed counts (Appendix B, tab:app-seed-orderings, tab:app-pairwise)). Old: `old line 5811: \todo{Per-seed orderings and pairwise seed counts (\texttt{o3\_seed\_level.csv}, \texttt{o3b\_pairwise\_seed\_...`. Deleted.
- **Todo deleted** (scripts of Section 5.3 (Appendix E scripts table)). Old: `old line 5952: \todo{Add the scripts \texttt{flop\_analysis/section\_5\_3\_analysis.py} and \texttt{section\_5\_3\_render.py}...`. Deleted.
- **Todo deleted** (Chapter 6 todo: move app:clock-provenance (Appendix F is active)). Old: `old line 6305: \todo{Appendix \zcref{app:clock-provenance} (Provenance of the Two Runs) is a commented block at the end of th...`. Deleted.
- **Todo deleted** (Chapter 6 todo: move app:clock-details (Appendix F is active)). Old: `old line 6370: \todo{Appendix \zcref{app:clock-details} (Allocated Sub-Segments and Idle Windows) is a commented block at the...`. Deleted.

## New positions (after the edits)

- line 7016: `\chapter{Hyperparameters and Configuration}`
- line 7314: `\chapter{Full Result Tables}`
- line 10524: `\chapter{FLOP-Counting Details}`
- line 10898: `\chapter{Learning check details}`
- line 11266: `\chapter{Reproducibility details}`
- line 11534: `\chapter{Clock Experiment Details}`

## Other changes made by the insertion

- Only the insertion region and the nine deleted todo lines differ from the backup (checked with `diff`, see `appendix_report.md`, section 8).
- The hard-coded letters "Appendix~A", "Appendix~B" and "Appendix C" occurred only inside the three todos that were deleted; no other occurrence of "Appendix" outside `app:*` references exists in the text, so no `\zcref` replacement was needed.
- No other text of `thesis-main.tex` was changed. Todos that wait for the Linux machine or the author were kept (`NOT RUN` and `AUTHOR` todos are listed in `appendix_report.md`).
- Figure files added to `Thesis Report/figures/`: `app-return-{sac,td3,mbpo,tdmpc2}-{hc,ant}.tex` and `.csv` (16 files).

## Proposed, not applied (outside the insertion points of the instructions)

1. **Section 3.5.2 (`sec:flop-per-call`), sentence "The measurement was run on the GPU of the experiment machine."** `flop_analysis/flops_per_call.json` records `"_device_used_for_measurement": "cpu"`. Appendix C states the recorded value. Proposed text: "The counts depend only on the tensor shapes; the measurement device recorded in the output file is the CPU." (The next sentence of the thesis, that the counts would be identical on the CPU, can stay.)
2. **Section 3.5.6 (`sec:flop-verification`), paragraph "Consistency of the totals", "maximum relative deviation of 4.8e-13".** The value 4.8e-13 is the result of parsing the per-task CodeCarbon file with the default fast float parser of pandas; with exact parsing (`check_totals.py`) the maximum relative deviation is 2.2e-16 (3.6e-16 with `float_precision='round_trip'`). Proposed text: "... with a maximum relative deviation of 2.2e-16 (4.8e-13 if the per-task file is read with the default float parser of pandas; \zcref{app:flop-details})".
3. **Section 3.5.6, the sentence on the analytic cross-check** ("the analytic and measured counts agree exactly in the configuration inspected") could now say that all 188 comparisons of the 42 canonical signatures are exact (`tab:app-flop-crosscheck`). Not applied because the commit of `verify_analytic_flops.py` is still open (the four AUTHOR todos "Commit ... and cite the commit" stay).
4. **`tab:tdmpc2-base-energy`, `target_update`, HalfCheetah-v5, sd of the energy**: printed `109.8`, recomputed `109.7498` (rounds to `109.7` at four significant digits; the context file `tdmpc2_cross_seed_summary.csv` has the same value 109.749763). All other numbers of 50 tables agree to their printed digits.
5. **Titles of the appendices.** `Learning check details` and `Reproducibility details` (existing, not renamed as instructed) are in sentence case, the other four appendices (`Hyperparameters and Configuration`, `Full Result Tables`, `FLOP-Counting Details`, `Clock Experiment Details`) follow the title case of the other chapters. Proposed: title case for the two existing ones, or sentence case for the other four.
6. **Repository documentation (outside `thesis-main.tex`).** `README.md` (section on known data issues) and `CLAUDE.md` describe the 15 development runs with null clock fields as equivalent to 2000 MHz; the commits show that an unset field meant "lowest supported clock" at all commits recorded by these runs and "200 MHz" at `f109bd4` (Appendix F, `app:clock-provenance`). `README.md` also says that 7 runs ran under torch 2.11.0 / CodeCarbon 3.2.9--3.3.0; the run directories show 10 runs with torch 2.11.0 (SAC, HalfCheetah-v5, seeds 0 and 42).
7. **`todos.md`, `todo-appendix.md`** and `thesis-layout.md` still describe the appendices as not written ("196 of 196 comparisons" includes 8 comparisons of a development environment; the canonical number is 188). Not edited.
8. **Author decisions that were delivered as optional blocks** (kept in the text with `\todo{AUTHOR: ...}`): see `appendix_report.md`, section "Decisions delivered as variants".
