"""Renders contexts/section_5.4.md from contexts/section_5.4_data/*.csv + findings.json (run build_section_5_4.py first).
Also reads (read-only) the thesis thesis-main.tex / bibliography.bib for Section 1 of the md.
Run from the repo root:  python -I -W ignore contexts/section_5.4_data/render_section_5_4.py"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
D = REPO / "contexts" / "section_5.4_data"
TEXDIR = Path(r"C:\Users\saman\Desktop\Thesis\Thesis Report")
TEX = TEXDIR / "thesis-main.tex"
BIB = TEXDIR / "bibliography.bib"
HC, ANT = "HalfCheetah-v5", "Ant-v5"

F = json.loads((D / "findings.json").read_text(encoding="utf-8"))
P = pd.read_csv(D / "env_comparison_paired.csv")
POOL = pd.read_csv(D / "pooled_summary.csv")
X1 = pd.read_csv(D / "x1_crosscheck_vs_thesis.csv")
X2 = pd.read_csv(D / "x2_crosscheck_vs_contexts_4x.csv")
C1 = pd.read_csv(D / "c1_mbpo_hc_base_vs_ant_rollout1.csv")
C3 = pd.read_csv(D / "c3_per_call_flop_ratios.csv")
C4 = pd.read_csv(D / "c4_episode_counts.csv")
C5 = pd.read_csv(D / "c5_rollout_wallclock_per_1000_steps.csv")
C7 = pd.read_csv(D / "c7_mbpo_last10pct_return.csv")
DIMS = pd.read_csv(D / "c3b_obs_act_dims.csv")
tex = TEX.read_text(encoding="utf-8")
texl = tex.splitlines()
bib = BIB.read_text(encoding="utf-8")
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()

SEGS_SHORT = {"rollout": "rollout", "gradient_updates": "GU", "dynamics_model_update": "fit", "synthetic_rollout_generation": "synth",
              "world_model_pretrain": "pretrain", "buffer_sample": "buf", "critic_update": "critic", "actor_update": "actor", "target_update": "target"}
MEAS = {"sac": ["rollout", "gradient_updates"], "td3": ["rollout", "gradient_updates"],
        "mbpo": ["rollout", "gradient_updates", "dynamics_model_update", "synthetic_rollout_generation"],
        "tdmpc2": ["rollout", "gradient_updates", "world_model_pretrain"]}
SUBS = ["buffer_sample", "critic_update", "actor_update", "target_update"]
ALGO_NAME = {"sac": "SAC", "td3": "TD3", "mbpo": "MBPO", "tdmpc2": "TD-MPC2"}


def g4(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4g}"


def tbl(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex_block(a, b):
    return "\n".join(texl[a - 1:b])


def find_line(pat, start=0):
    for i in range(start, len(texl)):
        if re.search(pat, texl[i]):
            return i + 1
    return None


out: list[str] = []
w = out.append

# ================================================================================================ header
w("# Section 5.4 — Environment effect, quantified — context\n")
w("Data context for thesis Section 5.4 (HalfCheetah-v5 vs. Ant-v5, all four algorithms). Facts, numbers and provenance only — no interpretation, no LaTeX prose. "
  "Canonical seeds [331, 958, 14577, 43611, 85062] only (dev runs ignored); 280 runs; **gross** energies (idle floor included); joules. "
  "HC = HalfCheetah-v5, Ant = Ant-v5; every ratio/difference is **Ant relative to HC**, runs paired by seed number. "
  "Labelling rule: anything not found/computed/verified is written `NOT FOUND: <what, where looked>`; anything computed here and not by a repo script is tagged `DERIVED (not a repo script)`. "
  "Four significant digits here; full precision in the CSVs.\n")
w(f"Generated 2026-10-08 from git HEAD `{head}` (branch master); Windows dev checkout with synced `results/`. Nothing under `results/`, `flop_analysis/output/`, or the thesis `.tex`/`.bib` was modified; the thesis files were only read.\n")
w("**Files written** (all under `contexts/`): this file; `contexts/section_5.4_data/build_section_5_4.py` (computes everything → CSVs + `findings.json`; "
  "**new script**, `DERIVED (not a repo script)` in the sense that it is not one of the Chapter-4 generators, but its definitions are copied from them — see §2 of this file and the two cross-checks); "
  "`contexts/section_5.4_data/render_section_5_4.py` (renders this file). Re-run: `.venv/Scripts/python -I -W ignore contexts/section_5.4_data/build_section_5_4.py` then `.venv/Scripts/python -I -W ignore contexts/section_5.4_data/render_section_5_4.py`.\n")

# ================================================================================================ 1. Thesis state
w("## 1. Thesis state\n")
w(f"**Source files.** The thesis lives in one file: `{TEX}` ({len(texl)} lines; chapters are inline, no `\\input`/`\\include` of chapter files). "
  f"`{TEXDIR / 'bibliography.bib'}` is the bibliography; `commands.tex`, `abbreviations.tex` are small helper inputs. `main-english.tex` and `main-english-university-of-hamburg.tex` in the same folder are other (shorter, ~2100-line) documents; a search for `ch:comparison` / `sec:env-effect` in `main-english.tex` finds nothing, so they hold no Chapter 5.\n")
ch5 = find_line(r"\\chapter\{Cross-Algorithm Comparison\}")
secs = [(m, find_line(rf"\\label\{{{m}\}}", ch5 - 1)) for m in ("sec:segment-shares", "sec:energy-vs-compute", "sec:robustness")]
app = find_line(r"\\chapter\{My first appendix\}")
bounds = [s for _, s in secs] + [app]
w(f"**Chapter 5**: `\\chapter{{Cross-Algorithm Comparison}}\\label{{ch:comparison}}` at `thesis-main.tex:{ch5}`; the chapter runs to line {app - 1} (followed by `\\chapter{{My first appendix}}` at line {app}). "
  f"Chapter 5 intro is written (lines {ch5}–{secs[0][1] - 5}) and already announces Section 5.4 (\"\\Zcref{{sec:env-effect}} compares the two environments with the definitions of \\zcref{{sec:rqs}} for all four algorithms (RQ3)\", line {find_line('sec:env-effect compares|Zcref\\{sec:env-effect')}).\n")
rows = []
for (lab, s), e in zip(secs, bounds[1:]):
    body = [l for l in texl[s - 1:e - 1] if l.strip() and not l.strip().startswith("%")]
    name = re.search(r"\\section\{([^}]*)\}", texl[s - 1]).group(1)
    nsub = sum(1 for l in texl[s - 1:e - 1] if l.startswith("\\subsection"))
    rows.append([lab, name, f"{s}–{e - 1}", len(body), nsub, "written (non-empty)"])
rows.append(["`sec:env-effect`", "(Section 5.4 — does not exist)", "—", 0, 0, "NOT FOUND: no `\\section` / `\\label{sec:env-effect}` definition in thesis-main.tex (label only *referenced*)"])
w("**Sections 5.1–5.3 and 5.4**\n")
w(tbl(["label", "title", "lines", "non-comment non-blank lines", "# subsections", "state"], rows))
w("")
w("**Labels** (definition line in `thesis-main.tex`):\n")
lab_rows = []
for lab in ["ch:comparison", "sec:segment-shares", "sec:energy-vs-compute", "sec:robustness", "sec:rob-basis", "sec:rob-energy", "sec:rob-jpf", "sec:rob-rollout", "sec:rob-tdmpc2", "sec:rob-summary",
            "sec:env-effect", "sec:environments", "tab:envs", "tab:sac-env", "tab:mbpo-env", "tab:td3-env", "tab:tdmpc2-env", "sec:rqs", "sec:seeds", "sec:sweeps", "sec:baselines", "sec:threats", "ch:evaluation"]:
    d = find_line(rf"\\label\{{{re.escape(lab)}\}}")
    refs = [i + 1 for i, l in enumerate(texl) if re.search(rf"\{{{re.escape(lab)}\}}", l) and "\\label" not in l]
    lab_rows.append([f"`{lab}`", f"line {d}" if d else "NOT FOUND (no `\\label` definition)", f"{len(refs)} reference(s)" + (f", e.g. lines {', '.join(map(str, refs[:5]))}" if refs else "")])
w(tbl(["label", "definition", "references"], lab_rows))
w("\nAny `sec:env-*` label **defined**: only `sec:environments` (the \"Environments\" subsection of the experimental design, line 2843 — about the environments, not the comparison). `sec:env-effect` is referenced at lines "
  f"{', '.join(str(i + 1) for i, l in enumerate(texl) if 'sec:env-effect' in l)} but never defined (placeholder, as the todo at the end of the MBPO rollout subsection says). `sec:threats` and `ch:evaluation` are likewise referenced but not defined (see the table).\n")
# bib
keys = [(m.group(1), bib[:m.start()].count("\n") + 1) for m in re.finditer(r"^@\w+\{([^,\s]+),", bib, re.M)]
from collections import Counter
cnt = Counter(k for k, _ in keys)
want = {"MuJoCo": ["todorov2012mujoco"], "Gymnasium": ["towers2024gymnasium"], "CodeCarbon": ["codecarbon"], "SAC": ["haarnoja2018sac"], "TD3": ["fujimoto2018td3"],
        "MBPO": ["janner2019mbpo"], "TD-MPC2": ["hansen2024tdmpc2"]}
brow = []
for n, ks in want.items():
    for k in ks:
        ls = [l for kk, l in keys if kk == k]
        brow.append([n, f"`{k}`", ", ".join(map(str, ls)) if ls else "NOT FOUND"])
w("**Bib keys** (`bibliography.bib`):\n")
w(tbl(["topic", "key", "line(s) of definition"], brow))
related = ["chua2018pets", "williams2017mppi", "hansen2022tdmpc", "schulman2016gae", "wawrzynski2009cat", "henderson2018matters", "colas2018seeds", "agarwal2021precipice",
           "henderson2020systematic", "strubell2019energy", "schwartz2020green", "patterson2021carbon", "desislavov2023trends", "getzner2023accuracy", "khan2018rapl", "nvml", "pytorchflopcounter", "paszke2019pytorch", "virtuanen2020scipy", "virtanen2020scipy"]
w("\nOther keys present that 5.4 may reuse (key present = yes): " + ", ".join(f"`{k}`" for k in related if k in cnt) + ".")
dups = sorted(k for k, c in cnt.items() if c > 1)
w(f"\n**Duplicate keys in `bibliography.bib`**: {', '.join(f'`{k}` (lines {', '.join(str(l) for kk, l in keys if kk == k)})' for k in dups)}. (Observation only; `bibliography.bib` was not modified.)")
w("\n**Citation about environment-dependent energy or simulator cost**: NOT FOUND — no key in `bibliography.bib` whose title/fields concern simulator cost or environment-dependent energy (searched the file for `simulat`, `energy`: only `strubell2019energy`, `desislavov2023trends`, `getzner2023accuracy`, `henderson2020systematic`, which are general ML-energy references).\n")
# macros
def cnt_re(p):
    return len(re.findall(p, tex))
mac = [["`\\zcref{...}` / `\\Zcref{...}` (zref-clever; `\\Zcref` at sentence start)", cnt_re(r"\\zcref\{"), cnt_re(r"\\Zcref\{")],
       ["`\\todo{...}` (defined line 584: `TODO!` + `\\sidecomment`/`pdfcomment`)", cnt_re(r"\\todo\{"), "—"],
       ["siunitx: `\\num{}` (also `\\num{x+-y}`), `\\SI{}{\\percent}`, `\\SI{}{\\joule}` etc. (loaded line 611 with `group-minimum-digits=4,per-mode=fraction`; the thesis writes `\\SI{2.084}{\\percent}-points`)", cnt_re(r"\\num\{"), cnt_re(r"\\SI\{")],
       ["booktabs: `\\toprule \\midrule \\bottomrule`; `\\cmidrule(lr){a-b}`", cnt_re(r"\\toprule"), f"cmidrule {cnt_re(r'\\cmidrule')}"],
       ["`\\resizebox{\\textwidth}{!}{%` around `tabular` (tables `[tbp]`, `\\centering`, `\\footnotesize`, `@{}` column ends)", cnt_re(r"\\resizebox"), "—"],
       ["`\\paragraph{Name.}` blocks for sub-results; math `$\\bar P_r/\\bar P_{\\mathrm{GU}}$`, `$|\\Delta$share$|$`", cnt_re(r"\\paragraph\{"), "—"],
       ["`multirow` package is loaded (line 494) but `\\multirow{` is used 0 times; `makecell`, `tabularx`, `cellcolor`, `rowcolors`: not used", cnt_re(r"\\multirow\{"), "—"]]
w("**Macros / packages in use** (counts over `thesis-main.tex`):\n")
w(tbl(["macro / package", "count", "second count"], mac))
w("")
# tables
a = find_line(r"\\label\{tab:sac-env\}")
ts = max(i for i in range(1, a) if "\\begin{table}" in texl[i - 1])
te = min(i for i in range(a, len(texl)) if "\\end{table}" in texl[i - 1])
w(f"**Exact LaTeX of `tab:sac-env`** (`thesis-main.tex:{ts}–{te}`; the table that 5.4 should match; `tab:mbpo-env` has the identical column structure at the line given above):\n")
w("```latex\n" + tex_block(ts, te) + "\n```\n")
w(f"**Chapter 5 opening** (`thesis-main.tex:{ch5}–{secs[0][1] - 5}`):\n")
w("```latex\n" + tex_block(ch5, secs[0][1] - 5) + "\n```\n")
s53 = secs[2][1]
w(f"**Section 5.3 opening** (`thesis-main.tex:{s53}–{s53 + 11}`, a section-level opening for style):\n")
w("```latex\n" + tex_block(s53, s53 + 11) + "\n```\n")
w("Related thesis facts 5.4 must be consistent with (verbatim from `thesis-main.tex`): RQ3 part (iii) at line "
  f"{find_line(r'\\(iii\\)~The two environments')} (\"The two environments are compared by the difference of the shares of each segment and by the ratio of their energies, with the per-seed pairing of \\zcref{{sec:seeds}}. The paired differences are assessed with an exact sign-flip test, which cannot go below $p = \\num{{0.0625}}$ with five pairs, so the sign of a difference is stated but no stronger claim is made.\"); "
  f"environments table `tab:envs` (line {find_line(r'label.tab:envs')}) lists Ant-v5 observation dimension **105**, action dimension **8**, HalfCheetah-v5 **17**/**6** and carries an open `\\todo` to check the Ant-v5 observation dimension (resolved in §4.3 below); "
  "the todo at the end of the TD-MPC2 chapter (line 5406) says the termination classifier confounds the environment comparison; the todo at the MBPO rollout-length subsection (line 4667) asks whether Ant `L_max = 1` is the comparator (§4.1 below). "
  "The SAC/TD3/MBPO/TD-MPC2 baseline sentences already in the thesis: `Paired by seed, the total energy of Ant-v5 is \\num{1.057} / \\num{1.086} / \\num{1.221} / \\num{1.103} times that of HalfCheetah-v5 (sd of the paired ratios \\num{0.01179} / \\num{0.01747} / \\num{0.1145} / \\num{0.006616})`. "
  "Tables with this content exist for SAC (`tab:sac-env`) and MBPO (`tab:mbpo-env`) only; TD3 and TD-MPC2: NOT FOUND (no `tab:td3-env`/`tab:tdmpc2-env`; their environment comparison is in the prose paragraphs).\n")

# ================================================================================================ 2. Method and conventions
w("## 2. Method and conventions in use\n")
bm = P[P.config_tag == "base"].set_index("algo")
w("**Which statistic is \"E_total ratio\" in the existing tables?** **Ratio of means** — `mean(E_Ant over 5 seeds) / mean(E_HC over 5 seeds)`; the captions of `tab:sac-env` and `tab:mbpo-env` say \"Energy ratios are ratios of means\", and the code (`contexts/section_4.1_data/build_section_4_1.py` part5, 5.4 block; `contexts/Section_4.2_data/s4_blocks.py` `env_comparison_block`) computes `np.mean(E_ant)/np.mean(E_hc)`. "
  "The \"(sd of the paired ratios …)\" in the per-algorithm text is the sample sd (ddof = 1) of the five **per-seed** ratios `E_Ant,s / E_HC,s`; the headline number in the sentence (\"is 1.057 times\") is the ratio of means, **not** the mean of the paired ratios. Side by side (baselines):\n")
w(tbl(["algo", "ratio of means (thesis number)", "mean of 5 paired ratios", "sd of paired ratios (thesis `sd`)", "number printed in the thesis"],
      [[ALGO_NAME[a], g4(bm.loc[a, "E_total_ratio_of_means"]), g4(bm.loc[a, "E_total_paired_mean"]), g4(bm.loc[a, "E_total_paired_sd"]),
        {"sac": 1.057, "td3": 1.086, "mbpo": 1.221, "tdmpc2": 1.103}[a]] for a in ["sac", "td3", "mbpo", "tdmpc2"]]))
w("\nSource: `env_comparison_paired.csv` columns `E_total_ratio_of_means`, `E_total_paired_mean`, `E_total_paired_sd`. For the power ratio and the J/FLOP ratio the convention is also ratio of means (J/FLOP ratio = (mean E_Ant / mean F_Ant) / (mean E_HC / mean F_HC); the (P_rollout/P_GU) column is the double ratio built from the env means; the paired versions are in the CSV).\n")
w("**Share differences — per seed first, then averaged?** Yes. For each seed the share is `100 · E_segment / E_TOTAL_MEASURED_TRAINING` of that run, the paired difference is `share_Ant,s − share_HC,s` (pp), then mean and sd (ddof = 1) over the 5 differences; \"same sign\" = number of the 5 differences whose sign equals the sign of their mean (build_section_4_1.py part5 5.2 counts `> 0`, s4_blocks.py counts `sign == sign(mean)`; identical outcome whenever all 5 agree). "
  "`max |Δshare|` is `|mean of paired differences|` (equal to the difference of the mean shares) maximised over the measured non-GU segments **and the four allocated GU sub-segments**; `gradient_updates` itself is excluded because it is the sum of its sub-segments (this is why \"critic\"/\"actor\" can attain the maximum in `tab:sac-env`). Allocated sub-segments are included in the CSV (`dshare_<segment>_*`).\n")
w("**Sign-flip p-values — script/output.** Test: exact two-sided paired sign-flip permutation test, statistic |mean of the 5 paired differences (Ant − HC)|, all 2⁵ = 32 sign patterns, p = #{|mean| ≥ observed}/32 (tolerance 1e-12). Implemented as `signflip_p()` in `contexts/section_4.1_data/build_section_4_1.py:1416` (SAC; output `contexts/section_4.1_data/sac_env_comparison.csv`, metrics `signflip_p_two_sided::rollout share [pp]` and `::TOTAL energy [J]`) and verbatim in `contexts/Section_4.2_data/s4_data.py:169` (used by `s4_blocks.env_comparison_block`, line ~342, for TD3 / MBPO / TD-MPC2; output only inside `contexts/Section_4.2.md`, `Section_4.3.md`, `Section_4.4.md` — no CSV export of those env comparisons: NOT FOUND). "
  "Copied verbatim into `contexts/section_5.4_data/build_section_5_4.py` for this section. Tested quantities: (a) the paired difference of the rollout **share** (pp), (b) the paired difference of the **total energy** (J).\n")
w("**Which script produced `tab:sac-env` / `tab:mbpo-env`?** SAC: `contexts/section_4.1_data/build_section_4_1.py` (part5) → `contexts/section_4.1.md` / `sac_env_comparison.csv`. MBPO (and TD3, TD-MPC2): `contexts/Section_4.2_data/s4_blocks.py::env_comparison_block`, invoked by `build_section_4_2.py` / `build_section_4_3.py` / `build_section_4_4.py` → `Section_4.2.md` / `Section_4.3.md` / `Section_4.4.md`. "
  "Those scripts load `results/` themselves; the 27-row table below is produced by the **new** `build_section_5_4.py`, which re-implements the same definitions on top of `flop_analysis/output/per_run_energy_per_flop.csv` and `flop_analysis/section_5_2_analysis.parse()` (the config labelling used in 5.2/5.3). Two independent cross-checks (below) show it reproduces the earlier numbers.\n")
x1ok, x2ok = int(X1.ok.sum()), int(X2.ok.sum())
w("**Cross-checks (all passed)**\n")
w(f"- **vs. thesis draft tables** (`x1_crosscheck_vs_thesis.csv`): every cell of `tab:sac-env` and `tab:mbpo-env` (7 configs × 6 columns + segment label each, parsed directly from `thesis-main.tex`), plus the baseline E_total ratio and paired sd of SAC/TD3/MBPO/TD-MPC2 from the thesis text, plus the instruction's TD3 numbers: **{x1ok}/{len(X1)} reproduced** at the printed 4 digits. "
  "Specifically: baseline E_total ratio SAC 1.057 ✓, MBPO 1.221 ✓, TD3 1.086 ✓; max|Δshare| SAC 2.631 pp (rollout) ✓, MBPO 3.818 pp (fit) ✓, TD3 5.045 pp (rollout) ✓. "
  "**One cell is a rounding artefact, not a mismatch**: MBPO baseline (P_rollout/P_GU) ratio — computed 0.9731496, the repo context file prints 0.973150 (6 digits) and the thesis prints 0.9732 (the 6-digit value rounded half-up); a direct 4-digit rounding would give 0.9731.")
w(f"- **vs. the repo's own Chapter-4 context files** (`x2_crosscheck_vs_contexts_4x.csv`): the env-comparison tables of `Section_4.2.md` (TD3, 8 configs), `Section_4.3.md` (MBPO, 7), `Section_4.4.md` (TD-MPC2, 5) — 6 columns + segment label each: **{x2ok}/{len(X2)} reproduced** (max relative deviation {X2.rel_dev.max():.1e}, i.e. 6-digit rounding). TD3 and TD-MPC2 numbers in this file are therefore identical to the Chapter-4 contexts.\n")
w("**A pitfall found while reproducing the power column** (`DERIVED (not a repo script)` finding): in `flop_analysis/output/per_run_energy_per_flop.csv` the four GU sub-segment rows carry `duration_s` = the allocated `perf_counter` time and `mean_power_w` = E_sub/that time (e.g. 150.4 W), **not** the power of the measured `gradient_updates` task (145.4 W for the same SAC-Ant run). The thesis' (P_rollout/P_GU) uses E_GU / (sum of the `gradient_updates_<i>` task durations), so `build_section_5_4.py` reads the task durations from `emissions_*.csv`. Using the sub-row power instead shifts the double ratio by up to ~0.9 % (SAC batch 1024: 0.9279 vs 0.9194).\n")
w("**Existing figures that could be reused** (paths only; none is a static plot of per-env segment shares → NOT FOUND for a ready-made per-env segment-share figure):\n")
w("- `dashboard.py` (Dash app; per-run phase duration/power/energy bars — one run at a time)\n- `flop_dashboard.py` (Dash app over `flop_analysis/output/*.csv`; `env_id` is a pivot dimension, but the y-axes are energy / J-per-FLOP / ΔE/ΔF, not shares)\n- `flop_analysis/output/correlation/correlation_plots.html` (return/energy vs FLOPs per env)\n- `idle_power_report.html` (idle-baseline drift)\n- no `.png`/`.svg`/`.pdf` figure files exist in the repo outside `.venv`.\n")

# ================================================================================================ 3. Paired comparison
w("## 3. Paired comparison table references and per-algorithm summaries\n")
w("**CSV (the §2 deliverable): `contexts/section_5.4_data/env_comparison_paired.csv`** — 27 rows = 27 (algorithm, configuration) pairs present on both environments: "
  f"SAC {F['pair_counts']['sac']}, TD3 {F['pair_counts']['td3']}, MBPO {F['pair_counts']['mbpo']}, TD-MPC2 {F['pair_counts']['tdmpc2']} (count check ✓; Ant-only MBPO `rollout_max_length` 1 / 15 are not among them, see §4.1). "
  "Columns: `algo, config, config_tag`; `E_total_ratio_of_means`, `E_total_paired_mean`, `E_total_paired_sd`; same three for `E_rollout`; for every measured segment of the algorithm and the four allocated GU sub-segments `dshare_<segment>_mean_pp / _sd_pp / _same_sign_k_of_5`; "
  "`max_abs_dshare_pp, max_abs_dshare_segment`; `signflip_p_rollout_share, signflip_patterns_rollout_share, signflip_n_positive_rollout_share`, same for `total_energy`; `P_total_ratio_of_means` (+ paired mean/sd); `Proll_over_PGU_HC / _Ant / double_ratio_Ant_over_HC` (+ paired mean/sd); "
  "`jpf_total_HC/Ant_J_per_FLOP, jpf_total_ratio_Ant_over_HC`; `flops_total_ratio_Ant_over_HC` (+ the two FLOP totals); `tag_td3_batch_matched_to_sac_mbpo`; `algo_config_diff_HC_vs_Ant`, `experiment_config_diff_HC_vs_Ant`, `env_vs_config_confounded`, `git_commits_HC/Ant`.\n")
w("**TD3 matching flag**: TD3 rows `baseline, UTD 2, UTD 4, width 256, width 512` are TD3's own batch-100 runs (warmup 10000) — flagged *unmatched* to SAC/MBPO (batch 256); the TD3 batch rows use `b256 / b512 / b1024` and are the matched points (column `tag_td3_batch_matched_to_sac_mbpo`). HC vs. Ant within TD3 is always the same configuration on both sides.\n")
w("**Env-vs-config confound flag** (column `env_vs_config_confounded`, from the logged `algo_config` of the HC and Ant run groups — each group has an identical `algo_config` over its 5 seeds, asserted): SAC 0/7 configs differ between HC and Ant, TD3 0/8, **MBPO 7/7** (`rollout_max_length` 1 vs 25 and `rollout_max_epoch` 150 vs 100), **TD-MPC2 5/5** (`episodic` false vs true). "
  "`experiment_config` (protocol: warmup, steps, clocks, …) is identical between HC and Ant in all pairs except the SAC baseline, where the only difference is the logged `thermal_gate_reference_file` field (`null` on some runs vs `results/_thermal_reference.json`; irrelevant to the measurement). "
  "Git commits of the compared groups differ in several pairs (`git_commits_HC` vs `git_commits_Ant`: SAC utd2/utd4 `ecb98bb` vs `aa9186b`, SAC/TD3/MBPO batch rows `9b4b44c` vs `788ffcd`, SAC baseline Ant has two commits `92dcee0,f23f448`) — i.e. HC and Ant runs of the same configuration were launched by different sweep scripts at different times for those pairs.\n")
w("**Sign-flip result in one line**: in all 27 pairs the exact two-sided p returned is **0.0625** for the rollout share **and** for the total energy (the minimum possible with 5 pairs); the rollout-share difference is positive (Ant > HC) in all 5 seeds in all 27 pairs; the total energy is higher on Ant in all 5 seeds in all 27 pairs.\n")

for a in ["sac", "td3", "mbpo", "tdmpc2"]:
    sub = P[P.algo == a]
    w(f"### 3.{['sac', 'td3', 'mbpo', 'tdmpc2'].index(a) + 1} {ALGO_NAME[a]} — ratios, max |Δshare|, p, power, J/FLOP\n")
    rows = []
    for _, r in sub.iterrows():
        rows.append([r.config + ("*" if a == "td3" and "unmatched" in r.tag_td3_batch_matched_to_sac_mbpo else ""),
                     g4(r.E_total_ratio_of_means), f"{g4(r.E_total_paired_mean)} ± {g4(r.E_total_paired_sd)}",
                     g4(r.E_rollout_ratio_of_means), f"{g4(r.E_rollout_paired_mean)} ± {g4(r.E_rollout_paired_sd)}",
                     f"{g4(r.max_abs_dshare_pp)} ({SEGS_SHORT[r.max_abs_dshare_segment]})", f"{r.signflip_p_rollout_share:.4f} / {r.signflip_p_total_energy:.4f}",
                     g4(r.P_total_ratio_of_means), g4(r.Proll_over_PGU_double_ratio_Ant_over_HC), g4(r.jpf_total_ratio_Ant_over_HC), g4(r.flops_total_ratio_Ant_over_HC)])
    w(tbl(["config", "E_total ratio (of means)", "E_total paired mean ± sd", "E_rollout ratio (of means)", "E_rollout paired mean ± sd", "max abs Δshare pp (segment)",
           "p rollout share / p total", "P_total ratio", "(P_roll/P_GU) Ant/HC", "J/FLOP total ratio", "FLOPs total ratio"], rows))
    if a == "td3":
        w("\n`*` = TD3 batch-100 run (unmatched to SAC/MBPO batch 256).")
    w("")
    w(f"Per-segment share difference Ant − HC [pp], mean ± sd of the 5 paired differences, (k of 5 seeds with the same sign as the mean). Measured segments: {', '.join(f'`{s}`' for s in MEAS[a])}; allocated GU sub-segments are in the CSV.\n")
    rows = []
    for _, r in sub.iterrows():
        rows.append([r.config] + [f"{g4(r[f'dshare_{s}_mean_pp'])} ± {g4(r[f'dshare_{s}_sd_pp'])} ({int(r[f'dshare_{s}_same_sign_k_of_5'])}/5)" for s in MEAS[a]])
    w(tbl(["config"] + [SEGS_SHORT[s] for s in MEAS[a]], rows))
    w("")

w("### 3.5 Pooled summary across each algorithm's configurations (and all 27 pairs)\n")
w("min–max over the configurations of the E ratios (ratio of means), max|Δshare| (with the segment that attains it at the minimum and at the maximum, and the set of segments that attain it in any configuration), total-power ratio, J/FLOP ratio; and in how many configurations the rollout-share difference has the same sign in all 5 seeds. Source: `pooled_summary.csv`.\n")
rows = []
for _, r in POOL.iterrows():
    nm = ALGO_NAME.get(r.group, r.group)
    rows.append([nm, int(r.n_configs), f"{g4(r.E_total_min)} – {g4(r.E_total_max)}", f"{g4(r.E_rollout_min)} – {g4(r.E_rollout_max)}",
                 f"{g4(r.max_abs_dshare_min)} ({SEGS_SHORT[r.max_abs_dshare_segment_at_min]}) – {g4(r.max_abs_dshare_max)} ({SEGS_SHORT[r.max_abs_dshare_segment_at_max]})",
                 "; ".join(SEGS_SHORT[s] for s in r.max_abs_dshare_segments_all.split(";")),
                 f"{g4(r.P_total_min)} – {g4(r.P_total_max)}", f"{g4(r.jpf_total_min)} – {g4(r.jpf_total_max)}",
                 f"{int(r.rollout_share_same_sign_all5_n_configs)} of {int(r.n_configs)}"])
w(tbl(["algorithm", "# configs", "E_total ratio", "E_rollout ratio", "max abs Δshare pp (segment at min – at max)", "segments attaining max in any config", "P_total ratio", "J/FLOP total ratio", "configs with rollout-share Δ same sign in all 5 seeds"], rows))
w("\nAdditional pooled counts (all 27 pairs): rollout-share Δ positive (Ant higher) in 27/27 configurations, all 5 seeds each; total energy higher on Ant in 27/27, all 5 seeds each; every returned two-sided sign-flip p = 0.0625 (rollout share and total energy).\n")

# ================================================================================================ 4. Confound checks
w("## 4. Confound checks\n")
c = F["mbpo_confound"]
w("### 4.1 MBPO schedule confound\n")
w("**Logged config differences** (from each run group's `metadata.json` → `algo_config` / `experiment_config`; identical across the 5 seeds of a group — asserted):\n")
rows = [["HC baseline vs Ant baseline", json.dumps(c["algo_config_diff_HC_base_vs_Ant_base"]), json.dumps(c["experiment_config_diff_HC_base_vs_Ant_base"]) if c["experiment_config_diff_HC_base_vs_Ant_base"] else "none"],
        ["HC baseline vs Ant `rollout_max_length = 1`", json.dumps(c["algo_config_diff_HC_base_vs_Ant_rollout1"]), json.dumps(c["experiment_config_diff_HC_base_vs_Ant_rollout1"]) if c["experiment_config_diff_HC_base_vs_Ant_rollout1"] else "none"],
        ["Ant baseline vs Ant `rollout_max_length = 1`", json.dumps(c["algo_config_diff_Ant_base_vs_Ant_rollout1"]), "—"],
        ["Ant baseline vs Ant `rollout_max_length = 15`", json.dumps(c["algo_config_diff_Ant_base_vs_Ant_rollout15"]), "—"]]
w(tbl(["comparison", "algo_config differences (key: [first, second])", "experiment_config differences"], rows))
w("\nDifferences are `rollout_max_length` 1 (HC) vs 25 (Ant) and `rollout_max_epoch` 150 (HC) vs 100 (Ant) — nothing else (hidden sizes, batch, UTD, ensemble, `real_ratio`, `rollout_min_epoch` = 20, `rollout_min_length` = 1, model training settings are equal; checked by the full dict diff). "
  f"Rollout regimes (`flop_keys.mbpo_rollout_regime`): HC baseline `{c['regimes']['HC_base'][0]}`, Ant baseline `{c['regimes']['Ant_base'][0]}`, Ant L1 `{c['regimes']['Ant_rollout1'][0]}`, Ant L15 `{c['regimes']['Ant_rollout15'][0]}`. "
  f"Git commits: HC baseline `{c['git_commits']['HC_base'][0]}`, Ant baseline `{c['git_commits']['Ant_base'][0]}`, **Ant L1 / L15 `{c['git_commits']['Ant_rollout1'][0]}`** (a later sweep). "
  f"Run start times (UTC): HC baseline {c['start_utc_range']['HC_base'][0][:16]} – {c['start_utc_range']['HC_base'][1][:16]}; Ant baseline {c['start_utc_range']['Ant_base'][0][:16]} – {c['start_utc_range']['Ant_base'][1][:16]}; Ant L1 {c['start_utc_range']['Ant_rollout1'][0][:16]} – {c['start_utc_range']['Ant_rollout1'][1][:16]}.\n")
w("**Is any other configuration difference left for HC baseline vs Ant `rollout_max_length = 1`?** Logged: only `rollout_max_epoch` 150 vs 100. Verified that this difference has no effect on the executed schedule: `algorithms/mbpo.py::_rollout_length` interpolates between `rollout_min_length` and `rollout_max_length`, which are both 1, so it returns 1 at every epoch whatever the epoch bounds; "
  "the logged `rollout_length` per epoch is `{1}` in all 10 runs (HC baseline and Ant L1; read from `training_metrics.json`), the model-buffer capacity `max(10000, rollout_batch_size · rollout_max_length · model_retain_epochs)` depends on `rollout_max_length` only, and the synthetic samples per run are "
  f"{F['mbpo_synthetic_samples']['HC_base']['min']:,} in every HC-baseline and every Ant-L1 run (vs. Ant baseline {F['mbpo_synthetic_samples']['Ant_base']['min']:,}–{F['mbpo_synthetic_samples']['Ant_base']['max']:,}). "
  "What **does** remain: (i) the environment itself; (ii) the Ant L1 runs were produced at a later commit (`788ffcd`, UTC start times 2026-09-17) than the HC baseline (`0bbbbb7`, 2026-09-08) — `git diff --name-status 0bbbbb7 788ffcd -- algorithms utils configs experiment_runner.py run_experiment.py` shows `algorithms/td3.py`, `algorithms/tdmpc2.py`, the `configs/overrides/*.json` files added, `configs/config.py` (+147 lines, insertions only per `git diff --stat`) and `experiment_runner.py` (+8 lines: the `td3` and `tdmpc2` dispatch branches added after the unchanged `mbpo` branch of `_dispatch_train`, verified with `git diff 0bbbbb7 788ffcd -- experiment_runner.py`) modified, and **no change to `algorithms/mbpo.py`, `sac.py`, `dynamics_model.py`, `replay_buffer.py`, `termination_fns.py` or `utils/`** (consistent with the thesis todo); (iii) the HC baseline and Ant L1 runs were launched in different sessions (different days). "
  "A same-commit, same-session HC-vs-Ant L1 pair does not exist: NOT FOUND (see section 5 below).\n")
w("**Paired comparison HC baseline vs Ant `rollout_max_length = 1`** (new, `DERIVED (not a repo script)`: same definitions as §2; `c1_mbpo_hc_base_vs_ant_rollout1.csv`, one-row env summary `c1b_mbpo_hc_base_vs_ant_rollout1_envrow.csv`). "
  "Energy ratio = ratio of means [paired mean ± sd; seeds with ratio > 1]; the last column repeats the ratio of means of the **baseline** pairing (HC baseline vs Ant baseline, L_max = 25) for comparison. Share Δ in pp, mean ± sd (k/5 same sign); p = exact two-sided sign-flip p of the paired share difference / of the paired energy difference.\n")
rows = []
for _, r in C1.iterrows():
    sh = "—" if np.isnan(r.get("dshare_mean_pp", np.nan)) else f"{g4(r.share_HC_mean_pct)} → {g4(r.share_Ant_mean_pct)}; {g4(r.dshare_mean_pp)} ± {g4(r.dshare_sd_pp)} ({int(r.dshare_same_sign_k_of_5)}/5); p {r.signflip_p_share:.4f}"
    rows.append([SEGS_SHORT.get(r.segment, "TOTAL"), g4(r.E_HC_mean_J), g4(r.E_Ant_mean_J), f"{g4(r.E_ratio_of_means)} [{g4(r.E_ratio_paired_mean)} ± {g4(r.E_ratio_paired_sd)}; {int(r.E_ratio_seeds_gt1)}/5]",
                 f"p {r.signflip_p_energy:.4f}", sh, g4(r.E_ratio_of_means_baseline_pairing)])
w(tbl(["segment", "E HC base [J]", "E Ant L1 [J]", "E ratio of means [paired mean ± sd; seeds>1]", "energy sign-flip", "share HC → Ant (%); Δ pp; p", "ratio of means, baseline pairing"], rows))
l1 = F["mbpo_l1_envrow"]
w(f"\nOther columns for this pairing (from `c1b…envrow.csv`): max abs Δshare {g4(l1['max_abs_dshare_pp'])} pp attained by `{l1['max_abs_dshare_segment']}`; P_total ratio {g4(l1['P_total_ratio_of_means'])}; (P_rollout/P_GU) double ratio {g4(l1['Proll_over_PGU_double_ratio_Ant_over_HC'])}; J/FLOP total ratio {g4(l1['jpf_total_ratio_Ant_over_HC'])}; FLOPs total ratio {g4(l1['flops_total_ratio_Ant_over_HC'])}; sign-flip p rollout share {l1['signflip_p_rollout_share']:.4f}, total energy {l1['signflip_p_total_energy']:.4f}. "
  "The ratios of means for the total (1.161), fit (1.207), rollout (1.751), synthetic (1.617) and GU (1.021) reproduce the numbers in the thesis paragraph \"Regime-matched environment comparison\" (line 4664); the paired sd and the sign-flip tests that the thesis todo asks for are the new columns above.\n")

w("### 4.2 TD-MPC2 episodic confound\n")
ec = F["tdmpc2_episodic_counts"]
w("`algo_config.episodic` in `metadata.json` over **all** TD-MPC2 runs in `results/` (canonical and non-canonical): " + "; ".join(f"`{k}`: {v} run(s)" for k, v in ec.items()) +
  ". → Ant uses `episodic: true` in all 25 Ant runs (`configs/overrides/tdmpc2_ant*.json`), HC uses `episodic: false` in all 26 HC runs. "
  "Any Ant run with `episodic: false`: NOT FOUND (0 runs; searched `results/tdmpc2/Ant-v5/*/*/metadata.json`). Any HC run with `episodic: true`: NOT FOUND (0 runs). "
  "`flops_per_call.json` likewise has `episodic` = false for every HC signature (5) and true for every Ant signature (5); no environment has both values.\n")
w("FLOP-per-call difference attributable to the termination classifier: NOT FOUND — `flop_analysis/measure_flops.py` / `flops_per_call.json` / `compute_energy_per_flop.py` contain no isolated termination-head or classifier FLOP count (the `episodic` flag only switches the termination loss in the world-model step, `measure_flops.py:381-385`, and the imagined-trajectory termination in `algorithms/tdmpc2.py`), and there is no run with the classifier on HC or off on Ant. "
  "The per-call ratios in §4.3 (TD-MPC2) are therefore *environment + classifier combined*.\n")

w("### 4.3 Per-call FLOP ratios Ant/HC at the baselines, and logged observation/action dimensions\n")
w("Source: `flop_analysis/flops_per_call.json` (matmul FLOPs per call, measured with `torch.utils.flop_counter` on the real classes by `measure_flops.py`). Baseline signatures: SAC/MBPO `bs256_h1024x1024`, TD3 `bs100_h1024x1024`, TD-MPC2 `bs256_h3_…_nq5_…_epFalse` (HC) vs `…_epTrue` (Ant). Full table: `c3_per_call_flop_ratios.csv`.\n")
rows = []
for _, r in C3.iterrows():
    rows.append([ALGO_NAME[r.algo], r.quantity, f"{r.HC:.6g}", f"{r.Ant:.6g}", g4(r.ratio_Ant_over_HC), "DERIVED (not a repo script)" if str(r.source).startswith("DERIVED") else "flops_per_call.json"])
w(tbl(["algo", "quantity (per call)", "HC", "Ant", "Ant/HC", "source"], rows))
w("\nMapping to the segments: rollout = `actor_forward_bs1` (SAC/TD3/MBPO) or `rollout_per_env_step` (TD-MPC2, MPPI planning per env step); critic = `critic_fwdbwd`; actor = `actor_fwdbwd`; TD-MPC2 world-model step = `gradient_update_critic`, policy-prior step = `gradient_update_actor`; MBPO fit = `dynamics_member_fwdbwd` (per 256-sample batch per member) / per sample `dynamics_member_fwdbwd / model_train_batch_size`; MBPO synthetic per sample = `actor_forward_bs1 + dynamics_ensemble_forward_all_bs1` (as in `Section_4.2_data/s4_data.py::mbpo_extras`). The target update is elementwise (ops, not FLOPs).\n")
dm = F["dims"]
w(f"**Observation / action dimensions actually logged**: `flops_per_call.json` (every algorithm × every signature; `measure_flops.py:84-85` reads `env.observation_space.shape[0]` / `env.action_space.shape[0]` from the instantiated Gymnasium env) gives HalfCheetah-v5 obs **{dm[HC][0][0]}** / act **{dm[HC][0][1]}** and Ant-v5 obs **{dm[ANT][0][0]}** / act **{dm[ANT][0][1]}** — consistent across all {len(DIMS)} (algo, env, signature) entries (`c3b_obs_act_dims.csv`). "
  f"This resolves the open Ant-v5 observation-dimension todo in `tab:envs`: **105 is confirmed** (act 8). Cross-check by instantiating the env on this Windows machine (`gym.make`; Gymnasium {F.get('gymnasium_version_this_machine', 'NOT RUN')}): {F['live_env_dims_this_machine']} — note this is *this* machine's Gymnasium, not the Linux box's. "
  "`metadata.json` logs **neither** the dimensions nor Gymnasium/MuJoCo versions or the Ant-v5 kwargs (NOT FOUND: obs dim in metadata.json; Gymnasium/MuJoCo version of the Linux experiment box — `contexts/Section_3.1_3.2_3.3.md` records the same gap); default `gym.make(\"Ant-v5\")` settings are assumed by the code (no kwargs in `run_experiment.py`).\n")

w("### 4.4 Episode counts per run (completed training episodes)\n")
w("Source: `training_metrics.json` → `episodes` with `phase == \"train\"` (the same set `aggregate_results.py`/`correlation_analysis.py` use for returns); min / median / max over the 5 seeds. Baselines below; all 56 configurations in `c4_episode_counts.csv` (also warmup-phase episodes and episode-length min/median/max). HC: every episode lasts 1000 steps (never terminates) → exactly 100 train episodes per run, in every config. Ant episodes end early, so resets happen inside the `rollout` task.\n")
rows = []
for a in ["sac", "td3", "mbpo", "tdmpc2"]:
    h = C4[(C4.algo == a) & (C4.env == HC) & (C4.config == "base")].iloc[0]
    n = C4[(C4.algo == a) & (C4.env == ANT) & (C4.config == "base")].iloc[0]
    allant = C4[(C4.algo == a) & (C4.env == ANT)]
    rows.append([ALGO_NAME[a], f"{int(h.train_episodes_min)} / {g4(h.train_episodes_median)} / {int(h.train_episodes_max)}",
                 f"{int(n.train_episodes_min)} / {g4(n.train_episodes_median)} / {int(n.train_episodes_max)}",
                 f"{int(n.train_episode_length_min)} / {g4(n.train_episode_length_median)} / {int(n.train_episode_length_max)}",
                 f"{int(allant.train_episodes_min.min())} – {int(allant.train_episodes_max.max())} ({len(allant)} configs)"])
w(tbl(["algorithm", "HC baseline: min / median / max", "Ant baseline: min / median / max", "Ant baseline train-episode length min / median / max [steps]", "Ant, all configs: lowest min – highest max"], rows))
w("\nPer-config table for Ant (train episodes min / median / max over the 5 seeds):\n")
rows = []
for a in ["sac", "td3", "mbpo", "tdmpc2"]:
    for _, r in C4[(C4.algo == a) & (C4.env == ANT)].iterrows():
        rows.append([ALGO_NAME[a], r.config, f"{int(r.train_episodes_min)} / {g4(r.train_episodes_median)} / {int(r.train_episodes_max)}"])
w(tbl(["algorithm", "config", "Ant train episodes"], rows))
hcall = C4[C4.env == HC]
w(f"\nHC: all {len(hcall)} (algo, config) groups have min = median = max = 100 train episodes ({'verified' if (hcall.train_episodes_min == 100).all() and (hcall.train_episodes_max == 100).all() else 'NOT all 100 — see CSV'}).\n")

w("### 4.5 Wall-clock per rollout step\n")
w(f"`rollout` task duration per 1000 env steps = sum of the 100 per-epoch CodeCarbon `rollout_<i>` task `duration`s ÷ (100 epochs × 1000 steps) × 1000, from each run's per-task CSV `emissions_*.csv` (equal to the pipeline's `duration_s` of the `rollout` row ÷ 100000 × 1000; max relative deviation between the two routes {F['wallclock_pipeline_vs_taskcsv_max_rel_dev']:.1e}). "
  "Mean ± sd over 5 seeds. The rollout task contains action selection, `env.step`, buffer insertion and any `env.reset` (for TD-MPC2: the MPPI planning). Baselines; all configs in `c5_rollout_wallclock_per_1000_steps.csv`.\n")
rows = []
for a in ["sac", "td3", "mbpo", "tdmpc2"]:
    h = C5[(C5.algo == a) & (C5.env == HC) & (C5.config == "base")].iloc[0]
    n = C5[(C5.algo == a) & (C5.env == ANT) & (C5.config == "base")].iloc[0]
    m = C5[C5.algo == a].pivot(index="config", columns="env", values="rollout_s_per_1000_steps_mean")
    rr = (m[ANT] / m[HC]).dropna()
    rows.append([ALGO_NAME[a], f"{g4(h.rollout_s_per_1000_steps_mean)} ± {g4(h.rollout_s_per_1000_steps_sd)}", f"{g4(n.rollout_s_per_1000_steps_mean)} ± {g4(n.rollout_s_per_1000_steps_sd)}",
                 g4(n.rollout_s_per_1000_steps_mean / h.rollout_s_per_1000_steps_mean), f"{g4(rr.min())} – {g4(rr.max())}"])
w(tbl(["algorithm", "HC [s / 1000 steps]", "Ant [s / 1000 steps]", "Ant/HC (baseline)", "Ant/HC range over all paired configs"], rows))
w("\n(Rollout-energy ratio Ant/HC at the baselines for comparison: SAC 1.736, TD3 2.177, MBPO 1.768, TD-MPC2 1.191 — §3.)\n")

w("### 4.6 Simulator-only cost\n")
w("NOT FOUND: no measurement of MuJoCo step time, or of a simulator share of the `rollout` task, for either environment. Looked at: (a) `algorithms/{sac,td3,mbpo,tdmpc2}.py` — `perf_counter` timers exist only inside `update()` (e.g. " + ", ".join(F["algorithms_files_with_timers"][:6]) + ", …) for the allocated GU sub-segments; none wraps `env.step`; (b) `training_metrics.json` per-epoch keys: " + ", ".join(f"`{k}`" for k in F["training_metrics_epoch_keys"]) + " — no env-step time; (c) `segment_energy.json` keys: " + ", ".join(f"`{k}`" for k in F["segment_energy_keys"]) + "; (d) the earlier context files already record it as not measured (`contexts/section_4.1.md:2143-2144` \"No per-step simulator timing is logged, so the simulator's share of the `rollout` task duration is **NOT MEASURED**\"; `section_4.1.md:2632` open item 14 \"Simulator share of rollout time\"). "
  "README.md line 138 only states that env stepping is \"often CPU-bound / MuJoCo-simulation-bound\" — a statement, not a measurement. No new experiment was run.\n")

w("### 4.7 MBPO learning on Ant\n")
a_ = C7[C7.env == ANT]
am = F["mbpo_ant_returns"]
w(f"Mean return of the last 10 % of train-phase episodes (`len//10` episodes; `flop_analysis/correlation_analysis.last_10pct_return` definition = `aggregate_results.py`'s; recomputed here and equal to `summary.csv` for all {F['return_check_vs_summary_csv']['n_matched']} MBPO canonical runs, max abs deviation {F['return_check_vs_summary_csv']['max_abs_dev']:.1e}): "
  f"**Ant: negative in all {am['n_negative_mean']} of {am['n_configs']} MBPO configurations** (config means {g4(a_.last10pct_return_mean.min())} … {g4(a_.last10pct_return_mean.max())}; {am['n_all_seeds_negative']} of 9 configs negative in every seed — the exception is batch 512 with a seed max of {g4(a_[a_.config == 'b512'].last10pct_return_max.iloc[0])}). "
  f"HC: positive in all {F['mbpo_hc_returns']['n_configs']} configurations ({g4(F['mbpo_hc_returns']['min'])} … {g4(F['mbpo_hc_returns']['max'])}). Table (mean ± sd over 5 seeds; `c7_mbpo_last10pct_return.csv`):\n")
rows = [[r.env, r.config, f"{g4(r.last10pct_return_mean)} ± {g4(r.last10pct_return_sd)}", f"{g4(r.last10pct_return_min)} … {g4(r.last10pct_return_max)}"] for _, r in C7.iterrows()]
w(tbl(["env", "config", "last-10 % return mean ± sd", "seed min … max"], rows))
w("")

# ================================================================================================ 5. NOT FOUND
w("## 5. NOT FOUND list\n")
nf = [
    "`sec:env-effect` — no `\\section`/`\\label` for Section 5.4 in `thesis-main.tex` (only referenced at lines " + ", ".join(str(i + 1) for i, l in enumerate(texl) if "sec:env-effect" in l) + ").",
    "`tab:td3-env`, `tab:tdmpc2-env` — no TD3 / TD-MPC2 environment-comparison tables in the thesis (only prose paragraphs); `tab:sac-env` and `tab:mbpo-env` exist.",
    "Bib key about environment-dependent energy or simulator cost — none in `bibliography.bib` (searched `simulat`, `energy`).",
    "CSV export of the TD3 / MBPO / TD-MPC2 environment comparisons from the Chapter-4 scripts — only in `Section_4.{2,3,4}.md` (a SAC CSV, `sac_env_comparison.csv`, does exist). Now covered by `env_comparison_paired.csv`.",
    "Observation/action dimensions and Gymnasium/MuJoCo versions in `metadata.json` — not logged (dims are in `flops_per_call.json`: HC 17/6, Ant 105/8; Gymnasium/MuJoCo versions of the Linux box appear in no `metadata.json` or `run.log` under `results/`, and `requirements.txt` does not pin them).",
    "Any Ant TD-MPC2 run with `episodic: false`, or HC TD-MPC2 run with `episodic: true` — 0 runs.",
    "FLOP-per-call difference attributable to the TD-MPC2 termination classifier — no script/output isolates it.",
    "Simulator-only cost (MuJoCo step time, simulator share of `rollout`) for either environment — never measured.",
    "A same-commit, same-session HC-vs-Ant `rollout_max_length = 1` pair for MBPO — the Ant L1 runs were produced at `788ffcd` (UTC start 2026-09-17), the HC baseline at `0bbbbb7` (2026-09-08); the code difference between those commits does not touch MBPO (§4.1), but session effects cannot be separated.",
    "A ready-made static figure of per-environment segment shares — none in the repo (only the interactive dashboards listed in §2).",
    "Environment-matched comparisons of TD-MPC2 (episodic classifier) and of MBPO beyond `rollout_max_length = 1` — no runs exist (e.g. HC with L_max = 25, or HC TD-MPC2 with `episodic: true`).",
]
for n in nf:
    w("- NOT FOUND: " + n)
w("")

(REPO / "contexts" / "section_5.4.md").write_text("\n".join(out), encoding="utf-8")
print("written", REPO / "contexts" / "section_5.4.md", len("\n".join(out)), "chars")
