"""Assemble contexts/section_7.md from the CSVs in contexts/section_7_data/ (read-only on results/, flop_analysis/, configs/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd, numpy as np
from mdtab import md, fmt

D = Path(__file__).parent
REPO = D.parents[1]
rd = lambda n: pd.read_csv(D / n)

# ------------------------------------------------------------------ item 1 claims
c1 = pd.concat([rd("item1_claims_partial.csv"), rd("item1_claims_part2.csv")], ignore_index=True)
c1 = c1[~c1["item"].str.contains("dynamics_model_update share %") | c1["item"].str.contains("per-run shares")]
d2 = rd("item1_src_d2.csv"); cs = rd("item1_src_corr_summary.csv")
extra = []
for _, r in d2.iterrows():
    exp = {"HalfCheetah-v5": (0.8357, 27, 0.4715, 0.6110), "Ant-v5": (0.8463, 29, 0.5071, 0.6253)}[r.env]
    extra.append(dict(claim=4, item=f"pooled Spearman energy vs FLOPs {r.env}", found=f"rho={r.spearman_rho:.4f}, n={int(r.n)}",
                      status="VERIFIED" if round(r.spearman_rho, 4) == exp[0] and r.n == exp[1] else "MISMATCH",
                      source="contexts/section_5.2_data/d2_total_correlation.csv spearman_rho,n"))
    extra.append(dict(claim=5, item=f"pooled log-log slope / R2 {r.env}", found=f"slope={r.loglog_slope:.4f}, R2={r.loglog_r2:.4f}",
                      status="VERIFIED" if round(r.loglog_slope, 4) == exp[2] and round(r.loglog_r2, 4) == exp[3] else "MISMATCH",
                      source="d2_total_correlation.csv loglog_slope,loglog_r2"))
for env, rho, p in [("HalfCheetah-v5", 0.417, 0.0303), ("Ant-v5", -0.326, 0.0842)]:
    r = cs[(cs.env == env) & (cs.analysis == "return_vs_flops")].iloc[0]
    extra.append(dict(claim=4, item=f"Spearman return vs FLOPs {env}", found=f"rho={r.spearman_rho:.4f}, p={r.p_value:.4f}, n={int(r.n_configs)}",
                      status="VERIFIED" if round(r.spearman_rho, 3) == rho and round(r.p_value, 4) == p else "MISMATCH",
                      source="flop_analysis/output/correlation/correlation_summary.csv"))
c1 = pd.concat([c1, pd.DataFrame(extra)], ignore_index=True).sort_values("claim", kind="stable")
c1.to_csv(D / "item1_claims_all.csv", index=False)

# ------------------------------------------------------------------ item 2
L = rd("learning_check.csv"); Q = rd("learning_check_per_step_supplement.csv")
flag = L[L.flag_last_minus_first_le_0][["algo", "env", "config", "last10_return_mean", "first10_return_mean", "last_minus_first", "n_seeds_last_gt_first",
                                          "frac_early_terminated_last10", "first10_mean_length", "last10_mean_length"]]
rng = L.groupby(["env", "algo"]).agg(n_cfg=("config", "size"), last10_min=("last10_return_mean", "min"), last10_max=("last10_return_mean", "max"),
                                     n_flagged=("flag_last_minus_first_le_0", "sum"), n_eps_min=("n_episodes_min", "min"), n_eps_max=("n_episodes_max", "max")).reset_index()
ant_td = L[(L.algo == "tdmpc2") & (L.env == "Ant-v5")][["config", "last10_return_mean", "n_episodes_mean", "first10_mean_length", "last10_mean_length", "frac_early_terminated_last10"]]
qf = Q[(Q.env == "Ant-v5") & (Q.last_minus_first_per_step <= 0)][["algo", "config", "first10_return_per_step", "last10_return_per_step", "last_minus_first_per_step"]]

# ------------------------------------------------------------------ item 3b
S3 = rd("item3b_short_task_stats.csv")
s3 = S3[S3.segment.isin(["rollout", "gradient_updates"])][["algo", "env", "segment", "n_tasks", "duration_s_min", "duration_s_median", "duration_s_max",
                                                           "energy_J_min", "energy_J_median", "energy_J_max", "power_W_min", "power_W_median", "power_W_max",
                                                           "n_gpu_energy_le0", "n_cpu_energy_le0", "n_total_energy_le0"]]
s3s = S3[S3.segment.isin(["synthetic_rollout_generation", "dynamics_model_update"])][["algo", "env", "segment", "n_tasks", "duration_s_min", "duration_s_median", "duration_s_max",
                                                                                      "energy_J_median", "power_W_median", "power_W_max", "n_gpu_energy_le0", "n_cpu_energy_le0"]]
bk = rd("item3b_gpu_zero_by_duration_bucket_all280.csv")

# ------------------------------------------------------------------ item 4
E4 = rd("epoch_stationarity.csv")
e4 = E4[["algo", "env", "segment", "mean_epoch_energy_J", "ratio_ep0_9_over_ep10_99_mean_seed", "slope_pct_of_mean_per_100_epochs", "slope_pct_min_over_seeds",
         "slope_pct_max_over_seeds", "cv_within_run_mean", "r2_cum_energy_vs_cum_steps"]]

# ------------------------------------------------------------------ item 5
A5 = rd("item5a_idle_drift.csv")
a5 = A5[["scope", "convention", "n_runs", "n_tail_gt_head", "rel_pct_median", "rel_pct_q1", "rel_pct_q3", "rel_pct_max", "abs_W_median", "abs_W_q1", "abs_W_q3", "abs_W_max", "head_W_mean", "tail_W_mean"]]
G = rd("gross_vs_net.csv"); GR = rd("gross_vs_net_ratios.csv")
tot = G[(G.segment == "TRAINING_TOTAL")]
t_main = tot[tot.variant == "mean_head_tail"][["algo", "env", "n_runs", "E_gross_J", "E_net_J", "net_over_gross", "P_idle_W_mean", "jpf_gross_pJ", "jpf_net_pJ"]]
t_br = tot.pivot_table(index=["algo", "env"], columns="variant", values="E_net_J").reset_index()
t_brj = tot.pivot_table(index=["algo", "env"], columns="variant", values="jpf_net_pJ").reset_index()
sh = G[(G.variant == "mean_head_tail") & (G.segment != "TRAINING_TOTAL")][["algo", "env", "segment", "share_gross_pct", "share_net_pct"]]
ord_ = GR[["variant", "env", "basis", "ordering", "sac/td3", "mbpo/sac", "tdmpc2/mbpo", "jpf_ordering"]]

# ------------------------------------------------------------------ item 6
RN = rd("execution_timeline_runs.csv"); SS = rd("execution_timeline.csv"); PR = rd("execution_timeline_pairs.csv"); SO = rd("execution_timeline_seed_order.csv")
can = RN[RN.canonical]
sess = SS[SS.n_canonical > 0][["session", "first_start_utc", "last_end_utc", "span_h", "n_runs", "n_canonical", "n_dev", "algo_env_configs", "git_commits"]]
dev_s = SS[SS.n_canonical == 0]
pb = PR[PR.kind == "baseline_vs_sweep"]; pe = PR[PR.kind == "HC_vs_Ant_pair"]
pb_same = pb[~pb.different_day][["algo", "env", "config_b", "date_a", "date_b", "same_session", "hours_apart"]]
pe_diff = pe[pe.different_day][["algo", "config_a", "date_a", "date_b", "same_session", "hours_apart"]]
bhrs = pb.hours_apart

# ------------------------------------------------------------------ item 7
V = rd("item7_per_run_versions.csv")

# ------------------------------------------------------------------ item 9
LB = rd("item9_labels.csv"); BK = rd("item9_bibkeys.csv")
UR = rd("item9_undefined_refs.csv")
lab_by_ch = LB.groupby("chapter", sort=False).label.apply(lambda s: ", ".join(f"`{x}`" for x in s))

# =================================================================== write
o = []
w = o.append
w("""# Section 7 (Evaluation) — fact collection

Generated from the `rl-dissections` checkout at `C:\\Users\\saman\\Desktop\\Thesis\\Project Repository\\rl-dissections` (git HEAD `050b3c3`, run data for all 300 run directories present). Read-only on `results/`, `flop_analysis/`, `configs/`. All scripts and companion CSVs are in `contexts/section_7_data/` (shared loader `common7.py`, task cache `build_task_table.py`; `_task_table.pkl` is a cache of all 72,890 `emissions_base_*.csv` task rows of the 280 canonical runs).

**IMPORTANT scope note.** This session ran on the Windows dev machine, **not** the Linux research PC. The run data are here, so items 1–6, 8 and 9 were computed/read directly. Item 7 needs commands on the research PC (`/etc/os-release`, `nvidia-smi -q`, `pip freeze` of `rl-exp`, `lscpu`, ...); those could not be run and are marked `[NOT FOUND]` (the local `nvidia-smi` shows a different GPU, an RTX 4050 Laptop, and was deliberately not used). Where the run metadata records a fact, it is reported.

## Summary block (every `[NOT FOUND]`, `[UNVERIFIED]`, `[MISMATCH]`)

**[MISMATCH]**
- M1 (claim 11): "All `rollout` tasks are shorter than 1 s" holds for SAC (max 0.532 s), TD3 (0.349 s), MBPO (0.408 s) but **not TD-MPC2**: all 5000 baseline-sweep `rollout` tasks are 6.6–16.7 s (median 11.07 s; baseline rows 10.1–12.0 s). Source: `item1_claim11_task_durations.csv`.
- M2 (item 4 vs draft statement "energy per epoch roughly constant after the first epochs"): true for SAC, TD3, TD-MPC2 (epochs 0–9 / 10–99 ratio 0.96–1.09), **false for MBPO `dynamics_model_update`** (ratio 0.26 Ant / 0.33 HC; +71 % / +134 % linear growth over 100 epochs), MBPO `synthetic_rollout_generation` on Ant (ratio 0.16, slope +257 %) and hence for the MBPO training total (ratio 0.44 / 0.52; R² of cumulative energy vs steps 0.993 / 0.984 instead of ~1.000). Source: `epoch_stationarity.csv`.
- M3 (claim 2/3 share convention, not a numeric error): the draft shares 70.46 %/74.28 % (MBPO dynamics) and 47.83 %/51.62 % (TD-MPC2 rollout) are **means of per-run shares**; the ratio-of-means shares from `d1_config_table.csv` are 70.49 %/74.41 % and 47.82 %/51.62 %. State the convention. Source: `item1_claims_all.csv`, `gross_vs_net.csv` (`share_gross_pct`).

**[NOT FOUND]** (item 7, all need the research PC; looked in: `metadata.json` (only `torch_version`, `codecarbon_version`, `cuda_device_name`, `device`), `emissions_base_*.csv` (OS string, Python, CPU, GPU model, RAM total), `run.log`, `README.md`, `CLAUDE.md`, `cli_commands.txt`, `requirements.txt`, `contexts/*.md`)
- N1 `rl-exp/bin/pip freeze`; gymnasium, MuJoCo, `nvidia-ml-py` versions on the Linux box (not recorded in `metadata.json`; `requirements.txt` has unpinned names only). CUDA runtime version beyond the `+cu130` torch build tag.
- N2 `/etc/os-release` and `uname -r` (see U1).
- N3 NVIDIA driver version, open-vs-proprietary kernel modules, VBIOS.
- N4 `nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS`: power limits, current/max clocks, supported graphics-clock range (whether 200 MHz is inside it).
- N5 `torch.cuda.get_device_properties(0)`: SM count, compute capability → peak FP32 TFLOP/s at 2000 MHz cannot be derived (no datasheet number used; formula in item 7).
- N6 RAM type/speed, mainboard model/vendor, storage model (`lscpu`/`free -h`/`lsblk`/`/sys/class/dmi/id/*` not run; DIMM details need `sudo dmidecode`).

**[UNVERIFIED]**
- U1 "Debian" in the draft hardware table: the recorded OS string is `Linux-7.0.0-31-generic-x86_64-with-glibc2.43` (all 280 canonical runs). It does not name a distribution. Check: `cat /etc/os-release` on the research PC.
- U2 FP32 vs TF32: no module in the repo sets `allow_tf32` / `set_float32_matmul_precision` (grep over all `*.py`, `*.sh`, `*.md`, `*.txt` outside `.venv`/`contexts`: no match). PyTorch's defaults (`'highest'`, `allow_tf32=False`) were confirmed only on the *local* build (torch 2.14.0+cpu); the run stack is torch 2.13.0+cu130. Check: `python -c "import torch;print(torch.__version__, torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)"` in `rl-exp`.
- U3 Mechanism behind GPU-energy zeros on very short tasks (item 3): the data show a cut-off at ≈0.1 s (below), consistent with an NVML total-energy counter that updates in ≈100 ms steps; the update period itself was not measured. Check: poll `nvmlDeviceGetTotalEnergyConsumption` at ~1 kHz under load and histogram the step spacing.
- U4 CPU energy came from RAPL in all runs: source shows RAPL is used only if CodeCarbon selects `intel_rapl` mode; `run.log` does not log the mode. Supporting evidence: per-task `cpu_power` varies (e.g. 10.0 W idle, 29.3 W run average, not a constant TDP value) and `CLAUDE.md` records zero "default power consumption of 4 W per thread" warnings. Check: CodeCarbon's own log line "Tracking Intel CPU via RAPL interface" in a fresh run, or `ls /sys/class/powercap/intel-rapl*`.
- U5 Whether the GPU-counter quantisation biases the sum of short `rollout` tasks (item 3): no bias is visible (rollout mean power 117–122 W median for every algorithm, including TD3-HC whose median task is 0.110 s) but this was not tested directly.
""")

# ------------------------------------------------------------------ 1
w("\n---\n\n## 1. Claim check (P1)\n\nAll claims below were recomputed from the run data/pipeline outputs. Full table: `item1_claims_all.csv`. Scripts: `item1_claims.py`, `item1b_claims.py`.\n\n")
w(md(c1[["claim", "item", "found", "status", "source"]].assign(item=lambda d: d["item"].str.replace("�", "±"), found=lambda d: d["found"].str.replace("�", "±")), 4))
_rn = rd("execution_timeline_runs.csv"); _dv = _rn[~_rn.canonical]
DEVCOMP = "; ".join(f"{a}/{e}/seed_{s}: {n}" for (a, e, s), n in _dv.groupby(["algo", "env", "seed"]).size().items())
_pr = rd("item2_learning_per_run.csv")
HC_N = int((_pr.env == "HalfCheetah-v5").sum()); HC_BAD = int((~_pr[_pr.env == "HalfCheetah-v5"].all_len_1000).sum())
ANT_N = int((_pr.env == "Ant-v5").sum()); ANT_BAD = int((~_pr[_pr.env == "Ant-v5"].all_len_1000).sum())
w(f"""
Notes on the table:
- Claim 2/3 percent shares: see M3 above. The table shows the per-run-mean convention as VERIFIED; the ratio-of-means version differs in the second decimal (70.49 / 74.41 for MBPO dynamics).
- Claim 8: 1.1055 (lower bound of rollout ratio) rounds to 1.106 at three decimals; shown as VERIFIED.
- Claim 9 numbers come from summing all non-idle, non-warmup task rows of the two seed-0 MBPO HalfCheetah hidden-256 runs `results/mbpo/HalfCheetah-v5/seed_0/20260904_122958` (clock-lock 2000/2000) and `.../20260904_130915` (200/200); they also match `contexts/Section_6.md` lines 178–179 (189 511 J / 1638.1 s and 568 210 J / 5731.98 s).
- Claim 10: the 20 non-canonical directories are (algo/env/seed: count) {DEVCOMP}; the exact list is in `execution_timeline_runs.csv` (`canonical=False`).
- Claim 11 TD-MPC2 values requested in the draft: `rollout` min 6.616 s, median 11.066 s, max 16.678 s, 5000 tasks (all ≥ 1 s) — see M1. MBPO `synthetic_rollout_generation`: median 0.008523 s over 8000 tasks, max 0.116 s.
""")
w("Task duration table behind claim 11 (all canonical sweep runs; `item1_claim11_task_durations.csv`):\n\n")
w(md(rd("item1_claim11_task_durations.csv"), 4))

# ------------------------------------------------------------------ 2
w("""
---

## 2. Learning sanity check (P1)

`[DERIVED]` Script `item2_learning_check.py`; output `learning_check.csv` (56 rows, one per algo×env×config; extra columns `first10_mean_length`, `last10_mean_length`, `frac_early_terminated_first10`, `n_seeds_last_gt_first`, `flag_last_minus_first_le_0`), per-run table `item2_learning_per_run.csv`, supplementary `learning_check_per_step_supplement.csv`.

Definitions (same as `aggregate_results.py` lines 56–65): episodes with `phase == "train"` only (warm-up episodes excluded); `last10` = mean return of the last `max(1, n//10)` episodes, `first10` = of the first `max(1, n//10)` episodes of that run; per-config values are means over the 5 seeds (sd = sample sd, ddof=1, of the per-seed last10 values). `frac_early_terminated_last10` (Ant only) = fraction of the last-10 % episodes with `length < 1000`, averaged over seeds.

**Flagged configs (`last_minus_first <= 0`): 14 of 56 — all on Ant-v5.** (HalfCheetah-v5: 0 of 27; every HC config has all 5 seeds improving.)

""")
w(md(flag, 3))
w("\nPer algorithm/env summary of the last-10 % return across configs:\n\n")
w(md(rng, 2))
w(f"""
**Do first-vs-last comparisons use episodes of different lengths on Ant? — Yes.** HalfCheetah episodes are always 1000 steps ({HC_BAD} of {HC_N} HC runs contain any shorter episode). On Ant, {ANT_BAD} of the {ANT_N} Ant runs contain at least one episode shorter than 1000 steps; the number of train episodes per run varies strongly (SAC 178–511, MBPO 154–431, TD3 171–546, **TD-MPC2 884–2556**) because 100 000 train steps are split into episodes of varying length. Episode returns are per-episode sums, so an episode that ends early scores less regardless of per-step quality. Handling: the headline columns follow the requested definition (episode-count-based first/last 10 %, no length normalisation) and the lengths are reported next to them (`first10_mean_length`, `last10_mean_length`). As a sensitivity check (supplementary file) the return was also normalised per step: sum of returns / sum of lengths over the same first/last 10 % episodes. With per-step normalisation 4 of the 14 flagged configs remain non-improving (listed below); the other 10 flagged configs show a positive per-step change, i.e. their negative `last_minus_first` is driven (at least partly) by shorter late episodes.

""")
w(md(qf, 3))
w("""
TD-MPC2 on Ant: the small last-10 % episode returns (126–438) come from very short late episodes: mean episode length of the last 10 % is 38–209 steps (base: 45.5) with `frac_early_terminated_last10` 0.9–1.0, and 884–2556 episodes per run. Per step the last-10 % return is 2.0–3.5 vs 0.05–0.50 for the first 10 %. Rows:

""")
w(md(ant_td, 3))

# ------------------------------------------------------------------ 3
w("""
---

## 3. CodeCarbon and tasks shorter than the polling interval (P1)

### 3a. Source reading (`[FOUND]`; installed `codecarbon` 3.3.0 — the Windows dev venv has the same version as the research PC's recorded `codecarbon_version`; paths relative to `.venv/Lib/site-packages/codecarbon/`)

1. **What `start_task` / `stop_task` do** (`emissions_tracker.py:730` / `:795`). `start_task` stops the periodic energy scheduler (`self._scheduler.stop()`, ~l.752), starts a 1 s background *monitor* thread that only samples utilisation/CPU power history (`_scheduler_monitor_power`, interval=1, l.349–352; `_monitor_power` l.1136), resets the clock and **re-baselines every device counter**:
   `for hardware in self._hardware:` / `    hardware.start()` (l.776–777); then snapshots the running totals (`_prepare_emissions_data`, l.778). `stop_task` forces one final reading, `self._measure_power_and_energy()` (l.808), and returns `totals(stop) − snapshot(start)` via `emissions_data_delta.compute_delta_emission(self._active_task_emissions_at_start)` (l.828–830; `output_methods/emissions_data.py:55–63`); duration = `perf_counter()` difference (l.~842).
2. **Counters or polled power?** Per device:
   - GPU: cumulative NVML counter. `gpu_nvidia.py:54` `return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)`; `core/gpu_device.py:34–35` `def start(self): self.last_energy = self._get_energy_kwh()`; `:51,55` `new_last_energy = energy = self._get_energy_kwh()` … `self.energy_delta = energy - self.last_energy`.
   - CPU (only in `intel_rapl` mode): cumulative RAPL file. `core/rapl.py:67–68` `def start(self): self.last_energy = self._get_value()`, `:83` `self.energy_delta = energy - self.last_energy`; `external/hardware.py:405–408` `if self._mode in ("intel_rapl", "windows_emi"): energy = self._get_energy_from_cpus(...)`. (Other CPU modes use power × time; see U4.)
   - RAM: **power × duration**. `RAM` does not override `measure_power_and_energy` (grep: none in `external/ram.py`), so `BaseHardware` applies (`external/hardware.py:56`) `energy = Energy.from_power_and_time(power=power, time=Time.from_seconds(last_duration))`.
3. **A task shorter than the polling interval.** Because the scheduler is stopped inside a task and `stop_task` forces a measurement, task energy for GPU/CPU is a counter difference between the `start_task` baseline and the `stop_task` read; `measure_power_secs = 1.0` only governs the background scheduler (which is off during tasks), so task energy is **not** quantised to 1 s. Its resolution is instead that of the counters. Energy accrued between tasks (after one `stop_task`, before the next `start_task`) is not attributed to any task, because `hardware.start()` re-baselines the counters.
4. **RAM term.** `external/ram.py:312–341` `total_power()` → `_calculate_ram_power(memory_GB)` (DIMM-count model, `RAM_SLOT_POWER_X86 = 5` W, l.14, `:141`); in `tracking_mode='machine'` `memory_GB` is machine RAM (60.8 GB). Observed: `ram_power` is exactly **20.0 W in all 72 890 task rows** (min = max = 20.0). RAM task energy is therefore 20 W × task wall-clock: a constant-power model, not a measurement.

`[UNVERIFIED]` What the source cannot settle is the *update granularity* of the NVML counter; that is answered only empirically (3b, U3).

### 3b. Empirical check on the 8 canonical baselines (4 algos × 2 envs; 500 tasks per segment = 100 epochs × 5 seeds)

`[DERIVED]` `item3b_short_tasks.py` → `item3b_short_task_stats.csv`. Source: `emissions_base_*.csv` (task rows `rollout_<epoch>`, `gradient_updates_<epoch>`, …). Energy = `energy_consumed` (CPU+GPU+RAM) in J; power = energy ÷ task duration. Counts are the numbers of tasks with energy ≤ 0.

""")
w(md(s3, 3))
w("\nExtra segments (MBPO, same source):\n\n")
w(md(s3s, 3))
w("""
Findings (all `[DERIVED]` from the tables above and `item3b_gpu_zero_by_duration_bucket_all280.csv`):
- No `rollout` or `gradient_updates` task in the 8 baselines has zero/negative GPU, CPU or total energy (columns `n_*_le0` = 0). Per-task mean power is 86.9–220.7 W for rollouts and 126.9–189.0 W for gradient updates (plausible for this GPU+CPU+RAM total); rollout task power medians are 117–122 W for SAC/TD3/MBPO and 211–213 W for TD-MPC2.
- **MBPO `synthetic_rollout_generation` is different**: 720 of 1000 baseline tasks (466/500 HC, 254/500 Ant) have **exactly 0 GPU energy** (none negative); CPU energy is never ≤ 0. Median task duration 6.1 ms (HC) / 50.3 ms (Ant), and some tasks show implausible mean power (max 823–1200 W) when a counter step happens to land in a short window. Its recorded energy is 0.04 % (HC) / 0.29 % (Ant) of the MBPO baseline training total, so the effect on totals is small, but the *segment's own* energy is not reliable (GPU term mostly missing).
- Zero-GPU-energy tasks vs duration, **all 280 canonical runs, all segments except idle** (`item3b_gpu_zero_by_duration_bucket_all280.csv`):

""")
w(md(bk, 3))
w("""
  Every task ≥ 0.1 s has GPU energy > 0, almost every task ≤ 0.01 s has GPU energy = 0. This is consistent with an NVML energy-counter update period of order 0.1 s (U3). `rollout` tasks of SAC/MBPO/TD3 are 0.10–0.53 s (smallest: TD3-HalfCheetah, min 0.105 s, median 0.110 s), i.e. at or a few multiples of that scale: individual task energies are coarsely quantised, but all are non-zero and the per-task energy is only weakly correlated with duration (Pearson r 0.02–0.53 per algo/env, `item3b_short_tasks.py` output). TD-MPC2 rollouts (6.6–16.7 s) are two orders above it.
- CPU: no task of any length has CPU energy ≤ 0 (RAPL resolution is not an issue at these durations).
- Verdict: for `rollout` the energy is a valid counter-difference measurement but with coarse per-task resolution for the shortest (TD3-HC) tasks; for MBPO `synthetic_rollout_generation` the GPU part is largely missing (`[UNVERIFIED]` whether the sum over the run is unbiased, see U3/U5).
""")

# ------------------------------------------------------------------ 4
w("""
---

## 4. Per-epoch stationarity (P1)

`[DERIVED]` `item4_epoch_stationarity.py` → `epoch_stationarity.csv`. Baselines of 4 algos × 2 envs, 5 seeds each, 100 epochs (task names `<segment>_<epoch>`, epoch 0–99; per-epoch energy = task energy in J). `ratio` = mean(epochs 0–9) ÷ mean(epochs 10–99), computed per seed then averaged (the ratio of seed-mean series is also in the CSV and agrees to ≤ 0.002). `slope` = OLS slope of per-epoch energy on epoch × 100, as % of the run's mean epoch energy, averaged over seeds (min/max over seeds shown). `cv` = sd/mean of the 100 epoch values within a run, averaged over seeds. `R²` = squared Pearson correlation of cumulative training energy vs cumulative env steps (1000·(epoch+1)), per seed, averaged. `TRAINING_TOTAL` per epoch = sum of the per-epoch measured segments present (rollout + gradient_updates [+ MBPO dynamics + synthetic]); TD-MPC2's one-off `world_model_pretrain` has no epoch index and is excluded. Segments not present for an algorithm are omitted.

""")
w(md(e4, 3))
w("""
Reading (facts only): SAC and TD3 are flat (ratio 0.98–1.09; total-energy R² 0.9998–1.0000); TD-MPC2 is flat (ratio 0.96–0.99; within-run CV of rollout 0.5–0.6 %, total 1.5 %; R² 1.0000); MBPO `rollout` and `gradient_updates` are flat (ratio 0.99–1.03) but `dynamics_model_update` (ratio 0.26 Ant / 0.33 HC, CV 45 % / 72 %) and `synthetic_rollout_generation` on Ant (ratio 0.16) are strongly non-stationary — see M2. Individual-seed slopes of TD3 `gradient_updates` range from −9 % to +11 % per 100 epochs, so "flat" there means no systematic trend, not zero scatter.
""")

# ------------------------------------------------------------------ 5
w("""
---

## 5. Idle drift and gross-vs-net sensitivity (P1 for a, P2 for b)

### 5a. Idle drift over the 280 canonical runs

`[DERIVED]` `item5_idle.py` on `idle_power.csv` (repo root; produced by `idle_power_analysis.py`, last commit `57b3d21`; 280 rows, all canonical seeds, columns `head_total_w`/`tail_total_w` as recorded and `*_corr` window-corrected). Statistic: |P_tail − P_head| / P_head in %, and |P_tail − P_head| in W; median, quartiles (Q1/Q3), max. `n_tail_gt_head` = runs with tail > head. Output `item5a_idle_drift.csv`.

""")
w(md(a5, 2))
w("""
Headline (rows `ALL`): as recorded median |Δ| = 7.04 % (4.48 W, IQR 4.45–10.92 % / 2.86–7.12 W, max 23.2 % / 14.7 W; tail > head in 256/280); window-corrected median 10.02 % (6.23 W, IQR 7.61–14.36 % / 4.79–8.97 W, max 27.4 % / 16.8 W; tail > head in 261/280). Mean head 65.27 W (63.11 W corrected) vs mean tail 69.51 W.

### 5b. Sensitivity of the baseline results to the gross convention (sensitivity analysis only; thesis stays gross) — `[DERIVED]`

Script `item5_idle.py` → `gross_vs_net.csv` (segments, shares, totals, J/FLOP per variant) and `gross_vs_net_ratios.csv` (orderings and ratios). Population: the 8 baseline configs × 5 canonical seeds = 40 runs. Per run: `E_net(seg) = E(seg) − P_idle × T(seg)` with `E(seg)` = sum of the CodeCarbon task energies of the measured segment (`rollout`, `gradient_updates`, and where present `dynamics_model_update`, `synthetic_rollout_generation`, `world_model_pretrain`; sub-segments are allocated, not measured, so not used), `T(seg)` = summed task durations of that segment, `P_idle` = mean of window-corrected head and tail total power of that run (primary variant `mean_head_tail`); brackets `head_only` and `tail_only` use the window-corrected head or tail power alone. Sanity check: summed gross task energies equal the pipeline `TOTAL_MEASURED_TRAINING` energy per run (max relative error 3.2e-13). J/FLOP = mean energy ÷ mean matmul FLOPs (`TOTAL_MEASURED_TRAINING` FLOPs from `per_run_energy_per_flop.csv`), same ratio-of-means convention as `d1_config_table.csv`. Segment shares are means of per-run shares.

**Total training energy and J/FLOP, gross vs net (primary variant):**

""")
w(md(t_main, 3))
w("\n**Net total energy [J] for the three `P_idle` choices:**\n\n")
w(md(t_br, 1))
w("\n**Net training-total J/FLOP [pJ] for the three `P_idle` choices (gross for reference is in the previous table):**\n\n")
w(md(t_brj, 2))
w("\n**Algorithm orderings and ratios (energy ratios `sac/td3` = SAC÷TD3, etc.; J/FLOP ordering is lowest→highest):**\n\n")
w(md(ord_, 3))
w("\n**Measured-segment shares of training energy, gross vs net (primary variant; per-run mean shares):**\n\n")
w(md(sh, 2))
w("""
Facts: the energy ordering TD3 < SAC < MBPO < TD-MPC2 holds on both envs, gross and net, for all three `P_idle` variants. The J/FLOP ordering TD-MPC2 < SAC < TD3 < MBPO also holds gross and net, all variants, both envs. Net/gross total-energy ratio is 0.47–0.48 (TD3, MBPO), 0.52–0.53 (SAC), 0.65 (TD-MPC2). Between-algorithm energy ratios change (e.g. TD-MPC2/MBPO 1.95 → 2.67 on Ant, 2.16 → 2.93 on HC; SAC/TD3 2.11 → 2.35 on Ant). Shares move by ≤ 4.5 pp (largest: MBPO dynamics 70.46 → 66.07 % on HC, gradient_updates 28.33 → 32.77 %).
""")

# ------------------------------------------------------------------ 6
w(f"""
---

## 6. Execution order and timeline (P1)

`[DERIVED]` `item6_timeline.py`. Source: `start_time_utc` / `end_time_utc` / `git_commit` in all 300 `metadata.json`; config labels from `section_5_2_analysis.load()`. Outputs: `execution_timeline.csv` (sessions), `execution_timeline_runs.csv` (all runs with gap to previous run), `execution_timeline_configs.csv`, `execution_timeline_pairs.csv`, `execution_timeline_seed_order.csv`. A new session starts after a gap > 30 min between the end of one run and the start of the next; sessions were built over all 300 runs (dev runs included, flagged). Dates/times are UTC (local time at the machine is UTC+2: e.g. run `20260907_121335` has directory timestamp 12:13 local and `start_time_utc` 10:13).

**Totals.** The 280 canonical runs span {pd.Timestamp(can.start.min()):%Y-%m-%d %H:%M} to {pd.Timestamp(can.end.max()):%Y-%m-%d %H:%M} UTC. **Machine hours** = sum of (`end_time_utc` − `start_time_utc`) over runs (this includes settle, idle head/tail and teardown): **{can.dur_h.sum():.2f} h for the 280 canonical runs** ({RN.dur_h.sum():.2f} h for all 300). No two runs overlap (minimum gap between runs 0.021 min = 1.3 s). The 280 canonical runs carry {can.git_commit.nunique()} distinct `git_commit` values: {', '.join(sorted(can.git_commit.unique()))} (see per-session commits below; the commit is the repo HEAD at launch, not a record of which files differed).

**Sessions containing canonical runs** (sessions 1–9 and 13 contain only dev runs, {int(dev_s.n_runs.sum())} runs, 2026-08-13 to 2026-09-09; sessions 12 and 15 each include one dev run):

""")
w(md(sess.assign(first_start_utc=lambda d: d.first_start_utc.str[:16], last_end_utc=lambda d: d.last_end_utc.str[:16]), 2))
w(f"""
**Baseline vs sweep (48 pairs = every non-baseline config vs its baseline, same algo×env; `execution_timeline_pairs.csv`).** {int(pb.different_day.sum())} of 48 ran on a different UTC date than their baseline (gap {bhrs.min():.1f}–{bhrs.max():.1f} h between first starts) and {int((~pb.same_session).sum())} of 48 in a different session. The only same-day pairs are the {len(pb_same)} TD3 UTD pairs, all in session 14 together with the TD3 baseline:

""")
w(md(pb_same, 2))
w(f"""
**HalfCheetah vs Ant pairs (27 paired configs, as in `env_comparison_paired.csv`).** {int(pe.different_day.sum())} of 27 pairs have their two env legs on different UTC dates, {int((~pe.same_session).sum())} in different sessions (the difference between 11 and 9 is overnight sessions spanning midnight, e.g. MBPO w256/w512). The pairs on different days:

""")
w(md(pe_diff, 2))
w(f"""
**Seed and config order.**
- Seeds: for **{int(SO.in_list_order.sum())} of {len(SO)} canonical configs** the five seeds ran in the order of the seed list 331, 958, 14577, 43611, 85062, and for **{int(SO.consecutive_in_global_order.sum())} of {len(SO)}** they ran as five consecutive runs with no other run (canonical or dev) in between (`execution_timeline_seed_order.csv`). **Confirmed.**
- Config order within a sweep: **fixed, not randomised.** `grep -ln "shuf\\|RANDOM\\|sort -R" scripts/*.sh` returns nothing; the sweep scripts iterate nested `for` loops in listed order (e.g. `scripts/run_width_sweep.sh`: combos → widths → seeds; `run_batch_size_sweep.sh`: algo → batch → seed), and the observed run order matches (e.g. session 16: sac-HC w256, w512, sac-Ant w256, w512, mbpo-HC, mbpo-Ant, td3-HC, td3-Ant; session 18: sac, mbpo, td3 on HC with b512, b1024 (td3: b256, b512, b1024)). **Confirmed.**
- Drift/order confound: because the order is fixed, the config is confounded with time-of-run (baseline configs ran 2026-09-07 to 09-10; sweep configs ran from 2026-09-08 (SAC/TD3 UTD) through 2026-09-19, the width, MBPO-UTD, batch-size, Ant-rollout and TD-MPC2 sweeps on 09-12 to 09-19; within a sweep, algorithm and env follow loop order). This is a fact about the schedule; its effect size is not estimated here (see item 5a for idle-power drift).
- TD-MPC2 num_q/horizon sweep false start (described in `CLAUDE.md`; the log `results/tdmpc2/_numq_horizon_sweep.log` was not re-read here): session 20 contains exactly 40 runs, all canonical TD-MPC2 sweep configs, so no extra/partial run directory from the false start is present.
""")

# ------------------------------------------------------------------ 7
oss = V.os.unique().tolist()
w(f"""
---

## 7. Machine and software facts (P1)

Only what the run data record could be collected here (see scope note). Per-run table: `item7_per_run_versions.csv` (script `item7_versions.py`, from `metadata.json` + the first row of `emissions_base_*.csv` of each of the 280 canonical runs).

**Unique values across the 280 canonical runs `[FOUND]`** (one unique value per field, i.e. identical across all runs):

| Field | Value | Source |
|---|---|---|
| PyTorch | `{V.torch_version.iloc[0]}` (CUDA 13.0 build tag; this is the wheel's build tag, not a measured runtime version) | `metadata.json: torch_version`; unique count 1 |
| CodeCarbon | `{V.codecarbon_version.iloc[0]}` | `metadata.json` and `emissions_base_*.csv: codecarbon_version`; unique count 1 each |
| Python | `{V.python_version.iloc[0]}` | `emissions_base_*.csv: python_version`; unique count 1 |
| OS string | `{oss[0]}` | `emissions_base_*.csv: os`; unique count {len(oss)} |
| CPU | `{V.cpu_model.iloc[0]}`, `cpu_count` {V.cpu_count.iloc[0]} | `emissions_base_*.csv` |
| GPU | `{V.cuda_device_name.iloc[0]}` (`gpu_count` {V.gpu_count.iloc[0]}) | `metadata.json: cuda_device_name` |
| RAM total | {V.ram_total_size_gb.iloc[0]} GB (`ram_total_size` as CodeCarbon reports it) | `emissions_base_*.csv` |
| Tracking mode | `{V.tracking_mode.iloc[0]}` | `emissions_base_*.csv` |
| Device | `{V.device.iloc[0]}` | `metadata.json` |

`[NOT FOUND]` Gymnasium, MuJoCo, `nvidia-ml-py` versions on the research PC: `metadata.json` keys are only `{V.metadata_keys.iloc[0]}`. (`contexts/section_4.1.md` lists gymnasium 1.3.0 / mujoco 3.12.0, but those are from the local Windows venv, not the Linux box.) `rl-exp/bin/pip freeze` could not be run here.

Note: `contexts/Section_6.md` B4 quotes the kernel string `7.0.0-30-generic`; that value belongs to 7 non-canonical dev runs. All 280 canonical runs record `7.0.0-31-generic` (checked over all 300 `emissions_base` files: 280 canonical → `-31`; dev runs: 7 → `-30`, 3 → `-31`, 10 → Windows 11 `10.0.26200`, SAC dev runs from August on the same CPU/GPU).

**OS / Debian check (draft table says Debian)** — `[NOT FOUND]` `/etc/os-release`, `uname -r`; `[UNVERIFIED]` (U1). What is recorded: `Linux-7.0.0-31-generic-x86_64-with-glibc2.43`; no distribution name.

**GPU / driver** — `[NOT FOUND]` driver version, open kernel modules, VBIOS (N3); GPU name `[FOUND]` NVIDIA GeForce RTX 5090. **Power/clock limits and supported clock range** (is 200 MHz inside it) — `[NOT FOUND]` (N4). What the repo shows: GPU clocks were locked via `nvidia-smi` at 2000/2000 MHz for all 280 canonical runs (`experiment_config.gpu_min_clock_mhz/gpu_max_clock_mhz`); the one clock-study run was locked at 200/200 and completed, i.e. the driver accepted 200 MHz, which is evidence but not proof that 200 MHz is a supported clock (`results/mbpo/HalfCheetah-v5/seed_0/20260904_130915`, `metadata.json`). Command to run on the research PC: `nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS` and `nvidia-smi --query-gpu=name,driver_version,vbios_version,power.limit,power.default_limit,power.max_limit,clocks.max.graphics --format=csv`.

**Peak FP32 TFLOP/s at 2000 MHz** — `[NOT FOUND]` inputs (SM count, compute capability, FP32 lanes per SM). Formula to apply once `torch.cuda.get_device_properties(0).multi_processor_count` is read: `peak_FP32 [FLOP/s] = n_SM × FP32_lanes_per_SM × 2 [FLOP per FMA] × 2.000e9 [Hz]`; the lanes-per-SM figure for compute capability must be taken from NVIDIA's documentation for the reported capability, not from memory. Not computed.

**FP32 / TF32** — `[UNVERIFIED]` (U2). Repo grep result: no setting of `allow_tf32`, `set_float32_matmul_precision`, `autocast`, `bfloat16`, `float16` or `torch.compile` in any code path (only a docstring mention of `torch.compile` in `algorithms/tdmpc2.py:53` saying it is *not* used). Local default values seen: precision `'highest'`, `matmul.allow_tf32 = False` (torch 2.14.0+cpu, Windows venv only).

**RAM type/speed, mainboard model, storage** — `[NOT FOUND]` (N6). Recorded: RAM total 60.8 GB (CodeCarbon `ram_total_size`); CPU/threads as in the table.

**Commands to run on the research PC** (read-only): `cat /etc/os-release; uname -r; rl-exp/bin/pip freeze | grep -iE "torch|gymnasium|mujoco|nvidia|codecarbon|numpy|pandas"; nvidia-smi; nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS; lscpu; free -h; lsblk -d -o NAME,MODEL; cat /sys/class/dmi/id/board_name /sys/class/dmi/id/board_vendor; rl-exp/bin/python -c "import torch;p=torch.cuda.get_device_properties(0);print(p.name,p.multi_processor_count,p.major,p.minor,torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32)"`.
""")

# ------------------------------------------------------------------ 8
w("""
---

## 8. Pointers to result files (P2)

No recomputation. Sizes in bytes, `rows` = data rows (lines − 1). "Latest `flop_analysis/*.py` change" is commit `050b3c3` (2026-10-08), which only **added** `section_5_2_analysis.py`, `section_5_2_render.py`, `section_5_3_analysis.py`, `section_5_3_render.py`; the pipeline scripts `compute_energy_per_flop.py`, `flop_keys.py`, `measure_flops.py` and `flops_per_call.json` last changed in `1f93741` (2026-09-28), the same commit that last regenerated `flop_analysis/output/*.csv`, so the pipeline CSVs are **not** older than the code that produces them. `correlation_analysis.py` last changed `4247c30` (2026-09-28) and its outputs `4247c30`. Files under `contexts/Section_6*` are untracked (no commit).

| What | Path | Rows | Bytes | Key columns | Commit (date) |
|---|---|---|---|---|---|
| (a) per-config cross-seed energy and J/FLOP, 56 configs | `contexts/section_5.2_data/d1_config_table.csv` | 56 | 32 547 | `algo,env,config,…,n_seeds,total_E_J,total_E_sd_J,total_F,total_J_per_F,total_J_per_F_sd,{rollout,gradient_updates,dynamics_model_update,synthetic_rollout_generation,world_model_pretrain}_{E_J,E_sd_J,F,J_per_F,…}` | `050b3c3` (2026-10-08) |
| (a) compact | `contexts/section_5.3_data/t2_all_config_values.csv` | 56 | 3 562 | `algo,env,config,E_mean,jpf` | `050b3c3` |
| (a) pipeline source, per segment | `flop_analysis/output/cross_seed_energy_per_flop.csv` | 378 | 100 372 | `algo,env_id,architecture_signature,…,segment,flop_type,n_seeds,mean_energy_joules,mean_duration_s,mean_power_w,total_flops,mean_energy_per_flop_j_per_flop` | `1f93741` (2026-09-28) |
| (a) per-run source | `flop_analysis/output/per_run_energy_per_flop.csv` | 2917 | 1 253 232 | `algo,env_id,architecture_signature,seed,run_dir,segment,flop_type,call_count,total_flops,total_energy_joules,duration_s,mean_power_w,energy_per_flop_j_per_flop` | `1f93741` |
| (b) orderings (energy and J/FLOP) | `contexts/section_5.3_data/o1_orderings.csv` | 96 | 12 781 | `env,sweep,sweep_label,metric,rank,algo,config,mean,sd,cv_pct,ratio_to_previous,ordering` | `050b3c3` |
| (b) swaps vs baseline | `contexts/section_5.3_data/o2_swaps_vs_baseline.csv` | 32 | 2 460 | `env,sweep,metric,baseline_ordering,ordering,same_as_baseline,swapped_pairs` | `050b3c3` |
| (b) seed-level orderings | `contexts/section_5.3_data/o3_seed_level.csv` (+ `o3b_pairwise_seed_counts.csv`) | 32 (+96) | 4 842 | `…,seeds_same_as_mean_ordering,seed_331,…,seed_85062` | `050b3c3` |
| (c) env comparison, 27 pairs | `contexts/section_5.4_data/env_comparison_paired.csv` (+ `pooled_summary.csv`, 5 rows) | 27 | 23 779 | `algo,config,E_total_ratio_of_means,E_rollout_ratio_of_means,dshare_*,P_total_*,jpf_total_{HC,Ant}_J_per_FLOP,flops_total_{HC,Ant},git_commits_{HC,Ant},…` | `050b3c3` |
| (d) pooled Spearman + log-log (total) | `contexts/section_5.2_data/d2_total_correlation.csv` | 2 | 581 | `level,env,algo,segment,n,spearman_rho,spearman_p,loglog_slope,loglog_slope_se,loglog_r2,flops_min,flops_max` | `050b3c3` |
| (d) per-algorithm total | `contexts/section_5.2_data/d4_per_algo_total_correlation.csv` | 8 | 1 992 | same | `050b3c3` |
| (d) per-segment (pooled / per-algo) | `d3_segment_correlation.csv` (10 rows), `d5_per_algo_segment_correlation.csv` (22 rows) | — | 2 449 / 5 346 | same | `050b3c3` |
| (d) return-vs-FLOPs and energy-vs-FLOPs with bootstrap CIs | `flop_analysis/output/correlation/correlation_summary.csv` (+ `config_means.csv`, 56 rows, `correlation_plots.html`) | 4 | 708 | `env,analysis,n_configs,spearman_rho,ci_low,ci_high,p_value,loglog_slope,slope_ci_low,slope_ci_high,r2` | `4247c30` (2026-09-28) |
| (e) fixed/marginal fits | `contexts/section_5.2_data/a1_linear_fits.csv` | 144 | 22 702 | `algo,env,sweep,segment,configs,values,n,resid_df,a_J,b_J_per_FLOP,b_pJ_per_FLOP,r2,rse_J,fit_basis` | `050b3c3` |
| (e) delta pairs | `contexts/section_5.2_data/a2_delta_pairs.csv` | 150 | 27 382 | `algo,env,sweep,segment,from_value,to_value,from_config,to_config,E_from_J,E_to_J,dE_J,F_from,F_to,dF,dE_per_dF_pJ,status` | `050b3c3` |
| (f) clock study | `contexts/Section_6_data/C_segment_summary.csv` (24 rows, 4 200 B: `clock,segment,kind,energy_j,…,share_of_training_total,duration_s,mean_power_w,total_flops,…`), `D_ratios_200_over_2000.csv` (9 rows, 2 560 B), `C_consistency_checks.csv`, `C5_whole_run_row.csv`; narrative `contexts/Section_6.md` (427 lines, 53 359 B) | | | | **untracked** (no commit) |

`[NOT FOUND]`: none of the (a)–(f) files is missing. Two caveats: (1) the clock-study files are untracked in git; (2) `contexts/section_5.4_data/env_comparison_paired.csv` records the HC/Ant `git_commits` per pair, and 12 distinct launch commits occur among the canonical runs (item 6), so "same code" across sweeps rests on the `algo_config_diff_HC_vs_Ant`/`experiment_config_diff_HC_vs_Ant` columns there, not on commit equality.
""")

# ------------------------------------------------------------------ 9
bibkeys_all = BK.bibkey.tolist()
uniq = sorted(set(bibkeys_all))
dups = {k: bibkeys_all.count(k) for k in uniq if bibkeys_all.count(k) > 1}
import re
tex = Path(r"C:/Users/saman/Desktop/Thesis/Thesis Report/thesis-main.tex").read_text(encoding="utf8", errors="replace")
cited = set()
BSL = re.escape(chr(92))
for line in tex.splitlines():
    line = re.sub("(?<!" + BSL + ")%.*$", "", line)
    for g in re.findall(BSL + r"(?:[Pp]arencite|[Tt]extcite|[Aa]utocite|[Cc]ite[a-z]*|[Ff]ootcite|[Cc]ites|[Ss]martcite|[Nn]ocite)\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}", line):
        cited.update(k.strip() for k in g.split(",") if k.strip())
uncited = sorted(set(uniq) - cited)
ur = UR.groupby("key").agg(n=("line", "size"), first_line=("line", "min"), chapters=("chapter", lambda x: ", ".join(sorted(set(x))))).reset_index()
w(f"""
---

## 9. Thesis sources and bibliography (P2)

`[FOUND]` The thesis is at `C:\\Users\\saman\\Desktop\\Thesis\\Thesis Report\\` (searched `Desktop`, `Documents`, `Downloads` under the home directory; `Documents` and `Downloads` contain no `.tex`/`.bib`). Files: `thesis-main.tex` (6710 lines, 581 KB, modified 2026-10-11 10:23, **this is the working thesis**: title "Energy Profiling of Reinforcement Learning via Algorithmic Decomposition…"), `bibliography.bib` (modified 2026-10-11 10:11), plus template leftovers `main-english.tex` and `main-english-university-of-hamburg.tex` (unmodified template, dated 2026-09-28), `commands.tex`, `abbreviations.tex`. Also found but unrelated: `Desktop\\Thesis\\Project Proposal\\{{project-proposal.tex, preamble.tex, lit.bib}}`, and two seminar projects under `Desktop\\Uni Stuttgart\\WS25\\` (not analysed). Script `item9_tex.py` (comments after unescaped `%` stripped; reads only `thesis-main.tex` and `bibliography.bib`; nothing in the thesis folder was modified).

Chapters in `thesis-main.tex`: Introduction (l.1019), LaTeX Hints (l.1349, template text), Problem Exposition and Method (l.2089), Per-Algorithm Energy Analysis (l.3123), Cross-Algorithm Comparison (l.5412), GPU Clock Frequency (l.6436), My first appendix (l.6702, template). **No Evaluation chapter exists yet** (`ch:evaluation` is referenced but undefined). Everything lives in one file, so "grouped by file" = one file, grouped by chapter. {len(LB)} labels, no duplicate labels (`item9_labels.csv`).

**`\\label` names by chapter** (all in `thesis-main.tex`):

""")
for ch, s in lab_by_ch.items():
    w(f"- **{ch}** ({(LB.chapter == ch).sum()}): {s}\n")
w(f"""
**Citation keys in `bibliography.bib`:** {len(bibkeys_all)} entries but only **{len(uniq)} unique keys**. The following keys are defined more than once (number of entries): {', '.join(f'`{k}` ({n})' for k, n in sorted(dups.items()))}. Duplicate keys in a biber `.bib` file are an issue the bibliography owner should resolve (biber keeps one and warns). Unique keys (`item9_bibkeys.csv` has all entries):

{', '.join(f'`{k}`' for k in uniq)}

**Cite keys used but not defined (`\\cite`-family, from `thesis-main.tex` vs the bib): none.** Cited unique keys: {len(cited)}. Defined but never cited: {', '.join(f'`{k}`' for k in uncited) or 'none'}.

**`\\zcref` / `\\ref`-family keys used but not defined as a `\\label`** (`item9_undefined_refs.csv` has every occurrence with line number; `#3` is a macro argument in the preamble, `lst:*` are in the template chapter "LaTeX Hints"):

""")
w(md(ur, 0))
w("""
Interpretation-free summary: the real undefined targets are forward references to chapters/sections/appendices not yet written (`ch:evaluation`, `ch:conclusion`, `sec:threats` (21 uses), `app:*`, `sec:related`, `sec:td3-*`, …).
""")

w("""
---

## Final checklist

- Summary block lists every `[NOT FOUND]`, `[UNVERIFIED]`, `[MISMATCH]`: yes (top of file).
- Edits outside `contexts/`: none. Nothing under `results/`, `flop_analysis/`, `configs/` was modified; the thesis folder was only read.
- No `sudo`, no training runs, no clock/governor changes; no `nvidia-smi` calls (the local GPU is not the research GPU).
- Scripts and CSVs: `contexts/section_7_data/` (`build_section_7_md.py` regenerates this file from them).
""")

(D.parent / "section_7.md").write_text("".join(o), encoding="utf8")
print("written", (D.parent / "section_7.md").stat().st_size, "bytes")
