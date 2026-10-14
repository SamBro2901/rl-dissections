"""Builds 02_subinterval_tasks.md from 02_task_duration_stats.csv + 02_task_empirical_checks.json
(produced by task_durations.py). Prose in 2B is hand-written from reading the CodeCarbon 3.3.0 source."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT

S = pd.read_csv(os.path.join(OUT, "02_task_duration_stats.csv"))
C = json.load(open(os.path.join(OUT, "02_task_empirical_checks.json")))
ORDER = {"warmup": 0, "rollout": 1, "synthetic_rollout_generation": 2, "dynamics_model_update": 3,
         "gradient_updates": 4, "world_model_pretrain": 5}


def g(x, p=4):
    return f"{x:.{p}g}"


def table(scope, pooled=False):
    d = S[(S.scope == scope) & ((S.env == "POOLED") if pooled else (S.env != "POOLED"))].copy()
    d["o"] = d.segment.map(ORDER)
    d = d.sort_values(["algo", "env", "o"])
    out = ["| algo | env | segment | n | min | p5 | median | p95 | max | <1 s | <2 s | span dev med / max | span dev >10 % | zero GPU | zero CPU | zero total | median W (tot / GPU) |",
           "|" + "---|" * 17]
    for _, r in d.iterrows():
        out.append(f"| {r.algo} | {r.env} | {r.segment} | {int(r.n_tasks)} | {g(r.dur_min)} | {g(r.dur_p5)} | {g(r.dur_median)} | {g(r.dur_p95)} | {g(r.dur_max)} | "
                   f"{r.frac_lt_1s:.4f} | {r.frac_lt_2s:.4f} | {r.span_dev_median:.4f} / {r.span_dev_max:.4f} | {r.frac_span_dev_gt_10pct:.4f} | "
                   f"{r.frac_zero_gpu:.4f} | {r.frac_zero_cpu:.4f} | {r.frac_zero_total:.4f} | {r.median_total_w:.1f} / {r.median_gpu_w:.1f} |")
    return "\n".join(out)


L = []
P = L.append
P("# Task 2: Sub-interval tasks and CodeCarbon's accounting\n")
P("## Question\n")
P("How long are the per-epoch tasks really (vs. the 1 s polling interval), and what does CodeCarbon 3.3.0 do for tasks shorter than that interval?\n")
P("## Method\n")
P("- 2A script: `contexts/step1/scripts/task_durations.py` (run with the repo `.venv`; canonical list from `common.py`, 280 runs). Data: every task row of each run's `emissions_base_*.csv` except the two idle windows (`duration`, `cpu/gpu/ram/total energy`, `ram_power`). Full-precision output: `02_task_duration_stats.csv` (scopes `all_configs` and `baseline_only`, per algo × env and POOLED over envs), `02_task_empirical_checks.json`, `02_sum_tasks_vs_whole_run.csv`.")
P("- `measure_power_secs` is 1.0 in the `metadata.json` of all 280 runs (`experiment_config.measure_power_secs`); `codecarbon_version` is 3.3.0 in all 280.")
P("- \"span\" = `ram_energy[J] / ram_power[W]` (same proxy as Task 1); span deviation = |span − duration| / duration.")
P("- 2B: read the installed CodeCarbon 3.3.0 source in the repo `.venv` (`.venv/Lib/site-packages/codecarbon/`, the same version the canonical runs recorded; it is the Windows dev venv, the Linux `rl-exp` venv was not accessible from this checkout, so the package contents are assumed identical for the same version number).")
P("- Task row names found: SAC/TD3 `warmup`, `rollout_i`, `gradient_updates_i` (i = 0…99); MBPO additionally `dynamics_model_update_i`, `synthetic_rollout_generation_i`; TD-MPC2 `warmup`, `rollout_i`, `gradient_updates_i`, `world_model_pretrain`.\n")

P("## Results\n")
P("### 2A. Task duration statistics (durations in s)\n")
P("Table A1: **all 280 canonical runs** (all configurations), per algorithm × environment × segment.\n")
P(table("all_configs"))
P("\nTable A2: **baseline configuration only** (each algo × env, 5 seeds; n = 500 per epoch-level segment).\n")
P(table("baseline_only"))
P("\nTable A3: pooled over both environments (all configurations), as the thesis seems to quote medians.\n")
P(table("all_configs", pooled=True))
P("\nTable A4: pooled over both environments, baseline only.\n")
P(table("baseline_only", pooled=True))

P("\n### Cross-checks against the numbers you already have\n")
P("| claim | value reproduced | verdict |")
P("|---|---|---|")
P(f"| SAC rollout median 0.3409 s | {C['sac_rollout_allconfigs_median_pooled']:.4f} s = all 7 SAC configs × 2 envs pooled (n = 7000); baseline-only pooled gives {C['sac_rollout_baseline_median_pooled_envs']:.4f} s; per env baseline: HalfCheetah {C['sac_rollout_baseline_median_HalfCheetah-v5']:.4f}, Ant {C['sac_rollout_baseline_median_Ant-v5']:.4f} | reproduced **only with all configurations pooled, not baseline-only** — if the thesis calls 0.3409 s the \"baseline\" value, that is wrong (baseline pooled = 0.3436 s) |")
P(f"| TD3 rollout median 0.2175 s | {C['td3_rollout_allconfigs_median_pooled']:.4f} s = all 8 TD3 configs × 2 envs pooled (n = 8000); baseline-only pooled {C['td3_rollout_baseline_median_pooled_envs']:.4f} s; per env baseline: HalfCheetah {C['td3_rollout_baseline_median_HalfCheetah-v5']:.4f}, Ant {C['td3_rollout_baseline_median_Ant-v5']:.4f} | same: reproduced only for all configs pooled (baseline pooled = 0.2165 s) |")
P(f"| 8000 MBPO rollout tasks and 8000 synthetic-rollout tasks shorter than the polling interval | rollout: {C['mbpo_rollout_n_lt_1s']} of {C['mbpo_rollout_n']}; synthetic_rollout_generation: {C['mbpo_synthetic_rollout_generation_n_lt_1s']} of {C['mbpo_synthetic_rollout_generation_n']} | match. 8000 = 16 MBPO configurations (7 HalfCheetah + 9 Ant) × 5 seeds × 100 epochs = 80 runs × 100 (all configurations, not baseline: baseline has 1000) |")
P(f"| 720 of 1000 baseline `synthetic_rollout_generation` tasks have zero GPU energy | {C['mbpo_synthetic_rollout_generation_baseline_n_zero_gpu']} of {C['mbpo_synthetic_rollout_generation_baseline_n']} (HalfCheetah and Ant baselines pooled) | match |")
P(f"| \"all rollout tasks are shorter than 1 s: true for SAC, TD3, MBPO, not TD-MPC2\" | SAC {C['sac_rollout_all_configs_n_lt_1s']}/{C['sac_rollout_all_configs_n']} (max {C['sac_rollout_all_configs_max_dur']:.3f} s), MBPO {C['mbpo_rollout_all_configs_n_lt_1s']}/{C['mbpo_rollout_all_configs_n']} (max {C['mbpo_rollout_all_configs_max_dur']:.3f} s), TD3 {C['td3_rollout_all_configs_n_lt_1s']}/{C['td3_rollout_all_configs_n']} (max {C['td3_rollout_all_configs_max_dur']:.3f} s), TD-MPC2 {C['tdmpc2_rollout_all_configs_n_lt_1s']}/{C['tdmpc2_rollout_all_configs_n']} (min 6.6 s, max {C['tdmpc2_rollout_all_configs_max_dur']:.1f} s) | match |")
P("| \"synthetic integration span departs from task duration by up to 28 %\" (Section_8 B11, thesis-only figure) | max span deviation of `synthetic_rollout_generation`: 28.60 % (HalfCheetah), 23.15 % (Ant); median 5.84 % / 1.81 % | order of magnitude confirmed; exact maximum is 28.6 % (HalfCheetah), thesis says 28 % — round down/check wording |")
P("\nAlso notable: of the other segments, the `warmup` task is also sub-second (0.15–0.67 s, TD3-Ant 1.3 s) and `gradient_updates` is ≥ 1.8 s for every algorithm (TD3 ≈ 2 s, SAC/MBPO ≈ 4 s, TD-MPC2 ≈ 12 s); `dynamics_model_update` is ≥ 0.89 s (1 task < 1 s of 8000).\n")

P("### 2B. What CodeCarbon 3.3.0 does for a task shorter than the polling interval\n")
P("Source: `.venv/Lib/site-packages/codecarbon/` (version 3.3.0, equal to `codecarbon_version` in all 280 `metadata.json`). All excerpts are verbatim, trimmed.\n")
P("**Finding 1: the periodic 1 s scheduler is stopped for the whole duration of a task; no periodic energy sample ever falls inside a task window.**\n")
P("`emissions_tracker.py` L750-758 (`start_task`):\n```python\n        # Stop scheduler as we do not want it to interfere with the task measurement\n        if self._scheduler:\n            self._scheduler.stop()\n\n        # Task background thread for measuring power\n        self._scheduler_monitor_power.start()\n```\nThe stopped scheduler is the one running `_measure_power_and_energy` every `measure_power_secs` (`_initialize_scheduler_state`, L345-348). It is not restarted in `stop_task` (the only other `.stop()`/`.start()` calls are in `start` L725 and `stop` L909-911). The scheduler that remains (`_scheduler_monitor_power`, fixed `interval=1`, L349-352) calls `_monitor_power` (L1136-1166), which only appends CPU-power / utilisation samples to history lists; it never computes energy.\n")
P("**Finding 2: a task's energy is a difference of cumulative counters between `start_task` and `stop_task`, with one measurement at stop.**\n")
P("`emissions_tracker.py` L767-778 (`start_task`):\n```python\n        self._last_measured_time = self._start_time = time.perf_counter()\n        ...\n        # Read initial energy for hardware\n        for hardware in self._hardware:\n            hardware.start()\n        prepared_data_for_task_start = self._prepare_emissions_data()\n```\n`emissions_tracker.py` L805-831 (`stop_task`):\n```python\n        self._measure_power_and_energy()\n        emissions_data = (\n            self._prepare_emissions_data()\n        )  # This is emissions_data_at_stop\n        ...\n            emissions_data_delta = dataclasses.replace(emissions_data)\n            emissions_data_delta.compute_delta_emission(\n                self._active_task_emissions_at_start\n            )\n```\n`stop_task` therefore triggers a **fresh measurement at stop time** (L808), and the task value is `cumulative_at_stop − cumulative_at_start` (`emissions_data.py` L55-64: `self.gpu_energy -= previous_emission.gpu_energy`, etc.). The cumulative `_total_*_energy` only grows inside `_do_measurements` (L1195 for GPU: `self._total_gpu_energy += energy`). `hardware.start()` at task start re-baselines each counter, so energy between the previous `stop_task` and this `start_task` is attributed to nothing (and is also excluded from the run total, see the empirical check below).\n")
P("**Finding 3: per component.** `last_duration` passed to every component is `time.perf_counter() − self._last_measured_time` (`_do_measurements` L1168-1173), i.e. time since `start_task` L767.\n")
P("- **GPU** — measured, not sampled: `gpu_device.py` L47-56 (`delta`): `new_last_energy = energy = self._get_energy_kwh()` then `self.energy_delta = energy - self.last_energy`, where `_get_energy_kwh` converts `pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)` (mJ, `gpu_nvidia.py` L49-54). Energy = NVML cumulative-counter difference between the `hardware.start()` read and the read at stop. The power field is derived afterwards from that energy and `last_duration` (`Power.from_energies_and_delay`, `units.py` L126-141). Whether the NVML counter itself advances continuously is **not** in CodeCarbon's source — see the empirical part.")
P("- **CPU (RAPL)** — measured, not sampled: `hardware.py` L405-410 (`CPU.measure_power_and_energy`, mode `intel_rapl`): `energy = self._get_energy_from_cpus(delay=...)`, which sums `RAPLFile.delta` (`rapl.py` L70-83: current counter minus `last_energy`, wrap-around handled). Counter difference between start and stop; no interpolation.")
P("- **RAM** — modelled, not measured: `BaseHardware.measure_power_and_energy` (`hardware.py` L50-59): `power = self.total_power(); energy = Energy.from_power_and_time(power=power, time=Time.from_seconds(last_duration))`. RAM power is a constant (20 W in all runs' `ram_power` column; `RAM.total_power`, `ram.py` L312), so RAM energy = 20 W × `last_duration`. This is why RAM energy ÷ RAM power recovers the integration window (Task 1).\n")
P("**Finding 4: no sample inside the task window.** Nothing happens periodically inside a task (Finding 1), so the question \"what if no periodic sample falls inside the window\" does not arise for energy: the energy of a 0.2 s task is the counter difference over that 0.2 s window, exactly as for a 12 s task. The 1 s polling interval does **not** bound the resolution of per-task energy in CodeCarbon's task mode. The effective resolution is set by the hardware counters (RAPL µJ counter; NVML energy counter — update behaviour not documented in the source) and by the start/stop overheads (below).\n")
P("**Finding 5: which timestamps define duration and span.**\n- `duration` (reported) = `time.perf_counter() − Task.start_time` (`emissions_tracker.py` L836-838); `Task.start_time` is set in `Task.__init__` (`external/task.py`: `self.start_time = time.perf_counter()`), i.e. **after** `hardware.start()` and `_prepare_emissions_data()` of `start_task`, and the end timestamp is taken **after** `_measure_power_and_energy()` and `_prepare_emissions_data()` in `stop_task`.\n- the integration window of the counters / of RAM energy is from `hardware.start()` (after L767) to the measurement in `_measure_power_and_energy` (L808). The two windows differ by the start/stop bookkeeping (sub-millisecond: median span − duration is −0.0003 s for every duration class in `02_task_empirical_checks.json`, but up to 28 % of the duration for 6 ms tasks).\n- The idle head task is the exception (+3.08 s span): `start_task` L777-778 runs `hardware.start()` and then `_prepare_emissions_data()`, which calls `_update_emissions()` → `_ensure_geo_metadata()` (L962, L378-388) — a lazy first-use geolocation lookup — for the first time in the run. This is **inferred from the source order** (the lookup happens after the counters were baselined and the lookup time is not measured by this script); it fits the observed ≈3.08 s excess and the fact that no later task shows it.\n")

P("**Conclusion (plain language).**\n")
P("1. Established from source: CodeCarbon's task mode does not poll during a task. Each task's energy is the difference of the hardware energy counters (NVML for the GPU, RAPL for the CPU) between `start_task` and `stop_task`, plus modelled RAM power (constant 20 W) × elapsed time. A 0.2–0.4 s rollout task is therefore *not* \"zero or one polling sample\"; it is a counter difference over its own window.")
P("2. Established from source: the energy between `stop_task(i)` and `start_task(i+1)` is attributed to no task and is not in the run total either (counters are re-baselined by `hardware.start()`); the whole-run cumulative energy equals the sum of the task energies to within 0.0125 % (max over 280 runs, median 8.5e-5 %, `02_sum_tasks_vs_whole_run.csv`) — this is an accounting identity, not an independent validation.")
P("3. Established from data, mechanism inferred: the GPU energy of sub-second tasks is **quantised**. For MBPO `synthetic_rollout_generation` on HalfCheetah (median duration 6 ms) 3283 of 3500 tasks have exactly 0 J GPU energy, and the 217 non-zero ones are all between 5.86 and 6.92 J (median 6.36 J), independent of the task duration; on Ant (68 ms median for non-zero tasks) the non-zero values range 5.6–24 J. 217/3500 = 6.2 % ≈ 6.1 ms task / ≈ 100 ms: consistent with an NVML energy counter that advances in lumps of ≈6 J roughly every 100 ms (≈ 60-65 W × 0.1 s). This is **inferred**, not documented in CodeCarbon's source or checked against NVIDIA documentation here; `nvml_poll.py` (2C) would verify it.")
P("4. Consequence (inferred): for a task of 0.2–0.4 s (SAC/TD3/MBPO rollouts, ≈ 2-4 counter updates) the recorded GPU energy is a staircase value, i.e. it can be off by about ± one lump (≈ ±6 J out of ≈ 14-25 J, so ≈ ±25-40 % per task for 0.1-0.3 s rollouts), in either direction; for tasks of a few ms (synthetic rollout generation) the value is either 0 or one whole lump, i.e. off by up to ±100 % or more per task. Over many tasks the sum is plausibly unbiased (the probability of catching a lump scales with task duration: pooled mean GPU power of the HalfCheetah synthetic tasks is 62.9 W, Ant 86.1 W, vs 68-82 W mean for rollouts), but this is only reasoned from the data, not tested. CPU (RAPL µJ counter) and RAM (model) do not show zero energy for any task (zero fraction 0.0000 everywhere in table A1).")
P("5. Per-task power implausibilities in the data: for tasks < 1 s, GPU energy ÷ duration has p1 = p5 = 0 W, p99 = 618 W, max = 1214.7 W (1.2 kW is far above any plausible board power of this GPU; the card power limit itself was not checked here, see Task 3), all from `synthetic_rollout_generation`; for tasks ≥ 2 s the GPU power is 63.6-180.7 W (p1-max). So extreme per-task power values arise only in the millisecond-scale tasks.\n")

P("### 2B step 4: empirical check against the data\n")
P(f"- Sub-second tasks with non-zero GPU energy: {C['subsecond_gpu_tasks_nonzero']} of {C['subsecond_tasks_n']} sub-second tasks; {C['subsecond_gpu_mJ_distinct_values']} distinct energy values (not on a coarse grid at the mJ level: the counter itself is mJ-resolution, the lumps are not).")
P(f"- Sub-second total power (energy ÷ duration, W), quantiles p1 / p5 / p50 / p95 / p99 = {', '.join(f'{x:.1f}' for x in C['subsecond_total_w_quantiles_p1_p5_p50_p95_p99'])}; GPU power p1 / p5 / p50 / p95 / p99 / max = {', '.join(f'{x:.1f}' for x in C['subsecond_gpu_w_quantiles_p1_p5_p50_p95_p99_max'])}; for tasks ≥ 2 s: {', '.join(f'{x:.1f}' for x in C['long_tasks_ge2s_gpu_w_quantiles_p1_p5_p50_p95_p99_max'])}.")
P("- Rollout tasks (0.1-0.36 s): none has zero GPU energy (0 of 7000 SAC, 0 of 8000 TD3, 0 of 8000 MBPO), median GPU power 65-72 W, coefficient of variation of per-task GPU power 16 % (SAC) to 36 % (TD3 HalfCheetah, the shortest rollouts at 0.108 s) — the scatter grows as the task gets shorter, as expected for lumpy counters. Not a proof, but consistent.")
P("- Sum of task energies vs. the whole measured run (`emissions.csv` last row = cumulative energy since `tracker.start()`): relative difference median 8.5e-5 %, max 0.0125 % over 280 runs (per component all < 0.03 %). As explained in Finding 2 this is an identity of CodeCarbon's accounting and says nothing about the quality of the per-task values. What is *not* in either number is the (unmeasured) energy between tasks.\n")

P("### 2C. Optional follow-up experiments (NOT RUN — they need the research PC)\n")
P("1. `contexts/step1/scripts/sync_timing_run.py` — one baseline run per algorithm with `torch.cuda.synchronize()` before every `perf_counter()` call in `algorithms/{sac,td3,mbpo,tdmpc2}.py` (by replacing the module-level `time` name with a shim; no repo file is changed), then prints ratio = Σ sub-phase wall times / Σ measured `gradient_updates` durations, also for a canonical baseline run (`--compare-dir`). Commands are in the script header. Run on a quiet machine with clocks locked as in the protocol (sudo).")
P("2. `contexts/step1/scripts/nvml_poll.py` — standalone ~1 kHz poller of `nvmlDeviceGetTotalEnergyConsumption`, logs every counter change and prints a histogram of the spacing between updates; run it in parallel with a short SAC baseline run (commands in the header). Expected if the inference above is right: spacing clustered near 100 ms with ≈6 J steps.\n")

P("## Caveats / things that look odd\n")
P("- 2B was done on the Windows dev venv's CodeCarbon 3.3.0 install, not the Linux `rl-exp` venv; same version number as recorded in `metadata.json`, but package contents on the Linux box were not inspected.")
P("- The NVML-lump explanation and the \"≈100 ms\" figure come from the data only; they are not documented in the CodeCarbon source and were not checked against NVIDIA documentation or a live counter.")
P("- The 3.08 s head excess is attributed to the first-use geolocation lookup by reading the source order; not measured.")
P("- Thesis medians 0.3409 / 0.2175 s correspond to all-configuration pooling, not baseline-only pooling (0.3436 / 0.2165 s) — check the wording \"baseline\" in the thesis.")
P("- `warmup` is a task too (35-40 per algo-env group, one per run); TD3 Ant warmup (10000 steps) is 1.3 s, all other warmups are < 1 s.")
open(os.path.join(OUT, "02_subinterval_tasks.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("ok")
