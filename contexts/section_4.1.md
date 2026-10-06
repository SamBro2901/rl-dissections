# Section 4.1 context — SAC per-algorithm energy analysis

This file is the complete data context for thesis Section 4.1 (SAC per-algorithm energy analysis). It covers the 70 canonical SAC runs (7 configurations × 2 environments × 5 seeds [331, 958, 14577, 43611, 85062]) recorded on the Linux experiment box (RTX 5090, clocks requested at 2000 MHz, torch 2.13.0+cu130, CodeCarbon 3.3.0) and was generated on the Windows dev checkout from the synced `results/` and `flop_analysis/` directories by `contexts/section_4.1_data/build_section_4_1.py` (read-only on data). All energy is gross; `rollout` and `gradient_updates` are measured, the four sub-segments are allocated by wall-clock share. Headline: HC-base TOTAL_MEASURED_TRAINING energy 60910 ± 254.1 J, Ant-base 64380 ± 588.3 J; rollout share HC 4.097 ± 0.05304 %, Ant 6.729 ± 0.03840 %. 35 checks were run, 0 FAILED; all recomputed values reconcile with the pipeline CSVs (Part 4.8). Discrepancies and open items are in Part 11. Nothing here is interpreted; 'by construction' notes are confined to Parts 2 and 5.

**Contents**
- Schema samples (S1–S6)
- Part 0 — Provenance
- Part 1 — Audit of the SAC canonical run set
- Part 2 — SAC implementation facts
- Part 3 — FLOP facts for SAC
- Part 4 — Per-configuration results (Tables 4.1–4.8)
- Part 5 — Environment comparison (HalfCheetah-v5 vs Ant-v5)
- Part 6 — Effect of each sweep (baseline-relative)
- Part 7 — Energy composition and idle-floor diagnostics
- Part 8 — Measurement-quality diagnostics
- Part 9 — Learning-performance sanity check
- Part 10 — Optional backlog (not produced in this pass)
- Part 11 — Discrepancies, open questions, things not found
- Part 12 — Index of produced files

**Conventions used throughout.** Tags: `HC` = HalfCheetah-v5, `Ant` = Ant-v5; `base` = hidden (1024,1024), batch 256,
UTD 1; `utd2`/`utd4`, `w256`/`w512`, `b512`/`b1024` change exactly that one factor. Short segment names in tables:
`buf_sample` = buffer_sample, `critic` = critic_update, `actor` = actor_update, `target` = target_update,
`grad_updates`/`GU` = gradient_updates (= Σ of the four allocated), `TOTAL` = TOTAL_MEASURED_TRAINING (= rollout +
gradient_updates). Unless a table says otherwise: n = 5 seeds, mean ± sample sd (ddof = 1), 4 significant digits,
gross energy (idle floor included, never subtracted), 1 kWh = 3.6e6 J, J/FLOP = energy per **matmul** FLOP as a ratio
of means; `target` is reported in J per Polyak elementwise op, never as J/FLOP; `buffer_sample` has no FLOPs.
Companion CSVs carry full precision.

## Schema samples

**S1. `segment_energy.json`** — complete file of `results/sac/HalfCheetah-v5/seed_331/20260907_121335` (HC-base, seed 331):

```json
{
  "idle_baseline_head": 0.0016559092599871344,
  "warmup": 5.199447985954975e-06,
  "rollout": 0.0006883597055185396,
  "buffer_sample": 0.0004296243175465504,
  "critic_update": 0.006866485174144906,
  "actor_update": 0.008239057547029834,
  "target_update": 0.0006991743689062385,
  "_sub_segment_wall_time_seconds": {
    "buffer_sample": 10.441313098989895,
    "critic_update": 166.87863946400296,
    "actor_update": 200.23675563898462,
    "target_update": 16.992284185002404
  },
  "idle_baseline_tail": 0.0017299364773605413,
  "_total_kg_co2eq": "<CO2e value omitted: CO2e is not reported in this file>"
}
```
Key list shared by all 70 runs (identical, in this order):


| key | unit | meaning |
|---|---|---|
| `idle_baseline_head` | kWh | measured CodeCarbon task, 90 s idle before warmup (excluded from totals) |
| `warmup` | kWh | measured task `warmup_0`, 5000 random-action env steps (excluded from totals) |
| `rollout` | kWh | **measured**: Σ of the 100 `rollout_i` tasks |
| `buffer_sample` | kWh | **allocated**: Σ `gradient_updates_i` energy × T_buffer_sample/ΣT |
| `critic_update` | kWh | **allocated** (same rule) |
| `actor_update` | kWh | **allocated** (same rule) |
| `target_update` | kWh | **allocated** (same rule) |
| `_sub_segment_wall_time_seconds` | s | `_`-prefixed bookkeeping: Σ perf_counter times of the 4 sub-phases over the run |
| `idle_baseline_tail` | kWh | measured task, 90 s idle after the last epoch (excluded) |
| `_total_kg_co2eq` | kg CO₂e | `_`-prefixed bookkeeping: tracker.stop() return value (CO₂e, not used anywhere in this file) |

There is no `gradient_updates` key: the measured fused-task energy is replaced by its 4-way allocation (it equals the sum of the 4 allocated keys, Part 3.5).

**S2. `metadata.json`** — top-level key structure (types), then values of the relevant blocks for the same run (the complete verbatim files of one HC-base and one Ant-base run are in Part 2.1):

```json
{
  "algo_name": "str",
  "env_id": "str",
  "seed": "int",
  "git_commit": "str",
  "torch_version": "str",
  "device": "str",
  "cuda_device_name": "str",
  "start_time_utc": "str",
  "experiment_config": {
    "algo_name": "str",
    "env_id": "str",
    "seed": "int",
    "settle_seconds": "float",
    "idle_baseline_seconds": "float",
    "idle_tail_seconds": "float",
    "warmup_steps": "int",
    "train_steps": "int",
    "steps_per_epoch": "int",
    "lock_gpu_clocks": "bool",
    "gpu_min_clock_mhz": "int",
    "gpu_max_clock_mhz": "int",
    "set_persistence_mode": "bool",
    "set_cpu_performance_governor": "bool",
    "thermal_gate_enabled": "bool",
    "thermal_gate_temp_tolerance_c": "float",
    "thermal_gate_power_tolerance_w": "float",
    "thermal_gate_max_wait_seconds": "float",
    "thermal_gate_poll_interval_seconds": "float",
    "thermal_gate_reference_file": "str",
    "measure_power_secs": "float",
    "tracking_mode": "str",
    "output_dir": "str",
    "country_iso_code": "str",
    "force_cpu_power_w": "NoneType",
    "device": "str"
  },
  "algo_config": {
    "hidden_sizes": "list",
    "actor_lr": "float",
    "critic_lr": "float",
    "alpha_lr": "float",
    "gamma": "float",
    "tau": "float",
    "batch_size": "int",
    "buffer_capacity": "int",
    "target_entropy": "NoneType",
    "autotune_alpha": "bool",
    "init_alpha": "float",
    "updates_per_env_step": "int",
    "policy_update_delay": "int"
  },
  "codecarbon_version": "str",
  "thermal_gate": {
    "gated": "bool",
    "reason": "str",
    "waited_seconds": "float",
    "final_temp_c": "float",
    "final_power_w": "float"
  },
  "end_time_utc": "str"
}
```
```json
{
  "git_commit": "f23f448480ecdf1dd6dc552b45f9194b2602bc11",
  "torch_version": "2.13.0+cu130",
  "codecarbon_version": "3.3.0",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "device": "cuda",
  "start_time_utc": "2026-09-07T10:13:36.036705",
  "end_time_utc": "2026-09-07T10:24:28.627367",
  "experiment_config.gpu_min_clock_mhz": 2000,
  "experiment_config.gpu_max_clock_mhz": 2000,
  "algo_config": {
    "hidden_sizes": [
      1024,
      1024
    ],
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "alpha_lr": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_capacity": 1000000,
    "target_entropy": null,
    "autotune_alpha": true,
    "init_alpha": 0.2,
    "updates_per_env_step": 1,
    "policy_update_delay": 1
  },
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 1.1682510375976562e-05,
    "final_temp_c": 45.0,
    "final_power_w": 26.94
  }
}
```

**S3. CodeCarbon CSVs.** Two files per run: `emissions.csv` (one row, written by `tracker.stop()`) and the **per-task log** `emissions_<experiment>_<run_id>.csv` (here `emissions_base_2ab8fba5-bfcc-4267-8cf2-8fe3bddfe83b.csv`), which holds one row per task (`idle_baseline_head`, `warmup_0`, `rollout_0…99`, `gradient_updates_0…99`, `idle_baseline_tail`) and is the file used for every task-level number in this document (the pipeline reads the same file). Per-task CSV columns and pandas dtypes:

```
task_name: str
timestamp: str
project_name: str
run_id: str
duration: float64
emissions: float64
emissions_rate: float64
cpu_power: float64
gpu_power: float64
ram_power: float64
cpu_energy: float64
gpu_energy: float64
ram_energy: float64
energy_consumed: float64
water_consumed: float64
country_name: str
country_iso_code: str
region: str
cloud_provider: float64
cloud_region: float64
os: str
python_version: str
codecarbon_version: str
cpu_count: int64
cpu_model: str
gpu_count: int64
gpu_model: str
longitude: float64
latitude: float64
ram_total_size: float64
tracking_mode: str
cpu_utilization_percent: float64
gpu_utilization_percent: float64
ram_utilization_percent: float64
ram_used_gb: float64
on_cloud: str
```
First 3 rows (all columns):

```
task_name,timestamp,project_name,run_id,duration,emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed,water_consumed,country_name,country_iso_code,region,cloud_provider,cloud_region,os,python_version,codecarbon_version,cpu_count,cpu_model,gpu_count,gpu_model,longitude,latitude,ram_total_size,tracking_mode,cpu_utilization_percent,gpu_utilization_percent,ram_utilization_percent,ram_used_gb,on_cloud
idle_baseline_head,2026-09-07T12:15:51,sac_HalfCheetah-v5_seed331,2ab8fba5-bfcc-4267-8cf2-8fe3bddfe83b,90.00358154399999,<CO2e omitted>,<CO2e omitted>,10.033112075295069,13.627352604117268,20.0,0.0004220009223226,0.0007116714026699,0.0005222369349944,0.0016559092599871,0.0,Germany,DEU,baden-wurttemberg,,,Linux-7.0.0-31-generic-x86_64-with-glibc2.43,3.14.4,3.3.0,24,Intel(R) Core(TM) Ultra 9 285K,1,1 x NVIDIA GeForce RTX 5090,9.1827,48.767,60.81614303588867,machine,0.0,0.0,0.0,0.0,N
warmup_0,2026-09-07T12:15:52,sac_HalfCheetah-v5_seed331,2ab8fba5-bfcc-4267-8cf2-8fe3bddfe83b,0.1516346260000318,<CO2e omitted>,<CO2e omitted>,18.53704326539932,31.546017986185635,20.0,1.5191114930658209e-06,2.838335604000118e-06,8.42000888889036e-07,5.199447985954975e-06,0.0,Germany,DEU,baden-wurttemberg,,,Linux-7.0.0-31-generic-x86_64-with-glibc2.43,3.14.4,3.3.0,24,Intel(R) Core(TM) Ultra 9 285K,1,1 x NVIDIA GeForce RTX 5090,9.1827,48.767,60.81614303588867,machine,0.0,0.0,0.0,0.0,N
rollout_0,2026-09-07T12:15:52,sac_HalfCheetah-v5_seed331,2ab8fba5-bfcc-4267-8cf2-8fe3bddfe83b,0.3697663079999529,<CO2e omitted>,<CO2e omitted>,21.091142564121203,39.44992170926282,20.0,2.9690668196958453e-06,6.4811162959997615e-06,2.0517128000001068e-06,1.1501895915695714e-05,0.0,Germany,DEU,baden-wurttemberg,,,Linux-7.0.0-31-generic-x86_64-with-glibc2.43,3.14.4,3.3.0,24,Intel(R) Core(TM) Ultra 9 285K,1,1 x NVIDIA GeForce RTX 5090,9.1827,48.767,60.81614303588867,machine,0.0,0.0,0.0,0.0,N
```
`emissions.csv` (top-level) columns/dtypes and its single row:

```
timestamp: str
project_name: str
run_id: str
experiment_id: str
duration: float64
emissions: float64
emissions_rate: float64
cpu_power: float64
gpu_power: float64
ram_power: float64
cpu_energy: float64
gpu_energy: float64
ram_energy: float64
energy_consumed: float64
water_consumed: float64
country_name: str
country_iso_code: str
region: str
cloud_provider: float64
cloud_region: float64
os: str
python_version: str
codecarbon_version: str
cpu_count: int64
cpu_model: str
gpu_count: int64
gpu_model: str
longitude: float64
latitude: float64
ram_total_size: float64
tracking_mode: str
cpu_utilization_percent: float64
gpu_utilization_percent: float64
ram_utilization_percent: float64
ram_used_gb: float64
on_cloud: str
pue: float64
wue: float64
```
```
timestamp,project_name,run_id,experiment_id,duration,emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed,water_consumed,country_name,country_iso_code,region,cloud_provider,cloud_region,os,python_version,codecarbon_version,cpu_count,cpu_model,gpu_count,gpu_model,longitude,latitude,ram_total_size,tracking_mode,cpu_utilization_percent,gpu_utilization_percent,ram_utilization_percent,ram_used_gb,on_cloud,pue,wue
2026-09-07T12:24:28,sac_HalfCheetah-v5_seed331,2ab8fba5-bfcc-4267-8cf2-8fe3bddfe83b,5b0fa12a-3dd7-45bb-9766-cc326314d9f1,90.00570463899999,<CO2e omitted>,<CO2e omitted>,29.313081311565508,84.68412374636605,20.0,0.0038037780288533,0.0131223574423219,0.0033876408256166,0.020313776296792,0.0,Germany,DEU,baden-wurttemberg,,,Linux-7.0.0-31-generic-x86_64-with-glibc2.43,3.14.4,3.3.0,24,Intel(R) Core(TM) Ultra 9 285K,1,1 x NVIDIA GeForce RTX 5090,9.1827,48.767,60.81614303588867,machine,0.1707865168539326,0.0,12.634831460674157,7.694958376080803,N,1.0,0.0
```
(Values of the CO₂e columns `emissions` and `emissions_rate` are replaced by a placeholder in these samples; no CO₂e figure is used anywhere in this file.)

**S4. Pipeline CSVs.** `per_run_energy_per_flop.csv` columns, dtypes, meaning (from `compute_energy_per_flop.py`):


| column | dtype | meaning |
|---|---|---|
| `algo` | str | algorithm |
| `env_id` | str | Gymnasium env id |
| `architecture_signature` | str | flop_keys.signature() (hidden sizes + batch for SAC) |
| `hidden_sizes` | str | 'h1xh2' |
| `updates_per_env_step` | int64 | UTD (separate group key) |
| `mbpo_rollout_regime` | str | MBPO only (empty for SAC) |
| `seed` | int64 | run seed |
| `run_dir` | str | run directory relative to repo root |
| `included_in_cross_seed_avg` | bool | True iff seed in the canonical set |
| `segment` | str | segment name (incl. TOTAL_MEASURED_TRAINING) |
| `flop_type` | str | matmul \| elementwise (target_update) \| none \| mixed_total |
| `call_count` | float64 | derived calls (Part 3.2) |
| `total_flops` | float64 | per-call constant × call_count (target: op count) |
| `total_energy_kwh` | float64 | segment_energy.json value (TOTAL: sum) |
| `total_energy_joules` | float64 | × 3.6e6 |
| `duration_s` | float64 | measured task duration (allocated: perf_counter time; TOTAL: rollout+GU tasks) |
| `mean_power_w` | float64 | (cpu+gpu+ram energy)/duration |
| `mean_cpu_power_w` | float64 | CPU share of the above |
| `mean_gpu_power_w` | float64 | GPU share |
| `mean_ram_power_w` | float64 | RAM share |
| `energy_per_flop_j_per_flop` | float64 | total_energy_joules / total_flops (target_update: J per op) |
| `note` | str | free text |
| `dynamics_train_flops` | float64 | MBPO only |
| `dynamics_holdout_flops` | float64 | MBPO only |


Rows of `results/sac/HalfCheetah-v5/seed_331/20260907_121335` (`note` column omitted):

```
algo,env_id,architecture_signature,hidden_sizes,updates_per_env_step,mbpo_rollout_regime,seed,run_dir,included_in_cross_seed_avg,segment,flop_type,call_count,total_flops,total_energy_kwh,total_energy_joules,duration_s,mean_power_w,mean_cpu_power_w,mean_gpu_power_w,mean_ram_power_w,energy_per_flop_j_per_flop,dynamics_train_flops,dynamics_holdout_flops
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,rollout,matmul,100000.0,215654400000.0,0.0006883597055185,2478.0949398667426,20.244417399000213,122.4088049078201,30.36312607232672,72.06930871085926,19.976370124634094,1.1491047434537588e-08,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,buffer_sample,none,100000.0,0.0,0.0004296243175465,1546.6475431675815,10.441313098989896,148.12768552235127,25.58564640795036,101.9922087476625,20.5498303667384,,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,critic_update,matmul,100000.0,492358860800000.0,0.0068664851741449,24719.346626921662,166.87863946400296,148.12768552235127,25.58564640795036,101.99220874766252,20.5498303667384,5.020595462983422e-11,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,actor_update,matmul,100000.0,384512819200000.0,0.0082390575470298,29660.6071693074,200.23675563898465,148.12768552235124,25.58564640795036,101.9922087476625,20.5498303667384,7.713814907658455e-11,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,target_update,elementwise,100000.0,430080400000.0,0.0006991743689062,2517.027728062458,16.992284185002404,148.12768552235127,25.58564640795036,101.99220874766252,20.5498303667384,5.852458582308001e-09,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,warmup,none,,,5.199447985954975e-06,18.71801274943791,0.1516346260000318,123.4415465860283,36.065650170402414,67.38571818285277,19.99017823277312,,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,idle_baseline_head,none,,,0.0016559092599871,5961.273335953683,90.00358154399999,66.23373463243126,16.879365179695572,28.465723315238385,20.888646137497314,,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,idle_baseline_tail,none,,,0.0017299364773605,6227.771318497948,90.00354786399998,69.19473138890524,16.214900969494927,32.98011517440727,19.99971524500305,,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,331,results/sac/HalfCheetah-v5/seed_331/20260907_121335,True,TOTAL_MEASURED_TRAINING,mixed_total,,877087334400000.0,0.016922701113146,60921.72400732584,425.6760496229996,143.1175751167612,25.15874411317684,97.96164101044532,19.997189993139045,6.945913094161977e-11,,
```
`cross_seed_energy_per_flop.csv` columns, dtypes, meaning:


| column | dtype | meaning |
|---|---|---|
| `algo` | str | algorithm |
| `env_id` | str | Gymnasium env id |
| `architecture_signature` | str | flop_keys.signature() (hidden sizes + batch for SAC) |
| `hidden_sizes` | str | 'h1xh2' |
| `updates_per_env_step` | int64 | UTD (separate group key) |
| `mbpo_rollout_regime` | str | MBPO only (empty for SAC) |
| `segment` | str | segment name (incl. TOTAL_MEASURED_TRAINING) |
| `flop_type` | str | matmul \| elementwise (target_update) \| none \| mixed_total |
| `n_seeds` | int64 | number of runs averaged |
| `mean_energy_kwh` | float64 | mean over seeds |
| `mean_energy_joules` | float64 | × 3.6e6 |
| `mean_duration_s` | float64 | mean over seeds |
| `mean_power_w` | float64 | MEAN of per-seed powers (not ratio of means) |
| `mean_cpu_power_w` | float64 | mean of per-seed values |
| `mean_gpu_power_w` | float64 | mean of per-seed values |
| `mean_ram_power_w` | float64 | mean of per-seed values |
| `total_flops` | float64 | mean of non-zero per-seed FLOPs |
| `mean_energy_per_flop_j_per_flop` | float64 | mean energy J / mean FLOPs (ratio of means) |
| `dynamics_train_flops` | float64 | MBPO only |
| `dynamics_holdout_flops` | float64 | MBPO only |


Rows for HC-base:

```
algo,env_id,architecture_signature,hidden_sizes,updates_per_env_step,mbpo_rollout_regime,segment,flop_type,n_seeds,mean_energy_kwh,mean_energy_joules,mean_duration_s,mean_power_w,mean_cpu_power_w,mean_gpu_power_w,mean_ram_power_w,total_flops,mean_energy_per_flop_j_per_flop,dynamics_train_flops,dynamics_holdout_flops
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,TOTAL_MEASURED_TRAINING,mixed_total,5,0.0169197292222799,60911.02520020765,425.3428092093994,143.2052861562394,25.625758618618853,97.58227199882649,19.99725553879407,877087334400000.0,6.944693283237959e-11,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,actor_update,matmul,5,0.00826206445271,29743.432029756008,200.50247887203813,148.34452060721915,26.08008378524796,101.70351356529838,20.56092325667281,384512819200000.0,7.735355115504041e-11,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,buffer_sample,none,5,0.0004306037615091,1550.17354143308,10.449852958206066,148.34452060721918,26.08008378524796,101.70351356529838,20.56092325667281,0.0,,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,critic_update,matmul,5,0.0068264137224087,24575.089400671408,165.66446918999708,148.34452060721918,26.08008378524796,101.70351356529838,20.56092325667281,492358860800000.0,4.9912962591433895e-11,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,rollout,matmul,5,0.0006932510325253,2495.703717091418,20.477446562599837,121.87601002848469,30.7585836919795,71.13778091606947,19.979645420435723,215654400000.0,1.1572700195736413e-08,,
sac,HalfCheetah-v5,bs256_h1024x1024,1024x1024,1,,target_update,elementwise,5,0.0007073962531265,2546.626511255743,17.16697405005168,148.34452060721918,26.08008378524796,101.70351356529838,20.56092325667281,430080400000.0,5.921280093805119e-09,,
```

**S5. `training_metrics.json` keys:** top level `['episodes', 'epochs']`; each episode `['episode_idx', 'phase', 'epoch', 'env_step', 'return', 'length']`; each epoch `['epoch', 'env_step', 'cumulative_reward', 'epoch_reward_sum', 'epoch_reward_mean_per_step', 'num_episodes_completed', 'mean_episode_return', 'critic_loss_mean', 'actor_loss_mean', 'alpha_end', 'buffer_size']`.

**S6. Failure-mode strings** searched (case-insensitive substring) in every `run.log` and in every available sweep-log block (sources: README 'Known limitations', `utils/gpu_control.py`, `utils/thermal_gate.py`, `experiment_runner.py`, and CodeCarbon 3.3.0 `core/cpu.py`, `core/rapl.py`, `core/resource_tracker.py`, `external/geography.py` in the local venv):

```
default power consumption of 4 W per thread
Resorting to a default power consumption
No CPU tracking mode found
Unable to read RAPL value
Unable to read max_energy_range_uj
Unable to access geographical location
Using 'Canada' as the default value
Thermal gate timed out
Thermal reference did not stabilize
No NVML reference available
NVML read failed
Command failed (
Command not found
Command timed out
Could not query GPU clock range
Could not read current CPU governor
CUDA requested but not available
Permission denied
permission error
Insufficient Permissions
```

## Part 0 — Provenance


| Item | Value |
|---|---|
| Generated (local time) | 2026-10-05T18:51:12 |
| Machine / OS | Windows 11 (10.0.26200), AMD64 — this is the **Windows dev checkout**, not the Linux experiment box |
| Repo path | C:\Users\saman\Desktop\Thesis\Project Repository\rl-dissections |
| Python / pandas / numpy | 3.13.9 / 3.0.5 / 2.5.2 |
| torch / gymnasium / mujoco (local venv, used only for Part 2.4 instantiation) | 2.14.0+cpu / 1.3.0 / 3.12.0 |
| git HEAD | 57b3d21c577f0aee149ef0106bb9c3bbda7d1eb6 |
| branch | master |
| git status --porcelain | DIRTY, see below |

```
?? contexts/
```
(The only untracked/changed paths should be under `contexts/`, i.e. this generator's own output.)

**Pipeline files** (mtime = filesystem modification time on this checkout, which reflects git checkout/pull time, not the time the file was generated on the Linux box; the last-commit column is the more meaningful timestamp):


| File | SHA-256 | mtime (local) | last git commit touching it |
|---|---|---|---|
| `flop_analysis/output/per_run_energy_per_flop.csv` | 6e7f8f250a797fe3cca22c85a22e0e93897a91f1e0a5af5d575a4067ecc6d748 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |
| `flop_analysis/output/cross_seed_energy_per_flop.csv` | 3ebcc97fa64628cd50115ad58dc54dee129769375e2276babadf25d5096e5ec7 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |
| `flop_analysis/flops_per_call.json` | 7a562e0b092b68b6fbaea089354298114e864db72c6309847db44d0565e957a7 | 2026-09-28T17:41:20 | 1f93741 2026-09-28 09:44:38 +0200 |


**Freshness check.** Newest `segment_energy.json` among the 70 SAC runs: `results/sac/HalfCheetah-v5/seed_958/20260915_134531`, mtime 2026-09-23T18:04:24; oldest of the two CSVs' mtime 2026-09-28T17:41:20. **PASS** — filesystem mtime: newest SAC segment_energy.json older than the pipeline CSVs

Git view: last commit touching `results/sac/` = `5a21db9 2026-09-18 11:08:58 +0200`; last commit touching anything under `results/` = `88170be 2026-09-21 11:32:24 +0200`; last commit touching `per_run_energy_per_flop.csv` = `1f93741 2026-09-28 09:44:38 +0200`. Because mtimes on a git checkout are not generation times, staleness is decided by content instead: Part 4.8 recomputes every SAC energy/FLOP/J-per-FLOP value from `results/` and compares it to the CSV rows (if that reconciliation PASSes, the CSVs are not stale for SAC regardless of timestamps).

**Code-version consistency of the 70 runs** (`metadata.json` → `git_commit`):


| git_commit | # runs | commit | configurations |
|---|---|---|---|
| `f23f448480` | 8 | f23f448 2026-09-07 Setting baseline config for all experiments | Ant-base, HC-base |
| `92dcee0763` | 2 | 92dcee0 2026-09-07 Recorded runs for seeds - 331, 958, 14577, 43611, 85062 | Ant-base |
| `ecb98bbba4` | 10 | ecb98bb 2026-09-08 Finished remaining baseline runs for Ant/v5 | HC-utd2, HC-utd4 |
| `aa9186b0e5` | 10 | aa9186b 2026-09-09 Setup UTD sweep run script for SAC and TD3 across both envs and 5 seeds | Ant-utd2, Ant-utd4 |
| `7e672ef12c` | 20 | 7e672ef 2026-09-12 Setting up width sweep for SAC, TD3 and MBPO | Ant-w256, Ant-w512, HC-w256, HC-w512 |
| `9b4b44cd2f` | 10 | 9b4b44c 2026-09-15 Preparing batch size sweep for Half Cheetah | HC-b1024, HC-b512 |
| `788ffcd841` | 10 | 788ffcd 2026-09-17 Scheduling batch sweep for ant and new rollout sweep for MBPO | Ant-b1024, Ant-b512 |


Pairwise `git diff --stat <c1> <c2> -- algorithms/sac.py algorithms/replay_buffer.py algorithms/tracker_utils.py experiment_runner.py configs/config.py utils/ run_experiment.py` (chronological order; tracked `utils/__pycache__/*.pyc` files are excluded with `':(exclude)*.pyc'`; the content of the changes is summarised underneath):


| pair | changed files in the listed paths | summary |
|---|---|---|
| `f23f448`→`92dcee0` | experiment_runner.py, utils/thermal_gate.py | 2 files changed, 27 insertions(+), 3 deletions(-) |
| `f23f448`→`ecb98bb` | experiment_runner.py, utils/thermal_gate.py | 2 files changed, 27 insertions(+), 3 deletions(-) |
| `f23f448`→`aa9186b` | configs/config.py, experiment_runner.py, utils/thermal_gate.py | 3 files changed, 81 insertions(+), 4 deletions(-) |
| `f23f448`→`7e672ef` | configs/config.py, experiment_runner.py, utils/thermal_gate.py | 3 files changed, 183 insertions(+), 4 deletions(-) |
| `f23f448`→`9b4b44c` | configs/config.py, experiment_runner.py, utils/thermal_gate.py | 3 files changed, 183 insertions(+), 4 deletions(-) |
| `f23f448`→`788ffcd` | configs/config.py, experiment_runner.py, utils/thermal_gate.py | 3 files changed, 183 insertions(+), 4 deletions(-) |
| `92dcee0`→`ecb98bb` | (no change) |  |
| `92dcee0`→`aa9186b` | configs/config.py, experiment_runner.py | 2 files changed, 54 insertions(+), 1 deletion(-) |
| `92dcee0`→`7e672ef` | configs/config.py, experiment_runner.py | 2 files changed, 156 insertions(+), 1 deletion(-) |
| `92dcee0`→`9b4b44c` | configs/config.py, experiment_runner.py | 2 files changed, 156 insertions(+), 1 deletion(-) |
| `92dcee0`→`788ffcd` | configs/config.py, experiment_runner.py | 2 files changed, 156 insertions(+), 1 deletion(-) |
| `ecb98bb`→`aa9186b` | configs/config.py, experiment_runner.py | 2 files changed, 54 insertions(+), 1 deletion(-) |
| `ecb98bb`→`7e672ef` | configs/config.py, experiment_runner.py | 2 files changed, 156 insertions(+), 1 deletion(-) |
| `ecb98bb`→`9b4b44c` | configs/config.py, experiment_runner.py | 2 files changed, 156 insertions(+), 1 deletion(-) |
| `ecb98bb`→`788ffcd` | configs/config.py, experiment_runner.py | 2 files changed, 156 insertions(+), 1 deletion(-) |
| `aa9186b`→`7e672ef` | configs/config.py, experiment_runner.py | 2 files changed, 102 insertions(+) |
| `aa9186b`→`9b4b44c` | configs/config.py, experiment_runner.py | 2 files changed, 102 insertions(+) |
| `aa9186b`→`788ffcd` | configs/config.py, experiment_runner.py | 2 files changed, 102 insertions(+) |
| `7e672ef`→`9b4b44c` | (no change) |  |
| `7e672ef`→`788ffcd` | (no change) |  |
| `9b4b44c`→`788ffcd` | (no change) |  |


Whole-span diff `f23f448`→`788ffcd` restricted to those paths: 274 diff lines; files: configs/config.py, experiment_runner.py, utils/thermal_gate.py.
- `algorithms/sac.py`: identical across all run commits
- `algorithms/replay_buffer.py`: identical across all run commits
- `algorithms/tracker_utils.py`: identical across all run commits
- `experiment_runner.py`: +25 / -3 lines
- `configs/config.py`: +148 / -1 lines
- `run_experiment.py`: identical
- `utils/gpu_control.py`: identical
- `utils/thermal_gate.py`: +10 / -0 lines

`configs/config.py`, compared class by class between `f23f448` and `788ffcd` (AST source segments):


| definition | status | changed lines |
|---|---|---|
| `ExperimentConfig` | identical |  |
| `MBPOConfig` | changed | -    hidden_sizes: Tuple[int, int] = (256, 256)   # kept consistent with SACConfig's defaults in this repo; +    hidden_sizes: Tuple[int, int] = (1024, 1024)   # kept consistent with SACConfig's defaults in this repo |
| `SACConfig` | identical |  |
| `TD3Config` | added |  |
| `TDMPC2Config` | added |  |
| `ALGO_CONFIGS` (module-level) | changed | entries added: td3, tdmpc2 |

Verbatim changed lines of the other files (`+` added / `-` removed):

`experiment_runner.py`:
```diff
- from utils.thermal_gate import wait_for_thermal_baseline
-         "experiment_config": exp_cfg.__dict__,
-             metadata["thermal_gate"] = gate_info
+ from utils.thermal_gate import wait_for_thermal_baseline, read_current_state
+         "experiment_config": {k: v for k, v in exp_cfg.__dict__.items() if k != "thermal_gate_reference_file"},
+         thermal_gate_info = {}
+             thermal_gate_info.update(gate_info)
+         run_start_state = read_current_state()
+         if run_start_state is not None:
+             thermal_gate_info["run_start_temp_c"] = run_start_state["temp_c"]
+             thermal_gate_info["run_start_power_w"] = run_start_state["power_w"]
+             run_end_state = read_current_state()
+             if run_end_state is not None:
+                 thermal_gate_info["run_end_temp_c"] = run_end_state["temp_c"]
+                 thermal_gate_info["run_end_power_w"] = run_end_state["power_w"]
+     if thermal_gate_info:
+         metadata["thermal_gate"] = thermal_gate_info
+     if exp_cfg.algo_name == "td3":
+         from algorithms.td3 import train as td3_train
+         return td3_train(env, algo_cfg, exp_cfg, tracker, device, logger,
+                           steps_per_epoch=exp_cfg.steps_per_epoch)
+     if exp_cfg.algo_name == "tdmpc2":
+         from algorithms.tdmpc2 import train as tdmpc2_train
+         return tdmpc2_train(env, algo_cfg, exp_cfg, tracker, device, logger,
+                              steps_per_epoch=exp_cfg.steps_per_epoch)
```

`utils/thermal_gate.py`:
```diff
+ def read_current_state(gpu_index: int = 0) -> Optional[dict]:
+     """Single instantaneous GPU temp/power reading, for before/after-run snapshots
+     (as opposed to the gate's own reference/final readings, which are about
+     matching a cooldown target rather than bracketing the measured run)."""
+     state = _read_gpu_state(gpu_index)
+     if state is None:
+         return None
+     return {"temp_c": state.temp_c, "power_w": state.power_w}
```

**Working-tree caveat.** `git_commit` records HEAD, not uncommitted edits. The `thermal_gate` block of `metadata.json` has different key sets across the 70 runs:


| thermal_gate keys | # runs | runs (if ≤ 5) |
|---|---|---|
| final_power_w, final_temp_c, gated, reason, run_end_power_w, run_end_temp_c, run_start_power_w, run_start_temp_c, waited_seconds | 66 | … |
| final_power_w, final_temp_c, gated, reason, waited_seconds | 3 | HC-base/s331 (20260907_121335, commit f23f448); HC-base/s958 (20260907_123149, commit f23f448); HC-base/s14577 (20260907_124434, commit f23f448) |
| final_power_w, final_temp_c, gated, reason, reference_power_w, reference_temp_c, waited_seconds | 1 | HC-base/s43611 (20260907_125849, commit f23f448) |


The `run_start_*`/`run_end_*` keys were added to `experiment_runner.py` in commit `92dcee0` (diff above is logging-only: `read_current_state()` NVML snapshot before the head idle window and after the tail). Runs that carry those keys but are labelled with the earlier commit `f23f448`, and the one run with `reference_*` keys, show that some runs executed with an uncommitted working tree. The observed differences are in metadata logging only; whether any other uncommitted edit existed cannot be determined from the data.

All 70 runs can be regarded as running the same SAC training/measurement code: `algorithms/sac.py`, `algorithms/replay_buffer.py` and `algorithms/tracker_utils.py` are byte-identical across every recorded commit, `SACConfig` and `ExperimentConfig` are unchanged, and the remaining changes in the runner/config/utils files (listed above) concern metadata logging and other algorithms' dispatch/config — subject to the working-tree caveat.

## Part 1 — Audit of the SAC canonical run set

**1.1 Inventory** (one row per configuration; values from each run's logged `metadata.json`, identical within a configuration — checked below):


| tag | env | hidden_sizes | batch | UTD | policy_update_delay | warmup | train_steps | steps/epoch | architecture signature (`flop_keys.sig_sac_td3`) | override / script | # runs | seeds present | # duplicates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HC-base | HalfCheetah-v5 | (1024, 1024) | 256 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024` | `none (SACConfig defaults; no script found -- see Part 1)`; none found (manual CLI; commits 92dcee0/ecb98bb record them) | 5 | 331,958,14577,43611,85062 | 0 |
| HC-utd2 | HalfCheetah-v5 | (1024, 1024) | 256 | 2 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024` | `configs/overrides/sac_utd2.json`; scripts/run_utd_sweep.sh @ d1cebb4 (HC) | 5 | 331,958,14577,43611,85062 | 0 |
| HC-utd4 | HalfCheetah-v5 | (1024, 1024) | 256 | 4 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024` | `configs/overrides/sac_utd4.json`; scripts/run_utd_sweep.sh @ d1cebb4 (HC) | 5 | 331,958,14577,43611,85062 | 0 |
| HC-w256 | HalfCheetah-v5 | (256, 256) | 256 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h256x256` | `configs/overrides/sac_width256.json`; scripts/run_width_sweep.sh | 5 | 331,958,14577,43611,85062 | 0 |
| HC-w512 | HalfCheetah-v5 | (512, 512) | 256 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h512x512` | `configs/overrides/sac_width512.json`; scripts/run_width_sweep.sh | 5 | 331,958,14577,43611,85062 | 0 |
| HC-b512 | HalfCheetah-v5 | (1024, 1024) | 512 | 1 | 1 | 5000 | 100000 | 1000 | `bs512_h1024x1024` | `configs/overrides/sac_batch512.json`; scripts/run_batch_size_sweep.sh | 5 | 331,958,14577,43611,85062 | 0 |
| HC-b1024 | HalfCheetah-v5 | (1024, 1024) | 1024 | 1 | 1 | 5000 | 100000 | 1000 | `bs1024_h1024x1024` | `configs/overrides/sac_batch1024.json`; scripts/run_batch_size_sweep.sh | 5 | 331,958,14577,43611,85062 | 0 |
| Ant-base | Ant-v5 | (1024, 1024) | 256 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024` | `none (SACConfig defaults; no script found -- see Part 1)`; none found (manual CLI; commits 92dcee0/ecb98bb record them) | 5 | 331,958,14577,43611,85062 | 0 |
| Ant-utd2 | Ant-v5 | (1024, 1024) | 256 | 2 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024` | `configs/overrides/sac_utd2.json`; scripts/run_utd_sweep.sh @ aa9186b/24f41b0 (Ant) | 5 | 331,958,14577,43611,85062 | 0 |
| Ant-utd4 | Ant-v5 | (1024, 1024) | 256 | 4 | 1 | 5000 | 100000 | 1000 | `bs256_h1024x1024` | `configs/overrides/sac_utd4.json`; scripts/run_utd_sweep.sh @ aa9186b/24f41b0 (Ant) | 5 | 331,958,14577,43611,85062 | 0 |
| Ant-w256 | Ant-v5 | (256, 256) | 256 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h256x256` | `configs/overrides/sac_width256.json`; scripts/run_width_sweep.sh | 5 | 331,958,14577,43611,85062 | 0 |
| Ant-w512 | Ant-v5 | (512, 512) | 256 | 1 | 1 | 5000 | 100000 | 1000 | `bs256_h512x512` | `configs/overrides/sac_width512.json`; scripts/run_width_sweep.sh | 5 | 331,958,14577,43611,85062 | 0 |
| Ant-b512 | Ant-v5 | (1024, 1024) | 512 | 1 | 1 | 5000 | 100000 | 1000 | `bs512_h1024x1024` | `configs/overrides/sac_batch512.json`; scripts/run_ant_overnight_sweep.sh | 5 | 331,958,14577,43611,85062 | 0 |
| Ant-b1024 | Ant-v5 | (1024, 1024) | 1024 | 1 | 1 | 5000 | 100000 | 1000 | `bs1024_h1024x1024` | `configs/overrides/sac_batch1024.json`; scripts/run_ant_overnight_sweep.sh | 5 | 331,958,14577,43611,85062 | 0 |


CLI used, from the scripts: every sweep invocation is `sudo rl-exp/bin/python run_experiment.py --algo sac --env <env> --seed <seed> --algo-config-overrides <override>` (no `--warmup-steps` for SAC, so the CLI default 5000 applies). The SAC baselines have no script in the repo or in git history; their `metadata.json` (no override fields differ from `SACConfig`, all `ExperimentConfig` fields at CLI defaults) is consistent with `sudo rl-exp/bin/python run_experiment.py --algo sac --env <env> --seed <seed>` as in `cli_commands.txt`. `metadata.json` does not record the CLI arguments or the override-file path, so the override column is inferred from the logged config + scripts, not read from the run.

**1.2 Full run list** → `contexts/section_4.1_data/sac_run_inventory.csv` (70 rows).

**1.3 Checks**

- **PASS** — exactly 70 SAC canonical runs (found 70; unmatched configs: 0)
- **PASS** — exactly 5 runs per configuration (14 configurations) (all 14 have 5)
- **PASS** — no duplicate (algo, env, seed, config) runs
- **PASS** — every run has emissions.csv, segment_energy.json, training_metrics.json, metadata.json, run.log
- **PASS** — every run has exactly one per-task CodeCarbon log `emissions_*.csv` (the file holding the rollout_i / gradient_updates_i rows; see note)
- **PASS** — train_steps = 100000, steps_per_epoch = 1000, warmup_steps = 5000 in every run
- **PASS** — per-task CSV has exactly rollout_0..99, gradient_updates_0..99, warmup_0, idle_baseline_head, idle_baseline_tail (203 rows)
- **PASS** — gradient_updates call structure = 1000 × UTD updates per epoch — **indirect**: the number of update() calls is not logged; `sac.train()` runs exactly `steps_per_epoch × updates_per_env_step` calls unless `len(buffer) < batch_size`, and the logged `buffer_size` per epoch is 6000…105000 (≥ 1024 > every batch size) with a non-null critic loss in all 100 epochs of every run
- **PASS** — GPU = NVIDIA GeForce RTX 5090 (metadata `cuda_device_name`)
  - per-task CSV `gpu_model`: {'1 x NVIDIA GeForce RTX 5090': 70}; `cpu_model`: {'Intel(R) Core(TM) Ultra 9 285K': 70}
- **PASS** — clock lock requested at 2000/2000 MHz (not null, not 200) with lock_gpu_clocks = true
- **PASS** — torch 2.13.0+cu130 and codecarbon 3.3.0 in every run
- **PASS** — logged config matches tag, and every other algo_config field equals the SACConfig default and every experiment_config field equals the HC-base value, symmetric over the union of keys (one-factor-at-a-time; seed/env_id/algo_name ignored) (same reference for both envs, so HC and Ant also differ only in env_id; the key `thermal_gate_reference_file` is present in 3 runs' experiment_config and absent in the others (metadata format change at 92dcee0; values where present: ['results/_thermal_reference.json']; excluded from the comparison))
- **PASS** — `run.log` of every run contains none of the 20 failure-mode strings (list in 'Schema samples' §6)
- **PASS** — sweep-log block (stdout+stderr captured by the sweep script) contains none of the strings — available for 60 of 70 runs
- **PASS** — metadata thermal_gate: gated = true, reason = reached_reference (no timeout) in every run
- **PASS** — data-side RAPL check: per-task `cpu_power` varies across the 100 gradient_updates tasks in every run (a TDP fallback would give a constant value) (min distinct values per run = 100)
- **PASS** — data-side geolocation check: no run's per-task CSV shows the Canada/Quebec fallback (country_name, region) ({('Germany', 'baden-wurttemberg'): 70})

Grep caveat: `run.log` is written only by the `experiment_runner` logger. The `gpu_control`, `thermal_gate` and CodeCarbon loggers are **not** attached to `run.log` (their warnings go to stderr only), so a clean `run.log` is not by itself evidence that the clock lock / governor / RAPL / geolocation calls succeeded. The sweep scripts tee stdout+stderr into their sweep log, which covers 60/70 runs (all except the 10 baselines, which were launched manually and whose stderr was not saved). The only warning text in those blocks is CodeCarbon's "Multiple instances of codecarbon are allowed to run at the same time." (Part 8.5). Thermal-gate timeouts are additionally recorded in `metadata.json` (checked above). For the 10 baselines the clock lock is therefore verified only as *requested* (metadata), not as *applied*.

- **PASS** — UTD sweep provenance: utd2/utd4 exist for all five canonical seeds in both envs, at (1024,1024) and the current stack; no non-canonical-seed run is in the analysed set
  - HC-utd2: seeds [331, 958, 14577, 43611, 85062], hidden_sizes [(1024, 1024)], stack [('2.13.0+cu130', '3.3.0')], timestamps 20260908_122254…20260908_133331, commits ['ecb98bb']
  - HC-utd4: seeds [331, 958, 14577, 43611, 85062], hidden_sizes [(1024, 1024)], stack [('2.13.0+cu130', '3.3.0')], timestamps 20260908_135102…20260908_155540, commits ['ecb98bb']
  - Ant-utd2: seeds [331, 958, 14577, 43611, 85062], hidden_sizes [(1024, 1024)], stack [('2.13.0+cu130', '3.3.0')], timestamps 20260909_134544…20260909_145811, commits ['aa9186b']
  - Ant-utd4: seeds [331, 958, 14577, 43611, 85062], hidden_sizes [(1024, 1024)], stack [('2.13.0+cu130', '3.3.0')], timestamps 20260909_151617…20260909_172327, commits ['aa9186b']
  - non-canonical SAC runs with UTD ≠ 1 (excluded by seed filter): [('results/sac/HalfCheetah-v5/seed_42/20260826_111906', 42)]
  - `results/sac/HalfCheetah-v5/_utd_sweep_status.json`: 10 entries, seeds [331, 958, 14577, 43611, 85062], statuses {'done': 10}, started 2026-09-08T10:22:54.255569+00:00.

- Excluded SAC runs (counts only, no data used):

| env | seed | # runs |
|---|---|---|
| HalfCheetah-v5 | 0 | 7 |
| HalfCheetah-v5 | 42 | 3 |
| HalfCheetah-v5 | 56 | 3 |
| Humanoid-v5 | 56 | 2 |


  Total excluded: 15; SAC run dirs not matching any of the 7 tags at a canonical seed: 0 

Sweep bookkeeping files (for completeness; run directories are the evidence):


| status file | # entries | # SAC entries | statuses | every SAC entry's run_dir is one of the 70 |
|---|---|---|---|---|
| `results/_width_sweep_status.json` | 60 | 20 | {'done': 60} | yes |
| `results/_batch_size_sweep_status.json` | 35 | 10 | {'done': 35} | yes |
| `results/_ant_overnight_sweep_status.json` | 45 | 10 | {'done': 45} | yes |
| `results/sac/HalfCheetah-v5/_utd_sweep_status.json` | 10 | 10 | {'done': 10} | yes |
| `results/_utd_sweep_status.json` | 20 | 0 | {'done': 20} | n/a (no SAC entries) |

`results/_utd_sweep_status.json` now holds the MBPO UTD sweep (0 SAC entries), as documented. The SAC-Ant utd2/utd4 runs are evidenced by their run directories and by the retained `results/_utd_sweep.log` block.

**1.4 Cross-seed CSV rows for SAC** (`cross_seed_energy_per_flop.csv`, `algo == 'sac'`):


| env | signature | UTD | # rows | segments | n_seeds values |
|---|---|---|---|---|---|
| Ant-v5 | `bs1024_h1024x1024` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| Ant-v5 | `bs256_h1024x1024` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| Ant-v5 | `bs256_h1024x1024` | 2 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| Ant-v5 | `bs256_h1024x1024` | 4 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| Ant-v5 | `bs256_h256x256` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| Ant-v5 | `bs256_h512x512` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| Ant-v5 | `bs512_h1024x1024` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| HalfCheetah-v5 | `bs1024_h1024x1024` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| HalfCheetah-v5 | `bs256_h1024x1024` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| HalfCheetah-v5 | `bs256_h1024x1024` | 2 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| HalfCheetah-v5 | `bs256_h1024x1024` | 4 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| HalfCheetah-v5 | `bs256_h256x256` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| HalfCheetah-v5 | `bs256_h512x512` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |
| HalfCheetah-v5 | `bs512_h1024x1024` | 1 | 6 | TOTAL_MEASURED_TRAINING, actor_update, buffer_sample, critic_update, rollout, target_update | [5] |

- **PASS** — every SAC cross-seed row has n_seeds = 5 (the cross-seed CSV only contains canonical-seed runs; `included_in_cross_seed_avg` lives in the per-run CSV) (84 rows)
- **PASS** — all 70 runs present in per_run CSV with included_in_cross_seed_avg = True (70 run_dirs)
- Non-canonical SAC rows in per-run CSV: 15 runs, included flag values ['False'] (these are the excluded dev runs).
- **PASS** — set of SAC signatures in the cross-seed CSV equals the 5 derived from the 14 configurations (CSV: ['bs1024_h1024x1024', 'bs256_h1024x1024', 'bs256_h256x256', 'bs256_h512x512', 'bs512_h1024x1024']; runs: ['bs1024_h1024x1024', 'bs256_h1024x1024', 'bs256_h256x256', 'bs256_h512x512', 'bs512_h1024x1024'])
- **PASS** — utd2/utd4 share the baseline's signature (UTD is not part of the signature; it is a separate group key `updates_per_env_step`)
- **PASS** — cross-seed SAC groups = 14 = the 14 configurations (segment sets per group: {('TOTAL_MEASURED_TRAINING', 'actor_update', 'buffer_sample', 'critic_update', 'rollout', 'target_update'): 14})
- The cross-seed CSV has no `warmup`/`idle_baseline_*` rows (the pipeline only aggregates rows with a FLOP entry; those segments exist only in the per-run CSV) and no `gradient_updates` row in either CSV.

## Part 2 — SAC implementation facts

**2.1 `SACConfig` defaults** (`configs/config.py`, dumped via `dataclasses.asdict(SACConfig())`):

```json
{
  "hidden_sizes": [
    1024,
    1024
  ],
  "actor_lr": 0.0003,
  "critic_lr": 0.0003,
  "alpha_lr": 0.0003,
  "gamma": 0.99,
  "tau": 0.005,
  "batch_size": 256,
  "buffer_capacity": 1000000,
  "target_entropy": null,
  "autotune_alpha": true,
  "init_alpha": 0.2,
  "updates_per_env_step": 1,
  "policy_update_delay": 1
}
```
`ExperimentConfig` defaults (protocol, shared by all algorithms):

```json
{
  "algo_name": "sac",
  "env_id": "HalfCheetah-v5",
  "seed": 0,
  "settle_seconds": 30.0,
  "idle_baseline_seconds": 90.0,
  "idle_tail_seconds": 90.0,
  "warmup_steps": 5000,
  "train_steps": 100000,
  "steps_per_epoch": 1000,
  "lock_gpu_clocks": true,
  "gpu_min_clock_mhz": 2000,
  "gpu_max_clock_mhz": 2000,
  "set_persistence_mode": true,
  "set_cpu_performance_governor": true,
  "thermal_gate_enabled": true,
  "thermal_gate_temp_tolerance_c": 2.0,
  "thermal_gate_power_tolerance_w": 5.0,
  "thermal_gate_max_wait_seconds": 600.0,
  "thermal_gate_poll_interval_seconds": 5.0,
  "thermal_gate_reference_file": "results/_thermal_reference.json",
  "measure_power_secs": 1.0,
  "tracking_mode": "machine",
  "output_dir": "results",
  "country_iso_code": "DEU",
  "force_cpu_power_w": null,
  "device": "cuda"
}
```

`metadata.json` of `results/sac/HalfCheetah-v5/seed_331/20260907_121335` (HC-base, seed 331) — verbatim, complete:

```json
{
  "algo_name": "sac",
  "env_id": "HalfCheetah-v5",
  "seed": 331,
  "git_commit": "f23f448480ecdf1dd6dc552b45f9194b2602bc11",
  "torch_version": "2.13.0+cu130",
  "device": "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "start_time_utc": "2026-09-07T10:13:36.036705",
  "experiment_config": {
    "algo_name": "sac",
    "env_id": "HalfCheetah-v5",
    "seed": 331,
    "settle_seconds": 30.0,
    "idle_baseline_seconds": 90.0,
    "idle_tail_seconds": 90.0,
    "warmup_steps": 5000,
    "train_steps": 100000,
    "steps_per_epoch": 1000,
    "lock_gpu_clocks": true,
    "gpu_min_clock_mhz": 2000,
    "gpu_max_clock_mhz": 2000,
    "set_persistence_mode": true,
    "set_cpu_performance_governor": true,
    "thermal_gate_enabled": true,
    "thermal_gate_temp_tolerance_c": 2.0,
    "thermal_gate_power_tolerance_w": 5.0,
    "thermal_gate_max_wait_seconds": 600.0,
    "thermal_gate_poll_interval_seconds": 5.0,
    "thermal_gate_reference_file": "results/_thermal_reference.json",
    "measure_power_secs": 1.0,
    "tracking_mode": "machine",
    "output_dir": "results",
    "country_iso_code": "DEU",
    "force_cpu_power_w": null,
    "device": "cuda"
  },
  "algo_config": {
    "hidden_sizes": [
      1024,
      1024
    ],
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "alpha_lr": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_capacity": 1000000,
    "target_entropy": null,
    "autotune_alpha": true,
    "init_alpha": 0.2,
    "updates_per_env_step": 1,
    "policy_update_delay": 1
  },
  "codecarbon_version": "3.3.0",
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 1.1682510375976562e-05,
    "final_temp_c": 45.0,
    "final_power_w": 26.94
  },
  "end_time_utc": "2026-09-07T10:24:28.627367"
}
```

`metadata.json` of `results/sac/Ant-v5/seed_331/20260907_132907` (Ant-base, seed 331) — verbatim, complete:

```json
{
  "algo_name": "sac",
  "env_id": "Ant-v5",
  "seed": 331,
  "git_commit": "f23f448480ecdf1dd6dc552b45f9194b2602bc11",
  "torch_version": "2.13.0+cu130",
  "device": "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 5090",
  "start_time_utc": "2026-09-07T11:29:07.280745",
  "experiment_config": {
    "algo_name": "sac",
    "env_id": "Ant-v5",
    "seed": 331,
    "settle_seconds": 30.0,
    "idle_baseline_seconds": 90.0,
    "idle_tail_seconds": 90.0,
    "warmup_steps": 5000,
    "train_steps": 100000,
    "steps_per_epoch": 1000,
    "lock_gpu_clocks": true,
    "gpu_min_clock_mhz": 2000,
    "gpu_max_clock_mhz": 2000,
    "set_persistence_mode": true,
    "set_cpu_performance_governor": true,
    "thermal_gate_enabled": true,
    "thermal_gate_temp_tolerance_c": 2.0,
    "thermal_gate_power_tolerance_w": 5.0,
    "thermal_gate_max_wait_seconds": 600.0,
    "thermal_gate_poll_interval_seconds": 5.0,
    "measure_power_secs": 1.0,
    "tracking_mode": "machine",
    "output_dir": "results",
    "country_iso_code": "DEU",
    "force_cpu_power_w": null,
    "device": "cuda"
  },
  "algo_config": {
    "hidden_sizes": [
      1024,
      1024
    ],
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "alpha_lr": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_capacity": 1000000,
    "target_entropy": null,
    "autotune_alpha": true,
    "init_alpha": 0.2,
    "updates_per_env_step": 1,
    "policy_update_delay": 1
  },
  "codecarbon_version": "3.3.0",
  "thermal_gate": {
    "gated": true,
    "reason": "reached_reference",
    "waited_seconds": 1.0967254638671875e-05,
    "final_temp_c": 46.0,
    "final_power_w": 26.814,
    "run_start_temp_c": 46.0,
    "run_start_power_w": 26.635,
    "run_end_temp_c": 43.0,
    "run_end_power_w": 27.256
  },
  "end_time_utc": "2026-09-07T11:40:50.145665"
}
```

**2.2 Verbatim code** — `algorithms/sac.py` (line numbers as in the file at HEAD; byte-identical at every run commit, Part 0).

Networks (`_mlp`, `GaussianPolicy`, `QNetwork`, `UpdateInfo`):

```python
  37  def _mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
  38      layers = []
  39      for i in range(len(sizes) - 1):
  40          act = activation if i < len(sizes) - 2 else output_activation
  41          layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
  42      return nn.Sequential(*layers)
  43  
  44  
  45  class GaussianPolicy(nn.Module):
  46      def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
  47          super().__init__()
  48          self.net = _mlp([obs_dim, *hidden_sizes], output_activation=nn.ReLU)
  49          self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
  50          self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
  51          self.register_buffer("act_limit", torch.as_tensor(act_limit, dtype=torch.float32))
  52  
  53      def forward(self, obs, deterministic=False, with_logprob=True):
  54          h = self.net(obs)
  55          mu = self.mu_layer(h)
  56          log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
  57          std = log_std.exp()
  58          dist = Normal(mu, std)
  59  
  60          if deterministic:
  61              pre_tanh = mu
  62          else:
  63              pre_tanh = dist.rsample()  # reparameterization trick
  64  
  65          action = torch.tanh(pre_tanh)
  66  
  67          if with_logprob:
  68              logp = dist.log_prob(pre_tanh).sum(dim=-1)
  69              # tanh squashing correction (SAC paper, appendix C)
  70              logp -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1)
  71          else:
  72              logp = None
  73  
  74          return action * self.act_limit, logp
  75  
  76  
  77  class QNetwork(nn.Module):
  78      def __init__(self, obs_dim, act_dim, hidden_sizes):
  79          super().__init__()
  80          self.net = _mlp([obs_dim + act_dim, *hidden_sizes, 1])
  81  
  82      def forward(self, obs, act):
  83          return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)
  84  
  85  
  86  @dataclass
  87  class UpdateInfo:
  88      """Wall-clock time (seconds) spent in each sub-segment of one call to update()."""
  89      buffer_sample_s: float = 0.0
  90      critic_update_s: float = 0.0
  91      actor_update_s: float = 0.0
  92      target_update_s: float = 0.0
  93      critic_loss: Optional[float] = None
  94      actor_loss: Optional[float] = None
  95      alpha: Optional[float] = None
  96  
  97
```
`SACAgent.__init__`, `select_action`, `update()`:

```python
  98  class SACAgent:
  99      def __init__(self, obs_dim, act_dim, act_limit, cfg: SACConfig, device: torch.device):
 100          self.cfg = cfg
 101          self.device = device
 102          self.act_dim = act_dim
 103  
 104          self.actor = GaussianPolicy(obs_dim, act_dim, cfg.hidden_sizes, act_limit).to(device)
 105          self.q1 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
 106          self.q2 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
 107          self.q1_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
 108          self.q2_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
 109          self.q1_targ.load_state_dict(self.q1.state_dict())
 110          self.q2_targ.load_state_dict(self.q2.state_dict())
 111          for p in self.q1_targ.parameters():
 112              p.requires_grad = False
 113          for p in self.q2_targ.parameters():
 114              p.requires_grad = False
 115  
 116          self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
 117          self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
 118          self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
 119  
 120          self.autotune_alpha = cfg.autotune_alpha
 121          if cfg.autotune_alpha:
 122              self.target_entropy = (
 123                  cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
 124              )
 125              self.log_alpha = torch.tensor(
 126                  np.log(cfg.init_alpha), dtype=torch.float32, requires_grad=True, device=device
 127              )
 128              self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
 129          else:
 130              self.log_alpha = torch.tensor(np.log(cfg.init_alpha), device=device)
 131  
 132          self._update_counter = 0
 133  
 134      @property
 135      def alpha(self):
 136          return self.log_alpha.exp()
 137  
 138      @torch.no_grad()
 139      def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
 140          obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
 141          action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
 142          return action.squeeze(0).cpu().numpy()
 143  
 144      def update(self, buffer: ReplayBuffer, batch_size: int) -> UpdateInfo:
 145          info = UpdateInfo()
 146  
 147          # --- buffer_sample ---
 148          t0 = time.perf_counter()
 149          obs, act, rew, next_obs, done = buffer.sample(batch_size)
 150          obs_t = torch.as_tensor(obs, device=self.device)
 151          act_t = torch.as_tensor(act, device=self.device)
 152          rew_t = torch.as_tensor(rew, device=self.device).squeeze(-1)
 153          next_obs_t = torch.as_tensor(next_obs, device=self.device)
 154          done_t = torch.as_tensor(done, device=self.device).squeeze(-1)
 155          info.buffer_sample_s = time.perf_counter() - t0
 156  
 157          # --- critic_update ---
 158          t0 = time.perf_counter()
 159          with torch.no_grad():
 160              next_action, next_logp = self.actor(next_obs_t)
 161              q1_next = self.q1_targ(next_obs_t, next_action)
 162              q2_next = self.q2_targ(next_obs_t, next_action)
 163              q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
 164              target_q = rew_t + (1.0 - done_t) * self.cfg.gamma * q_next
 165  
 166          q1_pred = self.q1(obs_t, act_t)
 167          q2_pred = self.q2(obs_t, act_t)
 168          q1_loss = F.mse_loss(q1_pred, target_q)
 169          q2_loss = F.mse_loss(q2_pred, target_q)
 170  
 171          self.q1_opt.zero_grad(set_to_none=True)
 172          q1_loss.backward()
 173          self.q1_opt.step()
 174  
 175          self.q2_opt.zero_grad(set_to_none=True)
 176          q2_loss.backward()
 177          self.q2_opt.step()
 178          info.critic_update_s = time.perf_counter() - t0
 179          info.critic_loss = (q1_loss.item() + q2_loss.item()) / 2.0
 180  
 181          # --- actor_update (+ alpha) ---
 182          do_actor_update = (self._update_counter % self.cfg.policy_update_delay) == 0
 183          if do_actor_update:
 184              t0 = time.perf_counter()
 185              for p in self.q1.parameters():
 186                  p.requires_grad = False
 187              for p in self.q2.parameters():
 188                  p.requires_grad = False
 189  
 190              action, logp = self.actor(obs_t)
 191              q1_pi = self.q1(obs_t, action)
 192              q2_pi = self.q2(obs_t, action)
 193              q_pi = torch.min(q1_pi, q2_pi)
 194              actor_loss = (self.alpha.detach() * logp - q_pi).mean()
 195  
 196              self.actor_opt.zero_grad(set_to_none=True)
 197              actor_loss.backward()
 198              self.actor_opt.step()
 199  
 200              for p in self.q1.parameters():
 201                  p.requires_grad = True
 202              for p in self.q2.parameters():
 203                  p.requires_grad = True
 204  
 205              if self.autotune_alpha:
 206                  alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
 207                  self.alpha_opt.zero_grad(set_to_none=True)
 208                  alpha_loss.backward()
 209                  self.alpha_opt.step()
 210  
 211              info.actor_update_s = time.perf_counter() - t0
 212              info.actor_loss = actor_loss.item()
 213              info.alpha = self.alpha.item()
 214  
 215          # --- target_update ---
 216          t0 = time.perf_counter()
 217          with torch.no_grad():
 218              for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
 219                  p_targ.data.mul_(1 - self.cfg.tau)
 220                  p_targ.data.add_(self.cfg.tau * p.data)
 221              for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
 222                  p_targ.data.mul_(1 - self.cfg.tau)
 223                  p_targ.data.add_(self.cfg.tau * p.data)
 224          info.target_update_s = time.perf_counter() - t0
 225  
 226          self._update_counter += 1
 227          return info
 228  
 229
```
`train()`:

```python
 230  def train(
 231      env,
 232      sac_cfg: SACConfig,
 233      exp_cfg: ExperimentConfig,
 234      tracker,
 235      device: torch.device,
 236      logger,
 237      steps_per_epoch: int = 1000,
 238  ):
 239      """
 240      Runs SAC warmup + measured training. Returns:
 241        - agent: the trained SACAgent
 242        - energy_log: dict of energy_consumed (kWh) per segment: rollout,
 243          buffer_sample, critic_update, actor_update, target_update, plus
 244          'warmup' (kept separate from the measured segments).
 245        - metrics: dict with two lists for RL-performance analysis (separate
 246          from the energy accounting above):
 247            "episodes": one row per completed episode -- episode_idx, phase
 248              ("warmup"/"train"/"train_incomplete"), epoch (None during
 249              warmup), env_step (cumulative measured-training step at which
 250              the episode ended), return, length.
 251            "epochs": one row per training epoch -- epoch, env_step,
 252              cumulative_reward (running sum of reward since the start of
 253              measured training), epoch_reward_sum, epoch_reward_mean_per_step,
 254              num_episodes_completed, mean_episode_return (None if no episode
 255              finished within the epoch), critic_loss_mean, actor_loss_mean,
 256              alpha_end, buffer_size.
 257  
 258      `rollout` and the combined `gradient_updates` bucket are real CodeCarbon
 259      measurements (one task per epoch, unique-named). `buffer_sample`,
 260      `critic_update`, `actor_update`, `target_update` are obtained by
 261      allocating each epoch's `gradient_updates` energy proportionally to the
 262      wall-clock time share of each sub-segment within that epoch. This is a
 263      deliberate approximation -- see the module docstring and the thesis
 264      write-up caveat in README.md.
 265      """
 266      obs_dim = env.observation_space.shape[0]
 267      act_dim = env.action_space.shape[0]
 268      act_limit = float(env.action_space.high[0])
 269  
 270      agent = SACAgent(obs_dim, act_dim, act_limit, sac_cfg, device)
 271      buffer = ReplayBuffer(sac_cfg.buffer_capacity, obs_dim, act_dim)
 272  
 273      energy_log: Dict[str, float] = {}
 274      sub_time_totals = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
 275  
 276      episodes_log = []
 277      episode_idx = 0
 278      episode_return = 0.0
 279      episode_length = 0
 280  
 281      def _record_episode(phase, epoch, env_step):
 282          nonlocal episode_idx, episode_return, episode_length
 283          episodes_log.append({
 284              "episode_idx": episode_idx,
 285              "phase": phase,
 286              "epoch": epoch,
 287              "env_step": env_step,
 288              "return": episode_return,
 289              "length": episode_length,
 290          })
 291          episode_idx += 1
 292          episode_return = 0.0
 293          episode_length = 0
 294  
 295      obs, _ = env.reset(seed=exp_cfg.seed)
 296  
 297      # ---------------- warmup (random policy, fills buffer; excluded from analysis segments) ----------------
 298      logger.info("Starting warmup: %d steps", exp_cfg.warmup_steps)
 299      with TrackerTask(tracker, "warmup", 0, energy_log):
 300          for warmup_step in range(exp_cfg.warmup_steps):
 301              action = env.action_space.sample()
 302              next_obs, reward, terminated, truncated, _ = env.step(action)
 303              buffer.add(obs, action, reward, next_obs, terminated)
 304              episode_return += reward
 305              episode_length += 1
 306              obs = next_obs
 307              if terminated or truncated:
 308                  _record_episode("warmup", None, warmup_step + 1)
 309                  obs, _ = env.reset()
 310  
 311      # ---------------- measured training ----------------
 312      n_epochs = max(1, exp_cfg.train_steps // steps_per_epoch)
 313      logger.info("Starting measured training: %d epochs x %d steps = %d env steps",
 314                  n_epochs, steps_per_epoch, n_epochs * steps_per_epoch)
 315  
 316      epochs_log = []
 317      cumulative_reward = 0.0
 318      global_step = 0
 319  
 320      for epoch in range(n_epochs):
 321          # --- rollout block ---
 322          epoch_reward_sum = 0.0
 323          epoch_episode_returns = []
 324          with TrackerTask(tracker, "rollout", epoch, energy_log):
 325              for _ in range(steps_per_epoch):
 326                  action = agent.select_action(obs, deterministic=False)
 327                  next_obs, reward, terminated, truncated, _ = env.step(action)
 328                  buffer.add(obs, action, reward, next_obs, terminated)
 329                  episode_return += reward
 330                  episode_length += 1
 331                  cumulative_reward += reward
 332                  epoch_reward_sum += reward
 333                  global_step += 1
 334                  obs = next_obs
 335                  if terminated or truncated:
 336                      epoch_episode_returns.append(episode_return)
 337                      _record_episode("train", epoch, global_step)
 338                      obs, _ = env.reset()
 339  
 340          # --- gradient update block ---
 341          epoch_sub_times = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
 342          critic_losses, actor_losses, alphas = [], [], []
 343          n_updates = steps_per_epoch * sac_cfg.updates_per_env_step
 344          with TrackerTask(tracker, "gradient_updates", epoch, energy_log):
 345              for _ in range(n_updates):
 346                  if len(buffer) < sac_cfg.batch_size:
 347                      break
 348                  info = agent.update(buffer, sac_cfg.batch_size)
 349                  epoch_sub_times["buffer_sample"] += info.buffer_sample_s
 350                  epoch_sub_times["critic_update"] += info.critic_update_s
 351                  epoch_sub_times["actor_update"] += info.actor_update_s
 352                  epoch_sub_times["target_update"] += info.target_update_s
 353                  if info.critic_loss is not None:
 354                      critic_losses.append(info.critic_loss)
 355                  if info.actor_loss is not None:
 356                      actor_losses.append(info.actor_loss)
 357                  if info.alpha is not None:
 358                      alphas.append(info.alpha)
 359  
 360          # Sub-segment times accumulate across all epochs and are reconciled against the
 361          # total measured "gradient_updates" energy once, after the loop (see below) --
 362          # simpler than a per-epoch split and equally defensible under the constant-power
 363          # approximation this scheme relies on.
 364          for k in sub_time_totals:
 365              sub_time_totals[k] += epoch_sub_times[k]
 366  
 367          epochs_log.append({
 368              "epoch": epoch,
 369              "env_step": global_step,
 370              "cumulative_reward": cumulative_reward,
 371              "epoch_reward_sum": epoch_reward_sum,
 372              "epoch_reward_mean_per_step": epoch_reward_sum / steps_per_epoch,
 373              "num_episodes_completed": len(epoch_episode_returns),
 374              "mean_episode_return": (
 375                  sum(epoch_episode_returns) / len(epoch_episode_returns)
 376                  if epoch_episode_returns else None
 377              ),
 378              "critic_loss_mean": (sum(critic_losses) / len(critic_losses)) if critic_losses else None,
 379              "actor_loss_mean": (sum(actor_losses) / len(actor_losses)) if actor_losses else None,
 380              "alpha_end": alphas[-1] if alphas else None,
 381              "buffer_size": len(buffer),
 382          })
 383  
 384          if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
 385              logger.info(
 386                  "Epoch %d/%d done (buffer size=%d, cumulative_reward=%.2f, mean_episode_return=%s)",
 387                  epoch + 1, n_epochs, len(buffer), cumulative_reward,
 388                  epochs_log[-1]["mean_episode_return"],
 389              )
 390  
 391      # record a trailing partial episode (if training ended mid-episode) so no reward is lost
 392      if episode_length > 0:
 393          _record_episode("train_incomplete", n_epochs - 1, global_step)
 394  
 395      # ---------------- reconcile: split total gradient_updates energy by aggregate time share ----------------
 396      total_gradient_energy = energy_log.pop("gradient_updates", 0.0)
 397      total_time = sum(sub_time_totals.values())
 398      for k, t in sub_time_totals.items():
 399          share = (t / total_time) if total_time > 0 else 0.0
 400          energy_log[k] = total_gradient_energy * share
 401  
 402      energy_log["_sub_segment_wall_time_seconds"] = sub_time_totals
 403      metrics = {"episodes": episodes_log, "epochs": epochs_log}
 404      return agent, energy_log, metrics
```
`algorithms/replay_buffer.py` (complete) and `algorithms/tracker_utils.py` (complete):

```python
   1  """Fixed-capacity circular replay buffer, numpy-backed (fast, no torch overhead
   2  until the sampled batch is converted to tensors at train-step time -- keeps
   3  the 'buffer sample' segment's energy cost isolated from tensor/GPU transfer,
   4  which you may want to sub-segment further later)."""
   5  from __future__ import annotations
   6  
   7  import numpy as np
   8  
   9  
  10  class ReplayBuffer:
  11      def __init__(self, capacity: int, obs_dim: int, act_dim: int):
  12          self.capacity = capacity
  13          self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
  14          self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
  15          self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
  16          self.rewards = np.zeros((capacity, 1), dtype=np.float32)
  17          self.dones = np.zeros((capacity, 1), dtype=np.float32)
  18          self.ptr = 0
  19          self.size = 0
  20  
  21      def add(self, obs, action, reward, next_obs, done):
  22          self.obs[self.ptr] = obs
  23          self.actions[self.ptr] = action
  24          self.rewards[self.ptr, 0] = reward
  25          self.next_obs[self.ptr] = next_obs
  26          self.dones[self.ptr, 0] = float(done)
  27          self.ptr = (self.ptr + 1) % self.capacity
  28          self.size = min(self.size + 1, self.capacity)
  29  
  30      def add_batch(self, obs, action, reward, next_obs, done):
  31          """Vectorized insert of a batch of transitions. Used by MBPO's branched
  32          model rollouts, where thousands of synthetic transitions are generated
  33          per epoch -- inserting them one at a time in a Python loop would
  34          dominate wall-clock cost. Behaves as a no-op for n == 0."""
  35          n = obs.shape[0]
  36          if n == 0:
  37              return
  38          idx = (self.ptr + np.arange(n)) % self.capacity
  39          self.obs[idx] = obs
  40          self.actions[idx] = action
  41          self.rewards[idx, 0] = reward
  42          self.next_obs[idx] = next_obs
  43          self.dones[idx, 0] = done.astype(np.float32)
  44          self.ptr = (self.ptr + n) % self.capacity
  45          self.size = min(self.size + n, self.capacity)
  46  
  47      def sample(self, batch_size: int):
  48          idx = np.random.randint(0, self.size, size=batch_size)
  49          return (
  50              self.obs[idx],
  51              self.actions[idx],
  52              self.rewards[idx],
  53              self.next_obs[idx],
  54              self.dones[idx],
  55          )
  56  
  57      def __len__(self):
  58          return self.size
```
```python
   1  """Shared CodeCarbon task-timing helpers used by every algorithm's train()
   2  loop (see algorithms/sac.py and algorithms/mbpo.py). Kept in one place so
   3  adding a new algorithm doesn't mean re-deriving the unique-task-name-per-call
   4  workaround described in README.md ("Why epoch-level tagging..." section)."""
   5  from __future__ import annotations
   6  
   7  from typing import Dict
   8  
   9  
  10  class NullTask:
  11      """No-op context manager used when tracker is None (e.g. dry-run/debug)."""
  12      def __enter__(self):
  13          return self
  14  
  15      def __exit__(self, *a):
  16          return False
  17  
  18  
  19  class TrackerTask:
  20      """Wraps tracker.start_task/stop_task as a context manager and stores the
  21      returned EmissionsData's energy_consumed (kWh) into a results dict keyed
  22      by a stable prefix (unique task *names* per call avoid a CodeCarbon bug
  23      where reusing a task name corrupts internal accounting -- see README)."""
  24  
  25      def __init__(self, tracker, prefix: str, counter: int, energy_log: Dict[str, float]):
  26          self.tracker = tracker
  27          self.name = f"{prefix}_{counter}"
  28          self.prefix = prefix
  29          self.energy_log = energy_log
  30  
  31      def __enter__(self):
  32          if self.tracker is not None:
  33              self.tracker.start_task(self.name)
  34          return self
  35  
  36      def __exit__(self, *a):
  37          if self.tracker is not None:
  38              data = self.tracker.stop_task(self.name)
  39              self.energy_log[self.prefix] = self.energy_log.get(self.prefix, 0.0) + (
  40                  data.energy_consumed if data is not None else 0.0
  41              )
  42          return False
```

**What each timer contains** (facts read from the code above):

| sub-segment | timer lines | contents | device sync inside the timer? |
|---|---|---|---|
| `buffer_sample` | 147–155 | `ReplayBuffer.sample()` = `np.random.randint` + 5 NumPy fancy-index gathers on **host** (float32 arrays, pageable memory), then 5 × `torch.as_tensor(x, device=cuda)` = blocking (non-`non_blocking`) host→device copies from pageable memory (no `pin_memory`) | no explicit `torch.cuda.synchronize()`; how long each blocking copy waits for previously queued GPU work is determined by PyTorch/CUDA copy semantics and is **not measured** here |
| `critic_update` | 157–178 | no-grad target: actor forward on `next_obs` (sampling + log-prob), both target-Q forwards, min, entropy term, Bellman target; Q1/Q2 forward, MSE losses; `q1` zero_grad/backward/Adam step, then `q2` zero_grad/backward/Adam step | **none**: `q1_loss.item()`/`q2_loss.item()` are on line 179, *after* the timer stops |
| `actor_update` (+ α) | 181–211 | freeze Q params (`requires_grad=False`), actor forward on `obs` (rsample + tanh-corrected log-prob), Q1/Q2 forward on (obs, new action), min, actor loss, backward, Adam step, unfreeze Q params, α loss, backward, Adam step on `log_alpha` (runs every update since `policy_update_delay = 1`) | **none**: `actor_loss.item()` and `self.alpha.item()` are on lines 212–213, *after* the timer stops |
| `target_update` | 215–224 | Polyak averaging under `no_grad`: for every parameter of q1→q1_targ and q2→q2_targ, `mul_(1-τ)` then `add_(τ·p)` (a Python loop over 2 × 6 parameter tensors; `τ·p` creates a temporary) | none |

- `time.perf_counter()` measures **host wall-clock** time. CUDA kernels are launched asynchronously and there is no
  `torch.cuda.synchronize()` anywhere in the repo (grep, 2.3), so GPU work launched inside one timer can still be
  executing when that timer stops; the `.item()` calls *between* the timers (lines 179, 212–213)
  are device→host copies of results, so they wait for the kernels producing them, and they lie outside every timer.
  Host time spent waiting in those `.item()` calls is inside the CodeCarbon `gradient_updates` task but in no
  sub-timer (→ allocation coverage < 100 %, Part 8.2).
- Allocation (lines 395–402): the *whole run's* `gradient_updates` energy (sum of the 100 measured tasks)
  is split by the *whole run's* summed sub-timer shares `T_k / Σ_j T_j` (one split per run, not per epoch, despite
  the `train()` docstring saying "within that epoch").
- `select_action` (line 139): `torch.as_tensor(obs, float32, device)` (H2D copy of one observation), actor
  forward with `with_logprob=False` (stochastic: `dist.rsample()`), `.cpu().numpy()` (D2H copy = device sync every
  env step).
- **`rollout` task contents** (lines 324–342): per env step: `select_action`, `env.step(action)` (MuJoCo on CPU),
  `buffer.add(...)` (5 host array writes), reward bookkeeping, and on `terminated or truncated` an episode record +
  `env.reset()`. The `rollout` task therefore includes episode resets.
- **Evaluation episodes: none.** `train()` contains no deterministic/evaluation rollout anywhere — neither inside nor
  outside the measured tasks. All logged returns are those of the stochastic training policy.
- Between the two tasks of an epoch, only Python bookkeeping (dict updates, `epochs_log.append`, a `logger.info` every
  10 epochs) runs outside any task. `warmup` (random actions, `env.action_space.sample()`, 5000 steps) is its own task.
- Seeding: `experiment_runner` seeds `random`, `numpy`, `torch.manual_seed` with the run seed; `_make_env` calls
  `env.reset(seed)` and `env.action_space.seed(seed)`; `train()` calls `env.reset(seed=exp_cfg.seed)` again. No
  CUDA determinism flags are set.

**2.3 Numerics / performance flags** — searched all 24 `.py` files of the repo (excluding `.venv/` and `contexts/`):


| pattern | occurrences |
|---|---|
| `allow_tf32` | none found |
| `set_float32_matmul_precision` | none found |
| `torch.compile` | algorithms/tdmpc2.py:53 (`encoder, no `torch.compile`/vmap ensemble tricks: the Q-ensemble is a`) |
| `autocast` | none found |
| `torch.cuda.amp` | none found |
| `GradScaler` | none found |
| `cuda.graphs` | none found |
| `CUDAGraph` | none found |
| `cudnn.deterministic` | none found |
| `cudnn.benchmark` | none found |
| `pin_memory` | none found |
| `non_blocking` | none found |
| `synchronize` | flop_analysis/make_fix_report.py:264 (`timer ran next. Together with the missing `torch.cuda.synchronize()`, this shift`) |
| `set_default_dtype` | none found |
| `.half(` | none found |
| `bfloat16` | none found |
| `float16` | none found |
| `set_num_threads` | none found |
| `use_deterministic_algorithms` | none found |


- The hits listed (with the matching line) are prose inside a docstring / report text, not code; none is in
  `algorithms/sac.py`, `replay_buffer.py`, `tracker_utils.py`, `experiment_runner.py` or `run_experiment.py`. The
  code therefore **sets none** of TF32, matmul precision, `torch.compile`,
  AMP/autocast, CUDA graphs, cuDNN flags or pinned memory. Whatever the PyTorch defaults of torch 2.13.0+cu130 are,
  they apply.
- dtype: all networks are default-constructed `nn.Linear` (float32 parameters); the replay buffer stores float32;
  `select_action` casts observations to float32. → all SAC matmuls are float32-typed tensors.
- Local evidence about defaults (Windows venv, torch 2.14.0+cpu, **not** the run stack):
  `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.get_float32_matmul_precision() = 'highest'`. Whether FP32 matmuls ran as
  TF32 on the RTX 5090 under 2.13.0+cu130 is **NOT VERIFIED** from the run data (no run logs these flags); confirm on the
  Linux box with `rl-exp/bin/python -c "import torch;print(torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())"`.

**2.4 Network sizes** — real `GaussianPolicy`/`QNetwork` classes instantiated on CPU with the env dims below (parameter count = `sum(p.numel())`):


| env | hidden | actor (incl. μ and log σ heads) | one Q-network | both critics | both target critics | all 5 networks |
|---|---|---|---|---|---|---|
| HalfCheetah-v5 | (256,256) | 73484 | 72193 | 144386 | 144386 | 362256 |
| HalfCheetah-v5 | (512,512) | 278028 | 275457 | 550914 | 550914 | 1379856 |
| HalfCheetah-v5 | (1024,1024) | 1080332 | 1075201 | 2150402 | 2150402 | 5381136 |
| Ant-v5 | (256,256) | 97040 | 95233 | 190466 | 190466 | 477972 |
| Ant-v5 | (512,512) | 325136 | 321537 | 643074 | 643074 | 1611284 |
| Ant-v5 | (1024,1024) | 1174544 | 1167361 | 2334722 | 2334722 | 5843988 |


Environment dimensions (`gymnasium.make(env_id)` with **no kwargs**, exactly as `experiment_runner._make_env` does; instantiated locally with gymnasium 1.3.0). `metadata.json` does **not** store obs/act dims or gymnasium/mujoco versions; `flops_per_call.json` stores the dims `measure_flops.py` saw:


| env | obs dim | act dim | action bounds | max_episode_steps | frame_skip | dt (s) | flops_per_call.json obs/act | spec kwargs |
|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | 17 | 6 | [-1.0, 1.0] | 1000 | 5 | 0.05 | 17 / 6 | {} |
| Ant-v5 | 105 | 8 | [-1.0, 1.0] | 1000 | 5 | 0.05 | 105 / 8 | {} |


- **PASS** — local env dims equal the dims recorded in flops_per_call.json (HC 17/6, Ant 105/8 — the draft's assumed Ant obs = 105 is confirmed)
- `HalfCheetah-v5` unwrapped settings (defaults, since `gym.make` gets no kwargs): `{"_forward_reward_weight": 1.0, "_ctrl_cost_weight": 0.1, "_reset_noise_scale": 0.1, "_exclude_current_positions_from_observation": true}`
- `Ant-v5` unwrapped settings (defaults, since `gym.make` gets no kwargs): `{"_forward_reward_weight": 1, "_ctrl_cost_weight": 0.5, "_contact_cost_weight": 0.0005, "_healthy_reward": 1.0, "_terminate_when_unhealthy": true, "_healthy_z_range": [0.2, 1.0], "_contact_force_range": [-1.0, 1.0], "_main_body": 1, "_reset_noise_scale": 0.1, "_exclude_current_positions_from_observation": true, "_include_cfrc_ext_in_observation": true}`
- Ant-v5 therefore runs with Gymnasium's **default contact-force settings** (no kwargs are passed): the 105-dim observation includes the external contact forces (`include_cfrc_ext_in_observation=True` above), and `terminate_when_unhealthy=True` with the default `healthy_z_range`. HalfCheetah-v5 has no termination condition (episodes end only by the 1000-step `TimeLimit` truncation).

**2.5 Phase constants actually used** (identical in all 70 runs — Part 1.3 one-factor check covers every
`experiment_config` field):

| constant | value | source |
|---|---|---|
| settle (unmeasured) | 30.0 s | `experiment_config.settle_seconds` |
| idle baseline head / tail | 90.0 s / 90.0 s | `experiment_config` |
| warmup | 5000 env steps (random actions) | `experiment_config.warmup_steps` |
| training | 100000 env steps = 100 epochs × 1000 | `experiment_config` |
| CodeCarbon `measure_power_secs` | 1.0 s | `experiment_config` |
| CodeCarbon `tracking_mode` | `machine` | `experiment_config` |
| `force_cpu_power` | `None` (→ RAPL used) | `experiment_config.force_cpu_power_w` |
| RAM power | no override in code; distinct per-task CSV `ram_power` values over all 70 runs: 20.00 W (CodeCarbon's RAM model) | `emissions_*.csv` |
| tracker other options | `save_to_file=True`, `output_file="emissions.csv"`, `log_level="warning"`, `allow_multiple_runs=True`; `country_iso_code` is in `ExperimentConfig` but **not passed** to `EmissionsTracker` | `experiment_runner.py` |
| GPU clock lock | 2000/2000 MHz, persistence mode True, CPU governor `performance` True | `experiment_config`, `utils/gpu_control.py` |
| thermal gate | {"thermal_gate_enabled": true, "thermal_gate_temp_tolerance_c": 2.0, "thermal_gate_power_tolerance_w": 5.0, "thermal_gate_max_wait_seconds": 600.0, "thermal_gate_poll_interval_seconds": 5.0, "thermal_gate_reference_file": "results/_thermal_reference.json"}; reference = stable reading (3 consecutive polls within 0.5 °C, 5 s apart, ≤ 120 s) captured fresh in each process | `experiment_config`, `utils/thermal_gate.py` |
| tracker lifetime | one `EmissionsTracker` started after settle; tasks: `idle_baseline_head`, `warmup_0`, `rollout_i`/`gradient_updates_i` (i = 0…99), `idle_baseline_tail`; `tracker.stop()` writes the one-row `emissions.csv` | `experiment_runner.py`, `sac.py` |


## Part 3 — FLOP facts for SAC

**3.1 `flops_per_call.json` SAC entries** (verbatim; `_device_used_for_measurement` = `cpu`). Only the 5 signatures × 2 envs used by the 70 runs are shown (Humanoid-v5 entries exist for dev runs and are omitted):

```json
{
  "HalfCheetah-v5": {
    "bs1024_h1024x1024": {
      "obs_dim": 17,
      "act_dim": 6,
      "hidden_sizes": [
        1024,
        1024
      ],
      "batch_size": 1024,
      "actor_forward_bs1": 2156544,
      "critic_fwdbwd": 19694354432,
      "actor_fwdbwd": 15380512768,
      "target_update_elementwise_ops": 4300804,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 2156544,
        "critic_fwdbwd_analytic_approx": 13186891776
      }
    },
    "bs256_h1024x1024": {
      "obs_dim": 17,
      "act_dim": 6,
      "hidden_sizes": [
        1024,
        1024
      ],
      "batch_size": 256,
      "actor_forward_bs1": 2156544,
      "critic_fwdbwd": 4923588608,
      "actor_fwdbwd": 3845128192,
      "target_update_elementwise_ops": 4300804,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 2156544,
        "critic_fwdbwd_analytic_approx": 3296722944
      }
    },
    "bs256_h256x256": {
      "obs_dim": 17,
      "act_dim": 6,
      "hidden_sizes": [
        256,
        256
      ],
      "batch_size": 256,
      "actor_forward_bs1": 145920,
      "critic_fwdbwd": 324927488,
      "actor_fwdbwd": 256638976,
      "target_update_elementwise_ops": 288772,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 145920,
        "critic_fwdbwd_analytic_approx": 220200960
      }
    },
    "bs256_h512x512": {
      "obs_dim": 17,
      "act_dim": 6,
      "hidden_sizes": [
        512,
        512
      ],
      "batch_size": 256,
      "actor_forward_bs1": 553984,
      "critic_fwdbwd": 1253834752,
      "actor_fwdbwd": 983040000,
      "target_update_elementwise_ops": 1101828,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 553984,
        "critic_fwdbwd_analytic_approx": 843055104
      }
    },
    "bs512_h1024x1024": {
      "obs_dim": 17,
      "act_dim": 6,
      "hidden_sizes": [
        1024,
        1024
      ],
      "batch_size": 512,
      "actor_forward_bs1": 2156544,
      "critic_fwdbwd": 9847177216,
      "actor_fwdbwd": 7690256384,
      "target_update_elementwise_ops": 4300804,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 2156544,
        "critic_fwdbwd_analytic_approx": 6593445888
      }
    }
  },
  "Ant-v5": {
    "bs1024_h1024x1024": {
      "obs_dim": 105,
      "act_dim": 8,
      "hidden_sizes": [
        1024,
        1024
      ],
      "batch_size": 1024,
      "actor_forward_bs1": 2344960,
      "critic_fwdbwd": 21019754496,
      "actor_fwdbwd": 16529752064,
      "target_update_elementwise_ops": 4669444,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 2344960,
        "critic_fwdbwd_analytic_approx": 14319353856
      }
    },
    "bs256_h1024x1024": {
      "obs_dim": 105,
      "act_dim": 8,
      "hidden_sizes": [
        1024,
        1024
      ],
      "batch_size": 256,
      "actor_forward_bs1": 2344960,
      "critic_fwdbwd": 5254938624,
      "actor_fwdbwd": 4132438016,
      "target_update_elementwise_ops": 4669444,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 2344960,
        "critic_fwdbwd_analytic_approx": 3579838464
      }
    },
    "bs256_h256x256": {
      "obs_dim": 105,
      "act_dim": 8,
      "hidden_sizes": [
        256,
        256
      ],
      "batch_size": 256,
      "actor_forward_bs1": 193024,
      "critic_fwdbwd": 407764992,
      "actor_fwdbwd": 328466432,
      "target_update_elementwise_ops": 380932,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 193024,
        "critic_fwdbwd_analytic_approx": 290979840
      }
    },
    "bs256_h512x512": {
      "obs_dim": 105,
      "act_dim": 8,
      "hidden_sizes": [
        512,
        512
      ],
      "batch_size": 256,
      "actor_forward_bs1": 648192,
      "critic_fwdbwd": 1419509760,
      "actor_fwdbwd": 1126694912,
      "target_update_elementwise_ops": 1286148,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 648192,
        "critic_fwdbwd_analytic_approx": 984612864
      }
    },
    "bs512_h1024x1024": {
      "obs_dim": 105,
      "act_dim": 8,
      "hidden_sizes": [
        1024,
        1024
      ],
      "batch_size": 512,
      "actor_forward_bs1": 2344960,
      "critic_fwdbwd": 10509877248,
      "actor_fwdbwd": 8264876032,
      "target_update_elementwise_ops": 4669444,
      "_analytic_cross_check": {
        "actor_forward_bs1_analytic": 2344960,
        "critic_fwdbwd_analytic_approx": 7159676928
      }
    }
  }
}
```
Field meanings (from `measure_flops.measure_sac`, run under `torch.utils.flop_counter.FlopCounterMode`
on the real classes; FlopCounterMode counts matmul/addmm/bmm-type FLOPs = 2·m·n·k, i.e. one multiply-add = 2 FLOPs):
- `actor_forward_bs1` — `rollout` per-call FLOPs: one `GaussianPolicy` forward at batch 1, no grad, `with_logprob=False`.
- `critic_fwdbwd` — `critic_update` per call at batch B: the no-grad target computation (actor forward on next_obs +
  both target-Q forwards) **plus** Q1/Q2 forward, backward and Adam steps (Adam's elementwise ops are not matmuls → 0).
- `actor_fwdbwd` — `actor_update` per call at batch B: actor forward, Q1/Q2 forward on the new action, backward
  (actor weight grads + input grads through the frozen critics), α loss/backward (elementwise → 0).
- `target_update_elementwise_ops` — `2 × (params(q1) + params(q2))` = counted as 2 elementwise ops (mul + add) per
  critic parameter per Polyak update (the `τ·p` temporary is not counted). **Not a matmul FLOP count.**
- `_analytic_cross_check` — stored by `measure_flops.py`: `actor_forward_bs1_analytic` = 2·Σ n_in·n_out (incl. both
  heads); `critic_fwdbwd_analytic_approx` = 3 × 2 × (Q forward at batch B), i.e. twin-Q fwd+bwd only — it omits the
  target forward pass and the reduced first-layer backward, so it is not expected to match.

**3.2 Call-count formulas** (`compute_energy_per_flop.sac_like_call_counts` / `rows_for_sac_or_td3`):

```
n_epochs = max(1, train_steps // steps_per_epoch)        # = 100
n_env    = n_epochs * steps_per_epoch                      # = 100000  (warmup steps are NOT counted; warmup has no FLOP row)
n_upd    = n_env * updates_per_env_step                    # critic_update and buffer_sample calls
n_act    = ceil(n_upd / policy_update_delay)               # = n_upd for SAC (delay = 1)
target_update calls = n_upd                                # SAC: unconditional (TD3: n_act)
F_rollout = actor_forward_bs1 * n_env
F_critic  = critic_fwdbwd    * n_upd
F_actor   = actor_fwdbwd     * n_act
OPS_target = target_update_elementwise_ops * n_upd
```

| tag | n_env (rollout calls) | n_upd (critic / buffer_sample calls) | n_act (actor calls) | target_update calls |
|---|---|---|---|---|
| base | 100000 | 100000 | 100000 | 100000 |
| utd2 | 100000 | 200000 | 200000 | 200000 |
| utd4 | 100000 | 400000 | 400000 | 400000 |


- **PASS** — call counts above equal the `call_count` column of per_run_energy_per_flop.csv for all 70 runs

**3.3 Per-run segment FLOPs** (matmul FLOPs; target = elementwise op count, not FLOPs):


| config | signature | F_rollout | F_critic | F_actor | F_grad_updates = F_critic+F_actor | F_TOTAL = F_roll+F_crit+F_act | target-update ops | identical across 5 seeds |
|---|---|---|---|---|---|---|---|---|
| HC-base | `bs256_h1024x1024` | 2.157e+11 | 4.924e+14 | 3.845e+14 | 8.769e+14 | 8.771e+14 | 4.301e+11 | yes |
| HC-utd2 | `bs256_h1024x1024` | 2.157e+11 | 9.847e+14 | 7.69e+14 | 1.754e+15 | 1.754e+15 | 8.602e+11 | yes |
| HC-utd4 | `bs256_h1024x1024` | 2.157e+11 | 1.969e+15 | 1.538e+15 | 3.507e+15 | 3.508e+15 | 1.72e+12 | yes |
| HC-w256 | `bs256_h256x256` | 1.459e+10 | 3.249e+13 | 2.566e+13 | 5.816e+13 | 5.817e+13 | 2.888e+10 | yes |
| HC-w512 | `bs256_h512x512` | 5.54e+10 | 1.254e+14 | 9.83e+13 | 2.237e+14 | 2.237e+14 | 1.102e+11 | yes |
| HC-b512 | `bs512_h1024x1024` | 2.157e+11 | 9.847e+14 | 7.69e+14 | 1.754e+15 | 1.754e+15 | 4.301e+11 | yes |
| HC-b1024 | `bs1024_h1024x1024` | 2.157e+11 | 1.969e+15 | 1.538e+15 | 3.507e+15 | 3.508e+15 | 4.301e+11 | yes |
| Ant-base | `bs256_h1024x1024` | 2.345e+11 | 5.255e+14 | 4.132e+14 | 9.387e+14 | 9.39e+14 | 4.669e+11 | yes |
| Ant-utd2 | `bs256_h1024x1024` | 2.345e+11 | 1.051e+15 | 8.265e+14 | 1.877e+15 | 1.878e+15 | 9.339e+11 | yes |
| Ant-utd4 | `bs256_h1024x1024` | 2.345e+11 | 2.102e+15 | 1.653e+15 | 3.755e+15 | 3.755e+15 | 1.868e+12 | yes |
| Ant-w256 | `bs256_h256x256` | 1.93e+10 | 4.078e+13 | 3.285e+13 | 7.362e+13 | 7.364e+13 | 3.809e+10 | yes |
| Ant-w512 | `bs256_h512x512` | 6.482e+10 | 1.42e+14 | 1.127e+14 | 2.546e+14 | 2.547e+14 | 1.286e+11 | yes |
| Ant-b512 | `bs512_h1024x1024` | 2.345e+11 | 1.051e+15 | 8.265e+14 | 1.877e+15 | 1.878e+15 | 4.669e+11 | yes |
| Ant-b1024 | `bs1024_h1024x1024` | 2.345e+11 | 2.102e+15 | 1.653e+15 | 3.755e+15 | 3.755e+15 | 4.669e+11 | yes |


- **PASS** — FLOPs identical across the 5 seeds of every configuration Exact integers (full precision) are in `sac_cross_seed_summary.csv` (`flops_mean`) and `sac_segments_long.csv` (`flops`).

**3.4 Analytic cross-check.** Stored in the repo: `_analytic_cross_check` (columns 4 and 7). Additionally, *derived by the context generator (not a pipeline output)*: a full dense-layer count that mirrors the code path — forward = 2·B·n_in·n_out per layer; backward = weight-grad (2·B·n_in·n_out, every layer with trainable weights) + input-grad (2·B·n_in·n_out, every layer whose input needs a gradient, i.e. all but the first layer of a network fed by data; for the frozen critics in the actor loss only input-grads, including layer 1 since the action input needs a gradient); critic = target (actor fwd + 2 target-Q fwd) + 2 × (Q fwd + Q bwd); actor = actor fwd + actor bwd + 2 Q fwd + 2 Q input-grad bwd. Exact integer FLOPs:


| env | signature | rollout measured | rollout analytic (stored) | rollout analytic (derived) | critic measured | critic analytic_approx (stored) | critic analytic (derived) | actor measured | actor analytic (derived) | derived vs measured |
|---|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v5 | `bs256_h256x256` | 145920 | 145920 | 145920 | 324927488 | 220200960 | 324927488 | 256638976 | 256638976 | exact |
| HalfCheetah-v5 | `bs256_h512x512` | 553984 | 553984 | 553984 | 1253834752 | 843055104 | 1253834752 | 983040000 | 983040000 | exact |
| HalfCheetah-v5 | `bs256_h1024x1024` | 2156544 | 2156544 | 2156544 | 4923588608 | 3296722944 | 4923588608 | 3845128192 | 3845128192 | exact |
| HalfCheetah-v5 | `bs512_h1024x1024` | 2156544 | 2156544 | 2156544 | 9847177216 | 6593445888 | 9847177216 | 7690256384 | 7690256384 | exact |
| HalfCheetah-v5 | `bs1024_h1024x1024` | 2156544 | 2156544 | 2156544 | 19694354432 | 13186891776 | 19694354432 | 15380512768 | 15380512768 | exact |
| Ant-v5 | `bs256_h256x256` | 193024 | 193024 | 193024 | 407764992 | 290979840 | 407764992 | 328466432 | 328466432 | exact |
| Ant-v5 | `bs256_h512x512` | 648192 | 648192 | 648192 | 1419509760 | 984612864 | 1419509760 | 1126694912 | 1126694912 | exact |
| Ant-v5 | `bs256_h1024x1024` | 2344960 | 2344960 | 2344960 | 5254938624 | 3579838464 | 5254938624 | 4132438016 | 4132438016 | exact |
| Ant-v5 | `bs512_h1024x1024` | 2344960 | 2344960 | 2344960 | 10509877248 | 7159676928 | 10509877248 | 8264876032 | 8264876032 | exact |
| Ant-v5 | `bs1024_h1024x1024` | 2344960 | 2344960 | 2344960 | 21019754496 | 14319353856 | 21019754496 | 16529752064 | 16529752064 | exact |


- **PASS** — stored rollout analytic value equals the measured count for every SAC signature
- **PASS** — derived full analytic count equals FlopCounterMode exactly for rollout, critic and actor in all 10 (env, signature) entries
- The stored `critic_fwdbwd_analytic_approx` is 67.77 % of the measured value for HC-base (it omits the target forward pass), as expected from its definition.

**3.5 TOTAL_MEASURED_TRAINING definition in `compute_energy_per_flop.py`:**
- numerator = Σ energy of `TRAINING_SEGMENTS ∩ keys(segment_energy.json)` = `['actor_update', 'buffer_sample', 'critic_update', 'rollout', 'target_update']`
  for SAC = `E_rollout + (E_buffer_sample + E_critic + E_actor + E_target)` = `E_rollout + E_gradient_updates`;
- denominator = Σ `total_flops` of rows with `flop_type == "matmul"` = `F_rollout + F_critic + F_actor`
  (`target_update` is `elementwise`, `buffer_sample` is `none`/0);
- duration/power of the TOTAL row = measured `rollout` + measured `gradient_updates` task durations/energies.

- **PASS** — numerically, for all 70 runs: CSV TOTAL energy = E_rollout + Σ4 sub-segments (from segment_energy.json) (max rel. err 2.479e-16)
- **PASS** — CSV TOTAL FLOPs = F_rollout + F_critic + F_actor exactly
- **PASS** — Σ4 allocated sub-segments = Σ measured `gradient_updates_i` task energy in the per-task CSV (max rel. err 4.072e-13)

## Part 4 — Per-configuration results

**Table 4.1 — absolute energy per segment [J].** n = 5 seeds per row; mean ± sample sd (ddof = 1); 4 significant digits. Energy is **gross** (idle floor included, nothing subtracted). `rollout` and `gradient_updates` are **measured** CodeCarbon tasks; `buffer_sample`, `critic`, `actor`, `target` are **allocated** from `gradient_updates` by the run's wall-clock (`perf_counter`) time share; TOTAL = rollout + gradient_updates.


| config | rollout | buf_sample | critic | actor | target | grad_updates | TOTAL |
|---|---|---|---|---|---|---|---|
| HC-base | 2496 ± 33.35 | 1550 ± 20.91 | 24580 ± 234.9 | 29740 ± 127.1 | 2547 ± 28.80 | 58420 ± 247.8 | 60910 ± 254.1 |
| HC-utd2 | 2513 ± 28.54 | 3047 ± 32.20 | 48490 ± 414.6 | 58950 ± 230.7 | 5024 ± 24.75 | 115500 ± 621.5 | 118000 ± 623.7 |
| HC-utd4 | 2553 ± 47.45 | 6072 ± 41.76 | 97910 ± 628.5 | 118300 ± 561.9 | 10080 ± 72.58 | 232400 ± 1075 | 235000 ± 1107 |
| HC-w256 | 2278 ± 20.10 | 1189 ± 9.147 | 18970 ± 277.6 | 22990 ± 176.8 | 1976 ± 10.61 | 45130 ± 458.2 | 47410 ± 458.5 |
| HC-w512 | 2327 ± 10.01 | 1244 ± 5.171 | 19810 ± 447.3 | 24100 ± 85.68 | 2073 ± 11.87 | 47230 ± 515.4 | 49550 ± 510.5 |
| HC-b512 | 2647 ± 37.78 | 2103 ± 17.73 | 29860 ± 660.5 | 35470 ± 106.5 | 2947 ± 45.12 | 70380 ± 547.2 | 73020 ± 517.6 |
| HC-b1024 | 2748 ± 32.58 | 2960 ± 28.58 | 33300 ± 502.6 | 40040 ± 167.8 | 3321 ± 28.87 | 79620 ± 577.5 | 82370 ± 589.8 |
| Ant-base | 4332 ± 50.19 | 1923 ± 50.68 | 25130 ± 295.7 | 30440 ± 252.2 | 2558 ± 32.29 | 60050 ± 545.0 | 64380 ± 588.3 |
| Ant-utd2 | 4309 ± 66.09 | 3830 ± 39.39 | 49260 ± 437.9 | 60660 ± 240.3 | 5124 ± 42.09 | 118900 ± 666.3 | 123200 ± 726.7 |
| Ant-utd4 | 4294 ± 55.70 | 7617 ± 54.95 | 98310 ± 459.4 | 120900 ± 478.8 | 10230 ± 73.67 | 237000 ± 1031 | 241300 ± 1066 |
| Ant-w256 | 4101 ± 47.87 | 1457 ± 17.16 | 19300 ± 405.7 | 23130 ± 286.5 | 1996 ± 22.50 | 45880 ± 711.9 | 49990 ± 725.5 |
| Ant-w512 | 4154 ± 38.39 | 1545 ± 10.35 | 20060 ± 384.3 | 24350 ± 144.1 | 2097 ± 22.01 | 48050 ± 495.3 | 52200 ± 479.3 |
| Ant-b512 | 4367 ± 46.25 | 2887 ± 15.08 | 30240 ± 363.9 | 36020 ± 39.89 | 2990 ± 20.61 | 72130 ± 375.0 | 76500 ± 399.8 |
| Ant-b1024 | 4541 ± 46.10 | 4864 ± 41.13 | 34810 ± 446.0 | 41450 ± 148.2 | 3400 ± 27.33 | 84520 ± 417.1 | 89060 ± 410.8 |


Baseline rows in **kWh** (same statistic):


| config | rollout | buf_sample | critic | actor | target | grad_updates | TOTAL |
|---|---|---|---|---|---|---|---|
| HC-base | 0.0006933 ± 9.264e-06 | 0.0004306 ± 5.807e-06 | 0.006826 ± 6.525e-05 | 0.008262 ± 3.531e-05 | 0.0007074 ± 8e-06 | 0.01623 ± 6.884e-05 | 0.01692 ± 7.059e-05 |
| Ant-base | 0.001203 ± 1.394e-05 | 0.0005341 ± 1.408e-05 | 0.006981 ± 8.213e-05 | 0.008455 ± 7.006e-05 | 0.0007104 ± 8.97e-06 | 0.01668 ± 0.0001514 | 0.01788 ± 0.0001634 |


Excluded phases (not part of any total; kept for reference) — energy [J] and measured task duration [s], mean ± sd, n = 5:


| config | warmup E | idle head E | idle tail E | warmup dur | idle head dur | idle tail dur |
|---|---|---|---|---|---|---|
| HC-base | 18.38 ± 2.370 | 6028 ± 109.1 | 6262 ± 65.90 | 0.1516 ± 0.001431 | 90.00 ± 0.001185 | 90.00 ± 0.002178 |
| HC-utd2 | 16.74 ± 3.089 | 6075 ± 542.0 | 6110 ± 53.06 | 0.1527 ± 0.002528 | 90.00 ± 0.0003938 | 90.00 ± 0.0001134 |
| HC-utd4 | 17.87 ± 3.041 | 5821 ± 30.56 | 6200 ± 63.94 | 0.1535 ± 0.002633 | 90.00 ± 5.892e-05 | 90.00 ± 0.0002601 |
| HC-w256 | 16.76 ± 3.252 | 6228 ± 478.0 | 6636 ± 515.8 | 0.1513 ± 0.001999 | 90.00 ± 0.0002347 | 90.00 ± 0.0001248 |
| HC-w512 | 19.49 ± 2.619 | 6372 ± 551.9 | 6412 ± 544.6 | 0.1515 ± 0.003196 | 90.00 ± 9.878e-05 | 90.00 ± 0.0001067 |
| HC-b512 | 19.45 ± 3.294 | 5760 ± 110.8 | 6100 ± 31.33 | 0.1540 ± 0.004070 | 90.00 ± 0.0005039 | 90.00 ± 0.000178 |
| HC-b1024 | 16.46 ± 3.092 | 5689 ± 62.32 | 6380 ± 53.84 | 0.1536 ± 0.001981 | 90.00 ± 6.022e-05 | 90.00 ± 4.445e-05 |
| Ant-base | 74.67 ± 3.704 | 5966 ± 183.1 | 6517 ± 302.5 | 0.6654 ± 0.007628 | 90.00 ± 0.0004579 | 90.00 ± 0.0005883 |
| Ant-utd2 | 76.39 ± 3.972 | 6213 ± 223.9 | 6420 ± 140.7 | 0.6715 ± 0.003009 | 90.00 ± 0.0008972 | 90.00 ± 0.0003768 |
| Ant-utd4 | 78.81 ± 0.8415 | 6160 ± 35.81 | 6401 ± 37.15 | 0.6727 ± 0.007365 | 90.00 ± 0.0001526 | 90.00 ± 0.0005033 |
| Ant-w256 | 78.23 ± 2.610 | 6157 ± 491.5 | 6624 ± 513.9 | 0.6724 ± 0.01011 | 90.00 ± 0.001145 | 90.00 ± 9.324e-05 |
| Ant-w512 | 78.23 ± 4.336 | 6400 ± 534.7 | 6430 ± 583.5 | 0.6675 ± 0.006800 | 90.00 ± 0.0005227 | 90.00 ± 0.0002429 |
| Ant-b512 | 73.75 ± 3.176 | 5813 ± 171.9 | 6074 ± 42.78 | 0.6670 ± 0.005694 | 90.00 ± 0.0003999 | 90.00 ± 0.0002998 |
| Ant-b1024 | 75.57 ± 2.711 | 5648 ± 59.60 | 6283 ± 42.07 | 0.6710 ± 0.006886 | 90.00 ± 5.287e-05 | 90.00 ± 0.0001594 |


**Table 4.2 — shares [%].** Left block: share of `TOTAL_MEASURED_TRAINING` energy, mean ± sd of the per-seed shares; right block: share of each allocated sub-segment **within** `gradient_updates` (= its time share `T_k/ΣT_j`, by construction). n = 5, ddof = 1. **PASS** — shares sum to 100 % in every run (both blocks) (max |Σ−100| = 2.84e-14 pp)


| config | rollout | buf_sample | critic | actor | target | grad_updates (= 100 − rollout) | buf \| GU | critic \| GU | actor \| GU | target \| GU |
|---|---|---|---|---|---|---|---|---|---|---|
| HC-base | 4.097 ± 0.05304 | 2.545 ± 0.02884 | 40.35 ± 0.2984 | 48.83 ± 0.2100 | 4.181 ± 0.04275 | 95.90 ± 0.05304 | 2.654 ± 0.03101 | 42.07 ± 0.2901 | 50.92 ± 0.2443 | 4.360 ± 0.04565 |
| HC-utd2 | 2.130 ± 0.02556 | 2.582 ± 0.01936 | 41.08 ± 0.1710 | 49.95 ± 0.1357 | 4.257 ± 0.02196 | 97.87 ± 0.02556 | 2.638 ± 0.01947 | 41.98 ± 0.1667 | 51.04 ± 0.1483 | 4.349 ± 0.02323 |
| HC-utd4 | 1.086 ± 0.01713 | 2.584 ± 0.01470 | 41.67 ± 0.1217 | 50.37 ± 0.08554 | 4.290 ± 0.04630 | 98.91 ± 0.01713 | 2.613 ± 0.01488 | 42.13 ± 0.1218 | 50.92 ± 0.09096 | 4.337 ± 0.04628 |
| HC-w256 | 4.806 ± 0.06171 | 2.509 ± 0.01472 | 40.02 ± 0.2237 | 48.50 ± 0.1434 | 4.168 ± 0.02750 | 95.19 ± 0.06171 | 2.636 ± 0.01672 | 42.04 ± 0.2100 | 50.94 ± 0.1773 | 4.378 ± 0.03143 |
| HC-w512 | 4.696 ± 0.06084 | 2.511 ± 0.02701 | 39.97 ± 0.4959 | 48.64 ± 0.3475 | 4.184 ± 0.06353 | 95.30 ± 0.06084 | 2.635 ± 0.02990 | 41.94 ± 0.4939 | 51.04 ± 0.3969 | 4.390 ± 0.06943 |
| HC-b512 | 3.625 ± 0.07336 | 2.880 ± 0.03909 | 40.89 ± 0.6248 | 48.57 ± 0.4315 | 4.037 ± 0.08678 | 96.37 ± 0.07336 | 2.989 ± 0.04250 | 42.42 ± 0.6175 | 50.40 ± 0.4845 | 4.189 ± 0.09298 |
| HC-b1024 | 3.337 ± 0.03725 | 3.594 ± 0.05691 | 40.43 ± 0.3510 | 48.61 ± 0.2454 | 4.032 ± 0.04482 | 96.66 ± 0.03725 | 3.718 ± 0.05882 | 41.83 ± 0.3541 | 50.29 ± 0.2653 | 4.171 ± 0.04752 |
| Ant-base | 6.729 ± 0.03840 | 2.987 ± 0.06432 | 39.04 ± 0.2427 | 47.28 ± 0.1424 | 3.972 ± 0.03581 | 93.27 ± 0.03840 | 3.202 ± 0.07023 | 41.85 ± 0.2445 | 50.69 ± 0.1699 | 4.259 ± 0.03889 |
| Ant-utd2 | 3.498 ± 0.03565 | 3.109 ± 0.02467 | 39.99 ± 0.1704 | 49.25 ± 0.1483 | 4.160 ± 0.02586 | 96.50 ± 0.03565 | 3.222 ± 0.02568 | 41.44 ± 0.1792 | 51.03 ± 0.1467 | 4.310 ± 0.02695 |
| Ant-utd4 | 1.779 ± 0.01907 | 3.156 ± 0.01163 | 40.74 ± 0.04405 | 50.09 ± 0.05042 | 4.240 ± 0.01629 | 98.22 ± 0.01907 | 3.213 ± 0.01156 | 41.47 ± 0.05059 | 51.00 ± 0.04316 | 4.316 ± 0.01661 |
| Ant-w256 | 8.205 ± 0.1269 | 2.916 ± 0.01575 | 38.62 ± 0.2970 | 46.27 ± 0.1600 | 3.993 ± 0.03075 | 91.80 ± 0.1269 | 3.176 ± 0.01952 | 42.07 ± 0.2729 | 50.41 ± 0.2305 | 4.350 ± 0.03676 |
| Ant-w512 | 7.959 ± 0.1226 | 2.959 ± 0.02145 | 38.42 ± 0.4021 | 46.64 ± 0.2658 | 4.018 ± 0.05715 | 92.04 ± 0.1226 | 3.215 ± 0.02506 | 41.74 ± 0.4017 | 50.68 ± 0.3271 | 4.365 ± 0.06336 |
| Ant-b512 | 5.709 ± 0.04980 | 3.774 ± 0.03167 | 39.52 ± 0.2709 | 47.08 ± 0.2009 | 3.909 ± 0.04598 | 94.29 ± 0.04980 | 4.003 ± 0.03322 | 41.92 ± 0.2876 | 49.93 ± 0.2101 | 4.146 ± 0.04872 |
| Ant-b1024 | 5.099 ± 0.05861 | 5.462 ± 0.06766 | 39.08 ± 0.3477 | 46.54 ± 0.2182 | 3.818 ± 0.04178 | 94.90 ± 0.05861 | 5.755 ± 0.07334 | 41.18 ± 0.3464 | 49.04 ± 0.2509 | 4.023 ± 0.04629 |


**Table 4.3 — duration [s] and mean power [W].** Durations of `rollout`/`gradient_updates` = Σ of the 100 CodeCarbon task `duration`s (per-task CSV); allocated sub-segments = summed `perf_counter` times stored in `segment_energy.json["_sub_segment_wall_time_seconds"]` (host wall time, see Part 2.2); TOTAL = rollout + gradient_updates task durations. Power = per-seed energy / duration, then mean ± sd (n = 5). Power of allocated sub-segments is not tabulated here (it is `E_k/T_k` = gradient_updates energy / ΣT, identical for all four by construction; see the long CSV).


| config | T rollout | T buf_sample | T critic | T actor | T target | T grad_updates | T TOTAL | P rollout | P grad_updates | P TOTAL |
|---|---|---|---|---|---|---|---|---|---|---|
| HC-base | 20.48 ± 0.2653 | 10.45 ± 0.1429 | 165.7 ± 1.831 | 200.5 ± 0.8227 | 17.17 ± 0.1924 | 404.9 ± 2.235 | 425.3 ± 2.266 | 121.9 ± 0.4881 | 144.3 ± 0.2193 | 143.2 ± 0.1949 |
| HC-utd2 | 20.39 ± 0.3523 | 20.67 ± 0.1846 | 328.9 ± 3.054 | 399.9 ± 2.041 | 34.08 ± 0.2107 | 805.2 ± 4.927 | 825.6 ± 5.178 | 123.3 ± 1.137 | 143.5 ± 0.4785 | 143.0 ± 0.4725 |
| HC-utd4 | 20.49 ± 0.3876 | 41.22 ± 0.2172 | 664.6 ± 5.448 | 803.3 ± 4.975 | 68.41 ± 0.3872 | 1621 ± 9.772 | 1641 ± 10.04 | 124.6 ± 1.033 | 143.4 ± 0.6495 | 143.2 ± 0.6451 |
| HC-w256 | 19.70 ± 0.08942 | 10.18 ± 0.06132 | 162.4 ± 1.882 | 196.8 ± 0.6891 | 16.91 ± 0.07667 | 395.5 ± 2.521 | 415.2 ± 2.465 | 115.7 ± 0.6538 | 114.1 ± 0.5897 | 114.2 ± 0.5769 |
| HC-w512 | 19.71 ± 0.1116 | 10.20 ± 0.04743 | 162.4 ± 4.371 | 197.5 ± 1.508 | 16.99 ± 0.07852 | 397.2 ± 5.738 | 416.9 ± 5.756 | 118.1 ± 0.8487 | 118.9 ± 0.5117 | 118.9 ± 0.5006 |
| HC-b512 | 20.42 ± 0.4229 | 12.15 ± 0.08407 | 172.6 ± 5.076 | 204.9 ± 1.375 | 17.03 ± 0.1673 | 417.2 ± 6.049 | 437.7 ± 5.932 | 129.6 ± 1.388 | 168.7 ± 1.162 | 166.9 ± 1.111 |
| HC-b1024 | 20.48 ± 0.3517 | 15.16 ± 0.09508 | 170.6 ± 3.737 | 205.0 ± 1.985 | 17.00 ± 0.1690 | 418.5 ± 5.649 | 439.0 ± 5.900 | 134.2 ± 1.611 | 190.2 ± 1.223 | 187.6 ± 1.207 |
| Ant-base | 35.99 ± 0.2880 | 12.77 ± 0.2667 | 166.9 ± 2.033 | 202.1 ± 0.9564 | 16.99 ± 0.1912 | 413.1 ± 2.852 | 449.1 ± 2.955 | 120.4 ± 0.7125 | 145.4 ± 0.7271 | 143.4 ± 0.7101 |
| Ant-utd2 | 36.24 ± 0.4416 | 25.55 ± 0.3168 | 328.6 ± 3.776 | 404.7 ± 2.471 | 34.18 ± 0.3297 | 820.9 ± 6.901 | 857.1 ± 7.269 | 118.9 ± 1.030 | 144.8 ± 0.5215 | 143.7 ± 0.5065 |
| Ant-utd4 | 36.47 ± 0.2393 | 51.00 ± 0.4182 | 658.2 ± 3.830 | 809.4 ± 4.166 | 68.51 ± 0.5788 | 1642 ± 8.835 | 1678 ± 9.020 | 117.7 ± 1.464 | 144.4 ± 0.2214 | 143.8 ± 0.1999 |
| Ant-w256 | 35.34 ± 0.2785 | 12.49 ± 0.1434 | 165.4 ± 3.278 | 198.2 ± 2.271 | 17.11 ± 0.1822 | 402.4 ± 5.653 | 437.8 ± 5.831 | 116.0 ± 1.144 | 114.0 ± 0.3764 | 114.2 ± 0.4104 |
| Ant-w512 | 35.25 ± 0.1413 | 12.51 ± 0.05577 | 162.5 ± 3.384 | 197.3 ± 1.147 | 16.99 ± 0.1153 | 402.1 ± 4.392 | 437.3 ± 4.429 | 117.8 ± 0.8314 | 119.5 ± 0.6624 | 119.4 ± 0.5551 |
| Ant-b512 | 35.54 ± 0.08347 | 16.37 ± 0.09527 | 171.4 ± 2.814 | 204.2 ± 1.102 | 16.95 ± 0.06125 | 422.9 ± 3.954 | 458.5 ± 3.996 | 122.9 ± 1.449 | 170.6 ± 0.7048 | 166.9 ± 0.5840 |
| Ant-b1024 | 35.74 ± 0.3016 | 24.19 ± 0.1495 | 173.1 ± 3.262 | 206.1 ± 1.813 | 16.91 ± 0.1215 | 435.4 ± 4.704 | 471.1 ± 4.926 | 127.1 ± 1.291 | 194.1 ± 1.168 | 189.0 ± 1.147 |


**Table 4.4 — energy per FLOP.** J/FLOP = **ratio of means** (mean energy over 5 seeds / mean FLOPs; FLOPs are identical across seeds so this equals the mean of per-seed ratios), shown as `ratio [min, max] sd` of the 5 per-seed ratios. `target` is **J per Polyak elementwise op, not J/FLOP**. Gross energy; critic/actor/target are allocated values.


| config | rollout J/FLOP | critic J/FLOP | actor J/FLOP | TOTAL J/FLOP | target J/op |
|---|---|---|---|---|---|
| HC-base | 1.157e-08 [1.146e-08, 1.184e-08] sd 1.547e-10 | 4.991e-11 [4.936e-11, 5.03e-11] sd 4.771e-13 | 7.735e-11 [7.703e-11, 7.789e-11] sd 3.306e-13 | 6.945e-11 [6.896e-11, 6.97e-11] sd 2.897e-13 | 5.921e-09 [5.852e-09, 6.005e-09] sd 6.697e-11 |
| HC-utd2 | 1.165e-08 [1.152e-08, 1.186e-08] sd 1.324e-10 | 4.924e-11 [4.86e-11, 4.971e-11] sd 4.21e-13 | 7.666e-11 [7.634e-11, 7.701e-11] sd 2.999e-13 | 6.729e-11 [6.68e-11, 6.767e-11] sd 3.556e-13 | 5.84e-09 [5.802e-09, 5.868e-09] sd 2.877e-11 |
| HC-utd4 | 1.184e-08 [1.159e-08, 1.215e-08] sd 2.2e-10 | 4.972e-11 [4.932e-11, 5.005e-11] sd 3.191e-13 | 7.694e-11 [7.643e-11, 7.737e-11] sd 3.653e-13 | 6.698e-11 [6.653e-11, 6.735e-11] sd 3.155e-13 | 5.859e-09 [5.801e-09, 5.901e-09] sd 4.219e-11 |
| HC-w256 | 1.561e-07 [1.541e-07, 1.576e-07] sd 1.377e-09 | 5.839e-10 [5.734e-10, 5.94e-10] sd 8.543e-12 | 8.958e-10 [8.89e-10, 9.065e-10] sd 6.889e-12 | 8.15e-10 [8.055e-10, 8.255e-10] sd 7.882e-12 | 6.842e-08 [6.806e-08, 6.894e-08] sd 3.676e-10 |
| HC-w512 | 4.2e-08 [4.187e-08, 4.23e-08] sd 1.806e-10 | 1.58e-10 [1.535e-10, 1.619e-10] sd 3.568e-12 | 2.452e-10 [2.441e-10, 2.463e-10] sd 8.715e-13 | 2.215e-10 [2.19e-10, 2.241e-10] sd 2.282e-12 | 1.881e-08 [1.868e-08, 1.898e-08] sd 1.077e-10 |
| HC-b512 | 1.227e-08 [1.208e-08, 1.254e-08] sd 1.752e-10 | 3.032e-11 [2.967e-11, 3.112e-11] sd 6.707e-13 | 4.612e-11 [4.599e-11, 4.634e-11] sd 1.385e-13 | 4.163e-11 [4.127e-11, 4.199e-11] sd 2.951e-13 | 6.853e-09 [6.74e-09, 6.978e-09] sd 1.049e-10 |
| HC-b1024 | 1.274e-08 [1.257e-08, 1.289e-08] sd 1.511e-10 | 1.691e-11 [1.65e-11, 1.712e-11] sd 2.552e-13 | 2.603e-11 [2.595e-11, 2.621e-11] sd 1.091e-13 | 2.348e-11 [2.324e-11, 2.365e-11] sd 1.681e-13 | 7.721e-09 [7.638e-09, 7.798e-09] sd 6.712e-11 |
| Ant-base | 1.847e-08 [1.819e-08, 1.879e-08] sd 2.14e-10 | 4.783e-11 [4.702e-11, 4.861e-11] sd 5.626e-13 | 7.365e-11 [7.277e-11, 7.448e-11] sd 6.104e-13 | 6.857e-11 [6.756e-11, 6.916e-11] sd 6.265e-13 | 5.477e-09 [5.39e-09, 5.55e-09] sd 6.916e-11 |
| Ant-utd2 | 1.837e-08 [1.796e-08, 1.87e-08] sd 2.818e-10 | 4.687e-11 [4.627e-11, 4.734e-11] sd 4.167e-13 | 7.34e-11 [7.31e-11, 7.383e-11] sd 2.908e-13 | 6.56e-11 [6.508e-11, 6.617e-11] sd 3.87e-13 | 5.487e-09 [5.431e-09, 5.556e-09] sd 4.507e-11 |
| Ant-utd4 | 1.831e-08 [1.799e-08, 1.859e-08] sd 2.375e-10 | 4.677e-11 [4.653e-11, 4.707e-11] sd 2.185e-13 | 7.313e-11 [7.285e-11, 7.357e-11] sd 2.897e-13 | 6.426e-11 [6.399e-11, 6.469e-11] sd 2.838e-13 | 5.478e-09 [5.424e-09, 5.534e-09] sd 3.944e-11 |
| Ant-w256 | 2.124e-07 [2.096e-07, 2.153e-07] sd 2.48e-09 | 4.734e-10 [4.629e-10, 4.878e-10] sd 9.95e-12 | 7.041e-10 [6.978e-10, 7.188e-10] sd 8.721e-12 | 6.788e-10 [6.715e-10, 6.944e-10] sd 9.852e-12 | 5.24e-08 [5.169e-08, 5.321e-08] sd 5.908e-10 |
| Ant-w512 | 6.409e-08 [6.357e-08, 6.478e-08] sd 5.922e-10 | 1.413e-10 [1.377e-10, 1.445e-10] sd 2.708e-12 | 2.161e-10 [2.139e-10, 2.173e-10] sd 1.279e-12 | 2.05e-10 [2.02e-10, 2.067e-10] sd 1.882e-12 | 1.631e-08 [1.606e-08, 1.649e-08] sd 1.712e-10 |
| Ant-b512 | 1.862e-08 [1.828e-08, 1.877e-08] sd 1.972e-10 | 2.877e-11 [2.848e-11, 2.934e-11] sd 3.463e-13 | 4.358e-11 [4.353e-11, 4.366e-11] sd 4.827e-14 | 4.074e-11 [4.056e-11, 4.109e-11] sd 2.129e-13 | 6.404e-09 [6.338e-09, 6.447e-09] sd 4.413e-11 |
| Ant-b1024 | 1.936e-08 [1.904e-08, 1.956e-08] sd 1.966e-10 | 1.656e-11 [1.622e-11, 1.675e-11] sd 2.122e-13 | 2.507e-11 [2.498e-11, 2.522e-11] sd 8.964e-14 | 2.372e-11 [2.356e-11, 2.383e-11] sd 1.094e-13 | 7.281e-09 [7.186e-09, 7.347e-09] sd 5.854e-11 |


Same in **pJ/FLOP** (target in **µJ/op**), ratio of means ± sd of per-seed ratios:


| config | rollout pJ/FLOP | critic pJ/FLOP | actor pJ/FLOP | TOTAL pJ/FLOP | target µJ/op |
|---|---|---|---|---|---|
| HC-base | 11570 ± 154.7 | 49.91 ± 0.4771 | 77.35 ± 0.3306 | 69.45 ± 0.2897 | 0.005921 ± 6.697e-05 |
| HC-utd2 | 11650 ± 132.4 | 49.24 ± 0.4210 | 76.66 ± 0.2999 | 67.29 ± 0.3556 | 0.005840 ± 2.877e-05 |
| HC-utd4 | 11840 ± 220.0 | 49.72 ± 0.3191 | 76.94 ± 0.3653 | 66.98 ± 0.3155 | 0.005859 ± 4.219e-05 |
| HC-w256 | 156100 ± 1377 | 583.9 ± 8.543 | 895.8 ± 6.889 | 815.0 ± 7.882 | 0.06842 ± 0.0003676 |
| HC-w512 | 42000 ± 180.6 | 158.0 ± 3.568 | 245.2 ± 0.8715 | 221.5 ± 2.282 | 0.01881 ± 0.0001077 |
| HC-b512 | 12270 ± 175.2 | 30.32 ± 0.6707 | 46.12 ± 0.1385 | 41.63 ± 0.2951 | 0.006853 ± 0.0001049 |
| HC-b1024 | 12740 ± 151.1 | 16.91 ± 0.2552 | 26.03 ± 0.1091 | 23.48 ± 0.1681 | 0.007721 ± 6.712e-05 |
| Ant-base | 18470 ± 214.0 | 47.83 ± 0.5626 | 73.65 ± 0.6104 | 68.57 ± 0.6265 | 0.005477 ± 6.916e-05 |
| Ant-utd2 | 18370 ± 281.8 | 46.87 ± 0.4167 | 73.40 ± 0.2908 | 65.60 ± 0.3870 | 0.005487 ± 4.507e-05 |
| Ant-utd4 | 18310 ± 237.5 | 46.77 ± 0.2185 | 73.13 ± 0.2897 | 64.26 ± 0.2838 | 0.005478 ± 3.944e-05 |
| Ant-w256 | 212400 ± 2480 | 473.4 ± 9.950 | 704.1 ± 8.721 | 678.8 ± 9.852 | 0.05240 ± 0.0005908 |
| Ant-w512 | 64090 ± 592.2 | 141.3 ± 2.708 | 216.1 ± 1.279 | 205.0 ± 1.882 | 0.01631 ± 0.0001712 |
| Ant-b512 | 18620 ± 197.2 | 28.77 ± 0.3463 | 43.58 ± 0.04827 | 40.74 ± 0.2129 | 0.006404 ± 4.413e-05 |
| Ant-b1024 | 19360 ± 196.6 | 16.56 ± 0.2122 | 25.07 ± 0.08964 | 23.72 ± 0.1094 | 0.007281 ± 5.854e-05 |


**Table 4.5 — `gradient_updates` J/FLOP** — *derived by the context generator, not a pipeline output*: `E_gradient_updates / (F_critic + F_actor)` (measured energy, no time-share allocation involved). Ratio of means; spread over per-seed ratios.


| config | J/FLOP | pJ/FLOP | sd (pJ/FLOP) | min (pJ/FLOP) | max (pJ/FLOP) |
|---|---|---|---|---|---|
| HC-base | 6.662e-11 | 66.62 | 0.2826 | 66.16 | 66.89 |
| HC-utd2 | 6.587e-11 | 65.87 | 0.3544 | 65.37 | 66.26 |
| HC-utd4 | 6.626e-11 | 66.26 | 0.3066 | 65.83 | 66.61 |
| HC-w256 | 7.76e-10 | 776.0 | 7.878 | 766.6 | 786.5 |
| HC-w512 | 2.111e-10 | 211.1 | 2.304 | 208.5 | 213.8 |
| HC-b512 | 4.013e-11 | 40.13 | 0.3120 | 39.76 | 40.51 |
| HC-b1024 | 2.27e-11 | 22.70 | 0.1646 | 22.45 | 22.86 |
| Ant-base | 6.397e-11 | 63.97 | 0.5806 | 63.04 | 64.48 |
| Ant-utd2 | 6.332e-11 | 63.32 | 0.3549 | 62.84 | 63.84 |
| Ant-utd4 | 6.313e-11 | 63.13 | 0.2747 | 62.86 | 63.53 |
| Ant-w256 | 6.232e-10 | 623.2 | 9.669 | 615.3 | 638.6 |
| Ant-w512 | 1.887e-10 | 188.7 | 1.945 | 185.6 | 190.6 |
| Ant-b512 | 3.842e-11 | 38.42 | 0.1997 | 38.25 | 38.75 |
| Ant-b1024 | 2.251e-11 | 22.51 | 0.1111 | 22.34 | 22.62 |


Relation used in the thesis (allocated-jpf): within one run every allocated sub-segment has the same mean power `P̄ = E_GU / ΣT_j`, so `E_k/F_k = P̄ · T_k / F_k`; critic and actor J/FLOP differ only through their time per FLOP `T_k/F_k`.

**Table 4.6 — achieved throughput [GFLOP/s]** — *derived by the context generator*: FLOPs / duration per seed, mean ± sd (n = 5). Critic/actor columns divide by their `perf_counter` host time (Part 2.2 caveat: host time ≠ GPU busy time).


| config | rollout | gradient_updates | TOTAL | critic (/T_perf) | actor (/T_perf) |
|---|---|---|---|---|---|
| HC-base | 10.53 ± 0.1348 | 2166 ± 12.01 | 2062 ± 11.04 | 2972 ± 32.98 | 1918 ± 7.876 |
| HC-utd2 | 10.58 ± 0.1788 | 2178 ± 13.36 | 2125 ± 13.34 | 2994 ± 28.00 | 1923 ± 9.779 |
| HC-utd4 | 10.53 ± 0.1966 | 2164 ± 12.97 | 2137 ± 13.00 | 2963 ± 24.20 | 1915 ± 11.78 |
| HC-w256 | 0.7407 ± 0.003350 | 147.1 ± 0.9396 | 140.1 ± 0.8336 | 200.1 ± 2.329 | 130.4 ± 0.4570 |
| HC-w512 | 2.811 ± 0.01597 | 563.2 ± 8.162 | 536.7 ± 7.430 | 772.6 ± 20.94 | 497.6 ± 3.804 |
| HC-b512 | 10.56 ± 0.2135 | 4204 ± 60.84 | 4008 ± 54.26 | 5711 ± 167.0 | 3753 ± 25.27 |
| HC-b1024 | 10.53 ± 0.1783 | 8382 ± 114.2 | 7991 ± 108.3 | 11550 ± 257.6 | 7502 ± 72.83 |
| Ant-base | 6.516 ± 0.05214 | 2272 ± 15.77 | 2091 ± 13.84 | 3148 ± 38.20 | 2044 ± 9.721 |
| Ant-utd2 | 6.472 ± 0.07797 | 2287 ± 19.15 | 2191 ± 18.49 | 3199 ± 36.75 | 2042 ± 12.42 |
| Ant-utd4 | 6.430 ± 0.04204 | 2287 ± 12.25 | 2237 ± 11.97 | 3193 ± 18.50 | 2042 ± 10.46 |
| Ant-w256 | 0.5462 ± 0.004286 | 183.0 ± 2.537 | 168.2 ± 2.211 | 246.6 ± 4.844 | 165.8 ± 1.872 |
| Ant-w512 | 1.839 ± 0.007351 | 633.3 ± 6.908 | 582.4 ± 5.890 | 873.9 ± 18.12 | 571.2 ± 3.315 |
| Ant-b512 | 6.597 ± 0.01548 | 4440 ± 41.13 | 4096 ± 35.39 | 6133 ± 99.19 | 4048 ± 21.73 |
| Ant-b1024 | 6.562 ± 0.05520 | 8625 ± 93.43 | 7972 ± 83.52 | 12150 ± 231.6 | 8020 ± 70.15 |


GPU theoretical FP32 peak: **NOT DERIVED**. The repo stores neither SM count nor FP32 lanes per SM nor boost/locked-clock throughput, and `torch.cuda.get_device_properties` is not available on this Windows checkout (CPU-only torch; the RTX 5090 is on the Linux box). To derive it on the Linux box: `peak = 2 × (FP32 lanes per SM) × multi_processor_count × 2.000 GHz` with `multi_processor_count` from `torch.cuda.get_device_properties(0)` and the lanes-per-SM figure from NVIDIA's architecture documentation (not in the repo).

**Table 4.7 — per-seed totals and coefficient of variation.** Per-seed values in seed order [331, 958, 14577, 43611, 85062]; CV = sd(ddof=1)/mean × 100 %. Full precision: `sac_per_seed_totals.csv`.


| config | TOTAL energy [J] per seed | TOTAL duration [s] per seed | TOTAL mean power [W] per seed | CV E_TOTAL % | CV E_rollout % | CV E_GU % | CV T_rollout % | CV T_GU % |
|---|---|---|---|---|---|---|---|---|
| HC-base | 60920 / 61060 / 61140 / 60480 / 60950 | 425.7 / 427.1 / 427.1 / 421.6 / 425.3 | 143.1 / 143.0 / 143.1 / 143.5 / 143.3 | 0.4172 | 1.336 | 0.4243 | 1.295 | 0.5520 |
| HC-utd2 | 118700 / 118600 / 117800 / 117200 / 117900 | 826.6 / 832.4 / 826.5 / 818.0 / 824.3 | 143.6 / 142.4 / 142.6 / 143.2 / 143.0 | 0.5284 | 1.136 | 0.5381 | 1.727 | 0.6119 |
| HC-utd4 | 233400 / 236200 / 234800 / 234600 / 235700 | 1634 / 1658 / 1643 / 1635 / 1636 | 142.8 / 142.5 / 142.9 / 143.5 / 144.1 | 0.4711 | 1.859 | 0.4627 | 1.891 | 0.6030 |
| HC-w256 | 47410 / 47660 / 46860 / 48020 / 47100 | 416.7 / 417.1 / 412.3 / 417.2 / 412.7 | 113.8 / 114.3 / 113.6 / 115.1 / 114.1 | 0.9672 | 0.8822 | 1.015 | 0.4539 | 0.6375 |
| HC-w512 | 50140 / 49120 / 50000 / 49530 / 48990 | 422.3 / 412.9 / 422.6 / 417.5 / 409.5 | 118.7 / 118.9 / 118.3 / 118.6 / 119.7 | 1.030 | 0.4301 | 1.091 | 0.5663 | 1.445 |
| HC-b512 | 72960 / 73430 / 73650 / 72390 / 72690 | 435.5 / 443.1 / 444.5 / 430.5 / 434.8 | 167.6 / 165.7 / 165.7 / 168.2 / 167.2 | 0.7088 | 1.427 | 0.7776 | 2.070 | 1.450 |
| HC-b1024 | 82040 / 82780 / 82540 / 82960 / 81510 | 435.9 / 443.5 / 441.4 / 444.0 / 430.1 | 188.2 / 186.7 / 187.0 / 186.8 / 189.5 | 0.7160 | 1.186 | 0.7253 | 1.717 | 1.350 |
| Ant-base | 64820 / 64340 / 63440 / 64370 / 64940 | 451.7 / 449.3 / 444.2 / 451.0 / 449.4 | 143.5 / 143.2 / 142.8 / 142.7 / 144.5 | 0.9137 | 1.159 | 0.9076 | 0.8003 | 0.6903 |
| Ant-utd2 | 123100 / 124200 / 122200 / 123200 / 123200 | 851.9 / 868.4 / 849.7 / 857.3 / 858.3 | 144.5 / 143.1 / 143.8 / 143.7 / 143.5 | 0.5899 | 1.534 | 0.5605 | 1.219 | 0.8407 |
| Ant-utd4 | 240400 / 242900 / 241400 / 241600 / 240300 | 1671 / 1693 / 1676 / 1681 / 1672 | 143.9 / 143.5 / 144.0 / 143.8 / 143.7 | 0.4416 | 1.297 | 0.4351 | 0.6560 | 0.5381 |
| Ant-w256 | 49610 / 50260 / 49460 / 51140 / 49450 | 435.6 / 438.0 / 432.7 / 447.7 / 434.9 | 113.9 / 114.8 / 114.3 / 114.2 / 113.7 | 1.452 | 1.167 | 1.551 | 0.7880 | 1.405 |
| Ant-w512 | 52350 / 52020 / 51460 / 52540 / 52640 | 435.4 / 435.2 / 432.1 / 442.4 / 441.5 | 120.2 / 119.5 / 119.1 / 118.7 / 119.2 | 0.9182 | 0.9240 | 1.031 | 0.4008 | 1.092 |
| Ant-b512 | 76160 / 76540 / 76210 / 76420 / 77160 | 455.6 / 458.6 / 455.4 / 457.4 / 465.2 | 167.2 / 166.9 / 167.3 / 167.1 / 165.9 | 0.5226 | 1.059 | 0.5199 | 0.2348 | 0.9350 |
| Ant-b1024 | 88840 / 89170 / 89500 / 89310 / 88460 | 467.3 / 473.7 / 476.9 / 472.9 / 464.8 | 190.1 / 188.2 / 187.7 / 188.9 / 190.3 | 0.4613 | 1.015 | 0.4935 | 0.8439 | 1.080 |


Outlier flag rule (as requested): |x − median| > 3 × MAD within the configuration (MAD = median absolute deviation, **unscaled**, n = 5), applied to the energy of every segment in Table 4.1 (incl. GU and TOTAL). Flagged, **not excluded**: **53** of 490 (run, segment) cells; the largest flagged deviation from its median is 3.188 % of the median. For reference, with the normal-consistent scaled MAD (1.4826 × MAD) the same rule flags 38 cells.


| config | seed | segment | value [J] | median [J] | MAD [J] | \|x−med\|/MAD | \|x−med\| [% of median] |
|---|---|---|---|---|---|---|---|
| HC-base | 85062 | rollout | 2554 | 2481 | 9.084 | 7.983 | 2.923 |
| HC-base | 43611 | critic | 24340 | 24720 | 48.64 | 7.876 | 1.550 |
| HC-base | 85062 | critic | 24300 | 24720 | 48.64 | 8.594 | 1.691 |
| HC-base | 14577 | target | 2583 | 2532 | 14.91 | 3.409 | 2.007 |
| HC-base | 43611 | grad_updates | 58010 | 58440 | 125.6 | 3.439 | 0.7394 |
| HC-base | 43611 | TOTAL | 60480 | 60950 | 112.2 | 4.158 | 0.7657 |
| HC-utd2 | 331 | buf_sample | 3098 | 3035 | 16.02 | 3.973 | 2.098 |
| HC-utd4 | 331 | critic | 97140 | 98220 | 345.9 | 3.114 | 1.097 |
| HC-w256 | 14577 | buf_sample | 1174 | 1191 | 3.689 | 4.784 | 1.481 |
| HC-w256 | 43611 | actor | 23260 | 22930 | 111.2 | 3.033 | 1.471 |
| HC-w256 | 958 | target | 1991 | 1972 | 6.243 | 3.064 | 0.9700 |
| HC-w512 | 85062 | rollout | 2343 | 2322 | 2.484 | 8.712 | 0.9319 |
| HC-w512 | 85062 | target | 2091 | 2071 | 4.245 | 4.793 | 0.9825 |
| HC-b512 | 85062 | actor | 35640 | 35420 | 45.70 | 4.819 | 0.6218 |
| HC-b1024 | 85062 | critic | 32490 | 33560 | 148.1 | 7.225 | 3.188 |
| HC-b1024 | 958 | actor | 40060 | 39940 | 27.23 | 4.459 | 0.3040 |
| HC-b1024 | 43611 | actor | 40320 | 39940 | 27.23 | 13.71 | 0.9349 |
| Ant-base | 14577 | rollout | 4266 | 4330 | 8.959 | 7.241 | 1.498 |
| Ant-base | 85062 | rollout | 4406 | 4330 | 8.959 | 8.477 | 1.754 |
| Ant-base | 331 | critic | 25540 | 25130 | 24.38 | 16.92 | 1.641 |
| Ant-base | 14577 | critic | 24710 | 25130 | 24.38 | 17.34 | 1.683 |
| Ant-base | 14577 | actor | 30070 | 30430 | 75.89 | 4.634 | 1.156 |
| Ant-base | 85062 | actor | 30780 | 30430 | 75.89 | 4.664 | 1.163 |
| Ant-utd2 | 958 | buf_sample | 3893 | 3826 | 20.03 | 3.315 | 1.735 |
| Ant-utd2 | 958 | target | 5189 | 5123 | 10.51 | 6.289 | 1.290 |
| Ant-utd2 | 85062 | target | 5072 | 5123 | 10.51 | 4.860 | 0.9973 |
| Ant-utd2 | 958 | grad_updates | 119900 | 118900 | 128.8 | 7.623 | 0.8261 |
| Ant-utd2 | 14577 | grad_updates | 118000 | 118900 | 128.8 | 6.919 | 0.7498 |
| Ant-utd2 | 958 | TOTAL | 124200 | 123200 | 116.0 | 9.168 | 0.8630 |
| Ant-utd2 | 14577 | TOTAL | 122200 | 123200 | 116.0 | 8.480 | 0.7982 |
| Ant-utd4 | 331 | target | 10130 | 10240 | 22.59 | 4.687 | 1.034 |
| Ant-utd4 | 958 | target | 10340 | 10240 | 22.59 | 4.449 | 0.9819 |
| Ant-w256 | 43611 | buf_sample | 1487 | 1452 | 4.073 | 8.707 | 2.443 |
| Ant-w256 | 43611 | actor | 23610 | 23000 | 73.77 | 8.353 | 2.680 |
| Ant-w256 | 43611 | grad_updates | 47020 | 45570 | 268.4 | 5.399 | 3.180 |
| Ant-w256 | 958 | TOTAL | 50260 | 49610 | 161.7 | 4.022 | 1.311 |
| Ant-w256 | 43611 | TOTAL | 51140 | 49610 | 161.7 | 9.419 | 3.069 |
| Ant-w512 | 14577 | rollout | 4193 | 4133 | 12.14 | 4.906 | 1.441 |
| Ant-w512 | 43611 | rollout | 4199 | 4133 | 12.14 | 5.460 | 1.604 |
| Ant-w512 | 331 | actor | 24490 | 24380 | 15.95 | 6.618 | 0.4330 |
| Ant-w512 | 14577 | actor | 24100 | 24380 | 15.95 | 17.45 | 1.142 |
| Ant-w512 | 14577 | grad_updates | 47260 | 48220 | 301.9 | 3.166 | 1.982 |
| Ant-b512 | 331 | rollout | 4287 | 4380 | 13.79 | 6.726 | 2.118 |
| Ant-b512 | 85062 | actor | 36080 | 36010 | 20.92 | 3.466 | 0.2014 |
| Ant-b512 | 85062 | grad_updates | 72760 | 72050 | 171.1 | 4.154 | 0.9862 |
| Ant-b512 | 85062 | TOTAL | 77160 | 76420 | 206.7 | 3.580 | 0.9682 |
| Ant-b1024 | 958 | rollout | 4465 | 4550 | 12.56 | 6.834 | 1.886 |
| Ant-b1024 | 85062 | buf_sample | 4926 | 4857 | 19.34 | 3.576 | 1.424 |
| Ant-b1024 | 85062 | critic | 34080 | 35030 | 172.4 | 5.460 | 2.687 |
| Ant-b1024 | 14577 | actor | 41690 | 41410 | 49.47 | 5.663 | 0.6765 |
| Ant-b1024 | 958 | target | 3356 | 3403 | 4.801 | 9.953 | 1.404 |
| Ant-b1024 | 85062 | target | 3431 | 3403 | 4.801 | 5.671 | 0.7999 |
| Ant-b1024 | 85062 | grad_updates | 83900 | 84710 | 245.6 | 3.290 | 0.9540 |


**4.8 Reconciliation against the pipeline outputs.** Every (run, segment) and (configuration, segment) value recomputed here from `segment_energy.json`, the per-task CSV and `flops_per_call.json` is compared with `per_run_energy_per_flop.csv` (segments: rollout, the 4 allocated, TOTAL, warmup, idle head/tail) and `cross_seed_energy_per_flop.csv` (rollout, the 4 allocated, TOTAL — the only SAC segments it contains). Columns compared: energy, duration, mean power, FLOPs, J/FLOP. Note: the pipeline's cross-seed `mean_power_w` is the **mean of per-seed powers**; it is compared against the same statistic here.


| CSV | column | max relative deviation |
|---|---|---|
| cross_seed | `mean_duration_s` | 1.735e-16 |
| cross_seed | `mean_energy_joules` | 4.324e-16 |
| cross_seed | `mean_energy_per_flop_j_per_flop` | 4.058e-16 |
| cross_seed | `mean_power_w` | 1.361e-14 |
| cross_seed | `total_flops` | 0.000e+00 |
| per_run | `duration_s` | 2.096e-16 |
| per_run | `energy_per_flop_j_per_flop` | 3.861e-16 |
| per_run | `mean_power_w` | 3.777e-14 |
| per_run | `total_energy_joules` | 2.479e-16 |
| per_run | `total_flops` | 0.000e+00 |


- **PASS** — all deviations ≤ 1e-9 relative

## Part 5 — Environment comparison (HalfCheetah-v5 vs Ant-v5)

All statistics pair HalfCheetah-v5 and Ant-v5 runs **by seed** (the 5 seeds are shared). n = 5, ddof = 1, gross energy, 4 significant digits. Shares are of `TOTAL_MEASURED_TRAINING` energy. Full precision in `sac_env_comparison.csv`.

**5.1 Max absolute share difference** (difference of the mean shares, Ant − HC, percentage points; `gradient_updates` share = 100 − rollout share, so it is not listed separately):


| tag | max \|Δshare\| [pp] | attained by | Δ rollout | Δ buf_sample | Δ critic | Δ actor | Δ target |
|---|---|---|---|---|---|---|---|
| base | 2.631 | rollout | 2.631 | 0.4416 | -1.309 | -1.556 | -0.2085 |
| utd2 | 1.368 | rollout | 1.368 | 0.5268 | -1.095 | -0.7029 | -0.09683 |
| utd4 | 0.9364 | critic | 0.6926 | 0.5717 | -0.9364 | -0.2779 | -0.05004 |
| w256 | 3.399 | rollout | 3.399 | 0.4065 | -1.405 | -2.226 | -0.1742 |
| w512 | 3.263 | rollout | 3.263 | 0.4484 | -1.552 | -1.994 | -0.1656 |
| b512 | 2.084 | rollout | 2.084 | 0.8942 | -1.363 | -1.487 | -0.1274 |
| b1024 | 2.068 | actor | 1.762 | 1.868 | -1.349 | -2.068 | -0.2138 |


**5.2 Per-segment share difference Ant − HC, paired by seed** (mean ± sd of the 5 per-seed differences in pp; count of seeds with a positive difference):


| tag | rollout | buf_sample | critic | actor | target |
|---|---|---|---|---|---|
| base | 2.631 ± 0.02911 (5/5 > 0) | 0.4416 ± 0.05404 (5/5 > 0) | -1.309 ± 0.1997 (0/5 > 0) | -1.556 ± 0.2094 (0/5 > 0) | -0.2085 ± 0.03741 (0/5 > 0) |
| utd2 | 1.368 ± 0.03395 (5/5 > 0) | 0.5268 ± 0.03204 (5/5 > 0) | -1.095 ± 0.2878 (0/5 > 0) | -0.7029 ± 0.2426 (0/5 > 0) | -0.09683 ± 0.04478 (0/5 > 0) |
| utd4 | 0.6926 ± 0.01741 (5/5 > 0) | 0.5717 ± 0.02440 (5/5 > 0) | -0.9364 ± 0.1637 (0/5 > 0) | -0.2779 ± 0.1336 (0/5 > 0) | -0.05004 ± 0.06085 (1/5 > 0) |
| w256 | 3.399 ± 0.09963 (5/5 > 0) | 0.4065 ± 0.01515 (5/5 > 0) | -1.405 ± 0.1416 (0/5 > 0) | -2.226 ± 0.07728 (0/5 > 0) | -0.1742 ± 0.01840 (0/5 > 0) |
| w512 | 3.263 ± 0.1608 (5/5 > 0) | 0.4484 ± 0.04253 (5/5 > 0) | -1.552 ± 0.8102 (0/5 > 0) | -1.994 ± 0.5503 (0/5 > 0) | -0.1656 ± 0.09532 (0/5 > 0) |
| b512 | 2.084 ± 0.09889 (5/5 > 0) | 0.8942 ± 0.06350 (5/5 > 0) | -1.363 ± 0.8412 (0/5 > 0) | -1.487 ± 0.5692 (0/5 > 0) | -0.1274 ± 0.1222 (1/5 > 0) |
| b1024 | 1.762 ± 0.06440 (5/5 > 0) | 1.868 ± 0.06223 (5/5 > 0) | -1.349 ± 0.1310 (0/5 > 0) | -2.068 ± 0.1531 (0/5 > 0) | -0.2138 ± 0.02445 (0/5 > 0) |


**5.3 Power.** Mean power [W] = energy / measured task duration, per seed, mean ± sd:


| tag | P rollout HC | P rollout Ant | P GU HC | P GU Ant | P TOTAL HC | P TOTAL Ant |
|---|---|---|---|---|---|---|
| base | 121.9 ± 0.4881 | 120.4 ± 0.7125 | 144.3 ± 0.2193 | 145.4 ± 0.7271 | 143.2 ± 0.1949 | 143.4 ± 0.7101 |
| utd2 | 123.3 ± 1.137 | 118.9 ± 1.030 | 143.5 ± 0.4785 | 144.8 ± 0.5215 | 143.0 ± 0.4725 | 143.7 ± 0.5065 |
| utd4 | 124.6 ± 1.033 | 117.7 ± 1.464 | 143.4 ± 0.6495 | 144.4 ± 0.2214 | 143.2 ± 0.6451 | 143.8 ± 0.1999 |
| w256 | 115.7 ± 0.6538 | 116.0 ± 1.144 | 114.1 ± 0.5897 | 114.0 ± 0.3764 | 114.2 ± 0.5769 | 114.2 ± 0.4104 |
| w512 | 118.1 ± 0.8487 | 117.8 ± 0.8314 | 118.9 ± 0.5117 | 119.5 ± 0.6624 | 118.9 ± 0.5006 | 119.4 ± 0.5551 |
| b512 | 129.6 ± 1.388 | 122.9 ± 1.449 | 168.7 ± 1.162 | 170.6 ± 0.7048 | 166.9 ± 1.111 | 166.9 ± 0.5840 |
| b1024 | 134.2 ± 1.611 | 127.1 ± 1.291 | 190.2 ± 1.223 | 194.1 ± 1.168 | 187.6 ± 1.207 | 189.0 ± 1.147 |


Ratios: within-env `P_rollout / P_GU` (per seed, mean ± sd); ratio of ratios `(P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC` from the env means and paired by seed (mean ± sd); between-env `P_Ant / P_HC` per measured task as ratio of means (paired-by-seed mean ± sd in parentheses):


| tag | P_roll/P_GU HC | P_roll/P_GU Ant | ratio of ratios (means) | ratio of ratios (paired) | P_Ant/P_HC rollout | P_Ant/P_HC GU | P_Ant/P_HC TOTAL |
|---|---|---|---|---|---|---|---|
| base | 0.8447 ± 0.003958 | 0.8281 ± 0.002694 | 0.9803 | 0.9803 ± 0.003761 | 0.9876 (0.9876 ± 0.003703) | 1.007 (1.007 ± 0.005202) | 1.001 (1.001 ± 0.005073) |
| utd2 | 0.8591 ± 0.008208 | 0.8211 ± 0.007369 | 0.9557 | 0.9558 ± 0.01316 | 0.9647 (0.9648 ± 0.01172) | 1.009 (1.009 ± 0.002564) | 1.005 (1.005 ± 0.002195) |
| utd4 | 0.8687 ± 0.007469 | 0.8155 ± 0.01091 | 0.9388 | 0.9389 ± 0.01814 | 0.9451 (0.9452 ± 0.01759) | 1.007 (1.007 ± 0.004654) | 1.004 (1.004 ± 0.004584) |
| w256 | 1.014 ± 0.005622 | 1.018 ± 0.008326 | 1.004 | 1.004 ± 0.003942 | 1.003 (1.003 ± 0.008241) | 0.9992 (0.9992 ± 0.005258) | 0.9999 (1.000 ± 0.005586) |
| w512 | 0.9930 ± 0.007151 | 0.9862 ± 0.01182 | 0.9931 | 0.9931 ± 0.006006 | 0.9981 (0.9981 ± 0.007483) | 1.005 (1.005 ± 0.006243) | 1.004 (1.004 ± 0.005987) |
| b512 | 0.7684 ± 0.007286 | 0.7204 ± 0.01013 | 0.9376 | 0.9377 ± 0.01971 | 0.9480 (0.9481 ± 0.01876) | 1.011 (1.011 ± 0.008704) | 1.000 (1.000 ± 0.008062) |
| b1024 | 0.7054 ± 0.006726 | 0.6545 ± 0.003512 | 0.9279 | 0.9280 ± 0.01322 | 0.9468 (0.9469 ± 0.01468) | 1.020 (1.020 ± 0.002929) | 1.008 (1.008 ± 0.003252) |


**5.4 Energy ratio Ant/HC** (ratio of means; sd of the 5 paired per-seed ratios in parentheses):


| tag | rollout | buf_sample | critic | actor | target | grad_updates | TOTAL |
|---|---|---|---|---|---|---|---|
| base | 1.736 (±0.01307) | 1.240 (±0.03083) | 1.023 (±0.01608) | 1.023 (±0.007331) | 1.004 (±0.01837) | 1.028 (±0.01174) | 1.057 (±0.01179) |
| utd2 | 1.714 (±0.02001) | 1.257 (±0.01696) | 1.016 (±0.01293) | 1.029 (±0.003377) | 1.020 (±0.008954) | 1.029 (±0.006542) | 1.044 (±0.006636) |
| utd4 | 1.682 (±0.02150) | 1.254 (±0.01242) | 1.004 (±0.007035) | 1.021 (±0.004234) | 1.015 (±0.01426) | 1.020 (±0.004496) | 1.027 (±0.004492) |
| w256 | 1.800 (±0.02392) | 1.225 (±0.01318) | 1.017 (±0.008948) | 1.006 (±0.005755) | 1.010 (±0.008153) | 1.017 (±0.006991) | 1.054 (±0.006940) |
| w512 | 1.785 (±0.02044) | 1.242 (±0.004441) | 1.012 (±0.03729) | 1.010 (±0.007357) | 1.012 (±0.01320) | 1.017 (±0.01832) | 1.053 (±0.01735) |
| b512 | 1.650 (±0.02540) | 1.373 (±0.01511) | 1.013 (±0.03137) | 1.016 (±0.002324) | 1.015 (±0.02116) | 1.025 (±0.01143) | 1.048 (±0.01077) |
| b1024 | 1.652 (±0.02809) | 1.643 (±0.01916) | 1.045 (±0.002536) | 1.035 (±0.006485) | 1.024 (±0.009030) | 1.062 (±0.003716) | 1.081 (±0.004123) |


**J/FLOP ratio Ant/HC** (ratio of the two ratio-of-means J/FLOP values; `grad_updates` column uses the derived Table 4.5 quantity; `target` is a J/op ratio):


| tag | rollout | critic | actor | grad_updates (derived) | TOTAL | target (J/op) |
|---|---|---|---|---|---|---|
| base | 1.596 | 0.9582 | 0.9522 | 0.9602 | 0.9873 | 0.9250 |
| utd2 | 1.577 | 0.9518 | 0.9575 | 0.9613 | 0.9749 | 0.9394 |
| utd4 | 1.547 | 0.9407 | 0.9504 | 0.9527 | 0.9594 | 0.9350 |
| w256 | 1.361 | 0.8107 | 0.7860 | 0.8031 | 0.8329 | 0.7659 |
| w512 | 1.526 | 0.8943 | 0.8814 | 0.8937 | 0.9254 | 0.8667 |
| b512 | 1.517 | 0.9488 | 0.9449 | 0.9574 | 0.9785 | 0.9345 |
| b1024 | 1.520 | 0.9793 | 0.9633 | 0.9916 | 1.010 | 0.9431 |


**5.5 Exact paired sign-flip permutation test** (two-sided; statistic |mean of paired differences Ant − HC|; all 2⁵ = 32 sign patterns enumerated; p = #{patterns with |mean| ≥ observed}/32). With n = 5 the smallest attainable two-sided p is 2/32 = 0.0625, so no difference can reach p < 0.05 with this test.


| tag | quantity | per-seed differences (seed order 331,958,14577,43611,85062) | mean diff | count | p (two-sided) |
|---|---|---|---|---|---|
| base | rollout share [pp] | 2.613 / 2.656 / 2.665 / 2.626 / 2.596 | 2.631 | 2/32 | 0.06250 |
| base | TOTAL energy [J] | 3896 / 3281 / 2305 / 3890 / 3987 | 3472 | 2/32 | 0.06250 |
| utd2 | rollout share [pp] | 1.415 / 1.372 / 1.330 / 1.382 / 1.341 | 1.368 | 2/32 | 0.06250 |
| utd2 | TOTAL energy [J] | 4370 / 5679 / 4367 / 6067 / 5307 | 5158 | 2/32 | 0.06250 |
| utd4 | rollout share [pp] | 0.7113 / 0.6860 / 0.6668 / 0.6943 / 0.7048 | 0.6926 | 2/32 | 0.06250 |
| utd4 | TOTAL energy [J] | 7040 / 6673 / 6526 / 7051 / 4550 | 6368 | 2/32 | 0.06250 |
| w256 | rollout share [pp] | 3.412 / 3.397 / 3.557 / 3.310 / 3.318 | 3.399 | 2/32 | 0.06250 |
| w256 | TOTAL energy [J] | 2204 / 2609 / 2601 / 3115 / 2357 | 2577 | 2/32 | 0.06250 |
| w512 | rollout share [pp] | 3.253 / 3.219 / 3.490 / 3.310 / 3.045 | 3.263 | 2/32 | 0.06250 |
| w512 | TOTAL energy [J] | 2207 / 2899 / 1461 / 3012 / 3650 | 2646 | 2/32 | 0.06250 |
| b512 | rollout share [pp] | 2.019 / 2.141 / 2.227 / 2.047 / 1.985 | 2.084 | 2/32 | 0.06250 |
| b512 | TOTAL energy [J] | 3198 / 3113 / 2564 / 4028 / 4473 | 3475 | 2/32 | 0.06250 |
| b1024 | rollout share [pp] | 1.806 / 1.649 / 1.792 / 1.788 / 1.777 | 1.762 | 2/32 | 0.06250 |
| b1024 | TOTAL energy [J] | 6803 / 6390 / 6967 / 6346 / 6950 | 6691 | 2/32 | 0.06250 |


**5.6 Facts that differ between the environments by construction** (for interpretation; not interpreted here):
- Observation / action dims: HalfCheetah-v5 17/6, Ant-v5 105/8 (Part 2.4). Only the first layer of each network
  (input `obs` for the actor, `obs+act` for the Q-nets) and the output heads (2·act for the actor) depend on them, so
  FLOPs per call differ by the ratios below (Ant / HC), shrinking with width as the hidden×hidden layer dominates:


| signature | rollout (actor fwd bs1) | critic_fwdbwd | actor_fwdbwd | target ops |
|---|---|---|---|---|
| `bs256_h256x256` | 1.323 | 1.255 | 1.280 | 1.319 |
| `bs256_h512x512` | 1.170 | 1.132 | 1.146 | 1.167 |
| `bs256_h1024x1024` | 1.087 | 1.067 | 1.075 | 1.086 |
| `bs512_h1024x1024` | 1.087 | 1.067 | 1.075 | 1.086 |
| `bs1024_h1024x1024` | 1.087 | 1.067 | 1.075 | 1.086 |


- Episode structure inside the `rollout` task (same code for both envs; `env.reset()` is called inside the task):
  HalfCheetah-v5 never terminates (only the 1000-step truncation): completed train-phase episodes per run =
  100 in all 35 HC runs (episode length
  1000–1000 steps), i.e. 100 `env.reset()` calls inside the 100 `rollout` tasks.
  Ant-v5 has `terminate_when_unhealthy=True`: completed train-phase episodes per run
  178–511 (median 319) over the 35 Ant runs, episode length
  min/median/max 7/121/1000 steps; each completed
  episode triggers one `env.reset()` inside `rollout`. (Per-configuration episode counts: Part 9.)
- Simulator: both are MuJoCo envs stepped on the CPU (`frame_skip` = 5 for HC, 5 for Ant; Ant has more
  bodies/contacts and computes the 78-dim contact-force part of its observation). No per-step simulator timing is
  logged, so the simulator's share of the `rollout` task duration is **NOT MEASURED**.
- The `gradient_updates` code path is identical across envs; only the per-call FLOPs (table above) and the stored
  `done` flags differ (Ant terminations enter the Bellman target via `(1 − done)`; HC's `done` is always 0 since
  `buffer.add` stores `terminated`, not `truncated`).


## Part 6 — Effect of each sweep (baseline-relative)

Relative change vs the same env's `base`, in %: **paired** = per-seed (config_seed / base_seed − 1), then mean ± sd over the 5 seeds; **means** = (mean_config / mean_base − 1). Gross energy. Share changes are in percentage points (paired). Full precision in `sac_sweep_effects.csv`.


**6.1 HalfCheetah-v5 — relative change [%]** (cells: `paired mean ± sd [from means]`):


| config | E rollout | E buf_sample | E critic | E actor | E target | E grad_updates | E TOTAL | T rollout |
|---|---|---|---|---|---|---|---|---|
| HC-utd2 | 0.7203 ± 1.562 [0.7086] | 96.63 ± 3.955 [96.59] | 97.30 ± 1.103 [97.30] | 98.21 ± 1.111 [98.21] | 97.29 ± 2.663 [97.27] | 97.74 ± 0.8071 [97.74] | 93.77 ± 0.7924 [93.77] | -0.3922 ± 2.244 [-0.4073] |
| HC-utd4 | 2.309 ± 2.522 [2.291] | 291.8 ± 6.367 [291.7] | 298.5 ± 4.239 [298.4] | 297.9 ± 2.016 [297.9] | 295.8 ± 5.629 [295.8] | 297.9 ± 2.170 [297.9] | 285.7 ± 2.036 [285.7] | 0.09803 ± 2.560 [0.07936] |
| HC-w256 | -8.698 ± 0.8947 [-8.706] | -23.26 ± 1.466 [-23.27] | -22.79 ± 1.430 [-22.79] | -22.70 ± 0.8328 [-22.70] | -22.41 ± 1.137 [-22.42] | -22.74 ± 1.053 [-22.74] | -22.17 ± 1.028 [-22.17] | -3.785 ± 0.8169 [-3.794] |
| HC-w512 | -6.761 ± 0.8742 [-6.770] | -19.73 ± 1.026 [-19.75] | -19.39 ± 1.545 [-19.39] | -18.97 ± 0.5109 [-18.97] | -18.60 ± 0.7337 [-18.60] | -19.15 ± 0.8984 [-19.15] | -18.64 ± 0.8860 [-18.65] | -3.746 ± 1.398 [-3.759] |
| HC-b512 | 6.065 ± 0.9991 [6.062] | 35.69 ± 1.630 [35.67] | 21.49 ± 1.735 [21.50] | 19.24 ± 0.3621 [19.24] | 15.75 ± 2.315 [15.74] | 20.48 ± 0.5190 [20.48] | 19.89 ± 0.4798 [19.89] | -0.2637 ± 0.9579 [-0.2572] |
| HC-b1024 | 10.13 ± 1.775 [10.12] | 90.95 ± 2.183 [90.93] | 35.51 ± 1.701 [35.51] | 34.60 ± 1.003 [34.60] | 30.40 ± 1.588 [30.39] | 36.30 ± 1.245 [36.30] | 35.23 ± 1.270 [35.22] | 0.03908 ± 2.555 [0.01802] |



| config | T grad_updates | T TOTAL | P rollout | P grad_updates | P TOTAL | J/FLOP TOTAL | J/FLOP GU (derived) |
|---|---|---|---|---|---|---|---|
| HC-utd2 | 98.87 ± 0.4253 [98.87] | 94.09 ± 0.5096 [94.09] | 1.133 ± 1.205 [1.130] | -0.5685 ± 0.3209 [-0.5686] | -0.1682 ± 0.2952 [-0.1682] | -3.105 ± 0.3962 [-3.105] | -1.129 ± 0.4035 [-1.130] |
| HC-utd4 | 300.3 ± 1.976 [300.3] | 285.8 ± 1.988 [285.8] | 2.217 ± 1.248 [2.213] | -0.6101 ± 0.3368 [-0.6099] | -0.02554 ± 0.3489 [-0.02529] | -3.546 ± 0.5090 [-3.547] | -0.5361 ± 0.5426 [-0.5371] |
| HC-w256 | -2.313 ± 0.9389 [-2.316] | -2.384 ± 0.9188 [-2.387] | -5.106 ± 0.6850 [-5.107] | -20.91 ± 0.3460 [-20.91] | -20.27 ± 0.3429 [-20.27] | 1074 ± 15.50 [1074] | 1065 ± 15.87 [1065] |
| HC-w512 | -1.881 ± 1.449 [-1.883] | -1.971 ± 1.419 [-1.973] | -3.124 ± 0.9279 [-3.126] | -17.60 ± 0.3683 [-17.60] | -17.00 ± 0.3505 [-17.00] | 218.9 ± 3.473 [218.9] | 216.9 ± 3.522 [216.9] |
| HC-b512 | 3.051 ± 1.013 [3.055] | 2.892 ± 0.9349 [2.895] | 6.350 ± 1.179 [6.349] | 16.91 ± 0.6906 [16.91] | 16.52 ± 0.6621 [16.52] | -40.05 ± 0.2399 [-40.05] | -39.76 ± 0.2595 [-39.76] |
| HC-b1024 | 3.376 ± 1.520 [3.373] | 3.215 ± 1.563 [3.212] | 10.11 ± 1.242 [10.11] | 31.86 ± 0.8139 [31.86] | 31.02 ± 0.8158 [31.02] | -66.19 ± 0.3176 [-66.19] | -65.93 ± 0.3112 [-65.93] |


Share change vs base [pp], paired mean ± sd (HalfCheetah-v5):


| config | rollout | buf_sample | critic | actor | target | grad_updates |
|---|---|---|---|---|---|---|
| HC-utd2 | -1.968 ± 0.05252 | 0.03710 ± 0.04363 | 0.7358 ± 0.2444 | 1.119 ± 0.1395 | 0.07564 ± 0.04101 | 1.968 ± 0.05252 |
| HC-utd4 | -3.011 ± 0.06064 | 0.03943 ± 0.03347 | 1.327 ± 0.3007 | 1.536 ± 0.1975 | 0.1088 ± 0.06639 | 3.011 ± 0.06064 |
| HC-w256 | 0.7091 ± 0.05162 | -0.03597 ± 0.02163 | -0.3245 ± 0.2832 | -0.3355 ± 0.2270 | -0.01318 ± 0.03896 | -0.7091 ± 0.05162 |
| HC-w512 | 0.5986 ± 0.02973 | -0.03417 ± 0.03583 | -0.3746 ± 0.3479 | -0.1925 ± 0.2441 | 0.002703 ± 0.06166 | -0.5986 ± 0.02973 |
| HC-b512 | -0.4721 ± 0.04213 | 0.3353 ± 0.03895 | 0.5413 ± 0.4408 | -0.2603 ± 0.2759 | -0.1442 ± 0.09480 | 0.4721 ± 0.04213 |
| HC-b1024 | -0.7607 ± 0.03200 | 1.049 ± 0.03923 | 0.08493 ± 0.1782 | -0.2236 ± 0.1359 | -0.1493 ± 0.04028 | 0.7607 ± 0.03200 |


**6.1 Ant-v5 — relative change [%]** (cells: `paired mean ± sd [from means]`):


| config | E rollout | E buf_sample | E critic | E actor | E target | E grad_updates | E TOTAL | T rollout |
|---|---|---|---|---|---|---|---|---|
| Ant-utd2 | -0.5375 ± 1.576 [-0.5435] | 99.28 ± 6.074 [99.15] | 96.00 ± 2.200 [95.98] | 99.32 ± 1.776 [99.31] | 100.4 ± 3.572 [100.3] | 97.97 ± 1.599 [97.96] | 91.34 ± 1.550 [91.33] | 0.6856 ± 1.316 [0.6821] |
| Ant-utd4 | -0.8833 ± 1.466 [-0.8906] | 296.3 ± 11.27 [296.1] | 291.2 ± 5.290 [291.1] | 297.2 ± 4.300 [297.1] | 300.1 ± 6.344 [300.1] | 294.8 ± 4.646 [294.7] | 274.9 ± 4.403 [274.8] | 1.338 ± 0.8274 [1.334] |
| Ant-w256 | -5.323 ± 2.051 [-5.341] | -24.18 ± 2.037 [-24.22] | -23.19 ± 1.611 [-23.19] | -24.01 ± 1.214 [-24.02] | -21.95 ± 0.8446 [-21.95] | -23.59 ± 1.318 [-23.59] | -22.36 ± 1.320 [-22.36] | -1.801 ± 1.294 [-1.808] |
| Ant-w512 | -4.091 ± 1.853 [-4.107] | -19.63 ± 2.051 [-19.67] | -20.20 ± 1.321 [-20.20] | -20.00 ± 0.4686 [-20.01] | -17.99 ± 1.550 [-18.00] | -19.99 ± 0.3201 [-19.99] | -18.92 ± 0.3353 [-18.92] | -2.044 ± 0.9717 [-2.050] |
| Ant-b512 | 0.8202 ± 1.507 [0.8104] | 50.24 ± 4.242 [50.15] | 20.32 ± 1.926 [20.30] | 18.34 ± 0.8820 [18.34] | 16.95 ± 2.217 [16.92] | 20.12 ± 0.9038 [20.12] | 18.82 ± 0.9332 [18.82] | -1.235 ± 0.7409 [-1.240] |
| Ant-b1024 | 4.832 ± 1.602 [4.821] | 153.0 ± 4.506 [152.9] | 38.51 ± 2.783 [38.49] | 36.18 ± 1.471 [36.17] | 32.95 ± 1.427 [32.94] | 40.76 ± 1.916 [40.74] | 38.34 ± 1.859 [38.33] | -0.6868 ± 1.366 [-0.6940] |



| config | T grad_updates | T TOTAL | P rollout | P grad_updates | P TOTAL | J/FLOP TOTAL | J/FLOP GU (derived) |
|---|---|---|---|---|---|---|---|
| Ant-utd2 | 98.71 ± 1.882 [98.70] | 90.85 ± 1.718 [90.85] | -1.216 ± 0.7479 [-1.216] | -0.3717 ± 0.6538 [-0.3739] | 0.2559 ± 0.6249 [0.2539] | -4.319 ± 0.7749 [-4.324] | -1.017 ± 0.7994 [-1.022] |
| Ant-utd4 | 297.5 ± 3.541 [297.4] | 273.7 ± 3.215 [273.7] | -2.191 ± 1.325 [-2.193] | -0.6804 ± 0.5436 [-0.6825] | 0.3033 ± 0.5436 [0.3012] | -6.267 ± 1.101 [-6.274] | -1.311 ± 1.162 [-1.319] |
| Ant-w256 | -2.584 ± 1.133 [-2.584] | -2.522 ± 1.085 [-2.522] | -3.590 ± 1.375 [-3.596] | -21.56 ± 0.5989 [-21.56] | -20.35 ± 0.6202 [-20.35] | 890.0 ± 16.83 [889.9] | 874.3 ± 16.80 [874.3] |
| Ant-w512 | -2.669 ± 0.9015 [-2.669] | -2.619 ± 0.7912 [-2.620] | -2.094 ± 1.210 [-2.099] | -17.79 ± 0.5102 [-17.80] | -16.74 ± 0.4671 [-16.74] | 198.9 ± 1.236 [198.9] | 195.0 ± 1.180 [195.0] |
| Ant-b512 | 2.376 ± 1.151 [2.372] | 2.086 ± 1.024 [2.083] | 2.085 ± 1.624 [2.079] | 17.34 ± 1.024 [17.34] | 16.40 ± 0.9534 [16.40] | -40.58 ± 0.4667 [-40.58] | -39.94 ± 0.4519 [-39.94] |
| Ant-b1024 | 5.395 ± 1.674 [5.388] | 4.908 ± 1.627 [4.900] | 5.557 ± 0.7191 [5.558] | 33.55 ± 0.5183 [33.55] | 31.87 ± 0.4973 [31.87] | -65.41 ± 0.4648 [-65.41] | -64.81 ± 0.4789 [-64.81] |


Share change vs base [pp], paired mean ± sd (Ant-v5):


| config | rollout | buf_sample | critic | actor | target | grad_updates |
|---|---|---|---|---|---|---|
| Ant-utd2 | -3.231 ± 0.05969 | 0.1223 ± 0.07903 | 0.9494 ± 0.3591 | 1.972 ± 0.2185 | 0.1873 ± 0.05174 | 3.231 ± 0.05969 |
| Ant-utd4 | -4.950 ± 0.04399 | 0.1695 ± 0.06399 | 1.699 ± 0.2240 | 2.814 ± 0.1348 | 0.2673 ± 0.03614 | 4.950 ± 0.04399 |
| Ant-w256 | 1.476 ± 0.1258 | -0.07103 ± 0.05794 | -0.4206 ± 0.2934 | -1.006 ± 0.1522 | 0.02108 ± 0.04751 | -1.476 ± 0.1258 |
| Ant-w512 | 1.230 ± 0.1385 | -0.02738 ± 0.08101 | -0.6176 ± 0.5467 | -0.6310 ± 0.3311 | 0.04555 ± 0.09072 | -1.230 ± 0.1385 |
| Ant-b512 | -1.020 ± 0.04809 | 0.7879 ± 0.09321 | 0.4871 ± 0.4693 | -0.1920 ± 0.3046 | -0.06320 ± 0.07087 | 1.020 ± 0.04809 |
| Ant-b1024 | -1.630 ± 0.06525 | 2.475 ± 0.04083 | 0.04476 ± 0.3282 | -0.7355 ± 0.2156 | -0.1546 ± 0.04554 | 1.630 ± 0.06525 |


**6.2 Scaling check — FLOPs ratio vs energy ratio (config / base, ratio of means)** (`target` uses op counts):


HalfCheetah-v5:


| config | rollout | critic | actor | target | grad_updates | TOTAL |
|---|---|---|---|---|---|---|
| HC-utd2 | F×1.000 / E×1.007 | F×2.000 / E×1.973 | F×2.000 / E×1.982 | F×2.000 / E×1.973 | F×2.000 / E×1.977 | F×2.000 / E×1.938 |
| HC-utd4 | F×1.000 / E×1.023 | F×4.000 / E×3.984 | F×4.000 / E×3.979 | F×4.000 / E×3.958 | F×4.000 / E×3.979 | F×3.999 / E×3.857 |
| HC-w256 | F×0.06766 / E×0.9129 | F×0.06599 / E×0.7721 | F×0.06674 / E×0.7730 | F×0.06714 / E×0.7758 | F×0.06632 / E×0.7726 | F×0.06632 / E×0.7783 |
| HC-w512 | F×0.2569 / E×0.9323 | F×0.2547 / E×0.8061 | F×0.2557 / E×0.8103 | F×0.2562 / E×0.8140 | F×0.2551 / E×0.8085 | F×0.2551 / E×0.8135 |
| HC-b512 | F×1.000 / E×1.061 | F×2.000 / E×1.215 | F×2.000 / E×1.192 | F×1.000 / E×1.157 | F×2.000 / E×1.205 | F×2.000 / E×1.199 |
| HC-b1024 | F×1.000 / E×1.101 | F×4.000 / E×1.355 | F×4.000 / E×1.346 | F×1.000 / E×1.304 | F×4.000 / E×1.363 | F×3.999 / E×1.352 |


Ant-v5:


| config | rollout | critic | actor | target | grad_updates | TOTAL |
|---|---|---|---|---|---|---|
| Ant-utd2 | F×1.000 / E×0.9946 | F×2.000 / E×1.960 | F×2.000 / E×1.993 | F×2.000 / E×2.003 | F×2.000 / E×1.980 | F×2.000 / E×1.913 |
| Ant-utd4 | F×1.000 / E×0.9911 | F×4.000 / E×3.911 | F×4.000 / E×3.971 | F×4.000 / E×4.001 | F×4.000 / E×3.947 | F×3.999 / E×3.748 |
| Ant-w256 | F×0.08231 / E×0.9466 | F×0.07760 / E×0.7681 | F×0.07948 / E×0.7598 | F×0.08158 / E×0.7805 | F×0.07843 / E×0.7641 | F×0.07843 / E×0.7764 |
| Ant-w512 | F×0.2764 / E×0.9589 | F×0.2701 / E×0.7980 | F×0.2726 / E×0.7999 | F×0.2754 / E×0.8200 | F×0.2712 / E×0.8001 | F×0.2712 / E×0.8108 |
| Ant-b512 | F×1.000 / E×1.008 | F×2.000 / E×1.203 | F×2.000 / E×1.183 | F×1.000 / E×1.169 | F×2.000 / E×1.201 | F×2.000 / E×1.188 |
| Ant-b1024 | F×1.000 / E×1.048 | F×4.000 / E×1.385 | F×4.000 / E×1.362 | F×1.000 / E×1.329 | F×4.000 / E×1.407 | F×3.999 / E×1.383 |


**6.3 ΔEnergy/ΔFLOPs (marginal J per extra FLOP).** `compute_delta_pairs()` was **imported from `flop_dashboard.py` and called unmodified** (import is side-effect free: module level defines constants and functions only, the Dash app is built inside `main()` under `if __name__ == "__main__"` — confirmed). Inputs: the pipeline's `per_run_energy_per_flop.csv` (SAC, `included_in_cross_seed_avg`, loaded with the dashboard's own `load_per_run`) with `PER_RUN_DIMENSIONS` (seed held fixed → pairs are within-seed), energy `total_energy_joules`, FLOPs `total_flops`; and the cross-seed CSV with `DIMENSIONS` (`mean_energy_joules`, `total_flops`). Rule (dashboard): pairs differ only in the X dimension, all other config dims + flop_type (+ seed) fixed; matmul/mixed_total rows only; pairs with |ΔF| < 1% of the reference dropped; pooled value = ΣΔE/ΣΔF. Reference = smallest X value (`first`) or next-smaller (`previous`). Pair lists: `sac_delta_pairs.csv`.


| sweep | reference | segment | X pairs (x←ref) | HC ΣΔE/ΣΔF [pJ/FLOP] | Ant ΣΔE/ΣΔF [pJ/FLOP] | pairs used HC / Ant | pairs dropped HC / Ant |
|---|---|---|---|---|---|---|---|
| width | first | rollout | 1024x1024←256x256, 512x512←256x256 | 1098 | 1093 | 10 / 10 | 0 / 0 |
| width | first | critic | 1024x1024←256x256, 512x512←256x256 | 11.64 | 11.23 | 10 / 10 | 0 / 0 |
| width | first | actor | 1024x1024←256x256, 512x512←256x256 | 18.22 | 18.53 | 10 / 10 | 0 / 0 |
| width | first | TOTAL | 1024x1024←256x256, 512x512←256x256 | 15.90 | 15.88 | 10 / 10 | 0 / 0 |
| width | previous | rollout | 1024x1024←512x512, 512x512←256x256 | 1081 | 1075 | 10 / 10 | 0 / 0 |
| width | previous | critic | 1024x1024←512x512, 512x512←256x256 | 12.18 | 12.03 | 10 / 10 | 0 / 0 |
| width | previous | actor | 1024x1024←512x512, 512x512←256x256 | 18.82 | 19.22 | 10 / 10 | 0 / 0 |
| width | previous | TOTAL | 1024x1024←512x512, 512x512←256x256 | 16.49 | 16.64 | 10 / 10 | 0 / 0 |
| batch size | first | rollout | — | dropped (ΔF ≈ 0) | dropped (ΔF ≈ 0) | 0 / 0 | 10 / 10 |
| batch size | first | critic | 1024←256, 512←256 | 7.115 | 7.030 | 10 / 10 | 0 / 0 |
| batch size | first | actor | 1024←256, 512←256 | 10.41 | 10.04 | 10 / 10 | 0 / 0 |
| batch size | first | TOTAL | 1024←256, 512←256 | 9.570 | 9.798 | 10 / 10 | 0 / 0 |
| batch size | previous | rollout | — | dropped (ΔF ≈ 0) | dropped (ΔF ≈ 0) | 0 / 0 | 10 / 10 |
| batch size | previous | critic | 1024←512, 512←256 | 5.908 | 6.136 | 10 / 10 | 0 / 0 |
| batch size | previous | actor | 1024←512, 512←256 | 8.922 | 8.881 | 10 / 10 | 0 / 0 |
| batch size | previous | TOTAL | 1024←512, 512←256 | 8.156 | 8.762 | 10 / 10 | 0 / 0 |
| UTD | first | rollout | — | dropped (ΔF ≈ 0) | dropped (ΔF ≈ 0) | 0 / 0 | 10 / 10 |
| UTD | first | critic | UTD 2←UTD 1, UTD 4←UTD 1 | 49.38 | 46.29 | 10 / 10 | 0 / 0 |
| UTD | first | actor | UTD 2←UTD 1, UTD 4←UTD 1 | 76.60 | 73.00 | 10 / 10 | 0 / 0 |
| UTD | first | TOTAL | UTD 2←UTD 1, UTD 4←UTD 1 | 65.91 | 62.78 | 10 / 10 | 0 / 0 |
| UTD | previous | rollout | — | dropped (ΔF ≈ 0) | dropped (ΔF ≈ 0) | 0 / 0 | 10 / 10 |
| UTD | previous | critic | UTD 2←UTD 1, UTD 4←UTD 2 | 49.65 | 46.42 | 10 / 10 | 0 / 0 |
| UTD | previous | actor | UTD 2←UTD 1, UTD 4←UTD 2 | 76.81 | 72.95 | 10 / 10 | 0 / 0 |
| UTD | previous | TOTAL | UTD 2←UTD 1, UTD 4←UTD 2 | 66.16 | 62.83 | 10 / 10 | 0 / 0 |


Pooled values are from the per-run level (5 seeds × the listed X pairs). The cross-seed level gives the same pooled values (max relative difference 3.5e-15), because FLOPs are identical across seeds so ΣΔE/ΣΔF over seeds = Δ(mean E)/Δ(mean F). Dropped pairs have |ΔF| < 1 % of the reference (rollout FLOPs do not change with UTD or batch size). The pipeline CSVs contain no `gradient_updates` row, so the dashboard rule yields no gradient_updates ΔE/ΔF; the OLS slope b in 6.4 is the closest equivalent.

**6.4 (exploratory; belongs to the open Section 5.2 analysis (a)) Fixed-versus-marginal fit `E = a + b·F`** by ordinary least squares over per-run points (each run = one point; gross energy). Width sweep: w256, w512, base (batch 256, UTD 1); batch sweep: base, b512, b1024 (width 1024, UTD 1); UTD sweep added for completeness: base, utd2, utd4. Residual SE = sqrt(SSR/(n−2)). Full precision in `sac_ols_fits.csv`.


| sweep | env | segment | n | a [J] | b [J/FLOP] | b [pJ/FLOP] | R² | residual SE [J] |
|---|---|---|---|---|---|---|---|---|
| width | HalfCheetah-v5 | grad_updates | 15 | 43890 | 1.648e-11 | 16.48 | 0.9938 | 494.6 |
| width | HalfCheetah-v5 | TOTAL | 15 | 46160 | 1.674e-11 | 16.74 | 0.9940 | 492.9 |
| width | Ant-v5 | grad_updates | 15 | 44270 | 1.67e-11 | 16.70 | 0.9896 | 686.5 |
| width | Ant-v5 | TOTAL | 15 | 48360 | 1.696e-11 | 16.96 | 0.9896 | 696.9 |
| batch | HalfCheetah-v5 | grad_updates | 15 | 53790 | 7.661e-12 | 7.661 | 0.9296 | 2477 |
| batch | HalfCheetah-v5 | TOTAL | 15 | 56240 | 7.752e-12 | 7.752 | 0.9294 | 2509 |
| batch | Ant-v5 | grad_updates | 15 | 53860 | 8.389e-12 | 8.389 | 0.9653 | 2000 |
| batch | Ant-v5 | TOTAL | 15 | 58100 | 8.466e-12 | 8.466 | 0.9663 | 1988 |
| UTD | HalfCheetah-v5 | grad_updates | 15 | -32.28 | 6.621e-11 | 66.21 | 0.9999 | 833.3 |
| UTD | HalfCheetah-v5 | TOTAL | 15 | 2429 | 6.624e-11 | 66.24 | 0.9999 | 847.3 |
| UTD | Ant-v5 | grad_updates | 15 | 970.8 | 6.286e-11 | 62.86 | 0.9999 | 750.0 |
| UTD | Ant-v5 | TOTAL | 15 | 5296 | 6.284e-11 | 62.84 | 0.9999 | 791.5 |


## Part 7 — Energy composition and idle-floor diagnostics (descriptive only)

All values in this part are **diagnostics, not subtracted from any reported value**; energy stays gross everywhere else.

**7.1 CPU / GPU / RAM split of the two measured tasks** — per-task CSV columns `cpu_energy`, `gpu_energy`, `ram_energy` [kWh → J], summed over the 100 tasks of each prefix per run; mean ± sd over 5 seeds in J, and (in parentheses) mean ± sd of the per-run percentage of the task energy. CPU = RAPL package energy, GPU = NVML, RAM = CodeCarbon's constant-power RAM model (20 W, Part 2.5). Check: cpu+gpu+ram = energy_consumed per task, max rel. err 4.07e-13.


| config | rollout CPU | rollout GPU | rollout RAM | GU CPU | GU GPU | GU RAM |
|---|---|---|---|---|---|---|
| HC-base | 629.9 ± 11.10 (25.24 ± 0.2565 %) | 1457 ± 20.44 (58.37 ± 0.2958 %) | 409.1 ± 5.370 (16.39 ± 0.06402 %) | 10270 ± 114.8 (17.58 ± 0.1808 %) | 40050 ± 188.1 (68.56 ± 0.1734 %) | 8097 ± 44.67 (13.86 ± 0.02103 %) |
| HC-utd2 | 626.7 ± 13.20 (24.93 ± 0.3439 %) | 1479 ± 16.42 (58.85 ± 0.4899 %) | 407.5 ± 7.076 (16.21 ± 0.1499 %) | 20210 ± 213.6 (17.50 ± 0.09622 %) | 79200 ± 336.8 (68.56 ± 0.09651 %) | 16100 ± 98.62 (13.94 ± 0.04653 %) |
| HC-utd4 | 626.2 ± 10.68 (24.53 ± 0.2125 %) | 1517 ± 31.45 (59.43 ± 0.2543 %) | 409.4 ± 7.808 (16.04 ± 0.1324 %) | 40610 ± 399.3 (17.47 ± 0.1119 %) | 159400 ± 618.1 (68.58 ± 0.07559 %) | 32410 ± 195.5 (13.95 ± 0.06303 %) |
| HC-w256 | 612.1 ± 5.199 (26.86 ± 0.1766 %) | 1273 ± 15.65 (55.86 ± 0.2608 %) | 393.7 ± 1.778 (17.28 ± 0.09943 %) | 9955 ± 91.24 (22.06 ± 0.1105 %) | 27270 ± 335.5 (60.41 ± 0.1518 %) | 7909 ± 50.44 (17.53 ± 0.08992 %) |
| HC-w512 | 616.5 ± 4.584 (26.49 ± 0.1045 %) | 1316 ± 6.807 (56.58 ± 0.1458 %) | 393.9 ± 2.251 (16.93 ± 0.1231 %) | 10150 ± 124.9 (21.49 ± 0.06875 %) | 29130 ± 284.0 (61.69 ± 0.1077 %) | 7944 ± 114.8 (16.82 ± 0.07233 %) |
| HC-b512 | 645.3 ± 11.27 (24.38 ± 0.1946 %) | 1594 ± 21.88 (60.21 ± 0.3174 %) | 408.1 ± 8.451 (15.42 ± 0.1652 %) | 10880 ± 158.5 (15.46 ± 0.1127 %) | 51150 ± 276.9 (72.68 ± 0.1845 %) | 8344 ± 121.0 (11.86 ± 0.08181 %) |
| HC-b1024 | 655.6 ± 12.53 (23.85 ± 0.3422 %) | 1683 ± 23.89 (61.26 ± 0.5034 %) | 409.1 ± 7.052 (14.89 ± 0.1806 %) | 11030 ± 113.8 (13.85 ± 0.05337 %) | 60220 ± 356.5 (75.64 ± 0.1057 %) | 8370 ± 113.0 (10.51 ± 0.06728 %) |
| Ant-base | 1147 ± 12.96 (26.47 ± 0.2781 %) | 2466 ± 38.87 (56.93 ± 0.3051 %) | 719.1 ± 5.681 (16.60 ± 0.1000 %) | 10700 ± 203.6 (17.82 ± 0.2578 %) | 41090 ± 381.2 (68.42 ± 0.2113 %) | 8262 ± 57.10 (13.76 ± 0.06853 %) |
| Ant-utd2 | 1149 ± 11.10 (26.67 ± 0.5101 %) | 2436 ± 60.88 (56.53 ± 0.5666 %) | 723.9 ± 8.842 (16.80 ± 0.1465 %) | 21420 ± 150.7 (18.02 ± 0.05441 %) | 81040 ± 402.0 (68.17 ± 0.06450 %) | 16420 ± 138.0 (13.81 ± 0.04966 %) |
| Ant-utd4 | 1130 ± 18.48 (26.32 ± 0.3844 %) | 2435 ± 46.63 (56.71 ± 0.4583 %) | 728.6 ± 4.844 (16.97 ± 0.2117 %) | 42350 ± 216.7 (17.87 ± 0.08127 %) | 161800 ± 757.0 (68.28 ± 0.06717 %) | 32840 ± 176.7 (13.85 ± 0.02124 %) |
| Ant-w256 | 1140 ± 12.88 (27.79 ± 0.2532 %) | 2255 ± 37.81 (54.98 ± 0.3706 %) | 706.4 ± 5.533 (17.23 ± 0.1703 %) | 10270 ± 124.2 (22.39 ± 0.1532 %) | 27560 ± 492.6 (60.07 ± 0.1935 %) | 8048 ± 113.1 (17.54 ± 0.05786 %) |
| Ant-w512 | 1144 ± 16.07 (27.55 ± 0.4075 %) | 2305 ± 35.37 (55.49 ± 0.4304 %) | 704.6 ± 2.891 (16.96 ± 0.1178 %) | 10360 ± 169.2 (21.57 ± 0.3172 %) | 29640 ± 382.9 (61.69 ± 0.3764 %) | 8041 ± 87.84 (16.74 ± 0.09245 %) |
| Ant-b512 | 1158 ± 8.394 (26.53 ± 0.2130 %) | 2498 ± 42.07 (57.20 ± 0.3859 %) | 710.4 ± 1.686 (16.27 ± 0.1949 %) | 11040 ± 140.3 (15.30 ± 0.1401 %) | 52630 ± 205.3 (72.97 ± 0.1735 %) | 8458 ± 79.15 (11.72 ± 0.04876 %) |
| Ant-b1024 | 1171 ± 7.534 (25.78 ± 0.3267 %) | 2656 ± 46.22 (58.49 ± 0.4810 %) | 714.2 ± 6.097 (15.73 ± 0.1611 %) | 11590 ± 138.9 (13.72 ± 0.1058 %) | 64220 ± 204.8 (75.98 ± 0.1575 %) | 8706 ± 94.08 (10.30 ± 0.06201 %) |


**7.2 Idle floor.** `P_head`, `P_tail` = idle-task energy / its CodeCarbon `duration` (as recorded), per run; mean ± sd over 5 seeds:


| config | P_head [W] | P_tail [W] | \|P_tail−P_head\|/P_head [%] | P_tail − P_head [W] |
|---|---|---|---|---|
| HC-base | 66.97 ± 1.212 | 69.58 ± 0.7324 | 3.933 ± 2.805 | 2.608 ± 1.818 |
| HC-utd2 | 67.50 ± 6.022 | 67.89 ± 0.5895 | 5.962 ± 4.440 | 0.3898 ± 5.754 |
| HC-utd4 | 64.68 ± 0.3396 | 68.89 ± 0.7104 | 6.510 ± 0.9562 | 4.211 ± 0.6178 |
| HC-w256 | 69.19 ± 5.311 | 73.73 ± 5.731 | 15.50 ± 5.302 | 4.540 ± 11.01 |
| HC-w512 | 70.80 ± 6.132 | 71.24 ± 6.050 | 15.31 ± 5.665 | 0.4416 ± 12.16 |
| HC-b512 | 63.99 ± 1.231 | 67.78 ± 0.3482 | 5.939 ± 1.641 | 3.785 ± 0.9985 |
| HC-b1024 | 63.21 ± 0.6924 | 70.89 ± 0.5982 | 12.16 ± 1.857 | 7.677 ± 1.086 |
| Ant-base | 66.28 ± 2.035 | 72.41 ± 3.361 | 9.400 ± 7.662 | 6.129 ± 4.769 |
| Ant-utd2 | 69.03 ± 2.487 | 71.33 ± 1.563 | 5.031 ± 2.641 | 2.298 ± 3.378 |
| Ant-utd4 | 68.44 ± 0.3979 | 71.12 ± 0.4130 | 3.917 ± 1.082 | 2.678 ± 0.7251 |
| Ant-w256 | 68.41 ± 5.461 | 73.60 ± 5.709 | 16.16 ± 6.238 | 5.187 ± 11.14 |
| Ant-w512 | 71.10 ± 5.941 | 71.45 ± 6.483 | 15.41 ± 5.851 | 0.3419 ± 12.38 |
| Ant-b512 | 64.59 ± 1.909 | 67.48 ± 0.4753 | 4.541 ± 2.838 | 2.891 ± 1.741 |
| Ant-b1024 | 62.76 ± 0.6622 | 69.81 ± 0.4675 | 11.25 ± 1.472 | 7.053 ± 0.8656 |


Over all 70 runs: |P_tail − P_head| / P_head — **median 8.370 %, maximum 23.19 %** (mean 9.359 %; tail > head in 58/70 runs). Largest: Ant-w512 s43611: 23.19 %; Ant-base s43611: 22.71 %; HC-w512 s43611: 22.44 %.

Integration-window diagnostic (derived; same method as the repo's `idle_power_analysis.py`): CodeCarbon models RAM at a constant 20 W, so `ram_energy / ram_power` is the time span over which a task's energy was integrated. For `idle_baseline_head` this span is 93.10 ± 0.1163 s vs a recorded duration of 90.00 ± 0.0007448 s; for the tail 90.00 ± 0.0002985 s vs 90.00 ± 0.000718 s. Dividing by the integrated span instead (window-corrected) gives |ΔP|/P_head median 9.534 %, max 27.39 % over the 70 runs. The as-recorded values above are the ones that follow the stated protocol (energy / duration); the corrected ones are for reference only. Per-run values: `sac_idle_floor.csv`.

**7.3 Idle-floor fraction (derived, descriptive):** `P̄_idle × duration / E` per run, with `P̄_idle` = mean of that run's as-recorded head and tail idle power, in %; mean ± sd over 5 seeds. This is a diagnostic of how large the idle floor is inside the gross value — **not** a net-energy result.


| config | rollout [%] | gradient_updates [%] | TOTAL [%] |
|---|---|---|---|
| HC-base | 56.02 ± 0.5541 | 47.32 ± 0.2745 | 47.68 ± 0.2852 |
| HC-utd2 | 54.94 ± 2.851 | 47.18 ± 2.077 | 47.35 ± 2.092 |
| HC-utd4 | 53.62 ± 0.7705 | 46.57 ± 0.3212 | 46.65 ± 0.3254 |
| HC-w256 | 61.80 ± 0.7436 | 62.63 ± 0.5885 | 62.59 ± 0.5925 |
| HC-w512 | 60.16 ± 0.6910 | 59.73 ± 0.3626 | 59.75 ± 0.3731 |
| HC-b512 | 50.84 ± 0.6239 | 39.06 ± 0.4330 | 39.49 ± 0.4292 |
| HC-b1024 | 49.97 ± 0.7124 | 35.25 ± 0.4019 | 35.74 ± 0.4049 |
| Ant-base | 57.61 ± 1.066 | 47.71 ± 0.8856 | 48.37 ± 0.8976 |
| Ant-utd2 | 59.02 ± 0.8540 | 48.46 ± 0.7321 | 48.83 ± 0.7316 |
| Ant-utd4 | 59.28 ± 0.6709 | 48.33 ± 0.1772 | 48.53 ± 0.1706 |
| Ant-w256 | 61.19 ± 0.5902 | 62.28 ± 0.3924 | 62.19 ± 0.3878 |
| Ant-w512 | 60.49 ± 0.6981 | 59.65 ± 0.5052 | 59.72 ± 0.4859 |
| Ant-b512 | 53.76 ± 1.406 | 38.72 ± 0.5389 | 39.58 ± 0.5782 |
| Ant-b1024 | 52.17 ± 0.5660 | 34.14 ± 0.2282 | 35.06 ± 0.2339 |


## Part 8 — Measurement-quality diagnostics for SAC

**8.1 Single-task durations** over all 100 epochs × 5 seeds (500 tasks per cell), min / median / max in s, and the shortest task in units of the polling interval `measure_power_secs` = 1.0 s. Last two columns (derived diagnostic): integration span / duration = (`ram_energy`/`ram_power`) / `duration` per task (1 = CodeCarbon integrated energy over exactly the task's duration).


| config | rollout_i dur min/med/max [s] | shortest rollout / poll | gradient_updates_i dur min/med/max [s] | shortest GU / poll | rollout span/dur min/med/max | GU span/dur min/med/max |
|---|---|---|---|---|---|---|
| HC-base | 0.1947 / 0.2026 / 0.3698 | 0.1947 | 3.878 / 3.995 / 4.500 | 3.878 | 0.9896 / 0.9990 / 1.007 | 0.9995 / 0.9999 / 1.000 |
| HC-utd2 | 0.1951 / 0.2028 / 0.3492 | 0.1951 | 7.761 / 7.950 / 8.786 | 7.761 | 0.9902 / 0.9991 / 1.008 | 0.9997 / 0.9999 / 1.000 |
| HC-utd4 | 0.1948 / 0.2033 / 0.3467 | 0.1948 | 15.61 / 15.97 / 17.40 | 15.61 | 0.9899 / 0.9991 / 1.008 | 0.9999 / 1.000 / 1.000 |
| HC-w256 | 0.1896 / 0.1943 / 0.3564 | 0.1896 | 3.793 / 3.871 / 4.418 | 3.793 | 0.9905 / 0.9992 / 1.008 | 0.9995 / 0.9999 / 1.000 |
| HC-w512 | 0.1900 / 0.1945 / 0.3224 | 0.1900 | 3.811 / 3.887 / 4.465 | 3.811 | 0.9902 / 0.9992 / 1.008 | 0.9995 / 0.9999 / 1.000 |
| HC-b512 | 0.1949 / 0.2011 / 0.3617 | 0.1949 | 4.001 / 4.092 / 4.605 | 4.001 | 0.9908 / 0.9991 / 1.008 | 0.9995 / 0.9999 / 1.000 |
| HC-b1024 | 0.1948 / 0.2027 / 0.3479 | 0.1948 | 4.038 / 4.113 / 4.702 | 4.038 | 0.9903 / 0.9990 / 1.008 | 0.9995 / 0.9999 / 1.000 |
| Ant-base | 0.3434 / 0.3567 / 0.5322 | 0.3434 | 3.979 / 4.070 / 4.519 | 3.979 | 0.9942 / 0.9991 / 1.005 | 0.9994 / 0.9999 / 1.000 |
| Ant-utd2 | 0.3437 / 0.3605 / 0.4968 | 0.3437 | 8.001 / 8.138 / 8.995 | 8.001 | 0.9939 / 0.9988 / 1.004 | 0.9997 / 0.9999 / 1.000 |
| Ant-utd4 | 0.3470 / 0.3632 / 0.4953 | 0.3470 | 15.96 / 16.28 / 17.88 | 15.96 | 0.9939 / 0.9989 / 1.004 | 0.9999 / 1.000 / 1.000 |
| Ant-w256 | 0.3385 / 0.3508 / 0.4732 | 0.3385 | 3.820 / 3.957 / 4.531 | 3.820 | 0.9946 / 0.9995 / 1.005 | 0.9995 / 0.9999 / 1.000 |
| Ant-w512 | 0.3393 / 0.3498 / 0.4777 | 0.3393 | 3.867 / 3.946 / 4.469 | 3.867 | 0.9943 / 0.9995 / 1.005 | 0.9995 / 0.9999 / 1.000 |
| Ant-b512 | 0.3422 / 0.3528 / 0.4900 | 0.3422 | 4.053 / 4.142 / 4.705 | 4.053 | 0.9944 / 0.9994 / 1.004 | 0.9995 / 0.9999 / 1.000 |
| Ant-b1024 | 0.3418 / 0.3550 / 0.4948 | 0.3418 | 4.158 / 4.285 / 4.718 | 4.158 | 0.9943 / 0.9993 / 1.004 | 0.9994 / 0.9999 / 1.000 |


Across all 70 runs: rollout tasks 0.1896–0.5322 s (median 0.3409 s); 7000 of 7000 rollout tasks are shorter than one polling interval. gradient_updates tasks 3.793–17.88 s.

**8.2 Allocation coverage** `Σ_k T_k / duration(gradient_updates)` (Σ of the four `perf_counter` sub-phase times, from `segment_energy.json["_sub_segment_wall_time_seconds"]`, divided by the summed CodeCarbon durations of the 100 `gradient_updates_i` tasks), in %, mean ± sd / min / max over 5 seeds; and the time shares `T_k / Σ_j T_j` [%] used by the allocation. **PASS** — time shares equal the allocated-energy shares within gradient_updates (Table 4.2 right block) (max |Δ| = 1.42e-14 pp)


| config | coverage [%] mean ± sd | min | max | T buf_sample | T critic | T actor | T target |
|---|---|---|---|---|---|---|---|
| HC-base | 97.26 ± 0.08761 | 97.15 | 97.36 | 2.654 ± 0.03101 | 42.07 ± 0.2901 | 50.92 ± 0.2443 | 4.360 ± 0.04565 |
| HC-utd2 | 97.32 ± 0.05276 | 97.24 | 97.38 | 2.638 ± 0.01947 | 41.98 ± 0.1667 | 51.04 ± 0.1483 | 4.349 ± 0.02323 |
| HC-utd4 | 97.34 ± 0.04420 | 97.27 | 97.38 | 2.613 ± 0.01488 | 42.13 ± 0.1218 | 50.92 ± 0.09096 | 4.337 ± 0.04628 |
| HC-w256 | 97.66 ± 0.03134 | 97.62 | 97.69 | 2.636 ± 0.01672 | 42.04 ± 0.2100 | 50.94 ± 0.1773 | 4.378 ± 0.03143 |
| HC-w512 | 97.45 ± 0.08468 | 97.35 | 97.55 | 2.635 ± 0.02990 | 41.94 ± 0.4939 | 51.04 ± 0.3969 | 4.390 ± 0.06943 |
| HC-b512 | 97.47 ± 0.07009 | 97.37 | 97.54 | 2.989 ± 0.04250 | 42.42 ± 0.6175 | 50.40 ± 0.4845 | 4.189 ± 0.09298 |
| HC-b1024 | 97.43 ± 0.05372 | 97.33 | 97.46 | 3.718 ± 0.05882 | 41.83 ± 0.3541 | 50.29 ± 0.2653 | 4.171 ± 0.04752 |
| Ant-base | 96.54 ± 0.1465 | 96.30 | 96.69 | 3.202 ± 0.07023 | 41.85 ± 0.2445 | 50.69 ± 0.1699 | 4.259 ± 0.03889 |
| Ant-utd2 | 96.61 ± 0.05368 | 96.53 | 96.67 | 3.222 ± 0.02568 | 41.44 ± 0.1792 | 51.03 ± 0.1467 | 4.310 ± 0.02695 |
| Ant-utd4 | 96.66 ± 0.04939 | 96.58 | 96.72 | 3.213 ± 0.01156 | 41.47 ± 0.05059 | 51.00 ± 0.04316 | 4.316 ± 0.01661 |
| Ant-w256 | 97.70 ± 0.03782 | 97.67 | 97.76 | 3.176 ± 0.01952 | 42.07 ± 0.2729 | 50.41 ± 0.2305 | 4.350 ± 0.03676 |
| Ant-w512 | 96.81 ± 0.04506 | 96.76 | 96.86 | 3.215 ± 0.02506 | 41.74 ± 0.4017 | 50.68 ± 0.3271 | 4.365 ± 0.06336 |
| Ant-b512 | 96.69 ± 0.03385 | 96.65 | 96.73 | 4.003 ± 0.03322 | 41.92 ± 0.2876 | 49.93 ± 0.2101 | 4.146 ± 0.04872 |
| Ant-b1024 | 96.54 ± 0.06988 | 96.46 | 96.63 | 5.755 ± 0.07334 | 41.18 ± 0.3464 | 49.04 ± 0.2509 | 4.023 ± 0.04629 |


**8.3 Per-epoch stationarity** (per-epoch values in `sac_per_epoch.csv`). For each run and task: mean and sd across the 100 epochs, CV across epochs, relative difference (mean of epochs 0–9 − mean of epochs 10–99) / mean of 10–99, and the OLS slope vs epoch index (also as % of the run's mean per 100 epochs); each statistic is then averaged over the 5 seeds (± sd over seeds).


Per-epoch **energy** (J):


| config | rollout mean ± sd-across-epochs [J] | rollout CV % | rollout ep0–9 vs 10–99 % | rollout slope [J/epoch] | rollout slope % of mean /100 ep | GU mean ± sd-across-epochs [J] | GU CV % | GU ep0–9 vs 10–99 % | GU slope [J/epoch] | GU slope % of mean /100 ep |
|---|---|---|---|---|---|---|---|---|---|---|
| HC-base | 24.96 ± 2.363 | 9.454 ± 1.957 | 3.319 ± 4.932 | -0.01238 ± 0.01013 | -4.947 ± 4.027 | 584.2 ± 14.31 | 2.449 ± 0.3120 | -0.7267 ± 1.894 | -0.01304 ± 0.05495 | -0.2223 ± 0.9410 |
| HC-utd2 | 25.13 ± 2.275 | 9.053 ± 1.368 | 5.315 ± 2.055 | -0.01172 ± 0.004359 | -4.676 ± 1.769 | 1155 ± 23.31 | 2.017 ± 0.2673 | 1.629 ± 0.5591 | -0.1942 ± 0.2161 | -1.680 ± 1.870 |
| HC-utd4 | 25.53 ± 2.325 | 9.088 ± 1.778 | 5.069 ± 3.401 | -0.008624 ± 0.007038 | -3.393 ± 2.806 | 2324 ± 50.34 | 2.165 ± 0.1910 | 1.851 ± 1.559 | -0.4510 ± 0.2478 | -1.938 ± 1.058 |
| HC-w256 | 22.78 ± 2.209 | 9.697 ± 1.583 | 6.788 ± 2.532 | -0.01039 ± 0.007859 | -4.547 ± 3.435 | 451.3 ± 16.24 | 3.597 ± 0.6813 | -0.004636 ± 4.764 | 0.09200 ± 0.3277 | 2.062 ± 7.268 |
| HC-w512 | 23.27 ± 1.992 | 8.560 ± 0.5078 | 7.480 ± 2.225 | -0.01697 ± 0.008653 | -7.289 ± 3.718 | 472.3 ± 15.70 | 3.323 ± 0.4127 | 2.367 ± 2.030 | 0.001284 ± 0.2576 | 0.03665 ± 5.461 |
| HC-b512 | 26.47 ± 2.776 | 10.48 ± 1.351 | 4.283 ± 3.363 | -0.004453 ± 0.01138 | -1.654 ± 4.321 | 703.8 ± 12.06 | 1.713 ± 0.3158 | 1.872 ± 1.212 | -0.1602 ± 0.07855 | -2.271 ± 1.102 |
| HC-b1024 | 27.48 ± 3.089 | 11.24 ± 0.4592 | 1.545 ± 4.416 | -0.001457 ± 0.004417 | -0.5337 ± 1.614 | 796.2 ± 13.55 | 1.701 ± 0.2558 | 0.6413 ± 1.109 | 0.05491 ± 0.1928 | 0.7009 ± 2.423 |
| Ant-base | 43.32 ± 4.480 | 10.34 ± 0.5843 | -0.9047 ± 1.431 | -0.006141 ± 0.01478 | -1.418 ± 3.406 | 600.5 ± 15.64 | 2.604 ± 0.09714 | -0.3675 ± 2.436 | 0.002994 ± 0.1702 | 0.03759 ± 2.844 |
| Ant-utd2 | 43.09 ± 4.292 | 9.967 ± 0.6757 | 5.107 ± 1.822 | -0.02464 ± 0.01511 | -5.726 ± 3.510 | 1189 ± 19.28 | 1.621 ± 0.2610 | 0.8243 ± 1.340 | -0.2088 ± 0.07102 | -1.756 ± 0.5956 |
| Ant-utd4 | 42.94 ± 4.265 | 9.935 ± 0.4440 | 6.217 ± 3.184 | -0.01864 ± 0.01347 | -4.319 ± 3.073 | 2370 ± 38.64 | 1.630 ± 0.1408 | 1.202 ± 1.523 | -0.3037 ± 0.4118 | -1.277 ± 1.732 |
| Ant-w256 | 41.01 ± 3.635 | 8.866 ± 0.4807 | 1.259 ± 5.655 | 0.002305 ± 0.03032 | 0.6036 ± 7.411 | 458.8 ± 15.14 | 3.297 ± 0.4981 | 1.577 ± 2.327 | -0.03940 ± 0.1555 | -0.8358 ± 3.383 |
| Ant-w512 | 41.54 ± 3.888 | 9.360 ± 0.3058 | 1.764 ± 5.584 | -0.01470 ± 0.03647 | -3.566 ± 8.767 | 480.5 ± 13.67 | 2.848 ± 0.6529 | 2.325 ± 4.399 | -0.1878 ± 0.2174 | -3.922 ± 4.568 |
| Ant-b512 | 43.67 ± 4.935 | 11.30 ± 0.3723 | -0.3495 ± 2.314 | 0.006672 ± 0.007634 | 1.523 ± 1.741 | 721.3 ± 14.71 | 2.040 ± 0.1114 | 1.441 ± 1.547 | -0.09239 ± 0.2170 | -1.276 ± 3.001 |
| Ant-b1024 | 45.41 ± 5.598 | 12.33 ± 0.4460 | 2.690 ± 3.823 | -0.001630 ± 0.01124 | -0.3618 ± 2.471 | 845.2 ± 13.85 | 1.637 ± 0.4920 | -0.4386 ± 1.332 | 0.1585 ± 0.2524 | 1.874 ± 2.978 |


Per-epoch **duration** (s):


| config | rollout mean ± sd-across-epochs [s] | rollout CV % | rollout ep0–9 vs 10–99 % | rollout slope [s/epoch] | rollout slope % of mean /100 ep | GU mean ± sd-across-epochs [s] | GU CV % | GU ep0–9 vs 10–99 % | GU slope [s/epoch] | GU slope % of mean /100 ep |
|---|---|---|---|---|---|---|---|---|---|---|
| HC-base | 0.2048 ± 0.01486 | 7.260 ± 0.8097 | 8.512 ± 1.021 | -0.0001204 ± 2.515e-05 | -5.886 ± 1.279 | 4.049 ± 0.1251 | 3.089 ± 0.3357 | 0.7361 ± 2.341 | 0.0003462 ± 0.0004402 | 0.8576 ± 1.092 |
| HC-utd2 | 0.2039 ± 0.01457 | 7.147 ± 0.09857 | 8.350 ± 0.8697 | -0.0001248 ± 2.121e-05 | -6.122 ± 1.045 | 8.052 ± 0.2352 | 2.920 ± 0.3652 | 1.857 ± 1.344 | 0.0001608 ± 0.001799 | 0.1978 ± 2.236 |
| HC-utd4 | 0.2049 ± 0.01460 | 7.129 ± 0.1903 | 7.916 ± 0.2928 | -0.000107 ± 2.021e-05 | -5.209 ± 0.8918 | 16.21 ± 0.4947 | 3.052 ± 0.08001 | 1.343 ± 1.840 | -0.001071 ± 0.002667 | -0.6559 ± 1.643 |
| HC-w256 | 0.1970 ± 0.01334 | 6.774 ± 0.9393 | 7.838 ± 1.232 | -0.0001021 ± 2.265e-05 | -5.184 ± 1.164 | 3.955 ± 0.1425 | 3.603 ± 0.2247 | 1.275 ± 4.632 | 0.0005564 ± 0.001893 | 1.398 ± 4.793 |
| HC-w512 | 0.1971 ± 0.01309 | 6.643 ± 0.06102 | 7.906 ± 0.7592 | -0.0001065 ± 4.317e-06 | -5.404 ± 0.2139 | 3.972 ± 0.1393 | 3.502 ± 0.4983 | 3.082 ± 2.807 | 0.0006789 ± 0.001046 | 1.688 ± 2.628 |
| HC-b512 | 0.2042 ± 0.01477 | 7.231 ± 0.1535 | 7.879 ± 0.7069 | -0.0001056 ± 9.878e-06 | -5.169 ± 0.4423 | 4.172 ± 0.1223 | 2.929 ± 0.3887 | 4.450 ± 2.455 | -0.002213 ± 0.0007886 | -5.288 ± 1.826 |
| HC-b1024 | 0.2048 ± 0.01443 | 7.049 ± 0.1724 | 8.250 ± 0.4747 | -0.000122 ± 1.668e-05 | -5.961 ± 0.8320 | 4.185 ± 0.1335 | 3.183 ± 0.6080 | 3.851 ± 1.973 | -0.001103 ± 0.001933 | -2.598 ± 4.582 |
| Ant-base | 0.3599 ± 0.01581 | 4.392 ± 0.5785 | 4.457 ± 0.8932 | -6.167e-05 ± 4.583e-05 | -1.718 ± 1.272 | 4.131 ± 0.1258 | 3.044 ± 0.2647 | 1.190 ± 1.505 | 3.925e-06 ± 0.001099 | 0.002666 ± 2.661 |
| Ant-utd2 | 0.3624 ± 0.01510 | 4.168 ± 0.09854 | 3.312 ± 0.8147 | -6.396e-05 ± 1.522e-05 | -1.764 ± 0.4143 | 8.209 ± 0.1803 | 2.195 ± 0.4919 | -0.09406 ± 1.280 | 0.0004741 ± 0.0003672 | 0.5775 ± 0.4500 |
| Ant-utd4 | 0.3647 ± 0.01502 | 4.117 ± 0.06448 | 2.935 ± 0.7575 | -4.494e-05 ± 2.718e-05 | -1.231 ± 0.7403 | 16.42 ± 0.3911 | 2.382 ± 0.2171 | 0.2433 ± 1.715 | -0.0001465 ± 0.002930 | -0.08318 ± 1.783 |
| Ant-w256 | 0.3534 ± 0.01337 | 3.782 ± 0.09354 | 3.730 ± 0.8979 | -4.149e-05 ± 3.942e-05 | -1.172 ± 1.108 | 4.024 ± 0.1467 | 3.645 ± 0.1447 | 3.642 ± 3.542 | -0.0009109 ± 0.001587 | -2.306 ± 3.947 |
| Ant-w512 | 0.3525 ± 0.01364 | 3.869 ± 0.1849 | 4.227 ± 0.4676 | -8.331e-05 ± 2.531e-05 | -2.364 ± 0.7219 | 4.021 ± 0.1293 | 3.215 ± 0.2613 | 3.186 ± 2.764 | -0.001174 ± 0.001926 | -2.918 ± 4.740 |
| Ant-b512 | 0.3554 ± 0.01489 | 4.188 ± 0.03324 | 3.870 ± 0.8957 | -7.44e-05 ± 3.256e-05 | -2.092 ± 0.9135 | 4.229 ± 0.1453 | 3.435 ± 0.2437 | 2.488 ± 2.854 | -0.0008424 ± 0.002592 | -1.970 ± 6.098 |
| Ant-b1024 | 0.3574 ± 0.01491 | 4.171 ± 0.06099 | 3.989 ± 0.4624 | -7.131e-05 ± 1.244e-05 | -1.996 ± 0.3579 | 4.354 ± 0.1319 | 3.026 ± 0.6272 | 1.664 ± 3.285 | 0.0001129 ± 0.002958 | 0.2559 ± 6.764 |


Epoch 0 vs mean of epochs 1–99 (energy ratio), all 70 runs: rollout 1.402 ± 0.1666, gradient_updates 1.049 ± 0.02748.

**8.4 Execution order and thermal state.** The 70 runs sorted by `metadata.start_time_utc` (directory timestamp = local start time). Session = maximal block of runs from **all 300** recorded runs (all algorithms) with < 30 min between one run's end and the next run's start. Gap = minutes since the previous *SAC* run ended. Thermal fields from `metadata.json["thermal_gate"]` (`final_*` = reading at which the gate passed; `run_start_*` = NVML reading after the 30 s settle, just before the tracker starts; `run_end_*` = after the idle tail; blank = key absent, see Part 0).


| # | dir timestamp | config | seed | session | gap [min] | run length [min] | gate waited [s] | gate temp [°C] | gate power [W] | gated | reason | start temp | start power [W] | end temp | end power [W] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20260907_121335 | HC-base | 331 | 9 | NaN | 10.88 | 1.168e-05 | 45.00 | 26.94 | True | reached_reference | NaN | NaN | NaN | NaN |
| 2 | 20260907_123149 | HC-base | 958 | 9 | 7.356 | 10.88 | 1.097e-05 | 45.00 | 26.48 | True | reached_reference | NaN | NaN | NaN | NaN |
| 3 | 20260907_124434 | HC-base | 14577 | 9 | 1.859 | 10.88 | 1.144e-05 | 45.00 | 27.84 | True | reached_reference | NaN | NaN | NaN | NaN |
| 4 | 20260907_125849 | HC-base | 43611 | 9 | 3.379 | 10.95 | 1.001e-05 | 46.00 | 28.17 | True | reached_reference | NaN | NaN | NaN | NaN |
| 5 | 20260907_131709 | HC-base | 85062 | 9 | 7.383 | 10.85 | 1.049e-05 | 47.00 | 28.16 | True | reached_reference | 47.00 | 28.99 | 45.00 | 27.90 |
| 6 | 20260907_132907 | Ant-base | 331 | 9 | 1.108 | 11.71 | 1.097e-05 | 46.00 | 26.81 | True | reached_reference | 46.00 | 26.64 | 43.00 | 27.26 |
| 7 | 20260907_134651 | Ant-base | 958 | 9 | 6.026 | 11.26 | 1.025e-05 | 45.00 | 28.42 | True | reached_reference | 45.00 | 27.00 | 44.00 | 26.90 |
| 8 | 20260907_135857 | Ant-base | 14577 | 9 | 0.8347 | 11.17 | 1.121e-05 | 45.00 | 27.66 | True | reached_reference | 45.00 | 27.42 | 45.00 | 28.81 |
| 9 | 20260908_115230 | Ant-base | 43611 | 10 | 1302 | 11.37 | 1.144e-05 | 42.00 | 24.62 | True | reached_reference | 42.00 | 25.09 | 45.00 | 30.24 |
| 10 | 20260908_120632 | Ant-base | 85062 | 10 | 2.649 | 11.26 | 1.073e-05 | 45.00 | 27.64 | True | reached_reference | 46.00 | 29.05 | 45.00 | 27.96 |
| 11 | 20260908_122254 | HC-utd2 | 331 | 10 | 5.118 | 17.71 | 9.537e-06 | 47.00 | 44.71 | True | reached_reference | 47.00 | 44.66 | 44.00 | 26.60 |
| 12 | 20260908_124038 | HC-utd2 | 958 | 10 | 0.02199 | 17.80 | 1.192e-05 | 45.00 | 27.54 | True | reached_reference | 45.00 | 27.51 | 45.00 | 26.64 |
| 13 | 20260908_125827 | HC-utd2 | 14577 | 10 | 0.02183 | 17.54 | 1.001e-05 | 45.00 | 25.55 | True | reached_reference | 45.00 | 25.07 | 45.00 | 28.16 |
| 14 | 20260908_131601 | HC-utd2 | 43611 | 10 | 0.02201 | 17.48 | 1.144e-05 | 45.00 | 27.32 | True | reached_reference | 45.00 | 27.07 | 45.00 | 27.26 |
| 15 | 20260908_133331 | HC-utd2 | 85062 | 10 | 0.02241 | 17.50 | 1.001e-05 | 45.00 | 27.64 | True | reached_reference | 45.00 | 27.83 | 45.00 | 27.43 |
| 16 | 20260908_135102 | HC-utd4 | 331 | 10 | 0.02237 | 31.00 | 1.097e-05 | 45.00 | 28.24 | True | reached_reference | 45.00 | 27.72 | 45.00 | 28.70 |
| 17 | 20260908_142203 | HC-utd4 | 958 | 10 | 0.02233 | 31.39 | 9.537e-06 | 45.00 | 27.09 | True | reached_reference | 45.00 | 27.62 | 45.00 | 28.29 |
| 18 | 20260908_145328 | HC-utd4 | 14577 | 10 | 0.02208 | 31.14 | 9.06e-06 | 45.00 | 26.69 | True | reached_reference | 45.00 | 26.94 | 45.00 | 27.37 |
| 19 | 20260908_152438 | HC-utd4 | 43611 | 10 | 0.02210 | 31.01 | 1.025e-05 | 45.00 | 26.48 | True | reached_reference | 45.00 | 26.42 | 45.00 | 30.04 |
| 20 | 20260908_155540 | HC-utd4 | 85062 | 10 | 0.02236 | 31.02 | 1.025e-05 | 45.00 | 28.09 | True | reached_reference | 45.00 | 28.52 | 45.00 | 29.35 |
| 21 | 20260909_134544 | Ant-utd2 | 331 | 13 | 1279 | 17.97 | 9.537e-06 | 51.00 | 33.96 | True | reached_reference | 51.00 | 32.45 | 45.00 | 27.99 |
| 22 | 20260909_140343 | Ant-utd2 | 958 | 13 | 0.02218 | 18.33 | 1.049e-05 | 45.00 | 28.65 | True | reached_reference | 45.00 | 28.00 | 45.00 | 26.99 |
| 23 | 20260909_142204 | Ant-utd2 | 14577 | 13 | 0.02198 | 18.01 | 9.775e-06 | 45.00 | 27.81 | True | reached_reference | 45.00 | 28.57 | 45.00 | 29.33 |
| 24 | 20260909_144006 | Ant-utd2 | 43611 | 13 | 0.02215 | 18.06 | 1.025e-05 | 45.00 | 27.99 | True | reached_reference | 45.00 | 28.80 | 45.00 | 28.45 |
| 25 | 20260909_145811 | Ant-utd2 | 85062 | 13 | 0.02217 | 18.07 | 1.049e-05 | 45.00 | 28.94 | True | reached_reference | 45.00 | 28.51 | 45.00 | 31.18 |
| 26 | 20260909_151617 | Ant-utd4 | 331 | 13 | 0.02240 | 31.61 | 1.144e-05 | 45.00 | 31.45 | True | reached_reference | 45.00 | 28.61 | 45.00 | 29.96 |
| 27 | 20260909_154755 | Ant-utd4 | 958 | 13 | 0.02193 | 31.98 | 1.073e-05 | 45.00 | 29.40 | True | reached_reference | 45.00 | 28.91 | 45.00 | 29.57 |
| 28 | 20260909_161955 | Ant-utd4 | 14577 | 13 | 0.02222 | 31.70 | 9.775e-06 | 45.00 | 29.83 | True | reached_reference | 45.00 | 28.67 | 45.00 | 28.45 |
| 29 | 20260909_165139 | Ant-utd4 | 43611 | 13 | 0.02199 | 31.78 | 1.001e-05 | 45.00 | 28.08 | True | reached_reference | 45.00 | 27.94 | 45.00 | 29.82 |
| 30 | 20260909_172327 | Ant-utd4 | 85062 | 13 | 0.02213 | 31.63 | 9.537e-06 | 45.00 | 28.54 | True | reached_reference | 45.00 | 29.74 | 45.00 | 30.17 |
| 31 | 20260912_191506 | HC-w256 | 331 | 15 | 4400 | 10.71 | 1.168e-05 | 45.00 | 31.04 | True | reached_reference | 45.00 | 30.42 | 54.00 | 38.22 |
| 32 | 20260912_192550 | HC-w256 | 958 | 15 | 0.02661 | 10.71 | 1.049e-05 | 54.00 | 37.75 | True | reached_reference | 54.00 | 37.66 | 43.00 | 26.67 |
| 33 | 20260912_193634 | HC-w256 | 14577 | 15 | 0.02169 | 10.63 | 9.537e-06 | 43.00 | 26.34 | True | reached_reference | 43.00 | 26.79 | 53.00 | 37.29 |
| 34 | 20260912_194713 | HC-w256 | 43611 | 15 | 0.02162 | 10.71 | 9.537e-06 | 53.00 | 37.03 | True | reached_reference | 53.00 | 37.07 | 44.00 | 28.06 |
| 35 | 20260912_195757 | HC-w256 | 85062 | 15 | 0.02227 | 10.64 | 1.097e-05 | 44.00 | 27.21 | True | reached_reference | 44.00 | 28.75 | 54.00 | 37.45 |
| 36 | 20260912_200837 | HC-w512 | 331 | 15 | 0.02189 | 10.80 | 9.775e-06 | 54.00 | 37.14 | True | reached_reference | 54.00 | 36.73 | 43.00 | 26.89 |
| 37 | 20260912_201926 | HC-w512 | 958 | 15 | 0.02200 | 10.64 | 1.001e-05 | 43.00 | 26.09 | True | reached_reference | 43.00 | 26.71 | 55.00 | 37.26 |
| 38 | 20260912_203006 | HC-w512 | 14577 | 15 | 0.02180 | 10.89 | 9.537e-06 | 55.00 | 38.72 | True | reached_reference | 54.00 | 37.44 | 43.00 | 27.72 |
| 39 | 20260912_204101 | HC-w512 | 43611 | 15 | 0.02162 | 10.72 | 1.788e-05 | 43.00 | 26.56 | True | reached_reference | 43.00 | 26.43 | 55.00 | 37.93 |
| 40 | 20260912_205145 | HC-w512 | 85062 | 15 | 0.02149 | 10.75 | 1.025e-05 | 54.00 | 37.10 | True | reached_reference | 54.00 | 36.63 | 43.00 | 26.70 |
| 41 | 20260912_210231 | Ant-w256 | 331 | 15 | 0.02220 | 11.03 | 1.049e-05 | 43.00 | 26.48 | True | reached_reference | 43.00 | 26.39 | 54.00 | 36.76 |
| 42 | 20260912_211334 | Ant-w256 | 958 | 15 | 0.02189 | 11.07 | 1.049e-05 | 54.00 | 36.76 | True | reached_reference | 53.00 | 37.05 | 43.00 | 28.01 |
| 43 | 20260912_212440 | Ant-w256 | 14577 | 15 | 0.02173 | 10.98 | 1.025e-05 | 43.00 | 27.05 | True | reached_reference | 44.00 | 27.66 | 54.00 | 37.81 |
| 44 | 20260912_213540 | Ant-w256 | 43611 | 15 | 0.02172 | 11.23 | 1.073e-05 | 54.00 | 37.96 | True | reached_reference | 54.00 | 37.97 | 43.00 | 25.87 |
| 45 | 20260912_214655 | Ant-w256 | 85062 | 15 | 0.02200 | 11.02 | 1.025e-05 | 43.00 | 26.15 | True | reached_reference | 43.00 | 25.64 | 54.00 | 37.81 |
| 46 | 20260912_215758 | Ant-w512 | 331 | 15 | 0.02177 | 11.03 | 9.537e-06 | 54.00 | 37.06 | True | reached_reference | 53.00 | 36.79 | 43.00 | 27.32 |
| 47 | 20260912_220900 | Ant-w512 | 958 | 15 | 0.02166 | 11.02 | 9.775e-06 | 43.00 | 26.55 | True | reached_reference | 43.00 | 27.36 | 55.00 | 38.46 |
| 48 | 20260912_222003 | Ant-w512 | 14577 | 15 | 0.02171 | 10.97 | 9.775e-06 | 55.00 | 38.03 | True | reached_reference | 55.00 | 37.78 | 42.00 | 28.36 |
| 49 | 20260912_223103 | Ant-w512 | 43611 | 15 | 0.02230 | 11.14 | 1.025e-05 | 42.00 | 25.43 | True | reached_reference | 43.00 | 26.05 | 55.00 | 38.21 |
| 50 | 20260912_224212 | Ant-w512 | 85062 | 15 | 0.02163 | 11.13 | 1.097e-05 | 55.00 | 37.31 | True | reached_reference | 54.00 | 38.35 | 42.00 | 25.77 |
| 51 | 20260915_123859 | HC-b512 | 331 | 17 | 3706 | 11.02 | 1.144e-05 | 47.00 | 28.17 | True | reached_reference | 47.00 | 27.59 | 43.00 | 25.24 |
| 52 | 20260915_125002 | HC-b512 | 958 | 17 | 0.02205 | 11.23 | 1.001e-05 | 43.00 | 25.63 | True | reached_reference | 43.00 | 24.61 | 43.00 | 25.54 |
| 53 | 20260915_130117 | HC-b512 | 14577 | 17 | 0.02202 | 11.17 | 9.775e-06 | 43.00 | 25.44 | True | reached_reference | 44.00 | 25.27 | 43.00 | 24.97 |
| 54 | 20260915_131229 | HC-b512 | 43611 | 17 | 0.02233 | 10.93 | 1.049e-05 | 43.00 | 24.83 | True | reached_reference | 44.00 | 25.39 | 43.00 | 26.23 |
| 55 | 20260915_132326 | HC-b512 | 85062 | 17 | 0.02218 | 11.01 | 1.001e-05 | 43.00 | 26.32 | True | reached_reference | 43.00 | 25.34 | 43.00 | 25.92 |
| 56 | 20260915_133428 | HC-b1024 | 331 | 17 | 0.02197 | 11.03 | 9.775e-06 | 43.00 | 24.86 | True | reached_reference | 43.00 | 24.41 | 44.00 | 27.30 |
| 57 | 20260915_134531 | HC-b1024 | 958 | 17 | 0.02241 | 11.15 | 9.775e-06 | 44.00 | 26.08 | True | reached_reference | 42.00 | 25.38 | 44.00 | 28.29 |
| 58 | 20260915_135641 | HC-b1024 | 14577 | 17 | 0.02235 | 11.12 | 1.001e-05 | 44.00 | 26.48 | True | reached_reference | 42.00 | 25.16 | 44.00 | 27.99 |
| 59 | 20260915_140749 | HC-b1024 | 43611 | 17 | 0.02225 | 11.16 | 9.775e-06 | 44.00 | 27.18 | True | reached_reference | 42.00 | 26.45 | 44.00 | 27.92 |
| 60 | 20260915_141900 | HC-b1024 | 85062 | 17 | 0.02198 | 10.93 | 9.537e-06 | 44.00 | 28.40 | True | reached_reference | 42.00 | 26.50 | 44.00 | 26.17 |
| 61 | 20260917_121044 | Ant-b512 | 331 | 18 | 2741 | 11.37 | 1.168e-05 | 46.00 | 28.02 | True | reached_reference | 46.00 | 30.82 | 44.00 | 27.38 |
| 62 | 20260917_122208 | Ant-b512 | 958 | 18 | 0.02160 | 11.50 | 9.537e-06 | 45.00 | 27.72 | True | reached_reference | 45.00 | 26.90 | 44.00 | 27.78 |
| 63 | 20260917_123339 | Ant-b512 | 14577 | 18 | 0.02192 | 11.36 | 1.788e-05 | 44.00 | 27.39 | True | reached_reference | 45.00 | 27.63 | 45.00 | 25.96 |
| 64 | 20260917_124502 | Ant-b512 | 43611 | 18 | 0.02164 | 11.39 | 1.216e-05 | 45.00 | 25.55 | True | reached_reference | 45.00 | 25.02 | 45.00 | 27.57 |
| 65 | 20260917_125627 | Ant-b512 | 85062 | 18 | 0.02157 | 11.52 | 1.24e-05 | 45.00 | 26.02 | True | reached_reference | 45.00 | 25.84 | 45.00 | 27.42 |
| 66 | 20260917_130759 | Ant-b1024 | 331 | 18 | 0.02248 | 11.56 | 1.168e-05 | 45.00 | 26.57 | True | reached_reference | 45.00 | 26.20 | 43.00 | 26.06 |
| 67 | 20260917_131934 | Ant-b1024 | 958 | 18 | 0.02178 | 11.66 | 9.775e-06 | 43.00 | 25.23 | True | reached_reference | 41.00 | 24.45 | 43.00 | 27.71 |
| 68 | 20260917_133115 | Ant-b1024 | 14577 | 18 | 0.02159 | 11.88 | 9.775e-06 | 42.00 | 25.87 | True | reached_reference | 42.00 | 24.43 | 43.00 | 26.47 |
| 69 | 20260917_134310 | Ant-b1024 | 43611 | 18 | 0.02206 | 11.82 | 9.06e-06 | 42.00 | 26.27 | True | reached_reference | 42.00 | 26.09 | 43.00 | 26.84 |
| 70 | 20260917_135500 | Ant-b1024 | 85062 | 18 | 0.02199 | 11.52 | 1.073e-05 | 43.00 | 25.87 | True | reached_reference | 42.00 | 24.82 | 43.00 | 28.05 |


Sessions containing SAC runs (times UTC):


| session | first start | last end | # runs (all algos) | algos | # SAC | SAC configs in order | order |
|---|---|---|---|---|---|---|---|
| 9 | 2026-09-07T10:13 | 2026-09-07T12:10 | 8 | {'sac': 8} | 8 | HC-base → Ant-base | configuration-major |
| 10 | 2026-09-08T09:52 | 2026-09-08T14:26 | 12 | {'sac': 12} | 12 | Ant-base → HC-utd2 → HC-utd4 | configuration-major |
| 13 | 2026-09-09T10:22 | 2026-09-09T20:34 | 40 | {'td3': 30, 'sac': 10} | 10 | Ant-utd2 → Ant-utd4 | configuration-major (each config's seeds contiguous, seeds in order 331,958,14577,43611,85062) |
| 15 | 2026-09-12T17:15 | 2026-09-13T10:44 | 60 | {'sac': 20, 'mbpo': 20, 'td3': 20} | 20 | HC-w256 → HC-w512 → Ant-w256 → Ant-w512 | configuration-major (each config's seeds contiguous, seeds in order 331,958,14577,43611,85062) |
| 17 | 2026-09-15T10:39 | 2026-09-15T19:32 | 35 | {'sac': 10, 'mbpo': 10, 'td3': 15} | 10 | HC-b512 → HC-b1024 | configuration-major (each config's seeds contiguous, seeds in order 331,958,14577,43611,85062) |
| 18 | 2026-09-17T10:10 | 2026-09-18T02:15 | 45 | {'sac': 10, 'mbpo': 20, 'td3': 15} | 10 | Ant-b512 → Ant-b1024 | configuration-major (each config's seeds contiguous, seeds in order 331,958,14577,43611,85062) |


Within a session, consecutive SAC runs are separated by 1.289–443.0 s (median 1.323 s) between one run's `end_time_utc` and the next run's `start_time_utc`. 'Run length' = `end_time_utc − start_time_utc`; `start_time_utc` is taken before the thermal gate, so run length includes the gate's reference poll, the 30 s settle and both 90 s idle windows. Each sweep script powers the machine off at its end (`sudo shutdown -h +1`); for the manually launched baseline runs no shutdown is scripted, so whether the machine was off between the baseline sessions (9, 10) is unknown. Ordering: the 5 seeds of every configuration ran consecutively (no other SAC configuration in between) — true for all 14; all 5 seeds of a configuration lie in one session — true for 13 of 14, exception(s): Ant-base (sessions [9, 10]). Seed order within every configuration is 331,958,14577,43611,85062: true for all 14 (configuration-major, seed-minor).

Thermal gate across the 70 runs: waited_seconds max 1.788e-05 s; gate temperature 42.00–55.00 °C; gate power 24.62–44.71 W. The gate passes on its first poll because the reference is captured fresh in each process right before (utils/thermal_gate.py); the effective cool-down is the reference stabilisation poll + 30 s settle.

**8.5 Warning/error lines.** `run.log`: lines containing `WARNING`, `ERROR` or `Traceback` (case-insensitive) per run — total over 70 runs = 0, max per run = 0.
No such line in any of the 70 `run.log` files.

Sweep-log blocks (stdout+stderr, 60 runs), distinct matching lines:


| line | count |
|---|---|
| `[codecarbon WARNING @ <time>] Multiple instances of codecarbon are allowed to run at the same time.` | 60 |


## Part 9 — Learning-performance sanity check (SAC)

Definitions (from `aggregate_results.py`, reproduced exactly): episodes = `training_metrics.json["episodes"]` with `phase == "train"` (warmup episodes and the trailing `train_incomplete` partial episode excluded), in order; **final-10 %-mean return** = mean of the last `max(1, n_episodes // 10)` of them; **max return** = max over them. Returns are undiscounted sums of env reward of the **stochastic training policy** (no evaluation episodes exist, Part 2.2). Per-epoch: `training_metrics.json["epochs"][e]["mean_episode_return"]` = mean return of episodes that *ended* during epoch e; "epoch N" = the N-th epoch (0-based index N−1, env_step = N×1000); `None` when no episode ended in that epoch (n < 5 shown). mean ± sd over seeds (ddof = 1). Sanity check only.


| config | final-10%-mean return | max return | # completed train episodes (range) | env steps (measured training) | epoch 10 | epoch 25 | epoch 50 | epoch 75 | epoch 100 |
|---|---|---|---|---|---|---|---|---|---|
| HC-base | 4696 ± 356.1 | 4935 ± 442.2 | 100–100 | 100000 | 155.0 ± 285.9 | 1790 ± 818.7 | 2691 ± 1489 | 4425 ± 658.6 | 4803 ± 334.3 |
| HC-utd2 | 4481 ± 251.7 | 4800 ± 253.6 | 100–100 | 100000 | 303.4 ± 337.7 | 1728 ± 426.2 | 3301 ± 346.3 | 3993 ± 376.4 | 4730 ± 184.8 |
| HC-utd4 | 4108 ± 698.8 | 4457 ± 628.4 | 100–100 | 100000 | 340.7 ± 173.9 | 1294 ± 372.9 | 2628 ± 258.8 | 3709 ± 504.3 | 4415 ± 594.9 |
| HC-w256 | 4885 ± 502.7 | 5113 ± 436.7 | 100–100 | 100000 | -72.31 ± 67.45 | 1696 ± 679.0 | 3387 ± 625.6 | 4244 ± 843.9 | 5028 ± 485.0 |
| HC-w512 | 4364 ± 397.5 | 4591 ± 442.5 | 100–100 | 100000 | 139.2 ± 293.3 | 1634 ± 338.4 | 2413 ± 864.6 | 3737 ± 222.2 | 4420 ± 392.6 |
| HC-b512 | 3819 ± 487.7 | 4183 ± 475.4 | 100–100 | 100000 | 60.90 ± 264.5 | 1091 ± 452.8 | 2593 ± 543.4 | 3239 ± 430.8 | 3963 ± 649.2 |
| HC-b1024 | 2921 ± 337.7 | 3414 ± 299.7 | 100–100 | 100000 | 31.72 ± 180.2 | 660.1 ± 207.5 | 1151 ± 785.2 | 2695 ± 282.7 | 3266 ± 363.8 |
| Ant-base | 158.3 ± 22.30 | 545.4 ± 57.88 | 234–271 | 100000 | 5.870 ± 27.40 | -72.95 ± 128.4 | -4.084 ± 91.86 | 69.81 ± 44.30 | 155.0 ± 64.42 |
| Ant-utd2 | -122.4 ± 57.71 | 146.5 ± 65.97 | 288–327 | 100000 | -200.9 ± 146.7 | -265.0 ± 299.7 | -105.6 ± 127.0 | -218.6 ± 263.5 | -145.6 ± 184.5 |
| Ant-utd4 | -220.1 ± 94.31 | 52.93 ± 21.46 | 375–433 | 100000 | -361.8 ± 373.7 | -321.3 ± 305.9 | -249.7 ± 146.9 | -621.5 ± 304.8 | -539.2 ± 366.4 |
| Ant-w256 | 408.6 ± 107.8 | 1084 ± 181.3 | 178–216 | 100000 | 86.80 ± 99.44 | 198.1 ± 77.91 | 417.8 ± 240.9 | 596.1 ± 236.0 | 297.2 ± 168.4 |
| Ant-w512 | 246.5 ± 58.12 | 611.7 ± 95.41 | 255–325 | 100000 | -34.65 ± 24.19 | -66.10 ± 117.1 | 83.62 ± 115.3 | 190.4 ± 63.65 | 268.8 ± 239.6 |
| Ant-b512 | -54.02 ± 21.25 | 226.7 ± 163.8 | 328–355 | 100000 | -47.60 ± 30.19 | -175.8 ± 165.6 | -205.7 ± 238.7 | -127.3 ± 145.9 | -127.7 ± 142.8 |
| Ant-b1024 | -104.5 ± 12.31 | 80.84 ± 18.09 | 379–511 | 100000 | -85.02 ± 80.19 | -31.09 ± 20.91 | -63.27 ± 32.14 | -319.0 ± 225.4 | -167.9 ± 162.0 |


`training_metrics.json` keys: top level `episodes`, `epochs`; episode records ['episode_idx', 'phase', 'epoch', 'env_step', 'return', 'length']; epoch records ['epoch', 'env_step', 'cumulative_reward', 'epoch_reward_sum', 'epoch_reward_mean_per_step', 'num_episodes_completed', 'mean_episode_return', 'critic_loss_mean', 'actor_loss_mean', 'alpha_end', 'buffer_size'].

## Part 10 — Optional backlog

Not produced in this pass (excluded by the user's instruction). Nothing in Parts 0–9 depends on it.

## Part 11 — Discrepancies, open questions, and things not found

**Contradictions between documentation and code/data:**

1. Some SAC runs labelled `git_commit=f23f448` carry metadata keys (`thermal_gate.run_start_*`) that only exist from commit `92dcee0` onward, and one run carries `thermal_gate.reference_*` keys that no committed version writes: those runs executed with an uncommitted working tree, so `git_commit` alone does not pin their exact code. The visible differences are metadata-logging only.
2. No launcher script exists (in the tree or in git history) for the 10 SAC baseline runs; their CLI is inferred from metadata + `cli_commands.txt`. `metadata.json` never records CLI args or the override path.
3. CLAUDE.md/README claim zero GPU-clock-lock/CPU-governor permission failures and zero RAPL/geolocation fallbacks 'by grepping every run.log'. That grep cannot detect those failures: the `gpu_control`, `thermal_gate` and CodeCarbon loggers do not write to `run.log`. For SAC the claim is supported for the 60 sweep runs by their sweep logs (stderr captured) and for all 70 by data-side checks (varying RAPL CPU power, non-fallback geolocation, metadata thermal-gate reason); the clock lock of the 10 manually launched baselines is verified only as requested (metadata), not as applied.
4. CLAUDE.md says the SAC-HalfCheetah UTD sweep (`d1cebb4`, `results/sac/HalfCheetah-v5/_utd_sweep_status.json`) 'predat[es] the canonical seed set'. The script at `d1cebb4` uses `SEEDS=(331 958 14577 43611 85062)`, the status file lists exactly those seeds, and the 10 HC-utd2/utd4 runs (2026-09-08, metadata commit `ecb98bb`) are at (1024,1024) on the current stack, i.e. they are canonical-seed runs recorded after the HC baselines (2026-09-07).
5. CLAUDE.md says the original SAC/TD3 UTD sweep's bookkeeping was 'silently overwritten'. Only `results/_utd_sweep_status.json` was replaced; `results/_utd_sweep.log` is appended to (`tee -a`) and still contains the full SAC-Ant/TD3 30-run block (2026-09-09, 'Sweep complete: 30/30') followed by the MBPO 20-run block.
6. `algorithms/sac.py`'s `train()` docstring says each epoch's `gradient_updates` energy is allocated 'proportionally to the wall-clock time share of each sub-segment within that epoch'; the code (reconcile step at the end of `train()`) does one split of the whole run's summed energy by the whole run's summed sub-timer times, as README states. The numbers here follow the code.
7. gymnasium/mujoco versions of the Linux run environment are not recorded in `metadata.json`; Part 2.4 env facts come from the local Windows venv (gymnasium 1.3.0). The dims agree with `flops_per_call.json`, but default env kwargs could in principle differ between versions.
8. CodeCarbon's `idle_baseline_head` task integrates energy over a span a few seconds longer than its recorded `duration` (RAM energy / RAM power ≈ 93–94 s vs 90 s; tail spans match), so as-recorded head idle power is biased high by a few percent. This is already noted in `idle_power_analysis.py`; it affects the head–tail drift figure (both versions given in Part 7.2).
9. The context request refers to per-epoch `rollout_{i}`/`gradient_updates_{i}` tasks 'in `emissions.csv`'. In this repo `emissions.csv` holds a single row written by `tracker.stop()`; the per-task rows are in `emissions_<experiment>_<run_id>.csv` in the same run directory (as `compute_energy_per_flop.py`'s docstring says). All task-level numbers here use the per-task file.

**Open questions (facts the thesis author should look at):**

- Part 9 sanity check: HalfCheetah-v5 final-10%-mean returns per configuration are 2921–4885; Ant-v5 values are -220.1–408.6, negative in 4 of 7 configurations (utd2, utd4, b512, b1024). Whether this satisfies the 'SAC learns' sanity check for Ant-v5 within 100k steps is for the thesis author to judge (stated as a fact, not interpreted).

**NOT FOUND / NOT COMPUTED / NOT VERIFIED:**

10. GPU theoretical FP32 peak: NOT DERIVED (Part 4.6 — no SM/lane data in repo; CUDA not available on this checkout).
11. TF32 / float32 matmul precision actually in effect on the run stack: NOT VERIFIED (Part 2.3 — code sets nothing; local CPU torch reports its defaults only).
12. Launcher script/CLI for the 10 SAC baseline runs: NOT FOUND (Part 1.1 — inferred from metadata + cli_commands.txt).
13. Number of update() calls per epoch: NOT LOGGED; verified indirectly (Part 1.3).
14. Simulator share of rollout time: NOT MEASURED (Part 5.6).
15. gymnasium/mujoco versions on the Linux box: NOT RECORDED in metadata (Part 2.4).
16. Applied (vs requested) GPU clock lock for the 10 baseline runs: NOT VERIFIABLE from saved logs (Part 1.3).

**Failed checks:**

- No check FAILED.

**Decisions taken:**

17. Configuration tags are assigned from `metadata.json → algo_config` by diffing against `SACConfig()` defaults (exactly one factor may differ), not from directory names.
18. `gradient_updates` energy per run = Σ of the four allocated keys in `segment_energy.json` (equals the per-task CSV sum to ≤ 2.2e-16 relative); durations of measured tasks = Σ CodeCarbon task `duration` from the per-task CSV.
19. Mean power in all tables = per-seed energy/duration, then mean ± sd over seeds (the pipeline's cross-seed `mean_power_w` uses the same statistic).
20. Allocated sub-segment 'duration' = summed perf_counter host time (as in the pipeline), even though coverage < 100 %.
21. Outlier rule uses the unscaled MAD.
22. 'Epoch N' in Part 9 = 0-based epoch index N−1.
23. Sessions in Part 8.4 are defined with all 300 runs (all algorithms), gap threshold 30 min end→start.
24. Part 10 (optional backlog over all 280 runs) was deliberately not produced, per the user's instruction.

Check tally: 35 checks, 35 PASS, 0 FAIL.

## Part 12 — Index of produced files


| file | description / columns |
|---|---|
| `contexts/section_4.1.md` | this file |
| `contexts/section_4.1_data/build_section_4_1.py` | generator script (read-only on data); rerun to regenerate everything |
| `contexts/section_4.1_data/sac_run_inventory.csv` | 70 runs: tag, env, seed, run_dir, timestamp, git_commit, torch_version, codecarbon_version, gpu_name, gpu_min_clock_mhz, gpu_max_clock_mhz, hidden_sizes, batch_size, utd, warmup_steps, train_steps, steps_per_epoch |
| `contexts/section_4.1_data/sac_segments_long.csv` | 70 runs × 10 segments: tag, env, seed, segment, status, energy_kwh, energy_j, duration_s, mean_power_w, flops, flop_type, j_per_flop, share_of_total_pct, share_of_gradient_updates_pct (target_update: `j_per_flop` holds J per elementwise op; flops = op count) |
| `contexts/section_4.1_data/sac_cross_seed_summary.csv` | 14 configs × 10 segments, cross-seed values of Tables 4.1–4.6: tag, env, config, segment, status, n, architecture_signature, utd, energy_j_mean, energy_j_sd, energy_j_min, energy_j_max, energy_kwh_mean, duration_s_mean, duration_s_sd, mean_power_w_mean, mean_power_w_sd, share_of_total_pct_mean, share_of_total_pct_sd, flops_mean, flop_type, j_per_flop_ratio_of_means, j_per_flop_per_seed_mean, j_per_flop_per_seed_sd, j_per_flop_per_seed_min, j_per_flop_per_seed_max, pj_per_flop_ratio_of_means, jpf_note, gflops_per_s_mean, gflops_per_s_sd, share_of_gu_pct_mean, share_of_gu_pct_sd |
| `contexts/section_4.1_data/sac_per_seed_totals.csv` | Table 4.7 per run: tag, env, seed, run_dir, total_energy_j, total_duration_s, total_mean_power_w, rollout_energy_j, gradient_updates_energy_j, rollout_duration_s, gradient_updates_duration_s |
| `contexts/section_4.1_data/sac_env_comparison.csv` | Part 5 quantities (long format): tag, metric, segment, value |
| `contexts/section_4.1_data/sac_sweep_effects.csv` | Part 6.1: env, tag, metric, paired_mean_pct, paired_sd_pct, from_means_pct (share metrics in pp) |
| `contexts/section_4.1_data/sac_delta_pairs.csv` | Part 6.3 pair list from flop_dashboard.compute_delta_pairs (+ x_dim, ref_mode, level) |
| `contexts/section_4.1_data/sac_ols_fits.csv` | Part 6.4: sweep, env, segment, n, a_J, b_J_per_FLOP, r2, rse_J |
| `contexts/section_4.1_data/sac_energy_components.csv` | Part 7.1: env, tag, task, component, energy_j_mean, energy_j_sd, pct_mean, pct_sd |
| `contexts/section_4.1_data/sac_idle_floor.csv` | Part 7.2 per run: tag, env, seed, run_dir, P_head_w, P_tail_w, rel_head_tail_diff, signed_tail_minus_head_w, head_duration_s, head_integration_window_s, tail_duration_s, tail_integration_window_s, P_head_window_corrected_w, P_tail_window_corrected_w, rel_diff_window_corrected |
| `contexts/section_4.1_data/sac_allocation_coverage.csv` | Part 8.2 per run: env, tag, seed, coverage, T_buffer_sample_s, T_critic_update_s, T_actor_update_s, T_target_update_s, gradient_updates_task_duration_s |
| `contexts/section_4.1_data/sac_per_epoch.csv` | Part 8.3 (14 000 rows): tag, env, seed, epoch, task, energy_j, duration_s, mean_power_w, ram_integration_window_s |
| `contexts/section_4.1_data/sac_per_epoch_stationarity.csv` | Part 8.3 per run×task×quantity: env, tag, seed, task, quantity, mean, sd, cv, early, slope, slope_pct |
| `contexts/section_4.1_data/sac_learning_performance.csv` | Part 9 per run: env, tag, seed, final_10pct_mean_return, max_return, n_train_episodes, env_steps, mean_episode_return_epoch10, mean_episode_return_epoch25, mean_episode_return_epoch50, mean_episode_return_epoch75, mean_episode_return_epoch100 |

