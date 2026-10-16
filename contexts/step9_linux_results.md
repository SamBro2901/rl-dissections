# Step 9: Linux-pending results

- Date (UTC): Fri 2026-10-16 09:12 to about 09:30 UTC (`date -u`, raw: `contexts/step9/header.txt`)
- Hostname: `samanth-research-pc`
- Repo HEAD: `788eb49` (`2026-10-14 20:37:56 +0200 Context for latex builds`), branch `linux-main`
- `git status --short` at start: clean (only `contexts/step9/` appeared after I created it)
- Session type: `/home/samanth` exists, `/sys/class/powercap` exists, venv `/home/samanth/rl-dissection-linux/rl-exp/bin/python` exists
- **All driver, clock, governor and package values below are today's (2026-10-16), not necessarily those of 2026-09-07 to 2026-09-19.**
- Session constraint found: `sudo -n true` fails here ("no new privileges" flag is set), so nothing needing root could run in this session (raw: `A2.txt`, `A5.txt`).
- One deviation from the instructions: the user installed `scipy` into `rl-exp` during the session (needed for A12). The `pip freeze` in A6 was taken **before** that install (no scipy); the post-install versions are in `A6_analysis_versions_after_scipy.txt`.

## Status table

| ID | Title | Status | Raw file | Thesis label waiting |
|---|---|---|---|---|
| A1 | OS and kernel | DONE | `A1.txt` | `tab:hardware`, `tab:thr-provenance` Distribution row, 7.2 |
| A2 | CPU, board, storage, RAM | DONE, RAM type/speed NOT AVAILABLE (needs sudo) | `A2.txt` | `tab:hardware`, `tab:thr-provenance` row 5 |
| A3 | GPU, driver, clocks, power | DONE | `A3.txt`, `A3_clock_range.txt` | `tab:hardware`, `tab:thr-provenance` rows 2-3, 6.4, `tab:app-repro` |
| A4 | CPU governor (current state) | DONE | `A4.txt` | `sec:confounds`, `tab:control-evidence` |
| A5 | RAPL | DONE, RAPL mode NOT confirmed today | `A5.txt`, `A5_python.txt`, `A5_runlog_check.txt` | 7.1.1, `sec:units` |
| A6 | Python environment | DONE | `A6.txt`, `A6_analysis_versions_after_scipy.txt` | `tab:software`, `tab:thr-provenance` row 1 |
| A7 | TF32 and device properties | DONE | `A7.txt` | TF32 todos 4.1-4.4, 7.2, `tab:thr-provenance` row 4 |
| A8 | Environment facts | DONE, MBPO reference code NOT AVAILABLE | `A8.txt` | 3.6, 4.3, 4.4, `sec:thr-design` |
| A9 | Random-policy reference | DONE | `A9.txt`, `A9_below_random.txt` | `sec:thr-learning`, `tab:app-random`, 8.4 |
| A10 | Repository facts | DONE | `A10.txt` | 8.2 contribution 5, Chapter 5 scripts |
| A11 | Data inventory | DONE | `A11.txt` | clock appendix, README |
| A12 | Cross-platform reproduction | DONE, Windows outputs not on this machine so no diff | `A12_summary.txt`, `A12_summary_rerun.txt`, `section_*_checks_linux.txt`, `verify_*_linux.txt` | Chapters 5-8, Appendix C |
| B1 | NVML counter step spacing | NEEDS USER (sudo; precondition failed) | none | 7.1.1 |
| B2 | Sub-phase timers with synchronisation | NEEDS USER (sudo; precondition failed) | none | 3.4.3, 4.1, 7.1.6 |
| B3 | Lock mechanism test | NEEDS USER (sudo; ASK FIRST item, user chose to run it themselves) | none | 6.4.1, `sec:data-quality` |
| B4 | Buffer-sampler profile | DONE | `B4.txt`, `scratch/profile_sampler.py` | 4.4 |
| B5 | Tracker overhead control | SKIPPED by user decision; script written, not run | `scratch/overhead_control.py` (not run) | 7.1.6 par. 2, 8.4 |

Part B preconditions (rule 5), checked 09:13 UTC: `nvidia-smi --query-compute-apps` listed 4 processes (VS Code GPU process pid 10709, Chrome pid 12469, claude-desktop pid 24397, nautilus pid 25855), so the first condition failed; `uptime` load average was 0.30, 0.58, 0.35 (1 min < 0.5, OK); `sudo -n true` failed. The user was asked and chose "skip, list commands" for B1-B3. B4 is CPU only and was run with the user's agreement.

---

## A1 Operating system and kernel (raw `A1.txt`)

```
PRETTY_NAME="Ubuntu 26.04.1 LTS"   VERSION_CODENAME=resolute
Linux samanth-research-pc 7.0.0-38-generic #38-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 09:10:14 UTC 2026 x86_64
ldd (Ubuntu GLIBC 2.43-2ubuntu2.4) 2.43
```

Resolves: point release is **26.04.1 LTS**; glibc 2.43 confirmed. **Kernel today is 7.0.0-38-generic, the thesis says 7.0.0-31-generic**: a different kernel (updated since the runs; the runs' kernel is not recoverable from this machine today). Proposed cell: "Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic (kernel of the runs 7.0.0-31-generic as recorded earlier; updated afterwards), glibc 2.43".

## A2 CPU, board, storage, RAM (raw `A2.txt`)

```
Model name: Intel(R) Core(TM) Ultra 9 285K | CPU(s): 24 | Thread(s) per core: 1 | Core(s) per socket: 24 | Socket(s): 1
CPU max MHz: 5700.0000 | CPU min MHz: 800.0000 | L3 cache: 36 MiB | L2 40 MiB (12 instances)
board_vendor: ASUSTeK COMPUTER INC. | board_name: ROG MAXIMUS Z890 HERO | bios_version: 3202
nvme0n1 Samsung SSD 9100 PRO 4TB   3.6T  ROTA 0  TRAN nvme
Mem: total 60Gi (free -h); lshw: "64GiB System memory"
```

RAM type/speed/manufacturer: **NOT AVAILABLE**. `dmidecode` needs root (`sudo -n` failed), `/sys/devices/system/edac/.../dimm_mem_type` does not exist, `lshw` without root printed only the total ("64GiB System memory"). Command for the user below. Proposed cell: "Intel Core Ultra 9 285K (24 cores, 24 threads), 64 GiB RAM (60 GiB usable per `free`), type/speed NOT AVAILABLE, ASUS ROG MAXIMUS Z890 HERO (BIOS 3202), Samsung SSD 9100 PRO 4 TB (NVMe)". Note: "BIOS 3202" is today's value.

## A3 GPU, driver, clocks, power limits (raw `A3.txt`, `A3_clock_range.txt`)

```
NVIDIA-SMI 615.71.09   KMD Version: 615.71.09   CUDA UMD Version: 13.4
name: NVIDIA GeForce RTX 5090 | driver 615.71.09 | vbios 98.02.2E.40.7F | persistence_mode Disabled
power.limit 600 W | default 600 W | min 400 W | max 600 W
clocks.max.sm 3105 MHz | clocks.max.graphics 3105 MHz | clocks.max.memory 14001 MHz
clocks.default_applications.graphics: [Requested functionality has been deprecated]
NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.71.09  Release Build  (Fri Sep  4 22:33:33 UTC 2026)
modinfo: filename /lib/modules/7.0.0-38-generic/updates/dkms/nvidia.ko.zst, version 615.71.09, license Dual MIT/GPL
```

Supported graphics clocks (memory 14001 MHz list, 1651 entries): **min 180 MHz, max 3090 MHz** (`A3_clock_range.txt`). **200 MHz and 2000 MHz are inside the supported range (180 to 3090) but neither is an exact entry of the supported list** (`has200=0 has2000=0`; the list is stepped about 7.5 MHz). The request `-lgc 200,200` / `2000,2000` is therefore within the range; whether the driver snapped to a neighbouring step is what B3 tests (it is not answered by A3). Kernel modules: **open** modules (the `/proc/driver/nvidia/version` line names them). `clocks.max.graphics` 3105 MHz (the supported list tops out at 3090). Persistence mode is Disabled today (idle desktop state). The ASUS card model is not shown by `nvidia-smi` (only "NVIDIA GeForce RTX 5090"). The driver build date (Sep 4 2026) predates the runs (Sep 7-19), but the installed driver today may still differ from the one used.

## A4 CPU governor (current state, not that of the runs) (raw `A4.txt`)

```
driver: intel_pstate | energy performance preference: performance | hardware limits: 800 MHz - 4.70 GHz
available cpufreq governors: performance powersave | current policy governor: "powersave"
cpu0 scaling_governor: powersave | scaling_driver: intel_pstate | boost state: supported yes, active yes
```

Today the governor is `powersave` (the harness switches it to `performance` during runs with sudo; this state is the idle state after the runs). Note `cpupower` reports a hardware limit of 4.70 GHz while `lscpu` shows CPU max MHz 5700; I did not resolve this difference.

## A5 RAPL (raw `A5.txt`, `A5_python.txt`, `A5_runlog_check.txt`)

```
/sys/class/powercap/intel-rapl, intel-rapl-mmio, intel-rapl-mmio:0, intel-rapl:0, intel-rapl:0:0, intel-rapl:0:1
intel-rapl:0 = package-0 ; intel-rapl:0:0 = core ; intel-rapl:0:1 = uncore ; intel-rapl-mmio:0 = package-0
-r-------- 1 root root ... energy_uj      (all four files, root only)
cat intel-rapl:0/energy_uj -> Permission denied ; sudo -n cat -> "no new privileges" error
CodeCarbon 3.3.0: is_rapl_available() -> False
WARNING: RAPL - Permission denied reading RAPL file /sys/class/powercap/intel-rapl/subsystem/intel-rapl-mmio/intel-rapl-mmio:0/energy_uj ...
run.log check over all 300 runs: "default power consumption" matches 0, "rapl" mentions 0 (log_level=warning)
```

Resolves: RAPL interfaces exist (package-0, core, uncore) but `energy_uj` is **readable only by root**; as a normal user CodeCarbon falls back (`is_rapl_available()` False), which is why the harness is run with sudo. **"RAPL mode confirmed" is NOT established by this session** (I cannot read the files as root here, and `run.log` is at warning level so it does not print the mode). What the data support: zero CodeCarbon fallback warnings in all 300 `run.log` files (the 4 W per thread message would be a warning). Proposed sentence (plain, not applied): "The RAPL energy files are readable by root only; the runs were launched with sudo, and no run log contains the CodeCarbon fallback warning; the RAPL mode itself was not read out by a separate call."

## A6 Python environment (raw `A6.txt`, `A6_analysis_versions_after_scipy.txt`)

```
Python 3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0]
pip freeze: codecarbon==3.3.0 gymnasium==1.3.0 mujoco==3.12.0 numpy==2.5.2 nvidia-ml-py==13.610.43 pandas==3.0.5 torch==2.13.0
torch 2.13.0+cu130, CUDA 13.0, cuDNN 92000
(scipy, matplotlib: not installed at A6 time)
After user installed scipy: scipy 1.18.1, pandas 3.0.5, numpy 2.5.2, Python 3.14.4
```

Compared with the values the instructions quote for the run files (torch 2.13.0+cu130, CodeCarbon 3.3.0, Python 3.14.4): **identical**; I did not re-verify them against the run files. The analysis versions requested for the reproducibility appendix differ from the Windows analysis (Python 3.13.9, SciPy 1.16.2, pandas 2.3.3, NumPy 2.3.4) on this machine: Python 3.14.4, SciPy 1.18.1, pandas 3.0.5, NumPy 2.5.2. `matplotlib` is not installed. Note: pandas 3.0.5 vs 2.3.3 is a major version difference (relevant to A12).

## A7 TF32 and device properties (raw `A7.txt`)

```
float32_matmul_precision: highest
matmul.allow_tf32: False  cudnn.allow_tf32: True
_CudaDeviceProperties(name='NVIDIA GeForce RTX 5090', major=12, minor=0, total_memory=32199MB, multi_processor_count=170, ..., L2_cache_size=96MB)
SMs: 170  capability: 12 0  total_memory: 33763426304
grep over repo *.py/*.sh/*.txt/*.json for allow_tf32|set_float32_matmul_precision|float32_matmul|TORCH_|CUBLAS|NVIDIA_TF32 (excluding rl-exp, .venv, contexts/): no matches
env NVIDIA_TF32_OVERRIDE / TORCH_* / CUBLAS*: none set in this shell
```

Resolves: PyTorch defaults are in force; the code sets no TF32 flag. FP32 matmuls run in **full FP32** (`float32_matmul_precision` "highest", `matmul.allow_tf32` False); only cuDNN convolutions may use TF32 (`cudnn.allow_tf32` True), and the networks are MLPs (I did not check for convolutions in code). Caveat: this is the state in this Python process today; env vars of the original `sudo` runs were not recorded (sudo resets the environment anyway). Device: 170 SMs, compute capability 12.0, 32199 MB (`total_memory` 33 763 426 304 bytes), L2 96 MB.

## A8 Environment facts (raw `A8.txt`)

```
gymnasium 1.3.0  mujoco 3.12.0
HalfCheetah-v5 obs (17,) act (6,) kwargs {} max_episode_steps 1000
Ant-v5        obs (105,) act (8,) kwargs {} max_episode_steps 1000 ; _include_cfrc_ext_in_observation: True
grep obs_dim/observation in results/mbpo/Ant-v5/seed_331/*/metadata.json: no output
find / -path "*mbpo/env/ant.py": nothing found
```

Confirmed: 17/6 (HalfCheetah-v5) and 105/8 (Ant-v5); the Ant-v5 default **includes contact forces** (`_include_cfrc_ext_in_observation=True`; 105 dimensions). MBPO reference code (27-dimension Ant observation): **reference code not available here**, so that claim stays unverified.

## A9 Random-policy reference return (raw `A9.txt`, `A9_below_random.txt`)

Uniform random actions, 5 canonical seeds x 20 episodes, `reset(seed=s*1000+ep)`, CPU only:

```
HalfCheetah-v5 episodes 100 return mean -272.4 sd 74.7 min -444.5 max -116.8 length mean 1000.0 median 1000.0 terminated 0
  per-seed means: 331 -249.7, 958 -278.4, 14577 -258.6, 43611 -271.7, 85062 -303.5
Ant-v5 episodes 100 return mean -48.6 sd 85.1 min -360.7 max 25.9 length mean 141.1 median 55.0 terminated 93
  per-seed means: 331 -40.4, 958 -40.4, 14577 -80.4, 43611 -41.6, 85062 -40.0
```

Comparison with the thesis last-tenth config-mean returns (the ranges in the instructions match `flop_analysis/output/correlation/config_means.csv` exactly):
- HalfCheetah-v5 (random mean -272.4): every algorithm's lowest config mean (SAC 2920.6, TD3 3684.6, MBPO 4534.8, TD-MPC2 5790.9) is above it; no configuration is below.
- Ant-v5 (random mean -48.6): **below the random mean: all 9 MBPO configs (-756.7 to -244.6) and 4 of 7 SAC configs** (bs1024 utd1 -104.5, bs256 utd2 -122.4, bs256 utd4 -220.1, bs512 utd1 -54.0). **Above it:** SAC bs256 utd1 158.3, h256 408.6, h512 246.5; all 8 TD3 configs (min -30.31); all 5 TD-MPC2 configs (min 125.8). No causes are interpreted.

## A10 Repository facts (raw `A10.txt`)

```
HEAD 788eb49 2026-10-14 20:37:56 +0200 Context for latex builds | tags: none | status: clean
2b4882f commit, 050b3c3 commit, 94e1de2 commit
git ls-tree 2b4882f | grep section_5_(2|3)_(analysis|render): (no output)
git ls-tree 050b3c3: flop_analysis/section_5_2_analysis.py, section_5_2_render.py, section_5_3_analysis.py, section_5_3_render.py
git log: 050b3c3 Section 5 context and data; 2b4882f Section 4.2, 4.3 and 4.4 context and data; 788ffcd; 0bbbbb7; f23f448; f109bd4; ae38808 all present
LICENSE*: none (ls: cannot access) | remote: https://github.com/SamBro2901/rl-dissections.git
```

Resolves: the Chapter 5 scripts (`section_5_{2,3}_{analysis,render}.py`) are in `050b3c3`, **not** in `2b4882f`. No LICENSE file and no git tag exist. **The Linux checkout contains the scripts the instructions called untracked**: `flop_analysis/verify_analytic_flops.py`, `section_5_checks.py` to `section_8_checks.py` exist, are tracked (`git ls-files` lists them), and `contexts/step1/` and `contexts/step1/scripts/` (with `nvml_poll.py`, `sync_timing_run.py`) exist. I did not check which commit introduced the check scripts.

## A11 Data inventory (raw `A11.txt`)

300 `metadata.json` in total; canonical-seed runs 280 = SAC 70, TD3 80, MBPO 80, TD-MPC2 50 (matches the thesis). Non-canonical 20: SAC HC 13, SAC Humanoid 2, MBPO HC 3, TD3 HC 1, TD-MPC2 HC 1 (matches). Per (algo, env), all/canonical: MBPO Ant 45/45, MBPO HC 38/35, SAC Ant 35/35, SAC HC 48/35, SAC Humanoid 2/0, TD3 Ant 40/40, TD3 HC 41/40, TD-MPC2 Ant 25/25, TD-MPC2 HC 26/25.

Unset clock fields (`gpu_min_clock_mhz`/`gpu_max_clock_mhz` null): **15 runs** in the whole dataset (all non-canonical SAC: HC seed 0 x7, seed 42 x3, seed 56 x3, Humanoid seed 56 x2), `start_time_utc` 2026-08-13T09:26:21 to 2026-09-02T11:52:48. Clock value distribution over all 300: (2000, 2000) 284; (200, 200) 1; (None, None) 15. So **"fifteen" is correct, not "sixteen"**. Non-canonical run list (algo, env, seed, start, commit, min, max): see `A11.txt`. The single 200/200 run is MBPO HC seed 0, 2026-09-04T11:09:15, commit ae38808 (one-off, already excluded).

## A12 Cross-platform reproduction (raw `A12_summary.txt`, `A12_summary_rerun.txt`, `*_linux.txt`)

First pass: sections 5-8 exited 1 with `ModuleNotFoundError: No module named 'scipy'` (4-line outputs, kept in `A12_summary.txt`). After the user installed scipy 1.18.1 into `rl-exp`, rerun with `rl-exp/bin/python -I`:

| Script | Exit | Lines (stdout+stderr) | sha256 of `*_linux.txt` | Windows reference |
|---|---|---|---|---|
| section_5_checks | 0 | 668 | `98c44eb050af99c3a6eebecf84a6c24896d9304ba41e14bef13969553b524abc` | 668 lines, `efee977d...` |
| section_6_checks | 0 | 162 | `3d855a765f195edd16b8320c00af276e120252b255fc50410722ea61abba208f` | `41c5c963...` (line count not given) |
| section_7_checks | 0 | 760 (756 stdout + 4 stderr UserWarning lines) | `5f34c48c09b4064b6352d4e2b857d1ef7d0c3938b18a38b142a704c48180055b` (stdout only, stderr dropped: `3dc12a834b10a48beb47832fb273bbb6f5fcc6e091ae1e23e5038a2b0a7b897c`) | 756 lines, `34f35a29...` |
| section_8_checks | 0 | 285 | `886bf235be9a35de83b60788da0ca618d2870b4c6af8280d90ce140c37676f32` | 285 lines, `f9676917...` |
| verify_analytic_flops | 0 | 199 | `37484ca8e1c0e0b873a653cc8c0c2a0a467a1684767b6eb208fee6396582dced` | expects "196 comparisons, 0 differ": **got "196 comparisons, 0 differ"** |

Hashes differ from the Windows ones, as the instructions expected (the output prints the root path and versions; Python 3.14.4/SciPy 1.18.1/pandas 3.0.5/NumPy 2.5.2 here). The Windows outputs (`contexts/section_*_checks_windows.txt`) are **not present** on this machine, so no `diff` was made. Line counts for sections 5, 7 (stdout) and 8 equal the Windows line counts. Section 7 prints a pandas `UserWarning` ("obj.round has no effect with datetime...", `section_7_checks.py:540`) to stderr, which the Windows line count of 756 does not include.

Key numbers for later comparison (all verbatim from the Linux outputs):
- Section 5: HC with TD-MPC2, total: n=27 rho=0.8357 p=5.78e-08 slope=0.4715+-0.0752 R2=0.6110; Ant total: n=29 rho=0.8463 p=7.33e-09 slope=0.5071+-0.0755 R2=0.6253; HC rollout rho=0.8836 slope=0.4231; Ant rollout rho=0.8834 slope=0.3815; HC without TD-MPC2 total n=22 rho=0.6994 slope=0.3772; Ant without TD-MPC2 total n=24 rho=0.7381 slope=0.4372.
- Section 6: R2000 energy 189511.2211 J, duration 1638.110 s, mean power 115.6890 W, J/FLOP 6.0098e-10; R200 energy 568209.9519 J, duration 5731.985 s, power 99.1297 W, J/FLOP 1.8019e-09; ratios R200/R2000 energy 2.99829, duration 3.49915, power 0.85686.
- Section 7: gross HC n=27 rho=0.8357 slope=0.4715 J/FLOP spread 74.38 (min 19.11 pJ, max 1421.26 pJ); gross Ant n=29 rho=0.8463 slope=0.5071 spread 62.18 (min 19.82 pJ, max 1232.48 pJ); net_mean HC rho=0.8812 slope=0.5586 spread 42.03; net_mean Ant rho=0.8904 slope=0.5945 spread 34.92; Spearman(return, FLOPs) HC rho=0.417 p=0.0303, Ant rho=-0.326 p=0.0842.
- Section 8: 280 canonical runs, 56 configurations (Ant 29, HC 27); max sample sd of quoted shares SAC/TD3/TD-MPC2 0.3413 pp, MBPO 2.0932 pp; baseline rollout share min 1.171 % (MBPO HC), max 10.064 % (TD3 Ant); all 46 SAC/TD3/MBPO configs: min 0.637 % (MBPO HC utd4), max 11.307 % (TD3 Ant w256).

`verify_mbpo_fit_flops.py` (CPU, `-I`): exit 0, 16 lines, **byte-identical to the saved log** `flop_analysis/output/verify_mbpo_fit_flops.log` (`diff` empty), last line "ALL PASSED (acceptance: rel diff < 0.5% for every N)", all `rel diff=0.000e+00 -> PASS`, holdout share 7.99 % (raw `verify_mbpo_fit_flops_linux.txt`).

## B1, B2, B3: NEEDS USER (not run)

Not run because the precondition `nvidia-smi --query-compute-apps` is empty failed (desktop GPU processes) and `sudo` does not work in this session. The scripts exist and were read fully: `contexts/step1/scripts/nvml_poll.py` (needs `pynvml`, provided by installed `nvidia-ml-py`) and `sync_timing_run.py`. Commands are in "Commands for the user to run". Corrections to the scripts' docstrings: the docstring examples use `--output-dir results_poll_check` / `results_sync_check` (inside the repo, not `results/`); the instructions ask for `/tmp`. I used `/tmp`.

## B4 Buffer-sampler profile (raw `B4.txt`; script `contexts/step9/scratch/profile_sampler.py`)

Source: `_EpisodeSequenceBuffer.sample()` in `algorithms/tdmpc2.py` (lines 383-402) builds `eligible = [ep for ep in self._episodes if ep["length"] >= horizon]` and a weights array over **all stored episodes at every call**, then a Python loop of `batch_size` slice copies. Settings from the canonical `metadata.json` (TD-MPC2 baseline, both envs): `batch_size=256`, `horizon=3`, `buffer_capacity=1000000`. 200 timed calls after 10 warm-up calls, CPU only, no tracker, `time.perf_counter`:

```
length pool: Ant-v5 TD-MPC2 baseline seed 331 (n=923 episodes, mean 113.8, median 50.0, max 1000)
Ant-like pool lengths     episodes=  105 transitions=  14938  mean=0.4247 ms sd=0.0123
Ant-like pool lengths     episodes=  478 transitions=  59088  mean=0.4669 ms sd=0.0121
Ant-like pool lengths     episodes=  923 transitions= 104187  mean=0.4995 ms sd=0.0118
Ant-like pool lengths     episodes= 2587 transitions= 288880  mean=0.6093 ms sd=0.0115
Ant-like fixed length 100 episodes=  105 mean=0.4042 ms sd=0.0076
Ant-like fixed length 100 episodes=  478 mean=0.4408 ms sd=0.0073
Ant-like fixed length 100 episodes= 2587 mean=0.5742 ms sd=0.0127
HalfCheetah-like 105x1000 episodes=105 transitions=105000 mean=0.4182 ms sd=0.0117
linear fit (pool lengths):  time_ms = 0.4267 + 0.000072 * episodes (R^2=0.9904, 4 points)
linear fit (fixed length):  time_ms = 0.4025 + 0.000067 * episodes (R^2=0.9957, 3 points)
```

Resolves: the time per call grows with the number of stored episodes, 7.2e-5 ms (72 ns) per additional stored episode on top of about 0.40-0.43 ms fixed (batch copy) cost; independent of episode length (fixed length 100 gives the same slope). The absolute cost at 2587 episodes (0.61 ms) is the sampler's time per call; the cost per update is that plus the rest of `update()`, which I did not measure here. Episode counts 478 and 2587 are those given in the instructions; 923 is the count of the actual seed-331 Ant TD-MPC2 baseline run. The lengths were resampled from the real pool (not the real sequence).

## B5 Tracker overhead control: SKIPPED (user decision)

Script written but **not run**: `contexts/step9/scratch/overhead_control.py` (syntax checked with `py_compile` only; untested). It builds the tracker with the same arguments as `experiment_runner.py` lines 152-162 via `ExperimentConfig()` defaults (`measure_power_secs=1.0`, `tracking_mode="machine"`, `force_cpu_power_w=None`). Durations 4.0 s and 0.2 s and 30 repetitions are the values given in the instructions. If run later it must be reported as a new control run, not the validation run the README recommends. Command below.

---

## Part C: derived tables (plain text rows, values only from the outputs above)

**1. `tab:hardware`** (today's state)
- CPU: Intel Core Ultra 9 285K, 24 cores, 24 threads (1 thread/core), max 5700 MHz per `lscpu`
- RAM: 64 GiB (60 GiB usable per `free -h`), type NOT AVAILABLE, speed NOT AVAILABLE
- Mainboard: ASUS ROG MAXIMUS Z890 HERO, BIOS 3202
- Storage: Samsung SSD 9100 PRO 4 TB (NVMe)
- GPU: NVIDIA GeForce RTX 5090 (32 GB, 170 SMs, compute capability 12.0), VBIOS 98.02.2E.40.7F, power limit 600 W (range 400-600 W)
- Driver: 615.71.09, open kernel modules (NVIDIA UNIX Open Kernel Module), CUDA UMD 13.4
- OS: Ubuntu 26.04.1 LTS; kernel 7.0.0-38-generic today (thesis: 7.0.0-31-generic); glibc 2.43

**2. `tab:software`**: Python 3.14.4; PyTorch 2.13.0+cu130 (CUDA 13.0, cuDNN 9.2.0 = 92000); Gymnasium 1.3.0; MuJoCo 3.12.0; nvidia-ml-py 13.610.43; CodeCarbon 3.3.0; NumPy 2.5.2; pandas 3.0.5; SciPy 1.18.1 (installed by the user on 2026-10-16, after A6). Source: `pip freeze` today (A6); today's torch, CodeCarbon and Python equal the values the instructions quote from the run files (I did not re-read the run files for the versions this session, only `metadata.json` fields for clocks and seeds); gymnasium, mujoco, numpy and nvidia-ml-py versions of the runs are not established, so these are today's values.

**3. `tab:thr-provenance` / `tab:app-repro` open rows:** row 1 `pip freeze`: see 2 above; row 2-3 `nvidia-smi`: driver 615.71.09, power limit 600/600 W (default/current), supported graphics clock range 180-3090 MHz (200 and 2000 inside, not exact list entries), `clocks.max.graphics` 3105 MHz, persistence mode Disabled today; row 4 device properties: 170 SMs, cc 12.0, 32199 MB, L2 96 MB, `float32_matmul_precision=highest`, `matmul.allow_tf32=False`; row 5 `lscpu`/`lsblk`/`dmidecode`: CPU/board/SSD as above, RAM type/speed NOT AVAILABLE (dmidecode needs sudo). Distribution row: Ubuntu 26.04.1 LTS.

**4. What changed relative to the Windows-session claims:** (a) the kernel today is 7.0.0-38-generic, not 7.0.0-31-generic; (b) Ubuntu point release is 26.04.1; (c) Ant-v5 observation is 105-dimensional with contact forces, confirmed; the 27-dimension MBPO reference observation could not be checked (no reference code here); (d) TF32: PyTorch defaults, FP32 matmul in full FP32, no flag set in the code; (e) RAPL: files are root-only and `is_rapl_available()` is False without root, so the RAPL mode is not confirmed in this session (only that no run log shows the fallback warning); (f) the clock request values 200 and 2000 MHz are inside the supported range 180-3090 MHz but not exact entries of the supported list; (g) unset clock fields: 15 runs, not 16; (h) the check scripts (`verify_analytic_flops.py`, `section_5_checks.py` to `section_8_checks.py`) are tracked in the Linux checkout, not missing; (i) no LICENSE file and no git tag exist; (j) the Chapter 5 scripts are in `050b3c3`, not `2b4882f`; (k) `scipy` was not installed in `rl-exp` (the check scripts could not run until the user installed it), and analysis versions differ from the Windows ones (Python 3.14.4 vs 3.13.9, SciPy 1.18.1 vs 1.16.2, pandas 3.0.5 vs 2.3.3, NumPy 2.5.2 vs 2.3.4); (l) the Linux machine's GPU compute-process list is not empty during a normal desktop session, so any future run needs the desktop applications closed; (m) in A9, 4 of 7 SAC Ant configs and all 9 MBPO Ant configs have last-tenth returns below the random-policy mean of -48.6.

---

## Commands for the user to run

All need a normal terminal with working sudo, a quiet machine (close VS Code, Chrome, Claude desktop, nautilus so `nvidia-smi --query-compute-apps=pid,name --format=csv` prints no process, load average < 0.5, no experiment running). Run one at a time.

**A2 RAM type and speed:**
```bash
sudo dmidecode -t memory | grep -E "Type:|Speed|Manufacturer|Part Number|Size|Configured" | sort | uniq -c > contexts/step9/A2_dmidecode.txt
```

**A5 RAPL as root (confirms the files CodeCarbon reads):**
```bash
sudo rl-exp/bin/python -c "from codecarbon.core.cpu import is_rapl_available; print(is_rapl_available())" | tee contexts/step9/A5_rapl_root.txt
```

**B1 NVML counter spacing** (terminal 1, then terminal 2 immediately):
```bash
rl-exp/bin/python contexts/step1/scripts/nvml_poll.py --seconds 120 --out /tmp/nvml_poll_sac.csv | tee contexts/step9/B1_poll_summary.txt
```
```bash
sudo rl-exp/bin/python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 331 --train-steps 5000 --warmup-steps 1000 --steps-per-epoch 1000 --output-dir /tmp/results_poll_check
```
(argument names checked in `run_experiment.py`: `--train-steps`, `--warmup-steps`, `--steps-per-epoch`, `--output-dir`.) Afterwards ask me to analyse `/tmp/nvml_poll_sac.csv` (distinct values, spacing median/IQR, step size in J, idle vs load).

**B2 sync timing** (one per algorithm; the matching canonical HalfCheetah-v5 seed-331 baseline dirs found from `metadata.json`: SAC `20260907_121335`, TD3 `20260909_122204` (batch 100, UTD 1, warm-up 10000), MBPO `20260908_194455`, TD-MPC2 `20260910_122845`). I did not ask which length to use because B2 was deferred to you; the full default length is `--train-steps 100000` (SAC about 27 min per the instructions, TD-MPC2 several hours); if you want a reduced run, add `--train-steps <N>` and the ratio is then at that length.
```bash
sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo sac --env HalfCheetah-v5 --seed 331 --output-dir /tmp/results_sync_check --compare-dir results/sac/HalfCheetah-v5/seed_331/20260907_121335 | tee contexts/step9/B2_sac.txt
```
```bash
sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo td3 --env HalfCheetah-v5 --seed 331 --warmup-steps 10000 --output-dir /tmp/results_sync_check --compare-dir results/td3/HalfCheetah-v5/seed_331/20260909_122204 | tee contexts/step9/B2_td3.txt
```
```bash
sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo mbpo --env HalfCheetah-v5 --seed 331 --output-dir /tmp/results_sync_check --compare-dir results/mbpo/HalfCheetah-v5/seed_331/20260908_194455 | tee contexts/step9/B2_mbpo.txt
```
```bash
sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo tdmpc2 --env HalfCheetah-v5 --seed 331 --output-dir /tmp/results_sync_check --compare-dir results/tdmpc2/HalfCheetah-v5/seed_331/20260910_122845 | tee contexts/step9/B2_tdmpc2.txt
```

**B3 lock mechanism test** (follow the instructions' sequence; record the "before" state first; restore with `sudo nvidia-smi -rgc` at the end; sample with the query line given in the instructions):
```bash
nvidia-smi -q -d CLOCK > contexts/step9/B3_before.txt
```

**B5 overhead control (optional, untested script; verify it first):**
```bash
rl-exp/bin/python -m py_compile contexts/step9/scratch/overhead_control.py && echo COMPILE_OK
```
```bash
sudo rl-exp/bin/python contexts/step9/scratch/overhead_control.py --output-dir /tmp/overhead_check | tee contexts/step9/B5_overhead.txt
```
Verify the result by checking that `/tmp/overhead_check/overhead_control.json` exists with 30 energies per duration (4.0 s and 0.2 s) and that `run.log`-style CodeCarbon warnings (RAPL/fallback) are absent in the terminal output.

## Surprises (statements that conflict with the thesis or the instructions)

1. Kernel today is 7.0.0-38-generic, thesis says 7.0.0-31-generic.
2. `sudo` is unusable inside this Claude session ("no new privileges"), so all root-dependent items (dmidecode, RAPL read, B1-B3, B5) cannot run from here.
3. `scipy` was missing in `rl-exp` (A12 first pass failed); it was installed by the user during the session.
4. Instructions say the date is 2026-10-15; the machine date is 2026-10-16 (UTC).
5. The check scripts and `contexts/step1/` are present in the Linux checkout (instructions treat them as possibly untracked).
6. `cpupower` reports a hardware frequency limit of 4.70 GHz whereas `lscpu` reports CPU max MHz 5700 (not resolved).

## Final check

Every item ID (A1-A12, B1-B5) has a status line in the table above. `git status --short` at the end: only `?? contexts/step9/` and `?? contexts/step9_linux_results.md` (no tracked file changed; nothing committed).
