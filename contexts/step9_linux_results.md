# Step 9: Linux-pending results

- Date (UTC): Part A and B4 on Fri 2026-10-16 09:12 to about 09:30 UTC (`date -u`, raw: `contexts/step9/header.txt`); B1 to B3 and B5 run by the user on 2026-10-16/17; compiled 2026-10-17 12:51 UTC
- Hostname: `samanth-research-pc`
- Repo HEAD at start: `788eb49` (`2026-10-14 20:37:56 +0200 Context for latex builds`), branch `linux-main`; at compile time HEAD is `2d086f2` ("Linux data", the user committed the step 9 files in between)
- `git status --short` at start: clean (only `contexts/step9/` appeared after I created it)
- Session type: `/home/samanth` exists, `/sys/class/powercap` exists, venv `/home/samanth/rl-dissection-linux/rl-exp/bin/python` exists
- **All driver, clock, governor and package values below are those of 2026-10-16/17, not necessarily those of 2026-09-07 to 2026-09-19.**
- Session constraint found: `sudo -n true` fails here ("no new privileges" flag is set), so nothing needing root could run in this session (raw: `A2.txt`, `A5.txt`); the root items were run by the user in their own terminal.
- **The machine rebooted on 2026-10-17 at 10:49 UTC** (`uptime -s` 12:49:40 local), which cleared `/tmp`: the SAC sync run directory (B2) and the B1 poll CSV are lost; see B1 and B2.
- One deviation from the instructions: the user installed `scipy` into `rl-exp` during the session (needed for A12). The `pip freeze` in A6 was taken **before** that install (no scipy); the post-install versions are in `A6_analysis_versions_after_scipy.txt`.

## Status table

| ID | Title | Status | Raw file | Thesis label waiting |
|---|---|---|---|---|
| A1 | OS and kernel | DONE | `A1.txt` | `tab:hardware`, `tab:thr-provenance` Distribution row, 7.2 |
| A2 | CPU, board, storage, RAM | DONE (RAM from user-run `dmidecode`, `A2_dmidecode.txt`) | `A2.txt` | `tab:hardware`, `tab:thr-provenance` row 5 |
| A3 | GPU, driver, clocks, power | DONE | `A3.txt`, `A3_clock_range.txt` | `tab:hardware`, `tab:thr-provenance` rows 2-3, 6.4, `tab:app-repro` |
| A4 | CPU governor (current state) | DONE | `A4.txt` | `sec:confounds`, `tab:control-evidence` |
| A5 | RAPL | DONE (as root: available) | `A5.txt`, `A5_python.txt`, `A5_rapl_root.txt`, `A5_runlog_check.txt` | 7.1.1, `sec:units` |
| A6 | Python environment | DONE | `A6.txt`, `A6_analysis_versions_after_scipy.txt` | `tab:software`, `tab:thr-provenance` row 1 |
| A7 | TF32 and device properties | DONE | `A7.txt` | TF32 todos 4.1-4.4, 7.2, `tab:thr-provenance` row 4 |
| A8 | Environment facts | DONE, MBPO reference code NOT AVAILABLE | `A8.txt` | 3.6, 4.3, 4.4, `sec:thr-design` |
| A9 | Random-policy reference | DONE | `A9.txt`, `A9_below_random.txt` | `sec:thr-learning`, `tab:app-random`, 8.4 |
| A10 | Repository facts | DONE | `A10.txt` | 8.2 contribution 5, Chapter 5 scripts |
| A11 | Data inventory | DONE | `A11.txt` | clock appendix, README |
| A12 | Cross-platform reproduction | DONE, Windows outputs not on this machine so no diff | `A12_summary.txt`, `A12_summary_rerun.txt`, `section_*_checks_linux.txt`, `verify_*_linux.txt` | Chapters 5-8, Appendix C |
| B1 | NVML counter step spacing | DONE (first run lost its CSV; re-run supplied step size; no idle/load split possible) | `B1_poll_summary.txt`, `B1_poll_summary_rerun.txt`, `nvml_poll_sac.csv`, `B1_analysis.txt` | 7.1.1 |
| B2 | Sub-phase timers with synchronisation | DONE, full length (SAC run directory lost, log kept) | `B2_sac.txt`, `B2_td3.txt`, `B2_mbpo.txt`, `B2_tdmpc2.txt` | 3.4.3, 4.1, 7.1.6 |
| B3 | Lock mechanism test | DONE, state restored (verified); "unlocked" control invalid | `B3_*` | 6.4.1, `sec:data-quality` |
| B4 | Buffer-sampler profile | DONE | `B4.txt`, `scratch/profile_sampler.py` | 4.4 |
| B5 | Tracker overhead control | DONE by the user (new control run, not a validation run) | `B5_trial.txt`, `B5_overhead.txt`, `scratch/overhead_control.py` | 7.1.6 par. 2, 8.4 |

Part B preconditions (rule 5), checked 09:13 UTC: `nvidia-smi --query-compute-apps` listed 4 processes (VS Code GPU process pid 10709, Chrome pid 12469, claude-desktop pid 24397, nautilus pid 25855), so the first condition failed; `uptime` load average was 0.30, 0.58, 0.35 (1 min < 0.5, OK); `sudo -n true` failed. The user was asked and chose "skip, list commands" for B1-B3 and later ran them in their own terminal (B3 and B5 after I expanded their instructions). B4 is CPU only and was run with the user's agreement. Preconditions for the user-run B3 (12:32 UTC): `nvidia-smi` listed `ptyxis`, VS Code and claude-desktop GPU processes, load average 0.56 (above 0.5); for B5 (14:40 local): only `ptyxis`, load average 0.51.

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

RAM, from `sudo dmidecode -t memory` run by the user (`A2_dmidecode.txt`, summarised with `sort | uniq -c`, so counts are per field value over the 4 slots): 2 modules installed (`Size: 32 GB` x2, `Size: No Module Installed` x2), **Type DDR5**, manufacturer G Skill Intl, part number `F5-6400J3239G32G`, `Speed: 6400 MT/s`, `Configured Memory Speed: 6400 MT/s`, configured voltage 1.4 V, error correction none. Proposed cell: "Intel Core Ultra 9 285K (24 cores, 24 threads), 64 GiB DDR5-6400 (2 x 32 GB G.Skill F5-6400J3239G32G, configured 6400 MT/s), ASUS ROG MAXIMUS Z890 HERO (BIOS 3202), Samsung SSD 9100 PRO 4 TB (NVMe)". Note: "BIOS 3202" is today's value; the summary does not say which DIMM slots are populated.

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
CodeCarbon 3.3.0: is_rapl_available() -> False (as normal user)
sudo rl-exp/bin/python -c "from codecarbon.core.cpu import is_rapl_available; print(is_rapl_available())"  (run by the user, A5_rapl_root.txt) -> True
WARNING: RAPL - Permission denied reading RAPL file /sys/class/powercap/intel-rapl/subsystem/intel-rapl-mmio/intel-rapl-mmio:0/energy_uj ...
run.log check over all 300 runs: "default power consumption" matches 0, "rapl" mentions 0 (log_level=warning)
```

Resolves: RAPL interfaces exist (package-0, core, uncore) but `energy_uj` is **readable only by root**; as a normal user CodeCarbon 3.3.0 reports RAPL unavailable (`is_rapl_available()` False), **as root it reports True** (user-run, `A5_rapl_root.txt`). That is consistent with the harness being launched with sudo and with zero CodeCarbon fallback warnings in all 300 `run.log` files. It does not read out which RAPL domain CodeCarbon sums, and it is today's check, not a record of the 2026-09 runs. Proposed sentence (plain, not applied): "The RAPL energy files are readable by root only; CodeCarbon 3.3.0 reports RAPL available when run as root (checked 2026-10-17), the runs were launched with sudo, and no run log contains the CodeCarbon fallback warning."

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

## B1 NVML counter step spacing (raw `B1_poll_summary.txt` first run, `B1_poll_summary_rerun.txt`, `nvml_poll_sac.csv`, `B1_run_rerun.txt`, `B1_analysis.txt`)

Run twice by the user with `nvml_poll.py`. **First run** (120 s, SAC run concurrent, output not saved; poll CSV lost in the `/tmp` wipe of the 2026-10-17 reboot): 1200 counter changes, spacing min 0.0961, p5 0.0991, median 0.1000, p95 0.1008, max 0.1037 s, 592 intervals in [0.090, 0.100) s and 607 in [0.100, 0.110) s. Superseded by the re-run for everything except being a second confirmation of the 100 ms period.

**Re-run** (2026-10-17, poll `--seconds 150 --out contexts/step9/nvml_poll_sac.csv`, nominal 1 kHz; 113530 polls = 757 Hz effective), with a concurrent short SAC run (`--train-steps 5000 --warmup-steps 1000 --settle-seconds 10 --idle-baseline-seconds 20 --idle-tail-seconds 20`, run 15:00:25 to 15:02:13 local, training 15:01:17 to 15:01:53, `B1_run_rerun.txt`). The poller was started by hand a moment before; the CSV has no wall-clock column, so the exact overlap with the run is not recorded.

```
rows 1500 (1500 distinct counter values, monotonic, 264 distinct step values, 1 mJ resolution); t_s 0.060 to 149.952
total energy over the window 11235.2 J, mean power 74.95 W
all 1499 full intervals (first row, a partial interval of 5.140 J in 0.0597 s, excluded):
  spacing s:  min 0.0836  p5 0.0992  Q1 0.0998  median 0.1000  Q3 0.1002  p95 0.1008  max 0.1168   (IQR 0.0003 s)
              histogram: [0.080,0.090) 1; [0.090,0.100) 713; [0.100,0.110) 784; [0.110,0.120) 1
  step J:     min 6.989  p5 7.429  Q1 7.439  median 7.464  mean 7.495  Q3 7.553  p95 7.634  max 8.502
              histogram: [6,7) J 1; [7,8) J 1496; [8,9) J 2
  implied power (step / spacing): min 65.0  median 74.7  max 100.8 W
  per 10 s bin: mean power 74.3 to 76.2 W in all 15 bins (step median 7.435 to 7.619 J)
  correlation: step vs spacing r = -0.167; step vs 5-interval rolling power r = 0.880; slope of step on rolling power 0.1017 J per W
  idle vs load split: not possible (see below)
```

Resolves: the NVML total-energy counter advances every 100 ms (median 0.1000 s, IQR 0.0003 s, 99.9 % of intervals between 0.090 and 0.110 s), in steps whose size is **not fixed**: median 7.46 J here, and the step scales with the power during the interval (step is about power x 0.1 s; r = 0.88 with power, slope 0.102 J/W). At 75 W the step is about 7.5 J; the thesis value (median 6.36 J, range 5.86 to 6.92 J) corresponds to a lower power (about 64 W at 0.1 s), which I did not check against the runs. So "about 6 J every 100 ms" is a property of the power level, and the 100 ms period is the stable quantity.

**What this re-run cannot show (idle vs load):** the GPU sat at about 75 W in every 10 s bin, so the poll contains no separate idle and load phase. The polled run's own CodeCarbon GPU energy gives the same level: idle_baseline_head 85.7 W (includes clock ramp), rollout/gradient_updates tasks 73.2 to 78.4 W, idle_baseline_tail 74.4 W (`emissions_base_*.csv`, GPU energy / task duration). The harness locks the GPU at 2000 MHz (metadata `gpu_min_clock_mhz` 2000), which holds the card at about 75 W whether or not the small SAC network is working. The dependence of step size on power is therefore established only within roughly 65 to 101 W, not between the 24 W unlocked idle of B3 and the 500 W of the matmul load of B3.

## B2 Sub-phase timers with `torch.cuda.synchronize()` (raw `B2_sac.txt`, `B2_td3.txt`, `B2_mbpo.txt`, `B2_tdmpc2.txt`)

Run by the user at **full length** (100 epochs x 1000 steps, 100000 env steps, standard warm-up: 5000, TD3 10000), seed 331, HalfCheetah-v5, `--output-dir /tmp/results_sync_check`, each with `--compare-dir` the canonical baseline. Ratio = (sum of the four allocated sub-phase timers) / (measured gradient-update task duration), from the scripts' own output:

| Algorithm | ratio with synchronisation | canonical seed-331 run (without), same formula | thesis value (without) | run duration vs canonical | training energy vs canonical |
|---|---|---|---|---|---|
| SAC | 0.97691 (395.52 s / 404.87 s) | 0.97316 (`20260907_121335`) | 97.26 % | 650.6 s (log) vs 652.6 s, -0.30 % | -0.86 % |
| TD3 (`20261017_133538`) | 0.97268 (189.87 s / 195.20 s) | 0.93303 (`20260909_122204`) | 93.42 % | 431.8 s vs 431.0 s, +0.17 % | -4.69 % |
| MBPO | 0.97682 (394.58 s / 403.94 s) | 0.96968 (`20260908_194455`) | 96.97 % | 1823.5 s vs 1823.5 s, +0.00 % | -3.54 % |
| TD-MPC2 | 0.98384 (1266.04 s / 1286.84 s) | 0.98542 (`20260910_122845`) | 98.50 % | 2586.2 s vs 2635.6 s, -1.88 % | -3.58 % |

Notes (facts only):
- "training energy" = sum of `rollout`, `buffer_sample`, `critic_update`, `actor_update`, `target_update` (plus MBPO's two extras, TD-MPC2's `world_model_pretrain` where present), idle and warm-up excluded, from `segment_energy.json`; the SAC value is computed from the energies printed in `B2_sac.txt`, because the SAC run directory is gone (see below).
- The thesis percentages are as quoted in the instructions; the seed-331 canonical runs give slightly different values for SAC (97.32 %), TD3 (93.30 %) and TD-MPC2 (98.54 %), and equal it for MBPO (96.97 %). I did not recompute the thesis aggregates.
- With synchronisation the ratio rises for SAC (+0.38 percentage points), TD3 (+3.97 pp), MBPO (+0.71 pp) and falls for TD-MPC2 (-0.16 pp), all relative to the seed-331 canonical run. Each is one run (one seed), so no spread is available.
- **TD3 was run twice**: `20261017_125344` (metadata start 10:53:44 UTC; its console output was not saved in `B2_td3.txt`, which is the second run, start 11:35:38 UTC) gives ratio 0.97338 (194.754 s / 200.080 s), computed by me from its `segment_energy.json` and `emissions_base_*.csv` with the script's own formula. The first TD3 run has warm-up 10000 and 100000 train steps in its metadata (clock and torch fields not checked); the second has warm-up 10000, clock 2000/2000, torch 2.13.0+cu130.
- **The SAC sync run directory no longer exists.** It was written on 2026-10-16 under `/tmp/results_sync_check/sac/...` and removed by the reboot of 2026-10-17; the SAC numbers come from the verbatim console log `B2_sac.txt` (the log times there are local, UTC+2). The other three run directories still exist in `/tmp` (volatile: copy them elsewhere if they should be kept).
- All four runs: clocks locked 2000/2000, `torch 2.13.0+cu130`, thermal gate reason `reached_reference` with waited time about 0 s.

## B3 Lock mechanism test (raw `B3_before.txt`, `B3_lock200_*`, `B3_lock2000_*`, `B3_unlocked_*`, `B3_summary.txt`)

Run by the user on 2026-10-17 (14:32 to 14:38 local, 12:32 to 12:38 UTC). State before: persistence mode **Enabled** already (it was Disabled in A3 on 2026-10-16), idle SM clock 195 MHz, P8, 24.05 W, no lock visible in `nvidia-smi -q -d CLOCK` (this driver prints no "locked clock" field). Step 3 (`-pm 1`) was therefore not needed and not run.

```
sudo nvidia-smi -lgc 200,200   -> GPU clocks set to "(gpuClkMin 200, gpuClkMax 200)" for GPU 00000000:02:00.0 / All done.
sudo nvidia-smi -lgc 2000,2000 -> GPU clocks set to "(gpuClkMin 2000, gpuClkMax 2000)" for GPU 00000000:02:00.0 / All done.
clocks.sm samples under load (15 x 1 s):
  lock 200:   n=15 min=202 median=202 max=202 MHz   P1   power 76.15 to 79.89 W   (110 matmuls 8192x8192 fp32 in 20.3 s)
  lock 2000:  n=15 min=1972 median=1980 max=1980 MHz P1  power 498.27 to 515.76 W (1110 matmuls in 20.2 s)
  "unlocked": n=15 min=1972 median=1972 max=1980 MHz P1  power 501.77 to 513.03 W (1100 matmuls in 20.0 s)
```

Resolves: `nvidia-smi -lgc` is accepted by this card and driver today and takes effect. The realised clocks are the nearest entries of the supported list: 200 MHz gave **202 MHz** and 2000 MHz gave **1972/1980 MHz** (202, 1972 and 1980 are entries of the supported list in `A3.txt`; 200 and 2000 are not). So the Chapter 6 request values 200/2000 were accepted and realised within +2 MHz and -28/-20 MHz respectively. At 202 MHz the same load ran about 10 times slower (110 vs 1110 matmuls in about 20 s) at 76 to 80 W instead of about 500 to 516 W.

**Invalid control, do not use:** the "unlocked" samples (14:37:55 to 14:38:09 local) were taken **before** the restore step and after `-lgc 2000,2000`; they show the 2000 lock again (1972 to 1980 MHz, 1100 matmuls) and are not an unlocked reference. My instruction order allowed this (optional step 6 stood before the restore step 8).

**Restore (done later by the user, verified by me 2026-10-17 12:55 UTC):** when I first checked (12:51 UTC) the restore had not been run and the idle GPU showed `Enabled, 1980 MHz, 1980 MHz, P5, 42.95 W`, consistent with the 2000 MHz lock still being active. Afterwards the user ran `sudo nvidia-smi -rgc` (`B3_restore_cmd.txt`: "All done."); `B3_after.txt` ends with `Enabled, 202 MHz, P8, 23.65 W`, and my own query a minute later gave `Enabled, 652 MHz, P8, 27.24 W` (no load from me). So the lock is released (idle P8, about 24 to 27 W, idle clock back to the 195 to 650 MHz range), and persistence mode is Enabled as before the test. I did not diff the full before/after `nvidia-smi -q -d CLOCK` files.

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

## B5 Tracker overhead control (raw `B5_trial.txt`, `B5_overhead.txt`; script `contexts/step9/scratch/overhead_control.py`)

**This is a new control run done on 2026-10-17 (14:42 to 14:46 local). It is not the validation run the README recommends having been performed before the experiments.** Run by the user with `sudo`, idle, tracker built as in `experiment_runner.py` (`measure_power_secs=1.0`, `tracking_mode="machine"`, `force_cpu_power_w=None`, `allow_multiple_runs=True`; the only warning printed is CodeCarbon's "Multiple instances of codecarbon are allowed to run at the same time"; no RAPL permission-denied or 4 W fallback line). The script was corrected once before the run (`ExperimentConfig` needs `algo_name` and `env_id`).

```
trial (n=2):  4.0 s: 328.696 J sd 127.887 J, wall 5.536 s | 0.2 s: 14.196 J sd 0.188 J, wall 0.204 s
full (n=30):  4.0 s: mean 237.498 J, sd 2.603 J, mean wall 4.005 s, mean power 59.3 W
              0.2 s: mean 14.355 J, sd 1.862 J, mean wall 0.205 s, mean power 70.1 W
/tmp/overhead_check/overhead_control.json (30 energies per duration)  [volatile: /tmp]
```

Arithmetic from these numbers (no interpretation): fitting E = a + P x t through the two means gives P = (237.498 - 14.355) / (4.005 - 0.205) = 58.7 W and a = 2.3 J per empty task. For comparison, from the canonical SAC seed-331 baseline `20260907_121335` (`segment_energy.json`): `target_update` 2517.0 J over 100 epochs = 25.17 J per epoch task pair; `buffer_sample` 15.5 J and `rollout` 24.8 J per epoch; idle head 5961.3 J over 90.00 s = 66.2 W as recorded. The empty 0.2 s task cost 14.355 J (57 % of the per-epoch `target_update` energy), of which the fitted intercept is 2.3 J (9 %). Caveats: the intercept rests on two durations only; the first task of the trial was visibly larger (trial 4.0 s sd 127.9 J); the GPU was probably still locked at 2000 MHz (P5 idle) during this run because the B3 restore was done only afterwards (see B3), so the idle machine power here (59.3 W) is probably not the unlocked idle state; the tasks here are the harness's per-epoch tasks, while `target_update` itself is an allocated sub-segment, not a tracker task.


---

## Part C: derived tables (plain text rows, values only from the outputs above)

**1. `tab:hardware`** (today's state)
- CPU: Intel Core Ultra 9 285K, 24 cores, 24 threads (1 thread/core), max 5700 MHz per `lscpu`
- RAM: 64 GiB DDR5, 2 x 32 GB G.Skill F5-6400J3239G32G, 6400 MT/s (configured 6400 MT/s, 1.4 V); 60 GiB usable per `free -h`
- Mainboard: ASUS ROG MAXIMUS Z890 HERO, BIOS 3202
- Storage: Samsung SSD 9100 PRO 4 TB (NVMe)
- GPU: NVIDIA GeForce RTX 5090 (32 GB, 170 SMs, compute capability 12.0), VBIOS 98.02.2E.40.7F, power limit 600 W (range 400-600 W)
- Driver: 615.71.09, open kernel modules (NVIDIA UNIX Open Kernel Module), CUDA UMD 13.4
- OS: Ubuntu 26.04.1 LTS; kernel 7.0.0-38-generic today (thesis: 7.0.0-31-generic); glibc 2.43

**2. `tab:software`**: Python 3.14.4; PyTorch 2.13.0+cu130 (CUDA 13.0, cuDNN 9.2.0 = 92000); Gymnasium 1.3.0; MuJoCo 3.12.0; nvidia-ml-py 13.610.43; CodeCarbon 3.3.0; NumPy 2.5.2; pandas 3.0.5; SciPy 1.18.1 (installed by the user on 2026-10-16, after A6). Source: `pip freeze` today (A6); today's torch, CodeCarbon and Python equal the values the instructions quote from the run files (I did not re-read the run files for the versions this session, only `metadata.json` fields for clocks and seeds); gymnasium, mujoco, numpy and nvidia-ml-py versions of the runs are not established, so these are today's values.

**3. `tab:thr-provenance` / `tab:app-repro` open rows:** row 1 `pip freeze`: see 2 above; row 2-3 `nvidia-smi`: driver 615.71.09, power limit 600/600 W (default/current), supported graphics clock range 180-3090 MHz (200 and 2000 inside, not exact list entries), `clocks.max.graphics` 3105 MHz, persistence mode Disabled on 2026-10-16 and Enabled on 2026-10-17 (before B3); row 4 device properties: 170 SMs, cc 12.0, 32199 MB, L2 96 MB, `float32_matmul_precision=highest`, `matmul.allow_tf32=False`; row 5 `lscpu`/`lsblk`/`dmidecode`: CPU/board/SSD/RAM as above (dmidecode run by the user as root). Distribution row: Ubuntu 26.04.1 LTS.

**4. What changed relative to the Windows-session claims:** (a) the kernel today is 7.0.0-38-generic, not 7.0.0-31-generic; (b) Ubuntu point release is 26.04.1; (c) Ant-v5 observation is 105-dimensional with contact forces, confirmed; the 27-dimension MBPO reference observation could not be checked (no reference code here); (d) TF32: PyTorch defaults, FP32 matmul in full FP32, no flag set in the code; (e) RAPL: files are root-only; `is_rapl_available()` is False as a normal user and True as root (user-run), and no run log shows the fallback warning; the RAPL domain used is still not read out; (f) the clock request values 200 and 2000 MHz are inside the supported range 180-3090 MHz but not exact entries of the supported list; `nvidia-smi -lgc` accepted both and the realised clocks were 202 MHz and 1972/1980 MHz (B3); (g) unset clock fields: 15 runs, not 16; (h) the check scripts (`verify_analytic_flops.py`, `section_5_checks.py` to `section_8_checks.py`) are tracked in the Linux checkout, not missing; (i) no LICENSE file and no git tag exist; (j) the Chapter 5 scripts are in `050b3c3`, not `2b4882f`; (k) `scipy` was not installed in `rl-exp` (the check scripts could not run until the user installed it), and analysis versions differ from the Windows ones (Python 3.14.4 vs 3.13.9, SciPy 1.18.1 vs 1.16.2, pandas 3.0.5 vs 2.3.3, NumPy 2.5.2 vs 2.3.4); (l) the Linux machine's GPU compute-process list is not empty during a normal desktop session, so any future run needs the desktop applications closed; (m) NVML energy counter: updates every 100 ms (median 0.1000 s, IQR 0.0003 s), step about power x 0.1 s: median 7.46 J (range 6.99 to 8.50 J) at about 75 W, not a fixed 6 J; no idle/load split because the GPU stayed at about 75 W; (n) the synchronised sub-phase timers raise the sum-of-timers / measured ratio for SAC, TD3 and MBPO (to 97.69 %, 97.27 %, 97.68 %) and lower it slightly for TD-MPC2 (98.38 %); the TD3 value moves most (canonical seed-331 run 93.30 %); (o) empty tracker task energy: 14.355 J (0.2 s) and 237.498 J (4.0 s), intercept 2.3 J (B5); (p) sampler time per call grows by 72 ns per stored episode on a 0.40 to 0.43 ms base (B4); (q) in A9, 4 of 7 SAC Ant configs and all 9 MBPO Ant configs have last-tenth returns below the random-policy mean of -48.6.

---

## Commands outstanding for the user (all optional)

1. B3 restore: done and verified (see B3). Nothing mandatory remains.
2. **Optional, valid unlocked control** (now that the lock is released, same load script, same sampling): `rl-exp/bin/python contexts/step9/scratch/b3_load.py 20 &` together with the `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.gr,power.draw,pstate --format=csv -l 1` sampler for 15 s, saved as `B3_unlocked2_samples.csv`.
3. B1 re-run: done (see B1). An idle (about 25 W, unlocked) versus heavy-load (about 500 W) poll, e.g. during the B3 matmul load script, would be the only way to test the power dependence over a wide range; it was not done.
4. **Optional, copy the volatile B2/B5 data out of `/tmp`** (the reboot already removed the SAC sync run and the B1 CSV): `cp -r /tmp/results_sync_check /tmp/overhead_check contexts/step9/scratch/` (adds files only under `contexts/step9/`).

Commands already executed by the user and recorded above: `dmidecode` (A2), root RAPL check (A5), B1, B2 (all four algorithms, full length), B3 steps 1 to 7, B5 trial and full run.

## Surprises (statements that conflict with the thesis or the instructions)

1. Kernel today is 7.0.0-38-generic, thesis says 7.0.0-31-generic.
2. `sudo` is unusable inside this Claude session ("no new privileges"); all root items were run by the user in their own terminal.
3. `scipy` was missing in `rl-exp` (A12 first pass failed); it was installed by the user during the session.
4. Instructions say the date is 2026-10-15; the machine date is 2026-10-16/17 (UTC).
5. The check scripts and `contexts/step1/` are present in the Linux checkout (instructions treat them as possibly untracked).
6. `cpupower` reports a hardware frequency limit of 4.70 GHz whereas `lscpu` reports CPU max MHz 5700 (not resolved).
7. The B3 restore (`-rgc`) was not run when I first checked (idle GPU at 1980 MHz, P5, 42.95 W); it was run afterwards and verified (idle P8, 23.65 W and 27.24 W).
8. **`/tmp` was wiped by a reboot** (2026-10-17 10:49 UTC): the SAC sync run directory and the first B1 poll CSV are lost (B1 was re-run and its CSV kept in `contexts/step9/`).
9. The "unlocked" B3 control was taken while the 2000 MHz lock was still set (my ordering of the optional step allowed it): invalid.
10. Persistence mode was Disabled on 2026-10-16 (A3) and Enabled on 2026-10-17 (B3 before); who or what enabled it is unknown.
11. `results/_thermal_reference.json` (tracked, inside `results/`) was overwritten by the B2 sync runs even with `--output-dir /tmp/results_sync_check` (the harness writes the thermal reference to its fixed path `results/_thermal_reference.json`); the committed version already holds the SAC sync run's values (temp 47.0, power 28.534, pid 138900) and the working tree holds the TD-MPC2 sync run's (46.0, 27.636). CLAUDE.md calls this file audit-only; no run data under `results/` changed (checked via `git status`: only that file is modified there).
12. Two TD3 sync runs exist (`20261017_125344` and `20261017_133538`); only the second one's console output was saved.
13. The poll rate of B1 was 727 Hz (first run) and 757 Hz (re-run), not the nominal 1 kHz.
14. The NVML step is not a fixed 6 J: it was 7.46 J (median) at about 75 W in the re-run.

## Final check

Every item ID (A1-A12, B1-B5) has a status line in the table above (checked with `grep "^| <ID> "`). `git status --short` at compile time (2026-10-17 12:55 UTC), relative to HEAD `2d086f2`:
- modified tracked files: `contexts/step9/B3_before.txt` (overwritten by the user's B3 step 2), `contexts/step9/scratch/overhead_control.py` (my fix of the `ExperimentConfig` call, before B5), `contexts/step9_linux_results.md` (this file), and `results/_thermal_reference.json` (written by the harness during the B2 runs, see Surprises 11);
- all other changes are new untracked files under `contexts/step9/`.
Nothing was committed or pushed by me. No run data under `results/` was changed.
