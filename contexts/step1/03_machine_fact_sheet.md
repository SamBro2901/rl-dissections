# Task 3: Machine fact sheet

**STATUS: NOT RUN.** This session ran on the Windows dev checkout (`C:\Users\saman\Desktop\Thesis\Project Repository\rl-dissections`, hostname `Samanth`, Windows 11, no NVIDIA/MuJoCo experiment stack), not on the Linux research PC. The task says to stop in that case and list the commands. None of the hardware, driver, kernel, RAPL, TF32-runtime, peak-FP32 or environment-dimension values below was measured; everything marked NOT DETERMINABLE needs the research PC.

## Question

What are the hardware/software/provenance facts of the machine that produced the canonical runs (OS, kernel, GPU, driver, CPU, RAM, board, governor/RAPL access, package versions, TF32 state, peak FP32, env dimensions, git state)?

## Method

- Script (the exact commands, to be run on the research PC): `contexts/step1/scripts/machine_fact_sheet.sh`. It covers items 1-10 of the task, prints raw output, and does a ~3 s large-matmul benchmark (TF32 off) outside the protocol. It asks for no password (`sudo -n` only).

```bash
cd /home/samanth/rl-dissection-linux          # repo root on the research PC
bash contexts/step1/scripts/machine_fact_sheet.sh rl-exp/bin/python 2>&1 | tee contexts/step1/03_machine_fact_sheet_raw.txt
# if `sudo -n dmidecode` printed nothing (needs a password), run this yourself:
sudo dmidecode -t memory -t baseboard
```

- Item 3 (`dmidecode`): not run, needs sudo — exact command above.
- Paste `03_machine_fact_sheet_raw.txt` back and I will turn it into the summary table (the raw output replaces the "NOT DETERMINABLE" cells).

## Results

### A. What the research PC would have to answer (NOT DETERMINABLE here)

| item | status | needs |
|---|---|---|
| 1 OS release, kernel `uname -r` | NOT DETERMINABLE (partially inferable, see C) | script §1 |
| 2 GPU name, driver version, VBIOS, power limit, default/max/locked clocks, persistence mode, open vs proprietary kernel modules | NOT DETERMINABLE | script §2 (`nvidia-smi`, `-q -d POWER,CLOCK,SUPPORTED_CLOCKS`, `/proc/driver/nvidia/version`, `modinfo nvidia`) |
| 3 CPU (`lscpu`), RAM capacity/type/speed/modules, mainboard, disks | NOT DETERMINABLE (CPU model, thread count and total RAM are in the run data, see C) | script §3; `dmidecode` needs sudo |
| 4 governor, `/sys/class/powercap`, RAPL readability, bare metal | NOT DETERMINABLE | script §4 |
| 5 Python / torch / codecarbon / nvidia-ml-py / gymnasium / mujoco / numpy / scipy / pandas versions | torch and codecarbon recorded in metadata (see C); others NOT DETERMINABLE | script §5 |
| 6 `get_float32_matmul_precision()`, `matmul.allow_tf32`, `cudnn.allow_tf32` at runtime | NOT DETERMINABLE; code-side result in C | script §6 |
| 7 compute capability, SM count, peak FP32 at 2000 MHz and at boost, achieved matmul TFLOP/s | NOT DETERMINABLE | script §7 |
| 8 observation/action space shapes, contact-force setting | NOT DETERMINABLE; construction call in C | script §8 |

### B. Peak FP32: what is known and what is not

Not computed. The task asks for `SM count × FP32 cores per SM × 2 × clock` with the locked 2000 MHz SM clock (protocol value, recorded in all 280 `metadata.json`: `gpu_min_clock_mhz = gpu_max_clock_mhz = 2000`) and the boost clock from `nvidia-smi`. SM count, compute capability, FP32 cores per SM and the boost clock are not available here; **I did not look up or assume them in the md** — the script prints `torch.cuda.get_device_properties(0)`, assumes 128 FP32 cores/SM for compute capability ≥ 8.6 and 64 for 7.x/8.0 (marked in the script as an assumption to verify against NVIDIA's architecture documentation), and measures an achieved 8192² FP32 matmul with TF32 disabled as a sanity upper bound.

### C. Facts that can be established from this checkout (repository / run data), labelled by source

These do **not** replace the machine facts above; they are what the repo and the 280 canonical runs already record.

**C1. Run provenance in the 280 canonical runs** (`metadata.json` and `emissions_base_*.csv`; all values single-valued unless noted):

| field | value | source | runs |
|---|---|---|---|
| GPU model | NVIDIA GeForce RTX 5090 (`cuda_device_name`; `1 x NVIDIA GeForce RTX 5090`) | metadata, emissions CSV | 280 |
| torch | 2.13.0+cu130 | `metadata.json.torch_version` | 280 |
| codecarbon | 3.3.0 | `metadata.json.codecarbon_version` | 280 |
| Python | 3.14.4 | `emissions_base_*.csv.python_version` (not in metadata.json) | 280 |
| OS string | `Linux-7.0.0-31-generic-x86_64-with-glibc2.43` (kernel 7.0.0-31-generic, glibc 2.43) | `emissions_base_*.csv.os` | 280 |
| CPU | Intel(R) Core(TM) Ultra 9 285K, `cpu_count` 24 | emissions CSV | 280 |
| RAM total | 60.816 GB (CodeCarbon `ram_total_size`) | emissions CSV | 280 |
| tracking mode / country | machine / DEU | emissions CSV, metadata | 280 |
| clock lock requested | 2000/2000 MHz, persistence on, CPU governor `performance` requested, `measure_power_secs` 1.0 | metadata `experiment_config` | 280 |
| git commit | 12 distinct commits (`f23f448` … `a3582b5`), see Task 4 item 8 | metadata | 280 |
| **not recorded in any run** | driver version, CUDA runtime/driver, hostname, nvidia-ml-py / gymnasium / mujoco / numpy versions, TF32 flags, dirty-tree flag | – | – |

Package installs "now" vs "recorded": NOT DETERMINABLE here (no access to the `rl-exp` venv; the local Windows `.venv` is a different environment, torch 2.14.0+cpu per earlier notes in `contexts/section_7_data`).

**C2. TF32 / precision, code side.** Search over all `*.py` in the repo (excluding `.venv`): no occurrence of `set_float32_matmul_precision`, `allow_tf32`, `autocast`, `bfloat16`, `float16`, `half(` in any training, agent, runner, FLOP or tooling script (the only hits are in `contexts/**/build_section_*.py`, which are analysis scripts that search for these names; `algorithms/tdmpc2.py:53` mentions `torch.compile` in a docstring saying it is *not* used, and `torch.compile` is not in the requested pattern list). So no code path changes the precision settings; the canonical runs therefore ran with **whatever the PyTorch defaults of torch 2.13.0+cu130 are**. Stated plainly: whether those matmuls ran as full FP32 or TF32 is **not recorded in `metadata.json`** and **not determinable here**; PyTorch's documented defaults are `torch.set_float32_matmul_precision('highest')` and `torch.backends.cuda.matmul.allow_tf32 = False` (i.e. full FP32 for matmuls; cuDNN convolutions default to TF32-allowed, but there are no convolutions in these MLP networks) — this is from PyTorch's documented defaults, not verified on the research PC. Run script §6 to confirm.

**C3. Environment construction** (for item 8): `experiment_runner.py:237-242` `_make_env`: `env = gym.make(env_id)` with **no keyword arguments**, then `env.reset(seed=seed)`, `env.action_space.seed(seed)`. So the Gymnasium defaults of `HalfCheetah-v5` and `Ant-v5` apply, in particular Ant-v5's default `use_contact_forces`. Observation/action dimensions were not instantiated here (MuJoCo not available in this environment); script §8 prints them. The expected values from earlier notes (17/6 and 105/8) are **not confirmed**.

**C4. Git (item 10), this checkout:**
- `git rev-parse HEAD` = `94e1de288e7aa8e3709568c66f3b136048961c90` ("Chapter 6, 7, 8 Context and data", 2026-10-11 19:28:17 +02:00); branch `master`.
- `git status --short`: only `?? contexts/step1/` (this task's outputs; nothing else modified).
- `LICENSE`: **no LICENSE file** in the repository root.
- `git remote -v`: `origin https://github.com/SamBro2901/rl-dissections.git` (fetch and push). `git tag`: **no tags**.
- Commits referenced in the todos: `2b4882f` exists — `2b4882f26e3f9aa96c8a05860d4084ff3c2e5cb0`, 2026-10-08 08:57:07 +02:00, "Section 4.2, 4.3 and 4.4 context and data"; `050b3c3` exists — `050b3c3d2618b7536f0cf5c5069e312fbebd4526`, 2026-10-08 16:48:12 +02:00, "Section 5 context and data".
- The research PC's checkout (`/home/samanth/rl-dissection-linux`) was not available, so its HEAD/status cannot be reported.

## Caveats / things that look odd

- The "provenance" table in C1 is read from what the runs recorded about themselves; none of it was independently verified on the machine.
- The Windows `.venv` here has CodeCarbon 3.3.0 (same version number as recorded); that is what Task 2B used to read the source.
- Python 3.14.4 is the Linux runs' interpreter (CodeCarbon CSV), vs. the Windows dev venv's 3.13 — irrelevant for the data but means the dev venv cannot reproduce the run environment.
- If the 20 hardware/software todos only need C1-C4 plus the script's output, one `tee` of the script on the research PC resolves all of them.
