# Research-PC checklist (NOT RUN, nothing here was executed)

Written for the appendices of `thesis-main.tex` (line numbers are those of the file **after** the insertion of the appendices, 2026-10-14; they drift as you edit).
Run on the Linux research PC in `/home/samanth/rl-dissection-linux` with the environment `rl-exp`.
Precondition for every item that touches the GPU: `nvidia-smi` shows no compute process and `uptime` shows a load below 0.5. None of the items is a measurement run.
The script `contexts/step1/scripts/machine_fact_sheet.sh` covers items 1 to 9 in one go (`bash contexts/step1/scripts/machine_fact_sheet.sh rl-exp/bin/python 2>&1 | tee fact_sheet.txt`); the single commands are listed so that each output can be pasted into its place.

| # | Fills (location in `thesis-main.tex`) | Command | Output to paste | sudo |
|---|---|---|---|---|
| 1 | `tab:thr-provenance`, row "Gymnasium, MuJoCo, `nvidia-ml-py`" (L11299); `tab:software` (Section 3.7); todos at L2616, L2818, L4563 | `rl-exp/bin/python -m pip freeze` and `rl-exp/bin/python -c "import gymnasium, mujoco, torch; print(gymnasium.__version__, mujoco.__version__, torch.__version__)"` | versions of Gymnasium, MuJoCo, `nvidia-ml-py` and every other package; also `rl-exp/bin/python --version` (the metadata says 3.14.4) | no |
| 2 | `tab:thr-provenance`, row "Driver, kernel modules, VBIOS" (L11300); `tab:hardware`/`tab:software` | `nvidia-smi`; `nvidia-smi --query-gpu=name,driver_version,vbios_version,persistence_mode,power.limit,power.default_limit --format=csv`; `cat /proc/driver/nvidia/version`; `modinfo nvidia \| head -12` | driver version, VBIOS version, whether the open kernel modules are used | no |
| 3 | `tab:thr-provenance`, row "Power limits, supported clocks" (L11301); Chapter 6 todo L6429 (clock range, power limit, driver) | `nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS` | the full listing (long: goes to the appendix or is summarised), label every value with the date of the run (the state of 2026-09-04 was not recorded) | no |
| 4 | `tab:thr-provenance`, row "SM count, compute capability, peak FP32" (L11302) | `rl-exp/bin/python -c "import torch; p=torch.cuda.get_device_properties(0); print(p); print(p.multi_processor_count, p.major, p.minor)"` | device properties | no |
| 5 | `tab:thr-provenance`, row "RAM type/speed, mainboard, storage" (L11303); `tab:hardware` | `lscpu`; `lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN`; `cat /sys/class/dmi/id/board_vendor /sys/class/dmi/id/board_name`; `sudo dmidecode -t memory -t baseboard` | CPU model line, storage model and size, mainboard, RAM type and speed | `dmidecode` only |
| 6 | `tab:thr-provenance`, row "Distribution" (point release of Ubuntu 26.04 LTS) | `cat /etc/os-release` | `VERSION=` line | no |
| 7 | Todo at L6475 (NOT RUN: RAPL mode); todo at L2294 | `ls -d /sys/class/powercap/intel-rapl*`; `cat /sys/class/powercap/intel-rapl:0/name`; the CodeCarbon start-up line of a short run (`python run_experiment.py --algo sac --env Pendulum-v1 --train-steps 2000 --warmup-steps 500 --steps-per-epoch 500 --settle-seconds 2 --idle-baseline-seconds 3 --idle-tail-seconds 3 --no-gpu-lock --no-thermal-gate --output-dir results_smoketest`) | whether `intel-rapl` exists and which mode CodeCarbon reports | no |
| 8 | Todo at L6759 (NOT RUN: TF32); todos in Sections 4.1 and 7.2 | `rl-exp/bin/python -c "import torch; print(torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)"` | the three values | no |
| 9 | Todos L6426 (NOT RUN: lock check) and L6429 | `nvidia-smi -q -d CLOCK` (state before); `sudo nvidia-smi -lgc 200,200`; short GPU load while `nvidia-smi --query-gpu=clocks.sm,clocks.gr --format=csv -l 1`; `sudo nvidia-smi -rgc`; then restore the lock of the harness (`sudo nvidia-smi -lgc 2000,2000`). Author decision (F-6.2 of `todo-changes-step6.md`); a test today shows that the mechanism works, not that the lock was applied on 2026-09-04 | clocks during the load | **yes** |
| 10 | Random-policy returns: todo in Appendix D (L11247), also `sec:sac` (L2965) and `sec:tdmpc2` | command F3 of `todo-changes-step4.md` (below) | return mean and sd, length mean, number of episodes (100 per environment) into `tab:app-random` | no |
| 11 | Todo L6495 (NOT RUN: NVML counter update period) | terminal 1: `python contexts/step1/scripts/nvml_poll.py --seconds 120 --out nvml_poll_sac.csv`; terminal 2, right after: `sudo rl-exp/bin/python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 331 --train-steps 5000 --warmup-steps 1000 --steps-per-epoch 1000 --output-dir results_poll_check` (script written in Step 1, not run; quiet machine, clocks locked as in the protocol) | update period of `nvmlDeviceGetTotalEnergyConsumption` (histogram printed by the script) | the run: yes |
| 12 | Todos L2147 and L3167 (`torch.cuda.synchronize()` run) | one baseline run per algorithm, e.g. `sudo rl-exp/bin/python contexts/step1/scripts/sync_timing_run.py --algo sac --env HalfCheetah-v5 --seed 331 --output-dir results_sync_check` (TD3 with `--warmup-steps 10000`; `--compare-dir` takes the matching canonical run; script written in Step 1, not run; quiet machine) | ratio of the summed sub-phase times to the measured gradient-update duration, printed by the script, for the run and for the canonical baseline | yes |
| 13 | Todo L6809 (NOT RUN: "fill from the research PC", `tab:thr-summary`) | outputs of items 1 to 9 | the "Incomplete provenance" row | no |
| 14 | Todo L10868 in Appendix C (NOT RUN: rerun of `verify_mbpo_fit_flops.py`, optional, any machine with PyTorch) | `rl-exp/bin/python flop_analysis/verify_mbpo_fit_flops.py` | the six counted/predicted pairs of `tab:app-flop-fitcheck` (must be identical) | no |
| 15 | Todo L11447 in Appendix E (NOT RUN: package versions of the experiment environment) | item 1 | versions | no |

Command F3 (random-policy reference return; 5 canonical seeds x 20 episodes = 100 episodes per environment, same environment settings, no learning, no energy measurement):

```bash
rl-exp/bin/python - <<'EOF'
import numpy as np, gymnasium as gym
seeds = [331, 958, 14577, 43611, 85062]
for env_id in ("HalfCheetah-v5", "Ant-v5"):
    rets, lens = [], []
    for s in seeds:
        env = gym.make(env_id)
        env.action_space.seed(s)
        for ep in range(20):
            obs, _ = env.reset(seed=s * 1000 + ep)
            done, R, n = False, 0.0, 0
            while not done:
                obs, r, term, trunc, _ = env.step(env.action_space.sample())
                R += r; n += 1; done = term or trunc
            rets.append(R); lens.append(n)
    print(env_id, "episodes", len(rets), "return mean %.1f sd %.1f" % (np.mean(rets), np.std(rets, ddof=1)),
          "length mean %.1f" % np.mean(lens))
EOF
```

Not on the Linux machine but open for you: commit/tag/LICENSE (contribution 5, L11529), commit of `flop_analysis/verify_analytic_flops.py` (L10822), the README/CLAUDE.md statement on the unset clock fields (L11560).
