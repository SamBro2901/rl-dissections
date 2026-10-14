#!/usr/bin/env bash
# Task 3 machine fact sheet. NOT RUN by Claude: the session was on the Windows dev checkout, not the
# research PC. Run from the repo root on the Linux research PC:
#
#   bash contexts/step1/scripts/machine_fact_sheet.sh rl-exp/bin/python 2>&1 | tee contexts/step1/03_machine_fact_sheet_raw.txt
#
# $1 = interpreter of the experiment venv (default rl-exp/bin/python). Read-only except a ~3 s matmul
# benchmark (run it OUTSIDE the measurement protocol, nothing else running). No password is requested:
# sudo is only tried non-interactively (sudo -n).
PY="${1:-rl-exp/bin/python}"
hr() { echo; echo "=================== $* ==================="; }

hr "1. OS / kernel"
cat /etc/os-release; uname -r; uname -a

hr "2a. nvidia-smi"
nvidia-smi
hr "2b. driver / kernel module flavour"
cat /proc/driver/nvidia/version 2>&1
modinfo nvidia 2>&1 | head -12
hr "2c. nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS (appendix: long)"
nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS
hr "2d. persistence mode, VBIOS, locked clocks (compact)"
nvidia-smi --query-gpu=name,driver_version,vbios_version,persistence_mode,power.limit,power.default_limit,clocks.sm,clocks.max.sm,clocks.applications.graphics,clocks.default_applications.graphics --format=csv

hr "3. CPU / RAM / board / disks"
lscpu
lsblk -d -o NAME,MODEL,SIZE,ROTA
free -h
echo "--- dmidecode (non-interactive sudo only; if it prints nothing it needs: sudo dmidecode -t memory -t baseboard)"
sudo -n dmidecode -t memory -t baseboard 2>&1 | head -150

hr "4. Governor / RAPL"
(cpupower frequency-info 2>&1 | head -30) || true
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1
ls /sys/class/powercap/ 2>&1
ls -l /sys/class/powercap/intel-rapl*/ 2>&1 | head -40
for f in /sys/class/powercap/intel-rapl:*/energy_uj; do echo "$f: $(cat "$f" 2>&1)"; done
echo "hypervisor flag in lscpu (bare metal if empty):"; lscpu | grep -i hypervisor
systemd-detect-virt 2>&1

hr "5. Python packages"
"$PY" --version
"$PY" -m pip freeze 2>/dev/null | grep -i -E '^(torch|codecarbon|nvidia-ml-py|pynvml|gymnasium|mujoco|numpy|scipy|pandas)=='
"$PY" -c "import torch;print('torch',torch.__version__,'| built with CUDA',torch.version.cuda)"

hr "6. TF32 / precision (runtime state, default as the training scripts see it)"
"$PY" - <<'PYEOF'
import torch
print("get_float32_matmul_precision:", torch.get_float32_matmul_precision())
print("backends.cuda.matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
print("backends.cudnn.allow_tf32:", torch.backends.cudnn.allow_tf32)
PYEOF
echo "--- repo grep (code only)"
grep -rnE "set_float32_matmul_precision|allow_tf32|autocast|bfloat16|float16|half\(" --include=*.py --exclude-dir=rl-exp --exclude-dir=.venv --exclude-dir=contexts --exclude-dir=results . || echo "no hits"

hr "7. Peak FP32 throughput"
"$PY" - <<'PYEOF'
import torch, subprocess, time
p = torch.cuda.get_device_properties(0)
print("name:", p.name, "| compute capability:", f"{p.major}.{p.minor}", "| SMs:", p.multi_processor_count,
      "| total memory GiB:", round(p.total_memory / 2**30, 2))
# FP32 cores per SM ASSUMED from NVIDIA architecture docs: 128 for cc 8.6/8.9/9.0/10.x/12.x (Ampere GA10x, Ada, Hopper, Blackwell),
# 64 for cc 8.0/7.x. VERIFY against the NVIDIA architecture whitepaper / CUDA programming guide table "Compute capabilities".
cores = {(8,0):64,(7,0):64,(7,5):64}.get((p.major,p.minor),128)
print("assumed FP32 cores/SM:", cores)
def smi(q):
    return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], text=True).strip()
boost = float(smi("clocks.max.sm"))
for label, mhz in (("locked clock 2000 MHz (protocol)", 2000.0), (f"boost clock {boost:.0f} MHz (nvidia-smi clocks.max.sm)", boost)):
    print(f"peak FP32 [{label}]: {p.multi_processor_count*cores*2*mhz*1e6/1e12:.2f} TFLOP/s")
# achieved matmul, TF32 OFF, outside protocol
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
n = 8192
a = torch.randn(n, n, device="cuda"); b = torch.randn(n, n, device="cuda")
for _ in range(3): a @ b
torch.cuda.synchronize(); t = time.perf_counter(); k = 20
for _ in range(k): a @ b
torch.cuda.synchronize(); dt = time.perf_counter() - t
print(f"achieved FP32 (TF32 off) {n}x{n} matmul: {2*n**3*k/dt/1e12:.2f} TFLOP/s  (clocks as currently set; NOT under the protocol)")
PYEOF

hr "8. Environment dimensions (gym.make(env_id) with NO kwargs, as experiment_runner._make_env does)"
"$PY" - <<'PYEOF'
import gymnasium as gym
for e in ("HalfCheetah-v5", "Ant-v5"):
    env = gym.make(e)
    print(e, "obs", env.observation_space.shape, "act", env.action_space.shape,
          "| spec kwargs:", env.spec.kwargs, "| use_contact_forces:", getattr(env.unwrapped, "_use_contact_forces", "n/a"))
PYEOF

hr "9. Provenance recorded in the 280 canonical metadata.json"
"$PY" - <<'PYEOF'
import glob, json
from collections import Counter
seeds = {331, 958, 14577, 43611, 85062}
metas = []
for p in glob.glob("results/*/*/seed_*/*/metadata.json"):
    m = json.load(open(p))
    if m.get("seed") in seeds: metas.append(m)
print("n canonical:", len(metas))
for k in ("torch_version","codecarbon_version","device","cuda_device_name","git_commit"):
    print(k, dict(Counter(m.get(k, "<missing>") for m in metas)))
print("fields present in metadata:", sorted(set().union(*(m.keys() for m in metas))))
PYEOF
echo "(per-run python version / OS string / driver are in emissions_base_*.csv columns python_version, os, codecarbon_version; not in metadata.json)"

hr "10. Git"
git rev-parse HEAD; git status --short | head; ls LICENSE* 2>&1; git remote -v; git tag
git log -1 --format='%H %ad %s' 2b4882f; git log -1 --format='%H %ad %s' 050b3c3
