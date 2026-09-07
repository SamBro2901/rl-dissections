"""
Utilities to reduce clock/frequency-scaling confounds between runs.

These wrap nvidia-smi / cpupower calls (Linux). All functions fail *soft*:
if a command isn't available (e.g. testing on a machine without an NVIDIA
GPU) or isn't permitted (e.g. nvidia-smi clock changes and cpupower governor
changes both need root), we log a warning and continue rather than crash the
experiment. Run the whole experiment under `sudo` if you want these confound
controls actually applied; otherwise they no-op with a warning and the run
proceeds unlocked. Every "set" function has a matching "reset" function that
MUST be called at teardown, even on error paths (the runner does this in a
try/finally).
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("gpu_control")

DEFAULT_LOCK_MHZ = 2000  # fixed clock used when no explicit min/max is given --
                          # representative of a real compute workload, not the idle floor


def _run(cmd: list[str]) -> Tuple[bool, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            logger.warning("Command failed (%s): %s", " ".join(cmd), result.stderr.strip())
            return False, result.stderr.strip()
        return True, result.stdout.strip()
    except FileNotFoundError:
        logger.warning("Command not found: %s (skipping)", cmd[0])
        return False, "command not found"
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out: %s", " ".join(cmd))
        return False, "timeout"


def query_gpu_clock_range(gpu_index: int = 0) -> Optional[Tuple[int, int]]:
    """Returns (min_mhz, max_mhz) supported graphics clocks, or None if unavailable."""
    ok, out = _run([
        "nvidia-smi", "-i", str(gpu_index),
        "--query-supported-clocks=graphics", "--format=csv,noheader,nounits",
    ])
    if not ok or not out:
        return None
    try:
        clocks = sorted({int(line.strip()) for line in out.splitlines() if line.strip()})
        return clocks[0], clocks[-1]
    except ValueError:
        return None


def set_persistence_mode(enabled: bool = True, gpu_index: int = 0) -> bool:
    ok, _ = _run(["nvidia-smi", "-i", str(gpu_index), "-pm", "1" if enabled else "0"])
    return ok


def lock_gpu_clocks(min_mhz: Optional[int], max_mhz: Optional[int], gpu_index: int = 0) -> bool:
    """
    Locks GPU graphics clock to a fixed range to suppress boost-clock variability
    between/within runs. If min/max not given, queries the device and locks to
    DEFAULT_LOCK_MHZ (clamped to the device's supported range) -- a fixed clock
    representative of an actual compute workload, not the idle floor.
    """
    if min_mhz is None or max_mhz is None:
        queried = query_gpu_clock_range(gpu_index)
        if queried is None:
            logger.warning("Could not query GPU clock range; skipping clock lock.")
            return False
        qmin, qmax = queried
        target = min(max(DEFAULT_LOCK_MHZ, qmin), qmax)
        min_mhz = min_mhz or target
        max_mhz = max_mhz or target
    ok, _ = _run(["nvidia-smi", "-i", str(gpu_index), "-lgc", f"{min_mhz},{max_mhz}"])
    if ok:
        logger.info("Locked GPU %d clocks to [%d, %d] MHz", gpu_index, min_mhz, max_mhz)
    return ok


def reset_gpu_clocks(gpu_index: int = 0) -> bool:
    ok, _ = _run(["nvidia-smi", "-i", str(gpu_index), "-rgc"])
    if ok:
        logger.info("Reset GPU %d clocks to driver defaults", gpu_index)
    return ok


def set_cpu_governor(governor: str = "performance") -> bool:
    """
    Sets the scaling governor on every CPU core via `cpupower frequency-set -g`
    (cpupower applies to all cores in one call). "performance" pins each core
    to its max frequency, suppressing the idle/boost frequency-scaling
    variance that would otherwise confound the idle-baseline and per-segment
    measurements. Writing to the governor sysfs node needs root -- run under
    sudo, or this fails soft with a warning and the run proceeds unlocked.
    """
    ok, _ = _run(["cpupower", "frequency-set", "-g", governor])
    if ok:
        logger.info("Set CPU governor to '%s'", governor)
    return ok


def get_cpu_governor() -> Optional[str]:
    """Reads the current governor from cpu0's sysfs node. set_cpu_governor
    sets all cores in lockstep, so cpu0 is representative of the rest."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except OSError as e:
        logger.warning("Could not read current CPU governor: %s", e)
        return None


@dataclass
class GpuCpuGuard:
    """
    Context manager that applies persistence mode + clock lock + CPU governor
    on entry and guarantees reset on exit, even if the run raises.

    Usage:
        with GpuCpuGuard(min_mhz=None, max_mhz=None):
            ... run experiment ...
    """
    min_mhz: Optional[int] = None
    max_mhz: Optional[int] = None
    gpu_index: int = 0
    set_persistence: bool = True
    set_governor: bool = True
    _prior_governor: Optional[str] = None

    def __enter__(self):
        if self.set_persistence:
            set_persistence_mode(True, self.gpu_index)
        lock_gpu_clocks(self.min_mhz, self.max_mhz, self.gpu_index)
        if self.set_governor:
            self._prior_governor = get_cpu_governor()
            set_cpu_governor("performance")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_gpu_clocks(self.gpu_index)
        # Deliberately leaving persistence mode ON is usually fine/desired
        # long-term on a dedicated experiment box; not resetting it here.
        # Restore whatever governor was active before, if we changed it.
        if self.set_governor and self._prior_governor:
            set_cpu_governor(self._prior_governor)
        return False  # do not suppress exceptions
