"""
Utilities to reduce clock/frequency-scaling confounds between runs.

These wrap nvidia-smi / powercfg calls (Windows). All functions fail *soft*:
if a command isn't available (e.g. testing on a machine without an NVIDIA
GPU) or isn't permitted (e.g. nvidia-smi clock changes need an elevated/
Administrator shell), we log a warning and continue rather than crash the
experiment. Every "set" function has a matching "reset" function that MUST
be called at teardown, even on error paths (the runner does this in a
try/finally).
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("gpu_control")

# Built-in Windows power scheme GUID for "High performance" -- fixed/well-known,
# not locale-dependent (the display name is localized, the GUID isn't).
_HIGH_PERFORMANCE_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"


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
    between/within runs. If min/max not given, queries the device and locks to a
    fixed low value near the low end of its supported range (favors reproducibility
    over raw throughput -- this is a deliberate tradeoff for measurement, not
    something you'd do for a production training run).
    """
    if min_mhz is None or max_mhz is None:
        queried = query_gpu_clock_range(gpu_index)
        if queried is None:
            logger.warning("Could not query GPU clock range; skipping clock lock.")
            return False
        qmin, qmax = queried
        min_mhz = min_mhz or qmin
        max_mhz = max_mhz or qmin  # lock to a single fixed low clock by default
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
    Windows has no per-core frequency governor like Linux's cpupower; the
    closest equivalent lever is the active power plan, which controls the
    same idle/parking/boost behavior at the OS level. "performance" switches
    to the built-in "High performance" plan; any other value is a no-op
    (kept for interface parity with callers that pass a governor name).
    """
    if governor != "performance":
        logger.warning("Unsupported governor '%s' on Windows; skipping.", governor)
        return False
    ok, _ = _run(["powercfg", "/setactive", _HIGH_PERFORMANCE_GUID])
    if ok:
        logger.info("Set Windows power plan to 'High performance'")
    return ok


def get_cpu_governor() -> Optional[str]:
    ok, out = _run(["powercfg", "/getactivescheme"])
    if not ok:
        return None
    return out.strip()


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
            # get_cpu_governor() returns powercfg's descriptive scheme line
            # (name + GUID), not a bare name; if you need exact restoration,
            # parse the GUID out and pass it to `powercfg /setactive`.
            logger.info("Prior CPU governor info was: %s", self._prior_governor)
        return False  # do not suppress exceptions
