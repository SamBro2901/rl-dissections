"""
Thermal gate: before starting a new measured run, wait until the GPU has
cooled back down toward a reference "cold" state, rather than waiting a
fixed sleep duration. This matters because thermal recovery time is not
constant across a multi-hour session (it gets slower as the room/heatsink
soaks), so a fixed cooldown timer under-cools later runs and over-cools
early ones.

The reference is captured fresh at the start of every process (not
reused from a previous script invocation's saved file), since a "cold"
baseline from a run several hours ago is not a meaningful target for a
run starting now. Because two runs are often launched back-to-back, the
capture first polls until the reading stabilizes -- this stops a run
from locking in a still-cooling-down reading (left over from the
*previous* run's workload) as its own baseline, which would silently
defeat the gate by making it pass immediately.

Falls back to a fixed sleep if pynvml/NVML is unavailable (e.g. no GPU
present, or you're doing a CPU-only reference run).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("thermal_gate")

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

# Cached per-process so repeated calls within the same run (e.g. gating
# before each of several episodes) reuse the same captured reference
# instead of re-running the stabilization poll every time. A new script
# invocation is a new process, so this is naturally empty again on
# every run -- nothing to reset by hand.
_RUN_REFERENCE: Optional["GpuState"] = None


@dataclass
class GpuState:
    temp_c: float
    power_w: float


def _read_gpu_state(gpu_index: int = 0) -> Optional[GpuState]:
    if not _NVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
        return GpuState(temp_c=float(temp), power_w=power_mw / 1000.0)
    except Exception as e:  # pragma: no cover - defensive; NVML errors are varied
        logger.warning("NVML read failed: %s", e)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _capture_stable_reference(
    gpu_index: int = 0,
    stabilize_window: int = 3,
    stabilize_tolerance_c: float = 0.5,
    stabilize_poll_interval_seconds: float = 5.0,
    stabilize_max_wait_seconds: float = 120.0,
) -> Optional[GpuState]:
    """
    Polls the GPU until its temperature stops moving (stabilize_window
    consecutive readings all within stabilize_tolerance_c of each other),
    then returns that reading. This avoids capturing a reference mid-way
    through a cooldown (e.g. right after a prior run's workload ended),
    which would lock in a still-hot value as the "reference" and make the
    gate a no-op.

    Gives up after stabilize_max_wait_seconds and returns the last reading
    with a warning, rather than blocking forever.
    """
    start = time.time()
    recent: list[float] = []
    last_state: Optional[GpuState] = None
    while True:
        state = _read_gpu_state(gpu_index)
        if state is None:
            return None
        last_state = state
        recent.append(state.temp_c)
        recent = recent[-stabilize_window:]
        if len(recent) == stabilize_window and (max(recent) - min(recent)) <= stabilize_tolerance_c:
            return state
        if time.time() - start >= stabilize_max_wait_seconds:
            logger.warning(
                "Thermal reference did not stabilize within %.0fs (recent temps: %s); "
                "using last reading anyway.",
                stabilize_max_wait_seconds, recent,
            )
            return last_state
        time.sleep(stabilize_poll_interval_seconds)


def load_or_init_reference(reference_file: str, gpu_index: int = 0) -> Optional[GpuState]:
    """
    Captures this process's own thermal reference. Never reused from a
    previous script invocation's saved file -- each new run gets a fresh
    baseline, since GPU/room conditions drift across a multi-hour session.

    Within a single run, repeated calls reuse the same in-process reading
    rather than re-polling every time.

    The reference file is still written for audit purposes (so you can see
    what each run was gated against after the fact), but it is always
    overwritten, never read back.
    """
    global _RUN_REFERENCE
    if _RUN_REFERENCE is not None:
        return _RUN_REFERENCE

    state = _capture_stable_reference(gpu_index)
    if state is None:
        return None

    os.makedirs(os.path.dirname(reference_file) or ".", exist_ok=True)
    with open(reference_file, "w") as f:
        json.dump(
            {
                "temp_c": state.temp_c,
                "power_w": state.power_w,
                "pid": os.getpid(),
                "captured_at": time.time(),
            },
            f,
        )
    logger.info(
        "Captured thermal reference for this run (pid=%d): %.1f C, %.1f W",
        os.getpid(), state.temp_c, state.power_w,
    )

    _RUN_REFERENCE = state
    return state


def wait_for_thermal_baseline(
    reference_file: str,
    temp_tolerance_c: float = 3.0,
    power_tolerance_w: float = 5.0,
    max_wait_seconds: float = 600.0,
    poll_interval_seconds: float = 5.0,
    gpu_index: int = 0,
) -> dict:
    """
    Blocks until GPU temp/power are within tolerance of the reference cold
    state, or until max_wait_seconds elapses (returns anyway, but logs a
    warning and records this fact so you can flag/exclude the run later).

    Returns a small dict summarizing the wait, meant to be dumped into the
    run's metadata.json for auditability.
    """
    reference = load_or_init_reference(reference_file, gpu_index)
    if reference is None:
        logger.warning("No NVML reference available; falling back to fixed 30s sleep.")
        time.sleep(30)
        return {"gated": False, "reason": "nvml_unavailable", "waited_seconds": 30}

    start = time.time()
    while True:
        state = _read_gpu_state(gpu_index)
        if state is None:
            time.sleep(poll_interval_seconds)
            continue
        temp_ok = abs(state.temp_c - reference.temp_c) <= temp_tolerance_c
        power_ok = abs(state.power_w - reference.power_w) <= power_tolerance_w
        elapsed = time.time() - start
        if temp_ok and power_ok:
            return {
                "gated": True,
                "reason": "reached_reference",
                "waited_seconds": elapsed,
                "final_temp_c": state.temp_c,
                "final_power_w": state.power_w,
            }
        if elapsed >= max_wait_seconds:
            logger.warning(
                "Thermal gate timed out after %.0fs (temp=%.1fC vs ref %.1fC, "
                "power=%.1fW vs ref %.1fW). Proceeding anyway -- flag this run.",
                elapsed, state.temp_c, reference.temp_c, state.power_w, reference.power_w,
            )
            return {
                "gated": False,
                "reason": "timeout",
                "waited_seconds": elapsed,
                "final_temp_c": state.temp_c,
                "final_power_w": state.power_w,
            }
        time.sleep(poll_interval_seconds)
