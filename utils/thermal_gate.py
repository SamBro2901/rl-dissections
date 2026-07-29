"""
Thermal gate: before starting a new measured run, wait until the GPU has
cooled back down toward a reference "cold" state, rather than waiting a
fixed sleep duration. This matters because thermal recovery time is not
constant across a multi-hour session (it gets slower as the room/heatsink
soaks), so a fixed cooldown timer under-cools later runs and over-cools
early ones.

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


def load_or_init_reference(reference_file: str, gpu_index: int = 0) -> Optional[GpuState]:
    """
    On the very first run of a session, there is no prior reference state,
    so we read the current (assumed-cold) GPU state and persist it to disk.
    Subsequent runs (even from a fresh process) load this same reference,
    so every run in a multi-day experiment campaign is gated against the
    *same* original cold baseline, not against a drifting a-la-carte state.
    """
    os.makedirs(os.path.dirname(reference_file) or ".", exist_ok=True)
    if os.path.exists(reference_file):
        with open(reference_file) as f:
            data = json.load(f)
        return GpuState(**data)

    state = _read_gpu_state(gpu_index)
    if state is None:
        return None
    with open(reference_file, "w") as f:
        json.dump({"temp_c": state.temp_c, "power_w": state.power_w}, f)
    logger.info("Initialized thermal reference: %.1f C, %.1f W", state.temp_c, state.power_w)
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
