"""
Orchestrates one full, documented experiment run:

    [settle] -> [idle baseline] -> [warmup + measured training] -> [idle tail] -> [teardown]

Produces, per run, a self-contained directory:

    results/{algo}/{env}/seed_{n}/{timestamp}/
        emissions.csv       <- CodeCarbon's native per-task log
        segment_energy.json <- reconciled per-segment energy breakdown (kWh)
        metadata.json        <- full run provenance (git hash, versions, config, thermal gate info)
        run.log               <- this run's log output

Run via:
    python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 0

See README.md for the methodological caveats around the segment_energy.json
allocation scheme (rollout / gradient_updates are directly measured;
buffer_sample / critic_update / actor_update / target_update are allocated
from gradient_updates by wall-clock time share).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from configs.config import ExperimentConfig, SACConfig
from utils.gpu_control import GpuCpuGuard
from utils.thermal_gate import wait_for_thermal_baseline


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("experiment_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(run_dir / "run.log")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _make_run_dir(exp_cfg: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(exp_cfg.output_dir) / exp_cfg.algo_name / exp_cfg.env_id / f"seed_{exp_cfg.seed}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _select_device(requested: str, logger: logging.Logger) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def run_experiment(exp_cfg: ExperimentConfig, algo_cfg, seed_everything: bool = True):
    """
    algo_cfg: an algorithm-specific config object (currently SACConfig).
    Dispatch on exp_cfg.algo_name to pick the right train() function; add
    elif branches here as you add PPO/MBPO/PETS.
    """
    run_dir = _make_run_dir(exp_cfg)
    logger = _setup_logging(run_dir)
    logger.info("=== Experiment run: algo=%s env=%s seed=%d ===", exp_cfg.algo_name, exp_cfg.env_id, exp_cfg.seed)
    logger.info("Run directory: %s", run_dir)

    if seed_everything:
        import random
        import numpy as np
        random.seed(exp_cfg.seed)
        np.random.seed(exp_cfg.seed)
        torch.manual_seed(exp_cfg.seed)

    device = _select_device(exp_cfg.device, logger)
    logger.info("Using device: %s", device)

    metadata = {
        "algo_name": exp_cfg.algo_name,
        "env_id": exp_cfg.env_id,
        "seed": exp_cfg.seed,
        "git_commit": _git_commit_hash(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "start_time_utc": datetime.utcnow().isoformat(),
        "experiment_config": exp_cfg.__dict__,
        "algo_config": algo_cfg.__dict__,
    }

    try:
        import codecarbon
        metadata["codecarbon_version"] = codecarbon.__version__
    except ImportError:
        logger.error("codecarbon is not installed. Install it with: pip install codecarbon")
        raise

    # ---------------- GPU/CPU guard + thermal gate (before touching the tracker) ----------------
    with GpuCpuGuard(
        min_mhz=exp_cfg.gpu_min_clock_mhz,
        max_mhz=exp_cfg.gpu_max_clock_mhz,
        set_persistence=exp_cfg.set_persistence_mode,
        set_governor=exp_cfg.set_cpu_performance_governor,
    ) if exp_cfg.lock_gpu_clocks else _NullGuard():

        if exp_cfg.thermal_gate_enabled:
            logger.info("Waiting for thermal baseline before starting...")
            gate_info = wait_for_thermal_baseline(
                reference_file=exp_cfg.thermal_gate_reference_file,
                temp_tolerance_c=exp_cfg.thermal_gate_temp_tolerance_c,
                power_tolerance_w=exp_cfg.thermal_gate_power_tolerance_w,
                max_wait_seconds=exp_cfg.thermal_gate_max_wait_seconds,
                poll_interval_seconds=exp_cfg.thermal_gate_poll_interval_seconds,
            )
            logger.info("Thermal gate result: %s", gate_info)
            metadata["thermal_gate"] = gate_info

        # ---------------- settle (unmeasured) ----------------
        logger.info("Settling for %.0fs (unmeasured)...", exp_cfg.settle_seconds)
        time.sleep(exp_cfg.settle_seconds)

        # ---------------- start CodeCarbon tracker for the whole remaining lifecycle ----------------
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(
            project_name=f"{exp_cfg.algo_name}_{exp_cfg.env_id}_seed{exp_cfg.seed}",
            measure_power_secs=exp_cfg.measure_power_secs,
            tracking_mode=exp_cfg.tracking_mode,
            output_dir=str(run_dir),
            output_file="emissions.csv",
            save_to_file=True,
            log_level="warning",
            allow_multiple_runs=True,
        )
        tracker.start()

        energy_log = {}
        try:
            # ---------------- idle baseline (head) ----------------
            logger.info("Recording idle baseline for %.0fs...", exp_cfg.idle_baseline_seconds)
            tracker.start_task("idle_baseline_head")
            time.sleep(exp_cfg.idle_baseline_seconds)
            data = tracker.stop_task("idle_baseline_head")
            energy_log["idle_baseline_head"] = data.energy_consumed if data else 0.0

            # ---------------- warmup + measured training ----------------
            env = _make_env(exp_cfg.env_id, exp_cfg.seed)
            agent, algo_energy_log = _dispatch_train(exp_cfg, algo_cfg, env, tracker, device, logger)
            energy_log.update(algo_energy_log)
            env.close()

            # ---------------- idle baseline (tail) ----------------
            logger.info("Recording idle tail for %.0fs...", exp_cfg.idle_tail_seconds)
            tracker.start_task("idle_baseline_tail")
            time.sleep(exp_cfg.idle_tail_seconds)
            data = tracker.stop_task("idle_baseline_tail")
            energy_log["idle_baseline_tail"] = data.energy_consumed if data else 0.0

        finally:
            total_data = tracker.stop()
            energy_log["_total_kg_co2eq"] = total_data

    # ---------------- write outputs ----------------
    energy_log_serializable = {
        k: (v if not isinstance(v, dict) else v) for k, v in energy_log.items()
    }
    with open(run_dir / "segment_energy.json", "w") as f:
        json.dump(energy_log_serializable, f, indent=2)

    metadata["end_time_utc"] = datetime.utcnow().isoformat()
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Run complete. Segment energy summary (kWh unless noted):")
    for k, v in energy_log.items():
        logger.info("  %s: %s", k, v)

    return run_dir, energy_log


class _NullGuard:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _make_env(env_id: str, seed: int):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def _dispatch_train(exp_cfg: ExperimentConfig, algo_cfg, env, tracker, device, logger):
    if exp_cfg.algo_name == "sac":
        from algorithms.sac import train as sac_train
        return sac_train(env, algo_cfg, exp_cfg, tracker, device, logger,
                          steps_per_epoch=exp_cfg.steps_per_epoch)
    raise ValueError(f"Unknown algo_name '{exp_cfg.algo_name}' -- add a dispatch branch in experiment_runner.py")
