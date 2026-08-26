"""
Central configuration objects.

Keeping experiment-protocol config (timing, GPU locking, thermal gating)
separate from algorithm hyperparameters (SAC config) so that adding a new
algorithm later (PPO, MBPO, PETS) only means adding a new dataclass here,
not touching the runner.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ExperimentConfig:
    """Controls the timing/phase structure of a single run, independent of algorithm."""

    algo_name: str
    env_id: str
    seed: int = 0

    # --- phase durations / sizes ---
    settle_seconds: float = 30.0          # unmeasured pre-run settle time
    idle_baseline_seconds: float = 90.0   # measured idle baseline (head)
    idle_tail_seconds: float = 90.0       # measured idle baseline (tail)
    warmup_steps: int = 5_000             # env steps, excluded from "measured" analysis segment
    train_steps: int = 100_000            # env steps in the measured training segment
    steps_per_epoch: int = 1000           # env steps per CodeCarbon-tagged rollout/gradient_updates task pair

    # --- GPU / CPU control ---
    lock_gpu_clocks: bool = True
    gpu_min_clock_mhz: Optional[int] = None   # None => query device and use its min
    gpu_max_clock_mhz: Optional[int] = None   # None => query device and use a fixed low-variance value
    set_persistence_mode: bool = True
    set_cpu_performance_governor: bool = True

    # --- thermal gating between repeated runs ---
    thermal_gate_enabled: bool = True
    thermal_gate_temp_tolerance_c: float = 2.0
    thermal_gate_power_tolerance_w: float = 5.0
    thermal_gate_max_wait_seconds: float = 600.0
    thermal_gate_poll_interval_seconds: float = 5.0
    thermal_gate_reference_file: str = "results/_thermal_reference.json"

    # --- codecarbon ---
    measure_power_secs: float = 1.0
    tracking_mode: str = "machine"  # "machine" recommended for dedicated single-purpose PC
    output_dir: str = "results"
    country_iso_code: str = "DEU"  # adjust to your grid region; affects CO2e conversion, not energy (kWh)
    force_cpu_power_w: Optional[float] = 125.0  # Core Ultra 9 285K base TDP; codecarbon's table doesn't know this CPU yet

    # --- misc ---
    device: str = "cuda"  # falls back to cpu automatically if unavailable


@dataclass
class SACConfig:
    """SAC hyperparameters (Haarnoja et al. 2018, automatic entropy tuning variant)."""

    hidden_sizes: Tuple[int, int] = (1024, 1024)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005            # target network Polyak averaging coefficient
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    target_entropy: Optional[float] = None  # None => -action_dim (standard heuristic)
    autotune_alpha: bool = True
    init_alpha: float = 0.2
    updates_per_env_step: int = 1  # gradient steps per environment step, after warmup
    policy_update_delay: int = 1   # e.g. set to 2 for TD3-style delayed actor updates (kept at 1 for vanilla SAC)


# Maps --algo name -> its hyperparameter config dataclass. run_experiment.py
# uses this to pick the right config instead of hardcoding each algorithm's
# hyperparameters as CLI flags. Add an entry here when adding a new algorithm
# (PPO, MBPO, PETS, ...); no changes to run_experiment.py are needed.
ALGO_CONFIGS = {
    "sac": SACConfig,
}
