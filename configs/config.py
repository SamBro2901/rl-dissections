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
    gpu_min_clock_mhz: Optional[int] = 2000   # None => use gpu_control.DEFAULT_LOCK_MHZ (2000), clamped to device range
    gpu_max_clock_mhz: Optional[int] = 2000   # None => use gpu_control.DEFAULT_LOCK_MHZ (2000), clamped to device range
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
    force_cpu_power_w: Optional[float] = None  # None => let CodeCarbon read real RAPL energy (Linux); only set
                                                # this to a fixed watts figure as a fallback if RAPL is unreadable

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


@dataclass
class MBPOConfig:
    """
    Model-Based Policy Optimization (Janner, Fu, Zhang & Levine, NeurIPS 2019,
    "When to Trust Your Model: Model-Based Policy Optimization",
    https://arxiv.org/abs/1906.08253), using SAC (Haarnoja et al. 2018/2019)
    as the underlying model-free policy-optimization subroutine, exactly as
    in the paper's reference implementation (https://github.com/JannerM/mbpo).

    Defaults below reproduce the paper's HalfCheetah-v2 configuration
    (Appendix A, Table 2) -- the closest published setting to this harness's
    default `--env HalfCheetah-v5`. Hopper/Walker2d/Ant/Humanoid need a
    longer, scheduled model rollout length and (for Humanoid) a wider model;
    see configs/overrides/mbpo_*.json for the paper's per-env settings and
    pass one via `--algo-config-overrides`.
    """

    # ---- SAC (policy) hyperparameters -- consumed by algorithms.sac.SACAgent,
    # which MBPO reuses unmodified as its policy-optimization subroutine. ----
    hidden_sizes: Tuple[int, int] = (1024, 1024)   # kept consistent with SACConfig's defaults in this repo
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005                # target network Polyak averaging coefficient
    target_entropy: Optional[float] = None  # None => -action_dim (standard heuristic)
    autotune_alpha: bool = True
    init_alpha: float = 0.2
    policy_update_delay: int = 1

    # ---- SAC gradient updates ("G" in the paper). MBPO uses far more updates
    # per env step than vanilla SAC because most of the SAC minibatch is cheap
    # model-generated data, not real env transitions. ----
    updates_per_env_step: int = 1    # paper uses 40 for Hopper specifically
    batch_size: int = 256             # SAC minibatch size, mixed real/model per real_ratio below
    real_ratio: float = 0.05          # fraction of each SAC minibatch drawn from the *real* env buffer

    # ---- real environment replay buffer (same role as SACConfig.buffer_capacity) ----
    buffer_capacity: int = 1_000_000

    # ---- probabilistic dynamics model ensemble (Chua et al. 2018, PETS-style) ----
    ensemble_size: int = 7
    num_elites: int = 5
    model_hidden_sizes: Tuple[int, ...] = (200, 200, 200, 200)   # (400, 400, 400, 400) for Humanoid in the paper
    model_lr: float = 1e-3
    # Per-layer Adam weight decay (one entry per hidden layer + one for the output
    # heads, so len == len(model_hidden_sizes) + 1), matching the PETS/MBPO
    # reference implementation's FC-ensemble default.
    model_weight_decays: Tuple[float, ...] = (2.5e-5, 5e-5, 7.5e-5, 1e-4, 1e-4)
    model_train_batch_size: int = 256
    model_holdout_ratio: float = 0.2   # fraction of each retrain's data held out for early stopping / elite selection
    model_max_train_epochs: int = 200  # cap on passes over the training split per retrain call
    model_train_patience: int = 5      # early-stop a retrain if holdout MSE hasn't improved for this many epochs
    model_train_freq: int = 250        # retrain the ensemble once real env steps since the last retrain reach this
    deterministic_model: bool = False  # False => sample from the predicted Gaussian (paper default); True => use the mean

    # ---- branched model rollouts (the core MBPO mechanism, Sec 3.3) ----
    rollout_batch_size: int = 10_000  # number of real states branched from each time model rollouts are generated
    model_retain_epochs: int = 1       # model buffer capacity is sized to hold only this many epochs' worth of rollout data
    # Rollout length k is linearly scheduled from rollout_min_length to
    # rollout_max_length between epochs rollout_min_epoch and rollout_max_epoch
    # (paper Table 2). Defaults below reproduce HalfCheetah-v2's setting: a
    # fixed length of 1 (no scheduling) -- the paper notes HalfCheetah gets no
    # benefit from longer rollouts. Override for other envs, e.g. Hopper:
    # (20, 100, 1, 15); Walker2d: (20, 150, 1, 15); Ant: (20, 100, 1, 25);
    # Humanoid: (20, 300, 1, 25).
    rollout_min_epoch: int = 20
    rollout_max_epoch: int = 150
    rollout_min_length: int = 1
    rollout_max_length: int = 1


@dataclass
class TD3Config:
    """
    Twin Delayed Deep Deterministic Policy Gradient (Fujimoto, van Hoof &
    Meger, ICML 2018, "Addressing Function Approximation Error in
    Actor-Critic Methods", https://arxiv.org/abs/1802.09477), consumed by
    algorithms/td3.py.

    TD3 is DDPG (Lillicrap et al. 2015) plus three fixes for actor-critic
    overestimation bias (paper Sections 4.2/5.2/5.3, Algorithm 1): clipped
    double Q-learning (twin critics, target = min over both), delayed policy
    updates (actor + both target networks updated once every
    `policy_update_delay` critic updates), and target policy smoothing
    (clipped Gaussian noise added to the target action).

    Defaults below reproduce the paper's own hyperparameters (Section 6.1,
    Table 3), which the paper applies *unmodified* across every MuJoCo-v1
    task in its evaluation, including HalfCheetah-v1 and Ant-v1 -- so unlike
    MBPOConfig, no per-env override file is needed here.

    The one paper-specified setting that *does* vary by environment -- the
    length of the purely-random warmup phase used to fill the buffer before
    training starts -- lives on `ExperimentConfig.warmup_steps` (shared
    across algorithms), not here: the paper uses 10,000 steps for
    HalfCheetah-v1/Ant-v1 ("stable length environments") vs. 1,000 for the
    rest (Section 6.1). Pass `--warmup-steps 10000` on the CLI for
    HalfCheetah-v5/Ant-v5 runs to match.

    hidden_sizes is the one deliberate deviation from the paper (which uses
    (400, 300), Appendix C): kept at (1024, 1024) to match
    SACConfig/MBPOConfig's defaults in this repo, so the network-capacity
    term is held constant across algorithms when comparing energy usage.
    """

    hidden_sizes: Tuple[int, int] = (1024, 1024)   # paper: (400, 300)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005                  # target network Polyak averaging coefficient
    batch_size: int = 100
    buffer_capacity: int = 1_000_000
    exploration_noise: float = 0.1      # std of Gaussian action noise added during rollout, as a fraction of act_limit
    target_policy_noise: float = 0.2    # std of clipped noise added to the target action (target policy smoothing), as a fraction of act_limit
    target_noise_clip: float = 0.5      # clip range for target smoothing noise, as a fraction of act_limit
    policy_update_delay: int = 2        # d in the paper -- actor + both target nets updated once every d critic updates
    updates_per_env_step: int = 1       # gradient steps per environment step, after warmup ("iterations per time step" in Table 3)


@dataclass
class TDMPC2Config:
    """
    TD-MPC2 (Hansen, Su & Wang, ICLR 2024, "TD-MPC2: Scalable, Robust World
    Models for Continuous Control", https://arxiv.org/abs/2310.16828),
    consumed by algorithms/tdmpc2.py.

    TD-MPC2 learns a decoder-free latent world model (encoder, latent
    dynamics, reward model, a 5-network Q-ensemble, and a Gaussian policy
    prior) and selects actions via MPPI trajectory optimization in latent
    space (Section 3, Algorithm 1) rather than by directly querying the
    policy -- planning dominates its per-step compute, unlike the other
    (model-free) algorithms in this repo.

    Defaults below are the paper's own single-task, state-observation
    hyperparameters -- i.e. the values in the authors' reference
    implementation's `config.yaml` (https://github.com/nicklashansen/tdmpc2),
    which the paper reports using *unmodified* across all 104 evaluated
    single-task environments. Two values are nonetheless environment-shaped
    and are computed from `episode_length` rather than hardcoded here (see
    `algorithms/tdmpc2.py`'s `TDMPC2Agent._get_discount`):
      - discount = clip((episode_length/discount_denom - 1) / (episode_length
        /discount_denom), discount_min, discount_max). For HalfCheetah-v5 and
        Ant-v5 (both 1000-step episodes by default), this evaluates to 0.995.
      - the paper's seed-steps heuristic is max(5 * episode_length, 1000) =
        5000 for 1000-step episodes -- which is exactly this harness's
        default `ExperimentConfig.warmup_steps`, so no override is needed
        for HalfCheetah-v5/Ant-v5.

    `episodic` controls whether an auxiliary termination classifier is
    learned and used to truncate imagined MPPI rollouts (paper Section 3,
    "Handling termination"). It defaults to False (matching the paper's
    default for fixed-length continuous-control tasks) since HalfCheetah-v5
    never terminates early; `configs/overrides/tdmpc2_ant.json` sets it to
    True for Ant-v5, since Gymnasium's default `terminate_when_unhealthy=True`
    means Ant *can* end an episode early, and the planner should account for
    that when imagining trajectories (the real TD-target already bootstraps
    correctly off the env's own termination signal regardless of this flag --
    see `TDMPC2Agent.update`'s `td_targets` computation).
    """

    # ---- architecture (paper's 5M-parameter state-based default) ----
    num_enc_layers: int = 2
    enc_dim: int = 256
    mlp_dim: int = 512
    latent_dim: int = 512
    num_q: int = 5
    dropout: float = 0.01
    simnorm_dim: int = 8

    # ---- optimization ----
    lr: float = 3e-4
    enc_lr_scale: float = 0.3      # encoder LR = lr * enc_lr_scale
    grad_clip_norm: float = 20.0
    batch_size: int = 256
    tau: float = 0.01              # target Q-ensemble Polyak coefficient; also RunningScale's EMA rate
    rho: float = 0.5               # temporal loss-weighting decay (rho^t) across the planning horizon
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1.0  # only used when episodic=True
    entropy_coef: float = 1e-4
    updates_per_env_step: int = 1  # gradient steps per environment step, after the pretraining burst

    # ---- discount heuristic (see class docstring) ----
    discount_denom: float = 5.0
    discount_min: float = 0.95
    discount_max: float = 0.995
    episode_length: int = 1000     # fallback only; algorithms/tdmpc2.py prefers env.spec.max_episode_steps

    # ---- MPPI planning (Algorithm 1) ----
    horizon: int = 3
    iterations: int = 6            # +2 automatically for action_dim >= 20 (paper heuristic; irrelevant for
                                    # HalfCheetah-v5 (6) / Ant-v5 (8))
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24         # of num_samples, this many trajectories are seeded from the policy prior
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5

    # ---- Gaussian policy prior ----
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    # ---- discrete regression (two-hot, symlog-transformed) for reward/value prediction ----
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    # ---- episodic termination model (see class docstring) ----
    episodic: bool = False

    # ---- sequence replay buffer ----
    buffer_capacity: int = 1_000_000


# Maps --algo name -> its hyperparameter config dataclass. run_experiment.py
# uses this to pick the right config instead of hardcoding each algorithm's
# hyperparameters as CLI flags. Add an entry here when adding a new algorithm
# (PPO, PETS, ...); no changes to run_experiment.py are needed.
ALGO_CONFIGS = {
    "sac": SACConfig,
    "mbpo": MBPOConfig,
    "td3": TD3Config,
    "tdmpc2": TDMPC2Config,
}
