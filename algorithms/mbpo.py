"""
Model-Based Policy Optimization (Janner, Fu, Zhang & Levine, 2019, "When to
Trust Your Model: Model-Based Policy Optimization", NeurIPS 2019).

MBPO trains a probabilistic ensemble dynamics model (algorithms/dynamics_model.py)
on real environment transitions, uses it to generate short "branched" rollouts
starting from real states (Section 3.3), and trains SAC (Haarnoja et al. 2018)
on a mix of real and model-generated transitions. This module reuses
`SACAgent` from algorithms/sac.py completely unmodified as the policy
optimizer -- the only MBPO-specific addition on the policy-learning side is
`_MixedReplayBuffer`, which presents the same `.sample()` interface as
`ReplayBuffer` so `SACAgent.update()` doesn't need to know its data is mixed.

Segment structure (see train() at the bottom), matching the naming used in
README.md's "Extending to PPO / MBPO / PETS" section:
  - "rollout": env.step() + action selection + real buffer insertion (identical to SAC)
  - "dynamics_model_update": retraining the ensemble on the real buffer
  - "synthetic_rollout_generation": branched model rollouts filling the model buffer
  - "gradient_updates": SAC updates on mixed real/model batches, sub-timed into
    "buffer_sample" / "critic_update" / "actor_update" / "target_update" exactly
    as in algorithms/sac.py (see that module's docstring and README.md for why
    this sub-split is a time-proportional allocation, not an independent
    hardware measurement).

Unlike SAC, "dynamics_model_update" and "synthetic_rollout_generation" are
directly-measured CodeCarbon tasks in their own right (not time-allocated),
since they are algorithmically and often computationally distinct enough from
"gradient_updates" that a shared-power approximation between them would be
weaker than the one already used for the four SAC sub-segments.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch

from algorithms.dynamics_model import EnsembleDynamicsModel
from algorithms.replay_buffer import ReplayBuffer
from algorithms.sac import SACAgent
from algorithms.termination_fns import get_termination_fn
from algorithms.tracker_utils import TrackerTask
from configs.config import MBPOConfig, ExperimentConfig


class _MixedReplayBuffer:
    """Presents a `.sample(batch_size)` / `__len__` interface identical to
    ReplayBuffer, drawing `real_ratio` of each minibatch from the real
    environment buffer and the remainder from the model (branched-rollout)
    buffer -- the mechanism (Janner et al. 2019, Sec 3.3) that lets
    algorithms/sac.py's SACAgent.update() be reused completely unmodified."""

    def __init__(self, real_buffer: ReplayBuffer, model_buffer: ReplayBuffer, real_ratio: float):
        self.real_buffer = real_buffer
        self.model_buffer = model_buffer
        self.real_ratio = real_ratio

    def __len__(self):
        return len(self.real_buffer) + len(self.model_buffer)

    def sample(self, batch_size: int):
        if len(self.model_buffer) == 0:
            # Before the first model rollout batch exists (start of training),
            # fall back to pure real data rather than starving the SAC update.
            return self.real_buffer.sample(batch_size)

        n_real = min(max(int(round(batch_size * self.real_ratio)), 0), batch_size)
        n_model = batch_size - n_real

        real = self.real_buffer.sample(n_real) if n_real > 0 else None
        model = self.model_buffer.sample(n_model) if n_model > 0 else None

        if real is None:
            return model
        if model is None:
            return real
        return tuple(np.concatenate([r, m], axis=0) for r, m in zip(real, model))


def _rollout_length(epoch: int, cfg: MBPOConfig) -> int:
    """Linear schedule for the branched model-rollout length k, per Janner et
    al. 2019 Table 2 / Appendix A."""
    if epoch <= cfg.rollout_min_epoch:
        return cfg.rollout_min_length
    if epoch >= cfg.rollout_max_epoch:
        return cfg.rollout_max_length
    span = max(1, cfg.rollout_max_epoch - cfg.rollout_min_epoch)
    frac = (epoch - cfg.rollout_min_epoch) / span
    length = cfg.rollout_min_length + frac * (cfg.rollout_max_length - cfg.rollout_min_length)
    return int(round(length))


def _generate_model_rollouts(
    agent: SACAgent,
    model: EnsembleDynamicsModel,
    real_buffer: ReplayBuffer,
    model_buffer: ReplayBuffer,
    cfg: MBPOConfig,
    rollout_length: int,
    device: torch.device,
    termination_fn,
) -> int:
    """Branches `rollout_batch_size` k-step imagined trajectories off real
    states sampled from `real_buffer`, using the current policy to act and
    the dynamics ensemble to predict transitions, and stores every generated
    transition in `model_buffer` (Janner et al. 2019, Algorithm 1 lines 6-9).
    Each partial trajectory stops early once its termination function fires.
    Returns the number of synthetic transitions generated."""
    n_start = min(cfg.rollout_batch_size, len(real_buffer))
    if n_start == 0:
        return 0

    idx = np.random.randint(0, len(real_buffer), size=n_start)
    obs = real_buffer.obs[idx].copy()
    n_generated = 0

    for _ in range(rollout_length):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            act_t, _ = agent.actor(obs_t, deterministic=False, with_logprob=False)
            next_obs_t, rew_t = model.predict(obs_t, act_t)

        act = act_t.cpu().numpy()
        next_obs = next_obs_t.cpu().numpy()
        rew = rew_t.cpu().numpy()
        done = termination_fn(next_obs)

        model_buffer.add_batch(obs, act, rew, next_obs, done)
        n_generated += obs.shape[0]

        keep = ~done
        if not np.any(keep):
            break
        obs = next_obs[keep]

    return n_generated


def train(
    env,
    mbpo_cfg: MBPOConfig,
    exp_cfg: ExperimentConfig,
    tracker,
    device: torch.device,
    logger,
    steps_per_epoch: int = 1000,
):
    """
    Runs MBPO warmup + measured training. Returns (agent, energy_log, metrics)
    with the same shapes as algorithms.sac.train() (see that function's
    docstring), plus MBPO-specific fields on each "epochs" row: rollout_length,
    model_buffer_size, model_holdout_mse, model_train_epochs.

    `rollout`, `dynamics_model_update`, `synthetic_rollout_generation`, and
    the combined `gradient_updates` bucket are real CodeCarbon measurements
    (one task per epoch, unique-named, per the pattern in algorithms/sac.py
    and README.md). `buffer_sample`, `critic_update`, `actor_update`,
    `target_update` are obtained by allocating `gradient_updates`' energy
    proportionally to wall-clock time share, exactly as in SAC.
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_limit, mbpo_cfg, device)
    model = EnsembleDynamicsModel(obs_dim, act_dim, mbpo_cfg, device)
    termination_fn = get_termination_fn(exp_cfg.env_id, logger)

    real_buffer = ReplayBuffer(mbpo_cfg.buffer_capacity, obs_dim, act_dim)
    # Sized to hold `model_retain_epochs` epochs' worth of rollouts at the
    # longest scheduled rollout length; a plain circular buffer then
    # naturally approximates "retain only the most recent N epochs of model
    # data" (Janner et al. 2019 Sec 3.3) as older entries get overwritten.
    model_buffer_capacity = max(
        10_000, mbpo_cfg.rollout_batch_size * mbpo_cfg.rollout_max_length * mbpo_cfg.model_retain_epochs
    )
    model_buffer = ReplayBuffer(model_buffer_capacity, obs_dim, act_dim)
    mixed_buffer = _MixedReplayBuffer(real_buffer, model_buffer, mbpo_cfg.real_ratio)

    energy_log: Dict[str, float] = {}
    sub_time_totals = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}

    episodes_log = []
    episode_idx = 0
    episode_return = 0.0
    episode_length = 0

    def _record_episode(phase, epoch, env_step):
        nonlocal episode_idx, episode_return, episode_length
        episodes_log.append({
            "episode_idx": episode_idx,
            "phase": phase,
            "epoch": epoch,
            "env_step": env_step,
            "return": episode_return,
            "length": episode_length,
        })
        episode_idx += 1
        episode_return = 0.0
        episode_length = 0

    obs, _ = env.reset(seed=exp_cfg.seed)

    # ---------------- warmup (random policy, fills real buffer; excluded from analysis segments) ----------------
    logger.info("Starting warmup: %d steps", exp_cfg.warmup_steps)
    with TrackerTask(tracker, "warmup", 0, energy_log):
        for warmup_step in range(exp_cfg.warmup_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            real_buffer.add(obs, action, reward, next_obs, terminated)
            episode_return += reward
            episode_length += 1
            obs = next_obs
            if terminated or truncated:
                _record_episode("warmup", None, warmup_step + 1)
                obs, _ = env.reset()

    # ---------------- measured training ----------------
    n_epochs = max(1, exp_cfg.train_steps // steps_per_epoch)
    logger.info("Starting measured training: %d epochs x %d steps = %d env steps",
                n_epochs, steps_per_epoch, n_epochs * steps_per_epoch)

    epochs_log = []
    cumulative_reward = 0.0
    global_step = 0
    last_model_train_step = -mbpo_cfg.model_train_freq  # forces a retrain before the first epoch's rollouts

    for epoch in range(n_epochs):
        # --- dynamics_model_update: retrain the ensemble on all real data seen so far ---
        model_diag = {"holdout_mse": None, "train_epochs": None}
        if (global_step - last_model_train_step) >= mbpo_cfg.model_train_freq:
            with TrackerTask(tracker, "dynamics_model_update", epoch, energy_log):
                model_diag = model.fit(real_buffer)
            last_model_train_step = global_step

        # --- rollout block (real env; identical in structure to SAC) ---
        epoch_reward_sum = 0.0
        epoch_episode_returns = []
        with TrackerTask(tracker, "rollout", epoch, energy_log):
            for _ in range(steps_per_epoch):
                action = agent.select_action(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                real_buffer.add(obs, action, reward, next_obs, terminated)
                episode_return += reward
                episode_length += 1
                cumulative_reward += reward
                epoch_reward_sum += reward
                global_step += 1
                obs = next_obs
                if terminated or truncated:
                    epoch_episode_returns.append(episode_return)
                    _record_episode("train", epoch, global_step)
                    obs, _ = env.reset()

        # --- synthetic_rollout_generation: branched model rollouts ---
        rollout_length = _rollout_length(epoch, mbpo_cfg)
        with TrackerTask(tracker, "synthetic_rollout_generation", epoch, energy_log):
            n_synthetic = _generate_model_rollouts(
                agent, model, real_buffer, model_buffer, mbpo_cfg, rollout_length, device, termination_fn
            )

        # --- gradient_updates block (SAC updates on mixed real/model batches) ---
        epoch_sub_times = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
        critic_losses, actor_losses, alphas = [], [], []
        n_updates = steps_per_epoch * mbpo_cfg.updates_per_env_step
        with TrackerTask(tracker, "gradient_updates", epoch, energy_log):
            for _ in range(n_updates):
                if len(mixed_buffer) < mbpo_cfg.batch_size:
                    break
                info = agent.update(mixed_buffer, mbpo_cfg.batch_size)
                epoch_sub_times["buffer_sample"] += info.buffer_sample_s
                epoch_sub_times["critic_update"] += info.critic_update_s
                epoch_sub_times["actor_update"] += info.actor_update_s
                epoch_sub_times["target_update"] += info.target_update_s
                if info.critic_loss is not None:
                    critic_losses.append(info.critic_loss)
                if info.actor_loss is not None:
                    actor_losses.append(info.actor_loss)
                if info.alpha is not None:
                    alphas.append(info.alpha)

        # Sub-segment times accumulate across all epochs and are reconciled against the
        # total measured "gradient_updates" energy once, after the loop (see below) --
        # identical scheme to algorithms/sac.py.
        for k in sub_time_totals:
            sub_time_totals[k] += epoch_sub_times[k]

        epochs_log.append({
            "epoch": epoch,
            "env_step": global_step,
            "cumulative_reward": cumulative_reward,
            "epoch_reward_sum": epoch_reward_sum,
            "epoch_reward_mean_per_step": epoch_reward_sum / steps_per_epoch,
            "num_episodes_completed": len(epoch_episode_returns),
            "mean_episode_return": (
                sum(epoch_episode_returns) / len(epoch_episode_returns)
                if epoch_episode_returns else None
            ),
            "critic_loss_mean": (sum(critic_losses) / len(critic_losses)) if critic_losses else None,
            "actor_loss_mean": (sum(actor_losses) / len(actor_losses)) if actor_losses else None,
            "alpha_end": alphas[-1] if alphas else None,
            "buffer_size": len(real_buffer),
            "rollout_length": rollout_length,
            "synthetic_transitions_generated": n_synthetic,
            "model_buffer_size": len(model_buffer),
            "model_holdout_mse": model_diag["holdout_mse"],
            "model_train_epochs": model_diag["train_epochs"],
        })

        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            logger.info(
                "Epoch %d/%d done (real_buffer=%d, model_buffer=%d, rollout_len=%d, "
                "cumulative_reward=%.2f, mean_episode_return=%s)",
                epoch + 1, n_epochs, len(real_buffer), len(model_buffer), rollout_length,
                cumulative_reward, epochs_log[-1]["mean_episode_return"],
            )

    # record a trailing partial episode (if training ended mid-episode) so no reward is lost
    if episode_length > 0:
        _record_episode("train_incomplete", n_epochs - 1, global_step)

    # ---------------- reconcile: split total gradient_updates energy by aggregate time share ----------------
    total_gradient_energy = energy_log.pop("gradient_updates", 0.0)
    total_time = sum(sub_time_totals.values())
    for k, t in sub_time_totals.items():
        share = (t / total_time) if total_time > 0 else 0.0
        energy_log[k] = total_gradient_energy * share

    energy_log["_sub_segment_wall_time_seconds"] = sub_time_totals
    metrics = {"episodes": episodes_log, "epochs": epochs_log}
    return agent, energy_log, metrics
