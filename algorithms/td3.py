"""
Twin Delayed Deep Deterministic Policy Gradient (Fujimoto, van Hoof & Meger,
ICML 2018, "Addressing Function Approximation Error in Actor-Critic
Methods", https://arxiv.org/abs/1802.09477).

TD3 is DDPG (Lillicrap et al., 2015) with three fixes for actor-critic
overestimation bias (paper Sections 4.2, 5.2, 5.3; Algorithm 1):
  1. Clipped double Q-learning: twin critics, target uses min(Q1_targ, Q2_targ).
  2. Delayed policy updates: the actor and *both* target networks are only
     updated once every `policy_update_delay` critic updates.
  3. Target policy smoothing: clipped Gaussian noise is added to the target
     action before it is fed to the target critics, regularizing the target
     against sharp, easily-overfit Q-value peaks.

Segment structure (see train() at the bottom), matching sac.py:
  - "rollout": env.step() + action selection (+ exploration noise) + buffer insertion
  - "buffer_sample": drawing a minibatch from the replay buffer
  - "critic_update": twin-Q forward/backward pass + optimizer step
  - "actor_update": policy forward/backward pass + optimizer step (delayed)
  - "target_update": Polyak averaging of target actor + target critic weights
    (delayed, tied to the same schedule as actor_update -- see Algorithm 1)

These are timed internally with time.perf_counter() at sub-epoch granularity
and reconciled against CodeCarbon's real hardware-measured energy for each
epoch's two top-level tasks ("rollout_epoch{i}", "gradient_updates_epoch{i}"),
exactly as in sac.py -- see that module's docstring and the README for the
allocation scheme this relies on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.replay_buffer import ReplayBuffer
from algorithms.tracker_utils import TrackerTask
from configs.config import TD3Config, ExperimentConfig


def _mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden_sizes, act_dim], output_activation=nn.Tanh)
        self.register_buffer("act_limit", torch.as_tensor(act_limit, dtype=torch.float32))

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.net = _mlp([obs_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


@dataclass
class UpdateInfo:
    """Wall-clock time (seconds) spent in each sub-segment of one call to update()."""
    buffer_sample_s: float = 0.0
    critic_update_s: float = 0.0
    actor_update_s: float = 0.0
    target_update_s: float = 0.0
    critic_loss: Optional[float] = None
    actor_loss: Optional[float] = None


class TD3Agent:
    def __init__(self, obs_dim, act_dim, act_limit, cfg: TD3Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.actor = DeterministicPolicy(obs_dim, act_dim, cfg.hidden_sizes, act_limit).to(device)
        self.actor_targ = DeterministicPolicy(obs_dim, act_dim, cfg.hidden_sizes, act_limit).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.q1 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.q1_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.q2_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        for p in self.actor_targ.parameters():
            p.requires_grad = False
        for p in self.q1_targ.parameters():
            p.requires_grad = False
        for p in self.q2_targ.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        # Target policy smoothing noise std/clip, scaled from the paper's
        # normalized ([0, 1] fraction of act_limit) values -- matches the
        # reference implementation, which pre-scales these by max_action
        # before constructing the agent (main.py: `policy_noise * max_action`).
        self.target_policy_noise = cfg.target_policy_noise * act_limit
        self.target_noise_clip = cfg.target_noise_clip * act_limit

        self._update_counter = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action from the current policy (no exploration noise --
        the caller adds that, since it depends on the rollout phase, not the agent)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_t)
        return action.squeeze(0).cpu().numpy()

    def update(self, buffer: ReplayBuffer, batch_size: int) -> UpdateInfo:
        info = UpdateInfo()

        # --- buffer_sample ---
        t0 = time.perf_counter()
        obs, act, rew, next_obs, done = buffer.sample(batch_size)
        obs_t = torch.as_tensor(obs, device=self.device)
        act_t = torch.as_tensor(act, device=self.device)
        rew_t = torch.as_tensor(rew, device=self.device).squeeze(-1)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        done_t = torch.as_tensor(done, device=self.device).squeeze(-1)
        info.buffer_sample_s = time.perf_counter() - t0

        # --- critic_update (clipped double Q-learning + target policy smoothing) ---
        t0 = time.perf_counter()
        with torch.no_grad():
            noise = (torch.randn_like(act_t) * self.target_policy_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_action = (self.actor_targ(next_obs_t) + noise).clamp(-self.act_limit, self.act_limit)
            q1_next = self.q1_targ(next_obs_t, next_action)
            q2_next = self.q2_targ(next_obs_t, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = rew_t + (1.0 - done_t) * self.cfg.gamma * q_next

        q1_pred = self.q1(obs_t, act_t)
        q2_pred = self.q2(obs_t, act_t)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()
        info.critic_update_s = time.perf_counter() - t0
        info.critic_loss = (q1_loss.item() + q2_loss.item()) / 2.0

        # --- actor_update + target_update (delayed, Algorithm 1: both gated on t mod d) ---
        do_delayed_update = (self._update_counter % self.cfg.policy_update_delay) == 0
        if do_delayed_update:
            t0 = time.perf_counter()
            for p in self.q1.parameters():
                p.requires_grad = False

            actor_loss = -self.q1(obs_t, self.actor(obs_t)).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            for p in self.q1.parameters():
                p.requires_grad = True

            info.actor_update_s = time.perf_counter() - t0
            info.actor_loss = actor_loss.item()

            t0 = time.perf_counter()
            with torch.no_grad():
                for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau)
                    p_targ.data.add_(self.cfg.tau * p.data)
                for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau)
                    p_targ.data.add_(self.cfg.tau * p.data)
                for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau)
                    p_targ.data.add_(self.cfg.tau * p.data)
            info.target_update_s = time.perf_counter() - t0

        self._update_counter += 1
        return info


def train(
    env,
    td3_cfg: TD3Config,
    exp_cfg: ExperimentConfig,
    tracker,
    device: torch.device,
    logger,
    steps_per_epoch: int = 1000,
):
    """
    Runs TD3 warmup + measured training. Returns:
      - agent: the trained TD3Agent
      - energy_log: dict of energy_consumed (kWh) per segment: rollout,
        buffer_sample, critic_update, actor_update, target_update, plus
        'warmup' (kept separate from the measured segments).
      - metrics: dict with "episodes" and "epochs" lists, same schema as
        sac.py's train() (see that module's docstring for field details;
        TD3 has no alpha/entropy term so "alpha_end" is simply omitted here).

    Warmup uses a purely random policy to fill the buffer, matching the
    paper (Section 6.1): 10,000 steps for HalfCheetah-v1/Ant-v1, 1,000 for
    other envs -- pass the right `--warmup-steps` on the CLI (this harness's
    `ExperimentConfig.warmup_steps` is shared across algorithms, so it isn't
    baked into TD3Config). After warmup, actions are the deterministic
    policy plus N(0, exploration_noise * act_limit) Gaussian noise, clipped
    to the action bounds (paper Section 6.1; Algorithm 1).
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3Agent(obs_dim, act_dim, act_limit, td3_cfg, device)
    buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, act_dim)

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

    def _explore_action(obs):
        action = agent.select_action(obs)
        noise = np.random.normal(0.0, td3_cfg.exploration_noise * act_limit, size=act_dim)
        return np.clip(action + noise, -act_limit, act_limit)

    obs, _ = env.reset(seed=exp_cfg.seed)

    # ---------------- warmup (random policy, fills buffer; excluded from analysis segments) ----------------
    logger.info("Starting warmup: %d steps", exp_cfg.warmup_steps)
    with TrackerTask(tracker, "warmup", 0, energy_log):
        for warmup_step in range(exp_cfg.warmup_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)
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

    for epoch in range(n_epochs):
        # --- rollout block ---
        epoch_reward_sum = 0.0
        epoch_episode_returns = []
        with TrackerTask(tracker, "rollout", epoch, energy_log):
            for _ in range(steps_per_epoch):
                action = _explore_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                buffer.add(obs, action, reward, next_obs, terminated)
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

        # --- gradient update block ---
        epoch_sub_times = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
        critic_losses, actor_losses = [], []
        n_updates = steps_per_epoch * td3_cfg.updates_per_env_step
        with TrackerTask(tracker, "gradient_updates", epoch, energy_log):
            for _ in range(n_updates):
                if len(buffer) < td3_cfg.batch_size:
                    break
                info = agent.update(buffer, td3_cfg.batch_size)
                epoch_sub_times["buffer_sample"] += info.buffer_sample_s
                epoch_sub_times["critic_update"] += info.critic_update_s
                epoch_sub_times["actor_update"] += info.actor_update_s
                epoch_sub_times["target_update"] += info.target_update_s
                if info.critic_loss is not None:
                    critic_losses.append(info.critic_loss)
                if info.actor_loss is not None:
                    actor_losses.append(info.actor_loss)

        # Sub-segment times accumulate across all epochs and are reconciled against the
        # total measured "gradient_updates" energy once, after the loop (see below) --
        # simpler than a per-epoch split and equally defensible under the constant-power
        # approximation this scheme relies on.
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
            "buffer_size": len(buffer),
        })

        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            logger.info(
                "Epoch %d/%d done (buffer size=%d, cumulative_reward=%.2f, mean_episode_return=%s)",
                epoch + 1, n_epochs, len(buffer), cumulative_reward,
                epochs_log[-1]["mean_episode_return"],
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
