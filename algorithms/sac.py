"""
Soft Actor-Critic (Haarnoja et al., 2018) for continuous control, with
automatic entropy coefficient tuning (Haarnoja et al., 2019).

Segment structure (see train() at the bottom):
  - "rollout": env.step() + action selection + buffer insertion
  - "buffer_sample": drawing a minibatch from the replay buffer
  - "critic_update": twin-Q forward/backward pass + optimizer step
  - "actor_update": policy forward/backward pass + optimizer step (+ alpha update)
  - "target_update": Polyak averaging of target critic weights

These are timed internally with time.perf_counter() at sub-epoch granularity
and reconciled against CodeCarbon's real hardware-measured energy for each
epoch's two top-level tasks ("rollout_epoch{i}", "gradient_updates_epoch{i}").
See experiment_runner.py / README for why this two-level scheme is used
instead of tagging every single gradient step as its own CodeCarbon task.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from algorithms.replay_buffer import ReplayBuffer
from configs.config import SACConfig, ExperimentConfig

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def _mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden_sizes], output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.register_buffer("act_limit", torch.as_tensor(act_limit, dtype=torch.float32))

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)

        if deterministic:
            pre_tanh = mu
        else:
            pre_tanh = dist.rsample()  # reparameterization trick

        action = torch.tanh(pre_tanh)

        if with_logprob:
            logp = dist.log_prob(pre_tanh).sum(dim=-1)
            # tanh squashing correction (SAC paper, appendix C)
            logp -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1)
        else:
            logp = None

        return action * self.act_limit, logp


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
    alpha: Optional[float] = None


class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, cfg: SACConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.act_dim = act_dim

        self.actor = GaussianPolicy(obs_dim, act_dim, cfg.hidden_sizes, act_limit).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.q1_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.q2_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())
        for p in self.q1_targ.parameters():
            p.requires_grad = False
        for p in self.q2_targ.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        self.autotune_alpha = cfg.autotune_alpha
        if cfg.autotune_alpha:
            self.target_entropy = (
                cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
            )
            self.log_alpha = torch.tensor(
                np.log(cfg.init_alpha), dtype=torch.float32, requires_grad=True, device=device
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(cfg.init_alpha), device=device)

        self._update_counter = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
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

        # --- critic_update ---
        t0 = time.perf_counter()
        with torch.no_grad():
            next_action, next_logp = self.actor(next_obs_t)
            q1_next = self.q1_targ(next_obs_t, next_action)
            q2_next = self.q2_targ(next_obs_t, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
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

        # --- actor_update (+ alpha) ---
        do_actor_update = (self._update_counter % self.cfg.policy_update_delay) == 0
        if do_actor_update:
            t0 = time.perf_counter()
            for p in self.q1.parameters():
                p.requires_grad = False
            for p in self.q2.parameters():
                p.requires_grad = False

            action, logp = self.actor(obs_t)
            q1_pi = self.q1(obs_t, action)
            q2_pi = self.q2(obs_t, action)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * logp - q_pi).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            for p in self.q1.parameters():
                p.requires_grad = True
            for p in self.q2.parameters():
                p.requires_grad = True

            if self.autotune_alpha:
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()

            info.actor_update_s = time.perf_counter() - t0
            info.actor_loss = actor_loss.item()
            info.alpha = self.alpha.item()

        # --- target_update ---
        t0 = time.perf_counter()
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)
        info.target_update_s = time.perf_counter() - t0

        self._update_counter += 1
        return info


class _NullTask:
    """No-op context manager used when tracker is None (e.g. dry-run/debug)."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _TrackerTask:
    """Wraps tracker.start_task/stop_task as a context manager and stores the
    returned EmissionsData's energy_consumed (kWh) into a results dict keyed
    by a stable prefix (unique task *names* per call avoid a CodeCarbon bug
    where reusing a task name corrupts internal accounting -- see README)."""

    def __init__(self, tracker, prefix: str, counter: int, energy_log: Dict[str, float]):
        self.tracker = tracker
        self.name = f"{prefix}_{counter}"
        self.prefix = prefix
        self.energy_log = energy_log

    def __enter__(self):
        if self.tracker is not None:
            self.tracker.start_task(self.name)
        return self

    def __exit__(self, *a):
        if self.tracker is not None:
            data = self.tracker.stop_task(self.name)
            self.energy_log[self.prefix] = self.energy_log.get(self.prefix, 0.0) + (
                data.energy_consumed if data is not None else 0.0
            )
        return False


def train(
    env,
    sac_cfg: SACConfig,
    exp_cfg: ExperimentConfig,
    tracker,
    device: torch.device,
    logger,
    steps_per_epoch: int = 1000,
):
    """
    Runs SAC warmup + measured training. Returns a dict of energy_consumed
    (kWh) per segment: rollout, buffer_sample, critic_update, actor_update,
    target_update, plus 'warmup' (kept separate from the measured segments).

    `rollout` and the combined `gradient_updates` bucket are real CodeCarbon
    measurements (one task per epoch, unique-named). `buffer_sample`,
    `critic_update`, `actor_update`, `target_update` are obtained by
    allocating each epoch's `gradient_updates` energy proportionally to the
    wall-clock time share of each sub-segment within that epoch. This is a
    deliberate approximation -- see the module docstring and the thesis
    write-up caveat in README.md.
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_limit, sac_cfg, device)
    buffer = ReplayBuffer(sac_cfg.buffer_capacity, obs_dim, act_dim)

    energy_log: Dict[str, float] = {}
    sub_time_totals = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}

    obs, _ = env.reset(seed=exp_cfg.seed)

    # ---------------- warmup (random policy, fills buffer; excluded from analysis segments) ----------------
    logger.info("Starting warmup: %d steps", exp_cfg.warmup_steps)
    with _TrackerTask(tracker, "warmup", 0, energy_log):
        for _ in range(exp_cfg.warmup_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

    # ---------------- measured training ----------------
    n_epochs = max(1, exp_cfg.train_steps // steps_per_epoch)
    logger.info("Starting measured training: %d epochs x %d steps = %d env steps",
                n_epochs, steps_per_epoch, n_epochs * steps_per_epoch)

    for epoch in range(n_epochs):
        # --- rollout block ---
        with _TrackerTask(tracker, "rollout", epoch, energy_log):
            for _ in range(steps_per_epoch):
                action = agent.select_action(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                buffer.add(obs, action, reward, next_obs, terminated)
                obs = next_obs
                if terminated or truncated:
                    obs, _ = env.reset()

        # --- gradient update block ---
        epoch_sub_times = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
        n_updates = steps_per_epoch * sac_cfg.updates_per_env_step
        with _TrackerTask(tracker, "gradient_updates", epoch, energy_log):
            for _ in range(n_updates):
                if len(buffer) < sac_cfg.batch_size:
                    break
                info = agent.update(buffer, sac_cfg.batch_size)
                epoch_sub_times["buffer_sample"] += info.buffer_sample_s
                epoch_sub_times["critic_update"] += info.critic_update_s
                epoch_sub_times["actor_update"] += info.actor_update_s
                epoch_sub_times["target_update"] += info.target_update_s

        # Sub-segment times accumulate across all epochs and are reconciled against the
        # total measured "gradient_updates" energy once, after the loop (see below) --
        # simpler than a per-epoch split and equally defensible under the constant-power
        # approximation this scheme relies on.
        for k in sub_time_totals:
            sub_time_totals[k] += epoch_sub_times[k]

        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            logger.info("Epoch %d/%d done (buffer size=%d)", epoch + 1, n_epochs, len(buffer))

    # ---------------- reconcile: split total gradient_updates energy by aggregate time share ----------------
    total_gradient_energy = energy_log.pop("gradient_updates", 0.0)
    total_time = sum(sub_time_totals.values())
    for k, t in sub_time_totals.items():
        share = (t / total_time) if total_time > 0 else 0.0
        energy_log[k] = total_gradient_energy * share

    energy_log["_sub_segment_wall_time_seconds"] = sub_time_totals
    return agent, energy_log
