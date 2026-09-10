"""
TD-MPC2 (Hansen, Su & Wang, ICLR 2024, "TD-MPC2: Scalable, Robust World
Models for Continuous Control", https://arxiv.org/abs/2310.16828).

TD-MPC2 learns a decoder-free latent world model (encoder + latent dynamics
+ reward model + Q-ensemble + a Gaussian policy prior used to seed planning)
and selects actions via MPPI trajectory optimization in latent space rather
than by directly querying the policy. This module is a single-task,
state-observation port of the authors' reference implementation
(https://github.com/nicklashansen/tdmpc2), adapted to fit this harness's
env/config/TrackerTask conventions (see algorithms/sac.py, algorithms/td3.py)
-- the numerics of every loss, the MPPI planner, and the default
hyperparameters (configs/config.py TDMPC2Config) match the reference
`tdmpc2/tdmpc2.py`, `common/world_model.py`, `common/math.py`,
`common/layers.py` and `config.yaml` as of the ICLR 2024 release.

Key building blocks, ported from the reference source:
  - SimNorm (simplicial normalization, https://arxiv.org/abs/2204.00616):
    stabilizes the latent space by softmax-normalizing groups of `simnorm_dim`
    units, used as the output activation of the encoder and the dynamics net.
  - Discrete regression for reward/value (`two_hot`/`two_hot_inv`/`soft_ce`):
    rewards and Q-values are predicted as a soft two-hot distribution over
    `num_bins` bins in a symlog-transformed space and trained with
    cross-entropy, rather than direct MSE regression.
  - The "consistency loss": only the FIRST latent state in a training
    sequence is obtained by encoding a real observation; every subsequent
    latent is produced by unrolling the *learned* dynamics model, and is
    pulled (MSE, stop-gradient on the target) toward the encoding of the
    corresponding real next-observation. This is what makes the latent
    dynamics model usable for multi-step planning.
  - MPPI planning (`TDMPC2Agent.act` / `_plan`): at every environment step,
    `num_samples` action sequences (a fraction seeded by the policy prior,
    the rest by the current Gaussian search distribution) are rolled out
    through the *imagined* latent dynamics, scored by reward-model rollout +
    terminal Q-value, and the top `num_elites` are used to re-fit the search
    distribution's mean/std (`iterations` times) before one action is sampled
    from it. Unlike SAC/TD3/MBPO in this repo, action *selection* itself is
    the dominant compute cost here, not the gradient updates -- worth calling
    out explicitly when comparing energy profiles across algorithms.

Deliberate adaptations to this harness (documented so results aren't
mis-read against the paper):
  - The reference's `OnlineTrainer` interleaves one env step with one
    gradient update strictly in lockstep, and does a one-off burst of
    `seed_steps` gradient updates the instant warmup ends ("pretraining on
    seed data"). This module keeps that pretraining burst (as its own
    directly-measured segment, "world_model_pretrain") but -- like every
    other algorithm in this repo -- separates each epoch's rollout from its
    `steps_per_epoch * updates_per_env_step` gradient updates so the two can
    be tagged as independent CodeCarbon tasks (see algorithms/sac.py's
    docstring and README.md's "Why epoch-level tagging" section).
  - No multi-task machinery (task embeddings/masks), no pixel-observation
    encoder, no `torch.compile`/vmap ensemble tricks: the Q-ensemble is a
    plain `nn.ModuleList` of independent MLPs evaluated in a Python loop.
    Mathematically identical to the reference's vmapped ensemble; just not
    fused into one kernel. Fine at the batch sizes used here.
  - Q-parameter freezing during the policy-loss backward pass uses this
    repo's existing `requires_grad = False` / restore pattern (see
    algorithms/td3.py) instead of the reference's separate
    `TensorDictParams`-based "detached" parameter view -- same effect
    (policy gradients flow through z/action into the policy net only, never
    into the Q-network weights), simpler to read.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.tracker_utils import TrackerTask
from configs.config import TDMPC2Config, ExperimentConfig

LOG_PROB_CONST = 0.9189385175704956  # 0.5 * log(2*pi)


# ------------------------------------------------------------------------
# Math utilities (ported from tdmpc2/common/math.py)
# ------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x: torch.Tensor, vmin: float, vmax: float, num_bins: int, bin_size: float) -> torch.Tensor:
    """Converts a batch of scalars (shape (B, 1)) to soft two-hot encoded
    targets (shape (B, num_bins)) for discrete regression, in symlog space."""
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(-1)
    bin_idx = torch.floor((x - vmin) / bin_size)
    bin_offset = ((x - vmin) / bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x: torch.Tensor, vmin: float, vmax: float, num_bins: int) -> torch.Tensor:
    """Converts (soft) two-hot / discrete-regression logits back to a scalar
    expectation, mapped back out of symlog space."""
    bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
    probs = F.softmax(x, dim=-1)
    out = torch.sum(probs * bins, dim=-1, keepdim=True)
    return symexp(out)


def soft_ce(pred: torch.Tensor, target: torch.Tensor, vmin: float, vmax: float, num_bins: int, bin_size: float) -> torch.Tensor:
    """Cross-entropy between predicted bin logits and a soft two-hot target."""
    logp = F.log_softmax(pred, dim=-1)
    target_soft = two_hot(target, vmin, vmax, num_bins, bin_size)
    return -(target_soft * logp).sum(-1, keepdim=True)


def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    residual = -0.5 * eps.pow(2) - log_std
    return (residual - LOG_PROB_CONST).sum(-1, keepdim=True)


def squash(mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor):
    """tanh-squashes the mean and sampled action, correcting the log-prob."""
    mu = torch.tanh(mu)
    pi_t = torch.tanh(pi)
    squashed = torch.log(F.relu(1 - pi_t.pow(2)) + 1e-6)
    log_pi = log_pi - squashed.sum(-1, keepdim=True)
    return mu, pi_t, log_pi


def gumbel_softmax_sample(p: torch.Tensor, temperature: float = 1.0, dim: int = 0) -> torch.Tensor:
    logits = p.log()
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1).reshape(-1)


class RunningScale:
    """Running trimmed (5th-95th percentile) scale estimator used to
    normalize Q-values before combining them with the entropy bonus in the
    policy loss (tdmpc2/common/scale.py)."""

    def __init__(self, tau: float, device: torch.device):
        self.tau = tau
        self.value = torch.ones(1, device=device)

    def _percentile(self, x: torch.Tensor) -> torch.Tensor:
        x_sorted, _ = torch.sort(x, dim=0)
        n = x_sorted.shape[0]
        percentiles = torch.tensor([5.0, 95.0], device=x.device)
        positions = percentiles * (n - 1) / 100
        floored = torch.floor(positions)
        ceiled = torch.clamp(floored + 1, max=n - 1)
        w_ceiled = (positions - floored).unsqueeze(-1)
        w_floored = 1.0 - w_ceiled
        d0 = x_sorted[floored.long()] * w_floored
        d1 = x_sorted[ceiled.long()] * w_ceiled
        return d0 + d1

    def update(self, x: torch.Tensor):
        p = self._percentile(x.detach())
        value = torch.clamp(p[1] - p[0], min=1.0)
        self.value = self.value * (1 - self.tau) + value * self.tau

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.value


# ------------------------------------------------------------------------
# Network building blocks (ported from tdmpc2/common/layers.py)
# ------------------------------------------------------------------------

class SimNorm(nn.Module):
    """Simplicial normalization (https://arxiv.org/abs/2204.00616): splits
    the last dimension into groups of `dim` units and softmaxes each group."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Linear):
    """Linear layer + LayerNorm + activation (Mish by default) + optional dropout."""

    def __init__(self, in_features, out_features, dropout: float = 0.0, act: Optional[nn.Module] = None):
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.drop is not None:
            x = self.drop(x)
        return self.act(self.ln(x))


def mlp(in_dim: int, mlp_dims: List[int], out_dim: int, act: Optional[nn.Module] = None, dropout: float = 0.0) -> nn.Sequential:
    """TD-MPC2's basic MLP block: NormedLinear hidden layers (dropout only on
    the first), followed by either a plain nn.Linear output (act=None, used
    for reward/Q/policy-prior logits) or a NormedLinear output with a custom
    activation (used for the encoder/dynamics, act=SimNorm)."""
    dims = [in_dim] + list(mlp_dims) + [out_dim]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout if i == 0 else 0.0))
    if act is not None:
        layers.append(NormedLinear(dims[-2], dims[-1], act=act))
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def _weight_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ------------------------------------------------------------------------
# World model
# ------------------------------------------------------------------------

class WorldModel(nn.Module):
    """Implicit (decoder-free) latent world model: encoder h, latent dynamics
    d, reward R, terminal-value Q-ensemble, and Gaussian policy prior p
    (tdmpc2/common/world_model.py, single-task/state-obs subset)."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: TDMPC2Config):
        super().__init__()
        self.cfg = cfg
        self.act_dim = act_dim

        enc_hidden = max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim]
        self.encoder = mlp(obs_dim, enc_hidden, cfg.latent_dim, act=SimNorm(cfg.simnorm_dim))
        self.dynamics = mlp(cfg.latent_dim + act_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg.simnorm_dim))
        self.reward_net = mlp(cfg.latent_dim + act_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.num_bins)
        self.termination_net = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 1) if cfg.episodic else None
        self.pi_net = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 2 * act_dim)
        self.qs = nn.ModuleList([
            mlp(cfg.latent_dim + act_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.num_bins, dropout=cfg.dropout)
            for _ in range(cfg.num_q)
        ])

        self.apply(_weight_init)
        nn.init.zeros_(self.reward_net[-1].weight)
        for net in self.qs:
            nn.init.zeros_(net[-1].weight)

        import copy
        self.qs_target = copy.deepcopy(self.qs)
        for p in self.qs_target.parameters():
            p.requires_grad = False

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max - cfg.log_std_min))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def next_latent(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.reward_net(torch.cat([z, a], dim=-1))

    def termination(self, z: torch.Tensor, unnormalized: bool = False) -> torch.Tensor:
        out = self.termination_net(z)
        return out if unnormalized else torch.sigmoid(out)

    def pi(self, z: torch.Tensor):
        mean, log_std = self.pi_net(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * self.log_std_dif * (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mean)
        log_prob = gaussian_logprob(eps, log_std)

        action = mean + eps * log_std.exp()
        mean, action, log_prob = squash(mean, action, log_prob)

        size = eps.shape[-1]
        scaled_log_prob = log_prob * size
        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def Q(self, z: torch.Tensor, a: torch.Tensor, return_type: str = "min", target: bool = False) -> torch.Tensor:
        assert return_type in {"min", "avg", "all"}
        x = torch.cat([z, a], dim=-1)
        nets = self.qs_target if target else self.qs
        out = torch.stack([net(x) for net in nets], dim=0)  # (num_q, ..., num_bins)
        if return_type == "all":
            return out
        idx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        qvals = two_hot_inv(out[idx], self.cfg.vmin, self.cfg.vmax, self.cfg.num_bins)  # (2, ..., 1)
        if return_type == "min":
            return qvals.min(0).values
        return qvals.sum(0) / 2.0


# ------------------------------------------------------------------------
# Sequence replay buffer (episodes -> random length-`horizon` subsequences)
# ------------------------------------------------------------------------

class _EpisodeSequenceBuffer:
    """Stores whole episodes and samples random length-`horizon` (+1 obs)
    contiguous subsequences, uniformly over *transitions* (an episode of
    length L contributes L - horizon + 1 valid start positions, so it is
    sampled with weight proportional to that count) -- the same effective
    distribution as the reference's SliceSampler-based torchrl buffer
    (tdmpc2/common/buffer.py), implemented with plain numpy since this
    harness doesn't depend on torchrl.

    Capacity is enforced in whole episodes (oldest evicted first, at least
    one episode always kept) once the total transition count exceeds it --
    an episode-granular approximation of the reference's flat transition
    circular buffer."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, horizon: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self._episodes: "collections.deque[dict]" = collections.deque()
        self._total_transitions = 0
        self._reset_open()

    def _reset_open(self):
        self._open_obs = []
        self._open_actions = []
        self._open_rewards = []
        self._open_terminateds = []

    def start_episode(self, obs0: np.ndarray):
        self._reset_open()
        self._open_obs.append(np.asarray(obs0, dtype=np.float32))

    def add(self, action, reward, next_obs, terminated):
        self._open_actions.append(np.asarray(action, dtype=np.float32))
        self._open_rewards.append(np.float32(reward))
        self._open_terminateds.append(np.float32(terminated))
        self._open_obs.append(np.asarray(next_obs, dtype=np.float32))

    def end_episode(self):
        length = len(self._open_actions)
        if length == 0:
            self._reset_open()
            return
        episode = {
            "obs": np.stack(self._open_obs).astype(np.float32),
            "actions": np.stack(self._open_actions).astype(np.float32),
            "rewards": np.asarray(self._open_rewards, dtype=np.float32).reshape(-1, 1),
            "terminateds": np.asarray(self._open_terminateds, dtype=np.float32).reshape(-1, 1),
            "length": length,
        }
        self._episodes.append(episode)
        self._total_transitions += length
        while self._total_transitions > self.capacity and len(self._episodes) > 1:
            old = self._episodes.popleft()
            self._total_transitions -= old["length"]
        self._reset_open()

    def __len__(self):
        return self._total_transitions

    def sample(self, batch_size: int, horizon: int):
        eligible = [ep for ep in self._episodes if ep["length"] >= horizon]
        if not eligible:
            return None
        weights = np.array([ep["length"] - horizon + 1 for ep in eligible], dtype=np.float64)
        probs = weights / weights.sum()
        choices = np.random.choice(len(eligible), size=batch_size, p=probs)

        obs = np.empty((horizon + 1, batch_size, self.obs_dim), dtype=np.float32)
        actions = np.empty((horizon, batch_size, self.act_dim), dtype=np.float32)
        rewards = np.empty((horizon, batch_size, 1), dtype=np.float32)
        terminateds = np.empty((horizon, batch_size, 1), dtype=np.float32)
        for b, ci in enumerate(choices):
            ep = eligible[ci]
            start = np.random.randint(0, ep["length"] - horizon + 1)
            obs[:, b] = ep["obs"][start:start + horizon + 1]
            actions[:, b] = ep["actions"][start:start + horizon]
            rewards[:, b] = ep["rewards"][start:start + horizon]
            terminateds[:, b] = ep["terminateds"][start:start + horizon]
        return obs, actions, rewards, terminateds


# ------------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------------

@dataclass
class UpdateInfo:
    """Wall-clock time (seconds) spent in each sub-segment of one call to
    update(), matching the buffer_sample/critic_update/actor_update/
    target_update naming used by sac.py/td3.py/mbpo.py -- here "critic_update"
    is the world-model step (consistency + reward + value losses) and
    "actor_update" is the policy-prior step, since those are TD-MPC2's
    closest analogues to SAC/TD3's critic and actor updates."""
    buffer_sample_s: float = 0.0
    critic_update_s: float = 0.0
    actor_update_s: float = 0.0
    target_update_s: float = 0.0
    consistency_loss: Optional[float] = None
    reward_loss: Optional[float] = None
    value_loss: Optional[float] = None
    pi_loss: Optional[float] = None
    total_loss: Optional[float] = None


class TDMPC2Agent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, cfg: TDMPC2Config, device: torch.device, episode_length: int):
        self.cfg = cfg
        self.device = device
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

        self.model = WorldModel(obs_dim, act_dim, cfg).to(device)

        world_model_params = [
            {"params": self.model.encoder.parameters(), "lr": cfg.lr * cfg.enc_lr_scale},
            {"params": self.model.dynamics.parameters()},
            {"params": self.model.reward_net.parameters()},
            {"params": self.model.qs.parameters()},
        ]
        if cfg.episodic:
            world_model_params.append({"params": self.model.termination_net.parameters()})
        self.optim = torch.optim.Adam(world_model_params, lr=cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model.pi_net.parameters(), lr=cfg.lr, eps=1e-5)

        self._world_model_modules = [self.model.encoder, self.model.dynamics, self.model.reward_net, self.model.qs]
        if cfg.episodic:
            self._world_model_modules.append(self.model.termination_net)

        self.scale = RunningScale(cfg.tau, device)
        self.iterations = cfg.iterations + 2 * int(act_dim >= 20)  # heuristic for large action spaces (paper)
        self.discount = self._get_discount(episode_length)
        self._prev_mean = torch.zeros(cfg.horizon, act_dim, device=device)

    def _get_discount(self, episode_length: int) -> float:
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

    def _world_model_parameters(self):
        for m in self._world_model_modules:
            yield from m.parameters()

    @torch.no_grad()
    def _estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Rolls a batch of imagined action sequences forward through the
        latent dynamics/reward models and returns discounted return + terminal
        Q-value (paper Eq. for MPPI's trajectory objective)."""
        n = actions.shape[1]
        G = torch.zeros(n, 1, device=self.device)
        discount = 1.0
        termination = torch.zeros(n, 1, device=self.device)
        for t in range(self.cfg.horizon):
            reward = two_hot_inv(self.model.reward(z, actions[t]), self.cfg.vmin, self.cfg.vmax, self.cfg.num_bins)
            z = self.model.next_latent(z, actions[t])
            G = G + discount * (1 - termination) * reward
            discount = discount * self.discount
            if self.cfg.episodic:
                termination = torch.clip(termination + (self.model.termination(z) > 0.5).float(), max=1.0)
        action, _ = self.model.pi(z)
        return G + discount * (1 - termination) * self.model.Q(z, action, return_type="avg")

    @torch.no_grad()
    def act(self, obs: np.ndarray, t0: bool = False, eval_mode: bool = False) -> np.ndarray:
        """Selects an action by MPPI planning in latent space (tdmpc2's `_plan`)."""
        cfg = self.cfg
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.model.encode(obs_t)

        pi_actions = None
        if cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(cfg.horizon, cfg.num_pi_trajs, self.act_dim, device=self.device)
            _z = z.repeat(cfg.num_pi_trajs, 1)
            for t in range(cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z)
                _z = self.model.next_latent(_z, pi_actions[t])
            pi_actions[-1], _ = self.model.pi(_z)

        z = z.repeat(cfg.num_samples, 1)
        mean = torch.zeros(cfg.horizon, self.act_dim, device=self.device)
        std = torch.full((cfg.horizon, self.act_dim), cfg.max_std, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]

        actions = torch.empty(cfg.horizon, cfg.num_samples, self.act_dim, device=self.device)
        if pi_actions is not None:
            actions[:, :cfg.num_pi_trajs] = pi_actions

        score = None
        elite_actions = None
        for _ in range(self.iterations):
            r = torch.randn(cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, self.act_dim, device=self.device)
            actions_sample = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
            actions[:, cfg.num_pi_trajs:] = actions_sample

            value = self._estimate_value(z, actions).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(cfg.min_std, cfg.max_std)

        rand_idx = gumbel_softmax_sample(score.squeeze(1))
        chosen = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std_final = chosen[0], std[0]
        if not eval_mode:
            a = a + std_final * torch.randn(self.act_dim, device=self.device)
        self._prev_mean = mean
        a = a.clamp(-1, 1)
        return (a * self.act_limit).cpu().numpy()

    def _update_pi(self, zs_detached: torch.Tensor) -> Dict[str, float]:
        cfg = self.cfg
        for p in self.model.qs.parameters():
            p.requires_grad = False

        action, info = self.model.pi(zs_detached)
        q_pi = self.model.Q(zs_detached, action, return_type="avg")  # (H+1, B, 1)
        self.scale.update(q_pi[0])
        q_pi_scaled = self.scale(q_pi)

        rho_weights = torch.pow(cfg.rho, torch.arange(zs_detached.shape[0], device=self.device))
        per_step = -(cfg.entropy_coef * info["scaled_entropy"] + q_pi_scaled).mean(dim=1).squeeze(-1)
        pi_loss = (per_step * rho_weights).mean()

        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.pi_net.parameters(), cfg.grad_clip_norm)
        self.pi_optim.step()

        for p in self.model.qs.parameters():
            p.requires_grad = True
        return {"pi_loss": pi_loss.item()}

    def update(self, buffer: _EpisodeSequenceBuffer, batch_size: int) -> UpdateInfo:
        info = UpdateInfo()
        cfg = self.cfg

        t0 = time.perf_counter()
        batch = buffer.sample(batch_size, cfg.horizon)
        if batch is None:
            return info
        obs_np, action_np, reward_np, terminated_np = batch
        obs = torch.as_tensor(obs_np, device=self.device)
        action = torch.as_tensor(action_np, device=self.device)
        reward = torch.as_tensor(reward_np, device=self.device)
        terminated = torch.as_tensor(terminated_np, device=self.device)
        info.buffer_sample_s = time.perf_counter() - t0

        # --- critic_update: world model (consistency + reward + value losses) ---
        t0 = time.perf_counter()
        self.model.train()
        with torch.no_grad():
            next_z = self.model.encode(obs[1:])  # (H, B, latent_dim), real next-obs encodings (loss targets)
            next_action, _ = self.model.pi(next_z)
            target_q = self.model.Q(next_z, next_action, return_type="min", target=True)  # (H, B, 1)
            td_targets = reward + self.discount * (1 - terminated) * target_q

        zs = torch.empty(cfg.horizon + 1, batch_size, cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0])
        zs[0] = z
        consistency_loss = torch.zeros((), device=self.device)
        for t in range(cfg.horizon):
            z = self.model.next_latent(z, action[t])
            consistency_loss = consistency_loss + F.mse_loss(z, next_z[t]) * (cfg.rho ** t)
            zs[t + 1] = z
        consistency_loss = consistency_loss / cfg.horizon

        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, return_type="all")  # (num_q, H, B, num_bins)
        reward_preds = self.model.reward(_zs, action)  # (H, B, num_bins)

        reward_loss = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)
        for t in range(cfg.horizon):
            reward_loss = reward_loss + soft_ce(
                reward_preds[t], reward[t], cfg.vmin, cfg.vmax, cfg.num_bins, self.bin_size
            ).mean() * (cfg.rho ** t)
            for qi in range(cfg.num_q):
                value_loss = value_loss + soft_ce(
                    qs[qi, t], td_targets[t], cfg.vmin, cfg.vmax, cfg.num_bins, self.bin_size
                ).mean() * (cfg.rho ** t)
        reward_loss = reward_loss / cfg.horizon
        value_loss = value_loss / (cfg.horizon * cfg.num_q)

        if cfg.episodic:
            termination_pred = self.model.termination(zs[1:], unnormalized=True)
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
        else:
            termination_loss = torch.zeros((), device=self.device)

        total_loss = (
            cfg.consistency_coef * consistency_loss
            + cfg.reward_coef * reward_loss
            + cfg.value_coef * value_loss
            + cfg.termination_coef * termination_loss
        )

        self.optim.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._world_model_parameters(), cfg.grad_clip_norm)
        self.optim.step()
        info.critic_update_s = time.perf_counter() - t0
        info.consistency_loss = consistency_loss.item()
        info.reward_loss = reward_loss.item()
        info.value_loss = value_loss.item()
        info.total_loss = total_loss.item()

        # --- actor_update: policy prior (Q-maximization + entropy bonus) ---
        t0 = time.perf_counter()
        pi_info = self._update_pi(zs.detach())
        info.actor_update_s = time.perf_counter() - t0
        info.pi_loss = pi_info["pi_loss"]

        # --- target_update: Polyak averaging of target Q-ensemble ---
        t0 = time.perf_counter()
        with torch.no_grad():
            for p, p_targ in zip(self.model.qs.parameters(), self.model.qs_target.parameters()):
                p_targ.data.mul_(1 - cfg.tau)
                p_targ.data.add_(cfg.tau * p.data)
        info.target_update_s = time.perf_counter() - t0

        self.model.eval()
        return info


# ------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------

def train(
    env,
    tdmpc2_cfg: TDMPC2Config,
    exp_cfg: ExperimentConfig,
    tracker,
    device: torch.device,
    logger,
    steps_per_epoch: int = 1000,
):
    """
    Runs TD-MPC2 warmup + world-model pretraining burst + measured training.
    Returns (agent, energy_log, metrics) in the same shape as sac.py's
    train() (see that module's docstring for the full "episodes"/"epochs"
    schema); "epochs" rows additionally carry consistency_loss_mean,
    reward_loss_mean, value_loss_mean, pi_loss_mean, total_loss_mean.

    Segment structure:
      - "warmup": purely random actions filling the sequence buffer (same
        role as in sac.py/td3.py/mbpo.py).
      - "world_model_pretrain": a one-off burst of `warmup_steps` gradient
        updates on the seed data the instant warmup ends, matching the
        reference OnlineTrainer's `num_updates = seed_steps` pretraining
        step -- tracked as its own directly-measured CodeCarbon task since
        it is algorithmically distinct from the steady-state 1-update-per-
        env-step regime that follows.
      - per epoch: "rollout" (MPPI planning + env.step, the dominant cost
        for this algorithm -- unlike SAC/TD3/MBPO) and "gradient_updates",
        sub-split into buffer_sample/critic_update/actor_update/target_update
        by wall-clock time share exactly as in sac.py.
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    episode_length = getattr(getattr(env, "spec", None), "max_episode_steps", None) or tdmpc2_cfg.episode_length

    agent = TDMPC2Agent(obs_dim, act_dim, act_limit, tdmpc2_cfg, device, episode_length=episode_length)
    buffer_capacity = min(tdmpc2_cfg.buffer_capacity, exp_cfg.warmup_steps + exp_cfg.train_steps)
    buffer = _EpisodeSequenceBuffer(buffer_capacity, obs_dim, act_dim, tdmpc2_cfg.horizon)

    energy_log: Dict[str, float] = {}
    sub_time_totals = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}

    episodes_log = []
    episode_idx = 0
    episode_return = 0.0
    episode_length_count = 0

    def _record_episode(phase, epoch, env_step):
        nonlocal episode_idx, episode_return, episode_length_count
        episodes_log.append({
            "episode_idx": episode_idx,
            "phase": phase,
            "epoch": epoch,
            "env_step": env_step,
            "return": episode_return,
            "length": episode_length_count,
        })
        episode_idx += 1
        episode_return = 0.0
        episode_length_count = 0

    obs, _ = env.reset(seed=exp_cfg.seed)
    buffer.start_episode(obs)
    is_t0 = True

    # ---------------- warmup (random policy, fills buffer; excluded from analysis segments) ----------------
    logger.info("Starting warmup: %d steps", exp_cfg.warmup_steps)
    with TrackerTask(tracker, "warmup", 0, energy_log):
        for warmup_step in range(exp_cfg.warmup_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(action, reward, next_obs, terminated)
            episode_return += reward
            episode_length_count += 1
            obs = next_obs
            if terminated or truncated:
                buffer.end_episode()
                _record_episode("warmup", None, warmup_step + 1)
                obs, _ = env.reset()
                buffer.start_episode(obs)
                is_t0 = True

    # ---------------- world-model pretraining burst on seed data ----------------
    logger.info("Pretraining world model on seed data: %d updates", exp_cfg.warmup_steps)
    pretrain_total_losses = []
    with TrackerTask(tracker, "world_model_pretrain", 0, energy_log):
        for _ in range(exp_cfg.warmup_steps):
            info = agent.update(buffer, tdmpc2_cfg.batch_size)
            if info.total_loss is not None:
                pretrain_total_losses.append(info.total_loss)
    if pretrain_total_losses:
        logger.info(
            "Pretrain burst done: %d updates, mean total_loss=%.4f",
            len(pretrain_total_losses), sum(pretrain_total_losses) / len(pretrain_total_losses),
        )

    # ---------------- measured training ----------------
    n_epochs = max(1, exp_cfg.train_steps // steps_per_epoch)
    logger.info("Starting measured training: %d epochs x %d steps = %d env steps",
                n_epochs, steps_per_epoch, n_epochs * steps_per_epoch)

    epochs_log = []
    cumulative_reward = 0.0
    global_step = 0

    for epoch in range(n_epochs):
        # --- rollout block (MPPI planning selects every action) ---
        epoch_reward_sum = 0.0
        epoch_episode_returns = []
        with TrackerTask(tracker, "rollout", epoch, energy_log):
            for _ in range(steps_per_epoch):
                action = agent.act(obs, t0=is_t0, eval_mode=False)
                is_t0 = False
                next_obs, reward, terminated, truncated, _ = env.step(action)
                buffer.add(action, reward, next_obs, terminated)
                episode_return += reward
                episode_length_count += 1
                cumulative_reward += reward
                epoch_reward_sum += reward
                global_step += 1
                obs = next_obs
                if terminated or truncated:
                    buffer.end_episode()
                    epoch_episode_returns.append(episode_return)
                    _record_episode("train", epoch, global_step)
                    obs, _ = env.reset()
                    buffer.start_episode(obs)
                    is_t0 = True

        # --- gradient update block ---
        epoch_sub_times = {"buffer_sample": 0.0, "critic_update": 0.0, "actor_update": 0.0, "target_update": 0.0}
        consistency_losses, reward_losses, value_losses, pi_losses, total_losses = [], [], [], [], []
        n_updates = steps_per_epoch * tdmpc2_cfg.updates_per_env_step
        with TrackerTask(tracker, "gradient_updates", epoch, energy_log):
            for _ in range(n_updates):
                info = agent.update(buffer, tdmpc2_cfg.batch_size)
                epoch_sub_times["buffer_sample"] += info.buffer_sample_s
                epoch_sub_times["critic_update"] += info.critic_update_s
                epoch_sub_times["actor_update"] += info.actor_update_s
                epoch_sub_times["target_update"] += info.target_update_s
                if info.consistency_loss is not None:
                    consistency_losses.append(info.consistency_loss)
                if info.reward_loss is not None:
                    reward_losses.append(info.reward_loss)
                if info.value_loss is not None:
                    value_losses.append(info.value_loss)
                if info.pi_loss is not None:
                    pi_losses.append(info.pi_loss)
                if info.total_loss is not None:
                    total_losses.append(info.total_loss)

        for k in sub_time_totals:
            sub_time_totals[k] += epoch_sub_times[k]

        def _mean(xs):
            return (sum(xs) / len(xs)) if xs else None

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
            "consistency_loss_mean": _mean(consistency_losses),
            "reward_loss_mean": _mean(reward_losses),
            "value_loss_mean": _mean(value_losses),
            "pi_loss_mean": _mean(pi_losses),
            "total_loss_mean": _mean(total_losses),
            "buffer_size": len(buffer),
        })

        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            logger.info(
                "Epoch %d/%d done (buffer size=%d, cumulative_reward=%.2f, mean_episode_return=%s)",
                epoch + 1, n_epochs, len(buffer), cumulative_reward,
                epochs_log[-1]["mean_episode_return"],
            )

    # record a trailing partial episode (if training ended mid-episode) so no reward is lost
    if episode_length_count > 0:
        buffer.end_episode()
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
