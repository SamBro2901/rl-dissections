"""
Probabilistic ensemble dynamics model for MBPO (Janner, Fu, Zhang & Levine,
2019, "When to Trust Your Model: Model-Based Policy Optimization", NeurIPS
2019), using the network design from Chua et al. (2018, "Deep Reinforcement
Learning in a Handful of Trials using Probabilistic Dynamics Models" / PETS)
that the MBPO reference implementation (https://github.com/JannerM/mbpo)
reuses directly.

Each ensemble member is an MLP predicting a diagonal Gaussian
N(mean, diag(exp(logvar))) over the target [delta_obs, reward], given
[obs, act] as input. The ensemble is trained by penalized Gaussian
negative-log-likelihood with learned, softly-bounded log-variance limits
(Chua et al. 2018, Appendix A.1), using a held-out validation split for
early stopping. At rollout time, predictions are drawn from a randomly
selected "elite" member per sample (the `num_elites` members with the lowest
validation error) -- this bootstrap-ensemble mechanism is what MBPO relies on
to keep short model rollouts from compounding a single member's bias
(Janner et al. 2019, Section 3.1).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.replay_buffer import ReplayBuffer


class _Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GaussianEnsembleMLP(nn.Module):
    """`ensemble_size` independent MLPs, each predicting a diagonal Gaussian
    over [delta_obs, reward] given [obs, act]. Kept as separate members
    (rather than a batched/vmapped implementation) for readability, matching
    the twin-Q style already used in algorithms/sac.py."""

    def __init__(self, obs_dim: int, act_dim: int, ensemble_size: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        in_dim = obs_dim + act_dim
        out_dim = obs_dim + 1  # predicted delta_obs (obs_dim dims) + reward (1 dim)
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.ensemble_size = ensemble_size

        self.trunks = nn.ModuleList()
        for _ in range(ensemble_size):
            layers = []
            prev = in_dim
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), _Swish()]
                prev = h
            self.trunks.append(nn.Sequential(*layers))

        self.mean_heads = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_dim) for _ in range(ensemble_size)])
        self.logvar_heads = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_dim) for _ in range(ensemble_size)])

        # Learned, softly-clamped log-variance bounds (Chua et al. 2018, Appendix A.1):
        # keeps predicted variance from collapsing to 0 or exploding early in training.
        self.max_logvar = nn.Parameter(torch.full((out_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((out_dim,), -10.0))

    def _bound_logvar(self, logvar):
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return logvar

    def forward_member(self, member_idx: int, obs_act: torch.Tensor):
        h = self.trunks[member_idx](obs_act)
        mean = self.mean_heads[member_idx](h)
        logvar = self._bound_logvar(self.logvar_heads[member_idx](h))
        return mean, logvar

    def forward_all(self, obs_act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (means, logvars), each of shape [ensemble_size, batch, out_dim]."""
        means, logvars = [], []
        for i in range(self.ensemble_size):
            mean, logvar = self.forward_member(i, obs_act)
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)


class RunningNormalizer(nn.Module):
    """Standardizes [obs, act] inputs to the dynamics model using the mean/std
    of the current training split. Refit every time the ensemble is retrained
    (MBPO/PETS convention) -- keeps the model's input scale stable as the
    replay buffer grows and its distribution shifts over training."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x: torch.Tensor):
        with torch.no_grad():
            self.mean.copy_(x.mean(dim=0))
            std = x.std(dim=0)
            std[std < 1e-12] = 1.0
            self.std.copy_(std)

    def forward(self, x):
        return (x - self.mean) / self.std


class EnsembleDynamicsModel:
    """Owns a GaussianEnsembleMLP plus its training loop (penalized Gaussian
    NLL with holdout early stopping, per-layer weight decay matching the
    PETS/MBPO reference code) and elite-based prediction for model rollouts."""

    def __init__(self, obs_dim: int, act_dim: int, cfg, device: torch.device):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.device = device

        hidden_sizes = cfg.model_hidden_sizes
        decays = cfg.model_weight_decays
        if len(decays) != len(hidden_sizes) + 1:
            raise ValueError(
                f"MBPOConfig.model_weight_decays must have len(model_hidden_sizes) + 1 = "
                f"{len(hidden_sizes) + 1} entries (one per hidden layer, one for the output "
                f"heads), got {len(decays)}."
            )

        self.net = GaussianEnsembleMLP(obs_dim, act_dim, cfg.ensemble_size, hidden_sizes).to(device)
        self.normalizer = RunningNormalizer(obs_dim + act_dim).to(device)

        param_groups = []
        for i in range(cfg.ensemble_size):
            layer_linears = [m for m in self.net.trunks[i] if isinstance(m, nn.Linear)]
            for linear, wd in zip(layer_linears, decays[:-1]):
                param_groups.append({"params": linear.parameters(), "weight_decay": wd})
            param_groups.append({"params": self.net.mean_heads[i].parameters(), "weight_decay": decays[-1]})
            param_groups.append({"params": self.net.logvar_heads[i].parameters(), "weight_decay": decays[-1]})
        param_groups.append({"params": [self.net.max_logvar, self.net.min_logvar], "weight_decay": 0.0})

        self.optimizer = torch.optim.Adam(param_groups, lr=cfg.model_lr)
        # Until the first fit() call, treat the first `num_elites` members as elites
        # (arbitrary but harmless: warmup fills the real buffer before any model
        # rollout is ever generated, so fit() always runs at least once first).
        self.elite_indices = list(range(cfg.num_elites))

    def fit(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """Trains the full ensemble from scratch on the current contents of
        `buffer` (MBPO retrains periodically on *all* real data seen so far,
        not just a minibatch -- see Janner et al. 2019 Algorithm 1, line 4).
        Each member is trained on its own bootstrap resample of the training
        split (Chua et al. 2018 Appendix A.1). Returns diagnostics for logging.
        """
        n = buffer.size
        obs = buffer.obs[:n]
        act = buffer.actions[:n]
        rew = buffer.rewards[:n, 0]
        next_obs = buffer.next_obs[:n]

        inputs = np.concatenate([obs, act], axis=1)
        targets = np.concatenate([next_obs - obs, rew[:, None]], axis=1)

        n_holdout = max(1, int(n * self.cfg.model_holdout_ratio))
        perm = np.random.permutation(n)
        holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]

        inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        targets_t = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        self.normalizer.fit(inputs_t[train_idx])

        n_train = len(train_idx)
        # Bootstrap resample per ensemble member (with replacement), fixed for this fit() call.
        bootstrap_idx = np.random.randint(0, n_train, size=(self.cfg.ensemble_size, n_train))
        member_train_idx = train_idx[bootstrap_idx]  # [ensemble_size, n_train]

        best_holdout_mse = None
        epochs_since_improved = 0
        epochs_run = 0
        batch_size = self.cfg.model_train_batch_size

        for epoch in range(self.cfg.model_max_train_epochs):
            epochs_run = epoch + 1
            # shuffle each member's bootstrap sample independently
            for i in range(self.cfg.ensemble_size):
                np.random.shuffle(member_train_idx[i])

            for start in range(0, n_train, batch_size):
                end = start + batch_size
                total_loss = 0.0
                self.optimizer.zero_grad(set_to_none=True)
                for i in range(self.cfg.ensemble_size):
                    idx = member_train_idx[i, start:end]
                    x = self.normalizer(inputs_t[idx])
                    y = targets_t[idx]
                    mean, logvar = self.net.forward_member(i, x)
                    inv_var = torch.exp(-logvar)
                    nll = (((mean - y) ** 2) * inv_var + logvar).mean()
                    total_loss = total_loss + nll
                total_loss = total_loss + 0.01 * (
                    self.net.max_logvar.sum() - self.net.min_logvar.sum()
                )
                total_loss.backward()
                self.optimizer.step()

            holdout_mse = self._holdout_mse(inputs_t[holdout_idx], targets_t[holdout_idx])
            mean_holdout_mse = float(holdout_mse.mean())
            if best_holdout_mse is None or mean_holdout_mse < best_holdout_mse - 1e-4:
                best_holdout_mse = mean_holdout_mse
                best_per_member = holdout_mse
                epochs_since_improved = 0
            else:
                epochs_since_improved += 1
                if epochs_since_improved >= self.cfg.model_train_patience:
                    break

        self.elite_indices = list(
            np.argsort(best_per_member.cpu().numpy())[: self.cfg.num_elites]
        )

        return {
            "holdout_mse": best_holdout_mse,
            "train_epochs": epochs_run,
            "train_transitions": int(n),
        }

    @torch.no_grad()
    def _holdout_mse(self, inputs_t: torch.Tensor, targets_t: torch.Tensor) -> torch.Tensor:
        """Per-member MSE (not NLL) on the holdout split -- used for both early
        stopping and elite selection, following the PETS/MBPO convention of
        ranking ensemble members by plain prediction error rather than
        likelihood (which the learned logvar could otherwise game)."""
        x = self.normalizer(inputs_t)
        means, _ = self.net.forward_all(x)  # [ensemble_size, n_holdout, out_dim]
        return ((means - targets_t.unsqueeze(0)) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """For each row in the batch, samples a random *elite* ensemble member
        (bootstrap ensemble prediction, Chua et al. 2018 Sec 3.2 / Janner et
        al. 2019 Sec 3.1) and returns (next_obs, reward)."""
        batch_size = obs.shape[0]
        obs_act = torch.cat([obs, act], dim=-1)
        obs_act_n = self.normalizer(obs_act)

        means, logvars = self.net.forward_all(obs_act_n)  # [ensemble_size, batch, out_dim]
        elite = torch.as_tensor(self.elite_indices, device=self.device)
        choice = elite[torch.randint(0, len(elite), (batch_size,), device=self.device)]
        rows = torch.arange(batch_size, device=self.device)

        mean = means[choice, rows]
        logvar = logvars[choice, rows]

        if self.cfg.deterministic_model:
            sample = mean
        else:
            std = torch.exp(0.5 * logvar)
            sample = mean + std * torch.randn_like(mean)

        delta_obs = sample[:, : self.obs_dim]
        reward = sample[:, self.obs_dim]
        next_obs = obs + delta_obs
        return next_obs, reward
