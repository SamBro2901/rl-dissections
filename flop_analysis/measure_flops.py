"""
Step 1 of the FLOP-per-segment methodology (see flop_calculation_methodology.md).

Measures per-call FLOPs for every (algorithm, environment) combination that
actually appears under results/, using torch's native FlopCounterMode against
the REAL network classes and, for TD-MPC2, the REAL TDMPC2Agent.act()/update()
methods -- not hand-derived analytic estimates. FLOP count is deterministic
given (architecture, batch size, planner hyperparameters), so this only needs
to run once per distinct (algo, env_id) pair, not once per run/seed.

Output: flop_analysis/flops_per_call.json

Usage (from repo root, with the project venv active):
    python3 flop_analysis/measure_flops.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from algorithms.sac import GaussianPolicy as SACActor, QNetwork as SACQNetwork  # noqa: E402
from algorithms.td3 import DeterministicPolicy as TD3Actor, QNetwork as TD3QNetwork  # noqa: E402
from algorithms.dynamics_model import GaussianEnsembleMLP  # noqa: E402
from algorithms.tdmpc2 import TDMPC2Agent, soft_ce  # noqa: E402
from configs.config import TDMPC2Config  # noqa: E402
from flop_keys import signature  # noqa: E402

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACT_LIMIT = 1.0  # a scale constant only -- never touches matmul shapes, doesn't affect FLOP counts


# ------------------------------------------------------------------------
# Discovery: find every distinct (algo, env_id, architecture-signature)
# combination under results/ -- NOT just (algo, env_id). This results/
# directory contains more than the UTD sweep: e.g. sac/HalfCheetah-v5 has
# runs at hidden_sizes (256,256), (512,512) and (1024,1024) left over from
# the project's evolution before settling on 1024x1024. FLOP count depends
# on architecture, so every distinct combination that actually appears must
# be measured (see flop_keys.py).
# ------------------------------------------------------------------------

def discover_configs(results_dir: str) -> dict:
    configs = {}
    n_meta = 0
    for meta_path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "seed_*", "*", "metadata.json"))):
        n_meta += 1
        with open(meta_path) as f:
            meta = json.load(f)
        algo = meta["algo_name"]
        env_id = meta["env_id"]
        sig = signature(algo, meta["algo_config"])
        key = (algo, env_id, sig)
        if key in configs:
            continue
        configs[key] = {
            "algo_config": meta["algo_config"],
            "experiment_config": meta["experiment_config"],
            "example_run": meta_path,
        }
    print(f"Scanned {n_meta} metadata.json files -> {len(configs)} distinct (algo, env, architecture) combos")
    return configs


def env_dims(env_id: str):
    import gymnasium as gym
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    episode_length = getattr(getattr(env, "spec", None), "max_episode_steps", None) or 1000
    env.close()
    return obs_dim, act_dim, episode_length


# ------------------------------------------------------------------------
# Analytic cross-check (doc Step 1, optional but recommended): dense-layer
# FLOP formula, forward ~= 2*B*n_in*n_out per layer, fwd+bwd ~= 3x forward.
# ------------------------------------------------------------------------

def analytic_forward_flops(layer_dims, batch_size: int) -> int:
    """layer_dims: list of (n_in, n_out) pairs for each Linear layer."""
    return sum(2 * batch_size * n_in * n_out for n_in, n_out in layer_dims)


# ------------------------------------------------------------------------
# SAC / TD3 (and MBPO, which reuses SAC's actor/critic classes unmodified)
# ------------------------------------------------------------------------

def measure_sac(obs_dim, act_dim, hidden_sizes, batch_size):
    actor = SACActor(obs_dim, act_dim, hidden_sizes, ACT_LIMIT).to(DEVICE)
    q1 = SACQNetwork(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    q2 = SACQNetwork(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    q1t = SACQNetwork(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    q2t = SACQNetwork(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())

    q1_opt = torch.optim.Adam(q1.parameters(), lr=3e-4)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=3e-4)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    log_alpha = torch.zeros(1, device=DEVICE, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)

    # --- rollout: action selection, bs=1, no grad (matches SACAgent.select_action) ---
    obs1 = torch.randn(1, obs_dim, device=DEVICE)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        actor(obs1, deterministic=False, with_logprob=False)
    actor_forward_bs1 = fc.get_total_flops()

    # --- critic_update: twin-Q forward/backward (matches SACAgent.update()'s critic block) ---
    obsB = torch.randn(batch_size, obs_dim, device=DEVICE)
    actB = torch.rand(batch_size, act_dim, device=DEVICE) * 2 - 1
    rewB = torch.randn(batch_size, device=DEVICE)
    nobsB = torch.randn(batch_size, obs_dim, device=DEVICE)
    doneB = torch.zeros(batch_size, device=DEVICE)

    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            next_action, next_logp = actor(nobsB)
            q1n = q1t(nobsB, next_action)
            q2n = q2t(nobsB, next_action)
            q_next = torch.min(q1n, q2n) - log_alpha.exp().detach() * next_logp
            target_q = rewB + (1.0 - doneB) * 0.99 * q_next
        q1p = q1(obsB, actB)
        q2p = q2(obsB, actB)
        q1_loss = F.mse_loss(q1p, target_q)
        q2_loss = F.mse_loss(q2p, target_q)
        q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        q1_opt.step()
        q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        q2_opt.step()
    critic_fwdbwd = fc.get_total_flops()

    # --- actor_update (+alpha): policy fwd/bwd + critic forward (no grad through critic) ---
    with FlopCounterMode(display=False) as fc:
        for p in q1.parameters():
            p.requires_grad = False
        for p in q2.parameters():
            p.requires_grad = False
        action, logp = actor(obsB)
        q1_pi = q1(obsB, action)
        q2_pi = q2(obsB, action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (log_alpha.exp().detach() * logp - q_pi).mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()
        for p in q1.parameters():
            p.requires_grad = True
        for p in q2.parameters():
            p.requires_grad = True
        alpha_loss = -(log_alpha * (logp.detach() + (-float(act_dim)))).mean()
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_opt.step()
    actor_fwdbwd = fc.get_total_flops()

    target_update_ops = 2 * (sum(p.numel() for p in q1.parameters()) + sum(p.numel() for p in q2.parameters()))

    analytic_actor_fwd = analytic_forward_flops(
        [(obs_dim, hidden_sizes[0]), (hidden_sizes[0], hidden_sizes[1])], 1
    ) + 2 * analytic_forward_flops([(hidden_sizes[1], act_dim)], 1)  # mu + log_std heads
    analytic_critic_fwdbwd = 3 * 2 * analytic_forward_flops(
        [(obs_dim + act_dim, hidden_sizes[0]), (hidden_sizes[0], hidden_sizes[1]), (hidden_sizes[1], 1)],
        batch_size,
    )  # x2 twin heads, x3 for fwd+bwd

    return {
        "obs_dim": obs_dim, "act_dim": act_dim, "hidden_sizes": list(hidden_sizes), "batch_size": batch_size,
        "actor_forward_bs1": actor_forward_bs1,
        "critic_fwdbwd": critic_fwdbwd,
        "actor_fwdbwd": actor_fwdbwd,
        "target_update_elementwise_ops": target_update_ops,
        "_analytic_cross_check": {
            "actor_forward_bs1_analytic": analytic_actor_fwd,
            "critic_fwdbwd_analytic_approx": analytic_critic_fwdbwd,
        },
    }


def measure_td3(obs_dim, act_dim, hidden_sizes, batch_size):
    actor = TD3Actor(obs_dim, act_dim, hidden_sizes, ACT_LIMIT).to(DEVICE)
    actor_targ = TD3Actor(obs_dim, act_dim, hidden_sizes, ACT_LIMIT).to(DEVICE)
    actor_targ.load_state_dict(actor.state_dict())
    q1 = TD3QNetwork(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    q2 = TD3QNetwork(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    q1t = TD3QNetwork(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    q2t = TD3QNetwork(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())

    q1_opt = torch.optim.Adam(q1.parameters(), lr=1e-3)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=1e-3)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)

    obs1 = torch.randn(1, obs_dim, device=DEVICE)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        actor(obs1)
    actor_forward_bs1 = fc.get_total_flops()

    obsB = torch.randn(batch_size, obs_dim, device=DEVICE)
    actB = torch.rand(batch_size, act_dim, device=DEVICE) * 2 - 1
    rewB = torch.randn(batch_size, device=DEVICE)
    nobsB = torch.randn(batch_size, obs_dim, device=DEVICE)
    doneB = torch.zeros(batch_size, device=DEVICE)

    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            noise = (torch.randn_like(actB) * 0.2).clamp(-0.5, 0.5)
            next_action = (actor_targ(nobsB) + noise).clamp(-1.0, 1.0)
            q1n = q1t(nobsB, next_action)
            q2n = q2t(nobsB, next_action)
            q_next = torch.min(q1n, q2n)
            target_q = rewB + (1.0 - doneB) * 0.99 * q_next
        q1p = q1(obsB, actB)
        q2p = q2(obsB, actB)
        q1_loss = F.mse_loss(q1p, target_q)
        q2_loss = F.mse_loss(q2p, target_q)
        q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        q1_opt.step()
        q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        q2_opt.step()
    critic_fwdbwd = fc.get_total_flops()

    with FlopCounterMode(display=False) as fc:
        for p in q1.parameters():
            p.requires_grad = False
        actor_loss = -q1(obsB, actor(obsB)).mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()
        for p in q1.parameters():
            p.requires_grad = True
    actor_fwdbwd = fc.get_total_flops()

    target_update_ops = 2 * (
        sum(p.numel() for p in actor.parameters())
        + sum(p.numel() for p in q1.parameters())
        + sum(p.numel() for p in q2.parameters())
    )

    return {
        "obs_dim": obs_dim, "act_dim": act_dim, "hidden_sizes": list(hidden_sizes), "batch_size": batch_size,
        "actor_forward_bs1": actor_forward_bs1,
        "critic_fwdbwd": critic_fwdbwd,
        "actor_fwdbwd": actor_fwdbwd,
        "target_update_elementwise_ops": target_update_ops,
    }


def measure_mbpo_dynamics(obs_dim, act_dim, ensemble_size, hidden_sizes, batch_size):
    net = GaussianEnsembleMLP(obs_dim, act_dim, ensemble_size, hidden_sizes).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    xB = torch.randn(batch_size, obs_dim + act_dim, device=DEVICE)
    yB = torch.randn(batch_size, obs_dim + 1, device=DEVICE)
    with FlopCounterMode(display=False) as fc:
        mean, logvar = net.forward_member(0, xB)
        inv_var = torch.exp(-logvar)
        nll = (((mean - yB) ** 2) * inv_var + logvar).mean()
        opt.zero_grad(set_to_none=True)
        nll.backward()
        opt.step()
    member_fwdbwd = fc.get_total_flops()

    # forward_all() already loops over every ensemble member -- this constant
    # IS the full per-sample ensemble forward cost used by predict() during
    # synthetic_rollout_generation (see dynamics_model.py: predict() always
    # scores every member before selecting an elite's prediction per row).
    x1 = torch.randn(1, obs_dim + act_dim, device=DEVICE)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net.forward_all(x1)
    ensemble_forward_all_bs1 = fc.get_total_flops()

    dims = [obs_dim + act_dim] + list(hidden_sizes)
    layer_dims = list(zip(dims[:-1], dims[1:]))
    trunk_fwd = analytic_forward_flops(layer_dims, batch_size)
    head_fwd = 2 * analytic_forward_flops([(hidden_sizes[-1], obs_dim + 1)], batch_size)  # mean + logvar heads
    analytic_member_fwdbwd = 3 * (trunk_fwd + head_fwd)

    return {
        "obs_dim": obs_dim, "act_dim": act_dim, "ensemble_size": ensemble_size,
        "model_hidden_sizes": list(hidden_sizes), "model_train_batch_size": batch_size,
        "dynamics_member_fwdbwd": member_fwdbwd,
        "dynamics_ensemble_forward_all_bs1": ensemble_forward_all_bs1,
        "_analytic_cross_check": {"dynamics_member_fwdbwd_analytic_approx": analytic_member_fwdbwd},
    }


# ------------------------------------------------------------------------
# TD-MPC2
# ------------------------------------------------------------------------

class _FakeSequenceBuffer:
    """Same .sample(batch_size, horizon) contract as _EpisodeSequenceBuffer,
    filled with random data of the right shape/dtype -- only shapes matter
    for FLOP counting, not values."""

    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def sample(self, batch_size, horizon):
        obs = np.random.randn(horizon + 1, batch_size, self.obs_dim).astype(np.float32)
        actions = (np.random.rand(horizon, batch_size, self.act_dim).astype(np.float32) * 2 - 1)
        rewards = np.random.randn(horizon, batch_size, 1).astype(np.float32)
        terminateds = np.zeros((horizon, batch_size, 1), dtype=np.float32)
        return obs, actions, rewards, terminateds


def measure_tdmpc2_gradient_breakdown(agent: TDMPC2Agent, batch_size, obs_dim, act_dim):
    """Replicates TDMPC2Agent.update()'s body (tdmpc2.py lines ~576-647) in
    three separate FlopCounterMode windows sharing the same intermediate `zs`
    tensor, so critic_update (world-model)/actor_update (_update_pi, a real
    standalone method called directly)/target_update can be reported
    separately -- matching the same buffer_sample/critic_update/actor_update/
    target_update granularity segment_energy.json already reconciles by
    wall-clock time share."""
    cfg = agent.cfg
    H = cfg.horizon
    obs = torch.randn(H + 1, batch_size, obs_dim, device=DEVICE)
    action = torch.rand(H, batch_size, act_dim, device=DEVICE) * 2 - 1
    reward = torch.randn(H, batch_size, 1, device=DEVICE)
    terminated = torch.zeros(H, batch_size, 1, device=DEVICE)

    agent.model.train()
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            next_z = agent.model.encode(obs[1:])
            next_action, _ = agent.model.pi(next_z)
            target_q = agent.model.Q(next_z, next_action, return_type="min", target=True)
            td_targets = reward + agent.discount * (1 - terminated) * target_q

        zs = torch.empty(H + 1, batch_size, cfg.latent_dim, device=DEVICE)
        z = agent.model.encode(obs[0])
        zs[0] = z
        consistency_loss = torch.zeros((), device=DEVICE)
        for t in range(H):
            z = agent.model.next_latent(z, action[t])
            consistency_loss = consistency_loss + F.mse_loss(z, next_z[t]) * (cfg.rho ** t)
            zs[t + 1] = z
        consistency_loss = consistency_loss / H

        _zs = zs[:-1]
        qs = agent.model.Q(_zs, action, return_type="all")
        reward_preds = agent.model.reward(_zs, action)

        reward_loss = torch.zeros((), device=DEVICE)
        value_loss = torch.zeros((), device=DEVICE)
        for t in range(H):
            reward_loss = reward_loss + soft_ce(
                reward_preds[t], reward[t], cfg.vmin, cfg.vmax, cfg.num_bins, agent.bin_size
            ).mean() * (cfg.rho ** t)
            for qi in range(cfg.num_q):
                value_loss = value_loss + soft_ce(
                    qs[qi, t], td_targets[t], cfg.vmin, cfg.vmax, cfg.num_bins, agent.bin_size
                ).mean() * (cfg.rho ** t)
        reward_loss = reward_loss / H
        value_loss = value_loss / (H * cfg.num_q)

        if cfg.episodic:
            termination_pred = agent.model.termination(zs[1:], unnormalized=True)
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
        else:
            termination_loss = torch.zeros((), device=DEVICE)

        total_loss = (
            cfg.consistency_coef * consistency_loss
            + cfg.reward_coef * reward_loss
            + cfg.value_coef * value_loss
            + cfg.termination_coef * termination_loss
        )
        agent.optim.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent._world_model_parameters(), cfg.grad_clip_norm)
        agent.optim.step()
    critic_flops = fc.get_total_flops()

    with FlopCounterMode(display=False) as fc:
        agent._update_pi(zs.detach())
    actor_flops = fc.get_total_flops()

    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            for p, p_targ in zip(agent.model.qs.parameters(), agent.model.qs_target.parameters()):
                p_targ.data.mul_(1 - cfg.tau)
                p_targ.data.add_(cfg.tau * p.data)
    target_flops = fc.get_total_flops()
    target_ops = 2 * sum(p.numel() for p in agent.model.qs.parameters())

    agent.model.eval()
    return critic_flops, actor_flops, target_flops, target_ops


def measure_tdmpc2(obs_dim, act_dim, episode_length, algo_config):
    cfg = TDMPC2Config(**{k: v for k, v in algo_config.items()})
    agent = TDMPC2Agent(obs_dim, act_dim, ACT_LIMIT, cfg, DEVICE, episode_length=episode_length)
    agent.model.eval()

    # --- rollout: one full MPPI planning call (agent.act(), unmodified) ---
    obs = np.random.randn(obs_dim).astype(np.float32)
    with FlopCounterMode(display=False) as fc:
        agent.act(obs, t0=False, eval_mode=False)
    rollout_flops = fc.get_total_flops()

    # --- cross-check: full agent.update() black-box vs. the 3-way breakdown ---
    fake_buffer = _FakeSequenceBuffer(obs_dim, act_dim)
    with FlopCounterMode(display=False) as fc:
        agent.update(fake_buffer, cfg.batch_size)
    gradient_update_total_blackbox = fc.get_total_flops()

    critic_flops, actor_flops, target_flops, target_ops = measure_tdmpc2_gradient_breakdown(
        agent, cfg.batch_size, obs_dim, act_dim
    )
    breakdown_sum = critic_flops + actor_flops + target_flops

    return {
        "obs_dim": obs_dim, "act_dim": act_dim, "batch_size": cfg.batch_size,
        "horizon": cfg.horizon, "num_samples": cfg.num_samples, "iterations": agent.iterations,
        "num_pi_trajs": cfg.num_pi_trajs, "num_elites": cfg.num_elites, "num_q": cfg.num_q,
        "episodic": cfg.episodic, "episode_length": episode_length,
        "rollout_per_env_step": rollout_flops,
        "gradient_update_critic": critic_flops,
        "gradient_update_actor": actor_flops,
        "gradient_update_target": target_flops,
        "gradient_update_target_elementwise_ops": target_ops,
        "gradient_update_total": breakdown_sum,
        "_cross_check_blackbox_vs_breakdown": {
            "blackbox_total": gradient_update_total_blackbox,
            "breakdown_total": breakdown_sum,
            "pct_diff": (
                100.0 * (breakdown_sum - gradient_update_total_blackbox) / gradient_update_total_blackbox
                if gradient_update_total_blackbox else None
            ),
        },
    }


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    results_dir = os.path.join(REPO_ROOT, "results")
    configs = discover_configs(results_dir)
    print(f"Discovered {len(configs)} (algo, env, architecture) combinations under {results_dir}:")
    for (algo, env_id, sig) in configs:
        print(f"  - {algo} / {env_id} / {sig}")

    out = {"sac": {}, "td3": {}, "mbpo": {}, "tdmpc2": {}, "_device_used_for_measurement": str(DEVICE)}

    for (algo, env_id, sig), cfg in configs.items():
        algo_config = cfg["algo_config"]
        obs_dim, act_dim, episode_length = env_dims(env_id)
        print(f"Measuring {algo} / {env_id} / {sig} (obs_dim={obs_dim}, act_dim={act_dim}) ...")
        out[algo].setdefault(env_id, {})

        if algo == "sac":
            out["sac"][env_id][sig] = measure_sac(
                obs_dim, act_dim, tuple(algo_config["hidden_sizes"]), algo_config["batch_size"]
            )
        elif algo == "td3":
            out["td3"][env_id][sig] = measure_td3(
                obs_dim, act_dim, tuple(algo_config["hidden_sizes"]), algo_config["batch_size"]
            )
        elif algo == "mbpo":
            sac_part = measure_sac(
                obs_dim, act_dim, tuple(algo_config["hidden_sizes"]), algo_config["batch_size"]
            )
            dyn_part = measure_mbpo_dynamics(
                obs_dim, act_dim, algo_config["ensemble_size"],
                tuple(algo_config["model_hidden_sizes"]), algo_config["model_train_batch_size"],
            )
            merged = dict(sac_part)
            merged.update(dyn_part)
            out["mbpo"][env_id][sig] = merged
        elif algo == "tdmpc2":
            out["tdmpc2"][env_id][sig] = measure_tdmpc2(obs_dim, act_dim, episode_length, algo_config)
        else:
            print(f"  (skipping unknown algo {algo!r})")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flops_per_call.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
