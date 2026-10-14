"""
Analytic cross-check of the measured per-call FLOP constants in
flops_per_call.json.

The constants were measured with torch.utils.flop_counter.FlopCounterMode on
the real network classes (measure_flops.py).  FlopCounterMode counts matrix
products only (mm / addmm / bmm), i.e. 2 * rows * n_in * n_out per dense layer
and per matrix product.  This script derives the same numbers from the layer
dimensions alone, following the code path of each segment:

  forward of a dense layer          2 * rows * n_in * n_out
  backward, weight gradient         2 * rows * n_in * n_out   (parameter requires grad)
  backward, input gradient          2 * rows * n_in * n_out   (not for a first layer whose
                                                                input needs no gradient)

It needs only the standard library and the project's torch-free
configs/config.py (for TDMPC2Config defaults); no GPU and no torch.

Covered:
  SAC / TD3 (and the SAC learner inside every MBPO signature):
      actor_forward_bs1, critic_fwdbwd, actor_fwdbwd, target_update_elementwise_ops
  MBPO dynamics ensemble:
      dynamics_member_fwdbwd (one member, forward + backward, minibatch of
      model_train_batch_size), dynamics_ensemble_forward_all_bs1
  TD-MPC2:
      rollout_per_env_step (one planning call), gradient_update_critic
      (world-model step), gradient_update_actor (policy-prior step),
      gradient_update_target_elementwise_ops

Every analytic value is compared with the measured one.  The script prints a
table and exits with status 1 if any relative difference is not exactly 0.

Usage (from the repository root):
    python flop_analysis/verify_analytic_flops.py
"""
from __future__ import annotations

import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from configs.config import TDMPC2Config  # noqa: E402

JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flops_per_call.json")


# ------------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------------

def mm(rows: int, n_in: int, n_out: int) -> int:
    return 2 * rows * n_in * n_out


def fwd(rows: int, layers) -> int:
    """Forward FLOPs of a stack of dense layers; layers = [(n_in, n_out), ...]."""
    return sum(mm(rows, i, o) for i, o in layers)


def bwd(rows: int, layers, first_input_grad: bool, weights: bool = True) -> int:
    """Backward FLOPs of a stack of dense layers.

    weights: parameter gradients are computed (parameters require grad).
    first_input_grad: the input of the first layer needs a gradient; the inputs
    of all later layers always do when anything upstream needs one.  A layer
    whose input needs no gradient and whose parameters need none costs nothing.
    """
    total = 0
    for k, (i, o) in enumerate(layers):
        if weights:
            total += mm(rows, i, o)
        if k > 0 or first_input_grad:
            total += mm(rows, i, o)
    return total


def n_params(layers) -> int:
    return sum(i * o + o for i, o in layers)


def dims(*sizes):
    return list(zip(sizes[:-1], sizes[1:]))


# ------------------------------------------------------------------------
# SAC / TD3
# ------------------------------------------------------------------------

def sac_analytic(obs, act, hidden, B):
    h1, h2 = hidden
    trunk = dims(obs, h1, h2)
    heads = dims(h2, act)  # mu head; the log-std head has the same shape
    q = dims(obs + act, h1, h2, 1)

    actor_fwd = lambda rows: fwd(rows, trunk) + 2 * fwd(rows, heads)
    q_fwd = lambda rows: fwd(rows, q)

    out = {}
    out["actor_forward_bs1"] = actor_fwd(1)

    # critic update: target actor-free policy forward (no grad), two target Q forwards,
    # two online Q forwards, backward of both online Qs (weights; no input grad at L1)
    out["critic_fwdbwd"] = actor_fwd(B) + 4 * q_fwd(B) + 2 * bwd(B, q, first_input_grad=False)

    # actor update: policy forward, two Q forwards (parameters frozen), backward through both Qs
    # (input gradients only, down to the action part of the first-layer input), then the policy
    # backward (two heads with weights and input gradients, trunk L2 both, trunk L1 weights only)
    q_back = bwd(B, q, first_input_grad=True, weights=False)
    pol_back = 2 * bwd(B, heads, first_input_grad=True) + bwd(B, trunk, first_input_grad=False)
    out["actor_fwdbwd"] = actor_fwd(B) + 2 * q_fwd(B) + 2 * q_back + pol_back

    # Polyak: two operations per parameter of both online critics
    out["target_update_elementwise_ops"] = 2 * (2 * n_params(q))
    return out


def td3_analytic(obs, act, hidden, B):
    h1, h2 = hidden
    pi = dims(obs, h1, h2, act)
    q = dims(obs + act, h1, h2, 1)

    out = {}
    out["actor_forward_bs1"] = fwd(1, pi)
    # critic update: target actor forward (no grad), two target Qs, two online Qs fwd + bwd
    out["critic_fwdbwd"] = fwd(B, pi) + 4 * fwd(B, q) + 2 * bwd(B, q, first_input_grad=False)
    # actor update: policy forward, one Q forward (frozen), Q input gradient, policy backward
    out["actor_fwdbwd"] = (
        fwd(B, pi) + fwd(B, q)
        + bwd(B, q, first_input_grad=True, weights=False)
        + bwd(B, pi, first_input_grad=False)
    )
    # Polyak: two operations per parameter of the actor and both critics
    out["target_update_elementwise_ops"] = 2 * (n_params(pi) + 2 * n_params(q))
    return out


# ------------------------------------------------------------------------
# MBPO dynamics ensemble
# ------------------------------------------------------------------------

def mbpo_dynamics_analytic(obs, act, ens, mh, mb):
    trunk = dims(obs + act, *mh)
    heads = dims(mh[-1], obs + 1)  # mean head; the log-variance head has the same shape
    member_fwd = lambda rows: fwd(rows, trunk) + 2 * fwd(rows, heads)
    out = {}
    out["dynamics_member_fwdbwd"] = (
        member_fwd(mb)
        + bwd(mb, trunk, first_input_grad=False)
        + 2 * bwd(mb, heads, first_input_grad=True)
    )
    out["dynamics_ensemble_forward_all_bs1"] = ens * member_fwd(1)
    return out


# ------------------------------------------------------------------------
# TD-MPC2
# ------------------------------------------------------------------------

def tdmpc2_analytic(rec, sig):
    cfg = TDMPC2Config()
    obs, act = rec["obs_dim"], rec["act_dim"]
    B, H = rec["batch_size"], rec["horizon"]
    S, P, K = rec["num_samples"], rec["num_pi_trajs"], rec["num_q"]
    iters, episodic = rec["iterations"], rec["episodic"]
    mlp = rec["mlp_dim"]
    m = re.search(r"enc(\d+)x(\d+)_mlp(\d+)_lat(\d+)", sig)
    n_enc, enc_dim, _, lat = (int(x) for x in m.groups())
    nb = cfg.num_bins

    enc = dims(obs, *([enc_dim] * max(n_enc - 1, 1)), lat)
    dyn = dims(lat + act, mlp, mlp, lat)
    rew = dims(lat + act, mlp, mlp, nb)
    term = dims(lat, mlp, mlp, 1)
    pi = dims(lat, mlp, mlp, 2 * act)
    q = dims(lat + act, mlp, mlp, nb)  # one Q network

    # ---- planning call (agent.act), batch sizes as in TDMPC2Agent.act ----
    plan = fwd(1, enc)
    plan += H * fwd(P, pi) + (H - 1) * fwd(P, dyn)  # policy-seeded trajectories
    per_value = H * (fwd(S, rew) + fwd(S, dyn) + (fwd(S, term) if episodic else 0))
    per_value += fwd(S, pi) + K * fwd(S, q)  # terminal policy action and all Q-networks
    plan += iters * per_value

    # ---- world-model step (critic_update) ----
    HB = H * B
    wm = fwd(HB, enc)  # encode(obs[1:]), no grad
    wm += fwd(HB, pi)  # next-state policy action, no grad
    wm += K * fwd(HB, q)  # all target Q-networks, no grad
    wm += fwd(B, enc)  # encode(obs[0])
    wm += H * fwd(B, dyn)  # unrolled dynamics
    wm += K * fwd(HB, q)  # all online Q-networks
    wm += fwd(HB, rew)
    if episodic:
        wm += fwd(HB, term)
    # backward: encoder of obs[0] (weights; no input gradient), dynamics (weights + inputs),
    # reward and Q-networks (weights + inputs), termination head (weights + inputs)
    wm += bwd(B, enc, first_input_grad=False)
    wm += H * bwd(B, dyn, first_input_grad=True)
    wm += K * bwd(HB, q, first_input_grad=True)
    wm += bwd(HB, rew, first_input_grad=True)
    if episodic:
        wm += bwd(HB, term, first_input_grad=True)

    # ---- policy-prior step (actor_update) on (H + 1) * B latents ----
    R = (H + 1) * B
    pr = fwd(R, pi) + K * fwd(R, q)
    # the Q-networks are frozen (inputs only); the loss selects two of them at random, but
    # autograd still runs the backward pass of every network on the (zero) gradient
    pr += K * bwd(R, q, first_input_grad=True, weights=False)
    pr += bwd(R, pi, first_input_grad=False)

    return {
        "rollout_per_env_step": plan,
        "gradient_update_critic": wm,
        "gradient_update_actor": pr,
        # Polyak: two operations per parameter of the Q-ensemble; each Q-network also has the
        # weight and bias of its two LayerNorms (2 * mlp each) besides the dense layers
        "gradient_update_target_elementwise_ops": 2 * K * (n_params(q) + 2 * 2 * mlp),
    }


# ------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------

def main() -> int:
    with open(JSON_PATH) as f:
        data = json.load(f)

    rows = []  # (algo, env, signature, quantity, analytic, measured)

    for algo in ("sac", "td3", "mbpo"):
        for env, sigs in data[algo].items():
            for sig, rec in sigs.items():
                obs, act, hid, B = rec["obs_dim"], rec["act_dim"], rec["hidden_sizes"], rec["batch_size"]
                fn = td3_analytic if algo == "td3" else sac_analytic
                for key, val in fn(obs, act, hid, B).items():
                    rows.append((algo, env, sig, key, val, rec[key]))
                if algo == "mbpo":
                    d = mbpo_dynamics_analytic(
                        obs, act, rec["ensemble_size"], rec["model_hidden_sizes"], rec["model_train_batch_size"]
                    )
                    for key, val in d.items():
                        rows.append((algo, env, sig, key, val, rec[key]))

    for env, sigs in data["tdmpc2"].items():
        for sig, rec in sigs.items():
            for key, val in tdmpc2_analytic(rec, sig).items():
                rows.append(("tdmpc2", env, sig, key, val, rec[key]))

    n_bad = 0
    print(f"{'algo':7s} {'env':15s} {'signature':72s} {'quantity':36s} {'analytic':>16s} {'measured':>16s} {'rel.diff':>10s}")
    for algo, env, sig, key, ana, mea in rows:
        rel = (ana - mea) / mea if mea else (0.0 if ana == 0 else float("inf"))
        flag = "" if rel == 0 else "  <-- DIFFERS"
        n_bad += rel != 0
        print(f"{algo:7s} {env:15s} {sig:72s} {key:36s} {ana:16d} {mea:16d} {rel:10.3g}{flag}")

    print(f"\n{len(rows)} comparisons, {n_bad} differ")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
