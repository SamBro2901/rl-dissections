"""
Ground-truth check for compute_energy_per_flop.mbpo_fit_flops(), the formula
used for MBPO's dynamics_model_update FLOPs.

  1. Linearity: FlopCounterMode on forward_member fwd+bwd (measure_flops.py's
     own measure_mbpo_dynamics()) at batch 256 and 100 -- the latter must be
     exactly the former x 100/256, which is what justifies per-sample constants.
  2. For MBPOConfig defaults on HalfCheetah-v5 dims and the Ant-v5 override
     (configs/overrides/mbpo_ant.json): build a real EnsembleDynamicsModel,
     fill a ReplayBuffer with N random transitions, run the full, unmodified
     model.fit(buffer) under FlopCounterMode (model_max_train_epochs=3,
     patience high enough never to stop early) and compare the measured total
     with mbpo_fit_flops() evaluated on the returned train_epochs, using the
     constants from flops_per_call.json.

Acceptance: relative difference < 0.5% for every N. Exits non-zero otherwise.
Needs gymnasium[mujoco] for the env dims; CPU is fine (FLOP counts don't
depend on device).

Usage (from repo root):
    python flop_analysis/verify_mbpo_fit_flops.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from algorithms.dynamics_model import EnsembleDynamicsModel  # noqa: E402
from algorithms.replay_buffer import ReplayBuffer  # noqa: E402
from configs.config import MBPOConfig  # noqa: E402
from compute_energy_per_flop import FLOPS_JSON, mbpo_fit_flops  # noqa: E402
from flop_keys import signature  # noqa: E402
from measure_flops import env_dims, measure_mbpo_dynamics  # noqa: E402

DEVICE = torch.device("cpu")
BUFFER_SIZES = (5_000, 12_345, 30_000)
FIT_EPOCHS = 3
REL_TOL = 0.005


def load_config(overrides_path=None):
    overrides = {}
    if overrides_path:
        with open(os.path.join(REPO_ROOT, overrides_path)) as f:
            overrides = json.load(f)
    return MBPOConfig(**overrides)


def check_linearity(obs_dim, act_dim, cfg):
    hidden = tuple(cfg.model_hidden_sizes)
    f256 = measure_mbpo_dynamics(obs_dim, act_dim, cfg.ensemble_size, hidden, 256)["dynamics_member_fwdbwd"]
    f100 = measure_mbpo_dynamics(obs_dim, act_dim, cfg.ensemble_size, hidden, 100)["dynamics_member_fwdbwd"]
    ok = f100 * 256 == f256 * 100
    print(f"  linearity: member fwd+bwd bs256={f256}, bs100={f100}, "
          f"bs256*100/256={f256 * 100 / 256:.1f} -> {'exact' if ok else 'NOT exact'}")
    return ok, f256


def check_fit(label, env_id, overrides_path, flops_json):
    cfg = load_config(overrides_path)
    obs_dim, act_dim, _ = env_dims(env_id)
    print(f"\n{label}: {env_id}, obs_dim={obs_dim}, act_dim={act_dim}, "
          f"ensemble={cfg.ensemble_size}, model_hidden={tuple(cfg.model_hidden_sizes)}, "
          f"batch={cfg.model_train_batch_size}, holdout_ratio={cfg.model_holdout_ratio}")

    lin_ok, f256 = check_linearity(obs_dim, act_dim, cfg)
    algo_config = dataclasses.asdict(cfg)
    sig = signature("mbpo", algo_config)
    flops_key = flops_json["mbpo"][env_id][sig]
    const_ok = flops_key["dynamics_member_fwdbwd"] == f256 or cfg.model_train_batch_size != 256
    print(f"  flops_per_call.json[{sig}]: member_fwdbwd={flops_key['dynamics_member_fwdbwd']}, "
          f"forward_all_bs1={flops_key['dynamics_ensemble_forward_all_bs1']} "
          f"({'matches' if const_ok else 'DIFFERS FROM'} fresh measurement)")

    fit_cfg = dataclasses.replace(cfg, model_max_train_epochs=FIT_EPOCHS, model_train_patience=10_000)
    ok = lin_ok and const_ok
    for n in BUFFER_SIZES:
        rng = np.random.default_rng(n)
        buf = ReplayBuffer(n, obs_dim, act_dim)
        buf.add_batch(
            rng.standard_normal((n, obs_dim)).astype(np.float32),
            rng.uniform(-1, 1, (n, act_dim)).astype(np.float32),
            rng.standard_normal(n).astype(np.float32),
            rng.standard_normal((n, obs_dim)).astype(np.float32),
            np.zeros(n, dtype=bool),
        )
        torch.manual_seed(0)
        np.random.seed(0)
        model = EnsembleDynamicsModel(obs_dim, act_dim, fit_cfg, DEVICE)
        with FlopCounterMode(display=False) as fc:
            diag = model.fit(buf)
        measured = fc.get_total_flops()
        train_f, holdout_f, steps = mbpo_fit_flops(n, diag["train_epochs"], algo_config, flops_key)
        predicted = train_f + holdout_f
        rel = abs(predicted - measured) / measured
        passed = rel < REL_TOL
        ok &= passed
        print(f"  N={n:>6}: train_epochs={diag['train_epochs']} optimizer_steps={steps} "
              f"measured={measured} predicted={predicted} (train={train_f}, holdout={holdout_f}, "
              f"holdout share={100 * holdout_f / predicted:.2f}%) rel diff={rel:.3e} "
              f"-> {'PASS' if passed else 'FAIL'}")
    return ok


def main():
    with open(FLOPS_JSON) as f:
        flops_json = json.load(f)
    ok = check_fit("MBPOConfig defaults", "HalfCheetah-v5", None, flops_json)
    ok &= check_fit("Ant override", "Ant-v5", "configs/overrides/mbpo_ant.json", flops_json)
    print(f"\n{'ALL PASSED' if ok else 'FAILED'} (acceptance: rel diff < {REL_TOL:.1%} for every N)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
