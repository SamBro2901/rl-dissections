"""
Environment-specific termination ("done") heuristics for model-generated
transitions, matching the per-environment `termination_fn`s from the MBPO
reference implementation (Janner et al., 2019,
https://github.com/JannerM/mbpo/tree/master/mbpo/static), which in turn
mirror each Gymnasium MuJoCo environment's built-in `terminate_when_unhealthy`
early-termination checks.

The learned dynamics model (algorithms/dynamics_model.py) only predicts
(next_obs, reward) -- it has no notion of "the robot fell over". Without a
termination function, branched model rollouts would keep "walking" through
physically invalid states (e.g. a fallen-over Hopper) instead of ending the
episode there, which the paper's ablations show significantly hurts
performance on environments that have early termination in the real env.

These assume the Gymnasium MuJoCo default observation layout
(`exclude_current_positions_from_observation=True`, the default for all
`gym.make(...)` calls here), under which index 0 of the observation is the
torso height ("z") for the walking/running envs below.
"""
from __future__ import annotations

import numpy as np


def _never_done(next_obs: np.ndarray) -> np.ndarray:
    return np.zeros((next_obs.shape[0],), dtype=bool)


def _hopper_done(next_obs: np.ndarray) -> np.ndarray:
    height, angle = next_obs[:, 0], next_obs[:, 1]
    healthy = (
        np.isfinite(next_obs).all(axis=-1)
        & (np.abs(next_obs[:, 1:]) < 100).all(axis=-1)
        & (height > 0.7)
        & (np.abs(angle) < 0.2)
    )
    return ~healthy


def _walker2d_done(next_obs: np.ndarray) -> np.ndarray:
    height, angle = next_obs[:, 0], next_obs[:, 1]
    healthy = (height > 0.8) & (height < 2.0) & (angle > -1.0) & (angle < 1.0)
    return ~healthy


def _ant_done(next_obs: np.ndarray) -> np.ndarray:
    z = next_obs[:, 0]
    healthy = np.isfinite(next_obs).all(axis=-1) & (z >= 0.2) & (z <= 1.0)
    return ~healthy


def _humanoid_done(next_obs: np.ndarray) -> np.ndarray:
    z = next_obs[:, 0]
    return (z < 1.0) | (z > 2.0)


# Matched against env_id by lowercase substring (see get_termination_fn).
# HalfCheetah/InvertedPendulum/InvertedDoublePendulum never terminate early
# in Gymnasium, so `_never_done` is the *correct* function for them, not a
# fallback.
_REGISTRY = {
    "halfcheetah": _never_done,
    "hopper": _hopper_done,
    "walker2d": _walker2d_done,
    "ant": _ant_done,
    "humanoid": _humanoid_done,
    "invertedpendulum": _never_done,
    "inverteddoublependulum": _never_done,
}


def get_termination_fn(env_id: str, logger=None):
    """Returns a function `next_obs: np.ndarray[N, obs_dim] -> done: np.ndarray[N] (bool)`.

    Falls back to `_never_done` (with a logged warning) for unrecognized
    environments -- correct for non-terminating tasks, but a known-imprecise
    approximation for anything else (model rollouts on that env will run to
    the full scheduled rollout length even through what would be a terminal
    state in the real environment)."""
    key = env_id.lower()
    for name, fn in _REGISTRY.items():
        if name in key:
            return fn
    if logger is not None:
        logger.warning(
            "No known MBPO termination function for env_id=%s; model rollouts "
            "will never terminate early. Add one to algorithms/termination_fns.py "
            "if this environment has early termination in the real env.", env_id,
        )
    return _never_done
