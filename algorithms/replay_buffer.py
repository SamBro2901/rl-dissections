"""Fixed-capacity circular replay buffer, numpy-backed (fast, no torch overhead
until the sampled batch is converted to tensors at train-step time -- keeps
the 'buffer sample' segment's energy cost isolated from tensor/GPU transfer,
which you may want to sub-segment further later)."""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr, 0] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, action, reward, next_obs, done):
        """Vectorized insert of a batch of transitions. Used by MBPO's branched
        model rollouts, where thousands of synthetic transitions are generated
        per epoch -- inserting them one at a time in a Python loop would
        dominate wall-clock cost. Behaves as a no-op for n == 0."""
        n = obs.shape[0]
        if n == 0:
            return
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx, 0] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx, 0] = done.astype(np.float32)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size
