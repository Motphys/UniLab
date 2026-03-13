"""Minimal zero-copy replay buffer for off-policy RL."""

import torch
from typing import Dict


class ReplayBuffer:
    """Shared tensor replay buffer with device-adaptive sampling.

    Design:
    - Host tensors shared via .share_memory_() for IPC
    - CUDA: lazy sync to GPU on sample() call
    - MPS: direct access (unified memory)
    - CPU: direct access
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str):
        self.capacity = capacity
        self.device = device
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        # Single packed host tensor (IPC); layout: [obs | next_obs | actions | rew | done | trunc]
        total_dim = 2 * obs_dim + action_dim + 3
        self._storage = torch.zeros(capacity, total_dim).share_memory_()

        c = 0
        self._obs_sl    = slice(c, c + obs_dim);    c += obs_dim
        self._nobs_sl   = slice(c, c + obs_dim);    c += obs_dim
        self._act_sl    = slice(c, c + action_dim); c += action_dim
        self._rew_col   = c;  c += 1
        self._done_col  = c;  c += 1
        self._trunc_col = c

        self.ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.size = torch.zeros(1, dtype=torch.int64).share_memory_()

        # CUDA: GPU cache for zero-copy sampling
        if device == "cuda":
            self.obs_gpu = torch.empty(capacity, obs_dim, device='cuda')
            self.next_obs_gpu = torch.empty(capacity, obs_dim, device='cuda')
            self.actions_gpu = torch.empty(capacity, action_dim, device='cuda')
            self.rewards_gpu = torch.empty(capacity, device='cuda')
            self.dones_gpu = torch.empty(capacity, device='cuda')
            self.truncated_gpu = torch.empty(capacity, device='cuda')
            self._gpu_synced_ptr = 0

    # ---------------------------------------------------------------------------
    # Column-view properties — maintain backward-compat for CUDA path and tests
    # ---------------------------------------------------------------------------

    @property
    def obs(self):
        return self._storage[:, self._obs_sl]

    @property
    def next_obs(self):
        return self._storage[:, self._nobs_sl]

    @property
    def actions(self):
        return self._storage[:, self._act_sl]

    @property
    def rewards(self):
        return self._storage[:, self._rew_col]

    @property
    def dones(self):
        return self._storage[:, self._done_col]

    @property
    def truncated(self):
        return self._storage[:, self._trunc_col]

    def add(self, obs, actions, rewards, next_obs, dones, truncated):
        """Add batch (called by collector)."""
        n = obs.shape[0]
        idx = int(self.ptr[0]) % self.capacity

        row = torch.cat([
            obs, next_obs, actions,
            rewards.unsqueeze(1), dones.unsqueeze(1), truncated.unsqueeze(1),
        ], dim=1)

        if idx + n <= self.capacity:
            self._storage[idx:idx+n] = row
        else:
            split = self.capacity - idx
            self._storage[idx:] = row[:split]
            self._storage[:n-split] = row[split:]

        self.ptr[0] += n
        self.size[0] = min(int(self.size[0]) + n, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch (called by learner)."""
        size = int(self.size[0])
        indices = torch.randint(0, size, (batch_size,))

        if self.device == "cuda":
            # Lazy sync: only sync new data since last sample
            ptr = int(self.ptr[0])
            if ptr > self._gpu_synced_ptr:
                delta = ptr - self._gpu_synced_ptr
                idx = self._gpu_synced_ptr % self.capacity

                if idx + delta <= self.capacity:
                    self.obs_gpu[idx:idx+delta].copy_(self.obs[idx:idx+delta], non_blocking=True)
                    self.next_obs_gpu[idx:idx+delta].copy_(self.next_obs[idx:idx+delta], non_blocking=True)
                    self.actions_gpu[idx:idx+delta].copy_(self.actions[idx:idx+delta], non_blocking=True)
                    self.rewards_gpu[idx:idx+delta].copy_(self.rewards[idx:idx+delta], non_blocking=True)
                    self.dones_gpu[idx:idx+delta].copy_(self.dones[idx:idx+delta], non_blocking=True)
                    self.truncated_gpu[idx:idx+delta].copy_(self.truncated[idx:idx+delta], non_blocking=True)
                else:
                    split = self.capacity - idx
                    self.obs_gpu[idx:].copy_(self.obs[idx:], non_blocking=True)
                    self.obs_gpu[:delta-split].copy_(self.obs[:delta-split], non_blocking=True)
                    self.next_obs_gpu[idx:].copy_(self.next_obs[idx:], non_blocking=True)
                    self.next_obs_gpu[:delta-split].copy_(self.next_obs[:delta-split], non_blocking=True)
                    self.actions_gpu[idx:].copy_(self.actions[idx:], non_blocking=True)
                    self.actions_gpu[:delta-split].copy_(self.actions[:delta-split], non_blocking=True)
                    self.rewards_gpu[idx:].copy_(self.rewards[idx:], non_blocking=True)
                    self.rewards_gpu[:delta-split].copy_(self.rewards[:delta-split], non_blocking=True)
                    self.dones_gpu[idx:].copy_(self.dones[idx:], non_blocking=True)
                    self.dones_gpu[:delta-split].copy_(self.dones[:delta-split], non_blocking=True)
                    self.truncated_gpu[idx:].copy_(self.truncated[idx:], non_blocking=True)
                    self.truncated_gpu[:delta-split].copy_(self.truncated[:delta-split], non_blocking=True)

                self._gpu_synced_ptr = ptr

            indices = indices.to('cuda')
            return {
                "obs": self.obs_gpu[indices],
                "actions": self.actions_gpu[indices],
                "rewards": self.rewards_gpu[indices],
                "next_obs": self.next_obs_gpu[indices],
                "dones": self.dones_gpu[indices],
                "truncated": self.truncated_gpu[indices],
            }
        else:
            chunk = self._storage[indices].to(self.device)
            return {
                "obs":       chunk[:, self._obs_sl],
                "next_obs":  chunk[:, self._nobs_sl],
                "actions":   chunk[:, self._act_sl],
                "rewards":   chunk[:, self._rew_col],
                "dones":     chunk[:, self._done_col],
                "truncated": chunk[:, self._trunc_col],
            }
