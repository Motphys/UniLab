"""Optimized GPU buffer with on-demand sync."""

import torch
import numpy as np
from typing import Dict


class OptimizedGPUReplayBuffer:
    """GPU buffer with on-demand incremental sync."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str):
        self.capacity = capacity
        self.device = device

        # GPU tensors
        self.obs = torch.empty(capacity, obs_dim, dtype=torch.float32, device=device)
        self.next_obs = torch.empty(capacity, obs_dim, dtype=torch.float32, device=device)
        self.actions = torch.empty(capacity, action_dim, dtype=torch.float32, device=device)
        self.rewards = torch.empty(capacity, dtype=torch.float32, device=device)
        self.dones = torch.empty(capacity, dtype=torch.float32, device=device)
        self.truncated = torch.empty(capacity, dtype=torch.float32, device=device)

        self._ptr = 0
        self._size = 0
        self._last_host_ptr = 0

    def sync_new_data(self, shared_buffer) -> int:
        """Sync only new data since last call."""
        # Read host ptr (minimal lock time)
        with shared_buffer._lock:
            host_ptr = int(shared_buffer._meta[0])

        new_samples = host_ptr - self._last_host_ptr
        if new_samples <= 0:
            return 0

        # Read new data (outside lock for most of the time)
        start_idx = self._last_host_ptr % shared_buffer.capacity

        if start_idx + new_samples <= shared_buffer.capacity:
            # Contiguous read
            obs_np = shared_buffer.obs[start_idx:start_idx + new_samples]
            next_obs_np = shared_buffer.next_obs[start_idx:start_idx + new_samples]
            actions_np = shared_buffer.actions[start_idx:start_idx + new_samples]
            rewards_np = shared_buffer.rewards[start_idx:start_idx + new_samples]
            dones_np = shared_buffer.dones[start_idx:start_idx + new_samples]
            truncated_np = shared_buffer.truncated[start_idx:start_idx + new_samples]
        else:
            # Wrap-around
            first = shared_buffer.capacity - start_idx
            obs_np = np.concatenate([
                shared_buffer.obs[start_idx:],
                shared_buffer.obs[:new_samples - first]
            ])
            next_obs_np = np.concatenate([
                shared_buffer.next_obs[start_idx:],
                shared_buffer.next_obs[:new_samples - first]
            ])
            actions_np = np.concatenate([
                shared_buffer.actions[start_idx:],
                shared_buffer.actions[:new_samples - first]
            ])
            rewards_np = np.concatenate([
                shared_buffer.rewards[start_idx:],
                shared_buffer.rewards[:new_samples - first]
            ])
            dones_np = np.concatenate([
                shared_buffer.dones[start_idx:],
                shared_buffer.dones[:new_samples - first]
            ])
            truncated_np = np.concatenate([
                shared_buffer.truncated[start_idx:],
                shared_buffer.truncated[:new_samples - first]
            ])

        # Async transfer to GPU (non-blocking for CUDA only)
        non_blocking = self.device.startswith('cuda')
        obs_gpu = torch.from_numpy(obs_np).to(self.device, non_blocking=non_blocking)
        next_obs_gpu = torch.from_numpy(next_obs_np).to(self.device, non_blocking=non_blocking)
        actions_gpu = torch.from_numpy(actions_np).to(self.device, non_blocking=non_blocking)
        rewards_gpu = torch.from_numpy(rewards_np).to(self.device, non_blocking=non_blocking)
        dones_gpu = torch.from_numpy(dones_np).to(self.device, non_blocking=non_blocking)
        truncated_gpu = torch.from_numpy(truncated_np).to(self.device, non_blocking=non_blocking)

        # Write to GPU buffer
        gpu_start = self._ptr % self.capacity
        if gpu_start + new_samples <= self.capacity:
            self.obs[gpu_start:gpu_start + new_samples] = obs_gpu
            self.next_obs[gpu_start:gpu_start + new_samples] = next_obs_gpu
            self.actions[gpu_start:gpu_start + new_samples] = actions_gpu
            self.rewards[gpu_start:gpu_start + new_samples] = rewards_gpu
            self.dones[gpu_start:gpu_start + new_samples] = dones_gpu
            self.truncated[gpu_start:gpu_start + new_samples] = truncated_gpu
        else:
            first = self.capacity - gpu_start
            self.obs[gpu_start:] = obs_gpu[:first]
            self.obs[:new_samples - first] = obs_gpu[first:]
            self.next_obs[gpu_start:] = next_obs_gpu[:first]
            self.next_obs[:new_samples - first] = next_obs_gpu[first:]
            self.actions[gpu_start:] = actions_gpu[:first]
            self.actions[:new_samples - first] = actions_gpu[first:]
            self.rewards[gpu_start:] = rewards_gpu[:first]
            self.rewards[:new_samples - first] = rewards_gpu[first:]
            self.dones[gpu_start:] = dones_gpu[:first]
            self.dones[:new_samples - first] = dones_gpu[first:]
            self.truncated[gpu_start:] = truncated_gpu[:first]
            self.truncated[:new_samples - first] = truncated_gpu[first:]

        self._ptr += new_samples
        self._size = min(self._size + new_samples, self.capacity)
        self._last_host_ptr = host_ptr

        return new_samples

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Zero-copy GPU sampling."""
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size} samples, need {batch_size}")

        indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
            "truncated": self.truncated[indices],
        }

    @property
    def size(self) -> int:
        return self._size
