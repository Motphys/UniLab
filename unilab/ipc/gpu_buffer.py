"""GPU-resident replay buffer for off-policy RL algorithms."""

import torch
import numpy as np
from typing import Dict


class GPUReplayBuffer:
    """GPU-resident ring buffer with periodic host sync.

    Stores replay data in GPU memory for zero-copy sampling.
    Periodically syncs new data from host SharedReplayBuffer.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # GPU tensors
        self.obs = torch.zeros(capacity, obs_dim, dtype=torch.float32, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, action_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.truncated = torch.zeros(capacity, dtype=torch.float32, device=device)

        # Metadata (CPU scalars, no lock needed)
        self._ptr = 0
        self._size = 0
        self._last_synced_ptr = 0

    def sync_from_host(self, shared_buffer) -> int:
        """Pull new data from host SharedReplayBuffer.

        Args:
            shared_buffer: SharedReplayBuffer instance

        Returns:
            Number of samples synced
        """
        # Read host state (minimal lock time)
        with shared_buffer._lock:
            host_ptr = int(shared_buffer._meta[0])
            host_size = int(shared_buffer._meta[1])

        # Calculate new samples
        new_samples = host_ptr - self._last_synced_ptr
        if new_samples <= 0:
            return 0

        # Determine host read range (handle wrap)
        start = self._last_synced_ptr % shared_buffer.capacity

        if start + new_samples <= shared_buffer.capacity:
            # Contiguous read
            obs_np = shared_buffer.obs[start:start + new_samples]
            next_obs_np = shared_buffer.next_obs[start:start + new_samples]
            actions_np = shared_buffer.actions[start:start + new_samples]
            rewards_np = shared_buffer.rewards[start:start + new_samples]
            dones_np = shared_buffer.dones[start:start + new_samples]
            truncated_np = shared_buffer.truncated[start:start + new_samples]
        else:
            # Wrap-around: read in 2 chunks
            first = shared_buffer.capacity - start
            obs_np = np.concatenate([shared_buffer.obs[start:], shared_buffer.obs[:new_samples - first]])
            next_obs_np = np.concatenate([shared_buffer.next_obs[start:], shared_buffer.next_obs[:new_samples - first]])
            actions_np = np.concatenate([shared_buffer.actions[start:], shared_buffer.actions[:new_samples - first]])
            rewards_np = np.concatenate([shared_buffer.rewards[start:], shared_buffer.rewards[:new_samples - first]])
            dones_np = np.concatenate([shared_buffer.dones[start:], shared_buffer.dones[:new_samples - first]])
            truncated_np = np.concatenate([shared_buffer.truncated[start:], shared_buffer.truncated[:new_samples - first]])

        # Async transfer to GPU
        obs_gpu = torch.from_numpy(obs_np).to(self.device, non_blocking=True)
        next_obs_gpu = torch.from_numpy(next_obs_np).to(self.device, non_blocking=True)
        actions_gpu = torch.from_numpy(actions_np).to(self.device, non_blocking=True)
        rewards_gpu = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        dones_gpu = torch.from_numpy(dones_np).to(self.device, non_blocking=True)
        truncated_gpu = torch.from_numpy(truncated_np).to(self.device, non_blocking=True)

        # Write to GPU buffer (handle wrap)
        gpu_start = self._ptr % self.capacity

        if gpu_start + new_samples <= self.capacity:
            # Contiguous write
            self.obs[gpu_start:gpu_start + new_samples] = obs_gpu
            self.next_obs[gpu_start:gpu_start + new_samples] = next_obs_gpu
            self.actions[gpu_start:gpu_start + new_samples] = actions_gpu
            self.rewards[gpu_start:gpu_start + new_samples] = rewards_gpu
            self.dones[gpu_start:gpu_start + new_samples] = dones_gpu
            self.truncated[gpu_start:gpu_start + new_samples] = truncated_gpu
        else:
            # Wrap-around write
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

        # Sync device
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

        # Update metadata
        self._ptr += new_samples
        self._size = min(self._size + new_samples, self.capacity)
        self._last_synced_ptr = host_ptr

        return new_samples

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Zero-copy GPU sampling.

        Args:
            batch_size: Number of samples to draw

        Returns:
            Dict of tensors on GPU
        """
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size} samples, need {batch_size}")

        # GPU random indexing
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        # Zero-copy views
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
        """Current valid data count."""
        return self._size
