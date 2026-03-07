"""Safe replay buffer with guaranteed data integrity in multiprocessing."""

from __future__ import annotations
import numpy as np
import torch
from unilab.ipc.shared_buffer import SharedReplayBuffer


class SafeReplayBuffer(SharedReplayBuffer):
    """Replay buffer with safe multiprocessing data transfer.

    Extends SharedReplayBuffer with explicit synchronization to prevent
    race conditions in async data transfer (macOS NaN issue fix).
    """

    def sample_torch(self, batch_size: int, device: str = "cpu"):
        """Sample batch with guaranteed data integrity.

        Three-layer protection:
        1. Copy data within lock
        2. Non-blocking transfer for performance
        3. Explicit sync to ensure completion
        """
        # Layer 1: Copy in lock
        with self._lock:
            current_size = int(self._meta[1])
            indices = np.random.randint(0, current_size, size=batch_size)
            obs_copy = self.obs[indices].copy()
            actions_copy = self.actions[indices].copy()
            rewards_copy = self.rewards[indices].copy()
            next_obs_copy = self.next_obs[indices].copy()
            dones_copy = self.dones[indices].copy()
            truncated_copy = self.truncated[indices].copy()

        # Layer 2: Non-blocking transfer
        result = {
            "obs": torch.from_numpy(obs_copy).to(device, non_blocking=True),
            "actions": torch.from_numpy(actions_copy).to(device, non_blocking=True),
            "rewards": torch.from_numpy(rewards_copy).to(device, non_blocking=True),
            "next_obs": torch.from_numpy(next_obs_copy).to(device, non_blocking=True),
            "dones": torch.from_numpy(dones_copy).to(device, non_blocking=True),
            "truncated": torch.from_numpy(truncated_copy).to(device, non_blocking=True),
        }

        # Layer 3: Explicit sync
        if device != "cpu":
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

        return result
