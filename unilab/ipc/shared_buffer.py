"""Shared replay buffer for off-policy RL algorithms."""

from __future__ import annotations
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict
import numpy as np

_SPAWN_CTX = mp.get_context("spawn")


class SharedReplayBuffer:
    """Zero-copy ring buffer in shared memory for (obs, act, rew, next_obs, done, truncated)."""

    _META_INTS = 2  # ptr, size

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        *,
        create: bool = True,
        shm_name: str | None = None,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        _f32 = np.dtype(np.float32).itemsize
        _i32 = np.dtype(np.int32).itemsize
        self._obs_bytes = capacity * obs_dim * _f32
        self._act_bytes = capacity * action_dim * _f32
        self._scalar_bytes = capacity * _f32

        data_bytes = (
            2 * self._obs_bytes + self._act_bytes + 3 * self._scalar_bytes
        )
        meta_bytes = self._META_INTS * _i32
        total_bytes = data_bytes + meta_bytes

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
            self._lock = _SPAWN_CTX.Lock()
        else:
            assert shm_name is not None
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
            self._lock = None

        buf = self._shm.buf
        offset = 0

        self.obs = np.ndarray((capacity, obs_dim), dtype=np.float32, buffer=buf[offset:])
        offset += self._obs_bytes
        self.next_obs = np.ndarray((capacity, obs_dim), dtype=np.float32, buffer=buf[offset:])
        offset += self._obs_bytes
        self.actions = np.ndarray((capacity, action_dim), dtype=np.float32, buffer=buf[offset:])
        offset += self._act_bytes
        self.rewards = np.ndarray((capacity,), dtype=np.float32, buffer=buf[offset:])
        offset += self._scalar_bytes
        self.dones = np.ndarray((capacity,), dtype=np.float32, buffer=buf[offset:])
        offset += self._scalar_bytes
        self.truncated = np.ndarray((capacity,), dtype=np.float32, buffer=buf[offset:])
        offset += self._scalar_bytes

        self._meta = np.ndarray((self._META_INTS,), dtype=np.int32, buffer=buf[offset:])
        if create:
            self._meta[:] = 0
            self.obs[:] = 0.0
            self.next_obs[:] = 0.0
            self.actions[:] = 0.0
            self.rewards[:] = 0.0
            self.dones[:] = 0.0
            self.truncated[:] = 0.0

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def ptr(self) -> int:
        return int(self._meta[0])

    @property
    def size(self) -> int:
        return int(self._meta[1])

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        batch_size = obs.shape[0]
        start = int(self._meta[0]) % self.capacity

        if start + batch_size <= self.capacity:
            self.obs[start : start + batch_size] = obs
            self.next_obs[start : start + batch_size] = next_obs
            self.actions[start : start + batch_size] = actions
            self.rewards[start : start + batch_size] = rewards
            self.dones[start : start + batch_size] = dones
            self.truncated[start : start + batch_size] = truncated
        else:
            first = self.capacity - start
            self.obs[start:] = obs[:first]; self.obs[:batch_size - first] = obs[first:]
            self.next_obs[start:] = next_obs[:first]; self.next_obs[:batch_size - first] = next_obs[first:]
            self.actions[start:] = actions[:first]; self.actions[:batch_size - first] = actions[first:]
            self.rewards[start:] = rewards[:first]; self.rewards[:batch_size - first] = rewards[first:]
            self.dones[start:] = dones[:first]; self.dones[:batch_size - first] = dones[first:]
            self.truncated[start:] = truncated[:first]; self.truncated[:batch_size - first] = truncated[first:]

        with self._lock:
            self._meta[0] += batch_size
            self._meta[1] = min(int(self._meta[1]) + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        with self._lock:
            current_size = int(self._meta[1])
        indices = np.random.randint(0, current_size, size=batch_size)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
            "truncated": self.truncated[indices],
        }

    def sample_torch(self, batch_size: int, device: str = "cpu"):
        import torch
        with self._lock:
            current_size = int(self._meta[1])
        indices = np.random.randint(0, current_size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[indices]).to(device, non_blocking=True),
            "actions": torch.from_numpy(self.actions[indices]).to(device, non_blocking=True),
            "rewards": torch.from_numpy(self.rewards[indices]).to(device, non_blocking=True),
            "next_obs": torch.from_numpy(self.next_obs[indices]).to(device, non_blocking=True),
            "dones": torch.from_numpy(self.dones[indices]).to(device, non_blocking=True),
            "truncated": torch.from_numpy(self.truncated[indices]).to(device, non_blocking=True),
        }

    def sample_mlx(self, batch_size: int):
        import mlx.core as mx
        with self._lock:
            current_size = int(self._meta[1])
        indices = np.random.randint(0, current_size, size=batch_size)
        return {
            "obs": mx.array(self.obs[indices]),
            "actions": mx.array(self.actions[indices]),
            "rewards": mx.array(self.rewards[indices]),
            "next_obs": mx.array(self.next_obs[indices]),
            "dones": mx.array(self.dones[indices]),
            "truncated": mx.array(self.truncated[indices]),
        }

    def utilization(self) -> float:
        return self.size / self.capacity

    def is_nearly_full(self, threshold: float = 0.95) -> bool:
        return self.utilization() >= threshold

    def cleanup(self) -> None:
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass
