"""Shared on-policy storage for APPO-style algorithms."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")


class SharedOnPolicyStorage:
    """Double-buffered rollout storage for on-policy algorithms."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        *,
        create: bool = True,
        shm_name_prefix: str | None = None,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        _f32 = np.dtype(np.float32).itemsize
        n = num_envs * num_steps
        per_buffer = (
            n * obs_dim * _f32 + n * action_dim * _f32 + n * _f32 * 4 + num_envs * obs_dim * _f32
        )
        total_bytes = 2 * per_buffer

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        else:
            assert shm_name_prefix is not None
            self._shm = shared_memory.SharedMemory(name=shm_name_prefix, create=False)

        self._per_buffer = per_buffer
        self._buffers = [self._make_views(0), self._make_views(per_buffer)]

        if create:
            self._write_idx = _SPAWN_CTX.Value("l", 0)
            self._read_idx = _SPAWN_CTX.Value("l", 0)
            self._ready = [_SPAWN_CTX.Event(), _SPAWN_CTX.Event()]
        else:
            self._write_idx = None
            self._read_idx = None
            self._ready = None

    def attach_sync_primitives(self, write_idx, read_idx, ready_events):
        self._write_idx = write_idx
        self._read_idx = read_idx
        self._ready = ready_events

    def _make_views(self, base_offset: int) -> Dict[str, np.ndarray]:
        buf = self._shm.buf
        n = self.num_envs * self.num_steps
        _f32 = np.dtype(np.float32).itemsize
        offset = base_offset
        views = {}

        views["obs"] = np.ndarray(
            (self.num_envs, self.num_steps, self.obs_dim), dtype=np.float32, buffer=buf[offset:]
        )
        offset += n * self.obs_dim * _f32
        views["actions"] = np.ndarray(
            (self.num_envs, self.num_steps, self.action_dim), dtype=np.float32, buffer=buf[offset:]
        )
        offset += n * self.action_dim * _f32

        for name in ["rewards", "dones", "truncated", "log_probs"]:
            views[name] = np.ndarray(
                (self.num_envs, self.num_steps), dtype=np.float32, buffer=buf[offset:]
            )
            offset += n * _f32

        views["last_obs"] = np.ndarray(
            (self.num_envs, self.obs_dim), dtype=np.float32, buffer=buf[offset:]
        )
        return views

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def write_buffer(self):
        return self._buffers[self._write_idx.value % 2]

    @property
    def read_buffer(self):
        return self._buffers[self._read_idx.value % 2]

    def signal_write_done(self) -> None:
        idx = self._write_idx.value % 2
        self._ready[idx].set()
        self._write_idx.value += 1

    def wait_for_data(self, timeout: float = 30.0) -> bool:
        idx = self._read_idx.value % 2
        result = self._ready[idx].wait(timeout=timeout)
        if result:
            self._ready[idx].clear()
        return result

    def advance_read(self) -> None:
        self._read_idx.value += 1

    def read_torch(self, device: str = "cpu"):
        import torch

        views = self.read_buffer
        return {k: torch.from_numpy(v).to(device) for k, v in views.items()}

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
