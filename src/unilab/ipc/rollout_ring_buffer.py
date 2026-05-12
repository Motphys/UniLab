"""Shared rollout IPC ring buffer for APPO / async PPO."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")

_FIELD_SHAPES = {
    "obs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, od),
    "critic": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, cd),
    "actions": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, ad),
    "log_probs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "rewards": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "dones": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "truncated": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "last_obs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, od),
    "last_critic": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, cd),
}


class RolloutRingBuffer:
    """N-slot shared-memory ring buffer for raw rollout payloads."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        *,
        critic_dim: int = 0,
        num_slots: int = 4,
        create: bool = True,
        shm_name_prefix: Dict[str, str] | None = None,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_dim = critic_dim
        self.num_slots = num_slots

        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}

        fields_to_allocate = {k: v for k, v in _FIELD_SHAPES.items()}
        if critic_dim == 0:
            fields_to_allocate.pop("critic", None)
            fields_to_allocate.pop("last_critic", None)

        for field, shape_fn in fields_to_allocate.items():
            shape = shape_fn(
                num_slots,
                num_envs,
                num_steps,
                obs_dim,
                action_dim,
                critic_dim,
            )
            nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize

            if create:
                shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            else:
                assert shm_name_prefix is not None, "shm_name_prefix required when create=False"
                shm = shared_memory.SharedMemory(name=shm_name_prefix[field], create=False)

            self._shm_blocks[field] = shm
            self._arrays[field] = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

        if create:
            self._write_ptr = _SPAWN_CTX.Value("l", 0)
            self._read_ptr = _SPAWN_CTX.Value("l", 0)

    @property
    def name(self) -> Dict[str, str]:
        return {field: shm.name for field, shm in self._shm_blocks.items()}

    def attach_sync_primitives(self, write_ptr, read_ptr) -> None:
        self._write_ptr = write_ptr
        self._read_ptr = read_ptr

    @property
    def write_slot(self) -> int:
        return int(self._write_ptr.value) % self.num_slots

    @property
    def write_buffer(self) -> Dict[str, np.ndarray]:
        s = self.write_slot
        return {field: arr[s] for field, arr in self._arrays.items()}

    def signal_write_done(self) -> None:
        with self._write_ptr.get_lock():
            self._write_ptr.value += 1

    def available(self) -> int:
        return int(self._write_ptr.value) - int(self._read_ptr.value)

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while self.available() == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    @property
    def read_slot(self) -> int:
        return int(self._read_ptr.value) % self.num_slots

    def read_torch(self, device: str) -> dict:
        import torch

        s = self.read_slot
        return {
            field: torch.from_numpy(arr[s].copy()).to(device) for field, arr in self._arrays.items()
        }

    def advance_read(self) -> None:
        with self._read_ptr.get_lock():
            rp = self._read_ptr.value + 1
            wp = self._write_ptr.value
            if wp - rp > self.num_slots:
                rp = wp - self.num_slots + 1
            self._read_ptr.value = rp

    def cleanup(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def close(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass
