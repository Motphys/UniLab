"""Shared on-policy rollout storage for APPO / async PPO."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")

# Fields stored per rollout and their shape constructors.
# Shape: (num_slots, num_envs, num_steps, *trailing) for time-series fields,
#        (num_slots, num_envs, obs_dim) for last_obs.
_FIELD_SHAPES = {
    "obs": lambda ns_slots, ne, ns, od, ad: (ns_slots, ne, ns, od),
    "actions": lambda ns_slots, ne, ns, od, ad: (ns_slots, ne, ns, ad),
    "log_probs": lambda ns_slots, ne, ns, od, ad: (ns_slots, ne, ns),
    "rewards": lambda ns_slots, ne, ns, od, ad: (ns_slots, ne, ns),
    "dones": lambda ns_slots, ne, ns, od, ad: (ns_slots, ne, ns),
    "truncated": lambda ns_slots, ne, ns, od, ad: (ns_slots, ne, ns),
    "last_obs": lambda ns_slots, ne, ns, od, ad: (ns_slots, ne, od),
}


class SharedOnPolicyStorage:
    """N-slot ring-buffer shared-memory store for on-policy rollouts.

    The writer (collector subprocess) fills the current write slot and calls
    ``signal_write_done()`` — this atomically advances ``_write_ptr`` and
    returns immediately (no blocking).  The reader (learner) calls
    ``wait_for_data()``, reads with ``read_torch()``, then calls
    ``advance_read()`` to advance ``_read_ptr``.

    When the collector is faster than the learner the buffer can fill up.
    In that case the collector continues writing and the oldest unread slots
    are silently overwritten (IMPALA/APPO overwrite semantics).  The learner
    detects and skips overwritten slots inside ``advance_read()``.
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        *,
        num_slots: int = 4,
        create: bool = True,
        shm_name_prefix: Dict[str, str] | None = None,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_slots = num_slots

        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}

        for field, shape_fn in _FIELD_SHAPES.items():
            shape = shape_fn(num_slots, num_envs, num_steps, obs_dim, action_dim)
            nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize

            if create:
                shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            else:
                assert shm_name_prefix is not None, "shm_name_prefix required when create=False"
                shm = shared_memory.SharedMemory(name=shm_name_prefix[field], create=False)

            self._shm_blocks[field] = shm
            self._arrays[field] = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

        if create:
            # Monotonically increasing write / read pointers.
            # slot index = ptr % num_slots
            self._write_ptr = _SPAWN_CTX.Value("l", 0)
            self._read_ptr = _SPAWN_CTX.Value("l", 0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> Dict[str, str]:
        """Return {field: shm_name} dict — pass as shm_name_prefix to attach."""
        return {field: shm.name for field, shm in self._shm_blocks.items()}

    # ------------------------------------------------------------------
    # Sync primitives (attached in worker subprocess)
    # ------------------------------------------------------------------

    def attach_sync_primitives(self, write_ptr, read_ptr) -> None:
        """Called in the worker to attach the sync primitives from the parent."""
        self._write_ptr = write_ptr
        self._read_ptr = read_ptr

    # ------------------------------------------------------------------
    # Writer API (collector subprocess)
    # ------------------------------------------------------------------

    @property
    def write_slot(self) -> int:
        return self._write_ptr.value % self.num_slots

    @property
    def write_buffer(self) -> Dict[str, np.ndarray]:
        """Return numpy arrays for the current write slot."""
        s = self.write_slot
        return {field: arr[s] for field, arr in self._arrays.items()}

    def signal_write_done(self) -> None:
        """Atomically advance write pointer (non-blocking)."""
        with self._write_ptr.get_lock():
            self._write_ptr.value += 1

    # ------------------------------------------------------------------
    # Reader API (learner / main process)
    # ------------------------------------------------------------------

    def available(self) -> int:
        """Number of slots ready to consume."""
        return self._write_ptr.value - self._read_ptr.value

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        """Spin-wait (with sleep) until at least one slot is available."""
        import time

        deadline = time.monotonic() + timeout
        while self.available() == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    @property
    def read_slot(self) -> int:
        return self._read_ptr.value % self.num_slots

    def read_torch(self, device: str) -> dict:
        """Copy current read slot into CPU tensors and move to *device*."""
        import torch

        s = self.read_slot
        return {
            field: torch.from_numpy(arr[s].copy()).to(device)
            for field, arr in self._arrays.items()
        }

    def advance_read(self) -> None:
        """Advance read pointer, skipping any slots overwritten by the collector."""
        with self._read_ptr.get_lock():
            rp = self._read_ptr.value + 1
            wp = self._write_ptr.value
            # If the collector has run far ahead, fast-forward past overwritten slots.
            if wp - rp > self.num_slots:
                rp = wp - self.num_slots + 1
            self._read_ptr.value = rp

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Close and unlink all shared memory (call from owner process)."""
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def close(self) -> None:
        """Close handles without unlinking (call from attached processes)."""
        for shm in self._shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass
