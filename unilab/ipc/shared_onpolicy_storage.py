"""Shared on-policy rollout storage for APPO / async PPO."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")

# Fields stored per rollout and their shape constructors.
# Shape: (num_envs, num_steps, *trailing) for time-series fields,
#        (num_envs, obs_dim) for last_obs.
_FIELD_SHAPES = {
    "obs": lambda ne, ns, od, ad: (ne, ns, od),
    "actions": lambda ne, ns, od, ad: (ne, ns, ad),
    "log_probs": lambda ne, ns, od, ad: (ne, ns),
    "rewards": lambda ne, ns, od, ad: (ne, ns),
    "dones": lambda ne, ns, od, ad: (ne, ns),
    "truncated": lambda ne, ns, od, ad: (ne, ns),
    "last_obs": lambda ne, ns, od, ad: (ne, od),
}


class SharedOnPolicyStorage:
    """Single-buffer shared-memory store for on-policy rollouts.

    The writer (collector subprocess) fills the buffer and calls
    ``signal_write_done()``.  The reader (learner) calls
    ``wait_for_data()``, reads with ``read_torch()``, then calls
    ``advance_read()`` to clear the signal and allow the next write.
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        *,
        create: bool = True,
        shm_name_prefix: Dict[str, str] | None = None,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}

        for field, shape_fn in _FIELD_SHAPES.items():
            shape = shape_fn(num_envs, num_steps, obs_dim, action_dim)
            nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize

            if create:
                shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            else:
                assert shm_name_prefix is not None, "shm_name_prefix required when create=False"
                shm = shared_memory.SharedMemory(name=shm_name_prefix[field], create=False)

            self._shm_blocks[field] = shm
            self._arrays[field] = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

        if create:
            self._ready: mp.Event = _SPAWN_CTX.Event()
            self._write_idx = _SPAWN_CTX.Value("i", 0)
            self._read_idx = _SPAWN_CTX.Value("i", 0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> Dict[str, str]:
        """Return {field: shm_name} dict — pass as shm_name_prefix to attach."""
        return {field: shm.name for field, shm in self._shm_blocks.items()}

    @property
    def write_buffer(self) -> Dict[str, np.ndarray]:
        """Return raw numpy arrays for the worker to write into."""
        return self._arrays

    # ------------------------------------------------------------------
    # Sync primitives (attached in worker subprocess)
    # ------------------------------------------------------------------

    def attach_sync_primitives(self, write_idx, read_idx, ready) -> None:
        """Called in the worker to attach the sync primitives from the parent."""
        self._write_idx = write_idx
        self._read_idx = read_idx
        self._ready = ready

    # ------------------------------------------------------------------
    # Writer API (collector subprocess)
    # ------------------------------------------------------------------

    def signal_write_done(self) -> None:
        """Signal that the buffer has been fully written."""
        with self._write_idx.get_lock():
            self._write_idx.value += 1
        self._ready.set()

    # ------------------------------------------------------------------
    # Reader API (learner / main process)
    # ------------------------------------------------------------------

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        """Block until data is available; return True if data arrived."""
        return self._ready.wait(timeout=timeout)

    def read_torch(self, device: str) -> dict:
        """Copy all fields into CPU tensors and move to *device*."""
        import torch

        return {
            field: torch.from_numpy(arr.copy()).to(device)
            for field, arr in self._arrays.items()
        }

    def advance_read(self) -> None:
        """Clear the ready flag so the writer can write the next rollout."""
        with self._read_idx.get_lock():
            self._read_idx.value += 1
        self._ready.clear()

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
