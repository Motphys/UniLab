"""Lightweight async runner framework using Python native multiprocessing + shared memory.

Uses zero-copy shared memory for inter-process communication.
Designed for Mac (MPS) with extensibility for NPU.

Key components:
- SharedReplayBuffer: Zero-copy ring buffer in shared memory (off-policy)
- SharedOnPolicyStorage: Double-buffered rollout storage (on-policy / APPO)
- SharedWeightSync: Actor weight synchronization via shared memory
- AsyncRunner: Base class for all async RL algorithms
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# SharedReplayBuffer — zero-copy ring buffer for off-policy algorithms
# ---------------------------------------------------------------------------

class SharedReplayBuffer:
    """Cross-process zero-copy ring buffer for (obs, act, rew, next_obs, done) transitions.

    Uses ``multiprocessing.shared_memory.SharedMemory`` so both the collector
    and learner processes can read/write the same numpy arrays without any
    serialisation overhead.

    ptr and size are stored as int32 values at the END of the shared memory
    block, so both processes see the same counters.
    """

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
            2 * self._obs_bytes       # obs + next_obs
            + self._act_bytes         # actions
            + 2 * self._scalar_bytes  # rewards + dones
        )
        meta_bytes = self._META_INTS * _i32  # ptr, size
        total_bytes = data_bytes + meta_bytes

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        else:
            assert shm_name is not None
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)

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

        # Meta counters — stored in the shared memory block itself
        self._meta = np.ndarray((self._META_INTS,), dtype=np.int32, buffer=buf[offset:])
        if create:
            self._meta[0] = 0  # ptr
            self._meta[1] = 0  # size
            # Zero-initialize all data arrays to prevent garbage data (NaN/Inf)
            # from being sampled before the collector fills them
            self.obs[:] = 0.0
            self.next_obs[:] = 0.0
            self.actions[:] = 0.0
            self.rewards[:] = 0.0
            self.dones[:] = 0.0

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
    ) -> None:
        batch_size = obs.shape[0]
        start = int(self._meta[0]) % self.capacity

        if start + batch_size <= self.capacity:
            self.obs[start : start + batch_size] = obs
            self.next_obs[start : start + batch_size] = next_obs
            self.actions[start : start + batch_size] = actions
            self.rewards[start : start + batch_size] = rewards
            self.dones[start : start + batch_size] = dones
        else:
            first = self.capacity - start
            self.obs[start:] = obs[:first];           self.obs[:batch_size - first] = obs[first:]
            self.next_obs[start:] = next_obs[:first]; self.next_obs[:batch_size - first] = next_obs[first:]
            self.actions[start:] = actions[:first];   self.actions[:batch_size - first] = actions[first:]
            self.rewards[start:] = rewards[:first];   self.rewards[:batch_size - first] = rewards[first:]
            self.dones[start:] = dones[:first];       self.dones[:batch_size - first] = dones[first:]

        self._meta[0] += batch_size
        self._meta[1] = min(int(self._meta[1]) + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_obs": self.next_obs[indices].copy(),
            "dones": self.dones[indices].copy(),
        }

    def sample_torch(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        data = self.sample(batch_size)
        return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in data.items()}

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


# ---------------------------------------------------------------------------
# SharedOnPolicyStorage — double-buffered rollout storage for APPO
# ---------------------------------------------------------------------------

class SharedOnPolicyStorage:
    """Double-buffered rollout storage for on-policy algorithms (APPO).

    Two buffers alternate: while the collector writes to buffer A, the
    learner reads from buffer B, and vice versa.

    Each buffer stores a full rollout: ``(num_envs, num_steps, ...)``.
    """

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

        obs_bytes = n * obs_dim * _f32
        act_bytes = n * action_dim * _f32
        scalar_bytes = n * _f32
        last_obs_bytes = num_envs * obs_dim * _f32

        per_buffer = (
            obs_bytes
            + act_bytes
            + scalar_bytes * 5  # rewards, dones, truncated, log_probs, values
            + last_obs_bytes
        )

        total_bytes = 2 * per_buffer

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        else:
            assert shm_name_prefix is not None
            self._shm = shared_memory.SharedMemory(name=shm_name_prefix, create=False)

        self._per_buffer = per_buffer
        self._buffers = [
            self._make_views(0),
            self._make_views(per_buffer),
        ]

        if create:
            self._write_idx = mp.Value("i", 0)
            self._read_idx = mp.Value("i", 0)
            self._ready = [mp.Event(), mp.Event()]
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
        views: Dict[str, np.ndarray] = {}

        views["obs"] = np.ndarray(
            (self.num_envs, self.num_steps, self.obs_dim),
            dtype=np.float32, buffer=buf[offset:]
        )
        offset += n * self.obs_dim * _f32

        views["actions"] = np.ndarray(
            (self.num_envs, self.num_steps, self.action_dim),
            dtype=np.float32, buffer=buf[offset:]
        )
        offset += n * self.action_dim * _f32

        for name in ["rewards", "dones", "truncated", "log_probs", "values"]:
            views[name] = np.ndarray(
                (self.num_envs, self.num_steps),
                dtype=np.float32, buffer=buf[offset:]
            )
            offset += n * _f32

        views["last_obs"] = np.ndarray(
            (self.num_envs, self.obs_dim),
            dtype=np.float32, buffer=buf[offset:]
        )
        offset += self.num_envs * self.obs_dim * _f32

        return views

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def write_buffer(self) -> Dict[str, np.ndarray]:
        return self._buffers[self._write_idx.value % 2]

    @property
    def read_buffer(self) -> Dict[str, np.ndarray]:
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

    def read_torch(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        views = self.read_buffer
        return {k: torch.from_numpy(v.copy()).to(device) for k, v in views.items()}

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


# ---------------------------------------------------------------------------
# SharedWeightSync — actor weight synchronisation via shared memory
# ---------------------------------------------------------------------------

class SharedWeightSync:
    """Synchronise actor network weights between learner and collector processes.

    The learner writes its ``actor.state_dict()`` into a flat shared buffer;
    the collector reads and loads the weights.

    The version counter is stored as an int32 at the END of the shared memory
    block so both processes see the same value.
    """

    def __init__(
        self,
        param_shapes: Dict[str, torch.Size],
        *,
        create: bool = True,
        shm_name: str | None = None,
    ):
        self._param_shapes = param_shapes
        self._param_names = list(param_shapes.keys())

        total_numel = sum(s.numel() for s in param_shapes.values())
        _f32 = np.dtype(np.float32).itemsize
        _i32 = np.dtype(np.int32).itemsize
        data_bytes = total_numel * _f32
        meta_bytes = _i32  # version counter
        total_bytes = data_bytes + meta_bytes

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=max(total_bytes, 1))
        else:
            assert shm_name is not None
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)

        self._buffer = np.ndarray((total_numel,), dtype=np.float32, buffer=self._shm.buf)
        # Version counter at the end of the buffer
        self._version_arr = np.ndarray((1,), dtype=np.int32,
                                        buffer=self._shm.buf[data_bytes:])
        if create:
            self._version_arr[0] = 0
        self._total_numel = total_numel

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def version(self) -> int:
        return int(self._version_arr[0])

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], **kwargs) -> "SharedWeightSync":
        param_shapes = {name: p.shape for name, p in state_dict.items()}
        obj = cls(param_shapes, **kwargs)
        obj.write_weights(state_dict)
        return obj

    def write_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        offset = 0
        for name in self._param_names:
            param = state_dict[name]
            flat = param.detach().cpu().numpy().ravel()
            n = flat.shape[0]
            self._buffer[offset : offset + n] = flat
            offset += n
        self._version_arr[0] += 1

    def read_weights_into(self, state_dict: Dict[str, torch.Tensor]) -> int:
        offset = 0
        for name in self._param_names:
            param = state_dict[name]
            n = param.numel()
            data = self._buffer[offset : offset + n].copy()
            param.data.copy_(torch.from_numpy(data.reshape(param.shape)))
            offset += n
        return int(self._version_arr[0])

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


# ---------------------------------------------------------------------------
# AsyncRunner — lightweight base class for all async RL algorithms
# ---------------------------------------------------------------------------

def _get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AsyncRunner(ABC):
    """Lightweight base class for async RL algorithms.

    Subclasses implement ``_build_learner()`` and ``_collector_fn()``.
    The base class manages:
    - Shared memory allocation / cleanup
    - Collector process lifecycle
    - Main training loop skeleton
    """

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        *,
        device: str | None = None,
        collector_device: str | None = None,
        num_envs: int = 4096,
        **kwargs,
    ):
        self.env_name = env_name
        self.env_cfg_overrides = env_cfg_overrides
        self.rl_cfg = rl_cfg
        self.device = device or _get_default_device()
        self.collector_device = collector_device or self.device
        self.num_envs = num_envs
        self.extra_kwargs = kwargs

        self._collector_process: mp.Process | None = None
        self._stop_event = mp.Event()
        self._shared_resources: list = []

    @abstractmethod
    def _build_learner(self) -> Any:
        ...

    @abstractmethod
    def _collector_fn(self, stop_event: mp.Event, **kwargs) -> None:
        ...

    @abstractmethod
    def learn(self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs") -> None:
        ...

    def _start_collector(self, target_fn: Callable, kwargs: dict) -> None:
        self._collector_process = mp.Process(
            target=target_fn,
            kwargs=kwargs,
            daemon=True,
        )
        self._collector_process.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._collector_process is not None and self._collector_process.is_alive():
            self._collector_process.join(timeout=10)
            if self._collector_process.is_alive():
                self._collector_process.terminate()
                self._collector_process.join(timeout=5)
        for resource in self._shared_resources:
            if hasattr(resource, "cleanup"):
                resource.cleanup()
            elif hasattr(resource, "close"):
                resource.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
