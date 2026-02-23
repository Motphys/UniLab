"""Lightweight async runner framework using Python native multiprocessing + shared memory.

Replaces Ray with zero-copy shared memory for inter-process communication.
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

    Buffer capacity = ``capacity`` (total transitions, should be ``buffer_n * num_envs``).
    """

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

        # Compute byte sizes for each field (float32 = 4 bytes)
        _f32 = np.dtype(np.float32).itemsize
        self._obs_bytes = capacity * obs_dim * _f32
        self._act_bytes = capacity * action_dim * _f32
        self._scalar_bytes = capacity * _f32  # reward / done each

        total_bytes = (
            2 * self._obs_bytes       # obs + next_obs
            + self._act_bytes         # actions
            + 2 * self._scalar_bytes  # rewards + dones
        )

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        else:
            assert shm_name is not None, "Must provide shm_name when create=False"
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)

        # Build numpy views (zero-copy) over the shared buffer
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

        # Atomic counters (shared between processes)
        self._ptr = mp.Value("i", 0)
        self._size = mp.Value("i", 0)

    # -- properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def ptr(self) -> int:
        return self._ptr.value

    @property
    def size(self) -> int:
        return self._size.value

    # -- write (collector side) ----------------------------------------------

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Insert a batch of transitions into the ring buffer.

        All arrays have leading dimension ``batch_size``.
        """
        batch_size = obs.shape[0]
        start = self._ptr.value % self.capacity

        if start + batch_size <= self.capacity:
            # Contiguous write
            self.obs[start : start + batch_size] = obs
            self.next_obs[start : start + batch_size] = next_obs
            self.actions[start : start + batch_size] = actions
            self.rewards[start : start + batch_size] = rewards
            self.dones[start : start + batch_size] = dones
        else:
            # Wrap-around write
            first = self.capacity - start
            self.obs[start:] = obs[:first]
            self.obs[:batch_size - first] = obs[first:]
            self.next_obs[start:] = next_obs[:first]
            self.next_obs[:batch_size - first] = next_obs[first:]
            self.actions[start:] = actions[:first]
            self.actions[:batch_size - first] = actions[first:]
            self.rewards[start:] = rewards[:first]
            self.rewards[:batch_size - first] = rewards[first:]
            self.dones[start:] = dones[:first]
            self.dones[:batch_size - first] = dones[first:]

        self._ptr.value += batch_size
        self._size.value = min(self._size.value + batch_size, self.capacity)

    # -- read (learner side) -------------------------------------------------

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Uniformly sample a batch from the buffer (numpy arrays)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_obs": self.next_obs[indices].copy(),
            "dones": self.dones[indices].copy(),
        }

    def sample_torch(
        self, batch_size: int, device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """Sample and convert to torch tensors on the given device."""
        data = self.sample(batch_size)
        return {
            k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in data.items()
        }

    # -- lifecycle -----------------------------------------------------------

    def cleanup(self) -> None:
        """Release shared memory (call from the *creating* process only)."""
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass

    def close(self) -> None:
        """Detach from shared memory (call from non-creating process)."""
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

        # Per-buffer byte sizes
        obs_bytes = n * obs_dim * _f32         # obs
        act_bytes = n * action_dim * _f32      # actions
        scalar_bytes = n * _f32                # rewards / dones / log_probs / mu / sigma each

        # We also store last_obs = (num_envs, obs_dim)
        last_obs_bytes = num_envs * obs_dim * _f32

        # Total per buffer
        per_buffer = (
            obs_bytes           # obs
            + act_bytes         # actions
            + scalar_bytes * 5  # rewards, dones, truncated, log_probs, values
            + last_obs_bytes    # last_obs
        )

        total_bytes = 2 * per_buffer  # double buffer

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

        # Synchronisation primitives
        self._write_idx = mp.Value("i", 0)     # which buffer the collector writes to
        self._ready = [mp.Event(), mp.Event()]  # signalled when a buffer is filled

    def _make_views(self, base_offset: int) -> Dict[str, np.ndarray]:
        """Create numpy views for one buffer starting at ``base_offset``."""
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
        return self._buffers[(self._write_idx.value + 1) % 2]

    def signal_write_done(self) -> None:
        """Collector calls this after filling a buffer."""
        idx = self._write_idx.value % 2
        self._ready[idx].set()
        self._write_idx.value += 1

    def wait_for_data(self, timeout: float = 30.0) -> bool:
        """Learner blocks until the *read* buffer is ready."""
        read_idx = (self._write_idx.value + 1) % 2
        result = self._ready[read_idx].wait(timeout=timeout)
        if result:
            self._ready[read_idx].clear()
        return result

    def read_torch(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Read the current read-buffer and convert to torch tensors."""
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

        # Calculate total number of float32 parameters
        total_numel = sum(s.numel() for s in param_shapes.values())
        total_bytes = total_numel * np.dtype(np.float32).itemsize

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=max(total_bytes, 1))
        else:
            assert shm_name is not None
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)

        self._buffer = np.ndarray((total_numel,), dtype=np.float32, buffer=self._shm.buf)
        self._lock = mp.Lock()
        self._version = mp.Value("i", 0)
        self._total_numel = total_numel

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def version(self) -> int:
        return self._version.value

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], **kwargs) -> "SharedWeightSync":
        """Create from a model's ``state_dict()``."""
        param_shapes = {name: p.shape for name, p in state_dict.items()}
        obj = cls(param_shapes, **kwargs)
        obj.write_weights(state_dict)
        return obj

    def write_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Learner writes weights (called from learner process)."""
        with self._lock:
            offset = 0
            for name in self._param_names:
                param = state_dict[name]
                flat = param.detach().cpu().numpy().ravel()
                n = flat.shape[0]
                self._buffer[offset : offset + n] = flat
                offset += n
            self._version.value += 1

    def read_weights_into(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Collector reads weights into an existing state_dict (in-place).

        Returns the version number that was read.
        """
        with self._lock:
            offset = 0
            for name in self._param_names:
                param = state_dict[name]
                n = param.numel()
                data = self._buffer[offset : offset + n].copy()
                param.data.copy_(torch.from_numpy(data.reshape(param.shape)))
                offset += n
            return self._version.value

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
    """Return the best available device string for this platform."""
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

    Designed for Mac (MPS) with extensibility hooks for NPU.
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
        # Collector can run on a separate device (default: same as learner)
        self.collector_device = collector_device or self.device
        self.num_envs = num_envs
        self.extra_kwargs = kwargs

        # Will be initialised by subclass
        self._collector_process: mp.Process | None = None
        self._stop_event = mp.Event()
        self._shared_resources: list = []  # track for cleanup

    # -- abstract methods (subclass must implement) --------------------------

    @abstractmethod
    def _build_learner(self) -> Any:
        """Create and return the learner object on ``self.device``."""
        ...

    @abstractmethod
    def _collector_fn(
        self,
        stop_event: mp.Event,
        **kwargs,
    ) -> None:
        """Entry point for the collector subprocess.

        Must create the environment, run rollouts, and write to the shared
        replay buffer / storage.  Should check ``stop_event`` periodically.
        """
        ...

    @abstractmethod
    def learn(
        self,
        max_iterations: int,
        save_interval: int = 50,
        log_dir: str = "logs",
    ) -> None:
        """Main training loop.  Subclass implements the full loop."""
        ...

    # -- lifecycle -----------------------------------------------------------

    def _start_collector(self, target_fn: Callable, kwargs: dict) -> None:
        """Launch the collector subprocess."""
        self._collector_process = mp.Process(
            target=target_fn,
            kwargs=kwargs,
            daemon=True,
        )
        self._collector_process.start()

    def close(self) -> None:
        """Cleanly shut down the collector process and release shared memory."""
        # Signal collector to stop
        self._stop_event.set()

        if self._collector_process is not None and self._collector_process.is_alive():
            self._collector_process.join(timeout=10)
            if self._collector_process.is_alive():
                self._collector_process.terminate()
                self._collector_process.join(timeout=5)

        # Clean up shared memory
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
