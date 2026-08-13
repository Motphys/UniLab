"""Single-slot observation/action exchange for learner-owned inference."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
import torch

_SPAWN_CTX = mp.get_context("spawn")

_IDLE = 0
_OBS_READY = 1
_ACTION_READY = 2


class SharedInferenceSlot:
    """Fixed shared-memory slot with strict single-request ownership."""

    def __init__(self, num_envs: int, obs_dim: int, action_dim: int) -> None:
        if min(num_envs, obs_dim, action_dim) <= 0:
            raise ValueError("SharedInferenceSlot dimensions must be positive")
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.observations = torch.empty(
            (self.num_envs, self.obs_dim), dtype=torch.float32
        ).share_memory_()
        self.dones = torch.empty(self.num_envs, dtype=torch.float32).share_memory_()
        self.actions = torch.empty(
            (self.num_envs, self.action_dim), dtype=torch.float32
        ).share_memory_()
        self._state = _SPAWN_CTX.Value("i", _IDLE)
        self._request_tick = _SPAWN_CTX.Value("q", -1)
        self._response_tick = _SPAWN_CTX.Value("q", -1)
        self._policy_version = _SPAWN_CTX.Value("q", -1)
        self._lock = _SPAWN_CTX.Lock()

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.observations, self.dones, self.actions)
        )

    def publish_observation(
        self,
        *,
        tick_id: int,
        observations: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        observations = np.asarray(observations, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)
        expected_obs_shape = (self.num_envs, self.obs_dim)
        if observations.shape != expected_obs_shape:
            raise ValueError(
                f"Inference observation shape must be {expected_obs_shape}, "
                f"got {observations.shape}"
            )
        if dones.shape != (self.num_envs,):
            raise ValueError(f"Inference dones shape must be {(self.num_envs,)}, got {dones.shape}")
        with self._lock:
            if self._state.value != _IDLE:
                raise RuntimeError("Inference slot cannot be reused before action consumption")
            np.copyto(self.observations.numpy(), observations)
            np.copyto(self.dones.numpy(), dones)
            self._request_tick.value = int(tick_id)
            self._state.value = _OBS_READY

    def copy_observation_to(
        self,
        *,
        tick_id: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
        non_blocking: bool = False,
    ) -> None:
        if tuple(observations.shape) != (self.num_envs, self.obs_dim):
            raise ValueError("Learner observation destination shape mismatch")
        if tuple(dones.shape) != (self.num_envs,):
            raise ValueError("Learner dones destination shape mismatch")
        with self._lock:
            self._require_state(_OBS_READY, tick_id, self._request_tick, "observation")
            observations.copy_(self.observations, non_blocking=non_blocking)
            dones.copy_(self.dones, non_blocking=non_blocking)

    def publish_action(
        self,
        *,
        tick_id: int,
        policy_version: int,
        actions: torch.Tensor,
        non_blocking: bool = False,
    ) -> None:
        if tuple(actions.shape) != (self.num_envs, self.action_dim):
            raise ValueError("Learner action shape mismatch")
        with self._lock:
            self._require_state(_OBS_READY, tick_id, self._request_tick, "observation")
            self.actions.copy_(actions, non_blocking=non_blocking)
            self._response_tick.value = int(tick_id)
            self._policy_version.value = int(policy_version)
            self._state.value = _ACTION_READY

    def consume_action(self, *, tick_id: int) -> tuple[np.ndarray, int]:
        with self._lock:
            self._require_state(_ACTION_READY, tick_id, self._response_tick, "action")
            actions = self.actions.numpy().copy()
            policy_version = int(self._policy_version.value)
            self._state.value = _IDLE
            return actions, policy_version

    def _require_state(self, state: int, tick_id: int, tick_value: Any, label: str) -> None:
        if self._state.value != state:
            raise RuntimeError(f"Inference {label} is not ready")
        actual_tick = int(tick_value.value)
        if actual_tick != int(tick_id):
            raise RuntimeError(
                f"Inference {label} tick mismatch: expected {tick_id}, got {actual_tick}"
            )

    def close(self) -> None:
        return None

    cleanup = close
