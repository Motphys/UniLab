from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import gymnasium as gym
import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.base import ABEnv, EnvCfg, EnvPlayCapabilities
from unilab.dr import DomainRandomizationManager, DomainRandomizationProvider
from unilab.dtype_config import get_global_dtype

if TYPE_CHECKING:
    from unilab.base.augmentation import SymmetryAugmentation


@dataclass
class NpEnvState:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]
    final_observation: dict[str, np.ndarray] | None = None

    @property
    def done(self) -> np.ndarray:
        done: np.ndarray = np.logical_or(self.terminated, self.truncated)
        return done

    def replace(self, **updates: Any) -> "NpEnvState":
        return dataclasses.replace(self, **updates)


class NpEnv(ABEnv):
    """统一的 numpy 环境基类（backend-agnostic）"""

    def __init__(self, cfg: EnvCfg, backend: SimBackend, num_envs: int):
        self._cfg = cfg
        self._backend: Any = backend
        self._num_envs = num_envs
        self._state: Optional[NpEnvState] = None
        self._truncated_scratch: np.ndarray = np.zeros((self._num_envs,), dtype=bool)
        self._final_observation_scratch: dict[str, np.ndarray] | None = None
        self.step_counter = 0
        self._dr_manager: DomainRandomizationManager | None = None
        self._init_randomization_applied = False

    @property
    def cfg(self) -> EnvCfg:
        return self._cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def state(self) -> Optional[NpEnvState]:
        return self._state

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """Return observation group dimensions, e.g. {"obs": 98, "critic": 101}.

        Subclasses MUST override this property.
        """
        raise NotImplementedError

    @property
    def observation_space(self) -> gym.Space:
        total = sum(self.obs_groups_spec.values())
        return gym.spaces.Box(-np.inf, np.inf, shape=(total,), dtype=np.float64)

    def build_symmetry_augmentation(self, *, device: str) -> "SymmetryAugmentation | None":
        """Return an env-owned runtime symmetry adapter when the task/backend supports it."""
        return None

    def init_state(self) -> NpEnvState:
        dtype = get_global_dtype()
        obs = {
            k: np.zeros((self._num_envs, d), dtype=dtype) for k, d in self.obs_groups_spec.items()
        }
        reward = np.zeros((self._num_envs,), dtype=dtype)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        if self._cfg.max_episode_steps:
            steps = np.random.randint(
                0, self._cfg.max_episode_steps, size=(self._num_envs,), dtype=np.uint32
            )
        else:
            steps = np.zeros((self._num_envs,), dtype=np.uint32)
        info: dict = {"steps": steps}

        self._state = NpEnvState(obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._clear_step_final_observation()
        return self._state

    def step(self, actions: np.ndarray) -> NpEnvState:
        import time

        step_t0 = time.perf_counter()

        if self._state is None:
            self.init_state()

        assert self._state is not None

        t0 = time.perf_counter()
        ctrl = self.apply_action(actions, self._state)
        apply_action_time = time.perf_counter() - t0

        if self._dr_manager is not None:
            self._dr_manager.apply_interval_randomization_if_due(self.step_counter)
        self._state.truncated.fill(False)
        self._clear_step_final_observation()

        t0 = time.perf_counter()
        backend_result = self._backend.step(ctrl, self._cfg.sim_substeps)
        step_core_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._state = self.update_state(self._state)
        update_state_time = time.perf_counter() - t0

        self._state.info["steps"] += 1
        self.step_counter += 1
        truncated = self._compute_truncated(self._state)
        np.logical_or(self._state.truncated, truncated, out=self._state.truncated)

        done = self._state.done
        t0 = time.perf_counter()
        if np.any(done):
            self._reset_done_envs()
        reset_done_time = time.perf_counter() - t0

        timing = self._state.info.setdefault("timing", {})
        timing["env_step_total_ms"] = (time.perf_counter() - step_t0) * 1000.0
        timing["apply_action_ms"] = apply_action_time * 1000.0
        timing["step_core_ms"] = step_core_time * 1000.0
        timing["update_state_ms"] = update_state_time * 1000.0
        timing["reset_done_ms"] = reset_done_time * 1000.0
        if backend_result is not None:
            backend_timing = backend_result.get("timing")
            if backend_timing:
                for k, v in backend_timing.items():
                    timing[f"backend_{k}"] = v

        return self._state

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        done = self._state.done
        if not np.any(done):
            return

        env_indices = np.flatnonzero(done).astype(np.int32)
        self._state.info["steps"][env_indices] = 0

        final_observation = self._ensure_final_observation_scratch()
        compat_final_observation, compat_terminal_mask = (
            self._ensure_final_observation_compat_buffers()
        )
        compat_terminal_mask[:] = False
        compat_terminal_mask[env_indices] = True
        for key in self._state.obs:
            final_observation[key][env_indices] = self._state.obs[key][env_indices]
            compat_final_observation[key][env_indices] = final_observation[key][env_indices]

        self._state.final_observation = final_observation

        new_obs, info1 = self.reset(env_indices)
        for key in self._state.obs:
            self._state.obs[key][env_indices] = new_obs[key]

        if info1:
            for key, value in info1.items():
                if key not in self._state.info:
                    if isinstance(value, np.ndarray):
                        full_shape = (self._num_envs,) + value.shape[1:]
                        self._state.info[key] = np.zeros(full_shape, dtype=value.dtype)
                        self._state.info[key][env_indices] = value
                    else:
                        self._state.info[key] = value
                elif isinstance(value, np.ndarray):
                    self._state.info[key][env_indices] = value

    def _ensure_final_observation_scratch(self) -> dict[str, np.ndarray]:
        assert self._state is not None
        obs = self._state.obs
        scratch = self._final_observation_scratch
        if scratch is None or set(scratch) != set(obs):
            scratch = {key: np.zeros_like(value) for key, value in obs.items()}
            self._final_observation_scratch = scratch
        else:
            for key, value in obs.items():
                if scratch[key].shape != value.shape or scratch[key].dtype != value.dtype:
                    scratch = {
                        obs_key: np.zeros_like(obs_value) for obs_key, obs_value in obs.items()
                    }
                    self._final_observation_scratch = scratch
                    break
        assert scratch is not None
        return scratch

    def _ensure_final_observation_compat_buffers(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        assert self._state is not None
        obs = self._state.obs
        compat_final_observation = self._state.info.get("final_observation")
        if not isinstance(compat_final_observation, dict) or set(compat_final_observation) != set(
            obs
        ):
            compat_final_observation = {key: np.zeros_like(value) for key, value in obs.items()}
            self._state.info["final_observation"] = compat_final_observation
        else:
            for key, value in obs.items():
                if (
                    compat_final_observation[key].shape != value.shape
                    or compat_final_observation[key].dtype != value.dtype
                ):
                    compat_final_observation = {
                        obs_key: np.zeros_like(obs_value) for obs_key, obs_value in obs.items()
                    }
                    self._state.info["final_observation"] = compat_final_observation
                    break
        compat_terminal_mask = self._state.info.get("_final_observation")
        if not isinstance(compat_terminal_mask, np.ndarray) or compat_terminal_mask.shape != (
            self._num_envs,
        ):
            compat_terminal_mask = np.zeros((self._num_envs,), dtype=bool)
            self._state.info["_final_observation"] = compat_terminal_mask
        return compat_final_observation, compat_terminal_mask

    def _clear_step_final_observation(self) -> None:
        assert self._state is not None
        self._state.final_observation = None
        compat_terminal_mask = self._state.info.get("_final_observation")
        if isinstance(compat_terminal_mask, np.ndarray):
            compat_terminal_mask.fill(False)

    def _init_domain_randomization(self, provider: "DomainRandomizationProvider") -> None:
        from unilab.dr import DomainRandomizationManager

        self._dr_manager = DomainRandomizationManager(self, provider)
        if not self._init_randomization_applied:
            self._init_randomization_applied = self._dr_manager.apply_init_randomization()
        self._backend.materialize()

    def reset(self, env_indices: np.ndarray) -> Tuple[dict[str, np.ndarray], dict]:
        if self._dr_manager is None:  # pragma: no cover - constructor integration error
            raise RuntimeError("Domain-randomization manager has not been initialized")
        return self._dr_manager.reset(env_indices)

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        """Compute truncation conditions.

        By default, episodes are truncated only when the configured maximum
        episode length is reached. Subclasses may override this to add
        task-specific truncation conditions while remaining compatible with the
        existing done/reset contract.
        """
        truncated = cast(np.ndarray | None, getattr(self, "_truncated_scratch", None))
        if truncated is None or truncated.shape != (self._num_envs,):
            truncated = np.zeros((self._num_envs,), dtype=bool)
            self._truncated_scratch = truncated
        truncated.fill(False)
        if self._cfg.max_episode_steps:
            np.greater_equal(state.info["steps"], self._cfg.max_episode_steps, out=truncated)
        return truncated

    def init_play_renderer(self, render_spacing: float | None = None) -> None:
        """Initialize backend-native interactive playback when available."""
        if not self.play_capabilities.supports_native_interactive_renderer:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support native interactive playback"
            )
        if render_spacing is None:
            self._backend.init_renderer()
            return
        try:
            self._backend.init_renderer(spacing=render_spacing)
        except TypeError:
            self._backend.init_renderer()

    def render_play_frame(self) -> None:
        """Render one interactive playback frame through the env contract."""
        if not self.play_capabilities.supports_native_interactive_renderer:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support native interactive playback"
            )
        self._backend.render()

    def get_physics_state_snapshot(self) -> np.ndarray:
        """Return a detached physics snapshot for offline playback/video export."""
        if not self.play_capabilities.supports_physics_state_playback:
            raise NotImplementedError(
                f"{self._backend.__class__.__name__} does not support physics-state playback"
            )
        physics_state = cast(
            np.ndarray, np.asarray(self._backend.get_physics_state(), dtype=np.float32)
        )
        snapshot = cast(np.ndarray, physics_state.copy())
        return snapshot

    @abc.abstractmethod
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """子类实现：action → ctrl"""

    @abc.abstractmethod
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """子类实现：计算 obs/reward/terminated"""

    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        capabilities = self._backend.get_play_capabilities()
        return EnvPlayCapabilities(
            supports_native_interactive_renderer=capabilities.supports_native_interactive_renderer,
            supports_physics_state_playback=capabilities.supports_physics_state_playback,
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the backend playback model for one env in a vectorized batch.

        Args:
            env_index: Optional vectorized environment index.

        Returns:
            The backend-specific playback model.
        """
        return self._backend.get_playback_model(env_index)

    def close(self) -> None:
        """关闭环境"""
        pass

    def _supports_backend_property(self, name: str) -> bool:
        return hasattr(self._backend, name)
