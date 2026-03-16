from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.base import ABEnv, EnvCfg
from unilab.base.dtype_config import get_global_dtype


@dataclass
class NpEnvState:
    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict
    critic_obs: Optional[np.ndarray] = None

    @property
    def done(self) -> np.ndarray:
        return np.asarray(np.logical_or(self.terminated, self.truncated))

    def replace(self, **updates: Any) -> "NpEnvState":
        return dataclasses.replace(self, **updates)


class NpEnv(ABEnv):
    """统一的 numpy 环境基类（backend-agnostic）"""

    def __init__(self, cfg: EnvCfg, backend: SimBackend, num_envs: int):
        self._cfg = cfg
        self._backend: Any = backend
        self._num_envs = num_envs
        self._state: Optional[NpEnvState] = None
        self.step_counter = 0
        self.push_robots_flag = False
        if getattr(self._backend, "backend_type", None) == "motrix":
            self._backend._process_rigid_body_props(cfg)  # type: ignore[attr-defined]
            domain_rand = getattr(self._cfg, "domain_rand", None)
            if domain_rand and domain_rand.push_robots:
                self.push_robots_flag = True

    @property
    def cfg(self) -> EnvCfg:
        return self._cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def state(self) -> Optional[NpEnvState]:
        return self._state

    def init_state(self) -> NpEnvState:
        dtype = get_global_dtype()
        obs_shape = self.observation_space.shape
        assert obs_shape is not None
        obs = np.zeros((self._num_envs, obs_shape[0]), dtype=dtype)
        reward = np.zeros((self._num_envs,), dtype=dtype)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info: dict = {"steps": np.zeros((self._num_envs,), dtype=np.uint32)}

        self._state = NpEnvState(obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        return self._state

    def step(self, actions: np.ndarray) -> NpEnvState:
        import time

        step_t0 = time.perf_counter()

        if self._state is None:
            self.init_state()

        assert self._state is not None
        ctrl = self.apply_action(actions, self._state)

        self.push_robots()

        t0 = time.perf_counter()
        self._backend.step(ctrl, self._cfg.sim_substeps)
        step_core_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._state = self.update_state(self._state)
        update_state_time = time.perf_counter() - t0

        self._state.info["steps"] += 1
        self.step_counter += 1
        if self._cfg.max_episode_steps:
            np.greater_equal(
                self._state.info["steps"], self._cfg.max_episode_steps, out=self._state.truncated
            )

        done = self._state.done
        t0 = time.perf_counter()
        if np.any(done):
            self._reset_done_envs()
        reset_done_time = time.perf_counter() - t0

        timing = self._state.info.setdefault("timing", {})
        timing["env_step_total_ms"] = (time.perf_counter() - step_t0) * 1000.0
        timing["step_core_ms"] = step_core_time * 1000.0
        timing["update_state_ms"] = update_state_time * 1000.0
        timing["reset_done_ms"] = reset_done_time * 1000.0

        return self._state

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        done = self._state.done
        if not np.any(done):
            return

        env_indices = np.flatnonzero(done).astype(np.int32)
        self._state.info["steps"][env_indices] = 0

        if "final_observation" not in self._state.info:
            self._state.info["final_observation"] = np.zeros_like(self._state.obs)
            self._state.info["_final_observation"] = np.zeros((self._num_envs,), dtype=bool)

        self._state.info["_final_observation"][:] = False
        self._state.info["_final_observation"][env_indices] = True
        self._state.info["final_observation"][env_indices] = self._state.obs[env_indices]

        new_obs, _, info1 = self.reset(env_indices)
        self._state.obs[env_indices] = new_obs

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

    def push_robots(self) -> None:
        if self.push_robots_flag:
            domain_rand = getattr(self._cfg, "domain_rand", None)
            if domain_rand and self.step_counter % domain_rand.push_interval == 0:
                self._backend.push_robots(domain_rand.max_force)

    @abc.abstractmethod
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """子类实现：action → ctrl"""

    @abc.abstractmethod
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """子类实现：计算 obs/reward/terminated"""

    @abc.abstractmethod
    def reset(self, env_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """子类实现：重置指定环境"""

    def close(self) -> None:
        """关闭环境"""
        pass
