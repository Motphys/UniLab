import abc
import dataclasses
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import gymnasium as gym

from unilab.envs.base import ABEnv, EnvCfg
from unilab.envs.backend import ISimBackend


@dataclass
class NpEnvState:
    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict

    @property
    def done(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)

    def replace(self, **updates) -> "NpEnvState":
        return dataclasses.replace(self, **updates)


class NpEnv(ABEnv):
    """统一的 numpy 环境基类（backend-agnostic）"""

    def __init__(self, cfg: EnvCfg, backend: ISimBackend, num_envs: int):
        self._cfg = cfg
        self._backend = backend
        self._num_envs = num_envs
        self._state = None

    @property
    def cfg(self) -> EnvCfg:
        return self._cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def state(self) -> NpEnvState:
        return self._state

    def init_state(self) -> NpEnvState:
        obs = np.zeros((self._num_envs, self.observation_space.shape[0]), dtype=np.float32)
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {"steps": np.zeros((self._num_envs,), dtype=np.uint32)}

        self._state = NpEnvState(obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        return self._state

    def step(self, actions: np.ndarray) -> NpEnvState:
        if self._state is None:
            self.init_state()

        ctrl = self.apply_action(actions, self._state)
        self._backend.step(ctrl, self._cfg.sim_substeps)

        self._state = self.update_state(self._state)
        self._state.info["steps"] += 1

        if self._cfg.max_episode_steps:
            np.greater_equal(self._state.info["steps"], self._cfg.max_episode_steps, out=self._state.truncated)

        done = self._state.done
        if np.any(done):
            self._reset_done_envs()

        return self._state

    def _reset_done_envs(self):
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
                    self._state.info[key] = value
                elif isinstance(value, np.ndarray):
                    self._state.info[key][env_indices] = value

    @abc.abstractmethod
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """子类实现：action → ctrl"""

    @abc.abstractmethod
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """子类实现：计算 obs/reward/terminated"""

    @abc.abstractmethod
    def reset(self, env_indices: np.ndarray) -> Tuple[np.ndarray, dict]:
        """子类实现：重置指定环境"""

    def close(self):
        """关闭环境"""
        pass
