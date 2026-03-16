import abc
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np


@dataclass
class EnvCfg:
    """
    Config for the environment

    """

    model_file: Optional[str] = None
    sim_dt: float = 0.01
    max_episode_seconds: Optional[float] = None
    ctrl_dt: float = 0.01
    render_spacing: float = 1.0

    @property
    def max_episode_steps(self) -> Optional[int]:
        """
        return the max episode steps
        """
        if self.max_episode_seconds is None:
            return None
        return int(self.max_episode_seconds / self.ctrl_dt)

    @property
    def sim_substeps(self) -> int:
        """
        return the number of simulation steps per control step
        """
        return int(round(self.ctrl_dt / self.sim_dt))

    def validate(self):
        """
        validate the config
        """
        if self.sim_dt > self.ctrl_dt:
            raise ValueError("sim_dt must be less than or equal to ctrl_dt")


class ABEnv(abc.ABC):
    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """
        return the size of the env if it is vectorized
        """

    @property
    @abc.abstractmethod
    def cfg(self) -> EnvCfg:
        """
        The configuration of the environment
        """

    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.Space:
        """Observation space"""

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        """Action space"""

    @property
    @abc.abstractmethod
    def state(self) -> Any:
        """Current environment state (None before first reset)"""

    @abc.abstractmethod
    def init_state(self) -> Any:
        """Initialize environment and return initial state"""

    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> Any:
        """Step the environment with given actions, return new state"""

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up environment resources"""
