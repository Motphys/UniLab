"""G1 SAC environment - inherits from PPO for code reuse."""
from __future__ import annotations

from dataclasses import dataclass, field
from etils import epath
import numpy as np

from unilab.envs import registry
from unilab.envs.backend import create_backend
from unilab.envs.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.envs.locomotion.g1.joystick import G1JoystickPPO, InitState, Commands


@dataclass
class RewardConfigSAC:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "alive": 1.0,
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.754
    min_base_height: float = 0.55
    max_tilt_deg: float = 25.0


@registry.envcfg("G1WalkTaskMjSAC")
@dataclass
class G1JoystickSACCfg(G1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfigSAC = field(default_factory=RewardConfigSAC)


@registry.env("G1WalkTaskMjSAC", sim_backend="mujoco")
@registry.env("G1WalkTaskMjSAC", sim_backend="motrix")
class G1WalkTaskMjSAC(G1JoystickPPO):
    """G1 SAC environment - inherits from PPO, overrides rewards."""

    def __init__(self, cfg: G1JoystickSACCfg, num_envs=1, backend_type="mujoco"):
        backend = create_backend(backend_type, cfg.model_file, num_envs, cfg.sim_dt, body_name=cfg.asset.body_name)
        G1BaseEnv.__init__(self, cfg, backend, num_envs)
        self._enable_reward_log = True
        self._init_obs_space()
        self._init_reward_functions()

    def _init_reward_functions(self):
        """Override with SAC-specific rewards."""
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "base_height": self._reward_base_height,
            "action_rate": self._reward_action_rate,
            "similar_to_default": self._reward_similar_to_default,
            "alive": self._reward_alive,
        }

    def _reward_similar_to_default(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        return np.sum(np.abs(dof_pos - self.default_angles), axis=1)

    def _reward_alive(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        return np.ones((self._num_envs,), dtype=get_global_dtype())
