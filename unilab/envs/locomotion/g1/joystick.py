from __future__ import annotations

from dataclasses import dataclass, field
from etils import epath
import gymnasium as gym
import numpy as np

from unilab.envs import registry
from unilab.envs.np_env import NpEnvState
from unilab.envs.backend import create_backend
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.754]


@dataclass
class Commands:
    vel_limit = [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.25,
            "base_height": -500.0,
            "orientation": -5.0,
            "action_rate": -0.01,
            "pose": -0.1,
            "alive": 1.0,
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.754
    min_base_height: float = 0.55
    max_tilt_deg: float = 25.0


@registry.envcfg("G1WalkTaskMjSAC")
@registry.envcfg("G1JoystickFlatTerrain")
@dataclass
class G1JoystickCfg(G1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@registry.env("G1WalkTaskMjSAC", sim_backend="mujoco")
@registry.env("G1JoystickFlatTerrain", sim_backend="mujoco")
class G1WalkTask(G1BaseEnv):
    def __init__(self, cfg: G1JoystickCfg, num_envs=1, backend_type="mujoco"):
        backend = create_backend(backend_type, cfg.model_file, num_envs, cfg.sim_dt, body_name=cfg.asset.body_name)
        super().__init__(cfg, backend, num_envs)
        self._init_obs_space()
        self._init_reward_functions()

    def _init_reward_functions(self):
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "base_height": self._reward_base_height,
            "orientation": self._reward_orientation,
            "action_rate": self._reward_action_rate,
            "pose": self._reward_pose,
            "alive": self._reward_alive,
        }

    def _init_obs_space(self):
        num_obs = 3 + 3 + 3 + self._num_action + self._num_action + self._num_action + 3
        self._observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(num_obs,), dtype=float
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        qpos = self._backend.get_qpos()

        max_tilt_rad = np.deg2rad(self._cfg.reward_config.max_tilt_deg)
        tilt_terminated = gravity[:, 2] < np.cos(max_tilt_rad)
        height_terminated = qpos[:, 2] < self._cfg.reward_config.min_base_height
        terminated = np.logical_or(tilt_terminated, height_terminated)

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, qpos)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        return np.concatenate([linvel, gyro, -gravity, diff, dof_vel, last_actions, command], axis=1)

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        cfg = self._cfg.reward_config

        for name, scale in cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            reward += self._reward_fns[name](info, linvel, gyro, gravity, dof_pos, qpos) * scale

        return reward

    def _reward_tracking_lin_vel(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        commands = info["commands"]
        lin_vel_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        commands = info["commands"]
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        return np.exp(-ang_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_lin_vel_z(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        return np.square(linvel[:, 2])

    def _reward_ang_vel_xy(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_base_height(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        base_height = qpos[:, 2]
        height_error = base_height - self._cfg.reward_config.base_height_target
        below_min = base_height < self._cfg.reward_config.min_base_height
        return np.where(below_min, 100.0 * np.square(height_error), np.square(height_error))

    def _reward_orientation(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        return np.square(gravity[:, 2] - 1.0)

    def _reward_action_rate(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_pose(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        return np.sum(np.square(dof_pos - self.default_angles), axis=1)

    def _reward_alive(self, info: dict, linvel, gyro, gravity, dof_pos, qpos) -> np.ndarray:
        return np.ones((self._num_envs,), dtype=np.float32)

    def reset(self, env_indices: np.ndarray):
        num_reset = len(env_indices)
        qpos = np.tile(self._init_qpos, (num_reset, 1))
        qvel = np.tile(self._init_qvel, (num_reset, 1))

        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        quat_yaw = np_yaw_to_quat(yaw)
        qpos[:, 3:7] = np_quat_mul(quat_yaw, qpos[:, 3:7])

        self._backend.set_state(env_indices, qpos, qvel)

        commands = np.random.uniform(
            low=self._cfg.commands.vel_limit[0],
            high=self._cfg.commands.vel_limit[1],
            size=(num_reset, 3),
        )

        info = {
            "commands": np.zeros((self._num_envs, 3), dtype=np.float32),
            "current_actions": np.zeros((self._num_envs, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((self._num_envs, self._num_action), dtype=np.float32),
        }
        info["commands"][env_indices] = commands

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        obs = self._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return obs[env_indices], obs[env_indices], {k: v[env_indices] for k, v in info.items()}
