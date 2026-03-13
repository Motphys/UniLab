from __future__ import annotations

from etils import epath
import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.backend import create_backend
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat

from unilab.envs.locomotion.go2.base import Go2BaseEnv, Go2BaseCfg


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.42]


@dataclass
class Commands:
    vel_limit = [
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]

@dataclass
class Domain_Rand:
        # randomize_friction = True
        # friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]

        random_com = True
        com_offset_x = [-0.05, 0.05]

        push_robots = True
        push_interval = 750 #step
        max_force = [1, 1, 0.5]

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.02,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "alive": 0.0,
            "foot_lift_reward": 0.2,
            "foot_drag_penalty": -0.0,
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3
    target_foot_height: float = 0.08
    foot_clearance_sigma: float = 0.02


@registry.envcfg("Go2JoystickFlatTerrain")
@dataclass
class Go2JoystickCfg(Go2BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    domain_rand: Domain_Rand = field(default_factory=Domain_Rand)


@registry.env("Go2JoystickFlatTerrain", sim_backend="mujoco")
@registry.env("Go2JoystickFlatTerrain", sim_backend="motrix")
class Go2WalkTask(Go2BaseEnv):
    def __init__(self, cfg: Go2JoystickCfg, num_envs=1, backend_type="mujoco"):
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
            "action_rate": self._reward_action_rate,
            "similar_to_default": self._reward_similar_to_default,
            "alive": self._reward_alive,
            "foot_lift_reward": self._reward_foot_lift,
            "foot_drag_penalty": self._reward_foot_drag,
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
        terminated = self._compute_termination()
        reward = self._compute_reward(state.info)
        obs = self._compute_obs(state.info)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(self, info: dict) -> np.ndarray:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))

        return np.concatenate([linvel, gyro, -gravity, diff, dof_vel, last_actions, command], axis=1)

    def _compute_reward(self, info: dict) -> np.ndarray:
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        cfg = self._cfg.reward_config

        for name, scale in cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](info)
            reward += rew * scale
            if "log" not in info:
                info["log"] = {}
            info["log"][f"reward/{name}"] = float(np.mean(rew * scale))

        reward *= self._cfg.ctrl_dt
        return reward

    def _reward_tracking_lin_vel(self, info: dict) -> np.ndarray:
        linvel = self.get_local_linvel()
        commands = info["commands"]
        lin_vel_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, info: dict) -> np.ndarray:
        gyro = self.get_gyro()
        commands = info["commands"]
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        return np.exp(-ang_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_lin_vel_z(self, info: dict) -> np.ndarray:
        global_linvel = self._backend.get_sensor_data("global_linvel")
        return np.square(global_linvel[:, 2])

    def _reward_ang_vel_xy(self, info: dict) -> np.ndarray:
        gyro = self.get_gyro()
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_base_height(self, info: dict) -> np.ndarray:
        base_height = self._backend.get_qpos()[:, 2]
        return np.square(base_height - self._cfg.reward_config.base_height_target)

    def _reward_action_rate(self, info: dict) -> np.ndarray:
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_similar_to_default(self, info: dict) -> np.ndarray:
        return np.sum(np.abs(self.get_dof_pos() - self.default_angles), axis=1)

    def _reward_alive(self, info: dict) -> np.ndarray:
        return np.ones((self._num_envs,), dtype=np.float32)

    def _reward_foot_lift(self, info: dict) -> np.ndarray:
        foot_pos = self.get_foot_pos()
        foot_heights = foot_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = (foot_contact < 0.5)
        target_height = self._cfg.reward_config.target_foot_height
        sigma = self._cfg.reward_config.foot_clearance_sigma
        error_sq = np.square(foot_heights - target_height)
        reward = np.exp(-error_sq / sigma) * is_swing
        return np.sum(reward, axis=1)

    def _reward_foot_drag(self, info: dict) -> np.ndarray:
        foot_pos = self.get_foot_pos()
        foot_heights = foot_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = (foot_contact < 0.5)
        safe_height = self._cfg.reward_config.target_foot_height / 2.0
        height_error = np.clip(safe_height - foot_heights, 0.0, None)
        error = np.square(height_error) * is_swing
        return np.sum(error, axis=1)

    def _compute_termination(self) -> np.ndarray:
        gravity = self._backend.get_sensor_data("upvector")
        return gravity[:, 2] < 0.5

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
        }
        info["commands"][env_indices] = commands

        obs = self._compute_obs(info)
        return obs[env_indices], obs[env_indices], {k: v[env_indices] for k, v in info.items()}

