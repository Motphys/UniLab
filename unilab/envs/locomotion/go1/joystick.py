from __future__ import annotations

from etils import epath
import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field

from unilab.envs import registry
from unilab.envs.np_env import NpEnvState
from unilab.envs.backend import create_backend
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat
from unilab.envs.dtype_config import get_global_dtype

from unilab.envs.locomotion.go1.base import Go1BaseEnv, Go1BaseCfg


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.45]


@dataclass
class Commands:
    vel_limit = [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3

@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet_force = ['FL_foot_contact', 'FR_foot_contact',
                  'RL_foot_contact', 'RR_foot_contact']
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

@registry.envcfg("Go1JoystickFlatTerrain")
@dataclass
class Go1JoystickCfg(Go1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    sensor: Sensor = field(default_factory=Sensor)
    domain_rand: Domain_Rand = field(default_factory=Domain_Rand)

@registry.env("Go1JoystickFlatTerrain", sim_backend="mujoco")
@registry.env("Go1JoystickFlatTerrain", sim_backend="motrix")
@registry.env("Go1JoystickFlatTerrain", sim_backend="motrix_numba")

class Go1WalkTask(Go1BaseEnv):
    def __init__(self, cfg: Go1JoystickCfg, num_envs=1, backend_type="mujoco"):
        backend = create_backend(backend_type, cfg.model_file, num_envs, cfg.sim_dt, body_name=cfg.asset.body_name)
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._init_obs_space()
        self._init_reward_functions()
        self.phase = np.zeros(
            (num_envs,), dtype = np.float32
        )
        self.feet_phase = np.zeros(
            (num_envs,
            len(cfg.sensor.feet_force)),
            dtype = np.float32
        )
        self.gait_frequency = 2
        self.feet_force = np.zeros(
            (num_envs,
            len(cfg.sensor.feet_force), 3),
            dtype = np.float32
        )

    def _init_reward_functions(self):
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "base_height": self._reward_base_height,
            "action_rate": self._reward_action_rate,
            "similar_to_default": self._reward_similar_to_default,
            "contact": self._reward_contact,
        }

    def _init_obs_space(self):
        num_obs = 3 + 3 + 3 + self._num_action + self._num_action + self._num_action + 3 + 4
        self._observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(num_obs,), dtype=float
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def update_state(self, state: NpEnvState) -> NpEnvState:
        period = 0.6
        self.phase = np.fmod(self.phase + self._cfg.ctrl_dt * self.gait_frequency, 1.0)
        self.feet_phase[:, 0] = self.phase
        self.feet_phase[:, 3] = self.phase

        self.feet_phase[:, 1] = (self.phase + 0.5) % 1
        self.feet_phase[:, 2] = (self.phase + 0.5) % 1

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        qpos = self._backend.get_qpos()
        self.feet_force[:, :, :] = 0
        for i in range(len(self._cfg.sensor.feet_force)):
            self.feet_force[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_force[i])
        terminated = gravity[:, 2] <= 0.5
        reward = self._compute_reward(state.info, linvel, gyro, dof_pos, qpos)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel, self.feet_phase)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel, feet_phase) -> np.ndarray:
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        return np.concatenate([linvel, gyro, -gravity, diff, dof_vel, last_actions, command, feet_phase], axis=1, dtype=get_global_dtype())

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._cfg.reward_config

        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](info, linvel, gyro, dof_pos, qpos)
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _reward_tracking_lin_vel(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        commands = info["commands"]
        lin_vel_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        commands = info["commands"]
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        return np.exp(-ang_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_lin_vel_z(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        return np.square(linvel[:, 2])

    def _reward_ang_vel_xy(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_base_height(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        base_height = qpos[:, 2]
        return np.square(base_height - self._cfg.reward_config.base_height_target)

    def _reward_action_rate(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_similar_to_default(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        return np.sum(np.abs(dof_pos - self.default_angles), axis=1)

    def _reward_contact(self, info: dict, linvel, gyro, dof_pos, qpos) -> np.ndarray:
        contact = self.feet_force[:, :, 2] > 0.1
        res = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(len(self._cfg.sensor.feet_force)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (self.gait_frequency < 1.0e-8)
            res += ~(contact[:, i] ^ is_contact)
        return res

    def _reward_swing_feet_z(self):
        is_contact = (self.feet_phase < 0.6) | (self.gait_frequency < 1.0e-8).unsqueeze(1)
        pos_error = np.square((self.feet_pos[:, :, 2] - 0.1)) * ~is_contact
        return torch.sum(pos_error, dim=1) 

    def reset(self, env_indices: np.ndarray):
        num_reset = len(env_indices)
        qpos = np.tile(self._init_qpos, (num_reset, 1))
        qvel = np.tile(self._init_qvel, (num_reset, 1))

        # Domain Randomization
        dxy = np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 0:2] += dxy
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        quat_yaw = np_yaw_to_quat(yaw)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], quat_yaw)
        qvel[:, 0:6] = np.random.uniform(-0.5, 0.5, (num_reset, 6))

        self._backend.set_state(env_indices, qpos, qvel)

        commands = np.random.uniform(
            low=self._cfg.commands.vel_limit[0],
            high=self._cfg.commands.vel_limit[1],
            size=(num_reset, 3),
        )

        info = {
            "commands": commands,
            "current_actions": np.zeros((num_reset, self._num_action), dtype=get_global_dtype()),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=get_global_dtype()),
        }

        linvel = self.get_local_linvel()[env_indices]
        gyro = self.get_gyro()[env_indices]
        gravity = self._backend.get_sensor_data("upvector")[env_indices]
        dof_pos = self.get_dof_pos()[env_indices]
        dof_vel = self.get_dof_vel()[env_indices]
        obs = self._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel, self.feet_phase[env_indices])
        return obs, obs, info

