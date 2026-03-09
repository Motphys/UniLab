"""G1 Joystick environments - PPO and SAC variants."""
from __future__ import annotations

from dataclasses import dataclass, field
from etils import epath
import gymnasium as gym
import math
import numpy as np

from unilab.envs import registry
from unilab.envs.np_env import NpEnvState
from unilab.envs.backend import create_backend
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat
from unilab.envs.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.754]


@dataclass
class Commands:
    vel_limit = [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


@dataclass
class RewardConfigPPO:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.2,
            "feet_phase": 1.0,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.25,
            "base_height": -500.0,
            "orientation": -5.0,
            "action_rate": -0.01,
            "pose": -0.1,
        }
    )
    tracking_sigma: float = 0.25
    gait_frequency: float = 1.5
    feet_phase_swing_height: float = 0.09
    feet_phase_tracking_sigma: float = 0.008
    base_height_target: float = 0.754
    min_base_height: float = 0.55
    max_tilt_deg: float = 25.0
    pose_weights: list[float] = field(
        default_factory=lambda: [
            0.01, 1.0, 5.0, 0.01, 5.0, 5.0,
            0.01, 1.0, 5.0, 0.01, 5.0, 5.0,
            50.0, 50.0, 50.0,
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        ]
    )


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


# PPO Environment
@registry.envcfg("G1JoystickFlatTerrain")
@dataclass
class G1JoystickPPOCfg(G1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfigPPO = field(default_factory=RewardConfigPPO)


@registry.env("G1JoystickFlatTerrain", sim_backend="mujoco")
class G1JoystickPPO(G1BaseEnv):
    def __init__(self, cfg: G1JoystickPPOCfg, num_envs=1, backend_type="mujoco"):
        backend = create_backend(backend_type, cfg.model_file, num_envs, cfg.sim_dt, body_name=cfg.asset.body_name)
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True

        self._gait_phase_delta = float(2.0 * math.pi * cfg.reward_config.gait_frequency * cfg.ctrl_dt)
        self._pose_weights = np.array(cfg.reward_config.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError(f"pose_weights length mismatch")

        self._init_obs_space()
        self._init_reward_functions()

    def _init_reward_functions(self):
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "feet_phase": self._reward_feet_phase,
            "lin_vel_z": self._reward_lin_vel_z,
            "orientation": self._reward_orientation,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "action_rate": self._reward_action_rate,
            "base_height": self._reward_base_height,
            "pose": self._reward_pose,
        }

    def _init_obs_space(self):
        num_obs = 3 + 3 + 3 + self._num_action + self._num_action + self._num_action + 3 + 1
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
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(tilt > max_tilt_rad, qpos[:, 2] < self._cfg.reward_config.min_base_height)

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel, qpos)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 1), dtype=get_global_dtype()))
        return np.concatenate([linvel, gyro, -gravity, diff, dof_vel, last_actions, command, gait_phase], axis=1, dtype=get_global_dtype())

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel, qpos) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._cfg.reward_config

        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](info, linvel, gyro, gravity, dof_pos, dof_vel, qpos)
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _reward_tracking_lin_vel(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        commands = info["commands"]
        lin_vel_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        commands = info["commands"]
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        return np.exp(-ang_vel_error / self._cfg.reward_config.tracking_sigma)

    def _reward_feet_phase(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 1), dtype=get_global_dtype()))

        target_height = self._cfg.reward_config.feet_phase_swing_height * np.abs(np.sin(gait_phase[:, 0]))
        left_error = np.square(left_foot[:, 2] - target_height)
        right_error = np.square(right_foot[:, 2] - target_height)
        return np.exp(-(left_error + right_error) / self._cfg.reward_config.feet_phase_tracking_sigma)

    def _reward_lin_vel_z(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        return np.square(linvel[:, 2])

    def _reward_ang_vel_xy(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_orientation(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        return np.square(gravity[:, 0]) + np.square(gravity[:, 1])

    def _reward_base_height(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        return np.square(qpos[:, 2] - self._cfg.reward_config.base_height_target)

    def _reward_action_rate(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_pose(self, info, linvel, gyro, gravity, dof_pos, dof_vel, qpos):
        diff = dof_pos - self.default_angles
        return np.sum(self._pose_weights * np.square(diff), axis=1)

    def reset(self, env_indices: np.ndarray):
        dtype = get_global_dtype()
        num_reset = len(env_indices)
        qpos = np.tile(self._init_qpos, (num_reset, 1))
        qvel = np.tile(self._init_qvel, (num_reset, 1))

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
            "current_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "gait_phase": np.random.uniform(0, 2 * np.pi, (num_reset, 1)).astype(dtype),
        }

        linvel = self.get_local_linvel()[env_indices]
        gyro = self.get_gyro()[env_indices]
        gravity = self._backend.get_sensor_data("upvector")[env_indices]
        dof_pos = self.get_dof_pos()[env_indices]
        dof_vel = self.get_dof_vel()[env_indices]
        obs = self._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return obs, obs, info

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get("gait_phase", np.zeros((self._num_envs, 1), dtype=get_global_dtype()))
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        return actions * self._cfg.control_config.action_scale + self.default_angles


# SAC Environment (inherits from PPO)
@registry.envcfg("G1WalkTaskMjSAC")
@dataclass
class G1JoystickCfg(G1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfigSAC = field(default_factory=RewardConfigSAC)


@registry.env("G1WalkTaskMjSAC", sim_backend="mujoco")
class G1WalkTask(G1JoystickPPO):
    """SAC environment - simpler rewards than PPO version."""

    def __init__(self, cfg: G1JoystickCfg, num_envs=1, backend_type="mujoco"):
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
