"""G1 Joystick environments - PPO and SAC variants."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from etils import epath

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.754]


@dataclass
class Commands:
    vel_limit = [
        [-0.6, -0.4, -0.8],  # [vx_min, vy_min, vyaw_min]
        [1.0, 0.4, 0.8],  # [vx_max, vy_max, vyaw_max]
    ]


@dataclass
class RewardConfigPPO:
    scales: dict[str, float]
    tracking_sigma: float
    gait_frequency: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    base_height_target: float
    min_base_height: float
    max_tilt_deg: float
    pose_weights: list[float]


# PPO Environment
@registry.envcfg("G1JoystickFlatTerrain")
@dataclass
class G1JoystickPPOCfg(G1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfigPPO | None = None


@registry.env("G1JoystickFlatTerrain", sim_backend="mujoco")
@registry.env("G1JoystickFlatTerrain", sim_backend="motrix")
class G1JoystickPPO(G1BaseEnv):
    _cfg: G1JoystickPPOCfg

    def __init__(self, cfg: G1JoystickPPOCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type, cfg.model_file, num_envs, cfg.sim_dt, base_name=cfg.asset.base_name
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config

        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")

        self._init_reward_functions()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(29) + dof_vel(29) + action(29) + cmd(3) + phase(2) = 98
        return {"obs": 98, "privileged": 3}

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

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._backend.get_base_pos()[:, 2] < self._reward_cfg.min_base_height,
        )

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        actor = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions, command, gait_phase],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": actor, "privileged": linvel}

    def get_obs_structure(self) -> dict:
        """Return observation structure for symmetry augmentation."""
        return {
            "linvel": 3,
            "gyro": 3,
            "gravity": 3,
            "dof_pos": self._num_action,
            "dof_vel": self._num_action,
            "actions": self._num_action,
            "command": 3,
            "gait_phase": 2,
        }

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        reward = np.zeros((self._num_envs,), dtype=dtype)
        cfg = self._reward_cfg

        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](info, linvel, gyro, gravity, dof_pos, dof_vel)
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _reward_tracking_lin_vel(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        commands = info["commands"]
        lin_vel_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self._reward_cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        commands = info["commands"]
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        return np.exp(-ang_vel_error / self._reward_cfg.tracking_sigma)

    def _reward_feet_phase(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        """步态相位奖励：鼓励正确的摆动腿高度"""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))

        def cubic_bezier_height(phi, swing_height):
            # Convert phi from [0, 2π] to [-π, π]
            phi_normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
            x = (phi_normalized + np.pi) / (2 * np.pi)

            def cubic_bezier_interpolation(y_start, y_end, t):
                y_diff = y_end - y_start
                bezier = t**3 + 3 * (t**2 * (1 - t))
                return y_start + y_diff * bezier

            stance = cubic_bezier_interpolation(
                np.zeros_like(x), np.full_like(x, swing_height), 2 * x
            )
            swing = cubic_bezier_interpolation(
                np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1
            )
            return np.where(x <= 0.5, stance, swing)

        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target = cubic_bezier_height(gait_phase[:, 0], swing_height)
        right_target = cubic_bezier_height(gait_phase[:, 1], swing_height)
        left_error = np.square(left_foot[:, 2] - left_target)
        right_error = np.square(right_foot[:, 2] - right_target)
        return np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)

    def _reward_lin_vel_z(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        return np.square(linvel[:, 2])

    def _reward_ang_vel_xy(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_orientation(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        return np.square(gravity[:, 0]) + np.square(gravity[:, 1])

    def _reward_base_height(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        return np.square(self._backend.get_base_pos()[:, 2] - self._reward_cfg.base_height_target)

    def _reward_action_rate(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_pose(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
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
            "gait_phase": np.column_stack(
                [
                    np.random.uniform(0, 2 * np.pi, num_reset),
                    np.random.uniform(0, 2 * np.pi, num_reset) + np.pi,
                ]
            ).astype(dtype),
        }

        linvel = self.get_local_linvel()[env_indices]
        gyro = self.get_gyro()[env_indices]
        gravity = self._backend.get_sensor_data("upvector")[env_indices]
        dof_pos = self.get_dof_pos()[env_indices]
        dof_vel = self.get_dof_vel()[env_indices]
        obs = self._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return obs, info

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        return np.asarray(actions * self._cfg.control_config.action_scale + self.default_angles)
