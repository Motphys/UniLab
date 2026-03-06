from __future__ import annotations

from dataclasses import dataclass, field

from etils import epath
import gymnasium as gym
import math
import numpy as np

from unilab.envs import registry
from unilab.envs.mujoco_env.mj_env import MjNpEnvState
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseMjEnv
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.754]


@dataclass
class Commands:
    # Fixed forward command for fastest convergence.
    vel_limit = [
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]


@dataclass
class RewardConfig:
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
            # "similar_to_default": -0.02,
            "pose": -0.1, #-0.02,
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

@registry.envcfg("G1JoystickFlatTerrain")
@dataclass
class G1JoystickCfg(G1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@registry.env("G1JoystickFlatTerrain", sim_backend="mujoco")
class G1WalkTaskMj(G1BaseMjEnv):
    def __init__(self, cfg: G1JoystickCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._enable_reward_log = True
        self._idx_left_foot_pos = self._get_sensor_indices("left_foot_pos")
        self._idx_right_foot_pos = self._get_sensor_indices("right_foot_pos")
        if self._idx_left_foot_pos is None or self._idx_right_foot_pos is None:
            raise ValueError("Sensors 'left_foot_pos' and 'right_foot_pos' are required for feet_phase reward.")
        self._gait_phase_delta = float(2.0 * math.pi * self.cfg.reward_config.gait_frequency * self.cfg.ctrl_dt)
        self._pose_weights = np.array(self.cfg.reward_config.pose_weights, dtype=self._np_dtype)
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError(
                f"pose_weights length {self._pose_weights.shape[0]} does not match dof count {self._num_action}"
            )
        self._init_reward_functions()
        self._init_obs_space()

    def _init_reward_functions(self):
        self._reward_fns = {
            "tracking_lin_vel": lambda s: self._reward_tracking_lin_vel(s, s.info["commands"]),
            "tracking_ang_vel": lambda s: self._reward_tracking_ang_vel(s, s.info["commands"]),
            "feet_phase": self._reward_feet_phase,
            "lin_vel_z": self._reward_lin_vel_z,
            "orientation": self._reward_orientation,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "action_rate": lambda s: self._reward_action_rate(s.info),
            "base_height": self._reward_base_height,
            # "similar_to_default": self._reward_similar_to_default,
            "pose": self._reward_pose,
        }

    def _init_obs_space(self):
        num_dof_vel = self._num_action
        num_joint_angle = self._num_action
        num_linvel = 3
        num_gyro = 3
        num_gravity = 3
        num_actions = self._num_action
        num_command = 3

        num_obs = num_linvel + num_gyro + num_gravity + num_joint_angle + num_dof_vel + num_actions + num_command
        self._observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(num_obs,), dtype=float
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def _reward_base_height(self, state: MjNpEnvState):
        base_height = state.physics_state[:, self._idx_qpos + 2]
        target_height = self._cfg.reward_config.base_height_target
        return np.square(base_height - target_height)

    def _reward_similar_to_default(self, state: MjNpEnvState):
        return np.sum(np.abs(self.get_dof_pos(state) - self.default_angles), axis=1)

    def _reward_ang_vel_xy(self, state: MjNpEnvState):
        gyro = self.get_gyro(state)
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_orientation(self, state: MjNpEnvState):
        torso_upvector = state.sensor_data[:, self._idx_torso_upvector]
        return np.sum(np.square(torso_upvector[:, :2]), axis=1)

    def _reward_pose(self, state: MjNpEnvState):
        dof_pos = self.get_dof_pos(state)
        pose_error = np.square(dof_pos - self.default_angles)
        return np.sum(pose_error * self._pose_weights[None, :], axis=1)

    def _expected_foot_height(self, phase: np.ndarray) -> np.ndarray:
        x = (phase + math.pi) / (2.0 * math.pi)
        swing_height = self.cfg.reward_config.feet_phase_swing_height
        stance_x = np.clip(2.0 * x, 0.0, 1.0)
        swing_x = np.clip(2.0 * x - 1.0, 0.0, 1.0)
        stance = swing_height * (stance_x * stance_x * (3.0 - 2.0 * stance_x))
        swing = swing_height * (1.0 - swing_x * swing_x * (3.0 - 2.0 * swing_x))
        return np.where(x <= 0.5, stance, swing)

    def _reward_feet_phase(self, state: MjNpEnvState):
        phase = state.info["gait_phase"]
        left_phase = phase
        right_phase = phase + math.pi
        left_expected = self._expected_foot_height(left_phase)
        right_expected = self._expected_foot_height(right_phase)
        left_height, right_height = self._get_feet_height_rel_ground(state)
        total_error = np.square(left_height - left_expected) + np.square(right_height - right_expected)
        return np.exp(-total_error / self.cfg.reward_config.feet_phase_tracking_sigma)

    def _get_feet_height_rel_ground(self, state: MjNpEnvState) -> tuple[np.ndarray, np.ndarray]:
        left_world_z = state.sensor_data[:, self._idx_left_foot_pos][:, 2]
        right_world_z = state.sensor_data[:, self._idx_right_foot_pos][:, 2]
        # Flat terrain: floor plane is z=0. Keep this as explicit interface for future terrain queries.
        ground_z = 0.0
        return left_world_z - ground_z, right_world_z - ground_z

    def _advance_gait_phase(self, info: dict):
        phase = info.get("gait_phase")
        if phase is None:
            phase = np.zeros((self._num_envs,), dtype=self._np_dtype)
            info["gait_phase"] = phase
        phase += self._gait_phase_delta
        phase = np.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
        info["gait_phase"] = phase

    def _get_obs(self, state: MjNpEnvState, info: dict) -> np.ndarray:
        linear_vel = self.get_local_linvel(state)
        gyro = self.get_gyro(state)
        local_gravity = -self.get_upvector(state)
        dof_pos = self.get_dof_pos(state)
        dof_vel = self.get_dof_vel(state)

        noise_cfg = self.cfg.noise_config
        if noise_cfg.level > 0.0:
            def add_noise(val, scale):
                noise = (np.random.uniform(size=val.shape).astype(self._np_dtype) * 2.0 - 1.0) * noise_cfg.level * scale
                return val + noise

            gyro = add_noise(gyro, noise_cfg.scale_gyro)
            local_gravity = add_noise(local_gravity, noise_cfg.scale_gravity)
            dof_pos = add_noise(dof_pos, noise_cfg.scale_joint_angle)
            dof_vel = add_noise(dof_vel, noise_cfg.scale_joint_vel)
            linear_vel = add_noise(linear_vel, noise_cfg.scale_linvel)

        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info["current_actions"]

        return np.concatenate([linear_vel, gyro, local_gravity, diff, dof_vel, last_actions, command], axis=1)

    def update_observation(self, state: MjNpEnvState):
        obs = self._get_obs(state, state.info)
        state.obs = obs
        return state

    def _compute_rewards(self, state: MjNpEnvState) -> MjNpEnvState:
        self._advance_gait_phase(state.info)
        total_reward = np.zeros((self._num_envs,), dtype=self._np_dtype)
        step_count = state.info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else state.info.get("log", {})

        for name, scale in self.cfg.reward_config.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](state)
            weighted_rew = rew * scale
            total_reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        state.info["log"] = log
        state.info["reward_components"] = {}
        total_reward *= self.cfg.ctrl_dt
        state.reward = total_reward
        return state

    def update_terminated(self, state: MjNpEnvState):
        local_gravity = -self.get_upvector(state)
        sin_limit = math.sin(math.radians(self.cfg.reward_config.max_tilt_deg))
        bad_roll_pitch = np.logical_or(
            np.abs(local_gravity[:, 0]) > sin_limit,
            np.abs(local_gravity[:, 1]) > sin_limit,
        )
        base_height = state.physics_state[:, self._idx_qpos + 2]
        low_height = base_height < self.cfg.reward_config.min_base_height
        state.terminated = np.logical_or(bad_roll_pitch, low_height)
        return state

    def update_state(self, state: MjNpEnvState, obs_required: bool = True) -> MjNpEnvState:
        state = self.update_terminated(state)
        state = self._compute_rewards(state)
        if obs_required:
            state = self.update_observation(state)
        return state

    def resample_commands(self, num_envs: int):
        low = np.array(self.cfg.commands.vel_limit[0], dtype=self._np_dtype)
        high = np.array(self.cfg.commands.vel_limit[1], dtype=self._np_dtype)
        return low + (high - low) * np.random.uniform(size=(num_envs, 3)).astype(self._np_dtype)

    def reset(self, env_indices: np.ndarray):
        num_reset = len(env_indices)
        init_qpos_np = np.asarray(self._init_qpos, dtype=np.float64)
        init_dof_vel_np = np.asarray(self._init_dof_vel, dtype=np.float64)
        qpos_batch = np.broadcast_to(init_qpos_np[None, :], (num_reset, init_qpos_np.shape[0])).copy()
        qvel_batch = np.zeros((num_reset, self.nv), dtype=np.float64)
        qvel_batch[:, 6:] = init_dof_vel_np

        dxy = np.random.uniform(-0.2, 0.2, (num_reset, 2))
        qpos_batch[:, 0:2] += dxy
        yaw = np.random.uniform(-(math.pi / 6.0), math.pi / 6.0, num_reset)
        quat_yaw = np_yaw_to_quat(yaw)
        qpos_batch[:, 3:7] = np_quat_mul(qpos_batch[:, 3:7], quat_yaw)
        qvel_batch[:, 0:6] = np.random.uniform(-0.2, 0.2, (num_reset, 6))

        commands = self.resample_commands(num_reset)
        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=self._np_dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=self._np_dtype),
            "commands": commands,
            "gait_phase": (np.random.uniform(size=(num_reset,)).astype(self._np_dtype) * 2.0 - 1.0) * math.pi,
        }

        obs_physics_state_np = np.zeros((num_reset, self.physics_state_dim), dtype=np.float64)
        obs_physics_state_np[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch
        obs_physics_state_np[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch

        sensor_batch = self._compute_sensor_batch_from_state(obs_physics_state_np)
        obs_physics_state = np.asarray(obs_physics_state_np, dtype=self._np_dtype)

        obs_state = MjNpEnvState(
            physics_state=obs_physics_state,
            sensor_data=sensor_batch,
            obs=None,
            reward=None,
            terminated=None,
            truncated=None,
            ctrl=None,
            info=info,
        )
        obs_batch = self._get_obs(obs_state, info)
        return obs_physics_state, obs_batch, info

    def _reward_tracking_lin_vel(self, state: MjNpEnvState, commands: np.ndarray):
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(state)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, state: MjNpEnvState, commands: np.ndarray):
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(state)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)


# ---------------------------------------------------------------------------
# SAC-specific configuration and environment
# ---------------------------------------------------------------------------

@dataclass
class SACCommands:
    vel_limit = [
        [-1.0, -0.5, -1.0],
        [1.0, 0.5, 1.0],
    ]
    zero_cmd_prob: float = 0.15


@dataclass
class SACRewardConfig(RewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.0,
            "feet_phase": 3.0,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.25,
            "orientation": -5.0,
            "action_rate": -0.5,
            "pose": -0.5,
            "alive": 10.0,
            "penalty_close_feet_xy": -10.0,
        }
    )
    close_feet_threshold: float = 0.15


@registry.envcfg("G1JoystickFlatTerrainSAC")
@dataclass
class G1JoystickSACCfg(G1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: SACCommands = field(default_factory=SACCommands)
    reward_config: SACRewardConfig = field(default_factory=SACRewardConfig)


@registry.env("G1JoystickFlatTerrainSAC", sim_backend="mujoco")
class G1WalkTaskMjSAC(G1WalkTaskMj):
    """SAC variant: obs scaling, sin/cos gait phase, close-feet penalty, wider randomization."""

    def _init_reward_functions(self):
        super()._init_reward_functions()
        self._reward_fns["penalty_close_feet_xy"] = self._reward_penalty_close_feet_xy

    def _init_obs_space(self):
        num_obs = (
            3   # linvel
            + 3   # gyro
            + 3   # gravity
            + self._num_action  # joint angle diff
            + self._num_action  # dof_vel
            + self._num_action  # last actions
            + 3   # command
            + 2   # gait phase sin/cos
        )
        self._observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(num_obs,), dtype=float
        )

    def _reward_penalty_close_feet_xy(self, state: MjNpEnvState):
        left_xy = state.sensor_data[:, self._idx_left_foot_pos][:, :2]
        right_xy = state.sensor_data[:, self._idx_right_foot_pos][:, :2]
        dist = np.linalg.norm(left_xy - right_xy, axis=1)
        return (dist < self.cfg.reward_config.close_feet_threshold).astype(self._np_dtype)

    def _get_obs(self, state: MjNpEnvState, info: dict) -> np.ndarray:
        linear_vel = self.get_local_linvel(state)
        gyro = self.get_gyro(state) * 0.25
        local_gravity = -self.get_upvector(state)
        dof_pos = self.get_dof_pos(state)
        dof_vel = self.get_dof_vel(state) * 0.05

        noise_cfg = self.cfg.noise_config
        if noise_cfg.level > 0.0:
            def add_noise(val, scale):
                noise = (np.random.uniform(size=val.shape).astype(self._np_dtype) * 2.0 - 1.0) * noise_cfg.level * scale
                return val + noise

            gyro = add_noise(gyro, noise_cfg.scale_gyro)
            local_gravity = add_noise(local_gravity, noise_cfg.scale_gravity)
            dof_pos = add_noise(dof_pos, noise_cfg.scale_joint_angle)
            dof_vel = add_noise(dof_vel, noise_cfg.scale_joint_vel)
            linear_vel = add_noise(linear_vel, noise_cfg.scale_linvel)

        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info["current_actions"]
        phase = info["gait_phase"]
        gait_obs = np.stack([np.sin(phase), np.cos(phase)], axis=1)

        return np.concatenate([linear_vel, gyro, local_gravity, diff, dof_vel, last_actions, command, gait_obs], axis=1)

    def resample_commands(self, num_envs: int):
        low = np.array(self.cfg.commands.vel_limit[0], dtype=self._np_dtype)
        high = np.array(self.cfg.commands.vel_limit[1], dtype=self._np_dtype)
        commands = low + (high - low) * np.random.uniform(size=(num_envs, 3)).astype(self._np_dtype)
        zero_mask = np.random.uniform(size=(num_envs,)) < self.cfg.commands.zero_cmd_prob
        commands[zero_mask] = 0.0
        return commands

    def _reset_body(self, num_reset: int):
        """Wider randomization for SAC."""
        init_qpos_np = np.asarray(self._init_qpos, dtype=np.float64)
        init_dof_vel_np = np.asarray(self._init_dof_vel, dtype=np.float64)
        qpos_batch = np.broadcast_to(init_qpos_np[None, :], (num_reset, init_qpos_np.shape[0])).copy()
        qvel_batch = np.zeros((num_reset, self.nv), dtype=np.float64)
        qvel_batch[:, 6:] = init_dof_vel_np

        dxy = np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos_batch[:, 0:2] += dxy
        yaw = np.random.uniform(-math.pi, math.pi, num_reset)
        quat_yaw = np_yaw_to_quat(yaw)
        qpos_batch[:, 3:7] = np_quat_mul(qpos_batch[:, 3:7], quat_yaw)
        qvel_batch[:, 0:6] = np.random.uniform(-0.5, 0.5, (num_reset, 6))
        return qpos_batch, qvel_batch

    def reset(self, env_indices: np.ndarray):
        # Ensure env_indices is numpy array
        env_indices = np.asarray(env_indices, dtype=np.int32)

        # Flush episodic log for truncated envs
        if self._enable_reward_log and len(env_indices) > 0:
            active_mask = self._episode_steps[env_indices] > 0
            active_idx = env_indices[active_mask]
            if len(active_idx) > 0:
                log = {}
                for name in self._episode_reward_sums:
                    log[f"reward/{name}"] = float(
                        np.mean(self._episode_reward_sums[name][active_idx]) / self._max_episode_steps
                    )
                self._pending_reset_log = log
        for name in self._episode_reward_sums:
            self._episode_reward_sums[name][env_indices] = 0.0
        self._episode_steps[env_indices] = 0

        num_reset = len(env_indices)
        qpos_batch, qvel_batch = self._reset_body(num_reset)

        commands = self.resample_commands(num_reset)
        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=self._np_dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=self._np_dtype),
            "commands": commands,
            "gait_phase": (np.random.uniform(size=(num_reset,)).astype(self._np_dtype) * 2.0 - 1.0) * math.pi,
        }

        obs_physics_state_np = np.zeros((num_reset, self.physics_state_dim), dtype=np.float64)
        obs_physics_state_np[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch
        obs_physics_state_np[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch

        sensor_batch = self._compute_sensor_batch_from_state(obs_physics_state_np)
        obs_physics_state = np.asarray(obs_physics_state_np, dtype=self._np_dtype)

        obs_state = MjNpEnvState(
            physics_state=obs_physics_state,
            sensor_data=sensor_batch,
            obs=None, reward=None, terminated=None, truncated=None, ctrl=None,
            info=info,
        )
        obs_batch = self._get_obs(obs_state, info)
        return obs_physics_state, obs_batch, info
