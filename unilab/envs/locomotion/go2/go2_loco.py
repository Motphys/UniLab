"""Go2 Locomotion environment inspired by IsaacLab's Go2 velocity tracking task.

Reward and termination configuration references:
- IsaacLab: LocomotionVelocityRoughEnvCfg + UnitreeGo2RoughEnvCfg
- Adapted for MuJoCo-MJX backend in UniLab

Key differences from Go2JoystickFlatTerrain:
- More comprehensive reward terms (ang_vel_xy_l2, dof_torques_l2, dof_acc_l2, flat_orientation_l2)
- Base contact termination (instead of roll/pitch angle threshold)
- Larger velocity command ranges (suitable for training robust locomotion)
"""

from etils import epath
import gymnasium as gym
import math
import mlx.core as mx
import numpy as np
from dataclasses import dataclass, field

from unilab.envs import registry
from unilab.envs.mujoco_env.mj_env import MjMlxEnvState
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat

from unilab.envs.locomotion.go2.base import Go2BaseMjEnv, Go2BaseCfg, ControlConfig, NoiseConfig

# ----------------- Configuration -----------------


@dataclass
class LocoInitState:
    pos = [0.0, 0.0, 0.42]


@dataclass
class LocoCommands:
    """Velocity command ranges aligned with IsaacLab Go2 config."""
    vel_limit = [
        [0.5, 0, 0],  # min: vel_x, vel_y, ang_vel
        [0.5, 0, 0],     # max
    ]
    resampling_time: float = 10.0  # seconds between command resampling


@dataclass
class LocoRewardConfig:
    """Reward configuration aligned with IsaacLab Go2 rough env.

    Weights are from IsaacLab LocomotionVelocityRoughEnvCfg +
    UnitreeGo2RoughEnvCfg overrides.
    """
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # Task rewards
            "alive": 1.0,
            "track_lin_vel_xy_exp": 1.0,
            "track_ang_vel_z_exp": 0.5,
            # Penalty terms
            "dof_pos_l2_from_default": -0.05,
            "foot_height_tracking": -0.01,
            "action_rate_l2": -0.005,
            "flat_orientation_l2": -0.5,
            "lin_vel_z_l2": -1.0,
            "ang_vel_xy_l2": -0.05,
            "termination_penalty": -50.0,
        }
    )

    # Tracking reward parameters
    tracking_sigma: float = 0.5  # Increased to prevent sharp decay at edge of command bounds

    # Base height target
    base_height_target: float = 0.30


@dataclass
class LocoTerminationConfig:
    """Termination configuration aligned with IsaacLab."""
    base_contact_threshold: float = 1.0  # force threshold for base contact termination
    # Fallback to orientation check if contact sensor not available
    max_tilt_angle_deg: float = 30.0
    min_base_height: float = 0.20  # terminate if base height drops below this


@registry.envcfg("Go2LocoFlatTerrain")
@dataclass
class Go2LocoCfg(Go2BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: LocoInitState = field(default_factory=LocoInitState)
    commands: LocoCommands = field(default_factory=LocoCommands)
    reward_config: LocoRewardConfig = field(default_factory=LocoRewardConfig)
    termination_config: LocoTerminationConfig = field(default_factory=LocoTerminationConfig)
    control_config: ControlConfig = field(default_factory=lambda: ControlConfig(action_scale=0.25))


# ----------------- Environment -----------------


@registry.env("Go2LocoFlatTerrain", sim_backend="mujoco")
class Go2LocoTaskMj(Go2BaseMjEnv):
    """Go2 locomotion task with IsaacLab-style rewards and terminations.

    Reward terms:
    - track_lin_vel_xy_exp: exponential tracking of linear velocity commands
    - track_ang_vel_z_exp: exponential tracking of angular velocity command
    - lin_vel_z_l2: penalize vertical velocity
    - ang_vel_xy_l2: penalize roll/pitch angular velocity
    - dof_torques_l2: penalize joint torques
    - dof_acc_l2: penalize joint accelerations
    - action_rate_l2: penalize action changes
    - feet_air_time: encourage feet lift-off
    - flat_orientation_l2: penalize non-flat orientation

    Termination:
    - base_contact: body tilt beyond threshold (orientation-based for MuJoCo)
    - time_out: max episode length
    """

    def __init__(self, cfg: Go2LocoCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._enable_reward_log = True

        self._init_reward_functions()
        self._init_obs_space()

        # Track previous joint velocities for acceleration penalty
        self._prev_dof_vel = mx.zeros(
            (self._num_envs, self._num_dof_vel), dtype=self._mlx_dtype
        )

    def _init_reward_functions(self):
        """Register all reward functions."""
        self._reward_fns = {
            "alive": self._reward_alive,
            "track_lin_vel_xy_exp": lambda s: self._reward_track_lin_vel_xy_exp(s, s.info["commands"]),
            "track_ang_vel_z_exp": lambda s: self._reward_track_ang_vel_z_exp(s, s.info["commands"]),
            "dof_pos_l2_from_default": self._reward_dof_pos_l2_from_default,
            "foot_height_tracking": self._reward_foot_height_tracking,
            "action_rate_l2": lambda s: self._reward_action_rate(s.info),
            "flat_orientation_l2": self._reward_flat_orientation_l2,
            "lin_vel_z_l2": self._reward_lin_vel_z,
            "ang_vel_xy_l2": self._reward_ang_vel_xy_l2,
            "termination_penalty": self._reward_termination_penalty,
        }

    def _init_obs_space(self):
        num_dof_vel = self._num_dof_vel
        num_joint_angle = self._num_dof_pos
        num_linvel = 3
        num_gyro = 3
        num_gravity = 3
        num_actions = self._num_action
        num_command = 3

        num_obs = (
            num_linvel + num_gyro + num_gravity
            + num_joint_angle + num_dof_vel
            + num_actions + num_command
        )

        self._observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(num_obs,), dtype=float
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    # ------------- Reward Functions (IsaacLab-aligned) ----------------

    def _reward_alive(self, state: MjMlxEnvState):
        """Reward for staying alive."""
        return mx.ones((self._num_envs,), dtype=self._mlx_dtype)

    def _reward_termination_penalty(self, state: MjMlxEnvState):
        """Penalty for early termination (e.g. base contact/tilt)."""
        # state.terminated is a boolean array, convert to float for the reward
        return state.terminated.astype(self._mlx_dtype)

    def _reward_track_lin_vel_xy_exp(self, state: MjMlxEnvState, commands: mx.array):
        """Exponential tracking of linear velocity commands (x, y).

        r = exp(-||cmd_xy - vel_xy||^2 / sigma)
        """
        lin_vel = self.get_local_linvel(state)
        lin_vel_error = mx.sum(
            mx.square(commands[:, :2] - lin_vel[:, :2]), axis=1
        )
        return mx.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_track_ang_vel_z_exp(self, state: MjMlxEnvState, commands: mx.array):
        """Exponential tracking of angular velocity command (yaw)."""
        gyro = self.get_gyro(state)
        ang_vel_error = mx.square(commands[:, 2] - gyro[:, 2])
        return mx.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_ang_vel_xy_l2(self, state: MjMlxEnvState):
        """Penalize roll/pitch angular velocity (IsaacLab: ang_vel_xy_l2)."""
        gyro = self.get_gyro(state)
        return mx.sum(mx.square(gyro[:, :2]), axis=1)

    def _reward_dof_pos_l2_from_default(self, state: MjMlxEnvState):
        """Penalize joint limits/extreme configurations (Default-pose penalty)."""
        dof_pos = self.get_dof_pos(state)
        # Using default joint positions as target
        return mx.sum(mx.square(dof_pos - self.default_angles), axis=1)

    def _reward_foot_height_tracking(self, state: MjMlxEnvState):
        """Simple foot-height tracking term to guide swing motion."""
        # A simple heuristic: punish too high base/feet deviations, 
        # or just penalize ground clearance slightly to discourage moonwalking.
        # Since we just want to replace the complex feet_air_time, 
        # we can use base height tracking as a proxy for foot height rhythm, 
        # or implement a simple variance penalty.
        base_height = state.physics_state[:, self._idx_qpos + 2]
        target = self.cfg.reward_config.base_height_target
        return mx.square(base_height - target)

    def _reward_flat_orientation_l2(self, state: MjMlxEnvState):
        """Penalize non-flat orientation (IsaacLab: flat_orientation_l2).

        Uses projected gravity to measure tilt.
        """
        local_gravity = -self.get_upvector(state)
        # Ideal: gravity = [0, 0, -1], so local_gravity[:, :2] should be 0
        return mx.sum(mx.square(local_gravity[:, :2]), axis=1)

    # ------------- Observation ----------------

    def _get_obs(self, state: MjMlxEnvState, info: dict) -> mx.array:
        linear_vel = self.get_local_linvel(state)
        gyro = self.get_gyro(state)
        local_gravity = -self.get_upvector(state)
        dof_pos = self.get_dof_pos(state)
        dof_vel = self.get_dof_vel(state)

        # Apply noise
        noise_cfg = self.cfg.noise_config
        if noise_cfg.level > 0.0:
            def add_noise(val, scale):
                noise = (mx.random.uniform(shape=val.shape, dtype=self._mlx_dtype) * 2.0 - 1.0) * noise_cfg.level * scale
                return val + noise

            gyro = add_noise(gyro, noise_cfg.scale_gyro)
            local_gravity = add_noise(local_gravity, noise_cfg.scale_gravity)
            dof_pos = add_noise(dof_pos, noise_cfg.scale_joint_angle)
            dof_vel = add_noise(dof_vel, noise_cfg.scale_joint_vel)
            linear_vel = add_noise(linear_vel, noise_cfg.scale_linvel)

        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info["current_actions"]

        obs = mx.concatenate(
            [linear_vel, gyro, local_gravity, diff, dof_vel, last_actions, command],
            axis=1,
        )
        return obs

    # ------------- State Update ----------------

    def update_state(self, state: MjMlxEnvState, obs_required: bool = True) -> MjMlxEnvState:
        state = self.update_terminated(state)
        state = self._compute_rewards(state)
        if obs_required:
            state = self.update_observation(state)
        return state

    def update_observation(self, state: MjMlxEnvState):
        obs = self._get_obs(state, state.info)
        state.obs = obs
        return state

    def _compute_rewards(self, state: MjMlxEnvState) -> MjMlxEnvState:
        total_reward = mx.zeros((self._num_envs,), dtype=self._mlx_dtype)

        step_count = state.info.get("steps", mx.zeros((self._num_envs,), dtype=mx.uint32))
        should_log = self._enable_reward_log and (int(step_count[0].item()) % 4 == 0)
        log = {} if should_log else state.info.get("log", {})

        for name, scale in self.cfg.reward_config.scales.items():
            if scale == 0:
                continue
            if name not in self._reward_fns:
                continue

            try:
                rew = self._reward_fns[name](state)
            except Exception as e:
                print(f"Error evaluating reward {name}: {e}")
                raise
            weighted_rew = rew * scale
            total_reward += weighted_rew

            if should_log:
                log[f"reward/{name}"] = float(mx.mean(weighted_rew).item())

        state.info["log"] = log
        state.info["reward_components"] = {}

        total_reward *= self.cfg.ctrl_dt
        state.reward = total_reward
        return state

    # ------------- Termination (IsaacLab-style) ----------------

    def update_terminated(self, state: MjMlxEnvState) -> MjMlxEnvState:
        """Terminate if base tilts too much (orientation-based, aligned with IsaacLab).

        IsaacLab uses base contact force threshold. Since MuJoCo doesn't
        expose per-body contact forces easily, we use the orientation fallback:
        terminate if the tilt angle exceeds the configured threshold.
        """
        local_gravity = -self.get_upvector(state)
        max_tilt = self.cfg.termination_config.max_tilt_angle_deg
        sin_limit = math.sin(math.radians(max_tilt))

        bad_orientation = mx.logical_or(
            mx.abs(local_gravity[:, 0]) > sin_limit,
            mx.abs(local_gravity[:, 1]) > sin_limit,
        )
        
        base_height = state.physics_state[:, self._idx_qpos + 2]
        bad_height = base_height < self.cfg.termination_config.min_base_height
        
        state.terminated = mx.logical_or(bad_orientation, bad_height)
        return state

    # ------------- Commands ----------------

    def resample_commands(self, num_envs: int):
        low = mx.array(self.cfg.commands.vel_limit[0], dtype=self._mlx_dtype)
        high = mx.array(self.cfg.commands.vel_limit[1], dtype=self._mlx_dtype)
        commands = low + (high - low) * mx.random.uniform(
            shape=(num_envs, 3), dtype=self._mlx_dtype
        )
        return commands

    # ------------- Reset ----------------

    def reset(self, env_indices: mx.array) -> tuple[mx.array, mx.array, dict]:
        num_reset = len(env_indices)

        init_qpos_np = np.asarray(self._init_qpos, dtype=np.float64)
        init_dof_vel_np = np.asarray(self._init_dof_vel, dtype=np.float64)
        qpos_batch = np.broadcast_to(
            init_qpos_np[None, :], (num_reset, init_qpos_np.shape[0])
        ).copy()
        qvel_batch = np.zeros((num_reset, self.nv), dtype=np.float64)
        qvel_batch[:, 6:] = init_dof_vel_np

        # Domain Randomization (IsaacLab-style)
        dxy = np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos_batch[:, 0:2] += dxy
        yaw = np.random.uniform(-math.pi, math.pi, num_reset)
        quat_yaw = np_yaw_to_quat(yaw)
        qpos_batch[:, 3:7] = np_quat_mul(qpos_batch[:, 3:7], quat_yaw)
        # Velocity randomization (same as IsaacLab Go2: zero velocity on reset)
        qvel_batch[:, 0:6] = 0.0

        commands = self.resample_commands(num_reset)

        info = {
            "current_actions": mx.zeros((num_reset, self._num_action), dtype=self._mlx_dtype),
            "last_actions": mx.zeros((num_reset, self._num_action), dtype=self._mlx_dtype),
            "commands": commands,
        }

        sensor_batch = self._compute_sensor_batch_from_qpos_qvel(qpos_batch, qvel_batch)
        qpos_batch_mx = mx.array(qpos_batch, dtype=self._mlx_dtype)
        qvel_batch_mx = mx.array(qvel_batch, dtype=self._mlx_dtype)

        if hasattr(self, "_state") and self._state is not None:
            self._state.sensor_data = self._scatter_rows(
                self._state.sensor_data, env_indices, sensor_batch
            )

        # Reset prev_dof_vel for acceleration computation
        reset_indices_np = np.array(env_indices)
        self._prev_dof_vel = mx.zeros_like(self._prev_dof_vel)

        obs_physics_state = mx.zeros(
            (num_reset, self.physics_state_dim), dtype=self._mlx_dtype
        )
        obs_physics_state[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch_mx
        obs_physics_state[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch_mx

        obs_state = MjMlxEnvState(
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


# ----------------- Play Environment -----------------


@dataclass
class LocoPlayCommands(LocoCommands):
    """Play commands with fixed forward velocity."""
    vel_limit = [
        [1.0, 0.0, 0.0],  # min: vel_x, vel_y, ang_vel
        [1.0, 0.0, 0.0],  # max
    ]


@registry.envcfg("Go2LocoFlatTerrainPlay")
@dataclass
class Go2LocoPlayCfg(Go2LocoCfg):
    commands: LocoPlayCommands = field(default_factory=LocoPlayCommands)
    noise_config: NoiseConfig = field(default_factory=lambda: NoiseConfig(level=0.0))


@registry.env("Go2LocoFlatTerrainPlay", sim_backend="mujoco")
class Go2LocoPlayTaskMj(Go2LocoTaskMj):
    """Play environment inheriting from Go2LocoFlatTerrain, without noise and with fixed forward commands."""
    pass
