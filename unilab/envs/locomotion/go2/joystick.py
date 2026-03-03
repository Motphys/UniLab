from __future__ import annotations

from etils import epath
import gymnasium as gym
import math
try:
    import mlx.core as mx
except Exception:
    mx = None
import numpy as np
from dataclasses import dataclass, field

from unilab.envs import registry
from unilab.envs.mujoco_env.mj_env import MjMlxEnvState
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat

from unilab.envs.locomotion.go2.base import Go2BaseMjEnv, Go2BaseCfg

# ----------------- Configuration -----------------


@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.42]


@dataclass
class Commands:
    vel_limit = [
        [0.5, 0.0, 0.0],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
        [0.5, 0.0, 0.0],  # max
    ]


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        }
    )

    tracking_sigma: float = 0.25
    base_height_target: float = 0.3


@registry.envcfg("Go2JoystickFlatTerrain")
@dataclass
class Go2JoystickCfg(Go2BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig = field(default_factory=RewardConfig)


# ----------------- Environment -----------------


@registry.env("Go2JoystickFlatTerrain", sim_backend="mujoco")
class Go2WalkTaskMj(Go2BaseMjEnv):
    def __init__(self, cfg: Go2JoystickCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._enable_reward_log = True

        self._init_reward_functions()
        self._init_obs_space()

    def _init_reward_functions(self):
        """Register reward functions."""
        # Genesis go2_train.py reward terms.
        self._reward_fns = {
            "tracking_lin_vel": lambda s: self._reward_tracking_lin_vel(s, s.info["commands"]),
            "tracking_ang_vel": lambda s: self._reward_tracking_ang_vel(s, s.info["commands"]),
            "lin_vel_z": self._reward_lin_vel_z,
            "action_rate": lambda s: self._reward_action_rate(s.info),
            "base_height": lambda s: self._reward_base_height(s),
            "similar_to_default": lambda s: self._reward_similar_to_default(s),
        }

    def _init_obs_space(self):
        num_dof_vel = self._num_dof_vel
        num_joint_angle = self._num_dof_pos
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

    def _reward_base_height(self, state: MjMlxEnvState):
        # Penalize base height deviation from target.
        base_height = state.physics_state[:, self._idx_qpos + 2]
        target_height = self._cfg.reward_config.base_height_target
        return mx.square(base_height - target_height)

    def _reward_similar_to_default(self, state: MjMlxEnvState):
        # Penalize joint pose deviations from default posture.
        return mx.sum(mx.abs(self.get_dof_pos(state) - self.default_angles), axis=1)

    def _get_obs(self, state: MjMlxEnvState, info: dict) -> mx.array:
        # Get raw data (copy to allow noise injection without side effects)
        linear_vel = self.get_local_linvel(state)
        gyro = self.get_gyro(state)
        local_gravity = -self.get_upvector(state)
        dof_pos = self.get_dof_pos(state)
        dof_vel = self.get_dof_vel(state)

        # Apply Observation Noise if enabled
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
            [
                linear_vel,
                gyro,
                local_gravity,
                diff,
                dof_vel,
                last_actions,
                command,
            ],
            axis=1,
        )
        return obs

    def update_state(self, state: MjMlxEnvState, obs_required: bool = True) -> MjMlxEnvState:
        # 1. Check Termination
        state = self.update_terminated(state)

        # 2. Compute Rewards
        state = self._compute_rewards(state)

        # 3. Update Observation (if required)
        if obs_required:
            state = self.update_observation(state)

        return state

    def update_observation(self, state: MjMlxEnvState):
        obs = self._get_obs(state, state.info)
        state.obs = obs
        return state

    def _compute_rewards(self, state: MjMlxEnvState) -> MjMlxEnvState:
        total_reward = mx.zeros((self._num_envs,), dtype=self._mlx_dtype)

        # Only compute per-component logging every 4th step to reduce np.mean overhead
        step_count = state.info.get("steps", mx.zeros((self._num_envs,), dtype=mx.uint32))
        should_log = self._enable_reward_log and (int(step_count[0].item()) % 4 == 0)

        log = {} if should_log else state.info.get("log", {})

        for name, scale in self.cfg.reward_config.scales.items():
            if scale == 0:
                continue
            if name not in self._reward_fns:
                continue

            rew = self._reward_fns[name](state)
            weighted_rew = rew * scale
            total_reward += weighted_rew

            if should_log:
                log[f"reward/{name}"] = float(mx.mean(weighted_rew).item())

        state.info["log"] = log
        state.info["reward_components"] = {}

        # Match genesis behavior: sum(reward_i * scale_i * dt).
        total_reward *= self.cfg.ctrl_dt

        state.reward = total_reward
        return state

    def update_terminated(self, state: MjMlxEnvState) -> MjMlxEnvState:
        # Genesis termination uses roll/pitch absolute angle > 10 degrees.
        local_gravity = -self.get_upvector(state)
        sin_limit = math.sin(math.radians(10.0))
        bad_roll_or_pitch = mx.logical_or(
            mx.abs(local_gravity[:, 0]) > sin_limit,
            mx.abs(local_gravity[:, 1]) > sin_limit,
        )
        state.terminated = bad_roll_or_pitch
        return state

    def resample_commands(self, num_envs: int):
        low = mx.array(self.cfg.commands.vel_limit[0], dtype=self._mlx_dtype)
        high = mx.array(self.cfg.commands.vel_limit[1], dtype=self._mlx_dtype)
        commands = low + (high - low) * mx.random.uniform(shape=(num_envs, 3), dtype=self._mlx_dtype)
        return commands

    def reset(self, env_indices: mx.array) -> tuple[mx.array, mx.array, dict]:
        num_reset = len(env_indices)

        # Build reset states on host to avoid MLX->NumPy sync in sensor-forward path.
        init_qpos_np = np.asarray(self._init_qpos, dtype=np.float64)
        init_dof_vel_np = np.asarray(self._init_dof_vel, dtype=np.float64)
        qpos_batch = np.broadcast_to(init_qpos_np[None, :], (num_reset, init_qpos_np.shape[0])).copy()
        qvel_batch = np.zeros((num_reset, self.nv), dtype=np.float64)
        qvel_batch[:, 6:] = init_dof_vel_np

        # Domain Randomization
        dxy = np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos_batch[:, 0:2] += dxy
        yaw = np.random.uniform(-math.pi, math.pi, num_reset)
        quat_yaw = np_yaw_to_quat(yaw)
        qpos_batch[:, 3:7] = np_quat_mul(qpos_batch[:, 3:7], quat_yaw)
        qvel_batch[:, 0:6] = np.random.uniform(-0.5, 0.5, (num_reset, 6))

        commands = self.resample_commands(num_reset)

        info = {
            "current_actions": mx.zeros((num_reset, self._num_action), dtype=self._mlx_dtype),
            "last_actions": mx.zeros((num_reset, self._num_action), dtype=self._mlx_dtype),
            "commands": commands,
        }

        sensor_batch = self._compute_sensor_batch_from_qpos_qvel(qpos_batch, qvel_batch)
        qpos_batch_mx = mx.array(qpos_batch, dtype=self._mlx_dtype)
        qvel_batch_mx = mx.array(qvel_batch, dtype=self._mlx_dtype)

        # Update global sensor cache without host-side index conversion.
        if hasattr(self, "_state") and self._state is not None:
            self._state.sensor_data = self._scatter_rows(self._state.sensor_data, env_indices, sensor_batch)

        # Reconstruct physics state
        obs_physics_state = mx.zeros((num_reset, self.physics_state_dim), dtype=self._mlx_dtype)
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

        # Call _get_obs ONCE for the entire batch
        obs_batch = self._get_obs(obs_state, info)

        # MjMlxEnv expects: new_physics_states, new_obs, info
        return obs_physics_state, obs_batch, info

    def _reward_tracking_lin_vel(self, state, commands: mx.array):
        lin_vel_error = mx.sum(mx.square(commands[:, :2] - self.get_local_linvel(state)[:, :2]), axis=1)
        return mx.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, state, commands: mx.array):
        ang_vel_error = mx.square(commands[:, 2] - self.get_gyro(state)[:, 2])
        return mx.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)
