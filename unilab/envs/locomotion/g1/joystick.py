from dataclasses import dataclass, field

from etils import epath
import gymnasium as gym
import math
import mlx.core as mx

from unilab.envs import registry
from unilab.envs.mujoco_env.mj_env import MjMlxEnvState
from unilab.envs.utils.math_utils import axis_angle_to_quat, quat_mul
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseMjEnv


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
        self._idx_left_foot_pos = self._get_sensor_indices("left_foot_pos")
        self._idx_right_foot_pos = self._get_sensor_indices("right_foot_pos")
        if self._idx_left_foot_pos is None or self._idx_right_foot_pos is None:
            raise ValueError("Sensors 'left_foot_pos' and 'right_foot_pos' are required for feet_phase reward.")
        self._gait_phase_delta = float(2.0 * math.pi * self.cfg.reward_config.gait_frequency * self.cfg.ctrl_dt)
        self._pose_weights = mx.array(self.cfg.reward_config.pose_weights, dtype=mx.float32)
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

    def _reward_base_height(self, state: MjMlxEnvState):
        base_height = state.physics_state[:, self._idx_qpos + 2]
        target_height = self._cfg.reward_config.base_height_target
        return mx.square(base_height - target_height)

    def _reward_similar_to_default(self, state: MjMlxEnvState):
        return mx.sum(mx.abs(self.get_dof_pos(state) - self.default_angles), axis=1)

    def _reward_ang_vel_xy(self, state: MjMlxEnvState):
        gyro = self.get_gyro(state)
        return mx.sum(mx.square(gyro[:, :2]), axis=1)

    def _reward_orientation(self, state: MjMlxEnvState):
        torso_upvector = state.sensor_data[:, self._idx_torso_upvector]
        return mx.sum(mx.square(torso_upvector[:, :2]), axis=1)

    def _reward_pose(self, state: MjMlxEnvState):
        dof_pos = self.get_dof_pos(state)
        pose_error = mx.square(dof_pos - self.default_angles)
        return mx.sum(pose_error * self._pose_weights[None, :], axis=1)

    def _expected_foot_height(self, phase: mx.array) -> mx.array:
        x = (phase + math.pi) / (2.0 * math.pi)
        swing_height = self.cfg.reward_config.feet_phase_swing_height
        stance_x = mx.clip(2.0 * x, 0.0, 1.0)
        swing_x = mx.clip(2.0 * x - 1.0, 0.0, 1.0)
        stance = swing_height * (stance_x * stance_x * (3.0 - 2.0 * stance_x))
        swing = swing_height * (1.0 - swing_x * swing_x * (3.0 - 2.0 * swing_x))
        return mx.where(x <= 0.5, stance, swing)

    def _reward_feet_phase(self, state: MjMlxEnvState):
        phase = state.info["gait_phase"]
        left_phase = phase
        right_phase = phase + math.pi
        left_expected = self._expected_foot_height(left_phase)
        right_expected = self._expected_foot_height(right_phase)
        left_height, right_height = self._get_feet_height_rel_ground(state)
        total_error = mx.square(left_height - left_expected) + mx.square(right_height - right_expected)
        return mx.exp(-total_error / self.cfg.reward_config.feet_phase_tracking_sigma)

    def _get_feet_height_rel_ground(self, state: MjMlxEnvState) -> tuple[mx.array, mx.array]:
        left_world_z = state.sensor_data[:, self._idx_left_foot_pos][:, 2]
        right_world_z = state.sensor_data[:, self._idx_right_foot_pos][:, 2]
        # Flat terrain: floor plane is z=0. Keep this as explicit interface for future terrain queries.
        ground_z = 0.0
        return left_world_z - ground_z, right_world_z - ground_z

    def _advance_gait_phase(self, info: dict):
        phase = info.get("gait_phase")
        if phase is None:
            phase = mx.zeros((self._num_envs,), dtype=mx.float32)
            info["gait_phase"] = phase
        phase += self._gait_phase_delta
        phase = mx.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
        info["gait_phase"] = phase

    def _get_obs(self, state: MjMlxEnvState, info: dict) -> mx.array:
        linear_vel = self.get_local_linvel(state)
        gyro = self.get_gyro(state)
        local_gravity = -self.get_upvector(state)
        dof_pos = self.get_dof_pos(state)
        dof_vel = self.get_dof_vel(state)

        noise_cfg = self.cfg.noise_config
        if noise_cfg.level > 0.0:
            def add_noise(val, scale):
                noise = (mx.random.uniform(shape=val.shape, dtype=mx.float32) * 2.0 - 1.0) * noise_cfg.level * scale
                return val + noise

            gyro = add_noise(gyro, noise_cfg.scale_gyro)
            local_gravity = add_noise(local_gravity, noise_cfg.scale_gravity)
            dof_pos = add_noise(dof_pos, noise_cfg.scale_joint_angle)
            dof_vel = add_noise(dof_vel, noise_cfg.scale_joint_vel)
            linear_vel = add_noise(linear_vel, noise_cfg.scale_linvel)

        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info["current_actions"]

        return mx.concatenate([linear_vel, gyro, local_gravity, diff, dof_vel, last_actions, command], axis=1)

    def update_observation(self, state: MjMlxEnvState):
        obs = self._get_obs(state, state.info)
        return state.replace(obs=obs)

    def _compute_rewards(self, state: MjMlxEnvState) -> MjMlxEnvState:
        self._advance_gait_phase(state.info)
        total_reward = mx.zeros((self._num_envs,), dtype=mx.float32)
        log = {}

        for name, scale in self.cfg.reward_config.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](state)
            weighted_rew = rew * scale
            total_reward += weighted_rew
            log[f"reward/{name}"] = float(mx.mean(weighted_rew).item())

        state.info["log"] = log
        state.info["reward_components"] = {}
        total_reward *= self.cfg.ctrl_dt
        return state.replace(reward=total_reward)

    def update_terminated(self, state: MjMlxEnvState):
        local_gravity = -self.get_upvector(state)
        sin_limit = math.sin(math.radians(self.cfg.reward_config.max_tilt_deg))
        bad_roll_pitch = mx.logical_or(
            mx.abs(local_gravity[:, 0]) > sin_limit,
            mx.abs(local_gravity[:, 1]) > sin_limit,
        )
        base_height = state.physics_state[:, self._idx_qpos + 2]
        low_height = base_height < self.cfg.reward_config.min_base_height
        return state.replace(terminated=mx.logical_or(bad_roll_pitch, low_height))

    def update_state(self, state: MjMlxEnvState, obs_required: bool = True) -> MjMlxEnvState:
        state = self.update_terminated(state)
        state = self._compute_rewards(state)
        if obs_required:
            state = self.update_observation(state)
        return state

    def resample_commands(self, num_envs: int):
        low = mx.array(self.cfg.commands.vel_limit[0], dtype=mx.float32)
        high = mx.array(self.cfg.commands.vel_limit[1], dtype=mx.float32)
        return low + (high - low) * mx.random.uniform(shape=(num_envs, 3), dtype=mx.float32)

    def reset(self, env_indices: mx.array):
        num_reset = len(env_indices)
        qpos_batch = mx.broadcast_to(self._init_qpos[None, :], (num_reset, self._init_qpos.shape[0]))
        qvel_batch = mx.zeros((num_reset, self.nv), dtype=mx.float32)
        qvel_batch[:, 6:] = mx.broadcast_to(self._init_dof_vel[None, :], (num_reset, self._init_dof_vel.shape[0]))

        # Light randomization keeps fast convergence while avoiding overfitting.
        dxy = (mx.random.uniform(shape=(num_reset, 2), dtype=mx.float32) * 2.0 - 1.0) * 0.2
        qpos_batch[:, 0:2] += dxy
        yaw = (mx.random.uniform(shape=(num_reset,), dtype=mx.float32) * 2.0 - 1.0) * (math.pi / 6.0)
        axis = mx.zeros((num_reset, 3), dtype=mx.float32)
        axis[:, 2] = 1.0
        quat_yaw = axis_angle_to_quat(axis, yaw)
        qpos_batch[:, 3:7] = quat_mul(qpos_batch[:, 3:7], quat_yaw)
        qvel_batch[:, 0:6] = (mx.random.uniform(shape=(num_reset, 6), dtype=mx.float32) * 2.0 - 1.0) * 0.2

        if hasattr(self, "_state") and self._state is not None:
            idx_list = [int(i) for i in env_indices.tolist()]
            prev = self._state.physics_state[idx_list]
            prev[:, 0] = 0.0
            prev[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch
            prev[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch
            idx_act = self._idx_qvel + self.nv
            prev[:, idx_act:] = 0.0
            self._state.physics_state = self._scatter_rows(self._state.physics_state, idx_list, prev)

        commands = self.resample_commands(num_reset)
        info = {
            "current_actions": mx.zeros((num_reset, self._num_action), dtype=mx.float32),
            "last_actions": mx.zeros((num_reset, self._num_action), dtype=mx.float32),
            "commands": commands,
            "gait_phase": (mx.random.uniform(shape=(num_reset,), dtype=mx.float32) * 2.0 - 1.0) * math.pi,
        }

        sensor_batch = self._compute_sensor_batch_from_qpos_qvel(qpos_batch, qvel_batch)

        if hasattr(self, "_state") and self._state is not None:
            idx_list = [int(i) for i in env_indices.tolist()]
            self._state.sensor_data = self._scatter_rows(self._state.sensor_data, idx_list, sensor_batch)

        obs_physics_state = mx.zeros((num_reset, self.physics_state_dim), dtype=mx.float32)
        obs_physics_state[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch
        obs_physics_state[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch

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

    def _reward_tracking_lin_vel(self, state: MjMlxEnvState, commands: mx.array):
        lin_vel_error = mx.sum(mx.square(commands[:, :2] - self.get_local_linvel(state)[:, :2]), axis=1)
        return mx.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, state: MjMlxEnvState, commands: mx.array):
        ang_vel_error = mx.square(commands[:, 2] - self.get_gyro(state)[:, 2])
        return mx.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)
