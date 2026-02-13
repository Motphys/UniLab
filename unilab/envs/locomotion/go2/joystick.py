from etils import epath

import gymnasium as gym
import mujoco
import numpy as np
from dataclasses import dataclass, field


from unilab.envs import registry
from unilab.envs.mujoco_env.mj_env import MjNpEnvState
from unilab.envs.utils.math_utils import quat_rotate_inverse, quat_mul, axis_angle_to_quat

from unilab.envs.locomotion.go2.base import Go2BaseMjEnv, Go2BaseCfg

# ----------------- Configuration -----------------


@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.278]


@dataclass
class Commands:
    vel_limit = [
        [-3.0, -1.5, -6.28],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
        [3.0, 1.5, 6.28],  # max
    ]


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # Tracking
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.0,
            # Base
            "lin_vel_z": -0.5,
            "ang_vel_xy": -0.01,
            "orientation": -1.0,
            # "base_height": -5.0,
            # Other
            "dof_pos_limits": -1.0,
            "pose": 0.5,
            "termination": -1.0,
            "stand_still": -0.2,
            # Regularization
            "torques": -0.0002,
            "action_rate": -0.01,
            "energy": -0.001,
            # Feet
            "feet_clearance": -2.0,
            "feet_height": -0.2,
            "feet_slip": -0.1,
            "feet_air_time": 0.1,
            "base_height": -10.0,  # defaulting to 0 as generic reward
        }
    )

    tracking_sigma: float = 0.25
    max_foot_height: float = 0.1


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

        self._init_reward_functions()
        self._init_obs_space()

    def _init_reward_functions(self):
        """Register reward functions."""
        # Mix of base rewards and task specific rewards
        self._reward_fns = {
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "orientation": self._reward_orientation,
            "torques": self._reward_torques,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": lambda s: self._reward_dof_acc(s, s.info),
            "action_rate": lambda s: self._reward_action_rate(s.info),
            "termination": lambda s: self._reward_termination(s.terminated),
            "dof_pos_limits": lambda s: self._cost_joint_pos_limits(s),
            "pose": lambda s: self._reward_pose(s),
            "energy": lambda s: self._cost_energy(s),
            # Task specific
            "tracking_lin_vel": lambda s: self._reward_tracking_lin_vel(s, s.info["commands"]),
            "tracking_ang_vel": lambda s: self._reward_tracking_ang_vel(s, s.info["commands"]),
            "stand_still": lambda s: self._reward_stand_still(s, s.info["commands"]),
            "feet_air_time": lambda s: self._reward_feet_air_time(s.info["commands"], s.info),
            "feet_clearance": lambda s: self._cost_feet_clearance(s),
            "feet_height": lambda s: self._cost_feet_height(s.info),
            "feet_slip": lambda s: self._cost_feet_slip(s, s.info),
            "base_height": lambda s: self._reward_base_height(s),
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
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(num_obs,), dtype=np.float32
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def _reward_base_height(self, state: MjNpEnvState):
        # Penalize base height deviation from target
        base_height = state.physics_state[:, self._idx_qpos + 2]
        target_height = self._cfg.init_state.pos[2]
        return np.square(base_height - target_height)

    def _get_obs(self, state: MjNpEnvState, info: dict) -> np.ndarray:
        # Get raw data (copy to allow noise injection without side effects)
        linear_vel = self.get_local_linvel(state).copy()
        gyro = self.get_gyro(state).copy()
        local_gravity = info["local_gravity"].copy()
        dof_pos = self.get_dof_pos(state).copy()
        dof_vel = self.get_dof_vel(state).copy()

        # Apply Observation Noise if enabled
        noise_cfg = self.cfg.noise_config
        if noise_cfg.level > 0.0:

            def add_noise(val, scale):
                noise = (np.random.rand(*val.shape) * 2 - 1) * noise_cfg.level * scale
                return val + noise

            gyro = add_noise(gyro, noise_cfg.scale_gyro)
            local_gravity = add_noise(local_gravity, noise_cfg.scale_gravity)
            dof_pos = add_noise(dof_pos, noise_cfg.scale_joint_angle)
            dof_vel = add_noise(dof_vel, noise_cfg.scale_joint_vel)
            linear_vel = add_noise(linear_vel, noise_cfg.scale_linvel)

        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info["current_actions"]

        obs = np.hstack(
            [
                linear_vel,
                gyro,
                local_gravity,
                diff,
                dof_vel,
                last_actions,
                command,
            ]
        )
        return obs

    def update_state(self, state: MjNpEnvState, obs_required: bool = True) -> MjNpEnvState:
        # 1. Update Physics Cache (contacts, foot pos, etc.)
        self._update_cache(state)

        # 2. Check Termination
        state = self.update_terminated(state)

        # 3. Compute Rewards
        state = self._compute_rewards(state)

        # 4. Update Observation (if required)
        if obs_required:
            state = self.update_observation(state)

        return state

    def update_observation(self, state: MjNpEnvState):
        obs = self._get_obs(state, state.info)
        return state.replace(obs=obs)

    def _compute_rewards(self, state: MjNpEnvState) -> MjNpEnvState:
        total_reward = np.zeros(self._num_envs, dtype=np.float32)

        # Initialize dictionary for logging
        log = {}
        # Also store raw components for episodic accumulation in the wrapper
        reward_components = {}

        for name, scale in self.cfg.reward_config.scales.items():
            if scale == 0:
                continue
            if name not in self._reward_fns:
                continue

            rew = self._reward_fns[name](state)
            weighted_rew = rew * scale
            total_reward += weighted_rew

            # Store mean weighted reward per step for logging (gs_playground style)
            log[f"reward/{name}"] = np.mean(weighted_rew)
            # Store raw for aggregation
            reward_components[name] = weighted_rew

        # Log other info metrics
        if "feet_air_time" in state.info:
            log["metrics/feet_air_time"] = np.mean(state.info["feet_air_time"])
        if "contacts" in state.info:
            log["metrics/contact_rate"] = np.mean(state.info.get("contacts", np.zeros_like(total_reward)).astype(float))

        state.info["log"] = log
        state.info["reward_components"] = reward_components

        # Scale reward by dt (mujoco_playground style)
        total_reward *= self.cfg.ctrl_dt

        # Clip reward to non-negative to prevent early suicide (agent terminating
        # on purpose to avoid accumulating negative rewards)
        total_reward = np.clip(total_reward, 0.0, 10000.0)

        return state.replace(reward=total_reward)

    def _update_cache(self, state: MjNpEnvState):
        """Update cached info based on current physics/sensor state."""
        super()._update_cache(state)
        info = state.info
        current_contacts = info["contacts"]

        # C. Update Air Time
        if "feet_air_time" not in info:
            info["feet_air_time"] = np.zeros((self._num_envs, 4), dtype=np.float32)

        info["feet_air_time"] += self.cfg.ctrl_dt

        # Capture air time at contact (before reset)
        info["air_time_at_contact"] = info["feet_air_time"] * current_contacts

        # Reset air time for feet in contact
        info["feet_air_time"] *= ~current_contacts

        # D. Update Swing Peak & Foot Tracking (requires global foot pos)
        self._update_foot_tracking(state, info, current_contacts)

    def _update_foot_tracking(self, state, info, current_contacts):
        # Calculate foot global Z for swing height reward
        batch_size = current_contacts.shape[0]

        if "swing_peak" not in info:
            info["swing_peak"] = np.zeros((batch_size, 4), dtype=np.float32)

        foot_rel_pos_list = []
        for idx_list in self.foot_pos_sensor_indices:
            foot_rel_pos_list.append(state.sensor_data[:, idx_list])
        foot_rel_pos = np.stack(foot_rel_pos_list, axis=1)  # (N, 4, 3)

        base_pos = state.sensor_data[:, self.idx_global_pos]  # (N, 3)
        base_quat = state.sensor_data[:, self.idx_orientation]  # (N, 4)

        # Calculate Forward Kinematics manually: Global = Base + R * Local
        base_quat_conj = base_quat.copy()
        base_quat_conj[:, 1:] *= -1  # Conjugate

        foot_global_z = np.zeros((batch_size, 4), dtype=np.float32)
        for i in range(4):
            vec = foot_rel_pos[:, i, :]
            # quat_rotate_inverse(q_conj, v) = q * v * q^-1 (Standard Rotate)
            vec_rot = quat_rotate_inverse(base_quat_conj, vec)
            foot_global_z[:, i] = (vec_rot + base_pos)[:, 2]

        info["foot_pos_z"] = foot_global_z

        # update swing peak
        p_fz = foot_global_z
        info["swing_peak"] = np.maximum(info["swing_peak"], p_fz)
        # Reset peak on contact
        info["swing_peak"] *= ~current_contacts
        info["swing_peak"] = np.maximum(info["swing_peak"], foot_global_z)
        info["swing_peak_at_contact"] = info["swing_peak"] * current_contacts
        info["swing_peak"] *= ~current_contacts

    def update_terminated(self, state: MjNpEnvState) -> MjNpEnvState:
        local_gravity = state.info["local_gravity"]
        up_z = -local_gravity[:, 2]

        is_fallen = up_z <= 0.5

        return state.replace(
            terminated=is_fallen,
        )

    def resample_commands(self, num_envs: int):
        commands = np.random.uniform(
            low=self.cfg.commands.vel_limit[0],
            high=self.cfg.commands.vel_limit[1],
            size=(num_envs, 3),
        )

        # Standard practice: set small percentage of commands to zero to train standing still
        mask = np.random.random(num_envs) < 0.05
        commands[mask] = 0.0

        return commands

    def reset(self, env_indices: np.ndarray) -> tuple[np.ndarray, dict]:
        num_reset = len(env_indices)

        qpos_batch = np.tile(self._init_qpos, (num_reset, 1))

        qvel_batch = np.zeros((num_reset, self.nv), dtype=np.float64)
        qvel_batch[:, 6:] = self._init_dof_vel

        # Domain Randomization (joystick.py reference)
        # 1. Base Position Noise (x, y) ~ U(-0.5, 0.5)
        dxy = np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos_batch[:, 0:2] += dxy

        # 2. Base Orientation Noise (yaw) ~ U(-pi, pi)
        yaw = np.random.uniform(-np.pi, np.pi, num_reset)
        axis = np.zeros((num_reset, 3))
        axis[:, 2] = 1.0  # Z-axis
        quat_yaw = axis_angle_to_quat(axis, yaw)

        # q_new = q_old * q_yaw (Quaternion multiplication)
        qpos_batch[:, 3:7] = quat_mul(qpos_batch[:, 3:7], quat_yaw)

        # 3. Base Velocity Noise ~ U(-0.5, 0.5) for 6DoF
        qvel_batch[:, 0:6] = np.random.uniform(-0.5, 0.5, (num_reset, 6))

        if hasattr(self, "_state") and self._state is not None:
            self._state.physics_state[env_indices, 0] = 0.0
            self._state.physics_state[env_indices, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch
            self._state.physics_state[env_indices, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch
            idx_act = self._idx_qvel + self.nv
            self._state.physics_state[env_indices, idx_act:] = 0.0

        commands = self.resample_commands(num_reset)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": commands,
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "feet_air_time": np.zeros((num_reset, 4), dtype=np.float32),
            "contacts": np.zeros((num_reset, 4), dtype=bool),
        }

        sensor_batch = np.zeros((num_reset, self._model.nsensordata), dtype=np.float32)
        mj_data = self._worker_data[0]  # Use first worker for utility

        for i in range(num_reset):
            mj_data.time = 0.0
            mj_data.qpos[:] = qpos_batch[i]
            mj_data.qvel[:] = qvel_batch[i]
            mj_data.ctrl[:] = 0.0
            mj_data.qacc[:] = 0.0
            mj_data.qacc_warmstart[:] = 0.0

            mujoco.mj_forward(self._model, mj_data)

            sensor_batch[i] = mj_data.sensordata

        # Update Global Sensor State
        if hasattr(self, "_state") and self._state is not None:
            self._state.sensor_data[env_indices] = sensor_batch

        # Reconstruct physics state
        obs_physics_state = np.zeros((num_reset, self.physics_state_dim), dtype=np.float64)
        obs_physics_state[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch
        obs_physics_state[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch

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

        # Manually call update_cache to populate local_gravity/contacts/etc.
        self._update_cache(obs_state)
        # Reset feet_air_time to 0.0 as it was incremented by update_cache
        info["feet_air_time"][:] = 0.0

        # Call _get_obs ONCE for the entire batch
        obs_batch = self._get_obs(obs_state, info)

        # MjNpEnv expects: new_physics_states, new_obs, info
        return obs_physics_state, obs_batch, info

    def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
        # Reward for taking long steps: (air_time - threshold) * first_contact
        air_time = info.get("air_time_at_contact", np.zeros((self._num_envs, 4)))

        rew_air_time = np.sum((air_time - 0.1) * (air_time > 0.0), axis=1)

        # Reward is only non-zero when commands are non-zero
        rew_air_time *= np.linalg.norm(commands[:, :3], axis=1) > 0.01
        return rew_air_time

    def _reward_tracking_lin_vel(self, state, commands: np.ndarray):
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(state)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, state, commands: np.ndarray):
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(state)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_stand_still(self, state, commands: np.ndarray):
        # Penalize motion (joint deviation) at zero commands
        cmd_norm = np.linalg.norm(commands, axis=1)
        return np.sum(np.abs(self.get_dof_pos(state) - self.default_angles), axis=1) * (cmd_norm < 0.01)

    def _cost_feet_slip(self, state, info):
        # Penalize foot velocity while in contact
        vals = []
        for idx_list in self.foot_linvel_sensor_indices:
            vals.append(state.sensor_data[:, idx_list])  # (N, 3)

        feet_vel = np.stack(vals, axis=1)  # (N, 4, 3)
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = np.sum(np.square(vel_xy), axis=-1)

        contacts = info.get("contacts", np.zeros((self._num_envs, 4)))
        cmd_norm = np.linalg.norm(info["commands"], axis=1)

        return np.sum(vel_xy_norm_sq * contacts, axis=1) * (cmd_norm > 0.01)

    def _cost_feet_clearance(self, state):
        # Penalize deviation from target height during swing
        # Get foot velocity XY
        vals = []
        for idx_list in self.foot_linvel_sensor_indices:
            vals.append(state.sensor_data[:, idx_list])
        feet_vel = np.stack(vals, axis=1)
        vel_xy = feet_vel[..., :2]
        vel_norm = np.sqrt(np.linalg.norm(vel_xy, axis=-1))

        # Get Foot Z
        foot_z = state.info.get("foot_pos_z", np.zeros((self._num_envs, 4)))

        target = self.cfg.reward_config.max_foot_height
        delta = np.square(foot_z - target)
        return np.sum(delta * vel_norm, axis=1)

    def _cost_feet_height(self, info):
        # Penalize swing feet that don't reach target height
        peak = info.get("swing_peak_at_contact", np.zeros((self._num_envs, 4)))

        target = self.cfg.reward_config.max_foot_height
        if target < 0.0001:
            raise ValueError(f"Invalid target feet height: {target}")

        error = peak / target - 1.0
        mask = peak > 0.001

        cmd_norm = np.linalg.norm(info["commands"], axis=1)
        return np.sum(np.square(error) * mask, axis=1) * (cmd_norm > 0.01)

    def _reward_termination(self, done):
        return done
