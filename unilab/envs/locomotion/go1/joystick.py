from etils import epath
import gymnasium as gym
import mujoco
import numpy as np
from dataclasses import dataclass, field

from unilab.envs import registry
from unilab.envs.mujoco_env.mj_env import MjNpEnvState
from unilab.envs.utils.math_utils import quat_mul, axis_angle_to_quat

from unilab.envs.locomotion.go1.base import Go1BaseMjEnv, Go1BaseCfg

# ----------------- Configuration -----------------

@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.45]

@dataclass
class Commands:
    vel_limit = [
        # [-1.0, -0.5, -1.0],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
        # [ 1.0,  0.5,  1.0],  # max
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # Keep Go1 reward template aligned with Go2 simplified setup.
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.02,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        }
    )

    tracking_sigma: float = 0.25
    base_height_target: float = 0.3


@registry.envcfg("Go1JoystickFlatTerrain")
@dataclass
class Go1JoystickCfg(Go1BaseCfg):
    model_file: str = str(epath.Path(__file__).parent / "xml" / "scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

# ----------------- Environment -----------------

@registry.env("Go1JoystickFlatTerrain", sim_backend="mujoco")
class Go1WalkTaskMj(Go1BaseMjEnv):
    def __init__(self, cfg: Go1JoystickCfg, num_envs=1):
        super().__init__(cfg, num_envs)

        self._init_reward_functions()
        self._init_obs_space()

    def _init_reward_functions(self):
        """Register reward functions."""
        # Keep reward terms aligned with Go2 simplified setup.
        self._reward_fns = {
            "tracking_lin_vel": lambda s: self._reward_tracking_lin_vel(s, s.info["commands"]),
            "tracking_ang_vel": lambda s: self._reward_tracking_ang_vel(s, s.info["commands"]),
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
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
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(num_obs,), dtype=np.float32
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def _reward_base_height(self, state: MjNpEnvState):
        # Penalize base height deviation from target.
        base_height = state.physics_state[:, self._idx_qpos + 2]
        target_height = self._cfg.reward_config.base_height_target
        return np.square(base_height - target_height)

    def _reward_similar_to_default(self, state: MjNpEnvState):
        # Penalize joint pose deviations from default posture.
        return np.sum(np.abs(self.get_dof_pos(state) - self.default_angles), axis=1)

    def _get_obs(self, state: MjNpEnvState, info: dict) -> np.ndarray:
        # Get raw data (copy to allow noise injection without side effects)
        linear_vel = self.get_local_linvel(state).copy()
        gyro = self.get_gyro(state).copy()
        local_gravity = (-self.get_upvector(state)).copy()
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
        # 1. Check Termination
        state = self.update_terminated(state)

        # 2. Compute Rewards
        state = self._compute_rewards(state)

        # 3. Update Observation (if required)
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
            
        state.info["log"] = log
        state.info["reward_components"] = {}

        # Keep the same dt scaling style used in Go2.
        total_reward *= self.cfg.ctrl_dt
    
        return state.replace(reward=total_reward)

    def update_terminated(self, state: MjNpEnvState) -> MjNpEnvState:
        is_fallen = (self.get_upvector(state)[:, 2] <= 0.5)
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
        # mask = np.random.random(num_envs) < 0.05
        # commands[mask] = 0.0
        
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
        }

        sensor_batch = self._compute_sensor_batch_from_qpos_qvel(qpos_batch, qvel_batch)

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

        # Call _get_obs ONCE for the entire batch
        obs_batch = self._get_obs(obs_state, info)

        # MjNpEnv expects: new_physics_states, new_obs, info
        return obs_physics_state, obs_batch, info

    def _reward_tracking_lin_vel(self, state, commands: np.ndarray):
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(state)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, state, commands: np.ndarray):
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(state)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_ang_vel_xy(self, state: MjNpEnvState):
        gyro = self.get_gyro(state)
        return np.sum(np.square(gyro[:, :2]), axis=1)
