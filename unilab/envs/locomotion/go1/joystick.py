from __future__ import annotations

from etils import epath
import gymnasium as gym
import mujoco
import math
import numpy as np
from dataclasses import dataclass, field

from unilab.envs import registry
from unilab.envs.mujoco_env.mj_env import MjNpEnvState
from unilab.utils.math_utils import np_quat_mul, np_yaw_to_quat

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
            "ang_vel_xy": -0.1, #-0.02 #ppo
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
        self._enable_reward_log = True

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
            low=-float("inf"), high=float("inf"), shape=(num_obs,), dtype=float
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
        linear_vel = self.get_local_linvel(state)
        gyro = self.get_gyro(state)
        local_gravity = -self.get_upvector(state)
        dof_pos = self.get_dof_pos(state)
        dof_vel = self.get_dof_vel(state)

        # Apply Observation Noise if enabled
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

        obs = np.concatenate(
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
        state.obs = obs
        return state

    def _compute_rewards(self, state: MjNpEnvState) -> MjNpEnvState:
        total_reward = np.zeros((self._num_envs,), dtype=self._np_dtype)
        step_count = state.info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
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
                log[f"reward/{name}"] = float(np.mean(weighted_rew))
            
        state.info["log"] = log
        state.info["reward_components"] = {}

        # Keep the same dt scaling style used in Go2.
        total_reward *= self.cfg.ctrl_dt
    
        state.reward = total_reward
        return state

    def update_terminated(self, state: MjNpEnvState) -> MjNpEnvState:
        is_fallen = (self.get_upvector(state)[:, 2] <= 0.5)
        state.terminated = is_fallen
        return state

    def resample_commands(self, num_envs: int):
        low = np.array(self.cfg.commands.vel_limit[0], dtype=self._np_dtype)
        high = np.array(self.cfg.commands.vel_limit[1], dtype=self._np_dtype)
        commands = low + (high - low) * np.random.uniform(size=(num_envs, 3)).astype(self._np_dtype)

        # Standard practice: set small percentage of commands to zero to train standing still
        # mask = np.random.uniform(size=(num_envs,)) < 0.05
        # commands[mask] = 0.0
        
        return commands

    def reset(self, env_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        num_reset = len(env_indices)

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
            "current_actions": np.zeros((num_reset, self._num_action), dtype=self._np_dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=self._np_dtype),
            "commands": commands,
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

        # Call _get_obs ONCE for the entire batch
        obs_batch = self._get_obs(obs_state, info)

        # Environment reset returns: new_physics_states, new_obs, info
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
