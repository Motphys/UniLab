from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from dataclasses import dataclass, field

from unilab.envs.base import EnvCfg
from unilab.envs.mujoco_env.mj_env import MjNpEnv, MjNpEnvState

# ----------------- Configuration -----------------

@dataclass
class NoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale: float = 0.25
    Kp: float = 35.0
    Kd: float = 0.5
    simulate_action_latency: bool = True


@dataclass
class Asset:
    body_name = "base"
    foot_name = "foot"
    ground = "floor"

@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"

@dataclass
class Go2BaseCfg(EnvCfg):
    model_file: str = field(default=str(""))
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02

# ----------------- Environment -----------------

class Go2BaseMjEnv(MjNpEnv):
    def __init__(self, cfg: Go2BaseCfg, num_envs=1):
        super().__init__(cfg, num_envs)

        # Modify PD gains
        self._model.dof_damping[6:] = cfg.control_config.Kd
        self._model.actuator_gainprm[:, 0] = cfg.control_config.Kp
        self._model.actuator_biasprm[:, 1] = -cfg.control_config.Kp

        self.nq = self._model.nq
        self.nv = self._model.nv
        self._idx_qpos = 1
        self._idx_qvel = 1 + self.nq

        self._num_dof_pos = self.nq - 7 
        self._num_dof_vel = self.nv - 6
        
        self._init_action_space()
        self._num_action = self._action_space.shape[0]

        # Init init_dof_vel which is used in reset
        self._init_dof_vel = np.zeros(
            (self._num_dof_vel,),
            dtype=self._np_dtype,
        )
        self._init_qpos = np.array(self._model.qpos0.copy(), dtype=self._np_dtype)
        
        self._init_buffer()
        self._init_sensor_indices()

    def _init_action_space(self):
        model = self.model
        # nu = number of actuators
        low = model.actuator_ctrlrange[:, 0].copy()
        high = model.actuator_ctrlrange[:, 1].copy()
        self._action_space = gym.spaces.Box(
            low,
            high,
            (model.nu,),
            dtype=float,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def get_dof_pos(self, state: MjNpEnvState):
        return state.physics_state[:, self._idx_qpos + 7 : self._idx_qpos + self.nq]

    def get_dof_vel(self, state: MjNpEnvState):
        return state.physics_state[:, self._idx_qvel + 6 : self._idx_qvel + self.nv]

    def _init_buffer(self):
        # Generic buffers
        self.reset_buf = np.ones((self._num_envs,), dtype=bool)
        self.default_angles = np.zeros((self._num_action,), dtype=self._np_dtype)
        
        # Try to find "home" keyframe to init default pose
        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            self._init_qpos = np.array(self._model.key_qpos[key_id].copy(), dtype=self._np_dtype)
            self.default_angles = self._init_qpos[7:]
        else:
            raise ValueError("Keyframe 'home' not found in model.")
        
    def _init_sensor_indices(self):
        super()._init_sensor_indices()

        # Resolve 'local_linvel' and 'gyro'
        self.idx_linvel = self._get_sensor_indices(self._cfg.sensor.local_linvel)
        self.idx_gyro = self._get_sensor_indices(self._cfg.sensor.gyro)
        
        # Resolve required sensors for observation/tracking.
        self.idx_global_linvel = self._get_sensor_indices("global_linvel")
        self.idx_upvector = self._get_sensor_indices("upvector")
        
        self.foot_names = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]
        self.idx_foot_pos = [self._get_sensor_indices(name) for name in self.foot_names]

        self.foot_vel_names = ["FL_vel", "FR_vel", "RL_vel", "RR_vel"]
        self.idx_foot_vel = [self._get_sensor_indices(name) for name in self.foot_vel_names]

        self.foot_contact_names = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
        self.idx_foot_contact = [self._get_sensor_indices(name) for name in self.foot_contact_names]

    def _get_sensor_indices(self, name):
        if name not in self.sensor_indices:
             raise ValueError(f"Sensor '{name}' not found.")
        sensor_id = self.sensor_indices[name]
        adr = self._model.sensor_adr[sensor_id]
        dim = self._model.sensor_dim[sensor_id]
        return list(range(adr, adr + dim))
    
    def apply_action(self, actions, state):
        state.info["last_actions"] = np.array(state.info["current_actions"])
        state.info["current_actions"] = actions
        
        # Match genesis setup: one-step action latency on the actuator command.
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else state.info["current_actions"]
        )
        ctrl = self._compute_target_jq(exec_actions)
        return ctrl

    def _compute_target_jq(self, actions):
        # Compute target position from actions.
        target_jq = actions * self.cfg.control_config.action_scale + self.default_angles
        return target_jq

    def get_local_linvel(self, state: MjNpEnvState) -> np.ndarray:
        return state.sensor_data[:, self.idx_linvel]

    def get_gyro(self, state: MjNpEnvState) -> np.ndarray:
        return state.sensor_data[:, self.idx_gyro]

    def get_global_linvel(self, state: MjNpEnvState) -> np.ndarray:
        return state.sensor_data[:, self.idx_global_linvel]

    def get_upvector(self, state: MjNpEnvState) -> np.ndarray:
        return state.sensor_data[:, self.idx_upvector]

    def get_foot_pos(self, state: MjNpEnvState) -> np.ndarray:
        # returns shape (num_envs, 4, 3)
        foot_pos = [state.sensor_data[:, idx] for idx in self.idx_foot_pos]
        return np.stack(foot_pos, axis=1)

    def get_foot_vel(self, state: MjNpEnvState) -> np.ndarray:
        # returns shape (num_envs, 4, 3)
        foot_vel = [state.sensor_data[:, idx] for idx in self.idx_foot_vel]
        return np.stack(foot_vel, axis=1)

    def get_foot_contact(self, state: MjNpEnvState) -> np.ndarray:
        # returns shape (num_envs, 4)
        foot_contact = [state.sensor_data[:, idx[0]] for idx in self.idx_foot_contact]
        return np.stack(foot_contact, axis=1)
        
    def _reward_lin_vel_z(self, state):
        global_linvel = self.get_global_linvel(state)
        return np.square(global_linvel[:, 2])

    def _reward_action_rate(self, info: dict):
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

