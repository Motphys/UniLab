from __future__ import annotations

import gymnasium as gym
import mujoco
try:
    import mlx.core as mx
except Exception:
    mx = None
from dataclasses import dataclass, field

from unilab.envs.base import EnvCfg
from unilab.envs.mujoco_env.mjx_env import MjMlxEnv, MjMlxEnvState


@dataclass
class NoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.02
    scale_joint_vel: float = 0.3
    scale_gyro: float = 0.1
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    # target angle = action_scale * action + default_angle
    action_scale: float = 0.25
    Kp: float = 80.0
    Kd: float = 2.0
    simulate_action_latency: bool = False


@dataclass
class Asset:
    body_name = "pelvis"
    foot_name = "ankle_roll_link"
    ground = "floor"


@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "imu-pelvis-angular-velocity"


@dataclass
class G1BaseCfg(EnvCfg):
    model_file: str = field(default=str(""))
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02


class G1BaseMjEnv(MjMlxEnv):
    def __init__(self, cfg: G1BaseCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)

        # Override default PD gains for faster early training convergence.
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

        self._init_dof_vel = mx.zeros((self._num_dof_vel,), dtype=self._mlx_dtype)
        self._init_qpos = mx.array(self._model.qpos0.copy(), dtype=self._mlx_dtype)

        self._init_buffer()
        self._init_sensor_indices()

    def _init_action_space(self):
        low = self._model.actuator_ctrlrange[:, 0].copy()
        high = self._model.actuator_ctrlrange[:, 1].copy()
        self._action_space = gym.spaces.Box(low, high, (self._model.nu,), dtype=float)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def _init_buffer(self):
        self.reset_buf = mx.ones((self._num_envs,), dtype=mx.bool_)
        self.default_angles = mx.zeros((self._num_action,), dtype=self._mlx_dtype)

        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id < 0:
            raise ValueError("Keyframe 'stand' not found in model.")

        self._init_qpos = mx.array(self._model.key_qpos[key_id].copy(), dtype=self._mlx_dtype)
        self.default_angles = self._init_qpos[7 : 7 + self._num_action]

    def _get_sensor_indices(self, name: str):
        if name not in self.sensor_indices:
            return None
        sensor_id = self.sensor_indices[name]
        adr = self._model.sensor_adr[sensor_id]
        dim = self._model.sensor_dim[sensor_id]
        return list(range(adr, adr + dim))

    def _init_sensor_indices(self):
        super()._init_sensor_indices()
        self.idx_linvel = self._get_sensor_indices(self._cfg.sensor.local_linvel)
        self.idx_gyro = self._get_sensor_indices(self._cfg.sensor.gyro)
        self.idx_upvector = self._get_sensor_indices("upvector")
        self._idx_torso_upvector = self._get_sensor_indices("torso_upvector")
        if self.idx_linvel is None:
            raise ValueError("Sensor 'local_linvel' is required for G1.")
        if self.idx_gyro is None:
            raise ValueError(f"Sensor '{self._cfg.sensor.gyro}' is required for G1.")
        if self.idx_upvector is None:
            raise ValueError("Sensor 'upvector' is required for G1.")
        if self._idx_torso_upvector is None:
            raise ValueError("Sensor 'torso_upvector' is required for G1.")

    def get_dof_pos(self, state: MjMlxEnvState):
        return state.physics_state[:, self._idx_qpos + 7 : self._idx_qpos + 7 + self._num_action]

    def get_dof_vel(self, state: MjMlxEnvState):
        return state.physics_state[:, self._idx_qvel + 6 : self._idx_qvel + 6 + self._num_action]

    def get_global_linvel(self, state: MjMlxEnvState) -> mx.array:
        return state.physics_state[:, self._idx_qvel : self._idx_qvel + 3]

    def get_local_linvel(self, state: MjMlxEnvState) -> mx.array:
        return state.sensor_data[:, self.idx_linvel]

    def get_gyro(self, state: MjMlxEnvState) -> mx.array:
        return state.sensor_data[:, self.idx_gyro]

    def get_upvector(self, state: MjMlxEnvState) -> mx.array:
        return state.sensor_data[:, self.idx_upvector]

    def apply_action(self, actions: mx.array, state: MjMlxEnvState) -> mx.array:
        state.info["last_actions"] = mx.array(state.info["current_actions"])
        state.info["current_actions"] = actions

        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else state.info["current_actions"]
        )
        return exec_actions * self._cfg.control_config.action_scale + self.default_angles

    def _reward_lin_vel_z(self, state: MjMlxEnvState):
        global_linvel = self.get_global_linvel(state)
        return mx.square(global_linvel[:, 2])

    def _reward_action_rate(self, info: dict):
        action_diff = info["current_actions"] - info["last_actions"]
        return mx.sum(mx.square(action_diff), axis=1)
