import gymnasium as gym
import mujoco
import numpy as np
from dataclasses import dataclass, field

from unilab import ROOT_PATH
from unilab.envs.base import EnvCfg
from unilab.envs.mujoco_env.mj_env import MjNpEnv, MjNpEnvState
from unilab.envs.utils.math_utils import quat_rotate_inverse, quat_mul, axis_angle_to_quat

# ----------------- Configuration -----------------

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale: float = 0.5
    Kp: float = 35.0
    Kd: float = 0.5


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
    sim_dt: float = 0.004
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
            dtype=np.float32,
        )
        # Compute init dof pos from keyframe 0 or qpos0
        self._init_qpos = self._model.qpos0.copy()
        
        self._init_buffer()
        self._init_sensor_indices()

    def _init_action_space(self):
        model = self.model
        # nu = number of actuators
        self._action_space = gym.spaces.Box(
            np.array(model.actuator_ctrlrange[:, 0]),
            np.array(model.actuator_ctrlrange[:, 1]),
            (model.nu,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def get_dof_pos(self, state: MjNpEnvState):
        # qpos[7:]
        # Extract qpos from physics_state
        return state.physics_state[:, self._idx_qpos + 7 : self._idx_qpos + self.nq]

    def get_dof_vel(self, state: MjNpEnvState):
        # qvel[6:]
        return state.physics_state[:, self._idx_qvel + 6 : self._idx_qvel + self.nv]

    def _init_buffer(self):
        # Generic buffers
        self.reset_buf = np.ones(self._num_envs, dtype=bool)
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.hip_indices = []
        self.calf_indices = []
        
        # Try to find "home" keyframe to init default pose
        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            print(f"Using keyframe 'home' (id {key_id}) for initial state.")
            self._init_qpos = self._model.key_qpos[key_id].copy()
            self.default_angles = self._init_qpos[7:].astype(np.float32)
        else:
            raise ValueError("Keyframe 'home' not found in model.")

        # Populate indices
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if not name: continue
                    
            if "hip" in name:
                self.hip_indices.append(i)
            if "calf" in name:
                self.calf_indices.append(i)
        
        # Foot stuff handle in _init_sensor_indices
        self._init_foot_linvel_sensor_indices()
        
        if self._model.njnt > 1:
             self.dof_pos_limits = self._model.jnt_range[1:1+self._num_dof_pos].copy()
             self.soft_dof_pos_limits = self.dof_pos_limits * 0.95
        else:
             self.dof_pos_limits = np.zeros((self._num_dof_pos, 2))
             self.soft_dof_pos_limits = np.zeros((self._num_dof_pos, 2))

    def _init_sensor_indices(self):
        super()._init_sensor_indices()
        
        foot_name = self._cfg.asset.foot_name
        prefixes = ["FL", "FR", "RL", "RR"]
        expected_names = [f"{p}_{foot_name}_contact" for p in prefixes]
        
        self.contact_sensor_indices = []
        for name in expected_names:
            if name not in self.sensor_indices:
                raise ValueError(f"Required contact sensor '{name}' not found.")
            # Map sensor ID to data indices
            self.contact_sensor_indices.extend(self._get_sensor_indices(name))
            
        # Resolve 'local_linvel' and 'gyro'
        self.idx_linvel = self._get_sensor_indices(self._cfg.sensor.local_linvel)
        self.idx_gyro = self._get_sensor_indices(self._cfg.sensor.gyro)
        
        # Resolve required sensors for observation/tracking (Global sensors)
        self.idx_global_linvel = self._get_sensor_indices("global_linvel")
        self.idx_global_angvel = self._get_sensor_indices("global_angvel")
        self.idx_upvector = self._get_sensor_indices("upvector")
        
        if "global_position" in self.sensor_indices:
             self.idx_global_pos = self._get_sensor_indices("global_position")
        
        self.idx_orientation = self._get_sensor_indices("orientation")

        # Foot position sensors
        foot_names = ["FL", "FR", "RL", "RR"]
        self.foot_pos_sensor_indices = [] 
        for name in foot_names:
            sname = f"{name}_pos"
            self.foot_pos_sensor_indices.append(self._get_sensor_indices(sname))

    def _get_sensor_indices(self, name):
        if name not in self.sensor_indices:
             raise ValueError(f"Sensor '{name}' not found.")
        sensor_id = self.sensor_indices[name]
        adr = self._model.sensor_adr[sensor_id]
        dim = self._model.sensor_dim[sensor_id]
        return list(range(adr, adr + dim))
    
    def _init_foot_linvel_sensor_indices(self):
        foot_sites = ["FL", "FR", "RL", "RR"]
        self.foot_linvel_sensor_indices = []
        for site in foot_sites:
            name = f"{site}_global_linvel"
            self.foot_linvel_sensor_indices.append(self._get_sensor_indices(name))

    def apply_action(self, actions, state):
        # Update info for rewards
        state.info["last_dof_vel"] = self.get_dof_vel(state)
        state.info["last_last_actions"] = state.info["last_actions"] # Keep history of last last
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        
        # Compute control
        ctrl = self._compute_target_jq(actions)
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

    def get_global_angvel(self, state: MjNpEnvState) -> np.ndarray:
        return state.sensor_data[:, self.idx_global_angvel]

    def get_upvector(self, state: MjNpEnvState) -> np.ndarray:
        return state.sensor_data[:, self.idx_upvector]

    def _update_cache(self, state: MjNpEnvState):
        """Update cached info based on current physics/sensor state."""
        info = state.info
        
        # A. Update Local Gravity
        base_quat = state.physics_state[:, self._idx_qpos+3 : self._idx_qpos+7]
        local_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        info["local_gravity"] = local_gravity
        
        # B. Update Contacts
        if len(self.contact_sensor_indices) > 0:
            contact_vals = state.sensor_data[:, self.contact_sensor_indices]
            current_contacts = (contact_vals > 0.1)
        else:
            current_contacts = np.zeros((self._num_envs, 4), dtype=bool)
        
        info["contacts"] = current_contacts

    def _reward_lin_vel_z(self, state):
        global_linvel = self.get_global_linvel(state)
        return np.square(global_linvel[:, 2])

    def _reward_ang_vel_xy(self, state):
        global_angvel = self.get_global_angvel(state)
        return np.sum(np.square(global_angvel[:, :2]), axis=1)

    def _reward_orientation(self, state):
        upvector = self.get_upvector(state)
        return np.sum(np.square(upvector[:, :2]), axis=1)

    def _reward_torques(self, state):
        return np.sum(np.square(state.ctrl), axis=1)

    def _reward_dof_vel(self, state):
        return np.sum(np.square(self.get_dof_vel(state)), axis=1)

    def _reward_dof_acc(self, state, info):
        return np.sum(
            np.square((info["last_dof_vel"] - self.get_dof_vel(state)) / self.cfg.ctrl_dt),
            axis=1,
        )

    def _reward_action_rate(self, info: dict):
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _cost_energy(self, state: MjNpEnvState):
        return np.sum(np.abs(self.get_dof_vel(state)) * np.abs(state.ctrl), axis=1)

    def _reward_pose(self, state: MjNpEnvState):
        qpos = self.get_dof_pos(state)
        weight = np.tile(np.array([1.0, 1.0, 0.1]), 4)
        error = np.sum(np.square(qpos - self.default_angles) * weight, axis=1)
        return np.exp(-error)

    def _cost_joint_pos_limits(self, state: MjNpEnvState):
        qpos = self.get_dof_pos(state)
        soft_lower = self.soft_dof_pos_limits[:, 0]
        soft_upper = self.soft_dof_pos_limits[:, 1]
        out_of_limits = -np.clip(qpos - soft_lower, None, 0.0)
        out_of_limits += np.clip(qpos - soft_upper, 0.0, None)
        return np.sum(out_of_limits, axis=1)
