import gymnasium as gym
import mujoco
import numpy as np

from unilab.envs import registry
from unilab.envs.locomotion.go1.cfg import Go1WalkNpEnvCfg
from unilab.envs.mujoco_env.mj_env import MjNpEnv, MjNpEnvState
from unilab.envs.utils.math_utils import quat_rotate_inverse, quat_mul, axis_angle_to_quat, quat_rotate

@registry.env("go1-flat-terrain-walk", sim_backend="mujoco")
class Go1WalkTaskMj(MjNpEnv):
    def __init__(self, cfg: Go1WalkNpEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)

        # Modify PD gains to match mujoco_playground go2/base.py
        self._model.dof_damping[6:] = cfg.control_config.Kd
        self._model.actuator_gainprm[:, 0] = cfg.control_config.Kp
        self._model.actuator_biasprm[:, 1] = -cfg.control_config.Kp

        self.nq = self._model.nq
        self.nv = self._model.nv
        # Offsets in physics_state (mjSTATE_FULLPHYSICS: time, qpos, qvel, act, qacc_warmstart)
        # assuming time (1), qpos(nq), qvel(nv) ...
        self._idx_qpos = 1
        self._idx_qvel = 1 + self.nq

        self._num_dof_pos = self.nq - 7 # Floating base 7
        self._num_dof_vel = self.nv - 6 # Floating base 6
        
        self._init_action_space()
        self._num_action = self._action_space.shape[0]
        self._init_obs_space()
        self._num_observation = self._observation_space.shape[0]
        
        self._init_dof_vel = np.zeros(
            (self._num_dof_vel,),
            dtype=np.float32,
        )
        # Compute init dof pos from keyframe 0 or qpos0
        # MjModel.qpos0 contains default position
        self._init_qpos = self._model.qpos0.copy()
        
        self._init_buffer()
        self._init_sensor_indices()
        self._init_reward_functions()

    def _init_reward_functions(self):
        """Register reward functions with standardized signature (state -> term)."""
        self._reward_fns = {
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "orientation": self._reward_orientation,
            "torques": self._reward_torques,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": lambda s: self._reward_dof_acc(s, s.info),
            "action_rate": lambda s: self._reward_action_rate(s.info),
            "tracking_lin_vel": lambda s: self._reward_tracking_lin_vel(s, s.info["commands"]),
            "tracking_ang_vel": lambda s: self._reward_tracking_ang_vel(s, s.info["commands"]),
            "stand_still": lambda s: self._reward_stand_still(s, s.info["commands"]),
            "feet_air_time": lambda s: self._reward_feet_air_time(s.info["commands"], s.info),
            "termination": lambda s: self._reward_termination(s.terminated),
            "dof_pos_limits": lambda s: self._cost_joint_pos_limits(s),
            "pose": lambda s: self._reward_pose(s),
            "energy": lambda s: self._cost_energy(s),
            "feet_clearance": lambda s: self._cost_feet_clearance(s),
            "feet_height": lambda s: self._cost_feet_height(s.info),
            "feet_slip": lambda s: self._cost_feet_slip(s, s.info),
            "base_height": lambda s: self._reward_base_height(s),
        }

    def _reward_base_height(self, state: MjNpEnvState):
        # Penalize base height deviation from target
        base_height = state.physics_state[:, self._idx_qpos + 2]
        target_height = self._cfg.init_state.pos[2]
        return np.square(base_height - target_height)

    def _init_sensor_indices(self):
        super()._init_sensor_indices()
        
        # 1. Contact Sensors
        # Strict match: expected names FR_floor_found etc. based on xml definition
        prefixes = ["FR", "FL", "RR", "RL"]
        contact_names = [f"{p}_floor_found" for p in prefixes]
        
        self.contact_sensor_indices = []
        for name in contact_names:
            self.contact_sensor_indices.extend(self._get_sensor_indices(name))
        
        print(f"Mapped contact sensors: {contact_names} -> {self.contact_sensor_indices}")

        # 2. Global State Sensors
        # Enforce exact naming from scene_mjx_feetonly_flat_terrain.xml
        self.idx_global_pos = self._get_sensor_indices("position")
        self.idx_orientation = self._get_sensor_indices("orientation")
        self.idx_global_linvel = self._get_sensor_indices("global_linvel")
        self.idx_global_angvel = self._get_sensor_indices("global_angvel")
        self.idx_upvector = self._get_sensor_indices("upvector")

        # 3. Local/Proprioceptive Sensors
        self.idx_linvel = self._get_sensor_indices("local_linvel")
        self.idx_gyro = self._get_sensor_indices("gyro")

        # 4. Foot Position Sensors
        self.foot_pos_sensor_indices = []
        for p in prefixes:
            self.foot_pos_sensor_indices.append(self._get_sensor_indices(f"{p}_pos"))
            
    def _get_sensor_indices(self, name):
        """Helper to get data indices from sensor name."""
        if name not in self.sensor_indices:
             # Allow optional sensors or raise? original raised.
             # I will raise to fail fast if important.
             raise ValueError(f"Sensor '{name}' not found.")
        sensor_id = self.sensor_indices[name]
        adr = self._model.sensor_adr[sensor_id]
        dim = self._model.sensor_dim[sensor_id]
        return list(range(adr, adr + dim))


    def _init_obs_space(self):
        # model = self.model
        num_dof_vel = self._num_dof_vel
        num_joint_angle = self._num_dof_pos
        num_linvel = 3
        num_gyro = 3
        num_gravity = 3
        num_actions = self._num_action
        num_command = 3

        num_obs = num_linvel + num_gyro + num_gravity + num_joint_angle + num_dof_vel + num_actions + num_command

        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

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

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_dof_pos(self, state: MjNpEnvState):
        # qpos[7:]
        # Extract qpos from physics_state
        return state.physics_state[:, self._idx_qpos + 7 : self._idx_qpos + self.nq]

    def get_dof_vel(self, state: MjNpEnvState):
        # qvel[6:]
        return state.physics_state[:, self._idx_qvel + 6 : self._idx_qvel + self.nv]

    def _init_buffer(self):
        cfg = self._cfg
        assert isinstance(cfg, Go1WalkNpEnvCfg)
        # init buffers

        self.reset_buf = np.ones(self._num_envs, dtype=bool)
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        # self.commands_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

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
        
        print("Default joint angles:", self.default_angles)

        # Foot stuff handle in _init_sensor_indices
        self._init_foot_linvel_sensor_indices()
        
        # Cache joint limits for reward
        if self._model.njnt > 1:
             self.dof_pos_limits = self._model.jnt_range[1:1+self._num_dof_pos].copy()
             self.soft_dof_pos_limits = self.dof_pos_limits * 0.95
        else:
             self.dof_pos_limits = np.zeros((self._num_dof_pos, 2))
             self.soft_dof_pos_limits = np.zeros((self._num_dof_pos, 2))

    def _init_foot_linvel_sensor_indices(self):
        # Match XML order: FR, FL, RR, RL
        foot_sites = ["FR", "FL", "RR", "RL"]
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

    def update_state(self, state, obs_required=True):
        # 1. Always update intermediate state info (sensors, math) needed for rewards/termination
        self._update_cache(state)
        
        # 2. Compute Observation if required (for agent)
        if obs_required:
            state = self.update_observation(state)
            
        # 3. Check termination and calculate reward
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

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

        # C. Update Air Time
        if "feet_air_time" not in info:
             info["feet_air_time"] = np.zeros((self._num_envs, 4), dtype=np.float32)

        info["feet_air_time"] += self.cfg.ctrl_dt
        
        # D. Update Swing Peak & Foot Tracking (requires global foot pos)
        self._update_foot_tracking(state, info, current_contacts)

        # Apply reset for next step
        info["feet_air_time"] *= ~current_contacts
        info["contacts"] = current_contacts

    def _update_foot_tracking(self, state, info, current_contacts):
        # Calculate foot global Z for swing height reward
        batch_size = current_contacts.shape[0]
        
        if "swing_peak" not in info:
            info["swing_peak"] = np.zeros((batch_size, 4), dtype=np.float32)
            
        foot_rel_pos_list = []
        for idx_list in self.foot_pos_sensor_indices:
             foot_rel_pos_list.append(state.sensor_data[:, idx_list])
        foot_rel_pos = np.stack(foot_rel_pos_list, axis=1) # (N, 4, 3)
        
        base_pos = state.sensor_data[:, self.idx_global_pos] # (N, 3)
        base_quat = state.sensor_data[:, self.idx_orientation] # (N, 4)
        
        # Calculate Forward Kinematics manually: Global = Base + R * Local
        base_quat_conj = base_quat.copy()
        base_quat_conj[:, 1:] *= -1 # Conjugate
        
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

    def update_observation(self, state: MjNpEnvState):
        obs = self._get_obs(state, state.info)
        return state.replace(obs=obs)

    def update_terminated(self, state: MjNpEnvState) -> MjNpEnvState:
        local_gravity = state.info["local_gravity"]
        up_z = -local_gravity[:, 2]
        
        # 1. Orientation termination
        is_fallen = up_z <= 0.5

        return state.replace(
            terminated=is_fallen,
        )

    def update_feet_air_time(self, info: dict):
        feet_air_time = info["feet_air_time"]
        feet_air_time += self.cfg.ctrl_dt
        feet_air_time *= ~info["contacts"]
        return feet_air_time

    def resample_commands(self, num_envs: int):
        commands = np.random.uniform(
            low=self.cfg.commands.vel_limit[0],
            high=self.cfg.commands.vel_limit[1],
            size=(num_envs, 3),
        )

        mask = np.random.random(num_envs) < 0.05
        commands[mask] = 0.0
        
        return commands

    def update_reward(self, state: MjNpEnvState) -> MjNpEnvState:
        # Optimized: Calculate reward accumulatively using registered functions
        total_reward = np.zeros(self._num_envs, dtype=np.float32)
        scales = self.cfg.reward_config.scales
        
        # Logging dictionary for rsl_rl
        log = {}
        
        for name, scale in scales.items():
            if scale == 0.0:
                continue
                
            if name in self._reward_fns:
                # Call standardized lambda/method
                term = self._reward_fns[name](state)
                weighted_reward = term * scale
                total_reward += weighted_reward
                
                # Log average weighted reward per step
                log[f"reward/{name}"] = np.mean(weighted_reward)
        
        # Log other info metrics
        if "feet_air_time" in state.info:
            log["metrics/feet_air_time"] = np.mean(state.info["feet_air_time"])
        if "contacts" in state.info:
            log["metrics/contact_rate"] = np.mean(state.info["contacts"].astype(float))

        # Store log in info
        state.info["log"] = log
        
        # Clip reward
        total_reward = np.clip(total_reward, 0.0, 10000.0)
        
        return state.replace(reward=total_reward)

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
        
        if hasattr(self, '_state') and self._state is not None:
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

        # Call _get_obs ONCE for the entire batch
        obs_batch = self._get_obs(obs_state, info)

        # MjNpEnv expects: new_physics_states, new_obs, info
        return obs_physics_state, obs_batch, info

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, state):
        # Penalize z axis base linear velocity
        # Matches joystick.py _cost_lin_vel_z using global_linvel
        global_linvel = self.get_global_linvel(state)
        return np.square(global_linvel[:, 2])

    def _reward_ang_vel_xy(self, state):
        # Penalize xy axes base angular velocity
        # Matches joystick.py _cost_ang_vel_xy using global_angvel
        global_angvel = self.get_global_angvel(state)
        return np.sum(np.square(global_angvel[:, :2]), axis=1)

    def _reward_orientation(self, state):
        # Penalize non flat base orientation
        # Matches joystick.py _cost_orientation using upvector (Global Z axis of body)
        upvector = self.get_upvector(state)
        return np.sum(np.square(upvector[:, :2]), axis=1)

    def _reward_torques(self, state):
        return np.sum(np.square(state.ctrl), axis=1)

    def _reward_dof_vel(self, state):
        # Penalize dof velocities
        return np.sum(np.square(self.get_dof_vel(state)), axis=1)

    def _reward_dof_acc(self, state, info):
        # Penalize dof accelerations
        return np.sum(
            np.square((info["last_dof_vel"] - self.get_dof_vel(state)) / self.cfg.ctrl_dt),
            axis=1,
        )

    def _reward_action_rate(self, info: dict):
        # Penalize changes in actions
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_termination(self, done):
        # Terminal reward / penalty
        return done

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
        return np.sum(np.abs(self.get_dof_pos(state) - self.default_angles), axis=1) * (
            cmd_norm < 0.01
        )

    def _cost_energy(self, state: MjNpEnvState):
        # Energy = sum(abs(qvel) * abs(torque))
        return np.sum(np.abs(self.get_dof_vel(state)) * np.abs(state.ctrl), axis=1)

    def _reward_pose(self, state: MjNpEnvState):
        # Penalize deviation from default pose
        qpos = self.get_dof_pos(state)
        weight = np.tile(np.array([1.0, 1.0, 0.1]), 4)
        error = np.sum(np.square(qpos - self.default_angles) * weight, axis=1)
        return np.exp(-error)

    def _cost_joint_pos_limits(self, state: MjNpEnvState):
        # Penalize joints if they cross soft limits.
        qpos = self.get_dof_pos(state)
        soft_lower = self.soft_dof_pos_limits[:, 0]
        soft_upper = self.soft_dof_pos_limits[:, 1]
        
        # Lower violation
        out_of_limits = -np.clip(qpos - soft_lower, None, 0.0)
        # Upper violation
        out_of_limits += np.clip(qpos - soft_upper, 0.0, None)
        
        return np.sum(out_of_limits, axis=1)

    def _cost_feet_slip(self, state, info):
        # Penalize foot velocity while in contact
        vals = []
        for idx_list in self.foot_linvel_sensor_indices:
             vals.append(state.sensor_data[:, idx_list]) #(N, 3)
        
        feet_vel = np.stack(vals, axis=1) # (N, 4, 3)
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
        mask = (peak > 0.001) 
        
        cmd_norm = np.linalg.norm(info["commands"], axis=1)
        return np.sum(np.square(error) * mask, axis=1) * (cmd_norm > 0.01)
