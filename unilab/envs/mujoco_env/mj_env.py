
import abc
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from multiprocessing import cpu_count

import mujoco
from mujoco import rollout
import numpy as np

from unilab.envs.base import ABEnv, EnvCfg

@dataclass
class MjNpEnvState:
    physics_state: np.ndarray  # (num_envs, nstate) - MjState (full physics)
    sensor_data: np.ndarray    # (num_envs, nsensordata) - MjData.sensordata
    ctrl: np.ndarray           # (num_envs, ncontrol) - Current control input
    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict

    @property
    def done(self) -> np.ndarray:
        """
        Check if the environment is done.
        """
        return np.logical_or(self.terminated, self.truncated)

    def replace(self, **updates) -> "MjNpEnvState":
        return dataclasses.replace(self, **updates)

    def validate(self):
        num_envs = self.physics_state.shape[0]
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape
        assert self.ctrl.shape[0] == num_envs, self.ctrl.shape


class MjNpEnv(ABEnv):
    _model: mujoco.MjModel
    _cfg: EnvCfg
    _state: MjNpEnvState = None
    _num_envs: int
    _rollout_runner: rollout.Rollout = None
    _worker_data: List[mujoco.MjData] = None # Pool of workers for rollout
    _last_sensor_traj: np.ndarray = None

    def __init__(self, cfg: EnvCfg, num_envs: int = 1):
        self._cfg = cfg
        self._num_envs = num_envs
        self._model = mujoco.MjModel.from_xml_path(cfg.model_file)
        self._model.opt.timestep = cfg.sim_dt
        
        # MjData is not thread-safe for write access, so we need one per thread for parallel stepping.
        # We separate the "Logic" state (MujocoEnvState) from the "Compute" resources (Worker Data).
        
        # Validate that model timestep matches config
        # self._model.opt.timestep = cfg.sim_dt # Already set
        
        # Configure Thread Pool for Rollout
        # We use min(num_envs, cpu_count) threads.
        self._n_threads = min(num_envs, cpu_count())
        
        # Create worker MjData pool
        # These are purely for computation and do not hold persistent environment state.
        self._worker_data = [mujoco.MjData(self._model) for _ in range(self._n_threads)]
        
        # Using persistent rollout runner
        self._rollout_runner = rollout.Rollout(nthread=self._n_threads)

        self._init_sensor_indices()

    def _init_sensor_indices(self):
        """
        Build a dictionary mapping sensor names to their indices.
        """
        self.sensor_indices = {}
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self.sensor_indices[name] = i

    @property
    def physics_state_dim(self) -> int:
        return mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    def close(self):
        if self._rollout_runner is not None:
             self._rollout_runner = None

    @property
    def model(self) -> mujoco.MjModel:
        """
        Get the mujoco model
        """
        return self._model

    @property
    def state(self) -> MjNpEnvState:
        """
        Get the current environment state
        """
        return self._state

    @property
    def cfg(self) -> EnvCfg:
        """
        Get the environment configuration
        """
        return self._cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def init_state(self) -> MjNpEnvState:
        """
        Create a new environment state
        """
        nstate = mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        nsensordata = self._model.nsensordata
        ncontrol = self._model.nu
        
        physics_state = np.zeros((self._num_envs, nstate), dtype=np.float64)
        sensor_data = np.zeros((self._num_envs, nsensordata), dtype=np.float64)
        ctrl = np.zeros((self._num_envs, ncontrol), dtype=np.float64)
        
        obs = np.zeros((self._num_envs, self.observation_space.shape[0]), dtype=np.float32)
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {"steps": np.zeros((self._num_envs,), dtype=np.uint64)}
        
        self._state = MjNpEnvState(physics_state, sensor_data, ctrl, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def _reset_done_envs(self):
        """
        Reset the environments that are done. 
        """
        state = self._state
        done = state.done
        assert done.shape == (self._num_envs,)
        if not np.any(done):
            return

        np.putmask(state.info["steps"], done, 0)
        
        # Indices of envs to reset
        indices = np.where(done)[0]
        
        # Call reset. 
        # Note: reset now is responsible for returning new physics states for these indices
        new_physics_states, new_obs, info1 = self.reset(indices)
        
        # Update state
        state.physics_state[indices] = new_physics_states
        if new_obs is not None:
             state.obs[indices] = new_obs
        
        # NOTE: sensor_data is NOT automatically updated by setting physics_state 
        # until the next physics_step via rollout!
        # If reset() returned None for obs, it implies we expect env to compute it from sensor_data?
        # BUT sensor_data is stale (pre-reset).
        # We must either:
        # A) Compute Obs manually in reset using analytical kinematics (hard)
        # B) Run a 0-step forward kinematics rollout to update sensor_data (cleaner)
        # C) Or just rely on next step.
        # But RSL-RL wrapper calls reset_all() -> reset() -> _update_buffers(obs).
        
        # If new_obs is None, we need to compute it.
        if new_obs is None:
            # This means we relied on sensor_data in _compute_obs, but sensor_data is stale!
            # We should probably run a minimal forward step or re-compute.
            pass

        # Update info
        if info1:

            def replace_dict_values(dst, new_values, mask):
                for key, value in new_values.items():
                    if key not in dst:
                        dst[key] = value
                    else:
                        if isinstance(value, np.ndarray):
                            dst[key][mask] = value
                        elif isinstance(value, dict):
                            assert isinstance(dst[key], dict)
                            replace_dict_values(dst[key], value, mask)

            replace_dict_values(state.info, info1, done)
        
        # Since we reset state, we assume sensor data might be stale until next step, 
        # or reset should computed it. Ideally physics_step will update sensor_data.

    def _update_truncate(self):
        """
        Truncate the environments that have reached max episode length
        """
        if not self._cfg.max_episode_steps:
            return
        self._state.truncated = self._state.info["steps"] >= self._cfg.max_episode_steps

    @abc.abstractmethod
    def apply_action(self, actions: np.ndarray, state: MjNpEnvState) -> np.ndarray:
        """
        Compute control input from actions.
        
        Returns:
            np.ndarray: The control input (ctrl) for the physics step. Shape (num_envs, ncontrol)
        """

    @abc.abstractmethod
    def update_state(self, state: MjNpEnvState, obs_required: bool = True) -> MjNpEnvState:
        """
        Update the environment state after physics step (e.g. compute obs, rewards)
        """

    @abc.abstractmethod
    def reset(
        self,
        env_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Reset the environment for the done envs

        Args:
            env_indices (np.ndarray): The indices of the envs being reset

        Returns:
            tuple:
                - new_physics_states (np.ndarray): (len(indices), nstate)
                - new_obs (np.ndarray): (len(indices), obs_dim)
                - info (dict): Additional info
        """
        pass

    def physics_step(self):
        """
        Step the physics simulation for all environments in parallel using mujoco.rollout.
        """
        nsubsteps = self._cfg.sim_substeps
        
        # Prepare inputs
        initial_state = self._state.physics_state
        ctrl = self._state.ctrl
        
        # Rollout expects control shape (nbatch, nstep, ncontrol)
        # We assume zero-order hold (constant action) across substeps
        # Tile ctrl: (B, D) -> (B, nsubsteps, D)
        control_traj = np.tile(ctrl[:, None, :], (1, nsubsteps, 1))
        
        # Execute rollout
        # Note: We pass self._worker_data which has length == self._n_threads
        # rollout will handle checking len(data) == nthread and distributing nbatch tasks
        state_traj, sensor_traj = self._rollout_runner.rollout(
            self._model, 
            self._worker_data, 
            initial_state=initial_state, 
            control=control_traj,
            nstep=nsubsteps
        )

        # Store sensor data for potential rendering usage
        # We only really care about the LAST step sensor data for rendering the current frame
        if sensor_traj is not None and sensor_traj.size > 0:
            self._last_sensor_traj = sensor_traj[:, -1, :]
            # Update state sensor data
            self._state.sensor_data[:] = self._last_sensor_traj
        
        # Update physics state (for next step)
        # Get the final state from the trajectory
        if state_traj is not None and state_traj.size > 0:
            self._state.physics_state[:] = state_traj[:, -1, :]

    def _prev_physics_step(self):
        state = self._state
        state.reward.fill(0.0)
        state.terminated.fill(False)
        state.truncated.fill(False)

    def _before_chunk_step(self, data: Any):
        """
        Hook called before executing a chunk of actions.
        """
        pass

    def step(self, actions: np.ndarray) -> MjNpEnvState:
        if self._state is None:
            self.init_state()

        # Handle action dimensions
        # 1. auto crop if input action dim > action_space dim
        if actions.shape[-1] > self.action_space.shape[0]:
            actions = actions[..., :self.action_space.shape[0]]

        # 2. handle chunk action (B, T, D) vs single action (B, D)
        if actions.ndim == 2:
            # (B, D) -> (B, 1, D)
            actions = actions[:, None, :]
        
        # Now actions is (B, T, D)
        num_steps = actions.shape[1]
        
        # Hook for chunk start
        self._before_chunk_step(None) # No longer passing list of MjData

        cumulative_reward = np.zeros(self._num_envs, dtype=np.float32)
        chunk_terminated = np.zeros(self._num_envs, dtype=bool)
        chunk_truncated = np.zeros(self._num_envs, dtype=bool)

        for t in range(num_steps):
            self._prev_physics_step()
            
            # Apply Action: Now updates self._state.ctrl
            self._state.ctrl[:] = self.apply_action(actions[:, t], self._state)
            
            self.physics_step()
            
            # Optimization: only compute obs on last step
            is_last_step = (t == num_steps - 1)
            self._state = self.update_state(self._state, obs_required=is_last_step)
                
            self._state.info["steps"] += 1
            
            # Accumulate reward before reset might clear it
            cumulative_reward += self._state.reward

            self._update_truncate()
            
            # Accumulate done flags
            chunk_terminated |= self._state.terminated
            chunk_truncated |= self._state.truncated
        
        # Apply accumulated flags to state
        self._state.terminated = chunk_terminated
        self._state.truncated = chunk_truncated
        self._state.reward = cumulative_reward
        
        # Reset done envs at the very end of the chunk
        self._reset_done_envs()
        
        return self._state

    def _get_sensor_range(self, name, dim):
        id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if id == -1:
            raise ValueError(f"Sensor {name} not found in model")
        adr = self._model.sensor_adr[id]
        return np.arange(adr, adr + dim)
