
import abc
import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from multiprocessing import cpu_count

import mujoco
from mujoco import mlx_step
import mlx.core as mx
import numpy as np

from unilab.envs.base import ABEnv, EnvCfg

@dataclass
class MjMlxEnvState:
    physics_state: mx.array  # (num_envs, nstate) - MjState (full physics)
    sensor_data: mx.array    # (num_envs, nsensordata) - MjData.sensordata
    ctrl: mx.array           # (num_envs, ncontrol) - Current control input
    obs: mx.array
    reward: mx.array
    terminated: mx.array
    truncated: mx.array
    info: dict

    @property
    def done(self) -> mx.array:
        """
        Check if the environment is done.
        """
        return mx.logical_or(self.terminated, self.truncated)

    def replace(self, **updates) -> "MjMlxEnvState":
        return dataclasses.replace(self, **updates)

    def validate(self):
        num_envs = self.physics_state.shape[0]
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape
        assert self.ctrl.shape[0] == num_envs, self.ctrl.shape


class MjMlxEnv(ABEnv):
    _model: mujoco.MjModel
    _cfg: EnvCfg
    _state: MjMlxEnvState = None
    _num_envs: int
    _step_runner: mlx_step.MlxStepRunner = None
    _worker_data: List[mujoco.MjData] = None # Preallocated MuJoCo compute workers
    _reset_forward_executor: Optional[ThreadPoolExecutor] = None
    _last_sensor_traj: mx.array = None

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
        self._reset_forward_executor = ThreadPoolExecutor(max_workers=self._n_threads) if self._n_threads > 1 else None
        
        # Persistent MLX simulation-step runner.
        self._step_runner = mlx_step.MlxStepRunner(nthread=self._n_threads)

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
        if self._step_runner is not None:
             self._step_runner = None
        if self._reset_forward_executor is not None:
            self._reset_forward_executor.shutdown(wait=True)
            self._reset_forward_executor = None

    @staticmethod
    def _scatter_rows(base: mx.array, indices: List[int], updates: mx.array) -> mx.array:
        if len(indices) == 0:
            return base
        out = base.tolist()
        upd = updates.tolist()
        for k, idx in enumerate(indices):
            out[idx] = upd[k]
        return mx.array(out, dtype=base.dtype)

    @staticmethod
    def _forward_sensor_chunk(
        model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        qpos_batch: np.ndarray,
        qvel_batch: np.ndarray,
        sensor_batch: list,
        start: int,
        end: int,
    ) -> None:
        for i in range(start, end):
            mj_data.time = 0.0
            mj_data.qpos[:] = qpos_batch[i]
            mj_data.qvel[:] = qvel_batch[i]
            mj_data.ctrl[:] = 0.0
            mj_data.qacc[:] = 0.0
            mj_data.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(model, mj_data)
            sensor_batch[i] = mj_data.sensordata.copy()

    def _compute_sensor_batch_from_qpos_qvel(
        self,
        qpos_batch: mx.array,
        qvel_batch: mx.array,
    ) -> mx.array:
        num_reset = qpos_batch.shape[0]
        sensor_batch = [None] * num_reset
        if num_reset == 0:
            return mx.zeros((0, self._model.nsensordata), dtype=mx.float32)
        # Convert once on main thread; worker threads should not touch MLX tensors.
        qpos_np = np.asarray(qpos_batch, dtype=np.float64)
        qvel_np = np.asarray(qvel_batch, dtype=np.float64)

        # For small resets, single-thread path avoids extra scheduling overhead.
        if self._reset_forward_executor is None or num_reset < 64:
            self._forward_sensor_chunk(
                self._model, self._worker_data[0], qpos_np, qvel_np, sensor_batch, 0, num_reset
            )
            return mx.array(np.stack(sensor_batch, axis=0), dtype=mx.float32)

        nworkers = min(self._n_threads, num_reset)
        chunk_size = (num_reset + nworkers - 1) // nworkers
        futures = []
        for worker_id in range(nworkers):
            start = worker_id * chunk_size
            end = min(start + chunk_size, num_reset)
            if start >= end:
                break
            futures.append(
                self._reset_forward_executor.submit(
                    self._forward_sensor_chunk,
                    self._model,
                    self._worker_data[worker_id],
                    qpos_np,
                    qvel_np,
                    sensor_batch,
                    start,
                    end,
                )
            )
        for fut in futures:
            fut.result()
        return mx.array(np.stack(sensor_batch, axis=0), dtype=mx.float32)

    @property
    def model(self) -> mujoco.MjModel:
        """
        Get the mujoco model
        """
        return self._model

    @property
    def state(self) -> MjMlxEnvState:
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

    def init_state(self) -> MjMlxEnvState:
        """
        Create a new environment state
        """
        nstate = mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        nsensordata = self._model.nsensordata
        ncontrol = self._model.nu
        
        physics_state = mx.zeros((self._num_envs, nstate), dtype=mx.float32)
        sensor_data = mx.zeros((self._num_envs, nsensordata), dtype=mx.float32)
        ctrl = mx.zeros((self._num_envs, ncontrol), dtype=mx.float32)

        obs = mx.zeros((self._num_envs, self.observation_space.shape[0]), dtype=mx.float32)
        reward = mx.zeros((self._num_envs,), dtype=mx.float32)
        terminated = mx.ones((self._num_envs,), dtype=mx.bool_)
        truncated = mx.zeros((self._num_envs,), dtype=mx.bool_)
        info = {"steps": mx.zeros((self._num_envs,), dtype=mx.uint32)}
        
        self._state = MjMlxEnvState(physics_state, sensor_data, ctrl, obs, reward, terminated, truncated, info)
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
        done_list = [bool(x) for x in done.tolist()]
        if not any(done_list):
            return

        indices = [i for i, v in enumerate(done_list) if v]
        steps = state.info["steps"].tolist()
        for idx in indices:
            steps[idx] = 0
        state.info["steps"] = mx.array(steps, dtype=state.info["steps"].dtype)
        
        # Call reset. 
        # Note: reset now is responsible for returning new physics states for these indices
        env_indices = mx.array(indices, dtype=mx.int32)
        new_physics_states, new_obs, info1 = self.reset(env_indices)
        
        # Update state
        state.physics_state = self._scatter_rows(state.physics_state, indices, new_physics_states)
        if new_obs is not None:
            state.obs = self._scatter_rows(state.obs, indices, new_obs)
        
        # NOTE: sensor_data is NOT automatically updated by setting physics_state
        # until the next simulation step.
        # If reset() returned None for obs, it implies we expect env to compute it from sensor_data?
        # BUT sensor_data is stale (pre-reset).
        # We must either:
        # A) Compute Obs manually in reset using analytical kinematics (hard)
        # B) Run a forward kinematics update to refresh sensor_data (cleaner)
        # C) Or just rely on next step.
        # But RSL-RL wrapper calls reset_all() -> reset() -> _update_buffers(obs).
        
        # If new_obs is None, we need to compute it.
        if new_obs is None:
            # This means we relied on sensor_data in _compute_obs, but sensor_data is stale!
            # We should probably run a minimal forward step or re-compute.
            pass

        # Update info
        if info1:

            def replace_dict_values(dst, new_values):
                for key, value in new_values.items():
                    if key not in dst:
                        if isinstance(value, mx.array):
                            full_shape = (self._num_envs,) + tuple(value.shape[1:])
                            dst[key] = mx.zeros(full_shape, dtype=value.dtype)
                        elif isinstance(value, dict):
                            dst[key] = {}
                        else:
                            dst[key] = value

                    if isinstance(value, mx.array):
                        dst[key] = self._scatter_rows(dst[key], indices, value)
                    elif isinstance(value, dict):
                        assert isinstance(dst[key], dict)
                        replace_dict_values(dst[key], value)
                    else:
                        dst[key] = value

            replace_dict_values(state.info, info1)
        
        # Since we reset state, sensor data may be stale until next step,
        # unless reset path already computed it.

    def _update_truncate(self):
        """
        Truncate the environments that have reached max episode length
        """
        if not self._cfg.max_episode_steps:
            return
        self._state.truncated = self._state.info["steps"] >= self._cfg.max_episode_steps

    @abc.abstractmethod
    def apply_action(self, actions: mx.array, state: MjMlxEnvState) -> mx.array:
        """
        Compute control input from actions.
        
        Returns:
            mx.array: The control input (ctrl) for the physics step. Shape (num_envs, ncontrol)
        """

    @abc.abstractmethod
    def update_state(self, state: MjMlxEnvState, obs_required: bool = True) -> MjMlxEnvState:
        """
        Update the environment state after physics step (e.g. compute obs, rewards)
        """

    @abc.abstractmethod
    def reset(
        self,
        env_indices: mx.array,
    ) -> Tuple[mx.array, mx.array, dict]:
        """
        Reset the environment for the done envs

        Args:
            env_indices (mx.array): The indices of the envs being reset

        Returns:
            tuple:
                - new_physics_states (mx.array): (len(indices), nstate)
                - new_obs (mx.array): (len(indices), obs_dim)
                - info (dict): Additional info
        """
        pass

    def _step_core(self):
        """
        Step the physics simulation for all environments in parallel using mujoco.mlx_step.
        """
        nsubsteps = self._cfg.sim_substeps
        
        # Prepare inputs
        initial_state = self._state.physics_state
        ctrl = self._state.ctrl
        
        # MLX step runner expects control shape (B, T, D); zero-order hold across substeps.
        control_traj = mx.broadcast_to(ctrl[:, None, :], (self._num_envs, nsubsteps, ctrl.shape[-1]))
        model_batch = [self._model] * self._num_envs
        step_out = self._step_runner.step(
            model=model_batch,
            data=self._worker_data,
            initial_state=initial_state,
            control=control_traj,
            nstep=nsubsteps,
            out_dtype=mx.float32,
        )
        if isinstance(step_out, tuple):
            state_traj, sensor_traj = step_out
        else:
            state_traj = step_out.state_mx
            sensor_traj = step_out.sensordata_mx

        # Store sensor data for potential rendering usage
        # We only really care about the LAST step sensor data for rendering the current frame
        if sensor_traj is not None and sensor_traj.size > 0:
            self._last_sensor_traj = sensor_traj[:, -1, :]
            # Update state sensor data
            self._state.sensor_data = self._last_sensor_traj
        
        # Update physics state (for next step)
        # Get the final state from the trajectory
        if state_traj is not None and state_traj.size > 0:
            self._state.physics_state = state_traj[:, -1, :]

    def _pre_step(self):
        state = self._state
        state.reward = mx.zeros_like(state.reward)
        state.terminated = mx.zeros_like(state.terminated)
        state.truncated = mx.zeros_like(state.truncated)

    def _before_chunk_step(self, data: Any):
        """
        Hook called before executing a chunk of actions.
        """
        pass

    def step(self, actions: mx.array) -> MjMlxEnvState:
        if self._state is None:
            self.init_state()

        actions = mx.array(actions, dtype=mx.float32)

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

        cumulative_reward = mx.zeros((self._num_envs,), dtype=mx.float32)
        chunk_terminated = mx.zeros((self._num_envs,), dtype=mx.bool_)
        chunk_truncated = mx.zeros((self._num_envs,), dtype=mx.bool_)

        for t in range(num_steps):
            self._pre_step()
            
            # Apply Action: Now updates self._state.ctrl
            self._state.ctrl[:] = self.apply_action(actions[:, t], self._state)
            
            self._step_core()
            
            # Optimization: only compute obs on last step
            is_last_step = (t == num_steps - 1)
            self._state = self.update_state(self._state, obs_required=is_last_step)
                
            self._state.info["steps"] += 1
            
            # Accumulate reward before reset might clear it
            cumulative_reward += self._state.reward

            self._update_truncate()
            
            # Accumulate done flags
            chunk_terminated = mx.logical_or(chunk_terminated, self._state.terminated)
            chunk_truncated = mx.logical_or(chunk_truncated, self._state.truncated)
        
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
        return mx.arange(adr, adr + dim)
