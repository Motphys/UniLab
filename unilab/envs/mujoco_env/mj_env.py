from __future__ import annotations

import abc
import dataclasses
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from multiprocessing import cpu_count

import mujoco
try:
    from mujoco import mlx_step
    import mlx.core as mx
except Exception:
    mlx_step = None
    mx = None
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
        
        # Configure thread pool for rollout.
        # Allow explicit override by env var; otherwise auto-tune for large batches.
        thread_override = os.getenv("UNILAB_MLX_STEP_THREADS")
        if thread_override is not None:
            self._n_threads = min(num_envs, max(1, int(thread_override)))
        else:
            host_threads = cpu_count()
            if num_envs >= 4096:
                auto_threads = max(host_threads, 56)
            elif num_envs >= 2048:
                auto_threads = max(host_threads, 32)
            else:
                auto_threads = host_threads
            self._n_threads = min(num_envs, auto_threads)
        self._step_chunk_size = max(1, int(os.getenv("UNILAB_MLX_STEP_CHUNK", "16")))
        
        # Create worker MjData pool
        # These are purely for computation and do not hold persistent environment state.
        self._worker_data = [mujoco.MjData(self._model) for _ in range(self._n_threads)]
        self._reset_forward_executor = ThreadPoolExecutor(max_workers=self._n_threads) if self._n_threads > 1 else None
        
        # Persistent MLX simulation-step runner.
        self._step_runner = mlx_step.MlxStepRunner(nthread=self._n_threads)

        # Float dtype for MLX state/obs/ctrl/reward (global option: UNILAB_MLX_DTYPE=fp16|float16 for FP16)
        _dtype_str = os.getenv("UNILAB_MLX_DTYPE", "float32").strip().lower()
        self._mlx_dtype = mx.float16 if _dtype_str in ("float16", "fp16") else mx.float32

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
    def _scatter_rows(base: mx.array, indices: mx.array, updates: mx.array) -> mx.array:
        if indices.size == 0:
            return base
        idx = indices.astype(mx.int32)
        current = mx.take(base, idx, axis=0)
        return base.at[idx].add(updates - current)

    @staticmethod
    def _forward_sensor_chunk(
        model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        qpos_batch: np.ndarray,
        qvel_batch: np.ndarray,
        sensor_batch: np.ndarray,
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
            sensor_batch[i, :] = mj_data.sensordata

    def _compute_sensor_batch_from_qpos_qvel(
        self,
        qpos_batch,
        qvel_batch,
    ) -> mx.array:
        num_reset = qpos_batch.shape[0]
        if num_reset == 0:
            return mx.zeros((0, self._model.nsensordata), dtype=self._mlx_dtype)
        sensor_batch = np.empty((num_reset, self._model.nsensordata), dtype=np.float32)
        qpos_np = np.asarray(qpos_batch, dtype=np.float64)
        qvel_np = np.asarray(qvel_batch, dtype=np.float64)

        # For small resets, single-thread path avoids extra scheduling overhead.
        if self._reset_forward_executor is None or num_reset < 64:
            self._forward_sensor_chunk(
                self._model, self._worker_data[0], qpos_np, qvel_np, sensor_batch, 0, num_reset
            )
            return mx.array(sensor_batch, dtype=self._mlx_dtype)

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
        return mx.array(sensor_batch, dtype=self._mlx_dtype)

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
        
        physics_state = mx.zeros((self._num_envs, nstate), dtype=self._mlx_dtype)
        sensor_data = mx.zeros((self._num_envs, nsensordata), dtype=self._mlx_dtype)
        ctrl = mx.zeros((self._num_envs, ncontrol), dtype=self._mlx_dtype)

        obs = mx.zeros((self._num_envs, self.observation_space.shape[0]), dtype=self._mlx_dtype)
        reward = mx.zeros((self._num_envs,), dtype=self._mlx_dtype)
        terminated = mx.ones((self._num_envs,), dtype=mx.bool_)
        truncated = mx.zeros((self._num_envs,), dtype=mx.bool_)
        info = {
            "steps": mx.zeros((self._num_envs,), dtype=mx.uint32),
            "timing": {
                "env_step_total_ms": 0.0,
                "step_core_ms": 0.0,
                "update_state_ms": 0.0,
                "reset_done_ms": 0.0,
                "reset_index_extract_ms": 0.0,
                "reset_call_ms": 0.0,
                "reset_scatter_ms": 0.0,
                "reset_info_merge_ms": 0.0,
            },
        }
        
        self._state = MjMlxEnvState(physics_state, sensor_data, ctrl, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def _reset_done_envs(self):
        """
        Reset the environments that are done. 
        """
        t_reset_start = time.perf_counter()
        state = self._state
        done = state.done
        assert done.shape == (self._num_envs,)
        t_index0 = time.perf_counter()
        done_np = np.asarray(done, dtype=np.bool_)
        idx_np = np.flatnonzero(done_np)
        done_count = int(idx_np.size)
        if done_count == 0:
            timing = state.info.setdefault("timing", {})
            timing["reset_done_ms"] = (time.perf_counter() - t_reset_start) * 1000.0
            timing["reset_index_extract_ms"] = (time.perf_counter() - t_index0) * 1000.0
            timing["reset_call_ms"] = 0.0
            timing["reset_scatter_ms"] = 0.0
            timing["reset_info_merge_ms"] = 0.0
            return
        env_indices = mx.array(idx_np.astype(np.int32), dtype=mx.int32)
        index_extract_time = time.perf_counter() - t_index0
        scatter_time = 0.0

        t_scatter0 = time.perf_counter()
        steps = state.info["steps"]
        state.info["steps"] = self._scatter_rows(
            steps,
            env_indices,
            mx.zeros((done_count,), dtype=steps.dtype),
        )
        scatter_time += time.perf_counter() - t_scatter0
        
        # Call reset. 
        # Note: reset now is responsible for returning new physics states for these indices
        t_call0 = time.perf_counter()
        new_physics_states, new_obs, info1 = self.reset(env_indices)
        reset_call_time = time.perf_counter() - t_call0
        
        # Update state
        t_scatter0 = time.perf_counter()
        state.physics_state = self._scatter_rows(state.physics_state, env_indices, new_physics_states)
        if new_obs is not None:
            state.obs = self._scatter_rows(state.obs, env_indices, new_obs)
        scatter_time += time.perf_counter() - t_scatter0
        
        assert new_obs is not None

        # Update info
        info_merge_time = 0.0
        if info1:
            t_info0 = time.perf_counter()

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
                        dst[key] = self._scatter_rows(dst[key], env_indices, value)
                    elif isinstance(value, dict):
                        assert isinstance(dst[key], dict)
                        replace_dict_values(dst[key], value)
                    else:
                        dst[key] = value

            replace_dict_values(state.info, info1)
            info_merge_time = time.perf_counter() - t_info0

        timing = state.info.setdefault("timing", {})
        timing["reset_done_ms"] = (time.perf_counter() - t_reset_start) * 1000.0
        timing["reset_index_extract_ms"] = index_extract_time * 1000.0
        timing["reset_call_ms"] = reset_call_time * 1000.0
        timing["reset_scatter_ms"] = scatter_time * 1000.0
        timing["reset_info_merge_ms"] = info_merge_time * 1000.0
        
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
        state_traj, sensor_traj = self._step_runner.step(
            model=self._model,
            data=self._worker_data,
            initial_state=initial_state,
            control=control_traj,
            nstep=nsubsteps,
            chunk_size=self._step_chunk_size,
            out_dtype=self._mlx_dtype,
            return_last_only=True,
        )
        self._last_sensor_traj = sensor_traj
        self._state.sensor_data[:] = self._last_sensor_traj
        self._state.physics_state[:] = state_traj

    def _pre_step(self):
        state = self._state
        state.reward[:] = 0.0
        state.terminated[:] = False
        state.truncated[:] = False

    def _before_chunk_step(self, data: Any):
        """
        Hook called before executing a chunk of actions.
        """
        pass

    def step(self, actions: mx.array) -> MjMlxEnvState:
        """
        Step with actions of shape (B, D) only.
        B = num_envs, D = action_space.shape[0].
        """
        step_t0 = time.perf_counter()
        if self._state is None:
            self.init_state()

        actions = mx.array(actions, dtype=self._mlx_dtype)
        assert actions.ndim == 2, f"actions must be (B, D), got ndim={actions.ndim}"
        assert actions.shape[0] == self._num_envs, (
            f"actions.shape[0] must be num_envs={self._num_envs}, got {actions.shape[0]}"
        )
        assert actions.shape[1] == self.action_space.shape[0], (
            f"actions.shape[1] must be action_dim={self.action_space.shape[0]}, got {actions.shape[1]}"
        )

        self._before_chunk_step(None)

        self._pre_step()
        self._state.ctrl[:] = self.apply_action(actions, self._state)

        t_core0 = time.perf_counter()
        self._step_core()
        step_core_time = time.perf_counter() - t_core0

        t_upd0 = time.perf_counter()
        self._state = self.update_state(self._state, obs_required=True)
        update_state_time = time.perf_counter() - t_upd0

        self._state.info["steps"] += 1
        self._update_truncate()

        t_reset0 = time.perf_counter()
        self._reset_done_envs()
        reset_done_time = time.perf_counter() - t_reset0
        timing = self._state.info.setdefault("timing", {})
        timing["env_step_total_ms"] = (time.perf_counter() - step_t0) * 1000.0
        timing["step_core_ms"] = step_core_time * 1000.0
        timing["update_state_ms"] = update_state_time * 1000.0
        timing["reset_done_ms"] = reset_done_time * 1000.0

        return self._state

    def _get_sensor_range(self, name, dim):
        id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if id == -1:
            raise ValueError(f"Sensor {name} not found in model")
        adr = self._model.sensor_adr[id]
        return mx.arange(adr, adr + dim)
