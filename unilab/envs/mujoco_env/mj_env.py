import abc
import dataclasses
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Any
from multiprocessing import cpu_count

import mujoco
from mujoco import batch_forward
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
    _forward_runner: batch_forward.BatchForwardRunner = None
    _worker_data: List[mujoco.MjData] = None # Pool of workers for rollout
    _last_sensor_traj: np.ndarray = None
    _np_dtype: np.dtype

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
        thread_override = os.getenv("UNILAB_MUJOCO_STEP_THREADS")
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
        self._step_chunk_size = max(1, int(os.getenv("UNILAB_MUJOCO_STEP_CHUNK", "16")))
        
        # Create worker MjData pool
        # These are purely for computation and do not hold persistent environment state.
        self._worker_data = [mujoco.MjData(self._model) for _ in range(self._n_threads)]
        
        # Using persistent rollout runner
        self._rollout_runner = rollout.Rollout(nthread=self._n_threads)

        # Persistent C++ batch-forward runner for fast reset sensor refresh.
        self._forward_runner = batch_forward.BatchForwardRunner(nthread=self._n_threads)

        _dtype_str = os.getenv("UNILAB_MLX_DTYPE", "float32").strip().lower()
        self._np_dtype = np.float16 if _dtype_str in ("float16", "fp16") else np.float32

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
        if self._forward_runner is not None:
            self._forward_runner.close()
            self._forward_runner = None

    @staticmethod
    def _scatter_rows(base: np.ndarray, indices: np.ndarray, updates: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return base
        out = np.array(base, copy=True)
        out[indices] = updates
        return out

    def _compute_sensor_batch_from_state(self, physics_state_batch: np.ndarray) -> np.ndarray:
        num_reset = int(physics_state_batch.shape[0])
        if num_reset == 0:
            return np.zeros((0, self._model.nsensordata), dtype=self._np_dtype)

        state_np = np.asarray(physics_state_batch, dtype=np.float64)
        _, sensor_np = self._forward_runner.forward(
            model=self._model,
            data=self._worker_data,
            initial_state=state_np,
            chunk_size=self._step_chunk_size,
            skipsensor=False,
            out_dtype=self._np_dtype,
            return_state=True,
        )
        return np.asarray(sensor_np, dtype=self._np_dtype)

    def _compute_sensor_batch_from_qpos_qvel(
        self,
        qpos_batch,
        qvel_batch,
    ) -> np.ndarray:
        num_reset = qpos_batch.shape[0]
        if num_reset == 0:
            return np.zeros((0, self._model.nsensordata), dtype=self._np_dtype)

        qpos_np = np.asarray(qpos_batch, dtype=np.float64)
        qvel_np = np.asarray(qvel_batch, dtype=np.float64)
        state_np = np.zeros((num_reset, self.physics_state_dim), dtype=np.float64)
        state_np[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos_np
        state_np[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel_np
        return self._compute_sensor_batch_from_state(state_np)

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
        
        physics_state = np.zeros((self._num_envs, nstate), dtype=self._np_dtype)
        sensor_data = np.zeros((self._num_envs, nsensordata), dtype=self._np_dtype)
        ctrl = np.zeros((self._num_envs, ncontrol), dtype=self._np_dtype)

        obs = np.zeros((self._num_envs, self.observation_space.shape[0]), dtype=self._np_dtype)
        reward = np.zeros((self._num_envs,), dtype=self._np_dtype)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {
            "steps": np.zeros((self._num_envs,), dtype=np.uint32),
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
        
        self._state = MjNpEnvState(physics_state, sensor_data, ctrl, obs, reward, terminated, truncated, info)
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
        env_indices = idx_np.astype(np.int32)
        index_extract_time = time.perf_counter() - t_index0
        scatter_time = 0.0

        t_scatter0 = time.perf_counter()
        steps = state.info["steps"]
        state.info["steps"] = self._scatter_rows(
            steps,
            env_indices,
            np.zeros((done_count,), dtype=steps.dtype),
        )
        scatter_time += time.perf_counter() - t_scatter0

        if "final_observation" not in state.info:
            state.info["final_observation"] = np.zeros_like(state.obs)
            state.info["_final_observation"] = np.zeros((self._num_envs,), dtype=np.bool_)

        state.info["_final_observation"][:] = False
        state.info["_final_observation"] = self._scatter_rows(
            state.info["_final_observation"],
            env_indices,
            np.ones((done_count,), dtype=np.bool_),
        )

        terminal_obs = np.take(state.obs, env_indices, axis=0)
        state.info["final_observation"] = self._scatter_rows(
            state.info["final_observation"],
            env_indices,
            terminal_obs,
        )
        
        # Call reset. 
        # Note: reset now is responsible for returning new physics states for these indices
        t_call0 = time.perf_counter()
        new_physics_states, new_obs, info1 = self.reset(env_indices)
        reset_call_time = time.perf_counter() - t_call0

        # Update state
        t_scatter0 = time.perf_counter()
        state.physics_state = self._scatter_rows(state.physics_state, env_indices, new_physics_states)
        state.sensor_data = self._scatter_rows(
            state.sensor_data,
            env_indices,
            self._compute_sensor_batch_from_state(new_physics_states),
        )
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
                        if isinstance(value, np.ndarray):
                            full_shape = (self._num_envs,) + tuple(value.shape[1:])
                            dst[key] = np.zeros(full_shape, dtype=value.dtype)
                        elif isinstance(value, dict):
                            dst[key] = {}
                        else:
                            dst[key] = value

                    if isinstance(value, np.ndarray):
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

    def _step_core(self):
        """
        Step the physics simulation for all environments in parallel using mujoco.rollout.
        """
        nsubsteps = self._cfg.sim_substeps
        
        # Prepare inputs
        initial_state = self._state.physics_state
        ctrl = self._state.ctrl
        
        # Rollout expects control shape (B, T, D); zero-order hold across substeps.
        control_traj = np.broadcast_to(ctrl[:, None, :], (self._num_envs, nsubsteps, ctrl.shape[-1]))
        
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

        if state_traj is not None:
            state_traj = np.asarray(state_traj, dtype=np.float32)
        if sensor_traj is not None:
            sensor_traj = np.asarray(sensor_traj, dtype=np.float32)

        if sensor_traj is not None and sensor_traj.size > 0:
            self._last_sensor_traj = sensor_traj[:, -1, :]
            self._state.sensor_data[:] = self._last_sensor_traj

        if state_traj is not None and state_traj.size > 0:
            self._state.physics_state[:] = state_traj[:, -1, :]

    def _pre_step(self):
        state = self._state
        state.reward.fill(0.0)
        state.terminated.fill(False)
        state.truncated.fill(False)

    def step(self, actions: np.ndarray) -> MjNpEnvState:
        step_t0 = time.perf_counter()
        if self._state is None:
            self.init_state()

        actions = np.asarray(actions, dtype=self._np_dtype)
        assert actions.ndim == 2, f"actions must be (B, D), got ndim={actions.ndim}"
        assert actions.shape[0] == self._num_envs, (
            f"actions.shape[0] must be num_envs={self._num_envs}, got {actions.shape[0]}"
        )
        assert actions.shape[1] == self.action_space.shape[0], (
            f"actions.shape[1] must be action_dim={self.action_space.shape[0]}, got {actions.shape[1]}"
        )

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
        return np.arange(adr, adr + dim)
