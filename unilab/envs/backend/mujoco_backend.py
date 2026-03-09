import os
from multiprocessing import cpu_count
import mujoco
from mujoco import rollout, batch_forward
import numpy as np
from .base import ISimBackend
from ..dtype_config import get_global_dtype


class MuJoCoBackend(ISimBackend):
    """MuJoCo 后端实现"""

    def __init__(self, model_file: str, num_envs: int, sim_dt: float, body_name: str = None, np_dtype=None):
        self._model = mujoco.MjModel.from_xml_path(model_file)
        self._model.opt.timestep = sim_dt
        self._num_envs = num_envs
        self._np_dtype = np_dtype if np_dtype is not None else get_global_dtype()

        # 线程配置
        thread_override = os.getenv("UNILAB_MUJOCO_STEP_THREADS")
        if thread_override:
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

        # Worker pool
        self._worker_data = [mujoco.MjData(self._model) for _ in range(self._n_threads)]
        self._rollout = rollout.Rollout(nthread=self._n_threads)
        self._forward_runner = batch_forward.BatchForwardRunner(nthread=self._n_threads)

        # 状态存储
        nstate = mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._physics_state = np.zeros((num_envs, nstate), dtype=self._np_dtype)
        self._sensor_data = np.zeros((num_envs, self._model.nsensordata), dtype=self._np_dtype)

        # 索引
        self.nq = self._model.nq
        self.nv = self._model.nv
        self._idx_qpos = 1
        self._idx_qvel = 1 + self.nq
        self._num_dof_pos = self.nq - 7
        self._num_dof_vel = self.nv - 6

        # 缓存视图
        self._dof_pos_view = self._physics_state[:, self._idx_qpos + 7 : self._idx_qpos + self.nq]
        self._dof_vel_view = self._physics_state[:, self._idx_qvel + 6 : self._idx_qvel + self.nv]
        self._qpos_view = self._physics_state[:, self._idx_qpos : self._idx_qpos + self.nq]

        # 传感器索引
        self._sensor_indices = {}
        self._sensor_views = {}
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                adr = self._model.sensor_adr[i]
                dim = self._model.sensor_dim[i]
                self._sensor_indices[name] = list(range(adr, adr + dim))
                self._sensor_views[name] = self._sensor_data[:, adr:adr + dim]

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        control_traj = np.broadcast_to(ctrl[:, None, :], (self._num_envs, nsteps, ctrl.shape[-1]))
        state_traj, sensor_traj = self._rollout.rollout(
            self._model, self._worker_data,
            initial_state=self._physics_state.astype(np.float64),
            control=control_traj, nstep=nsteps
        )
        self._physics_state[:] = state_traj[:, -1, :].astype(self._np_dtype)
        self._sensor_data[:] = sensor_traj[:, -1, :].astype(self._np_dtype)

    def get_dof_pos(self) -> np.ndarray:
        return self._dof_pos_view

    def get_dof_vel(self) -> np.ndarray:
        return self._dof_vel_view

    def get_qpos(self) -> np.ndarray:
        return self._qpos_view

    def get_sensor_data(self, name: str) -> np.ndarray:
        if name not in self._sensor_views:
            raise ValueError(f"Sensor '{name}' not found")
        return self._sensor_views[name]

    def set_state(self, env_indices: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        num_reset = len(env_indices)
        state_np = np.zeros((num_reset, self._physics_state.shape[1]), dtype=np.float64)
        state_np[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos
        state_np[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel

        _, sensor_np = self._forward_runner.forward(
            model=self._model,
            data=self._worker_data,
            initial_state=state_np,
            chunk_size=max(1, num_reset // self._n_threads),
            skipsensor=False,
            out_dtype=np.float64,
            return_state=True,
        )

        self._physics_state[env_indices] = state_np.astype(self._np_dtype)
        self._sensor_data[env_indices] = sensor_np.astype(self._np_dtype)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self):
        return self._model

    def get_physics_state(self) -> np.ndarray:
        return self._physics_state
