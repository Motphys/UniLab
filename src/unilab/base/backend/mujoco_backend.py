import os
from multiprocessing import cpu_count
from typing import Optional

import mujoco
import numpy as np
from mujoco.batch_env import BatchEnvPool

from ..dtype_config import get_global_dtype
from .base import SimBackend


class MuJoCoBackend(SimBackend):
    """MuJoCo 后端实现"""

    def __init__(
        self,
        model_file: str,
        num_envs: int,
        sim_dt: float,
        base_name: Optional[str] = None,
        np_dtype=None,
        add_body_sensors: bool = False,
    ):
        self.add_body_sensors = add_body_sensors

        if self.add_body_sensors:
            from unilab.utils.xml_utils import inject_mujoco_tracking_sensors

            tmp_path, self._tracked_body_ids, valid_bnames = inject_mujoco_tracking_sensors(
                model_file, baselink_name=base_name
            )
            try:
                self._model = mujoco.MjModel.from_xml_path(tmp_path)
            finally:
                os.remove(tmp_path)

            self._body_id_to_tracked_idx = np.full(self._model.nbody, -1, dtype=int)
            for idx, bid in enumerate(self._tracked_body_ids):
                self._body_id_to_tracked_idx[bid] = idx
        else:
            self._model = mujoco.MjModel.from_xml_path(model_file)
            valid_bnames = []

        self._model.opt.timestep = sim_dt
        self._num_envs = num_envs
        self._np_dtype = np_dtype if np_dtype is not None else get_global_dtype()
        self.backend_type = "mujoco"

        # 线程配置
        thread_override = os.getenv("UNILAB_MUJOCO_STEP_THREADS")
        if thread_override:
            self._n_threads = min(num_envs, max(1, int(thread_override)))
        else:
            host_threads = cpu_count()
            self._n_threads = min(num_envs, host_threads * 2)

        self._pool = BatchEnvPool(self._model, nbatch=num_envs, nthread=self._n_threads)

        # 索引
        self.nq = self._model.nq
        self.nv = self._model.nv
        self._idx_qpos = 1
        self._idx_qvel = 1 + self.nq
        self._num_dof_pos = self.nq - 7
        self._num_dof_vel = self.nv - 6

        # 状态存储
        nstate = mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._physics_state = np.zeros((num_envs, nstate), dtype=self._np_dtype)
        # 用模型默认 qpos（含 identity 四元数）初始化所有环境
        self._physics_state[:, self._idx_qpos : self._idx_qpos + self._model.nq] = self._model.qpos0
        self._sensor_data = np.zeros((num_envs, self._model.nsensordata), dtype=self._np_dtype)

        # 缓存视图
        self._dof_pos_view = self._physics_state[:, self._idx_qpos + 7 : self._idx_qpos + self.nq]
        self._dof_vel_view = self._physics_state[:, self._idx_qvel + 6 : self._idx_qvel + self.nv]
        self._qpos_view = self._physics_state[:, self._idx_qpos : self._idx_qpos + self.nq]
        self._base_pos_view = self._physics_state[:, self._idx_qpos : self._idx_qpos + 3]
        self._base_quat_view = self._physics_state[:, self._idx_qpos + 3 : self._idx_qpos + 7]
        self._base_lin_vel_view = self._physics_state[:, self._idx_qvel : self._idx_qvel + 3]
        self._base_ang_vel_view = self._physics_state[:, self._idx_qvel + 3 : self._idx_qvel + 6]

        # 传感器索引
        self._sensor_indices = {}
        self._sensor_views = {}
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                adr = self._model.sensor_adr[i]
                dim = self._model.sensor_dim[i]
                self._sensor_indices[name] = list(range(adr, adr + dim))
                self._sensor_views[name] = self._sensor_data[:, adr : adr + dim]

        # 针对追踪身体传感器的零拷贝视图映射
        if self.add_body_sensors and valid_bnames:

            def _get_sensor_view(prefix, dim):
                adrs = [
                    self._model.sensor_adr[
                        mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}_{nb}")
                    ]
                    for nb in valid_bnames
                ]
                return self._sensor_data[:, adrs[0] : adrs[-1] + dim].reshape(
                    num_envs, len(valid_bnames), dim
                )

            # Global (world) sensors
            self._tracked_pos_w_all = _get_sensor_view("track_pos_w", 3)
            self._tracked_quat_w_all = _get_sensor_view("track_quat_w", 4)
            self._tracked_linvel_w_all = _get_sensor_view("track_linvel_w", 3)
            self._tracked_angvel_w_all = _get_sensor_view("track_angvel_w", 3)

            # Local (baselink) sensors
            self._tracked_pos_b_all = _get_sensor_view("track_pos_b", 3)
            self._tracked_quat_b_all = _get_sensor_view("track_quat_b", 4)
            self._tracked_linvel_b_all = _get_sensor_view("track_linvel_b", 3)
            self._tracked_angvel_b_all = _get_sensor_view("track_angvel_b", 3)

        # 对初始 qpos0 状态运行一次 forward pass，确保传感器数据有效
        if self._model.nsensordata > 0:
            _, sensor_init = self._pool.forward(self._physics_state)
            self._sensor_data[:] = sensor_init.astype(self._np_dtype)

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self):
        return self._model

    # ------------------------------------------------------------------ #
    # Simulation control                                                 #
    # ------------------------------------------------------------------ #

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        control_traj = np.broadcast_to(ctrl[:, None, :], (self._num_envs, nsteps, ctrl.shape[-1]))
        state_np = self._pool.step(
            self._physics_state,
            nstep=nsteps,
            control=control_traj,
            control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
        )
        self._physics_state[:] = state_np.astype(self._np_dtype)

        # Recompute sensors from the final physics state to keep state/sensor
        # alignment consistent with set_state() and initialization paths.
        if self._model.nsensordata > 0:
            _, sensor_np = self._pool.forward(self._physics_state)
            self._sensor_data[:] = sensor_np.astype(self._np_dtype)

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: dict[str, np.ndarray] | None = None,
    ) -> None:
        if len(env_indices) == 0:
            return

        num_reset = len(env_indices)
        state_np = np.zeros((num_reset, self._physics_state.shape[1]), dtype=np.float64)
        state_np[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos
        state_np[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel

        env_ids = np.asarray(env_indices, dtype=np.int32)
        state_out, sensor_np = self._pool.reset(
            env_ids=env_ids,
            initial_state=state_np,
            randomization=randomization,
        )

        self._physics_state[env_ids] = state_out.astype(self._np_dtype)
        if self._model.nsensordata > 0:
            self._sensor_data[env_ids] = sensor_np.astype(self._np_dtype)

    # ------------------------------------------------------------------ #
    # Base kinematics                                                    #
    # ------------------------------------------------------------------ #

    def get_base_pos(self) -> np.ndarray:
        return self._base_pos_view

    def get_base_quat(self) -> np.ndarray:
        return self._base_quat_view

    def get_base_lin_vel(self) -> np.ndarray:
        return self._base_lin_vel_view

    def get_base_ang_vel(self) -> np.ndarray:
        return self._base_ang_vel_view

    # ------------------------------------------------------------------ #
    # DOF state                                                          #
    # ------------------------------------------------------------------ #

    def get_dof_pos(self) -> np.ndarray:
        return self._dof_pos_view

    def get_dof_vel(self) -> np.ndarray:
        return self._dof_vel_view

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                      #
    # ------------------------------------------------------------------ #

    def _get_mapped_indices(self, body_ids: np.ndarray) -> np.ndarray:
        # if not self.add_body_sensors:
        #     raise NotImplementedError(
        #         "Slow kinematics computation has been removed for performance reasons. "
        #         "Please pass add_body_sensors=True during initialization to enable tracking."
        #     )
        mapped_indices = self._body_id_to_tracked_idx[body_ids]
        # if np.any(mapped_indices == -1):
        #     raise ValueError(
        #         "Cannot query untracked (unnamed) bodies with fast sensor method. Please ensure bodies are named in XML."
        #     )
        return mapped_indices  # type: ignore[no-any-return]

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._tracked_pos_w_all[:, self._get_mapped_indices(body_ids), :])

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._tracked_quat_w_all[:, self._get_mapped_indices(body_ids), :])

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._tracked_linvel_w_all[:, self._get_mapped_indices(body_ids), :])

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._tracked_angvel_w_all[:, self._get_mapped_indices(body_ids), :])

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                   #
    # ------------------------------------------------------------------ #

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._tracked_pos_b_all[:, self._get_mapped_indices(body_ids), :])

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._tracked_quat_b_all[:, self._get_mapped_indices(body_ids), :])

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._tracked_linvel_b_all[:, self._get_mapped_indices(body_ids), :])

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._tracked_angvel_b_all[:, self._get_mapped_indices(body_ids), :])

    # ------------------------------------------------------------------ #
    # Sensors                                                            #
    # ------------------------------------------------------------------ #

    def get_sensor_data(self, name: str) -> np.ndarray:
        # if name not in self._sensor_views:
        #     raise ValueError(f"Sensor '{name}' not found")
        return self._sensor_views[name]

    # ------------------------------------------------------------------ #
    # Mujoco-specific                                                 #
    # ------------------------------------------------------------------ #

    def get_physics_state(self) -> np.ndarray:
        return self._physics_state
