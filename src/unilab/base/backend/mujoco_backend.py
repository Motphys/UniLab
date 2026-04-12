import os
import time
from collections.abc import Sequence
from multiprocessing import cpu_count
from typing import Optional, cast

import mujoco
import numpy as np
from mujoco.batch_env import BatchEnvPool

from unilab.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IQUAT,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

from ..dtype_config import get_global_dtype
from .base import BackendPlayCapabilities, SimBackend


def _root_state_dims(model) -> tuple[int, int]:
    if model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    return 0, 0


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
        position_actuator_gains: dict | None = None,
        iterations: int | None = None,
    ):
        self.add_body_sensors = add_body_sensors
        self._base_name = base_name
        self._model_file = model_file
        from unilab.utils.xml_utils import create_discardvisual_xml

        model_path = create_discardvisual_xml(model_file)
        tmp_paths = [model_path]

        if self.add_body_sensors:
            from unilab.utils.xml_utils import inject_mujoco_tracking_sensors

            model_path, self._tracked_body_ids, valid_bnames = inject_mujoco_tracking_sensors(
                model_path,
                baselink_name=base_name,
            )
            tmp_paths.append(model_path)
        else:
            valid_bnames = []

        try:
            self._model = mujoco.MjModel.from_xml_path(model_path)
        finally:
            for tmp_path in reversed(tmp_paths):
                os.remove(tmp_path)

        if self.add_body_sensors:
            self._body_id_to_tracked_idx = np.full(self._model.nbody, -1, dtype=int)
            for idx, bid in enumerate(self._tracked_body_ids):
                self._body_id_to_tracked_idx[bid] = idx

        self._model.opt.timestep = sim_dt
        if iterations is not None:
            self._model.opt.iterations = int(iterations)
        if position_actuator_gains is not None:
            self._apply_position_actuator_gains_to_model(self._model, **position_actuator_gains)
        self._base_body_id = (
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, base_name)
            if base_name is not None
            else -1
        )
        self._base_body_mass = np.asarray(self._model.body_mass).copy()
        self._base_body_ipos = np.asarray(self._model.body_ipos).copy()
        self._num_envs = num_envs
        self._np_dtype = np_dtype if np_dtype is not None else get_global_dtype()
        self.backend_type = "mujoco"

        # 线程配置
        self._n_threads = min(num_envs, cpu_count() * 2)

        self._pool = BatchEnvPool(self._model, nbatch=num_envs, nthread=self._n_threads)

        # 索引
        self.nq = self._model.nq
        self.nv = self._model.nv
        self._idx_qpos = 1
        self._idx_qvel = 1 + self.nq
        self._root_qpos_dim, self._root_qvel_dim = _root_state_dims(self._model)
        self._num_dof_pos = self.nq - self._root_qpos_dim
        self._num_dof_vel = self.nv - self._root_qvel_dim

        # 状态存储
        nstate = mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._physics_state = np.zeros((num_envs, nstate), dtype=self._np_dtype)
        # 用模型默认 qpos（含 identity 四元数）初始化所有环境
        self._physics_state[:, self._idx_qpos : self._idx_qpos + self._model.nq] = self._model.qpos0
        self._sensor_data = np.zeros((num_envs, self._model.nsensordata), dtype=self._np_dtype)

        # 缓存视图
        self._dof_pos_view = self._physics_state[
            :, self._idx_qpos + self._root_qpos_dim : self._idx_qpos + self.nq
        ]
        self._dof_vel_view = self._physics_state[
            :, self._idx_qvel + self._root_qvel_dim : self._idx_qvel + self.nv
        ]
        self._qpos_view = self._physics_state[:, self._idx_qpos : self._idx_qpos + self.nq]
        if self._root_qpos_dim == 7:
            self._base_pos_view = self._physics_state[:, self._idx_qpos : self._idx_qpos + 3]
            self._base_quat_view = self._physics_state[:, self._idx_qpos + 3 : self._idx_qpos + 7]
            self._base_lin_vel_view = self._physics_state[:, self._idx_qvel : self._idx_qvel + 3]
            self._base_ang_vel_view = self._physics_state[
                :, self._idx_qvel + 3 : self._idx_qvel + 6
            ]
        else:
            if self._base_body_id >= 0:
                data0 = mujoco.MjData(self._model)
                mujoco.mj_forward(self._model, data0)
                base_pos = np.asarray(data0.xpos[self._base_body_id], dtype=self._np_dtype).copy()
                base_quat = np.asarray(data0.xquat[self._base_body_id], dtype=self._np_dtype).copy()
            else:
                base_pos = np.zeros((3,), dtype=self._np_dtype)
                base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=self._np_dtype)
            self._base_pos_view = np.broadcast_to(base_pos, (num_envs, 3)).copy()
            self._base_quat_view = np.broadcast_to(base_quat, (num_envs, 4)).copy()
            self._base_lin_vel_view = np.zeros((num_envs, 3), dtype=self._np_dtype)
            self._base_ang_vel_view = np.zeros((num_envs, 3), dtype=self._np_dtype)

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
        sensor_init = self._pool.forward(self._physics_state)
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
    # Model properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def num_actuators(self) -> int:
        return int(self._model.nu)

    @property
    def num_dof_vel(self) -> int:
        return int(self._num_dof_vel)

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.array(self._model.actuator_ctrlrange, dtype=self._np_dtype)

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id < 0:
            raise ValueError(f"Keyframe '{name}' not found in MuJoCo model")
        return np.array(self._model.key_qpos[key_id].copy(), dtype=self._np_dtype)

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self.nv,), dtype=self._np_dtype)

    def get_body_ids(self, names: "Sequence[str]") -> np.ndarray:
        ids: list[int] = []
        for name in names:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body '{name}' not found in MuJoCo model")
            ids.append(bid)
        return np.array(ids, dtype=np.int32)

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_joint_range(self) -> np.ndarray | None:
        if self._root_qpos_dim > 0:
            return np.array(self._model.jnt_range[1:], dtype=self._np_dtype)
        return np.array(self._model.jnt_range, dtype=self._np_dtype)

    # ------------------------------------------------------------------ #
    # Simulation control                                                 #
    # ------------------------------------------------------------------ #

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        t0 = time.perf_counter()
        control_traj = np.broadcast_to(ctrl[:, None, :], (self._num_envs, nsteps, ctrl.shape[-1]))
        set_ctrl_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        state_np = self._pool.step(
            self._physics_state,
            nstep=nsteps,
            control=control_traj,
            control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
        )
        self._physics_state[:] = state_np.astype(self._np_dtype)
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        sensor_np = self._pool.forward(self._physics_state)
        self._sensor_data[:] = sensor_np.astype(self._np_dtype)
        refresh_cache_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "timing": {
                "set_ctrl_ms": set_ctrl_ms,
                "physics_ms": physics_ms,
                "refresh_cache_ms": refresh_cache_ms,
            }
        }

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        if len(env_indices) == 0:
            return

        num_reset = len(env_indices)
        state_np = np.zeros((num_reset, self._physics_state.shape[1]), dtype=np.float64)
        state_np[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos
        state_np[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel

        state_out, sensor_np = self._pool.reset(
            env_ids=np.asarray(env_indices, dtype=np.int32),
            initial_state=state_np,
            randomization=self._translate_reset_randomization(randomization, num_reset),
        )

        self._physics_state[env_indices] = state_out.astype(self._np_dtype)
        self._sensor_data[env_indices] = sensor_np.astype(self._np_dtype)

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                {
                    RESET_TERM_BASE_MASS,
                    RESET_TERM_BASE_COM,
                    RESET_TERM_BODY_IQUAT,
                    RESET_TERM_BODY_INERTIA,
                    RESET_TERM_KP,
                    RESET_TERM_KD,
                }
            ),
            supports_interval_push=True,
        )

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.push_perturbation_limit is None:
            return
        velocity_delta = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3))
        velocity_delta *= np.asarray(plan.push_perturbation_limit, dtype=np.float64)
        self._base_lin_vel_view[:] += velocity_delta.astype(self._np_dtype)

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(supports_physics_state_playback=True)

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
        return np.asarray(self._body_id_to_tracked_idx[body_ids])

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
        return self._sensor_views[name]

    # ------------------------------------------------------------------ #
    # Mujoco-specific                                                 #
    # ------------------------------------------------------------------ #

    def get_physics_state(self) -> np.ndarray:
        return self._physics_state

    def _coerce_reset_field(
        self,
        value: np.ndarray,
        *,
        name: str,
        num_reset: int,
        shaped_tail: tuple[int, ...],
    ) -> np.ndarray:
        arr = cast(np.ndarray, np.asarray(value, dtype=np.float64))
        flat_tail = int(np.prod(shaped_tail))
        flat_shape = (num_reset, flat_tail)
        shaped = (num_reset, *shaped_tail)
        if arr.shape == flat_shape:
            return cast(np.ndarray, arr.copy())
        if arr.shape == shaped:
            return cast(np.ndarray, arr.reshape(num_reset, flat_tail).copy())
        raise ValueError(f"{name} must have shape {flat_shape} or {shaped}, got {arr.shape}")

    def _translate_reset_randomization(
        self,
        randomization: ResetRandomizationPayload | None,
        num_reset: int,
    ) -> dict[str, np.ndarray] | None:
        if randomization is None or randomization.is_empty():
            return None
        if (
            randomization.base_mass_delta is not None or randomization.base_com_offset is not None
        ) and self._base_body_id < 0:
            raise ValueError(f"Body '{self._base_name}' not found in MuJoCo model")

        translated: dict[str, np.ndarray] = {}
        if randomization.base_mass_delta is not None:
            body_mass = np.broadcast_to(self._base_body_mass, (num_reset, self._model.nbody)).copy()
            body_mass[:, self._base_body_id] += np.asarray(randomization.base_mass_delta)
            translated["body_mass"] = body_mass

        if randomization.base_com_offset is not None:
            body_ipos = np.broadcast_to(
                self._base_body_ipos, (num_reset, self._model.nbody, 3)
            ).copy()
            body_ipos[:, self._base_body_id, :] += np.asarray(randomization.base_com_offset)
            translated["body_ipos"] = body_ipos.reshape(num_reset, -1)

        if randomization.body_iquat is not None:
            translated["body_iquat"] = self._coerce_reset_field(
                randomization.body_iquat,
                name="body_iquat",
                num_reset=num_reset,
                shaped_tail=(self._model.nbody, 4),
            )

        if randomization.body_inertia is not None:
            translated["body_inertia"] = self._coerce_reset_field(
                randomization.body_inertia,
                name="body_inertia",
                num_reset=num_reset,
                shaped_tail=(self._model.nbody, 3),
            )

        if randomization.kp is not None:
            translated["kp"] = self._coerce_reset_field(
                randomization.kp,
                name="kp",
                num_reset=num_reset,
                shaped_tail=(self._model.nu,),
            )

        if randomization.kd is not None:
            translated["kd"] = self._coerce_reset_field(
                randomization.kd,
                name="kd",
                num_reset=num_reset,
                shaped_tail=(self._model.nu,),
            )

        return translated or None

    def _apply_position_actuator_gains_to_model(
        self,
        model,
        *,
        kp: float | np.ndarray,
        kd: float | np.ndarray,
        actuator_ids=slice(None),
    ) -> None:
        kp_arr = np.asarray(kp, dtype=np.float64)
        kd_arr = np.asarray(kd, dtype=np.float64)
        model.actuator_gainprm[actuator_ids, 0] = kp_arr
        model.actuator_biasprm[actuator_ids, 1] = -kp_arr
        model.actuator_biasprm[actuator_ids, 2] = -kd_arr
