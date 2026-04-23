import os
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, get_context
from typing import Optional, cast

import mujoco
import numpy as np
from mujoco.batch_env import BatchEnvPool

from unilab.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IQUAT,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ModelVariantSpec,
    ResetRandomizationPayload,
)

from ..dtype_config import get_global_dtype
from .base import BackendPlayCapabilities, SimBackend


def _root_state_dims(model) -> tuple[int, int]:
    if model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    return 0, 0


def _prepare_variant_model_xml(
    model_file: str,
    *,
    add_body_sensors: bool,
    base_name: str | None,
) -> tuple[str, list[str]]:
    from unilab.base.backend.xml import create_discardvisual_xml, inject_mujoco_tracking_sensors

    model_path = create_discardvisual_xml(model_file)
    tmp_paths = [model_path]
    if add_body_sensors:
        model_path, _, _ = inject_mujoco_tracking_sensors(
            model_path,
            baselink_name=base_name,
        )
        tmp_paths.append(model_path)
    return model_path, tmp_paths


def _compile_model_variant_chunk_to_mjb(
    *,
    model_file: str,
    add_body_sensors: bool,
    base_name: str | None,
    sim_dt: float,
    iterations: int | None,
    position_actuator_gains: dict | None,
    variants: tuple[ModelVariantSpec, ...],
) -> tuple[str, ...]:
    model_path, tmp_paths = _prepare_variant_model_xml(
        model_file,
        add_body_sensors=add_body_sensors,
        base_name=base_name,
    )
    output_dir = tempfile.mkdtemp(prefix="unilab-mj-variant-")
    try:
        base_spec = mujoco.MjSpec.from_file(model_path)
        output_paths: list[str] = []
        for idx, variant in enumerate(variants):
            spec = base_spec.copy()
            for override in variant.geom_size_overrides:
                geom = spec.geom(override.geom_name)
                if geom is None:
                    raise ValueError(
                        f"Geom '{override.geom_name}' not found in MuJoCo model '{model_file}'"
                    )
                geom.size = list(override.size)
            model = spec.compile()
            model.opt.timestep = sim_dt
            if iterations is not None:
                model.opt.iterations = int(iterations)
            if position_actuator_gains is not None:
                kp_arr = np.asarray(position_actuator_gains["kp"], dtype=np.float64)
                kd_arr = np.asarray(position_actuator_gains["kd"], dtype=np.float64)
                model.actuator_gainprm[:, 0] = kp_arr
                model.actuator_biasprm[:, 1] = -kp_arr
                model.actuator_biasprm[:, 2] = -kd_arr
            output_path = os.path.join(output_dir, f"variant_{idx}.mjb")
            mujoco.mj_saveModel(model, output_path)
            output_paths.append(output_path)
        return tuple(output_paths)
    finally:
        for tmp_path in reversed(tmp_paths):
            os.remove(tmp_path)


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
        push_body_name: Optional[str] = None,
    ):
        self.add_body_sensors = add_body_sensors
        self._base_name = base_name
        self._push_body_name = push_body_name
        self._model_file = model_file
        self._sim_dt = float(sim_dt)
        self._iterations = None if iterations is None else int(iterations)
        self._position_actuator_gains = (
            None if position_actuator_gains is None else dict(position_actuator_gains)
        )
        self._model = self._load_base_model()
        self._base_body_id = (
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, base_name)
            if base_name is not None
            else -1
        )
        self._push_body_id = self._resolve_push_body_id(self._model)
        self._push_body_force_slice = self._resolve_push_body_force_slice(self._push_body_id)
        self._base_body_mass = np.asarray(self._model.body_mass).copy()
        self._base_body_ipos = np.asarray(self._model.body_ipos).copy()
        self._num_envs = num_envs
        self._np_dtype = np_dtype if np_dtype is not None else get_global_dtype()
        self.backend_type = "mujoco"
        self._pending_xfrc_applied = np.zeros((num_envs, 6 * self._model.nbody), dtype=np.float64)

        # 线程配置
        self._n_threads = min(num_envs, cpu_count() * 2)

        self._model_variants: tuple[mujoco.MjModel, ...] = (self._model,)
        self._model_assignments = np.zeros((num_envs,), dtype=np.int32)
        self._pool: BatchEnvPool | None = None

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
        if self.add_body_sensors and self._valid_bnames:

            def _get_sensor_view(prefix, dim):
                adrs = [
                    self._model.sensor_adr[
                        mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}_{nb}")
                    ]
                    for nb in self._valid_bnames
                ]
                return self._sensor_data[:, adrs[0] : adrs[-1] + dim].reshape(
                    num_envs, len(self._valid_bnames), dim
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

    def _load_base_model(self) -> mujoco.MjModel:
        model_path, tmp_paths, tracked_body_ids, valid_bnames = self._prepare_model_xml()
        try:
            model = mujoco.MjModel.from_xml_path(model_path)
        finally:
            for tmp_path in reversed(tmp_paths):
                os.remove(tmp_path)

        self._tracked_body_ids = tracked_body_ids
        if self.add_body_sensors:
            self._body_id_to_tracked_idx = np.full(model.nbody, -1, dtype=int)
            for idx, bid in enumerate(self._tracked_body_ids):
                self._body_id_to_tracked_idx[bid] = idx
        self._valid_bnames = valid_bnames
        self._configure_model(model)
        return model

    def _prepare_model_xml(self) -> tuple[str, list[str], list[int], list[str]]:
        from unilab.base.backend.xml import create_discardvisual_xml, inject_mujoco_tracking_sensors

        model_path = create_discardvisual_xml(self._model_file)
        tmp_paths = [model_path]
        if self.add_body_sensors:
            model_path, tracked_body_ids, valid_bnames = inject_mujoco_tracking_sensors(
                model_path,
                baselink_name=self._base_name,
            )
            tmp_paths.append(model_path)
        else:
            tracked_body_ids = []
            valid_bnames = []
        return model_path, tmp_paths, tracked_body_ids, valid_bnames

    def _configure_model(self, model: mujoco.MjModel) -> None:
        model.opt.timestep = self._sim_dt
        if self._iterations is not None:
            model.opt.iterations = self._iterations
        if self._position_actuator_gains is not None:
            self._apply_position_actuator_gains_to_model(model, **self._position_actuator_gains)

    def _resolve_push_body_id(self, model: mujoco.MjModel) -> int:
        body_name = self._push_body_name if self._push_body_name is not None else self._base_name
        if body_name is None:
            return -1
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Push body '{body_name}' not found in MuJoCo model")
        return int(body_id)

    def _resolve_push_body_force_slice(self, body_id: int) -> slice:
        if body_id < 0:
            return slice(0, 0)
        start = 6 * body_id
        return slice(start, start + 3)

    def _compile_model_variants(
        self,
        variant_specs: Sequence[ModelVariantSpec],
    ) -> tuple[mujoco.MjModel, ...]:
        variants = tuple(variant_specs)
        if not variants:
            return tuple()
        if len(variants) == 1:
            mjb_paths = _compile_model_variant_chunk_to_mjb(
                model_file=self._model_file,
                add_body_sensors=self.add_body_sensors,
                base_name=self._base_name,
                sim_dt=self._sim_dt,
                iterations=self._iterations,
                position_actuator_gains=self._position_actuator_gains,
                variants=variants,
            )
            try:
                models = tuple(mujoco.MjModel.from_binary_path(path) for path in mjb_paths)
            finally:
                for path in mjb_paths:
                    if os.path.exists(path):
                        os.remove(path)
                for path in mjb_paths:
                    parent = os.path.dirname(path)
                    if parent and os.path.isdir(parent):
                        try:
                            os.rmdir(parent)
                        except OSError:
                            pass
            return models

        max_workers = min(len(variants), max(1, cpu_count()))
        chunk_size = max(1, (len(variants) + max_workers - 1) // max_workers)
        chunks = tuple(
            tuple(variants[idx : idx + chunk_size]) for idx in range(0, len(variants), chunk_size)
        )
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=get_context("spawn"),
            ) as executor:
                futures = [
                    executor.submit(
                        _compile_model_variant_chunk_to_mjb,
                        model_file=self._model_file,
                        add_body_sensors=self.add_body_sensors,
                        base_name=self._base_name,
                        sim_dt=self._sim_dt,
                        iterations=self._iterations,
                        position_actuator_gains=self._position_actuator_gains,
                        variants=chunk,
                    )
                    for chunk in chunks
                ]
            mjb_paths_nested = [future.result() for future in futures]
        except PermissionError:
            mjb_paths_nested = [
                _compile_model_variant_chunk_to_mjb(
                    model_file=self._model_file,
                    add_body_sensors=self.add_body_sensors,
                    base_name=self._base_name,
                    sim_dt=self._sim_dt,
                    iterations=self._iterations,
                    position_actuator_gains=self._position_actuator_gains,
                    variants=chunk,
                )
                for chunk in chunks
            ]
        flat_paths = [path for paths in mjb_paths_nested for path in paths]
        try:
            models = tuple(mujoco.MjModel.from_binary_path(path) for path in flat_paths)
        finally:
            for path in flat_paths:
                if os.path.exists(path):
                    os.remove(path)
            for path in flat_paths:
                parent = os.path.dirname(path)
                if parent and os.path.isdir(parent):
                    try:
                        os.rmdir(parent)
                    except OSError:
                        pass
        return models

    def _current_model_sequence(self) -> mujoco.MjModel | list[mujoco.MjModel]:
        if len(self._model_variants) == 1 and np.all(self._model_assignments == 0):
            return self._model_variants[0]
        return [self._model_variants[int(idx)] for idx in self._model_assignments]

    def _build_pool(self) -> BatchEnvPool:
        pool = BatchEnvPool(
            self._current_model_sequence(),
            nbatch=self._num_envs,
            nthread=self._n_threads,
        )
        sensor_init = pool.forward(self._physics_state)
        self._sensor_data[:] = sensor_init.astype(self._np_dtype)
        return pool

    def _apply_model_assignments(
        self,
        model_variants: tuple[mujoco.MjModel, ...],
        model_assignments: np.ndarray,
    ) -> None:
        if len(model_assignments) != self._num_envs:
            raise ValueError(
                f"model_assignments must have length {self._num_envs}, got {len(model_assignments)}"
            )
        if len(model_variants) == 0:
            raise ValueError("model_variants must be non-empty")
        if np.any(model_assignments < 0) or np.any(model_assignments >= len(model_variants)):
            raise ValueError(
                f"model_assignments must be in [0, {len(model_variants) - 1}], got {model_assignments}"
            )

        self._model_variants = model_variants
        self._model_assignments = np.asarray(model_assignments, dtype=np.int32).copy()
        self._model = model_variants[int(self._model_assignments[0])]
        self._base_body_id = (
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._base_name)
            if self._base_name is not None
            else -1
        )
        self._push_body_id = self._resolve_push_body_id(self._model)
        self._push_body_force_slice = self._resolve_push_body_force_slice(self._push_body_id)
        self._base_body_mass = np.asarray(self._model.body_mass).copy()
        self._base_body_ipos = np.asarray(self._model.body_ipos).copy()
        self._pending_xfrc_applied = np.zeros(
            (self._num_envs, 6 * self._model.nbody), dtype=np.float64
        )
        self._physics_state[:, self._idx_qpos : self._idx_qpos + self._model.nq] = self._model.qpos0

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
        control_spec = int(mujoco.mjtState.mjSTATE_CTRL)
        if np.any(self._pending_xfrc_applied):
            control_spec |= int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
            xfrc_traj = np.broadcast_to(
                self._pending_xfrc_applied[:, None, :],
                (self._num_envs, nsteps, self._pending_xfrc_applied.shape[-1]),
            )
            control_traj = np.concatenate((control_traj, xfrc_traj), axis=-1)
        set_ctrl_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        state_np = self._pool.step(  # type: ignore[union-attr]
            self._physics_state,
            nstep=nsteps,
            control=control_traj,
            control_spec=control_spec,
        )
        if control_spec & int(mujoco.mjtState.mjSTATE_XFRC_APPLIED):
            self._pending_xfrc_applied.fill(0.0)
        self._physics_state[:] = state_np.astype(self._np_dtype)
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        sensor_np = self._pool.forward(self._physics_state)  # type: ignore[union-attr]
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

        state_out, sensor_np = self._pool.reset(  # type: ignore[union-attr]
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
                    RESET_TERM_GRAVITY,
                    RESET_TERM_BODY_IQUAT,
                    RESET_TERM_BODY_INERTIA,
                    RESET_TERM_KP,
                    RESET_TERM_KD,
                }
            ),
            supports_interval_push=self._push_body_id >= 0,
        )

    def apply_init_randomization(self, plan: InitRandomizationPlan) -> None:
        if plan.is_empty():
            return
        if self._pool is not None:
            raise RuntimeError("MuJoCo init randomization must run before pool materialization")
        model_assignments = np.asarray(plan.model_assignments, dtype=np.int32)
        model_variants = self._compile_model_variants(plan.model_variants)
        self._apply_model_assignments(model_variants, model_assignments)

    def materialize(self) -> None:
        if self._pool is not None:
            raise RuntimeError("MuJoCo backend pool is already materialized")
        self._pool = self._build_pool()

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.push_perturbation_limit is None:
            return
        self.push_robots(plan.push_perturbation_limit)

    def push_robots(self, force_range: Sequence[float] | np.ndarray) -> None:
        ex_force = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3))
        ex_force *= force_range
        self._pending_xfrc_applied.fill(0.0)
        self._pending_xfrc_applied[:, self._push_body_force_slice] = ex_force

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
        return self._body_id_to_tracked_idx[body_ids]  # type: ignore[no-any-return]

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_pos_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_quat_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_linvel_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_angvel_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                   #
    # ------------------------------------------------------------------ #

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_pos_b_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_quat_b_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_linvel_b_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_angvel_b_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

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

    def get_playback_model(self, env_index: int | None = None):
        """Return the MuJoCo model used by playback for one vectorized env.

        Args:
            env_index: Optional vectorized environment index.

        Returns:
            The MuJoCo model assigned to that env, or the current backend model
            when no explicit index is requested.
        """
        if env_index is None:
            return self._model
        idx = int(env_index)
        if idx < 0 or idx >= self._num_envs:
            raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        return self._model_variants[int(self._model_assignments[idx])]

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

        if randomization.gravity is not None:
            translated["gravity"] = self._coerce_reset_field(
                randomization.gravity,
                name="gravity",
                num_reset=num_reset,
                shaped_tail=(3,),
            )

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

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint (kp, kd) arrays read from the current model state."""
        kp = np.asarray(self._model.actuator_gainprm[:, 0], dtype=np.float64).copy()
        kd = np.asarray(-self._model.actuator_biasprm[:, 2], dtype=np.float64).copy()
        return kp, kd

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
