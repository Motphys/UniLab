import os
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import cpu_count, current_process, get_context
from typing import Any, Optional, cast

import mujoco
import numpy as np
from mujoco.batch_env import BatchEnvPool

from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_IQUAT,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ModelVariantSpec,
    ResetRandomizationPayload,
)
from unilab.dtype_config import get_global_dtype

from ..base import (
    BackendHeightScanner,
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    SimBackend,
    normalize_play_render_mode,
)
from .playback import run_mujoco_playback


def _root_state_dims(model) -> tuple[int, int]:
    if model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    return 0, 0


@dataclass
class _MuJoCoHeightScanner(BackendHeightScanner):
    backend: "MuJoCoBackend"
    hfield_geom_id: int
    offsets: np.ndarray
    frame_body_id: int
    alignment: str
    output: str

    def scan(self) -> np.ndarray:
        pool = self.backend._pool
        if pool is None:
            raise RuntimeError("MuJoCo backend pool must be materialized before hfield scanning")

        heights = pool.sample_hfield_height(
            self.backend._physics_state,
            hfield_geom_id=self.hfield_geom_id,
            offsets=self.offsets,
            frame_body_id=self.frame_body_id,
            alignment=self.alignment,
            output=self.output,
        )
        return np.asarray(heights, dtype=self.backend._np_dtype)


def _prepare_variant_model_xml(
    model_file: str,
    *,
    add_body_sensors: bool,
    base_name: str | None,
) -> tuple[str, list[str]]:
    from unilab.base.backend.mujoco.xml import (
        create_discardvisual_xml,
        inject_mujoco_tracking_sensors,
    )

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
                _apply_position_actuator_gains_to_mj_model(model, **position_actuator_gains)
            output_path = os.path.join(output_dir, f"variant_{idx}.mjb")
            mujoco.mj_saveModel(model, output_path)
            output_paths.append(output_path)
        return tuple(output_paths)
    finally:
        for tmp_path in reversed(tmp_paths):
            os.remove(tmp_path)


def _actuator_ids_from_selector(model, actuator_ids) -> np.ndarray:
    ids = np.arange(model.nu)[actuator_ids]
    return np.atleast_1d(np.asarray(ids, dtype=np.int32))


def _assert_position_actuator_targets(model, actuator_ids=slice(None)) -> None:
    ids = _actuator_ids_from_selector(model, actuator_ids)
    if ids.size == 0:
        return
    affine_bias = int(mujoco.mjtBias.mjBIAS_AFFINE)
    invalid = ids[np.asarray(model.actuator_biastype[ids], dtype=np.int32) != affine_bias]
    if invalid.size == 0:
        return
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(idx)) or str(int(idx))
        for idx in invalid[:8]
    ]
    suffix = "" if invalid.size <= 8 else f", ... ({invalid.size} total)"
    raise ValueError(
        "position_actuator_gains can only target MuJoCo position actuators; "
        f"non-position actuator ids/names: {', '.join(names)}{suffix}"
    )


def _apply_position_actuator_gains_to_mj_model(
    model,
    *,
    kp: float | np.ndarray,
    kd: float | np.ndarray,
    actuator_ids=slice(None),
) -> None:
    _assert_position_actuator_targets(model, actuator_ids)
    kp_arr = np.asarray(kp, dtype=np.float64)
    kd_arr = np.asarray(kd, dtype=np.float64)
    model.actuator_gainprm[actuator_ids, 0] = kp_arr
    model.actuator_biasprm[actuator_ids, 1] = -kp_arr
    model.actuator_biasprm[actuator_ids, 2] = -kd_arr


class _TempXmlCleanup:
    def __init__(self, path: str) -> None:
        self.path = path

    def cleanup(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


@dataclass
class _MuJoCoSceneContext:
    model_source: str | mujoco.MjModel
    model_file: str
    visual_model_file: str | None = None
    artifacts_dir: str | None = None
    terrain_origins: np.ndarray | None = None
    terrain_surface_sampler: Any | None = None
    cleanup_handle: Any | None = None


def _build_mujoco_scene_context(scene: SceneCfg) -> _MuJoCoSceneContext:
    from unilab.base.backend.mujoco.xml import (
        materialize_mujoco_hfield_attached_scene,
        materialize_scene_fragments,
    )

    if scene is None:
        raise ValueError("SceneCfg must be provided")
    if not scene.model_file:
        raise ValueError("SceneCfg.model_file must be provided")

    if scene.terrain is None:
        if not scene.fragment_files:
            return _MuJoCoSceneContext(
                model_source=scene.model_file,
                model_file=scene.model_file,
                visual_model_file=scene.model_file,
            )
        model_source = materialize_scene_fragments(
            scene.model_file,
            fragment_files=scene.fragment_files,
        )
        return _MuJoCoSceneContext(
            model_source=model_source,
            model_file=scene.model_file,
            visual_model_file=model_source,
            cleanup_handle=_TempXmlCleanup(model_source),
        )

    if scene.terrain.generator is None:
        raise ValueError("SceneCfg.terrain.generator must be configured for terrain scenes")

    output_dir = tempfile.TemporaryDirectory(prefix="unilab_scene_")
    try:
        model, terrain_origins, terrain_surface_sampler = materialize_mujoco_hfield_attached_scene(
            model_file=scene.model_file,
            terrain_cfg=scene.terrain.generator,
            output_dir=output_dir.name,
            fragment_files=scene.fragment_files,
            hfield_name=scene.terrain.hfield_name,
            geom_name=scene.terrain.geom_name or "floor",
            return_surface_sampler=True,
        )
    except Exception:
        output_dir.cleanup()
        raise

    return _MuJoCoSceneContext(
        model_source=model,
        model_file=scene.model_file,
        visual_model_file=os.path.join(output_dir.name, "scene.xml"),
        artifacts_dir=output_dir.name,
        terrain_origins=terrain_origins,
        terrain_surface_sampler=terrain_surface_sampler,
        cleanup_handle=output_dir,
    )


class MuJoCoBackend(SimBackend):
    """MuJoCo 后端实现"""

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        base_name: Optional[str] = None,
        np_dtype=None,
        add_body_sensors: bool = False,
        position_actuator_gains: dict | None = None,
        iterations: int | None = None,
        push_body_name: Optional[str] = None,
        post_step_forward_sensor: bool = False,
    ):
        scene_context = _build_mujoco_scene_context(scene)
        self.scene_model_file = scene_context.model_file
        self.scene_visual_model_file = scene_context.visual_model_file
        self.scene_artifacts_dir = scene_context.artifacts_dir
        self.terrain_origins = scene_context.terrain_origins
        self.terrain_surface_sampler = scene_context.terrain_surface_sampler
        self._scene_cleanup_handle = scene_context.cleanup_handle
        self.add_body_sensors = add_body_sensors
        self._base_name = base_name
        self._push_body_name = push_body_name
        self._model_file = scene_context.model_source
        self._sim_dt = float(sim_dt)
        self._iterations = None if iterations is None else int(iterations)
        self._post_step_forward_sensor = bool(post_step_forward_sensor)
        self._position_actuator_gains = (
            None if position_actuator_gains is None else dict(position_actuator_gains)
        )
        self._pre_step_control_fn = None
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
        if isinstance(self._model_file, mujoco.MjModel):
            if self.add_body_sensors:
                raise ValueError("add_body_sensors is not supported for precompiled MuJoCo models")
            self._tracked_body_ids = []
            self._valid_bnames = []
            model = self._model_file
            self._configure_model(model)
            return model

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
        from unilab.base.backend.mujoco.xml import (
            create_discardvisual_xml,
            inject_mujoco_tracking_sensors,
        )

        model_path = create_discardvisual_xml(str(self._model_file))
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

    def _sample_push_force(self, force_range: Sequence[float] | np.ndarray) -> np.ndarray:
        """Sample one world-frame push force vector per environment.

        Args:
            force_range: Per-axis push-force magnitude range.

        Returns:
            Array with shape ``(num_envs, 3)`` containing sampled forces.
        """
        ex_force = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3))
        ex_force *= np.asarray(force_range, dtype=np.float64)
        return ex_force.astype(np.float64, copy=False)

    def _compile_model_variants(
        self,
        variant_specs: Sequence[ModelVariantSpec],
    ) -> tuple[mujoco.MjModel, ...]:
        variants = tuple(variant_specs)
        if not variants:
            return tuple()
        if isinstance(self._model_file, mujoco.MjModel):
            raise ValueError(
                "MuJoCo model variants are not supported for precompiled materialized scenes"
            )

        def _load_compiled_models_and_cleanup(paths: Sequence[str]) -> tuple[mujoco.MjModel, ...]:
            try:
                return tuple(mujoco.MjModel.from_binary_path(path) for path in paths)
            finally:
                for path in paths:
                    if os.path.exists(path):
                        os.remove(path)
                for path in paths:
                    parent = os.path.dirname(path)
                    if parent and os.path.isdir(parent):
                        try:
                            os.rmdir(parent)
                        except OSError:
                            pass

        if len(variants) == 1 or current_process().daemon:
            mjb_paths = _compile_model_variant_chunk_to_mjb(
                model_file=self._model_file,
                add_body_sensors=self.add_body_sensors,
                base_name=self._base_name,
                sim_dt=self._sim_dt,
                iterations=self._iterations,
                position_actuator_gains=self._position_actuator_gains,
                variants=variants,
            )
            return _load_compiled_models_and_cleanup(mjb_paths)

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
        return _load_compiled_models_and_cleanup(flat_paths)

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

    def get_default_qpos(self) -> np.ndarray:
        return np.asarray(self._model.qpos0, dtype=np.float64).copy()

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

    def get_geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Geom '{name}' not found in MuJoCo model")
        return int(geom_id)

    def get_geom_size(self, name: str) -> np.ndarray:
        return np.asarray(self._model.geom_size[self.get_geom_id(name)], dtype=np.float64).copy()

    def create_hfield_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> BackendHeightScanner:
        offsets_np = np.ascontiguousarray(np.asarray(offsets, dtype=np.float64))
        if offsets_np.ndim != 2 or offsets_np.shape[1] != 2:
            raise ValueError(f"offsets must have shape (num_points, 2), got {offsets_np.shape}")

        return _MuJoCoHeightScanner(
            backend=self,
            hfield_geom_id=int(hfield_geom_id),
            offsets=offsets_np,
            frame_body_id=int(frame_body_id),
            alignment=alignment,
            output=output,
        )

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        subtree_ids = {int(root_body_id)}
        changed = True
        while changed:
            changed = False
            for body_id in range(self._model.nbody):
                parent_id = int(self._model.body_parentid[body_id])
                if body_id not in subtree_ids and parent_id in subtree_ids:
                    subtree_ids.add(body_id)
                    changed = True
        return np.asarray(sorted(subtree_ids), dtype=np.int32)

    def get_geom_names(self) -> tuple[str, ...]:
        return tuple(
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in range(self._model.ngeom)
        )

    def get_geom_body_ids(self) -> np.ndarray:
        return np.asarray(self._model.geom_bodyid, dtype=np.int32).copy()

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._model.geom_contype, dtype=np.int32).copy(),
            np.asarray(self._model.geom_conaffinity, dtype=np.int32).copy(),
        )

    def get_geom_friction(self) -> np.ndarray:
        return np.asarray(self._model.geom_friction, dtype=np.float64).copy()

    def get_gravity(self) -> np.ndarray:
        return np.asarray(self._model.opt.gravity, dtype=np.float64).copy()

    def get_body_mass(self) -> np.ndarray:
        return np.asarray(self._model.body_mass, dtype=np.float64).copy()

    def get_body_ipos(self) -> np.ndarray:
        return np.asarray(self._model.body_ipos, dtype=np.float64).copy()

    def get_dof_armature(self) -> np.ndarray:
        return np.asarray(self._model.dof_armature, dtype=np.float64).copy()

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        ids: list[int] = []
        for name in names:
            sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"Site '{name}' not found in MuJoCo model")
            ids.append(sid)
        return np.array(ids, dtype=np.int32)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            indices.append(int(self._model.jnt_dofadr[jid]))
        return np.array(indices, dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            if int(self._model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"Joint '{name}' is not a single-DoF joint")
            indices.append(int(self._model.jnt_qposadr[jid]) - self._root_qpos_dim)
        return np.array(indices, dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_indices(names) - self._root_qvel_dim

    def get_joint_range(self) -> np.ndarray | None:
        jnt_range = self._model.jnt_range
        mask = self._model.jnt_type != int(mujoco.mjtJoint.mjJNT_FREE)
        return np.array(jnt_range[mask], dtype=self._np_dtype)

    # ------------------------------------------------------------------ #
    # Simulation control                                                 #
    # ------------------------------------------------------------------ #

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        if self._pre_step_control_fn is not None:
            return self._step_with_pre_step_control(ctrl, nsteps)

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
        state_np, sensor_np = self._pool.step(  # type: ignore[union-attr]
            self._physics_state,
            nstep=nsteps,
            control=control_traj,
            control_spec=control_spec,
            return_sensor=True,
            post_step_forward_sensor=self._post_step_forward_sensor,
        )
        if control_spec & int(mujoco.mjtState.mjSTATE_XFRC_APPLIED):
            self._pending_xfrc_applied.fill(0.0)
        self._physics_state[:] = state_np.astype(self._np_dtype)
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._sensor_data[:] = sensor_np.astype(self._np_dtype)
        refresh_cache_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "timing": {
                "set_ctrl_ms": set_ctrl_ms,
                "physics_ms": physics_ms,
                "refresh_cache_ms": refresh_cache_ms,
            }
        }

    def _step_with_pre_step_control(
        self, ctrl: np.ndarray, nsteps: int
    ) -> dict[str, dict[str, float]]:
        set_ctrl_ms = 0.0
        physics_ms = 0.0
        refresh_cache_ms = 0.0
        has_pending_xfrc = bool(np.any(self._pending_xfrc_applied))

        for _ in range(nsteps):
            t0 = time.perf_counter()
            native_ctrl = self._apply_pre_step_control(ctrl)
            control_traj = native_ctrl[:, None, :]
            control_spec = int(mujoco.mjtState.mjSTATE_CTRL)
            if has_pending_xfrc:
                control_spec |= int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
                xfrc_traj = self._pending_xfrc_applied[:, None, :]
                control_traj = np.concatenate((control_traj, xfrc_traj), axis=-1)
            set_ctrl_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            state_np, sensor_np = self._pool.step(  # type: ignore[union-attr]
                self._physics_state,
                nstep=1,
                control=control_traj,
                control_spec=control_spec,
                return_sensor=True,
                post_step_forward_sensor=self._post_step_forward_sensor,
            )
            self._physics_state[:] = state_np.astype(self._np_dtype)
            physics_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            self._sensor_data[:] = sensor_np.astype(self._np_dtype)
            refresh_cache_ms += (time.perf_counter() - t0) * 1000.0

        if has_pending_xfrc:
            self._pending_xfrc_applied.fill(0.0)

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
                    RESET_TERM_BODY_IPOS,
                    RESET_TERM_BODY_MASS,
                    RESET_TERM_DOF_ARMATURE,
                    RESET_TERM_GEOM_FRICTION,
                    RESET_TERM_KP,
                    RESET_TERM_KD,
                }
            ),
            supports_interval_push=self._push_body_id >= 0,
            supports_interval_body_force=True,
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
        if plan.is_empty():
            return
        self._pending_xfrc_applied.fill(0.0)
        if plan.push_perturbation_limit is not None:
            self.push_robots(plan.push_perturbation_limit)
        if plan.body_force is not None:
            if plan.body_ids is None:
                raise ValueError("Interval body-force perturbation requires body_ids")
            self.apply_body_force(plan.body_ids, plan.body_force)
        if plan.body_linear_velocity_delta is not None:
            if plan.body_ids is None:
                raise ValueError("Interval body-velocity perturbation requires body_ids")
            self.apply_body_linear_velocity_delta(plan.body_ids, plan.body_linear_velocity_delta)

    def push_robots(self, force_range: Sequence[float] | np.ndarray) -> None:
        self._pending_xfrc_applied.fill(0.0)
        self._pending_xfrc_applied[:, self._push_body_force_slice] = self._sample_push_force(
            force_range
        )

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
    ) -> None:
        """Accumulate one external world-frame force vector per target body.

        Args:
            body_ids: Body ids to perturb.
            force: Force tensor with shape ``(num_envs, len(body_ids), 3)``.

        Returns:
            None. The force is staged in ``xfrc_applied`` for the next step.
        """
        body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        force_np = np.asarray(force, dtype=np.float64)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
        if force_np.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {force_np.shape}")
        for body_offset, body_id in enumerate(body_ids_np):
            self._pending_xfrc_applied[:, self._resolve_push_body_force_slice(int(body_id))] += (
                force_np[:, body_offset, :]
            )

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(supports_physics_state_playback=True)

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        effective_mode = "record" if mode == "auto" else mode
        if effective_mode == "none":
            return BackendPlayRenderPlan(
                mode=effective_mode,
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if effective_mode == "interactive":
            raise NotImplementedError("MuJoCo playback does not support interactive rendering.")
        assert effective_mode == "record"
        if play_steps is None:
            raise ValueError("MuJoCo record playback requires a finite training.play_steps value.")
        if output_video is None:
            raise ValueError("MuJoCo record playback requires an output video path.")
        return BackendPlayRenderPlan(
            mode=effective_mode,
            headless=True,
            record_video=True,
            num_steps=int(play_steps),
            output_video=output_video,
        )

    def run_playback(
        self,
        *,
        env: Any,
        initialize,
        step,
        num_steps: int | None,
        output_video: str | os.PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter=None,
        camera_kwargs: dict[str, Any] | None = None,
        extra_data_getter=None,
    ) -> str | None:
        del render_offset_mode
        should_record_video = (
            bool(record_video) if record_video is not None else output_video is not None
        )
        should_run_headless = bool(headless) if headless is not None else should_record_video
        return run_mujoco_playback(
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            headless=should_run_headless,
            record_video=should_record_video,
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera_kwargs,
            extra_data_getter=extra_data_getter,
        )

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

    def get_body_pose_w_rows(
        self, env_ids: np.ndarray, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(env_ids, dtype=np.intp)
        mapped = self._get_mapped_indices(body_ids)
        return self._tracked_pos_w_all[rows[:, None], mapped], self._tracked_quat_w_all[
            rows[:, None], mapped
        ]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_linvel_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_angvel_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_state_w(
        self, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mapped = self._get_mapped_indices(body_ids)
        return (
            self._tracked_pos_w_all[:, mapped, :],
            self._tracked_quat_w_all[:, mapped, :],
            self._tracked_linvel_w_all[:, mapped, :],
            self._tracked_angvel_w_all[:, mapped, :],
        )

    def copy_body_state_w(
        self,
        body_ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mapped = self._get_mapped_indices(body_ids)
        np.take(self._tracked_pos_w_all, mapped, axis=1, out=out_pos)
        np.take(self._tracked_quat_w_all, mapped, axis=1, out=out_quat)
        np.take(self._tracked_linvel_w_all, mapped, axis=1, out=out_lin_vel)
        np.take(self._tracked_angvel_w_all, mapped, axis=1, out=out_ang_vel)
        return out_pos, out_quat, out_lin_vel, out_ang_vel

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

    def get_sensor_data_rows(self, name: str, env_ids: np.ndarray) -> np.ndarray:
        return self._sensor_views[name][np.asarray(env_ids, dtype=np.intp)]

    def get_sensor_data_batch(self, names: Sequence[str]) -> np.ndarray:
        sensor_names = tuple(names)
        if not sensor_names:
            return np.empty((self._num_envs, 0), dtype=self._np_dtype)
        values = [self._sensor_views[name].reshape(self._num_envs, -1) for name in sensor_names]
        return np.concatenate(values, axis=1)

    def get_site_jacobian_w(
        self,
        site_id: int,
        dof_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量 Jacobian，shape (num_envs, 3, len(dof_indices))。

        使用 BatchEnvPool.compute_site_jacobians 原生接口，无需每 env 独立 MjData。
        scalar site_id → pool 返回 (N, 3, nv)（k 维被 squeeze）。
        """
        dof_indices = np.asarray(dof_indices, dtype=np.int32).reshape(-1)
        jp, jr = self._pool.compute_site_jacobians(  # type: ignore[union-attr]
            self._physics_state.astype(np.float64),
            int(site_id),
            jacp=True,
            jacr=True,
        )
        return (
            jp[:, :, dof_indices].astype(self._np_dtype),
            jr[:, :, dof_indices].astype(self._np_dtype),
        )

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
        body_mass = None
        if randomization.body_mass is not None:
            body_mass = self._coerce_reset_field(
                randomization.body_mass,
                name="body_mass",
                num_reset=num_reset,
                shaped_tail=(self._model.nbody,),
            )
        if randomization.base_mass_delta is not None:
            if body_mass is None:
                body_mass = np.broadcast_to(
                    self._base_body_mass, (num_reset, self._model.nbody)
                ).copy()
            body_mass[:, self._base_body_id] += np.asarray(randomization.base_mass_delta)
        if body_mass is not None:
            translated["body_mass"] = body_mass

        body_ipos = None
        if randomization.body_ipos is not None:
            body_ipos = self._coerce_reset_field(
                randomization.body_ipos,
                name="body_ipos",
                num_reset=num_reset,
                shaped_tail=(self._model.nbody, 3),
            )
        if randomization.base_com_offset is not None:
            if body_ipos is None:
                body_ipos = np.broadcast_to(
                    self._base_body_ipos, (num_reset, self._model.nbody, 3)
                ).copy()
            body_ipos[:, self._base_body_id, :] += np.asarray(randomization.base_com_offset)
        if body_ipos is not None:
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

        if randomization.geom_friction is not None:
            translated["geom_friction"] = self._coerce_reset_field(
                randomization.geom_friction,
                name="geom_friction",
                num_reset=num_reset,
                shaped_tail=(self._model.ngeom, 3),
            )

        if randomization.dof_armature is not None:
            translated["dof_armature"] = self._coerce_reset_field(
                randomization.dof_armature,
                name="dof_armature",
                num_reset=num_reset,
                shaped_tail=(self._model.nv,),
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
        _apply_position_actuator_gains_to_mj_model(
            model,
            kp=kp,
            kd=kd,
            actuator_ids=actuator_ids,
        )
