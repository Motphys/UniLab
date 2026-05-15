import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import numpy as np

from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

try:
    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderSettings

    MOTRIX_AVAILABLE = True
except ImportError:
    MOTRIX_AVAILABLE = False

from ..base import (
    BackendHeightScanner,
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    SimBackend,
    normalize_play_render_mode,
)
from ..motrix_camera import (
    MotrixTrackingCamera,
    render_offsets,
    resolve_system_camera_view,
    tracking_camera_lookat,
)
from .playback import run_motrix_playback

T = TypeVar("T")
DEFAULT_MOTRIX_MAX_ITERATIONS = 3


def _require_not_none(value: T | None, error_message: str) -> T:
    if value is None:
        raise ValueError(error_message)
    return value


def _first_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float32)
    return float(arr.reshape(-1)[0])


@dataclass
class _MotrixSceneContext:
    model: "mtx.SceneModel"
    terrain_origins: np.ndarray | None = None
    terrain_surface_sampler: object | None = None
    cleanup_handle: object | None = None


@dataclass
class _MotrixTerrainScanner(BackendHeightScanner):
    scanner: "mtx.TerrainScanner"
    data: "mtx.SceneData"
    out: np.ndarray

    def scan(self) -> np.ndarray:
        heights = np.asarray(self.scanner.scan(self.data, out=self.out))
        if heights.shape != self.out.shape:
            raise ValueError(
                f"Motrix TerrainScanner.scan returned shape {heights.shape}, "
                f"expected {self.out.shape}"
            )
        return heights


def _build_motrix_scene_context(
    scene: SceneCfg,
    *,
    add_body_sensors: bool,
    base_name: str,
) -> _MotrixSceneContext:
    from unilab.base.backend.motrix.scene import (
        materialize_motrix_hfield_attached_scene,
        materialize_motrix_scene,
    )

    if scene is None:
        raise ValueError("SceneCfg must be provided")
    if not scene.model_file:
        raise ValueError("SceneCfg.model_file must be provided")

    if scene.terrain is None:
        model = materialize_motrix_scene(
            model_file=scene.model_file,
            fragment_files=scene.fragment_files,
            add_body_sensors=add_body_sensors,
            base_name=base_name,
        )
        return _MotrixSceneContext(model=model)

    if scene.terrain.generator is None:
        raise ValueError("SceneCfg.terrain.generator must be configured for terrain scenes")

    model, terrain_origins, terrain_surface_sampler = materialize_motrix_hfield_attached_scene(
        model_file=scene.model_file,
        terrain_cfg=scene.terrain.generator,
        fragment_files=scene.fragment_files,
        hfield_name=scene.terrain.hfield_name,
        geom_name=scene.terrain.geom_name or "floor",
        add_body_sensors=add_body_sensors,
        base_name=base_name,
        return_surface_sampler=True,
    )
    return _MotrixSceneContext(
        model=model,
        terrain_origins=terrain_origins,
        terrain_surface_sampler=terrain_surface_sampler,
    )


class MotrixBackend(SimBackend):
    """MotrixSim 后端实现"""

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        base_name: str = "base",
        np_dtype=np.float32,
        add_body_sensors: bool = False,
        max_iterations: int | None = DEFAULT_MOTRIX_MAX_ITERATIONS,
        push_body_name: str | None = None,
    ):
        if not MOTRIX_AVAILABLE:
            raise ImportError("motrixsim not available")

        scene_context = _build_motrix_scene_context(
            scene,
            add_body_sensors=add_body_sensors,
            base_name=base_name,
        )
        self._scene = scene
        self.scene_artifacts_dir = None
        self.terrain_origins = scene_context.terrain_origins
        self.terrain_surface_sampler = scene_context.terrain_surface_sampler
        self._scene_cleanup_handle = scene_context.cleanup_handle
        self._base_name = base_name

        self._model = scene_context.model
        self._body_id_to_name = {  # type: ignore[assignment]
            link.index: link.name for link in self._model.links if link.name
        }

        self._model.options.timestep = sim_dt
        if max_iterations is None:
            max_iterations = DEFAULT_MOTRIX_MAX_ITERATIONS
        self._model.options.max_iterations = int(max_iterations)
        self._num_envs = num_envs
        self._np_dtype = np_dtype
        self._pre_step_control_fn = None

        self._data = mtx.SceneData(self._model, batch=[num_envs])  # pyright: ignore[reportPossiblyUnbound]
        self._body: "mtx.Body" = _require_not_none(
            self._model.get_body(base_name), f"Body '{base_name}' not found in Motrix model"
        )
        self._body_link: "mtx.Link" = _require_not_none(
            self._model.get_link(base_name), f"Link '{base_name}' not found in Motrix model"
        )
        push_body = push_body_name if push_body_name is not None else base_name
        self._push_body_link: "mtx.Link" = _require_not_none(
            self._model.get_link(push_body), f"Push link '{push_body}' not found in Motrix model"
        )
        self._body_floatingbase = self._body.floatingbase
        self._joint_dof_pos_indices = np.asarray(self._model.joint_dof_pos_indices, dtype=np.intp)
        self._joint_dof_vel_indices = np.asarray(self._model.joint_dof_vel_indices, dtype=np.intp)
        position_actuators: list["mtx.PositionActuator"] = []
        for actuator in self._model.actuators:
            if actuator.typ == "position":
                position_actuators.append(cast("mtx.PositionActuator", actuator))
        self._position_actuators = position_actuators
        self._supports_position_actuator_gains = len(self._position_actuators) == int(
            self._model.num_actuators
        )
        self._default_actuator_kp = np.zeros((self.num_actuators,), dtype=np.float32)
        self._default_actuator_kd = np.zeros((self.num_actuators,), dtype=np.float32)
        for actuator in self._position_actuators:
            idx = int(actuator.index)
            # TODO: switch to motrixsim model-level actuator gain API once available.
            self._default_actuator_kp[idx] = _first_scalar(actuator.get_kp_override(self._data))
            self._default_actuator_kd[idx] = _first_scalar(actuator.get_kd_override(self._data))
        self._floating_base_quat_indices: tuple[np.ndarray, ...] = tuple(
            np.asarray(floating_base.dof_pos_indices[3:7], dtype=np.intp)
            for floating_base in getattr(self._model, "floating_bases", [])
            if len(floating_base.dof_pos_indices) >= 7
        )
        self._links_by_id: dict[int, "mtx.Link"] = {
            int(link.index): link for link in self._model.links
        }
        self._supports_external_force = all(
            callable(getattr(link, "add_external_force", None))
            for link in self._links_by_id.values()
        )
        self._applied_body_forces: dict[int, np.ndarray] = {}
        self._geoms_by_id: dict[int, "mtx.Geom"] = {
            int(geom.index): geom for geom in self._model.geoms
        }
        # TODO(motrixsim): once pure visual geoms either stop exposing friction
        # override methods or safely no-op them, drop this collision-mask filter.
        self._geom_friction_override_ids = tuple(
            geom_id
            for geom_id, geom in self._geoms_by_id.items()
            if (
                int(getattr(geom, "collision_group", 0)) != 0
                or int(getattr(geom, "collision_affinity", 0)) != 0
            )
        )
        self._supports_geom_friction_override = all(
            callable(getattr(geom, "get_friction_override", None))
            and callable(getattr(geom, "set_friction_override", None))
            for geom_id, geom in self._geoms_by_id.items()
            if geom_id in self._geom_friction_override_ids
        )
        self._supports_gravity_override = callable(
            getattr(self._model, "get_gravity_override", None)
        ) and callable(getattr(self._model, "set_gravity_override", None))
        self._default_body_mass = np.zeros((int(self._model.num_links),), dtype=np.float32)
        self._default_body_ipos = np.zeros((int(self._model.num_links), 3), dtype=np.float32)
        for link_id, link in self._links_by_id.items():
            self._default_body_mass[link_id] = _first_scalar(link.get_mass_override(self._data))
            self._default_body_ipos[link_id] = np.asarray(
                link.get_center_of_mass_override(self._data),
                dtype=np.float32,
            ).reshape(self._num_envs, 3)[0]
        self._default_geom_friction = np.zeros((int(self._model.num_geoms), 3), dtype=np.float32)
        if self._supports_geom_friction_override:
            for geom_id in self._geom_friction_override_ids:
                geom = self._geoms_by_id[geom_id]
                self._default_geom_friction[geom_id] = np.asarray(
                    geom.get_friction_override(self._data),
                    dtype=np.float32,
                ).reshape(self._num_envs, 3)[0]
        self._init_geom_size_overrides: dict[int, np.ndarray] = {}
        self._render_app: "RenderApp | None" = None
        self._render_headless: bool | None = None
        self._render_capture_enabled = False
        self._render_offsets_np: np.ndarray | None = None
        self._render_tracking_camera: MotrixTrackingCamera | None = None
        self.backend_type = "motrix"
        self._link_velocity_cache: np.ndarray | None = None

        # Pre-cache link objects to avoid repeated get_link() lookups.
        self._link_cache: dict[int, "mtx.Link"] = {}
        for link in self._model.links:
            if link.name:
                self._link_cache[link.index] = link

        # 运行一次正向运动学，确保初始 link 位置和传感器数据有效。
        self._model.forward_kinematic(self._data)
        self._refresh_link_pose_cache()

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        ids: list[int] = []
        for name in names:
            link_id = self._model.get_link_index(name)
            if link_id is None or link_id < 0:
                raise ValueError(f"Motion body '{name}' not found in Motrix model")
            # Motion datasets use MuJoCo-style body ids, where worldbody is id 0.
            ids.append(int(link_id) + 1)
        return np.array(ids, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    # ------------------------------------------------------------------ #
    # Model properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def num_actuators(self) -> int:
        return int(self._model.num_actuators)

    @property
    def num_dof_vel(self) -> int:
        return int(len(self._joint_dof_vel_indices))

    def get_actuator_ctrl_range(self) -> np.ndarray:
        arr: np.ndarray = np.array(self._model.actuator_ctrl_limits, dtype=self._np_dtype)
        result: np.ndarray = arr.T.copy()
        return result

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if hasattr(self._model, "keyframes") and self._model.num_keyframes > 0:
            qpos = np.array(self._model.keyframes[0].dof_pos, dtype=self._np_dtype)
        else:
            qpos = np.array(self._model.compute_init_dof_pos(), dtype=self._np_dtype)
        return self._motrix_qpos_to_mujoco(qpos)

    def get_default_qpos(self) -> np.ndarray:
        qpos = np.array(self._model.compute_init_dof_pos(), dtype=self._np_dtype)
        return self._motrix_qpos_to_mujoco(qpos)

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._model.num_dof_vel,), dtype=self._np_dtype)

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        ids: list[int] = []
        for name in names:
            bid = self._model.get_link_index(name)
            if bid is None or bid < 0:
                raise ValueError(f"Body '{name}' not found in Motrix model")
            ids.append(int(bid))
        return np.array(ids, dtype=np.int32)

    def get_geom_id(self, name: str) -> int:
        geom_id = self._model.get_geom_index(name)
        if geom_id is None or geom_id < 0:
            raise ValueError(f"Geom '{name}' not found in Motrix model")
        return int(geom_id)

    def get_geom_size(self, name: str) -> np.ndarray:
        geom = _require_not_none(
            self._model.get_geom(name),
            f"Geom '{name}' not found in Motrix model",
        )
        return np.asarray(geom.size, dtype=np.float64).copy()

    def get_body_mass(self) -> np.ndarray:
        return self._default_body_mass.copy()

    def get_body_ipos(self) -> np.ndarray:
        return self._default_body_ipos.copy()

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        root_id = int(root_body_id)
        if root_id < 0 or root_id >= int(self._model.num_links):
            raise ValueError(f"root_body_id out of range: {root_id}")
        if root_id != int(self._body_link.index):
            raise NotImplementedError(
                "MotrixBackend only exposes the configured base articulation subtree"
            )

        subtree_ids = {root_id}
        for link in self._model.links:
            link_id = int(link.index)
            joint_indices = getattr(link, "joint_indices", ())
            if link_id != root_id and len(joint_indices) > 0:
                subtree_ids.add(link_id)
        return np.asarray(sorted(subtree_ids), dtype=np.int32)

    def get_geom_names(self) -> tuple[str, ...]:
        return tuple(
            str(getattr(self._geoms_by_id[geom_id], "name", "") or "")
            for geom_id in range(int(self._model.num_geoms))
        )

    def get_geom_body_ids(self) -> np.ndarray:
        body_ids = np.zeros((int(self._model.num_geoms),), dtype=np.int32)
        for geom_id in range(int(self._model.num_geoms)):
            link = getattr(self._geoms_by_id[geom_id], "link", None)
            if link is None:
                body_ids[geom_id] = -1
            else:
                body_ids[geom_id] = int(link.index)
        return body_ids

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        contype = np.zeros((int(self._model.num_geoms),), dtype=np.int32)
        conaffinity = np.zeros((int(self._model.num_geoms),), dtype=np.int32)
        for geom_id in range(int(self._model.num_geoms)):
            geom = self._geoms_by_id[geom_id]
            if not hasattr(geom, "collision_group") or not hasattr(geom, "collision_affinity"):
                raise NotImplementedError("Motrix geom objects do not expose contact masks")
            contype[geom_id] = int(geom.collision_group)
            conaffinity[geom_id] = int(geom.collision_affinity)
        return contype, conaffinity

    def get_geom_friction(self) -> np.ndarray:
        if not self._supports_geom_friction_override:
            raise NotImplementedError("Motrix geom friction override is not available")
        return self._default_geom_friction.copy()

    def get_gravity(self) -> np.ndarray:
        return np.asarray(self._model.options.gravity, dtype=np.float64).copy()

    def get_joint_range(self) -> np.ndarray | None:
        return None

    # ------------------------------------------------------------------ #
    # Simulation control                                                 #
    # ------------------------------------------------------------------ #

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        if self._pre_step_control_fn is not None:
            return self._step_with_pre_step_control(ctrl, nsteps)

        t0 = time.perf_counter()
        self._data.actuator_ctrls = np.ascontiguousarray(ctrl)
        set_ctrl_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if nsteps == 1:
            self._model.step(self._data)
        else:
            self._model.step_n(self._data, nsteps)
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_link_pose_cache()
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

        for _ in range(nsteps):
            t0 = time.perf_counter()
            native_ctrl = self._apply_pre_step_control(ctrl)
            self._data.actuator_ctrls = np.ascontiguousarray(native_ctrl)
            set_ctrl_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            self._model.step(self._data)
            physics_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            self._refresh_link_pose_cache()
            refresh_cache_ms += (time.perf_counter() - t0) * 1000.0

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
        qpos_motrix = self._mujoco_qpos_to_motrix(qpos)

        # Create mask for batch operation
        mask = np.zeros(self._num_envs, dtype=bool)
        mask[env_indices] = True
        data_slice = self._data[mask]

        # Batch set state
        data_slice.reset(self._model)
        self._clear_applied_body_forces(env_indices)
        self._apply_init_geom_size_overrides(data_slice, env_indices)
        self._apply_reset_randomization(data_slice, env_indices, randomization)
        data_slice.set_dof_vel(qvel)
        data_slice.set_dof_pos(qpos_motrix, self._model)

        if self._supports_position_actuator_gains:
            ctrl = qpos_motrix[:, self._joint_dof_pos_indices]
        else:
            ctrl = np.zeros((len(env_indices), self.num_actuators), dtype=self._np_dtype)
        data_slice.actuator_ctrls = np.ascontiguousarray(ctrl)

        self._model.forward_kinematic(data_slice)
        self._refresh_link_pose_cache(env_indices)

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        supported_reset_terms = {
            RESET_TERM_BASE_MASS,
            RESET_TERM_BASE_COM,
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_IPOS,
        }
        if getattr(self, "_supports_position_actuator_gains", False):
            supported_reset_terms.update({RESET_TERM_KP, RESET_TERM_KD})
        if getattr(self, "_supports_geom_friction_override", False):
            supported_reset_terms.add(RESET_TERM_GEOM_FRICTION)
        if getattr(self, "_supports_gravity_override", False):
            supported_reset_terms.add(RESET_TERM_GRAVITY)
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(supported_reset_terms),
            supports_interval_push=True,
            supports_interval_body_velocity_delta=False,
            supports_interval_body_force=getattr(self, "_supports_external_force", False),
        )

    def apply_init_randomization(self, plan: InitRandomizationPlan) -> None:
        if plan.is_empty():
            return
        model_assignments = np.asarray(plan.model_assignments, dtype=np.int32)
        if model_assignments.shape != (self._num_envs,):
            raise ValueError(
                f"model_assignments must have shape ({self._num_envs},), "
                f"got {model_assignments.shape}"
            )
        if np.any(model_assignments < 0) or np.any(model_assignments >= len(plan.model_variants)):
            raise ValueError(
                "model_assignments must refer to entries in InitRandomizationPlan.model_variants"
            )

        geom_size_overrides: dict[int, np.ndarray] = {}
        for variant_id, variant in enumerate(plan.model_variants):
            env_indices = np.flatnonzero(model_assignments == variant_id)
            if env_indices.size == 0:
                continue
            for override in variant.geom_size_overrides:
                geom_id = self.get_geom_id(override.geom_name)
                geom = _require_not_none(
                    self._model.get_geom(geom_id),
                    f"Geom '{override.geom_name}' not found in Motrix model",
                )
                override_shape = np.asarray(geom.get_size_override(self._data)).shape
                if len(override_shape) != 2:
                    raise ValueError(
                        f"Motrix geom '{override.geom_name}' size override must be rank-2, "
                        f"got shape {override_shape}"
                    )
                width = int(override_shape[1])
                size = np.asarray(override.size, dtype=np.float64).reshape(-1)
                if size.size < width:
                    raise ValueError(
                        f"GeomSizeOverride for '{override.geom_name}' has {size.size} values, "
                        f"but Motrix expects at least {width}"
                    )
                values = geom_size_overrides.setdefault(
                    geom_id,
                    np.asarray(geom.get_size_override(self._data), dtype=np.float64).copy(),
                )
                values[env_indices, :] = size[:width]

        self._init_geom_size_overrides = geom_size_overrides
        self._apply_init_geom_size_overrides(self._data, np.arange(self._num_envs, dtype=np.int32))

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
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

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=True,
            supports_native_video_capture=True,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        effective_mode = "interactive" if mode == "auto" else mode
        if effective_mode == "none":
            return BackendPlayRenderPlan(
                mode=effective_mode,
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if effective_mode == "interactive":
            return BackendPlayRenderPlan(
                mode=effective_mode,
                headless=False,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        assert effective_mode == "record"
        if play_steps is None:
            raise ValueError("Motrix record playback requires a finite training.play_steps value.")
        if output_video is None:
            raise ValueError("Motrix record playback requires an output video path.")
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
        del frame_state_getter, extra_data_getter
        should_record_video = (
            bool(record_video) if record_video is not None else output_video is not None
        )
        should_run_headless = bool(headless) if headless is not None else should_record_video
        try:
            return run_motrix_playback(
                backend=self,
                env=env,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                output_video=output_video,
                render_spacing=render_spacing,
                render_offset_mode=render_offset_mode,
                headless=should_run_headless,
                record_video=should_record_video,
                camera_kwargs=camera_kwargs,
            )
        except Exception as e:
            if (
                not should_run_headless
                and not should_record_video
                and "RenderClosedError" in type(e).__name__
            ):
                print("Render window closed.")
                return None
            raise

    # ------------------------------------------------------------------ #
    # Base kinematics                                                    #
    # ------------------------------------------------------------------ #

    def get_base_pos(self) -> np.ndarray:
        if self._body_floatingbase is not None:
            return self._body_floatingbase.get_translation(self._data)  # type: ignore[no-any-return]
        return self._body_link.get_pose(self._data)[:, :3]  # type: ignore[no-any-return]

    def get_base_quat(self) -> np.ndarray:
        if self._body_floatingbase is not None:
            quat = self._body_floatingbase.get_rotation(self._data)
        else:
            quat = self._body_link.get_rotation(self._data)
        return self._xyzw_to_wxyz(quat)

    def get_base_lin_vel(self) -> np.ndarray:
        if self._body_floatingbase is not None:
            return self._body_floatingbase.get_global_linear_velocity(self._data)  # type: ignore[no-any-return]
        return self._body_link.get_linear_velocity(self._data)  # type: ignore[no-any-return]

    def get_base_ang_vel(self) -> np.ndarray:
        if self._body_floatingbase is not None:
            return self._body_floatingbase.get_global_angular_velocity(self._data)  # type: ignore[no-any-return]
        return self._body_link.get_angular_velocity(self._data)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    # DOF state                                                          #
    # ------------------------------------------------------------------ #

    def get_dof_pos(self) -> np.ndarray:
        return self._body.get_joint_dof_pos(self._data)  # type: ignore[no-any-return]

    def get_dof_vel(self) -> np.ndarray:
        return self._body.get_joint_dof_vel(self._data)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                      #
    # ------------------------------------------------------------------ #

    def _as_body_ids(self, body_ids: np.ndarray) -> np.ndarray:
        return np.asarray(body_ids, dtype=np.int32)

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_poses_w(body_ids)[:, :, :3]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._xyzw_to_wxyz(self._get_link_poses_w(body_ids)[:, :, 3:])

    def get_body_pose_w_rows(
        self, env_ids: np.ndarray, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(env_ids, dtype=np.intp)
        poses_w = self._link_poses[rows[:, None], self._as_body_ids(body_ids), :]
        return poses_w[:, :, :3], self._xyzw_to_wxyz(poses_w[:, :, 3:])

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_lin_vel_w(body_ids)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_ang_vel_w(body_ids)

    def get_body_state_w(
        self, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        poses_w = self._get_link_poses_w(body_ids)
        return (
            poses_w[:, :, :3],
            self._xyzw_to_wxyz(poses_w[:, :, 3:]),
            self._get_link_lin_vel_w(body_ids),
            self._get_link_ang_vel_w(body_ids),
        )

    def copy_body_state_w(
        self,
        body_ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ids = self._as_body_ids(body_ids)
        poses_w = self._get_link_poses_w(ids)
        out_pos[..., 0] = poses_w[..., 0]
        out_pos[..., 1] = poses_w[..., 1]
        out_pos[..., 2] = poses_w[..., 2]
        out_quat[..., 0] = poses_w[..., 6]
        out_quat[..., 1] = poses_w[..., 3]
        out_quat[..., 2] = poses_w[..., 4]
        out_quat[..., 3] = poses_w[..., 5]

        link_velocity_cache = self._model.get_link_velocities(self._data)
        if self._link_velocity_cache is None or self._link_velocity_cache.shape != (
            self._num_envs,
            len(ids),
            6,
        ):
            self._link_velocity_cache = np.empty(
                (self._num_envs, len(ids), 6), dtype=self._np_dtype
            )
        np.take(link_velocity_cache, ids, axis=1, out=self._link_velocity_cache)
        out_lin_vel[..., 0] = self._link_velocity_cache[..., 0]
        out_lin_vel[..., 1] = self._link_velocity_cache[..., 1]
        out_lin_vel[..., 2] = self._link_velocity_cache[..., 2]
        out_ang_vel[..., 0] = self._link_velocity_cache[..., 3]
        out_ang_vel[..., 1] = self._link_velocity_cache[..., 4]
        out_ang_vel[..., 2] = self._link_velocity_cache[..., 5]
        return out_pos, out_quat, out_lin_vel, out_ang_vel

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                   #
    # ------------------------------------------------------------------ #

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_body_sensor_values(body_ids, "track_pos_b")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        # motrixsim framequat sensor 输出 xyzw，转换为接口约定的 wxyz
        return self._xyzw_to_wxyz(self._get_body_sensor_values(body_ids, "track_quat_b"))

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_body_sensor_values(body_ids, "track_linvel_b")

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_body_sensor_values(body_ids, "track_angvel_b")

    # ------------------------------------------------------------------ #
    # Sensors                                                            #
    # ------------------------------------------------------------------ #

    def get_sensor_data(self, name: str) -> np.ndarray:
        return self._model.get_sensor_value(name, self._data)  # type: ignore[no-any-return]

    def get_sensor_data_rows(self, name: str, env_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_ids, dtype=np.intp)
        mask = np.zeros(self._num_envs, dtype=bool)
        mask[rows] = True
        return self._model.get_sensor_value(name, self._data[mask])  # type: ignore[no-any-return]

    def get_sensor_data_batch(self, names: Sequence[str]) -> np.ndarray:
        sensor_names = tuple(names)
        if not sensor_names:
            return np.empty((self._num_envs, 0), dtype=self._np_dtype)
        values = self._model.get_sensor_values(sensor_names, self._data)
        return np.asarray(values, dtype=self._np_dtype)

    # ------------------------------------------------------------------ #
    # MotrixSim-specific                                                 #
    # ------------------------------------------------------------------ #

    def _get_body_names(self, body_ids: np.ndarray) -> list[str]:
        return [self._body_id_to_name[int(bid)] for bid in self._as_body_ids(body_ids)]

    def _get_link_poses_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.ascontiguousarray(self._link_poses[:, ids, :])

    def _get_link_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.stack(
            [self._link_cache[int(bid)].get_linear_velocity(self._data) for bid in ids],
            axis=1,
        )

    def _get_link_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.stack(
            [self._link_cache[int(bid)].get_angular_velocity(self._data) for bid in ids],
            axis=1,
        )

    def _get_body_sensor_values(self, body_ids: np.ndarray, prefix: str) -> np.ndarray:
        return np.stack(
            [
                self._model.get_sensor_value(f"{prefix}_{name}", self._data)
                for name in self._get_body_names(body_ids)
            ],
            axis=1,
        )

    def _xyzw_to_wxyz(self, q: np.ndarray) -> np.ndarray:
        """motrix xyzw → wxyz"""
        return q[..., [3, 0, 1, 2]]

    def _mujoco_qpos_to_motrix(self, qpos: np.ndarray) -> np.ndarray:
        """Convert every MuJoCo freejoint quaternion slice from wxyz to xyzw."""
        qpos_motrix = np.array(qpos, copy=True)
        for quat_indices in self._floating_base_quat_indices:
            qpos_motrix[..., quat_indices] = qpos[..., quat_indices[[1, 2, 3, 0]]]
        return qpos_motrix

    def _motrix_qpos_to_mujoco(self, qpos: np.ndarray) -> np.ndarray:
        """Convert every Motrix freejoint quaternion slice from xyzw to wxyz."""
        qpos_mujoco = np.array(qpos, copy=True)
        for quat_indices in self._floating_base_quat_indices:
            qpos_mujoco[..., quat_indices] = qpos[..., quat_indices[[3, 0, 1, 2]]]
        return qpos_mujoco

    def _refresh_link_pose_cache(self, env_indices: np.ndarray | None = None) -> None:
        if env_indices is None:
            self._link_poses = self._model.get_link_poses(self._data)
        else:
            mask = np.zeros(self._num_envs, dtype=bool)
            mask[env_indices] = True
            self._link_poses[env_indices] = self._model.get_link_poses(self._data[mask])

    def _coerce_reset_field(
        self,
        value: np.ndarray,
        *,
        name: str,
        num_reset: int,
        shaped_tail: tuple[int, ...],
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        shaped = (num_reset, *shaped_tail)
        flat_shape = (num_reset, int(np.prod(shaped_tail)))
        if arr.shape == shaped:
            return arr.copy()
        if arr.shape == flat_shape:
            return arr.reshape(shaped).copy()
        raise ValueError(f"{name} must have shape {shaped} or {flat_shape}, got {arr.shape}")

    def _apply_init_geom_size_overrides(self, data_slice, env_indices: np.ndarray) -> None:
        if not self._init_geom_size_overrides:
            return
        env_ids = np.asarray(env_indices, dtype=np.intp)
        for geom_id, values in self._init_geom_size_overrides.items():
            geom = _require_not_none(
                self._model.get_geom(int(geom_id)),
                f"Geom id {geom_id} not found in Motrix model",
            )
            geom.set_size_override(
                data_slice,
                np.ascontiguousarray(np.asarray(values[env_ids], dtype=np.float32)),
            )

    def _set_link_mass_overrides(self, data_slice, body_mass: np.ndarray) -> None:
        for link_id, link in self._links_by_id.items():
            link.set_mass_override(
                data_slice,
                np.ascontiguousarray(np.asarray(body_mass[:, link_id], dtype=np.float32)),
            )

    def _set_link_ipos_overrides(self, data_slice, body_ipos: np.ndarray) -> None:
        for link_id, link in self._links_by_id.items():
            link.set_center_of_mass_override(
                data_slice,
                np.ascontiguousarray(np.asarray(body_ipos[:, link_id, :], dtype=np.float32)),
            )

    def _set_geom_friction_overrides(self, data_slice, geom_friction: np.ndarray) -> None:
        if not self._supports_geom_friction_override:
            raise NotImplementedError("Motrix geom friction override is not available")
        override_ids = getattr(self, "_geom_friction_override_ids", tuple(self._geoms_by_id))
        unsupported_ids = sorted(set(self._geoms_by_id) - set(override_ids))
        if unsupported_ids:
            unsupported_values = geom_friction[:, unsupported_ids, :]
            default_values = self._default_geom_friction[None, unsupported_ids, :]
            if not np.allclose(unsupported_values, default_values):
                raise ValueError(
                    "Motrix geom friction override only supports collision geoms; "
                    f"non-collision geom ids were modified: {unsupported_ids}"
                )
        for geom_id in override_ids:
            geom = self._geoms_by_id[int(geom_id)]
            geom.set_friction_override(
                data_slice,
                np.ascontiguousarray(np.asarray(geom_friction[:, geom_id, :], dtype=np.float32)),
            )

    def _clear_applied_body_forces(self, env_indices: np.ndarray) -> None:
        if not self._applied_body_forces:
            return
        env_ids = np.asarray(env_indices, dtype=np.intp)
        for applied_force in self._applied_body_forces.values():
            applied_force[env_ids, :] = 0.0

    def push_robots(self, force_range):
        ex_force = np.random.rand(self.num_envs, 3) * 2 - 1  # [x_force, y_force, z_force]
        ex_force[:, 0] *= force_range[0]
        ex_force[:, 1] *= force_range[1]
        ex_force[:, 2] *= force_range[2]
        self._push_body_link.add_external_force(self._data, ex_force, local=True)

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
    ) -> None:
        """Apply absolute world-frame external forces through Motrix Link API."""
        if not getattr(self, "_supports_external_force", False):
            raise NotImplementedError("Motrix link external-force API is not available")
        body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        force_np = np.asarray(force, dtype=np.float64)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
        if force_np.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {force_np.shape}")
        for body_offset, body_id in enumerate(body_ids_np):
            link_id = int(body_id)
            link = self._links_by_id.get(link_id)
            if link is None:
                raise ValueError(f"Body id {link_id} not found in Motrix model")
            target_force = np.asarray(force_np[:, body_offset, :], dtype=np.float64)
            applied_force = self._applied_body_forces.setdefault(
                link_id,
                np.zeros((self._num_envs, 3), dtype=np.float64),
            )
            delta_force = target_force - applied_force
            if np.any(delta_force):
                link.add_external_force(
                    self._data,
                    np.ascontiguousarray(delta_force.astype(np.float32)),
                    local=False,
                )
                applied_force[:] = target_force

    def create_hfield_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> BackendHeightScanner:
        offsets_np = np.ascontiguousarray(np.asarray(offsets, dtype=np.float32))
        if offsets_np.ndim != 2 or offsets_np.shape[1] != 2:
            raise ValueError(f"offsets must have shape (num_points, 2), got {offsets_np.shape}")

        if alignment != "yaw":
            raise ValueError(f"MotrixBackend only supports alignment='yaw', got {alignment!r}")
        if output not in {"height", "clearance"}:
            raise ValueError(f"Unsupported hfield sampling output: {output!r}")

        geom_id = int(hfield_geom_id)
        if geom_id < 0 or geom_id >= int(self._model.num_geoms):
            raise ValueError(f"hfield_geom_id out of range: {geom_id}")

        body_id = int(frame_body_id)
        if body_id < 0 or body_id >= int(self._model.num_links):
            raise ValueError(f"frame_body_id out of range: {body_id}")

        terrain = self._model.get_geom(geom_id)
        if terrain is None:
            raise ValueError(f"Geom id {geom_id} not found in Motrix model")
        if not isinstance(terrain, mtx.GeomHField):
            raise ValueError(f"Geom id {geom_id} is not backed by a Motrix hfield")
        frame = self._link_cache[body_id]
        scanner = mtx.TerrainScanner(
            terrain,
            frame,
            offsets_np,
            alignment=alignment,
            output=output,
        )
        return _MotrixTerrainScanner(
            scanner=scanner,
            data=self._data,
            out=np.empty((self._num_envs, offsets_np.shape[0]), dtype=self._np_dtype),
        )

    def _update_tracking_camera_view(self) -> None:
        if (
            self._render_app is None
            or self._render_tracking_camera is None
            or self._render_offsets_np is None
        ):
            return
        lookat = tracking_camera_lookat(
            self.get_base_pos(),
            self._render_tracking_camera,
            self._render_offsets_np,
        )
        self._render_app.system_camera.set_view(
            lookat,
            self._render_tracking_camera.distance,
            self._render_tracking_camera.elevation,
            self._render_tracking_camera.azimuth,
        )

    def _assert_render_context_available(self, *, headless: bool, capture: bool) -> None:
        if self._render_app is None:
            return
        if self._render_headless != headless:
            raise RuntimeError(
                "Motrix renderer is already initialized with "
                f"headless={self._render_headless!r}; cannot reuse it with headless={headless!r}"
            )
        if capture and not self._render_capture_enabled:
            raise RuntimeError(
                "Motrix renderer is already initialized without video capture; "
                "cannot enable capture on the existing renderer"
            )
        return

    def init_renderer(
        self,
        spacing: float = 1.0,
        *,
        offset_mode: str = "grid",
        headless: bool = False,
        capture: bool = False,
        width: int = 1280,
        height: int = 720,
        camera_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a Motrix renderer, optionally enabling system-camera capture."""
        headless = bool(headless)
        capture = bool(capture)
        self._assert_render_context_available(headless=headless, capture=capture)
        if self._render_app is not None:
            return

        settings = RenderSettings.performance()
        settings.enable_shadow = True
        offsets = render_offsets(
            self._num_envs,
            float(spacing),
            offset_mode=str(offset_mode),
        )
        offsets_np = np.asarray(offsets, dtype=np.float64)
        self._render_offsets_np = offsets_np
        use_configured_camera = capture or camera_kwargs is not None
        if use_configured_camera:
            base_positions = (
                self.get_base_pos()
                if bool(dict(camera_kwargs or {}).get("cam_tracking", False))
                else None
            )
            camera_view = resolve_system_camera_view(
                self._num_envs,
                base_positions,
                offsets,
                camera_kwargs,
            )
            tracking_camera = camera_view.tracking
        else:
            tracking_camera = None
        if capture:
            self._model.cameras.set_system_render_target("image", int(width), int(height))
        render_app = RenderApp(headless=headless)
        render_app.launch(
            self._model,
            batch=self._num_envs,
            render_offset=offsets,
            render_settings=settings,
        )
        if use_configured_camera:
            render_app.system_camera.set_view(
                camera_view.lookat,
                camera_view.distance,
                camera_view.elevation,
                camera_view.azimuth,
            )
            if not capture:
                render_app.set_main_camera(None)
        self._render_app = render_app
        self._render_headless = headless
        self._render_capture_enabled = capture
        self._render_tracking_camera = tracking_camera

    def render(self):
        """Render current state (interactive visualization)"""
        if self._render_app is None:
            self.init_renderer()
        self._assert_render_context_available(headless=False, capture=False)
        assert self._render_app is not None
        self._update_tracking_camera_view()
        self._render_app.sync(data=self._data)

    def capture_video_frame(self) -> np.ndarray:
        """Capture one RGB frame from Motrix's system camera."""
        if self._render_app is None:
            self.init_renderer(headless=True, capture=True)
        if not self._render_capture_enabled:
            raise RuntimeError("Motrix renderer is not initialized for video capture")
        assert self._render_app is not None

        self._update_tracking_camera_view()
        task = self._render_app.system_camera.capture()
        self._render_app.sync(data=self._data, wait=True)
        image = task.take_image()
        if image is None:
            raise RuntimeError("Motrix system camera capture did not return an image")

        pixels = np.asarray(image.pixels)
        if pixels.ndim != 3:
            raise RuntimeError(
                f"Motrix system camera capture must return an HWC image, got shape {pixels.shape}"
            )
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]
        if pixels.shape[-1] != 3:
            raise RuntimeError(
                f"Motrix system camera capture must return RGB/RGBA pixels, got shape {pixels.shape}"
            )
        return np.ascontiguousarray(pixels, dtype=np.uint8)

    def _apply_reset_randomization(
        self,
        data_slice,
        env_indices: np.ndarray,
        randomization: ResetRandomizationPayload | None,
    ) -> None:
        if randomization is None or randomization.is_empty():
            return
        unsupported = (
            randomization.requested_terms() - self.get_dr_capabilities().supported_reset_terms
        )
        if unsupported:
            terms = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"{self.backend_type} backend does not support reset randomization terms: {terms}"
            )

        env_ids = np.asarray(env_indices, dtype=np.intp)
        num_reset = len(env_ids)
        body_mass = None
        if randomization.body_mass is not None:
            body_mass = self._coerce_reset_field(
                randomization.body_mass,
                name="body_mass",
                num_reset=num_reset,
                shaped_tail=(int(self._model.num_links),),
            )
        if randomization.base_mass_delta is not None:
            if body_mass is None:
                body_mass = np.broadcast_to(
                    self._default_body_mass,
                    (num_reset, int(self._model.num_links)),
                ).copy()
            body_mass[:, int(self._body_link.index)] += np.asarray(
                randomization.base_mass_delta, dtype=np.float32
            )
        if body_mass is not None:
            self._set_link_mass_overrides(data_slice, body_mass)

        body_ipos = None
        if randomization.body_ipos is not None:
            body_ipos = self._coerce_reset_field(
                randomization.body_ipos,
                name="body_ipos",
                num_reset=num_reset,
                shaped_tail=(int(self._model.num_links), 3),
            )
        if randomization.base_com_offset is not None:
            if body_ipos is None:
                body_ipos = np.broadcast_to(
                    self._default_body_ipos,
                    (num_reset, int(self._model.num_links), 3),
                ).copy()
            body_ipos[:, int(self._body_link.index), :] += np.asarray(
                randomization.base_com_offset, dtype=np.float32
            )
        if body_ipos is not None:
            self._set_link_ipos_overrides(data_slice, body_ipos)

        if randomization.geom_friction is not None:
            geom_friction = self._coerce_reset_field(
                randomization.geom_friction,
                name="geom_friction",
                num_reset=num_reset,
                shaped_tail=(int(self._model.num_geoms), 3),
            )
            self._set_geom_friction_overrides(data_slice, geom_friction)

        if randomization.gravity is not None:
            gravity = self._coerce_reset_field(
                randomization.gravity,
                name="gravity",
                num_reset=num_reset,
                shaped_tail=(3,),
            )
            self._set_gravity_override(data_slice, gravity)

        if randomization.kp is not None:
            kp = np.asarray(randomization.kp, dtype=np.float32)
            expected_shape = (num_reset, self.num_actuators)
            if kp.shape != expected_shape:
                raise ValueError(f"kp must have shape {expected_shape}, got {kp.shape}")
            self._set_position_actuator_kp_override(data_slice, kp)

        if randomization.kd is not None:
            kd = np.asarray(randomization.kd, dtype=np.float32)
            expected_shape = (num_reset, self.num_actuators)
            if kd.shape != expected_shape:
                raise ValueError(f"kd must have shape {expected_shape}, got {kd.shape}")
            self._set_position_actuator_kd_override(data_slice, kd)

    def _set_gravity_override(self, data_slice, gravity: np.ndarray) -> None:
        if not getattr(self, "_supports_gravity_override", False):
            raise NotImplementedError("Motrix gravity override is not available")
        self._model.set_gravity_override(
            data_slice,
            np.ascontiguousarray(np.asarray(gravity, dtype=np.float32)),
        )

    def _set_position_actuator_kp_override(self, data_slice, kp: np.ndarray) -> None:
        if not self._supports_position_actuator_gains:
            raise NotImplementedError(
                "Motrix actuator kp override is only available for all-position-actuator models"
            )
        for actuator in self._position_actuators:
            # TODO(motrixsim#1384): drop the copy once strided NumPy views are accepted.
            actuator.set_kp_override(data_slice, np.ascontiguousarray(kp[:, int(actuator.index)]))

    def _set_position_actuator_kd_override(self, data_slice, kd: np.ndarray) -> None:
        if not self._supports_position_actuator_gains:
            raise NotImplementedError(
                "Motrix actuator kd override is only available for all-position-actuator models"
            )
        for actuator in self._position_actuators:
            # TODO(motrixsim#1384): drop the copy once strided NumPy views are accepted.
            actuator.set_damping_override(
                data_slice,
                np.ascontiguousarray(kd[:, int(actuator.index)]),
            )

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._supports_position_actuator_gains:
            raise NotImplementedError(
                "Motrix actuator gains are only exposed for all-position-actuator models"
            )
        return self._default_actuator_kp.copy(), self._default_actuator_kd.copy()
