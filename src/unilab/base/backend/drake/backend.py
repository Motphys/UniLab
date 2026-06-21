"""UniLab adapter for the DrakeUni batch runtime.

UniLab owns task logic, reset sampling, named sensor views, and training flow.
DrakeUni owns Drake model construction, batched stepping, and raw sensor
evaluation. This module translates the ``SimBackend`` contract into DrakeUni
runtime calls and keeps UniLab's cached state/sensor views synchronized.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from multiprocessing import cpu_count
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    SimBackend,
    normalize_play_render_mode,
)
from unilab.base.backend.drake.playback import run_drake_playback
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)


# DrakeUni availability globals. These are cheap import-time probes so callers
# can ask whether Drake support exists without constructing a backend.
def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


DRAKE_AVAILABLE = _module_available("drakeuni")
DRAKE_IMPORT_ERROR: ImportError | None = None
DRAKE_BATCH_AVAILABLE = _module_available("drakeuni")
DRAKE_BATCH_IMPORT_ERROR: ImportError | None = None
DrakeBatchConfig = None
create_drake_runtime = None

_DRAKEUNI_SYMBOLS_LOADED = False


# Lazy import and pydrake guard helpers.
def _pydrake_loaded() -> bool:
    # DrakeUni's batch extension owns Drake symbol loading; mixing it with an
    # already-imported pydrake module has produced unstable process state.
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)


def _load_drakeuni_symbols() -> None:
    """Load DrakeUni only when a Drake backend is actually constructed."""

    global DRAKE_AVAILABLE
    global DRAKE_BATCH_AVAILABLE
    global DRAKE_BATCH_IMPORT_ERROR
    global DrakeBatchConfig
    global create_drake_runtime
    global _DRAKEUNI_SYMBOLS_LOADED

    if _DRAKEUNI_SYMBOLS_LOADED:
        return
    try:
        from drakeuni.runtime import DrakeBatchConfig as ImportedDrakeBatchConfig
        from drakeuni.runtime import batch_diagnostics
        from drakeuni.runtime import create_runtime as imported_create_runtime
    except ImportError as exc:  # pragma: no cover - optional local package.
        DRAKE_AVAILABLE = False
        DRAKE_BATCH_AVAILABLE = False
        DRAKE_BATCH_IMPORT_ERROR = exc
        raise ImportError("DrakeUni batch runtime is not installed.") from exc

    diagnostics = batch_diagnostics()
    if not diagnostics.batch_available:
        detail = diagnostics.batch_import_error
        import_error = ImportError(detail or "DrakeEnvPool batch extension has not been built.")
        DRAKE_AVAILABLE = False
        DRAKE_BATCH_AVAILABLE = False
        DRAKE_BATCH_IMPORT_ERROR = import_error
        raise ImportError("DrakeEnvPool batch extension has not been built.") from import_error

    DrakeBatchConfig = ImportedDrakeBatchConfig
    create_drake_runtime = imported_create_runtime
    DRAKE_AVAILABLE = True
    DRAKE_BATCH_AVAILABLE = True
    DRAKE_BATCH_IMPORT_ERROR = None
    _DRAKEUNI_SYMBOLS_LOADED = True


def ensure_drake_batch_available() -> tuple[bool, ImportError | None]:
    """Report whether the DrakeUni batch extension can be used."""

    try:
        _load_drakeuni_symbols()
    except ImportError as exc:
        return False, exc
    return True, None


# Floating-base compact state starts with xyz + quaternion in qpos and
# 3 linear + 3 angular components in qvel. UniLab usually wants only the
# actuated joint slices behind those root coordinates.
ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6


# Small helper types.
@dataclass(frozen=True)
class _DrakeUniModelView:
    """Read-only model-shape facade for UniLab's ``backend.model`` API.

    DrakeUni exposes model dimensions through ``model_info``. UniLab's backend
    contract expects a model-like object with dimension methods, so this facade
    carries those shape queries without exposing Drake internals.
    """

    nq: int
    nv: int
    nu: int

    def num_positions(self) -> int:
        return self.nq

    def num_velocities(self) -> int:
        return self.nv

    def num_actuators(self) -> int:
        return self.nu


# Path and thread helpers.
def _resolve_batch_nthread(num_envs: int, requested: int) -> int:
    """Resolve a worker count without creating idle workers above num_envs."""

    env_count = max(1, int(num_envs))
    requested_count = int(requested)
    if requested_count > 0:
        return min(env_count, requested_count)
    return min(env_count, max(1, cpu_count() * 2))


def _resolve_scene_path(scene: SceneCfg) -> Path:
    """Convert UniLab's scene pointer into an absolute model path."""

    if not scene.model_file:
        raise ValueError("DrakeBackend requires SceneCfg.model_file")
    path = Path(scene.model_file)
    return path if path.is_absolute() else Path.cwd() / path


class DrakeBackend(SimBackend):
    """UniLab ``SimBackend`` implementation backed by DrakeUni batch runtime.

    The backend keeps the public UniLab API stable while delegating model
    construction, integration, and raw sensor evaluation to DrakeUni.
    """

    backend_type = "drake"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        drake_backend_mode: str = "batch",
        nthread: int = 0,
    ) -> None:
        # Validate the backend mode at construction so Hydra/config mistakes
        # fail at the backend boundary.
        mode = str(drake_backend_mode or "batch").strip().lower()
        if mode != "batch":
            raise ValueError(
                "UniLab DrakeBackend requires drake_backend_mode='batch'. "
                f"Got {drake_backend_mode!r}."
            )
        if _pydrake_loaded():
            raise ImportError(
                "Drake batch backend cannot be loaded after pydrake has already "
                "been imported in this process. Start a fresh process before "
                "constructing DrakeBackend."
            )
        if int(num_envs) < 1:
            raise ValueError(f"DrakeUni batch backend requires num_envs >= 1, got {num_envs}")
        _load_drakeuni_symbols()
        if DrakeBatchConfig is None or create_drake_runtime is None:
            detail = DRAKE_BATCH_IMPORT_ERROR
            message = "DrakeUni runtime is not available."
            if detail is not None:
                message = f"{message} Import error: {detail}"
            raise ImportError(message) from detail

        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._scene_model_file = str(_resolve_scene_path(scene))
        # DrakeUni receives only generic batch facts. Task concepts such as
        # base bodies, push targets, and observation semantics stay in UniLab.
        config = DrakeBatchConfig(
            model_file=self._scene_model_file,
            num_envs=self._num_envs,
            sim_dt=self._sim_dt,
            nthread=int(nthread),
        )
        self._runtime = create_drake_runtime(config)
        model_info = self._runtime.model_info()
        # Cache static model metadata once and expose copies through the
        # UniLab backend contract.
        self._home_qpos_mujoco = model_info.home_qpos.copy()
        self._home_qvel_mujoco = model_info.home_qvel.copy()
        self._ctrl_limits = model_info.ctrl_limits.copy()
        self._joint_ranges = model_info.joint_ranges.copy()
        self._actuator_stiffness = model_info.actuator_stiffness.copy()
        self._actuator_damping = model_info.actuator_damping.copy()
        self._actuator_qpos_adr = model_info.actuator_qpos_adr.astype(np.intp, copy=True)
        self._actuator_qvel_adr = model_info.actuator_qvel_adr.astype(np.intp, copy=True)
        self._sensor_names = tuple(model_info.sensor_names)
        self._sensor_adr = model_info.sensor_adr.copy()
        self._sensor_dim = model_info.sensor_dim.copy()
        self._site_name_to_id = {
            str(name): index for index, name in enumerate(getattr(model_info, "site_names", ()))
        }
        self._joint_qpos_adr_by_name = {
            str(name): int(adr)
            for name, adr in zip(
                getattr(model_info, "joint_names", ()),
                getattr(model_info, "joint_qpos_adr", ()),
                strict=True,
            )
        }
        self._joint_qvel_adr_by_name = {
            str(name): int(adr)
            for name, adr in zip(
                getattr(model_info, "joint_names", ()),
                getattr(model_info, "joint_qvel_adr", ()),
                strict=True,
            )
        }
        self._joint_dims_by_name = {
            str(name): (int(qpos_dim), int(qvel_dim))
            for name, qpos_dim, qvel_dim in zip(
                getattr(model_info, "joint_names", ()),
                getattr(model_info, "joint_qpos_dim", ()),
                getattr(model_info, "joint_qvel_dim", ()),
                strict=True,
            )
        }
        self._root_qpos_dim = int(np.min(self._actuator_qpos_adr)) if self._actuator_qpos_adr.size else 0
        self._root_qvel_dim = int(np.min(self._actuator_qvel_adr)) if self._actuator_qvel_adr.size else 0
        self._num_bodies = int(model_info.num_bodies)
        self._pending_body_forces = np.zeros((self._num_envs, self._num_bodies, 3), dtype=np.float64)
        self._model = _DrakeUniModelView(
            nq=int(model_info.nq),
            nv=int(model_info.nv),
            nu=int(model_info.nu),
        )
        self._nthread = int(getattr(self._runtime, "nthread", int(nthread)))
        # Runtime state and raw sensor views are refreshed after reset/step.
        self._physics_state = self._runtime.physics_state()
        self._sensor_data = np.zeros(
            (self._num_envs, int(model_info.nsensordata)),
            dtype=np.float64,
        )
        self._sensor_views: dict[str, np.ndarray] = {}
        self._sync_runtime_state()

    # Static model contract.
    #
    # These accessors expose stable dimensions, limits, and reset defaults from
    # the cached model metadata.
    @property
    def scene_model_file(self) -> str:
        return self._scene_model_file

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def nthread(self) -> int:
        return self._nthread

    @property
    def model(self) -> _DrakeUniModelView:
        return self._model

    @property
    def num_actuators(self) -> int:
        return self._model.nu

    @property
    def num_dof_vel(self) -> int:
        return int(self._actuator_qvel_adr.size)

    # Return copies for arrays that UniLab may clamp, concatenate, or normalize.
    # The backend cache should not be mutated by task-side code.
    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._ctrl_limits.copy()

    def get_joint_range(self) -> np.ndarray | None:
        return self._joint_ranges.copy()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if name == "home":
            return self._home_qpos_mujoco.copy()
        return self._runtime.keyframe_qpos(str(name))

    def get_default_qpos(self) -> np.ndarray:
        return self._home_qpos_mujoco.copy()

    def get_init_qvel(self) -> np.ndarray:
        return self._home_qvel_mujoco.copy()

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._actuator_stiffness.copy(), self._actuator_damping.copy())

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        # Body IDs are owned by DrakeUni because they depend on the materialized
        # Drake model, not on UniLab's scene pointer.
        return self._runtime.body_ids(tuple(str(name) for name in names))

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        ids: list[int] = []
        for name in names:
            key = str(name)
            try:
                ids.append(self._site_name_to_id[key])
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain MJCF site {key!r}") from exc
        return np.asarray(ids, dtype=np.int32)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            key = str(name)
            self._require_single_dof_joint(key)
            try:
                indices.append(self._joint_qvel_adr_by_name[key])
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain joint {key!r}") from exc
        return np.asarray(indices, dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            key = str(name)
            self._require_single_dof_joint(key)
            try:
                indices.append(self._joint_qpos_adr_by_name[key] - self._root_qpos_dim)
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain joint {key!r}") from exc
        return np.asarray(indices, dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            key = str(name)
            self._require_single_dof_joint(key)
            try:
                indices.append(self._joint_qvel_adr_by_name[key] - self._root_qvel_dim)
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain joint {key!r}") from exc
        return np.asarray(indices, dtype=np.int32)

    # Stepping and reset.
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        # UniLab passes one actuator command per env. An optional pre-step hook
        # can convert policy actions into backend-native position targets.
        step_count = int(nsteps)
        if step_count < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (self._num_envs, self.num_actuators):
            raise ValueError(
                "DrakeUni batch backend step expected ctrl shape "
                f"({self._num_envs}, {self.num_actuators}), got {values.shape}"
            )
        start = time.perf_counter()
        try:
            if self._pre_step_control_fn is None:
                output = self._runtime.step(values, step_count, self._pending_body_forces_or_none())
                self._sync_runtime_state(output)
            else:
                output = None
                for _ in range(step_count):
                    native_ctrl = self._apply_pre_step_control(values)
                    output = self._runtime.step(native_ctrl, 1, self._pending_body_forces_or_none())
                    self._sync_runtime_state(output)
        finally:
            self._pending_body_forces.fill(0.0)
        timing = dict(output.get("timing", {}))
        timing.setdefault("step_ms", (time.perf_counter() - start) * 1000.0)
        return {"timing": timing}

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        # Reset is the handoff from UniLab's sampled state tensors into
        # DrakeUni's per-env runtime contexts.
        if randomization is not None and not randomization.is_empty():
            raise NotImplementedError("DrakeUni batch backend does not apply reset randomization yet")
        indices = np.asarray(env_indices, dtype=np.int32)
        qpos_rows = np.asarray(qpos, dtype=np.float64)
        qvel_rows = np.asarray(qvel, dtype=np.float64)
        if indices.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= self._num_envs):
            raise IndexError(
                f"env_indices must be in [0, {self._num_envs - 1}], got {indices.tolist()}"
            )
        if qpos_rows.shape != (indices.size, self._model.nq):
            raise ValueError(f"qpos must have shape ({indices.size}, {self._model.nq})")
        if qvel_rows.shape != (indices.size, self._model.nv):
            raise ValueError(f"qvel must have shape ({indices.size}, {self._model.nv})")
        self._runtime.reset(indices, qpos_rows, qvel_rows)
        self._sync_runtime_state()

    # Playback and domain randomization.
    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        # Unsupported randomization knobs fail explicitly instead of silently
        # becoming no-ops.
        return DomainRandomizationCapabilities(supports_interval_body_force=True)

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        self._pending_body_forces.fill(0.0)
        if plan.push_perturbation_limit is not None:
            raise NotImplementedError(
                "DrakeBackend interval pushes require explicit body_ids and body_force"
            )
        if plan.body_force is not None:
            if plan.body_ids is None:
                raise ValueError("Interval body-force perturbation requires body_ids")
            self.apply_body_force(plan.body_ids, plan.body_force)
        if plan.body_linear_velocity_delta is not None:
            raise NotImplementedError(
                "DrakeUni batch backend does not support interval body velocity perturbation"
            )

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        # Drake advances playback physics, while the shared playback helper
        # handles recording. There is no native interactive Drake viewer path.
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=False,
            supports_physics_state_playback=True,
            supports_native_video_capture=False,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        if mode in {"none", "auto"}:
            return BackendPlayRenderPlan(
                mode=mode,
                headless=True,
                record_video=False,
                num_steps=play_steps,
                output_video=None,
            )
        if mode == "interactive":
            raise NotImplementedError("DrakeUni batch backend does not support interactive rendering")
        if play_steps is None:
            raise ValueError("DrakeUni record playback requires a finite play_steps value.")
        if output_video is None:
            raise ValueError("DrakeUni record playback requires an output video path.")
        return BackendPlayRenderPlan(
            mode="record",
            headless=True,
            record_video=True,
            num_steps=int(play_steps),
            output_video=output_video,
        )

    def run_playback(
        self,
        *,
        env: Any,
        initialize: Callable[[], Any],
        step: Callable[[Any], Any],
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Callable[[], np.ndarray] | None = None,
        camera_kwargs: dict[str, Any] | None = None,
        extra_data_getter: Callable[[], np.ndarray | None] | None = None,
    ) -> str | None:
        # Playback keeps Drake as the physics backend. The helper owns rendering
        # and video capture so training code can use one playback contract.
        return run_drake_playback(
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            headless=bool(headless),
            record_video=bool(record_video),
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera_kwargs,
            extra_data_getter=extra_data_getter,
        )

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
        del spacing, offset_mode, headless, capture, width, height, camera_kwargs
        raise NotImplementedError("DrakeUni batch backend records through run_playback")

    def render(self) -> None:
        raise NotImplementedError("DrakeUni batch backend does not support interactive rendering")

    def capture_video_frame(self) -> np.ndarray:
        raise NotImplementedError("DrakeUni batch backend records through run_playback")

    # Runtime state getters.
    #
    # ``physics_state`` is DrakeUni's compact per-env packet used by playback
    # and debugging. Sensor-specific getters below expose named slices/packets.
    def get_physics_state(self) -> np.ndarray:
        return self._physics_state.copy()

    def get_playback_model(self, env_index: int | None = None) -> str:
        if env_index is not None:
            idx = int(env_index)
            if idx < 0 or idx >= self._num_envs:
                raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        return self._scene_model_file

    def diagnostics(self) -> Any:
        return self._runtime.diagnostics()

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
    ) -> None:
        ids = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        values = np.asarray(force, dtype=np.float64)
        expected_shape = (self._num_envs, ids.size, 3)
        if values.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {values.shape}")
        for offset, body_id in enumerate(ids):
            if body_id < 0 or body_id >= self._num_bodies:
                raise IndexError(f"body id {int(body_id)} is outside [0, {self._num_bodies - 1}]")
            self._pending_body_forces[:, int(body_id), :] += values[:, offset, :]

    # Sensor access.
    #
    # DrakeUni returns one flat sensor array; this class owns the MuJoCo-style
    # named views over that array.
    def get_base_pos(self) -> np.ndarray:
        self._require_floating_root()
        return self._physics_state[:, 1:4].copy()

    def get_base_quat(self) -> np.ndarray:
        self._require_floating_root()
        return self._physics_state[:, 4:8].copy()

    def get_base_lin_vel(self) -> np.ndarray:
        qvel_start = 1 + self._model.nq
        return self._physics_state[:, qvel_start : qvel_start + 3].copy()

    def get_base_ang_vel(self) -> np.ndarray:
        qvel_start = 1 + self._model.nq
        return self._physics_state[:, qvel_start + 3 : qvel_start + 6].copy()

    def get_dof_pos(self) -> np.ndarray:
        return self._physics_state[:, 1 + self._actuator_qpos_adr].copy()

    def get_dof_vel(self) -> np.ndarray:
        qvel_start = 1 + self._model.nq
        return self._physics_state[:, qvel_start + self._actuator_qvel_adr].copy()

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_state(body_ids)["pos"]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_state(body_ids)["quat"]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_state(body_ids)["linvel"]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_state(body_ids)["angvel"]

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        body_state = self._body_state(body_ids)
        base_pos = self.get_base_pos()
        base_rot = _quat_to_rotation_matrix(self.get_base_quat())
        delta = body_state["pos"] - base_pos[:, None, :]
        return np.einsum("nij,nkj->nki", np.swapaxes(base_rot, 1, 2), delta)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        body_quat = self._body_state(body_ids)["quat"]
        base_inv = _quat_conjugate(self.get_base_quat())
        return _quat_multiply(base_inv[:, None, :], body_quat)

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        base_rot = _quat_to_rotation_matrix(self.get_base_quat())
        return np.einsum(
            "nij,nkj->nki",
            np.swapaxes(base_rot, 1, 2),
            self._body_state(body_ids)["linvel"],
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        base_rot = _quat_to_rotation_matrix(self.get_base_quat())
        return np.einsum(
            "nij,nkj->nki",
            np.swapaxes(base_rot, 1, 2),
            self._body_state(body_ids)["angvel"],
        )

    def get_sensor_data(self, name: str) -> np.ndarray:
        if name in self._sensor_views:
            return self._sensor_views[name].copy()
        raise KeyError(f"Unknown DrakeUni sensor: {name}")

    # Internal helpers.
    def _sync_runtime_state(self, output: dict[str, Any] | None = None) -> None:
        # Keep UniLab's cached state/sensor views aligned after every DrakeUni update.
        if output is None:
            self._physics_state = self._runtime.physics_state()
            sensor_data = self._runtime.sensor_data()
        else:
            self._physics_state = np.asarray(output["state"], dtype=np.float64).copy()
            sensor_data = output["sensor_data"]
        self._sensor_data = np.asarray(sensor_data, dtype=np.float64).copy()
        self._rebuild_sensor_views()

    def _rebuild_sensor_views(self) -> None:
        self._sensor_views = {}
        for index, name in enumerate(self._sensor_names):
            adr = int(self._sensor_adr[index])
            dim = int(self._sensor_dim[index])
            self._sensor_views[name] = self._sensor_data[:, adr : adr + dim]

    def _body_state(self, body_ids: np.ndarray) -> dict[str, np.ndarray]:
        ids = np.asarray(body_ids, dtype=np.int32)
        if ids.ndim != 1:
            raise ValueError(f"body_ids must be one-dimensional, got {ids.shape}")
        return self._runtime.compute_body_state(ids)

    def _pending_body_forces_or_none(self) -> np.ndarray | None:
        if np.any(self._pending_body_forces):
            return self._pending_body_forces
        return None

    def _require_single_dof_joint(self, name: str) -> None:
        dims = self._joint_dims_by_name.get(name)
        if dims is None:
            raise ValueError(f"Drake model does not contain joint {name!r}")
        if dims != (1, 1):
            raise ValueError(f"Drake joint {name!r} is not a single-DoF joint")

    def _require_floating_root(self) -> None:
        if self._model.nq < ROOT_QPOS_DIM or self._model.nv < ROOT_QVEL_DIM:
            raise NotImplementedError(
                "DrakeBackend root-state helpers require a floating-root compact state"
            )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64).copy()
    values[..., 1:] *= -1.0
    return values


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    q = np.divide(values, np.maximum(norm, 1.0e-12))
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


__all__ = [
    "DRAKE_AVAILABLE",
    "DRAKE_IMPORT_ERROR",
    "DRAKE_BATCH_AVAILABLE",
    "DRAKE_BATCH_IMPORT_ERROR",
    "DrakeBackend",
    "_resolve_batch_nthread",
    "ensure_drake_batch_available",
]
