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


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


DRAKE_AVAILABLE = _module_available("drakeuni")
DRAKE_IMPORT_ERROR: ImportError | None = None
DRAKE_BATCH_AVAILABLE = _module_available("drakeuni")
DRAKE_BATCH_IMPORT_ERROR: ImportError | None = None
DrakeRuntimeConfig = None
create_drake_runtime = None

_DRAKEUNI_SYMBOLS_LOADED = False


def _pydrake_loaded() -> bool:
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)


def _load_drakeuni_symbols() -> None:
    global DRAKE_AVAILABLE
    global DRAKE_BATCH_AVAILABLE
    global DRAKE_BATCH_IMPORT_ERROR
    global DrakeRuntimeConfig
    global create_drake_runtime
    global _DRAKEUNI_SYMBOLS_LOADED

    if _DRAKEUNI_SYMBOLS_LOADED:
        return
    try:
        from drakeuni.runtime import DrakeRuntimeConfig as ImportedDrakeRuntimeConfig
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

    DrakeRuntimeConfig = ImportedDrakeRuntimeConfig
    create_drake_runtime = imported_create_runtime
    DRAKE_AVAILABLE = True
    DRAKE_BATCH_AVAILABLE = True
    DRAKE_BATCH_IMPORT_ERROR = None
    _DRAKEUNI_SYMBOLS_LOADED = True


def ensure_drake_batch_available() -> tuple[bool, ImportError | None]:
    try:
        _load_drakeuni_symbols()
    except ImportError as exc:
        return False, exc
    return True, None


ROOT_QVEL_DIM = 6


@dataclass(frozen=True)
class _DrakeUniModelView:
    nq: int
    nv: int
    nu: int

    def num_positions(self) -> int:
        return self.nq

    def num_velocities(self) -> int:
        return self.nv

    def num_actuators(self) -> int:
        return self.nu


def _resolve_batch_nthread(num_envs: int, requested: int) -> int:
    env_count = max(1, int(num_envs))
    requested_count = int(requested)
    if requested_count > 0:
        return min(env_count, requested_count)
    return min(env_count, max(1, cpu_count() * 2))


def _resolve_scene_path(scene: SceneCfg) -> Path:
    if not scene.model_file:
        raise ValueError("DrakeBackend requires SceneCfg.model_file")
    path = Path(scene.model_file)
    return path if path.is_absolute() else Path.cwd() / path


class DrakeBackend(SimBackend):
    """Public UniLab Drake adapter backed by the DrakeUni batch runtime."""

    backend_type = "drake"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        drake_backend_mode: str = "batch",
        nthread: int = 0,
        **kwargs: Any,
    ) -> None:
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
        self._impl: SimBackend = _DrakeUniBatchBackend(
            scene,
            num_envs,
            sim_dt,
            nthread=nthread,
            **kwargs,
        )
        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None

    def diagnostics(self) -> Any:
        diagnostics = getattr(self._impl, "diagnostics", None)
        if diagnostics is None:
            return {"mode": "batch", "available": False}
        return diagnostics()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)

    @property
    def scene_model_file(self) -> str:
        return self._impl.scene_model_file

    @property
    def num_envs(self) -> int:
        return self._impl.num_envs

    @property
    def nthread(self) -> int:
        return int(getattr(self._impl, "nthread", 0))

    @property
    def model(self):
        return self._impl.model

    @property
    def num_actuators(self) -> int:
        return self._impl.num_actuators

    @property
    def num_dof_vel(self) -> int:
        return self._impl.num_dof_vel

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._impl.get_actuator_ctrl_range()

    def get_joint_range(self) -> np.ndarray | None:
        return self._impl.get_joint_range()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        return self._impl.get_keyframe_qpos(name)

    def get_default_qpos(self) -> np.ndarray:
        return self._impl.get_default_qpos()

    def get_init_qvel(self) -> np.ndarray:
        return self._impl.get_init_qvel()

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return self._impl.get_actuator_gains()

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self._impl.get_body_ids(names)

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        return self._impl.step(ctrl, nsteps)

    def set_pre_step_control(self, fn: Callable[[Any, np.ndarray], np.ndarray] | None) -> None:
        self._pre_step_control_fn = fn
        self._impl.set_pre_step_control(fn)

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        self._impl.set_state(env_indices, qpos, qvel, randomization)

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return self._impl.get_dr_capabilities()

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        self._impl.apply_interval_randomization(plan)

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return self._impl.get_play_capabilities()

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        return self._impl.resolve_play_render_plan(
            play_render_mode=play_render_mode,
            play_steps=play_steps,
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
        return self._impl.run_playback(
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            headless=headless,
            record_video=record_video,
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
        self._impl.init_renderer(
            spacing=spacing,
            offset_mode=offset_mode,
            headless=headless,
            capture=capture,
            width=width,
            height=height,
            camera_kwargs=camera_kwargs,
        )

    def render(self) -> None:
        self._impl.render()

    def capture_video_frame(self) -> np.ndarray:
        return self._impl.capture_video_frame()

    def get_physics_state(self) -> np.ndarray:
        return self._impl.get_physics_state()

    def get_playback_model(self, env_index: int | None = None) -> Any:
        return self._impl.get_playback_model(env_index)

    def cleanup_scene_assets(self) -> None:
        self._impl.cleanup_scene_assets()

    def get_base_pos(self) -> np.ndarray:
        return self._impl.get_base_pos()

    def get_base_quat(self) -> np.ndarray:
        return self._impl.get_base_quat()

    def get_base_lin_vel(self) -> np.ndarray:
        return self._impl.get_base_lin_vel()

    def get_base_ang_vel(self) -> np.ndarray:
        return self._impl.get_base_ang_vel()

    def get_dof_pos(self) -> np.ndarray:
        return self._impl.get_dof_pos()

    def get_dof_vel(self) -> np.ndarray:
        return self._impl.get_dof_vel()

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._impl.get_body_pos_w(body_ids)

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._impl.get_body_quat_w(body_ids)

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._impl.get_body_lin_vel_w(body_ids)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._impl.get_body_ang_vel_w(body_ids)

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._impl.get_body_pos_b(body_ids)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._impl.get_body_quat_b(body_ids)

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._impl.get_body_lin_vel_b(body_ids)

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._impl.get_body_ang_vel_b(body_ids)

    def get_sensor_data(self, name: str) -> np.ndarray:
        return self._impl.get_sensor_data(name)


class _DrakeUniBatchBackend(SimBackend):
    """DrakeUni-backed UniLab contract adapter."""

    backend_type = "drake"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        position_actuator_gains: dict[str, float] | None = None,
        nthread: int = 0,
        robot_profile: str | None = None,
        **_: Any,
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError(f"DrakeUni batch backend requires num_envs >= 1, got {num_envs}")
        if not robot_profile:
            raise ValueError("DrakeUni batch backend requires a robot_profile from the task.")
        if not base_name:
            raise ValueError("DrakeUni batch backend requires a base_name from the task.")
        _load_drakeuni_symbols()
        if DrakeRuntimeConfig is None or create_drake_runtime is None:
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
        self._kp = float((position_actuator_gains or {}).get("kp", 35.0))
        self._kd = float((position_actuator_gains or {}).get("kd", 0.5))
        self._base_name = str(base_name)
        self._push_body_name = str(push_body_name or base_name)
        self._robot_profile = str(robot_profile)
        self._pending_push_force = np.zeros((self._num_envs, 3), dtype=np.float64)

        config = DrakeRuntimeConfig(
            model_file=self._scene_model_file,
            num_envs=self._num_envs,
            sim_dt=self._sim_dt,
            mode="batch",
            base_name=self._base_name,
            push_body_name=self._push_body_name,
            kp=self._kp,
            kd=self._kd,
            nthread=int(nthread),
            robot_profile=self._robot_profile,
        )
        self._runtime = create_drake_runtime(config)
        model_info = self._runtime.model_info()
        self._home_qpos_mujoco = model_info.home_qpos.copy()
        self._home_qvel_mujoco = model_info.home_qvel.copy()
        self._ctrl_limits = model_info.ctrl_limits.copy()
        self._joint_ranges = model_info.joint_ranges.copy()
        self._sensor_names = tuple(model_info.sensor_names)
        self._base_body_id = int(self._runtime.body_ids([self._base_name])[0])
        self._model = _DrakeUniModelView(
            nq=int(model_info.nq),
            nv=int(model_info.nv),
            nu=int(model_info.nu),
        )
        self._nthread = int(getattr(self._runtime, "nthread", int(nthread)))
        self._physics_state = self._runtime.physics_state()
        self._sensor_packet: dict[str, np.ndarray] = {}
        self._sync_runtime_state()

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
        return self._model.nv - ROOT_QVEL_DIM

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._ctrl_limits.copy()

    def get_joint_range(self) -> np.ndarray | None:
        return self._joint_ranges.copy()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if name != "home":
            raise KeyError(f"DrakeUni batch backend only exposes keyframe 'home', got {name!r}")
        return self._home_qpos_mujoco.copy()

    def get_default_qpos(self) -> np.ndarray:
        return self._home_qpos_mujoco.copy()

    def get_init_qvel(self) -> np.ndarray:
        return self._home_qvel_mujoco.copy()

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.num_actuators, self._kp, dtype=np.float64),
            np.full(self.num_actuators, self._kd, dtype=np.float64),
        )

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self._runtime.body_ids(tuple(str(name) for name in names))

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (self._num_envs, self.num_actuators):
            raise ValueError(
                "DrakeUni batch backend step expected ctrl shape "
                f"({self._num_envs}, {self.num_actuators}), got {values.shape}"
            )
        values = self._apply_pre_step_control(values)
        start = time.perf_counter()
        output = self._runtime.step(values, int(nsteps), self._pending_push_force)
        self._pending_push_force.fill(0.0)
        self._sync_runtime_state(output)
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

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(supports_interval_push=True)

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        self._pending_push_force.fill(0.0)
        if plan.push_perturbation_limit is not None:
            self._pending_push_force[:] = self._sample_push_force(plan.push_perturbation_limit)
        if plan.body_force is not None or plan.body_linear_velocity_delta is not None:
            raise NotImplementedError(
                "DrakeUni batch backend currently supports push_perturbation_limit only"
            )

    def get_play_capabilities(self) -> BackendPlayCapabilities:
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

    def get_base_pos(self) -> np.ndarray:
        return self._sensor_packet["base_pos"].copy()

    def get_base_quat(self) -> np.ndarray:
        return self._sensor_packet["base_quat"].copy()

    def get_base_lin_vel(self) -> np.ndarray:
        qvel_start = 1 + self._model.nq
        return self._physics_state[:, qvel_start : qvel_start + 3].copy()

    def get_base_ang_vel(self) -> np.ndarray:
        qvel_start = 1 + self._model.nq
        return self._physics_state[:, qvel_start + 3 : qvel_start + 6].copy()

    def get_dof_pos(self) -> np.ndarray:
        return self._sensor_packet["dof_pos"].copy()

    def get_dof_vel(self) -> np.ndarray:
        return self._sensor_packet["dof_vel"].copy()

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_base_body_only(body_ids)
        return np.repeat(self.get_base_pos()[:, None, :], len(body_ids), axis=1)

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_base_body_only(body_ids)
        return np.repeat(self.get_base_quat()[:, None, :], len(body_ids), axis=1)

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_base_body_only(body_ids)
        return np.repeat(self.get_base_lin_vel()[:, None, :], len(body_ids), axis=1)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_base_body_only(body_ids)
        return np.repeat(self.get_base_ang_vel()[:, None, :], len(body_ids), axis=1)

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_base_body_only(body_ids)
        return np.zeros((self._num_envs, len(body_ids), 3), dtype=np.float64)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_base_body_only(body_ids)
        quat = np.zeros((self._num_envs, len(body_ids), 4), dtype=np.float64)
        quat[:, :, 0] = 1.0
        return quat

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_base_body_only(body_ids)
        return np.repeat(self._sensor_packet["local_linvel"][:, None, :], len(body_ids), axis=1)

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_base_body_only(body_ids)
        return np.repeat(self._sensor_packet["gyro"][:, None, :], len(body_ids), axis=1)

    def get_sensor_data(self, name: str) -> np.ndarray:
        if name in self._sensor_packet:
            return self._sensor_packet[name].copy()
        raise KeyError(f"Unknown DrakeUni sensor: {name}")

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

    def _sync_runtime_state(self, output: dict[str, Any] | None = None) -> None:
        if output is None:
            self._physics_state = self._runtime.physics_state()
            packet = {name: self._runtime.sensor(name) for name in self._sensor_names}
        else:
            self._physics_state = np.asarray(output["state"], dtype=np.float64).copy()
            raw_sensor = output.get("sensor", {})
            packet = {
                key: np.asarray(value, dtype=np.float64).copy() for key, value in raw_sensor.items()
            }
        packet.setdefault("position", packet["base_pos"])
        self._sensor_packet = packet

    def _require_base_body_only(self, body_ids: np.ndarray) -> None:
        ids = np.asarray(body_ids, dtype=np.int32)
        if ids.ndim != 1:
            raise ValueError(f"body_ids must be one-dimensional, got {ids.shape}")
        if np.any(ids != self._base_body_id):
            raise NotImplementedError(
                "DrakeUni batch backend only exposes configured base body kinematics "
                "in this milestone"
            )

    def _sample_push_force(self, force_range: Sequence[float] | np.ndarray) -> np.ndarray:
        limit = np.asarray(force_range, dtype=np.float64)
        if limit.shape != (3,):
            raise ValueError(f"DrakeUni push force range must have shape (3,), got {limit.shape}")
        direction = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3))
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.divide(direction, np.maximum(norm, 1.0e-12))
        magnitude = np.random.uniform(0.0, 1.0, size=(self._num_envs, 1))
        return direction * magnitude * limit.reshape(1, 3)


__all__ = [
    "DRAKE_AVAILABLE",
    "DRAKE_IMPORT_ERROR",
    "DRAKE_BATCH_AVAILABLE",
    "DRAKE_BATCH_IMPORT_ERROR",
    "DrakeBackend",
    "_resolve_batch_nthread",
    "ensure_drake_batch_available",
]
