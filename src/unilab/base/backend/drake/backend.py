from __future__ import annotations

import logging
import sys
import time
import xml.etree.ElementTree as ET
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
from unilab.base.backend.drake.pool import DrakeEnvPool, DrakePoolOutput, SensorPacket
from unilab.base.backend.mujoco.playback import run_mujoco_playback
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

DRAKE_AVAILABLE = find_spec("pydrake") is not None
DRAKE_IMPORT_ERROR: ImportError | None = None
DRAKE_BATCH_AVAILABLE = find_spec("drakeuni") is not None
DRAKE_BATCH_IMPORT_ERROR: ImportError | None = None
DrakeRuntimeConfig = None
create_drake_runtime = None

_DRAKEUNI_SYMBOLS_LOADED = False
_PYDRAKE_SYMBOLS_LOADED = False

AddMultibodyPlantSceneGraph = None
BodyIndex = None
CameraInfo = None
CollisionFilterDeclaration = None
ClippingRange = None
ContactModel = None
DepthRange = None
DepthRenderCamera = None
DiagramBuilder = None
DiscreteContactApproximation = None
ExternallyAppliedSpatialForce = None
GeometrySet = None
JointActuatorIndex = None
LightParameter = None
MakeRenderEngineVtk = None
MeshcatVisualizer = None
Parser = None
PdControllerGains = None
RenderCameraCore = None
RenderEngineVtkParams = None
RgbdSensor = None
RigidTransform = None
RotationMatrix = None
SpatialForce = None
StartMeshcat = None
Simulator = None


def _pydrake_loaded() -> bool:
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)


def _load_drakeuni_symbols() -> None:
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
        DRAKE_BATCH_AVAILABLE = False
        DRAKE_BATCH_IMPORT_ERROR = exc
        raise ImportError("DrakeUni batch runtime is not installed.") from exc

    diagnostics = batch_diagnostics()
    if not diagnostics.batch_available:
        detail = diagnostics.batch_import_error
        import_error = ImportError(detail or "DrakeEnvPool batch extension has not been built.")
        DRAKE_BATCH_AVAILABLE = False
        DRAKE_BATCH_IMPORT_ERROR = import_error
        raise ImportError("DrakeEnvPool batch extension has not been built.") from import_error

    DrakeRuntimeConfig = ImportedDrakeRuntimeConfig
    create_drake_runtime = imported_create_runtime
    DRAKE_BATCH_AVAILABLE = True
    DRAKE_BATCH_IMPORT_ERROR = None
    _DRAKEUNI_SYMBOLS_LOADED = True


def ensure_drake_batch_available() -> tuple[bool, ImportError | None]:
    try:
        _load_drakeuni_symbols()
    except ImportError as exc:
        return False, exc
    return True, None


def _load_pydrake_symbols() -> None:
    global AddMultibodyPlantSceneGraph
    global BodyIndex
    global CameraInfo
    global CollisionFilterDeclaration
    global ClippingRange
    global ContactModel
    global DRAKE_AVAILABLE
    global DRAKE_IMPORT_ERROR
    global DepthRange
    global DepthRenderCamera
    global DiagramBuilder
    global DiscreteContactApproximation
    global ExternallyAppliedSpatialForce
    global GeometrySet
    global JointActuatorIndex
    global LightParameter
    global MakeRenderEngineVtk
    global MeshcatVisualizer
    global Parser
    global PdControllerGains
    global RenderCameraCore
    global RenderEngineVtkParams
    global RgbdSensor
    global RigidTransform
    global RotationMatrix
    global Simulator
    global SpatialForce
    global StartMeshcat
    global _PYDRAKE_SYMBOLS_LOADED

    if _PYDRAKE_SYMBOLS_LOADED:
        return
    try:  # pragma: no cover - exercised by integration smoke tests when Drake is installed.
        from pydrake.all import (
            AddMultibodyPlantSceneGraph,
            CameraInfo,
            ClippingRange,
            DepthRange,
            DepthRenderCamera,
            DiagramBuilder,
            ExternallyAppliedSpatialForce,
            JointActuatorIndex,
            LightParameter,
            MakeRenderEngineVtk,
            MeshcatVisualizer,
            Parser,
            RenderCameraCore,
            RenderEngineVtkParams,
            RgbdSensor,
            RigidTransform,
            RotationMatrix,
            Simulator,
            StartMeshcat,
        )
        from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
        from pydrake.multibody.math import SpatialForce
        from pydrake.multibody.plant import ContactModel, DiscreteContactApproximation
        from pydrake.multibody.tree import BodyIndex, PdControllerGains
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        DRAKE_AVAILABLE = False
        DRAKE_IMPORT_ERROR = exc
        raise ImportError("Drake backend requested, but pydrake is not installed.") from exc

    DRAKE_AVAILABLE = True
    DRAKE_IMPORT_ERROR = None
    _PYDRAKE_SYMBOLS_LOADED = True


LOGGER = logging.getLogger(__name__)
ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6
BASE_SENSOR_NAMES = frozenset(
    {
        "gyro",
        "local_linvel",
        "global_linvel",
        "global_angvel",
        "position",
        "upvector",
    }
)
GO1_FOOT_SENSOR_NAMES = ("FL_pos", "FR_pos", "RL_pos", "RR_pos")
GO1_FOOT_CONTACT_SENSOR_NAMES = (
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
)


@dataclass(frozen=True)
class DrakeModelMetadata:
    """UniLab replay metadata parsed from the Drake model files."""

    name: str
    ctrl_limits: np.ndarray
    torque_limits: np.ndarray
    joint_ranges: np.ndarray
    foot_sensor_to_body: dict[str, str]
    foot_sensor_offsets: dict[str, np.ndarray]
    contact_sensors: frozenset[str]


@dataclass(frozen=True)
class _DrakeRuntime:
    context: Any
    plant_context: Any
    simulator: Any


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


def _require_drake() -> None:
    _load_pydrake_symbols()


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


def _load_xml_roots(scene_path: Path) -> list[tuple[Path, ET.Element]]:
    roots: list[tuple[Path, ET.Element]] = []
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        root = ET.parse(resolved).getroot()
        roots.append((resolved, root))
        for include in root.findall(".//include"):
            include_file = include.attrib.get("file")
            if include_file:
                visit(resolved.parent / include_file)

    visit(scene_path)
    return roots


def _parse_vector(text: str | None, *, expected: int | None = None) -> np.ndarray | None:
    if text is None:
        return None
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if expected is not None and values.shape != (expected,):
        return None
    return values if values.size else None


def _required_pair(text: str | None, description: str) -> np.ndarray:
    values = _parse_vector(text, expected=2)
    if values is None:
        raise ValueError(f"Expected two values for {description}, got {text!r}")
    return values


def _collect_default_classes(
    roots: Sequence[tuple[Path, ET.Element]],
) -> dict[str, dict[str, dict[str, str]]]:
    defaults: dict[str, dict[str, dict[str, str]]] = {}

    def walk_default(
        node: ET.Element,
        inherited: dict[str, dict[str, str]],
    ) -> None:
        current = {tag: dict(attrs) for tag, attrs in inherited.items()}
        for child in node:
            if child.tag in {"joint", "position"}:
                current.setdefault(child.tag, {}).update(child.attrib)

        class_name = node.attrib.get("class")
        if class_name:
            defaults[class_name] = {tag: dict(attrs) for tag, attrs in current.items()}

        for child in node:
            if child.tag == "default":
                walk_default(child, current)

    for _, root in roots:
        for default_node in root.findall("./default"):
            walk_default(default_node, {})
    return defaults


def _merged_default_attrs(
    defaults: dict[str, dict[str, dict[str, str]]],
    class_name: str | None,
    tag: str,
    attrs: dict[str, str],
) -> dict[str, str]:
    merged = dict(defaults.get(class_name or "", {}).get(tag, {}))
    merged.update(attrs)
    return merged


def _extract_actuator_metadata(
    roots: Sequence[tuple[Path, ET.Element]],
    defaults: dict[str, dict[str, dict[str, str]]],
    joint_ranges_by_name: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctrl_limits: list[np.ndarray] = []
    torque_limits: list[float] = []
    joint_ranges: list[np.ndarray] = []

    for _, root in roots:
        for actuator in root.findall(".//actuator/position"):
            attrs = _merged_default_attrs(
                defaults,
                actuator.attrib.get("class"),
                "position",
                actuator.attrib,
            )
            actuator_name = actuator.attrib.get("name", "<unnamed>")
            ctrl_range = _required_pair(attrs.get("ctrlrange"), f"{actuator_name} ctrlrange")
            force_range = _required_pair(attrs.get("forcerange"), f"{actuator_name} forcerange")
            ctrl_limits.append(ctrl_range)
            torque_limits.append(float(np.max(np.abs(force_range))))

            joint_name = actuator.attrib.get("joint")
            if joint_name and joint_name in joint_ranges_by_name:
                joint_ranges.append(joint_ranges_by_name[joint_name])
            else:
                joint_ranges.append(ctrl_range)

    if not ctrl_limits:
        raise ValueError("DrakeBackend requires position actuators with ctrlrange metadata")
    return (
        np.asarray(ctrl_limits, dtype=np.float64),
        np.asarray(torque_limits, dtype=np.float64),
        np.asarray(joint_ranges, dtype=np.float64),
    )


def _extract_joint_ranges(
    roots: Sequence[tuple[Path, ET.Element]],
    defaults: dict[str, dict[str, dict[str, str]]],
) -> dict[str, np.ndarray]:
    ranges: dict[str, np.ndarray] = {}
    for _, root in roots:
        for joint in root.findall(".//worldbody//joint"):
            name = joint.attrib.get("name")
            if not name:
                continue
            attrs = _merged_default_attrs(
                defaults,
                joint.attrib.get("class"),
                "joint",
                joint.attrib,
            )
            joint_range = _parse_vector(attrs.get("range"), expected=2)
            if joint_range is not None:
                ranges[name] = joint_range
    return ranges


def _extract_sites(
    roots: Sequence[tuple[Path, ET.Element]],
) -> dict[str, tuple[str, np.ndarray]]:
    sites: dict[str, tuple[str, np.ndarray]] = {}

    def walk_body(body: ET.Element) -> None:
        body_name = body.attrib.get("name")
        if body_name:
            for site in body.findall("./site"):
                site_name = site.attrib.get("name")
                site_pos = _parse_vector(site.attrib.get("pos"), expected=3)
                if site_name and site_pos is not None:
                    sites[site_name] = (body_name, site_pos)
        for child in body.findall("./body"):
            walk_body(child)

    for _, root in roots:
        for body in root.findall("./worldbody/body"):
            walk_body(body)
    return sites


def _extract_sensor_metadata(
    roots: Sequence[tuple[Path, ET.Element]],
    sites: dict[str, tuple[str, np.ndarray]],
) -> tuple[dict[str, str], dict[str, np.ndarray], frozenset[str]]:
    foot_sensor_to_body: dict[str, str] = {}
    foot_sensor_offsets: dict[str, np.ndarray] = {}
    contact_sensors: set[str] = set()

    for _, root in roots:
        for sensor in root.findall(".//sensor/framepos"):
            name = sensor.attrib.get("name")
            obj_name = sensor.attrib.get("objname")
            if (
                not name
                or not obj_name
                or name in BASE_SENSOR_NAMES
                or sensor.attrib.get("objtype") != "site"
                or obj_name not in sites
            ):
                continue
            body_name, site_offset = sites[obj_name]
            foot_sensor_to_body[name] = body_name
            foot_sensor_offsets[name] = site_offset

        for sensor in root.findall(".//sensor/contact"):
            name = sensor.attrib.get("name")
            if name:
                contact_sensors.add(name)

    return foot_sensor_to_body, foot_sensor_offsets, frozenset(contact_sensors)


def _load_model_metadata(scene_path: Path) -> DrakeModelMetadata:
    roots = _load_xml_roots(scene_path)
    defaults = _collect_default_classes(roots)
    joint_ranges_by_name = _extract_joint_ranges(roots, defaults)
    ctrl_limits, torque_limits, joint_ranges = _extract_actuator_metadata(
        roots,
        defaults,
        joint_ranges_by_name,
    )
    foot_sensor_to_body, foot_sensor_offsets, contact_sensors = _extract_sensor_metadata(
        roots,
        _extract_sites(roots),
    )
    return DrakeModelMetadata(
        name=roots[0][1].attrib.get("model", scene_path.stem),
        ctrl_limits=ctrl_limits,
        torque_limits=torque_limits,
        joint_ranges=joint_ranges,
        foot_sensor_to_body=foot_sensor_to_body,
        foot_sensor_offsets=foot_sensor_offsets,
        contact_sensors=contact_sensors,
    )


def _read_keyframe_qpos(scene_path: Path, name: str) -> np.ndarray | None:
    try:
        roots = _load_xml_roots(scene_path)
    except ET.ParseError:
        return None
    for _, root in roots:
        for key in root.findall(".//key"):
            if key.attrib.get("name") != name:
                continue
            values = _parse_vector(key.attrib.get("qpos"))
            if values is not None:
                return values
    return None


def _mujoco_qpos_to_drake(qpos: np.ndarray) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float64)
    return np.concatenate([values[3:7], values[0:3], values[7:]])


def _drake_qpos_to_mujoco(qpos: np.ndarray) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float64)
    return np.concatenate([values[4:7], values[0:4], values[7:]])


def _mujoco_qvel_to_drake(qvel: np.ndarray) -> np.ndarray:
    values = np.asarray(qvel, dtype=np.float64)
    return np.concatenate([values[3:6], values[0:3], values[6:]])


def _drake_qvel_to_mujoco(qvel: np.ndarray) -> np.ndarray:
    values = np.asarray(qvel, dtype=np.float64)
    return np.concatenate([values[3:6], values[0:3], values[6:]])


def _quat_from_rotation(rotation: Any) -> np.ndarray:
    quat = rotation.ToQuaternion()
    return np.array([quat.w(), quat.x(), quat.y(), quat.z()], dtype=np.float64)


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * scale,
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array(
            [
                (m[2, 1] - m[1, 2]) / scale,
                0.25 * scale,
                (m[0, 1] + m[1, 0]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    if m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array(
            [
                (m[0, 2] - m[2, 0]) / scale,
                (m[0, 1] + m[1, 0]) / scale,
                0.25 * scale,
                (m[1, 2] + m[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array(
        [
            (m[1, 0] - m[0, 1]) / scale,
            (m[0, 2] + m[2, 0]) / scale,
            (m[1, 2] + m[2, 1]) / scale,
            0.25 * scale,
        ],
        dtype=np.float64,
    )


class DrakeBackend(SimBackend):
    """Public UniLab Drake adapter over private pydrake and DrakeUni runtimes."""

    backend_type = "drake"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        drake_backend_mode: str = "pydrake",
        nthread: int = 0,
        **kwargs: Any,
    ) -> None:
        mode = str(drake_backend_mode or "pydrake").strip().lower()
        if mode in {"batch", "drakeuni"}:
            if _pydrake_loaded():
                raise ImportError(
                    "Drake batch backend cannot be loaded after pydrake has already "
                    "been imported in this process. Start a fresh process for "
                    "drake_backend_mode='batch', or select drake_backend_mode='pydrake'."
                )
            self._impl: SimBackend = _DrakeUniBatchBackend(
                scene,
                num_envs,
                sim_dt,
                nthread=nthread,
                **kwargs,
            )
        elif mode in {"pydrake", "python"}:
            self._impl = _PydrakeDrakeBackend(scene, num_envs, sim_dt, **kwargs)
        else:
            raise ValueError(
                "drake_backend_mode must be one of pydrake, python, batch, "
                f"drakeuni; got {drake_backend_mode!r}"
            )
        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None

    def diagnostics(self) -> Any:
        diagnostics = getattr(self._impl, "diagnostics", None)
        if diagnostics is None:
            return {"mode": "pydrake", "available": bool(DRAKE_AVAILABLE)}
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


class _PydrakeDrakeBackend(SimBackend):
    """Minimal Drake backend for the replay milestone.

    The contract intentionally mirrors MuJoCo-shaped qpos/qvel at the UniLab
    boundary while storing Drake's quaternion-first floating-base order inside
    the plant. Robot-specific replay metadata is parsed from the model files.
    """

    backend_type = "drake"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str = "trunk",
        push_body_name: str | None = None,
        position_actuator_gains: dict[str, float] | None = None,
        **_: Any,
    ) -> None:
        _require_drake()
        if int(num_envs) < 1:
            raise ValueError(f"DrakeBackend requires num_envs >= 1, got {num_envs}")

        logging.getLogger("drake").setLevel(logging.ERROR)
        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._scene_model_file = str(_resolve_scene_path(scene))
        self._model_metadata = _load_model_metadata(Path(self._scene_model_file))
        self._home_qpos_mujoco = self._resolve_home_qpos(Path(self._scene_model_file))
        self._home_qvel_mujoco = np.empty(0, dtype=np.float64)
        self._kp = float((position_actuator_gains or {}).get("kp", 35.0))
        self._kd = float((position_actuator_gains or {}).get("kd", 0.5))
        self._base_name = base_name
        self._push_body_name = push_body_name or base_name
        self._meshcat = None
        self._meshcat_url: str | None = None
        self._rgbd_sensor = None
        self._rgbd_width = 0
        self._rgbd_height = 0
        self._runtimes: list[_DrakeRuntime] = []
        self._physics_state = np.empty((0, 0), dtype=np.float64)
        self._sensor_packet: SensorPacket = {}
        self._pool: DrakeEnvPool | None = None
        self._target_realtime_rate = 0.0
        self._pending_push_force = np.zeros((self._num_envs, 3), dtype=np.float64)

        self._build_drake_system(meshcat=None, camera_pose=None)
        self._pool = self._make_pool()
        LOGGER.info(
            "Initialized Drake %s backend from %s; filtered %d robot collision geometries",
            self._model_metadata.name,
            self._scene_model_file,
            self._num_filtered_geometries,
        )

    def _resolve_home_qpos(self, scene_path: Path) -> np.ndarray:
        qpos = _read_keyframe_qpos(scene_path, "home")
        if qpos is None:
            raise ValueError(f"DrakeBackend requires keyframe 'home' in {scene_path}")
        return qpos.copy()

    def _build_drake_system(
        self,
        *,
        meshcat: Any | None,
        camera_pose: Any | None,
        camera_width: int = 640,
        camera_height: int = 360,
        qpos_mujoco: np.ndarray | None = None,
        qvel_mujoco: np.ndarray | None = None,
    ) -> None:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=self._sim_dt)
        target_realtime_rate = 1.0 if meshcat is not None else 0.0
        self._rgbd_sensor = None
        self._rgbd_width = 0
        self._rgbd_height = 0
        if camera_pose is not None:
            render_params = RenderEngineVtkParams()
            render_params.lights = [
                LightParameter(
                    type="directional",
                    frame="camera",
                    direction=np.array([0.0, 0.0, 1.0]),
                    intensity=3.0,
                )
            ]
            scene_graph.AddRenderer("drake_renderer", MakeRenderEngineVtk(render_params))
        plant.set_contact_model(ContactModel.kPointContactOnly)
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
        model_instances = Parser(plant).AddModels(self._scene_model_file)
        if len(model_instances) != 1:
            raise RuntimeError(
                f"DrakeBackend expected one model instance from {self._scene_model_file}, "
                f"got {len(model_instances)}"
            )
        self._model_instance = model_instances[0]

        for i, effort_limit in enumerate(self._model_metadata.torque_limits):
            actuator = plant.get_joint_actuator(JointActuatorIndex(i))
            actuator.set_effort_limit(float(effort_limit))
            actuator.set_controller_gains(PdControllerGains(p=self._kp, d=self._kd))

        self._num_filtered_geometries = self._exclude_robot_self_collisions(
            plant, scene_graph, self._model_instance
        )
        plant.Finalize()
        if meshcat is not None:
            MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        if camera_pose is not None:
            intrinsics = CameraInfo(camera_width, camera_height, np.deg2rad(55.0))
            camera_core = RenderCameraCore(
                "drake_renderer",
                intrinsics,
                ClippingRange(0.05, 100.0),
                RigidTransform(),
            )
            depth_camera = DepthRenderCamera(camera_core, DepthRange(0.1, 50.0))
            rgbd_sensor = builder.AddSystem(
                RgbdSensor(scene_graph.world_frame_id(), camera_pose, depth_camera, False)
            )
            builder.Connect(
                scene_graph.get_query_output_port(),
                rgbd_sensor.query_object_input_port(),
            )
            self._rgbd_sensor = rgbd_sensor
            self._rgbd_width = int(camera_width)
            self._rgbd_height = int(camera_height)

        self._plant = plant
        self._scene_graph = scene_graph
        self._trunk = self._plant.GetBodyByName(self._base_name, self._model_instance)
        self._push_body = self._plant.GetBodyByName(self._push_body_name, self._model_instance)
        self._diagram = builder.Build()
        self._target_realtime_rate = target_realtime_rate

        if self._home_qpos_mujoco.shape != (self._plant.num_positions(),):
            raise RuntimeError(
                f"DrakeBackend home keyframe has shape {self._home_qpos_mujoco.shape}, "
                f"but plant nq={self._plant.num_positions()}"
            )
        if self._model_metadata.ctrl_limits.shape != (self._plant.num_actuators(), 2):
            raise RuntimeError(
                f"DrakeBackend parsed ctrl limits with shape "
                f"{self._model_metadata.ctrl_limits.shape}, but plant nu={self._plant.num_actuators()}"
            )
        if self._model_metadata.torque_limits.shape != (self._plant.num_actuators(),):
            raise RuntimeError(
                f"DrakeBackend parsed torque limits with shape "
                f"{self._model_metadata.torque_limits.shape}, but plant nu={self._plant.num_actuators()}"
            )
        if self._model_metadata.joint_ranges.shape != (self._plant.num_actuators(), 2):
            raise RuntimeError(
                f"DrakeBackend parsed joint ranges with shape "
                f"{self._model_metadata.joint_ranges.shape}, but plant nu={self._plant.num_actuators()}"
            )
        if self._home_qvel_mujoco.shape != (self._plant.num_velocities(),):
            self._home_qvel_mujoco = np.zeros(self._plant.num_velocities(), dtype=np.float64)

        reset_qpos, reset_qvel = self._runtime_state_batches(qpos_mujoco, qvel_mujoco)
        self._runtimes = [
            self._make_runtime(
                reset_qpos[env_index],
                reset_qvel[env_index],
                target_realtime_rate=self._target_realtime_rate,
            )
            for env_index in range(self._num_envs)
        ]
        self._bind_primary_runtime()
        self._diagram.ForcedPublish(self._context)
        self._refresh_cached_outputs_from_live_contexts()

    def _make_pool(self) -> DrakeEnvPool:
        sensor_shapes = {key: value.shape for key, value in self._sensor_packet.items()}
        return DrakeEnvPool(
            nbatch=self._num_envs,
            state_dim=self._state_dim,
            control_dim=self.num_actuators,
            sensor_shapes=sensor_shapes,
            step_impl=self._pool_step_impl,
            reset_impl=self._pool_reset_impl,
        )

    def _runtime_state_batches(
        self,
        qpos_mujoco: np.ndarray | None,
        qvel_mujoco: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos = self._home_qpos_mujoco if qpos_mujoco is None else np.asarray(qpos_mujoco)
        qvel = self._home_qvel_mujoco if qvel_mujoco is None else np.asarray(qvel_mujoco)
        if qpos.ndim == 1:
            qpos = np.broadcast_to(qpos, (self._num_envs, qpos.shape[0])).copy()
        if qvel.ndim == 1:
            qvel = np.broadcast_to(qvel, (self._num_envs, qvel.shape[0])).copy()
        expected_qpos = (self._num_envs, self._plant.num_positions())
        expected_qvel = (self._num_envs, self._plant.num_velocities())
        if qpos.shape != expected_qpos:
            raise ValueError(f"qpos_mujoco must have shape {expected_qpos}, got {qpos.shape}")
        if qvel.shape != expected_qvel:
            raise ValueError(f"qvel_mujoco must have shape {expected_qvel}, got {qvel.shape}")
        return qpos.astype(np.float64, copy=False), qvel.astype(np.float64, copy=False)

    def _make_runtime(
        self,
        qpos_mujoco: np.ndarray,
        qvel_mujoco: np.ndarray,
        *,
        target_realtime_rate: float,
    ) -> _DrakeRuntime:
        context = self._diagram.CreateDefaultContext()
        plant_context = self._plant.GetMyMutableContextFromRoot(context)
        self._plant.SetPositions(plant_context, _mujoco_qpos_to_drake(qpos_mujoco))
        self._plant.SetVelocities(plant_context, _mujoco_qvel_to_drake(qvel_mujoco))
        self._plant.get_actuation_input_port(self._model_instance).FixValue(
            plant_context, np.zeros(self.num_actuators, dtype=np.float64)
        )
        self._set_pd_target(qpos_mujoco[ROOT_QPOS_DIM:], plant_context=plant_context)
        simulator = Simulator(self._diagram, context)
        simulator.set_target_realtime_rate(float(target_realtime_rate))
        simulator.Initialize()
        return _DrakeRuntime(context=context, plant_context=plant_context, simulator=simulator)

    def _bind_primary_runtime(self) -> None:
        runtime = self._runtimes[0]
        self._context = runtime.context
        self._plant_context = runtime.plant_context
        self._simulator = runtime.simulator

    @property
    def scene_model_file(self) -> str:
        return self._scene_model_file

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> Any:
        return self._plant

    @property
    def num_actuators(self) -> int:
        return int(self._plant.num_actuators())

    @property
    def num_dof_vel(self) -> int:
        return int(self._plant.num_velocities() - ROOT_QVEL_DIM)

    @property
    def _state_dim(self) -> int:
        return int(1 + self._plant.num_positions() + self._plant.num_velocities())

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._model_metadata.ctrl_limits.copy()

    def get_joint_range(self) -> np.ndarray | None:
        return self._model_metadata.joint_ranges.copy()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if name != "home":
            raise KeyError(f"DrakeBackend only exposes keyframe 'home', got {name!r}")
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
        body_ids: list[int] = []
        for name in names:
            try:
                body = self._plant.GetBodyByName(str(name), self._model_instance)
            except RuntimeError as exc:
                raise ValueError(f"Unknown Drake body: {name}") from exc
            body_ids.append(int(body.index()))
        return np.asarray(body_ids, dtype=np.int32)

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (self._num_envs, self.num_actuators):
            raise ValueError(
                f"DrakeBackend.step expected ctrl shape ({self._num_envs}, {self.num_actuators}), "
                f"got {values.shape}"
            )
        values = self._apply_pre_step_control(values)
        if self._pool is None:
            raise RuntimeError("DrakeEnvPool is not initialized")
        output = self._pool.step(
            self._physics_state,
            nstep=int(nsteps),
            control=values,
            push_force=self._pending_push_force,
            return_sensor=True,
        )
        self._pending_push_force.fill(0.0)
        self._apply_pool_output(output)
        self._bind_primary_runtime()
        return {"timing": dict(output.timing)}

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        indices = np.asarray(env_indices, dtype=np.int32)
        if randomization is not None and not randomization.is_empty():
            raise NotImplementedError("DrakeBackend does not apply reset randomization yet")

        qpos_rows = np.asarray(qpos, dtype=np.float64)
        qvel_rows = np.asarray(qvel, dtype=np.float64)
        if indices.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= self._num_envs):
            raise IndexError(
                f"env_indices must be in [0, {self._num_envs - 1}], got {indices.tolist()}"
            )
        if qpos_rows.shape != (indices.size, self._plant.num_positions()):
            raise ValueError(
                f"qpos must have shape ({indices.size}, {self._plant.num_positions()}), "
                f"got {qpos_rows.shape}"
            )
        if qvel_rows.shape != (indices.size, self._plant.num_velocities()):
            raise ValueError(
                f"qvel must have shape ({indices.size}, {self._plant.num_velocities()}), "
                f"got {qvel_rows.shape}"
            )

        if self._pool is None:
            raise RuntimeError("DrakeEnvPool is not initialized")
        initial_state = self._pack_state_rows(qpos_rows, qvel_rows)
        output = self._pool.reset(indices, initial_state)
        self._apply_pool_output(output)
        self._bind_primary_runtime()

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
                "DrakeBackend currently supports Go1 push_robots interval randomization only"
            )

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=True,
            supports_physics_state_playback=True,
            supports_native_video_capture=True,
        )

    def get_base_pos(self) -> np.ndarray:
        return self._sensor_packet["base_pos"].copy()

    def get_base_quat(self) -> np.ndarray:
        return self._physics_state[:, 1 + 3 : 1 + 7].copy()

    def get_base_lin_vel(self) -> np.ndarray:
        qvel_start = 1 + self._plant.num_positions()
        return self._physics_state[:, qvel_start : qvel_start + 3].copy()

    def get_base_ang_vel(self) -> np.ndarray:
        qvel_start = 1 + self._plant.num_positions()
        return self._physics_state[:, qvel_start + 3 : qvel_start + 6].copy()

    def get_dof_pos(self) -> np.ndarray:
        return self._sensor_packet["dof_pos"].copy()

    def get_dof_vel(self) -> np.ndarray:
        return self._sensor_packet["dof_vel"].copy()

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        bodies = [self._plant.get_body(BodyIndex(int(body_id))) for body_id in body_ids]
        return np.asarray(
            [
                [self._body_pos(body, env_index) for body in bodies]
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        ).reshape(self._num_envs, len(bodies), 3)

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        bodies = [self._plant.get_body(BodyIndex(int(body_id))) for body_id in body_ids]
        return np.asarray(
            [
                [self._body_quat(body, env_index) for body in bodies]
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        ).reshape(self._num_envs, len(bodies), 4)

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        bodies = [self._plant.get_body(BodyIndex(int(body_id))) for body_id in body_ids]
        return np.asarray(
            [
                [self._body_spatial_velocity(body, env_index).translational() for body in bodies]
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        ).reshape(self._num_envs, len(bodies), 3)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        bodies = [self._plant.get_body(BodyIndex(int(body_id))) for body_id in body_ids]
        return np.asarray(
            [
                [self._body_spatial_velocity(body, env_index).rotational() for body in bodies]
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        ).reshape(self._num_envs, len(bodies), 3)

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        pos_w = self.get_body_pos_w(body_ids)
        values = []
        for env_index in range(self._num_envs):
            x_wb = self._body_pose(self._trunk, env_index)
            rotation_bw = x_wb.rotation().matrix().T
            base_pos = np.asarray(x_wb.translation(), dtype=np.float64)
            values.append((pos_w[env_index] - base_pos) @ rotation_bw.T)
        return np.asarray(values, dtype=np.float64).reshape(self._num_envs, len(body_ids), 3)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        values = []
        for env_index in range(self._num_envs):
            x_wb = self._body_pose(self._trunk, env_index)
            r_bw = x_wb.rotation().matrix().T
            row = []
            for body_id in body_ids:
                body = self._plant.get_body(BodyIndex(int(body_id)))
                r_wi = self._body_pose(body, env_index).rotation().matrix()
                row.append(_quat_from_matrix(r_bw @ r_wi))
            values.append(row)
        return np.asarray(values, dtype=np.float64).reshape(self._num_envs, len(body_ids), 4)

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        vel_w = self.get_body_lin_vel_w(body_ids)
        values = []
        for env_index in range(self._num_envs):
            rotation_bw = self._body_pose(self._trunk, env_index).rotation().matrix().T
            values.append(vel_w[env_index] @ rotation_bw.T)
        return np.asarray(values, dtype=np.float64).reshape(self._num_envs, len(body_ids), 3)

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        vel_w = self.get_body_ang_vel_w(body_ids)
        values = []
        for env_index in range(self._num_envs):
            rotation_bw = self._body_pose(self._trunk, env_index).rotation().matrix().T
            values.append(vel_w[env_index] @ rotation_bw.T)
        return np.asarray(values, dtype=np.float64).reshape(self._num_envs, len(body_ids), 3)

    def get_sensor_data(self, name: str) -> np.ndarray:
        if name in self._sensor_packet:
            return self._sensor_packet[name].copy()
        raise KeyError(f"Unknown Drake sensor: {name}")

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        if mode == "none":
            return BackendPlayRenderPlan(
                mode="none",
                headless=True,
                record_video=False,
                num_steps=play_steps,
                output_video=None,
            )
        if mode == "auto":
            return BackendPlayRenderPlan(
                mode="auto",
                headless=True,
                record_video=False,
                num_steps=play_steps,
                output_video=None,
            )
        if mode == "interactive":
            return BackendPlayRenderPlan(
                mode="interactive",
                headless=False,
                record_video=False,
                num_steps=play_steps,
                output_video=None,
            )
        assert mode == "record"
        if play_steps is None:
            raise ValueError("Drake record playback requires a finite training.play_steps value.")
        if output_video is None:
            raise ValueError("Drake record playback requires an output video path.")
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
        del render_offset_mode

        interactive = not bool(headless)
        should_record = bool(record_video)
        if should_record:
            return run_mujoco_playback(
                env=env,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                output_video=output_video,
                render_spacing=render_spacing,
                headless=True,
                record_video=True,
                frame_state_getter=frame_state_getter,
                camera_kwargs=camera_kwargs,
                extra_data_getter=extra_data_getter,
            )

        if interactive:
            url = self._ensure_meshcat()
            print(f"Drake Meshcat: {url}")

        state = initialize()
        steps_run = 0
        while num_steps is None or steps_run < int(num_steps):
            state = step(state)
            if interactive:
                self.render()
            steps_run += 1
        return None

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
        del spacing, offset_mode
        if capture:
            self._ensure_rgbd_sensor(
                width=int(width),
                height=int(height),
                camera_kwargs=dict(camera_kwargs or {}),
            )
            return
        if headless:
            raise NotImplementedError("DrakeBackend interactive renderer requires headless=false")
        self._ensure_meshcat()

    def render(self) -> None:
        if self._meshcat is None:
            raise NotImplementedError("Drake Meshcat renderer is not initialized")
        self._diagram.ForcedPublish(self._context)

    def capture_video_frame(self) -> np.ndarray:
        if self._rgbd_sensor is None:
            self._ensure_rgbd_sensor(width=640, height=360, camera_kwargs={})
        assert self._rgbd_sensor is not None
        sensor_context = self._rgbd_sensor.GetMyContextFromRoot(self._context)
        image = self._rgbd_sensor.color_image_output_port().Eval(sensor_context)
        return np.asarray(image.data, dtype=np.uint8)[:, :, :3].copy()

    def _pool_step_impl(
        self,
        state0: np.ndarray,
        control: np.ndarray,
        nstep: int,
        push_force: np.ndarray | None,
        return_sensor: bool,
    ) -> DrakePoolOutput:
        del state0
        step_start = time.perf_counter()
        push_values = (
            np.zeros((self._num_envs, 3), dtype=np.float64)
            if push_force is None
            else np.asarray(push_force, dtype=np.float64)
        )
        if control.ndim == 2:
            targets = np.clip(
                control,
                self._model_metadata.ctrl_limits[:, 0],
                self._model_metadata.ctrl_limits[:, 1],
            )
            advance_dt = float(nstep) * self._sim_dt
            for env_index, runtime in enumerate(self._runtimes):
                self._set_pd_target(targets[env_index], plant_context=runtime.plant_context)
                self._set_external_push_force(
                    push_values[env_index],
                    plant_context=runtime.plant_context,
                )
                target_time = runtime.simulator.get_context().get_time() + advance_dt
                runtime.simulator.AdvanceTo(target_time)
        else:
            targets = np.clip(
                control,
                self._model_metadata.ctrl_limits[:, 0],
                self._model_metadata.ctrl_limits[:, 1],
            )
            for env_index, runtime in enumerate(self._runtimes):
                for substep in range(int(nstep)):
                    self._set_pd_target(
                        targets[env_index, substep],
                        plant_context=runtime.plant_context,
                    )
                    self._set_external_push_force(
                        push_values[env_index],
                        plant_context=runtime.plant_context,
                    )
                    target_time = runtime.simulator.get_context().get_time() + self._sim_dt
                    runtime.simulator.AdvanceTo(target_time)

        self._bind_primary_runtime()
        timing = {"step_ms": (time.perf_counter() - step_start) * 1000.0}
        sensors = self._make_sensor_packet_from_live_contexts() if return_sensor else {}
        return DrakePoolOutput(
            state=self._make_physics_state_from_live_contexts(),
            sensor=sensors,
            timing=timing,
        )

    def _pool_reset_impl(
        self,
        env_ids: np.ndarray,
        initial_state: np.ndarray,
    ) -> DrakePoolOutput:
        qpos_rows, qvel_rows = self._unpack_state_rows(initial_state)
        for row_index, env_index in enumerate(env_ids):
            runtime = self._make_runtime(
                qpos_rows[row_index],
                qvel_rows[row_index],
                target_realtime_rate=self._target_realtime_rate,
            )
            self._runtimes[int(env_index)] = runtime
            self._diagram.ForcedPublish(runtime.context)
        self._bind_primary_runtime()
        return DrakePoolOutput(
            state=self._make_physics_state_from_live_contexts(),
            sensor=self._make_sensor_packet_from_live_contexts(),
        )

    def _apply_pool_output(self, output: DrakePoolOutput) -> None:
        self._physics_state = np.asarray(output.state, dtype=np.float64).copy()
        self._sensor_packet = {
            key: np.asarray(value, dtype=np.float64).copy() for key, value in output.sensor.items()
        }

    def _refresh_cached_outputs_from_live_contexts(self) -> None:
        self._apply_pool_output(
            DrakePoolOutput(
                state=self._make_physics_state_from_live_contexts(),
                sensor=self._make_sensor_packet_from_live_contexts(),
            )
        )

    def _pack_state_rows(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qpos_rows = np.asarray(qpos, dtype=np.float64)
        qvel_rows = np.asarray(qvel, dtype=np.float64)
        state = np.zeros((qpos_rows.shape[0], self._state_dim), dtype=np.float64)
        state[:, 1 : 1 + self._plant.num_positions()] = qpos_rows
        state[:, 1 + self._plant.num_positions() :] = qvel_rows
        return state

    def _unpack_state_rows(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(state, dtype=np.float64)
        qpos_start = 1
        qvel_start = qpos_start + self._plant.num_positions()
        return values[:, qpos_start:qvel_start], values[:, qvel_start:]

    def _make_physics_state_from_live_contexts(self) -> np.ndarray:
        qpos, qvel = self._current_mujoco_state_batch()
        state = self._pack_state_rows(qpos, qvel)
        state[:, 0] = [runtime.context.get_time() for runtime in self._runtimes]
        return state

    def _make_sensor_packet_from_live_contexts(self) -> SensorPacket:
        base_pos = np.asarray(
            [self._body_pos(self._trunk, env_index) for env_index in range(self._num_envs)],
            dtype=np.float64,
        )
        base_quat = np.asarray(
            [self._body_quat(self._trunk, env_index) for env_index in range(self._num_envs)],
            dtype=np.float64,
        )
        base_lin_vel = np.asarray(
            [
                self._body_spatial_velocity(self._trunk, env_index).translational()
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        )
        base_ang_vel = np.asarray(
            [
                self._body_spatial_velocity(self._trunk, env_index).rotational()
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        )
        local_spatial = [self._local_spatial_velocity(env_index) for env_index in range(self._num_envs)]
        gyro = np.asarray([values[0] for values in local_spatial], dtype=np.float64)
        local_linvel = np.asarray([values[1] for values in local_spatial], dtype=np.float64)
        upvector = np.asarray(
            [
                self._body_pose(self._trunk, env_index).rotation().matrix()[:, 2]
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        )
        dof_pos = np.asarray(
            [
                np.asarray(
                    self._plant.GetPositions(self._runtime(env_index).plant_context),
                    dtype=np.float64,
                )[ROOT_QPOS_DIM:]
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        )
        dof_vel = np.asarray(
            [
                np.asarray(
                    self._plant.GetVelocities(self._runtime(env_index).plant_context),
                    dtype=np.float64,
                )[ROOT_QVEL_DIM:]
                for env_index in range(self._num_envs)
            ],
            dtype=np.float64,
        )

        foot_names = [
            name for name in GO1_FOOT_SENSOR_NAMES if name in self._model_metadata.foot_sensor_to_body
        ]
        foot_pos = np.zeros((self._num_envs, len(foot_names), 3), dtype=np.float64)
        for foot_index, sensor_name in enumerate(foot_names):
            foot_pos[:, foot_index, :] = self._foot_sensor_pos_from_live_contexts(sensor_name)

        contact_names = [
            name
            for name in GO1_FOOT_CONTACT_SENSOR_NAMES
            if name in self._model_metadata.contact_sensors
        ]
        feet_contact_force = np.zeros((self._num_envs, len(contact_names), 3), dtype=np.float64)

        packet: SensorPacket = {
            "gyro": gyro,
            "local_linvel": local_linvel,
            "global_linvel": base_lin_vel,
            "global_angvel": base_ang_vel,
            "position": base_pos,
            "upvector": upvector,
            "base_pos": base_pos,
            "base_quat": base_quat,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "feet_pos": foot_pos,
            "feet_contact_force": feet_contact_force,
        }
        for foot_index, sensor_name in enumerate(foot_names):
            packet[sensor_name] = foot_pos[:, foot_index, :]
        for foot_index, sensor_name in enumerate(contact_names):
            packet[sensor_name] = feet_contact_force[:, foot_index, :]
        return packet

    def _foot_sensor_pos_from_live_contexts(self, name: str) -> np.ndarray:
        foot_body_name = self._model_metadata.foot_sensor_to_body[name]
        body = self._plant.GetBodyByName(foot_body_name, self._model_instance)
        foot_offset = self._model_metadata.foot_sensor_offsets[name]
        values = []
        for env_index in range(self._num_envs):
            x_wc = self._body_pose(body, env_index)
            values.append(x_wc.translation() + x_wc.rotation().matrix() @ foot_offset)
        return np.asarray(values, dtype=np.float64).reshape(self._num_envs, 3)

    def _set_pd_target(
        self, target_q: np.ndarray, *, plant_context: Any | None = None
    ) -> None:
        desired_state = np.concatenate(
            [
                np.asarray(target_q, dtype=np.float64),
                np.zeros(self.num_actuators, dtype=np.float64),
            ]
        )
        target_context = self._plant_context if plant_context is None else plant_context
        self._plant.get_desired_state_input_port(self._model_instance).FixValue(
            target_context, desired_state
        )

    def _sample_push_force(self, force_range: Sequence[float] | np.ndarray) -> np.ndarray:
        limit = np.asarray(force_range, dtype=np.float64)
        if limit.shape != (3,):
            raise ValueError(f"Drake push force range must have shape (3,), got {limit.shape}")
        direction = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3))
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.divide(direction, np.maximum(norm, 1.0e-12))
        magnitude = np.random.uniform(0.0, 1.0, size=(self._num_envs, 1))
        return direction * magnitude * limit.reshape(1, 3)

    def _set_external_push_force(self, force: np.ndarray, *, plant_context: Any) -> None:
        if np.allclose(force, 0.0):
            forces: list[Any] = []
        else:
            applied = ExternallyAppliedSpatialForce()
            applied.body_index = self._push_body.index()
            applied.p_BoBq_B = np.zeros(3, dtype=np.float64)
            applied.F_Bq_W = SpatialForce(tau=np.zeros(3, dtype=np.float64), f=force)
            forces = [applied]
        self._plant.get_applied_spatial_force_input_port().FixValue(plant_context, forces)

    def _runtime(self, env_index: int) -> _DrakeRuntime:
        if env_index < 0 or env_index >= self._num_envs:
            raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {env_index}")
        return self._runtimes[env_index]

    def _body_pose(self, body: Any, env_index: int = 0) -> Any:
        return self._plant.EvalBodyPoseInWorld(self._runtime(env_index).plant_context, body)

    def _body_pos(self, body: Any, env_index: int = 0) -> np.ndarray:
        return np.asarray(self._body_pose(body, env_index).translation(), dtype=np.float64)

    def _body_quat(self, body: Any, env_index: int = 0) -> np.ndarray:
        return _quat_from_rotation(self._body_pose(body, env_index).rotation())

    def _body_spatial_velocity(self, body: Any, env_index: int = 0) -> Any:
        return self._plant.EvalBodySpatialVelocityInWorld(
            self._runtime(env_index).plant_context, body
        )

    def _local_spatial_velocity(self, env_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        x_wb = self._body_pose(self._trunk, env_index)
        r_bw = x_wb.rotation().matrix().T
        velocity_w = self._body_spatial_velocity(self._trunk, env_index)
        return (
            r_bw @ np.asarray(velocity_w.rotational(), dtype=np.float64),
            r_bw @ np.asarray(velocity_w.translational(), dtype=np.float64),
        )

    def _current_mujoco_state(self, env_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        runtime = self._runtime(env_index)
        q = np.asarray(self._plant.GetPositions(runtime.plant_context), dtype=np.float64)
        v = np.asarray(self._plant.GetVelocities(runtime.plant_context), dtype=np.float64)
        return _drake_qpos_to_mujoco(q), _drake_qvel_to_mujoco(v)

    def _current_mujoco_state_batch(self) -> tuple[np.ndarray, np.ndarray]:
        qpos: list[np.ndarray] = []
        qvel: list[np.ndarray] = []
        for env_index in range(self._num_envs):
            qpos_row, qvel_row = self._current_mujoco_state(env_index)
            qpos.append(qpos_row)
            qvel.append(qvel_row)
        return np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)

    def get_physics_state(self) -> np.ndarray:
        return self._physics_state.copy()

    def get_playback_model(self, env_index: int | None = None) -> str:
        if env_index is not None:
            idx = int(env_index)
            if idx < 0 or idx >= self._num_envs:
                raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        return self._scene_model_file

    def _ensure_meshcat(self) -> str:
        if self._meshcat is None:
            qpos_mujoco, qvel_mujoco = self._unpack_state_rows(self._physics_state)
            self._meshcat = StartMeshcat()
            self._meshcat_url = str(self._meshcat.web_url())
            self._build_drake_system(
                meshcat=self._meshcat,
                camera_pose=None,
                qpos_mujoco=qpos_mujoco,
                qvel_mujoco=qvel_mujoco,
            )
            LOGGER.info("Started Drake Meshcat at %s", self._meshcat_url)
        assert self._meshcat_url is not None
        return self._meshcat_url

    def _ensure_rgbd_sensor(
        self,
        *,
        width: int,
        height: int,
        camera_kwargs: dict[str, Any],
    ) -> None:
        if (
            self._rgbd_sensor is not None
            and self._rgbd_width == width
            and self._rgbd_height == height
        ):
            return
        qpos_mujoco, qvel_mujoco = self._unpack_state_rows(self._physics_state)
        self._build_drake_system(
            meshcat=None,
            camera_pose=self._make_record_camera_pose(camera_kwargs),
            camera_width=width,
            camera_height=height,
            qpos_mujoco=qpos_mujoco,
            qvel_mujoco=qvel_mujoco,
        )

    def _make_record_camera_pose(self, camera_kwargs: dict[str, Any]) -> Any:
        target_value = camera_kwargs.get("cam_lookat")
        if target_value is None:
            target = np.array([2.0, 0.0, 0.35], dtype=np.float64)
        else:
            target = np.asarray(target_value, dtype=np.float64)
            if target.shape != (3,):
                target = np.array([2.0, 0.0, 0.35], dtype=np.float64)

        distance = float(camera_kwargs.get("cam_distance", 2.0))
        azimuth = np.deg2rad(float(camera_kwargs.get("cam_azimuth", 90.0)))
        elevation = np.deg2rad(abs(float(camera_kwargs.get("cam_elevation", -20.0))))
        horizontal = distance * np.cos(elevation)
        eye = target + np.array(
            [
                horizontal * np.cos(azimuth),
                horizontal * np.sin(azimuth),
                distance * np.sin(elevation),
            ],
            dtype=np.float64,
        )
        return self._look_at_transform(eye, target)

    def _look_at_transform(self, eye: np.ndarray, target: np.ndarray) -> Any:
        forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm <= 1.0e-12:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= right_norm
        down = np.cross(forward, right)
        return RigidTransform(RotationMatrix(np.column_stack([right, down, forward])), eye)

    def _exclude_robot_self_collisions(
        self,
        plant: Any,
        scene_graph: Any,
        model_instance: Any,
    ) -> int:
        robot_geometries = GeometrySet()
        count = 0
        for body_index in plant.GetBodyIndices(model_instance):
            body = plant.get_body(body_index)
            for geometry_id in plant.GetCollisionGeometriesForBody(body):
                robot_geometries.Add(geometry_id)
                count += 1
        if count:
            scene_graph.collision_filter_manager().Apply(
                CollisionFilterDeclaration().ExcludeWithin(robot_geometries)
            )
        return count


class _DrakeUniBatchBackend(SimBackend):
    """Go1-only DrakeUni backend that keeps pydrake out of the process."""

    backend_type = "drake"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str = "trunk",
        push_body_name: str | None = None,
        position_actuator_gains: dict[str, float] | None = None,
        nthread: int = 0,
        **_: Any,
    ) -> None:
        _load_drakeuni_symbols()
        if DrakeRuntimeConfig is None or create_drake_runtime is None:
            detail = DRAKE_BATCH_IMPORT_ERROR
            message = "DrakeUni runtime is not available."
            if detail is not None:
                message = f"{message} Import error: {detail}"
            raise ImportError(message) from detail
        if int(num_envs) < 1:
            raise ValueError(f"DrakeUni batch backend requires num_envs >= 1, got {num_envs}")

        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._scene_model_file = str(_resolve_scene_path(scene))
        self._kp = float((position_actuator_gains or {}).get("kp", 35.0))
        self._kd = float((position_actuator_gains or {}).get("kd", 0.5))
        self._base_name = str(base_name)
        self._push_body_name = str(push_body_name or base_name)
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
            robot_profile="go1",
        )
        self._runtime = create_drake_runtime(config)
        model_info = self._runtime.model_info()
        self._home_qpos_mujoco = model_info.home_qpos.copy()
        self._home_qvel_mujoco = model_info.home_qvel.copy()
        self._ctrl_limits = model_info.ctrl_limits.copy()
        self._joint_ranges = model_info.joint_ranges.copy()
        self._sensor_names = tuple(model_info.sensor_names)
        self._trunk_body_id = int(self._runtime.body_ids([self._base_name])[0])
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
                "DrakeUni batch backend currently supports Go1 push_robots interval randomization only"
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
        del render_offset_mode
        if bool(record_video):
            return run_mujoco_playback(
                env=env,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                output_video=output_video,
                render_spacing=render_spacing,
                headless=True,
                record_video=True,
                frame_state_getter=frame_state_getter,
                camera_kwargs=camera_kwargs,
                extra_data_getter=extra_data_getter,
            )
        if not bool(headless):
            raise NotImplementedError("DrakeUni batch backend does not support interactive rendering")
        state = initialize()
        steps_run = 0
        while num_steps is None or steps_run < int(num_steps):
            state = step(state)
            steps_run += 1
        return None

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
        self._require_trunk_only(body_ids)
        return np.repeat(self.get_base_pos()[:, None, :], len(body_ids), axis=1)

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_trunk_only(body_ids)
        return np.repeat(self.get_base_quat()[:, None, :], len(body_ids), axis=1)

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_trunk_only(body_ids)
        return np.repeat(self.get_base_lin_vel()[:, None, :], len(body_ids), axis=1)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_trunk_only(body_ids)
        return np.repeat(self.get_base_ang_vel()[:, None, :], len(body_ids), axis=1)

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_trunk_only(body_ids)
        return np.zeros((self._num_envs, len(body_ids), 3), dtype=np.float64)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_trunk_only(body_ids)
        quat = np.zeros((self._num_envs, len(body_ids), 4), dtype=np.float64)
        quat[:, :, 0] = 1.0
        return quat

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_trunk_only(body_ids)
        return np.repeat(self._sensor_packet["local_linvel"][:, None, :], len(body_ids), axis=1)

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_trunk_only(body_ids)
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

    def _require_trunk_only(self, body_ids: np.ndarray) -> None:
        ids = np.asarray(body_ids, dtype=np.int32)
        if ids.ndim != 1:
            raise ValueError(f"body_ids must be one-dimensional, got {ids.shape}")
        if np.any(ids != self._trunk_body_id):
            raise NotImplementedError(
                "DrakeUni batch backend only exposes trunk body kinematics in this milestone"
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
