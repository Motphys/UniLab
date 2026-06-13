from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from unilab.base.backend.drake.native import (
    NativeDrakeEnvPool,
    native_available,
    native_import_error,
)
from unilab.base.backend.mujoco.playback import run_mujoco_playback
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

NATIVE_DRAKE_AVAILABLE = bool(native_available())
NATIVE_DRAKE_IMPORT_ERROR = native_import_error()
ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6
GO1_FOOT_SENSOR_NAMES = ("FL_pos", "FR_pos", "RL_pos", "RR_pos")
GO1_FOOT_CONTACT_SENSOR_NAMES = (
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
)
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

# Drake body indices for src/unilab/assets/robots/go1/scene_flat_drake.xml.
# This backend is intentionally Go1-only until DrakeUni grows a native metadata
# query layer.
GO1_BODY_INDICES = {
    "trunk": 1,
    "FR_hip": 2,
    "FR_thigh": 3,
    "FR_calf": 4,
    "FL_hip": 5,
    "FL_thigh": 6,
    "FL_calf": 7,
    "RR_hip": 8,
    "RR_thigh": 9,
    "RR_calf": 10,
    "RL_hip": 11,
    "RL_thigh": 12,
    "RL_calf": 13,
}


@dataclass(frozen=True)
class _NativeDrakeModelView:
    nq: int
    nv: int
    nu: int

    def num_positions(self) -> int:
        return self.nq

    def num_velocities(self) -> int:
        return self.nv

    def num_actuators(self) -> int:
        return self.nu


@dataclass(frozen=True)
class _Go1Metadata:
    ctrl_limits: np.ndarray
    torque_limits: np.ndarray
    joint_ranges: np.ndarray
    foot_sensor_to_body: dict[str, str]
    foot_sensor_offsets: dict[str, np.ndarray]
    contact_sensors: frozenset[str]


def _resolve_scene_path(scene: SceneCfg) -> Path:
    if not scene.model_file:
        raise ValueError("NativeDrakeBackend requires SceneCfg.model_file")
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

    def walk_default(node: ET.Element, inherited: dict[str, dict[str, str]]) -> None:
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
            joint_ranges.append(joint_ranges_by_name.get(joint_name or "", ctrl_range))

    if not ctrl_limits:
        raise ValueError("NativeDrakeBackend requires position actuators")
    return (
        np.asarray(ctrl_limits, dtype=np.float64),
        np.asarray(torque_limits, dtype=np.float64),
        np.asarray(joint_ranges, dtype=np.float64),
    )


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


def _load_go1_metadata(scene_path: Path) -> _Go1Metadata:
    roots = _load_xml_roots(scene_path)
    defaults = _collect_default_classes(roots)
    joint_ranges = _extract_joint_ranges(roots, defaults)
    ctrl_limits, torque_limits, joint_limits = _extract_actuator_metadata(
        roots,
        defaults,
        joint_ranges,
    )
    foot_sensor_to_body, foot_sensor_offsets, contact_sensors = _extract_sensor_metadata(
        roots,
        _extract_sites(roots),
    )
    return _Go1Metadata(
        ctrl_limits=ctrl_limits,
        torque_limits=torque_limits,
        joint_ranges=joint_limits,
        foot_sensor_to_body=foot_sensor_to_body,
        foot_sensor_offsets=foot_sensor_offsets,
        contact_sensors=contact_sensors,
    )


def _read_keyframe_qpos(scene_path: Path, name: str) -> np.ndarray:
    for _, root in _load_xml_roots(scene_path):
        for key in root.findall(".//key"):
            if key.attrib.get("name") == name:
                values = _parse_vector(key.attrib.get("qpos"))
                if values is not None:
                    return values.copy()
    raise ValueError(f"NativeDrakeBackend requires keyframe {name!r} in {scene_path}")


def _resolve_native_nthread(num_envs: int, requested: int) -> int:
    env_count = max(1, int(num_envs))
    requested_count = int(requested)
    if requested_count > 0:
        return min(env_count, requested_count)
    return min(env_count, max(1, cpu_count() * 2))


class NativeDrakeBackend(SimBackend):
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
        if NativeDrakeEnvPool is None:
            detail = NATIVE_DRAKE_IMPORT_ERROR
            message = "Native DrakeEnvPool extension has not been built."
            if detail is not None:
                message = f"{message} Import error: {detail}"
            raise ImportError(message) from detail
        if int(num_envs) < 1:
            raise ValueError(f"NativeDrakeBackend requires num_envs >= 1, got {num_envs}")

        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._scene_model_file = str(_resolve_scene_path(scene))
        self._metadata = _load_go1_metadata(Path(self._scene_model_file))
        self._home_qpos_mujoco = _read_keyframe_qpos(Path(self._scene_model_file), "home")
        self._kp = float((position_actuator_gains or {}).get("kp", 35.0))
        self._kd = float((position_actuator_gains or {}).get("kd", 0.5))
        self._base_name = str(base_name)
        self._push_body_name = str(push_body_name or base_name)
        self._nthread = _resolve_native_nthread(self._num_envs, int(nthread))
        self._pending_push_force = np.zeros((self._num_envs, 3), dtype=np.float64)

        base_body_index = self._body_index(self._base_name)
        push_body_index = self._body_index(self._push_body_name)
        foot_body_indices, foot_offsets = self._native_foot_metadata()
        self._pool = NativeDrakeEnvPool(
            self._scene_model_file,
            self._num_envs,
            self._sim_dt,
            self._metadata.ctrl_limits,
            self._metadata.torque_limits,
            base_body_index,
            push_body_index,
            foot_body_indices,
            foot_offsets,
            self._kp,
            self._kd,
            self._nthread,
        )
        nv = int(self._pool.state_dim) - 1 - int(self._home_qpos_mujoco.size)
        if nv <= ROOT_QVEL_DIM:
            raise RuntimeError(f"Native Drake pool returned invalid nv={nv}")
        self._home_qvel_mujoco = np.zeros(nv, dtype=np.float64)
        self._model = _NativeDrakeModelView(
            nq=int(self._home_qpos_mujoco.size),
            nv=nv,
            nu=int(self._pool.control_dim),
        )
        self._physics_state = np.zeros((self._num_envs, int(self._pool.state_dim)), dtype=np.float64)
        self._sensor_packet: dict[str, np.ndarray] = {}
        qpos = np.broadcast_to(self._home_qpos_mujoco, (self._num_envs, self._model.nq)).copy()
        qvel = np.zeros((self._num_envs, self._model.nv), dtype=np.float64)
        self.set_state(np.arange(self._num_envs, dtype=np.int32), qpos, qvel)

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
    def model(self) -> _NativeDrakeModelView:
        return self._model

    @property
    def num_actuators(self) -> int:
        return self._model.nu

    @property
    def num_dof_vel(self) -> int:
        return self._model.nv - ROOT_QVEL_DIM

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._metadata.ctrl_limits.copy()

    def get_joint_range(self) -> np.ndarray | None:
        return self._metadata.joint_ranges.copy()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if name != "home":
            raise KeyError(f"NativeDrakeBackend only exposes keyframe 'home', got {name!r}")
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
        return np.asarray([self._body_index(str(name)) for name in names], dtype=np.int32)

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (self._num_envs, self.num_actuators):
            raise ValueError(
                "NativeDrakeBackend.step expected ctrl shape "
                f"({self._num_envs}, {self.num_actuators}), got {values.shape}"
            )
        values = self._apply_pre_step_control(values)
        start = time.perf_counter()
        output = self._pool.step(
            self._physics_state,
            int(nsteps),
            values,
            self._pending_push_force,
        )
        self._pending_push_force.fill(0.0)
        self._apply_native_output(output)
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
            raise NotImplementedError("NativeDrakeBackend does not apply reset randomization yet")
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
        output = self._pool.reset(indices, self._pack_state_rows(qpos_rows, qvel_rows))
        self._apply_native_output(output)

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
                "NativeDrakeBackend currently supports Go1 push_robots interval randomization only"
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
            raise NotImplementedError("NativeDrakeBackend does not support interactive rendering")
        if play_steps is None:
            raise ValueError("Native Drake record playback requires a finite play_steps value.")
        if output_video is None:
            raise ValueError("Native Drake record playback requires an output video path.")
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
            raise NotImplementedError("NativeDrakeBackend does not support interactive rendering")
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
        raise NotImplementedError("NativeDrakeBackend records through run_playback")

    def render(self) -> None:
        raise NotImplementedError("NativeDrakeBackend does not support interactive rendering")

    def capture_video_frame(self) -> np.ndarray:
        raise NotImplementedError("NativeDrakeBackend records through run_playback")

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
        raise KeyError(f"Unknown Native Drake sensor: {name}")

    def get_physics_state(self) -> np.ndarray:
        return self._physics_state.copy()

    def get_playback_model(self, env_index: int | None = None) -> str:
        if env_index is not None:
            idx = int(env_index)
            if idx < 0 or idx >= self._num_envs:
                raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        return self._scene_model_file

    def _apply_native_output(self, output: dict[str, Any]) -> None:
        self._physics_state = np.asarray(output["state"], dtype=np.float64).copy()
        raw_sensor = output.get("sensor", {})
        packet = {key: np.asarray(value, dtype=np.float64).copy() for key, value in raw_sensor.items()}
        feet_pos = packet.get("feet_pos")
        if feet_pos is not None:
            for foot_index, sensor_name in enumerate(GO1_FOOT_SENSOR_NAMES):
                if foot_index < feet_pos.shape[1]:
                    packet[sensor_name] = feet_pos[:, foot_index, :]
        feet_contact = packet.get("feet_contact_force")
        if feet_contact is not None:
            for foot_index, sensor_name in enumerate(GO1_FOOT_CONTACT_SENSOR_NAMES):
                if foot_index < feet_contact.shape[1]:
                    packet[sensor_name] = feet_contact[:, foot_index, :]
        packet.setdefault("position", packet["base_pos"])
        self._sensor_packet = packet

    def _pack_state_rows(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qpos_rows = np.asarray(qpos, dtype=np.float64)
        qvel_rows = np.asarray(qvel, dtype=np.float64)
        state = np.zeros((qpos_rows.shape[0], int(self._pool.state_dim)), dtype=np.float64)
        state[:, 1 : 1 + self._model.nq] = qpos_rows
        state[:, 1 + self._model.nq :] = qvel_rows
        return state

    def _native_foot_metadata(self) -> tuple[list[int], np.ndarray]:
        body_indices: list[int] = []
        offsets: list[np.ndarray] = []
        for sensor_name in GO1_FOOT_SENSOR_NAMES:
            body_name = self._metadata.foot_sensor_to_body.get(sensor_name)
            if body_name is None:
                raise ValueError(f"NativeDrakeBackend missing foot sensor {sensor_name!r}")
            body_indices.append(self._body_index(body_name))
            offsets.append(self._metadata.foot_sensor_offsets[sensor_name])
        return body_indices, np.asarray(offsets, dtype=np.float64)

    def _body_index(self, name: str) -> int:
        try:
            return GO1_BODY_INDICES[name]
        except KeyError as exc:
            raise ValueError(f"NativeDrakeBackend only knows Go1 body {name!r}") from exc

    def _require_trunk_only(self, body_ids: np.ndarray) -> None:
        ids = np.asarray(body_ids, dtype=np.int32)
        if ids.ndim != 1:
            raise ValueError(f"body_ids must be one-dimensional, got {ids.shape}")
        trunk_id = GO1_BODY_INDICES["trunk"]
        if np.any(ids != trunk_id):
            raise NotImplementedError(
                "NativeDrakeBackend only exposes trunk body kinematics in this milestone"
            )

    def _sample_push_force(self, force_range: Sequence[float] | np.ndarray) -> np.ndarray:
        limit = np.asarray(force_range, dtype=np.float64)
        if limit.shape != (3,):
            raise ValueError(f"Native Drake push force range must have shape (3,), got {limit.shape}")
        direction = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3))
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.divide(direction, np.maximum(norm, 1.0e-12))
        magnitude = np.random.uniform(0.0, 1.0, size=(self._num_envs, 1))
        return direction * magnitude * limit.reshape(1, 3)


__all__ = ["NATIVE_DRAKE_AVAILABLE", "NATIVE_DRAKE_IMPORT_ERROR", "NativeDrakeBackend"]
