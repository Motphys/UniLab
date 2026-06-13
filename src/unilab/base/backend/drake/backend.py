from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
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
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

try:  # pragma: no cover - exercised by integration smoke tests when Drake is installed.
    from pydrake.all import (
        AddMultibodyPlantSceneGraph,
        CameraInfo,
        ClippingRange,
        DepthRange,
        DepthRenderCamera,
        DiagramBuilder,
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
    from pydrake.multibody.plant import ContactModel, DiscreteContactApproximation
    from pydrake.multibody.tree import BodyIndex, PdControllerGains

    DRAKE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard.
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
    StartMeshcat = None
    Simulator = None
    DRAKE_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

DEFAULT_GO1_QPOS_MUJOCO = np.array(
    [
        0.0,
        0.0,
        0.27,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.9,
        -1.8,
        0.0,
        0.9,
        -1.8,
        0.0,
        1.0,
        -1.8,
        0.0,
        1.0,
        -1.8,
    ],
    dtype=np.float64,
)
GO1_CTRL_LIMITS = np.array(
    [
        [-0.863, 0.863],
        [-0.686, 4.501],
        [-2.818, -0.888],
    ]
    * 4,
    dtype=np.float64,
)
GO1_TORQUE_LIMITS = np.array([23.7, 23.7, 35.55] * 4, dtype=np.float64)
GO1_FOOT_OFFSET_IN_CALF = np.array([0.0, 0.0, -0.213], dtype=np.float64)
GO1_FOOT_SENSOR_TO_BODY = {
    "FR_pos": "FR_calf",
    "FL_pos": "FL_calf",
    "RR_pos": "RR_calf",
    "RL_pos": "RL_calf",
}
GO1_CONTACT_SENSORS = {
    "FR_foot_contact",
    "FL_foot_contact",
    "RR_foot_contact",
    "RL_foot_contact",
}


def _require_drake() -> None:
    if not DRAKE_AVAILABLE:
        raise ImportError("Drake backend requested, but pydrake is not installed.")


def _resolve_scene_path(scene: SceneCfg) -> Path:
    if not scene.model_file:
        raise ValueError("DrakeBackend requires SceneCfg.model_file")
    path = Path(scene.model_file)
    return path if path.is_absolute() else Path.cwd() / path


def _read_home_qpos(scene_path: Path) -> np.ndarray:
    try:
        tree = ET.parse(scene_path)
    except ET.ParseError:
        return DEFAULT_GO1_QPOS_MUJOCO.copy()
    for key in tree.findall(".//key"):
        if key.attrib.get("name") != "home":
            continue
        qpos_text = key.attrib.get("qpos", "")
        values = np.fromstring(qpos_text, sep=" ", dtype=np.float64)
        if values.shape == DEFAULT_GO1_QPOS_MUJOCO.shape:
            return values
    return DEFAULT_GO1_QPOS_MUJOCO.copy()


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
    """Minimal Drake backend for the Go1JoystickFlat replay milestone.

    The contract intentionally mirrors MuJoCo-shaped qpos/qvel at the UniLab
    boundary while storing Drake's quaternion-first floating-base order inside
    the plant. This keeps the existing locomotion reset and observation code
    reusable for the first Drake replay milestone.
    """

    backend_type = "drake"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str = "trunk",
        position_actuator_gains: dict[str, float] | None = None,
        **_: Any,
    ) -> None:
        _require_drake()
        if num_envs != 1:
            raise NotImplementedError("DrakeBackend currently supports exactly one environment.")

        logging.getLogger("drake").setLevel(logging.ERROR)
        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._scene_model_file = str(_resolve_scene_path(scene))
        self._home_qpos_mujoco = _read_home_qpos(Path(self._scene_model_file))
        self._home_qpos_drake = _mujoco_qpos_to_drake(self._home_qpos_mujoco)
        self._home_qvel_mujoco = np.zeros(18, dtype=np.float64)
        self._kp = float((position_actuator_gains or {}).get("kp", 35.0))
        self._kd = float((position_actuator_gains or {}).get("kd", 0.5))
        self._base_name = base_name
        self._meshcat = None
        self._meshcat_url: str | None = None
        self._rgbd_sensor = None
        self._rgbd_width = 0
        self._rgbd_height = 0

        self._build_drake_system(meshcat=None, camera_pose=None)
        LOGGER.info(
            "Initialized Drake Go1 backend from %s; filtered %d robot collision geometries",
            self._scene_model_file,
            self._num_filtered_geometries,
        )

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

        for i, effort_limit in enumerate(GO1_TORQUE_LIMITS):
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
            self._rgbd_sensor = builder.AddSystem(
                RgbdSensor(scene_graph.world_frame_id(), camera_pose, depth_camera, False)
            )
            builder.Connect(
                scene_graph.get_query_output_port(),
                self._rgbd_sensor.query_object_input_port(),
            )
            self._rgbd_width = int(camera_width)
            self._rgbd_height = int(camera_height)

        self._diagram = builder.Build()
        self._context = self._diagram.CreateDefaultContext()
        self._plant = plant
        self._scene_graph = scene_graph
        self._plant_context = self._plant.GetMyMutableContextFromRoot(self._context)
        self._trunk = self._plant.GetBodyByName(self._base_name, self._model_instance)

        if (
            self._plant.num_positions() != 19
            or self._plant.num_velocities() != 18
            or self._plant.num_actuators() != 12
        ):
            raise RuntimeError(
                "DrakeBackend currently supports Go1 dimensions only: "
                f"nq={self._plant.num_positions()} "
                f"nv={self._plant.num_velocities()} "
                f"nu={self._plant.num_actuators()}"
            )

        reset_qpos = self._home_qpos_mujoco if qpos_mujoco is None else qpos_mujoco
        reset_qvel = self._home_qvel_mujoco if qvel_mujoco is None else qvel_mujoco
        self._plant.SetPositions(self._plant_context, _mujoco_qpos_to_drake(reset_qpos))
        self._plant.SetVelocities(self._plant_context, _mujoco_qvel_to_drake(reset_qvel))
        self._plant.get_actuation_input_port(self._model_instance).FixValue(
            self._plant_context, np.zeros(12, dtype=np.float64)
        )
        self._set_native_pd_target(reset_qpos[7:])

        self._simulator = Simulator(self._diagram, self._context)
        self._simulator.set_target_realtime_rate(1.0 if meshcat is not None else 0.0)
        self._simulator.Initialize()
        self._diagram.ForcedPublish(self._context)

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
        return int(self._plant.num_velocities() - 6)

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return GO1_CTRL_LIMITS.copy()

    def get_joint_range(self) -> np.ndarray | None:
        return GO1_CTRL_LIMITS.copy()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if name != "home":
            raise KeyError(f"DrakeBackend only exposes Go1 keyframe 'home', got {name!r}")
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
        step_start = time.perf_counter()
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (1, self.num_actuators):
            raise ValueError(f"DrakeBackend.step expected ctrl shape (1, 12), got {values.shape}")
        values = self._apply_pre_step_control(values)
        target_q = np.clip(values[0], GO1_CTRL_LIMITS[:, 0], GO1_CTRL_LIMITS[:, 1])
        self._set_native_pd_target(target_q)
        target_time = self._simulator.get_context().get_time() + float(nsteps) * self._sim_dt
        self._simulator.AdvanceTo(target_time)
        return {"timing": {"step_ms": (time.perf_counter() - step_start) * 1000.0}}

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        indices = np.asarray(env_indices, dtype=np.int32)
        if indices.shape != (1,) or int(indices[0]) != 0:
            raise NotImplementedError("DrakeBackend.set_state currently supports only env index 0")
        if randomization is not None and not randomization.is_empty():
            raise NotImplementedError("DrakeBackend does not apply reset randomization yet")

        qpos_rows = np.asarray(qpos, dtype=np.float64)
        qvel_rows = np.asarray(qvel, dtype=np.float64)
        if qpos_rows.shape != (1, self._plant.num_positions()):
            raise ValueError(f"qpos must have shape (1, 19), got {qpos_rows.shape}")
        if qvel_rows.shape != (1, self._plant.num_velocities()):
            raise ValueError(f"qvel must have shape (1, 18), got {qvel_rows.shape}")

        self._plant.SetPositions(self._plant_context, _mujoco_qpos_to_drake(qpos_rows[0]))
        self._plant.SetVelocities(self._plant_context, _mujoco_qvel_to_drake(qvel_rows[0]))
        self._set_native_pd_target(qpos_rows[0, 7:])
        self._diagram.ForcedPublish(self._context)

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities()

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        raise NotImplementedError("DrakeBackend does not support interval randomization yet")

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=True,
            supports_native_video_capture=True,
        )

    def get_base_pos(self) -> np.ndarray:
        return self._body_pos(self._trunk).reshape(1, 3)

    def get_base_quat(self) -> np.ndarray:
        return self._body_quat(self._trunk).reshape(1, 4)

    def get_base_lin_vel(self) -> np.ndarray:
        return np.asarray(self._body_spatial_velocity(self._trunk).translational()).reshape(1, 3)

    def get_base_ang_vel(self) -> np.ndarray:
        return np.asarray(self._body_spatial_velocity(self._trunk).rotational()).reshape(1, 3)

    def get_dof_pos(self) -> np.ndarray:
        q = np.asarray(self._plant.GetPositions(self._plant_context), dtype=np.float64)
        return q[7:].reshape(1, -1)

    def get_dof_vel(self) -> np.ndarray:
        v = np.asarray(self._plant.GetVelocities(self._plant_context), dtype=np.float64)
        return v[6:].reshape(1, -1)

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        bodies = [self._plant.get_body(BodyIndex(int(body_id))) for body_id in body_ids]
        values = [self._body_pos(body) for body in bodies]
        return np.asarray(values, dtype=np.float64).reshape(1, len(values), 3)

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        bodies = [self._plant.get_body(BodyIndex(int(body_id))) for body_id in body_ids]
        values = [self._body_quat(body) for body in bodies]
        return np.asarray(values, dtype=np.float64).reshape(1, len(values), 4)

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        bodies = [self._plant.get_body(BodyIndex(int(body_id))) for body_id in body_ids]
        values = [self._body_spatial_velocity(body).translational() for body in bodies]
        return np.asarray(values, dtype=np.float64).reshape(1, len(values), 3)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        bodies = [self._plant.get_body(BodyIndex(int(body_id))) for body_id in body_ids]
        values = [self._body_spatial_velocity(body).rotational() for body in bodies]
        return np.asarray(values, dtype=np.float64).reshape(1, len(values), 3)

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        pos_w = self.get_body_pos_w(body_ids)[0]
        x_wb = self._body_pose(self._trunk)
        rotation_bw = x_wb.rotation().matrix().T
        base_pos = np.asarray(x_wb.translation(), dtype=np.float64)
        return ((pos_w - base_pos) @ rotation_bw.T).reshape(1, len(body_ids), 3)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        x_wb = self._body_pose(self._trunk)
        r_bw = x_wb.rotation().matrix().T
        values = []
        for body_id in body_ids:
            body = self._plant.get_body(BodyIndex(int(body_id)))
            r_wi = self._body_pose(body).rotation().matrix()
            values.append(_quat_from_matrix(r_bw @ r_wi))
        return np.asarray(values, dtype=np.float64).reshape(1, len(values), 4)

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        vel_w = self.get_body_lin_vel_w(body_ids)[0]
        rotation_bw = self._body_pose(self._trunk).rotation().matrix().T
        return (vel_w @ rotation_bw.T).reshape(1, len(body_ids), 3)

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        vel_w = self.get_body_ang_vel_w(body_ids)[0]
        rotation_bw = self._body_pose(self._trunk).rotation().matrix().T
        return (vel_w @ rotation_bw.T).reshape(1, len(body_ids), 3)

    def get_sensor_data(self, name: str) -> np.ndarray:
        if name == "gyro":
            return self._local_spatial_velocity()[0].reshape(1, 3)
        if name == "local_linvel":
            return self._local_spatial_velocity()[1].reshape(1, 3)
        if name == "global_linvel":
            return self.get_base_lin_vel()
        if name == "global_angvel":
            return self.get_base_ang_vel()
        if name == "position":
            return self.get_base_pos()
        if name == "upvector":
            return self._body_pose(self._trunk).rotation().matrix()[:, 2].reshape(1, 3)
        if name in GO1_FOOT_SENSOR_TO_BODY:
            body = self._plant.GetBodyByName(GO1_FOOT_SENSOR_TO_BODY[name], self._model_instance)
            x_wc = self._body_pose(body)
            foot_pos = x_wc.translation() + x_wc.rotation().matrix() @ GO1_FOOT_OFFSET_IN_CALF
            return np.asarray(foot_pos, dtype=np.float64).reshape(1, 3)
        if name in GO1_CONTACT_SENSORS:
            return np.zeros((1, 3), dtype=np.float64)
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
        del render_spacing, render_offset_mode
        del frame_state_getter, extra_data_getter

        interactive = not bool(headless)
        should_record = bool(record_video)
        if should_record:
            if num_steps is None:
                raise ValueError("Drake video recording requires a finite num_steps value.")
            if output_video is None:
                raise ValueError("Drake video recording requires an output_video path.")
            self.init_renderer(
                headless=True,
                capture=True,
                width=640,
                height=360,
                camera_kwargs=dict(camera_kwargs or {}),
            )
            state = initialize()
            frames: list[np.ndarray] = []
            for _ in range(int(num_steps)):
                state = step(state)
                frames.append(self.capture_video_frame())
            import mediapy as media

            ctrl_dt = float(getattr(getattr(env, "cfg", None), "ctrl_dt", 1.0 / 50.0))
            fps = max(1, int(round(1.0 / ctrl_dt)))
            media.write_video(str(output_video), frames, fps=fps)
            return str(output_video)

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

    def _set_native_pd_target(self, target_q: np.ndarray) -> None:
        desired_state = np.concatenate(
            [
                np.asarray(target_q, dtype=np.float64),
                np.zeros(self.num_actuators, dtype=np.float64),
            ]
        )
        self._plant.get_desired_state_input_port(self._model_instance).FixValue(
            self._plant_context, desired_state
        )

    def _body_pose(self, body: Any) -> Any:
        return self._plant.EvalBodyPoseInWorld(self._plant_context, body)

    def _body_pos(self, body: Any) -> np.ndarray:
        return np.asarray(self._body_pose(body).translation(), dtype=np.float64)

    def _body_quat(self, body: Any) -> np.ndarray:
        return _quat_from_rotation(self._body_pose(body).rotation())

    def _body_spatial_velocity(self, body: Any) -> Any:
        return self._plant.EvalBodySpatialVelocityInWorld(self._plant_context, body)

    def _local_spatial_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        x_wb = self._body_pose(self._trunk)
        r_bw = x_wb.rotation().matrix().T
        velocity_w = self._body_spatial_velocity(self._trunk)
        return (
            r_bw @ np.asarray(velocity_w.rotational(), dtype=np.float64),
            r_bw @ np.asarray(velocity_w.translational(), dtype=np.float64),
        )

    def _current_mujoco_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(self._plant.GetPositions(self._plant_context), dtype=np.float64)
        v = np.asarray(self._plant.GetVelocities(self._plant_context), dtype=np.float64)
        return _drake_qpos_to_mujoco(q), _drake_qvel_to_mujoco(v)

    def _ensure_meshcat(self) -> str:
        if self._meshcat is None:
            qpos_mujoco, qvel_mujoco = self._current_mujoco_state()
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
        if self._rgbd_sensor is not None and self._rgbd_width == width and self._rgbd_height == height:
            return
        qpos_mujoco, qvel_mujoco = self._current_mujoco_state()
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

        distance = float(camera_kwargs.get("cam_distance", 6.0))
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
