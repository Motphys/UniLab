import os
import time
from collections.abc import Sequence
from typing import Any, TypeVar, cast

import numpy as np

from unilab.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

try:
    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderSettings

    MOTRIX_AVAILABLE = True
except ImportError:
    MOTRIX_AVAILABLE = False

from .base import BackendPlayCapabilities, SimBackend

T = TypeVar("T")
DEFAULT_MOTRIX_MAX_ITERATIONS = 3


def _require_not_none(value: T | None, error_message: str) -> T:
    if value is None:
        raise ValueError(error_message)
    return value


class MotrixBackend(SimBackend):
    """MotrixSim 后端实现"""

    def __init__(
        self,
        model_file: str,
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

        self.add_body_sensors = add_body_sensors
        self._base_name = base_name
        self._model_file = model_file

        if self.add_body_sensors:
            from unilab.base.backend.xml import inject_motrix_tracking_sensors

            tmp_path, _, valid_bnames = inject_motrix_tracking_sensors(
                model_file, baselink_name=base_name
            )
            try:
                self._model = mtx.load_model(tmp_path)  # pyright: ignore[reportPossiblyUnbound]
            finally:
                os.remove(tmp_path)

            # 用 motrixsim link index 作 key（Link 覆盖所有关节体，Body 只有 freejoint 根体）
            self._body_id_to_name: dict[int, str] = {
                idx: name
                for name in valid_bnames
                if (idx := self._model.get_link_index(name)) is not None
            }
        else:
            from unilab.base.backend.xml import create_motrix_compatible_xml

            tmp_path = create_motrix_compatible_xml(model_file)
            try:
                self._model = mtx.load_model(tmp_path)  # pyright: ignore[reportPossiblyUnbound]
            finally:
                os.remove(tmp_path)

            # 枚举所有具名 link，用 link index 作 key
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
        self._default_actuator_kp = np.zeros((self.num_actuators,), dtype=np.float64)
        self._default_actuator_kd = np.zeros((self.num_actuators,), dtype=np.float64)
        for actuator in self._position_actuators:
            idx = int(actuator.index)
            # TODO: switch to motrixsim model-level actuator gain API once available.
            self._default_actuator_kp[idx] = float(
                np.asarray(actuator.get_kp_override(self._data), dtype=np.float64)[0]
            )
            self._default_actuator_kd[idx] = float(
                np.asarray(actuator.get_kd_override(self._data), dtype=np.float64)[0]
            )
        self._floating_base_quat_indices: tuple[np.ndarray, ...] = tuple(
            np.asarray(floating_base.dof_pos_indices[3:7], dtype=np.intp)
            for floating_base in getattr(self._model, "floating_bases", [])
            if len(floating_base.dof_pos_indices) >= 7
        )
        self._default_base_mass_override = np.array(self._body_link.get_mass_override(self._data))
        self._default_base_com_override = np.array(
            self._body_link.get_center_of_mass_override(self._data)
        )
        self._render_app: "RenderApp | None" = None
        self._render_headless: bool | None = None
        self._render_capture_enabled = False
        self.backend_type = "motrix"

        # Pre-cache link objects to avoid repeated get_link() lookups.
        self._link_cache: dict[int, "mtx.Link"] = {}
        for link in self._model.links:
            if link.name:
                self._link_cache[link.index] = link

        # 运行一次正向运动学，确保初始 link 位置和传感器数据有效。
        self._model.forward_kinematic(self._data)
        self._refresh_link_pose_cache()

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
            return np.array(self._model.keyframes[0].dof_pos, dtype=self._np_dtype)
        return np.array(self._model.compute_init_dof_pos(), dtype=self._np_dtype)

    def get_default_qpos(self) -> np.ndarray:
        return np.array(self._model.compute_init_dof_pos(), dtype=self._np_dtype)

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
        supported_reset_terms = {RESET_TERM_BASE_MASS, RESET_TERM_BASE_COM}
        if self._supports_position_actuator_gains:
            supported_reset_terms.update({RESET_TERM_KP, RESET_TERM_KD})
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(supported_reset_terms),
            supports_interval_push=True,
            supports_interval_body_velocity_delta=False,
        )

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.push_perturbation_limit is None:
            return
        self.push_robots(plan.push_perturbation_limit)

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=True,
            supports_native_video_capture=True,
        )

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

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_lin_vel_w(body_ids)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_ang_vel_w(body_ids)

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
            qpos_motrix[:, quat_indices] = qpos[:, quat_indices[[1, 2, 3, 0]]]
        return qpos_motrix

    def _refresh_link_pose_cache(self, env_indices: np.ndarray | None = None) -> None:
        if env_indices is None:
            self._link_poses = self._model.get_link_poses(self._data)
        else:
            mask = np.zeros(self._num_envs, dtype=bool)
            mask[env_indices] = True
            self._link_poses[env_indices] = self._model.get_link_poses(self._data[mask])

    def push_robots(self, force_range):
        ex_force = np.random.rand(self.num_envs, 3) * 2 - 1  # [x_force, y_force, z_force]
        ex_force[:, 0] *= force_range[0]
        ex_force[:, 1] *= force_range[1]
        ex_force[:, 2] *= force_range[2]
        self._push_body_link.add_external_force(self._data, ex_force, local=True)

    def _render_offsets(self, spacing: float) -> list[list[float]]:
        cols = int(np.ceil(np.sqrt(self._num_envs)))
        offsets = []
        for i in range(self._num_envs):
            row = i // cols
            col = i % cols
            offsets.append([col * spacing, row * spacing, 0.0])
        return offsets

    def _resolve_system_camera_view(
        self,
        offsets: Sequence[Sequence[float]],
        camera_kwargs: dict[str, Any] | None,
    ) -> tuple[list[float], float, float, float]:
        cam_kw = dict(camera_kwargs or {})
        if bool(cam_kw.get("cam_tracking", False)):
            raise NotImplementedError("Motrix native video capture does not support cam_tracking")

        lookat_raw = cam_kw.get("cam_lookat")
        if lookat_raw is None:
            offsets_np = np.asarray(offsets, dtype=np.float64)
            lookat = [
                float(np.mean(offsets_np[:, 0])),
                float(np.mean(offsets_np[:, 1])),
                0.75,
            ]
        else:
            lookat_arr = np.asarray(lookat_raw, dtype=np.float64).reshape(-1)
            if lookat_arr.shape != (3,):
                raise ValueError(f"cam_lookat must contain 3 values, got {lookat_raw!r}")
            lookat = [float(lookat_arr[0]), float(lookat_arr[1]), float(lookat_arr[2])]

        return (
            lookat,
            float(cam_kw.get("cam_distance", 2.0)),
            float(cam_kw.get("cam_elevation", -20.0)),
            float(cam_kw.get("cam_azimuth", 90.0)),
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
        offsets = self._render_offsets(float(spacing))
        if capture:
            self._model.cameras.set_system_render_target("image", int(width), int(height))
            lookat, distance, elevation, azimuth = self._resolve_system_camera_view(
                offsets,
                camera_kwargs,
            )
        render_app = RenderApp(headless=headless)
        render_app.launch(
            self._model,
            batch=self._num_envs,
            render_offset=offsets,
            render_settings=settings,
        )
        if capture:
            render_app.system_camera.set_view(lookat, distance, elevation, azimuth)
        self._render_app = render_app
        self._render_headless = headless
        self._render_capture_enabled = capture

    def render(self):
        """Render current state (interactive visualization)"""
        if self._render_app is None:
            self.init_renderer()
        self._assert_render_context_available(headless=False, capture=False)
        assert self._render_app is not None
        self._render_app.sync(data=self._data)

    def capture_video_frame(self) -> np.ndarray:
        """Capture one RGB frame from Motrix's system camera."""
        if self._render_app is None:
            self.init_renderer(headless=True, capture=True)
        if not self._render_capture_enabled:
            raise RuntimeError("Motrix renderer is not initialized for video capture")
        assert self._render_app is not None

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
        if randomization.base_mass_delta is not None:
            base_mass = self._default_base_mass_override[env_ids].copy()
            randomized_mass = base_mass + randomization.base_mass_delta
            self._body_link.set_mass_override(data_slice, randomized_mass)

        if randomization.base_com_offset is not None:
            base_com = self._default_base_com_override[env_ids].copy()
            randomized_com = base_com + randomization.base_com_offset
            self._body_link.set_center_of_mass_override(data_slice, randomized_com)

        num_reset = len(env_ids)
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

    def _set_position_actuator_kp_override(self, data_slice, kp: np.ndarray) -> None:
        for actuator in self._position_actuators:
            actuator.set_kp_override(data_slice, kp[:, int(actuator.index)])

    def _set_position_actuator_kd_override(self, data_slice, kd: np.ndarray) -> None:
        for actuator in self._position_actuators:
            actuator.set_kd_override(data_slice, kd[:, int(actuator.index)])

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._supports_position_actuator_gains:
            raise NotImplementedError(
                "Motrix actuator gains are only exposed for all-position-actuator models"
            )
        return self._default_actuator_kp.copy(), self._default_actuator_kd.copy()
