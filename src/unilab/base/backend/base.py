import abc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np

from unilab.dr.types import (
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

PreStepControlFn = Callable[[Any, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BackendPlayCapabilities:
    """Backend-native play/render capabilities surfaced through env contracts."""

    supports_native_interactive_renderer: bool = False
    supports_physics_state_playback: bool = False
    supports_native_video_capture: bool = False


class BackendHeightScanner(abc.ABC):
    """Backend-owned height-field scanner created on the env init path."""

    @abc.abstractmethod
    def scan(self) -> np.ndarray:
        """Return sampled values with shape ``(num_envs, num_points)``."""


PLAY_RENDER_MODES = frozenset({"auto", "interactive", "record", "none"})


@dataclass(frozen=True)
class BackendPlayRenderPlan:
    """Backend-resolved playback rendering behavior."""

    mode: str
    headless: bool
    record_video: bool
    num_steps: int | None
    output_video: str | PathLike[str] | None


def normalize_play_render_mode(play_render_mode: str | None) -> str:
    mode = "auto" if play_render_mode is None else str(play_render_mode).strip().lower()
    if mode not in PLAY_RENDER_MODES:
        joined = ", ".join(sorted(PLAY_RENDER_MODES))
        raise ValueError(f"training.play_render_mode must be one of: {joined}; got {mode!r}.")
    return mode


class SimBackend(abc.ABC):
    """Unified simulation backend contract."""

    _pre_step_control_fn: PreStepControlFn | None
    _scene_cleanup_handle: Any | None
    backend_type: str

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """Number of vectorized environments."""

    @property
    @abc.abstractmethod
    def model(self):
        """Underlying physics model."""

    # ------------------------------------------------------------------ #
    # Model properties                                                     #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def num_actuators(self) -> int:
        """Number of actuators."""

    @property
    @abc.abstractmethod
    def num_dof_vel(self) -> int:
        """Number of joint velocity DoFs, excluding the floating base."""

    @abc.abstractmethod
    def get_actuator_ctrl_range(self) -> np.ndarray:
        """Return actuator control ranges.

        Returns:
            Array with shape ``(num_actuators, 2)`` and columns ``[low, high]``.
        """

    @abc.abstractmethod
    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        """Return the full qpos for a named keyframe, including the floating base.

        Args:
            name: Keyframe name such as ``"stand"`` or ``"home"``.

        Returns:
            Array with shape ``(nq,)``.
        """

    def get_default_qpos(self) -> np.ndarray:
        """Return the backend/model default qpos through a stable contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose default qpos")

    @abc.abstractmethod
    def get_init_qvel(self) -> np.ndarray:
        """Return a zero-initialized qvel vector compatible with ``set_state``.

        Returns:
            Zero-filled qvel array.
        """

    @abc.abstractmethod
    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve body/link names to backend integer IDs.

        Args:
            names: Body/link names.

        Returns:
            ``int32`` array with shape ``(len(names),)``.

        Raises:
            ValueError: If any name is not found.
        """

    def get_body_id(self, name: str) -> int:
        """Resolve one body/link name through the backend contract."""
        return int(self.get_body_ids([name])[0])

    def get_geom_id(self, name: str) -> int:
        """Resolve one geom name through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom ids")

    def get_geom_size(self, name: str) -> np.ndarray:
        """Return one geom size vector through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom sizes")

    def create_hfield_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> BackendHeightScanner:
        """Create a reusable height-field scanner on the init/cold path.

        Backends that support height-field terrain scan must override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native height-field scanners"
        )

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        """Return body ids in the subtree rooted at ``root_body_id``."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body subtree ids")

    def get_geom_names(self) -> tuple[str, ...]:
        """Return backend geom names in backend id order."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom names")

    def get_geom_body_ids(self) -> np.ndarray:
        """Return the owning body id for each geom."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom body ids")

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-geom contact type and affinity masks."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom contact masks")

    def get_geom_friction(self) -> np.ndarray:
        """Return the backend geom-friction table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom friction")

    def get_gravity(self) -> np.ndarray:
        """Return the backend gravity vector."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose gravity")

    def get_body_mass(self) -> np.ndarray:
        """Return the backend body-mass table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body mass")

    def get_body_ipos(self) -> np.ndarray:
        """Return the backend body inertial-position table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body ipos")

    def get_dof_armature(self) -> np.ndarray:
        """Return the backend dof-armature table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose dof armature")

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve MuJoCo-style body IDs used by motion datasets."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose motion body ids")

    def cleanup_scene_assets(self) -> None:
        """Release cold-path scene artifacts owned by the backend."""
        cleanup_handle = getattr(self, "_scene_cleanup_handle", None)
        if cleanup_handle is None:
            return
        cleanup_handle.cleanup()
        self._scene_cleanup_handle = None

    def __del__(self) -> None:
        try:
            self.cleanup_scene_assets()
        except Exception:
            pass

    @abc.abstractmethod
    def get_joint_range(self) -> np.ndarray | None:
        """Return joint position limits, excluding the floating base.

        Returns:
            Array with shape ``(num_dof, 2)`` and columns ``[low, high]``, or
            ``None`` when the backend does not expose limits.
        """

    # ------------------------------------------------------------------ #
    # Simulation control                                                   #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        """Advance physics.

        Args:
            ctrl: Control input with shape ``(num_envs, nu)``.
            nsteps: Number of physics substeps.

        Returns:
            Optional dictionary. Backends may include a ``"timing"`` key with
            per-phase timings in milliseconds.
        """

    def set_pre_step_control(self, fn: PreStepControlFn | None) -> None:
        """Register an env-owned policy-control to physics-control converter.

        The callback receives ``(backend, ctrl)`` so owner code can read the
        backend's freshly-updated sensor contract before every physics substep.
        It must return backend-native actuator control with the same shape.
        Position-actuator envs leave this unset and keep the direct control path.
        """
        self._pre_step_control_fn = fn

    def _apply_pre_step_control(self, ctrl: np.ndarray) -> np.ndarray:
        if self._pre_step_control_fn is None:
            return ctrl
        converted = np.asarray(self._pre_step_control_fn(self, ctrl), dtype=ctrl.dtype)
        if converted.shape != ctrl.shape:
            raise ValueError(
                f"pre-step control must return shape {ctrl.shape}, got {converted.shape}"
            )
        return converted

    @abc.abstractmethod
    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        """Set physics state for selected environments.

        Args:
            env_indices: Environment indices.
            qpos: Position state.
            qvel: Velocity state.
            randomization: Optional backend randomization payload.
        """

    @abc.abstractmethod
    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Return supported domain-randomization capabilities for this backend."""

    def apply_init_randomization(self, plan: InitRandomizationPlan) -> None:
        """Apply cold-path model/materialization randomization."""
        if plan.is_empty():
            return
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support init-lifecycle randomization"
        )

    def materialize(self) -> None:
        """Finalize cold-path backend resources before reset/step."""

    @abc.abstractmethod
    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        """Apply a scheduled interval randomization plan."""

    def apply_body_linear_velocity_delta(
        self,
        body_ids: np.ndarray,
        velocity_delta: np.ndarray,
    ) -> None:
        """Apply a world-frame linear-velocity delta to specific bodies.

        Args:
            body_ids: Body ids whose linear velocities should be perturbed.
            velocity_delta: Velocity delta with shape ``(num_envs, len(body_ids), 3)``.

        Returns:
            None. Backends that support this mutate their pending simulation state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support interval body velocity perturbation"
        )

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
    ) -> None:
        """Apply a world-frame force to specific bodies for the upcoming step.

        Args:
            body_ids: Body ids whose external forces should be perturbed.
            force: Force values with shape ``(num_envs, len(body_ids), 3)``.

        Returns:
            None. Backends that support this mutate their pending simulation state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support interval body force perturbation"
        )

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        """Return backend-native play/render capabilities."""
        return BackendPlayCapabilities()

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        """Resolve high-level playback mode into backend-owned render parameters."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define playback render mode semantics"
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
        """Execute backend-owned playback for an env wrapper."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support playback execution")

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
        """Initialize a backend-native renderer.

        ``headless`` controls whether a native window is opened. ``capture``
        controls whether ``capture_video_frame`` is valid for the renderer.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support native rendering")

    def render(self) -> None:
        """Render one frame through a backend-native interactive renderer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native interactive rendering"
        )

    def capture_video_frame(self) -> np.ndarray:
        """Capture one RGB frame through a backend-native renderer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native video capture"
        )

    def get_physics_state(self) -> np.ndarray:
        """Return a physics snapshot suitable for offline playback/video export."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support physics-state playback"
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the playback model for a specific env when variants exist.

        Args:
            env_index: Optional vectorized environment index.

        Returns:
            The backend model object used by playback tooling.
        """
        return self.model

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint (kp, kd) arrays from the backend model."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support reading actuator gains"
        )

    # ------------------------------------------------------------------ #
    # Base kinematics                                                      #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_base_pos(self) -> np.ndarray:
        """Return base position in the world frame.

        Returns:
            (num_envs, 3)
        """

    @abc.abstractmethod
    def get_base_quat(self) -> np.ndarray:
        """Return base quaternion in the world frame as ``wxyz``.

        Returns:
            (num_envs, 4)
        """

    @abc.abstractmethod
    def get_base_lin_vel(self) -> np.ndarray:
        """Return base linear velocity in the world frame.

        This is the first three dimensions of generalized velocity ``qvel``,
        expressed in world coordinates.

        Returns:
            (num_envs, 3)
        """

    @abc.abstractmethod
    def get_base_ang_vel(self) -> np.ndarray:
        """Return base angular velocity in the world frame.

        This is dimensions 3-5 of generalized velocity ``qvel``, expressed in
        world coordinates. It differs from gyro readings: gyro sensors report
        angular velocity components in the body/sensor local frame, while this
        contract returns world-frame values. Use the matching sensor contract
        when body-frame angular velocity is required.

        Returns:
            (num_envs, 3)
        """

    # ------------------------------------------------------------------ #
    # DOF state                                                            #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_dof_pos(self) -> np.ndarray:
        """Return joint positions, excluding the base.

        Returns:
            (num_envs, num_dof)
        """

    @abc.abstractmethod
    def get_dof_vel(self) -> np.ndarray:
        """Return joint velocities, excluding the base.

        Returns:
            (num_envs, num_dof)
        """

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                        #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body positions in the world frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body quaternions in the world frame as ``wxyz``.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 4)
        """

    def get_body_pose_w(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return selected body positions and quaternions in the world frame."""
        return self.get_body_pos_w(body_ids), self.get_body_quat_w(body_ids)

    @abc.abstractmethod
    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body linear velocities in the world frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    def get_body_vel_w(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return selected body linear and angular velocities in the world frame."""
        return self.get_body_lin_vel_w(body_ids), self.get_body_ang_vel_w(body_ids)

    @abc.abstractmethod
    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body angular velocities in the world frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    def get_body_state_w(
        self, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get selected body position, quaternion, linear velocity, and angular velocity."""
        return (
            self.get_body_pos_w(body_ids),
            self.get_body_quat_w(body_ids),
            self.get_body_lin_vel_w(body_ids),
            self.get_body_ang_vel_w(body_ids),
        )

    def copy_body_state_w(
        self,
        body_ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Copy selected world-frame body state into caller-owned buffers."""
        pos, quat, lin_vel, ang_vel = self.get_body_state_w(body_ids)
        out_pos[...] = pos
        out_quat[...] = quat
        out_lin_vel[...] = lin_vel
        out_ang_vel[...] = ang_vel
        return out_pos, out_quat, out_lin_vel, out_ang_vel

    def get_body_pose_w_rows(
        self, env_ids: np.ndarray, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get selected env rows of world-frame body position and quaternion."""
        rows = np.asarray(env_ids, dtype=np.intp)
        return self.get_body_pos_w(body_ids)[rows], self.get_body_quat_w(body_ids)[rows]

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                     #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body positions in the baselink frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body quaternions in the baselink frame as ``wxyz``.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 4)
        """

    @abc.abstractmethod
    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body linear velocities in the baselink frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body angular velocities in the baselink frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    # ------------------------------------------------------------------ #
    # Kinematics / Jacobian                                                #
    # ------------------------------------------------------------------ #

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve site names to integer ID arrays.

        Args:
            names: Site names.

        Returns:
            ``int32`` ID array with shape ``(len(names),)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_site_ids")

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to DoF indices in velocity space (qvel).

        Args:
            names: Joint names.

        Returns:
            ``int32`` index array with shape ``(len(names),)`` relative to
            the qvel start.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_joint_dof_indices")

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to DoF indices in position space (qpos).

        Only single-DoF joints are supported; free joints are excluded.

        Args:
            names: Joint names.

        Returns:
            ``int32`` index array with shape ``(len(names),)`` relative to
            the joint section of qpos.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_joint_dof_pos_indices"
        )

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to DoF indices in velocity space (qvel).

        Args:
            names: Joint names.

        Returns:
            ``int32`` index array with shape ``(len(names),)`` relative to
            the joint section start.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_joint_dof_vel_indices"
        )

    def get_site_jacobian_w(
        self,
        site_id: int,
        dof_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute world-frame Jacobians for one site and selected DoF columns.

        Args:
            site_id: Integer site ID.
            dof_indices: DoF column indices to extract, with shape ``(n_dof,)``.

        Returns:
            ``(jacp, jacr)`` translation/rotation Jacobians, each with shape
            ``(num_envs, 3, n_dof)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_site_jacobian_w")

    # ------------------------------------------------------------------ #
    # Sensors                                                              #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_sensor_data(self, name: str) -> np.ndarray:
        """Return sensor data.

        Args:
            name: Sensor name.

        Returns:
            Sensor data array.
        """

    def get_sensor_data_rows(self, name: str, env_ids: np.ndarray) -> np.ndarray:
        """Get selected env rows of a sensor array."""
        return self.get_sensor_data(name)[np.asarray(env_ids, dtype=np.intp)]

    def get_sensor_data_batch(self, names: Sequence[str]) -> np.ndarray:
        """Fetch multiple sensors and concatenate their flattened values.

        Args:
            names: Sensor names in output order.

        Returns:
            Array with shape ``(num_envs, total_sensor_values)``.
        """
        sensor_names = tuple(names)
        if not sensor_names:
            return np.empty((self.num_envs, 0), dtype=np.float64)
        values = [np.asarray(self.get_sensor_data(name)) for name in sensor_names]
        flat_values = [value.reshape(value.shape[0], -1) for value in values]
        return np.concatenate(flat_values, axis=1)
