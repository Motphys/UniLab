"""Host-compatibility implementation of the independent ``mjwarp`` backend.

``mjwarp`` is not a MuJoCo backend mode.  It uploads a CPU MuJoCo model to
``mujoco_warp`` and owns its own device data and host cache.  The cache is
refreshed exactly at explicit step/reset barriers; legacy getters only return
views into that cache and therefore never trigger an implicit Warp ``.numpy``
transfer.
"""

from __future__ import annotations

import gc
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from os import PathLike
from typing import Any, NoReturn

import numpy as np

from unilab.base.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    SimBackend,
    normalize_play_render_mode,
)
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
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unilab.utils.rotation import np_quat_apply_inverse_batched

from ..body_state import copy_selected_body_state
from .dependencies import load_mjwarp_dependencies
from .materialization import materialize_mjwarp_scene
from .playback import run_mjwarp_playback, validate_mjwarp_visual_model
from .randomization import expand_model_fields

_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)
# Reset scratch storage is deliberately bounded.  The original 128-world
# allocation covers the smaller G1 owners, while the 8192-world motion
# tracking owner routinely resets about 250 rows per vector step.  Scale the
# cold-path allocation with the world count and cap it at 512 so that this
# workload stays on the sparse route without making an unbounded per-task
# allocation.  Keeping the minimum at 128 preserves the #1288 route for the
# 1024/2048-world owners.
_RESET_SCRATCH_CAPACITY = 512
_RESET_SCRATCH_MIN_CAPACITY = 128
_RESET_SCRATCH_MIN_BATCH_SIZE = 8 * _RESET_SCRATCH_MIN_CAPACITY
_RESET_SCRATCH_WORLD_FRACTION = 16


def _reset_scratch_capacity_for_batch(num_envs: int) -> int:
    """Choose a bounded, power-of-two scratch capacity on the cold path.

    A fixed shape is required by the captured reset graphs.  The capacity is
    rounded down to a power of two so graph/data shapes stay predictable, and
    is never allowed below the proven 128-world route or above the 512-world
    memory bound.  Small batches retain the eager/full-forward fallback rather
    than paying for a second ``mujoco_warp.Data`` instance.
    """
    if num_envs < _RESET_SCRATCH_MIN_BATCH_SIZE:
        return 0
    target = max(_RESET_SCRATCH_MIN_CAPACITY, num_envs // _RESET_SCRATCH_WORLD_FRACTION)
    power_of_two = 1 << (target.bit_length() - 1)
    return min(_RESET_SCRATCH_CAPACITY, power_of_two)


@contextmanager
def _suspend_gc() -> Iterator[None]:
    """Keep graph finalizers from running inside a new Warp capture."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _cuda_graph_eligibility(warp: Any, device: Any) -> tuple[bool, str | None]:
    """Return the cold-path CUDA graph decision and a fallback diagnostic."""
    if not bool(device.is_cuda):
        return False, "active Warp device is not CUDA"

    try:
        driver_version = warp.get_cuda_driver_version()
    except Exception as exc:
        return False, f"CUDA driver query failed: {type(exc).__name__}: {exc}"
    if driver_version is None:
        return False, "CUDA driver version is unavailable"

    try:
        mempool_enabled = bool(warp.is_mempool_enabled(device))
    except Exception as exc:
        return False, f"CUDA mempool query failed: {type(exc).__name__}: {exc}"

    reasons: list[str] = []
    if tuple(driver_version) < _GRAPH_CAPTURE_MIN_DRIVER:
        reasons.append(f"CUDA driver {driver_version[0]}.{driver_version[1]} is older than 12.4")
    if not mempool_enabled:
        reasons.append("CUDA mempool is disabled")
    if reasons:
        return False, "; ".join(reasons)
    return True, None


class MjwarpBackend(SimBackend):
    """Independent CUDA backend exposed through the host NumPy profile.

    State and control cross the host/device boundary only at explicit
    step/reset barriers with bounded, statically declared transfers.  Reset DR
    writes per-world rows of cold-path-expanded model arrays in place and
    recomputes derived constants with the graded ``set_const*`` family;
    interval DR stages ``xfrc_applied`` pushes/body forces for the next step
    barrier or kicks root velocity through the reset upload+forward path.
    Per-world gravity DR, native rendering, and host-substep-controller
    combinations remain fail-closed. Detached host snapshots support finite
    MuJoCo-based offline recording.
    """

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        add_body_sensors: bool = False,
        **unexpected_kwargs: Any,
    ) -> None:
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"MjwarpBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
        if not isinstance(add_body_sensors, bool):
            raise TypeError(
                f"MjwarpBackend add_body_sensors must be bool, got {type(add_body_sensors).__name__}"
            )
        nconmax = self._require_capacity(nconmax, name="nconmax", default=512)
        njmax = self._require_capacity(njmax, name="njmax", default=512)
        if push_body_name is not None and not isinstance(push_body_name, str):
            raise TypeError(
                "MjwarpBackend push_body_name must be str or None, got "
                f"{type(push_body_name).__name__}"
            )

        deps = load_mjwarp_dependencies()
        device = deps.warp.get_device()
        if not bool(device.is_cuda):
            raise RuntimeError(
                "mjwarp backend requires an active CUDA Warp device; choose a CUDA-capable "
                "host or select the mujoco backend."
            )

        scene_context = materialize_mjwarp_scene(scene, add_body_sensors=add_body_sensors)
        self._scene_cleanup_handle = scene_context.cleanup_handle
        self.scene_model_file = scene_context.diagnostic_model_file
        self.scene_visual_model_file = str(scene.visual_model_file or scene.model_file)
        self._playback_model_validated = False
        self._pre_step_control_fn = None
        self.backend_type = "mjwarp"
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._base_name = base_name
        self._push_body_name = push_body_name
        self._nconmax = nconmax
        self._njmax = njmax
        self._add_body_sensors = add_body_sensors
        self._tracked_body_names = scene_context.tracked_body_names

        self._mujoco = deps.mujoco
        self._mujoco_warp = deps.mujoco_warp
        self._warp = deps.warp
        try:
            self._cpu_model = deps.mujoco.MjModel.from_xml_path(scene_context.source_model_file)
        finally:
            # The materialized source (fragment merge and/or injected tracking
            # sensors) is only needed to compile the model; release the
            # temporary files immediately like the MuJoCo backend does.
            self.cleanup_scene_assets()
        self._cpu_model.opt.timestep = self._sim_dt
        self._device_model = deps.mujoco_warp.put_model(self._cpu_model)
        self._device_data = deps.mujoco_warp.make_data(
            self._cpu_model,
            nworld=self._num_envs,
            # These capacities are owner-configured cold-path physical storage
            # limits.  They are intentionally not inferred or changed during a
            # rollout: a task must select and validate its own safe budget.
            nconmax=nconmax,
            njmax=njmax,
        )

        self._nq = int(self._cpu_model.nq)
        self._nv = int(self._cpu_model.nv)
        self._nu = int(self._cpu_model.nu)
        self._nbody = int(self._cpu_model.nbody)
        self._root_qpos_dim, self._root_qvel_dim = self._root_state_dims()
        self._num_dof_pos = self._nq - self._root_qpos_dim
        self._num_dof_vel = self._nv - self._root_qvel_dim

        self._sensor_slots = self._bind_sensor_slots()
        self._keyframe_qpos = self._bind_keyframes()
        self._body_ids = self._bind_names(deps.mujoco.mjtObj.mjOBJ_BODY, self._nbody)
        self._joint_ids = self._bind_names(
            deps.mujoco.mjtObj.mjOBJ_JOINT,
            int(self._cpu_model.njnt),
        )
        if self._base_name is None:
            self._base_body_id: int | None = None
        else:
            try:
                self._base_body_id = self._body_ids[self._base_name]
            except KeyError as exc:
                raise ValueError(
                    f"Base body {self._base_name!r} not found in mjwarp model"
                ) from exc
        self._geom_ids = self._bind_names(deps.mujoco.mjtObj.mjOBJ_GEOM, int(self._cpu_model.ngeom))
        self._site_ids = self._bind_names(deps.mujoco.mjtObj.mjOBJ_SITE, int(self._cpu_model.nsite))
        self._push_body_id = self._resolve_push_body_id()
        self._interval_root_velocity_qvel_ids = self._resolve_interval_root_velocity_qvel_ids()
        # Per-world DR expansion replaces model arrays, so it must run on the
        # cold path before the first forward and before CUDA graph capture
        # (captured pointers would otherwise keep reading the shared arrays).
        # Later DR writes are in-place ``assign`` uploads into the expanded
        # arrays and therefore stay graph-safe.
        expand_model_fields(self._warp, self._device_model, self._num_envs)
        self._bind_dr_host_mirrors()
        self._xfrc_staging = np.zeros((self._num_envs, self._nbody, 6), dtype=np.float32)
        self._xfrc_pending = False
        self._actuator_names = tuple(
            deps.mujoco.mj_id2name(
                self._cpu_model,
                deps.mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )
            or ""
            for actuator_id in range(self._nu)
        )
        self._actuator_ctrl_range = np.asarray(
            self._cpu_model.actuator_ctrlrange,
            dtype=np.float32,
        ).copy()
        self._joint_range = self._bind_joint_range()

        # All legacy getters below return views into these stable pinned host
        # buffers. They are refreshed only by _refresh_host_cache(), called
        # after a device step or a reset/forward lifecycle barrier. Keeping the
        # Warp storage alive lets D2H copies target the public NumPy cache
        # directly instead of allocating a temporary array on every refresh.
        self._qpos_cache_storage, self._qpos_cache = self._allocate_pinned_host_cache(
            self._device_data.qpos
        )
        self._qvel_cache_storage, self._qvel_cache = self._allocate_pinned_host_cache(
            self._device_data.qvel
        )
        self._time_cache = np.zeros((self._num_envs,), dtype=np.float32)
        self._sensor_cache_storage, self._sensor_cache = self._allocate_pinned_host_cache(
            self._device_data.sensordata
        )
        self._ctrl_staging = np.zeros((self._num_envs, self._nu), dtype=np.float32)
        self._reset_mask_host = np.zeros((self._num_envs,), dtype=np.bool_)
        self._reset_mask_device = deps.warp.zeros(self._num_envs, dtype=bool)
        # A bounded secondary Data avoids running reset-time forward over every
        # production world when only a small row set terminated.  It is built
        # only for batches large enough to amortize the extra graph and copies.
        self._reset_scratch_capacity = _reset_scratch_capacity_for_batch(self._num_envs)
        self._reset_scratch_data: Any | None = None
        self._reset_scratch_mask_device: Any | None = None
        self._reset_scratch_qpos_staging: np.ndarray | None = None
        self._reset_scratch_qvel_staging: np.ndarray | None = None
        self._reset_scratch_sensor_storage: Any | None = None
        self._reset_scratch_sensor_cache: np.ndarray | None = None
        # Tracked-body views are zero-copy slices of _sensor_cache; they must be
        # bound before the first forward barrier refreshes the cache below.
        self._body_id_to_tracked_idx: np.ndarray | None = None
        if self._add_body_sensors:
            self._bind_tracked_body_state()
        # Begin from explicit model defaults, run a forward barrier, and cache
        # the resulting sensors/kinematics.  This avoids an uninitialized host
        # cache before NpEnv's first selected-row reset.
        defaults = np.broadcast_to(
            np.asarray(self._cpu_model.qpos0, dtype=np.float32),
            (self._num_envs, self._nq),
        )
        np.copyto(self._qpos_cache, defaults)
        self._qvel_cache.fill(0.0)
        self._upload(self._device_data.qpos, self._qpos_cache)
        self._upload(self._device_data.qvel, self._qvel_cache)
        self._mujoco_warp.forward(self._device_model, self._device_data)
        self._synchronize()
        self._refresh_host_cache()
        self._initialize_cuda_graphs(device)

    # ------------------------------------------------------------------ #
    # Cold-path model binding                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_capacity(value: int | None, *, name: str, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"mjwarp {name} must be a positive integer, got {value!r}")
        return value

    def _root_state_dims(self) -> tuple[int, int]:
        if int(self._cpu_model.njnt) == 0:
            return 0, 0
        free_joint = int(self._mujoco.mjtJoint.mjJNT_FREE)
        if int(self._cpu_model.jnt_type[0]) == free_joint:
            return 7, 6
        return 0, 0

    def _bind_names(self, object_type: Any, count: int) -> dict[str, int]:
        names: dict[str, int] = {}
        for object_id in range(count):
            name = self._mujoco.mj_id2name(self._cpu_model, object_type, object_id)
            if name is not None:
                names[str(name)] = object_id
        return names

    def _bind_sensor_slots(self) -> dict[str, tuple[int, int]]:
        slots: dict[str, tuple[int, int]] = {}
        sensor_type = self._mujoco.mjtObj.mjOBJ_SENSOR
        for sensor_id in range(int(self._cpu_model.nsensor)):
            name = self._mujoco.mj_id2name(self._cpu_model, sensor_type, sensor_id)
            if name is None:
                continue
            slots[str(name)] = (
                int(self._cpu_model.sensor_adr[sensor_id]),
                int(self._cpu_model.sensor_dim[sensor_id]),
            )
        return slots

    def _bind_keyframes(self) -> dict[str, np.ndarray]:
        keyframes: dict[str, np.ndarray] = {}
        key_type = self._mujoco.mjtObj.mjOBJ_KEY
        for key_id in range(int(self._cpu_model.nkey)):
            name = self._mujoco.mj_id2name(self._cpu_model, key_type, key_id)
            if name is not None:
                keyframes[str(name)] = np.asarray(
                    self._cpu_model.key_qpos[key_id],
                    dtype=np.float32,
                ).copy()
        return keyframes

    def _bind_joint_range(self) -> np.ndarray | None:
        free_joint = int(self._mujoco.mjtJoint.mjJNT_FREE)
        mask = np.asarray(self._cpu_model.jnt_type, dtype=np.int32) != free_joint
        joint_range = np.asarray(self._cpu_model.jnt_range, dtype=np.float32)[mask]
        return None if joint_range.size == 0 else joint_range.copy()

    def _resolve_push_body_id(self) -> int | None:
        """Resolve the interval-push target body on the cold path."""
        body_name = self._push_body_name if self._push_body_name is not None else self._base_name
        if body_name is None:
            return None
        try:
            return self._body_ids[body_name]
        except KeyError as exc:
            raise ValueError(f"Push body {body_name!r} not found in mjwarp model") from exc

    def _resolve_interval_root_velocity_qvel_ids(self) -> tuple[int, int, int] | None:
        """Bind the configured free root's world-linear qvel columns on the cold path."""
        if self._base_name is None or self._base_body_id is None:
            return None
        try:
            layout = self.get_root_state_layout(self._base_name)
        except (NotImplementedError, ValueError):
            return None
        linear_ids = layout.qvel_indices[:3]
        if linear_ids != tuple(range(linear_ids[0], linear_ids[0] + 3)):
            return None
        return int(linear_ids[0]), int(linear_ids[1]), int(linear_ids[2])

    def _bind_dr_host_mirrors(self) -> None:
        """Allocate per-world host mirrors of the DR-writable model fields.

        Reset randomization stages rows into these mirrors and uploads them
        with in-place ``assign`` into the expanded device arrays, so the
        mirrors are also the source of truth for the rows a reset did not
        touch.  Immutable model defaults are kept separately for the delta
        payload terms (``base_mass_delta`` / ``base_com_offset``).
        """
        cpu = self._cpu_model
        num_envs = self._num_envs
        self._default_body_mass = np.asarray(cpu.body_mass, dtype=np.float32).copy()
        self._default_body_ipos = np.asarray(cpu.body_ipos, dtype=np.float32).copy()
        self._dr_body_mass = np.broadcast_to(
            self._default_body_mass, (num_envs, self._nbody)
        ).copy()
        self._dr_body_ipos = np.broadcast_to(
            self._default_body_ipos, (num_envs, self._nbody, 3)
        ).copy()
        self._dr_body_iquat = np.broadcast_to(
            np.asarray(cpu.body_iquat, dtype=np.float32), (num_envs, self._nbody, 4)
        ).copy()
        self._dr_body_inertia = np.broadcast_to(
            np.asarray(cpu.body_inertia, dtype=np.float32), (num_envs, self._nbody, 3)
        ).copy()
        self._dr_dof_armature = np.broadcast_to(
            np.asarray(cpu.dof_armature, dtype=np.float32), (num_envs, self._nv)
        ).copy()
        self._dr_geom_friction = np.broadcast_to(
            np.asarray(cpu.geom_friction, dtype=np.float32), (num_envs, int(cpu.ngeom), 3)
        ).copy()
        self._dr_actuator_gainprm = np.broadcast_to(
            np.asarray(cpu.actuator_gainprm, dtype=np.float32), (num_envs, self._nu, 10)
        ).copy()
        self._dr_actuator_biasprm = np.broadcast_to(
            np.asarray(cpu.actuator_biasprm, dtype=np.float32), (num_envs, self._nu, 10)
        ).copy()

    def _bind_tracked_body_state(self) -> None:
        """Bind zero-copy tracked-body views into the per-step sensor cache.

        Sensor columns follow the ``tracked_body_names`` insertion order from
        the cold-path injection; body ids are rebuilt from the compiled model
        because MjSpec compilation can reorder bodies (same reasoning as the
        MuJoCo backend).
        """
        names = self._tracked_body_names
        if not names:
            raise ValueError(
                "mjwarp add_body_sensors requires at least one named body in the model"
            )
        body_type = self._mujoco.mjtObj.mjOBJ_BODY
        tracked_ids = [self._mujoco.mj_name2id(self._cpu_model, body_type, name) for name in names]
        missing = [name for name, body_id in zip(names, tracked_ids, strict=True) if body_id < 0]
        if missing:
            raise ValueError(
                "Injected mjwarp body tracking sensors reference bodies missing from "
                f"the compiled model: {missing}"
            )
        mapping = np.full(self._nbody, -1, dtype=np.intp)
        for index, body_id in enumerate(tracked_ids):
            mapping[body_id] = index
        self._body_id_to_tracked_idx = mapping
        self._tracked_pos_w_all = self._tracked_sensor_view("track_pos_w", 3)
        self._tracked_quat_w_all = self._tracked_sensor_view("track_quat_w", 4)
        self._tracked_linvel_w_all = self._tracked_sensor_view("track_linvel_w", 3)
        self._tracked_angvel_w_all = self._tracked_sensor_view("track_angvel_w", 3)

    def _tracked_sensor_view(self, prefix: str, dim: int) -> np.ndarray:
        count = len(self._tracked_body_names)
        addresses = []
        for name in self._tracked_body_names:
            sensor_name = f"{prefix}_{name}"
            try:
                address, sensor_dim = self._sensor_slots[sensor_name]
            except KeyError as exc:
                raise ValueError(
                    f"Injected mjwarp tracking sensor {sensor_name!r} is missing from the "
                    "compiled model"
                ) from exc
            if sensor_dim != dim:
                raise ValueError(
                    f"Injected mjwarp tracking sensor {sensor_name!r} has dim {sensor_dim}; "
                    f"expected {dim}"
                )
            addresses.append(address)
        first = addresses[0]
        if addresses != [first + index * dim for index in range(count)]:
            raise ValueError(
                f"Injected mjwarp tracking sensors {prefix}_* are not one contiguous "
                "sensor block in tracked-body order"
            )
        return self._sensor_cache[:, first : first + count * dim].reshape(
            self._num_envs, count, dim
        )

    def _mapped_tracked_ids(self, operation: str, body_ids: np.ndarray) -> np.ndarray:
        mapping = self._body_id_to_tracked_idx
        if mapping is None:
            self._unsupported_body_kinematics(operation)
        mapped = mapping[np.asarray(body_ids, dtype=np.intp)]
        if np.any(mapped < 0):
            raise ValueError(
                f"mjwarp {operation} received body ids without injected tracking sensors: "
                f"{np.asarray(body_ids)[mapped < 0].tolist()}"
            )
        return mapped

    # ------------------------------------------------------------------ #
    # Explicit host-cache barriers                                        #
    # ------------------------------------------------------------------ #

    def _allocate_pinned_host_cache(self, device_array: Any) -> tuple[Any, np.ndarray]:
        """Allocate stable CPU storage for one fixed-shape device-state cache."""
        storage = self._warp.empty(
            device_array.shape,
            dtype=device_array.dtype,
            device="cpu",
            pinned=True,
        )
        return storage, storage.numpy()

    def _refresh_host_cache(self) -> None:
        """Copy all legacy-visible device state at one explicit lifecycle barrier."""
        self._download(self._device_data.qpos, self._qpos_cache_storage)
        self._download(self._device_data.qvel, self._qvel_cache_storage)
        self._download(self._device_data.sensordata, self._sensor_cache_storage)
        self._synchronize()

    def _upload(self, device_array: Any, host_array: np.ndarray) -> None:
        device_array.assign(host_array)

    def _download(self, device_array: Any, host_array: Any) -> None:
        self._warp.copy(host_array, device_array)

    def _synchronize(self) -> None:
        self._warp.synchronize_device()

    def _disable_cuda_graphs(self, reason: str) -> None:
        """Atomically select the eager path and release any captured graphs."""
        self._cuda_graph_enabled = False
        self._step_graph = None
        self._forward_graph = None
        self._reset_graph = None
        self._reset_scratch_reset_graph = None
        self._reset_scratch_forward_graph = None
        self._cuda_graph_disable_reason: str | None = reason

    def _prepare_reset_scratch(self) -> None:
        """Materialize and warm the bounded reset-forward data on the cold path."""
        if self._reset_scratch_capacity == 0 or self._reset_scratch_data is not None:
            return

        capacity = self._reset_scratch_capacity
        data = self._mujoco_warp.make_data(
            self._cpu_model,
            nworld=capacity,
            nconmax=self._nconmax,
            njmax=self._njmax,
        )
        qpos = np.broadcast_to(
            np.asarray(self._cpu_model.qpos0, dtype=np.float32),
            (capacity, self._nq),
        ).copy()
        qvel = np.zeros((capacity, self._nv), dtype=np.float32)
        reset_mask = self._warp.ones(capacity, dtype=bool)
        sensor_storage, sensor_cache = self._allocate_pinned_host_cache(data.sensordata)

        self._reset_scratch_data = data
        self._reset_scratch_mask_device = reset_mask
        self._reset_scratch_qpos_staging = qpos
        self._reset_scratch_qvel_staging = qvel
        self._reset_scratch_sensor_storage = sensor_storage
        self._reset_scratch_sensor_cache = sensor_cache

        # Warm dynamically specialized reset/forward kernels before capture;
        # compiling or allocating from inside a CUDA capture is unsupported.
        self._mujoco_warp.reset_data(self._device_model, data, reset=reset_mask)
        self._upload(data.qpos, qpos)
        self._upload(data.qvel, qvel)
        self._mujoco_warp.forward(self._device_model, data)
        self._synchronize()

    def _initialize_cuda_graphs(self, device: Any) -> None:
        """Capture fixed-address device operations or retain the eager fallback.

        Current uploads mutate existing Warp arrays with ``assign``. Any future
        owner-layer operation that replaces a model or data array must call this
        method afterward so captured pointers cannot become stale.
        """
        self._disable_cuda_graphs("CUDA graph capture has not been initialized")
        eligible, reason = _cuda_graph_eligibility(self._warp, device)
        if not eligible:
            assert reason is not None
            self._cuda_graph_disable_reason = reason
            warnings.warn(
                f"mjwarp CUDA graphs disabled; using eager execution: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        try:
            self._prepare_reset_scratch()
            # Assign only after all captures succeed. This keeps step/reset on
            # one execution mode if any MJWarp operation is not capturable.
            with _suspend_gc(), self._warp.ScopedDevice(device):
                with self._warp.ScopedCapture() as step_capture:
                    self._mujoco_warp.step(self._device_model, self._device_data)
                with self._warp.ScopedCapture() as forward_capture:
                    self._mujoco_warp.forward(self._device_model, self._device_data)
                with self._warp.ScopedCapture() as reset_capture:
                    self._mujoco_warp.reset_data(
                        self._device_model,
                        self._device_data,
                        reset=self._reset_mask_device,
                    )
                reset_scratch_reset_capture = None
                reset_scratch_forward_capture = None
                if self._reset_scratch_data is not None:
                    assert self._reset_scratch_mask_device is not None
                    with self._warp.ScopedCapture() as reset_scratch_reset_capture:
                        self._mujoco_warp.reset_data(
                            self._device_model,
                            self._reset_scratch_data,
                            reset=self._reset_scratch_mask_device,
                        )
                    with self._warp.ScopedCapture() as reset_scratch_forward_capture:
                        self._mujoco_warp.forward(
                            self._device_model,
                            self._reset_scratch_data,
                        )
            step_graph = step_capture.graph
            forward_graph = forward_capture.graph
            reset_graph = reset_capture.graph
            reset_scratch_reset_graph = (
                None if reset_scratch_reset_capture is None else reset_scratch_reset_capture.graph
            )
            reset_scratch_forward_graph = (
                None
                if reset_scratch_forward_capture is None
                else reset_scratch_forward_capture.graph
            )
        except Exception as exc:
            reason = f"capture failed: {type(exc).__name__}: {exc}"
            self._disable_cuda_graphs(reason)
            warnings.warn(
                f"mjwarp CUDA graphs disabled; using eager execution: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._step_graph = step_graph
        self._forward_graph = forward_graph
        self._reset_graph = reset_graph
        self._reset_scratch_reset_graph = reset_scratch_reset_graph
        self._reset_scratch_forward_graph = reset_scratch_forward_graph
        self._cuda_graph_enabled = True
        self._cuda_graph_disable_reason = None

    def _execute_device_steps(self, nsteps: int) -> None:
        """Advance fixed-shape device state through graph replay or eager calls."""
        if self._cuda_graph_enabled:
            assert self._step_graph is not None
            for _ in range(nsteps):
                self._warp.capture_launch(self._step_graph)
            return
        for _ in range(nsteps):
            self._mujoco_warp.step(self._device_model, self._device_data)

    def _execute_device_reset(self) -> None:
        """Clear selected device rows before the host state upload."""
        if self._cuda_graph_enabled:
            assert self._reset_graph is not None
            self._warp.capture_launch(self._reset_graph)
            return
        self._mujoco_warp.reset_data(
            self._device_model,
            self._device_data,
            reset=self._reset_mask_device,
        )

    def _execute_device_forward(self) -> None:
        """Refresh kinematics after the host state upload."""
        if self._cuda_graph_enabled:
            assert self._forward_graph is not None
            self._warp.capture_launch(self._forward_graph)
            return
        self._mujoco_warp.forward(self._device_model, self._device_data)

    def _can_use_reset_scratch(self, num_rows: int) -> bool:
        return (
            self._cuda_graph_enabled
            and 0 < num_rows <= self._reset_scratch_capacity
            and self._reset_scratch_data is not None
            and self._reset_scratch_reset_graph is not None
            and self._reset_scratch_forward_graph is not None
        )

    def _execute_reset_scratch_forward(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> None:
        """Forward reset rows in bounded scratch storage without touching main rows."""
        data = self._reset_scratch_data
        qpos_staging = self._reset_scratch_qpos_staging
        qvel_staging = self._reset_scratch_qvel_staging
        assert data is not None
        assert qpos_staging is not None and qvel_staging is not None
        assert self._reset_scratch_reset_graph is not None
        assert self._reset_scratch_forward_graph is not None

        num_rows = len(qpos)
        np.copyto(qpos_staging[:num_rows], qpos)
        np.copyto(qvel_staging[:num_rows], qvel)
        self._warp.capture_launch(self._reset_scratch_reset_graph)
        self._upload(data.qpos, qpos_staging)
        self._upload(data.qvel, qvel_staging)
        self._warp.capture_launch(self._reset_scratch_forward_graph)

    def _refresh_reset_scratch_cache(self, row_ids: np.ndarray) -> None:
        """Publish scratch sensor rows while retaining complement host-cache rows."""
        data = self._reset_scratch_data
        storage = self._reset_scratch_sensor_storage
        cache = self._reset_scratch_sensor_cache
        assert data is not None and storage is not None and cache is not None
        self._download(data.sensordata, storage)
        self._synchronize()
        self._sensor_cache[row_ids] = cache[: len(row_ids)]

    def _validate_rows(self, env_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_indices, dtype=np.intp)
        if rows.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got shape {rows.shape}")
        if np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError(f"env_indices must be in [0, {self._num_envs}), got {rows}")
        if np.unique(rows).size != rows.size:
            raise ValueError("env_indices must not contain duplicate rows")
        return rows

    # ------------------------------------------------------------------ #
    # SimBackend properties and cold metadata                             #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> Any:
        """Return the backend-owned device model, never a MuJoCo backend model."""
        return self._device_model

    @property
    def num_actuators(self) -> int:
        return self._nu

    @property
    def num_dof_vel(self) -> int:
        return self._num_dof_vel

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._actuator_ctrl_range.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        return self._actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        supported_transmissions = {
            int(self._mujoco.mjtTrn.mjTRN_JOINT),
            int(self._mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        }
        supported_joint_types = {
            int(self._mujoco.mjtJoint.mjJNT_HINGE),
            int(self._mujoco.mjtJoint.mjJNT_SLIDE),
        }
        names: list[str] = []
        for actuator_id, actuator_name in enumerate(self._actuator_names):
            transmission = int(self._cpu_model.actuator_trntype[actuator_id])
            joint_id = int(self._cpu_model.actuator_trnid[actuator_id, 0])
            if transmission not in supported_transmissions or joint_id < 0:
                raise NotImplementedError(
                    "backend 'mjwarp' capability 'actuator target joint' requires a "
                    f"joint transmission; actuator '{actuator_name}' uses "
                    f"transmission type {transmission}"
                )
            if int(self._cpu_model.jnt_type[joint_id]) not in supported_joint_types:
                raise NotImplementedError(
                    "backend 'mjwarp' capability 'actuator target joint' requires a "
                    f"single-DoF joint; actuator '{actuator_name}' targets joint id {joint_id}"
                )
            joint_name = self._mujoco.mj_id2name(
                self._cpu_model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not joint_name:
                raise NotImplementedError(
                    "backend 'mjwarp' capability 'actuator target joint' requires named "
                    f"joints; actuator '{actuator_name}' targets unnamed joint id {joint_id}"
                )
            names.append(str(joint_name))
        return tuple(names)

    def get_scene_model_file(self) -> str | None:
        return self.scene_model_file

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        try:
            return self._keyframe_qpos[name].copy()
        except KeyError as exc:
            available = ", ".join(sorted(self._keyframe_qpos))
            raise ValueError(f"Keyframe {name!r} not found; available: {available}") from exc

    def get_default_qpos(self) -> np.ndarray:
        return np.asarray(self._cpu_model.qpos0, dtype=np.float32).copy()

    def get_default_dof_pos(self) -> np.ndarray:
        return np.asarray(self._cpu_model.qpos0[self._root_qpos_dim :], dtype=np.float32).copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._nv,), dtype=np.float32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        try:
            body_id = self._body_ids[root_body_name]
        except KeyError as exc:
            raise ValueError(f"Body {root_body_name!r} not found in mjwarp model") from exc
        joint_count = int(self._cpu_model.body_jntnum[body_id])
        joint_id = int(self._cpu_model.body_jntadr[body_id])
        free_joint = int(self._mujoco.mjtJoint.mjJNT_FREE)
        if (
            joint_count != 1
            or joint_id < 0
            or int(self._cpu_model.jnt_type[joint_id]) != free_joint
        ):
            raise NotImplementedError(
                "backend 'mjwarp' capability 'root-state layout' requires body "
                f"{root_body_name!r} to own exactly one free joint"
            )
        qpos_start = int(self._cpu_model.jnt_qposadr[joint_id])
        qvel_start = int(self._cpu_model.jnt_dofadr[joint_id])
        return BackendRootStateLayout(
            qpos_indices=tuple(range(qpos_start, qpos_start + 7)),
            qvel_indices=tuple(range(qvel_start, qvel_start + 6)),
        )

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._body_ids[str(name)])
            except KeyError as exc:
                raise ValueError(f"Body {name!r} not found in mjwarp model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_geom_id(self, name: str) -> int:
        try:
            return int(self._geom_ids[name])
        except KeyError as exc:
            raise ValueError(f"Geom {name!r} not found in mjwarp model") from exc

    def get_geom_size(self, name: str) -> np.ndarray:
        return np.asarray(
            self._cpu_model.geom_size[self.get_geom_id(name)], dtype=np.float32
        ).copy()

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        root = int(root_body_id)
        if root < 0 or root >= self._nbody:
            raise ValueError(f"root_body_id must be in [0, {self._nbody}), got {root}")
        descendants = {root}
        changed = True
        parent_ids = np.asarray(self._cpu_model.body_parentid, dtype=np.int32)
        while changed:
            changed = False
            for body_id, parent_id in enumerate(parent_ids):
                if body_id not in descendants and int(parent_id) in descendants:
                    descendants.add(body_id)
                    changed = True
        return np.asarray(sorted(descendants), dtype=np.int32)

    def get_geom_names(self) -> tuple[str, ...]:
        names = [""] * int(self._cpu_model.ngeom)
        for name, geom_id in self._geom_ids.items():
            names[geom_id] = name
        return tuple(names)

    def get_geom_body_ids(self) -> np.ndarray:
        return np.asarray(self._cpu_model.geom_bodyid, dtype=np.int32).copy()

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._cpu_model.geom_contype, dtype=np.int32).copy(),
            np.asarray(self._cpu_model.geom_conaffinity, dtype=np.int32).copy(),
        )

    def get_geom_friction(self) -> np.ndarray:
        return np.asarray(self._cpu_model.geom_friction, dtype=np.float32).copy()

    def get_gravity(self) -> np.ndarray:
        return np.asarray(self._cpu_model.opt.gravity, dtype=np.float32).copy()

    def get_body_mass(self) -> np.ndarray:
        return np.asarray(self._cpu_model.body_mass, dtype=np.float32).copy()

    def get_body_ipos(self) -> np.ndarray:
        return np.asarray(self._cpu_model.body_ipos, dtype=np.float32).copy()

    def get_dof_armature(self) -> np.ndarray:
        return np.asarray(self._cpu_model.dof_armature, dtype=np.float32).copy()

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_joint_range(self) -> np.ndarray | None:
        return None if self._joint_range is None else self._joint_range.copy()

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._site_ids[str(name)])
            except KeyError as exc:
                raise ValueError(f"Site {name!r} not found in mjwarp model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joint qvel coordinates on the cold metadata path."""

        resolved: list[int] = []
        for name in names:
            try:
                joint_id = self._joint_ids[str(name)]
            except KeyError as exc:
                raise ValueError(f"Joint {name!r} not found in mjwarp model") from exc
            resolved.append(int(self._cpu_model.jnt_dofadr[joint_id]))
        return np.asarray(resolved, dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named single-DoF qpos coordinates excluding the free root."""

        single_dof_types = {
            int(self._mujoco.mjtJoint.mjJNT_HINGE),
            int(self._mujoco.mjtJoint.mjJNT_SLIDE),
        }
        resolved: list[int] = []
        for name in names:
            try:
                joint_id = self._joint_ids[str(name)]
            except KeyError as exc:
                raise ValueError(f"Joint {name!r} not found in mjwarp model") from exc
            if int(self._cpu_model.jnt_type[joint_id]) not in single_dof_types:
                raise ValueError(f"Joint {name!r} is not a single-DoF joint")
            resolved.append(int(self._cpu_model.jnt_qposadr[joint_id]) - self._root_qpos_dim)
        return np.asarray(resolved, dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joint qvel coordinates excluding the free root."""

        return self.get_joint_dof_indices(names) - self._root_qvel_dim

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joints to full reset qpos columns."""
        return self.get_joint_dof_pos_indices(names) + self._root_qpos_dim

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joints to full reset qvel columns."""
        return self.get_joint_dof_vel_indices(names) + self._root_qvel_dim

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Expose immutable model defaults; this does not advertise gain DR support."""
        kp = np.asarray(self._cpu_model.actuator_gainprm[:, 0], dtype=np.float32).copy()
        kd = np.asarray(-self._cpu_model.actuator_biasprm[:, 2], dtype=np.float32).copy()
        return kp, kd

    def _execute_host_step(
        self,
        ctrl: np.ndarray,
        nsteps: int,
    ) -> dict[str, float]:
        """Execute the owner-layer host-cache barrier for one legacy step."""
        t0 = time.perf_counter()
        np.copyto(self._ctrl_staging, ctrl)
        self._upload(self._device_data.ctrl, self._ctrl_staging)
        if self._xfrc_pending:
            # Staged interval push/body forces apply for the whole upcoming
            # step (all substeps), matching the MuJoCo backend's pending
            # ``xfrc_applied`` semantics.
            self._upload(self._device_data.xfrc_applied, self._xfrc_staging)
        control_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._execute_device_steps(nsteps)
        if self._xfrc_pending:
            self._xfrc_staging.fill(0.0)
            self._upload(self._device_data.xfrc_applied, self._xfrc_staging)
            self._xfrc_pending = False
        self._synchronize()
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_host_cache()
        self._time_cache += np.float32(nsteps * self._sim_dt)
        host_cache_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "control_upload_ms": control_upload_ms,
            "physics_ms": physics_ms,
            "host_cache_refresh_ms": host_cache_ms,
        }

    def _execute_host_reset(
        self,
        row_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        reset_qpos: np.ndarray,
        reset_qvel: np.ndarray,
        *,
        force_full_forward: bool = False,
    ) -> dict[str, float]:
        """Commit one explicit reset barrier from host staging.

        Callers validate and own the staging source. The helper preserves the
        backend-owned transfer ordering: reset mask/qpos/qvel H2D,
        forward/sync, then cache D2H.  ``force_full_forward`` bypasses the
        bounded scratch route: reset-time model randomization recomputes
        derived constants on the main device data, and the scratch worlds
        would index per-world model rows with scratch-local world ids.
        """

        use_scratch = self._can_use_reset_scratch(len(row_ids)) and not force_full_forward
        t0 = time.perf_counter()
        self._reset_mask_host.fill(False)
        self._reset_mask_host[row_ids] = True
        self._upload(self._reset_mask_device, self._reset_mask_host)
        self._execute_device_reset()
        # Full-cache uploads are intentional for the host compatibility
        # profile: they preserve complement worlds after reset_data cleared
        # selected transient state, while keeping all D2H materialization at
        # one explicit barrier.
        self._upload(self._device_data.qpos, qpos)
        self._upload(self._device_data.qvel, qvel)
        if use_scratch:
            self._execute_reset_scratch_forward(reset_qpos, reset_qvel)
        reset_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if not use_scratch:
            self._execute_device_forward()
        self._synchronize()
        reset_forward_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if use_scratch:
            self._refresh_reset_scratch_cache(row_ids)
        else:
            self._refresh_host_cache()
        self._time_cache[row_ids] = 0.0
        host_cache_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "reset_upload_ms": reset_upload_ms,
            "reset_forward_ms": reset_forward_ms,
            "host_cache_refresh_ms": host_cache_ms,
        }

    def set_pre_step_control(self, fn: Any | None) -> None:
        if fn is not None:
            raise NotImplementedError(
                "mjwarp host_numpy profile rejects host pre-step callbacks; a per-substep "
                "CPU callback would introduce an implicit device synchronization."
            )
        self._pre_step_control_fn = None

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
        if isinstance(nsteps, bool) or int(nsteps) <= 0:
            raise ValueError(f"nsteps must be a positive integer, got {nsteps!r}")
        ctrl_array = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, self._nu)
        if ctrl_array.shape != expected:
            raise ValueError(f"ctrl must have shape {expected}, got {ctrl_array.shape}")
        timings = self._execute_host_step(ctrl_array, int(nsteps))
        return {"timing": timings}

    # All backends report the same set_state key set for column stability;
    # sub-keys that don't apply to the mjwarp host profile report 0.0.
    _SET_STATE_TIMING_ZERO_KEYS = (
        "set_state_mask_ms",
        "set_state_data_slice_ms",
        "set_state_data_reset_ms",
        "set_state_clear_forces_ms",
        "set_state_geom_overrides_ms",
        "set_state_reset_rand_ms",
        "set_state_set_dof_vel_ms",
        "set_state_set_dof_pos_ms",
        "set_state_actuator_ctrl_ms",
        "set_state_forward_kinematic_ms",
        "set_state_refresh_pose_cache_ms",
        "set_state_invalidate_velocity_ms",
        "set_state_qpos_convert_ms",
        "set_state_pool_reset_ms",
        "set_state_state_scatter_ms",
    )

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict[str, dict[str, float]]:
        rows = self._validate_rows(env_indices)
        qpos_array = np.asarray(qpos, dtype=np.float32)
        qvel_array = np.asarray(qvel, dtype=np.float32)
        expected_qpos = (rows.size, self._nq)
        expected_qvel = (rows.size, self._nv)
        if qpos_array.shape != expected_qpos:
            raise ValueError(f"qpos must have shape {expected_qpos}, got {qpos_array.shape}")
        if qvel_array.shape != expected_qvel:
            raise ValueError(f"qvel must have shape {expected_qvel}, got {qvel_array.shape}")
        if randomization is not None and not randomization.is_empty():
            unsupported = self.get_dr_capabilities().get_unsupported_reset_terms(
                randomization.requested_terms()
            )
            if unsupported:
                requested = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    "mjwarp host_numpy profile does not support reset domain randomization "
                    f"terms: {requested}."
                )
        timing: dict[str, float] = {key: 0.0 for key in self._SET_STATE_TIMING_ZERO_KEYS}
        timing.update(
            {
                "set_state_reset_upload_ms": 0.0,
                "set_state_reset_forward_ms": 0.0,
                "set_state_host_cache_refresh_ms": 0.0,
                "set_state_internal_gap_ms": 0.0,
            }
        )
        if rows.size == 0:
            return {"timing": timing}

        outer_t0 = time.perf_counter()
        self._qpos_cache[rows] = qpos_array
        self._qvel_cache[rows] = qvel_array
        has_model_dr = False
        if randomization is not None and not randomization.is_empty():
            t0 = time.perf_counter()
            has_model_dr = self._apply_reset_randomization(rows, randomization)
            self._synchronize()
            timing["set_state_reset_rand_ms"] = (time.perf_counter() - t0) * 1000.0
        # reset_data clears device xfrc_applied on the reset rows; keep the
        # staged host mirror consistent so a pending push cannot resurrect.
        self._xfrc_staging[rows] = 0.0
        timings = self._execute_host_reset(
            rows,
            self._qpos_cache,
            self._qvel_cache,
            qpos_array,
            qvel_array,
            force_full_forward=has_model_dr,
        )
        timing["set_state_reset_upload_ms"] = timings["reset_upload_ms"]
        timing["set_state_reset_forward_ms"] = timings["reset_forward_ms"]
        timing["set_state_host_cache_refresh_ms"] = timings["host_cache_refresh_ms"]
        outer_total_ms = (time.perf_counter() - outer_t0) * 1000.0
        measured_ms = (
            timing["set_state_reset_upload_ms"]
            + timing["set_state_reset_forward_ms"]
            + timing["set_state_host_cache_refresh_ms"]
        )
        timing["set_state_internal_gap_ms"] = outer_total_ms - measured_ms
        return {"timing": timing}

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Advertise the per-world model mutation set validated by effect tests."""
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                {
                    RESET_TERM_BASE_MASS,
                    RESET_TERM_BASE_COM,
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
            supports_interval_push=self._push_body_id is not None,
            supports_interval_body_velocity_delta=(
                self._interval_root_velocity_qvel_ids is not None
            ),
            supports_interval_body_force=True,
        )

    # ------------------------------------------------------------------ #
    # Reset domain randomization: per-world model row writes              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_dr_field(
        name: str,
        values: np.ndarray,
        num_reset: int,
        shaped_tail: tuple[int, ...],
    ) -> np.ndarray:
        """Accept the flat or shaped per-row layout, matching the MuJoCo backend."""
        array = np.asarray(values, dtype=np.float32)
        flat_tail = int(np.prod(shaped_tail)) if shaped_tail else 1
        shaped = (num_reset, *shaped_tail)
        if array.shape == shaped:
            return np.ascontiguousarray(array)
        if array.shape == (num_reset, flat_tail):
            return np.ascontiguousarray(array.reshape(shaped))
        raise ValueError(
            f"{name} must have shape {shaped} or {(num_reset, flat_tail)}, got {array.shape}"
        )

    def _require_dr_base_body(self, term: str) -> int:
        if self._base_body_id is None:
            raise ValueError(
                f"mjwarp reset randomization term {term!r} requires base_name to "
                "identify the base body"
            )
        return self._base_body_id

    def _apply_reset_randomization(
        self,
        rows: np.ndarray,
        randomization: ResetRandomizationPayload,
    ) -> bool:
        """Stage payload rows into the per-world host mirrors and upload in place.

        Writes are whole-array ``assign`` uploads of the expanded model fields,
        which keeps the captured CUDA graphs valid.  Derived constants are
        recomputed with the graded ``set_const*`` family *before* the kp/kd
        upload so its dampratio re-resolution cannot clobber literal gain
        writes.  Returns True whenever model rows changed, which forces the
        full-forward reset route.
        """
        num_reset = rows.size
        needs_set_const = False
        needs_set_const_0 = False
        wrote_fields: list[str] = []

        if randomization.body_mass is not None:
            self._dr_body_mass[rows] = self._coerce_dr_field(
                "body_mass", randomization.body_mass, num_reset, (self._nbody,)
            )
            needs_set_const = True
            wrote_fields.append("body_mass")
        if randomization.base_mass_delta is not None:
            base_id = self._require_dr_base_body("base_mass_delta")
            delta = np.asarray(randomization.base_mass_delta, dtype=np.float32)
            if delta.shape != (num_reset,):
                raise ValueError(
                    f"base_mass_delta must have shape ({num_reset},), got {delta.shape}"
                )
            if randomization.body_mass is not None:
                self._dr_body_mass[rows, base_id] += delta
            else:
                self._dr_body_mass[rows, base_id] = self._default_body_mass[base_id] + delta
                wrote_fields.append("body_mass")
            needs_set_const = True

        if randomization.body_ipos is not None:
            self._dr_body_ipos[rows] = self._coerce_dr_field(
                "body_ipos", randomization.body_ipos, num_reset, (self._nbody, 3)
            )
            needs_set_const = True
            wrote_fields.append("body_ipos")
        if randomization.base_com_offset is not None:
            base_id = self._require_dr_base_body("base_com_offset")
            offset = np.asarray(randomization.base_com_offset, dtype=np.float32)
            if offset.shape != (num_reset, 3):
                raise ValueError(
                    f"base_com_offset must have shape ({num_reset}, 3), got {offset.shape}"
                )
            if randomization.body_ipos is not None:
                self._dr_body_ipos[rows, base_id, :] += offset
            else:
                self._dr_body_ipos[rows, base_id, :] = self._default_body_ipos[base_id] + offset
                wrote_fields.append("body_ipos")
            needs_set_const = True

        if randomization.body_iquat is not None:
            self._dr_body_iquat[rows] = self._coerce_dr_field(
                "body_iquat", randomization.body_iquat, num_reset, (self._nbody, 4)
            )
            needs_set_const = True
            wrote_fields.append("body_iquat")
        if randomization.body_inertia is not None:
            self._dr_body_inertia[rows] = self._coerce_dr_field(
                "body_inertia", randomization.body_inertia, num_reset, (self._nbody, 3)
            )
            needs_set_const_0 = True
            wrote_fields.append("body_inertia")
        if randomization.dof_armature is not None:
            self._dr_dof_armature[rows] = self._coerce_dr_field(
                "dof_armature", randomization.dof_armature, num_reset, (self._nv,)
            )
            needs_set_const_0 = True
            wrote_fields.append("dof_armature")
        if randomization.geom_friction is not None:
            self._dr_geom_friction[rows] = self._coerce_dr_field(
                "geom_friction",
                randomization.geom_friction,
                num_reset,
                (int(self._cpu_model.ngeom), 3),
            )
            wrote_fields.append("geom_friction")

        for field in wrote_fields:
            mirror = getattr(self, f"_dr_{field}")
            self._upload(getattr(self._device_model, field), mirror)
        if needs_set_const:
            self._mujoco_warp.set_const(self._device_model, self._device_data)
        elif needs_set_const_0:
            self._mujoco_warp.set_const_0(self._device_model, self._device_data)

        if randomization.kp is not None:
            kp = self._coerce_dr_field("kp", randomization.kp, num_reset, (self._nu,))
            self._dr_actuator_gainprm[rows, :, 0] = kp
            self._upload(self._device_model.actuator_gainprm, self._dr_actuator_gainprm)
        if randomization.kd is not None:
            kd = self._coerce_dr_field("kd", randomization.kd, num_reset, (self._nu,))
            self._dr_actuator_biasprm[rows, :, 2] = kd
            self._upload(self._device_model.actuator_biasprm, self._dr_actuator_biasprm)
        return True

    # ------------------------------------------------------------------ #
    # Interval domain randomization                                       #
    # ------------------------------------------------------------------ #

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
            self._apply_body_linear_velocity_delta(plan.body_ids, plan.body_linear_velocity_delta)

    def push_robots(self, force_range: Sequence[float] | np.ndarray) -> None:
        """Sample one world-frame push force per env and stage it for the next step."""
        if self._push_body_id is None:
            raise NotImplementedError(
                "mjwarp interval push requires base_name or push_body_name to identify "
                "a push target body"
            )
        limit = np.asarray(force_range, dtype=np.float32)
        if limit.shape != (3,) or not np.isfinite(limit).all():
            raise ValueError(f"push force_range must be a finite (3,) array, got {limit.shape}")
        self._xfrc_staging.fill(0.0)
        sampled = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3)).astype(np.float32)
        self._xfrc_staging[:, self._push_body_id, 0:3] = sampled * limit
        self._xfrc_pending = True

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Accumulate world-frame forces on the staged ``xfrc_applied`` rows."""
        if torque is not None:
            raise NotImplementedError("mjwarp backend does not support interval body torque yet")
        body_ids_np = np.asarray(body_ids, dtype=np.intp).reshape(-1)
        if np.any(body_ids_np < 0) or np.any(body_ids_np >= self._nbody):
            raise ValueError(f"body_ids must be in [0, {self._nbody}), got {body_ids_np}")
        force_np = np.asarray(force, dtype=np.float32)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
        if force_np.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {force_np.shape}")
        if not np.isfinite(force_np).all():
            raise ValueError("body force contains NaN or Inf")
        for body_offset, body_id in enumerate(body_ids_np):
            self._xfrc_staging[:, int(body_id), 0:3] += force_np[:, body_offset, :]
        self._xfrc_pending = True

    def _apply_body_linear_velocity_delta(
        self,
        body_ids: np.ndarray,
        velocity_delta: np.ndarray,
    ) -> None:
        """Apply a row-selective world-frame velocity kick to the configured free root."""
        qvel_ids = self._interval_root_velocity_qvel_ids
        if qvel_ids is None:
            raise NotImplementedError(
                "mjwarp interval body velocity perturbation requires base_name to identify "
                "a body with exactly one free joint"
            )

        raw_body_ids = np.asarray(body_ids)
        if (
            raw_body_ids.ndim != 1
            or not np.issubdtype(raw_body_ids.dtype, np.integer)
            or np.issubdtype(raw_body_ids.dtype, np.bool_)
        ):
            raise TypeError(
                "mjwarp interval body velocity perturbation body_ids must be a 1-D "
                f"integer array, got shape={raw_body_ids.shape}, dtype={raw_body_ids.dtype}"
            )
        resolved_body_ids = np.asarray(raw_body_ids, dtype=np.int32)
        expected_body_ids = np.asarray([self._base_body_id], dtype=np.int32)
        if not np.array_equal(resolved_body_ids, expected_body_ids):
            raise NotImplementedError(
                "mjwarp interval body velocity perturbation only supports the configured "
                f"free root body {self._base_name!r} (id={self._base_body_id}); "
                f"received body_ids={resolved_body_ids.tolist()}"
            )

        if not isinstance(velocity_delta, np.ndarray):
            raise TypeError(
                "mjwarp interval body velocity perturbation must be an np.ndarray, "
                f"got {type(velocity_delta).__name__}"
            )
        expected_shape = (self._num_envs, 1, 3)
        if velocity_delta.shape != expected_shape:
            raise ValueError(
                "mjwarp interval body velocity perturbation has shape "
                f"{velocity_delta.shape}; expected {expected_shape}"
            )
        if not np.issubdtype(velocity_delta.dtype, np.floating):
            raise TypeError(
                "mjwarp interval body velocity perturbation must have floating dtype, "
                f"got {velocity_delta.dtype}"
            )
        if not np.isfinite(velocity_delta).all():
            raise ValueError("mjwarp interval body velocity perturbation contains NaN or Inf")

        active_rows = np.flatnonzero(np.any(velocity_delta[:, 0, :] != 0.0, axis=1)).astype(
            np.intp,
            copy=False,
        )
        if active_rows.size == 0:
            return

        # Kick the host-cache rows and re-commit them through the existing
        # upload + forward barrier so sensors/kinematics stay coherent.
        qvel_columns = np.asarray(qvel_ids, dtype=np.intp)
        self._qvel_cache[active_rows[:, None], qvel_columns[None, :]] += velocity_delta[
            active_rows, 0, :
        ].astype(np.float32)
        self._upload(self._device_data.qvel, self._qvel_cache)
        self._execute_device_forward()
        self._synchronize()
        self._refresh_host_cache()

    def materialize(self) -> None:
        """Resources are fully materialized during the constructor cold path."""

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(supports_physics_state_playback=True)

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
                num_steps=None,
                output_video=None,
            )
        if mode == "auto":
            raise NotImplementedError(
                "mjwarp playback does not support auto mode; select record or none explicitly."
            )
        if mode == "interactive":
            raise NotImplementedError(
                "mjwarp playback does not support interactive or native rendering; "
                "select record or none."
            )
        if isinstance(play_steps, bool) or play_steps is None or int(play_steps) <= 0:
            raise ValueError(
                "mjwarp record playback requires a positive finite training.play_steps value."
            )
        if output_video is None:
            raise ValueError("mjwarp record playback requires an output video path.")
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
        initialize: Any,
        step: Any,
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Any = None,
        camera_kwargs: dict[str, Any] | None = None,
        extra_data_getter: Any = None,
    ) -> str | None:
        del render_offset_mode
        should_record = bool(record_video) if record_video is not None else output_video is not None
        should_run_headless = bool(headless) if headless is not None else should_record
        return run_mjwarp_playback(
            backend=self,
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            headless=should_run_headless,
            record_video=should_record,
            snapshot_shape=(self._num_envs, 1 + self._nq + self._nv),
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera_kwargs,
            extra_data_getter=extra_data_getter,
        )

    def get_physics_state(self) -> np.ndarray:
        state = np.empty((self._num_envs, 1 + self._nq + self._nv), dtype=np.float32)
        state[:, 0] = self._time_cache
        state[:, 1 : 1 + self._nq] = self._qpos_cache
        state[:, 1 + self._nq :] = self._qvel_cache
        return state

    def get_playback_model(self, env_index: int | None = None) -> str:
        if env_index is not None:
            idx = int(env_index)
            if idx < 0 or idx >= self._num_envs:
                raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        if not self._playback_model_validated:
            self.scene_visual_model_file = validate_mjwarp_visual_model(
                mujoco=self._mujoco,
                physics_model=self._cpu_model,
                model_file=self.scene_visual_model_file,
            )
            self._playback_model_validated = True
        return self.scene_visual_model_file

    # ------------------------------------------------------------------ #
    # Legacy getters: cache views only, never direct Warp transfers       #
    # ------------------------------------------------------------------ #

    def _require_free_root(self, operation: str) -> None:
        if self._root_qpos_dim != 7 or self._root_qvel_dim != 6:
            raise NotImplementedError(
                f"{operation} requires a first free joint; mjwarp host_numpy profile is "
                "currently validated only for floating-base G1 layouts."
            )

    def get_base_pos(self) -> np.ndarray:
        self._require_free_root("get_base_pos")
        return self._qpos_cache[:, 0:3]

    def get_base_quat(self) -> np.ndarray:
        self._require_free_root("get_base_quat")
        return self._qpos_cache[:, 3:7]

    def get_base_lin_vel(self) -> np.ndarray:
        self._require_free_root("get_base_lin_vel")
        return self._qvel_cache[:, 0:3]

    def get_base_ang_vel(self) -> np.ndarray:
        self._require_free_root("get_base_ang_vel")
        return self._qvel_cache[:, 3:6]

    def get_dof_pos(self) -> np.ndarray:
        return self._qpos_cache[:, self._root_qpos_dim :]

    def get_dof_vel(self) -> np.ndarray:
        return self._qvel_cache[:, self._root_qvel_dim :]

    def _unsupported_body_kinematics(self, operation: str) -> NoReturn:
        raise NotImplementedError(
            f"mjwarp host_numpy profile does not expose {operation}; the G1 host adapter "
            "supports base, dof, and configured sensor cache reads, plus tracked body "
            "kinematics when constructed with body_state_required/add_body_sensors."
        )

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        mapped = self._mapped_tracked_ids("world-frame body positions", body_ids)
        return self._tracked_pos_w_all[:, mapped, :]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        mapped = self._mapped_tracked_ids("world-frame body orientations", body_ids)
        return self._tracked_quat_w_all[:, mapped, :]

    def get_body_pose_w_rows(
        self, env_ids: np.ndarray, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather world-frame body pose for selected environments only."""
        rows = np.asarray(env_ids, dtype=np.intp)
        mapped = self._mapped_tracked_ids("world-frame body poses", body_ids)
        return self._tracked_pos_w_all[rows[:, None], mapped], self._tracked_quat_w_all[
            rows[:, None], mapped
        ]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        mapped = self._mapped_tracked_ids("world-frame body linear velocities", body_ids)
        return self._tracked_linvel_w_all[:, mapped, :]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        mapped = self._mapped_tracked_ids("world-frame body angular velocities", body_ids)
        return self._tracked_angvel_w_all[:, mapped, :]

    def get_body_lin_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        """Gather world-frame body linear velocity for selected rows."""
        rows = np.asarray(env_ids, dtype=np.intp)
        mapped = self._mapped_tracked_ids("world-frame body linear velocities", body_ids)
        return self._tracked_linvel_w_all[rows[:, None], mapped]

    def get_body_ang_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        """Gather world-frame body angular velocity for selected rows."""
        rows = np.asarray(env_ids, dtype=np.intp)
        mapped = self._mapped_tracked_ids("world-frame body angular velocities", body_ids)
        return self._tracked_angvel_w_all[rows[:, None], mapped]

    def copy_body_state_w(
        self,
        body_ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mapped = self._mapped_tracked_ids("world-frame body state", body_ids)
        copy_selected_body_state(
            self._tracked_pos_w_all,
            self._tracked_quat_w_all,
            self._tracked_linvel_w_all,
            self._tracked_angvel_w_all,
            mapped,
            out_pos,
            out_quat,
            out_lin_vel,
            out_ang_vel,
        )
        return out_pos, out_quat, out_lin_vel, out_ang_vel

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("base-frame body positions")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("base-frame body orientations")

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Analytical per the SimBackend contract (#1254): world-frame velocity
        # rotated into each body's own frame, matching MuJoCoBackend.
        mapped = self._mapped_tracked_ids("base-frame body linear velocities", body_ids)
        return np_quat_apply_inverse_batched(
            self._tracked_quat_w_all[:, mapped, :],
            self._tracked_linvel_w_all[:, mapped, :],
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        mapped = self._mapped_tracked_ids("base-frame body angular velocities", body_ids)
        return np_quat_apply_inverse_batched(
            self._tracked_quat_w_all[:, mapped, :],
            self._tracked_angvel_w_all[:, mapped, :],
        )

    def get_sensor_data(self, name: str) -> np.ndarray:
        try:
            address, dimension = self._sensor_slots[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._sensor_slots))
            raise ValueError(f"Sensor {name!r} not found; available: {available}") from exc
        return self._sensor_cache[:, address : address + dimension]

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> Callable[[], np.ndarray]:
        """Capture numeric host-cache slots for a zero-metadata hot-path view."""
        slots = tuple(self._sensor_slots[name] for name in names)

        def read() -> np.ndarray:
            values = [
                self._sensor_cache[:, address : address + dimension] for address, dimension in slots
            ]
            return np.concatenate(values, axis=1)

        return read
