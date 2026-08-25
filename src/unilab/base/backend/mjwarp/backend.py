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
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unilab.utils.rotation import np_quat_apply_inverse_batched

from ..body_state import copy_selected_body_state
from .dependencies import load_mjwarp_dependencies
from .materialization import materialize_mjwarp_scene
from .playback import run_mjwarp_playback, validate_mjwarp_visual_model

_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)
_RESET_SCRATCH_CAPACITY = 128
_RESET_SCRATCH_MIN_BATCH_SIZE = 8 * _RESET_SCRATCH_CAPACITY


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
    step/reset barriers with bounded, statically declared transfers.  Interval
    DR, native rendering, and host-substep-controller
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
        if push_body_name is not None:
            raise NotImplementedError(
                "mjwarp host_numpy profile does not support interval push or external wrench "
                "randomization; remove domain_rand.push_body_name and disable push_robots."
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
        self._reset_scratch_capacity = (
            _RESET_SCRATCH_CAPACITY if self._num_envs >= _RESET_SCRATCH_MIN_BATCH_SIZE else 0
        )
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
        control_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._execute_device_steps(nsteps)
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
    ) -> dict[str, float]:
        """Commit one explicit reset barrier from host staging.

        Callers validate and own the staging source. The helper preserves the
        backend-owned transfer ordering: reset mask/qpos/qvel H2D,
        forward/sync, then cache D2H.
        """

        use_scratch = self._can_use_reset_scratch(len(row_ids))
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
        if randomization is not None and not randomization.is_empty():
            requested = ", ".join(sorted(randomization.requested_terms()))
            raise NotImplementedError(
                "mjwarp host_numpy profile does not support reset domain randomization "
                f"terms: {requested}."
            )
        rows = self._validate_rows(env_indices)
        qpos_array = np.asarray(qpos, dtype=np.float32)
        qvel_array = np.asarray(qvel, dtype=np.float32)
        expected_qpos = (rows.size, self._nq)
        expected_qvel = (rows.size, self._nv)
        if qpos_array.shape != expected_qpos:
            raise ValueError(f"qpos must have shape {expected_qpos}, got {qpos_array.shape}")
        if qvel_array.shape != expected_qvel:
            raise ValueError(f"qvel must have shape {expected_qvel}, got {qvel_array.shape}")
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
        timings = self._execute_host_reset(
            rows,
            self._qpos_cache,
            self._qvel_cache,
            qpos_array,
            qvel_array,
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
        """Advertise no legacy DR until per-world model mutation is effect-tested."""
        return DomainRandomizationCapabilities()

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        raise NotImplementedError(
            "mjwarp host_numpy profile does not support interval randomization; disable "
            "push/body-force/body-velocity terms in the owner YAML."
        )

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

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        mapped = self._mapped_tracked_ids("world-frame body linear velocities", body_ids)
        return self._tracked_linvel_w_all[:, mapped, :]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        mapped = self._mapped_tracked_ids("world-frame body angular velocities", body_ids)
        return self._tracked_angvel_w_all[:, mapped, :]

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
