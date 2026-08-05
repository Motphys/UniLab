"""Host-compatibility implementation of the independent ``mjwarp`` backend.

``mjwarp`` is not a MuJoCo backend mode.  It uploads a CPU MuJoCo model to
``mujoco_warp`` and owns its own device data and host cache.  The cache is
refreshed exactly at explicit step/reset barriers; legacy getters only return
views into that cache and therefore never trigger an implicit Warp ``.numpy``
transfer.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np

from unilab.base.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    SimBackend,
    normalize_play_render_mode,
)
from unilab.base.backend.batch import (
    BackendBatchContractError,
    BackendBatchDiagnostics,
    BackendIORequirements,
    BackendMutationBatch,
    BackendReadResult,
    BackendResetResult,
    BackendStepResult,
    BackendTiming,
    BoundBackendPlan,
    ControlBatch,
    ExecutionProfile,
    RowSelection,
    StateBatchPhase,
)
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

from .batch import (
    MjwarpHostBatchPlan,
    _worst_reset_counters,
    _worst_step_counters,
    bind_mjwarp_host_batch,
)
from .dependencies import load_mjwarp_dependencies
from .materialization import materialize_mjwarp_scene
from .playback import run_mjwarp_playback, validate_mjwarp_visual_model


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
        **unexpected_kwargs: Any,
    ) -> None:
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"MjwarpBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
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

        scene_context = materialize_mjwarp_scene(scene)
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

        self._mujoco = deps.mujoco
        self._mujoco_warp = deps.mujoco_warp
        self._warp = deps.warp
        self._cpu_model = deps.mujoco.MjModel.from_xml_path(scene_context.source_model_file)
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
        self._sensor_ids = self._bind_names(
            deps.mujoco.mjtObj.mjOBJ_SENSOR,
            int(self._cpu_model.nsensor),
        )
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

        # All legacy getters below return views into these stable host buffers.
        # They are refreshed only by _refresh_host_cache(), called after a
        # device step or a reset/forward lifecycle barrier.
        self._qpos_cache = np.zeros((self._num_envs, self._nq), dtype=np.float32)
        self._qvel_cache = np.zeros((self._num_envs, self._nv), dtype=np.float32)
        self._time_cache = np.zeros((self._num_envs,), dtype=np.float32)
        self._sensor_cache = np.zeros(
            (self._num_envs, int(self._cpu_model.nsensordata)),
            dtype=np.float32,
        )
        self._ctrl_staging = np.zeros((self._num_envs, self._nu), dtype=np.float32)
        self._reset_mask_host = np.zeros((self._num_envs,), dtype=np.bool_)
        self._reset_mask_device = deps.warp.zeros(self._num_envs, dtype=bool)
        self._batch_instance_id = f"mjwarp:{id(self):x}"
        self._host_batch_plans: dict[str, MjwarpHostBatchPlan] = {}

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

    # ------------------------------------------------------------------ #
    # Explicit host-cache barriers                                        #
    # ------------------------------------------------------------------ #

    def _refresh_host_cache(self) -> None:
        """Copy all legacy-visible device state at one explicit lifecycle barrier."""
        self._download(self._device_data.qpos, self._qpos_cache)
        self._download(self._device_data.qvel, self._qvel_cache)
        self._download(self._device_data.sensordata, self._sensor_cache)

    def _upload(self, device_array: Any, host_array: np.ndarray) -> None:
        device_array.assign(host_array)

    def _download(self, device_array: Any, host_array: np.ndarray) -> None:
        np.copyto(host_array, device_array.numpy())

    def _synchronize(self) -> None:
        self._warp.synchronize_device()

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

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._nv,), dtype=np.float32)

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._body_ids[str(name)])
            except KeyError as exc:
                raise ValueError(f"Body {name!r} not found in mjwarp model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_sensor_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve exact sensor names from constructor-bound CPU metadata."""

        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._sensor_ids[str(name)])
            except KeyError as exc:
                raise ValueError(f"Sensor {name!r} not found in mjwarp model") from exc
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

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Expose immutable model defaults; this does not advertise gain DR support."""
        kp = np.asarray(self._cpu_model.actuator_gainprm[:, 0], dtype=np.float32).copy()
        kd = np.asarray(-self._cpu_model.actuator_biasprm[:, 2], dtype=np.float32).copy()
        return kp, kd

    # ------------------------------------------------------------------ #
    # Typed host batch control and reset                                  #
    # ------------------------------------------------------------------ #

    def bind_task_io(self, requirements: BackendIORequirements) -> BoundBackendPlan:
        if not isinstance(requirements, BackendIORequirements):
            raise BackendBatchContractError(
                "mjwarp batch requirements must be BackendIORequirements"
            )
        if requirements.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise BackendBatchContractError(
                "mjwarp does not support execution profile "
                f"{requirements.execution_profile.value!r}"
            )
        host_bound = bind_mjwarp_host_batch(
            self,
            requirements,
            backend_instance_id=self._batch_instance_id,
        )
        existing_host = self._host_batch_plans.get(host_bound.public_plan.fingerprint)
        if existing_host is not None:
            existing_host.public_plan.require_compatible(host_bound.public_plan)
            return existing_host.public_plan
        self._host_batch_plans[host_bound.public_plan.fingerprint] = host_bound
        return host_bound.public_plan

    def _require_host_batch_plan(self, plan: BoundBackendPlan) -> MjwarpHostBatchPlan:
        if not isinstance(plan, BoundBackendPlan):
            raise BackendBatchContractError("mjwarp batch plan must be a BoundBackendPlan")
        plan.require_owner(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
        )
        try:
            bound = self._host_batch_plans[plan.fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp batch plan was not bound by this backend instance"
            ) from exc
        bound.public_plan.require_compatible(plan)
        return bound

    def _invalidate_host_batch_state(self) -> None:
        for bound in self._host_batch_plans.values():
            bound.lease.invalidate()

    def read_state_batch(
        self,
        plan: BoundBackendPlan,
        rows: RowSelection,
        *,
        phase: StateBatchPhase = StateBatchPhase.CURRENT,
    ) -> BackendReadResult:
        if not isinstance(rows, RowSelection):
            raise BackendBatchContractError("mjwarp rows must be a RowSelection")
        if rows.universe_size != self._num_envs:
            raise BackendBatchContractError("mjwarp row universe does not match backend num_envs")
        if not isinstance(phase, StateBatchPhase):
            raise BackendBatchContractError("mjwarp state phase must be a StateBatchPhase")
        return self._require_host_batch_plan(plan).materialize(rows, phase)

    def _execute_host_step(
        self,
        ctrl: np.ndarray,
        nsteps: int,
    ) -> dict[str, float]:
        """Execute the single owner-layer host-cache barrier used by both APIs."""
        t0 = time.perf_counter()
        np.copyto(self._ctrl_staging, ctrl)
        self._upload(self._device_data.ctrl, self._ctrl_staging)
        control_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(nsteps):
            self._mujoco_warp.step(self._device_model, self._device_data)
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
    ) -> dict[str, float]:
        """Commit one explicit reset barrier from host staging.

        Callers validate and own the staging source.  This helper intentionally
        has no legacy API dependency, so ``reset_batch`` never delegates to
        ``set_state`` while both paths preserve the same backend-owned transfer
        ordering: reset mask/qpos/qvel H2D, forward/sync, then cache D2H.
        """

        t0 = time.perf_counter()
        self._reset_mask_host.fill(False)
        self._reset_mask_host[row_ids] = True
        self._upload(self._reset_mask_device, self._reset_mask_host)
        self._mujoco_warp.reset_data(
            self._device_model,
            self._device_data,
            reset=self._reset_mask_device,
        )
        # Full-cache uploads are intentional for the host compatibility
        # profile: they preserve complement worlds after reset_data cleared
        # selected transient state, while keeping all D2H materialization at
        # one explicit barrier.
        self._upload(self._device_data.qpos, qpos)
        self._upload(self._device_data.qvel, qvel)
        reset_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._mujoco_warp.forward(self._device_model, self._device_data)
        self._synchronize()
        reset_forward_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_host_cache()
        self._time_cache[row_ids] = 0.0
        host_cache_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "reset_upload_ms": reset_upload_ms,
            "reset_forward_ms": reset_forward_ms,
            "host_cache_refresh_ms": host_cache_ms,
        }

    def step_batch(
        self,
        plan: BoundBackendPlan,
        control_batch: ControlBatch,
        *,
        mutation_batch: BackendMutationBatch | None = None,
        nsteps: int = 1,
    ) -> BackendStepResult:
        bound = self._require_host_batch_plan(plan)
        if self._pre_step_control_fn is not None:
            raise BackendBatchContractError(
                "mjwarp managed host batches do not support pre-step control callbacks"
            )
        if not isinstance(control_batch, ControlBatch):
            raise BackendBatchContractError("mjwarp control must be a ControlBatch")
        plan.require_compatible(control_batch.plan)
        if not control_batch.rows.is_all:
            raise BackendBatchContractError("mjwarp physics steps require controls for all rows")
        if (
            isinstance(nsteps, bool)
            or not isinstance(nsteps, int)
            or nsteps != plan.control.physics_substeps_per_control
        ):
            raise BackendBatchContractError(
                "mjwarp nsteps does not match the bound control cadence"
            )
        if mutation_batch is not None:
            raise BackendBatchContractError(
                "mjwarp host_numpy typed step does not support mutation batches"
            )
        control = control_batch.buffer.handle
        if not isinstance(control, np.ndarray):
            raise BackendBatchContractError("mjwarp host control handle must be a numpy array")
        if control.dtype.name != plan.control.buffer.dtype:
            raise BackendBatchContractError("mjwarp control handle dtype does not match the plan")
        if control.shape != (self._num_envs, *plan.control.buffer.row_shape):
            raise BackendBatchContractError("mjwarp control handle shape does not match the plan")
        if not control.flags.c_contiguous:
            raise BackendBatchContractError("mjwarp host control must be C-contiguous")

        self._invalidate_host_batch_state()
        timings = self._execute_host_step(control, nsteps)
        read_result = bound.materialize(
            RowSelection.all(self._num_envs),
            StateBatchPhase.TERMINAL,
        )
        diagnostics = BackendBatchDiagnostics(
            counters=_worst_step_counters(self),
            timings=(
                *(BackendTiming(phase, milliseconds) for phase, milliseconds in timings.items()),
                *read_result.diagnostics.timings,
            ),
        )
        return BackendStepResult(
            terminal_state=read_result.state,
            diagnostics=diagnostics,
        )

    def reset_batch(
        self,
        plan: BoundBackendPlan,
        rows: RowSelection,
        *,
        mutation_batch: BackendMutationBatch | None = None,
    ) -> BackendResetResult:
        """Reset selected rows to the model defaults."""
        bound = self._require_host_batch_plan(plan)
        if not isinstance(rows, RowSelection):
            raise BackendBatchContractError("mjwarp rows must be a RowSelection")
        if rows.universe_size != self._num_envs:
            raise BackendBatchContractError("mjwarp row universe does not match backend num_envs")
        if mutation_batch is not None:
            raise BackendBatchContractError("mjwarp identity reset does not support mutations")
        row_ids = (
            np.arange(self._num_envs, dtype=np.intp)
            if rows.is_all
            else np.asarray(rows.indices, dtype=np.intp)
        )
        self._qpos_cache[row_ids] = np.asarray(self._cpu_model.qpos0, dtype=np.float32)
        self._qvel_cache[row_ids] = 0.0

        self._invalidate_host_batch_state()
        timings = self._execute_host_reset(row_ids, self._qpos_cache, self._qvel_cache)
        read_result = bound.materialize(rows, StateBatchPhase.RESET)
        diagnostics = BackendBatchDiagnostics(
            counters=_worst_reset_counters(self),
            timings=(
                *(BackendTiming(phase, milliseconds) for phase, milliseconds in timings.items()),
                *read_result.diagnostics.timings,
            ),
        )
        return BackendResetResult(reset_state=read_result.state, diagnostics=diagnostics)

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
        self._invalidate_host_batch_state()
        timings = self._execute_host_step(ctrl_array, int(nsteps))
        return {"timing": timings}

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
        if rows.size == 0:
            return {"timing": {"set_state_reset_ms": 0.0, "set_state_cache_refresh_ms": 0.0}}

        self._invalidate_host_batch_state()
        self._qpos_cache[rows] = qpos_array
        self._qvel_cache[rows] = qvel_array
        timings = self._execute_host_reset(rows, self._qpos_cache, self._qvel_cache)
        return {
            "timing": {
                "set_state_reset_ms": timings["reset_upload_ms"] + timings["reset_forward_ms"],
                "set_state_cache_refresh_ms": timings["host_cache_refresh_ms"],
            }
        }

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

    def _unsupported_body_kinematics(self, operation: str) -> None:
        raise NotImplementedError(
            f"mjwarp host_numpy profile does not expose {operation}; bind it through a typed "
            "state plan after its frame and cache contract are implemented."
        )

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("world-frame body positions")

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("world-frame body orientations")

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("world-frame body linear velocities")

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("world-frame body angular velocities")

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("base-frame body positions")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("base-frame body orientations")

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("base-frame body linear velocities")

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("base-frame body angular velocities")

    def get_sensor_data(self, name: str) -> np.ndarray:
        try:
            address, dimension = self._sensor_slots[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._sensor_slots))
            raise ValueError(f"Sensor {name!r} not found; available: {available}") from exc
        return self._sensor_cache[:, address : address + dimension]
