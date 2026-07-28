"""Production host-compatibility implementation of the independent ``mjwarp`` backend.

``mjwarp`` is not a MuJoCo backend mode.  It uploads a CPU MuJoCo model to
``mujoco_warp`` and owns its own device data and host cache.  The cache is
refreshed exactly at explicit step/reset barriers; legacy getters only return
views into that cache and therefore never trigger an implicit Warp ``.numpy``
transfer.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from unilab.base.backend.base import SimBackend
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
    RowSelection,
    StateBatchPhase,
)
from unilab.base.backend.mutation import (
    BoundMutationPlan,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationSpec,
    MutationTargetKind,
    MutationTargetSpec,
)
from unilab.base.backend.mutation import (
    bind_mutation_plan as bind_typed_mutation_plan,
)
from unilab.base.backend.mutation_batch import TypedBackendMutationBatch
from unilab.base.backend.telemetry import (
    BackendTransferBuffer,
    BackendTransferCounters,
    BackendTransferProfile,
    BackendTransferTrace,
)
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

from .batch import (
    MjwarpHostBatchPlan,
    bind_mjwarp_host_batch,
    transfer_delta_to_batch_counters,
)
from .dependencies import load_mjwarp_dependencies
from .materialization import materialize_mjwarp_scene
from .mutation import MjwarpHostMutationPlan, mjwarp_host_mutation_capabilities
from .telemetry import MJWARP_HOST_TRANSFER_PROFILE, MjwarpTransferTelemetry


class MjwarpBackend(SimBackend):
    """CUDA-only, host-NumPy compatibility profile for ``mujoco_warp``.

    The profile is intentionally narrow: it provides the state/control/reset
    surface required by ``g1_walk_flat`` while unsupported render, terrain,
    Jacobian, typed model-mutation, interval-DR, and host-substep-controller paths
    remain fail-closed.  It is a correctness migration path, not a performance
    claim or a replacement for the later device-resident executor.
    """

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        **unexpected_kwargs: Any,
    ) -> None:
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"MjwarpBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
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
        self._pre_step_control_fn = None
        self.backend_type = "mjwarp"
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._base_name = base_name
        self._cuda_device_name = str(device)

        self._mujoco = deps.mujoco
        self._mujoco_warp = deps.mujoco_warp
        self._warp = deps.warp
        self._transfer_telemetry = MjwarpTransferTelemetry()
        self._cpu_model = deps.mujoco.MjModel.from_xml_path(scene_context.source_model_file)
        self._cpu_model.opt.timestep = self._sim_dt
        self._device_model = deps.mujoco_warp.put_model(self._cpu_model)
        self._device_data = deps.mujoco_warp.make_data(
            self._cpu_model,
            nworld=self._num_envs,
            # G1 flat regularly has multiple simultaneous contacts.  These
            # capacities are cold-path backend allocation parameters, not task
            # semantics, and avoid silently overflowing the device constraint
            # buffers under a normal standing pose.
            nconmax=512,
            njmax=512,
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
        self._sensor_cache = np.zeros(
            (self._num_envs, int(self._cpu_model.nsensordata)),
            dtype=np.float32,
        )
        self._ctrl_staging = np.zeros((self._num_envs, self._nu), dtype=np.float32)
        self._reset_mask_host = np.zeros((self._num_envs,), dtype=np.bool_)
        self._reset_mask_device = deps.warp.zeros(self._num_envs, dtype=bool)
        self._batch_instance_id = f"mjwarp:{id(self):x}"
        self._host_batch_plans: dict[str, MjwarpHostBatchPlan] = {}
        self._host_mutation_plans: dict[str, MjwarpHostMutationPlan] = {}

        # Begin from explicit model defaults, run a forward barrier, and cache
        # the resulting sensors/kinematics.  This avoids an uninitialized host
        # cache before NpEnv's first selected-row reset.
        defaults = np.broadcast_to(
            np.asarray(self._cpu_model.qpos0, dtype=np.float32),
            (self._num_envs, self._nq),
        )
        np.copyto(self._qpos_cache, defaults)
        self._qvel_cache.fill(0.0)
        self._transfer_telemetry.begin_barrier("init")
        self._upload("qpos", self._device_data.qpos, self._qpos_cache)
        self._upload("qvel", self._device_data.qvel, self._qvel_cache)
        self._mujoco_warp.forward(self._device_model, self._device_data)
        self._synchronize()
        self._refresh_host_cache()

    # ------------------------------------------------------------------ #
    # Cold-path model binding                                             #
    # ------------------------------------------------------------------ #

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
        self._download("qpos", self._device_data.qpos, self._qpos_cache)
        self._download("qvel", self._device_data.qvel, self._qvel_cache)
        self._download("sensordata", self._device_data.sensordata, self._sensor_cache)

    def _upload(self, buffer_name: str, device_array: Any, host_array: np.ndarray) -> None:
        device_array.assign(host_array)
        self._transfer_telemetry.host_to_device(buffer_name, int(host_array.nbytes))

    def _download(self, buffer_name: str, device_array: Any, host_array: np.ndarray) -> None:
        np.copyto(host_array, device_array.numpy())
        self._transfer_telemetry.device_to_host(buffer_name, int(host_array.nbytes))

    def _synchronize(self) -> None:
        self._warp.synchronize_device()
        self._transfer_telemetry.synchronize()

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

    def get_transfer_profile(self) -> BackendTransferProfile:
        """Expose the declared bounded transfers for this host cache profile."""
        return MJWARP_HOST_TRANSFER_PROFILE

    def get_transfer_counters(self) -> BackendTransferCounters:
        """Return telemetry without reading a Warp array or synchronizing the device."""
        return self._transfer_telemetry.counters()

    def get_transfer_buffers(self) -> tuple[BackendTransferBuffer, ...]:
        """Return stable host-cache byte sizes named by the transfer profile."""
        return (
            BackendTransferBuffer("control", int(self._ctrl_staging.nbytes)),
            BackendTransferBuffer("reset_mask", int(self._reset_mask_host.nbytes)),
            BackendTransferBuffer("qpos", int(self._qpos_cache.nbytes)),
            BackendTransferBuffer("qvel", int(self._qvel_cache.nbytes)),
            BackendTransferBuffer("sensordata", int(self._sensor_cache.nbytes)),
        )

    def get_transfer_trace(self) -> BackendTransferTrace:
        """Materialize immutable profiler events only when diagnostics request them."""
        return self._transfer_telemetry.trace()

    def reset_transfer_telemetry(self) -> None:
        """Clear transfer diagnostics; simulator state and stable cache remain untouched."""
        self._transfer_telemetry.reset()

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

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Expose immutable model defaults; this does not advertise gain DR support."""
        kp = np.asarray(self._cpu_model.actuator_gainprm[:, 0], dtype=np.float32).copy()
        kd = np.asarray(-self._cpu_model.actuator_biasprm[:, 2], dtype=np.float32).copy()
        return kp, kd

    # ------------------------------------------------------------------ #
    # Control, reset, and deliberately narrow DR surface                 #
    # ------------------------------------------------------------------ #

    def bind_task_io(self, requirements: BackendIORequirements) -> BoundBackendPlan:
        bound = bind_mjwarp_host_batch(
            self,
            requirements,
            backend_instance_id=self._batch_instance_id,
        )
        existing = self._host_batch_plans.get(bound.public_plan.fingerprint)
        if existing is not None:
            existing.public_plan.require_compatible(bound.public_plan)
            return existing.public_plan
        self._host_batch_plans[bound.public_plan.fingerprint] = bound
        for mutation_plan in self._host_mutation_plans.values():
            mutation_plan.register_batch_plan(bound.public_plan)
        return bound.public_plan

    def _resolve_mjwarp_typed_mutation_selector(
        self,
        spec: MutationTargetSpec,
    ) -> tuple[int, ...]:
        """Resolve raw Warp state coordinates exactly once during plan binding."""

        if spec.selector is None:
            raise MutationContractError("mjwarp typed mutation selector must be explicit")
        if spec.target_kind is not MutationTargetKind.SIMULATION_STATE:
            raise MutationContractError("mjwarp typed mutation selector has an unsupported target")

        if spec.entity_kind is MutationEntityKind.BODY:
            root_targets = {
                "state.root.position": MutationFieldKind.POSITION,
                "state.root.orientation": MutationFieldKind.ORIENTATION,
                "state.root.linear_velocity": MutationFieldKind.LINEAR_VELOCITY,
                "state.root.angular_velocity": MutationFieldKind.ANGULAR_VELOCITY,
            }
            expected_field = root_targets.get(spec.target_key)
            if expected_field is None or spec.field_kind is not expected_field:
                raise MutationContractError(
                    "mjwarp typed body reset only supports floating-root state targets"
                )
            if self._base_body_id is None or (self._root_qpos_dim, self._root_qvel_dim) != (7, 6):
                raise MutationContractError(
                    "mjwarp typed root reset requires a named first free-joint base body"
                )
            try:
                body_id = self._body_ids[spec.selector]
            except KeyError as exc:
                raise MutationContractError(
                    f"mjwarp typed root selector {spec.selector!r} did not resolve a body"
                ) from exc
            if body_id != self._base_body_id:
                raise MutationContractError(
                    f"mjwarp typed root selector {spec.selector!r} must resolve the base body"
                )
            return (body_id,)

        if spec.entity_kind is not MutationEntityKind.DOF:
            raise MutationContractError(
                "mjwarp typed reset selector must name a floating root body or hinge DoF"
            )
        field_targets = {
            "state.dof.position": MutationFieldKind.POSITION,
            "state.dof.angular_velocity": MutationFieldKind.ANGULAR_VELOCITY,
        }
        expected_field = field_targets.get(spec.target_key)
        if expected_field is None or spec.field_kind is not expected_field:
            raise MutationContractError("mjwarp typed DoF reset has an unsupported field kind")
        joint_id = self._mujoco.mj_name2id(
            self._cpu_model,
            self._mujoco.mjtObj.mjOBJ_JOINT,
            spec.selector,
        )
        if joint_id < 0 or int(self._cpu_model.jnt_type[joint_id]) != int(
            self._mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise MutationContractError(
                f"mjwarp typed reset selector {spec.selector!r} must resolve one hinge joint"
            )
        if spec.field_kind is MutationFieldKind.POSITION:
            coordinate = int(self._cpu_model.jnt_qposadr[joint_id]) - self._root_qpos_dim
            count = self._num_dof_pos
        else:
            coordinate = int(self._cpu_model.jnt_dofadr[joint_id]) - self._root_qvel_dim
            count = self._num_dof_vel
        if coordinate < 0 or coordinate >= count:
            raise MutationContractError("mjwarp typed reset coordinate is out of range")
        return (coordinate,)

    def bind_mutation_plan(self, specs: tuple[MutationSpec, ...]) -> BoundMutationPlan:
        """Bind the narrow typed reset contract without exposing raw Warp objects."""

        bound = bind_typed_mutation_plan(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
            num_envs=self._num_envs,
            specs=specs,
            capabilities=mjwarp_host_mutation_capabilities(self),
            resolve_selector=self._resolve_mjwarp_typed_mutation_selector,
        )
        existing = self._host_mutation_plans.get(bound.fingerprint)
        if existing is not None:
            existing.public_plan.require_compatible(bound)
            return existing.public_plan
        runtime_plan = MjwarpHostMutationPlan(
            public_plan=bound,
            nq=self._nq,
            nv=self._nv,
            state_dtype=np.dtype(self._qpos_cache.dtype).name,
            root_qpos_dim=self._root_qpos_dim,
            root_qvel_dim=self._root_qvel_dim,
        )
        for batch_plan in self._host_batch_plans.values():
            runtime_plan.register_batch_plan(batch_plan.public_plan)
        self._host_mutation_plans[bound.fingerprint] = runtime_plan
        return bound

    def _require_host_mutation_plan(
        self,
        mutation_batch: TypedBackendMutationBatch,
        plan: BoundBackendPlan,
    ) -> MjwarpHostMutationPlan:
        mutation_batch.plan.require_owner(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
        )
        try:
            runtime_plan = self._host_mutation_plans[mutation_batch.plan_fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp typed mutation plan was not bound by this backend instance"
            ) from exc
        runtime_plan.public_plan.require_compatible(mutation_batch.plan)
        runtime_plan.require_registered_batch_plan(plan)
        return runtime_plan

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
        bound = self._require_host_batch_plan(plan)
        if not isinstance(rows, RowSelection):
            raise BackendBatchContractError("mjwarp rows must be a RowSelection")
        if rows.universe_size != self._num_envs:
            raise BackendBatchContractError("mjwarp row universe does not match backend num_envs")
        if not isinstance(phase, StateBatchPhase):
            raise BackendBatchContractError("mjwarp state phase must be a StateBatchPhase")
        return bound.materialize(rows, phase)

    def _execute_host_step(
        self,
        ctrl: np.ndarray,
        nsteps: int,
    ) -> tuple[dict[str, float], BackendTransferCounters]:
        """Execute the single owner-layer host-cache barrier used by both APIs."""
        before = self._transfer_telemetry.counters()
        t0 = time.perf_counter()
        np.copyto(self._ctrl_staging, ctrl)
        self._transfer_telemetry.begin_barrier("step")
        self._upload("control", self._device_data.ctrl, self._ctrl_staging)
        control_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(nsteps):
            self._mujoco_warp.step(self._device_model, self._device_data)
        self._synchronize()
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_host_cache()
        host_cache_ms = (time.perf_counter() - t0) * 1000.0
        transfer_delta = self._transfer_telemetry.counters().delta(before)
        return (
            {
                "control_upload_ms": control_upload_ms,
                "physics_ms": physics_ms,
                "host_cache_refresh_ms": host_cache_ms,
            },
            transfer_delta,
        )

    def _execute_host_reset(
        self,
        row_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> tuple[dict[str, float], BackendTransferCounters]:
        """Commit one explicit typed/legacy reset barrier from host staging.

        Callers validate and own the staging source.  This helper intentionally
        has no legacy API dependency, so ``reset_batch`` never delegates to
        ``set_state`` while both paths preserve the same backend-owned transfer
        ordering: reset mask/qpos/qvel H2D, forward/sync, then cache D2H.
        """

        before = self._transfer_telemetry.counters()
        t0 = time.perf_counter()
        self._transfer_telemetry.begin_barrier("reset")
        self._reset_mask_host.fill(False)
        self._reset_mask_host[row_ids] = True
        self._upload("reset_mask", self._reset_mask_device, self._reset_mask_host)
        self._mujoco_warp.reset_data(
            self._device_model,
            self._device_data,
            reset=self._reset_mask_device,
        )
        # Full-cache uploads are intentional for the host compatibility
        # profile: they preserve complement worlds after reset_data cleared
        # selected transient state, while keeping all D2H materialization at
        # one explicit barrier.
        self._upload("qpos", self._device_data.qpos, qpos)
        self._upload("qvel", self._device_data.qvel, qvel)
        reset_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._mujoco_warp.forward(self._device_model, self._device_data)
        self._synchronize()
        reset_forward_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_host_cache()
        host_cache_ms = (time.perf_counter() - t0) * 1000.0
        transfer_delta = self._transfer_telemetry.counters().delta(before)
        return (
            {
                "reset_upload_ms": reset_upload_ms,
                "reset_forward_ms": reset_forward_ms,
                "host_cache_refresh_ms": host_cache_ms,
            },
            transfer_delta,
        )

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
        timings, transfer_delta = self._execute_host_step(control, nsteps)
        read_result = bound.materialize(
            RowSelection.all(self._num_envs),
            StateBatchPhase.TERMINAL,
        )
        diagnostics = BackendBatchDiagnostics(
            counters=transfer_delta_to_batch_counters(
                transfer_delta,
                allocations=bound.step_allocations,
                state_materializations=1,
            ),
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
        bound = self._require_host_batch_plan(plan)
        if not isinstance(rows, RowSelection):
            raise BackendBatchContractError("mjwarp rows must be a RowSelection")
        if rows.universe_size != self._num_envs:
            raise BackendBatchContractError("mjwarp row universe does not match backend num_envs")
        if not isinstance(mutation_batch, TypedBackendMutationBatch):
            raise BackendBatchContractError(
                "mjwarp typed reset requires a TypedBackendMutationBatch"
            )
        if mutation_batch.rows != rows:
            raise BackendBatchContractError(
                "mjwarp typed reset rows must match the mutation envelope"
            )
        mutation_runtime = self._require_host_mutation_plan(mutation_batch, plan)
        qpos, qvel, row_ids = mutation_runtime.stage_reset_state(
            mutation_batch,
            self._qpos_cache,
            self._qvel_cache,
        )

        self._invalidate_host_batch_state()
        timings, transfer_delta = self._execute_host_reset(row_ids, qpos, qvel)
        read_result = bound.materialize(rows, StateBatchPhase.RESET)
        diagnostics = BackendBatchDiagnostics(
            counters=transfer_delta_to_batch_counters(
                transfer_delta,
                allocations=bound.step_allocations,
                state_materializations=1,
            ),
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
        timings, _ = self._execute_host_step(ctrl_array, int(nsteps))
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
        timings, _ = self._execute_host_reset(rows, self._qpos_cache, self._qvel_cache)
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

    def get_playback_model(self, env_index: int | None = None) -> Any:
        del env_index
        raise NotImplementedError(
            "mjwarp host_numpy profile does not support playback model export or rendering."
        )

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
