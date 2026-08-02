"""Production host-compatibility implementation of the independent ``mjwarp`` backend.

``mjwarp`` is not a MuJoCo backend mode.  It uploads a CPU MuJoCo model to
``mujoco_warp`` and owns its own device data and host cache.  The cache is
refreshed exactly at explicit step/reset barriers; legacy getters only return
views into that cache and therefore never trigger an implicit Warp ``.numpy``
transfer.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

import numpy as np
import torch

from unilab.base.backend.base import SimBackend
from unilab.base.backend.batch import (
    BackendBatchContractError,
    BackendBatchCounters,
    BackendBatchDiagnostics,
    BackendIORequirements,
    BackendMutationBatch,
    BackendReadResult,
    BackendResetResult,
    BackendStepResult,
    BackendTiming,
    BoundBackendPlan,
    BufferPlacement,
    ControlBatch,
    ExecutionProfile,
    RowSelection,
    StateBatchPhase,
)
from unilab.base.backend.device import (
    DeviceBufferContractError,
    DeviceBufferLease,
    DeviceTensorView,
    require_device_tensor_view,
)
from unilab.base.backend.graph import (
    DeviceGraphBufferAddress,
    DeviceGraphCaptureKey,
    DeviceGraphDiagnostics,
    DeviceGraphExecutionMode,
)
from unilab.base.backend.mutation import (
    BoundMutationPlan,
    MutationCapabilityManifest,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationGraphInvalidation,
    MutationSelectorMode,
    MutationSpec,
    MutationTargetKind,
    MutationTargetSpec,
)
from unilab.base.backend.mutation import (
    bind_mutation_plan as bind_typed_mutation_plan,
)
from unilab.base.backend.mutation_batch import (
    DeviceResetMutationBatch,
    TypedBackendMutationBatch,
)
from unilab.base.backend.performance import (
    BackendDeviceLifecycleDiagnostics,
    BackendModelFieldDiagnostics,
    BackendModelMaterializationDiagnostics,
    BackendMutationPerformanceDiagnostics,
)
from unilab.base.backend.phase_timing import (
    DevicePhaseTimingError,
    DeviceResetPhaseTimingSampleToken,
    DeviceResetPhaseTimingSession,
)
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
from .device import MjwarpDeviceBatchPlan, bind_mjwarp_device_batch
from .device_mutation import MjwarpDeviceMutationPlan, mjwarp_device_mutation_capabilities
from .materialization import (
    MJWARP_MODEL_INVALIDATIONS,
    MjwarpModelFieldReceipt,
    MjwarpModelFieldRole,
    MjwarpModelInvalidationOutcome,
    MjwarpModelInvalidationReceipt,
    MjwarpModelMaterializationContractError,
    MjwarpModelMaterializationCoordinator,
    MjwarpModelMaterializationReceipt,
    MjwarpModelMaterializationRequest,
    materialize_mjwarp_scene,
)
from .model_mutation import MjwarpDeviceModelMutationPlan
from .mutation import MjwarpHostMutationPlan, mjwarp_host_mutation_capabilities
from .recompute import (
    MjwarpModelRecomputeContract,
    MjwarpModelRecomputeDiagnostics,
    MjwarpModelRecomputeKind,
    MjwarpModelRecomputeRuntime,
    MjwarpModelRecomputeWorkspace,
    compile_model_recompute_contract,
)
from .telemetry import MJWARP_HOST_TRANSFER_PROFILE, MjwarpTransferTelemetry

if TYPE_CHECKING:
    from .armature_recompute import MjwarpArmatureRecomputeWorkspace

_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)


@contextmanager
def _suspend_gc():
    """Prevent stale Warp graph destructors from entering a new capture."""

    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@dataclass(frozen=True)
class _MjwarpDeviceBridge:
    """Cold-bound Torch aliases and stream/event objects for CUDA batches."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    sensordata: torch.Tensor
    reset_mask: torch.Tensor
    physics_stream: torch.cuda.Stream
    warp_physics_stream: Any
    step_event: torch.cuda.Event
    read_event: torch.cuda.Event
    reset_event: torch.cuda.Event


@dataclass(frozen=True)
class _MjwarpDeviceGraphBundle:
    """One reset/forward/step capture transaction under a complete key."""

    key: DeviceGraphCaptureKey
    reset_graph: Any
    forward_graph: Any
    step_graph: Any


@dataclass(frozen=True)
class _MjwarpModelDefaultBaseline:
    """Backend-owned immutable-by-contract device defaults for one direct field."""

    compiled_default: Any
    per_world_default: Any | None


@dataclass(frozen=True)
class _MjwarpStagedModelField:
    """Fully validated allocation staged before any Model pointer is replaced."""

    field_name: str
    original: Any
    materialized: Any
    baseline: _MjwarpModelDefaultBaseline | None
    receipt: MjwarpModelFieldReceipt
    baseline_bytes: int


@dataclass(frozen=True)
class _MjwarpDeviceGraphState:
    """Rollback snapshot for graph identity and diagnostics counters."""

    bundles: dict[str, _MjwarpDeviceGraphBundle]
    storage_generation: int
    storage_buffers: tuple[DeviceGraphBufferAddress, ...]
    storage_fingerprint: str
    storage_poisoned: bool
    captures: int
    launches: int
    recaptures: int
    stale_rejections: int
    eager_fallbacks: int
    storage_verifications: int


@dataclass(frozen=True)
class MjwarpDeviceCapacityDiagnostics:
    """Low-frequency evidence for a cold-bound mjwarp storage budget.

    Reading these values synchronizes and materializes small device counters,
    so this diagnostic is deliberately unavailable to rollout hot paths.
    """

    nconmax_per_world: int
    njmax_per_world: int
    global_contact_capacity: int
    global_contact_count: int
    max_constraints_per_world: int
    overflow_world_count: int
    overflow_mask: int


class MjwarpBackend(SimBackend):
    """Independent CUDA backend with explicit host and device execution profiles.

    ``host_numpy`` remains a bounded-transfer compatibility path.  The separate
    ``device_resident`` batch path exposes only typed CUDA state/control/reset
    buffers and events.  Only mandatory-tested reset-time Model mutations are
    exposed; render, terrain, interval DR, and host-substep-controller
    combinations remain fail-closed.
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
        self._transfer_telemetry = MjwarpTransferTelemetry()
        self._reset_phase_timing_session: DeviceResetPhaseTimingSession | None = None
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
        self._actuator_ids = {
            name: actuator_id for actuator_id, name in enumerate(self._actuator_names) if name
        }
        self._position_actuator_ids = self._bind_position_actuator_ids()
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
        self._device_batch_plans: dict[str, MjwarpDeviceBatchPlan] = {}
        self._host_mutation_plans: dict[str, MjwarpHostMutationPlan] = {}
        self._device_mutation_plans: dict[str, MjwarpDeviceMutationPlan] = {}
        self._device_bridge: _MjwarpDeviceBridge | None = None
        self._model_bridge_cache: dict[str, torch.Tensor] = {}
        self._model_bridge_generation = 0
        self._model_sensor_context: Any | None = None
        self._model_sensor_generation = 0
        self._model_default_baselines: dict[str, _MjwarpModelDefaultBaseline] = {}
        self._expanded_model_fields: frozenset[str] = frozenset()
        self._model_materialization_receipt: MjwarpModelMaterializationReceipt | None = None
        self._model_materialization_in_progress = False
        self._model_materialization_poisoned = False
        self._runtime_barrier_count = 0
        # ``UNTIL_STEP_COMPLETE`` controls are runner-owned.  Keep only a
        # weak epoch watermark per owner lease so replaying an already queued
        # action cannot silently race a subsequent physics barrier, while
        # discarded runner buffers do not accumulate in a long rollout.
        self._device_control_epochs: WeakKeyDictionary[DeviceBufferLease, int] = WeakKeyDictionary()
        self._device_reset_epochs: WeakKeyDictionary[DeviceBufferLease, int] = WeakKeyDictionary()
        self._device_graph_bundles: dict[str, _MjwarpDeviceGraphBundle] = {}
        self._device_graph_captures = 0
        self._device_graph_launches = 0
        self._device_step_graph_launches = 0
        self._device_reset_graph_launches = 0
        self._device_forward_graph_launches = 0
        self._device_state_refreshes = 0
        self._device_graph_recaptures = 0
        self._device_graph_stale_rejections = 0
        self._device_graph_eager_fallbacks = 0
        self._device_graph_storage_verifications = 0
        self._device_graph_storage_generation = 0
        self._device_graph_storage_poisoned = False
        self._device_graph_storage_buffers = self._snapshot_device_graph_storage()
        self._device_graph_storage_fingerprint = self._graph_storage_fingerprint(
            self._device_graph_storage_buffers
        )

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

    def _bind_position_actuator_ids(self) -> tuple[int, ...]:
        """Identify native MuJoCo position servos with the supported slot ABI."""

        model = self._cpu_model
        fixed_gain = int(self._mujoco.mjtGain.mjGAIN_FIXED)
        affine_bias = int(self._mujoco.mjtBias.mjBIAS_AFFINE)
        no_dynamics = int(self._mujoco.mjtDyn.mjDYN_NONE)
        joint_transmission = int(self._mujoco.mjtTrn.mjTRN_JOINT)
        gain = np.asarray(model.actuator_gainprm)
        bias = np.asarray(model.actuator_biasprm)
        supported: list[int] = []
        for actuator_id in range(self._nu):
            gain_row = gain[actuator_id]
            bias_row = bias[actuator_id]
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            if (
                not self._actuator_names[actuator_id]
                or int(model.actuator_gaintype[actuator_id]) != fixed_gain
                or int(model.actuator_biastype[actuator_id]) != affine_bias
                or int(model.actuator_dyntype[actuator_id]) != no_dynamics
                or int(model.actuator_trntype[actuator_id]) != joint_transmission
                or joint_id < 0
                or joint_id >= int(model.njnt)
                or not np.isfinite(gain_row).all()
                or not np.isfinite(bias_row).all()
                or float(gain_row[0]) <= 0.0
                or float(bias_row[1]) != -float(gain_row[0])
                or float(bias_row[2]) > 0.0
                or np.any(gain_row[1:] != 0.0)
                or float(bias_row[0]) != 0.0
                or np.any(bias_row[3:] != 0.0)
            ):
                continue
            supported.append(actuator_id)
        return tuple(supported)

    # ------------------------------------------------------------------ #
    # Cold-path Model field materialization                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _model_field_parent(model: Any, field_name: str) -> tuple[Any, str]:
        parts = field_name.split(".")
        if any(not part for part in parts):
            raise AttributeError(field_name)
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    @classmethod
    def _get_model_field_storage(cls, model: Any, field_name: str) -> Any:
        parent, leaf = cls._model_field_parent(model, field_name)
        return getattr(parent, leaf)

    @classmethod
    def _set_model_field_storage(cls, model: Any, field_name: str, value: Any) -> None:
        parent, leaf = cls._model_field_parent(model, field_name)
        setattr(parent, leaf, value)

    @staticmethod
    def _model_host_fingerprint(value: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {"shape": contiguous.shape, "dtype": contiguous.dtype.str},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(contiguous.tobytes(order="C"))
        return f"mjwarp-model-default-v1:{digest.hexdigest()}"

    def _snapshot_compiled_model_default(
        self,
        field_name: str,
        source_host: np.ndarray,
    ) -> np.ndarray:
        """Clone one final compiled CPU default before any Model mutation."""

        try:
            raw_default = self._get_model_field_storage(self._cpu_model, field_name)
        except AttributeError as exc:
            raise BackendBatchContractError(
                f"mjwarp CPU model has no default field {field_name!r}"
            ) from exc
        default = np.ascontiguousarray(np.asarray(raw_default, dtype=source_host.dtype))
        if field_name == "stat.meaninertia" and default.shape == (1,):
            # MuJoCo exposes mjStatistic scalars as length-one views, while
            # MJWarp uses this array axis for its per-world storage contract.
            default = default.reshape(())
        expected = source_host.shape[1:]
        if default.shape != expected:
            raise BackendBatchContractError(
                f"mjwarp model field {field_name!r} default shape {default.shape} "
                f"does not match Warp storage tail {expected}"
            )
        return default

    def _allocate_model_array(
        self,
        field_name: str,
        source: Any,
        host_values: np.ndarray,
    ) -> Any:
        """Allocate and populate one staged Warp array without publishing it."""

        logical_shape = (int(host_values.shape[0]), *tuple(int(dim) for dim in source.shape[1:]))
        try:
            allocated = self._warp.empty(
                shape=logical_shape,
                dtype=source.dtype,
                device=source.device,
            )
            allocated.assign(np.ascontiguousarray(host_values))
        except Exception as exc:
            raise BackendBatchContractError(
                f"mjwarp failed to allocate staged model field {field_name!r}"
            ) from exc
        return allocated

    @staticmethod
    def _host_rows_equal(left: np.ndarray, right: np.ndarray) -> bool:
        if np.issubdtype(left.dtype, np.inexact):
            return bool(np.array_equal(left, right, equal_nan=True))
        return bool(np.array_equal(left, right))

    def _stage_model_field(
        self,
        request: MjwarpModelMaterializationRequest,
        field_name: str,
    ) -> _MjwarpStagedModelField:
        try:
            source = self._get_model_field_storage(self._device_model, field_name)
        except AttributeError as exc:
            raise BackendBatchContractError(
                f"mjwarp device model has no field {field_name!r}"
            ) from exc
        if not isinstance(source, self._warp.array):
            raise BackendBatchContractError(
                f"mjwarp model field {field_name!r} is not Warp array storage"
            )
        source_shape = tuple(int(dim) for dim in source.shape)
        if not source_shape:
            raise BackendBatchContractError(
                f"mjwarp model field {field_name!r} has no world dimension"
            )
        if source_shape[0] not in {1, self._num_envs}:
            raise BackendBatchContractError(
                f"mjwarp model field {field_name!r} has illegal world dimension "
                f"{source_shape[0]}; expected 1 or {self._num_envs}"
            )

        source_host = np.ascontiguousarray(source.numpy())
        compiled_default = self._snapshot_compiled_model_default(field_name, source_host)
        compiled_world = np.ascontiguousarray(compiled_default[np.newaxis, ...])
        per_world = field_name in request.per_world_default_fields
        if per_world and source_shape[0] != self._num_envs:
            raise BackendBatchContractError(
                f"mjwarp per-world default field {field_name!r} must already have "
                f"{self._num_envs} worlds"
            )
        requires_compiled_match = not per_world and (
            field_name in request.direct_fields or source_shape[0] == 1
        )
        if requires_compiled_match:
            expected_current = np.broadcast_to(
                compiled_world,
                (source_shape[0], *compiled_world.shape[1:]),
            )
            if not self._host_rows_equal(source_host, expected_current):
                raise BackendBatchContractError(
                    f"mjwarp model field {field_name!r} differs from its compiled default "
                    "before baseline materialization"
                )

        if source_shape[0] == 1 and self._num_envs > 1:
            materialized_host = np.ascontiguousarray(
                np.broadcast_to(
                    source_host,
                    (self._num_envs, *source_host.shape[1:]),
                )
            )
            materialized = self._allocate_model_array(
                field_name,
                source,
                materialized_host,
            )
        else:
            materialized_host = source_host
            materialized = source

        baseline: _MjwarpModelDefaultBaseline | None = None
        baseline_bytes = 0
        if field_name in request.direct_fields:
            compiled_device = self._allocate_model_array(
                f"{field_name}.compiled_default",
                source,
                compiled_world,
            )
            per_world_device = None
            if per_world:
                per_world_device = self._allocate_model_array(
                    f"{field_name}.per_world_default",
                    source,
                    source_host,
                )
            baseline = _MjwarpModelDefaultBaseline(
                compiled_default=compiled_device,
                per_world_default=per_world_device,
            )
            baseline_bytes = int(compiled_world.nbytes)
            if per_world_device is not None:
                baseline_bytes += int(source_host.nbytes)

        role = (
            MjwarpModelFieldRole.DIRECT
            if field_name in request.direct_fields
            else MjwarpModelFieldRole.DERIVED
        )
        per_world_shape = tuple(int(dim) for dim in source_host.shape) if per_world else None
        receipt = MjwarpModelFieldReceipt(
            field_name=field_name,
            role=role,
            source_shape=source_shape,
            materialized_shape=tuple(int(dim) for dim in materialized.shape),
            source_address=int(source.ptr or 0),
            materialized_address=int(materialized.ptr or 0),
            replaced=materialized is not source,
            model_bytes=int(materialized_host.nbytes),
            compiled_default_shape=tuple(int(dim) for dim in compiled_default.shape),
            compiled_default_fingerprint=self._model_host_fingerprint(compiled_default),
            per_world_default_shape=per_world_shape,
            per_world_default_fingerprint=(
                self._model_host_fingerprint(source_host) if per_world else None
            ),
        )
        return _MjwarpStagedModelField(
            field_name=field_name,
            original=source,
            materialized=materialized,
            baseline=baseline,
            receipt=receipt,
            baseline_bytes=baseline_bytes,
        )

    def _prepare_model_sensor_context(self, fields: frozenset[str]) -> Any | None:
        """Prepare the optional model-dependent sensor owner before commit.

        UniLab's current mjwarp state sensors consume stable ``Data.sensordata``
        storage and therefore have no separate context.  Keeping this explicit
        preparation boundary makes absence observable and gives future custom
        sensors one transactional owner rather than an ad-hoc cache probe.
        """

        del fields
        return self._model_sensor_context

    def _prepare_model_bridge_cache(
        self,
        staged: tuple[_MjwarpStagedModelField, ...],
        existing: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build aliases for every existing cache entry without publishing them."""

        replacements = {item.field_name: item.materialized for item in staged}
        prepared: dict[str, torch.Tensor] = {}
        for field_name in sorted(existing):
            try:
                model_field = replacements.get(field_name)
                if model_field is None:
                    model_field = self._get_model_field_storage(self._device_model, field_name)
                bridge = self._warp.to_torch(model_field)
            except Exception as exc:
                raise BackendBatchContractError(
                    f"mjwarp failed to rebuild model bridge {field_name!r}"
                ) from exc
            if int(bridge.data_ptr()) != int(model_field.ptr or 0):
                raise BackendBatchContractError(
                    f"mjwarp rebuilt model bridge {field_name!r} does not alias Model storage"
                )
            prepared[field_name] = bridge
        return prepared

    def _get_model_field_bridge(self, field_name: str) -> torch.Tensor:
        """Return a cold-bound Torch alias for a materialized Model field."""

        try:
            return self._model_bridge_cache[field_name]
        except KeyError:
            pass
        if field_name not in self._expanded_model_fields:
            raise BackendBatchContractError(
                f"mjwarp model field {field_name!r} was not materialized"
            )
        try:
            model_field = self._get_model_field_storage(self._device_model, field_name)
        except AttributeError as exc:  # pragma: no cover - guarded by receipt verification.
            raise BackendBatchContractError(
                f"mjwarp materialized field {field_name!r} disappeared"
            ) from exc
        bridge = self._warp.to_torch(model_field)
        self._model_bridge_cache[field_name] = bridge
        return bridge

    def _get_model_default_baseline(
        self,
        field_name: str,
        *,
        per_world: bool,
    ) -> Any:
        """Return backend-owned device baseline storage for a future mutation plan."""

        try:
            baseline = self._model_default_baselines[field_name]
        except KeyError as exc:
            raise BackendBatchContractError(
                f"mjwarp model field {field_name!r} has no materialized default"
            ) from exc
        if not per_world:
            return baseline.compiled_default
        if baseline.per_world_default is None:
            raise BackendBatchContractError(
                f"mjwarp model field {field_name!r} has no per-world variant default"
            )
        return baseline.per_world_default

    def _get_model_default_bridge(self, field_name: str) -> torch.Tensor:
        """Return a cold-bound Torch alias for one immutable compiled default."""

        baseline = self._get_model_default_baseline(field_name, per_world=False)
        bridge = self._warp.to_torch(baseline)
        if int(bridge.data_ptr()) != int(baseline.ptr or 0):
            raise BackendBatchContractError(
                f"mjwarp compiled default bridge {field_name!r} does not alias its owner"
            )
        return bridge

    def _materialize_bound_device_model_fields(
        self,
        contract: MjwarpModelRecomputeContract,
    ) -> MjwarpModelMaterializationReceipt:
        """Freeze only direct and derived fields required by one bound plan."""

        derived_fields = contract.derived_fields
        if contract.kind in {
            MjwarpModelRecomputeKind.SET_CONST_0,
            MjwarpModelRecomputeKind.SET_CONST,
        }:
            # MJWarp declares meaninertia as per-world solver state, but uploads
            # singleton storage.  Keep this internal field in the same atomic
            # cold-path transaction as the public recompute descriptor fields.
            derived_fields = tuple(sorted((*derived_fields, "stat.meaninertia")))
        request = MjwarpModelMaterializationRequest(
            num_worlds=self._num_envs,
            direct_fields=contract.direct_fields,
            derived_fields=derived_fields,
        )
        return self._materialize_model_fields(request)

    def _model_invalidation_receipts(
        self,
        *,
        storage_replaced: bool,
        bridge_entries: int,
        sensor_present: bool,
        graph_keys: int,
    ) -> tuple[MjwarpModelInvalidationReceipt, ...]:
        counts = {
            MutationGraphInvalidation.MODEL_BRIDGE_CACHE: bridge_entries,
            MutationGraphInvalidation.SENSOR_CONTEXT: int(sensor_present),
            MutationGraphInvalidation.STEP_GRAPH: graph_keys,
            MutationGraphInvalidation.FORWARD_GRAPH: graph_keys,
            MutationGraphInvalidation.RESET_GRAPH: graph_keys,
            MutationGraphInvalidation.SENSE_GRAPH: 0,
        }
        receipts: list[MjwarpModelInvalidationReceipt] = []
        for consumer in MJWARP_MODEL_INVALIDATIONS:
            count = counts[consumer]
            if not count:
                outcome = MjwarpModelInvalidationOutcome.NOT_PRESENT
            elif not storage_replaced:
                outcome = MjwarpModelInvalidationOutcome.UNCHANGED
            else:
                outcome = MjwarpModelInvalidationOutcome.REBUILT
            receipts.append(
                MjwarpModelInvalidationReceipt(
                    consumer=consumer,
                    outcome=outcome,
                    affected_count=count if storage_replaced else 0,
                )
            )
        return tuple(receipts)

    def _verify_model_materialization_receipt(
        self,
        receipt: MjwarpModelMaterializationReceipt,
    ) -> None:
        try:
            receipt.verify_fingerprint()
        except MjwarpModelMaterializationContractError as exc:
            self._model_materialization_poisoned = True
            self._device_graph_storage_poisoned = True
            raise BackendBatchContractError(
                "mjwarp model materialization receipt identity is corrupt"
            ) from exc
        self._verify_device_graph_storage()
        for field in receipt.fields:
            try:
                actual = self._get_model_field_storage(
                    self._device_model,
                    field.field_name,
                )
            except AttributeError as exc:
                self._model_materialization_poisoned = True
                raise BackendBatchContractError(
                    f"mjwarp materialized model field {field.field_name!r} disappeared"
                ) from exc
            if (
                tuple(int(dim) for dim in actual.shape) != field.materialized_shape
                or int(actual.ptr or 0) != field.materialized_address
            ):
                self._model_materialization_poisoned = True
                self._device_graph_storage_poisoned = True
                raise BackendBatchContractError(
                    f"mjwarp materialized model field {field.field_name!r} changed address or shape"
                )

    def _materialize_model_fields(
        self,
        request: MjwarpModelMaterializationRequest,
    ) -> MjwarpModelMaterializationReceipt:
        """Delegate the atomic transaction to the cold-path owner module."""

        return MjwarpModelMaterializationCoordinator(self).materialize(request)

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

    def _ensure_device_bridge(self) -> _MjwarpDeviceBridge:
        """Cold-bind stable Torch aliases and one explicit physics stream.

        This method is only reached by ``bind_task_io`` for an explicit
        ``device_resident`` plan.  It never downloads a Warp array and all
        aliases retain backend ownership for the life of the backend instance.
        """

        existing = self._device_bridge
        if existing is not None:
            return existing
        qpos = self._warp.to_torch(self._device_data.qpos)
        qvel = self._warp.to_torch(self._device_data.qvel)
        ctrl = self._warp.to_torch(self._device_data.ctrl)
        sensordata = self._warp.to_torch(self._device_data.sensordata)
        reset_mask = self._warp.to_torch(self._reset_mask_device)
        expected = {
            "qpos": (qpos, (self._num_envs, self._nq)),
            "qvel": (qvel, (self._num_envs, self._nv)),
            "ctrl": (ctrl, (self._num_envs, self._nu)),
            "sensordata": (sensordata, (self._num_envs, int(self._cpu_model.nsensordata))),
        }
        for name, (tensor, shape) in expected.items():
            if tensor.device.type != "cuda" or tensor.dtype is not torch.float32:
                raise BackendBatchContractError(
                    f"mjwarp device bridge {name} must be a CUDA float32 Torch alias"
                )
            if tuple(int(dim) for dim in tensor.shape) != shape or not tensor.is_contiguous():
                raise BackendBatchContractError(
                    f"mjwarp device bridge {name} has an unexpected shape or layout"
                )
        if (
            reset_mask.device != qpos.device
            or reset_mask.dtype is not torch.bool
            or tuple(int(dim) for dim in reset_mask.shape) != (self._num_envs,)
            or not reset_mask.is_contiguous()
        ):
            raise BackendBatchContractError(
                "mjwarp device bridge reset_mask must be a contiguous CUDA bool Torch alias"
            )
        device = qpos.device
        physics_stream = torch.cuda.Stream(device=device)
        bridge = _MjwarpDeviceBridge(
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            sensordata=sensordata,
            reset_mask=reset_mask,
            physics_stream=physics_stream,
            warp_physics_stream=self._warp.stream_from_torch(physics_stream),
            step_event=torch.cuda.Event(enable_timing=False),
            read_event=torch.cuda.Event(enable_timing=False),
            reset_event=torch.cuda.Event(enable_timing=False),
        )
        self._device_bridge = bridge
        return bridge

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

    def capture_device_capacity_diagnostics(self) -> MjwarpDeviceCapacityDiagnostics:
        """Synchronize once on a cold diagnostic path and report capacity headroom.

        This method exists for backend-owner validation of an explicit task
        allocation budget.  It must not be called from env/runner hot paths:
        the Warp ``numpy()`` reads are intentionally observable host
        materialization.
        """

        self._warp.synchronize_device()
        global_contacts = np.asarray(self._device_data.nacon.numpy(), dtype=np.int64)
        constraints = np.asarray(self._device_data.nefc.numpy(), dtype=np.int64)
        overflow = np.asarray(self._device_data.overflow.numpy(), dtype=np.int64)
        return MjwarpDeviceCapacityDiagnostics(
            nconmax_per_world=self._nconmax,
            njmax_per_world=self._njmax,
            global_contact_capacity=int(self._device_data.naconmax),
            global_contact_count=int(global_contacts.max(initial=0)),
            max_constraints_per_world=int(constraints.max(initial=0)),
            overflow_world_count=int(np.count_nonzero(overflow)),
            overflow_mask=int(np.bitwise_or.reduce(overflow, initial=0)),
        )

    def reset_transfer_telemetry(self) -> None:
        """Clear transfer diagnostics; simulator state and stable cache remain untouched."""
        self._transfer_telemetry.reset()

    def create_reset_phase_timing_session(self, *, capacity: int) -> DeviceResetPhaseTimingSession:
        """Preallocate one explicit CUDA reset timing window."""

        if not self._device_batch_plans:
            raise BackendBatchContractError(
                "mjwarp reset phase timing requires a bound device_resident plan"
            )
        existing = self._reset_phase_timing_session
        if existing is not None and not existing.is_materialized:
            raise BackendBatchContractError(
                "mjwarp already owns an unmaterialized reset phase timing session"
            )
        session = DeviceResetPhaseTimingSession(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
            placement=self._device_plan_placement(),
            capacity=capacity,
        )
        self._reset_phase_timing_session = session
        return session

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
    # Control, reset, and deliberately narrow DR surface                 #
    # ------------------------------------------------------------------ #

    def bind_task_io(self, requirements: BackendIORequirements) -> BoundBackendPlan:
        if not isinstance(requirements, BackendIORequirements):
            raise BackendBatchContractError(
                "mjwarp batch requirements must be BackendIORequirements"
            )
        if requirements.execution_profile is ExecutionProfile.HOST_NUMPY:
            if self._device_batch_plans or self._device_mutation_plans:
                raise BackendBatchContractError(
                    "mjwarp cannot mix host_numpy and device_resident batch plans in one backend "
                    "instance; construct a dedicated backend for each explicit profile"
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
            for host_mutation_plan in self._host_mutation_plans.values():
                host_mutation_plan.register_batch_plan(host_bound.public_plan)
            return host_bound.public_plan
        if requirements.execution_profile is ExecutionProfile.DEVICE_RESIDENT:
            if self._host_batch_plans or self._host_mutation_plans:
                raise BackendBatchContractError(
                    "mjwarp cannot mix host_numpy and device_resident batch plans in one backend "
                    "instance; construct a dedicated backend for each explicit profile"
                )
            device_bound = bind_mjwarp_device_batch(
                self,
                requirements,
                backend_instance_id=self._batch_instance_id,
            )
            existing_device = self._device_batch_plans.get(device_bound.public_plan.fingerprint)
            if existing_device is not None:
                existing_device.public_plan.require_compatible(device_bound.public_plan)
                self._ensure_device_graphs(bound=existing_device)
                return existing_device.public_plan

            fingerprint = device_bound.public_plan.fingerprint
            graph_state = self._snapshot_device_graph_state()
            self._device_batch_plans[fingerprint] = device_bound
            try:
                if self._snapshot_device_graph_storage() != self._device_graph_storage_buffers:
                    self._recapture_device_graphs_after_storage_change()
                self._ensure_device_graphs(bound=device_bound)
            except Exception:
                self._device_batch_plans.pop(fingerprint, None)
                self._restore_device_graph_state(graph_state)
                raise
            for device_mutation_plan in self._device_mutation_plans.values():
                device_mutation_plan.register_batch_plan(device_bound.public_plan)
            return device_bound.public_plan
        raise BackendBatchContractError(
            f"mjwarp does not support execution profile {requirements.execution_profile.value!r}"
        )

    def _resolve_mjwarp_typed_mutation_selector(
        self,
        spec: MutationTargetSpec,
    ) -> tuple[int, ...]:
        """Resolve semantic entities exactly once during plan binding."""

        selector = spec.selector_spec
        if selector is None:
            raise MutationContractError("mjwarp typed mutation selector must be explicit")
        if spec.target_kind is MutationTargetKind.MODEL_PARAMETER:
            if selector.mode is not MutationSelectorMode.EXACT:
                raise MutationContractError(
                    "mjwarp typed Model mutation only supports exact mutation selectors"
                )
            expected_pd_field = {
                "actuator.pd_stiffness": MutationFieldKind.STIFFNESS,
                "actuator.pd_damping": MutationFieldKind.DAMPING,
            }.get(spec.target_key)
            if expected_pd_field is not None:
                if (
                    spec.entity_kind is not MutationEntityKind.ACTUATOR
                    or spec.field_kind is not expected_pd_field
                ):
                    raise MutationContractError(
                        "mjwarp typed actuator Model mutation has an invalid semantic target"
                    )
                supported = frozenset(self._position_actuator_ids)
                actuator_ids: list[int] = []
                for raw_selector in selector.expressions:
                    try:
                        actuator_id = self._actuator_ids[raw_selector]
                    except KeyError as exc:
                        raise MutationContractError(
                            f"mjwarp typed actuator selector {raw_selector!r} did not resolve "
                            "an actuator"
                        ) from exc
                    if actuator_id not in supported:
                        raise MutationContractError(
                            f"mjwarp typed actuator selector {raw_selector!r} is not a supported "
                            "native position servo"
                        )
                    actuator_ids.append(actuator_id)
                if len(set(actuator_ids)) != len(actuator_ids):
                    raise MutationContractError(
                        "mjwarp typed actuator selector resolved duplicate actuators"
                    )
                return tuple(actuator_ids)

            if spec.target_key == "joint.armature":
                if (
                    spec.entity_kind is not MutationEntityKind.DOF
                    or spec.field_kind is not MutationFieldKind.ARMATURE
                ):
                    raise MutationContractError(
                        "mjwarp typed armature mutation has an invalid semantic target"
                    )
                armature_coordinates: list[int] = []
                for raw_selector in selector.expressions:
                    joint_id = self._mujoco.mj_name2id(
                        self._cpu_model,
                        self._mujoco.mjtObj.mjOBJ_JOINT,
                        raw_selector,
                    )
                    if joint_id < 0:
                        raise MutationContractError(
                            f"mjwarp typed armature selector {raw_selector!r} did not resolve "
                            "a joint"
                        )
                    if int(self._cpu_model.jnt_type[joint_id]) != int(
                        self._mujoco.mjtJoint.mjJNT_HINGE
                    ):
                        raise MutationContractError(
                            f"mjwarp typed armature selector {raw_selector!r} must resolve "
                            "one hinge joint"
                        )
                    coordinate = int(self._cpu_model.jnt_dofadr[joint_id]) - self._root_qvel_dim
                    if coordinate < 0 or coordinate >= self._num_dof_vel:
                        raise MutationContractError(
                            "mjwarp typed armature coordinate is out of range"
                        )
                    armature_coordinates.append(coordinate)
                if len(set(armature_coordinates)) != len(armature_coordinates):
                    raise MutationContractError(
                        "mjwarp typed armature selector resolved duplicate DoFs"
                    )
                return tuple(armature_coordinates)

            if spec.target_key == "body.gravity_compensation":
                if (
                    spec.entity_kind is not MutationEntityKind.BODY
                    or spec.field_kind is not MutationFieldKind.GRAVITY_COMPENSATION
                ):
                    raise MutationContractError(
                        "mjwarp typed gravity compensation mutation has an invalid semantic target"
                    )
                body_ids: list[int] = []
                for raw_selector in selector.expressions:
                    try:
                        body_id = self._body_ids[raw_selector]
                    except KeyError as exc:
                        raise MutationContractError(
                            f"mjwarp typed gravity compensation selector {raw_selector!r} "
                            "did not resolve a body"
                        ) from exc
                    if body_id <= 0:
                        raise MutationContractError(
                            "mjwarp typed gravity compensation cannot target the world body"
                        )
                    body_ids.append(body_id)
                if len(set(body_ids)) != len(body_ids):
                    raise MutationContractError(
                        "mjwarp typed gravity compensation selector resolved duplicate bodies"
                    )
                return tuple(body_ids)

            raise MutationContractError("mjwarp typed Model mutation target is unsupported")

        if spec.target_kind is not MutationTargetKind.SIMULATION_STATE:
            raise MutationContractError("mjwarp typed mutation selector has an unsupported target")

        if spec.entity_kind is MutationEntityKind.BODY:
            raw_selector = selector.require_exact_singleton(context="mjwarp typed root reset")
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
                body_id = self._body_ids[raw_selector]
            except KeyError as exc:
                raise MutationContractError(
                    f"mjwarp typed root selector {raw_selector!r} did not resolve a body"
                ) from exc
            if body_id != self._base_body_id:
                raise MutationContractError(
                    f"mjwarp typed root selector {raw_selector!r} must resolve the base body"
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
        if selector.mode is not MutationSelectorMode.EXACT:
            raise MutationContractError(
                "mjwarp typed DoF reset only supports exact mutation selectors"
            )
        coordinates: list[int] = []
        for raw_selector in selector.expressions:
            joint_id = self._mujoco.mj_name2id(
                self._cpu_model,
                self._mujoco.mjtObj.mjOBJ_JOINT,
                raw_selector,
            )
            if joint_id < 0 or int(self._cpu_model.jnt_type[joint_id]) != int(
                self._mujoco.mjtJoint.mjJNT_HINGE
            ):
                raise MutationContractError(
                    f"mjwarp typed reset selector {raw_selector!r} must resolve one hinge joint"
                )
            if spec.field_kind is MutationFieldKind.POSITION:
                coordinate = int(self._cpu_model.jnt_qposadr[joint_id]) - self._root_qpos_dim
                count = self._num_dof_pos
            else:
                coordinate = int(self._cpu_model.jnt_dofadr[joint_id]) - self._root_qvel_dim
                count = self._num_dof_vel
            if coordinate < 0 or coordinate >= count:
                raise MutationContractError("mjwarp typed reset coordinate is out of range")
            coordinates.append(coordinate)
        if len(set(coordinates)) != len(coordinates):
            raise MutationContractError(
                "mjwarp typed DoF reset selector resolved duplicate coordinates"
            )
        return tuple(coordinates)

    def bind_mutation_plan(self, specs: tuple[MutationSpec, ...]) -> BoundMutationPlan:
        """Bind the narrow typed reset contract without exposing raw Warp objects."""

        if self._device_batch_plans:
            if self._host_batch_plans:
                raise BackendBatchContractError(
                    "mjwarp cannot bind host and device mutation plans in one backend instance"
                )
            capability_manifest = self.get_mutation_capability_manifest(
                ExecutionProfile.DEVICE_RESIDENT
            )
            device_bound = bind_typed_mutation_plan(
                backend_type=self.backend_type,
                backend_instance_id=self._batch_instance_id,
                num_envs=self._num_envs,
                specs=specs,
                capabilities=capability_manifest.capabilities,
                resolve_selector=self._resolve_mjwarp_typed_mutation_selector,
                capability_manifest=capability_manifest,
            )
            existing_device = self._device_mutation_plans.get(device_bound.fingerprint)
            if existing_device is not None:
                existing_device.public_plan.require_compatible(device_bound)
                return existing_device.public_plan
            has_model_specs = any(
                spec.target.target_kind is MutationTargetKind.MODEL_PARAMETER
                for spec in device_bound.specs
            )
            model_plan = None
            recompute_runtime = None
            if has_model_specs:
                from .armature_recompute import MjwarpArmatureRecomputeWorkspace

                recompute_contract = compile_model_recompute_contract(
                    device_bound,
                    capability_manifest,
                )
                receipt = self._materialize_bound_device_model_fields(recompute_contract)
                model_fields = {
                    field_name: self._get_model_field_bridge(field_name)
                    for field_name in recompute_contract.direct_fields
                }
                default_fields = {
                    field_name: self._get_model_default_bridge(field_name)
                    for field_name in recompute_contract.direct_fields
                }
                model_plan = MjwarpDeviceModelMutationPlan(
                    public_plan=device_bound,
                    placement=self._device_plan_placement(),
                    model_fields=model_fields,
                    default_fields=default_fields,
                    root_qvel_dim=self._root_qvel_dim,
                )
                condition_reduction = None
                graph_condition = None
                warp_graph_condition = None
                recompute_workspace: (
                    MjwarpModelRecomputeWorkspace | MjwarpArmatureRecomputeWorkspace | None
                ) = None
                if (
                    recompute_contract.kind is not MjwarpModelRecomputeKind.NONE
                    and self._warp.is_conditional_graph_supported()
                ):
                    device = self._ensure_device_bridge().qpos.device
                    condition_reduction = torch.empty((), dtype=torch.bool, device=device)
                    graph_condition = torch.zeros((1,), dtype=torch.int32, device=device)
                    warp_graph_condition = self._warp.from_torch(graph_condition)
                    if MjwarpArmatureRecomputeWorkspace.supports(
                        recompute_contract.direct_fields,
                        recompute_contract.kind.value,
                        self._cpu_model,
                        self._mujoco,
                    ):
                        recompute_workspace = MjwarpArmatureRecomputeWorkspace.create(
                            mujoco=self._mujoco,
                            mujoco_warp=self._mujoco_warp,
                            model=self._cpu_model,
                            device_model=self._device_model,
                            device_data=self._device_data,
                            active_mask=model_plan.effective_mask,
                            num_worlds=self._num_envs,
                        )
                    elif recompute_contract.kind in {
                        MjwarpModelRecomputeKind.SET_CONST_0,
                        MjwarpModelRecomputeKind.SET_CONST,
                    }:
                        try:
                            recompute_workspace = MjwarpModelRecomputeWorkspace(
                                self._warp,
                                self._mujoco_warp,
                            )
                        except Exception as exc:
                            raise BackendBatchContractError(
                                "mjwarp failed to capture Model recompute graph "
                                f"{recompute_contract.kind.value!r}"
                            ) from exc
                recompute_runtime = MjwarpModelRecomputeRuntime(
                    public_plan=device_bound,
                    contract=recompute_contract,
                    graph=self._capture_model_recompute_graph(
                        recompute_contract,
                        condition=warp_graph_condition,
                        workspace=recompute_workspace,
                    ),
                    condition_reduction=condition_reduction,
                    graph_condition=graph_condition,
                    warp_graph_condition=warp_graph_condition,
                    workspace=recompute_workspace,
                    storage_generation=self._device_graph_storage_generation,
                    storage_fingerprint=self._device_graph_storage_fingerprint,
                    materialization_receipt=receipt,
                )
            device_runtime_plan = MjwarpDeviceMutationPlan(
                public_plan=device_bound,
                nq=self._nq,
                nv=self._nv,
                root_qpos_dim=self._root_qpos_dim,
                root_qvel_dim=self._root_qvel_dim,
                placement=self._device_plan_placement(),
                model_plan=model_plan,
                recompute_runtime=recompute_runtime,
            )
            for device_batch_plan in self._device_batch_plans.values():
                device_runtime_plan.register_batch_plan(device_batch_plan.public_plan)
            self._device_mutation_plans[device_bound.fingerprint] = device_runtime_plan
            return device_bound

        capability_manifest = self.get_mutation_capability_manifest(ExecutionProfile.HOST_NUMPY)
        bound = bind_typed_mutation_plan(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
            num_envs=self._num_envs,
            specs=specs,
            capabilities=capability_manifest.capabilities,
            resolve_selector=self._resolve_mjwarp_typed_mutation_selector,
            capability_manifest=capability_manifest,
        )
        existing = self._host_mutation_plans.get(bound.fingerprint)
        if existing is not None:
            existing.public_plan.require_compatible(bound)
            return existing.public_plan
        host_runtime_plan = MjwarpHostMutationPlan(
            public_plan=bound,
            nq=self._nq,
            nv=self._nv,
            state_dtype=np.dtype(self._qpos_cache.dtype).name,
            root_qpos_dim=self._root_qpos_dim,
            root_qvel_dim=self._root_qvel_dim,
        )
        for host_batch_plan in self._host_batch_plans.values():
            host_runtime_plan.register_batch_plan(host_batch_plan.public_plan)
        self._host_mutation_plans[bound.fingerprint] = host_runtime_plan
        return bound

    def get_mutation_capability_manifest(
        self,
        execution_profile: ExecutionProfile,
    ) -> MutationCapabilityManifest:
        """Return only effect-tested typed mutations for the requested profile."""

        if not isinstance(execution_profile, ExecutionProfile):
            raise MutationContractError(
                "mjwarp mutation capability profile must be an ExecutionProfile"
            )
        if execution_profile is ExecutionProfile.HOST_NUMPY:
            if self._device_batch_plans or self._device_mutation_plans:
                raise BackendBatchContractError(
                    "mjwarp cannot advertise host_numpy mutation capabilities after a "
                    "device_resident plan was bound"
                )
            capabilities = mjwarp_host_mutation_capabilities(self)
        elif execution_profile is ExecutionProfile.DEVICE_RESIDENT:
            if self._host_batch_plans or self._host_mutation_plans:
                raise BackendBatchContractError(
                    "mjwarp cannot advertise device_resident mutation capabilities after a "
                    "host_numpy plan was bound"
                )
            capabilities = mjwarp_device_mutation_capabilities(self)
        else:  # pragma: no cover - ExecutionProfile is currently exhaustive.
            raise MutationContractError(
                f"mjwarp does not support mutation profile {execution_profile.value!r}"
            )
        return MutationCapabilityManifest(
            backend_type=self.backend_type,
            execution_profile=execution_profile,
            capabilities=capabilities,
        )

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

    def _device_plan_placement(self) -> BufferPlacement:
        """Return the cold-bound CUDA placement without exposing Warp to callers."""

        bridge = self._ensure_device_bridge()
        index = bridge.qpos.device.index
        if index is None:
            raise BackendBatchContractError("mjwarp device bridge has no CUDA device index")
        return BufferPlacement.device("cuda", int(index))

    def _snapshot_device_graph_state(self) -> _MjwarpDeviceGraphState:
        return _MjwarpDeviceGraphState(
            bundles=dict(self._device_graph_bundles),
            storage_generation=self._device_graph_storage_generation,
            storage_buffers=self._device_graph_storage_buffers,
            storage_fingerprint=self._device_graph_storage_fingerprint,
            storage_poisoned=self._device_graph_storage_poisoned,
            captures=self._device_graph_captures,
            launches=self._device_graph_launches,
            recaptures=self._device_graph_recaptures,
            stale_rejections=self._device_graph_stale_rejections,
            eager_fallbacks=self._device_graph_eager_fallbacks,
            storage_verifications=self._device_graph_storage_verifications,
        )

    def _restore_device_graph_state(self, state: _MjwarpDeviceGraphState) -> None:
        self._device_graph_bundles = dict(state.bundles)
        self._device_graph_storage_generation = state.storage_generation
        self._device_graph_storage_buffers = state.storage_buffers
        self._device_graph_storage_fingerprint = state.storage_fingerprint
        self._device_graph_storage_poisoned = state.storage_poisoned
        self._device_graph_captures = state.captures
        self._device_graph_launches = state.launches
        self._device_graph_recaptures = state.recaptures
        self._device_graph_stale_rejections = state.stale_rejections
        self._device_graph_eager_fallbacks = state.eager_fallbacks
        self._device_graph_storage_verifications = state.storage_verifications

    def _snapshot_device_graph_storage(self) -> tuple[DeviceGraphBufferAddress, ...]:
        """Scan graph-reachable Warp storage on a diagnostics/capture cold path."""

        buffers: list[DeviceGraphBufferAddress] = []
        visited: set[int] = set()

        def visit(value: Any, path: str, depth: int) -> None:
            if isinstance(value, self._warp.array):
                dtype = getattr(value.dtype, "__name__", str(value.dtype))
                buffers.append(
                    DeviceGraphBufferAddress(
                        name=path,
                        address=int(value.ptr or 0),
                        shape=tuple(int(dim) for dim in value.shape),
                        dtype=str(dtype),
                        device=str(value.device),
                    )
                )
                return
            if (
                depth >= 4
                or value is None
                or isinstance(value, (str, bytes, bool, int, float, complex, type))
            ):
                return
            identity = id(value)
            if identity in visited:
                return
            visited.add(identity)
            if isinstance(value, dict):
                for key in sorted(value, key=str):
                    visit(value[key], f"{path}[{key!r}]", depth + 1)
                return
            if isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]", depth + 1)
                return
            try:
                attributes = vars(value)
            except TypeError:
                return
            for name in sorted(attributes):
                if not name.startswith("_"):
                    visit(attributes[name], f"{path}.{name}", depth + 1)

        visit(self._device_model, "model", 0)
        visit(self._device_data, "data", 0)
        visit(self._reset_mask_device, "reset_mask", 0)
        for fingerprint, plan in sorted(self._device_batch_plans.items()):
            if plan.controller is not None:
                buffers.extend(
                    plan.controller.storage_buffers(prefix=f"controller[{fingerprint!r}]")
                )
        snapshot = tuple(sorted(buffers, key=lambda item: item.name))
        names = tuple(buffer.name for buffer in snapshot)
        if not snapshot or len(set(names)) != len(names):
            raise BackendBatchContractError(
                "mjwarp graph storage inventory must be non-empty and uniquely named"
            )
        required = {"data.qpos", "data.qvel", "data.ctrl", "data.sensordata", "reset_mask"}
        missing = tuple(sorted(required - set(names)))
        if missing:
            raise BackendBatchContractError(
                f"mjwarp graph storage inventory lacks device ABI buffers: {missing!r}"
            )
        return snapshot

    @staticmethod
    def _graph_storage_fingerprint(
        buffers: tuple[DeviceGraphBufferAddress, ...],
    ) -> str:
        payload = tuple(
            {
                "name": buffer.name,
                "address": buffer.address,
                "shape": buffer.shape,
                "dtype": buffer.dtype,
                "device": buffer.device,
            }
            for buffer in buffers
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _verify_device_graph_storage(self) -> None:
        actual = self._snapshot_device_graph_storage()
        self._device_graph_storage_verifications += 1
        if actual == self._device_graph_storage_buffers:
            return
        self._device_graph_storage_poisoned = True
        self._device_graph_stale_rejections += 1
        raise BackendBatchContractError(
            "mjwarp graph storage addresses changed without an owner recapture barrier"
        )

    def _device_graph_key(self, *, plan: BoundBackendPlan, nsteps: int) -> DeviceGraphCaptureKey:
        if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps <= 0:
            raise BackendBatchContractError("mjwarp device graph requires positive nsteps")
        state_dtypes = tuple(sorted({field.buffer.dtype for field in plan.state.fields}))
        if not state_dtypes:
            raise BackendBatchContractError("mjwarp graph key requires bound state fields")
        return DeviceGraphCaptureKey(
            backend_type=self.backend_type,
            plan_fingerprint=plan.fingerprint,
            num_envs=plan.num_envs,
            state_dtype="+".join(state_dtypes),
            control_dtype=plan.control.buffer.dtype,
            physics_substeps=nsteps,
            storage_generation=self._device_graph_storage_generation,
            storage_fingerprint=self._device_graph_storage_fingerprint,
        )

    def _capture_model_recompute_graph(
        self,
        contract: MjwarpModelRecomputeContract,
        *,
        condition: Any | None,
        workspace: MjwarpModelRecomputeWorkspace | MjwarpArmatureRecomputeWorkspace | None,
    ) -> Any | None:
        """Capture one plan-scoped derived-constant operation without fallback."""

        if contract.kind is MjwarpModelRecomputeKind.NONE:
            return None
        operation = {
            MjwarpModelRecomputeKind.SET_CONST_FIXED: self._mujoco_warp.set_const_fixed,
            MjwarpModelRecomputeKind.SET_CONST_0: self._mujoco_warp.set_const_0,
            MjwarpModelRecomputeKind.SET_CONST: self._mujoco_warp.set_const,
        }[contract.kind]

        def recompute() -> None:
            if workspace is not None:
                with workspace.capture_body(contract.kind) as workspace_recompute:
                    workspace_recompute(self._device_model, self._device_data)
            elif contract.kind is MjwarpModelRecomputeKind.SET_CONST_FIXED:
                operation(self._device_model, self._device_data)
            else:
                operation(self._device_model, self._device_data, restore=False)

        bridge = self._ensure_device_bridge()
        try:
            with (
                _suspend_gc(),
                torch.cuda.stream(bridge.physics_stream),
                self._warp.ScopedStream(bridge.warp_physics_stream),
            ):
                if workspace is not None:
                    workspace.prepare(
                        contract.kind,
                        self._device_model,
                        self._device_data,
                    )
                with self._warp.ScopedCapture() as capture:
                    if condition is None:
                        recompute()
                    else:
                        self._warp.capture_if(condition, on_true=recompute)
        except Exception as exc:
            raise BackendBatchContractError(
                f"mjwarp failed to capture Model recompute graph {contract.kind.value!r}"
            ) from exc
        if capture.graph is None:
            raise BackendBatchContractError(
                f"mjwarp Model recompute capture {contract.kind.value!r} produced no graph"
            )
        return capture.graph

    def _capture_device_graph_bundle(
        self,
        key: DeviceGraphCaptureKey,
        *,
        bound: MjwarpDeviceBatchPlan,
        recapture: bool,
    ) -> _MjwarpDeviceGraphBundle:
        plan = bound.public_plan
        if (
            plan.fingerprint != key.plan_fingerprint
            or plan.control.physics_substeps_per_control != key.physics_substeps
        ):
            raise BackendBatchContractError(
                "mjwarp graph capture runtime does not match its immutable capture key"
            )
        bridge = self._ensure_device_bridge()
        device = self._warp.get_device()
        driver = self._warp.get_cuda_driver_version()
        if (
            not bool(device.is_cuda)
            or driver is None
            or driver < _GRAPH_CAPTURE_MIN_DRIVER
            or not bool(self._warp.is_mempool_enabled(device))
        ):
            raise BackendBatchContractError(
                "mjwarp device_resident execution requires CUDA graph support "
                "(CUDA driver >= 12.4 and Warp mempool enabled); eager physics "
                "performs host roundtrips and is not an allowed fallback"
            )
        try:
            with (
                _suspend_gc(),
                torch.cuda.stream(bridge.physics_stream),
                self._warp.ScopedStream(bridge.warp_physics_stream),
            ):
                with self._warp.ScopedCapture() as reset_capture:
                    self._mujoco_warp.reset_data(
                        self._device_model,
                        self._device_data,
                        reset=self._reset_mask_device,
                    )
                with self._warp.ScopedCapture() as forward_capture:
                    self._mujoco_warp.forward(self._device_model, self._device_data)
                with self._warp.ScopedCapture() as step_capture:
                    for _ in range(key.physics_substeps):
                        if bound.controller is not None:
                            bound.controller.compute()
                        self._mujoco_warp.step(self._device_model, self._device_data)
        except Exception as exc:
            raise BackendBatchContractError(
                "mjwarp failed to capture the required device physics graph bundle"
            ) from exc
        self._device_graph_captures += 1
        if recapture:
            self._device_graph_recaptures += 1
        return _MjwarpDeviceGraphBundle(
            key=key,
            reset_graph=reset_capture.graph,
            forward_graph=forward_capture.graph,
            step_graph=step_capture.graph,
        )

    def _ensure_device_graphs(self, *, bound: MjwarpDeviceBatchPlan) -> None:
        """Capture graph-only device physics under a complete cold-path key."""

        plan = bound.public_plan
        nsteps = plan.control.physics_substeps_per_control
        self._verify_device_graph_storage()
        key = self._device_graph_key(plan=plan, nsteps=nsteps)
        existing = self._device_graph_bundles.get(plan.fingerprint)
        if existing is not None:
            if existing.key != key:
                self._device_graph_stale_rejections += 1
                raise BackendBatchContractError(
                    "mjwarp refused to reuse a device graph under a changed capture key"
                )
            return
        self._device_graph_bundles[plan.fingerprint] = self._capture_device_graph_bundle(
            key,
            bound=bound,
            recapture=False,
        )

    def _require_device_graph_bundle(
        self, *, plan: BoundBackendPlan, nsteps: int
    ) -> _MjwarpDeviceGraphBundle:
        """Perform the constant-time generation/key check used by the hot path."""

        if self._device_graph_storage_poisoned:
            self._device_graph_stale_rejections += 1
            raise BackendBatchContractError(
                "mjwarp device graph storage is stale; rebuild or recapture on the owner path"
            )
        try:
            bundle = self._device_graph_bundles[plan.fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp device graph was not captured for this plan and control cadence"
            ) from exc
        key = bundle.key
        if (
            key.plan_fingerprint != plan.fingerprint
            or key.num_envs != plan.num_envs
            or key.control_dtype != plan.control.buffer.dtype
            or key.physics_substeps != nsteps
            or key.storage_generation != self._device_graph_storage_generation
            or key.storage_fingerprint != self._device_graph_storage_fingerprint
        ):
            self._device_graph_storage_poisoned = True
            self._device_graph_stale_rejections += 1
            raise BackendBatchContractError(
                "mjwarp rejected a stale graph generation or changed capture key"
            )
        return bundle

    def _recapture_device_graphs_after_storage_change(self) -> None:
        """Owner-only recovery after model storage replacement on a cold path.

        Device state/control/reset-mask aliases cannot be repaired in place;
        replacing one of those allocations poisons this backend instance. Model
        field expansion may recapture all existing graph keys atomically.
        """

        before = self._snapshot_device_graph_state()
        actual = self._snapshot_device_graph_storage()
        if actual == self._device_graph_storage_buffers:
            raise BackendBatchContractError(
                "mjwarp graph recapture requires an observable storage replacement"
            )
        previous = {buffer.name: buffer for buffer in self._device_graph_storage_buffers}
        current = {buffer.name: buffer for buffer in actual}
        bridge_names = {
            "data.qpos",
            "data.qvel",
            "data.ctrl",
            "data.sensordata",
            "reset_mask",
        }
        changed_bridge = tuple(
            sorted(name for name in bridge_names if previous.get(name) != current.get(name))
        )
        if changed_bridge:
            self._device_graph_storage_poisoned = True
            self._device_graph_stale_rejections += 1
            raise BackendBatchContractError(
                "mjwarp cannot recapture after device ABI storage replacement: "
                f"changed={changed_bridge!r}"
            )

        next_generation = self._device_graph_storage_generation + 1
        next_fingerprint = self._graph_storage_fingerprint(actual)
        new_bundles: dict[str, _MjwarpDeviceGraphBundle] = {}
        try:
            for old_bundle in before.bundles.values():
                old_key = old_bundle.key
                try:
                    bound = self._device_batch_plans[old_key.plan_fingerprint]
                except KeyError as exc:
                    raise BackendBatchContractError(
                        "mjwarp graph recapture lacks its bound device runtime plan"
                    ) from exc
                new_key = DeviceGraphCaptureKey(
                    backend_type=old_key.backend_type,
                    plan_fingerprint=old_key.plan_fingerprint,
                    num_envs=old_key.num_envs,
                    state_dtype=old_key.state_dtype,
                    control_dtype=old_key.control_dtype,
                    physics_substeps=old_key.physics_substeps,
                    storage_generation=next_generation,
                    storage_fingerprint=next_fingerprint,
                )
                new_bundles[new_key.plan_fingerprint] = self._capture_device_graph_bundle(
                    new_key,
                    bound=bound,
                    recapture=True,
                )
        except Exception:
            self._restore_device_graph_state(before)
            self._device_graph_storage_poisoned = True
            raise
        self._device_graph_bundles = new_bundles
        self._device_graph_storage_generation = next_generation
        self._device_graph_storage_buffers = actual
        self._device_graph_storage_fingerprint = next_fingerprint
        self._device_graph_storage_poisoned = False

    def get_device_graph_diagnostics(
        self, *, verify_storage: bool = False
    ) -> DeviceGraphDiagnostics:
        """Return graph counters and optionally rescan all captured storage."""

        if not isinstance(verify_storage, bool):
            raise BackendBatchContractError("verify_storage must be a bool")
        if verify_storage:
            self._verify_device_graph_storage()
        keys: tuple[DeviceGraphCaptureKey, ...] = ()
        if not self._device_graph_storage_poisoned:
            keys = tuple(
                sorted(
                    (
                        bundle.key
                        for bundle in self._device_graph_bundles.values()
                        if bundle.key.storage_generation == self._device_graph_storage_generation
                        and bundle.key.storage_fingerprint == self._device_graph_storage_fingerprint
                    ),
                    key=lambda key: key.canonical_order,
                )
            )
        return DeviceGraphDiagnostics(
            backend_type=self.backend_type,
            execution_mode=DeviceGraphExecutionMode.CUDA_GRAPH,
            active_keys=keys,
            storage_buffers=self._device_graph_storage_buffers,
            storage_generation=self._device_graph_storage_generation,
            storage_fingerprint=self._device_graph_storage_fingerprint,
            capture_count=self._device_graph_captures,
            launch_count=self._device_graph_launches,
            recapture_count=self._device_graph_recaptures,
            stale_rejection_count=self._device_graph_stale_rejections,
            eager_fallback_count=self._device_graph_eager_fallbacks,
            storage_verification_count=self._device_graph_storage_verifications,
            instrumentation_complete=True,
        )

    def get_model_recompute_diagnostics(
        self,
        plan: BoundMutationPlan,
    ) -> MjwarpModelRecomputeDiagnostics | None:
        """Return low-frequency evidence for one bound mutation plan."""

        if not isinstance(plan, BoundMutationPlan):
            raise BackendBatchContractError(
                "mjwarp Model recompute diagnostics require a bound mutation plan"
            )
        plan.require_owner(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
        )
        try:
            runtime_plan = self._device_mutation_plans[plan.fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp Model recompute diagnostics plan was not published"
            ) from exc
        runtime_plan.public_plan.require_compatible(plan)
        return runtime_plan.recompute_diagnostics

    def get_mutation_performance_diagnostics(
        self,
        plan: BoundMutationPlan,
    ) -> BackendMutationPerformanceDiagnostics:
        """Project backend-owned Model and lifecycle evidence at a cold boundary."""

        if not isinstance(plan, BoundMutationPlan):
            raise BackendBatchContractError(
                "mjwarp mutation performance diagnostics require a bound mutation plan"
            )
        plan.require_owner(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
        )
        try:
            runtime_plan = self._device_mutation_plans[plan.fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp mutation performance diagnostics plan was not published"
            ) from exc
        runtime_plan.public_plan.require_compatible(plan)

        model_targets = tuple(
            sorted(
                {
                    spec.target.target_key
                    for spec in plan.specs
                    if spec.target.target_kind is MutationTargetKind.MODEL_PARAMETER
                }
            )
        )
        recompute = runtime_plan.recompute_diagnostics
        materialization = None
        recompute_kind = "none"
        direct_fields: tuple[str, ...] = ()
        derived_fields: tuple[str, ...] = ()
        recompute_capture_count = 0
        recompute_launch_count = 0
        if model_targets:
            if recompute is None or runtime_plan.recompute_runtime is None:
                raise BackendBatchContractError(
                    "mjwarp Model performance diagnostics lack a recompute owner"
                )
            receipt = runtime_plan.recompute_runtime.materialization_receipt
            if receipt is not self._model_materialization_receipt:
                raise BackendBatchContractError(
                    "mjwarp Model performance diagnostics have a stale materialization receipt"
                )
            self._verify_model_materialization_receipt(receipt)
            if (
                recompute.mutation_plan_fingerprint != plan.fingerprint
                or recompute.materialization_receipt_fingerprint != receipt.fingerprint
                or recompute.storage_generation != receipt.storage_generation_after
                or recompute.storage_fingerprint != receipt.storage_fingerprint_after
            ):
                raise BackendBatchContractError(
                    "mjwarp recompute and materialization diagnostics have different identities"
                )
            materialization = BackendModelMaterializationDiagnostics(
                receipt_fingerprint=receipt.fingerprint,
                num_worlds=receipt.num_worlds,
                fields=tuple(
                    BackendModelFieldDiagnostics(
                        field_name=field.field_name,
                        role=field.role.value,
                        materialized_shape=field.materialized_shape,
                        materialized_address=field.materialized_address,
                        model_bytes=field.model_bytes,
                        replaced=field.replaced,
                        compiled_default_shape=field.compiled_default_shape,
                        per_world_default_shape=field.per_world_default_shape,
                    )
                    for field in receipt.fields
                ),
                expanded_model_bytes=receipt.expanded_model_bytes,
                baseline_bytes=receipt.baseline_bytes,
                storage_generation=receipt.storage_generation_after,
                storage_fingerprint=receipt.storage_fingerprint_after,
            )
            recompute_kind = recompute.kind.value
            direct_fields = recompute.direct_fields
            derived_fields = recompute.derived_fields
            recompute_capture_count = recompute.capture_count
            recompute_launch_count = recompute.launch_count
        elif recompute is not None:
            raise BackendBatchContractError(
                "mjwarp state-only performance diagnostics unexpectedly own Model recompute"
            )

        return BackendMutationPerformanceDiagnostics(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
            mutation_plan_fingerprint=plan.fingerprint,
            model_targets=model_targets,
            recompute_kind=recompute_kind,
            direct_fields=direct_fields,
            derived_fields=derived_fields,
            recompute_capture_count=recompute_capture_count,
            recompute_launch_count=recompute_launch_count,
            materialization=materialization,
            lifecycle=BackendDeviceLifecycleDiagnostics(
                runtime_barriers=self._runtime_barrier_count,
                step_graph_launches=self._device_step_graph_launches,
                reset_graph_launches=self._device_reset_graph_launches,
                forward_graph_launches=self._device_forward_graph_launches,
                state_refreshes=self._device_state_refreshes,
            ),
            graph=self.get_device_graph_diagnostics(verify_storage=True),
        )

    def _require_device_mutation_plan(
        self,
        mutation_batch: DeviceResetMutationBatch,
        plan: BoundBackendPlan,
    ) -> MjwarpDeviceMutationPlan:
        if not isinstance(mutation_batch, DeviceResetMutationBatch):
            raise BackendBatchContractError(
                "mjwarp device reset requires a DeviceResetMutationBatch"
            )
        if mutation_batch.rows != RowSelection.all(self._num_envs):
            raise BackendBatchContractError(
                "mjwarp device reset requires an all-world typed reset envelope"
            )
        mutation_batch.plan.require_owner(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
        )
        try:
            runtime_plan = self._device_mutation_plans[mutation_batch.plan_fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp device reset mutation plan was not bound by this backend instance"
            ) from exc
        runtime_plan.public_plan.require_compatible(mutation_batch.plan)
        runtime_plan.require_registered_batch_plan(plan)
        runtime_plan.require_current_recompute(
            storage_generation=self._device_graph_storage_generation,
            storage_fingerprint=self._device_graph_storage_fingerprint,
            materialization_receipt=self._model_materialization_receipt,
        )
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

    def _require_device_batch_plan(self, plan: BoundBackendPlan) -> MjwarpDeviceBatchPlan:
        if not isinstance(plan, BoundBackendPlan):
            raise BackendBatchContractError("mjwarp batch plan must be a BoundBackendPlan")
        if plan.execution_profile is not ExecutionProfile.DEVICE_RESIDENT:
            raise BackendBatchContractError("mjwarp plan is not a device-resident batch plan")
        plan.require_owner(
            backend_type=self.backend_type,
            backend_instance_id=self._batch_instance_id,
        )
        try:
            bound = self._device_batch_plans[plan.fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp device batch plan was not bound by this backend instance"
            ) from exc
        bound.public_plan.require_compatible(plan)
        return bound

    def _invalidate_host_batch_state(self) -> None:
        for bound in self._host_batch_plans.values():
            bound.lease.invalidate()

    def _invalidate_device_batch_state(self) -> None:
        """Invalidate every borrowed device view before a physics mutation."""

        for bound in self._device_batch_plans.values():
            bound.invalidate()

    def _enqueue_device_refresh(
        self,
        bound: MjwarpDeviceBatchPlan,
        *,
        event: torch.cuda.Event,
    ) -> None:
        """Pack bound state fields on the dedicated physics stream only."""

        bridge = self._ensure_device_bridge()
        with torch.cuda.stream(bridge.physics_stream):
            with self._warp.ScopedStream(bridge.warp_physics_stream):
                bound.refresh()
                self._device_state_refreshes += 1
            event.record(bridge.physics_stream)

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
        if plan.execution_profile is ExecutionProfile.HOST_NUMPY:
            return self._require_host_batch_plan(plan).materialize(rows, phase)
        if plan.execution_profile is ExecutionProfile.DEVICE_RESIDENT:
            bound = self._require_device_batch_plan(plan)
            bridge = self._ensure_device_bridge()
            self._enqueue_device_refresh(bound, event=bridge.read_event)
            return bound.materialize(
                rows=rows,
                phase=phase,
                completion_event=bridge.read_event,
            )
        raise BackendBatchContractError(
            f"mjwarp cannot read unsupported execution profile {plan.execution_profile.value!r}"
        )

    def _execute_host_step(
        self,
        ctrl: np.ndarray,
        nsteps: int,
    ) -> tuple[dict[str, float], BackendTransferCounters]:
        """Execute the single owner-layer host-cache barrier used by both APIs."""
        self._runtime_barrier_count += 1
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

        self._runtime_barrier_count += 1
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

    def _execute_device_step(
        self,
        bound: MjwarpDeviceBatchPlan,
        control: DeviceTensorView,
        nsteps: int,
    ) -> BackendReadResult:
        """Queue action handoff, physics and state packing without host sync.

        The runner must record the action producer event.  The dedicated
        mjwarp physics stream waits for that event, executes Warp physics on
        the same native stream, packs the declared state fields, then records
        one backend completion event for the device consumer.
        """

        bridge = self._ensure_device_bridge()
        producer = control.require_completion()
        if producer.placement.device_index != bridge.ctrl.device.index:
            raise BackendBatchContractError(
                "mjwarp device control completion event belongs to another CUDA device"
            )
        action = control.torch()
        start = time.perf_counter()
        graph_bundle = self._require_device_graph_bundle(
            plan=bound.public_plan,
            nsteps=nsteps,
        )
        self._runtime_barrier_count += 1
        self._invalidate_device_batch_state()
        with torch.cuda.stream(bridge.physics_stream):
            producer.wait(bridge.physics_stream)
            command = bridge.ctrl if bound.controller is None else bound.controller.command
            command.copy_(action, non_blocking=True)
            with self._warp.ScopedStream(bridge.warp_physics_stream):
                self._warp.capture_launch(graph_bundle.step_graph)
                self._device_graph_launches += 1
                self._device_step_graph_launches += 1
                bound.refresh()
                self._device_state_refreshes += 1
            bridge.step_event.record(bridge.physics_stream)
        result = bound.materialize(
            rows=RowSelection.all(self._num_envs),
            phase=StateBatchPhase.TERMINAL,
            completion_event=bridge.step_event,
        )
        return BackendReadResult(
            state=result.state,
            diagnostics=BackendBatchDiagnostics(
                counters=BackendBatchCounters(
                    state_materializations=1,
                    instrumentation_complete=True,
                ),
                timings=(
                    BackendTiming("device_enqueue", (time.perf_counter() - start) * 1000.0),
                    *result.diagnostics.timings,
                ),
                completion_event=result.diagnostics.completion_event,
            ),
        )

    def _execute_device_reset(
        self,
        bound: MjwarpDeviceBatchPlan,
        mutation_plan: MjwarpDeviceMutationPlan,
        mutation_batch: DeviceResetMutationBatch,
        phase_timing_session: DeviceResetPhaseTimingSession | None,
        phase_timing_token: DeviceResetPhaseTimingSampleToken | None,
    ) -> BackendReadResult:
        """Queue one masked CUDA reset/forward/state-pack lifecycle barrier."""

        timing: tuple[DeviceResetPhaseTimingSession, DeviceResetPhaseTimingSampleToken] | None
        if phase_timing_session is None:
            if phase_timing_token is not None:
                raise BackendBatchContractError(
                    "mjwarp reset phase timing token requires its owning session"
                )
            timing = None
        else:
            if phase_timing_token is None:
                raise BackendBatchContractError(
                    "mjwarp reset phase timing session requires an active token"
                )
            timing = (phase_timing_session, phase_timing_token)

        bridge = self._ensure_device_bridge()
        completion = mutation_batch.completion
        if completion.placement.device_index != bridge.qpos.device.index:
            raise BackendBatchContractError(
                "mjwarp device reset completion event belongs to another CUDA device"
            )
        start = time.perf_counter()
        graph_bundle = self._require_device_graph_bundle(
            plan=bound.public_plan,
            nsteps=bound.public_plan.control.physics_substeps_per_control,
        )
        self._runtime_barrier_count += 1
        self._invalidate_device_batch_state()
        with torch.cuda.stream(bridge.physics_stream):
            completion.wait(bridge.physics_stream)
            active_mask = mutation_plan.active_mask(mutation_batch)
            effective_mask = mutation_plan.effective_active_mask(
                mutation_batch,
                active_mask=active_mask,
            )
            bridge.reset_mask.copy_(effective_mask, non_blocking=True)
            with self._warp.ScopedStream(bridge.warp_physics_stream):
                self._warp.capture_launch(graph_bundle.reset_graph)
                self._device_graph_launches += 1
                self._device_reset_graph_launches += 1
                if timing is not None:
                    timing[0].begin_backend_phase(
                        timing[1],
                        "mutation_commit",
                        bridge.physics_stream,
                    )
                mutation_plan.commit_model(
                    mutation_batch,
                    active_mask=effective_mask,
                )
                if timing is not None:
                    timing[0].end_backend_phase(
                        timing[1],
                        "mutation_commit",
                        bridge.physics_stream,
                    )
                    timing[0].begin_backend_phase(
                        timing[1],
                        "recompute_constants",
                        bridge.physics_stream,
                    )
                mutation_plan.launch_model_recompute(
                    self._warp,
                    active_mask=effective_mask,
                )
                if timing is not None:
                    timing[0].end_backend_phase(
                        timing[1],
                        "recompute_constants",
                        bridge.physics_stream,
                    )
                mutation_plan.stage_reset_state(
                    mutation_batch,
                    qpos=bridge.qpos,
                    qvel=bridge.qvel,
                    active_mask=effective_mask,
                )
                if timing is not None:
                    timing[0].begin_backend_phase(
                        timing[1],
                        "reset_forward",
                        bridge.physics_stream,
                    )
                self._warp.capture_launch(graph_bundle.forward_graph)
                self._device_graph_launches += 1
                self._device_forward_graph_launches += 1
                if timing is not None:
                    timing[0].end_backend_phase(
                        timing[1],
                        "reset_forward",
                        bridge.physics_stream,
                    )
                bound.refresh_masked(bridge.reset_mask)
                self._device_state_refreshes += 1
                if timing is not None:
                    timing[0].end_sample(
                        timing[1],
                        bridge.physics_stream,
                    )
            bridge.reset_event.record(bridge.physics_stream)
        result = bound.materialize(
            rows=RowSelection.all(self._num_envs),
            phase=StateBatchPhase.RESET,
            completion_event=bridge.reset_event,
        )
        return BackendReadResult(
            state=result.state,
            diagnostics=BackendBatchDiagnostics(
                counters=BackendBatchCounters(
                    state_materializations=1,
                    instrumentation_complete=True,
                ),
                timings=(
                    BackendTiming("device_reset_enqueue", (time.perf_counter() - start) * 1000.0),
                    *result.diagnostics.timings,
                ),
                completion_event=result.diagnostics.completion_event,
            ),
        )

    def _require_unconsumed_device_control(self, control: DeviceTensorView) -> None:
        """Reject reuse of a runner control before its lease advances.

        A device action is not merely an address: its producer event denotes
        one concrete write.  Reusing that event/lease epoch after a physics
        barrier violates ``UNTIL_STEP_COMPLETE`` even if the tensor pointer is
        stable.  The runner creates the next view only after advancing its own
        lease epoch.
        """

        previous_epoch = self._device_control_epochs.get(control.lease)
        if previous_epoch is not None and control.epoch <= previous_epoch:
            raise DeviceBufferContractError(
                "mjwarp device control lease epoch was already consumed; "
                "record a new producer event after advancing the runner lease"
            )

    def _mark_device_control_consumed(self, control: DeviceTensorView) -> None:
        self._device_control_epochs[control.lease] = control.epoch

    def _require_unconsumed_device_reset(self, batch: DeviceResetMutationBatch) -> None:
        mask = batch.active_mask.handle
        if not isinstance(mask, DeviceTensorView):  # pragma: no cover - envelope invariant.
            raise DeviceBufferContractError("mjwarp device reset mask is not a DeviceTensorView")
        mask.assert_valid()
        previous_epoch = self._device_reset_epochs.get(mask.lease)
        if previous_epoch is not None and mask.epoch <= previous_epoch:
            raise DeviceBufferContractError(
                "mjwarp device reset lease epoch was already committed; "
                "advance the manager lease and publish a new event"
            )

    def _mark_device_reset_consumed(self, batch: DeviceResetMutationBatch) -> None:
        mask = batch.active_mask.handle
        assert isinstance(mask, DeviceTensorView)  # validated by the envelope.
        self._device_reset_epochs[mask.lease] = mask.epoch

    def step_batch(
        self,
        plan: BoundBackendPlan,
        control_batch: ControlBatch,
        *,
        mutation_batch: BackendMutationBatch | None = None,
        nsteps: int = 1,
    ) -> BackendStepResult:
        if plan.execution_profile is ExecutionProfile.DEVICE_RESIDENT:
            return self._step_device_batch(
                plan,
                control_batch,
                mutation_batch=mutation_batch,
                nsteps=nsteps,
            )
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

    def _step_device_batch(
        self,
        plan: BoundBackendPlan,
        control_batch: ControlBatch,
        *,
        mutation_batch: BackendMutationBatch | None,
        nsteps: int,
    ) -> BackendStepResult:
        """Advance a device-resident plan with explicit event ownership."""

        bound = self._require_device_batch_plan(plan)
        if self._pre_step_control_fn is not None:
            raise BackendBatchContractError("mjwarp device batches reject host pre-step callbacks")
        if not isinstance(control_batch, ControlBatch):
            raise BackendBatchContractError("mjwarp control must be a ControlBatch")
        plan.require_compatible(control_batch.plan)
        if not control_batch.rows.is_all:
            raise BackendBatchContractError(
                "mjwarp device physics steps require controls for all rows"
            )
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
                "mjwarp device physics steps do not support mutation batches; "
                "simulation-state mutation is only accepted by reset_batch"
            )
        control = require_device_tensor_view(
            control_batch.buffer.handle,
            contract=plan.control.buffer,
            require_completion=True,
        )
        expected = (self._num_envs, *plan.control.buffer.row_shape)
        if control.shape != expected:
            raise BackendBatchContractError(
                f"mjwarp device control shape must be {expected}, got {control.shape}"
            )
        self._require_unconsumed_device_control(control)
        result = self._execute_device_step(bound, control, nsteps)
        self._mark_device_control_consumed(control)
        return BackendStepResult(terminal_state=result.state, diagnostics=result.diagnostics)

    def reset_batch(
        self,
        plan: BoundBackendPlan,
        rows: RowSelection,
        *,
        mutation_batch: BackendMutationBatch | None = None,
        phase_timing: DeviceResetPhaseTimingSampleToken | None = None,
    ) -> BackendResetResult:
        if plan.execution_profile is ExecutionProfile.DEVICE_RESIDENT:
            return self._reset_device_batch(
                plan,
                rows,
                mutation_batch=mutation_batch,
                phase_timing=phase_timing,
            )
        if phase_timing is not None:
            raise BackendBatchContractError(
                "mjwarp host_numpy reset does not support CUDA phase timing"
            )
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

    def _reset_device_batch(
        self,
        plan: BoundBackendPlan,
        rows: RowSelection,
        *,
        mutation_batch: BackendMutationBatch | None,
        phase_timing: DeviceResetPhaseTimingSampleToken | None,
    ) -> BackendResetResult:
        """Commit an all-world device reset whose active rows stay on CUDA."""

        bound = self._require_device_batch_plan(plan)
        if not isinstance(rows, RowSelection) or rows != RowSelection.all(self._num_envs):
            raise BackendBatchContractError(
                "mjwarp device reset requires RowSelection.all; selected rows live in the CUDA mask"
            )
        if not isinstance(mutation_batch, DeviceResetMutationBatch):
            raise BackendBatchContractError(
                "mjwarp device reset requires a DeviceResetMutationBatch"
            )
        mutation_plan = self._require_device_mutation_plan(mutation_batch, plan)
        self._require_unconsumed_device_reset(mutation_batch)
        mutation_plan.validate_batch(mutation_batch)
        phase_timing_session: DeviceResetPhaseTimingSession | None = None
        if phase_timing is not None:
            phase_timing_session = self._reset_phase_timing_session
            if phase_timing_session is None:
                raise BackendBatchContractError(
                    "mjwarp reset phase timing token was not created by this backend"
                )
            try:
                phase_timing_session.require_backend_sample(
                    phase_timing,
                    backend_type=self.backend_type,
                    backend_instance_id=self._batch_instance_id,
                    placement=self._device_plan_placement(),
                )
            except DevicePhaseTimingError as exc:
                raise BackendBatchContractError(
                    f"mjwarp reset phase timing token is invalid: {exc}"
                ) from exc
        result = self._execute_device_reset(
            bound,
            mutation_plan,
            mutation_batch,
            phase_timing_session,
            phase_timing,
        )
        self._mark_device_reset_consumed(mutation_batch)
        return BackendResetResult(reset_state=result.state, diagnostics=result.diagnostics)

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
