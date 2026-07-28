"""Backend-owned typed simulation-state reset support for ``mjwarp``.

The public mutation contract intentionally exposes only semantic targets and
manager-owned value buffers.  This module is the sole owner of the mapping to
the independent ``mujoco_warp`` qpos/qvel layout.  It is cold-bound once and
keeps all reset staging buffers stable for subsequent managed reset barriers.

Only the small Phase 3C simulation-state slice is implemented here: a
floating root and single-DoF hinge coordinates.  Model parameter mutations,
external wrenches, task state, device graphs, and controllers remain outside
this adapter and are rejected before physics is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..batch import (
    BackendBatchContractError,
    BoundBackendPlan,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    ExecutionProfile,
    RowSelection,
)
from ..mutation import (
    BoundMutationPlan,
    BoundMutationSpec,
    MutationBaseline,
    MutationCapability,
    MutationCommitPhase,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationTargetKind,
    MutationTrigger,
)
from ..mutation_batch import MutationValueBatch, TypedBackendMutationBatch

if TYPE_CHECKING:
    from .backend import MjwarpBackend


_ROOT_TARGETS = frozenset(
    {
        "state.root.position",
        "state.root.orientation",
        "state.root.linear_velocity",
        "state.root.angular_velocity",
    }
)
_DOF_TARGETS = frozenset({"state.dof.position", "state.dof.angular_velocity"})


def _manager_value_template(*, row_shape: tuple[int, ...], dtype: str) -> BufferContract:
    """Return the one permitted host value contract for this reference adapter."""

    return BufferContract(
        row_shape=row_shape,
        dtype=dtype,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=BufferPlacement.host(),
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=False,
    )


def _reset_capability(
    *,
    target_key: str,
    entity_kind: MutationEntityKind,
    field_kind: MutationFieldKind,
    entity_count: int,
    row_shape: tuple[int, ...],
    dtype: str,
) -> MutationCapability:
    """Construct one fully typed capability without runtime option inference."""

    return MutationCapability(
        target_key=target_key,
        target_kind=MutationTargetKind.SIMULATION_STATE,
        entity_kind=entity_kind,
        field_kind=field_kind,
        entity_count=entity_count,
        value_template=_manager_value_template(row_shape=row_shape, dtype=dtype),
        triggers=frozenset({MutationTrigger.RESET}),
        commit_phases=frozenset({MutationCommitPhase.RESET}),
        operations=frozenset({MutationOperation.SET}),
        baselines=frozenset({MutationBaseline.DEFAULT}),
        persistences=frozenset({MutationPersistence.EPISODE}),
        recompute_levels=frozenset({MutationRecomputeLevel.KINEMATICS}),
    )


def mjwarp_host_mutation_capabilities(backend: MjwarpBackend) -> tuple[MutationCapability, ...]:
    """Declare the cold-bound, effect-tested state reset surface.

    A body selector represents the first free root only.  A DOF selector is
    resolved target-specifically to a hinge qpos or qvel coordinate below.
    This keeps qpos/qvel domains separate even for future models where their
    numeric coordinates differ.
    """

    dtype = np.dtype(backend._qpos_cache.dtype).name
    capabilities: list[MutationCapability] = []
    if backend._base_body_id is not None and (backend._root_qpos_dim, backend._root_qvel_dim) == (
        7,
        6,
    ):
        for target_key, field_kind, row_shape in (
            ("state.root.position", MutationFieldKind.POSITION, (3,)),
            ("state.root.orientation", MutationFieldKind.ORIENTATION, (4,)),
            ("state.root.linear_velocity", MutationFieldKind.LINEAR_VELOCITY, (3,)),
            (
                "state.root.angular_velocity",
                MutationFieldKind.ANGULAR_VELOCITY,
                (3,),
            ),
        ):
            capabilities.append(
                _reset_capability(
                    target_key=target_key,
                    entity_kind=MutationEntityKind.BODY,
                    field_kind=field_kind,
                    entity_count=backend._nbody,
                    row_shape=row_shape,
                    dtype=dtype,
                )
            )
    if backend._num_dof_pos > 0:
        capabilities.append(
            _reset_capability(
                target_key="state.dof.position",
                entity_kind=MutationEntityKind.DOF,
                field_kind=MutationFieldKind.POSITION,
                entity_count=backend._num_dof_pos,
                row_shape=(1,),
                dtype=dtype,
            )
        )
    if backend._num_dof_vel > 0:
        capabilities.append(
            _reset_capability(
                target_key="state.dof.angular_velocity",
                entity_kind=MutationEntityKind.DOF,
                field_kind=MutationFieldKind.ANGULAR_VELOCITY,
                entity_count=backend._num_dof_vel,
                row_shape=(1,),
                dtype=dtype,
            )
        )
    if not capabilities:
        raise BackendBatchContractError(
            "mjwarp typed state mutation requires a floating root or at least one DoF"
        )
    return tuple(capabilities)


@dataclass
class MjwarpHostMutationPlan:
    """Cold-allocated qpos/qvel staging for one public mutation plan."""

    public_plan: BoundMutationPlan
    nq: int
    nv: int
    state_dtype: str
    root_qpos_dim: int
    root_qvel_dim: int
    _reset_qpos: np.ndarray = field(init=False, repr=False)
    _reset_qvel: np.ndarray = field(init=False, repr=False)
    _reset_row_ids: np.ndarray = field(init=False, repr=False)
    _registered_batch_plans: dict[str, BoundBackendPlan] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.nq <= 0 or self.nv <= 0:
            raise BackendBatchContractError("mjwarp mutation binding requires positive nq and nv")
        dtype = np.dtype(self.state_dtype)
        self._reset_qpos = np.empty((self.public_plan.num_envs, self.nq), dtype=dtype)
        self._reset_qvel = np.empty((self.public_plan.num_envs, self.nv), dtype=dtype)
        self._reset_row_ids = np.arange(self.public_plan.num_envs, dtype=np.intp)

    def register_batch_plan(self, plan: BoundBackendPlan) -> None:
        """Pair a state/control plan on the cold path before any reset commit."""

        if not isinstance(plan, BoundBackendPlan):
            raise BackendBatchContractError("mjwarp state plan must be a BoundBackendPlan")
        if plan.num_envs != self.public_plan.num_envs:
            raise BackendBatchContractError(
                "mjwarp mutation and state plans must use the same row universe"
            )
        if plan.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise BackendBatchContractError(
                "mjwarp typed state mutations only support host_numpy plans"
            )
        previous = self._registered_batch_plans.get(plan.fingerprint)
        if previous is not None:
            previous.require_compatible(plan)
            return
        self._registered_batch_plans[plan.fingerprint] = plan

    def require_registered_batch_plan(self, plan: BoundBackendPlan) -> None:
        try:
            registered = self._registered_batch_plans[plan.fingerprint]
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp typed mutation plan was not cold-path paired with this state/control plan"
            ) from exc
        registered.require_compatible(plan)

    @staticmethod
    def _require_value_handle(value: MutationValueBatch, rows: RowSelection) -> np.ndarray:
        """Validate the opaque manager buffer without coercion or allocation."""

        # ``TypedBackendMutationBatch`` has already checked the envelope and
        # metadata.  Keep the only raw-handle check at the backend owner layer.
        if not isinstance(value, MutationValueBatch):
            raise BackendBatchContractError("mjwarp typed mutation value is malformed")
        handle = value.buffer.handle
        if not isinstance(handle, np.ndarray):
            raise BackendBatchContractError(
                "mjwarp typed mutation values require numpy host buffer handles"
            )
        expected_shape = (rows.count, *value.spec.value_buffer.row_shape)
        if handle.shape != expected_shape:
            raise BackendBatchContractError(
                "mjwarp typed mutation value handle shape does not match its bound contract"
            )
        if handle.dtype.name != value.spec.value_buffer.dtype:
            raise BackendBatchContractError(
                "mjwarp typed mutation value handle dtype does not match its bound contract"
            )
        if not handle.flags.c_contiguous:
            raise BackendBatchContractError("mjwarp typed mutation values must be C-contiguous")
        return handle

    def _validate_state_values(
        self,
        batch: TypedBackendMutationBatch,
    ) -> tuple[tuple[BoundMutationSpec, np.ndarray], ...]:
        if batch.model.values or batch.wrench.values or batch.task_state.values:
            raise BackendBatchContractError(
                "mjwarp typed reset only supports simulation-state values"
            )
        if not batch.state.values:
            raise BackendBatchContractError("mjwarp typed reset requires at least one state value")

        values: list[tuple[BoundMutationSpec, np.ndarray]] = []
        for value in batch.state.values:
            spec = value.spec
            if spec.target.target_kind is not MutationTargetKind.SIMULATION_STATE:
                raise MutationContractError(
                    "mjwarp typed reset plan contains a non-state mutation target"
                )
            target_key = spec.target.target_key
            if target_key in _ROOT_TARGETS:
                if (
                    spec.target.entity_kind is not MutationEntityKind.BODY
                    or len(spec.target.entity_ids) != 1
                ):
                    raise MutationContractError(
                        "mjwarp root reset target must bind exactly one base body"
                    )
            elif target_key in _DOF_TARGETS:
                if (
                    spec.target.entity_kind is not MutationEntityKind.DOF
                    or not spec.target.entity_ids
                    or spec.value_buffer.row_shape[-1] != 1
                ):
                    raise MutationContractError(
                        "mjwarp DoF reset target must bind one-value hinge coordinates"
                    )
            else:
                raise MutationContractError(
                    "mjwarp typed reset plan contains an unsupported simulation-state target"
                )
            values.append((spec, self._require_value_handle(value, batch.rows)))
        return tuple(values)

    def _row_ids(self, rows: RowSelection) -> np.ndarray:
        if rows.universe_size != self.public_plan.num_envs:
            raise BackendBatchContractError(
                "mjwarp typed mutation rows do not match the backend row universe"
            )
        if not rows.is_all:
            assert rows.indices is not None
            self._reset_row_ids[: rows.count] = rows.indices
        return self._reset_row_ids[: rows.count]

    def stage_reset_state(
        self,
        batch: TypedBackendMutationBatch,
        qpos_cache: np.ndarray,
        qvel_cache: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Patch typed values into full-world host staging without mutating live cache."""

        self.public_plan.require_compatible(batch.plan)
        if qpos_cache.shape != self._reset_qpos.shape or qvel_cache.shape != self._reset_qvel.shape:
            raise BackendBatchContractError(
                "mjwarp typed reset source cache does not match the bound mutation plan"
            )
        if qpos_cache.dtype.name != self.state_dtype or qvel_cache.dtype.name != self.state_dtype:
            raise BackendBatchContractError(
                "mjwarp typed reset source cache dtype does not match the bound mutation plan"
            )

        values = self._validate_state_values(batch)
        row_ids = self._row_ids(batch.rows)
        np.copyto(self._reset_qpos, qpos_cache)
        np.copyto(self._reset_qvel, qvel_cache)

        for spec, value in values:
            target_key = spec.target.target_key
            if target_key == "state.root.position":
                for row_offset, row_id in enumerate(row_ids):
                    self._reset_qpos[row_id, 0:3] = value[row_offset, 0, :]
            elif target_key == "state.root.orientation":
                for row_offset, row_id in enumerate(row_ids):
                    self._reset_qpos[row_id, 3:7] = value[row_offset, 0, :]
            elif target_key == "state.root.linear_velocity":
                for row_offset, row_id in enumerate(row_ids):
                    self._reset_qvel[row_id, 0:3] = value[row_offset, 0, :]
            elif target_key == "state.root.angular_velocity":
                for row_offset, row_id in enumerate(row_ids):
                    self._reset_qvel[row_id, 3:6] = value[row_offset, 0, :]
            elif target_key == "state.dof.position":
                for dof_offset, dof_id in enumerate(spec.target.entity_ids):
                    coordinate = self.root_qpos_dim + dof_id
                    for row_offset, row_id in enumerate(row_ids):
                        self._reset_qpos[row_id, coordinate] = value[row_offset, dof_offset, 0]
            elif target_key == "state.dof.angular_velocity":
                for dof_offset, dof_id in enumerate(spec.target.entity_ids):
                    coordinate = self.root_qvel_dim + dof_id
                    for row_offset, row_id in enumerate(row_ids):
                        self._reset_qvel[row_id, coordinate] = value[row_offset, dof_offset, 0]
            else:  # Guard future target additions from silently entering this path.
                raise MutationContractError(
                    "mjwarp typed reset plan contains an unsupported simulation-state target"
                )
        return self._reset_qpos, self._reset_qvel, row_ids


__all__ = ["MjwarpHostMutationPlan", "mjwarp_host_mutation_capabilities"]
