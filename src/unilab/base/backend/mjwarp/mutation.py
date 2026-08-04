"""Adapter-owned typed simulation-state reset support for ``mjwarp``."""

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
    TypedBackendMutationBatch,
)
from .capability import mjwarp_host_reset_descriptor

if TYPE_CHECKING:
    from .backend import MjwarpBackend


_ROOT_TARGETS = {
    "state.root.position": ("qpos", slice(0, 3)),
    "state.root.orientation": ("qpos", slice(3, 7)),
    "state.root.linear_velocity": ("qvel", slice(0, 3)),
    "state.root.angular_velocity": ("qvel", slice(3, 6)),
}
_DOF_TARGETS = frozenset({"state.dof.position", "state.dof.angular_velocity"})


def _value_template(*, row_shape: tuple[int, ...], dtype: str) -> BufferContract:
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
    return MutationCapability(
        target_key=target_key,
        target_kind=MutationTargetKind.SIMULATION_STATE,
        entity_kind=entity_kind,
        field_kind=field_kind,
        entity_count=entity_count,
        value_template=_value_template(row_shape=row_shape, dtype=dtype),
        triggers=frozenset({MutationTrigger.RESET}),
        commit_phases=frozenset({MutationCommitPhase.RESET}),
        operations=frozenset({MutationOperation.SET}),
        baselines=frozenset({MutationBaseline.DEFAULT}),
        persistences=frozenset({MutationPersistence.EPISODE}),
        recompute_levels=frozenset({MutationRecomputeLevel.KINEMATICS}),
        descriptor=mjwarp_host_reset_descriptor(target_key=target_key),
    )


def mjwarp_host_mutation_capabilities(backend: MjwarpBackend) -> tuple[MutationCapability, ...]:
    """Return only the selected-row state reset surface verified by the pilot."""

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
            ("state.root.angular_velocity", MutationFieldKind.ANGULAR_VELOCITY, (3,)),
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
    for target_key, field_kind, count in (
        ("state.dof.position", MutationFieldKind.POSITION, backend._num_dof_pos),
        (
            "state.dof.angular_velocity",
            MutationFieldKind.ANGULAR_VELOCITY,
            backend._num_dof_vel,
        ),
    ):
        if count > 0:
            capabilities.append(
                _reset_capability(
                    target_key=target_key,
                    entity_kind=MutationEntityKind.DOF,
                    field_kind=field_kind,
                    entity_count=count,
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
    _row_ids: np.ndarray = field(init=False, repr=False)
    _registered_batch_plans: dict[str, BoundBackendPlan] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.nq <= 0 or self.nv <= 0:
            raise BackendBatchContractError("mjwarp mutation binding requires positive nq and nv")
        dtype = np.dtype(self.state_dtype)
        self._reset_qpos = np.empty((self.public_plan.num_envs, self.nq), dtype=dtype)
        self._reset_qvel = np.empty((self.public_plan.num_envs, self.nv), dtype=dtype)
        self._row_ids = np.arange(self.public_plan.num_envs, dtype=np.intp)

    def register_batch_plan(self, plan: BoundBackendPlan) -> None:
        if plan.num_envs != self.public_plan.num_envs:
            raise BackendBatchContractError(
                "mjwarp mutation and state plans must use the same row universe"
            )
        if plan.execution_profile is not ExecutionProfile.HOST_NUMPY:
            raise BackendBatchContractError("mjwarp typed mutations only support host_numpy plans")
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
    def _require_value_handle(value: object, rows: RowSelection) -> np.ndarray:
        buffer = getattr(value, "buffer", None)
        spec = getattr(value, "spec", None)
        handle = getattr(buffer, "handle", None)
        if not isinstance(handle, np.ndarray) or not isinstance(spec, BoundMutationSpec):
            raise BackendBatchContractError("mjwarp typed mutation value is malformed")
        expected_shape = (rows.count, *spec.value_buffer.row_shape)
        if handle.shape != expected_shape or handle.dtype.name != spec.value_buffer.dtype:
            raise BackendBatchContractError(
                "mjwarp typed mutation value handle does not match its bound contract"
            )
        if not handle.flags.c_contiguous:
            raise BackendBatchContractError("mjwarp typed mutation values must be C-contiguous")
        return handle

    def _state_values(
        self, batch: TypedBackendMutationBatch
    ) -> tuple[tuple[int, BoundMutationSpec, np.ndarray], ...]:
        if batch.model.values or batch.wrench.values or batch.task_state.values:
            raise BackendBatchContractError(
                "mjwarp typed reset only supports simulation-state values"
            )
        window = batch.state.bound_buffer_window
        if window is not None:
            values = tuple(
                (index, self.public_plan.specs[index], window.buffer_at(index)[: batch.rows.count])
                for index in window.field_indices
            )
        else:
            values = tuple(
                (value.field_index, value.spec, self._require_value_handle(value, batch.rows))
                for value in batch.state.values
            )
        if tuple(sorted(index for index, _, _ in values)) != tuple(
            range(len(self.public_plan.specs))
        ):
            raise BackendBatchContractError(
                "mjwarp typed reset must supply every bound simulation-state field once"
            )
        return values

    def stage_reset_state(
        self,
        batch: TypedBackendMutationBatch,
        qpos_cache: np.ndarray,
        qvel_cache: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Patch selected semantic values into stable full-world host staging."""

        self.public_plan.require_compatible(batch.plan)
        if qpos_cache.shape != self._reset_qpos.shape or qvel_cache.shape != self._reset_qvel.shape:
            raise BackendBatchContractError(
                "mjwarp typed reset source cache does not match the bound mutation plan"
            )
        if qpos_cache.dtype.name != self.state_dtype or qvel_cache.dtype.name != self.state_dtype:
            raise BackendBatchContractError(
                "mjwarp typed reset source cache dtype does not match the bound mutation plan"
            )
        rows = batch.rows
        if not rows.is_all:
            assert rows.indices is not None
            self._row_ids[: rows.count] = rows.indices
        row_ids = self._row_ids[: rows.count]
        np.copyto(self._reset_qpos, qpos_cache)
        np.copyto(self._reset_qvel, qvel_cache)

        for _, spec, value in self._state_values(batch):
            target_key = spec.target.target_key
            root_target = _ROOT_TARGETS.get(target_key)
            if root_target is not None:
                storage_name, columns = root_target
                storage = self._reset_qpos if storage_name == "qpos" else self._reset_qvel
                storage[row_ids, columns] = value[:, 0, :]
                continue
            if (
                target_key not in _DOF_TARGETS
                or spec.target.entity_kind is not MutationEntityKind.DOF
            ):
                raise MutationContractError(
                    "mjwarp typed reset plan contains an unsupported simulation-state target"
                )
            offset = (
                self.root_qpos_dim if target_key == "state.dof.position" else self.root_qvel_dim
            )
            storage = self._reset_qpos if target_key == "state.dof.position" else self._reset_qvel
            columns = np.asarray(spec.target.entity_ids, dtype=np.intp) + offset
            storage[np.ix_(row_ids, columns)] = value[:, :, 0]
        return self._reset_qpos, self._reset_qvel, row_ids


__all__ = ["MjwarpHostMutationPlan", "mjwarp_host_mutation_capabilities"]
