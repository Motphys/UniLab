"""CUDA reset-time mutation support for the ``mjwarp`` device profile.

The device lifecycle deliberately keeps dynamic done/reset membership on CUDA.
``RowSelection.all`` describes the stable all-world value layout while the
typed :class:`~unilab.base.backend.mutation_batch.DeviceResetMutationBatch`
carries the selected rows as one CUDA bool mask.  This module coordinates the
backend-owned state and Model plans; manager and runner code never receive raw
physics tensors or MuJoCo parameter slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

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
)
from ..device import DeviceTensorView, require_device_tensor_view
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
from ..mutation_batch import DeviceResetMutationBatch, MutationValueBatch
from .capability import mjwarp_actuator_pd_descriptor, mjwarp_state_reset_descriptor
from .model_mutation import MjwarpDeviceModelMutationPlan

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


def _device_manager_value_template(
    *, placement: BufferPlacement, row_shape: tuple[int, ...]
) -> BufferContract:
    """Return the sole device mutation value ABI supported by this slice."""

    return BufferContract(
        row_shape=row_shape,
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=True,
    )


def _reset_capability(
    *,
    target_key: str,
    entity_kind: MutationEntityKind,
    field_kind: MutationFieldKind,
    entity_count: int,
    placement: BufferPlacement,
    row_shape: tuple[int, ...],
) -> MutationCapability:
    return MutationCapability(
        target_key=target_key,
        target_kind=MutationTargetKind.SIMULATION_STATE,
        entity_kind=entity_kind,
        field_kind=field_kind,
        entity_count=entity_count,
        value_template=_device_manager_value_template(
            placement=placement,
            row_shape=row_shape,
        ),
        triggers=frozenset({MutationTrigger.RESET}),
        commit_phases=frozenset({MutationCommitPhase.RESET}),
        operations=frozenset({MutationOperation.SET}),
        baselines=frozenset({MutationBaseline.DEFAULT}),
        persistences=frozenset({MutationPersistence.EPISODE}),
        recompute_levels=frozenset({MutationRecomputeLevel.KINEMATICS}),
        descriptor=mjwarp_state_reset_descriptor(
            target_key=target_key,
            execution_profile=ExecutionProfile.DEVICE_RESIDENT,
        ),
    )


def _model_capability(
    *,
    target_key: str,
    field_kind: MutationFieldKind,
    entity_count: int,
    placement: BufferPlacement,
) -> MutationCapability:
    return MutationCapability(
        target_key=target_key,
        target_kind=MutationTargetKind.MODEL_PARAMETER,
        entity_kind=MutationEntityKind.ACTUATOR,
        field_kind=field_kind,
        entity_count=entity_count,
        value_template=_device_manager_value_template(
            placement=placement,
            row_shape=(1,),
        ),
        triggers=frozenset({MutationTrigger.RESET}),
        commit_phases=frozenset({MutationCommitPhase.RESET}),
        operations=frozenset({MutationOperation.SET, MutationOperation.SCALE}),
        baselines=frozenset({MutationBaseline.DEFAULT}),
        persistences=frozenset({MutationPersistence.EPISODE}),
        recompute_levels=frozenset({MutationRecomputeLevel.NONE}),
        descriptor=mjwarp_actuator_pd_descriptor(target_key=target_key),
    )


def mjwarp_device_mutation_capabilities(backend: MjwarpBackend) -> tuple[MutationCapability, ...]:
    """Declare the explicit CUDA reset-state capability surface.

    State values and the verified native position-actuator Model fields use
    CUDA manager buffers and one active mask.  Other randomization and Event
    features remain unavailable until they have their own effect evidence.
    """

    bridge = backend._ensure_device_bridge()
    index = bridge.qpos.device.index
    if index is None:
        raise BackendBatchContractError("mjwarp CUDA mutation bridge has no device index")
    placement = BufferPlacement.device("cuda", int(index))
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
                    placement=placement,
                    row_shape=row_shape,
                )
            )
    if backend._num_dof_pos > 0:
        capabilities.append(
            _reset_capability(
                target_key="state.dof.position",
                entity_kind=MutationEntityKind.DOF,
                field_kind=MutationFieldKind.POSITION,
                entity_count=backend._num_dof_pos,
                placement=placement,
                row_shape=(1,),
            )
        )
    if backend._num_dof_vel > 0:
        capabilities.append(
            _reset_capability(
                target_key="state.dof.angular_velocity",
                entity_kind=MutationEntityKind.DOF,
                field_kind=MutationFieldKind.ANGULAR_VELOCITY,
                entity_count=backend._num_dof_vel,
                placement=placement,
                row_shape=(1,),
            )
        )
    if backend._position_actuator_ids:
        capabilities.extend(
            (
                _model_capability(
                    target_key="actuator.pd_stiffness",
                    field_kind=MutationFieldKind.STIFFNESS,
                    entity_count=backend._nu,
                    placement=placement,
                ),
                _model_capability(
                    target_key="actuator.pd_damping",
                    field_kind=MutationFieldKind.DAMPING,
                    entity_count=backend._nu,
                    placement=placement,
                ),
            )
        )
    if not capabilities:
        raise BackendBatchContractError(
            "mjwarp device typed state mutation requires a floating root or at least one DoF"
        )
    return tuple(capabilities)


@dataclass(frozen=True)
class _DeviceResetTarget:
    """Cold-bound raw coordinate map for one semantic mutation field."""

    field_index: int
    spec: BoundMutationSpec
    coordinates: torch.Tensor | None = field(repr=False, compare=False)


@dataclass
class MjwarpDeviceMutationPlan:
    """Coordinate state and Model commits under one all-world reset mask."""

    public_plan: BoundMutationPlan
    nq: int
    nv: int
    root_qpos_dim: int
    root_qvel_dim: int
    placement: BufferPlacement
    model_plan: MjwarpDeviceModelMutationPlan | None = field(default=None, repr=False)
    _reset_qpos: torch.Tensor | None = field(init=False, repr=False)
    _reset_qvel: torch.Tensor | None = field(init=False, repr=False)
    _masked_qpos: torch.Tensor | None = field(init=False, repr=False)
    _masked_qvel: torch.Tensor | None = field(init=False, repr=False)
    _targets: tuple[_DeviceResetTarget, ...] = field(init=False, repr=False)
    _registered_batch_plans: dict[str, BoundBackendPlan] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.nq <= 0 or self.nv <= 0:
            raise BackendBatchContractError("mjwarp device mutation requires positive nq and nv")
        if self.placement.device_index is None:
            raise BackendBatchContractError("mjwarp device mutation placement needs a CUDA index")
        device = torch.device(f"cuda:{self.placement.device_index}")
        shape_qpos = (self.public_plan.num_envs, self.nq)
        shape_qvel = (self.public_plan.num_envs, self.nv)
        targets: list[_DeviceResetTarget] = []
        for field_index, spec in enumerate(self.public_plan.specs):
            if spec.target.target_kind is not MutationTargetKind.SIMULATION_STATE:
                continue
            target_key = spec.target.target_key
            if target_key in _ROOT_TARGETS:
                if (
                    spec.target.entity_kind is not MutationEntityKind.BODY
                    or len(spec.target.entity_ids) != 1
                ):
                    raise MutationContractError(
                        "mjwarp device root reset target must bind one base body"
                    )
                coordinates = None
            elif target_key in _DOF_TARGETS:
                if (
                    spec.target.entity_kind is not MutationEntityKind.DOF
                    or not spec.target.entity_ids
                    or spec.value_buffer.row_shape[-1] != 1
                ):
                    raise MutationContractError(
                        "mjwarp device DoF reset target must bind hinge coordinates"
                    )
                offset = (
                    self.root_qpos_dim if target_key == "state.dof.position" else self.root_qvel_dim
                )
                coordinates = torch.tensor(
                    tuple(offset + index for index in spec.target.entity_ids),
                    dtype=torch.int64,
                    device=device,
                )
            else:
                raise MutationContractError(
                    "mjwarp device reset plan contains an unsupported simulation-state target"
                )
            targets.append(
                _DeviceResetTarget(
                    field_index=field_index,
                    spec=spec,
                    coordinates=coordinates,
                )
            )
        self._targets = tuple(targets)
        if targets:
            self._reset_qpos = torch.empty(shape_qpos, dtype=torch.float32, device=device)
            self._reset_qvel = torch.empty(shape_qvel, dtype=torch.float32, device=device)
            self._masked_qpos = torch.empty(shape_qpos, dtype=torch.float32, device=device)
            self._masked_qvel = torch.empty(shape_qvel, dtype=torch.float32, device=device)
        else:
            self._reset_qpos = None
            self._reset_qvel = None
            self._masked_qpos = None
            self._masked_qvel = None
        has_model_specs = any(
            spec.target.target_kind is MutationTargetKind.MODEL_PARAMETER
            for spec in self.public_plan.specs
        )
        if has_model_specs != bool(self.model_plan and self.model_plan.has_targets):
            raise MutationContractError(
                "mjwarp device mutation plan Model targets do not match their runtime owner"
            )
        if self.model_plan is not None:
            self.model_plan.public_plan.require_compatible(self.public_plan)

    def register_batch_plan(self, plan: BoundBackendPlan) -> None:
        """Pair one device state/control plan on the cold path."""

        if not isinstance(plan, BoundBackendPlan):
            raise BackendBatchContractError("mjwarp device state plan must be a BoundBackendPlan")
        if plan.num_envs != self.public_plan.num_envs:
            raise BackendBatchContractError(
                "mjwarp device mutation and state plans need one row universe"
            )
        if plan.execution_profile is not ExecutionProfile.DEVICE_RESIDENT:
            raise BackendBatchContractError(
                "mjwarp device mutation only pairs with device_resident batch plans"
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
                "mjwarp device mutation plan was not cold-path paired with this batch plan"
            ) from exc
        registered.require_compatible(plan)

    @staticmethod
    def _value_tensor(value: MutationValueBatch, *, expected: BoundMutationSpec) -> torch.Tensor:
        if not isinstance(value, MutationValueBatch) or value.spec != expected:
            raise BackendBatchContractError(
                "mjwarp device reset value does not match its bound field"
            )
        view = require_device_tensor_view(
            value.buffer.handle,
            contract=expected.value_buffer,
            require_completion=True,
        )
        return view.torch()

    def _ordered_values(
        self,
        batch: DeviceResetMutationBatch,
    ) -> tuple[torch.Tensor, ...]:
        mutation = batch.mutation
        values_by_index = {value.field_index: value for value in mutation.state.values}
        expected_indices = tuple(target.field_index for target in self._targets)
        if tuple(sorted(values_by_index)) != tuple(sorted(expected_indices)):
            raise BackendBatchContractError(
                "mjwarp device reset must supply every bound simulation-state field once"
            )
        return tuple(
            self._value_tensor(values_by_index[target.field_index], expected=target.spec)
            for target in self._targets
        )

    def active_mask(self, batch: DeviceResetMutationBatch) -> torch.Tensor:
        """Return the validated CUDA row mask for the reset-data barrier.

        The caller copies this manager-owned view into the backend-owned Warp
        reset mask before ``reset_data``.  Keeping that copy at the backend
        owner layer lets the mutation plan remain agnostic to raw Warp arrays
        while preserving one device-only partial-reset transaction.
        """

        self.public_plan.require_compatible(batch.plan)
        view = require_device_tensor_view(
            batch.active_mask.handle,
            contract=batch.active_mask.contract,
            require_completion=True,
        )
        mask = view.torch()
        if (
            tuple(mask.shape) != (self.public_plan.num_envs,)
            or mask.dtype is not torch.bool
            or mask.device != torch.device(f"cuda:{self.placement.device_index}")
        ):
            raise BackendBatchContractError(
                "mjwarp device reset active mask does not match the cold-bound CUDA plan"
            )
        return mask

    @property
    def numeric_buffer_addresses(self) -> tuple[int, ...]:
        """Return stable scratch identity for low-frequency allocation gates."""

        buffers = tuple(
            value
            for value in (
                self._reset_qpos,
                self._reset_qvel,
                self._masked_qpos,
                self._masked_qvel,
            )
            if value is not None
        )
        addresses = tuple(int(value.data_ptr()) for value in buffers)
        if self.model_plan is not None:
            addresses = (*addresses, *self.model_plan.numeric_buffer_addresses)
        return addresses

    def validate_batch(self, batch: DeviceResetMutationBatch) -> None:
        """Validate complete typed coverage before launching a reset graph."""

        self.public_plan.require_compatible(batch.plan)
        self.active_mask(batch)
        self._ordered_values(batch)
        if self.model_plan is None:
            if batch.mutation.model.values:
                raise BackendBatchContractError(
                    "mjwarp device reset supplied Model values without a bound Model plan"
                )
        else:
            self.model_plan.ordered_values(batch)

    def commit_model(
        self,
        batch: DeviceResetMutationBatch,
        *,
        active_mask: torch.Tensor,
    ) -> None:
        if self.model_plan is not None:
            self.model_plan.commit(batch, active_mask=active_mask)

    def effective_active_mask(
        self,
        batch: DeviceResetMutationBatch,
        *,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gate malformed Model values on CUDA before any reset-side write."""

        if self.model_plan is None:
            return active_mask
        return self.model_plan.effective_active_mask(batch, active_mask=active_mask)

    def stage_reset_state(
        self,
        batch: DeviceResetMutationBatch,
        *,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        """Apply all semantic values under a CUDA mask without host indexing."""

        self.public_plan.require_compatible(batch.plan)
        expected_qpos = (self.public_plan.num_envs, self.nq)
        expected_qvel = (self.public_plan.num_envs, self.nv)
        if (
            tuple(qpos.shape) != expected_qpos
            or tuple(qvel.shape) != expected_qvel
            or qpos.dtype is not torch.float32
            or qvel.dtype is not torch.float32
            or qpos.device != torch.device(f"cuda:{self.placement.device_index}")
            or qvel.device != torch.device(f"cuda:{self.placement.device_index}")
        ):
            raise BackendBatchContractError(
                "mjwarp device reset source tensors do not match the cold-bound plan"
            )
        values = self._ordered_values(batch)
        if not self._targets:
            return
        mask = active_mask
        if (
            tuple(mask.shape) != (self.public_plan.num_envs,)
            or mask.dtype is not torch.bool
            or mask.device != qpos.device
        ):
            raise BackendBatchContractError(
                "mjwarp device reset effective mask does not match the cold-bound plan"
            )
        assert self._reset_qpos is not None
        assert self._reset_qvel is not None
        assert self._masked_qpos is not None
        assert self._masked_qvel is not None
        self._reset_qpos.copy_(qpos, non_blocking=True)
        self._reset_qvel.copy_(qvel, non_blocking=True)
        for target, value in zip(self._targets, values, strict=True):
            target_key = target.spec.target.target_key
            if target_key == "state.root.position":
                self._reset_qpos[:, 0:3].copy_(value[:, 0, :], non_blocking=True)
            elif target_key == "state.root.orientation":
                self._reset_qpos[:, 3:7].copy_(value[:, 0, :], non_blocking=True)
            elif target_key == "state.root.linear_velocity":
                self._reset_qvel[:, 0:3].copy_(value[:, 0, :], non_blocking=True)
            elif target_key == "state.root.angular_velocity":
                self._reset_qvel[:, 3:6].copy_(value[:, 0, :], non_blocking=True)
            elif target_key == "state.dof.position":
                assert target.coordinates is not None
                self._reset_qpos.index_copy_(1, target.coordinates, value[:, :, 0])
            elif target_key == "state.dof.angular_velocity":
                assert target.coordinates is not None
                self._reset_qvel.index_copy_(1, target.coordinates, value[:, :, 0])
            else:  # pragma: no cover - cold-path target validation above.
                raise MutationContractError(
                    "mjwarp device reset plan contains an unsupported target"
                )
        mask_2d = mask.view(self.public_plan.num_envs, 1)
        torch.where(mask_2d, self._reset_qpos, qpos, out=self._masked_qpos)
        torch.where(mask_2d, self._reset_qvel, qvel, out=self._masked_qvel)
        qpos.copy_(self._masked_qpos, non_blocking=True)
        qvel.copy_(self._masked_qvel, non_blocking=True)


__all__ = ["MjwarpDeviceMutationPlan", "mjwarp_device_mutation_capabilities"]
