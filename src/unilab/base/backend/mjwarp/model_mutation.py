"""Device-resident Model mutation owned by the ``mjwarp`` backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..batch import BackendBatchContractError, BufferPlacement
from ..device import require_device_tensor_view
from ..mutation import (
    BoundMutationPlan,
    BoundMutationSpec,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationOperation,
    MutationTargetKind,
)
from ..mutation_batch import DeviceResetMutationBatch, MutationValueBatch


@dataclass(frozen=True)
class _DevicePdTarget:
    field_index: int
    spec: BoundMutationSpec
    actuator_indices: torch.Tensor = field(repr=False, compare=False)
    default_gain: torch.Tensor | None = field(repr=False, compare=False)
    default_bias: torch.Tensor = field(repr=False, compare=False)
    target_values: torch.Tensor = field(repr=False, compare=False)
    current_values: torch.Tensor = field(repr=False, compare=False)
    masked_values: torch.Tensor = field(repr=False, compare=False)
    value_above_lower: torch.Tensor = field(repr=False, compare=False)
    value_below_upper: torch.Tensor = field(repr=False, compare=False)
    valid_worlds: torch.Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class _DeviceScalarTarget:
    field_index: int
    spec: BoundMutationSpec
    entity_indices: torch.Tensor = field(repr=False, compare=False)
    model_field: torch.Tensor = field(repr=False, compare=False)
    default_values: torch.Tensor = field(repr=False, compare=False)
    nonnegative: bool
    target_values: torch.Tensor = field(repr=False, compare=False)
    current_values: torch.Tensor = field(repr=False, compare=False)
    masked_values: torch.Tensor = field(repr=False, compare=False)
    value_above_lower: torch.Tensor = field(repr=False, compare=False)
    value_below_upper: torch.Tensor = field(repr=False, compare=False)
    valid_worlds: torch.Tensor = field(repr=False, compare=False)


_DeviceModelTarget = _DevicePdTarget | _DeviceScalarTarget


@dataclass
class MjwarpDeviceModelMutationPlan:
    """Preallocated selected-world commits for supported Model parameters."""

    public_plan: BoundMutationPlan
    placement: BufferPlacement
    model_fields: dict[str, torch.Tensor] = field(repr=False)
    default_fields: dict[str, torch.Tensor] = field(repr=False)
    root_qvel_dim: int
    _targets: tuple[_DeviceModelTarget, ...] = field(init=False, repr=False)
    _effective_mask: torch.Tensor = field(init=False, repr=False)
    _effective_mask_2d: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.placement.device_index is None:
            raise BackendBatchContractError("mjwarp Model mutation placement needs a CUDA index")
        if (
            isinstance(self.root_qvel_dim, bool)
            or not isinstance(self.root_qvel_dim, int)
            or self.root_qvel_dim < 0
        ):
            raise BackendBatchContractError("mjwarp Model mutation root_qvel_dim is invalid")
        device = torch.device(f"cuda:{self.placement.device_index}")
        expected_fields: set[str] = set()
        for spec in self.public_plan.specs:
            if spec.target.target_kind is not MutationTargetKind.MODEL_PARAMETER:
                continue
            expected_fields.update(
                {
                    "actuator.pd_stiffness": ("actuator_biasprm", "actuator_gainprm"),
                    "actuator.pd_damping": ("actuator_biasprm",),
                    "joint.armature": ("dof_armature",),
                    "body.gravity_compensation": ("body_gravcomp",),
                }.get(spec.target.target_key, ())
            )
        if set(self.model_fields) != expected_fields or set(self.default_fields) != expected_fields:
            raise BackendBatchContractError(
                "mjwarp Model mutation fields do not exactly match the bound plan"
            )
        for field_name in sorted(expected_fields):
            model_field = self.model_fields[field_name]
            default_field = self.default_fields[field_name]
            if (
                not isinstance(model_field, torch.Tensor)
                or not isinstance(default_field, torch.Tensor)
                or model_field.device != device
                or default_field.device != device
                or model_field.dtype is not torch.float32
                or default_field.dtype is not torch.float32
                or model_field.ndim not in {2, 3}
                or default_field.ndim != model_field.ndim
                or model_field.shape[0] != self.public_plan.num_envs
                or tuple(default_field.shape) != (1, *tuple(model_field.shape[1:]))
            ):
                raise BackendBatchContractError(
                    f"mjwarp Model field {field_name!r} or its default has an invalid CUDA layout"
                )

        targets: list[_DeviceModelTarget] = []
        for field_index, spec in enumerate(self.public_plan.specs):
            if spec.target.target_kind is not MutationTargetKind.MODEL_PARAMETER:
                continue
            target_key = spec.target.target_key
            if not spec.target.entity_ids or spec.value_buffer.row_shape != (
                len(spec.target.entity_ids),
                1,
            ):
                raise MutationContractError(
                    "mjwarp Model mutation target has an invalid selector or value layout"
                )
            shape = (self.public_plan.num_envs, len(spec.target.entity_ids))
            scratch = self._target_scratch(shape, device=device)
            if target_key in {"actuator.pd_stiffness", "actuator.pd_damping"}:
                targets.append(
                    self._bind_pd_target(
                        field_index=field_index,
                        spec=spec,
                        device=device,
                        scratch=scratch,
                    )
                )
            elif target_key in {"joint.armature", "body.gravity_compensation"}:
                targets.append(
                    self._bind_scalar_target(
                        field_index=field_index,
                        spec=spec,
                        device=device,
                        scratch=scratch,
                    )
                )
            else:
                raise MutationContractError(f"unsupported mjwarp Model target {target_key!r}")
        self._targets = tuple(targets)
        self._effective_mask = torch.empty(
            (self.public_plan.num_envs,),
            dtype=torch.bool,
            device=device,
        )
        self._effective_mask_2d = self._effective_mask.view(self.public_plan.num_envs, 1)

    @staticmethod
    def _target_scratch(
        shape: tuple[int, int],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty((shape[0],), dtype=torch.bool, device=device),
        )

    def _bind_pd_target(
        self,
        *,
        field_index: int,
        spec: BoundMutationSpec,
        device: torch.device,
        scratch: tuple[torch.Tensor, ...],
    ) -> _DevicePdTarget:
        target_key = spec.target.target_key
        expected_field = {
            "actuator.pd_stiffness": MutationFieldKind.STIFFNESS,
            "actuator.pd_damping": MutationFieldKind.DAMPING,
        }[target_key]
        gain = self.model_fields.get("actuator_gainprm")
        bias = self.model_fields["actuator_biasprm"]
        default_gainprm = self.default_fields.get("actuator_gainprm")
        default_biasprm = self.default_fields["actuator_biasprm"]
        if (
            spec.target.entity_kind is not MutationEntityKind.ACTUATOR
            or spec.target.field_kind is not expected_field
            or bias.ndim != 3
            or bias.shape[2] < 3
            or default_biasprm.ndim != 3
        ):
            raise MutationContractError(
                "mjwarp Model mutation plan contains an unsupported actuator target"
            )
        actuator_indices = torch.tensor(
            spec.target.entity_ids,
            dtype=torch.int64,
            device=device,
        )
        if min(spec.target.entity_ids) < 0 or max(spec.target.entity_ids) >= bias.shape[1]:
            raise MutationContractError("mjwarp actuator Model coordinate is out of range")
        default_gain = None
        if target_key == "actuator.pd_stiffness":
            if gain is None or default_gainprm is None or gain.ndim != 3 or gain.shape[2] < 1:
                raise MutationContractError("mjwarp stiffness Model fields are incomplete")
            default_gain = torch.index_select(
                default_gainprm[:, :, 0],
                1,
                actuator_indices,
            ).contiguous()
            default_bias = torch.index_select(
                default_biasprm[:, :, 1],
                1,
                actuator_indices,
            ).contiguous()
        else:
            default_bias = torch.index_select(
                default_biasprm[:, :, 2],
                1,
                actuator_indices,
            ).contiguous()
        return _DevicePdTarget(
            field_index=field_index,
            spec=spec,
            actuator_indices=actuator_indices,
            default_gain=default_gain,
            default_bias=default_bias,
            target_values=scratch[0],
            current_values=scratch[1],
            masked_values=scratch[2],
            value_above_lower=scratch[3],
            value_below_upper=scratch[4],
            valid_worlds=scratch[5],
        )

    def _bind_scalar_target(
        self,
        *,
        field_index: int,
        spec: BoundMutationSpec,
        device: torch.device,
        scratch: tuple[torch.Tensor, ...],
    ) -> _DeviceScalarTarget:
        target_key = spec.target.target_key
        if target_key == "joint.armature":
            expected_entity = MutationEntityKind.DOF
            expected_kind = MutationFieldKind.ARMATURE
            field_name = "dof_armature"
            raw_ids = tuple(self.root_qvel_dim + entity_id for entity_id in spec.target.entity_ids)
            nonnegative = True
        else:
            expected_entity = MutationEntityKind.BODY
            expected_kind = MutationFieldKind.GRAVITY_COMPENSATION
            field_name = "body_gravcomp"
            raw_ids = spec.target.entity_ids
            nonnegative = False
        model_field = self.model_fields[field_name]
        default_field = self.default_fields[field_name]
        if (
            spec.target.entity_kind is not expected_entity
            or spec.target.field_kind is not expected_kind
            or model_field.ndim != 2
            or default_field.ndim != 2
            or min(raw_ids) < 0
            or max(raw_ids) >= model_field.shape[1]
        ):
            raise MutationContractError(
                f"mjwarp Model mutation plan contains an invalid {target_key!r} target"
            )
        entity_indices = torch.tensor(raw_ids, dtype=torch.int64, device=device)
        default_values = torch.index_select(
            default_field,
            1,
            entity_indices,
        ).contiguous()
        return _DeviceScalarTarget(
            field_index=field_index,
            spec=spec,
            entity_indices=entity_indices,
            model_field=model_field,
            default_values=default_values,
            nonnegative=nonnegative,
            target_values=scratch[0],
            current_values=scratch[1],
            masked_values=scratch[2],
            value_above_lower=scratch[3],
            value_below_upper=scratch[4],
            valid_worlds=scratch[5],
        )

    @property
    def has_targets(self) -> bool:
        return bool(self._targets)

    @property
    def numeric_buffer_addresses(self) -> tuple[int, ...]:
        buffers: list[torch.Tensor] = []
        for target in self._targets:
            if isinstance(target, _DevicePdTarget):
                buffers.extend((target.actuator_indices, target.default_bias))
                if target.default_gain is not None:
                    buffers.append(target.default_gain)
            else:
                buffers.extend((target.entity_indices, target.default_values))
            buffers.extend(
                (
                    target.target_values,
                    target.current_values,
                    target.masked_values,
                    target.value_above_lower,
                    target.value_below_upper,
                    target.valid_worlds,
                )
            )
        buffers.append(self._effective_mask)
        return tuple(int(value.data_ptr()) for value in buffers)

    @property
    def effective_mask(self) -> torch.Tensor:
        return self._effective_mask

    @staticmethod
    def _value_tensor(value: MutationValueBatch, *, expected: BoundMutationSpec) -> torch.Tensor:
        if not isinstance(value, MutationValueBatch) or value.spec != expected:
            raise BackendBatchContractError(
                "mjwarp Model mutation value does not match its bound field"
            )
        view = require_device_tensor_view(
            value.buffer.handle,
            contract=expected.value_buffer,
            require_completion=True,
        )
        return view.torch()

    def ordered_values(self, batch: DeviceResetMutationBatch) -> tuple[torch.Tensor, ...]:
        values_by_index = {value.field_index: value for value in batch.mutation.model.values}
        expected_indices = tuple(target.field_index for target in self._targets)
        if tuple(sorted(values_by_index)) != tuple(sorted(expected_indices)):
            raise BackendBatchContractError(
                "mjwarp device reset must supply every bound Model field once"
            )
        values = tuple(
            self._value_tensor(values_by_index[target.field_index], expected=target.spec)
            for target in self._targets
        )
        for target, value in zip(self._targets, values, strict=True):
            expected_shape = (
                self.public_plan.num_envs,
                len(target.spec.target.entity_ids),
                1,
            )
            if (
                tuple(value.shape) != expected_shape
                or value.dtype is not torch.float32
                or value.device != target.target_values.device
            ):
                raise BackendBatchContractError(
                    "mjwarp Model mutation value differs from its cold-bound CUDA plan"
                )
        return values

    def effective_active_mask(
        self,
        batch: DeviceResetMutationBatch,
        *,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gate malformed Model values per row entirely on CUDA."""

        values = self.ordered_values(batch)
        if (
            tuple(active_mask.shape) != (self.public_plan.num_envs,)
            or active_mask.dtype is not torch.bool
            or active_mask.device != self._effective_mask.device
        ):
            raise BackendBatchContractError(
                "mjwarp Model mutation mask differs from its cold-bound CUDA plan"
            )
        self._effective_mask.copy_(active_mask, non_blocking=True)
        for target, value in zip(self._targets, values, strict=True):
            source = value[:, :, 0]
            if isinstance(target, _DeviceScalarTarget):
                if target.spec.operation is MutationOperation.SET:
                    target.target_values.copy_(source, non_blocking=True)
                elif target.spec.operation is MutationOperation.SCALE:
                    torch.mul(target.default_values, source, out=target.target_values)
                else:  # pragma: no cover - capability binding rejects this.
                    raise MutationContractError("unsupported mjwarp scalar Model operation")
                candidate = target.target_values
                lower = 0.0 if target.nonnegative else float("-inf")
            else:
                default = target.default_gain
                damping = target.spec.target.target_key == "actuator.pd_damping"
                if damping:
                    default = target.default_bias
                if target.spec.operation is MutationOperation.SET:
                    target.target_values.copy_(source, non_blocking=True)
                elif target.spec.operation is MutationOperation.SCALE:
                    assert default is not None
                    torch.mul(default, source, out=target.target_values)
                    if damping:
                        torch.neg(target.target_values, out=target.target_values)
                else:  # pragma: no cover - capability binding rejects this.
                    raise MutationContractError("unsupported mjwarp actuator Model operation")
                candidate = target.target_values
                lower = 0.0
            if lower == float("-inf"):
                torch.gt(candidate, lower, out=target.value_above_lower)
            else:
                torch.ge(candidate, lower, out=target.value_above_lower)
            torch.lt(candidate, float("inf"), out=target.value_below_upper)
            torch.logical_and(
                target.value_above_lower,
                target.value_below_upper,
                out=target.value_above_lower,
            )
            torch.all(target.value_above_lower, dim=1, out=target.valid_worlds)
            torch.logical_and(
                self._effective_mask,
                target.valid_worlds,
                out=self._effective_mask,
            )
        return self._effective_mask

    @staticmethod
    def _commit_slot(
        *,
        model_field: torch.Tensor,
        slot: int,
        target: _DevicePdTarget,
        active_mask_2d: torch.Tensor,
    ) -> None:
        field_slot = model_field[:, :, slot]
        torch.index_select(field_slot, 1, target.actuator_indices, out=target.current_values)
        torch.where(
            active_mask_2d,
            target.target_values,
            target.current_values,
            out=target.masked_values,
        )
        field_slot.index_copy_(1, target.actuator_indices, target.masked_values)

    @staticmethod
    def _commit_scalar(
        *,
        target: _DeviceScalarTarget,
        active_mask_2d: torch.Tensor,
    ) -> None:
        torch.index_select(
            target.model_field,
            1,
            target.entity_indices,
            out=target.current_values,
        )
        torch.where(
            active_mask_2d,
            target.target_values,
            target.current_values,
            out=target.masked_values,
        )
        target.model_field.index_copy_(1, target.entity_indices, target.masked_values)

    def commit(self, batch: DeviceResetMutationBatch, *, active_mask: torch.Tensor) -> None:
        """Commit selected rows from immutable defaults without warm allocation."""

        values = self.ordered_values(batch)
        if not self._targets:
            return
        if (
            tuple(active_mask.shape) != (self.public_plan.num_envs,)
            or active_mask.dtype is not torch.bool
            or active_mask.device != self._targets[0].target_values.device
        ):
            raise BackendBatchContractError(
                "mjwarp Model mutation mask differs from its cold-bound CUDA plan"
            )
        active_mask_2d = (
            self._effective_mask_2d
            if active_mask.data_ptr() == self._effective_mask.data_ptr()
            else active_mask.view(self.public_plan.num_envs, 1)
        )
        for target, value in zip(self._targets, values, strict=True):
            source = value[:, :, 0]
            if isinstance(target, _DeviceScalarTarget):
                if target.spec.operation is MutationOperation.SET:
                    target.target_values.copy_(source, non_blocking=True)
                elif target.spec.operation is MutationOperation.SCALE:
                    torch.mul(target.default_values, source, out=target.target_values)
                else:  # pragma: no cover - capability binding rejects this.
                    raise MutationContractError("unsupported mjwarp scalar Model operation")
                self._commit_scalar(target=target, active_mask_2d=active_mask_2d)
                continue

            target_key = target.spec.target.target_key
            if target_key == "actuator.pd_stiffness":
                if target.spec.operation is MutationOperation.SET:
                    target.target_values.copy_(source, non_blocking=True)
                elif target.spec.operation is MutationOperation.SCALE:
                    assert target.default_gain is not None
                    torch.mul(target.default_gain, source, out=target.target_values)
                else:  # pragma: no cover - capability binding rejects this.
                    raise MutationContractError("unsupported mjwarp stiffness operation")
                self._commit_slot(
                    model_field=self.model_fields["actuator_gainprm"],
                    slot=0,
                    target=target,
                    active_mask_2d=active_mask_2d,
                )
                if target.spec.operation is MutationOperation.SET:
                    torch.neg(source, out=target.target_values)
                else:
                    torch.mul(target.default_bias, source, out=target.target_values)
                self._commit_slot(
                    model_field=self.model_fields["actuator_biasprm"],
                    slot=1,
                    target=target,
                    active_mask_2d=active_mask_2d,
                )
            elif target_key == "actuator.pd_damping":
                if target.spec.operation is MutationOperation.SET:
                    torch.neg(source, out=target.target_values)
                elif target.spec.operation is MutationOperation.SCALE:
                    torch.mul(target.default_bias, source, out=target.target_values)
                else:  # pragma: no cover - capability binding rejects this.
                    raise MutationContractError("unsupported mjwarp damping operation")
                self._commit_slot(
                    model_field=self.model_fields["actuator_biasprm"],
                    slot=2,
                    target=target,
                    active_mask_2d=active_mask_2d,
                )


__all__ = ["MjwarpDeviceModelMutationPlan"]
