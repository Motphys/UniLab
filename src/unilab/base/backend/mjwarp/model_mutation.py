"""Device-resident actuator Model mutation owned by the ``mjwarp`` backend."""

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
class _DeviceModelTarget:
    """Cold-bound actuator coordinates, defaults, and reusable CUDA scratch."""

    field_index: int
    spec: BoundMutationSpec
    actuator_indices: torch.Tensor = field(repr=False, compare=False)
    default_gain: torch.Tensor | None = field(repr=False, compare=False)
    default_bias: torch.Tensor = field(repr=False, compare=False)
    target_values: torch.Tensor = field(repr=False, compare=False)
    current_values: torch.Tensor = field(repr=False, compare=False)
    masked_values: torch.Tensor = field(repr=False, compare=False)
    value_in_range: torch.Tensor = field(repr=False, compare=False)
    value_below_infinity: torch.Tensor = field(repr=False, compare=False)
    valid_worlds: torch.Tensor = field(repr=False, compare=False)


@dataclass
class MjwarpDeviceModelMutationPlan:
    """Preallocated selected-world commits for native position actuators.

    MuJoCo position servos encode ``kp`` in both ``gainprm[0]`` and
    ``-biasprm[1]`` and encode ``kd`` in ``-biasprm[2]``.  The plan binds that
    physical layout once and never exposes it to a manager or runtime.
    """

    public_plan: BoundMutationPlan
    placement: BufferPlacement
    actuator_gainprm: torch.Tensor = field(repr=False)
    actuator_biasprm: torch.Tensor = field(repr=False)
    default_gainprm: torch.Tensor = field(repr=False)
    default_biasprm: torch.Tensor = field(repr=False)
    _targets: tuple[_DeviceModelTarget, ...] = field(init=False, repr=False)
    _effective_mask: torch.Tensor = field(init=False, repr=False)
    _effective_mask_2d: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.placement.device_index is None:
            raise BackendBatchContractError("mjwarp Model mutation placement needs a CUDA index")
        device = torch.device(f"cuda:{self.placement.device_index}")
        tensors = (
            self.actuator_gainprm,
            self.actuator_biasprm,
            self.default_gainprm,
            self.default_biasprm,
        )
        if any(
            not isinstance(value, torch.Tensor)
            or value.device != device
            or value.dtype is not torch.float32
            or value.ndim != 3
            for value in tensors
        ):
            raise BackendBatchContractError(
                "mjwarp actuator Model mutation requires three-dimensional CUDA float32 storage"
            )
        gain_shape = tuple(int(dim) for dim in self.actuator_gainprm.shape)
        bias_shape = tuple(int(dim) for dim in self.actuator_biasprm.shape)
        default_gain_shape = tuple(int(dim) for dim in self.default_gainprm.shape)
        default_bias_shape = tuple(int(dim) for dim in self.default_biasprm.shape)
        if (
            gain_shape[0] != self.public_plan.num_envs
            or bias_shape[0] != self.public_plan.num_envs
            or gain_shape[1] != bias_shape[1]
            or gain_shape[2] < 1
            or bias_shape[2] < 3
            or default_gain_shape != (1, *gain_shape[1:])
            or default_bias_shape != (1, *bias_shape[1:])
        ):
            raise BackendBatchContractError(
                "mjwarp actuator Model fields or compiled defaults have incompatible shapes"
            )

        targets: list[_DeviceModelTarget] = []
        for field_index, spec in enumerate(self.public_plan.specs):
            if spec.target.target_kind is not MutationTargetKind.MODEL_PARAMETER:
                continue
            target_key = spec.target.target_key
            expected_field = {
                "actuator.pd_stiffness": MutationFieldKind.STIFFNESS,
                "actuator.pd_damping": MutationFieldKind.DAMPING,
            }.get(target_key)
            if (
                spec.target.entity_kind is not MutationEntityKind.ACTUATOR
                or spec.target.field_kind is not expected_field
                or not spec.target.entity_ids
                or spec.value_buffer.row_shape != (len(spec.target.entity_ids), 1)
            ):
                raise MutationContractError(
                    "mjwarp Model mutation plan contains an unsupported actuator target"
                )
            actuator_indices = torch.tensor(
                spec.target.entity_ids,
                dtype=torch.int64,
                device=device,
            )
            default_gain = None
            if target_key == "actuator.pd_stiffness":
                default_gain = torch.index_select(
                    self.default_gainprm[:, :, 0],
                    1,
                    actuator_indices,
                ).contiguous()
                default_bias = torch.index_select(
                    self.default_biasprm[:, :, 1],
                    1,
                    actuator_indices,
                ).contiguous()
            elif target_key == "actuator.pd_damping":
                default_bias = torch.index_select(
                    self.default_biasprm[:, :, 2],
                    1,
                    actuator_indices,
                ).contiguous()
            else:  # pragma: no cover - guarded by the semantic map above.
                raise MutationContractError("unsupported mjwarp actuator Model target")
            shape = (self.public_plan.num_envs, len(spec.target.entity_ids))
            targets.append(
                _DeviceModelTarget(
                    field_index=field_index,
                    spec=spec,
                    actuator_indices=actuator_indices,
                    default_gain=default_gain,
                    default_bias=default_bias,
                    target_values=torch.empty(shape, dtype=torch.float32, device=device),
                    current_values=torch.empty(shape, dtype=torch.float32, device=device),
                    masked_values=torch.empty(shape, dtype=torch.float32, device=device),
                    value_in_range=torch.empty(shape, dtype=torch.bool, device=device),
                    value_below_infinity=torch.empty(shape, dtype=torch.bool, device=device),
                    valid_worlds=torch.empty(
                        (self.public_plan.num_envs,),
                        dtype=torch.bool,
                        device=device,
                    ),
                )
            )
        self._targets = tuple(targets)
        self._effective_mask = torch.empty(
            (self.public_plan.num_envs,),
            dtype=torch.bool,
            device=device,
        )
        self._effective_mask_2d = self._effective_mask.view(self.public_plan.num_envs, 1)

    @property
    def has_targets(self) -> bool:
        return bool(self._targets)

    @property
    def numeric_buffer_addresses(self) -> tuple[int, ...]:
        """Expose stable allocation identity for low-frequency regression tests."""

        buffers: list[torch.Tensor] = []
        for target in self._targets:
            buffers.extend(
                (
                    target.actuator_indices,
                    target.default_bias,
                    target.target_values,
                    target.current_values,
                    target.masked_values,
                    target.value_in_range,
                    target.value_below_infinity,
                    target.valid_worlds,
                )
            )
            if target.default_gain is not None:
                buffers.append(target.default_gain)
        buffers.append(self._effective_mask)
        return tuple(int(value.data_ptr()) for value in buffers)

    @property
    def effective_mask(self) -> torch.Tensor:
        """Expose the preallocated validity-gated mask for diagnostics."""

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
        """Validate complete Model coverage before any physics or field write."""

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
        """Reject non-finite or negative PD values per world without a host read."""

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
            torch.ge(source, 0.0, out=target.value_in_range)
            torch.lt(source, float("inf"), out=target.value_below_infinity)
            torch.logical_and(
                target.value_in_range,
                target.value_below_infinity,
                out=target.value_in_range,
            )
            torch.all(target.value_in_range, dim=1, out=target.valid_worlds)
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
        target: _DeviceModelTarget,
        active_mask_2d: torch.Tensor,
    ) -> None:
        field_slot = model_field[:, :, slot]
        torch.index_select(
            field_slot,
            1,
            target.actuator_indices,
            out=target.current_values,
        )
        torch.where(
            active_mask_2d,
            target.target_values,
            target.current_values,
            out=target.masked_values,
        )
        field_slot.index_copy_(1, target.actuator_indices, target.masked_values)

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
        if active_mask.data_ptr() == self._effective_mask.data_ptr():
            active_mask_2d = self._effective_mask_2d
        else:
            active_mask_2d = active_mask.view(self.public_plan.num_envs, 1)
        for target, value in zip(self._targets, values, strict=True):
            source = value[:, :, 0]
            if target.spec.target.target_key == "actuator.pd_stiffness":
                if target.spec.operation is MutationOperation.SET:
                    target.target_values.copy_(source, non_blocking=True)
                elif target.spec.operation is MutationOperation.SCALE:
                    assert target.default_gain is not None
                    torch.mul(target.default_gain, source, out=target.target_values)
                else:  # pragma: no cover - capability binding rejects this.
                    raise MutationContractError("unsupported mjwarp stiffness operation")
                self._commit_slot(
                    model_field=self.actuator_gainprm,
                    slot=0,
                    target=target,
                    active_mask_2d=active_mask_2d,
                )
                if target.spec.operation is MutationOperation.SET:
                    torch.neg(source, out=target.target_values)
                else:
                    torch.mul(target.default_bias, source, out=target.target_values)
                self._commit_slot(
                    model_field=self.actuator_biasprm,
                    slot=1,
                    target=target,
                    active_mask_2d=active_mask_2d,
                )
            elif target.spec.target.target_key == "actuator.pd_damping":
                if target.spec.operation is MutationOperation.SET:
                    torch.neg(source, out=target.target_values)
                elif target.spec.operation is MutationOperation.SCALE:
                    torch.mul(target.default_bias, source, out=target.target_values)
                else:  # pragma: no cover - capability binding rejects this.
                    raise MutationContractError("unsupported mjwarp damping operation")
                self._commit_slot(
                    model_field=self.actuator_biasprm,
                    slot=2,
                    target=target,
                    active_mask_2d=active_mask_2d,
                )


__all__ = ["MjwarpDeviceModelMutationPlan"]
