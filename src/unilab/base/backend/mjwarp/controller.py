"""Cold-bound device substep controllers owned by the ``mjwarp`` backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from unilab.base.backend.batch import (
    BackendBatchContractError,
    ControlImplementation,
    ControllerStateReadPhase,
    ControlSpec,
    StateEntityKind,
    StateFieldKind,
    StateFieldSpec,
)
from unilab.base.backend.graph import DeviceGraphBufferAddress

if TYPE_CHECKING:
    from .backend import MjwarpBackend


MJWARP_JOINT_PD_TORQUE_CONTROLLER = "mjwarp.joint_pd_torque.v1"

_POSITION_READ = "dof.position"
_VELOCITY_READ = "dof.velocity"
_PARAMETER_KEYS = ("damping", "effort_limit", "stiffness")


def _tensor_address(name: str, tensor: torch.Tensor) -> DeviceGraphBufferAddress:
    return DeviceGraphBufferAddress(
        name=name,
        address=int(tensor.data_ptr()),
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device=str(tensor.device),
    )


@dataclass(frozen=True)
class MjwarpDeviceControllerRuntime:
    """Stable CUDA storage and graph-capturable joint PD implementation."""

    implementation_key: str
    command: torch.Tensor = field(repr=False)
    position_source: torch.Tensor = field(repr=False)
    velocity_source: torch.Tensor = field(repr=False)
    position_indices: torch.Tensor = field(repr=False)
    velocity_indices: torch.Tensor = field(repr=False)
    position: torch.Tensor = field(repr=False)
    velocity: torch.Tensor = field(repr=False)
    error: torch.Tensor = field(repr=False)
    damping_force: torch.Tensor = field(repr=False)
    stiffness: torch.Tensor = field(repr=False)
    damping: torch.Tensor = field(repr=False)
    effort_lower: torch.Tensor = field(repr=False)
    effort_upper: torch.Tensor = field(repr=False)
    native_ctrl: torch.Tensor = field(repr=False)

    def compute(self) -> None:
        """Compute native torque from state read at the current substep boundary."""

        torch.index_select(
            self.position_source,
            dim=1,
            index=self.position_indices,
            out=self.position,
        )
        torch.index_select(
            self.velocity_source,
            dim=1,
            index=self.velocity_indices,
            out=self.velocity,
        )
        torch.sub(self.command, self.position, out=self.error)
        torch.mul(self.error, self.stiffness, out=self.native_ctrl)
        torch.mul(self.velocity, self.damping, out=self.damping_force)
        torch.sub(self.native_ctrl, self.damping_force, out=self.native_ctrl)
        torch.maximum(self.native_ctrl, self.effort_lower, out=self.native_ctrl)
        torch.minimum(self.native_ctrl, self.effort_upper, out=self.native_ctrl)

    def storage_buffers(self, *, prefix: str) -> tuple[DeviceGraphBufferAddress, ...]:
        """Return every controller-owned allocation reachable from a step graph."""

        tensors = {
            "command": self.command,
            "damping": self.damping,
            "damping_force": self.damping_force,
            "effort_lower": self.effort_lower,
            "effort_upper": self.effort_upper,
            "error": self.error,
            "position": self.position,
            "position_indices": self.position_indices,
            "stiffness": self.stiffness,
            "velocity": self.velocity,
            "velocity_indices": self.velocity_indices,
        }
        return tuple(_tensor_address(f"{prefix}.{name}", tensors[name]) for name in sorted(tensors))

    @property
    def numeric_buffer_addresses(self) -> tuple[int, ...]:
        """Expose stable owner addresses for low-frequency allocation gates."""

        return tuple(item.address for item in self.storage_buffers(prefix="controller"))


def _require_source(
    state_sources: tuple[tuple[StateFieldSpec, torch.Tensor], ...],
    *,
    semantic_key: str,
    field_kind: StateFieldKind,
) -> tuple[StateFieldSpec, torch.Tensor]:
    matches = tuple(item for item in state_sources if item[0].semantic_key == semantic_key)
    if len(matches) != 1:
        raise BackendBatchContractError(
            f"mjwarp controller read {semantic_key!r} must bind exactly one state source"
        )
    spec, source = matches[0]
    if (
        spec.identity.entity_kind is not StateEntityKind.DOF
        or spec.identity.field_kind is not field_kind
    ):
        raise BackendBatchContractError(
            f"mjwarp controller read {semantic_key!r} has incompatible state semantics"
        )
    if source.device.type != "cuda" or source.dtype is not torch.float32 or source.ndim != 2:
        raise BackendBatchContractError(
            f"mjwarp controller read {semantic_key!r} requires a CUDA float32 matrix"
        )
    expected = (len(spec.identity.entity_ids),)
    if tuple(int(dim) for dim in source.shape[1:]) != expected:
        raise BackendBatchContractError(
            f"mjwarp controller read {semantic_key!r} source shape does not match its IDs"
        )
    return spec, source


def _expand_parameter(
    control: ControlSpec,
    semantic_key: str,
    *,
    width: int,
) -> np.ndarray:
    assert control.controller is not None
    values = np.asarray(control.controller.parameter(semantic_key).values, dtype=np.float64)
    if values.size == 1:
        values = np.full((width,), float(values[0]), dtype=np.float64)
    elif values.size != width:
        raise BackendBatchContractError(
            f"mjwarp controller parameter {semantic_key!r} requires length 1 or {width}"
        )
    values32 = values.astype(np.float32)
    if not np.isfinite(values32).all():
        raise BackendBatchContractError(
            f"mjwarp controller parameter {semantic_key!r} is not finite in float32"
        )
    return values32


def _bind_actuator_coordinates(
    backend: MjwarpBackend,
    *,
    position_spec: StateFieldSpec,
    velocity_spec: StateFieldSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = backend._cpu_model
    mujoco = backend._mujoco
    fixed_gain = int(mujoco.mjtGain.mjGAIN_FIXED)
    no_bias = int(mujoco.mjtBias.mjBIAS_NONE)
    no_dynamics = int(mujoco.mjtDyn.mjDYN_NONE)
    joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    position_columns = {
        int(coordinate): column
        for column, coordinate in enumerate(position_spec.identity.entity_ids)
    }
    velocity_columns = {
        int(coordinate): column
        for column, coordinate in enumerate(velocity_spec.identity.entity_ids)
    }
    if len(position_columns) != len(position_spec.identity.entity_ids) or len(
        velocity_columns
    ) != len(velocity_spec.identity.entity_ids):
        raise BackendBatchContractError("mjwarp controller state coordinate IDs must be unique")

    position_indices: list[int] = []
    velocity_indices: list[int] = []
    logical_coordinates: list[tuple[int, int]] = []
    direct_gain = np.zeros((int(model.actuator_gainprm.shape[1]),), dtype=np.float64)
    direct_gain[0] = 1.0
    direct_gear = np.zeros((int(model.actuator_gear.shape[1]),), dtype=np.float64)
    direct_gear[0] = 1.0
    named_joint_ids = set(backend._joint_ids.values())
    for actuator_id in range(backend._nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        valid_joint = 0 <= joint_id < int(model.njnt)
        if (
            not backend._actuator_names[actuator_id]
            or int(model.actuator_gaintype[actuator_id]) != fixed_gain
            or int(model.actuator_biastype[actuator_id]) != no_bias
            or int(model.actuator_dyntype[actuator_id]) != no_dynamics
            or int(model.actuator_trntype[actuator_id]) != joint_transmission
            or not valid_joint
            or joint_id not in named_joint_ids
            or int(model.actuator_trnid[actuator_id, 1]) != -1
            or (valid_joint and int(model.jnt_type[joint_id]) != hinge)
            or not np.array_equal(model.actuator_gainprm[actuator_id], direct_gain)
            or not np.array_equal(
                model.actuator_biasprm[actuator_id],
                np.zeros_like(model.actuator_biasprm[actuator_id]),
            )
            or not np.array_equal(model.actuator_gear[actuator_id], direct_gear)
            or not bool(model.actuator_ctrllimited[actuator_id])
        ):
            raise BackendBatchContractError(
                "mjwarp joint PD controller requires named single-DoF hinge direct motors "
                "with unit joint transmission, fixed unit gain, no bias/dynamics, and ctrl limits"
            )
        qpos_coordinate = int(model.jnt_qposadr[joint_id]) - backend._root_qpos_dim
        qvel_coordinate = int(model.jnt_dofadr[joint_id]) - backend._root_qvel_dim
        try:
            position_indices.append(position_columns[qpos_coordinate])
            velocity_indices.append(velocity_columns[qvel_coordinate])
        except KeyError as exc:
            raise BackendBatchContractError(
                "mjwarp controller state reads do not cover every actuator coordinate"
            ) from exc
        logical_coordinates.append((qpos_coordinate, qvel_coordinate))
    if len(set(logical_coordinates)) != backend._nu:
        raise BackendBatchContractError(
            "mjwarp controller requires a one-to-one actuator/DoF coordinate mapping"
        )
    ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    if (
        ctrl_range.shape != (backend._nu, 2)
        or not np.isfinite(ctrl_range).all()
        or np.any(ctrl_range[:, 0] >= 0.0)
        or np.any(ctrl_range[:, 1] <= 0.0)
    ):
        raise BackendBatchContractError(
            "mjwarp controller requires finite actuator ctrl ranges spanning zero"
        )
    return (
        np.asarray(position_indices, dtype=np.int64),
        np.asarray(velocity_indices, dtype=np.int64),
        ctrl_range,
    )


def bind_mjwarp_device_controller(
    backend: MjwarpBackend,
    control: ControlSpec,
    *,
    state_sources: tuple[tuple[StateFieldSpec, torch.Tensor], ...],
) -> MjwarpDeviceControllerRuntime | None:
    """Validate and allocate one production controller on a cold bind path."""

    if control.implementation is ControlImplementation.CONTROL_STEP_CONSTANT:
        return None
    if control.implementation is not ControlImplementation.DEVICE_SUBSTEP_CONTROLLER:
        raise BackendBatchContractError(
            f"mjwarp does not support control implementation {control.implementation.value!r}"
        )
    controller = control.controller
    assert controller is not None
    if controller.implementation_key != MJWARP_JOINT_PD_TORQUE_CONTROLLER:
        raise BackendBatchContractError(
            f"mjwarp does not support device controller {controller.implementation_key!r}"
        )
    read_keys = tuple(item.semantic_key for item in controller.state_reads)
    if read_keys != (_POSITION_READ, _VELOCITY_READ) or any(
        item.phase is not ControllerStateReadPhase.PRE_SUBSTEP for item in controller.state_reads
    ):
        raise BackendBatchContractError(
            "mjwarp joint PD controller requires canonical pre_substep reads "
            "('dof.position', 'dof.velocity')"
        )
    parameter_keys = tuple(item.semantic_key for item in controller.parameters)
    if parameter_keys != _PARAMETER_KEYS:
        raise BackendBatchContractError(
            f"mjwarp joint PD controller requires parameters {_PARAMETER_KEYS!r}"
        )

    position_spec, position_source = _require_source(
        state_sources,
        semantic_key=_POSITION_READ,
        field_kind=StateFieldKind.POSITION,
    )
    velocity_spec, velocity_source = _require_source(
        state_sources,
        semantic_key=_VELOCITY_READ,
        field_kind=StateFieldKind.ANGULAR_VELOCITY,
    )
    position_indices, velocity_indices, ctrl_range = _bind_actuator_coordinates(
        backend,
        position_spec=position_spec,
        velocity_spec=velocity_spec,
    )
    stiffness = _expand_parameter(control, "stiffness", width=backend._nu)
    damping = _expand_parameter(control, "damping", width=backend._nu)
    effort_limit = _expand_parameter(control, "effort_limit", width=backend._nu)
    if np.any(stiffness < 0.0) or np.any(damping < 0.0):
        raise BackendBatchContractError(
            "mjwarp controller stiffness and damping must be non-negative"
        )
    symmetric_model_limit = np.minimum(-ctrl_range[:, 0], ctrl_range[:, 1]).astype(np.float32)
    if np.any(effort_limit <= 0.0) or np.any(effort_limit > symmetric_model_limit):
        raise BackendBatchContractError(
            "mjwarp controller effort_limit must be positive and within actuator ctrl ranges"
        )

    bridge = backend._ensure_device_bridge()
    device = bridge.ctrl.device
    shape = (backend._num_envs, backend._nu)
    if position_source.device != device or velocity_source.device != device:
        raise BackendBatchContractError(
            "mjwarp controller state sources must share the native control CUDA device"
        )
    return MjwarpDeviceControllerRuntime(
        implementation_key=controller.implementation_key,
        command=torch.empty(shape, dtype=torch.float32, device=device),
        position_source=position_source,
        velocity_source=velocity_source,
        position_indices=torch.tensor(position_indices, dtype=torch.int64, device=device),
        velocity_indices=torch.tensor(velocity_indices, dtype=torch.int64, device=device),
        position=torch.empty(shape, dtype=torch.float32, device=device),
        velocity=torch.empty(shape, dtype=torch.float32, device=device),
        error=torch.empty(shape, dtype=torch.float32, device=device),
        damping_force=torch.empty(shape, dtype=torch.float32, device=device),
        stiffness=torch.tensor(stiffness, dtype=torch.float32, device=device),
        damping=torch.tensor(damping, dtype=torch.float32, device=device),
        effort_lower=torch.tensor(-effort_limit, dtype=torch.float32, device=device),
        effort_upper=torch.tensor(effort_limit, dtype=torch.float32, device=device),
        native_ctrl=bridge.ctrl,
    )


__all__ = [
    "MJWARP_JOINT_PD_TORQUE_CONTROLLER",
    "MjwarpDeviceControllerRuntime",
    "bind_mjwarp_device_controller",
]
