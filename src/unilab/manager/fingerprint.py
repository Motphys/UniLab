"""Canonical serialization for managed task and policy fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from unilab.base.backend.batch import (
    BufferContract,
    BufferPlacement,
    PhysicalUnit,
    ReferenceFrame,
)
from unilab.base.backend.mutation import MutationSelectorSpec

from .entities import ManagerContractError
from .plan import MANAGER_TASK_CONTRACT_VERSION, CompiledTaskPlan
from .spec import ParameterValue, TensorSpec

MANAGED_POLICY_ABI_SNAPSHOT_VERSION = "managed-policy-abi-snapshot-v1"
"""Wire-schema version for a manager policy contract saved with a training run."""


class ManagedPolicyABISnapshotError(ValueError):
    """Raised when a serialized managed policy ABI is not a complete contract.

    The snapshot deliberately contains only semantic policy information.  In
    particular it must not be used as a transport for backend selector IDs or
    backend-local execution fingerprints: those differ legitimately between
    independent physics backends and are not policy I/O semantics.
    """


def canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _placement_payload(placement: BufferPlacement) -> dict[str, Any]:
    return {
        "memory_space": placement.memory_space.value,
        "device_type": placement.device_type,
        "device_index": placement.device_index,
    }


def _buffer_payload(buffer: BufferContract) -> dict[str, Any]:
    return {
        "row_shape": list(buffer.row_shape),
        "dtype": buffer.dtype,
        "layout": buffer.layout.value,
        "placement": _placement_payload(buffer.placement),
        "owner": buffer.owner.value,
        "mutability": buffer.mutability.value,
        "lifetime": buffer.lifetime.value,
        "dlpack_exportable": buffer.dlpack_exportable,
        "address_stable": buffer.address_stable,
    }


def tensor_payload(tensor: TensorSpec) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
        "frame": tensor.frame.value,
        "unit": tensor.unit.value,
        "quaternion_order": tensor.quaternion_order.value,
    }


def managed_policy_abi_snapshot(plan: CompiledTaskPlan) -> dict[str, Any]:
    """Render the backend-neutral policy contract of an immutable task plan.

    This is intentionally a *semantic* snapshot.  It preserves ordered
    observation layout and per-output tensor semantics, but excludes selector
    bindings and every backend execution identity.  Consequently a MuJoCo and
    an ``mjwarp`` instance may share it only when their compiled policy I/O is
    actually compatible.

    The returned object contains fresh JSON primitives, so retaining or
    modifying it cannot mutate the compiled plan.
    """

    if not isinstance(plan, CompiledTaskPlan):
        raise ManagedPolicyABISnapshotError("managed policy ABI requires a CompiledTaskPlan")
    policy = plan.policy_abi
    snapshot = {
        "schema_version": MANAGED_POLICY_ABI_SNAPSHOT_VERSION,
        "manager_contract_version": plan.contract_version,
        "task_key": plan.task_key,
        "plan_fingerprint": plan.fingerprint,
        "policy_abi_fingerprint": policy.fingerprint,
        "executor_key": plan.executor_key,
        "execution_profile": plan.backend_io.execution_profile.value,
        "observation_groups": [
            {
                "key": group.key,
                "width": group.width,
                "dtype": group.dtype,
                "outputs": [
                    {
                        "semantic_key": output.semantic_key,
                        "start": output.output.start,
                        "stop": output.output.stop,
                        "tensor": tensor_payload(output.output.tensor),
                    }
                    for output in group.outputs
                ],
            }
            for group in policy.observation_groups
        ],
        "action": {
            "key": policy.action_key,
            "dim": policy.action_dim,
            "dtype": policy.action_dtype,
            "scale": [float(value) for value in policy.action_scale],
        },
        "normalization": policy.normalization.value,
    }
    # Keep renderer and parser as a single wire contract.  This also makes a
    # future plan extension fail at the owner layer instead of silently being
    # dropped from a cross-backend policy check.
    return normalize_managed_policy_abi_snapshot(snapshot)


def _require_mapping(value: object, *, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManagedPolicyABISnapshotError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ManagedPolicyABISnapshotError(f"{name} keys must be strings")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ManagedPolicyABISnapshotError(
            f"{name} has an invalid key set; missing={missing}, extra={extra}"
        )
    return value


def _require_non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManagedPolicyABISnapshotError(f"{name} must be a non-empty string")
    return value.strip()


def _require_dtype(value: object, *, name: str) -> str:
    raw = _require_non_empty_string(value, name=name)
    try:
        return str(np.dtype(raw).name)
    except (TypeError, ValueError) as exc:
        raise ManagedPolicyABISnapshotError(f"{name} is not a valid NumPy dtype") from exc


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManagedPolicyABISnapshotError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManagedPolicyABISnapshotError(f"{name} must be a non-negative integer")
    return value


def _canonical_tensor_snapshot(value: object, *, name: str) -> dict[str, Any]:
    tensor = _require_mapping(
        value,
        name=name,
        keys=frozenset({"shape", "dtype", "frame", "unit", "quaternion_order"}),
    )
    raw_shape = tensor["shape"]
    if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
        raise ManagedPolicyABISnapshotError(f"{name}.shape must be a sequence")
    shape_values = [_require_positive_int(item, name=f"{name}.shape item") for item in raw_shape]
    if not shape_values:
        raise ManagedPolicyABISnapshotError(f"{name}.shape must not be empty")
    quaternion_order = _require_non_empty_string(
        tensor["quaternion_order"], name=f"{name}.quaternion_order"
    )
    if quaternion_order not in {"none", "wxyz", "xyzw"}:
        raise ManagedPolicyABISnapshotError(f"{name}.quaternion_order is unsupported")
    frame = _require_non_empty_string(tensor["frame"], name=f"{name}.frame")
    if frame not in {item.value for item in ReferenceFrame}:
        raise ManagedPolicyABISnapshotError(f"{name}.frame is unsupported")
    unit = _require_non_empty_string(tensor["unit"], name=f"{name}.unit")
    if unit not in {item.value for item in PhysicalUnit}:
        raise ManagedPolicyABISnapshotError(f"{name}.unit is unsupported")
    if (unit == PhysicalUnit.QUATERNION.value) != (quaternion_order != "none"):
        raise ManagedPolicyABISnapshotError(f"{name}.unit and quaternion_order must agree")
    if quaternion_order != "none" and shape_values[-1] != 4:
        raise ManagedPolicyABISnapshotError(
            f"{name}.quaternion_order requires a trailing dimension of four"
        )
    return {
        "shape": shape_values,
        "dtype": _require_dtype(tensor["dtype"], name=f"{name}.dtype"),
        "frame": frame,
        "unit": unit,
        "quaternion_order": quaternion_order,
    }


def _tensor_width(tensor: Mapping[str, object]) -> int:
    width = 1
    raw_shape = tensor["shape"]
    assert isinstance(raw_shape, list)
    for dimension in raw_shape:
        assert isinstance(dimension, int)
        width *= dimension
    return width


def _canonical_observation_groups(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManagedPolicyABISnapshotError("observation_groups must be a sequence")
    if not value:
        raise ManagedPolicyABISnapshotError("observation_groups must not be empty")
    groups: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for group_index, raw_group in enumerate(value):
        name = f"observation_groups[{group_index}]"
        group = _require_mapping(
            raw_group,
            name=name,
            keys=frozenset({"key", "width", "dtype", "outputs"}),
        )
        key = _require_non_empty_string(group["key"], name=f"{name}.key")
        if key in seen_groups:
            raise ManagedPolicyABISnapshotError("observation group keys must be unique")
        seen_groups.add(key)
        width = _require_positive_int(group["width"], name=f"{name}.width")
        dtype = _require_dtype(group["dtype"], name=f"{name}.dtype")
        raw_outputs = group["outputs"]
        if not isinstance(raw_outputs, Sequence) or isinstance(raw_outputs, (str, bytes)):
            raise ManagedPolicyABISnapshotError(f"{name}.outputs must be a sequence")
        if not raw_outputs:
            raise ManagedPolicyABISnapshotError(f"{name}.outputs must not be empty")
        outputs: list[dict[str, Any]] = []
        seen_outputs: set[str] = set()
        cursor = 0
        for output_index, raw_output in enumerate(raw_outputs):
            output_name = f"{name}.outputs[{output_index}]"
            output = _require_mapping(
                raw_output,
                name=output_name,
                keys=frozenset({"semantic_key", "start", "stop", "tensor"}),
            )
            semantic_key = _require_non_empty_string(
                output["semantic_key"], name=f"{output_name}.semantic_key"
            )
            if semantic_key in seen_outputs:
                raise ManagedPolicyABISnapshotError(f"{name} output semantic keys must be unique")
            seen_outputs.add(semantic_key)
            start = _require_non_negative_int(output["start"], name=f"{output_name}.start")
            stop = _require_positive_int(output["stop"], name=f"{output_name}.stop")
            if start != cursor or stop <= start:
                raise ManagedPolicyABISnapshotError(
                    f"{name} outputs must be contiguous and non-overlapping"
                )
            tensor = _canonical_tensor_snapshot(output["tensor"], name=f"{output_name}.tensor")
            if tensor["dtype"] != dtype:
                raise ManagedPolicyABISnapshotError(f"{name} output dtype differs from group dtype")
            if stop - start != _tensor_width(tensor):
                raise ManagedPolicyABISnapshotError(
                    f"{output_name} slice width differs from tensor shape"
                )
            cursor = stop
            outputs.append(
                {
                    "semantic_key": semantic_key,
                    "start": start,
                    "stop": stop,
                    "tensor": tensor,
                }
            )
        if cursor != width:
            raise ManagedPolicyABISnapshotError(
                f"{name} output slices do not cover the declared width"
            )
        groups.append({"key": key, "width": width, "dtype": dtype, "outputs": outputs})
    return groups


def _canonical_action_snapshot(value: object) -> dict[str, Any]:
    action = _require_mapping(
        value,
        name="action",
        keys=frozenset({"key", "dim", "dtype", "scale"}),
    )
    dim = _require_positive_int(action["dim"], name="action.dim")
    raw_scale = action["scale"]
    if not isinstance(raw_scale, Sequence) or isinstance(raw_scale, (str, bytes)):
        raise ManagedPolicyABISnapshotError("action.scale must be a sequence")
    if len(raw_scale) != dim:
        raise ManagedPolicyABISnapshotError("action.scale must have one value per action dim")
    scale: list[float] = []
    for value in raw_scale:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManagedPolicyABISnapshotError("action.scale values must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ManagedPolicyABISnapshotError("action.scale values must be finite numbers")
        scale.append(number)
    return {
        "key": _require_non_empty_string(action["key"], name="action.key"),
        "dim": dim,
        "dtype": _require_dtype(action["dtype"], name="action.dtype"),
        "scale": scale,
    }


def normalize_managed_policy_abi_snapshot(value: object) -> dict[str, Any]:
    """Validate and deep-copy a managed policy ABI wire snapshot.

    Resolver callers use this before comparison so malformed, partial, or
    backend-leaking extensions cannot be treated as a matching policy merely
    because their common subset happens to compare equal.
    """

    snapshot = _require_mapping(
        value,
        name="managed policy ABI",
        keys=frozenset(
            {
                "schema_version",
                "manager_contract_version",
                "task_key",
                "plan_fingerprint",
                "policy_abi_fingerprint",
                "executor_key",
                "execution_profile",
                "observation_groups",
                "action",
                "normalization",
            }
        ),
    )
    schema_version = _require_non_empty_string(
        snapshot["schema_version"], name="managed policy ABI.schema_version"
    )
    if schema_version != MANAGED_POLICY_ABI_SNAPSHOT_VERSION:
        raise ManagedPolicyABISnapshotError("managed policy ABI uses an unsupported schema version")
    manager_contract_version = _require_non_empty_string(
        snapshot["manager_contract_version"],
        name="managed policy ABI.manager_contract_version",
    )
    if manager_contract_version != MANAGER_TASK_CONTRACT_VERSION:
        raise ManagedPolicyABISnapshotError(
            "managed policy ABI uses an unsupported manager contract version"
        )
    execution_profile = _require_non_empty_string(
        snapshot["execution_profile"], name="managed policy ABI.execution_profile"
    )
    if execution_profile not in {"host_numpy", "device_resident"}:
        raise ManagedPolicyABISnapshotError("managed policy ABI has invalid execution_profile")
    normalization = _require_non_empty_string(
        snapshot["normalization"], name="managed policy ABI.normalization"
    )
    if normalization not in {"none", "empirical"}:
        raise ManagedPolicyABISnapshotError("managed policy ABI has invalid normalization")
    observation_groups = _canonical_observation_groups(snapshot["observation_groups"])
    action = _canonical_action_snapshot(snapshot["action"])
    policy_fingerprint = _require_non_empty_string(
        snapshot["policy_abi_fingerprint"],
        name="managed policy ABI.policy_abi_fingerprint",
    )
    policy_payload = {
        "groups": [
            {
                "key": group["key"],
                "width": group["width"],
                "dtype": group["dtype"],
                "outputs": [
                    {
                        "key": output["semantic_key"],
                        "start": output["start"],
                        "stop": output["stop"],
                        "tensor": output["tensor"],
                    }
                    for output in group["outputs"]
                ],
            }
            for group in observation_groups
        ],
        "action": action,
        "normalization": normalization,
    }
    expected_policy_fingerprint = f"managed-policy-abi-v1:{canonical_digest(policy_payload)}"
    if policy_fingerprint != expected_policy_fingerprint:
        raise ManagedPolicyABISnapshotError(
            "managed policy ABI policy_abi_fingerprint does not match its semantic fields"
        )
    return {
        "schema_version": schema_version,
        "manager_contract_version": manager_contract_version,
        "task_key": _require_non_empty_string(
            snapshot["task_key"], name="managed policy ABI.task_key"
        ),
        "plan_fingerprint": _require_non_empty_string(
            snapshot["plan_fingerprint"], name="managed policy ABI.plan_fingerprint"
        ),
        "policy_abi_fingerprint": policy_fingerprint,
        "executor_key": _require_non_empty_string(
            snapshot["executor_key"], name="managed policy ABI.executor_key"
        ),
        "execution_profile": execution_profile,
        "observation_groups": observation_groups,
        "action": action,
        "normalization": normalization,
    }


def _parameter_payload(value: ParameterValue) -> object:
    return list(value) if isinstance(value, tuple) else value


def _mutation_selector_payload(
    selector: MutationSelectorSpec | None,
    *,
    include_bindings: bool,
) -> dict[str, Any] | None:
    """Serialize selector semantics separately from compiler-local IDs."""

    if selector is None:
        return None
    return {
        "semantic_key": selector.semantic_key,
        "mode": selector.mode.value,
        "expressions": list(selector.expressions),
        **({"entity_ids": list(selector.entity_ids)} if include_bindings else {}),
    }


def compiled_plan_payload(
    plan: CompiledTaskPlan,
    *,
    include_bindings: bool,
) -> dict[str, Any]:
    """Serialize plan semantics, optionally including backend-local selector IDs."""
    return {
        "contract_version": plan.contract_version,
        "task_key": plan.task_key,
        "selectors": [
            {
                "key": item.key,
                "entity": item.entity,
                "kind": item.kind.value,
                "mode": item.mode.value,
                "expressions": list(item.expressions),
                **({"entity_ids": list(item.entity_ids)} if include_bindings else {}),
            }
            for item in plan.selectors
        ],
        "terms": [
            {
                "key": item.key,
                "definition_key": item.definition_key,
                "definition_version": item.definition_version,
                "phase": item.phase.value,
                "role": item.role.value,
                "dependencies": list(item.dependency_indices),
                "state_fields": list(item.state_field_indices),
                "mutations": list(item.mutation_indices),
                "parameters": [[key, _parameter_payload(value)] for key, value in item.parameters],
                "output": None
                if item.output is None
                else {
                    "channel": item.output.channel,
                    "start": item.output.start,
                    "stop": item.output.stop,
                    "tensor": tensor_payload(item.output.tensor),
                },
            }
            for item in plan.terms
        ],
        "backend_io": {
            "execution_profile": plan.backend_io.execution_profile.value,
            "state_fields": [
                {
                    "semantic_key": item.semantic_key,
                    "entity_kind": item.identity.entity_kind.value,
                    "field_kind": item.identity.field_kind.value,
                    **({"entity_ids": list(item.identity.entity_ids)} if include_bindings else {}),
                    "frame": item.frame.value,
                    "unit": item.unit.value,
                    "buffer": _buffer_payload(item.buffer),
                }
                for item in plan.backend_io.state_fields
            ],
            "control": {
                "semantic_key": plan.backend_io.control.semantic_key,
                "buffer": _buffer_payload(plan.backend_io.control.buffer),
                "physics_substeps_per_control": (
                    plan.backend_io.control.physics_substeps_per_control
                ),
                "implementation": plan.backend_io.control.implementation.value,
                "controller": None
                if plan.backend_io.control.controller is None
                else {
                    "contract_version": plan.backend_io.control.controller.contract_version,
                    "implementation_key": (plan.backend_io.control.controller.implementation_key),
                    "state_reads": [
                        {
                            "semantic_key": item.semantic_key,
                            "phase": item.phase.value,
                        }
                        for item in plan.backend_io.control.controller.state_reads
                    ],
                    "parameters": [
                        {
                            "semantic_key": item.semantic_key,
                            "values": list(item.values),
                        }
                        for item in plan.backend_io.control.controller.parameters
                    ],
                },
            },
            "hot_path_budget": None
            if plan.backend_io.hot_path_budget is None
            else list(plan.backend_io.hot_path_budget.items()),
            "reset_hot_path_budget": None
            if plan.backend_io.reset_hot_path_budget is None
            else list(plan.backend_io.reset_hot_path_budget.items()),
        },
        "mutations": [
            {
                "term_key": item.term_key,
                "target_key": item.target.target_key,
                "target_kind": item.target.target_kind.value,
                "entity_kind": item.target.entity_kind.value,
                "field_kind": item.target.field_kind.value,
                "selector": _mutation_selector_payload(
                    item.target.selector_spec,
                    include_bindings=include_bindings,
                ),
                "trigger": item.trigger.value,
                "commit_phase": item.commit_phase.value,
                "operation": item.operation.value,
                "baseline": item.baseline.value,
                "persistence": item.persistence.value,
                "recompute": int(item.recompute),
                "value_template": _buffer_payload(item.value_template),
            }
            for item in plan.mutation_specs
        ],
        **(
            {
                "mutation_events": [
                    {
                        "mutation_index": item.mutation_index,
                        "term_index": item.term_index,
                        "term_key": item.term_key,
                        "term_version": item.term_version,
                        "trigger": item.trigger.value,
                        "commit_phase": item.commit_phase.value,
                        "distribution": item.distribution.value,
                        "parameters": list(item.parameters),
                        "correlation": item.correlation.value,
                        "algorithm": item.algorithm,
                    }
                    for item in plan.mutation_events
                ]
            }
            if plan.mutation_events
            else {}
        ),
        "output_channels": [
            {"key": item.key, "buffer": _buffer_payload(item.buffer)}
            for item in plan.output_channels
        ],
        "policy_abi_fingerprint": plan.policy_abi.fingerprint,
        "executor_key": plan.executor_key,
        "required_capabilities": list(plan.required_capabilities),
    }


def validate_compiled_plan_fingerprints(plan: CompiledTaskPlan) -> None:
    """Fail closed when a frozen plan was forged after compilation."""

    if not isinstance(plan, CompiledTaskPlan):
        raise ManagerContractError("compiled plan integrity requires a CompiledTaskPlan")
    semantic = (
        f"{plan.contract_version}:"
        f"{canonical_digest(compiled_plan_payload(plan, include_bindings=False))}"
    )
    binding = (
        "manager-selector-binding-v1:"
        f"{canonical_digest(compiled_plan_payload(plan, include_bindings=True))}"
    )
    if plan.fingerprint != semantic or plan.selector_binding_fingerprint != binding:
        raise ManagerContractError(
            "compiled task plan fingerprints do not match its immutable payload"
        )
