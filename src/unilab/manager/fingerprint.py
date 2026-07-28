"""Canonical serialization for managed task and policy fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from unilab.base.backend.batch import BufferContract, BufferPlacement
from unilab.base.backend.mutation import MutationSelectorSpec

from .plan import CompiledTaskPlan
from .spec import ParameterValue, TensorSpec


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
        "output_channels": [
            {"key": item.key, "buffer": _buffer_payload(item.buffer)}
            for item in plan.output_channels
        ],
        "policy_abi_fingerprint": plan.policy_abi.fingerprint,
        "executor_key": plan.executor_key,
        "required_capabilities": list(plan.required_capabilities),
    }
