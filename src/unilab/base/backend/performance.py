"""Cold-path performance evidence contracts for managed device backends."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import DeviceGraphDiagnostics

BACKEND_PERFORMANCE_DIAGNOSTICS_VERSION = 1


class BackendPerformanceDiagnosticsError(ValueError):
    """Raised when backend-owned performance evidence is incomplete."""


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendPerformanceDiagnosticsError(f"{label} must be a non-empty string")
    return value


def _count(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendPerformanceDiagnosticsError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class BackendModelFieldDiagnostics:
    """One direct or derived Model allocation in a materialization receipt."""

    field_name: str
    role: str
    materialized_shape: tuple[int, ...]
    materialized_address: int
    model_bytes: int
    replaced: bool
    compiled_default_shape: tuple[int, ...]
    per_world_default_shape: tuple[int, ...] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_name", _name(self.field_name, "field_name"))
        if self.role not in {"direct", "derived"}:
            raise BackendPerformanceDiagnosticsError(
                "model field role must be 'direct' or 'derived'"
            )
        for label in ("materialized_shape", "compiled_default_shape"):
            shape = getattr(self, label)
            if (
                not isinstance(shape, tuple)
                or not shape
                or any(
                    isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
                )
            ):
                raise BackendPerformanceDiagnosticsError(f"{label} is invalid")
        if self.per_world_default_shape is not None and (
            not isinstance(self.per_world_default_shape, tuple)
            or not self.per_world_default_shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
                for dim in self.per_world_default_shape
            )
        ):
            raise BackendPerformanceDiagnosticsError("per_world_default_shape is invalid")
        object.__setattr__(
            self,
            "materialized_address",
            _count(self.materialized_address, "materialized_address"),
        )
        object.__setattr__(self, "model_bytes", _count(self.model_bytes, "model_bytes"))
        if not isinstance(self.replaced, bool):
            raise BackendPerformanceDiagnosticsError("replaced must be a bool")


@dataclass(frozen=True)
class BackendModelMaterializationDiagnostics:
    """Public, immutable projection of backend-owned Model materialization."""

    receipt_fingerprint: str
    num_worlds: int
    fields: tuple[BackendModelFieldDiagnostics, ...]
    expanded_model_bytes: int
    baseline_bytes: int
    storage_generation: int
    storage_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_fingerprint",
            _name(self.receipt_fingerprint, "receipt_fingerprint"),
        )
        if isinstance(self.num_worlds, bool) or not isinstance(self.num_worlds, int):
            raise BackendPerformanceDiagnosticsError("num_worlds must be a positive integer")
        if self.num_worlds <= 0:
            raise BackendPerformanceDiagnosticsError("num_worlds must be a positive integer")
        if (
            not isinstance(self.fields, tuple)
            or not self.fields
            or any(not isinstance(field, BackendModelFieldDiagnostics) for field in self.fields)
        ):
            raise BackendPerformanceDiagnosticsError(
                "materialization fields must be a non-empty typed tuple"
            )
        names = tuple(field.field_name for field in self.fields)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise BackendPerformanceDiagnosticsError(
                "materialization fields must be canonical and unique"
            )
        if any(field.materialized_shape[0] != self.num_worlds for field in self.fields):
            raise BackendPerformanceDiagnosticsError(
                "materialization field world dimensions differ from num_worlds"
            )
        object.__setattr__(
            self,
            "expanded_model_bytes",
            _count(self.expanded_model_bytes, "expanded_model_bytes"),
        )
        object.__setattr__(self, "baseline_bytes", _count(self.baseline_bytes, "baseline_bytes"))
        if self.expanded_model_bytes != sum(
            field.model_bytes for field in self.fields if field.replaced
        ):
            raise BackendPerformanceDiagnosticsError(
                "expanded_model_bytes differs from replaced field bytes"
            )
        object.__setattr__(
            self,
            "storage_generation",
            _count(self.storage_generation, "storage_generation"),
        )
        object.__setattr__(
            self,
            "storage_fingerprint",
            _name(self.storage_fingerprint, "storage_fingerprint"),
        )


@dataclass(frozen=True)
class BackendDeviceLifecycleDiagnostics:
    """Cumulative graph launches split by semantic lifecycle operation."""

    runtime_barriers: int
    step_graph_launches: int
    reset_graph_launches: int
    forward_graph_launches: int
    state_refreshes: int
    instrumentation_complete: bool = True

    def __post_init__(self) -> None:
        for name in (
            "runtime_barriers",
            "step_graph_launches",
            "reset_graph_launches",
            "forward_graph_launches",
            "state_refreshes",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if not isinstance(self.instrumentation_complete, bool):
            raise BackendPerformanceDiagnosticsError(
                "lifecycle instrumentation_complete must be a bool"
            )


@dataclass(frozen=True)
class BackendMutationPerformanceDiagnostics:
    """Plan-scoped Model mutation, storage, and lifecycle evidence."""

    backend_type: str
    backend_instance_id: str
    mutation_plan_fingerprint: str
    model_targets: tuple[str, ...]
    recompute_kind: str
    direct_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    recompute_capture_count: int
    recompute_launch_count: int
    materialization: BackendModelMaterializationDiagnostics | None
    lifecycle: BackendDeviceLifecycleDiagnostics
    graph: DeviceGraphDiagnostics
    instrumentation_complete: bool = True
    contract_version: int = BACKEND_PERFORMANCE_DIAGNOSTICS_VERSION

    def __post_init__(self) -> None:
        for name in ("backend_type", "backend_instance_id", "mutation_plan_fingerprint"):
            object.__setattr__(self, name, _name(getattr(self, name), name))
        for name in ("model_targets", "direct_fields", "derived_fields"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or values != tuple(sorted(values))
                or len(values) != len(set(values))
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise BackendPerformanceDiagnosticsError(f"{name} must be canonical and unique")
        object.__setattr__(self, "recompute_kind", _name(self.recompute_kind, "recompute_kind"))
        for name in ("recompute_capture_count", "recompute_launch_count"):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if not isinstance(self.lifecycle, BackendDeviceLifecycleDiagnostics):
            raise BackendPerformanceDiagnosticsError("lifecycle diagnostics are invalid")
        if not isinstance(self.graph, DeviceGraphDiagnostics):
            raise BackendPerformanceDiagnosticsError("graph diagnostics are invalid")
        has_model = bool(self.model_targets)
        if has_model != (self.materialization is not None):
            raise BackendPerformanceDiagnosticsError(
                "Model targets and materialization diagnostics must be present together"
            )
        if has_model:
            assert self.materialization is not None
            if not self.direct_fields:
                raise BackendPerformanceDiagnosticsError(
                    "Model performance diagnostics require direct fields"
                )
            materialized_direct = tuple(
                field.field_name for field in self.materialization.fields if field.role == "direct"
            )
            materialized_derived = tuple(
                field.field_name for field in self.materialization.fields if field.role == "derived"
            )
            if (
                materialized_direct != self.direct_fields
                or materialized_derived != self.derived_fields
            ):
                raise BackendPerformanceDiagnosticsError(
                    "materialization field roles differ from recompute diagnostics"
                )
            if (
                self.materialization.storage_generation != self.graph.storage_generation
                or self.materialization.storage_fingerprint != self.graph.storage_fingerprint
            ):
                raise BackendPerformanceDiagnosticsError(
                    "materialization and graph storage identities differ"
                )
        if not has_model and (
            self.recompute_kind != "none"
            or self.direct_fields
            or self.derived_fields
            or self.recompute_capture_count
            or self.recompute_launch_count
        ):
            raise BackendPerformanceDiagnosticsError(
                "state-only plans cannot report Model recompute evidence"
            )
        if self.recompute_kind == "none" and (
            self.derived_fields or self.recompute_capture_count or self.recompute_launch_count
        ):
            raise BackendPerformanceDiagnosticsError(
                "recompute kind 'none' cannot report a graph or derived fields"
            )
        if self.recompute_kind != "none" and self.recompute_capture_count == 0:
            raise BackendPerformanceDiagnosticsError(
                "non-trivial Model recompute requires a captured graph"
            )
        if self.graph.backend_type != self.backend_type:
            raise BackendPerformanceDiagnosticsError(
                "mutation and graph diagnostics report different backend types"
            )
        semantic_graph_launches = (
            self.lifecycle.step_graph_launches
            + self.lifecycle.reset_graph_launches
            + self.lifecycle.forward_graph_launches
        )
        if semantic_graph_launches != self.graph.launch_count:
            raise BackendPerformanceDiagnosticsError(
                "semantic lifecycle graph launches differ from graph diagnostics"
            )
        if self.lifecycle.runtime_barriers != (
            self.lifecycle.step_graph_launches + self.lifecycle.reset_graph_launches
        ):
            raise BackendPerformanceDiagnosticsError(
                "runtime barriers differ from step and reset graph launches"
            )
        if self.lifecycle.state_refreshes < (
            self.lifecycle.step_graph_launches + self.lifecycle.forward_graph_launches
        ):
            raise BackendPerformanceDiagnosticsError(
                "state refreshes do not cover every state-producing graph"
            )
        if not isinstance(self.instrumentation_complete, bool):
            raise BackendPerformanceDiagnosticsError(
                "mutation instrumentation_complete must be a bool"
            )
        if self.instrumentation_complete and (
            not self.lifecycle.instrumentation_complete or not self.graph.instrumentation_complete
        ):
            raise BackendPerformanceDiagnosticsError(
                "complete mutation diagnostics require complete nested instrumentation"
            )
        if self.contract_version != BACKEND_PERFORMANCE_DIAGNOSTICS_VERSION:
            raise BackendPerformanceDiagnosticsError(
                f"unsupported performance diagnostics version {self.contract_version}"
            )


__all__ = [
    "BACKEND_PERFORMANCE_DIAGNOSTICS_VERSION",
    "BackendDeviceLifecycleDiagnostics",
    "BackendModelFieldDiagnostics",
    "BackendModelMaterializationDiagnostics",
    "BackendMutationPerformanceDiagnostics",
    "BackendPerformanceDiagnosticsError",
]
