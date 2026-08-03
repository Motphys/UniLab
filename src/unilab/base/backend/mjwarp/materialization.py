"""Cold-path scene materialization owned by the independent ``mjwarp`` backend."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, cast

from unilab.base.scene import SceneCfg

from ..batch import BackendBatchContractError
from ..mutation import MutationGraphInvalidation

MJWARP_MODEL_MATERIALIZATION_VERSION = "mjwarp-model-materialization-v1"
MJWARP_MODEL_INVALIDATIONS = (
    MutationGraphInvalidation.MODEL_BRIDGE_CACHE,
    MutationGraphInvalidation.SENSOR_CONTEXT,
    MutationGraphInvalidation.STEP_GRAPH,
    MutationGraphInvalidation.FORWARD_GRAPH,
    MutationGraphInvalidation.RESET_GRAPH,
    MutationGraphInvalidation.SENSE_GRAPH,
)


class MjwarpModelMaterializationContractError(ValueError):
    """Raised when a cold-path model materialization contract is malformed."""


class MjwarpModelFieldRole(str, Enum):
    DIRECT = "direct"
    DERIVED = "derived"


class MjwarpModelInvalidationOutcome(str, Enum):
    REBUILT = "rebuilt"
    UNCHANGED = "unchanged"
    NOT_PRESENT = "not_present"


def _canonical_fields(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise MjwarpModelMaterializationContractError(f"{name} must be a tuple")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise MjwarpModelMaterializationContractError(f"{name} must contain non-empty field names")
    normalized = tuple(value.strip() for value in values)
    if normalized != tuple(sorted(normalized)) or len(set(normalized)) != len(normalized):
        raise MjwarpModelMaterializationContractError(f"{name} must be canonical and unique")
    return normalized


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MjwarpModelMaterializationRequest:
    """Immutable one-shot request for per-world MJWarp Model storage."""

    num_worlds: int
    direct_fields: tuple[str, ...]
    derived_fields: tuple[str, ...] = ()
    per_world_default_fields: tuple[str, ...] = ()
    contract_version: str = MJWARP_MODEL_MATERIALIZATION_VERSION
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_worlds, bool)
            or not isinstance(self.num_worlds, int)
            or self.num_worlds <= 0
        ):
            raise MjwarpModelMaterializationContractError("num_worlds must be a positive integer")
        if self.contract_version != MJWARP_MODEL_MATERIALIZATION_VERSION:
            raise MjwarpModelMaterializationContractError(
                "unsupported mjwarp model materialization contract version"
            )
        direct = _canonical_fields(self.direct_fields, name="direct_fields")
        derived = _canonical_fields(self.derived_fields, name="derived_fields")
        per_world = _canonical_fields(
            self.per_world_default_fields,
            name="per_world_default_fields",
        )
        if not direct:
            raise MjwarpModelMaterializationContractError(
                "model materialization requires at least one direct field"
            )
        overlap = set(direct).intersection(derived)
        if overlap:
            raise MjwarpModelMaterializationContractError(
                "direct_fields and derived_fields overlap: " + ", ".join(sorted(overlap))
            )
        unknown_defaults = set(per_world).difference(direct)
        if unknown_defaults:
            raise MjwarpModelMaterializationContractError(
                "per_world_default_fields must be direct fields: "
                + ", ".join(sorted(unknown_defaults))
            )
        object.__setattr__(self, "direct_fields", direct)
        object.__setattr__(self, "derived_fields", derived)
        object.__setattr__(self, "per_world_default_fields", per_world)
        payload = {
            "contract_version": self.contract_version,
            "num_worlds": self.num_worlds,
            "direct_fields": direct,
            "derived_fields": derived,
            "per_world_default_fields": per_world,
        }
        object.__setattr__(
            self,
            "fingerprint",
            f"{MJWARP_MODEL_MATERIALIZATION_VERSION}:{_digest(payload)}",
        )

    @property
    def all_fields(self) -> tuple[str, ...]:
        return tuple(sorted((*self.direct_fields, *self.derived_fields)))

    def verify_fingerprint(self) -> None:
        """Reject a request whose immutable identity was altered after construction."""

        canonical = MjwarpModelMaterializationRequest(
            num_worlds=self.num_worlds,
            direct_fields=self.direct_fields,
            derived_fields=self.derived_fields,
            per_world_default_fields=self.per_world_default_fields,
            contract_version=self.contract_version,
        )
        if self != canonical:
            raise MjwarpModelMaterializationContractError(
                "model materialization request fingerprint does not match its payload"
            )


@dataclass(frozen=True)
class MjwarpModelFieldReceipt:
    """Address, shape, allocation, and default evidence for one Model field."""

    field_name: str
    role: MjwarpModelFieldRole
    source_shape: tuple[int, ...]
    materialized_shape: tuple[int, ...]
    source_address: int
    materialized_address: int
    replaced: bool
    model_bytes: int
    compiled_default_shape: tuple[int, ...]
    compiled_default_fingerprint: str
    per_world_default_shape: tuple[int, ...] | None = None
    per_world_default_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise MjwarpModelMaterializationContractError(
                "field receipt requires a non-empty field_name"
            )
        if not isinstance(self.role, MjwarpModelFieldRole):
            raise MjwarpModelMaterializationContractError("field receipt role is invalid")
        for name in ("source_shape", "materialized_shape", "compiled_default_shape"):
            shape = getattr(self, name)
            if not isinstance(shape, tuple) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
            ):
                raise MjwarpModelMaterializationContractError(f"{name} is invalid")
        for name in ("source_address", "materialized_address", "model_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MjwarpModelMaterializationContractError(f"{name} is invalid")
        if not isinstance(self.replaced, bool):
            raise MjwarpModelMaterializationContractError("replaced must be a bool")
        if not self.compiled_default_fingerprint:
            raise MjwarpModelMaterializationContractError(
                "compiled_default_fingerprint must be non-empty"
            )
        if (self.per_world_default_shape is None) != (self.per_world_default_fingerprint is None):
            raise MjwarpModelMaterializationContractError(
                "per-world default shape and fingerprint must be declared together"
            )


@dataclass(frozen=True)
class MjwarpModelInvalidationReceipt:
    consumer: MutationGraphInvalidation
    outcome: MjwarpModelInvalidationOutcome
    affected_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.consumer, MutationGraphInvalidation):
            raise MjwarpModelMaterializationContractError("invalidation consumer is invalid")
        if not isinstance(self.outcome, MjwarpModelInvalidationOutcome):
            raise MjwarpModelMaterializationContractError("invalidation outcome is invalid")
        if (
            isinstance(self.affected_count, bool)
            or not isinstance(self.affected_count, int)
            or self.affected_count < 0
        ):
            raise MjwarpModelMaterializationContractError("invalidation affected_count is invalid")
        changed = self.outcome is MjwarpModelInvalidationOutcome.REBUILT
        if changed != (self.affected_count > 0):
            raise MjwarpModelMaterializationContractError(
                "changed invalidations require a positive affected_count"
            )


@dataclass(frozen=True)
class MjwarpModelMaterializationReceipt:
    """Atomic commit evidence for one backend-owned materialization transaction."""

    request_fingerprint: str
    backend_instance_id: str
    num_worlds: int
    fields: tuple[MjwarpModelFieldReceipt, ...]
    invalidations: tuple[MjwarpModelInvalidationReceipt, ...]
    storage_generation_before: int
    storage_generation_after: int
    storage_fingerprint_before: str
    storage_fingerprint_after: str
    graph_plan_fingerprints_before: tuple[str, ...]
    graph_plan_fingerprints_after: tuple[str, ...]
    model_bridge_generation_before: int
    model_bridge_generation_after: int
    sensor_generation_before: int
    sensor_generation_after: int
    expanded_model_bytes: int
    baseline_bytes: int
    contract_version: str = MJWARP_MODEL_MATERIALIZATION_VERSION
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_fingerprint, str)
            or not self.request_fingerprint.strip()
            or not isinstance(self.backend_instance_id, str)
            or not self.backend_instance_id.strip()
        ):
            raise MjwarpModelMaterializationContractError(
                "materialization receipt identities must be non-empty"
            )
        if self.contract_version != MJWARP_MODEL_MATERIALIZATION_VERSION:
            raise MjwarpModelMaterializationContractError(
                "materialization receipt contract version is invalid"
            )
        if (
            isinstance(self.num_worlds, bool)
            or not isinstance(self.num_worlds, int)
            or self.num_worlds <= 0
        ):
            raise MjwarpModelMaterializationContractError("receipt num_worlds is invalid")
        if not isinstance(self.fields, tuple) or not self.fields:
            raise MjwarpModelMaterializationContractError("receipt fields must be non-empty")
        if any(not isinstance(item, MjwarpModelFieldReceipt) for item in self.fields):
            raise MjwarpModelMaterializationContractError("receipt contains an invalid field")
        field_names = tuple(item.field_name for item in self.fields)
        if field_names != tuple(sorted(field_names)) or len(set(field_names)) != len(field_names):
            raise MjwarpModelMaterializationContractError(
                "receipt fields must be canonical and unique"
            )
        if not isinstance(self.invalidations, tuple) or any(
            not isinstance(item, MjwarpModelInvalidationReceipt) for item in self.invalidations
        ):
            raise MjwarpModelMaterializationContractError(
                "receipt contains an invalid invalidation"
            )
        consumers = tuple(item.consumer for item in self.invalidations)
        if consumers != MJWARP_MODEL_INVALIDATIONS:
            raise MjwarpModelMaterializationContractError(
                "receipt must account for every model-storage consumer"
            )
        for name in (
            "storage_generation_before",
            "storage_generation_after",
            "model_bridge_generation_before",
            "model_bridge_generation_after",
            "sensor_generation_before",
            "sensor_generation_after",
            "expanded_model_bytes",
            "baseline_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MjwarpModelMaterializationContractError(f"receipt {name} is invalid")
        for name in ("graph_plan_fingerprints_before", "graph_plan_fingerprints_after"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise MjwarpModelMaterializationContractError(
                    f"receipt {name} must contain non-empty strings"
                )
            if values != tuple(sorted(values)) or len(set(values)) != len(values):
                raise MjwarpModelMaterializationContractError(
                    f"receipt {name} must be canonical and unique"
                )
        if any(
            not isinstance(value, str) or not value
            for value in (self.storage_fingerprint_before, self.storage_fingerprint_after)
        ):
            raise MjwarpModelMaterializationContractError(
                "receipt storage fingerprints must be non-empty strings"
            )

        replaced = any(item.replaced for item in self.fields)
        generation_delta = int(replaced)
        for before_name, after_name in (
            ("storage_generation_before", "storage_generation_after"),
            ("model_bridge_generation_before", "model_bridge_generation_after"),
            ("sensor_generation_before", "sensor_generation_after"),
        ):
            if getattr(self, after_name) != getattr(self, before_name) + generation_delta:
                raise MjwarpModelMaterializationContractError(
                    f"receipt {after_name} does not match model replacement state"
                )
        if (self.storage_fingerprint_before != self.storage_fingerprint_after) is not replaced:
            raise MjwarpModelMaterializationContractError(
                "receipt storage fingerprint transition does not match model replacement state"
            )
        if self.graph_plan_fingerprints_before != self.graph_plan_fingerprints_after:
            raise MjwarpModelMaterializationContractError(
                "materialization cannot change the active graph plan set"
            )
        if self.expanded_model_bytes != sum(
            item.model_bytes for item in self.fields if item.replaced
        ):
            raise MjwarpModelMaterializationContractError(
                "expanded_model_bytes does not match replaced field storage"
            )
        for item in self.fields:
            if not item.source_shape or not item.materialized_shape:
                raise MjwarpModelMaterializationContractError(
                    "materialized field shapes require a world dimension"
                )
            if item.materialized_shape[0] != self.num_worlds:
                raise MjwarpModelMaterializationContractError(
                    "materialized field world dimension does not match the receipt"
                )
            if item.replaced:
                if item.source_shape[0] != 1:
                    raise MjwarpModelMaterializationContractError(
                        "replaced model fields must originate from singleton storage"
                    )
            elif item.source_shape != item.materialized_shape:
                raise MjwarpModelMaterializationContractError(
                    "unreplaced model field shape changed"
                )
            if item.per_world_default_shape is not None and (
                item.role is not MjwarpModelFieldRole.DIRECT
                or not item.per_world_default_shape
                or item.per_world_default_shape[0] != self.num_worlds
            ):
                raise MjwarpModelMaterializationContractError(
                    "per-world defaults require a direct field with matching worlds"
                )

        invalidations = {item.consumer: item for item in self.invalidations}
        graph_count = len(self.graph_plan_fingerprints_before)
        expected_graph_outcome = (
            MjwarpModelInvalidationOutcome.REBUILT
            if replaced and graph_count
            else (
                MjwarpModelInvalidationOutcome.UNCHANGED
                if graph_count
                else MjwarpModelInvalidationOutcome.NOT_PRESENT
            )
        )
        expected_graph_count = graph_count if replaced else 0
        for consumer in (
            MutationGraphInvalidation.STEP_GRAPH,
            MutationGraphInvalidation.FORWARD_GRAPH,
            MutationGraphInvalidation.RESET_GRAPH,
        ):
            invalidation = invalidations[consumer]
            if (
                invalidation.outcome is not expected_graph_outcome
                or invalidation.affected_count != expected_graph_count
            ):
                raise MjwarpModelMaterializationContractError(
                    f"receipt {consumer.value} invalidation does not match active graph plans"
                )
        if not replaced and any(
            item.outcome is MjwarpModelInvalidationOutcome.REBUILT for item in self.invalidations
        ):
            raise MjwarpModelMaterializationContractError(
                "unchanged storage cannot report a rebuilt consumer"
            )
        payload = {
            "contract_version": self.contract_version,
            "request_fingerprint": self.request_fingerprint,
            "backend_instance_id": self.backend_instance_id,
            "num_worlds": self.num_worlds,
            "fields": [
                {
                    "field_name": item.field_name,
                    "role": item.role.value,
                    "source_shape": item.source_shape,
                    "materialized_shape": item.materialized_shape,
                    "source_address": item.source_address,
                    "materialized_address": item.materialized_address,
                    "replaced": item.replaced,
                    "model_bytes": item.model_bytes,
                    "compiled_default_shape": item.compiled_default_shape,
                    "compiled_default_fingerprint": item.compiled_default_fingerprint,
                    "per_world_default_shape": item.per_world_default_shape,
                    "per_world_default_fingerprint": item.per_world_default_fingerprint,
                }
                for item in self.fields
            ],
            "invalidations": [
                {
                    "consumer": item.consumer.value,
                    "outcome": item.outcome.value,
                    "affected_count": item.affected_count,
                }
                for item in self.invalidations
            ],
            "storage_generation_before": self.storage_generation_before,
            "storage_generation_after": self.storage_generation_after,
            "storage_fingerprint_before": self.storage_fingerprint_before,
            "storage_fingerprint_after": self.storage_fingerprint_after,
            "graph_plan_fingerprints_before": self.graph_plan_fingerprints_before,
            "graph_plan_fingerprints_after": self.graph_plan_fingerprints_after,
            "model_bridge_generation_before": self.model_bridge_generation_before,
            "model_bridge_generation_after": self.model_bridge_generation_after,
            "sensor_generation_before": self.sensor_generation_before,
            "sensor_generation_after": self.sensor_generation_after,
            "expanded_model_bytes": self.expanded_model_bytes,
            "baseline_bytes": self.baseline_bytes,
        }
        object.__setattr__(
            self,
            "fingerprint",
            f"{MJWARP_MODEL_MATERIALIZATION_VERSION}-receipt:{_digest(payload)}",
        )

    def verify_fingerprint(self) -> None:
        """Reject backend-owned evidence altered after atomic publication."""

        canonical = MjwarpModelMaterializationReceipt(
            request_fingerprint=self.request_fingerprint,
            backend_instance_id=self.backend_instance_id,
            num_worlds=self.num_worlds,
            fields=self.fields,
            invalidations=self.invalidations,
            storage_generation_before=self.storage_generation_before,
            storage_generation_after=self.storage_generation_after,
            storage_fingerprint_before=self.storage_fingerprint_before,
            storage_fingerprint_after=self.storage_fingerprint_after,
            graph_plan_fingerprints_before=self.graph_plan_fingerprints_before,
            graph_plan_fingerprints_after=self.graph_plan_fingerprints_after,
            model_bridge_generation_before=self.model_bridge_generation_before,
            model_bridge_generation_after=self.model_bridge_generation_after,
            sensor_generation_before=self.sensor_generation_before,
            sensor_generation_after=self.sensor_generation_after,
            expanded_model_bytes=self.expanded_model_bytes,
            baseline_bytes=self.baseline_bytes,
            contract_version=self.contract_version,
        )
        if self != canonical:
            raise MjwarpModelMaterializationContractError(
                "model materialization receipt fingerprint does not match its payload"
            )


class MjwarpModelMaterializationOwner(Protocol):
    """Minimum backend identity retained by the transaction coordinator."""

    _batch_instance_id: str
    _num_envs: int


@dataclass(frozen=True)
class MjwarpModelMaterializationCoordinator:
    """Own atomic Model storage publication, invalidation, and rollback."""

    owner: MjwarpModelMaterializationOwner

    def materialize(
        self,
        request: MjwarpModelMaterializationRequest,
    ) -> MjwarpModelMaterializationReceipt:
        owner = cast(Any, self.owner)
        if not isinstance(request, MjwarpModelMaterializationRequest):
            raise BackendBatchContractError(
                "mjwarp model materialization requires an immutable request"
            )
        try:
            request.verify_fingerprint()
        except MjwarpModelMaterializationContractError as exc:
            raise BackendBatchContractError(
                "mjwarp model materialization request identity is corrupt"
            ) from exc
        if request.num_worlds != owner._num_envs:
            raise BackendBatchContractError(
                "mjwarp model materialization world count does not match the backend"
            )
        if owner._model_materialization_poisoned:
            raise BackendBatchContractError(
                "mjwarp model materialization owner is permanently poisoned"
            )
        if owner._model_materialization_in_progress:
            raise BackendBatchContractError("mjwarp model materialization is not reentrant")

        existing = owner._model_materialization_receipt
        if existing is not None:
            owner._verify_model_materialization_receipt(existing)
            required_roles = {
                **{field_name: MjwarpModelFieldRole.DIRECT for field_name in request.direct_fields},
                **{
                    field_name: MjwarpModelFieldRole.DERIVED
                    for field_name in request.derived_fields
                },
            }
            existing_fields = {field.field_name: field for field in existing.fields}
            missing = tuple(sorted(set(required_roles).difference(existing_fields)))
            mismatched = tuple(
                sorted(
                    field_name
                    for field_name, role in required_roles.items()
                    if field_name in existing_fields
                    and existing_fields[field_name].role is not role
                )
            )
            missing_defaults = tuple(
                sorted(
                    field_name
                    for field_name in request.per_world_default_fields
                    if field_name in existing_fields
                    and existing_fields[field_name].per_world_default_shape is None
                )
            )
            if missing or mismatched or missing_defaults:
                raise BackendBatchContractError(
                    "mjwarp model fields were already frozen without the requested subset: "
                    f"missing={missing!r}, mismatched={mismatched!r}, "
                    f"missing_defaults={missing_defaults!r}"
                )
            return cast(MjwarpModelMaterializationReceipt, existing)
        if owner._runtime_barrier_count:
            raise BackendBatchContractError(
                "mjwarp model fields must materialize before the first runtime physics barrier"
            )

        owner._verify_device_graph_storage()
        owner._model_materialization_in_progress = True
        staged: tuple[Any, ...] = ()
        graph_before = owner._snapshot_device_graph_state()
        bridge_cache_before = dict(owner._model_bridge_cache)
        bridge_generation_before = owner._model_bridge_generation
        sensor_before = owner._model_sensor_context
        sensor_generation_before = owner._model_sensor_generation
        baselines_before = dict(owner._model_default_baselines)
        expanded_before = owner._expanded_model_fields
        try:
            staged = tuple(
                owner._stage_model_field(request, field_name) for field_name in request.all_fields
            )
            # Staging uses Warp's current stream; graph capture uses the
            # dedicated physics stream. Publication begins only after both
            # Model and baseline copies are globally visible.
            owner._warp.synchronize_device()
            storage_replaced = any(item.receipt.replaced for item in staged)
            prepared_bridge_cache = bridge_cache_before
            prepared_sensor = sensor_before
            if storage_replaced:
                prepared_bridge_cache = owner._prepare_model_bridge_cache(
                    staged,
                    bridge_cache_before,
                )
                prepared_sensor = owner._prepare_model_sensor_context(
                    frozenset((*owner._expanded_model_fields, *request.all_fields))
                )
                owner._warp.synchronize_device()
            graph_keys_before = tuple(sorted(graph_before.bundles))

            try:
                for item in staged:
                    if item.receipt.replaced:
                        owner._set_model_field_storage(
                            owner._device_model,
                            item.field_name,
                            item.materialized,
                        )
                owner._model_default_baselines.update(
                    {item.field_name: item.baseline for item in staged if item.baseline is not None}
                )
                owner._expanded_model_fields = frozenset(
                    (*owner._expanded_model_fields, *request.all_fields)
                )
                if storage_replaced:
                    owner._model_bridge_cache = prepared_bridge_cache
                    owner._model_bridge_generation += 1
                    owner._model_sensor_context = prepared_sensor
                    owner._model_sensor_generation += 1
                    owner._recapture_device_graphs_after_storage_change()

                graph_after = owner._snapshot_device_graph_state()
                graph_keys_after = tuple(sorted(graph_after.bundles))
                if graph_keys_after != graph_keys_before:
                    raise BackendBatchContractError(
                        "mjwarp model materialization changed the active graph plan set"
                    )
                receipt = MjwarpModelMaterializationReceipt(
                    request_fingerprint=request.fingerprint,
                    backend_instance_id=owner._batch_instance_id,
                    num_worlds=owner._num_envs,
                    fields=tuple(item.receipt for item in staged),
                    invalidations=owner._model_invalidation_receipts(
                        storage_replaced=storage_replaced,
                        bridge_entries=len(bridge_cache_before),
                        sensor_present=prepared_sensor is not None,
                        graph_keys=len(graph_keys_before),
                    ),
                    storage_generation_before=graph_before.storage_generation,
                    storage_generation_after=graph_after.storage_generation,
                    storage_fingerprint_before=graph_before.storage_fingerprint,
                    storage_fingerprint_after=graph_after.storage_fingerprint,
                    graph_plan_fingerprints_before=graph_keys_before,
                    graph_plan_fingerprints_after=graph_keys_after,
                    model_bridge_generation_before=bridge_generation_before,
                    model_bridge_generation_after=owner._model_bridge_generation,
                    sensor_generation_before=sensor_generation_before,
                    sensor_generation_after=owner._model_sensor_generation,
                    expanded_model_bytes=sum(
                        item.receipt.model_bytes for item in staged if item.receipt.replaced
                    ),
                    baseline_bytes=sum(item.baseline_bytes for item in staged),
                )
                owner._model_materialization_receipt = receipt
                return receipt
            except Exception as exc:
                rollback_error: Exception | None = None
                try:
                    for item in staged:
                        owner._set_model_field_storage(
                            owner._device_model,
                            item.field_name,
                            item.original,
                        )
                    owner._model_bridge_cache = bridge_cache_before
                    owner._model_bridge_generation = bridge_generation_before
                    owner._model_sensor_context = sensor_before
                    owner._model_sensor_generation = sensor_generation_before
                    owner._model_default_baselines = baselines_before
                    owner._expanded_model_fields = expanded_before
                    owner._restore_device_graph_state(graph_before)
                    actual = owner._snapshot_device_graph_storage()
                    if actual != graph_before.storage_buffers:
                        raise BackendBatchContractError(
                            "mjwarp model materialization rollback left stale storage"
                        )
                except Exception as rollback_exc:  # pragma: no cover - hard fault path.
                    rollback_error = rollback_exc
                if rollback_error is not None:
                    owner._model_materialization_poisoned = True
                    owner._device_graph_storage_poisoned = True
                    owner._device_graph_bundles.clear()
                    raise BackendBatchContractError(
                        "mjwarp model materialization failed and rollback could not restore "
                        "graph storage; backend is permanently poisoned"
                    ) from rollback_error
                raise BackendBatchContractError(
                    "mjwarp model materialization transaction rolled back"
                ) from exc
        except BackendBatchContractError:
            raise
        except Exception as exc:
            raise BackendBatchContractError(
                "mjwarp model materialization preparation failed before publication"
            ) from exc
        finally:
            owner._model_materialization_in_progress = False


class _TemporarySceneCleanup:
    """Own the one temporary XML created while merging scene fragments."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class MjwarpSceneContext:
    """Cold-path scene source and cleanup ownership for one backend instance."""

    source_model_file: str
    diagnostic_model_file: str
    cleanup_handle: Any | None = None


def materialize_mjwarp_scene(scene: SceneCfg) -> MjwarpSceneContext:
    """Resolve a flat/fragments scene before CUDA model upload.

    Height-field terrain construction is intentionally rejected in the first
    correctness profile.  The rejection happens before model upload so an
    unsupported owner cannot silently fall back to a different terrain path.
    """
    if scene is None or not scene.model_file:
        raise ValueError("MjwarpBackend requires SceneCfg.model_file")
    if scene.terrain is not None:
        raise NotImplementedError(
            "mjwarp host_numpy profile does not support generated terrain or height-field "
            "scanners; select a flat owner YAML or a backend with terrain support."
        )
    if not scene.fragment_files:
        return MjwarpSceneContext(
            source_model_file=str(scene.model_file),
            diagnostic_model_file=str(scene.model_file),
        )

    # This is intentionally in a cold-path-only module.  The shared XML
    # composition helper is not a sibling runtime backend dependency.
    from unilab.base.backend.mujoco.xml import materialize_scene_fragments

    materialized = materialize_scene_fragments(
        str(scene.model_file),
        fragment_files=scene.fragment_files,
    )
    return MjwarpSceneContext(
        source_model_file=materialized,
        diagnostic_model_file=str(scene.model_file),
        cleanup_handle=_TemporarySceneCleanup(materialized),
    )
