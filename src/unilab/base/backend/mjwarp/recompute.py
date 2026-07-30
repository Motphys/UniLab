"""Plan-scoped Model constant recomputation for the ``mjwarp`` backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..batch import BackendBatchContractError
from ..mutation import (
    BoundMutationPlan,
    MutationCapabilityManifest,
    MutationContractError,
    MutationRecomputeLevel,
    MutationTargetKind,
    mutation_capability_fingerprint,
)
from .materialization import MjwarpModelFieldRole, MjwarpModelMaterializationReceipt


class MjwarpModelRecomputeKind(str, Enum):
    """Backend operation selected by the Model recompute lattice."""

    NONE = "none"
    SET_CONST_FIXED = "set_const_fixed"
    SET_CONST_0 = "set_const_0"
    SET_CONST = "set_const"


_FIXED_DERIVED_FIELDS = ("body_subtreemass",)
_REFERENCE_DERIVED_FIELDS = (
    "actuator_acc0",
    "body_invweight0",
    "dof_invweight0",
    "tendon_invweight0",
    "tendon_length0",
)
_RECOMPUTE_BITS = {
    MjwarpModelRecomputeKind.NONE: 0,
    MjwarpModelRecomputeKind.SET_CONST_FIXED: 1,
    MjwarpModelRecomputeKind.SET_CONST_0: 2,
    MjwarpModelRecomputeKind.SET_CONST: 3,
}
_BITS_TO_RECOMPUTE = {bits: kind for kind, bits in _RECOMPUTE_BITS.items()}
_MODEL_DIRECT_FIELDS = {
    "actuator.pd_damping": ("actuator_biasprm",),
    "actuator.pd_stiffness": ("actuator_biasprm", "actuator_gainprm"),
    "body.gravity_compensation": ("body_gravcomp",),
    "joint.armature": ("dof_armature",),
}
_MODEL_RECOMPUTE_LEVELS = {
    "actuator.pd_damping": MutationRecomputeLevel.NONE,
    "actuator.pd_stiffness": MutationRecomputeLevel.NONE,
    "body.gravity_compensation": MutationRecomputeLevel.KINEMATICS,
    "joint.armature": MutationRecomputeLevel.DYNAMICS,
}


def model_recompute_kind(
    target_kind: MutationTargetKind,
    level: MutationRecomputeLevel,
) -> MjwarpModelRecomputeKind:
    """Map backend-neutral semantics without treating state forward as Model work."""

    if not isinstance(target_kind, MutationTargetKind):
        raise MutationContractError("mjwarp recompute target kind is invalid")
    if not isinstance(level, MutationRecomputeLevel):
        raise MutationContractError("mjwarp recompute level is invalid")
    if target_kind is not MutationTargetKind.MODEL_PARAMETER:
        return MjwarpModelRecomputeKind.NONE
    return {
        MutationRecomputeLevel.NONE: MjwarpModelRecomputeKind.NONE,
        MutationRecomputeLevel.KINEMATICS: MjwarpModelRecomputeKind.SET_CONST_FIXED,
        MutationRecomputeLevel.DYNAMICS: MjwarpModelRecomputeKind.SET_CONST_0,
        MutationRecomputeLevel.FULL: MjwarpModelRecomputeKind.SET_CONST,
    }[level]


def join_model_recompute(
    requirements: tuple[MjwarpModelRecomputeKind, ...],
) -> MjwarpModelRecomputeKind:
    """Join fixed/reference requirements as a two-bit lattice."""

    if not isinstance(requirements, tuple) or any(
        not isinstance(item, MjwarpModelRecomputeKind) for item in requirements
    ):
        raise MutationContractError("mjwarp recompute requirements must be a typed tuple")
    bits = 0
    for requirement in requirements:
        bits |= _RECOMPUTE_BITS[requirement]
    return _BITS_TO_RECOMPUTE[bits]


def recompute_derived_fields(kind: MjwarpModelRecomputeKind) -> tuple[str, ...]:
    """Return the exact Model outputs that must have per-world storage."""

    if not isinstance(kind, MjwarpModelRecomputeKind):
        raise MutationContractError("mjwarp recompute kind is invalid")
    if kind is MjwarpModelRecomputeKind.NONE:
        return ()
    if kind is MjwarpModelRecomputeKind.SET_CONST_FIXED:
        return _FIXED_DERIVED_FIELDS
    if kind is MjwarpModelRecomputeKind.SET_CONST_0:
        return _REFERENCE_DERIVED_FIELDS
    return tuple(sorted((*_FIXED_DERIVED_FIELDS, *_REFERENCE_DERIVED_FIELDS)))


@dataclass(frozen=True)
class MjwarpModelRecomputeContract:
    """Cold-compiled Model fields and one joined recompute operation."""

    kind: MjwarpModelRecomputeKind
    direct_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    state_forward_required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MjwarpModelRecomputeKind):
            raise MutationContractError("mjwarp Model recompute contract has an invalid kind")
        for name in ("direct_fields", "derived_fields"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or values != tuple(sorted(values))
                or len(values) != len(set(values))
            ):
                raise MutationContractError(
                    f"mjwarp Model recompute {name} must be canonical and unique"
                )
        if not self.direct_fields:
            raise MutationContractError("mjwarp Model recompute contract requires direct fields")
        if set(self.direct_fields).intersection(self.derived_fields):
            raise MutationContractError("mjwarp Model direct and derived field roles overlap")
        if self.derived_fields != recompute_derived_fields(self.kind):
            raise MutationContractError(
                "mjwarp Model recompute derived fields do not match the joined operation"
            )
        if not isinstance(self.state_forward_required, bool):
            raise MutationContractError("mjwarp state forward requirement must be a bool")


def compile_model_recompute_contract(
    plan: BoundMutationPlan,
    manifest: MutationCapabilityManifest,
) -> MjwarpModelRecomputeContract:
    """Compile and verify the Model field/recompute contract for one bound plan."""

    if not isinstance(plan, BoundMutationPlan):
        raise MutationContractError("mjwarp recompute compilation requires a bound mutation plan")
    if not isinstance(manifest, MutationCapabilityManifest):
        raise MutationContractError("mjwarp recompute compilation requires a capability manifest")
    manifest.require_valid()
    if plan.capability_manifest_fingerprint != manifest.fingerprint:
        raise MutationContractError(
            "mjwarp mutation plan and capability manifest identities do not match"
        )

    capabilities = {capability.target_key: capability for capability in manifest.capabilities}
    direct_fields: set[str] = set()
    derived_fields: set[str] = set()
    requirements: list[MjwarpModelRecomputeKind] = []
    state_forward_required = False
    has_model = False
    for spec in plan.specs:
        if spec.target.target_kind is MutationTargetKind.SIMULATION_STATE:
            state_forward_required |= spec.recompute is not MutationRecomputeLevel.NONE
            continue
        if spec.target.target_kind is not MutationTargetKind.MODEL_PARAMETER:
            continue
        has_model = True
        try:
            capability = capabilities[spec.target.target_key]
        except KeyError as exc:  # pragma: no cover - shared binding already rejects this.
            raise MutationContractError(
                f"mjwarp Model target {spec.target.target_key!r} has no capability"
            ) from exc
        if spec.capability_fingerprint != mutation_capability_fingerprint(capability):
            raise MutationContractError(
                f"mjwarp Model target {spec.target.target_key!r} has a stale capability identity"
            )
        expected_level = _MODEL_RECOMPUTE_LEVELS.get(spec.target.target_key)
        if expected_level is None or spec.recompute is not expected_level:
            raise MutationContractError(
                f"mjwarp Model target {spec.target.target_key!r} has an invalid recompute level"
            )
        descriptor = capability.descriptor
        if descriptor is None:
            raise MutationContractError(
                f"mjwarp Model target {spec.target.target_key!r} lacks a field descriptor"
            )
        expected_direct = _MODEL_DIRECT_FIELDS.get(spec.target.target_key)
        if expected_direct is None or descriptor.direct_fields != expected_direct:
            raise MutationContractError(
                f"mjwarp Model target {spec.target.target_key!r} has invalid direct fields"
            )
        requirement = model_recompute_kind(spec.target.target_kind, spec.recompute)
        expected_derived = recompute_derived_fields(requirement)
        if descriptor.derived_fields != expected_derived:
            raise MutationContractError(
                f"mjwarp Model target {spec.target.target_key!r} has invalid derived fields"
            )
        overlap = direct_fields.intersection(
            descriptor.derived_fields
        ) | derived_fields.intersection(descriptor.direct_fields)
        if overlap:
            raise MutationContractError(
                "mjwarp Model capabilities disagree on field roles: " + ", ".join(sorted(overlap))
            )
        direct_fields.update(descriptor.direct_fields)
        derived_fields.update(descriptor.derived_fields)
        requirements.append(requirement)

    if not has_model:
        raise MutationContractError("mjwarp Model recompute compilation found no Model targets")
    joined = join_model_recompute(tuple(requirements))
    expected_joined_fields = set(recompute_derived_fields(joined))
    if derived_fields != expected_joined_fields:
        raise MutationContractError(
            "mjwarp joined Model recompute fields are incomplete or contain extras"
        )
    return MjwarpModelRecomputeContract(
        kind=joined,
        direct_fields=tuple(sorted(direct_fields)),
        derived_fields=tuple(sorted(derived_fields)),
        state_forward_required=state_forward_required,
    )


def _receipt_identity(
    receipt: MjwarpModelMaterializationReceipt,
) -> tuple[tuple[str, MjwarpModelFieldRole, tuple[int, ...], int], ...]:
    return tuple(
        (field.field_name, field.role, field.materialized_shape, field.materialized_address)
        for field in receipt.fields
    )


@dataclass(frozen=True)
class MjwarpModelRecomputeDiagnostics:
    mutation_plan_fingerprint: str
    kind: MjwarpModelRecomputeKind
    direct_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    state_forward_required: bool
    capture_count: int
    launch_count: int
    storage_generation: int
    storage_fingerprint: str
    materialization_receipt_fingerprint: str
    instrumentation_complete: bool = True


@dataclass
class MjwarpModelRecomputeRuntime:
    """Captured recompute graph tied to one mutation and storage identity."""

    public_plan: BoundMutationPlan
    contract: MjwarpModelRecomputeContract
    graph: Any | None = field(repr=False)
    storage_generation: int
    storage_fingerprint: str
    materialization_receipt: MjwarpModelMaterializationReceipt = field(repr=False)
    _receipt_identity_snapshot: tuple[
        tuple[str, MjwarpModelFieldRole, tuple[int, ...], int], ...
    ] = field(init=False, repr=False)
    _launch_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.materialization_receipt.verify_fingerprint()
        if (
            self.public_plan.backend_instance_id != self.materialization_receipt.backend_instance_id
            or self.public_plan.num_envs != self.materialization_receipt.num_worlds
        ):
            raise BackendBatchContractError(
                "mjwarp recompute plan and materialization receipt have different owners"
            )
        if (
            self.storage_generation != self.materialization_receipt.storage_generation_after
            or self.storage_fingerprint != self.materialization_receipt.storage_fingerprint_after
        ):
            raise BackendBatchContractError(
                "mjwarp recompute storage identity does not match its materialization receipt"
            )
        required_roles = {
            **{name: MjwarpModelFieldRole.DIRECT for name in self.contract.direct_fields},
            **{name: MjwarpModelFieldRole.DERIVED for name in self.contract.derived_fields},
        }
        receipt_roles = {item.field_name: item.role for item in self.materialization_receipt.fields}
        missing = tuple(sorted(set(required_roles).difference(receipt_roles)))
        mismatched = tuple(
            sorted(
                name
                for name, role in required_roles.items()
                if receipt_roles.get(name) is not None and receipt_roles[name] is not role
            )
        )
        if missing or mismatched:
            raise BackendBatchContractError(
                "mjwarp recompute receipt does not satisfy required field roles: "
                f"missing={missing!r}, mismatched={mismatched!r}"
            )
        if (self.contract.kind is MjwarpModelRecomputeKind.NONE) != (self.graph is None):
            raise BackendBatchContractError(
                "mjwarp recompute graph presence does not match the compiled operation"
            )
        if self.storage_generation < 0 or not self.storage_fingerprint:
            raise BackendBatchContractError("mjwarp recompute storage identity is invalid")
        self._receipt_identity_snapshot = _receipt_identity(self.materialization_receipt)

    def require_current(
        self,
        *,
        plan: BoundMutationPlan,
        storage_generation: int,
        storage_fingerprint: str,
        materialization_receipt: MjwarpModelMaterializationReceipt | None,
    ) -> None:
        self.public_plan.require_compatible(plan)
        if (
            storage_generation != self.storage_generation
            or storage_fingerprint != self.storage_fingerprint
        ):
            raise BackendBatchContractError(
                "mjwarp Model recompute graph has a stale storage identity"
            )
        if (
            materialization_receipt is not self.materialization_receipt
            or materialization_receipt.fingerprint != self.materialization_receipt.fingerprint
            or _receipt_identity(materialization_receipt) != self._receipt_identity_snapshot
        ):
            raise BackendBatchContractError(
                "mjwarp Model recompute materialization receipt changed after binding"
            )

    def launch(self, warp: Any) -> bool:
        if self.graph is None:
            return False
        warp.capture_launch(self.graph)
        self._launch_count += 1
        return True

    @property
    def diagnostics(self) -> MjwarpModelRecomputeDiagnostics:
        return MjwarpModelRecomputeDiagnostics(
            mutation_plan_fingerprint=self.public_plan.fingerprint,
            kind=self.contract.kind,
            direct_fields=self.contract.direct_fields,
            derived_fields=self.contract.derived_fields,
            state_forward_required=self.contract.state_forward_required,
            capture_count=int(self.graph is not None),
            launch_count=self._launch_count,
            storage_generation=self.storage_generation,
            storage_fingerprint=self.storage_fingerprint,
            materialization_receipt_fingerprint=self.materialization_receipt.fingerprint,
        )


__all__ = [
    "MjwarpModelRecomputeContract",
    "MjwarpModelRecomputeDiagnostics",
    "MjwarpModelRecomputeKind",
    "MjwarpModelRecomputeRuntime",
    "compile_model_recompute_contract",
    "join_model_recompute",
    "model_recompute_kind",
    "recompute_derived_fields",
]
