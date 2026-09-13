"""Task-owned fixed model variants and immutable assignment materialization.

UniLab owns *which* variant each environment uses.  It does not compile engine
models or select an executor representation; those responsibilities stay behind
the UniSim backend contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from unisim.dr.types import (
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
)


@dataclass(frozen=True)
class FixedModelVariantCfg:
    """One named, pickle-safe source descriptor for a fixed model variant."""

    name: str
    source_model_file: str


@dataclass(frozen=True)
class FixedModelVariantAssignmentCfg:
    """Declare how a task maps environments to named fixed variants."""

    mode: Literal["round_robin", "explicit"] = "round_robin"
    explicit_variant_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "explicit_variant_names", _string_tuple(self.explicit_variant_names)
        )
        if self.mode not in ("round_robin", "explicit"):
            raise ValueError(
                "FixedModelVariantAssignmentCfg.mode must be 'round_robin' or 'explicit'; "
                f"got {self.mode!r}"
            )
        for index, name in enumerate(self.explicit_variant_names):
            if not name.strip():
                raise ValueError(
                    "FixedModelVariantAssignmentCfg.explicit_variant_names"
                    f"[{index}] must be a non-empty string"
                )


@dataclass(frozen=True)
class FixedModelVariantCatalogCfg:
    """Task-owned catalog of same-public-layout model/tool sources."""

    variants: tuple[FixedModelVariantCfg, ...] = field(default_factory=tuple)
    assignment: FixedModelVariantAssignmentCfg = field(
        default_factory=FixedModelVariantAssignmentCfg
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variants, (list, tuple)):
            raise TypeError(
                "FixedModelVariantCatalogCfg.variants must be a sequence of "
                f"FixedModelVariantCfg, got {type(self.variants).__name__}"
            )
        object.__setattr__(self, "variants", tuple(self.variants))
        if not isinstance(self.assignment, FixedModelVariantAssignmentCfg):
            raise TypeError(
                "FixedModelVariantCatalogCfg.assignment must be "
                f"FixedModelVariantAssignmentCfg, got {type(self.assignment).__name__}"
            )
        _validate_catalog(self)


@dataclass(frozen=True)
class FixedModelVariantMaterialization:
    """The final immutable variant selection handed to the owner boundary.

    ``model_assignments`` is an ``int32`` array with shape ``(num_envs,)`` and
    is marked read-only.  Consumers that need a writable projection must call
    :meth:`copy_model_assignments`; they must never mutate the final task
    identity in place.
    """

    variants: tuple[FixedModelVariantCfg, ...]
    model_assignments: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.variants, (list, tuple)):
            raise TypeError(
                "FixedModelVariantMaterialization.variants must be a sequence of "
                f"FixedModelVariantCfg, got {type(self.variants).__name__}"
            )
        object.__setattr__(self, "variants", tuple(self.variants))
        if not isinstance(self.model_assignments, np.ndarray):
            raise TypeError(
                "Fixed model variant model_assignments must be np.ndarray, "
                f"got {type(self.model_assignments).__name__}"
            )
        if self.model_assignments.dtype.kind not in "iu":
            raise ValueError(
                "Fixed model variant model_assignments must have an integer dtype; "
                f"got {self.model_assignments.dtype}"
            )
        object.__setattr__(
            self,
            "model_assignments",
            np.ascontiguousarray(self.model_assignments, dtype=np.int32),
        )
        self.model_assignments.setflags(write=False)

    @property
    def variant_names(self) -> tuple[str, ...]:
        return tuple(variant.name for variant in self.variants)

    def copy_model_assignments(self) -> np.ndarray:
        """Return a writable backend-local copy of the final assignment."""

        return np.array(self.model_assignments, dtype=np.int32, copy=True)


def _validate_catalog(catalog: FixedModelVariantCatalogCfg) -> None:
    if not catalog.variants:
        raise ValueError("FixedModelVariantCatalogCfg.variants must not be empty")
    _validate_variants(catalog.variants, "FixedModelVariantCatalogCfg.variants")
    if catalog.assignment.mode == "round_robin" and catalog.assignment.explicit_variant_names:
        raise ValueError(
            "FixedModelVariantAssignmentCfg.explicit_variant_names must be empty "
            "when assignment mode is 'round_robin'"
        )


def _validate_variants(variants: tuple[FixedModelVariantCfg, ...], label_prefix: str) -> None:
    if not variants:
        raise ValueError(f"{label_prefix} must not be empty")

    names: set[str] = set()
    for index, variant in enumerate(variants):
        label = f"{label_prefix}[{index}]"
        if not isinstance(variant, FixedModelVariantCfg):
            raise TypeError(f"{label} must be FixedModelVariantCfg, got {type(variant).__name__}")
        if not isinstance(variant.name, str) or not variant.name.strip():
            raise ValueError(f"{label}.name must be a non-empty string")
        if not isinstance(variant.source_model_file, str) or not variant.source_model_file.strip():
            raise ValueError(
                f"{label}('{variant.name}').source_model_file must be a non-empty string"
            )
        if variant.name in names:
            raise ValueError(
                "Fixed model variant names must be unique; duplicate "
                f"{variant.name!r} was declared more than once"
            )
        names.add(variant.name)


def _string_tuple(values: object) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise TypeError(f"Expected a sequence of strings, got {type(values).__name__}")
    result = tuple(values)
    if any(not isinstance(value, str) for value in result):
        kinds = sorted({type(value).__name__ for value in result if not isinstance(value, str)})
        raise TypeError(f"Expected a sequence of strings, got {kinds}")
    return result


def materialize_fixed_model_variants(
    catalog: FixedModelVariantCatalogCfg, num_envs: int
) -> FixedModelVariantMaterialization:
    """Materialize a final assignment without touching an engine object.

    This is deliberately a cold-path operation: it resolves only task names and
    integer indices.  It never opens or compiles ``source_model_file``.
    """

    _validate_catalog(catalog)
    if isinstance(num_envs, bool) or not isinstance(num_envs, (int, np.integer)):
        raise TypeError(f"num_envs must be a positive integer, got {num_envs!r}")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    variant_indices = {variant.name: index for index, variant in enumerate(catalog.variants)}
    if catalog.assignment.mode == "round_robin":
        if catalog.assignment.explicit_variant_names:
            raise ValueError(
                "FixedModelVariantAssignmentCfg.explicit_variant_names must be empty "
                "when assignment mode is 'round_robin'"
            )
        assignments = np.arange(num_envs, dtype=np.int32) % np.int32(len(catalog.variants))
    else:
        requested = catalog.assignment.explicit_variant_names
        if len(requested) != num_envs:
            raise ValueError(
                "Explicit fixed-variant assignment must contain exactly num_envs names; "
                f"expected {num_envs}, got {len(requested)}"
            )
        unknown = [name for name in requested if name not in variant_indices]
        if unknown:
            available = [variant.name for variant in catalog.variants]
            raise ValueError(
                f"Explicit fixed-variant assignment references unknown variants {unknown}; "
                f"available variants are {available}"
            )
        assignments = np.fromiter(
            (variant_indices[name] for name in requested),
            dtype=np.int32,
            count=num_envs,
        )

    materialization = FixedModelVariantMaterialization(catalog.variants, assignments)
    validate_fixed_model_variant_materialization(materialization, num_envs)
    return materialization


def prepare_fixed_model_variants(
    catalog: FixedModelVariantCatalogCfg,
    num_envs: int,
    capabilities: DomainRandomizationCapabilities,
) -> FixedModelVariantMaterialization:
    """Materialize a task assignment after negotiating the single DR contract.

    This owner-layer helper is the integration seam used by env construction.
    It intentionally accepts only the already-materialized backend capability
    object; it never probes a backend type or optional engine package.
    """

    require_fixed_model_variant_support(capabilities)
    return materialize_fixed_model_variants(catalog, num_envs)


def build_fixed_variant_plan(
    materialization: FixedModelVariantMaterialization,
) -> FixedVariantPlan:
    """Translate the final task selection into UniSim's neutral variant plan."""

    num_envs = int(materialization.model_assignments.size)
    validate_fixed_model_variant_materialization(materialization, num_envs)
    return FixedVariantPlan(
        assignment=materialization.model_assignments,
        variants=tuple(
            ModelSourceDescriptor(model_file=variant.source_model_file)
            for variant in materialization.variants
        ),
        layout=FixedVariantLayout.SAME_LAYOUT,
    )


def validate_fixed_model_variant_materialization(
    materialization: FixedModelVariantMaterialization,
    num_envs: int,
    *,
    require_immutable_assignment: bool = True,
) -> None:
    """Validate the final assignment shape, range, and immutable state."""

    if not isinstance(materialization, FixedModelVariantMaterialization):
        raise TypeError(
            "Fixed model variant materialization must be "
            f"FixedModelVariantMaterialization, got {type(materialization).__name__}"
        )
    if isinstance(num_envs, bool) or not isinstance(num_envs, (int, np.integer)) or num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs!r}")
    _validate_variants(materialization.variants, "FixedModelVariantMaterialization.variants")
    assignments = materialization.model_assignments
    if not isinstance(assignments, np.ndarray):
        raise TypeError(
            "Fixed model variant model_assignments must be np.ndarray, "
            f"got {type(assignments).__name__}"
        )
    if assignments.shape != (num_envs,):
        raise ValueError(
            f"Fixed model variant model_assignments must have shape ({num_envs},); "
            f"got {assignments.shape}"
        )
    if assignments.dtype.kind not in "iu":
        raise ValueError(
            "Fixed model variant model_assignments must have an integer dtype; "
            f"got {assignments.dtype}"
        )
    if np.any(assignments < 0) or np.any(assignments >= len(materialization.variants)):
        raise ValueError("Fixed model variant model_assignments contains an out-of-range index")
    if require_immutable_assignment and assignments.flags.writeable:
        raise ValueError(
            "Final fixed model variant model_assignments must be read-only after materialization"
        )


def require_fixed_model_variant_support(
    capabilities: DomainRandomizationCapabilities,
) -> None:
    """Fail closed unless UniSim explicitly declares fixed-variant support.

    The authoritative declaration remains UniSim's DR capability object.  This
    helper intentionally does not infer support from a backend type, installed
    engine, or optional import.
    """

    declared = getattr(capabilities, "supports_fixed_variants", False)
    if declared is not True:
        raise NotImplementedError(
            f"{type(capabilities).__name__} does not support fixed model variants "
            f"(supports_fixed_variants={declared!r})"
        )


__all__ = [
    "FixedModelVariantAssignmentCfg",
    "FixedModelVariantCatalogCfg",
    "FixedModelVariantCfg",
    "FixedModelVariantMaterialization",
    "build_fixed_variant_plan",
    "materialize_fixed_model_variants",
    "prepare_fixed_model_variants",
    "require_fixed_model_variant_support",
    "validate_fixed_model_variant_materialization",
]
