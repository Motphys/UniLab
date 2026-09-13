"""Task-owned fixed model variant declarations.

UniLab owns *which* variant each environment uses. It does not compile engine
models or select an executor representation; those responsibilities stay behind
the UniSim backend contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
class FixedModelVariantCatalogCfg:
    """Task-owned variant sources and their optional explicit assignment."""

    variants: tuple[FixedModelVariantCfg, ...] = field(default_factory=tuple)
    explicit_variant_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.variants, (list, tuple)):
            raise TypeError(
                "FixedModelVariantCatalogCfg.variants must be a sequence of "
                f"FixedModelVariantCfg, got {type(self.variants).__name__}"
            )
        object.__setattr__(self, "variants", tuple(self.variants))
        object.__setattr__(
            self,
            "explicit_variant_names",
            _string_tuple(self.explicit_variant_names, "explicit_variant_names"),
        )
        _validate_catalog(self)


def _build_fixed_variant_plan(
    catalog: FixedModelVariantCatalogCfg,
    num_envs: int,
) -> FixedVariantPlan:
    """Build the immutable UniSim construction-time variant identity."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, (int, np.integer)):
        raise TypeError(f"num_envs must be a positive integer, got {num_envs!r}")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs!r}")

    if catalog.explicit_variant_names:
        requested = catalog.explicit_variant_names
        if len(requested) != num_envs:
            raise ValueError(
                "Explicit fixed-variant assignment must contain exactly num_envs names; "
                f"expected {num_envs}, got {len(requested)}"
            )
        variant_indices = {variant.name: index for index, variant in enumerate(catalog.variants)}
        assignments = np.fromiter(
            (variant_indices[name] for name in requested),
            dtype=np.int32,
            count=num_envs,
        )
    else:
        assignments = np.arange(num_envs, dtype=np.int32) % np.int32(len(catalog.variants))
    assignments.setflags(write=False)

    return FixedVariantPlan(
        assignment=assignments,
        variants=tuple(
            ModelSourceDescriptor(model_file=variant.source_model_file)
            for variant in catalog.variants
        ),
        layout=FixedVariantLayout.SAME_LAYOUT,
    )


def _require_fixed_variant_support(
    capabilities: DomainRandomizationCapabilities,
    plan: FixedVariantPlan,
) -> None:
    """Fail closed unless UniSim can realize the complete variant plan."""

    rejections = capabilities.fixed_variant_rejections(plan)
    if rejections:
        rendered = "; ".join(rejections)
        raise NotImplementedError(
            f"{type(capabilities).__name__} cannot realize the fixed variant plan: {rendered}"
        )


def _validate_catalog(catalog: FixedModelVariantCatalogCfg) -> None:
    if not catalog.variants:
        raise ValueError("FixedModelVariantCatalogCfg.variants must not be empty")

    names: set[str] = set()
    for index, variant in enumerate(catalog.variants):
        label = f"FixedModelVariantCatalogCfg.variants[{index}]"
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

    unknown = [name for name in catalog.explicit_variant_names if name not in names]
    if unknown:
        available = [variant.name for variant in catalog.variants]
        raise ValueError(
            "Explicit fixed-variant assignment references unknown variants "
            f"{unknown}; available variants are {available}"
        )


def _string_tuple(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of strings, got {type(values).__name__}")
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


__all__ = [
    "FixedModelVariantCatalogCfg",
    "FixedModelVariantCfg",
]
