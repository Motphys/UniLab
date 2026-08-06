"""Cold-path term resolution and pre-bound NumPy Tier 1 execution."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .errors import TermBindingError, TermContractError, TermPlanError, TermRegistrationError
from .registry import TermRegistry
from .spec import (
    NamedTensorSpec,
    NumpyTermContext,
    ParameterValue,
    TensorSpec,
    TermConfig,
    TermDefinition,
    TermKind,
)


@dataclass(frozen=True)
class ResolvedTerm:
    name: str
    definition: TermDefinition
    scale: float
    parameters: Mapping[str, ParameterValue]


@dataclass(frozen=True)
class ResolvedTermPlan:
    terms: tuple[ResolvedTerm, ...]
    input_specs: Mapping[str, TensorSpec]
    workspace_specs: Mapping[str, TensorSpec]
    output_specs: Mapping[str, TensorSpec]

    def materialize(self, *, num_envs: int, inputs: Mapping[str, np.ndarray]) -> NumpyTermRuntime:
        """Validate input views once and allocate all owned buffers."""
        if isinstance(num_envs, bool) or not isinstance(num_envs, numbers.Integral) or num_envs < 1:
            raise TermBindingError(f"num_envs must be a positive integer; got {num_envs!r}")
        count = int(num_envs)
        bound_inputs: dict[str, np.ndarray] = {}
        for name, spec in self.input_specs.items():
            view = inputs.get(name)
            if not isinstance(view, np.ndarray):
                raise TermBindingError(f"missing ndarray input {name!r}")
            expected_shape = (count, *spec.shape)
            if view.shape != expected_shape:
                raise TermBindingError(
                    f"input {name!r} shape mismatch: {expected_shape} != {view.shape}"
                )
            if view.dtype != spec.numpy_dtype:
                raise TermBindingError(
                    f"input {name!r} dtype mismatch: {spec.numpy_dtype} != {view.dtype}"
                )
            bound_inputs[name] = view

        outputs = _allocate(self.output_specs, count)
        workspace = _allocate(self.workspace_specs, count)
        bound: list[_BoundTerm] = []
        for term in self.terms:
            definition = term.definition
            context = NumpyTermContext(
                inputs=_select(definition.inputs, bound_inputs),
                parameters=term.parameters,
                output=outputs[term.name],
                workspace=_select(definition.workspace, workspace),
            )
            bound.append(_BoundTerm(term=term, context=context))
        return NumpyTermRuntime(terms=tuple(bound), outputs=outputs, workspace=workspace)


@dataclass(frozen=True)
class _BoundTerm:
    term: ResolvedTerm
    context: NumpyTermContext


class NumpyTermRuntime:
    """Pre-bound, allocation-stable Python dispatcher for vectorized terms."""

    def __init__(
        self,
        *,
        terms: tuple[_BoundTerm, ...],
        outputs: dict[str, np.ndarray],
        workspace: dict[str, np.ndarray],
    ) -> None:
        self._terms = terms
        self._indices = {bound.term.name: index for index, bound in enumerate(terms)}
        self._scales = np.asarray([bound.term.scale for bound in terms], dtype=np.float64)
        self.outputs: Mapping[str, np.ndarray] = MappingProxyType(outputs)
        self.workspace: Mapping[str, np.ndarray] = MappingProxyType(workspace)

    def set_scale(self, name: str, scale: float) -> None:
        """Update one runtime scale without rebuilding the resolved plan."""
        index = self._indices.get(name)
        if index is None:
            raise TermBindingError(f"unknown term instance {name!r}")
        self._scales[index] = _validate_scale(
            self._terms[index].term.definition, scale, name=name, error_type=TermBindingError
        )

    def execute(self) -> Mapping[str, np.ndarray]:
        """Execute in config order using only pre-bound views and buffers."""
        for index, bound in enumerate(self._terms):
            scale = self._scales[index]
            output = bound.context.output
            if scale == 0.0:
                output.fill(0)
                continue
            bound.term.definition.numpy_fn(bound.context)
            if scale != 1.0:
                np.multiply(output, scale, out=output)
        return self.outputs


def resolve_term_plan(registry: TermRegistry, configs: Sequence[TermConfig]) -> ResolvedTermPlan:
    """Resolve config declaration order into one immutable execution plan."""
    terms: list[ResolvedTerm] = []
    seen_names: set[str] = set()
    input_specs: dict[str, TensorSpec] = {}
    workspace_specs: dict[str, TensorSpec] = {}
    output_specs: dict[str, TensorSpec] = {}

    for config in configs:
        if config.name in seen_names:
            raise TermPlanError(f"duplicate term instance name {config.name!r}")
        seen_names.add(config.name)
        definition = registry.resolve(config.term_key)
        parameters = _resolve_parameters(definition, config.parameters)
        scale = _validate_scale(definition, config.scale, name=config.name)
        for item in definition.inputs:
            _merge_tensor_spec(input_specs, item, owner=definition.key, category="input")
        for item in definition.workspace:
            _merge_tensor_spec(workspace_specs, item, owner=definition.key, category="workspace")
        terms.append(ResolvedTerm(config.name, definition, scale, parameters))
        output_specs[config.name] = definition.output

    namespace_overlap = set(input_specs) & set(workspace_specs)
    if namespace_overlap:
        raise TermPlanError(
            f"plan uses names as both input and workspace: {sorted(namespace_overlap)}"
        )
    return ResolvedTermPlan(
        terms=tuple(terms),
        input_specs=MappingProxyType(input_specs),
        workspace_specs=MappingProxyType(workspace_specs),
        output_specs=MappingProxyType(output_specs),
    )


def _allocate(specs: Mapping[str, TensorSpec], count: int) -> dict[str, np.ndarray]:
    return {
        name: np.zeros((count, *spec.shape), dtype=spec.numpy_dtype) for name, spec in specs.items()
    }


def _select(
    specs: Sequence[NamedTensorSpec], views: Mapping[str, np.ndarray]
) -> Mapping[str, np.ndarray]:
    return MappingProxyType({item.name: views[item.name] for item in specs})


def _merge_tensor_spec(
    merged: dict[str, TensorSpec],
    item: NamedTensorSpec,
    *,
    owner: str,
    category: str,
) -> None:
    existing = merged.get(item.name)
    if existing is not None and existing != item.tensor:
        raise TermPlanError(
            f"{category} {item.name!r} has conflicting specs: {existing!r} vs "
            f"{item.tensor!r} from term {owner!r}"
        )
    merged[item.name] = item.tensor


def _resolve_parameters(
    definition: TermDefinition, configured: Mapping[str, object]
) -> Mapping[str, ParameterValue]:
    specs = {item.name: item for item in definition.parameters}
    unknown = set(configured) - set(specs)
    if unknown:
        raise TermPlanError(f"term {definition.key!r} has unknown parameters {sorted(unknown)}")
    resolved: dict[str, ParameterValue] = {}
    for name, spec in specs.items():
        if name in configured:
            value = configured[name]
        elif spec.required:
            raise TermPlanError(f"term {definition.key!r} is missing parameter {name!r}")
        else:
            value = spec.default
        try:
            resolved[name] = spec.normalize(value)
        except TermRegistrationError as exc:
            raise TermPlanError(f"term {definition.key!r}: {exc}") from exc
    return MappingProxyType(resolved)


def _validate_scale(
    definition: TermDefinition,
    scale: object,
    *,
    name: str,
    error_type: type[TermContractError] = TermPlanError,
) -> float:
    if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
        raise error_type(f"term {name!r} scale must be a finite scalar")
    normalized = float(scale)
    if not math.isfinite(normalized):
        raise error_type(f"term {name!r} scale must be finite")
    if definition.kind is TermKind.TERMINATION and normalized not in (0.0, 1.0):
        raise error_type(f"termination term {name!r} scale must be 0.0 or 1.0; got {normalized}")
    return normalized
