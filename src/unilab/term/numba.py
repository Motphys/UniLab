"""Cold-path assembly and in-process caching for trusted Numba term plans."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .errors import TermBindingError, TermPlanError
from .plan import NumpyTermRuntime, ResolvedTermPlan

try:  # pragma: no cover - optional dependency boundary
    from numba import njit, prange
    from numba.np.ufunc.parallel import get_thread_id

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    get_thread_id = njit = prange = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class FusedOutputLayout:
    """Route resolved term instances into fused task-owned outputs."""

    rewards: tuple[str, ...]
    observations: Mapping[str, tuple[str, ...]]
    terminations: tuple[str, ...]
    preambles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rewards", tuple(self.rewards))
        object.__setattr__(self, "terminations", tuple(self.terminations))
        object.__setattr__(self, "preambles", tuple(self.preambles))
        object.__setattr__(
            self,
            "observations",
            MappingProxyType({name: tuple(terms) for name, terms in self.observations.items()}),
        )


@dataclass(frozen=True)
class FusedCompileInfo:
    cache_key: str
    cache_hit: bool
    assembly_ms: float


class FusedNumbaRuntime:
    def __init__(
        self,
        *,
        plan: ResolvedTermPlan,
        numpy_runtime: NumpyTermRuntime,
        layout: FusedOutputLayout,
        kernel: Any,
        compile_info: FusedCompileInfo,
        parameters: np.ndarray,
        inputs: Mapping[str, np.ndarray],
        observations: Mapping[str, np.ndarray],
        reward: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        self.plan = plan
        self.layout = layout
        self.compile_info = compile_info
        self._numpy_runtime = numpy_runtime
        self._kernel = kernel
        self._inputs = tuple(inputs[name] for name in plan.input_specs)
        self._workspace = tuple(numpy_runtime.workspace[name] for name in plan.workspace_specs)
        self._observations = tuple(observations[name] for name in layout.observations)
        self.reward = reward
        self.terminated = terminated
        self._indices = {term.name: index for index, term in enumerate(plan.terms)}
        self.scales = np.asarray([term.scale for term in plan.terms], dtype=np.float64)
        self.parameters = parameters

    def bind_inputs(self, inputs: Mapping[str, np.ndarray]) -> None:
        """Rebind task-owned views, validating only identities that changed."""
        views = []
        for index, (name, spec) in enumerate(self.plan.input_specs.items()):
            view = inputs.get(name)
            if not isinstance(view, np.ndarray):
                raise TermBindingError(f"missing ndarray input {name!r}")
            if view is not self._inputs[index]:
                expected = (self.reward.shape[0], *spec.shape)
                if view.shape != expected or view.dtype != spec.numpy_dtype:
                    raise TermBindingError(
                        f"input {name!r} must have shape {expected} and dtype {spec.numpy_dtype}"
                    )
            views.append(view)
        self._inputs = tuple(views)

    def set_scale(self, name: str, scale: float) -> None:
        index = self._indices.get(name)
        if index is None:
            raise TermBindingError(f"unknown term instance {name!r}")
        self._numpy_runtime.set_scale(name, scale)
        self.scales[index] = float(scale)

    def execute(self, *, reward_multiplier: float, log_scratch: np.ndarray) -> None:
        self._kernel(
            self._inputs,
            self.parameters,
            self._workspace,
            self.scales,
            self._observations,
            self.reward,
            self.terminated,
            log_scratch,
            float(reward_multiplier),
        )


_KERNEL_CACHE: dict[str, Any] = {}


def clear_numba_plan_cache() -> None:
    """Test helper for deterministic cache-hit assertions."""
    _KERNEL_CACHE.clear()


def materialize_numba_plan(
    plan: ResolvedTermPlan,
    layout: FusedOutputLayout,
    *,
    num_envs: int,
    inputs: Mapping[str, np.ndarray],
    observations: Mapping[str, np.ndarray],
    reward: np.ndarray,
    terminated: np.ndarray,
) -> FusedNumbaRuntime:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba fused term execution requires the optional numba dependency")
    assert njit is not None
    _validate_layout(plan, layout, observations, num_envs=num_envs)
    numpy_runtime = plan.materialize(num_envs=num_envs, inputs=inputs)
    parameters = _flatten_parameters(plan)
    _validate_numba_definitions(plan)
    _validate_output("reward", reward, (num_envs,), np.floating)
    _validate_output("terminated", terminated, (num_envs,), np.bool_)

    key = _cache_key(plan, layout)
    started = time.perf_counter()
    kernel = _KERNEL_CACHE.get(key)
    cache_hit = kernel is not None
    if kernel is None:
        function, _namespace = _build_kernel(plan, layout)
        kernel = njit(parallel=True, fastmath=True, nogil=True)(function)
        _KERNEL_CACHE[key] = kernel
    info = FusedCompileInfo(key, cache_hit, (time.perf_counter() - started) * 1e3)
    return FusedNumbaRuntime(
        plan=plan,
        numpy_runtime=numpy_runtime,
        layout=layout,
        kernel=kernel,
        compile_info=info,
        parameters=parameters,
        inputs=inputs,
        observations=observations,
        reward=reward,
        terminated=terminated,
    )


def _validate_output(name: str, value: np.ndarray, shape: tuple[int, ...], dtype: Any) -> None:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise TermBindingError(f"fused output {name!r} must have shape {shape}")
    if dtype is np.bool_:
        valid = value.dtype == np.dtype(np.bool_)
    else:
        valid = np.issubdtype(value.dtype, dtype)
    if not valid:
        raise TermBindingError(f"fused output {name!r} has invalid dtype {value.dtype}")


def _validate_layout(
    plan: ResolvedTermPlan,
    layout: FusedOutputLayout,
    observations: Mapping[str, np.ndarray],
    *,
    num_envs: int,
) -> None:
    routed = [*layout.preambles, *layout.rewards, *layout.terminations]
    routed.extend(name for names in layout.observations.values() for name in names)
    expected = [term.name for term in plan.terms]
    if len(routed) != len(set(routed)) or set(routed) != set(expected):
        raise TermPlanError("fused output layout must route every resolved term exactly once")
    if set(observations) != set(layout.observations):
        raise TermBindingError("fused observation buffers must exactly match the output layout")
    for group, names in layout.observations.items():
        width = sum(plan.output_specs[name].shape[0] for name in names)
        value = observations.get(group)
        _validate_output(group, value, (num_envs, width), np.floating)  # type: ignore[arg-type]


def _flatten_parameters(plan: ResolvedTermPlan) -> np.ndarray:
    values: list[float] = []
    for term in plan.terms:
        for spec in term.definition.parameters:
            value = term.parameters[spec.name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TermPlanError(
                    f"Numba fused term {term.name!r} requires scalar numeric parameters"
                )
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _validate_numba_definitions(plan: ResolvedTermPlan) -> None:
    missing = [term.definition.key for term in plan.terms if term.definition.numba_item_fn is None]
    if missing:
        raise TermPlanError(f"terms have no Numba item implementation: {missing}")


def _cache_key(plan: ResolvedTermPlan, layout: FusedOutputLayout) -> str:
    def specs(items: Mapping[str, Any]) -> list[tuple[str, tuple[int, ...], str]]:
        return [(name, spec.shape, spec.dtype) for name, spec in items.items()]

    payload = {
        "terms": [
            (term.name, term.definition.key, term.definition.implementation_version)
            for term in plan.terms
        ],
        "inputs": specs(plan.input_specs),
        "workspace": specs(plan.workspace_specs),
        "outputs": specs(plan.output_specs),
        "layout": {
            "preambles": layout.preambles,
            "rewards": layout.rewards,
            "observations": list(layout.observations.items()),
            "terminations": layout.terminations,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_kernel(plan: ResolvedTermPlan, layout: FusedOutputLayout) -> tuple[Any, dict[str, Any]]:
    input_indices = {name: index for index, name in enumerate(plan.input_specs)}
    workspace_indices = {name: index for index, name in enumerate(plan.workspace_specs)}
    roles = {name: "preamble" for name in layout.preambles}
    roles.update({name: "reward" for name in layout.rewards})
    roles.update({name: "termination" for name in layout.terminations})
    observation_routes = {}
    for group_index, (group, names) in enumerate(layout.observations.items()):
        offset = 0
        for name in names:
            roles[name] = "observation"
            observation_routes[name] = (group_index, offset)
            offset += plan.output_specs[name].shape[0]

    namespace: dict[str, Any] = {"prange": prange, "get_thread_id": get_thread_id}
    lines = [
        "def fused(inputs, parameters, workspace, scales, observations, reward, terminated, log_scratch, reward_multiplier):",
        "    n = reward.shape[0]",
        "    for i in prange(n):",
        "        tid = get_thread_id()",
        "        reward[i] = 0.0",
        "        terminated[i] = False",
    ]
    parameter_offset = 0
    for index, term in enumerate(plan.terms):
        definition = term.definition
        if definition.numba_item_fn is None:
            raise TermPlanError(f"term {definition.key!r} has no Numba item implementation")
        fn_name = f"item_{index}"
        namespace[fn_name] = definition.numba_item_fn
        args = [f"inputs[{input_indices[item.name]}]" for item in definition.inputs]
        args.extend(f"parameters[{parameter_offset + i}]" for i in range(len(term.parameters)))
        args.extend(f"workspace[{workspace_indices[item.name]}]" for item in definition.workspace)
        joined = ", ".join(args)
        joined = f"{joined}, " if joined else ""
        role = roles[term.name]
        lines.append(f"        if scales[{index}] != 0.0:")
        if role == "reward":
            lines.extend(
                (
                    f"            value = {fn_name}({joined}i)",
                    f"            weighted = value * scales[{index}]",
                    "            reward[i] += weighted",
                    f"            log_scratch[tid, {index}] += weighted",
                )
            )
        elif role == "termination":
            lines.append(f"            terminated[i] |= {fn_name}({joined}i)")
        elif role == "observation":
            group_index, offset = observation_routes[term.name]
            lines.append(f"            {fn_name}({joined}observations[{group_index}], {offset}, i)")
        else:
            lines.append(f"            {fn_name}({joined}i)")
        parameter_offset += len(term.parameters)
    lines.append("        reward[i] *= reward_multiplier")
    exec("\n".join(lines), namespace)  # noqa: S102 - trusted registry functions only
    return namespace["fused"], namespace
