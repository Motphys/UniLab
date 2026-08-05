from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from unilab.term import (  # noqa: E402
    NamedTensorSpec,
    NumpyTermContext,
    ParameterKind,
    ParameterSpec,
    TensorSpec,
    TermConfig,
    TermDefinition,
    TermKind,
    TermPlanError,
    TermRegistry,
    resolve_term_plan,
)
from unilab.term.numba import (  # noqa: E402
    FusedOutputLayout,
    clear_numba_plan_cache,
    materialize_numba_plan,
)


@numba.njit(inline="always", fastmath=True, nogil=True)
def _reward_item(state, gain, scratch, index):
    scratch[index, 0] = state[index, 0] + state[index, 1]
    return scratch[index, 0] * gain


def _reward_numpy(context: NumpyTermContext) -> None:
    np.sum(context.inputs["state"], axis=1, out=context.workspace["scratch"][:, 0])
    np.multiply(context.workspace["scratch"][:, 0], context.parameters["gain"], out=context.output)


@numba.njit(inline="always", fastmath=True, nogil=True)
def _typed_parameter_item(state, enabled, offset, indices, index):
    value = 0.0
    if enabled:
        for column in indices:
            value += state[index, column]
    return value + offset


def _typed_parameter_numpy(context: NumpyTermContext) -> None:
    context.output.fill(float(context.parameters["offset"]))
    if context.parameters["enabled"]:
        context.output += context.inputs["state"][:, context.parameters["indices"]].sum(axis=1)


def _plan(
    *,
    dtype: type[np.floating] = np.float32,
    workspace_width: int = 1,
    version: int = 1,
    order: Sequence[str] = ("a",),
    item_fn=_reward_item,
):
    registry = TermRegistry()
    for label in ("a", "b"):
        registry.register(
            TermDefinition(
                f"synthetic.reward.{label}.v1",
                TermKind.REWARD,
                _reward_numpy,
                TensorSpec((), dtype),
                inputs=(NamedTensorSpec("state", TensorSpec((2,), dtype)),),
                workspace=(NamedTensorSpec("scratch", TensorSpec((workspace_width,), dtype)),),
                parameters=(ParameterSpec("gain", ParameterKind.FLOAT),),
                numba_item_fn=item_fn,
                implementation_version=version,
            )
        )
    return resolve_term_plan(
        registry,
        tuple(
            TermConfig(label, f"synthetic.reward.{label}.v1", parameters={"gain": 0.25})
            for label in order
        ),
    )


def _materialize(plan, layout: FusedOutputLayout | None = None):
    count = 8
    dtype = plan.input_specs["state"].numpy_dtype
    inputs = {"state": np.arange(count * 2, dtype=dtype).reshape(count, 2)}
    reward = np.empty((count,), dtype=dtype)
    terminated = np.empty((count,), dtype=np.bool_)
    runtime = materialize_numba_plan(
        plan,
        layout
        or FusedOutputLayout(
            rewards=tuple(term.name for term in plan.terms), observations={}, terminations=()
        ),
        num_envs=count,
        inputs=inputs,
        observations={},
        reward=reward,
        terminated=terminated,
    )
    return runtime, inputs


def test_fused_plan_cache_reuses_callable_and_runtime_buffers() -> None:
    clear_numba_plan_cache()
    plan = _plan()
    first, inputs = _materialize(plan)
    second, _ = _materialize(plan)

    assert not first.compile_info.cache_hit
    assert second.compile_info.cache_hit
    assert first.compile_info.cache_key == second.compile_info.cache_key
    assert first._kernel is second._kernel

    reward_id = id(first.reward)
    scratch = np.zeros((numba.get_num_threads(), len(plan.terms)), dtype=np.float64)
    first.execute(reward_multiplier=2.0, log_scratch=scratch)
    expected = inputs["state"].sum(axis=1) * 0.5
    np.testing.assert_allclose(first.reward, expected)

    first.parameters[0] = 0.5
    first.execute(reward_multiplier=2.0, log_scratch=scratch)
    np.testing.assert_allclose(first.reward, expected * 2.0)
    first.set_scale("a", 0.0)
    first.execute(reward_multiplier=2.0, log_scratch=scratch)
    np.testing.assert_array_equal(first.reward, 0.0)
    assert id(first.reward) == reward_id


def test_fused_cache_key_covers_plan_structure_dtype_and_version() -> None:
    clear_numba_plan_cache()
    base, _ = _materialize(_plan(order=("a", "b")))
    variants = (
        _materialize(_plan(order=("b", "a")))[0],
        _materialize(_plan(workspace_width=2, order=("a", "b")))[0],
        _materialize(_plan(dtype=np.float64, order=("a", "b")))[0],
        _materialize(_plan(version=2, order=("a", "b")))[0],
        _materialize(
            _plan(order=("a", "b")),
            FusedOutputLayout(rewards=(), observations={}, terminations=(), preambles=("a", "b")),
        )[0],
    )
    keys = {runtime.compile_info.cache_key for runtime in variants}
    assert base.compile_info.cache_key not in keys
    assert len(keys) == 5
    assert all(not runtime.compile_info.cache_hit for runtime in variants)


def test_fused_materialization_fails_closed_without_numba_item() -> None:
    clear_numba_plan_cache()
    with pytest.raises(TermPlanError, match="no Numba item implementation"):
        _materialize(_plan(item_fn=None))


def test_fused_plan_executes_bool_int_and_typed_tuple_parameters() -> None:
    registry = TermRegistry()
    registry.register(
        TermDefinition(
            "synthetic.reward.typed.v1",
            TermKind.REWARD,
            _typed_parameter_numpy,
            TensorSpec((), np.float32),
            inputs=(NamedTensorSpec("state", TensorSpec((2,), np.float32)),),
            parameters=(
                ParameterSpec("enabled", ParameterKind.BOOLEAN),
                ParameterSpec("offset", ParameterKind.INTEGER),
                ParameterSpec("indices", ParameterKind.INTEGER, tuple_value=True),
            ),
            numba_item_fn=_typed_parameter_item,
        )
    )
    plan = resolve_term_plan(
        registry,
        (
            TermConfig(
                "typed",
                "synthetic.reward.typed.v1",
                parameters={"enabled": True, "offset": 3, "indices": (0, 1)},
            ),
        ),
    )
    runtime, inputs = _materialize(plan)
    scratch = np.zeros((numba.get_num_threads(), 1), dtype=np.float64)

    runtime.execute(reward_multiplier=1.0, log_scratch=scratch)

    np.testing.assert_allclose(runtime.reward, inputs["state"].sum(axis=1) + 3.0)
