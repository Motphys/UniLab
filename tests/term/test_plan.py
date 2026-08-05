from typing import cast

import numpy as np
import pytest

from unilab.term import (
    NamedTensorSpec,
    NumpyTermContext,
    ParameterKind,
    ParameterSpec,
    TensorSpec,
    TermBindingError,
    TermConfig,
    TermDefinition,
    TermKind,
    TermPlanError,
    TermRegistrationError,
    TermRegistry,
    resolve_term_plan,
)

STATE = NamedTensorSpec("state", TensorSpec((2,), np.float32))
SCRATCH = NamedTensorSpec("scratch", TensorSpec((), np.float32))


def _registry(calls: list[str] | None = None) -> TermRegistry:
    registry = TermRegistry()

    def reward(context: NumpyTermContext) -> None:
        calls is not None and calls.append("reward")
        np.sum(context.inputs["state"], axis=1, out=context.workspace["scratch"])
        np.multiply(
            context.workspace["scratch"],
            cast(float, context.parameters["gain"]),
            out=context.output,
        )

    def item(state, gain, output, scratch, index) -> None:
        scratch[index] = state[index, 0] + state[index, 1]
        output[index] = scratch[index] * gain

    def observation(context: NumpyTermContext) -> None:
        calls is not None and calls.append("observation")
        np.add(
            context.inputs["state"],
            cast(tuple[float, ...], context.parameters["offsets"]),
            out=context.output,
        )

    def termination(context: NumpyTermContext) -> None:
        calls is not None and calls.append("termination")
        np.greater(
            context.inputs["state"][:, 0],
            cast(float, context.parameters["threshold"]),
            out=context.output,
        )

    registry.register(
        TermDefinition(
            "synthetic.reward.sum.v1",
            TermKind.REWARD,
            reward,
            TensorSpec((), np.float32),
            inputs=(STATE,),
            workspace=(SCRATCH,),
            parameters=(ParameterSpec("gain", ParameterKind.FLOAT),),
            numba_item_fn=item,
        )
    )
    registry.register(
        TermDefinition(
            "synthetic.observation.offset.v1",
            TermKind.OBSERVATION,
            observation,
            TensorSpec((2,), np.float32),
            inputs=(STATE,),
            workspace=(SCRATCH,),
            parameters=(
                ParameterSpec("offsets", ParameterKind.FLOAT, tuple_value=True, default=(0.0, 0.0)),
            ),
        )
    )
    registry.register(
        TermDefinition(
            "synthetic.termination.threshold.v1",
            TermKind.TERMINATION,
            termination,
            TensorSpec((), np.bool_),
            inputs=(STATE,),
            parameters=(ParameterSpec("threshold", ParameterKind.FLOAT, default=0.0),),
        )
    )
    return registry


def _configs() -> tuple[TermConfig, ...]:
    return (
        TermConfig(
            "actor_offset",
            "synthetic.observation.offset.v1",
            parameters={"offsets": [0.5, -0.5]},
        ),
        TermConfig(
            "tracking_reward",
            "synthetic.reward.sum.v1",
            scale=2.0,
            parameters={"gain": 0.25},
        ),
        TermConfig("fell", "synthetic.termination.threshold.v1", parameters={"threshold": 2.0}),
    )


def test_plan_executes_in_config_order_and_reuses_buffers() -> None:
    calls: list[str] = []
    state = np.array([[1.0, 3.0], [4.0, -2.0]], dtype=np.float32)
    runtime = resolve_term_plan(_registry(calls), _configs()).materialize(
        num_envs=2, inputs={"state": state}
    )
    outputs = runtime.execute()
    ids = [id(value) for value in (*outputs.values(), *runtime.workspace.values())]

    assert calls == ["observation", "reward", "termination"]
    assert tuple(outputs) == ("actor_offset", "tracking_reward", "fell")
    np.testing.assert_array_equal(outputs["actor_offset"], [[1.5, 2.5], [4.5, -2.5]])
    np.testing.assert_array_equal(outputs["tracking_reward"], [2.0, 1.0])
    np.testing.assert_array_equal(outputs["fell"], [False, True])
    definition = _registry().resolve("synthetic.reward.sum.v1")
    item_output, item_scratch = np.empty(2, np.float32), np.empty(2, np.float32)
    assert definition.numba_item_fn is not None
    for index in range(2):
        definition.numba_item_fn(state, 0.25, item_output, item_scratch, index)
    np.testing.assert_array_equal(item_output, outputs["tracking_reward"] / 2.0)

    calls.clear()
    runtime.set_scale("tracking_reward", 0.0)
    state += 1.0
    assert runtime.execute() is outputs
    assert calls == ["observation", "termination"]
    assert [id(value) for value in (*outputs.values(), *runtime.workspace.values())] == ids
    np.testing.assert_array_equal(outputs["tracking_reward"], 0.0)


@pytest.mark.parametrize(
    ("configs", "message"),
    [
        ((TermConfig("x", "synthetic.reward.unknown.v1"),), "unknown term key"),
        ((TermConfig("x", "synthetic.reward.sum.v1"),), "missing parameter"),
        (
            (TermConfig("x", "synthetic.reward.sum.v1", parameters={"gain": "fast"}),),
            "must be float",
        ),
        ((TermConfig("x", "synthetic.termination.threshold.v1", scale=0.5),), "0.0 or 1.0"),
    ],
)
def test_plan_rejects_invalid_config(configs, message: str) -> None:
    with pytest.raises(TermPlanError, match=message):
        resolve_term_plan(_registry(), configs)


def test_plan_rejects_conflicting_workspace_specs() -> None:
    registry = _registry()

    registry.register(
        TermDefinition(
            "synthetic.reward.conflict.v1",
            TermKind.REWARD,
            lambda context: context.output.fill(0),
            TensorSpec((), np.float32),
            inputs=(STATE,),
            workspace=(NamedTensorSpec("scratch", TensorSpec((2,), np.float64)),),
        )
    )
    configs = (
        TermConfig("obs", "synthetic.observation.offset.v1"),
        TermConfig("conflict", "synthetic.reward.conflict.v1"),
    )
    with pytest.raises(TermPlanError, match="workspace 'scratch' has conflicting specs"):
        resolve_term_plan(registry, configs)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"state": np.zeros((3, 2), np.float32)}, "shape mismatch"),
        ({"state": np.zeros((2, 2), np.float64)}, "dtype mismatch"),
    ],
)
def test_materialize_rejects_invalid_bound_views(inputs, message: str) -> None:
    plan = resolve_term_plan(_registry(), _configs())
    with pytest.raises(TermBindingError, match=message):
        plan.materialize(num_envs=2, inputs=inputs)


def test_registry_rejects_duplicate_key() -> None:
    registry = _registry()
    with pytest.raises(TermRegistrationError, match="already registered"):
        registry.register(registry.resolve("synthetic.reward.sum.v1"))
