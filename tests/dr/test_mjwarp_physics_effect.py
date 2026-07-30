"""Next-step physics-effect oracles for advertised mjwarp Model mutations."""

from __future__ import annotations

import pytest
import torch
from tests.dr.mjwarp_model_mutation_support import (
    ModelMutationRuntime,
    PlanKey,
    ResetBatchBuffers,
    control_batch,
    model_mutation_runtime,
    state_tensor,
    wait_result,
)

from unilab.base.backend import MutationOperation, RowSelection

pytestmark = pytest.mark.slow

_NUM_ENVS = 8
_SELECTED = (0, 2)
_PAIRED_CONTROLS = (1, 3)
_UNTOUCHED_LEFT = (4, 6)
_UNTOUCHED_RIGHT = (5, 7)
_CUDA_PAIR_ATOL = 5.0e-6


@pytest.fixture(scope="module")
def runtime() -> ModelMutationRuntime:
    keys = tuple(
        PlanKey(target_key, MutationOperation.SET, mixed_state=True)
        for target_key in ("actuator.pd_stiffness", "actuator.pd_damping")
    )
    with model_mutation_runtime(num_envs=_NUM_ENVS, plan_keys=keys) as value:
        yield value


@pytest.mark.parametrize(
    "target_key",
    ("actuator.pd_stiffness", "actuator.pd_damping"),
)
def test_each_supported_mutation_has_next_step_effect(
    target_key: str,
    runtime: ModelMutationRuntime,
) -> None:
    """Paired worlds isolate stiffness and damping effects above the registered threshold."""

    runtime.restore_compiled_model_defaults()
    target_position = 0.15
    target_velocity = 0.0 if target_key == "actuator.pd_stiffness" else 1.4
    qpos, _ = runtime.set_uniform_state(
        target_position=target_position,
        target_velocity=target_velocity,
    )
    key = PlanKey(target_key, MutationOperation.SET, mixed_state=True)
    plan = runtime.mutation_plans[key]
    buffers = ResetBatchBuffers(runtime, plan)
    buffers.active_mask[list(_SELECTED)] = True
    actuator_id = runtime.actuator_id
    default = (
        runtime.default_gain[0, actuator_id, 0]
        if target_key == "actuator.pd_stiffness"
        else -runtime.default_bias[0, actuator_id, 2]
    )
    buffers.values["model.value"][list(_SELECTED), 0, 0] = default * 2.0
    buffers.values["state.position"][list(_SELECTED), 0, 0] = target_position
    buffers.values["state.velocity"][list(_SELECTED), 0, 0] = target_velocity

    reset = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=buffers.publish(),
    )
    wait_result(reset)
    immediate_position = state_tensor(reset.reset_state, "dof.position")
    immediate_velocity = state_tensor(reset.reset_state, "dof.angular_velocity")
    torch.testing.assert_close(
        immediate_position[list(_SELECTED), runtime.dof_position_index],
        immediate_position[list(_PAIRED_CONTROLS), runtime.dof_position_index],
    )
    torch.testing.assert_close(
        immediate_velocity[list(_SELECTED), runtime.dof_velocity_index],
        immediate_velocity[list(_PAIRED_CONTROLS), runtime.dof_velocity_index],
    )

    control = runtime.position_hold_control(qpos)
    if target_key == "actuator.pd_stiffness":
        control[:, actuator_id] += 0.25
    terminal = runtime.backend.step_batch(
        runtime.plan,
        control_batch(runtime, control, owner=f"physics-effect-{target_key}"),
        nsteps=1,
    )
    wait_result(terminal)
    velocity = state_tensor(terminal.terminal_state, "dof.angular_velocity")
    selected = velocity[list(_SELECTED), runtime.dof_velocity_index]
    paired = velocity[list(_PAIRED_CONTROLS), runtime.dof_velocity_index]
    assert bool(torch.isfinite(selected).all())
    assert bool(torch.isfinite(paired).all())
    assert float(torch.max(torch.abs(selected - paired))) > 1.0e-7
    torch.testing.assert_close(
        velocity[list(_UNTOUCHED_LEFT), runtime.dof_velocity_index],
        velocity[list(_UNTOUCHED_RIGHT), runtime.dof_velocity_index],
        atol=_CUDA_PAIR_ATOL,
        rtol=0.0,
    )

    complement = sorted(set(range(runtime.num_envs)) - set(_SELECTED))
    if target_key == "actuator.pd_stiffness":
        torch.testing.assert_close(
            runtime.gain[complement, actuator_id, 0],
            default.expand(len(complement)),
        )
        torch.testing.assert_close(
            runtime.bias[:, actuator_id, 1],
            -runtime.gain[:, actuator_id, 0],
        )
    else:
        torch.testing.assert_close(
            -runtime.bias[complement, actuator_id, 2],
            default.expand(len(complement)),
        )
