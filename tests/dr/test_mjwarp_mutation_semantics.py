"""Cross-category reset semantics for mjwarp typed mutation plans."""

from __future__ import annotations

import pytest
import torch
from tests._support.mjwarp_model_mutation import (
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
_ROWS_SCALE = (0, 3)
_ROWS_SET = (2, 6)


@pytest.fixture(scope="module")
def runtime() -> ModelMutationRuntime:
    keys = tuple(
        PlanKey(target_key, operation, mixed_state=True)
        for target_key in ("actuator.pd_stiffness", "actuator.pd_damping")
        for operation in (MutationOperation.SET, MutationOperation.SCALE)
    )
    with model_mutation_runtime(num_envs=_NUM_ENVS, plan_keys=keys) as value:
        yield value


@pytest.mark.parametrize(
    "target_key",
    ("actuator.pd_stiffness", "actuator.pd_damping"),
)
def test_operations_baselines_rows_and_persistence(
    target_key: str,
    runtime: ModelMutationRuntime,
) -> None:
    """Mixed state/Model commits preserve default semantics and episode values."""

    runtime.restore_compiled_model_defaults()
    qpos, _ = runtime.set_uniform_state(target_position=0.11, target_velocity=0.0)
    scale_key = PlanKey(target_key, MutationOperation.SCALE, mixed_state=True)
    scale_plan = runtime.mutation_plans[scale_key]
    scale = ResetBatchBuffers(runtime, scale_plan)
    scale.active_mask[list(_ROWS_SCALE)] = True
    scale.values["model.value"][list(_ROWS_SCALE), 0, 0] = 1.5
    scale.values["state.position"][list(_ROWS_SCALE), 0, 0] = 0.23
    scale.values["state.velocity"][list(_ROWS_SCALE), 0, 0] = -0.37

    first = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=scale.publish(),
    )
    wait_result(first)
    first_position = state_tensor(first.reset_state, "dof.position")
    first_velocity = state_tensor(first.reset_state, "dof.angular_velocity")
    torch.testing.assert_close(
        first_position[list(_ROWS_SCALE), runtime.dof_position_index],
        torch.full((len(_ROWS_SCALE),), 0.23, device=runtime.device),
    )
    torch.testing.assert_close(
        first_velocity[list(_ROWS_SCALE), runtime.dof_velocity_index],
        torch.full((len(_ROWS_SCALE),), -0.37, device=runtime.device),
    )

    actuator_id = runtime.actuator_id
    default = (
        runtime.default_gain[0, actuator_id, 0]
        if target_key == "actuator.pd_stiffness"
        else -runtime.default_bias[0, actuator_id, 2]
    )

    def physical() -> torch.Tensor:
        if target_key == "actuator.pd_stiffness":
            return runtime.gain[:, actuator_id, 0]
        return -runtime.bias[:, actuator_id, 2]

    torch.testing.assert_close(
        physical()[list(_ROWS_SCALE)],
        default.expand(len(_ROWS_SCALE)) * 1.5,
    )

    control = runtime.position_hold_control(qpos)
    stepped = runtime.backend.step_batch(
        runtime.plan,
        control_batch(runtime, control, owner=f"semantic-{target_key}"),
        nsteps=1,
    )
    wait_result(stepped)
    torch.testing.assert_close(
        physical()[list(_ROWS_SCALE)],
        default.expand(len(_ROWS_SCALE)) * 1.5,
    )

    set_key = PlanKey(target_key, MutationOperation.SET, mixed_state=True)
    set_plan = runtime.mutation_plans[set_key]
    absolute = default * 0.8
    set_buffers = ResetBatchBuffers(runtime, set_plan)
    set_buffers.active_mask[list(_ROWS_SET)] = True
    set_buffers.values["model.value"][list(_ROWS_SET), 0, 0] = absolute
    set_buffers.values["state.position"][list(_ROWS_SET), 0, 0] = -0.19
    set_buffers.values["state.velocity"][list(_ROWS_SET), 0, 0] = 0.41
    second = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=set_buffers.publish(),
    )
    wait_result(second)

    torch.testing.assert_close(
        physical()[list(_ROWS_SCALE)],
        default.expand(len(_ROWS_SCALE)) * 1.5,
    )
    torch.testing.assert_close(
        physical()[list(_ROWS_SET)],
        absolute.expand(len(_ROWS_SET)),
    )
    untouched = sorted(set(range(runtime.num_envs)) - set(_ROWS_SCALE) - set(_ROWS_SET))
    torch.testing.assert_close(physical()[untouched], default.expand(len(untouched)))
    second_position = state_tensor(second.reset_state, "dof.position")
    second_velocity = state_tensor(second.reset_state, "dof.angular_velocity")
    torch.testing.assert_close(
        second_position[list(_ROWS_SET), runtime.dof_position_index],
        torch.full((len(_ROWS_SET),), -0.19, device=runtime.device),
    )
    torch.testing.assert_close(
        second_velocity[list(_ROWS_SET), runtime.dof_velocity_index],
        torch.full((len(_ROWS_SET),), 0.41, device=runtime.device),
    )
