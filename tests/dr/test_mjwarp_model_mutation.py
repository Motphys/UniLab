"""Production CUDA semantics and mandatory evidence for mjwarp actuator PD mutation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch
from tests.dr.mjwarp_model_mutation_support import (
    ACTUATOR_NAME,
    ModelMutationRuntime,
    PlanKey,
    ResetBatchBuffers,
    bind_combined_pd_plan,
    bind_model_plan,
    control_batch,
    model_mutation_runtime,
)
from tests.training.device_runtime_harness import forbid_host_roundtrip

from unilab.base.backend import (
    BackendBatchContractError,
    ModelParameterMutationBatch,
    MutationBaseline,
    MutationCommitPhase,
    MutationContractError,
    MutationEntityKind,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationSelectorMode,
    MutationSelectorSpec,
    MutationSpec,
    MutationTargetKind,
    MutationTargetSpec,
    MutationTrigger,
    RowSelection,
)

pytestmark = pytest.mark.slow

_NUM_ENVS = 8
_SELECTED = (1, 5)


def _wait(result: Any) -> None:
    completion = result.diagnostics.completion_event
    assert completion is not None
    completion.handle.event.synchronize()


@dataclass(frozen=True)
class _ModelCase:
    target_key: str
    field_kind: MutationFieldKind
    operation: MutationOperation

    @property
    def parameter_id(self) -> str:
        return f"device_resident-{self.target_key}-{self.operation.value}"

    @property
    def key(self) -> PlanKey:
        return PlanKey(self.target_key, self.operation)


_CASES = tuple(
    _ModelCase(target_key, field_kind, operation)
    for target_key, field_kind in (
        ("actuator.pd_stiffness", MutationFieldKind.STIFFNESS),
        ("actuator.pd_damping", MutationFieldKind.DAMPING),
    )
    for operation in (MutationOperation.SET, MutationOperation.SCALE)
)
_PLAN_KEYS = tuple(case.key for case in _CASES)


@pytest.fixture(scope="module")
def runtime() -> ModelMutationRuntime:
    with model_mutation_runtime(num_envs=_NUM_ENVS, plan_keys=_PLAN_KEYS) as value:
        yield value


def _source_values(case: _ModelCase, runtime: ModelMutationRuntime) -> torch.Tensor:
    if case.operation is MutationOperation.SCALE:
        return torch.tensor((1.35, 1.70), dtype=torch.float32, device=runtime.device)
    default = (
        runtime.default_gain[0, runtime.actuator_id, 0]
        if case.target_key == "actuator.pd_stiffness"
        else -runtime.default_bias[0, runtime.actuator_id, 2]
    )
    return default * torch.tensor((1.25, 1.55), dtype=torch.float32, device=runtime.device)


def _expected_physical_values(
    case: _ModelCase,
    runtime: ModelMutationRuntime,
    source: torch.Tensor,
) -> torch.Tensor:
    if case.operation is MutationOperation.SET:
        return source
    default = (
        runtime.default_gain[0, runtime.actuator_id, 0]
        if case.target_key == "actuator.pd_stiffness"
        else -runtime.default_bias[0, runtime.actuator_id, 2]
    )
    return default * source


def _commit(
    case: _ModelCase,
    runtime: ModelMutationRuntime,
    *,
    source: torch.Tensor | None = None,
) -> tuple[ResetBatchBuffers, Any]:
    plan = runtime.mutation_plans[case.key]
    buffers = ResetBatchBuffers(runtime, plan)
    buffers.active_mask[list(_SELECTED)] = True
    source = _source_values(case, runtime) if source is None else source
    buffers.values["model.value"][list(_SELECTED), 0, 0] = source
    with forbid_host_roundtrip(runtime.backend):
        result = runtime.backend.reset_batch(
            runtime.plan,
            RowSelection.all(runtime.num_envs),
            mutation_batch=buffers.publish(),
        )
    _wait(result)
    return buffers, result


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.parameter_id)
def test_mjwarp_advertised_model_capability_case(
    case: _ModelCase,
    runtime: ModelMutationRuntime,
) -> None:
    """Every advertised case commits exact selected rows through the public batch ABI."""

    runtime.restore_compiled_model_defaults()
    graph_before = runtime.backend.get_device_graph_diagnostics(verify_storage=True)
    plan_owner = runtime.backend._device_mutation_plans[
        runtime.mutation_plans[case.key].fingerprint
    ]
    scratch_before = plan_owner.numeric_buffer_addresses
    producer_addresses = None
    gain_address = int(runtime.gain.data_ptr())
    bias_address = int(runtime.bias.data_ptr())
    source = _source_values(case, runtime)

    buffers, result = _commit(case, runtime, source=source)
    producer_addresses = buffers.numeric_addresses
    expected = _expected_physical_values(case, runtime, source)
    actuator_id = runtime.actuator_id
    selected = list(_SELECTED)
    complement = sorted(set(range(runtime.num_envs)) - set(_SELECTED))
    default_gain = runtime.default_gain[0, actuator_id]
    default_bias = runtime.default_bias[0, actuator_id]

    if case.target_key == "actuator.pd_stiffness":
        torch.testing.assert_close(runtime.gain[selected, actuator_id, 0], expected)
        torch.testing.assert_close(runtime.bias[selected, actuator_id, 1], -expected)
        torch.testing.assert_close(
            runtime.bias[:, actuator_id, 2],
            runtime.default_bias[:, actuator_id, 2].expand(runtime.num_envs),
        )
    else:
        torch.testing.assert_close(runtime.bias[selected, actuator_id, 2], -expected)
        torch.testing.assert_close(
            runtime.gain[:, actuator_id, 0],
            runtime.default_gain[:, actuator_id, 0].expand(runtime.num_envs),
        )
        torch.testing.assert_close(
            runtime.bias[:, actuator_id, 1],
            runtime.default_bias[:, actuator_id, 1].expand(runtime.num_envs),
        )
    torch.testing.assert_close(
        runtime.gain[complement, actuator_id],
        default_gain.expand(len(complement), -1),
    )
    torch.testing.assert_close(
        runtime.bias[complement, actuator_id],
        default_bias.expand(len(complement), -1),
    )

    counters = result.diagnostics.counters
    assert counters.host_to_device_transfers == 0
    assert counters.device_to_host_transfers == 0
    assert counters.global_synchronizations == 0
    assert counters.allocations == 0
    assert counters.instrumentation_complete
    assert buffers.numeric_addresses == producer_addresses
    assert plan_owner.numeric_buffer_addresses == scratch_before
    assert int(runtime.gain.data_ptr()) == gain_address
    assert int(runtime.bias.data_ptr()) == bias_address
    graph_after = runtime.backend.get_device_graph_diagnostics(verify_storage=True)
    assert graph_after.storage_generation == graph_before.storage_generation
    assert graph_after.storage_fingerprint == graph_before.storage_fingerprint


@pytest.mark.parametrize(
    "target_key",
    ("actuator.pd_stiffness", "actuator.pd_damping"),
)
def test_scale_uses_compiled_default_and_never_accumulates(
    target_key: str,
    runtime: ModelMutationRuntime,
) -> None:
    runtime.restore_compiled_model_defaults()
    case = next(
        item
        for item in _CASES
        if item.target_key == target_key and item.operation is MutationOperation.SCALE
    )
    plan = runtime.mutation_plans[case.key]
    buffers = ResetBatchBuffers(runtime, plan)
    buffers.active_mask[list(_SELECTED)] = True
    buffers.values["model.value"][list(_SELECTED), 0, 0] = 1.6
    first = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=buffers.publish(),
    )
    _wait(first)
    slot = (
        runtime.gain[:, runtime.actuator_id, 0]
        if target_key == "actuator.pd_stiffness"
        else runtime.bias[:, runtime.actuator_id, 2]
    )
    first_values = slot.clone()

    second = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=buffers.publish(),
    )
    _wait(second)
    torch.testing.assert_close(slot, first_values)


def test_invalid_model_values_gate_the_entire_reset_row_before_physics(
    runtime: ModelMutationRuntime,
) -> None:
    runtime.restore_compiled_model_defaults()
    case = next(
        item
        for item in _CASES
        if item.target_key == "actuator.pd_stiffness" and item.operation is MutationOperation.SET
    )
    plan = runtime.mutation_plans[case.key]
    buffers = ResetBatchBuffers(runtime, plan)
    invalid_rows = (0, 2, 4)
    buffers.active_mask[list(invalid_rows)] = True
    buffers.values["model.value"][list(invalid_rows), 0, 0] = torch.tensor(
        (-1.0, float("nan"), float("inf")),
        dtype=torch.float32,
        device=runtime.device,
    )
    qpos, _ = runtime.set_uniform_state(target_position=0.37)
    before = runtime.backend.read_state_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
    ).state
    before_position = before.buffer("dof.position").handle.torch().clone()

    result = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=buffers.publish(),
    )
    _wait(result)
    after = result.reset_state.buffer("dof.position").handle.torch()
    torch.testing.assert_close(after, before_position)
    torch.testing.assert_close(
        runtime.gain,
        runtime.default_gain.expand_as(runtime.gain),
    )
    torch.testing.assert_close(
        runtime.bias,
        runtime.default_bias.expand_as(runtime.bias),
    )
    assert qpos[0, runtime.raw_qpos_index] == pytest.approx(0.37)
    owner = runtime.backend._device_mutation_plans[plan.fingerprint]
    assert owner.model_plan is not None
    assert not bool(owner.model_plan.effective_mask[list(invalid_rows)].any())


def test_combined_pd_and_state_plan_rejects_an_invalid_row_atomically(
    runtime: ModelMutationRuntime,
) -> None:
    runtime.restore_compiled_model_defaults()
    runtime.set_uniform_state(target_position=0.12, target_velocity=-0.08)
    plan = bind_combined_pd_plan(runtime, mixed_state=True)
    buffers = ResetBatchBuffers(runtime, plan)
    valid_row, invalid_row = 0, 4
    buffers.active_mask[[valid_row, invalid_row]] = True
    buffers.values["model.stiffness"][[valid_row, invalid_row], 0, 0] = torch.tensor(
        (37.0, 41.0), dtype=torch.float32, device=runtime.device
    )
    buffers.values["model.damping"][[valid_row, invalid_row], 0, 0] = torch.tensor(
        (3.0, float("nan")), dtype=torch.float32, device=runtime.device
    )
    buffers.values["state.position"][[valid_row, invalid_row], 0, 0] = 0.31
    buffers.values["state.velocity"][[valid_row, invalid_row], 0, 0] = -0.27

    result = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=buffers.publish(),
    )
    _wait(result)

    actuator_id = runtime.actuator_id
    torch.testing.assert_close(
        runtime.gain[valid_row, actuator_id, 0],
        torch.tensor(37.0, device=runtime.device),
    )
    torch.testing.assert_close(
        runtime.bias[valid_row, actuator_id, 1],
        torch.tensor(-37.0, device=runtime.device),
    )
    torch.testing.assert_close(
        runtime.bias[valid_row, actuator_id, 2],
        torch.tensor(-3.0, device=runtime.device),
    )
    torch.testing.assert_close(
        runtime.gain[invalid_row, actuator_id],
        runtime.default_gain[0, actuator_id],
    )
    torch.testing.assert_close(
        runtime.bias[invalid_row, actuator_id],
        runtime.default_bias[0, actuator_id],
    )
    position = result.reset_state.buffer("dof.position").handle.torch()
    velocity = result.reset_state.buffer("dof.angular_velocity").handle.torch()
    torch.testing.assert_close(
        position[valid_row, runtime.dof_position_index],
        torch.tensor(0.31, device=runtime.device),
    )
    torch.testing.assert_close(
        velocity[valid_row, runtime.dof_velocity_index],
        torch.tensor(-0.27, device=runtime.device),
    )
    torch.testing.assert_close(
        position[invalid_row, runtime.dof_position_index],
        torch.tensor(0.12, device=runtime.device),
    )
    torch.testing.assert_close(
        velocity[invalid_row, runtime.dof_velocity_index],
        torch.tensor(-0.08, device=runtime.device),
    )


def _unsupported_spec(
    runtime: ModelMutationRuntime,
    *,
    selector: MutationSelectorSpec,
) -> MutationSpec:
    capability = next(
        item
        for item in runtime.backend.get_mutation_capability_manifest(
            runtime.plan.execution_profile
        ).capabilities
        if item.target_key == "actuator.pd_stiffness"
    )
    return MutationSpec(
        term_key=f"unsupported.{selector.semantic_key}",
        target=MutationTargetSpec(
            target_key=capability.target_key,
            target_kind=MutationTargetKind.MODEL_PARAMETER,
            entity_kind=MutationEntityKind.ACTUATOR,
            field_kind=MutationFieldKind.STIFFNESS,
            selector=selector,
        ),
        trigger=MutationTrigger.RESET,
        commit_phase=MutationCommitPhase.RESET,
        operation=MutationOperation.SET,
        baseline=MutationBaseline.DEFAULT,
        persistence=MutationPersistence.EPISODE,
        recompute=MutationRecomputeLevel.NONE,
        value_template=capability.value_template,
    )


def test_actuator_selector_and_layout_fail_closed(runtime: ModelMutationRuntime) -> None:
    regex = MutationSelectorSpec(
        semantic_key="regex-actuator",
        mode=MutationSelectorMode.REGEX,
        expressions=("left_.*",),
    )
    with pytest.raises(MutationContractError, match="only supports exact"):
        runtime.backend.bind_mutation_plan((_unsupported_spec(runtime, selector=regex),))

    unknown = MutationSelectorSpec.exact("missing_actuator")
    with pytest.raises(MutationContractError, match="did not resolve an actuator"):
        runtime.backend.bind_mutation_plan((_unsupported_spec(runtime, selector=unknown),))

    with pytest.raises(MutationContractError, match="must be unique"):
        MutationSelectorSpec(
            semantic_key="duplicate-actuator",
            mode=MutationSelectorMode.EXACT,
            expressions=(ACTUATOR_NAME, ACTUATOR_NAME),
        )

    original = runtime.backend._position_actuator_ids
    runtime.backend._position_actuator_ids = tuple(
        actuator_id for actuator_id in original if actuator_id != runtime.actuator_id
    )
    try:
        with pytest.raises(MutationContractError, match="not a supported native position servo"):
            runtime.backend.bind_mutation_plan(
                (
                    _unsupported_spec(
                        runtime,
                        selector=MutationSelectorSpec.exact(ACTUATOR_NAME),
                    ),
                )
            )
    finally:
        runtime.backend._position_actuator_ids = original


def test_position_servo_binding_excludes_unnamed_or_nonservo_actuators(
    runtime: ModelMutationRuntime,
) -> None:
    backend = runtime.backend
    actuator_id = runtime.actuator_id
    original_names = backend._actuator_names
    unnamed = list(original_names)
    unnamed[actuator_id] = ""
    backend._actuator_names = tuple(unnamed)
    try:
        assert actuator_id not in backend._bind_position_actuator_ids()
    finally:
        backend._actuator_names = original_names

    model = backend._cpu_model
    original_gain_type = int(model.actuator_gaintype[actuator_id])
    model.actuator_gaintype[actuator_id] = int(backend._mujoco.mjtGain.mjGAIN_USER)
    try:
        assert actuator_id not in backend._bind_position_actuator_ids()
    finally:
        model.actuator_gaintype[actuator_id] = original_gain_type


def test_exact_multi_actuator_selector_commits_vectorized_slots(
    runtime: ModelMutationRuntime,
) -> None:
    runtime.restore_compiled_model_defaults()
    second_id = next(
        actuator_id
        for actuator_id in runtime.backend._position_actuator_ids
        if actuator_id != runtime.actuator_id
    )
    second_name = runtime.backend.get_actuator_names()[second_id]
    selector = MutationSelectorSpec(
        semantic_key="two-position-actuators",
        mode=MutationSelectorMode.EXACT,
        expressions=(ACTUATOR_NAME, second_name),
    )
    plan = runtime.backend.bind_mutation_plan((_unsupported_spec(runtime, selector=selector),))
    assert plan.specs[0].target.entity_ids == (runtime.actuator_id, second_id)
    assert plan.specs[0].value_buffer.row_shape == (2, 1)

    buffers = ResetBatchBuffers(runtime, plan)
    buffers.active_mask[list(_SELECTED)] = True
    expected = torch.tensor((37.0, 53.0), dtype=torch.float32, device=runtime.device)
    buffers.values["unsupported.two-position-actuators"][list(_SELECTED), :, 0] = expected
    result = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=buffers.publish(),
    )
    _wait(result)

    actuator_ids = (runtime.actuator_id, second_id)
    selected = list(_SELECTED)
    complement = sorted(set(range(runtime.num_envs)) - set(_SELECTED))
    torch.testing.assert_close(
        runtime.gain[selected][:, actuator_ids, 0],
        expected.expand(len(selected), -1),
    )
    torch.testing.assert_close(
        runtime.bias[selected][:, actuator_ids, 1],
        -expected.expand(len(selected), -1),
    )
    torch.testing.assert_close(
        runtime.gain[complement][:, actuator_ids, 0],
        runtime.default_gain[:, actuator_ids, 0].expand(len(complement), -1),
    )
    torch.testing.assert_close(
        runtime.bias[complement][:, actuator_ids, 1],
        runtime.default_bias[:, actuator_ids, 1].expand(len(complement), -1),
    )


def test_warm_commits_keep_numeric_addresses_and_cuda_allocations_stable(
    runtime: ModelMutationRuntime,
) -> None:
    runtime.restore_compiled_model_defaults()
    key = PlanKey("actuator.pd_stiffness", MutationOperation.SCALE)
    plan = runtime.mutation_plans[key]
    owner = runtime.backend._device_mutation_plans[plan.fingerprint]
    buffers = ResetBatchBuffers(runtime, plan)
    buffers.active_mask[list(_SELECTED)] = True
    buffers.values["model.value"].fill_(1.1)

    warmup = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=buffers.publish(),
    )
    _wait(warmup)
    addresses = (
        *buffers.numeric_addresses,
        *owner.numeric_buffer_addresses,
        int(runtime.gain.data_ptr()),
        int(runtime.bias.data_ptr()),
    )
    allocation_count = torch.cuda.memory_stats(runtime.device)["allocation.all.allocated"]

    for _ in range(8):
        with forbid_host_roundtrip(runtime.backend):
            result = runtime.backend.reset_batch(
                runtime.plan,
                RowSelection.all(runtime.num_envs),
                mutation_batch=buffers.publish(),
            )
        _wait(result)
        assert result.diagnostics.counters.allocations == 0

    assert torch.cuda.memory_stats(runtime.device)["allocation.all.allocated"] == allocation_count
    assert addresses == (
        *buffers.numeric_addresses,
        *owner.numeric_buffer_addresses,
        int(runtime.gain.data_ptr()),
        int(runtime.bias.data_ptr()),
    )
    receipt = runtime.backend._model_materialization_receipt
    assert receipt is not None
    assert tuple(field.field_name for field in receipt.fields) == (
        "actuator_biasprm",
        "actuator_gainprm",
    )


def test_model_plan_binding_after_first_runtime_barrier_fails_without_recapture() -> None:
    with model_mutation_runtime(num_envs=4, plan_keys=()) as late_runtime:
        control = torch.zeros(
            (late_runtime.num_envs, late_runtime.backend.num_actuators),
            dtype=torch.float32,
            device=late_runtime.device,
        )
        stepped = late_runtime.backend.step_batch(
            late_runtime.plan,
            control_batch(late_runtime, control, owner="late-model-bind"),
            nsteps=1,
        )
        _wait(stepped)
        before = late_runtime.backend.get_device_graph_diagnostics(verify_storage=True)

        with pytest.raises(BackendBatchContractError, match="before the first runtime"):
            bind_model_plan(
                late_runtime,
                PlanKey("actuator.pd_stiffness", MutationOperation.SET),
            )

        after = late_runtime.backend.get_device_graph_diagnostics(verify_storage=True)
        assert late_runtime.backend._model_materialization_receipt is None
        assert late_runtime.backend._expanded_model_fields == frozenset()
        assert after.storage_verification_count == before.storage_verification_count + 1
        assert (
            replace(
                after,
                storage_verification_count=before.storage_verification_count,
            )
            == before
        )


def test_incomplete_model_envelope_fails_before_graph_launch_or_model_write(
    runtime: ModelMutationRuntime,
) -> None:
    runtime.restore_compiled_model_defaults()
    plan = bind_combined_pd_plan(runtime, mixed_state=False)
    buffers = ResetBatchBuffers(runtime, plan)
    buffers.active_mask[list(_SELECTED)] = True
    buffers.values["model.stiffness"][list(_SELECTED), 0, 0] = 37.0
    published = buffers.publish()
    incomplete_mutation = replace(
        published.mutation,
        model=ModelParameterMutationBatch(published.mutation.model.values[:-1]),
    )
    incomplete = replace(published, mutation=incomplete_mutation)
    gain_before = runtime.gain.clone()
    bias_before = runtime.bias.clone()
    graph_before = runtime.backend.get_device_graph_diagnostics(verify_storage=True)

    with pytest.raises(BackendBatchContractError, match="supply every bound Model field once"):
        runtime.backend.reset_batch(
            runtime.plan,
            RowSelection.all(runtime.num_envs),
            mutation_batch=incomplete,
        )

    graph_after = runtime.backend.get_device_graph_diagnostics(verify_storage=True)
    torch.testing.assert_close(runtime.gain, gain_before)
    torch.testing.assert_close(runtime.bias, bias_before)
    assert graph_after.storage_verification_count == graph_before.storage_verification_count + 1
    assert (
        replace(
            graph_after,
            storage_verification_count=graph_before.storage_verification_count,
        )
        == graph_before
    )
