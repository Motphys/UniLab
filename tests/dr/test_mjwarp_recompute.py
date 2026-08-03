"""Model recompute lattice and production CUDA evidence for ``mjwarp``."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tests.dr.mjwarp_model_mutation_support import (
    ModelMutationRuntime,
    PlanKey,
    ResetBatchBuffers,
    bind_combined_model_plan,
    bind_model_plan,
    bind_state_reset_plan,
    control_batch,
    model_mutation_runtime,
    model_mutation_spec,
    state_tensor,
    wait_result,
)
from tests.training.device_runtime_harness import forbid_host_roundtrip

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import (
    BackendBatchContractError,
    ExecutionProfile,
    MutationCapabilityManifest,
    MutationContractError,
    MutationOperation,
    MutationRecomputeLevel,
    MutationSelectorMode,
    MutationSelectorSpec,
    MutationTargetKind,
    RowSelection,
)
from unilab.base.backend.mjwarp.armature_recompute import (
    MjwarpArmatureRecomputeWorkspace,
)
from unilab.base.backend.mjwarp.materialization import (
    MjwarpModelFieldRole,
    MjwarpModelMaterializationContractError,
)
from unilab.base.backend.mjwarp.recompute import (
    MjwarpModelRecomputeKind,
    MjwarpModelRecomputeRuntime,
    MjwarpModelRecomputeWorkspace,
    join_model_recompute,
    model_recompute_kind,
    recompute_derived_fields,
)


@pytest.mark.parametrize(
    ("level", "expected"),
    (
        (MutationRecomputeLevel.NONE, MjwarpModelRecomputeKind.NONE),
        (MutationRecomputeLevel.KINEMATICS, MjwarpModelRecomputeKind.SET_CONST_FIXED),
        (MutationRecomputeLevel.DYNAMICS, MjwarpModelRecomputeKind.SET_CONST_0),
        (MutationRecomputeLevel.FULL, MjwarpModelRecomputeKind.SET_CONST),
    ),
)
def test_model_semantic_maps_to_backend_recompute(
    level: MutationRecomputeLevel,
    expected: MjwarpModelRecomputeKind,
) -> None:
    assert model_recompute_kind(MutationTargetKind.MODEL_PARAMETER, level) is expected


@pytest.mark.parametrize(
    "target_kind",
    (
        MutationTargetKind.SIMULATION_STATE,
        MutationTargetKind.EXTERNAL_WRENCH,
        MutationTargetKind.TASK_STATE,
    ),
)
def test_non_model_semantic_never_enters_model_recompute(
    target_kind: MutationTargetKind,
) -> None:
    assert (
        model_recompute_kind(target_kind, MutationRecomputeLevel.KINEMATICS)
        is MjwarpModelRecomputeKind.NONE
    )


@pytest.mark.parametrize(
    ("requirements", "expected"),
    (
        ((), MjwarpModelRecomputeKind.NONE),
        (
            (MjwarpModelRecomputeKind.SET_CONST_FIXED,),
            MjwarpModelRecomputeKind.SET_CONST_FIXED,
        ),
        ((MjwarpModelRecomputeKind.SET_CONST_0,), MjwarpModelRecomputeKind.SET_CONST_0),
        (
            (
                MjwarpModelRecomputeKind.SET_CONST_FIXED,
                MjwarpModelRecomputeKind.SET_CONST_0,
            ),
            MjwarpModelRecomputeKind.SET_CONST,
        ),
        (
            (
                MjwarpModelRecomputeKind.SET_CONST_0,
                MjwarpModelRecomputeKind.SET_CONST_FIXED,
            ),
            MjwarpModelRecomputeKind.SET_CONST,
        ),
        (
            (
                MjwarpModelRecomputeKind.SET_CONST_FIXED,
                MjwarpModelRecomputeKind.SET_CONST,
            ),
            MjwarpModelRecomputeKind.SET_CONST,
        ),
    ),
)
def test_recompute_join_is_a_lattice_not_enum_max(
    requirements: tuple[MjwarpModelRecomputeKind, ...],
    expected: MjwarpModelRecomputeKind,
) -> None:
    assert join_model_recompute(requirements) is expected


def test_recompute_derived_fields_match_backend_operations() -> None:
    assert recompute_derived_fields(MjwarpModelRecomputeKind.NONE) == ()
    assert recompute_derived_fields(MjwarpModelRecomputeKind.SET_CONST_FIXED) == (
        "body_subtreemass",
    )
    assert recompute_derived_fields(MjwarpModelRecomputeKind.SET_CONST_0) == (
        "actuator_acc0",
        "body_invweight0",
        "dof_invweight0",
        "tendon_invweight0",
        "tendon_length0",
    )


def test_armature_specialization_support_matrix_is_fail_closed() -> None:
    mujoco = SimpleNamespace(
        mjMINVAL=1.0e-15,
        mjtBias=SimpleNamespace(mjBIAS_AFFINE=1),
    )
    gain = np.zeros((1, 10), dtype=np.float64)
    bias = np.zeros((1, 10), dtype=np.float64)
    model = SimpleNamespace(
        nv=35,
        actuator_gainprm=gain,
        actuator_biasprm=bias,
        actuator_biastype=np.asarray([1], dtype=np.int32),
    )

    assert MjwarpArmatureRecomputeWorkspace.supports(
        ("dof_armature",), "set_const_0", model, mujoco
    )
    assert MjwarpArmatureRecomputeWorkspace.supports(
        ("body_gravcomp", "dof_armature"), "set_const", model, mujoco
    )
    assert not MjwarpArmatureRecomputeWorkspace.supports(
        ("body_gravcomp",), "set_const_fixed", model, mujoco
    )

    model.nv = 65
    assert not MjwarpArmatureRecomputeWorkspace.supports(
        ("dof_armature",), "set_const_0", model, mujoco
    )
    model.nv = 35
    gain[0, 0] = 10.0
    bias[0, 1] = -10.0
    bias[0, 2] = 0.9
    assert not MjwarpArmatureRecomputeWorkspace.supports(
        ("dof_armature",), "set_const_0", model, mujoco
    )


_NUM_ENVS = 8
_SELECTED = (1, 5)
_PAIRED_CONTROLS = (0, 4)
_SHARPA_MODEL = str(ASSETS_ROOT_PATH / "robots" / "sharpa_wave" / "scene.xml")


@dataclass(frozen=True)
class _RecomputeCase:
    target_key: str
    operation: MutationOperation
    runtime_key: str
    direct_field: str
    kind: MjwarpModelRecomputeKind

    @property
    def parameter_id(self) -> str:
        return f"device_resident-{self.target_key}-{self.operation.value}"

    @property
    def plan_key(self) -> PlanKey:
        return PlanKey(self.target_key, self.operation)


_RECOMPUTE_CASES = tuple(
    _RecomputeCase(target_key, operation, runtime_key, direct_field, kind)
    for target_key, runtime_key, direct_field, kind in (
        (
            "joint.armature",
            "armature",
            "dof_armature",
            MjwarpModelRecomputeKind.SET_CONST_0,
        ),
        (
            "body.gravity_compensation",
            "gravcomp",
            "body_gravcomp",
            MjwarpModelRecomputeKind.SET_CONST_FIXED,
        ),
    )
    for operation in (MutationOperation.SET, MutationOperation.SCALE)
)


@pytest.fixture(scope="module")
def recompute_runtimes() -> Iterator[dict[str, ModelMutationRuntime]]:
    armature_keys = tuple(
        case.plan_key for case in _RECOMPUTE_CASES if case.runtime_key == "armature"
    )
    gravcomp_keys = tuple(
        case.plan_key for case in _RECOMPUTE_CASES if case.runtime_key == "gravcomp"
    )
    with model_mutation_runtime(
        num_envs=_NUM_ENVS,
        plan_keys=armature_keys,
    ) as armature_runtime:
        with model_mutation_runtime(
            num_envs=_NUM_ENVS,
            plan_keys=gravcomp_keys,
            model_file=_SHARPA_MODEL,
            base_name="right_hand_C_MC",
            body_name="right_index_PP",
            joint_name="right_index_MCP_AA",
            actuator_name="right_index_MCP_AA_ctrl",
            keyframe_name="home",
        ) as gravcomp_runtime:
            yield {
                "armature": armature_runtime,
                "gravcomp": gravcomp_runtime,
            }


def _raw_entity_id(case: _RecomputeCase, runtime: ModelMutationRuntime) -> int:
    if case.target_key == "joint.armature":
        return runtime.raw_qvel_index
    return runtime.body_id


def _case_values(
    case: _RecomputeCase,
    runtime: ModelMutationRuntime,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_id = _raw_entity_id(case, runtime)
    default = runtime.model_default(case.direct_field)[0, raw_id]
    if case.operation is MutationOperation.SCALE:
        source = torch.tensor(1.75, dtype=torch.float32, device=runtime.device)
        return source, default * source
    if case.target_key == "joint.armature":
        source = default + 0.05
    else:
        source = default * 0.25
    return source, source


def _reset_with_buffers(
    runtime: ModelMutationRuntime,
    buffers: ResetBatchBuffers,
) -> Any:
    result = runtime.backend.reset_batch(
        runtime.plan,
        RowSelection.all(runtime.num_envs),
        mutation_batch=buffers.publish(),
    )
    wait_result(result)
    return result


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    _RECOMPUTE_CASES,
    ids=lambda case: case.parameter_id,
)
def test_mjwarp_advertised_recompute_capability_case(
    case: _RecomputeCase,
    recompute_runtimes: dict[str, ModelMutationRuntime],
) -> None:
    """Each advertised case proves direct commit, derived storage, and graph use."""

    runtime = recompute_runtimes[case.runtime_key]
    runtime.restore_compiled_model_defaults()
    plan = runtime.mutation_plans[case.plan_key]
    raw_id = _raw_entity_id(case, runtime)

    baseline_buffers = ResetBatchBuffers(runtime, plan)
    _reset_with_buffers(runtime, baseline_buffers)
    derived_before = {
        field_name: runtime.model_field(field_name).clone()
        for field_name in recompute_derived_fields(case.kind)
    }
    derived_addresses = tuple(
        int(runtime.model_field(field_name).data_ptr()) for field_name in derived_before
    )
    direct_default = runtime.model_default(case.direct_field)[0, raw_id]
    diagnostics_before = runtime.backend.get_model_recompute_diagnostics(plan)
    assert diagnostics_before is not None

    source, expected = _case_values(case, runtime)
    buffers = ResetBatchBuffers(runtime, plan)
    buffers.active_mask[list(_SELECTED)] = True
    buffers.values["model.value"][list(_SELECTED), 0, 0] = source
    addresses_before = (
        *buffers.numeric_addresses,
        *runtime.backend._device_mutation_plans[plan.fingerprint].numeric_buffer_addresses,
        int(runtime.model_field(case.direct_field).data_ptr()),
        *derived_addresses,
    )

    with forbid_host_roundtrip(runtime.backend):
        result = _reset_with_buffers(runtime, buffers)
    assert result.diagnostics.counters.host_to_device_transfers == 0
    assert result.diagnostics.counters.device_to_host_transfers == 0
    assert result.diagnostics.counters.global_synchronizations == 0
    assert result.diagnostics.counters.allocations == 0

    direct = runtime.model_field(case.direct_field)
    complement = sorted(set(range(runtime.num_envs)).difference(_SELECTED))
    torch.testing.assert_close(direct[list(_SELECTED), raw_id], expected.expand(len(_SELECTED)))
    torch.testing.assert_close(
        direct[complement, raw_id],
        direct_default.expand(len(complement)),
    )
    for field_name, before in derived_before.items():
        current = runtime.model_field(field_name)
        assert bool(torch.isfinite(current).all())
        torch.testing.assert_close(current[complement], before[complement])
    if case.target_key == "joint.armature":
        after_invweight = runtime.model_field("dof_invweight0")
        assert not torch.allclose(
            after_invweight[list(_SELECTED), raw_id],
            derived_before["dof_invweight0"][list(_SELECTED), raw_id],
        )

    diagnostics_after = runtime.backend.get_model_recompute_diagnostics(plan)
    assert diagnostics_after is not None
    assert diagnostics_after.kind is case.kind
    assert diagnostics_after.capture_count == 1
    assert diagnostics_after.launch_count == diagnostics_before.launch_count + 1
    assert not diagnostics_after.state_forward_required
    assert addresses_before == (
        *buffers.numeric_addresses,
        *runtime.backend._device_mutation_plans[plan.fingerprint].numeric_buffer_addresses,
        int(runtime.model_field(case.direct_field).data_ptr()),
        *(int(runtime.model_field(name).data_ptr()) for name in derived_before),
    )

    if case.operation is MutationOperation.SCALE:
        _reset_with_buffers(runtime, buffers)
        torch.testing.assert_close(
            direct[list(_SELECTED), raw_id],
            expected.expand(len(_SELECTED)),
        )


@pytest.mark.slow
def test_specialized_armature_recompute_matches_cpu_mujoco_per_world() -> None:
    selected = (0, 2, 5)
    key = PlanKey("joint.armature", MutationOperation.SET)
    with model_mutation_runtime(num_envs=6, plan_keys=(key,)) as runtime:
        runtime.restore_compiled_model_defaults()
        plan = runtime.mutation_plans[key]
        owner = runtime.backend._device_mutation_plans[plan.fingerprint]
        assert owner.recompute_runtime is not None
        assert isinstance(
            owner.recompute_runtime.workspace,
            MjwarpArmatureRecomputeWorkspace,
        )

        derived_names = (
            "dof_invweight0",
            "body_invweight0",
            "actuator_acc0",
            "stat.meaninertia",
        )
        derived_before = {name: runtime.model_field(name).clone() for name in derived_names}
        buffers = ResetBatchBuffers(runtime, plan)
        buffers.active_mask[list(selected)] = True
        buffers.values["model.value"][list(selected), 0, 0] = torch.tensor(
            (0.02, 0.08, 0.3),
            dtype=torch.float32,
            device=runtime.device,
        )
        _reset_with_buffers(runtime, buffers)

        complement = sorted(set(range(runtime.num_envs)).difference(selected))
        for name, before in derived_before.items():
            assert torch.equal(runtime.model_field(name)[complement], before[complement])

        armature = runtime.model_field("dof_armature").detach().cpu().numpy()
        actual = {name: runtime.model_field(name).detach().cpu() for name in derived_names}
        for world_id in range(runtime.num_envs):
            reference_model = copy.deepcopy(runtime.backend._cpu_model)
            reference_model.dof_armature[:] = armature[world_id]
            reference_data = runtime.backend._mujoco.MjData(reference_model)
            runtime.backend._mujoco.mj_setConst(reference_model, reference_data)
            expected = {
                "dof_invweight0": reference_model.dof_invweight0,
                "body_invweight0": reference_model.body_invweight0,
                "actuator_acc0": reference_model.actuator_acc0,
                "stat.meaninertia": reference_model.stat.meaninertia,
            }
            for name in derived_names:
                torch.testing.assert_close(
                    actual[name][world_id],
                    torch.as_tensor(expected[name], dtype=torch.float32),
                    rtol=5.0e-4,
                    atol=5.0e-4,
                )


@pytest.mark.slow
def test_recompute_diagnostics_distinguish_model_state_and_mixed_plans() -> None:
    with model_mutation_runtime(num_envs=4, plan_keys=()) as runtime:
        mixed = bind_combined_model_plan(
            runtime,
            targets=(
                ("joint.armature", MutationOperation.SET, "model.armature"),
                (
                    "body.gravity_compensation",
                    MutationOperation.SET,
                    "model.gravcomp",
                ),
            ),
            mixed_state=True,
        )
        mixed_diagnostics = runtime.backend.get_model_recompute_diagnostics(mixed)
        assert mixed_diagnostics is not None
        assert mixed_diagnostics.kind is MjwarpModelRecomputeKind.SET_CONST
        assert mixed_diagnostics.state_forward_required
        assert mixed_diagnostics.launch_count == 0

    with model_mutation_runtime(num_envs=4, plan_keys=()) as runtime:
        state_only = bind_state_reset_plan(runtime)
        assert runtime.backend.get_model_recompute_diagnostics(state_only) is None


@pytest.mark.slow
def test_strongest_recompute_runs_once_per_barrier() -> None:
    """Mixed fixed/reference work follows the exact reset publication order."""

    with model_mutation_runtime(num_envs=4, plan_keys=()) as runtime:
        plan = bind_combined_model_plan(
            runtime,
            targets=(
                ("joint.armature", MutationOperation.SET, "model.armature"),
                (
                    "body.gravity_compensation",
                    MutationOperation.SET,
                    "model.gravcomp",
                ),
            ),
            mixed_state=True,
        )
        owner = runtime.backend._device_mutation_plans[plan.fingerprint]
        assert owner.recompute_runtime is not None
        recompute_graph = owner.recompute_runtime.graph
        bundle = runtime.backend._device_graph_bundles[runtime.plan.fingerprint]
        bound = runtime.backend._device_batch_plans[runtime.plan.fingerprint]
        buffers = ResetBatchBuffers(runtime, plan)
        buffers.active_mask[[1, 3]] = True
        buffers.values["model.armature"][[1, 3], 0, 0] = 0.08
        buffers.values["model.gravcomp"][[1, 3], 0, 0] = 0.5
        buffers.values["state.position"][[1, 3], 0, 0] = 0.1
        buffers.values["state.velocity"][[1, 3], 0, 0] = 0.0
        trace: list[str] = []

        original_launch = runtime.backend._warp.capture_launch
        original_commit = owner.commit_model
        original_stage = owner.stage_reset_state
        original_refresh = bound.refresh_masked

        def traced_launch(graph: Any) -> None:
            if graph is bundle.reset_graph:
                trace.append("reset")
            elif graph is recompute_graph:
                trace.append("set_const")
            elif graph is bundle.forward_graph:
                trace.append("forward")
            else:  # pragma: no cover - this barrier only owns these three graphs.
                trace.append("unexpected")
            original_launch(graph)

        def traced_commit(*args: Any, **kwargs: Any) -> None:
            trace.append("model commit")
            original_commit(*args, **kwargs)

        def traced_stage(*args: Any, **kwargs: Any) -> None:
            trace.append("state staging")
            original_stage(*args, **kwargs)

        def traced_refresh(*args: Any, **kwargs: Any) -> None:
            trace.append("refresh")
            original_refresh(*args, **kwargs)

        diagnostics_before = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics_before is not None
        with (
            patch.object(runtime.backend._warp, "capture_launch", side_effect=traced_launch),
            patch.object(owner, "commit_model", side_effect=traced_commit),
            patch.object(owner, "stage_reset_state", side_effect=traced_stage),
            patch.object(bound, "refresh_masked", side_effect=traced_refresh),
        ):
            _reset_with_buffers(runtime, buffers)

        assert trace == [
            "reset",
            "model commit",
            "set_const",
            "state staging",
            "forward",
            "refresh",
        ]
        diagnostics_after = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics_after is not None
        assert diagnostics_after.kind is MjwarpModelRecomputeKind.SET_CONST
        assert diagnostics_after.launch_count == diagnostics_before.launch_count + 1


@pytest.fixture(scope="module")
def effect_runtimes() -> Iterator[dict[str, ModelMutationRuntime]]:
    with model_mutation_runtime(
        num_envs=_NUM_ENVS,
        plan_keys=(PlanKey("joint.armature", MutationOperation.SET, mixed_state=True),),
    ) as armature_runtime:
        with model_mutation_runtime(
            num_envs=_NUM_ENVS,
            plan_keys=(
                PlanKey(
                    "body.gravity_compensation",
                    MutationOperation.SET,
                    mixed_state=True,
                ),
            ),
            model_file=_SHARPA_MODEL,
            base_name="right_hand_C_MC",
            body_name="right_index_PP",
            joint_name="right_index_MCP_AA",
            actuator_name="right_index_MCP_AA_ctrl",
            keyframe_name="home",
        ) as gravcomp_runtime:
            yield {
                "joint.armature": armature_runtime,
                "body.gravity_compensation": gravcomp_runtime,
            }


@pytest.mark.slow
@pytest.mark.parametrize(
    "target_key",
    ("joint.armature", "body.gravity_compensation"),
)
def test_recompute_mutation_changes_next_step_physics(
    target_key: str,
    effect_runtimes: dict[str, ModelMutationRuntime],
) -> None:
    runtime = effect_runtimes[target_key]
    runtime.restore_compiled_model_defaults()
    target_position = 0.15
    qpos, _ = runtime.set_uniform_state(
        target_position=target_position,
        target_velocity=0.0,
    )
    key = PlanKey(target_key, MutationOperation.SET, mixed_state=True)
    plan = runtime.mutation_plans[key]
    buffers = ResetBatchBuffers(runtime, plan)
    buffers.active_mask[list(_SELECTED)] = True
    if target_key == "joint.armature":
        buffers.values["model.value"][list(_SELECTED), 0, 0] = 0.15
    else:
        assert runtime.model_default("body_gravcomp")[0, runtime.body_id] == 1.0
        buffers.values["model.value"][list(_SELECTED), 0, 0] = 0.0
    buffers.values["state.position"][list(_SELECTED), 0, 0] = target_position
    buffers.values["state.velocity"][list(_SELECTED), 0, 0] = 0.0

    reset = _reset_with_buffers(runtime, buffers)
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
    if target_key == "joint.armature":
        control[:, runtime.actuator_id] += 0.25
    terminal = runtime.backend.step_batch(
        runtime.plan,
        control_batch(runtime, control, owner=f"recompute-effect-{target_key}"),
        nsteps=1,
    )
    wait_result(terminal)
    velocity = state_tensor(terminal.terminal_state, "dof.angular_velocity")
    selected = velocity[list(_SELECTED), runtime.dof_velocity_index]
    controls = velocity[list(_PAIRED_CONTROLS), runtime.dof_velocity_index]
    assert bool(torch.isfinite(selected).all())
    assert bool(torch.isfinite(controls).all())
    assert float(torch.max(torch.abs(selected - controls))) > 1.0e-5


def _replace_manifest_capability(
    manifest: MutationCapabilityManifest,
    target_key: str,
    capability: Any,
) -> MutationCapabilityManifest:
    capabilities = tuple(
        capability if item.target_key == target_key else item for item in manifest.capabilities
    )
    return MutationCapabilityManifest(
        backend_type=manifest.backend_type,
        execution_profile=manifest.execution_profile,
        capabilities=capabilities,
    )


@pytest.mark.slow
def test_descriptor_fields_and_recompute_tier_fail_closed_before_materialization() -> None:
    with model_mutation_runtime(num_envs=4, plan_keys=()) as runtime:
        backend = runtime.backend
        manifest = backend.get_mutation_capability_manifest(ExecutionProfile.DEVICE_RESIDENT)
        armature = next(
            capability
            for capability in manifest.capabilities
            if capability.target_key == "joint.armature"
        )
        assert armature.descriptor is not None

        with pytest.raises(MutationContractError, match="has no field-level descriptor"):
            _replace_manifest_capability(
                manifest,
                armature.target_key,
                replace(armature, descriptor=None),
            )

        extra_direct = replace(
            armature,
            descriptor=replace(
                armature.descriptor,
                direct_fields=("dof_armature", "dof_damping"),
            ),
        )
        extra_derived = replace(
            armature,
            descriptor=replace(
                armature.descriptor,
                derived_fields=(
                    "actuator_acc0",
                    "body_invweight0",
                    "body_subtreemass",
                    "dof_invweight0",
                    "tendon_invweight0",
                    "tendon_length0",
                ),
            ),
        )
        fixed_cases = tuple(
            replace(case, recompute=MutationRecomputeLevel.KINEMATICS)
            for case in armature.descriptor.cases
        )
        wrong_tier = replace(
            armature,
            recompute_levels=frozenset({MutationRecomputeLevel.KINEMATICS}),
            descriptor=replace(
                armature.descriptor,
                derived_fields=("body_subtreemass",),
                cases=fixed_cases,
            ),
        )
        scenarios = (
            (extra_direct, "invalid direct fields"),
            (extra_derived, "invalid derived fields"),
            (wrong_tier, "invalid recompute level"),
        )
        for capability, message in scenarios:
            changed = _replace_manifest_capability(manifest, armature.target_key, capability)
            with patch.object(
                backend,
                "get_mutation_capability_manifest",
                return_value=changed,
            ):
                spec = model_mutation_spec(
                    runtime,
                    target_key="joint.armature",
                    operation=MutationOperation.SET,
                    term_key=f"fault.{message}",
                )
                with pytest.raises(MutationContractError, match=message):
                    backend.bind_mutation_plan((spec,))
            assert backend._device_mutation_plans == {}
            assert backend._model_materialization_receipt is None


@pytest.mark.slow
def test_model_selectors_fail_closed_before_plan_publication() -> None:
    with model_mutation_runtime(
        num_envs=4,
        plan_keys=(),
        model_file=_SHARPA_MODEL,
        base_name="right_hand_C_MC",
        body_name="right_index_PP",
        joint_name="right_index_MCP_AA",
        actuator_name="right_index_MCP_AA_ctrl",
        keyframe_name="home",
    ) as runtime:
        backend = runtime.backend
        armature = model_mutation_spec(
            runtime,
            target_key="joint.armature",
            operation=MutationOperation.SET,
            term_key="selector.armature",
        )
        gravcomp = model_mutation_spec(
            runtime,
            target_key="body.gravity_compensation",
            operation=MutationOperation.SET,
            term_key="selector.gravcomp",
        )

        for selector, message in (
            ("missing_joint", "did not resolve a joint"),
            ("object_joint", "must resolve one hinge joint"),
        ):
            changed = replace(
                armature,
                target=replace(armature.target, selector=selector),
            )
            with pytest.raises(MutationContractError, match=message):
                backend.bind_mutation_plan((changed,))

        original_name2id = backend._mujoco.mj_name2id

        def alias_name2id(model: Any, object_type: Any, name: str) -> int:
            if name == "right_index_MCP_AA_alias":
                name = runtime.joint_name
            return int(original_name2id(model, object_type, name))

        duplicate_selector = MutationSelectorSpec(
            semantic_key="duplicate-armature",
            mode=MutationSelectorMode.EXACT,
            expressions=(runtime.joint_name, "right_index_MCP_AA_alias"),
        )
        duplicate = replace(
            armature,
            target=replace(armature.target, selector=duplicate_selector),
        )
        with patch.object(backend._mujoco, "mj_name2id", side_effect=alias_name2id):
            with pytest.raises(MutationContractError, match="duplicate DoFs"):
                backend.bind_mutation_plan((duplicate,))

        backend._body_ids["world"] = 0
        try:
            world = replace(
                gravcomp,
                target=replace(gravcomp.target, selector="world"),
            )
            with pytest.raises(MutationContractError, match="cannot target the world body"):
                backend.bind_mutation_plan((world,))
        finally:
            backend._body_ids.pop("world", None)

        assert backend._device_mutation_plans == {}
        assert backend._model_materialization_receipt is None


@pytest.mark.slow
def test_recompute_capture_failure_does_not_publish_and_retry_is_deterministic() -> None:
    with model_mutation_runtime(num_envs=4, plan_keys=()) as runtime:
        key = PlanKey("joint.armature", MutationOperation.SET)
        with patch.object(
            MjwarpArmatureRecomputeWorkspace,
            "_recompute",
            autospec=True,
            side_effect=RuntimeError("injected recompute capture failure"),
        ):
            with pytest.raises(BackendBatchContractError, match="failed to capture"):
                bind_model_plan(runtime, key)

        assert runtime.backend._device_mutation_plans == {}
        retained_receipt = runtime.backend._model_materialization_receipt
        assert retained_receipt is not None
        assert runtime.backend._runtime_barrier_count == 0

        plan = bind_model_plan(runtime, key)
        diagnostics = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics is not None
        assert diagnostics.capture_count == 1
        assert diagnostics.launch_count == 0
        assert runtime.backend._model_materialization_receipt is retained_receipt


@pytest.mark.slow
def test_dampratio_model_uses_generic_recompute_workspace() -> None:
    with model_mutation_runtime(
        num_envs=4,
        plan_keys=(),
        model_file=_SHARPA_MODEL,
        base_name="right_hand_C_MC",
        body_name="right_index_PP",
        joint_name="right_index_MCP_AA",
        actuator_name="right_index_MCP_AA_ctrl",
        keyframe_name="home",
    ) as runtime:
        model = runtime.backend._cpu_model
        bias = model.actuator_biasprm[runtime.actuator_id]
        gain = model.actuator_gainprm[runtime.actuator_id]
        assert np.isclose(gain[0], -bias[1], rtol=0.0, atol=model.opt.tolerance)
        bias[2] = 0.9
        assert not MjwarpArmatureRecomputeWorkspace.supports(
            ("dof_armature",),
            "set_const_0",
            model,
            runtime.backend._mujoco,
        )
        key = PlanKey("joint.armature", MutationOperation.SET)
        plan = bind_model_plan(runtime, key)
        owner = runtime.backend._device_mutation_plans[plan.fingerprint]
        assert owner.recompute_runtime is not None
        assert isinstance(owner.recompute_runtime.workspace, MjwarpModelRecomputeWorkspace)


@pytest.mark.slow
def test_recompute_values_with_wrong_shape_dtype_or_device_fail_before_launch() -> None:
    key = PlanKey("joint.armature", MutationOperation.SET)
    with model_mutation_runtime(num_envs=4, plan_keys=(key,)) as runtime:
        plan = runtime.mutation_plans[key]
        diagnostics_before = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics_before is not None
        direct_before = runtime.model_field("dof_armature").clone()
        invalid_values = (
            torch.zeros((4, 2, 1), dtype=torch.float32, device=runtime.device),
            torch.zeros((4, 1, 1), dtype=torch.float64, device=runtime.device),
            torch.zeros((4, 1, 1), dtype=torch.float32),
        )
        for invalid in invalid_values:
            buffers = ResetBatchBuffers(runtime, plan)
            buffers.active_mask[1] = True
            buffers.values["model.value"] = invalid
            with pytest.raises(BackendBatchContractError):
                runtime.backend.reset_batch(
                    runtime.plan,
                    RowSelection.all(runtime.num_envs),
                    mutation_batch=buffers.publish(),
                )

        diagnostics_after = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics_after is not None
        assert diagnostics_after.launch_count == diagnostics_before.launch_count
        torch.testing.assert_close(runtime.model_field("dof_armature"), direct_before)


@pytest.mark.slow
def test_recompute_receipt_roles_and_storage_identity_fail_closed() -> None:
    key = PlanKey("joint.armature", MutationOperation.SET)
    with model_mutation_runtime(num_envs=4, plan_keys=(key,)) as runtime:
        plan = runtime.mutation_plans[key]
        backend = runtime.backend
        owner = backend._device_mutation_plans[plan.fingerprint]
        recompute = owner.recompute_runtime
        receipt = backend._model_materialization_receipt
        assert recompute is not None
        assert receipt is not None

        fields = tuple(
            replace(field, role=MjwarpModelFieldRole.DERIVED)
            if field.field_name == "dof_armature"
            else field
            for field in receipt.fields
        )
        wrong_role_receipt = replace(receipt, fields=fields)
        with pytest.raises(BackendBatchContractError, match="field roles"):
            MjwarpModelRecomputeRuntime(
                public_plan=plan,
                contract=recompute.contract,
                graph=recompute.graph,
                condition_reduction=recompute.condition_reduction,
                graph_condition=recompute.graph_condition,
                warp_graph_condition=recompute.warp_graph_condition,
                workspace=recompute.workspace,
                storage_generation=recompute.storage_generation,
                storage_fingerprint=recompute.storage_fingerprint,
                materialization_receipt=wrong_role_receipt,
            )

        corrupt_receipt = replace(receipt)
        object.__setattr__(corrupt_receipt, "fingerprint", "tampered-recompute-receipt")
        with pytest.raises(MjwarpModelMaterializationContractError, match="fingerprint"):
            MjwarpModelRecomputeRuntime(
                public_plan=plan,
                contract=recompute.contract,
                graph=recompute.graph,
                condition_reduction=recompute.condition_reduction,
                graph_condition=recompute.graph_condition,
                warp_graph_condition=recompute.warp_graph_condition,
                workspace=recompute.workspace,
                storage_generation=recompute.storage_generation,
                storage_fingerprint=recompute.storage_fingerprint,
                materialization_receipt=corrupt_receipt,
            )

        buffers = ResetBatchBuffers(runtime, plan)
        buffers.active_mask[1] = True
        buffers.values["model.value"][1, 0, 0] = 0.1
        direct_before = runtime.model_field("dof_armature").clone()

        backend._model_materialization_receipt = replace(receipt)
        try:
            with pytest.raises(BackendBatchContractError, match="receipt changed"):
                runtime.backend.reset_batch(
                    runtime.plan,
                    RowSelection.all(runtime.num_envs),
                    mutation_batch=buffers.publish(),
                )
        finally:
            backend._model_materialization_receipt = receipt
        torch.testing.assert_close(runtime.model_field("dof_armature"), direct_before)

        backend._device_graph_storage_generation += 1
        try:
            with pytest.raises(BackendBatchContractError, match="stale storage identity"):
                runtime.backend.reset_batch(
                    runtime.plan,
                    RowSelection.all(runtime.num_envs),
                    mutation_batch=buffers.publish(),
                )
        finally:
            backend._device_graph_storage_generation -= 1
        torch.testing.assert_close(runtime.model_field("dof_armature"), direct_before)


@pytest.mark.slow
def test_invalid_model_rows_skip_recompute_but_still_reset_state() -> None:
    key = PlanKey("joint.armature", MutationOperation.SCALE, mixed_state=True)
    with model_mutation_runtime(num_envs=6, plan_keys=(key,)) as runtime:
        runtime.restore_compiled_model_defaults()
        plan = runtime.mutation_plans[key]
        owner = runtime.backend._device_mutation_plans[plan.fingerprint]
        assert owner.model_plan is not None
        scalar_target = owner.model_plan._targets[0]
        assert hasattr(scalar_target, "default_values")
        original_default = scalar_target.default_values.clone()
        scalar_target.default_values.fill_(2.0)
        try:
            buffers = ResetBatchBuffers(runtime, plan)
            buffers.active_mask[[0, 1, 2, 3, 4]] = True
            values = buffers.values["model.value"][:, 0, 0]
            values[0] = 0.5
            values[1] = float("nan")
            values[2] = -1.0
            values[3] = float("inf")
            values[4] = torch.finfo(torch.float32).max
            buffers.values["state.position"][:, 0, 0] = 0.37
            buffers.values["state.velocity"][:, 0, 0] = 0.0
            diagnostics_before = runtime.backend.get_mutation_performance_diagnostics(plan)

            with forbid_host_roundtrip(runtime.backend):
                result = _reset_with_buffers(runtime, buffers)
        finally:
            scalar_target.default_values.copy_(original_default)

        effective = owner.model_plan.effective_mask
        assert bool(effective[0])
        assert not bool(effective[[1, 2, 3, 4]].any())
        direct = runtime.model_field("dof_armature")
        torch.testing.assert_close(
            direct[0, runtime.raw_qvel_index],
            torch.tensor(1.0, device=runtime.device),
        )
        default = runtime.model_default("dof_armature")[0, runtime.raw_qvel_index]
        torch.testing.assert_close(
            direct[[1, 2, 3, 4], runtime.raw_qvel_index],
            default.expand(4),
        )
        reset_position = state_tensor(result.reset_state, "dof.position")
        torch.testing.assert_close(
            reset_position[[0, 1, 2, 3, 4], runtime.dof_position_index],
            torch.full((5,), 0.37, dtype=torch.float32, device=runtime.device),
        )
        diagnostics_after = runtime.backend.get_mutation_performance_diagnostics(plan)
        assert (
            diagnostics_after.lifecycle.invalid_model_sample_rows
            == diagnostics_before.lifecycle.invalid_model_sample_rows + 4
        )


@pytest.mark.slow
def test_gravity_compensation_accepts_finite_negative_values() -> None:
    key = PlanKey("body.gravity_compensation", MutationOperation.SET)
    with model_mutation_runtime(
        num_envs=4,
        plan_keys=(key,),
        model_file=_SHARPA_MODEL,
        base_name="right_hand_C_MC",
        body_name="right_index_PP",
        joint_name="right_index_MCP_AA",
        actuator_name="right_index_MCP_AA_ctrl",
        keyframe_name="home",
    ) as runtime:
        buffers = ResetBatchBuffers(runtime, runtime.mutation_plans[key])
        buffers.active_mask[2] = True
        buffers.values["model.value"][2, 0, 0] = -0.5
        _reset_with_buffers(runtime, buffers)
        torch.testing.assert_close(
            runtime.model_field("body_gravcomp")[2, runtime.body_id],
            torch.tensor(-0.5, device=runtime.device),
        )


@pytest.mark.slow
def test_all_false_mask_is_device_noop_without_host_predicate() -> None:
    key = PlanKey("joint.armature", MutationOperation.SET, mixed_state=True)
    with model_mutation_runtime(num_envs=4, plan_keys=(key,)) as runtime:
        plan = runtime.mutation_plans[key]
        buffers = ResetBatchBuffers(runtime, plan)
        buffers.values["model.value"].fill_(0.25)
        buffers.values["state.position"].fill_(0.4)
        _reset_with_buffers(runtime, buffers)
        direct_before = runtime.model_field("dof_armature").clone()
        runtime.model_field("dof_invweight0").fill_(0.125)
        torch.cuda.synchronize(runtime.device)
        derived_before = runtime.model_field("dof_invweight0").clone()
        bridge = runtime.backend._ensure_device_bridge()
        qpos_before = bridge.qpos.clone()
        qvel_before = bridge.qvel.clone()
        diagnostics_before = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics_before is not None

        with forbid_host_roundtrip(runtime.backend):
            _reset_with_buffers(runtime, buffers)

        torch.testing.assert_close(runtime.model_field("dof_armature"), direct_before)
        torch.testing.assert_close(runtime.model_field("dof_invweight0"), derived_before)
        torch.testing.assert_close(bridge.qpos, qpos_before)
        torch.testing.assert_close(bridge.qvel, qvel_before)
        diagnostics_after = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics_after is not None
        assert diagnostics_after.launch_count == diagnostics_before.launch_count + 1
        owner = runtime.backend._device_mutation_plans[plan.fingerprint]
        assert owner.recompute_runtime is not None
        assert owner.recompute_runtime.graph_condition is not None
        assert owner.recompute_runtime.graph_condition.item() == 0


@pytest.mark.slow
def test_warm_recompute_keeps_addresses_and_cuda_allocations_stable() -> None:
    key = PlanKey("joint.armature", MutationOperation.SCALE)
    with model_mutation_runtime(num_envs=8, plan_keys=(key,)) as runtime:
        plan = runtime.mutation_plans[key]
        owner = runtime.backend._device_mutation_plans[plan.fingerprint]
        buffers = ResetBatchBuffers(runtime, plan)
        buffers.active_mask[[1, 3, 5, 7]] = True
        buffers.values["model.value"].fill_(1.1)
        _reset_with_buffers(runtime, buffers)
        receipt = runtime.backend._model_materialization_receipt
        assert receipt is not None
        model_addresses = tuple(
            int(runtime.model_field(field.field_name).data_ptr()) for field in receipt.fields
        )
        addresses = (
            *buffers.numeric_addresses,
            *owner.numeric_buffer_addresses,
            *model_addresses,
        )
        allocation_count = torch.cuda.memory_stats(runtime.device)["allocation.all.allocated"]
        diagnostics_before = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics_before is not None

        for _ in range(8):
            with forbid_host_roundtrip(runtime.backend):
                result = _reset_with_buffers(runtime, buffers)
            assert result.diagnostics.counters.host_to_device_transfers == 0
            assert result.diagnostics.counters.device_to_host_transfers == 0
            assert result.diagnostics.counters.global_synchronizations == 0
            assert result.diagnostics.counters.allocations == 0

        assert (
            torch.cuda.memory_stats(runtime.device)["allocation.all.allocated"] == allocation_count
        )
        assert addresses == (
            *buffers.numeric_addresses,
            *owner.numeric_buffer_addresses,
            *(int(runtime.model_field(field.field_name).data_ptr()) for field in receipt.fields),
        )
        diagnostics_after = runtime.backend.get_model_recompute_diagnostics(plan)
        assert diagnostics_after is not None
        assert diagnostics_after.launch_count == diagnostics_before.launch_count + 8
    assert recompute_derived_fields(MjwarpModelRecomputeKind.SET_CONST) == (
        "actuator_acc0",
        "body_invweight0",
        "body_subtreemass",
        "dof_invweight0",
        "tendon_invweight0",
        "tendon_length0",
    )
