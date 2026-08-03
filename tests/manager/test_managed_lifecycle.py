from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend
from unilab.base.backend.batch import (
    BACKEND_BATCH_CONTRACT_VERSION,
    BackendBatchContractError,
    BackendReadResult,
    BackendResetResult,
    BackendStepResult,
    BoundBackendPlan,
    BoundStatePlan,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    ControlBatch,
    ControlSpec,
    ExecutionProfile,
    PhysicalUnit,
    ReferenceFrame,
    RowSelection,
    StaleStateBatchError,
    StateBatch,
    StateBatchLease,
    StateBatchPhase,
    StateFieldKind,
)
from unilab.base.backend.mutation import (
    BoundMutationPlan,
    BoundMutationSpec,
    BoundMutationTarget,
    MutationBaseline,
    MutationCommitPhase,
    MutationFieldKind,
    MutationOperation,
    MutationPersistence,
    MutationRecomputeLevel,
    MutationSpec,
    MutationTargetKind,
    MutationTrigger,
)
from unilab.base.backend.mutation_batch import TypedBackendMutationBatch
from unilab.manager import (
    EntityKind,
    EntitySelector,
    ManagedKernelBinding,
    ManagedLifecyclePhase,
    ManagedMetric,
    ManagedReferenceRuntime,
    ManagedResetRequest,
    ManagedRuntimeError,
    MutationTemplate,
    PolicySpec,
    StateRequirement,
    TaskCompiler,
    TaskSpec,
    TensorSpec,
    TermDefinition,
    TermInvocation,
    TermPhase,
    TermRegistry,
    TermRole,
)


class _Resolver:
    def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
        assert selector.key == "robot.base"
        return (0,)


def _control_buffer(*, placement: BufferPlacement) -> BufferContract:
    return BufferContract(
        row_shape=(1,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
        dlpack_exportable=False,
    )


def _mutation_buffer(*, placement: BufferPlacement) -> BufferContract:
    return BufferContract(
        row_shape=(3,),
        dtype="float32",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=False,
    )


def _build_plan(
    *,
    profile: ExecutionProfile = ExecutionProfile.HOST_NUMPY,
    with_mutation: bool = False,
):
    placement = (
        BufferPlacement.host()
        if profile is ExecutionProfile.HOST_NUMPY
        else BufferPlacement.device("cuda", 0)
    )
    base = EntitySelector(
        key="robot.base",
        entity="robot",
        kind=EntityKind.BODY,
        expressions=("base",),
    )
    registry = TermRegistry()
    if with_mutation:
        registry.register(
            TermDefinition(
                key="event.reset_base",
                version="1",
                phase=TermPhase.RESET,
                role=TermRole.EVENT,
                mutation_templates=(
                    MutationTemplate(
                        key_suffix="",
                        target_key="state.body.position",
                        target_kind=MutationTargetKind.SIMULATION_STATE,
                        selector=base,
                        field_kind=MutationFieldKind.POSITION,
                        trigger=MutationTrigger.RESET,
                        commit_phase=MutationCommitPhase.RESET,
                        operation=MutationOperation.SET,
                        baseline=MutationBaseline.DEFAULT,
                        persistence=MutationPersistence.EPISODE,
                        recompute=MutationRecomputeLevel.KINEMATICS,
                        value_template=_mutation_buffer(placement=placement),
                    ),
                ),
            )
        )
    registry.register(
        TermDefinition(
            key="obs.base_scalar",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="robot.base.position",
                    selector=base,
                    field_kind=StateFieldKind.POSITION,
                    tensor=TensorSpec(
                        shape=(1, 1),
                        dtype="float32",
                        frame=ReferenceFrame.WORLD,
                        unit=PhysicalUnit.METER,
                    ),
                ),
            ),
            output=TensorSpec(shape=(1,), dtype="float32"),
        )
    )
    terms = [
        TermInvocation.create(
            key="base_scalar",
            definition_key="obs.base_scalar",
            observation_group="policy",
        )
    ]
    if with_mutation:
        terms.append(TermInvocation.create(key="reset_base", definition_key="event.reset_base"))
    return TaskCompiler(registry).compile(
        TaskSpec.create(
            key="managed_lifecycle_fixture",
            terms=terms,
            control=ControlSpec(
                semantic_key="robot.command",
                buffer=_control_buffer(placement=placement),
            ),
            execution_profile=profile,
            executor_key="recording.reference.v1",
            policy=PolicySpec(observation_groups=("policy",), action_scale=(1.0,)),
        ),
        resolver=_Resolver(),
        capabilities=frozenset({"state.body.position"}),
    )


class _RecordingBackend:
    """Typed-only backend oracle; every legacy path fails if it is reached."""

    backend_type = "recording"
    _instance_id = "recording:managed-lifecycle"

    def __init__(
        self,
        *,
        num_envs: int = 3,
        invalidate_on_reset: bool = True,
        reset_requires_mutation_batch: bool = False,
    ) -> None:
        self._num_envs = num_envs
        self._invalidate_on_reset = invalidate_on_reset
        self._reset_requires_mutation_batch = reset_requires_mutation_batch
        self._bound_plan: BoundBackendPlan | None = None
        self._mutation_plan: BoundMutationPlan | None = None
        self._lease = StateBatchLease(self._instance_id)
        self._values = np.zeros((num_envs, 1, 1), dtype=np.float32)
        self._scratch = np.empty_like(self._values)
        self.bind_calls = 0
        self.reset_calls: list[RowSelection] = []
        self.step_calls = 0
        self.control_batches: list[ControlBatch] = []
        self.legacy_step_calls = 0
        self.legacy_set_state_calls = 0
        self.terminal_batches: list[StateBatch] = []
        self._reset_generation = 0

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def bind_task_io(self, requirements: Any) -> BoundBackendPlan:
        self.bind_calls += 1
        state = BoundStatePlan(
            backend_type=self.backend_type,
            backend_instance_id=self._instance_id,
            num_envs=self._num_envs,
            fields=requirements.state_fields,
            execution_profile=requirements.execution_profile,
            fingerprint="recording-state-v1",
        )
        self._bound_plan = BoundBackendPlan(
            state=state,
            control=requirements.control,
            execution_profile=requirements.execution_profile,
            fingerprint=BACKEND_BATCH_CONTRACT_VERSION,
            hot_path_budget=requirements.hot_path_budget,
            reset_hot_path_budget=requirements.reset_hot_path_budget,
            reset_requires_mutation_batch=self._reset_requires_mutation_batch,
        )
        return self._bound_plan

    def bind_mutation_plan(self, specs: tuple[MutationSpec, ...]) -> BoundMutationPlan:
        if self._bound_plan is None:
            raise BackendBatchContractError("recording backend binds mutation after task I/O")
        if len(specs) != 1:
            raise BackendBatchContractError("recording fixture expects one mutation spec")
        spec = specs[0]
        bound_spec = BoundMutationSpec(
            term_key=spec.term_key,
            target=BoundMutationTarget(
                target_key=spec.target.target_key,
                target_kind=spec.target.target_kind,
                entity_kind=spec.target.entity_kind,
                field_kind=spec.target.field_kind,
                entity_ids=(0,),
            ),
            trigger=spec.trigger,
            commit_phase=spec.commit_phase,
            operation=spec.operation,
            baseline=spec.baseline,
            persistence=spec.persistence,
            recompute=spec.recompute,
            value_buffer=replace(
                spec.value_template, row_shape=(1, *spec.value_template.row_shape)
            ),
            capability_fingerprint="recording-capability-v1",
        )
        self._mutation_plan = BoundMutationPlan(
            backend_type=self.backend_type,
            backend_instance_id=self._instance_id,
            num_envs=self._num_envs,
            specs=(bound_spec,),
            capability_manifest_fingerprint="recording-capability-manifest-v1",
            fingerprint="recording-mutation-v1",
        )
        return self._mutation_plan

    def _require_plan(self, plan: BoundBackendPlan) -> BoundBackendPlan:
        if self._bound_plan is None:
            raise BackendBatchContractError("recording backend was not bound")
        self._bound_plan.require_compatible(plan)
        return self._bound_plan

    def _row_indices(self, rows: RowSelection) -> np.ndarray:
        if rows.universe_size != self._num_envs:
            raise BackendBatchContractError("recording row universe differs from backend")
        if rows.is_all:
            return np.arange(self._num_envs, dtype=np.intp)
        assert rows.indices is not None
        return np.asarray(rows.indices, dtype=np.intp)

    def _state(self, rows: RowSelection, phase: StateBatchPhase) -> StateBatch:
        plan = self._require_plan(cast(BoundBackendPlan, self._bound_plan))
        indices = self._row_indices(rows)
        scratch = self._scratch[: rows.count]
        np.take(self._values, indices, axis=0, out=scratch)
        view = scratch.view()
        view.flags.writeable = False
        descriptor = BufferView(
            handle=view,
            shape=view.shape,
            contract=plan.state.fields[0].buffer,
        )
        return StateBatch(
            plan=plan,
            rows=rows,
            phase=phase,
            descriptors=(descriptor,),
            lease=self._lease,
        )

    def read_state_batch(
        self,
        plan: BoundBackendPlan,
        rows: RowSelection,
        *,
        phase: StateBatchPhase = StateBatchPhase.CURRENT,
    ) -> BackendReadResult:
        self._require_plan(plan)
        return BackendReadResult(state=self._state(rows, phase))

    def step_batch(
        self,
        plan: BoundBackendPlan,
        control_batch: ControlBatch,
        *,
        mutation_batch: object | None = None,
        nsteps: int = 1,
    ) -> BackendStepResult:
        self._require_plan(plan)
        if mutation_batch is not None:
            raise BackendBatchContractError("recording fixture has no pre-physics mutation")
        if not control_batch.rows.is_all or control_batch.plan != plan:
            raise BackendBatchContractError("recording fixture requires full-row matching control")
        if nsteps != 1:
            raise BackendBatchContractError("recording fixture requires one physics step")
        control = control_batch.buffer.handle
        if not isinstance(control, np.ndarray) or control.shape != (self._num_envs, 1):
            raise BackendBatchContractError("recording fixture received invalid control")
        self.control_batches.append(control_batch)
        self._lease.invalidate()
        self._values[:, 0, 0] += control[:, 0]
        self.step_calls += 1
        terminal = self._state(RowSelection.all(self._num_envs), StateBatchPhase.TERMINAL)
        self.terminal_batches.append(terminal)
        return BackendStepResult(terminal_state=terminal)

    def reset_batch(
        self,
        plan: BoundBackendPlan,
        rows: RowSelection,
        *,
        mutation_batch: object | None = None,
    ) -> BackendResetResult:
        self._require_plan(plan)
        if self._mutation_plan is not None:
            if not isinstance(mutation_batch, TypedBackendMutationBatch):
                raise BackendBatchContractError("recording reset requires a typed mutation batch")
            self._mutation_plan.require_compatible(mutation_batch.plan)
            if mutation_batch.rows != rows:
                raise BackendBatchContractError("recording mutation rows differ from reset rows")
        elif mutation_batch is not None:
            raise BackendBatchContractError("recording fixture has no mutation plan")
        if self._invalidate_on_reset:
            self._lease.invalidate()
        self._reset_generation += 1
        indices = self._row_indices(rows)
        self._values[indices, 0, 0] = 100.0 * self._reset_generation + indices
        self.reset_calls.append(rows)
        return BackendResetResult(reset_state=self._state(rows, StateBatchPhase.RESET))

    def step(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.legacy_step_calls += 1
        raise AssertionError("managed runtime must not call legacy step")

    def set_state(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.legacy_set_state_calls += 1
        raise AssertionError("managed runtime must not call legacy set_state")


class _ForeignStateBackend(_RecordingBackend):
    def step_batch(
        self,
        plan: BoundBackendPlan,
        control_batch: ControlBatch,
        *,
        mutation_batch: object | None = None,
        nsteps: int = 1,
    ) -> BackendStepResult:
        result = super().step_batch(
            plan,
            control_batch,
            mutation_batch=mutation_batch,
            nsteps=nsteps,
        )
        foreign_state_plan = BoundStatePlan(
            backend_type=self.backend_type,
            backend_instance_id="recording:foreign-state",
            num_envs=self.num_envs,
            fields=plan.state.fields,
            execution_profile=plan.execution_profile,
            fingerprint="foreign-state-plan-v1",
        )
        foreign_plan = BoundBackendPlan(
            state=foreign_state_plan,
            control=plan.control,
            execution_profile=plan.execution_profile,
            fingerprint="foreign-backend-plan-v1",
        )
        terminal = result.terminal_state
        descriptor = BufferView(
            handle=terminal.buffer_at(0).handle,
            shape=terminal.buffer_at(0).shape,
            contract=plan.state.fields[0].buffer,
        )
        return BackendStepResult(
            terminal_state=StateBatch(
                plan=foreign_plan,
                rows=RowSelection.all(self.num_envs),
                phase=StateBatchPhase.TERMINAL,
                descriptors=(descriptor,),
                lease=StateBatchLease("recording:foreign-state"),
            )
        )


SimBackend.register(_RecordingBackend)


class _RecordingKernel:
    executor_key = "recording.reference.v1"

    def __init__(self, termination_masks: tuple[np.ndarray, ...]) -> None:
        self._termination_masks = termination_masks
        self.binding: ManagedKernelBinding | None = None

    def bind(self, *, binding: ManagedKernelBinding) -> None:
        self.binding = binding

    def create_task_state(self, *, num_envs: int, dtype: np.dtype[Any]) -> object:
        return {
            "commands": np.zeros((num_envs,), dtype=dtype),
            "history": np.zeros((num_envs,), dtype=np.int32),
            "evaluation": 0,
            "resets": [],
        }

    def apply_action(
        self,
        *,
        actions: np.ndarray,
        task_state: object,
        control_out: np.ndarray,
    ) -> None:
        state = cast(dict[str, Any], task_state)
        np.copyto(control_out, actions)
        np.copyto(state["commands"], actions[:, 0])
        state["history"] += 1

    def build_pre_physics_mutation(self, *, task_state: object) -> None:
        del task_state
        return None

    def evaluate_termination(
        self,
        *,
        state: StateBatch,
        task_state: object,
        terminated_out: np.ndarray,
    ) -> None:
        del state
        kernel_state = cast(dict[str, Any], task_state)
        index = min(kernel_state["evaluation"], len(self._termination_masks) - 1)
        np.copyto(terminated_out, self._termination_masks[index])
        kernel_state["evaluation"] += 1

    def evaluate_reward(
        self,
        *,
        state: StateBatch,
        task_state: object,
        reward_out: np.ndarray,
    ) -> None:
        del task_state
        values = cast(np.ndarray, state.buffer_at(0).handle)
        np.copyto(reward_out, values[:, 0, 0])

    def evaluate_metrics(
        self,
        *,
        state: StateBatch,
        task_state: object,
        terminated: np.ndarray,
    ) -> tuple[ManagedMetric, ...]:
        del task_state, terminated
        values = cast(np.ndarray, state.buffer_at(0).handle)
        return (ManagedMetric("mean_terminal", float(np.mean(values[:, 0, 0]))),)

    def write_observations(
        self,
        *,
        state: StateBatch,
        task_state: object,
        observation_buffers: tuple[np.ndarray, ...],
    ) -> None:
        del task_state
        values = cast(np.ndarray, state.buffer_at(0).handle)
        if state.rows.is_all:
            indices: slice | np.ndarray = slice(None)
        else:
            assert state.rows.indices is not None
            indices = np.asarray(state.rows.indices, dtype=np.intp)
        observation_buffers[0][indices, 0] = values[:, 0, 0]

    def prepare_reset(self, *, rows: RowSelection, task_state: object) -> ManagedResetRequest:
        del task_state
        return ManagedResetRequest(rows=rows, kernel_state={"reset_count": rows.count})

    def complete_reset(
        self,
        *,
        request: ManagedResetRequest,
        state: StateBatch,
        task_state: object,
    ) -> None:
        state.assert_valid()
        kernel_state = cast(dict[str, Any], task_state)
        if request.rows.is_all:
            indices: slice | np.ndarray = slice(None)
        else:
            assert request.rows.indices is not None
            indices = np.asarray(request.rows.indices, dtype=np.intp)
        kernel_state["commands"][indices] = 0.0
        kernel_state["history"][indices] = -1
        kernel_state["resets"].append(request.rows)


class _MutationKernel(_RecordingKernel):
    def __init__(self, *, mode: str = "valid") -> None:
        super().__init__((np.zeros((3,), dtype=bool),))
        self._mode = mode

    def prepare_reset(self, *, rows: RowSelection, task_state: object) -> ManagedResetRequest:
        del task_state
        assert self.binding is not None
        assert self.binding.mutation_plan is not None
        if self._mode == "missing":
            return ManagedResetRequest(rows=rows)
        if self._mode == "foreign":
            plan = replace(self.binding.mutation_plan, fingerprint="foreign-mutation-plan")
        else:
            plan = self.binding.mutation_plan
        batch = TypedBackendMutationBatch(plan=plan, rows=rows)
        return ManagedResetRequest(rows=rows, mutation_batch=batch)


class _WrongRowsKernel(_RecordingKernel):
    def prepare_reset(self, *, rows: RowSelection, task_state: object) -> ManagedResetRequest:
        request = super().prepare_reset(rows=rows, task_state=task_state)
        if cast(dict[str, Any], task_state)["evaluation"]:
            return ManagedResetRequest(rows=RowSelection.selected(rows.universe_size, (1,)))
        return request


class _NoResetRequestKernel(_RecordingKernel):
    def prepare_reset(self, *, rows: RowSelection, task_state: object) -> ManagedResetRequest:
        del rows, task_state
        return cast(ManagedResetRequest, None)


class _MissingObservationKernel(_RecordingKernel):
    def write_observations(
        self,
        *,
        state: StateBatch,
        task_state: object,
        observation_buffers: tuple[np.ndarray, ...],
    ) -> None:
        del state, task_state, observation_buffers


class _LateTerminalReadKernel(_RecordingKernel):
    def __init__(self) -> None:
        super().__init__((np.array((True, False, False)),))
        self._terminal: StateBatch | None = None

    def evaluate_termination(
        self,
        *,
        state: StateBatch,
        task_state: object,
        terminated_out: np.ndarray,
    ) -> None:
        self._terminal = state
        super().evaluate_termination(
            state=state,
            task_state=task_state,
            terminated_out=terminated_out,
        )

    def complete_reset(
        self,
        *,
        request: ManagedResetRequest,
        state: StateBatch,
        task_state: object,
    ) -> None:
        if self._terminal is not None:
            self._terminal.buffer_at(0).handle
        super().complete_reset(request=request, state=state, task_state=task_state)


def _trace(
    runtime: ManagedReferenceRuntime,
) -> list[tuple[ManagedLifecyclePhase, tuple[int, ...] | None]]:
    return [(event.phase, event.rows) for event in runtime.last_trace]


def test_terminal_and_autoreset_lifecycle_trace() -> None:
    backend = _RecordingBackend()
    kernel = _RecordingKernel(
        (
            np.array((True, False, True)),
            np.array((False, False, False)),
        )
    )
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=kernel,
        max_episode_steps=2,
        record_lifecycle=True,
    )

    initial = runtime.init_state()
    assert initial.final_observation is None
    assert kernel.binding is not None
    assert kernel.binding.mutation_plan is None
    assert kernel.binding.state_field_indices == (("robot.base.position", 0),)
    assert _trace(runtime) == [
        (ManagedLifecyclePhase.INITIAL_RESET_REQUEST, None),
        (ManagedLifecyclePhase.RESET_BACKEND, None),
        (ManagedLifecyclePhase.TASK_STATE_RESET, None),
        (ManagedLifecyclePhase.OBSERVATION, None),
        (ManagedLifecyclePhase.COMPLETE, None),
    ]
    np.testing.assert_array_equal(initial.obs["policy"][:, 0], (100.0, 101.0, 102.0))
    obs_address = initial.obs["policy"].ctypes.data
    final_address = initial.info["final_observation"]["policy"].ctypes.data

    first = runtime.step(np.array(((1.0,), (2.0,), (3.0,)), dtype=np.float32))

    np.testing.assert_array_equal(first.terminated, (True, False, True))
    np.testing.assert_array_equal(first.truncated, (False, False, False))
    np.testing.assert_array_equal(first.info["steps"], (0, 1, 0))
    np.testing.assert_array_equal(first.obs["policy"][:, 0], (200.0, 103.0, 202.0))
    assert first.final_observation is not None
    np.testing.assert_array_equal(first.final_observation["policy"][(0, 2), 0], (101.0, 105.0))
    np.testing.assert_array_equal(first.info["_final_observation"], (True, False, True))
    assert first.obs["policy"].ctypes.data == obs_address
    assert first.info["final_observation"]["policy"].ctypes.data == final_address
    with pytest.raises(StaleStateBatchError):
        backend.terminal_batches[0].assert_valid()
    assert _trace(runtime) == [
        (ManagedLifecyclePhase.ACTION, None),
        (ManagedLifecyclePhase.PRE_PHYSICS, None),
        (ManagedLifecyclePhase.PHYSICS, None),
        (ManagedLifecyclePhase.TERMINATION, None),
        (ManagedLifecyclePhase.REWARD, None),
        (ManagedLifecyclePhase.METRIC, None),
        (ManagedLifecyclePhase.TERMINAL_OBSERVATION, None),
        (ManagedLifecyclePhase.TIMEOUT, None),
        (ManagedLifecyclePhase.FINAL_OBSERVATION, (0, 2)),
        (ManagedLifecyclePhase.AUTORESET, (0, 2)),
        (ManagedLifecyclePhase.RESET_REQUEST, (0, 2)),
        (ManagedLifecyclePhase.RESET_BACKEND, (0, 2)),
        (ManagedLifecyclePhase.TASK_STATE_RESET, (0, 2)),
        (ManagedLifecyclePhase.OBSERVATION, (0, 2)),
        (ManagedLifecyclePhase.COMPLETE, None),
    ]

    second = runtime.step(np.zeros((3, 1), dtype=np.float32))

    np.testing.assert_array_equal(second.terminated, (False, False, False))
    np.testing.assert_array_equal(second.truncated, (False, True, False))
    np.testing.assert_array_equal(second.info["steps"], (1, 0, 1))
    np.testing.assert_array_equal(second.obs["policy"][:, 0], (200.0, 301.0, 202.0))
    assert second.final_observation is not None
    np.testing.assert_array_equal(second.final_observation["policy"][1], (103.0,))
    np.testing.assert_array_equal(second.info["_final_observation"], (False, True, False))
    assert [rows.indices for rows in backend.reset_calls] == [None, (0, 2), (1,)]
    task_state = cast(dict[str, Any], runtime.task_state)
    assert [rows.indices for rows in task_state["resets"]] == [None, (0, 2), (1,)]
    np.testing.assert_array_equal(task_state["history"], (0, -1, 0))
    assert backend.legacy_step_calls == 0
    assert backend.legacy_set_state_calls == 0
    assert len(backend.control_batches) == 2
    assert backend.control_batches[0] is backend.control_batches[1]


def test_no_done_branch_clears_final_observation_without_reallocating_outputs() -> None:
    backend = _RecordingBackend()
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=_RecordingKernel((np.zeros((3,), dtype=bool),)),
        max_episode_steps=None,
        record_lifecycle=True,
    )
    initial = runtime.init_state()
    obs_address = initial.obs["policy"].ctypes.data
    compat_address = initial.info["final_observation"]["policy"].ctypes.data

    state = runtime.step(np.zeros((3, 1), dtype=np.float32))

    assert state.final_observation is None
    np.testing.assert_array_equal(state.info["_final_observation"], (False, False, False))
    assert state.obs["policy"].ctypes.data == obs_address
    assert state.info["final_observation"]["policy"].ctypes.data == compat_address
    assert [rows.indices for rows in backend.reset_calls] == [None]
    assert ManagedLifecyclePhase.FINAL_OBSERVATION not in [
        event.phase for event in runtime.last_trace
    ]
    assert ManagedLifecyclePhase.AUTORESET not in [event.phase for event in runtime.last_trace]


def test_step_before_init_state_fails_without_touching_backend() -> None:
    backend = _RecordingBackend()
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=_RecordingKernel((np.zeros((3,), dtype=bool),)),
        max_episode_steps=None,
    )

    with pytest.raises(ManagedRuntimeError, match=r"requires init_state\(\) first"):
        runtime.step(np.zeros((3, 1), dtype=np.float32))

    assert backend.reset_calls == []
    assert backend.step_calls == 0


def test_runtime_rejects_missing_required_reset_mutation_at_bind_time() -> None:
    backend = _RecordingBackend(reset_requires_mutation_batch=True)

    with pytest.raises(ManagedRuntimeError, match="requires a bound mutation plan"):
        ManagedReferenceRuntime(
            backend=cast(SimBackend, backend),
            plan=_build_plan(),
            kernel=_RecordingKernel((np.zeros((3,), dtype=bool),)),
            max_episode_steps=None,
        )

    assert backend.reset_calls == []


def test_observation_term_validation_reports_unwritten_semantic_key() -> None:
    backend = _RecordingBackend()
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=_MissingObservationKernel((np.zeros((3,), dtype=bool),)),
        max_episode_steps=None,
        validate_observation_terms=True,
    )

    with pytest.raises(ManagedRuntimeError, match="semantic_keys=\\('base_scalar',\\)"):
        runtime.init_state()


@pytest.mark.parametrize(
    ("shape", "dtype", "match"),
    (
        ((3, 2), np.dtype(np.float32), r"must have shape \(3, 1\)"),
        ((3, 1), np.dtype(np.float64), "must have dtype float32"),
    ),
    ids=("width", "dtype"),
)
def test_observation_buffer_validation_uses_policy_abi(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    match: str,
) -> None:
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, _RecordingBackend()),
        plan=_build_plan(),
        kernel=_RecordingKernel((np.zeros((3,), dtype=bool),)),
        max_episode_steps=None,
    )
    runtime._observation_buffers = (np.zeros(shape, dtype=dtype),)

    with pytest.raises(ManagedRuntimeError, match=match):
        runtime.init_state()


@pytest.mark.parametrize("forbidden_name", ("backend", "env", "_backend", "_env"))
def test_kernel_owner_capture_fails_before_backend_bind(forbidden_name: str) -> None:
    backend = _RecordingBackend()
    kernel = _RecordingKernel((np.zeros((3,), dtype=bool),))
    setattr(kernel, forbidden_name, backend)

    with pytest.raises(ManagedRuntimeError, match="forbidden"):
        ManagedReferenceRuntime(
            backend=cast(SimBackend, backend),
            plan=_build_plan(),
            kernel=kernel,
            max_episode_steps=None,
        )

    assert backend.bind_calls == 0


def test_reference_runtime_rejects_device_profile_before_backend_bind() -> None:
    backend = _RecordingBackend()

    with pytest.raises(ManagedRuntimeError, match="host_numpy"):
        ManagedReferenceRuntime(
            backend=cast(SimBackend, backend),
            plan=_build_plan(profile=ExecutionProfile.DEVICE_RESIDENT),
            kernel=_RecordingKernel((np.zeros((3,), dtype=bool),)),
            max_episode_steps=None,
        )

    assert backend.bind_calls == 0


def test_wrong_reset_rows_fail_before_backend_reset_barrier() -> None:
    backend = _RecordingBackend()
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=_WrongRowsKernel((np.array((True, False, True)),)),
        max_episode_steps=None,
    )
    runtime.init_state()

    with pytest.raises(ManagedRuntimeError, match="reset request rows"):
        runtime.step(np.zeros((3, 1), dtype=np.float32))

    assert [rows.indices for rows in backend.reset_calls] == [None]


def test_missing_reset_request_fails_before_backend_reset_barrier() -> None:
    backend = _RecordingBackend()
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=_NoResetRequestKernel((np.zeros((3,), dtype=bool),)),
        max_episode_steps=None,
    )

    with pytest.raises(ManagedRuntimeError, match="prepare_reset"):
        runtime.init_state()

    assert backend.reset_calls == []


@pytest.mark.parametrize(
    ("mode", "match"),
    (("missing", "requires a typed mutation batch"), ("foreign", "different bound mutation plan")),
)
def test_reset_mutation_requests_fail_closed(mode: str, match: str) -> None:
    backend = _RecordingBackend()
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(with_mutation=True),
        kernel=_MutationKernel(mode=mode),
        max_episode_steps=None,
    )

    with pytest.raises(ManagedRuntimeError, match=match):
        runtime.init_state()

    assert backend.reset_calls == []


def test_late_terminal_state_read_fails_at_reset_barrier() -> None:
    backend = _RecordingBackend()
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=_LateTerminalReadKernel(),
        max_episode_steps=None,
    )
    runtime.init_state()

    with pytest.raises(StaleStateBatchError):
        runtime.step(np.zeros((3, 1), dtype=np.float32))


def test_reset_barrier_requires_terminal_lease_invalidation() -> None:
    backend = _RecordingBackend(invalidate_on_reset=False)
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=_RecordingKernel((np.array((True, False, False)),)),
        max_episode_steps=None,
    )
    runtime.init_state()

    with pytest.raises(ManagedRuntimeError, match="did not invalidate"):
        runtime.step(np.zeros((3, 1), dtype=np.float32))


def test_foreign_terminal_state_plan_fails_at_step_barrier() -> None:
    backend = _ForeignStateBackend()
    runtime = ManagedReferenceRuntime(
        backend=cast(SimBackend, backend),
        plan=_build_plan(),
        kernel=_RecordingKernel((np.zeros((3,), dtype=bool),)),
        max_episode_steps=None,
    )
    runtime.init_state()

    with pytest.raises(ManagedRuntimeError, match="different bound backend plan"):
        runtime.step(np.zeros((3, 1), dtype=np.float32))
