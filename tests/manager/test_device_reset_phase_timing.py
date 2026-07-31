"""Real-CUDA contracts for allocation-stable mjwarp reset phase timing."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tests.training.device_runtime_harness import forbid_host_roundtrip, runtime_harness

import unilab.base.backend.phase_timing as phase_timing_module
from unilab.base.backend import (
    DEVICE_RESET_TIMING_PHASES,
    BackendBatchContractError,
    DevicePhaseTimingError,
    DeviceResetPhaseTimingSampleToken,
    DeviceResetPhaseTimingSession,
)
from unilab.manager import DeviceManagedRuntimeError

pytestmark = pytest.mark.slow


def test_phase_timing_is_allocation_free_when_disabled() -> None:
    with patch.object(
        phase_timing_module,
        "_create_timing_event",
        side_effect=AssertionError("disabled timing created a CUDA event"),
    ):
        with runtime_harness(num_envs=4, seed=8290, max_episode_steps=128) as harness:
            assert harness.runtime.reset_phase_timing_diagnostics is None
            transition = harness.runtime.reset()
            transition.completion.event.synchronize()
            assert harness.runtime.reset_phase_timing_diagnostics is None


def test_phase_timing_preallocates_and_materializes_once_after_window() -> None:
    created_events: list[torch.cuda.Event] = []
    synchronized_events: list[torch.cuda.Event] = []
    create_event = phase_timing_module._create_timing_event
    synchronize_event = phase_timing_module._synchronize_timing_event

    def counted_create() -> torch.cuda.Event:
        event = create_event()
        created_events.append(event)
        return event

    def counted_synchronize(event: torch.cuda.Event) -> None:
        synchronized_events.append(event)
        synchronize_event(event)

    with (
        patch.object(phase_timing_module, "_create_timing_event", side_effect=counted_create),
        patch.object(
            phase_timing_module,
            "_synchronize_timing_event",
            side_effect=counted_synchronize,
        ),
        runtime_harness(
            num_envs=8,
            seed=8291,
            max_episode_steps=128,
            randomize_dof_armature=True,
        ) as harness,
    ):
        runtime = harness.runtime
        runtime.begin_reset_phase_timing(capacity=2)
        diagnostics = runtime.reset_phase_timing_diagnostics
        assert diagnostics is not None
        assert diagnostics.events_preallocated == 2 * len(DEVICE_RESET_TIMING_PHASES) * 2
        assert len(created_events) == diagnostics.events_preallocated
        assert len({id(event) for event in created_events}) == len(created_events)
        assert diagnostics.priming_synchronizations == 1
        assert len(synchronized_events) == 1

        with (
            patch.object(
                phase_timing_module,
                "_create_timing_event",
                side_effect=AssertionError("measurement window allocated a timing event"),
            ),
            patch.object(
                phase_timing_module,
                "_synchronize_timing_event",
                side_effect=AssertionError("measurement window synchronized a timing event"),
            ),
            forbid_host_roundtrip(harness.backend),
        ):
            runtime.reset()
            runtime.reset()

        diagnostics = runtime.reset_phase_timing_diagnostics
        assert diagnostics is not None
        assert diagnostics.recorded_samples == 2
        assert diagnostics.materializations == 0
        assert diagnostics.materialization_synchronizations == 0
        assert not diagnostics.sample_open
        assert len(created_events) == diagnostics.events_preallocated
        assert len(synchronized_events) == 1

        session = cast(Any, harness.backend)._reset_phase_timing_session
        trace = runtime.materialize_reset_phase_timings()
        assert trace.backend_type == "mjwarp"
        assert trace.capacity == 2
        assert len(trace.samples) == 2
        assert trace.events_preallocated == diagnostics.events_preallocated
        assert trace.priming_synchronizations == 1
        assert trace.materialization_synchronizations == 1
        assert len(synchronized_events) == 2
        assert synchronized_events[-1] is created_events[-1]
        assert runtime.reset_phase_timing_diagnostics is None

        for index, sample in enumerate(trace.samples):
            assert sample.sample_index == index
            assert tuple(interval.phase for interval in sample.intervals) == (
                DEVICE_RESET_TIMING_PHASES
            )
            assert all(np.isfinite(interval.milliseconds) for interval in sample.intervals)
            assert sample.interval("reset_barrier").milliseconds > 0.0

        assert session is not None
        final_diagnostics = session.diagnostics
        assert final_diagnostics.materializations == 1
        assert final_diagnostics.materialization_synchronizations == 1
        with pytest.raises(DevicePhaseTimingError, match="only materialize once"):
            session.materialize()


def test_phase_timing_capacity_overflow_rejects_before_graph_launch() -> None:
    with runtime_harness(
        num_envs=4,
        seed=8292,
        max_episode_steps=128,
        randomize_dof_armature=True,
    ) as harness:
        runtime = harness.runtime
        runtime.begin_reset_phase_timing(capacity=1)
        runtime.reset()
        graph_before = harness.backend.get_device_graph_diagnostics()

        with pytest.raises(DeviceManagedRuntimeError, match="capacity 1 is exhausted"):
            runtime.reset()

        graph_after = harness.backend.get_device_graph_diagnostics()
        assert graph_after.launch_count == graph_before.launch_count
        diagnostics = runtime.reset_phase_timing_diagnostics
        assert diagnostics is not None
        assert diagnostics.recorded_samples == 1
        assert diagnostics.overflow_attempts == 1
        assert not diagnostics.sample_open
        trace = runtime.materialize_reset_phase_timings()
        assert len(trace.samples) == 1


@pytest.mark.parametrize("token_kind", ("forged", "foreign"))
def test_phase_timing_rejects_invalid_token_before_graph_launch(token_kind: str) -> None:
    with runtime_harness(
        num_envs=4,
        seed=8293,
        max_episode_steps=128,
        randomize_dof_armature=True,
    ) as harness:
        runtime = harness.runtime
        runtime.begin_reset_phase_timing(capacity=1)
        foreign_token: DeviceResetPhaseTimingSampleToken | None = None
        if token_kind == "foreign":
            foreign_session = DeviceResetPhaseTimingSession(
                backend_type="mjwarp",
                backend_instance_id="foreign-backend",
                placement=harness.placement,
                capacity=1,
            )
            foreign_token = foreign_session.begin_sample(runtime._task_stream)

        original_reset = harness.backend.reset_batch

        def replace_token(
            plan: Any,
            rows: Any,
            *,
            mutation_batch: Any = None,
            phase_timing: DeviceResetPhaseTimingSampleToken | None = None,
        ) -> Any:
            assert phase_timing is not None
            if token_kind == "forged":
                invalid_token = DeviceResetPhaseTimingSampleToken(
                    sample_index=phase_timing.sample_index,
                )
            else:
                assert foreign_token is not None
                invalid_token = foreign_token
            return original_reset(
                plan,
                rows,
                mutation_batch=mutation_batch,
                phase_timing=invalid_token,
            )

        graph_before = harness.backend.get_device_graph_diagnostics()
        with (
            patch.object(harness.backend, "reset_batch", side_effect=replace_token),
            pytest.raises(BackendBatchContractError, match="foreign or forged"),
        ):
            runtime.reset()
        graph_after = harness.backend.get_device_graph_diagnostics()
        assert graph_after.launch_count == graph_before.launch_count
