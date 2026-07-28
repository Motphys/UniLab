"""Pure-contract tests for backend transfer telemetry snapshots."""

from __future__ import annotations

import pytest

from unilab.base.backend import (
    BackendTransferBarrier,
    BackendTransferBuffer,
    BackendTransferCounters,
    BackendTransferDirection,
    BackendTransferEvent,
    BackendTransferProfile,
    BackendTransferTelemetryError,
    BackendTransferTrace,
)
from unilab.base.backend.mjwarp.telemetry import MjwarpTransferTelemetry


def test_trace_reconstructs_counters_independently_from_events() -> None:
    trace = BackendTransferTrace(
        events=(
            BackendTransferEvent(
                sequence=0,
                barrier="step",
                direction=BackendTransferDirection.HOST_TO_DEVICE,
                buffer_name="control",
                nbytes=64,
            ),
            BackendTransferEvent(
                sequence=1,
                barrier="step",
                direction=BackendTransferDirection.SYNCHRONIZE,
                buffer_name=None,
            ),
            BackendTransferEvent(
                sequence=2,
                barrier="step",
                direction=BackendTransferDirection.DEVICE_TO_HOST,
                buffer_name="state",
                nbytes=128,
            ),
        )
    )

    assert trace.counters() == BackendTransferCounters(
        host_to_device_transfers=1,
        device_to_host_transfers=1,
        host_to_device_bytes=64,
        device_to_host_bytes=128,
        global_synchronizations=1,
    )
    assert trace.overflow_count == 0


def test_telemetry_contract_rejects_ambiguous_buffers_sequences_and_regression() -> None:
    with pytest.raises(BackendTransferTelemetryError, match="nbytes must be > 0"):
        BackendTransferBuffer("control", 0)
    with pytest.raises(BackendTransferTelemetryError, match="duplicates"):
        BackendTransferBarrier("step", host_to_device_buffers=("control", "control"))
    with pytest.raises(BackendTransferTelemetryError, match="dense"):
        BackendTransferTrace(
            events=(
                BackendTransferEvent(
                    sequence=1,
                    barrier="step",
                    direction=BackendTransferDirection.SYNCHRONIZE,
                    buffer_name=None,
                ),
            )
        )
    with pytest.raises(BackendTransferTelemetryError, match="monotonic"):
        BackendTransferCounters().delta(BackendTransferCounters(host_to_device_transfers=1))

    profile = BackendTransferProfile(
        name="test-host",
        execution_profile="host_numpy",
        barriers=(BackendTransferBarrier("step", host_to_device_buffers=("control",)),),
    )
    assert profile.barrier("step").host_to_device_buffers == ("control",)
    with pytest.raises(KeyError, match="available"):
        profile.barrier("reset")


def test_mjwarp_fixed_capacity_trace_exposes_overflow_without_dropping_counters() -> None:
    telemetry = MjwarpTransferTelemetry(capacity=1)
    telemetry.begin_barrier("step")
    telemetry.host_to_device("control", 64)
    telemetry.host_to_device("control", 64)

    assert telemetry.counters() == BackendTransferCounters(
        host_to_device_transfers=2,
        host_to_device_bytes=128,
    )
    trace = telemetry.trace()
    assert trace.overflow_count == 1
    assert trace.counters() == BackendTransferCounters(
        host_to_device_transfers=1,
        host_to_device_bytes=64,
    )
