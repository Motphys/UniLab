"""Profiler/counter reconciliation for the managed MuJoCo/MJWarp rollout all-device rollout path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
import torch
from tests._support.device_runtime import forbid_host_roundtrip, runtime_harness
from torch.profiler import ProfilerActivity, profile, record_function

from unilab.manager import DeviceRuntimeTrafficDiagnostics

pytestmark = pytest.mark.slow

_ROLLOUT_SCOPE = "manager_mjwarp.device_rollout"
_PHYSICS_STEP_SCOPE = "manager_mjwarp.device_physics_step"
_PHYSICS_RESET_SCOPE = "manager_mjwarp.device_physics_reset"


def _traffic_signature(value: DeviceRuntimeTrafficDiagnostics) -> tuple[int, ...]:
    return (
        value.host_to_device_transfers,
        value.device_to_host_transfers,
        value.host_to_device_bytes,
        value.device_to_host_bytes,
        value.global_synchronizations,
        value.backend_allocations,
        value.dynamic_getter_calls,
        value.selector_resolutions,
        value.asset_metadata_reads,
        value.registry_lookups,
    )


def _profiled_host_traffic(
    trace: dict[str, Any], *, scope_name: str = _ROLLOUT_SCOPE
) -> tuple[int, int, int]:
    events = trace.get("traceEvents")
    assert isinstance(events, list)
    scopes = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("name") == scope_name
        and event.get("cat") == "user_annotation"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    assert scopes
    intervals = tuple(
        (float(scope["ts"]), float(scope["ts"]) + float(scope["dur"])) for scope in scopes
    )
    h2d = d2h = global_sync = 0
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("ts"), (int, float)):
            continue
        timestamp = float(event["ts"])
        if not any(start <= timestamp <= end for start, end in intervals):
            continue
        payload = (
            str(event.get("name", "")) + " " + json.dumps(event.get("args", {}), sort_keys=True)
        ).lower()
        if any(token in payload for token in ("htod", "host to device", "host -> device")):
            h2d += 1
        if any(token in payload for token in ("dtoh", "device to host", "device -> host")):
            d2h += 1
        if "cudadevicesynchronize" in payload:
            global_sync += 1
    return h2d, d2h, global_sync


def _profiled_call(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with record_function(name):
            return function(*args, **kwargs)

    return wrapped


def test_rollout_has_no_per_step_host_roundtrip(tmp_path: Path) -> None:
    """Typed zero-transfer counters must agree with an independent CUPTI trace."""

    steps = 8
    with runtime_harness(
        num_envs=128,
        seed=0,
        max_episode_steps=10_000,
    ) as harness:
        for _ in range(3):
            harness.step(0.01)
        harness.wait()
        baseline = harness.runtime.traffic_diagnostics
        action_pointer = int(harness.action.data_ptr())

        with profile(
            activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            with record_function(_ROLLOUT_SCOPE):
                with (
                    patch.object(
                        harness.backend,
                        "step_batch",
                        side_effect=_profiled_call(_PHYSICS_STEP_SCOPE, harness.backend.step_batch),
                    ),
                    patch.object(
                        harness.backend,
                        "reset_batch",
                        side_effect=_profiled_call(
                            _PHYSICS_RESET_SCOPE, harness.backend.reset_batch
                        ),
                    ),
                    forbid_host_roundtrip(harness.backend),
                ):
                    for index in range(steps):
                        harness.step(0.005 * float(index % 3))
                        assert int(harness.action.data_ptr()) == action_pointer
                    # Extend the profiler annotation through completion of all
                    # queued work so an asynchronous memcpy cannot escape the
                    # measured timestamp interval. This is one low-frequency
                    # event wait, not a per-step or device-global synchronize.
                    harness.wait()

        trace_path = tmp_path / "device_rollout_trace.json"
        profiler.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        total_traffic = _profiled_host_traffic(trace)
        physics_step_traffic = _profiled_host_traffic(trace, scope_name=_PHYSICS_STEP_SCOPE)
        physics_reset_traffic = _profiled_host_traffic(trace, scope_name=_PHYSICS_RESET_SCOPE)
        assert total_traffic == (0, 0, 0), (
            f"total={total_traffic}, physics_step={physics_step_traffic}, "
            f"physics_reset={physics_reset_traffic}"
        )

        observed = harness.runtime.traffic_diagnostics
        assert _traffic_signature(observed) == (0,) * 10
        assert observed.policy_steps - baseline.policy_steps == steps
        assert observed.step_barriers - baseline.step_barriers == steps
        assert observed.reset_barriers - baseline.reset_barriers == steps
        assert observed.state_materializations - baseline.state_materializations == 2 * steps
        assert observed.instrumentation_complete

        diagnostics = harness.runtime.stability_diagnostics
        assert diagnostics is not None and diagnostics.instrumentation_complete
        assert diagnostics.warm_numeric_allocations == 0
        assert diagnostics.address_churn == 0

    # Lifecycle logging and reward-term count may change CPU descriptors and
    # task math, but never the device transfer/synchronization contract.
    signatures: list[tuple[int, ...]] = []
    for minimal_rewards, record_lifecycle in ((True, False), (False, True)):
        with runtime_harness(
            num_envs=128,
            seed=0,
            max_episode_steps=10_000,
            minimal_rewards=minimal_rewards,
            record_lifecycle=record_lifecycle,
        ) as harness:
            with forbid_host_roundtrip(harness.backend):
                for _ in range(4):
                    harness.step()
            signatures.append(_traffic_signature(harness.runtime.traffic_diagnostics))
    assert signatures == [(0,) * 10, (0,) * 10]
