"""Real-CUDA transfer/profile reconciliation for ``mjwarp`` host compatibility."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unilab.base.backend import (
    BackendTransferCounters,
    BackendTransferDirection,
    BackendTransferProfile,
    BackendTransferTrace,
    create_backend,
)
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow

_NUM_ENVS = 128


def _require_cuda_mjwarp() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp transfer accounting requires an active CUDA Warp device")


def _backend() -> Any:
    _require_cuda_mjwarp()
    from unilab.assets import ASSETS_ROOT_PATH

    return create_backend(
        "mjwarp",
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")),
        _NUM_ENVS,
        0.02 / 3.0,
        base_name="pelvis",
    )


def _expected_events(
    profile: BackendTransferProfile,
    *,
    barrier_name: str,
    repeats: int,
    buffer_nbytes: dict[str, int],
) -> list[tuple[str, BackendTransferDirection, str | None, int]]:
    barrier = profile.barrier(barrier_name)
    events: list[tuple[str, BackendTransferDirection, str | None, int]] = []
    for _ in range(repeats):
        events.extend(
            (barrier.name, BackendTransferDirection.HOST_TO_DEVICE, name, buffer_nbytes[name])
            for name in barrier.host_to_device_buffers
        )
        events.extend(
            (barrier.name, BackendTransferDirection.SYNCHRONIZE, None, 0)
            for _ in range(barrier.global_synchronizations)
        )
        events.extend(
            (barrier.name, BackendTransferDirection.DEVICE_TO_HOST, name, buffer_nbytes[name])
            for name in barrier.device_to_host_buffers
        )
    return events


def _counters_from_expected(
    events: list[tuple[str, BackendTransferDirection, str | None, int]],
) -> BackendTransferCounters:
    h2d = [event for event in events if event[1] is BackendTransferDirection.HOST_TO_DEVICE]
    d2h = [event for event in events if event[1] is BackendTransferDirection.DEVICE_TO_HOST]
    sync = [event for event in events if event[1] is BackendTransferDirection.SYNCHRONIZE]
    return BackendTransferCounters(
        host_to_device_transfers=len(h2d),
        device_to_host_transfers=len(d2h),
        host_to_device_bytes=sum(event[3] for event in h2d),
        device_to_host_bytes=sum(event[3] for event in d2h),
        global_synchronizations=len(sync),
    )


def _assert_trace_matches_plan(
    trace: BackendTransferTrace,
    counters: BackendTransferCounters,
    expected: list[tuple[str, BackendTransferDirection, str | None, int]],
) -> None:
    assert trace.overflow_count == 0
    assert trace.counters() == counters
    assert counters == _counters_from_expected(expected)
    assert [
        (event.barrier, event.direction, event.buffer_name, event.nbytes) for event in trace.events
    ] == expected


def _getter_probes(backend: Any) -> None:
    """Exercise every legacy G1 cache adapter that the host profile supports."""
    _ = backend.get_base_pos()
    _ = backend.get_base_quat()
    _ = backend.get_base_lin_vel()
    _ = backend.get_base_ang_vel()
    _ = backend.get_dof_pos()
    _ = backend.get_dof_vel()
    _ = backend.get_sensor_data("torso_upvector")
    _ = backend.get_sensor_data("pelvis_local_linvel")
    _ = backend.get_sensor_data("torso_gyro")


def test_host_profile_transfer_count_matches_bound_plan() -> None:
    """Counters, fixed profiler trace, and public profile agree exactly on CUDA."""
    backend = _backend()
    profile = backend.get_transfer_profile()
    assert profile.name == "mjwarp-host-cache-v1"
    assert profile.execution_profile == "host_numpy"
    buffer_nbytes = {buffer.name: buffer.nbytes for buffer in backend.get_transfer_buffers()}
    declared = {
        name
        for barrier in profile.barriers
        for name in (*barrier.host_to_device_buffers, *barrier.device_to_host_buffers)
    }
    assert set(buffer_nbytes) == declared

    # Construction is itself an explicit host-cache lifecycle barrier.  Check
    # it before clearing telemetry for the independently scoped reset/step
    # measurements below, so every barrier declared by the profile is covered.
    _assert_trace_matches_plan(
        backend.get_transfer_trace(),
        backend.get_transfer_counters(),
        _expected_events(
            profile,
            barrier_name="init",
            repeats=1,
            buffer_nbytes=buffer_nbytes,
        ),
    )

    qpos = np.tile(backend.get_keyframe_qpos("stand"), (_NUM_ENVS, 1)).astype(np.float32)
    qvel = np.zeros((_NUM_ENVS, backend.get_init_qvel().size), dtype=np.float32)
    rows = np.arange(_NUM_ENVS, dtype=np.int32)

    backend.reset_transfer_telemetry()
    backend.set_state(rows, qpos, qvel)
    reset_counters = backend.get_transfer_counters()
    reset_trace = backend.get_transfer_trace()
    _assert_trace_matches_plan(
        reset_trace,
        reset_counters,
        _expected_events(
            profile,
            barrier_name="reset",
            repeats=1,
            buffer_nbytes=buffer_nbytes,
        ),
    )

    # Reading more public legacy views is a pure host-cache operation.  It
    # must neither add a transfer nor mutate the profiler trace.
    before_getters = backend.get_transfer_counters()
    before_getter_trace = backend.get_transfer_trace()
    _getter_probes(backend)
    assert backend.get_transfer_counters() == before_getters
    assert backend.get_transfer_trace() == before_getter_trace

    backend.reset_transfer_telemetry()
    ctrl = np.zeros((_NUM_ENVS, backend.num_actuators), dtype=np.float32)
    for _ in range(3):
        backend.step(ctrl)
    step_counters = backend.get_transfer_counters()
    step_trace = backend.get_transfer_trace()
    _assert_trace_matches_plan(
        step_trace,
        step_counters,
        _expected_events(
            profile,
            barrier_name="step",
            repeats=3,
            buffer_nbytes=buffer_nbytes,
        ),
    )

    before_getters = backend.get_transfer_counters()
    before_getter_trace = backend.get_transfer_trace()
    _getter_probes(backend)
    assert backend.get_transfer_counters() == before_getters
    assert backend.get_transfer_trace() == before_getter_trace
