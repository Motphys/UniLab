"""Warm-path allocation and address-contract tests for managed G1 fusion.

The test intentionally consumes only public runtime diagnostics and the typed
backend result counters.  It does not inspect MuJoCo model/data/private pool
arrays, so a future backend can satisfy the same contract with a different
storage implementation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from unilab.base.backend import (
    BackendBatchCounters,
    BackendBatchDiagnostics,
    SimBackend,
    create_backend,
    env_backend_kwargs,
)
from unilab.base.backend.batch import (
    BackendStepResult,
    BufferView,
    StateBatch,
    StateBatchLease,
)
from unilab.envs.locomotion.g1.joystick import G1WalkFlatCfg, G1WalkRewardConfig
from unilab.envs.locomotion.g1.managed_fused import create_g1_managed_fused_runtime
from unilab.manager import ManagedReferenceRuntime, ManagedRuntimeError


def _reward_config(*, expanded: bool) -> G1WalkRewardConfig:
    scales: dict[str, float] = {
        "tracking_lin_vel": 2.0,
        "alive": 0.1,
    }
    if expanded:
        scales.update(
            {
                "tracking_ang_vel": 0.2,
                "forward_progress": 0.25,
                "under_speed": -0.1,
                "feet_phase": 1.0,
                "feet_phase_contrast": 0.25,
                "lin_vel_z": -1.0,
                "ang_vel_xy": -0.25,
                "base_height": -500.0,
                "orientation": -5.0,
                "action_rate": -0.01,
                "pose": -0.1,
                "upper_body_pose": -0.05,
                "penalty_close_feet_xy": -0.1,
            }
        )
    return G1WalkRewardConfig(
        scales=scales,
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.754,
        min_base_height=0.55,
        max_tilt_deg=25.0,
        pose_weights=[0.01, 1.0, 5.0, 0.01, 5.0, 5.0, 0.01, 1.0, 5.0, 0.01, 5.0, 5.0] + [50.0] * 17,
    )


def _cfg(*, expanded_terms: bool) -> G1WalkFlatCfg:
    cfg = G1WalkFlatCfg(reward_config=_reward_config(expanded=expanded_terms))
    cfg.max_episode_seconds = None
    cfg.curriculum.enabled = False
    cfg.domain_rand.randomize_kp = False
    cfg.domain_rand.randomize_kd = False
    cfg.noise_config.level = 0.0
    cfg.noise_config.seed = None
    cfg.commands.resampling_time = 0.0
    cfg.commands.heading_command = False
    return cfg


def _backend(cfg: G1WalkFlatCfg, *, num_envs: int) -> SimBackend:
    assert cfg.scene is not None
    return create_backend(
        "mujoco",
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
        push_body_name=cfg.domain_rand.push_body_name,
        **env_backend_kwargs(cfg),
    )


def _runtime(
    *, num_envs: int, expanded_terms: bool, record_lifecycle: bool
) -> tuple[SimBackend, ManagedReferenceRuntime, G1WalkFlatCfg]:
    cfg = _cfg(expanded_terms=expanded_terms)
    backend = _backend(deepcopy(cfg), num_envs=num_envs)
    runtime = create_g1_managed_fused_runtime(
        backend=backend,
        cfg=cfg,
        reset_seed=7,
        record_lifecycle=record_lifecycle,
        enable_stability_instrumentation=True,
    )
    return backend, runtime, cfg


def _force_terminal_rows(*, backend: SimBackend, rows: np.ndarray, min_height: float) -> None:
    """Use the public legacy setup API outside the measured managed hot path."""

    default_qpos = backend.get_keyframe_qpos("stand")
    default_qvel = backend.get_init_qvel()
    qpos = np.broadcast_to(default_qpos, (len(rows), default_qpos.size)).copy()
    qvel = np.broadcast_to(default_qvel, (len(rows), default_qvel.size)).copy()
    qpos[:, 2] = min_height - 0.2
    backend.set_state(rows.astype(np.int32, copy=False), qpos, qvel)


def _diagnostics(runtime: ManagedReferenceRuntime):
    diagnostics = runtime.stability_diagnostics
    assert diagnostics is not None
    assert diagnostics.instrumentation_complete
    assert diagnostics.warm_numeric_allocations == 0
    assert diagnostics.address_churn == 0
    assert diagnostics.backend_reset_counters is not None
    return diagnostics


def _transfer_signature(counters: BackendBatchCounters) -> tuple[int, int, int, int, int]:
    return (
        counters.host_to_device_transfers,
        counters.device_to_host_transfers,
        counters.host_to_device_bytes,
        counters.device_to_host_bytes,
        counters.global_synchronizations,
    )


def _state_with_copied_buffers(state: StateBatch) -> StateBatch:
    """Model a backend that allocates fresh public state storage after warmup.

    This deliberately reconstructs only the public typed ``StateBatch``
    envelope.  It never reaches into a backend-owned plan/cache/lease, and the
    fresh lease is sufficient because the failure is expected before a reset
    barrier consumes the forged terminal state.
    """

    descriptors = tuple(
        BufferView(
            handle=np.array(state.buffer_at(index).handle, copy=True),
            shape=state.buffer_at(index).shape,
            contract=state.buffer_at(index).contract,
        )
        for index in range(len(state.plan.state.fields))
    )
    return StateBatch(
        plan=state.plan,
        rows=state.rows,
        phase=state.phase,
        descriptors=descriptors,
        lease=StateBatchLease(state.plan.backend_instance_id),
    )


@pytest.mark.parametrize(
    "num_envs",
    [128, pytest.param(4096, marks=pytest.mark.slow)],
)
def test_warm_loop_has_stable_addresses_and_allocations(num_envs: int) -> None:
    """No-done/sparse/full reset and trace logging keep warm buffers stable."""

    observed_step_counters: list[BackendBatchCounters] = []
    sparse_reset_counters: list[BackendBatchCounters] = []
    full_reset_counters: list[BackendBatchCounters] = []
    for record_lifecycle in (False, True):
        backend, runtime, cfg = _runtime(
            num_envs=num_envs,
            expanded_terms=True,
            record_lifecycle=record_lifecycle,
        )
        try:
            state = runtime.init_state()
            diagnostics = _diagnostics(runtime)
            baseline_buffers = diagnostics.buffers
            assert diagnostics.state_buffers
            actions = np.zeros((num_envs, 29), dtype=np.float32)

            # Repetitions cover warmed no-done execution and normal reward-log
            # cadence, whose Python metric wrappers are explicitly outside the
            # numeric buffer counter while their task buffers remain stable.
            for _ in range(3):
                state = runtime.step(actions)
                assert not np.any(state.terminated | state.truncated)
                diagnostics = _diagnostics(runtime)
                assert diagnostics.buffers == baseline_buffers
                assert diagnostics.backend_step_counters is not None
                observed_step_counters.append(diagnostics.backend_step_counters)

            assert cfg.reward_config is not None
            sparse_rows = np.asarray((1, num_envs - 1), dtype=np.int32)
            for _ in range(3):
                _force_terminal_rows(
                    backend=backend,
                    rows=sparse_rows,
                    min_height=cfg.reward_config.min_base_height,
                )
                state = runtime.step(actions)
                assert np.array_equal(np.flatnonzero(state.terminated), sparse_rows)
                diagnostics = _diagnostics(runtime)
                assert diagnostics.buffers == baseline_buffers
                assert diagnostics.backend_step_counters is not None
                observed_step_counters.append(diagnostics.backend_step_counters)
                reset_counters = diagnostics.backend_reset_counters
                assert reset_counters is not None
                sparse_reset_counters.append(reset_counters)

            all_rows = np.arange(num_envs, dtype=np.int32)
            for _ in range(3):
                _force_terminal_rows(
                    backend=backend,
                    rows=all_rows,
                    min_height=cfg.reward_config.min_base_height,
                )
                state = runtime.step(actions)
                assert np.all(state.terminated)
                diagnostics = _diagnostics(runtime)
                assert diagnostics.buffers == baseline_buffers
                assert diagnostics.backend_step_counters is not None
                observed_step_counters.append(diagnostics.backend_step_counters)
                reset_counters = diagnostics.backend_reset_counters
                assert reset_counters is not None
                full_reset_counters.append(reset_counters)
        finally:
            backend.cleanup_scene_assets()

    assert observed_step_counters
    assert all(counter == observed_step_counters[0] for counter in observed_step_counters)
    assert sparse_reset_counters
    assert full_reset_counters
    sparse_reset = sparse_reset_counters[0]
    assert all(counter == sparse_reset for counter in sparse_reset_counters)
    assert all(counter == full_reset_counters[0] for counter in full_reset_counters)
    assert _transfer_signature(observed_step_counters[0]) == (0, 0, 0, 0, 0)
    assert _transfer_signature(sparse_reset) == (0, 0, 0, 0, 0)

    # Adding independent fused reward terms changes only task-owned reward
    # scratch width.  It must not add state materialization or transfer work.
    term_signatures: list[tuple[int, int, int, int, int]] = []
    term_counters: list[BackendBatchCounters] = []
    for expanded_terms in (False, True):
        backend, runtime, _ = _runtime(
            num_envs=num_envs,
            expanded_terms=expanded_terms,
            record_lifecycle=False,
        )
        try:
            runtime.init_state()
            runtime.step(np.zeros((num_envs, 29), dtype=np.float32))
            diagnostics = _diagnostics(runtime)
            assert diagnostics.backend_step_counters is not None
            counters = diagnostics.backend_step_counters
            term_signatures.append(_transfer_signature(counters))
            term_counters.append(counters)
            assert counters.state_materializations == 1
            assert counters.dynamic_getter_calls == 0
            assert counters.selector_resolutions == 0
            assert counters.asset_metadata_reads == 0
            assert counters.registry_lookups == 0
        finally:
            backend.cleanup_scene_assets()
    assert term_signatures[0] == term_signatures[1]
    assert term_counters[0] == term_counters[1]


def test_fused_stability_instrumentation_fails_closed() -> None:
    """Buffer replacement and incomplete backend telemetry cannot be hidden."""

    backend, runtime, _ = _runtime(num_envs=4, expanded_terms=False, record_lifecycle=False)
    try:
        runtime.init_state()
        task = runtime.task_state
        assert task is not None
        # The task state is intentionally public only as a diagnostic object;
        # replacement models an accidental warm allocation in a future kernel.
        task.commands = np.empty_like(task.commands)  # type: ignore[attr-defined]
        with pytest.raises(ManagedRuntimeError, match="warm buffer stability violated"):
            runtime.step(np.zeros((4, 29), dtype=np.float32))
    finally:
        backend.cleanup_scene_assets()

    backend, runtime, _ = _runtime(num_envs=4, expanded_terms=False, record_lifecycle=False)
    try:
        runtime.init_state()
        actions = np.zeros((4, 29), dtype=np.float32)
        # Establish the terminal-state address baseline before modelling a
        # post-warm backend materialization that silently allocates copies.
        runtime.step(actions)
        original_step = backend.step_batch

        def _address_churn_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            assert isinstance(result, BackendStepResult)
            return replace(result, terminal_state=_state_with_copied_buffers(result.terminal_state))

        with patch.object(backend, "step_batch", side_effect=_address_churn_step):
            with pytest.raises(ManagedRuntimeError, match="StateBatch address changed"):
                runtime.step(actions)
    finally:
        backend.cleanup_scene_assets()

    backend, runtime, _ = _runtime(num_envs=4, expanded_terms=False, record_lifecycle=False)
    try:
        runtime.init_state()
        original_step = backend.step_batch

        def _incomplete_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            assert isinstance(result, BackendStepResult)
            return replace(
                result,
                diagnostics=BackendBatchDiagnostics(
                    counters=BackendBatchCounters(instrumentation_complete=False)
                ),
            )

        with patch.object(backend, "step_batch", side_effect=_incomplete_step):
            with pytest.raises(ManagedRuntimeError, match="requires complete backend"):
                runtime.step(np.zeros((4, 29), dtype=np.float32))
    finally:
        backend.cleanup_scene_assets()

    backend, runtime, _ = _runtime(num_envs=4, expanded_terms=False, record_lifecycle=False)
    try:
        original_reset = backend.reset_batch

        def _incomplete_reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            return replace(
                result,
                diagnostics=BackendBatchDiagnostics(
                    counters=BackendBatchCounters(instrumentation_complete=False)
                ),
            )

        with patch.object(backend, "reset_batch", side_effect=_incomplete_reset):
            with pytest.raises(ManagedRuntimeError, match="requires complete backend"):
                runtime.init_state()
    finally:
        backend.cleanup_scene_assets()
