"""Acceptance tests for the compiled G1 host-Numba executor.

The reference and fused candidates receive independent physics instances.  The
test never treats the fused kernel's arrays as an oracle: generated vectors,
the separate reference executor, terminal/final observations, and lifecycle
trace all have to agree through the public ``ManagedEnvState`` contract.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np
import pytest

from unilab.base.backend import (
    BoundMutationValueBufferGroup,
    BoundMutationValueBuffers,
    RowSelection,
    SimBackend,
    StateBatch,
    TypedBackendMutationBatch,
    create_backend,
    env_backend_kwargs,
)
from unilab.envs.locomotion.g1 import managed_fused as fused_module
from unilab.envs.locomotion.g1.joystick import G1WalkEnv, G1WalkFlatCfg, G1WalkRewardConfig
from unilab.envs.locomotion.g1.managed_fused import (
    G1ManagedFusedError,
    G1ManagedFusedKernel,
    compile_g1_managed_fused_task,
    create_g1_managed_fused_runtime,
)
from unilab.envs.locomotion.g1.managed_reference import (
    G1ManagedReferenceKernel,
    compile_g1_managed_reference_task,
    create_g1_managed_reference_runtime,
)
from unilab.envs.locomotion.g1.managed_schema import (
    G1_STATE_KEYS,
    build_g1_kernel_config,
)
from unilab.manager import ManagedEnvState, ManagedReferenceRuntime, ManagedRuntimeError

_ATOL = 1.0e-6
_RTOL = 1.0e-5


def _reward_config() -> G1WalkRewardConfig:
    return G1WalkRewardConfig(
        scales={
            "tracking_lin_vel": 2.0,
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
            "alive": 0.1,
        },
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.754,
        min_base_height=0.55,
        max_tilt_deg=25.0,
        pose_weights=[0.01, 1.0, 5.0, 0.01, 5.0, 5.0, 0.01, 1.0, 5.0, 0.01, 5.0, 5.0] + [50.0] * 17,
    )


def _cfg(*, noise_level: float, max_episode_seconds: float | None = None) -> G1WalkFlatCfg:
    cfg = G1WalkFlatCfg(reward_config=_reward_config())
    cfg.max_episode_seconds = max_episode_seconds
    cfg.curriculum.enabled = False
    cfg.domain_rand.randomize_kp = False
    cfg.domain_rand.randomize_kd = False
    cfg.noise_config.level = noise_level
    cfg.noise_config.seed = 41 if noise_level > 0.0 else None
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


def _create_pair(
    cfg: G1WalkFlatCfg, *, num_envs: int, reset_seed: int
) -> tuple[SimBackend, ManagedReferenceRuntime, SimBackend, ManagedReferenceRuntime]:
    reference_cfg = deepcopy(cfg)
    fused_cfg = deepcopy(cfg)
    reference_backend = _backend(reference_cfg, num_envs=num_envs)
    fused_backend = _backend(fused_cfg, num_envs=num_envs)
    reference = create_g1_managed_reference_runtime(
        backend=reference_backend,
        cfg=reference_cfg,
        reset_seed=reset_seed,
        record_lifecycle=True,
    )
    fused = create_g1_managed_fused_runtime(
        backend=fused_backend,
        cfg=fused_cfg,
        reset_seed=reset_seed,
        record_lifecycle=True,
    )
    return reference_backend, reference, fused_backend, fused


def _assert_array_close(*, expected: np.ndarray, actual: np.ndarray, label: str) -> None:
    assert expected.shape == actual.shape, f"{label}: shape mismatch"
    if expected.dtype.kind == "b" or actual.dtype.kind == "b":
        assert np.array_equal(expected, actual), f"{label}: boolean mismatch"
        return
    np.testing.assert_allclose(expected, actual, atol=_ATOL, rtol=_RTOL, err_msg=label)


def _assert_state_close(*, expected: ManagedEnvState, actual: ManagedEnvState, label: str) -> None:
    assert tuple(expected.obs) == tuple(actual.obs), f"{label}: observation keys differ"
    for key in expected.obs:
        _assert_array_close(
            expected=expected.obs[key], actual=actual.obs[key], label=f"{label}.obs[{key}]"
        )
    _assert_array_close(expected=expected.reward, actual=actual.reward, label=f"{label}.reward")
    _assert_array_close(
        expected=expected.terminated, actual=actual.terminated, label=f"{label}.terminated"
    )
    _assert_array_close(
        expected=expected.truncated, actual=actual.truncated, label=f"{label}.truncated"
    )
    expected_mask = np.asarray(expected.info["_final_observation"], dtype=bool)
    actual_mask = np.asarray(actual.info["_final_observation"], dtype=bool)
    _assert_array_close(expected=expected_mask, actual=actual_mask, label=f"{label}.final_mask")
    if np.any(expected_mask):
        assert expected.final_observation is not None, (
            f"{label}: reference final observation missing"
        )
        assert actual.final_observation is not None, f"{label}: fused final observation missing"
        for key in expected.final_observation:
            _assert_array_close(
                expected=expected.final_observation[key][expected_mask],
                actual=actual.final_observation[key][actual_mask],
                label=f"{label}.final_observation[{key}]",
            )
            _assert_array_close(
                expected=expected.info["final_observation"][key][expected_mask],
                actual=actual.info["final_observation"][key][actual_mask],
                label=f"{label}.compat_final_observation[{key}]",
            )
    else:
        assert expected.final_observation is None
        assert actual.final_observation is None
    expected_log = expected.info["log"]
    actual_log = actual.info["log"]
    assert set(expected_log) == set(actual_log), f"{label}: reward log keys differ"
    for key in expected_log:
        _assert_array_close(
            expected=np.asarray(expected_log[key]),
            actual=np.asarray(actual_log[key]),
            label=f"{label}.log[{key}]",
        )


def _generated_actions(*, num_envs: int, action_dim: int) -> tuple[np.ndarray, ...]:
    base = np.arange(action_dim, dtype=np.float32)
    offsets = np.arange(num_envs, dtype=np.float32)[:, None] * 0.002
    return (
        np.ascontiguousarray(np.zeros((num_envs, action_dim), dtype=np.float32)),
        np.ascontiguousarray(np.sin(base[None, :] * 0.17 + offsets) * 0.08),
        np.ascontiguousarray(
            np.clip(
                np.linspace(-1.0, 1.0, action_dim, dtype=np.float32)[None, :] + offsets, -1.0, 1.0
            )
        ),
    )


def _force_terminal_rows(*, backend: SimBackend, rows: np.ndarray, min_height: float) -> None:
    default_qpos = backend.get_keyframe_qpos("stand")
    default_qvel = backend.get_init_qvel()
    qpos = np.broadcast_to(default_qpos, (len(rows), default_qpos.size)).copy()
    qvel = np.broadcast_to(default_qvel, (len(rows), default_qvel.size)).copy()
    qpos[:, 2] = min_height - 0.2
    backend.set_state(rows.astype(np.int32, copy=False), qpos, qvel)


@contextmanager
def _forbid_fused_hot_path_fallbacks(backend: SimBackend) -> Iterator[None]:
    """Make accidental getter/selector/asset/reference delegation observable."""

    with ExitStack() as stack:
        for method in (
            "get_actuator_names",
            "get_body_ids",
            "get_joint_dof_pos_indices",
            "get_joint_dof_vel_indices",
            "get_sensor_ids",
            "get_keyframe_qpos",
            "get_init_qvel",
            "get_base_pos",
            "get_base_quat",
            "get_base_lin_vel",
            "get_base_ang_vel",
            "get_dof_pos",
            "get_dof_vel",
            "get_sensor_data",
            "set_state",
        ):
            stack.enter_context(
                patch.object(backend, method, side_effect=AssertionError(f"fallback: {method}"))
            )
        for method in (
            "apply_action",
            "evaluate_termination",
            "evaluate_reward",
            "evaluate_metrics",
            "write_observations",
        ):
            stack.enter_context(
                patch.object(
                    G1ManagedReferenceKernel,
                    method,
                    side_effect=AssertionError(f"reference delegation: {method}"),
                )
            )
        stack.enter_context(
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset read fallback"))
        )
        stack.enter_context(
            patch.object(Path, "read_text", side_effect=AssertionError("asset read fallback"))
        )
        yield


def _cleanup(*backends: SimBackend) -> None:
    for backend in backends:
        backend.cleanup_scene_assets()


def test_fused_executor_matches_reference_generated_vectors() -> None:
    """Normal, boundary, reset, noise, logs, and lifecycle match independently."""

    for reset_seed, num_envs, noise_level in (
        (0, 1, 0.0),
        (1, 128, 0.0),
        (2, 4096, 0.35),
    ):
        cfg = _cfg(noise_level=noise_level)
        reference_backend, reference, fused_backend, fused = _create_pair(
            cfg, num_envs=num_envs, reset_seed=reset_seed
        )
        try:
            expected = reference.init_state()
            actual = fused.init_state()
            _assert_state_close(
                expected=expected,
                actual=actual,
                label=f"seed={reset_seed}/batch={num_envs}.initial",
            )
            assert reference.plan.executor_key != fused.plan.executor_key
            assert reference.plan.fingerprint != fused.plan.fingerprint
            assert reference.plan.backend_io.state_fields == fused.plan.backend_io.state_fields
            assert reference.plan.backend_io.control == fused.plan.backend_io.control
            assert reference.plan.policy_abi == fused.plan.policy_abi
            assert reference.last_trace == fused.last_trace
            for index, actions in enumerate(_generated_actions(num_envs=num_envs, action_dim=29)):
                expected = reference.step(actions.copy())
                with _forbid_fused_hot_path_fallbacks(fused_backend):
                    actual = fused.step(actions.copy())
                _assert_state_close(
                    expected=expected,
                    actual=actual,
                    label=f"seed={reset_seed}/batch={num_envs}.normal={index}",
                )
                assert reference.last_trace == fused.last_trace
                assert not np.any(expected.terminated | expected.truncated)

            if num_envs > 1:
                partial = np.asarray((1, num_envs - 1), dtype=np.int32)
                _force_terminal_rows(
                    backend=reference_backend,
                    rows=partial,
                    min_height=cfg.reward_config.min_base_height,  # type: ignore[union-attr]
                )
                _force_terminal_rows(
                    backend=fused_backend,
                    rows=partial,
                    min_height=cfg.reward_config.min_base_height,  # type: ignore[union-attr]
                )
                actions = _generated_actions(num_envs=num_envs, action_dim=29)[1]
                expected = reference.step(actions.copy())
                with _forbid_fused_hot_path_fallbacks(fused_backend):
                    actual = fused.step(actions.copy())
                _assert_state_close(
                    expected=expected,
                    actual=actual,
                    label=f"seed={reset_seed}/batch={num_envs}.partial_reset",
                )
                assert reference.last_trace == fused.last_trace

            all_rows = np.arange(num_envs, dtype=np.int32)
            _force_terminal_rows(
                backend=reference_backend,
                rows=all_rows,
                min_height=cfg.reward_config.min_base_height,  # type: ignore[union-attr]
            )
            _force_terminal_rows(
                backend=fused_backend,
                rows=all_rows,
                min_height=cfg.reward_config.min_base_height,  # type: ignore[union-attr]
            )
            actions = _generated_actions(num_envs=num_envs, action_dim=29)[2]
            expected = reference.step(actions.copy())
            with _forbid_fused_hot_path_fallbacks(fused_backend):
                actual = fused.step(actions.copy())
            _assert_state_close(
                expected=expected,
                actual=actual,
                label=f"seed={reset_seed}/batch={num_envs}.all_reset",
            )
            assert reference.last_trace == fused.last_trace
        finally:
            _cleanup(reference_backend, fused_backend)

    # A one-step timeout verifies final-observation/autoreset behavior through
    # the same plan with a different lifecycle schedule.
    timeout_cfg = _cfg(noise_level=0.2, max_episode_seconds=0.02)
    reference_backend, reference, fused_backend, fused = _create_pair(
        timeout_cfg, num_envs=4, reset_seed=7
    )
    try:
        _assert_state_close(
            expected=reference.init_state(), actual=fused.init_state(), label="timeout.initial"
        )
        actions = _generated_actions(num_envs=4, action_dim=29)[1]
        expected = reference.step(actions.copy())
        with _forbid_fused_hot_path_fallbacks(fused_backend):
            actual = fused.step(actions.copy())
        _assert_state_close(expected=expected, actual=actual, label="timeout.step")
        assert np.all(expected.truncated)
        assert reference.last_trace == fused.last_trace
    finally:
        _cleanup(reference_backend, fused_backend)

    # The phase-3 reference differential and this executor differential meet
    # at one independent three-way vector: no state, reward, log, or policy
    # observation is allowed to become a two-implementation agreement only.
    triple_cfg = _cfg(noise_level=0.0)
    legacy = G1WalkEnv(deepcopy(triple_cfg), num_envs=1, backend_type="mujoco")
    reference_backend, reference, fused_backend, fused = _create_pair(
        triple_cfg, num_envs=1, reset_seed=19
    )
    try:
        np.random.seed(19)
        legacy.init_state()
        reference.init_state()
        fused.init_state()
        action = _generated_actions(num_envs=1, action_dim=29)[2]
        expected = legacy.step(action.copy())
        reference_state = reference.step(action.copy())
        with _forbid_fused_hot_path_fallbacks(fused_backend):
            fused_state = fused.step(action.copy())
        _assert_state_close(expected=expected, actual=reference_state, label="triple.reference")
        _assert_state_close(expected=expected, actual=fused_state, label="triple.fused")
    finally:
        _cleanup(legacy._backend, reference_backend, fused_backend)

    # The task math is keyed by a numeric cold-bound term table, so dictionary
    # ordering must not change values, logs, reset behavior, or observations.
    original_cfg = _cfg(noise_level=0.0)
    permuted_cfg = deepcopy(original_cfg)
    assert permuted_cfg.reward_config is not None
    permuted_cfg.reward_config.scales = dict(reversed(permuted_cfg.reward_config.scales.items()))
    original_backend = _backend(original_cfg, num_envs=4)
    permuted_backend = _backend(permuted_cfg, num_envs=4)
    try:
        original = create_g1_managed_fused_runtime(
            backend=original_backend, cfg=original_cfg, reset_seed=13
        )
        permuted = create_g1_managed_fused_runtime(
            backend=permuted_backend, cfg=permuted_cfg, reset_seed=13
        )
        _assert_state_close(
            expected=original.init_state(),
            actual=permuted.init_state(),
            label="term_permutation.initial",
        )
        actions = _generated_actions(num_envs=4, action_dim=29)[2]
        _assert_state_close(
            expected=original.step(actions.copy()),
            actual=permuted.step(actions.copy()),
            label="term_permutation.step",
        )
    finally:
        _cleanup(original_backend, permuted_backend)


def test_fused_executor_uses_serial_numba_hot_kernels_for_host_profile() -> None:
    """Guard the measured host strategy against accidental ``prange`` regression."""

    dispatchers = (
        fused_module._apply_action_kernel,
        fused_module._write_observations_kernel,
        fused_module._compute_terminal_kernel,
        fused_module._copy_rows_kernel,
        fused_module._gather_task_rows_kernel,
        fused_module._complete_reset_task_state_kernel,
    )
    assert all(
        dispatcher.targetoptions.get("parallel", False) is False for dispatcher in dispatchers
    )


def test_fused_terminal_state_views_are_validated_once_per_borrowed_batch() -> None:
    """Terminal reward and observation share one valid typed-state mapping.

    The cache remains keyed by the strong ``StateBatch`` object, rather than
    its integer id, and the kernel still invokes ``assert_valid`` for each
    lifecycle consumer.  This counts only field descriptor reads: a terminal
    step needs the eleven G1 fields once for termination/reward math and must
    not re-read them while writing its terminal observation.
    """

    cfg = _cfg(noise_level=0.0)
    backend = _backend(deepcopy(cfg), num_envs=2)
    try:
        runtime = create_g1_managed_fused_runtime(backend=backend, cfg=cfg, reset_seed=17)
        runtime.init_state()
        original_buffer_at = StateBatch.buffer_at
        field_reads: list[int] = []

        def _record_buffer_at(state: StateBatch, field_index: int):
            field_reads.append(field_index)
            return original_buffer_at(state, field_index)

        actions = _generated_actions(num_envs=2, action_dim=29)[1]
        with patch.object(StateBatch, "buffer_at", new=_record_buffer_at):
            runtime.step(actions)

        state_indices = dict(runtime.kernel_binding.state_field_indices)
        assert field_reads == [state_indices[key] for key in G1_STATE_KEYS]
    finally:
        _cleanup(backend)


def test_fused_reset_uses_cold_bound_complete_mutation_window() -> None:
    """Reset descriptor construction stays cold-bound for the fused profile."""

    cfg = _cfg(noise_level=0.0)
    backend = _backend(deepcopy(cfg), num_envs=4)
    try:
        runtime = create_g1_managed_fused_runtime(backend=backend, cfg=cfg, reset_seed=29)
        runtime.init_state()
        task = runtime.task_state
        assert task is not None
        buffer_set = getattr(task, "reset_value_buffer_set")
        assert isinstance(buffer_set, BoundMutationValueBuffers)
        assert all(isinstance(group, BoundMutationValueBufferGroup) for group in buffer_set.groups)
        assert len(buffer_set.groups) == 2
        position_values = getattr(task, "reset_dof_position_values")
        velocity_values = getattr(task, "reset_dof_velocity_values")
        position_indices = runtime._kernel._dof_position_reset_indices  # type: ignore[attr-defined]
        velocity_indices = runtime._kernel._dof_velocity_reset_indices  # type: ignore[attr-defined]
        assert position_indices is not None
        assert velocity_indices is not None
        for dof_index, field_index in enumerate(position_indices):
            assert np.shares_memory(buffer_set.buffers[field_index], position_values[dof_index])
            assert buffer_set.buffers[field_index].flags.c_contiguous
        for dof_index, field_index in enumerate(velocity_indices):
            assert np.shares_memory(buffer_set.buffers[field_index], velocity_values[dof_index])
            assert buffer_set.buffers[field_index].flags.c_contiguous
        mutation_plan = runtime.kernel_binding.mutation_plan
        assert mutation_plan is not None
        mutation_runtime = backend._host_mutation_plans[mutation_plan.fingerprint]  # type: ignore[attr-defined]
        prepared = mutation_runtime._prepared_buffer_sets[id(buffer_set)]
        assert prepared.owner is buffer_set
        assert len(prepared.groups) == 2
        assert len(prepared.individual) == 4
        rows = RowSelection.selected(backend.num_envs, (3, 1))
        request = runtime._kernel.prepare_reset(rows=rows, task_state=task)  # type: ignore[attr-defined]
        assert isinstance(request.mutation_batch, TypedBackendMutationBatch)
        assert request.mutation_batch.state.values == ()
        window = request.mutation_batch.state.bound_buffer_window
        assert window is not None
        assert window.buffers is buffer_set
        assert window.rows == rows
        assert window.plan == runtime.kernel_binding.mutation_plan
    finally:
        _cleanup(backend)


def test_fused_executor_never_silently_falls_back() -> None:
    """Unsupported/faulted dispatch fails before physics or uses only fused math."""

    cfg = _cfg(noise_level=0.0)
    backend = _backend(deepcopy(cfg), num_envs=2)
    try:
        runtime = create_g1_managed_fused_runtime(backend=backend, cfg=cfg, reset_seed=3)
        dispatchers = (
            fused_module._apply_action_kernel,
            fused_module._write_observations_kernel,
            fused_module._compute_terminal_kernel,
            fused_module._copy_rows_kernel,
            fused_module._gather_task_rows_kernel,
            fused_module._complete_reset_task_state_kernel,
        )
        signatures_at_bind = tuple(len(dispatcher.signatures) for dispatcher in dispatchers)
        runtime.init_state()
        assert tuple(len(dispatcher.signatures) for dispatcher in dispatchers) == signatures_at_bind
        valid = _generated_actions(num_envs=2, action_dim=29)[1]
        with _forbid_fused_hot_path_fallbacks(backend):
            runtime.step(valid.copy())
        assert tuple(len(dispatcher.signatures) for dispatcher in dispatchers) == signatures_at_bind
        for invalid in (np.nan, np.inf, -np.inf):
            actions = valid.copy()
            actions[0, 0] = invalid
            with pytest.raises(ManagedRuntimeError, match="non-finite"):
                runtime.step(actions)

        # A terminal state with one non-finite public typed field must fail at
        # fused dispatch rather than producing a reference/getter fallback.
        original_step = backend.step_batch
        root_index = dict(runtime.kernel_binding.state_field_indices)["g1.root.position"]

        def _nonfinite_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            values = result.terminal_state.buffer_at(root_index).handle
            assert isinstance(values, np.ndarray)
            values.flags.writeable = True
            values[0, 2] = np.nan
            values.flags.writeable = False
            return result

        with patch.object(backend, "step_batch", side_effect=_nonfinite_step):
            with pytest.raises(G1ManagedFusedError, match="non-finite typed state"):
                runtime.step(valid.copy())
    finally:
        _cleanup(backend)

    unsupported_cfg = _cfg(noise_level=0.0)
    unsupported_cfg.domain_rand.randomize_kp = True
    unsupported_backend = _backend(deepcopy(unsupported_cfg), num_envs=1)
    try:
        with patch.object(
            unsupported_backend,
            "get_actuator_names",
            side_effect=AssertionError("unsupported profile reached binding"),
        ):
            with pytest.raises(G1ManagedFusedError, match="typed DR/Event"):
                create_g1_managed_fused_runtime(backend=unsupported_backend, cfg=unsupported_cfg)
    finally:
        _cleanup(unsupported_backend)

    unavailable_backend = _backend(_cfg(noise_level=0.0), num_envs=1)
    try:
        with patch.object(fused_module, "NUMBA_AVAILABLE", False):
            with patch.object(
                unavailable_backend,
                "get_actuator_names",
                side_effect=AssertionError("Numba-unavailable path reached binding"),
            ):
                with pytest.raises(G1ManagedFusedError, match="requires numba"):
                    create_g1_managed_fused_runtime(
                        backend=unavailable_backend,
                        cfg=_cfg(noise_level=0.0),
                    )
    finally:
        _cleanup(unavailable_backend)

    # A foreign executor cannot pass runtime dispatch even if its state/policy
    # shape happens to look G1-compatible.
    wrong_backend = _backend(_cfg(noise_level=0.0), num_envs=1)
    try:
        wrong_cfg = _cfg(noise_level=0.0)
        reference_plan = compile_g1_managed_reference_task(backend=wrong_backend, cfg=wrong_cfg)
        wrong_backend.materialize()
        kernel = G1ManagedFusedKernel(
            build_g1_kernel_config(
                backend=wrong_backend,
                cfg=wrong_cfg,
                reset_seed=0,
                observation_noise_seed=None,
                profile_name="fused executor",
                error_type=G1ManagedFusedError,
            ),
            expected_plan_fingerprint=reference_plan.fingerprint,
        )
        with pytest.raises(ManagedRuntimeError, match="executor_key"):
            ManagedReferenceRuntime(
                backend=wrong_backend,
                plan=reference_plan,
                kernel=kernel,
                max_episode_steps=wrong_cfg.max_episode_steps,
            )
    finally:
        _cleanup(wrong_backend)

    # A stale expected plan fingerprint is a bind-time hard error; it cannot
    # pick a reference kernel or a newly compiled plan at runtime.
    stale_backend = _backend(_cfg(noise_level=0.0), num_envs=1)
    try:
        stale_cfg = _cfg(noise_level=0.0)
        fused_plan = compile_g1_managed_fused_task(backend=stale_backend, cfg=stale_cfg)
        stale_backend.materialize()
        stale_kernel = G1ManagedFusedKernel(
            build_g1_kernel_config(
                backend=stale_backend,
                cfg=stale_cfg,
                reset_seed=0,
                observation_noise_seed=None,
                profile_name="fused executor",
                error_type=G1ManagedFusedError,
            ),
            expected_plan_fingerprint="stale-plan-fingerprint",
        )
        with pytest.raises(G1ManagedFusedError, match="stale or foreign"):
            ManagedReferenceRuntime(
                backend=stale_backend,
                plan=fused_plan,
                kernel=stale_kernel,
                max_episode_steps=stale_cfg.max_episode_steps,
            )
    finally:
        _cleanup(stale_backend)
