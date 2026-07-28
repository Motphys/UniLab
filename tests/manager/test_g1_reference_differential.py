"""Independent numerical oracle for the managed G1 host-reference kernel.

The hand-written :class:`G1WalkEnv` is deliberately kept as the expected
implementation in this test.  The managed candidate gets a separate MuJoCo
backend, so matching observations alone cannot be explained by shared task
state, a legacy getter, or a reset fallback.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np
import pytest

from unilab.base.backend import SimBackend, create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.g1.joystick import G1WalkEnv, G1WalkFlatCfg, G1WalkRewardConfig
from unilab.envs.locomotion.g1.managed_reference import (
    G1ManagedReferenceError,
    create_g1_managed_reference_runtime,
)
from unilab.manager import ManagedLifecyclePhase, ManagedReferenceRuntime

_ATOL = 1.0e-6
_RTOL = 1.0e-5


def _reward_config() -> G1WalkRewardConfig:
    """Use the common MuJoCo owner reward slice supported by this child issue."""

    return G1WalkRewardConfig(
        scales={
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.2,
            "feet_phase": 1.0,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.25,
            "base_height": -500.0,
            "orientation": -5.0,
            "action_rate": -0.01,
            "pose": -0.1,
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


def _cfg(
    *,
    max_episode_seconds: float | None,
    observation_noise_level: float,
    observation_noise_seed: int | None,
) -> G1WalkFlatCfg:
    cfg = G1WalkFlatCfg(reward_config=_reward_config())
    cfg.max_episode_seconds = max_episode_seconds
    cfg.curriculum.enabled = False
    cfg.domain_rand.randomize_kp = False
    cfg.domain_rand.randomize_kd = False
    cfg.noise_config.level = observation_noise_level
    cfg.noise_config.seed = observation_noise_seed
    cfg.commands.resampling_time = 0.0
    cfg.commands.heading_command = False
    return cfg


def _create_pair(
    cfg: G1WalkFlatCfg,
    *,
    num_envs: int,
    reset_seed: int,
) -> tuple[G1WalkEnv, SimBackend, ManagedReferenceRuntime]:
    """Build isolated legacy-oracle and managed-candidate physics instances."""

    legacy = G1WalkEnv(deepcopy(cfg), num_envs=num_envs, backend_type="mujoco")
    manager_cfg = deepcopy(cfg)
    assert manager_cfg.scene is not None
    managed_backend = create_backend(
        "mujoco",
        manager_cfg.scene,
        num_envs,
        manager_cfg.sim_dt,
        base_name=manager_cfg.asset.base_name,
        push_body_name=manager_cfg.domain_rand.push_body_name,
        **env_backend_kwargs(manager_cfg),
    )
    managed = create_g1_managed_reference_runtime(
        backend=managed_backend,
        cfg=manager_cfg,
        reset_seed=reset_seed,
        record_lifecycle=True,
    )
    return legacy, managed_backend, managed


def _assert_close(*, expected: np.ndarray, actual: np.ndarray, label: str) -> None:
    """Fail with the first mismatch and global max error, not only a shape diff."""

    if expected.shape != actual.shape:
        raise AssertionError(
            f"{label}: shape mismatch expected={expected.shape}, actual={actual.shape}"
        )
    if expected.dtype.kind == "b" or actual.dtype.kind == "b":
        mismatches = np.argwhere(expected != actual)
        if mismatches.size:
            first = tuple(int(value) for value in mismatches[0])
            raise AssertionError(
                f"{label}: first boolean mismatch at {first}: "
                f"expected={expected[first]!r}, actual={actual[first]!r}; "
                f"count={len(mismatches)}"
            )
        return
    close = np.isclose(expected, actual, atol=_ATOL, rtol=_RTOL, equal_nan=False)
    if bool(np.all(close)):
        return
    mismatches = np.argwhere(~close)
    first = tuple(int(value) for value in mismatches[0])
    max_error = float(np.max(np.abs(expected - actual)))
    raise AssertionError(
        f"{label}: first mismatch at {first}: expected={expected[first]!r}, "
        f"actual={actual[first]!r}, abs_error={abs(expected[first] - actual[first]):.7g}, "
        f"max_abs_error={max_error:.7g}, atol={_ATOL}, rtol={_RTOL}"
    )


def _assert_state_matches(*, expected: NpEnvState, actual: NpEnvState, label: str) -> None:
    assert tuple(expected.obs) == tuple(actual.obs), f"{label}: observation group keys differ"
    for key in expected.obs:
        _assert_close(
            expected=expected.obs[key],
            actual=actual.obs[key],
            label=f"{label}.obs[{key}]",
        )
    _assert_close(expected=expected.reward, actual=actual.reward, label=f"{label}.reward")
    _assert_close(
        expected=expected.terminated,
        actual=actual.terminated,
        label=f"{label}.terminated",
    )
    _assert_close(
        expected=expected.truncated,
        actual=actual.truncated,
        label=f"{label}.truncated",
    )

    expected_mask = np.asarray(expected.info["_final_observation"], dtype=bool)
    actual_mask = np.asarray(actual.info["_final_observation"], dtype=bool)
    _assert_close(
        expected=expected_mask,
        actual=actual_mask,
        label=f"{label}.final_observation_mask",
    )
    if not np.any(expected_mask):
        assert expected.final_observation is None, (
            f"{label}: legacy unexpectedly has final observation"
        )
        assert actual.final_observation is None, (
            f"{label}: manager unexpectedly has final observation"
        )
    else:
        assert expected.final_observation is not None, f"{label}: legacy lacks final observation"
        assert actual.final_observation is not None, f"{label}: manager lacks final observation"
        for key in expected.final_observation:
            _assert_close(
                expected=expected.final_observation[key][expected_mask],
                actual=actual.final_observation[key][actual_mask],
                label=f"{label}.final_observation[{key}]",
            )
            _assert_close(
                expected=expected.info["final_observation"][key][expected_mask],
                actual=actual.info["final_observation"][key][actual_mask],
                label=f"{label}.compat_final_observation[{key}]",
            )

    expected_log = expected.info.get("log", {})
    actual_log = actual.info.get("log", {})
    assert tuple(expected_log) == tuple(actual_log), (
        f"{label}: reward log keys differ expected={tuple(expected_log)}, actual={tuple(actual_log)}"
    )
    for key in expected_log:
        _assert_close(
            expected=np.asarray(expected_log[key]),
            actual=np.asarray(actual_log[key]),
            label=f"{label}.log[{key}]",
        )


@contextmanager
def _forbid_managed_hot_path_fallbacks(backend: SimBackend) -> Iterator[None]:
    """Prove the candidate cannot consult legacy selector/getter/reset APIs."""

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
        stack.enter_context(
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset read fallback"))
        )
        stack.enter_context(
            patch.object(Path, "read_text", side_effect=AssertionError("asset read fallback"))
        )
        yield


def _step_pair(
    *,
    legacy: G1WalkEnv,
    managed: ManagedReferenceRuntime,
    managed_backend: SimBackend,
    actions: np.ndarray,
    label: str,
) -> tuple[NpEnvState, NpEnvState]:
    expected = legacy.step(actions.copy())
    with _forbid_managed_hot_path_fallbacks(managed_backend):
        actual = managed.step(actions.copy())
    _assert_state_matches(expected=expected, actual=actual, label=label)
    return expected, actual


def _assert_trace(
    runtime: ManagedReferenceRuntime,
    *,
    done_rows: tuple[int, ...] | None,
    label: str,
) -> None:
    expected_phases = [
        ManagedLifecyclePhase.ACTION,
        ManagedLifecyclePhase.PRE_PHYSICS,
        ManagedLifecyclePhase.PHYSICS,
        ManagedLifecyclePhase.TERMINATION,
        ManagedLifecyclePhase.REWARD,
        ManagedLifecyclePhase.METRIC,
        ManagedLifecyclePhase.TERMINAL_OBSERVATION,
        ManagedLifecyclePhase.TIMEOUT,
    ]
    expected_rows: list[tuple[int, ...] | None] = [None] * len(expected_phases)
    if done_rows is not None:
        expected_phases.extend(
            (
                ManagedLifecyclePhase.FINAL_OBSERVATION,
                ManagedLifecyclePhase.AUTORESET,
                ManagedLifecyclePhase.RESET_REQUEST,
                ManagedLifecyclePhase.RESET_BACKEND,
                ManagedLifecyclePhase.TASK_STATE_RESET,
                ManagedLifecyclePhase.OBSERVATION,
            )
        )
        expected_rows.extend([done_rows] * 6)
    expected_phases.append(ManagedLifecyclePhase.COMPLETE)
    expected_rows.append(None)
    actual = tuple((event.phase, event.rows) for event in runtime.last_trace)
    expected = tuple(zip(expected_phases, expected_rows, strict=True))
    assert actual == expected, (
        f"{label}: lifecycle trace mismatch actual={actual}, expected={expected}"
    )


def _force_terminal_rows(
    *,
    backend: SimBackend,
    rows: np.ndarray,
    min_base_height: float,
) -> None:
    """Use the public reset contract only to schedule a known terminal state."""

    default_qpos = backend.get_keyframe_qpos("stand")
    default_qvel = backend.get_init_qvel()
    qpos = np.broadcast_to(default_qpos, (len(rows), default_qpos.size)).copy()
    qvel = np.broadcast_to(default_qvel, (len(rows), default_qvel.size)).copy()
    qpos[:, 2] = min_base_height - 0.2
    backend.set_state(rows.astype(np.int32, copy=False), qpos, qvel)


def _action(step: int, *, num_envs: int, action_dim: int) -> np.ndarray:
    base = np.sin(np.arange(action_dim, dtype=np.float32) * 0.17 + step * 0.31) * 0.08
    offsets = np.arange(num_envs, dtype=np.float32)[:, None] * 0.002
    return np.ascontiguousarray(base[None, :] + offsets, dtype=np.float32)


@pytest.mark.parametrize(
    ("unsupported", "match"),
    (
        ("kp", "typed DR/Event"),
        ("numba", "Numba executor"),
        ("reward", "does not implement reward terms"),
    ),
)
def test_g1_managed_reference_rejects_unsupported_profile_before_backend_binding(
    unsupported: str,
    match: str,
) -> None:
    """Unsupported legacy semantics must not reach selector binding or physics."""

    cfg = _cfg(
        max_episode_seconds=None,
        observation_noise_level=0.0,
        observation_noise_seed=None,
    )
    if unsupported == "kp":
        cfg.domain_rand.randomize_kp = True
    elif unsupported == "numba":
        cfg.numba_acceleration = True
    else:
        assert isinstance(cfg.reward_config, G1WalkRewardConfig)
        cfg.reward_config.scales["penalty_feet_ori"] = 0.0

    assert cfg.scene is not None
    backend = create_backend(
        "mujoco",
        cfg.scene,
        1,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
        push_body_name=cfg.domain_rand.push_body_name,
        **env_backend_kwargs(cfg),
    )
    try:
        with patch.object(
            backend,
            "get_actuator_names",
            side_effect=AssertionError("unsupported profile reached backend binding"),
        ):
            with pytest.raises(G1ManagedReferenceError, match=match):
                create_g1_managed_reference_runtime(backend=backend, cfg=cfg)
    finally:
        backend.cleanup_scene_assets()


@pytest.mark.parametrize(
    ("seed", "num_envs", "noise_level", "noise_seed"),
    ((0, 1, 0.0, None), (1, 32, 0.0, None), (2, 128, 0.35, 41)),
)
def test_g1_managed_reference_matches_handwritten_env(
    seed: int,
    num_envs: int,
    noise_level: float,
    noise_seed: int | None,
) -> None:
    """Differential across no-done, partial reset, and all-done schedules."""

    cfg = _cfg(
        max_episode_seconds=None,
        observation_noise_level=noise_level,
        observation_noise_seed=noise_seed,
    )
    assert isinstance(cfg.reward_config, G1WalkRewardConfig)
    min_base_height = cfg.reward_config.min_base_height
    legacy, managed_backend, managed = _create_pair(cfg, num_envs=num_envs, reset_seed=seed)

    np.random.seed(seed)
    expected_initial = legacy.init_state()
    actual_initial = managed.init_state()
    # NpEnv historically exposes terminal=True immediately after its initial
    # autoreset.  The manager canonical lifecycle intentionally starts clean;
    # initial policy observations remain the shared contract under test.
    for key in expected_initial.obs:
        _assert_close(
            expected=expected_initial.obs[key],
            actual=actual_initial.obs[key],
            label=f"seed={seed}/batch={num_envs}.initial.obs[{key}]",
        )
    assert tuple((event.phase, event.rows) for event in managed.last_trace) == (
        (ManagedLifecyclePhase.INITIAL_RESET_REQUEST, None),
        (ManagedLifecyclePhase.RESET_BACKEND, None),
        (ManagedLifecyclePhase.TASK_STATE_RESET, None),
        (ManagedLifecyclePhase.OBSERVATION, None),
        (ManagedLifecyclePhase.COMPLETE, None),
    )

    action_dim = legacy.action_space.shape[0]
    for step in range(3):
        expected, _ = _step_pair(
            legacy=legacy,
            managed=managed,
            managed_backend=managed_backend,
            actions=_action(step, num_envs=num_envs, action_dim=action_dim),
            label=f"seed={seed}/batch={num_envs}.no_done.step={step}",
        )
        assert not np.any(expected.terminated | expected.truncated)
        _assert_trace(
            managed,
            done_rows=None,
            label=f"seed={seed}/batch={num_envs}.no_done.step={step}",
        )

    if num_envs > 1:
        partial_rows = np.asarray((num_envs - 1,), dtype=np.int32)
        _force_terminal_rows(
            backend=legacy._backend,
            rows=partial_rows,
            min_base_height=min_base_height,
        )
        _force_terminal_rows(
            backend=managed_backend,
            rows=partial_rows,
            min_base_height=min_base_height,
        )
        expected, _ = _step_pair(
            legacy=legacy,
            managed=managed,
            managed_backend=managed_backend,
            actions=_action(3, num_envs=num_envs, action_dim=action_dim),
            label=f"seed={seed}/batch={num_envs}.partial_reset",
        )
        assert np.array_equal(
            expected.terminated,
            np.asarray([False] * (num_envs - 1) + [True]),
        )
        assert np.array_equal(np.asarray(expected.info["_final_observation"]), expected.terminated)
        _assert_trace(
            managed,
            done_rows=(num_envs - 1,),
            label=f"seed={seed}/batch={num_envs}.partial_reset",
        )

    all_rows = np.arange(num_envs, dtype=np.int32)
    _force_terminal_rows(
        backend=legacy._backend,
        rows=all_rows,
        min_base_height=min_base_height,
    )
    _force_terminal_rows(
        backend=managed_backend,
        rows=all_rows,
        min_base_height=min_base_height,
    )
    expected, _ = _step_pair(
        legacy=legacy,
        managed=managed,
        managed_backend=managed_backend,
        actions=_action(4, num_envs=num_envs, action_dim=action_dim),
        label=f"seed={seed}/batch={num_envs}.all_done",
    )
    assert np.all(expected.terminated)
    _assert_trace(
        managed,
        done_rows=tuple(range(num_envs)),
        label=f"seed={seed}/batch={num_envs}.all_done",
    )


@pytest.mark.parametrize(("seed", "num_envs"), ((3, 1), (4, 4)))
def test_g1_managed_reference_matches_handwritten_timeout_autoreset(
    seed: int,
    num_envs: int,
) -> None:
    """A one-control-step episode exercises timeout/final-observation/reset ordering."""

    cfg = _cfg(
        max_episode_seconds=0.02,
        observation_noise_level=0.2,
        observation_noise_seed=73,
    )
    assert cfg.max_episode_steps == 1
    legacy, managed_backend, managed = _create_pair(cfg, num_envs=num_envs, reset_seed=seed)

    np.random.seed(seed)
    expected_initial = legacy.init_state()
    actual_initial = managed.init_state()
    for key in expected_initial.obs:
        _assert_close(
            expected=expected_initial.obs[key],
            actual=actual_initial.obs[key],
            label=f"timeout.seed={seed}/batch={num_envs}.initial.obs[{key}]",
        )

    expected, actual = _step_pair(
        legacy=legacy,
        managed=managed,
        managed_backend=managed_backend,
        actions=_action(0, num_envs=num_envs, action_dim=legacy.action_space.shape[0]),
        label=f"timeout.seed={seed}/batch={num_envs}.step",
    )
    assert not np.any(expected.terminated)
    assert np.all(expected.truncated)
    assert np.all(np.asarray(expected.info["_final_observation"]))
    assert actual.final_observation is not None
    _assert_trace(
        managed,
        done_rows=tuple(range(num_envs)),
        label=f"timeout.seed={seed}/batch={num_envs}.step",
    )
