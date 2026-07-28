"""Real-CUDA acceptance for one G1 manager plan on independent backends."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from unilab.base.backend import (
    RowSelection,
    SimBackend,
    create_backend,
    env_backend_kwargs,
)
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.envs.locomotion.g1.joystick import G1WalkFlatCfg, G1WalkRewardConfig
from unilab.envs.locomotion.g1.managed_reference import create_g1_managed_reference_runtime
from unilab.manager import ManagedReferenceRuntime, managed_policy_abi_snapshot

pytestmark = pytest.mark.slow

_ATOL = 1.0e-4
_RTOL = 1.0e-3
_NUM_ENVS = 32


def _reward_config() -> G1WalkRewardConfig:
    """The strict host-reference slice shared by both G1 owner profiles."""

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


def _cfg() -> G1WalkFlatCfg:
    cfg = G1WalkFlatCfg(reward_config=_reward_config())
    cfg.max_episode_seconds = None
    cfg.control_config.action_scale = 0.25
    cfg.curriculum.enabled = False
    cfg.domain_rand.randomize_kp = False
    cfg.domain_rand.randomize_kd = False
    cfg.commands.resampling_time = 0.0
    cfg.commands.heading_command = False
    return cfg


def _backend(backend_type: str, cfg: G1WalkFlatCfg) -> SimBackend:
    assert cfg.scene is not None
    return create_backend(
        backend_type,
        cfg.scene,
        _NUM_ENVS,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
        push_body_name=cfg.domain_rand.push_body_name,
        **env_backend_kwargs(cfg),
    )


def _actions(step: int) -> np.ndarray:
    action_dim = 29
    # Exercise a nonzero control schedule while keeping the three-step CUDA
    # differential within the Phase 3 pre-registered trajectory tolerance.
    base = np.sin(np.arange(action_dim, dtype=np.float32) * 0.17 + step * 0.31) * 0.04
    offsets = np.arange(_NUM_ENVS, dtype=np.float32)[:, None] * 0.002
    return np.ascontiguousarray(base[None, :] + offsets, dtype=np.float32)


def _copy_public_state(
    backend: SimBackend, runtime: ManagedReferenceRuntime
) -> dict[str, np.ndarray]:
    """Read only through the public bound state view and copy before next mutation."""

    result = backend.read_state_batch(
        runtime.bound_plan,
        RowSelection.all(_NUM_ENVS),
    )
    state = result.state
    return {
        field.key: np.array(state.buffer_at(index).handle, copy=True)
        for index, field in enumerate(state.plan.state.fields)
    }


def _align_quaternion_sign(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Quaternion signs encode the same orientation; compare the closest form."""

    dot = np.sum(expected * actual, axis=-1, keepdims=True)
    return actual * np.where(dot < 0.0, -1.0, 1.0)


def _assert_arrays_close(*, expected: np.ndarray, actual: np.ndarray, label: str) -> None:
    if expected.shape != actual.shape:
        raise AssertionError(f"{label}: shape mismatch {expected.shape} != {actual.shape}")
    candidate = actual
    if expected.shape[-1:] == (4,) and "orientation" in label:
        candidate = _align_quaternion_sign(expected, actual)
    assert_allclose(expected, candidate, atol=_ATOL, rtol=_RTOL, err_msg=label)


def _assert_public_states_match(
    expected: dict[str, np.ndarray], actual: dict[str, np.ndarray], *, label: str
) -> None:
    assert tuple(expected) == tuple(actual), f"{label}: state field keys differ"
    for key in expected:
        _assert_arrays_close(
            expected=expected[key],
            actual=actual[key],
            label=f"{label}.state[{key}]",
        )


def _assert_transition_match(
    expected: Any,
    actual: Any,
    *,
    label: str,
) -> None:
    assert tuple(expected.obs) == tuple(actual.obs), f"{label}: observation groups differ"
    for key in expected.obs:
        _assert_arrays_close(
            expected=np.asarray(expected.obs[key]),
            actual=np.asarray(actual.obs[key]),
            label=f"{label}.obs[{key}]",
        )
    _assert_arrays_close(
        expected=np.asarray(expected.reward),
        actual=np.asarray(actual.reward),
        label=f"{label}.reward",
    )
    assert np.array_equal(expected.terminated, actual.terminated), f"{label}: terminated differs"
    assert np.array_equal(expected.truncated, actual.truncated), f"{label}: truncated differs"
    assert np.array_equal(
        np.asarray(expected.info["_final_observation"]),
        np.asarray(actual.info["_final_observation"]),
    ), f"{label}: final-observation mask differs"
    assert expected.final_observation is None
    assert actual.final_observation is None
    assert tuple(expected.info["log"]) == tuple(actual.info["log"]), f"{label}: log keys differ"
    for key in expected.info["log"]:
        _assert_arrays_close(
            expected=np.asarray(expected.info["log"][key]),
            actual=np.asarray(actual.info["log"][key]),
            label=f"{label}.log[{key}]",
        )


@contextmanager
def _forbid_managed_hot_path_fallbacks(backends: tuple[SimBackend, ...]) -> Iterator[None]:
    """Turn any getter/selector/reset/asset fallback into a deterministic failure."""

    methods = (
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
    )
    with ExitStack() as stack:
        for backend in backends:
            for method in methods:
                stack.enter_context(
                    patch.object(
                        backend,
                        method,
                        side_effect=AssertionError(f"managed hot-path fallback: {method}"),
                    )
                )
        stack.enter_context(
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset read fallback"))
        )
        stack.enter_context(
            patch.object(Path, "read_text", side_effect=AssertionError("asset read fallback"))
        )
        yield


@pytest.mark.parametrize("seed", (0, 1))
def test_g1_plan_is_shared_by_mujoco_and_mjwarp(seed: int) -> None:
    """One manager plan must execute through both independent production backends."""

    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("cross-backend manager acceptance requires an active CUDA Warp device")

    mujoco_cfg = _cfg()
    mjwarp_cfg = deepcopy(mujoco_cfg)
    mujoco_backend = _backend("mujoco", mujoco_cfg)
    mjwarp_backend = _backend("mjwarp", mjwarp_cfg)
    try:
        mujoco_runtime = create_g1_managed_reference_runtime(
            backend=mujoco_backend,
            cfg=mujoco_cfg,
            reset_seed=seed,
            record_lifecycle=True,
        )
        mjwarp_runtime = create_g1_managed_reference_runtime(
            backend=mjwarp_backend,
            cfg=mjwarp_cfg,
            reset_seed=seed,
            record_lifecycle=True,
        )

        assert mujoco_runtime.plan.fingerprint == mjwarp_runtime.plan.fingerprint
        assert mujoco_runtime.plan.policy_abi == mjwarp_runtime.plan.policy_abi
        assert managed_policy_abi_snapshot(mujoco_runtime.plan) == managed_policy_abi_snapshot(
            mjwarp_runtime.plan
        )
        assert mujoco_runtime.bound_plan.backend_type == "mujoco"
        assert mjwarp_runtime.bound_plan.backend_type == "mjwarp"
        assert (
            mujoco_runtime.bound_plan.backend_instance_id
            != mjwarp_runtime.bound_plan.backend_instance_id
        )
        assert mujoco_runtime.bound_plan.fingerprint != mjwarp_runtime.bound_plan.fingerprint

        initial_mujoco = mujoco_runtime.init_state()
        initial_mjwarp = mjwarp_runtime.init_state()
        _assert_transition_match(initial_mujoco, initial_mjwarp, label=f"seed={seed}.initial")
        _assert_public_states_match(
            _copy_public_state(mujoco_backend, mujoco_runtime),
            _copy_public_state(mjwarp_backend, mjwarp_runtime),
            label=f"seed={seed}.initial",
        )

        for step in range(3):
            actions = _actions(step)
            with _forbid_managed_hot_path_fallbacks((mujoco_backend, mjwarp_backend)):
                transition_mujoco = mujoco_runtime.step(actions.copy())
                transition_mjwarp = mjwarp_runtime.step(actions.copy())
            _assert_transition_match(
                transition_mujoco,
                transition_mjwarp,
                label=f"seed={seed}.step={step}",
            )
            _assert_public_states_match(
                _copy_public_state(mujoco_backend, mujoco_runtime),
                _copy_public_state(mjwarp_backend, mjwarp_runtime),
                label=f"seed={seed}.step={step}",
            )
            assert mujoco_runtime.last_trace == mjwarp_runtime.last_trace
    finally:
        mujoco_backend.cleanup_scene_assets()
        mjwarp_backend.cleanup_scene_assets()
