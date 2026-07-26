"""Guard tests for the extracted motion-tracking reward module.

These lock the ``common.rewards`` term functions and dispatch against
hand-computed values on a small synthetic ``RewardContext`` and guard against
numpy/numba reward-term drift.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.motion_tracking.common import numba as numba_mod
from unilab.envs.motion_tracking.common import rewards
from unilab.envs.motion_tracking.common.rewards import RewardConfig, RewardContext


def _make_ctx(*, scales: dict[str, float] | None = None) -> RewardContext:
    """Build a small deterministic RewardContext (2 envs, 3 bodies, 2 joints)."""
    rng = np.random.default_rng(7)
    num_envs, n_body, n_action = 2, 3, 2
    dtype = np.float64

    def rq(shape: tuple[int, ...]) -> np.ndarray:
        q = rng.normal(size=(*shape, 4)).astype(dtype)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return q

    reward_config = RewardConfig()
    if scales is not None:
        reward_config.scales = scales

    motion_data = SimpleNamespace(
        joint_pos=rng.normal(size=(num_envs, n_action)).astype(dtype),
        joint_vel=rng.normal(size=(num_envs, n_action)).astype(dtype),
        body_pos_w=rng.normal(size=(num_envs, n_body, 3)).astype(dtype),
        body_quat_w=rq((num_envs, n_body)),
        body_lin_vel_w=rng.normal(size=(num_envs, n_body, 3)).astype(dtype),
        body_ang_vel_w=rng.normal(size=(num_envs, n_body, 3)).astype(dtype),
    )

    robot_body_pos_w = rng.normal(size=(num_envs, n_body, 3)).astype(dtype)
    # Force a deterministic contact pattern for the undesired_contacts term.
    robot_body_pos_w[:, :, 2] = np.array([[0.01, 0.20, 0.03], [0.10, 0.02, 0.30]], dtype=dtype)
    undesired_idx = np.array([0, 1], dtype=np.int32)

    current_actions = rng.normal(size=(num_envs, n_action)).astype(dtype)
    last_actions = rng.normal(size=(num_envs, n_action)).astype(dtype)

    return RewardContext(
        info={
            "current_actions": current_actions,
            "last_actions": last_actions,
            "steps": np.zeros((num_envs,), dtype=np.uint32),
        },
        motion_data=motion_data,
        robot_body_pos_w=robot_body_pos_w,
        robot_body_quat_w=rq((num_envs, n_body)),
        robot_body_lin_vel_w=rng.normal(size=(num_envs, n_body, 3)).astype(dtype),
        robot_body_ang_vel_w=rng.normal(size=(num_envs, n_body, 3)).astype(dtype),
        ref_body_pos_w=rng.normal(size=(num_envs, n_body, 3)).astype(dtype),
        ref_body_quat_w=rq((num_envs, n_body)),
        dof_pos=rng.normal(size=(num_envs, n_action)).astype(dtype),
        dof_vel=rng.normal(size=(num_envs, n_action)).astype(dtype),
        reward_config=reward_config,
        anchor_body_idx=0,
        ee_body_indices=np.array([2], dtype=np.int32),
        undesired_contact_body_indices=undesired_idx,
        joint_lower=np.array([-0.5, -0.5], dtype=dtype),
        joint_upper=np.array([0.5, 0.5], dtype=dtype),
        undesired_contact_z_threshold=0.05,
        num_envs=num_envs,
        body_vec_error=np.empty((num_envs, n_body, 3), dtype=dtype),
        joint_error=np.empty((num_envs, n_action), dtype=dtype),
        joint_error_upper=np.empty((num_envs, n_action), dtype=dtype),
        env_error=np.empty((num_envs,), dtype=dtype),
        env_error2=np.empty((num_envs,), dtype=dtype),
        reward_term=np.empty((num_envs,), dtype=dtype),
        weighted_reward=np.empty((num_envs,), dtype=dtype),
        quat_error_w=np.empty((num_envs, n_body), dtype=dtype),
        quat_error_x=np.empty((num_envs, n_body), dtype=dtype),
        ee_pos_error_z=np.empty((num_envs, 1), dtype=dtype),
        undesired_contact_mask=np.empty((num_envs, undesired_idx.size), dtype=bool),
    )


def test_build_reward_functions_matches_numba_term_order():
    assert set(rewards.build_reward_functions()) == set(numba_mod.TERM_ORDER)


def test_motion_joint_pos_term_matches_hand_computed():
    ctx = _make_ctx()
    out = rewards.motion_joint_pos(ctx).copy()
    error = np.mean(np.square(ctx.motion_data.joint_pos - ctx.dof_pos), axis=1)
    expected = np.exp(-error / ctx.reward_config.std_joint_pos**2)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_action_rate_l2_term_matches_hand_computed():
    ctx = _make_ctx()
    out = rewards.action_rate_l2(ctx).copy()
    expected = np.sum(np.square(ctx.info["current_actions"] - ctx.info["last_actions"]), axis=1)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_undesired_contacts_term_matches_hand_computed():
    ctx = _make_ctx()
    out = rewards.undesired_contacts(ctx).copy()
    idx = ctx.undesired_contact_body_indices
    expected = np.sum(
        ctx.robot_body_pos_w[:, idx, 2] < ctx.undesired_contact_z_threshold, axis=1
    ).astype(np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_compute_reward_matches_hand_computed_weighted_sum():
    scales = {
        "motion_joint_pos": 1.0,
        "action_rate_l2": -0.1,
        "undesired_contacts": -0.5,
        # zero-weighted terms are skipped by compute_reward
        "motion_body_pos": 0.0,
    }
    ctrl_dt = 0.02
    fns = rewards.build_reward_functions()

    # Reference terms computed on an untouched ctx (fresh buffers per call).
    joint_pos_ref = rewards.motion_joint_pos(_make_ctx()).copy()
    action_rate_ref = rewards.action_rate_l2(_make_ctx()).copy()
    undesired_ref = rewards.undesired_contacts(_make_ctx()).copy()

    ctx = _make_ctx(scales=scales)
    reward = rewards.compute_reward(
        ctx,
        active_reward_fns=fns,
        all_reward_fns=fns,
        scales=scales,
        ctrl_dt=ctrl_dt,
        enable_log=False,
    ).copy()

    expected = (1.0 * joint_pos_ref - 0.1 * action_rate_ref - 0.5 * undesired_ref) * ctrl_dt
    np.testing.assert_allclose(reward, expected, rtol=1e-12, atol=1e-12)
