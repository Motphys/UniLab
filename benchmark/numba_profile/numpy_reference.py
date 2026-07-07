"""Faithful **numpy** reference for ``g1_motion_tracking`` reward + termination.

This is the golden oracle: each function mirrors the corresponding
``_reward_*`` / ``_compute_terminations`` in
``src/unilab/envs/motion_tracking/g1/tracking.py`` (line refs inline).  The
numba kernel is validated bit-for-tolerance against this file, and it doubles
as the human-readable spec of what the kernel computes.

Style note: the real task hand-rolls ``out=`` numpy calls to reuse scratch
buffers; here we favour plain vectorised expressions for readability — the
arithmetic is identical, only the memory management differs.
"""

from __future__ import annotations

import numpy as np

from . import spec
from .state import Batch

_DT = np.float32


# ── error helpers (tracking.py:1262-1303) ───────────────────────────────────
def _mean_body_xyz_squared_error(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Mean over bodies of the per-body squared xyz error (tracking.py:1262)."""
    diff = reference - actual  # (N, n_body, 3)
    per_body = np.sum(diff * diff, axis=2)  # (N, n_body)
    return np.sum(per_body, axis=1) / reference.shape[1]  # (N,)


def _quat_error_sq_body(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """(2*acos(|dot|))^2 per body (tracking.py:1280)."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))  # (N, n_body)
    dot = np.clip(dot, 0.0, 1.0)
    ang = 2.0 * np.arccos(dot)
    return ang * ang


def _exp_reward(error: np.ndarray, std: float) -> np.ndarray:
    """exp(-error / std^2)  (tracking.py:1299)."""
    return np.exp(error / -(std * std))


# ── reward terms (one function per tracking.py:_reward_*) ────────────────────
def motion_global_root_pos(b: Batch) -> np.ndarray:  # tracking.py:1306
    a = spec.ANCHOR_BODY_IDX
    diff = b.motion_body_pos_w[:, a] - b.robot_body_pos_w[:, a]
    err = np.sum(diff * diff, axis=1)
    return _exp_reward(err, spec.REWARD_SPEC.std_root_pos)


def motion_global_root_ori(b: Batch) -> np.ndarray:  # tracking.py:1322
    a = spec.ANCHOR_BODY_IDX
    dot = np.abs(np.sum(b.motion_body_quat_w[:, a] * b.robot_body_quat_w[:, a], axis=1))
    dot = np.clip(dot, 0.0, 1.0)
    ang = 2.0 * np.arccos(dot)
    return _exp_reward(ang * ang, spec.REWARD_SPEC.std_root_ori)


def motion_body_pos(b: Batch) -> np.ndarray:  # tracking.py:1341
    err = _mean_body_xyz_squared_error(b.ref_body_pos_relative_w, b.robot_body_pos_w)
    return _exp_reward(err, spec.REWARD_SPEC.std_body_pos)


def motion_body_ori(b: Batch) -> np.ndarray:  # tracking.py:1346
    per_body = _quat_error_sq_body(b.ref_body_quat_relative_w, b.robot_body_quat_w)
    err = np.sum(per_body, axis=1) / per_body.shape[1]
    return _exp_reward(err, spec.REWARD_SPEC.std_body_ori)


def motion_body_lin_vel(b: Batch) -> np.ndarray:  # tracking.py:1355
    err = _mean_body_xyz_squared_error(b.motion_body_lin_vel_w, b.robot_body_lin_vel_w)
    return _exp_reward(err, spec.REWARD_SPEC.std_body_lin_vel)


def motion_body_ang_vel(b: Batch) -> np.ndarray:  # tracking.py:1361
    err = _mean_body_xyz_squared_error(b.motion_body_ang_vel_w, b.robot_body_ang_vel_w)
    return _exp_reward(err, spec.REWARD_SPEC.std_body_ang_vel)


def motion_ee_body_pos_z(b: Batch) -> np.ndarray:  # tracking.py:1367
    ee = spec.EE_BODY_INDICES
    diff = b.ref_body_pos_relative_w[:, ee, 2] - b.robot_body_pos_w[:, ee, 2]
    err = np.sum(diff * diff, axis=1) / ee.size
    return _exp_reward(err, spec.REWARD_SPEC.std_body_pos)


def motion_joint_pos(b: Batch) -> np.ndarray:  # tracking.py:1379
    diff = b.motion_joint_pos - b.dof_pos
    err = np.sum(diff * diff, axis=1) / b.dof_pos.shape[1]
    return _exp_reward(err, spec.REWARD_SPEC.std_joint_pos)


def motion_joint_vel(b: Batch) -> np.ndarray:  # tracking.py:1388
    diff = b.motion_joint_vel - b.dof_vel
    err = np.sum(diff * diff, axis=1) / b.dof_vel.shape[1]
    return _exp_reward(err, spec.REWARD_SPEC.std_joint_vel)


def action_rate_l2(b: Batch) -> np.ndarray:  # tracking.py:1408
    diff = b.current_actions - b.last_actions
    return np.sum(diff * diff, axis=1)


def joint_limit(b: Batch) -> np.ndarray:  # tracking.py:1414
    low = np.maximum(b.joint_lower - b.dof_pos, 0.0)
    high = np.maximum(b.dof_pos - b.joint_upper, 0.0)
    viol = low + high
    return np.sum(viol * viol, axis=1)


# Registry pairing name -> numpy fn (index-aligned with spec.TERM_ORDER).
NUMPY_TERMS = {
    "motion_global_root_pos": motion_global_root_pos,
    "motion_global_root_ori": motion_global_root_ori,
    "motion_body_pos": motion_body_pos,
    "motion_body_ori": motion_body_ori,
    "motion_body_lin_vel": motion_body_lin_vel,
    "motion_body_ang_vel": motion_body_ang_vel,
    "motion_ee_body_pos_z": motion_ee_body_pos_z,
    "motion_joint_pos": motion_joint_pos,
    "motion_joint_vel": motion_joint_vel,
    "action_rate_l2": action_rate_l2,
    "joint_limit": joint_limit,
}


def compute_reward(b: Batch, scales: dict[str, float] | None = None) -> tuple[np.ndarray, dict]:
    """Faithful port of ``_compute_reward`` (tracking.py:1209).

    Returns ``(reward, per_term_weighted_mean)``.  Skips ``scale == 0`` terms
    exactly like the real loop, then multiplies the sum by ``ctrl_dt``.
    """
    s = spec.REWARD_SPEC.scales if scales is None else scales
    n = b.num_envs
    reward = np.zeros(n, dtype=_DT)
    log: dict[str, float] = {}
    for name in spec.TERM_ORDER:
        scale = s.get(name, 0.0)
        if scale == 0:
            continue
        weighted = NUMPY_TERMS[name](b).astype(_DT) * _DT(scale)
        reward += weighted
        log[f"reward/{name}"] = float(np.sum(weighted) / weighted.size)
    reward *= _DT(spec.CTRL_DT)
    return reward, log


def compute_terminations(b: Batch) -> np.ndarray:
    """Faithful port of ``_compute_terminations`` (tracking.py:978)."""
    a = spec.ANCHOR_BODY_IDX
    terminated = np.zeros(b.num_envs, dtype=bool)

    # anchor position error (z only)
    dz = np.abs(b.motion_body_pos_w[:, a, 2] - b.robot_body_pos_w[:, a, 2])
    terminated |= dz > spec.ANCHOR_POS_Z_THRESHOLD

    # anchor orientation via body-frame gravity-z (tracking.py:276)
    if spec.ANCHOR_ORI_THRESHOLD < 2.0:
        mq = b.motion_body_quat_w[:, a]
        rq = b.robot_body_quat_w[:, a]
        m_gz = 2.0 * (mq[:, 1] * mq[:, 1] + mq[:, 2] * mq[:, 2]) - 1.0
        r_gz = 2.0 * (rq[:, 1] * rq[:, 1] + rq[:, 2] * rq[:, 2]) - 1.0
        terminated |= np.abs(m_gz - r_gz) > spec.ANCHOR_ORI_THRESHOLD

    # end-effector z error (any)
    ee = spec.EE_BODY_INDICES
    ee_dz = np.abs(b.ref_body_pos_relative_w[:, ee, 2] - b.robot_body_pos_w[:, ee, 2])
    terminated |= np.any(ee_dz > spec.EE_BODY_POS_Z_THRESHOLD, axis=1)

    if spec.TERMINATE_ON_UNDESIRED_CONTACTS:
        uc = spec.UNDESIRED_CONTACT_INDICES
        below = b.robot_body_pos_w[:, uc, 2] < spec.UNDESIRED_CONTACT_Z_THRESHOLD
        terminated |= np.any(below, axis=1)

    return terminated


def update_state(b: Batch, scales: dict[str, float] | None = None):
    """Reward + termination — the numpy half of ``update_state`` overhead."""
    reward, log = compute_reward(b, scales)
    terminated = compute_terminations(b)
    return reward, terminated, log
