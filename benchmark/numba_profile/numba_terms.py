"""Single-source **numba** device functions — one per reward term.

Each ``@njit(inline="always")`` function computes *one* term for *one*
environment row ``i`` and mirrors the matching function in
``numpy_reference.py`` line-for-line (scalar instead of vectorised).  Three
payoffs from one definition (support pillar 1 of the plan):

  * the fused kernel ``inline``s them -> machine-code fusion, no intermediate
    ``(N, ...)`` arrays (the 2.3-3.8x single-core win in issue #665);
  * ``fn.py_func`` is the plain-Python version -> set a breakpoint, unit-test it;
  * scalar form reads closer to the math than the ``out=`` numpy in tracking.py.

Term identity/ordering is owned by ``spec.TERM_ORDER``; this module only
supplies the arithmetic.
"""

from __future__ import annotations

import math

from numba import njit


def _dev(fn):
    """Device-function decorator: inlined into the kernel, GIL-free, cached.

    ``inline="always"`` is what turns these separate readable functions into a
    single fused machine-code loop when called from the prange kernel.
    """
    return njit(inline="always", fastmath=True, cache=True, nogil=True)(fn)


@_dev
def _exp_reward(error, std):
    return math.exp(error / -(std * std))


# ── anchor (root) terms ─────────────────────────────────────────────────────
@_dev
def motion_global_root_pos_i(motion_pos, robot_pos, a, std, i):
    dx = motion_pos[i, a, 0] - robot_pos[i, a, 0]
    dy = motion_pos[i, a, 1] - robot_pos[i, a, 1]
    dz = motion_pos[i, a, 2] - robot_pos[i, a, 2]
    return _exp_reward(dx * dx + dy * dy + dz * dz, std)


@_dev
def _quat_angle_sq(qa, qb, i, a):
    dot = (
        qa[i, a, 0] * qb[i, a, 0]
        + qa[i, a, 1] * qb[i, a, 1]
        + qa[i, a, 2] * qb[i, a, 2]
        + qa[i, a, 3] * qb[i, a, 3]
    )
    dot = abs(dot)
    if dot > 1.0:
        dot = 1.0
    ang = 2.0 * math.acos(dot)
    return ang * ang


@_dev
def motion_global_root_ori_i(motion_quat, robot_quat, a, std, i):
    return _exp_reward(_quat_angle_sq(motion_quat, robot_quat, i, a), std)


# ── per-body mean terms ─────────────────────────────────────────────────────
@_dev
def _mean_body_xyz_sq_err_i(ref, act, n_body, i):
    acc = 0.0
    for bdy in range(n_body):
        dx = ref[i, bdy, 0] - act[i, bdy, 0]
        dy = ref[i, bdy, 1] - act[i, bdy, 1]
        dz = ref[i, bdy, 2] - act[i, bdy, 2]
        acc += dx * dx + dy * dy + dz * dz
    return acc / n_body


@_dev
def motion_body_pos_i(ref_pos, robot_pos, n_body, std, i):
    return _exp_reward(_mean_body_xyz_sq_err_i(ref_pos, robot_pos, n_body, i), std)


@_dev
def motion_body_ori_i(ref_quat, robot_quat, n_body, std, i):
    acc = 0.0
    for bdy in range(n_body):
        acc += _quat_angle_sq(ref_quat, robot_quat, i, bdy)
    return _exp_reward(acc / n_body, std)


@_dev
def motion_body_lin_vel_i(motion_v, robot_v, n_body, std, i):
    return _exp_reward(_mean_body_xyz_sq_err_i(motion_v, robot_v, n_body, i), std)


@_dev
def motion_body_ang_vel_i(motion_w, robot_w, n_body, std, i):
    return _exp_reward(_mean_body_xyz_sq_err_i(motion_w, robot_w, n_body, i), std)


# ── end-effector / joint terms ──────────────────────────────────────────────
@_dev
def motion_ee_body_pos_z_i(ref_pos, robot_pos, ee_idx, std, i):
    acc = 0.0
    for k in range(ee_idx.shape[0]):
        e = ref_pos[i, ee_idx[k], 2] - robot_pos[i, ee_idx[k], 2]
        acc += e * e
    return _exp_reward(acc / ee_idx.shape[0], std)


@_dev
def _mean_joint_sq_err_i(ref, act, n_action, i):
    acc = 0.0
    for j in range(n_action):
        d = ref[i, j] - act[i, j]
        acc += d * d
    return acc / n_action


@_dev
def motion_joint_pos_i(motion_jp, dof_pos, n_action, std, i):
    return _exp_reward(_mean_joint_sq_err_i(motion_jp, dof_pos, n_action, i), std)


@_dev
def motion_joint_vel_i(motion_jv, dof_vel, n_action, std, i):
    return _exp_reward(_mean_joint_sq_err_i(motion_jv, dof_vel, n_action, i), std)


@_dev
def action_rate_l2_i(cur, last, n_action, i):
    acc = 0.0
    for j in range(n_action):
        d = cur[i, j] - last[i, j]
        acc += d * d
    return acc


@_dev
def joint_limit_i(dof_pos, lower, upper, n_action, i):
    acc = 0.0
    for j in range(n_action):
        lo = lower[j] - dof_pos[i, j]
        if lo < 0.0:
            lo = 0.0
        hi = dof_pos[i, j] - upper[j]
        if hi < 0.0:
            hi = 0.0
        v = lo + hi
        acc += v * v
    return acc
