"""Fused numba ``update_state`` kernel for ``g1_motion_tracking``.

Realises the structured scheme:

  * **Pillar 2 (config-first, no codegen):** the kernel is a *static superset*
    of every term in ``spec.TERM_ORDER``.  A dense ``scale`` vector (built on the
    cold path from the config dict) gates each term — ``scale==0`` contributes
    exactly 0, matching the numpy loop's ``continue``.  Weights stay in config,
    never baked into compiled code.
  * **Pillar 1 (single source):** each term is the inlined device function from
    ``numba_terms.py`` — same code the ``.py_func`` debug path uses.
  * **Pillar 4 (per-thread log scratch):** per-term reward means accumulate into
    ``(nthreads, N_TERMS)`` scratch indexed by thread id, summed on the main
    thread afterwards.  Writing a shared ``log[k] += ...`` would false-share one
    cache line across all threads and cap scaling at ~5x (issue #665 §2).
  * **Pillar 6 (fallback):** ``update_state`` transparently drops to the numpy
    oracle for small batches or when explicitly forced.

Termination is fused into the same prange loop (issue #665: reset/termination is
the co-bottleneck; folding it in costs nothing once we already stream each row).
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads

from . import numba_terms as T
from . import spec
from .numpy_reference import compute_terminations
from .numpy_reference import update_state as _numpy_update_state
from .state import Batch


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _fused_update_kernel(
    motion_pos,
    motion_quat,
    motion_lin_vel,
    motion_ang_vel,
    motion_jp,
    motion_jv,
    ref_pos,
    ref_quat,
    robot_pos,
    robot_quat,
    robot_lin_vel,
    robot_ang_vel,
    dof_pos,
    dof_vel,
    cur_act,
    last_act,
    joint_lower,
    joint_upper,
    scale,  # (N_TERMS,) float64 — term gates/weights, TERM_ORDER
    std,  # (N_TERMS,) float64 — per-term exp std
    anchor,  # int — anchor body index
    ee_idx,  # (n_ee,) int32
    n_body,
    n_action,
    ctrl_dt,
    anchor_pos_z_thr,
    anchor_ori_thr,
    ee_pos_z_thr,
    reward,  # (N,) out
    terminated,  # (N,) bool out
    log_scratch,  # (nthreads, N_TERMS) out — indexed by get_thread_id()
):
    n = reward.shape[0]
    for i in prange(n):
        tid = get_thread_id()  # actual executing thread -> no cross-thread scratch collisions

        # ── reward: static superset, each term gated by its scale ───────────
        r = 0.0
        # index k must match spec.TERM_ORDER
        w = T.motion_global_root_pos_i(motion_pos, robot_pos, anchor, std[0], i) * scale[0]
        r += w
        log_scratch[tid, 0] += w
        w = T.motion_global_root_ori_i(motion_quat, robot_quat, anchor, std[1], i) * scale[1]
        r += w
        log_scratch[tid, 1] += w
        w = T.motion_body_pos_i(ref_pos, robot_pos, n_body, std[2], i) * scale[2]
        r += w
        log_scratch[tid, 2] += w
        w = T.motion_body_ori_i(ref_quat, robot_quat, n_body, std[3], i) * scale[3]
        r += w
        log_scratch[tid, 3] += w
        w = T.motion_body_lin_vel_i(motion_lin_vel, robot_lin_vel, n_body, std[4], i) * scale[4]
        r += w
        log_scratch[tid, 4] += w
        w = T.motion_body_ang_vel_i(motion_ang_vel, robot_ang_vel, n_body, std[5], i) * scale[5]
        r += w
        log_scratch[tid, 5] += w
        w = T.motion_ee_body_pos_z_i(ref_pos, robot_pos, ee_idx, std[6], i) * scale[6]
        r += w
        log_scratch[tid, 6] += w
        w = T.motion_joint_pos_i(motion_jp, dof_pos, n_action, std[7], i) * scale[7]
        r += w
        log_scratch[tid, 7] += w
        w = T.motion_joint_vel_i(motion_jv, dof_vel, n_action, std[8], i) * scale[8]
        r += w
        log_scratch[tid, 8] += w
        w = T.action_rate_l2_i(cur_act, last_act, n_action, i) * scale[9]
        r += w
        log_scratch[tid, 9] += w
        w = T.joint_limit_i(dof_pos, joint_lower, joint_upper, n_action, i) * scale[10]
        r += w
        log_scratch[tid, 10] += w
        reward[i] = r * ctrl_dt

        # ── termination (fused) ─────────────────────────────────────────────
        term = False
        if abs(motion_pos[i, anchor, 2] - robot_pos[i, anchor, 2]) > anchor_pos_z_thr:
            term = True
        mgz = 2.0 * (motion_quat[i, anchor, 1] ** 2 + motion_quat[i, anchor, 2] ** 2) - 1.0
        rgz = 2.0 * (robot_quat[i, anchor, 1] ** 2 + robot_quat[i, anchor, 2] ** 2) - 1.0
        if abs(mgz - rgz) > anchor_ori_thr:
            term = True
        for k in range(ee_idx.shape[0]):
            if abs(ref_pos[i, ee_idx[k], 2] - robot_pos[i, ee_idx[k], 2]) > ee_pos_z_thr:
                term = True
        terminated[i] = term


class FusedUpdateState:
    """Cold-path-initialised driver holding the config-derived vectors.

    Mirrors how the real env would build ``scale``/``std`` once at reset and call
    a hot kernel each step — no per-step config parsing.
    """

    def __init__(self, scales: dict[str, float] | None = None, num_threads: int | None = None):
        self.scale = spec.scale_vector(scales)
        self.std = spec.std_vector()
        self.ee_idx = spec.EE_BODY_INDICES
        self.num_threads = num_threads

    def __call__(self, b: Batch) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.num_threads is not None:
            set_num_threads(self.num_threads)
        nthreads = get_num_threads()
        n = b.num_envs

        reward = np.empty(n, dtype=np.float32)
        terminated = np.empty(n, dtype=np.bool_)
        log_scratch = np.zeros((nthreads, spec.N_TERMS), dtype=np.float64)

        _fused_update_kernel(
            b.motion_body_pos_w,
            b.motion_body_quat_w,
            b.motion_body_lin_vel_w,
            b.motion_body_ang_vel_w,
            b.motion_joint_pos,
            b.motion_joint_vel,
            b.ref_body_pos_relative_w,
            b.ref_body_quat_relative_w,
            b.robot_body_pos_w,
            b.robot_body_quat_w,
            b.robot_body_lin_vel_w,
            b.robot_body_ang_vel_w,
            b.dof_pos,
            b.dof_vel,
            b.current_actions,
            b.last_actions,
            b.joint_lower,
            b.joint_upper,
            self.scale,
            self.std,
            spec.ANCHOR_BODY_IDX,
            self.ee_idx,
            spec.N_BODY,
            spec.NUM_ACTION,
            spec.CTRL_DT,
            spec.ANCHOR_POS_Z_THRESHOLD,
            spec.ANCHOR_ORI_THRESHOLD,
            spec.EE_BODY_POS_Z_THRESHOLD,
            reward,
            terminated,
            log_scratch,
        )

        # aggregate per-thread scratch -> per-term mean, for active terms only
        term_sums = log_scratch.sum(axis=0)  # (N_TERMS,)
        log = {
            f"reward/{name}": float(term_sums[k] / n)
            for k, name in enumerate(spec.TERM_ORDER)
            if self.scale[k] != 0.0
        }
        return reward, terminated, log


# Min batch below which the parallel kernel's launch/barrier cost is not worth
# it and we fall back to numpy (pillar 6). Tuned conservatively; see bench.py.
_FALLBACK_MIN_ENVS = 256


def update_state(
    b: Batch,
    scales: dict[str, float] | None = None,
    num_threads: int | None = None,
    force: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reward + termination via numba, with numpy fallback for small batches.

    ``force`` in {"auto", "numba", "numpy"} pins the path (bench/test use it).
    """
    if force == "numpy" or (force == "auto" and b.num_envs < _FALLBACK_MIN_ENVS):
        return _numpy_update_state(b, scales)
    driver = FusedUpdateState(scales=scales, num_threads=num_threads)
    return driver(b)


__all__ = ["FusedUpdateState", "update_state", "compute_terminations"]
