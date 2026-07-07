"""Deterministic synthetic batch reproducing the arrays that
``g1_motion_tracking``'s reward + termination read every step.

We do **not** run MuJoCo/Motrix — the numba target is the backend-agnostic
``update_state`` overhead (obs/reward/termination), which is pure array math on
robot state vs. a motion reference (issue #663/#665).  So we synthesise those
arrays at the exact shapes/dtypes the real task uses, with magnitudes chosen so
rewards land in a sane range and terminations stay sparse (~2%), matching a
mid-training regime.

Field provenance (arrays consumed by tracking.py reward/termination):

  motion_data.body_pos_w      (N, 14, 3)   _reward_motion_global_root_pos / body_pos
  motion_data.body_quat_w     (N, 14, 4)   _reward_motion_global_root_ori / body_ori
  motion_data.body_lin_vel_w  (N, 14, 3)   _reward_motion_body_lin_vel
  motion_data.body_ang_vel_w  (N, 14, 3)   _reward_motion_body_ang_vel
  motion_data.joint_pos/vel   (N, 29)      _reward_motion_joint_pos/vel  (scale 0)
  ref_body_pos_relative_w     (N, 14, 3)   _reward_motion_body_pos / ee / termination
  ref_body_quat_relative_w    (N, 14, 4)   _reward_motion_body_ori
  robot_body_*_w              same shapes  robot side of every error
  dof_pos / dof_vel           (N, 29)
  current_actions/last_actions(N, 29)      _reward_action_rate_l2
  joint_lower / joint_upper   (29,)        _reward_joint_limit
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec


@dataclass
class Batch:
    """All arrays a single ``update_state`` call reads (float32, C-contiguous)."""

    # motion reference (world frame)
    motion_body_pos_w: np.ndarray
    motion_body_quat_w: np.ndarray
    motion_body_lin_vel_w: np.ndarray
    motion_body_ang_vel_w: np.ndarray
    motion_joint_pos: np.ndarray
    motion_joint_vel: np.ndarray
    # motion reference expressed in the anchor frame (body_pos/quat_relative_w)
    ref_body_pos_relative_w: np.ndarray
    ref_body_quat_relative_w: np.ndarray
    # robot state (world frame)
    robot_body_pos_w: np.ndarray
    robot_body_quat_w: np.ndarray
    robot_body_lin_vel_w: np.ndarray
    robot_body_ang_vel_w: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    current_actions: np.ndarray
    last_actions: np.ndarray
    # static joint limits (shared across envs)
    joint_lower: np.ndarray
    joint_upper: np.ndarray

    @property
    def num_envs(self) -> int:
        return self.dof_pos.shape[0]


def _unit_quats(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Random unit quaternions (wxyz), last axis size 4."""
    q = rng.standard_normal((*shape, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def _perturb_quats(rng: np.random.Generator, q: np.ndarray, sigma: float) -> np.ndarray:
    """Small perturbation of unit quats, renormalised — keeps ori error modest."""
    out = q + sigma * rng.standard_normal(q.shape).astype(np.float32)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def make_batch(num_envs: int, seed: int = 0, dtype=np.float32) -> Batch:
    """Build a reproducible batch of ``num_envs`` environments.

    Faithful tracking regime: motion reference, the anchor-relative reference,
    and the robot all track a common per-env target with small errors, so
    exponential rewards are non-degenerate and terminations stay sparse (~2%,
    per issue #663) — the robot is *mostly* following the clip.
    """
    rng = np.random.default_rng(seed)
    n, nb, na = num_envs, spec.N_BODY, spec.NUM_ACTION

    def f32(a):
        return np.ascontiguousarray(a, dtype=dtype)

    # ── common per-env targets the robot is tracking ────────────────────────
    tgt_pos = rng.uniform(-1.0, 1.0, (n, nb, 3)).astype(np.float32)
    tgt_quat = _unit_quats(rng, (n, nb))
    tgt_lin_vel = rng.uniform(-2.0, 2.0, (n, nb, 3)).astype(np.float32)
    tgt_ang_vel = rng.uniform(-3.0, 3.0, (n, nb, 3)).astype(np.float32)
    tgt_jp = rng.uniform(-1.0, 1.0, (n, na)).astype(np.float32)
    tgt_jv = rng.uniform(-2.0, 2.0, (n, na)).astype(np.float32)

    # small per-source jitter around the shared target
    p, v, qs = 0.04, 0.10, 0.02

    # ── motion reference (world) ≈ target ───────────────────────────────────
    motion_body_pos_w = f32(tgt_pos + p * rng.standard_normal((n, nb, 3)))
    motion_body_quat_w = f32(_perturb_quats(rng, tgt_quat, qs))
    motion_body_lin_vel_w = f32(tgt_lin_vel + v * rng.standard_normal((n, nb, 3)))
    motion_body_ang_vel_w = f32(tgt_ang_vel + v * rng.standard_normal((n, nb, 3)))
    motion_joint_pos = f32(tgt_jp + 0.03 * rng.standard_normal((n, na)))
    motion_joint_vel = f32(tgt_jv + 0.05 * rng.standard_normal((n, na)))

    # ── anchor-relative reference ≈ target ──────────────────────────────────
    ref_body_pos_relative_w = f32(tgt_pos + p * rng.standard_normal((n, nb, 3)))
    ref_body_quat_relative_w = f32(_perturb_quats(rng, tgt_quat, qs))

    # ── robot state ≈ target + slightly larger error ────────────────────────
    robot_body_pos_w = f32(tgt_pos + 0.06 * rng.standard_normal((n, nb, 3)))
    robot_body_quat_w = f32(_perturb_quats(rng, tgt_quat, 0.03))
    robot_body_lin_vel_w = f32(tgt_lin_vel + v * rng.standard_normal((n, nb, 3)))
    robot_body_ang_vel_w = f32(tgt_ang_vel + v * rng.standard_normal((n, nb, 3)))
    dof_pos = f32(tgt_jp + 0.05 * rng.standard_normal((n, na)))
    dof_vel = f32(tgt_jv + 0.10 * rng.standard_normal((n, na)))
    current_actions = f32(rng.uniform(-1.0, 1.0, (n, na)))
    last_actions = f32(current_actions + 0.1 * rng.standard_normal((n, na)))

    # ── static joint limits, with dof_pos mostly inside them ─────────────────
    joint_lower = f32(np.full(na, -2.5))
    joint_upper = f32(np.full(na, 2.5))

    # Inject ~2% terminations on the anchor-Z channel so both paths exercise the
    # termination branch identically.
    n_term = max(1, int(0.02 * n))
    term_ids = rng.choice(n, size=n_term, replace=False)
    robot_body_pos_w[term_ids, spec.ANCHOR_BODY_IDX, 2] += 0.6  # > 0.25 threshold

    return Batch(
        motion_body_pos_w=motion_body_pos_w,
        motion_body_quat_w=motion_body_quat_w,
        motion_body_lin_vel_w=motion_body_lin_vel_w,
        motion_body_ang_vel_w=motion_body_ang_vel_w,
        motion_joint_pos=motion_joint_pos,
        motion_joint_vel=motion_joint_vel,
        ref_body_pos_relative_w=ref_body_pos_relative_w,
        ref_body_quat_relative_w=ref_body_quat_relative_w,
        robot_body_pos_w=robot_body_pos_w,
        robot_body_quat_w=robot_body_quat_w,
        robot_body_lin_vel_w=robot_body_lin_vel_w,
        robot_body_ang_vel_w=robot_body_ang_vel_w,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        current_actions=current_actions,
        last_actions=last_actions,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
    )
