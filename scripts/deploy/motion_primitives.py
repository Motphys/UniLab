#!/usr/bin/env python3
"""Shared WBT motion-bin primitives for the deploy warmup/cooldown scripts.

Single owner of the interpolation and bin-I/O primitives that
``prepend_warmup.py`` and ``append_cooldown.py`` previously each carried a
verbatim copy of (#983 conclusion 4.2):

- ``load_motion_bin`` / ``save_motion_bin`` — flat binary layout shared with
  State_WBT.cpp / sim_prototype.py / export_motion_bin.py: header
  (4 int32: fps, F, J, B) then six float32 blocks.
- ``compute_fixstand_body_states`` — MuJoCo FK at the 'stand' keyframe over
  the same tracked body ids the C++ State_WBT consumes.
- ``hermite`` — cubic Hermite position + analytic velocity.
- ``slerp_smoothstep`` — quaternion SLERP along quintic smoothstep.
- ``quat_seq_ang_vel`` — world-frame angular velocity by finite difference.

Note: ``quat_seq_ang_vel`` deliberately keeps its own arithmetic (endpoint
one-sided differences; small-angle ``2*Δq.xyz/h``) instead of
``unilab.utils.rotation.np_quat_angular_velocity`` — the two differ at the
endpoints (one-sided vs copied-interior) and in the interior (small-angle vs
exact axis-angle), and switching would silently change generated bin content.
"""

from __future__ import annotations

import struct
from pathlib import Path

import mujoco
import numpy as np

# ----------------------------------------------------------------------------
# bin I/O — same layout State_WBT.cpp / sim_prototype.py / export_motion_bin.py
# all use: header (4 int32: fps, F, J, B) then six float32 blocks.
# ----------------------------------------------------------------------------


def load_motion_bin(path: Path) -> dict:
    with open(path, "rb") as f:
        fps, nf, nj, nb = struct.unpack("<iiii", f.read(16))
        jp = np.frombuffer(f.read(nf * nj * 4), "<f4").reshape(nf, nj).copy()
        jv = np.frombuffer(f.read(nf * nj * 4), "<f4").reshape(nf, nj).copy()
        bp = np.frombuffer(f.read(nf * nb * 3 * 4), "<f4").reshape(nf, nb, 3).copy()
        bq = np.frombuffer(f.read(nf * nb * 4 * 4), "<f4").reshape(nf, nb, 4).copy()
        bv = np.frombuffer(f.read(nf * nb * 3 * 4), "<f4").reshape(nf, nb, 3).copy()
        bav = np.frombuffer(f.read(nf * nb * 3 * 4), "<f4").reshape(nf, nb, 3).copy()
    return dict(fps=fps, nf=nf, nj=nj, nb=nb, jp=jp, jv=jv, bp=bp, bq=bq, bv=bv, bav=bav)


def save_motion_bin(path: Path, fps: int, jp, jv, bp, bq, bv, bav) -> None:
    nf, nj = jp.shape
    nb = bp.shape[1]
    assert jv.shape == jp.shape, "jv shape mismatch"
    assert bp.shape == (nf, nb, 3) and bq.shape == (nf, nb, 4), "body shape mismatch"
    assert bv.shape == (nf, nb, 3) and bav.shape == (nf, nb, 3), "body vel shape mismatch"
    with open(path, "wb") as f:
        f.write(struct.pack("<iiii", fps, nf, nj, nb))
        for arr in (jp, jv, bp, bq, bv, bav):
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


# ----------------------------------------------------------------------------
# FixStand FK — compute body world states at the 'stand' keyframe.
# Uses the same MuJoCo body ids the bin records (tracked_body_mujoco_ids from
# deploy_config.yaml), so the result is layer-compatible with the original bin.
# ----------------------------------------------------------------------------


def compute_fixstand_body_states(
    scene: Path, tracked_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise SystemExit(f"'stand' keyframe not found in {scene}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    jp = np.asarray(data.qpos[7:], dtype=np.float64).copy()
    bp = np.stack([np.asarray(data.xpos[i], dtype=np.float64).copy() for i in tracked_ids])
    bq = np.stack(
        [
            np.asarray(data.xquat[i], dtype=np.float64).copy()  # wxyz
            for i in tracked_ids
        ]
    )
    return jp, bp, bq


# ----------------------------------------------------------------------------
# Cubic Hermite — analytic position and velocity, kinematically consistent.
# Boundary conditions: p(0)=p0, p'(0)=v0, p(T)=p1, p'(T)=v1.
# Returns (p(t), p'(t)) where t may be an array with shape (N,) and p* may
# have additional trailing dims (broadcasting).
# ----------------------------------------------------------------------------


def hermite(p0, v0, p1, v1, t, T):
    s = t / T
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    h00d = 6 * s2 - 6 * s
    h10d = 3 * s2 - 4 * s + 1
    h01d = -6 * s2 + 6 * s
    h11d = 3 * s2 - 2 * s

    # Reshape s-dim coefficients for broadcasting against p*'s trailing dims.
    def _r(a):
        return a.reshape(a.shape + (1,) * (np.ndim(p0)))

    p = _r(h00) * p0 + _r(h10) * T * v0 + _r(h01) * p1 + _r(h11) * T * v1
    pd = _r(h00d / T) * p0 + _r(h10d) * v0 + _r(h01d / T) * p1 + _r(h11d) * v1
    return p, pd


# ----------------------------------------------------------------------------
# Quaternion SLERP along quintic smoothstep s(u) = 6u^5 - 15u^4 + 10u^3.
# u in [0, 1]. Operates on (4,) wxyz quaternions, returns (N, 4).
# ----------------------------------------------------------------------------


def slerp_smoothstep(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # shortest path
        q1 = -q1
        dot = -dot
    s = 6 * u**5 - 15 * u**4 + 10 * u**3
    if dot > 0.9995:  # near-parallel: lerp + renormalise
        out = (1 - s)[:, None] * q0 + s[:, None] * q1
        return out / np.linalg.norm(out, axis=-1, keepdims=True)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * s
    a = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    b = np.sin(theta) / sin_theta_0
    return a[:, None] * q0 + b[:, None] * q1


# ----------------------------------------------------------------------------
# World-frame angular velocity from a quaternion sequence by finite difference.
# Δq = q[k+1] * q[k]^-1  ;  ω_w ≈ 2 * Δq.xyz / dt  (small-angle, shortest path).
# Central difference for interior, forward/backward at endpoints.
# ----------------------------------------------------------------------------


def quat_seq_ang_vel(q_seq: np.ndarray, dt: float) -> np.ndarray:
    n = q_seq.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)

    def diff(q_a, q_b, h):  # ω over interval h, expressed in world
        aw, ax, ay, az = q_a
        bw, bx, by, bz = q_b
        # Δq = q_b * q_a^{-1}
        dw = bw * aw + bx * ax + by * ay + bz * az
        dx = -bw * ax + bx * aw - by * az + bz * ay
        dy = -bw * ay + bx * az + by * aw - bz * ax
        dz = -bw * az - bx * ay + by * ax + bz * aw
        if dw < 0.0:  # shortest path
            dw, dx, dy, dz = -dw, -dx, -dy, -dz
        return np.array([2 * dx / h, 2 * dy / h, 2 * dz / h])

    for i in range(n):
        if i == 0:
            out[i] = diff(q_seq[0], q_seq[1], dt)
        elif i == n - 1:
            out[i] = diff(q_seq[n - 2], q_seq[n - 1], dt)
        else:
            out[i] = diff(q_seq[i - 1], q_seq[i + 1], 2 * dt)
    return out
