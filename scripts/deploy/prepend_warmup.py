#!/usr/bin/env python3
"""Prepend a stand->dance-frame-0 warmup prefix to a WBT motion bin.

Why this exists
---------------
The deploy-side FSM transitions FixStand -> State_WBT at t=0, but the original
dance bin's frame 0 is mid-dance (e.g. dance1.bin frame 0 has L_shoulder_roll
at +1.41 rad vs default_angles +0.20 — a ~70 deg jump). The training pipeline
uses Reference State Initialization (RSI): the env resets the robot AT the
motion's frame 0 pose, so the policy never sees "robot at default_angles,
commanded to track dance frame 0". That mismatch produces a transient action
burst at deploy time which exceeds the ankle's holding capacity, causing the
real robot's feet to slip for ~3 s before the policy stabilises.

Fix: insert N seconds of kinematically self-consistent interpolation frames at
the head of the bin. After this the policy sees a slow, smooth tracking task
(ramping the command from FixStand to dance frame 0), which is in-distribution
for a tracking policy. The dance frames are then concatenated verbatim.

What this does NOT change
-------------------------
- Training pipeline (no retrain needed).
- State_WBT.cpp / deploy_config.yaml / FSM yaml time_start.
- The original dance frames (they are appended unchanged after the warmup).

Interpolation scheme
--------------------
- joint_pos (J=29): cubic Hermite per joint with boundaries
    (default_angles, 0) -> (orig.jp[0], orig.jv[0])
  joint_vel is the analytic derivative of the polynomial (kinematically
  consistent with joint_pos by construction).
- body_pos (B,3): cubic Hermite per axis with boundaries
    (FK(stand), 0) -> (orig.bp[0], orig.bv[0])
  body_lin_vel is the analytic derivative.
- body_quat (B,4): SLERP along quintic smoothstep s(u) = 6u^5 - 15u^4 + 10u^3
  (s'(0) = s''(0) = s'(1) = s''(1) = 0). Shortest-path; near-parallel fallback.
- body_ang_vel: central difference on the SLERP path. The seam frame is
  differenced against the original bin's frame 0 so the discrete derivative is
  continuous across the seam.

FixStand body states come from MuJoCo FK on the 'stand' keyframe in scene XML,
using the same tracked_body_mujoco_ids the C++ State_WBT consumes — so the
warmup's frame 0 (body_pos/body_quat) is exactly what default_angles produces.

Use
---
    uv run python scripts/deploy/prepend_warmup.py \
        --input  ../deploy_ws/assets/dance1.bin \
        --output ../deploy_ws/assets/dance1_warmup.bin \
        --config ../deploy_ws/assets/deploy_config.yaml \
        --warmup-sec 1.5

Validate in sim BEFORE swapping on the real robot:
    uv run python scripts/deploy/sim_prototype.py \
        --motion ../deploy_ws/assets/dance1_warmup.bin \
        --init-mode stand --render
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_WS = REPO_ROOT.parent / "deploy_ws"
DEFAULT_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
DEFAULT_CFG = DEPLOY_WS / "assets/deploy_config.yaml"
DEFAULT_IN = DEPLOY_WS / "assets/dance1.bin"
DEFAULT_OUT = DEPLOY_WS / "assets/dance1_warmup.bin"
DEFAULT_WARMUP_SEC = 1.5


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
    return dict(fps=fps, nf=nf, nj=nj, nb=nb,
                jp=jp, jv=jv, bp=bp, bq=bq, bv=bv, bav=bav)


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

def compute_fixstand_body_states(scene: Path, tracked_ids: list[int]
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise SystemExit(f"'stand' keyframe not found in {scene}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    jp = np.asarray(data.qpos[7:], dtype=np.float64).copy()
    bp = np.stack([np.asarray(data.xpos[i], dtype=np.float64).copy()
                   for i in tracked_ids])
    bq = np.stack([np.asarray(data.xquat[i], dtype=np.float64).copy()  # wxyz
                   for i in tracked_ids])
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
    if dot < 0.0:                  # shortest path
        q1 = -q1
        dot = -dot
    s = 6 * u**5 - 15 * u**4 + 10 * u**3
    if dot > 0.9995:               # near-parallel: lerp + renormalise
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

    def diff(q_a, q_b, h):                # ω over interval h, expressed in world
        aw, ax, ay, az = q_a
        bw, bx, by, bz = q_b
        # Δq = q_b * q_a^{-1}
        dw = bw * aw + bx * ax + by * ay + bz * az
        dx = -bw * ax + bx * aw - by * az + bz * ay
        dy = -bw * ay + bx * az + by * aw - bz * ax
        dz = -bw * az - bx * ay + by * ax + bz * aw
        if dw < 0.0:                      # shortest path
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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG,
                    help="deploy_config.yaml (for default_angles, tracked ids).")
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE,
                    help="MuJoCo XML with 'stand' keyframe (for FixStand FK).")
    ap.add_argument("--warmup-sec", type=float, default=DEFAULT_WARMUP_SEC,
                    help="Length of prepended warmup interval [seconds].")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    default_angles = np.asarray(cfg["default_angles"], dtype=np.float64)
    tracked_ids = list(cfg["tracked_body_mujoco_ids"])

    orig = load_motion_bin(args.input)
    fps, J, B = orig["fps"], orig["nj"], orig["nb"]
    if J != len(default_angles):
        raise SystemExit(f"J mismatch: bin J={J}, default_angles len={len(default_angles)}")
    if B != len(tracked_ids):
        raise SystemExit(f"B mismatch: bin B={B}, tracked_body_mujoco_ids len={len(tracked_ids)}")

    dt = 1.0 / fps
    N = int(round(args.warmup_sec * fps))
    if N < 2:
        raise SystemExit(f"warmup-sec={args.warmup_sec}s too short for fps={fps} (need >= 2 frames)")
    T = N * dt   # so frame N in the new bin == original frame 0 exactly

    fk_jp, fk_bp, fk_bq = compute_fixstand_body_states(args.scene, tracked_ids)

    keyframe_diff = float(np.abs(fk_jp - default_angles).max())
    if keyframe_diff > 1e-3:
        print(f"WARN: 'stand' keyframe joint pos differs from deploy_config "
              f"default_angles by {keyframe_diff:.4f} rad — using deploy_config "
              "values for warmup start (so command_joint_pos matches what "
              "FixStand actually holds on the real robot).", file=sys.stderr)

    # --- joint pos/vel: analytic Hermite -----------------------------------
    ts = np.arange(N) * dt                                     # (N,)
    jp_w, jv_w = hermite(default_angles, np.zeros(J),
                         orig["jp"][0], orig["jv"][0], ts, T)

    # --- body pos / lin_vel: analytic Hermite per axis ---------------------
    bp_w, bv_w = hermite(fk_bp, np.zeros((B, 3)),
                         orig["bp"][0], orig["bv"][0], ts, T)

    # --- body quat: SLERP along quintic smoothstep -------------------------
    u = ts / T
    bq_w = np.zeros((N, B, 4), dtype=np.float64)
    for b in range(B):
        bq_w[:, b, :] = slerp_smoothstep(fk_bq[b], orig["bq"][0, b], u)

    # --- body ang_vel: central diff, with the seam differenced against the
    # original bin's first two frames so the derivative is continuous across
    # the warmup/dance boundary --------------------------------------------
    bav_w = np.zeros((N, B, 3), dtype=np.float64)
    for b in range(B):
        ext = np.concatenate([bq_w[:, b, :], orig["bq"][:2, b, :]], axis=0)
        bav_w[:, b, :] = quat_seq_ang_vel(ext, dt)[:N]

    # --- concatenate warmup + original -------------------------------------
    new_jp = np.concatenate([jp_w, orig["jp"]], axis=0)
    new_jv = np.concatenate([jv_w, orig["jv"]], axis=0)
    new_bp = np.concatenate([bp_w, orig["bp"]], axis=0)
    new_bq = np.concatenate([bq_w, orig["bq"]], axis=0)
    new_bv = np.concatenate([bv_w, orig["bv"]], axis=0)
    new_bav = np.concatenate([bav_w, orig["bav"]], axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_motion_bin(args.output, fps, new_jp, new_jv, new_bp, new_bq, new_bv, new_bav)

    # --- report ------------------------------------------------------------
    seam_jp = float(np.abs(jp_w[-1] - orig["jp"][0]).max())
    seam_jv = float(np.abs(jv_w[-1] - orig["jv"][0]).max())
    seam_bp = float(np.abs(bp_w[-1] - orig["bp"][0]).max())
    seam_bv = float(np.abs(bv_w[-1] - orig["bv"][0]).max())
    # Quaternion seam: angle between bq_w[-1] and orig.bq[0] for each body
    seam_qang_deg = []
    for b in range(B):
        d = float(np.clip(abs(np.dot(bq_w[-1, b], orig["bq"][0, b])), 0.0, 1.0))
        seam_qang_deg.append(np.degrees(2 * np.arccos(d)))
    seam_qang_max = max(seam_qang_deg)
    # Warmup-start command joint deviation from default (rate / sec)
    init_jvel_l2 = float(np.linalg.norm(jv_w[0]))

    print(f"Wrote {args.output}")
    print(f"  fps={fps}  J={J}  B={B}")
    print(f"  frames: {orig['nf']} (orig)  -> {new_jp.shape[0]} ({N} warmup + {orig['nf']} dance)")
    print(f"  duration: {orig['nf']/fps:.2f}s -> {new_jp.shape[0]/fps:.2f}s "
          f"(warmup_sec={args.warmup_sec})")
    print(f"  init (frame 0) command_joint_pos == default_angles? "
          f"max_diff={float(np.abs(new_jp[0] - default_angles).max()):.6f} rad")
    print(f"  init (frame 0) command_joint_vel L2 = {init_jvel_l2:.6f} rad/s (should be 0)")
    print("  seam check (warmup last frame vs original frame 0; one-dt-step apart):")
    print(f"    |Δjoint_pos|_∞       = {seam_jp:.4f} rad")
    print(f"    |Δjoint_vel|_∞       = {seam_jv:.4f} rad/s")
    print(f"    |Δbody_pos|_∞        = {seam_bp:.4f} m")
    print(f"    |Δbody_lin_vel|_∞    = {seam_bv:.4f} m/s")
    print(f"    max body quat angle  = {seam_qang_max:.4f} deg")


if __name__ == "__main__":
    main()
