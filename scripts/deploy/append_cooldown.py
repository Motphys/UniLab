#!/usr/bin/env python3
"""Append a dance-final-frame->FixStand cooldown suffix to a WBT motion bin.

Why this exists
---------------
Symmetric to prepend_warmup.py.  The original dance bin's last frame is mid-
motion (joint vel ~18 rad/s L2, pelvis still descending at 0.3 m/s, joints
~42 deg from FixStand stand qpos).  When State_WBT's time-end check fires
and the FSM hands off to State_FixStand, FixStand only does a joint-space PD
interpolation from current motor q -> stand qpos over 3 s; it has no balance
controller, so a robot left with angular momentum and an off-balance pose
collapses despite the PD targets pulling it back to stand.

Fix: append N seconds of kinematically self-consistent interpolation frames
at the tail of the bin.  The tracking policy decelerates joints, brings the
torso back to upright, and lands on FixStand stand qpos before the FSM exit
triggers — so the FixStand interpolation starts from an already-balanced
pose and just holds station.

What this does NOT change
-------------------------
- Training pipeline (no retrain needed).
- State_WBT.cpp / deploy_config.yaml / FSM yaml time_end (leave null -> full
  duration so the cooldown plays automatically).
- The original dance frames (they are preserved verbatim before the cooldown).

Interpolation scheme — shared primitives from motion_primitives.py, run in
reverse direction:
- joint_pos (J=29): cubic Hermite per joint with boundaries
    (orig.jp[-1], orig.jv[-1]) -> (default_angles, 0)
  joint_vel is the analytic derivative.
- body_pos (B,3): cubic Hermite per axis with boundaries
    (orig.bp[-1], orig.bv[-1]) -> (FK(stand), 0)
  body_lin_vel is the analytic derivative.
- body_quat (B,4): SLERP along quintic smoothstep s(u) = 6u^5 - 15u^4 + 10u^3
  (s'(0) = s''(0) = s'(1) = s''(1) = 0). Shortest-path; near-parallel fallback.
- body_ang_vel: central difference over [orig.bq[-2:], cooldown_bq] so the
  seam ang_vel is continuous with the original bin's final ang_vel.

Use
---
    uv run scripts/deploy/append_cooldown.py \
        --input  ../deploy_ws/assets/dance1.bin \
        --output ../deploy_ws/assets/dance1_cooldown.bin \
        --config ../deploy_ws/assets/deploy_config.yaml \
        --cooldown-sec 2.0

Validate in sim BEFORE swapping on the real robot:
    uv run scripts/deploy/sim_prototype.py \
        --motion ../deploy_ws/assets/dance1_cooldown.bin \
        --init-mode stand --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from motion_primitives import (
    compute_fixstand_body_states,
    hermite,
    load_motion_bin,
    quat_seq_ang_vel,
    save_motion_bin,
    slerp_smoothstep,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_WS = REPO_ROOT.parent / "deploy_ws"
DEFAULT_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
DEFAULT_CFG = DEPLOY_WS / "assets/deploy_config.yaml"
DEFAULT_IN = DEPLOY_WS / "assets/dance1.bin"
DEFAULT_OUT = DEPLOY_WS / "assets/dance1_cooldown.bin"
DEFAULT_COOLDOWN_SEC = 3.0
DEFAULT_HOLD_SEC = 0.5


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CFG,
        help="deploy_config.yaml (for default_angles, tracked ids).",
    )
    ap.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="MuJoCo XML with 'stand' keyframe (for FixStand FK).",
    )
    ap.add_argument(
        "--cut-frame",
        type=int,
        default=None,
        help="If set, truncate the input bin so that frame "
        "CUT_FRAME becomes the new last frame (inclusive). "
        "Frames after that are dropped, and the cooldown ramps "
        "from this frame to default. Use this when the dance's "
        "actual last frame has too much momentum for any post-"
        "hoc interpolation to stabilize. Find a low-energy "
        "frame (low joint vel, near-upright torso, near-zero "
        "pelvis vertical vel) and cut there.",
    )
    ap.add_argument(
        "--cooldown-sec",
        type=float,
        default=DEFAULT_COOLDOWN_SEC,
        help="Length of Hermite deceleration ramp [seconds]. "
        "Lowers joint vel from orig.jv[-1] to 0 and joint pos "
        "to default_angles over this interval.",
    )
    ap.add_argument(
        "--hold-sec",
        type=float,
        default=DEFAULT_HOLD_SEC,
        help="Length of trailing static-hold segment [seconds]. "
        "Appended AFTER the Hermite ramp; frames are pure "
        "(default_angles, 0, FixStand FK body, 0). Lets the "
        "tracking policy physically converge to stand before "
        "the FSM timeout swaps to State_FixStand. Set to 0 to "
        "disable. Recommended >= 0.3s.",
    )
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

    if args.cut_frame is not None:
        cf = int(args.cut_frame)
        if cf < 1 or cf >= orig["nf"]:
            raise SystemExit(f"cut-frame={cf} out of range [1, {orig['nf'] - 1}]")
        # Keep frames [0, cf] (cf becomes the new last frame, inclusive).
        keep = cf + 1
        print(
            f"Truncating input: keep frames [0, {cf}]  "
            f"({orig['nf']} -> {keep} frames, {orig['nf'] / fps:.2f}s -> {keep / fps:.2f}s)",
            file=sys.stderr,
        )
        for k in ("jp", "jv", "bp", "bq", "bv", "bav"):
            orig[k] = orig[k][:keep]
        orig["nf"] = keep
        # Report cut-point energy so the user sees what cool-down starts from.
        a = cfg.get("anchor_body_idx_in_tracked", 7)
        jv_l2 = float(np.linalg.norm(orig["jv"][-1]))
        jp_dev = float(np.abs(orig["jp"][-1] - default_angles).max())
        bv_z = float(abs(orig["bv"][-1, a, 2]))
        print(
            f"  cut-point energy: |jv|={jv_l2:.2f} rad/s  jp_dev={jp_dev:.3f} rad  "
            f"|bv_z|={bv_z:.3f} m/s",
            file=sys.stderr,
        )

    dt = 1.0 / fps
    N = int(round(args.cooldown_sec * fps))
    if N < 2:
        raise SystemExit(
            f"cooldown-sec={args.cooldown_sec}s too short for fps={fps} (need >= 2 frames)"
        )
    K = int(round(args.hold_sec * fps)) if args.hold_sec > 0.0 else 0
    if args.hold_sec > 0.0 and K < 1:
        raise SystemExit(
            f"hold-sec={args.hold_sec}s rounds to 0 frames at fps={fps}; pass 0 or >= {dt:.3f}s"
        )
    T = N * dt  # frame N of the cooldown == FixStand target exactly

    fk_jp, fk_bp, fk_bq = compute_fixstand_body_states(args.scene, tracked_ids)

    keyframe_diff = float(np.abs(fk_jp - default_angles).max())
    if keyframe_diff > 1e-3:
        print(
            f"WARN: 'stand' keyframe joint pos differs from deploy_config "
            f"default_angles by {keyframe_diff:.4f} rad — using deploy_config "
            "values for cooldown end (so command_joint_pos matches what "
            "FixStand will hold on the real robot).",
            file=sys.stderr,
        )

    # --- joint pos/vel: analytic Hermite, orig[-1] -> default_angles --------
    # ts[k] = (k+1)*dt so the first cooldown frame is one dt after orig[-1]
    # and the last cooldown frame (k = N-1) lands exactly at t = T = N*dt,
    # i.e. precisely on the (default_angles, 0) boundary.
    ts = (np.arange(N) + 1) * dt
    jp_c, jv_c = hermite(orig["jp"][-1], orig["jv"][-1], default_angles, np.zeros(J), ts, T)

    # --- body pos / lin_vel: analytic Hermite per axis ----------------------
    bp_c, bv_c = hermite(orig["bp"][-1], orig["bv"][-1], fk_bp, np.zeros((B, 3)), ts, T)

    # --- body quat: SLERP along quintic smoothstep --------------------------
    u = ts / T
    bq_c = np.zeros((N, B, 4), dtype=np.float64)
    for b in range(B):
        bq_c[:, b, :] = slerp_smoothstep(orig["bq"][-1, b], fk_bq[b], u)

    # --- body ang_vel: central diff, with the seam differenced against the
    # ORIGINAL bin's last two frames so the derivative is continuous across
    # the dance/cooldown boundary -------------------------------------------
    bav_c = np.zeros((N, B, 3), dtype=np.float64)
    for b in range(B):
        ext = np.concatenate([orig["bq"][-2:, b, :], bq_c[:, b, :]], axis=0)
        # extended sequence has 2+N frames; cooldown frames live at [2, 2+N).
        bav_c[:, b, :] = quat_seq_ang_vel(ext, dt)[2:]

    # --- static hold segment: K frames all at (default, 0, FK, 0). Gives the
    # tracking policy time to physically converge to the FixStand target before
    # the FSM time-end check fires and hands off to State_FixStand. Without
    # this segment, the cooldown's last (default, 0) frame is commanded for
    # only one step_dt (~0.02s) before the swap, and any residual body momentum
    # carried through the Hermite ramp ends up as the FixStand starting pose. -
    if K > 0:
        jp_h = np.broadcast_to(default_angles, (K, J)).copy()
        jv_h = np.zeros((K, J), dtype=np.float64)
        bp_h = np.broadcast_to(fk_bp, (K, B, 3)).copy()
        bq_h = np.broadcast_to(fk_bq, (K, B, 4)).copy()
        bv_h = np.zeros((K, B, 3), dtype=np.float64)
        bav_h = np.zeros((K, B, 3), dtype=np.float64)

    # --- concatenate original + cooldown (+ hold) -------------------------
    parts_jp = [orig["jp"], jp_c]
    parts_jv = [orig["jv"], jv_c]
    parts_bp = [orig["bp"], bp_c]
    parts_bq = [orig["bq"], bq_c]
    parts_bv = [orig["bv"], bv_c]
    parts_bav = [orig["bav"], bav_c]
    if K > 0:
        parts_jp.append(jp_h)
        parts_jv.append(jv_h)
        parts_bp.append(bp_h)
        parts_bq.append(bq_h)
        parts_bv.append(bv_h)
        parts_bav.append(bav_h)
    new_jp = np.concatenate(parts_jp, axis=0)
    new_jv = np.concatenate(parts_jv, axis=0)
    new_bp = np.concatenate(parts_bp, axis=0)
    new_bq = np.concatenate(parts_bq, axis=0)
    new_bv = np.concatenate(parts_bv, axis=0)
    new_bav = np.concatenate(parts_bav, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_motion_bin(args.output, fps, new_jp, new_jv, new_bp, new_bq, new_bv, new_bav)

    # --- report ------------------------------------------------------------
    # Seam = boundary between orig[-1] and cooldown[0], which are one dt apart.
    seam_jp = float(np.abs(jp_c[0] - orig["jp"][-1]).max())
    seam_jv = float(np.abs(jv_c[0] - orig["jv"][-1]).max())
    seam_bp = float(np.abs(bp_c[0] - orig["bp"][-1]).max())
    seam_bv = float(np.abs(bv_c[0] - orig["bv"][-1]).max())
    seam_qang_deg = []
    for b in range(B):
        d = float(np.clip(abs(np.dot(bq_c[0, b], orig["bq"][-1, b])), 0.0, 1.0))
        seam_qang_deg.append(np.degrees(2 * np.arccos(d)))
    seam_qang_max = max(seam_qang_deg)
    # End state must match FixStand boundary (this is the whole point).
    end_jp_diff = float(np.abs(new_jp[-1] - default_angles).max())
    end_jv_l2 = float(np.linalg.norm(new_jv[-1]))

    print(f"Wrote {args.output}")
    print(f"  fps={fps}  J={J}  B={B}")
    print(
        f"  frames: {orig['nf']} (orig)  -> {new_jp.shape[0]} "
        f"({orig['nf']} dance + {N} cooldown + {K} hold)"
    )
    print(
        f"  duration: {orig['nf'] / fps:.2f}s -> {new_jp.shape[0] / fps:.2f}s "
        f"(cooldown_sec={args.cooldown_sec}, hold_sec={args.hold_sec})"
    )
    print(f"  end (last frame) command_joint_pos == default_angles? max_diff={end_jp_diff:.6f} rad")
    print(f"  end (last frame) command_joint_vel L2 = {end_jv_l2:.6f} rad/s (should be 0)")
    print("  seam check (orig last frame vs cooldown first frame; one-dt-step apart):")
    print(f"    |Δjoint_pos|_∞       = {seam_jp:.4f} rad")
    print(f"    |Δjoint_vel|_∞       = {seam_jv:.4f} rad/s")
    print(f"    |Δbody_pos|_∞        = {seam_bp:.4f} m")
    print(f"    |Δbody_lin_vel|_∞    = {seam_bv:.4f} m/s")
    print(f"    max body quat angle  = {seam_qang_max:.4f} deg")


if __name__ == "__main__":
    main()
