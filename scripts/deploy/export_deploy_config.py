#!/usr/bin/env python3
"""Export deploy_config.yaml for the C++ G1-29DOF WBT deployment side.

Reads g1.xml + scene_flat.xml + tracking.py defaults to emit a single yaml
that the deploy framework (~/deploy_ws/unitree_rl_lab/.../State_WBT) can load
to drive the actor at runtime.

This file is the SINGLE SOURCE OF TRUTH for the actor obs schema:
  * Training side (tracking.py) assembles obs in the order documented here.
  * Deploy side (State_WBT.cpp + ObservationManager) reads obs_layout from
    this yaml and assembles in matching order with per-term history buffers.
  * Alignment test (tests/test_obs_alignment_g1_wbt.py) verifies both code
    paths produce byte-identical obs from the same inputs.

obs_layout schema v2 (per-term history_length):
  Each entry: {name, dim, history_length, source}
  * Reference terms (command_*, motion_anchor_*) use history_length=1
    (single-step, refs come fresh from the motion clip every tick).
  * Proprio terms (gyro, joint_pos_rel, dof_vel, last_actions) carry
    history_length=H (5 for the deploy profile), flattened oldest-first.
  * Deploy framework reads via use_gym_history=false (group-by-term mode),
    so each term independently flattens its full history → total obs_dim is
    the sum of dim * history_length over all entries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
DEFAULT_OUT = REPO_ROOT / "logs/deploy/deploy_config.yaml"

TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
ANCHOR_BODY_NAME = "torso_link"

ACTION_SCALE = 2.0
# EMA alpha for q_target smoothing on the deploy side. Training applies q_target
# directly with no smoothing; alpha=1.0 means deploy also applies directly (best
# for sim2sim correctness). Lower it (~0.5–0.8) only on real hardware if jitter
# requires smoothing — but every step of lag pushes obs out of training
# distribution, so verify sim2sim impact at the chosen alpha first.
EMA_ALPHA = 1.0
CTRL_DT = 0.02
KEYFRAME_NAME = "stand"
ROOT_QPOS_DIM = 7  # free joint: xyz + quat(wxyz)

# Default obs history length — matches mujoco_deploy.yaml's
# `noise_config.obs_history_length`. Override via --obs-history-length when
# exporting for other training profiles (e.g. mujoco.yaml uses 1).
DEFAULT_OBS_HISTORY_LENGTH = 5


def _round_list(arr, ndigits=6):
    return [round(float(v), ndigits) for v in arr]


def _build_obs_layout(num_action: int, hist_len: int,
                      enable_zero_anchor_pos: bool, enable_zero_linvel: bool):
    """Build obs_layout in the exact order tracking.py:_compute_obs assembles.

    Returns (layout_list, total_obs_dim).
    Order = single-step refs first, then per-term proprio history blocks,
    matching the training-side actor obs concatenation order.
    """
    layout = [
        # ---- single-step reference terms (history_length=1) ----
        {"name": "command_joint_pos", "dim": num_action, "history_length": 1,
         "source": "motion_ref_frame.joint_pos"},
        {"name": "command_joint_vel", "dim": num_action, "history_length": 1,
         "source": "motion_ref_frame.joint_vel"},
    ]
    if not enable_zero_anchor_pos:
        layout.append({
            "name": "motion_anchor_pos_b", "dim": 3, "history_length": 1,
            "source": "subtract_frame(robot_anchor_w, motion_anchor_w).pos",
        })
    layout.append({
        "name": "motion_anchor_ori_b", "dim": 6, "history_length": 1,
        "source": "rotation_matrix(subtract_frame(...).quat)[:, :2].flatten()",
    })
    if not enable_zero_linvel:
        layout.append({
            "name": "base_lin_vel", "dim": 3, "history_length": 1,
            "source": "imu.local_linvel (state-estimated)",
        })
    # ---- proprio terms with H-step oldest-first history ----
    layout.extend([
        {"name": "gyro", "dim": 3, "history_length": hist_len,
         "source": "imu.gyroscope"},
        {"name": "joint_pos_rel", "dim": num_action, "history_length": hist_len,
         "source": "dof_pos - default_angles"},
        {"name": "dof_vel", "dim": num_action, "history_length": hist_len,
         "source": "dof_vel"},
        {"name": "last_actions", "dim": num_action, "history_length": hist_len,
         "source": "previous raw actor output"},
    ])
    total = sum(seg["dim"] * seg["history_length"] for seg in layout)
    return layout, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE,
                    help="MuJoCo scene file containing the 'stand' keyframe.")
    ap.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--obs-history-length", type=int, default=DEFAULT_OBS_HISTORY_LENGTH,
                    help="Proprio history length H. Must match training-side "
                         "noise_config.obs_history_length. Default 5 = current "
                         "mujoco_deploy.yaml. Set 1 for the legacy 154-d schema.")
    ap.add_argument("--enable-zero-anchor-pos", action="store_true", default=True,
                    help="Drop motion_anchor_pos_b from actor obs (mjlab parity). "
                         "Matches mujoco_deploy.yaml's noise_config flag.")
    ap.add_argument("--enable-zero-linvel", action="store_true", default=True,
                    help="Drop base_lin_vel from actor obs (mjlab parity). "
                         "Matches mujoco_deploy.yaml's noise_config flag.")
    args = ap.parse_args()

    if not args.scene.exists():
        raise SystemExit(f"Scene not found: {args.scene}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))

    if model.nu != 29:
        raise SystemExit(f"Expected 29 actuators, got {model.nu}")

    # Joint names in actuator order (action[i] drives actuator i).
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]

    # kp from XML <position> actuators.
    kp = model.actuator_gainprm[:, 0].copy()  # gainprm[0] is kp for position actuator
    if not (kp > 0).all():
        raise SystemExit(f"kp parsing produced non-positive values: {kp}")
    # kv recovery: for position actuator, biasprm = [0, -kp, -kv]
    kv = -model.actuator_biasprm[:, 2].copy()
    if not (kv > 0).all():
        raise SystemExit(f"kv parsing produced non-positive values: {kv}")

    # Joint limits: skip the floating-root joint (jnt 0).
    if model.njnt < 1 + model.nu:
        raise SystemExit(f"Insufficient joints: njnt={model.njnt}")
    jnt_range = model.jnt_range[1:1 + model.nu].copy()
    joint_lower = jnt_range[:, 0]
    joint_upper = jnt_range[:, 1]

    # Force range from actuator forcerange.
    force_range = model.actuator_forcerange.copy()
    force_lower = force_range[:, 0]
    force_upper = force_range[:, 1]

    # default_angles = stand keyframe qpos[7:36].
    if model.nkey == 0:
        raise SystemExit(f"No keyframes in {args.scene}; expected '{KEYFRAME_NAME}'.")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    if key_id < 0:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        raise SystemExit(f"Keyframe '{KEYFRAME_NAME}' not found; have {names}")
    stand_qpos = model.key_qpos[key_id]
    if len(stand_qpos) != ROOT_QPOS_DIM + model.nu:
        raise SystemExit(
            f"stand qpos len {len(stand_qpos)} != {ROOT_QPOS_DIM}+{model.nu}"
        )
    default_angles = stand_qpos[ROOT_QPOS_DIM:].copy()

    # 14 tracked body indices (in MuJoCo body-id space; deploy side won't use
    # these directly but they are useful for debugging / cross-checks).
    tracked_body_ids = []
    for nm in TRACKED_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            raise SystemExit(f"Tracked body '{nm}' missing from model")
        tracked_body_ids.append(int(bid))
    anchor_body_idx_in_tracked = TRACKED_BODY_NAMES.index(ANCHOR_BODY_NAME)

    # Build the obs layout (single source of truth for actor obs assembly).
    obs_layout, obs_dim = _build_obs_layout(
        num_action=model.nu,
        hist_len=args.obs_history_length,
        enable_zero_anchor_pos=args.enable_zero_anchor_pos,
        enable_zero_linvel=args.enable_zero_linvel,
    )

    cfg = {
        # ---- meta ----
        # obs_dim = sum over obs_layout of dim * history_length.
        # For the deploy profile (H=5, both zero flags ON) this is:
        #   command_joint_pos(29*1) + command_joint_vel(29*1)
        #   + motion_anchor_ori_b(6*1) + gyro(3*5)
        #   + joint_pos_rel(29*5) + dof_vel(29*5) + last_actions(29*5) = 514
        # Aligns with deploy-profile training run (mujoco_deploy.yaml) which
        # drops motion_anchor_pos_b and base_lin_vel from actor obs to match
        # Unitree's verified mjlab "No-State-Estimation" deploy yaml and
        # adds BeyondMimic-style proprio history (default H=5).
        "obs_dim": obs_dim,
        "obs_history_length": args.obs_history_length,
        # Deploy ObservationManager mode. False = group-by-term (each term's
        # full history flattened oldest-first, then concat across terms).
        # True = group-by-time-step (legacy gym style). Training side uses
        # the false convention, so MUST stay false here for byte alignment.
        "use_gym_history": False,
        "action_dim": int(model.nu),
        "ctrl_dt": CTRL_DT,
        "action_scale": ACTION_SCALE,
        "ema_alpha": EMA_ALPHA,

        # ---- joint config (in MuJoCo actuator order; deploy side assumes
        # this matches the SDK motor index 1:1 for G1-29DOF — verify per-motor
        # before real-robot run) ----
        "joint_names": list(joint_names),
        # Identity SDK mapping; replace with measured order if 1:1 assumption fails.
        "joint_ids_map": list(range(model.nu)),
        "default_angles": _round_list(default_angles),
        "kp": _round_list(kp),
        "kd": _round_list(kv),
        "joint_lower": _round_list(joint_lower),
        "joint_upper": _round_list(joint_upper),
        "force_lower": _round_list(force_lower, 3),
        "force_upper": _round_list(force_upper, 3),

        # ---- motion / anchor ----
        "tracked_body_names": list(TRACKED_BODY_NAMES),
        "tracked_body_mujoco_ids": tracked_body_ids,
        "anchor_body_name": ANCHOR_BODY_NAME,
        "anchor_body_idx_in_tracked": int(anchor_body_idx_in_tracked),

        # ---- noise (NOT applied at deploy; documentation only — per-step
        # uniform noise scales used during training, plus persistent encoder
        # bias absorbed into joint_pos_rel) ----
        "training_noise_scales": {
            "joint_angle": 0.01,
            "joint_vel": 0.5,
            "gyro": 0.2,
            "anchor_ori": 0.05,
            "joint_pos_encoder_bias_per_episode": 0.01,
        },

        # ---- obs layout (SINGLE SOURCE OF TRUTH for both ends) ----
        # Each entry: name, dim (per-step), history_length, source.
        # Total obs_dim = sum(dim * history_length).
        # Order MUST match tracking.py:_compute_obs assembly order.
        # State_WBT.cpp:build_env_cfg translates names via its alias table.
        "obs_layout": obs_layout,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, width=120)

    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"  joints: {model.nu}, tracked bodies: {len(TRACKED_BODY_NAMES)}, "
          f"anchor='{ANCHOR_BODY_NAME}' (idx_in_tracked={anchor_body_idx_in_tracked})")
    print(f"  obs_history_length={args.obs_history_length}, total obs_dim={obs_dim}")
    print("  obs_layout segments:")
    for seg in obs_layout:
        print(f"    {seg['name']:24s} dim={seg['dim']:3d}  H={seg['history_length']:1d}  "
              f"contrib={seg['dim'] * seg['history_length']}")
    print(f"  default_angles[:6] = {_round_list(default_angles[:6], 3)}")
    print(f"  kp[:3] = {_round_list(kp[:3], 3)}, kd[:3] = {_round_list(kv[:3], 3)}")


if __name__ == "__main__":
    main()
