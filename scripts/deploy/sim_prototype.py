#!/usr/bin/env python3
"""Python prototype of State_WBT — drives the ONNX policy in MuJoCo using the
exact obs assembly the C++ State_WBT will use, so we can validate the
deploy_config.yaml + dance1.bin against training-side expectations BEFORE
writing C++.

Inputs (defaults match the artifacts produced by export_*.py):
  ~/deploy_ws/assets/policy.onnx   (optional — if missing or --no-onnx, the
                                    prototype skips inference and only sanity-
                                    checks obs assembly with default-angle ctrl)
  ~/deploy_ws/assets/deploy_config.yaml
  ~/deploy_ws/assets/dance1.bin

What this verifies:
  1. Obs assembly matches training segment-for-segment, driven by
     cfg["obs_layout"] (no hard-coded dimension) — so when training flips
     enable_zero_linvel / enable_zero_anchor_pos the prototype follows.
  2. obs total length equals cfg["obs_dim"]; ONNX input width (when provided)
     matches cfg["obs_dim"].
  3. With an ONNX file: q_target = action*2.0 + default_angles + clip + EMA
     produces motion that visually tracks the reference in MuJoCo.

Differences from training-side eval (deliberate):
  - obs construction is reimplemented in pure numpy here, NOT routed through
    the env class — this is the same code path State_WBT will reproduce in C++.
  - No noise injected on obs (matches deploy convention).
  - Robot anchor pos in world is locked to the first motion frame's torso pos
    (since real robot has no GPS/SLAM).
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from unilab.envs.common.rotation import (  # noqa: E402
    np_matrix_from_quat,
    np_subtract_frame_transforms,
)

DEFAULT_ONNX = Path.home() / "deploy_ws/assets/policy.onnx"
DEFAULT_CFG = Path.home() / "deploy_ws/assets/deploy_config.yaml"
DEFAULT_BIN = Path.home() / "deploy_ws/assets/dance1.bin"
DEFAULT_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"


def load_motion_bin(path: Path) -> dict:
    with open(path, "rb") as f:
        fps, nf, nj, nb = struct.unpack("<iiii", f.read(16))
        jp = np.frombuffer(f.read(nf * nj * 4), dtype="<f4").reshape(nf, nj).copy()
        jv = np.frombuffer(f.read(nf * nj * 4), dtype="<f4").reshape(nf, nj).copy()
        bp = np.frombuffer(f.read(nf * nb * 3 * 4), dtype="<f4").reshape(nf, nb, 3).copy()
        bq = np.frombuffer(f.read(nf * nb * 4 * 4), dtype="<f4").reshape(nf, nb, 4).copy()
        blv = np.frombuffer(f.read(nf * nb * 3 * 4), dtype="<f4").reshape(nf, nb, 3).copy()
        bav = np.frombuffer(f.read(nf * nb * 3 * 4), dtype="<f4").reshape(nf, nb, 3).copy()
    return {
        "fps": fps,
        "num_frames": nf,
        "num_joints": nj,
        "num_bodies": nb,
        "joint_pos": jp,
        "joint_vel": jv,
        "body_pos_w": bp,
        "body_quat_w": bq,
        "body_lin_vel_w": blv,
        "body_ang_vel_w": bav,
    }


def build_obs_from_layout(
    layout: list[dict], segments: dict[str, np.ndarray], obs_dim: int
) -> np.ndarray:
    """Concatenate segments in the order dictated by cfg['obs_layout'].

    Every layout entry must resolve to a known segment with matching dim.
    Raises SystemExit on any mismatch — the prototype refuses to fabricate
    a vector that disagrees with the deploy contract.

    Single-step path only; history-aware assembly goes through ``ObsAssembler``.
    """
    parts: list[np.ndarray] = []
    for seg in layout:
        name = seg["name"]
        dim = int(seg["dim"])
        if name not in segments:
            raise SystemExit(f"obs_layout segment '{name}' has no value provider in prototype")
        arr = np.asarray(segments[name], dtype=np.float32).reshape(-1)
        if arr.size != dim:
            raise SystemExit(f"obs segment '{name}': expected dim {dim}, got {arr.size}")
        parts.append(arr)
    obs = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if obs.size != obs_dim:
        raise SystemExit(f"assembled obs dim {obs.size} != cfg.obs_dim {obs_dim}")
    return obs


class ObsAssembler:
    """Schema-driven actor obs assembler with per-term history buffers.

    Mirrors the deploy-side ObservationManager + ObservationTermCfg behaviour
    byte-for-byte so sim_prototype validates the same obs vector State_WBT
    will produce on the real robot:

      * Per layout term, owns a (H, dim) deque-like ring buffer (oldest at row 0).
      * ``reset(segments)`` fills all H rows with the current value (matches
        deploy ObservationTermCfg::reset which calls add() H times).
      * ``step(segments)`` shifts oldest out, writes current at the last row
        (matches deploy add()).
      * ``assemble()`` concatenates each term's full history oldest-first,
        then concatenates across terms in layout order (matches deploy
        use_gym_history=false mode in ObservationManager::compute_group).

    History length per term is read from ``layout[i]['history_length']``,
    defaulting to 1 (single-step / no history). Total assembled dim is
    sum(dim * history_length) over all terms.
    """

    def __init__(self, cfg: dict) -> None:
        self.obs_dim = int(cfg["obs_dim"])
        self.layout = cfg["obs_layout"]
        if cfg.get("use_gym_history", False):
            raise SystemExit(
                "sim_prototype assumes use_gym_history=false (group-by-term flatten); "
                "set it to false in deploy_config.yaml or extend ObsAssembler"
            )
        self._buffers: dict[str, np.ndarray] = {}
        self._hist_len: dict[str, int] = {}
        layout_total = 0
        for seg in self.layout:
            name = seg["name"]
            dim = int(seg["dim"])
            h_len = int(seg.get("history_length", 1))
            if h_len < 1:
                raise SystemExit(f"obs term '{name}' has invalid history_length={h_len}")
            self._buffers[name] = np.zeros((h_len, dim), dtype=np.float32)
            self._hist_len[name] = h_len
            layout_total += dim * h_len
        if layout_total != self.obs_dim:
            raise SystemExit(
                f"cfg internal inconsistency: sum(dim*history_length)={layout_total} "
                f"!= obs_dim={self.obs_dim}"
            )
        self._primed = False

    def reset(self, segments: dict[str, np.ndarray]) -> np.ndarray:
        """Fill every history slot with the current segment values (deploy parity)."""
        for seg in self.layout:
            name = seg["name"]
            self._set_validated(name, segments[name])
            self._buffers[name][:] = self._buffers[name][-1:, :]
        self._primed = True
        return self.assemble()

    def step(self, segments: dict[str, np.ndarray]) -> np.ndarray:
        """Push current values; auto-primes on the first call."""
        if not self._primed:
            return self.reset(segments)
        for seg in self.layout:
            name = seg["name"]
            buf = self._buffers[name]
            buf[:-1] = buf[1:]
            self._set_validated(name, segments[name])
        return self.assemble()

    def assemble(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for seg in self.layout:
            name = seg["name"]
            parts.append(self._buffers[name].reshape(-1))
        obs = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if obs.size != self.obs_dim:
            raise SystemExit(f"assembled obs dim {obs.size} != cfg.obs_dim {self.obs_dim}")
        return obs

    def _set_validated(self, name: str, value) -> None:
        buf = self._buffers[name]
        dim = buf.shape[1]
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size != dim:
            raise SystemExit(f"obs segment '{name}': expected dim {dim}, got {arr.size}")
        buf[-1, :] = arr


def compute_obs_segments(
    cfg: dict,
    motion_frame: dict,
    *,
    robot_torso_pos_w: np.ndarray,
    robot_torso_quat_w: np.ndarray,
    gyro: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    last_actions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the current-step value of every potential obs segment.

    Returns a dict keyed by segment name. Only the segments listed in
    cfg['obs_layout'] are consumed downstream; unused keys are harmless.

    All inputs are batch-less (1D arrays). Each output segment is float32.
    """
    default_angles = np.asarray(cfg["default_angles"], dtype=np.float32)
    anchor_idx = int(cfg["anchor_body_idx_in_tracked"])

    ref_joint_pos = motion_frame["joint_pos"]
    ref_joint_vel = motion_frame["joint_vel"]
    ref_torso_pos_w = motion_frame["body_pos_w"][anchor_idx]
    ref_torso_quat_w = motion_frame["body_quat_w"][anchor_idx]

    pos_b, ori_q = np_subtract_frame_transforms(
        robot_torso_pos_w[None, :],
        robot_torso_quat_w[None, :],
        ref_torso_pos_w[None, :],
        ref_torso_quat_w[None, :],
    )
    motion_anchor_pos_b = pos_b[0].astype(np.float32)
    ori_R = np_matrix_from_quat(ori_q)[0]
    motion_anchor_ori_b = ori_R[:, :2].reshape(6).astype(np.float32)

    # linvel: deploy default = zero (no real-robot sensor on G1).
    linvel_strategy = str(cfg.get("linvel_strategy", "zero"))
    if linvel_strategy != "zero":
        raise SystemExit(
            f"linvel_strategy='{linvel_strategy}' not supported in prototype; "
            "G1 deploy must use 'zero'"
        )
    base_lin_vel = np.zeros(3, dtype=np.float32)

    return {
        "command_joint_pos": ref_joint_pos.astype(np.float32),
        "command_joint_vel": ref_joint_vel.astype(np.float32),
        "motion_anchor_pos_b": motion_anchor_pos_b,
        "motion_anchor_ori_b": motion_anchor_ori_b,
        # Two aliases so legacy schemas (which used 'linvel') still work.
        "base_lin_vel": base_lin_vel,
        "linvel": base_lin_vel,
        "gyro": gyro.astype(np.float32),
        "joint_pos_rel": (dof_pos - default_angles).astype(np.float32),
        "dof_vel": dof_vel.astype(np.float32),
        "last_actions": last_actions.astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
        help="ONNX policy. If absent or --no-onnx, prototype runs a "
        "default-angle ctrl sanity check (obs assembly only).",
    )
    ap.add_argument(
        "--no-onnx", action="store_true", help="Skip ONNX inference even if the file exists."
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--motion", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument(
        "--render", action="store_true", help="Open MuJoCo passive viewer (requires display)."
    )
    ap.add_argument(
        "--max-steps", type=int, default=0, help="0 = play to end of clip then loop once."
    )
    ap.add_argument(
        "--cheat-anchor",
        action="store_true",
        help="Use sim-true robot torso pos for anchor (debug; "
        "would not be available on real robot).",
    )
    ap.add_argument(
        "--init-mode",
        choices=["rsi", "stand"],
        default="rsi",
        help="rsi = teleport robot to motion frame 0 (training-time "
        "reset condition; default). stand = leave robot in the "
        "'stand' keyframe pose (== FixStand default_angles, "
        "matching the deploy-time FSM transition).",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    motion = load_motion_bin(args.motion)

    expected_dim = int(cfg["obs_dim"])
    layout_total = sum(int(s["dim"]) * int(s.get("history_length", 1)) for s in cfg["obs_layout"])
    if layout_total != expected_dim:
        raise SystemExit(
            f"cfg internal inconsistency: sum(obs_layout dim*history_length)={layout_total} "
            f"!= obs_dim={expected_dim}"
        )
    obs_assembler = ObsAssembler(cfg)

    use_onnx = (not args.no_onnx) and args.onnx.exists()
    sess = None
    inp_name = out_name = None
    if use_onnx:
        import onnxruntime as ort  # local import: optional in sanity mode

        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        onnx_in_shape = sess.get_inputs()[0].shape
        onnx_in_dim = int(onnx_in_shape[-1]) if isinstance(onnx_in_shape[-1], int) else -1
        print(
            f"ONNX: input={inp_name} {onnx_in_shape}, output={out_name} "
            f"{sess.get_outputs()[0].shape}"
        )
        if onnx_in_dim != expected_dim:
            raise SystemExit(
                f"ONNX input dim {onnx_in_dim} != cfg.obs_dim {expected_dim}. "
                "Retrain (or re-export ONNX) so the policy matches the deploy contract."
            )
    else:
        reason = "missing" if not args.onnx.exists() else "disabled (--no-onnx)"
        print(
            f"ONNX: {reason} — running OBS-ASSEMBLY SANITY CHECK only "
            "(ctrl = default_angles, no policy in the loop)"
        )

    print(
        f"obs_dim={expected_dim}, layout segments=["
        + ", ".join(
            f"{s['name']}({s['dim']}×{int(s.get('history_length', 1))})" for s in cfg["obs_layout"]
        )
        + "]"
    )

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    ctrl_dt = float(cfg["ctrl_dt"])
    sim_dt = float(model.opt.timestep)
    substeps = max(1, int(round(ctrl_dt / sim_dt)))
    print(f"sim_dt={sim_dt:.5f}, ctrl_dt={ctrl_dt:.3f}, substeps/ctrl={substeps}")

    # Init pose: two modes.
    #   rsi   — Reference State Initialization: teleport robot to motion frame
    #           0's pose (matches the training env's reset condition).
    #   stand — leave robot in the "stand" keyframe pose (== FixStand
    #           default_angles), which is what the C++ FSM sees at the
    #           FixStand → State_WBT transition on the real robot.
    # Use --init-mode stand to reproduce the deploy-time "first few seconds of
    # slipping while the policy snaps the robot from default_angles to dance
    # frame 0" scenario in simulation, without touching hardware.
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise SystemExit("'stand' keyframe not found")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    pelvis_id_in_tracked = 0  # 'pelvis' is first in TRACKED_BODY_NAMES
    if args.init_mode == "rsi":
        data.qpos[0:3] = motion["body_pos_w"][0, pelvis_id_in_tracked]
        data.qpos[3:7] = motion["body_quat_w"][0, pelvis_id_in_tracked]  # wxyz
        data.qpos[7:] = motion["joint_pos"][0]
        data.qvel[0:3] = motion["body_lin_vel_w"][0, pelvis_id_in_tracked]
        data.qvel[3:6] = motion["body_ang_vel_w"][0, pelvis_id_in_tracked]
        data.qvel[6:] = motion["joint_vel"][0]
    mujoco.mj_forward(model, data)
    print(f"init_mode={args.init_mode}: base xyz={data.qpos[:3]}, base quat={data.qpos[3:7]}")

    default_angles = np.asarray(cfg["default_angles"], dtype=np.float32)
    action_scale = float(cfg["action_scale"])
    ema_alpha = float(cfg["ema_alpha"])
    joint_lower = np.asarray(cfg["joint_lower"], dtype=np.float32)
    joint_upper = np.asarray(cfg["joint_upper"], dtype=np.float32)
    anchor_idx = int(cfg["anchor_body_idx_in_tracked"])
    anchor_body_name = cfg["anchor_body_name"]
    anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name)

    gyro_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
    if gyro_sid < 0:
        raise SystemExit("'gyro' sensor not found in model")
    gyro_adr = int(model.sensor_adr[gyro_sid])
    gyro_dim = int(model.sensor_dim[gyro_sid])
    if gyro_dim != 3:
        raise SystemExit(f"gyro sensor has dim {gyro_dim}, expected 3")

    robot_anchor_pos_w_locked = motion["body_pos_w"][0, anchor_idx].astype(np.float32)
    print(f"robot_anchor_pos_w (locked) = {robot_anchor_pos_w_locked}")

    last_actions = np.zeros(29, dtype=np.float32)
    q_target_smoothed = default_angles.copy()
    n_frames = motion["num_frames"]
    if args.max_steps > 0:
        total_steps = args.max_steps
    elif use_onnx:
        total_steps = n_frames
    else:
        total_steps = min(50, n_frames)  # sanity mode: short run

    obs_norms = []
    action_amplitudes = []
    z_errors = []

    viewer = None
    if args.render:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    t_wall = time.time()
    for step in range(total_steps):
        frame_idx = step % n_frames

        robot_torso_quat_w = data.xquat[anchor_body_id].astype(np.float32)
        gyro = data.sensordata[gyro_adr : gyro_adr + gyro_dim].astype(np.float32)
        dof_pos = data.qpos[7:].astype(np.float32)
        dof_vel = data.qvel[6:].astype(np.float32)

        motion_frame = {
            "joint_pos": motion["joint_pos"][frame_idx],
            "joint_vel": motion["joint_vel"][frame_idx],
            "body_pos_w": motion["body_pos_w"][frame_idx],
            "body_quat_w": motion["body_quat_w"][frame_idx],
        }
        if args.cheat_anchor:
            robot_torso_pos_w_used = data.xpos[anchor_body_id].astype(np.float32)
        else:
            robot_torso_pos_w_used = robot_anchor_pos_w_locked
        current_segments = compute_obs_segments(
            cfg,
            motion_frame,
            robot_torso_pos_w=robot_torso_pos_w_used,
            robot_torso_quat_w=robot_torso_quat_w,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            last_actions=last_actions,
        )
        # Stateful: first call auto-resets (fills history with current segments,
        # matching State_WBT.cpp's env_->reset() at FSM enter); subsequent calls
        # push current and drop oldest (matching ObservationTermCfg::add).
        obs = obs_assembler.step(current_segments)
        if not np.all(np.isfinite(obs)):
            raise SystemExit(f"non-finite obs at step {step}")

        if use_onnx:
            action = sess.run([out_name], {inp_name: obs[None, :].astype(np.float32)})[0][0]
            action = action.astype(np.float32)
            last_actions = action.copy()
            q_target = action * action_scale + default_angles
            q_target = np.clip(q_target, joint_lower, joint_upper)
            q_target_smoothed = ema_alpha * q_target + (1.0 - ema_alpha) * q_target_smoothed
        else:
            # Sanity mode: keep robot at default angles, freeze last_actions at 0.
            q_target_smoothed = default_angles.copy()

        data.ctrl[:] = q_target_smoothed

        for _ in range(substeps):
            mujoco.mj_step(model, data)

        obs_norms.append(float(np.linalg.norm(obs)))
        action_amplitudes.append(float(np.max(np.abs(last_actions))))
        robot_z = float(data.xpos[anchor_body_id, 2])
        ref_z = float(motion["body_pos_w"][frame_idx, anchor_idx, 2])
        z_errors.append(abs(robot_z - ref_z))

        if viewer is not None:
            viewer.sync()
            elapsed = time.time() - t_wall
            target = (step + 1) * ctrl_dt
            if elapsed < target:
                time.sleep(target - elapsed)

        if step % 50 == 0:
            print(
                f"step {step:4d} frame={frame_idx:3d}  "
                f"obs_norm={obs_norms[-1]:7.2f}  "
                f"|action|={action_amplitudes[-1]:5.2f}  "
                f"z_err={z_errors[-1]:.3f}m  "
                f"q_target[:3]={q_target_smoothed[:3]}"
            )

        if not np.all(np.isfinite(data.qpos)):
            print(f"!! NaN at step {step}, aborting")
            break
        if use_onnx and z_errors[-1] > 0.6:
            print(f"!! z_err {z_errors[-1]:.2f}m exceeds 0.6m, robot likely fell at step {step}")
            break

    if viewer is not None:
        viewer.close()

    n = len(obs_norms)
    print()
    print(f"Ran {n} ctrl steps ({n * ctrl_dt:.2f}s of motion).")
    print(f"obs_norm:        mean={np.mean(obs_norms):.3f}  max={np.max(obs_norms):.3f}")
    print(
        f"|action| max:    mean={np.mean(action_amplitudes):.3f}  max={np.max(action_amplitudes):.3f}"
    )
    print(f"|torso z err|:   mean={np.mean(z_errors):.4f}m  max={np.max(z_errors):.4f}m")

    if not use_onnx:
        # Sanity mode never claims tracking — only that obs assembly is finite & shaped.
        print(
            "SANITY OK — obs assembly produced finite vectors of expected width "
            f"({expected_dim}); end-to-end tracking requires running with a "
            f"matching ONNX (input dim {expected_dim})."
        )
    elif np.max(z_errors) < 0.2 and n == total_steps:
        print("PROTOTYPE OK — obs assembly + ONNX inference produces tracking behavior.")
    elif n < total_steps:
        print("WARNING: prototype aborted before clip end (see message above).")
    else:
        print("WARNING: large torso z error — obs assembly may be off, or policy weak.")


if __name__ == "__main__":
    main()
