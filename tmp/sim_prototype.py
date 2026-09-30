#!/usr/bin/env python3
"""Run the SONICMimic actor contract in a small MuJoCo deployment prototype.

Only the actor group is assembled: ten-frame histories of gravity, gyro, joint
position/velocity and action, plus ten future motion-reference frames. Critic
observations are privileged and never enter deployment. Action latency is not
emulated because the effective SONICMimic configuration has it disabled.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from unilab.utils.rotation import (  # noqa: E402
    np_matrix_first_two_cols_from_quat,
    np_quat_apply_inverse,
    np_quat_mul,
)

RUN_DIR = REPO_ROOT / "logs/warp_sac/G1SonicMimicDR/2026-09-29_23-06-34_mjwarp"
DEFAULT_ONNX = RUN_DIR / "policy.onnx"
DEFAULT_CFG = REPO_ROOT / "tmp/sonicmimic_dr_140k_deploy_config.yaml"
DEFAULT_MOTION = REPO_ROOT / "src/unilab/assets/motions/g1/dance1_subject2_part.npz"
DEFAULT_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"


def load_motion(path: Path) -> dict[str, np.ndarray | int]:
    """Load the training NPZ or the legacy deployment binary."""
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                        "body_lin_vel_w", "body_ang_vel_w")
            missing = [key for key in required if key not in data]
            if missing:
                raise SystemExit(f"motion NPZ missing fields: {missing}")
            result: dict[str, np.ndarray | int] = {
                key: np.asarray(data[key], dtype=np.float32).copy() for key in required
            }
            result["fps"] = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50
        result["num_frames"] = int(result["joint_pos"].shape[0])
        result["num_joints"] = int(result["joint_pos"].shape[1])
        result["num_bodies"] = int(result["body_pos_w"].shape[1])
        return result

    with path.open("rb") as stream:
        fps, nf, nj, nb = struct.unpack("<iiii", stream.read(16))
        fields = {"joint_pos": (nf, nj), "joint_vel": (nf, nj),
                  "body_pos_w": (nf, nb, 3), "body_quat_w": (nf, nb, 4),
                  "body_lin_vel_w": (nf, nb, 3), "body_ang_vel_w": (nf, nb, 3)}
        result = {"fps": fps, "num_frames": nf, "num_joints": nj, "num_bodies": nb}
        for name, shape in fields.items():
            result[name] = np.frombuffer(
                stream.read(int(np.prod(shape)) * 4), dtype="<f4"
            ).reshape(shape).copy()
    return result


class ObsAssembler:
    """Flatten per-term history buffers in deploy-contract order."""

    def __init__(self, cfg: dict) -> None:
        self.obs_dim = int(cfg["obs_dim"])
        self.layout = cfg["obs_layout"]
        if cfg.get("use_gym_history", False):
            raise SystemExit("use_gym_history=true is unsupported")
        self.buffers: dict[str, np.ndarray] = {}
        total = 0
        for term in self.layout:
            name, dim = term["name"], int(term["dim"])
            history = int(term.get("history_length", 1))
            if history < 1:
                raise SystemExit(f"invalid history_length for {name}: {history}")
            self.buffers[name] = np.zeros((history, dim), dtype=np.float32)
            total += history * dim
        if total != self.obs_dim:
            raise SystemExit(f"layout total {total} != obs_dim {self.obs_dim}")
        self.primed = False

    def _write(self, name: str, value: np.ndarray) -> None:
        buf = self.buffers[name]
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
        if flat.size != buf.shape[1]:
            raise SystemExit(f"segment {name}: expected {buf.shape[1]}, got {flat.size}")
        buf[-1] = flat

    def push(self, segments: dict[str, np.ndarray]) -> np.ndarray:
        for term in self.layout:
            name = term["name"]
            if name not in segments:
                raise SystemExit(f"no provider for obs segment {name}")
            if not self.primed:
                self._write(name, segments[name])
                self.buffers[name][:] = self.buffers[name][-1]
            else:
                self.buffers[name][:-1] = self.buffers[name][1:]
                self._write(name, segments[name])
        self.primed = True
        obs = np.concatenate([self.buffers[t["name"]].reshape(-1) for t in self.layout])
        if obs.size != self.obs_dim or not np.all(np.isfinite(obs)):
            raise SystemExit(f"invalid assembled observation (shape={obs.shape})")
        return obs.astype(np.float32, copy=False)


def future_reference(motion: dict, frame: int, future_steps: list[int],
                     robot_anchor_quat: np.ndarray, anchor_motion_idx: int) -> np.ndarray:
    """Match motion_command_multi_future: [joint_pos, joint_vel, ori6d]."""
    last = int(motion["num_frames"]) - 1
    frames = [min(frame + int(step), last) for step in future_steps]
    q_ref = np.asarray(motion["body_quat_w"])[frames, anchor_motion_idx]
    q_robot_inv = np.array([robot_anchor_quat[0], -robot_anchor_quat[1],
                            -robot_anchor_quat[2], -robot_anchor_quat[3]], dtype=np.float32)
    q_rel = np.asarray([np_quat_mul(q_robot_inv, q) for q in q_ref], dtype=np.float32)
    ori = np.asarray(np_matrix_first_two_cols_from_quat(q_rel), dtype=np.float32)
    pos = np.asarray(motion["joint_pos"])[frames]
    vel = np.asarray(motion["joint_vel"])[frames]
    return np.concatenate((pos, vel, ori), axis=-1).reshape(-1).astype(np.float32)


def compute_segments(cfg: dict, motion: dict, frame: int, robot_root_quat: np.ndarray,
                     robot_anchor_quat: np.ndarray, gyro: np.ndarray,
                     dof_pos: np.ndarray, dof_vel: np.ndarray,
                     last_action: np.ndarray) -> dict[str, np.ndarray]:
    default = np.asarray(cfg["default_angles"], dtype=np.float32)
    gravity = np_quat_apply_inverse(
        robot_root_quat, np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    )
    return {
        "command_multi_future": future_reference(
            motion, frame, [int(x) for x in cfg["future_steps"]],
            robot_anchor_quat, int(cfg["anchor_body_idx_in_motion"])
        ),
        "projected_gravity": np.asarray(gravity, dtype=np.float32),
        "base_ang_vel": np.asarray(gyro, dtype=np.float32),
        "joint_pos": (dof_pos - default).astype(np.float32),
        "joint_vel": np.asarray(dof_vel, dtype=np.float32),
        "actions": np.asarray(last_action, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--init-mode", choices=("rsi", "stand"), default="rsi")
    args = parser.parse_args()

    with args.config.open() as stream:
        cfg = yaml.safe_load(stream)
    motion = load_motion(args.motion)
    if int(motion["num_joints"]) != int(cfg["action_dim"]):
        raise SystemExit("motion/action joint dimensions do not match")
    assembler = ObsAssembler(cfg)
    expected_dim, action_dim = int(cfg["obs_dim"]), int(cfg["action_dim"])

    session = None
    if not args.no_onnx:
        if not args.onnx.exists():
            raise SystemExit(f"ONNX policy not found: {args.onnx}")
        import onnxruntime as ort
        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
        input_dim, output_dim = input_meta.shape[-1], output_meta.shape[-1]
        if input_dim != expected_dim or output_dim != action_dim:
            raise SystemExit(f"ONNX shape {input_meta.shape}/{output_meta.shape} != "
                             f"{expected_dim}/{action_dim}")
        print(f"ONNX: {args.onnx} input={input_meta.shape} output={output_meta.shape}")
    else:
        input_meta = output_meta = None
        print("ONNX disabled: observation-only sanity run")

    print("obs layout: " + ", ".join(
        f"{term['name']}({term['dim']}x{term.get('history_length', 1)})"
        for term in cfg["obs_layout"]
    ))
    print(f"obs_dim={expected_dim}, action_dim={action_dim}, fps={motion['fps']}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    ctrl_dt, sim_dt = float(cfg["ctrl_dt"]), float(model.opt.timestep)
    substeps = max(1, int(round(ctrl_dt / sim_dt)))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise SystemExit("scene has no stand keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["anchor_body_name"])
    if pelvis_id < 0 or anchor_id < 0:
        raise SystemExit("pelvis/anchor body not found")
    if args.init_mode == "rsi":
        # The training motion is pelvis-rooted (pelvis position is zero). Keep
        # stand xyz while applying its reference orientation and joint pose.
        data.qpos[3:7] = np.asarray(motion["body_quat_w"])[0, 0]
        data.qpos[7:] = np.asarray(motion["joint_pos"])[0]
        data.qvel[3:6] = np.asarray(motion["body_ang_vel_w"])[0, 0]
        data.qvel[6:] = np.asarray(motion["joint_vel"])[0]
    mujoco.mj_forward(model, data)
    print(f"sim_dt={sim_dt:.6f}, ctrl_dt={ctrl_dt:.3f}, substeps={substeps}, init={args.init_mode}")

    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_gyro")
    if sensor_id < 0:
        raise SystemExit("pelvis_gyro sensor not found")
    sensor_adr, sensor_dim = int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id])
    if sensor_dim != 3:
        raise SystemExit(f"pelvis_gyro dimension is {sensor_dim}, expected 3")

    default = np.asarray(cfg["default_angles"], dtype=np.float32)
    lower, upper = np.asarray(cfg["joint_lower"], dtype=np.float32), np.asarray(cfg["joint_upper"], dtype=np.float32)
    scale, ema = float(cfg["action_scale"]), float(cfg.get("ema_alpha", 1.0))
    last_action = np.zeros(action_dim, dtype=np.float32)
    q_target, nframes = default.copy(), int(motion["num_frames"])
    total = args.max_steps if args.max_steps > 0 else nframes
    norms, amplitudes, z_errors = [], [], []
    viewer = None
    if args.render:
        from mujoco import viewer
        viewer = viewer.launch_passive(model, data)
    wall_start = time.time()

    for step in range(total):
        frame = step % nframes
        root_quat = np.asarray(data.xquat[pelvis_id], dtype=np.float32)
        anchor_quat = np.asarray(data.xquat[anchor_id], dtype=np.float32)
        segments = compute_segments(
            cfg, motion, frame, root_quat, anchor_quat,
            np.asarray(data.sensordata[sensor_adr:sensor_adr + 3]),
            np.asarray(data.qpos[7:]), np.asarray(data.qvel[6:]), last_action,
        )
        obs = assembler.push(segments)
        if session is not None:
            action = np.asarray(session.run([output_meta.name], {
                input_meta.name: obs[None, :]
            })[0][0], dtype=np.float32)
            if not np.all(np.isfinite(action)):
                raise SystemExit(f"non-finite action at step {step}")
            last_action = action
            target = np.clip(action * scale + default, lower, upper)
            q_target = ema * target + (1.0 - ema) * q_target
        else:
            q_target = default.copy()
        data.ctrl[:] = q_target
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        norms.append(float(np.linalg.norm(obs)))
        amplitudes.append(float(np.max(np.abs(last_action))))
        ref_z = float(np.asarray(motion["body_pos_w"])[frame, int(cfg["anchor_body_idx_in_motion"]), 2])
        z_errors.append(abs(float(data.xpos[anchor_id, 2]) - ref_z))
        if not np.all(np.isfinite(data.qpos)):
            print(f"!! non-finite qpos at step {step}; aborting")
            break
        if step % 50 == 0:
            print(f"step {step:4d} frame={frame:3d} obs_norm={norms[-1]:7.2f} "
                  f"|action|={amplitudes[-1]:5.2f} z_err={z_errors[-1]:.3f}m")
        if viewer is not None:
            viewer.sync()
            delay = (step + 1) * ctrl_dt - (time.time() - wall_start)
            if delay > 0:
                time.sleep(delay)

    n = len(norms)
    print(f"Ran {n} control steps ({n * ctrl_dt:.2f}s).")
    print(f"obs_norm: mean={np.mean(norms):.3f} max={np.max(norms):.3f}")
    print(f"|action|: mean={np.mean(amplitudes):.3f} max={np.max(amplitudes):.3f}")
    print(f"anchor z error: mean={np.mean(z_errors):.4f}m max={np.max(z_errors):.4f}m")
    if viewer is not None:
        viewer.close()
    if n != total:
        raise SystemExit("prototype terminated before requested steps")
    print(f"PROTOTYPE OK — finite SONIC actor obs ({expected_dim}) and actions ({action_dim}).")


if __name__ == "__main__":
    main()
