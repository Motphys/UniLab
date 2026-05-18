#!/usr/bin/env python3
"""Minimal G1MotionTrackingDeploy ONNX runner on raw MuJoCo.

Dependencies are intentionally limited to NumPy, MuJoCo, and ONNX Runtime.
The script does not import UniLab.  It spells out the deploy policy contract:

    [motion joint pos, motion joint vel, anchor orientation, gyro,
     pelvis gyro, joint pos relative to stand pose, joint vel, last action] -> policy
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent
TASK_NAME = "G1MotionTrackingDeploy"
ALGO_LOG_NAME = "rsl_rl_ppo"
DEFAULT_MODEL_FILE = ROOT_DIR / "src/unilab/assets/robots/g1/scene_flat.xml"
DEFAULT_MOTION_FILE = ROOT_DIR / "src/unilab/assets/motions/g1/dance1_subject2_part.npz"

ROOT_BODY_NAME = "pelvis"
ANCHOR_BODY_NAME = "torso_link"
GYRO_SENSOR_NAME = "pelvis_gyro"
STAND_KEYFRAME_NAME = "stand"

NUM_ACTIONS = 29
OBS_DIM = 154
ACTION_SCALE = np.array(
    [
        0.5475464629911068,
        0.35066146637882434,
        0.5475464629911068,
        0.35066146637882434,
        0.43857731392336724,
        0.43857731392336724,
        0.5475464629911068,
        0.35066146637882434,
        0.5475464629911068,
        0.35066146637882434,
        0.43857731392336724,
        0.43857731392336724,
        0.5475464629911068,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
    ],
    dtype=np.float64,
)
CTRL_DT = 0.02
SIM_SUBSTEPS = 4
SIM_DT = CTRL_DT / SIM_SUBSTEPS


@dataclass(frozen=True)
class Motion:
    fps: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


def _resolve_path(value: str | None, *, default: Path | None, label: str) -> Path:
    path: Path | None
    if value is None:
        path = default
    else:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
    if path is None or not path.is_file():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def _latest_policy_onnx() -> Path | None:
    task_log_root = ROOT_DIR / "logs" / ALGO_LOG_NAME / TASK_NAME
    if not task_log_root.exists():
        return None
    candidates = sorted(task_log_root.glob("*/policy.onnx"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _resolve_onnx_path(value: str | None) -> Path:
    if value is not None:
        return _resolve_path(value, default=None, label="--onnx-path")
    latest = _latest_policy_onnx()
    if latest is None:
        raise SystemExit(
            "--onnx-path was not provided and no policy.onnx exists under "
            f"logs/{ALGO_LOG_NAME}/{TASK_NAME}/."
        )
    return latest


def _load_motion(path: Path, *, model: mujoco.MjModel) -> Motion:
    with np.load(path) as data:
        required = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        missing = required.difference(data.files)
        if missing:
            raise SystemExit(f"motion file is missing keys: {sorted(missing)}")
        motion = Motion(
            fps=int(np.asarray(data["fps"]).reshape(-1)[0]),
            joint_pos=np.asarray(data["joint_pos"], dtype=np.float64),
            joint_vel=np.asarray(data["joint_vel"], dtype=np.float64),
            body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float64),
            body_quat_w=np.asarray(data["body_quat_w"], dtype=np.float64),
            body_lin_vel_w=np.asarray(data["body_lin_vel_w"], dtype=np.float64),
            body_ang_vel_w=np.asarray(data["body_ang_vel_w"], dtype=np.float64),
        )
    if motion.fps != int(round(1.0 / CTRL_DT)):
        raise SystemExit(f"motion fps must be {int(round(1.0 / CTRL_DT))}, got {motion.fps}")
    if motion.joint_pos.shape != motion.joint_vel.shape:
        raise SystemExit("motion joint_pos and joint_vel shapes must match")
    if motion.joint_pos.shape[1] != NUM_ACTIONS:
        raise SystemExit(
            f"motion joint dimension must be {NUM_ACTIONS}, got {motion.joint_pos.shape[1]}"
        )
    if motion.body_pos_w.shape[1] != model.nbody:
        raise SystemExit(
            "motion body axis must use the MuJoCo body-id layout; "
            f"got {motion.body_pos_w.shape[1]} bodies, model has {model.nbody}"
        )
    return motion


def _make_session(onnx_path: Path, providers: Sequence[str]) -> ort.InferenceSession:
    available = set(ort.get_available_providers())
    selected = [provider for provider in providers if provider in available]
    if not selected:
        selected = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=selected)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        raise SystemExit("policy.onnx must have exactly one input and at least one output")
    input_shape = list(inputs[0].shape)
    output_shape = list(outputs[0].shape)
    if len(input_shape) != 2 or isinstance(input_shape[1], int) and input_shape[1] != OBS_DIM:
        raise SystemExit(f"policy ONNX input must have shape [batch, {OBS_DIM}], got {input_shape}")
    if (
        len(output_shape) != 2
        or isinstance(output_shape[1], int)
        and output_shape[1] != NUM_ACTIONS
    ):
        raise SystemExit(
            f"policy ONNX output must have shape [batch, {NUM_ACTIONS}], got {output_shape}"
        )
    return session


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _name_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    idx = int(mujoco.mj_name2id(model, objtype, name))
    if idx < 0:
        raise SystemExit(f"MuJoCo model is missing {objtype.name} named {name!r}")
    return idx


def _sensor_slice(model: mujoco.MjModel, sensor_name: str) -> slice:
    sensor_id = _name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + int(model.sensor_dim[sensor_id]))


def _stand_angles(model: mujoco.MjModel) -> np.ndarray:
    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, STAND_KEYFRAME_NAME)
    return np.asarray(model.key_qpos[key_id][-NUM_ACTIONS:], dtype=np.float64).copy()


def _reset_to_motion_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: Motion,
    *,
    frame: int,
    root_body_id: int,
) -> None:
    data.qpos[:3] = motion.body_pos_w[frame, root_body_id]
    data.qpos[3:7] = motion.body_quat_w[frame, root_body_id]
    data.qpos[7:] = motion.joint_pos[frame]
    data.qvel[:3] = motion.body_lin_vel_w[frame, root_body_id]
    data.qvel[3:6] = motion.body_ang_vel_w[frame, root_body_id]
    data.qvel[6:] = motion.joint_vel[frame]
    data.ctrl[:] = motion.joint_pos[frame]
    mujoco.mj_forward(model, data)


def _build_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: Motion,
    *,
    frame: int,
    anchor_body_id: int,
    gyro_slice: slice,
    stand_angles: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    robot_anchor_quat = np.asarray(data.xquat[anchor_body_id], dtype=np.float64)
    motion_anchor_quat = motion.body_quat_w[frame, anchor_body_id]
    anchor_rel_quat = _quat_mul(_quat_inv(robot_anchor_quat), motion_anchor_quat)
    anchor_ori = _matrix_from_quat(anchor_rel_quat)[:, :2].reshape(-1)
    obs = np.concatenate(
        [
            motion.joint_pos[frame],
            motion.joint_vel[frame],
            anchor_ori,
            np.asarray(data.sensordata[gyro_slice], dtype=np.float64),
            np.asarray(data.qpos[7:] - stand_angles, dtype=np.float64),
            np.asarray(data.qvel[6:], dtype=np.float64),
            last_action,
        ]
    ).astype(np.float32)
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"deploy observation must have shape ({OBS_DIM},), got {obs.shape}")
    return obs[None, :]


def _policy_action(
    session: ort.InferenceSession,
    *,
    input_name: str,
    output_name: str,
    obs: np.ndarray,
) -> np.ndarray:
    action = session.run([output_name], {input_name: obs})[0]
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (1, NUM_ACTIONS):
        raise RuntimeError(f"policy action must have shape (1, {NUM_ACTIONS}), got {action.shape}")
    return action[0]


def _advance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: np.ndarray,
    *,
    stand_angles: np.ndarray,
) -> None:
    data.ctrl[:] = action * ACTION_SCALE + stand_angles
    mujoco.mj_step(model, data, nstep=SIM_SUBSTEPS)


def _next_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: Motion,
    *,
    frame: int,
    root_body_id: int,
    loop_motion: bool,
) -> int:
    next_frame = frame + 1
    if next_frame < motion.num_frames:
        return next_frame
    if not loop_motion:
        return motion.num_frames - 1
    _reset_to_motion_frame(model, data, motion, frame=0, root_body_id=root_body_id)
    return 0


def _print_step(
    step: int, *, frame: int, action: np.ndarray, data: mujoco.MjData, start: float
) -> None:
    elapsed = max(time.perf_counter() - start, 1e-9)
    print(
        f"step={step} frame={frame} fps={step / elapsed:.1f} "
        f"base_z={float(data.qpos[2]):.3f} action_abs_max={float(np.max(np.abs(action))):.3f}",
        flush=True,
    )


def _run(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: Motion,
    session: ort.InferenceSession,
    root_body_id: int,
    anchor_body_id: int,
    gyro_slice: slice,
    stand_angles: np.ndarray,
    steps: int,
    start_frame: int,
    loop_motion: bool,
    log_interval: int,
    render: bool,
) -> None:
    frame = int(start_frame)
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    _reset_to_motion_frame(model, data, motion, frame=frame, root_body_id=root_body_id)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    start = time.perf_counter()

    viewer_ctx = mujoco.viewer.launch_passive(model, data) if render else None
    if viewer_ctx is None:
        for step in range(1, steps + 1):
            obs = _build_obs(
                model,
                data,
                motion,
                frame=frame,
                anchor_body_id=anchor_body_id,
                gyro_slice=gyro_slice,
                stand_angles=stand_angles,
                last_action=last_action,
            )
            action = _policy_action(
                session, input_name=input_name, output_name=output_name, obs=obs
            )
            _advance(model, data, action, stand_angles=stand_angles)
            last_action = action
            frame = _next_frame(
                model,
                data,
                motion,
                frame=frame,
                root_body_id=root_body_id,
                loop_motion=loop_motion,
            )
            if step == 1 or step % log_interval == 0:
                _print_step(step, frame=frame, action=action, data=data, start=start)
        return

    print("Opening MuJoCo viewer. Close the window or press Esc to quit.", flush=True)
    with viewer_ctx as viewer:
        step = 0
        while viewer.is_running() and step < steps:
            tick = time.perf_counter()
            obs = _build_obs(
                model,
                data,
                motion,
                frame=frame,
                anchor_body_id=anchor_body_id,
                gyro_slice=gyro_slice,
                stand_angles=stand_angles,
                last_action=last_action,
            )
            action = _policy_action(
                session, input_name=input_name, output_name=output_name, obs=obs
            )
            _advance(model, data, action, stand_angles=stand_angles)
            last_action = action
            frame = _next_frame(
                model,
                data,
                motion,
                frame=frame,
                root_body_id=root_body_id,
                loop_motion=loop_motion,
            )
            viewer.sync()
            step += 1
            if step == 1 or step % log_interval == 0:
                _print_step(step, frame=frame, action=action, data=data, start=start)
            sleep_s = CTRL_DT - (time.perf_counter() - tick)
            if sleep_s > 0.0:
                time.sleep(sleep_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deploy ONNX policy with raw MuJoCo only.")
    parser.add_argument("--onnx-path", default=None, help="Path to exported policy.onnx.")
    parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE))
    parser.add_argument("--motion-file", default=str(DEFAULT_MOTION_FILE))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--no-loop-motion", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo viewer.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="ONNX Runtime provider preference. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.log_interval < 1:
        raise SystemExit("--log-interval must be >= 1")

    np.random.seed(int(args.seed))
    model_path = _resolve_path(args.model_file, default=DEFAULT_MODEL_FILE, label="--model-file")
    motion_path = _resolve_path(
        args.motion_file, default=DEFAULT_MOTION_FILE, label="--motion-file"
    )
    onnx_path = _resolve_onnx_path(args.onnx_path)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    if model.nu != NUM_ACTIONS or model.nq != 7 + NUM_ACTIONS or model.nv != 6 + NUM_ACTIONS:
        raise SystemExit(
            "unexpected G1 model dimensions: "
            f"nq={model.nq}, nv={model.nv}, nu={model.nu}; expected 36, 35, 29"
        )
    motion = _load_motion(motion_path, model=model)
    if args.start_frame < 0 or args.start_frame >= motion.num_frames:
        raise SystemExit(f"--start-frame must be in [0, {motion.num_frames - 1}]")

    root_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY_NAME)
    anchor_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY_NAME)
    gyro_slice = _sensor_slice(model, GYRO_SENSOR_NAME)
    stand_angles = _stand_angles(model)
    providers = args.provider or ["CPUExecutionProvider"]
    session = _make_session(onnx_path, providers)
    data = mujoco.MjData(model)

    print(
        f"model={model_path} motion={motion_path} onnx={onnx_path} "
        f"obs_dim={OBS_DIM} action_dim={NUM_ACTIONS} providers={session.get_providers()}",
        flush=True,
    )
    _run(
        model=model,
        data=data,
        motion=motion,
        session=session,
        root_body_id=root_body_id,
        anchor_body_id=anchor_body_id,
        gyro_slice=gyro_slice,
        stand_angles=stand_angles,
        steps=int(args.steps),
        start_frame=int(args.start_frame),
        loop_motion=not bool(args.no_loop_motion),
        log_interval=int(args.log_interval),
        render=bool(args.render),
    )


if __name__ == "__main__":
    main()
