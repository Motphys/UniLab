from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SIM2SIM_PATH = ROOT_DIR / "scripts/play_go2_arm_onnx_sim2sim.py"

DEFAULT_TARGETS = np.asarray(
    [
        [0.15, 0.0, 0.25],
        [0.20, 0.0, 0.25],
        [0.25, 0.0, 0.25],
        [0.30, 0.0, 0.25],
        [0.25, 0.08, 0.25],
        [0.25, -0.08, 0.25],
        [0.25, 0.0, 0.15],
        [0.25, 0.0, 0.35],
    ],
    dtype=np.float64,
)


@dataclass
class CandidateResult:
    roll: float
    pitch_offset: float
    mean_pos_err: float
    max_pos_err: float
    mean_orn_err: float
    max_orn_err: float
    min_margin: float
    score: float
    feasible: bool


def _load_sim2sim_module() -> Any:
    spec = importlib.util.spec_from_file_location("play_go2_arm_onnx_sim2sim", SIM2SIM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SIM2SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_target(raw: list[float]) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"--target must have 3 values, got {raw}")
    return arr


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep default_orn_roll and arm_induced_pitch for Go2Arm EE orientation."
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="Exported policy.onnx used by the sim2sim script.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=ROOT_DIR / "src/unilab/assets/robots/go2_arm/scene_flat.xml",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        action="append",
        help="Target positions in arm-base local frame. Repeatable.",
    )
    parser.add_argument("--roll-min", type=float, default=-np.pi)
    parser.add_argument("--roll-max", type=float, default=np.pi)
    parser.add_argument("--roll-steps", type=int, default=9)
    parser.add_argument("--pitch-min", type=float, default=0.0)
    parser.add_argument("--pitch-max", type=float, default=1.2)
    parser.add_argument("--pitch-steps", type=int, default=13)
    parser.add_argument("--steps-per-target", type=int, default=120)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--ctrl-dt", type=float, default=0.02)
    parser.add_argument("--limit-margin-threshold", type=float, default=0.05)
    parser.add_argument("--pos-threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser


def _build_sim2sim_args(
    module: Any, args: argparse.Namespace, target: np.ndarray
) -> argparse.Namespace:
    base_argv = [
        "--onnx-path",
        str(args.onnx_path),
        "--model-file",
        str(args.model_file),
        "--command-source",
        "zero",
        "--target-mode",
        "fixed",
        "--target-orientation-mode",
        "auto",
        "--initial-goal",
        *(str(x) for x in target.tolist()),
        "--arm-action-scale",
        "0.0",
        "--zero-arm-action",
        "--lock-legs-at-zero-command",
        "--sim-dt",
        str(args.sim_dt),
        "--ctrl-dt",
        str(args.ctrl_dt),
        "--headless-steps",
        "0",
    ]
    return module.parse_args(base_argv)


def _reset_candidate(
    module: Any,
    ctx: dict[str, Any],
    module_args: argparse.Namespace,
    target_local: np.ndarray,
    *,
    roll: float,
    pitch_offset: float,
) -> None:
    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData = ctx["data"]
    state = ctx["state"]

    mujoco.mj_resetDataKeyframe(model, data, ctx["home_id"])
    mujoco.mj_forward(model, data)

    state.command[:] = 0.0
    state.phase = 0.0
    state.control_step = 0
    state.last_action[:] = 0.0
    state.target_local_pos[:] = target_local
    state.target_local_quat[:] = module._target_quat_from_goal(
        target_local,
        default_orn_roll=roll,
        arm_induced_pitch=pitch_offset,
    )

    target_world = module._local_to_world(data, ctx["armbase_site_id"], state.target_local_pos)
    target_world_quat = module._local_quat_to_world(
        data,
        ctx["armbase_site_id"],
        state.target_local_quat,
    )
    data.mocap_pos[ctx["mocap_id"]] = target_world
    data.mocap_quat[ctx["mocap_id"]] = target_world_quat
    state.last_target_world[:] = target_world
    state.last_target_world_quat[:] = target_world_quat
    mujoco.mj_forward(model, data)

    module_args.default_orn_roll = float(roll)
    module_args.arm_induced_pitch = float(pitch_offset)
    module_args.initial_goal = target_local.copy()


def _evaluate_candidate(
    module: Any,
    ctx: dict[str, Any],
    module_args: argparse.Namespace,
    target_local: np.ndarray,
    *,
    roll: float,
    pitch_offset: float,
    steps_per_target: int,
    margin_threshold: float,
    pos_threshold: float,
) -> dict[str, float | bool]:
    _reset_candidate(
        module,
        ctx,
        module_args,
        target_local,
        roll=roll,
        pitch_offset=pitch_offset,
    )

    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData = ctx["data"]
    qpos_ids = ctx["qpos_ids"]
    arm_qpos_ids = qpos_ids[12:]
    low = model.actuator_ctrlrange[12:, 0]
    high = model.actuator_ctrlrange[12:, 1]

    pos_hist: list[float] = []
    orn_hist: list[float] = []
    margin_hist: list[float] = []

    for _ in range(steps_per_target):
        status = module._run_control_step(ctx, module_args)
        pos_hist.append(float(status["ee_err"]))
        orn_hist.append(float(status["orn_err"]))
        arm_qpos = data.qpos[arm_qpos_ids].copy()
        margin = float(np.min(np.minimum(arm_qpos - low, high - arm_qpos)))
        margin_hist.append(margin)

    tail_len = max(10, steps_per_target // 4)
    tail_pos = np.asarray(pos_hist[-tail_len:], dtype=np.float64)
    tail_orn = np.asarray(orn_hist[-tail_len:], dtype=np.float64)
    min_margin = float(np.min(margin_hist))
    mean_pos = float(np.mean(tail_pos))
    max_pos = float(np.max(tail_pos))
    mean_orn = float(np.mean(tail_orn))
    max_orn = float(np.max(tail_orn))

    feasible = mean_pos <= pos_threshold and min_margin >= margin_threshold
    score = (
        mean_pos
        + 0.25 * mean_orn
        + 8.0 * max(0.0, margin_threshold - min_margin)
        + 4.0 * max(0.0, mean_pos - pos_threshold)
    )
    return {
        "mean_pos_err": mean_pos,
        "max_pos_err": max_pos,
        "mean_orn_err": mean_orn,
        "max_orn_err": max_orn,
        "min_margin": min_margin,
        "score": score,
        "feasible": feasible,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    targets = np.asarray(args.target, dtype=np.float64) if args.target else DEFAULT_TARGETS
    if args.roll_steps <= 0 or args.pitch_steps <= 0:
        raise ValueError("--roll-steps and --pitch-steps must be positive")
    if args.steps_per_target <= 0:
        raise ValueError("--steps-per-target must be positive")

    module = _load_sim2sim_module()
    base_target = targets[0]
    module_args = _build_sim2sim_args(module, args, base_target)
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = module._build_context(module_args)
    ctx["home_id"] = module._named_id(ctx["model"], mujoco.mjtObj.mjOBJ_KEY, "home")

    rolls = np.linspace(args.roll_min, args.roll_max, args.roll_steps, dtype=np.float64)
    pitch_offsets = np.linspace(args.pitch_min, args.pitch_max, args.pitch_steps, dtype=np.float64)

    rows: list[CandidateResult] = []
    for roll in rolls:
        for pitch_offset in pitch_offsets:
            per_target = []
            for target in targets:
                metrics = _evaluate_candidate(
                    module,
                    ctx,
                    module_args,
                    target,
                    roll=float(roll),
                    pitch_offset=float(pitch_offset),
                    steps_per_target=args.steps_per_target,
                    margin_threshold=args.limit_margin_threshold,
                    pos_threshold=args.pos_threshold,
                )
                per_target.append(metrics)

            mean_pos = float(np.mean([item["mean_pos_err"] for item in per_target]))
            max_pos = float(np.max([item["max_pos_err"] for item in per_target]))
            mean_orn = float(np.mean([item["mean_orn_err"] for item in per_target]))
            max_orn = float(np.max([item["max_orn_err"] for item in per_target]))
            min_margin = float(np.min([item["min_margin"] for item in per_target]))
            feasible = all(bool(item["feasible"]) for item in per_target)
            score = (
                mean_pos
                + 0.25 * mean_orn
                + 8.0 * max(0.0, args.limit_margin_threshold - min_margin)
                + 4.0 * max(0.0, mean_pos - args.pos_threshold)
            )
            rows.append(
                CandidateResult(
                    roll=float(roll),
                    pitch_offset=float(pitch_offset),
                    mean_pos_err=mean_pos,
                    max_pos_err=max_pos,
                    mean_orn_err=mean_orn,
                    max_orn_err=max_orn,
                    min_margin=min_margin,
                    score=score,
                    feasible=feasible,
                )
            )

    rows.sort(key=lambda item: (not item.feasible, item.score, item.mean_pos_err, item.min_margin))

    print("Top candidates:")
    for row in rows[: max(1, args.top_k)]:
        print(
            f"roll={row.roll:+.3f} pitch_offset={row.pitch_offset:+.3f} "
            f"mean_pos={row.mean_pos_err:.4f} max_pos={row.max_pos_err:.4f} "
            f"mean_orn={row.mean_orn_err:.4f} max_orn={row.max_orn_err:.4f} "
            f"min_margin={row.min_margin:.4f} score={row.score:.4f} feasible={row.feasible}"
        )

    best = rows[0]
    print(
        "\nRecommended start point: "
        f"default_orn_roll={best.roll:+.3f}, arm_induced_pitch={best.pitch_offset:+.3f}"
    )

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "roll",
                    "pitch_offset",
                    "mean_pos_err",
                    "max_pos_err",
                    "mean_orn_err",
                    "max_orn_err",
                    "min_margin",
                    "score",
                    "feasible",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)
        print(f"\nSaved CSV to {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
