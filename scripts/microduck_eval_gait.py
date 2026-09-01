"""Evaluate a Manager-Based MicroDuck PPO checkpoint on fixed straight commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from rsl_rl.runners import OnPolicyRunner

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.train_rsl_rl import (  # noqa: E402
    _resolve_ppo_wrapper_cls,
    apply_ppo_runtime_flags,
    build_ppo_play_env_cfg_override,
)
from unilab.algos.rsl_rl import get_policy_obs_dims, normalize_ppo_train_cfg  # noqa: E402
from unilab.base.config_adapter import create_env  # noqa: E402
from unilab.training import (  # noqa: E402
    algo_config_dict,
    ensure_registries,
    parse_checkpoint_path,
)
from unilab.utils.checkpoint import get_entrypoint_log_root  # noqa: E402
from unilab.utils.device import get_default_device  # noqa: E402
from unilab.utils.rotation import (  # noqa: E402
    np_quat_apply_inverse,
    np_yaw_from_quat,
)
from unilab.visualization.interactive_playback import (  # noqa: E402
    RslRlPlaybackConfig,
    create_rsl_rl_playback_session,
    infer_checkpoint_actor_input_dim,
    make_sim2sim_preflight,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="microduck_velocity_flat/mujoco")
    parser.add_argument("--vx", type=float, action="append", default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument(
        "--heading-hold",
        action="store_true",
        help="Keep a zero-world-heading target and allow feedback yaw-rate commands.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _zero_ranges(width: int) -> list[list[float]]:
    return [[0.0, 0.0] for _ in range(width)]


def _compose_eval_cfg(args: argparse.Namespace, vx: float) -> DictConfig:
    overrides = [
        f"task={args.task}",
        f"algo.load_run={args.load_run}",
        f"algo.checkpoint={args.checkpoint}",
        f"training.play_env_num={args.num_envs}",
        "+env.seed=0",
        "env.commands.twist.rel_standing_envs=0.0",
        "env.commands.twist.rel_world_envs=0.0",
        "env.commands.twist.rel_forward_envs=0.0",
        "env.commands.twist.turn_in_place_fraction=0.0",
        f"env.commands.twist.ranges.lin_vel_x=[{vx},{vx}]",
        "env.commands.twist.ranges.lin_vel_y=[0.0,0.0]",
        f"env.commands.head_pose.ranges={_zero_ranges(4)}",
        f"env.commands.body_pose.ranges={_zero_ranges(6)}",
        "env.events.base_com=null",
        "env.events.head_com=null",
        "env.events.encoder_bias=null",
        "env.events.foot_friction=null",
        "env.events.randomize_armature=null",
        "env.events.push_robot=null",
        "env.curriculum.standing_envs=null",
        "env.curriculum.head_pose_range=null",
        "env.curriculum.base_com_range=null",
        "env.curriculum.head_com_range=null",
    ]
    if args.heading_hold:
        overrides.extend(
            [
                "env.commands.twist.heading_command=true",
                "env.commands.twist.rel_heading_envs=1.0",
                "env.commands.twist.ranges.ang_vel_z=[-1.0,1.0]",
                "++env.commands.twist.ranges.heading=[0.0,0.0]",
            ]
        )
    else:
        overrides.extend(
            [
                "env.commands.twist.heading_command=false",
                "env.commands.twist.rel_heading_envs=0.0",
                "env.commands.twist.ranges.ang_vel_z=[0.0,0.0]",
                "++env.commands.twist.ranges.heading=null",
            ]
        )
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "ppo"), version_base="1.3"):
        return compose("config", overrides=overrides)


def _make_session(cfg: DictConfig) -> tuple[Any, Path]:
    ensure_registries()
    device = get_default_device()
    load_path, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None:
        raise SystemExit("Could not resolve checkpoint path")

    rl_cfg = algo_config_dict(cfg)

    def normalize(train_cfg: dict[str, Any]) -> dict[str, Any]:
        result = normalize_ppo_train_cfg(train_cfg)
        apply_ppo_runtime_flags(result, cfg, training_enabled=False)
        return result

    session, _, checkpoint_path = create_rsl_rl_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task=str(cfg.training.task_name),
            load_run=str(cfg.algo.load_run),
            checkpoint=str(cfg.algo.checkpoint),
            action_mode="policy",
            policy_obs_mode="flat",
            algo_log_name=str(cfg.algo.algo_log_name),
            log_root=None,
            num_envs=int(cfg.training.play_env_num),
        ),
        env_factory=lambda n: create_env(
            cfg,
            num_envs=n,
            env_cfg_override=build_ppo_play_env_cfg_override(cfg),
        ),
        algo_config=rl_cfg,
        root_dir=ROOT_DIR,
        device=device,
        checkpoint_resolver=lambda *_args: str(load_path),
        checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
        entrypoint_log_root=get_entrypoint_log_root,
        wrapper_cls=_resolve_ppo_wrapper_cls(rl_cfg),
        runner_cls=OnPolicyRunner,
        policy_obs_dims_getter=get_policy_obs_dims,
        train_cfg_normalizer=normalize,
        sim2sim_preflight=make_sim2sim_preflight(cfg, algo_name="ppo"),
        guard_algo_name="ppo",
    )
    if checkpoint_path is None:
        raise SystemExit("Playback session did not load a checkpoint")
    return session, Path(checkpoint_path)


def _pitch_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    normalized = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = normalized.T
    return np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))


def _nanpercentile_range(values: np.ndarray, joint_indices: list[int]) -> np.ndarray:
    selected = values[:, :, joint_indices]
    return np.nanpercentile(selected, 95, axis=0) - np.nanpercentile(selected, 5, axis=0)


def _evaluate(session: Any, *, vx: float, num_steps: int) -> dict[str, float | int]:
    env = session.env
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
    contact_view = env.scene.bind_sensor_data(["left_foot_contact", "right_foot_contact"])
    contact_columns = np.asarray([0, contact_view.dimensions[0]], dtype=np.intp)
    joint_ids, _ = robot.find_joints(
        [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
        ],
        preserve_order=True,
    )

    session.reset()
    start_pos = np.asarray(robot.data.root_link_pos_w, dtype=np.float64).copy()
    start_yaw = np_yaw_from_quat(np.asarray(robot.data.root_link_quat_w, dtype=np.float64))
    latest_pos = start_pos.copy()
    latest_yaw = start_yaw.copy()
    num_envs = int(env.num_envs)
    active = np.ones(num_envs, dtype=bool)
    failed = np.zeros(num_envs, dtype=bool)
    previous_contact = contact_view.read()[:, contact_columns] > 0.1
    air_time = np.zeros((num_envs, 2), dtype=np.float64)
    last_touchdown_foot = np.full(num_envs, -1, dtype=np.int8)
    touchdown_counts = np.zeros((num_envs, 2), dtype=np.int32)
    alternating_touchdowns = np.zeros(num_envs, dtype=np.int32)
    total_touchdowns = np.zeros(num_envs, dtype=np.int32)
    contact_steps = np.zeros((num_envs, 2), dtype=np.float64)
    both_air_steps = np.zeros(num_envs, dtype=np.float64)
    double_stance_steps = np.zeros(num_envs, dtype=np.float64)
    active_steps = np.zeros(num_envs, dtype=np.float64)

    vx_values: list[np.ndarray] = []
    wz_values: list[np.ndarray] = []
    dof_values: list[np.ndarray] = []
    height_values: list[np.ndarray] = []
    pitch_values: list[np.ndarray] = []
    separation_values: list[np.ndarray] = []
    swing_height_values: list[np.ndarray] = []
    touchdown_advance_values: list[np.ndarray] = []
    touchdown_air_time_values: list[np.ndarray] = []

    with torch.inference_mode():
        for _ in range(num_steps):
            session.step_once()
            state = env.state
            done = np.asarray(state.terminated | state.truncated, dtype=bool)
            sample = active & ~done
            root_pos = np.asarray(robot.data.root_link_pos_w, dtype=np.float64)
            root_quat = np.asarray(robot.data.root_link_quat_w, dtype=np.float64)
            root_yaw = np_yaw_from_quat(root_quat)
            lin_vel_b = np.asarray(robot.data.root_link_lin_vel_b, dtype=np.float64)
            ang_vel_b = np.asarray(robot.data.root_link_ang_vel_b, dtype=np.float64)
            joint_pos = np.asarray(robot.data.joint_pos[:, joint_ids], dtype=np.float64)
            foot_pos_w = np.asarray(robot.data.body_link_pos_w[:, foot_ids], dtype=np.float64)
            relative_foot_pos = foot_pos_w - root_pos[:, None, :]
            foot_pos_b = np_quat_apply_inverse(
                np.repeat(root_quat[:, None, :], 2, axis=1).reshape(-1, 4),
                relative_foot_pos.reshape(-1, 3),
            ).reshape(num_envs, 2, 3)
            contact = contact_view.read()[:, contact_columns] > 0.1

            previous_air_time = air_time.copy()
            air_time = np.where(contact, 0.0, air_time + float(env.step_dt))
            touchdown = contact & ~previous_contact & sample[:, None] & (previous_air_time >= 0.06)
            touchdown_air_time_values.append(np.where(touchdown, previous_air_time, np.nan))
            advance = np.full(num_envs, np.nan, dtype=np.float64)
            advance[touchdown[:, 0]] = (
                foot_pos_b[touchdown[:, 0], 0, 0] - foot_pos_b[touchdown[:, 0], 1, 0]
            )
            advance[touchdown[:, 1]] = (
                foot_pos_b[touchdown[:, 1], 1, 0] - foot_pos_b[touchdown[:, 1], 0, 0]
            )
            touchdown_advance_values.append(advance)
            for foot in (0, 1):
                ids = np.flatnonzero(touchdown[:, foot])
                touchdown_counts[ids, foot] += 1
                total_touchdowns[ids] += 1
                alternating_touchdowns[ids] += last_touchdown_foot[ids] == (1 - foot)
                last_touchdown_foot[ids] = foot

            vx_values.append(np.where(sample, lin_vel_b[:, 0], np.nan))
            wz_values.append(np.where(sample, ang_vel_b[:, 2], np.nan))
            dof_values.append(np.where(sample[:, None], joint_pos, np.nan))
            height_values.append(np.where(sample, root_pos[:, 2], np.nan))
            pitch_values.append(np.where(sample, _pitch_from_quat_wxyz(root_quat), np.nan))
            separation_values.append(
                np.where(sample, foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1], np.nan)
            )
            swing_height_values.append(
                np.where((~contact) & sample[:, None], foot_pos_w[:, :, 2], np.nan)
            )
            active_steps += sample
            contact_steps += contact * sample[:, None]
            both_air_steps += (~contact.any(axis=1)) * sample
            double_stance_steps += contact.all(axis=1) * sample
            latest_pos[sample] = root_pos[sample]
            latest_yaw[sample] = root_yaw[sample]
            failed |= np.asarray(state.terminated, dtype=bool) & active
            active &= ~done
            previous_contact = contact
            if not active.any():
                break

    vx_arr = np.stack(vx_values)
    wz_arr = np.stack(wz_values)
    dof_arr = np.stack(dof_values)
    height_arr = np.stack(height_values)
    pitch_arr = np.stack(pitch_values)
    separation_arr = np.stack(separation_values)
    swing_height_arr = np.stack(swing_height_values)
    advance_arr = np.stack(touchdown_advance_values)
    touchdown_air_arr = np.stack(touchdown_air_time_values)
    valid_advance = advance_arr[np.isfinite(advance_arr)]
    valid_air = touchdown_air_arr[np.isfinite(touchdown_air_arr)]
    initial_forward = np.stack((np.cos(start_yaw), np.sin(start_yaw)), axis=1)
    progress = np.sum((latest_pos[:, :2] - start_pos[:, :2]) * initial_forward, axis=1)
    signed_yaw_drift = np.arctan2(
        np.sin(latest_yaw - start_yaw),
        np.cos(latest_yaw - start_yaw),
    )
    support_duty = contact_steps / np.maximum(active_steps[:, None], 1.0)
    alternation = alternating_touchdowns / np.maximum(total_touchdowns - 1, 1)
    hip_range = _nanpercentile_range(dof_arr, [2, 6])
    knee_range = _nanpercentile_range(dof_arr, [3, 7])
    horizon_s = num_steps * float(env.step_dt)

    return {
        "vx_command_mps": vx,
        "horizon_s": horizon_s,
        "survival_rate": float(np.mean(~failed)),
        "mean_vx_mps": float(np.nanmean(vx_arr)),
        "vx_mae_mps": float(np.nanmean(np.abs(vx_arr - vx))),
        "world_progress_mean_m": float(np.mean(progress)),
        "world_progress_ratio": float(np.mean(progress) / max(vx * horizon_s, 1.0e-6)),
        "yaw_rate_abs_p95_rad_s": float(np.nanpercentile(np.abs(wz_arr), 95)),
        "yaw_drift_abs_p95_deg": float(np.degrees(np.percentile(np.abs(signed_yaw_drift), 95))),
        "yaw_drift_signed_mean_deg": float(np.degrees(np.mean(signed_yaw_drift))),
        "base_height_median_m": float(np.nanmedian(height_arr)),
        "torso_pitch_abs_p95_deg": float(np.degrees(np.nanpercentile(np.abs(pitch_arr), 95))),
        "left_hip_pitch_motion_range_median_deg": float(np.degrees(np.nanmedian(hip_range[:, 0]))),
        "right_hip_pitch_motion_range_median_deg": float(np.degrees(np.nanmedian(hip_range[:, 1]))),
        "min_side_hip_pitch_motion_range_median_deg": float(
            np.degrees(np.nanmedian(np.min(hip_range, axis=1)))
        ),
        "left_knee_motion_range_median_deg": float(np.degrees(np.nanmedian(knee_range[:, 0]))),
        "right_knee_motion_range_median_deg": float(np.degrees(np.nanmedian(knee_range[:, 1]))),
        "swing_foot_height_p95_m": float(np.nanpercentile(swing_height_arr, 95)),
        "foot_lateral_separation_median_m": float(np.nanmedian(separation_arr)),
        "forward_touchdown_fraction": float(np.mean(valid_advance > 0.0)),
        "forward_touchdown_median_m": float(np.median(valid_advance)),
        "touchdown_air_time_median_s": float(np.median(valid_air)),
        "both_air_fraction": float(np.sum(both_air_steps) / max(np.sum(active_steps), 1.0)),
        "double_stance_fraction": float(
            np.sum(double_stance_steps) / max(np.sum(active_steps), 1.0)
        ),
        "left_support_fraction": float(np.mean(support_duty[:, 0])),
        "right_support_fraction": float(np.mean(support_duty[:, 1])),
        "support_duty_imbalance": float(np.mean(np.abs(support_duty[:, 0] - support_duty[:, 1]))),
        "alternating_touchdown_ratio": float(np.mean(alternation)),
        "full_gait_cycles_median": float(np.median(np.min(touchdown_counts, axis=1))),
        "terminated_envs": int(np.sum(failed)),
    }


def main() -> None:
    args = _parse_args()
    speeds = args.vx or [0.2, 0.25, 0.3]
    results: list[dict[str, float | int]] = []
    checkpoint_path: Path | None = None
    for speed in speeds:
        cfg = _compose_eval_cfg(args, speed)
        session, checkpoint_path = _make_session(cfg)
        try:
            results.append(_evaluate(session, vx=speed, num_steps=args.num_steps))
        finally:
            session.env.close()

    payload = {
        "task": args.task,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
