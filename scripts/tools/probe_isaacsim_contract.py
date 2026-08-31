#!/usr/bin/env python3
"""Bounded headless probe for the IsaacSim/IsaacLab backend contract.

This developer tool runs inside the external ``UNILAB_ISAACSIM_HOME`` Python
3.11 runtime.  It exercises only cold-path materialization and the native
IsaacLab articulation tensor API; production backend behavior belongs under
``src/unilab/base/backend/isaacsim``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _as_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> int:
    args = _parse_args()
    if args.num_envs < 2:
        raise ValueError("--num-envs must be at least 2 to probe masked writes")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    model_file = args.model_file.expanduser().resolve()
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")

    # IsaacSim requires the Kit app before importing simulator modules.
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher({"headless": True, "device": args.device}).app

    import torch

    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.utils.extensions import enable_extension
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    result: dict[str, Any] = {
        "model_file": str(model_file),
        "num_envs": args.num_envs,
        "device": args.device,
    }

    try:
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 150.0, device=args.device)
        sim = sim_utils.SimulationContext(sim_cfg)

        enable_extension("isaacsim.asset.importer.mjcf")
        converter = MjcfConverter(
            MjcfConverterCfg(
                asset_path=str(model_file),
                fix_base=False,
                import_sites=True,
                make_instanceable=True,
            )
        )
        result["converted_usd"] = converter.usd_path

        origins = []
        for env_id in range(args.num_envs):
            origin = (float(env_id) * 2.0, 0.0, 0.0)
            origins.append(origin)
            prim_utils.create_prim(f"/World/envs/env_{env_id}", "Xform", translation=origin)

        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            articulation_root_prim_path="/pelvis/pelvis",
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=40.0,
                    damping=2.0,
                    effort_limit_sim=200.0,
                )
            },
        )
        robot = Articulation(robot_cfg)
        sim.reset()
        robot.update(sim_cfg.dt)

        result.update(
            {
                "joint_names": list(robot.joint_names),
                "body_names": list(robot.body_names),
                "num_joints": robot.num_joints,
                "num_bodies": robot.num_bodies,
                "root_quat_order": "wxyz",
                "root_ang_vel_frame": "world",
                "shapes": {
                    "root_pos_w": list(robot.data.root_pos_w.shape),
                    "root_quat_w": list(robot.data.root_quat_w.shape),
                    "root_lin_vel_w": list(robot.data.root_lin_vel_w.shape),
                    "root_ang_vel_w": list(robot.data.root_ang_vel_w.shape),
                    "joint_pos": list(robot.data.joint_pos.shape),
                    "joint_vel": list(robot.data.joint_vel.shape),
                    "body_pos_w": list(robot.data.body_pos_w.shape),
                    "body_quat_w": list(robot.data.body_quat_w.shape),
                    "body_lin_vel_w": list(robot.data.body_lin_vel_w.shape),
                    "body_ang_vel_w": list(robot.data.body_ang_vel_w.shape),
                },
            }
        )

        env_ids = torch.tensor([1], dtype=torch.long, device=robot.device)
        before_root = robot.data.root_pos_w.clone()
        before_joint = robot.data.joint_pos.clone()

        root_pose = robot.data.root_state_w[env_ids, :7].clone()
        root_pose[:, 2] += 0.05
        root_velocity = torch.zeros((1, 6), device=robot.device)
        root_velocity[:, 3:] = torch.tensor([[0.1, 0.2, 0.3]], device=robot.device)
        joint_pos = robot.data.joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        if robot.num_joints:
            joint_pos[:, 0] += 0.05

        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot.reset(env_ids)
        robot.update(sim_cfg.dt)

        masked_root_changed = float(robot.data.root_pos_w[1, 2] - before_root[1, 2])
        untouched_root_delta = float(
            torch.max(torch.abs(robot.data.root_pos_w[0] - before_root[0])).item()
        )
        untouched_joint_delta = float(
            torch.max(torch.abs(robot.data.joint_pos[0] - before_joint[0])).item()
        )
        result["masked_write"] = {
            "selected_root_z_delta": masked_root_changed,
            "untouched_root_max_delta": untouched_root_delta,
            "untouched_joint_max_delta": untouched_joint_delta,
            "selected_root_ang_vel_w": _as_list(robot.data.root_ang_vel_w[1]),
        }

        targets = robot.data.joint_pos.clone()
        if robot.num_joints:
            targets[:, 0] += 0.1
        before_pd = robot.data.joint_pos.clone()
        for _ in range(args.steps):
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_cfg.dt)
        result["pd_target"] = {
            "steps": args.steps,
            "first_joint_delta": _as_list(robot.data.joint_pos[:, 0] - before_pd[:, 0])
            if robot.num_joints
            else [],
        }
        result["sample"] = {
            "root_quat_w": _as_list(robot.data.root_quat_w[0]),
            "body_quat_w": _as_list(robot.data.body_quat_w[0, 0]),
        }

        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException:
        # Kit can suppress buffered Python diagnostics during shutdown.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
