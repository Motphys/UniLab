from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.envs.locomotion.go2_arm.base import build_go2_arm_position_gains
from unilab.envs.locomotion.go2_arm.manip_loco import Go2ArmManipLocoCfg

TARGET_BODY = "ik_mocap_target"


def _parse_vec3(raw: list[float], *, name: str) -> np.ndarray:
    if len(raw) != 3:
        raise ValueError(f"{name} must contain exactly 3 floats, got {raw}")
    return np.asarray(raw, dtype=np.float64)


def _write_scene_with_mocap(source: Path) -> Path:
    raw = source.read_text()
    marker = "  </worldbody>"
    if marker not in raw:
        raise ValueError(f"Could not find scene worldbody closing tag in {source}")
    mocap_xml = f"""
    <body name="{TARGET_BODY}" mocap="true" pos="0.3 0 0.3">
      <geom name="{TARGET_BODY}_geom" type="sphere" size="0.035"
            rgba="1 0.78 0.05 0.65" contype="0" conaffinity="0"/>
      <site name="{TARGET_BODY}_site" type="sphere" size="0.012" rgba="1 0.5 0 1"/>
    </body>
"""
    xml = raw.replace(marker, mocap_xml + marker, 1)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_go2_arm_ik_only_",
        suffix=".xml",
        dir=str(source.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.write_text(xml)
    return tmp_path


def _load_model(args: argparse.Namespace) -> tuple[mujoco.MjModel, Go2ArmManipLocoCfg]:
    cfg = Go2ArmManipLocoCfg()
    source = Path(cfg.model_file)
    tmp_xml = _write_scene_with_mocap(source)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    finally:
        tmp_xml.unlink(missing_ok=True)

    model.opt.timestep = float(args.sim_dt)
    gains = build_go2_arm_position_gains(cfg.control_config)
    kp = np.asarray(gains["kp"], dtype=np.float64)
    kd = np.asarray(gains["kd"], dtype=np.float64)
    if kp.shape != (model.nu,) or kd.shape != (model.nu,):
        raise ValueError(f"Expected gain shape ({model.nu},), got kp={kp.shape}, kd={kd.shape}")
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kd
    return model, cfg


def _named_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"{name!r} not found in MuJoCo model")
    return int(obj_id)


def _home_state(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    key_id = _named_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model.key_qpos[key_id].copy(), model.key_ctrl[key_id].copy()


def _joint_qpos_indices(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    indices = []
    for name in names:
        jid = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        indices.append(int(model.jnt_qposadr[jid]))
    return np.asarray(indices, dtype=np.int32)


def _joint_qvel_indices(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    indices = []
    for name in names:
        jid = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        indices.append(int(model.jnt_dofadr[jid]))
    return np.asarray(indices, dtype=np.int32)


def _arm_actuator_ids(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> np.ndarray:
    ids = np.arange(model.nu - len(joint_names), model.nu, dtype=np.int32)
    joint_ids = [_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    mapped = [int(model.actuator_trnid[aid, 0]) for aid in ids]
    if mapped != joint_ids:
        print(
            "warning: last actuator joint mapping does not match arm joint names; "
            f"using last {len(joint_names)} actuators anyway. mapped={mapped}, expected={joint_ids}"
        )
    return ids


def _reset_home(model: mujoco.MjModel, data: mujoco.MjData, home_qpos: np.ndarray) -> None:
    data.qpos[:] = home_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _hold_go2_default(
    data: mujoco.MjData,
    *,
    home_qpos: np.ndarray,
    home_ctrl: np.ndarray,
) -> None:
    data.qpos[:19] = home_qpos[:19]
    data.qvel[:18] = 0.0
    data.ctrl[:12] = home_ctrl[:12]


def _site_pos_rot(data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()


def _local_to_world(data: mujoco.MjData, site_id: int, local: np.ndarray) -> np.ndarray:
    pos, rot = _site_pos_rot(data, site_id)
    return pos + rot @ np.asarray(local, dtype=np.float64)


def _world_to_local(data: mujoco.MjData, site_id: int, world: np.ndarray) -> np.ndarray:
    pos, rot = _site_pos_rot(data, site_id)
    return rot.T @ (np.asarray(world, dtype=np.float64) - pos)


def _fullphysics_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.zeros(
        (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS),),
        dtype=np.float64,
    )
    mujoco.mj_getState(model, data, state, int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
    return state


def _direct_site_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_site_id: int,
    arm_qvel_ids: np.ndarray,
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    return jacp[:, arm_qvel_ids]


def _batch_site_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    batch_pool: Any,
    ee_site_id: int,
    arm_qvel_ids: np.ndarray,
) -> np.ndarray:
    jp, _ = batch_pool.compute_site_jacobians(
        _fullphysics_state(model, data)[None, :],
        int(ee_site_id),
        jacp=True,
        jacr=True,
    )
    if jp.ndim == 4:
        jp = jp[:, 0]
    jac = jp[0]
    if jac.shape == (3, model.nv):
        return jac[:, arm_qvel_ids]
    if jac.shape == (model.nv, 3):
        return jac[arm_qvel_ids, :].T
    raise ValueError(f"Unexpected batch Jacobian shape: {jac.shape}")


def _site_jacobian_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    jacobian_source: str,
    batch_pool: Any,
    ee_site_id: int,
    arm_qvel_ids: np.ndarray,
) -> np.ndarray:
    if jacobian_source == "direct":
        return _direct_site_jacobian(
            model,
            data,
            ee_site_id=ee_site_id,
            arm_qvel_ids=arm_qvel_ids,
        )
    if jacobian_source == "batch":
        return _batch_site_jacobian(
            model,
            data,
            batch_pool=batch_pool,
            ee_site_id=ee_site_id,
            arm_qvel_ids=arm_qvel_ids,
        )
    raise ValueError(f"Unsupported jacobian source: {jacobian_source}")


def _compute_ik_delta(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    batch_pool: Any,
    jacobian_source: str,
    compare_jacobians: bool,
    ee_site_id: int,
    ref_site_id: int,
    arm_qvel_ids: np.ndarray,
    goal_local_pos: np.ndarray,
    curr_local_pos: np.ndarray,
    damping: float,
    dq_clip: float,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    jacp_world = _site_jacobian_world(
        model,
        data,
        jacobian_source=jacobian_source,
        batch_pool=batch_pool,
        ee_site_id=ee_site_id,
        arm_qvel_ids=arm_qvel_ids,
    )
    compare_abs_max = None
    if compare_jacobians:
        jacp_direct = _direct_site_jacobian(
            model,
            data,
            ee_site_id=ee_site_id,
            arm_qvel_ids=arm_qvel_ids,
        )
        jacp_batch = _batch_site_jacobian(
            model,
            data,
            batch_pool=batch_pool,
            ee_site_id=ee_site_id,
            arm_qvel_ids=arm_qvel_ids,
        )
        compare_abs_max = float(np.max(np.abs(jacp_direct - jacp_batch)))

    _, ref_rot = _site_pos_rot(data, ref_site_id)
    jac_local = ref_rot.T @ jacp_world
    pos_err = goal_local_pos - curr_local_pos
    lhs = jac_local @ jac_local.T + np.eye(3, dtype=np.float64) * (damping**2)
    dq = jac_local.T @ np.linalg.solve(lhs, pos_err)
    if dq_clip > 0.0:
        dq = np.clip(dq, -dq_clip, dq_clip)
    return dq, jac_local, compare_abs_max


def _step_controller(ctx: dict[str, Any]) -> dict[str, Any]:
    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData = ctx["data"]
    goal_local = _world_to_local(
        data,
        ctx["armbase_site_id"],
        data.mocap_pos[ctx["target_mocap_id"]],
    )
    ee_local = data.sensor("endpoint_pos").data.copy()
    dq, jac_local, jacobian_compare_abs_max = _compute_ik_delta(
        model,
        data,
        batch_pool=ctx["batch_pool"],
        jacobian_source=ctx["args"].jacobian_source,
        compare_jacobians=ctx["args"].compare_jacobians,
        ee_site_id=ctx["ee_site_id"],
        ref_site_id=ctx["armbase_site_id"],
        arm_qvel_ids=ctx["arm_qvel_ids"],
        goal_local_pos=goal_local,
        curr_local_pos=ee_local,
        damping=ctx["args"].damping,
        dq_clip=ctx["args"].dq_clip,
    )
    arm_qpos = data.qpos[ctx["arm_qpos_ids"]].copy()
    arm_target = arm_qpos + ctx["args"].gain * dq
    arm_ctrl_ids = ctx["arm_actuator_ids"]
    ctrl_low = model.actuator_ctrlrange[arm_ctrl_ids, 0]
    ctrl_high = model.actuator_ctrlrange[arm_ctrl_ids, 1]
    data.ctrl[arm_ctrl_ids] = np.clip(arm_target, ctrl_low, ctrl_high)
    return {
        "goal_local": goal_local,
        "ee_local": ee_local,
        "dq": dq,
        "jac_local": jac_local,
        "jacobian_compare_abs_max": jacobian_compare_abs_max,
        "arm_qpos": arm_qpos,
        "arm_ctrl": data.ctrl[arm_ctrl_ids].copy(),
        "err": float(np.linalg.norm(goal_local - ee_local)),
    }


def _init_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    armbase_site_id: int,
    initial_goal_local: np.ndarray,
) -> int:
    target_body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    mocap_id = int(model.body_mocapid[target_body_id])
    if mocap_id < 0:
        raise ValueError(f"{TARGET_BODY!r} is not a mocap body")
    data.mocap_pos[mocap_id] = _local_to_world(data, armbase_site_id, initial_goal_local)
    data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    return mocap_id


def _print_status(step: int, status: dict[str, Any]) -> None:
    compare = status.get("jacobian_compare_abs_max")
    compare_text = "" if compare is None else f" jac_direct_batch_maxdiff={compare:.3e}"
    print(
        f"step={step:>6} err={status['err']:.5f} "
        f"goal_local={np.array2string(status['goal_local'], precision=4)} "
        f"ee_local={np.array2string(status['ee_local'], precision=4)} "
        f"dq={np.array2string(status['dq'], precision=4)}"
        f"{compare_text}"
    )


def _draw_debug_geoms(viewer: Any, data: mujoco.MjData, ee_site_id: int) -> None:
    viewer.user_scn.ngeom = 1
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.018, 0.0, 0.0],
        pos=data.site_xpos[ee_site_id].copy(),
        mat=np.eye(3).reshape(-1),
        rgba=np.array([0.1, 0.35, 1.0, 0.85]),
    )


def _build_context(args: argparse.Namespace) -> dict[str, Any]:
    model, cfg = _load_model(args)
    data = mujoco.MjData(model)
    home_qpos, home_ctrl = _home_state(model)
    _reset_home(model, data, home_qpos)

    armbase_site_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "armbasepoint")
    ee_site_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.asset.ee_site_name)
    target_mocap_id = _init_target(
        model,
        data,
        armbase_site_id=armbase_site_id,
        initial_goal_local=args.initial_goal,
    )
    joint_names = tuple(cfg.asset.arm_joint_names)
    ctx = {
        "args": args,
        "model": model,
        "data": data,
        "home_qpos": home_qpos,
        "home_ctrl": home_ctrl,
        "armbase_site_id": armbase_site_id,
        "ee_site_id": ee_site_id,
        "target_mocap_id": target_mocap_id,
        "arm_qpos_ids": _joint_qpos_indices(model, joint_names),
        "arm_qvel_ids": _joint_qvel_indices(model, joint_names),
        "arm_actuator_ids": _arm_actuator_ids(model, joint_names),
        "batch_pool": None,
    }
    if args.jacobian_source == "batch" or args.compare_jacobians:
        from mujoco.batch_env import BatchEnvPool

        ctx["batch_pool"] = BatchEnvPool(model, nbatch=1, nthread=1)
    return ctx


def _close_context(ctx: dict[str, Any]) -> None:
    batch_pool = ctx.get("batch_pool")
    if batch_pool is not None:
        batch_pool.close()


def _run_one_sim_step(ctx: dict[str, Any], step: int) -> dict[str, Any]:
    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData = ctx["data"]
    _hold_go2_default(data, home_qpos=ctx["home_qpos"], home_ctrl=ctx["home_ctrl"])
    mujoco.mj_forward(model, data)
    if step % ctx["control_every"] == 0:
        ctx["last_status"] = _step_controller(ctx)
    mujoco.mj_step(model, data)
    _hold_go2_default(data, home_qpos=ctx["home_qpos"], home_ctrl=ctx["home_ctrl"])
    return ctx["last_status"]


def run_headless(ctx: dict[str, Any], steps: int) -> None:
    ctx["control_every"] = max(1, int(round(ctx["args"].ctrl_dt / ctx["args"].sim_dt)))
    ctx["last_status"] = _step_controller(ctx)
    for step in range(steps):
        status = _run_one_sim_step(ctx, step)
        if step < 5 or step % ctx["args"].print_every == 0 or step == steps - 1:
            _print_status(step, status)


def run_viewer(ctx: dict[str, Any]) -> None:
    import mujoco.viewer

    args = ctx["args"]
    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData = ctx["data"]
    ctx["control_every"] = max(1, int(round(args.ctrl_dt / args.sim_dt)))
    ctx["last_status"] = _step_controller(ctx)

    print("MuJoCo viewer controls:")
    print("  Drag the yellow mocap sphere to move the IK target.")
    print("  Blue sphere marks the current endpoint site.")
    print("  Go2 base and legs are forced back to the home keyframe each step.")

    step = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            status = _run_one_sim_step(ctx, step)
            _draw_debug_geoms(viewer, data, ctx["ee_site_id"])
            if step % args.print_every == 0:
                _print_status(step, status)
            viewer.sync()
            step += 1
            sleep_s = args.sim_dt - (time.perf_counter() - start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactively test Go2 arm Jacobian IK with UniLab XML and no policy."
    )
    parser.add_argument("--initial-goal", type=float, nargs="+", default=[0.30, 0.0, 0.25])
    parser.add_argument("--sim-dt", type=float, default=0.004)
    parser.add_argument("--ctrl-dt", type=float, default=0.02)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--dq-clip", type=float, default=0.2)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--jacobian-source",
        choices=["batch", "direct"],
        default="batch",
        help="Use UniLab training path BatchEnvPool.compute_site_jacobians or direct mj_jacSite.",
    )
    parser.add_argument(
        "--compare-jacobians",
        action="store_true",
        help="Print max abs difference between direct mj_jacSite and batch Jacobian.",
    )
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="Run without launching the viewer for this many simulation steps.",
    )
    args = parser.parse_args()
    args.initial_goal = _parse_vec3(args.initial_goal, name="--initial-goal")
    if args.sim_dt <= 0.0 or args.ctrl_dt <= 0.0:
        raise ValueError("--sim-dt and --ctrl-dt must be positive")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")

    np.set_printoptions(precision=5, suppress=False, linewidth=160)
    ctx = _build_context(args)
    print("Go2 arm IK-only play")
    print(f"  initial_goal local(armbasepoint): {np.array2string(args.initial_goal, precision=4)}")
    print(f"  ik: damping={args.damping}, gain={args.gain}, dq_clip={args.dq_clip}")
    print(f"  sim_dt={args.sim_dt}, ctrl_dt={args.ctrl_dt}")
    print(f"  jacobian_source={args.jacobian_source}, compare_jacobians={args.compare_jacobians}")
    print(f"  arm_qpos_ids={ctx['arm_qpos_ids'].tolist()}")
    print(f"  arm_qvel_ids={ctx['arm_qvel_ids'].tolist()}")
    print(f"  arm_actuator_ids={ctx['arm_actuator_ids'].tolist()}")

    try:
        if args.headless_steps > 0:
            run_headless(ctx, args.headless_steps)
        else:
            run_viewer(ctx)
    finally:
        _close_context(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
