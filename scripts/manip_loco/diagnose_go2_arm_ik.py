from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.common.rotation import np_matrix_from_quat
from unilab.envs.locomotion.go2_arm.manip_loco import RewardConfig


def _parse_vec3(raw: list[float], *, name: str) -> list[float]:
    if len(raw) != 3:
        raise ValueError(f"{name} must contain exactly 3 floats, got {raw}")
    return [float(v) for v in raw]


def _reward_cfg() -> RewardConfig:
    return RewardConfig(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "object_distance": 2.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.3,
        object_sigma=0.1,
    )


def _make_env(args: argparse.Namespace):
    env_cfg_override: dict[str, Any] = {
        "reward_config": _reward_cfg(),
        "arm_stage": {
            "freeze_arm_joints": False,
            "disable_ee_goal_trajectory": True,
            "fixed_ee_goal_cart": args.fixed_goal,
        },
        "ik": {
            "damping": args.damping,
            "gain": args.gain,
            "dq_clip": args.dq_clip,
            "use_orientation": False,
        },
    }
    if args.disable_gain_randomization:
        env_cfg_override["domain_rand"] = {
            "randomize_kp": False,
            "randomize_kd": False,
        }

    ensure_registries()
    return registry.make(
        "Go2ArmManipLoco",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override=env_cfg_override,
    )


def _state_qpos_qvel(backend: Any) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(backend.get_physics_state()[0], dtype=np.float64)
    qpos = state[backend._idx_qpos : backend._idx_qpos + backend.nq].copy()
    qvel = state[backend._idx_qvel : backend._idx_qvel + backend.nv].copy()
    return qpos, qvel


def _restore_state(env: Any, qpos: np.ndarray, qvel: np.ndarray) -> None:
    env._backend.set_state(
        np.array([0], dtype=np.int32),
        qpos[None, :],
        qvel[None, :],
    )


def _force_go2_home(env: Any, home_qpos: np.ndarray) -> None:
    """Clamp floating base and leg joints to home while preserving arm state."""
    backend = env._backend
    qpos, qvel = _state_qpos_qvel(backend)
    first_arm_qpos = backend._root_qpos_dim + int(env.arm_dof_pos_indices[0])
    first_arm_qvel = int(env.arm_jacobian_dof_indices[0])
    qpos[:first_arm_qpos] = home_qpos[:first_arm_qpos]
    qvel[:first_arm_qvel] = 0.0
    _restore_state(env, qpos, qvel)


def _local_position_jacobian(env: Any) -> np.ndarray:
    jacp_w, _ = env._backend.get_site_jacobian_w(
        env._ee_site_id,
        env.arm_jacobian_dof_indices,
    )
    ref_rot_w = np_matrix_from_quat(
        env._backend.get_sensor_data(env._cfg.sensor.arm_ref_world_quat)
    )
    return np.matmul(np.swapaxes(ref_rot_w, 1, 2), jacp_w)[0]


def run_jacobian_fd_check(env: Any, eps: float) -> dict[str, Any]:
    backend = env._backend
    qpos0, qvel0 = _state_qpos_qvel(backend)
    ee0 = env.get_ee_local_pose()[0].copy()[0]
    jac = _local_position_jacobian(env)
    fd = np.zeros_like(jac)

    for col, qpos_idx_rel in enumerate(env.arm_dof_pos_indices):
        qpos = qpos0.copy()
        qpos[backend._root_qpos_dim + qpos_idx_rel] += eps
        _restore_state(env, qpos, qvel0)
        ee_pos = env.get_ee_local_pose()[0].copy()[0]
        fd[:, col] = (ee_pos - ee0) / eps

    _restore_state(env, qpos0, qvel0)
    diff = jac - fd
    return {
        "jac": jac,
        "fd": fd,
        "max_abs_err": float(np.max(np.abs(diff))),
        "col_abs_err": np.max(np.abs(diff), axis=0),
    }


def run_direction_check(env: Any, delta: list[float]) -> dict[str, Any]:
    ee_pos = env.get_ee_local_pose()[0].copy()
    desired_delta = np.asarray(delta, dtype=ee_pos.dtype)[None, :]
    goal = ee_pos + desired_delta
    dq = env.compute_arm_ik_delta(goal, ee_pos)
    predicted_delta = np.matmul(_local_position_jacobian(env), dq[0])
    target = desired_delta[0]
    denom = np.linalg.norm(predicted_delta) * np.linalg.norm(target)
    cosine = float(np.dot(predicted_delta, target) / max(denom, 1e-12))
    return {
        "ee_pos": ee_pos[0],
        "goal": goal[0],
        "dq": dq[0],
        "predicted_delta": predicted_delta,
        "target_delta": target,
        "cosine": cosine,
    }


def run_closed_loop_check(env: Any, steps: int, *, go2_home_mode: str) -> dict[str, Any]:
    zero_actions = np.zeros((1, 18), dtype=np.float32)
    errors: list[float] = []
    dq_norms: list[float] = []
    arm_action_norms: list[float] = []
    rows: list[dict[str, Any]] = []
    if go2_home_mode not in {"none", "once", "reset-each-step"}:
        raise ValueError(f"Unsupported go2_home_mode: {go2_home_mode}")

    home_qpos = env._backend.get_keyframe_qpos("home") if go2_home_mode != "none" else None
    if go2_home_mode in {"once", "reset-each-step"}:
        assert home_qpos is not None
        _force_go2_home(env, home_qpos)

    for step in range(steps):
        if go2_home_mode == "reset-each-step":
            assert home_qpos is not None
            _force_go2_home(env, home_qpos)
        ee_pos = env.get_ee_local_pose()[0].copy()
        goal = env.curr_ee_goal_cart.copy()
        dq = env.compute_arm_ik_delta(goal, ee_pos)
        arm_qpos = env.get_arm_dof_pos().copy()
        ctrl = env.apply_action(zero_actions, env._state)
        arm_ctrl = ctrl[:, 12:18].copy()
        err = float(np.linalg.norm(goal - ee_pos, axis=1)[0])
        errors.append(err)
        dq_norms.append(float(np.linalg.norm(dq[0])))
        arm_action_norms.append(float(np.linalg.norm(arm_ctrl[0] - arm_qpos[0])))

        if step < 5 or step % 10 == 0 or step == steps - 1:
            rows.append(
                {
                    "step": step,
                    "err": err,
                    "ee": ee_pos[0].copy(),
                    "goal": goal[0].copy(),
                    "dq": dq[0].copy(),
                    "arm_qpos": arm_qpos[0].copy(),
                    "arm_ctrl": arm_ctrl[0].copy(),
                }
            )

        state = env.step(zero_actions)
        if go2_home_mode == "reset-each-step":
            assert home_qpos is not None
            _force_go2_home(env, home_qpos)
        if rows and rows[-1]["step"] == step:
            rows[-1]["terminated"] = bool(state.terminated[0])
            rows[-1]["truncated"] = bool(state.truncated[0])
            rows[-1]["gravity_z"] = float(env._backend.get_sensor_data("upvector")[0, 2])

    return {
        "go2_home_mode": go2_home_mode,
        "errors": np.asarray(errors, dtype=np.float64),
        "dq_norms": np.asarray(dq_norms, dtype=np.float64),
        "arm_action_norms": np.asarray(arm_action_norms, dtype=np.float64),
        "rows": rows,
    }


def _fmt_vec(v: np.ndarray) -> str:
    return np.array2string(np.asarray(v), precision=5, suppress_small=False)


def _print_jacobian_report(result: dict[str, Any], atol: float) -> bool:
    ok = result["max_abs_err"] <= atol
    print("\n[1] Jacobian finite-difference check")
    print(
        f"    max_abs_err: {result['max_abs_err']:.6g}  "
        f"threshold: {atol:.6g}  status: {'PASS' if ok else 'FAIL'}"
    )
    print(f"    col_abs_err: {_fmt_vec(result['col_abs_err'])}")
    print("    analytic J:")
    print(_fmt_vec(result["jac"]))
    print("    finite-diff J:")
    print(_fmt_vec(result["fd"]))
    return ok


def _print_direction_report(result: dict[str, Any], min_cosine: float) -> bool:
    ok = result["cosine"] >= min_cosine
    print("\n[2] One-step IK direction check")
    print(f"    ee:              {_fmt_vec(result['ee_pos'])}")
    print(f"    goal:            {_fmt_vec(result['goal'])}")
    print(f"    target_delta:    {_fmt_vec(result['target_delta'])}")
    print(f"    dq:              {_fmt_vec(result['dq'])}")
    print(f"    J @ dq:          {_fmt_vec(result['predicted_delta'])}")
    print(
        f"    cosine: {result['cosine']:.6f}  "
        f"threshold: {min_cosine:.6f}  status: {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _print_closed_loop_report(result: dict[str, Any]) -> None:
    errors = result["errors"]
    mode = result["go2_home_mode"]
    print(f"\n[3] Fixed-goal zero-action closed-loop check ({mode})")
    print(
        "    error summary: "
        f"initial={errors[0]:.6f}, min={np.min(errors):.6f}, "
        f"final={errors[-1]:.6f}, argmin={int(np.argmin(errors))}"
    )
    print(
        "    norm summary: "
        f"dq_mean={np.mean(result['dq_norms']):.6f}, "
        f"ctrl_minus_qpos_mean={np.mean(result['arm_action_norms']):.6f}"
    )
    print("    sampled rows:")
    for row in result["rows"]:
        print(
            f"      step={row['step']:>4} err={row['err']:.6f} "
            f"ee={_fmt_vec(row['ee'])} goal={_fmt_vec(row['goal'])} "
            f"done={row.get('terminated', False) or row.get('truncated', False)} "
            f"gravity_z={row.get('gravity_z', float('nan')):.3f}"
        )
        print(f"           dq={_fmt_vec(row['dq'])}")
        print(f"           arm_qpos={_fmt_vec(row['arm_qpos'])}")
        print(f"           arm_ctrl={_fmt_vec(row['arm_ctrl'])}")

    if errors[-1] > errors[0]:
        print(
            "    note: final error is larger than initial error. "
            "This points to closed-loop tuning, force limits, or action/IK interaction rather than Jacobian sign alone."
        )
    if mode == "reset-each-step":
        print(
            "    note: reset-each-step calls backend.set_state()/BatchEnvPool.reset inside the "
            "control loop; use go2-home-mode=once to test IK without that reset disturbance."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Go2ArmManipLoco end-effector IK without a trained policy."
    )
    parser.add_argument("--fixed-goal", type=float, nargs="+", default=[0.30, 0.0, 0.25])
    parser.add_argument("--direction-delta", type=float, nargs="+", default=[0.02, 0.0, 0.0])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--dq-clip", type=float, default=0.2)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--jacobian-atol", type=float, default=2e-3)
    parser.add_argument("--min-direction-cosine", type=float, default=0.9)
    parser.add_argument("--closed-loop-only", action="store_true")
    parser.add_argument(
        "--go2-home-mode",
        choices=["none", "once", "reset-each-step"],
        default="once",
        help=(
            "How to put Go2 at the home pose for closed-loop IK. "
            "'once' is the clean IK diagnostic; 'reset-each-step' is a reset-disturbance check."
        ),
    )
    parser.add_argument(
        "--force-go2-home",
        action="store_true",
        help=(
            "Backward-compatible alias for --go2-home-mode reset-each-step. "
            "This intentionally exercises the reset disturbance path."
        ),
    )
    parser.add_argument(
        "--disable-gain-randomization",
        action="store_true",
        help="Disable reset-time Kp/Kd randomization to isolate IK/controller behavior.",
    )
    args = parser.parse_args()

    args.fixed_goal = _parse_vec3(args.fixed_goal, name="--fixed-goal")
    args.direction_delta = _parse_vec3(args.direction_delta, name="--direction-delta")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.fd_eps <= 0.0:
        raise ValueError("--fd-eps must be positive")
    if args.force_go2_home:
        args.go2_home_mode = "reset-each-step"

    np.set_printoptions(precision=5, suppress=False, linewidth=180)
    env = _make_env(args)
    env.init_state()

    print("Go2ArmManipLoco IK diagnostic")
    print(f"  fixed_goal: {args.fixed_goal}")
    print(f"  ik: damping={args.damping}, gain={args.gain}, dq_clip={args.dq_clip}")
    print(f"  go2_home_mode: {args.go2_home_mode}")
    print(f"  disable_gain_randomization: {args.disable_gain_randomization}")
    print(f"  arm_dof_pos_indices: {env.arm_dof_pos_indices.tolist()}")
    print(f"  arm_jacobian_dof_indices: {env.arm_jacobian_dof_indices.tolist()}")
    print(f"  default_arm_qpos: {_fmt_vec(env.default_angles[12:18])}")

    checks_ok = True
    if not args.closed_loop_only:
        checks_ok &= _print_jacobian_report(
            run_jacobian_fd_check(env, args.fd_eps),
            args.jacobian_atol,
        )
        checks_ok &= _print_direction_report(
            run_direction_check(env, args.direction_delta),
            args.min_direction_cosine,
        )

    _print_closed_loop_report(
        run_closed_loop_check(env, args.steps, go2_home_mode=args.go2_home_mode)
    )
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
