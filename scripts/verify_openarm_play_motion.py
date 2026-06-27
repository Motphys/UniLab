#!/usr/bin/env python3
"""Check that OpenArmDemoPick playback physics shows left-arm joint motion (env 0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (SRC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _left_arm_qpos_indices(model: Any) -> list[int]:
    import mujoco

    m: Any = model
    indices: list[int] = []
    for i in range(1, 8):
        jn = f"openarm_left_joint{i}"
        jid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn))
        if jid < 0:
            raise SystemExit(f"missing joint {jn!r}")
        adr = int(m.jnt_qposadr[jid])
        jtype = int(m.jnt_type[jid])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            raise SystemExit(f"unexpected free joint {jn}")
        width = 7 if jtype == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        for k in range(width):
            indices.append(adr + k)
    return indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--motion-threshold", type=float, default=0.012)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use policy mean only (default: stochastic sampling, easier to see joint motion).",
    )
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir not found: {run_dir}")

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(ROOT / "conf" / "ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=openarm_demo_pick/mujoco",
                f"algo.load_run={run_dir}",
                "algo.checkpoint=-1",
                "algo.num_envs=8",
                "training.no_play=true",
            ],
        )

    import importlib.util

    path = ROOT / "scripts" / "train_rsl_rl.py"
    spec = importlib.util.spec_from_file_location("train_rsl_rl_verify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load train_rsl_rl")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    from unilab.training import create_env, ensure_registries, parse_checkpoint_path
    from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
    from unilab.training.sim2sim import policy_load_dim_guard
    from unilab.utils.device import get_default_device

    ensure_registries()
    device = get_default_device()
    env_cfg_override = tr.build_ppo_play_env_cfg_override(cfg)
    env = create_env(cfg, num_envs=8, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)

    rl_cfg = tr._algo_config_dict(cfg)
    wrapper_cls = tr._resolve_ppo_wrapper_cls(rl_cfg)
    wrapped = wrapper_cls(env, device=device)
    train_cfg = normalize_ppo_train_cfg(rl_cfg)
    tr.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"

    from rsl_rl.runners import OnPolicyRunner

    ckpt, _ = parse_checkpoint_path(cfg, root_dir=ROOT)
    if ckpt is None or not ckpt.is_file():
        raise SystemExit(f"no checkpoint under {run_dir}")

    runner = cast(
        Any,
        OnPolicyRunner(cast(Any, wrapped), train_cfg, log_dir=None, device=device),
    )
    with policy_load_dim_guard(
        env_obs_dim=getattr(wrapped, "num_obs", None),
        env_action_dim=getattr(wrapped, "num_actions", None),
        algo_name="ppo",
    ):
        runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)

    model = env._backend.model
    qadr = np.asarray(_left_arm_qpos_indices(model), dtype=np.intp)
    idx_qpos = int(getattr(env._backend, "_idx_qpos", 1))

    obs_td, _ = wrapped.reset()
    prev = None
    max_delta = 0.0
    with torch.inference_mode():
        for _ in range(int(args.steps)):
            act = policy(obs_td, stochastic_output=not bool(args.deterministic))
            obs_td, _, _, _ = wrapped.step(act)
            snap = np.asarray(env.get_physics_state_snapshot(), dtype=np.float64)
            q = snap[0, idx_qpos + qadr]
            if prev is not None:
                max_delta = max(max_delta, float(np.max(np.abs(q - prev))))
            prev = q.copy()

    env.close()
    print(
        f"verify_openarm_play_motion: max_abs_left_arm_qpos_delta={max_delta:.6f} thr={args.motion_threshold}"
    )
    if max_delta < float(args.motion_threshold):
        raise SystemExit(2)
    print("OK: arm motion detected in physics rollout.")


if __name__ == "__main__":
    main()
