"""Headless rollout eval for the OpenArm demo pick task.

Loads a trained PPO checkpoint and rolls the deterministic policy out across many
parallel envs (no rendering), then reports objective pick metrics: fraction of
envs that ever reach ``pick_success``, fraction holding success at the final
step, fraction that dropped the cube, and the mean final cube height. Use this
instead of eyeballing the single on-camera env in the play video.

Example:
    HIP_VISIBLE_DEVICES=0 uv run scripts/eval_openarm_success.py \
        task=openarm_demo_pick/mujoco_lift3d_contgrip \
        algo.load_run=logs/rsl_rl_ppo/OpenArmDemoPick/<run>_mujoco \
        training.eval_envs=512 training.play_steps=200
"""

import sys
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
for _p in (SRC_DIR, ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rsl_rl.runners import OnPolicyRunner
from train_rsl_rl import (  # type: ignore[import-not-found]
    _algo_config_dict,
    _resolve_ppo_wrapper_cls,
    apply_ppo_runtime_flags,
    build_ppo_play_env_cfg_override,
)

from unilab.training import (
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    parse_checkpoint_path,
)
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from unilab.training.sim2sim import policy_load_dim_guard
from unilab.utils.device import get_default_device


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = get_default_device()

    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or not load_path.exists():
        raise SystemExit(f"No checkpoint resolved (algo.load_run={cfg.algo.load_run!r}).")

    num_envs = int(OmegaConf.select(cfg, "training.eval_envs", default=512))
    num_steps = int(OmegaConf.select(cfg, "training.play_steps", default=200))

    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=build_ppo_play_env_cfg_override(cfg),
    )
    wrapper_cls = _resolve_ppo_wrapper_cls(_algo_config_dict(cfg))
    wrapped_env = wrapper_cls(env, device=device)

    train_cfg = normalize_ppo_train_cfg(_algo_config_dict(cfg))
    apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    train_cfg.setdefault("runner", {})["logger"] = "none"

    runner = cast(
        Any, OnPolicyRunner(cast(Any, wrapped_env), train_cfg, log_dir=None, device=device)
    )
    with policy_load_dim_guard(
        env_obs_dim=getattr(wrapped_env, "num_obs", None),
        env_action_dim=getattr(wrapped_env, "num_actions", None),
        algo_name="ppo",
    ):
        runner.load(str(load_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    ever_success = np.zeros(num_envs, dtype=bool)
    ever_fallen = np.zeros(num_envs, dtype=bool)

    # --- Staged-grasp adherence tracking (does the policy go above + open, then
    # close over the cube, rather than swiping it up?) ---
    ALIGN_XY = 0.05  # xy considered "aligned over the cube"
    ABOVE_LO, ABOVE_HI = 0.03, 0.18  # tcp height band above the cube for pre-grasp
    OPEN_THRESH = 0.25  # closure below this counts as "gripper open"
    CLOSE_THRESH = 0.5  # closure above this counts as "gripper closed"
    first_close_step = np.full(num_envs, -1, dtype=np.int64)
    first_close_xy = np.full(num_envs, np.nan, dtype=np.float32)
    was_above_open = np.zeros(num_envs, dtype=bool)  # ever pre-grasp-posed before closing

    def _info_flag(name: str) -> np.ndarray:
        val = env.state.info.get(name)
        return np.asarray(val, dtype=bool).reshape(-1)

    def _stage_signals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ee = np.asarray(env._grasp_point_w(), dtype=np.float32)
        cube = np.asarray(
            env._backend.get_body_pos_w(env._cube_body_ids)[:, 0, :], dtype=np.float32
        )
        xy = np.linalg.norm(ee[:, :2] - cube[:, :2], axis=-1)
        above = ee[:, 2] - cube[:, 2]
        closure = np.asarray(env._finger_closure(env.state.info), dtype=np.float32).reshape(-1)
        return xy, above, closure

    with torch.inference_mode():
        obs = wrapped_env.reset()[0]
        for t in range(num_steps):
            obs = wrapped_env.step(policy(obs))[0]
            ever_success |= _info_flag("pick_success")
            ever_fallen |= _info_flag("pick_fallen")

            xy, above, closure = _stage_signals()
            open_now = closure < OPEN_THRESH
            not_closed_yet = first_close_step < 0
            pre_grasp_pose = (xy < ALIGN_XY) & (above > ABOVE_LO) & (above < ABOVE_HI) & open_now
            was_above_open |= pre_grasp_pose & not_closed_yet
            just_closed = (closure > CLOSE_THRESH) & not_closed_yet
            first_close_step[just_closed] = t
            first_close_xy[just_closed] = xy[just_closed]

    final_success = _info_flag("pick_success")
    cube_w = np.asarray(env._backend.get_body_pos_w(env._cube_body_ids)[:, 0, :], dtype=np.float32)
    cube_z = cube_w[:, 2]
    goal = np.asarray(env.state.info["goal_pos"], dtype=np.float32).reshape(num_envs, 3)
    dist3d = np.linalg.norm(cube_w - goal, axis=-1)

    table_z = float(env._reward_cfg.table_z)
    margin = float(env._reward_cfg.lift_margin)
    held_clear = cube_z > (table_z + margin)
    final_closure = np.asarray(env._finger_closure(env.state.info), dtype=np.float32).reshape(-1)

    print("=" * 60)
    print(f"checkpoint     : {load_path}")
    print(f"envs / steps   : {num_envs} / {num_steps}")
    print(
        f"ever success   : {ever_success.mean() * 100:.1f}%   (3D dist<{env._reward_cfg.success_dist} & lifted)"
    )
    print(f"final success  : {final_success.mean() * 100:.1f}%  (holding at last step)")
    print(f"ever dropped   : {ever_fallen.mean() * 100:.1f}%")
    print(f"held clear     : {held_clear.mean() * 100:.1f}%   (final z > table+{margin})")
    held_closure = final_closure[held_clear]
    closure_str = f"{held_closure.mean():.3f}" if held_closure.size else "n/a"
    print(f"grasp firmness : mean finger closure on held envs = {closure_str}  (0=open..1=closed)")
    print(
        f"final cube z   : mean={cube_z.mean():.3f}  min={cube_z.min():.3f}  max={cube_z.max():.3f}"
    )
    print(
        f"final 3D dist  : mean={dist3d.mean():.3f}  min={dist3d.min():.3f}  (goal z={goal[0, 2]:.3f})"
    )
    print(f"within 0.10 m  : {(dist3d < 0.10).mean() * 100:.1f}%")
    print(f"within 0.05 m  : {(dist3d < 0.05).mean() * 100:.1f}%")
    print("-" * 60)
    print("staged-grasp adherence:")
    ever_closed = first_close_step >= 0
    closed_over_cube = ever_closed & (first_close_xy < ALIGN_XY)
    print(
        f"  pre-grasp pose : {was_above_open.mean() * 100:.1f}%   (above+open before first close)"
    )
    print(f"  ever closed    : {ever_closed.mean() * 100:.1f}%")
    if ever_closed.any():
        print(
            f"  closed over cube: {closed_over_cube.mean() * 100:.1f}%   "
            f"(first-close xy<{ALIGN_XY}; mean first-close xy={np.nanmean(first_close_xy):.3f})"
        )
        print(f"  mean first-close step: {first_close_step[ever_closed].mean():.1f}")
    print("=" * 60)

    if num_envs <= 32:
        order = np.argsort(-cube_z)
        print("per-env (sorted by final z):")
        print("  env  final_z  held  dropped")
        for i in order:
            print(
                f"  {int(i):>3}  {cube_z[i]:7.3f}  {str(bool(held_clear[i])):>5}  "
                f"{str(bool(ever_fallen[i])):>5}"
            )
        best = int(order[0])
        for i in order:
            if held_clear[i] and not ever_fallen[i]:
                best = int(i)
                break
        print(f"best demo env (held, no drop, highest z): {best}")
        print("=" * 60)


if __name__ == "__main__":
    main()
