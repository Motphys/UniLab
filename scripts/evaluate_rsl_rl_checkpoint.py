from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rsl_rl.runners import OnPolicyRunner

from scripts.train_rsl_rl import _algo_config_dict, _resolve_ppo_wrapper_cls
from unilab.base.backend.xml import materialize_scene_visual_override
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.training.rsl_rl import normalize_ppo_train_cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint without rendering.")
    parser.add_argument("--task", required=True, help="Hydra task choice, for example g1_flip_tracking/mujoco.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. May be repeated.",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    overrides = [
        f"task={args.task}",
        "training.no_play=true",
        *args.override,
    ]
    with hydra.initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "ppo"), version_base="1.3"
    ):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    ensure_registries()
    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    ).build_task_env_cfg_override()

    env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=env_cfg_override)
    wrapper_cls = _resolve_ppo_wrapper_cls(_algo_config_dict(cfg))
    wrapped_env = wrapper_cls(env, device=args.device)
    try:
        train_cfg = normalize_ppo_train_cfg(_algo_config_dict(cfg))
        train_cfg.setdefault("runner", {})["logger"] = "none"
        runner = cast(
            Any,
            OnPolicyRunner(cast(Any, wrapped_env), train_cfg, log_dir=None, device=args.device),
        )
        runner.load(str(checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)

        obs = wrapped_env.get_observations()
        lengths = np.zeros(args.num_envs, dtype=np.int32)
        returns = np.zeros(args.num_envs, dtype=np.float64)
        completed_lengths: list[int] = []
        completed_returns: list[float] = []
        log_values: dict[str, list[float]] = {}
        terminated_episodes = 0
        truncated_episodes = 0

        with torch.inference_mode():
            for _step in range(args.max_steps):
                actions = policy(obs)
                obs, rew, dones, infos = wrapped_env.step(actions)
                rew_np = rew.detach().cpu().numpy()
                done_np = dones.detach().cpu().numpy().astype(bool)

                lengths += 1
                returns += rew_np

                log = infos.get("log") if isinstance(infos, dict) else None
                if isinstance(log, dict):
                    for key, value in log.items():
                        try:
                            log_values.setdefault(str(key), []).append(float(value))
                        except (TypeError, ValueError):
                            pass

                if not np.any(done_np):
                    continue

                done_idx = np.flatnonzero(done_np)
                completed_lengths.extend(int(lengths[i]) for i in done_idx)
                completed_returns.extend(float(returns[i]) for i in done_idx)

                time_outs = infos.get("time_outs") if isinstance(infos, dict) else None
                if time_outs is not None:
                    timeout_np = time_outs.detach().cpu().numpy().astype(bool)
                    truncated_episodes += int(timeout_np[done_idx].sum())
                    terminated_episodes += int((~timeout_np[done_idx]).sum())

                lengths[done_idx] = 0
                returns[done_idx] = 0.0
                if len(completed_lengths) >= args.episodes:
                    break
    finally:
        wrapped_env.close()

    return {
        "checkpoint": str(checkpoint),
        "task": args.task,
        "overrides": overrides,
        "num_envs": args.num_envs,
        "episodes": len(completed_lengths),
        "mean_episode_length": (
            float(statistics.mean(completed_lengths)) if completed_lengths else None
        ),
        "median_episode_length": (
            float(statistics.median(completed_lengths)) if completed_lengths else None
        ),
        "min_episode_length": int(min(completed_lengths)) if completed_lengths else None,
        "max_episode_length": int(max(completed_lengths)) if completed_lengths else None,
        "mean_return": float(statistics.mean(completed_returns)) if completed_returns else None,
        "terminated_episodes": terminated_episodes,
        "truncated_episodes": truncated_episodes,
        "reward_log_mean": {
            key: float(statistics.mean(values))
            for key, values in sorted(log_values.items())
            if values
        },
    }


def main() -> None:
    args = _parse_args()
    summary = evaluate(args)
    output = args.output or args.checkpoint.parent / f"eval_{args.checkpoint.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
