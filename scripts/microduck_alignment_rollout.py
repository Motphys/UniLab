"""Controlled statistical rollout for MicroDuck alignment comparison (issue #1453, child 1/5).

Builds ``microduck_velocity_flat`` on the mjwarp backend with a fixed seed,
holds the action at zero (the JointPositionActionCfg default-pose target, so
the protocol is deterministic and reproducible by the upstream repo), and
collects: per-reward-term step means, episode-length distribution, root z and
tilt at episode end, and NaN counts. Requires CUDA; without a GPU the script
emits a skipped JSON result instead of failing.

    uv run scripts/microduck_alignment_rollout.py
    uv run scripts/microduck_alignment_rollout.py --num-envs 64 --steps 200 --output out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "src" / "unilab" / "conf" / "ppo"


def _compose_mjwarp_owner() -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=microduck_velocity_flat/mjwarp"])


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_rollout(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    from unisim.backend.mjwarp.dependencies import load_mjwarp_dependencies

    from unilab.base import registry
    from unilab.base.config_adapter import BackendAdapter
    from unilab.tasks.locomotion.microduck.deploy_contract import MICRODUCK_NUM_ACTION

    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        return {
            "skipped": True,
            "reason": "microduck alignment rollout requires an active CUDA Warp device",
        }

    cfg = _compose_mjwarp_owner()
    registry.ensure_registries()
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    # Fixed seed and manual resets so terminal root z / tilt are captured
    # before the autoreset overwrites them.
    override["seed"] = seed
    override["auto_reset"] = False
    env = cast(
        Any,
        registry.make(
            "MicroduckVelocityFlat",
            sim_backend="mjwarp",
            num_envs=num_envs,
            env_cfg_override=override,
        ),
    )

    reward_sums: dict[str, float] = {}
    reward_steps = 0
    episode_lengths: list[float] = []
    terminal_root_z: list[float] = []
    terminal_tilt_rad: list[float] = []
    num_terminated = 0
    num_time_out = 0
    nan_obs_env_steps = 0
    nan_reward_env_steps = 0

    action = np.zeros((num_envs, MICRODUCK_NUM_ACTION), dtype=np.float32)
    try:
        env.reset()
        for _ in range(steps):
            state = env.step(action)

            log = state.info.get("log", {})
            for key, value in log.items():
                if key.startswith("reward/"):
                    reward_sums[key[len("reward/") :]] = reward_sums.get(
                        key[len("reward/") :], 0.0
                    ) + float(value)
            reward_steps += 1

            obs = state.obs
            nan_obs_env_steps += int(
                np.sum(~np.all(np.isfinite(obs["obs"]), axis=1))
                + np.sum(~np.all(np.isfinite(obs["critic"]), axis=1))
            )
            nan_reward_env_steps += int(np.sum(~np.isfinite(state.reward)))

            done = state.terminated | state.truncated
            if np.any(done):
                done_ids = np.flatnonzero(done)
                num_terminated += int(np.sum(state.terminated))
                num_time_out += int(np.sum(state.truncated))
                episode_lengths.extend(env.episode_length_buf[done_ids].astype(float).tolist())
                robot = env.scene["robot"]
                with env.scene._scoped_state_reads():
                    terminal_root_z.extend(
                        robot.data.root_link_pos_w[done_ids, 2].astype(float).tolist()
                    )
                    gravity = robot.data.projected_gravity_b[done_ids]
                    tilt = np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
                    terminal_tilt_rad.extend(tilt.astype(float).tolist())
                env.reset(env_indices=done_ids)
    finally:
        env.close()

    return {
        "skipped": False,
        "task": "microduck_velocity_flat",
        "backend": "mjwarp",
        "seed": seed,
        "num_envs": num_envs,
        "ctrl_steps": steps,
        "action": "zeros (default-pose target)",
        "reward_term_step_mean": {
            name: value / max(reward_steps, 1) for name, value in sorted(reward_sums.items())
        },
        "episode_length": _distribution(episode_lengths),
        "episodes": {
            "total": num_terminated + num_time_out,
            "terminated": num_terminated,
            "time_out": num_time_out,
        },
        "terminal_root_z": _distribution(terminal_root_z),
        "terminal_tilt_rad": _distribution(terminal_tilt_rad),
        "nan": {
            "obs_env_steps": nan_obs_env_steps,
            "reward_env_steps": nan_reward_env_steps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Control steps per rollout (1000 = one 20 s episode at ctrl_dt=0.02).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=str, default=None, help="JSON output path (default: stdout)."
    )
    args = parser.parse_args()

    result = run_rollout(args.num_envs, args.steps, args.seed)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
