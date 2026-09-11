#!/usr/bin/env python3
"""Capture a deterministic drift-characterization baseline for the mujoco backend.

Roadmap issue #1552 replaces the mujoco backend's native executor
(mujoco-uni-runtime, import name ``mujoco_uni``) with mjbatch. Numerical drift
between the two executors is expected and accepted; the quality gate is a
fixed model/seed/action-sequence trajectory diff against this baseline
(issue #1554), not a bit-exact gate.

This script builds each task's training environment through the exact Hydra
owner path used by the trainer (``conf/ppo`` task config -> ``BackendAdapter``
-> ``registry.make``), seeds every consumed RNG, pins the executor thread
count via ``cpu_ids``, drives the env with a fixed pseudo-random
action sequence (seeded ``numpy`` Generator, stored in the artifact; no neural
network), and records per-step observations (all obs groups), rewards, done
flags, and backend qpos/qvel read through the public ``SimBackend.get_state``
interface.

Per task it writes ``<output>/<task>.npz`` plus ``<output>/<task>.metadata.json``
(commit hash, package versions, task/seed/step configuration, per-array
SHA-256 digests). The default ``--output`` is the committed BEFORE baseline
location; after the dependency switch, re-run with the identical configuration
and a different ``--output`` (e.g. ``.../drift_baseline/after``) to produce the
AFTER artifact, then diff the two with any npz-aware comparison.

Note: the .npz zip container embeds file timestamps, so two runs are compared
on array contents (the metadata records per-array SHA-256 digests for exactly
this), not on container bytes.

Usage:
    # BEFORE capture (writes the committed baseline):
    uv run scripts/tools/capture_mujoco_drift_baseline.py

    # AFTER capture (identical configuration, different output dir):
    uv run scripts/tools/capture_mujoco_drift_baseline.py \
        --output scripts/tools/drift_baseline/after

    # Single task / smoke run:
    uv run scripts/tools/capture_mujoco_drift_baseline.py \
        --tasks go2w_joystick_flat/mujoco --steps 50 --output /tmp/drift_smoke

    # Determinism check: run twice into different dirs, compare array contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (str(SRC_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.training import ensure_registries

CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "scripts" / "tools" / "drift_baseline" / "before"

# Default tasks: Go2WJoystickFlat exercises the per-substep state-feedback
# control path (Go2WMixedAction via SimBackend.set_pre_step_control), the
# highest-risk surface for the executor swap; Go2JoystickFlat covers the plain
# position-action path on the same robot family.
DEFAULT_TASKS = ("go2w_joystick_flat/mujoco", "go2_joystick_flat/mujoco")

# Executor determinism contract: pin the pool worker count via cpu_ids so
# every run steps an identical partition. Recorded in metadata for the AFTER
# comparison. (The mujoco_uni-era chunk_size/adaptive_chunk_size knobs no
# longer exist in EnvCfg; mjbatch schedules per-sim work without a chunk
# knob, so there is nothing else to pin.)
DEFAULT_CPU_IDS = (0, 1, 2, 3)

METADATA_PACKAGES = ("mujoco", "mjbatch", "unisim-core", "numpy")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        metavar="TASK/BACKEND",
        help="Hydra task config paths under conf/ppo/task (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory for <task>.npz + <task>.metadata.json (default: %(default)s).",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Env seed: ManagerBasedRlEnvCfg.seed and env.reset(seed=...) (default: 42).",
    )
    parser.add_argument(
        "--action-seed",
        type=int,
        default=1234,
        help="Seed of the recorded numpy Generator action sequence (default: 1234).",
    )
    parser.add_argument(
        "--cpu-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_CPU_IDS),
        help="Explicit CPU ids; also fixes the executor pool worker count (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in METADATA_PACKAGES:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = "<not installed>"
    return versions


def _compose_task_cfg(task_path: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task={task_path}"])


def _build_env(task_path: str, args: argparse.Namespace):
    hydra_cfg = _compose_task_cfg(task_path)
    task_name = str(hydra_cfg.training.task_name)
    env_cfg_override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg_override.update(
        {
            "seed": args.seed,
            "cpu_ids": list(args.cpu_ids),
        }
    )
    env = registry.make(
        task_name,
        sim_backend=str(hydra_cfg.training.sim_backend),
        env_cfg_override=env_cfg_override,
        num_envs=args.num_envs,
    )
    return task_name, env


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_action_sequence(steps: int, num_envs: int, action_dim: int, seed: int) -> np.ndarray:
    """Fixed recorded action sequence: seeded Generator, no policy network.

    Base: iid uniform(-1, 1) for every env. Every 4th env (indices 3, 7, ...)
    instead holds a seeded saturated +/-1 action vector for the first half of
    the run and its negation for the second half; the saturated pose plus the
    mid-run reversal topples those envs, so the baseline also exercises
    termination, autoreset, and the backend set_state path mid-trajectory.
    """
    rng = np.random.default_rng(seed)
    actions = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(steps, num_envs, action_dim),
    ).astype(np.float32)
    held_envs = [i for i in range(num_envs) if i % 4 == 3]
    held = rng.choice([-1.0, 1.0], size=(len(held_envs), action_dim)).astype(np.float32)
    half = steps // 2
    for row, env_index in enumerate(held_envs):
        actions[:half, env_index] = held[row]
        actions[half:, env_index] = -held[row]
    return actions


def capture_task(task_path: str, args: argparse.Namespace) -> dict[str, Any]:
    task_name, env = _build_env(task_path, args)
    try:
        action_dim = int(env.action_space.shape[0])
        actions = make_action_sequence(args.steps, args.num_envs, action_dim, args.action_seed)

        # Belt and braces for any global-numpy consumer on the reset/step path;
        # the authoritative env RNG is seeded explicitly below and via cfg.seed.
        np.random.seed(args.seed)
        reset_obs, _ = env.reset(seed=args.seed)

        backend_state = env._backend.get_state
        obs_groups = sorted(env.obs_groups_spec)
        init_backend = backend_state(("qpos", "qvel"))

        traces: dict[str, list[np.ndarray]] = {
            "reward": [],
            "terminated": [],
            "truncated": [],
            "qpos": [],
            "qvel": [],
        }
        traces.update({f"obs/{name}": [] for name in obs_groups})

        for step in range(args.steps):
            state = env.step(actions[step])
            backend = backend_state(("qpos", "qvel"))
            for name in obs_groups:
                traces[f"obs/{name}"].append(np.asarray(state.obs[name]).copy())
            traces["reward"].append(np.asarray(state.reward).copy())
            traces["terminated"].append(np.asarray(state.terminated).copy())
            traces["truncated"].append(np.asarray(state.truncated).copy())
            traces["qpos"].append(np.asarray(backend["qpos"]).copy())
            traces["qvel"].append(np.asarray(backend["qvel"]).copy())
            if (step + 1) % 100 == 0:
                print(f"[capture] {task_name}: {step + 1}/{args.steps} steps", flush=True)

        arrays: dict[str, np.ndarray] = {name: np.stack(values) for name, values in traces.items()}
        arrays["actions"] = actions
        for name in obs_groups:
            arrays[f"obs_init/{name}"] = np.asarray(reset_obs[name]).copy()
        arrays["qpos_init"] = np.asarray(init_backend["qpos"]).copy()
        arrays["qvel_init"] = np.asarray(init_backend["qvel"]).copy()

        metadata: dict[str, Any] = {
            "task_name": task_name,
            "hydra_task_path": task_path,
            "sim_backend": env._backend.backend_type,
            "num_envs": args.num_envs,
            "num_steps": args.steps,
            "seed": args.seed,
            "action_seed": args.action_seed,
            "action_generator": (
                "make_action_sequence: default_rng(action_seed) iid uniform(-1, 1); "
                "envs i%4==3 hold a seeded +/-1 vector for the first half, negated after"
            ),
            "cpu_ids": list(args.cpu_ids),
            "sim_dt": float(env.cfg.sim_dt),
            "ctrl_dt": float(env.cfg.ctrl_dt),
            "obs_groups_spec": {k: int(v) for k, v in env.obs_groups_spec.items()},
            "action_dim": action_dim,
            "git_commit": _git_commit(),
            "package_versions": _package_versions(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "array_sha256": {name: _sha256(value) for name, value in arrays.items()},
        }
        return {"task_name": task_name, "arrays": arrays, "metadata": metadata}
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_envs < 1:
        raise SystemExit(f"--num-envs must be >= 1, got {args.num_envs}")
    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")

    ensure_registries()
    args.output.mkdir(parents=True, exist_ok=True)

    for task_path in args.tasks:
        print(f"[capture] task={task_path} num_envs={args.num_envs} steps={args.steps}", flush=True)
        result = capture_task(task_path, args)
        task_name = result["task_name"]
        npz_path = args.output / f"{task_name}.npz"
        metadata_path = args.output / f"{task_name}.metadata.json"
        np.savez_compressed(npz_path, **result["arrays"])
        metadata_path.write_text(json.dumps(result["metadata"], indent=2) + "\n")
        print(
            f"[capture] wrote {npz_path} ({npz_path.stat().st_size / 1e6:.2f} MB) "
            f"and {metadata_path.name}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
