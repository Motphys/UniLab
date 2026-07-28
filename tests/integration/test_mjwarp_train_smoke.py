"""Real-CUDA PPO liveness test for the production ``mjwarp`` owner route."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies

ROOT_DIR = Path(__file__).resolve().parents[2]
_NUM_ENVS = 128
_STEPS_PER_ENV = 2

pytestmark = pytest.mark.slow


def _require_cuda_mjwarp() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp PPO liveness requires an active CUDA Warp device")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _assert_finite_if_present(value: object, name: str) -> None:
    if value is not None:
        assert isinstance(value, (int, float)), f"{name} must be numeric or null"
        assert math.isfinite(float(value)), f"{name} must be finite"


def test_g1_one_iteration_uses_production_mjwarp(tmp_path: Path) -> None:
    """The registered owner completes rollout, learner, and checkpoint on CUDA.

    This deliberately invokes the public training CLI instead of importing a
    backend object or a benchmark helper.  The persisted run artifacts are the
    oracle for owner/backend identity and completion; stdout independently
    proves that RSL-RL reached both rollout collection and the learner phase.
    """
    _require_cuda_mjwarp()
    log_root = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_rsl_rl.py",
            "task=g1_walk_flat/mjwarp",
            f"algo.num_envs={_NUM_ENVS}",
            f"algo.num_steps_per_env={_STEPS_PER_ENV}",
            "algo.max_iterations=1",
            "algo.save_interval=1",
            "training.no_play=true",
            "training.logger=tensorboard",
            f"training.log_root={log_root}",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, (
        "mjwarp PPO one-iteration smoke failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Using device: cuda" in result.stdout
    assert "Learning iteration 0/1" in result.stdout
    assert "Collection time:" in result.stdout
    assert "Learning time:" in result.stdout

    run_dirs = sorted((log_root / "G1WalkFlat").glob("*_mjwarp"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run_config = _load_json(run_dir / "run_config.json")
    run_summary = _load_json(run_dir / "run_summary.json")

    run_metadata = run_config["run"]
    resolved_config = run_config["config"]
    assert run_metadata["sim_backend"] == "mjwarp"
    assert str(run_metadata["device"]).startswith("cuda")
    assert resolved_config["training"]["sim_backend"] == "mjwarp"
    assert resolved_config["training"]["task_name"] == "G1WalkFlat"
    assert resolved_config["algo"]["num_envs"] == _NUM_ENVS
    assert resolved_config["algo"]["num_steps_per_env"] == _STEPS_PER_ENV
    assert resolved_config["algo"]["max_iterations"] == 1

    assert run_summary["status"] == "completed"
    assert run_summary["sim_backend"] == "mjwarp"
    assert run_summary["completed_iterations"] == 1
    assert run_summary["total_env_steps"] == _NUM_ENVS * _STEPS_PER_ENV
    for name in (
        "training_wall_time_sec",
        "wall_time_sec",
        "peak_process_rss_bytes",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
        "final_mean_reward",
        "best_mean_reward",
        "mean_episode_length",
    ):
        _assert_finite_if_present(run_summary.get(name), name)
    assert float(run_summary["training_wall_time_sec"]) > 0.0
    assert int(run_summary["peak_gpu_memory_allocated_bytes"]) > 0
    assert int(run_summary["peak_gpu_memory_reserved_bytes"]) > 0

    checkpoint = run_dir / "model_0.pt"
    assert Path(run_summary["last_checkpoint"]) == checkpoint
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0
