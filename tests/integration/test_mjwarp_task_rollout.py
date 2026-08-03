"""Capability-derived production task rollout for managed MuJoCo/MJWarp rollout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tooling.acceptance.task_rollout import ROLLOUT_PLAN_PATH, load_task_rollout_plan
from tooling.acceptance.task_rollout_run import validate_task_rollout_run

from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies

ROOT_DIR = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.slow


def _require_cuda_mjwarp() -> None:
    try:
        dependencies = load_mjwarp_dependencies()
        is_cuda = bool(dependencies.warp.get_device().is_cuda)
    except Exception as exc:  # noqa: BLE001 - unavailable infrastructure is a hard failure
        pytest.fail(f"mjwarp task rollout could not initialize Warp: {type(exc).__name__}: {exc}")
    if not is_cuda:
        pytest.fail("mjwarp task rollout requires an active CUDA Warp device")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_promoted_tasks_pass_capability_derived_matrix(tmp_path: Path) -> None:
    """Run every currently promoted task/owner pair through the public CLI.

    The plan is intentionally the source for the seed and budget matrix.  Each
    subprocess gets an isolated log root so a stale run cannot satisfy a later
    seed's receipt checks.
    """

    _require_cuda_mjwarp()
    plan = load_task_rollout_plan(ROOT_DIR / ROLLOUT_PLAN_PATH)
    assert len(plan.entries) == 1
    entry = plan.entries[0]

    for seed in entry.seeds:
        log_root = tmp_path / f"seed-{seed}"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/train_rsl_rl.py",
                f"task={entry.task_slug}/{entry.backend}",
                f"algo.seed={seed}",
                f"algo.num_envs={entry.num_envs}",
                f"algo.num_steps_per_env={entry.num_steps_per_env}",
                f"algo.max_iterations={entry.max_iterations}",
                "algo.save_interval=1",
                "algo.capture_performance_diagnostics=true",
                "training.no_play=true",
                "training.logger=tensorboard",
                f"training.log_root={log_root}",
            ],
            cwd=ROOT_DIR,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"managed MuJoCo/MJWarp rollout task rollout failed for seed={seed}:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        run_dirs = sorted((log_root / entry.env_name).glob(f"*_{entry.backend}"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        report = validate_task_rollout_run(
            entry,
            seed=seed,
            run_dir=run_dir,
            run_config=_load_json(run_dir / "run_config.json"),
            run_summary=_load_json(run_dir / "run_summary.json"),
            stdout=result.stdout,
        )
        assert report.ok, report.errors
