from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _drake_batch_available() -> bool:
    return _module_available("drakeuni.compiled._drake_env_pool")


@pytest.mark.slow
@pytest.mark.skipif(
    not _drake_batch_available(),
    reason="optional DrakeUni batch extension has not been built",
)
@pytest.mark.parametrize("task", ["go1_joystick_flat/drake", "go2_joystick_flat/drake"])
def test_drake_ppo_one_iteration_training_smoke(task: str, tmp_path: Path) -> None:
    """Drake task configs can run the real RSL-RL training entry point."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_rsl_rl.py",
            f"task={task}",
            "training.no_play=true",
            f"training.log_root={tmp_path / 'logs'}",
            "algo.num_envs=4",
            "algo.num_steps_per_env=4",
            "algo.max_iterations=1",
            "algo.save_interval=100",
            "env.drake_nthread=1",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"Drake PPO smoke failed for {task}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Learning iteration 0/1" in result.stdout
