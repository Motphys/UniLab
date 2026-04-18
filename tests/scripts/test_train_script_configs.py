"""Test that training scripts can start with all task configs.

These tests verify that Hydra configs are complete and scripts don't crash on startup.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("mujoco")


def _mlx_runtime_usable() -> bool:
    """Probe whether importing mlx.core is safe in a subprocess on this host."""
    if sys.platform != "darwin":
        return True
    result = subprocess.run(
        [sys.executable, "-c", "import mlx.core"], capture_output=True, text=True, timeout=10
    )
    return result.returncode == 0


_MLX_RUNTIME_USABLE = _mlx_runtime_usable()


@pytest.mark.veryslow
@pytest.mark.parametrize(
    "task",
    [
        "go1_joystick_flat/mujoco",
        "go2_joystick_flat/mujoco",
        "g1_joystick_flat/mujoco",
        "g1_motion_tracking/mujoco",
        "g1_flip_tracking/mujoco",
    ],
)
def test_appo_task_configs_load(task):
    """APPO can start training with all supported task configs."""
    if not _MLX_RUNTIME_USABLE:
        pytest.skip("mlx runtime aborts in subprocess on this host")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_appo.py",
            f"task={task}",
            "algo.max_iterations=1",
            "training.no_play=true",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"APPO {task} failed:\n{result.stderr}"


@pytest.mark.veryslow
@pytest.mark.parametrize(
    "task",
    ["sac/go1_joystick_flat/mujoco", "sac/go2_joystick_flat/mujoco"],
)
def test_offpolicy_task_configs_load(task):
    """Off-policy SAC can start training with all supported task configs."""
    if not _MLX_RUNTIME_USABLE:
        pytest.skip("mlx runtime aborts in subprocess on this host")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_offpolicy.py",
            "algo=sac",
            f"task={task}",
            "algo.max_iterations=1",
            "training.no_play=true",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"SAC {task} failed:\n{result.stderr}"
