"""Test that training scripts can start with all task configs.

These tests verify that Hydra configs are complete and scripts don't crash on startup.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("mujoco")


@pytest.mark.veryslow
@pytest.mark.parametrize(
    "task",
    ["go1_joystick", "go2_joystick", "g1_joystick", "g1_motion_tracking"],
)
def test_appo_task_configs_load(task):
    """APPO can start training with all task configs."""
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
    ["go1_joystick", "go2_joystick"],
)
def test_offpolicy_task_configs_load(task):
    """Off-policy SAC can start training with all task configs."""
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
