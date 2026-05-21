"""Test that training scripts can start with all task configs.

These tests verify that Hydra configs are complete and scripts don't crash on startup.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
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


def _write_sharpa_smoke_cache(cache_prefix, scale_values: list[float]) -> None:
    from unilab.envs.manipulation.sharpa_inhand.base import (
        SOURCE_DEFAULT_HAND_JOINT_POS_DEG,
        resolve_grasp_cache_file,
    )

    hand_qpos = np.deg2rad(np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float64))
    object_pose = np.asarray([-0.09559, -0.00517, 0.61906, 1.0, 0.0, 0.0, 0.0])
    cache = np.broadcast_to(np.concatenate([hand_qpos, object_pose]), (32, 29)).copy()
    for scale_value in scale_values:
        np.save(
            resolve_grasp_cache_file(str(cache_prefix), float(scale_value)),
            cache.astype(np.float32),
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "task",
    [
        "go1_joystick_flat/mujoco",
        "go2_joystick_flat/mujoco",
        "g1_walk_flat/mujoco",
        "g1_motion_tracking/mujoco",
        "g1_box_tracking/mujoco",
        "g1_flip_tracking/mujoco",
        "g1_wall_flip_tracking/mujoco",
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "task",
    ["sac/g1_walk_flat/mujoco", "sac/g1_walk_rough/mujoco", "td3/g1_walk_flat/mujoco"],
)
def test_offpolicy_task_configs_load(task):
    """Off-policy task configs can start training with supported MuJoCo owners."""
    if not _MLX_RUNTIME_USABLE:
        pytest.skip("mlx runtime aborts in subprocess on this host")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_offpolicy.py",
            f"algo={task.split('/', 1)[0]}",
            f"task={task}",
            "algo.max_iterations=1",
            "training.no_play=true",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Off-policy {task} failed:\n{result.stderr}"


@pytest.mark.slow
def test_ppo_sharpa_motrix_one_iteration_training_smoke(tmp_path):
    """Sharpa Motrix owner can run a minimal RSL-RL learn loop."""
    pytest.importorskip("motrixsim", reason="motrixsim not installed")
    cache_prefix = tmp_path / "sharpa_grasp"
    _write_sharpa_smoke_cache(
        cache_prefix,
        [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_rsl_rl.py",
            "task=sharpa_inhand/motrix",
            "algo.num_envs=16",
            "algo.num_steps_per_env=2",
            "algo.max_iterations=1",
            "algo.save_interval=100",
            "training.no_play=true",
            f"training.log_root={tmp_path / 'logs'}",
            f"env.grasp_cache_path={cache_prefix}",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        "Sharpa Motrix PPO one-iteration smoke failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Learning iteration 0/1" in result.stdout
    assert "reward/total" in result.stdout
