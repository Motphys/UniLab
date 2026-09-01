"""Test that training scripts can start with all task configs.

These tests verify that Hydra configs are complete and scripts don't crash on startup.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

ROOT_DIR = Path(__file__).resolve().parents[2]


def _write_sharpa_smoke_cache(cache_prefix, scale_values: list[float]) -> None:
    from unilab.tasks.manipulation.sharpa_inhand.base import (
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


APPO_MUJOCO_SMOKE_TASKS = [
    "go1_joystick_flat/mujoco",
    "go2_joystick_flat/mujoco",
    "g1_walk_flat/mujoco",
    "g1_motion_tracking/mujoco",
    "g1_flip_tracking/mujoco",
    "g1_wall_flip_tracking/mujoco",
]

APPO_MOTION_SMOKE_TASKS = {
    "g1_motion_tracking/mujoco",
    "g1_flip_tracking/mujoco",
    "g1_wall_flip_tracking/mujoco",
}


def _write_g1_motion_smoke_npz(path: Path) -> None:
    num_frames = 4
    num_joints = 29
    num_bodies = 128
    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_pos_w[..., 2] = 0.8
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    np.savez(
        path,
        fps=np.array([50], dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )


def _appo_motion_file_overrides(task: str, tmp_path: Path) -> list[str]:
    if task not in APPO_MOTION_SMOKE_TASKS:
        return []
    motion_file = tmp_path / f"{task.split('/', 1)[0]}_smoke_motion.npz"
    _write_g1_motion_smoke_npz(motion_file)
    return [f"env.commands.motion.params.motion_file={motion_file}"]


def test_appo_mujoco_smoke_tasks_have_owner_configs():
    """APPO runtime smoke coverage must not declare tasks without owner YAML."""
    missing = [
        task
        for task in APPO_MUJOCO_SMOKE_TASKS
        if not (ROOT_DIR / "src" / "unilab" / "conf" / "appo" / "task" / f"{task}.yaml").is_file()
    ]
    assert missing == []


@pytest.mark.slow
@pytest.mark.parametrize("task", APPO_MUJOCO_SMOKE_TASKS)
def test_appo_task_configs_load(task, tmp_path):
    """APPO can start training with selected MuJoCo owner configs."""
    motion_overrides = _appo_motion_file_overrides(task, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "src/unilab/scripts/train_appo.py",
            f"task={task}",
            "algo.max_iterations=1",
            "training.no_play=true",
            *motion_overrides,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"APPO {task} failed:\n{result.stderr}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("algo", "task"),
    [
        ("sac", "g1_walk_flat/mujoco"),
        ("sac", "g1_walk_rough/mujoco"),
        ("td3", "g1_walk_flat/mujoco"),
    ],
)
def test_offpolicy_task_configs_load(algo, task):
    """Off-policy task configs can start training with supported MuJoCo owners."""
    result = subprocess.run(
        [
            sys.executable,
            f"src/unilab/scripts/train_{algo}.py",
            f"task={task}",
            "algo.max_iterations=1",
            "training.no_play=true",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Off-policy {algo} {task} failed:\n{result.stderr}"


@pytest.mark.slow
def test_ppo_sharpa_motrix_one_iteration_training_smoke(tmp_path):
    """Sharpa Motrix owner can run a minimal RSL-RL learn loop."""
    pytest.importorskip("motrixsim", reason="motrixsim not installed")
    cache_prefix = tmp_path / "sharpa_grasp"
    _write_sharpa_smoke_cache(
        cache_prefix,
        [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    )

    result = subprocess.run(
        [
            sys.executable,
            "src/unilab/scripts/train_rsl_rl.py",
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 CUDA devices")
@pytest.mark.slow
@pytest.mark.parametrize(
    ("task", "task_name"),
    [
        ("go2_joystick_flat/mujoco", "Go2JoystickFlat"),
        ("g1_motion_tracking/mujoco", "G1MotionTracking"),
    ],
)
def test_ppo_two_gpu_rsl_rl_training_smoke(task: str, task_name: str, tmp_path: Path) -> None:
    """RSL-RL PPO uses one env/storage replica per rank and rank-0 artifacts."""
    result = subprocess.run(
        [
            sys.executable,
            "src/unilab/scripts/train_rsl_rl.py",
            f"task={task}",
            "training.devices=[0,1]",
            "training.play_render_mode=record",
            "training.play_steps=2",
            "training.play_env_num=2",
            "training.nan_guard.enabled=false",
            f"training.log_root={tmp_path}",
            "algo.num_envs=64",
            "algo.num_steps_per_env=4",
            "algo.max_iterations=1",
            "algo.save_interval=100",
            "algo.algorithm.num_learning_epochs=1",
            "algo.algorithm.num_mini_batches=2",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"PPO two-GPU smoke failed for {task}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Synchronizing parameters for rank 0" in result.stdout
    assert "Synchronizing parameters for rank 1" in result.stdout
    assert "Done." in result.stdout
    assert "device used by this process is currently unknown" not in result.stderr

    run_dirs = list((tmp_path / task_name).glob("*_mujoco_gpux2"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert len(list(run_dir.glob("events.out.tfevents.*"))) == 1
    assert len(list(run_dir.glob("model_*.pt"))) == 1
    assert (run_dir / "play_video.mp4").is_file()
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["world_size"] == 2
    assert summary["num_envs_per_rank"] == 64
    assert summary["global_num_envs"] == 128
    assert summary["samples_per_iteration"] == 512
    assert summary["run_env_steps"] == 512
