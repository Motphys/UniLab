from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoPreset:
    preset_id: str
    algo: str
    task: str
    sim: str
    task_name: str
    algo_log_name: str
    source_run: str
    checkpoint_filename: str
    demo_run: str
    expected_output: str = "play_video.mp4"


DEMO_PRESETS = {
    "go2_joystick_mujoco_ppo": DemoPreset(
        preset_id="go2_joystick_mujoco_ppo",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        task_name="Go2JoystickFlatTerrain",
        algo_log_name="rsl_rl_ppo",
        source_run="2026-04-24_01-36-01_mujoco",
        checkpoint_filename="model_999.pt",
        demo_run="demo_go2_joystick_mujoco_ppo_v0",
    )
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_demo_preset(preset_name: str) -> DemoPreset:
    try:
        return DEMO_PRESETS[preset_name]
    except KeyError as exc:
        available = ", ".join(sorted(DEMO_PRESETS))
        raise SystemExit(
            f"Unknown demo preset {preset_name!r}. Available presets: {available}"
        ) from exc


def _task_log_root(root: Path, preset: DemoPreset) -> Path:
    return root / "logs" / preset.algo_log_name / preset.task_name


def _source_run_dir(root: Path, preset: DemoPreset) -> Path:
    return _task_log_root(root, preset) / preset.source_run


def _demo_run_dir(root: Path, preset: DemoPreset) -> Path:
    return _task_log_root(root, preset) / preset.demo_run


def materialize_demo_run(*, root: Path, preset: DemoPreset, refresh: bool) -> Path:
    source_run_dir = _source_run_dir(root, preset)
    source_checkpoint = source_run_dir / preset.checkpoint_filename
    if not source_checkpoint.is_file():
        raise SystemExit(
            "Demo checkpoint not found. Expected local asset at "
            f"{source_checkpoint}."
        )

    demo_run_dir = _demo_run_dir(root, preset)
    if refresh and demo_run_dir.exists():
        shutil.rmtree(demo_run_dir)

    demo_checkpoint = demo_run_dir / preset.checkpoint_filename
    if not demo_checkpoint.is_file():
        demo_run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, demo_checkpoint)

    return demo_run_dir


def build_demo_command(*, preset: DemoPreset, device: str | None = None) -> list[str]:
    command = [
        sys.executable,
        str(_repo_root() / "scripts" / "train_rsl_rl.py"),
        f"task={preset.task}/{preset.sim}",
        "training.play_only=true",
        f"algo.load_run={preset.demo_run}",
    ]
    if device is not None:
        command.append(f"training.device={device}")
    return command


def run_demo(*, preset_name: str, refresh: bool, device: str | None = None) -> int:
    root = _repo_root()
    preset = get_demo_preset(preset_name)
    demo_run_dir = materialize_demo_run(root=root, preset=preset, refresh=refresh)
    command = build_demo_command(preset=preset, device=device)
    env = os.environ.copy()
    env.setdefault("UV_PROJECT_ENVIRONMENT", str(root / ".venv"))
    returncode = subprocess.run(command, check=False, env=env).returncode
    if returncode == 0:
        video_path = demo_run_dir / preset.expected_output
        print(f"Demo preset: {preset.preset_id}")
        if video_path.is_file():
            print(f"Demo video: {video_path}")
        else:
            print(f"Demo completed, but expected video was not found: {video_path}")
    return returncode
