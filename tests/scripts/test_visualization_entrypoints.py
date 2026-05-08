from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _load_script(name: str) -> Any:
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_legacy_visualization_env_entrypoint_is_removed():
    assert not (_SCRIPTS_DIR / "visualization_env.py").exists()


def test_visualize_task_env_keeps_canonical_defaults():
    mod = _load_script("visualize_task_env")

    args = mod._parse_args([])

    assert args.task == "Go2JoystickFlat"
    assert args.backend == "mujoco"
    assert args.num_envs == 4
    assert args.env_spacing is None


def test_visualize_task_env_parses_explicit_args():
    mod = _load_script("visualize_task_env")

    args = mod._parse_args(
        [
            "--task",
            "Go2JoystickRough",
            "--backend",
            "motrix",
            "--num_envs",
            "4",
            "--env_spacing",
            "2",
        ]
    )

    assert args.task == "Go2JoystickRough"
    assert args.backend == "motrix"
    assert args.num_envs == 4
    assert args.env_spacing == 2.0
