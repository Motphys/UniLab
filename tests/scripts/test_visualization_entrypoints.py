from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

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


def test_visualize_task_env_parses_explicit_args():
    mod = _load_script("visualize_task_env")

    args = mod._parse_args(
        [
            "--task",
            "Go2JoystickRough",
            "--backend",
            "motrix",
            "--num_envs",
            "8",
        ]
    )

    assert args.task == "Go2JoystickRough"
    assert args.backend == "motrix"
    assert args.num_envs == 8


def test_motrix_camera_kwargs_focuses_single_terrain_spawn():
    mod = _load_script("visualize_task_env")

    class FakeSpawn:
        def origins_for(self, env_ids):
            assert env_ids.tolist() == [0]
            return np.asarray([[10.0, 20.0, 0.25]], dtype=np.float64)

    class FakeEnv:
        _spawn = FakeSpawn()

    camera_kwargs = mod._motrix_camera_kwargs(FakeEnv(), 1)

    assert camera_kwargs == {
        "cam_lookat": [10.0, 20.0, 0.75],
        "cam_distance": 4.0,
        "cam_elevation": -25.0,
        "cam_azimuth": 135.0,
    }


def test_motrix_camera_kwargs_frames_multiple_terrain_spawns():
    mod = _load_script("visualize_task_env")

    class FakeSpawn:
        def origins_for(self, env_ids):
            assert env_ids.tolist() == [0, 1, 2, 3]
            return np.asarray(
                [
                    [-36.0, 36.0, 0.0],
                    [36.0, -12.0, 0.0],
                    [-4.0, -44.0, 0.0],
                    [-12.0, -44.0, 0.0],
                ],
                dtype=np.float64,
            )

    class FakeEnv:
        _spawn = FakeSpawn()

    camera_kwargs = mod._motrix_camera_kwargs(FakeEnv(), 4)

    assert camera_kwargs["cam_lookat"] == [0.0, -4.0, 0.5]
    assert camera_kwargs["cam_distance"] > 4.0
    assert camera_kwargs["cam_elevation"] == -25.0
    assert camera_kwargs["cam_azimuth"] == 135.0


def test_mujoco_visual_xml_paths_prefer_backend_visual_scene(tmp_path: Path):
    mod = _load_script("visualize_task_env")
    robot_xml = tmp_path / "robot.xml"
    visual_xml = tmp_path / "scene.xml"

    class FakeScene:
        model_file = str(robot_xml)

    class FakeBackend:
        scene_visual_model_file = str(visual_xml)

    class FakeEnv:
        _backend = FakeBackend()
        cfg = type("Cfg", (), {"scene": FakeScene()})()

    parent, robot = mod._mujoco_visual_xml_paths(FakeEnv())

    assert parent == visual_xml
    assert robot == robot_xml
