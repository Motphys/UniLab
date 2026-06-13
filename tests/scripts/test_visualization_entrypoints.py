from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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


def _keyboard_env(
    with_commands: bool = True,
    *,
    env_cls_name: str = "Env",
    cfg_cls_name: str = "Cfg",
    module: str = "tests.fake_env",
    obs_contains_command: bool = False,
) -> Any:
    info: dict[str, Any] = (
        {"commands": np.asarray([[0.37, -0.23, 0.19]], dtype=np.float32)}
        if with_commands
        else {"steps": 0}
    )
    if obs_contains_command:
        info["commands"] = np.asarray([[0.37, -0.23, 0.19]], dtype=np.float32)
    commands_cfg = (
        type(
            "Cmds",
            (),
            {
                "vel_limit": [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]],
                "heading_command": True,
                "resampling_time": 10.0,
            },
        )()
        if with_commands
        else None
    )
    cfg_type = type(cfg_cls_name, (), {"__module__": module})
    cfg = cfg_type()
    cfg.commands = commands_cfg
    obs = (
        {"obs": np.asarray([[1.0, 0.37, -0.23, 0.19, 2.0]], dtype=np.float32)}
        if obs_contains_command
        else {"obs": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)}
    )
    state = type("State", (), {"info": info, "obs": obs})()
    env_type = type(env_cls_name, (), {"__module__": module})
    env = env_type()
    env.state = state
    env.cfg = cfg
    return env


def test_build_keyboard_commander_disabled_when_flag_off():
    mod = _load_script("play_interactive")
    args = type("Args", (), {"keyboard": False})()
    assert mod._build_keyboard_commander(_keyboard_env(), args) is None


def test_build_keyboard_commander_ignored_without_commands(capsys):
    mod = _load_script("play_interactive")
    args = type("Args", (), {"keyboard": True})()
    assert mod._build_keyboard_commander(_keyboard_env(with_commands=False), args) is None
    assert "no velocity 'commands'" in capsys.readouterr().out


def test_build_keyboard_commander_makes_keyboard_authoritative():
    mod = _load_script("play_interactive")
    env = _keyboard_env()
    args = type(
        "Args", (), {"keyboard": True, "keyboard_step_lin": 0.15, "keyboard_step_ang": 0.25}
    )()

    commander = mod._build_keyboard_commander(env, args)

    assert commander is not None
    assert commander.step_lin == 0.15
    assert commander.step_ang == 0.25
    # Heading P-control and resampling are turned off so they cannot fight the keyboard.
    assert env.cfg.commands.heading_command is False
    assert env.cfg.commands.resampling_time == 0.0
    assert env.state.info["commands"].tolist() == [[0.0, 0.0, 0.0]]


def test_velocity_arrows_require_velocity_command_task_and_policy_obs():
    mod = _load_script("play_interactive")

    joystick_env = _keyboard_env(
        env_cls_name="Go2WalkTask",
        cfg_cls_name="Go2JoystickCfg",
        module="unilab.envs.locomotion.go2.joystick",
        obs_contains_command=True,
    )
    manip_loco_env = _keyboard_env(
        env_cls_name="Go2ArmManipLocoEnv",
        cfg_cls_name="Go2ArmManipLocoCfg",
        module="unilab.envs.locomotion.go2_arm.manip_loco",
        obs_contains_command=True,
    )
    missing_obs_command_env = _keyboard_env(
        env_cls_name="Go2WalkTask",
        cfg_cls_name="Go2JoystickCfg",
        module="unilab.envs.locomotion.go2.joystick",
        obs_contains_command=False,
    )

    assert mod._should_render_velocity_arrows(joystick_env) is True
    assert mod._should_render_velocity_arrows(manip_loco_env) is False
    assert mod._should_render_velocity_arrows(missing_obs_command_env) is False


def test_handle_command_key_maps_drive_style_keys():
    mod = _load_script("play_interactive")
    commander = mod.KeyboardCommander.from_vel_limit([[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]])

    mod._handle_command_key(commander, mod._KEY_UP)  # forward (vx +)
    mod._handle_command_key(commander, mod._KEY_LEFT)  # turn left (vyaw +)
    assert commander.command == pytest.approx([0.1, 0.0, 0.2])

    mod._handle_command_key(commander, mod._KEY_RIGHT)  # turn right cancels yaw
    assert commander.command[2] == pytest.approx(0.0)

    before = commander.command.copy()
    mod._handle_command_key(commander, ord("q"))  # unmapped key is a no-op
    assert commander.command.tolist() == before.tolist()

    mod._handle_command_key(commander, mod._KEY_ENTER)  # full stop
    assert commander.command.tolist() == [0.0, 0.0, 0.0]


def test_play_interactive_viewer_model_uses_shared_render_playback_resolver(
    tmp_path: Path, monkeypatch
):
    mod = _load_script("play_interactive")
    visual_xml = tmp_path / "scene.xml"
    visual_xml.write_text("<mujoco/>", encoding="utf-8")

    import mujoco

    loaded_binary: list[str] = []
    resolved: dict[str, object] = {}
    viewer_model = object()

    def fake_from_binary_path(path: str):
        loaded_binary.append(path)
        return viewer_model

    def fake_resolve_render_play_model_files(env, *, num_envs: int, tmp_dir: str | Path):
        resolved["env"] = env
        resolved["num_envs"] = num_envs
        resolved["tmp_dir"] = tmp_dir
        output_path = Path(tmp_dir) / "model_0.mjb"
        output_path.write_bytes(b"fake-mjb")
        return str(output_path)

    monkeypatch.setattr(mujoco.MjModel, "from_binary_path", fake_from_binary_path)
    monkeypatch.setattr(
        mod,
        "resolve_render_play_model_files",
        fake_resolve_render_play_model_files,
    )

    class FakeBackend:
        scene_visual_model_file = str(visual_xml)

    class FakeEnv:
        _backend = FakeBackend()

    env = FakeEnv()
    model = mod._load_viewer_model(env, use_env_visual_model=False)

    assert model is viewer_model
    assert len(loaded_binary) == 1
    assert Path(loaded_binary[0]).name == "model_0.mjb"
    assert resolved["env"] is env
    assert resolved["num_envs"] == 1
    assert Path(resolved["tmp_dir"]).name.startswith("unilab-interactive-viewer-")
