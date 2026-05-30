from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base.backend.motrix.backend import MotrixBackend
from unilab.base.backend.motrix.playback import run_motrix_playback
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.backend.mujoco.playback import run_mujoco_playback
from unilab.base.scene import SceneCfg
from unilab.training import (
    BackendAdapter,
    get_latest_checkpoint,
    get_latest_run,
    parse_checkpoint_path,
    resolve_checkpoint_path,
    resolve_task_checkpoint_path,
)
from unilab.visualization.playback import render_play_mode

_ROOT_DIR = Path(__file__).resolve().parents[2]
_CONF_DIR = _ROOT_DIR / "conf"


def _resolve_low_level_playback_flags(kwargs: dict[str, object]) -> dict[str, object]:
    record_video = kwargs.get("record_video")
    resolved_record_video = (
        bool(record_video) if record_video is not None else kwargs.get("output_video") is not None
    )
    headless = kwargs.get("headless")
    resolved_headless = bool(headless) if headless is not None else resolved_record_video
    updated = dict(kwargs)
    updated["record_video"] = resolved_record_video
    updated["headless"] = resolved_headless
    return updated


def _normalize_overrides(overrides: list[str] | None, *, offpolicy: bool = False) -> list[str]:
    normalized: list[str] = []
    algo = "sac"
    task_selected = False

    for override in overrides or []:
        if override.startswith("algo="):
            algo = override.split("=", 1)[1]
            normalized.append(override)
            continue
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        if offpolicy:
            normalized.append(f"task={algo}/g1_walk_flat/mujoco")
        else:
            normalized.append("task=go1_joystick_flat/mujoco")
    return normalized


def _ppo_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _offpolicy_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "offpolicy"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides, offpolicy=True))


def test_get_latest_run_and_checkpoint_support_shared_checkpoint_resolution(tmp_path: Path):
    task_dir = tmp_path / "logs" / "custom_ppo" / "MyTask"
    older_run = task_dir / "2024-01-01_00-00-00_mujoco"
    newer_run = task_dir / "2024-02-01_00-00-00_mujoco"
    older_run.mkdir(parents=True)
    newer_run.mkdir(parents=True)
    (older_run / "model_1.pt").write_bytes(b"")
    (newer_run / "model_5.pt").write_bytes(b"")
    (newer_run / "model_9.pt").write_bytes(b"")

    latest_run = get_latest_run(task_dir)
    latest_checkpoint = get_latest_checkpoint(newer_run)

    assert latest_run == newer_run
    assert latest_checkpoint == newer_run / "model_9.pt"


def test_resolve_checkpoint_path_accepts_integer_latest_run(tmp_path: Path):
    task_dir = tmp_path / "logs" / "sac" / "MyTask"
    run_dir = task_dir / "2024-02-01_00-00-00_motrix"
    run_dir.mkdir(parents=True)
    selected_model = run_dir / "model_9.pt"
    selected_model.write_bytes(b"")

    checkpoint_path, checkpoint_dir = resolve_checkpoint_path(task_dir, -1)

    assert checkpoint_path == selected_model
    assert checkpoint_dir == run_dir


def test_parse_checkpoint_path_uses_algo_log_name_from_cfg(tmp_path: Path):
    cfg = _ppo_cfg()
    cfg.algo.algo_log_name = "custom_ppo"

    run_dir = (
        tmp_path / "logs" / "custom_ppo" / cfg.training.task_name / "2024-01-01_00-00-00_mujoco"
    )
    run_dir.mkdir(parents=True)
    model_path = run_dir / "model_12.pt"
    model_path.write_bytes(b"")

    checkpoint_path, checkpoint_dir = parse_checkpoint_path(cfg, root_dir=tmp_path)

    assert checkpoint_path == model_path
    assert checkpoint_dir == run_dir


def test_parse_checkpoint_path_supports_explicit_checkpoint_from_cfg(tmp_path: Path):
    cfg = _ppo_cfg()
    cfg.algo.algo_log_name = "custom_ppo"
    cfg.algo.checkpoint = 12

    run_dir = (
        tmp_path / "logs" / "custom_ppo" / cfg.training.task_name / "2024-01-01_00-00-00_mujoco"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_9.pt").write_bytes(b"")
    selected_model = run_dir / "model_12.pt"
    selected_model.write_bytes(b"")

    checkpoint_path, checkpoint_dir = parse_checkpoint_path(cfg, root_dir=tmp_path)

    assert checkpoint_path == selected_model
    assert checkpoint_dir == run_dir


def test_parse_checkpoint_path_returns_run_dir_when_requested_checkpoint_is_missing(tmp_path: Path):
    cfg = _ppo_cfg()
    cfg.algo.algo_log_name = "custom_ppo"
    cfg.algo.checkpoint = 12

    run_dir = (
        tmp_path / "logs" / "custom_ppo" / cfg.training.task_name / "2024-01-01_00-00-00_mujoco"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_9.pt").write_bytes(b"")

    checkpoint_path, checkpoint_dir = parse_checkpoint_path(cfg, root_dir=tmp_path)

    assert checkpoint_path is None
    assert checkpoint_dir == run_dir


def test_resolve_task_checkpoint_path_supports_explicit_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_00-00-00_mujoco"
    run_dir.mkdir(parents=True)
    (run_dir / "model_9.pt").write_bytes(b"")
    selected_model = run_dir / "model_12.pt"
    selected_model.write_bytes(b"")

    checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
        tmp_path,
        task_name="MyTask",
        load_run="-1",
        algo_log_name="custom_ppo",
        checkpoint="12",
    )

    assert checkpoint_path == selected_model
    assert checkpoint_dir == run_dir


def test_resolve_task_checkpoint_path_returns_run_dir_when_checkpoint_missing(tmp_path: Path):
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_00-00-00_mujoco"
    run_dir.mkdir(parents=True)

    checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
        tmp_path,
        task_name="MyTask",
        load_run="-1",
        algo_log_name="custom_ppo",
        checkpoint="12",
    )

    assert checkpoint_path is None
    assert checkpoint_dir == run_dir


def test_resolve_task_checkpoint_path_accepts_integer_latest_run(tmp_path: Path):
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_00-00-00_mujoco"
    run_dir.mkdir(parents=True)
    selected_model = run_dir / "model_12.pt"
    selected_model.write_bytes(b"")

    checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
        tmp_path,
        task_name="MyTask",
        load_run=-1,
        algo_log_name="custom_ppo",
    )

    assert checkpoint_path == selected_model
    assert checkpoint_dir == run_dir


def test_backend_adapter_env_cfg_override_for_motrix_sac_g1_walk_flat():
    """Env cfg override carries reward + env preset fields. Algo is NOT touched."""
    cfg = _offpolicy_cfg(["task=sac/g1_walk_flat/motrix"])

    adapter = BackendAdapter(cfg, root_dir=_ROOT_DIR, algo_name="sac")
    env_cfg_override = adapter.build_task_env_cfg_override()

    # env_cfg_override has reward + env preset fields
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(2.2)
    assert env_cfg_override["domain_rand"]["randomize_kp"] is False
    assert env_cfg_override["domain_rand"]["randomize_kd"] is False
    # algo values come straight from YAML compose — no mutation, matches task owner values
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000


def test_backend_adapter_builds_play_scene_override():
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix", "training.play_only=true"])
    assert cfg.training.play_env_num == 16
    captured: dict[str, object] = {}

    def _fake_materializer(source_model_file: str, **kwargs) -> str:
        captured["source_model_file"] = source_model_file
        captured.update(kwargs)
        return "/tmp/g1_motion_tracking_play_scene.xml"

    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=_ROOT_DIR,
        algo_name="ppo",
        scene_materializer=_fake_materializer,
    ).build_play_env_cfg_override()

    assert cfg.training.play_env_num == 16
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)
    assert env_cfg_override["scene"].model_file == "/tmp/g1_motion_tracking_play_scene.xml"
    assert captured["ground_texture_file"] == str(
        _ROOT_DIR / "src/unilab/assets/robots/g1/textures/floor.png"
    )


def test_render_play_mode_uses_env_interactive_contract():
    class FakeEnv:
        def __init__(self):
            self.cfg = type(
                "Cfg", (), {"ctrl_dt": 0.02, "scene": SceneCfg(model_file="scene.xml")}
            )()
            self.init_calls: list[dict[str, object]] = []
            self.render_calls = 0

        def run_playback(self, **kwargs):
            kwargs = _resolve_low_level_playback_flags(kwargs)
            kwargs.pop("frame_state_getter", None)
            return run_motrix_playback(backend=self, env=self, **kwargs)

        def init_renderer(self, **kwargs):
            self.init_calls.append(kwargs)

        def render(self):
            self.render_calls += 1

    env = FakeEnv()
    seen: list[int] = []

    result = render_play_mode(
        env,
        sim_backend="motrix",
        initialize=lambda: 0,
        step=lambda obs: seen.append(obs) or obs + 1,
        num_steps=3,
        render_spacing=2.5,
    )

    assert result is None
    assert env.init_calls == [{"spacing": 2.5, "offset_mode": "grid"}]
    assert env.render_calls == 3
    assert seen == [0, 1, 2]


def test_backend_play_render_plan_maps_auto_by_backend(tmp_path: Path):
    motrix = MotrixBackend.__new__(MotrixBackend)
    motrix_plan = motrix.resolve_play_render_plan(
        play_render_mode="auto",
        play_steps=37,
        output_video=tmp_path / "motrix.mp4",
    )
    assert motrix_plan.mode == "interactive"
    assert motrix_plan.headless is False
    assert motrix_plan.record_video is False
    assert motrix_plan.num_steps is None
    assert motrix_plan.output_video is None

    mujoco = MuJoCoBackend.__new__(MuJoCoBackend)
    mujoco_plan = mujoco.resolve_play_render_plan(
        play_render_mode="auto",
        play_steps=37,
        output_video=tmp_path / "mujoco.mp4",
    )
    assert mujoco_plan.mode == "record"
    assert mujoco_plan.headless is True
    assert mujoco_plan.record_video is True
    assert mujoco_plan.num_steps == 37
    assert mujoco_plan.output_video == tmp_path / "mujoco.mp4"


def test_motrix_record_play_render_plan_is_headless(tmp_path: Path):
    backend = MotrixBackend.__new__(MotrixBackend)
    plan = backend.resolve_play_render_plan(
        play_render_mode="record",
        play_steps=12,
        output_video=tmp_path / "motrix.mp4",
    )

    assert plan.mode == "record"
    assert plan.headless is True
    assert plan.record_video is True
    assert plan.num_steps == 12


def test_motrix_interactive_run_playback_treats_window_close_as_done(capsys):
    class RenderClosedError(RuntimeError):
        pass

    class FakeEnv:
        cfg = type("Cfg", (), {"render_spacing": 1.0, "ctrl_dt": 0.02})()

    backend = MotrixBackend.__new__(MotrixBackend)
    backend.init_renderer = lambda **kwargs: None

    def _render():
        raise RenderClosedError("closed")

    backend.render = _render
    backend.capture_video_frame = lambda: np.zeros((2, 2, 3), dtype=np.uint8)

    result = backend.run_playback(
        env=FakeEnv(),
        initialize=lambda: 0,
        step=lambda obs: obs + 1,
        num_steps=None,
        output_video=None,
        headless=False,
        record_video=False,
    )

    assert result is None
    assert "Render window closed." in capsys.readouterr().out


def test_motrix_record_run_playback_does_not_swallow_render_closed(tmp_path: Path):
    class RenderClosedError(RuntimeError):
        pass

    class FakeEnv:
        cfg = type("Cfg", (), {"render_spacing": 1.0, "ctrl_dt": 0.02})()

    backend = MotrixBackend.__new__(MotrixBackend)
    backend.init_renderer = lambda **kwargs: None

    def _capture_video_frame():
        raise RenderClosedError("closed")

    backend.capture_video_frame = _capture_video_frame

    with pytest.raises(RenderClosedError):
        backend.run_playback(
            env=FakeEnv(),
            initialize=lambda: 0,
            step=lambda obs: obs + 1,
            num_steps=1,
            output_video=tmp_path / "play.mp4",
            headless=True,
            record_video=True,
        )


def test_render_play_mode_uses_motrix_native_video_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    class FakeEnv:
        def __init__(self):
            self.cfg = type(
                "Cfg", (), {"ctrl_dt": 0.05, "scene": SceneCfg(model_file="scene.xml")}
            )()
            self.init_calls: list[dict[str, object]] = []
            self.capture_calls = 0

        def run_playback(self, **kwargs):
            kwargs = _resolve_low_level_playback_flags(kwargs)
            kwargs.pop("frame_state_getter", None)
            return run_motrix_playback(backend=self, env=self, **kwargs)

        def init_renderer(self, **kwargs):
            self.init_calls.append(kwargs)

        def capture_video_frame(self) -> np.ndarray:
            self.capture_calls += 1
            return np.full((2, 3, 3), self.capture_calls, dtype=np.uint8)

    fake_media = types.ModuleType("mediapy")
    fake_media.write_video = lambda path, frames, fps: captured.update(  # type: ignore[attr-defined]
        {"video_path": path, "frames": frames, "fps": fps}
    )
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)

    env = FakeEnv()
    output_path = tmp_path / "motrix.mp4"
    result = render_play_mode(
        env,
        sim_backend="motrix",
        initialize=lambda: 0,
        step=lambda obs: obs + 1,
        num_steps=2,
        output_video=output_path,
        render_spacing=2.5,
        camera_kwargs={"cam_distance": 3.0},
    )

    assert result == str(output_path)
    assert env.init_calls == [
        {
            "spacing": 2.5,
            "offset_mode": "grid",
            "headless": True,
            "capture": True,
            "width": 1280,
            "height": 720,
            "camera_kwargs": {"cam_distance": 3.0},
        }
    ]
    assert env.capture_calls == 2
    assert captured["video_path"] == str(output_path)
    assert captured["fps"] == 20
    frames = captured["frames"]
    assert isinstance(frames, list)
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], np.full((2, 3, 3), 1, dtype=np.uint8))
    np.testing.assert_array_equal(frames[1], np.full((2, 3, 3), 2, dtype=np.uint8))


def test_render_play_mode_rejects_motrix_record_with_interactive_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    class FakeEnv:
        def __init__(self):
            self.cfg = type(
                "Cfg", (), {"ctrl_dt": 0.05, "scene": SceneCfg(model_file="scene.xml")}
            )()
            self.init_calls: list[dict[str, object]] = []
            self.capture_calls = 0

        def run_playback(self, **kwargs):
            kwargs = _resolve_low_level_playback_flags(kwargs)
            kwargs.pop("frame_state_getter", None)
            return run_motrix_playback(backend=self, env=self, **kwargs)

        def init_renderer(self, **kwargs):
            self.init_calls.append(kwargs)

        def capture_video_frame(self) -> np.ndarray:
            self.capture_calls += 1
            return np.full((2, 3, 3), self.capture_calls, dtype=np.uint8)

    fake_media = types.ModuleType("mediapy")
    fake_media.write_video = lambda path, frames, fps: captured.update(  # type: ignore[attr-defined]
        {"video_path": path, "frames": frames, "fps": fps}
    )
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)

    env = FakeEnv()
    output_path = tmp_path / "motrix_interactive.mp4"
    with pytest.raises(ValueError, match="Motrix video recording requires headless=true"):
        render_play_mode(
            env,
            sim_backend="motrix",
            initialize=lambda: 0,
            step=lambda obs: obs + 1,
            num_steps=2,
            output_video=output_path,
            headless=False,
            record_video=True,
            render_spacing=1.5,
        )

    assert env.init_calls == []
    assert env.capture_calls == 0
    assert captured == {}


def test_render_play_mode_defaults_to_env_physics_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    class FakeEnv:
        def __init__(self):
            self.cfg = type(
                "Cfg", (), {"ctrl_dt": 0.05, "scene": SceneCfg(model_file="scene.xml")}
            )()
            self.snapshot_calls = 0

        def get_physics_state_snapshot(self) -> np.ndarray:
            self.snapshot_calls += 1
            return np.full((2, 4), self.snapshot_calls, dtype=np.float32)

        def run_playback(self, **kwargs):
            kwargs = _resolve_low_level_playback_flags(kwargs)
            kwargs.pop("render_offset_mode", None)
            return run_mujoco_playback(env=self, **kwargs)

    def _render_states_get_frames(state_list, model_file, **kwargs):
        captured["states"] = state_list
        captured["model_file"] = model_file
        captured["camera_kwargs"] = kwargs
        return [np.zeros((2, 2, 3), dtype=np.uint8)]

    fake_media = types.ModuleType("mediapy")
    fake_media.write_video = lambda path, frames, fps: captured.update(  # type: ignore[attr-defined]
        {"video_path": path, "frames": frames, "fps": fps}
    )

    monkeypatch.setitem(sys.modules, "mediapy", fake_media)
    monkeypatch.setattr(
        "unilab.visualization.render_many.render_states_get_frames",
        _render_states_get_frames,
    )

    env = FakeEnv()
    output_path = tmp_path / "play.mp4"
    result = render_play_mode(
        env,
        sim_backend="mujoco",
        initialize=lambda: 0,
        step=lambda obs: obs + 1,
        num_steps=2,
        output_video=output_path,
    )

    assert result == str(output_path)
    assert env.snapshot_calls == 2
    assert captured["model_file"] == "scene.xml"
    assert captured["fps"] == 20


def test_render_play_mode_uses_visualized_per_env_playback_models_for_video_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import mujoco

    captured: dict[str, object] = {}
    visual_xml = """
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="2 2 0.1"/>
        <body name="hand" pos="0 0 0.3">
          <geom name="hand_geom" type="box" size="0.05 0.05 0.05"/>
        </body>
        <body name="object_body" pos="0 0 0.6">
          <geom name="object" type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    visual_model_path = tmp_path / "scene.xml"
    visual_model_path.write_text(visual_xml)

    class FakeEnv:
        def __init__(self):
            self.cfg = type(
                "Cfg",
                (),
                {"ctrl_dt": 0.05, "scene": SceneCfg(model_file=str(visual_model_path))},
            )()
            self.snapshot_calls = 0
            self._models = [
                mujoco.MjModel.from_xml_string(
                    "<mujoco><worldbody><body><geom name='object' type='box' size='0.1 0.1 0.1'/></body></worldbody></mujoco>"
                ),
                mujoco.MjModel.from_xml_string(
                    "<mujoco><worldbody><body><geom name='object' type='box' size='0.2 0.2 0.2'/></body></worldbody></mujoco>"
                ),
            ]

        def get_physics_state_snapshot(self) -> np.ndarray:
            self.snapshot_calls += 1
            return np.full((2, 2), self.snapshot_calls, dtype=np.float32)

        def get_playback_model(self, env_index: int | None = None):
            idx = 0 if env_index is None else int(env_index)
            return self._models[idx]

        def run_playback(self, **kwargs):
            kwargs = _resolve_low_level_playback_flags(kwargs)
            kwargs.pop("render_offset_mode", None)
            return run_mujoco_playback(env=self, **kwargs)

    def _render_states_get_frames(state_list, model_file, **kwargs):
        del kwargs
        captured["states"] = state_list
        captured["model_file"] = model_file
        assert isinstance(model_file, list)
        model0 = mujoco.MjModel.from_binary_path(model_file[0])
        model1 = mujoco.MjModel.from_binary_path(model_file[1])
        object0 = mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_GEOM, "object")
        object1 = mujoco.mj_name2id(model1, mujoco.mjtObj.mjOBJ_GEOM, "object")
        hand0 = mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_GEOM, "hand_geom")
        hand1 = mujoco.mj_name2id(model1, mujoco.mjtObj.mjOBJ_GEOM, "hand_geom")
        ground0 = mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        ground1 = mujoco.mj_name2id(model1, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        captured["object0_size"] = model0.geom_size[object0].copy()
        captured["object1_size"] = model1.geom_size[object1].copy()
        captured["hand0_size"] = model0.geom_size[hand0].copy()
        captured["hand1_size"] = model1.geom_size[hand1].copy()
        captured["ground0_size"] = model0.geom_size[ground0].copy()
        captured["ground1_size"] = model1.geom_size[ground1].copy()
        return [np.zeros((2, 2, 3), dtype=np.uint8)]

    fake_media = types.ModuleType("mediapy")
    fake_media.write_video = lambda path, frames, fps: captured.update(  # type: ignore[attr-defined]
        {"video_path": path, "frames": frames, "fps": fps}
    )

    monkeypatch.setitem(sys.modules, "mediapy", fake_media)
    monkeypatch.setattr(
        "unilab.visualization.render_many.render_states_get_frames",
        _render_states_get_frames,
    )

    env = FakeEnv()
    output_path = tmp_path / "play.mp4"
    result = render_play_mode(
        env,
        sim_backend="mujoco",
        initialize=lambda: 0,
        step=lambda obs: obs + 1,
        num_steps=2,
        output_video=output_path,
    )

    assert result == str(output_path)
    assert env.snapshot_calls == 2
    assert isinstance(captured["model_file"], list)
    model_files = captured["model_file"]
    assert len(model_files) == 2
    np.testing.assert_allclose(captured["object0_size"], [0.1, 0.1, 0.1])
    np.testing.assert_allclose(captured["object1_size"], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(captured["hand0_size"], captured["hand1_size"])
    np.testing.assert_allclose(captured["ground0_size"], captured["ground1_size"])


def test_render_play_mode_requires_env_snapshot_contract_for_video_export(tmp_path: Path):
    class FakeEnv:
        def __init__(self):
            self.cfg = type(
                "Cfg", (), {"ctrl_dt": 0.05, "scene": SceneCfg(model_file="scene.xml")}
            )()

        def get_physics_state_snapshot(self) -> np.ndarray:
            raise NotImplementedError("unsupported")

        def run_playback(self, **kwargs):
            kwargs = _resolve_low_level_playback_flags(kwargs)
            kwargs.pop("render_offset_mode", None)
            return run_mujoco_playback(env=self, **kwargs)

    with pytest.raises(NotImplementedError, match="unsupported"):
        render_play_mode(
            FakeEnv(),
            sim_backend="mujoco",
            initialize=lambda: 0,
            step=lambda obs: obs + 1,
            num_steps=1,
            output_video=tmp_path / "play.mp4",
        )
