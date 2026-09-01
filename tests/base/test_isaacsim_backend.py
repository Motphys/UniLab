"""Contract and lifecycle tests for the IsaacSim subprocess adapter.

The tests use the deterministic protocol worker already used by the IsaacGym
adapter.  It exercises the host-side IPC/shared-memory lifecycle without
requiring Kit or a GPU, while the small worker helpers are tested with tiny
fake USD objects.
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unilab.base.backend import create_backend
from unilab.base.backend.base import RenderClosedError
from unilab.base.backend.isaacgym.backend import IsaacGymWorkerError
from unilab.base.backend.isaacsim.backend import (
    IsaacSimBackend,
    IsaacSimRenderError,
    IsaacSimWorkerError,
)
from unilab.base.backend.isaacsim.dependencies import (
    ENV_HOME,
    ENV_PYTHON,
    IsaacSimDependencyError,
    build_worker_env,
    resolve_isaacsim_runtime,
)
from unilab.base.backend.isaacsim.worker import (
    _quat_rotate_wxyz,
    _resolve_articulation_root_prim_path,
)
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg

_MOCK_WORKER = str(Path(__file__).resolve().parent / "isaacgym_mock_worker.py")
SIM_DT = 0.005
NUM_ENVS = 2

_ROBOT_XML = """
<mujoco>
  <worldbody>
    <geom name="floor_geom" type="plane" size="0 0 0.05"/>
    <body name="base">
      <freejoint/>
      <site name="imu_site"/>
      <geom name="base_geom" size="0.1"/>
      <body name="link0">
        <joint name="j0" type="hinge" range="-1.5 1.5"/>
        <geom name="g0" size="0.1"/>
        <body name="link1">
          <joint name="j1" type="hinge" range="-1.5 1.5"/>
          <geom name="g1" size="0.1"/>
          <body name="link2">
            <joint name="j2" type="hinge" range="-1.5 1.5"/>
            <geom name="foot_geom" size="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="j0" joint="j0" kp="10" kv="0.5"/>
    <position name="j1" joint="j1" kp="20" kv="1.0"/>
    <position name="j2" joint="j2" kp="30"/>
  </actuator>
</mujoco>
"""

_SCENE_XML = """
<mujoco>
  <include file="robot.xml"/>
  <sensor>
    <gyro name="base_gyro" site="imu_site"/>
    <contact name="foot_contact" geom1="floor_geom" geom2="foot_geom" data="found" num="1"/>
  </sensor>
  <keyframe>
    <key name="home" qpos="0 0 0.8 1 0 0 0 0.1 0.2 0.3"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture()
def scene_file(tmp_path: Path) -> str:
    (tmp_path / "robot.xml").write_text(textwrap.dedent(_ROBOT_XML), encoding="utf-8")
    scene = tmp_path / "scene.xml"
    scene.write_text(textwrap.dedent(_SCENE_XML), encoding="utf-8")
    return str(scene)


def _make_backend(scene_file: str, **kwargs: Any) -> IsaacSimBackend:
    kwargs.setdefault("worker_command", [sys.executable, _MOCK_WORKER])
    kwargs.setdefault("worker_timeout_s", 30.0)
    backend = create_backend(
        "isaacsim",
        SceneCfg(model_file=scene_file),
        NUM_ENVS,
        SIM_DT,
        base_name="base",
        **kwargs,
    )
    assert isinstance(backend, IsaacSimBackend)
    return backend


@pytest.fixture()
def backend(scene_file: str) -> IsaacSimBackend:
    instance = _make_backend(scene_file)
    instance.materialize()
    try:
        yield instance
    finally:
        instance.close()


def _make_runtime_tree(home: Path) -> None:
    python = home / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    (home / "venv" / "lib" / "python3.11" / "site-packages").mkdir(parents=True)
    (home / "IsaacLab" / "source").mkdir(parents=True)


def test_factory_routes_isaacsim_without_importing_kit(scene_file: str) -> None:
    backend = _make_backend(scene_file)
    try:
        assert backend.backend_type == "isaacsim"
        assert isinstance(backend, IsaacSimBackend)
        assert backend._proc is None
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"isaacsim_render_mode": "video"}, "isaacsim_render_mode"),
        ({"isaacsim_render_width": 0}, "isaacsim_render_width"),
        ({"isaacsim_render_height": True}, "isaacsim_render_height"),
    ],
)
def test_env_cfg_rejects_invalid_isaacsim_render_settings(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EnvCfg(**kwargs).validate()


def test_dependencies_resolve_default_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    _make_runtime_tree(tmp_path)
    runtime = resolve_isaacsim_runtime()
    assert runtime.python == tmp_path / "venv" / "bin" / "python"
    assert runtime.package_path == tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    assert runtime.isaaclab_source == tmp_path / "IsaacLab" / "source"
    env = build_worker_env(runtime)
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(tmp_path / "venv" / "lib")
    assert env["PATH"].split(":")[0] == str(tmp_path / "venv" / "bin")
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path / "IsaacLab" / "source")
    assert env["OMNI_KIT_ACCEPT_EULA"] == "1"


def test_dependencies_override_and_missing_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    _make_runtime_tree(tmp_path)
    custom = tmp_path / "custom-python"
    custom.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(ENV_PYTHON, str(custom))
    assert resolve_isaacsim_runtime().python == custom

    empty = tmp_path / "empty"
    monkeypatch.setenv(ENV_HOME, str(empty))
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    with pytest.raises(IsaacSimDependencyError, match="setup_isaacsim_env.sh"):
        resolve_isaacsim_runtime()


def test_materialize_uses_shared_protocol_and_exposes_state(backend: IsaacSimBackend) -> None:
    assert backend.model.num_dof == 3
    assert backend.model.num_bodies == 4
    assert backend.get_actuator_names() == ("j0", "j1", "j2")
    np.testing.assert_allclose(backend.get_default_qpos(), [0, 0, 0.8, 1, 0, 0, 0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(backend.get_base_pos(), [[0, 0, 0.8]] * NUM_ENVS)
    np.testing.assert_allclose(backend.get_dof_pos(), [[0.1, 0.2, 0.3]] * NUM_ENVS)
    result = backend.step(np.ones((NUM_ENVS, 3), dtype=np.float32), nsteps=1)
    assert result["timing"]["physics_ms"] >= 0.0


def test_env_cleanup_hook_reaps_isaacsim_worker(backend: IsaacSimBackend) -> None:
    """The NpEnv cleanup hook must not leave the external worker alive."""
    proc = backend._proc
    assert proc is not None and proc.poll() is None
    shm_names = [handle.name for handle in backend._shm_handles.values()]

    backend.cleanup_scene_assets()

    assert proc.poll() is not None
    assert backend._proc is None
    from multiprocessing import shared_memory

    for name in shm_names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)


def test_contact_sensor_is_explicitly_unsupported(backend: IsaacSimBackend) -> None:
    metadata = backend._scene_metadata
    assert metadata is not None
    assert "foot_contact" in metadata.unsupported_sensors
    with pytest.raises(NotImplementedError, match="contact-force reporting"):
        backend.get_sensor_data("foot_contact")
    # Non-contact sensors remain available through the inherited cached path.
    assert backend.get_sensor_data("base_gyro").shape == (NUM_ENVS, 3)


def test_play_contract_advertises_native_rendering(
    backend: IsaacSimBackend,
    scene_file: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps = backend.get_play_capabilities()
    assert caps.supports_native_interactive_renderer
    assert caps.supports_native_video_capture
    assert not caps.supports_physics_state_playback

    plan = backend.resolve_play_render_plan(
        play_render_mode="none", play_steps=3, output_video=tmp_path / "ignored.mp4"
    )
    assert plan.mode == "none" and plan.headless and not plan.record_video
    with pytest.raises(IsaacSimRenderError, match="before env creation"):
        backend.resolve_play_render_plan(
            play_render_mode="auto", play_steps=3, output_video=tmp_path / "play.mp4"
        )

    monkeypatch.setenv("DISPLAY", ":0")
    interactive_backend = _make_backend(scene_file, render_mode="auto")
    try:
        interactive = interactive_backend.resolve_play_render_plan(
            play_render_mode="auto", play_steps=3, output_video=None
        )
        assert interactive.mode == "interactive" and not interactive.headless
    finally:
        interactive_backend.close()

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    record_backend = _make_backend(scene_file, render_mode="auto")
    try:
        record = record_backend.resolve_play_render_plan(
            play_render_mode="auto", play_steps=3, output_video=tmp_path / "play.mp4"
        )
        assert record.mode == "record" and record.headless and record.record_video
        explicit = record_backend.resolve_play_render_plan(
            play_render_mode="record", play_steps=7, output_video=tmp_path / "explicit.mp4"
        )
        assert explicit.mode == "record" and explicit.num_steps == 7
        with pytest.raises(IsaacSimRenderError, match="playback requested 'interactive'"):
            record_backend.resolve_play_render_plan(
                play_render_mode="interactive", play_steps=3, output_video=None
            )
        with pytest.raises(ValueError, match="play_steps"):
            record_backend.resolve_play_render_plan(
                play_render_mode="record", play_steps=None, output_video=tmp_path / "play.mp4"
            )
        with pytest.raises(ValueError, match="positive finite"):
            record_backend.resolve_play_render_plan(
                play_render_mode="record", play_steps=0, output_video=tmp_path / "play.mp4"
            )
        with pytest.raises(ValueError, match="output video path"):
            record_backend.resolve_play_render_plan(
                play_render_mode="record", play_steps=3, output_video=None
            )
    finally:
        record_backend.close()


def test_record_renderer_roundtrip(scene_file: str) -> None:
    backend = _make_backend(
        scene_file,
        render_mode="record",
        render_width=64,
        render_height=48,
    )
    backend.materialize()
    try:
        backend.init_renderer(headless=True, capture=True, width=64, height=48)
        frame = backend.capture_video_frame()
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8
        assert frame.flags.c_contiguous
        assert int(np.ptp(frame)) > 0
        with pytest.raises(IsaacSimRenderError, match="worker started"):
            backend.init_renderer(headless=False, capture=False, width=64, height=48)
        with pytest.raises(IsaacSimRenderError, match="positive integers"):
            backend.init_renderer(headless=True, capture=True, width=True, height=48)
    finally:
        backend.close()


def test_record_playback_writes_video(scene_file: str, tmp_path: Path) -> None:
    output = tmp_path / "play_video.mp4"
    backend = _make_backend(
        scene_file,
        render_mode="record",
        render_width=64,
        render_height=48,
    )
    backend.materialize()
    try:
        result = backend.run_playback(
            env=SimpleNamespace(cfg=SimpleNamespace(ctrl_dt=0.02)),
            initialize=lambda: 0,
            step=lambda obs: obs + 1,
            num_steps=3,
            output_video=output,
            headless=True,
            record_video=True,
        )
        assert result == str(output)
        assert output.exists() and output.stat().st_size > 0
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("behavior", "message"),
    [
        ("capture_uniform", "empty/uniform"),
        ("capture_float", "dtype=float32"),
        ("capture_wrong_shape", "invalid RGB frame"),
    ],
)
def test_record_frame_contract_fails_closed(
    scene_file: str,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    message: str,
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", behavior)
    backend = _make_backend(
        scene_file,
        render_mode="record",
        render_width=64,
        render_height=48,
    )
    backend.materialize()
    try:
        backend.init_renderer(headless=True, capture=True, width=64, height=48)
        with pytest.raises(IsaacSimRenderError, match=message):
            backend.capture_video_frame()
    finally:
        backend.close()


def test_interactive_renderer_roundtrip(scene_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    backend = _make_backend(scene_file, render_mode="interactive")
    backend.materialize()
    try:
        backend.init_renderer(headless=False, capture=False)
        backend.render()
    finally:
        backend.close()


def test_interactive_playback_routes_startup_dimensions_and_camera(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    backend = _make_backend(
        scene_file,
        render_mode="interactive",
        render_width=64,
        render_height=48,
    )
    init_calls: list[dict[str, Any]] = []
    original_init_renderer = backend.init_renderer

    def record_init_renderer(*args: Any, **kwargs: Any) -> None:
        init_calls.append(dict(kwargs))
        original_init_renderer(*args, **kwargs)

    backend.init_renderer = record_init_renderer  # type: ignore[method-assign]
    backend.materialize()
    try:
        result = backend.run_playback(
            env=SimpleNamespace(cfg=None),
            initialize=lambda: 0,
            step=lambda obs: obs + 1,
            num_steps=1,
            headless=False,
            record_video=False,
            camera_kwargs={
                "cam_distance": 3.0,
                "cam_elevation": -15.0,
                "cam_azimuth": 45.0,
            },
        )
        assert result is None
        assert init_calls == [
            {
                "headless": False,
                "width": 64,
                "height": 48,
                "camera_kwargs": {
                    "cam_distance": 3.0,
                    "cam_elevation": -15.0,
                    "cam_azimuth": 45.0,
                },
            }
        ]
    finally:
        backend.close()


def test_interactive_window_close_maps_to_interface_error(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "close_on_render")
    backend = _make_backend(scene_file, render_mode="interactive")
    backend.materialize()
    try:
        backend.init_renderer(headless=False, capture=False)
        with pytest.raises(RenderClosedError, match="closed"):
            backend.render()
    finally:
        backend.close()


def test_interactive_playback_stops_when_window_closes(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "close_on_render")
    backend = _make_backend(scene_file, render_mode="interactive")
    backend.materialize()
    steps: list[int] = []
    try:
        result = backend.run_playback(
            env=SimpleNamespace(cfg=None),
            initialize=lambda: 0,
            step=lambda obs: steps.append(obs) or (obs + 1),
            num_steps=None,
            headless=False,
            record_video=False,
        )
        assert result is None
        assert steps == [0]
    finally:
        backend.close()


def test_explicit_interactive_without_display_fails_closed(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    backend = _make_backend(scene_file, render_mode="interactive")
    try:
        with pytest.raises(IsaacSimRenderError, match="no local display"):
            backend.materialize()
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("behavior", "message"),
    [
        ("render_meta_missing", "missing render startup fields"),
        ("render_meta_mode_mismatch", "render_mode does not match"),
        ("render_meta_size_mismatch", "render dimensions do not match"),
        ("render_meta_graphics_mismatch", "graphics_enabled does not match"),
    ],
)
def test_render_startup_metadata_mismatch_fails_closed(
    scene_file: str,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    message: str,
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", behavior)
    backend = _make_backend(scene_file, render_mode="record", render_width=64, render_height=48)
    try:
        with pytest.raises(IsaacSimWorkerError, match=message):
            backend.materialize()
    finally:
        backend.close()


def test_isaacsim_worker_error_is_not_isaacgym_error_type(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "fail_init")
    backend = _make_backend(scene_file)
    try:
        with pytest.raises(IsaacSimWorkerError) as caught:
            backend.materialize()
        assert type(caught.value) is IsaacSimWorkerError
        assert isinstance(caught.value, IsaacGymWorkerError)
    finally:
        backend.close()


def test_isaacsim_worker_timeout_has_backend_diagnostic(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "hang_on_step")
    backend = _make_backend(scene_file, worker_timeout_s=0.2)
    backend.materialize()
    try:
        with pytest.raises(IsaacSimWorkerError, match="isaacsim worker did not answer STEP"):
            backend.step(np.zeros((NUM_ENVS, 3), dtype=np.float32))
        assert backend._proc is not None and backend._proc.poll() is not None
    finally:
        backend.close()


def test_root_angular_velocity_helper_converts_body_to_world() -> None:
    half = np.sqrt(0.5)
    quat = np.array([[half, half, 0.0, 0.0]], dtype=np.float32)
    body_angvel = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(_quat_rotate_wxyz(quat, body_angvel), [[0.0, 0.0, -1.0]], atol=1e-6)


class _FakePrim:
    def __init__(self, path: str, *, root_api: bool = False, valid: bool = True) -> None:
        self.path = path
        self.root_api = root_api
        self.valid = valid

    def GetPath(self) -> str:
        return self.path

    def IsValid(self) -> bool:
        return self.valid

    def HasAPI(self, api: object) -> bool:
        del api
        return self.root_api


class _FakeStage:
    def __init__(self, default_path: str, prims: list[_FakePrim]) -> None:
        self.default = _FakePrim(default_path)
        self.prims = prims

    def GetDefaultPrim(self) -> _FakePrim:
        return self.default

    def Traverse(self) -> list[_FakePrim]:
        return self.prims


def _install_fake_pxr(monkeypatch: pytest.MonkeyPatch, stage: _FakeStage) -> None:
    class _Stage:
        @staticmethod
        def Open(path: str) -> _FakeStage:
            del path
            return stage

    usd = types.SimpleNamespace(Stage=_Stage)
    usd_physics = types.SimpleNamespace(ArticulationRootAPI=object())
    pxr = types.ModuleType("pxr")
    pxr.Usd = usd  # type: ignore[attr-defined]
    pxr.UsdPhysics = usd_physics  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pxr", pxr)


def test_resolve_articulation_root_prim_path_discovers_unique_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _FakeStage(
        "/Asset",
        [
            _FakePrim("/Asset", root_api=False),
            _FakePrim("/Asset/pelvis", root_api=True),
            _FakePrim("/Asset/pelvis/mesh", root_api=False),
        ],
    )
    _install_fake_pxr(monkeypatch, stage)
    assert _resolve_articulation_root_prim_path("asset.usd", "pelvis") == "/pelvis"


@pytest.mark.parametrize(
    "prims",
    [
        [_FakePrim("/Asset/other", root_api=True)],
        [
            _FakePrim("/Asset/pelvis/a", root_api=True),
            _FakePrim("/Asset/pelvis/b", root_api=True),
        ],
    ],
)
def test_resolve_articulation_root_prim_path_fails_on_missing_or_ambiguous(
    monkeypatch: pytest.MonkeyPatch, prims: list[_FakePrim]
) -> None:
    _install_fake_pxr(monkeypatch, _FakeStage("/Asset", prims))
    with pytest.raises(RuntimeError, match="expected one ArticulationRootAPI"):
        _resolve_articulation_root_prim_path("asset.usd", "pelvis")
