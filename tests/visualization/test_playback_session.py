"""Tests for the embeddable snapshot playback session and camera config surface."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from unisim.backend.base import CameraCfg, DebugPrimitive

from unilab.base.base import EnvPlayCapabilities
from unilab.visualization.playback import camera_cfg_from_training, render_play_mode
from unilab.visualization.playback_session import SnapshotPlaybackSession


class _FakeEnv:
    def __init__(
        self,
        *,
        num_envs: int = 2,
        capabilities: EnvPlayCapabilities | None = None,
        ctrl_dt: float = 0.02,
        render_spacing: float = 1.5,
    ):
        self.num_envs = num_envs
        self.play_capabilities = capabilities or EnvPlayCapabilities(
            supports_physics_state_playback=True,
            supports_debug_overlay=True,
        )
        self.cfg = SimpleNamespace(ctrl_dt=ctrl_dt, render_spacing=render_spacing)
        self._snapshot_count = 0

    def get_physics_state_snapshot(self) -> np.ndarray:
        self._snapshot_count += 1
        return np.full((self.num_envs, 4), float(self._snapshot_count), dtype=np.float32)


def _patch_render_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Swap the mujoco-dependent render pipeline for fakes; returns call log."""
    log: dict[str, Any] = {}

    playback_mod = types.ModuleType("unisim.backend.mujoco.playback")

    def _resolve_render_play_model_files(env: Any, *, num_envs: int, tmp_dir: str) -> str:
        log["resolve_model_files"] = {"num_envs": num_envs}
        return "fake_model.xml"

    playback_mod.resolve_render_play_model_files = _resolve_render_play_model_files  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unisim.backend.mujoco.playback", playback_mod)

    render_many_mod = types.ModuleType("unisim.visualization.render_many")

    def _render_states_get_frames(state_list, model_path, **kwargs):
        log["render_states_get_frames"] = {"state_list": list(state_list), "kwargs": kwargs}
        return [np.zeros((2, 4, 3), dtype=np.uint8) for _ in state_list]

    def _render_states_get_frames_tracking(state_list, model_path, **kwargs):
        log["render_states_get_frames_tracking"] = {
            "state_list": list(state_list),
            "kwargs": kwargs,
        }
        return [np.ones((2, 4, 3), dtype=np.uint8) for _ in state_list]

    render_many_mod.render_states_get_frames = _render_states_get_frames  # type: ignore[attr-defined]
    render_many_mod.render_states_get_frames_tracking = _render_states_get_frames_tracking  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unisim.visualization.render_many", render_many_mod)
    visualization_pkg = sys.modules.get("unisim.visualization")
    if visualization_pkg is not None:
        monkeypatch.setattr(visualization_pkg, "render_many", render_many_mod, raising=False)

    from unisim.backend import playback_common

    def _write_playback_video(path: str, frames: list[np.ndarray], *, fps: int) -> None:
        log["write_playback_video"] = {"path": path, "frames": list(frames), "fps": fps}

    monkeypatch.setattr(playback_common, "write_playback_video", _write_playback_video)
    return log


def test_session_requires_physics_state_playback() -> None:
    env = _FakeEnv(capabilities=EnvPlayCapabilities())
    with pytest.raises(NotImplementedError, match="_FakeEnv"):
        SnapshotPlaybackSession(env)


def test_session_overlay_getter_requires_debug_overlay_support() -> None:
    env = _FakeEnv(capabilities=EnvPlayCapabilities(supports_physics_state_playback=True))
    with pytest.raises(NotImplementedError, match="supports_debug_overlay"):
        SnapshotPlaybackSession(env, overlay_getter=lambda: None)


def test_snapshot_caches_detached_states_and_clear_resets() -> None:
    env = _FakeEnv()
    session = SnapshotPlaybackSession(env)
    first = session.snapshot()
    first[:] = -1.0  # mutating the returned array must not corrupt the cache
    session.snapshot()
    assert len(session) == 2
    assert session.snapshots[0][0, 0] == pytest.approx(1.0)
    assert session.snapshots[1][0, 0] == pytest.approx(2.0)
    session.clear()
    assert len(session) == 0


def test_render_snapshots_requires_cached_states() -> None:
    session = SnapshotPlaybackSession(_FakeEnv())
    with pytest.raises(ValueError, match="no cached snapshots"):
        session.render_snapshots(output_video="out.mp4")


def test_render_snapshots_writes_video_with_camera_and_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _patch_render_pipeline(monkeypatch)
    session = SnapshotPlaybackSession(_FakeEnv(ctrl_dt=0.05))
    session.snapshot()
    session.snapshot()

    out = session.render_snapshots(
        output_video="trial.mp4",
        camera={"cam_distance": 3.5, "cam_azimuth": 45.0},
    )
    assert out == "trial.mp4"
    render_kwargs = log["render_states_get_frames"]["kwargs"]
    assert render_kwargs["cam_distance"] == pytest.approx(3.5)
    assert render_kwargs["cam_azimuth"] == pytest.approx(45.0)
    assert render_kwargs["render_spacing"] == pytest.approx(1.5)
    assert len(log["render_states_get_frames"]["state_list"]) == 2
    assert log["write_playback_video"]["fps"] == 20  # 1 / ctrl_dt


def test_render_snapshots_tracking_camera_uses_tracking_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _patch_render_pipeline(monkeypatch)
    session = SnapshotPlaybackSession(_FakeEnv())
    session.snapshot()
    session.render_snapshots(
        output_video="tracked.mp4",
        camera=CameraCfg(cam_tracking=True, cam_tracking_env_idx=1),
    )
    assert log["render_states_get_frames_tracking"]["kwargs"]["tracking_env_idx"] == 1


def test_render_snapshots_applies_on_frame_and_overlay_getter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _patch_render_pipeline(monkeypatch)
    env = _FakeEnv(num_envs=2)
    session = SnapshotPlaybackSession(env)
    session.snapshot()
    session.snapshot()

    primitive = DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 1.0), size=(0.05,))
    overlay_calls = []

    def _overlay_getter():
        overlay_calls.append(1)
        return [[primitive], None]

    def _on_frame(index: int, frame: np.ndarray) -> np.ndarray | None:
        if index == 0:
            return None
        return np.full_like(frame, 7)

    session.render_snapshots(
        output_video="overlay.mp4",
        overlay_getter=_overlay_getter,
        on_frame=_on_frame,
    )
    assert len(overlay_calls) == 2
    overlays = log["render_states_get_frames"]["kwargs"]["debug_overlays_list"]
    assert overlays[0][0] == [primitive]
    assert overlays[1][0] == [primitive]
    written = log["write_playback_video"]["frames"]
    assert written[0].sum() == 0  # frame 0 kept (on_frame returned None)
    assert np.all(written[1] == 7)


def test_render_snapshots_validates_overlay_env_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render_pipeline(monkeypatch)
    session = SnapshotPlaybackSession(_FakeEnv(num_envs=2))
    session.snapshot()
    with pytest.raises(ValueError, match="one entry per env"):
        session.render_snapshots(
            output_video="bad.mp4",
            overlay_getter=lambda: [[DebugPrimitive(kind="sphere", pos=(0, 0, 1), size=(0.1,))]],
        )


def test_render_snapshots_rejects_unknown_camera_keys() -> None:
    session = SnapshotPlaybackSession(_FakeEnv())
    session.snapshot()
    with pytest.raises(ValueError, match="unknown camera_kwargs"):
        session.render_snapshots(output_video="x.mp4", camera={"elevation_deg": -20.0})


def test_camera_cfg_from_training_normalizes_hydra_fields() -> None:
    training = SimpleNamespace(
        cam_distance=2.5,
        cam_elevation=-30.0,
        cam_azimuth=120.0,
        cam_lookat=[0.0, 0.0, 0.5],
        cam_tracking=True,
        cam_tracking_env_idx=1,
        cam_tracking_extra_envs=3,
    )
    camera = camera_cfg_from_training(training)
    assert isinstance(camera, CameraCfg)
    assert camera.cam_distance == pytest.approx(2.5)
    assert camera.cam_lookat == (0.0, 0.0, 0.5)
    assert camera.cam_tracking is True
    assert camera.cam_tracking_env_idx == 1
    assert camera.cam_tracking_extra_envs == 3
    assert camera.cam_fov is None


def test_render_play_mode_forwards_overlay_and_on_frame() -> None:
    captured: dict[str, Any] = {}

    class _Env:
        def run_playback(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "out.mp4"

    getter = lambda: None  # noqa: E731
    on_frame = lambda i, frame: frame  # noqa: E731
    result = render_play_mode(
        _Env(),
        sim_backend="mujoco",
        initialize=lambda: None,
        step=lambda obs: obs,
        num_steps=1,
        debug_overlay_getter=getter,
        on_frame=on_frame,
    )
    assert result == "out.mp4"
    assert captured["debug_overlay_getter"] is getter
    assert captured["on_frame"] is on_frame
