from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mjwarp playback validation requires mujoco")

import mujoco
from omegaconf import OmegaConf

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.base import BackendPlayCapabilities
from unilab.base.backend.mjwarp.backend import MjwarpBackend
from unilab.base.backend.mjwarp.playback import (
    run_mjwarp_playback,
    validate_mjwarp_visual_model,
)
from unilab.base.scene import SceneCfg

G1_SCENE = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _stub_backend(model_file: Path = G1_SCENE, *, num_envs: int = 2) -> MjwarpBackend:
    backend = object.__new__(MjwarpBackend)
    model = mujoco.MjModel.from_xml_path(str(model_file))
    backend._mujoco = mujoco
    backend._cpu_model = model
    backend._num_envs = num_envs
    backend._nq = int(model.nq)
    backend._nv = int(model.nv)
    backend._qpos_cache = np.broadcast_to(
        np.asarray(model.qpos0, dtype=np.float32), (num_envs, int(model.nq))
    ).copy()
    backend._qvel_cache = np.zeros((num_envs, int(model.nv)), dtype=np.float32)
    backend._time_cache = np.zeros((num_envs,), dtype=np.float32)
    backend.scene_visual_model_file = str(model_file)
    backend._playback_model_validated = False
    backend._scene_cleanup_handle = None
    return backend


def test_mjwarp_plan_is_explicit_and_finite() -> None:
    backend = _stub_backend(num_envs=1)
    assert backend.get_play_capabilities() == BackendPlayCapabilities(
        supports_physics_state_playback=True
    )
    assert (
        backend.resolve_play_render_plan(
            play_render_mode="record", play_steps=3, output_video="play.mp4"
        ).num_steps
        == 3
    )
    assert (
        backend.resolve_play_render_plan(
            play_render_mode="none", play_steps=None, output_video=None
        ).mode
        == "none"
    )
    with pytest.raises(NotImplementedError, match="auto mode"):
        backend.resolve_play_render_plan(
            play_render_mode="auto", play_steps=3, output_video="play.mp4"
        )
    with pytest.raises(NotImplementedError, match="interactive"):
        backend.resolve_play_render_plan(
            play_render_mode="interactive", play_steps=3, output_video="play.mp4"
        )
    with pytest.raises(ValueError, match="positive finite"):
        backend.resolve_play_render_plan(
            play_render_mode="record", play_steps=0, output_video="play.mp4"
        )


def test_g1_mjwarp_owner_enables_explicit_record_profile() -> None:
    owner = OmegaConf.load(REPO_ROOT / "conf" / "ppo" / "task" / "g1_walk_flat" / "mjwarp.yaml")

    assert OmegaConf.select(owner, "training.no_play") is None
    assert owner.training.play_render_mode == "record"
    assert owner.play_profile.enabled is True
    assert owner.play_profile.env.render_spacing == 2.0


def test_mjwarp_snapshot_is_detached_time_qpos_qvel_layout() -> None:
    backend = _stub_backend(num_envs=2)
    backend._time_cache[:] = [0.12, 0.34]
    backend._qpos_cache[:, 0] = [1.0, 2.0]
    backend._qvel_cache[:, 0] = [3.0, 4.0]

    snapshot = backend.get_physics_state()

    assert snapshot.shape == (2, 1 + backend._nq + backend._nv)
    np.testing.assert_allclose(snapshot[:, 0], [0.12, 0.34])
    np.testing.assert_allclose(snapshot[:, 1 : 1 + backend._nq], backend._qpos_cache)
    np.testing.assert_allclose(snapshot[:, 1 + backend._nq :], backend._qvel_cache)
    snapshot.fill(99.0)
    assert backend._time_cache[0] == pytest.approx(0.12)
    assert backend._qpos_cache[0, 0] == pytest.approx(1.0)


def test_g1_visual_model_and_replay_forward_camera_and_spacing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = _stub_backend()
    env = SimpleNamespace(
        cfg=SimpleNamespace(ctrl_dt=0.02, scene=SceneCfg(model_file=str(G1_SCENE))),
        get_playback_model=backend.get_playback_model,
        get_physics_state_snapshot=backend.get_physics_state,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr("unilab.visualization.render_many.render_backend_usable", lambda: True)
    monkeypatch.setattr(
        "unilab.visualization.render_many.render_states_get_frames",
        lambda states, model_file, **kwargs: (
            captured.update(states=states, model_file=model_file, render_kwargs=kwargs)
            or [np.zeros((2, 2, 3), dtype=np.uint8)]
        ),
    )
    monkeypatch.setattr(
        "unilab.base.backend.playback_common.imageio.mimsave",
        lambda path, frames, fps: captured.update(video_path=str(path), fps=fps),
    )

    output = tmp_path / "g1.mp4"
    result = run_mjwarp_playback(
        backend=backend,
        env=env,
        initialize=lambda: 0,
        step=lambda obs: obs,
        num_steps=1,
        output_video=output,
        render_spacing=2.0,
        headless=True,
        record_video=True,
        snapshot_shape=(2, 1 + backend._nq + backend._nv),
        frame_state_getter=None,
        camera_kwargs={"cam_distance": 4.0, "cam_tracking": False},
    )

    assert result == str(output)
    assert captured["model_file"] == str(G1_SCENE)
    assert captured["states"][0].shape == (  # type: ignore[index]
        2,
        1 + backend._nq + backend._nv,
    )
    render_kwargs = captured["render_kwargs"]
    assert render_kwargs["render_spacing"] == 2.0  # type: ignore[index]
    assert render_kwargs["cam_distance"] == 4.0  # type: ignore[index]


def test_mjwarp_playback_rejects_bad_snapshot_and_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = _stub_backend(num_envs=1)
    env = SimpleNamespace(get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32))
    monkeypatch.setattr("unilab.visualization.render_many.render_backend_usable", lambda: True)
    with pytest.raises(ValueError, match=r"\[time, qpos, qvel\]"):
        run_mjwarp_playback(
            backend=backend,
            env=env,
            initialize=lambda: 0,
            step=lambda obs: obs,
            num_steps=1,
            output_video=tmp_path / "bad.mp4",
            render_spacing=1.0,
            headless=True,
            record_video=True,
            snapshot_shape=(1, 1 + backend._nq + backend._nv),
            frame_state_getter=None,
            camera_kwargs=None,
        )

    monkeypatch.setattr("unilab.visualization.render_many.render_backend_usable", lambda: False)
    with pytest.raises(RuntimeError, match="usable MuJoCo off-screen renderer"):
        run_mjwarp_playback(
            backend=backend,
            env=SimpleNamespace(get_physics_state_snapshot=backend.get_physics_state),
            initialize=lambda: 0,
            step=lambda obs: obs,
            num_steps=1,
            output_video=tmp_path / "unavailable.mp4",
            render_spacing=1.0,
            headless=True,
            record_video=True,
            snapshot_shape=(1, 1 + backend._nq + backend._nv),
            frame_state_getter=None,
            camera_kwargs=None,
        )


def test_mjwarp_visual_model_diagnostics(tmp_path: Path) -> None:
    physics = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint name='root'/><geom size='.1'/></body>"
        "</worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="does not exist or is not a file"):
        validate_mjwarp_visual_model(
            mujoco=mujoco, physics_model=physics, model_file=tmp_path / "missing.xml"
        )

    incompatible = tmp_path / "incompatible.xml"
    incompatible.write_text(
        "<mujoco><worldbody><body><joint name='root' type='hinge'/><geom size='.1'/></body>"
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="state dimensions are incompatible"):
        validate_mjwarp_visual_model(mujoco=mujoco, physics_model=physics, model_file=incompatible)

    mismatched = tmp_path / "mismatched.xml"
    mismatched.write_text(
        "<mujoco><worldbody><body><freejoint name='other'/><geom size='.1'/></body>"
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="joint layout is incompatible"):
        validate_mjwarp_visual_model(mujoco=mujoco, physics_model=physics, model_file=mismatched)
