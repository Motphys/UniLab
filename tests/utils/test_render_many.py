"""Tests for MuJoCo GL backend resolution in unilab.visualization.render_many."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="GitHub Actions runners do not provide stable EGL/GLFW rendering backends.",
)


def _reload_render_many(monkeypatch):
    monkeypatch.setitem(sys.modules, "mujoco", types.SimpleNamespace())
    sys.modules.pop("unilab.visualization.render_many", None)
    return importlib.import_module("unilab.visualization.render_many")


def test_resolve_gl_backend_uses_egl_when_probe_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: True)

    assert render_many._resolve_gl_backend() == "egl"


def test_resolve_gl_backend_uses_osmesa_when_headless_and_egl_unavailable(monkeypatch) -> None:
    # Headless host (no DISPLAY): glfw cannot work, so software rendering wins.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "osmesa"


def test_resolve_gl_backend_uses_glfw_when_display_present_and_egl_unavailable(monkeypatch) -> None:
    # A display is available: glfw can create an off-screen context.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "glfw"


def test_resolve_gl_backend_preserves_explicit_safe_value(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "osmesa"


def test_resolve_gl_backend_uses_glfw_on_windows_without_display(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "glfw"


def test_resolve_gl_backend_rejects_linux_only_backend_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    render_many = _reload_render_many(monkeypatch)

    assert render_many._resolve_gl_backend() == "glfw"


def test_egl_runtime_usable_sets_default_device_id(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)

    def _fake_run(cmd, env, check, stdout, stderr, timeout):
        assert cmd[0] == sys.executable
        assert env["MUJOCO_GL"] == "egl"
        assert env["MUJOCO_EGL_DEVICE_ID"] == "0"
        assert check is True
        assert stdout is subprocess.DEVNULL
        assert stderr is subprocess.DEVNULL
        assert timeout == 10
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(render_many.subprocess, "run", _fake_run)

    assert render_many._egl_runtime_usable() is True
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"


def test_egl_runtime_usable_returns_false_on_probe_failure(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    def _fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(render_many.subprocess, "run", _fake_run)

    assert render_many._egl_runtime_usable() is False


def _reload_render_many_with_geom_enums(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        types.SimpleNamespace(
            mjtGeom=types.SimpleNamespace(mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_BOX=6),
        ),
    )
    sys.modules.pop("unilab.visualization.render_many", None)
    return importlib.import_module("unilab.visualization.render_many")


def test_replicable_terrain_geom_indices_selects_worldbody_box(monkeypatch) -> None:
    # The x2 wall-flip render twin declares the wall as a group-0 worldbody box
    # geom precisely so this selector picks it up and the grid renderer
    # replicates one wall per env cell. Lock that contract in.
    render_many = _reload_render_many_with_geom_enums(monkeypatch)

    model = types.SimpleNamespace(
        ngeom=4,
        # 0: floor plane (worldbody)  1: robot geom (body 5)
        # 2: wall box (worldbody)     3: group-2 worldbody box (non-default group)
        geom_group=np.array([0, 0, 0, 2], dtype=np.int32),
        geom_bodyid=np.array([0, 5, 0, 0], dtype=np.int32),
        geom_type=np.array([0, 6, 6, 6], dtype=np.int32),
    )

    indices = render_many._replicable_terrain_geom_indices(model)

    # Only the worldbody box wall (geom 2) is replicable: the plane is skipped,
    # the body-attached robot geom is skipped, and the non-group-0 geom is skipped.
    assert indices.tolist() == [2]


def test_offset_freejoint_object_qpos_handles_arbitrary_object_body(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    model = types.SimpleNamespace(
        nbody=4,
        body_jntadr=np.array([-1, 0, 1, -1], dtype=np.int32),
        body_jntnum=np.array([0, 1, 1, 0], dtype=np.int32),
        jnt_type=np.array([0, 0], dtype=np.int32),
        jnt_qposadr=np.array([0, 7], dtype=np.int32),
    )
    data = types.SimpleNamespace(qpos=np.zeros((14,), dtype=np.float32))

    shifted = render_many._offset_freejoint_object_qpos(
        model, data, np.array([1.5, -2.0], dtype=np.float32)
    )

    assert shifted == {2}
    assert data.qpos[0] == pytest.approx(0.0)
    assert data.qpos[1] == pytest.approx(0.0)
    assert data.qpos[7] == pytest.approx(1.5)
    assert data.qpos[8] == pytest.approx(-2.0)


def test_render_backend_usable_reflects_resolved_backend(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    seen: dict[str, str] = {}

    def _fake_probe(backend: str) -> bool:
        seen["backend"] = backend
        return backend == "egl"

    monkeypatch.setattr(render_many, "_gl_backend_runtime_usable", _fake_probe)

    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert render_many.render_backend_usable() is True
    assert seen["backend"] == "egl"

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    assert render_many.render_backend_usable() is False


def test_render_states_get_frames_skips_when_backend_unusable(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: False)

    frames = render_many.render_states_get_frames(
        [np.zeros((1, 8), dtype=np.float32)],
        "/no/such/model.xml",
        num_processes=4,
    )

    assert frames == []


def test_render_states_get_frames_tracking_skips_when_backend_unusable(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: False)

    frames = render_many.render_states_get_frames_tracking(
        [np.zeros((1, 8), dtype=np.float32)],
        "/no/such/model.xml",
    )

    assert frames == []


def test_render_states_get_frames_fails_fast_on_worker_init_error(monkeypatch) -> None:
    """A failing pool initializer must NOT respawn workers forever (issue #605).

    ProcessPoolExecutor raises BrokenProcessPool quickly instead of hanging, and
    render_states_get_frames degrades to an empty result + warning.
    """
    # Skip the EGL probe in spawned workers (they inherit MUJOCO_GL via os.environ).
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    render_many = _reload_render_many(monkeypatch)
    # Bypass the parent pre-flight so we exercise the pool's fail-fast path.
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: True)

    frames = render_many.render_states_get_frames(
        [np.zeros((1, 8), dtype=np.float32)],
        "/nonexistent/model/path.xml",  # init_worker raises while loading this
        num_processes=2,
    )

    assert frames == []
