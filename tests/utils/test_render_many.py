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


def test_resolve_gl_backend_falls_back_to_glfw_when_probe_fails(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "glfw"


def test_resolve_gl_backend_preserves_explicit_safe_value(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "osmesa"


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
