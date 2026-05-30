"""Tests for the teaser scene renderer (``unilab.tools.render_teaser``)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from unilab.tools import render_teaser


def test_set_teaser_system_camera_view_uses_packaged_view():
    class FakeSystemCamera:
        def __init__(self):
            self.view = None

        def set_view(self, lookat, distance, elevation, azimuth):
            self.view = {
                "lookat": lookat,
                "distance": distance,
                "elevation": elevation,
                "azimuth": azimuth,
            }

    class FakeRender:
        def __init__(self):
            self.main_camera = "unset"
            self.system_camera = FakeSystemCamera()

        def set_main_camera(self, camera):
            self.main_camera = camera

    render = FakeRender()
    render_teaser._set_teaser_system_camera_view(render)

    assert render.main_camera is None
    assert render.system_camera.view == {
        "lookat": render_teaser.TEASER_SYSTEM_CAMERA_LOOKAT,
        "distance": render_teaser.TEASER_SYSTEM_CAMERA_DISTANCE,
        "elevation": render_teaser.TEASER_SYSTEM_CAMERA_ELEVATION,
        "azimuth": render_teaser.TEASER_SYSTEM_CAMERA_AZIMUTH,
    }


def test_render_teaser_has_no_cli_parameters():
    assert list(inspect.signature(render_teaser.main).parameters) == []


def test_load_teaser_model_calls_resolve_scene_dir():
    """_load_teaser_model must call resolve_scene_dir before loading."""
    mock_resolve = MagicMock(return_value=render_teaser.TEASER_DIR)

    fake_model = MagicMock()
    fake_data = MagicMock()
    fake_mtx = MagicMock()
    fake_mtx.load_model.return_value = fake_model
    fake_mtx.SceneData.return_value = fake_data

    with (
        patch("unilab.assets.hub.resolve_scene_dir", mock_resolve),
        patch.dict("sys.modules", {"motrixsim": fake_mtx}),
        patch.object(
            render_teaser, "DEFAULT_SCENE", MagicMock(is_file=MagicMock(return_value=True))
        ),
    ):
        render_teaser._load_teaser_model()

    mock_resolve.assert_called_once_with("scenes/teaser")
