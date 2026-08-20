"""Render the packaged MotrixSim teaser scene.

This is a visual-only entrypoint for the paper-cover teaser asset. It loads the
MJCF scene once, creates static scene data, and keeps the Motrix renderer alive
without advancing simulation.

Scene assets are downloaded automatically from HuggingFace on first run.
"""

from __future__ import annotations

import time
from typing import Any

from unilab.assets import ASSETS_ROOT_PATH

TEASER_DIR = ASSETS_ROOT_PATH / "scenes" / "teaser"
DEFAULT_SCENE = TEASER_DIR / "teaser.xml"
TEASER_SYSTEM_CAMERA_LOOKAT = [-1.0e-7, -0.0035363, 0.75]
TEASER_SYSTEM_CAMERA_DISTANCE = 5.6876
TEASER_SYSTEM_CAMERA_ELEVATION = -30.0
TEASER_SYSTEM_CAMERA_AZIMUTH = 90.0
TEASER_RENDER_FPS = 60.0
TEASER_LOG_LEVEL = "WARN"


def _render_settings() -> Any:
    from motrixsim.render import RenderSettings

    settings = RenderSettings.quality()
    settings.enable_shadow = True
    settings.enable_ssgi = True
    settings.simplify_render_mesh = False
    return settings


def _set_teaser_system_camera_view(render: Any) -> None:
    render.set_main_camera(None)
    render.system_camera.set_view(
        TEASER_SYSTEM_CAMERA_LOOKAT,
        TEASER_SYSTEM_CAMERA_DISTANCE,
        TEASER_SYSTEM_CAMERA_ELEVATION,
        TEASER_SYSTEM_CAMERA_AZIMUTH,
    )


def _load_teaser_model() -> tuple[Any, Any]:
    import motrixsim as mtx

    from unilab.assets.hub import resolve_scene_dir

    resolve_scene_dir("scenes/teaser")

    if not DEFAULT_SCENE.is_file():
        raise FileNotFoundError(f"Teaser scene file does not exist: {DEFAULT_SCENE}")
    model = mtx.load_model(str(DEFAULT_SCENE))
    data = mtx.SceneData(model)
    model.forward_kinematic(data)
    return model, data


def _run() -> None:
    from motrixsim.render import RenderApp

    model, data = _load_teaser_model()
    frame_dt = 1.0 / TEASER_RENDER_FPS

    with RenderApp(TEASER_LOG_LEVEL) as render:
        settings = _render_settings()
        render.launch(model, render_settings=settings)
        _set_teaser_system_camera_view(render)

        try:
            while not render.is_closed:
                render.sync(data)
                time.sleep(frame_dt)
        except KeyboardInterrupt:
            return


def main() -> None:
    _run()


if __name__ == "__main__":
    main()
