"""Render-only visual model override for the MuJoCo scene context.

These cover the cold-path wiring that lets offline playback render a visual twin
of a scene (e.g. a per-env replicable wall) without touching the trained physics
model. See ``SceneCfg.visual_model_file`` and ``_build_mujoco_scene_context``.
"""

from __future__ import annotations

from pathlib import Path

from unilab.base.backend.mujoco.backend import _build_mujoco_scene_context
from unilab.base.scene import SceneCfg


def test_visual_model_file_defaults_to_model_file() -> None:
    ctx = _build_mujoco_scene_context(SceneCfg(model_file="/tmp/phys.xml"))

    assert ctx.model_file == "/tmp/phys.xml"
    assert ctx.model_source == "/tmp/phys.xml"
    # Unset override => renderer uses the physics model (behaviour unchanged for
    # every task that does not opt in).
    assert ctx.visual_model_file == "/tmp/phys.xml"


def test_visual_model_file_override_is_render_only() -> None:
    ctx = _build_mujoco_scene_context(
        SceneCfg(model_file="/tmp/phys.xml", visual_model_file="/tmp/visual.xml")
    )

    # Physics keeps using model_file; only the render model is swapped.
    assert ctx.model_file == "/tmp/phys.xml"
    assert ctx.model_source == "/tmp/phys.xml"
    assert ctx.visual_model_file == "/tmp/visual.xml"


def test_x2_wall_flip_wires_render_only_visual_twin() -> None:
    from unilab.envs.motion_tracking.x2.flip_tracking import X2WallFlipTrackingEnvCfg

    cfg = X2WallFlipTrackingEnvCfg()

    # Physics = trained <body> wall; render = worldbody-geom twin.
    assert cfg.scene.model_file.endswith("scene_flat_with_wall.xml")
    assert cfg.scene.visual_model_file is not None
    assert cfg.scene.visual_model_file.endswith("scene_flat_with_wall_visual.xml")
    assert cfg.scene.model_file != cfg.scene.visual_model_file
    assert Path(cfg.scene.model_file).is_file()
    assert Path(cfg.scene.visual_model_file).is_file()
