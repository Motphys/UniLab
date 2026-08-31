"""Fail-closed public capability matrix for the ``mjwarp`` host profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unilab.base.backend import create_backend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg
from unilab.dr.types import IntervalRandomizationPlan, ResetRandomizationPayload

pytestmark = pytest.mark.slow


def _backend() -> Any:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp capability tests require an active CUDA Warp device")

    from unilab.assets import ASSETS_ROOT_PATH

    scene = SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))
    return create_backend("mjwarp", scene, 1, 0.02 / 3.0, base_name="pelvis")


def test_unsupported_matrix_fails_before_step() -> None:
    """Every currently unadvertised public path errors before a physics step."""
    backend = _backend()
    with pytest.raises(NotImplementedError, match="host pre-step callbacks"):
        backend.set_pre_step_control(lambda _backend, control: control)
    with pytest.raises(NotImplementedError, match="body positions"):
        backend.get_body_pos_w(np.asarray([1], dtype=np.int32))
    with pytest.raises(NotImplementedError, match="height-field scanners"):
        backend.create_hfield_scanner(
            hfield_geom_id=0,
            offsets=np.zeros((1, 2), dtype=np.float32),
            frame_body_id=1,
        )
    assert backend.get_play_capabilities().supports_physics_state_playback
    assert not backend.get_play_capabilities().supports_native_interactive_renderer
    assert not backend.get_play_capabilities().supports_native_video_capture
    assert Path(backend.get_playback_model()).is_file()
    # Per-world gravity DR stays out of scope for the mjwarp host profile.
    capabilities = backend.get_dr_capabilities()
    assert not capabilities.supports_reset_term("gravity")
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (1, 1))
    qvel = np.zeros((1, backend.get_init_qvel().size), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="reset domain randomization"):
        backend.set_state(
            np.asarray([0], dtype=np.int32),
            qpos,
            qvel,
            randomization=ResetRandomizationPayload(gravity=np.zeros((1, 3), dtype=np.float32)),
        )


def test_interval_push_and_velocity_require_named_bodies() -> None:
    """Without base/push body names the interval capabilities stay fail-closed."""
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp capability tests require an active CUDA Warp device")

    from unilab.assets import ASSETS_ROOT_PATH

    scene = SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))
    backend = create_backend("mjwarp", scene, 1, 0.02 / 3.0)
    capabilities = backend.get_dr_capabilities()
    assert not capabilities.supports_interval_push
    assert not capabilities.supports_interval_body_velocity_delta
    with pytest.raises(NotImplementedError, match="push target body"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(push_perturbation_limit=np.ones((3,), dtype=np.float32))
        )
    with pytest.raises(NotImplementedError, match="exactly one free joint"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                body_ids=np.asarray([1], dtype=np.int32),
                body_linear_velocity_delta=np.zeros((1, 1, 3), dtype=np.float32),
            )
        )
    with pytest.raises(ValueError, match="Push body 'missing' not found"):
        create_backend("mjwarp", scene, 1, 0.02 / 3.0, push_body_name="missing")
