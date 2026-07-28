"""Fail-closed public capability matrix for the ``mjwarp`` host profile."""

from __future__ import annotations

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
    with pytest.raises(NotImplementedError, match="playback model export"):
        backend.get_playback_model()
    with pytest.raises(NotImplementedError, match="interval randomization"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(push_perturbation_limit=np.ones((3,), dtype=np.float32))
        )
    with pytest.raises(NotImplementedError, match="typed backend batches"):
        backend.bind_task_io(None)  # type: ignore[arg-type]

    qpos = np.tile(backend.get_keyframe_qpos("stand"), (1, 1))
    qvel = np.zeros((1, backend.get_init_qvel().size), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="reset domain randomization"):
        backend.set_state(
            np.asarray([0], dtype=np.int32),
            qpos,
            qvel,
            randomization=ResetRandomizationPayload(kp=np.ones((1, backend.num_actuators))),
        )
