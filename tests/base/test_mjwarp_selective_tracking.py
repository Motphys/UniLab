"""Real-CUDA MJWarp acceptance for UniLab's selective tracked-body routing."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from unisim.backend.base import SimBackend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityInitialState, SceneEntitySpec

from unilab.base.backend_factory import create_backend
from unilab.base.entity import EntityCfg, EntityScene, SceneBodyPairContactReadSpec
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow


def _require_cuda_mjwarp() -> None:
    if sys.platform == "darwin":
        pytest.skip("mjwarp is a CUDA-only backend; macOS has no supported CUDA runtime")
    try:
        from unisim.backend.mjwarp.dependencies import (
            MjwarpDependencyError,
            load_mjwarp_dependencies,
        )
    except ImportError as exc:
        pytest.skip(f"mjwarp runtime is unavailable: {exc}")
    try:
        dependencies = load_mjwarp_dependencies()
    except MjwarpDependencyError as exc:
        pytest.skip(str(exc))
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.skip("MJWarp selective tracking acceptance requires an active CUDA Warp device")


def _source(tmp_path: Path, name: str) -> ModelSourceDescriptor:
    path = tmp_path / f"{name}.xml"
    path.write_text(
        '<mujoco><option gravity="0 0 0" integrator="Euler"/>'
        '<worldbody><body name="base"><freejoint name="root"/>'
        '<inertial pos=".03 .01 .02" mass="1" diaginertia=".1 .2 .3"/>'
        '<geom name="base_geom" size=".05" contype="0" conaffinity="0"/>'
        '<body name="hand" pos="0 0 .3"><joint name="hinge" axis="0 1 0"/>'
        '<geom name="hand_geom" size=".03" mass=".2" contype="0" conaffinity="0"/>'
        "</body></body></worldbody>"
        '<actuator><general name="drive" joint="hinge" dyntype="filter" dynprm=".1"/>'
        "</actuator></mujoco>",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _scene(tmp_path: Path) -> SceneCfg:
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "left",
                _source(tmp_path, "left"),
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.0)),
            ),
            SceneEntitySpec(
                "right",
                _source(tmp_path, "right"),
                initial_state=EntityInitialState(position=(1.0, 0.0, 0.0)),
            ),
        ),
        entities={
            "robot": EntityCfg(
                root_body_name="right/base",
                physical_entity="right",
                body_names=("right/base", "left/hand"),
                actuator_names=("right/drive",),
            )
        },
        primary_entity="right",
    )


def test_mjwarp_factory_selective_tracking_and_body_pair_fail_closed(tmp_path: Path) -> None:
    _require_cuda_mjwarp()
    scene = _scene(tmp_path)
    scene.tracked_body_names = ("left/hand", "right/base")
    backend = create_backend(
        "mjwarp",
        scene,
        2,
        0.002,
        body_state_required=True,
        tracked_body_names=scene.tracked_body_names,
        mjwarp_nconmax=32,
        mjwarp_njmax=64,
    )
    try:
        injected = backend.get_sensor_names()
        assert "track_pos_w_left/hand" in injected
        assert "track_pos_w_right/base" in injected
        assert "track_pos_w_left/base" not in injected
        assert "track_pos_w_right/hand" not in injected

        views = backend.get_tracked_body_views(device="cuda")
        assert views.body_names == ("left/hand", "right/base")
        assert views.pos_w.shape == (2, 2, 3)
        assert views.pos_w.device.type == "cuda"
        assert views.pos_w.dtype == torch.float32
        ctrl = torch.zeros((2, backend.num_actuators), dtype=torch.float32, device="cuda")
        backend.step_tensor(ctrl, nsteps=4)
        torch.testing.assert_close(
            views.pos_w[:, 0],
            backend.get_sensor_view("track_pos_w_left/hand", device="cuda"),
            rtol=2e-5,
            atol=2e-6,
        )

        assert not backend.get_tensor_capabilities().batched_body_pair_contacts
        entity_scene = EntityScene.from_scene_cfg(scene, backend, device="cuda")
        with pytest.raises(NotImplementedError, match="batched_body_pair_contacts capability"):
            entity_scene.compile_tensor_reads(
                "cuda",
                (),
                (SceneBodyPairContactReadSpec(entity="robot", sensor_names=("pair",)),),
            )
    finally:
        cast(SimBackend, backend).close()
