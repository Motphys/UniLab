"""Real Newton acceptance for UniLab's batched body-pair Manager read plan."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from unisim.backend.base import SimBackend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityInitialState, SceneEntitySpec

from unilab.base.backend_factory import create_backend
from unilab.base.entity import EntityCfg, EntityScene, SceneBodyPairContactReadSpec
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow

_MODEL = """
<mujoco model="manager-body-pair-contacts">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="base" pos="0 0 0.2">
      <freejoint/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="base_floor_second" body1="base" body2="world" data="found force pos"/>
    <contact name="base_floor_first" body1="base" body2="world" data="found force pos"/>
  </sensor>
</mujoco>
"""
_NAMES = ("base_floor_second", "base_floor_first")
_FIELDS = ("active", "count", "position_w", "force_w")


def _portable_source(tmp_path: Path, name: str, *, floating: bool) -> ModelSourceDescriptor:
    path = tmp_path / f"{name}.xml"
    root = '<body name="base"><freejoint/>' if floating else '<body name="base">'
    path.write_text(
        '<mujoco><compiler angle="radian"/>'
        "<worldbody>"
        f"{root}"
        '<geom name="base_geom" type="box" size=".08 .08 .08" mass="1"/>'
        '<body name="hand" pos="0 0 .1"><geom name="hand_geom" type="sphere" size=".05" mass=".2"/>'
        "</body></body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _require_cuda_newton() -> str:
    if sys.platform == "darwin":
        pytest.skip("newton is a CUDA-only backend; macOS has no supported CUDA runtime")
    pytest.importorskip("newton", reason="newton requires the newton extra")
    from unisim.backend.newton.dependencies import (
        NewtonDependencyError,
        load_newton_dependencies,
    )

    try:
        dependencies = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    dependencies.warp.init()
    device = dependencies.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton Manager body-pair reads require an active CUDA Warp device")
    return str(device)


def test_newton_scene_read_plan_matches_single_views_and_manager_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _require_cuda_newton()
    model_file = tmp_path / "body_pair.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    scene = SceneCfg(
        model_file=str(model_file),
        entities={"robot": EntityCfg(root_body_name="base", joint_names=(), body_names=("base",))},
    )
    backend = create_backend(
        "newton",
        scene,
        2,
        0.005,
        base_name="base",
        newton_device=device,
        capacity_check_steps=1,
        newton_deferred_contact_publication=True,
    )
    try:
        assert backend.get_tensor_capabilities().batched_body_pair_contacts
        entity_scene = EntityScene.from_scene_cfg(scene, backend, device=device)
        plan = entity_scene.compile_tensor_reads(
            device,
            (),
            (SceneBodyPairContactReadSpec(entity="robot", sensor_names=_NAMES),),
        )
        singles = {name: backend.get_sensor_view(name) for name in _NAMES}
        ctrl = torch.zeros((2, 0), dtype=torch.float32, device=device)
        backend.step_tensor(ctrl, nsteps=40)

        with monkeypatch.context() as hot_path:

            def forbidden_host_read(self: torch.Tensor) -> torch.Tensor:
                raise AssertionError("unexpected host transfer in Manager contact phase")

            hot_path.setattr(torch.Tensor, "cpu", forbidden_host_read)
            hot_path.setattr(torch.Tensor, "item", forbidden_host_read)
            plan.refresh()
            view = plan.body_pair_contact_view("robot", _NAMES)

        assert view.sensor_names == _NAMES
        assert view.active.shape == view.count.shape == (2, len(_NAMES))
        assert view.position_w.shape == view.force_w.shape == (2, len(_NAMES), 3)
        assert view.active.dtype == torch.bool
        assert view.count.dtype == torch.int32
        assert view.position_w.dtype == view.force_w.dtype == torch.float32
        assert view.active.device == view.count.device == torch.device(device)
        for slot, name in enumerate(_NAMES):
            for field in _FIELDS:
                torch.testing.assert_close(
                    getattr(view, field)[:, slot],
                    getattr(singles[name], field),
                    rtol=2e-5,
                    atol=2e-6,
                )

        retained = {
            name: {field: getattr(data, field).clone() for field in _FIELDS}
            for name, data in singles.items()
        }
        state_views = cast(Any, backend.get_state_views(("qpos", "qvel"), device=device))
        moved_qpos = state_views["qpos"][1:2].clone()
        moved_qpos[0, 0] = 20.0
        moved_qpos[0, 2] = 5.0
        backend.set_state_tensor(
            torch.tensor([1], dtype=torch.int64, device=device),
            moved_qpos,
            state_views["qvel"][1:2].contiguous(),
        )
        plan.refresh_selected()
        view = plan.body_pair_contact_view("robot", _NAMES)
        for slot, name in enumerate(_NAMES):
            for field in _FIELDS:
                torch.testing.assert_close(getattr(view, field)[0, slot], retained[name][field][0])
            assert not bool(view.active[1, slot])
            assert not bool(view.count[1, slot])
        assert torch.count_nonzero(view.position_w[1]) == 0
        assert torch.count_nonzero(view.force_w[1]) == 0
    finally:
        cast(SimBackend, backend).close()


def test_newton_multi_entity_batch_matches_ordered_single_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _require_cuda_newton()
    names = tuple(f"hand_object_{index}" for index in range(8))
    fragment = tmp_path / "contacts.xml"
    fragment.write_text(
        "<mujoco><sensor>"
        + "".join(
            f'<contact name="{name}" body1="robot/hand" body2="object/base" '
            'data="found force pos" num="1"/>'
            for name in names
        )
        + "</sensor></mujoco>",
        encoding="utf-8",
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                _portable_source(tmp_path, "robot", floating=True),
                initial_state=EntityInitialState((0.0, 0.0, 0.1)),
            ),
            SceneEntitySpec(
                "object",
                _portable_source(tmp_path, "object", floating=True),
                initial_state=EntityInitialState((0.0, 0.0, 0.2)),
            ),
            SceneEntitySpec(
                "table",
                _portable_source(tmp_path, "table", floating=False),
                root_mode="fixed",
            ),
        ),
        fragment_files=[str(fragment)],
        entities={
            "robot": EntityCfg(
                root_body_name="robot/base",
                physical_entity="robot",
                body_names=("robot/base", "robot/hand"),
            ),
            "object": EntityCfg(root_body_name="object/base", physical_entity="object"),
            "table": EntityCfg(root_body_name="table/base", physical_entity="table"),
        },
        primary_entity="robot",
    )
    backend = create_backend(
        "newton",
        scene,
        2,
        0.005,
        newton_device=device,
        capacity_check_steps=1,
        newton_deferred_contact_publication=True,
    )
    try:
        entity_scene = EntityScene.from_scene_cfg(scene, backend, device=device)
        plan = entity_scene.compile_tensor_reads(
            device,
            (),
            (SceneBodyPairContactReadSpec(entity="robot", sensor_names=names),),
        )
        singles = {name: backend.get_sensor_view(name) for name in names}
        ctrl = torch.zeros((2, backend.num_actuators), dtype=torch.float32, device=device)
        backend.step_tensor(ctrl, nsteps=4)
        with monkeypatch.context() as hot_path:

            def forbidden_host_read(self: torch.Tensor) -> torch.Tensor:
                raise AssertionError("unexpected host transfer in Manager contact phase")

            hot_path.setattr(torch.Tensor, "cpu", forbidden_host_read)
            hot_path.setattr(torch.Tensor, "item", forbidden_host_read)
            plan.refresh()
            view = plan.body_pair_contact_view("robot", names)

        assert torch.all(view.active)
        assert torch.all(view.count > 0)
        for slot, name in enumerate(names):
            for field in _FIELDS:
                torch.testing.assert_close(
                    getattr(view, field)[:, slot],
                    getattr(singles[name], field),
                    rtol=2e-5,
                    atol=2e-6,
                )
    finally:
        cast(SimBackend, backend).close()
