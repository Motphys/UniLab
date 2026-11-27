from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import mujoco
import numpy as np
import pytest
from unisim.visualization.mujoco_geom import geom_mesh

from unilab.visualization import viser_scene
from unilab.visualization.viser_scene import (
    MujocoViserBatchScene,
    MujocoViserScene,
    build_visible_env_indices,
)


class _FakeHandle:
    def __init__(self) -> None:
        self.position = None
        self.wxyz = None
        self.batched_positions = None
        self.batched_wxyzs = None
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _FakeScene:
    def __init__(self) -> None:
        self.handles: list[_FakeHandle] = []
        self.up: str | None = None
        self.grids: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.simple_meshes: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.batched_meshes: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def set_up_direction(self, value: str) -> None:
        self.up = value

    def _handle(self) -> _FakeHandle:
        handle = _FakeHandle()
        self.handles.append(handle)
        return handle

    def add_grid(self, *args, **kwargs):
        self.grids.append((args, kwargs))
        return self._handle()

    def add_icosphere(self, *args, **kwargs):
        del args, kwargs
        return self._handle()

    def add_mesh_trimesh(self, *args, **kwargs):
        del args, kwargs
        return self._handle()

    def add_cylinder(self, *args, **kwargs):
        del args, kwargs
        return self._handle()

    def add_box(self, *args, **kwargs):
        del args, kwargs
        return self._handle()

    def add_mesh_simple(self, *args, **kwargs):
        self.simple_meshes.append((args, kwargs))
        return self._handle()

    def add_batched_meshes_simple(self, *args, **kwargs):
        self.batched_meshes.append((args, kwargs))
        return self._handle()


class _FakeServer:
    def __init__(self) -> None:
        self.scene = _FakeScene()

    def atomic(self):
        return nullcontext()


def test_mujoco_viser_scene_applies_position_offset_and_close() -> None:
    xml = """
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="2 2 0.1"/>
        <body name="box_body" pos="0 0 0.5">
          <geom name="box" type="box" size="0.1 0.2 0.3"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)  # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]

    server = _FakeServer()
    scene = MujocoViserScene(
        server,
        model,
        name_prefix="/mujoco/test",
        position_offset=(1.0, 2.0, 0.0),
        render_plane=False,
    )
    scene.update(data)

    assert server.scene.up == "+z"
    assert len(server.scene.handles) == 1
    expected = data.geom_xpos[1] + np.array([1.0, 2.0, 0.0], dtype=np.float64)
    assert server.scene.handles[0].position == (
        float(expected[0]),
        float(expected[1]),
        float(expected[2]),
    )

    scene.close()
    assert server.scene.handles[0].removed is True


def test_build_visible_env_indices_spreads_slots_across_full_batch() -> None:
    indices = build_visible_env_indices(num_envs=64, visible_envs=16)
    np.testing.assert_array_equal(indices, np.arange(0, 64, 4, dtype=np.int32))


def test_mujoco_viser_batch_scene_updates_instances() -> None:
    xml = """
    <mujoco>
      <worldbody>
        <body name="box_body" pos="0 0 0.5">
          <geom name="box" type="box" size="0.1 0.2 0.3"/>
        </body>
      </worldbody>
    </mujoco>
    """
    models = [
        mujoco.MjModel.from_xml_string(xml),  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.MjModel.from_xml_string(xml),  # pyright: ignore[reportAttributeAccessIssue]
    ]
    data = [mujoco.MjData(model) for model in models]  # pyright: ignore[reportAttributeAccessIssue]
    for model, item in zip(models, data, strict=True):
        mujoco.mj_forward(model, item)  # pyright: ignore[reportAttributeAccessIssue]

    server = _FakeServer()
    scene = MujocoViserBatchScene(
        server,
        models,
        position_offsets=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
        render_plane=False,
    )
    scene.update(data)

    assert len(server.scene.handles) == 1
    handle = server.scene.handles[0]
    expected = np.stack([item.geom_xpos[0] for item in data]).astype(np.float32)
    expected += np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(handle.batched_positions, expected)
    assert handle.batched_wxyzs.shape == (2, 4)

    scene.close()
    assert handle.removed is True


def _grouped_geoms_xml() -> str:
    return """
    <mujoco>
      <worldbody>
        <body name="body" pos="0 0 0.5">
          <geom name="visual" type="box" size="0.1 0.1 0.1" group="2"/>
          <geom name="collision_marker" type="box" size="0.2 0.2 0.2" group="3"/>
        </body>
      </worldbody>
    </mujoco>
    """


def test_mujoco_viser_scene_hides_geom_groups_beyond_mujoco_default() -> None:
    # MuJoCo's default mjvOption.geomgroup enables groups 0-2 only; the viser
    # scene must match so group-3 collision markers stay hidden.
    model = mujoco.MjModel.from_xml_string(_grouped_geoms_xml())  # pyright: ignore[reportAttributeAccessIssue]

    server = _FakeServer()
    scene = MujocoViserScene(server, model, render_plane=False)

    assert model.geom_group.tolist() == [2, 3]
    assert list(scene._handles) == [0]
    assert len(server.scene.handles) == 1


def test_mujoco_viser_scene_visible_geom_groups_override() -> None:
    model = mujoco.MjModel.from_xml_string(_grouped_geoms_xml())  # pyright: ignore[reportAttributeAccessIssue]

    server = _FakeServer()
    scene = MujocoViserScene(server, model, render_plane=False, visible_geom_groups=(0, 1, 2, 3))

    assert list(scene._handles) == [0, 1]


def test_mujoco_viser_batch_scene_hides_geom_groups_beyond_mujoco_default() -> None:
    models = [
        mujoco.MjModel.from_xml_string(_grouped_geoms_xml()),  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.MjModel.from_xml_string(_grouped_geoms_xml()),  # pyright: ignore[reportAttributeAccessIssue]
    ]

    server = _FakeServer()
    scene = MujocoViserBatchScene(server, models, render_plane=False)

    assert list(scene._handles) == [0]
    assert len(server.scene.handles) == 1


def _every_meshable_geom_kind_xml() -> str:
    # One geom per kind ``geom_mesh`` supports, behind a plane that the batch
    # scene draws as a grid instead of a mesh.
    return """
    <mujoco>
      <asset>
        <mesh name="tetra" vertex="0 0 0 0.1 0 0 0 0.1 0 0 0 0.1" face="0 2 1 0 1 3 0 3 2 1 2 3"/>
      </asset>
      <worldbody>
        <geom name="ground" type="plane" size="2 2 0.1"/>
        <geom name="sphere" type="sphere" size="0.1"/>
        <geom name="capsule" type="capsule" size="0.05 0.2"/>
        <geom name="ellipsoid" type="ellipsoid" size="0.1 0.2 0.15"/>
        <geom name="cylinder" type="cylinder" size="0.1 0.3"/>
        <geom name="box" type="box" size="0.1 0.2 0.3"/>
        <geom name="tetra" type="mesh" mesh="tetra"/>
      </worldbody>
    </mujoco>
    """


def _assert_carries_geom_mesh(
    kwargs: dict[str, Any], model: mujoco.MjModel, geom_index: int
) -> None:
    """Check that ``kwargs`` hold the ``geom_mesh`` output of one geom, cast as today."""
    vertices, faces = geom_mesh(model, geom_index)
    assert kwargs["vertices"].dtype == np.float32
    assert kwargs["faces"].dtype == np.int32
    np.testing.assert_array_equal(kwargs["vertices"], vertices.astype(np.float32))
    np.testing.assert_array_equal(kwargs["faces"], faces.astype(np.int32))


def test_mujoco_viser_batch_scene_draws_every_geom_mesh_kind_through_geom_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = mujoco.MjModel.from_xml_string(_every_meshable_geom_kind_xml())  # pyright: ignore[reportAttributeAccessIssue]
    requested: list[int] = []

    def recording_geom_mesh(model: mujoco.MjModel, geom_index: int):
        requested.append(geom_index)
        return geom_mesh(model, geom_index)

    monkeypatch.setattr(viser_scene, "geom_mesh", recording_geom_mesh)
    server = _FakeServer()
    scene = MujocoViserBatchScene(server, [model])

    # Every geom behind the plane is resolved by the shared helper, and only those.
    assert requested == [1, 2, 3, 4, 5, 6]
    assert list(scene._handles) == requested
    assert [args for args, _ in server.scene.batched_meshes] == [
        (f"/mujoco/batch/geom/{geom_index}",) for geom_index in requested
    ]
    for geom_index, (_, kwargs) in zip(requested, server.scene.batched_meshes, strict=True):
        _assert_carries_geom_mesh(kwargs, model, geom_index)

    # The plane is still one grid and nothing else: no scene node is created
    # besides that grid and the six batched meshes.
    assert [args for args, _ in server.scene.grids] == [("/mujoco/batch/ground",)]
    _, grid_kwargs = server.scene.grids[0]
    assert (grid_kwargs["width"], grid_kwargs["height"], grid_kwargs["cell_size"]) == (
        4.0,
        4.0,
        0.5,
    )
    assert len(server.scene.handles) == len(server.scene.grids) + len(server.scene.batched_meshes)


def test_mujoco_viser_scene_draws_mesh_geoms_through_geom_mesh() -> None:
    model = mujoco.MjModel.from_xml_string(_every_meshable_geom_kind_xml())  # pyright: ignore[reportAttributeAccessIssue]
    server = _FakeServer()
    scene = MujocoViserScene(server, model, render_plane=True)

    # The native Viser primitives keep drawing plane, sphere, capsule, ellipsoid,
    # cylinder, and box; only the mesh geom goes through ``add_mesh_simple``.
    assert list(scene._handles) == [0, 1, 2, 3, 4, 5, 6]
    assert len(server.scene.simple_meshes) == 1
    args, kwargs = server.scene.simple_meshes[0]
    assert args == ("/mujoco/geom/6",)
    _assert_carries_geom_mesh(kwargs, model, 6)

    # A mesh geom without compiled mesh data is skipped, as before the switch.
    model.geom_dataid[6] = -1
    server = _FakeServer()
    scene = MujocoViserScene(server, model, render_plane=False)
    assert server.scene.simple_meshes == []
    assert list(scene._handles) == [1, 2, 3, 4, 5]
