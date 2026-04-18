from __future__ import annotations

from contextlib import nullcontext

import mujoco
import numpy as np
import pytest

from unilab.utils.viser_scene import VISER_AVAILABLE, MujocoViserScene, build_visible_env_indices


class _FakeHandle:
    def __init__(self) -> None:
        self.position = None
        self.wxyz = None
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _FakeScene:
    def __init__(self) -> None:
        self.handles: list[_FakeHandle] = []
        self.up: str | None = None

    def set_up_direction(self, value: str) -> None:
        self.up = value

    def _handle(self) -> _FakeHandle:
        handle = _FakeHandle()
        self.handles.append(handle)
        return handle

    def add_grid(self, *args, **kwargs):
        del args, kwargs
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
        del args, kwargs
        return self._handle()


class _FakeServer:
    def __init__(self) -> None:
        self.scene = _FakeScene()

    def atomic(self):
        return nullcontext()


@pytest.mark.skipif(not VISER_AVAILABLE, reason="viser optional dependency is not installed")
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
