"""Tests for the interactive-path DebugPrimitive injection helpers."""

from __future__ import annotations

import numpy as np
import pytest
from unisim.backend.base import DebugPrimitive

from unilab.visualization.debug_primitives import (
    append_debug_primitives_to_scene,
    quat_from_z_axis,
    segment_arrow,
)

mujoco = pytest.importorskip("mujoco")


def _quat_rotate(quat: tuple[float, float, float, float], vec: np.ndarray) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.array(quat, dtype=np.float64))
    return mat.reshape(3, 3) @ vec


def _make_scene(maxgeom: int = 64):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>'
    )
    return mujoco.MjvScene(model, maxgeom=maxgeom)


def test_quat_from_z_axis_rotates_z_onto_direction() -> None:
    quat = quat_from_z_axis([1.0, 0.0, 0.0])
    rotated = _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(rotated, [1.0, 0.0, 0.0], atol=1e-9)


def test_quat_from_z_axis_handles_opposite_and_zero() -> None:
    quat = quat_from_z_axis([0.0, 0.0, -2.0])
    rotated = _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(rotated, [0.0, 0.0, -1.0], atol=1e-9)
    with pytest.raises(ValueError, match="non-zero"):
        quat_from_z_axis([0.0, 0.0, 0.0])


def test_segment_arrow_spans_endpoints() -> None:
    arrow = segment_arrow([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], rgba=(1.0, 0.0, 0.0, 1.0))
    assert arrow is not None
    assert arrow.kind == "arrow"
    assert arrow.size == pytest.approx((2.0,))
    direction = _quat_rotate(arrow.quat, np.array([0.0, 0.0, 1.0])) * arrow.size[0]
    assert np.allclose(arrow.pos + direction, [0.0, 0.0, 2.0], atol=1e-9)


def test_segment_arrow_returns_none_when_degenerate() -> None:
    assert segment_arrow([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], rgba=(1.0, 1.0, 1.0, 1.0)) is None


def test_append_primitives_adds_expected_geom_counts() -> None:
    scene = _make_scene()
    primitives = [
        DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.05,)),
        DebugPrimitive(kind="box", pos=(0.2, 0.0, 0.5), size=(0.05, 0.05, 0.05)),
        DebugPrimitive(kind="frame", pos=(0.0, 0.2, 0.5), size=(0.1,)),
        DebugPrimitive(
            kind="arrow",
            pos=(0.0, 0.0, 0.5),
            quat=quat_from_z_axis([1.0, 0.0, 0.0]),
            size=(0.3,),
        ),
        DebugPrimitive(kind="text", pos=(0.0, 0.0, 0.8), text="label"),
    ]
    # sphere(1) + box(1) + frame(3 axes) + arrow(1); text is a documented no-op.
    added = append_debug_primitives_to_scene(scene, primitives)
    assert added == 6
    assert scene.ngeom == 6


def test_append_primitives_stops_at_maxgeom() -> None:
    scene = _make_scene(maxgeom=1)
    primitives = [
        DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.05,)),
        DebugPrimitive(kind="sphere", pos=(0.1, 0.0, 0.5), size=(0.05,)),
    ]
    added = append_debug_primitives_to_scene(scene, primitives)
    assert added == 1
    assert scene.ngeom == 1


def test_append_ghost_geom_requires_resolvable_mesh() -> None:
    scene = _make_scene()
    ghost = DebugPrimitive(kind="ghost_geom", pos=(0.0, 0.0, 0.5), mesh_asset="missing_mesh")
    with pytest.raises(ValueError, match="missing_mesh"):
        append_debug_primitives_to_scene(scene, [ghost])
