"""Minimal regression test for MjSpec sensor compile failure.

See: https://github.com/google-deepmind/mujoco/issues/XXX
"""

from __future__ import annotations

import pytest


def test_mjspec_sensor_compile() -> None:
    """Adding a frame sensor via MjSpec should not crash to_xml/compile."""
    mujoco = pytest.importorskip("mujoco")

    from unilab.assets import ASSETS_ROOT_PATH

    model_file = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    spec = mujoco.MjSpec.from_file(model_file)
    spec.add_sensor(
        name="track_pos_w_pelvis",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="pelvis",
    )
    # This should succeed but currently raises UnicodeDecodeError / ValueError
    # in mujoco 3.6.0 due to a bug in MjSpec sensor serialization.
    xml = spec.to_xml()
    assert isinstance(xml, str)
    assert "track_pos_w_pelvis" in xml
