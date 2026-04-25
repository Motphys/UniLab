"""Tests for MuJoCo-backed XML editing helpers."""

from __future__ import annotations

import os

import pytest
from unilab.utils.xml_utils import inject_motrix_tracking_sensors, inject_mujoco_tracking_sensors

from unilab.assets import ASSETS_ROOT_PATH


def _g1_scene() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def test_inject_mujoco_tracking_sensors_uses_mjspec_and_preserves_contract() -> None:
    mujoco = pytest.importorskip("mujoco")

    tmp_xml, tracked_body_ids, valid_bnames = inject_mujoco_tracking_sensors(
        _g1_scene(),
        baselink_name="pelvis",
    )
    try:
        assert tracked_body_ids == list(range(1, len(valid_bnames) + 1))
        assert valid_bnames[0] == "pelvis"

        model = mujoco.MjModel.from_xml_path(tmp_xml)
        for sensor_name in (
            "track_pos_w_pelvis",
            "track_quat_w_pelvis",
            "track_linvel_w_pelvis",
            "track_angvel_w_pelvis",
            "track_pos_b_pelvis",
            "track_quat_b_pelvis",
            "track_linvel_b_pelvis",
            "track_angvel_b_pelvis",
        ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0
    finally:
        os.remove(tmp_xml)


def test_inject_motrix_tracking_sensors_only_adds_baselink_relative_sensors() -> None:
    mujoco = pytest.importorskip("mujoco")

    tmp_xml, _, _ = inject_motrix_tracking_sensors(_g1_scene(), baselink_name="pelvis")
    try:
        model = mujoco.MjModel.from_xml_path(tmp_xml)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_pos_b_pelvis") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_quat_b_pelvis") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_linvel_b_pelvis") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_angvel_b_pelvis") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "track_pos_w_pelvis") == -1
    finally:
        os.remove(tmp_xml)
