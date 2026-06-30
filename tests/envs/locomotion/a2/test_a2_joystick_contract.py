"""Contract tests for the A2JoystickFlat environment (leg-only Unitree A2).

The A2 leg-only MJCF mirrors the Go2 joystick sensor/geom/leg-ordering
contract (legs FL,FR,RL,RR; foot geoms+sites FL/FR/RL/RR; Go2-named IMU/foot
sensors) and uses <position> actuators, so the env reuses Go2WalkTask
unchanged. These tests prove the A2 model + scene + config + env chain
constructs and steps in MuJoCo as a 12-DOF joystick task."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH


def _skip_if_no_mujoco():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")


def test_a2_robot_xml_compiles_with_12_position_actuators():
    """a2.xml loads standalone and exposes exactly 12 position-style leg
    actuators in the FL,FR,RL,RR x hip,thigh,calf order."""
    mujoco = pytest.importorskip("mujoco")
    xml = ASSETS_ROOT_PATH / "robots" / "a2" / "a2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.nu == 12
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]
    assert names == [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]
    # Position actuators carry an affine bias (kp in gainprm[0]); motor actuators do not.
    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    assert all(int(model.actuator_biastype[i]) == affine for i in range(model.nu))
