"""Cold-path public selector contract coverage for managed task compilation."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg


def _scene() -> SceneCfg:
    from unilab.assets import ASSETS_ROOT_PATH

    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))


def _backend() -> MuJoCoBackend:
    backend = MuJoCoBackend(
        _scene(),
        1,
        0.02 / 3.0,
        base_name="pelvis",
        np_dtype=np.float32,
    )
    backend.materialize()
    return backend


def test_mujoco_public_cold_selector_queries_are_exact_and_fail_closed() -> None:
    backend = _backend()
    try:
        sensor_names = ("pelvis_local_linvel", "torso_gyro", "torso_upvector")
        joint_names = ("left_hip_pitch_joint", "right_hip_pitch_joint")

        sensor_ids = backend.get_sensor_ids(sensor_names)
        dof_ids = backend.get_joint_dof_pos_indices(joint_names)

        assert sensor_ids.dtype == np.dtype(np.int32)
        assert sensor_ids.shape == (len(sensor_names),)
        assert np.all(sensor_ids >= 0)
        assert len(set(sensor_ids.tolist())) == len(sensor_names)
        assert dof_ids.dtype == np.dtype(np.int32)
        assert dof_ids.shape == (len(joint_names),)
        assert np.all(dof_ids >= 0)
        assert len(set(dof_ids.tolist())) == len(joint_names)

        with pytest.raises(ValueError, match="Sensor .* not found"):
            backend.get_sensor_ids(("missing_sensor",))
        with pytest.raises(ValueError, match="Joint .* not found"):
            backend.get_joint_dof_pos_indices(("missing_joint",))
        with pytest.raises(ValueError, match="not a single-DoF"):
            backend.get_joint_dof_pos_indices(("floating_base_joint",))
    finally:
        assert backend._pool is not None
        backend._pool.close()
