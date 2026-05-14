"""Tests for MuJoCo backend site / Jacobian contract."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    from mujoco.batch_env import BatchEnvPool  # noqa: F401
except Exception:
    pytest.skip(
        "mujoco.batch_env not available (platform/libstdc++ issue)", allow_module_level=True
    )

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg

MODEL_FILE = str(ASSETS_ROOT_PATH / "robots" / "go2_arm" / "scene_flat.xml")
NUM_ENVS = 4
ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
EE_SITE_NAME = "endpoint"


@pytest.fixture(scope="module")
def backend():
    b = MuJoCoBackend(
        SceneCfg(model_file=MODEL_FILE),
        num_envs=NUM_ENVS,
        sim_dt=0.01,
        base_name="base",
    )
    b.materialize()
    return b


def test_get_site_ids(backend):
    site_ids = backend.get_site_ids([EE_SITE_NAME])
    assert site_ids.shape == (1,)
    assert site_ids.dtype == np.int32
    assert site_ids[0] >= 0


def test_get_site_ids_not_found(backend):
    with pytest.raises(ValueError, match="not found"):
        backend.get_site_ids(["nonexistent_site_xyz"])


def test_get_joint_dof_indices(backend):
    indices = backend.get_joint_dof_indices(list(ARM_JOINT_NAMES))
    assert indices.shape == (6,)
    assert indices.dtype == np.int32
    assert np.all(indices >= 0)
    assert np.all(indices < backend.nv)


def test_get_joint_dof_pos_indices(backend):
    indices = backend.get_joint_dof_pos_indices(list(ARM_JOINT_NAMES))
    assert indices.shape == (6,)
    assert indices.dtype == np.int32
    # 所有索引在合法 dof_pos 范围内
    assert np.all(indices >= 0)
    assert np.all(indices < backend._num_dof_pos)


def test_get_joint_dof_vel_indices(backend):
    vel_indices = backend.get_joint_dof_vel_indices(list(ARM_JOINT_NAMES))
    dof_indices = backend.get_joint_dof_indices(list(ARM_JOINT_NAMES))
    # vel_indices = dof_indices - root_qvel_dim
    assert np.all(vel_indices == dof_indices - backend._root_qvel_dim)


@pytest.mark.slow
def test_get_site_jacobian_shape(backend):
    site_ids = backend.get_site_ids([EE_SITE_NAME])
    dof_indices = backend.get_joint_dof_indices(list(ARM_JOINT_NAMES))
    jacp, jacr = backend.get_site_jacobian_w(int(site_ids[0]), dof_indices)
    assert jacp.shape == (NUM_ENVS, 3, 6)
    assert jacr.shape == (NUM_ENVS, 3, 6)
    assert np.all(np.isfinite(jacp))
    assert np.all(np.isfinite(jacr))


@pytest.mark.slow
def test_get_site_jacobian_matches_serial(backend):
    """并行结果与串行逐 env 计算一致（差值 < 1e-6）。"""
    import mujoco

    site_ids = backend.get_site_ids([EE_SITE_NAME])
    site_id = int(site_ids[0])
    dof_indices = backend.get_joint_dof_indices(list(ARM_JOINT_NAMES))

    jacp_par, jacr_par = backend.get_site_jacobian_w(site_id, dof_indices)

    # 串行参考
    jacp_ser = np.zeros((NUM_ENVS, 3, 6), dtype=np.float64)
    jacr_ser = np.zeros((NUM_ENVS, 3, 6), dtype=np.float64)
    for env_idx in range(NUM_ENVS):
        variant_idx = int(backend._model_assignments[env_idx])
        model = backend._model_variants[variant_idx]
        data = mujoco.MjData(model)
        state = np.asarray(backend._physics_state[env_idx], dtype=np.float64)
        mujoco.mj_setState(model, data, state, int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_forward(model, data)
        jacp_full = np.zeros((3, model.nv), dtype=np.float64)
        jacr_full = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp_full, jacr_full, site_id)
        jacp_ser[env_idx] = jacp_full[:, dof_indices]
        jacr_ser[env_idx] = jacr_full[:, dof_indices]

    np.testing.assert_allclose(jacp_par, jacp_ser, atol=1e-6)
    np.testing.assert_allclose(jacr_par, jacr_ser, atol=1e-6)
