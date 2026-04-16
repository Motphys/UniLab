from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    import mujoco
    from mujoco.batch_env import SUPPORTED_FIELDS, BatchEnvPool
except Exception:
    pytest.skip(
        "mujoco.batch_env not available (platform/libstdc++ issue)", allow_module_level=True
    )

from unilab.assets import ASSETS_ROOT_PATH

mj: Any = mujoco

EXPECTED_SUPPORTED_FIELDS = {
    "body_mass",
    "body_ipos",
    "body_iquat",
    "body_inertia",
    "dof_armature",
    "geom_friction",
    "kp",
    "kd",
}


@dataclass
class _PoolCtx:
    model: Any
    pool: BatchEnvPool
    initial_state: np.ndarray


def _xml(robot: str, scene: str = "scene_flat.xml") -> str:
    return str(ASSETS_ROOT_PATH / "robots" / robot / scene)


def _make_initial_state(model: Any) -> np.ndarray:
    nstate = mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
    initial_state = np.zeros((1, nstate), dtype=np.float64)
    initial_state[:, 1 : 1 + model.nq] = model.qpos0
    return initial_state


@pytest.fixture
def pool_ctx() -> Iterator[_PoolCtx]:
    model = mj.MjModel.from_xml_path(_xml("go2"))
    pool = BatchEnvPool(model, nbatch=2, nthread=1)
    try:
        yield _PoolCtx(model=model, pool=pool, initial_state=_make_initial_state(model))
    finally:
        pool.close()


def _reset_and_assert_field_applied(
    pool_ctx: _PoolCtx, field_name: str, updated: np.ndarray
) -> None:
    original_0 = pool_ctx.pool.get_field(0, field_name).copy()
    original_1 = pool_ctx.pool.get_field(1, field_name).copy()

    assert updated.shape == original_1.shape
    assert not np.allclose(updated, original_1)

    pool_ctx.pool.reset(
        env_ids=[1],
        initial_state=pool_ctx.initial_state.copy(),
        randomization={field_name: updated[None, :]},
    )

    np.testing.assert_array_equal(pool_ctx.pool.get_field(0, field_name), original_0)
    np.testing.assert_allclose(pool_ctx.pool.get_field(1, field_name), updated)


@pytest.mark.slow
def test_batch_env_supported_fields_match_documented_reset_randomization_fields() -> None:
    assert set(SUPPORTED_FIELDS) == EXPECTED_SUPPORTED_FIELDS


@pytest.mark.slow
def test_batch_env_reset_applies_body_mass_randomization(pool_ctx: _PoolCtx) -> None:
    updated = pool_ctx.pool.get_field(1, "body_mass").copy()
    updated[1] += 0.25
    _reset_and_assert_field_applied(pool_ctx, "body_mass", updated)


@pytest.mark.slow
def test_batch_env_reset_applies_body_ipos_randomization(pool_ctx: _PoolCtx) -> None:
    updated = pool_ctx.pool.get_field(1, "body_ipos").reshape(pool_ctx.model.nbody, 3).copy()
    updated[1] += np.array([0.01, -0.02, 0.03], dtype=np.float64)
    _reset_and_assert_field_applied(pool_ctx, "body_ipos", updated.reshape(-1))


@pytest.mark.slow
def test_batch_env_reset_applies_body_iquat_randomization(pool_ctx: _PoolCtx) -> None:
    updated = pool_ctx.pool.get_field(1, "body_iquat").reshape(pool_ctx.model.nbody, 4).copy()
    quat = np.array([0.92387953, 0.0, 0.38268343, 0.0], dtype=np.float64)
    updated[1] = quat / np.linalg.norm(quat)
    _reset_and_assert_field_applied(pool_ctx, "body_iquat", updated.reshape(-1))


@pytest.mark.slow
def test_batch_env_reset_applies_body_inertia_randomization(pool_ctx: _PoolCtx) -> None:
    updated = pool_ctx.pool.get_field(1, "body_inertia").reshape(pool_ctx.model.nbody, 3).copy()
    updated[1] *= 1.25
    _reset_and_assert_field_applied(pool_ctx, "body_inertia", updated.reshape(-1))


@pytest.mark.slow
def test_batch_env_reset_applies_dof_armature_randomization(pool_ctx: _PoolCtx) -> None:
    updated = pool_ctx.pool.get_field(1, "dof_armature").copy()
    updated += np.linspace(0.01, 0.03, updated.size, dtype=np.float64)
    _reset_and_assert_field_applied(pool_ctx, "dof_armature", updated)


@pytest.mark.slow
def test_batch_env_reset_applies_geom_friction_randomization(pool_ctx: _PoolCtx) -> None:
    updated = pool_ctx.pool.get_field(1, "geom_friction").reshape(pool_ctx.model.ngeom, 3).copy()
    updated[0] += np.array([0.1, 0.002, 0.0002], dtype=np.float64)
    _reset_and_assert_field_applied(pool_ctx, "geom_friction", updated.reshape(-1))


@pytest.mark.slow
def test_batch_env_reset_applies_kp_randomization(pool_ctx: _PoolCtx) -> None:
    updated = pool_ctx.pool.get_field(1, "kp").copy()
    updated += 1.25
    _reset_and_assert_field_applied(pool_ctx, "kp", updated)


@pytest.mark.slow
def test_batch_env_reset_applies_kd_randomization(pool_ctx: _PoolCtx) -> None:
    updated = pool_ctx.pool.get_field(1, "kd").copy()
    updated += 0.25
    _reset_and_assert_field_applied(pool_ctx, "kd", updated)
