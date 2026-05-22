from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    import mujoco
    from mujoco.batch_env import BatchEnvPool
except Exception:
    pytest.skip(
        "mujoco.batch_env not available (platform/libstdc++ issue)", allow_module_level=True
    )

if not hasattr(BatchEnvPool, "compute_site_jacobians"):
    # This is a mujoco-uni 3.8 API. UniLab currently pins mujoco-uni 3.7.0rc0
    # because Sharpa simulates incorrectly on 3.8, so keep this coverage skipped.
    pytest.skip(
        "BatchEnvPool.compute_site_jacobians requires mujoco-uni>=3.8.0rc0",
        allow_module_level=True,
    )

from unilab.assets import ASSETS_ROOT_PATH

mj: Any = mujoco

GO2_SITE_NAME = "imu"


@dataclass
class _PoolCtx:
    model: Any
    pool: BatchEnvPool
    initial_state: np.ndarray
    site_id: int


def _xml(robot: str, scene: str = "scene_flat.xml") -> str:
    return str(ASSETS_ROOT_PATH / "robots" / robot / scene)


def _make_initial_state(model: Any, nbatch: int, rng: np.random.Generator) -> np.ndarray:
    nstate = mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
    state0 = np.zeros((nbatch, nstate), dtype=np.float64)
    state0[:, 1 : 1 + model.nq] = model.qpos0
    # Perturb dof positions / velocities so each env produces a distinct Jacobian.
    state0[:, 1 + 7 : 1 + model.nq] += 0.05 * rng.standard_normal((nbatch, model.nq - 7))
    state0[:, 1 + model.nq + 6 : 1 + model.nq + model.nv] += 0.02 * rng.standard_normal(
        (nbatch, model.nv - 6)
    )
    return state0


def _reference_site_jacobian(
    model: Any, state_row: np.ndarray, site_id: int, jacp: bool, jacr: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    data = mj.MjData(model)
    mj.mj_setState(model, data, state_row, int(mj.mjtState.mjSTATE_FULLPHYSICS))
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    jp = np.zeros((3, model.nv), dtype=np.float64) if jacp else None
    jr = np.zeros((3, model.nv), dtype=np.float64) if jacr else None
    mj.mj_jacSite(model, data, jp, jr, site_id)
    return jp, jr


@pytest.fixture
def pool_ctx() -> Iterator[_PoolCtx]:
    rng = np.random.default_rng(0)
    model = mj.MjModel.from_xml_path(_xml("go2"))
    site_id = mj.mj_name2id(model, int(mj.mjtObj.mjOBJ_SITE), GO2_SITE_NAME)
    assert site_id >= 0, f"go2 model is expected to expose site '{GO2_SITE_NAME}'"
    nbatch = 3
    pool = BatchEnvPool(model, nbatch=nbatch, nthread=2)
    try:
        yield _PoolCtx(
            model=model,
            pool=pool,
            initial_state=_make_initial_state(model, nbatch, rng),
            site_id=site_id,
        )
    finally:
        pool.close()


def test_compute_site_jacobians_matches_reference_jacp_only(pool_ctx: _PoolCtx) -> None:
    jp, jr = pool_ctx.pool.compute_site_jacobians(
        pool_ctx.initial_state, [pool_ctx.site_id], jacp=True, jacr=False
    )
    assert jr is None
    assert jp.shape == (pool_ctx.initial_state.shape[0], 1, 3, pool_ctx.model.nv)
    for i in range(pool_ctx.initial_state.shape[0]):
        ref_jp, _ = _reference_site_jacobian(
            pool_ctx.model, pool_ctx.initial_state[i], pool_ctx.site_id, True, False
        )
        np.testing.assert_allclose(jp[i, 0], ref_jp, atol=1e-12, rtol=0)


def test_compute_site_jacobians_matches_reference_jacp_and_jacr(pool_ctx: _PoolCtx) -> None:
    jp, jr = pool_ctx.pool.compute_site_jacobians(
        pool_ctx.initial_state, [pool_ctx.site_id], jacp=True, jacr=True
    )
    assert jp.shape == (pool_ctx.initial_state.shape[0], 1, 3, pool_ctx.model.nv)
    assert jr.shape == (pool_ctx.initial_state.shape[0], 1, 3, pool_ctx.model.nv)
    for i in range(pool_ctx.initial_state.shape[0]):
        ref_jp, ref_jr = _reference_site_jacobian(
            pool_ctx.model, pool_ctx.initial_state[i], pool_ctx.site_id, True, True
        )
        np.testing.assert_allclose(jp[i, 0], ref_jp, atol=1e-12, rtol=0)
        np.testing.assert_allclose(jr[i, 0], ref_jr, atol=1e-12, rtol=0)


def test_compute_site_jacobians_scalar_site_squeezes_k_dim(pool_ctx: _PoolCtx) -> None:
    jp, jr = pool_ctx.pool.compute_site_jacobians(
        pool_ctx.initial_state, pool_ctx.site_id, jacp=True, jacr=True
    )
    assert jp.shape == (pool_ctx.initial_state.shape[0], 3, pool_ctx.model.nv)
    assert jr.shape == (pool_ctx.initial_state.shape[0], 3, pool_ctx.model.nv)
    for i in range(pool_ctx.initial_state.shape[0]):
        ref_jp, ref_jr = _reference_site_jacobian(
            pool_ctx.model, pool_ctx.initial_state[i], pool_ctx.site_id, True, True
        )
        np.testing.assert_allclose(jp[i], ref_jp, atol=1e-12, rtol=0)
        np.testing.assert_allclose(jr[i], ref_jr, atol=1e-12, rtol=0)


def test_compute_site_jacobians_requires_at_least_one_flag(pool_ctx: _PoolCtx) -> None:
    with pytest.raises(ValueError):
        pool_ctx.pool.compute_site_jacobians(
            pool_ctx.initial_state, [pool_ctx.site_id], jacp=False, jacr=False
        )


def test_compute_site_jacobians_rejects_invalid_site_id(pool_ctx: _PoolCtx) -> None:
    with pytest.raises(ValueError):
        pool_ctx.pool.compute_site_jacobians(
            pool_ctx.initial_state, [pool_ctx.model.nsite], jacp=True
        )


def test_compute_site_jacobians_rejects_wrong_state_shape(pool_ctx: _PoolCtx) -> None:
    bad_state = pool_ctx.initial_state[:-1]
    with pytest.raises(ValueError):
        pool_ctx.pool.compute_site_jacobians(bad_state, [pool_ctx.site_id], jacp=True)
