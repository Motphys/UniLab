"""Contract + parity tests for the extended ``SimBackend.set_state`` timing dict.

Issue #679 extends the ``SimBackend.set_state`` return type from ``None`` to
``dict | None`` so backends can surface per-substep timing to the DR manager. This
module locks in three invariants:

1. Both MuJoCo and Motrix ``set_state`` return a dict with a ``"timing"`` sub-dict
   whose keys are the schema documented in
   ``unilab.base.np_env.BACKEND_SET_STATE_DETAIL_TIMING_KEYS``.
2. Every reported sub-timing is a non-negative float.
3. The reported sub-timings sum to within ~5 ms of the outer wall-clock cost
   (``set_state_internal_gap_ms`` catches the residual). This is the same
   consistency check the benchmark harness relies on when computing % share.

The tests double as parity anchors for the Step 5 Motrix refactor: pre-refactor
they capture the current behavior (all schema keys populated, gap tiny); post-
refactor they must still hold, so any regression that skips a sub-key or blows
up the gap is caught before the benchmark run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.np_env import BACKEND_SET_STATE_DETAIL_TIMING_KEYS
from unilab.base.scene import SceneCfg
from unilab.dr.types import ResetRandomizationPayload

pytest.importorskip("mujoco", reason="mujoco not installed")


NUM_ENVS = 2
SIM_DT = 0.005
_G1_MODEL_FILE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")


def _identity_qpos_mujoco(nq: int, xyz=(0.0, 0.0, 0.8)) -> np.ndarray:
    q = np.zeros((1, nq))
    q[0, :3] = xyz
    q[0, 3] = 1.0
    return q


def _assert_timing_dict_shape(result: Any) -> dict[str, float]:
    """Common assertions across backends."""
    assert isinstance(result, dict), f"set_state must return a dict, got {type(result)!r}"
    timing = result.get("timing")
    assert isinstance(timing, dict), "set_state result must contain a 'timing' sub-dict"

    missing = [key for key in BACKEND_SET_STATE_DETAIL_TIMING_KEYS if key not in timing]
    assert not missing, f"missing keys in set_state timing: {missing}"

    for key in BACKEND_SET_STATE_DETAIL_TIMING_KEYS:
        value = timing[key]
        assert isinstance(value, float), f"{key} must be float, got {type(value)!r}"
        # Gap can be very slightly negative due to perf_counter noise; the
        # sum-based check below is what we really care about.
        if not key.endswith("internal_gap_ms"):
            assert value >= 0.0, f"{key} must be non-negative, got {value}"

    return timing


def _assert_gap_bounded(timing: dict[str, float]) -> None:
    """Sub-timings should sum to ~ the outer wall-clock cost; gap is the residual.

    We just assert the reported ``set_state_internal_gap_ms`` stays within a
    generous absolute bound so a refactor that forgets to track a new sub-step
    (making the gap balloon) is caught.
    """
    gap = timing["set_state_internal_gap_ms"]
    assert abs(gap) < 5.0, f"set_state_internal_gap_ms out of bounds: {gap}"


def test_mujoco_set_state_returns_schema_conformant_timing() -> None:
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    backend = MuJoCoBackend(
        SceneCfg(model_file=_G1_MODEL_FILE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()
    qpos = _identity_qpos_mujoco(backend.model.nq, xyz=(1.0, 2.0, 0.8))
    qvel = np.zeros((1, backend.model.nv))

    result = backend.set_state(np.array([0]), qpos, qvel)

    timing = _assert_timing_dict_shape(result)
    _assert_gap_bounded(timing)
    # MuJoCo-specific sub-keys must be populated on the mujoco path.
    assert timing["set_state_pool_reset_ms"] > 0.0
    assert timing["set_state_state_scatter_ms"] > 0.0
    # Motrix-only sub-keys report 0.0 on the mujoco backend.
    assert timing["set_state_mask_ms"] == 0.0
    assert timing["set_state_data_slice_ms"] == 0.0
    assert timing["set_state_forward_kinematic_ms"] == 0.0
    assert timing["set_state_refresh_pose_cache_ms"] == 0.0


def test_mujoco_set_state_empty_env_indices_still_returns_timing_dict() -> None:
    """Backends must never return None even on the empty-reset fast path."""
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    backend = MuJoCoBackend(
        SceneCfg(model_file=_G1_MODEL_FILE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()

    result = backend.set_state(
        np.array([], dtype=np.int32),
        np.zeros((0, backend.model.nq)),
        np.zeros((0, backend.model.nv)),
    )

    timing = _assert_timing_dict_shape(result)
    # Empty-reset path skips all substeps.
    assert timing["set_state_pool_reset_ms"] == 0.0
    assert timing["set_state_state_scatter_ms"] == 0.0
    assert timing["set_state_internal_gap_ms"] == 0.0


def test_motrix_set_state_returns_schema_conformant_timing() -> None:
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend

    backend = MotrixBackend(
        SceneCfg(model_file=_G1_MODEL_FILE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    nq = backend.get_dof_pos().shape[-1] + 7
    nv = backend.get_dof_vel().shape[-1] + 6
    qpos = _identity_qpos_mujoco(nq, xyz=(1.0, 2.0, 0.8))
    qvel = np.zeros((1, nv))
    randomization = ResetRandomizationPayload(
        gravity=np.asarray([[0.5, 0.5, -3.0]], dtype=np.float64)
    )

    result = backend.set_state(np.array([0]), qpos, qvel, randomization=randomization)

    timing = _assert_timing_dict_shape(result)
    _assert_gap_bounded(timing)
    # Motrix path must populate the sub-steps it actually runs.
    assert timing["set_state_mask_ms"] > 0.0
    assert timing["set_state_data_slice_ms"] > 0.0
    assert timing["set_state_data_reset_ms"] > 0.0
    assert timing["set_state_set_dof_vel_ms"] > 0.0
    assert timing["set_state_set_dof_pos_ms"] > 0.0
    assert timing["set_state_forward_kinematic_ms"] > 0.0
    assert timing["set_state_refresh_pose_cache_ms"] >= 0.0
    # A gravity payload was provided, so the DR sub-step should be non-zero.
    assert timing["set_state_reset_rand_ms"] > 0.0
    # MuJoCo-only sub-keys report 0.0 on motrix.
    assert timing["set_state_pool_reset_ms"] == 0.0
    assert timing["set_state_state_scatter_ms"] == 0.0


def test_motrix_set_state_without_randomization_leaves_reset_rand_zero() -> None:
    pytest.importorskip("motrixsim")

    from unilab.base.backend.motrix.backend import MotrixBackend

    backend = MotrixBackend(
        SceneCfg(model_file=_G1_MODEL_FILE),
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
    )
    nq = backend.get_dof_pos().shape[-1] + 7
    nv = backend.get_dof_vel().shape[-1] + 6
    qpos = _identity_qpos_mujoco(nq, xyz=(1.0, 2.0, 0.8))
    qvel = np.zeros((1, nv))

    result = backend.set_state(np.array([0]), qpos, qvel)

    timing = _assert_timing_dict_shape(result)
    # No payload → the fast-return branch in _apply_reset_randomization keeps
    # this sub-step at essentially zero.
    assert timing["set_state_reset_rand_ms"] < 1.0
