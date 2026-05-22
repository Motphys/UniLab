"""IK unit tests for Go2Arm base environment."""

from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.go2_arm.base import Go2ArmBaseCfg, Go2ArmBaseEnv


class _IkHarness(Go2ArmBaseEnv):
    def apply_action(self, actions, state):
        raise NotImplementedError

    def update_state(self, state):
        raise NotImplementedError


class _FakeBackend:
    def __init__(self, jacp: np.ndarray, jacr: np.ndarray):
        self._jacp = jacp
        self._jacr = jacr

    def get_site_jacobian_w(self, site_id: int, dof_indices: np.ndarray):
        del site_id, dof_indices
        return self._jacp, self._jacr

    def get_sensor_data(self, name: str) -> np.ndarray:
        del name
        return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)


def _ik_env(*, use_orientation: bool, orientation_mode: str) -> Go2ArmBaseEnv:
    jacp = np.asarray(
        [
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    jacr = np.asarray(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    env = object.__new__(_IkHarness)
    cfg = Go2ArmBaseCfg()
    cfg.ik.use_orientation = use_orientation
    cfg.ik.orientation_mode = orientation_mode
    cfg.ik.damping = 0.0
    cfg.ik.dq_clip = 0.0
    env._cfg = cfg
    env._backend = _FakeBackend(jacp, jacr)
    env._ee_site_id = 0
    env._arm_jacobian_dof_indices = np.arange(6, dtype=np.int32)
    return env


def test_go2_arm_ik_zero_error_orientation_regularizes_rotation_nullspace():
    goal = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    curr = np.zeros((1, 3), dtype=np.float64)

    position_only = _ik_env(use_orientation=False, orientation_mode="target").compute_arm_ik_delta(
        goal,
        curr,
    )
    zero_error = _ik_env(
        use_orientation=True,
        orientation_mode="zero_error",
    ).compute_arm_ik_delta(goal, curr)

    np.testing.assert_allclose(position_only, [[0.5, 1.0, 1.5, 0.5, 1.0, 1.5]])
    np.testing.assert_allclose(zero_error, [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])


def test_go2_arm_ik_rejects_unknown_orientation_mode():
    env = _ik_env(use_orientation=True, orientation_mode="invalid")
    goal = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    curr = np.zeros((1, 3), dtype=np.float64)

    try:
        env.compute_arm_ik_delta(goal, curr)
    except ValueError as exc:
        assert "ik.orientation_mode" in str(exc)
    else:
        raise AssertionError("expected invalid ik.orientation_mode to raise ValueError")
