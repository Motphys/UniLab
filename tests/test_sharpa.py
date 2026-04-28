from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.common.rotation import np_quat_apply
from unilab.envs.manipulation.sharpa_inhand.base import SharpaDomainRandConfig
from unilab.envs.manipulation.sharpa_inhand.grasp_gen import (
    SharpaInhandRotationGraspCfg,
    SharpaInhandRotationGraspEnv,
)
from unilab.envs.manipulation.sharpa_inhand.rotation import (
    SharpaInhandRotationEnv,
    sample_random_quaternion,
)


def test_sharpa_gravity_direction_randomization_matches_rotated_gravity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gravity-direction DR should rotate a fixed-magnitude downward vector.

    Args:
        monkeypatch: Pytest helper used to replace quaternion sampling.

    Returns:
        None. The assertions validate the exact gravity vectors produced.
    """
    fixed_quat = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    def _fake_sample_random_quaternion(num_envs: int) -> np.ndarray:
        assert num_envs == 2
        return fixed_quat.copy()

    monkeypatch.setitem(
        SharpaInhandRotationEnv._build_gravity_direction_randomization.__globals__,
        "sample_random_quaternion",
        _fake_sample_random_quaternion,
    )

    env = SimpleNamespace(
        _cfg=SimpleNamespace(
            domain_rand=SimpleNamespace(
                randomize_gravity_direction=True,
                gravity_direction_magnitude=9.81,
            )
        )
    )

    gravity = SharpaInhandRotationEnv._build_gravity_direction_randomization(env, batch_size=2)

    assert gravity is not None
    expected = np_quat_apply(
        fixed_quat,
        np.asarray([[0.0, 0.0, -9.81]], dtype=np.float64),
    )
    np.testing.assert_allclose(gravity, expected)
    np.testing.assert_allclose(np.linalg.norm(gravity, axis=1), np.full((2,), 9.81))


def test_sharpa_gravity_direction_randomization_disabled_returns_none() -> None:
    env = SimpleNamespace(
        _cfg=SimpleNamespace(
            domain_rand=SimpleNamespace(
                randomize_gravity_direction=False,
                gravity_direction_magnitude=9.81,
            )
        )
    )

    gravity = SharpaInhandRotationEnv._build_gravity_direction_randomization(env, batch_size=3)

    assert gravity is None


def test_sharpa_grasp_env_rejects_gravity_randomization() -> None:
    """Sharpa grasp collection should reject gravity DR explicitly.

    Args:
        None.

    Returns:
        None. The assertion validates the grasp-task contract.
    """
    cfg = SharpaInhandRotationGraspCfg(
        domain_rand=SharpaDomainRandConfig(randomize_gravity_direction=True)
    )

    with pytest.raises(ValueError, match="does not support gravity randomization"):
        SharpaInhandRotationGraspEnv(cfg, num_envs=1, backend_type="mujoco")
