"""Real-CUDA correctness tests for the production ``mjwarp`` host profile.

These are slow by design.  They fail when explicitly invoked without the
``mjwarp`` extra or CUDA; the normal optional-backend unit lane deselects them
instead of treating an unavailable fake implementation as evidence.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow


def _require_cuda_mjwarp() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp correctness tests require an active CUDA Warp device")


def _scene() -> SceneCfg:
    from unilab.assets import ASSETS_ROOT_PATH

    return SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))


def _backend(num_envs: int = 3) -> Any:
    _require_cuda_mjwarp()
    return create_backend("mjwarp", _scene(), num_envs, 0.02 / 3.0, base_name="pelvis")


def _stand_state(backend: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.tile(backend.get_keyframe_qpos("stand"), (count, 1))
    qvel = np.zeros((count, backend.get_init_qvel().size), dtype=np.float32)
    return qpos.astype(np.float32), qvel


def test_real_cuda_init_reset_step() -> None:
    backend = _backend(2)
    assert backend.backend_type == "mjwarp"
    assert backend.num_actuators == 29
    assert backend.num_dof_vel == 29

    qpos, qvel = _stand_state(backend, 2)
    backend.set_state(np.asarray([0, 1], dtype=np.int32), qpos, qvel)
    before = backend.get_base_pos().copy()
    result = backend.step(np.tile(qpos[0, -backend.num_actuators :], (2, 1)), nsteps=1)

    assert set(result["timing"]) == {"control_upload_ms", "physics_ms", "host_cache_refresh_ms"}
    assert np.isfinite(backend.get_base_pos()).all()
    assert np.isfinite(backend.get_dof_pos()).all()
    assert np.isfinite(backend.get_sensor_data("torso_upvector")).all()
    assert not np.array_equal(backend.get_base_pos(), before)


def test_selected_row_reset_isolated() -> None:
    backend = _backend(4)
    qpos, qvel = _stand_state(backend, 4)
    qpos[:, 0] = np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=np.float32)
    qvel[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    backend.set_state(np.arange(4, dtype=np.int32), qpos, qvel)
    previous_pos = backend.get_base_pos().copy()
    previous_vel = backend.get_base_lin_vel().copy()

    rows = np.asarray([3, 1], dtype=np.int32)
    reset_qpos, reset_qvel = _stand_state(backend, 2)
    reset_qpos[:, 0] = np.asarray([0.45, -0.45], dtype=np.float32)
    reset_qvel[:, 0] = np.asarray([0.8, -0.8], dtype=np.float32)
    backend.set_state(rows, reset_qpos, reset_qvel)

    np.testing.assert_allclose(backend.get_base_pos()[rows, 0], reset_qpos[:, 0])
    np.testing.assert_allclose(backend.get_base_lin_vel()[rows, 0], reset_qvel[:, 0])
    complement = np.asarray([0, 2], dtype=np.int32)
    np.testing.assert_allclose(backend.get_base_pos()[complement], previous_pos[complement])
    np.testing.assert_allclose(backend.get_base_lin_vel()[complement], previous_vel[complement])
    assert np.isfinite(backend.get_sensor_data("pelvis_local_linvel")).all()


def _g1_reward_config():
    from unilab.envs.locomotion.g1.joystick import G1WalkRewardConfig

    return G1WalkRewardConfig(
        scales={
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.2,
            "feet_phase": 1.0,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.25,
            "base_height": -500.0,
            "orientation": -5.0,
            "action_rate": -0.01,
            "pose": -0.1,
        },
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.008,
        base_height_target=0.754,
        min_base_height=0.55,
        max_tilt_deg=25.0,
        pose_weights=[0.01, 1.0, 5.0, 0.01, 5.0, 5.0, 0.01, 1.0, 5.0, 0.01, 5.0, 5.0] + [50.0] * 17,
    )


def test_g1_host_profile_init_reset_step() -> None:
    _require_cuda_mjwarp()
    from unilab.base.registry import ensure_registries

    ensure_registries()
    env = cast(
        Any,
        registry.make(
            "G1WalkFlat",
            sim_backend="mjwarp",
            num_envs=2,
            env_cfg_override={
                "reward_config": _g1_reward_config(),
                "domain_rand": {"randomize_kp": False, "randomize_kd": False},
                "curriculum": {"enabled": False},
            },
        ),
    )
    try:
        initial = env.init_state()
        stepped = env.step(np.zeros((2, env.action_space.shape[0]), dtype=np.float32))
    finally:
        env.close()

    assert initial.obs["obs"].shape == (2, 98)
    assert stepped.obs["critic"].shape == (2, 101)
    assert np.isfinite(stepped.obs["obs"]).all()
    assert np.isfinite(stepped.reward).all()
