"""Real-CUDA correctness tests for the production ``mjwarp`` host profile.

These are slow by design.  They fail when explicitly invoked without the
``mjwarp`` extra or CUDA; the normal optional-backend unit lane deselects them
instead of treating an unavailable fake implementation as evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.config_adapter import BackendAdapter
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow
REPO_ROOT = Path(__file__).resolve().parents[2]


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
    root_layout = backend.get_root_state_layout("pelvis")
    assert root_layout.qpos_indices == tuple(range(7))
    assert root_layout.qvel_indices == tuple(range(6))

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


@pytest.mark.parametrize(
    ("config_group", "overrides", "algo_name"),
    [
        pytest.param("ppo", ["task=g1_walk_flat/mjwarp"], "ppo", id="ppo"),
        pytest.param(
            "sac",
            ["task=g1_walk_flat/mjwarp"],
            "sac",
            id="sac",
        ),
    ],
)
def test_g1_walk_flat_owner_one_step(
    config_group: str,
    overrides: list[str],
    algo_name: str,
) -> None:
    _require_cuda_mjwarp()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf" / config_group), version_base="1.3"
    ):
        cfg = compose("config", overrides=overrides)

    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=REPO_ROOT,
        algo_name=algo_name,
    ).build_task_env_cfg_override()
    registry.ensure_registries()
    env = registry.make(
        str(cfg.training.task_name),
        sim_backend=str(cfg.training.sim_backend),
        env_cfg_override=env_cfg_override,
        num_envs=2,
    )

    action_dim = int(env.action_space.shape[-1])
    state = env.step(np.zeros((2, action_dim), dtype=np.float32))

    assert set(state.obs) == {"obs", "critic"}
    assert state.obs["obs"].shape == (2, 98)
    assert state.obs["critic"].shape == (2, 101)
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()
    assert np.isfinite(state.reward).all()
