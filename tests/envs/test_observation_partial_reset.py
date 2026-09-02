"""Row-scoped partial-reset observation rebuild tests (issue #1259 R2).

On the partial-reset path the observation manager returns only the reset rows
instead of a full batch that the env then slices. These tests pin the env-level
contract:

- reset observations arrive row-shaped, are finite, and are scattered into the
  env state at exactly the reset rows;
- untouched rows of ``state.obs`` keep their per-step values bit-identically;
- a subsequent step recomputes full-batch observations normally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg

_ROOT = Path(__file__).parents[2]


def _make_env(config_root: str, task: str, backend: str, identity: str, num_envs: int):
    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_ROOT / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        owner = compose("config", overrides=[f"task={task}/{backend}"])
    cfg = registry.materialize_env_config(identity)
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    override = BackendAdapter(
        owner, root_dir=_ROOT, algo_name=config_root
    ).build_task_env_cfg_override()
    apply_cfg_overrides(cfg, override)
    return registry.make(
        identity, num_envs=num_envs, sim_backend=backend, env_cfg_override=override
    )


def test_observation_partial_reset_row_contract() -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco_uni.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco_uni.batch_env not available")

    num_envs = 4
    env = _make_env("sac", "g1_motion_tracking", "mujoco", "G1MotionTrackingSAC", num_envs)
    try:
        env.init_state()
        action_dim = 29
        rng = np.random.default_rng(7)
        for _ in range(5):
            env.step((0.1 * rng.standard_normal((num_envs, action_dim))).astype(np.float32))

        assert env.state is not None
        obs_before = {name: values.copy() for name, values in env.state.obs.items()}
        reset_ids = np.array([0, 2], dtype=np.int32)
        keep_ids = np.array([1, 3], dtype=np.int32)
        reset_obs, _ = env.reset(env_ids=reset_ids)

        for group, values in reset_obs.items():
            assert values.shape == (len(reset_ids), obs_before[group].shape[1])
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(env.state.obs[group][reset_ids], values)
            np.testing.assert_array_equal(
                env.state.obs[group][keep_ids],
                obs_before[group][keep_ids],
                err_msg=f"untouched rows changed for group {group}",
            )

        # The next per-step compute rebuilds full-batch observations normally.
        env.step((0.1 * rng.standard_normal((num_envs, action_dim))).astype(np.float32))
        for group, values in env.state.obs.items():
            assert values.shape == obs_before[group].shape
            assert np.isfinite(values).all()
    finally:
        env.close()
