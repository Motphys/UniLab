"""Full-DR smoke for the MicroDuck SAC mjwarp owner (issue #1401).

The owner inherits ``base_com`` (reset body_ipos randomization) and
``push_robot`` (interval velocity kicks) from the shared base config; this test
builds the env on the mjwarp backend, verifies the reset payload visibly
reaches the per-world device model, and steps past the push interval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.base.config_adapter import BackendAdapter
from unilab.tasks.locomotion.microduck.deploy_contract import MICRODUCK_NUM_ACTION

pytestmark = pytest.mark.slow

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "sac"


def _compose_mjwarp_owner() -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=microduck_velocity_flat/mjwarp"])


def test_microduck_mjwarp_owner_enables_base_com_and_push_events() -> None:
    cfg = _compose_mjwarp_owner()
    assert cfg.training.sim_backend == "mjwarp"
    events = cfg.env.events
    assert events.base_com is not None
    assert events.base_com.func == "unilab.envs.mdp.randomize_rigid_body_com"
    assert events.push_robot is not None
    assert events.push_robot.func == "unilab.envs.mdp.push_by_setting_velocity"
    assert events.push_robot.mode == "interval"
    # Legacy-recipe DR events added by #1402 must also reach the mjwarp owner.
    assert events.head_com is not None
    assert events.head_com.func == "unilab.envs.mdp.randomize_rigid_body_com"
    assert events.foot_friction is not None
    assert events.foot_friction.func == "unilab.envs.mdp.geom_friction"
    assert events.randomize_armature is not None
    assert events.randomize_armature.func == "unilab.envs.mdp.joint_armature"


def test_microduck_mjwarp_full_dr_reset_and_interval_smoke() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("microduck mjwarp DR smoke requires an active CUDA Warp device")

    num_envs = 4
    cfg = _compose_mjwarp_owner()
    registry.ensure_registries()
    env = cast(
        Any,
        registry.make(
            "MicroduckVelocityFlat",
            sim_backend="mjwarp",
            num_envs=num_envs,
            env_cfg_override=BackendAdapter(
                cfg,
                root_dir=ROOT_DIR,
                algo_name="sac",
            ).build_task_env_cfg_override(),
        ),
    )
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert np.isfinite(obs["obs"]).all()
        assert np.isfinite(obs["critic"]).all()

        # The base_com reset payload reached the per-world device model rows.
        body_ipos = np.asarray(env._backend._device_model.body_ipos.numpy())
        base_id = env._backend._base_body_id
        assert base_id is not None
        world_ipos = body_ipos[:, base_id, :]
        assert np.ptp(world_ipos, axis=0).max() > 1e-5

        # The #1402 foot-friction (abs, [0.7, 1.3]) and armature (scale,
        # [0.9, 1.1]) reset payloads also reached the per-world device model.
        geom_friction = np.asarray(env._backend._device_model.geom_friction.numpy())
        foot_geom_ids = [
            env._backend.get_geom_id(name)
            for name in ("left_foot_collision", "right_foot_collision")
        ]
        foot_friction = geom_friction[:, foot_geom_ids, 0]
        assert np.all(foot_friction >= 0.7 - 1e-6)
        assert np.all(foot_friction <= 1.3 + 1e-6)
        dof_armature = np.asarray(env._backend._device_model.dof_armature.numpy())
        assert np.ptp(dof_armature, axis=0).max() > 0.0

        # Step past the 3s push interval (150 control steps at ctrl_dt=0.02)
        # so both the reset and interval DR paths execute on device.
        action = np.zeros((num_envs, MICRODUCK_NUM_ACTION), dtype=np.float32)
        for _ in range(155):
            state = env.step(action)
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()
