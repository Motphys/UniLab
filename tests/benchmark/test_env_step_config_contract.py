from __future__ import annotations

import pytest
from hydra.errors import ConfigCompositionException
from scripts.benchmark.env import benchmark_env_step as bench

from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.tasks.locomotion.common.rough_manager_terms import QuadrupedRoughTerrainCfg


def test_go2w_flat_benchmark_uses_production_manager_owner() -> None:
    cfg = bench.TASK_CONFIGS["go2w"].build_cfg("mujoco")

    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    assert list(cfg.actions) == ["motor"]
    assert cfg.critic_observation_group == "critic"
    assert bench.TASK_CONFIGS["go2w"].env_cls_factory() is make_manager_based_rl_env
    assert bench.DEFAULT_NUM_ENVS == 4096


def test_go2w_rough_cfg_matches_ppo_owner_yaml() -> None:
    cfg = bench.TASK_CONFIGS["go2w_rough"].build_cfg("mujoco")

    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    assert cfg.scene is not None
    assert cfg.scene.terrain is not None
    assert isinstance(cfg.scene.terrain.generator, QuadrupedRoughTerrainCfg)
    assert cfg.scene.model_file.endswith("go2w_mujoco.xml")
    assert cfg.actions["motor"].wheel_action_scale == pytest.approx(5.0)
    assert cfg.rewards["tracking_lin_vel"].weight == pytest.approx(3.0)
    assert bench.TASK_CONFIGS["go2w_rough"].env_cls_factory() is make_manager_based_rl_env


def test_env_and_reward_overrides_use_hydra_composition() -> None:
    cfg = bench.TASK_CONFIGS["go2w_rough"].build_cfg(
        "mujoco",
        [
            "env.actions.motor.leg_action_scale=0.125",
            "reward.tracking_lin_vel.weight=2.25",
        ],
    )

    assert cfg.actions["motor"].leg_action_scale == pytest.approx(0.125)
    assert cfg.rewards["tracking_lin_vel"].weight == pytest.approx(2.25)


def test_unknown_env_override_fails_in_hydra() -> None:
    with pytest.raises(ConfigCompositionException, match="env.not_a_real_field"):
        bench.TASK_CONFIGS["go2w_rough"].build_cfg(
            "mujoco",
            ["env.not_a_real_field=1"],
        )


@pytest.mark.parametrize(
    "override",
    ["training.sim_backend=motrix", "+training.sim_backend=motrix"],
)
def test_training_sim_backend_override_is_rejected(override: str) -> None:
    with pytest.raises(ValueError, match=r"task=<task>/<backend>"):
        bench._resolve_task_and_backend(["task=go2w_joystick_rough/mujoco", override])


def test_only_owner_config_overrides_are_forwarded() -> None:
    overrides = [
        "task=go2w_joystick_rough/mujoco",
        "env.actions.motor.leg_action_scale=0.125",
        "reward.tracking_lin_vel.weight=2.25",
    ]

    assert bench._owner_config_overrides(overrides) == overrides[1:]
    with pytest.raises(ValueError, match="Unsupported benchmark config override"):
        bench._owner_config_overrides(["algo.learning_rate=0.1"])
