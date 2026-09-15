from __future__ import annotations

import pytest
from hydra.errors import ConfigCompositionException
from scripts.benchmark.env import benchmark_env_step as bench

from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

# Importing benchmark_env_step installs a process-wide create_backend patch for
# the benchmark script. These tests only build configs, so undo the patch to
# keep the global factory pristine for the rest of the pytest session.
bench._uninstall_mjwarp_patch()


def test_go2_flat_benchmark_uses_production_manager_owner() -> None:
    cfg = bench.TASK_CONFIGS["go2"].build_cfg("mujoco")

    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    assert list(cfg.actions) == ["joint_pos"]
    assert cfg.critic_observation_group == "critic"
    assert bench.TASK_CONFIGS["go2"].env_cls_factory() is make_manager_based_rl_env
    assert bench.DEFAULT_NUM_ENVS == 4096


def test_env_and_reward_overrides_use_hydra_composition() -> None:
    cfg = bench.TASK_CONFIGS["go2"].build_cfg(
        "mujoco",
        [
            "env.sim_dt=0.004",
            "reward.tracking_lin_vel.weight=2.25",
        ],
    )

    assert cfg.sim_dt == pytest.approx(0.004)
    assert cfg.rewards["tracking_lin_vel"].weight == pytest.approx(2.25)


def test_unknown_env_override_fails_in_hydra() -> None:
    with pytest.raises(ConfigCompositionException, match="env.not_a_real_field"):
        bench.TASK_CONFIGS["go2"].build_cfg(
            "mujoco",
            ["env.not_a_real_field=1"],
        )


@pytest.mark.parametrize(
    "override",
    ["training.sim_backend=motrix", "+training.sim_backend=motrix"],
)
def test_training_sim_backend_override_is_rejected(override: str) -> None:
    with pytest.raises(ValueError, match=r"task=<task>/<backend>"):
        bench._resolve_task_and_backend(["task=go2_joystick_flat/mujoco", override])


def test_only_owner_config_overrides_are_forwarded() -> None:
    overrides = [
        "task=go2_joystick_flat/mujoco",
        "env.scene.sim_dt=0.004",
        "reward.tracking_lin_vel.weight=2.25",
    ]

    assert bench._owner_config_overrides(overrides) == overrides[1:]
    with pytest.raises(ValueError, match="Unsupported benchmark config override"):
        bench._owner_config_overrides(["algo.learning_rate=0.1"])
