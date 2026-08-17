from __future__ import annotations

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from scripts.benchmark.env import benchmark_env_step as bench


def test_go2w_rough_cfg_matches_ppo_owner_yaml() -> None:
    cfg = bench.TASK_CONFIGS["go2w_rough"].build_cfg("mujoco")
    owner_path = (
        Path(bench.ROOT_DIR) / "conf" / "ppo" / "task" / "go2w_joystick_rough" / "mujoco.yaml"
    )
    owner_cfg = OmegaConf.load(owner_path)

    assert cfg.reward_config.scales == OmegaConf.to_container(
        owner_cfg.reward.scales,
        resolve=True,
    )
    assert cfg.reward_config.tracking_sigma == owner_cfg.reward.tracking_sigma
    assert cfg.reward_config.base_height_target == owner_cfg.reward.base_height_target
    assert cfg.scene.model_file == owner_cfg.env.scene.model_file


def test_env_and_reward_overrides_use_hydra_composition() -> None:
    cfg = bench.TASK_CONFIGS["go2w_rough"].build_cfg(
        "mujoco",
        [
            "env.control_config.action_scale=0.125",
            "reward.tracking_sigma=${reward.base_height_target}",
        ],
    )

    assert cfg.control_config.action_scale == pytest.approx(0.125)
    assert cfg.reward_config.tracking_sigma == pytest.approx(0.4)
    assert isinstance(cfg.reward_config.tracking_sigma, float)


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
        "env.control_config.action_scale=0.125",
        "+reward.scales.custom_term=1.0",
    ]

    assert bench._owner_config_overrides(overrides) == overrides[1:]
    with pytest.raises(ValueError, match="Unsupported benchmark config override"):
        bench._owner_config_overrides(["algo.learning_rate=0.1"])
