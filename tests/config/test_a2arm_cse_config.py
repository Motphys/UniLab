from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def test_a2arm_cse_owner_preserves_training_dimensions_and_tuning() -> None:
    config_dir = Path(__file__).parents[2] / "conf" / "ppo_cse"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose("config", overrides=["task=a2arm_pos_force/mujoco"])

    assert cfg.training.task_name == "A2ArmPosForce"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.sim_dt == pytest.approx(0.005)
    assert cfg.env.ctrl_dt == pytest.approx(0.02)
    assert cfg.algo.num_one_step_obs == 73
    assert cfg.algo.num_actor_history == 32
    assert cfg.algo.num_critic_history == 3
    assert cfg.algo.estimator.target_start == 0
    assert list(cfg.algo.estimator.target_group_sizes) == [3, 3, 3, 3]
    assert list(cfg.algo.estimator.target_weights) == [0.2, 0.2, 1.0, 1.0]
    assert cfg.algo.algorithm.learning_rate == pytest.approx(5.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-2)
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.max_iterations == 20000
