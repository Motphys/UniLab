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
    assert cfg.reward.base_height.params.target == pytest.approx(0.435)
    assert cfg.reward.torque_limits.params.soft_limit == pytest.approx(0.9)
    assert cfg.reward.feet_contact_forces.params.threshold == pytest.approx(200.0)


def test_a2arm_history_terms_follow_algo_history_overrides() -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    root = Path(__file__).resolve().parents[2]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(root / "conf" / "ppo_cse"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=a2arm_pos_force/mujoco",
                "algo.num_actor_history=4",
                "algo.num_critic_history=2",
            ],
        )

    assert cfg.env.observations.policy.terms.history.params.history_length == 4
    assert cfg.env.observations.critic.terms.history.params.history_length == 2
