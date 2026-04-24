"""Slow integration tests for OffPolicyRunner (FastSAC/FastTD3).

Requires MuJoCo to be installed. Run with:
    uv run pytest -m slow -v           # init + close + full training iteration
"""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("mujoco")

from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.algos.torch.fast_td3.learner import FastTD3Learner
from unilab.algos.torch.offpolicy.runner import OffPolicyRunner
from unilab.config.structured_configs import SACConfig


def _make_sac_runner(env_name: str) -> OffPolicyRunner:
    cfg = SACConfig().to_dict()
    obs_dim = 8
    action_dim = 3

    learner = FastSACLearner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=obs_dim,
        device="cpu",
        actor_hidden_dim=cfg.get("actor_hidden_dim", 64),
        critic_hidden_dim=cfg.get("critic_hidden_dim", 64),
        use_layer_norm=False,
    )

    runner = OffPolicyRunner(
        learner=learner,
        env_name=env_name,
        algo_type="sac",
        num_envs=4,
        replay_buffer_n=8,
        batch_size=16,
        learning_starts=0,
        updates_per_step=1,
        device="cpu",
    )
    return runner


@pytest.mark.slow
def test_offpolicy_runner_sac_init_no_crash(mock_env_name):
    runner = _make_sac_runner(mock_env_name)
    runner.close()


@pytest.mark.slow
def test_offpolicy_runner_sac_learn_two_iterations(mock_env_name):
    runner = _make_sac_runner(mock_env_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        runner.learn(max_iterations=2, save_interval=0, log_dir=tmpdir)
    runner.close()


@pytest.mark.slow
def test_offpolicy_runner_sac_close_is_idempotent(mock_env_name):
    runner = _make_sac_runner(mock_env_name)
    runner.close()
    runner.close()  # must not raise


# ---------------------------------------------------------------------------
# FastTD3
# ---------------------------------------------------------------------------


def _make_td3_runner(env_name: str) -> OffPolicyRunner:
    obs_dim = 8
    action_dim = 3

    learner = FastTD3Learner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=obs_dim,
        num_envs=4,
        device="cpu",
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        num_atoms=11,
        obs_normalization=False,
    )

    runner = OffPolicyRunner(
        learner=learner,
        env_name=env_name,
        algo_type="td3",
        num_envs=4,
        replay_buffer_n=8,
        batch_size=16,
        learning_starts=0,
        updates_per_step=1,
        policy_frequency=1,
        device="cpu",
    )
    return runner


@pytest.mark.slow
def test_offpolicy_runner_td3_init_no_crash(mock_env_name):
    runner = _make_td3_runner(mock_env_name)
    runner.close()


@pytest.mark.slow
def test_offpolicy_runner_td3_learn_two_iterations(mock_env_name):
    runner = _make_td3_runner(mock_env_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        runner.learn(max_iterations=2, save_interval=0, log_dir=tmpdir)
    runner.close()


@pytest.mark.slow
def test_offpolicy_runner_td3_close_is_idempotent(mock_env_name):
    runner = _make_td3_runner(mock_env_name)
    runner.close()
    runner.close()  # must not raise
