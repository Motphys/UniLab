"""Slow integration tests for OffPolicyRunner (FastSAC/FastTD3).

Requires MuJoCo to be installed. Run with:
    uv run pytest -m slow -v
"""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("mujoco")

from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.algos.torch.offpolicy.runner import OffPolicyRunner
from unilab.config.locomotion_params import offpolicy_config


def _make_sac_runner(env_name: str) -> OffPolicyRunner:
    cfg = offpolicy_config("sac", env_name).to_dict()
    obs_dim = 8
    action_dim = 3

    learner = FastSACLearner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device="cpu",
        hidden_dim=cfg.get("actor_hidden_dim", 64),
        use_layer_norm=False,
    )

    runner = OffPolicyRunner(
        learner=learner,
        env_name=env_name,
        algo_type="sac",
        num_envs=4,
        replay_buffer_n=8,
        batch_size=16,
        warmup_steps=0,
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
def test_offpolicy_runner_close_is_idempotent(mock_env_name):
    runner = _make_sac_runner(mock_env_name)
    runner.close()
    runner.close()  # must not raise
