from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch
from tensordict import TensorDict


def test_hora_rsl_wrapper_uses_explicit_np_env_state_contract() -> None:
    """HORA wrapper must not probe required NpEnvState fields dynamically."""
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper

    source = textwrap.dedent(inspect.getsource(HoraRslRlVecEnvWrapper.step))
    tree = ast.parse(source)
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"getattr", "hasattr"}:
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        if node.args[0].id == "state":
            forbidden_calls.append(node.func.id)

    assert forbidden_calls == []


def test_hora_appo_learner_derives_priv_info_from_critic_contract() -> None:
    from unilab.algos.torch.hora.appo_learner import _derive_priv_info_from_critic

    actor_obs = torch.zeros((2, 3, 4), dtype=torch.float32)
    priv_info = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    critic_obs = torch.cat([actor_obs, priv_info], dim=-1)

    torch.testing.assert_close(
        _derive_priv_info_from_critic(actor_obs, critic_obs, context="test"),
        priv_info,
    )

    with pytest.raises(ValueError, match="privileged tail"):
        _derive_priv_info_from_critic(actor_obs, actor_obs, context="test")


def _make_hora_appo_learner(**algorithm_overrides):
    from unilab.algos.torch.hora.appo_learner import HoraAPPOLearner
    from unilab.algos.torch.hora.models import (
        HoraActorModel,
        HoraCriticModel,
        HoraSharedActorCritic,
    )

    obs = TensorDict(
        {
            "actor": torch.zeros(4, 5),
            "priv_info": torch.zeros(4, 2),
        },
        batch_size=4,
    )
    shared = HoraSharedActorCritic(
        obs_dim=5,
        action_dim=3,
        priv_info_dim=2,
        priv_info_embed_dim=2,
        actor_hidden_dims=(8,),
        priv_mlp_hidden_dims=(4, 2),
    )
    actor = HoraActorModel(obs, {}, "actor", 3, shared_model=shared)
    critic = HoraCriticModel(obs, {}, "critic", 1, shared_model=shared)
    kwargs = {
        "actor": actor,
        "critic": critic,
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
        "device": "cpu",
    }
    kwargs.update(algorithm_overrides)
    return HoraAPPOLearner(**kwargs)


def test_hora_appo_learner_uses_one_shared_actor_critic_core() -> None:
    learner = _make_hora_appo_learner()

    assert learner.actor.shared is learner.critic.shared


def test_hora_appo_runner_builds_shared_actor_critic_core() -> None:
    from unilab.algos.torch.hora.appo_runner import HoraAPPORunner

    runner = HoraAPPORunner.__new__(HoraAPPORunner)
    runner.num_envs = 4
    runner.obs_dim = 5
    runner.action_dim = 3
    runner.priv_info_dim = 2
    runner.device = "cpu"
    runner.rl_cfg = {
        "obs_groups": {
            "actor": {"actor": 5, "priv_info": 2},
            "critic": {"actor": 5, "priv_info": 2},
        },
        "actor": {
            "class_name": "unilab.algos.torch.hora:HoraActorModel",
            "hidden_dims": [8],
            "priv_info_embed_dim": 2,
            "priv_mlp_hidden_dims": [4, 2],
        },
        "critic": {
            "class_name": "unilab.algos.torch.hora:HoraCriticModel",
            "priv_info_embed_dim": 2,
            "priv_mlp_hidden_dims": [4, 2],
        },
        "algorithm": {
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
        },
    }

    learner = runner._build_learner()

    assert learner.actor.shared is learner.critic.shared


def test_hora_appo_combined_optimizer_has_unique_parameters() -> None:
    learner = _make_hora_appo_learner()

    combined_ids = [id(param) for param in learner._combined_params]
    optimizer_ids = [
        id(param) for group in learner.optimizer.param_groups for param in group["params"]
    ]

    assert len(combined_ids) == len(set(combined_ids))
    assert optimizer_ids == combined_ids
    assert len(optimizer_ids) == len(set(optimizer_ids))


def test_hora_appo_update_uses_joint_shared_optimizer() -> None:
    torch.manual_seed(13)
    learner = _make_hora_appo_learner(learning_rate=1e-3)
    observations = torch.randn(2, 3, 5)
    priv_info = torch.randn(2, 3, 2)
    last_obs = torch.randn(3, 5)
    last_priv_info = torch.randn(3, 2)
    batch = {
        "observations": observations,
        "critic": torch.cat([observations, priv_info], dim=-1),
        "actions": torch.randn(2, 3, 3),
        "actions_log_prob": torch.zeros(2, 3),
        "rewards": torch.randn(2, 3),
        "dones": torch.zeros(2, 3),
        "last_obs": last_obs,
        "last_critic": torch.cat([last_obs, last_priv_info], dim=-1),
    }

    trunk_before = [param.detach().clone() for param in learner.actor.shared.trunk.parameters()]

    learner.process_batch(batch)
    metrics = learner.update(batch)

    trunk_after = list(learner.actor.shared.trunk.parameters())

    assert metrics["appo/updates_executed"] == pytest.approx(1.0)
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(trunk_before, trunk_after, strict=True)
    )
