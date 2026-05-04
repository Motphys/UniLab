from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch


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
