from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.offpolicy.worker import (
    resolve_offpolicy_actor_priv_info,
    sample_offpolicy_actions,
)


class _DummyActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        assert dones is not None
        self.calls.append((obs.clone(), dones.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


@pytest.mark.parametrize("algo_type", ["sac", "td3", "flashsac"])
def test_sample_offpolicy_actions_uses_actor_explore(algo_type: str) -> None:
    actor = _DummyActor()
    obs = torch.zeros(4, 5)
    dones = torch.zeros(4)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type=algo_type,
        obs_torch=obs,
        prev_dones_torch=dones,
    )

    assert len(actor.calls) == 1
    assert actor.calls[0][2] is False
    assert actions.shape == (4, 3)


def test_sample_offpolicy_actions_rejects_unknown_algo() -> None:
    actor = _DummyActor()

    with pytest.raises(ValueError, match="Unsupported off-policy algo_type"):
        sample_offpolicy_actions(
            actor=actor,
            algo_type="unknown",
            obs_torch=torch.zeros(2, 4),
            prev_dones_torch=torch.zeros(2),
        )


class _DummyHoraActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.calls.append((obs.clone(), priv_info.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def test_sample_offpolicy_actions_passes_hora_priv_info() -> None:
    actor = _DummyHoraActor()
    obs = torch.zeros(4, 5)
    priv_info = torch.randn(4, 2)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type="hora_sac",
        obs_torch=obs,
        prev_dones_torch=torch.zeros(4),
        priv_info_torch=priv_info,
    )

    assert actions.shape == (4, 3)
    assert len(actor.calls) == 1
    torch.testing.assert_close(actor.calls[0][1], priv_info)


def test_resolve_offpolicy_actor_priv_info_prefers_explicit_info() -> None:
    import numpy as np

    obs = np.zeros((2, 3), dtype=np.float32)
    critic_tail = np.ones((2, 2), dtype=np.float32)
    critic = np.concatenate([obs, critic_tail], axis=1)
    explicit = np.full((2, 2), 7.0, dtype=np.float32)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="hora_sac",
        obs_np=obs,
        critic_np=critic,
        info={"critic_info": explicit},
    )

    np.testing.assert_allclose(resolved, explicit)


def test_resolve_offpolicy_actor_priv_info_uses_critic_tail() -> None:
    import numpy as np

    obs = np.zeros((2, 3), dtype=np.float32)
    critic_tail = np.arange(4, dtype=np.float32).reshape(2, 2)
    critic = np.concatenate([obs, critic_tail], axis=1)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="hora_sac",
        obs_np=obs,
        critic_np=critic,
        info={},
    )

    np.testing.assert_allclose(resolved, critic_tail)
