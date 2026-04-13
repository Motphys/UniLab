from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.offpolicy.worker import sample_offpolicy_actions


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
