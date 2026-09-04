"""MicroDuck bilateral policy-contract tests."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from unilab.tasks.locomotion.microduck.symmetry import microduck_velocity_symmetry


def _mirror(actor: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    obs = TensorDict(
        {"policy": actor, "critic": torch.zeros((actor.shape[0], 76))},
        batch_size=[actor.shape[0]],
    )
    mirrored_obs, mirrored_action = microduck_velocity_symmetry(None, obs, action)
    assert mirrored_obs is not None
    assert mirrored_action is not None
    return mirrored_obs["policy"][actor.shape[0] :], mirrored_action[action.shape[0] :]


def test_microduck_velocity_mirror_is_an_involution() -> None:
    actor = torch.arange(61, dtype=torch.float32).unsqueeze(0)
    action = torch.arange(14, dtype=torch.float32).unsqueeze(0)

    mirrored_actor, mirrored_action = _mirror(actor, action)
    restored_actor, restored_action = _mirror(mirrored_actor, mirrored_action)

    torch.testing.assert_close(restored_actor, actor)
    torch.testing.assert_close(restored_action, action)


def test_microduck_velocity_mirror_flips_lateral_and_yaw_commands() -> None:
    actor = torch.zeros((1, 61))
    actor[0, 48:51] = torch.tensor((0.3, 0.2, 0.7))
    action = torch.zeros((1, 14))

    mirrored_actor, _ = _mirror(actor, action)

    torch.testing.assert_close(mirrored_actor[0, 48:51], torch.tensor((0.3, -0.2, -0.7)))
