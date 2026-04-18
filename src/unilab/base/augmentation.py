from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


SymmetryObsLayout = tuple[tuple[str, int], ...]


class SymmetryAugmentation(Protocol):
    """Runtime symmetry augmentation contract owned by env/backend adapters."""

    batch_multiplier: int

    def augment_obs_and_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        obs_group: str = "obs",
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def mirror_obs(
        self,
        obs: torch.Tensor,
        *,
        obs_group: str = "obs",
    ) -> torch.Tensor: ...
