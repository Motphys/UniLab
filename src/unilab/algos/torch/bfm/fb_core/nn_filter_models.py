import typing as tp

import gymnasium
import torch
from torch import nn
from torch.utils._pytree import tree_map

from .nn_models import (
    Actor,
    ActorArchiConfig,
    BackwardArchiConfig,
    BackwardMap,
    Discriminator,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    ForwardMap,
    ResidualActorArchiConfig,
    SimpleActorArchiConfig,
)


def filter_space(obs_space: gymnasium.spaces.Dict, filter: list[int]) -> gymnasium.spaces.Dict:
    assert isinstance(obs_space, gymnasium.spaces.Dict), (
        "BackwardFilterArchiConfig requires a Dict observation space."
    )
    assert len(obs_space.spaces) == 1 and "proprio" in obs_space.spaces, (
        "Filter nn modules are currently expecting humenv observations only"
    )
    obs_space = obs_space.spaces["proprio"]
    filtered_space = gymnasium.spaces.Box(
        low=obs_space.low[filter], high=obs_space.high[filter], shape=(len(filter),)
    )
    filtered_space = gymnasium.spaces.Dict({"proprio": filtered_space})
    return filtered_space


class FilterBackwardMap(nn.Module):
    def __init__(self, nn_base: BackwardMap, filter: list[int]):
        super().__init__()
        self._nn_base = nn_base
        self._filter = filter

    def forward(self, obs):
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base(filtered_obs)


class BackwardFilterArchiConfig(BackwardArchiConfig):
    name: tp.Literal["BackwardFilterArchi"] = "BackwardFilterArchi"
    filter: list[int] = None

    def build(self, obs_space, z_dim):
        filtered_space = filter_space(obs_space, self.filter)
        return FilterBackwardMap(super().build(obs_space=filtered_space, z_dim=z_dim), self.filter)


class FilterForwardMap(nn.Module):
    def __init__(self, nn_base: ForwardMap, filter: list[int]):
        super().__init__()
        self._nn_base = nn_base
        self._filter = filter

    def forward(
        self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor
    ):
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base(filtered_obs, z, action)


class ForwardFilterArchiConfig(ForwardArchiConfig):
    name: tp.Literal["ForwardFilterArchi"] = "ForwardFilterArchi"
    filter: list[int] = None

    def build(self, obs_space, z_dim, action_dim, output_dim=None):
        filtered_space = filter_space(obs_space, self.filter)
        return FilterForwardMap(
            super().build(
                obs_space=filtered_space, z_dim=z_dim, action_dim=action_dim, output_dim=output_dim
            ),
            self.filter,
        )


class FilterActor(nn.Module):
    def __init__(self, nn_base: Actor, filter: list[int], filter_z: bool):
        super().__init__()
        self._nn_base = nn_base
        self._filter = filter
        self._filter_z = filter_z

    def forward(self, obs, z, std):
        if self._filter_z:
            z = z[:, self._filter]
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base(filtered_obs, z, std)


class ActorFilterArchiConfig(ActorArchiConfig):
    name: tp.Literal["ActorFilterArchi"] = "ActorFilterArchi"
    filter: list[int] = None
    filter_z: bool = False

    def build(self, obs_space, z_dim, action_dim):
        if self.filter_z:
            z_dim = len(self.filter)
        filtered_space = filter_space(obs_space, self.filter)
        return FilterActor(
            super().build(obs_space=filtered_space, z_dim=z_dim, action_dim=action_dim),
            self.filter,
            self.filter_z,
        )


class SimpleActorFilterArchiConfig(SimpleActorArchiConfig):
    name: tp.Literal["SimpleActorFilterArchi"] = "SimpleActorFilterArchi"
    filter: list[int] = None
    filter_z: bool = False

    def build(self, obs_space, z_dim, action_dim) -> "Actor":
        if self.filter_z:
            z_dim = len(self.filter)
        filtered_space = filter_space(obs_space, self.filter)
        return FilterActor(
            super().build(obs_space=filtered_space, z_dim=z_dim, action_dim=action_dim),
            self.filter,
            self.filter_z,
        )


class ResidualActorFilterArchiConfig(ResidualActorArchiConfig):
    name: tp.Literal["ResidualActorFilterArchi"] = "ResidualActorFilterArchi"
    filter: list[int] = None
    filter_z: bool = False

    def build(self, obs_space, z_dim, action_dim) -> "Actor":
        if self.filter_z:
            z_dim = len(self.filter)
        filtered_space = filter_space(obs_space, self.filter)
        return FilterActor(
            super().build(obs_space=filtered_space, z_dim=z_dim, action_dim=action_dim),
            self.filter,
            self.filter_z,
        )


class FilterDiscriminator(nn.Module):
    def __init__(self, nn_base: Discriminator, filter: list[int]):
        super().__init__()
        self._nn_base = nn_base
        self._filter = filter

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base.forward(filtered_obs, z)

    def compute_logits(
        self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor
    ) -> torch.Tensor:
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base.compute_logits(filtered_obs, z)

    def compute_reward(
        self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base.compute_reward(filtered_obs, z, eps)


class DiscriminatorFilterArchiConfig(DiscriminatorArchiConfig):
    name: tp.Literal["DiscriminatorFilterArchi"] = "DiscriminatorFilterArchi"
    filter: list[int] = None

    def build(self, obs_space, z_dim):
        filtered_space = filter_space(obs_space, self.filter)
        return FilterDiscriminator(
            super().build(obs_space=filtered_space, z_dim=z_dim), self.filter
        )
