import typing as tp

import gymnasium
import numpy as np
import pydantic
import torch
from torch import nn

from .base import BaseConfig


class IdentityInputFilterConfig(BaseConfig):
    def build(self, space):
        nn_module = nn.Identity()

        nn_module.output_space = space
        return nn_module


class DictInputFilterConfig(BaseConfig):
    key: str | list[str] | tuple[str, ...]

    def build(self, space):
        if isinstance(self.key, str):
            return DictInputFilter(space, self)
        else:
            if len(self.key) == 1:
                return DictInputFilter(space, DictInputFilterConfig(key=self.key[0]))

            return DictInputConcatFilter(space, self)


class DictInputFilter(nn.Module):
    def __init__(self, space, cfg: DictInputFilterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.output_space = self.get_output_space(space, cfg)

    @classmethod
    def get_output_space(cls, space, cfg: DictInputFilterConfig):
        assert isinstance(space, gymnasium.spaces.Dict), "space must be a Dict space"
        assert cfg.key in space.spaces, (
            f"key {cfg.key} not found in space of keys {space.spaces.keys()}"
        )
        return space.spaces[cfg.key]

    def forward(self, _input: torch.Tensor | dict[str, torch.Tensor]):
        if isinstance(_input, dict):
            _input = _input[self.cfg.key]

        return _input


class DictInputConcatFilter(nn.Module):
    def __init__(self, space, cfg: DictInputFilterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.output_space = self.get_output_space(space, cfg)

    @classmethod
    def get_output_space(cls, input_space, cfg: DictInputFilterConfig):
        assert isinstance(input_space, gymnasium.spaces.Dict), "input_space must be a Dict space"
        assert all(key in input_space.spaces for key in cfg.key), (
            f"keys {cfg.key} not found in obs_space {input_space.spaces.keys()}"
        )
        assert all(isinstance(input_space[key], gymnasium.spaces.Box) for key in cfg.key), (
            "All keys must be Box spaces"
        )
        assert all(len(input_space[key].shape) == 1 for key in cfg.key), (
            f"All key spaces must have 1D shape, got {[input_space[key].shape for key in cfg.key]}"
        )
        first_dtype = input_space.spaces[cfg.key[0]].dtype
        assert all(input_space.spaces[key].dtype == first_dtype for key in cfg.key), (
            f"All keys must have the same dtype {input_space.spaces[cfg.key[0]].dtype}"
        )
        return gymnasium.spaces.Box(
            low=np.concatenate([input_space.spaces[key].low for key in cfg.key]),
            high=np.concatenate([input_space.spaces[key].high for key in cfg.key]),
            dtype=input_space.spaces[cfg.key[0]].dtype,
        )

    def forward(self, _input: torch.Tensor | dict[str, torch.Tensor]):
        if isinstance(_input, dict):
            _input = torch.cat([_input[key] for key in self.cfg.key], dim=-1)

        return _input


NNFilter = tp.Annotated[
    tp.Union[
        IdentityInputFilterConfig,
        DictInputFilterConfig,
    ],
    pydantic.Field(discriminator="name"),
]
