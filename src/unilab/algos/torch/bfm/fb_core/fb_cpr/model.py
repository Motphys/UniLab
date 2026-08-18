import copy
import typing as tp

import pydantic
import torch
from torch.amp import autocast

from ..fb.model import FBModel, FBModelArchiConfig, FBModelConfig
from ..nn_filter_models import DiscriminatorFilterArchiConfig, ForwardFilterArchiConfig
from ..nn_models import DiscriminatorArchiConfig, ForwardArchiConfig


class FBcprModelArchiConfig(FBModelArchiConfig):
    critic: ForwardArchiConfig | ForwardFilterArchiConfig = pydantic.Field(
        ForwardArchiConfig(), discriminator="name"
    )
    discriminator: DiscriminatorArchiConfig | DiscriminatorFilterArchiConfig = pydantic.Field(
        DiscriminatorArchiConfig(), discriminator="name"
    )


class FBcprModelConfig(FBModelConfig):
    name: tp.Literal["FBcprModel"] = "FBcprModel"
    archi: FBcprModelArchiConfig = FBcprModelArchiConfig()

    @property
    def object_class(self):
        return FBcprModel


class FBcprModel(FBModel):
    config_class = FBcprModelConfig

    def __init__(self, obs_space, action_dim, cfg: FBcprModelConfig):
        super().__init__(obs_space, action_dim, cfg)

        self.cfg: FBcprModelConfig = cfg
        self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)
        self._critic = cfg.archi.critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)

        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_critic = copy.deepcopy(self._critic)

    @torch.no_grad()
    def critic(
        self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor
    ):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._discriminator(self._normalize(obs), z)
