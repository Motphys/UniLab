import copy
import math
import typing as tp

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..base_model import BaseModel, BaseModelConfig
from ..nn_filter_models import (
    ActorFilterArchiConfig,
    BackwardFilterArchiConfig,
    ForwardFilterArchiConfig,
    ResidualActorFilterArchiConfig,
    SimpleActorFilterArchiConfig,
)
from ..nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    ForwardArchiConfig,
    ResidualActorArchiConfig,
    SimpleActorArchiConfig,
    eval_mode,
)
from ..normalizers import ObsNormalizerConfig
from ..pytree_utils import tree_get_batch_size


class FBModelArchiConfig(BaseConfig):
    z_dim: int = 100
    norm_z: bool = True
    f: ForwardArchiConfig | ForwardFilterArchiConfig = pydantic.Field(
        ForwardArchiConfig(), discriminator="name"
    )
    b: BackwardArchiConfig | BackwardFilterArchiConfig = pydantic.Field(
        BackwardArchiConfig(), discriminator="name"
    )

    actor: (
        ActorArchiConfig
        | ActorFilterArchiConfig
        | SimpleActorArchiConfig
        | ResidualActorArchiConfig
        | SimpleActorFilterArchiConfig
        | ResidualActorFilterArchiConfig
    ) = pydantic.Field(SimpleActorArchiConfig(), discriminator="name")


class FBModelConfig(BaseModelConfig):
    name: tp.Literal["FBModel"] = "FBModel"

    archi: FBModelArchiConfig = FBModelArchiConfig()
    obs_normalizer: ObsNormalizerConfig = ObsNormalizerConfig()
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    amp: bool = False

    def build(self, obs_space, action_dim) -> "FBModel":
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBModel


class FBModel(BaseModel):
    config_class = FBModelConfig

    def __init__(self, obs_space, action_dim, cfg: FBModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: FBModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device
        self.amp_dtype = torch.bfloat16

        self._backward_map = arch.b.build(obs_space, arch.z_dim)
        self._forward_map = arch.f.build(obs_space, arch.z_dim, action_dim)
        self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:

        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(
        self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor
    ):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, std: float):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._actor(self._normalize(obs), z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(
        self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True
    ) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()

    def reward_inference(
        self,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        reward: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            batch_size = tree_get_batch_size(next_obs)
            num_batches = int(np.ceil(batch_size / self.cfg.inference_batch_size))
            z = 0
            wr = reward if weight is None else reward * weight
            for i in range(num_batches):
                start_idx, end_idx = (
                    i * self.cfg.inference_batch_size,
                    (i + 1) * self.cfg.inference_batch_size,
                )
                next_obs_slice = tree_map(lambda x: x[start_idx:end_idx].to(self.device), next_obs)
                B = self.backward_map(next_obs_slice)
                z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(
        self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor
    ) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)
