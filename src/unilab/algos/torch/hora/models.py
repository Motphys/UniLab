from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, GaussianDistribution


def _build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}")


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | tuple[int, ...], activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(_build_activation(activation))
            current_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProprioAdaptTConv(nn.Module):
    """Temporal adaptation encoder used by HORA stage-2 distillation."""

    def __init__(self, frame_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(frame_dim, frame_dim),
            nn.ReLU(inplace=True),
            nn.Linear(frame_dim, frame_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(frame_dim, frame_dim, kernel_size=9, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_dim, frame_dim, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_dim, frame_dim, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(frame_dim * 3, latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                fan_out = module.kernel_size[0] * module.out_channels
                module.weight.data.normal_(mean=0.0, std=(2.0 / fan_out) ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        x = self.channel_transform(proprio_hist)
        x = x.permute(0, 2, 1)
        x = self.temporal_aggregation(x)
        return self.low_dim_proj(x.flatten(1))


@dataclass
class HoraCoreOutput:
    policy_obs: torch.Tensor
    trunk_latent: torch.Tensor
    privileged_latent: torch.Tensor
    privileged_target: torch.Tensor


class HoraSharedActorCritic(nn.Module):
    """Shared-backbone HORA actor-critic with optional adaptation encoder."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        priv_info_dim: int,
        priv_info_embed_dim: int = 8,
        actor_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        priv_mlp_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 8),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
        use_student_encoder: bool = False,
        proprio_hist_len: int = 30,
        proprio_frame_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.priv_info_dim = int(priv_info_dim)
        self.priv_info_embed_dim = int(priv_info_embed_dim)
        self.use_student_encoder = bool(use_student_encoder)
        self.proprio_hist_len = int(proprio_hist_len)
        self.proprio_frame_dim = (
            int(proprio_frame_dim) if proprio_frame_dim is not None else self.obs_dim // 3
        )

        self.obs_normalizer = (
            EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        )
        self.priv_encoder = _MLP(self.priv_info_dim, list(priv_mlp_hidden_dims), activation)
        self.trunk = _MLP(self.obs_dim + self.priv_info_embed_dim, list(actor_hidden_dims), activation)
        self.value_head = nn.Linear(self.trunk.output_dim, 1)
        self.mu_head = nn.Linear(self.trunk.output_dim, self.action_dim)
        self.distribution = GaussianDistribution(
            self.action_dim,
            **(
                {
                    key: value
                    for key, value in (
                        distribution_cfg
                        if distribution_cfg is not None
                        else {"init_std": 1.0, "std_type": "scalar"}
                    ).items()
                    if key != "class_name"
                }
            ),
        )
        self.adapt_tconv = (
            ProprioAdaptTConv(self.proprio_frame_dim, self.priv_info_embed_dim)
            if self.use_student_encoder
            else None
        )
        self._init_linear_biases()

    def _init_linear_biases(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _normalize_actor_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.obs_normalizer(actor_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(obs["actor"])

    def _zero_privileged_latent(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((batch_size, self.priv_info_embed_dim), device=device, dtype=dtype)

    def encode_privileged_info(self, priv_info: torch.Tensor | None) -> torch.Tensor:
        if priv_info is None:
            raise ValueError("priv_info is required to compute the HORA teacher latent")
        return torch.tanh(self.priv_encoder(priv_info))

    def encode_proprio_history(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        if self.adapt_tconv is None:
            raise RuntimeError("HORA adaptation encoder is not enabled")
        return torch.tanh(self.adapt_tconv(proprio_hist))

    def build_core_output(self, obs: TensorDict, *, prefer_student: bool) -> HoraCoreOutput:
        actor_obs = obs["actor"]
        policy_obs = self._normalize_actor_obs(actor_obs)
        priv_info = obs.get("priv_info")
        proprio_hist = obs.get("proprio_hist")

        privileged_target = (
            self.encode_privileged_info(priv_info)
            if priv_info is not None
            else self._zero_privileged_latent(actor_obs.shape[0], actor_obs.device, actor_obs.dtype)
        )

        if prefer_student and self.adapt_tconv is not None and proprio_hist is not None:
            privileged_latent = self.encode_proprio_history(proprio_hist)
        elif priv_info is not None:
            privileged_latent = privileged_target
        else:
            privileged_latent = self._zero_privileged_latent(
                actor_obs.shape[0], actor_obs.device, actor_obs.dtype
            )

        trunk_input = torch.cat([policy_obs, privileged_latent], dim=-1)
        trunk_latent = self.trunk(trunk_input)
        return HoraCoreOutput(
            policy_obs=policy_obs,
            trunk_latent=trunk_latent,
            privileged_latent=privileged_latent,
            privileged_target=privileged_target,
        )

    def policy_mean(self, obs: TensorDict, *, prefer_student: bool) -> tuple[torch.Tensor, HoraCoreOutput]:
        core_output = self.build_core_output(obs, prefer_student=prefer_student)
        return self.mu_head(core_output.trunk_latent), core_output

    def value(self, obs: TensorDict, *, prefer_student: bool) -> tuple[torch.Tensor, HoraCoreOutput]:
        core_output = self.build_core_output(obs, prefer_student=prefer_student)
        return self.value_head(core_output.trunk_latent), core_output


class _HoraInferenceModule(nn.Module):
    input_names = ["actor", "priv_info", "proprio_hist"]
    output_names = ["actions"]

    def __init__(
        self,
        *,
        obs_normalizer: nn.Module,
        priv_encoder: nn.Module,
        trunk: nn.Module,
        mu_head: nn.Module,
        obs_dim: int,
        priv_info_dim: int,
        proprio_hist_len: int,
        proprio_frame_dim: int,
        verbose: bool = False,
        adapt_tconv: nn.Module | None = None,
        prefer_student: bool = False,
    ) -> None:
        super().__init__()
        self.obs_normalizer = obs_normalizer
        self.priv_encoder = priv_encoder
        self.trunk = trunk
        self.mu_head = mu_head
        self.adapt_tconv = adapt_tconv
        self.prefer_student = bool(prefer_student)
        self.obs_dim = int(obs_dim)
        self.priv_info_dim = int(priv_info_dim)
        self.proprio_hist_len = int(proprio_hist_len)
        self.proprio_frame_dim = int(proprio_frame_dim)
        self.verbose = bool(verbose)

    def forward(self, actor: torch.Tensor, priv_info: torch.Tensor, proprio_hist: torch.Tensor) -> torch.Tensor:
        policy_obs = self.obs_normalizer(actor)
        if self.prefer_student:
            if self.adapt_tconv is None:
                raise RuntimeError("HORA adaptation encoder export requires adapt_tconv")
            privileged_latent = torch.tanh(self.adapt_tconv(proprio_hist))
        else:
            privileged_latent = torch.tanh(self.priv_encoder(priv_info))
        trunk_input = torch.cat([policy_obs, privileged_latent], dim=-1)
        return self.mu_head(self.trunk(trunk_input))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.obs_dim),
            torch.zeros(1, self.priv_info_dim),
            torch.zeros(1, self.proprio_hist_len, self.proprio_frame_dim),
        )


class HoraActorModel(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        shared_model: HoraSharedActorCritic | None = None,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
        priv_info_dim: int | None = None,
        priv_info_embed_dim: int = 8,
        priv_mlp_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 8),
        use_student_encoder: bool = False,
        proprio_hist_len: int = 30,
        proprio_frame_dim: int | None = None,
    ) -> None:
        del obs_groups, obs_set
        super().__init__()
        if shared_model is None:
            shared_model = HoraSharedActorCritic(
                obs_dim=int(obs["actor"].shape[-1]),
                action_dim=output_dim,
                priv_info_dim=int(
                    priv_info_dim if priv_info_dim is not None else obs.get("priv_info").shape[-1]
                ),
                priv_info_embed_dim=priv_info_embed_dim,
                actor_hidden_dims=hidden_dims,
                priv_mlp_hidden_dims=priv_mlp_hidden_dims,
                activation=activation,
                obs_normalization=obs_normalization,
                distribution_cfg=distribution_cfg,
                use_student_encoder=use_student_encoder,
                proprio_hist_len=proprio_hist_len,
                proprio_frame_dim=proprio_frame_dim
                if proprio_frame_dim is not None
                else (int(obs["proprio_hist"].shape[-1]) if "proprio_hist" in obs else None),
            )
        self.shared = shared_model
        self.prefer_student = bool(use_student_encoder)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state
        mean, _ = self.shared.policy_mean(obs, prefer_student=self.prefer_student)
        self.shared.distribution.update(mean)
        if stochastic_output:
            return self.shared.distribution.sample()
        return self.shared.distribution.deterministic_output(mean)

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        return self.shared.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.shared.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.shared.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.shared.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.shared.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.shared.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        self.shared.update_normalization(obs)

    def as_jit(self) -> nn.Module:
        return _HoraInferenceModule(
            obs_normalizer=copy.deepcopy(self.shared.obs_normalizer),
            priv_encoder=copy.deepcopy(self.shared.priv_encoder),
            trunk=copy.deepcopy(self.shared.trunk),
            mu_head=copy.deepcopy(self.shared.mu_head),
            obs_dim=self.shared.obs_dim,
            priv_info_dim=self.shared.priv_info_dim,
            proprio_hist_len=self.shared.proprio_hist_len,
            proprio_frame_dim=self.shared.proprio_frame_dim,
            adapt_tconv=copy.deepcopy(self.shared.adapt_tconv),
            prefer_student=self.prefer_student,
        )

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _HoraInferenceModule(
            obs_normalizer=copy.deepcopy(self.shared.obs_normalizer),
            priv_encoder=copy.deepcopy(self.shared.priv_encoder),
            trunk=copy.deepcopy(self.shared.trunk),
            mu_head=copy.deepcopy(self.shared.mu_head),
            obs_dim=self.shared.obs_dim,
            priv_info_dim=self.shared.priv_info_dim,
            proprio_hist_len=self.shared.proprio_hist_len,
            proprio_frame_dim=self.shared.proprio_frame_dim,
            verbose=verbose,
            adapt_tconv=copy.deepcopy(self.shared.adapt_tconv),
            prefer_student=self.prefer_student,
        )


class HoraCriticModel(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        shared_model: HoraSharedActorCritic | None = None,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        priv_info_dim: int | None = None,
        priv_info_embed_dim: int = 8,
        priv_mlp_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 8),
    ) -> None:
        del obs_groups, obs_set, output_dim
        super().__init__()
        if shared_model is None:
            shared_model = HoraSharedActorCritic(
                obs_dim=int(obs["actor"].shape[-1]),
                action_dim=1,
                priv_info_dim=int(
                    priv_info_dim if priv_info_dim is not None else obs.get("priv_info").shape[-1]
                ),
                priv_info_embed_dim=priv_info_embed_dim,
                actor_hidden_dims=hidden_dims,
                priv_mlp_hidden_dims=priv_mlp_hidden_dims,
                activation=activation,
                obs_normalization=obs_normalization,
            )
        self.shared = shared_model

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state, stochastic_output
        value, _ = self.shared.value(obs, prefer_student=False)
        return value

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: TensorDict) -> None:
        del obs
