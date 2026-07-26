"""FlashSAC learner adapted to UniLab's off-policy contract."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from unilab.algos.torch.common.compile import get_torch_compile_for_cuda
from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.flash_sac.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from unilab.algos.torch.flash_sac.update import (
    build_lr_lambda,
    resolve_target_entropy,
    select_min_q_log_probs,
)
from unilab.algos.torch.offpolicy.distributed import normalize_distributed_sync_mode


@dataclass
class RunningMeanStd:
    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    @classmethod
    def create(cls, device: torch.device) -> "RunningMeanStd":
        return cls(
            mean=torch.zeros(1, device=device, dtype=torch.float32),
            var=torch.ones(1, device=device, dtype=torch.float32),
            count=torch.tensor(1e-4, device=device, dtype=torch.float32),
        )

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1).to(dtype=torch.float32)
        if x.numel() == 0:
            return
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = torch.tensor(float(x.numel()), device=x.device, dtype=torch.float32)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.pow(2) * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


class RewardNormalizer:
    """Adaptive reward scaling with running discounted-return statistics."""

    def __init__(
        self,
        gamma: float,
        g_max: float,
        device: torch.device,
        eps: float = 1e-8,
    ):
        self.gamma = gamma
        self.g_max = g_max
        self.eps = eps
        self.device = device
        self.rms = RunningMeanStd.create(device)
        self.g_r = torch.zeros(0, device=device, dtype=torch.float32)
        self.g_r_max = torch.tensor(0.0, device=device, dtype=torch.float32)

    def _ensure_g_r_shape(self, num_envs: int) -> None:
        if self.g_r.shape == (num_envs,):
            return
        self.g_r = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

    def update_from_transitions(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        rewards = rewards.to(device=self.device, dtype=torch.float32)
        dones = dones.to(device=self.device, dtype=torch.float32)

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)
        if rewards.numel() == 0:
            return

        num_envs = int(rewards.shape[-1])
        self._ensure_g_r_shape(num_envs)
        done = torch.clamp(dones, min=0.0, max=1.0)

        for step in range(rewards.shape[0]):
            self.g_r = self.gamma * (1.0 - done[step]) * self.g_r + rewards[step]
            self.g_r_max = torch.maximum(self.g_r_max, self.g_r.abs().max())
            self.rms.update(self.g_r)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        denominator = torch.maximum(
            torch.sqrt(self.rms.var + self.eps),
            self.g_r_max / max(self.g_max, self.eps),
        )
        return rewards / denominator

    def state_dict(self) -> dict[str, Any]:
        return {
            "rms": self.rms.state_dict(),
            "g_r": self.g_r,
            "g_r_max": self.g_r_max,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rms.load_state_dict(state_dict["rms"])
        self.g_r = state_dict["g_r"]
        self.g_r_max = state_dict["g_r_max"]

    def broadcast(self, src: int = 0) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        g_r_size = torch.tensor([self.g_r.numel()], device=self.device, dtype=torch.int64)
        dist.broadcast(g_r_size, src=src)
        num_envs = int(g_r_size.item())
        self._ensure_g_r_shape(num_envs)
        for tensor in (
            self.rms.mean,
            self.rms.var,
            self.rms.count,
            self.g_r_max,
        ):
            dist.broadcast(tensor, src=src)
        if num_envs > 0:
            dist.broadcast(self.g_r, src=src)


class FlashSACLearner:
    supports_multi_gpu = True
    supports_multi_gpu_symmetry = False
    supported_multi_gpu_sync_modes = frozenset({"sync_sgd", "local_sgd"})
    supports_cuda_graph_packed_staging = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        actor_num_blocks: int = 2,
        critic_num_blocks: int = 2,
        num_atoms: int = 101,
        critic_min_v: float = -5.0,
        critic_max_v: float = 5.0,
        temp_initial_value: float = 0.01,
        temp_target_sigma: float = 0.15,
        temp_target_entropy: float | None = None,
        actor_bc_alpha: float = 0.0,
        actor_noise_zeta_mu: float = 2.0,
        actor_noise_zeta_max: int = 16,
        learning_rate_init: float = 3e-4,
        learning_rate_peak: float = 3e-4,
        learning_rate_end: float = 1.5e-4,
        learning_rate_warmup_steps: int = 0,
        learning_rate_decay_steps: int = 500000,
        normalize_reward: bool = True,
        normalized_g_max: float = 5.0,
        n_step: int = 1,
        obs_normalization: bool = False,
        use_amp: bool = False,
        amp_dtype: str = "auto",
        use_compile: bool = False,
        use_cuda_graph_critic: bool = False,
        use_cuda_graph_actor: bool = False,
        use_cuda_graph_critic_packed_staging: bool = False,
        use_cuda_graph_actor_packed_staging: bool = False,
        world_size: int = 1,
        distributed_sync_mode: str = "sync_sgd",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.actor_bc_alpha = actor_bc_alpha
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.update_count = 0
        self.use_amp = bool(use_amp and self.device.type in ("cuda", "xpu"))
        self.amp_dtype = amp_dtype
        self._amp_dtype = self._resolve_amp_dtype(amp_dtype, self.device.type)
        self.use_compile = bool(
            use_compile and get_torch_compile_for_cuda(self.device, warn=True) is not None
        )
        self.use_cuda_graph_critic = bool(use_cuda_graph_critic)
        self.use_cuda_graph_actor = bool(use_cuda_graph_actor)
        self.use_cuda_graph_critic_packed_staging = bool(
            use_cuda_graph_critic_packed_staging and self.use_cuda_graph_critic
        )
        self.use_cuda_graph_actor_packed_staging = bool(
            use_cuda_graph_actor_packed_staging and self.use_cuda_graph_actor
        )
        self.world_size = int(world_size)
        self.distributed_sync_mode = normalize_distributed_sync_mode(distributed_sync_mode)

        self.actor = FlashSACActor(
            num_blocks=actor_num_blocks,
            input_dim=obs_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
            noise_zeta_mu=actor_noise_zeta_mu,
            noise_zeta_max=actor_noise_zeta_max,
            device=self.device,
        )
        self.critic = FlashSACDoubleCritic(
            num_blocks=critic_num_blocks,
            input_dim=self.critic_obs_dim + action_dim,
            hidden_dim=critic_hidden_dim,
            num_bins=num_atoms,
            min_v=critic_min_v,
            max_v=critic_max_v,
            device=self.device,
        )
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.eval()
        self.temperature = FlashSACTemperature(temp_initial_value).to(self.device)

        self.target_entropy = resolve_target_entropy(
            action_dim=action_dim,
            target_sigma=temp_target_sigma,
            target_entropy=temp_target_entropy,
        )

        self.obs_normalizer: EmpiricalNormalization | nn.Identity
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=self.device)
        else:
            self.obs_normalizer = nn.Identity()

        self.reward_normalizer = (
            RewardNormalizer(gamma=self.gamma, g_max=normalized_g_max, device=self.device)
            if normalize_reward
            else None
        )

        # GradScaler is only needed for fp16 (cuda); bf16 on xpu doesn't need it.
        self.scaler: Any | None = (
            getattr(torch.amp, "GradScaler")("cuda")
            if self._should_use_grad_scaler(self.use_amp, self.device.type, self._amp_dtype)
            else None
        )
        lr_peak = learning_rate_peak if learning_rate_peak > 0 else actor_lr
        optimizer_kwargs: dict[str, Any] = {"fused": self.device.type == "cuda"}
        if self.device.type == "cuda":
            optimizer_kwargs["capturable"] = True
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_peak, **optimizer_kwargs)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_peak, **optimizer_kwargs)
        self.temperature_optimizer = optim.Adam(
            self.temperature.parameters(), lr=lr_peak, **optimizer_kwargs
        )
        self._cuda_graph_critic: torch.cuda.CUDAGraph | None = None
        self._cuda_graph_critic_static_inputs: dict[str, torch.Tensor] | None = None
        self._cuda_graph_sac_static_packed_input: torch.Tensor | None = None
        self._cuda_graph_sac_static_source_ptr: int | None = None
        self._cuda_graph_critic_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cuda_graph_critic_shapes: dict[str, torch.Size] | None = None
        self._cuda_graph_actor: torch.cuda.CUDAGraph | None = None
        self._cuda_graph_actor_static_inputs: dict[str, torch.Tensor] | None = None
        self._cuda_graph_actor_static_packed_input: torch.Tensor | None = None
        self._cuda_graph_actor_outputs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._cuda_graph_actor_shapes: dict[str, torch.Size] | None = None

        scheduler_fn = build_lr_lambda(
            init_lr=learning_rate_init,
            peak_lr=lr_peak,
            end_lr=learning_rate_end,
            warmup_steps=learning_rate_warmup_steps,
            decay_steps=learning_rate_decay_steps,
        )
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, scheduler_fn)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, scheduler_fn)
        self.temperature_scheduler = optim.lr_scheduler.LambdaLR(
            self.temperature_optimizer, scheduler_fn
        )

        if self.use_compile:
            self._compile_training_methods()

    def _compile_training_methods(self) -> None:
        compile_fn = get_torch_compile_for_cuda(self.device, warn=True)
        if compile_fn is None:
            return

        compile_kwargs = {"options": {"triton.cudagraphs": False}}
        self.actor.get_mean_and_std = compile_fn(  # type: ignore[method-assign]
            self.actor.get_mean_and_std, **compile_kwargs
        )
        if not self.use_cuda_graph_critic:
            self._critic_loss_tensors = compile_fn(  # type: ignore[method-assign]
                self._critic_loss_tensors, **compile_kwargs
            )
        if not self.use_cuda_graph_actor:
            self._actor_loss_tensors = compile_fn(  # type: ignore[method-assign]
                self._actor_loss_tensors, **compile_kwargs
            )

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str, device_type: str) -> torch.dtype:
        normalized = amp_dtype.lower()
        if normalized == "auto":
            return torch.bfloat16
        if normalized == "fp16":
            return torch.float16
        if normalized == "bf16":
            return torch.bfloat16
        raise ValueError("FlashSAC amp_dtype must be one of: auto, fp16, bf16")

    @staticmethod
    def _should_use_grad_scaler(
        use_amp: bool,
        device_type: str,
        amp_dtype: torch.dtype,
    ) -> bool:
        return bool(use_amp) and device_type == "cuda" and amp_dtype == torch.float16

    def _maybe_normalize_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if isinstance(self.obs_normalizer, nn.Identity):
            return obs
        normalizer = cast(EmpiricalNormalization, self.obs_normalizer)
        if update:
            self._update_obs_normalizer(obs)
            return cast(torch.Tensor, normalizer(obs, update=False))
        return cast(torch.Tensor, normalizer(obs, update=False))

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type, dtype=self._amp_dtype, enabled=self.use_amp
        )

    def _distributed_normalization_ready(self) -> bool:
        return self.world_size > 1 and dist.is_available() and dist.is_initialized()

    @torch.no_grad()
    def _update_obs_normalizer(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_normalizer, nn.Identity):
            return
        normalizer = cast(EmpiricalNormalization, self.obs_normalizer)
        if not self._distributed_normalization_ready():
            normalizer.update(obs)
            return

        obs_for_stats = obs.detach().to(dtype=normalizer._mean.dtype)
        obs_dim = int(obs_for_stats.shape[-1])
        moment_payload = torch.cat(
            [
                obs_for_stats.sum(dim=0),
                obs_for_stats.square().sum(dim=0),
                torch.tensor(
                    [obs_for_stats.shape[0]],
                    device=obs_for_stats.device,
                    dtype=obs_for_stats.dtype,
                ),
            ]
        )
        dist.all_reduce(moment_payload, op=dist.ReduceOp.SUM)
        batch_count = moment_payload[-1].clamp_min(1.0)
        batch_mean = (moment_payload[:obs_dim] / batch_count).view_as(normalizer._mean)
        batch_var = (
            moment_payload[obs_dim : 2 * obs_dim] / batch_count - batch_mean.view(-1).square()
        ).clamp_min(0.0)
        normalizer.update_from_moments(
            batch_mean,
            batch_var.view_as(normalizer._var),
            batch_count.round().to(dtype=normalizer.count.dtype),
        )

    def _reduce_gradients(self, module: nn.Module) -> bool:
        if self.world_size <= 1 or self.distributed_sync_mode != "sync_sgd":
            return True
        grads = [param.grad.reshape(-1) for param in module.parameters() if param.grad is not None]
        if not grads:
            return True
        flat = torch.cat(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= self.world_size
        if not bool(torch.isfinite(flat).all().item()):
            return False
        offset = 0
        for param in module.parameters():
            if param.grad is None:
                continue
            n = param.grad.numel()
            param.grad.copy_(flat[offset : offset + n].view_as(param.grad))
            offset += n
        return True

    def _backoff_grad_scaler(self) -> None:
        if self.scaler is None:
            return
        self.scaler.update(self.scaler.get_scale() * self.scaler.get_backoff_factor())

    def _parameter_sync_tensors(self) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        for module in (self.actor, self.critic, self.target_critic, self.temperature):
            tensors.extend(
                tensor
                for tensor in module.state_dict().values()
                if torch.is_tensor(tensor) and tensor.is_floating_point()
            )
        return tensors

    @torch.no_grad()
    def average_distributed_parameters(self) -> None:
        if self.world_size <= 1:
            return
        tensors = self._parameter_sync_tensors()
        if not tensors:
            return
        flat = torch.cat([tensor.reshape(-1) for tensor in tensors])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= self.world_size
        offset = 0
        for tensor in tensors:
            n = tensor.numel()
            tensor.copy_(flat[offset : offset + n].view_as(tensor))
            offset += n
        self.actor.normalize_parameters()
        self.critic.normalize_parameters()
        self.target_critic.normalize_parameters()

    def sync_initial_parameters(self, src: int = 0) -> None:
        if self.world_size <= 1:
            return
        for module in (self.actor, self.critic, self.target_critic, self.temperature):
            for tensor in module.state_dict().values():
                if torch.is_tensor(tensor):
                    dist.broadcast(tensor, src=src)
        if not isinstance(self.obs_normalizer, nn.Identity):
            for tensor in self.obs_normalizer.state_dict().values():
                if torch.is_tensor(tensor):
                    dist.broadcast(tensor, src=src)
        self.sync_reward_normalizer(src=src)

    def sync_reward_normalizer(self, src: int = 0) -> None:
        if self.world_size <= 1 or self.reward_normalizer is None:
            return
        self.reward_normalizer.broadcast(src=src)

    def update_reward_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if self.reward_normalizer is None:
            return
        self.reward_normalizer.update_from_transitions(rewards, dones)

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _critic_loss_tensors(
        self,
        next_q_values: torch.Tensor,
        next_q_log_probs_full: torch.Tensor,
        support: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
        actor_entropy: torch.Tensor,
        pred_log_probs: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        next_q_log_probs = select_min_q_log_probs(next_q_values, next_q_log_probs_full)
        batch_size, num_bins = next_q_log_probs.shape
        support_view = support.view(1, -1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        truncated = truncated.view(-1, 1)
        actor_entropy = actor_entropy.view(-1, 1)

        bootstrap = torch.clamp(1.0 - dones + truncated, 0.0, 1.0)
        support_min = support_view.min()
        support_max = support_view.max()
        target_bin_values = rewards + bootstrap * gamma * (support_view - actor_entropy)
        target_bin_values = torch.clamp(target_bin_values, support_min, support_max)

        bin_width = torch.clamp(support_view[0, 1] - support_view[0, 0], min=1e-8)
        offsets = (target_bin_values - support_min) / bin_width
        lower = torch.floor(offsets).long().clamp(0, num_bins - 1)
        upper = torch.ceil(offsets).long().clamp(0, num_bins - 1)
        frac = offsets - lower.float()

        probs = next_q_log_probs.exp()
        target_probs = torch.zeros(batch_size, num_bins, dtype=probs.dtype, device=probs.device)
        target_probs.scatter_add_(1, lower, probs * (1.0 - frac))
        target_probs.scatter_add_(1, upper, probs * frac)
        return cast(torch.Tensor, -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1).mean())

    def _actor_loss_tensors(
        self,
        log_probs: torch.Tensor,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        expert_actions: torch.Tensor,
        temp_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        min_q = torch.min(q_values[0], q_values[1])
        actor_loss = (temp_value.detach() * log_probs - min_q).mean()
        if self.actor_bc_alpha > 0:
            bc_loss = torch.mean((actions - expert_actions) ** 2)
            actor_loss = actor_loss + self.actor_bc_alpha * min_q.abs().mean().detach() * bc_loss
        entropy = -log_probs.detach().mean()
        return actor_loss, entropy

    @staticmethod
    def _critic_graph_input_keys() -> tuple[str, ...]:
        return (
            "obs",
            "actions",
            "rewards",
            "next_obs",
            "dones",
            "truncated",
            "critic",
            "next_critic",
        )

    def _critic_graph_input_shapes(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Size]:
        return {key: inputs[key].shape for key in self._critic_graph_input_keys()}

    @staticmethod
    def _graph_width(shape: torch.Size) -> int:
        value = 1
        for dim in shape[1:]:
            value *= int(dim)
        return value

    @classmethod
    def _sac_graph_offsets(
        cls,
        actor_shapes: dict[str, torch.Size],
        critic_shapes: dict[str, torch.Size],
    ) -> dict[str, tuple[int, int]]:
        widths = {
            "obs": cls._graph_width(actor_shapes["obs"]),
            "critic": cls._graph_width(critic_shapes["critic"]),
            "actions": cls._graph_width(critic_shapes["actions"]),
            "rewards": cls._graph_width(critic_shapes["rewards"]),
            "next_obs": cls._graph_width(critic_shapes["next_obs"]),
            "next_critic": cls._graph_width(critic_shapes["next_critic"]),
            "dones": cls._graph_width(critic_shapes["dones"]),
            "truncated": cls._graph_width(critic_shapes["truncated"]),
        }
        offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for key in (
            "obs",
            "critic",
            "actions",
            "rewards",
            "next_obs",
            "next_critic",
            "dones",
            "truncated",
        ):
            key_width = widths[key]
            offsets[key] = (offset, key_width)
            offset += key_width
        return offsets

    @classmethod
    def _critic_graph_static_views_from_sac_packed(
        cls,
        packed: torch.Tensor,
        critic_shapes: dict[str, torch.Size],
        actor_shapes: dict[str, torch.Size],
    ) -> dict[str, torch.Tensor]:
        offsets = cls._sac_graph_offsets(actor_shapes, critic_shapes)
        views: dict[str, torch.Tensor] = {}
        for key in cls._critic_graph_input_keys():
            offset, width = offsets[key]
            views[key] = packed.narrow(1, offset, width).view(critic_shapes[key])
        return views

    @classmethod
    def _actor_graph_static_views_from_sac_packed(
        cls,
        packed: torch.Tensor,
        actor_shapes: dict[str, torch.Size],
    ) -> dict[str, torch.Tensor]:
        batch_size = int(actor_shapes["obs"][0])
        critic_shapes = {
            "critic": actor_shapes["critic"],
            "actions": actor_shapes["actions"],
            "rewards": torch.Size((batch_size,)),
            "next_obs": actor_shapes["next_obs"],
            "next_critic": actor_shapes["critic"],
            "dones": torch.Size((batch_size,)),
            "truncated": torch.Size((batch_size,)),
        }
        offsets = cls._sac_graph_offsets(actor_shapes, critic_shapes)
        views: dict[str, torch.Tensor] = {}
        for key in cls._actor_graph_input_keys():
            source_key = "actions" if key == "actions" else key
            offset, width = offsets[source_key]
            views[key] = packed.narrow(1, offset, width).view(actor_shapes[key])
        return views

    def _prepare_critic_graph_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)
        truncated = batch["truncated"].to(self.device)
        critic_obs = batch["critic"].to(self.device)
        critic_next_obs = batch["next_critic"].to(self.device)

        obs = self._maybe_normalize_obs(obs, update=True)
        next_obs = self._maybe_normalize_obs(next_obs, update=False)
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer.normalize(rewards)

        prepared = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "truncated": truncated,
            "critic": critic_obs,
            "next_critic": critic_next_obs,
        }
        if "sac_graph_packed_source" in batch:
            prepared["sac_graph_packed_source"] = batch["sac_graph_packed_source"].to(self.device)
        return prepared

    def _copy_critic_graph_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        assert self._cuda_graph_critic_static_inputs is not None
        packed_source = inputs.get("sac_graph_packed_source")
        if packed_source is not None and self._cuda_graph_sac_static_packed_input is not None:
            self._cuda_graph_sac_static_packed_input.copy_(packed_source)
            self._cuda_graph_sac_static_source_ptr = int(packed_source.data_ptr())
            return
        for key, tensor in self._cuda_graph_critic_static_inputs.items():
            tensor.copy_(inputs[key])

    def _update_critic_capture_candidate(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = inputs["actions"]
        rewards = inputs["rewards"]
        next_obs = inputs["next_obs"]
        dones = inputs["dones"]
        truncated = inputs["truncated"]
        critic_obs = inputs["critic"]
        critic_next_obs = inputs["next_critic"]

        gamma = self.gamma**self.n_step
        obs_all = torch.cat([critic_obs, critic_next_obs], dim=0)

        with torch.no_grad():
            with self._autocast():
                next_actions, actor_info = self.actor(next_obs, training=False)
                actor_entropy = self.temperature().detach() * actor_info["log_prob"]
                act_all = torch.cat([actions, next_actions], dim=0)
                qs_all, q_info_all = self.target_critic(obs_all, act_all, training=True)
                next_q_values = qs_all.chunk(2, dim=1)[1]
                next_q_log_probs_full = q_info_all["log_prob"].chunk(2, dim=1)[1]
                support = cast(torch.Tensor, self.target_critic.predictor.support)

        with self._autocast():
            _, pred_info_all = self.critic(obs_all, act_all, training=True)
            pred_log_probs = pred_info_all["log_prob"].chunk(2, dim=1)[0]
            critic_loss = self._critic_loss_tensors(
                next_q_values,
                next_q_log_probs_full,
                support,
                rewards,
                dones,
                truncated,
                actor_entropy,
                pred_log_probs,
                gamma,
            )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        reward_scale_std = (
            torch.sqrt(self.reward_normalizer.rms.var)
            if self.reward_normalizer is not None
            else torch.ones((), device=self.device)
        )
        return critic_loss, reward_scale_std

    def _reset_critic_cuda_graph(self) -> None:
        self._cuda_graph_critic = None
        self._cuda_graph_critic_static_inputs = None
        self._cuda_graph_sac_static_packed_input = None
        self._cuda_graph_sac_static_source_ptr = None
        self._cuda_graph_critic_outputs = None
        self._cuda_graph_critic_shapes = None

    def _materialize_capturable_critic_optimizer_state(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> None:
        optimizer_lrs = [group["lr"] for group in self.critic_optimizer.param_groups]
        optimizer_weight_decays = [
            group["weight_decay"] for group in self.critic_optimizer.param_groups
        ]
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if self.device.type == "cuda" else None
        try:
            for group in self.critic_optimizer.param_groups:
                group["lr"] = 0.0
                group["weight_decay"] = 0.0
            self._update_critic_capture_candidate(inputs)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            for group, lr, weight_decay in zip(
                self.critic_optimizer.param_groups,
                optimizer_lrs,
                optimizer_weight_decays,
                strict=True,
            ):
                group["lr"] = lr
                group["weight_decay"] = weight_decay

        self.critic_optimizer.zero_grad(set_to_none=True)
        for state in self.critic_optimizer.state.values():
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                step.zero_()
            elif step is not None:
                state["step"] = 0
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                tensor = state.get(name)
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()

    def _capture_critic_cuda_graph(self, inputs: dict[str, torch.Tensor]) -> None:
        self._cuda_graph_critic_shapes = self._critic_graph_input_shapes(inputs)
        packed_source = inputs.get("sac_graph_packed_source")
        if self.use_cuda_graph_critic_packed_staging and packed_source is not None:
            self._cuda_graph_sac_static_packed_input = packed_source.detach().clone()
            actor_shapes = self._actor_graph_input_shapes(inputs)
            self._cuda_graph_critic_static_inputs = self._critic_graph_static_views_from_sac_packed(
                self._cuda_graph_sac_static_packed_input,
                self._cuda_graph_critic_shapes,
                actor_shapes,
            )
        else:
            self._cuda_graph_sac_static_packed_input = None
            self._cuda_graph_critic_static_inputs = {
                key: inputs[key].detach().clone() for key in self._critic_graph_input_keys()
            }
        self._copy_critic_graph_inputs(inputs)

        graph = torch.cuda.CUDAGraph()
        capture_stream = cast(torch.cuda.Stream, torch.cuda.Stream())
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
            self._cuda_graph_critic_outputs = self._update_critic_capture_candidate(
                self._cuda_graph_critic_static_inputs
            )
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._cuda_graph_critic = graph

    def _critic_graph_output_metrics(self, *, read_items: bool = True) -> dict[str, float]:
        if not read_items:
            return {}
        assert self._cuda_graph_critic_outputs is not None
        critic_loss, reward_scale_std = self._cuda_graph_critic_outputs
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "reward_scale_std": float(reward_scale_std.detach().cpu()),
        }

    def update_critic_cuda_graph(
        self,
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> dict[str, float]:
        if not self.use_cuda_graph_critic:
            return self.update_critic(batch)
        if self.device.type != "cuda":
            return self.update_critic(batch)
        if self.scaler is not None or self.world_size > 1:
            return self.update_critic(batch)
        if not isinstance(self.obs_normalizer, nn.Identity):
            return self.update_critic(batch)

        inputs = self._prepare_critic_graph_inputs(batch)
        if self._cuda_graph_critic_shapes != self._critic_graph_input_shapes(inputs):
            self._reset_critic_cuda_graph()
            self._materialize_capturable_critic_optimizer_state(inputs)
            self._capture_critic_cuda_graph(inputs)
            self.critic_scheduler.step()
            self.critic.normalize_parameters()
            return self._critic_graph_output_metrics(read_items=read_metrics)

        assert self._cuda_graph_critic is not None
        self._copy_critic_graph_inputs(inputs)
        self._cuda_graph_critic.replay()
        self.critic_scheduler.step()
        self.critic.normalize_parameters()
        return self._critic_graph_output_metrics(read_items=read_metrics)

    @staticmethod
    def _actor_graph_input_keys() -> tuple[str, ...]:
        return ("obs", "next_obs", "actions", "critic")

    def _actor_graph_input_shapes(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Size]:
        return {key: inputs[key].shape for key in self._actor_graph_input_keys()}

    def _prepare_actor_graph_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        expert_actions = batch["actions"].to(self.device)
        critic_obs = batch["critic"].to(self.device)
        prepared = {
            "obs": self._maybe_normalize_obs(obs, update=False),
            "next_obs": self._maybe_normalize_obs(next_obs, update=False),
            "actions": expert_actions,
            "critic": critic_obs,
        }
        if "sac_graph_packed_source" in batch:
            prepared["sac_graph_packed_source"] = batch["sac_graph_packed_source"].to(self.device)
        return prepared

    def _copy_actor_graph_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        assert self._cuda_graph_actor_static_inputs is not None
        packed_source = inputs.get("sac_graph_packed_source")
        if packed_source is not None:
            static_packed = self._cuda_graph_actor_static_packed_input
            if static_packed is None:
                static_packed = self._cuda_graph_sac_static_packed_input
            if static_packed is not None:
                source_ptr = int(packed_source.data_ptr())
                if (
                    static_packed is not self._cuda_graph_sac_static_packed_input
                    or self._cuda_graph_sac_static_source_ptr != source_ptr
                ):
                    static_packed.copy_(packed_source)
                    if static_packed is self._cuda_graph_sac_static_packed_input:
                        self._cuda_graph_sac_static_source_ptr = source_ptr
                return
        for key, tensor in self._cuda_graph_actor_static_inputs.items():
            tensor.copy_(inputs[key])

    def _update_actor_capture_candidate(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = inputs["obs"]
        next_obs = inputs["next_obs"]
        expert_actions = inputs["actions"]
        critic_obs = inputs["critic"]
        obs_all = torch.cat([obs, next_obs], dim=0)

        with self._autocast():
            actions_all, actor_info_all = self.actor(obs_all, training=True)
            actions = actions_all.chunk(2, dim=0)[0]
            log_probs = actor_info_all["log_prob"].chunk(2, dim=0)[0]

            self._set_requires_grad(self.critic, False)
            q_values, _ = self.critic(critic_obs, actions, training=False)
            self._set_requires_grad(self.critic, True)
            temp_value = self.temperature()
            actor_loss, entropy = self._actor_loss_tensors(
                log_probs, q_values, actions, expert_actions, temp_value
            )

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        temp_loss = temp_value * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temp_loss.backward()
        self.temperature_optimizer.step()
        return actor_loss, entropy, temp_value, temp_loss

    def _reset_actor_cuda_graph(self) -> None:
        self._cuda_graph_actor = None
        self._cuda_graph_actor_static_inputs = None
        self._cuda_graph_actor_static_packed_input = None
        self._cuda_graph_actor_outputs = None
        self._cuda_graph_actor_shapes = None

    def _materialize_capturable_actor_optimizer_state(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> None:
        optimizers = (self.actor_optimizer, self.temperature_optimizer)
        optimizer_lrs = [
            [group["lr"] for group in optimizer.param_groups] for optimizer in optimizers
        ]
        optimizer_weight_decays = [
            [group["weight_decay"] for group in optimizer.param_groups] for optimizer in optimizers
        ]
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if self.device.type == "cuda" else None
        try:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = 0.0
                    group["weight_decay"] = 0.0
            self._update_actor_capture_candidate(inputs)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            for optimizer, lrs, weight_decays in zip(
                optimizers,
                optimizer_lrs,
                optimizer_weight_decays,
                strict=True,
            ):
                for group, lr, weight_decay in zip(
                    optimizer.param_groups,
                    lrs,
                    weight_decays,
                    strict=True,
                ):
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
            for state in optimizer.state.values():
                step = state.get("step")
                if isinstance(step, torch.Tensor):
                    step.zero_()
                elif step is not None:
                    state["step"] = 0
                for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    tensor = state.get(name)
                    if isinstance(tensor, torch.Tensor):
                        tensor.zero_()

    def _capture_actor_cuda_graph(self, inputs: dict[str, torch.Tensor]) -> None:
        self._cuda_graph_actor_shapes = self._actor_graph_input_shapes(inputs)
        packed_source = inputs.get("sac_graph_packed_source")
        if self.use_cuda_graph_actor_packed_staging and packed_source is not None:
            if (
                self._cuda_graph_sac_static_packed_input is not None
                and self._cuda_graph_sac_static_packed_input.shape == packed_source.shape
            ):
                self._cuda_graph_actor_static_packed_input = (
                    self._cuda_graph_sac_static_packed_input
                )
            else:
                self._cuda_graph_actor_static_packed_input = packed_source.detach().clone()
            self._cuda_graph_actor_static_inputs = self._actor_graph_static_views_from_sac_packed(
                self._cuda_graph_actor_static_packed_input,
                self._cuda_graph_actor_shapes,
            )
        else:
            self._cuda_graph_actor_static_packed_input = None
            self._cuda_graph_actor_static_inputs = {
                key: inputs[key].detach().clone() for key in self._actor_graph_input_keys()
            }
        self._copy_actor_graph_inputs(inputs)

        graph = torch.cuda.CUDAGraph()
        capture_stream = cast(torch.cuda.Stream, torch.cuda.Stream())
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
            self._cuda_graph_actor_outputs = self._update_actor_capture_candidate(
                self._cuda_graph_actor_static_inputs
            )
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._cuda_graph_actor = graph

    def _actor_graph_output_metrics(self, *, read_items: bool = True) -> dict[str, float]:
        if not read_items:
            return {}
        assert self._cuda_graph_actor_outputs is not None
        actor_loss, entropy, temp_value, temp_loss = self._cuda_graph_actor_outputs
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_entropy": float(entropy.detach().cpu()),
            "temperature": float(temp_value.detach().cpu()),
            "temperature_loss": float(temp_loss.detach().cpu()),
        }

    def update_actor_cuda_graph(
        self,
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> dict[str, float]:
        if not self.use_cuda_graph_actor:
            return self.update_actor(batch)
        if self.device.type != "cuda":
            return self.update_actor(batch)
        if self.scaler is not None or self.world_size > 1:
            return self.update_actor(batch)
        if not isinstance(self.obs_normalizer, nn.Identity):
            return self.update_actor(batch)

        inputs = self._prepare_actor_graph_inputs(batch)
        if self._cuda_graph_actor_shapes != self._actor_graph_input_shapes(inputs):
            self._reset_actor_cuda_graph()
            self._materialize_capturable_actor_optimizer_state(inputs)
            self._capture_actor_cuda_graph(inputs)
            self.actor_scheduler.step()
            self.temperature_scheduler.step()
            self.actor.normalize_parameters()
            return self._actor_graph_output_metrics(read_items=read_metrics)

        assert self._cuda_graph_actor is not None
        self._copy_actor_graph_inputs(inputs)
        self._cuda_graph_actor.replay()
        self.actor_scheduler.step()
        self.temperature_scheduler.step()
        self.actor.normalize_parameters()
        return self._actor_graph_output_metrics(read_items=read_metrics)

    def update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)
        truncated = batch["truncated"].to(self.device)
        critic_obs = batch["critic"].to(self.device)
        critic_next_obs = batch["next_critic"].to(self.device)

        obs = self._maybe_normalize_obs(obs, update=True)
        next_obs = self._maybe_normalize_obs(next_obs, update=False)

        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer.normalize(rewards)

        gamma = self.gamma**self.n_step

        obs_all = torch.cat([critic_obs, critic_next_obs], dim=0)

        with torch.no_grad():
            with self._autocast():
                next_actions, actor_info = self.actor(next_obs, training=False)
                actor_entropy = self.temperature().detach() * actor_info["log_prob"]
                act_all = torch.cat([actions, next_actions], dim=0)
                qs_all, q_info_all = self.target_critic(obs_all, act_all, training=True)
                next_q_values = qs_all.chunk(2, dim=1)[1]
                next_q_log_probs_full = q_info_all["log_prob"].chunk(2, dim=1)[1]
                support = cast(torch.Tensor, self.target_critic.predictor.support)

        with self._autocast():
            _, pred_info_all = self.critic(obs_all, act_all, training=True)
            pred_log_probs = pred_info_all["log_prob"].chunk(2, dim=1)[0]
            critic_loss = self._critic_loss_tensors(
                next_q_values,
                next_q_log_probs_full,
                support,
                rewards,
                dones,
                truncated,
                actor_entropy,
                pred_log_probs,
                gamma,
            )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_step_ok = True
        if self.scaler is not None:
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            critic_step_ok = self._reduce_gradients(self.critic)
            if critic_step_ok:
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
            else:
                self._backoff_grad_scaler()
        else:
            critic_loss.backward()
            critic_step_ok = self._reduce_gradients(self.critic)
            if critic_step_ok:
                self.critic_optimizer.step()
        if critic_step_ok:
            self.critic_scheduler.step()
            self.critic.normalize_parameters()

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "reward_scale_std": float(
                torch.sqrt(self.reward_normalizer.rms.var).detach().cpu()
                if self.reward_normalizer is not None
                else torch.tensor(1.0)
            ),
        }

    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        expert_actions = batch["actions"].to(self.device)
        critic_obs = batch["critic"].to(self.device)

        obs = self._maybe_normalize_obs(obs, update=False)
        next_obs = self._maybe_normalize_obs(next_obs, update=False)

        obs_all = torch.cat([obs, next_obs], dim=0)

        with self._autocast():
            actions_all, actor_info_all = self.actor(obs_all, training=True)
            actions = actions_all.chunk(2, dim=0)[0]
            log_probs = actor_info_all["log_prob"].chunk(2, dim=0)[0]

            self._set_requires_grad(self.critic, False)
            q_values, _ = self.critic(critic_obs, actions, training=False)
            self._set_requires_grad(self.critic, True)
            actor_loss, entropy = self._actor_loss_tensors(
                log_probs, q_values, actions, expert_actions, self.temperature()
            )

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_step_ok = True
        if self.scaler is not None:
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            actor_step_ok = self._reduce_gradients(self.actor)
            if actor_step_ok:
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                self._backoff_grad_scaler()
        else:
            actor_loss.backward()
            actor_step_ok = self._reduce_gradients(self.actor)
            if actor_step_ok:
                self.actor_optimizer.step()
        if actor_step_ok:
            self.actor_scheduler.step()
            self.actor.normalize_parameters()

        temp_value = self.temperature()
        temp_loss = temp_value * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temp_loss.backward()
        if self._reduce_gradients(self.temperature):
            self.temperature_optimizer.step()
            self.temperature_scheduler.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_entropy": float(entropy.detach().cpu()),
            "temperature": float(temp_value.detach().cpu()),
            "temperature_loss": float(temp_loss.detach().cpu()),
        }

    def soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "temperature": self.temperature.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "temperature_scheduler": self.temperature_scheduler.state_dict(),
            "obs_normalizer": (
                self.obs_normalizer.state_dict()
                if hasattr(self.obs_normalizer, "state_dict")
                else None
            ),
            "reward_normalizer": (
                self.reward_normalizer.state_dict() if self.reward_normalizer is not None else None
            ),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.temperature.load_state_dict(state_dict["temperature"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.temperature_optimizer.load_state_dict(state_dict["temperature_optimizer"])
        self.actor_scheduler.load_state_dict(state_dict["actor_scheduler"])
        self.critic_scheduler.load_state_dict(state_dict["critic_scheduler"])
        self.temperature_scheduler.load_state_dict(state_dict["temperature_scheduler"])
        if state_dict.get("obs_normalizer") and hasattr(self.obs_normalizer, "load_state_dict"):
            self.obs_normalizer.load_state_dict(state_dict["obs_normalizer"])
        if self.reward_normalizer is not None and state_dict.get("reward_normalizer"):
            self.reward_normalizer.load_state_dict(state_dict["reward_normalizer"])
        self.update_count = int(state_dict.get("update_count", 0))
