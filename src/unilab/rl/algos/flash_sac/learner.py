"""FlashSAC learner adapted to UniLab's off-policy contract."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Sequence, cast

import torch
import torch.nn as nn
import torch.optim as optim

from unilab.rl.algos.common.compile import get_torch_compile_for_device
from unilab.rl.algos.common.learner_base import (
    WholeCycleLearnerMixin,
    fused_adam_supported,
    mps_bf16_autocast_supported,
    resolve_amp_selection,
    resolve_finite_check_flags,
    validate_amp_compile_paths,
)
from unilab.rl.algos.common.normalization import EmpiricalNormalization
from unilab.rl.algos.common.performance import resolve_learner_performance_tier
from unilab.rl.algos.flash_sac.actor_spec import (
    FlashSACActorSpec,
    build_builtin_actor_semantics,
    build_custom_actor,
    build_custom_actor_semantics,
    validate_actor_semantics,
    validate_aux_losses,
)
from unilab.rl.algos.flash_sac.layers import UnitBatchNorm, UnitLinear, UnitRMSNorm
from unilab.rl.algos.flash_sac.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from unilab.rl.algos.flash_sac.objectives import (
    build_lr_lambda,
    compute_categorical_td_target,
    resolve_target_entropy,
    select_min_q_log_probs,
)


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

        # Keep the original storage alive.  A whole-cycle CUDA Graph captures
        # these addresses, while the runner continues to update reward
        # statistics between learner cycles.
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state_dict["mean"])
        self.var.copy_(state_dict["var"])
        self.count.copy_(state_dict["count"])


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
            self.g_r.mul_(self.gamma * (1.0 - done[step])).add_(rewards[step])
            self.g_r_max.copy_(torch.maximum(self.g_r_max, self.g_r.abs().max()))
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
        if self.g_r.shape != state_dict["g_r"].shape:
            self.g_r = torch.empty_like(state_dict["g_r"], device=self.device)
        self.g_r.copy_(state_dict["g_r"])
        self.g_r_max.copy_(state_dict["g_r_max"])


def _disable_parameter_normalization(module: nn.Module) -> None:
    """Restore FlashSAC's pre-normalization parameter layout."""
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, UnitLinear):
                nn.init.orthogonal_(child.w.weight)
            elif isinstance(child, UnitBatchNorm):
                child.weight.fill_(1.0)
                child.bias.zero_()
            elif isinstance(child, UnitRMSNorm):
                child.weight.fill_(1.0)


def _noop_normalize_parameters() -> None:
    """Instance-level no-op used when parameter normalization is disabled."""


class FlashSACLearner(WholeCycleLearnerMixin):
    supports_deferred_update_metrics = True

    def inference_startup_memory_categories(self, batch_size: int) -> dict[str, int]:
        """Expose exact inference-owned persistent scratch before collection."""
        return cast(
            dict[str, int],
            cast(Any, self.actor).inference_startup_memory_categories(batch_size),
        )

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
        use_compile: bool | None = None,
        compile_full_objectives: bool | None = None,
        use_whole_cycle_cuda_graph: bool | None = None,
        dp_whole_cycle_cuda_graph: bool = False,
        actor_normalize_parameters: bool = True,
        critic_normalize_parameters: bool = True,
        actor_config: Mapping[str, Any] | None = None,
    ):
        self.actor_normalize_parameters = bool(actor_normalize_parameters)
        self.critic_normalize_parameters = bool(critic_normalize_parameters)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.actor_bc_alpha = actor_bc_alpha
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.update_count = 0
        self._device_type = self.device.type
        self._performance_tier = resolve_learner_performance_tier(self.device)
        self._nvidia_cuda = self._performance_tier.supports_whole_cycle_graph
        amp_selection = resolve_amp_selection(
            device_type=self._device_type,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            learner_name="FlashSAC",
        )
        self.use_amp = amp_selection.use_amp
        self.amp_policy = amp_selection.manifest()
        if self._device_type == "mps" and self.use_amp and not mps_bf16_autocast_supported():
            raise RuntimeError(
                "FlashSAC MPS BF16 AMP was requested, but this Torch/MPS runtime "
                "cannot execute BF16 autocast"
            )
        self.amp_dtype = amp_dtype
        self._amp_dtype = self._resolve_amp_dtype(amp_dtype, self.device.type)
        compile_requested = self._nvidia_cuda if use_compile is None else bool(use_compile)
        compile_fn = (
            get_torch_compile_for_device(self.device, warn=not self._nvidia_cuda)
            if compile_requested
            else None
        )
        if compile_requested and self._nvidia_cuda and compile_fn is None:
            raise RuntimeError("FlashSAC requires CUDA Inductor/Triton on NVIDIA CUDA")
        # ``None`` is the auto policy (compile requested on the validated
        # NVIDIA CUDA tier, off elsewhere); an explicit ``False`` opts out of
        # compile for debugging on any tier.
        self.use_compile = bool(compile_requested and compile_fn is not None)
        validate_amp_compile_paths(
            device_type=self._device_type,
            use_compile=self.use_compile,
            use_amp=self.use_amp,
            learner_name="FlashSAC",
        )
        full_objectives_requested = (
            self._nvidia_cuda if compile_full_objectives is None else bool(compile_full_objectives)
        )
        self.compile_full_objectives = bool(full_objectives_requested and self.use_compile)
        # Host-side ``Tensor.item``/truth checks synchronize the device.  The
        # compiled CUDA path uses the fused optimizer's device gate; MPS has no
        # device-side skip (its fused Adam kernel ignores ``found_inf``), so
        # host checks there run only on the metrics-reading update of each
        # iteration — see `resolve_finite_check_flags`.  CPU/eager CUDA paths
        # retain the explicit per-update safety behavior.
        self._host_finite_checks, self._metrics_finite_checks = resolve_finite_check_flags(
            self._device_type,
            device_gated=self._device_type == "cuda" and self.use_compile,
        )
        self._gradient_sync: Callable[[Iterable[torch.Tensor]], None] | None = None
        self._dp_whole_cycle_cuda_graph = bool(dp_whole_cycle_cuda_graph)
        self._gradient_graph_hooks: tuple[Callable[[], None], ...] | None = None
        self._gradient_loss_flag = torch.empty(
            (), device=device, dtype=torch.float32, requires_grad=True
        )
        self._gradient_loss_flag.grad = torch.zeros_like(self._gradient_loss_flag)
        self._actor_spec = (
            FlashSACActorSpec.from_config(actor_config) if actor_config is not None else None
        )
        graph_requested = (
            self._nvidia_cuda
            if use_whole_cycle_cuda_graph is None
            else bool(use_whole_cycle_cuda_graph)
        )
        if self._actor_spec is not None:
            if not self.actor_normalize_parameters:
                raise ValueError(
                    "actor_normalize_parameters only applies to the built-in "
                    "FlashSAC actor; custom actors (actor_config) own their "
                    "parameter normalization"
                )
            if graph_requested:
                raise ValueError(
                    "FlashSAC custom actors (actor_config) do not support the "
                    "whole-cycle CUDA graph; set "
                    "algo.algo_params.use_whole_cycle_cuda_graph=false"
                )
        self._aux_loss_names: tuple[str, ...] | None = None
        if self._actor_spec is not None:
            self.actor = build_custom_actor(
                self._actor_spec,
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=self.device,
            )
            self._actor_semantics = build_custom_actor_semantics(
                self._actor_spec,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_transform_id=cast(str, cast(Any, self.actor).action_transform_id),
            )
        else:
            self.actor = FlashSACActor(
                num_blocks=actor_num_blocks,
                input_dim=obs_dim,
                hidden_dim=actor_hidden_dim,
                action_dim=action_dim,
                noise_zeta_mu=actor_noise_zeta_mu,
                noise_zeta_max=actor_noise_zeta_max,
                device=self.device,
            )
            self._actor_semantics = build_builtin_actor_semantics(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_blocks=actor_num_blocks,
                hidden_dim=actor_hidden_dim,
                noise_zeta_mu=actor_noise_zeta_mu,
                noise_zeta_max=actor_noise_zeta_max,
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
        if not self.actor_normalize_parameters:
            _disable_parameter_normalization(self.actor)
            cast(Any, self.actor).normalize_parameters = _noop_normalize_parameters
        if not self.critic_normalize_parameters:
            _disable_parameter_normalization(self.critic)
            self.target_critic.load_state_dict(self.critic.state_dict())
            cast(Any, self.critic).normalize_parameters = _noop_normalize_parameters
        self.temperature = FlashSACTemperature(temp_initial_value).to(self.device)
        self._optimizer_grad_scale = torch.ones((), device=self.device)
        self._optimizer_found_inf = torch.zeros((), device=self.device)

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

        # GradScaler is only needed for fp16 (cuda); bf16 doesn't need it.
        self.scaler: Any | None = (
            getattr(torch.amp, "GradScaler")("cuda")
            if self._should_use_grad_scaler(self.use_amp, self.device.type, self._amp_dtype)
            else None
        )
        lr_peak = learning_rate_peak if learning_rate_peak > 0 else actor_lr
        # Fused Adam collapses the per-parameter host loop (and its per-param
        # blocking `.item()` syncs on MPS) into one kernel; `capturable`
        # stays CUDA-only.
        _fused = fused_adam_supported(self.device.type)
        optimizer_kwargs: dict[str, Any] = {"fused": _fused}
        if _fused and self._performance_tier.supports_capturable_optimizer:
            optimizer_kwargs["capturable"] = True
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_peak, **optimizer_kwargs)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_peak, **optimizer_kwargs)
        self.temperature_optimizer = optim.Adam(
            self.temperature.parameters(), lr=lr_peak, **optimizer_kwargs
        )
        self._pending_actor_metric_values: torch.Tensor | None = None

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

        self._lr_schedule_fn = scheduler_fn
        self._lr_peak = lr_peak
        self._pending_cycle_critic_metric_values: torch.Tensor | None = None
        self._pending_cycle_metric_values: torch.Tensor | None = None
        self._critic_update_finite = torch.ones((), device=self.device)
        self._target_tau = torch.full((), tau, device=self.device)
        self._update_cycle_graph: torch.cuda.CUDAGraph | None = None
        self._update_cycle_graph_cache_key: tuple[object, ...] | None = None
        self._update_cycle_static_batch: dict[str, torch.Tensor] | None = None
        self._update_cycle_graph_metric_values: torch.Tensor | None = None
        self._update_cycle_lr_specs: list[
            tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR, torch.Tensor, int]
        ] = []
        self._update_cycle_lr_cursors: dict[int, int] = {}
        self._capture_update_cycle = False
        self._update_cycle_capture_failed = False
        # Whole-cycle capture wraps compiled objectives, so disabling compile
        # also disables this dependent layer.  The graph switch otherwise does
        # not change which objective boundary Inductor compiles.
        self._compile_full_update_cycle = bool(
            graph_requested and self.use_compile and self._nvidia_cuda
        )
        if self._compile_full_update_cycle and self.scaler is not None:
            raise ValueError(
                "FlashSAC CUDA compile mode requires bf16 (or fp32); "
                "fp16 GradScaler is incompatible with the whole-cycle graph"
            )
        if self._compile_full_update_cycle:
            if obs_normalization:
                raise ValueError(
                    "FlashSAC whole-cycle CUDA graphs do not yet support obs normalization"
                )
        if self.use_compile:
            if self._compile_full_update_cycle:
                self._materialize_capturable_optimizer_state()
            self._compile_training_methods()

    @property
    def use_update_cycle(self) -> bool:
        """Whether this learner owns the whole-cycle update orchestration."""
        return bool(
            self._compile_full_update_cycle
            and (
                self._gradient_sync is None
                or (self._dp_whole_cycle_cuda_graph and self._gradient_graph_hooks is not None)
            )
        )

    def _whole_cycle_optimizers(self) -> tuple[optim.Optimizer, ...]:
        return (self.critic_optimizer, self.actor_optimizer, self.temperature_optimizer)

    def _whole_cycle_modules(self) -> tuple[nn.Module, ...]:
        return (self.actor, self.critic, self.target_critic, self.temperature)

    def _whole_cycle_schedulers(self) -> tuple[Any, ...]:
        return (self.critic_scheduler, self.actor_scheduler, self.temperature_scheduler)

    def _whole_cycle_finite_gate(self) -> torch.Tensor:
        return self._critic_update_finite

    def _whole_cycle_target_modules(self) -> tuple[nn.Module, nn.Module]:
        return self.target_critic, self.critic

    def set_gradient_sync(self, sync: Callable[[Iterable[torch.Tensor]], None] | None) -> None:
        """Attach DP reduction; graph collectives require explicit opt-in hooks."""
        if sync != self._gradient_sync:
            self._reset_update_cycle_graph()
            self._gradient_graph_hooks = None
        if (
            sync is not None
            and self._compile_full_update_cycle
            and not self._dp_whole_cycle_cuda_graph
        ):
            self._compile_full_update_cycle = False
            self._reset_update_cycle_graph()
        self._gradient_sync = sync

    def set_gradient_graph_hooks(
        self,
        begin_capture: Callable[[], None],
        end_capture: Callable[[], None],
        before_replay: Callable[[], None],
    ) -> None:
        """Bind graph lifetime to the distributed synchronizer's lifetime."""
        hooks = (begin_capture, end_capture, before_replay)
        if hooks != self._gradient_graph_hooks:
            self._reset_update_cycle_graph()
        self._gradient_graph_hooks = hooks

    def _sync_loss_gradients(self, parameters: Iterable[torch.Tensor], loss: torch.Tensor) -> None:
        """Reduce loss validity with gradients so all ranks gate each step alike."""
        if (
            self._gradient_sync is None
            or self._device_type != "cuda"
            or self._host_finite_checks
            or self.scaler is not None
        ):
            self._sync_gradients(parameters)
            return
        sentinel = self._gradient_loss_flag
        assert sentinel.grad is not None
        sentinel.grad.copy_(torch.logical_not(torch.isfinite(loss.detach()).all()))
        self._sync_gradients((*parameters, sentinel))

    def _compile_training_methods(self) -> None:
        compile_fn = get_torch_compile_for_device(self.device, warn=True)
        if compile_fn is None:
            return

        if self._compile_full_update_cycle:
            # The raw whole-cycle graph must not nest Inductor CUDA Graph Trees.
            # Architecture-portable max-autotune lets Triton select kernels on
            # the installed GPU instead of restoring Ada/Blackwell learner paths.
            compile_kwargs = {
                "dynamic": False,
                "mode": "max-autotune-no-cudagraphs",
            }
        else:
            compile_kwargs = {
                "dynamic": False,
                "options": {
                    "triton.cudagraphs": True,
                },
            }
        if self.compile_full_objectives:
            self._full_critic_objective = compile_fn(  # type: ignore[method-assign]
                self._full_critic_objective, **compile_kwargs
            )
            self._full_actor_objective = compile_fn(  # type: ignore[method-assign]
                self._full_actor_objective, **compile_kwargs
            )
            return
        self.actor.get_mean_and_std = compile_fn(  # type: ignore[method-assign]
            self.actor.get_mean_and_std, **compile_kwargs
        )
        self.__dict__["_critic_objective"] = compile_fn(
            self._critic_objective,
            **compile_kwargs,
        )
        self.__dict__["_actor_objective"] = compile_fn(
            self._actor_objective,
            **compile_kwargs,
        )

    def _arm_optimizer_finite_gate(
        self,
        optimizer: optim.Optimizer,
        loss: torch.Tensor,
    ) -> None:
        """Arm a persistent graph-safe gate for fused CUDA Adam."""
        if self._host_finite_checks or self._device_type != "cuda":
            return
        found_inf = self._optimizer_found_inf
        found_inf.copy_(torch.logical_not(torch.isfinite(loss.detach()).all()))
        if self._gradient_sync is not None:
            if self.scaler is None:
                assert self._gradient_loss_flag.grad is not None
                found_inf.copy_(self._gradient_loss_flag.grad > 0)
            gradients = [
                parameter.grad
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if gradients:
                torch._amp_foreach_non_finite_check_and_unscale_(
                    gradients,
                    found_inf,
                    self._optimizer_grad_scale,
                )
        if optimizer is self.critic_optimizer:
            self._critic_update_finite.copy_(torch.logical_not(found_inf.detach()))
        setattr(optimizer, "grad_scale", self._optimizer_grad_scale)
        setattr(optimizer, "found_inf", found_inf)

    def _stage_update_cycle_lr(self, optimizer: optim.Optimizer) -> None:
        """Bind the next preallocated LR tensor while capturing a cycle."""
        if not self._capture_update_cycle:
            return
        cursor = self._update_cycle_lr_cursors.get(id(optimizer), 0)
        specs = [spec for spec in self._update_cycle_lr_specs if spec[0] is optimizer]
        if cursor >= len(specs):
            raise RuntimeError("FlashSAC update-cycle LR staging exhausted")
        optimizer.param_groups[0]["lr"] = specs[cursor][2]
        self._update_cycle_lr_cursors[id(optimizer)] = cursor + 1

    def _maybe_normalize_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if isinstance(self.obs_normalizer, nn.Identity):
            return obs
        normalizer = cast(EmpiricalNormalization, self.obs_normalizer)
        if update:
            self._update_obs_normalizer(obs)
            return cast(torch.Tensor, normalizer(obs, update=False))
        return cast(torch.Tensor, normalizer(obs, update=False))

    def update_reward_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if self.reward_normalizer is None:
            return
        self.reward_normalizer.update_from_transitions(rewards, dones)

    def _critic_objective(
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
        target_probs = compute_categorical_td_target(
            support=support,
            target_log_probs=next_q_log_probs,
            reward=rewards,
            dones=dones,
            truncated=truncated,
            actor_entropy=actor_entropy,
            gamma=gamma,
        )
        return cast(torch.Tensor, -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1).mean())

    def _actor_objective(
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

    def _full_critic_objective(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
        critic_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full critic forward/target/projection/loss objective."""
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
            critic_loss = self._critic_objective(
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
        reward_scale_std = (
            torch.sqrt(self.reward_normalizer.rms.var)
            if self.reward_normalizer is not None
            else torch.ones((), device=self.device)
        )
        return critic_loss, reward_scale_std

    def _full_actor_objective(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        expert_actions: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Full actor/critic forward and actor objective.

        The third return value carries the actor-owned auxiliary losses
        (empty for the built-in actor); they are already included in
        ``actor_loss`` and are returned separately only for diagnostics.
        """
        obs_all = torch.cat([obs, next_obs], dim=0)
        with self._autocast():
            actions_all, actor_info_all = self.actor(obs_all, training=True)
            actions = actions_all.chunk(2, dim=0)[0]
            log_probs = actor_info_all["log_prob"].chunk(2, dim=0)[0]
            q_values, _ = self.critic(critic_obs, actions, training=False)
            actor_loss, entropy = self._actor_objective(
                log_probs, q_values, actions, expert_actions, self.temperature()
            )
            aux_losses = validate_aux_losses(
                actor_info_all.get("aux_losses"), context="FlashSAC actor"
            )
            for aux_loss in aux_losses.values():
                actor_loss = actor_loss + aux_loss
        return actor_loss, entropy, dict(aux_losses)

    def update_critic(
        self,
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> dict[str, float]:
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

        critic_loss, reward_scale_std = self._full_critic_objective(
            actions, rewards, next_obs, dones, truncated, critic_obs, critic_next_obs
        )

        self._zero_optimizer_gradients(self.critic_optimizer)
        if self._finite_check_ok(critic_loss, read_metrics):
            if self.scaler is not None:
                self.scaler.scale(critic_loss).backward()
                self._sync_loss_gradients(self.critic.parameters(), critic_loss)
                self.scaler.unscale_(self.critic_optimizer)
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
            else:
                critic_loss.backward()
                self._sync_loss_gradients(self.critic.parameters(), critic_loss)
                self._arm_optimizer_finite_gate(self.critic_optimizer, critic_loss)
                self._stage_update_cycle_lr(self.critic_optimizer)
                self.critic_optimizer.step()
        if not self._capture_update_cycle:
            self.critic_scheduler.step()
        self.critic.normalize_parameters()

        if not read_metrics:
            critic_tensors: tuple[torch.Tensor, ...] = (critic_loss,)
            if self.reward_normalizer is not None:
                critic_tensors = (*critic_tensors, reward_scale_std)
            self._pending_cycle_critic_metric_values = torch.stack(
                [tensor.detach().reshape(()) for tensor in critic_tensors]
            )
            return {}
        metric_names: tuple[str, ...] = ("Loss/critic",)
        metric_tensors: tuple[torch.Tensor, ...] = (critic_loss,)
        if self.reward_normalizer is not None:
            metric_names = (*metric_names, "Train/reward_scale_std")
            metric_tensors = (*metric_tensors, reward_scale_std)
        return self._read_metric_tensors(
            metric_names,
            metric_tensors,
        )

    def update_actor(
        self,
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> dict[str, float]:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        expert_actions = batch["actions"].to(self.device)
        critic_obs = batch["critic"].to(self.device)

        obs = self._maybe_normalize_obs(obs, update=False)
        next_obs = self._maybe_normalize_obs(next_obs, update=False)

        self._pending_actor_metric_values = None
        actor_loss, entropy, aux_losses = self._full_actor_objective(
            obs,
            next_obs,
            expert_actions,
            critic_obs,
        )
        aux_names = tuple(sorted(aux_losses))
        if self._aux_loss_names is None:
            self._aux_loss_names = aux_names
        elif aux_names != self._aux_loss_names:
            raise RuntimeError(
                "FlashSAC actor auxiliary loss names changed between updates: "
                f"expected {self._aux_loss_names}, got {aux_names}. Auxiliary "
                "losses must be structurally stable across training."
            )

        self._zero_optimizer_gradients(self.actor_optimizer)
        if self._finite_check_ok(actor_loss, read_metrics):
            if self.scaler is not None:
                self.scaler.scale(actor_loss).backward()
                self._sync_loss_gradients(self.actor.parameters(), actor_loss)
                self.scaler.unscale_(self.actor_optimizer)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                actor_loss.backward(inputs=list(self.actor.parameters()))
                self._sync_loss_gradients(self.actor.parameters(), actor_loss)
                self._arm_optimizer_finite_gate(self.actor_optimizer, actor_loss)
                self._stage_update_cycle_lr(self.actor_optimizer)
                self.actor_optimizer.step()
        if not self._capture_update_cycle:
            self.actor_scheduler.step()
        cast(Any, self.actor).normalize_parameters()

        temp_value = self.temperature()
        temp_loss = temp_value * (entropy - self.target_entropy)
        self._zero_optimizer_gradients(self.temperature_optimizer)
        if self._finite_check_ok(temp_loss, read_metrics):
            temp_loss.backward(inputs=list(self.temperature.parameters()))
            self._sync_loss_gradients(self.temperature.parameters(), temp_loss)
            self._arm_optimizer_finite_gate(self.temperature_optimizer, temp_loss)
            self._stage_update_cycle_lr(self.temperature_optimizer)
            self.temperature_optimizer.step()
        if not self._capture_update_cycle:
            self.temperature_scheduler.step()

        post_update_temperature = self.temperature()
        actor_metric_tensors = (actor_loss, entropy, post_update_temperature, temp_loss)
        if not read_metrics:
            # Keep a private device-side snapshot.  The cycle-end drain below
            # performs the only D2H read, after all compiled replays finish.
            # Auxiliary-loss diagnostics only exist on the metrics-reading
            # path; their values remain part of Loss/actor here.
            self._pending_actor_metric_values = torch.stack(
                [tensor.detach().reshape(()) for tensor in actor_metric_tensors]
            )
            return {}
        aux_metric_names = tuple(f"Loss/aux_{name}" for name in aux_names)
        aux_metric_tensors = tuple(aux_losses[name] for name in aux_names)
        return self._read_metric_tensors(
            (
                "Loss/actor",
                "Loss/entropy",
                "Policy/temperature",
                "Loss/temperature",
                *aux_metric_names,
            ),
            (*actor_metric_tensors, *aux_metric_tensors),
        )

    @staticmethod
    def _read_metric_tensors(
        names: tuple[str, ...],
        tensors: tuple[torch.Tensor, ...],
    ) -> dict[str, float]:
        values = torch.stack([tensor.detach().reshape(()) for tensor in tensors]).cpu().tolist()
        return {name: float(value) for name, value in zip(names, values, strict=True)}

    def read_deferred_actor_metrics(self) -> dict[str, float]:
        values = self._pending_actor_metric_values
        self._pending_actor_metric_values = None
        if values is None:
            return {}
        return {
            name: float(value)
            for name, value in zip(
                (
                    "Loss/actor",
                    "Loss/entropy",
                    "Policy/temperature",
                    "Loss/temperature",
                ),
                values.cpu().tolist(),
                strict=True,
            )
        }

    def read_deferred_cycle_metrics(self) -> dict[str, float]:
        values = self._pending_cycle_metric_values
        critic_metric_count = 2 if self.reward_normalizer is not None else 1
        has_actor = values is not None and values.numel() == critic_metric_count + 4
        self._pending_cycle_critic_metric_values = None
        self._pending_actor_metric_values = None
        self._pending_cycle_metric_values = None
        if values is None:
            return {}
        metric_names: tuple[str, ...] = ("Loss/critic",)
        if self.reward_normalizer is not None:
            metric_names = (*metric_names, "Train/reward_scale_std")
        if has_actor:
            metric_names = metric_names + (
                "Loss/actor",
                "Loss/entropy",
                "Policy/temperature",
                "Loss/temperature",
            )
        return {
            name: float(value)
            for name, value in zip(metric_names, values.cpu().tolist(), strict=True)
        }

    def _begin_update_cycle_metrics(self) -> None:
        self._pending_actor_metric_values = None
        self._pending_cycle_critic_metric_values = None
        self._pending_cycle_metric_values = None
        self._update_cycle_lr_cursors = {}

    def _update_cycle_graph_key(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> tuple[object, ...]:
        shapes = tuple((key, tuple(value.shape), value.dtype) for key, value in large_batch.items())
        reward_state_shape = (
            tuple(self.reward_normalizer.g_r.shape) if self.reward_normalizer is not None else None
        )
        return (
            updates_per_step,
            policy_frequency,
            target_frequency,
            policy_before_critic,
            shapes,
            reward_state_shape,
        )

    def _prepare_update_cycle_lr_specs(
        self,
        *,
        updates_per_step: int,
        policy_frequency: int,
    ) -> None:
        actor_updates = len(range(0, updates_per_step, policy_frequency))
        self._update_cycle_lr_specs = []
        for scheduler, optimizer, count in (
            (self.critic_scheduler, self.critic_optimizer, updates_per_step),
            (self.actor_scheduler, self.actor_optimizer, actor_updates),
            (self.temperature_scheduler, self.temperature_optimizer, actor_updates),
        ):
            for offset in range(count):
                tensor = torch.empty((), device=self.device, dtype=torch.float32)
                self._update_cycle_lr_specs.append((optimizer, scheduler, tensor, offset))

    def _fill_update_cycle_lr_specs(self, *, zero: bool) -> None:
        for _optimizer, scheduler, tensor, offset in self._update_cycle_lr_specs:
            if zero:
                tensor.zero_()
                continue
            lr = self._lr_peak * self._lr_schedule_fn(scheduler.last_epoch + offset)
            tensor.fill_(float(lr))

    def _advance_update_cycle_schedulers(self) -> None:
        counts: dict[int, tuple[optim.lr_scheduler.LambdaLR, int]] = {}
        for _optimizer, scheduler, _tensor, offset in self._update_cycle_lr_specs:
            key = id(scheduler)
            _scheduler, current = counts.get(key, (scheduler, 0))
            counts[key] = (scheduler, max(current, offset + 1))
        for scheduler, count in counts.values():
            for _ in range(count):
                scheduler.step()

    def _prepare_update_cycle_warmup(self, *, updates_per_step: int, policy_frequency: int) -> None:
        self._prepare_update_cycle_lr_specs(
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
        )
        self._fill_update_cycle_lr_specs(zero=True)

    def _before_update_cycle_capture(
        self,
        saved_groups: Sequence[dict[str, Any]],
        *,
        updates_per_step: int,
        policy_frequency: int,
    ) -> tuple[int, ...]:
        del saved_groups
        if self._gradient_graph_hooks is not None:
            self._gradient_graph_hooks[0]()
        return (updates_per_step, policy_frequency)

    def _prepare_update_cycle_capture(self, *capture_state: Any) -> None:
        updates_per_step, policy_frequency = capture_state
        self._prepare_update_cycle_lr_specs(
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
        )
        self._fill_update_cycle_lr_specs(zero=False)

    def _finish_update_cycle_capture(self, saved_groups: Sequence[dict[str, Any]]) -> None:
        try:
            self._restore_optimizer_groups(self._whole_cycle_optimizers(), saved_groups)
        finally:
            if self._gradient_graph_hooks is not None:
                self._gradient_graph_hooks[1]()

    @property
    def _update_cycle_capture_error_mode(self) -> str:
        return "thread_local"

    @property
    def _whole_cycle_capture_fallback(self) -> bool:
        # Mixed eager/graph collectives across ranks can deadlock or diverge.
        return not (self._gradient_sync is not None and self._dp_whole_cycle_cuda_graph)

    def _whole_cycle_capture_failure_message(self) -> str:
        return (
            "FlashSAC whole-cycle CUDA graph capture failed; falling back to "
            "eager update-cycle execution for the rest of training."
        )

    def update_cycle(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
        read_metrics: bool = False,
    ) -> None:
        """Run the complete learner block used by the off-policy runner."""
        del read_metrics
        if not self.use_update_cycle:
            raise RuntimeError(
                "FlashSAC update_cycle() requires the NVIDIA CUDA whole-cycle path; "
                "use the per-update methods on the compatibility performance tiers"
            )
        if not self._update_cycle_capture_failed:
            self._ensure_update_cycle_graph(
                large_batch,
                updates_per_step=updates_per_step,
                policy_frequency=policy_frequency,
                target_frequency=target_frequency,
                policy_before_critic=policy_before_critic,
            )
        if self._update_cycle_capture_failed:
            # Eager fallback after a failed capture: the compiled objectives stay
            # active, only graph replay is lost. Schedulers step inside the
            # per-update methods on this path (not capturing).
            self._run_update_cycle_core(
                large_batch,
                updates_per_step=updates_per_step,
                policy_frequency=policy_frequency,
                target_frequency=target_frequency,
                policy_before_critic=policy_before_critic,
            )
            return
        assert self._update_cycle_graph is not None
        self._fill_update_cycle_lr_specs(zero=False)
        if self._gradient_graph_hooks is not None:
            self._gradient_graph_hooks[2]()
        self._update_cycle_graph.replay()
        self._advance_update_cycle_schedulers()
        self._pending_cycle_metric_values = self._update_cycle_graph_metric_values

    def _reset_update_cycle_graph(self) -> None:
        """Invalidate captured storage after checkpoint or DP state changes."""
        self._update_cycle_capture_failed = False
        self._update_cycle_graph = None
        self._update_cycle_graph_cache_key = None
        self._update_cycle_static_batch = None
        self._update_cycle_graph_metric_values = None
        self._update_cycle_lr_specs = []
        self._update_cycle_lr_cursors = {}
        self._pending_actor_metric_values = None
        self._pending_cycle_critic_metric_values = None
        self._pending_cycle_metric_values = None

    def dp_initial_sync(self) -> dict[str, torch.Tensor]:
        """Model state broadcast once from rank 0 before collection starts.

        The values alias the parameter/buffer storage of ``actor``, ``critic``,
        ``target_critic`` and ``temperature`` rather than copies. Optimizer
        state starts empty and remains aligned because every actual optimizer
        update uses the same cross-rank mean gradient. Observation/reward
        normalizer statistics remain rank-local.
        """
        tensors: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("actor", self.actor),
            ("critic", self.critic),
            ("target_critic", self.target_critic),
            ("temperature", self.temperature),
        ):
            for key, value in module.state_dict().items():
                tensors[f"{prefix}.{key}"] = value
        return tensors

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
            # Actor construction/observation/action semantics, validated
            # fail-closed on load (see unilab.rl.algos.flash_sac.actor_spec).
            "actor_semantics": copy.deepcopy(self._actor_semantics),
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
        validate_actor_semantics(state_dict.get("actor_semantics"), self._actor_semantics)
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
        self._reset_update_cycle_graph()
