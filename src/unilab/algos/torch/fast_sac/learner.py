"""FastSAC Learner — replicated from holosoma's FastSAC implementation.

Network architecture:
- Actor: MLP with SiLU + LayerNorm, tanh-squashed Gaussian
- Critic: Distributional Q-Networks (C51 variant, num_atoms=101)
- Automatic entropy coefficient (alpha) learning

Hyperparameters aligned with holosoma FastSACConfig defaults.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unilab.base.augmentation import SymmetryAugmentation

# ---------------------------------------------------------------------------
# Actor Network (holosoma-style: SiLU + LayerNorm + Tanh squashing)
# ---------------------------------------------------------------------------


class SACActor(nn.Module):
    """Stochastic actor for SAC with tanh-squashed Gaussian policy.

    Architecture: Linear→LN→SiLU → Linear→LN→SiLU → Linear→LN→SiLU → fc_mu + fc_logstd
    Hidden dims: [hidden_dim, hidden_dim//2, hidden_dim//4]
    """

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: str | torch.device = "cpu",
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.use_tanh = use_tanh
        self.device_ = device  # avoid name collision with nn.Module.device

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 4, action_dim, device=device)
        self.fc_logstd = nn.Linear(hidden_dim // 4, action_dim, device=device)

        # Zero-init output heads (holosoma style)
        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

        # Action scaling
        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(action_dim, device=device))
        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(action_dim, device=device))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, mean, log_std)."""
        x = self.net(obs)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)

        # Squash log_std to [log_std_min, log_std_max] (SpinUp / Denis Yarats style)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        # NaN protection: clamp mean to prevent exploding values
        mean = torch.clamp(mean, -10.0, 10.0)
        mean = torch.nan_to_num(mean, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=self.log_std_min)

        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale + self.action_bias
        else:
            action = mean

        return action, mean, log_std

    def as_export_module(self) -> "nn.Module":
        """Return a single-input/single-output wrapper suitable for torch.onnx.export."""
        actor = self

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = actor

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                action, _, _ = self.base(obs)
                return cast(torch.Tensor, action)

        return _Wrapper()

    def get_actions_and_log_probs(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities. Returns (action, log_prob, log_std)."""
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
            log_prob = dist.log_prob(raw_action)
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        log_prob = log_prob.sum(1)
        return action, log_prob, log_std

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get exploration actions.

        Args:
            obs: Batched observations.
            dones: Unused for SAC; kept for API alignment with TD3 actor.
            deterministic: Whether to return deterministic policy actions.
        """
        # Backward compatibility: previous signature was explore(obs, deterministic=False).
        if isinstance(dones, bool):
            deterministic = dones
            dones = None
        _ = dones

        _, mean, log_std = self.forward(obs)
        if deterministic:
            if self.use_tanh:
                return torch.tanh(mean) * self.action_scale + self.action_bias
            return mean

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        return raw_action


# ---------------------------------------------------------------------------
# Distributional Q-Network (C51 variant, from holosoma)
# ---------------------------------------------------------------------------


class DistributionalQNetwork(nn.Module):
    """Single distributional Q-network (C51).

    Architecture: Linear→LN→SiLU → Linear→LN→SiLU → Linear→LN→SiLU → Linear(num_atoms)
    Input: concat(obs, action)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        input_dim = obs_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)  # type: ignore[no-any-return]

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Categorical projection for distributional RL."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)

        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1
        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class SACCritic(nn.Module):
    """Ensemble of distributional Q-networks for SAC.

    Uses ``num_q_networks`` independent DistributionalQNetwork instances.
    """

    q_support: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.num_q_networks = num_q_networks

        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    num_atoms=num_atoms,
                    v_min=v_min,
                    v_max=v_max,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                    device=device,
                )
                for _ in range(num_q_networks)
            ]
        )

        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns stacked logits: (num_q_nets, batch, num_atoms)."""
        outputs = [qnet(obs, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Project for all Q-networks: (num_q_nets, batch, num_atoms)."""
        projections = [
            qnet.projection(  # type: ignore[operator]
                obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device
            )
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from probabilities using support."""
        return torch.sum(probs * self.q_support, dim=-1)


# ---------------------------------------------------------------------------
# FastSACLearner — the training algorithm
# ---------------------------------------------------------------------------


class FastSACLearner:
    """FastSAC learner with holosoma-aligned hyperparameters.

    Key hyperparameters (aligned with holosoma FastSACConfig):
    - gamma=0.97, tau=0.125
    - batch_size=8192, num_updates=8, policy_frequency=4
    - alpha_init=0.001, target_entropy_ratio=0.0
    - AdamW with betas=(0.9, 0.95), weight_decay=0.001
    - Distributional critic (C51, num_atoms=101)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int,
        device: str = "cpu",
        # Hyperparameters aligned with holosoma
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.001,
        target_entropy_ratio: float = 0.0,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        num_q_networks: int = 2,
        use_layer_norm: bool = True,
        use_tanh: bool = True,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        weight_decay: float = 0.001,
        max_grad_norm: float = 0.0,
        use_autotune: bool = True,
        use_symmetry: bool = False,
        use_amp: bool = False,
        amp_dtype: str = "auto",
        use_compile: bool = False,
        symmetry_augmentation: SymmetryAugmentation | None = None,
        world_size: int = 1,
    ):
        self.device = device
        self._device_type = torch.device(device).type
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.use_autotune = use_autotune
        self.use_amp = bool(use_amp) and self._device_type in ("cuda", "xpu")
        self.use_compile = (
            bool(use_compile) and self._device_type == "cuda" and hasattr(torch, "compile")
        )
        self.amp_dtype = amp_dtype
        self._amp_dtype = self._resolve_amp_dtype(amp_dtype, self._device_type)
        self.world_size = world_size
        self.critic_obs_dim = critic_obs_dim

        # Build actor (uses obs only)
        self.actor = SACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )

        self.qnet = SACCritic(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            use_layer_norm=use_layer_norm,
            num_q_networks=num_q_networks,
            device=device,
        )

        # Target critic
        self.qnet_target = SACCritic(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            use_layer_norm=use_layer_norm,
            num_q_networks=num_q_networks,
            device=device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Entropy coefficient
        self.log_alpha = torch.tensor([math.log(alpha_init)], requires_grad=True, device=device)
        self.target_entropy = -action_dim * target_entropy_ratio

        # fused AdamW requires CUDA; MPS and CPU do not support it
        _fused = isinstance(device, str) and device.startswith("cuda")

        # Optimizers (AdamW with holosoma betas)
        self.q_optimizer = optim.AdamW(
            self.qnet.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
            fused=_fused,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            fused=_fused,
            betas=(0.9, 0.95),
        )
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha],
            lr=alpha_lr,
            fused=_fused,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

        # Step counter
        self.update_count = 0

        # AMP scaler for mixed precision (fp16 only; bf16 has fp32 range and skips scaler)
        self.scaler = (
            torch.amp.GradScaler("cuda")  # pyright: ignore[reportPrivateImportUsage]
            if self._should_use_grad_scaler(self.use_amp, self._device_type, self._amp_dtype)
            else None
        )

        self.symmetry = symmetry_augmentation
        if use_symmetry and symmetry_augmentation is None:
            raise ValueError(
                "FastSACLearner use_symmetry=True requires a symmetry_augmentation contract"
            )
        self.use_symmetry = use_symmetry
        if self.use_compile:
            self._compile_training_methods()

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str, device_type: str) -> torch.dtype:
        normalized = amp_dtype.lower()
        if normalized == "auto":
            return torch.bfloat16
        if normalized == "fp16":
            return torch.float16
        if normalized == "bf16":
            return torch.bfloat16
        raise ValueError("FastSAC amp_dtype must be one of: auto, fp16, bf16")

    @staticmethod
    def _should_use_grad_scaler(
        use_amp: bool,
        device_type: str,
        amp_dtype: torch.dtype,
    ) -> bool:
        return bool(use_amp) and device_type == "cuda" and amp_dtype == torch.float16

    def _compile_training_methods(self) -> None:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None or torch.device(self.device).type != "cuda":
            return

        compile_kwargs = {"options": {"triton.cudagraphs": False}}
        self._critic_loss_tensors = compile_fn(  # type: ignore[method-assign]
            self._critic_loss_tensors, **compile_kwargs
        )
        self._actor_loss_tensors = compile_fn(  # type: ignore[method-assign]
            self._actor_loss_tensors, **compile_kwargs
        )

    def _autocast(self):
        return torch.amp.autocast(  # pyright: ignore[reportPrivateImportUsage]
            self._device_type, dtype=self._amp_dtype, enabled=self.use_amp
        )

    def _reduce_gradients(self, model: nn.Module) -> None:
        """All-reduce gradients across all workers and divide by world_size.

        Must be called after ``backward()`` and, when using AMP, after
        ``scaler.unscale_(optimizer)`` so that gradients are in full precision.
        """
        if self.world_size <= 1:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= self.world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.grad.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    def _critic_loss_tensors(
        self,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bootstrap = torch.clamp(1.0 - dones.float() + truncated.float(), 0.0, 1.0)
        discount = torch.full_like(dones, self.gamma)

        with torch.no_grad():
            with self._autocast():
                next_actions, next_log_probs, _ = self.actor.get_actions_and_log_probs(next_obs)
            adjusted_rewards = (
                rewards - discount * bootstrap * self.log_alpha.exp() * next_log_probs
            )

            with self._autocast():
                target_distributions = self.qnet_target.projection(
                    critic_next_obs, next_actions, adjusted_rewards, bootstrap, discount
                )
                target_values = self.qnet_target.get_value(target_distributions)
                target_q_max = target_values.max()
                target_q_min = target_values.min()

        with self._autocast():
            q_outputs = self.qnet(critic_obs, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1).clamp(min=-30.0)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        return qf_loss, target_q_max, target_q_min, next_log_probs.detach()

    def _actor_loss_tensors(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._autocast():
            actions, log_probs, log_std = self.actor.get_actions_and_log_probs(obs)

        with torch.no_grad():
            action_std = log_std.exp().mean()
            policy_entropy = -log_probs.mean()

        with self._autocast():
            q_outputs = self.qnet(critic_obs, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = self.qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        return actor_loss, policy_entropy, action_std

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One critic update step."""
        obs = batch["obs"]
        critic_obs = batch["critic"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        critic_next_obs = batch["next_critic"]
        dones = batch["dones"]
        truncated = batch["truncated"]

        # Apply symmetry augmentation
        if self.use_symmetry:
            orig_actions = actions

            assert self.symmetry is not None
            obs, actions = self.symmetry.augment_obs_and_actions(obs, actions, obs_group="obs")
            next_obs, _ = self.symmetry.augment_obs_and_actions(
                next_obs, orig_actions, obs_group="obs"
            )

            critic_obs, _ = self.symmetry.augment_obs_and_actions(
                critic_obs,
                orig_actions,
                obs_group="critic",
            )
            critic_next_obs, _ = self.symmetry.augment_obs_and_actions(
                critic_next_obs,
                orig_actions,
                obs_group="critic",
            )

            # Double the batch size for other tensors
            rewards = rewards.repeat(2)
            dones = dones.repeat(2)
            truncated = truncated.repeat(2)

        qf_loss, target_q_max, target_q_min, next_log_probs = self._critic_loss_tensors(
            critic_obs,
            actions,
            rewards,
            next_obs,
            critic_next_obs,
            dones,
            truncated,
        )

        # Skip if NaN
        if torch.isfinite(qf_loss):
            self.q_optimizer.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(qf_loss).backward()
                self.scaler.unscale_(self.q_optimizer)
                self._reduce_gradients(self.qnet)
                if self.max_grad_norm > 0:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.qnet.parameters(), max_norm=self.max_grad_norm
                    )
                else:
                    critic_grad_norm = torch.tensor(0.0, device=self.device)
                self.scaler.step(self.q_optimizer)
                self.scaler.update()
            else:
                qf_loss.backward()
                self._reduce_gradients(self.qnet)
                if self.max_grad_norm > 0:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.qnet.parameters(), max_norm=self.max_grad_norm
                    )
                else:
                    critic_grad_norm = torch.tensor(0.0, device=self.device)
                self.q_optimizer.step()
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)

        # Alpha loss (temperature update) - matching holosoma
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.use_autotune:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (-self.log_alpha.exp() * (next_log_probs + self.target_entropy)).mean()
            if torch.isfinite(alpha_loss):
                alpha_loss.backward()
                if self.world_size > 1 and self.log_alpha.grad is not None:
                    dist.all_reduce(self.log_alpha.grad, op=dist.ReduceOp.SUM)
                    self.log_alpha.grad /= self.world_size
                self.alpha_optimizer.step()

        return {
            "qf_loss": qf_loss.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "target_q_max": target_q_max.item(),
            "target_q_min": target_q_min.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
        }

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One actor update step."""
        obs = batch["obs"]
        critic_obs = batch["critic"]

        # Apply symmetry augmentation
        if self.use_symmetry:
            assert self.symmetry is not None
            obs = torch.cat([obs, self.symmetry.mirror_obs(obs, obs_group="obs")], dim=0)
            critic_obs = torch.cat(
                [critic_obs, self.symmetry.mirror_obs(critic_obs, obs_group="critic")],
                dim=0,
            )

        actor_loss, policy_entropy, action_std = self._actor_loss_tensors(obs, critic_obs)

        # Skip if NaN
        if torch.isfinite(actor_loss):
            self.actor_optimizer.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                self._reduce_gradients(self.actor)
                if self.max_grad_norm > 0:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), max_norm=self.max_grad_norm
                    )
                else:
                    actor_grad_norm = torch.tensor(0.0, device=self.device)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                actor_loss.backward()
                self._reduce_gradients(self.actor)
                if self.max_grad_norm > 0:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), max_norm=self.max_grad_norm
                    )
                else:
                    actor_grad_norm = torch.tensor(0.0, device=self.device)
                self.actor_optimizer.step()
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)

        return {
            "actor_loss": actor_loss.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "policy_entropy": policy_entropy.item(),
            "action_std": action_std.item(),
        }

    def soft_update_target(self) -> None:
        """Polyak-average update of the target Q-network."""
        with torch.no_grad():
            for tgt, src in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                tgt.data.mul_(1.0 - self.tau).add_(src.data, alpha=self.tau)

    def get_state_dict(self) -> Dict[str, Any]:
        """Save all components."""
        return {
            "actor": self.actor.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load all components."""
        self.actor.load_state_dict(state_dict["actor"])
        self.qnet.load_state_dict(state_dict["qnet"])
        self.qnet_target.load_state_dict(state_dict["qnet_target"])
        self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.update_count = state_dict.get("update_count", 0)


# ---------------------------------------------------------------------------
