"""FlashSAC learner adapted to UniLab's off-policy contract."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.flash_sac.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from unilab.algos.torch.flash_sac.update import (
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
    """Adaptive reward scaling with running reward statistics."""

    def __init__(self, g_max: float, device: torch.device, eps: float = 1e-8):
        self.g_max = g_max
        self.eps = eps
        self.device = device
        self.rms = RunningMeanStd.create(device)
        self.reward_abs_max = torch.tensor(0.0, device=device, dtype=torch.float32)

    def update(self, rewards: torch.Tensor) -> None:
        rewards = rewards.reshape(-1).to(device=self.device, dtype=torch.float32)
        if rewards.numel() == 0:
            return
        self.rms.update(rewards)
        self.reward_abs_max = torch.maximum(self.reward_abs_max, rewards.abs().max())

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        denominator = torch.maximum(
            torch.sqrt(self.rms.var + self.eps),
            self.reward_abs_max / max(self.g_max, self.eps),
        )
        return rewards / denominator

    def state_dict(self) -> dict[str, Any]:
        return {
            "rms": self.rms.state_dict(),
            "reward_abs_max": self.reward_abs_max,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rms.load_state_dict(state_dict["rms"])
        self.reward_abs_max = state_dict["reward_abs_max"]


class FlashSACLearner:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int = 0,
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
        use_compile: bool = False,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.actor_bc_alpha = actor_bc_alpha
        self.obs_dim = obs_dim
        self.critic_obs_dim = obs_dim + privileged_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self.update_count = 0
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.use_compile = bool(
            use_compile and hasattr(torch, "compile") and self.device.type == "cuda"
        )

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
            RewardNormalizer(g_max=normalized_g_max, device=self.device)
            if normalize_reward
            else None
        )

        self.scaler: Any | None = getattr(torch.amp, "GradScaler")("cuda") if self.use_amp else None
        lr_peak = learning_rate_peak if learning_rate_peak > 0 else actor_lr
        fused = self.device.type == "cuda"
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_peak, fused=fused)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_peak, fused=fused)
        self.temperature_optimizer = optim.Adam(
            self.temperature.parameters(), lr=lr_peak, fused=fused
        )

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
            self.actor.get_mean_and_std = torch.compile(  # type: ignore[method-assign]
                self.actor.get_mean_and_std, mode="reduce-overhead"
            )

    def _maybe_normalize_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if isinstance(self.obs_normalizer, nn.Identity):
            return obs
        return cast(torch.Tensor, self.obs_normalizer(obs, update=update))

    def _build_critic_obs(
        self,
        obs: torch.Tensor,
        privileged: torch.Tensor | None,
    ) -> torch.Tensor:
        if privileged is None:
            return obs
        return torch.cat([obs, privileged], dim=-1)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp)

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        terminated = batch["dones"].to(self.device)
        truncated = batch.get("truncated", torch.zeros_like(terminated)).to(self.device)
        privileged = batch.get("privileged")
        next_privileged = batch.get("next_privileged")
        privileged = privileged.to(self.device) if privileged is not None else None
        next_privileged = next_privileged.to(self.device) if next_privileged is not None else None

        obs = self._maybe_normalize_obs(obs, update=True)
        next_obs = self._maybe_normalize_obs(next_obs, update=False)
        critic_obs = self._build_critic_obs(obs, privileged)
        critic_next_obs = self._build_critic_obs(next_obs, next_privileged)

        if self.reward_normalizer is not None:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        gamma = self.gamma**self.n_step

        with torch.no_grad():
            with self._autocast():
                next_actions, actor_info = self.actor(next_obs, training=False)
                entropy_bonus = -self.temperature().detach() * actor_info["log_prob"]
                next_q_values, next_q_info = self.target_critic(
                    critic_next_obs, next_actions, training=False
                )
                next_q_log_probs = select_min_q_log_probs(next_q_values, next_q_info["log_prob"])
                support = cast(torch.Tensor, self.target_critic.predictor.support)
                target_probs = compute_categorical_td_target(
                    support=support,
                    target_log_probs=next_q_log_probs,
                    reward=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    actor_entropy=entropy_bonus,
                    gamma=gamma,
                )

        with self._autocast():
            _, pred_info = self.critic(critic_obs, actions, training=True)
            critic_loss = -(target_probs.unsqueeze(0) * pred_info["log_prob"]).sum(dim=-1).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            critic_loss.backward()
            self.critic_optimizer.step()
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
        expert_actions = batch["actions"].to(self.device)
        privileged = batch.get("privileged")
        privileged = privileged.to(self.device) if privileged is not None else None

        obs = self._maybe_normalize_obs(obs, update=False)
        critic_obs = self._build_critic_obs(obs, privileged)

        with self._autocast():
            actions, actor_info = self.actor(obs, training=True)
            log_probs = actor_info["log_prob"]

            self._set_requires_grad(self.critic, False)
            q_values, _ = self.critic(critic_obs, actions, training=False)
            self._set_requires_grad(self.critic, True)
            min_q = torch.min(q_values[0], q_values[1])

            actor_loss = (self.temperature().detach() * log_probs - min_q).mean()
            if self.actor_bc_alpha > 0:
                bc_loss = torch.mean((actions - expert_actions) ** 2)
                actor_loss = (
                    actor_loss + self.actor_bc_alpha * min_q.abs().mean().detach() * bc_loss
                )

        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            actor_loss.backward()
            self.actor_optimizer.step()
        self.actor_scheduler.step()
        self.actor.normalize_parameters()

        entropy = -log_probs.detach().mean()
        temp_value = self.temperature()
        temp_loss = temp_value * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temp_loss.backward()
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
