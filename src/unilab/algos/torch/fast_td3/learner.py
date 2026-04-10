"""FastTD3 Learner — aligned with reference FastTD3 repository.

Architecture (from reference fast_td3.py):
- Actor: ReLU MLP (hidden → hidden//2 → hidden//4 → n_act, Tanh)
  - Per-env noise scales (sampled uniformly, resampled on episode done)
  - Small init scale for output layer
- Critic: Twin Distributional Q-Networks (C51 variant)
  - ReLU MLP with num_atoms output
- Observation normalization with EmpiricalNormalization
- AdamW optimizer with weight_decay=0.1
- Cosine LR scheduler

Hyperparameters aligned with reference Go1JoystickFlatTerrain config.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unilab.algos.torch.common.networks import Critic
from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.common.stability import check_nan_loss, clip_gradients

# ---------------------------------------------------------------------------
# Actor (deterministic, ReLU, per-env noise)
# ---------------------------------------------------------------------------


class TD3Actor(nn.Module):
    """Deterministic actor with per-environment exploration noise.

    Architecture: Linear→ReLU → Linear→ReLU → Linear→ReLU → Linear→Tanh
    Each environment has its own noise scale, sampled uniformly in [std_min, std_max].
    Noise scales are resampled when an episode ends.
    """

    noise_scales: torch.Tensor
    log_std_min: torch.Tensor
    log_std_max: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        log_std_min: float = -3.0,
        log_std_max: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )
        fc_mu_linear = self.fc_mu[0]
        assert isinstance(fc_mu_linear, nn.Linear)
        nn.init.normal_(fc_mu_linear.weight, 0.0, init_scale)
        nn.init.constant_(fc_mu_linear.bias, 0.0)

        std_min = float(math.exp(log_std_min))
        std_max = float(math.exp(log_std_max))
        noise_scales = torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("log_std_min", torch.as_tensor(log_std_min, device=device))
        self.register_buffer("log_std_max", torch.as_tensor(log_std_max, device=device))
        self.n_envs = num_envs
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        action: torch.Tensor = self.fc_mu(x)  # type: ignore[assignment]
        return action

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Forward pass with per-env exploration noise."""
        if isinstance(dones, bool):
            deterministic = dones
            dones = None

        if dones is not None and dones.sum() > 0:
            std_min = torch.exp(self.log_std_min)
            std_max = torch.exp(self.log_std_max)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device) * (std_max - std_min) + std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(torch.where(dones_view, new_scales, self.noise_scales))

        act = self.forward(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return (act + noise).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# FastTD3 Learner
# ---------------------------------------------------------------------------


class FastTD3Learner:
    """FastTD3 learner aligned with reference FastTD3 repository.

    Key hyperparameters (from Go1JoystickFlatTerrain):
    - gamma=0.97, tau=0.1
    - AdamW with weight_decay=0.1
    - Cosine LR schedule
    - Distributional critic (C51, num_atoms=101, v_min/max=±10)
    - CDQ (Clipped Double Q-learning) toggle
    - Observation normalization
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_envs: int = 1024,
        device: str = "cpu",
        # Hyperparameters from reference
        gamma: float = 0.97,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 1024,
        num_atoms: int = 101,
        v_min: float = -10.0,
        v_max: float = 10.0,
        init_scale: float = 0.01,
        log_std_min: float = -3.0,
        log_std_max: float = 0.0,
        weight_decay: float = 0.001,
        use_cdq: bool = True,
        # TD3-specific
        policy_noise: float = 0.1,
        noise_clip: float = 0.2,
        policy_frequency: int = 2,
        # Training
        max_iterations: int = 50000,
        obs_normalization: bool = True,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency
        self.use_cdq = use_cdq

        torch_device = torch.device(device)

        # Build actor
        self.actor = TD3Actor(
            obs_dim=obs_dim,
            n_act=action_dim,
            num_envs=num_envs,
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            device=torch_device,
        )
        self.actor_target = TD3Actor(
            obs_dim=obs_dim,
            n_act=action_dim,
            num_envs=num_envs,
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            device=torch_device,
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Build critic
        self.qnet = Critic(
            obs_dim=obs_dim,
            n_act=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            device=torch_device,
        )
        self.qnet_target = Critic(
            obs_dim=obs_dim,
            n_act=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            device=torch_device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Observation normalization
        self.obs_normalizer: Union[EmpiricalNormalization, nn.Identity]
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()

        # Optimizers (AdamW, reference style)
        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=torch.tensor(critic_lr, device=device),
            weight_decay=weight_decay,
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=torch.tensor(actor_lr, device=device),
            weight_decay=weight_decay,
        )

        self.update_count = 0
        self.weight_decay = weight_decay

    def normalize_obs(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        """Normalize observations using running statistics."""
        if not isinstance(self.obs_normalizer, nn.Identity):
            return self.obs_normalizer.forward(obs, update=update)
        return obs

    def update_critic(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One critic update step."""
        observations = self.normalize_obs(data["obs"], update=True)
        privileged = data.get("privileged", None)
        actions = data["actions"]
        rewards = data["rewards"]
        next_observations = self.normalize_obs(data["next_obs"], update=False)
        next_privileged = data.get("next_privileged", None)
        dones = data["dones"].bool()
        truncations = data["truncated"].bool()

        # Critic input: obs + privileged
        if privileged is not None:
            assert next_privileged is not None
            critic_obs = torch.cat([observations, privileged], dim=-1)
            critic_next_obs = torch.cat([next_observations, next_privileged], dim=-1)
        else:
            critic_obs = observations
            critic_next_obs = next_observations

        bootstrap = (truncations | ~dones).float()
        discount = torch.full_like(rewards, self.gamma)

        # Target policy smoothing
        clipped_noise = torch.randn_like(actions)
        clipped_noise = clipped_noise.mul(self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        next_state_actions = (self.actor_target(next_observations) + clipped_noise).clamp(-1.0, 1.0)

        with torch.no_grad():
            qf1_next_target_proj, qf2_next_target_proj = self.qnet_target.projection(
                critic_next_obs,
                next_state_actions,
                rewards,
                bootstrap,
                discount,
            )
            qf1_next_target_value = self.qnet_target.get_value(qf1_next_target_proj)
            qf2_next_target_value = self.qnet_target.get_value(qf2_next_target_proj)

            if self.use_cdq:
                # Clipped Double Q-learning: use distribution of the min-value Q
                qf_next_target_dist = torch.where(
                    qf1_next_target_value.unsqueeze(1) < qf2_next_target_value.unsqueeze(1),
                    qf1_next_target_proj,
                    qf2_next_target_proj,
                )
                qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
            else:
                qf1_next_target_dist = qf1_next_target_proj
                qf2_next_target_dist = qf2_next_target_proj

        qf1, qf2 = self.qnet(critic_obs, actions)
        qf1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
        qf2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
        qf_loss = qf1_loss + qf2_loss

        loss, nan_metrics = check_nan_loss(
            qf_loss,
            {
                "qf_loss": 0.0,
                "qf_max": 0.0,
                "qf_min": 0.0,
            },
        )
        if loss is None:
            return nan_metrics or {}

        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.weight_decay > 0:
            clip_gradients(self.qnet.parameters(), max_norm=10.0)
        self.q_optimizer.step()

        return {
            "qf_loss": qf_loss.item(),
            "qf_max": qf1_next_target_value.max().item(),
            "qf_min": qf1_next_target_value.min().item(),
        }

    def update_actor(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One actor update step."""
        observations = self.normalize_obs(data["obs"], update=False)

        qf1, qf2 = self.qnet(observations, self.actor(observations))
        qf1_value = self.qnet.get_value(F.softmax(qf1, dim=1))
        qf2_value = self.qnet.get_value(F.softmax(qf2, dim=1))

        if self.use_cdq:
            qf_value = torch.minimum(qf1_value, qf2_value)
        else:
            qf_value = (qf1_value + qf2_value) / 2.0

        actor_loss = -qf_value.mean()

        loss, nan_metrics = check_nan_loss(actor_loss, {"actor_loss": 0.0})
        if loss is None:
            return nan_metrics or {}

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.weight_decay > 0:
            clip_gradients(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item()}

    @torch.no_grad()
    def soft_update_target(self) -> None:
        """Polyak-average update of target networks."""
        src_ps = [p.data for p in self.qnet.parameters()]
        tgt_ps = [p.data for p in self.qnet_target.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - self.tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=self.tau)

        src_actor_ps = [p.data for p in self.actor.parameters()]
        tgt_actor_ps = [p.data for p in self.actor_target.parameters()]
        torch._foreach_mul_(tgt_actor_ps, 1.0 - self.tau)
        torch._foreach_add_(tgt_actor_ps, src_actor_ps, alpha=self.tau)

    @torch.no_grad()
    def soft_update(self) -> None:
        """Backward-compatible alias for older call sites."""
        self.soft_update_target()

    def get_state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "obs_normalizer": (
                self.obs_normalizer.state_dict()
                if hasattr(self.obs_normalizer, "state_dict")
                else None
            ),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        if "actor_target" in state_dict:
            self.actor_target.load_state_dict(state_dict["actor_target"])
        else:
            self.actor_target.load_state_dict(state_dict["actor"])
        self.qnet.load_state_dict(state_dict["qnet"])
        self.qnet_target.load_state_dict(state_dict["qnet_target"])
        if state_dict.get("obs_normalizer") and hasattr(self.obs_normalizer, "load_state_dict"):
            self.obs_normalizer.load_state_dict(state_dict["obs_normalizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.update_count = state_dict.get("update_count", 0)
