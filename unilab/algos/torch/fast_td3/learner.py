"""FastTD3 Learner — Twin Delayed DDPG with distributional critics.

Network architecture (replicated from holosoma):
- Actor: MLP with SiLU + LayerNorm, deterministic + exploration noise
- Critic: Distributional Q-Networks (C51 variant, num_atoms=101)

Hyperparameters aligned with holosoma FastSACConfig defaults.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# TD3 Actor (deterministic, SiLU + LayerNorm)
# ---------------------------------------------------------------------------

class TD3Actor(nn.Module):
    """Deterministic actor for TD3.

    Architecture: Linear→LN→SiLU → Linear→LN→SiLU → Linear→LN→SiLU → Linear→Tanh
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        use_layer_norm: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
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
            nn.Linear(hidden_dim // 4, action_dim, device=device),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# Reuse DistributionalQNetwork and Critic from FastSAC
from unilab.algos.torch.fast_sac.learner import DistributionalQNetwork, SACCritic


# ---------------------------------------------------------------------------
# FastTD3Learner
# ---------------------------------------------------------------------------

class FastTD3Learner:
    """FastTD3 learner with holosoma-aligned hyperparameters.

    Key hyperparameters (aligned with holosoma):
    - gamma=0.97, tau=0.125
    - batch_size=8192, policy_delay=4
    - AdamW with betas=(0.9, 0.95), weight_decay=0.001
    - Distributional critic (C51, num_atoms=101)
    - Target noise clipping for policy smoothing
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        # Hyperparameters aligned with holosoma
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        num_q_networks: int = 2,
        use_layer_norm: bool = True,
        weight_decay: float = 0.001,
        max_grad_norm: float = 0.0,
        # TD3-specific
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise

        # Build actor
        self.actor = TD3Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        self.actor_target = TD3Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Build critic ensemble (distributional)
        self.qnet = SACCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            use_layer_norm=use_layer_norm,
            num_q_networks=num_q_networks,
            device=device,
        )
        self.qnet_target = SACCritic(
            obs_dim=obs_dim,
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

        # Optimizers (AdamW, holosoma style)
        self.q_optimizer = optim.AdamW(
            self.qnet.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        self.update_count = 0

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One critic update step."""
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        bootstrap = (1.0 - dones).float()
        discount = torch.full_like(dones, self.gamma)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1, 1)

            # Distributional target
            target_distributions = self.qnet_target.projection(
                next_obs, next_actions, rewards, bootstrap, discount
            )
            target_values = self.qnet_target.get_value(target_distributions)

        # Critic loss (distributional cross-entropy)
        q_outputs = self.qnet(obs, actions)
        critic_log_probs = F.log_softmax(q_outputs, dim=-1)
        critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
        qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()

        critic_grad_norm = torch.tensor(0.0, device=self.device)
        if self.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.qnet.parameters(), max_norm=self.max_grad_norm
            )
        self.q_optimizer.step()

        return {
            "qf_loss": qf_loss.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "target_q_max": target_values.max().item(),
            "target_q_min": target_values.min().item(),
        }

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One actor update step."""
        obs = batch["obs"]
        actions = self.actor(obs)

        # Use mean Q-value across ensemble
        q_outputs = self.qnet(obs, actions)
        q_probs = F.softmax(q_outputs, dim=-1)
        q_values = self.qnet.get_value(q_probs)
        actor_loss = -q_values.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()

        actor_grad_norm = torch.tensor(0.0, device=self.device)
        if self.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.max_grad_norm
            )
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "actor_grad_norm": actor_grad_norm.item(),
        }

    def soft_update_targets(self) -> None:
        """Polyak-average update of target networks."""
        with torch.no_grad():
            # Actor target
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
            # Critic target
            for p, tp in zip(self.qnet.parameters(), self.qnet_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def get_state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.qnet.load_state_dict(state_dict["qnet"])
        self.qnet_target.load_state_dict(state_dict["qnet_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.update_count = state_dict.get("update_count", 0)
