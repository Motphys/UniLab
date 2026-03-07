"""Neural network architectures for RL algorithms."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalQNetwork(nn.Module):
    """Single distributional Q-network (C51 variant).

    Architecture: Linear→ReLU → Linear→ReLU → Linear→ReLU → Linear
    Outputs num_atoms logits over the value distribution.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x

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
        """Categorical projection (Bellman update on the distribution support)."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        is_int = (l == u)
        l_mask = is_int & (l > 0)
        u_mask = is_int & (l == 0)

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        return proj_dist


class Critic(nn.Module):
    """Twin distributional Q-networks for off-policy RL (SAC/TD3)."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs, n_act=n_act, num_atoms=num_atoms,
            v_min=v_min, v_max=v_max, hidden_dim=hidden_dim, device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs, n_act=n_act, num_atoms=num_atoms,
            v_min=v_min, v_max=v_max, hidden_dim=hidden_dim, device=device,
        )

        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
        )
        self.device = device

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ):
        """Projection operation using both Q-networks."""
        q1_proj = self.qnet1.projection(
            obs, actions, rewards, bootstrap, discount,
            self.q_support, self.q_support.device,
        )
        q2_proj = self.qnet2.projection(
            obs, actions, rewards, bootstrap, discount,
            self.q_support, self.q_support.device,
        )
        return q1_proj, q2_proj

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from probability distribution using support."""
        return torch.sum(probs * self.q_support, dim=1)
