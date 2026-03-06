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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict

# ---------------------------------------------------------------------------
# Distributional Q-Network (C51)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Critic (Twin Distributional Q-Networks)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """Twin distributional Q-networks for TD3."""

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


# ---------------------------------------------------------------------------
# Actor (deterministic, ReLU, per-env noise)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Deterministic actor with per-environment exploration noise.

    Architecture: Linear→ReLU → Linear→ReLU → Linear→ReLU → Linear→Tanh
    Each environment has its own noise scale, sampled uniformly in [std_min, std_max].
    Noise scales are resampled when an episode ends.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
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
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        action = self.fc_mu(x)
        return action

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        """Forward pass with per-env exploration noise."""
        # Resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )

        act = self(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return (act + noise).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Observation Normalization
# ---------------------------------------------------------------------------

class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of observations using running statistics."""

    def __init__(self, shape, device, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, center: bool = True, update: bool = True
    ) -> torch.Tensor:
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    def update(self, x):
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + batch_size

        # Welford's online algorithm
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class SimpleReplayBuffer(nn.Module):
    """Simple replay buffer shaped [n_env, buffer_size, ...].

    Matches the reference FastTD3 implementation with per-env circular buffers.
    """

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        device=None,
    ):
        super().__init__()
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.device = device

        self.observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (n_env, buffer_size, n_act), device=device, dtype=torch.float
        )
        self.rewards = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.dones = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.truncations = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.next_observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        self.ptr = 0

    @torch.no_grad()
    def extend(self, transition: Dict):
        """Add a single timestep of transitions for all environments."""
        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = transition["observations"]
        self.actions[:, ptr] = transition["actions"]
        self.rewards[:, ptr] = transition["rewards"]
        self.dones[:, ptr] = transition["dones"]
        self.truncations[:, ptr] = transition["truncations"]
        self.next_observations[:, ptr] = transition["next_observations"]
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int):
        """Sample a batch of transitions. Returns (n_env * batch_size) transitions."""
        max_idx = min(self.buffer_size, self.ptr)
        indices = torch.randint(
            0, max_idx, (self.n_env, batch_size), device=self.device
        )
        obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
        act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

        flat_size = self.n_env * batch_size
        observations = torch.gather(self.observations, 1, obs_indices).reshape(flat_size, self.n_obs)
        next_observations = torch.gather(self.next_observations, 1, obs_indices).reshape(flat_size, self.n_obs)
        actions = torch.gather(self.actions, 1, act_indices).reshape(flat_size, self.n_act)
        rewards = torch.gather(self.rewards, 1, indices).reshape(flat_size)
        dones = torch.gather(self.dones, 1, indices).reshape(flat_size)
        truncations = torch.gather(self.truncations, 1, indices).reshape(flat_size)

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "truncations": truncations,
            "next_observations": next_observations,
        }

    @property
    def size(self):
        return min(self.ptr, self.buffer_size) * self.n_env


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
        tau: float = 0.1,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 1024,
        num_atoms: int = 101,
        v_min: float = -10.0,
        v_max: float = 10.0,
        init_scale: float = 0.01,
        std_min: float = 0.2,
        std_max: float = 0.8,
        weight_decay: float = 0.1,
        use_cdq: bool = True,
        # TD3-specific
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
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
        self.max_iterations = max_iterations

        # Build actor
        self.actor = Actor(
            n_obs=obs_dim,
            n_act=action_dim,
            num_envs=num_envs,
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
            std_min=std_min,
            std_max=std_max,
            device=device,
        )

        # Build critic
        self.qnet = Critic(
            n_obs=obs_dim,
            n_act=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            device=device,
        )
        self.qnet_target = Critic(
            n_obs=obs_dim,
            n_act=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            device=device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Observation normalization
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

        # Cosine LR schedulers
        self.q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.q_optimizer,
            T_max=max_iterations,
            eta_min=torch.tensor(critic_lr, device=device),
        )
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=max_iterations,
            eta_min=torch.tensor(actor_lr, device=device),
        )

        self.update_count = 0

    def normalize_obs(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        """Normalize observations using running statistics."""
        if not isinstance(self.obs_normalizer, nn.Identity):
            return self.obs_normalizer(obs, update=update)
        return obs

    def update_critic(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One critic update step."""
        observations = data["observations"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_observations = data["next_observations"]
        dones = data["dones"].bool()
        truncations = data["truncations"].bool()

        bootstrap = (truncations | ~dones).float()
        discount = torch.full_like(rewards, self.gamma)

        # Target policy smoothing
        clipped_noise = torch.randn_like(actions)
        clipped_noise = clipped_noise.mul(self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        next_state_actions = (self.actor(next_observations) + clipped_noise).clamp(-1.0, 1.0)

        with torch.no_grad():
            qf1_next_target_proj, qf2_next_target_proj = self.qnet_target.projection(
                next_observations, next_state_actions, rewards, bootstrap, discount,
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

        qf1, qf2 = self.qnet(observations, actions)
        qf1_loss = -torch.sum(
            qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
        ).mean()
        qf2_loss = -torch.sum(
            qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
        ).mean()
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optimizer.step()

        return {
            "qf_loss": qf_loss.item(),
            "qf_max": qf1_next_target_value.max().item(),
            "qf_min": qf1_next_target_value.min().item(),
        }

    def update_actor(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One actor update step."""
        observations = data["observations"]

        qf1, qf2 = self.qnet(observations, self.actor(observations))
        qf1_value = self.qnet.get_value(F.softmax(qf1, dim=1))
        qf2_value = self.qnet.get_value(F.softmax(qf2, dim=1))

        if self.use_cdq:
            qf_value = torch.minimum(qf1_value, qf2_value)
        else:
            qf_value = (qf1_value + qf2_value) / 2.0

        actor_loss = -qf_value.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item()}

    @torch.no_grad()
    def soft_update(self):
        """Polyak-average update of target critic network."""
        src_ps = [p.data for p in self.qnet.parameters()]
        tgt_ps = [p.data for p in self.qnet_target.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - self.tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=self.tau)

    def step_schedulers(self):
        """Step learning rate schedulers."""
        self.q_scheduler.step()
        self.actor_scheduler.step()

    def get_state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
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
        self.qnet.load_state_dict(state_dict["qnet"])
        self.qnet_target.load_state_dict(state_dict["qnet_target"])
        if state_dict.get("obs_normalizer") and hasattr(self.obs_normalizer, "load_state_dict"):
            self.obs_normalizer.load_state_dict(state_dict["obs_normalizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.update_count = state_dict.get("update_count", 0)
