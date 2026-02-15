"""FastSAC Learner — Soft Actor-Critic with automatic entropy tuning.

Key features:
- Twin Q-networks (take min for target Q, same as TD3)
- Stochastic policy (tanh-squashed Gaussian)
- Automatic entropy coefficient (alpha) learning
- Soft target network updates
- Running reward normalization to prevent Q-value explosion
- Works with rsl_rl MLPModel for actor/critic
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_optimizer


class RunningMeanStd:
    """Running mean/std for reward normalization (Welford's algorithm)."""

    def __init__(self, device="cpu"):
        self.mean = torch.tensor(0.0, device=device)
        self.var = torch.tensor(1.0, device=device)
        self.count = 0

    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var() if x.numel() > 1 else torch.tensor(0.0, device=x.device)
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        if total_count == 0:
            return

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


class FastSACLearner:
    """SAC learner that trains on GPU from replay buffer samples."""

    def __init__(
        self,
        actor: MLPModel,
        critic1: MLPModel,
        critic2: MLPModel,
        # SAC hyper-parameters
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        target_entropy: float | None = None,
        num_actions: int = 12,
        max_grad_norm: float = 1.0,
        normalize_rewards: bool = True,
        optimizer: str = "adam",
        device: str = "cpu",
        **kwargs,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.learning_rate = actor_lr  # exposed for logging
        self.normalize_rewards = normalize_rewards

        # Running reward stats for normalization
        self.reward_rms = RunningMeanStd(device=device) if normalize_rewards else None

        # Networks
        self.actor = actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)

        # Target critics only (no target actor in SAC)
        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        for net in [self.target_critic1, self.target_critic2]:
            net.eval()
            for p in net.parameters():
                p.requires_grad = False

        # Automatic entropy tuning
        if target_entropy is None:
            self.target_entropy = -float(num_actions)
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(
            [float(torch.log(torch.tensor(init_alpha)))],
            dtype=torch.float32, device=device, requires_grad=True,
        )

        # Optimisers
        opt_cls = resolve_optimizer(optimizer)
        self.actor_optimizer = opt_cls(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = opt_cls(
            chain(self.critic1.parameters(), self.critic2.parameters()), lr=critic_lr,
        )
        self.alpha_optimizer = opt_cls([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def train_mode(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def get_weights(self):
        return {"actor_state_dict": self.actor.state_dict()}

    def update(self, obs, actions, rewards, next_obs, dones):
        """One SAC gradient step.

        All inputs: tensors on self.device.
        obs, next_obs: [B, D]  |  actions: [B, A]  |  rewards, dones: [B]
        """
        # Reward normalization to prevent Q-value explosion
        if self.reward_rms is not None:
            self.reward_rms.update(rewards)
            rewards = self.reward_rms.normalize(rewards)

        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            obs_td_next = TensorDict(
                {"policy": next_obs}, batch_size=next_obs.shape[0], device=self.device,
            )
            # Sample next actions from current policy
            next_actions = self.actor(obs_td_next, stochastic_output=True)
            next_log_prob = self.actor.get_output_log_prob(next_actions)  # [B]

            # Target Q
            q_input_next = TensorDict(
                {"policy": torch.cat([next_obs, next_actions], dim=-1)},
                batch_size=next_obs.shape[0],
                device=self.device,
            )
            target_q1 = self.target_critic1(q_input_next).squeeze(-1)
            target_q2 = self.target_critic2(q_input_next).squeeze(-1)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_value = rewards + (1.0 - dones) * self.gamma * target_q

        # Current Q
        q_input = TensorDict(
            {"policy": torch.cat([obs, actions], dim=-1)},
            batch_size=obs.shape[0],
            device=self.device,
        )
        q1 = self.critic1(q_input).squeeze(-1)
        q2 = self.critic2(q_input).squeeze(-1)

        critic_loss = nn.functional.mse_loss(q1, target_value) + nn.functional.mse_loss(q2, target_value)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            chain(self.critic1.parameters(), self.critic2.parameters()),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()

        # --- Actor update ---
        obs_td = TensorDict(
            {"policy": obs}, batch_size=obs.shape[0], device=self.device,
        )
        new_actions = self.actor(obs_td, stochastic_output=True)
        new_log_prob = self.actor.get_output_log_prob(new_actions)

        q_input_actor = TensorDict(
            {"policy": torch.cat([obs, new_actions], dim=-1)},
            batch_size=obs.shape[0],
            device=self.device,
        )
        q1_pi = self.critic1(q_input_actor).squeeze(-1)
        q2_pi = self.critic2(q_input_actor).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * new_log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft update targets ---
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "entropy": -new_log_prob.mean().item(),
        }

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)
        for tb, sb in zip(target.buffers(), source.buffers()):
            tb.data.copy_(sb.data)
