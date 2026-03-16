"""Asynchronous PPO (APPO) Learner.

Based on IMPACT (Luo et al. 2020): Importance Weighted Asynchronous
Architectures with Clipped Target Networks.

Key differences from standard PPO:
- V-trace importance sampling correction for off-policy data
- Target network with soft update for stable IS ratio computation
- PPO clipping applied over IS-corrected ratios
"""

import copy
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_optimizer
from tensordict import TensorDict


def vtrace_advantages(
    behavior_log_probs,  # [T, N]  log π_b(a|s) from worker
    target_log_probs,  # [T, N]  log π_target(a|s) from target network
    rewards,  # [T, N]
    values,  # [T, N]
    bootstrap_values,  # [N]     V(s_{T})
    dones,  # [T, N]  float
    gamma=0.99,
    clip_rho=1.0,
    clip_c=1.0,
):
    """Compute V-trace targets and advantages.

    V-trace (Espeholt et al., 2018) corrects for the off-policy nature
    of asynchronous data collection by using importance sampling ratios
    clipped at ρ̄ (rho_bar) and c̄ (c_bar).

    Returns:
        vs:         V-trace value targets  [T, N]
        advantages: Policy gradient advantages  [T, N]
    """
    T, N = rewards.shape
    device = values.device

    with torch.no_grad():
        # IS ratios: ρ_t = π_target(a_t|s_t) / π_behavior(a_t|s_t)
        log_rhos = target_log_probs - behavior_log_probs
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.clamp(rhos, max=clip_rho)
        cs = torch.clamp(rhos, max=clip_c)

        non_terminal = 1.0 - dones

        # Vectorized next_values: shift values by 1, fill last step with bootstrap
        next_values = torch.cat([values[1:], bootstrap_values.unsqueeze(0)], dim=0)

        # Temporal difference errors
        deltas = clipped_rhos * (rewards + gamma * next_values * non_terminal - values)

        # Backward accumulation of V-trace corrections — run on CPU numpy to avoid
        # T sequential GPU kernel launches (one-time transfer cost is cheaper).
        deltas_np = deltas.cpu().numpy()
        non_terminal_np = non_terminal.cpu().numpy()
        cs_np = cs.cpu().numpy()
        values_np = values.cpu().numpy()

        vs_np = np.empty_like(values_np)
        vs_minus_v = np.zeros(N, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            vs_minus_v = deltas_np[t] + gamma * non_terminal_np[t] * cs_np[t] * vs_minus_v
            vs_np[t] = values_np[t] + vs_minus_v

        vs = torch.from_numpy(vs_np).to(device)

        # Vectorized policy gradient advantages
        next_vs = torch.cat([vs[1:], bootstrap_values.unsqueeze(0)], dim=0)
        advantages = clipped_rhos * (rewards + gamma * next_vs * non_terminal - values)

    return vs, advantages


class APPOLearner:
    """Asynchronous PPO Learner.

    PPO update with V-trace off-policy correction and target network,
    decoupled from rollout collection.

    Key features:
    - V-trace importance sampling for off-policy advantage estimation
    - Target network with soft update (tau) for stable IS computation
    - Observation normalization updated centrally, synced to workers
    - Time-out (truncation) bootstrap correction
    - Adaptive learning rate via KL-divergence target
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        optimizer: str = "adam",
        # APPO-specific parameters
        tau: float = 1.0,
        target_update_freq: int = 1,
        vtrace_clip_rho: float = 1.0,
        vtrace_clip_c: float = 1.0,
        **kwargs,
    ):
        self.device = device
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Target actor for V-trace IS computation
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()
        for p in self.target_actor.parameters():
            p.requires_grad = False

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # APPO-specific parameters
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.vtrace_clip_rho = vtrace_clip_rho
        self.vtrace_clip_c = vtrace_clip_c
        self._update_counter = 0

        # Optimizer
        self.optimizer = resolve_optimizer(optimizer)(  # pyright: ignore[reportCallIssue]
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )

    def train_mode(self):
        """Set actor/critic to training mode (enables EmpiricalNormalization.update)."""
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        """Set actor/critic to eval mode."""
        self.actor.eval()
        self.critic.eval()

    def update_target_network(self):
        """Soft update target actor: target = tau * current + (1 - tau) * target."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # Also copy buffers (e.g. normalization stats)
        for target_buf, buf in zip(self.target_actor.buffers(), self.actor.buffers()):
            target_buf.data.copy_(buf.data)

    def get_weights(self):
        """Return actor state dict for syncing to workers.

        Workers use the behavior policy (which may be stale).
        Includes EmpiricalNormalization buffers.
        """
        return self.actor.state_dict()

    def get_state_dict(self):
        """Return full learner state for checkpointing."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def process_batch(self, batch_dict):
        """Compute V-trace targets on GPU.

        Uses target network log-probs and behavior log-probs to compute
        importance-sampling-corrected value targets and advantages.
        """
        obs = batch_dict["observations"]  # [T, N, D]
        rewards = batch_dict["rewards"]  # [T, N]
        dones = batch_dict["dones"].float()  # [T, N]
        truncated = batch_dict["truncated"].float()  # [T, N]
        last_obs = batch_dict["last_obs"]  # [N, D]
        behavior_log_probs = batch_dict["actions_log_prob"]  # [T, N]
        actions = batch_dict["actions"]  # [T, N, A]

        T, N = obs.shape[:2]
        obs_flat = obs.flatten(0, 1)  # [T*N, D]

        # Wrap in TensorDict
        obs_td = TensorDict({"policy": obs_flat}, batch_size=obs_flat.shape[0], device=self.device)
        last_obs_td = TensorDict({"policy": last_obs}, batch_size=N, device=self.device)

        # Update Observation Normalization
        if hasattr(self.actor, "update_normalization"):
            self.actor.update_normalization(obs_td)
            self.actor.update_normalization(last_obs_td)
        if hasattr(self.critic, "update_normalization"):
            self.critic.update_normalization(obs_td)
            self.critic.update_normalization(last_obs_td)

        with torch.inference_mode():
            # Compute values with current critic
            values_flat = self.critic(obs_td)  # [T*N, 1]
            last_values = self.critic(last_obs_td).squeeze(-1)  # [N]
        values = values_flat.view(T, N, -1).squeeze(-1)  # [T, N]

        # Compute target policy log-probs for V-trace IS ratios.
        # Also cache mu/sigma here so update() doesn't need a second forward pass.
        actions_flat = actions.flatten(0, 1)  # [T*N, A]
        with torch.inference_mode():
            self.target_actor(obs_td, stochastic_output=True)
            target_log_probs_flat = self.target_actor.get_output_log_prob(actions_flat)
            batch_dict["_old_mu"] = self.target_actor.output_mean.clone()
            batch_dict["_old_sigma"] = self.target_actor.output_std.clone()
        target_log_probs = target_log_probs_flat.view(T, N)

        # Time-out bootstrap correction (matching rsl_rl):
        #   rewards += gamma * V(s) * time_outs
        rewards += self.gamma * values.detach() * truncated

        # V-trace targets and advantages
        vs, advantages = vtrace_advantages(
            behavior_log_probs=behavior_log_probs,
            target_log_probs=target_log_probs,
            rewards=rewards,
            values=values,
            bootstrap_values=last_values,
            dones=dones,
            gamma=self.gamma,
            clip_rho=self.vtrace_clip_rho,
            clip_c=self.vtrace_clip_c,
        )

        batch_dict["values"] = values
        batch_dict["advantages"] = advantages
        batch_dict["returns"] = vs  # V-trace targets as returns
        batch_dict["target_log_probs"] = target_log_probs
        batch_dict["_obs_td"] = obs_td  # Cache for update()

        return batch_dict

    def update(self, batch_dict):
        """Perform APPO update with IS-corrected PPO clipping.

        Key difference from standard PPO:
        - Uses IS ratio: clip(π_b/π_target, 0, 2) * (π_current/π_b)
        - Advantages are already V-trace corrected

        Returns:
            dict with loss metrics
        """
        # Flatten [T, N, ...] -> [T*N, ...]
        obs_flat = batch_dict["observations"].flatten(0, 1)
        actions_flat = batch_dict["actions"].flatten(0, 1)
        returns_flat = batch_dict["returns"].flatten(0, 1)
        advantages_flat = batch_dict["advantages"].flatten(0, 1)
        behavior_log_probs_flat = batch_dict["actions_log_prob"].flatten(0, 1)
        old_values_flat = batch_dict["values"].flatten(0, 1)
        target_log_probs_flat = batch_dict["target_log_probs"].flatten(0, 1)
        # Normalize advantages globally
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        # Reuse cached TensorDict from process_batch if available
        obs_td = batch_dict.get("_obs_td")
        if obs_td is None:
            obs_td = TensorDict(
                {"policy": obs_flat}, batch_size=obs_flat.shape[0], device=self.device
            )

        # Use target policy mu/sigma cached by process_batch() — no second forward pass.
        with torch.inference_mode():
            old_mu_flat = batch_dict["_old_mu"]
            old_sigma_flat = batch_dict["_old_sigma"]

        batch_size = obs_flat.shape[0]
        mini_batch_size = batch_size // self.num_mini_batches

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        num_updates = 0

        for epoch in range(self.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_mini_td = obs_td[batch_idx]
                actions_mini = actions_flat[batch_idx]
                target_values_mini = returns_flat[batch_idx]
                advantages_mini = advantages_flat[batch_idx]
                behavior_logp_mini = behavior_log_probs_flat[batch_idx]
                old_values_mini = old_values_flat[batch_idx]
                target_logp_mini = target_log_probs_flat[batch_idx]
                old_mu_mini = old_mu_flat[batch_idx]
                old_sigma_mini = old_sigma_flat[batch_idx]

                # Forward pass
                _ = self.actor(obs_mini_td, stochastic_output=True)
                current_log_prob = self.actor.get_output_log_prob(actions_mini)
                value = self.critic(obs_mini_td).squeeze(-1)
                entropy = self.actor.output_entropy.mean()

                # Current policy mu/sigma for KL
                mu = self.actor.output_mean
                sigma = self.actor.output_std

                # V-trace style PPO ratio: min(π_current/π_target, π_current/π_b)
                # = clamp(π_b/π_target, max=1) * π_current/π_b
                # Fresh data (π_b≈π_target) → standard PPO ratio.
                # Stale data → single-sided clip downweights the update without
                # amplifying it (unlike IMPACT's [0,2] range).
                with torch.no_grad():
                    clipped_rho = torch.clamp(
                        torch.exp(behavior_logp_mini - target_logp_mini), max=1.0
                    )
                ratio = clipped_rho * torch.exp(current_log_prob - behavior_logp_mini)

                # PPO Surrogate Loss
                surrogate = -advantages_mini * ratio
                surrogate_clipped = -advantages_mini * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Adaptive LR via analytical KL
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma / old_sigma_mini + 1e-5)
                            + (old_sigma_mini.pow(2) + (old_mu_mini - mu).pow(2))
                            / (2.0 * sigma.pow(2))
                            - 0.5,
                            dim=-1,
                        )
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                        mean_kl += kl_mean.item()

                # Value Loss with clipping
                if self.use_clipped_value_loss:
                    value_clipped = old_values_mini + (value - old_values_mini).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value - target_values_mini).pow(2)
                    value_losses_clipped = (value_clipped - target_values_mini).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (value - target_values_mini).pow(2).mean()

                # Total loss
                loss = (
                    surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                )

                # Gradient step
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm
                )
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.item()
                num_updates += 1

        # Target network update
        self._update_counter += 1
        if self._update_counter % self.target_update_freq == 0:
            self.update_target_network()

        num_updates = max(num_updates, 1)
        return {
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "kl": mean_kl / num_updates if self.schedule == "adaptive" else 0.0,
        }
