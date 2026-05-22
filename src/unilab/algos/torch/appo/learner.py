"""Asynchronous PPO (APPO) Learner.

Based on IMPACT (Luo et al. 2020): Importance Weighted Asynchronous
Architectures with Clipped Target Networks.

Key differences from standard PPO:
- V-trace importance sampling correction for off-policy data
- Target network with soft update for stable IS ratio computation
- PPO clipping applied over IS-corrected ratios
"""

import copy
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_optimizer
from tensordict import TensorDict


def _sample_tensor_for_metric(tensor: torch.Tensor, max_items: int = 8192) -> torch.Tensor:
    """Return a deterministic bounded sample for scalar metrics."""
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= max_items:
        return flat
    stride = max(flat.numel() // max_items, 1)
    return flat[::stride][:max_items]


def _grad_norm(parameters) -> float:
    """Compute gradient norm without mutating gradients or clipping."""
    norms = [p.grad.detach().norm(2) for p in parameters if getattr(p, "grad", None) is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    """Return parameters without duplicate object ids, preserving first use order."""
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in parameters:
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        unique.append(param)
    return unique


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
        adaptive_kl_factor: float = 1.2,
        adaptive_lr_factor: float = 1.1,
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
        actor_params = _unique_parameters(self.actor.parameters())
        critic_params = _unique_parameters(self.critic.parameters())
        self._combined_params = _unique_parameters([*actor_params, *critic_params])

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
        if adaptive_kl_factor <= 1.0:
            raise ValueError(f"adaptive_kl_factor must be > 1.0, got {adaptive_kl_factor}")
        if adaptive_lr_factor <= 1.0:
            raise ValueError(f"adaptive_lr_factor must be > 1.0, got {adaptive_lr_factor}")
        self.adaptive_kl_factor = adaptive_kl_factor
        self.adaptive_lr_factor = adaptive_lr_factor

        # APPO-specific parameters
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.vtrace_clip_rho = vtrace_clip_rho
        self.vtrace_clip_c = vtrace_clip_c
        self._update_counter = 0
        self.last_update_metrics: dict[str, float] = {}

        # Optimizer
        self.optimizer = resolve_optimizer(optimizer)(  # pyright: ignore[reportCallIssue]
            self._combined_params, lr=learning_rate
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
        self.sync_target_actor_buffers()

    def _update_adaptive_learning_rate(self, kl_mean: float) -> None:
        """Update optimizer LR from KL according to the configured adaptive schedule."""
        if self.desired_kl is None or self.schedule != "adaptive":
            return

        if kl_mean > self.desired_kl * self.adaptive_kl_factor:
            self.learning_rate = max(1e-5, self.learning_rate / self.adaptive_lr_factor)
        elif 0.0 < kl_mean < self.desired_kl / self.adaptive_kl_factor:
            self.learning_rate = min(1e-2, self.learning_rate * self.adaptive_lr_factor)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def sync_target_actor_buffers(self):
        """Copy actor buffers such as observation-normalization stats to target actor."""
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
        critic_base = batch_dict.get("critic", None)  # [T, N, C] or None
        rewards = batch_dict["rewards"]  # [T, N]
        dones = batch_dict["dones"].float()  # [T, N]
        last_obs = batch_dict["last_obs"]  # [N, D]
        last_critic = batch_dict.get("last_critic", None)  # [N, C] or None
        behavior_log_probs = batch_dict["actions_log_prob"]  # [T, N]
        actions = batch_dict["actions"]  # [T, N, A]

        T, N = obs.shape[:2]
        obs_flat = obs.flatten(0, 1)  # [T*N, D]

        # Actor: obs only
        obs_td = TensorDict({"policy": obs_flat}, batch_size=obs_flat.shape[0], device=self.device)
        last_obs_td = TensorDict({"policy": last_obs}, batch_size=N, device=self.device)

        # Critic: explicit critic obs when available, else actor obs
        if critic_base is None:
            critic_base = obs
        if last_critic is None:
            last_critic = last_obs
        critic_obs = critic_base
        critic_last_obs = last_critic
        critic_obs_flat = critic_obs.flatten(0, 1)  # [T*N, D+P]
        critic_obs_td = TensorDict(
            {"policy": critic_obs_flat}, batch_size=critic_obs_flat.shape[0], device=self.device
        )
        critic_last_obs_td = TensorDict(
            {"policy": critic_last_obs}, batch_size=N, device=self.device
        )

        # Update Observation Normalization
        if hasattr(self.actor, "update_normalization"):
            self.actor.update_normalization(obs_td)
            self.actor.update_normalization(last_obs_td)
            self.sync_target_actor_buffers()
        if hasattr(self.critic, "update_normalization"):
            self.critic.update_normalization(critic_obs_td)
            self.critic.update_normalization(critic_last_obs_td)

        # Cache critic_obs_flat for update()
        batch_dict["_critic_obs_flat"] = critic_obs_flat

        with torch.inference_mode():
            # Compute values with current critic
            values_flat = self.critic(critic_obs_td)  # [T*N, 1]
            last_values = self.critic(critic_last_obs_td).squeeze(-1)  # [N]
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
        with torch.inference_mode():
            rhos = torch.exp(target_log_probs - behavior_log_probs)
            rho_sample = _sample_tensor_for_metric(rhos)
            batch_dict["_appo_process_metrics"] = {
                "vtrace/rho_clip_fraction": float(
                    (rhos > float(self.vtrace_clip_rho)).float().mean().item()
                ),
                "vtrace/rho_raw_p99": float(torch.quantile(rho_sample, 0.99).item()),
            }

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
        """Perform the original main APPO update with additional detached metrics."""
        obs_flat = batch_dict["observations"].flatten(0, 1)
        actions_flat = batch_dict["actions"].flatten(0, 1)
        returns_flat = batch_dict["returns"].flatten(0, 1)
        advantages_flat = batch_dict["advantages"].flatten(0, 1)
        behavior_log_probs_flat = batch_dict["actions_log_prob"].flatten(0, 1)
        old_values_flat = batch_dict["values"].flatten(0, 1)
        target_log_probs_flat = batch_dict["target_log_probs"].flatten(0, 1)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        obs_td = batch_dict.get("_obs_td")
        if obs_td is None:
            obs_td = TensorDict(
                {"policy": obs_flat}, batch_size=obs_flat.shape[0], device=self.device
            )

        critic_obs_td = batch_dict.get("_critic_obs_td")
        if critic_obs_td is None:
            critic_obs_flat = batch_dict.get("_critic_obs_flat")
            if critic_obs_flat is None:
                critic_base_flat = batch_dict.get("critic")
                critic_obs_flat = (
                    obs_flat if critic_base_flat is None else critic_base_flat.flatten(0, 1)
                )
            critic_obs_td = TensorDict(
                {"policy": critic_obs_flat},
                batch_size=critic_obs_flat.shape[0],
                device=self.device,
            )

        with torch.inference_mode():
            old_mu_flat = batch_dict["_old_mu"]
            old_sigma_flat = batch_dict["_old_sigma"]

        batch_size = obs_flat.shape[0]
        mini_batch_size = batch_size // self.num_mini_batches

        mean_surrogate_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        mean_clip_fraction = 0.0
        mean_behavior_to_current_kl = 0.0
        mean_target_to_current_kl = 0.0
        mean_global_grad_norm = 0.0
        num_updates = 0

        for epoch in range(self.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_mini_td = obs_td[batch_idx]
                critic_obs_mini_td = critic_obs_td[batch_idx]
                actions_mini = actions_flat[batch_idx]
                target_values_mini = returns_flat[batch_idx]
                advantages_mini = advantages_flat[batch_idx]
                behavior_logp_mini = behavior_log_probs_flat[batch_idx]
                old_values_mini = old_values_flat[batch_idx]
                target_logp_mini = target_log_probs_flat[batch_idx]
                old_mu_mini = old_mu_flat[batch_idx]
                old_sigma_mini = old_sigma_flat[batch_idx]

                _ = self.actor(obs_mini_td, stochastic_output=True)
                current_log_prob = self.actor.get_output_log_prob(actions_mini)
                value = self.critic(critic_obs_mini_td).squeeze(-1)
                entropy = self.actor.output_entropy.mean()
                mu = self.actor.output_mean
                sigma = self.actor.output_std

                with torch.no_grad():
                    clipped_rho = torch.clamp(
                        torch.exp(behavior_logp_mini - target_logp_mini), max=1.0
                    )
                ratio = clipped_rho * torch.exp(current_log_prob - behavior_logp_mini)

                surrogate = -advantages_mini * ratio
                surrogate_clipped = -advantages_mini * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                kl_mean_value: float | None = None
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
                        kl_mean_value = float(kl_mean.item())

                        self._update_adaptive_learning_rate(kl_mean_value)

                        mean_kl += kl_mean_value
                        mean_target_to_current_kl += kl_mean_value

                if self.use_clipped_value_loss:
                    value_clipped = old_values_mini + (value - old_values_mini).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value - target_values_mini).pow(2)
                    value_losses_clipped = (value_clipped - target_values_mini).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (value - target_values_mini).pow(2).mean()

                loss = (
                    surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                global_grad_norm = _grad_norm(self._combined_params)
                nn.utils.clip_grad_norm_(self._combined_params, self.max_grad_norm)
                self.optimizer.step()

                with torch.inference_mode():
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_param).float().mean().item()
                    behavior_to_current_kl = (behavior_logp_mini - current_log_prob).mean().item()
                    if kl_mean_value is None:
                        kl = torch.sum(
                            torch.log(sigma / old_sigma_mini + 1e-5)
                            + (old_sigma_mini.pow(2) + (old_mu_mini - mu).pow(2))
                            / (2.0 * sigma.pow(2))
                            - 0.5,
                            dim=-1,
                        )
                        kl_mean_value = float(torch.mean(kl).item())
                        mean_target_to_current_kl += kl_mean_value

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.item()
                mean_clip_fraction += float(clip_fraction)
                mean_behavior_to_current_kl += float(behavior_to_current_kl)
                mean_global_grad_norm += global_grad_norm
                num_updates += 1

        self._update_counter += 1
        if self._update_counter % self.target_update_freq == 0:
            self.update_target_network()

        num_updates = max(num_updates, 1)
        final_lr = float(self.learning_rate)
        self.learning_rate = final_lr

        metrics = {
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "kl": mean_kl / num_updates if self.schedule == "adaptive" else 0.0,
            "loss/policy_loss": mean_surrogate_loss / num_updates,
            "loss/value_loss": mean_value_loss / num_updates,
            "policy/entropy": mean_entropy / num_updates,
            "ppo/approx_kl": mean_target_to_current_kl / num_updates,
            "ppo/clip_fraction": mean_clip_fraction / num_updates,
            "grad/global_norm": mean_global_grad_norm / num_updates,
            "optim/learning_rate": final_lr,
            "policy_kl/behavior_to_current_kl": mean_behavior_to_current_kl / num_updates,
            "appo/updates_executed": float(num_updates),
        }
        metrics.update(batch_dict.get("_appo_process_metrics", {}))
        self.last_update_metrics = metrics
        return metrics
