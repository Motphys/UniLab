import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_optimizer
from itertools import chain


class BOPRLLearner:
    """Batch On-Policy RL Learner.

    PPO update logic decoupled from rollout collection.

    Key features matching rsl_rl PPO:
    - Observation normalization updated centrally, synced to workers via state_dict
    - Time-out (truncation) bootstrap correction
    - Adaptive learning rate via KL-divergence target
    - Proper train/eval mode management
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
        **kwargs,
    ):
        self.device = device
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

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

        # Optimizer
        self.optimizer = resolve_optimizer(optimizer)(
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

    def get_weights(self):
        """Return actor state dict for syncing to workers.

        Includes EmpiricalNormalization buffers (_mean, _var, _std, count)
        which are registered as buffers in state_dict, so workers get
        updated normalization statistics automatically.
        """
        return {
            "actor_state_dict": self.actor.state_dict(),
        }

    def process_batch(self, batch_dict):
        """Compute Values and GAE on GPU with proper bootstrap.

        Handles:
        - Observation normalization update (training mode)
        - Value computation with current critic
        - Time-out bootstrap correction (truncated != terminated, rsl_rl style)
        - GAE advantage estimation
        """
        obs = batch_dict["observations"]  # [T, N, D]
        rewards = batch_dict["rewards"]  # [T, N]
        dones = batch_dict["dones"].float()  # [T, N]
        truncated = batch_dict["truncated"].float()  # [T, N]
        last_obs = batch_dict["last_obs"]  # [N, D]

        T, N = obs.shape[:2]
        obs_flat = obs.flatten(0, 1)  # [T*N, D]

        # Wrap in TensorDict
        obs_td = TensorDict({"policy": obs_flat}, batch_size=obs_flat.shape[0], device=self.device)
        last_obs_td = TensorDict({"policy": last_obs}, batch_size=N, device=self.device)

        # Update Observation Normalization (matches rsl_rl: update every step with next_obs)
        if hasattr(self.actor, "update_normalization"):
            self.actor.update_normalization(obs_td)
            self.actor.update_normalization(last_obs_td)
        if hasattr(self.critic, "update_normalization"):
            self.critic.update_normalization(obs_td)
            self.critic.update_normalization(last_obs_td)

        with torch.inference_mode():
            values_flat = self.critic(obs_td)  # [T*N, 1]
            last_values = self.critic(last_obs_td).squeeze(-1)  # [N]
        values = values_flat.view(T, N, -1).squeeze(-1)  # [T, N]

        # Time-out bootstrap correction (matching rsl_rl process_env_step):
        #   rewards += gamma * V(s) * time_outs
        # This prevents truncation from being treated as true terminal.
        rewards = rewards.clone()
        rewards += self.gamma * values.detach() * truncated

        # Compute GAE
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values

        batch_dict["values"] = values
        batch_dict["advantages"] = advantages
        batch_dict["returns"] = returns
        batch_dict["_obs_td"] = obs_td  # Cache for update() to avoid rebuilding

        return batch_dict

    def update(self, batch_dict):
        """Perform PPO update with per-mini-batch adaptive LR (matching rsl_rl exactly).

        Uses analytical Gaussian KL divergence between old and new policy outputs
        (mu, sigma) for stable LR adaptation, matching rsl_rl's PPO implementation.

        Returns:
            dict with loss metrics
        """
        # Flatten [T, N, ...] -> [T*N, ...]
        obs_flat = batch_dict["observations"].flatten(0, 1)
        actions_flat = batch_dict["actions"].flatten(0, 1)
        returns_flat = batch_dict["returns"].flatten(0, 1)
        advantages_flat = batch_dict["advantages"].flatten(0, 1)
        old_log_probs_flat = batch_dict["actions_log_prob"].flatten(0, 1)
        old_values_flat = batch_dict["values"].flatten(0, 1)
        old_mu_flat = batch_dict["mu"].flatten(0, 1)  # [T*N, A]
        old_sigma_flat = batch_dict["sigma"].flatten(0, 1)  # [T*N, A]

        # Normalize advantages globally (before mini-batch split, matching rsl_rl)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # Reuse cached TensorDict from process_batch if available
        obs_td = batch_dict.get("_obs_td")
        if obs_td is None:
            obs_td = TensorDict({"policy": obs_flat}, batch_size=obs_flat.shape[0], device=self.device)

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
                old_log_probs_mini = old_log_probs_flat[batch_idx]
                old_values_mini = old_values_flat[batch_idx]
                old_mu_mini = old_mu_flat[batch_idx]
                old_sigma_mini = old_sigma_flat[batch_idx]

                # Forward pass
                _ = self.actor(obs_mini_td, stochastic_output=True)
                actions_log_prob = self.actor.get_output_log_prob(actions_mini)
                value = self.critic(obs_mini_td).squeeze(-1)
                entropy = self.actor.output_entropy.mean()

                # Current policy mu/sigma
                mu = self.actor.output_mean
                sigma = self.actor.output_std

                # Analytical Gaussian KL divergence (matching rsl_rl exactly):
                # D_KL(N_old || N_new) = sum_i [ln(σ_new/σ_old) + (σ_old² + (μ_old - μ_new)²)/(2σ_new²) - 0.5]
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma / old_sigma_mini + 1e-5)
                            + (old_sigma_mini.pow(2) + (old_mu_mini - mu).pow(2)) / (2.0 * sigma.pow(2))
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

                # PPO Surrogate Loss
                ratio = torch.exp(actions_log_prob - old_log_probs_mini)
                surrogate = -advantages_mini * ratio
                surrogate_clipped = -advantages_mini * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value Loss with clipping
                if self.use_clipped_value_loss:
                    value_clipped = old_values_mini + (value - old_values_mini).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value - target_values_mini).pow(2)
                    value_losses_clipped = (value_clipped - target_values_mini).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (value - target_values_mini).pow(2).mean()

                # Total loss
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Gradient step
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.item()
                num_updates += 1

        num_updates = max(num_updates, 1)
        return {
            "surrogate": mean_surrogate_loss / num_updates,
            "value_function": mean_value_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "kl": mean_kl / num_updates if self.schedule == "adaptive" else 0.0,
        }
