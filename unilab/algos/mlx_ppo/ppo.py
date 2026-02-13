"""PPO trainer implemented with MLX."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from unilab.algos.mlx_rl import RolloutBuffer, diag_gaussian_entropy, diag_gaussian_log_prob
from .model import MLPActorCritic


@dataclass
class PPOConfig:
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    learning_rate: float = 3e-4
    use_clipped_value_loss: bool = True
    max_grad_norm: float = 1.0
    log_ratio_clip: float = 20.0
    schedule: str = "fixed"
    desired_kl: float = 0.01
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2
    normalize_advantage_per_mini_batch: bool = False


class PPOTrainer:
    """PPO update logic for `MLPActorCritic` and `RolloutBuffer`."""

    def __init__(self, model: MLPActorCritic, cfg: PPOConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.learning_rate = float(cfg.learning_rate)
        self.optimizer = optim.Adam(learning_rate=self.learning_rate)
        self.loss_and_grad = nn.value_and_grad(model, self._loss_fn)

    @staticmethod
    def _all_finite(tree) -> bool:
        leaves = [leaf for _, leaf in tree_flatten(tree)]
        if not leaves:
            return True
        checks = [mx.all(mx.isfinite(leaf)) for leaf in leaves]
        mx.eval(*checks)
        return all(bool(c.item()) for c in checks)

    def _clip_grads(self, grads):
        """Global gradient clipping similar to rsl-rl max_grad_norm."""
        if self.cfg.max_grad_norm <= 0.0:
            return grads
        leaves = [leaf for _, leaf in tree_flatten(grads)]
        if not leaves:
            return grads
        sq_norm = mx.array(0.0, dtype=mx.float32)
        for leaf in leaves:
            sq_norm = sq_norm + mx.sum(leaf * leaf)
        global_norm = mx.sqrt(sq_norm + 1e-12)
        clip_coef = mx.minimum(1.0, self.cfg.max_grad_norm / (global_norm + 1e-6))
        return tree_map(lambda g: g * clip_coef, grads)

    def _loss_fn(self, model: MLPActorCritic, batch: Dict[str, mx.array]) -> mx.array:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_values = batch["old_values"]

        if self.cfg.normalize_advantage_per_mini_batch:
            advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

        mean, sigma, log_std = model.distribution_params(obs)
        values = model.value(obs)
        log_probs = diag_gaussian_log_prob(actions, mean, log_std)
        entropy = mx.mean(diag_gaussian_entropy(log_std))

        log_ratio = mx.clip(log_probs - old_log_probs, -self.cfg.log_ratio_clip, self.cfg.log_ratio_clip)
        ratio = mx.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = mx.clip(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages
        policy_loss = -mx.mean(mx.minimum(surr1, surr2))

        if self.cfg.use_clipped_value_loss:
            value_pred_clipped = old_values + mx.clip(values - old_values, -self.cfg.clip_param, self.cfg.clip_param)
            value_losses = (values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = 0.5 * mx.mean(mx.maximum(value_losses, value_losses_clipped))
        else:
            value_loss = 0.5 * mx.mean((returns - values) ** 2)

        return policy_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy

    def _metrics(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_values = batch["old_values"]
        old_mu = batch["old_mu"]
        old_sigma = batch["old_sigma"]

        if self.cfg.normalize_advantage_per_mini_batch:
            advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

        mean, sigma, log_std = self.model.distribution_params(obs)
        values = self.model.value(obs)
        log_probs = diag_gaussian_log_prob(actions, mean, log_std)
        entropy = mx.mean(diag_gaussian_entropy(log_std))
        sigma = mx.maximum(sigma, 1e-5)

        log_ratio = mx.clip(log_probs - old_log_probs, -self.cfg.log_ratio_clip, self.cfg.log_ratio_clip)
        ratio = mx.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = mx.clip(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages
        policy_loss = -mx.mean(mx.minimum(surr1, surr2))

        if self.cfg.use_clipped_value_loss:
            value_pred_clipped = old_values + mx.clip(values - old_values, -self.cfg.clip_param, self.cfg.clip_param)
            value_losses = (values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = 0.5 * mx.mean(mx.maximum(value_losses, value_losses_clipped))
        else:
            value_loss = 0.5 * mx.mean((returns - values) ** 2)

        # Match rsl-rl style analytic KL for adaptive LR.
        kl = mx.sum(
            mx.log(sigma / (old_sigma + 1e-5) + 1e-5)
            + (old_sigma**2 + (old_mu - mean) ** 2) / (2.0 * sigma**2)
            - 0.5,
            axis=-1,
        )
        kl_mean = mx.mean(kl)
        mx.eval(policy_loss, value_loss, entropy, kl_mean)
        return {
            "surrogate": float(policy_loss.item()),
            "value": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(kl_mean.item()),
        }

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        agg = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        updates = 0
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grads = 0
        rolled_back_updates = 0
        skipped_nonfinite_metrics = 0
        for batch in buffer.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            # Keep a copy for rollback if this update corrupts parameters.
            param_backup = tree_map(lambda x: mx.array(x), self.model.parameters())
            optim_state_backup = tree_map(lambda x: mx.array(x), self.optimizer.state)
            loss, grads = self.loss_and_grad(self.model, batch)
            if not mx.all(mx.isfinite(loss)).item():
                skipped_nonfinite_loss += 1
                continue
            if not self._all_finite(grads):
                skipped_nonfinite_grads += 1
                continue
            grads = self._clip_grads(grads)
            if not self._all_finite(grads):
                skipped_nonfinite_grads += 1
                continue
            self.optimizer.update(self.model, grads)
            mx.eval(loss, self.model.parameters(), self.optimizer.state)
            if not self._all_finite(self.model.parameters()):
                # Roll back this step if parameters become non-finite.
                self.model.update(param_backup)
                self.optimizer.state = optim_state_backup
                mx.eval(self.model.parameters())
                rolled_back_updates += 1
                continue

            metrics = self._metrics(batch)
            if not all(math.isfinite(v) for v in metrics.values()):
                skipped_nonfinite_metrics += 1
                continue

            if self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
                kl = metrics["approx_kl"]
                if kl > self.cfg.desired_kl * 2.0:
                    self.learning_rate = max(self.cfg.min_learning_rate, self.learning_rate / 1.5)
                elif 0.0 < kl < self.cfg.desired_kl / 2.0:
                    self.learning_rate = min(self.cfg.max_learning_rate, self.learning_rate * 1.5)
                self.optimizer.learning_rate = mx.array(self.learning_rate, dtype=mx.float32)

            for key in agg:
                agg[key] += metrics[key]
            updates += 1

        if updates == 0:
            return {
                **agg,
                "learning_rate": self.learning_rate,
                "updates_applied": 0.0,
                "skipped_nonfinite_loss": float(skipped_nonfinite_loss),
                "skipped_nonfinite_grads": float(skipped_nonfinite_grads),
                "rolled_back_updates": float(rolled_back_updates),
                "skipped_nonfinite_metrics": float(skipped_nonfinite_metrics),
            }
        out = {key: value / updates for key, value in agg.items()}
        out["learning_rate"] = self.learning_rate
        out["updates_applied"] = float(updates)
        out["skipped_nonfinite_loss"] = float(skipped_nonfinite_loss)
        out["skipped_nonfinite_grads"] = float(skipped_nonfinite_grads)
        out["rolled_back_updates"] = float(rolled_back_updates)
        out["skipped_nonfinite_metrics"] = float(skipped_nonfinite_metrics)
        return out
