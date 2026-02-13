"""Runner-style utilities for MLX PPO.

This module keeps train script entrypoints thin, similar to rsl-rl runner usage.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map

from unilab.algos.mlx_rl import EmpiricalDiscountedVariationNormalization, RolloutBuffer
from .model import MLPActorCritic
from .ppo import PPOConfig, PPOTrainer


class TensorboardScalarWriter:
    """Minimal scalar writer based on tensorboard event files."""

    def __init__(self, log_dir: Path) -> None:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._Event = Event
        self._Summary = Summary
        self._writer = EventFileWriter(str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary = self._Summary(value=[self._Summary.Value(tag=tag, simple_value=float(value))])
        event = self._Event(wall_time=time.time(), step=int(step), summary=summary)
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def get_latest_run(log_dir: Path) -> Path | None:
    """Find latest run directory under a task log root."""
    if not log_dir.exists():
        return None
    runs = sorted([p for p in log_dir.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def get_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the latest model_*.safetensors checkpoint in a run dir."""
    if not run_dir.exists():
        return None
    model_files = [p for p in run_dir.glob("model_*.safetensors") if p.is_file()]
    if not model_files:
        return None
    model_files.sort(key=lambda p: int(p.stem.split("_")[1]))
    return model_files[-1]


class MLXPPOAgent:
    """High-level PPO wrapper to keep train script lightweight."""

    def __init__(self, cfg, obs_dim: int, action_dim: int, learning_rate: float) -> None:
        policy_cfg = cfg.policy
        algo_cfg = cfg.algorithm

        init_noise_std = float(getattr(policy_cfg, "init_noise_std", 1.0))
        init_log_std = float(mx.log(mx.array(max(init_noise_std, 1e-6))).item())
        obs_norm = bool(getattr(cfg, "empirical_normalization", False))
        noise_std_type = str(getattr(policy_cfg, "noise_std_type", "scalar"))
        state_dependent_std = bool(getattr(policy_cfg, "state_dependent_std", False))

        self.model = MLPActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=policy_cfg.actor_hidden_dims,
            critic_hidden_dims=policy_cfg.critic_hidden_dims,
            activation=policy_cfg.activation,
            init_log_std=init_log_std,
            obs_normalization=obs_norm,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
        )

        ppo_cfg = PPOConfig(
            num_learning_epochs=int(algo_cfg.num_learning_epochs),
            num_mini_batches=int(algo_cfg.num_mini_batches),
            clip_param=float(algo_cfg.clip_param),
            gamma=float(algo_cfg.gamma),
            lam=float(algo_cfg.lam),
            value_loss_coef=float(algo_cfg.value_loss_coef),
            entropy_coef=float(algo_cfg.entropy_coef),
            learning_rate=learning_rate,
            use_clipped_value_loss=bool(algo_cfg.use_clipped_value_loss),
            max_grad_norm=float(getattr(algo_cfg, "max_grad_norm", 1.0)),
            schedule=str(getattr(algo_cfg, "schedule", "fixed")),
            desired_kl=float(getattr(algo_cfg, "desired_kl", 0.01)),
            normalize_advantage_per_mini_batch=bool(getattr(algo_cfg, "normalize_advantage_per_mini_batch", False)),
        )
        self.trainer = PPOTrainer(self.model, ppo_cfg)

        use_reward_norm = bool(getattr(algo_cfg, "reward_normalization", False))
        self.reward_normalizer = (
            EmpiricalDiscountedVariationNormalization(gamma=ppo_cfg.gamma) if use_reward_norm else None
        )

    @property
    def learning_rate(self) -> float:
        return self.trainer.learning_rate

    def update_normalization(self, obs: mx.array) -> None:
        self.model.update_normalization(obs)

    def act(self, obs: mx.array):
        return self.model.act(obs)

    def policy_mean(self, obs: mx.array) -> mx.array:
        return self.model.policy(obs)

    def normalize_rewards(self, rewards: mx.array) -> mx.array:
        if self.reward_normalizer is not None:
            return mx.squeeze(self.reward_normalizer(rewards), axis=-1)
        return rewards

    def current_action_std(self, action_shape: tuple[int, ...]) -> mx.array:
        return self.model.current_action_std(action_shape)

    def mean_noise_std(self) -> float:
        std = self.model.current_action_std((1, self.model.action_dim))
        mx.eval(std)
        return float(mx.mean(std).item())

    def update(self, buffer: RolloutBuffer, last_obs: mx.array):
        last_values = self.model.value(last_obs)
        buffer.compute_returns_and_advantages(last_values)
        return self.trainer.update(buffer)

    def load_weights(self, path: Path) -> None:
        self.model.load_weights(str(path), strict=True)

    def save_checkpoint(self, model_path: Path, trainer_state_path: Path, iteration: int) -> None:
        self.model.save_weights(str(model_path))
        payload = {
            "iteration": int(iteration),
            "learning_rate": float(self.trainer.learning_rate),
            "optimizer_state": tree_map(lambda x: x.tolist(), self.trainer.optimizer.state),
        }
        with trainer_state_path.open("wb") as f:
            pickle.dump(payload, f)

    def load_trainer_state(self, trainer_state_path: Path) -> int:
        with trainer_state_path.open("rb") as f:
            payload = pickle.load(f)
        self.trainer.learning_rate = float(payload.get("learning_rate", self.trainer.learning_rate))
        self.trainer.optimizer.learning_rate = mx.array(self.trainer.learning_rate, dtype=mx.float32)
        self.trainer.optimizer.state = tree_map(lambda x: mx.array(x), payload["optimizer_state"])
        return int(payload.get("iteration", -1))
