"""Typed dataclass configs for all training algorithms.

Replaces ml_collections.ConfigDict factory functions.
Use OmegaConf / Hydra to compose these at runtime.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional, cast


class BaseConfig:
    """Base class for typed configs with a plain-dict export helper."""

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain nested dict."""
        return cast(dict[str, Any], dataclasses.asdict(cast(Any, self)))


# ── Off-policy: SAC ──────────────────────────────────────────────────────────


@dataclass
class SACAlgoParams:
    """SAC entropy-temperature and optimization parameters."""

    # Learning rate for the entropy temperature (alpha).
    alpha_lr: float = 3e-4
    # Initial entropy temperature value.
    alpha_init: float = 0.01
    # Target entropy is -action_dim * ratio; 0 keeps a zero target.
    target_entropy_ratio: float = 0.0
    # Gradient clipping norm; 0 disables clipping.
    max_grad_norm: float = 0.0
    # AMP dtype selection ("auto" resolves per device).
    amp_dtype: str = "auto"
    # Enable torch.compile for the update path.
    use_compile: bool = True


@dataclass
class SACConfig(BaseConfig):
    """Top-level SAC training configuration."""

    # Algorithm registry key.
    algo: str = "sac"
    # Name used in logs and run directories.
    algo_log_name: str = "sac"
    # Optional runtime implementation override.
    runtime_impl: Optional[str] = None
    # Optional runtime resolver override.
    runtime_resolver: Optional[str] = None
    # Global random seed.
    seed: int = 1
    # Number of parallel environments.
    num_envs: int = 4096
    # Learner batch size for one SAC update.
    batch_size: int = 8192
    # Replay buffer length in steps per environment.
    replay_buffer_n: int = 512
    # Gradient updates per environment step.
    updates_per_step: int = 4
    # Collection steps before learning starts.
    learning_starts: int = 1
    # Environment steps between policy updates.
    policy_frequency: int = 4
    # Maximum number of training iterations.
    max_iterations: int = 500
    # Iterations between checkpoint saves.
    save_interval: int = 500
    # Discount factor.
    gamma: float = 0.97
    # Target network soft-update coefficient.
    tau: float = 0.125
    # Actor learning rate.
    actor_lr: float = 3e-4
    # Critic learning rate.
    critic_lr: float = 3e-4
    # Actor hidden layer width.
    actor_hidden_dim: int = 512
    # Critic hidden layer width.
    critic_hidden_dim: int = 768
    # Number of atoms in the distributional critic.
    num_atoms: int = 101
    # Enable observation normalization.
    obs_normalization: bool = False
    # Enable layer normalization in the networks.
    use_layer_norm: bool = True
    # Extra actor overrides forwarded to the actor factory.
    actor: dict[str, Any] = field(default_factory=dict)
    # Entropy-temperature and optimization parameters.
    algo_params: SACAlgoParams = field(default_factory=SACAlgoParams)


# ── Off-policy: FlashSAC ─────────────────────────────────────────────────────


@dataclass
class FlashSACAlgoParams:
    """FlashSAC learner parameters."""

    # Normalize rewards before computing returns.
    normalize_reward: bool = True
    # Clamp of the normalized return target magnitude.
    normalized_g_max: float = 5.0
    # Number of residual blocks in the actor.
    actor_num_blocks: int = 2
    # Number of residual blocks in the critic.
    critic_num_blocks: int = 2
    # Behavior-cloning loss weight on the actor; 0 disables it.
    actor_bc_alpha: float = 0.0
    # Mean of the actor noise zeta schedule.
    actor_noise_zeta_mu: float = 2.0
    # Maximum of the actor noise zeta schedule.
    actor_noise_zeta_max: int = 16
    # Lower bound of the distributional critic value support.
    critic_min_v: float = -5.0
    # Upper bound of the distributional critic value support.
    critic_max_v: float = 5.0
    # Initial entropy temperature value.
    temp_initial_value: float = 0.01
    # Target standard deviation for the temperature adaptation.
    temp_target_sigma: float = 0.15
    # Explicit target entropy; None derives it from the action dimension.
    temp_target_entropy: float | None = None
    # Initial learning rate of the schedule.
    learning_rate_init: float = 3e-4
    # Peak learning rate of the schedule.
    learning_rate_peak: float = 3e-4
    # Final learning rate of the schedule.
    learning_rate_end: float = 1.5e-4
    # Learning-rate warmup steps.
    learning_rate_warmup_steps: int = 0
    # Learning-rate decay steps.
    learning_rate_decay_steps: int = 500000
    # N-step return horizon.
    n_step: int = 1
    # AMP dtype selection ("auto" resolves per device).
    amp_dtype: str = "auto"
    # Enable torch.compile for the update path.
    use_compile: bool = True
    # Compile the full objective functions, not only the networks.
    compile_full_objectives: bool = True
    # Capture the whole training cycle in a CUDA graph.
    use_whole_cycle_cuda_graph: bool = True


@dataclass
class FlashSACConfig(BaseConfig):
    """Top-level FlashSAC training configuration."""

    # Algorithm registry key.
    algo: str = "flashsac"
    # Name used in logs and run directories.
    algo_log_name: str = "flash_sac"
    # Global random seed.
    seed: int = 1
    # Number of parallel environments.
    num_envs: int = 1024
    # Learner batch size for one update.
    batch_size: int = 2048
    # Replay buffer length in steps per environment.
    replay_buffer_n: int = 512
    # Gradient updates per environment step.
    updates_per_step: int = 2
    # Collection steps before learning starts.
    learning_starts: int = 98
    # Environment steps between policy updates.
    policy_frequency: int = 2
    # Maximum number of training iterations.
    max_iterations: int = 5000
    # Iterations between checkpoint saves.
    save_interval: int = 1000
    # Discount factor.
    gamma: float = 0.97
    # Target network soft-update coefficient.
    tau: float = 0.01
    # Actor learning rate.
    actor_lr: float = 3e-4
    # Critic learning rate.
    critic_lr: float = 3e-4
    # Actor hidden layer width.
    actor_hidden_dim: int = 128
    # Critic hidden layer width.
    critic_hidden_dim: int = 256
    # Number of atoms in the distributional critic.
    num_atoms: int = 101
    # Enable observation normalization.
    obs_normalization: bool = False
    # Enable layer normalization in the networks.
    use_layer_norm: bool = False
    # FlashSAC learner parameters.
    algo_params: FlashSACAlgoParams = field(default_factory=FlashSACAlgoParams)


# ── APPO ─────────────────────────────────────────────────────────────────────


@dataclass
class APPOAlgorithmConfig:
    """APPO PPO-style objective, schedule, and target-update parameters."""

    # Learning epochs per collected batch.
    num_learning_epochs: int = 5
    # Mini-batches per learning epoch.
    num_mini_batches: int = 4
    # PPO surrogate clipping range.
    clip_param: float = 0.2
    # Discount factor.
    gamma: float = 0.99
    # GAE lambda.
    lam: float = 0.95
    # Value loss coefficient.
    value_loss_coef: float = 1.0
    # Entropy bonus coefficient.
    entropy_coef: float = 0.01
    # Optimizer learning rate.
    learning_rate: float = 1e-3
    # Gradient clipping norm.
    max_grad_norm: float = 1.0
    # Clip the value loss around the old value estimate.
    use_clipped_value_loss: bool = True
    # Learning-rate schedule ("adaptive" tracks KL).
    schedule: str = "adaptive"
    # Desired KL divergence for the adaptive schedule.
    desired_kl: float = 0.01
    # KL threshold factor for the adaptive schedule; must exceed 1.0.
    adaptive_kl_factor: float = 1.2
    # Learning-rate adjustment factor for the adaptive schedule.
    adaptive_lr_factor: float = 1.1
    # Optimizer name.
    optimizer: str = "adam"
    # Target network soft-update coefficient.
    tau: float = 1.0
    # Updates between target network refreshes.
    target_update_freq: int = 1
    # V-trace importance-weight clip for rho.
    vtrace_clip_rho: float = 1.0
    # V-trace importance-weight clip for c.
    vtrace_clip_c: float = 1.0
    # Enable torch.compile for the update path.
    enable_compile: bool = True


@dataclass
class APPODistributionConfig:
    """Action distribution configuration for APPO."""

    # Distribution class path resolved by the actor factory.
    class_name: str = "rsl_rl.modules.distribution.GaussianDistribution"
    # Initial distribution standard deviation.
    init_std: float = 1.0
    # Standard-deviation parameterization ("scalar" shares one value).
    std_type: str = "scalar"


@dataclass
class APPOActorConfig:
    """APPO actor network configuration."""

    # Actor model class path.
    class_name: str = "rsl_rl.models.MLPModel"
    # Hidden layer widths.
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    # Activation function name.
    activation: str = "elu"
    # Action distribution configuration.
    distribution_cfg: APPODistributionConfig = field(default_factory=APPODistributionConfig)


@dataclass
class APPOCriticConfig:
    """APPO critic network configuration."""

    # Critic model class path.
    class_name: str = "rsl_rl.models.MLPModel"
    # Hidden layer widths.
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    # Activation function name.
    activation: str = "elu"


@dataclass
class APPOConfig(BaseConfig):
    """Top-level APPO training configuration."""

    # Algorithm registry key.
    algo: str = "appo"
    # Name used in logs and run directories.
    algo_log_name: str = "appo"
    # Global random seed.
    seed: int = 1
    # Number of parallel environments.
    num_envs: int = 2048
    # Rollout steps collected per environment per iteration.
    steps_per_env: int = 24
    # Maximum number of training iterations.
    max_iterations: int = 150
    # Iterations between checkpoint saves.
    save_interval: int = 50
    # Observation group assignment per network input.
    obs_groups: dict = field(default_factory=lambda: {"actor": {"policy": 0}})
    # Actor network configuration.
    actor: APPOActorConfig = field(default_factory=APPOActorConfig)
    # Critic network configuration.
    critic: APPOCriticConfig = field(default_factory=APPOCriticConfig)
    # Objective, schedule, and target-update parameters.
    algorithm: APPOAlgorithmConfig = field(default_factory=APPOAlgorithmConfig)


# ── PPO (rsl-rl) ─────────────────────────────────────────────────────────────


@dataclass
class PPODistributionConfig:
    """Action distribution configuration for PPO (rsl-rl)."""

    # Distribution class path resolved by the actor factory.
    class_name: str = "rsl_rl.modules.distribution.GaussianDistribution"
    # Initial distribution standard deviation.
    init_std: float = 1.0
    # Standard-deviation parameterization ("scalar" shares one value).
    std_type: str = "scalar"


@dataclass
class PPOActorConfig:
    """PPO actor network configuration."""

    # Actor model class path.
    class_name: str = "rsl_rl.models.MLPModel"
    # Hidden layer widths.
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    # Activation function name.
    activation: str = "elu"
    # Enable observation normalization in the actor.
    obs_normalization: bool = False
    # Action distribution configuration.
    distribution_cfg: PPODistributionConfig = field(default_factory=PPODistributionConfig)


@dataclass
class PPOCriticConfig:
    """PPO critic network configuration."""

    # Critic model class path.
    class_name: str = "rsl_rl.models.MLPModel"
    # Hidden layer widths.
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    # Activation function name.
    activation: str = "elu"
    # Enable observation normalization in the critic.
    obs_normalization: bool = False


@dataclass
class PPOAlgorithmConfig:
    """PPO objective and schedule parameters (rsl-rl)."""

    # Algorithm class path resolved by rsl-rl.
    class_name: str = "rsl_rl.algorithms:PPO"
    # Value loss coefficient.
    value_loss_coef: float = 1.0
    # Clip the value loss around the old value estimate.
    use_clipped_value_loss: bool = True
    # PPO surrogate clipping range.
    clip_param: float = 0.2
    # Entropy bonus coefficient.
    entropy_coef: float = 0.01
    # Learning epochs per collected batch.
    num_learning_epochs: int = 5
    # Mini-batches per learning epoch.
    num_mini_batches: int = 4
    # Optimizer learning rate.
    learning_rate: float = 1e-3
    # Learning-rate schedule ("adaptive" tracks KL).
    schedule: str = "adaptive"
    # Discount factor.
    gamma: float = 0.99
    # GAE lambda.
    lam: float = 0.95
    # Desired KL divergence for the adaptive schedule.
    desired_kl: float = 0.01
    # Gradient clipping norm.
    max_grad_norm: float = 1.0


@dataclass
class PPOConfig(BaseConfig):
    """Top-level PPO training configuration (direct rsl-rl integration)."""

    # Algorithm registry key.
    algo: str = "ppo"
    # Name used in logs and run directories.
    algo_log_name: str = "rsl_rl_ppo"
    # Global random seed.
    seed: int = 1
    # Number of parallel environments.
    num_envs: int = 4096
    # Rollout steps collected per environment per iteration.
    num_steps_per_env: int = 24
    # Maximum number of training iterations.
    max_iterations: int = 101
    # Iterations between checkpoint saves.
    save_interval: int = 100
    # Observation group assignment per network input.
    obs_groups: dict = field(default_factory=lambda: {"actor": ["policy"]})
    # Experiment name used for run directories.
    experiment_name: str = "test"
    # Optional run name suffix.
    run_name: str = ""
    # Resume training from a checkpoint.
    resume: bool = False
    # Run directory to load from; -1 selects the most recent.
    load_run: str = "-1"
    # Checkpoint iteration to load; -1 selects the latest.
    checkpoint: int = -1
    # Explicit checkpoint path overriding run/checkpoint selection.
    resume_path: Optional[str] = None
    # Actor network configuration.
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)
    # Critic network configuration.
    critic: PPOCriticConfig = field(default_factory=PPOCriticConfig)
    # Objective and schedule parameters.
    algorithm: PPOAlgorithmConfig = field(default_factory=PPOAlgorithmConfig)
