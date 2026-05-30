"""Typed dataclass configs for all training algorithms.

Replaces ml_collections.ConfigDict factory functions.
Use OmegaConf / Hydra to compose these at runtime.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional, cast


class BaseConfig:
    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], dataclasses.asdict(cast(Any, self)))


# ── Off-policy: SAC ──────────────────────────────────────────────────────────


@dataclass
class SACAlgoParams:
    alpha_lr: float = 3e-4
    alpha_init: float = 0.01
    target_entropy_ratio: float = 0.0
    max_grad_norm: float = 0.0
    amp_dtype: str = "auto"
    use_compile: bool = True


@dataclass
class SACConfig(BaseConfig):
    algo: str = "sac"
    algo_log_name: str = "fast_sac"
    runtime_impl: Optional[str] = None
    runtime_resolver: Optional[str] = None
    seed: int = 1
    num_envs: int = 4096
    batch_size: int = 8192
    replay_buffer_n: int = 512
    updates_per_step: int = 4
    learning_starts: int = 1
    policy_frequency: int = 4
    env_steps_per_sync: int = 1
    max_iterations: int = 500
    save_interval: int = 500
    gamma: float = 0.97
    tau: float = 0.125
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    num_atoms: int = 101
    obs_normalization: bool = True
    use_layer_norm: bool = True
    use_symmetry: bool = False
    actor: dict[str, Any] = field(default_factory=dict)
    algo_params: SACAlgoParams = field(default_factory=SACAlgoParams)


# ── Off-policy: TD3 ──────────────────────────────────────────────────────────


@dataclass
class TD3AlgoParams:
    weight_decay: float = 0.1
    v_min: float = -10.0
    v_max: float = 10.0
    init_scale: float = 0.01
    log_std_min: float = -0.9
    log_std_max: float = 0.0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    use_cdq: bool = True


@dataclass
class TD3Config(BaseConfig):
    algo: str = "td3"
    algo_log_name: str = "fast_td3"
    seed: int = 1
    num_envs: int = 4096
    batch_size: int = 8192
    replay_buffer_n: int = 1000
    updates_per_step: int = 4
    learning_starts: int = 1
    policy_frequency: int = 2
    env_steps_per_sync: int = 1
    max_iterations: int = 5000
    save_interval: int = 500
    gamma: float = 0.97
    tau: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 512
    num_atoms: int = 101
    obs_normalization: bool = True
    use_layer_norm: bool = False
    algo_params: TD3AlgoParams = field(default_factory=TD3AlgoParams)


# ── Off-policy: FlashSAC ─────────────────────────────────────────────────────


@dataclass
class FlashSACAlgoParams:
    normalize_reward: bool = True
    normalized_g_max: float = 5.0
    actor_num_blocks: int = 2
    critic_num_blocks: int = 2
    actor_bc_alpha: float = 0.0
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    critic_min_v: float = -5.0
    critic_max_v: float = 5.0
    temp_initial_value: float = 0.01
    temp_target_sigma: float = 0.15
    temp_target_entropy: float | None = None
    learning_rate_init: float = 3e-4
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 1.5e-4
    learning_rate_warmup_steps: int = 0
    learning_rate_decay_steps: int = 500000
    n_step: int = 1
    amp_dtype: str = "auto"
    use_compile: bool = True


@dataclass
class FlashSACConfig(BaseConfig):
    algo: str = "flashsac"
    algo_log_name: str = "flash_sac"
    seed: int = 1
    num_envs: int = 1024
    batch_size: int = 2048
    replay_buffer_n: int = 512
    updates_per_step: int = 2
    learning_starts: int = 98
    policy_frequency: int = 2
    env_steps_per_sync: int = 1
    max_iterations: int = 5000
    save_interval: int = 1000
    gamma: float = 0.97
    tau: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    num_atoms: int = 101
    obs_normalization: bool = False
    use_layer_norm: bool = False
    algo_params: FlashSACAlgoParams = field(default_factory=FlashSACAlgoParams)


# ── APPO ─────────────────────────────────────────────────────────────────────


@dataclass
class APPOAlgorithmConfig:
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    adaptive_kl_factor: float = 1.2
    adaptive_lr_factor: float = 1.1
    optimizer: str = "adam"
    tau: float = 1.0
    target_update_freq: int = 1
    vtrace_clip_rho: float = 1.0
    vtrace_clip_c: float = 1.0
    enable_compile: bool = True


@dataclass
class APPODistributionConfig:
    class_name: str = "rsl_rl.modules.distribution.GaussianDistribution"
    init_std: float = 1.0
    std_type: str = "scalar"


@dataclass
class APPOActorConfig:
    class_name: str = "rsl_rl.models.MLPModel"
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    distribution_cfg: APPODistributionConfig = field(default_factory=APPODistributionConfig)


@dataclass
class APPOCriticConfig:
    class_name: str = "rsl_rl.models.MLPModel"
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"


@dataclass
class APPOConfig(BaseConfig):
    algo: str = "appo"
    algo_log_name: str = "appo"
    seed: int = 1
    num_envs: int = 2048
    steps_per_env: int = 24
    max_iterations: int = 150
    save_interval: int = 50
    obs_groups: dict = field(default_factory=lambda: {"actor": {"policy": 0}})
    actor: APPOActorConfig = field(default_factory=APPOActorConfig)
    critic: APPOCriticConfig = field(default_factory=APPOCriticConfig)
    algorithm: APPOAlgorithmConfig = field(default_factory=APPOAlgorithmConfig)


# ── PPO (rsl-rl) ─────────────────────────────────────────────────────────────


@dataclass
class PPOPolicyConfig:
    init_noise_std: float = 1.0
    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    class_name: str = "ActorCritic"


@dataclass
class PPOAlgorithmConfig:
    class_name: str = "unilab.algos.torch.rsl_rl_ppo:FinalObservationAwarePPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    target_kl_stop: Optional[float] = None
    max_grad_norm: float = 1.0
    adaptive_kl_beta: float = 0.9
    adaptive_lr_growth: float = 1.1
    adaptive_lr_decay: float = 1.2
    adaptive_lr_update_interval: int = 5
    metrics_interval: int = 8
    finite_check_interval: int = 8
    enable_compile: bool = True
    warmup_strict_iters: int = 10
    warmup_metrics_interval: int = 2
    warmup_finite_check_interval: int = 2
    disable_finite_checks: bool = True


@dataclass
class PPOConfig(BaseConfig):
    algo: str = "ppo"
    algo_log_name: str = "rsl_rl_ppo"
    seed: int = 1
    num_envs: int = 4096
    num_steps_per_env: int = 24
    max_iterations: int = 101
    save_interval: int = 100
    empirical_normalization: bool = False
    runner_class_name: str = "OnPolicyRunner"
    obs_groups: dict = field(default_factory=lambda: {"default": ["policy"]})
    experiment_name: str = "test"
    run_name: str = ""
    resume: bool = False
    load_run: str = "-1"
    checkpoint: int = -1
    resume_path: Optional[str] = None
    policy: PPOPolicyConfig = field(default_factory=PPOPolicyConfig)
    algorithm: PPOAlgorithmConfig = field(default_factory=PPOAlgorithmConfig)
