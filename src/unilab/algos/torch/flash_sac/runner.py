"""FlashSAC runner using the shared off-policy runtime."""

from __future__ import annotations

from typing import Any

from unilab.algos.torch.flash_sac.learner import FlashSACLearner
from unilab.algos.torch.offpolicy.runner import OffPolicyRunner
from unilab.utils.device import get_default_device


class FlashSACRunner(OffPolicyRunner):
    def __init__(
        self,
        env_name: str,
        env_cfg_override: dict[str, Any] | None = None,
        device: str | None = None,
        num_envs: int = 2048,
        replay_buffer_n: int = 512,
        batch_size: int = 2048,
        learning_starts: int = 0,
        updates_per_step: int = 1,
        policy_frequency: int = 2,
        sync_collection: bool = True,
        env_steps_per_sync: int = 1,
        gamma: float = 0.99,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        obs_normalization: bool = False,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        num_atoms: int = 101,
        use_amp: bool = False,
        sim_backend: str = "mujoco",
        actor_num_blocks: int = 2,
        critic_num_blocks: int = 2,
        actor_bc_alpha: float = 0.0,
        actor_noise_zeta_mu: float = 2.0,
        actor_noise_zeta_max: int = 16,
        critic_min_v: float = -5.0,
        critic_max_v: float = 5.0,
        target_sigma: float = 0.15,
        target_entropy: float | None = None,
        temp_initial_value: float = 0.01,
        learning_rate_init: float = 3e-4,
        learning_rate_peak: float = 3e-4,
        learning_rate_end: float = 1.5e-4,
        learning_rate_warmup_steps: int = 0,
        learning_rate_decay_steps: int = 500000,
        normalize_reward: bool = True,
        normalized_g_max: float = 5.0,
        n_step: int = 1,
        use_compile: bool = False,
        seed: int | None = None,
        trace_enabled: bool = False,
        trace_output_dir: str | None = None,
        trace_thread_time: bool = False,
        trace_cuda_events: bool = True,
    ):
        from unilab.base import registry
        from unilab.base.observations import get_obs_dims
        from unilab.base.registry import ensure_registries
        from unilab.training.seed import apply_training_seed

        ensure_registries()
        apply_training_seed(seed, torch_runtime=True, cuda=True)
        env: Any = registry.make(
            env_name, num_envs=1, sim_backend=sim_backend, env_cfg_override=env_cfg_override
        )
        obs_dim, critic_obs_dim = get_obs_dims(env.obs_groups_spec)
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = int(action_shape[0])
        env.close()

        runtime_device = device or get_default_device()
        learner = FlashSACLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            critic_obs_dim=critic_obs_dim,
            device=runtime_device,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_hidden_dim=actor_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
            actor_num_blocks=actor_num_blocks,
            critic_num_blocks=critic_num_blocks,
            num_atoms=num_atoms,
            critic_min_v=critic_min_v,
            critic_max_v=critic_max_v,
            temp_initial_value=temp_initial_value,
            temp_target_sigma=target_sigma,
            temp_target_entropy=target_entropy,
            actor_bc_alpha=actor_bc_alpha,
            actor_noise_zeta_mu=actor_noise_zeta_mu,
            actor_noise_zeta_max=actor_noise_zeta_max,
            learning_rate_init=learning_rate_init,
            learning_rate_peak=learning_rate_peak,
            learning_rate_end=learning_rate_end,
            learning_rate_warmup_steps=learning_rate_warmup_steps,
            learning_rate_decay_steps=learning_rate_decay_steps,
            normalize_reward=normalize_reward,
            normalized_g_max=normalized_g_max,
            n_step=n_step,
            obs_normalization=obs_normalization,
            use_amp=use_amp,
            use_compile=use_compile,
        )

        super().__init__(
            learner=learner,
            env_name=env_name,
            algo_type="flashsac",
            num_envs=num_envs,
            replay_buffer_n=replay_buffer_n,
            batch_size=batch_size,
            learning_starts=learning_starts,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
            sync_collection=sync_collection,
            env_steps_per_sync=env_steps_per_sync,
            device=runtime_device,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=False,
            seed=seed,
            obs_normalization=obs_normalization,
            sim_backend=sim_backend,
            env_cfg_override=env_cfg_override,
            actor_kwargs={
                "actor_num_blocks": actor_num_blocks,
                "actor_noise_zeta_mu": actor_noise_zeta_mu,
                "actor_noise_zeta_max": actor_noise_zeta_max,
            },
            trace_enabled=trace_enabled,
            trace_output_dir=trace_output_dir,
            trace_thread_time=trace_thread_time,
            trace_cuda_events=trace_cuda_events,
        )
