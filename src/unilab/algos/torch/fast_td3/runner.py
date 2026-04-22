"""FastTD3 runner built on top of the unified off-policy infra."""

from unilab.algos.torch.fast_td3.learner import FastTD3Learner
from unilab.algos.torch.offpolicy.runner import OffPolicyRunner
from unilab.utils.device_utils import get_default_device, get_env_dims


class FastTD3Runner(OffPolicyRunner):
    """FastTD3 runner using the shared OffPolicyRunner training loop."""

    def __init__(
        self,
        env_name: str,
        env_cfg_override: dict | None = None,
        device: str | None = None,
        num_envs: int = 4096,
        replay_buffer_n: int = 1000,
        batch_size: int = 8192,
        learning_starts: int = 0,
        num_updates: int = 4,
        policy_frequency: int = 2,
        # Collection/training synchronization
        sync_collection: bool = True,
        env_steps_per_sync: int = 1,
        # Algorithm parameters
        gamma: float = 0.97,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 512,
        num_atoms: int = 101,
        v_min: float = -10.0,
        v_max: float = 10.0,
        init_scale: float = 0.01,
        log_std_min: float = -0.9,
        log_std_max: float = 0.0,
        policy_noise: float = 0.1,
        noise_clip: float = 0.2,
        weight_decay: float = 0.001,
        use_cdq: bool = True,
        obs_normalization: bool = True,
        sim_backend: str = "mujoco",
    ):
        obs_dim, action_dim, critic_obs_dim = get_env_dims(env_name, sim_backend, env_cfg_override)
        learner = FastTD3Learner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            critic_obs_dim=critic_obs_dim,
            num_envs=num_envs,
            device=device or get_default_device(),
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_hidden_dim=actor_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            init_scale=init_scale,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            weight_decay=weight_decay,
            use_cdq=use_cdq,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_frequency=policy_frequency,
            max_iterations=1,
            obs_normalization=obs_normalization,
        )

        super().__init__(
            learner=learner,
            env_name=env_name,
            algo_type="td3",
            num_envs=num_envs,
            replay_buffer_n=replay_buffer_n,
            batch_size=batch_size,
            learning_starts=learning_starts,
            updates_per_step=num_updates,
            policy_frequency=policy_frequency,
            sync_collection=sync_collection,
            env_steps_per_sync=env_steps_per_sync,
            device=device,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=False,
            obs_normalization=obs_normalization,
            sim_backend=sim_backend,
            env_cfg_override=env_cfg_override,
        )
