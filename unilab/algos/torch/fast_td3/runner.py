"""FastTD3 runner built on top of the unified off-policy infra."""

from unilab.algos.torch.fast_td3.learner import FastTD3Learner
from unilab.algos.torch.offpolicy.runner import OffPolicyRunner


class FastTD3Runner(OffPolicyRunner):
    """FastTD3 runner using the shared OffPolicyRunner training loop."""

    def __init__(
        self,
        env_name: str,
        device: str | None = None,
        num_envs: int = 4096,
        replay_buffer_n: int = 1000,
        batch_size: int = 8192,
        warmup_steps: int = 50,
        num_updates: int = 4,
        policy_frequency: int = 2,
        # Collection/training synchronization
        sync_collection: bool = True,
        env_steps_per_sync: int = 1,
        # Algorithm parameters
        gamma: float = 0.97,
        tau: float = 0.1,
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
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        weight_decay: float = 0.1,
        use_cdq: bool = True,
        obs_normalization: bool = True,
        sim_backend: str = "mujoco",
    ):
        obs_dim, action_dim = self._detect_obs_action_dims(env_name, sim_backend)
        learner = FastTD3Learner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_envs=num_envs,
            device=device or self._default_device(),
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
            warmup_steps=warmup_steps,
            updates_per_step=num_updates,
            policy_frequency=policy_frequency,
            sync_collection=sync_collection,
            env_steps_per_sync=env_steps_per_sync,
            device=device,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=False,
            obs_normalization=obs_normalization,
            sim_backend=sim_backend,
        )

    @staticmethod
    def _default_device() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _detect_obs_action_dims(env_name: str, sim_backend: str = "mujoco") -> tuple[int, int]:
        from unilab.base import registry
        from unilab.utils.algo_utils import ensure_registries

        ensure_registries()
        env = registry.make(env_name, num_envs=1, sim_backend=sim_backend)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        return obs_dim, action_dim
