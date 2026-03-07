"""SAC runner using unified off-policy framework."""

from unilab.algos.torch.offpolicy.runner import OffPolicyRunner
from unilab.algos.torch.fast_sac.learner import FastSACLearner


class UnifiedSACRunner(OffPolicyRunner):
    """SAC runner using unified infrastructure."""

    def __init__(
        self,
        env_name: str,
        device: str = None,
        num_envs: int = 4096,
        batch_size: int = 8192,
        warmup_steps: int = 0,
        updates_per_step: int = 8,
        policy_frequency: int = 4,
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.001,
        target_entropy_ratio: float = 0.0,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        use_layer_norm: bool = True,
        max_grad_norm: float = 0.0,
        sync_collection: bool = True,
        env_steps_per_sync: int = 1,
    ):
        # Detect dimensions
        from unilab.envs import registry
        from unilab.algos.torch.common.worker import ensure_registries
        ensure_registries()
        env = registry.make(env_name, num_envs=1, sim_backend="mujoco")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

        # Create learner
        learner = FastSACLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device or "mps",
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            alpha_init=alpha_init,
            target_entropy_ratio=target_entropy_ratio,
            actor_hidden_dim=actor_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
            num_atoms=num_atoms,
            use_layer_norm=use_layer_norm,
            max_grad_norm=max_grad_norm,
        )
        learner.policy_frequency = policy_frequency

        super().__init__(
            learner=learner,
            env_name=env_name,
            algo_type="sac",
            num_envs=num_envs,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
            sync_collection=sync_collection,
            env_steps_per_sync=env_steps_per_sync,
            device=device,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
        )
