"""FastSAC runner using unified OffPolicyRunner."""

from typing import Any

from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.algos.torch.offpolicy.runner import OffPolicyRunner
from unilab.utils.device_utils import get_default_device, get_env_dims


class FastSACRunner(OffPolicyRunner):
    """FastSAC using OffPolicyRunner infrastructure."""

    def __init__(
        self,
        env_name: str,
        env_cfg_override: dict[str, Any] | None = None,
        device: str | None = None,
        num_envs: int = 4096,
        replay_buffer_n: int = 1024,
        batch_size: int = 8192,
        warmup_steps: int = 0,
        updates_per_step: int = 8,
        policy_frequency: int = 4,
        sync_collection: bool = True,
        env_steps_per_sync: int = 1,
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.001,
        target_entropy_ratio: float = 1.0,
        obs_normalization: bool = True,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        use_layer_norm: bool = True,
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        sim_backend: str = "mujoco",
        use_symmetry: bool = False,
        world_size: int = 1,
    ):
        from unilab.base import registry
        from unilab.utils.algo_utils import ensure_registries

        ensure_registries()
        env: Any = registry.make(
            env_name, num_envs=1, sim_backend=sim_backend, env_cfg_override=env_cfg_override
        )
        from unilab.utils.obs_utils import get_obs_dims

        obs_dim, critic_obs_dim = get_obs_dims(env.obs_groups_spec)
        act_space_shape = env.action_space.shape
        assert act_space_shape is not None
        action_dim = act_space_shape[0]
        device = device or get_default_device()
        symmetry_augmentation = None
        if use_symmetry:
            symmetry_augmentation = env.build_symmetry_augmentation(device=device)
            if symmetry_augmentation is None:
                env.close()
                raise ValueError(
                    f"{env_name} with backend={sim_backend} does not provide symmetry augmentation"
                )
        env.close()

        learner = FastSACLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
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
            use_amp=use_amp,
            use_symmetry=use_symmetry,
            symmetry_augmentation=symmetry_augmentation,
            world_size=getattr(self, "world_size", world_size),
            critic_obs_dim=critic_obs_dim,
        )

        if symmetry_augmentation is not None:
            if batch_size % symmetry_augmentation.batch_multiplier != 0:
                raise ValueError(
                    "Symmetry augmentation requires algo.batch_size to be divisible by "
                    f"{symmetry_augmentation.batch_multiplier}, got {batch_size}"
                )
            batch_size = batch_size // symmetry_augmentation.batch_multiplier
            print(
                "[FastSAC] Symmetry enabled: "
                f"batch_size adjusted to {batch_size} "
                f"(effective: {batch_size * symmetry_augmentation.batch_multiplier})"
            )

        super().__init__(
            learner=learner,
            env_name=env_name,
            algo_type="sac",
            num_envs=num_envs,
            replay_buffer_n=replay_buffer_n,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
            sync_collection=sync_collection,
            env_steps_per_sync=env_steps_per_sync,
            device=device,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            obs_normalization=obs_normalization,
            sim_backend=sim_backend,
            env_cfg_override=env_cfg_override,
        )
