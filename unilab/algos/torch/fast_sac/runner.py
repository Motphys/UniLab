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
    ):
        from unilab.base import registry
        from unilab.utils.algo_utils import ensure_registries

        ensure_registries()
        env: Any = registry.make(env_name, num_envs=1, sim_backend=sim_backend)
        obs_space_shape = env.observation_space.shape
        act_space_shape = env.action_space.shape
        assert obs_space_shape is not None and act_space_shape is not None
        obs_dim = obs_space_shape[0]
        action_dim = act_space_shape[0]
        mujoco_model = getattr(env, "_backend", None)
        if mujoco_model is not None:
            mujoco_model = getattr(mujoco_model, "model", None)
        obs_structure = getattr(env, "get_obs_structure", lambda: None)()
        env.close()

        device = device or get_default_device()

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
            mujoco_model=mujoco_model,
            obs_structure=obs_structure,
        )

        # Auto-adjust batch_size when symmetry is enabled
        if use_symmetry:
            batch_size = batch_size // 2
            print(
                f"[FastSAC] Symmetry enabled: batch_size adjusted to {batch_size} (effective: {batch_size * 2})"
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
        )
