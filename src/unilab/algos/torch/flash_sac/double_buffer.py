"""FlashSAC builder for the CPU-pinned double-buffer replay path."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from unilab.algos.torch.flash_sac.learner import FlashSACLearner
from unilab.algos.torch.offpolicy.distributed import validate_distributed_learner_capability
from unilab.algos.torch.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from unilab.training import create_env, ensure_registries
from unilab.training.seed import apply_training_seed
from unilab.utils.device import get_default_device
from unilab.utils.nan_guard import NanGuardCfg


def _validate_flashsac_double_buffer_runtime(
    cfg: DictConfig,
    *,
    device: str,
    replay_prefetch_mode: str,
) -> None:
    if cfg.training.num_gpus > 1:
        if not str(device).startswith("cuda"):
            raise ValueError("FlashSAC multi-GPU training requires a CUDA device")
        validate_distributed_learner_capability(
            learner_cls=FlashSACLearner,
            algo_type="flashsac",
            learner_kwargs={},
            num_gpus=int(cfg.training.num_gpus),
            sync_mode=str(getattr(cfg.training, "multi_gpu_sync_mode", "local_sgd")),
        )
    if cfg.training.no_sync_collection:
        raise ValueError("FlashSAC-B cpu_pinned_double_buffer requires synchronized collection")
    if replay_prefetch_mode != "one_tick":
        raise ValueError(
            "FlashSAC-B cpu_pinned_double_buffer requires replay_prefetch_mode='one_tick'"
        )
    if cfg.algo.algo_params.n_step != 1:
        raise ValueError("FlashSAC-B initially supports n_step=1 only")


def build_flashsac_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    verbose_metrics: bool,
    nan_guard_cfg: NanGuardCfg | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
) -> Any:
    """Build FlashSAC with the opt-in CPU-pinned double-buffer replay pipeline."""
    from unilab.base.observations import get_obs_dims

    ensure_registries()
    apply_training_seed(cfg.algo.seed, torch_runtime=True, cuda=True)
    device = cfg.training.device or get_default_device()
    collector_infer_device = str(getattr(cfg.training, "collector_infer_device", "cpu") or "cpu")
    _validate_flashsac_double_buffer_runtime(
        cfg,
        device=device,
        replay_prefetch_mode=replay_prefetch_mode,
    )

    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    try:
        obs_dim, critic_obs_dim = get_obs_dims(env.obs_groups_spec)
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = int(action_shape[0])
    finally:
        env.close()

    learner_kwargs = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "critic_obs_dim": critic_obs_dim,
        "gamma": cfg.algo.gamma,
        "tau": cfg.algo.tau,
        "actor_lr": cfg.algo.actor_lr,
        "critic_lr": cfg.algo.critic_lr,
        "actor_hidden_dim": cfg.algo.actor_hidden_dim,
        "critic_hidden_dim": cfg.algo.critic_hidden_dim,
        "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
        "critic_num_blocks": cfg.algo.algo_params.critic_num_blocks,
        "num_atoms": cfg.algo.num_atoms,
        "critic_min_v": cfg.algo.algo_params.critic_min_v,
        "critic_max_v": cfg.algo.algo_params.critic_max_v,
        "temp_initial_value": cfg.algo.algo_params.temp_initial_value,
        "temp_target_sigma": cfg.algo.algo_params.temp_target_sigma,
        "temp_target_entropy": cfg.algo.algo_params.temp_target_entropy,
        "actor_bc_alpha": cfg.algo.algo_params.actor_bc_alpha,
        "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
        "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
        "learning_rate_init": cfg.algo.algo_params.learning_rate_init,
        "learning_rate_peak": cfg.algo.algo_params.learning_rate_peak,
        "learning_rate_end": cfg.algo.algo_params.learning_rate_end,
        "learning_rate_warmup_steps": cfg.algo.algo_params.learning_rate_warmup_steps,
        "learning_rate_decay_steps": cfg.algo.algo_params.learning_rate_decay_steps,
        "normalize_reward": cfg.algo.algo_params.normalize_reward,
        "normalized_g_max": cfg.algo.algo_params.normalized_g_max,
        "n_step": cfg.algo.algo_params.n_step,
        "obs_normalization": cfg.algo.obs_normalization,
        "use_amp": cfg.training.use_amp,
        "amp_dtype": cfg.algo.algo_params.amp_dtype,
        "use_compile": cfg.algo.algo_params.use_compile,
        "use_cuda_graph_critic": cfg.algo.algo_params.use_cuda_graph_critic,
        "use_cuda_graph_actor": cfg.algo.algo_params.use_cuda_graph_actor,
        "use_cuda_graph_critic_packed_staging": (
            cfg.algo.algo_params.use_cuda_graph_critic_packed_staging
        ),
        "use_cuda_graph_actor_packed_staging": (
            cfg.algo.algo_params.use_cuda_graph_actor_packed_staging
        ),
    }
    actor_kwargs = {
        "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
        "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
        "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
    }

    if cfg.training.num_gpus > 1:
        from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

        validate_distributed_learner_capability(
            learner_cls=FlashSACLearner,
            algo_type="flashsac",
            learner_kwargs=learner_kwargs,
            num_gpus=int(cfg.training.num_gpus),
            sync_mode=str(getattr(cfg.training, "multi_gpu_sync_mode", "local_sgd")),
        )
        learner = FlashSACLearner(device=device, **learner_kwargs)
        return MultiGPUOffPolicyRunner(
            learner=learner,
            env_name=cfg.training.task_name,
            algo_type="flashsac",
            learner_cls=FlashSACLearner,
            learner_kwargs=learner_kwargs,
            num_gpus=int(cfg.training.num_gpus),
            distributed_backend="nccl",
            multi_gpu_sync_mode=str(getattr(cfg.training, "multi_gpu_sync_mode", "local_sgd")),
            multi_gpu_sync_interval=int(getattr(cfg.training, "multi_gpu_sync_interval", 1)),
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=cfg.algo.batch_size,
            learning_starts=cfg.algo.learning_starts,
            updates_per_step=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            sync_collection=not cfg.training.no_sync_collection,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
            device=device,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            use_layer_norm=False,
            obs_normalization=cfg.algo.obs_normalization,
            sim_backend=cfg.training.sim_backend,
            env_cfg_override=env_cfg_override,
            actor_kwargs=actor_kwargs,
            seed=cfg.algo.seed,
            trace_enabled=cfg.training.trace_enabled,
            trace_output_dir=cfg.training.trace_output_dir,
            trace_thread_time=cfg.training.trace_thread_time,
            trace_cuda_events=cfg.training.trace_cuda_events,
            nan_guard_cfg=nan_guard_cfg,
            collector_infer_device=collector_infer_device,
            torch_thread_runtime=torch_thread_runtime,
        )

    learner = FlashSACLearner(device=device, **learner_kwargs)

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type="flashsac",
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=cfg.algo.batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=cfg.algo.updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        sync_collection=not cfg.training.no_sync_collection,
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        device=device,
        actor_hidden_dim=cfg.algo.actor_hidden_dim,
        use_layer_norm=False,
        obs_normalization=cfg.algo.obs_normalization,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
        actor_kwargs=actor_kwargs,
        seed=cfg.algo.seed,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        verbose_metrics=verbose_metrics,
        nan_guard_cfg=nan_guard_cfg,
        collector_infer_device=collector_infer_device,
        torch_thread_runtime=torch_thread_runtime,
    )
