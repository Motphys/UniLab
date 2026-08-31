"""FastSAC builder for the device-authoritative replay path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.algos.fast_sac.learner import FastSACLearner
from unilab.algos.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from unilab.algos.offpolicy.runtime import resolve_custom_offpolicy_runtime
from unilab.base.config_adapter import create_env
from unilab.base.np_env import NpEnv
from unilab.base.registry import ensure_registries
from unilab.utils.nan_guard import NanGuardCfg

if TYPE_CHECKING:
    from unilab.ipc.dp_sync import DpParameterSync


def build_sac_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    device: str,
    nan_guard_cfg: NanGuardCfg | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
    collector_cpu_ids: list[int] | None = None,
    dp_sync: DpParameterSync | None = None,
) -> Any:
    """Build SAC from its Hydra owner config without interpreting it in the entrypoint."""
    from unilab.base.observations import get_obs_dims

    ensure_registries()
    rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    custom_runtime = resolve_custom_offpolicy_runtime(rl_cfg)

    env = cast(NpEnv, create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override))
    try:
        action_shape = env.action_space.shape
        assert action_shape
        obs_dim, critic_obs_dim = get_obs_dims(env.obs_groups_spec)
        action_dim = int(action_shape[0])
    finally:
        env.close()

    batch_size = cfg.algo.batch_size

    learner_cls: type[Any] = FastSACLearner
    algo_type = "sac"
    learner_extra_kwargs: dict[str, Any] = {}
    if custom_runtime is not None:
        learner_extra_kwargs = cast(
            dict[str, Any],
            custom_runtime.build_model_kwargs(
                obs_dim=int(obs_dim),
                critic_obs_dim=int(critic_obs_dim),
            ),
        )
        if custom_runtime.learner_cls is not None:
            learner_cls = custom_runtime.learner_cls
        if custom_runtime.algo_type is not None:
            algo_type = str(custom_runtime.algo_type)

    learner_kwargs = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "gamma": cfg.algo.gamma,
        "tau": cfg.algo.tau,
        "actor_lr": cfg.algo.actor_lr,
        "critic_lr": cfg.algo.critic_lr,
        "alpha_lr": cfg.algo.algo_params.alpha_lr,
        "alpha_init": cfg.algo.algo_params.alpha_init,
        "target_entropy_ratio": cfg.algo.algo_params.target_entropy_ratio,
        "actor_hidden_dim": cfg.algo.actor_hidden_dim,
        "critic_hidden_dim": cfg.algo.critic_hidden_dim,
        "num_atoms": cfg.algo.num_atoms,
        "use_layer_norm": cfg.algo.use_layer_norm,
        "max_grad_norm": cfg.algo.algo_params.max_grad_norm,
        "use_amp": cfg.training.use_amp,
        "amp_dtype": cfg.algo.algo_params.amp_dtype,
        "use_compile": cfg.algo.algo_params.use_compile,
        "obs_normalization": cfg.algo.obs_normalization,
        "use_cuda_graph_critic": bool(
            getattr(cfg.algo.algo_params, "use_cuda_graph_critic", False)
        ),
        "use_cuda_graph_actor": bool(getattr(cfg.algo.algo_params, "use_cuda_graph_actor", False)),
        "use_cuda_graph_critic_packed_staging": bool(
            getattr(cfg.algo.algo_params, "use_cuda_graph_critic_packed_staging", False)
        ),
        "use_cuda_graph_actor_packed_staging": bool(
            getattr(cfg.algo.algo_params, "use_cuda_graph_actor_packed_staging", False)
        ),
        "nvtx_profile_ranges": bool(getattr(cfg.training, "nvtx_profile_ranges", False)),
        "critic_obs_dim": critic_obs_dim,
    }
    learner_kwargs.update(learner_extra_kwargs)
    learner = learner_cls(device=device, **learner_kwargs)

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type=algo_type,
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=cfg.algo.updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        device=device,
        obs_normalization=cfg.algo.obs_normalization,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        seed=cfg.algo.seed,
        nan_guard_cfg=nan_guard_cfg,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
        inference_request_timeout_sec=getattr(cfg.training, "inference_request_timeout_sec", None),
    )
