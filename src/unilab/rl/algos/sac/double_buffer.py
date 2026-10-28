"""SAC builder for the device-authoritative replay path."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.base.contracts.env import EnvFactory
from unilab.base.contracts.observations import get_obs_dims
from unilab.rl.algos.sac.learner import SACLearner
from unilab.rl.runtime.ipc.replay_pipelines.device import (
    require_offpolicy_replay_device,
)
from unilab.rl.runtime.offpolicy.actor_adapter import import_actor_adapter_modules
from unilab.rl.runtime.offpolicy.double_buffer_runner import (
    DoubleBufferOffPolicyRunner,
    validate_no_inference_request_timeout,
)
from unilab.rl.runtime.offpolicy.resolution import (
    resolve_actor_adapter_modules,
    resolve_custom_offpolicy_runtime,
)
from unilab.rl.runtime.placement import InferencePlacement
from unilab.rl.runtime.settings import RuntimeSettings

if TYPE_CHECKING:
    from unilab.rl.runtime.distributed.sync import DpParameterSync


def build_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_factory: EnvFactory,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    device: str,
    nan_guard_cfg: Any | None = None,
    nan_guard_factory: Callable[[Any, int, bool], Any] | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
    collector_cpu_ids: list[int] | None = None,
    dp_sync: DpParameterSync | None = None,
    backend_device_resolver: Callable[[str, str | None], str | None] | None = None,
    backend_device_configurer: Callable[[str, str | None], str | None] | None = None,
    runtime_settings: RuntimeSettings | None = None,
    inference_placement: InferencePlacement | None = None,
) -> Any:
    """Build SAC from its Hydra owner config without interpreting it in the entrypoint.

    ``runtime_settings`` and ``inference_placement`` are resolved from
    the owner config by the training assembly layer
    (:mod:`unilab.training.tensor_runtime`) and injected here; when omitted the
    runner-side defaults apply. Backend process-device resolution/binding is
    likewise injected by the owner layer (e.g.
    :mod:`unilab.base.process_device`).
    """
    device = require_offpolicy_replay_device(device)
    rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    custom_runtime = resolve_custom_offpolicy_runtime(rl_cfg)
    # Import registration modules before the learner builds its actor so
    # custom adapter registrations are live in this process; the runner
    # forwards the list to the spawn collector.
    actor_adapter_modules = resolve_actor_adapter_modules(rl_cfg, custom_runtime)
    import_actor_adapter_modules(actor_adapter_modules)
    if runtime_settings is not None:
        batch_size = runtime_settings.batch_size
        updates_per_step = runtime_settings.updates_per_step
    else:
        batch_size = int(cfg.algo.batch_size)
        updates_per_step = int(cfg.algo.updates_per_step)

    validate_no_inference_request_timeout(cfg.training)
    env = env_factory(1, env_cfg_override)
    try:
        action_shape = env.action_space.shape
        assert action_shape
        obs_dim, critic_obs_dim = get_obs_dims(dict(env.obs_groups_spec))
        action_dim = int(action_shape[0])
    finally:
        env.close()

    learner_cls: type[Any] = SACLearner
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
        "nvtx_profile_ranges": bool(getattr(cfg.training, "nvtx_profile_ranges", False)),
        "critic_obs_dim": critic_obs_dim,
    }
    learner_kwargs.update(learner_extra_kwargs)
    learner = learner_cls(device=device, **learner_kwargs)

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type=algo_type,
        env_factory=env_factory,
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=updates_per_step,
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
        nan_guard_factory=nan_guard_factory,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
        learner_prepare_hook=(
            custom_runtime.learner_prepare_hook if custom_runtime is not None else None
        ),
        backend_device_resolver=backend_device_resolver,
        backend_device_configurer=backend_device_configurer,
        actor_adapter_modules=actor_adapter_modules,
        inference_placement=inference_placement,
        runtime_settings=runtime_settings,
        log_interval=int(cfg.training.log_interval),
    )
