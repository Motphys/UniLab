import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from unisim.backend.base import DebugPrimitive, RenderClosedError, log_playback_plan
from unisim.backend.mujoco.xml import materialize_scene_visual_override

from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.base.process_device import (
    apply_backend_env_device_override,
    apply_collector_tensor_device_override,
    configure_backend_process_device,
    pin_genesis_device_before_cuda_init,
    resolve_backend_env_device_id,
)
from unilab.base.run_control import RunComplete
from unilab.interruption import graceful_keyboard_interrupt
from unilab.rl import (
    UNILAB_DP_LOG_DIR,
    RslRlVecEnvAdapter,
    apply_rsl_rl_rank_seed,
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
    finish_rsl_rl_distributed,
    launch_torchrun_workers,
    ppo_samples_per_iteration,
    resolve_collector_cpu_ids,
    resolve_rsl_rl_device,
    rsl_rl_single_process_topology,
)
from unilab.rl.runtime.tf32 import apply_tf32_matmul_from_cfg
from unilab.training import (
    algo_config_dict,
    apply_env_nan_guard,
    ensure_registries,
    format_play_checkpoint_error,
    get_log_root,
    is_viser_play_render_mode,
    nonfatal_play_step,
    resolve_configured_checkpoint_path,
    resolve_run_log_dir,
    should_run_playback,
)
from unilab.training.experiment import (
    ExperimentTracker,
    patch_rsl_rl_action_std_logging,
    patch_rsl_rl_resume_state,
    patch_rsl_rl_tensorboard_logging,
    patch_rsl_rl_wandb_writer,
)
from unilab.training.seed import apply_configured_training_seed
from unilab.training.tensor_runtime import (
    resolve_recorded_collector_tensor_device,
    select_collector_tensor_device,
)
from unilab.utils.device import get_default_device
from unilab.visualization.interactive_playback import (
    PlaybackConfig,
    create_rsl_rl_playback_session,
    infer_checkpoint_actor_input_dim,
    make_sim2sim_preflight,
    normalize_checkpoint_value,
)
from unilab.visualization.playback import camera_cfg_from_training

# Also defined for consumers that invoke main with a composed configuration.
EXPORT_POLICY = False

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)


def _backend_adapter(cfg: DictConfig) -> EnvConfigAdapter:
    return EnvConfigAdapter(
        cfg,
        root_dir=Path.cwd(),
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    )


def build_ppo_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    base = cast(dict[str, Any], _backend_adapter(cfg).build_task_env_cfg_override())
    rank = current_torch_distributed_rank()
    local_rank = current_torch_distributed_local_rank()
    world_size = current_torch_distributed_world_size()
    learner_device = resolve_rsl_rl_device(
        configured_device=OmegaConf.select(cfg, "training.device", default=None),
        world_size=world_size,
        local_rank=local_rank,
        default_device="cpu",
    )
    result = apply_backend_env_device_override(
        base,
        str(cfg.training.sim_backend),
        learner_device=learner_device,
    )
    result = apply_collector_tensor_device_override(
        result,
        str(cfg.training.sim_backend),
        collector_tensor_device=select_collector_tensor_device(cfg),
        learner_device=learner_device,
    )
    if world_size > 1:
        explicit = OmegaConf.select(cfg, "training.dp_collector_cpu_ids", default=None)
        explicit = OmegaConf.to_container(explicit, resolve=True) if explicit is not None else None
        result["cpu_ids"] = resolve_collector_cpu_ids(
            world_size,
            rank,
            None,
            explicit=explicit,
        )
    return result


def build_ppo_play_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    base = cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())
    local_rank = current_torch_distributed_local_rank()
    world_size = current_torch_distributed_world_size()
    learner_device = resolve_rsl_rl_device(
        configured_device=OmegaConf.select(cfg, "training.device", default=None),
        world_size=world_size,
        local_rank=local_rank,
        default_device="cpu",
    )
    result = apply_backend_env_device_override(
        base,
        str(cfg.training.sim_backend),
        learner_device=learner_device,
    )
    return apply_collector_tensor_device_override(
        result,
        str(cfg.training.sim_backend),
        collector_tensor_device=select_collector_tensor_device(cfg),
        learner_device=learner_device,
    )


def validate_ppo_run_completion_topology(
    cfg: DictConfig,
    *,
    world_size: int,
) -> None:
    """Reject grasp collection on a topology without run-completion coordination."""
    if cfg.training.play_only:
        return
    if OmegaConf.select(cfg, "env.grasp_collection_target", default=None) is None:
        return
    effective_world_size = world_size
    if effective_world_size > 1:
        raise ValueError(
            "Grasp collection run completion currently requires one process; "
            "multi-rank completion needs an explicit cross-rank lifecycle protocol"
        )


def _resolve_play_num_steps(cfg: DictConfig) -> int | None:
    play_steps = OmegaConf.select(cfg, "training.play_steps", default=None)
    if play_steps is None:
        return None
    return int(play_steps)


_EE_GOAL_MARKER_RADIUS = 0.025
_EE_GOAL_MARKER_RGBA = (1.0, 0.2, 0.2, 0.8)


def _ee_goal_debug_overlay_getter(env):
    """Build the per-frame EE-goal overlay getter for tasks that expose one.

    Returns ``None`` when the env has no ``curr_ee_goal_world`` marker or the
    backend does not advertise debug overlay support; in both cases playback
    runs without overlays (backends fail closed on unsupported getters).
    """
    if not hasattr(env, "curr_ee_goal_world"):
        return None
    capabilities = getattr(env, "play_capabilities", None)
    if capabilities is None or not capabilities.supports_debug_overlay:
        return None

    def _get_overlay():
        goals = getattr(env, "curr_ee_goal_world", None)
        if goals is None:
            return None
        goals = np.asarray(goals, dtype=np.float64)
        if goals.ndim != 2 or goals.shape[1] != 3:
            raise ValueError(f"curr_ee_goal_world must have shape (num_envs, 3); got {goals.shape}")
        return [
            (
                [
                    DebugPrimitive(
                        kind="sphere",
                        pos=(float(goal[0]), float(goal[1]), float(goal[2])),
                        size=(_EE_GOAL_MARKER_RADIUS,),
                        rgba=_EE_GOAL_MARKER_RGBA,
                    )
                ]
                if np.isfinite(goal).all()
                else None
            )
            for goal in goals
        ]

    return _get_overlay


def _playback_debug_overlay_getter(env):
    """Discover the play-path overlay getter for an env.

    Prefers task-owned overlays aggregated by ``get_playback_debug_overlays``
    (command terms implementing ``playback_debug_overlay_getter()``); falls
    back to the ``curr_ee_goal_world`` EE-goal special case. Returns ``None``
    when neither is available or the backend does not advertise debug overlay
    support. Mode-specific gating happens in ``run_playback_mode`` once the
    render plan is resolved: interactive playback on backends without
    ``supports_interactive_debug_overlay`` drops the getter with a warning
    instead of failing closed.
    """
    capabilities = getattr(env, "play_capabilities", None)
    if capabilities is None or not capabilities.supports_debug_overlay:
        return None
    discover = getattr(env, "get_playback_debug_overlays", None)
    if discover is not None:
        getter = discover()
        if getter is not None:
            return getter
    return _ee_goal_debug_overlay_getter(env)


def play_ppo(cfg: DictConfig, device: str) -> str | None:
    """Play mode for RSL-RL."""
    rl_cfg = algo_config_dict(cfg)

    task_log_root = get_log_root(Path.cwd(), cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = resolve_configured_checkpoint_path(cfg, root_dir=Path.cwd())
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            format_play_checkpoint_error(
                cfg,
                task_log_root=task_log_root,
                load_path=load_path,
                load_path_dir=load_path_dir,
            )
        )
        return None

    print(f"Loading latest model: {load_path}")
    _ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in _ckpt_keys:
        print(
            f"Checkpoint at {load_path} is not an rsl-rl checkpoint "
            f"(found keys: {_ckpt_keys}). Aborting play."
        )
        return None

    playback_cfg = PlaybackConfig(
        task=str(cfg.training.task_name),
        load_run=str(cfg.algo.load_run),
        checkpoint=normalize_checkpoint_value(
            OmegaConf.select(cfg, "algo.checkpoint", default=None)
        ),
        action_mode="policy",
        policy_obs_mode="flat",
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=None,
        num_envs=cfg.training.play_env_num,
    )
    # A non-zero Genesis request pins CUDA_VISIBLE_DEVICES here; adopt the bound
    # in-process device for both the policy and the env override.  CUDA-only
    # backends reject CPU/MPS learners before playback construction.
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound_device is not None:
        device = bound_device
    play_env_cfg_override = apply_backend_env_device_override(
        build_ppo_play_env_cfg_override(cfg),
        str(cfg.training.sim_backend),
        learner_device=device,
    )
    play_env_cfg_override = apply_collector_tensor_device_override(
        play_env_cfg_override,
        str(cfg.training.sim_backend),
        collector_tensor_device=select_collector_tensor_device(cfg),
        learner_device=device,
    )
    session, _policy_obs_mode, _checkpoint_path = create_rsl_rl_playback_session(
        playback_cfg=playback_cfg,
        env_factory=lambda n: create_env(
            cfg,
            num_envs=n,
            env_cfg_override=play_env_cfg_override,
        ),
        algo_config=rl_cfg,
        root_dir=Path.cwd(),
        device=device,
        checkpoint_resolver=lambda *_args: str(load_path),
        checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
        entrypoint_log_root=get_log_root,
        sim2sim_preflight=make_sim2sim_preflight(cfg, algo_name="ppo"),
        guard_algo_name="ppo",
    )
    env = session.env
    runner = session.runner
    if EXPORT_POLICY:
        # The checkpoint early-returns above guarantee a loaded runner here.
        assert runner is not None
        with nonfatal_play_step("policy export"):
            runner.export_policy_to_onnx(path=str(load_path_dir))
            runner.export_policy_to_jit(path=str(load_path_dir))
    if is_viser_play_render_mode(getattr(cfg.training, "play_render_mode", "auto")):
        # Browser-based viser playback renders through the shared MuJoCo
        # playback shell; it replaces backend-native playback and records no
        # video. Blocks until Ctrl+C. Imported lazily: the viser scene module
        # binds the optional `mujoco` package, which non-MuJoCo-shell training
        # environments do not install.
        from unilab.visualization.viser_playback import run_viser_playback_from_cfg

        run_viser_playback_from_cfg(session, cfg, entrypoint="train_rsl_rl play")
        return None
    num_steps = _resolve_play_num_steps(cfg)
    output_video = Path(load_path_dir) / "play_video.mp4"
    playback_mode: str | None = None
    play_video_path: str | None = None

    def _log_plan(plan) -> None:
        nonlocal playback_mode
        playback_mode = plan.mode
        log_playback_plan(plan)

    with nonfatal_play_step("video rendering"):
        try:
            with torch.inference_mode():
                play_video_path = env.run_playback_mode(
                    play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                    play_steps=num_steps,
                    output_video=output_video,
                    render_spacing=float(
                        getattr(
                            cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0)
                        )
                    ),
                    render_offset_mode=str(getattr(env.cfg, "render_offset_mode", "grid")),
                    initialize=session.reset,
                    step=lambda _obs: session.step_once(),
                    camera_kwargs=camera_cfg_from_training(cfg.training),
                    on_plan=_log_plan,
                    debug_overlay_getter=_playback_debug_overlay_getter(env),
                )
        except RenderClosedError:
            # Interface-level signal: the user closed the backend render window.
            print("Render window closed.")
    if playback_mode != "none" and num_steps is not None:
        print("Done.")
    return play_video_path


@graceful_keyboard_interrupt
@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    rank = current_torch_distributed_rank()
    local_rank = current_torch_distributed_local_rank()
    world_size = current_torch_distributed_world_size()
    configured_device = OmegaConf.select(cfg, "training.device", default=None)

    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK={rank} is out of range for WORLD_SIZE={world_size}")
    if cfg.training.play_only and world_size > 1:
        raise ValueError(
            "Distributed play-only execution is not supported; launch eval normally and "
            "select one device"
        )
    validate_ppo_run_completion_topology(cfg, world_size=world_size)

    if world_size > 1 and os.environ.get("LOCAL_RANK") == "0":
        # A launcher parent invokes this branch once. World size comes from the
        # torchrun environment; selected parent visibility has one entry/rank.
        launch_torchrun_workers(
            world_size=world_size,
            script_path=Path(__file__),
            argv=sys.argv[1:],
            log_dir=resolve_run_log_dir(cfg, world_size=world_size),
        )
        return

    if (
        world_size > 1
        and os.environ.get(UNILAB_DP_LOG_DIR) is None
        and OmegaConf.select(cfg, "training.log_dir", default=None) is None
    ):
        raise ValueError(
            "Distributed RSL-RL workers require one shared run directory; set "
            "training.log_dir explicitly or use the launcher"
        )

    # Genesis/Quadrants binds the first CUDA_VISIBLE_DEVICES entry, and even
    # torch.cuda.is_available() latches the variable in the CUDA runtime, so
    # the pin must precede registry bootstrap and device auto-detection
    # (issue #1508).  Pure config topology resolves without touching torch.
    pinned_device = pin_genesis_device_before_cuda_init(
        str(cfg.training.sim_backend),
        learner_device=str(configured_device) if configured_device is not None else None,
    )

    if pinned_device is not None and world_size > 1:
        # rsl_rl's multi-GPU guard expects device == cuda:{LOCAL_RANK}.  Inside
        # the pinned single-device namespace the rank-local index is 0, so
        # adopt it for downstream consumers (parameter sync keeps using the
        # unchanged global RANK/WORLD_SIZE).
        os.environ["LOCAL_RANK"] = "0"
        local_rank = 0

    ensure_registries()
    apply_rsl_rl_rank_seed(cfg, rank)
    device = resolve_rsl_rl_device(
        configured_device=str(configured_device) if configured_device is not None else None,
        world_size=world_size,
        local_rank=local_rank,
        default_device=get_default_device(),
    )
    if pinned_device is not None:
        # The process was pinned to its rank GPU; use the in-process index.
        device = pinned_device
    # PPO workers inherit one launcher-selected CUDA_VISIBLE_DEVICES entry, so
    # the rank-local ordinal 0 is the simulator payload id.  Route this before env
    # construction and before Genesis/torch global initialization.
    # Payload routing is handled by the env override; there is no separate
    # non-Genesis payload binding at this layer.
    # Genesis pins CUDA_VISIBLE_DEVICES for a non-zero request (Quadrants only
    # honors the first visible device); the bound value is the device this
    # process must actually use afterwards.  CUDA-only backends reject CPU/MPS
    # learners before training construction.
    bound_device = configure_backend_process_device(
        str(cfg.training.sim_backend),
        device,
    )
    if bound_device is not None:
        device = bound_device
    print(f"[rank {rank}/{world_size}] Using device: {device}")
    env_cfg_override = apply_backend_env_device_override(
        build_ppo_env_cfg_override(cfg),
        str(cfg.training.sim_backend),
        learner_device=device,
    )
    env_cfg_override = apply_collector_tensor_device_override(
        env_cfg_override,
        str(cfg.training.sim_backend),
        collector_tensor_device=select_collector_tensor_device(cfg),
        learner_device=device,
    )
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    tf32_matmul_policy = apply_tf32_matmul_from_cfg(cfg, device=device)
    resolved_collector_tensor_device = resolve_recorded_collector_tensor_device(
        str(cfg.training.sim_backend),
        env_cfg_override,
    )

    # Compute effective max_iterations (supports num_timesteps override)
    max_iterations = cfg.algo.max_iterations
    if cfg.training.num_timesteps:
        n_steps_per_iter = ppo_samples_per_iteration(
            num_envs=cfg.algo.num_envs,
            num_steps_per_env=cfg.algo.num_steps_per_env,
            world_size=world_size,
        )
        max_iterations = max(1, int(cfg.training.num_timesteps / n_steps_per_iter))
        print(
            f"Overriding max_iterations to {max_iterations} based on "
            f"num_timesteps {cfg.training.num_timesteps}"
        )

    if not cfg.training.play_only:
        log_dir: str | None = resolve_run_log_dir(cfg, world_size=world_size)
    else:
        log_dir = None

    tracker = None
    if not cfg.training.play_only and log_dir is not None and rank == 0:
        tracker = ExperimentTracker(
            root_dir=Path.cwd(),
            log_dir=log_dir,
            algo_name="ppo",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=device,
            configured_device=str(cfg.training.device or "auto"),
            device_resolution={
                "learner": {
                    "configured": str(cfg.training.device or "auto"),
                    "resolved": str(device),
                },
                **(
                    {
                        "collector_tensor": {
                            "configured": select_collector_tensor_device(cfg),
                            "resolved": resolved_collector_tensor_device,
                        }
                    }
                    if resolved_collector_tensor_device is not None
                    else {}
                ),
            },
            seed_info=seed_info,
            tf32_matmul_policy=tf32_matmul_policy,
        )
        tracker.start()

    training_succeeded = False
    run_completion: RunComplete | None = None
    pending_summary: dict[str, object] | None = None
    try:
        try:
            if not cfg.training.play_only:
                env = create_env(
                    cfg,
                    num_envs=cfg.algo.num_envs,
                    env_cfg_override=env_cfg_override,
                )
                try:
                    train_cfg = algo_config_dict(cfg)

                    apply_env_nan_guard(env, cfg.training)

                    wrapped_env = RslRlVecEnvAdapter(env, device=device)

                    logger_type = (
                        cfg.training.logger
                        if cfg.training.logger in ["tensorboard", "wandb"]
                        else "none"
                    )
                    train_cfg["logger"] = logger_type

                    patch_rsl_rl_resume_state()

                    if tracker is not None and logger_type == "wandb":
                        patch_rsl_rl_wandb_writer()
                        wandb_settings = tracker.wandb_settings
                        train_cfg["wandb_project"] = wandb_settings["project"]
                        train_cfg["wandb_entity"] = wandb_settings["entity"]
                        train_cfg["wandb_group"] = wandb_settings["group"]
                        train_cfg["wandb_job_type"] = wandb_settings["job_type"]
                        train_cfg["wandb_tags"] = wandb_settings["tags"]
                        train_cfg["wandb_notes"] = wandb_settings["notes"]
                        train_cfg["wandb_mode"] = wandb_settings["mode"]

                    runner = cast(
                        Any,
                        OnPolicyRunner(
                            cast(Any, wrapped_env), train_cfg, log_dir=log_dir, device=device
                        ),
                    )
                    patch_rsl_rl_action_std_logging(runner)
                    patch_rsl_rl_tensorboard_logging(
                        runner, log_interval=int(cfg.training.log_interval)
                    )

                    if cfg.algo.load_run != "-1":
                        resume_path, _ = resolve_configured_checkpoint_path(
                            cfg, root_dir=Path.cwd()
                        )
                        if resume_path:
                            print(f"Resuming from {resume_path}")
                            runner.load(str(resume_path), map_location=device)

                    initial_timesteps = int(getattr(runner.logger, "tot_timesteps", 0))
                    initial_training_time = float(getattr(runner.logger, "tot_time", 0.0))
                    train_start_wall = time.time()
                    try:
                        runner.learn(
                            num_learning_iterations=max_iterations,
                            init_at_random_ep_len=True,
                        )
                    except RunComplete as completion:
                        if world_size != 1:
                            raise RuntimeError(
                                "RSL-RL RunComplete handling requires a single-process topology"
                            ) from completion
                        run_completion = completion
                    if run_completion is None and rank == 0:
                        assert log_dir is not None
                        total_timesteps = int(getattr(runner.logger, "tot_timesteps", 0))
                        total_training_time = float(getattr(runner.logger, "tot_time", 0.0))
                        run_timesteps = total_timesteps - initial_timesteps
                        run_training_time = total_training_time - initial_training_time
                        pending_summary = {
                            "status": "completed",
                            "backend_diagnostics": env.get_backend_runtime_diagnostics(),
                            "tf32_matmul": tf32_matmul_policy.manifest(),
                            "completed_iterations": int(runner.current_learning_iteration),
                            "total_env_steps": total_timesteps,
                            "run_env_steps": run_timesteps,
                            "world_size": world_size,
                            "num_envs_per_rank": int(cfg.algo.num_envs),
                            "global_num_envs": int(cfg.algo.num_envs) * world_size,
                            "samples_per_iteration": ppo_samples_per_iteration(
                                num_envs=cfg.algo.num_envs,
                                num_steps_per_env=cfg.algo.num_steps_per_env,
                                world_size=world_size,
                            ),
                            "training_throughput_env_steps_per_sec": (
                                run_timesteps / run_training_time
                                if run_training_time > 0.0
                                else None
                            ),
                            "final_mean_reward": (
                                float(statistics.mean(runner.logger.rewbuffer))
                                if len(getattr(runner.logger, "rewbuffer", [])) > 0
                                else None
                            ),
                            "best_mean_reward": (
                                float(max(runner.logger.rewbuffer))
                                if len(getattr(runner.logger, "rewbuffer", [])) > 0
                                else None
                            ),
                            "mean_episode_length": (
                                float(statistics.mean(runner.logger.lenbuffer))
                                if len(getattr(runner.logger, "lenbuffer", [])) > 0
                                else None
                            ),
                            "last_checkpoint": str(
                                Path(log_dir) / f"model_{int(runner.current_learning_iteration)}.pt"
                            ),
                            "training_wall_time_sec": time.time() - train_start_wall,
                        }
                finally:
                    env.close()
                if run_completion is not None:
                    pending_summary = {
                        **dict(run_completion.summary),
                        "completion_reason": run_completion.reason,
                        "status": "collection_completed",
                    }
                if tracker is not None and pending_summary is not None:
                    tracker.update_summary(pending_summary)
                training_succeeded = True
        finally:
            # Keep non-main ranks alive until rank 0 has persisted its final
            # summary, then release NCCL before playback.
            finish_rsl_rl_distributed(training_succeeded=training_succeeded)

        if (
            run_completion is None
            and rank == 0
            and should_run_playback(
                play_only=cfg.training.play_only,
                no_play=cfg.training.no_play,
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            )
        ):
            # torchrun rank variables outlive the training process group. Mask
            # them while playback constructs a new runner so rank 0 does not
            # initialize a second NCCL group after sibling workers have exited.
            with rsl_rl_single_process_topology():
                play_video_path = play_ppo(cfg, device)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
