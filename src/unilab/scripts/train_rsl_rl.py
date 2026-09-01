import datetime
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.rsl_rl import (
    RslRlVecEnvWrapper,
    apply_rsl_rl_rank_seed,
    finish_rsl_rl_distributed,
    get_policy_obs_dims,
    normalize_ppo_train_cfg,
    ppo_samples_per_iteration,
    resolve_rsl_rl_device,
    rsl_rl_single_process_topology,
)
from unilab.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
from unilab.base.backend import RenderClosedError, materialize_scene_visual_override
from unilab.base.backend.base import log_playback_plan
from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.base.run_control import RunComplete
from unilab.ipc.dp_launcher import (
    UNILAB_DP_LOG_DIR,
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
    launch_torchrun_workers,
    resolve_dp_topology,
    validate_dp_launchable,
)
from unilab.training import (
    algo_config_dict,
    apply_env_nan_guard,
    build_run_dir_name,
    ensure_registries,
    format_play_checkpoint_error,
    get_log_root,
    parse_checkpoint_path,
    should_run_playback,
)
from unilab.training.experiment import (
    ExperimentTracker,
    patch_rsl_rl_action_std_logging,
    patch_rsl_rl_resume_state,
    patch_rsl_rl_wandb_writer,
)
from unilab.utils.checkpoint import get_entrypoint_log_root
from unilab.utils.device import get_default_device
from unilab.utils.seed import apply_configured_training_seed
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    create_rsl_rl_playback_session,
    infer_checkpoint_actor_input_dim,
    make_sim2sim_preflight,
    normalize_checkpoint_value,
)

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=Path.cwd(),
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    )


def build_ppo_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_task_env_cfg_override())


def build_ppo_play_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())


def run_motrix_rsl_play_loop(
    wrapped_env,
    policy,
    *,
    render_spacing: float,
    render_offset_mode: str,
    num_steps: int | None = None,
) -> None:
    env = wrapped_env.env

    with torch.inference_mode():
        env.run_playback(
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            num_steps=num_steps,
            initialize=lambda: wrapped_env.reset()[0],
            step=lambda obs: wrapped_env.step(policy(obs))[0],
        )


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(Path.cwd(), cfg))


def resolve_ppo_log_dir(
    cfg: DictConfig,
    *,
    world_size: int,
    timestamp: str | None = None,
) -> str:
    """Resolve one canonical run directory shared by all distributed ranks."""
    distributed_log_dir = os.environ.get(UNILAB_DP_LOG_DIR)
    if distributed_log_dir:
        return distributed_log_dir
    configured_log_dir = OmegaConf.select(cfg, "training.log_dir", default=None)
    if configured_log_dir:
        return str(configured_log_dir)
    timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return str(
        Path(_get_log_root(cfg))
        / str(cfg.training.task_name)
        / build_run_dir_name(
            timestamp,
            str(cfg.training.sim_backend),
            world_size=world_size,
        )
    )


def _resolve_ppo_wrapper_cls(rl_cfg: dict[str, Any]) -> type[RslRlVecEnvWrapper]:
    """Resolve the VecEnv wrapper class from the owner-selected PPO runtime.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.

    Returns:
        Wrapper class used to adapt the UniLab env contract to the active
        RSL-RL PPO runtime.
    """
    return resolve_rsl_rl_ppo_runtime(
        rl_cfg,
        default_wrapper_cls=RslRlVecEnvWrapper,
    ).wrapper_cls


def apply_ppo_runtime_flags(
    train_cfg: dict[str, Any],
    cfg: DictConfig,
    *,
    training_enabled: bool,
) -> None:
    algorithm_cfg = train_cfg.setdefault("algorithm", {})
    if not isinstance(algorithm_cfg, dict):
        return
    if not training_enabled:
        algorithm_cfg["enable_compile"] = False


def validate_ppo_run_completion_topology(
    cfg: DictConfig,
    *,
    devices: tuple[int, ...] | None,
    world_size: int,
) -> None:
    """Reject grasp collection on a topology without run-completion coordination."""
    if cfg.training.play_only:
        return
    if OmegaConf.select(cfg, "env.grasp_collection_target", default=None) is None:
        return
    effective_world_size = len(devices) if devices is not None and world_size == 1 else world_size
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


def play_rsl_rl(cfg: DictConfig, device: str) -> str | None:
    """Play mode for RSL-RL."""
    rl_cfg = algo_config_dict(cfg)

    task_log_root = get_log_root(Path.cwd(), cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=Path.cwd())
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

    def _normalize_play_train_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_ppo_train_cfg(train_cfg)
        apply_ppo_runtime_flags(normalized, cfg, training_enabled=False)
        return normalized

    playback_cfg = RslRlPlaybackConfig(
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
    session, _policy_obs_mode, _checkpoint_path = create_rsl_rl_playback_session(
        playback_cfg=playback_cfg,
        env_factory=lambda n: create_env(
            cfg,
            num_envs=n,
            env_cfg_override=build_ppo_play_env_cfg_override(cfg),
        ),
        algo_config=rl_cfg,
        root_dir=Path.cwd(),
        device=device,
        checkpoint_resolver=lambda *_args: str(load_path),
        checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
        entrypoint_log_root=get_entrypoint_log_root,
        wrapper_cls=_resolve_ppo_wrapper_cls(rl_cfg),
        runner_cls=OnPolicyRunner,
        policy_obs_dims_getter=get_policy_obs_dims,
        train_cfg_normalizer=_normalize_play_train_cfg,
        sim2sim_preflight=make_sim2sim_preflight(cfg, algo_name="ppo"),
        guard_algo_name="ppo",
    )
    env = session.env
    runner = session.runner
    if EXPORT_POLICY:
        # The checkpoint early-returns above guarantee a loaded runner here.
        assert runner is not None
        runner.export_policy_to_onnx(path=str(load_path_dir))
        runner.export_policy_to_jit(path=str(load_path_dir))
    num_steps = _resolve_play_num_steps(cfg)
    output_video = Path(load_path_dir) / "play_video.mp4"
    playback_mode: str | None = None
    play_video_path: str | None = None

    def _log_plan(plan) -> None:
        nonlocal playback_mode
        playback_mode = plan.mode
        log_playback_plan(plan)

    try:
        with torch.inference_mode():
            play_video_path = env.run_playback_mode(
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                play_steps=num_steps,
                output_video=output_video,
                render_spacing=float(
                    getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
                ),
                render_offset_mode=str(getattr(env.cfg, "render_offset_mode", "grid")),
                initialize=session.reset,
                step=lambda _obs: session.step_once(),
                camera_kwargs={
                    "cam_distance": cfg.training.cam_distance,
                    "cam_elevation": cfg.training.cam_elevation,
                    "cam_azimuth": cfg.training.cam_azimuth,
                    "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                    "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                    "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                    "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
                },
                on_plan=_log_plan,
                extra_data_getter=(
                    (lambda: getattr(env, "curr_ee_goal_world", None))
                    if hasattr(env, "curr_ee_goal_world")
                    else None
                ),
            )
    except RenderClosedError:
        # Interface-level signal: the user closed the backend render window.
        print("Render window closed.")
    if playback_mode != "none" and num_steps is not None:
        print("Done.")
    return play_video_path


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    devices = resolve_dp_topology(cfg.training.devices)
    rank = current_torch_distributed_rank()
    local_rank = current_torch_distributed_local_rank()
    world_size = current_torch_distributed_world_size()
    configured_device = OmegaConf.select(cfg, "training.device", default=None)

    if configured_device is not None and devices is not None:
        raise ValueError("Set either training.device or training.devices, not both")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK={rank} is out of range for WORLD_SIZE={world_size}")
    if cfg.training.play_only and world_size > 1:
        raise ValueError(
            "Distributed play-only execution is not supported; launch eval normally and "
            "select one device"
        )
    validate_ppo_run_completion_topology(cfg, devices=devices, world_size=world_size)

    # The parent only composes config and invokes torchrun. CUDA, registry,
    # env, tracker, and runner construction all happen inside workers.
    if devices is not None and len(devices) > 1 and world_size == 1:
        if not cfg.training.play_only:
            launch_torchrun_workers(
                devices,
                script_path=Path(__file__),
                argv=sys.argv[1:],
                log_dir=resolve_ppo_log_dir(cfg, world_size=len(devices)),
            )
            return
        validate_dp_launchable(devices)
    elif devices is not None and world_size == 1:
        validate_dp_launchable(devices)

    if (
        world_size > 1
        and os.environ.get(UNILAB_DP_LOG_DIR) is None
        and OmegaConf.select(cfg, "training.log_dir", default=None) is None
    ):
        raise ValueError(
            "Distributed RSL-RL workers require one shared run directory; use "
            "training.devices or set training.log_dir explicitly"
        )

    ensure_registries()
    apply_rsl_rl_rank_seed(cfg, rank)
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    env_cfg_override = build_ppo_env_cfg_override(cfg)

    device = resolve_rsl_rl_device(
        configured_device=str(configured_device) if configured_device is not None else None,
        devices=devices,
        world_size=world_size,
        local_rank=local_rank,
        default_device=get_default_device(),
    )
    print(f"[rank {rank}/{world_size}] Using device: {device}")

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
        log_dir: str | None = resolve_ppo_log_dir(cfg, world_size=world_size)
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
            seed_info=seed_info,
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
                    rl_cfg = algo_config_dict(cfg)
                    wrapper_cls = _resolve_ppo_wrapper_cls(rl_cfg)

                    apply_env_nan_guard(env, cfg.training)

                    wrapped_env = wrapper_cls(env, device=device)

                    train_cfg = normalize_ppo_train_cfg(rl_cfg)
                    apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=True)
                    if "runner" not in train_cfg:
                        train_cfg["runner"] = {}

                    logger_type = (
                        cfg.training.logger
                        if cfg.training.logger in ["tensorboard", "wandb"]
                        else "none"
                    )
                    train_cfg["runner"]["logger"] = logger_type
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

                    if cfg.algo.load_run != "-1":
                        resume_path, _ = parse_checkpoint_path(cfg, root_dir=Path.cwd())
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
                play_video_path = play_rsl_rl(cfg, device)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
