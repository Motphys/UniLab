import datetime
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig

EXPORT_POLICY = False  # set to True in __main__ block

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.him_ppo.runner import HIMOnPolicyRunner
from unilab.algos.rsl_rl import RslRlVecEnvWrapper, get_policy_obs_dims
from unilab.base.backend import materialize_scene_visual_override
from unilab.base.config_adapter import (
    BackendAdapter,
    create_env,
)
from unilab.training import (
    algo_config_dict,
    apply_env_nan_guard,
    build_run_dir_name,
    ensure_registries,
    format_play_checkpoint_error,
    get_log_root,
    parse_checkpoint_path,
)
from unilab.training.experiment import ExperimentTracker
from unilab.utils.checkpoint import get_entrypoint_log_root
from unilab.visualization import render_play_mode
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    create_rsl_rl_playback_session,
    infer_checkpoint_actor_input_dim,
    make_sim2sim_preflight,
    normalize_checkpoint_value,
)


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo_him",
        scene_materializer=materialize_scene_visual_override,
    )


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(ROOT_DIR, cfg))


def play_him_ppo(cfg: DictConfig, device: str) -> str | None:
    """Play mode for HIM-PPO."""
    rl_cfg = algo_config_dict(cfg)

    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
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
            f"Checkpoint at {load_path} is not a HIM-PPO checkpoint "
            f"(found keys: {_ckpt_keys}). Aborting play."
        )
        return None

    def _create_env(num_envs: int):
        env_cfg_override = cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())
        return create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override)

    session, _policy_obs_mode, _checkpoint_path = create_rsl_rl_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task=str(cfg.training.task_name),
            load_run=str(getattr(cfg.algo, "load_run", "-1")),
            checkpoint=normalize_checkpoint_value(getattr(cfg.algo, "checkpoint", None)),
            action_mode="policy",
            policy_obs_mode="flat",
            algo_log_name=str(cfg.algo.algo_log_name),
            log_root=getattr(cfg.training, "log_root", None),
            num_envs=int(cfg.training.play_env_num),
        ),
        env_factory=_create_env,
        algo_config=rl_cfg,
        root_dir=ROOT_DIR,
        device=device,
        # The checkpoint was already resolved above for the friendly early exit.
        checkpoint_resolver=lambda *_args: str(load_path),
        checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
        entrypoint_log_root=get_entrypoint_log_root,
        wrapper_cls=RslRlVecEnvWrapper,
        runner_cls=HIMOnPolicyRunner,
        # HIMOnPolicyRunner.load does not accept a load_cfg argument.
        runner_loader=lambda runner, path: runner.load(path),
        policy_obs_dims_getter=get_policy_obs_dims,
        train_cfg_normalizer=lambda train_cfg: train_cfg,
        sim2sim_preflight=make_sim2sim_preflight(cfg, algo_name="ppo"),
        guard_algo_name="him_ppo",
    )
    env = session.env
    assert session.runner is not None and session.policy is not None

    # HIM's inference policy consumes the flat actor tensor, not the full obs
    # TensorDict the session hands to ``policy``.
    him_policy = session.policy
    session.policy = lambda obs: him_policy(obs["actor"])

    if EXPORT_POLICY:
        session.runner.export_policy_to_onnx(path=str(load_path_dir))
        session.runner.export_policy_to_jit(path=str(load_path_dir))

    output_video = Path(load_path_dir) / "play_video.mp4"
    print(f"Rendering video to {output_video}...")
    print("Collecting physics states...")
    with torch.inference_mode():
        render_play_mode(
            env,
            sim_backend=cfg.training.sim_backend,
            render_spacing=float(
                getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
            ),
            num_steps=cfg.training.play_steps,
            output_video=output_video,
            initialize=lambda: session.reset()["actor"],
            step=lambda _obs: session.step_once()["actor"],
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
                "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
            },
            extra_data_getter=(
                (lambda: getattr(env, "curr_ee_goal_world", None))
                if hasattr(env, "curr_ee_goal_world")
                else None
            ),
        )
    print("Done.")
    return str(output_video)


@hydra.main(version_base="1.3", config_path="../conf/ppo_him", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    env_cfg_override = cast(dict[str, Any], _backend_adapter(cfg).build_task_env_cfg_override())

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Compute effective max_iterations
    max_iterations = cfg.algo.max_iterations
    if cfg.training.num_timesteps:
        n_steps_per_iter = cfg.algo.num_steps_per_env * cfg.algo.num_envs
        max_iterations = max(1, int(cfg.training.num_timesteps / n_steps_per_iter))
        print(
            f"Overriding max_iterations to {max_iterations} based on "
            f"num_timesteps {cfg.training.num_timesteps}"
        )

    if not cfg.training.play_only:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = _get_log_root(cfg)
        log_dir = str(
            Path(log_root)
            / cfg.training.task_name
            / build_run_dir_name(timestamp, str(cfg.training.sim_backend))
        )
    else:
        log_dir = None

    tracker = None
    if not cfg.training.play_only and log_dir is not None:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name="ppo_him",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=device,
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            env = create_env(cfg, num_envs=cfg.algo.num_envs, env_cfg_override=env_cfg_override)

            apply_env_nan_guard(env, cfg.training)

            wrapped_env = RslRlVecEnvWrapper(env, device=device)
            rl_cfg = algo_config_dict(cfg)
            runner = HIMOnPolicyRunner(wrapped_env, rl_cfg, log_dir=log_dir, device=device)

            if cfg.algo.load_run != "-1":
                resume_path, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
                if resume_path:
                    print(f"Resuming from {resume_path}")
                    runner.load(str(resume_path))

            train_start_wall = time.time()
            runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
            assert log_dir is not None
            train_summary = {
                "status": "completed",
                "completed_iterations": int(runner.current_learning_iteration),
                "total_env_steps": int(runner.logger.tot_timesteps),
                "final_mean_reward": (
                    float(statistics.mean(runner.logger.rewbuffer))
                    if len(runner.logger.rewbuffer) > 0
                    else None
                ),
                "best_mean_reward": (
                    float(max(runner.logger.rewbuffer))
                    if len(runner.logger.rewbuffer) > 0
                    else None
                ),
                "mean_episode_length": (
                    float(statistics.mean(runner.logger.lenbuffer))
                    if len(runner.logger.lenbuffer) > 0
                    else None
                ),
                "last_checkpoint": str(
                    Path(log_dir) / f"model_{int(runner.current_learning_iteration)}.pt"
                ),
                "training_wall_time_sec": time.time() - train_start_wall,
            }
            if tracker is not None:
                tracker.update_summary(train_summary)
            env.close()

        if cfg.training.play_only or not cfg.training.no_play:
            play_video_path = play_him_ppo(cfg, device)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
