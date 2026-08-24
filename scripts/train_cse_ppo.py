"""Train or play the CSE-PPO A2Arm policy."""

from __future__ import annotations

import datetime
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.cse_ppo import CSEOnPolicyRunner
from unilab.algos.rsl_rl import RslRlVecEnvWrapper, get_policy_obs_dims
from unilab.base.backend import materialize_scene_visual_override
from unilab.base.config_adapter import BackendAdapter, create_env
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
    make_sim2sim_preflight,
    normalize_checkpoint_value,
)

EXPORT_POLICY = False


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    """Compatibility helper matching the historical CSE entrypoint surface."""
    return algo_config_dict(cfg)


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo_cse",
        scene_materializer=materialize_scene_visual_override,
    )


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(ROOT_DIR, cfg))


def _format_play_checkpoint_error(
    cfg: DictConfig,
    *,
    task_log_root: Path,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
    """Historical CSE helper retained while delegating to shared diagnostics."""
    return format_play_checkpoint_error(
        cfg,
        task_log_root=task_log_root,
        load_path=load_path,
        load_path_dir=load_path_dir,
    )


def play_cse_ppo(cfg: DictConfig, device: str) -> str | None:
    """Resolve a checkpoint and render a CSE policy using the shared playback session."""
    rl_cfg = _algo_config_dict(cfg)
    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            _format_play_checkpoint_error(
                cfg, task_log_root=task_log_root, load_path=load_path, load_path_dir=load_path_dir
            )
        )
        return None
    keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in keys:
        print(
            f"Checkpoint at {load_path} is not a CSE-PPO checkpoint (found keys: {keys}). Aborting play."
        )
        return None

    def env_factory(num_envs: int):
        override = cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())
        return create_env(cfg, num_envs=num_envs, env_cfg_override=override)

    session, _mode, _checkpoint = create_rsl_rl_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task=str(cfg.training.task_name),
            load_run=str(cfg.algo.load_run),
            checkpoint=normalize_checkpoint_value(
                OmegaConf.select(cfg, "algo.checkpoint", default=None)
            ),
            action_mode="policy",
            policy_obs_mode="flat",
            algo_log_name=str(cfg.algo.algo_log_name),
            log_root=getattr(cfg.training, "log_root", None),
            num_envs=int(cfg.training.play_env_num),
        ),
        env_factory=env_factory,
        algo_config=rl_cfg,
        root_dir=ROOT_DIR,
        device=device,
        checkpoint_resolver=lambda *_args: str(load_path),
        # The first CSE actor layer consumes ``one_step_obs + latent`` rather
        # than the flattened history, so its state-dict width is not the
        # environment policy-input width used by the generic RSL guard.
        checkpoint_input_dim_reader=lambda _path: None,
        entrypoint_log_root=get_entrypoint_log_root,
        wrapper_cls=RslRlVecEnvWrapper,
        runner_cls=CSEOnPolicyRunner,
        runner_loader=lambda runner, path: runner.load(path),
        policy_obs_dims_getter=get_policy_obs_dims,
        train_cfg_normalizer=lambda train_cfg: train_cfg,
        sim2sim_preflight=make_sim2sim_preflight(cfg, algo_name="ppo_cse"),
        guard_algo_name="ppo_cse",
    )
    env = session.env
    assert session.runner is not None and session.policy is not None
    cse_policy = session.policy
    session.policy = lambda obs: cse_policy(obs["actor"])
    if EXPORT_POLICY:
        session.runner.export_policy_to_jit(path=str(load_path_dir))
    output_video = Path(load_path_dir) / "play_video.mp4"
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
            extra_data_getter=(lambda: getattr(env, "curr_ee_goal_world", None))
            if hasattr(env, "curr_ee_goal_world")
            else None,
        )
    return str(output_video)


@hydra.main(version_base="1.3", config_path="../conf/ppo_cse", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    override = cast(dict[str, Any], _backend_adapter(cfg).build_task_env_cfg_override())
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    max_iterations = int(cfg.algo.max_iterations)
    if cfg.training.num_timesteps:
        max_iterations = max(
            1, int(cfg.training.num_timesteps / (cfg.algo.num_steps_per_env * cfg.algo.num_envs))
        )
    log_dir: str | None = None
    if not cfg.training.play_only:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = str(
            Path(_get_log_root(cfg))
            / str(cfg.training.task_name)
            / build_run_dir_name(timestamp, str(cfg.training.sim_backend))
        )
    tracker = None
    if log_dir is not None:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name="ppo_cse",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=device,
        )
        tracker.start()
    try:
        if not cfg.training.play_only:
            env = create_env(cfg, num_envs=cfg.algo.num_envs, env_cfg_override=override)
            apply_env_nan_guard(env, cfg.training)
            runner = CSEOnPolicyRunner(
                RslRlVecEnvWrapper(env, device=device),
                _algo_config_dict(cfg),
                log_dir=log_dir,
                device=device,
            )
            if cfg.algo.load_run != "-1":
                resume_path, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
                if resume_path:
                    runner.load(str(resume_path))
            started = time.time()
            runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
            assert log_dir is not None
            if tracker is not None:
                tracker.update_summary(
                    {
                        "status": "completed",
                        "completed_iterations": int(runner.current_learning_iteration),
                        "total_env_steps": int(runner.logger.tot_timesteps),
                        "final_mean_reward": float(statistics.mean(runner.logger.rewbuffer))
                        if runner.logger.rewbuffer
                        else None,
                        "best_mean_reward": float(max(runner.logger.rewbuffer))
                        if runner.logger.rewbuffer
                        else None,
                        "mean_episode_length": float(statistics.mean(runner.logger.lenbuffer))
                        if runner.logger.lenbuffer
                        else None,
                        "last_checkpoint": str(
                            Path(log_dir) / f"model_{runner.current_learning_iteration}.pt"
                        ),
                        "training_wall_time_sec": time.time() - started,
                    }
                )
            env.close()
        if cfg.training.play_only or not cfg.training.no_play:
            output = play_cse_ppo(cfg, device)
            if tracker is not None:
                tracker.log_video(output)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
