"""Unified off-policy training entry for SAC and TD3."""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.training import (
    BackendAdapter,
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
    get_log_root,
    render_play_mode,
)
from unilab.training import (
    resolve_checkpoint_path as resolve_checkpoint_path_common,
)
from unilab.utils.experiment_tracking import ExperimentTracker


def default_device(torch_module, preferred: str | None = None) -> str:
    """Resolve runtime device with optional user override."""
    if preferred:
        return preferred
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint_path(
    root_dir: Path, algo_log_name: str, task: str, load_run: str
) -> tuple[str | None, str | None]:
    checkpoint_path, checkpoint_dir = resolve_checkpoint_path_common(
        Path(root_dir) / "logs" / algo_log_name / task,
        load_run,
        suffix=".pt",
    )
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def extract_reset_obs(reset_result):
    """Extract obs_dict from env.reset(...) using the current (obs_dict, info_dict) contract."""
    if isinstance(reset_result, tuple):
        if len(reset_result) == 2:
            obs_out, _ = reset_result
            return obs_out
    raise ValueError(f"Unexpected env.reset return format: {type(reset_result)!r}")


def resolve_play_obs_dim(obs_groups_spec: dict[str, int]) -> int:
    from unilab.utils.obs_utils import get_obs_dims

    obs_dim, _ = get_obs_dims(obs_groups_spec)
    return int(obs_dim)


def extract_play_obs(obs_dict):
    from unilab.utils.obs_utils import split_obs_dict

    obs_out, _ = split_obs_dict(obs_dict)
    return obs_out


def build_offpolicy_env_cfg_override(algo_name: str, cfg: DictConfig) -> dict[str, Any] | None:
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)
    return cast(
        dict[str, Any] | None,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name=algo_name).build_task_env_cfg_override(),
    )


def build_runner(algo_name: str, cfg: DictConfig):
    """Build algorithm runner from unified Hydra config."""
    env_cfg_override = build_offpolicy_env_cfg_override(algo_name, cfg)

    if algo_name == "flashsac" and cfg.training.num_gpus > 1:
        raise ValueError("FlashSAC does not support training.num_gpus > 1")

    if algo_name == "sac":
        from unilab.algos.torch.fast_sac.learner import FastSACLearner
        from unilab.algos.torch.fast_sac.runner import FastSACRunner
        from unilab.utils.device_utils import get_default_device, get_env_dims

        # Multi-GPU path
        if cfg.training.num_gpus > 1:
            from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner

            ensure_registries()
            device = cfg.training.device or get_default_device()
            env = create_env(
                cfg,
                num_envs=1,
                env_cfg_override=env_cfg_override,
            )
            assert env.action_space.shape
            from unilab.utils.obs_utils import get_obs_dims

            obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
            action_dim = env.action_space.shape[0]
            env.close()

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
                "critic_obs_dim": critic_dim,
                "use_symmetry": cfg.algo.use_symmetry,
            }
            MultiGPUOffPolicyRunner.validate_capabilities(
                algo_type="sac",
                learner_kwargs=learner_kwargs,
                num_gpus=cfg.training.num_gpus,
            )
            main_learner = FastSACLearner(device=device, **learner_kwargs)

            return MultiGPUOffPolicyRunner(
                learner=main_learner,
                env_name=cfg.training.task_name,
                algo_type="sac",
                learner_kwargs=learner_kwargs,
                num_gpus=cfg.training.num_gpus,
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
                use_layer_norm=cfg.algo.use_layer_norm,
                obs_normalization=False,
                sim_backend=cfg.training.sim_backend,
                env_cfg_override=env_cfg_override,
            )

        return FastSACRunner(
            env_name=cfg.training.task_name,
            env_cfg_override=env_cfg_override,
            device=cfg.training.device,
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=cfg.algo.batch_size,
            learning_starts=cfg.algo.learning_starts,
            updates_per_step=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            gamma=cfg.algo.gamma,
            tau=cfg.algo.tau,
            actor_lr=cfg.algo.actor_lr,
            critic_lr=cfg.algo.critic_lr,
            alpha_lr=cfg.algo.algo_params.alpha_lr,
            alpha_init=cfg.algo.algo_params.alpha_init,
            target_entropy_ratio=cfg.algo.algo_params.target_entropy_ratio,
            obs_normalization=cfg.algo.obs_normalization,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            critic_hidden_dim=cfg.algo.critic_hidden_dim,
            num_atoms=cfg.algo.num_atoms,
            use_layer_norm=cfg.algo.use_layer_norm,
            max_grad_norm=cfg.algo.algo_params.max_grad_norm,
            use_amp=cfg.training.use_amp,
            sync_collection=not cfg.training.no_sync_collection,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
            sim_backend=cfg.training.sim_backend,
            use_symmetry=cfg.algo.use_symmetry,
        )

    if algo_name == "td3":
        from unilab.algos.torch.fast_td3.runner import FastTD3Runner

        return FastTD3Runner(
            env_name=cfg.training.task_name,
            env_cfg_override=env_cfg_override,
            device=cfg.training.device,
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=cfg.algo.batch_size,
            learning_starts=cfg.algo.learning_starts,
            num_updates=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            sync_collection=not cfg.training.no_sync_collection,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
            gamma=cfg.algo.gamma,
            tau=cfg.algo.tau,
            actor_lr=cfg.algo.actor_lr,
            critic_lr=cfg.algo.critic_lr,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            critic_hidden_dim=cfg.algo.critic_hidden_dim,
            num_atoms=cfg.algo.num_atoms,
            v_min=cfg.algo.algo_params.v_min,
            v_max=cfg.algo.algo_params.v_max,
            init_scale=cfg.algo.algo_params.init_scale,
            log_std_min=cfg.algo.algo_params.log_std_min,
            log_std_max=cfg.algo.algo_params.log_std_max,
            policy_noise=cfg.algo.algo_params.policy_noise,
            noise_clip=cfg.algo.algo_params.noise_clip,
            weight_decay=cfg.algo.algo_params.weight_decay,
            use_cdq=cfg.algo.algo_params.use_cdq,
            obs_normalization=cfg.algo.obs_normalization,
            sim_backend=cfg.training.sim_backend,
        )

    if algo_name == "flashsac":
        from unilab.algos.torch.flash_sac.runner import FlashSACRunner

        return FlashSACRunner(
            env_name=cfg.training.task_name,
            env_cfg_override=env_cfg_override,
            device=cfg.training.device,
            num_envs=cfg.algo.num_envs,
            replay_buffer_n=cfg.algo.replay_buffer_n,
            batch_size=cfg.algo.batch_size,
            learning_starts=cfg.algo.learning_starts,
            updates_per_step=cfg.algo.updates_per_step,
            policy_frequency=cfg.algo.policy_frequency,
            gamma=cfg.algo.gamma,
            tau=cfg.algo.tau,
            actor_lr=cfg.algo.actor_lr,
            critic_lr=cfg.algo.critic_lr,
            obs_normalization=cfg.algo.obs_normalization,
            actor_hidden_dim=cfg.algo.actor_hidden_dim,
            critic_hidden_dim=cfg.algo.critic_hidden_dim,
            num_atoms=cfg.algo.num_atoms,
            use_amp=cfg.training.use_amp,
            sync_collection=not cfg.training.no_sync_collection,
            env_steps_per_sync=cfg.training.env_steps_per_sync,
            sim_backend=cfg.training.sim_backend,
            actor_num_blocks=cfg.algo.algo_params.actor_num_blocks,
            critic_num_blocks=cfg.algo.algo_params.critic_num_blocks,
            actor_bc_alpha=cfg.algo.algo_params.actor_bc_alpha,
            actor_noise_zeta_mu=cfg.algo.algo_params.actor_noise_zeta_mu,
            actor_noise_zeta_max=cfg.algo.algo_params.actor_noise_zeta_max,
            critic_min_v=cfg.algo.algo_params.critic_min_v,
            critic_max_v=cfg.algo.algo_params.critic_max_v,
            target_sigma=cfg.algo.algo_params.temp_target_sigma,
            target_entropy=cfg.algo.algo_params.temp_target_entropy,
            temp_initial_value=cfg.algo.algo_params.temp_initial_value,
            learning_rate_init=cfg.algo.algo_params.learning_rate_init,
            learning_rate_peak=cfg.algo.algo_params.learning_rate_peak,
            learning_rate_end=cfg.algo.algo_params.learning_rate_end,
            learning_rate_warmup_steps=cfg.algo.algo_params.learning_rate_warmup_steps,
            learning_rate_decay_steps=cfg.algo.algo_params.learning_rate_decay_steps,
            normalize_reward=cfg.algo.algo_params.normalize_reward,
            normalized_g_max=cfg.algo.algo_params.normalized_g_max,
            n_step=cfg.algo.algo_params.n_step,
            use_compile=cfg.algo.algo_params.use_compile,
        )

    raise ValueError(f"Unsupported algo: {algo_name}")


def play_offpolicy(algo_name: str, cfg: DictConfig) -> str | None:
    """Play pipeline for off-policy algorithms."""
    import numpy as np
    import torch

    from unilab.utils.algo_utils import build_actor

    env_cfg_override = build_offpolicy_env_cfg_override(algo_name, cfg)

    device = default_device(torch, cfg.training.device)
    print(f"Using device for play: {device}")

    env = cast(
        Any,
        create_env(
            cfg,
            num_envs=cfg.training.play_env_num,
            env_cfg_override=env_cfg_override,
        ),
    )
    obs_dim = resolve_play_obs_dim(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])

    normalizer = None
    if algo_name == "sac":
        actor = build_actor(
            "sac",
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device,
        )
    elif algo_name == "td3":
        import torch

        from unilab.algos.torch.fast_td3.learner import EmpiricalNormalization, TD3Actor

        actor = TD3Actor(
            obs_dim,
            action_dim,
            cfg.training.play_env_num,
            cfg.algo.algo_params.init_scale,
            cfg.algo.actor_hidden_dim,
            cfg.algo.algo_params.log_std_min,
            cfg.algo.algo_params.log_std_max,
            torch.device(device),
        )
        if cfg.algo.obs_normalization:
            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    elif algo_name == "flashsac":
        actor = build_actor(
            "flashsac",
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device,
            actor_num_blocks=cfg.algo.algo_params.actor_num_blocks,
            actor_noise_zeta_mu=cfg.algo.algo_params.actor_noise_zeta_mu,
            actor_noise_zeta_max=cfg.algo.algo_params.actor_noise_zeta_max,
        )
        if cfg.algo.obs_normalization:
            from unilab.algos.torch.common.normalization import EmpiricalNormalization

            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")

    actor.eval()

    load_path, load_path_dir = resolve_checkpoint_path(
        ROOT_DIR,
        cfg.algo.algo_log_name,
        cfg.training.task_name,
        cfg.algo.load_run,
    )
    if not load_path or not os.path.exists(load_path):
        print(f"Could not find checkpoint. load_path={load_path}")
        return None

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    if algo_name in ("sac", "flashsac"):
        actor.load_state_dict(checkpoint["actor"])
        if normalizer and checkpoint.get("obs_normalizer"):
            normalizer.load_state_dict(checkpoint["obs_normalizer"])
            normalizer.eval()
    else:
        actor_state = {k: v for k, v in checkpoint["actor"].items() if k not in ("noise_scales",)}
        actor.load_state_dict(actor_state, strict=False)
        if normalizer and checkpoint.get("obs_normalizer"):
            normalizer.load_state_dict(checkpoint["obs_normalizer"])
            normalizer.eval()

    if env.state is None:
        env.init_state()

    def _policy_step(obs_np: np.ndarray) -> np.ndarray:
        obs_torch = torch.from_numpy(obs_np).to(device)
        if normalizer:
            obs_torch = normalizer(obs_torch, update=False)
        actions_np = (
            actor.explore(obs_torch, deterministic=True).cpu().numpy()
            if algo_name in ("sac", "flashsac")
            else actor(obs_torch).cpu().numpy()
        )
        state = env.step(actions_np)
        return np.asarray(extract_play_obs(state.obs), dtype=np.float32)

    # Use Motrix native rendering
    if cfg.training.sim_backend == "motrix":
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")

        with torch.inference_mode():
            try:
                render_play_mode(
                    env,
                    sim_backend="motrix",
                    num_steps=None,
                    initialize=lambda: np.asarray(
                        extract_play_obs(
                            extract_reset_obs(
                                env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))
                            )
                        ),
                        dtype=np.float32,
                    ),
                    step=_policy_step,
                )
            except Exception as e:
                if "RenderClosedError" in str(type(e).__name__):
                    print("Render window closed.")
                else:
                    raise
        return None

    if load_path_dir is None:
        print(f"Could not resolve checkpoint directory. load_path_dir={load_path_dir}")
        return None

    output_video = os.path.join(load_path_dir, "play_video.mp4")
    print("Collecting physics states...")
    with torch.inference_mode():
        render_play_mode(
            env,
            sim_backend=cfg.training.sim_backend,
            num_steps=cfg.training.play_steps,
            output_video=output_video,
            initialize=lambda: np.asarray(
                extract_play_obs(
                    extract_reset_obs(
                        env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))
                    )
                ),
                dtype=np.float32,
            ),
            step=_policy_step,
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
            },
        )
    print(f"Saving video to {output_video} ...")
    print("Done.")
    return output_video


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    algo_name = cfg.algo.algo
    task_name = cfg.training.task_name
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = str(
            get_log_root(ROOT_DIR, cfg) / task_name / f"{timestamp}_{cfg.training.sim_backend}"
        )
    else:
        log_dir = cfg.training.log_dir

    import torch

    tracker = None
    if not cfg.training.play_only:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name=algo_name,
            task_name=task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=default_device(torch, cfg.training.device),
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            runner = build_runner(algo_name, cfg)
            try:
                runner.learn(
                    max_iterations=cfg.algo.max_iterations,
                    save_interval=cfg.algo.save_interval,
                    log_dir=log_dir,
                    logger_type=cfg.training.logger,
                )
                if tracker is not None:
                    tracker.update_summary(getattr(runner, "last_run_summary", None))
            finally:
                runner.close()

        if cfg.training.play_only or not cfg.training.no_play:
            print("@" * 50)
            play_video_path = play_offpolicy(algo_name, cfg)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
