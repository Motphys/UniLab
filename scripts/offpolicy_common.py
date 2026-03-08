"""Shared CLI utilities for off-policy training/play scripts."""

from __future__ import annotations

import os
import datetime
import argparse
from pathlib import Path

from unilab.algos.torch.common.utils import ensure_registries


def default_device(torch_module, preferred: str | None = None) -> str:
    """Resolve runtime device with optional user override."""
    if preferred:
        return preferred
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint_path(root_dir: Path, algo_log_name: str, task: str, load_run: str) -> tuple[str | None, str | None]:
    """Resolve latest or explicit checkpoint path for play mode."""
    base_log_dir = os.path.join(root_dir, "logs", algo_log_name, task)

    load_path = None
    load_path_dir = None
    if load_run == "-1":
        if os.path.exists(base_log_dir):
            all_runs = sorted(
                [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))]
            )
            if all_runs:
                latest_run_dir = os.path.join(base_log_dir, all_runs[-1])
                model_files = sorted(
                    [f for f in os.listdir(latest_run_dir) if f.startswith("model_") and f.endswith(".pt")],
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                )
                if model_files:
                    load_path = os.path.join(latest_run_dir, model_files[-1])
                    load_path_dir = latest_run_dir
    elif os.path.exists(load_run):
        load_path = load_run
        load_path_dir = os.path.dirname(load_path)
    else:
        potential_dir = os.path.join(base_log_dir, load_run)
        if os.path.isdir(potential_dir):
            model_files = sorted(
                [f for f in os.listdir(potential_dir) if f.startswith("model_") and f.endswith(".pt")],
                key=lambda x: int(x.split("_")[1].split(".")[0]),
            )
            if model_files:
                load_path = os.path.join(potential_dir, model_files[-1])
                load_path_dir = potential_dir

    return load_path, load_path_dir


def build_offpolicy_parser(
    *,
    description: str,
    default_task: str,
    include_algo: bool = False,
    default_algo: str = "sac",
) -> argparse.ArgumentParser:
    """Build a common parser for off-policy training scripts."""
    parser = argparse.ArgumentParser(description=description)
    if include_algo:
        parser.add_argument("--algo", type=str, default=default_algo, choices=["sac", "td3"])
    parser.add_argument("--task", type=str, default=default_task)
    parser.add_argument("--max_iterations", type=int, default=None, help="Override max iterations from config")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs from config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--collector_device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--no_sync_collection", action="store_true", help="Disable collection sync (async mode)")
    parser.add_argument("--env_steps_per_sync", type=int, default=1, help="Collector env.step calls before learner phase")
    parser.add_argument("--play_only", action="store_true", help="Skip training, only play")
    parser.add_argument("--no_play", action="store_true", help="Skip play after training")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load or checkpoint path")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none", "no_print"])
    return parser


def run_offpolicy(algo: str, args, root_dir: Path) -> None:
    """Shared train+play execution path for off-policy algorithms."""
    ensure_registries()

    from unilab.config.locomotion_params import offpolicy_config

    algo_name = algo.lower()
    cfg = offpolicy_config(algo_name, args.task)

    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    cfg.env_steps_per_sync = args.env_steps_per_sync

    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join(root_dir, "logs", cfg.algo_log_name, args.task, timestamp)

    if not args.play_only:
        runner = _build_runner(algo_name, args, cfg)
        try:
            runner.learn(
                max_iterations=cfg.max_iterations,
                save_interval=cfg.save_interval,
                log_dir=args.log_dir,
                logger_type=args.logger,
            )
        finally:
            runner.close()

    if args.play_only or not args.no_play:
        play_offpolicy(algo_name, args, cfg, root_dir)


def _build_runner(algo_name: str, args, cfg):
    """Build algorithm runner from unified config."""
    collector_device = args.collector_device or "cpu"

    if algo_name == "sac":
        from unilab.algos.torch.fast_sac.runner import FastSACRunner

        return FastSACRunner(
            env_name=args.task,
            device=args.device,
            collector_device=collector_device,
            num_envs=cfg.num_envs,
            replay_buffer_n=cfg.replay_buffer_n,
            batch_size=cfg.batch_size,
            warmup_steps=cfg.warmup_steps,
            updates_per_step=cfg.updates_per_step,
            policy_frequency=cfg.policy_frequency,
            gamma=cfg.gamma,
            tau=cfg.tau,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            alpha_lr=cfg.algo_params.alpha_lr,
            alpha_init=cfg.algo_params.alpha_init,
            target_entropy_ratio=cfg.algo_params.target_entropy_ratio,
            obs_normalization=cfg.obs_normalization,
            actor_hidden_dim=cfg.actor_hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim,
            num_atoms=cfg.num_atoms,
            use_layer_norm=cfg.use_layer_norm,
            max_grad_norm=cfg.algo_params.max_grad_norm,
            sync_collection=not args.no_sync_collection,
            env_steps_per_sync=cfg.env_steps_per_sync,
        )

    if algo_name == "td3":
        from unilab.algos.torch.fast_td3.runner import FastTD3Runner

        return FastTD3Runner(
            env_name=args.task,
            device=args.device,
            collector_device=collector_device,
            num_envs=cfg.num_envs,
            replay_buffer_n=cfg.replay_buffer_n,
            batch_size=cfg.batch_size,
            warmup_steps=cfg.warmup_steps,
            num_updates=cfg.updates_per_step,
            policy_frequency=cfg.policy_frequency,
            sync_collection=not args.no_sync_collection,
            env_steps_per_sync=cfg.env_steps_per_sync,
            gamma=cfg.gamma,
            tau=cfg.tau,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            actor_hidden_dim=cfg.actor_hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim,
            num_atoms=cfg.num_atoms,
            v_min=cfg.algo_params.v_min,
            v_max=cfg.algo_params.v_max,
            init_scale=cfg.algo_params.init_scale,
            std_min=cfg.algo_params.std_min,
            std_max=cfg.algo_params.std_max,
            policy_noise=cfg.algo_params.policy_noise,
            noise_clip=cfg.algo_params.noise_clip,
            weight_decay=cfg.algo_params.weight_decay,
            use_cdq=cfg.algo_params.use_cdq,
            obs_normalization=cfg.obs_normalization,
        )

    raise ValueError(f"Unsupported algo: {algo_name}")


def play_offpolicy(algo_name: str, args, cfg, root_dir: Path) -> None:
    """Shared play pipeline for off-policy algorithms."""
    import mediapy as media
    import numpy as np
    import torch
    from unilab.envs import registry
    from unilab.utils import render_many
    from unilab.algos.torch.common.utils import build_actor

    device = default_device(torch, args.device)
    print(f"Using device for play: {device}")

    env = registry.make(args.task, num_envs=args.play_env_num, sim_backend="mujoco")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    normalizer = None
    if algo_name == "sac":
        actor = build_actor(
            algo_type="sac",
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dim=cfg.actor_hidden_dim,
            use_layer_norm=cfg.use_layer_norm,
            device=device,
        )
    elif algo_name == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor, EmpiricalNormalization

        actor = TD3Actor(
            n_obs=obs_dim,
            n_act=action_dim,
            num_envs=args.play_env_num,
            init_scale=cfg.algo_params.init_scale,
            hidden_dim=cfg.actor_hidden_dim,
            std_min=cfg.algo_params.std_min,
            std_max=cfg.algo_params.std_max,
            device=device,
        )
        if cfg.obs_normalization:
            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")

    actor.eval()

    load_path, load_path_dir = resolve_checkpoint_path(
        root_dir,
        algo_log_name=cfg.algo_log_name,
        task=args.task,
        load_run=args.load_run,
    )
    if not load_path or not os.path.exists(load_path):
        print(f"Could not find checkpoint. load_path={load_path}")
        return

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    if algo_name == "sac":
        actor.load_state_dict(checkpoint["actor"])
    else:
        actor_state = {k: v for k, v in checkpoint["actor"].items() if k not in ("noise_scales",)}
        actor.load_state_dict(actor_state, strict=False)
        if normalizer is not None and checkpoint.get("obs_normalizer") is not None:
            normalizer.load_state_dict(checkpoint["obs_normalizer"])
            normalizer.eval()

    output_video = os.path.join(load_path_dir, "play_video.mp4")
    print(f"Rendering video to {output_video}...")

    if env.state is None:
        env.init_state()
    env_indices = np.arange(args.play_env_num, dtype=np.int32)
    _, obs_out, _ = env.reset(env_indices)
    obs_np = np.asarray(obs_out, dtype=np.float32)

    state_list = []
    num_steps = 150

    print("Collecting physics states...")
    with torch.inference_mode():
        for _ in range(num_steps):
            obs_torch = torch.from_numpy(obs_np).to(device)
            if normalizer is not None:
                obs_torch = normalizer(obs_torch, update=False)

            if algo_name == "sac":
                actions_np = actor.explore(obs_torch, deterministic=True).cpu().numpy()
            else:
                actions_np = actor(obs_torch).cpu().numpy()

            state = env.step(actions_np)
            obs_np = np.asarray(state.obs, dtype=np.float32)
            state_list.append(np.asarray(env.state.physics_state, dtype=np.float32).copy())

    print("Rendering frames...")
    frames = render_many.render_states_get_frames(
        state_list,
        env.cfg.model_file,
        width=1280,
        height=720,
        camera_id=-1,
    )

    print(f"Saving video to {output_video} with mediapy...")
    media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
    print("Done.")
