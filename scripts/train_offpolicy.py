"""Unified off-policy training entry for SAC and TD3."""

from __future__ import annotations

import os
import sys
import datetime
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.utils.algo_utils import ensure_registries


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


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="Train off-policy algorithm (SAC/TD3)")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "td3"])
    parser.add_argument("--task", type=str, default="Go1JoystickFlatTerrain")
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
    parser.add_argument("--play_steps", type=int, default=200, help="Number of steps for play video")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none", "no_print"])
    parser.add_argument("--sim_backend", type=str, default="mujoco", choices=["mujoco", "motrix"], help="Simulation backend")
    parser.add_argument("--cam_distance", type=float, default=6.0, help="Camera distance for play video")
    parser.add_argument("--cam_elevation", type=float, default=-20.0, help="Camera elevation angle (degrees) for play video")
    parser.add_argument("--cam_azimuth", type=float, default=90.0, help="Camera azimuth angle (degrees) for play video")
    return parser

def build_runner(algo_name: str, args, cfg):
    """Build algorithm runner from unified config."""
    collector_device = args.collector_device or "cpu"
    if collector_device == "gpu":
        import torch
        collector_device = "mps" if torch.backends.mps.is_available() else "cuda"

    if algo_name == "sac":
        from unilab.algos.torch.fast_sac.runner import FastSACRunner
        return FastSACRunner(
            env_name=args.task, device=args.device, collector_device=collector_device,
            num_envs=cfg.num_envs, replay_buffer_n=cfg.replay_buffer_n, batch_size=cfg.batch_size,
            warmup_steps=cfg.warmup_steps, updates_per_step=cfg.updates_per_step,
            policy_frequency=cfg.policy_frequency, gamma=cfg.gamma, tau=cfg.tau,
            actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr, alpha_lr=cfg.algo_params.alpha_lr,
            alpha_init=cfg.algo_params.alpha_init, target_entropy_ratio=cfg.algo_params.target_entropy_ratio,
            obs_normalization=cfg.obs_normalization, actor_hidden_dim=cfg.actor_hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim, num_atoms=cfg.num_atoms,
            use_layer_norm=cfg.use_layer_norm, max_grad_norm=cfg.algo_params.max_grad_norm,
            sync_collection=not args.no_sync_collection, env_steps_per_sync=cfg.env_steps_per_sync,
            sim_backend=args.sim_backend,
        )

    if algo_name == "td3":
        from unilab.algos.torch.fast_td3.runner import FastTD3Runner
        return FastTD3Runner(
            env_name=args.task, device=args.device, collector_device=collector_device,
            num_envs=cfg.num_envs, replay_buffer_n=cfg.replay_buffer_n, batch_size=cfg.batch_size,
            warmup_steps=cfg.warmup_steps, num_updates=cfg.updates_per_step,
            policy_frequency=cfg.policy_frequency, sync_collection=not args.no_sync_collection,
            env_steps_per_sync=cfg.env_steps_per_sync, gamma=cfg.gamma, tau=cfg.tau,
            actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr, actor_hidden_dim=cfg.actor_hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim, num_atoms=cfg.num_atoms,
            v_min=cfg.algo_params.v_min, v_max=cfg.algo_params.v_max,
            init_scale=cfg.algo_params.init_scale, log_std_min=cfg.algo_params.log_std_min,
            log_std_max=cfg.algo_params.log_std_max, policy_noise=cfg.algo_params.policy_noise,
            noise_clip=cfg.algo_params.noise_clip, weight_decay=cfg.algo_params.weight_decay,
            use_cdq=cfg.algo_params.use_cdq, obs_normalization=cfg.obs_normalization,
            sim_backend=args.sim_backend,
        )

    raise ValueError(f"Unsupported algo: {algo_name}")


def play_offpolicy(algo_name: str, args, cfg) -> None:
    """Play pipeline for off-policy algorithms."""
    import mediapy as media
    import numpy as np
    import torch
    from unilab.envs import registry
    from unilab.utils import render_many
    from unilab.utils.algo_utils import build_actor

    device = default_device(torch, args.device)
    print(f"Using device for play: {device}")

    env = registry.make(args.task, num_envs=args.play_env_num, sim_backend=args.sim_backend)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    normalizer = None
    if algo_name == "sac":
        actor = build_actor("sac", obs_dim, action_dim, cfg.actor_hidden_dim, cfg.use_layer_norm, device)
    elif algo_name == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor, EmpiricalNormalization
        actor = TD3Actor(obs_dim, action_dim, args.play_env_num, cfg.algo_params.init_scale,
                        cfg.actor_hidden_dim, cfg.algo_params.log_std_min, cfg.algo_params.log_std_max, device)
        if cfg.obs_normalization:
            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")

    actor.eval()

    load_path, load_path_dir = resolve_checkpoint_path(ROOT_DIR, cfg.algo_log_name, args.task, args.load_run)
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
        if normalizer and checkpoint.get("obs_normalizer"):
            normalizer.load_state_dict(checkpoint["obs_normalizer"])
            normalizer.eval()

    output_video = os.path.join(load_path_dir, "play_video.mp4")
    print(f"Rendering video to {output_video}...")

    if env.state is None:
        env.init_state()
    env_indices = np.arange(args.play_env_num, dtype=np.int32)
    _, obs_out, _ = env.reset(env_indices)
    obs_np = np.asarray(obs_out, dtype=np.float32)

    # Use Motrix native rendering
    if args.sim_backend == "motrix":
        print("Starting interactive visualization (motrix native renderer)...")
        print("Close the render window to exit.")
        env._backend.init_renderer()

        import time
        last_render_time = time.perf_counter()
        render_dt = 1.0 / 60.0

        with torch.inference_mode():
            try:
                while True:
                    obs_torch = torch.from_numpy(obs_np).to(device)
                    if normalizer:
                        obs_torch = normalizer(obs_torch, update=False)
                    actions_np = actor.explore(obs_torch, deterministic=True).cpu().numpy() if algo_name == "sac" else actor(obs_torch).cpu().numpy()
                    state = env.step(actions_np)
                    obs_np = np.asarray(state.obs, dtype=np.float32)

                    current_time = time.perf_counter()
                    elapsed = current_time - last_render_time
                    if elapsed < render_dt:
                        time.sleep(render_dt - elapsed)
                    last_render_time = time.perf_counter()

                    env._backend.render()
            except Exception as e:
                if "RenderClosedError" in str(type(e).__name__):
                    print("Render window closed.")
                else:
                    raise
        return

    # MuJoCo backend: render to video
    output_video = os.path.join(load_path_dir, "play_video.mp4")
    print(f"Rendering video to {output_video}...")

    state_list = []

    print("Collecting physics states...")
    with torch.inference_mode():
        for _ in range(args.play_steps):
            obs_torch = torch.from_numpy(obs_np).to(device)
            if normalizer:
                obs_torch = normalizer(obs_torch, update=False)
            actions_np = actor.explore(obs_torch, deterministic=True).cpu().numpy() if algo_name == "sac" else actor(obs_torch).cpu().numpy()
            state = env.step(actions_np)
            obs_np = np.asarray(state.obs, dtype=np.float32)
            state_list.append(np.asarray(env._backend.get_physics_state(), dtype=np.float32).copy())

    print("Rendering frames...")
    frames = render_many.render_states_get_frames(
        state_list, env.cfg.model_file, width=1280, height=720, camera_id=-1,
        cam_distance=args.cam_distance, cam_elevation=args.cam_elevation, cam_azimuth=args.cam_azimuth,
    )
    print(f"Saving video to {output_video}...")
    media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
    print("Done.")


def main() -> None:
    ensure_registries()
    parser = build_parser()
    args = parser.parse_args()

    from unilab.config import locomotion_params, manipulation_params
    params = manipulation_params if args.task in manipulation_params.DEFAULT_ENV_NUM_BY_TASK else locomotion_params
    algo_name = args.algo.lower()
    cfg = params.offpolicy_config(algo_name, args.task)

    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    cfg.env_steps_per_sync = args.env_steps_per_sync

    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join(ROOT_DIR, "logs", cfg.algo_log_name, args.task, f"{timestamp}_{args.sim_backend}")

    if not args.play_only:
        runner = build_runner(algo_name, args, cfg)
        try:
            runner.learn(max_iterations=cfg.max_iterations, save_interval=cfg.save_interval,
                        log_dir=args.log_dir, logger_type=args.logger)
        finally:
            runner.close()

    if args.play_only or not args.no_play:
        play_offpolicy(algo_name, args, cfg)


if __name__ == "__main__":
    main()
