"""Train FastTD3 agent — async multiprocessing training."""

import argparse
import sys
import os
import datetime
from pathlib import Path
import pkgutil
import importlib

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))


def ensure_registries():
    try:
        import unilab.envs.locomotion
        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


def play_td3(args, cfg):
    """Play mode for FastTD3."""
    import torch
    import numpy as np
    import mediapy as media
    from unilab.envs import registry
    from unilab.algos.torch.fast_td3.learner import TD3Actor, EmpiricalNormalization
    from unilab.utils import render_many

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for play: {device}")

    env = registry.make(args.task, num_envs=args.play_env_num, sim_backend="mujoco")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = TD3Actor(
        n_obs=obs_dim,
        n_act=action_dim,
        num_envs=args.play_env_num,
        init_scale=cfg.init_scale,
        hidden_dim=cfg.actor_hidden_dim,
        std_min=cfg.std_min,
        std_max=cfg.std_max,
        device=device,
    )
    actor.eval()

    # Load weights
    base_log_dir = os.path.join(ROOT_DIR, "logs", "fast_td3", args.task)
    load_path = None
    load_path_dir = None
    if args.load_run == "-1":
        if os.path.exists(base_log_dir):
            all_runs = sorted([d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))])
            if all_runs:
                latest_run_dir = os.path.join(base_log_dir, all_runs[-1])
                model_files = sorted(
                    [f for f in os.listdir(latest_run_dir) if f.startswith("model_") and f.endswith(".pt")],
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                )
                if model_files:
                    load_path = os.path.join(latest_run_dir, model_files[-1])
                    load_path_dir = latest_run_dir
    elif os.path.exists(args.load_run):
        load_path = args.load_run
        load_path_dir = os.path.dirname(load_path)
    else:
        potential_dir = os.path.join(base_log_dir, args.load_run)
        if os.path.isdir(potential_dir):
            model_files = sorted(
                [f for f in os.listdir(potential_dir) if f.startswith("model_") and f.endswith(".pt")],
                key=lambda x: int(x.split("_")[1].split(".")[0]),
            )
            if model_files:
                load_path = os.path.join(potential_dir, model_files[-1])
                load_path_dir = potential_dir

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find checkpoint. load_path={load_path}")
        return

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    actor_state = {k: v for k, v in checkpoint["actor"].items() if k not in ("noise_scales",)}
    actor.load_state_dict(actor_state, strict=False)

    if cfg.obs_normalization:
        normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        if "obs_normalizer" in checkpoint and checkpoint["obs_normalizer"] is not None:
            normalizer.load_state_dict(checkpoint["obs_normalizer"])
        normalizer.eval()
    else:
        normalizer = torch.nn.Identity()

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
            obs_torch = normalizer(obs_torch, update=False) if isinstance(normalizer, EmpiricalNormalization) else obs_torch
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


def main():
    parser = argparse.ArgumentParser(description="Train FastTD3")
    parser.add_argument("--task", type=str, default="Go2LocoFlatTerrain")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--collector_device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--no_sync_collection", action="store_true", help="Disable collection sync (async mode)")
    parser.add_argument("--env_steps_per_sync", type=int, default=1, help="Collector env.step calls to gather before each learner phase")
    parser.add_argument("--play_only", action="store_true", help="Skip training, only play")
    parser.add_argument("--no_play", action="store_true", help="Skip play after training")
    parser.add_argument("--load_run", type=str, default="-1", help="Run ID to load or path")
    parser.add_argument("--play_env_num", type=int, default=16, help="Number of play envs")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none", "no_print"])
    args = parser.parse_args()

    ensure_registries()

    from unilab.config.locomotion_params import fast_td3_config
    cfg = fast_td3_config(args.task)

    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs

    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join(ROOT_DIR, "logs", "fast_td3", args.task, timestamp)

    if not args.play_only:
        from unilab.algos.torch.fast_td3.runner import FastTD3Runner

        runner = FastTD3Runner(
            env_name=args.task,
            device=args.device,
            collector_device=args.collector_device,
            num_envs=cfg.num_envs,
            replay_buffer_n=cfg.buffer_size,
            batch_size=cfg.batch_size,
            warmup_steps=cfg.warmup_steps,
            num_updates=cfg.num_updates,
            policy_frequency=cfg.policy_frequency,
            sync_collection=not args.no_sync_collection,
            env_steps_per_sync=args.env_steps_per_sync,
            gamma=cfg.gamma,
            tau=cfg.tau,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            actor_hidden_dim=cfg.actor_hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim,
            num_atoms=cfg.num_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            init_scale=cfg.init_scale,
            std_min=cfg.std_min,
            std_max=cfg.std_max,
            policy_noise=cfg.policy_noise,
            noise_clip=cfg.noise_clip,
            weight_decay=cfg.weight_decay,
            use_cdq=cfg.use_cdq,
            obs_normalization=cfg.obs_normalization,
        )

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
        play_td3(args, cfg)


if __name__ == "__main__":
    main()

