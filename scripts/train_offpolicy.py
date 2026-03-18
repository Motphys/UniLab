"""Unified off-policy training entry for SAC and TD3."""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

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


def resolve_checkpoint_path(
    root_dir: Path, algo_log_name: str, task: str, load_run: str
) -> tuple[str | None, str | None]:
    """Resolve latest or explicit checkpoint path for play mode."""
    base_log_dir = os.path.join(root_dir, "logs", algo_log_name, task)

    load_path = None
    load_path_dir = None
    if load_run == "-1":
        if os.path.exists(base_log_dir):
            all_runs = sorted(
                [
                    d
                    for d in os.listdir(base_log_dir)
                    if os.path.isdir(os.path.join(base_log_dir, d))
                ]
            )
            if all_runs:
                latest_run_dir = os.path.join(base_log_dir, all_runs[-1])
                model_files = sorted(
                    [
                        f
                        for f in os.listdir(latest_run_dir)
                        if f.startswith("model_") and f.endswith(".pt")
                    ],
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
                [
                    f
                    for f in os.listdir(potential_dir)
                    if f.startswith("model_") and f.endswith(".pt")
                ],
                key=lambda x: int(x.split("_")[1].split(".")[0]),
            )
            if model_files:
                load_path = os.path.join(potential_dir, model_files[-1])
                load_path_dir = potential_dir

    return load_path, load_path_dir


def build_runner(algo_name: str, cfg: DictConfig):
    """Build algorithm runner from unified Hydra config."""
    from unilab.utils.reward_utils import extract_reward_config

    env_cfg_override = extract_reward_config(cfg)

    if algo_name == "sac":
        from unilab.algos.torch.fast_sac.learner import FastSACLearner
        from unilab.algos.torch.fast_sac.runner import FastSACRunner
        from unilab.utils.device_utils import get_default_device, get_env_dims

        # Multi-GPU path
        if cfg.training.num_gpus > 1:
            from unilab.algos.torch.offpolicy.multi_gpu_runner import MultiGPUOffPolicyRunner
            from unilab.base import registry
            from unilab.utils.algo_utils import ensure_registries

            ensure_registries()
            device = cfg.training.device or get_default_device()
            env = registry.make(
                cfg.training.task_name, num_envs=1, sim_backend=cfg.training.sim_backend
            )
            assert env.action_space.shape
            from unilab.utils.obs_utils import get_obs_dims
            obs_dim, privileged_dim = get_obs_dims(env.obs_groups_spec)
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
                "privileged_dim": privileged_dim,
                # symmetry not supported in multi-GPU mode (mujoco_model not picklable)
                "use_symmetry": False,
            }
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
                warmup_steps=cfg.algo.warmup_steps,
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
            warmup_steps=cfg.algo.warmup_steps,
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
            warmup_steps=cfg.algo.warmup_steps,
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

    raise ValueError(f"Unsupported algo: {algo_name}")


def play_offpolicy(algo_name: str, cfg: DictConfig) -> None:
    """Play pipeline for off-policy algorithms."""
    import mediapy as media
    import numpy as np
    import torch

    from unilab.base import registry
    from unilab.utils import render_many
    from unilab.utils.algo_utils import build_actor

    from unilab.utils.reward_utils import extract_reward_config

    env_cfg_override = extract_reward_config(cfg)

    device = default_device(torch, cfg.training.device)
    print(f"Using device for play: {device}")

    from unilab.utils.obs_utils import flatten_obs_dict

    env = registry.make(
        cfg.training.task_name,
        num_envs=cfg.training.play_env_num,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    obs_dim = sum(env.obs_groups_spec.values())
    action_dim = env.action_space.shape[0]

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
        from unilab.algos.torch.fast_td3.learner import EmpiricalNormalization, TD3Actor

        actor = TD3Actor(
            obs_dim,
            action_dim,
            cfg.training.play_env_num,
            cfg.algo.algo_params.init_scale,
            cfg.algo.actor_hidden_dim,
            cfg.algo.algo_params.log_std_min,
            cfg.algo.algo_params.log_std_max,
            device,
        )
        if cfg.algo.obs_normalization:
            normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    else:
        raise ValueError(f"Unsupported algo: {algo_name}")

    actor.eval()

    load_path, load_path_dir = resolve_checkpoint_path(
        ROOT_DIR,
        cfg.algo.algo_log_name,
        cfg.training.task_name,
        cfg.training.load_run,
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
        if normalizer and checkpoint.get("obs_normalizer"):
            normalizer.load_state_dict(checkpoint["obs_normalizer"])
            normalizer.eval()

    output_video = os.path.join(load_path_dir, "play_video.mp4")

    if env.state is None:
        env.init_state()
    env_indices = np.arange(cfg.training.play_env_num, dtype=np.int32)
    _, obs_out, _ = env.reset(env_indices)
    obs_np = np.asarray(flatten_obs_dict(obs_out), dtype=np.float32)

    # Use Motrix native rendering
    if cfg.training.sim_backend == "motrix":
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
                    actions_np = (
                        actor.explore(obs_torch, deterministic=True).cpu().numpy()
                        if algo_name == "sac"
                        else actor(obs_torch).cpu().numpy()
                    )
                    state = env.step(actions_np)
                    obs_np = np.asarray(flatten_obs_dict(state.obs), dtype=np.float32)

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
    state_list = []

    print("Collecting physics states...")
    with torch.inference_mode():
        for _ in range(cfg.training.play_steps):
            obs_torch = torch.from_numpy(obs_np).to(device)
            if normalizer:
                obs_torch = normalizer(obs_torch, update=False)
            actions_np = (
                actor.explore(obs_torch, deterministic=True).cpu().numpy()
                if algo_name == "sac"
                else actor(obs_torch).cpu().numpy()
            )
            state = env.step(actions_np)
            obs_np = np.asarray(flatten_obs_dict(state.obs), dtype=np.float32)
            state_list.append(np.asarray(env._backend.get_physics_state(), dtype=np.float32).copy())

    print("Rendering frames...")
    frames = render_many.render_states_get_frames(
        state_list, env.cfg.model_file, width=1280, height=720, camera_id=-1
    )
    print(f"Saving video to {output_video} ...")
    media.write_video(str(output_video), frames, fps=int(1.0 / env.cfg.ctrl_dt))
    print("Done.")


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    algo_name = cfg.algo.algo
    task_name = cfg.training.task_name

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(
            ROOT_DIR,
            "logs",
            cfg.algo.algo_log_name,
            task_name,
            f"{timestamp}_{cfg.training.sim_backend}",
        )
    else:
        log_dir = cfg.training.log_dir

    if not cfg.training.play_only:
        runner = build_runner(algo_name, cfg)
        try:
            runner.learn(
                max_iterations=cfg.algo.max_iterations,
                save_interval=cfg.algo.save_interval,
                log_dir=log_dir,
                logger_type=cfg.training.logger,
            )
        finally:
            runner.close()

    if cfg.training.play_only or not cfg.training.no_play:
        play_offpolicy(algo_name, cfg)


if __name__ == "__main__":
    main()
