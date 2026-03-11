from ml_collections import config_dict


def ppo_config(env_name: str) -> config_dict.ConfigDict:
    """Return unified PPO config for RSL-RL.

    Common keys are aligned across tasks to make training infra reusable.
    Task-specific options are configured per environment.
    """
    cfg = config_dict.create(
        seed=1,
        num_envs=4096,
        num_steps_per_env=24,
        max_iterations=101,
        save_interval=50,
        learning_rate=3.0e-4,
        entropy_coef=0.001,
        schedule="fixed",
        value_loss_coef=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        empirical_normalization=True,
    )

    if env_name == "Go1JoystickFlatTerrain":
        cfg.num_envs = 4096
        cfg.entropy_coef = 0.01
        cfg.learning_rate = 1.0e-3
        cfg.schedule = "adaptive"
        cfg.save_interval = 100
        cfg.max_iterations = 151
        cfg.empirical_normalization = False
    elif env_name == "G1JoystickFlatTerrain":
        cfg.num_envs = 2048
        cfg.entropy_coef = 0.01
        cfg.learning_rate = 1.0e-3
        cfg.schedule = "adaptive"
        cfg.save_interval = 50
        cfg.max_iterations = 220
        cfg.empirical_normalization = False
    elif env_name == "Go2JoystickFlatTerrain":
        cfg.num_envs = 4096
        cfg.entropy_coef = 0.01
        cfg.learning_rate = 1.0e-3
        cfg.schedule = "adaptive"
        cfg.save_interval = 100
        cfg.max_iterations = 101
        cfg.empirical_normalization = False

    return config_dict.create(
        algo="ppo",
        algo_log_name="rsl_rl_ppo",
        seed=cfg.seed,
        num_envs=cfg.num_envs,
        num_steps_per_env=cfg.num_steps_per_env,
        max_iterations=cfg.max_iterations,
        save_interval=cfg.save_interval,
        empirical_normalization=cfg.empirical_normalization,
        runner_class_name="OnPolicyRunner",
        obs_groups={"default": ["policy"]},
        experiment_name="test",
        run_name="",
        resume=False,
        load_run="-1",
        checkpoint=-1,
        resume_path=None,
        policy=config_dict.create(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            class_name="ActorCritic",
        ),
        algorithm=config_dict.create(
            class_name="PPO",
            value_loss_coef=cfg.value_loss_coef,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=cfg.entropy_coef,
            num_learning_epochs=cfg.num_learning_epochs,
            num_mini_batches=cfg.num_mini_batches,
            learning_rate=cfg.learning_rate,
            schedule=cfg.schedule,
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            target_kl_stop=None,
            max_grad_norm=1.0,
            adaptive_kl_beta=0.9,
            adaptive_lr_growth=1.1,
            adaptive_lr_decay=1.2,
            adaptive_lr_update_interval=5,
            fast_mode=True,
            metrics_interval=8,
            finite_check_interval=8,
            enable_compile=False,
            warmup_strict_iters=10,
            warmup_metrics_interval=2,
            warmup_finite_check_interval=2,
            disable_finite_checks=True,
        ),
    )


def offpolicy_config(algo: str, env_name: str) -> config_dict.ConfigDict:
    """Return a unified off-policy config schema for SAC/TD3.

    Common keys are aligned across algorithms to make training infra reusable.
    Algo-specific options are stored under ``algo_params``.
    """
    algo_name = algo.lower()
    if algo_name == "sac":
        cfg = config_dict.create(
            seed=1,
            actor_hidden_dim=512,
            critic_hidden_dim=768,
            use_layer_norm=True,
            num_atoms=101,
            num_envs=4096,
            batch_size=8192,
            updates_per_step=4,
            warmup_steps=1000,
            replay_buffer_n=512,
            env_steps_per_sync=1,
            max_iterations=1500,
            save_interval=500,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.97,
            tau=0.125,
            alpha_init=0.01,
            target_entropy_ratio=0.0,
            obs_normalization=True,
            policy_frequency=4,
            max_grad_norm=0.0,
        )

        if env_name in ("Go2JoystickFlatTerrain", "Go2LocoFlatTerrain"):
            cfg.gamma = 0.99
            cfg.num_envs = 1024
            cfg.max_iterations = 1500
        elif env_name in ("Go1JoystickFlatTerrain",):
            cfg.num_envs = 4096
            cfg.max_iterations = 2000
        elif env_name in ("G1JoystickFlatTerrain",):
            raise NotImplementedError("G1JoystickFlatTerrain config is not implemented for FastSAC, Please use G1JoystickFlatTerrainSAC instead.")
        elif env_name in ("G1JoystickFlatTerrainSAC",):
            cfg.updates_per_step = 8
            cfg.replay_buffer_n = 1024
            cfg.warmup_steps = 0
            cfg.alpha_init = 0.001
            cfg.max_iterations = 25000
            cfg.save_interval = 1000

        return config_dict.create(
            algo="sac",
            algo_log_name="fast_sac",
            seed=cfg.seed,
            num_envs=cfg.num_envs,
            batch_size=cfg.batch_size,
            replay_buffer_n=cfg.replay_buffer_n,
            updates_per_step=cfg.updates_per_step,
            warmup_steps=cfg.warmup_steps,
            policy_frequency=cfg.policy_frequency,
            env_steps_per_sync=cfg.env_steps_per_sync,
            max_iterations=cfg.max_iterations,
            save_interval=cfg.save_interval,
            gamma=cfg.gamma,
            tau=cfg.tau,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            actor_hidden_dim=cfg.actor_hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim,
            num_atoms=cfg.num_atoms,
            obs_normalization=cfg.obs_normalization,
            use_layer_norm=cfg.use_layer_norm,
            algo_params=config_dict.create(
                alpha_lr=cfg.alpha_lr,
                alpha_init=cfg.alpha_init,
                target_entropy_ratio=cfg.target_entropy_ratio,
                max_grad_norm=cfg.max_grad_norm,
            ),
        )

    if algo_name == "td3":
        cfg = config_dict.create(
            seed=1,
            actor_hidden_dim=256,
            critic_hidden_dim=512,
            num_atoms=101,
            init_scale=0.01,
            num_envs=4096,
            batch_size=8192,
            num_updates=4,
            warmup_steps=100,
            buffer_size=1000,
            max_iterations=5000,
            save_interval=500,
            actor_lr=3e-4,
            critic_lr=3e-4,
            weight_decay=0.1,
            gamma=0.97,
            tau=0.1,
            policy_frequency=2,
            policy_noise=0.2,
            noise_clip=0.5,
            log_std_min=-0.9,
            log_std_max=0.0,
            v_min=-10.0,
            v_max=10.0,
            use_cdq=True,
            obs_normalization=True,
        )

        if env_name in ("Go2JoystickFlatTerrain", "Go2LocoFlatTerrain"):
            cfg.max_iterations = 2000
        elif env_name in ("G1JoystickFlatTerrain",):
            cfg.num_envs = 2048

        return config_dict.create(
            algo="td3",
            algo_log_name="fast_td3",
            seed=cfg.seed,
            num_envs=cfg.num_envs,
            batch_size=cfg.batch_size,
            replay_buffer_n=cfg.buffer_size,
            updates_per_step=cfg.num_updates,
            warmup_steps=cfg.warmup_steps,
            policy_frequency=cfg.policy_frequency,
            env_steps_per_sync=1,
            max_iterations=cfg.max_iterations,
            save_interval=cfg.save_interval,
            gamma=cfg.gamma,
            tau=cfg.tau,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            actor_hidden_dim=cfg.actor_hidden_dim,
            critic_hidden_dim=cfg.critic_hidden_dim,
            num_atoms=cfg.num_atoms,
            obs_normalization=cfg.obs_normalization,
            use_layer_norm=False,
            algo_params=config_dict.create(
                weight_decay=cfg.weight_decay,
                v_min=cfg.v_min,
                v_max=cfg.v_max,
                init_scale=cfg.init_scale,
                log_std_min=cfg.log_std_min,
                log_std_max=cfg.log_std_max,
                policy_noise=cfg.policy_noise,
                noise_clip=cfg.noise_clip,
                use_cdq=cfg.use_cdq,
            ),
        )

    raise ValueError(f"Unsupported off-policy algo: {algo}")