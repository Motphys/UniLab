from ml_collections import config_dict


DEFAULT_ENV_NUM_BY_TASK: dict[str, int] = {
    "AllegroInhandRotation": 16384,
}


def get_default_env_num(env_name: str) -> int:
    """Returns default number of parallel environments for a task."""
    return int(DEFAULT_ENV_NUM_BY_TASK.get(env_name, 4096))


def ppo_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned RSL-RL PPO config for the given environment."""

    rl_config = config_dict.create(
        seed=1,
        runner_class_name="OnPolicyRunner",
        obs_groups={"default": ["policy"]},
        policy=config_dict.create(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            class_name="ActorCritic",
        ),
        algorithm=config_dict.create(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
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
        num_steps_per_env=24,
        max_iterations=1500,
        empirical_normalization=False,
        save_interval=100,
        experiment_name="test",
        run_name="",
        resume=False,
        load_run="-1",
        checkpoint=-1,
        resume_path=None,
    )

    if env_name == "AllegroInhandRotation":
        rl_config.max_iterations = 501
        rl_config.num_steps_per_env = 8
        rl_config.algorithm.num_mini_batches = 4
        rl_config.algorithm.learning_rate = 1.0e-3
        rl_config.algorithm.schedule = "adaptive"
        rl_config.empirical_normalization = True
        rl_config.algorithm.entropy_coef = 0.01
        rl_config.algorithm.value_loss_coef = 4.0
        rl_config.algorithm.desired_kl = 0.02

    return rl_config


def offpolicy_config(algo: str, env_name: str) -> config_dict.ConfigDict:
    """Return a unified off-policy config schema for SAC/TD3.

    Common keys are aligned across algorithms to make training infra reusable.
    Algo-specific options are stored under ``algo_params``.
    """
    algo_name = algo.lower()
    if algo_name == "sac":
        cfg = config_dict.create(
            algo="sac",
            algo_log_name="fast_sac",
            seed=1,
            num_envs=4096,
            batch_size=8192,
            replay_buffer_n=512,
            updates_per_step=4,
            warmup_steps=1000,
            policy_frequency=4,
            env_steps_per_sync=1,
            max_iterations=1500,
            save_interval=500,
            gamma=0.97,
            tau=0.125,
            actor_lr=3e-4,
            critic_lr=3e-4,
            actor_hidden_dim=512, # To be check
            critic_hidden_dim=768, # To be check
            num_atoms=101,
            obs_normalization=True,
            use_layer_norm=True,
            algo_params=config_dict.create(
                alpha_lr=3e-4,
                alpha_init=0.01,
                target_entropy_ratio=0.0,
                max_grad_norm=0.0,
            ),
        )

        if env_name == "AllegroInhandRotation":
            cfg.num_envs = 4096
            cfg.batch_size = 8192
            cfg.replay_buffer_n = 1024
            cfg.updates_per_step = 4
            cfg.gamma = 0.97
            cfg.tau = 0.125
            cfg.actor_lr = 3e-4
            # cfg.warmup_steps = 2000

            cfg.algo_params.alpha_init = 0.01
            cfg.algo_params.target_entropy_ratio = 1.0
            cfg.algo_params.max_grad_norm = 1.0
            
            cfg.obs_normalization = True
            cfg.max_iterations = 25000

        return cfg

    if algo_name == "td3":
        cfg = config_dict.create(
            algo="td3",
            algo_log_name="fast_td3",
            seed=1,
            num_envs=4096,
            batch_size=8192,
            replay_buffer_n=1000,
            updates_per_step=4,
            warmup_steps=100,
            policy_frequency=2,
            env_steps_per_sync=1,
            max_iterations=5000,
            save_interval=500,
            gamma=0.97,
            tau=0.1,
            actor_lr=3e-4,
            critic_lr=3e-4,
            actor_hidden_dim=256,
            critic_hidden_dim=512,
            num_atoms=101,
            obs_normalization=True,
            use_layer_norm=False,
            algo_params=config_dict.create(
                weight_decay=0.1,
                v_min=-10.0,
                v_max=10.0,
                init_scale=0.01,
                log_std_min=-0.9,
                log_std_max=0.0,
                policy_noise=0.2,
                noise_clip=0.5,
                use_cdq=True,
            ),
        )

        return cfg

    raise ValueError(f"Unsupported off-policy algo: {algo}")