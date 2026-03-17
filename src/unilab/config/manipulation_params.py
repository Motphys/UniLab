from ml_collections import config_dict

KNOWN_TASKS: set = {
    "AllegroInhandRotation",
    "AllegroInhandRotationSac",
}


def ppo_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned RSL-RL PPO config for the given environment."""

    cfg = config_dict.create(
        num_envs=4096,
        num_steps_per_env=24,
        max_iterations=1500,
        save_interval=100,
        empirical_normalization=False,
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        schedule="adaptive",
        value_loss_coef=1.0,
        num_mini_batches=4,
        desired_kl=0.01,
    )

    if env_name == "AllegroInhandRotation":
        cfg.num_envs = 16384
        cfg.max_iterations = 501
        cfg.num_steps_per_env = 8
        cfg.num_mini_batches = 4
        cfg.learning_rate = 1.0e-3
        cfg.schedule = "adaptive"
        cfg.empirical_normalization = True
        cfg.entropy_coef = 0.01
        cfg.value_loss_coef = 4.0
        cfg.desired_kl = 0.02
    elif env_name == "AllegroInhandRotationSac":
        raise NotImplementedError()

    return config_dict.create(
        seed=1,
        num_envs=cfg.num_envs,
        runner_class_name="OnPolicyRunner",
        obs_groups={"actor": ["actor"], "critic": ["actor"]},
        num_steps_per_env=cfg.num_steps_per_env,
        max_iterations=cfg.max_iterations,
        empirical_normalization=cfg.empirical_normalization,
        save_interval=cfg.save_interval,
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
            num_learning_epochs=5,
            num_mini_batches=cfg.num_mini_batches,
            learning_rate=cfg.learning_rate,
            schedule=cfg.schedule,
            gamma=0.99,
            lam=0.95,
            desired_kl=cfg.desired_kl,
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


def appo_config(env_name: str) -> config_dict.ConfigDict:
    """Return APPO config.

    Mirrors the structure of ppo_config/offpolicy_config so training
    infrastructure is reusable.  Algorithm hyperparams live under the
    ``algorithm`` sub-dict; network architecture under ``actor``/``critic``.
    """
    cfg = config_dict.create(
        seed=1,
        num_envs=2048,
        steps_per_env=24,
        max_iterations=150,
        save_interval=50,
        value_loss_coef=1.0,
        desired_kl=0.01,
        empirical_normalization=False,
    )

    if env_name == "AllegroInhandRotation":
        cfg.num_envs = 16384
        cfg.max_iterations = 501
        cfg.steps_per_env = 8
        cfg.value_loss_coef = 4.0
        cfg.desired_kl = 0.02
        cfg.save_interval = 100
        cfg.empirical_normalization = True
    elif env_name == "AllegroInhandRotationSac":
        raise NotImplementedError()

    return config_dict.create(
        algo="appo",
        algo_log_name="appo",
        seed=cfg.seed,
        num_envs=cfg.num_envs,
        steps_per_env=cfg.steps_per_env,
        max_iterations=cfg.max_iterations,
        save_interval=cfg.save_interval,
        empirical_normalization=cfg.empirical_normalization,
        obs_groups=config_dict.create(
            actor=config_dict.create(policy=0),
        ),
        actor=config_dict.create(
            class_name="rsl_rl.models.MLPModel",
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=cfg.empirical_normalization,
            distribution_cfg=config_dict.create(
                class_name="rsl_rl.modules.distribution.GaussianDistribution",
                init_std=1.0,
                std_type="scalar",
            ),
        ),
        critic=config_dict.create(
            class_name="rsl_rl.models.MLPModel",
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=cfg.empirical_normalization,
        ),
        algorithm=config_dict.create(
            num_learning_epochs=5,
            num_mini_batches=4,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
            value_loss_coef=cfg.value_loss_coef,
            entropy_coef=0.01,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            schedule="adaptive",
            desired_kl=cfg.desired_kl,
            optimizer="adam",
            tau=1.0,
            target_update_freq=1,
            vtrace_clip_rho=1.0,
            vtrace_clip_c=1.0,
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
            alpha_lr=3e-4,
            actor_hidden_dim=512,
            critic_hidden_dim=768,
            num_atoms=101,
            obs_normalization=True,
            use_layer_norm=True,
            use_symmetry=False,
            alpha_init=0.01,
            target_entropy_ratio=0.0,
            max_grad_norm=0.0,
        )

        if env_name == "AllegroInhandRotationSac":
            cfg.num_envs = 2048
            cfg.batch_size = 16384
            cfg.replay_buffer_n = 1024
            cfg.updates_per_step = 4
            cfg.gamma = 0.99
            cfg.tau = 0.005
            cfg.actor_lr = 1e-4
            cfg.obs_normalization = True
            cfg.alpha_init = 0.001
            cfg.target_entropy_ratio = 0.0
            cfg.max_grad_norm = 1.0
            cfg.save_interval = 1000
            cfg.max_iterations = 25000
        elif env_name == "AllegroInhandRotation":
            raise NotImplementedError(
                "AllegroInhandRotation config is not implemented for FastSAC, Please use AllegroInhandRotationSac instead."
            )

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
            use_symmetry=cfg.use_symmetry,
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
            weight_decay=0.1,
            v_min=-10.0,
            v_max=10.0,
            init_scale=0.01,
            log_std_min=-0.9,
            log_std_max=0.0,
            policy_noise=0.2,
            noise_clip=0.5,
            use_cdq=True,
        )

        return config_dict.create(
            algo="td3",
            algo_log_name="fast_td3",
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
