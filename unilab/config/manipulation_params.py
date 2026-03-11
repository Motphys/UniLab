from ml_collections import config_dict


DEFAULT_ENV_NUM_BY_TASK: dict[str, int] = {
    "AllegroInhandRotation": 16384,
}


def get_default_env_num(env_name: str) -> int:
    """Returns default number of parallel environments for a task."""
    return int(DEFAULT_ENV_NUM_BY_TASK.get(env_name, 4096))


def rsl_rl_config(env_name: str) -> config_dict.ConfigDict:
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
        rl_config.max_iterations = 10000
        rl_config.num_steps_per_env = 8

        rl_config.algorithm.num_mini_batches = 4
        rl_config.algorithm.learning_rate = 1.0e-3

        rl_config.algorithm.entropy_coef = 0.01
        rl_config.algorithm.value_loss_coef = 4.0
        rl_config.algorithm.desired_kl = 0.02

        rl_config.algorithm.schedule = "adaptive"
        rl_config.empirical_normalization = True

        rl_config.max_iterations = 10000


    return rl_config
