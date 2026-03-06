from ml_collections import config_dict


DEFAULT_ENV_NUM_BY_TASK: dict[str, int] = {
    "G1JoystickFlatTerrain": 2048,
    "Go1JoystickFlatTerrain": 4096,
    "Go2JoystickFlatTerrain": 4096,
}


def get_default_env_num(env_name: str) -> int:
    """Returns default number of parallel environments for a task."""
    return int(DEFAULT_ENV_NUM_BY_TASK.get(env_name, 4096))


def rsl_rl_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned RSL-RL PPO config for the given environment."""

    rl_config = config_dict.create(
        seed=1,
        runner_class_name="OnPolicyRunner",
        obs_groups={"default": ["policy"]}, # Compatibility with new rsl-rl
        policy=config_dict.create(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
            activation="elu",
            class_name="ActorCritic",
        ),
        algorithm=config_dict.create(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            # mini batch size = num_envs*nsteps / nminibatches
            num_mini_batches=4,
            learning_rate=3.0e-4,  # 5.e-4
            schedule="fixed",  # could be adaptive, fixed
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
        num_steps_per_env=24,  # per iteration
        max_iterations=101,  # number of policy updates
        empirical_normalization=True,
        # logging
        save_interval=50,  # check for potential saves every this many iterations
        experiment_name="test",
        run_name="",
        # load and resume
        resume=False,
        load_run="-1",  # -1 = last run
        checkpoint=-1,  # -1 = last saved model
        resume_path=None,  # updated from load_run and chkpt
    )

    if env_name == "Go1JoystickFlatTerrain":
        # Align Go1 training hyper-parameters with the current Go2 setup.
        rl_config.algorithm.entropy_coef = 0.01
        rl_config.algorithm.learning_rate = 1.0e-3
        rl_config.algorithm.schedule = "adaptive"
        rl_config.algorithm.value_loss_coef = 1.0
        rl_config.algorithm.num_learning_epochs = 5
        rl_config.algorithm.num_mini_batches = 4
        rl_config.num_steps_per_env = 24
        rl_config.save_interval = 100
        rl_config.max_iterations = 151
        rl_config.empirical_normalization = False
    elif env_name == "G1JoystickFlatTerrain":
        # Humanoid needs slightly longer horizon but keep aggressive defaults.
        rl_config.algorithm.entropy_coef = 0.01
        rl_config.algorithm.learning_rate = 1.0e-3
        rl_config.algorithm.schedule = "adaptive"
        rl_config.algorithm.value_loss_coef = 1.0
        rl_config.algorithm.num_learning_epochs = 5
        rl_config.algorithm.num_mini_batches = 4
        rl_config.num_steps_per_env = 24
        rl_config.save_interval = 50
        rl_config.max_iterations = 220
        rl_config.empirical_normalization = False
    elif env_name == "Go2JoystickFlatTerrain":
        rl_config.algorithm.entropy_coef = 0.01
        rl_config.algorithm.learning_rate = 1.0e-3
        rl_config.algorithm.schedule = "adaptive"
        rl_config.algorithm.value_loss_coef = 1.0
        rl_config.algorithm.num_learning_epochs = 5
        rl_config.algorithm.num_mini_batches = 4
        rl_config.num_steps_per_env = 24
        rl_config.save_interval = 100
        rl_config.max_iterations = 101
        rl_config.empirical_normalization = False

    return rl_config


def fast_td3_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned FastTD3 config for the given environment.

    Hyperparameters aligned with reference FastTD3 repository
    (Go1JoystickFlatTerrain / MuJoCoPlayground defaults).
    """

    rl_config = config_dict.create(
        seed=1,
        # Network architecture (reference FastTD3)
        actor_hidden_dim=256,
        critic_hidden_dim=512,
        num_atoms=101,          # distributional C51
        init_scale=0.01,        # actor output layer init
        # Training
        num_envs=4096,
        batch_size=8192,
        num_updates=4,
        warmup_steps=100,
        buffer_size=1000,      # per-env buffer size
        max_iterations=5000,
        save_interval=500,
        # Optimizer (AdamW)
        actor_lr=3e-4,
        critic_lr=3e-4,
        weight_decay=0.1,
        # Algorithm
        gamma=0.97,
        tau=0.1,
        policy_frequency=2,
        policy_noise=0.2,
        noise_clip=0.5,
        # Per-env exploration noise
        std_min=0.4,
        std_max=1.0,
        # Distributional
        v_min=-10.0,
        v_max=10.0,
        use_cdq=True,
        obs_normalization=True,
    )

    if env_name in ("Go2JoystickFlatTerrain", "Go2LocoFlatTerrain"):
        rl_config.max_iterations = 2000
        pass  # defaults are tuned for Go2
    elif env_name in ("Go1JoystickFlatTerrain",):
        pass  # same as default
    elif env_name in ("G1JoystickFlatTerrain",):
        rl_config.num_envs = 2048

    return rl_config


def fast_sac_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned FastSAC config for the given environment.

    Hyperparameters aligned with holosoma FastSACConfig defaults.
    """

    rl_config = config_dict.create(
        seed=1,
        # Network architecture (holosoma-aligned)
        actor_hidden_dim=512,
        critic_hidden_dim=768,
        use_layer_norm=True,
        num_atoms=101,         # distributional C51
        # Training
        num_envs=4096,
        batch_size=8192,
        updates_per_step=4,
        warmup_steps=1000,
        replay_buffer_n=512,
        env_steps_per_sync=1,
        max_iterations=1500,
        save_interval=500,
        # Optimizer (AdamW, holosoma-style)
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        # Algorithm
        gamma=0.97,
        tau=0.125,
        alpha_init=0.01,
        target_entropy_ratio=0.0,
        obs_normalization=True,
        policy_frequency=4,
        max_grad_norm=0.0,         # holosoma: max_grad_norm=0.0
    )

    if env_name in ("Go2JoystickFlatTerrain", "Go2LocoFlatTerrain"):
        rl_config.gamma = 0.99
        rl_config.num_envs = 1024
        rl_config.max_iterations = 1500
    elif env_name in ("Go1JoystickFlatTerrain",):
        rl_config.num_envs = 4096
        rl_config.max_iterations = 2000
    elif env_name in ("G1JoystickFlatTerrain",):
        raise NotImplementedError("G1JoystickFlatTerrain config is not implemented for FastSAC, Please use G1JoystickFlatTerrainSAC instead.")
    elif env_name in ("G1JoystickFlatTerrainSAC",):
        # G1 (29 DOF humanoid): overrides that differ from defaults
        rl_config.updates_per_step = 8       # holosoma: num_updates=8  (default: 4)
        rl_config.replay_buffer_n = 1024     # holosoma: buffer_size=1024 (default: 512)
        rl_config.warmup_steps = 0           # holosoma: learning_starts=10 (default: 10000)
        rl_config.alpha_init = 0.001         # holosoma: alpha_init=0.001 (default: 0.01)
        rl_config.max_iterations = 25000     # holosoma: num_learning_iterations=50000
        rl_config.save_interval = 1000       # holosoma: save_interval=1000

    return rl_config