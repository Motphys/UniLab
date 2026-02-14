from typing import Optional
from ml_collections import config_dict

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
            metrics_interval=4,
            finite_check_interval=4,
            enable_compile=False,
            forward_velocity_bonus_coef=0.0,
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
        rl_config.max_iterations = 101
        rl_config.empirical_normalization = False
        rl_config.algorithm.forward_velocity_bonus_coef = 0.5
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
        rl_config.algorithm.forward_velocity_bonus_coef = 0.5

    return rl_config
