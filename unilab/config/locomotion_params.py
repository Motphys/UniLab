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
            max_grad_norm=1.0,
        ),
        num_steps_per_env=24,  # per iteration
        max_iterations=100000,  # number of policy updates
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

    if env_name in (
        "Go1JoystickFlatTerrain",
        "Go2JoystickFlatTerrain",
    ):
        rl_config.max_iterations = 1000
    if env_name == "Go1JoystickFlatTerrain":
        rl_config.algorithm.learning_rate = 3e-4
        rl_config.algorithm.schedule = "fixed"
    elif env_name == "Go2JoystickFlatTerrain":
        rl_config.algorithm.entropy_coef = 0.01
        rl_config.algorithm.learning_rate = 1.0e-3
        rl_config.algorithm.schedule = "adaptive"
        rl_config.max_iterations = 1500

    return rl_config
