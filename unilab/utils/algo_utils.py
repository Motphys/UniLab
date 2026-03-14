"""Common utilities for RL algorithms."""

import pkgutil
import importlib


def ensure_registries():
    """Import all env modules so they are registered."""
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


def build_actor(
    algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, device, num_envs=1
):
    """Build the correct actor model based on algorithm type."""
    if algo_type == "sac":
        from unilab.algos.torch.fast_sac.learner import SACActor

        return SACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        )
    elif algo_type == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor

        return TD3Actor(
            n_obs=obs_dim,
            n_act=action_dim,
            num_envs=num_envs,
            hidden_dim=actor_hidden_dim,
            init_scale=0.01,
            log_std_min=-0.9,
            log_std_max=0.0,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")
