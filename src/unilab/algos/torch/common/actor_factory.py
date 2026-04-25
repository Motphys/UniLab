"""Actor factory helpers for torch off-policy algorithms."""

from __future__ import annotations


def build_actor(
    algo_type,
    obs_dim,
    action_dim,
    actor_hidden_dim,
    use_layer_norm,
    device,
    num_envs=1,
    actor_num_blocks: int = 2,
    actor_noise_zeta_mu: float = 2.0,
    actor_noise_zeta_max: int = 16,
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
    if algo_type == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor

        return TD3Actor(
            obs_dim=obs_dim,
            n_act=action_dim,
            num_envs=num_envs,
            hidden_dim=actor_hidden_dim,
            init_scale=0.01,
            log_std_min=-0.9,
            log_std_max=0.0,
            device=device,
        )
    if algo_type == "flashsac":
        from unilab.algos.torch.flash_sac.network import FlashSACActor

        return FlashSACActor(
            num_blocks=actor_num_blocks,
            input_dim=obs_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
            noise_zeta_mu=actor_noise_zeta_mu,
            noise_zeta_max=actor_noise_zeta_max,
            device=device,
        )
    raise ValueError(f"Unknown algo_type: {algo_type}")
