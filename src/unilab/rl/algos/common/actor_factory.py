"""Actor factory helpers for torch off-policy algorithms."""

from __future__ import annotations

from unilab.rl.runtime.offpolicy.actor_adapter import get_offpolicy_actor_adapter


def build_actor(
    algo_type,
    obs_dim,
    action_dim,
    actor_hidden_dim,
    use_layer_norm,
    device,
    actor_num_blocks: int = 2,
    actor_noise_zeta_mu: float = 2.0,
    actor_noise_zeta_max: int = 16,
    priv_info_dim: int | None = None,
    priv_info_embed_dim: int = 9,
    priv_mlp_hidden_dims: tuple[int, ...] | list[int] = (256, 128, 9),
    **kwargs,
):
    """Build the correct actor model based on algorithm type."""
    adapter = get_offpolicy_actor_adapter(str(algo_type))
    if adapter is not None:
        if adapter.build_actor is None:
            raise ValueError(
                f"OffPolicyActorAdapter for algo_type={algo_type!r} does not provide build_actor."
            )
        return adapter.build_actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
            priv_info_dim=priv_info_dim,
            priv_info_embed_dim=priv_info_embed_dim,
            priv_mlp_hidden_dims=priv_mlp_hidden_dims,
        )
    if algo_type == "sac":
        from unilab.rl.algos.sac.learner import SACActor

        return SACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        )
    if algo_type == "flashsac":
        actor_config = kwargs.get("actor_config")
        if actor_config is not None:
            from unilab.rl.algos.flash_sac.actor_spec import (
                FlashSACActorSpec,
                build_custom_actor,
            )

            return build_custom_actor(
                FlashSACActorSpec.from_config(actor_config),
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=device,
            )
        from unilab.rl.algos.flash_sac.network import FlashSACActor

        return FlashSACActor(
            num_blocks=actor_num_blocks,
            input_dim=obs_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
            noise_zeta_mu=actor_noise_zeta_mu,
            noise_zeta_max=actor_noise_zeta_max,
            device=device,
        )
    raise ValueError(
        f"Unknown algo_type: {algo_type}. Custom off-policy actor types must "
        "register an OffPolicyActorAdapter via register_offpolicy_actor_adapter() "
        "or list their registration module in actor_adapter_modules."
    )
