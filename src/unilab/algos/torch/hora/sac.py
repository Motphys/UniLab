"""HORA-owned SAC entry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unilab.algos.torch.hora.runtime import HORA_SAC_RUNTIME_IMPL, is_hora_sac_runtime
from unilab.algos.torch.hora.sac_learner import HoraSACLearner
from unilab.algos.torch.offpolicy.runtime import OffPolicyRuntime


@dataclass(frozen=True)
class HoraSACRuntime(OffPolicyRuntime):
    """Resolved HORA-SAC hooks consumed by the generic off-policy script."""

    learner_cls: type[Any] | None = HoraSACLearner
    algo_type: str | None = HORA_SAC_RUNTIME_IMPL
    supports_symmetry: bool = False
    actor_cfg: dict[str, Any] = field(default_factory=dict)

    def build_model_kwargs(self, *, obs_dim: int, critic_obs_dim: int) -> dict[str, Any]:
        """Build shared learner/collector actor kwargs for the HORA-SAC actor."""
        priv_info_dim = int(critic_obs_dim - obs_dim)
        if priv_info_dim <= 0:
            raise ValueError(
                "HORA-SAC requires critic observations to contain privileged tail "
                f"features; got obs_dim={obs_dim}, critic_obs_dim={critic_obs_dim}."
            )
        return {
            "priv_info_dim": priv_info_dim,
            "priv_info_embed_dim": int(self.actor_cfg.get("priv_info_embed_dim", 9)),
            "priv_mlp_hidden_dims": tuple(
                self.actor_cfg.get("priv_mlp_hidden_dims", (256, 128, 9))
            ),
        }


def resolve_hora_sac_runtime(rl_cfg: dict[str, Any]) -> HoraSACRuntime | None:
    """Resolve HORA-SAC hooks from an explicit owner-config runtime marker."""
    if not is_hora_sac_runtime(rl_cfg):
        return None
    actor_cfg_raw = rl_cfg.get("actor", {})
    actor_cfg = actor_cfg_raw if isinstance(actor_cfg_raw, dict) else {}
    return HoraSACRuntime(actor_cfg=dict(actor_cfg))


__all__ = ["HoraSACRuntime", "resolve_hora_sac_runtime"]
