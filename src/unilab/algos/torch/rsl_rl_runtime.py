"""Runtime resolution helpers for RSL-RL PPO script assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unilab.training.rsl_rl import RslRlVecEnvWrapper


@dataclass(frozen=True)
class RslRlPPORuntime:
    """Resolved PPO runtime consumed by the generic RSL-RL entrypoint."""

    wrapper_cls: type[RslRlVecEnvWrapper]


def resolve_rsl_rl_ppo_runtime(
    rl_cfg: dict[str, Any],
    *,
    default_wrapper_cls: type[RslRlVecEnvWrapper],
) -> RslRlPPORuntime:
    """Resolve the PPO runtime bundle from owner config."""
    runtime_resolver = rl_cfg.get("runtime_resolver")
    if runtime_resolver in (None, ""):
        runtime_impl = rl_cfg.get("runtime_impl")
        if runtime_impl not in (None, ""):
            raise ValueError(
                "PPO owner config selected "
                f"runtime_impl={runtime_impl!r} but did not define algo.runtime_resolver."
            )
        return RslRlPPORuntime(wrapper_cls=default_wrapper_cls)

    from rsl_rl.utils import resolve_callable

    resolver = resolve_callable(str(runtime_resolver))
    runtime = resolver(rl_cfg)
    if runtime is None:
        raise ValueError(
            f"PPO runtime resolver {runtime_resolver!r} returned None for rl_cfg runtime selection."
        )

    wrapper_cls = getattr(runtime, "wrapper_cls", None)
    if wrapper_cls is None:
        raise TypeError(
            f"PPO runtime resolver {runtime_resolver!r} must return an object with "
            "'wrapper_cls' attribute."
        )
    return RslRlPPORuntime(wrapper_cls=wrapper_cls)
