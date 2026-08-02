"""Runtime resolution helpers for RSL-RL PPO script assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rsl_rl.runners import OnPolicyRunner

from unilab.training.rsl_rl import RslRlVecEnvWrapper


@dataclass(frozen=True)
class RslRlPPORuntime:
    """Resolved PPO runtime consumed by the generic RSL-RL entrypoint."""

    wrapper_cls: type[Any]
    runner_cls: type[Any]
    wrapper_kwargs: dict[str, Any]
    required_backend: str | None = None
    required_execution_profile: str | None = None


def validate_rsl_rl_ppo_runtime_owner(
    runtime: RslRlPPORuntime,
    *,
    sim_backend: str,
    execution_profile: str | None,
) -> None:
    """Fail closed when an owner-selected runtime/profile pair is inconsistent.

    A resolver can declare backend/profile requirements without pushing backend
    conditionals into the generic training script.  This is deliberately a
    cold-path check, before an environment or physics backend is constructed.
    """

    if not isinstance(runtime, RslRlPPORuntime):
        raise TypeError("resolved PPO runtime must be an RslRlPPORuntime")
    if not isinstance(sim_backend, str) or not sim_backend.strip():
        raise ValueError("PPO owner sim_backend must be a non-empty string")
    required_backend = runtime.required_backend
    if required_backend is not None and sim_backend != required_backend:
        raise ValueError(
            f"PPO runtime requires training.sim_backend={required_backend!r}, got {sim_backend!r}"
        )
    required_profile = runtime.required_execution_profile
    if required_profile is None and execution_profile is not None:
        raise ValueError(
            "PPO runtime does not accept training.execution_profile; "
            f"got {execution_profile!r} for an owner without an execution profile"
        )
    if required_profile is not None and execution_profile != required_profile:
        raise ValueError(
            "PPO runtime requires "
            f"training.execution_profile={required_profile!r}, got {execution_profile!r}"
        )


def resolve_rsl_rl_ppo_runtime(
    rl_cfg: dict[str, Any],
    *,
    default_wrapper_cls: type[RslRlVecEnvWrapper],
    default_runner_cls: type[Any] = OnPolicyRunner,
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
        return RslRlPPORuntime(
            wrapper_cls=default_wrapper_cls,
            runner_cls=default_runner_cls,
            wrapper_kwargs={},
        )

    from rsl_rl.utils import resolve_callable

    resolver = resolve_callable(str(runtime_resolver))
    runtime = resolver(rl_cfg)
    if runtime is None:
        raise ValueError(
            f"PPO runtime resolver {runtime_resolver!r} returned None for rl_cfg runtime selection."
        )

    wrapper_cls = getattr(runtime, "wrapper_cls", None)
    if not isinstance(wrapper_cls, type):
        raise TypeError(
            f"PPO runtime resolver {runtime_resolver!r} must return an object with "
            "'wrapper_cls' attribute."
        )
    runner_cls = getattr(runtime, "runner_cls", default_runner_cls)
    if not isinstance(runner_cls, type):
        raise TypeError(
            f"PPO runtime resolver {runtime_resolver!r} returned an invalid runner_cls."
        )
    wrapper_kwargs = getattr(runtime, "wrapper_kwargs", {})
    if not isinstance(wrapper_kwargs, dict) or any(
        not isinstance(key, str) for key in wrapper_kwargs
    ):
        raise TypeError(
            f"PPO runtime resolver {runtime_resolver!r} returned invalid wrapper_kwargs."
        )
    required_backend = getattr(runtime, "required_backend", None)
    if required_backend is not None and (
        not isinstance(required_backend, str) or not required_backend.strip()
    ):
        raise TypeError(
            f"PPO runtime resolver {runtime_resolver!r} returned invalid required_backend."
        )
    if isinstance(required_backend, str):
        required_backend = required_backend.strip()
    required_execution_profile = getattr(runtime, "required_execution_profile", None)
    if required_execution_profile is not None and (
        not isinstance(required_execution_profile, str) or not required_execution_profile.strip()
    ):
        raise TypeError(
            "PPO runtime resolver "
            f"{runtime_resolver!r} returned invalid required_execution_profile."
        )
    if isinstance(required_execution_profile, str):
        required_execution_profile = required_execution_profile.strip()
    return RslRlPPORuntime(
        wrapper_cls=wrapper_cls,
        runner_cls=runner_cls,
        wrapper_kwargs=dict(wrapper_kwargs),
        required_backend=required_backend,
        required_execution_profile=required_execution_profile,
    )


__all__ = [
    "RslRlPPORuntime",
    "resolve_rsl_rl_ppo_runtime",
    "validate_rsl_rl_ppo_runtime_owner",
]
