"""Internal, cold-path compatibility seam for legacy task factories.

This module is intentionally task-owned and is not part of the base registry
contract.  It only admits legacy factories that already use UniLab's
``EnvCfg -> NpEnv`` lifecycle; it does not provide a fallback runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from unilab.base.base import ABEnv, EnvCfg
from unilab.base.np_env import NpEnv


class CompatibilityStatus(str, Enum):
    """Documented outcome for one legacy compatibility boundary."""

    ADAPTED = "Adapted"
    UNSUPPORTED = "Unsupported"


@dataclass(frozen=True)
class LegacyTaskCompatibility:
    """Immutable compatibility evidence attached to one task-family seam."""

    task_family: str
    status: CompatibilityStatus
    reason: str

    def __post_init__(self) -> None:
        if not self.task_family.strip():
            raise ValueError("legacy compatibility task_family must be non-empty")
        if not self.reason.strip():
            raise ValueError("legacy compatibility reason must be non-empty")


class LegacyEnvFactory(Protocol):
    """Existing registry-shaped legacy factory admitted by this seam."""

    def __call__(
        self,
        cfg: EnvCfg,
        *,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> ABEnv: ...


@dataclass(frozen=True)
class LegacyFactoryAdapter:
    """Validate one legacy factory at the existing env construction boundary."""

    factory: LegacyEnvFactory
    compatibility: LegacyTaskCompatibility

    def __post_init__(self) -> None:
        if self.compatibility.status is not CompatibilityStatus.ADAPTED:
            raise ValueError("LegacyFactoryAdapter compatibility status must be Adapted")

    def __call__(
        self,
        cfg: EnvCfg,
        *,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> NpEnv:
        family = self.compatibility.task_family
        if not isinstance(cfg, EnvCfg):
            raise TypeError(
                f"Legacy task family '{family}' expected EnvCfg, received {type(cfg).__name__}"
            )

        env = self.factory(cfg, num_envs=num_envs, backend_type=backend_type)
        if not isinstance(env, ABEnv):
            raise TypeError(
                f"Legacy task family '{family}' factory returned {type(env).__name__}, "
                "expected ABEnv"
            )
        if not isinstance(env, NpEnv):
            raise TypeError(
                f"Legacy task family '{family}' compatibility is Unsupported: "
                f"{type(env).__name__} does not use the NpEnv lifecycle"
            )
        return env


def adapt_legacy_factory(
    factory: LegacyEnvFactory,
    *,
    task_family: str,
    reason: str,
) -> LegacyFactoryAdapter:
    """Mark and wrap an existing ``EnvCfg -> NpEnv`` task factory.

    The wrapper runs only while the registry constructs an environment.  It
    forwards the registry's fixed arguments exactly once and rejects any
    other config or runtime shape instead of probing or falling back.
    """

    if not callable(factory):
        raise TypeError(f"legacy task family '{task_family}' factory must be callable")
    compatibility = LegacyTaskCompatibility(
        task_family=task_family,
        status=CompatibilityStatus.ADAPTED,
        reason=reason,
    )
    return LegacyFactoryAdapter(factory=factory, compatibility=compatibility)


def unsupported_legacy_task(*, task_family: str, reason: str) -> LegacyTaskCompatibility:
    """Record an explicit unsupported surface without creating a factory."""

    return LegacyTaskCompatibility(
        task_family=task_family,
        status=CompatibilityStatus.UNSUPPORTED,
        reason=reason,
    )


__all__ = [
    "CompatibilityStatus",
    "LegacyFactoryAdapter",
    "LegacyTaskCompatibility",
    "adapt_legacy_factory",
    "unsupported_legacy_task",
]
