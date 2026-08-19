"""Focused contract tests for the internal legacy-task compatibility seam."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest

from unilab.base.base import ABEnv, EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.tasks.compatibility import (
    CompatibilityStatus,
    LegacyFactoryAdapter,
    adapt_legacy_factory,
    unsupported_legacy_task,
)


@dataclass
class _Cfg(EnvCfg):
    pass


class _PlainABEnv(ABEnv):
    @property
    def num_envs(self) -> int:
        return 1

    @property
    def cfg(self) -> EnvCfg:
        return _Cfg()

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 1}

    @property
    def state(self) -> None:
        return None

    def init_state(self) -> None:
        return None

    def step(self, actions: np.ndarray) -> None:
        return None

    def close(self) -> None:
        return None


class _NpEnv(NpEnv):
    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 1}

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        return actions

    def update_state(self, state: NpEnvState) -> NpEnvState:
        return state


def _uninitialized_np_env() -> _NpEnv:
    return object.__new__(_NpEnv)


def test_adapter_records_fixed_adapted_metadata_and_forwards_registry_arguments() -> None:
    received: list[tuple[EnvCfg, int, str]] = []
    expected = _uninitialized_np_env()

    def factory(cfg: EnvCfg, *, num_envs: int, backend_type: str) -> ABEnv:
        received.append((cfg, num_envs, backend_type))
        return expected

    adapter = adapt_legacy_factory(
        factory,
        task_family="Go2ArmManipLoco",
        reason="existing task owner already constructs an NpEnv",
    )
    cfg = _Cfg()

    assert adapter(cfg, num_envs=4, backend_type="motrix") is expected
    assert received == [(cfg, 4, "motrix")]
    assert adapter.compatibility.task_family == "Go2ArmManipLoco"
    assert adapter.compatibility.status is CompatibilityStatus.ADAPTED
    assert adapter.compatibility.reason == "existing task owner already constructs an NpEnv"


def test_adapter_rejects_non_env_cfg_before_calling_factory() -> None:
    called = False

    def factory(cfg: EnvCfg, *, num_envs: int, backend_type: str) -> ABEnv:
        nonlocal called
        called = True
        return _uninitialized_np_env()

    adapter = adapt_legacy_factory(factory, task_family="Sharpa", reason="migration seam")

    with pytest.raises(TypeError, match=r"Sharpa.*expected EnvCfg.*dict"):
        adapter({})  # type: ignore[arg-type]

    assert called is False


@pytest.mark.parametrize(
    ("result", "match"),
    (
        (object(), r"Go2ArmManipLoco.*object.*expected ABEnv"),
        (_PlainABEnv(), r"Go2ArmManipLoco.*Unsupported.*_PlainABEnv.*NpEnv"),
    ),
)
def test_adapter_rejects_factories_outside_the_np_env_lifecycle(
    result: object,
    match: str,
) -> None:
    def factory(cfg: EnvCfg, *, num_envs: int, backend_type: str) -> object:
        return result

    adapter = adapt_legacy_factory(
        factory,  # type: ignore[arg-type]
        task_family="Go2ArmManipLoco",
        reason="migration seam",
    )

    with pytest.raises(TypeError, match=match):
        adapter(_Cfg())


def test_factory_exception_propagates_without_fallback() -> None:
    failure = RuntimeError("owner factory failed")

    def factory(cfg: EnvCfg, *, num_envs: int, backend_type: str) -> ABEnv:
        raise failure

    adapter = adapt_legacy_factory(factory, task_family="Sharpa", reason="migration seam")

    with pytest.raises(RuntimeError) as exc_info:
        adapter(_Cfg())

    assert exc_info.value is failure


def test_unsupported_metadata_is_explicit_and_does_not_create_a_factory() -> None:
    compatibility = unsupported_legacy_task(
        task_family="Sharpa foreign lifecycle",
        reason="only the existing NpEnv lifecycle is admitted",
    )

    assert compatibility.status is CompatibilityStatus.UNSUPPORTED
    assert compatibility.reason == "only the existing NpEnv lifecycle is admitted"

    with pytest.raises(ValueError, match="status must be Adapted"):
        LegacyFactoryAdapter(lambda cfg, **kwargs: _uninitialized_np_env(), compatibility)


@pytest.mark.parametrize(("task_family", "reason"), (("", "reason"), ("Sharpa", "")))
def test_compatibility_metadata_requires_stable_family_and_reason(
    task_family: str,
    reason: str,
) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        unsupported_legacy_task(task_family=task_family, reason=reason)
