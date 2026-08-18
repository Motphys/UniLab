from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.base import EnvCfg
from unilab.base.config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_TERM_MAPPING_POLICY,
)
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from unilab.managers import RewardTermCfg
from unilab.training.backend_adapter import BackendAdapter

_MANAGER_ENV = "_TestManagerRewardRoute"
_MANAGER_FACTORY_ENV = "_TestManagerFactoryRewardRoute"
_LEGACY_ENV = "_TestLegacyRewardRoute"
_MISSING_ENV = "_TestMissingRewardRoute"
_AMBIGUOUS_ENV = "_TestAmbiguousRewardRoute"


@dataclass
class _LegacyCfg(EnvCfg):
    reward_config: dict[str, object] = field(default_factory=dict)


@dataclass
class _MissingCfg(EnvCfg):
    pass


@dataclass
class _AmbiguousCfg(EnvCfg):
    reward_config: dict[str, object] = field(default_factory=dict)
    rewards: dict[str, object] = field(
        default_factory=dict,
        metadata={CONFIG_MAPPING_POLICY_KEY: MANAGER_TERM_MAPPING_POLICY},
    )


def _make_manager_cfg() -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg()


for _name, _cfg_factory in (
    (_MANAGER_ENV, ManagerBasedRlEnvCfg),
    (_MANAGER_FACTORY_ENV, _make_manager_cfg),
    (_LEGACY_ENV, _LegacyCfg),
    (_MISSING_ENV, _MissingCfg),
    (_AMBIGUOUS_ENV, _AmbiguousCfg),
):
    if not registry.contains(_name):
        registry.register_env_config(_name, _cfg_factory)


def _reward(_env, *, std: float) -> np.ndarray:
    del std
    return np.zeros(1, dtype=np.float32)


def _alive(_env) -> np.ndarray:
    return np.ones(1, dtype=np.float32)


def _cfg(task_name: str, *, env: dict[str, object] | None = None):
    return OmegaConf.create(
        {
            "training": {"task_name": task_name},
            "reward": {
                "tracking": {
                    "weight": 2.0,
                    "params": {"std": 0.25},
                }
            },
            "env": env or {"ctrl_dt": 0.02},
        }
    )


@pytest.mark.parametrize("env_name", [_MANAGER_ENV, _MANAGER_FACTORY_ENV])
def test_backend_adapter_routes_manager_reward_and_preserves_factory_terms(
    env_name: str,
) -> None:
    override = BackendAdapter(
        _cfg(env_name),
        root_dir=Path("."),
    ).build_task_env_cfg_override()

    assert "reward_config" not in override
    assert override["rewards"]["tracking"]["weight"] == pytest.approx(2.0)
    assert override["ctrl_dt"] == pytest.approx(0.02)

    manager_cfg = ManagerBasedRlEnvCfg(
        rewards={
            "tracking": RewardTermCfg(func=_reward, weight=1.0, params={"std": 0.5}),
            "alive": RewardTermCfg(func=_alive, weight=0.1),
        }
    )
    factory_tracking = manager_cfg.rewards["tracking"]
    assert factory_tracking is not None
    tracking_func = factory_tracking.func

    apply_cfg_overrides(manager_cfg, override)

    tracking = manager_cfg.rewards["tracking"]
    assert tracking is not None
    assert tracking.func is tracking_func
    assert tracking.weight == pytest.approx(2.0)
    assert tracking.params == {"std": 0.25}
    assert manager_cfg.rewards["alive"] is not None
    assert list(manager_cfg.rewards) == ["tracking", "alive"]


def test_backend_adapter_preserves_legacy_reward_target() -> None:
    override = BackendAdapter(
        _cfg(_LEGACY_ENV),
        root_dir=Path("."),
    ).build_task_env_cfg_override()

    assert "rewards" not in override
    assert override["reward_config"]["tracking"]["weight"] == pytest.approx(2.0)
    assert override["ctrl_dt"] == pytest.approx(0.02)


def test_backend_adapter_rejects_duplicate_manager_reward_sources() -> None:
    cfg = _cfg(_MANAGER_ENV, env={"rewards": {"tracking": {"weight": 3.0}}})

    with pytest.raises(ValueError, match="root 'reward'.*env.rewards"):
        BackendAdapter(cfg, root_dir=Path(".")).build_task_env_cfg_override()


@pytest.mark.parametrize(
    ("env_name", "match"),
    [
        ("_UnregisteredRewardRoute", "not registered"),
        (_MISSING_ENV, "declares no supported.*reward target"),
        (_AMBIGUOUS_ENV, "declares both.*rewards.*reward_config"),
    ],
)
def test_reward_target_resolution_fails_closed(env_name: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        registry.resolve_reward_override_field(env_name)
