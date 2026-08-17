from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from unilab.base.registry import apply_cfg_overrides
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from unilab.envs.mdp import JointPositionActionCfg
from unilab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
)


def _first_term(_env, *, command_name: str, std: float) -> np.ndarray:
    del command_name, std
    return np.zeros(1, dtype=np.float32)


def _second_term(_env) -> np.ndarray:
    return np.zeros(1, dtype=np.float32)


def _manager_cfg() -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "first": ObservationTermCfg(func=_first_term),
                    "second": ObservationTermCfg(func=_second_term),
                }
            )
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.25,
            )
        },
        events={
            "reset": EventTermCfg(
                func=_first_term,
                mode="reset",
                params={"command_name": "twist", "std": 0.5},
            )
        },
        rewards={
            "tracking": RewardTermCfg(
                func=_first_term,
                weight=1.0,
                params={
                    "command_name": "twist",
                    "std": 0.5,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                },
            ),
            "alive": RewardTermCfg(func=_second_term, weight=0.1),
            "disabled": None,
        },
    )


def test_manager_mapping_overlay_preserves_factory_terms_and_order() -> None:
    cfg = _manager_cfg()
    reward_func = cfg.rewards["tracking"].func

    apply_cfg_overrides(
        cfg,
        {
            "rewards": {
                "tracking": {
                    "weight": 2.0,
                    "params": {"std": 0.25, "asset_cfg": {"preserve_order": True}},
                },
                "alive": None,
            },
            "events": {"reset": {"min_step_count_between_reset": 3}},
            "actions": {"joint_pos": {"scale": 0.4}},
        },
    )

    tracking = cfg.rewards["tracking"]
    assert tracking is not None
    assert tracking.func is reward_func
    assert tracking.weight == pytest.approx(2.0)
    assert tracking.params["command_name"] == "twist"
    assert tracking.params["std"] == pytest.approx(0.25)
    assert tracking.params["asset_cfg"].joint_names == ".*"
    assert tracking.params["asset_cfg"].preserve_order is True
    assert cfg.rewards["alive"] is None
    assert list(cfg.rewards) == ["tracking", "alive", "disabled"]
    reset = cfg.events["reset"]
    assert reset is not None
    assert reset.func is _first_term
    assert reset.min_step_count_between_reset == 3
    action = cfg.actions["joint_pos"]
    assert action is not None
    assert action.entity_name == "robot"
    assert action.actuator_names == (".*",)
    assert action.scale == pytest.approx(0.4)


def test_observation_group_and_term_overlay_preserve_siblings() -> None:
    cfg = _manager_cfg()

    apply_cfg_overrides(
        cfg,
        {
            "observations": {
                "policy": {
                    "enable_corruption": True,
                    "terms": {"first": {"scale": 2.0}, "second": None},
                }
            }
        },
    )

    policy = cfg.observations["policy"]
    assert policy is not None
    assert policy.enable_corruption is True
    first = policy.terms["first"]
    assert first is not None
    assert first.func is _first_term
    assert first.scale == pytest.approx(2.0)
    assert policy.terms["second"] is None
    assert list(policy.terms) == ["first", "second"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"rewards": {"missing": {"weight": 1.0}}}, "rewards.*missing"),
        ({"rewards": {"disabled": {"weight": 1.0}}}, "disabled.*task Python factory"),
        ({"rewards": {"tracking": _second_term}}, "tracking.*replacing"),
        ({"rewards": []}, "rewards.*mapping"),
        ({"rewards": {"tracking": {"func": _second_term}}}, "func.*factory-owned"),
    ],
)
def test_manager_mapping_overlay_fails_closed(overrides: dict, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        apply_cfg_overrides(_manager_cfg(), overrides)


@dataclass
class _LegacyCfg:
    reward_config: dict[str, object] = field(
        default_factory=lambda: {
            "scales": {"tracking": 1.0, "alive": 0.1},
            "tracking_sigma": 0.25,
        }
    )


def test_legacy_plain_dict_keeps_replacement_semantics() -> None:
    cfg = _LegacyCfg()

    apply_cfg_overrides(cfg, {"reward_config": {"scales": {"alive": 1.0}}})

    assert cfg.reward_config == {"scales": {"alive": 1.0}}
