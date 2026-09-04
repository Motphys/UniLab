from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.base.registry import apply_cfg_overrides
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from unilab.envs.mdp import JointPositionActionCfg, UniformVelocityCommandCfg
from unilab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
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
    initial_tracking = cfg.rewards["tracking"]
    assert initial_tracking is not None
    reward_func = initial_tracking.func

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
    assert isinstance(action, JointPositionActionCfg)
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


def _assert_no_omegaconf(value: object) -> None:
    assert not OmegaConf.is_config(value)
    if is_dataclass(value) and not isinstance(value, type):
        for config_field in fields(value):
            _assert_no_omegaconf(getattr(value, config_field.name))
    elif isinstance(value, dict):
        for item in value.values():
            _assert_no_omegaconf(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_omegaconf(item)


def test_hydra_mapping_fully_materializes_empty_manager_config() -> None:
    cfg = ManagerBasedRlEnvCfg()
    hydra_mapping = OmegaConf.create(
        {
            "scene": {
                "model_file": "robot.xml",
                "entities": {
                    "robot": {
                        "root_body_name": "base",
                        "joint_names": ["joint"],
                        "actuator_names": ["motor"],
                    }
                },
            },
            "sim_dt": 0.01,
            "ctrl_dt": 0.02,
            "max_episode_seconds": 20.0,
            "observations": {
                "policy": {
                    "_target_": "unilab.managers.ObservationGroupCfg",
                    "terms": {
                        "joint_pos": {
                            "_target_": "unilab.managers.ObservationTermCfg",
                            "func": "unilab.envs.mdp.joint_pos_rel",
                            "params": {
                                "asset_cfg": {
                                    "_target_": "unilab.managers.SceneEntityCfg",
                                    "name": "robot",
                                    "joint_names": ".*",
                                }
                            },
                        },
                        "disabled": None,
                    },
                }
            },
            "actions": {
                "joint_pos": {
                    "_target_": "unilab.envs.mdp.JointPositionActionCfg",
                    "entity_name": "robot",
                    "actuator_names": [".*"],
                    "scale": 0.25,
                }
            },
            "commands": {
                "twist": {
                    "_target_": "unilab.envs.mdp.UniformVelocityCommandCfg",
                    "entity_name": "robot",
                    "resampling_time_range": [1.0, 1.0],
                    "ranges": {
                        "lin_vel_x": [-1.0, 1.0],
                        "lin_vel_y": [-0.5, 0.5],
                        "ang_vel_z": [-1.0, 1.0],
                    },
                }
            },
            "events": {
                "reset": {
                    "_target_": "unilab.managers.EventTermCfg",
                    "func": "unilab.envs.mdp.reset_scene_to_default",
                    "mode": "reset",
                }
            },
            "rewards": {
                "alive": {
                    "_target_": "unilab.managers.RewardTermCfg",
                    "func": "unilab.envs.mdp.is_alive",
                    "weight": 1.0,
                }
            },
            "terminations": {
                "time_out": {
                    "_target_": "unilab.managers.TerminationTermCfg",
                    "func": "unilab.envs.mdp.time_out",
                    "time_out": True,
                }
            },
            "curriculum": {
                "difficulty": {
                    "_target_": "unilab.managers.CurriculumTermCfg",
                    "func": "unilab.envs.mdp.is_alive",
                }
            },
            "metrics": {
                "alive": {
                    "_target_": "unilab.managers.MetricsTermCfg",
                    "func": "unilab.envs.mdp.is_alive",
                }
            },
            "recorders": {
                "trace": {
                    "_target_": "unilab.managers.RecorderTermCfg",
                    "func": "unilab.managers.RecorderTerm",
                }
            },
            "policy_observation_group": "policy",
        }
    )

    apply_cfg_overrides(cfg, hydra_mapping)
    cfg.validate()

    assert cfg.scene is not None
    assert cfg.scene.entities["robot"].joint_names == ["joint"]
    assert list(cfg.observations) == ["policy"]
    policy = cfg.observations["policy"]
    assert isinstance(policy, ObservationGroupCfg)
    assert list(policy.terms) == ["joint_pos", "disabled"]
    joint_obs = policy.terms["joint_pos"]
    assert isinstance(joint_obs, ObservationTermCfg)
    assert joint_obs.func is not None and callable(joint_obs.func)
    assert isinstance(joint_obs.params["asset_cfg"], SceneEntityCfg)
    assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    assert isinstance(twist.ranges, UniformVelocityCommandCfg.Ranges)
    assert isinstance(cfg.events["reset"], EventTermCfg)
    assert isinstance(cfg.rewards["alive"], RewardTermCfg)
    assert isinstance(cfg.terminations["time_out"], TerminationTermCfg)
    assert isinstance(cfg.curriculum["difficulty"], CurriculumTermCfg)
    assert isinstance(cfg.metrics["alive"], MetricsTermCfg)
    assert isinstance(cfg.recorders["trace"], RecorderTermCfg)
    _assert_no_omegaconf(cfg)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"actions": {"missing": {"scale": 0.25}}}, "missing.*_target_"),
        ({"rewards": {"tracking": _second_term}}, "tracking.*field mapping"),
        ({"rewards": []}, "rewards.*mapping"),
        ({"rewards": {"tracking": {"func": _second_term}}}, "func.*typed term"),
    ],
)
def test_manager_mapping_overlay_fails_closed(overrides: dict, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        apply_cfg_overrides(_manager_cfg(), overrides)


def test_manager_mapping_overlay_infers_concrete_term_target() -> None:
    cfg = _manager_cfg()

    apply_cfg_overrides(
        cfg,
        {
            "rewards": {
                "missing": {
                    "func": "unilab.envs.mdp.is_alive",
                    "weight": 0.5,
                },
                "disabled": {
                    "func": "unilab.envs.mdp.is_alive",
                    "weight": 0.2,
                },
            },
            "observations": {
                "critic": {
                    "terms": {
                        "joint_vel": {"func": "unilab.envs.mdp.joint_vel_rel"},
                    }
                }
            },
        },
    )

    missing = cfg.rewards["missing"]
    assert isinstance(missing, RewardTermCfg)
    assert callable(missing.func)
    assert missing.weight == pytest.approx(0.5)
    disabled = cfg.rewards["disabled"]
    assert isinstance(disabled, RewardTermCfg)
    assert disabled.weight == pytest.approx(0.2)
    assert list(cfg.rewards) == ["tracking", "alive", "disabled", "missing"]
    critic = cfg.observations["critic"]
    assert isinstance(critic, ObservationGroupCfg)
    joint_vel = critic.terms["joint_vel"]
    assert isinstance(joint_vel, ObservationTermCfg)
    assert callable(joint_vel.func)


def test_hydra_materialization_resolves_managers_short_name() -> None:
    cfg = ManagerBasedRlEnvCfg()

    apply_cfg_overrides(
        cfg,
        {
            "events": {
                "base_mass": {
                    "func": "unilab.envs.mdp.is_alive",
                    "mode": "reset",
                    "params": {
                        "asset_cfg": {"_target_": "SceneEntityCfg", "name": "robot"},
                    },
                }
            }
        },
    )

    base_mass = cfg.events["base_mass"]
    assert isinstance(base_mass, EventTermCfg)
    asset_cfg = base_mass.params["asset_cfg"]
    assert isinstance(asset_cfg, SceneEntityCfg)
    assert asset_cfg.name == "robot"


def test_hydra_materialization_rejects_unknown_short_name() -> None:
    with pytest.raises(ValueError, match="could not resolve"):
        apply_cfg_overrides(
            ManagerBasedRlEnvCfg(),
            {
                "rewards": {
                    "term": {
                        "_target_": "NoSuchCfg",
                        "func": "unilab.envs.mdp.is_alive",
                        "weight": 1.0,
                    }
                }
            },
        )


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (
            {
                "_target_": "unilab.managers.EventTermCfg",
                "func": "unilab.envs.mdp.is_alive",
                "mode": "reset",
            },
            "EventTermCfg.*RewardTermCfg",
        ),
        (
            {"_target_": "unilab.envs.mdp.is_alive", "func": "unilab.envs.mdp.is_alive"},
            "dataclass type",
        ),
        (
            {"_target_": "unilab.managers.MissingCfg", "func": "unilab.envs.mdp.is_alive"},
            "_target_.*could not resolve",
        ),
        (
            {
                "_target_": "unilab.managers.RewardTermCfg",
                "func": "unilab.envs.mdp.missing",
                "weight": 1.0,
            },
            "func.*could not resolve",
        ),
        (
            {
                "_target_": "unilab.managers.RewardTermCfg",
                "func": "unilab.base.config_overrides.CONFIG_MAPPING_POLICY_KEY",
                "weight": 1.0,
            },
            "expected a callable",
        ),
        (
            {
                "_target_": "unilab.managers.RewardTermCfg",
                "func": "unilab.envs.mdp.is_alive",
                "weight": 1.0,
                "unknown": 1,
            },
            "has no fields.*unknown",
        ),
        (
            {"_target_": "unilab.managers.RewardTermCfg", "func": "unilab.envs.mdp.is_alive"},
            "could not construct",
        ),
    ],
)
def test_hydra_manager_materialization_fails_closed(entry: dict, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        apply_cfg_overrides(ManagerBasedRlEnvCfg(), {"rewards": {"term": entry}})


def test_hydra_manager_materialization_rejects_abstract_action_config() -> None:
    with pytest.raises(TypeError, match="abstract config.*ActionTermCfg"):
        apply_cfg_overrides(
            ManagerBasedRlEnvCfg(),
            {
                "actions": {
                    "joint": {
                        "_target_": "unilab.managers.ActionTermCfg",
                        "entity_name": "robot",
                    }
                }
            },
        )


def test_hydra_manager_materialization_rejects_wrong_nested_config_type() -> None:
    with pytest.raises(TypeError, match="SceneCfg.*Ranges"):
        apply_cfg_overrides(
            ManagerBasedRlEnvCfg(),
            {
                "commands": {
                    "twist": {
                        "_target_": "unilab.envs.mdp.UniformVelocityCommandCfg",
                        "entity_name": "robot",
                        "resampling_time_range": [1.0, 1.0],
                        "ranges": {
                            "_target_": "unilab.base.scene.SceneCfg",
                            "model_file": "robot.xml",
                        },
                    }
                }
            },
        )


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
