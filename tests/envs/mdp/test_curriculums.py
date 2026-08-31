"""Tests for the generic stage-based curriculum terms (issue #1397).

Covers stage trigger timing, later-stage precedence, params merging, the
fail-closed validation error paths, and the CurriculumManager integration
(live term cfg mutation + logging snapshot extras).
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.envs import mdp
from unilab.managers import (
    CurriculumManager,
    CurriculumTermCfg,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)


def _reward(env: _FakeEnv, std: float, scale: float) -> np.ndarray:
    del scale
    return np.full(env.num_envs, std, dtype=np.float32)


def _termination(env: _FakeEnv, threshold: float) -> np.ndarray:
    del threshold
    return np.ones(env.num_envs, dtype=np.bool_)


class _FakeEnv:
    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self.scene = {}
        self.common_step_counter = 0
        self.reward_manager = RewardManager(
            {"pen": RewardTermCfg(func=_reward, weight=-0.1, params={"std": 0.5, "scale": 1.0})},
            self,
        )
        self.termination_manager = TerminationManager(
            {"energy": TerminationTermCfg(func=_termination, params={"threshold": np.inf})},
            self,
        )


@pytest.fixture
def fake_env() -> _FakeEnv:
    return _FakeEnv()


def _reward_curriculum_manager(env: _FakeEnv, stages: list[dict]) -> CurriculumManager:
    return CurriculumManager(
        {
            "pen_ramp": CurriculumTermCfg(
                func=mdp.reward_curriculum,
                params={"reward_name": "pen", "stages": stages},
            )
        },
        env,
    )


def _termination_curriculum_manager(env: _FakeEnv, stages: list[dict]) -> CurriculumManager:
    return CurriculumManager(
        {
            "energy_ramp": CurriculumTermCfg(
                func=mdp.termination_curriculum,
                params={"termination_name": "energy", "stages": stages},
            )
        },
        env,
    )


# Reward: weight stages.


def test_reward_weight_unchanged_before_threshold(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(fake_env, [{"step": 100, "weight": -1.0}])
    fake_env.common_step_counter = 99
    manager.compute()
    assert fake_env.reward_manager.get_term_cfg("pen").weight == pytest.approx(-0.1)


def test_reward_weight_applied_at_threshold(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(fake_env, [{"step": 100, "weight": -1.0}])
    fake_env.common_step_counter = 100
    manager.compute()
    assert fake_env.reward_manager.get_term_cfg("pen").weight == pytest.approx(-1.0)


def test_reward_weight_later_stage_wins(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(
        fake_env,
        [
            {"step": 0, "weight": -0.2},
            {"step": 100, "weight": -0.4},
            {"step": 400, "weight": -1.0},
        ],
    )
    fake_env.common_step_counter = 500
    manager.compute()
    assert fake_env.reward_manager.get_term_cfg("pen").weight == pytest.approx(-1.0)


def test_reward_weight_partial_application(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(
        fake_env,
        [
            {"step": 100, "weight": -0.4},
            {"step": 200, "weight": -1.0},
        ],
    )
    fake_env.common_step_counter = 150
    manager.compute()
    assert fake_env.reward_manager.get_term_cfg("pen").weight == pytest.approx(-0.4)


def test_step_zero_applies_on_first_compute(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(fake_env, [{"step": 0, "weight": -0.9}])
    manager.compute()
    assert fake_env.reward_manager.get_term_cfg("pen").weight == pytest.approx(-0.9)


def test_reapplication_is_idempotent(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(
        fake_env, [{"step": 0, "weight": -0.2}, {"step": 100, "params": {"std": 0.2}}]
    )
    fake_env.common_step_counter = 150
    manager.compute()
    manager.compute()
    term_cfg = fake_env.reward_manager.get_term_cfg("pen")
    assert term_cfg.weight == pytest.approx(-0.2)
    assert term_cfg.params["std"] == pytest.approx(0.2)


def test_ramped_weight_changes_computed_reward(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(fake_env, [{"step": 0, "weight": -1.0}])
    before = fake_env.reward_manager.compute(dt=1.0).copy()
    manager.compute()
    after = fake_env.reward_manager.compute(dt=1.0)
    np.testing.assert_allclose(after, before * 10.0)


# Reward: params stages.


def test_reward_params_updated_at_threshold(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(fake_env, [{"step": 100, "params": {"std": 0.2}}])
    fake_env.common_step_counter = 200
    manager.compute()
    assert fake_env.reward_manager.get_term_cfg("pen").params["std"] == pytest.approx(0.2)


def test_reward_params_unchanged_before_threshold(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(fake_env, [{"step": 100, "params": {"std": 0.2}}])
    manager.compute()
    assert fake_env.reward_manager.get_term_cfg("pen").params["std"] == pytest.approx(0.5)


def test_reward_stage_combines_weight_and_params(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(
        fake_env, [{"step": 100, "weight": -1.0, "params": {"std": 0.1, "scale": 2.0}}]
    )
    fake_env.common_step_counter = 100
    manager.compute()
    term_cfg = fake_env.reward_manager.get_term_cfg("pen")
    assert term_cfg.weight == pytest.approx(-1.0)
    assert term_cfg.params["std"] == pytest.approx(0.1)
    assert term_cfg.params["scale"] == pytest.approx(2.0)


# Termination: params and time_out stages.


def test_termination_params_follow_stages(fake_env: _FakeEnv) -> None:
    manager = _termination_curriculum_manager(
        fake_env,
        [
            {"step": 0, "params": {"threshold": 1000.0}},
            {"step": 100, "params": {"threshold": 700.0}},
            {"step": 400, "params": {"threshold": 400.0}},
        ],
    )
    manager.compute()
    assert fake_env.termination_manager.get_term_cfg("energy").params["threshold"] == 1000.0
    fake_env.common_step_counter = 200
    manager.compute()
    assert fake_env.termination_manager.get_term_cfg("energy").params["threshold"] == 700.0
    fake_env.common_step_counter = 500
    manager.compute()
    assert fake_env.termination_manager.get_term_cfg("energy").params["threshold"] == 400.0


def test_termination_time_out_stage_reroutes_done(fake_env: _FakeEnv) -> None:
    manager = _termination_curriculum_manager(fake_env, [{"step": 10, "time_out": True}])
    manager.compute()
    fake_env.termination_manager.compute()
    assert fake_env.termination_manager.terminated.all()
    assert not fake_env.termination_manager.time_outs.any()
    fake_env.common_step_counter = 10
    manager.compute()
    fake_env.termination_manager.compute()
    assert fake_env.termination_manager.time_outs.all()
    assert not fake_env.termination_manager.terminated.any()


# Validation error paths (fail-closed at manager construction).


def test_unknown_reward_param_key_raises(fake_env: _FakeEnv) -> None:
    with pytest.raises(KeyError, match="unknown param"):
        _reward_curriculum_manager(fake_env, [{"step": 0, "params": {"stdd": 0.2}}])


def test_unknown_termination_param_key_raises(fake_env: _FakeEnv) -> None:
    with pytest.raises(KeyError, match="unknown param"):
        _termination_curriculum_manager(fake_env, [{"step": 0, "params": {"threshld": 1.0}}])


def test_unknown_stage_field_raises(fake_env: _FakeEnv) -> None:
    with pytest.raises(AttributeError, match="Field 'weigth' does not exist"):
        _reward_curriculum_manager(fake_env, [{"step": 0, "weigth": -1.0}])


def test_unsorted_stages_raise(fake_env: _FakeEnv) -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        _reward_curriculum_manager(
            fake_env, [{"step": 200, "weight": -0.4}, {"step": 100, "weight": -1.0}]
        )


def test_unknown_target_term_raises(fake_env: _FakeEnv) -> None:
    with pytest.raises(ValueError, match="Term 'missing' not found"):
        CurriculumManager(
            {
                "bad": CurriculumTermCfg(
                    func=mdp.reward_curriculum,
                    params={"reward_name": "missing", "stages": [{"step": 0, "weight": -1.0}]},
                )
            },
            fake_env,
        )


def test_duplicate_step_stages_apply_in_order(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(
        fake_env,
        [
            {"step": 100, "weight": -0.4},
            {"step": 100, "params": {"std": 0.1}},
        ],
    )
    fake_env.common_step_counter = 200
    manager.compute()
    term_cfg = fake_env.reward_manager.get_term_cfg("pen")
    assert term_cfg.weight == pytest.approx(-0.4)
    assert term_cfg.params["std"] == pytest.approx(0.1)


# Logging snapshot.


def test_reset_extras_log_only_staged_scalar_keys(fake_env: _FakeEnv) -> None:
    manager = _reward_curriculum_manager(
        fake_env, [{"step": 100, "weight": -1.0, "params": {"std": 0.2}}]
    )
    fake_env.common_step_counter = 200
    manager.compute()
    extras = manager.reset()
    assert extras["Curriculum/pen_ramp/weight"] == pytest.approx(-1.0)
    assert extras["Curriculum/pen_ramp/std"] == pytest.approx(0.2)
    # Not staged by any stage -> not logged.
    assert "Curriculum/pen_ramp/scale" not in extras


def test_weight_not_logged_when_only_params_staged(fake_env: _FakeEnv) -> None:
    manager = _termination_curriculum_manager(
        fake_env, [{"step": 100, "params": {"threshold": 500.0}}]
    )
    fake_env.common_step_counter = 200
    manager.compute()
    extras = manager.reset()
    assert extras["Curriculum/energy_ramp/threshold"] == pytest.approx(500.0)
    assert "Curriculum/energy_ramp/weight" not in extras
