"""Unit contracts for the G1 Manager-Based terms (fake-env, no simulator)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.managers import ObservationTermCfg, RewardTermCfg
from unilab.tasks.locomotion.g1 import manager_terms as g1_terms
from unilab.tasks.locomotion.g1.manager_terms import (
    G1GaitPhase,
    G1PenaltyCurriculum,
    G1VelocityCommandCfg,
    compute_feet_phase_contact_targets,
    compute_feet_phase_height_targets,
)

# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow


def _fake_scene(sensor_data: dict[str, np.ndarray]) -> Any:
    class _Scene(dict):
        def bind_sensor_data(self, names):
            arrays = [sensor_data[name] for name in names]
            concatenated = np.concatenate(arrays, axis=1)
            dimensions = tuple(array.shape[1] for array in arrays)
            return SimpleNamespace(
                dimensions=dimensions,
                backend_type="fake",
                read=lambda: concatenated,
            )

    return _Scene()


class _FakeEnv:
    """Weakref-able stand-in for ManagerBasedRlEnv in term unit tests."""

    def __init__(
        self,
        sensor_data: dict[str, np.ndarray],
        *,
        num_envs: int = 2,
        counter: int = 0,
        command: np.ndarray | None = None,
    ) -> None:
        commands = {"twist": command if command is not None else np.zeros((num_envs, 3))}
        self.num_envs = num_envs
        self.common_step_counter = counter
        self.step_dt = 0.02
        self.rng = np.random.default_rng(0)
        self.scene = _fake_scene(sensor_data)
        self.command_manager = SimpleNamespace(get_command=lambda name: commands[name])


def _fake_env(
    sensor_data: dict[str, np.ndarray],
    *,
    num_envs: int = 2,
    counter: int = 0,
    command: np.ndarray | None = None,
) -> Any:
    return _FakeEnv(sensor_data, num_envs=num_envs, counter=counter, command=command)


def test_gait_phase_advances_with_counter_and_resamples_per_init_mode():
    env = _fake_env({}, num_envs=4)
    term = G1GaitPhase(
        ObservationTermCfg(
            func=G1GaitPhase, params={"frequency": 1.5, "init_mode": "offset_phase"}
        ),
        cast(Any, env),
    )

    np.testing.assert_array_equal(term(env), np.zeros((4, 2)))
    term.reset(np.arange(4, dtype=np.int32))
    phase = term(env)
    np.testing.assert_allclose(phase[:, 1] - phase[:, 0], np.pi, rtol=1.0e-6)
    assert np.all(phase[:, 0] >= 0.0) and np.all(phase[:, 0] < 2.0 * np.pi)

    env.common_step_counter = 1
    advanced = term(env)
    delta = 2.0 * np.pi * 1.5 * 0.02
    np.testing.assert_allclose(advanced, np.fmod(phase + delta, 2.0 * np.pi), rtol=1.0e-6)

    # Idempotent per counter: reading twice does not advance twice.
    np.testing.assert_array_equal(term(env), advanced)

    env.common_step_counter = 0
    with pytest.raises(ValueError, match="cannot move backwards"):
        term(env)


def test_gait_phase_independent_mode_samples_feet_independently():
    env = _fake_env({}, num_envs=64)
    term = G1GaitPhase(
        ObservationTermCfg(func=G1GaitPhase, params={"frequency": 1.5, "init_mode": "independent"}),
        cast(Any, env),
    )
    term.reset(np.arange(64, dtype=np.int32))
    phase = term(env)
    assert not np.allclose(phase[:, 1] - phase[:, 0], np.pi)


def test_bezier_targets_match_legacy_reference_values():
    phase = np.array([[0.0, np.pi], [np.pi / 2.0, 3.0 * np.pi / 2.0]])
    left, right = compute_feet_phase_height_targets(phase, 0.09)
    # phi=0 -> x=0.5 boundary -> stance peak; phi=pi -> x=0 -> zero;
    # phi=pi/2 -> x=0.75 -> swing midpoint 0.045.
    np.testing.assert_allclose(left, [0.09, 0.045], atol=1.0e-7)
    np.testing.assert_allclose(right, [0.0, 0.045], atol=1.0e-7)
    left_contact, right_contact = compute_feet_phase_contact_targets(phase, 0.09)
    np.testing.assert_array_equal(left_contact, [False, True])
    np.testing.assert_array_equal(right_contact, [True, True])


def test_feet_phase_reward_is_gated_by_forward_speed():
    sensors = {
        "left_foot_pos": np.zeros((2, 3), dtype=np.float32),
        "right_foot_pos": np.zeros((2, 3), dtype=np.float32),
        "pelvis_local_linvel": np.array([[0.01, 0.0, 0.0], [0.10, 0.0, 0.0]], dtype=np.float32),
    }
    env = _fake_env(sensors)
    term = g1_terms.feet_phase(
        RewardTermCfg(
            func=g1_terms.feet_phase,
            weight=1.0,
            params={
                "frequency": 1.5,
                "swing_height": 0.09,
                "tracking_sigma": 0.008,
                "min_forward_speed": 0.05,
                "command_name": "twist",
            },
        ),
        cast(Any, env),
    )

    reward = term(env)

    assert reward[0] == pytest.approx(0.0)
    assert reward[1] > 0.0


def test_feet_double_stance_masks_on_forward_command():
    sensors = {
        "left_foot_pos": np.zeros((2, 3), dtype=np.float32),
        "right_foot_pos": np.zeros((2, 3), dtype=np.float32),
        "pelvis_local_linvel": np.zeros((2, 3), dtype=np.float32),
        **{f"left_foot_contact_{i}": np.ones((2, 1)) for i in range(4)},
        **{f"right_foot_contact_{i}": np.ones((2, 1)) for i in range(4)},
    }
    command = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = _fake_env(sensors, command=command)
    term = g1_terms.feet_double_stance(
        RewardTermCfg(
            func=g1_terms.feet_double_stance,
            weight=-1.0,
            params={"frequency": 1.5, "command_name": "twist"},
        ),
        cast(Any, env),
    )

    np.testing.assert_array_equal(term(env), [1.0, 0.0])


def _curriculum_env(weights: dict[str, float], num_envs: int = 4) -> Any:
    cfgs = {name: SimpleNamespace(weight=value) for name, value in weights.items()}
    return SimpleNamespace(
        num_envs=num_envs,
        reward_manager=SimpleNamespace(
            active_terms=list(weights),
            get_term_cfg=lambda name: cfgs[name],
        ),
        reset_buf=np.zeros(num_envs, dtype=np.bool_),
        episode_length_buf=np.zeros(num_envs, dtype=np.int64),
        rng=np.random.default_rng(0),
    )


def test_penalty_curriculum_scales_only_negative_weights_and_tracks_episodes():
    env = _curriculum_env({"pose": -0.5, "alive": 10.0, "penalty_orientation": -10.0})
    term = G1PenaltyCurriculum(
        RewardTermCfg(
            func=G1PenaltyCurriculum,
            weight=1.0,
            params={"initial_scale": 0.5, "min_scale": 0.5, "max_scale": 1.0},
        ),
        cast(Any, env),
    )

    assert env.reward_manager.get_term_cfg("pose").weight == pytest.approx(-0.25)
    assert env.reward_manager.get_term_cfg("penalty_orientation").weight == pytest.approx(-5.0)
    assert env.reward_manager.get_term_cfg("alive").weight == pytest.approx(10.0)

    # Short episodes (< level_down_threshold=150 default) shrink the scale,
    # clamped at min_scale.
    env.reset_buf[:] = True
    env.episode_length_buf[:] = 10
    state = term(cast(Any, env), np.arange(4, dtype=np.int32))
    assert state["average_episode_length"] == pytest.approx(10.0)
    assert state["penalty_scale"] == pytest.approx(0.5)
    assert env.reward_manager.get_term_cfg("pose").weight == pytest.approx(-0.25)

    # Long episodes (> level_up_threshold=750 default) relax the scale.
    env.episode_length_buf[:] = 1000
    state = term(cast(Any, env), np.arange(4, dtype=np.int32))
    assert state["penalty_scale"] == pytest.approx(0.5 * (1.0 + 0.001))
    assert env.reward_manager.get_term_cfg("pose").weight == pytest.approx(
        -0.5 * 0.5 * (1.0 + 0.001)
    )


def test_penalty_curriculum_repeated_construction_never_mutates_source_cfg():
    """Regression guard for the legacy shared-override mutation.

    The legacy PenaltyCurriculum halved the shared override dict in place on
    every env construction (two probe envs + the collector in each offpolicy
    runner), so collectors silently trained at 1/8 of the configured penalty
    weights. The manager runtime must isolate each construction from the
    source config so repeated env builds keep identical effective weights.
    """
    from unilab.managers import RewardManager

    source_cfg = {
        "pose": RewardTermCfg(func=lambda env: np.zeros(env.num_envs), weight=-0.5),
        "alive": RewardTermCfg(func=lambda env: np.zeros(env.num_envs), weight=10.0),
    }
    effective_weights: list[float] = []
    for _ in range(3):  # legacy offpolicy runners built probe + probe + collector
        reward_manager = RewardManager(source_cfg, cast(Any, SimpleNamespace(num_envs=4)))
        env = SimpleNamespace(
            num_envs=4,
            reward_manager=reward_manager,
            reset_buf=np.zeros(4, dtype=np.bool_),
            episode_length_buf=np.zeros(4, dtype=np.int64),
            rng=np.random.default_rng(0),
        )
        G1PenaltyCurriculum(
            RewardTermCfg(
                func=G1PenaltyCurriculum,
                weight=1.0,
                params={"initial_scale": 0.125, "min_scale": 0.125, "max_scale": 0.25},
            ),
            cast(Any, env),
        )
        effective_weights.append(reward_manager.get_term_cfg("pose").weight)

    assert source_cfg["pose"].weight == -0.5
    assert source_cfg["alive"].weight == 10.0
    for weight in effective_weights:
        assert weight == pytest.approx(-0.0625)


def test_penalty_curriculum_shrinks_scale_below_initial_when_min_allows():
    env = _curriculum_env({"pose": -0.5})
    term = G1PenaltyCurriculum(
        RewardTermCfg(
            func=G1PenaltyCurriculum,
            weight=1.0,
            params={"initial_scale": 0.5, "min_scale": 0.0, "max_scale": 1.0},
        ),
        cast(Any, env),
    )

    env.reset_buf[:] = True
    env.episode_length_buf[:] = 10
    state = term(cast(Any, env), np.arange(4, dtype=np.int32))
    assert state["penalty_scale"] == pytest.approx(0.5 * (1.0 - 0.001))
    assert env.reward_manager.get_term_cfg("pose").weight == pytest.approx(
        -0.5 * 0.5 * (1.0 - 0.001)
    )


def test_velocity_command_dead_zone_zeroes_small_planar_commands():
    env = SimpleNamespace(
        num_envs=64,
        rng=np.random.default_rng(0),
        step_dt=0.02,
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_lin_vel_b=np.zeros((64, 3)),
                    root_link_ang_vel_b=np.zeros((64, 3)),
                    heading_w=np.zeros(64),
                )
            )
        },
    )
    cfg = G1VelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(20.0, 20.0),
        ranges=G1VelocityCommandCfg.Ranges(
            lin_vel_x=(-0.15, 0.15),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(0.0, 0.0),
        ),
    )
    term = cfg.build(cast(Any, env))
    term._resample_command(np.arange(64, dtype=np.int32))

    planar_norm = np.linalg.norm(term.vel_command_b[:, :2], axis=1)
    assert np.all((planar_norm == 0.0) | (planar_norm > 0.2))
    assert np.any(planar_norm == 0.0)


def test_velocity_command_fails_closed_on_heading_command():
    cfg = G1VelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(20.0, 20.0),
        heading_command=True,
        ranges=G1VelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-3.14, 3.14),
        ),
    )
    env = SimpleNamespace(num_envs=1, rng=np.random.default_rng(0), step_dt=0.02, scene={})
    with pytest.raises(NotImplementedError, match="heading command"):
        cfg.build(cast(Any, env))


def test_g1_hot_paths_use_only_cached_runtime_objects():
    for term in (
        G1GaitPhase,
        g1_terms.feet_phase,
        g1_terms.feet_phase_contrast,
        g1_terms.feet_phase_contact,
        g1_terms.feet_double_stance,
        g1_terms.feet_air_time,
        g1_terms.forward_progress,
        g1_terms.under_speed,
        g1_terms.g1_tilt_exceeded,
        g1_terms.penalty_feet_ori,
        g1_terms.penalty_close_feet_xy,
        g1_terms.G1PenaltyCurriculum,
    ):
        source = inspect.getsource(term.__call__)
        for forbidden in (
            "ASSETS_ROOT_PATH",
            "model_file",
            "getattr(",
            "hasattr(",
            "._backend",
        ):
            assert forbidden not in source, f"{term.__name__} hot path references {forbidden}"
