"""Hydra/registry evidence for the minimal MicroDuck Manager-Based owners."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unisim.backend.mjwarp.dependencies import load_mjwarp_dependencies

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.envs.mdp import (
    LifecycleCounterRecorder,
    UniformVelocityCommandCfg,
    root_height,
)
from unilab.tasks.locomotion.microduck.manager_terms import (
    GroundPickPhaseCommand,
    GroundPickPhaseCommandCfg,
    SitStandCommand,
    SitStandCommandCfg,
    phase_height_tracking,
    posture_height_tracking,
)

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "ppo"


def _materialize(task_owner: str) -> tuple[Any, ManagerBasedRlEnvCfg]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task_owner}"])
    registry.ensure_registries()
    task_name = str(cfg.training.task_name)
    env_cfg = registry.materialize_env_config(task_name)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override(),
    )
    env_cfg.validate()
    return cfg, env_cfg


def test_minimal_microduck_owners_are_registered_as_mjwarp_ppo_profiles() -> None:
    registry.ensure_registries()
    registered = registry.list_registered_envs()
    assert registered["MicroduckGroundPickFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mjwarp"],
    }
    assert registered["MicroduckSitStandFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mjwarp"],
    }


def test_ground_pick_owner_exercises_phase_metrics_recorder_and_reward_terms() -> None:
    cfg, env_cfg = _materialize("microduck_ground_pick_flat/mjwarp")

    assert cfg.training.task_name == "MicroduckGroundPickFlat"
    assert cfg.training.sim_backend == "mjwarp"
    assert isinstance(env_cfg.commands["twist"], GroundPickPhaseCommandCfg)
    assert tuple(env_cfg.metrics) == ("root_height",)
    assert tuple(env_cfg.recorders) == ("lifecycle",)
    assert env_cfg.metrics["root_height"].func is root_height
    assert env_cfg.recorders["lifecycle"].func is LifecycleCounterRecorder
    assert env_cfg.rewards["phase_height"].func is phase_height_tracking


def test_sitstand_owner_exercises_binary_posture_command() -> None:
    cfg, env_cfg = _materialize("microduck_sitstand_flat/mjwarp")

    assert cfg.training.task_name == "MicroduckSitStandFlat"
    assert cfg.training.sim_backend == "mjwarp"
    command = env_cfg.commands["twist"]
    assert isinstance(command, SitStandCommandCfg)
    assert command.sit_prob == 0.5
    assert command.ramp_s == 2.0
    assert env_cfg.rewards["posture_height"].func is posture_height_tracking
    assert env_cfg.rewards["posture_height"].weight == 1.0


def _velocity_ranges() -> UniformVelocityCommandCfg.Ranges:
    return UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
    )


def _command_env(num_envs: int = 2) -> SimpleNamespace:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=np.asarray(
                [[0.0, 0.0, 0.115], [0.0, 0.0, 0.060]],
                dtype=np.float32,
            ),
            root_link_lin_vel_b=np.zeros((num_envs, 3), dtype=np.float32),
            root_link_ang_vel_b=np.zeros((num_envs, 3), dtype=np.float32),
        )
    )
    return SimpleNamespace(
        num_envs=num_envs,
        rng=np.random.default_rng(11),
        scene={"robot": robot},
        step_dt=0.02,
        episode_length_buf=np.zeros(num_envs, dtype=np.int64),
    )


def test_ground_pick_phase_is_continuous_and_does_not_resample() -> None:
    env = _command_env()
    cfg = GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 1.0),
        ranges=_velocity_ranges(),
        period=4.0,
        randomize_phase=False,
    )
    term = cfg.build(env)
    assert isinstance(term, GroundPickPhaseCommand)

    ids = np.asarray([0, 1], dtype=np.int32)
    term.reset(ids)
    np.testing.assert_array_equal(term.phase, 0.0)
    np.testing.assert_array_equal(term.command_counter, 0)

    term.compute(1.0)
    np.testing.assert_allclose(term.phase, 0.25)
    np.testing.assert_allclose(term.command, [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-6)
    np.testing.assert_array_equal(term.command_counter, 0)

    with pytest.raises(ValueError, match="non-finite dt"):
        term.compute(float("nan"))


def test_sit_stand_command_slews_from_measured_height_to_sampled_target() -> None:
    env = _command_env()
    cfg = SitStandCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 1.0),
        ranges=_velocity_ranges(),
        sit_prob=1.0,
        ramp_s=2.0,
        sit_z=0.060,
        stand_z=0.115,
    )
    term = cfg.build(env)
    assert isinstance(term, SitStandCommand)

    ids = np.asarray([0, 1], dtype=np.int32)
    term.reset(ids)
    term.compute(0.0, env_ids=ids)
    np.testing.assert_allclose(term.command[:, 0], 1.0)
    np.testing.assert_allclose(term.alpha, [0.0, 1.0])

    term.compute(1.0)
    np.testing.assert_allclose(term.alpha, [0.5, 1.0])


def test_phase_and_posture_height_rewards_use_their_explicit_command_contracts() -> None:
    env = _command_env()
    ids = np.asarray([0, 1], dtype=np.int32)
    phase = GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 1.0),
        ranges=_velocity_ranges(),
        period=4.0,
        randomize_phase=False,
    ).build(env)
    phase.reset(ids)
    phase.compute(1.0)
    env.command_manager = SimpleNamespace(
        get_command=lambda name: phase.command,
        get_term=lambda name: phase,
    )
    phase_reward = phase_height_tracking(env)
    assert phase_reward[0] < 1.0e-2
    assert phase_reward[1] == pytest.approx(1.0)

    posture = SitStandCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 1.0),
        ranges=_velocity_ranges(),
        sit_prob=1.0,
        ramp_s=2.0,
        sit_z=0.060,
        stand_z=0.115,
    ).build(env)
    posture.reset(ids)
    posture.compute(0.0, env_ids=ids)
    env.command_manager = SimpleNamespace(
        get_command=lambda name: posture.command,
        get_term=lambda name: posture,
    )
    np.testing.assert_allclose(posture_height_tracking(env), 1.0)


def test_posture_commands_reject_invalid_tuning() -> None:
    with pytest.raises(ValueError, match="period must be finite"):
        GroundPickPhaseCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0, 1.0),
            ranges=_velocity_ranges(),
            period=float("nan"),
        ).build(_command_env())

    with pytest.raises(TypeError, match="ramp_s must be a real number"):
        SitStandCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0, 1.0),
            ranges=_velocity_ranges(),
            ramp_s=True,
        ).build(_command_env())


@pytest.mark.slow
def test_minimal_microduck_owners_reset_and_step_on_mjwarp() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.skip("minimal MicroDuck mjwarp smoke requires an active CUDA Warp device")

    for owner in ("microduck_ground_pick_flat/mjwarp", "microduck_sitstand_flat/mjwarp"):
        cfg, _ = _materialize(owner)
        task_name = str(cfg.training.task_name)
        env = registry.make(
            task_name,
            sim_backend="mjwarp",
            num_envs=2,
            env_cfg_override=BackendAdapter(
                cfg,
                root_dir=ROOT_DIR,
                algo_name="ppo",
            ).build_task_env_cfg_override(),
        )
        try:
            obs, _ = env.reset()
            assert np.isfinite(obs["obs"]).all()
            action = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
            state = env.step(action)
            assert np.isfinite(state.obs["obs"]).all()
            assert np.isfinite(state.reward).all()
            trace = env.recorder_manager.get_term("lifecycle")
            assert trace.post_reset_count == 2
            assert trace.post_step_count == 1
        finally:
            env.close()
