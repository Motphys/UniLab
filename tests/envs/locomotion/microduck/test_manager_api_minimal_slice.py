"""Hydra/registry evidence for the minimal MicroDuck Manager-Based owners."""

from __future__ import annotations

from pathlib import Path
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
from unilab.envs.mdp import GroundPickPhaseCommandCfg, SitStandCommandCfg
from unilab.tasks.locomotion.microduck.manager_terms import (
    MicroduckTraceRecorder,
    posture_height_tracking,
    root_height_metric,
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
    assert env_cfg.metrics["root_height"].func is root_height_metric
    assert env_cfg.recorders["lifecycle"].func is MicroduckTraceRecorder
    assert env_cfg.rewards["posture_height"].func is posture_height_tracking


def test_sitstand_owner_exercises_binary_posture_command() -> None:
    cfg, env_cfg = _materialize("microduck_sitstand_flat/mjwarp")

    assert cfg.training.task_name == "MicroduckSitStandFlat"
    assert cfg.training.sim_backend == "mjwarp"
    command = env_cfg.commands["twist"]
    assert isinstance(command, SitStandCommandCfg)
    assert command.sit_prob == 0.5
    assert command.ramp_s == 2.0
    assert env_cfg.rewards["posture_height"].weight == 1.0


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
