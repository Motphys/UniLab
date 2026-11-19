"""Contract and runtime evidence for the stage-curriculum demo owner YAML.

The demo owner YAML (``tests/fixtures/mjlab_cartpole/conf/stage_curriculum.yaml``)
reuses the pinned cartpole task and declares one ladder per public curriculum
term from ``unilab.envs.mdp.curriculums`` (issue #1397).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from tests.fixtures.mjlab_cartpole import FIXTURE_ENV_NAME
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg, mdp

ROOT_DIR = Path(__file__).parents[2]
FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures" / "mjlab_cartpole"


def _materialize() -> tuple[ManagerBasedRlEnvCfg, dict[str, Any]]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(FIXTURE_DIR / "conf"), version_base="1.3"):
        hydra_cfg: DictConfig = compose("stage_curriculum")
    override = EnvConfigAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(FIXTURE_ENV_NAME)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return env_cfg, override


def test_stage_curriculum_demo_owner_materializes_declared_ladders() -> None:
    env_cfg, _ = _materialize()

    assert list(env_cfg.curriculum) == ["smooth_reward_weight", "pole_tilt_limit"]
    weight_term = env_cfg.curriculum["smooth_reward_weight"]
    assert weight_term.func is mdp.reward_curriculum
    assert weight_term.params["reward_name"] == "smooth_reward"
    stages = weight_term.params["stages"]
    assert [stage["step"] for stage in stages] == [0, 2, 4]
    assert [stage["weight"] for stage in stages] == pytest.approx([1.0, 0.5, 0.25])
    tilt_term = env_cfg.curriculum["pole_tilt_limit"]
    assert tilt_term.func is mdp.termination_curriculum
    assert tilt_term.params["termination_name"] == "pole_tilt"
    assert tilt_term.params["stages"][1]["params"] == {"limit_angle": 0.2}

    assert list(env_cfg.terminations) == ["time_out", "pole_tilt"]
    assert env_cfg.terminations["pole_tilt"].func is mdp.bad_orientation
    assert env_cfg.terminations["pole_tilt"].params["limit_angle"] == pytest.approx(0.4)

    for term in env_cfg.curriculum.values():
        assert not OmegaConf.is_config(term.params["stages"])


def test_stage_curriculum_demo_real_backend_runs_packed_reset_lifecycle() -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )
    _, override = _materialize()
    # These packed-reset lifecycle tests run the eager selected-row path on CPU.
    override["autoreset_cuda_graph"] = False
    env = registry.make(
        FIXTURE_ENV_NAME,
        sim_backend="mujoco",
        env_cfg_override=override,
        num_envs=4,
    )
    try:
        state = env.init_state()
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert state.obs["obs"].shape[0] == 4
        obs, _ = env.reset(torch.tensor([1], dtype=torch.int64, device=env.device))
        assert obs["obs"].shape[0] == 4
    finally:
        env.close()


def test_stage_curriculum_demo_owner_invalid_stages_fail_closed() -> None:
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )
    _, override = _materialize()
    override["curriculum"]["smooth_reward_weight"]["params"]["stages"] = [
        {"step": 4, "weight": 0.5},
        {"step": 2, "weight": 0.25},
    ]
    # Keep the fail-closed assertion reachable: run the eager path on CPU.
    override["autoreset_cuda_graph"] = False
    with pytest.raises(ValueError, match="nondecreasing"):
        registry.make(
            FIXTURE_ENV_NAME,
            sim_backend="mujoco",
            env_cfg_override=override,
            num_envs=2,
        )
