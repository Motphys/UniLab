from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.registry import ensure_registries

_CONF_DIR = Path(__file__).resolve().parents[2] / "conf"
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"

_OBS_DIM = 15
_ACTION_DIM = 2


def _make_env(num_envs: int = 2):
    ensure_registries()
    return registry.make("StewartBalance", sim_backend="motrix", num_envs=num_envs)


def test_stewart_env_uses_backend_contract() -> None:
    """The task must go through the backend contract, not raw sim internals."""
    source = (_SRC_DIR / "unilab" / "envs" / "manipulation" / "stewart" / "balance.py").read_text(
        encoding="utf-8"
    )
    assert "import motrixsim" not in source
    assert "import mujoco" not in source
    assert "_backend.model" not in source


def test_stewart_registered_backends() -> None:
    ensure_registries()
    registered = registry.list_registered_envs()
    assert "StewartBalance" in registered
    # motrix is the validated training backend; mujoco is construct/step-capable
    # (closed-loop stability tuning for mujoco is a follow-up).
    assert set(registered["StewartBalance"]["available_backends"]) == {"motrix", "mujoco"}


def test_stewart_motrix_owner_cfg_composes() -> None:
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=stewart_balance/motrix", "algo.num_envs=2"])
    assert cfg.training.task_name == "StewartBalance"
    assert cfg.training.sim_backend == "motrix"
    # Reward block maps onto the env's StewartRewardConfig.
    reward = OmegaConf.to_container(cfg.reward, resolve=True)
    assert set(reward) == {"scales", "fall_penalty"}
    assert set(reward["scales"]) == {"center", "progress", "still"}
    assert cfg.algo.obs_groups.actor == ["policy"]


def test_stewart_mujoco_owner_cfg_composes() -> None:
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=stewart_balance/mujoco", "algo.num_envs=2"])
    # Inherits the motrix owner config, only switching the backend.
    assert cfg.training.task_name == "StewartBalance"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.obs_groups.actor == ["policy"]


def test_stewart_env_constructs_and_steps() -> None:
    env = _make_env(num_envs=2)
    assert env.obs_groups_spec == {"obs": _OBS_DIM}
    assert env.action_space.shape == (_ACTION_DIM,)

    state = None
    for _ in range(20):
        state = env.step(np.zeros((2, _ACTION_DIM), dtype=np.float32))
    assert state is not None
    obs = state.obs["obs"]
    assert obs.shape == (2, _OBS_DIM)
    assert np.isfinite(obs).all()
    assert np.isfinite(state.reward).all()
    assert state.terminated.dtype == bool


def test_stewart_ik_holds_level_platform() -> None:
    """At zero action the IK should hold the plate near its home height (z=1)."""
    env = _make_env(num_envs=2)
    env.step(np.zeros((2, _ACTION_DIM), dtype=np.float32))  # triggers reset + calibration
    # Level-hold control should be ~zero leg displacement and ~1.1 m neutral legs.
    np.testing.assert_allclose(env._leg0, 1.1, atol=1e-2)
    ctrl = env._leg_ctrl_for_tilt(np.zeros((2, _ACTION_DIM), dtype=np.float32))
    assert np.allclose(ctrl, 0.0, atol=1e-2)
    for _ in range(20):
        env.step(np.zeros((2, _ACTION_DIM), dtype=np.float32))
    top_z = env._backend.get_body_pos_w(env._top_body_ids)[:, 0, 2]
    assert np.all(np.abs(top_z - 1.0) < 0.1)


@pytest.mark.slow
def test_stewart_solver_stable_under_random_actions() -> None:
    """Regression for the closed-loop solver blow-up (NotPositiveDefinite):
    the fall-radius margin + softened actuator kp must keep it stable."""
    env = _make_env(num_envs=8)
    np.random.seed(0)
    for _ in range(400):
        state = env.step(np.random.uniform(-1.0, 1.0, (8, _ACTION_DIM)).astype(np.float32))
        assert np.isfinite(state.obs["obs"]).all()
