"""Tests for env config completeness and env instantiation.

Config-attribute tests (non-slow) verify that config dataclasses expose every
attribute accessed by their paired env class, WITHOUT running a simulation.
They still require MuJoCo to be importable because the config and env classes
live in the same module file.

Slow tests actually call registry.make() and run a simulation step.
"""

from __future__ import annotations

import pytest

# The G1 env modules import create_backend → mujoco.batch_forward at the top
# level, so all tests in this file need a working MuJoCo installation.
pytest.importorskip("mujoco", reason="mujoco not installed")

# Some environments also use mujoco.batch_forward (G1 backend). Guard against
# partial MuJoCo installations where the base package installs but platform
# extensions fail (e.g. wrong libstdc++ version).
try:
    from mujoco import batch_forward as _  # noqa: F401
except Exception:
    pytest.skip("mujoco.batch_forward not available (platform/libstdc++ issue)", allow_module_level=True)

from unilab.utils.algo_utils import ensure_registries  # noqa: E402

# ---------------------------------------------------------------------------
# Non-slow: config attribute completeness (no env.step(), no MuJoCo sim)
# ---------------------------------------------------------------------------


def test_g1_joystick_ppo_cfg_has_obs_config():
    """G1JoystickPPOCfg must have obs_config (baseline sanity check)."""
    from unilab.envs.locomotion.g1.joystick import G1JoystickPPOCfg

    cfg = G1JoystickPPOCfg()
    assert hasattr(cfg, "obs_config"), "G1JoystickPPOCfg missing obs_config"
    assert hasattr(cfg.obs_config, "obs_dict")
    assert hasattr(cfg.obs_config, "actor_obs")


def test_g1_joystick_sac_cfg_has_obs_config():
    """G1JoystickSACCfg must have obs_config because G1WalkTaskMjSAC inherits
    _init_obs_space() from G1JoystickPPO which reads self._cfg.obs_config.

    Regression test for:
        AttributeError: 'G1JoystickSACCfg' object has no attribute 'obs_config'
    """
    from unilab.envs.locomotion.g1.joystick_sac import G1JoystickSACCfg

    cfg = G1JoystickSACCfg()
    assert hasattr(cfg, "obs_config"), (
        "G1JoystickSACCfg must declare obs_config so that the inherited "
        "G1JoystickPPO._init_obs_space() can resolve self._cfg.obs_config"
    )
    assert hasattr(cfg.obs_config, "obs_dict")
    assert hasattr(cfg.obs_config, "actor_obs")


def test_g1_joystick_sac_obs_config_total_dim():
    """obs_config total dim must match what _compute_obs actually concatenates.

    G1JoystickPPO._compute_obs outputs (G1 has 29 DoF):
        linvel(3) + gyro(3) + gravity(3) + diff(29) + dof_vel(29)
        + last_actions(29) + command(3) + gait_phase(2) = 101
    """
    from unilab.envs.locomotion.g1.joystick_sac import G1JoystickSACCfg

    cfg = G1JoystickSACCfg()
    total = sum(cfg.obs_config.obs_dict.values())
    assert total == 101, (
        f"obs_config total dim is {total}, expected 101. "
        "obs_config.obs_dict does not match _compute_obs output."
    )


# ---------------------------------------------------------------------------
# Slow: actual env instantiation (runs MuJoCo physics)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "env_name",
    [
        "Go1JoystickFlatTerrain",
        "Go2JoystickFlatTerrain",
        "G1JoystickFlatTerrain",
        "G1WalkTaskMjSAC",
    ],
)
def test_locomotion_env_instantiates(env_name: str):
    """Every registered locomotion env must be constructible with num_envs=1
    and expose valid observation/action spaces."""
    ensure_registries()
    from unilab.base import registry

    env = registry.make(env_name, num_envs=1, sim_backend="mujoco")
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        assert obs_space is not None
        assert act_space is not None
        assert obs_space.shape is not None and len(obs_space.shape) == 1
        assert act_space.shape is not None and len(act_space.shape) == 1
        assert obs_space.shape[0] > 0
        assert act_space.shape[0] > 0
    finally:
        env.close()
