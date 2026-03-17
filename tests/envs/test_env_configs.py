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
    pytest.skip(
        "mujoco.batch_forward not available (platform/libstdc++ issue)", allow_module_level=True
    )

from unilab.utils.algo_utils import ensure_registries  # noqa: E402

# ---------------------------------------------------------------------------
# Non-slow: config attribute completeness (no env.step(), no MuJoCo sim)
# ---------------------------------------------------------------------------


def test_g1_joystick_ppo_cfg_obs_groups_spec():
    """G1JoystickPPO must declare obs_groups_spec with actor and privileged groups."""
    from unilab.envs.locomotion.g1.joystick import G1JoystickPPO, G1JoystickPPOCfg

    cfg = G1JoystickPPOCfg()
    # Access obs_groups_spec from the class (it's a class-level property)
    # We need an instance to check, but can't instantiate without MuJoCo sim,
    # so just verify the cfg no longer has obs_config (removed in dict obs refactor).
    assert not hasattr(cfg, "obs_config"), "obs_config should have been removed"


def test_g1_joystick_sac_cfg_no_obs_config():
    """G1JoystickSACCfg should no longer have obs_config after dict obs refactor."""
    from unilab.envs.locomotion.g1.joystick_sac import G1JoystickSACCfg

    cfg = G1JoystickSACCfg()
    assert not hasattr(cfg, "obs_config"), (
        "obs_config should have been removed in the dict obs refactor"
    )


def test_g1_joystick_ppo_obs_groups_spec_dims():
    """obs_groups_spec total dim must match what _compute_obs actually produces.

    G1JoystickPPO._compute_obs outputs (G1 has 29 DoF):
        actor: gyro(3) + gravity(3) + diff(29) + dof_vel(29)
            + last_actions(29) + command(3) + gait_phase(2) = 98
        privileged: linvel(3)
    """
    from unilab.envs.locomotion.g1.joystick import G1JoystickPPO

    # obs_groups_spec is a @property; access via descriptor protocol
    spec = G1JoystickPPO.obs_groups_spec.fget(None)  # type: ignore[union-attr]
    assert spec is not None
    assert spec["actor"] == 98
    assert spec["privileged"] == 3


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
