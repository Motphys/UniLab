"""Tests for env config completeness and env instantiation.

Config-attribute tests (non-slow) verify that config dataclasses expose every
attribute accessed by their paired env class, WITHOUT running a simulation.
They still require MuJoCo to be importable because the config and env classes
live in the same module file.

Slow tests actually call registry.make() and run reset + step.
"""

from __future__ import annotations

import numpy as np
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
    from unilab.envs.locomotion.g1.joystick import G1JoystickPPOCfg

    cfg = G1JoystickPPOCfg()
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
# Slow: env instantiation + reset + step (runs MuJoCo physics)
# ---------------------------------------------------------------------------

# Environments that don't need special config overrides
_STANDARD_ENVS = [
    "Go1JoystickFlatTerrain",
    "Go2JoystickFlatTerrain",
    "G1JoystickFlatTerrain",
    "G1WalkTaskMjSAC",
    "AllegroInhandRotation",
    "AllegroInhandRotationSac",
]


@pytest.mark.slow
@pytest.mark.parametrize("env_name", _STANDARD_ENVS)
def test_env_reset_and_step(
    env_name: str,
    default_go1_reward_config,
    default_go2_reward_config,
    default_g1_reward_config,
    default_g1_sac_reward_config,
):
    """Every registered env must be constructible, resetable, and steppable.

    Verifies:
    - observation/action spaces are valid
    - init_state + reset produces dict obs with correct keys and shapes
    - step with zero actions produces dict obs, scalar reward, bool done
    """
    ensure_registries()
    from unilab.base import registry

    # Provide reward_config for locomotion envs
    env_cfg_override = None
    if "Go1" in env_name:
        env_cfg_override = {"reward_config": default_go1_reward_config}
    elif "Go2" in env_name:
        env_cfg_override = {"reward_config": default_go2_reward_config}
    elif "G1WalkTaskMjSAC" in env_name:
        env_cfg_override = {"reward_config": default_g1_sac_reward_config}
    elif "G1" in env_name:
        env_cfg_override = {"reward_config": default_g1_reward_config}

    env = registry.make(env_name, num_envs=2, sim_backend="mujoco", env_cfg_override=env_cfg_override)
    try:
        # 1. Spaces
        obs_space = env.observation_space
        act_space = env.action_space
        assert obs_space.shape is not None and obs_space.shape[0] > 0
        assert act_space.shape is not None and act_space.shape[0] > 0

        # obs_groups_spec must sum to observation_space total dim
        spec = env.obs_groups_spec
        assert isinstance(spec, dict)
        assert sum(spec.values()) == obs_space.shape[0]

        # 2. Reset
        state = env.init_state()
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert key in state.obs, f"obs missing group '{key}'"
            assert state.obs[key].shape == (2, dim), (
                f"obs['{key}'] shape mismatch: {state.obs[key].shape} != (2, {dim})"
            )

        # 3. Step with zero actions
        actions = np.zeros((2, act_space.shape[0]))
        state = env.step(actions)
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert state.obs[key].shape == (2, dim)
        assert state.reward.shape == (2,)
        assert state.done.shape == (2,)
    finally:
        env.close()


@pytest.mark.slow
def test_g1_motion_tracking_reset_and_step():
    """G1MotionTracking needs a motion_file — skip if not available."""
    ensure_registries()
    from pathlib import Path

    from unilab.base import registry

    # Look for any motion file in the expected location
    motion_dir = (
        Path(__file__).parents[2]
        / "src"
        / "unilab"
        / "envs"
        / "locomotion"
        / "g1"
        / "motion_tracking"
        / "motions"
    )
    if not motion_dir.exists():
        pytest.skip(f"Motion data directory not found: {motion_dir}")

    npz_files = list(motion_dir.glob("*.npz"))
    if not npz_files:
        pytest.skip(f"No .npz motion files in {motion_dir}")

    motion_file = str(npz_files[0])
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={"motion_file": motion_file},
    )
    try:
        spec = env.obs_groups_spec
        assert isinstance(spec, dict)
        assert "actor" in spec
        assert sum(spec.values()) == env.observation_space.shape[0]

        state = env.init_state()
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert state.obs[key].shape == (2, dim)

        actions = np.zeros((2, env.action_space.shape[0]))
        state = env.step(actions)
        assert isinstance(state.obs, dict)
        assert state.reward.shape == (2,)
        assert state.done.shape == (2,)
    finally:
        env.close()
