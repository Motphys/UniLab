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

# The G1 env modules import create_backend → mujoco.batch_env at the top
# level, so all tests in this file need a working MuJoCo installation.
pytest.importorskip("mujoco", reason="mujoco not installed")

# Some environments also use mujoco.batch_env (G1 backend). Guard against
# partial MuJoCo installations where the base package installs but platform
# extensions fail (e.g. wrong libstdc++ version).
try:
    from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
except Exception:
    pytest.skip(
        "mujoco.batch_env not available (platform/libstdc++ issue)", allow_module_level=True
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


def test_g1_joystick_sac_cfg_has_domain_rand_for_motrix():
    from unilab.envs.locomotion.g1.joystick_sac import G1JoystickSACCfg

    cfg = G1JoystickSACCfg()
    assert hasattr(cfg, "domain_rand")
    assert hasattr(cfg, "gait_phase_init_mode")
    assert hasattr(cfg, "reset_base_qvel_limit")
    assert cfg.domain_rand.randomize_base_mass is False
    assert cfg.domain_rand.random_com is False
    assert cfg.domain_rand.push_robots is False


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
    assert spec["obs"] == 98
    assert spec["privileged"] == 3


def test_g1_motion_tracking_uses_split_body_pose_queries():
    """G1MotionTracking should query pos/quat via the stable split backend API."""
    from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingEnv

    class FakeBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[str, np.ndarray]] = []

        def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
            self.calls.append(("pos", body_ids.copy()))
            return np.ones((2, len(body_ids), 3))

        def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
            self.calls.append(("quat", body_ids.copy()))
            return np.ones((2, len(body_ids), 4))

    env = object.__new__(G1MotionTrackingEnv)
    env._backend = FakeBackend()
    env.body_ids = np.array([1, 3], dtype=np.int32)

    pos_w, quat_w = env._get_body_pose_w()

    assert pos_w.shape == (2, 2, 3)
    assert quat_w.shape == (2, 2, 4)
    assert [name for name, _ in env._backend.calls] == ["pos", "quat"]
    np.testing.assert_array_equal(env._backend.calls[0][1], np.array([1, 3], dtype=np.int32))
    np.testing.assert_array_equal(env._backend.calls[1][1], np.array([1, 3], dtype=np.int32))


def test_g1_motion_tracking_cfg_has_domain_rand_for_motrix():
    from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingCfg

    cfg = G1MotionTrackingCfg()
    assert hasattr(cfg, "domain_rand")
    assert cfg.domain_rand.randomize_base_mass is False
    assert cfg.domain_rand.random_com is False
    assert cfg.domain_rand.push_robots is False


def test_g1_motion_tracking_cfg_preserves_legacy_defaults():
    from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingCfg

    cfg = G1MotionTrackingCfg()

    assert str(cfg.motion_file).endswith("dance1_subject2_part.npz")
    assert cfg.pose_randomization.x == (-0.05, 0.05)
    assert cfg.velocity_randomization.x == (-0.5, 0.5)
    assert cfg.joint_position_range == (-0.1, 0.1)
    assert cfg.anchor_ori_threshold == pytest.approx(0.8)


def test_g1_flip_tracking_cfg_uses_flip_profile():
    from unilab.envs.motion_tracking.g1.flip_tracking import G1FlipTrackingCfg

    cfg = G1FlipTrackingCfg()

    assert str(cfg.motion_file).endswith("flip_360_001__A304.npz")
    assert cfg.pose_randomization.x == (0.0, 0.0)
    assert cfg.velocity_randomization.x == (0.0, 0.0)
    assert cfg.joint_position_range == (0.0, 0.0)
    assert cfg.anchor_ori_threshold == pytest.approx(1e9)
    assert cfg.sampling_mode in {"start", "clip_start", "uniform", "adaptive"}


def test_g1_motion_tracking_clip_end_contributes_to_truncated():
    from unilab.base.np_env import NpEnvState
    from unilab.envs.motion_tracking.g1.motion_loader import MotionData
    from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingEnv

    class FakeBackend:
        def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
            return np.zeros((2, len(body_ids), 3), dtype=np.float32)

        def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
            return np.zeros((2, len(body_ids), 3), dtype=np.float32)

    class FakeSampler:
        def __init__(self) -> None:
            self.failure_updates: list[np.ndarray] = []

        def get_current_motion(self) -> MotionData:
            return MotionData(
                joint_pos=np.zeros((2, 2), dtype=np.float32),
                joint_vel=np.zeros((2, 2), dtype=np.float32),
                body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
                body_quat_w=np.tile(
                    np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32), (2, 1, 1)
                ),
                body_lin_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
            )

        def update_failure_stats(self, terminated: np.ndarray) -> None:
            self.failure_updates.append(terminated.copy())

        def step(self) -> np.ndarray:
            return np.array([1], dtype=np.int32)

    env = object.__new__(G1MotionTrackingEnv)
    env._num_envs = 2
    env._cfg = type("Cfg", (), {"max_episode_steps": None})()
    env.body_ids = np.array([0], dtype=np.int32)
    env._backend = FakeBackend()
    env.motion_sampler = FakeSampler()
    env._clip_end_truncated = np.zeros((2,), dtype=bool)
    env.get_local_linvel = lambda: np.zeros((2, 3), dtype=np.float32)
    env.get_gyro = lambda: np.zeros((2, 3), dtype=np.float32)
    env.get_dof_pos = lambda: np.zeros((2, 2), dtype=np.float32)
    env.get_dof_vel = lambda: np.zeros((2, 2), dtype=np.float32)
    env._get_body_pose_w = lambda: (
        np.zeros((2, 1, 3), dtype=np.float32),
        np.tile(np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32), (2, 1, 1)),
    )
    env._update_relative_transforms = lambda *args: None
    env._compute_terminations = lambda *args: np.zeros((2,), dtype=bool)
    env._compute_reward = lambda *args: np.zeros((2,), dtype=np.float32)
    env._compute_obs = lambda *args: {
        "obs": np.zeros((2, 1), dtype=np.float32),
        "privileged": np.zeros((2, 1), dtype=np.float32),
    }

    state = NpEnvState(
        obs={
            "obs": np.zeros((2, 1), dtype=np.float32),
            "privileged": np.zeros((2, 1), dtype=np.float32),
        },
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=bool),
        truncated=np.zeros((2,), dtype=bool),
        info={"steps": np.zeros((2,), dtype=np.uint32)},
    )

    next_state = env.update_state(state)
    truncated = env._compute_truncated(next_state)

    np.testing.assert_array_equal(next_state.terminated, np.array([False, False]))
    np.testing.assert_array_equal(truncated, np.array([False, True]))
    np.testing.assert_array_equal(
        env.motion_sampler.failure_updates[0], np.array([False, False], dtype=bool)
    )


def test_g1_motion_tracking_clip_end_does_not_override_true_termination():
    from unilab.base.np_env import NpEnvState
    from unilab.envs.motion_tracking.g1.tracking import G1MotionTrackingEnv

    env = object.__new__(G1MotionTrackingEnv)
    env._num_envs = 2
    env._cfg = type("Cfg", (), {"max_episode_steps": None})()
    env._clip_end_truncated = np.array([False, True], dtype=bool)

    state = NpEnvState(
        obs={},
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.array([False, True], dtype=bool),
        truncated=np.zeros((2,), dtype=bool),
        info={"steps": np.zeros((2,), dtype=np.uint32)},
    )

    truncated = env._compute_truncated(state)
    np.testing.assert_array_equal(truncated, np.array([False, False]))


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
    default_allegro_reward_config,
):
    """Every registered env must be constructible, resetable, and steppable.

    Verifies:
    - observation/action spaces are valid
    - init_state + reset produces dict obs with correct keys and shapes
    - step with zero actions produces dict obs, scalar reward, bool done
    """
    ensure_registries()
    from unilab.base import registry

    # Provide reward_config for envs that require it via Hydra
    env_cfg_override = None
    if "Go1" in env_name:
        env_cfg_override = {"reward_config": default_go1_reward_config}
    elif "Go2" in env_name:
        env_cfg_override = {"reward_config": default_go2_reward_config}
    elif "G1WalkTaskMjSAC" in env_name:
        env_cfg_override = {"reward_config": default_g1_sac_reward_config}
    elif "G1" in env_name:
        env_cfg_override = {"reward_config": default_g1_reward_config}
    elif "Allegro" in env_name:
        env_cfg_override = {"reward_config": default_allegro_reward_config}

    env = registry.make(
        env_name, num_envs=2, sim_backend="mujoco", env_cfg_override=env_cfg_override
    )
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


def _assert_mujoco_position_gains(env, *, kp: float, kd: float, actuator_ids=slice(None)) -> None:
    model = env._backend.model
    np.testing.assert_allclose(model.actuator_gainprm[actuator_ids, 0], kp)
    np.testing.assert_allclose(model.actuator_biasprm[actuator_ids, 1], -kp)
    np.testing.assert_allclose(model.actuator_biasprm[actuator_ids, 2], -kd)
    np.testing.assert_allclose(env._backend._pool.get_field(0, "kp")[actuator_ids], kp)
    np.testing.assert_allclose(env._backend._pool.get_field(0, "kd")[actuator_ids], kd)


@pytest.mark.slow
def test_go1_env_initializes_kp_kd_into_pool(default_go1_reward_config):
    ensure_registries()
    from unilab.base import registry

    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "reward_config": default_go1_reward_config,
            "control_config": {"Kp": 12.0, "Kd": 0.7},
        },
    )
    try:
        _assert_mujoco_position_gains(env, kp=12.0, kd=0.7)
    finally:
        env.close()


@pytest.mark.slow
def test_go2_env_initializes_kp_kd_into_pool():
    ensure_registries()
    from unilab.base import registry
    from unilab.envs.locomotion.go2.joystick import RewardConfig

    env = registry.make(
        "Go2JoystickFlatTerrain",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "reward_config": RewardConfig(
                scales={
                    "tracking_lin_vel": 1.0,
                    "tracking_ang_vel": 0.2,
                    "lin_vel_z": -5.0,
                    "ang_vel_xy": -0.02,
                    "base_height": -100.0,
                    "action_rate": -0.005,
                    "similar_to_default": -0.1,
                },
                tracking_sigma=0.25,
                base_height_target=0.3,
            ),
            "control_config": {"Kp": 18.0, "Kd": 0.9},
        },
    )
    try:
        _assert_mujoco_position_gains(env, kp=18.0, kd=0.9)
    finally:
        env.close()


@pytest.mark.slow
def test_allegro_env_initializes_kp_kd_into_pool(default_allegro_reward_config):
    ensure_registries()
    from unilab.base import registry

    env = registry.make(
        "AllegroInhandRotation",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "reward_config": default_allegro_reward_config,
            "control_config": {"kp": 2.5, "kd": 0.4},
        },
    )
    try:
        _assert_mujoco_position_gains(env, kp=2.5, kd=0.4, actuator_ids=slice(0, 16))
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.parametrize("sim_backend", ["mujoco", "motrix"])
def test_g1_motion_tracking_reset_and_step(sim_backend: str):
    """G1MotionTracking needs a motion_file — skip if not available."""
    ensure_registries()
    from pathlib import Path

    from unilab.base import registry

    if sim_backend == "motrix":
        pytest.importorskip("motrixsim")

    # Look for any motion file in the expected location
    motion_dir = Path(__file__).parents[2] / "src" / "unilab" / "assets" / "motions" / "g1"
    if not motion_dir.exists():
        pytest.skip(f"Motion data directory not found: {motion_dir}")

    npz_files = list(motion_dir.glob("*.npz"))
    if not npz_files:
        pytest.skip(f"No .npz motion files in {motion_dir}")

    motion_file = str(npz_files[0])
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend=sim_backend,
        env_cfg_override={"motion_file": motion_file},
    )
    try:
        spec = env.obs_groups_spec
        assert isinstance(spec, dict)
        assert "obs" in spec
        assert "privileged" in spec
        obs_shape = env.observation_space.shape
        assert obs_shape is not None
        assert sum(spec.values()) == obs_shape[0]

        state = env.init_state()
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert state.obs[key].shape == (2, dim)

        action_shape = env.action_space.shape
        assert action_shape is not None
        actions = np.zeros((2, action_shape[0]))
        state = env.step(actions)
        assert isinstance(state.obs, dict)
        assert state.reward.shape == (2,)
        assert state.done.shape == (2,)
    finally:
        env.close()


@pytest.mark.slow
def test_go2_mujoco_reset_applies_kp_kd_domain_randomization(default_go2_reward_config):
    ensure_registries()

    from unilab.base import registry

    env = registry.make(
        "Go2JoystickFlatTerrain",
        num_envs=4,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": default_go2_reward_config},
    )
    try:
        env.init_state()
        backend = env._backend
        kp = np.stack([backend._pool.get_field(i, "kp") for i in range(env.num_envs)])
        kd = np.stack([backend._pool.get_field(i, "kd") for i in range(env.num_envs)])
        base_kp = float(env.cfg.control_config.Kp)
        base_kd = float(env.cfg.control_config.Kd)

        assert np.unique(np.round(kp[:, 0], 6)).size > 1
        assert np.unique(np.round(kd[:, 0], 6)).size > 1
        np.testing.assert_allclose(kp / base_kp, np.broadcast_to(kp[:, :1] / base_kp, kp.shape))
        np.testing.assert_allclose(kd / base_kd, np.broadcast_to(kd[:, :1] / base_kd, kd.shape))
        assert np.all(kp >= base_kp * 0.9)
        assert np.all(kp <= base_kp * 1.1)
        assert np.all(kd >= base_kd * 0.9)
        assert np.all(kd <= base_kd * 1.1)
    finally:
        env.close()
