"""Contract tests for Go2ArmManipLoco environment."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from gymnasium import spaces

from unilab.base.np_env import NpEnvState

_GO2_ARM_MANIP_LOCO_MODULE = "unilab.envs.locomotion.go2_arm.manip_loco"
_REGISTRY_MODULE = "unilab.base.registry"


def _skip_if_no_mujoco():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")


def _default_reward_cfg():
    from unilab.envs.locomotion.go2_arm.manip_loco import RewardConfig

    return RewardConfig(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "object_distance": 2.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.3,
    )


def _make_env(num_envs: int = 2, env_cfg_override: dict | None = None):
    _ensure_go2_arm_manip_loco_registered()
    registry = _registry_module()
    override = {"reward_config": _default_reward_cfg()}
    if env_cfg_override:
        override.update(env_cfg_override)
    return registry.make(
        "Go2ArmManipLoco",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=override,
    )


def _registry_module():
    return importlib.import_module(_REGISTRY_MODULE)


def _ensure_go2_arm_manip_loco_registered() -> None:
    registry = _registry_module()
    registry.ensure_registries()
    if registry.contains("Go2ArmManipLoco"):
        return
    module = sys.modules.get(_GO2_ARM_MANIP_LOCO_MODULE)
    if module is None:
        importlib.import_module(_GO2_ARM_MANIP_LOCO_MODULE)
    else:
        importlib.reload(module)


def test_go2_arm_manip_loco_cfg_registered():
    """验证配置类已正确注册。"""
    _ensure_go2_arm_manip_loco_registered()
    registry = _registry_module()
    assert registry.contains("Go2ArmManipLoco")


def test_go2_arm_manip_loco_cfg_declares_scene_for_playback():
    """MuJoCo video playback needs the original visual scene, not only legacy model_file."""
    from unilab.base.scene import SceneCfg
    from unilab.envs.locomotion.go2_arm.manip_loco import (
        Go2ArmManipLocoCfg,
        _resolve_go2_arm_scene,
    )

    cfg = Go2ArmManipLocoCfg(reward_config=_default_reward_cfg())

    assert isinstance(cfg.scene, SceneCfg)
    assert cfg.scene.model_file == cfg.model_file
    assert cfg.scene.model_file.endswith("robots/go2_arm/scene_flat.xml")

    cfg.model_file = "custom_scene.xml"
    scene = _resolve_go2_arm_scene(cfg)
    assert scene.model_file == "custom_scene.xml"
    assert cfg.scene is scene


def test_go2_arm_ee_goal_collision_check_matches_reference_semantics():
    """EE goal 路径任一点进入 collision box 或地下都应判为 unsafe。"""
    from unilab.envs.locomotion.go2_arm.manip_loco import (
        EEGoalConfig,
        Go2ArmManipLocoCfg,
        Go2ArmManipLocoEnv,
        _cart2sphere,
    )

    env = object.__new__(Go2ArmManipLocoEnv)
    cfg = Go2ArmManipLocoCfg(reward_config=_default_reward_cfg())
    cfg.goal_ee = EEGoalConfig(num_collision_check_samples=3)
    env._cfg = cfg

    starts = _cart2sphere(np.asarray([[0.4, 0.2, 0.0]], dtype=np.float32))
    through_collision_box = _cart2sphere(np.asarray([[0.0, 0.0, -0.3]], dtype=np.float32))
    below_ground = _cart2sphere(np.asarray([[0.4, 0.2, -0.8]], dtype=np.float32))
    clear_path = _cart2sphere(np.asarray([[0.4, 0.2, 0.2]], dtype=np.float32))

    assert env._collision_check_sphere(starts, through_collision_box).tolist() == [True]
    assert env._collision_check_sphere(starts, below_ground).tolist() == [True]
    assert env._collision_check_sphere(starts, clear_path).tolist() == [False]


def test_go2_arm_command_moving_mask_includes_all_velocity_axes():
    """vx/vy/vyaw 任一轴超过阈值都应视为运动 command。"""
    from unilab.envs.locomotion.go2_arm.manip_loco import Go2ArmManipLocoEnv

    env = object.__new__(Go2ArmManipLocoEnv)
    clip = env._CMD_CLIP
    commands = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.5 * clip, 0.0],
            [0.0, 0.0, 1.5 * clip],
            [1.5 * clip, 0.0, 0.0],
            [0.5 * clip, -0.5 * clip, 0.5 * clip],
        ],
        dtype=np.float32,
    )

    assert env._command_is_moving(commands).tolist() == [False, True, True, True, False]

    normalized = env._normalize_velocity_commands(commands)
    np.testing.assert_allclose(normalized[0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(normalized[4], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(normalized[1:4], commands[1:4])


def test_go2_arm_command_postprocess_can_force_zero_commands():
    """zero_command_prob 应在小命令归零之外显式注入 exact zero command。"""
    from unilab.envs.locomotion.go2_arm.manip_loco import (
        Go2ArmManipLocoCfg,
        Go2ArmManipLocoEnv,
    )

    env = object.__new__(Go2ArmManipLocoEnv)
    cfg = Go2ArmManipLocoCfg(reward_config=_default_reward_cfg())
    cfg.commands.zero_command_prob = 1.0
    env._cfg = cfg
    commands = np.asarray(
        [
            [0.5, 0.2, 0.3],
            [-0.5, -0.2, -0.3],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        env._postprocess_velocity_commands(commands), np.zeros_like(commands)
    )

    cfg.commands.zero_command_prob = 0.0
    np.testing.assert_allclose(env._postprocess_velocity_commands(commands), commands)


def test_go2_arm_stand_still_reward_uses_same_command_mask():
    """stand_still 不应惩罚 lateral/yaw/forward command 下的腿部姿态偏差。"""
    from unilab.envs.locomotion.common.rewards import RewardContext
    from unilab.envs.locomotion.go2_arm.manip_loco import Go2ArmManipLocoEnv

    env = object.__new__(Go2ArmManipLocoEnv)
    clip = env._CMD_CLIP
    commands = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.5 * clip, 0.0],
            [0.0, 0.0, 1.5 * clip],
            [1.5 * clip, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    dof_pos = np.ones((4, 18), dtype=np.float32)
    ctx = RewardContext(
        info={"commands": commands},
        linvel=np.zeros((4, 3), dtype=np.float32),
        gyro=np.zeros((4, 3), dtype=np.float32),
        dof_pos=dof_pos,
        dof_vel=np.zeros((4, 18), dtype=np.float32),
        default_angles=np.zeros(18, dtype=np.float32),
    )

    np.testing.assert_allclose(env._reward_stand_still(ctx), np.asarray([12.0, 0.0, 0.0, 0.0]))


def test_go2_arm_write_feet_phase_updates_indexed_envs():
    """reset env 子集时也要真正写回 feet_phase，不能被 fancy indexing 拷贝吞掉。"""
    from unilab.envs.locomotion.go2_arm.manip_loco import Go2ArmManipLocoEnv

    env = object.__new__(Go2ArmManipLocoEnv)
    env.phase = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
    env.feet_phase = np.ones((3, 4), dtype=np.float32)

    env._write_feet_phase(np.asarray([0, 2], dtype=np.int32), np.asarray([False, True]))

    np.testing.assert_allclose(env.feet_phase[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(env.feet_phase[1], np.ones(4, dtype=np.float32))
    np.testing.assert_allclose(
        env.feet_phase[2], np.asarray([0.6, 0.1, 0.1, 0.6], dtype=np.float32), atol=1e-6
    )


def test_go2_arm_apply_action_uses_arm_action_scale_for_arm_residual():
    """腿部 residual 用 action_scale，机械臂 residual 用 arm_action_scale。"""
    from unilab.envs.locomotion.go2_arm.manip_loco import (
        Go2ArmManipLocoCfg,
        Go2ArmManipLocoEnv,
    )

    env = object.__new__(Go2ArmManipLocoEnv)
    cfg = Go2ArmManipLocoCfg(reward_config=_default_reward_cfg())
    cfg.control_config.action_scale = 0.25
    cfg.control_config.arm_action_scale = 0.05
    cfg.ik.gain = 0.0
    env._cfg = cfg
    env._num_envs = 1
    env.default_angles = np.zeros(18, dtype=np.float64)
    env._action_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float64)
    env.curr_ee_goal_cart = np.zeros((1, 3), dtype=np.float64)
    env.ee_goal_orn_quat = np.zeros((1, 4), dtype=np.float64)
    env.get_ee_local_pose = lambda: (  # type: ignore[method-assign]
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 4), dtype=np.float64),
    )
    env.compute_arm_ik_delta = lambda *_args, **_kwargs: np.zeros(  # type: ignore[method-assign]
        (1, 6), dtype=np.float64
    )
    env.get_arm_dof_pos = lambda: np.ones((1, 6), dtype=np.float64)  # type: ignore[method-assign]

    state = NpEnvState(
        obs={},
        reward=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        info={},
    )
    ctrl = env.apply_action(np.ones((1, 18), dtype=np.float64), state)

    np.testing.assert_allclose(ctrl[0, :12], np.full(12, 0.25, dtype=np.float64))
    np.testing.assert_allclose(ctrl[0, 12:18], np.full(6, 1.05, dtype=np.float64))


@pytest.mark.slow
def test_go2_arm_playback_resolves_visual_scene_model(tmp_path):
    """Offline video export should re-materialize the visual XML for Go2Arm."""
    _skip_if_no_mujoco()
    import mujoco

    from unilab.visualization.playback import _resolve_render_play_model_files

    env = _make_env(num_envs=2)
    try:
        assert env.cfg.scene is not None
        model_file = _resolve_render_play_model_files(env, num_envs=2, tmp_dir=tmp_path)
        assert isinstance(model_file, str)

        model = mujoco.MjModel.from_binary_path(model_file)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_link_visual") >= 0
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_obs_groups_spec():
    """验证 obs_groups_spec 维度正确（actor 76，critic 79）。"""
    _skip_if_no_mujoco()
    env = _make_env(num_envs=1)
    assert env.obs_groups_spec == {"obs": 76, "critic": 79}


@pytest.mark.slow
def test_go2_arm_reset_step_contract():
    """验证 init_state/step 返回正确 shape。"""
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2)
    state = env.init_state()
    assert state.obs["obs"].shape == (2, 76)
    assert state.obs["critic"].shape == (2, 79)

    actions = np.zeros((2, 18))
    state = env.step(actions)
    assert state.reward.shape == (2,)
    assert state.obs["obs"].shape == (2, 76)


@pytest.mark.slow
def test_go2_arm_ee_goal_valid_after_reset():
    """验证 reset 后末端目标 shape 和有限性。"""
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2)
    env.init_state()
    assert env.curr_ee_goal_cart.shape == (2, 3)
    assert np.all(np.isfinite(env.curr_ee_goal_cart))


@pytest.mark.slow
def test_go2_arm_ee_goal_resampling():
    """验证 ee goal timer 到期后目标发生变化（新字段 _arm_goal_timer / _traj_total_steps）。"""
    _skip_if_no_mujoco()
    env = _make_env(num_envs=4)
    env.init_state()

    # 强制 timer 到期（设为总步数 - 1，step 后触发 >= 条件）
    env._arm_goal_timer[:] = env._traj_total_steps - 1
    goal_before = env.curr_ee_goal_cart.copy()

    actions = np.zeros((4, 18))
    env.step(actions)

    changed = not np.allclose(env.curr_ee_goal_cart, goal_before)
    assert changed, "ee goal should have changed after arm_goal_timer expiry"


@pytest.mark.slow
def test_go2_arm_ee_goal_interpolation():
    """验证 timer 处于运动阶段时，每步 curr_ee_goal_cart 在变化（球空间插值）。"""
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2)
    env.init_state()

    # 将 timer 置于运动阶段中间，确保不会触发 expiry
    env._arm_goal_timer[:] = 0
    env._traj_steps[:] = 100
    env._traj_total_steps[:] = 150

    pos0 = env.curr_ee_goal_cart.copy()
    env.step(np.zeros((2, 18)))
    pos1 = env.curr_ee_goal_cart.copy()

    assert not np.allclose(pos0, pos1), "EE goal should interpolate each step"


@pytest.mark.slow
def test_go2_arm_command_resampling():
    """验证 command timer 到期后 command 发生变化。"""
    _skip_if_no_mujoco()
    env = _make_env(num_envs=4, env_cfg_override={"commands": {"resample_time_s": 0.02}})
    env.init_state()

    # 强制 timer 到期
    env._cmd_timer[:] = env._cmd_resample_steps - 1
    cmd_before = env._state.info["commands"].copy()

    actions = np.zeros((4, 18))
    env.step(actions)

    # 至少部分 env 的 command 应该发生变化
    changed = not np.allclose(env._state.info["commands"], cmd_before)
    assert changed, "commands should have been resampled after timer expiry"
