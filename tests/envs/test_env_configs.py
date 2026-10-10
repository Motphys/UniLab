"""Tests for env config completeness and env instantiation.

The whole module is marked slow: Hydra composition and ``registry.make()``
over every owner are CPU-bound and too slow for the single-core CI runner.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from unilab.base.registry import ensure_registries

# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow


def _require_mujoco_runtime() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    pytest.importorskip("mjbatch", reason="mjbatch not installed")
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )


def _require_mjwarp_runtime() -> None:
    if sys.platform == "darwin":
        pytest.skip("mjwarp is a CUDA-only backend; macOS has no supported CUDA runtime")
    from unisim.backend.mjwarp.dependencies import load_mjwarp_dependencies

    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp runtime tests require an active CUDA Warp device")


def _require_genesis_runtime() -> None:
    from unisim.backend.genesis.dependencies import genesis_dependencies_available

    if not genesis_dependencies_available():
        pytest.skip("genesis requires the genesis-world extra")
    if not torch.cuda.is_available():
        pytest.skip("genesis runtime tests require a CUDA device")


def _require_newton_runtime() -> None:
    if sys.platform == "darwin":
        pytest.skip("newton is a CUDA-only backend; macOS has no supported CUDA runtime")
    pytest.importorskip("newton", reason="newton requires the newton extra")
    from unisim.backend.newton.dependencies import load_newton_dependencies

    dependencies = load_newton_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.skip("newton runtime tests require an active CUDA Warp device")


def _require_motrix_runtime() -> None:
    pytest.importorskip("motrixsim", reason="motrixsim not installed")


def _g1_manager_override(
    task: str = "g1_walk_flat", backend: str = "mujoco", config_group: str = "ppo"
) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir

    from unilab.base.config_adapter import BackendAdapter

    repo_root = Path(__file__).parents[2]
    with initialize_config_dir(
        config_dir=str(repo_root / "src" / "unilab" / "conf" / config_group), version_base="1.3"
    ):
        cfg = compose("config", overrides=[f"task={task}/{backend}"])
    return BackendAdapter(
        cfg, root_dir=repo_root, algo_name=config_group
    ).build_task_env_cfg_override()


def _motion_manager_override(
    task: str,
    backend: str,
    *,
    config_root: str = "ppo",
) -> tuple[str, dict[str, Any]]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.config_adapter import BackendAdapter

    repo_root = Path(__file__).parents[2]
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(repo_root / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        overrides = [f"task={task}/{backend}"]
        cfg = compose("config", overrides=overrides)
    return str(cfg.training.task_name), BackendAdapter(
        cfg,
        root_dir=repo_root,
        algo_name=config_root,
    ).build_task_env_cfg_override()


# ---------------------------------------------------------------------------
# Config attribute completeness (no env.step(), no MuJoCo sim)
# ---------------------------------------------------------------------------


def test_registry_bootstrap_and_config_imports_do_not_require_mujoco():
    repo_root = Path(__file__).parents[2]
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mujoco" or name.startswith("mujoco."):
                raise ImportError("mujoco blocked by test")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        from unilab.base import registry
        from unilab.base.backend_factory import create_backend
        from unilab.base.registry import ensure_registries

        ensure_registries()
        assert callable(create_backend)
        assert registry.contains("G1MotionTracking")
        metadata = registry.list_registered_envs()
        assert metadata["G1MotionTracking"]["config_factory"] == "ManagerBasedRlEnvCfg"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_g1_walk_flat_assets_define_contact_sensors_for_gait_rewards():
    repo_root = Path(__file__).parents[2]
    scene_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat.xml"
    ).read_text()
    model_text = (repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "g1.xml").read_text()

    for name in (
        "left_foot_contact_0",
        "left_foot_contact_1",
        "left_foot_contact_2",
        "left_foot_contact_3",
        "right_foot_contact_0",
        "right_foot_contact_1",
        "right_foot_contact_2",
        "right_foot_contact_3",
    ):
        assert name in scene_text

    for name in (
        "pelvis_local_linvel",
        "pelvis_gyro",
        "pelvis_acceleration",
        "pelvis_upvector",
        "torso_gyro",
        "torso_acceleration",
        "torso_upvector",
    ):
        assert name in model_text

    for name in (
        "left_foot_contact_0_geom",
        "left_foot_contact_1_geom",
        "left_foot_contact_2_geom",
        "left_foot_contact_3_geom",
        "right_foot_contact_0_geom",
        "right_foot_contact_1_geom",
        "right_foot_contact_2_geom",
        "right_foot_contact_3_geom",
    ):
        assert name in model_text


def test_g1_sphere_hand_assets_align_with_current_g1_sensor_names():
    repo_root = Path(__file__).parents[2]
    model_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "g1_sphere_hand.xml"
    ).read_text()

    for name in (
        "pelvis_local_linvel",
        "pelvis_gyro",
        "pelvis_acceleration",
        "pelvis_upvector",
        "torso_gyro",
        "torso_acceleration",
        "torso_upvector",
        "left_foot_quat",
        "right_foot_quat",
    ):
        assert name in model_text


# ---------------------------------------------------------------------------
# Fast env/backend smoke tests
# ---------------------------------------------------------------------------

# Environments that don't need special config overrides
_STANDARD_ENVS = [
    "G1WalkFlat",
]


@pytest.mark.parametrize("env_name", _STANDARD_ENVS)
def test_env_reset_and_step(env_name: str):
    """Every registered env must be constructible, resetable, and steppable.

    Verifies:
    - observation/action spaces are valid
    - init_state + reset produces dict obs with correct keys and shapes
    - step with zero actions produces dict obs, scalar reward, bool done
    """
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry

    # Provide config overrides for envs that require them via Hydra
    env_cfg_override = None
    if env_name == "G1WalkFlat":
        env_cfg_override = _g1_manager_override("g1_walk_flat")
    env = cast(
        Any,
        registry.make(
            env_name, num_envs=2, sim_backend="mujoco", env_cfg_override=env_cfg_override
        ),
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
        actions = torch.zeros((2, int(act_space.shape[0])), dtype=torch.float32)
        state = env.step(actions)
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert state.obs[key].shape == (2, dim)
        assert state.reward.shape == (2,)
        assert state.terminated.shape == (2,)
        assert state.truncated.shape == (2,)
    finally:
        env.close()


_MOTION_CORE_RUNTIME_CASES = (
    pytest.param("ppo", "g1_motion_tracking", "G1MotionTracking", 160, 286, 29, False),
    pytest.param("appo", "g1_motion_tracking", "G1MotionTracking", 160, 286, 29, False),
)


def test_g1_motion_core_registrations_are_manager_only() -> None:
    from unilab.base import registry

    ensure_registries()
    metadata = registry.list_registered_envs()
    # PPO/APPO use the MuJoCo owner, while SAC/FlashSAC may select the scoped
    # device-resident tensor owners through their Hydra task leaves.
    assert metadata["G1MotionTracking"]["config_factory"] == "ManagerBasedRlEnvCfg"
    assert set(metadata["G1MotionTracking"]["available_backends"]) >= {
        "mujoco",
        "mjwarp",
        "genesis",
    }


def test_g1_motion_manager_ppo_wraps_only_active_rows_in_one_state_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_registries()
    _require_mujoco_runtime()
    from unisim.backend.base import HostBridgeTransferPlan

    from unilab.base import registry

    _, override = _motion_manager_override("g1_motion_tracking", "mujoco")
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        command.time_steps[:] = command.tensor_sampler.current_clip_end_frames
        env.reset_buf.copy_(torch.tensor([True, False], device=env.device))

        reset_rows: list[torch.Tensor] = []

        def _record_host_apply_reset(
            self: HostBridgeTransferPlan,
            env_ids: torch.Tensor,
            qpos: torch.Tensor,
            qvel: torch.Tensor,
            *,
            randomization: Any = None,
        ) -> Any:
            reset_rows.append(env_ids.detach().clone())
            return self.__dict__.setdefault("_recorded_host_resets", len(reset_rows))

        host_plan = env.scene._tensor_read_plan.host_plan
        assert host_plan is not None
        env._reset_state.declare_packed_reset_device(env.device)
        monkeypatch.setattr(type(host_plan), "apply_reset", _record_host_apply_reset)
        all_rows = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
        with env._reset_state.scoped_device_owner_tensor_with_host_commit(all_rows, host_plan):
            env.command_manager.compute(dt=0.0)
        env.command_manager.post_compute()

        assert len(reset_rows) == 1
        torch.testing.assert_close(reset_rows[0], torch.tensor([1], device=env.device))
        assert command.time_steps[0] == command.tensor_sampler.current_clip_end_frames[0]
        assert command.time_steps[1] <= command.tensor_sampler.current_clip_end_frames[1]
        expected_motion = command.motion.get_motion_at_frame(command.time_steps)
        np.testing.assert_array_equal(command.joint_pos, expected_motion.joint_pos)
        torch.testing.assert_close(command._robot_body_pos_w, command.robot_body_pos_w)
        assert command._robot_cache_step == env.common_step_counter
    finally:
        env.close()


def test_g1_motion_manager_sac_clip_end_is_truncation() -> None:
    ensure_registries()
    _require_mujoco_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mujoco",
        config_root="sac",
    )
    override["auto_reset"] = False
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        command.time_steps[:] = command.tensor_sampler.current_clip_end_frames

        state = env.step(torch.zeros((2, 29), dtype=torch.float32))

        np.testing.assert_array_equal(state.terminated, [False, False])
        np.testing.assert_array_equal(state.truncated, [True, True])
        np.testing.assert_array_equal(
            command.time_steps, command.tensor_sampler.current_clip_end_frames
        )
    finally:
        env.close()


def test_sac_g1_motion_mjwarp_dr_runtime_applies_reset_and_interval_dr() -> None:
    ensure_registries()
    _require_mjwarp_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mjwarp",
        config_root="sac",
    )
    push_robot = override["events"]["push_robot"]
    push_robot["interval_range_s"] = [0.0, 0.0]
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="mjwarp",
        env_cfg_override=override,
    )
    try:
        state = env.init_state()
        assert torch.isfinite(state.obs["obs"]).all()
        assert torch.isfinite(state.obs["critic"]).all()

        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))
            assert torch.isfinite(state.obs["obs"]).all()
            assert torch.isfinite(state.obs["critic"]).all()
            assert torch.isfinite(state.reward).all()

        velocity_range = env.event_manager.get_term_cfg("push_robot").params["velocity_range"]
        assert velocity_range["x"] == [-0.5, 0.5]
        assert velocity_range["z"] == [-0.2, 0.2]
        assert all(velocity_range[axis] == [0.0, 0.0] for axis in ("roll", "pitch", "yaw"))
    finally:
        env.close()


def test_flashsac_g1_motion_mjwarp_anchor_observations_roll_out() -> None:
    """The canonical FlashSAC MJWarp owner exercises tensor-native anchors."""
    ensure_registries()
    _require_mjwarp_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mjwarp",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="mjwarp",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    command = env.command_manager.get_term("motion")
    assert command.tensor_carrier is True
    assert env.command_manager.uses_tensor_reset_rows()
    try:
        assert env.obs_groups_spec == {"obs": 160, "critic": 289}

        state = env.init_state()
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))

        from unisim.backend.base import SelectedResetPublication

        capabilities = env.backend.get_tensor_capabilities()
        assert capabilities.selected_reset_publication is (
            SelectedResetPublication.AUTHORITATIVE_VIEWS
        )
        step_calls = 0
        original_step = env.backend.step_tensor

        def count_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        env.backend.step_tensor = count_step  # type: ignore[method-assign]
        env.reset(env_indices=torch.tensor([1], dtype=torch.int64, device=env.device))
        assert step_calls == 0
        assert state.obs["obs"].shape == (2, 160)
        assert state.obs["critic"].shape == (2, 289)
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert torch.isfinite(state.reward).all()
    finally:
        env.close()


def test_flashsac_g1_motion_mujoco_tensor_command_roll_out() -> None:
    """The scoped MuJoCo host bridge owns a tensor motion command carrier."""
    ensure_registries()
    _require_mujoco_runtime()
    script = textwrap.dedent(
        """
        from pathlib import Path
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from unilab.base import registry
        from unilab.base.config_adapter import BackendAdapter
        from unilab.envs import ManagerBasedRlEnv

        registry.ensure_registries()
        GlobalHydra.instance().clear()
        root = Path.cwd()
        with initialize_config_dir(
            config_dir=str(root / "src/unilab/conf/flashsac"), version_base="1.3"
        ):
            owner = compose("config", overrides=["task=g1_motion_tracking/mujoco"])
        override = BackendAdapter(
            owner, root_dir=root, algo_name="flashsac"
        ).build_task_env_cfg_override()
        env = registry.make(
            "G1MotionTracking",
            num_envs=2,
            sim_backend="mujoco",
            env_cfg_override=override,
        )
        assert isinstance(env, ManagerBasedRlEnv)
        try:
            command = env.command_manager.get_term("motion")
            assert command.tensor_carrier is True
            assert env.torch_rng is not None
            state = env.reset(seed=7)[0]
            for _ in range(3):
                state = env.step(torch.zeros((2, 29), dtype=torch.float32))
            assert state.obs["obs"].shape == (2, 160)
            assert torch.isfinite(state.obs["obs"]).all()
            # Selected-row autoreset drives the tensor motion command through
            # the packed host-bridge commit; it must not raise or step physics.
            original_step = env.backend.step_tensor
            step_calls = []

            def count_step(*args, **kwargs):
                step_calls.append(1)
                return original_step(*args, **kwargs)

            env.backend.step_tensor = count_step
            env.reset(env_indices=torch.tensor([1], dtype=torch.int64))
            assert not step_calls
            state = env.step(torch.zeros((2, 29), dtype=torch.float32))
            assert torch.isfinite(state.obs["obs"]).all()
        finally:
            env.close()
        print("[mujoco tensor motion rollout] OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "[mujoco tensor motion rollout] OK" in result.stdout


@pytest.mark.parametrize("mode", ["startup", "reset"])
def test_flashsac_g1_motion_mujoco_tensor_command_model_field_dr_roll_out(
    mode: str,
) -> None:
    """MuJoCo tensor motion commits startup/reset model-field DR in one boundary."""
    ensure_registries()
    _require_mujoco_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override("g1_motion_tracking", "mujoco", config_root="flashsac")
    override["scene"]["entities"]["robot"]["geom_names"] = [
        f"{side}_foot{index}_collision" for side in ("left", "right") for index in range(1, 8)
    ]
    events = {
        "body_mass": {
            "_target_": "unilab.managers.event_manager.EventTermCfg",
            "func": "unilab.envs.mdp.randomize_rigid_body_mass",
            "mode": mode,
            "params": {
                "asset_cfg": {
                    "_target_": "unilab.managers.SceneEntityCfg",
                    "name": "robot",
                    "body_names": ["pelvis", "torso_link"],
                },
                "mass_distribution_params": [1.0, 1.0],
                "operation": "scale",
                "recompute_inertia": False,
            },
        },
        "body_com": {
            "_target_": "unilab.managers.event_manager.EventTermCfg",
            "func": "unilab.envs.mdp.randomize_rigid_body_com",
            "mode": mode,
            "params": {
                "asset_cfg": {
                    "_target_": "unilab.managers.SceneEntityCfg",
                    "name": "robot",
                    "body_names": ["pelvis", "torso_link"],
                },
                "com_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            },
        },
    }
    if mode == "reset":
        events.update(
            {
                "foot_friction": {
                    "_target_": "unilab.managers.event_manager.EventTermCfg",
                    "func": "unilab.envs.mdp.geom_friction",
                    "mode": "reset",
                    "params": {
                        "asset_cfg": {
                            "_target_": "unilab.managers.SceneEntityCfg",
                            "name": "robot",
                            "geom_names": "^(left|right)_foot[1-7]_collision$",
                        },
                        "ranges": [1.0, 1.0],
                        "operation": "abs",
                    },
                },
                "pd_gains": {
                    "_target_": "unilab.managers.event_manager.EventTermCfg",
                    "func": "unilab.envs.mdp.pd_gains",
                    "mode": "reset",
                    "params": {
                        "asset_cfg": {
                            "_target_": "unilab.managers.SceneEntityCfg",
                            "name": "robot",
                            "actuator_names": ".*",
                        },
                        "kp_range": [1.0, 1.0],
                        "kd_range": [1.0, 1.0],
                        "operation": "scale",
                    },
                },
            }
        )
    override["events"] = events

    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        command = env.command_manager.get_term("motion")
        assert command.tensor_carrier is True
        obs, _ = env.reset(seed=1812)
        assert all(torch.isfinite(values).all() for values in obs.values())
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32))
        assert torch.isfinite(state.reward).all()
        assert all(torch.isfinite(values).all() for values in state.obs.values())

        obs, _ = env.reset(env_indices=torch.tensor([1], dtype=torch.int64))
        assert all(torch.isfinite(values).all() for values in obs.values())
    finally:
        env.close()
        env._backend.close()


def test_flashsac_g1_motion_genesis_manager_tensor_command_roll_out() -> None:
    """The canonical FlashSAC Genesis owner exercises the Manager tensor path."""
    ensure_registries()
    _require_genesis_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "genesis",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="genesis",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": 160, "critic": 289}
        command = env.command_manager.get_term("motion")
        assert command.tensor_carrier is True
        assert env.command_manager.uses_tensor_reset_rows()

        state = env.init_state()
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))

        assert state.obs["obs"].shape == (2, 160)
        assert state.obs["critic"].shape == (2, 289)
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert torch.isfinite(state.reward).all()
    finally:
        env.close()
        env._backend.close()


def test_flashsac_g1_motion_newton_manager_tensor_command_roll_out() -> None:
    """The canonical FlashSAC Newton owner exercises the Manager tensor path."""
    ensure_registries()
    _require_newton_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "newton",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="newton",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": 160, "critic": 289}
        command = env.command_manager.get_term("motion")
        assert command.tensor_carrier is True
        assert env.command_manager.uses_tensor_reset_rows()

        state = env.init_state()
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))

        assert state.obs["obs"].shape == (2, 160)
        assert state.obs["critic"].shape == (2, 289)
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert torch.isfinite(state.reward).all()
    finally:
        env.close()
        env._backend.close()


def test_flashsac_g1_motion_motrix_manager_tensor_command_roll_out() -> None:
    """The canonical FlashSAC Motrix owner exercises the packed host bridge."""
    ensure_registries()
    _require_motrix_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "motrix",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="motrix",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": 160, "critic": 289}
        command = env.command_manager.get_term("motion")
        assert command.tensor_carrier is True
        assert env.command_manager.uses_tensor_reset_rows()

        state = env.init_state()
        for _ in range(3):
            state = env.step(torch.zeros((2, 29), dtype=torch.float32, device=env.device))

        assert state.obs["obs"].shape == (2, 160)
        assert state.obs["critic"].shape == (2, 289)
        assert all(torch.isfinite(values).all() for values in state.obs.values())
        assert torch.isfinite(state.reward).all()
    finally:
        env.close()
        env._backend.close()


@pytest.mark.parametrize(
    ("task", "backend"), [("g1_motion_tracking", "mjwarp"), ("g1_motion_tracking", "newton")]
)
def test_selected_reset_publication_requires_no_manager_readiness_step(
    task: str, backend: str
) -> None:
    """Selected reset publishes authoritative views without a control step."""
    ensure_registries()
    if backend == "mjwarp":
        _require_mjwarp_runtime()
    elif backend == "newton":
        _require_newton_runtime()
    else:
        _require_genesis_runtime()
    from unisim.backend.base import SelectedResetPublication

    from unilab.base import registry

    _, override = _motion_manager_override(
        task,
        backend,
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend=backend,
        env_cfg_override=override,
    )
    try:
        capabilities = env.backend.get_tensor_capabilities()
        assert capabilities.selected_reset
        assert (
            capabilities.selected_reset_publication is SelectedResetPublication.AUTHORITATIVE_VIEWS
        )
        env.init_state()

        step_calls = 0
        original_step = env.backend.step_tensor

        def count_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        env.backend.step_tensor = count_step  # type: ignore[method-assign]
        env.reset(env_indices=torch.tensor([1], dtype=torch.int64, device=env.device))
        assert step_calls == 0

        state_views = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
        assert state_views["qpos"].shape == (2, env.backend.get_public_state_widths().nq)
        assert state_views["qvel"].shape == (2, env.backend.get_public_state_widths().nv)
        assert torch.isfinite(state_views["qpos"]).all()
        assert torch.isfinite(state_views["qvel"]).all()
    finally:
        env.close()
        env._backend.close()


def test_newton_backend_close_disables_graphs_and_fails_closed() -> None:
    """Newton teardown publishes disabled graph diagnostics before rejecting use."""
    ensure_registries()
    _require_newton_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "newton",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="newton",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        before = env.backend.get_tensor_runtime_diagnostics()["cuda_graph"]
        assert before.requested is True
    finally:
        env.close()
        env.backend.close()

    after = env.backend.get_tensor_runtime_diagnostics()["cuda_graph"]
    assert after.enabled is False
    assert after.disable_reason is not None
    with pytest.raises(RuntimeError, match="newton backend is closed"):
        env.backend.get_state_views(("qpos", "qvel"), device=env.device)


@pytest.mark.parametrize(
    ("config_root", "task", "identity", "actor_dim", "critic_dim", "action_dim", "truncate"),
    _MOTION_CORE_RUNTIME_CASES,
)
@pytest.mark.parametrize("sim_backend", ["mujoco"])
def test_g1_motion_core_manager_reset_and_step(
    config_root: str,
    task: str,
    identity: str,
    actor_dim: int,
    critic_dim: int,
    action_dim: int,
    truncate: bool,
    sim_backend: str,
) -> None:
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    if sim_backend == "mujoco":
        _require_mujoco_runtime()
    else:
        pytest.importorskip("motrixsim")

    task_name, override = _motion_manager_override(
        task,
        sim_backend,
        config_root=config_root,
    )
    assert task_name == identity
    env = registry.make(
        identity,
        num_envs=2,
        sim_backend=sim_backend,
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": actor_dim, "critic": critic_dim}
        assert env.action_space.shape == (action_dim,)
        command = env.command_manager.get_term("motion")
        assert command.cfg.params.truncate_on_clip_end is truncate
        if sim_backend == "motrix" and "deploy" in task:
            assert env._cfg.events["foot_friction"] is None
            assert env._cfg.events["push_robot"] is None
        if sim_backend == "motrix" and config_root in {"ppo", "appo"}:
            root_pos = env._cfg.rewards["motion_global_root_pos"]
            action_rate = env._cfg.rewards["action_rate_l2"]
            assert root_pos is not None
            assert action_rate is not None
            expected_weights = (1.0, -0.05) if config_root == "ppo" else (0.5, -0.1)
            assert (root_pos.weight, action_rate.weight) == pytest.approx(expected_weights)

        state = env.init_state()
        assert state.obs["obs"].shape == (2, actor_dim)
        assert state.obs["critic"].shape == (2, critic_dim)

        state = env.step(torch.zeros((2, action_dim), dtype=torch.float32))
        assert state.reward.shape == (2,)
        assert state.terminated.shape == (2,)
        assert state.truncated.shape == (2,)
        assert torch.isfinite(state.reward).all()
        assert all(torch.isfinite(values).all() for values in state.obs.values())
    finally:
        env.close()


def test_flashsac_motion_reset_publishes_call_graph_counts() -> None:
    """Selected-reset diagnostics count Manager dispatch, not only wall time."""
    ensure_registries()
    _require_mjwarp_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mjwarp",
        config_root="flashsac",
    )
    env = registry.make(
        "G1MotionTracking",
        num_envs=4,
        sim_backend="mjwarp",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        env.init_state()
        # Force every row done so the selected-reset path executes deterministically.
        original_compute = env._compute_truncated

        def terminate_all(state):
            del state
            return torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)

        env._compute_truncated = terminate_all
        state = env.step(torch.zeros((4, 29), dtype=torch.float32, device=env.device))
        env._compute_truncated = original_compute
        timing = state.info["timing"]
        assert timing["reset_done_event_term_count"] == 0.0
        assert timing["reset_done_command_term_count"] == 1.0
        assert timing["reset_done_manager_reset_count"] == 7.0
        assert timing["reset_done_observation_term_count"] == 19.0
        assert timing["reset_done_sampler_dispatch_count"] == 1.0
        assert timing["reset_done_sampler_host_transfer_count"] == 0.0
    finally:
        env.close()
        env._backend.close()


def test_flashsac_motion_selected_reset_uses_generic_manager_lifecycle() -> None:
    """Tensor command reset composes with canonical Manager reset terms."""
    ensure_registries()
    _require_mjwarp_runtime()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand

    _, override = _motion_manager_override("g1_motion_tracking", "mjwarp", config_root="flashsac")
    env = registry.make(
        "G1MotionTracking",
        num_envs=8,
        sim_backend="mjwarp",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        env.reset(seed=1811)
        actions = torch.zeros((8, 29), dtype=torch.float32, device=env.device)
        env.step(actions)
        rows = torch.tensor([0, 2, 5, 7], dtype=torch.int64, device=env.device)
        env.reset_terminated.fill_(False)
        env.reset_time_outs.fill_(False)
        env.reset_terminated[rows] = True
        env.reset_time_outs[rows] = True
        env.reset_buf.fill_(False)
        env.reset_buf[rows] = True
        obs, _info = env.reset(env_indices=rows)

        assert all(torch.isfinite(values).all() for values in obs.values())
        command = env.command_manager.get_term("motion")
        assert isinstance(command, MotionCommand)
        timing = env._last_reset_manager_timing_ms
        assert timing["reset_done_manager_reset_count"] == 7.0
        assert timing["reset_done_sampler_dispatch_count"] == 1.0
        assert timing["reset_done_sampler_host_transfer_count"] == 0.0
    finally:
        env.close()
        env._backend.close()
