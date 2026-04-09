"""Tests for script entry-point utilities (pure functions and Hydra config defaults).

Coverage targets:
  - train_offpolicy.py: Hydra defaults, default_device(), resolve_checkpoint_path()
  - train_mlx_ppo.py:   get_latest_run(), get_latest_checkpoint()  (skipped if mlx absent)
  - play_interactive.py: resolve_checkpoint()                       (skipped if mujoco absent)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_CONF_DIR = Path(__file__).parent.parent.parent / "conf"


def _load_script(name: str):
    """Load a scripts/<name>.py as a fresh module (no __init__ required)."""
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import sys as _sys

    import mlx.core  # noqa: F401

    _HAS_MLX = _sys.platform == "darwin"
except ImportError:
    _HAS_MLX = False

try:
    import mujoco  # noqa: F401

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# train_offpolicy.py — Hydra config defaults
# ---------------------------------------------------------------------------


def _offpolicy_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "offpolicy"), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def _ppo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def _appo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "appo"), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def _train_rsl_rl(monkeypatch: pytest.MonkeyPatch):
    import types

    for module_name in list(sys.modules):
        if module_name == "unilab" or module_name.startswith("unilab."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    runners_mod = types.ModuleType("rsl_rl.runners")
    runners_mod.OnPolicyRunner = object
    rsl_pkg = types.ModuleType("rsl_rl")
    rsl_pkg.runners = runners_mod
    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_pkg)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners_mod)
    return _load_script("train_rsl_rl")


def _train_appo():
    return _load_script("train_appo")


def test_offpolicy_hydra_default_algo():
    cfg = _offpolicy_cfg()
    assert cfg.algo.algo == "sac"


def test_offpolicy_hydra_default_task():
    cfg = _offpolicy_cfg()
    assert cfg.training.task_name == "Go1JoystickFlatTerrain"


def test_offpolicy_hydra_default_logger():
    cfg = _offpolicy_cfg()
    assert cfg.training.logger == "tensorboard"


def test_offpolicy_hydra_default_wandb_fields():
    cfg = _offpolicy_cfg()
    assert cfg.training.wandb_project == "unilab"
    assert cfg.training.wandb_entity is None
    assert cfg.training.wandb_group is None
    assert cfg.training.wandb_job_type is None
    assert cfg.training.wandb_name is None
    assert cfg.training.wandb_tags == []
    assert cfg.training.wandb_notes is None
    assert cfg.training.wandb_mode is None


def test_offpolicy_hydra_default_sim_backend():
    cfg = _offpolicy_cfg()
    assert cfg.training.sim_backend == "mujoco"


def test_ppo_hydra_default_wandb_fields():
    cfg = _ppo_cfg()
    assert cfg.training.wandb_project == "unilab"
    assert cfg.training.wandb_entity is None
    assert cfg.training.wandb_group is None
    assert cfg.training.wandb_job_type is None
    assert cfg.training.wandb_name is None
    assert cfg.training.wandb_tags == []
    assert cfg.training.wandb_notes is None
    assert cfg.training.wandb_mode is None


def test_offpolicy_hydra_default_play_flags():
    cfg = _offpolicy_cfg()
    assert cfg.training.play_only is False
    assert cfg.training.no_play is False
    assert cfg.training.load_run == "-1"


def test_offpolicy_hydra_algo_td3():
    cfg = _offpolicy_cfg(["algo=td3"])
    assert cfg.algo.algo == "td3"


def test_offpolicy_task_go1_exposes_motrix_legacy():
    cfg = _offpolicy_cfg(["task=go1_joystick"])

    assert cfg.motrix_legacy.enabled is True
    assert cfg.motrix_legacy.applies_to.algo == "sac"
    assert cfg.motrix_legacy.algo_overrides.num_envs == 4096
    assert cfg.motrix_legacy.algo_overrides.max_iterations == 2000
    assert cfg.motrix_legacy.env_cfg_override.legacy_motrix_profile.enabled is True


def test_offpolicy_build_task_motrix_env_cfg_override_applies_go1_legacy():
    cfg = _offpolicy_cfg(["task=go1_joystick", "training.sim_backend=motrix"])

    env_cfg_override = _offpolicy().build_task_motrix_offpolicy_env_cfg_override("sac", cfg)

    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 2000
    assert env_cfg_override["legacy_motrix_profile"]["enabled"] is True
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)


def test_offpolicy_build_task_motrix_env_cfg_override_skips_td3():
    cfg = _offpolicy_cfg(["algo=td3", "task=go1_joystick", "training.sim_backend=motrix"])

    env_cfg_override = _offpolicy().build_task_motrix_offpolicy_env_cfg_override("td3", cfg)

    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 3000
    assert "legacy_motrix_profile" not in env_cfg_override


def test_offpolicy_build_task_motrix_env_cfg_override_respects_cli_algo_override(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _offpolicy_cfg(
        ["task=go1_joystick", "training.sim_backend=motrix", "algo.max_iterations=1"]
    )
    monkeypatch.setattr(sys, "argv", ["train_offpolicy.py", "algo.max_iterations=1"])

    _offpolicy().build_task_motrix_offpolicy_env_cfg_override("sac", cfg)

    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 1


def test_offpolicy_resolve_sac_use_symmetry_keeps_mujoco_setting():
    cfg = _offpolicy_cfg(["task=g1_sac", "training.sim_backend=mujoco"])

    assert _offpolicy().resolve_sac_use_symmetry(cfg) is True


def test_offpolicy_resolve_sac_use_symmetry_disables_motrix():
    cfg = _offpolicy_cfg(["task=g1_sac", "training.sim_backend=motrix"])

    assert _offpolicy().resolve_sac_use_symmetry(cfg) is False


def test_ppo_task_go1_exposes_motrix_legacy():
    cfg = _ppo_cfg(["task=go1_joystick"])

    assert cfg.motrix_legacy.enabled is True
    assert cfg.motrix_legacy.algo_overrides.max_iterations == 151
    assert cfg.motrix_legacy.algo_overrides.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.motrix_legacy.env_cfg_override.legacy_motrix_profile.enabled is True


def test_ppo_task_g1_exposes_motrix_legacy():
    cfg = _ppo_cfg(["task=g1_joystick"])

    assert cfg.motrix_legacy.enabled is True
    assert cfg.motrix_legacy.algo_overrides.obs_groups.actor == ["policy"]
    assert (
        cfg.motrix_legacy.env_cfg_override.backend_overrides.control_action_scale
        == pytest.approx(0.5)
    )


def test_ppo_task_go2_aligns_mujoco_with_motrix_defaults():
    cfg = _ppo_cfg(["task=go2_joystick"])

    assert cfg.algo.num_envs == 1024
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.reward.scales.tracking_ang_vel == pytest.approx(0.2)
    assert cfg.reward.scales.lin_vel_z == pytest.approx(-5.0)
    assert cfg.reward.scales.ang_vel_xy == pytest.approx(-0.1)
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)


def test_build_task_motrix_ppo_env_cfg_override_applies_go1_legacy(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go1_joystick", "training.sim_backend=motrix"])

    env_cfg_override = mod.build_task_motrix_ppo_env_cfg_override(cfg)

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)
    assert env_cfg_override["legacy_motrix_profile"]["enabled"] is True
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)


def test_build_task_motrix_ppo_env_cfg_override_applies_g1_legacy(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_joystick", "training.sim_backend=motrix"])

    env_cfg_override = mod.build_task_motrix_ppo_env_cfg_override(cfg)

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(5.0e-3)
    assert cfg.algo.obs_groups.actor == ["policy"]
    assert env_cfg_override["backend_overrides"]["enabled"] is True
    assert env_cfg_override["reward_config"]["scales"]["upper_body_pose"] == pytest.approx(-0.05)


def test_build_task_motrix_ppo_env_cfg_override_applies_go2_reward_motrix(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go2_joystick", "training.sim_backend=motrix"])

    env_cfg_override = mod.build_task_motrix_ppo_env_cfg_override(cfg)

    assert cfg.reward_motrix.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.algo.num_envs == 1024
    assert env_cfg_override["legacy_motrix_profile"]["enabled"] is True
    assert env_cfg_override["domain_rand"]["randomize_kp"] is False
    assert env_cfg_override["domain_rand"]["randomize_kd"] is False
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)
    assert env_cfg_override["reward_config"]["scales"]["tracking_ang_vel"] == pytest.approx(0.2)


def test_build_task_motrix_ppo_env_cfg_override_respects_cli_algo_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_joystick", "training.sim_backend=motrix", "algo.max_iterations=1"])
    monkeypatch.setattr(sys, "argv", ["train_rsl_rl.py", "algo.max_iterations=1"])

    mod.build_task_motrix_ppo_env_cfg_override(cfg)

    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.max_iterations == 1


def test_g1_motion_tracking_ppo_motrix_prefers_backend_specific_reward(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking", "training.sim_backend=motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)
    assert cfg.reward_motrix.scales.motion_body_pos == pytest.approx(1.0)

    cfg.reward.scales.motion_body_pos = 1.25
    cfg.reward_motrix.scales.motion_body_pos = 2.5

    env_cfg_override = mod.build_task_motrix_ppo_env_cfg_override(cfg)

    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(2.5)


def test_build_motrix_play_ppo_env_cfg_override_applies_g1_motion_tracking_play_profile(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        ["task=g1_motion_tracking", "training.sim_backend=motrix", "training.play_only=true"]
    )

    monkeypatch.setattr(
        mod,
        "materialize_scene_visual_override",
        lambda source_model_file, **kwargs: "/tmp/g1_motion_tracking_play_scene.xml",
    )

    env_cfg_override = mod.build_motrix_play_ppo_env_cfg_override(cfg)

    assert cfg.training.play_env_num == 128
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)
    assert env_cfg_override["model_file"] == "/tmp/g1_motion_tracking_play_scene.xml"
    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(1.0)


def test_build_motrix_play_ppo_env_cfg_override_respects_cli_play_env_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=g1_motion_tracking",
            "training.sim_backend=motrix",
            "training.play_only=true",
            "training.play_env_num=32",
        ]
    )
    monkeypatch.setattr(sys, "argv", ["train_rsl_rl.py", "training.play_env_num=32"])
    monkeypatch.setattr(
        mod,
        "materialize_scene_visual_override",
        lambda source_model_file, **kwargs: "/tmp/g1_motion_tracking_play_scene.xml",
    )

    env_cfg_override = mod.build_motrix_play_ppo_env_cfg_override(cfg)

    assert cfg.training.play_env_num == 32
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)


def test_build_motrix_play_ppo_env_cfg_override_resolves_relative_ground_texture(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        ["task=g1_motion_tracking", "training.sim_backend=motrix", "training.play_only=true"]
    )
    cfg.motrix_play_only.scene_override.ground_texture_file = (
        "src/unilab/assets/robots/g1/floor.png"
    )

    captured = {}

    def _fake_materialize(source_model_file, **kwargs):
        captured["source_model_file"] = source_model_file
        captured.update(kwargs)
        return "/tmp/g1_motion_tracking_play_scene.xml"

    monkeypatch.setattr(mod, "materialize_scene_visual_override", _fake_materialize)

    mod.build_motrix_play_ppo_env_cfg_override(cfg)

    assert captured["ground_texture_file"] == str(
        mod.ROOT_DIR / "src/unilab/assets/robots/g1/floor.png"
    )


def test_run_motrix_rsl_play_loop_uses_render_spacing():
    import numpy as np
    import torch
    from tensordict import TensorDict

    mod = _train_rsl_rl(pytest.MonkeyPatch())

    class FakePolicy:
        def __call__(self, obs):
            batch = obs.batch_size[0]
            return torch.zeros((batch, 3), dtype=torch.float32)

    class FakeBackend:
        def __init__(self):
            self.init_renderer_calls = []
            self.render_calls = 0

        def init_renderer(self, spacing=1.0):
            self.init_renderer_calls.append(spacing)

        def render(self):
            self.render_calls += 1

    class FakeEnv:
        def __init__(self):
            self._backend = FakeBackend()
            self.cfg = type("Cfg", (), {"render_spacing": 2.5})()

    class FakeWrapper:
        def __init__(self):
            self.env = FakeEnv()
            self.reset_calls = 0
            self.step_calls = 0

        def reset(self):
            self.reset_calls += 1
            return TensorDict({"policy": torch.ones((2, 5), dtype=torch.float32)}, batch_size=2), {}

        def step(self, actions):
            self.step_calls += 1
            return (
                TensorDict({"policy": torch.ones((2, 5), dtype=torch.float32)}, batch_size=2),
                torch.zeros((2,), dtype=torch.float32),
                torch.zeros((2,), dtype=torch.bool),
                {},
            )

    wrapped_env = FakeWrapper()

    mod.run_motrix_rsl_play_loop(
        wrapped_env=wrapped_env,
        policy=FakePolicy(),
        render_spacing=2.5,
        num_steps=3,
    )

    assert wrapped_env.reset_calls == 1
    assert wrapped_env.step_calls == 3
    assert wrapped_env.env._backend.init_renderer_calls == [2.5]
    assert wrapped_env.env._backend.render_calls == 3


def test_g1_motion_tracking_appo_reward_extraction_prefers_backend_specific_reward():
    from unilab.utils.reward_utils import extract_reward_config

    cfg = _appo_cfg(["task=g1_motion_tracking", "training.sim_backend=motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)
    assert cfg.reward_motrix.scales.motion_body_pos == pytest.approx(1.0)

    cfg.reward.scales.motion_body_pos = 1.5
    cfg.reward_motrix.scales.motion_body_pos = 3.0

    env_cfg_override = extract_reward_config(cfg)

    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(3.0)


def test_g1_motion_tracking_ppo_can_override_backend_reward_from_reward_group():
    cfg = _ppo_cfg(
        [
            "task=g1_motion_tracking",
            "training.sim_backend=motrix",
            "reward@reward_motrix=g1_motion_tracking_motrix",
        ]
    )

    assert cfg.reward_motrix.scales.motion_body_pos == pytest.approx(1.0)


def test_g1_motion_tracking_appo_can_override_backend_reward_from_reward_group():
    cfg = _appo_cfg(
        [
            "task=g1_motion_tracking",
            "training.sim_backend=motrix",
            "reward@reward_motrix=g1_motion_tracking_motrix",
        ]
    )

    assert cfg.reward_motrix.scales.motion_body_pos == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# train_appo.py — motrix runner / play helpers
# ---------------------------------------------------------------------------


def test_build_appo_runner_kwargs_forwards_sim_backend():
    mod = _train_appo()
    cfg = _appo_cfg(["task=g1_motion_tracking", "training.sim_backend=motrix"])

    runner_kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"reward_config": {"scales": {}}},
        collector_device="cpu",
    )

    assert runner_kwargs["env_name"] == "G1MotionTracking"
    assert runner_kwargs["sim_backend"] == "motrix"
    assert runner_kwargs["collector_device"] == "cpu"
    assert runner_kwargs["num_envs"] == cfg.algo.num_envs
    assert runner_kwargs["steps_per_env"] == cfg.algo.steps_per_env
    assert runner_kwargs["env_cfg_overrides"]["reward_config"]["scales"] == {}


def test_run_motrix_play_loop_runs_without_physics_state():
    import numpy as np
    import torch

    mod = _train_appo()

    class FakeActor:
        def __call__(self, td):
            batch = td.batch_size[0]
            return torch.zeros((batch, 3), dtype=torch.float32)

    class FakeBackend:
        def __init__(self):
            self.init_renderer_calls = 0
            self.render_calls = 0

        def init_renderer(self):
            self.init_renderer_calls += 1

        def render(self):
            self.render_calls += 1

    class FakeState:
        def __init__(self):
            self.obs = {"obs": np.ones((2, 5), dtype=np.float32)}

    class FakeEnv:
        def __init__(self):
            self.state = None
            self._backend = FakeBackend()
            self.init_state_calls = 0
            self.reset_calls = 0
            self.step_calls = 0

        def init_state(self):
            self.init_state_calls += 1
            self.state = object()

        def reset(self, env_indices):
            self.reset_calls += 1
            assert env_indices.shape == (2,)
            return {"obs": np.ones((2, 5), dtype=np.float32)}, {}

        def step(self, actions):
            self.step_calls += 1
            assert actions.shape == (2, 3)
            return FakeState()

    env = FakeEnv()

    mod.run_motrix_play_loop(
        env=env,
        actor=FakeActor(),
        device="cpu",
        play_env_num=2,
        num_steps=3,
    )

    assert env.init_state_calls == 1
    assert env.reset_calls == 1
    assert env.step_calls == 3
    assert env._backend.init_renderer_calls == 1
    assert env._backend.render_calls == 3


def test_resolve_appo_checkpoint_path_prefers_latest_model_in_explicit_dir(tmp_path):
    mod = _train_appo()
    run_dir = tmp_path / "logs" / "appo" / "MyTask" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "model_1.pt").write_bytes(b"")
    (run_dir / "model_7.pt").write_bytes(b"")

    checkpoint_path, checkpoint_dir = mod.resolve_appo_checkpoint_path(
        base_log_dir=tmp_path / "logs" / "appo" / "MyTask",
        load_run=str(run_dir),
    )

    assert checkpoint_path is not None
    assert checkpoint_path.endswith("model_7.pt")
    assert checkpoint_dir == str(run_dir)


# ---------------------------------------------------------------------------
# train_offpolicy.py — default_device()
# ---------------------------------------------------------------------------


def _offpolicy():
    return _load_script("train_offpolicy")


def test_offpolicy_default_device_preferred_cpu():
    mock_torch = MagicMock()
    assert _offpolicy().default_device(mock_torch, preferred="cpu") == "cpu"


def test_offpolicy_default_device_preferred_cuda():
    mock_torch = MagicMock()
    assert _offpolicy().default_device(mock_torch, preferred="cuda") == "cuda"


def test_offpolicy_default_device_cuda_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "cuda"


def test_offpolicy_default_device_mps_fallback():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "mps"


def test_offpolicy_default_device_cpu_fallback():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    assert _offpolicy().default_device(mock_torch) == "cpu"


# ---------------------------------------------------------------------------
# train_offpolicy.py — resolve_checkpoint_path()
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_no_base_dir(tmp_path):
    """load_run='-1' with no log directory → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_explicit_existing_file(tmp_path):
    """load_run = absolute path to existing .pt → returns that path."""
    model_file = tmp_path / "model_100.pt"
    model_file.write_bytes(b"")
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", str(model_file)
    )
    assert path == str(model_file)
    assert path_dir == str(tmp_path)


def test_resolve_checkpoint_latest_picks_highest_iter(tmp_path):
    """load_run='-1' picks model with numerically highest iteration."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)
    (task_dir / "model_10.pt").write_bytes(b"")
    (task_dir / "model_50.pt").write_bytes(b"")
    (task_dir / "model_100.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is not None
    assert "model_100.pt" in path


def test_resolve_checkpoint_explicit_run_name(tmp_path):
    """load_run = run-directory name under the log root."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "myrun"
    task_dir.mkdir(parents=True)
    (task_dir / "model_5.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "myrun")
    assert path is not None
    assert "model_5.pt" in path
    assert path_dir == str(task_dir)


def test_resolve_checkpoint_nonexistent_explicit_path(tmp_path):
    """load_run points to a path that doesn't exist → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", "/nonexistent/model.pt"
    )
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_empty_run_dir(tmp_path):
    """Run directory exists but has no model_*.pt → (None, None)."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None


def test_offpolicy_extract_reset_obs_handles_two_tuple():
    obs = {"obs": "value"}

    result = _offpolicy().extract_reset_obs((obs, {"info": 1}))

    assert result is obs


def test_offpolicy_extract_reset_obs_rejects_three_tuple():
    obs = {"obs": "value"}

    with pytest.raises(ValueError, match="Unexpected env.reset return format"):
        _offpolicy().extract_reset_obs(("ignored", obs, {"info": 1}))


def test_offpolicy_resolve_play_obs_dim_ignores_privileged():
    obs_dim = _offpolicy().resolve_play_obs_dim({"obs": 98, "privileged": 3})

    assert obs_dim == 98


def test_offpolicy_extract_play_obs_uses_obs_group_only():
    import numpy as np

    obs = {
        "obs": np.ones((2, 98), dtype=np.float32),
        "privileged": np.full((2, 3), 2.0, dtype=np.float32),
    }

    play_obs = _offpolicy().extract_play_obs(obs)

    assert play_obs.shape == (2, 98)
    assert np.allclose(play_obs, 1.0)


# ---------------------------------------------------------------------------
# train_mlx_ppo.py — get_latest_run() / get_latest_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_nonexistent_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_run(tmp_path / "nonexistent") is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_empty_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_run(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_returns_last_sorted(tmp_path):
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "2024-01-01_mujoco").mkdir()
    (tmp_path / "2024-03-15_mujoco").mkdir()
    (tmp_path / "2024-02-10_mujoco").mkdir()
    result = mod.get_latest_run(tmp_path)
    assert result is not None
    assert result.name == "2024-03-15_mujoco"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_nonexistent_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_checkpoint(tmp_path / "no_such_dir") is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_empty_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_checkpoint(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_picks_highest_iter(tmp_path):
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "model_0.safetensors").write_bytes(b"")
    (tmp_path / "model_50.safetensors").write_bytes(b"")
    (tmp_path / "model_200.safetensors").write_bytes(b"")
    result = mod.get_latest_checkpoint(tmp_path)
    assert result is not None
    assert result.name == "model_200.safetensors"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_ignores_non_safetensors(tmp_path):
    """Only .safetensors files count; .pt files must be ignored."""
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "model_999.pt").write_bytes(b"")  # should be ignored
    assert mod.get_latest_checkpoint(tmp_path) is None


# ---------------------------------------------------------------------------
# play_interactive.py — resolve_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_nonexistent_run(tmp_path):
    """Passing a non-existent explicit path returns None."""
    mod = _load_script("play_interactive")
    result = mod.resolve_checkpoint("MyTask", str(tmp_path / "no_run"))
    assert result is None


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_dir_with_model(tmp_path):
    """Directory path containing model_*.pt files resolves to the latest."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "2024-01-01_mujoco"
    run_dir.mkdir()
    (run_dir / "model_10.pt").write_bytes(b"")
    (run_dir / "model_50.pt").write_bytes(b"")

    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is not None
    assert "model_50.pt" in result


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_explicit_file(tmp_path):
    """Absolute path to existing .pt file returns that path unchanged."""
    mod = _load_script("play_interactive")
    model_file = tmp_path / "model_99.pt"
    model_file.write_bytes(b"")
    result = mod.resolve_checkpoint("MyTask", str(model_file))
    assert result == str(model_file)


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_empty_dir(tmp_path):
    """Directory with no model_*.pt files returns None."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is None
