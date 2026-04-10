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
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_CONF_DIR = Path(__file__).parent.parent.parent / "conf"


def _normalize_overrides(overrides: list[str] | None, *, offpolicy: bool = False) -> list[str]:
    normalized: list[str] = []
    algo = "sac"
    task_selected = False

    for override in overrides or []:
        if override.startswith("algo="):
            algo = override.split("=", 1)[1]
            normalized.append(override)
            continue
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        if offpolicy:
            normalized.append(f"task={algo}/go1_joystick/mujoco")
        else:
            normalized.append("task=go1_joystick/mujoco")
    return normalized


def _load_script(name: str) -> Any:
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

    _HAS_MLX = _sys.platform == "darwin" and importlib.util.find_spec("mlx.core") is not None
except Exception:
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
        return compose("config", overrides=_normalize_overrides(overrides, offpolicy=True))


def _ppo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _appo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "appo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _train_rsl_rl(monkeypatch: pytest.MonkeyPatch):
    import types

    for module_name in list(sys.modules):
        if module_name == "unilab" or module_name.startswith("unilab."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    runners_mod = cast(Any, types.ModuleType("rsl_rl.runners"))
    runners_mod.OnPolicyRunner = object
    rsl_pkg = cast(Any, types.ModuleType("rsl_rl"))
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
    assert cfg.algo.load_run == "-1"


def test_offpolicy_hydra_algo_td3():
    cfg = _offpolicy_cfg(["algo=td3"])
    assert cfg.algo.algo == "td3"


def test_offpolicy_go1_resolved_algo_matches_old_motrix_behavior():
    """Equivalence: Motrix SAC Go1 algo hyperparams match legacy values."""
    cfg = _offpolicy_cfg(["task=sac/go1_joystick/motrix"])

    # Legacy Motrix SAC Go1 values: num_envs=4096, max_iterations=2000
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 2000


def test_offpolicy_go1_env_cfg_override_has_reward_and_commands():
    cfg = _offpolicy_cfg(["task=sac/go1_joystick/motrix"])

    env_cfg_override = _offpolicy().build_offpolicy_env_cfg_override("sac", cfg)

    # env_cfg_override has reward + env preset fields (no bag structure)
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)
    assert env_cfg_override["commands"]["vel_limit"] == [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


def test_offpolicy_g1_sac_backend_scoped_use_symmetry():
    mujoco_cfg = _offpolicy_cfg(["task=sac/g1_sac/mujoco"])
    motrix_cfg = _offpolicy_cfg(["task=sac/g1_sac/motrix"])

    assert mujoco_cfg.algo.use_symmetry is True
    assert motrix_cfg.algo.use_symmetry is False


def test_ppo_go1_resolved_algo_matches_old_motrix_behavior():
    """Equivalence: PPO Go1 algo hyperparams match pre-refactor motrix values."""
    cfg = _ppo_cfg(["task=go1_joystick/motrix"])

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)


def test_ppo_g1_resolved_algo_matches_old_motrix_behavior():
    """Equivalence: PPO G1 algo hyperparams match pre-refactor motrix values.

    In particular, max_iterations=151 (the motrix value from the old bag),
    NOT 220 (the old mujoco base value which has been retired).
    """
    cfg = _ppo_cfg(["task=g1_joystick/motrix"])

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.actor == ["policy"]
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(5.0e-3)


def test_ppo_g1_mujoco_base_hyperparams_remain_separate():
    cfg = _ppo_cfg(["task=g1_joystick/mujoco"])

    assert cfg.algo.max_iterations == 220
    assert cfg.algo.empirical_normalization is False
    assert cfg.algo.obs_groups.actor == ["actor"]


def test_ppo_g1_env_preset_has_env_overrides():
    cfg = _ppo_cfg(["task=g1_joystick/motrix"])

    assert cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert cfg.env.gait_phase_init_mode == "independent"
    assert cfg.env.reset_base_qvel_limit == pytest.approx(0.05)


def test_ppo_task_go2_aligns_mujoco_with_motrix_defaults():
    cfg = _ppo_cfg(["task=go2_joystick/mujoco"])

    assert cfg.algo.num_envs == 1024
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.reward.scales.tracking_ang_vel == pytest.approx(0.2)
    assert cfg.reward.scales.lin_vel_z == pytest.approx(-5.0)
    assert cfg.reward.scales.ang_vel_xy == pytest.approx(-0.1)
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)


def test_build_ppo_env_cfg_override_go1_motrix(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go1_joystick/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    # env_cfg_override has reward + env preset commands
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)
    assert env_cfg_override["commands"]["vel_limit"] == [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


def test_build_ppo_env_cfg_override_g1_motrix(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_joystick/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    # env_cfg_override has reward + env preset fields (flat, matching env cfg structure)
    assert env_cfg_override["reward_config"]["scales"]["upper_body_pose"] == pytest.approx(-0.05)
    assert env_cfg_override["control_config"]["action_scale"] == pytest.approx(0.5)
    assert env_cfg_override["gait_phase_init_mode"] == "independent"
    assert env_cfg_override["reset_base_qvel_limit"] == pytest.approx(0.05)


def test_build_ppo_env_cfg_override_applies_go2_reward_motrix(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go2_joystick/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.algo.num_envs == 1024
    assert env_cfg_override["domain_rand"]["randomize_kp"] is False
    assert env_cfg_override["domain_rand"]["randomize_kd"] is False
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)
    assert env_cfg_override["reward_config"]["scales"]["tracking_ang_vel"] == pytest.approx(0.2)


def test_ppo_cli_algo_override_wins_over_base(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLI override takes precedence over base task algo values via Hydra compose."""
    cfg = _ppo_cfg(["task=g1_joystick/motrix", "algo.max_iterations=1"])

    assert cfg.algo.max_iterations == 1
    # Other base values remain intact
    assert cfg.algo.empirical_normalization is True


def test_g1_motion_tracking_ppo_motrix_prefers_backend_specific_reward(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)
    cfg.reward.scales.motion_body_pos = 1.25

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(1.25)


def test_build_ppo_play_env_cfg_override_applies_g1_motion_tracking_play_profile(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix", "training.play_only=true"])

    monkeypatch.setattr(
        mod,
        "materialize_scene_visual_override",
        lambda source_model_file, **kwargs: "/tmp/g1_motion_tracking_play_scene.xml",
    )

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert cfg.training.play_env_num == 128
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)
    assert env_cfg_override["model_file"] == "/tmp/g1_motion_tracking_play_scene.xml"
    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(1.0)


def test_build_ppo_play_env_cfg_override_respects_cli_play_env_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=g1_motion_tracking/motrix",
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

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert cfg.training.play_env_num == 32
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)


def test_build_ppo_play_env_cfg_override_resolves_relative_ground_texture(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix", "training.play_only=true"])
    cfg.motrix_play_only.scene_override.ground_texture_file = (
        "src/unilab/assets/robots/g1/floor.png"
    )

    captured = {}

    def _fake_materialize(source_model_file, **kwargs):
        captured["source_model_file"] = source_model_file
        captured.update(kwargs)
        return "/tmp/g1_motion_tracking_play_scene.xml"

    monkeypatch.setattr(mod, "materialize_scene_visual_override", _fake_materialize)

    mod.build_ppo_play_env_cfg_override(cfg)

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

    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)
    cfg.reward.scales.motion_body_pos = 1.5

    env_cfg_override = extract_reward_config(cfg)

    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(1.5)


def test_g1_motion_tracking_ppo_task_exposes_final_reward():
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)


def test_g1_motion_tracking_appo_task_exposes_final_reward():
    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# train_appo.py — motrix runner / play helpers
# ---------------------------------------------------------------------------


def test_build_appo_runner_kwargs_forwards_sim_backend():
    mod = _train_appo()
    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

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


# ---------------------------------------------------------------------------
# play_interactive.py — RslRlVecEnvWrapper contract behavior
# ---------------------------------------------------------------------------


def _play_interactive():
    """Load play_interactive.py as a module."""
    return _load_script("play_interactive")


def test_play_wrapper_imports_shared_implementation():
    """Verify play_interactive.py uses shared RslRlVecEnvWrapper."""
    from unilab.utils.rsl_rl_vec_env_wrapper import RslRlVecEnvWrapper as SharedWrapper

    mod = _play_interactive()
    # The wrapper class in play_interactive should be the shared one
    assert mod.RslRlVecEnvWrapper is SharedWrapper


def test_play_wrapper_uses_current_reset_contract():
    """Verify wrapper reset() uses current (obs, info) contract, not old (_, obs, _)."""
    import numpy as np
    from tensordict import TensorDict

    from unilab.utils.rsl_rl_vec_env_wrapper import RslRlVecEnvWrapper

    # Create a fake environment that returns (obs, info) tuple
    class FakeEnv:
        def __init__(self):
            self.num_envs = 2
            self.state = type("State", (), {"obs": {"obs": np.ones((2, 5), dtype=np.float32)}})()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (3,)})()
            self.obs_groups_spec = {"obs": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            # Returns current contract: (obs, info)
            return {"obs": np.ones((2, 5), dtype=np.float32)}, {}

    env = FakeEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="flat")

    # Reset should work with current contract
    obs_td, info = wrapper.reset()

    assert isinstance(obs_td, TensorDict)
    assert "policy" in obs_td
    assert "actor" in obs_td
    assert obs_td.batch_size == (2,)


def test_play_wrapper_policy_obs_mode_actor():
    """Verify wrapper supports policy_obs_mode='actor'."""
    import numpy as np

    from unilab.utils.rsl_rl_vec_env_wrapper import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type("State", (), {"obs": {"obs": np.ones((1, 3), dtype=np.float32)}})()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (3,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "privileged": 2}

        def init_state(self):
            pass

        def reset(self, env_indices):
            return {
                "obs": np.ones((1, 3), dtype=np.float32),
                "privileged": np.zeros((1, 2), dtype=np.float32),
            }, {}

    env = FakeEnv()

    # Test actor mode - num_obs should match actor obs dim only
    wrapper_actor = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="actor")
    assert wrapper_actor.num_obs == 3  # Only "obs" group
    assert wrapper_actor._actor_obs_dim == 3
    assert wrapper_actor._flat_obs_dim == 5  # obs + privileged

    obs_td, _ = wrapper_actor.reset()
    # In actor mode, policy obs should equal actor obs
    assert obs_td["policy"].shape == (1, 3)
    assert obs_td["actor"].shape == (1, 3)


# ---------------------------------------------------------------------------
# Issue #168: Unified log directory and load_run resolution
# ---------------------------------------------------------------------------


def test_ppo_hydra_default_algo_log_name():
    """Verify PPO config has algo_log_name in algo section."""
    cfg = _ppo_cfg()
    assert cfg.algo.algo_log_name == "rsl_rl_ppo"


def test_ppo_hydra_load_run_in_algo_not_training():
    """Verify load_run is in algo section, not training section (issue #168)."""
    from omegaconf import OmegaConf

    cfg = _ppo_cfg()
    assert cfg.algo.load_run == "-1"
    # training section should NOT have load_run anymore
    assert "load_run" not in cfg.training or OmegaConf.is_missing(cfg.training, "load_run")


def test_appo_hydra_default_algo_log_name():
    """Verify APPO config has algo_log_name in algo section."""
    cfg = _appo_cfg()
    assert cfg.algo.algo_log_name == "appo"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_sac_hydra_default_algo_log_name():
    """Verify SAC config has algo_log_name in algo section."""
    cfg = _offpolicy_cfg(["algo=sac"])
    assert cfg.algo.algo_log_name == "fast_sac"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_td3_hydra_default_algo_log_name():
    """Verify TD3 config has algo_log_name in algo section."""
    cfg = _offpolicy_cfg(["algo=td3"])
    assert cfg.algo.algo_log_name == "fast_td3"
    assert cfg.algo.load_run == "-1"


def test_train_rsl_rl_get_log_root_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):
    """Verify _get_log_root uses algo.algo_log_name (issue #168)."""
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg()

    # Override algo_log_name to test
    cfg.algo.algo_log_name = "test_rsl_rl_ppo"

    log_root = mod._get_log_root(cfg)
    assert "logs/test_rsl_rl_ppo" in log_root


def test_train_appo_get_log_root_uses_algo_log_name():
    """Verify APPO _get_log_root uses algo.algo_log_name (issue #168)."""
    mod = _train_appo()
    cfg = _appo_cfg()

    cfg.algo.algo_log_name = "test_appo"

    log_root = mod._get_log_root(cfg)
    assert "logs/test_appo" in log_root


def test_play_resolve_checkpoint_uses_algo_log_name(tmp_path):
    """Verify play_interactive.resolve_checkpoint uses algo_log_name (issue #168)."""
    mod = _play_interactive()

    # Create test directory structure with custom algo_log_name
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_mujoco"
    run_dir.mkdir(parents=True)
    (run_dir / "model_50.pt").write_bytes(b"")

    # Temporarily override ROOT_DIR to use tmp_path
    original_root = mod.ROOT_DIR
    try:
        mod.ROOT_DIR = tmp_path
        result = mod.resolve_checkpoint("MyTask", "-1", algo_log_name="custom_ppo")
        assert result is not None
        assert "model_50.pt" in result
    finally:
        mod.ROOT_DIR = original_root
