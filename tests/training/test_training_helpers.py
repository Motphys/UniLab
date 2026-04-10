from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.training import (
    BackendAdapter,
    get_latest_checkpoint,
    get_latest_run,
    parse_checkpoint_path,
    resolve_task_checkpoint_path,
)

_ROOT_DIR = Path(__file__).resolve().parents[2]
_CONF_DIR = _ROOT_DIR / "conf"


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


def _ppo_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _offpolicy_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "offpolicy"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides, offpolicy=True))


def test_get_latest_run_and_checkpoint_support_shared_checkpoint_resolution(tmp_path: Path):
    task_dir = tmp_path / "logs" / "custom_ppo" / "MyTask"
    older_run = task_dir / "2024-01-01_00-00-00_mujoco"
    newer_run = task_dir / "2024-02-01_00-00-00_mujoco"
    older_run.mkdir(parents=True)
    newer_run.mkdir(parents=True)
    (older_run / "model_1.pt").write_bytes(b"")
    (newer_run / "model_5.pt").write_bytes(b"")
    (newer_run / "model_9.pt").write_bytes(b"")

    latest_run = get_latest_run(task_dir)
    latest_checkpoint = get_latest_checkpoint(newer_run)

    assert latest_run == newer_run
    assert latest_checkpoint == newer_run / "model_9.pt"


def test_parse_checkpoint_path_uses_algo_log_name_from_cfg(tmp_path: Path):
    cfg = _ppo_cfg()
    cfg.algo.algo_log_name = "custom_ppo"

    run_dir = (
        tmp_path / "logs" / "custom_ppo" / cfg.training.task_name / "2024-01-01_00-00-00_mujoco"
    )
    run_dir.mkdir(parents=True)
    model_path = run_dir / "model_12.pt"
    model_path.write_bytes(b"")

    checkpoint_path, checkpoint_dir = parse_checkpoint_path(cfg, root_dir=tmp_path)

    assert checkpoint_path == model_path
    assert checkpoint_dir == run_dir


def test_resolve_task_checkpoint_path_supports_explicit_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_00-00-00_mujoco"
    run_dir.mkdir(parents=True)
    (run_dir / "model_9.pt").write_bytes(b"")
    selected_model = run_dir / "model_12.pt"
    selected_model.write_bytes(b"")

    checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
        tmp_path,
        task_name="MyTask",
        load_run="-1",
        algo_log_name="custom_ppo",
        checkpoint="12",
    )

    assert checkpoint_path == selected_model
    assert checkpoint_dir == run_dir


def test_resolve_task_checkpoint_path_returns_run_dir_when_checkpoint_missing(tmp_path: Path):
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_00-00-00_mujoco"
    run_dir.mkdir(parents=True)

    checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
        tmp_path,
        task_name="MyTask",
        load_run="-1",
        algo_log_name="custom_ppo",
        checkpoint="12",
    )

    assert checkpoint_path is None
    assert checkpoint_dir == run_dir


def test_backend_adapter_env_cfg_override_for_motrix_sac_go1():
    """Env cfg override carries reward + env preset fields. Algo is NOT touched."""
    cfg = _offpolicy_cfg(["task=sac/go1_joystick/motrix"])

    adapter = BackendAdapter(cfg, root_dir=_ROOT_DIR, algo_name="sac")
    env_cfg_override = adapter.build_task_env_cfg_override()

    # env_cfg_override has reward + env preset fields
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)
    assert env_cfg_override["commands"]["vel_limit"] == [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    # algo values come straight from YAML compose — no mutation, matches old motrix behavior
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 2000


def test_backend_adapter_builds_play_scene_override():
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix", "training.play_only=true"])
    assert cfg.training.play_env_num == 128
    captured: dict[str, object] = {}

    def _fake_materializer(source_model_file: str, **kwargs) -> str:
        captured["source_model_file"] = source_model_file
        captured.update(kwargs)
        return "/tmp/g1_motion_tracking_play_scene.xml"

    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=_ROOT_DIR,
        algo_name="ppo",
        scene_materializer=_fake_materializer,
    ).build_play_env_cfg_override()

    assert cfg.training.play_env_num == 128
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)
    assert env_cfg_override["model_file"] == "/tmp/g1_motion_tracking_play_scene.xml"
    assert captured["ground_texture_file"] == str(
        _ROOT_DIR / "src/unilab/assets/robots/g1/floor.png"
    )
