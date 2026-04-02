from __future__ import annotations

import json
import sys
from pathlib import Path

from unilab.utils.experiment_tracking import ExperimentTracker, build_wandb_settings
from unilab.utils.offpolicy_logger import OffPolicyLogger
from unilab.utils.onpolicy_logger import OnPolicyLogger


class _FakeConfig(dict):
    def update(self, payload, allow_val_change=False):  # noqa: FBT002
        super().update(payload)


class _FakeRun:
    def __init__(self):
        self.summary = {}
        self.config = _FakeConfig()
        self.url = "https://wandb.local/run/test"


class _FakeVideo:
    def __init__(self, path: str, format: str = "mp4"):
        self.path = path
        self.format = format


class _FakeWandb:
    def __init__(self, existing_run: _FakeRun | None = None):
        self.run = existing_run
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, int | None]] = []
        self.finish_calls = 0

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run = _FakeRun()
        return self.run

    def log(self, payload, step=None):
        self.log_calls.append((payload, step))

    def finish(self):
        self.finish_calls += 1
        self.run = None

    def Video(self, path: str, format: str = "mp4"):  # noqa: N802
        return _FakeVideo(path, format=format)


def test_build_wandb_settings_defaults_for_shared_workspace():
    settings = build_wandb_settings(
        {"wandb_project": "unilab"},
        algo_name="ppo",
        task_name="Go1JoystickFlatTerrain",
        sim_backend="mujoco",
        log_dir="logs/rsl_rl_train/Go1JoystickFlatTerrain/2026-04-02_00-00-00_mujoco",
    )

    assert settings["project"] == "unilab"
    assert settings["group"] == "Go1JoystickFlatTerrain"
    assert settings["job_type"] == "ppo"
    assert settings["name"].startswith("ppo__Go1JoystickFlatTerrain__")
    assert "ppo" in settings["tags"]
    assert "Go1JoystickFlatTerrain" in settings["tags"]
    assert "mujoco" in settings["tags"]


def test_experiment_tracker_writes_local_run_files(tmp_path):
    log_dir = tmp_path / "logs" / "run1"
    tracker = ExperimentTracker(
        root_dir=tmp_path,
        log_dir=log_dir,
        algo_name="appo",
        task_name="G1MotionTracking",
        sim_backend="mujoco",
        training_cfg={"logger": "tensorboard"},
        full_cfg={"training": {"logger": "tensorboard"}},
        device="cuda",
        collector_device="cpu",
    )

    tracker.start()
    tracker.update_summary({"final_mean_reward": 12.3, "completed_iterations": 10})
    tracker.finish()

    run_config = json.loads((log_dir / "run_config.json").read_text(encoding="utf-8"))
    run_summary = json.loads((log_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert run_config["run"]["algo"] == "appo"
    assert run_config["run"]["task"] == "G1MotionTracking"
    assert run_summary["final_mean_reward"] == 12.3
    assert run_summary["completed_iterations"] == 10
    assert run_summary["wall_time_sec"] >= 0.0


def test_onpolicy_logger_reuses_existing_wandb_run(monkeypatch):
    fake_run = _FakeRun()
    fake_wandb = _FakeWandb(existing_run=fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OnPolicyLogger(
        algo_name="PPO",
        env_name="Go1JoystickFlatTerrain",
        log_backend="wandb",
    )

    assert logger._wandb_run is fake_run
    assert fake_wandb.init_calls == []

    logger.finish()
    assert fake_wandb.finish_calls == 0


def test_offpolicy_logger_reuses_existing_wandb_run(monkeypatch):
    fake_run = _FakeRun()
    fake_wandb = _FakeWandb(existing_run=fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="Go2JoystickFlatTerrain",
        log_backend="wandb",
    )

    assert logger._wandb_run is fake_run
    assert fake_wandb.init_calls == []

    logger.finish()
    assert fake_wandb.finish_calls == 0
