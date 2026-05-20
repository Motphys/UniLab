from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import unilab.logging.common as common_module
import unilab.logging.offpolicy as offpolicy_module
from unilab.logging import OffPolicyLogger, OnPolicyLogger
from unilab.training.experiment import ExperimentTracker, build_wandb_settings


class _FakeConfig(dict):
    def update(self, *args: Any, allow_val_change: bool = False, **kwargs: Any) -> None:  # noqa: FBT002
        del allow_val_change
        super().update(*args, **kwargs)


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


class _FakeTensorBoardWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.close_calls = 0

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def close(self) -> None:
        self.close_calls += 1


def test_training_logger_defers_initial_live_render(monkeypatch):
    start_refresh_values: list[bool] = []

    class _FakeLive:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self, *, refresh: bool = False) -> None:
            start_refresh_values.append(refresh)

        def update(self, *args, **kwargs) -> None:
            del args, kwargs

        def stop(self) -> None:
            pass

    monkeypatch.setattr(common_module, "Live", _FakeLive)

    logger = OffPolicyLogger(log_backend="none")
    logger.start()
    logger.close()

    assert start_refresh_values == [False]


def test_offpolicy_training_terminal_refresh_uses_single_low_frequency_trigger(monkeypatch):
    update_refresh_values: list[bool | None] = []
    now = 100.0

    class _FakeLive:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self, *, refresh: bool = False) -> None:
            del refresh

        def update(self, *args, **kwargs) -> None:
            del args
            update_refresh_values.append(kwargs.get("refresh"))

        def stop(self) -> None:
            pass

    def _fake_time() -> float:
        return now

    monkeypatch.setattr(common_module, "Live", _FakeLive)
    monkeypatch.setattr(common_module.time, "time", _fake_time)

    logger = OffPolicyLogger(log_backend="none", refresh_per_second=4)
    logger.start()
    logger.log_step(iteration=1, train_time=0.01, wait_time=0.0)
    assert update_refresh_values == [True]

    logger.log_collector(total_steps=128, buffer_size=128, mean_reward=2.0)
    logger.log_status("Collector metrics updated")
    logger.log_save("/tmp/model_2.pt")
    assert update_refresh_values == [True]

    now += 0.3
    logger.log_step(iteration=2, train_time=0.01, wait_time=0.0)
    assert update_refresh_values == [True, True]

    logger.log_status("[red]ERROR: Collector died[/]")
    assert update_refresh_values == [True, True, True]

    logger.close()


def test_build_wandb_settings_defaults_for_shared_workspace():
    settings = build_wandb_settings(
        {"wandb_project": "unilab"},
        algo_name="ppo",
        task_name="Go1JoystickFlat",
        sim_backend="mujoco",
        log_dir="logs/rsl_rl_train/Go1JoystickFlat/2026-04-02_00-00-00_mujoco",
    )

    assert settings["project"] == "unilab"
    assert settings["group"] == "Go1JoystickFlat"
    assert settings["job_type"] == "ppo"
    assert settings["name"].startswith("ppo__Go1JoystickFlat__")
    assert "ppo" in settings["tags"]
    assert "Go1JoystickFlat" in settings["tags"]
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
        seed_info={
            "configured_seed": 5,
            "configured_seed_source": "algo.seed",
            "effective_seed": 5,
        },
    )

    tracker.start()
    tracker.update_summary({"final_mean_reward": 12.3, "completed_iterations": 10})
    tracker.finish()

    run_config = json.loads((log_dir / "run_config.json").read_text(encoding="utf-8"))
    run_summary = json.loads((log_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert run_config["run"]["algo"] == "appo"
    assert run_config["run"]["task"] == "G1MotionTracking"
    assert run_config["run"]["configured_seed"] == 5
    assert run_config["run"]["configured_seed_source"] == "algo.seed"
    assert run_config["run"]["effective_seed"] == 5
    assert run_summary["final_mean_reward"] == 12.3
    assert run_summary["completed_iterations"] == 10
    assert run_summary["configured_seed"] == 5
    assert run_summary["effective_seed"] == 5
    assert run_summary["wall_time_sec"] >= 0.0


def test_onpolicy_logger_reuses_existing_wandb_run(monkeypatch):
    fake_run = _FakeRun()
    fake_wandb = _FakeWandb(existing_run=fake_run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OnPolicyLogger(
        algo_name="PPO",
        env_name="Go1JoystickFlat",
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
        env_name="Go2JoystickFlat",
        log_backend="wandb",
    )

    assert logger._wandb_run is fake_run
    assert fake_wandb.init_calls == []

    logger.finish()
    assert fake_wandb.finish_calls == 0


def test_onpolicy_logger_creates_and_finishes_owned_wandb_run(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OnPolicyLogger(
        algo_name="PPO",
        env_name="Go1JoystickFlat",
        log_backend="wandb",
        wandb_project="unilab",
        wandb_entity="team",
        wandb_name="ppo-go1",
        wandb_group="go1",
        wandb_job_type="train",
        wandb_tags=["ppo", "go1"],
        wandb_notes="notes",
    )

    assert logger._owns_wandb_run is True
    assert len(fake_wandb.init_calls) == 1
    init_call = fake_wandb.init_calls[0]
    assert init_call["project"] == "unilab"
    assert init_call["entity"] == "team"
    assert init_call["name"] == "ppo-go1"
    assert init_call["group"] == "go1"
    assert init_call["job_type"] == "train"
    assert init_call["tags"] == ["ppo", "go1"]
    assert init_call["notes"] == "notes"
    assert init_call["config"]["algo"] == "PPO"
    assert init_call["config"]["env"] == "Go1JoystickFlat"
    assert init_call["config"]["num_envs"] == 4096

    logger.finish()
    assert fake_wandb.finish_calls == 1


def test_offpolicy_logger_creates_and_finishes_owned_wandb_run(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="Go2JoystickFlat",
        log_backend="wandb",
        obs_dim=48,
        action_dim=12,
        max_iterations=321,
        wandb_project="unilab",
        wandb_entity="team",
        wandb_name="sac-go2",
        wandb_group="go2",
        wandb_job_type="train",
        wandb_tags=["sac", "go2"],
        wandb_notes="notes",
    )

    assert logger._owns_wandb_run is True
    assert len(fake_wandb.init_calls) == 1
    init_call = fake_wandb.init_calls[0]
    assert init_call["project"] == "unilab"
    assert init_call["entity"] == "team"
    assert init_call["name"] == "sac-go2"
    assert init_call["group"] == "go2"
    assert init_call["job_type"] == "train"
    assert init_call["tags"] == ["sac", "go2"]
    assert init_call["notes"] == "notes"
    assert init_call["config"]["algo"] == "FastSAC"
    assert init_call["config"]["env"] == "Go2JoystickFlat"
    assert init_call["config"]["num_envs"] == 4096
    assert init_call["config"]["obs_dim"] == 48
    assert init_call["config"]["action_dim"] == 12
    assert init_call["config"]["max_iterations"] == 321

    logger.finish()
    assert fake_wandb.finish_calls == 1


def test_offpolicy_logger_close_releases_owned_wandb_run_once(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="Go2JoystickFlat",
        log_backend="wandb",
    )

    logger.close()
    assert fake_wandb.finish_calls == 1

    logger.finish()
    assert fake_wandb.finish_calls == 1


def test_offpolicy_logger_logs_wait_and_iter_throughput(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OffPolicyLogger(
        algo_name="APPO",
        env_name="Go2JoystickFlat",
        num_envs=2,
        log_backend="wandb",
    )
    logger.log_step(
        iteration=1,
        metrics={},
        train_time=0.75,
        wait_time=10.0,
        learner_incremental_h2d_time=0.02,
        weight_sync_time=0.05,
        extra_info={"throughput_steps": 8},
    )

    payload, step = fake_wandb.log_calls[-1]
    assert step == 1
    assert payload["timing/learner_wait_ms"] == 10_000.0
    assert "timing/learner_collect_ms" not in payload
    assert payload["timing/learner_incremental_h2d_ms"] == 20.0
    assert payload["timing/learner_train_ms"] == 750.0
    assert payload["timing/learner_weight_sync_ms"] == 50.0
    assert payload["perf/iter_ms"] == pytest.approx(820.0)
    assert payload["perf/steps_per_sec"] == pytest.approx(8.0 / 0.82)
    assert "perf/collect_train_ratio" not in payload

    logger.finish()


def test_offpolicy_logger_logs_collector_phase_timing_to_backends(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wandb_logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="Go2JoystickFlat",
        log_backend="wandb",
    )
    wandb_logger.update_collector_timing({"replay_ms": 1.25})
    wandb_logger.log_step(iteration=3, metrics={}, train_time=0.1)

    payload, _ = fake_wandb.log_calls[-1]
    assert payload["timing/collector_replay_ms"] == pytest.approx(1.25)
    wandb_logger.finish()

    tb_writer = _FakeTensorBoardWriter()
    tb_logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="Go2JoystickFlat",
        log_backend="none",
    )
    tb_logger._tb_writer = tb_writer
    tb_logger.update_collector_timing({"replay_ms": 2.5})
    tb_logger.log_step(iteration=4, metrics={}, train_time=0.1)

    assert ("timing/collector_replay_ms", 2.5, 4) in tb_writer.scalars
    tb_logger.finish()


def test_offpolicy_logger_logs_reward_comparison_metrics(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="G1WalkFlat",
        log_backend="wandb",
    )
    logger.log_collector(total_steps=128, buffer_size=128)
    logger.log_step(
        iteration=2,
        metrics={},
        reward=3.0,
        reward_metrics={"mean_ep100": 2.0},
    )

    payload, step = fake_wandb.log_calls[-1]
    assert step == 128
    assert payload["reward/mean"] == 3.0
    assert payload["reward/mean_ep100"] == 2.0
    assert "reward/mean_unilab_100x100" not in payload

    logger.finish()


def test_offpolicy_logger_omits_iteration_extra_fields_when_not_supplied(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="Go2JoystickFlat",
        log_backend="wandb",
    )
    logger._start_time = 1.0
    monkeypatch.setattr(offpolicy_module.time, "time", lambda: 2.0)
    logger.log_collector(total_steps=8, buffer_size=8)
    logger.log_step(
        iteration=1,
        metrics={},
        train_time=0.75,
        wait_time=1.0,
        learner_incremental_h2d_time=0.02,
        weight_sync_time=0.05,
    )

    payload, _ = fake_wandb.log_calls[-1]
    assert "timing/learner_collect_ms" not in payload
    assert "perf/steps_per_sec" not in payload

    logger.finish()
