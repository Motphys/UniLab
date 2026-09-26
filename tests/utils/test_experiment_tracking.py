from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest
import uni_rl.logging.common as common_module
from rich.console import Console
from uni_rl.logging import OffPolicyLogger, OnPolicyLogger

import unilab.training.experiment as experiment_module
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
    live_kwargs: list[dict[str, Any]] = []

    class _FakeLive:
        def __init__(self, *args, **kwargs):
            del args
            live_kwargs.append(kwargs)

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
    assert live_kwargs[0]["auto_refresh"] is True
    assert live_kwargs[0]["refresh_per_second"] == 2
    assert callable(live_kwargs[0]["get_renderable"])


def test_offpolicy_training_terminal_uses_fixed_clock_and_only_forces_errors(monkeypatch):
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
    logger.log_step(iteration=1, train_time=0.01, collector_wait_time=0.0)
    assert update_refresh_values == []

    logger.log_collector(total_steps=128, buffer_size=128)
    logger.log_status("Collector metrics updated")
    logger.log_save("/tmp/model_2.pt")
    assert update_refresh_values == []

    now += 0.3
    logger.log_step(iteration=2, train_time=0.01, collector_wait_time=0.0)
    assert update_refresh_values == []

    logger.log_status("[red]ERROR: Collector died[/]")
    assert update_refresh_values == [True]

    logger.close()


def test_onpolicy_logger_uses_offpolicy_terminal_layout():
    logger = OnPolicyLogger(
        algo_name="PPO",
        env_name="Go2JoystickFlat",
        max_iterations=10,
        num_envs=4,
        num_steps=8,
        log_backend="no_print",
    )
    logger.start()
    logger.log_step(
        iteration=1,
        metrics={"Loss/surrogate": 0.1, "Loss/value": 0.2},
        return_mean_ep100=3.0,
        collect_time=0.02,
        train_time=0.03,
    )
    logger.update_mean_episode_length(12.0)

    console = Console(record=True, width=120)
    console.print(logger._build_display())
    output = console.export_text()

    assert "Losses & Metrics" in output
    assert "Learner" in output
    assert "Collector" in output
    assert "Train" in output
    assert "Collect" in output
    assert "Steps/s" in output
    assert "Rollout Steps" not in output
    assert "Steps/Env" not in output
    assert "Policy Metrics" not in output

    logger.close()


def test_offpolicy_logger_terminal_keeps_core_bottleneck_timing_rows():
    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="G1WalkFlat",
        max_iterations=10,
        num_envs=4,
        log_backend="no_print",
    )
    logger.log_step(
        iteration=1,
        metrics={"Loss/critic": 0.1},
        train_time=0.5,
        collector_wait_time=1.0,
        replay_batch_wait_time=0.005,
        learner_replay_sample_time=0.005,
        sync_coordination_time=0.005,
        replay_ingress_h2d_submit_time=0.01,
        weight_sync_time=0.005,
        inference_h2d_time=0.01,
        inference_forward_time=0.16,
        inference_d2h_time=0.01,
        inference_time=0.18,
        iteration_time=1.6,
        extra_info={
            "throughput_steps": 8,
            "batch_size_per_rank": 8,
            "effective_batch_size": 8,
            "learner_replay_rows_per_iter": 64,
        },
    )

    console = Console(record=True, width=140)
    console.print(logger._build_display())
    output = console.export_text()

    assert "Replay Wait" not in output
    assert "Collector Wait" in output
    assert "Inference" in output
    assert "Learning" in output
    assert "Iter Wall" in output
    assert "Replay Batch Wait" in output
    assert "Replay Sample" in output
    assert "Inference H2D" not in output
    assert "Inference Forward" not in output
    assert "Inference D2H" not in output
    assert "Collector Release" in output
    assert "H2D Copy" not in output
    assert "Weight Publish" not in output
    assert "Rank Barrier" not in output
    assert "Param Sync" not in output
    assert "Unaccounted" not in output
    assert "Other Loop" not in output
    assert "GPUs 1" in output
    assert "Sync Mode" not in output
    assert "Sync Interval" not in output
    assert "Sync Collect" not in output
    assert "Terminated Rate" not in output
    assert "Staging Pool" not in output
    assert "Batch/Rank" in output
    assert "Batch/Update" not in output
    assert "Samples/Iter" not in output
    assert "Rows/s" in output

    logger.close()


def test_offpolicy_logger_terminal_shows_material_blocking_phases():
    logger = OffPolicyLogger(log_backend="no_print")
    logger.log_step(
        iteration=1,
        train_time=0.5,
        collector_wait_time=0.3,
        replay_batch_wait_time=0.01,
        learner_replay_sample_time=0.01,
        sync_coordination_time=0.01,
        replay_ingress_h2d_submit_time=0.01,
        weight_sync_time=0.01,
        iteration_time=1.0,
    )

    console = Console(record=True, width=140)
    console.print(logger._build_display())
    output = console.export_text()

    for label in (
        "Replay Batch Wait",
        "Replay Sample",
        "Collector Release",
    ):
        assert label in output
    assert "Replay Stage" not in output
    assert "Weight Publish" not in output

    logger.close()


def test_offpolicy_logger_terminal_shows_replay_rows_and_effective_batch():
    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="G1WalkFlat",
        max_iterations=10,
        num_envs=4,
        log_backend="no_print",
    )
    logger.log_step(
        iteration=1,
        metrics={"Loss/critic": 0.1},
        train_time=0.5,
        collector_wait_time=0.1,
        iteration_time=1.0,
        extra_info={
            "throughput_steps": 4,
            "batch_size_per_rank": 16,
            "effective_batch_size": 16,
            "learner_replay_rows_per_iter": 16,
        },
    )

    console = Console(record=True, width=120)
    console.print(logger._build_display())
    output = console.export_text()

    assert "Batch/Rank" in output
    assert "Replay/Iter" not in output
    assert "Samples/Iter" not in output
    assert "Rows/s" in output
    assert "Rank Barrier" not in output
    assert "Param Sync" not in output
    assert "Other Loop" not in output

    logger.close()


def test_build_wandb_settings_defaults_for_shared_workspace():
    settings = build_wandb_settings(
        {"wandb_project": "unilab"},
        algo_name="ppo",
        task_name="Go2JoystickFlat",
        sim_backend="mujoco",
        log_dir="logs/rsl_rl_train/Go2JoystickFlat/2026-04-02_00-00-00_mujoco",
    )

    assert settings["project"] == "unilab"
    assert settings["group"] == "Go2JoystickFlat"
    assert settings["job_type"] == "ppo"
    assert settings["name"].startswith("ppo__Go2JoystickFlat__")
    assert "ppo" in settings["tags"]
    assert "Go2JoystickFlat" in settings["tags"]
    assert "mujoco" in settings["tags"]


def test_experiment_device_info_uses_library_helper():
    import unilab.utils.device as device_module

    assert experiment_module.get_device_info_dict is device_module.get_device_info_dict


def test_experiment_tracker_writes_local_run_files(tmp_path, monkeypatch):
    hardware = {
        "platform": "test-platform",
        "chip": "test-cpu",
        "cpu_total_cores": "16",
        "gpu_name": "test-gpu",
        "memory": "64 GB",
    }
    monkeypatch.setattr(experiment_module, "get_device_info_dict", lambda: hardware)
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
    assert run_config["run"]["hardware"] == hardware
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
        env_name="Go2JoystickFlat",
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
        env_name="Go2JoystickFlat",
        log_backend="wandb",
        wandb_project="unilab",
        wandb_entity="team",
        wandb_name="ppo-go2",
        wandb_group="go2",
        wandb_job_type="train",
        wandb_tags=["ppo", "go2"],
        wandb_notes="notes",
    )

    assert logger._owns_wandb_run is True
    assert len(fake_wandb.init_calls) == 1
    init_call = fake_wandb.init_calls[0]
    assert init_call["project"] == "unilab"
    assert init_call["entity"] == "team"
    assert init_call["name"] == "ppo-go2"
    assert init_call["group"] == "go2"
    assert init_call["job_type"] == "train"
    assert init_call["tags"] == ["ppo", "go2"]
    assert init_call["notes"] == "notes"
    assert init_call["config"]["algo"] == "PPO"
    assert init_call["config"]["env"] == "Go2JoystickFlat"
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
        timing_profile="appo",
    )
    logger.log_step(
        iteration=1,
        metrics={},
        train_time=0.75,
        collector_wait_time=10.0,
        learner_replay_stage_time=0.02,
        weight_sync_time=0.05,
        iteration_time=10.9,
        extra_info={
            "throughput_steps": 8,
            "batch_size_per_rank": 8,
            "effective_batch_size": 8,
            "learner_replay_rows_per_iter": 32,
        },
    )

    payload, step = fake_wandb.log_calls[-1]
    assert step == 1
    assert payload["Perf/learner_collector_wait_ms"] == 10_000.0
    assert "timing/learner_collect_ms" not in payload
    assert "timing/learner_replay_wait_ms" not in payload
    assert payload["Perf/learner_replay_stage_ms"] == 20.0
    assert payload["Perf/learner_replay_sample_ms"] == 0.0
    assert payload["Perf/learning_time"] == pytest.approx(0.75)
    assert "timing/learner_param_sync_ms" not in payload
    assert payload["Perf/learner_weight_publish_ms"] == 50.0
    for key in (
        "Perf/learner_inference_ms",
        "Perf/learner_collector_release_ms",
        "Perf/learner_replay_batch_wait_ms",
        "Perf/learner_inference_h2d_ms",
        "Perf/learner_inference_forward_ms",
        "Perf/learner_inference_d2h_ms",
        "Perf/replay_ingress_h2d_submit_ms",
    ):
        assert key not in payload
    assert "timing/learner_other_ms" not in payload
    assert "perf/learner_pipeline_ms" not in payload
    assert payload["Perf/iteration_time"] == pytest.approx(10.9)
    assert "perf/iter_ms" not in payload
    assert "perf/iter_unaccounted_ms" not in payload
    assert payload["Perf/total_fps"] == pytest.approx(8.0 / 10.9)
    for key in ("axis/iteration", "axis/env_steps_total", "Train/iteration"):
        assert key not in payload
    assert not any(key.startswith("distributed/") for key in payload)
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
    wandb_logger.update_collector_timing({"replay_write_ms": 1.25})
    wandb_logger.log_step(iteration=3, metrics={}, train_time=0.1)

    payload, _ = fake_wandb.log_calls[-1]
    assert payload["Perf/collector_replay_write_ms"] == pytest.approx(1.25)
    wandb_logger.finish()

    tb_writer = _FakeTensorBoardWriter()
    tb_logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="Go2JoystickFlat",
        log_backend="none",
    )
    tb_logger._tb_writer = tb_writer
    tb_logger.update_collector_timing({"replay_write_ms": 2.5})
    tb_logger.log_step(iteration=4, metrics={}, train_time=0.1)

    assert ("Perf/collector_replay_write_ms", 2.5, 4) in tb_writer.scalars
    tb_logger.finish()


def test_offpolicy_logger_uses_same_canonical_timing_names_in_terminal_and_backends(
    monkeypatch,
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="G1WalkFlat",
        log_backend="wandb",
    )
    logger.update_collector_timing(
        {
            "inference_request_ms": 1.0,
            "learner_action_wait_ms": 2.0,
            "env_step_ms": 3.0,
            "replay_write_ms": 4.0,
        }
    )
    logger.log_step(
        iteration=1,
        collector_wait_time=0.01,
        inference_time=0.02,
        sync_coordination_time=0.03,
        replay_batch_wait_time=0.04,
        learner_replay_stage_time=0.08,
        learner_replay_sample_time=0.05,
        train_time=0.06,
        weight_sync_time=0.07,
        iteration_time=0.30,
    )

    payload, _ = fake_wandb.log_calls[-1]
    learner_labels = list(logger._build_timing_table().columns[0].cells)
    collector_labels = list(logger._build_timing_table().columns[2].cells)

    assert learner_labels[:8] == [
        "Collector Wait",
        "Inference",
        "Collector Release",
        "Replay Batch Wait",
        "Replay Sample",
        "Learning",
        "Other",
        "Iter Wall",
    ]
    for key in (
        "Perf/learner_collector_wait_ms",
        "Perf/learner_inference_ms",
        "Perf/learner_collector_release_ms",
        "Perf/learner_replay_batch_wait_ms",
        "Perf/learner_replay_sample_ms",
        "Perf/learning_time",
        "Perf/iteration_time",
    ):
        assert key in payload
    assert "Perf/learner_replay_stage_ms" not in payload
    assert "Perf/learner_weight_publish_ms" not in payload
    assert collector_labels[:4] == [
        "Inference Request",
        "Learner Action Wait",
        "Env Step",
        "Replay Write",
    ]
    for key in (
        "Perf/collector_inference_request_ms",
        "Perf/collector_learner_action_wait_ms",
        "Perf/collector_env_step_ms",
        "Perf/collector_replay_write_ms",
    ):
        assert key in payload
    assert "perf/collector_cycle_ms" not in payload
    assert "timing/collector_bookkeeping_ms" not in payload
    logger.finish()


def test_offpolicy_logger_tensorboard_logs_wall_clock_without_axis_scalars():
    tb_writer = _FakeTensorBoardWriter()
    logger = OffPolicyLogger(
        algo_name="FastSAC",
        env_name="G1WalkFlat",
        log_backend="none",
    )
    logger._tb_writer = tb_writer
    logger.log_step(
        iteration=5,
        metrics={},
        train_time=0.7,
        collector_wait_time=1.0,
        replay_batch_wait_time=0.02,
        learner_replay_sample_time=0.03,
        sync_coordination_time=0.04,
        replay_ingress_h2d_submit_time=0.05,
        weight_sync_time=0.1,
        inference_h2d_time=0.01,
        inference_forward_time=0.18,
        inference_d2h_time=0.01,
        inference_time=0.2,
        iteration_time=2.15,
        extra_info={
            "throughput_steps": 16,
            "batch_size_per_rank": 16,
            "effective_batch_size": 16,
            "learner_replay_rows_per_iter": 64,
        },
    )

    scalars = {tag: value for tag, value, _ in tb_writer.scalars}
    assert scalars["Perf/learner_collector_wait_ms"] == pytest.approx(1_000.0)
    assert "timing/learner_replay_wait_ms" not in scalars
    assert scalars["Perf/learner_replay_batch_wait_ms"] == pytest.approx(20.0)
    assert scalars["Perf/replay_ingress_h2d_submit_ms"] == pytest.approx(50.0)
    assert scalars["Perf/learner_replay_sample_ms"] == pytest.approx(30.0)
    assert scalars["Perf/learner_collector_release_ms"] == pytest.approx(40.0)
    assert scalars["Perf/learner_inference_h2d_ms"] == pytest.approx(10.0)
    assert scalars["Perf/learner_inference_forward_ms"] == pytest.approx(180.0)
    assert scalars["Perf/learner_inference_d2h_ms"] == pytest.approx(10.0)
    assert scalars["Perf/learner_inference_ms"] == pytest.approx(200.0)
    assert scalars["Perf/learning_time"] == pytest.approx(0.7)
    assert "timing/learner_param_sync_ms" not in scalars
    assert "Perf/learner_replay_stage_ms" not in scalars
    assert "Perf/learner_weight_publish_ms" not in scalars
    assert "timing/learner_other_ms" not in scalars
    assert "perf/learner_pipeline_ms" not in scalars
    assert scalars["Perf/iteration_time"] == pytest.approx(2.15)
    assert scalars["Perf/total_fps"] == pytest.approx(16.0 / 2.15)
    assert "Episode/timeout_rate" not in scalars
    assert "episode/timeout_rate" not in scalars
    assert "episode/terminated_rate" not in scalars
    for key in ("axis/iteration", "axis/env_steps_total"):
        assert key not in scalars
    assert not any(key.startswith("distributed/") for key in scalars)
    logger.finish()


def test_offpolicy_logger_logs_episode_return_and_reward_terms(monkeypatch):
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
        return_mean_ep100=2.0,
        reward_components={"tracking": 3.0},
    )

    payload, step = fake_wandb.log_calls[-1]
    assert step == 128
    assert payload["Train/mean_reward"] == 2.0
    assert payload["reward/tracking"] == 3.0
    assert "reward/mean" not in payload
    assert "reward/mean_ep100" not in payload
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
    logger.log_collector(total_steps=8, buffer_size=8)
    logger.log_step(
        iteration=1,
        metrics={},
        train_time=0.75,
        collector_wait_time=1.0,
        replay_ingress_h2d_submit_time=0.02,
        weight_sync_time=0.05,
    )

    payload, _ = fake_wandb.log_calls[-1]
    assert "timing/learner_collect_ms" not in payload
    assert "Perf/total_fps" not in payload
    assert "Perf/iteration_time" not in payload
    assert "Perf/learner_weight_publish_ms" not in payload
    assert payload["Perf/learning_time"] == pytest.approx(0.75)
    assert payload["Perf/learner_collector_wait_ms"] == pytest.approx(1_000.0)
    assert payload["Perf/replay_ingress_h2d_submit_ms"] == pytest.approx(20.0)

    logger.finish()


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def add_event(self, event: Any) -> None:
        self.events.append(event)


class _FakeTbWriter:
    def __init__(self) -> None:
        self.file_writer = _RecordingEventSink()
        self.scalars: list[tuple[str, float, int]] = []
        self.closed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), step))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _collect_event_scalars(writer: _FakeTbWriter) -> list[tuple[int, dict[str, float]]]:
    out = []
    for event in writer.file_writer.events:
        out.append((event.step, {value.tag: value.simple_value for value in event.summary.value}))
    return out


def test_patch_rsl_rl_tensorboard_logging_batches_scalars_per_step() -> None:
    from unilab.training.experiment import patch_rsl_rl_tensorboard_logging

    writer = _FakeTbWriter()

    class Logger:
        logger_type = "tensorboard"

        def __init__(self) -> None:
            self.writer: Any = writer

        def log(self, it: int, start_it: int, total_it: int) -> None:
            self.writer.add_scalar("Loss/value", float(it), it)
            self.writer.add_scalar("Loss/policy", float(it) * 2, it)
            self.writer.add_scalar("Train/mean_reward/time", float(it), it + 1000)

    logger = Logger()
    runner = types.SimpleNamespace(logger=logger)
    patch_rsl_rl_tensorboard_logging(runner, log_interval=1)

    assert logger.writer is not writer
    for it in (1, 2):
        logger.log(it, 1, 2)
    logger.writer.close()

    assert writer.scalars == []
    steps = _collect_event_scalars(writer)
    assert [step for step, _ in steps] == [1, 1001, 2, 1002]
    assert steps[0][1] == {"Loss/value": 1.0, "Loss/policy": 2.0}
    assert steps[2][1] == {"Loss/value": 2.0, "Loss/policy": 4.0}
    assert writer.closed


def test_patch_rsl_rl_tensorboard_logging_interval_gates_writes_not_console() -> None:
    from unilab.training.experiment import patch_rsl_rl_tensorboard_logging

    writer = _FakeTbWriter()
    console_calls: list[int] = []

    class Logger:
        logger_type = "tensorboard"

        def __init__(self) -> None:
            self.writer: Any = writer

        def log(self, it: int, start_it: int, total_it: int) -> None:
            console_calls.append(it)
            self.writer.add_scalar("Loss/value", float(it), it)

    logger = Logger()
    runner = types.SimpleNamespace(logger=logger)
    patch_rsl_rl_tensorboard_logging(runner, log_interval=2)

    for it in range(1, 6):
        logger.log(it, 1, 5)
    logger.writer.close()

    assert console_calls == [1, 2, 3, 4, 5]
    steps = _collect_event_scalars(writer)
    assert [step for step, _ in steps] == [2, 4, 5]


def test_patch_rsl_rl_tensorboard_logging_skips_non_tensorboard_and_missing_writer() -> None:
    from unilab.training.experiment import patch_rsl_rl_tensorboard_logging

    writer = _FakeTbWriter()

    class WandbLogger:
        logger_type = "wandb"

        def __init__(self) -> None:
            self.writer: Any = writer

    wandb_logger = WandbLogger()
    patch_rsl_rl_tensorboard_logging(types.SimpleNamespace(logger=wandb_logger))
    assert wandb_logger.writer is writer

    class EmptyLogger:
        logger_type = "tensorboard"
        writer = None

    empty_logger = EmptyLogger()
    patch_rsl_rl_tensorboard_logging(types.SimpleNamespace(logger=empty_logger))
    assert empty_logger.writer is None

    patch_rsl_rl_tensorboard_logging(types.SimpleNamespace(logger=None))
