from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

import unilab.logging.common as common_logger_module
import unilab.logging.offpolicy as offpolicy_logger_module
from unilab.logging.offpolicy import OffPolicyLogger


def test_offpolicy_logger_only_forces_refresh_for_errors(monkeypatch) -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="none",
    )
    refresh_calls: list[bool] = []

    def fake_refresh(*, force: bool = False) -> None:
        refresh_calls.append(force)

    monkeypatch.setattr(logger, "_refresh", fake_refresh)

    logger.log_buffer_fill(32, 64)
    logger.log_status("Replay storage: device-authoritative bounded ingress")

    assert refresh_calls == []
    assert logger._buffer_size == 32
    assert logger._buffer_target == 64

    logger.log_status("[red]ERROR: Collector died[/]")
    assert refresh_calls == [True]

    refresh_calls.clear()
    logger.log_step(
        iteration=1,
        metrics={"loss/q": 0.5},
        reward=1.0,
        extra_info={"throughput_steps": 8},
    )
    logger.log_status("Training")
    logger.log_buffer_fill(64, 64)

    assert refresh_calls == []


def test_offpolicy_logger_stop_live_lets_rich_do_final_refresh() -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="none",
    )

    class _FakeLive:
        def __init__(self) -> None:
            self.update_calls: list[bool] = []
            self.stop_calls = 0

        def update(self, renderable: Any, *, refresh: bool) -> None:
            del renderable
            self.update_calls.append(refresh)

        def stop(self) -> None:
            self.stop_calls += 1

    live = _FakeLive()
    logger._live = live  # type: ignore[assignment]
    logger._last_live_refresh_time = 123.0

    logger._stop_live()

    assert live.update_calls == [False]
    assert live.stop_calls == 1
    assert logger._live is None
    assert logger._last_live_refresh_time is None


def test_offpolicy_logger_displays_env_step_breakdown_as_indented_children() -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="none",
    )
    logger.update_collector_timing(
        {
            "inference_request_ms": 0.1,
            "learner_action_wait_ms": 0.2,
            "env_step_ms": 14.0,
            "env_step_backend_ms": 12.5,
            "env_step_update_state_ms": 1.0,
            "env_step_reset_done_ms": 0.5,
            "replay_write_ms": 0.3,
        }
    )

    table = logger._build_timing_table()
    collector_cells = list(table.columns[2].cells)[:7]
    collector_value_cells = list(table.columns[3].cells)[:7]

    assert collector_cells == [
        "Inference Request",
        "Learner Action Wait",
        "Env Step",
        "[dim]  Backend Step[/]",
        "[dim]  Update State[/]",
        "[dim]  Reset Done[/]",
        "Replay Write",
    ]
    assert collector_value_cells == [
        "    0.1ms    1%",
        "    0.2ms    1%",
        "   14.0ms   96%",
        "[dim cyan]   12.5ms  86%─┤[/]",
        "[dim cyan]    1.0ms   7%─┤[/]",
        "[dim cyan]    0.5ms   3%─┘[/]",
        "    0.3ms    2%",
    ]

    console = Console(width=100, record=True, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    connector_columns = [
        line.index(connector)
        for line in capture.get().splitlines()
        for connector in ("┤", "┘")
        if connector in line
    ]
    assert len(connector_columns) == 3
    assert len(set(connector_columns)) == 1


def test_offpolicy_logger_discards_retired_collector_timing_names() -> None:
    logger = OffPolicyLogger(log_backend="none")

    logger.update_collector_timing(
        {
            "inference_wait_ms": 2.0,
            "learner_action_wait_ms": 4.0,
            "sync_idle_ms": 3.0,
        }
    )

    assert logger._collector_timing == {"learner_action_wait_ms": 4.0}


def test_offpolicy_logger_waits_for_complete_collector_cycle_before_percentages() -> None:
    logger = OffPolicyLogger(log_backend="none")
    logger.update_collector_timing({"replay_write_ms": 2.5})

    assert logger._get_collector_cycle_ms() is None
    assert "%" not in list(logger._build_timing_table().columns[3].cells)[0]


def test_offpolicy_logger_rejects_unknown_timing_profile() -> None:
    with pytest.raises(ValueError, match="timing_profile"):
        OffPolicyLogger(log_backend="none", timing_profile="unknown")


def test_offpolicy_logger_shows_complete_additive_learner_timeline() -> None:
    logger = OffPolicyLogger(
        algo_name="FastSAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="none",
    )
    logger.log_step(
        iteration=1,
        collector_wait_time=0.10,
        inference_time=0.20,
        sync_coordination_time=0.03,
        replay_batch_wait_time=0.04,
        learner_replay_stage_time=0.07,
        learner_replay_sample_time=0.05,
        train_time=0.50,
        weight_sync_time=0.06,
        replay_ingress_h2d_submit_time=0.40,
        iteration_time=1.0,
    )

    table = logger._build_timing_table()

    assert table.columns[0].header == "Learner (Iter Wall)"
    assert list(table.columns[0].cells)[:8] == [
        "Collector Wait",
        "Inference",
        "Collector Release",
        "Replay Batch Wait",
        "Replay Sample",
        "Train",
        "Other",
        "Iter Wall",
    ]
    assert list(table.columns[1].cells)[:8] == [
        "[red]  100.0ms   10%[/]",
        "  200.0ms   20%",
        "   30.0ms    3%",
        "   40.0ms    4%",
        "   50.0ms    5%",
        "[green]  500.0ms   50%[/]",
        "   80.0ms    8%",
        " 1000.0ms  100%",
    ]
    assert logger._get_learner_accounted_time() == pytest.approx(0.92)
    assert logger._get_learner_other_time() == pytest.approx(0.08)


def test_offpolicy_logger_does_not_fold_parallel_h2d_into_accounted_time() -> None:
    logger = OffPolicyLogger(
        algo_name="FlashSAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="none",
    )
    logger.log_step(
        iteration=1,
        collector_wait_time=0.05,
        inference_time=0.10,
        replay_batch_wait_time=0.05,
        train_time=0.70,
        replay_ingress_h2d_submit_time=0.50,
        iteration_time=1.0,
    )

    assert logger._get_learner_accounted_time() == pytest.approx(0.90)
    assert logger._get_iter_pct(logger._get_learner_accounted_time()) == pytest.approx(90.0)
    assert "Replay H2D Submit" not in list(logger._build_timing_table().columns[0].cells)


def test_offpolicy_logger_appo_profile_only_shows_applicable_learner_phases() -> None:
    logger = OffPolicyLogger(log_backend="none", timing_profile="appo")
    logger.log_step(
        iteration=1,
        collector_wait_time=0.10,
        inference_time=0.20,
        sync_coordination_time=0.03,
        replay_batch_wait_time=0.04,
        learner_replay_stage_time=0.05,
        learner_replay_sample_time=0.06,
        train_time=0.50,
        weight_sync_time=0.07,
        replay_ingress_h2d_submit_time=0.40,
        iteration_time=1.0,
    )

    table = logger._build_timing_table()

    assert list(table.columns[0].cells)[:7] == [
        "Collector Wait",
        "Replay Stage",
        "Replay Sample",
        "Train",
        "Weight Publish",
        "Other",
        "Iter Wall",
    ]
    assert logger._get_learner_accounted_time() == pytest.approx(0.78)
    assert logger._get_learner_other_time() == pytest.approx(0.22)


def test_offpolicy_logger_rollout_collector_uses_milliseconds_without_cycle_total() -> None:
    logger = OffPolicyLogger(log_backend="none", timing_profile="appo")
    logger._unicode_console = False
    logger.update_collector_timing(
        {
            "mlp_infer_ms": 1.5,
            "env_step_ms": 2.5,
            "env_step_backend_ms": 2.0,
            "rollout_ms": 40.0,
        }
    )

    table = logger._build_timing_table()
    collector_values = list(table.columns[3].cells)[:4]

    assert table.columns[2].header == "Collector (own clock)"
    assert collector_values == [
        "1.5ms",
        "2.5ms",
        "[dim cyan]    2.0ms -'[/]",
        "40.0ms",
    ]
    assert all("%" not in value for value in collector_values)
    assert logger._get_collector_cycle_ms() is None


def test_offpolicy_reward_component_names_stay_on_one_line_at_narrow_width() -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
        max_iterations=2,
        num_envs=8,
        env_name="Dummy",
        log_backend="no_print",
    )
    logger._reward_history.extend([1.0, 2.0])
    logger._latest_reward_components = {
        "reward/penalty_action_rate": -0.1,
        "reward/penalty_ang_vel_xy": -0.2,
        "reward/penalty_orientation": -0.3,
    }
    console = Console(width=47, record=True, force_terminal=False)

    console.print(logger._build_reward_table())
    output_lines = console.export_text().splitlines()

    for component in ("penalty action rate", "penalty ang vel xy", "penalty orientation"):
        matching_lines = [line for line in output_lines if component in line]
        assert len(matching_lines) == 1


def test_offpolicy_logger_training_timer_excludes_warmup_from_elapsed_and_eta(
    monkeypatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(common_logger_module.time, "time", lambda: now)

    logger = OffPolicyLogger(
        algo_name="FastSAC",
        max_iterations=2,
        num_envs=8,
        env_name="G1WalkFlat",
        log_backend="no_print",
    )
    logger._unicode_console = False
    logger.start()

    now = 130.0
    assert logger.start_training_timer() == 130.0

    now = 132.0
    logger.log_step(iteration=1, extra_info={"throughput_steps": 8})

    header = logger._build_compact_header(include_status=False, include_identity=False)
    assert "time 2s" in header.plain
    assert "ETA 2s" in header.plain
    assert "30s" not in header.plain


def test_offpolicy_logger_moves_identity_and_iteration_to_panel_title() -> None:
    logger = OffPolicyLogger(
        algo_name="FastSAC",
        max_iterations=5000,
        num_envs=4096,
        env_name="G1WalkFlat",
        log_backend="no_print",
    )
    logger._unicode_console = True
    logger.start_training_timer()
    logger.log_step(iteration=5000, extra_info={"throughput_steps": 4096})

    display = logger._build_display()
    header = logger._build_compact_header(
        include_status=False,
        include_identity=False,
        include_iteration=False,
    )

    assert isinstance(display.title, type(header))
    assert display.title.plain == (
        " 🚀 UniLab Off-Policy Training | FastSAC | G1WalkFlat | GPUs 1 | iter 5000/5000 "
    )
    assert "FastSAC" not in header.plain
    assert "G1WalkFlat" not in header.plain
    assert "iter 5000/5000" not in header.plain
    assert "Training" not in header.plain


def test_offpolicy_terminal_averages_aggregated_samples_over_two_seconds(
    monkeypatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(offpolicy_logger_module.time, "monotonic", lambda: now)

    class _Writer:
        def __init__(self) -> None:
            self.scalars: list[tuple[str, float, int]] = []

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars.append((tag, value, step))

    logger = OffPolicyLogger(log_backend="none", num_gpus=2)
    writer = _Writer()
    logger._tb_writer = writer

    logger.update_timeout_rate(0.2)
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
        metrics={"critic_loss": 2.0},
        train_time=0.2,
        iteration_time=0.5,
        extra_info={
            "steps_per_sec": 400.0,
            "learner_samples_per_sec": 1_000.0,
        },
    )

    now = 101.0
    logger.update_timeout_rate(0.4)
    logger.update_collector_timing(
        {
            "inference_request_ms": 3.0,
            "learner_action_wait_ms": 4.0,
            "env_step_ms": 5.0,
            "replay_write_ms": 6.0,
        }
    )
    logger.log_step(
        iteration=2,
        metrics={"critic_loss": 4.0},
        train_time=0.4,
        iteration_time=1.0,
        extra_info={
            "steps_per_sec": 800.0,
            "learner_samples_per_sec": 3_000.0,
        },
    )

    snapshot = logger._terminal_snapshot
    assert snapshot is not None
    assert snapshot.sample_count == 2
    assert snapshot.metrics["critic_loss"] == pytest.approx(3.0)
    assert snapshot.scalars["_train_time"] == pytest.approx(0.3)
    assert snapshot.scalars["timeout_rate"] == pytest.approx(0.3)
    assert snapshot.scalars["steps_per_sec"] == pytest.approx(600.0)
    assert snapshot.scalars["samples_per_sec"] == pytest.approx(2_000.0)
    assert snapshot.collector_timing["env_step_ms"] == pytest.approx(4.0)
    collector_values = list(logger._build_timing_table().columns[3].cells)
    assert "    4.0ms   29%" in collector_values
    assert "Avg 2s (n=2)" in logger._build_compact_header(include_status=False).plain
    assert "GPUs 2" in logger._build_display().title.plain

    latest_backend_values = {tag: value for tag, value, _ in writer.scalars}
    assert latest_backend_values["train/critic_loss"] == pytest.approx(4.0)
    assert latest_backend_values["perf/steps_per_sec"] == pytest.approx(800.0)

    now = 103.1
    logger.log_step(
        iteration=3,
        metrics={"critic_loss": 10.0},
        train_time=1.0,
        iteration_time=2.0,
        extra_info={
            "steps_per_sec": 1_200.0,
            "learner_samples_per_sec": 5_000.0,
        },
    )
    snapshot = logger._terminal_snapshot
    assert snapshot is not None
    assert snapshot.sample_count == 1
    assert snapshot.metrics["critic_loss"] == pytest.approx(10.0)
