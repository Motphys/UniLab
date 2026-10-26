from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from rich.console import Console

import unilab.rl.logging.base as base_logger_module
import unilab.rl.logging.offpolicy as offpolicy_logger_module
import unilab.rl.logging.onpolicy as onpolicy_logger_module
from unilab.rl.logging.base import BaseTrainingLogger
from unilab.rl.logging.offpolicy import OffPolicyLogger
from unilab.rl.logging.onpolicy import OnPolicyLogger


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
        metrics={"Loss/critic": 0.5},
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


def test_offpolicy_logger_close_restores_cursor_and_backends_when_rich_stop_fails(
    monkeypatch,
) -> None:
    logger = OffPolicyLogger(log_backend="none")

    class _FailingLive:
        def update(self, renderable: Any, *, refresh: bool) -> None:
            del renderable, refresh

        def stop(self) -> None:
            raise RuntimeError("final render failed")

    cursor_calls: list[bool] = []
    backend_close_calls: list[bool] = []
    monkeypatch.setattr(logger._console, "show_cursor", cursor_calls.append)
    monkeypatch.setattr(logger, "_close_backends", lambda: backend_close_calls.append(True))
    logger._live = _FailingLive()  # type: ignore[assignment]
    logger._last_live_refresh_time = 123.0

    with pytest.raises(RuntimeError, match="final render failed"):
        logger.close()

    assert cursor_calls == [True]
    assert backend_close_calls == [True]
    assert logger._live is None
    assert logger._last_live_refresh_time is None
    assert logger._closed is True


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
            "transition_extract_ms": 0.1,
            "replay_write_ms": 0.3,
            "metrics_publish_ms": 0.1,
        }
    )

    table = logger._build_timing_table()
    logger._console = Console(width=140, record=True, force_terminal=False)
    table = logger._build_timing_table()
    collector_cells = [cell for cell in table.columns[2].cells if cell][:8]
    collector_value_cells = [cell for cell in table.columns[3].cells if cell][:8]

    assert collector_cells == [
        "Learner Action Wait",
        "Env Step",
        "[dim]  Backend Step[/]",
        "[dim]  Update State[/]",
        "[dim]  Reset Done[/]",
        "Replay Write",
    ]
    assert collector_value_cells == [
        "    0.2ms    1%",
        "   14.0ms   95%",
        "[dim cyan]   12.5ms  84%─┤[/]",
        "[dim cyan]    1.0ms   7%─┤[/]",
        "[dim cyan]    0.5ms   3%─┘[/]",
        "    0.3ms    2%",
    ]

    console = Console(width=140, record=True, force_terminal=False)
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


def test_offpolicy_logger_hides_small_collector_diagnostics_until_material() -> None:
    logger = OffPolicyLogger(log_backend="none")
    logger._unicode_console = False
    logger._console = Console(width=140, record=True, force_terminal=False)
    logger.update_collector_timing(
        {
            "inference_request_ms": 0.1,
            "learner_action_wait_ms": 1.0,
            "env_step_ms": 90.0,
            "env_step_action_validate_ms": 0.05,
            "env_step_apply_action_ms": 0.20,
            "env_step_backend_ms": 80.0,
            "transition_extract_ms": 0.10,
            "replay_write_ms": 1.0,
            "metrics_publish_ms": 0.01,
        }
    )

    quiet_cells = [cell for cell in logger._build_timing_table().columns[2].cells if cell]
    assert quiet_cells == [
        "Learner Action Wait",
        "Env Step",
        "[dim]  Backend Step[/]",
        "Replay Write",
    ]
    assert "Inference Request" not in quiet_cells
    assert "Action Validate" not in quiet_cells
    assert "Transition Extract" not in quiet_cells
    assert "Metrics Publish" not in quiet_cells

    logger.update_collector_timing(
        {
            "inference_request_ms": 0.1,
            "learner_action_wait_ms": 1.0,
            "env_step_ms": 50.0,
            "env_step_action_validate_ms": 0.05,
            "env_step_apply_action_ms": 20.0,
            "env_step_backend_ms": 10.0,
            "transition_extract_ms": 5.0,
            "replay_write_ms": 1.0,
            "metrics_publish_ms": 0.01,
        }
    )
    active_cells = [cell for cell in logger._build_timing_table().columns[2].cells if cell]
    assert "[dim]  Apply Action[/]" in active_cells
    assert "Transition Extract" in active_cells
    assert "Metrics Publish" not in active_cells


def test_offpolicy_metrics_table_separates_replay_ingress_from_losses() -> None:
    logger = OffPolicyLogger(log_backend="none")
    logger._console = Console(width=140, record=True, force_terminal=False)
    logger._latest_metrics = {
        "Loss/critic": 4.2,
        "Policy/temperature": 0.01,
        "Train/reward_scale_std": 0.3,
        "Train/replay_ingress_depth": 2.0,
        "Train/replay_ingress_backpressure_wait_ms": 12.0,
    }

    table = logger._build_metrics_table()
    labels = [cell for cell in table.columns[0].cells if cell]
    console = Console(width=140, record=True, force_terminal=False)
    with console.capture():
        console.print(table)
    output = console.export_text()

    assert labels == [
        "Critic",
        "Temperature",
        "  Reward Scale Std",
        "  Ingress Backpressure",
        "  Ingress Depth",
    ]
    assert labels.index("Critic") < labels.index("Temperature")
    assert labels.index("Temperature") < labels.index("  Ingress Depth")
    assert "Backpressure Wait Ms" not in output
    assert all("Wait Ms" not in line for line in output.splitlines())


def test_offpolicy_terminal_render_is_width_aware() -> None:
    metrics = {
        "Loss/critic": 4.2,
        "Policy/temperature": 0.01,
        "Train/reward_scale_std": 0.3,
        "Train/replay_ingress_depth": 2.0,
        "Train/replay_ingress_backpressure_wait_ms": 12.0,
    }
    collector_timing = {
        "inference_request_ms": 0.1,
        "learner_action_wait_ms": 1.0,
        "env_step_ms": 30.0,
        "env_step_apply_action_ms": 9.0,
        "env_step_backend_ms": 14.0,
        "env_step_update_state_ms": 2.0,
        "env_step_reset_done_ms": 4.0,
        "transition_extract_ms": 0.1,
        "replay_write_ms": 0.2,
        "metrics_publish_ms": 0.005,
    }

    expectations = {
        80: {
            "layout": "stacked",
            "headers": ("Metrics", "Rewards", "Timing"),
            "intact": (
                "Collector Wait",
                "Learner Action Wait",
                "Apply Action",
                "Backend Step",
                "Ingress Backpressure",
            ),
        },
        100: {
            "layout": "two-column",
            "headers": ("Losses & Policy", "Learner", "Collector"),
            "intact": ("Collector Wait", "Learner Action Wait", "Apply Action"),
        },
        120: {
            "layout": "three-column",
            "headers": ("Learner (Iter Wall)", "Collector (own clock)", "System"),
            "intact": ("Collector Wait", "Learner Action Wait", "Apply Action"),
        },
        140: {
            "layout": "three-column",
            "headers": ("Learner (Iter Wall)", "Collector (own clock)", "System"),
            "intact": ("Collector Wait", "Learner Action Wait", "Apply Action"),
        },
    }

    for width, expected in expectations.items():
        logger = OffPolicyLogger(log_backend="none")
        logger._console = Console(width=width, record=True, force_terminal=False)
        logger._unicode_console = False
        logger._latest_metrics = dict(metrics)
        logger.update_collector_timing(dict(collector_timing))
        logger._reward_history.extend([1.0, 2.0])
        logger._latest_reward_components = {"reward/alive": 1.0}

        console = Console(width=width, record=True, force_terminal=False)
        console.print(logger._build_display())
        output = console.export_text()
        lines = output.splitlines()

        for header in expected["headers"]:
            assert any(header in line for line in lines), (width, header, output)
        for label in expected["intact"]:
            assert any(label in line for line in lines), (width, label, output)
        if width >= 120:
            assert "…" not in output
        assert "Wait Ms" not in output


def test_offpolicy_logger_system_column_shows_cuda_mps_status() -> None:
    logger = OffPolicyLogger(log_backend="none")
    logger._console = Console(width=140, record=True, force_terminal=False)

    def system_cells() -> tuple[list[str], list[str]]:
        table = logger._build_timing_table()
        labels = [cell for cell in table.columns[4].cells if cell]
        values = [cell for cell in table.columns[5].cells if cell]
        return labels, values

    labels, _ = system_cells()
    assert "CUDA MPS" not in labels

    logger.update_runtime_manifest(
        {"cuda_process_sharing": {"effective": "mps", "ambient_server": False}}
    )
    labels, values = system_cells()
    assert labels[-1] == "CUDA MPS"
    assert values[-1] == "[green]on[/]"

    logger.update_runtime_manifest(
        {"cuda_process_sharing": {"effective": None, "ambient_server": True}}
    )
    labels, values = system_cells()
    assert labels[-1] == "CUDA MPS"
    assert values[-1] == "[yellow]ambient[/]"

    logger.update_runtime_manifest(
        {"cuda_process_sharing": {"effective": None, "ambient_server": False}}
    )
    labels, values = system_cells()
    assert labels[-1] == "CUDA MPS"
    assert values[-1] == "[dim]off[/]"


def test_offpolicy_logger_rejects_retired_collector_timing_names() -> None:
    logger = OffPolicyLogger(log_backend="none")

    with pytest.raises(ValueError, match="unregistered collector timing keys: sync_idle_ms"):
        logger.update_collector_timing({"learner_action_wait_ms": 4.0, "sync_idle_ms": 3.0})

    assert logger._collector_timing == {}


def test_offpolicy_logger_waits_for_complete_collector_cycle_before_percentages() -> None:
    logger = OffPolicyLogger(log_backend="none")
    logger._console = Console(width=140, record=True, force_terminal=False)
    logger.update_collector_timing({"replay_write_ms": 2.5})

    assert logger._get_collector_cycle_ms() is None
    assert "%" not in list(logger._build_timing_table().columns[3].cells)[0]


def test_offpolicy_logger_rejects_unknown_timing_profile() -> None:
    with pytest.raises(ValueError, match="timing_profile"):
        OffPolicyLogger(log_backend="none", timing_profile="unknown")


def test_offpolicy_logger_shows_complete_additive_learner_timeline() -> None:
    logger = OffPolicyLogger(
        algo_name="SAC",
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

    logger._console = Console(width=140, record=True, force_terminal=False)
    table = logger._build_timing_table()

    assert table.columns[0].header == "Learner (Iter Wall)"
    assert list(table.columns[0].cells)[:8] == [
        "Collector Wait",
        "Inference",
        "Collector Release",
        "Replay Batch Wait",
        "Replay Sample",
        "Learning",
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
        "Learning",
        "Weight Publish",
        "Other",
        "Iter Wall",
    ]
    assert logger._get_learner_accounted_time() == pytest.approx(0.78)
    assert logger._get_learner_other_time() == pytest.approx(0.22)


def test_offpolicy_logger_rollout_collector_uses_milliseconds_without_cycle_total() -> None:
    logger = OffPolicyLogger(log_backend="none", timing_profile="appo")
    logger._unicode_console = False
    logger._console = Console(width=140, record=True, force_terminal=False)
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
    monkeypatch.setattr(base_logger_module.time, "time", lambda: now)

    logger = OffPolicyLogger(
        algo_name="SAC",
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
        algo_name="SAC",
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
        " 🚀 UniLab Off-Policy Training | SAC | G1WalkFlat | GPUs 1 | iter 5000/5000 "
    )
    assert "SAC" not in header.plain
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
        metrics={"Loss/critic": 2.0},
        train_time=0.2,
        iteration_time=0.5,
        extra_info={
            "env_steps_per_sec": 400.0,
            "learner_replay_rows_per_sec": 1_000.0,
        },
    )

    now = 101.0
    logger.update_timeout_rate(0.4)
    logger.update_collector_timing(
        {
            "inference_request_ms": 3.0,
            "learner_action_wait_ms": 4.0,
            "env_step_ms": 5.0,
            "transition_extract_ms": 0.5,
            "replay_write_ms": 6.0,
            "metrics_publish_ms": 0.5,
        }
    )
    logger.log_step(
        iteration=2,
        metrics={"Loss/critic": 4.0},
        train_time=0.4,
        iteration_time=1.0,
        extra_info={
            "env_steps_per_sec": 800.0,
            "learner_replay_rows_per_sec": 3_000.0,
        },
    )

    snapshot = logger._terminal_snapshot
    assert snapshot is not None
    assert snapshot.sample_count == 2
    assert snapshot.metrics["Loss/critic"] == pytest.approx(3.0)
    assert snapshot.scalars["_train_time"] == pytest.approx(0.3)
    assert snapshot.scalars["timeout_rate"] == pytest.approx(0.3)
    assert snapshot.scalars["env_steps_per_sec"] == pytest.approx(600.0)
    assert snapshot.scalars["learner_replay_rows_per_sec"] == pytest.approx(2_000.0)
    assert snapshot.collector_timing["env_step_ms"] == pytest.approx(4.0)
    logger._console = Console(width=140, record=True, force_terminal=False)
    collector_values = list(logger._build_timing_table().columns[3].cells)
    assert "    4.0ms   27%" in collector_values
    assert "Avg 2s (n=2)" in logger._build_compact_header(include_status=False).plain
    assert "GPUs 2" in logger._build_display().title.plain

    latest_backend_values = {tag: value for tag, value, _ in writer.scalars}
    assert latest_backend_values["Loss/critic"] == pytest.approx(4.0)
    assert latest_backend_values["Perf/total_fps"] == pytest.approx(800.0)

    now = 103.1
    logger.log_step(
        iteration=3,
        metrics={"Loss/critic": 10.0},
        train_time=1.0,
        iteration_time=2.0,
        extra_info={
            "env_steps_per_sec": 1_200.0,
            "learner_replay_rows_per_sec": 5_000.0,
        },
    )
    snapshot = logger._terminal_snapshot
    assert snapshot is not None
    assert snapshot.sample_count == 1
    assert snapshot.metrics["Loss/critic"] == pytest.approx(10.0)


class _BatchedWriter:
    """Minimal SummaryWriter stand-in exposing the batched event path."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.scalars: list[tuple[str, float, int]] = []

        class _FileWriter:
            def __init__(self, sink: list[Any]) -> None:
                self._sink = sink

            def add_event(self, event: Any) -> None:
                self._sink.append(event)

        self.file_writer = _FileWriter(self.events)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))


def _batched_scalar_map(writer: _BatchedWriter) -> dict[str, float]:
    assert writer.events, "expected at least one batched event"
    result: dict[str, float] = {}
    for event in writer.events:
        tags = [value.tag for value in event.summary.value]
        assert len(tags) == len(set(tags)), f"duplicate backend tags: {tags}"
        for value in event.summary.value:
            result[value.tag] = value.simple_value
    return result


def test_offpolicy_backend_log_step_writes_one_batched_event() -> None:
    logger = OffPolicyLogger(log_backend="none", num_gpus=1)
    writer = _BatchedWriter()
    logger._tb_writer = writer

    logger.update_curriculum_metrics(
        {"Curriculum/voc/stage": 2.0, "Curriculum/voc/target_voc_scale": 0.5}
    )
    logger.log_step(
        iteration=1,
        metrics={"Loss/critic": 2.0},
        return_mean_ep100=1.25,
        reward_components={"alive": 1.0},
        train_time=0.2,
        iteration_time=0.5,
        extra_info={"env_steps_per_sec": 400.0, "throughput_steps": 4096},
    )

    assert len(writer.events) == 1
    assert writer.scalars == []
    event = writer.events[0]
    assert event.step == 1
    scalars = _batched_scalar_map(writer)
    assert scalars["Loss/critic"] == pytest.approx(2.0)
    assert scalars["Train/mean_reward"] == pytest.approx(1.25)
    assert scalars["reward/alive"] == pytest.approx(1.0)
    assert scalars["Curriculum/voc/stage"] == pytest.approx(2.0)
    assert scalars["Curriculum/voc/target_voc_scale"] == pytest.approx(0.5)
    assert scalars["Perf/total_fps"] == pytest.approx(400.0)
    assert "Perf/iteration_time" in scalars
    assert "Perf/learning_time" in scalars
    assert "perf/learner_train_pct" not in scalars
    assert "perf/collector_cycle_ms" not in scalars


def test_offpolicy_writer_does_not_emit_retired_summary_or_throughput_tags() -> None:
    """Guard the current event writer against retired direct-rename tags."""
    logger = OffPolicyLogger(log_backend="none", num_gpus=1)
    writer = _BatchedWriter()
    logger._tb_writer = writer

    logger.log_step(
        iteration=1,
        metrics={"Loss/critic": 2.0},
        return_mean_ep100=1.25,
        train_time=0.2,
        iteration_time=0.5,
        extra_info={"env_steps_per_sec": 400.0},
    )

    tags = set(_batched_scalar_map(writer))
    assert "Train/mean_reward" in tags
    assert "Perf/total_fps" in tags
    assert {"reward/mean", "reward/mean_ep100", "perf/steps_per_sec"}.isdisjoint(tags)


def test_offpolicy_source_episode_return_owner_fails_closed() -> None:
    logger = OffPolicyLogger(log_backend="none", max_iterations=10, log_interval=5)

    with pytest.raises(ValueError, match="logger-owned metrics"):
        logger.log_step(
            iteration=1,
            metrics={"Train/mean_reward": 1.0},
            return_mean_ep100=2.0,
        )


def test_offpolicy_runner_iteration_collision_fails_closed() -> None:
    logger = OffPolicyLogger(log_backend="none")
    logger.log_collector(total_steps=16, buffer_size=8)

    with pytest.raises(ValueError, match="OffPolicyLogger owns Train/iteration"):
        logger.log_step(iteration=1, metrics={"Train/iteration": 1.0})


def test_offpolicy_fallback_axis_iteration_collision_fails_closed() -> None:
    logger = OffPolicyLogger(log_backend="none")

    with pytest.raises(ValueError, match="OffPolicyLogger owns Train/iteration"):
        logger.log_step(iteration=1, metrics={"Train/iteration": 1.0})


def test_offpolicy_backend_log_step_falls_back_to_per_scalar_writes() -> None:
    class _PlainWriter:
        def __init__(self) -> None:
            self.scalars: list[tuple[str, float, int]] = []

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars.append((tag, value, step))

    logger = OffPolicyLogger(log_backend="none", num_gpus=1)
    writer = _PlainWriter()
    logger._tb_writer = writer

    logger.log_step(iteration=1, metrics={"Loss/critic": 2.0}, train_time=0.2)

    tags = [tag for tag, _, _ in writer.scalars]
    assert "Loss/critic" in tags
    assert "Perf/iteration_time" not in tags


def test_offpolicy_log_interval_gates_backend_but_not_terminal_state() -> None:
    logger = OffPolicyLogger(log_backend="none", num_gpus=1, max_iterations=10, log_interval=5)
    writer = _BatchedWriter()
    logger._tb_writer = writer

    for iteration in range(1, 11):
        logger.log_step(
            iteration=iteration,
            metrics={"Loss/critic": float(iteration)},
            train_time=0.1,
        )

    logged_steps = [event.step for event in writer.events]
    assert logged_steps == [5, 10]
    # Terminal/render state still advances every iteration.
    assert logger._iteration == 10
    assert logger._latest_metrics["Loss/critic"] == pytest.approx(10.0)


def test_offpolicy_final_iteration_logs_when_interval_is_not_divisible() -> None:
    logger = OffPolicyLogger(log_backend="none", max_iterations=9, log_interval=4)
    writer = _BatchedWriter()
    logger._tb_writer = writer

    for iteration in range(1, 10):
        logger.log_step(iteration=iteration, metrics={"Loss/critic": float(iteration)})

    assert [event.step for event in writer.events] == [4, 8, 9]


def test_onpolicy_final_iteration_logs_when_interval_is_not_divisible() -> None:
    logger = OnPolicyLogger(
        log_backend="none", num_envs=1, num_steps=1, max_iterations=9, log_interval=4
    )
    writer = _BatchedWriter()
    logger._tb_writer = writer

    for iteration in range(10):
        logger.log_step(iteration=iteration, metrics={"Loss/surrogate": float(iteration)})

    assert [event.step for event in writer.events] == [0, 4, 8, 9]


def test_offpolicy_source_contract_fails_even_when_backend_write_is_skipped() -> None:
    logger = OffPolicyLogger(log_backend="none", max_iterations=10, log_interval=5)

    with pytest.raises(ValueError, match="unregistered canonical training metric"):
        logger.log_step(iteration=1, metrics={"train/actor_loss": 1.0})
    with pytest.raises(ValueError, match="reserved reward component name"):
        logger.log_step(iteration=1, reward_components={"mean": 1.0})


@pytest.mark.parametrize(
    ("logger_factory", "tag"),
    [
        (lambda: OffPolicyLogger(log_backend="none"), "Train/mean_reward"),
        (lambda: OffPolicyLogger(log_backend="none"), "Train/mean_episode_length"),
        (lambda: OffPolicyLogger(log_backend="none"), "Episode/timeout_rate"),
        (lambda: OffPolicyLogger(log_backend="none"), "Perf/collection_time"),
        (
            lambda: OnPolicyLogger(log_backend="none", num_envs=1, num_steps=1),
            "Train/mean_episode_length",
        ),
    ],
)
def test_source_metrics_cannot_bypass_logger_owned_telemetry(
    logger_factory: Callable[[], BaseTrainingLogger], tag: str
) -> None:
    logger = logger_factory()

    with pytest.raises(ValueError, match="logger-owned metrics"):
        logger.log_step(iteration=1, metrics={tag: 1.0})


def test_onpolicy_source_contract_fails_even_when_backend_write_is_skipped() -> None:
    logger = OnPolicyLogger(log_backend="none", max_iterations=10, log_interval=5)

    with pytest.raises(ValueError, match="unregistered canonical training metric"):
        logger.log_step(iteration=1, metrics={"train/actor_loss": 1.0})
    with pytest.raises(ValueError, match="reserved reward component name"):
        logger.log_step(iteration=1, reward_components={"mean": 1.0})


def test_offpolicy_backend_step_axis_prefers_collector_env_steps() -> None:
    logger = OffPolicyLogger(log_backend="none")
    writer = _BatchedWriter()
    logger._tb_writer = writer

    logger.log_collector(total_steps=128, buffer_size=64)
    logger.log_collector(total_steps=256, buffer_size=64)
    logger.log_step(iteration=3, metrics={"Loss/critic": 1.0}, train_time=0.1)

    assert writer.events[0].step == 256
    assert _batched_scalar_map(writer)["Train/iteration"] == pytest.approx(3.0)


def test_offpolicy_collector_counters_do_not_update_reward_history() -> None:
    logger = OffPolicyLogger(log_backend="none")

    logger.log_collector(total_steps=16, buffer_size=8)

    assert not logger._reward_history


def test_offpolicy_log_step_return_feeds_terminal_reward_table() -> None:
    logger = OffPolicyLogger(log_backend="none")

    logger.log_step(iteration=1, metrics={"Loss/critic": 1.0}, return_mean_ep100=2.5)

    assert list(logger._reward_history) == [2.5]
    reward_cells = list(logger._build_reward_table().columns[0].cells)
    assert "Waiting for data..." not in reward_cells
    assert any("Reward" in cell for cell in reward_cells)


def test_offpolicy_reward_table_waits_for_first_data_without_source_marker() -> None:
    logger = OffPolicyLogger(log_backend="none")

    table = logger._build_reward_table()

    assert table.columns[0].header == "Rewards"
    reward_cells = list(table.columns[0].cells)
    assert any("Waiting for data..." in cell for cell in reward_cells)


def test_offpolicy_reward_table_keeps_last_data_after_window_expires(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(offpolicy_logger_module.time, "monotonic", lambda: now)
    logger = OffPolicyLogger(log_backend="none")

    logger.log_step(iteration=1, return_mean_ep100=2.5, reward_components={"alive": 0.5})
    assert "(iter 1)" in logger._build_reward_table().columns[0].header

    now = 103.1
    logger.log_step(iteration=2, metrics={"Loss/critic": 1.0})

    snapshot = logger._terminal_snapshot
    assert snapshot is not None
    assert not snapshot.reward_components
    table = logger._build_reward_table()
    reward_cells = list(table.columns[0].cells)
    assert "Waiting for data..." not in reward_cells
    assert any("Reward" in cell for cell in reward_cells)
    assert any("alive" in cell for cell in reward_cells)
    assert "(iter 1)" in table.columns[0].header

    logger.log_step(iteration=5, return_mean_ep100=3.0, reward_components={"alive": 0.7})
    assert "(iter 5)" in logger._build_reward_table().columns[0].header


def test_offpolicy_backend_step_axis_falls_back_to_iteration() -> None:
    logger = OffPolicyLogger(log_backend="none")
    writer = _BatchedWriter()
    logger._tb_writer = writer

    logger.log_step(iteration=7, metrics={"Loss/critic": 1.0}, train_time=0.1)

    assert writer.events[0].step == 7
    backend_scalars = _batched_scalar_map(writer)
    assert "Train/iteration" not in backend_scalars
    assert "Perf/iteration_time" not in backend_scalars
    assert "Episode/timeout_rate" not in backend_scalars

    logger.update_timeout_rate(0.25)
    logger.log_step(iteration=8, metrics={"Loss/critic": 1.5}, train_time=0.1)

    assert _batched_scalar_map(writer)["Episode/timeout_rate"] == pytest.approx(0.25)


def test_offpolicy_collection_time_waits_for_complete_rollout_measurement() -> None:
    logger = OffPolicyLogger(log_backend="none")
    writer = _BatchedWriter()
    logger._tb_writer = writer

    logger.update_collector_timing({"env_step_ms": 2.0})
    logger.log_step(iteration=1, metrics={"Loss/critic": 1.0})
    assert "Perf/collection_time" not in _batched_scalar_map(writer)

    logger.update_collector_timing({"rollout_ms": 100.0})
    logger.log_step(iteration=2, metrics={"Loss/critic": 2.0})
    assert _batched_scalar_map(writer)["Perf/collection_time"] == pytest.approx(0.1)


def test_tensorboard_and_wandb_use_the_same_canonical_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = OffPolicyLogger(log_backend="none")
    writer = _BatchedWriter()
    logger._tb_writer = writer
    logger._wandb_run = object()

    wandb_calls: list[dict[str, Any]] = []

    class _FakeWandb:
        @staticmethod
        def log(payload: dict[str, Any], *, step: int) -> None:
            wandb_calls.append({"payload": payload, "step": step})

    monkeypatch.setattr(offpolicy_logger_module, "_load_wandb", lambda: _FakeWandb)

    logger.log_step(
        iteration=4,
        metrics={"Loss/actor": 2.0},
        return_mean_ep100=2.5,
        reward_components={"alive": 0.5},
        train_time=0.1,
    )

    assert len(writer.events) == 1
    assert len(wandb_calls) == 1
    assert writer.events[0].step == wandb_calls[0]["step"] == 4
    wandb_payload = wandb_calls[0]["payload"]
    tb_payload = _batched_scalar_map(writer)
    assert set(wandb_payload) == set(tb_payload)
    for key, value in wandb_payload.items():
        assert value == pytest.approx(tb_payload[key])
    assert wandb_calls[0]["payload"]["Loss/actor"] == pytest.approx(2.0)


def test_offpolicy_replay_ingress_diagnostics_use_canonical_backend_payload() -> None:
    logger = OffPolicyLogger(log_backend="none")
    writer = _BatchedWriter()
    logger._tb_writer = writer

    logger.log_step(
        iteration=1,
        metrics={
            "Train/replay_ingress_depth": 2.0,
            "Train/replay_ingress_occupancy": 2.0,
            "Train/replay_ingress_high_water": 2.0,
            "Train/replay_ingress_backpressure_wait_ms": 250.0,
            "Train/replay_ingress_dropped_batches": 1.0,
        },
    )

    backend_scalars = _batched_scalar_map(writer)
    assert backend_scalars["Train/replay_ingress_depth"] == pytest.approx(2.0)
    assert backend_scalars["Train/replay_ingress_backpressure_wait_ms"] == pytest.approx(250.0)
    assert backend_scalars["Train/replay_ingress_dropped_batches"] == pytest.approx(1.0)


def test_onpolicy_tensorboard_and_wandb_use_the_same_canonical_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = OnPolicyLogger(log_backend="none", num_envs=10, num_steps=3)
    writer = _BatchedWriter()
    logger._tb_writer = writer
    logger._wandb_run = object()

    wandb_calls: list[dict[str, Any]] = []

    class _FakeWandb:
        @staticmethod
        def log(payload: dict[str, Any], *, step: int) -> None:
            wandb_calls.append({"payload": payload, "step": step})

    monkeypatch.setattr(onpolicy_logger_module, "_load_wandb", lambda: _FakeWandb)

    logger.log_step(
        iteration=4,
        metrics={"Loss/surrogate": 2.0},
        return_mean_ep100=2.5,
        reward_components={"alive": 0.5},
        collect_time=0.1,
        train_time=0.2,
    )

    assert len(writer.events) == len(wandb_calls) == 1
    assert writer.events[0].step == wandb_calls[0]["step"] == 4
    tb_payload = _batched_scalar_map(writer)
    wandb_payload = wandb_calls[0]["payload"]
    assert set(wandb_payload) == set(tb_payload)
    for key, value in wandb_payload.items():
        assert value == pytest.approx(tb_payload[key])


def test_onpolicy_backend_uses_canonical_reward_and_loss_tags() -> None:
    logger = OnPolicyLogger(log_backend="none", num_envs=10, num_steps=3)
    writer = _BatchedWriter()
    logger._tb_writer = writer

    logger.log_step(
        iteration=2,
        metrics={"Loss/surrogate": 1.0},
        return_mean_ep100=2.0,
        reward_components={"alive": 0.5},
        collect_time=0.1,
        train_time=0.2,
    )

    assert len(writer.events) == 1
    assert writer.events[0].step == 2
    scalars = _batched_scalar_map(writer)
    assert scalars["Loss/surrogate"] == pytest.approx(1.0)
    assert scalars["Train/mean_reward"] == pytest.approx(2.0)
    assert scalars["reward/alive"] == pytest.approx(0.5)
    assert scalars["Perf/total_fps"] == pytest.approx(100.0)
    assert scalars["Perf/collection_time"] == pytest.approx(0.1)
    assert scalars["Perf/learning_time"] == pytest.approx(0.2)
    assert "Train/iteration" not in scalars
    assert "Perf/iteration_time" not in scalars
    assert {"reward/mean", "reward/mean_ep100", "perf/steps_per_sec"}.isdisjoint(scalars)


def test_onpolicy_source_episode_return_owner_fails_closed() -> None:
    logger = OnPolicyLogger(log_backend="none")

    with pytest.raises(ValueError, match="logger-owned metrics"):
        logger.log_step(
            iteration=1,
            metrics={"Train/mean_reward": 1.0},
            return_mean_ep100=2.0,
        )


def test_onpolicy_runner_iteration_metric_fails_closed() -> None:
    logger = OnPolicyLogger(log_backend="none")

    with pytest.raises(ValueError, match="iteration as the step axis"):
        logger.log_step(iteration=1, metrics={"Train/iteration": 1.0})
