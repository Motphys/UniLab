# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), manager tests.
# Modified by UniLab for NumPy scheduling and unsupported capability errors; Apache-2.0.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

import unilab.managers as managers
from unilab.envs.mdp.recorders import LifecycleCounterRecorder
from unilab.managers import (
    ActionManager,
    ActionTerm,
    ActionTermCfg,
    CommandManager,
    CommandTerm,
    CommandTermCfg,
    EventManager,
    EventTermCfg,
    MetricsManager,
    MetricsTermCfg,
    NullMetricsManager,
    NullRecorderManager,
    RecorderManager,
    RecorderTerm,
    RecorderTermCfg,
)

from .conftest import FakeEnv


class DummyAction(ActionTerm):
    def __init__(self, cfg: DummyActionCfg, env: FakeEnv):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, cfg.dim), dtype=torch.float32)
        self.reset_ids: list[torch.Tensor] = []

    @property
    def action_dim(self) -> int:
        return self._raw.shape[1]

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw.copy_(actions)

    def apply_actions(self) -> None:
        return None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.reset_ids.append(torch.arange(self.num_envs, dtype=torch.int64))
            self._raw.zero_()
            return
        self.reset_ids.append(env_ids.clone())
        self._raw[env_ids] = 0.0


@dataclass(kw_only=True)
class DummyActionCfg(ActionTermCfg):
    dim: int

    def build(self, env: FakeEnv) -> DummyAction:
        return DummyAction(self, env)


def _record(env: FakeEnv, env_ids: np.ndarray | None, *, label: str) -> None:
    copied = None if env_ids is None else np.asarray(env_ids).copy()
    env.calls.append((label, copied))


def test_event_modes_interval_reset_throttle_and_order(fake_env: FakeEnv) -> None:
    cfg = {
        "startup": EventTermCfg(func=_record, params={"label": "startup"}, mode="startup"),
        "reset": EventTermCfg(
            func=_record,
            params={"label": "reset"},
            mode="reset",
            min_step_count_between_reset=3,
        ),
        "interval": EventTermCfg(
            func=_record,
            params={"label": "interval"},
            mode="interval",
            interval_range_s=(0.1, 0.1),
        ),
        "step": EventTermCfg(func=_record, params={"label": "step"}, mode="step"),
    }
    manager = EventManager(cfg, fake_env)
    assert list(manager.active_terms) == ["startup", "reset", "interval", "step"]
    manager.apply("startup", env_ids=torch.tensor([0, 2], dtype=torch.int64))
    manager.apply("step", dt=0.01)
    manager.apply("interval", dt=0.1)
    assert [label for label, _ in fake_env.calls] == ["startup", "step", "interval"]
    np.testing.assert_array_equal(fake_env.calls[2][1], np.arange(fake_env.num_envs))

    # First reset triggers every selected env (none has triggered before).
    fake_env.calls.clear()
    all_ids = torch.arange(fake_env.num_envs, dtype=torch.int64)
    manager.apply("reset", env_ids=all_ids, global_env_step_count=0)
    assert [label for label, _ in fake_env.calls] == ["reset"]
    np.testing.assert_array_equal(fake_env.calls[0][1], np.arange(fake_env.num_envs))

    # A reset inside the min-step window is filtered out per env row.
    fake_env.calls.clear()
    manager.apply("reset", env_ids=all_ids, global_env_step_count=1)
    assert fake_env.calls == []

    # Once enough steps elapsed, the reset term fires again.
    fake_env.calls.clear()
    manager.apply("reset", env_ids=all_ids, global_env_step_count=3)
    assert [label for label, _ in fake_env.calls] == ["reset"]
    np.testing.assert_array_equal(fake_env.calls[0][1], np.arange(fake_env.num_envs))


def test_event_validation_and_model_mutation_failure(fake_env: FakeEnv) -> None:
    with pytest.raises(ValueError, match="interval_range_s"):
        EventManager({"bad": EventTermCfg(func=_record, mode="interval")}, fake_env)

    def model_mutation(env: FakeEnv, env_ids: np.ndarray | None) -> None:
        pass

    model_mutation.model_fields = ("body_mass",)  # type: ignore[attr-defined]
    with pytest.raises(NotImplementedError, match="model-field mutation"):
        EventManager({"unsupported": EventTermCfg(func=model_mutation, mode="startup")}, fake_env)

    empty = EventManager({}, fake_env)
    empty.apply("interval")


def test_event_interval_rng_is_reproducible() -> None:
    cfg = {
        "interval": EventTermCfg(
            func=_record,
            params={"label": "interval"},
            mode="interval",
            interval_range_s=(0.1, 2.0),
        )
    }
    left = EventManager(cfg, FakeEnv(seed=17))
    right = EventManager(cfg, FakeEnv(seed=17))
    assert len(left._interval_term_time_left) == len(right._interval_term_time_left)
    for left_time, right_time in zip(
        left._interval_term_time_left, right._interval_term_time_left, strict=True
    ):
        assert isinstance(left_time, torch.Tensor)
        assert left_time.dtype == torch.float64
        assert left_time.device.type == torch.device("cpu").type
        torch.testing.assert_close(left_time, right_time)


def test_event_and_command_scheduling_fail_closed_without_rng() -> None:
    cfg = {
        "interval": EventTermCfg(
            func=_record,
            params={"label": "interval"},
            mode="interval",
            interval_range_s=(0.1, 2.0),
        )
    }
    without_rng = FakeEnv()
    without_rng.rng = None
    with pytest.raises(RuntimeError, match="Manager-owned Torch generator"):
        EventManager(cfg, without_rng)

    with pytest.raises(RuntimeError, match="Manager-owned Torch generator"):
        DummyCommandCfg(resampling_time_range=(1.0, 1.0)).build(without_rng)


class DummyCommand(CommandTerm):
    def __init__(self, cfg: DummyCommandCfg, env: FakeEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros((env.num_envs, 1), dtype=torch.float32)
        self.metrics["error"] = torch.arange(env.num_envs, dtype=torch.float32)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        self.metrics["error"] += 1.0

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        self._command.copy_(
            torch.where(
                reset_mask[:, None],
                self.command_counter.to(torch.float32)[:, None],
                self._command,
            )
        )

    def _update_command(self) -> None:
        pass

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask


@dataclass(kw_only=True)
class DummyCommandCfg(CommandTermCfg):
    def build(self, env: FakeEnv) -> DummyCommand:
        return DummyCommand(self, env)


def test_command_resample_metrics_validation_and_null(fake_env: FakeEnv) -> None:
    manager = CommandManager({"goal": DummyCommandCfg(resampling_time_range=(0.5, 0.5))}, fake_env)
    extras = manager.reset(torch.tensor([1, 2], dtype=torch.int64))
    assert extras == {"Metrics/goal/error": 1.5}
    np.testing.assert_array_equal(manager.get_command("goal")[[1, 2]], 0.0)
    manager.compute(0.5)
    np.testing.assert_array_equal(manager.get_term("goal").command[:, 0], [0, 1, 1, 0])
    assert manager.get_term("goal").command_counter.tolist() == [1, 2, 2, 1]


def test_deferred_command_metrics_drain_without_reset_on_boundary(fake_env: FakeEnv) -> None:
    setattr(fake_env, "defer_finite_validation", True)
    setattr(fake_env, "finite_validation_interval", 4)
    command = DummyCommand(DummyCommandCfg(resampling_time_range=(1.0, 1.0)), fake_env)

    setattr(fake_env, "step_counter", 1)
    assert command.reset(torch.tensor([1], dtype=torch.int64)) == {}
    command.metrics["error"].copy_(torch.tensor([4.0, 5.0, 6.0, 7.0]))
    setattr(fake_env, "step_counter", 5)
    assert command.reset(torch.tensor([2], dtype=torch.int64)) == {}
    command.metrics["error"].copy_(torch.tensor([8.0, 9.0, 10.0, 11.0]))
    setattr(fake_env, "step_counter", 9)
    assert command.reset(torch.tensor([0], dtype=torch.int64)) == {}

    # No reset lands on a multiple of four. The environment drains this
    # accumulator independently of reset at the next validation boundary.
    setattr(fake_env, "step_counter", 12)
    extras = command.publish_pending_reset_metrics()

    assert extras["error"] == pytest.approx((1.0 + 6.0 + 8.0) / 3.0)
    assert command.publish_pending_reset_metrics() == {}


def test_command_reset_rejects_out_of_range_env_ids(fake_env: FakeEnv) -> None:
    manager = CommandManager({"goal": DummyCommandCfg(resampling_time_range=(1.0, 1.0))}, fake_env)
    with pytest.raises(ValueError, match="CommandManager.reset"):
        manager.reset(torch.tensor([0, fake_env.num_envs], dtype=torch.int64))


def test_command_reset_uses_owner_env_ids_boundary(fake_env: FakeEnv) -> None:
    class IdsCommand(DummyCommand):
        def reset(
            self, env_ids: torch.Tensor | None = None, *, publish_metrics: bool = True
        ) -> dict[str, float]:
            del publish_metrics
            assert env_ids is not None
            self._command[env_ids] = 7.0
            self.command_counter[env_ids] = 0
            return {}

    @dataclass(kw_only=True)
    class IdsCfg(DummyCommandCfg):
        def build(self, env: FakeEnv) -> IdsCommand:
            return IdsCommand(self, env)

    manager = CommandManager({"goal": IdsCfg(resampling_time_range=(1.0, 1.0))}, fake_env)
    manager.get_term("goal")._command[:] = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    manager.get_term("goal").command_counter[:] = torch.arange(4, dtype=torch.int64)

    extras = manager.reset(torch.tensor([1, 3], dtype=torch.int64))

    assert extras == {}
    torch.testing.assert_close(
        manager.get_term("goal").command,
        torch.tensor([[0.0], [7.0], [2.0], [7.0]]),
    )
    torch.testing.assert_close(
        manager.get_term("goal").command_counter,
        torch.tensor([0, 0, 2, 0], dtype=torch.int64),
    )


def test_tensor_metric_clearing_does_not_mirror_reset_rows_on_host(
    fake_env: FakeEnv,
) -> None:
    command = DummyCommand(DummyCommandCfg(resampling_time_range=(1.0, 1.0)), fake_env)
    command.metrics = {
        "tensor": torch.arange(fake_env.num_envs, dtype=torch.float32),
        "other": torch.arange(fake_env.num_envs, dtype=torch.float32) * 2.0,
    }
    mask = torch.tensor([False, True, True, False], dtype=torch.bool)

    def fail_cpu(*args: object) -> torch.Tensor:
        raise AssertionError("all-tensor metric clearing must not mirror reset rows")

    original_cpu = torch.Tensor.cpu
    torch.Tensor.cpu = fail_cpu  # type: ignore[method-assign]
    try:
        command.clear_episode_metrics(mask)
    finally:
        torch.Tensor.cpu = original_cpu  # type: ignore[method-assign]

    torch.testing.assert_close(command.metrics["tensor"], torch.tensor([0.0, 0.0, 0.0, 3.0]))
    torch.testing.assert_close(command.metrics["other"], torch.tensor([0.0, 0.0, 0.0, 6.0]))


def test_command_viewer_request_fails_closed(fake_env: FakeEnv) -> None:
    with pytest.raises(NotImplementedError, match="viewer"):
        CommandManager(
            {"goal": DummyCommandCfg(resampling_time_range=(1.0, 1.0), debug_vis=True)},
            fake_env,
        )


def test_command_reset_refresh_does_not_validate_full_command_or_metrics(
    fake_env: FakeEnv,
) -> None:
    manager = CommandManager({"goal": DummyCommandCfg(resampling_time_range=(1.0, 1.0))}, fake_env)
    command = manager.get_term("goal")
    mask = torch.tensor([False, True, False, False], dtype=torch.bool)
    command._command[2, 0] = np.nan
    command.metrics["error"][1] = np.nan
    metrics_before = command.metrics["error"].clone()
    scalar_conversions = 0
    original_bool = torch.Tensor.__bool__

    def counted_bool(self: torch.Tensor) -> bool:
        nonlocal scalar_conversions
        scalar_conversions += 1
        return original_bool(self)

    torch.Tensor.__bool__ = counted_bool  # type: ignore[method-assign]
    try:
        manager.refresh_reset(mask)
    finally:
        torch.Tensor.__bool__ = original_bool  # type: ignore[method-assign]

    assert scalar_conversions == 0
    np.testing.assert_array_equal(command._command[:, 0], [0.0, 0.0, np.nan, 0.0])
    np.testing.assert_array_equal(command.metrics["error"], metrics_before)
    assert bool(torch.isnan(command.metrics["error"][1]))


def test_metrics_reductions_substeps_reset_and_finite_failure(fake_env: FakeEnv) -> None:
    manager = MetricsManager(
        {
            "mean": MetricsTermCfg(func=lambda env: torch.as_tensor(env.value), reduce="mean"),
            "max": MetricsTermCfg(func=lambda env: torch.as_tensor(env.value), reduce="max"),
            "sum": MetricsTermCfg(func=lambda env: torch.as_tensor(env.value), reduce="sum"),
            "last": MetricsTermCfg(func=lambda env: torch.as_tensor(env.value), reduce="last"),
            "substep": MetricsTermCfg(
                func=lambda env: torch.as_tensor(env.value), per_substep=True, reduce="mean"
            ),
        },
        fake_env,
    )
    manager.compute_substep()
    fake_env.value += 2
    manager.compute_substep()
    manager.compute()
    assert isinstance(manager._step_values, torch.Tensor)
    assert manager._step_values.device.type == "cpu"
    assert all(isinstance(values, torch.Tensor) for values in manager._episode_sums.values())
    extras = manager.reset(torch.tensor([1, 2], dtype=torch.int64))
    assert extras["Episode_Metrics/mean"] == pytest.approx(3.5)
    assert extras["Episode_Metrics/max"] == pytest.approx(3.5)
    assert extras["Episode_Metrics/sum"] == pytest.approx(3.5)
    assert extras["Episode_Metrics/last"] == pytest.approx(3.5)
    assert extras["Episode_Metrics/substep"] == pytest.approx(2.5)

    bad = MetricsManager(
        {"bad": MetricsTermCfg(func=lambda env: torch.full((env.num_envs,), torch.inf))},
        fake_env,
    )
    with pytest.raises(ValueError, match="MetricsManager term 'bad'"):
        bad.compute()
    wrong_dtype = MetricsManager(
        {
            "wrong_dtype": MetricsTermCfg(
                func=lambda env: torch.ones(env.num_envs, dtype=torch.float64)
            )
        },
        fake_env,
    )
    with pytest.raises(TypeError, match="expected float32"):
        wrong_dtype.compute()
    numpy_carrier = MetricsManager(
        {"numpy_carrier": MetricsTermCfg(func=lambda env: np.ones(env.num_envs))},
        fake_env,
    )
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        numpy_carrier.compute()
    assert NullMetricsManager().reset(torch.empty(0, dtype=torch.int64)) == {}


def test_metrics_deferred_finite_verdict_is_polled_at_boundary(fake_env: FakeEnv) -> None:
    setattr(fake_env, "defer_finite_validation", True)
    manager = MetricsManager(
        {"value": MetricsTermCfg(func=lambda env: torch.as_tensor(env.value))},
        fake_env,
    )

    manager.compute()
    assert manager.poll_finite_validation()

    fake_env.value[1] = torch.inf
    manager.compute()
    with pytest.raises(ValueError, match="MetricsManager returned non-finite"):
        manager.poll_finite_validation()


def test_deferred_metrics_drain_without_reset_on_boundary(fake_env: FakeEnv) -> None:
    setattr(fake_env, "defer_finite_validation", True)
    setattr(fake_env, "finite_validation_interval", 4)
    manager = MetricsManager(
        {"value": MetricsTermCfg(func=lambda env: env.value, reduce="mean")},
        fake_env,
    )

    setattr(fake_env, "step_counter", 1)
    manager.compute()
    assert manager.reset(torch.tensor([1], dtype=torch.int64)) == {}
    fake_env.value = torch.tensor([4.0, 5.0, 6.0, 7.0])
    setattr(fake_env, "step_counter", 5)
    manager.compute()
    assert manager.reset(torch.tensor([2], dtype=torch.int64)) == {}
    fake_env.value = torch.tensor([8.0, 9.0, 10.0, 11.0])
    setattr(fake_env, "step_counter", 9)
    manager.compute()
    assert manager.reset(torch.tensor([0], dtype=torch.int64)) == {}

    setattr(fake_env, "step_counter", 12)
    extras = manager.publish_pending_reset_metrics()

    assert extras["Episode_Metrics/value"] == pytest.approx(3.0)
    assert manager.publish_pending_reset_metrics() == {}


def test_metrics_reset_publishes_and_clears_only_selected_rows(fake_env: FakeEnv) -> None:
    class SelectedRowsMetric:
        def __init__(self) -> None:
            self.cleared: torch.Tensor | None = None

        def __call__(self, env: FakeEnv) -> torch.Tensor:
            return torch.as_tensor(env.value)

    term = SelectedRowsMetric()
    term_configs = [
        MetricsTermCfg(func=term, reduce=reduce) for reduce in ("mean", "max", "sum", "last")
    ]
    manager = MetricsManager(
        {name: cfg for name, cfg in zip(("mean", "max", "sum", "last"), term_configs)},
        fake_env,
    )
    manager.compute_substep()
    fake_env.value += 2
    manager.compute_substep()
    manager.compute()
    before = {key: value.clone() for key, value in manager._episode_sums.items()}
    mask = torch.tensor([False, True, False, True])

    extras = manager.reset(torch.tensor([1, 3], dtype=torch.int64))

    assert extras["Episode_Metrics/mean"] == pytest.approx(4.0)
    assert extras["Episode_Metrics/max"] == pytest.approx(4.0)
    assert extras["Episode_Metrics/sum"] == pytest.approx(4.0)
    assert extras["Episode_Metrics/last"] == pytest.approx(4.0)
    # The same callable backs all four reductions; manager-owned carriers
    # prove selected-row clearing without requiring a stateful metric term.
    for key, values in manager._episode_sums.items():
        assert values[mask].sum().item() == 0.0
        torch.testing.assert_close(values[~mask], before[key][~mask])
    assert manager._step_count[mask].sum().item() == 0
    assert manager._step_count[~mask].sum().item() > 0


class TraceRecorder(RecorderTerm):
    def __init__(self, cfg: RecorderTermCfg, env: FakeEnv):
        super().__init__(cfg, env)
        self.events: list[tuple[str, list[int] | None]] = []

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        self.events.append(("pre", env_ids.tolist()))

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        self.events.append(("post_reset", env_ids.tolist()))

    def record_post_step(self) -> None:
        self.events.append(("step", None))

    def close(self) -> None:
        self.events.append(("close", None))


def test_recorder_lifecycle_and_null(fake_env: FakeEnv) -> None:
    cfg = {"trace": RecorderTermCfg(func=TraceRecorder)}
    manager = RecorderManager(cfg, fake_env)
    env_ids = torch.tensor([0, 3], dtype=torch.int64)
    manager.record_pre_reset(env_ids)
    manager.record_post_reset(env_ids)
    manager.record_post_step()
    manager.close()
    assert manager.get_term("trace").events == [
        ("pre", [0, 3]),
        ("post_reset", [0, 3]),
        ("step", None),
        ("close", None),
    ]
    assert cfg["trace"].func is TraceRecorder
    null = NullRecorderManager()
    with pytest.raises(KeyError, match="has no terms"):
        null.get_term("trace")


def test_lifecycle_counter_recorder_is_task_independent(fake_env: FakeEnv) -> None:
    manager = RecorderManager(
        {"counter": RecorderTermCfg(func=LifecycleCounterRecorder)},
        fake_env,
    )
    recorder = manager.get_term("counter")
    env_ids = torch.tensor([0, 3], dtype=torch.int64)

    manager.record_pre_reset(env_ids)
    empty = torch.empty(0, dtype=torch.int64)
    manager.record_pre_reset(empty)
    manager.record_post_reset(env_ids)
    manager.record_post_step()

    assert isinstance(recorder, LifecycleCounterRecorder)
    assert recorder.pre_reset_count == 2
    assert recorder.post_reset_count == 2
    assert recorder.post_step_count == 1


def test_public_exports_and_repository_import_boundary() -> None:
    expected = {
        "ManagerBase",
        "ManagerTermBase",
        "ManagerTermBaseCfg",
        "ActionManager",
        "ObservationManager",
        "RewardManager",
        "TerminationManager",
        "EventManager",
        "CommandManager",
        "CurriculumManager",
        "MetricsManager",
        "RecorderManager",
        "SceneEntityCfg",
    }
    assert expected <= set(vars(managers))

    package_root = Path(managers.__file__).parent
    # ADR-0011 supersedes ADR-0006's NumPy-only Manager constraint. Torch is
    # the target execution carrier; mjlab remains a cold-path source baseline,
    # not a runtime dependency.
    forbidden_roots = {"mjlab"}
    forbidden_unilab = {"uni_rl", "unilab.runners", "unilab.scripts", "unilab.base.backend"}
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not ({name.split(".")[0] for name in imports} & forbidden_roots), path
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in forbidden_unilab
        ), path


def test_command_reset_can_skip_metric_publication(fake_env: FakeEnv) -> None:
    manager = CommandManager({"goal": DummyCommandCfg(resampling_time_range=(1.0, 1.0))}, fake_env)
    term = manager.get_term("goal")
    term.metrics["error"].fill_(2.0)
    env_ids = torch.tensor([1, 3], dtype=torch.int64)

    assert term.reset(env_ids, publish_metrics=False) == {}
    np.testing.assert_array_equal(term.metrics["error"], [2.0, 0.0, 2.0, 0.0])
    assert "reset_done_reset_validation_ms" not in manager.last_reset_timing_ms


def test_action_and_metric_state_clear_boundaries_match_selected_rows(fake_env: FakeEnv) -> None:
    action = ActionManager({"joint": DummyActionCfg(entity_name="robot", dim=1)}, fake_env)
    action.prev_action.fill_(1.0)
    action.prev_prev_action.fill_(2.0)
    action.action.fill_(3.0)
    action.clear_action_state(torch.tensor([0, 2], dtype=torch.int64))

    torch.testing.assert_close(action.action, torch.tensor([[0.0], [3.0], [0.0], [3.0]]))
    torch.testing.assert_close(action.prev_action, torch.tensor([[0.0], [1.0], [0.0], [1.0]]))
    torch.testing.assert_close(action.prev_prev_action, torch.tensor([[0.0], [2.0], [0.0], [2.0]]))

    metrics = MetricsManager(
        {
            "sum": MetricsTermCfg(func=lambda env: env.value.copy(), reduce="sum"),
            "max": MetricsTermCfg(func=lambda env: env.value.copy(), reduce="max"),
        },
        fake_env,
    )
    metrics._episode_sums["sum"].fill_(4.0)
    metrics._episode_max["max"].fill_(5.0)
    metrics._step_count.fill_(2)
    metrics.clear_episode_state(torch.tensor([1, 3], dtype=torch.int64))

    torch.testing.assert_close(metrics._episode_sums["sum"], torch.tensor([4.0, 0.0, 4.0, 0.0]))
    torch.testing.assert_close(
        metrics._episode_max["max"],
        torch.tensor([5.0, float("-inf"), 5.0, float("-inf")]),
    )
    torch.testing.assert_close(metrics._step_count, torch.tensor([2, 0, 2, 0], dtype=torch.int64))
