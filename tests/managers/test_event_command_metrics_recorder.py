# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), manager tests.
# Modified by UniLab for NumPy scheduling and unsupported capability errors; Apache-2.0.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import unilab.managers as managers
from unilab.managers import (
    CommandManager,
    CommandTerm,
    CommandTermCfg,
    EventManager,
    EventTermCfg,
    MetricsManager,
    MetricsTermCfg,
    NullCommandManager,
    NullMetricsManager,
    NullRecorderManager,
    RecorderManager,
    RecorderTerm,
    RecorderTermCfg,
)

from .conftest import FakeEnv


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
    manager.apply("startup", env_ids=np.array([0, 2]))
    manager.apply("step", dt=0.01)
    manager.apply("interval", dt=0.1)
    manager.apply("reset", env_ids=np.array([1, 3]), global_env_step_count=1)
    manager.apply("reset", env_ids=np.array([1, 3]), global_env_step_count=2)
    assert [label for label, _ in fake_env.calls] == ["startup", "step", "interval", "reset"]
    np.testing.assert_array_equal(fake_env.calls[2][1], np.arange(fake_env.num_envs))


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
    np.testing.assert_array_equal(left._interval_term_time_left, right._interval_term_time_left)


class DummyCommand(CommandTerm):
    def __init__(self, cfg: DummyCommandCfg, env: FakeEnv):
        super().__init__(cfg, env)
        self._command = np.zeros((env.num_envs, 1), dtype=np.float32)
        self.metrics["error"] = np.arange(env.num_envs, dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        return self._command

    def _update_metrics(self) -> None:
        self.metrics["error"] += 1.0

    def _resample_command(self, env_ids: np.ndarray) -> None:
        self._command[env_ids, 0] = self.command_counter[env_ids]

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        pass


@dataclass(kw_only=True)
class DummyCommandCfg(CommandTermCfg):
    def build(self, env: FakeEnv) -> DummyCommand:
        return DummyCommand(self, env)


def test_command_resample_metrics_validation_and_null(fake_env: FakeEnv) -> None:
    manager = CommandManager({"goal": DummyCommandCfg(resampling_time_range=(0.5, 0.5))}, fake_env)
    extras = manager.reset(np.array([1, 2]))
    assert extras == {"Metrics/goal/error": 1.5}
    np.testing.assert_array_equal(manager.get_command("goal")[[1, 2]], 0.0)
    manager.compute(0.5)
    np.testing.assert_array_equal(manager.get_term("goal").command[:, 0], [0, 1, 1, 0])
    assert manager.get_term("goal").command_counter.tolist() == [1, 2, 2, 1]

    manager.get_term("goal").command[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        manager.get_command("goal")

    null = NullCommandManager()
    assert null.get_command("missing") is None
    assert null.reset() == {}


def test_command_viewer_request_and_old_signature_fail_closed(fake_env: FakeEnv) -> None:
    with pytest.raises(NotImplementedError, match="viewer"):
        CommandManager(
            {"goal": DummyCommandCfg(resampling_time_range=(1.0, 1.0), debug_vis=True)},
            fake_env,
        )

    class OldCommand(DummyCommand):
        def _update_command(self) -> None:  # type: ignore[override]
            pass

    @dataclass(kw_only=True)
    class OldCfg(CommandTermCfg):
        def build(self, env: FakeEnv) -> OldCommand:
            return OldCommand(self, env)

    with pytest.raises(TypeError, match="must accept env_ids"):
        CommandManager({"old": OldCfg(resampling_time_range=(1.0, 1.0))}, fake_env)


def test_metrics_reductions_substeps_reset_and_finite_failure(fake_env: FakeEnv) -> None:
    manager = MetricsManager(
        {
            "mean": MetricsTermCfg(func=lambda env: env.value.copy(), reduce="mean"),
            "max": MetricsTermCfg(func=lambda env: env.value.copy(), reduce="max"),
            "sum": MetricsTermCfg(func=lambda env: env.value.copy(), reduce="sum"),
            "last": MetricsTermCfg(func=lambda env: env.value.copy(), reduce="last"),
            "substep": MetricsTermCfg(
                func=lambda env: env.value.copy(), per_substep=True, reduce="mean"
            ),
        },
        fake_env,
    )
    manager.compute_substep()
    fake_env.value += 2
    manager.compute_substep()
    manager.compute()
    extras = manager.reset(np.array([1, 2]))
    assert extras["Episode_Metrics/mean"] == pytest.approx(3.5)
    assert extras["Episode_Metrics/max"] == pytest.approx(3.5)
    assert extras["Episode_Metrics/sum"] == pytest.approx(3.5)
    assert extras["Episode_Metrics/last"] == pytest.approx(3.5)
    assert extras["Episode_Metrics/substep"] == pytest.approx(2.5)

    bad = MetricsManager(
        {"bad": MetricsTermCfg(func=lambda env: np.full(env.num_envs, np.inf))}, fake_env
    )
    with pytest.raises(ValueError, match="MetricsManager term 'bad'"):
        bad.compute()
    assert NullMetricsManager().reset() == {}


class TraceRecorder(RecorderTerm):
    def __init__(self, cfg: RecorderTermCfg, env: FakeEnv):
        super().__init__(cfg, env)
        self.events: list[tuple[str, list[int] | None]] = []

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        self.events.append(("pre", env_ids.tolist()))

    def record_post_reset(self, env_ids: np.ndarray) -> None:
        self.events.append(("post_reset", env_ids.tolist()))

    def record_post_step(self) -> None:
        self.events.append(("step", None))

    def close(self) -> None:
        self.events.append(("close", None))


def test_recorder_lifecycle_and_null(fake_env: FakeEnv) -> None:
    cfg = {"trace": RecorderTermCfg(func=TraceRecorder)}
    manager = RecorderManager(cfg, fake_env)
    ids = np.array([0, 3])
    manager.record_pre_reset(ids)
    manager.record_post_reset(ids)
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
    forbidden_roots = {"torch", "mjlab"}
    forbidden_unilab = {"unilab.ipc", "unilab.runners", "unilab.scripts", "unilab.base.backend"}
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
