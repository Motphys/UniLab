"""Fixed-shape (``ResetSelection``) autoreset, persistent step buffers, deferred
fail-closed checks and CUDA graph capture of the autoreset segment.

The env under test runs on a device-resident fake backend whose physics is pure
Torch, so the same scenario runs on CPU and, when available, on CUDA (where the
autoreset segment is captured into a CUDA graph).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from unisim.backend.base import (
    SelectedResetPublication,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)
from unisim.backend.graph_capture import ExternalGraphCapture, GraphCaptureError, closing_capture

from tests.envs.test_manager_based_rl_env import _ResetBackend
from unilab.base.entity import EntityCfg
from unilab.base.reset_selection import (
    ResetSelection,
    assign_rows_,
    fill_rows_,
    reset_selection_capable,
    supports_reset_selection,
)
from unilab.base.reset_selectors import ResetMaskBuffer, resolve_reset_env_ids
from unilab.base.scene import SceneCfg
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from unilab.managers import (
    ActionTerm,
    ActionTermCfg,
    CircularBuffer,
    CommandTerm,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

NUM_ENVS = 8
CUDA = torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="CUDA"))]


# ---------------------------------------------------------------------------
# Device-resident fake backend
# ---------------------------------------------------------------------------


class _TensorBackend(_ResetBackend):
    """Pure-Torch device-resident backend: no host reads on the hot path."""

    backend_type = "fake_tensor"

    def __init__(self, num_envs: int, device: str) -> None:
        super().__init__(num_envs)
        self.device = torch.device(device)
        self.qpos = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.qvel = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.capture_scopes = 0

    def get_tensor_execution(self) -> TensorExecution:
        return TensorExecution.DEVICE_RESIDENT

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            stepping=True,
            selected_reset=True,
            mask_selected_reset=True,
            reset_randomization=True,
            packed_reset=True,
            device_reset_randomization=True,
            packed_reset_fields=frozenset(("body_mass", "body_ipos", "geom_friction", "kp", "kd")),
            selected_reset_publication=SelectedResetPublication.AUTHORITATIVE_VIEWS,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="test",
            torch_devices=(self.device.type,),
            external_graph_capture=self.device.type == "cuda",
        )

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        views: dict[str, torch.Tensor] = {}
        if "qpos" in fields:
            views["qpos"] = self.qpos
        if "qvel" in fields:
            views["qvel"] = self.qvel
        return views

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> None:
        del randomization
        self.qpos[env_indices] = qpos
        self.qvel[env_indices] = qvel

    def set_state_tensor_mask(self, reset_mask, qpos, qvel, randomization=None, **_kwargs):
        del randomization
        self.qpos.copy_(torch.where(reset_mask[:, None], qpos, self.qpos))
        self.qvel.copy_(torch.where(reset_mask[:, None], qvel, self.qvel))
        return {"timing": {}}

    def step_tensor(self, ctrl: torch.Tensor, nsteps: int = 1) -> dict[str, Any]:
        self.qpos[:, 3:4].add_(ctrl * (0.01 * nsteps))
        return {"timing": {}}

    @contextmanager
    def torch_graph_capture_scope(
        self, torch_stream: Any | None = None, *, capture_error_mode: str = "global"
    ) -> Iterator[ExternalGraphCapture]:
        del capture_error_mode
        if not torch.cuda.is_current_stream_capturing():
            raise GraphCaptureError("scope requires an active Torch capture")
        self.capture_scopes += 1
        stream = torch_stream or torch.cuda.current_stream(self.device)
        with closing_capture(ExternalGraphCapture(torch_stream=stream)) as capture:
            yield capture


# ---------------------------------------------------------------------------
# Selection-capable manager terms
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class _DriveCfg(ActionTermCfg):
    def build(self, env) -> ActionTerm:
        return _DriveAction(self, env)


class _DriveAction(ActionTerm):
    supports_reset_selection = True

    def __init__(self, cfg: _DriveCfg, env) -> None:
        super().__init__(cfg, env)
        self._processed = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._processed.copy_(actions)

    def reset(self, env_ids: torch.Tensor | ResetSelection | None = None) -> None:
        fill_rows_(self._processed, env_ids, 0.0)

    def apply_actions(self) -> None:
        self._entity.data.write_ctrl(self._processed)


@dataclass(kw_only=True)
class _TargetCfg(CommandTermCfg):
    def build(self, env) -> CommandTerm:
        return _Target(self, env)


class _Target(CommandTerm):
    supports_reset_selection = True

    def __init__(self, cfg: _TargetCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=env.device)
        self.metrics["error"] = torch.zeros(self.num_envs, dtype=torch.float32, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        self.metrics["error"].add_(self._command[:, 0].abs())

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        samples = self._env.rng.uniform(-1.0, 1.0, self._command.shape, dtype=torch.float32)
        self._command.copy_(torch.where(reset_mask[:, None], samples, self._command))

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def post_compute_reset(self) -> None:
        return None

    def _update_command(self) -> None:
        return None


@reset_selection_capable
def _reset_joint(env: ManagerBasedRlEnv, env_ids: torch.Tensor | ResetSelection) -> None:
    """Reset event: joint position from the selection-capable command sample."""
    target = env.command_manager.get_command("target")
    env.scene["robot"].write_joint_state_to_sim_mask(
        target.clone(),
        torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device),
        reset_mask=env.event_term_mask_buffer.scatter(env_ids),
    )


def _joint_reset_ids_only(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Undeclared reset event (ids only)."""
    del env, env_ids


def _policy_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.column_stack(
        (
            env.episode_length_buf.to(torch.float32),
            env.action_manager.action[:, 0],
            env.command_manager.get_command("target")[:, 0],
            # The fake backend's joint qpos column (device view, no host read).
            cast(_TensorBackend, env.backend).qpos[:, 3],
        )
    )


def _critic_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf[:, None].to(torch.float32)


def _reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.action_manager.action[:, 0] - env.command_manager.get_command("target")[:, 0]


def _failure(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.action_manager.action[:, 0] > 0.8


def _time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf >= 4


def _metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf.to(torch.float32)


def _curriculum(env: ManagerBasedRlEnv, env_ids: Any) -> float:
    del env, env_ids
    return 0.0


def _make_cfg(device: str, **overrides: Any) -> ManagerBasedRlEnvCfg:
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file="fake.xml",
            entities={"robot": EntityCfg(joint_names=("joint",), actuator_names=("motor",))},
        ),
        sim_dt=0.01,
        ctrl_dt=0.02,
        max_episode_seconds=1.0,
        seed=11,
        collector_tensor_device=device,
        observations={
            "actor": ObservationGroupCfg(
                terms={"policy": ObservationTermCfg(func=_policy_obs, history_length=3)}
            ),
            "value": ObservationGroupCfg(terms={"critic": ObservationTermCfg(func=_critic_obs)}),
        },
        actions={"drive": _DriveCfg(entity_name="robot")},
        rewards={"track": RewardTermCfg(func=_reward, weight=1.0)},
        terminations={
            "failure": TerminationTermCfg(func=_failure),
            "time_out": TerminationTermCfg(func=_time_out, time_out=True),
        },
        events={"joint": EventTermCfg(func=_reset_joint, mode="reset")},
        commands={"target": _TargetCfg(resampling_time_range=(10.0, 10.0))},
        metrics={"length": MetricsTermCfg(func=_metric, reduce="mean")},
        policy_observation_group="actor",
        critic_observation_group="value",
        defer_finite_validation=True,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _make_env(device: str, **overrides: Any) -> tuple[ManagerBasedRlEnv, _TensorBackend]:
    backend = _TensorBackend(NUM_ENVS, device)
    env = ManagerBasedRlEnv(_make_cfg(device, **overrides), cast(SimBackend, backend), NUM_ENVS)
    return env, backend


def _actions(device: str, steps: int) -> list[torch.Tensor]:
    """Random actions; row 0 fails on every step.

    The masked autoreset dispatches its reset participants on every step, so
    full-width reset samplers draw RNG even when no row is done, whereas the
    selected-row path skips steps without a done row. Keeping one row failing
    every step makes both lifecycles consume the RNG stream identically, so
    same-seed trajectories can be compared exactly.
    """
    generator = torch.Generator().manual_seed(3)
    actions = []
    for _ in range(steps):
        action = torch.rand((NUM_ENVS, 1), generator=generator) * 2.0 - 1.0
        action[0] = 0.9
        actions.append(action.to(device))
    return actions


def _snapshot(env: ManagerBasedRlEnv, backend: _TensorBackend) -> dict[str, torch.Tensor]:
    state = env.state
    assert state is not None
    out = {f"obs.{k}": v.detach().clone() for k, v in state.obs.items()}
    out["reward"] = state.reward.detach().clone()
    out["terminated"] = state.terminated.detach().clone()
    out["truncated"] = state.truncated.detach().clone()
    out["steps"] = state.info["steps"].detach().clone()
    out["qpos"] = backend.qpos.detach().clone()
    out["command"] = env.command_manager.get_command("target").detach().clone()
    # The masked autoreset publishes final_observation every step (meaningful
    # on done rows only); the selected-row path publishes it only when some
    # row reset. Compare the done rows.
    done = state.terminated | state.truncated
    final = state.final_observation or {k: torch.zeros_like(v) for k, v in state.obs.items()}
    for name, value in final.items():
        out[f"final.{name}"] = torch.where(done[:, None], value, 0.0).detach().clone()
    return out


def _run(env: ManagerBasedRlEnv, backend: _TensorBackend, actions: list[torch.Tensor]):
    env.reset()
    trajectory = [_snapshot(env, backend)]
    logs = []
    for action in actions:
        env.step(action)
        trajectory.append(_snapshot(env, backend))
        logs.append(dict(env.state.info.get("log", {})))
    return trajectory, logs


def _assert_same(left, right) -> None:
    assert len(left) == len(right)
    for step, (a, b) in enumerate(zip(left, right, strict=True)):
        assert a.keys() == b.keys(), step
        for key in a:
            torch.testing.assert_close(a[key].cpu(), b[key].cpu(), msg=f"step {step} {key}")


# ---------------------------------------------------------------------------
# ResetSelection primitives
# ---------------------------------------------------------------------------


def test_reset_selection_helpers_write_only_selected_rows() -> None:
    selection = ResetSelection.allocate(4, "cpu")
    selection.mask.copy_(torch.tensor([True, False, True, False]))
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    fill_rows_(values, selection, -1.0)
    assert values[:, 0].tolist() == [-1.0, 2.0, -1.0, 6.0]

    history = torch.zeros((3, 4, 2))
    fill_rows_(history, selection, 5.0, row_dim=1)
    assert history[:, 1].abs().sum() == 0 and bool((history[:, 0] == 5.0).all())

    timers = torch.zeros(4)
    assign_rows_(timers, selection, lambda count: torch.full((count,), 7.0))
    assert timers.tolist() == [7.0, 0.0, 7.0, 0.0]
    assign_rows_(timers, torch.tensor([1]), lambda count: torch.full((count,), 3.0))
    assert timers.tolist() == [7.0, 3.0, 7.0, 0.0]


def test_reset_override_must_redeclare_selection_support() -> None:
    class Parent:
        supports_reset_selection = True

        def reset(self, env_ids=None) -> None:
            del env_ids

    class InheritsReset(Parent):
        pass

    class OverridesReset(Parent):
        def reset(self, env_ids=None) -> None:
            del env_ids

    class Redeclares(OverridesReset):
        supports_reset_selection = True

    class Withdraws(Parent):
        supports_reset_selection = False

    assert supports_reset_selection(Parent())
    assert supports_reset_selection(InheritsReset())
    assert not supports_reset_selection(OverridesReset())
    assert not supports_reset_selection(OverridesReset)
    assert supports_reset_selection(Redeclares())
    assert not supports_reset_selection(Withdraws())

    def term(env, env_ids) -> None:
        del env, env_ids

    assert not supports_reset_selection(term)
    assert supports_reset_selection(reset_selection_capable(term))


def test_selection_is_rejected_as_an_index_and_validated_by_resolvers() -> None:
    selection = ResetSelection.allocate(4, "cpu")
    with pytest.raises((TypeError, IndexError)):
        torch.zeros(4)[selection]  # type: ignore[index]
    assert resolve_reset_env_ids(selection, num_envs=4, device="cpu", owner="t") is selection
    with pytest.raises(ValueError, match="covers 4 rows"):
        resolve_reset_env_ids(selection, num_envs=5, device="cpu", owner="t")
    buffer = ResetMaskBuffer(4, torch.device("cpu"))
    selection.mask[2] = True
    assert buffer.scatter(selection).tolist() == [False, False, True, False]
    with pytest.raises(TypeError):
        ResetSelection(mask=torch.zeros(4), row_ids=torch.arange(4))


def test_circular_buffer_selection_reset_and_backfill_match_ids() -> None:
    data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    by_ids, by_mask = CircularBuffer(3, 4), CircularBuffer(3, 4)
    for buffer in (by_ids, by_mask):
        buffer.append(data)
        buffer.append(data + 100.0)
    ids = torch.tensor([1, 3])
    selection = ResetSelection.allocate(4, "cpu")
    selection.mask[ids] = True
    by_ids.reset(ids)
    by_ids.backfill(data * 2.0, ids)
    by_mask.reset(selection)
    by_mask.backfill(data * 2.0, selection)
    torch.testing.assert_close(by_ids.buffer, by_mask.buffer)
    by_ids.append(data + 7.0)
    by_mask.append(data + 7.0)
    torch.testing.assert_close(by_ids.buffer, by_mask.buffer)


# ---------------------------------------------------------------------------
# Masked autoreset: negotiation, equivalence, buffers, deferred checks
# ---------------------------------------------------------------------------


def test_masked_autoreset_requires_deferred_validation() -> None:
    with pytest.raises(ValueError, match="defer_finite_validation"):
        _make_cfg("cpu", masked_autoreset=True, defer_finite_validation=False).validate()
    with pytest.raises(ValueError, match="requires masked_autoreset"):
        _make_cfg("cpu", autoreset_cuda_graph=True).validate()


def test_masked_autoreset_reports_every_unsupported_participant() -> None:
    env, _ = _make_env(
        "cpu",
        masked_autoreset=True,
        events={"ids_only": EventTermCfg(func=_joint_reset_ids_only, mode="reset")},
        curriculum={"level": CurriculumTermCfg(func=_curriculum)},
    )
    with pytest.raises(NotImplementedError) as raised:
        env.reset()
    message = str(raised.value)
    assert "_joint_reset_ids_only" in message
    assert "CurriculumManager: term 'level'" in message
    assert supports_reset_selection(_reset_joint)
    assert not supports_reset_selection(_joint_reset_ids_only)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("depth", [0, 2])
def test_masked_autoreset_matches_selected_row_autoreset(device: str, depth: int) -> None:
    actions = _actions(device, 24)
    eager_env, eager_backend = _make_env(device)
    masked_env, masked_backend = _make_env(
        device,
        masked_autoreset=True,
        persistent_step_buffers=True,
        published_buffer_depth=depth,
    )
    eager, eager_logs = _run(eager_env, eager_backend, actions)
    masked, masked_logs = _run(masked_env, masked_backend, actions)
    # Both lifecycles draw reset randomness with full-width masked draws, so
    # same-seed trajectories are identical, including reset rows.
    _assert_same(eager, masked)
    assert eager_logs == masked_logs
    assert any(bool((snap["terminated"] | snap["truncated"]).any()) for snap in masked)


def test_persistent_buffers_keep_addresses_and_publish_a_stable_ring() -> None:
    env, backend = _make_env(
        "cpu", masked_autoreset=True, persistent_step_buffers=True, published_buffer_depth=2
    )
    env.reset()
    working = env._state
    assert working is not None
    working_ptrs = {name: value.data_ptr() for name, value in working.obs.items()}
    working_ptrs["reward"] = working.reward.data_ptr()
    assert env.obs_buf is working.obs

    published = []
    for action in _actions("cpu", 6):
        state = env.step(action)
        published.append((state, {k: v.clone() for k, v in state.obs.items()}))
        assert env._state is working
        assert {n: v.data_ptr() for n, v in working.obs.items()} == {
            n: p for n, p in working_ptrs.items() if n != "reward"
        }
        assert working.reward.data_ptr() == working_ptrs["reward"]
    # Ring of two persistent slots: alternate objects, and a published state is
    # unchanged by the following step (the off-policy collector's obs_t).
    assert published[0][0] is published[2][0] and published[1][0] is published[3][0]
    assert published[0][0] is not published[1][0]
    last_state, last_values = published[-1]
    previous_state, previous_values = published[-2]
    for name in previous_values:
        torch.testing.assert_close(previous_state.obs[name], previous_values[name])
        torch.testing.assert_close(last_state.obs[name], last_values[name])
    del backend


def test_masked_step_has_no_reset_gate_and_reports_reset_call_timing() -> None:
    env, _ = _make_env("cpu", masked_autoreset=True)
    env.reset()
    env.set_episode_length_buf(torch.zeros(NUM_ENVS, dtype=torch.int64))
    calls = []
    original = env.reset

    def counting_reset(env_ids=None, **kwargs):
        calls.append(type(env_ids).__name__)
        return original(env_ids, **kwargs)

    env.reset = counting_reset  # type: ignore[method-assign]
    state = env.step(torch.zeros((NUM_ENVS, 1)))
    assert calls == ["ResetSelection"]
    assert not bool((state.terminated | state.truncated).any())
    assert state.info["timing"]["reset_done_count"] == 0.0
    assert state.info["timing"]["reset_done_reset_call_ms"] >= 0.0


def test_deferred_named_checks_fail_closed_at_the_poll_boundary() -> None:
    env, _ = _make_env("cpu", masked_autoreset=True)
    env.reset()
    env.require(torch.tensor(True), "test.ok", "never raised")
    env.require(torch.tensor(False), "test.bad", "custom deferred check failed")
    with pytest.raises(ValueError, match="custom deferred check failed"):
        env.step(torch.zeros((NUM_ENVS, 1)))
    # The failed flag was cleared by the raise; the next boundary is clean.
    env.step(torch.zeros((NUM_ENVS, 1)))


def test_transaction_mask_mismatch_is_deferred_not_read() -> None:
    env, _ = _make_env("cpu", masked_autoreset=True)
    env.reset()
    transaction = env._reset_state
    active = torch.tensor([True] + [False] * (NUM_ENVS - 1))
    other = torch.tensor([False, True] + [False] * (NUM_ENVS - 2))
    assert transaction._same_mask(active, other, "mask mismatch")
    assert not transaction._mask_is_empty(torch.zeros(NUM_ENVS, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask mismatch"):
        env.poll_finite_validation(force=True)


def test_host_transitions_run_eagerly_outside_capture() -> None:
    env, _ = _make_env("cpu", masked_autoreset=True)
    env.reset()
    seen = []
    env.record_host_transition("probe", lambda: seen.append(1))
    assert seen == [1]


# ---------------------------------------------------------------------------
# CUDA graph capture of the autoreset segment
# ---------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(not CUDA, reason="requires CUDA")


@cuda_only
def test_autoreset_graph_replay_matches_eager_masked_autoreset() -> None:
    actions = _actions("cuda", 30)
    eager_env, eager_backend = _make_env(
        "cuda", masked_autoreset=True, persistent_step_buffers=True
    )
    graph_env, graph_backend = _make_env(
        "cuda",
        masked_autoreset=True,
        persistent_step_buffers=True,
        autoreset_cuda_graph=True,
        autoreset_graph_warmup=2,
    )
    eager, eager_logs = _run(eager_env, eager_backend, actions)
    graphed, graph_logs = _run(graph_env, graph_backend, actions)
    graph = graph_env.autoreset_graph
    assert graph is not None and graph.captured
    assert graph.replays == len(actions) - 2
    assert graph_backend.capture_scopes == 1
    _assert_same(eager, graphed)
    assert eager_logs == graph_logs
    names = graph.host_transition_names
    assert "env.final_observation" in names
    assert "env.observation.invalidate_cache" in names
    assert "env.metrics.pending_episode_dirty" in names
    assert any(bool((snap["terminated"] | snap["truncated"]).any()) for snap in graphed[3:])


@cuda_only
def test_graphed_masked_step_never_synchronizes_inside_the_autoreset() -> None:
    env, _ = _make_env(
        "cuda",
        masked_autoreset=True,
        persistent_step_buffers=True,
        autoreset_cuda_graph=True,
        autoreset_graph_warmup=1,
    )
    env.reset()
    for action in _actions("cuda", 3):
        env.step(action)
    graph = env.autoreset_graph
    assert graph is not None and graph.captured
    torch.cuda.set_sync_debug_mode("error")
    try:
        env._run_masked_autoreset()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@cuda_only
def test_eager_masked_autoreset_body_never_synchronizes() -> None:
    env, _ = _make_env("cuda", masked_autoreset=True, persistent_step_buffers=True)
    env.reset()
    for action in _actions("cuda", 3):
        env.step(action)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        env._masked_autoreset_body()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@cuda_only
def test_captured_reset_rejects_host_published_log_entries() -> None:
    env, _ = _make_env(
        "cuda",
        masked_autoreset=True,
        persistent_step_buffers=True,
        autoreset_cuda_graph=True,
        autoreset_graph_warmup=1,
    )
    env.reset()
    original = env.metrics_manager.reset

    def logging_reset(env_ids=None):
        original(env_ids)
        return {"Episode_Metrics/host": 1.0}

    env.metrics_manager.reset = logging_reset  # type: ignore[method-assign]
    env.step(_actions("cuda", 1)[0])
    with pytest.raises(GraphCaptureError, match="host log entries"):
        env.step(_actions("cuda", 1)[0])


def test_autoreset_graph_requires_cuda_and_backend_capability() -> None:
    with pytest.raises(ValueError, match="CUDA env device"):
        _make_env(
            "cpu",
            masked_autoreset=True,
            persistent_step_buffers=True,
            autoreset_cuda_graph=True,
        )
