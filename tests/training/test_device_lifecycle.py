"""Real-CUDA lifecycle tests for the managed G1 ``mjwarp`` runtime.

The host managed runtime already freezes terminal/final/next-observation
semantics.  This suite applies the same done/timeout schedule to the production
device runtime.  Dynamic reset membership remains a CUDA bool mask; D2H copies
exist only in the explicit oracle helper after waiting for the transition
completion event.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tests.manager.test_g1_reference_differential import _cfg

from unilab.base.backend import (
    BackendBatchCounters,
    BackendCompletionEvent,
    BackendResetResult,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    BufferView,
    DeviceBufferContractError,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceResetMutationBatch,
    DeviceTensorView,
    MutationValueBatch,
    RowSelection,
    SimulationStateMutationBatch,
    StateBatchPhase,
    TypedBackendMutationBatch,
    create_backend,
    env_backend_kwargs,
)
from unilab.base.backend.base import SimBackend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.envs.locomotion.g1.managed_device import create_g1_managed_device_runtime
from unilab.envs.locomotion.g1.managed_reference import (
    create_g1_managed_reference_runtime,
)
from unilab.manager import (
    DeviceManagedRuntime,
    DeviceManagedRuntimeError,
    DeviceTransition,
    ManagedLifecyclePhase,
    ManagedReferenceRuntime,
)

pytestmark = pytest.mark.slow

_OBSERVATION_KEYS = ("obs", "critic")
_STEP_TRACE = (
    ManagedLifecyclePhase.ACTION,
    ManagedLifecyclePhase.PRE_PHYSICS,
    ManagedLifecyclePhase.PHYSICS,
    ManagedLifecyclePhase.TERMINATION,
    ManagedLifecyclePhase.REWARD,
    ManagedLifecyclePhase.METRIC,
    ManagedLifecyclePhase.TERMINAL_OBSERVATION,
    ManagedLifecyclePhase.TIMEOUT,
    ManagedLifecyclePhase.FINAL_OBSERVATION,
    ManagedLifecyclePhase.AUTORESET,
    ManagedLifecyclePhase.RESET_REQUEST,
    ManagedLifecyclePhase.RESET_BACKEND,
    ManagedLifecyclePhase.TASK_STATE_RESET,
    ManagedLifecyclePhase.OBSERVATION,
    ManagedLifecyclePhase.COMPLETE,
)


@dataclass(frozen=True)
class _RuntimeFixture:
    backend: SimBackend
    runtime: DeviceManagedRuntime
    placement: BufferPlacement
    device: torch.device


@dataclass(frozen=True)
class _TransitionSnapshot:
    observations: dict[str, torch.Tensor]
    terminal_observations: dict[str, torch.Tensor]
    final_observations: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_observation_mask: torch.Tensor


@dataclass(frozen=True)
class _HostDeviceFixture:
    host_backend: SimBackend
    host_runtime: ManagedReferenceRuntime
    device: _RuntimeFixture


def _require_cuda() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("device managed lifecycle requires an active CUDA Warp device")


@contextmanager
def _runtime_fixture(
    *, num_envs: int, seed: int, max_episode_steps: int | None
) -> Iterator[_RuntimeFixture]:
    _require_cuda()
    cfg = _cfg(
        max_episode_seconds=None,
        observation_noise_level=0.0,
        observation_noise_seed=None,
    )
    assert cfg.scene is not None
    backend = create_backend(
        "mjwarp",
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
        push_body_name=cfg.domain_rand.push_body_name,
        **env_backend_kwargs(cfg),
    )
    try:
        runtime = create_g1_managed_device_runtime(
            backend=backend,
            cfg=cfg,
            reset_seed=seed,
            max_episode_steps=max_episode_steps,
            record_lifecycle=True,
        )
        placement = runtime.bound_plan.control.buffer.placement
        assert placement.device_index is not None
        yield _RuntimeFixture(
            backend=backend,
            runtime=runtime,
            placement=placement,
            device=torch.device(f"cuda:{placement.device_index}"),
        )
    finally:
        backend.cleanup_scene_assets()


@contextmanager
def _host_device_fixture(*, num_envs: int, seed: int) -> Iterator[_HostDeviceFixture]:
    """Build independent host-reference and device runtimes for a numeric oracle."""

    _require_cuda()
    cfg = _cfg(
        max_episode_seconds=None,
        observation_noise_level=0.0,
        observation_noise_seed=None,
    )
    assert cfg.reward_config is not None
    cfg.reward_config.scales.pop("feet_phase")
    cfg.reward_config.gait_frequency = 0.0
    cfg.reset_base_qvel_limit = 0.0
    cfg.commands.vel_limit = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert cfg.scene is not None

    host_cfg = deepcopy(cfg)
    device_cfg = deepcopy(cfg)
    host_backend = create_backend(
        "mujoco",
        host_cfg.scene,
        num_envs,
        host_cfg.sim_dt,
        base_name=host_cfg.asset.base_name,
        push_body_name=host_cfg.domain_rand.push_body_name,
        **env_backend_kwargs(host_cfg),
    )
    device_backend = create_backend(
        "mjwarp",
        device_cfg.scene,
        num_envs,
        device_cfg.sim_dt,
        base_name=device_cfg.asset.base_name,
        push_body_name=device_cfg.domain_rand.push_body_name,
        **env_backend_kwargs(device_cfg),
    )
    try:
        host_runtime = create_g1_managed_reference_runtime(
            backend=host_backend,
            cfg=host_cfg,
            reset_seed=seed,
            record_lifecycle=True,
        )
        device_runtime = create_g1_managed_device_runtime(
            backend=device_backend,
            cfg=device_cfg,
            reset_seed=seed,
            max_episode_steps=100,
            record_lifecycle=True,
        )
        placement = device_runtime.bound_plan.control.buffer.placement
        assert placement.device_index is not None
        yield _HostDeviceFixture(
            host_backend=host_backend,
            host_runtime=host_runtime,
            device=_RuntimeFixture(
                backend=device_backend,
                runtime=device_runtime,
                placement=placement,
                device=torch.device(f"cuda:{placement.device_index}"),
            ),
        )
    finally:
        host_backend.cleanup_scene_assets()
        device_backend.cleanup_scene_assets()


def _mask_contract(placement: BufferPlacement) -> BufferContract:
    return BufferContract(
        row_shape=(),
        dtype="bool",
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.MANAGER,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.UNTIL_COMMIT,
        dlpack_exportable=True,
        address_stable=True,
    )


def _action_view(
    fixture: _RuntimeFixture,
    *,
    producer_stream: torch.cuda.Stream,
    after: DeviceCompletion,
    value: float = 0.0,
    owner: str = "device-lifecycle-policy",
) -> DeviceTensorView:
    contract = fixture.runtime.bound_plan.control.buffer
    lease = DeviceBufferLease(owner)
    with torch.cuda.stream(producer_stream):
        after.wait(producer_stream)
        actions = torch.full(
            (fixture.backend.num_envs, *contract.row_shape),
            value,
            dtype=torch.float32,
            device=fixture.device,
        )
        completion = DeviceCompletion.record(
            placement=fixture.placement,
            owner_id=lease.owner_id,
            epoch=lease.epoch,
            stream=producer_stream,
        )
    return DeviceTensorView(
        tensor_handle=actions,
        contract=contract,
        lease=lease,
        completion=completion,
    )


def _snapshot(
    transition: DeviceTransition, *, consumer_stream: torch.cuda.Stream
) -> _TransitionSnapshot:
    device_values: dict[str, torch.Tensor] = {}
    with torch.cuda.stream(consumer_stream):
        transition.completion.wait(consumer_stream)
        for key in _OBSERVATION_KEYS:
            device_values[f"observation.{key}"] = transition.observation(key).torch().clone()
            device_values[f"terminal.{key}"] = transition.terminal_observation(key).torch().clone()
            device_values[f"final.{key}"] = transition.final_observation(key).torch().clone()
        device_values["reward"] = transition.reward.torch().clone()
        device_values["terminated"] = transition.terminated.torch().clone()
        device_values["truncated"] = transition.truncated.torch().clone()
        device_values["final_mask"] = transition.final_observation_mask.torch().clone()
    consumer_stream.synchronize()

    host = {key: value.cpu() for key, value in device_values.items()}
    return _TransitionSnapshot(
        observations={key: host[f"observation.{key}"] for key in _OBSERVATION_KEYS},
        terminal_observations={key: host[f"terminal.{key}"] for key in _OBSERVATION_KEYS},
        final_observations={key: host[f"final.{key}"] for key in _OBSERVATION_KEYS},
        reward=host["reward"],
        terminated=host["terminated"],
        truncated=host["truncated"],
        final_observation_mask=host["final_mask"],
    )


def _assert_device_counters(counters: BackendBatchCounters) -> None:
    assert counters.instrumentation_complete
    assert counters.host_to_device_transfers == 0
    assert counters.device_to_host_transfers == 0
    assert counters.host_to_device_bytes == 0
    assert counters.device_to_host_bytes == 0
    assert counters.global_synchronizations == 0
    assert counters.allocations == 0
    assert counters.dynamic_getter_calls == 0
    assert counters.selector_resolutions == 0
    assert counters.asset_metadata_reads == 0
    assert counters.registry_lookups == 0
    assert counters.state_materializations == 1


def _assert_transition_diagnostics(transition: DeviceTransition) -> None:
    assert transition.reset_diagnostics is not None
    _assert_device_counters(transition.reset_diagnostics.counters)
    if transition.step_diagnostics is not None:
        _assert_device_counters(transition.step_diagnostics.counters)


@contextmanager
def _forbid_hot_path_fallbacks(backend: SimBackend) -> Iterator[None]:
    with ExitStack() as stack:
        for method in (
            "get_actuator_names",
            "get_body_ids",
            "get_joint_dof_pos_indices",
            "get_joint_dof_vel_indices",
            "get_sensor_ids",
            "get_keyframe_qpos",
            "get_init_qvel",
            "get_base_pos",
            "get_base_quat",
            "get_base_lin_vel",
            "get_base_ang_vel",
            "get_dof_pos",
            "get_dof_vel",
            "get_sensor_data",
            "get_scene_model_file",
            "set_state",
        ):
            stack.enter_context(
                patch.object(backend, method, side_effect=AssertionError(f"hot-path {method}"))
            )
        stack.enter_context(
            patch.object(
                backend,
                "_resolve_mjwarp_typed_mutation_selector",
                side_effect=AssertionError("hot-path selector"),
            )
        )
        stack.enter_context(
            patch("torch.cuda.synchronize", side_effect=AssertionError("global sync is forbidden"))
        )
        for method in ("cpu", "numpy", "item", "tolist"):
            stack.enter_context(
                patch.object(
                    torch.Tensor,
                    method,
                    side_effect=AssertionError(f"host tensor extraction: {method}"),
                )
            )
        stack.enter_context(
            patch("torch.nonzero", side_effect=AssertionError("host done-index extraction"))
        )
        stack.enter_context(
            patch.object(Path, "read_text", side_effect=AssertionError("asset read"))
        )
        stack.enter_context(
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset read"))
        )
        yield


def _state_tensor(state: Any, key: str) -> DeviceTensorView:
    value = state.buffer(key).handle
    assert isinstance(value, DeviceTensorView)
    return value


def _force_root_state(
    fixture: _RuntimeFixture,
    *,
    rows: tuple[int, ...],
    height: float | None,
    producer_stream: torch.cuda.Stream,
    after: DeviceCompletion,
    orientation: tuple[float, float, float, float] | None = None,
) -> DeviceCompletion:
    """Inject a known terminal state through the public typed reset contract."""

    num_envs = fixture.backend.num_envs
    all_rows = RowSelection.all(num_envs)
    mutation_plan = fixture.runtime.kernel_binding.mutation_plan
    assert mutation_plan is not None
    read_result = fixture.backend.read_state_batch(
        fixture.runtime.bound_plan,
        all_rows,
        phase=StateBatchPhase.CURRENT,
    )
    state = read_result.state
    sources = {
        target: _state_tensor(state, field)
        for target, field in (
            ("state.root.position", "g1.root.position"),
            ("state.root.orientation", "g1.root.orientation"),
            ("state.root.linear_velocity", "g1.root.linear_velocity"),
            ("state.root.angular_velocity", "g1.root.angular_velocity"),
            ("state.dof.position", "g1.dof.position"),
            ("state.dof.angular_velocity", "g1.dof.angular_velocity"),
        )
    }
    state_ids = {
        field.key: field.identity.entity_ids for field in fixture.runtime.bound_plan.state.fields
    }
    dof_columns = {
        "state.dof.position": {
            entity_id: index for index, entity_id in enumerate(state_ids["g1.dof.position"])
        },
        "state.dof.angular_velocity": {
            entity_id: index for index, entity_id in enumerate(state_ids["g1.dof.angular_velocity"])
        },
    }

    lease = DeviceBufferLease("device-lifecycle-reset-injection")
    mask_contract = _mask_contract(fixture.placement)
    with torch.cuda.stream(producer_stream):
        after.wait(producer_stream)
        for source in sources.values():
            source.wait(producer_stream)
        active_mask = torch.zeros((num_envs,), dtype=torch.bool, device=fixture.device)
        active_mask[list(rows)] = True
        values: list[torch.Tensor] = []
        root_position: torch.Tensor | None = None
        root_orientation: torch.Tensor | None = None
        for spec in mutation_plan.specs:
            target_key = spec.target.target_key
            source = sources[target_key].torch()
            value = torch.empty(
                (num_envs, *spec.value_buffer.row_shape),
                dtype=torch.float32,
                device=fixture.device,
            )
            if target_key.startswith("state.root."):
                value[:, 0, :].copy_(source, non_blocking=True)
                if target_key == "state.root.position":
                    root_position = value
                elif target_key == "state.root.orientation":
                    root_orientation = value
            else:
                entity_id = spec.target.entity_ids[0]
                column = dof_columns[target_key][entity_id]
                value[:, 0, 0].copy_(source[:, column], non_blocking=True)
            values.append(value)
        assert root_position is not None
        assert root_orientation is not None
        if height is not None:
            root_position[list(rows), 0, 2] = height
        if orientation is not None:
            root_orientation[list(rows), 0, :] = torch.tensor(
                orientation,
                dtype=torch.float32,
                device=fixture.device,
            )
        completion = DeviceCompletion.record(
            placement=fixture.placement,
            owner_id=lease.owner_id,
            epoch=lease.epoch,
            stream=producer_stream,
        )

    mask_view = DeviceTensorView(
        tensor_handle=active_mask,
        contract=mask_contract,
        lease=lease,
        completion=completion,
    )
    entries = tuple(
        MutationValueBatch(
            plan=mutation_plan,
            field_index=index,
            rows=all_rows,
            buffer=BufferView(
                handle=DeviceTensorView(
                    tensor_handle=value,
                    contract=spec.value_buffer,
                    lease=lease,
                    completion=completion,
                ),
                shape=tuple(int(dim) for dim in value.shape),
                contract=spec.value_buffer,
            ),
        )
        for index, (spec, value) in enumerate(zip(mutation_plan.specs, values, strict=True))
    )
    mutation = TypedBackendMutationBatch(
        plan=mutation_plan,
        rows=all_rows,
        state=SimulationStateMutationBatch(entries),
    )
    reset = DeviceResetMutationBatch(
        plan=mutation_plan,
        rows=all_rows,
        mutation=mutation,
        active_mask=BufferView(
            handle=mask_view,
            shape=(num_envs,),
            contract=mask_contract,
        ),
    )
    result = fixture.backend.reset_batch(
        fixture.runtime.bound_plan,
        all_rows,
        mutation_batch=reset,
    )
    completion_event = result.diagnostics.completion_event
    assert completion_event is not None
    handle = completion_event.handle
    assert isinstance(handle, DeviceCompletion)
    return handle


def _assert_final_contract(snapshot: _TransitionSnapshot, expected_done: torch.Tensor) -> None:
    torch.testing.assert_close(snapshot.final_observation_mask, expected_done, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.logical_or(snapshot.terminated, snapshot.truncated),
        expected_done,
        rtol=0,
        atol=0,
    )
    for key in _OBSERVATION_KEYS:
        torch.testing.assert_close(
            snapshot.final_observations[key][expected_done],
            snapshot.terminal_observations[key][expected_done],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            snapshot.observations[key][~expected_done],
            snapshot.terminal_observations[key][~expected_done],
            rtol=1.0e-5,
            atol=1.0e-6,
        )


def _align_host_physics_from_device(
    fixture: _HostDeviceFixture,
    *,
    consumer_stream: torch.cuda.Stream,
) -> DeviceCompletion:
    """Test-only D2H oracle that aligns host qpos/qvel to public device state."""

    device = fixture.device
    result = device.backend.read_state_batch(
        device.runtime.bound_plan,
        RowSelection.all(device.backend.num_envs),
        phase=StateBatchPhase.CURRENT,
    )
    state = result.state
    tensors: dict[str, torch.Tensor] = {}
    with torch.cuda.stream(consumer_stream):
        for key in (
            "g1.root.position",
            "g1.root.orientation",
            "g1.root.linear_velocity",
            "g1.root.angular_velocity",
            "g1.dof.position",
            "g1.dof.angular_velocity",
        ):
            view = _state_tensor(state, key)
            view.wait(consumer_stream)
            tensors[key] = view.torch().clone()
    consumer_stream.synchronize()
    qpos = (
        torch.cat(
            (
                tensors["g1.root.position"],
                tensors["g1.root.orientation"],
                tensors["g1.dof.position"],
            ),
            dim=1,
        )
        .cpu()
        .numpy()
    )
    qvel = (
        torch.cat(
            (
                tensors["g1.root.linear_velocity"],
                tensors["g1.root.angular_velocity"],
                tensors["g1.dof.angular_velocity"],
            ),
            dim=1,
        )
        .cpu()
        .numpy()
    )
    rows = np.arange(device.backend.num_envs, dtype=np.int32)
    fixture.host_backend.set_state(rows, qpos, qvel)
    event = result.diagnostics.completion_event
    assert event is not None and isinstance(event.handle, DeviceCompletion)
    return event.handle


def _without_rng_gait_columns(key: str, value: np.ndarray) -> np.ndarray:
    if key == "obs":
        return value[:, :96]
    return np.concatenate((value[:, :96], value[:, 98:]), axis=1)


def _assert_host_device_transition(
    *,
    host: Any,
    device: _TransitionSnapshot,
    label: str,
) -> None:
    host_mask = np.asarray(host.info["_final_observation"], dtype=bool)
    np.testing.assert_array_equal(device.final_observation_mask.numpy(), host_mask)
    np.testing.assert_array_equal(device.terminated.numpy(), host.terminated)
    np.testing.assert_array_equal(device.truncated.numpy(), host.truncated)
    np.testing.assert_allclose(
        device.reward.numpy(),
        host.reward,
        atol=1.0e-4,
        rtol=1.0e-3,
        err_msg=f"{label}.reward",
    )
    for key in _OBSERVATION_KEYS:
        host_terminal = np.array(host.obs[key], copy=True)
        if host.final_observation is not None:
            host_terminal[host_mask] = host.final_observation[key][host_mask]
        np.testing.assert_allclose(
            _without_rng_gait_columns(key, device.terminal_observations[key].numpy()),
            _without_rng_gait_columns(key, host_terminal),
            atol=1.0e-4,
            rtol=1.0e-3,
            err_msg=f"{label}.terminal[{key}]",
        )
        np.testing.assert_allclose(
            _without_rng_gait_columns(key, device.observations[key].numpy()),
            _without_rng_gait_columns(key, np.asarray(host.obs[key])),
            atol=1.0e-4,
            rtol=1.0e-3,
            err_msg=f"{label}.next[{key}]",
        )
        if np.any(host_mask):
            assert host.final_observation is not None
            np.testing.assert_allclose(
                _without_rng_gait_columns(
                    key,
                    device.final_observations[key][torch.from_numpy(host_mask)].numpy(),
                ),
                _without_rng_gait_columns(key, host.final_observation[key][host_mask]),
                atol=1.0e-4,
                rtol=1.0e-3,
                err_msg=f"{label}.final[{key}]",
            )


@pytest.mark.parametrize(("seed", "num_envs"), ((0, 32), (1, 128), (2, 32)))
def test_device_adapter_matches_host_terminal_contract(seed: int, num_envs: int) -> None:
    """Freeze host terminal/final/next semantics on the production device path."""

    with _runtime_fixture(
        num_envs=num_envs,
        seed=seed,
        max_episode_steps=100,
    ) as fixture:
        producer = cast(torch.cuda.Stream, torch.cuda.Stream(device=fixture.device))
        consumer = cast(torch.cuda.Stream, torch.cuda.Stream(device=fixture.device))

        with _forbid_hot_path_fallbacks(fixture.backend):
            initial = fixture.runtime.reset()
        initial_snapshot = _snapshot(initial, consumer_stream=consumer)
        assert initial.trace == (
            ManagedLifecyclePhase.INITIAL_RESET_REQUEST,
            ManagedLifecyclePhase.RESET_BACKEND,
            ManagedLifecyclePhase.TASK_STATE_RESET,
            ManagedLifecyclePhase.OBSERVATION,
            ManagedLifecyclePhase.COMPLETE,
        )
        assert not initial_snapshot.terminated.any()
        assert not initial_snapshot.truncated.any()
        assert not initial_snapshot.final_observation_mask.any()
        assert torch.count_nonzero(initial_snapshot.reward) == 0
        _assert_transition_diagnostics(initial)
        stale_initial = initial.observation("obs")

        no_done_action = _action_view(
            fixture,
            producer_stream=producer,
            after=initial.completion,
        )
        with _forbid_hot_path_fallbacks(fixture.backend):
            no_done = fixture.runtime.step(no_done_action)
        no_done_snapshot = _snapshot(no_done, consumer_stream=consumer)
        expected_none = torch.zeros((num_envs,), dtype=torch.bool)
        _assert_final_contract(no_done_snapshot, expected_none)
        assert no_done.trace == _STEP_TRACE
        assert torch.isfinite(no_done_snapshot.reward).all()
        _assert_transition_diagnostics(no_done)
        with pytest.raises(DeviceBufferContractError, match="stale"):
            stale_initial.torch()

        partial_rows = (num_envs - 1,)
        forced = _force_root_state(
            fixture,
            rows=partial_rows,
            height=0.1,
            producer_stream=producer,
            after=no_done.completion,
        )
        partial_action = _action_view(
            fixture,
            producer_stream=producer,
            after=forced,
        )
        with _forbid_hot_path_fallbacks(fixture.backend):
            partial = fixture.runtime.step(partial_action)
        partial_snapshot = _snapshot(partial, consumer_stream=consumer)
        expected_partial = torch.zeros((num_envs,), dtype=torch.bool)
        expected_partial[-1] = True
        torch.testing.assert_close(
            partial_snapshot.terminated,
            expected_partial,
            rtol=0,
            atol=0,
        )
        assert not partial_snapshot.truncated.any()
        _assert_final_contract(partial_snapshot, expected_partial)
        assert partial.trace == _STEP_TRACE
        _assert_transition_diagnostics(partial)
        stale_partial = partial.terminal_observation("critic")

        forced_all = _force_root_state(
            fixture,
            rows=tuple(range(num_envs)),
            height=0.1,
            producer_stream=producer,
            after=partial.completion,
        )
        all_done_action = _action_view(
            fixture,
            producer_stream=producer,
            after=forced_all,
        )
        with _forbid_hot_path_fallbacks(fixture.backend):
            all_done = fixture.runtime.step(all_done_action)
        all_done_snapshot = _snapshot(all_done, consumer_stream=consumer)
        expected_all = torch.ones((num_envs,), dtype=torch.bool)
        torch.testing.assert_close(all_done_snapshot.terminated, expected_all, rtol=0, atol=0)
        assert not all_done_snapshot.truncated.any()
        _assert_final_contract(all_done_snapshot, expected_all)
        assert all_done.trace == _STEP_TRACE
        _assert_transition_diagnostics(all_done)
        with pytest.raises(DeviceBufferContractError, match="stale"):
            stale_partial.torch()


def test_device_g1_transition_matches_verified_host_runtime_on_aligned_state() -> None:
    """Compare independent host/device task math after public state alignment."""

    with _host_device_fixture(num_envs=8, seed=19) as fixture:
        device = fixture.device
        producer = cast(torch.cuda.Stream, torch.cuda.Stream(device=device.device))
        consumer = cast(torch.cuda.Stream, torch.cuda.Stream(device=device.device))
        fixture.host_runtime.init_state()
        device.runtime.reset()

        after = _align_host_physics_from_device(fixture, consumer_stream=consumer)
        action = _action_view(
            device,
            producer_stream=producer,
            after=after,
            value=0.015,
            owner="device-host-parity-step-0",
        )
        host_step = fixture.host_runtime.step(np.full((8, 29), 0.015, dtype=np.float32))
        with _forbid_hot_path_fallbacks(device.backend):
            device_step = device.runtime.step(action)
        snapshot = _snapshot(device_step, consumer_stream=consumer)
        _assert_host_device_transition(host=host_step, device=snapshot, label="no_done")

        forced = _force_root_state(
            device,
            rows=(7,),
            height=None,
            producer_stream=producer,
            after=device_step.completion,
            orientation=(0.95371695, 0.3007058, 0.0, 0.0),
        )
        forced.wait(consumer)
        after = _align_host_physics_from_device(fixture, consumer_stream=consumer)
        action = _action_view(
            device,
            producer_stream=producer,
            after=after,
            value=-0.01,
            owner="device-host-parity-step-1",
        )
        host_step = fixture.host_runtime.step(np.full((8, 29), -0.01, dtype=np.float32))
        with _forbid_hot_path_fallbacks(device.backend):
            device_step = device.runtime.step(action)
        snapshot = _snapshot(device_step, consumer_stream=consumer)
        expected_partial = np.zeros((8,), dtype=bool)
        expected_partial[-1] = True
        np.testing.assert_array_equal(host_step.info["_final_observation"], expected_partial)
        _assert_host_device_transition(host=host_step, device=snapshot, label="partial_reset")


def test_device_timeout_captures_terminal_before_all_world_reset() -> None:
    with _runtime_fixture(num_envs=32, seed=11, max_episode_steps=1) as fixture:
        producer = cast(torch.cuda.Stream, torch.cuda.Stream(device=fixture.device))
        consumer = cast(torch.cuda.Stream, torch.cuda.Stream(device=fixture.device))
        initial = fixture.runtime.reset()
        action = _action_view(
            fixture,
            producer_stream=producer,
            after=initial.completion,
            value=0.01,
        )
        transition = fixture.runtime.step(action)
        snapshot = _snapshot(transition, consumer_stream=consumer)

        expected = torch.ones((fixture.backend.num_envs,), dtype=torch.bool)
        assert not snapshot.terminated.any()
        torch.testing.assert_close(snapshot.truncated, expected, rtol=0, atol=0)
        _assert_final_contract(snapshot, expected)
        assert transition.trace == _STEP_TRACE
        _assert_transition_diagnostics(transition)


def test_device_runtime_rejects_raw_cpu_missing_event_and_replayed_action() -> None:
    with _runtime_fixture(num_envs=4, seed=13, max_episode_steps=100) as fixture:
        initial = fixture.runtime.reset()
        contract = fixture.runtime.bound_plan.control.buffer

        with pytest.raises(DeviceBufferContractError, match="DeviceTensorView"):
            fixture.runtime.step(cast(DeviceTensorView, torch.zeros((4, 29))))

        missing_lease = DeviceBufferLease("device-lifecycle-missing-event")
        missing = DeviceTensorView(
            tensor_handle=torch.zeros((4, 29), dtype=torch.float32, device=fixture.device),
            contract=contract,
            lease=missing_lease,
        )
        with pytest.raises(DeviceBufferContractError, match="no producer completion"):
            fixture.runtime.step(missing)

        producer = cast(torch.cuda.Stream, torch.cuda.Stream(device=fixture.device))
        valid = _action_view(
            fixture,
            producer_stream=producer,
            after=initial.completion,
            owner="device-lifecycle-replay",
        )
        fixture.runtime.step(valid)
        with pytest.raises(DeviceBufferContractError, match="already consumed"):
            fixture.runtime.step(valid)


@pytest.mark.parametrize("corruption", ("foreign_owner", "mismatched_event"))
def test_device_runtime_rejects_untrusted_backend_completion(corruption: str) -> None:
    """A diagnostics event cannot authenticate a different state transaction."""

    with _runtime_fixture(num_envs=4, seed=17, max_episode_steps=100) as fixture:
        original_reset = fixture.backend.reset_batch

        def corrupt_reset(*args: Any, **kwargs: Any) -> BackendResetResult:
            result = original_reset(*args, **kwargs)
            state_view = result.reset_state.buffer_at(0).handle
            assert isinstance(state_view, DeviceTensorView)
            state_completion = state_view.require_completion()
            owner_id = state_completion.owner_id
            event = state_completion.event
            if corruption == "foreign_owner":
                owner_id = "foreign-backend-instance"
            else:
                event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False))
                event.record(torch.cuda.current_stream(fixture.device))
            forged = DeviceCompletion(
                placement=state_completion.placement,
                owner_id=owner_id,
                epoch=state_completion.epoch,
                event=event,
            )
            diagnostics = replace(
                result.diagnostics,
                completion_event=BackendCompletionEvent(
                    backend_type=fixture.runtime.bound_plan.backend_type,
                    placement=fixture.placement,
                    handle=forged,
                ),
            )
            return BackendResetResult(
                reset_state=result.reset_state,
                diagnostics=diagnostics,
            )

        expected = "completion owner" if corruption == "foreign_owner" else "completion differs"
        with patch.object(fixture.backend, "reset_batch", side_effect=corrupt_reset):
            with pytest.raises(DeviceManagedRuntimeError, match=expected):
                fixture.runtime.reset()
