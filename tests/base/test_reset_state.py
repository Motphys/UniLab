"""Focused tests for the base-owned Manager-Based reset transaction."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from unisim.backend.base import BackendRootStateLayout, SimBackend
from unisim.dr.types import (
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    ResetRandomizationPayload,
)

from unilab.base.reset_state import ResetStateTransaction


class _Backend:
    backend_type = "fake"
    num_actuators = 3

    def __init__(
        self,
        *,
        qpos: Any = None,
        qvel: Any = None,
        fail_set_state: bool = False,
    ) -> None:
        self.num_envs = 4
        self.qpos = np.array([1.0, 2.0, 3.0]) if qpos is None else qpos
        self.qvel = np.array([0.0, 0.0]) if qvel is None else qvel
        self.fail_set_state = fail_set_state
        self.default_qpos_calls = 0
        self.init_qvel_calls = 0
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.randomization_calls: list[ResetRandomizationPayload | None] = []
        self.default_kp = np.array([10.0, 20.0, 30.0])
        self.default_kd = np.array([1.0, 2.0, 3.0])

    def get_default_qpos(self):
        self.default_qpos_calls += 1
        return self.qpos

    def get_init_qvel(self):
        self.init_qvel_calls += 1
        return self.qvel

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset((RESET_TERM_KP, RESET_TERM_KD))
        )

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return self.default_kp.copy(), self.default_kd.copy()

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> dict:
        if self.fail_set_state:
            raise NotImplementedError("reset upload disabled")
        self.set_state_calls.append((env_ids.copy(), qpos.copy(), qvel.copy()))
        self.randomization_calls.append(randomization)
        return {"timing": {"set_state_ms": 1.0}}


def _transaction(backend: _Backend) -> ResetStateTransaction:
    return ResetStateTransaction(cast(SimBackend, backend))


def test_transaction_is_lazy_and_combines_terms_into_one_commit() -> None:
    backend = _Backend()
    transaction = _transaction(backend)

    with transaction.scoped(np.array([0, 2, 3], dtype=np.int32)):
        assert transaction.active
    assert backend.default_qpos_calls == 0
    assert backend.init_qvel_calls == 0
    assert backend.set_state_calls == []

    with transaction.scoped(np.array([0, 2, 3], dtype=np.int32)):
        transaction.reset_to_default(
            np.array([2], dtype=np.int32),
            term_name="first",
        )
        transaction.reset_to_default(
            np.array([3, 0], dtype=np.int32),
            term_name="second",
        )
        assert backend.set_state_calls == []

    assert not transaction.active
    assert backend.default_qpos_calls == 1
    assert backend.init_qvel_calls == 1
    assert len(backend.set_state_calls) == 1
    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [0, 2, 3])
    np.testing.assert_array_equal(qpos, np.tile(backend.qpos, (3, 1)))
    np.testing.assert_array_equal(qvel, np.tile(backend.qvel, (3, 1)))

    with transaction.scoped(np.array([1], dtype=np.int32)):
        transaction.reset_to_default(np.array([1], dtype=np.int32), term_name="third")
    assert backend.default_qpos_calls == 1
    assert backend.init_qvel_calls == 1
    assert len(backend.set_state_calls) == 2


def test_transaction_reports_only_committed_dirty_rows_as_writes() -> None:
    backend = _Backend()
    transaction = _transaction(backend)

    assert not transaction.last_commit_had_writes
    with transaction.scoped(np.array([0], dtype=np.int32)):
        pass
    assert not transaction.last_commit_had_writes

    with transaction.scoped(np.array([0], dtype=np.int32)):
        transaction.reset_to_default(np.array([0], dtype=np.int32), term_name="dirty")
    assert transaction.last_commit_had_writes

    with pytest.raises(RuntimeError, match="abort"):
        with transaction.scoped(np.array([1], dtype=np.int32)):
            transaction.reset_to_default(np.array([1], dtype=np.int32), term_name="aborted")
            raise RuntimeError("abort")
    assert not transaction.last_commit_had_writes


def test_exception_aborts_without_backend_mutation_and_next_reset_is_clean() -> None:
    backend = _Backend()
    transaction = _transaction(backend)

    with pytest.raises(RuntimeError, match="term failed"):
        with transaction.scoped(np.array([0, 1], dtype=np.int32)):
            transaction.reset_to_default(np.array([0], dtype=np.int32), term_name="broken")
            raise RuntimeError("term failed")

    assert not transaction.active
    assert backend.set_state_calls == []

    with transaction.scoped(np.array([1], dtype=np.int32)):
        transaction.reset_to_default(np.array([1], dtype=np.int32), term_name="healthy")
    assert len(backend.set_state_calls) == 1
    np.testing.assert_array_equal(backend.set_state_calls[0][0], [1])


def test_actuator_gains_compose_with_state_in_one_reset_commit() -> None:
    backend = _Backend()
    transaction = _transaction(backend)
    columns, default_kp, default_kd = transaction.bind_actuator_gain_write(
        np.array([2, 0], dtype=np.int32),
        term_name="pd_gains:robot",
    )
    np.testing.assert_array_equal(columns, [2, 0])
    np.testing.assert_array_equal(default_kp, [30.0, 10.0])
    np.testing.assert_array_equal(default_kd, [3.0, 1.0])

    with transaction.scoped(np.array([0, 2], dtype=np.int32)):
        transaction.reset_to_default(np.array([0, 2], dtype=np.int32), term_name="default")
        transaction.write_actuator_gains(
            np.array([2, 0], dtype=np.int32),
            columns,
            np.array([[5.0, 6.0], [7.0, 8.0]]),
            np.array([[0.5, 0.6], [0.7, 0.8]]),
            term_name="pd_gains:robot",
        )

    payload = backend.randomization_calls[0]
    assert payload is not None
    np.testing.assert_array_equal(payload.kp, [[8.0, 20.0, 7.0], [6.0, 20.0, 5.0]])
    np.testing.assert_array_equal(payload.kd, [[0.8, 2.0, 0.7], [0.6, 2.0, 0.5]])


def test_actuator_gain_sparse_rows_abort_without_backend_mutation() -> None:
    backend = _Backend()
    transaction = _transaction(backend)
    columns, _, _ = transaction.bind_actuator_gain_write(
        np.array([0], dtype=np.int32),
        term_name="pd_gains:robot",
    )

    with pytest.raises(RuntimeError, match=r"cannot represent sparse rows.*missing env IDs \[1\]"):
        with transaction.scoped(np.array([0, 1], dtype=np.int32)):
            transaction.reset_to_default(
                np.array([0, 1], dtype=np.int32),
                term_name="default",
            )
            transaction.write_actuator_gains(
                np.array([0], dtype=np.int32),
                columns,
                np.array([[11.0]]),
                np.array([[1.1]]),
                term_name="pd_gains:robot",
            )

    assert backend.set_state_calls == []


def test_joint_writes_initialize_defaults_and_compose_by_column() -> None:
    backend = _Backend()
    transaction = _transaction(backend)

    with transaction.scoped(np.array([0, 2], dtype=np.int32)):
        transaction.write_joint_state(
            np.array([2, 0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([[9.0], [8.0]], dtype=np.float32),
            np.array([[-1.0], [-2.0]], dtype=np.float32),
            term_name="robot.write_joint_state_to_sim",
        )

    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [0, 2])
    np.testing.assert_array_equal(qpos, [[1.0, 8.0, 3.0], [1.0, 9.0, 3.0]])
    np.testing.assert_array_equal(qvel, [[-2.0, 0.0], [-1.0, 0.0]])


def test_root_pose_and_world_velocity_compose_at_nonzero_columns() -> None:
    default_qpos = np.array([99.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 88.0])
    default_qvel = np.array([77.0, 66.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 55.0])
    backend = _Backend(qpos=default_qpos, qvel=default_qvel)
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(1, 8)), tuple(range(2, 8)))
    half_sqrt = np.sqrt(0.5)
    poses = np.array(
        [
            [10.0, 11.0, 12.0, half_sqrt, 0.0, 0.0, half_sqrt],
            [20.0, 21.0, 22.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    velocities_w = np.array(
        [
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0],
            [4.0, 5.0, 6.0, 0.0, 1.0, 2.0],
        ]
    )

    with transaction.scoped(np.array([0, 2], dtype=np.int32)):
        transaction.write_root_pose(
            np.array([2, 0], dtype=np.int32),
            layout,
            poses,
            term_name="root_pose",
        )
        transaction.write_root_velocity(
            np.array([2, 0], dtype=np.int32),
            layout,
            velocities_w,
            term_name="root_velocity",
        )

    assert len(backend.set_state_calls) == 1
    ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [0, 2])
    np.testing.assert_array_equal(qpos[:, [0, 8]], [[99.0, 88.0], [99.0, 88.0]])
    np.testing.assert_allclose(qpos[0, 1:8], poses[1])
    np.testing.assert_allclose(qpos[1, 1:8], poses[0])
    np.testing.assert_array_equal(qvel[:, [0, 1, 8]], [[77.0, 66.0, 55.0]] * 2)
    np.testing.assert_allclose(qvel[0, 2:8], velocities_w[1])
    np.testing.assert_allclose(qvel[1, 2:5], velocities_w[0, :3])
    np.testing.assert_allclose(qvel[1, 5:8], [0.0, -1.0, 0.0], atol=1e-7)


def test_read_root_pose_returns_staged_or_default_pose_without_dirtying() -> None:
    default_qpos = np.array([99.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 88.0])
    default_qvel = np.zeros(9)
    backend = _Backend(qpos=default_qpos, qvel=default_qvel)
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(1, 8)), tuple(range(2, 8)))

    with transaction.scoped(np.array([0, 2], dtype=np.int32)):
        transaction.write_root_pose(
            np.array([2], dtype=np.int32),
            layout,
            np.array([[10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0]]),
            term_name="reset_base",
        )
        staged = transaction.read_root_pose(
            np.array([0, 2], dtype=np.int32),
            layout,
            term_name="random_prone_init",
        )
        # Env 2 reflects the earlier term's write; env 0 the backend default.
        np.testing.assert_allclose(staged[0], default_qpos[1:8])
        np.testing.assert_allclose(staged[1], [10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0])
        # The read is a detached copy: mutating it must not leak back.
        staged[1, 0] = -5.0
        reread = transaction.read_root_pose(
            np.array([2], dtype=np.int32),
            layout,
            term_name="random_prone_init",
        )
        assert reread[0, 0] == 10.0

    # Reading alone does not dirty a row: only the written env is committed.
    assert len(backend.set_state_calls) == 1
    ids, _, _ = backend.set_state_calls[0]
    np.testing.assert_array_equal(ids, [2])


def test_read_root_pose_fails_closed_outside_reset_scope() -> None:
    backend = _Backend(
        qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(6),
    )
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))

    with pytest.raises(RuntimeError, match="requires an active reset event"):
        transaction.read_root_pose(np.array([0], dtype=np.int32), layout, term_name="t")

    with transaction.scoped(np.array([0], dtype=np.int32)):
        with pytest.raises(ValueError, match="outside the active reset"):
            transaction.read_root_pose(np.array([1], dtype=np.int32), layout, term_name="t")


def test_combined_root_state_uses_staged_pose_for_angular_velocity() -> None:
    backend = _Backend(
        qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(6),
    )
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    half_sqrt = np.sqrt(0.5)
    root_state = np.array(
        [[1.0, 2.0, 3.0, half_sqrt, 0.0, 0.0, half_sqrt, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0]]
    )

    with transaction.scoped(np.array([1], dtype=np.int32)):
        transaction.write_root_state(
            np.array([1], dtype=np.int32),
            layout,
            root_state,
            term_name="root_state",
        )

    _, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_allclose(qpos[0], root_state[0, :7])
    np.testing.assert_allclose(qvel[0, :3], root_state[0, 7:10])
    np.testing.assert_allclose(qvel[0, 3:6], [0.0, -1.0, 0.0], atol=1e-7)


@pytest.mark.parametrize(
    ("root_state", "error", "match"),
    [
        (np.zeros((1, 12)), ValueError, "root state.*expected"),
        (np.zeros((1, 13), dtype=np.int32), TypeError, "root state.*floating"),
        (np.full((1, 13), np.nan), ValueError, "root state.*NaN or Inf"),
        (
            np.array([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            ValueError,
            "root quaternion must be unit length",
        ),
    ],
)
def test_root_state_values_fail_closed(root_state, error, match: str) -> None:
    backend = _Backend(
        qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(6),
    )
    transaction = _transaction(backend)
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    with pytest.raises(error, match=match):
        with transaction.scoped(np.array([0], dtype=np.int32)):
            transaction.write_root_state(
                np.array([0], dtype=np.int32),
                layout,
                root_state,
                term_name="bad_root",
            )
    assert backend.set_state_calls == []


def test_root_layout_bounds_and_reset_scope_fail_closed() -> None:
    backend = _Backend(
        qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(6),
    )
    transaction = _transaction(backend)
    out_of_bounds = BackendRootStateLayout(tuple(range(1, 8)), tuple(range(6)))

    with pytest.raises(IndexError, match="root qpos indices out of range"):
        with transaction.scoped(np.array([0], dtype=np.int32)):
            transaction.write_root_pose(
                np.array([0], dtype=np.int32),
                out_of_bounds,
                np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                term_name="bad_layout",
            )

    valid = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    with pytest.raises(ValueError, match="root-pose mutation outside the active reset"):
        with transaction.scoped(np.array([0], dtype=np.int32)):
            transaction.write_root_pose(
                np.array([1], dtype=np.int32),
                valid,
                np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                term_name="outside",
            )


@pytest.mark.parametrize(
    ("position", "velocity", "error", "match"),
    [
        (np.zeros((1, 2)), np.zeros((1, 1)), ValueError, "joint position.*expected"),
        (np.zeros((1, 1)), np.zeros((2, 1)), ValueError, "joint velocity.*expected"),
        (np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1)), TypeError, "must be floating"),
        (np.full((1, 1), np.nan), np.zeros((1, 1)), ValueError, "NaN or Inf"),
    ],
)
def test_joint_write_values_fail_closed(position, velocity, error, match: str) -> None:
    transaction = _transaction(_Backend())
    with pytest.raises(error, match=match):
        with transaction.scoped(np.array([0], dtype=np.int32)):
            transaction.write_joint_state(
                np.array([0], dtype=np.int32),
                np.array([1], dtype=np.int32),
                np.array([0], dtype=np.int32),
                position,
                velocity,
                term_name="joint_term",
            )


def test_mutation_must_stay_inside_active_reset() -> None:
    transaction = _transaction(_Backend())
    with transaction.scoped(np.array([1, 2], dtype=np.int32)):
        with pytest.raises(ValueError, match="outside the active reset.*3"):
            transaction.reset_to_default(np.array([3], dtype=np.int32), term_name="bad")

    with pytest.raises(RuntimeError, match="requires an active reset event"):
        transaction.reset_to_default(np.array([1], dtype=np.int32), term_name="late")


@pytest.mark.parametrize(
    ("ids", "error", "match"),
    [
        ([0], TypeError, "must be np.ndarray"),
        (np.array([[0]], dtype=np.int32), TypeError, "1-D integer"),
        (np.array([True]), TypeError, "1-D integer"),
        (np.array([-1], dtype=np.int32), IndexError, "out of range"),
        (np.array([4], dtype=np.int32), IndexError, "out of range"),
        (np.array([1, 1], dtype=np.int32), ValueError, "duplicates"),
    ],
)
def test_begin_rejects_invalid_environment_ids(ids, error, match: str) -> None:
    with pytest.raises(error, match=match):
        _transaction(_Backend()).begin(ids)


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    [
        ("qpos", [1.0], TypeError, "default qpos.*np.ndarray"),
        ("qpos", np.zeros((1, 1)), ValueError, "default qpos.*expected 1-D"),
        ("qpos", np.array([1], dtype=np.int32), TypeError, "default qpos.*floating"),
        ("qpos", np.array([np.nan]), ValueError, "default qpos.*NaN or Inf"),
        ("qvel", np.array([np.inf]), ValueError, "initial qvel.*NaN or Inf"),
    ],
)
def test_backend_default_state_contract_fails_at_mutation_boundary(
    field: str,
    value,
    error,
    match: str,
) -> None:
    kwargs = {field: value}
    transaction = _transaction(_Backend(**kwargs))
    with pytest.raises(error, match=match):
        with transaction.scoped(np.array([0], dtype=np.int32)):
            transaction.reset_to_default(
                np.array([0], dtype=np.int32),
                term_name="reset_scene_to_default",
            )
    assert not transaction.active


def test_missing_default_and_set_state_capabilities_name_term_and_backend() -> None:
    missing = type("MissingBackend", (), {"num_envs": 1, "backend_type": "missing"})()
    transaction = ResetStateTransaction(cast(SimBackend, missing))
    with pytest.raises(
        NotImplementedError,
        match="EventManager term 'reset_scene_to_default'.*default qpos.*backend 'missing'",
    ):
        with transaction.scoped(np.array([0], dtype=np.int32)):
            transaction.reset_to_default(
                np.array([0], dtype=np.int32),
                term_name="reset_scene_to_default",
            )

    backend = _Backend(fail_set_state=True)
    transaction = _transaction(backend)
    with pytest.raises(
        NotImplementedError,
        match="SimBackend.set_state.*reset_scene_to_default.*backend 'fake'",
    ):
        with transaction.scoped(np.array([0], dtype=np.int32)):
            transaction.reset_to_default(
                np.array([0], dtype=np.int32),
                term_name="reset_scene_to_default",
            )
    assert not transaction.active
