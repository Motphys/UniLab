"""Focused tests for the base-owned Manager-Based reset transaction."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend
from unilab.base.reset_state import ResetStateTransaction


class _Backend:
    backend_type = "fake"

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

    def get_default_qpos(self):
        self.default_qpos_calls += 1
        return self.qpos

    def get_init_qvel(self):
        self.init_qvel_calls += 1
        return self.qvel

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> dict:
        assert randomization is None
        if self.fail_set_state:
            raise NotImplementedError("reset upload disabled")
        self.set_state_calls.append((env_ids.copy(), qpos.copy(), qvel.copy()))
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
