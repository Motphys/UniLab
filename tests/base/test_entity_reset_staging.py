"""Sparse reset staging must avoid unused snapshots without changing writes."""

from types import SimpleNamespace

import numpy as np
import pytest
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout

from unilab.base.reset_state import ResetStateTransaction


def fixture(num_envs=5):
    entity = EntityLayout(
        "object",
        "articulation",
        "floating",
        "base",
        ("base", "a", "b"),
        (0, 1, 2),
        (None, "base", "a"),
        (JointLayout("a", "hinge", (7,), (6,), "a"), JointLayout("b", "hinge", (8,), (7,), "b")),
        (),
        (),
        (),
        tuple(range(7)),
        tuple(range(6)),
    )
    layout = CompiledSceneLayout((entity,), 9, 8, 0, 3)
    state = {
        "root_pose": np.tile([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], (num_envs, 1)),
        "root_velocity": np.zeros((num_envs, 6)),
        "joint_positions": np.tile([0.2, 0.4], (num_envs, 1)),
        "joint_velocities": np.tile([0.6, 0.8], (num_envs, 1)),
    }
    reads, commits = [], []

    def read(name):
        reads.append(name)
        return {key: values.copy() for key, values in state.items()}

    backend = SimpleNamespace(
        num_envs=num_envs,
        get_entity_state=read,
        reset_entities=commits.append,
        get_entity_default_state=lambda name, ids: {
            key: values[ids].copy() for key, values in state.items()
        },
    )
    return ResetStateTransaction(backend, scene_layout=layout), reads, commits


@pytest.mark.parametrize("default", [False, True])
def test_complete_fields_do_not_read_current_entity_state(default):
    transaction, reads, commits = fixture()
    ids = np.array([4, 1])
    with transaction.scoped(ids):
        if default:
            transaction.reset_to_default(ids, term_name="defaults")
        else:
            transaction.write_entity_state(
                "object",
                ids,
                term_name="pose",
                root_pose=np.array([[4.0, 0, 1, 1, 0, 0, 0], [1.0, 0, 1, 1, 0, 0, 0]]),
            )
    assert reads == [] and len(commits) == 1
    assert commits[0].env_ids == (1, 4)
    if not default:
        np.testing.assert_array_equal(commits[0].patches[0].root_pose[:, 0], [1, 4])


def test_mixed_joint_field_subsets_read_once_only_for_missing_merge_columns():
    transaction, reads, commits = fixture()
    ids = np.array([4, 1])
    with transaction.scoped(ids):
        transaction.write_entity_state(
            "object", ids, term_name="a", joint_names=("a",), joint_positions=np.array([[1], [2]])
        )
        transaction.write_entity_state(
            "object",
            ids[::-1],
            term_name="b",
            joint_names=("b",),
            joint_velocities=np.array([[3.5], [4.5]]),
        )
        assert reads == []
    assert reads == ["object"]
    patch = commits[0].patches[0]
    assert patch.joint_names == ("a", "b")
    np.testing.assert_allclose(patch.joint_positions, [[2, 0.4], [1, 0.4]])
    np.testing.assert_allclose(patch.joint_velocities, [[0.6, 3.5], [0.6, 4.5]])


def test_overlapping_joint_writes_preserve_fractions_and_last_write_wins():
    transaction, reads, commits = fixture()
    with transaction.scoped(np.array([1])):
        transaction.write_entity_state(
            "object",
            np.array([1]),
            term_name="first",
            joint_names=("a",),
            joint_positions=np.array([[1]]),
        )
        transaction.write_entity_state(
            "object",
            np.array([1]),
            term_name="second",
            joint_names=("a",),
            joint_positions=np.array([[0.25]]),
        )
    assert reads == []
    np.testing.assert_allclose(commits[0].patches[0].joint_positions, [[0.25]])


def test_late_invalid_patch_does_not_read_current_or_commit():
    transaction, reads, commits = fixture()
    with pytest.raises(ValueError), transaction.scoped(np.array([1])):
        transaction.write_entity_state(
            "object",
            np.array([1]),
            term_name="first",
            joint_names=("a",),
            joint_positions=np.array([[0.25]]),
        )
        transaction.write_entity_state(
            "object", np.array([1]), term_name="invalid", root_pose=np.zeros((1, 7))
        )
    assert reads == [] and commits == []


def test_root_read_after_only_joint_staging_uses_current_root_and_keeps_it_unwritten():
    transaction, reads, commits = fixture()
    ids = np.array([4, 1])
    original = transaction._backend.get_entity_state

    def current(entity):
        state = original(entity)
        state["root_pose"][:, 0] = np.arange(5) + 10
        return state

    transaction._backend.get_entity_state = current
    with transaction.scoped(ids):
        transaction.write_entity_state(
            "object",
            ids,
            term_name="joint",
            joint_names=("a",),
            joint_positions=np.array([[0.2], [0.3]]),
        )
        assert reads == []
        pose = transaction.read_entity_root_pose("object", ids)
        np.testing.assert_array_equal(pose[:, 0], [14, 11])
        pose[:] = 999
    assert reads == ["object"]
    assert commits[0].patches[0].root_pose is None
    assert commits[0].patches[0].root_velocity is None
    np.testing.assert_allclose(commits[0].patches[0].joint_positions, [[0.3], [0.2]])
