"""Tests for public, cold-path manager entity resolution."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend
from unilab.manager import BackendEntityResolver, EntityKind, EntitySelector, ManagerContractError
from unilab.manager.entities import SelectorMode


def _selector(
    key: str,
    kind: EntityKind,
    expressions: tuple[str, ...],
    *,
    mode: SelectorMode = SelectorMode.EXACT,
) -> EntitySelector:
    return EntitySelector(
        key=key,
        entity="robot",
        kind=kind,
        expressions=expressions,
        mode=mode,
    )


def _backend() -> Mock:
    backend = Mock(spec=SimBackend)
    backend.get_body_ids.side_effect = lambda names: np.asarray(
        [10 + index for index, _ in enumerate(names)], dtype=np.int32
    )
    backend.get_joint_dof_pos_indices.side_effect = lambda names: np.asarray(
        [20 + index for index, _ in enumerate(names)], dtype=np.int32
    )
    backend.get_sensor_ids.side_effect = lambda names: np.asarray(
        [30 + index for index, _ in enumerate(names)], dtype=np.int32
    )
    return backend


def test_backend_entity_resolver_uses_only_public_exact_queries() -> None:
    backend = _backend()
    resolver = BackendEntityResolver(backend)

    assert resolver.resolve(_selector("robot.root", EntityKind.ROOT, ("pelvis",))) == (10,)
    assert resolver.resolve(_selector("robot.bodies", EntityKind.BODY, ("pelvis", "torso"))) == (
        10,
        11,
    )
    assert resolver.resolve(_selector("robot.dofs", EntityKind.DOF, ("left_hip", "right_hip"))) == (
        20,
        21,
    )
    assert resolver.resolve(
        _selector("robot.sensors", EntityKind.SENSOR, ("gyro", "upvector"))
    ) == (30, 31)

    assert backend.get_body_ids.call_args_list[0].args == (("pelvis",),)
    assert backend.get_body_ids.call_args_list[1].args == (("pelvis", "torso"),)
    backend.get_joint_dof_pos_indices.assert_called_once_with(("left_hip", "right_hip"))
    backend.get_sensor_ids.assert_called_once_with(("gyro", "upvector"))


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        (
            _selector("robot.regex", EntityKind.BODY, (".*",), mode=SelectorMode.REGEX),
            "exact selectors only",
        ),
        (_selector("robot.site", EntityKind.SITE, ("foot",)), "does not support selector kind"),
        (_selector("robot.roots", EntityKind.ROOT, ("pelvis", "torso")), "exactly one body"),
    ],
)
def test_backend_entity_resolver_rejects_unsupported_selector_shapes(
    selector: EntitySelector,
    message: str,
) -> None:
    backend = _backend()

    with pytest.raises(ManagerContractError, match=message):
        BackendEntityResolver(backend).resolve(selector)

    backend.get_body_ids.assert_not_called()
    backend.get_joint_dof_pos_indices.assert_not_called()
    backend.get_sensor_ids.assert_not_called()


@pytest.mark.parametrize(
    "ids",
    [
        [1],
        np.asarray([[1]], dtype=np.int32),
        np.asarray([1], dtype=np.float32),
        np.asarray([-1], dtype=np.int32),
        np.asarray([1, 1], dtype=np.int32),
    ],
)
def test_backend_entity_resolver_rejects_invalid_public_id_results(ids: object) -> None:
    backend = _backend()
    backend.get_sensor_ids.return_value = ids
    backend.get_sensor_ids.side_effect = None

    with pytest.raises(
        ManagerContractError, match="id array|shape|non-integer|invalid or duplicate"
    ):
        BackendEntityResolver(backend).resolve(
            _selector("robot.sensor", EntityKind.SENSOR, ("gyro",))
        )


def test_backend_entity_resolver_normalizes_backend_failures() -> None:
    backend = _backend()
    backend.get_sensor_ids.side_effect = ValueError("missing exact sensor")

    with pytest.raises(ManagerContractError, match="failed to resolve backend selector"):
        BackendEntityResolver(backend).resolve(
            _selector("robot.sensor", EntityKind.SENSOR, ("missing",))
        )
