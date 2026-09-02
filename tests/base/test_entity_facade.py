from __future__ import annotations

import ast
import inspect
from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from unisim.backend.base import BackendRootStateLayout, SimBackend

import unilab.base.entity as entity_module
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.reset_state import ResetStateTransaction
from unilab.base.scene import SceneCfg
from unilab.managers import RewardManager, RewardTermCfg, SceneEntityCfg


class _StrictBackendProfile:
    """Strict public-contract fake shared by backend capability profiles."""

    num_envs = 3
    num_actuators = 5

    def __init__(self, backend_type: str, *, unsupported: frozenset[str] = frozenset()) -> None:
        self.backend_type = backend_type
        self.unsupported = unsupported
        self.calls: Counter[str] = Counter()
        self.joint_ids = {"hip": 2, "knee": 0, "ankle": 4}
        self.body_ids = {"base": 4, "foot": 7}
        self.site_ids = {"imu": 3}
        self.geom_names = ("floor", "base_collision", "foot_collision")
        self.actuator_names = ("knee", "unused", "hip", "unused_2", "ankle")
        self.actuator_joint_names = (
            "knee",
            "unused_joint",
            "hip",
            "unused_2_joint",
            "ankle",
        )
        self.dof_pos = np.arange(self.num_envs * 5, dtype=np.float32).reshape(self.num_envs, 5)
        self.dof_vel = self.dof_pos + 100.0
        base = np.arange(self.num_envs * 10 * 3, dtype=np.float32)
        self.body_pos = base.reshape(self.num_envs, 10, 3)
        self.body_quat = np.zeros((self.num_envs, 10, 4), dtype=np.float32)
        self.body_quat[..., 0] = 1.0
        self.body_lin_vel = self.body_pos + 200.0
        self.body_ang_vel = self.body_pos + 300.0
        self.body_lin_vel_b = self.body_pos + 400.0
        self.body_ang_vel_b = self.body_pos + 500.0
        self.default_qpos = np.array(
            [99.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 88.0],
            dtype=np.float32,
        )
        self.init_qvel = np.array(
            [77.0, 66.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 55.0],
            dtype=np.float32,
        )
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.joint_range = np.asarray(
            [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0], [-4.0, 4.0], [-5.0, 5.0]],
            dtype=np.float32,
        )

    def _check(self, capability: str) -> None:
        self.calls[capability] += 1
        if capability in self.unsupported:
            raise NotImplementedError(f"{self.backend_type} lacks {capability}")

    def get_body_ids(self, names) -> np.ndarray:
        self._check("body names")
        return np.asarray([self.body_ids[name] for name in names], dtype=np.int32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        self._check("root-state layout")
        if root_body_name != "base":
            raise ValueError(f"unknown root body {root_body_name}")
        return BackendRootStateLayout(tuple(range(1, 8)), tuple(range(2, 8)))

    def get_default_qpos(self) -> np.ndarray:
        self._check("default qpos")
        return self.default_qpos.copy()

    def get_init_qvel(self) -> np.ndarray:
        self._check("initial qvel")
        return self.init_qvel.copy()

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> None:
        assert randomization is None
        self.set_state_calls.append((env_ids.copy(), qpos.copy(), qvel.copy()))

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        self._check("joint position names")
        return np.asarray([self.joint_ids[name] for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        self._check("joint velocity names")
        return np.asarray([self.joint_ids[name] for name in names], dtype=np.int32)

    def get_site_ids(self, names) -> np.ndarray:
        self._check("site names")
        return np.asarray([self.site_ids[name] for name in names], dtype=np.int32)

    def get_geom_names(self) -> tuple[str, ...]:
        self._check("geom names")
        return self.geom_names

    def get_actuator_names(self) -> tuple[str, ...]:
        self._check("actuator names")
        return self.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        self._check("actuator target joints")
        return self.actuator_joint_names

    def get_actuator_ctrl_range(self) -> np.ndarray:
        self.calls["actuator range"] += 1
        return np.arange(10, dtype=np.float32).reshape(5, 2)

    def get_dof_pos(self) -> np.ndarray:
        self._check("joint position state")
        return self.dof_pos

    def get_default_dof_pos(self) -> np.ndarray:
        self._check("default joint position")
        return np.arange(5, dtype=np.float32) + 10.0

    def get_dof_vel(self) -> np.ndarray:
        self._check("joint velocity state")
        return self.dof_vel

    def get_joint_range(self) -> np.ndarray:
        self._check("joint position limits")
        return self.joint_range

    def get_body_pos_w(self, ids: np.ndarray) -> np.ndarray:
        self._check("body position state")
        return self.body_pos[:, ids]

    def get_body_quat_w(self, ids: np.ndarray) -> np.ndarray:
        self._check("body quaternion state")
        return self.body_quat[:, ids]

    def get_body_lin_vel_w(self, ids: np.ndarray) -> np.ndarray:
        self._check("body linear velocity state")
        return self.body_lin_vel[:, ids]

    def get_body_ang_vel_w(self, ids: np.ndarray) -> np.ndarray:
        self._check("body angular velocity state")
        return self.body_ang_vel[:, ids]

    def copy_body_state_w(
        self,
        ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._check("body state copy")
        np.take(self.body_pos, ids, axis=1, out=out_pos)
        np.take(self.body_quat, ids, axis=1, out=out_quat)
        np.take(self.body_lin_vel, ids, axis=1, out=out_lin_vel)
        np.take(self.body_ang_vel, ids, axis=1, out=out_ang_vel)
        return out_pos, out_quat, out_lin_vel, out_ang_vel

    def get_body_lin_vel_b(self, ids: np.ndarray) -> np.ndarray:
        self._check("body-frame linear velocity state")
        return self.body_lin_vel_b[:, ids]

    def get_body_ang_vel_b(self, ids: np.ndarray) -> np.ndarray:
        self._check("body-frame angular velocity state")
        return self.body_ang_vel_b[:, ids]


def _scene(backend_type: str = "mujoco") -> tuple[_StrictBackendProfile, EntityScene]:
    backend = _StrictBackendProfile(backend_type)
    cfg = SceneCfg(
        model_file="unused.xml",
        entities={
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=("ankle", "hip"),
                body_names=("foot", "base"),
                geom_names=("foot_collision", "base_collision"),
                site_names=("imu",),
                actuator_names=("ankle", "hip"),
            )
        },
    )
    return backend, EntityScene.from_scene_cfg(cfg, cast(SimBackend, backend))


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix", "drake"])
def test_backend_profiles_materialize_identical_local_entity_contract(backend_type: str) -> None:
    backend, scene = _scene(backend_type)
    robot = scene["robot"]

    assert robot.joint_names == ("ankle", "hip")
    assert robot.body_names == ("foot", "base")
    np.testing.assert_array_equal(robot.data.joint_pos, backend.dof_pos[:, [4, 2]])
    np.testing.assert_array_equal(robot.data.joint_vel, backend.dof_vel[:, [4, 2]])
    np.testing.assert_array_equal(robot.data.default_joint_vel, 0.0)
    np.testing.assert_array_equal(robot.data.soft_joint_pos_limits, backend.joint_range[[4, 2]])
    np.testing.assert_array_equal(robot.data.gravity_vec_w, [[0.0, 0.0, -1.0]] * 3)
    np.testing.assert_array_equal(robot.data.joint_pos_biased, backend.dof_pos[:, [4, 2]])
    np.testing.assert_array_equal(robot.data.body_link_pos_w, backend.body_pos[:, [7, 4]])
    np.testing.assert_array_equal(robot.data.root_link_pos_w, backend.body_pos[:, 4])
    np.testing.assert_array_equal(robot.data.root_link_lin_vel_b, backend.body_lin_vel_b[:, 4])
    np.testing.assert_array_equal(robot.data.root_link_ang_vel_b, backend.body_ang_vel_b[:, 4])
    np.testing.assert_array_equal(robot.data.heading_w, 0.0)
    np.testing.assert_array_equal(robot.data.projected_gravity_b, [[0.0, 0.0, -1.0]] * 3)
    np.testing.assert_array_equal(
        robot.data.default_root_state,
        np.tile([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (3, 1)),
    )
    assert not robot.data.default_root_state.flags.writeable
    assert not robot.data.soft_joint_pos_limits.flags.writeable
    assert not robot.data.gravity_vec_w.flags.writeable
    np.testing.assert_array_equal(
        robot.data.actuator_ctrl_range,
        np.arange(10, dtype=np.float32).reshape(5, 2)[[4, 2]],
    )


def test_body_state_copy_binding_freezes_local_to_backend_mapping() -> None:
    backend, scene = _scene()
    robot = scene["robot"]
    copy_body_state = robot.bind_body_state_copy(np.asarray([1, 0], dtype=np.int32))
    before = backend.calls.copy()
    out_pos = np.empty((backend.num_envs, 2, 3), dtype=np.float32)
    out_quat = np.empty((backend.num_envs, 2, 4), dtype=np.float32)
    out_lin_vel = np.empty_like(out_pos)
    out_ang_vel = np.empty_like(out_pos)

    result = copy_body_state(out_pos, out_quat, out_lin_vel, out_ang_vel)

    assert result == (out_pos, out_quat, out_lin_vel, out_ang_vel)
    np.testing.assert_array_equal(out_pos, backend.body_pos[:, [4, 7]])
    np.testing.assert_array_equal(out_quat, backend.body_quat[:, [4, 7]])
    np.testing.assert_array_equal(out_lin_vel, backend.body_lin_vel[:, [4, 7]])
    np.testing.assert_array_equal(out_ang_vel, backend.body_ang_vel[:, [4, 7]])
    assert backend.calls["body state copy"] == before["body state copy"] + 1
    for capability in (
        "body position state",
        "body quaternion state",
        "body linear velocity state",
        "body angular velocity state",
    ):
        assert backend.calls[capability] == before[capability]


@pytest.mark.parametrize(
    ("body_ids", "error_type", "message"),
    [
        (np.asarray([], dtype=np.int32), ValueError, "selected no bodies"),
        ([2], IndexError, "out of range"),
        ([0, 0], ValueError, "contain duplicates"),
        ([True], TypeError, "1-D integer array"),
    ],
)
def test_body_state_copy_binding_rejects_invalid_local_ids(
    body_ids: Any, error_type: type[Exception], message: str
) -> None:
    _, scene = _scene()
    with pytest.raises(error_type, match=message):
        scene["robot"].bind_body_state_copy(body_ids)


def test_state_read_cache_is_scoped_shared_and_explicitly_invalidated() -> None:
    backend, scene = _scene()
    robot = scene["robot"]
    before = backend.calls.copy()

    with scene._scoped_state_reads():
        np.testing.assert_array_equal(robot.data.joint_pos, robot.data.joint_pos)
        np.testing.assert_array_equal(robot.data.joint_vel, robot.data.joint_vel)
        np.testing.assert_array_equal(robot.data.heading_w, robot.data.projected_gravity_b[:, 0])
        np.testing.assert_array_equal(robot.data.root_link_pos_w, robot.data.root_link_pos_w)
        np.testing.assert_array_equal(robot.data.body_link_pos_w, robot.data.body_link_pos_w)

        assert backend.calls["joint position state"] == before["joint position state"] + 1
        assert backend.calls["joint velocity state"] == before["joint velocity state"] + 1
        assert backend.calls["body quaternion state"] == before["body quaternion state"] + 1
        # Root and full-body selectors remain distinct cache entries.
        assert backend.calls["body position state"] == before["body position state"] + 2

        scene._invalidate_state_reads()
        _ = robot.data.joint_pos
        assert backend.calls["joint position state"] == before["joint position state"] + 2

    # Reads outside the env-owned phase retain the previous per-access behavior.
    _ = robot.data.joint_pos
    _ = robot.data.joint_pos
    assert backend.calls["joint position state"] == before["joint position state"] + 4


def test_state_read_cache_scope_closes_after_exception() -> None:
    backend, scene = _scene()
    robot = scene["robot"]
    before = backend.calls["joint position state"]

    with pytest.raises(RuntimeError, match="term failed"):
        with scene._scoped_state_reads():
            _ = robot.data.joint_pos
            raise RuntimeError("term failed")

    with scene._scoped_state_reads():
        _ = robot.data.joint_pos
    assert backend.calls["joint position state"] == before + 2


def test_scene_entity_cfg_resolves_only_against_cached_names() -> None:
    backend, scene = _scene()
    cold_path_calls = backend.calls.copy()

    cfg = SceneEntityCfg(
        "robot",
        joint_names=("hip", "ankle"),
        body_names=".*",
        geom_names="foot_.*",
        site_names="imu",
        actuator_names=["hip", "ankle"],
        preserve_order=True,
    )
    cfg.resolve(scene)

    assert cfg.joint_ids == [1, 0]
    assert cfg.body_ids == slice(None)
    assert cfg.geom_ids == [0]
    assert cfg.site_ids == slice(None)
    assert cfg.actuator_ids == [1, 0]
    assert backend.calls == cold_path_calls

    for _ in range(3):
        scene["robot"].data.joint_pos
        scene["robot"].data.root_link_quat_w
    for key in (
        "body names",
        "joint position names",
        "joint velocity names",
        "site names",
        "geom names",
        "actuator names",
    ):
        assert backend.calls[key] == cold_path_calls[key]


def test_entity_control_write_uses_cached_actuator_columns_and_fails_closed() -> None:
    backend = _StrictBackendProfile("mujoco")
    control = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)
    scene = EntityScene(
        {"robot": EntityCfg(actuator_names=("ankle", "hip"))},
        cast(SimBackend, backend),
        control,
    )
    data = scene["robot"].data

    data.write_ctrl(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
    np.testing.assert_array_equal(control[:, 4], [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(control[:, 2], [2.0, 4.0, 6.0])
    data.write_ctrl(
        np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32),
        env_ids=np.array([2, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(control[:, 4], [7.0, 3.0, 9.0])
    np.testing.assert_array_equal(control[:, 2], [6.0, 4.0, 8.0])
    assert backend.calls["actuator names"] == 1

    with pytest.raises(ValueError, match="expected shape"):
        data.write_ctrl(np.zeros((backend.num_envs, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        data.write_ctrl(np.full((backend.num_envs, 2), np.nan, dtype=np.float32))
    with pytest.raises(IndexError, match="out of range"):
        data.write_ctrl(
            np.zeros((1, 2), dtype=np.float32),
            env_ids=np.array([backend.num_envs], dtype=np.int32),
        )

    _, read_only_scene = _scene()
    with pytest.raises(NotImplementedError, match="actuator control write.*not materialized"):
        read_only_scene["robot"].data.write_ctrl(np.zeros((backend.num_envs, 2), dtype=np.float32))


def test_entity_root_writes_use_cached_layout_and_one_reset_commit() -> None:
    backend = _StrictBackendProfile("mujoco")
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    scene = EntityScene(
        {"robot": EntityCfg(root_body_name="base")},
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    robot = scene["robot"]
    cold_layout_calls = backend.calls["root-state layout"]
    half_sqrt = np.sqrt(0.5)

    with transaction.scoped(np.array([0, 2], dtype=np.int32)):
        robot.write_root_link_pose_to_sim(
            np.array(
                [
                    [10.0, 11.0, 12.0, half_sqrt, 0.0, 0.0, half_sqrt],
                    [20.0, 21.0, 22.0, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            env_ids=np.array([2, 0], dtype=np.int32),
        )
        robot.write_root_link_velocity_to_sim(
            np.array(
                [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0], [4.0, 5.0, 6.0, 0.0, 1.0, 2.0]],
                dtype=np.float32,
            ),
            env_ids=np.array([2, 0], dtype=np.int32),
        )

    assert backend.calls["root-state layout"] == cold_layout_calls == 1
    assert len(backend.set_state_calls) == 1
    env_ids, qpos, qvel = backend.set_state_calls[0]
    np.testing.assert_array_equal(env_ids, [0, 2])
    np.testing.assert_allclose(qpos[0, 1:8], [20.0, 21.0, 22.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(qpos[1, 1:8], [10.0, 11.0, 12.0, half_sqrt, 0.0, 0.0, half_sqrt])
    np.testing.assert_allclose(qvel[0, 2:8], [4.0, 5.0, 6.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(qvel[1, 2:8], [1.0, 2.0, 3.0, 0.0, -1.0, 0.0], atol=1e-6)


def test_entity_caches_unsupported_root_layout_without_hot_path_probe() -> None:
    backend = _StrictBackendProfile(
        "drake",
        unsupported=frozenset({"root-state layout"}),
    )
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    scene = EntityScene(
        {"robot": EntityCfg(root_body_name="base")},
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    robot = scene["robot"]

    with pytest.raises(
        NotImplementedError,
        match="default root state.*backend 'drake'.*drake lacks root-state layout",
    ):
        _ = robot.data.default_root_state
    with pytest.raises(
        NotImplementedError,
        match="reset root-state layout.*backend 'drake'.*drake lacks root-state layout",
    ):
        with transaction.scoped(np.array([0], dtype=np.int32)):
            robot.write_root_state_to_sim(
                np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                env_ids=np.array([0], dtype=np.int32),
            )
    assert backend.calls["root-state layout"] == 1
    assert backend.set_state_calls == []


def test_entity_joint_position_target_maps_natural_joint_order_to_control_order() -> None:
    backend = _StrictBackendProfile("mujoco")
    control = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("ankle", "hip", "knee"),
                actuator_names=("ankle", "hip"),
            )
        },
        cast(SimBackend, backend),
        control,
    )
    robot = scene["robot"]

    ids, names = robot.find_joints_by_actuator_names(".*")
    assert ids == [0, 1]
    assert names == ["ankle", "hip"]
    np.testing.assert_array_equal(
        robot.data.default_joint_pos,
        np.asarray([[14.0, 12.0, 10.0]] * backend.num_envs, dtype=np.float32),
    )
    np.testing.assert_array_equal(robot.data.encoder_bias, 0.0)

    targets = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    robot.set_joint_position_target(targets, joint_ids=np.asarray(ids, dtype=np.int32))
    np.testing.assert_array_equal(control[:, 4], targets[:, 0])
    np.testing.assert_array_equal(control[:, 2], targets[:, 1])
    np.testing.assert_array_equal(control[:, [0, 1, 3]], 0.0)

    with pytest.raises(NotImplementedError, match="passive joints.*knee"):
        robot.set_joint_position_target(
            np.ones((backend.num_envs, 1), dtype=np.float32),
            joint_ids=np.asarray([2], dtype=np.int32),
        )


@pytest.mark.parametrize(
    ("actuator_joint_names", "joint_names", "message"),
    [
        (("hip", "unused", "hip", "unused_2", "knee"), ("hip",), "must be unique"),
        (
            ("knee", "unused_joint", "hip", "unused_2_joint", "ankle"),
            ("hip",),
            "outside its declared joint partition.*knee",
        ),
    ],
)
def test_entity_joint_actuator_mapping_rejects_ambiguous_or_missing_targets(
    actuator_joint_names: tuple[str, ...],
    joint_names: tuple[str, ...],
    message: str,
) -> None:
    backend = _StrictBackendProfile("mujoco")
    backend.actuator_joint_names = actuator_joint_names

    with pytest.raises(ValueError, match=message):
        EntityScene(
            {
                "robot": EntityCfg(
                    joint_names=joint_names,
                    actuator_names=("knee", "hip"),
                )
            },
            cast(SimBackend, backend),
        )


@pytest.mark.parametrize(
    ("ids", "error_type", "message"),
    [
        ([-1], ValueError, "EntityCfg entity 'robot' joint selector.*out of range"),
        ([2], ValueError, "EntityCfg entity 'robot' joint selector.*out of range"),
        ([True], TypeError, "EntityCfg entity 'robot' joint selector.*must be integers"),
        (["0"], TypeError, "EntityCfg entity 'robot' joint selector.*must be integers"),
    ],
)
def test_scene_entity_cfg_rejects_invalid_ids(ids, error_type, message: str) -> None:
    _, scene = _scene()
    cfg = SceneEntityCfg("robot", joint_ids=ids)
    with pytest.raises(error_type, match=message):
        cfg.resolve(scene)


def test_scene_entity_cfg_reports_entity_for_invalid_regex() -> None:
    _, scene = _scene()
    with pytest.raises(
        ValueError,
        match="SceneEntityCfg entity 'robot' joint selector.*Invalid entity selector regex",
    ):
        SceneEntityCfg("robot", joint_names="[").resolve(scene)


def test_missing_entity_namespace_and_backend_capability_fail_closed() -> None:
    backend = _StrictBackendProfile("drake", unsupported=frozenset({"actuator names"}))
    with pytest.raises(
        NotImplementedError,
        match="Entity 'robot'.*actuator.*backend 'drake'",
    ):
        EntityScene(
            {"robot": EntityCfg(actuator_names=("hip",))},
            cast(SimBackend, backend),
        )

    _, scene = _scene("motrix")
    with pytest.raises(
        NotImplementedError,
        match="Entity 'robot'.*tendon.*backend 'motrix'",
    ):
        SceneEntityCfg("robot", tendon_names=".*").resolve(scene)

    sparse = EntityScene(
        {"robot": EntityCfg(joint_names=("hip",))},
        cast(SimBackend, _StrictBackendProfile("mujoco")),
    )
    with pytest.raises(NotImplementedError, match="body.*not declared"):
        SceneEntityCfg("robot", body_names=".*").resolve(sparse)

    state_missing = _StrictBackendProfile("mujoco", unsupported=frozenset({"body position state"}))
    with pytest.raises(
        NotImplementedError,
        match="Entity 'robot'.*body position state.*backend 'mujoco'",
    ):
        EntityScene(
            {"robot": EntityCfg(root_body_name="base")},
            cast(SimBackend, state_missing),
        )


def test_entity_declaration_rejects_coercion_and_duplicate_names() -> None:
    backend = cast(SimBackend, _StrictBackendProfile("mujoco"))
    with pytest.raises(TypeError, match="sequence of strings, not a scalar"):
        EntityScene(
            {"robot": EntityCfg(joint_names=cast(Any, "hip"))},
            backend,
        )
    with pytest.raises(TypeError, match="joint names must be strings"):
        EntityScene(
            {"robot": EntityCfg(joint_names=cast(Any, ("hip", 1)))},
            backend,
        )
    with pytest.raises(ValueError, match="joint names must be unique"):
        EntityScene(
            {"robot": EntityCfg(joint_names=("hip", "hip"))},
            backend,
        )


def test_manager_resolution_error_has_manager_term_entity_capability_and_backend() -> None:
    _, scene = _scene("drake")
    env = SimpleNamespace(
        num_envs=3,
        scene=scene,
        rng=np.random.default_rng(7),
        max_episode_length_s=2.0,
    )

    with pytest.raises(
        NotImplementedError,
        match=(
            "RewardManager term 'unsupported' parameter 'asset_cfg'.*"
            "Entity 'robot'.*tendon.*backend 'drake'"
        ),
    ):
        RewardManager(
            {
                "unsupported": RewardTermCfg(
                    func=lambda env, asset_cfg: np.zeros(env.num_envs),
                    weight=1.0,
                    params={"asset_cfg": SceneEntityCfg("robot", tendon_names=".*")},
                )
            },
            cast(Any, env),
        )


def test_entity_facade_has_no_backend_model_or_asset_access() -> None:
    tree = ast.parse(inspect.getsource(entity_module))
    accessed_attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "model" not in accessed_attributes
    assert "scene_model_file" not in accessed_attributes


def test_real_mujoco_entity_selector_and_numpy_state_smoke() -> None:
    from unisim.backend.mujoco.backend import MuJoCoBackend

    joint_names = (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    )
    actuator_names = (
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
    )
    scene_cfg = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "go2" / "scene_flat.xml"),
        entities={
            "robot": EntityCfg(
                root_body_name="base",
                joint_names=joint_names,
                body_names=("base",),
                actuator_names=actuator_names,
            )
        },
    )
    backend = MuJoCoBackend(
        scene_cfg,
        num_envs=2,
        sim_dt=0.01,
        base_name="base",
        add_body_sensors=True,
    )
    backend.materialize()
    scene = EntityScene.from_scene_cfg(scene_cfg, backend)

    selector = SceneEntityCfg("robot", joint_names=".*_calf_joint")
    selector.resolve(scene)
    assert selector.joint_ids == [2, 5, 8, 11]
    assert scene["robot"].data.joint_pos.shape == (2, 12)
    assert scene["robot"].data.root_link_pose_w.shape == (2, 7)
    assert scene["robot"].data.root_link_lin_vel_b.shape == (2, 3)
    assert scene["robot"].data.root_link_ang_vel_b.shape == (2, 3)
    assert scene["robot"].data.heading_w.shape == (2,)
    assert scene["robot"].data.projected_gravity_b.shape == (2, 3)
    assert scene["robot"].data.default_joint_vel.shape == (2, 12)
    assert np.isfinite(scene["robot"].data.root_link_lin_vel_b).all()
    assert np.isfinite(scene["robot"].data.root_link_ang_vel_b).all()
    assert np.isfinite(scene["robot"].data.heading_w).all()
    assert np.isfinite(scene["robot"].data.projected_gravity_b).all()
    np.testing.assert_array_equal(scene["robot"].data.default_joint_vel, 0.0)


def test_scene_cfg_entity_defaults_are_not_shared() -> None:
    first = SceneCfg(model_file="first.xml")
    second = SceneCfg(model_file="second.xml")
    first.entities["robot"] = EntityCfg()
    assert second.entities == {}


def test_scene_exposes_read_only_community_entities_and_zero_origins() -> None:
    _, scene = _scene()

    assert scene.entities["robot"] is scene["robot"]
    with pytest.raises(TypeError):
        scene.entities["other"] = scene["robot"]  # type: ignore[index]

    assert scene.env_origins.shape == (3, 3)
    np.testing.assert_array_equal(scene.env_origins, 0.0)
    with pytest.raises(ValueError, match="read-only"):
        scene.env_origins[0, 0] = 1.0
