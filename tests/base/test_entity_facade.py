from __future__ import annotations

import ast
import inspect
from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    MocapPoseBinding,
    RootStateLayout,
    SimBackend,
    TensorExecution,
    TrackedBodyStateViews,
)
from unisim.dr.types import DomainRandomizationCapabilities
from unisim.scene_layout import CompiledSceneLayout, EntityLayout

import unilab.base.entity as entity_module
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.entity import Entity, EntityCfg, EntityScene
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

    def get_root_state_layout(self, root_body_name: str) -> RootStateLayout:
        self._check("root-state layout")
        if root_body_name != "base":
            raise ValueError(f"unknown root body {root_body_name}")
        return RootStateLayout(tuple(range(1, 8)), tuple(range(2, 8)))

    def get_default_qpos(self) -> np.ndarray:
        self._check("default qpos")
        return self.default_qpos.copy()

    def get_init_qvel(self) -> np.ndarray:
        self._check("initial qvel")
        return self.init_qvel.copy()

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> None:
        self.set_state_calls.append(
            (
                env_indices.detach().cpu().numpy().copy(),
                qpos.detach().cpu().numpy().copy(),
                qvel.detach().cpu().numpy().copy(),
            )
        )

    def get_state_views(self, fields, device=None) -> dict[str, torch.Tensor]:
        views = {}
        if "qpos" in fields:
            self._check("joint position state")
            views["qpos"] = torch.from_numpy(self.dof_pos)
        if "qvel" in fields:
            self._check("joint velocity state")
            views["qvel"] = torch.from_numpy(self.dof_vel)
        return views

    def get_sensor_view(self, name: str, device=None) -> torch.Tensor:
        self._check("tracked body state")
        prefix_by_field = {
            "track_pos_w_": self.body_pos,
            "track_quat_w_": self.body_quat,
            "track_linvel_w_": self.body_lin_vel,
            "track_angvel_w_": self.body_ang_vel,
        }
        for prefix, source in prefix_by_field.items():
            if not name.startswith(prefix):
                continue
            body_name = name[len(prefix) :]
            return torch.from_numpy(source[:, self.body_ids[body_name]])
        raise KeyError(name)

    def get_joint_dof_pos_indices(self, names) -> np.ndarray:
        self._check("joint position names")
        return np.asarray([self.joint_ids[name] for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names) -> np.ndarray:
        self._check("joint velocity names")
        return np.asarray([self.joint_ids[name] for name in names], dtype=np.int32)

    def get_joint_state_qpos_indices(self, names) -> np.ndarray:
        self._check("joint qpos layout")
        return np.asarray([self.joint_ids[name] for name in names], dtype=np.int32)

    def get_joint_state_qvel_indices(self, names) -> np.ndarray:
        self._check("joint qvel layout")
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

    def get_default_dof_pos(self) -> np.ndarray:
        self._check("default joint position")
        return np.arange(5, dtype=np.float32) + 10.0

    def get_joint_range(self) -> np.ndarray:
        self._check("joint position limits")
        return self.joint_range

    def get_tracked_body_views(self, body_names, device=None) -> TrackedBodyStateViews:
        capability = {
            "pos_w": "body position state",
            "quat_w": "body quaternion state",
            "lin_vel_w": "body linear velocity state",
            "ang_vel_w": "body angular velocity state",
        }
        self._check(capability["pos_w"])
        names = tuple(body_names)
        ids = np.asarray([self.body_ids[name] for name in names], dtype=np.int32)
        return TrackedBodyStateViews(
            body_names=names,
            pos_w=torch.from_numpy(self.body_pos[:, ids]),
            quat_w=torch.from_numpy(self.body_quat[:, ids]),
            lin_vel_w=torch.from_numpy(self.body_lin_vel[:, ids]),
            ang_vel_w=torch.from_numpy(self.body_ang_vel[:, ids]),
        )


def test_manipulation_entity_maps_local_columns_and_binds_mocap_without_floating_root():
    class Backend(_StrictBackendProfile):
        def __init__(self):
            super().__init__("mjwarp")
            self.payload = None
            self.poses = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (3, 1))

        def get_dr_capabilities(self):
            return DomainRandomizationCapabilities(
                supported_reset_terms=frozenset(("geom_size", "dof_damping"))
            )

        def get_default_geom_sizes(self):
            self._check("geom size defaults")
            return np.ones((3, 3))

        def get_dof_damping(self):
            self._check("damping defaults")
            return np.ones(9)

        def get_reset_term_default(self, term):
            self._check(f"{term} defaults")
            if term == "geom_size":
                return np.ones((3, 3))
            if term == "dof_damping":
                return np.ones(9)
            raise NotImplementedError(term)

        def get_joint_dof_indices(self, names):
            self._check("model DOF IDs")
            return np.array([{"hip": 7}[name] for name in names], dtype=np.int32)

        def bind_mocap_pose(self, name):
            self._check("mocap binding")

            def write(ids, poses):
                self.poses[ids] = poses

            return MocapPoseBinding(
                "mjwarp", name, self.num_envs, self.poses[0], lambda: self.poses, write
            )

        def set_state_tensor(self, env_indices, qpos, qvel, randomization=None):
            super().set_state_tensor(env_indices, qpos, qvel, randomization=randomization)
            self.payload = randomization
            self.poses[env_indices.detach().cpu().numpy(), 0] = 0.0

    backend = Backend()
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    hand = Entity(
        "hand",
        EntityCfg(
            body_names=("foot",),
            joint_names=("hip",),
            geom_names=("foot_collision", "base_collision"),
        ),
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    geom_ids, _ = hand.bind_geom_size_write([0], term_name="size")
    joint_ids, _ = hand.bind_joint_damping_write(term_name="damping")
    hand.bind_mocap_pose_write("foot", term_name="wrist")
    calls = backend.calls.copy()
    ids = torch.tensor([2], dtype=torch.int64)
    with pytest.raises(ValueError, match="every environment row"), transaction.scoped_startup(ids):
        pass
    assert backend.payload is None
    assert backend.calls == calls
    assert backend.calls["root-state layout"] == 0


def test_entity_binding_selects_per_world_default_columns() -> None:
    class Backend(_StrictBackendProfile):
        def __init__(self):
            super().__init__("mjwarp")
            self.payload = None

        def get_dr_capabilities(self):
            return DomainRandomizationCapabilities(supported_reset_terms=frozenset(("body_mass",)))

        def get_reset_term_default(self, term):
            self._check("body_mass defaults")
            if term != "body_mass":
                raise NotImplementedError(term)
            return np.asarray(
                [[1.0 + env_id] * 10 for env_id in range(self.num_envs)],
                dtype=np.float64,
            )

        def set_state_tensor(self, env_indices, qpos, qvel, randomization=None):
            super().set_state_tensor(env_indices, qpos, qvel, randomization=randomization)
            self.payload = randomization

    backend = Backend()
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    entity = Entity(
        "tool",
        EntityCfg(body_names=("foot",)),
        cast(SimBackend, backend),
        reset_state=transaction,
    )

    body_ids, defaults = entity.bind_body_mass_write(term_name="variant_mass")
    assert defaults.shape == (backend.num_envs, body_ids.size)
    np.testing.assert_allclose(defaults[:, 0], [1.0, 2.0, 3.0])

    with (
        pytest.raises(ValueError, match="every environment row"),
        transaction.scoped_startup(torch.tensor([1], dtype=torch.int64)),
    ):
        pass
    assert backend.payload is None
    assert backend.calls["body_mass defaults"] == 1


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
    np.testing.assert_array_equal(robot.data._host_joint_pos, backend.dof_pos[:, [4, 2]])
    np.testing.assert_array_equal(robot.data._host_joint_vel, backend.dof_vel[:, [4, 2]])
    soft_limits = robot.data.soft_joint_pos_limits(torch.device("cpu"))
    torch.testing.assert_close(soft_limits, torch.from_numpy(backend.joint_range[[4, 2]]))
    assert soft_limits.dtype == torch.float32
    assert robot.data.soft_joint_pos_limits(torch.device("cpu")) is soft_limits
    np.testing.assert_array_equal(robot.data._host_gravity_vec_w, [[0.0, 0.0, -1.0]] * 3)
    np.testing.assert_array_equal(robot.data._host_body_link_pos_w, backend.body_pos[:, [7, 4]])
    np.testing.assert_array_equal(robot.data._host_root_link_pos_w, backend.body_pos[:, 4])
    np.testing.assert_array_equal(robot.data._host_root_link_lin_vel_b, backend.body_lin_vel[:, 4])
    np.testing.assert_array_equal(robot.data._host_root_link_ang_vel_b, backend.body_ang_vel[:, 4])
    np.testing.assert_array_equal(robot.data._host_root_link_quat_w[:, 0], 1.0)
    np.testing.assert_array_equal(
        robot.data.default_root_state(torch.device("cpu")),
        np.tile([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (3, 1)),
    )
    default_vel = robot.data.default_joint_vel(torch.device("cpu"))
    torch.testing.assert_close(default_vel, torch.zeros_like(default_vel))


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
    assert backend.calls["body position state"] == before["body position state"] + 1
    for capability in (
        "body position state",
        "body quaternion state",
        "body linear velocity state",
        "body angular velocity state",
    ):
        if capability != "body position state":
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
        np.testing.assert_array_equal(robot.data._host_joint_pos, robot.data._host_joint_pos)
        np.testing.assert_array_equal(robot.data._host_joint_vel, robot.data._host_joint_vel)
        np.testing.assert_array_equal(
            robot.data._host_root_link_pos_w, robot.data._host_root_link_pos_w
        )
        np.testing.assert_array_equal(
            robot.data._host_body_link_pos_w, robot.data._host_body_link_pos_w
        )
        np.testing.assert_array_equal(
            robot.data._host_root_link_quat_w, robot.data._host_root_link_quat_w
        )

        assert backend.calls["joint position state"] == before["joint position state"] + 1
        assert backend.calls["joint velocity state"] == before["joint velocity state"] + 1
        # Root and full-body selectors remain distinct cache entries.  Each
        # named sensor view is one negotiated public backend boundary.
        assert backend.calls["tracked body state"] == before["tracked body state"] + 4

        scene._invalidate_state_reads()
        _ = robot.data._host_joint_pos
        assert backend.calls["joint position state"] == before["joint position state"] + 2

    # Reads outside the env-owned phase retain the previous per-access behavior.
    _ = robot.data._host_joint_pos
    _ = robot.data._host_joint_pos
    assert backend.calls["joint position state"] == before["joint position state"] + 4


def test_state_read_cache_scope_closes_after_exception() -> None:
    backend, scene = _scene()
    robot = scene["robot"]
    before = backend.calls["joint position state"]

    with pytest.raises(RuntimeError, match="term failed"):
        with scene._scoped_state_reads():
            _ = robot.data._host_joint_pos
            raise RuntimeError("term failed")

    with scene._scoped_state_reads():
        _ = robot.data._host_joint_pos
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
        scene["robot"].data._host_joint_pos
        scene["robot"].data._host_root_link_quat_w
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
        env_ids=torch.tensor([2, 0], dtype=torch.int64),
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
            env_ids=torch.tensor([backend.num_envs], dtype=torch.int64),
        )

    _, read_only_scene = _scene()
    with pytest.raises(NotImplementedError, match="actuator control write.*not materialized"):
        read_only_scene["robot"].data.write_ctrl(np.zeros((backend.num_envs, 2), dtype=np.float32))


def test_entity_root_writes_use_cached_layout_and_one_reset_commit() -> None:
    backend = _StrictBackendProfile("mujoco")
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    EntityScene(
        {"robot": EntityCfg(root_body_name="base")},
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    cold_layout_calls = backend.calls["root-state layout"]
    with (
        pytest.raises(ValueError, match="every environment row"),
        transaction.scoped_startup(torch.tensor([0, 2], dtype=torch.int64)),
    ):
        pass
    assert backend.calls["root-state layout"] == cold_layout_calls
    assert backend.set_state_calls == []


def test_entity_read_reset_root_pose_returns_staged_or_default_pose() -> None:
    backend = _StrictBackendProfile("mujoco")
    transaction = ResetStateTransaction(cast(SimBackend, backend))
    EntityScene(
        {"robot": EntityCfg(root_body_name="base")},
        cast(SimBackend, backend),
        reset_state=transaction,
    )
    with (
        pytest.raises(ValueError, match="every environment row"),
        transaction.scoped_startup(torch.tensor([0, 2], dtype=torch.int64)),
    ):
        pass
    assert backend.set_state_calls == []


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
        _ = robot.data.default_root_state(torch.device("cpu"))
    with (
        pytest.raises(ValueError, match="every environment row"),
        transaction.scoped_startup(torch.tensor([0], dtype=torch.int64)),
    ):
        pass
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
        robot.data.default_joint_pos(torch.device("cpu")),
        torch.tile(torch.tensor([14.0, 12.0, 10.0]), (backend.num_envs, 1)),
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


def test_entity_full_width_position_target_maps_natural_joint_order() -> None:
    backend = _StrictBackendProfile("mujoco")
    backend.num_actuators = 3
    backend.actuator_names = ("hip_motor", "unused_motor", "ankle_motor")
    backend.actuator_joint_names = ("hip", "unused", "ankle")
    backend.joint_ids = {"hip": 2, "knee": 0, "ankle": 4, "unused": 1}
    backend.get_actuator_ctrl_range = lambda: np.asarray([[-3.0, 3.0]] * 3, dtype=np.float32)
    control = torch.zeros((backend.num_envs, backend.num_actuators), dtype=torch.float32)
    scene = EntityScene(
        {
            "robot": EntityCfg(
                joint_names=("ankle", "hip", "unused"),
                actuator_names=backend.actuator_names,
            )
        },
        cast(SimBackend, backend),
        control,
    )

    target = torch.arange(1.0, 4.0, dtype=torch.float32).repeat(backend.num_envs, 1)
    scene["robot"].set_full_width_joint_position_target(target)

    torch.testing.assert_close(control, target[:, [1, 2, 0]])


def test_entity_cold_default_root_and_env_origin_torch_tables_are_cached() -> None:
    _, scene = _scene("mujoco")
    robot = scene["robot"]

    roots = robot.data.default_root_state(torch.device("cpu"))
    origins = scene.env_origins(torch.device("cpu"))
    assert roots.shape == (3, 13)
    assert origins.shape == (3, 3)
    assert roots.dtype == torch.float32
    assert origins.dtype == torch.float32
    assert robot.data.default_root_state(torch.device("cpu")) is roots
    assert scene.env_origins(torch.device("cpu")) is origins


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

    state_missing = _StrictBackendProfile("mujoco", unsupported=frozenset({"tracked body state"}))
    state_scene = EntityScene(
        {"robot": EntityCfg(root_body_name="base")},
        cast(SimBackend, state_missing),
    )
    with pytest.raises(
        NotImplementedError,
        match="Entity 'robot'.*tracked body 'pos_w'.*backend 'mujoco'",
    ):
        _ = state_scene["robot"].data._host_root_link_pos_w


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
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )
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
    assert scene["robot"].data._host_joint_pos.shape == (2, 12)
    assert scene["robot"].data._host_root_link_pos_w.shape == (2, 3)
    assert scene["robot"].data._host_root_link_quat_w.shape == (2, 4)
    assert scene["robot"].data._host_root_link_lin_vel_b.shape == (2, 3)
    assert scene["robot"].data._host_root_link_ang_vel_b.shape == (2, 3)
    assert scene["robot"].data._host_root_link_quat_w[:, 0].shape == (2,)
    assert scene["robot"].data.gravity_vec_w(torch.device("cpu")).shape == (2, 3)
    assert scene["robot"].data.default_joint_vel(torch.device("cpu")).shape == (2, 12)
    assert np.isfinite(scene["robot"].data._host_root_link_lin_vel_b).all()
    assert np.isfinite(scene["robot"].data._host_root_link_ang_vel_b).all()
    assert np.isfinite(scene["robot"].data._host_root_link_quat_w[:, 0:1]).all()
    assert bool(torch.isfinite(scene["robot"].data.gravity_vec_w("cpu")).all())
    default_vel = scene["robot"].data.default_joint_vel(torch.device("cpu"))
    torch.testing.assert_close(default_vel, torch.zeros_like(default_vel))


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

    origins = scene.env_origins(torch.device("cpu"))
    assert origins.shape == (3, 3)
    torch.testing.assert_close(origins, torch.zeros_like(origins))
    assert not origins.requires_grad


@pytest.mark.parametrize("excluded_world_rows", [0, 1], ids=["whole-mjcf", "packed-state"])
def test_mapped_multi_entity_body_addresses_use_backend_state_rows(
    excluded_world_rows: int,
) -> None:
    def physical_entity(name: str, first_body_id: int) -> EntityLayout:
        return EntityLayout(
            name=name,
            kind="rigid",
            root_mode="fixed",
            root_body="wrist",
            body_names=("wrist", "finger"),
            body_ids=(first_body_id, first_body_id + 1),
            body_parent_names=(None, "wrist"),
            joints=(),
            actuator_names=(),
            actuator_joint_names=(),
            actuator_indices=(),
        )

    layout = CompiledSceneLayout(
        entities=(physical_entity("left_hand", 1), physical_entity("right_hand", 3)),
        nq=0,
        nv=0,
        nu=0,
        nbody=5,
    )

    class Backend:
        backend_type = "fake"
        num_envs = 2
        num_actuators = 0

        def __init__(self) -> None:
            self.model_ids = {
                f"{entity.name}/{body_name}": body_id
                for entity in layout.entities
                for body_name, body_id in zip(entity.body_names, entity.body_ids, strict=True)
            }
            self.state_rows = {
                name: body_id - excluded_world_rows for name, body_id in self.model_ids.items()
            }
            self.body_pos = np.zeros(
                (self.num_envs, layout.nbody - excluded_world_rows, 3), dtype=np.float32
            )
            for name, row in self.state_rows.items():
                self.body_pos[:, row, 0] = self.model_ids[name]

        def get_tensor_capabilities(self):
            from unisim.backend.base import TensorDataPlane, TensorLifecycleCapabilities

            return TensorLifecycleCapabilities(
                execution=TensorExecution.DEVICE_RESIDENT,
                data_plane=TensorDataPlane.DIRECT,
                stream_event_ownership="test",
                torch_devices=("cpu",),
            )

        def get_entity_default_state(self, name: str):
            return {
                "root_pose": np.tile(
                    np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (self.num_envs, 1),
                ),
                "root_velocity": np.zeros((self.num_envs, 6), dtype=np.float32),
                "joint_positions": np.zeros((self.num_envs, 0), dtype=np.float32),
                "joint_velocities": np.zeros((self.num_envs, 0), dtype=np.float32),
            }

        def get_body_ids(self, names) -> np.ndarray:
            return np.asarray([self.state_rows[name] for name in names], dtype=np.int32)

        def get_tracked_body_views(self, body_names, device=None) -> TrackedBodyStateViews:
            del device
            names = tuple(body_names)
            rows = np.asarray([self.state_rows[name] for name in names], dtype=np.intp)
            quat = np.zeros((self.num_envs, len(names), 4), dtype=np.float32)
            quat[..., 0] = 1.0
            return TrackedBodyStateViews(
                body_names=names,
                pos_w=torch.from_numpy(self.body_pos[:, rows].copy()),
                quat_w=torch.from_numpy(quat),
                lin_vel_w=torch.zeros((self.num_envs, len(names), 3)),
                ang_vel_w=torch.zeros((self.num_envs, len(names), 3)),
            )

    backend = Backend()
    transaction = ResetStateTransaction(cast(SimBackend, backend), scene_layout=layout)
    entities = {}
    for name, first_body_id in (("left_hand", 1), ("right_hand", 3)):
        entities[name] = Entity(
            name,
            EntityCfg(
                root_body_name=f"{name}/wrist",
                body_names=(f"{name}/wrist", f"{name}/finger"),
                physical_entity=name,
            ),
            cast(SimBackend, backend),
            reset_state=transaction,
        )
        expected_rows = np.asarray(
            [first_body_id - excluded_world_rows, first_body_id + 1 - excluded_world_rows],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(entities[name]._root_body_ids, expected_rows[:1])
        np.testing.assert_array_equal(entities[name]._body_ids, expected_rows)
        _, _, reset_rows = entities[name]._bind_body_randomization(
            None, capability="test body reset rows"
        )
        np.testing.assert_array_equal(reset_rows, expected_rows)

        copy_body_state = entities[name].bind_body_state_copy()
        out_pos = np.zeros((2, 2, 3), dtype=np.float32)
        out_quat = np.zeros((2, 2, 4), dtype=np.float32)
        out_lin_vel = np.zeros((2, 2, 3), dtype=np.float32)
        out_ang_vel = np.zeros((2, 2, 3), dtype=np.float32)
        copy_body_state(out_pos, out_quat, out_lin_vel, out_ang_vel)
        np.testing.assert_array_equal(out_pos[0, :, 0], [first_body_id, first_body_id + 1])
