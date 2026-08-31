"""Host-compatibility implementation of the independent ``genesis`` backend.

The adapter serves the ``SimBackend`` NumPy contract on top of Genesis 1.3.3,
following the measured mappings of ``scripts/tools/genesis_feasibility/
REPORT.md`` (#1372): link-addressed root state (never entity-level getters,
REPORT §5.5), ``control_dofs_position`` inside an adapter-owned nsteps loop
honoring ``set_pre_step_control`` (§5.4), host caches refreshed once per
step/reset barrier (§5.9), MJCF-named sensor equivalents from link state plus
one IMUSensor per accelerometer site with clean (noise-free) data (§3.4/§5.3),
genesis-native recoded geom contact masks that must not be compared against
MuJoCo tables (§5.10), and a DR capability set restricted to the per-env
round-trip-measured items (§3.5 [8] / §5.7).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any

import numpy as np

from unilab.base.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    SimBackend,
    normalize_play_render_mode,
)
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unilab.utils.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_mul_batched,
)

from . import dependencies, materialization

_WORLD_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _make_device_cache(torch: Any, shape: tuple[int, ...]) -> tuple[Any, np.ndarray]:
    """One fixed-shape host cache: (pinned storage, zero-copy NumPy view)."""
    pinned = torch.empty(shape, dtype=torch.float32, pin_memory=torch.cuda.is_available())
    return pinned, pinned.numpy()


class GenesisBackend(SimBackend):
    """Independent Genesis backend exposed through the host NumPy profile.

    Construction performs dependency loading, the MJCF cold-path scan, the
    process-wide ``gs.init`` (once), and scene/entity/sensor creation;
    ``materialize()`` builds the batched solver state and binds all runtime
    caches.  Terrain, geom-name contracts, site Jacobians, playback, and
    rendering fail closed.  Call ``close()`` to end the process-wide Genesis
    session; re-initialization afterwards fails closed by design.
    """

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        integrator: str | None = None,
        constraint_solver: str | None = None,
        friction_cone: str | None = None,
        solver_iterations: int | None = None,
        **unexpected_kwargs: Any,
    ) -> None:
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"GenesisBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
        if solver_iterations is not None and (
            isinstance(solver_iterations, bool)
            or not isinstance(solver_iterations, int)
            or solver_iterations <= 0
        ):
            raise ValueError(
                f"genesis solver_iterations must be a positive integer or None, "
                f"got {solver_iterations!r}"
            )
        if push_body_name is not None:
            raise NotImplementedError(
                "genesis backend does not support interval push or external wrench "
                "randomization; remove domain_rand.push_body_name and disable push_robots."
            )

        deps = dependencies.load_genesis_dependencies()
        self._deps = deps
        self._torch = deps.torch
        self._gs = deps.genesis
        self._metadata = materialization.scan_genesis_model_metadata(deps.mujoco, scene)
        self._scene_cleanup_handle = self._metadata.cleanup_handle
        self._scene_model_file = str(scene.model_file)

        # One gs.init per process; re-init after destroy fails closed here.
        materialization.init_genesis_session(deps)
        self._device = (
            self._torch.device("cuda", self._torch.cuda.current_device())
            if (self._torch.cuda.is_available())
            else self._torch.device("cpu")
        )

        self._scene = materialization.build_genesis_scene(
            deps,
            sim_dt=float(sim_dt),
            gravity=self._metadata.gravity,
            integrator=integrator,
            constraint_solver=constraint_solver,
            friction_cone=friction_cone,
            solver_iterations=solver_iterations,
        )
        self._entity = self._scene.add_entity(
            self._gs.morphs.MJCF(file=self._metadata.source_model_file)
        )
        # One IMUSensor per accelerometer site (REPORT §3.4 equivalent); the
        # link index resolves pre-build from the cold-path entity structure.
        self._imu_sensors: dict[str, Any] = {}
        for plan in self._metadata.sensor_plans:
            if plan.kind != "accelerometer" or plan.name in self._imu_sensors:
                continue
            assert plan.site_pos is not None  # guaranteed by the cold-path scan
            link = self._entity.get_link(plan.body_name)
            self._imu_sensors[plan.name] = self._scene.add_sensor(
                self._gs.sensors.IMU(
                    entity_idx=self._entity.idx,
                    link_idx_local=int(link.idx_local),
                    pos_offset=tuple(plan.site_pos),
                )
            )

        self._pre_step_control_fn = None
        self.backend_type = "genesis"
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._base_name = base_name
        # Root dims and name->index maps come from the cold-path MJCF scan
        # (mjwarp-style), so cold metadata like get_default_dof_pos and the
        # joint index getters are correct before materialize(); the
        # materialize-time binding validates the live import against them.
        self._root_qpos_dim = self._metadata.root_qpos_dim
        self._root_qvel_dim = self._metadata.root_qvel_dim
        self._body_ids = {name: idx for idx, name in enumerate(self._metadata.body_names)}
        self._joint_dof_ids = dict(
            zip(self._metadata.joint_names, self._metadata.joint_dof_adrs, strict=True)
        )
        self._joint_qpos_ids = dict(
            zip(self._metadata.joint_names, self._metadata.joint_qpos_adrs, strict=True)
        )
        self._materialized = False
        self._closed = False

    # ------------------------------------------------------------------ #
    # Materialization-time binding                                        #
    # ------------------------------------------------------------------ #

    def materialize(self) -> None:
        """Build the batched scene, cross-check the import, and bind caches."""
        if self._closed:
            raise RuntimeError("genesis backend is closed")
        if self._materialized:
            raise RuntimeError("genesis backend is already materialized")
        self._scene.build(n_envs=self._num_envs)
        self._materialized = True
        self._bind_materialized_metadata()

    def _bind_materialized_metadata(self) -> None:
        metadata = self._metadata
        entity = self._entity
        if int(entity.n_dofs) != metadata.nv or int(entity.n_qs) != metadata.nq:
            raise RuntimeError(
                f"genesis MJCF import mismatch: n_dofs/n_qs {entity.n_dofs}/{entity.n_qs} "
                f"!= MJCF nv/nq {metadata.nv}/{metadata.nq}"
            )
        if int(entity.n_links) != metadata.nbody:
            raise RuntimeError(
                f"genesis MJCF import mismatch: n_links {entity.n_links} != MJCF nbody "
                f"{metadata.nbody}"
            )
        link_names = tuple(str(link.name) for link in entity.links)
        if link_names != metadata.body_names:
            raise RuntimeError(
                f"genesis MJCF import mismatch: link names/order {link_names} != MJCF body "
                f"names {metadata.body_names}"
            )
        one_dof_joints = [joint for joint in entity.joints if int(joint.n_dofs) == 1]
        joint_names = tuple(str(joint.name) for joint in one_dof_joints)
        if joint_names != metadata.joint_names:
            raise RuntimeError(
                f"genesis MJCF import mismatch: joint names/order {joint_names} != MJCF "
                f"single-DoF joints {metadata.joint_names}"
            )
        entity_dof_ids = {
            name: int(joint.dofs_idx_local[0])
            for name, joint in zip(metadata.joint_names, one_dof_joints, strict=True)
        }
        entity_qpos_ids = {
            name: int(joint.qs_idx_local[0])
            for name, joint in zip(metadata.joint_names, one_dof_joints, strict=True)
        }
        if entity_dof_ids != self._joint_dof_ids or entity_qpos_ids != self._joint_qpos_ids:
            raise RuntimeError(
                "genesis MJCF import mismatch: joint dof/qpos indices do not match the "
                "scanned MJCF layout"
            )
        # Actuator order: MJCF actuator-target joints map 1:1 onto actuated
        # dofs (REPORT §3.1 [1b]); gains are cross-checked against the import.
        self._actuated_dofs = [
            self._joint_dof_ids[joint_name] for joint_name in metadata.actuator_joint_names
        ]
        imported_kp = entity.get_dofs_kp().cpu().numpy()[0, self._actuated_dofs]
        imported_kv = entity.get_dofs_kv().cpu().numpy()[0, self._actuated_dofs]
        if not np.allclose(imported_kp, metadata.actuator_kp, atol=1e-4) or not np.allclose(
            imported_kv, metadata.actuator_kv, atol=1e-4
        ):
            raise RuntimeError(
                "genesis MJCF import mismatch: imported dof kp/kv do not match the MJCF "
                "position-actuator gains (REPORT §3.1 [3a] expects PD-reducible gains)."
            )

        self._base_link_idx: int | None = None
        if self._base_name is not None:
            try:
                self._base_link_idx = self._body_ids[self._base_name]
            except KeyError as exc:
                raise ValueError(
                    f"Base body {self._base_name!r} not found in genesis model"
                ) from exc
            root_layout = self.get_root_state_layout(self._base_name)
            if (
                len(root_layout.qpos_indices) != self._root_qpos_dim
                or len(root_layout.qvel_indices) != self._root_qvel_dim
            ):
                raise RuntimeError(
                    f"genesis MJCF import mismatch: base link {self._base_name!r} root "
                    "layout does not match the scanned free-root block"
                )
        self._link_start = int(entity.link_start)

        self._sensor_slots, sensor_constants, total_dim = self._bind_sensor_slots()
        self._sensor_constants = sensor_constants

        torch = self._torch
        n = self._num_envs
        self._qpos_cache = _make_device_cache(torch, (n, metadata.nq))
        self._qvel_cache = _make_device_cache(torch, (n, metadata.nv))
        self._links_pos_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
        self._links_quat_cache = _make_device_cache(torch, (n, metadata.nbody, 4))
        self._links_vel_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
        self._links_ang_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
        self._contact_force_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
        self._sensor_cache = np.zeros((n, total_dim), dtype=np.float32)
        self._imu_caches = {name: _make_device_cache(torch, (n, 3)) for name in self._imu_sensors}
        self._time_cache = np.zeros((n,), dtype=np.float32)
        self._refresh_host_cache()

    def _bind_sensor_slots(self) -> tuple[dict[str, tuple[int, int]], dict[str, tuple], int]:
        slots: dict[str, tuple[int, int]] = {}
        constants: dict[str, tuple] = {}
        address = 0
        for plan in self._metadata.sensor_plans:
            if plan.body_name not in self._body_ids:
                raise RuntimeError(
                    f"genesis sensor {plan.name!r} references missing body {plan.body_name!r}"
                )
            link_idx = self._body_ids[plan.body_name]
            slots[plan.name] = (address, plan.dim)
            if plan.kind == "contact":
                constants[plan.name] = (link_idx,)
            else:
                assert plan.site_pos is not None and plan.site_quat is not None
                constants[plan.name] = (
                    link_idx,
                    np.asarray(plan.site_pos, dtype=np.float64),
                    np.asarray(plan.site_quat, dtype=np.float64),
                )
            address += plan.dim
        return slots, constants, address

    # ------------------------------------------------------------------ #
    # Host-cache barriers                                                 #
    # ------------------------------------------------------------------ #

    def _require_running(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(f"genesis backend is closed; cannot run {operation}")
        if not self._materialized:
            raise RuntimeError(
                f"genesis backend is not materialized; call materialize() before {operation}"
            )

    def _refresh_host_cache(self) -> None:
        """Refresh every legacy-visible cache at one explicit lifecycle barrier."""
        entity = self._entity
        self._qpos_cache[0].copy_(entity.get_qpos())
        self._qvel_cache[0].copy_(entity.get_dofs_velocity())
        self._links_pos_cache[0].copy_(entity.get_links_pos())
        self._links_quat_cache[0].copy_(entity.get_links_quat())
        self._links_vel_cache[0].copy_(entity.get_links_vel())
        self._links_ang_cache[0].copy_(entity.get_links_ang())
        self._contact_force_cache[0].copy_(entity.get_links_net_contact_force())
        for name, sensor in self._imu_sensors.items():
            self._imu_caches[name][0].copy_(sensor.read().lin_acc)
        self._refresh_sensor_cache()

    def _refresh_sensor_cache(self) -> None:
        """Compute MJCF-named sensors from link caches (REPORT §3.4 mappings)."""
        for plan in self._metadata.sensor_plans:
            address, dim = self._sensor_slots[plan.name]
            out = self._sensor_cache[:, address : address + dim]
            if plan.kind == "contact":
                (link_idx,) = self._sensor_constants[plan.name]
                force = self._contact_force_cache[1][:, link_idx, :]
                magnitude = np.linalg.norm(force, axis=-1, keepdims=True)
                threshold = materialization.CONTACT_FOUND_FORCE_THRESHOLD_N
                out[...] = (magnitude > threshold).astype(np.float32)
                continue
            if plan.kind == "accelerometer":
                out[...] = self._imu_caches[plan.name][1]
                continue
            link_idx, site_pos, site_quat = self._sensor_constants[plan.name]
            link_quat = self._links_quat_cache[1][:, link_idx, :]
            batch3 = link_quat.shape[:-1] + (3,)
            site_quat_w = np_quat_mul_batched(
                link_quat, np.broadcast_to(site_quat, link_quat.shape)
            )
            if plan.kind == "gyro":
                out[...] = np_quat_apply_inverse_batched(
                    site_quat_w, self._links_ang_cache[1][:, link_idx, :]
                )
            elif plan.kind == "framequat":
                out[...] = site_quat_w
            elif plan.kind == "framezaxis":
                out[...] = np_quat_apply_batched(
                    site_quat_w, np.broadcast_to(_WORLD_Z_AXIS, batch3)
                )
            else:
                offset_w = np_quat_apply_batched(link_quat, np.broadcast_to(site_pos, batch3))
                if plan.kind == "velocimeter":
                    lin_vel_w = self._links_vel_cache[1][:, link_idx, :] + np.cross(
                        self._links_ang_cache[1][:, link_idx, :], offset_w
                    )
                    out[...] = np_quat_apply_inverse_batched(site_quat_w, lin_vel_w)
                elif plan.kind == "framepos":
                    out[...] = self._links_pos_cache[1][:, link_idx, :] + offset_w

    def _to_device(self, array: np.ndarray) -> Any:
        host = np.ascontiguousarray(array, dtype=np.float32)
        return self._torch.from_numpy(host).to(self._device)

    def _validate_rows(self, env_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_indices, dtype=np.intp)
        if rows.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got shape {rows.shape}")
        if np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError(f"env_indices must be in [0, {self._num_envs}), got {rows}")
        if np.unique(rows).size != rows.size:
            raise ValueError("env_indices must not contain duplicate rows")
        return rows

    # ------------------------------------------------------------------ #
    # SimBackend properties and cold metadata                             #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> Any:
        """Return the backend-owned Genesis rigid entity."""
        return self._entity

    @property
    def num_actuators(self) -> int:
        return len(self._metadata.actuator_names)

    @property
    def num_dof_vel(self) -> int:
        return self._metadata.nv - self._root_qvel_dim

    def get_actuator_ctrl_range(self) -> np.ndarray:
        """MJCF ``ctrlrange`` metadata; Genesis does not enforce it in-engine."""
        return self._metadata.actuator_ctrl_range.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        return self._metadata.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self._metadata.actuator_joint_names

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return self._metadata.actuator_kp.copy(), self._metadata.actuator_kv.copy()

    def get_scene_model_file(self) -> str | None:
        return self._scene_model_file

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        keyframes = dict(self._metadata.keyframe_qpos)
        try:
            return keyframes[name].copy()
        except KeyError as exc:
            available = ", ".join(sorted(keyframes))
            raise ValueError(f"Keyframe {name!r} not found; available: {available}") from exc

    def get_default_qpos(self) -> np.ndarray:
        return self._metadata.default_qpos.copy()

    def get_default_dof_pos(self) -> np.ndarray:
        return self._metadata.default_qpos[self._root_qpos_dim :].copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._metadata.nv,), dtype=np.float32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if root_body_name not in self._body_ids:
            raise ValueError(f"Body {root_body_name!r} not found in genesis model")
        link = self._entity.get_link(root_body_name)
        free_joints = [
            joint for joint in link.joints if int(joint.n_dofs) == 6 and int(joint.n_qs) == 7
        ]
        if len(free_joints) != 1:
            raise NotImplementedError(
                "backend 'genesis' capability 'root-state layout' requires body "
                f"{root_body_name!r} to own exactly one free joint"
            )
        joint = free_joints[0]
        return BackendRootStateLayout(
            qpos_indices=tuple(int(v) for v in joint.qs_idx_local),
            qvel_indices=tuple(int(v) for v in joint.dofs_idx_local),
        )

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._body_ids[str(name)])
            except KeyError as exc:
                raise ValueError(f"Body {name!r} not found in genesis model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Genesis-native recoded contype/conaffinity of collision geoms.

        Genesis re-synthesizes contype/conaffinity at import: the collision
        matrix semantics are preserved, but the integer values must NOT be
        compared against MuJoCo tables (REPORT #1372 §5.10).
        """
        self._require_running("get_geom_contact_masks")
        return (
            np.asarray([geom.contype for geom in self._entity.geoms], dtype=np.int32),
            np.asarray([geom.conaffinity for geom in self._entity.geoms], dtype=np.int32),
        )

    def get_gravity(self) -> np.ndarray:
        return self._metadata.gravity.copy()

    def get_body_mass(self) -> np.ndarray:
        return self._metadata.body_mass.copy()

    def get_body_ipos(self) -> np.ndarray:
        return self._metadata.body_ipos.copy()

    def get_dof_armature(self) -> np.ndarray:
        return self._metadata.dof_armature.copy()

    def get_joint_range(self) -> np.ndarray | None:
        joint_range = self._metadata.joint_range
        return None if joint_range is None else joint_range.copy()

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        return np.asarray(self._resolve_joint_ids(names, self._joint_dof_ids), dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        qpos_ids = np.asarray(self._resolve_joint_ids(names, self._joint_qpos_ids))
        return (qpos_ids - self._root_qpos_dim).astype(np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_indices(names) - self._root_qvel_dim

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names) + self._root_qpos_dim

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_vel_indices(names) + self._root_qvel_dim

    def _resolve_joint_ids(self, names: Sequence[str], table: dict[str, int]) -> list[int]:
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(table[str(name)])
            except KeyError as exc:
                raise ValueError(f"Joint {name!r} not found in genesis model") from exc
        return resolved

    # ------------------------------------------------------------------ #
    # Simulation control                                                  #
    # ------------------------------------------------------------------ #

    def _push_control(self, ctrl: np.ndarray) -> None:
        self._entity.control_dofs_position(
            self._to_device(ctrl), dofs_idx_local=self._actuated_dofs
        )

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
        self._require_running("step")
        if isinstance(nsteps, bool) or int(nsteps) <= 0:
            raise ValueError(f"nsteps must be a positive integer, got {nsteps!r}")
        ctrl_array = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, self.num_actuators)
        if ctrl_array.shape != expected:
            raise ValueError(f"ctrl must have shape {expected}, got {ctrl_array.shape}")

        t0 = time.perf_counter()
        if self._pre_step_control_fn is None:
            # control_dofs_position holds the target across scene.step() calls,
            # matching MuJoCo ctrl-broadcast semantics (REPORT §3.3 [3b]).
            self._push_control(ctrl_array)
            for _ in range(int(nsteps)):
                self._scene.step()
        else:
            # Adapter-side per-substep hook: the conversion reads the freshly
            # refreshed sensor contract before every physics substep (REPORT
            # §5.4); no new SimBackend surface is introduced.
            for _ in range(int(nsteps)):
                converted = self._apply_pre_step_control(ctrl_array)
                self._push_control(converted)
                self._scene.step()
                self._refresh_host_cache()
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_host_cache()
        self._time_cache += np.float32(int(nsteps) * self._sim_dt)
        host_cache_ms = (time.perf_counter() - t0) * 1000.0
        return {"timing": {"physics_ms": physics_ms, "host_cache_refresh_ms": host_cache_ms}}

    # All backends report the same set_state key set for column stability;
    # sub-keys that don't apply to the genesis host profile report 0.0.
    _SET_STATE_TIMING_ZERO_KEYS = (
        "set_state_mask_ms",
        "set_state_data_slice_ms",
        "set_state_data_reset_ms",
        "set_state_clear_forces_ms",
        "set_state_geom_overrides_ms",
        "set_state_reset_rand_ms",
        "set_state_set_dof_vel_ms",
        "set_state_set_dof_pos_ms",
        "set_state_actuator_ctrl_ms",
        "set_state_forward_kinematic_ms",
        "set_state_refresh_pose_cache_ms",
        "set_state_invalidate_velocity_ms",
        "set_state_qpos_convert_ms",
        "set_state_pool_reset_ms",
        "set_state_state_scatter_ms",
    )
    _SET_STATE_TIMING_OWN_KEYS = (
        "set_state_reset_upload_ms",
        "set_state_reset_forward_ms",
        "set_state_host_cache_refresh_ms",
        "set_state_internal_gap_ms",
    )

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict[str, dict[str, float]]:
        self._require_running("set_state")
        rows = self._validate_rows(env_indices)
        qpos_array = np.asarray(qpos, dtype=np.float32)
        qvel_array = np.asarray(qvel, dtype=np.float32)
        expected_qpos = (rows.size, self._metadata.nq)
        expected_qvel = (rows.size, self._metadata.nv)
        if qpos_array.shape != expected_qpos:
            raise ValueError(f"qpos must have shape {expected_qpos}, got {qpos_array.shape}")
        if qvel_array.shape != expected_qvel:
            raise ValueError(f"qvel must have shape {expected_qvel}, got {qvel_array.shape}")
        timing: dict[str, float] = {
            key: 0.0 for key in self._SET_STATE_TIMING_ZERO_KEYS + self._SET_STATE_TIMING_OWN_KEYS
        }
        if rows.size == 0:
            return {"timing": timing}

        outer_t0 = time.perf_counter()
        envs_idx = rows.tolist()
        t0 = time.perf_counter()
        # set_qpos runs forward kinematics for the touched envs, so positions
        # are immediately readable afterwards (REPORT §5.6).
        self._entity.set_qpos(self._to_device(qpos_array), envs_idx=envs_idx, zero_velocity=False)
        self._entity.set_dofs_velocity(self._to_device(qvel_array), envs_idx=envs_idx)
        timing["set_state_reset_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if randomization is not None and not randomization.is_empty():
            self._apply_reset_randomization(randomization, rows)
        self._refresh_host_cache()
        self._time_cache[rows] = 0.0
        timing["set_state_host_cache_refresh_ms"] = (time.perf_counter() - t0) * 1000.0

        measured_ms = (
            timing["set_state_reset_upload_ms"]
            + timing["set_state_reset_forward_ms"]
            + timing["set_state_host_cache_refresh_ms"]
        )
        total_ms = (time.perf_counter() - outer_t0) * 1000.0
        timing["set_state_internal_gap_ms"] = total_ms - measured_ms
        return {"timing": timing}

    # ------------------------------------------------------------------ #
    # Domain randomization (REPORT §3.5 [8] measured items only)          #
    # ------------------------------------------------------------------ #

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Declare only the per-env round-trip-measured DR items (REPORT §5.7).

        Measured: link inertial mass and dof kp/kv (require the materialize-
        time batch build flags), plus the solver-level external force API
        (call-verified; physical effect is a REPORT §8 follow-up).  Measured-
        but-unmappable items stay undeclared: frictionloss/damping/armature
        have no SimBackend reset term, and geom friction only has a per-env
        *ratio* API, so absolute geom_friction randomization is unsupported.
        """
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                {RESET_TERM_BODY_MASS, RESET_TERM_BASE_MASS, RESET_TERM_KP, RESET_TERM_KD}
            ),
            supports_interval_body_force=True,
        )

    _UNSUPPORTED_RESET_TERMS = (
        "gravity",
        "body_iquat",
        "body_inertia",
        "body_ipos",
        "base_com_offset",
        "dof_armature",
        "geom_friction",
    )

    def _apply_reset_randomization(
        self, randomization: ResetRandomizationPayload, rows: np.ndarray
    ) -> None:
        unsupported = [
            term
            for term in self._UNSUPPORTED_RESET_TERMS
            if getattr(randomization, term) is not None
        ]
        if unsupported:
            raise NotImplementedError(
                "genesis backend does not support reset domain randomization terms: "
                f"{', '.join(sorted(unsupported))} (REPORT #1372 §5.7 declares only the "
                "measured items)."
            )
        envs_idx = rows.tolist()
        body_mass = randomization.body_mass
        if randomization.base_mass_delta is not None:
            if self._base_link_idx is None:
                raise ValueError(
                    "genesis base_mass_delta randomization requires base_name to identify "
                    "the base link"
                )
            delta = np.asarray(randomization.base_mass_delta, dtype=np.float32).reshape(-1)
            if delta.shape != (rows.size,):
                raise ValueError(
                    f"base_mass_delta must have shape ({rows.size},), got {delta.shape}"
                )
            if body_mass is None:
                body_mass = np.broadcast_to(
                    self._metadata.body_mass, (rows.size, self._metadata.nbody)
                ).copy()
            else:
                body_mass = np.asarray(body_mass, dtype=np.float32).copy()
            body_mass[:, self._base_link_idx] += delta
        if body_mass is not None:
            mass = np.asarray(body_mass, dtype=np.float32)
            expected = (rows.size, self._metadata.nbody)
            if mass.shape != expected:
                raise ValueError(f"body_mass must have shape {expected}, got {mass.shape}")
            self._entity.set_links_inertial_mass(self._to_device(mass), envs_idx=envs_idx)
        for value, name, setter in (
            (randomization.kp, "kp", self._entity.set_dofs_kp),
            (randomization.kd, "kd", self._entity.set_dofs_kv),
        ):
            if value is None:
                continue
            gains = np.asarray(value, dtype=np.float32)
            expected = (rows.size, self.num_actuators)
            if gains.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {gains.shape}")
            setter(
                self._to_device(gains),
                dofs_idx_local=self._actuated_dofs,
                envs_idx=envs_idx,
            )

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        self._require_running("apply_interval_randomization")
        if plan.is_empty():
            return
        if plan.push_perturbation_limit is not None:
            raise NotImplementedError(
                "genesis backend does not support interval push perturbation; disable "
                "push_robots in the owner YAML."
            )
        if plan.body_linear_velocity_delta is not None:
            raise NotImplementedError(
                "genesis backend does not support interval body-velocity deltas; disable "
                "the term in the owner YAML."
            )
        if plan.body_force is not None:
            if plan.body_ids is None:
                raise ValueError("Interval body-force perturbation requires body_ids")
            self.apply_body_force(plan.body_ids, plan.body_force)

    def apply_body_force(self, body_ids: np.ndarray, force: np.ndarray) -> None:
        """Apply a world-frame force per body through the solver-level API."""
        self._require_running("apply_body_force")
        ids = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        force_array = np.asarray(force, dtype=np.float32)
        expected = (self._num_envs, ids.size, 3)
        if force_array.shape != expected:
            raise ValueError(f"body force must have shape {expected}, got {force_array.shape}")
        if np.any(ids < 0) or np.any(ids >= self._metadata.nbody):
            raise ValueError(f"body_ids must be in [0, {self._metadata.nbody}), got {ids}")
        solver = self._scene.sim.rigid_solver
        force_device = self._to_device(force_array)
        for offset, body_id in enumerate(ids):
            # Solver-level API uses global link indices (REPORT §3.5 [8]);
            # single-entity scenes keep global == link_start + local.
            solver.apply_links_external_force(
                force_device[:, offset, :],
                links_idx=[self._link_start + int(body_id)],
            )

    # ------------------------------------------------------------------ #
    # Play/render: declared honestly, fail closed                          #
    # ------------------------------------------------------------------ #

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities()

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        if mode == "none":
            return BackendPlayRenderPlan(
                mode="none",
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        raise NotImplementedError(
            f"genesis backend does not support playback mode {mode!r}; Genesis native "
            "rendering/playback is not a declared capability of this adapter (offscreen "
            "camera rendering is a measured-but-unintegrated follow-up, REPORT #1372 "
            "§3.5 [12]). Select training.play_render_mode=none."
        )

    # ------------------------------------------------------------------ #
    # Legacy getters: cache views only, never direct device transfers     #
    # ------------------------------------------------------------------ #

    def _require_free_root(self, operation: str) -> None:
        self._require_running(operation)
        if self._root_qpos_dim != 7 or self._root_qvel_dim != 6:
            raise NotImplementedError(
                f"{operation} requires a free root joint; genesis host profile is "
                "currently validated only for floating-base layouts."
            )

    def get_base_pos(self) -> np.ndarray:
        self._require_free_root("get_base_pos")
        return self._links_pos_cache[1][:, self._base_link_idx, :]

    def get_base_quat(self) -> np.ndarray:
        self._require_free_root("get_base_quat")
        return self._links_quat_cache[1][:, self._base_link_idx, :]

    def get_base_lin_vel(self) -> np.ndarray:
        self._require_free_root("get_base_lin_vel")
        # qvel[0:3] is the root linear velocity in world coordinates and stays
        # valid immediately after set_state (REPORT §5.6).
        return self._qvel_cache[1][:, 0:3]

    def get_base_ang_vel(self) -> np.ndarray:
        self._require_free_root("get_base_ang_vel")
        # qvel[3:6] is body-frame angular velocity; the contract wants world
        # frame.  Deriving it from qvel keeps the value fresh after reset
        # (genesis link velocity getters only refresh across a step barrier,
        # REPORT §5.6); it equals get_links_ang(root) after a step.
        return np_quat_apply_batched(self.get_base_quat(), self._qvel_cache[1][:, 3:6])

    def get_dof_pos(self) -> np.ndarray:
        self._require_running("get_dof_pos")
        return self._qpos_cache[1][:, self._root_qpos_dim :]

    def get_dof_vel(self) -> np.ndarray:
        self._require_running("get_dof_vel")
        return self._qvel_cache[1][:, self._root_qvel_dim :]

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_running("get_body_pos_w")
        return self._links_pos_cache[1][:, np.asarray(body_ids, dtype=np.intp), :]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_running("get_body_quat_w")
        return self._links_quat_cache[1][:, np.asarray(body_ids, dtype=np.intp), :]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_running("get_body_lin_vel_w")
        return self._links_vel_cache[1][:, np.asarray(body_ids, dtype=np.intp), :]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_running("get_body_ang_vel_w")
        return self._links_ang_cache[1][:, np.asarray(body_ids, dtype=np.intp), :]

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_base_frame_kinematics("base-frame body positions")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_base_frame_kinematics("base-frame body orientations")

    def _unsupported_base_frame_kinematics(self, operation: str) -> None:
        # Same boundary as the mjwarp host profile: no task consumes
        # base-frame body pose, and the tracked subsets live in sensors.
        raise NotImplementedError(f"genesis host profile does not expose {operation}")

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Analytical per the SimBackend contract (#1254): world-frame velocity
        # rotated into each body's own frame.
        ids = np.asarray(body_ids, dtype=np.intp)
        return np_quat_apply_inverse_batched(
            self._links_quat_cache[1][:, ids, :], self._links_vel_cache[1][:, ids, :]
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(body_ids, dtype=np.intp)
        return np_quat_apply_inverse_batched(
            self._links_quat_cache[1][:, ids, :], self._links_ang_cache[1][:, ids, :]
        )

    def get_sensor_data(self, name: str) -> np.ndarray:
        self._require_running(f"get_sensor_data({name!r})")
        try:
            address, dimension = self._sensor_slots[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._sensor_slots))
            raise ValueError(f"Sensor {name!r} not found; available: {available}") from exc
        return self._sensor_cache[:, address : address + dimension]

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> Callable[[], np.ndarray]:
        """Capture numeric host-cache slots for a zero-metadata hot-path view."""
        slots = tuple(self._sensor_slots[name] for name in names)

        def read() -> np.ndarray:
            values = [
                self._sensor_cache[:, address : address + dimension] for address, dimension in slots
            ]
            return np.concatenate(values, axis=1)

        return read

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """End the process-wide Genesis session; re-init afterwards fails closed."""
        if self._closed:
            return
        self._closed = True
        materialization.destroy_genesis_session(self._deps)
