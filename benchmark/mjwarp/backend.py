"""mjwarp ``SimBackend`` adapter used only by ``benchmark/benchmark_env_step.py``.

The goal is parity with ``MuJoCoBackend`` / ``MotrixBackend`` at the
``NpEnv.step`` boundary so the three backends share the same Python pipeline
(``apply_action`` / ``update_state`` / ``reset_done`` all stay on CPU numpy).
The only difference is what happens inside ``backend.step``: H2D ctrl,
``mujoco_warp.step`` on GPU, D2H of qpos/qvel/sensors. Those three segments
are reported in the timing dict so the benchmark's existing breakdown columns
work without changes.

This backend covers what ``g1_walk_flat`` actually invokes. Methods the
benchmark does not exercise raise ``NotImplementedError``. Phase 2 will
broaden the surface for other tasks.

Per-env actuator gain (``kp`` / ``kd``) randomization is NOT supported: in
mjwarp the model parameters are shared across the batched ``Data`` worlds.
``get_dr_capabilities`` reports ``supports_kp = supports_kd = False`` so the
incoming ``ResetRandomizationPayload`` is treated as a no-op for those terms.
This is acceptable for benchmark-only env_step timing comparison; it does not
change the physics pipeline being measured.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from unilab.base.backend.base import BackendHeightScanner, SimBackend
from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)


def _import_mujoco():
    import mujoco

    return mujoco


def _import_mjwarp():
    import mujoco_warp
    import warp as wp

    return mujoco_warp, wp


class _MjwarpHeightScanner(BackendHeightScanner):
    """GPU-accelerated height scanner using mujoco_warp.rays()."""

    def __init__(
        self,
        backend: "MjwarpBackend",
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str,
        output: str,
    ) -> None:
        self._backend = backend
        self._hfield_geom_id = hfield_geom_id
        self._offsets = np.asarray(offsets, dtype=np.float64)  # (num_points, 2)
        self._num_points = self._offsets.shape[0]
        self._frame_body_id = frame_body_id
        self._alignment = alignment
        self._output = output
        self._ray_height = 50.0

        wp = backend._wp
        n = backend._num_envs
        nray = self._num_points
        from mujoco_warp._src.types import vec6f

        self._vec6f = vec6f
        self._pnt_buf = wp.zeros((n, nray), dtype=wp.vec3f)
        self._vec_buf = wp.zeros((n, nray), dtype=wp.vec3f)
        self._dist_buf = wp.zeros((n, nray), dtype=wp.float32)
        self._geomid_buf = wp.zeros((n, nray), dtype=wp.int32)
        self._normal_buf = wp.zeros((n, nray), dtype=wp.vec3f)
        self._bodyexclude = wp.full((nray,), frame_body_id, dtype=wp.int32)

        # Pre-compute constant ray direction (0, 0, -1) for all envs/rays
        vec_np = np.zeros((n, nray, 3), dtype=np.float32)
        vec_np[:, :, 2] = -1.0
        self._vec_buf.assign(vec_np)

    def scan(self) -> np.ndarray:
        backend = self._backend
        wp = backend._wp

        base_pos = backend._cache_qpos[:, 0:3]  # (num_envs, 3)
        base_quat = backend._cache_qpos[:, 3:7]  # (num_envs, 4) wxyz

        # Extract yaw from quaternion (w, x, y, z)
        w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        cos_yaw = np.cos(yaw)[:, None]  # (num_envs, 1)
        sin_yaw = np.sin(yaw)[:, None]

        # Rotate offsets by yaw: R @ [ox, oy]^T
        ox = self._offsets[:, 0]  # (num_points,)
        oy = self._offsets[:, 1]
        # Broadcast: (num_envs, 1) * (1, num_points) → (num_envs, num_points)
        rx = cos_yaw * ox[None, :] - sin_yaw * oy[None, :]
        ry = sin_yaw * ox[None, :] + cos_yaw * oy[None, :]

        # Ray origins: base_pos + rotated offset + z_offset
        n = backend._num_envs
        nray = self._num_points
        pnt_np = np.zeros((n, nray, 3), dtype=np.float32)
        pnt_np[:, :, 0] = (base_pos[:, 0:1] + rx).astype(np.float32)
        pnt_np[:, :, 1] = (base_pos[:, 1:2] + ry).astype(np.float32)
        pnt_np[:, :, 2] = (base_pos[:, 2:3] + self._ray_height).astype(np.float32)

        self._pnt_buf.assign(pnt_np)
        geomgroup = self._vec6f(-1, -1, -1, -1, -1, -1)

        backend._mjwarp.rays(
            backend._mjw_model,
            backend._mjw_data,
            self._pnt_buf,
            self._vec_buf,
            geomgroup,
            True,
            self._bodyexclude,
            self._dist_buf,
            self._geomid_buf,
            self._normal_buf,
        )
        wp.synchronize_device()

        dist = self._dist_buf.numpy()  # (num_envs, num_points)
        # Height = origin_z - dist. Where dist == -1 (no hit), use 0.
        origin_z = pnt_np[:, :, 2]
        heights = np.where(dist >= 0, origin_z - dist, 0.0)
        return heights.astype(np.float64)


class MjwarpBackend(SimBackend):
    """Minimal mjwarp adapter for benchmark env_step timing.

    Implements the ``SimBackend`` surface that ``g1_walk_flat`` exercises during
    a benchmark run; everything else raises ``NotImplementedError``.
    """

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        position_actuator_gains: Any = None,
        **_ignored_kwargs: Any,
    ) -> None:
        del position_actuator_gains, _ignored_kwargs

        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None
        self.backend_type = "mjwarp"

        if scene is None or not scene.model_file:
            raise ValueError("MjwarpBackend requires a SceneCfg with a model_file")

        mujoco = _import_mujoco()
        mujoco_warp, wp = _import_mjwarp()
        self._mujoco = mujoco
        self._wp = wp
        self._mjwarp = mujoco_warp

        # Scene materialization: three paths depending on terrain / fragments.
        self.terrain_origins = None
        self.terrain_surface_sampler = None
        self._terrain_tmpdir = None

        if scene.terrain is not None:
            import tempfile

            from unilab.base.backend.mujoco.xml import (
                materialize_mujoco_hfield_attached_scene,
            )

            tmpdir = tempfile.TemporaryDirectory(prefix="mjwarp_terrain_")
            self._terrain_tmpdir = tmpdir
            result = materialize_mujoco_hfield_attached_scene(
                model_file=scene.model_file,
                terrain_cfg=scene.terrain.generator,
                output_dir=tmpdir.name,
                fragment_files=scene.fragment_files or [],
                hfield_name=scene.terrain.hfield_name,
                geom_name=scene.terrain.geom_name or "floor",
                return_surface_sampler=True,
            )
            self._mj_model, self.terrain_origins, self.terrain_surface_sampler = result
        elif scene.fragment_files:
            from unilab.base.backend.mujoco.xml import materialize_scene_fragments

            xml_path = materialize_scene_fragments(
                scene.model_file, fragment_files=scene.fragment_files
            )
            self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            self._mj_model = mujoco.MjModel.from_xml_path(scene.model_file)

        self._mj_model.opt.timestep = float(sim_dt)

        self._base_name = base_name
        self._push_body_name = push_body_name

        self._num_envs = int(num_envs)
        self._nq = int(self._mj_model.nq)
        self._nv = int(self._mj_model.nv)
        self._nu = int(self._mj_model.nu)

        # Free-joint slice assumption: G1 (and other locomotion tasks) place the
        # free joint at index 0, so DOF state for the env starts after it.
        # Cold-path verification at init avoids per-step hasattr probing.
        if self._mj_model.njnt == 0:
            raise ValueError("Model has no joints")
        first_joint_type = int(self._mj_model.jnt_type[0])
        # mujoco.mjtJoint.mjJNT_FREE == 0
        self._has_free_joint = first_joint_type == 0
        if self._has_free_joint:
            self._dof_qpos_start = 7
            self._dof_qvel_start = 6
            expected_dof = self._nq - 7
            if expected_dof != self._nu:
                raise ValueError(
                    f"DOF count mismatch: nq-7={expected_dof} but nu={self._nu}"
                )
        else:
            self._dof_qpos_start = 0
            self._dof_qvel_start = 0

        # Build sensor-name -> (adr, dim) map once on the cold path.
        self._sensor_adr: dict[str, tuple[int, int]] = {}
        for sid in range(int(self._mj_model.nsensor)):
            name = mujoco.mj_id2name(
                self._mj_model, int(mujoco.mjtObj.mjOBJ_SENSOR), sid
            )
            if name is None:
                continue
            self._sensor_adr[str(name)] = (
                int(self._mj_model.sensor_adr[sid]),
                int(self._mj_model.sensor_dim[sid]),
            )

        # Cache keyframe qpos (CPU numpy) for reset.
        self._keyframe_qpos: dict[str, np.ndarray] = {}
        for kid in range(int(self._mj_model.nkey)):
            kname = mujoco.mj_id2name(
                self._mj_model, int(mujoco.mjtObj.mjOBJ_KEY), kid
            )
            if kname is None:
                continue
            self._keyframe_qpos[str(kname)] = np.asarray(
                self._mj_model.key_qpos[kid], dtype=np.float64
            )

        # Cache actuator ctrl range and joint ranges for the env's cold path.
        self._actuator_ctrl_range = np.asarray(
            self._mj_model.actuator_ctrlrange, dtype=np.float64
        )

        # Upload to GPU. ``njmax`` controls per-world constraint capacity;
        # the default is too small for legged robots with multiple feet making
        # ground contact, producing ``nefc overflow`` warnings and numerical
        # corruption. 512 is a comfortable upper bound for g1_walk_flat
        # (observed max around 175 across 512 envs in early-run probes).
        self._mjw_model = mujoco_warp.put_model(self._mj_model)
        self._mjw_data = mujoco_warp.make_data(
            self._mj_model,
            nworld=self._num_envs,
            njmax=512,
            nconmax=512,
        )

        # Host-side cache buffers refreshed at the end of every step.
        self._cache_qpos = np.zeros((self._num_envs, self._nq), dtype=np.float64)
        self._cache_qvel = np.zeros((self._num_envs, self._nv), dtype=np.float64)
        self._cache_sensordata = np.zeros(
            (self._num_envs, int(self._mj_model.nsensordata)), dtype=np.float64
        )

        # Seed qpos to the first keyframe so the env's init_state has something
        # consistent to reset from. NpEnv's init_state marks every env as
        # terminated so reset() will overwrite these immediately, but writing a
        # valid pose here avoids a transient bad state before that runs.
        if self._mj_model.nkey > 0:
            init_qpos = np.tile(
                np.asarray(self._mj_model.key_qpos[0], dtype=np.float32),
                (self._num_envs, 1),
            )
            self._mjw_data.qpos.assign(init_qpos)
            self._wp.synchronize_device()

        # Pre-allocated H2D scratch for ctrl to avoid per-step warp.array
        # allocation in the hot path.
        self._ctrl_scratch_f32 = np.zeros((self._num_envs, self._nu), dtype=np.float32)

        # Resolve push body id (used by DR interval push for tasks like go1).
        # ``push_body_name`` falls back to ``base_name``; -1 means push is
        # unsupported for this env. Matches MuJoCoBackend semantics.
        push_target = self._push_body_name if self._push_body_name is not None else self._base_name
        if push_target is None:
            self._push_body_id = -1
        else:
            body_id = mujoco.mj_name2id(
                self._mj_model, int(mujoco.mjtObj.mjOBJ_BODY), push_target
            )
            if body_id < 0:
                raise ValueError(
                    f"Push body {push_target!r} not found in mjwarp model"
                )
            self._push_body_id = int(body_id)

        # Pending external wrench buffer ``(num_envs, nbody, 6)`` written into
        # ``data.xfrc_applied`` at the start of every ``step`` and cleared
        # afterwards. mujoco_warp's xfrc_applied dtype is spatial_vectorf
        # (6 floats: linear xyz + angular xyz).
        self._nbody = int(self._mj_model.nbody)
        self._pending_xfrc_applied = np.zeros(
            (self._num_envs, self._nbody, 6), dtype=np.float32
        )
        self._xfrc_dirty = False  # avoid H2D when no force is pending

    # ------------------------------------------------------------------ #
    # Hot-path: step                                                       #
    # ------------------------------------------------------------------ #

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        ctrl = self._apply_pre_step_control(ctrl)
        wp = self._wp

        # H2D copy of ctrl (and pending external wrench if any).
        t0 = time.perf_counter()
        np.copyto(self._ctrl_scratch_f32, np.asarray(ctrl, dtype=np.float32))
        self._mjw_data.ctrl.assign(self._ctrl_scratch_f32)
        if self._xfrc_dirty:
            self._mjw_data.xfrc_applied.assign(self._pending_xfrc_applied)
        wp.synchronize_device()
        set_ctrl_ms = (time.perf_counter() - t0) * 1000.0

        # GPU physics.
        t0 = time.perf_counter()
        for _ in range(int(nsteps)):
            self._mjwarp.step(self._mjw_model, self._mjw_data)
        wp.synchronize_device()
        physics_ms = (time.perf_counter() - t0) * 1000.0

        # Clear pending external wrench so the next step starts clean. MuJoCo's
        # xfrc_applied is sticky across steps; the env layer expects it to be
        # consumed and reset by the backend.
        if self._xfrc_dirty:
            self._pending_xfrc_applied.fill(0.0)
            self._mjw_data.xfrc_applied.assign(self._pending_xfrc_applied)
            self._xfrc_dirty = False

        # D2H refresh of qpos / qvel / sensordata into pre-allocated host buffers.
        t0 = time.perf_counter()
        self._refresh_cache()
        wp.synchronize_device()
        refresh_cache_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "timing": {
                "set_ctrl_ms": set_ctrl_ms,
                "physics_ms": physics_ms,
                "refresh_cache_ms": refresh_cache_ms,
            }
        }

    def _refresh_cache(self) -> None:
        np.copyto(self._cache_qpos, self._mjw_data.qpos.numpy().astype(np.float64, copy=False))
        np.copyto(self._cache_qvel, self._mjw_data.qvel.numpy().astype(np.float64, copy=False))
        np.copyto(
            self._cache_sensordata,
            self._mjw_data.sensordata.numpy().astype(np.float64, copy=False),
        )

    # ------------------------------------------------------------------ #
    # Properties                                                            #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> Any:
        return self._mjw_model

    @property
    def num_actuators(self) -> int:
        return self._nu

    @property
    def num_dof_vel(self) -> int:
        return self._nu if self._has_free_joint else self._nv

    # ------------------------------------------------------------------ #
    # Cold-path model queries                                              #
    # ------------------------------------------------------------------ #

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._actuator_ctrl_range.copy()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if name not in self._keyframe_qpos:
            raise ValueError(
                f"Keyframe {name!r} not found; available: {sorted(self._keyframe_qpos)}"
            )
        return self._keyframe_qpos[name].copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._nv,), dtype=np.float64)

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        ids = np.empty(len(names), dtype=np.int32)
        for i, name in enumerate(names):
            body_id = self._mujoco.mj_name2id(
                self._mj_model, int(self._mujoco.mjtObj.mjOBJ_BODY), str(name)
            )
            if body_id < 0:
                raise ValueError(f"Body {name!r} not found in mjwarp model")
            ids[i] = body_id
        return ids

    def get_joint_range(self) -> np.ndarray | None:
        jnt_range = self._mj_model.jnt_range
        mask = self._mj_model.jnt_type != 0  # exclude mjJNT_FREE
        filtered = jnt_range[mask]
        if filtered.size == 0:
            return None
        return np.asarray(filtered, dtype=np.float64)

    def get_geom_id(self, name: str) -> int:
        mujoco = self._mujoco
        gid = mujoco.mj_name2id(self._mj_model, int(mujoco.mjtObj.mjOBJ_GEOM), name)
        if gid < 0:
            raise ValueError(f"Geom {name!r} not found in mjwarp model")
        return int(gid)

    def create_hfield_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> BackendHeightScanner:
        offsets_np = np.ascontiguousarray(np.asarray(offsets, dtype=np.float64))
        if offsets_np.ndim != 2 or offsets_np.shape[1] != 2:
            raise ValueError(
                f"offsets must have shape (num_points, 2), got {offsets_np.shape}"
            )
        return _MjwarpHeightScanner(
            backend=self,
            hfield_geom_id=int(hfield_geom_id),
            offsets=offsets_np,
            frame_body_id=int(frame_body_id),
            alignment=alignment,
            output=output,
        )

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-actuator (kp, kd) read from the underlying mujoco model.

        g1_walk_flat invokes this on the cold path when ``randomize_kp`` or
        ``randomize_kd`` is true so the DR provider can stash a baseline. The
        returned values are never written back to mjwarp's shared model — see
        ``get_dr_capabilities`` for why per-env kp/kd is reported unsupported.
        """
        kp = np.asarray(self._mj_model.actuator_gainprm[:, 0], dtype=np.float64).copy()
        kd = np.asarray(-self._mj_model.actuator_biasprm[:, 2], dtype=np.float64).copy()
        return kp, kd

    # ------------------------------------------------------------------ #
    # Simulation control: reset and DR                                     #
    # ------------------------------------------------------------------ #

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        env_indices_np = np.asarray(env_indices, dtype=np.int64)
        if env_indices_np.size == 0:
            return

        # The randomization terms that mjwarp Phase 1 does not support
        # (kp/kd in particular) are filtered out by capabilities upstream,
        # so anything reaching here is dropped silently; capability honesty
        # lives in get_dr_capabilities below.
        del randomization

        qpos_np = np.asarray(qpos, dtype=np.float64)
        qvel_np = np.asarray(qvel, dtype=np.float64)
        if qpos_np.shape != (env_indices_np.size, self._nq):
            raise ValueError(
                f"qpos must have shape {(env_indices_np.size, self._nq)}, got {qpos_np.shape}"
            )
        if qvel_np.shape != (env_indices_np.size, self._nv):
            raise ValueError(
                f"qvel must have shape {(env_indices_np.size, self._nv)}, got {qvel_np.shape}"
            )

        # Update host cache so getters return consistent values immediately.
        self._cache_qpos[env_indices_np] = qpos_np
        self._cache_qvel[env_indices_np] = qvel_np

        # Read current device state, splice in the new rows, write back.
        # A targeted warp kernel could do this with one D2H+H2D for the touched
        # rows, but Phase 1 favors a simple full-array round-trip; reset is not
        # in the hot path.
        device_qpos = self._mjw_data.qpos.numpy()
        device_qvel = self._mjw_data.qvel.numpy()
        device_qpos[env_indices_np] = qpos_np.astype(device_qpos.dtype, copy=False)
        device_qvel[env_indices_np] = qvel_np.astype(device_qvel.dtype, copy=False)
        self._mjw_data.qpos.assign(device_qpos)
        self._mjw_data.qvel.assign(device_qvel)
        self._wp.synchronize_device()

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        # mjwarp's batched Data shares the Model across worlds, so per-env
        # actuator gain / mass / friction overrides are not supported.
        # Reset-time randomization payload terms (kp/kd/body_mass/com/etc.)
        # are filtered upstream by the DR manager based on the empty set
        # returned here. Interval push and per-body force ARE supported via
        # the GPU-side xfrc_applied buffer (see apply_body_force / step).
        push_supported = self._push_body_id >= 0
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(),
            supports_interval_push=push_supported,
            supports_interval_body_velocity_delta=False,
            supports_interval_body_force=True,
        )

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        # Start from a clean wrench buffer so successive applies do not
        # accumulate stale forces. Matches MuJoCoBackend.apply_interval_randomization
        # ([src/unilab/base/backend/mujoco/backend.py]).
        self._pending_xfrc_applied.fill(0.0)
        if plan.push_perturbation_limit is not None:
            self._sample_push_into_pending(plan.push_perturbation_limit)
            self._xfrc_dirty = True
        if plan.body_force is not None:
            if plan.body_ids is None:
                raise ValueError("Interval body-force perturbation requires body_ids")
            self.apply_body_force(plan.body_ids, plan.body_force)
        if plan.body_linear_velocity_delta is not None:
            raise NotImplementedError(
                "MjwarpBackend does not implement interval body linear velocity delta "
                "(no current benchmark task requires it)"
            )

    def _sample_push_into_pending(
        self, force_range: Sequence[float] | np.ndarray
    ) -> None:
        if self._push_body_id < 0:
            raise RuntimeError("Interval push requested but push_body_id is unresolved")
        ex_force = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3))
        ex_force *= np.asarray(force_range, dtype=np.float64)
        self._pending_xfrc_applied[:, self._push_body_id, 0:3] = ex_force.astype(
            np.float32, copy=False
        )

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
    ) -> None:
        """Accumulate one world-frame force vector per target body for the
        upcoming step. Mirrors MuJoCoBackend.apply_body_force semantics; the
        force is staged in the host-side buffer and uploaded to
        ``data.xfrc_applied`` at the start of ``step``.
        """
        body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        force_np = np.asarray(force, dtype=np.float64)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
        if force_np.shape != expected_shape:
            raise ValueError(
                f"body force must have shape {expected_shape}, got {force_np.shape}"
            )
        for body_offset, body_id in enumerate(body_ids_np):
            bid = int(body_id)
            self._pending_xfrc_applied[:, bid, 0:3] += force_np[:, body_offset, :].astype(
                np.float32, copy=False
            )
        self._xfrc_dirty = True

    def materialize(self) -> None:
        # mjwarp model/data are already finalized in __init__.
        return

    # ------------------------------------------------------------------ #
    # Base kinematics                                                      #
    # ------------------------------------------------------------------ #

    def get_base_pos(self) -> np.ndarray:
        if not self._has_free_joint:
            raise NotImplementedError(
                "get_base_pos requires a free joint at index 0 (g1_walk_flat layout)"
            )
        return self._cache_qpos[:, 0:3].copy()

    def get_base_quat(self) -> np.ndarray:
        if not self._has_free_joint:
            raise NotImplementedError("get_base_quat requires free-joint layout")
        return self._cache_qpos[:, 3:7].copy()

    def get_base_lin_vel(self) -> np.ndarray:
        if not self._has_free_joint:
            raise NotImplementedError("get_base_lin_vel requires free-joint layout")
        return self._cache_qvel[:, 0:3].copy()

    def get_base_ang_vel(self) -> np.ndarray:
        if not self._has_free_joint:
            raise NotImplementedError("get_base_ang_vel requires free-joint layout")
        return self._cache_qvel[:, 3:6].copy()

    # ------------------------------------------------------------------ #
    # DOF state                                                            #
    # ------------------------------------------------------------------ #

    def get_dof_pos(self) -> np.ndarray:
        return self._cache_qpos[:, self._dof_qpos_start : self._dof_qpos_start + self._nu].copy()

    def get_dof_vel(self) -> np.ndarray:
        return self._cache_qvel[:, self._dof_qvel_start : self._dof_qvel_start + self._nu].copy()

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                        #
    # ------------------------------------------------------------------ #

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "MjwarpBackend Phase 1 does not implement get_body_pos_w "
            "(g1_walk_flat reads world-frame body data via sensors instead)"
        )

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MjwarpBackend Phase 1 does not implement get_body_quat_w")

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MjwarpBackend Phase 1 does not implement get_body_lin_vel_w")

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MjwarpBackend Phase 1 does not implement get_body_ang_vel_w")

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                     #
    # ------------------------------------------------------------------ #

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MjwarpBackend Phase 1 does not implement get_body_pos_b")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MjwarpBackend Phase 1 does not implement get_body_quat_b")

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MjwarpBackend Phase 1 does not implement get_body_lin_vel_b")

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MjwarpBackend Phase 1 does not implement get_body_ang_vel_b")

    # ------------------------------------------------------------------ #
    # Sensors                                                              #
    # ------------------------------------------------------------------ #

    def get_sensor_data(self, name: str) -> np.ndarray:
        slot = self._sensor_adr.get(name)
        if slot is None:
            raise ValueError(
                f"Sensor {name!r} not found in mjwarp model; "
                f"available: {sorted(self._sensor_adr)}"
            )
        adr, dim = slot
        # Match MuJoCoBackend.get_sensor_data: always return shape
        # (num_envs, dim). Tasks rely on NumPy broadcasting on the dim axis
        # (e.g. go2 broadcasts dim=1 contact "found" sensors into a (num_envs, 3)
        # feet_force buffer). Squeezing dim==1 here would break that path.
        return self._cache_sensordata[:, adr : adr + dim].copy()
