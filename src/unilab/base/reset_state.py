"""Base-owned reset-state transaction for Manager-Based event terms.

The transaction composes NumPy state writes in memory and hands the finished
batch to :meth:`SimBackend.set_state` exactly once.  It deliberately knows
nothing about task configuration, IPC, runners, or backend-private state.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from unilab.base.backend.base import BackendRootStateLayout, SimBackend
from unilab.dr.types import (
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    ResetRandomizationPayload,
)
from unilab.utils.rotation import np_quat_apply_inverse


class ResetStateTransaction:
    """Reusable, fail-closed transaction for reset-mode state mutation."""

    def __init__(
        self,
        backend: SimBackend,
        *,
        default_qpos: np.ndarray | None = None,
    ) -> None:
        self._backend = backend
        self._num_envs = backend.num_envs
        self._selected_default_qpos = default_qpos
        self._active = False
        self._active_mask = np.zeros(self._num_envs, dtype=np.bool_)
        self._dirty_mask = np.zeros(self._num_envs, dtype=np.bool_)
        self._default_qpos: np.ndarray | None = None
        self._default_qvel: np.ndarray | None = None
        self._qpos: np.ndarray | None = None
        self._qvel: np.ndarray | None = None
        self._default_kp: np.ndarray | None = None
        self._default_kd: np.ndarray | None = None
        self._kp: np.ndarray | None = None
        self._kd: np.ndarray | None = None
        self._gain_dirty_mask = np.zeros(self._num_envs, dtype=np.bool_)
        self._randomization_defaults: dict[str, np.ndarray] = {}
        self._randomization_values: dict[str, np.ndarray] = {}
        self._randomization_dirty_masks: dict[str, np.ndarray] = {}
        self._requesting_terms: set[str] = set()
        self._last_commit_had_writes = False
        self._last_set_state_timing_ms: dict[str, float] = {}

    @property
    def active(self) -> bool:
        """Whether a reset lifecycle currently owns the transaction."""
        return self._active

    @property
    def last_commit_had_writes(self) -> bool:
        """Whether the most recent scoped commit submitted dirty rows to set_state."""
        return self._last_commit_had_writes

    @property
    def last_set_state_timing_ms(self) -> dict[str, float]:
        """Sub-timings from the most recent commit's set_state call.

        Always includes ``dr_reset_set_state_ms`` (outer wall-clock around the
        backend call); backend-reported ``set_state_*_ms`` sub-keys are merged
        in when the backend returns them. Empty when the last commit had no
        dirty rows.
        """
        return self._last_set_state_timing_ms

    @contextmanager
    def scoped(self, env_ids: np.ndarray) -> Iterator[ResetStateTransaction]:
        """Begin a reset transaction and commit it only after all terms succeed."""
        self.begin(env_ids)
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        else:
            self.commit()

    def begin(self, env_ids: np.ndarray) -> None:
        """Open a transaction for the concrete reset environment IDs."""
        if self._active:
            raise RuntimeError("ManagerBased reset-state transaction is already active")
        ids = self._validate_ids(env_ids, capability="begin")
        self._active_mask.fill(False)
        self._active_mask[ids] = True
        self._dirty_mask.fill(False)
        self._gain_dirty_mask.fill(False)
        for mask in self._randomization_dirty_masks.values():
            mask.fill(False)
        self._requesting_terms.clear()
        self._last_commit_had_writes = False
        self._last_set_state_timing_ms = {}
        self._active = True

    def bind_body_mass_write(
        self,
        body_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind body-mass columns and immutable backend defaults on the cold path."""
        default = self._materialize_randomization_default(
            RESET_TERM_BODY_MASS,
            getter=self._backend.get_body_mass,
            expected_tail=None,
            term_name=term_name,
        )
        columns = self._validate_columns(
            body_ids,
            width=default.shape[0],
            capability="body mass IDs",
            term_name=term_name,
        )
        return self._readonly_binding(columns, default[columns])

    def bind_body_ipos_write(
        self,
        body_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind body inertial-position columns and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_BODY_IPOS,
            getter=self._backend.get_body_ipos,
            expected_tail=(3,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            body_ids,
            width=default.shape[0],
            capability="body ipos IDs",
            term_name=term_name,
        )
        return self._readonly_binding(columns, default[columns])

    def bind_gravity_write(self, *, term_name: str) -> np.ndarray:
        """Bind the immutable backend gravity vector on the cold path."""
        return self._materialize_randomization_default(
            RESET_TERM_GRAVITY,
            getter=self._backend.get_gravity,
            expected_tail=(),
            term_name=term_name,
        )

    def bind_dof_armature_write(
        self,
        dof_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind DOF-armature columns and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_DOF_ARMATURE,
            getter=self._backend.get_dof_armature,
            expected_tail=None,
            term_name=term_name,
        )
        columns = self._validate_columns(
            dof_ids,
            width=default.shape[0],
            capability="DOF armature IDs",
            term_name=term_name,
        )
        return self._readonly_binding(columns, default[columns])

    def bind_geom_friction_write(
        self,
        geom_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind geom-friction rows and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_FRICTION,
            getter=self._backend.get_geom_friction,
            expected_tail=(3,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            geom_ids,
            width=default.shape[0],
            capability="geom friction IDs",
            term_name=term_name,
        )
        return self._readonly_binding(columns, default[columns])

    def write_body_mass(
        self,
        env_ids: np.ndarray,
        body_ids: np.ndarray,
        values: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage selected body masses in the exactly-once reset payload."""
        self._write_selected_randomization(
            RESET_TERM_BODY_MASS,
            env_ids,
            body_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def write_body_ipos(
        self,
        env_ids: np.ndarray,
        body_ids: np.ndarray,
        values: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage selected body inertial positions in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_BODY_IPOS,
            env_ids,
            body_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def write_gravity(
        self,
        env_ids: np.ndarray,
        values: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage per-environment gravity vectors in the reset payload."""
        ids = self._prepare_state_write(
            env_ids,
            capability="gravity",
            term_name=term_name,
        )
        default = self._require_randomization_default(RESET_TERM_GRAVITY, term_name)
        gravity = self._validate_values(
            values,
            shape=(ids.size, 3),
            capability="gravity",
            term_name=term_name,
        )
        buffer = self._randomization_values[RESET_TERM_GRAVITY]
        mask = self._randomization_dirty_masks[RESET_TERM_GRAVITY]
        uninitialized = ids[~mask[ids]]
        if uninitialized.size:
            buffer[uninitialized] = default
        buffer[ids] = gravity
        mask[ids] = True
        self._dirty_mask[ids] = True

    def write_dof_armature(
        self,
        env_ids: np.ndarray,
        dof_ids: np.ndarray,
        values: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage selected DOF armatures in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_DOF_ARMATURE,
            env_ids,
            dof_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def write_geom_friction(
        self,
        env_ids: np.ndarray,
        geom_ids: np.ndarray,
        values: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage selected three-axis geom friction in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_FRICTION,
            env_ids,
            geom_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def bind_actuator_gain_write(
        self,
        actuator_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve gain mutation capability and immutable defaults on the cold path."""
        columns = self._validate_columns(
            actuator_ids,
            width=self._backend.num_actuators,
            capability="actuator IDs",
            term_name=term_name,
        )
        self._materialize_default_actuator_gains(term_name)
        assert self._default_kp is not None
        assert self._default_kd is not None
        selected_kp = np.array(self._default_kp[columns], copy=True)
        selected_kd = np.array(self._default_kd[columns], copy=True)
        selected_kp.setflags(write=False)
        selected_kd.setflags(write=False)
        bound_columns = np.array(columns, copy=True)
        bound_columns.setflags(write=False)
        return bound_columns, selected_kp, selected_kd

    def write_actuator_gains(
        self,
        env_ids: np.ndarray,
        actuator_ids: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage selected per-environment actuator gains in the reset transaction."""
        ids = self._prepare_state_write(
            env_ids,
            capability="actuator-gain",
            term_name=term_name,
        )
        columns = self._validate_columns(
            actuator_ids,
            width=self._backend.num_actuators,
            capability="actuator IDs",
            term_name=term_name,
        )
        self._materialize_default_actuator_gains(term_name)
        gains_shape = (ids.size, columns.size)
        kp_values = self._validate_values(
            kp,
            shape=gains_shape,
            capability="actuator kp",
            term_name=term_name,
        )
        kd_values = self._validate_values(
            kd,
            shape=gains_shape,
            capability="actuator kd",
            term_name=term_name,
        )
        assert self._default_kp is not None
        assert self._default_kd is not None
        assert self._kp is not None
        assert self._kd is not None
        uninitialized = ids[~self._gain_dirty_mask[ids]]
        if uninitialized.size:
            self._kp[uninitialized] = self._default_kp
            self._kd[uninitialized] = self._default_kd
        if ids.size and columns.size:
            self._kp[ids[:, None], columns[None, :]] = kp_values
            self._kd[ids[:, None], columns[None, :]] = kd_values
        self._gain_dirty_mask[ids] = True
        self._dirty_mask[ids] = True

    def reset_to_default(self, env_ids: np.ndarray, *, term_name: str) -> None:
        """Stage backend default qpos/qvel for a subset of the active reset."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="reset_to_default")
        outside = ids[~self._active_mask[ids]]
        if outside.size:
            raise ValueError(
                "EventManager term "
                f"'{term_name}' attempted reset-state mutation outside the active reset: "
                f"{outside.tolist()}"
            )
        if ids.size == 0:
            return
        self._requesting_terms.add(term_name)
        self._materialize_default_state(term_name)
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        assert self._qpos is not None
        assert self._qvel is not None
        self._qpos[ids] = self._default_qpos
        self._qvel[ids] = self._default_qvel
        self._dirty_mask[ids] = True

    def write_joint_state(
        self,
        env_ids: np.ndarray,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage selected joint position and velocity columns in the reset batch."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="write_joint_state")
        outside = ids[~self._active_mask[ids]]
        if outside.size:
            raise ValueError(
                f"EventManager term '{term_name}' attempted joint-state mutation outside "
                f"the active reset: {outside.tolist()}"
            )
        self._requesting_terms.add(term_name)
        self._materialize_default_state(term_name)
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        assert self._qpos is not None
        assert self._qvel is not None

        pos_columns = self._validate_columns(
            qpos_indices,
            width=self._default_qpos.size,
            capability="qpos indices",
            term_name=term_name,
        )
        vel_columns = self._validate_columns(
            qvel_indices,
            width=self._default_qvel.size,
            capability="qvel indices",
            term_name=term_name,
        )
        if pos_columns.size != vel_columns.size:
            raise ValueError(
                f"EventManager term '{term_name}' joint-state qpos/qvel index counts differ: "
                f"{pos_columns.size} != {vel_columns.size}"
            )
        positions = self._validate_values(
            position,
            shape=(ids.size, pos_columns.size),
            capability="joint position",
            term_name=term_name,
        )
        velocities = self._validate_values(
            velocity,
            shape=(ids.size, vel_columns.size),
            capability="joint velocity",
            term_name=term_name,
        )

        uninitialized = ids[~self._dirty_mask[ids]]
        if uninitialized.size:
            self._qpos[uninitialized] = self._default_qpos
            self._qvel[uninitialized] = self._default_qvel
        if ids.size and pos_columns.size:
            self._qpos[ids[:, None], pos_columns[None, :]] = positions
            self._qvel[ids[:, None], vel_columns[None, :]] = velocities
        self._dirty_mask[ids] = True

    def write_root_state(
        self,
        env_ids: np.ndarray,
        layout: BackendRootStateLayout,
        root_state: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage a community 13-D world-frame root state."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="write_root_state")
        values = self._validate_values(
            root_state,
            shape=(ids.size, 13),
            capability="root state",
            term_name=term_name,
        )
        self.write_root_pose(ids, layout, values[:, :7], term_name=term_name)
        self.write_root_velocity(ids, layout, values[:, 7:], term_name=term_name)

    def write_root_pose(
        self,
        env_ids: np.ndarray,
        layout: BackendRootStateLayout,
        pose: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage world position and wxyz orientation for one floating root."""
        ids = self._prepare_state_write(env_ids, capability="root-pose", term_name=term_name)
        positions = self._validate_values(
            pose,
            shape=(ids.size, 7),
            capability="root pose",
            term_name=term_name,
        )
        self._validate_quaternions(positions[:, 3:7], term_name=term_name)
        qpos_columns, _ = self._validate_root_layout(layout, term_name=term_name)
        assert self._qpos is not None
        if ids.size:
            self._qpos[ids[:, None], qpos_columns[None, :]] = positions
        self._dirty_mask[ids] = True

    def write_root_velocity(
        self,
        env_ids: np.ndarray,
        layout: BackendRootStateLayout,
        velocity_w: np.ndarray,
        *,
        term_name: str,
    ) -> None:
        """Stage world-frame root velocity in generalized qvel columns.

        The public generalized-state contract stores free-root linear velocity
        in world coordinates and angular velocity in root-body coordinates.
        The conversion uses the pose already staged in this transaction.
        """
        ids = self._prepare_state_write(env_ids, capability="root-velocity", term_name=term_name)
        velocities = self._validate_values(
            velocity_w,
            shape=(ids.size, 6),
            capability="root velocity",
            term_name=term_name,
        )
        qpos_columns, qvel_columns = self._validate_root_layout(layout, term_name=term_name)
        assert self._qpos is not None
        assert self._qvel is not None
        if ids.size:
            quaternions = self._qpos[
                ids[:, None],
                qpos_columns[None, 3:7],
            ]
            self._validate_quaternions(quaternions, term_name=term_name)
            encoded_velocity = np.array(velocities, copy=True)
            encoded_velocity[:, 3:6] = np_quat_apply_inverse(
                quaternions,
                velocities[:, 3:6],
            )
            self._qvel[ids[:, None], qvel_columns[None, :]] = encoded_velocity
        self._dirty_mask[ids] = True

    def commit(self) -> dict | None:
        """Commit all staged rows through one public backend call."""
        self._require_active()
        dirty_ids = np.flatnonzero(self._dirty_mask).astype(np.int32, copy=False)
        self._last_commit_had_writes = bool(dirty_ids.size)
        try:
            if dirty_ids.size == 0:
                return None
            assert self._qpos is not None
            assert self._qvel is not None
            randomization = self._build_randomization_payload(dirty_ids)
            try:
                set_state_t0 = time.perf_counter()
                result = self._backend.set_state(
                    dirty_ids,
                    self._qpos[dirty_ids],
                    self._qvel[dirty_ids],
                    randomization=randomization,
                )
                timing: dict[str, float] = {
                    "dr_reset_set_state_ms": (time.perf_counter() - set_state_t0) * 1000.0
                }
                if isinstance(result, dict):
                    backend_timing = result.get("timing")
                    if isinstance(backend_timing, dict):
                        timing.update(backend_timing)
                self._last_set_state_timing_ms = timing
                return result
            except (AttributeError, NotImplementedError) as exc:
                terms = ", ".join(sorted(self._requesting_terms))
                raise NotImplementedError(
                    "EventManager reset-state capability 'SimBackend.set_state' is unavailable "
                    f"for term(s) [{terms}] on backend '{self._backend.backend_type}': {exc}"
                ) from exc
        finally:
            self._finish()

    def abort(self) -> None:
        """Discard staged rows without touching the backend."""
        if self._active:
            self._finish()

    def _materialize_default_state(self, term_name: str) -> None:
        if self._default_qpos is not None:
            return
        qpos = self._selected_default_qpos
        if qpos is None:
            try:
                qpos = self._backend.get_default_qpos()
            except (AttributeError, NotImplementedError) as exc:
                raise self._capability_error(term_name, "default qpos", exc) from exc
        try:
            qvel = self._backend.get_init_qvel()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, "initial qvel", exc) from exc

        default_qpos = self._validate_state_vector(qpos, "default qpos", term_name)
        default_qvel = self._validate_state_vector(qvel, "initial qvel", term_name)
        self._default_qpos = default_qpos
        self._default_qvel = default_qvel
        self._qpos = np.empty((self._num_envs, default_qpos.size), dtype=default_qpos.dtype)
        self._qvel = np.empty((self._num_envs, default_qvel.size), dtype=default_qvel.dtype)

    def _materialize_default_actuator_gains(self, term_name: str) -> None:
        if self._default_kp is not None:
            return
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, "actuator gain randomization", exc) from exc
        required = frozenset((RESET_TERM_KP, RESET_TERM_KD))
        unsupported = capabilities.get_unsupported_reset_terms(required)
        if unsupported:
            detail = ", ".join(sorted(unsupported))
            raise self._capability_error(
                term_name,
                "actuator gain randomization",
                NotImplementedError(f"unsupported reset payload fields: {detail}"),
            )
        try:
            kp, kd = self._backend.get_actuator_gains()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, "default actuator gains", exc) from exc
        default_kp = self._validate_gain_vector(kp, "default actuator kp", term_name)
        default_kd = self._validate_gain_vector(kd, "default actuator kd", term_name)
        self._default_kp = default_kp
        self._default_kd = default_kd
        self._kp = np.empty(
            (self._num_envs, self._backend.num_actuators),
            dtype=default_kp.dtype,
        )
        self._kd = np.empty(
            (self._num_envs, self._backend.num_actuators),
            dtype=default_kd.dtype,
        )

    def _materialize_randomization_default(
        self,
        field: str,
        *,
        getter,
        expected_tail: tuple[int, ...] | None,
        term_name: str,
    ) -> np.ndarray:
        cached = self._randomization_defaults.get(field)
        if cached is not None:
            return cached
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, f"{field} randomization", exc) from exc
        unsupported = capabilities.get_unsupported_reset_terms(frozenset((field,)))
        if unsupported:
            raise self._capability_error(
                term_name,
                f"{field} randomization",
                NotImplementedError(f"unsupported reset payload field: {field}"),
            )
        try:
            value = getter()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, f"default {field}", exc) from exc
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' must return np.ndarray, got "
                f"{type(value).__name__}"
            )
        expected_ndim = 1 if expected_tail is None else 1 + len(expected_tail)
        if value.ndim != expected_ndim:
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected "
                f"{expected_ndim}-D"
            )
        if expected_tail is not None and value.shape[1:] != expected_tail:
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected tail "
                f"{expected_tail}"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' must be floating, got {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned NaN or Inf"
            )
        default = np.array(value, copy=True)
        default.setflags(write=False)
        self._randomization_defaults[field] = default
        self._randomization_values[field] = np.empty(
            (self._num_envs, *default.shape),
            dtype=default.dtype,
        )
        self._randomization_dirty_masks[field] = np.zeros(self._num_envs, dtype=np.bool_)
        return default

    def _require_randomization_default(self, field: str, term_name: str) -> np.ndarray:
        try:
            return self._randomization_defaults[field]
        except KeyError as exc:
            raise RuntimeError(
                f"EventManager term '{term_name}' must bind reset field '{field}' "
                "during manager construction before writing it"
            ) from exc

    def _readonly_binding(
        self,
        columns: np.ndarray,
        selected_default: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        bound_columns = np.array(columns, copy=True)
        bound_columns.setflags(write=False)
        selected = np.array(selected_default, copy=True)
        selected.setflags(write=False)
        return bound_columns, selected

    def _write_selected_randomization(
        self,
        field: str,
        env_ids: np.ndarray,
        column_ids: np.ndarray,
        values: np.ndarray,
        *,
        value_tail: tuple[int, ...],
        term_name: str,
    ) -> None:
        ids = self._prepare_state_write(
            env_ids,
            capability=field,
            term_name=term_name,
        )
        default = self._require_randomization_default(field, term_name)
        columns = self._validate_columns(
            column_ids,
            width=default.shape[0],
            capability=f"{field} column IDs",
            term_name=term_name,
        )
        selected = self._validate_values(
            values,
            shape=(ids.size, columns.size, *value_tail),
            capability=field,
            term_name=term_name,
        )
        buffer = self._randomization_values[field]
        mask = self._randomization_dirty_masks[field]
        uninitialized = ids[~mask[ids]]
        if uninitialized.size:
            buffer[uninitialized] = default
        if ids.size and columns.size:
            buffer[ids[:, None], columns[None, :]] = selected
        mask[ids] = True
        self._dirty_mask[ids] = True

    def _build_randomization_payload(
        self,
        dirty_ids: np.ndarray,
    ) -> ResetRandomizationPayload | None:
        payload = ResetRandomizationPayload()
        for field in (
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_IPOS,
            RESET_TERM_DOF_ARMATURE,
            RESET_TERM_GEOM_FRICTION,
            RESET_TERM_GRAVITY,
        ):
            mask = self._randomization_dirty_masks.get(field)
            if mask is None or not np.any(mask):
                continue
            self._require_dense_randomization_rows(field, mask, dirty_ids)
            setattr(
                payload,
                field,
                np.array(self._randomization_values[field][dirty_ids], copy=True),
            )

        gain_ids = np.flatnonzero(self._gain_dirty_mask).astype(np.int32, copy=False)
        if gain_ids.size:
            self._require_dense_randomization_rows(
                "actuator gains", self._gain_dirty_mask, dirty_ids
            )
            assert self._kp is not None
            assert self._kd is not None
            payload.kp = np.array(self._kp[dirty_ids], copy=True)
            payload.kd = np.array(self._kd[dirty_ids], copy=True)
        return None if payload.is_empty() else payload

    def _require_dense_randomization_rows(
        self,
        field: str,
        mask: np.ndarray,
        dirty_ids: np.ndarray,
    ) -> None:
        missing = dirty_ids[~mask[dirty_ids]]
        if missing.size:
            terms = ", ".join(sorted(self._requesting_terms))
            raise RuntimeError(
                f"EventManager reset {field} payload cannot represent sparse rows in one "
                f"SimBackend.set_state call for term(s) [{terms}] on backend "
                f"'{self._backend.backend_type}'; missing env IDs {missing.tolist()}"
            )

    def _prepare_state_write(
        self,
        env_ids: np.ndarray,
        *,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        self._require_active()
        ids = self._validate_ids(env_ids, capability=f"write_{capability}")
        outside = ids[~self._active_mask[ids]]
        if outside.size:
            raise ValueError(
                f"EventManager term '{term_name}' attempted {capability} mutation outside "
                f"the active reset: {outside.tolist()}"
            )
        self._requesting_terms.add(term_name)
        self._materialize_default_state(term_name)
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        assert self._qpos is not None
        assert self._qvel is not None
        uninitialized = ids[~self._dirty_mask[ids]]
        if uninitialized.size:
            self._qpos[uninitialized] = self._default_qpos
            self._qvel[uninitialized] = self._default_qvel
        return ids

    def _validate_root_layout(
        self,
        layout: BackendRootStateLayout,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(layout, BackendRootStateLayout):
            raise TypeError(
                f"EventManager term '{term_name}' root-state layout must be "
                f"BackendRootStateLayout, got {type(layout).__name__}"
            )
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        qpos_columns = self._validate_columns(
            np.asarray(layout.qpos_indices, dtype=np.intp),
            width=self._default_qpos.size,
            capability="root qpos indices",
            term_name=term_name,
        )
        qvel_columns = self._validate_columns(
            np.asarray(layout.qvel_indices, dtype=np.intp),
            width=self._default_qvel.size,
            capability="root qvel indices",
            term_name=term_name,
        )
        return qpos_columns, qvel_columns

    def _validate_quaternions(self, values: np.ndarray, *, term_name: str) -> None:
        norms = np.linalg.norm(values, axis=1)
        invalid = ~np.isclose(norms, 1.0, rtol=1e-5, atol=1e-6)
        if np.any(invalid):
            raise ValueError(
                f"EventManager term '{term_name}' root quaternion must be unit length; "
                f"norms={norms[invalid].tolist()}"
            )

    def _validate_state_vector(
        self,
        value: np.ndarray,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' must return np.ndarray, got "
                f"{type(value).__name__}"
            )
        if value.ndim != 1:
            raise ValueError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected 1-D"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' must be floating, got {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' returned NaN or Inf"
            )
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result

    def _validate_gain_vector(
        self,
        value: np.ndarray,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        result = self._validate_state_vector(value, capability, term_name)
        expected = (self._backend.num_actuators,)
        if result.shape != expected:
            raise ValueError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' returned shape {result.shape}; expected {expected}"
            )
        return result

    def _validate_ids(self, env_ids: np.ndarray, *, capability: str) -> np.ndarray:
        if not isinstance(env_ids, np.ndarray):
            raise TypeError(
                f"ManagerBased reset-state {capability} env_ids must be np.ndarray, "
                f"got {type(env_ids).__name__}"
            )
        if (
            env_ids.ndim != 1
            or not np.issubdtype(env_ids.dtype, np.integer)
            or np.issubdtype(env_ids.dtype, np.bool_)
        ):
            raise TypeError(
                f"ManagerBased reset-state {capability} env_ids must be a 1-D integer "
                f"np.ndarray, got shape={env_ids.shape}, dtype={env_ids.dtype}"
            )
        ids = np.asarray(env_ids, dtype=np.int32)
        if np.any(ids < 0) or np.any(ids >= self._num_envs):
            raise IndexError(
                f"ManagerBased reset-state {capability} env_ids out of range for "
                f"{self._num_envs} environments: {ids.tolist()}"
            )
        # Duplicate check via bincount instead of np.unique: identical semantics
        # (ids are already range-checked above) but avoids the sort — ~30x faster
        # at num_envs=4096 and ~4x at typical partial-reset widths. This runs on
        # every reset-state write (~6x per env step), so the sort cost was
        # measurable in the collector host phase (issue #1352).
        if ids.size > 1 and np.bincount(ids, minlength=self._num_envs).max() > 1:
            raise ValueError(
                f"ManagerBased reset-state {capability} env_ids contain duplicates: {ids.tolist()}"
            )
        return ids

    def _validate_columns(
        self,
        values: np.ndarray,
        *,
        width: int,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be np.ndarray, "
                f"got {type(values).__name__}"
            )
        if (
            values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be a 1-D integer array"
            )
        columns = np.asarray(values, dtype=np.intp)
        if np.any(columns < 0) or np.any(columns >= width):
            raise IndexError(
                f"EventManager term '{term_name}' {capability} out of range for width "
                f"{width}: {columns.tolist()}"
            )
        if np.unique(columns).size != columns.size:
            raise ValueError(
                f"EventManager term '{term_name}' {capability} contain duplicates: "
                f"{columns.tolist()}"
            )
        return columns

    def _validate_values(
        self,
        values: np.ndarray,
        *,
        shape: tuple[int, ...],
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be np.ndarray, "
                f"got {type(values).__name__}"
            )
        if values.shape != shape:
            raise ValueError(
                f"EventManager term '{term_name}' {capability} has shape {values.shape}; "
                f"expected {shape}"
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be floating, got {values.dtype}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"EventManager term '{term_name}' {capability} contains NaN or Inf")
        return values

    def _capability_error(
        self,
        term_name: str,
        capability: str,
        exc: BaseException,
    ) -> NotImplementedError:
        return NotImplementedError(
            f"EventManager term '{term_name}' reset-state capability '{capability}' is "
            f"unavailable on backend '{self._backend.backend_type}': {exc}"
        )

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("ManagerBased reset-state mutation requires an active reset event")

    def _finish(self) -> None:
        self._active = False
        self._active_mask.fill(False)
        self._dirty_mask.fill(False)
        self._gain_dirty_mask.fill(False)
        for mask in self._randomization_dirty_masks.values():
            mask.fill(False)
        self._requesting_terms.clear()


__all__ = ["ResetStateTransaction"]
