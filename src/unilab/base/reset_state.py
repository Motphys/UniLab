"""Base-owned reset-state transaction for Manager-Based event terms.

The transaction composes NumPy state writes in memory and hands the finished
batch to :meth:`SimBackend.set_state` exactly once.  It deliberately knows
nothing about task configuration, IPC, runners, or backend-private state.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from unilab.base.backend.base import BackendRootStateLayout, SimBackend
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
        self._requesting_terms: set[str] = set()

    @property
    def active(self) -> bool:
        """Whether a reset lifecycle currently owns the transaction."""
        return self._active

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
        self._requesting_terms.clear()
        self._active = True

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
        try:
            if dirty_ids.size == 0:
                return None
            assert self._qpos is not None
            assert self._qvel is not None
            try:
                return self._backend.set_state(
                    dirty_ids,
                    self._qpos[dirty_ids],
                    self._qvel[dirty_ids],
                )
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
        if np.unique(ids).size != ids.size:
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
        shape: tuple[int, int],
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
        self._requesting_terms.clear()


__all__ = ["ResetStateTransaction"]
