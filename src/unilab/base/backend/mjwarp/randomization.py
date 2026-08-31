"""Per-world model-field expansion for the independent ``mjwarp`` backend.

MuJoCo Warp declares the leading dimension of randomizable ``Model`` fields as
``"*"``: every kernel reads ``field[worldid % field.shape[0]]``, so tiling a
field from ``(1, ...)`` to ``(nworld, ...)`` yields per-world semantics without
kernel changes.  Expansion replaces the array allocation, so it is a cold-path
operation: the backend expands the declared set once during construction,
*before* CUDA graph capture, and all later DR writes are in-place ``assign``
uploads into the same fixed-address arrays (graph-safe).

The field list is ported from mjlab's ``expand_model_fields`` usage and checked
against the pinned mujoco-warp 3.10 ``Model`` dataclass.  Derived fields are
expanded alongside their inputs because the ``set_const*`` recompute kernels
size their launch grid from the *output* field's leading dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Payload-writable fields that require ``mujoco_warp.set_const`` afterwards
# (mass/inertial family; superset of the set_const_0 level).
SET_CONST_FIELDS = ("body_mass", "body_ipos", "body_iquat")
# Payload-writable fields that require ``mujoco_warp.set_const_0`` afterwards.
SET_CONST_0_FIELDS = ("body_inertia", "dof_armature")
# Payload-writable fields that need no derived-quantity recomputation.
NO_RECOMPUTE_FIELDS = ("geom_friction", "actuator_gainprm", "actuator_biasprm")
# Derived fields recomputed by the ``set_const*`` family.  They must be
# per-world as well: the recompute kernels launch over ``field.shape[0]`` and
# the step/forward kernels index them per world.
DERIVED_FIELDS = (
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
)

EXPANDED_MODEL_FIELDS: tuple[str, ...] = (
    SET_CONST_FIELDS + SET_CONST_0_FIELDS + NO_RECOMPUTE_FIELDS + DERIVED_FIELDS
)


def expand_model_fields(warp: Any, model: Any, nworld: int) -> tuple[str, ...]:
    """Tile the declared DR model fields from ``(1, ...)`` to ``(nworld, ...)``.

    Returns the names actually expanded.  Fields already per-world (e.g. from a
    prior expansion) are skipped, mirroring mjlab's guard.  A single-world
    backend keeps the shared arrays untouched.
    """
    if nworld <= 1:
        return ()
    model_fields = getattr(model, "__dataclass_fields__", None)
    if model_fields is None:
        raise TypeError(
            f"mjwarp DR expansion requires a mujoco_warp.Model dataclass, got {type(model)}"
        )
    expanded: list[str] = []
    for name in EXPANDED_MODEL_FIELDS:
        if name not in model_fields:
            raise RuntimeError(
                f"mujoco-warp Model no longer declares DR field {name!r}; update the "
                "mjwarp expansion field list for the pinned mujoco-warp version"
            )
        array = getattr(model, name)
        if array.shape[0] == nworld:
            continue
        if array.shape[0] != 1:
            raise RuntimeError(
                f"mujoco-warp Model field {name!r} has unexpected leading dim "
                f"{array.shape[0]}; expected 1 (shared) or {nworld} (per-world)"
            )
        host = np.asarray(array.numpy())
        tiled = np.ascontiguousarray(np.broadcast_to(host, (nworld, *host.shape[1:])))
        replacement = warp.array(
            shape=(nworld, *array.shape[1:]),
            dtype=array.dtype,
            device=array.device,
        )
        replacement.assign(tiled)
        setattr(model, name, replacement)
        expanded.append(name)
    return tuple(expanded)
