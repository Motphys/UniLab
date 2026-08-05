"""Explicit retirement diagnostics for the removed device-resident mjwarp path.

Issue #886 (Phase 0 scope reset) removed the production device-resident
mjwarp runtime introduced by the #705 branch (``runtime_impl:
mjwarp_device_v1``, ``execution_profile: device_resident``, the
``unilab.training.rsl_rl_device`` resolver, and the ``entrypoints`` routing
block), while keeping the backend-neutral manager/physics infrastructure.

These helpers fail fast with an actionable :class:`RetiredDevicePathError`
instead of an obscure import/attribute/shape error when a stale owner YAML,
composed config, checkpoint, resume, or playback request still references
the retired path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


class RetiredDevicePathError(RuntimeError):
    """Raised when a request targets the retired device-resident mjwarp path."""


RETIRED_TASK_OWNERS: tuple[str, ...] = ()
"""Owner identities still retired after the host mjwarp adapter was restored."""

RETIRED_RUNTIME_IMPL = "mjwarp_device_v1"
"""Retired ``algo.runtime_impl`` marker for the device-resident runtime."""

RETIRED_EXECUTION_PROFILE = "device_resident"
"""Retired ``training.execution_profile`` marker for the device-resident runtime."""

RETIRED_RESOLVER_FRAGMENT = "unilab.training.rsl_rl_device"
"""Import-path fragment of the retired device runtime resolver module."""

RETIRED_MARKERS: tuple[str, ...] = (
    RETIRED_RUNTIME_IMPL,
    RETIRED_EXECUTION_PROFILE,
    RETIRED_RESOLVER_FRAGMENT,
)
"""String markers that identify artifacts of the retired device path."""

_MIGRATION_HINT = (
    "The production device-resident mjwarp runtime (runtime_impl "
    f"{RETIRED_RUNTIME_IMPL!r}, execution_profile {RETIRED_EXECUTION_PROFILE!r}, "
    f"resolver {RETIRED_RESOLVER_FRAGMENT!r}) was retired in issue #886 and no "
    "longer exists. task=g1_walk_flat/mjwarp now selects the backend-neutral "
    "host adapter; device-resident artifacts are incompatible "
    "with it and must be retrained."
)


def _normalize_owner_name(task: str) -> str:
    """Normalize an owner name so ``G1WalkFlat/mjwarp`` matches ``g1_walk_flat/mjwarp``."""

    normalized = str(task).strip().lower().replace("\\", "/")
    return normalized.replace("_", "").replace("-", "")


def check_retired_task_owner(task: str) -> None:
    """Raise :class:`RetiredDevicePathError` when ``task`` names a retired owner."""

    normalized = _normalize_owner_name(task)
    for retired_owner in RETIRED_TASK_OWNERS:
        if normalized == _normalize_owner_name(retired_owner):
            raise RetiredDevicePathError(
                f"Task owner {retired_owner!r} is retired (requested: {task!r}). " + _MIGRATION_HINT
            )


def check_retired_task_overrides(args: Sequence[str]) -> None:
    """Scan CLI-style overrides for a retired ``task=`` selection and fail fast.

    Hydra composes the task group before ``main()`` runs, so a deleted owner
    YAML would otherwise surface as a generic "could not find config" error.
    """

    for arg in args:
        text = str(arg).strip()
        if text.startswith("task="):
            check_retired_task_owner(text.split("=", 1)[1])


def _select_any(cfg: Any, path: str) -> Any:
    """Best-effort dotted-path read that never requires the field to exist."""

    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, path, default=None)
    node = cfg
    for key in path.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def check_retired_config(cfg: Any) -> None:
    """Raise :class:`RetiredDevicePathError` when a composed config carries retired markers."""

    marker_fields = (
        ("training.execution_profile", RETIRED_EXECUTION_PROFILE),
        ("algo.runtime_impl", RETIRED_RUNTIME_IMPL),
        ("algo.runtime_resolver", RETIRED_RESOLVER_FRAGMENT),
    )
    for path, marker in marker_fields:
        value = _select_any(cfg, path)
        if value is not None and marker in str(value):
            raise RetiredDevicePathError(
                f"Config field {path!r} references the retired device-resident "
                f"mjwarp path (value: {value!r}). " + _MIGRATION_HINT
            )
    if _select_any(cfg, "entrypoints") is not None:
        raise RetiredDevicePathError(
            "Config block 'entrypoints' belongs to the retired device-resident "
            "mjwarp routing layer. " + _MIGRATION_HINT
        )


def _find_retired_marker(node: Any) -> str | None:
    """Return the first retired marker string found anywhere in ``node``."""

    if isinstance(node, str):
        for marker in RETIRED_MARKERS:
            if marker in node:
                return marker
        return None
    if isinstance(node, Mapping):
        for value in node.values():
            hit = _find_retired_marker(value)
            if hit is not None:
                return hit
        return None
    if isinstance(node, (list, tuple)):
        for value in node:
            hit = _find_retired_marker(value)
            if hit is not None:
                return hit
    return None


def check_retired_run_config(data: Any, *, source: str) -> None:
    """Raise :class:`RetiredDevicePathError` when a parsed run_config carries markers.

    The search is recursive across the ``training``/``algo``/``contract_snapshot``
    subtrees, so a retired device ABI snapshot (``execution_profile:
    device_resident`` under ``manager.policy_abi``) is reported as
    :class:`RetiredDevicePathError` rather than a generic contract mismatch.
    """

    if not isinstance(data, Mapping):
        return
    hit = _find_retired_marker(data)
    if hit is not None:
        raise RetiredDevicePathError(
            f"Run config at {source} was produced by the retired device-resident "
            f"mjwarp path (marker: {hit!r}). " + _MIGRATION_HINT
        )


def check_retired_checkpoint(path: str | Path) -> None:
    """Raise :class:`RetiredDevicePathError` when a checkpoint comes from a retired run.

    Looks for ``run_config.json`` in the checkpoint's directory and its
    ancestors (the experiment tracker writes it into the run directory next
    to ``model_*.pt``).  Returns silently when no run config is found.
    """

    candidate = Path(path)
    directory = candidate if candidate.is_dir() else candidate.parent
    for run_dir in (directory, *list(directory.parents)[:5]):
        run_config = run_dir / "run_config.json"
        if not run_config.is_file():
            continue
        try:
            data = json.loads(run_config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        check_retired_run_config(data, source=str(run_config))
        return
