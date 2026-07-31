"""Versioned owner contract for training and policy operation entrypoints."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.manager.fingerprint import normalize_managed_policy_abi_snapshot
from unilab.training.sim2sim import policy_load_dim_guard, resolve_sim2sim_config

ENTRYPOINT_CONTRACT_FINGERPRINT = "issue705-entrypoints-v1"


class EntrypointContractError(RuntimeError):
    """Raised before runtime construction when an owner does not support a route."""


class EntrypointRoute(str, Enum):
    TRAIN = "train"
    PLAY = "play"
    VISUALIZE = "visualize"
    EXPORT = "export"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    RESUME = "resume"


class EntrypointDisposition(str, Enum):
    NATIVE = "native"
    EXPLICIT_ADAPTER = "explicit_adapter"
    UNSUPPORTED = "unsupported"


_POLICY_LOAD_ROUTES = frozenset(
    {
        EntrypointRoute.PLAY,
        EntrypointRoute.EXPORT,
        EntrypointRoute.CHECKPOINT_LOAD,
        EntrypointRoute.RESUME,
    }
)
_RENDER_ROUTES = frozenset({EntrypointRoute.PLAY, EntrypointRoute.VISUALIZE})


@dataclass(frozen=True)
class EntrypointExecutionIdentity:
    task_name: str
    backend: str
    execution_profile: str
    runtime_impl: str
    runtime_resolver: str | None

    def to_snapshot(self) -> dict[str, str | None]:
        return {
            "task_name": self.task_name,
            "backend": self.backend,
            "execution_profile": self.execution_profile,
            "runtime_impl": self.runtime_impl,
            "runtime_resolver": self.runtime_resolver,
        }


@dataclass(frozen=True)
class EntrypointRouteContract:
    fingerprint: str
    route: EntrypointRoute
    disposition: EntrypointDisposition
    identity: EntrypointExecutionIdentity
    checkpoint_required: bool
    config_guard_required: bool
    managed_abi_guard_required: bool
    dimension_guard_required: bool
    renderer_backend: str | None
    adapter_backend: str | None
    export_formats: tuple[str, ...]
    diagnostic: str

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "route": self.route.value,
            "disposition": self.disposition.value,
            "identity": self.identity.to_snapshot(),
            "checkpoint_required": self.checkpoint_required,
            "config_guard_required": self.config_guard_required,
            "managed_abi_guard_required": self.managed_abi_guard_required,
            "dimension_guard_required": self.dimension_guard_required,
            "renderer_backend": self.renderer_backend,
            "adapter_backend": self.adapter_backend,
            "export_formats": list(self.export_formats),
        }


@dataclass(frozen=True)
class PolicyLoadTarget:
    managed_policy_abi: dict[str, Any] | None
    managed_policy_abi_fingerprint: str | None
    observation_dim: int | None
    action_dim: int | None


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EntrypointContractError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, name=name)


def _route_declaration(
    cfg: DictConfig, route: EntrypointRoute
) -> tuple[EntrypointDisposition, str | None]:
    raw = OmegaConf.select(cfg, f"entrypoints.routes.{route.value}", default=None)
    if raw is None:
        raise EntrypointContractError(
            f"owner config does not declare entrypoints.routes.{route.value}; "
            f"use the {ENTRYPOINT_CONTRACT_FINGERPRINT} owner schema"
        )
    if isinstance(raw, str):
        try:
            return EntrypointDisposition(raw), None
        except ValueError as exc:
            raise EntrypointContractError(
                f"entrypoints.routes.{route.value} has unknown disposition {raw!r}"
            ) from exc
    if not OmegaConf.is_config(raw) and not isinstance(raw, Mapping):
        raise EntrypointContractError(
            f"entrypoints.routes.{route.value} must be a disposition or mapping"
        )
    disposition_raw = OmegaConf.select(cast(Any, raw), "disposition", default=None)
    try:
        disposition = EntrypointDisposition(
            _non_empty_string(
                disposition_raw,
                name=f"entrypoints.routes.{route.value}.disposition",
            )
        )
    except ValueError as exc:
        raise EntrypointContractError(
            f"entrypoints.routes.{route.value} has unknown disposition {disposition_raw!r}"
        ) from exc
    adapter_backend = _optional_string(
        OmegaConf.select(cast(Any, raw), "adapter_backend", default=None),
        name=f"entrypoints.routes.{route.value}.adapter_backend",
    )
    if disposition is EntrypointDisposition.EXPLICIT_ADAPTER and adapter_backend is None:
        raise EntrypointContractError(
            f"entrypoints.routes.{route.value} explicit_adapter requires adapter_backend"
        )
    if disposition is not EntrypointDisposition.EXPLICIT_ADAPTER and adapter_backend is not None:
        raise EntrypointContractError(
            f"entrypoints.routes.{route.value} may declare adapter_backend only for explicit_adapter"
        )
    return disposition, adapter_backend


def _export_formats(cfg: DictConfig) -> tuple[str, ...]:
    raw = OmegaConf.select(cfg, "entrypoints.export_formats", default=[])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise EntrypointContractError("entrypoints.export_formats must be a sequence")
    formats = tuple(
        _non_empty_string(value, name="entrypoints.export_formats item") for value in raw
    )
    if len(set(formats)) != len(formats):
        raise EntrypointContractError("entrypoints.export_formats must not contain duplicates")
    unknown = sorted(set(formats) - {"onnx", "jit"})
    if unknown:
        raise EntrypointContractError(
            f"entrypoints.export_formats contains unsupported values: {unknown}"
        )
    return formats


def _validate_declared_identity(
    cfg: DictConfig,
    *,
    identity: EntrypointExecutionIdentity,
) -> None:
    declared = OmegaConf.select(cfg, "entrypoints.identity", default=None)
    if declared is None:
        return
    if not OmegaConf.is_config(declared) and not isinstance(declared, Mapping):
        raise EntrypointContractError("entrypoints.identity must be a mapping")

    actual = identity.to_snapshot()
    for field, actual_value in actual.items():
        expected = OmegaConf.select(cast(Any, declared), field, default=None)
        if expected is None:
            continue
        expected_value = _non_empty_string(
            expected,
            name=f"entrypoints.identity.{field}",
        )
        if expected_value != actual_value:
            raise EntrypointContractError(
                f"owner-declared entrypoint identity mismatch for {field}: "
                f"expected {expected_value!r}, got {actual_value!r}. Select the matching "
                "task/backend owner YAML; do not override backend, execution profile, or "
                "runtime identity fields independently."
            )


def _require_policy_source_metadata(source_run_dir: str | Path) -> None:
    run_dir = Path(source_run_dir)
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise EntrypointContractError(
            f"Policy source metadata is missing: {config_path}. Use a checkpoint produced "
            "by a versioned UniLab entrypoint; policy loads cannot bypass config or managed "
            "ABI guards."
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EntrypointContractError(
            f"Policy source metadata is malformed: {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise EntrypointContractError(
            f"Policy source metadata must be a JSON object: {config_path}"
        )
    snapshot = payload.get("contract_snapshot")
    if not isinstance(snapshot, Mapping):
        raise EntrypointContractError(
            f"Policy source metadata has no contract_snapshot object: {config_path}. "
            "Retrain the policy instead of loading an unverifiable legacy run."
        )
    source_cfg = payload.get("config")
    if not isinstance(source_cfg, Mapping):
        raise EntrypointContractError(
            f"Policy source metadata has no resolved config object: {config_path}"
        )
    source_entrypoints = source_cfg.get("entrypoints")
    if not isinstance(source_entrypoints, Mapping):
        raise EntrypointContractError(
            f"Policy source metadata has no versioned entrypoint contract: {config_path}. "
            "Retrain the policy with the current owner config."
        )
    source_fingerprint = source_entrypoints.get("fingerprint")
    if source_fingerprint != ENTRYPOINT_CONTRACT_FINGERPRINT:
        raise EntrypointContractError(
            f"Policy source entrypoint fingerprint must be "
            f"{ENTRYPOINT_CONTRACT_FINGERPRINT!r}, got {source_fingerprint!r} in "
            f"{config_path}."
        )


def resolve_entrypoint_contract(
    cfg: DictConfig,
    route: EntrypointRoute | str,
) -> EntrypointRouteContract:
    """Resolve one route exclusively from the composed owner config."""

    try:
        resolved_route = route if isinstance(route, EntrypointRoute) else EntrypointRoute(route)
    except ValueError as exc:
        raise EntrypointContractError(f"unknown entrypoint route {route!r}") from exc
    fingerprint = _non_empty_string(
        OmegaConf.select(cfg, "entrypoints.fingerprint", default=None),
        name="entrypoints.fingerprint",
    )
    if fingerprint != ENTRYPOINT_CONTRACT_FINGERPRINT:
        raise EntrypointContractError(
            f"entrypoint owner fingerprint must be {ENTRYPOINT_CONTRACT_FINGERPRINT!r}, "
            f"got {fingerprint!r}"
        )
    disposition, adapter_backend = _route_declaration(cfg, resolved_route)
    backend = _non_empty_string(
        OmegaConf.select(cfg, "training.sim_backend", default=None),
        name="training.sim_backend",
    )
    profile = (
        _optional_string(
            OmegaConf.select(cfg, "training.execution_profile", default=None),
            name="training.execution_profile",
        )
        or "host_numpy"
    )
    runtime_impl = (
        _optional_string(
            OmegaConf.select(cfg, "algo.runtime_impl", default=None),
            name="algo.runtime_impl",
        )
        or "rsl_rl_default"
    )
    runtime_resolver = _optional_string(
        OmegaConf.select(cfg, "algo.runtime_resolver", default=None),
        name="algo.runtime_resolver",
    )
    renderer_backend = _optional_string(
        OmegaConf.select(cfg, "entrypoints.renderer_backend", default=None),
        name="entrypoints.renderer_backend",
    )
    if resolved_route in _RENDER_ROUTES:
        if disposition is EntrypointDisposition.NATIVE:
            if renderer_backend is None:
                renderer_backend = backend
            elif renderer_backend != backend:
                raise EntrypointContractError(
                    f"entrypoints.routes.{resolved_route.value}=native requires "
                    f"renderer_backend={backend!r}, got {renderer_backend!r}; declare an "
                    "explicit_adapter route instead of mixing backend identities"
                )
        elif disposition is EntrypointDisposition.EXPLICIT_ADAPTER:
            if renderer_backend is not None and renderer_backend != adapter_backend:
                raise EntrypointContractError(
                    f"entrypoints.routes.{resolved_route.value} declares adapter_backend="
                    f"{adapter_backend!r} but renderer_backend={renderer_backend!r}"
                )
            renderer_backend = adapter_backend
        elif renderer_backend is not None:
            raise EntrypointContractError(
                f"entrypoints.routes.{resolved_route.value}=unsupported requires "
                "entrypoints.renderer_backend=null"
            )
    else:
        renderer_backend = None

    export_formats = _export_formats(cfg) if resolved_route is EntrypointRoute.EXPORT else ()
    if (
        resolved_route is EntrypointRoute.EXPORT
        and disposition is not EntrypointDisposition.UNSUPPORTED
        and not export_formats
    ):
        raise EntrypointContractError(
            "a supported export route requires at least one entrypoints.export_formats value"
        )

    configured_diagnostic = OmegaConf.select(
        cfg,
        f"entrypoints.diagnostics.{resolved_route.value}",
        default=None,
    )
    if configured_diagnostic is None:
        diagnostic = (
            f"Entrypoint route {resolved_route.value!r} is unsupported for "
            f"backend={backend!r}, execution_profile={profile!r}, runtime={runtime_impl!r}. "
            f"The task/backend owner must declare entrypoints.routes.{resolved_route.value} "
            "as native or as an explicit adapter; do not substitute another physics backend."
        )
    else:
        diagnostic = _non_empty_string(
            configured_diagnostic,
            name=f"entrypoints.diagnostics.{resolved_route.value}",
        )

    policy_load = resolved_route in _POLICY_LOAD_ROUTES
    identity = EntrypointExecutionIdentity(
        task_name=_non_empty_string(
            OmegaConf.select(cfg, "training.task_name", default=None),
            name="training.task_name",
        ),
        backend=backend,
        execution_profile=profile,
        runtime_impl=runtime_impl,
        runtime_resolver=runtime_resolver,
    )
    _validate_declared_identity(cfg, identity=identity)
    return EntrypointRouteContract(
        fingerprint=fingerprint,
        route=resolved_route,
        disposition=disposition,
        identity=identity,
        checkpoint_required=policy_load,
        config_guard_required=policy_load,
        managed_abi_guard_required=policy_load,
        dimension_guard_required=policy_load,
        renderer_backend=renderer_backend,
        adapter_backend=adapter_backend,
        export_formats=export_formats,
        diagnostic=diagnostic,
    )


def require_entrypoint_route(
    contract: EntrypointRouteContract,
    *,
    renderer_backend: str | None = None,
) -> EntrypointRouteContract:
    """Fail before env construction for unsupported or identity-mixed routes."""

    if contract.disposition is EntrypointDisposition.UNSUPPORTED:
        raise EntrypointContractError(contract.diagnostic)
    if renderer_backend is not None:
        requested = _non_empty_string(renderer_backend, name="renderer_backend")
        if contract.renderer_backend != requested:
            raise EntrypointContractError(
                f"Entrypoint route {contract.route.value!r} for backend="
                f"{contract.identity.backend!r} declares renderer_backend="
                f"{contract.renderer_backend!r}, not {requested!r}. Do not rewrite "
                "the physics backend identity to launch a viewer."
            )
    return contract


def require_policy_load_contracts(
    cfg: DictConfig,
    route: EntrypointRoute | str,
) -> tuple[EntrypointRouteContract, EntrypointRouteContract]:
    """Require both an operation route and the shared checkpoint-load route."""

    operation = require_entrypoint_route(resolve_entrypoint_contract(cfg, route))
    if operation.route not in _POLICY_LOAD_ROUTES:
        raise EntrypointContractError(
            f"entrypoint route {operation.route.value!r} is not a policy-load operation"
        )
    checkpoint_load = (
        operation
        if operation.route is EntrypointRoute.CHECKPOINT_LOAD
        else require_entrypoint_route(
            resolve_entrypoint_contract(cfg, EntrypointRoute.CHECKPOINT_LOAD)
        )
    )
    return operation, checkpoint_load


def resolve_ppo_operation(cfg: DictConfig) -> EntrypointRoute:
    """Map the public PPO operation selector and legacy play flag to a route."""

    raw = str(OmegaConf.select(cfg, "training.operation", default="auto"))
    if raw == "auto":
        return (
            EntrypointRoute.PLAY
            if bool(OmegaConf.select(cfg, "training.play_only", default=False))
            else EntrypointRoute.TRAIN
        )
    if raw not in {"train", "play", "export"}:
        raise EntrypointContractError(
            "training.operation must be one of auto, train, play, or export"
        )
    route = EntrypointRoute(raw)
    play_only = bool(OmegaConf.select(cfg, "training.play_only", default=False))
    if play_only and route is not EntrypointRoute.PLAY:
        raise EntrypointContractError(
            "training.play_only=true conflicts with explicit training.operation=" + raw
        )
    return route


def policy_load_target(
    *,
    managed_policy_abi: Mapping[str, Any] | None,
    observation_dim: int | None,
    action_dim: int | None,
) -> PolicyLoadTarget:
    """Normalize target runtime metadata after cold-path materialization."""

    normalized = (
        None
        if managed_policy_abi is None
        else normalize_managed_policy_abi_snapshot(managed_policy_abi)
    )
    fingerprint = None if normalized is None else str(normalized["policy_abi_fingerprint"])
    for name, value in (("observation_dim", observation_dim), ("action_dim", action_dim)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise EntrypointContractError(f"{name} must be a positive integer or None")
    return PolicyLoadTarget(
        managed_policy_abi=normalized,
        managed_policy_abi_fingerprint=fingerprint,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )


def preflight_policy_source(
    *,
    source_run_dir: str | Path,
    target_cfg: DictConfig,
    algo_name: str,
    strict: bool,
) -> DictConfig:
    """Validate config fields before materializing the target managed runtime."""

    _require_policy_source_metadata(source_run_dir)
    return (
        resolve_sim2sim_config(
            source_run_dir,
            target_cfg,
            algo_name=algo_name,
            strict=strict,
            defer_managed_policy_abi=True,
        )
        or target_cfg
    )


@contextmanager
def guarded_policy_load(
    *,
    contract: EntrypointRouteContract,
    source_run_dir: str | Path,
    target_cfg: DictConfig,
    target: PolicyLoadTarget,
    algo_name: str,
    strict: bool,
) -> Iterator[None]:
    """Apply canonical managed ABI and tensor-dimension guards before a real load."""

    require_entrypoint_route(contract)
    if not contract.checkpoint_required:
        raise EntrypointContractError(
            f"entrypoint route {contract.route.value!r} is not a checkpoint-load operation"
        )
    _require_policy_source_metadata(source_run_dir)
    resolve_sim2sim_config(
        source_run_dir,
        target_cfg,
        algo_name=algo_name,
        strict=strict,
        managed_policy_abi=target.managed_policy_abi,
    )
    with policy_load_dim_guard(
        env_obs_dim=target.observation_dim,
        env_action_dim=target.action_dim,
        algo_name=algo_name,
        managed_policy_abi_fingerprint=target.managed_policy_abi_fingerprint,
    ):
        yield


def build_entrypoint_receipt(
    contract: EntrypointRouteContract,
    *,
    checkpoint: str | Path | None,
    target: PolicyLoadTarget | None,
    outputs: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build JSON-ready route evidence without exposing backend-private state."""

    return {
        "contract": contract.to_snapshot(),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "managed_policy_abi_fingerprint": (
            None if target is None else target.managed_policy_abi_fingerprint
        ),
        "observation_dim": None if target is None else target.observation_dim,
        "action_dim": None if target is None else target.action_dim,
        "outputs": [str(output) for output in outputs],
    }


__all__ = [
    "ENTRYPOINT_CONTRACT_FINGERPRINT",
    "EntrypointContractError",
    "EntrypointDisposition",
    "EntrypointExecutionIdentity",
    "EntrypointRoute",
    "EntrypointRouteContract",
    "PolicyLoadTarget",
    "build_entrypoint_receipt",
    "guarded_policy_load",
    "policy_load_target",
    "preflight_policy_source",
    "require_entrypoint_route",
    "require_policy_load_contracts",
    "resolve_entrypoint_contract",
    "resolve_ppo_operation",
]
