"""Cross-backend sim2sim contract snapshot and resolution."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.manager.fingerprint import (
    ManagedPolicyABISnapshotError,
    normalize_managed_policy_abi_snapshot,
)
from unilab.training.retirement import check_retired_checkpoint


class CrossBackendIncompatibleError(RuntimeError):
    """Raised when a target play config diverges from the source training contract."""


ALLOWLIST: list[str] = [
    "training.sim_backend",
    "env.scene",
    "training.play_steps",
    "env.domain_rand",
    "env.noise_config",
    "env.commands.vel_limit",
]

WARNING_LIST: list[str] = [
    "reward.scales",
    "reward.base_height_target",
    "reward.max_tilt_deg",
    "reward.min_base_height",
    "env.control_config.simulate_action_latency",
    "env.ctrl_dt",
]

DENYLIST: list[str] = [
    "algo.obs_groups",
    "env.control_config.action_scale",
    "algo.policy.actor_hidden_dims",
    "algo.policy.critic_hidden_dims",
    "algo.empirical_normalization",
    "algo.obs_normalization",
    "env.sampling_mode",
]

SNAPSHOT_FIELDS: list[str] = DENYLIST + WARNING_LIST

MANAGED_POLICY_ABI_SNAPSHOT_KEY = "manager.policy_abi"
"""Reserved run-snapshot key for caller-provided compiled manager policy ABI."""

ENV_STRUCTURAL_DENYLIST: list[str] = [path for path in DENYLIST if path.startswith("env.")]


def _select(cfg: Any, path: str) -> Any:
    """Return the effective value at a dotted path (or ``None`` if absent)."""
    return OmegaConf.select(cfg, path)


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def extract_contract_snapshot(
    full_cfg: Any,
    *,
    managed_policy_abi: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract config and optional compiled-manager policy contract fields.

    A compiled manager plan is intentionally supplied by the cold-path caller
    rather than inferred from Hydra config.  That keeps selector/model/backend
    materialization out of this generic training helper and makes an omitted
    ABI visible to the cross-backend resolver instead of silently guessed.
    """
    cfg: Any = full_cfg if OmegaConf.is_config(full_cfg) else OmegaConf.create(full_cfg)
    snapshot: dict[str, Any] = {}
    for path in SNAPSHOT_FIELDS:
        value = _select(cfg, path)
        if value is None:
            continue
        snapshot[path] = _to_plain(value)
    if managed_policy_abi is not None:
        try:
            snapshot[MANAGED_POLICY_ABI_SNAPSHOT_KEY] = normalize_managed_policy_abi_snapshot(
                managed_policy_abi
            )
        except ManagedPolicyABISnapshotError as exc:
            raise ValueError(f"invalid managed policy ABI snapshot: {exc}") from exc
    return snapshot


def _normalize(value: Any) -> Any:
    """Canonicalize a value for order-insensitive, type-tolerant comparison."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, bool):  # must precede int: bool is a subclass of int
        return value
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)  # 0 == 0.0; YAML-int vs JSON-float parity
    return value


def _values_equal(a: Any, b: Any) -> bool:
    return bool(_normalize(a) == _normalize(b))


def _format_value(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True)


def _diff_line(path: str, source_value: Any, target_value: Any) -> str:
    return f"{path}: source={_format_value(source_value)} target={_format_value(target_value)}"


def _asymmetric_line(path: str, present_value: Any, *, source_present: bool) -> str:
    """Format a denial for an env-structural field set on exactly one side."""
    value = _format_value(present_value)
    if source_present:
        return (
            f"{path}: source={value} target=<absent> (target omits this field and "
            "falls back to the env default, which may differ; set it explicitly in the "
            "target task YAML to make the contract verifiable)"
        )
    return (
        f"{path}: source=<absent> target={value} (the trained run omitted this field "
        "and used the env default; set it explicitly so the contract can be verified)"
    )


def _read_snapshot(run_dir: Path) -> dict[str, Any] | None:
    """Read ``contract_snapshot`` from ``run_dir/run_config.json`` (``None`` if absent)."""
    path = run_dir / "run_config.json"
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    snapshot = parsed.get("contract_snapshot")
    if not isinstance(snapshot, dict):
        return None
    return snapshot


def _normalize_managed_policy_abi_for_resolver(
    value: object,
    *,
    side: str,
) -> dict[str, Any]:
    """Turn malformed manager ABI metadata into an actionable fail-closed error."""

    try:
        return normalize_managed_policy_abi_snapshot(value)
    except ManagedPolicyABISnapshotError as exc:
        raise CrossBackendIncompatibleError(
            f"Invalid {side} managed policy ABI snapshot: {exc}. "
            "Recompile the managed task on the cold path and pass its canonical "
            "policy snapshot to resolve_sim2sim_config."
        ) from exc


# Fields that carry backend-local execution identity — they differ legitimately
# between independent compilation runs (different GPU ordinal, host vs device
# executor) and must NOT influence the policy I/O compatibility decision.
_ABI_EXECUTION_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"plan_fingerprint", "executor_key", "execution_profile"}
)


def _semantic_abi_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return the policy-semantic subset of a normalized ABI snapshot.

    Strips backend-local execution-identity fields (``plan_fingerprint``,
    ``executor_key``, ``execution_profile``) so that two snapshots compiled
    on different GPU ordinals or executor profiles compare equal whenever
    their observable policy I/O contract is identical.
    """
    return {k: v for k, v in snapshot.items() if k not in _ABI_EXECUTION_IDENTITY_FIELDS}


def _managed_policy_abi_asymmetric_line(*, source_present: bool) -> str:
    if source_present:
        return (
            f"{MANAGED_POLICY_ABI_SNAPSHOT_KEY}: source=<present> target=<absent> "
            "(a managed policy was trained, but the target did not supply a compiled "
            "policy ABI; construct/compile the target plan on its cold path first)"
        )
    return (
        f"{MANAGED_POLICY_ABI_SNAPSHOT_KEY}: source=<absent> target=<present> "
        "(the source run predates or omitted the managed policy ABI; retrain with a "
        "canonical snapshot rather than loading it as a verified managed policy)"
    )


def resolve_sim2sim_config(
    source_run_dir: str | Path | None,
    target_cfg: DictConfig,
    *,
    algo_name: str | None = None,
    strict: bool = True,
    managed_policy_abi: Mapping[str, Any] | None = None,
    defer_managed_policy_abi: bool = False,
) -> DictConfig | None:
    """Validate a target play config against the source training contract.

    Returns ``None`` if ``source_run_dir`` is ``None``; otherwise returns ``target_cfg``
    unchanged (never mutated). ``managed_policy_abi`` is an optional canonical
    snapshot rendered from a target :class:`~unilab.manager.CompiledTaskPlan`
    on its cold path.  If either source or target declares that extension, it
    is bidirectionally required and compared fail-closed alongside config
    DENYLIST fields.  This avoids a dimensionally compatible but semantically
    different managed policy being loaded across backends.

    ``defer_managed_policy_abi`` performs only the pre-materialization config
    pass.  It still validates a source ABI if present, but deliberately defers
    presence/equality checks until the caller has cold-materialized the target
    plan and invokes this function again with its canonical ABI.

    Raises :class:`CrossBackendIncompatibleError` under ``strict`` when any
    DENYLIST field differs, including asymmetric presence for
    :data:`ENV_STRUCTURAL_DENYLIST` paths.  A malformed managed ABI always
    raises, including in non-strict mode, because it cannot be compared safely.
    A source run produced by the retired device-resident mjwarp path raises
    :class:`~unilab.training.retirement.RetiredDevicePathError` before any
    contract comparison.
    """
    if not isinstance(defer_managed_policy_abi, bool):
        raise TypeError("defer_managed_policy_abi must be a bool")
    if defer_managed_policy_abi and managed_policy_abi is not None:
        raise ValueError("managed_policy_abi must be omitted during the pre-materialization pass")
    if source_run_dir is None:
        print("[sim2sim] no source run dir; skipping cross-backend contract check")
        return None

    run_dir = Path(source_run_dir)
    # A run produced by the retired device-resident mjwarp path must surface an
    # explicit retirement diagnostic, not a generic contract mismatch.
    check_retired_checkpoint(run_dir)
    snapshot = _read_snapshot(run_dir)
    if snapshot is None:
        print(
            f"[sim2sim] {run_dir}/run_config.json has no contract_snapshot "
            "(old run); skipping cross-backend enforcement"
        )
        return target_cfg

    denials: list[str] = []
    for path, source_value in snapshot.items():
        if path == MANAGED_POLICY_ABI_SNAPSHOT_KEY:
            continue
        target_value = _select(target_cfg, path)
        if target_value is None:
            if path in ENV_STRUCTURAL_DENYLIST:
                denials.append(_asymmetric_line(path, source_value, source_present=True))
            continue
        if _values_equal(source_value, target_value):
            continue
        line = _diff_line(path, source_value, target_value)
        if path in DENYLIST:
            denials.append(line)
        else:
            print(f"[sim2sim] WARNING override {line}")

    for path in ENV_STRUCTURAL_DENYLIST:
        if path in snapshot:
            continue
        if _select(target_cfg, path) is not None:
            denials.append(_asymmetric_line(path, _select(target_cfg, path), source_present=False))

    source_has_managed_abi = MANAGED_POLICY_ABI_SNAPSHOT_KEY in snapshot
    target_has_managed_abi = managed_policy_abi is not None
    if source_has_managed_abi:
        source_managed_abi = _normalize_managed_policy_abi_for_resolver(
            snapshot[MANAGED_POLICY_ABI_SNAPSHOT_KEY], side="source"
        )
    else:
        source_managed_abi = None
    if target_has_managed_abi:
        target_managed_abi = _normalize_managed_policy_abi_for_resolver(
            managed_policy_abi, side="target"
        )
    else:
        target_managed_abi = None
    if not defer_managed_policy_abi:
        if source_has_managed_abi and not target_has_managed_abi:
            denials.append(_managed_policy_abi_asymmetric_line(source_present=True))
        elif target_has_managed_abi and not source_has_managed_abi:
            denials.append(_managed_policy_abi_asymmetric_line(source_present=False))
        elif source_managed_abi is not None and target_managed_abi is not None:
            # Compare only the policy-semantic payload — strip backend-local
            # execution-identity fields (plan_fingerprint, executor_key,
            # execution_profile) that differ legitimately across GPU ordinals
            # and executor profiles without affecting policy I/O compatibility.
            source_semantic = _semantic_abi_payload(source_managed_abi)
            target_semantic = _semantic_abi_payload(target_managed_abi)
            if not _values_equal(source_semantic, target_semantic):
                denials.append(
                    _diff_line(
                        MANAGED_POLICY_ABI_SNAPSHOT_KEY,
                        source_semantic,
                        target_semantic,
                    )
                )

    if denials:
        message = (
            "Cross-backend sim2sim contract mismatch between the trained policy and "
            f"the target play config.\nSource run: {run_dir}\n"
            "The following policy-defining fields differ and must be reconciled in "
            "the target task YAML or compiled plan:\n  " + "\n  ".join(denials)
        )
        if strict:
            raise CrossBackendIncompatibleError(message)
        print(f"[sim2sim] WARNING (non-strict) {message}")

    return target_cfg


_DIM_MISMATCH_MARKERS: tuple[str, ...] = (
    "size mismatch",
    "copying a param",
    "shape",
    "dimension",
    "expected",
)


def _looks_like_dim_mismatch(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in _DIM_MISMATCH_MARKERS)


@contextmanager
def policy_load_dim_guard(
    *,
    env_obs_dim: int | None = None,
    env_action_dim: int | None = None,
    algo_name: str | None = None,
    managed_policy_abi_fingerprint: str | None = None,
) -> Iterator[None]:
    """Re-raise a tensor shape mismatch during checkpoint load as a sim2sim diagnostic.

    ``managed_policy_abi_fingerprint`` is diagnostic-only: compatibility must
    already have been checked by :func:`resolve_sim2sim_config` before an env
    or checkpoint is constructed.  Non-matching errors propagate unchanged,
    so a valid load is never blocked.
    """
    if managed_policy_abi_fingerprint is not None and (
        not isinstance(managed_policy_abi_fingerprint, str)
        or not managed_policy_abi_fingerprint.strip()
    ):
        raise ValueError("managed_policy_abi_fingerprint must be a non-empty string or None")
    try:
        yield
    except (RuntimeError, ValueError) as exc:
        if not _looks_like_dim_mismatch(str(exc)):
            raise
        managed_abi_line = ""
        if managed_policy_abi_fingerprint is not None:
            managed_abi_line = f"  managed policy ABI: {managed_policy_abi_fingerprint}\n"
        message = (
            (
                "Trained policy checkpoint does not fit this play environment -- likely a "
                "cross-backend sim2sim dimension mismatch.\n"
                f"  algo: {algo_name}\n"
                f"  env policy obs dim: {env_obs_dim}\n"
                f"  env action dim: {env_action_dim}\n"
            )
            + managed_abi_line
            + (
                "The checkpoint's tensor shapes do not match the env's observation/action "
                "dimensions. Check the task's obs_groups_spec and action space across "
                "backends; see resolve_sim2sim_config and run "
                "`uv run scripts/audit_sim2sim_contracts.py`.\n"
                f"Original load error:\n{exc}"
            )
        )
        raise CrossBackendIncompatibleError(message) from exc


class Sim2SimConfigResolver:
    """Object facade over the module-level sim2sim contract API."""

    ALLOWLIST = ALLOWLIST
    WARNING_LIST = WARNING_LIST
    DENYLIST = DENYLIST
    ENV_STRUCTURAL_DENYLIST = ENV_STRUCTURAL_DENYLIST
    MANAGED_POLICY_ABI_SNAPSHOT_KEY = MANAGED_POLICY_ABI_SNAPSHOT_KEY

    @staticmethod
    def extract_snapshot(
        full_cfg: Any,
        *,
        managed_policy_abi: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """See :func:`extract_contract_snapshot`."""
        return extract_contract_snapshot(full_cfg, managed_policy_abi=managed_policy_abi)

    @staticmethod
    def resolve(
        source_run_dir: str | Path | None,
        target_cfg: DictConfig,
        *,
        algo_name: str | None = None,
        strict: bool = True,
        managed_policy_abi: Mapping[str, Any] | None = None,
        defer_managed_policy_abi: bool = False,
    ) -> DictConfig | None:
        """See :func:`resolve_sim2sim_config`."""
        return resolve_sim2sim_config(
            source_run_dir,
            target_cfg,
            algo_name=algo_name,
            strict=strict,
            managed_policy_abi=managed_policy_abi,
            defer_managed_policy_abi=defer_managed_policy_abi,
        )

    @staticmethod
    def load_dim_guard(
        *,
        env_obs_dim: int | None = None,
        env_action_dim: int | None = None,
        algo_name: str | None = None,
        managed_policy_abi_fingerprint: str | None = None,
    ):
        """See :func:`policy_load_dim_guard`."""
        return policy_load_dim_guard(
            env_obs_dim=env_obs_dim,
            env_action_dim=env_action_dim,
            algo_name=algo_name,
            managed_policy_abi_fingerprint=managed_policy_abi_fingerprint,
        )
