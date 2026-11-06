"""Cross-backend sim2sim contract snapshot and resolution."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


class CrossBackendIncompatibleError(RuntimeError):
    """Raised when a target play config diverges from the source training contract."""


ALLOWLIST: list[str] = [
    "training.sim_backend",
    "env.scene",
    "training.play_steps",
    "env.noise_config",
]

WARNING_LIST: list[str] = [
    "reward.scales",
    "reward.base_height_target",
    "reward.max_tilt_deg",
    "reward.min_base_height",
    "env.control_config.simulate_action_latency",
]

DENYLIST: list[str] = [
    "algo.obs_groups",
    "env.control_config.action_scale",
    "env.observations",
    "env.actions",
    "env.policy_observation_group",
    "env.critic_observation_group",
    "algo.actor.hidden_dims",
    "algo.critic.hidden_dims",
    "algo.actor.obs_normalization",
    "algo.critic.obs_normalization",
    "algo.obs_normalization",
    "env.ctrl_dt",
    "env.commands.vel_limit",
    "env.commands.motion.params.sampling_mode",
]

SNAPSHOT_FIELDS: list[str] = DENYLIST + WARNING_LIST

# Action-term fields that only drive playback visualization (debug overlays).
# They never affect policy I/O or action semantics, so toggling them for
# interactive play is not a policy-defining change; the ``env.actions``
# comparison strips them before diffing.
VISUALIZATION_ONLY_ACTION_FIELDS: frozenset[str] = frozenset({"log_goal", "show_rays"})

ENV_STRUCTURAL_DENYLIST: list[str] = [path for path in DENYLIST if path.startswith("env.")]

_MINIMUM_SUPPORTED_FORMAT = (
    "Minimum supported format: a run_config.json whose 'contract_snapshot' is a "
    "mapping keyed by the canonical contract paths (the snapshot schema written "
    "by training runs on the UniLab 1.3.5 baseline). Older runs, legacy snapshot "
    "keys, and damaged files are not migrated or aliased; they are rejected.\n"
    "Recovery: retrain the policy with the current UniLab version so the run "
    "writes a supported snapshot (checkpoint weights themselves are unchanged), "
    "or play with no source run (source_run_dir=None) to explicitly skip the "
    "contract check."
)


def _unsupported_snapshot(run_dir: Path, reason: str) -> CrossBackendIncompatibleError:
    return CrossBackendIncompatibleError(
        f"Unsupported sim2sim source run: {reason}\n"
        f"Source run: {run_dir}\n"
        f"{_MINIMUM_SUPPORTED_FORMAT}"
    )


def _select(cfg: Any, path: str) -> Any:
    """Return the effective value at a dotted path (or ``None`` if absent)."""
    return OmegaConf.select(cfg, path)


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def extract_contract_snapshot(full_cfg: DictConfig) -> dict[str, Any]:
    """Extract the contract fields from a resolved training config keyed by dotted path."""
    cfg: Any = full_cfg if OmegaConf.is_config(full_cfg) else OmegaConf.create(full_cfg)
    snapshot: dict[str, Any] = {}
    for path in SNAPSHOT_FIELDS:
        value = _select(cfg, path)
        if value is None:
            continue
        snapshot[path] = _to_plain(value)
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


def _strip_visualization_only_fields(value: Any) -> Any:
    """Recursively drop playback-visualization-only leaves from an action mapping."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        return value
    stripped: dict[str, Any] = {}
    for key, child in value.items():
        if key in VISUALIZATION_ONLY_ACTION_FIELDS:
            continue
        stripped[str(key)] = _strip_visualization_only_fields(child)
    return stripped


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


def _read_snapshot(run_dir: Path) -> dict[str, Any]:
    """Read the canonical ``contract_snapshot`` mapping from a source run, failing closed.

    Raises :class:`CrossBackendIncompatibleError` when ``run_config.json`` is
    missing, is not valid JSON, is not a top-level mapping, or has no mapping
    ``contract_snapshot``. Historical ``run_config.json`` files are immutable
    inputs: nothing here repairs, rewrites, or migrates them.
    """
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise _unsupported_snapshot(run_dir, f"{path} does not exist.")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _unsupported_snapshot(run_dir, f"{path} is not valid JSON ({exc}).") from exc
    if not isinstance(parsed, dict):
        raise _unsupported_snapshot(run_dir, f"{path} top level is not a JSON object.")
    if "contract_snapshot" not in parsed:
        raise _unsupported_snapshot(run_dir, f"{path} has no 'contract_snapshot' field.")
    snapshot = parsed["contract_snapshot"]
    if not isinstance(snapshot, dict):
        raise _unsupported_snapshot(run_dir, f"{path} 'contract_snapshot' is not a mapping.")
    return snapshot


def resolve_sim2sim_config(
    source_run_dir: str | Path | None,
    target_cfg: DictConfig,
    *,
    algo_name: str | None = None,
    strict: bool = True,
) -> DictConfig | None:
    """Validate a target play config against the source training contract.

    Returns ``None`` if ``source_run_dir`` is ``None`` (an explicit "no source
    run" play); otherwise returns ``target_cfg`` unchanged (never mutated).
    Raises :class:`CrossBackendIncompatibleError` under ``strict`` when any
    DENYLIST field differs, including asymmetric presence for
    :data:`ENV_STRUCTURAL_DENYLIST` paths. Playback-visualization-only
    action fields (:data:`VISUALIZATION_ONLY_ACTION_FIELDS`) are exempt from
    the ``env.actions`` comparison. A source run whose contract snapshot
    is missing, malformed, or uses keys outside the current schema fails closed
    with the same error regardless of ``strict``.
    """
    if source_run_dir is None:
        print("[sim2sim] no source run dir; skipping cross-backend contract check")
        return None

    run_dir = Path(source_run_dir)
    snapshot = _read_snapshot(run_dir)

    unknown_paths = sorted(set(snapshot) - set(SNAPSHOT_FIELDS))
    if unknown_paths:
        raise _unsupported_snapshot(
            run_dir,
            "contract_snapshot contains keys outside the current contract schema: "
            + ", ".join(unknown_paths)
            + ". Retired legacy keys and snapshots from other versions are not "
            "resolved or skipped.",
        )

    denials: list[str] = []
    for path, source_value in snapshot.items():
        target_value = _select(target_cfg, path)
        if target_value is None:
            if path in ENV_STRUCTURAL_DENYLIST:
                denials.append(_asymmetric_line(path, source_value, source_present=True))
            continue
        if path == "env.actions":
            source_value = _strip_visualization_only_fields(source_value)
            target_value = _strip_visualization_only_fields(target_value)
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
        target_value = _select(target_cfg, path)
        if target_value is not None:
            denials.append(_asymmetric_line(path, target_value, source_present=False))

    if denials:
        message = (
            "Cross-backend sim2sim contract mismatch between the trained policy and "
            f"the target play config.\nSource run: {run_dir}\n"
            "The following policy-defining fields differ and must be reconciled in "
            "the target task YAML:\n  " + "\n  ".join(denials)
        )
        if strict:
            raise CrossBackendIncompatibleError(message)
        print(f"[sim2sim] WARNING (non-strict) {message}")

    return target_cfg


def _looks_like_dim_mismatch(message: str) -> bool:
    """Return whether ``load_state_dict`` reported a parameter size mismatch."""
    return "size mismatch for " in message.lower()


@contextmanager
def policy_load_dim_guard(
    *,
    env_obs_dim: int | None = None,
    env_action_dim: int | None = None,
    algo_name: str | None = None,
) -> Iterator[None]:
    """Re-raise a tensor shape mismatch during checkpoint load as a sim2sim diagnostic.

    Non-matching errors propagate unchanged, so a valid load is never blocked.
    """
    try:
        yield
    except (RuntimeError, ValueError) as exc:
        if not _looks_like_dim_mismatch(str(exc)):
            raise
        raise CrossBackendIncompatibleError(
            "Trained policy checkpoint does not fit this play environment -- likely a "
            "cross-backend sim2sim dimension mismatch.\n"
            f"  algo: {algo_name}\n"
            f"  env policy obs dim: {env_obs_dim}\n"
            f"  env action dim: {env_action_dim}\n"
            "The checkpoint's tensor shapes do not match the env's observation/action "
            "dimensions. Check the task's obs_groups_spec and action space across "
            "backends; see resolve_sim2sim_config and run "
            "`uv run scripts/audit_sim2sim_contracts.py`.\n"
            f"Original load error:\n{exc}"
        ) from exc
