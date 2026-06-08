"""Cross-backend sim2sim contract snapshot and resolution.

When a policy is trained on one simulation backend (MuJoCo / Motrix) and
"played" (sim2sim evaluation) on another, the target backend YAML is maintained
independently from the training config and frequently diverges. Checkpoints only
store weights, so a mismatch on a policy-defining field (observation grouping,
action scale, network width, observation normalization) silently corrupts the
loaded policy.

Training already writes ``run_config.json`` next to each checkpoint. This module
adds a compact ``contract_snapshot`` to that sidecar (see
:func:`extract_contract_snapshot`) and validates a target play config against the
source snapshot before the environment is created (see
:func:`resolve_sim2sim_config`):

* ``DENYLIST``     - a difference raises :class:`CrossBackendIncompatibleError`.
* ``WARNING_LIST`` - a difference is allowed but logged.
* ``ALLOWLIST``    - target-owned runtime/backend fields; never snapshotted nor
  compared.

The checkpoint format is untouched, so historical checkpoints keep working: a run
without ``contract_snapshot`` falls back to the target config with a warning.

This module intentionally depends only on the standard library and OmegaConf so it
can be imported from :mod:`unilab.training.experiment` without an import cycle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


class CrossBackendIncompatibleError(RuntimeError):
    """Raised when a target play config diverges from the source training contract
    on a policy-breaking (DENYLIST) field."""


# Fields the target backend may freely override; never snapshotted, never compared.
ALLOWLIST: list[str] = [
    "training.sim_backend",
    "env.scene",
    "training.play_steps",
    "env.domain_rand",
    "env.noise_config",
    "env.commands.vel_limit",
]

# Override allowed, but warn when the source training value differs from target.
WARNING_LIST: list[str] = [
    "reward.scales",
    "reward.base_height_target",
    "reward.max_tilt_deg",
    "reward.min_base_height",
    "env.control_config.simulate_action_latency",
    "env.ctrl_dt",
]

# A difference between source and target raises CrossBackendIncompatibleError.
# Scoped (per #579 decision) to fields that change policy I/O or network shape.
DENYLIST: list[str] = [
    "algo.obs_groups",
    "env.control_config.action_scale",
    "algo.policy.actor_hidden_dims",
    "algo.policy.critic_hidden_dims",
    "algo.empirical_normalization",  # PPO / APPO / MLX / HIM
    "algo.obs_normalization",  # off-policy (TD3 / SAC); skipped when absent
    "env.sampling_mode",  # motion-tracking tasks
]

# The snapshot stores exactly the fields we may need to compare at play time.
SNAPSHOT_FIELDS: list[str] = DENYLIST + WARNING_LIST


def _select(cfg: Any, path: str) -> Any:
    """Return the effective value at a dotted path (or ``None`` if absent)."""
    return OmegaConf.select(cfg, path)


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def extract_contract_snapshot(full_cfg: DictConfig) -> dict[str, Any]:
    """Extract the cross-backend contract fields from a resolved training config.

    Returns a flat mapping keyed by dotted config path. Fields that do not exist
    for the current algo/task are omitted (never stored as ``None``). Accepts a
    plain mapping as well as a ``DictConfig`` (some callers pass a plain dict).
    """
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


def _values_equal(a: Any, b: Any) -> bool:
    return bool(_normalize(a) == _normalize(b))


def _format_value(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True)


def _diff_line(path: str, source_value: Any, target_value: Any) -> str:
    return f"{path}: source={_format_value(source_value)} target={_format_value(target_value)}"


def _read_snapshot(run_dir: Path) -> dict[str, Any] | None:
    """Read ``contract_snapshot`` from ``run_dir/run_config.json``.

    Returns ``None`` for any missing/old/corrupt sidecar so playback never crashes
    on a bad file.
    """
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


def resolve_sim2sim_config(
    source_run_dir: str | Path | None,
    target_cfg: DictConfig,
    *,
    algo_name: str | None = None,
    strict: bool = True,
) -> DictConfig | None:
    """Validate a target play config against the source training contract.

    ``source_run_dir`` is the directory holding the source run's
    ``run_config.json`` (the checkpoint's run directory). The function never
    mutates ``target_cfg``; it returns:

    * ``None`` when there is no source directory to read (fresh/random play);
    * ``target_cfg`` unchanged when the contract validates, when the source run
      has no snapshot (old run), or when the sidecar is unreadable.

    Raises :class:`CrossBackendIncompatibleError` when ``strict`` and a DENYLIST
    field differs between the source snapshot and the target config. ``algo_name``
    is informational only (the normalization fields for every algo are in the
    DENYLIST and absent ones are skipped).
    """
    if source_run_dir is None:
        print("[sim2sim] no source run dir; skipping cross-backend contract check")
        return None

    run_dir = Path(source_run_dir)
    snapshot = _read_snapshot(run_dir)
    if snapshot is None:
        print(
            f"[sim2sim] {run_dir}/run_config.json has no contract_snapshot "
            "(old run); skipping cross-backend enforcement"
        )
        return target_cfg

    denials: list[str] = []
    for path, source_value in snapshot.items():
        target_value = _select(target_cfg, path)
        if target_value is None:
            continue  # target does not set this field; nothing to compare
        if _values_equal(source_value, target_value):
            continue
        line = _diff_line(path, source_value, target_value)
        if path in DENYLIST:
            denials.append(line)
        else:
            print(f"[sim2sim] WARNING override {line}")

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
