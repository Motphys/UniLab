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

* ``DENYLIST``     - a difference raises :class:`CrossBackendIncompatibleError`. For
  the env-structural subset (:data:`ENV_STRUCTURAL_DENYLIST`) an *asymmetric presence*
  (set on one side, absent on the other) also raises, because the absent side silently
  falls back to an env default that may differ (issue #579 P2).
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
from collections.abc import Iterator
from contextlib import contextmanager
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

# DENYLIST fields backed by an env dataclass default rather than a config.yaml default,
# so they are absent from the composed config when a backend does not set them. Unlike
# the algo-specific fields (empirical_normalization / obs_normalization, legitimately
# absent across algos), these are always meaningful at runtime, so an asymmetric
# presence between source and target is treated as a contract mismatch (fail closed)
# instead of skipped. See issue #579 P2 and ``resolve_sim2sim_config`` below.
ENV_STRUCTURAL_DENYLIST: list[str] = [path for path in DENYLIST if path.startswith("env.")]


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


def _asymmetric_line(path: str, present_value: Any, *, source_present: bool) -> str:
    """Format a denial for an env-structural field set on exactly one side.

    The other side omits it and falls back to an env dataclass default, which the guard
    cannot resolve, so the contract is unverifiable and we fail closed (issue #579 P2).
    """
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

    Also raises (when ``strict``) when an :data:`ENV_STRUCTURAL_DENYLIST` field is
    present on exactly one side (asymmetric presence): the absent side falls back to an
    env default the guard cannot resolve, so the contract is unverifiable (issue #579
    P2). The algo-specific fields keep the skip-on-absent behavior so legitimate
    cross-algo plays are unaffected.
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
            # Target does not set this field. Algo-specific fields are legitimately
            # absent across algos (e.g. PPO empirical_normalization vs off-policy
            # obs_normalization), so skip them. But an env structural field is backed
            # by a dataclass default that is always meaningful at runtime, so an
            # asymmetric presence is unverifiable -> fail closed (issue #579 P2).
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

    # Reverse asymmetry: an env structural field the target sets explicitly but the
    # source snapshot never recorded (the trained run used the env default). Also
    # unverifiable, so fail closed (issue #579 P2).
    for path in ENV_STRUCTURAL_DENYLIST:
        if path in snapshot:
            continue  # presence already handled by the snapshot loop above
        if _select(target_cfg, path) is not None:
            denials.append(_asymmetric_line(path, _select(target_cfg, path), source_present=False))

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


# Substrings that mark a tensor shape/size mismatch in torch / mlx load errors. Matched
# case-insensitively; used only to *re-label* a load that already failed, so a broad set
# is safe (non-matching errors are re-raised unchanged).
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
) -> Iterator[None]:
    """Wrap a play-time checkpoint load and turn a tensor shape/size mismatch into a
    clear cross-backend sim2sim diagnostic (issue #579 runtime dimension check).

    The trained policy network's weight shapes are fixed in the checkpoint. Loading it
    into an env whose actual observation/action dimensions differ -- a sim2sim mismatch
    the YAML-level guard cannot see, because the real dims come from
    ``env.obs_groups_spec`` / the action space, not the config -- makes
    ``load_state_dict`` / ``load_weights`` raise a cryptic size-mismatch error (and note
    that a size mismatch raises even under ``strict=False``). This wrapper re-raises that
    as an actionable :class:`CrossBackendIncompatibleError` naming the env dims.

    It only ever acts on a load that ALREADY failed (non-matching errors propagate
    unchanged), so it cannot block an otherwise-valid play. ``env_obs_dim`` /
    ``env_action_dim`` are informational only; pass ``None`` if not readily available.
    """
    try:
        yield
    except (RuntimeError, ValueError) as exc:  # torch -> RuntimeError, mlx -> ValueError
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


class Sim2SimConfigResolver:
    """Object facade for the cross-backend sim2sim contract (issue #579 RFC name).

    The RFC names this type; it is a thin, stateless wrapper over the module-level
    functions so callers may use either style. The field lists remain the single source
    of truth as module constants and are re-exposed here as class attributes.
    """

    ALLOWLIST = ALLOWLIST
    WARNING_LIST = WARNING_LIST
    DENYLIST = DENYLIST
    ENV_STRUCTURAL_DENYLIST = ENV_STRUCTURAL_DENYLIST

    @staticmethod
    def extract_snapshot(full_cfg: DictConfig) -> dict[str, Any]:
        """See :func:`extract_contract_snapshot`."""
        return extract_contract_snapshot(full_cfg)

    @staticmethod
    def resolve(
        source_run_dir: str | Path | None,
        target_cfg: DictConfig,
        *,
        algo_name: str | None = None,
        strict: bool = True,
    ) -> DictConfig | None:
        """See :func:`resolve_sim2sim_config`. ``strict=False`` downgrades DENYLIST
        denials to warnings (user-level bypass)."""
        return resolve_sim2sim_config(
            source_run_dir, target_cfg, algo_name=algo_name, strict=strict
        )

    @staticmethod
    def load_dim_guard(
        *,
        env_obs_dim: int | None = None,
        env_action_dim: int | None = None,
        algo_name: str | None = None,
    ):
        """See :func:`policy_load_dim_guard`."""
        return policy_load_dim_guard(
            env_obs_dim=env_obs_dim, env_action_dim=env_action_dim, algo_name=algo_name
        )
