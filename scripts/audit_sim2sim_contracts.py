"""Audit cross-backend sim2sim contract divergences across task owner YAMLs.

For every task with >=2 backend YAMLs, hydra-composes each backend's effective config
and compares the DENYLIST / WARNING_LIST fields from ``unilab.training.sim2sim``.

Read-only.

    uv run scripts/audit_sim2sim_contracts.py
    uv run scripts/audit_sim2sim_contracts.py --trees ppo appo offpolicy
    uv run scripts/audit_sim2sim_contracts.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.training.sim2sim import DENYLIST, ENV_STRUCTURAL_DENYLIST, WARNING_LIST, _normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_ROOT = REPO_ROOT / "conf"
ABSENT = "<absent>"

PRIMARY_PAIR = ("mujoco", "motrix")


def _values_equal(a: Any, b: Any) -> bool:
    return _normalize(a) == _normalize(b)


def _fmt(value: Any) -> str:
    if value is ABSENT:
        return ABSENT
    try:
        return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _select(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path)
    return ABSENT if value is None else value


def _compose(tree: str, task_variant: str) -> Any:
    conf_dir = str(CONF_ROOT / tree)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        return compose("config", overrides=[f"task={task_variant}"])


def _discover(tree: str) -> dict[str, list[str]]:
    base = CONF_ROOT / tree / "task"
    out: dict[str, list[str]] = {}
    if not base.is_dir():
        return out
    for task_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        backends = sorted(p.stem for p in task_dir.glob("*.yaml") if p.stem != "base")
        if backends:
            out[task_dir.name] = backends
    return out


def _diff_field(path: str, mj: Any, mx: Any) -> dict[str, Any] | None:
    mj_absent, mx_absent = mj is ABSENT, mx is ABSENT
    if mj_absent and mx_absent:
        return None
    if mj_absent != mx_absent:
        kind = "asymmetric-presence"
    elif _values_equal(mj, mx):
        return None
    else:
        kind = "value-diff"
    guard_enforced = kind == "value-diff" or (
        kind == "asymmetric-presence" and path in ENV_STRUCTURAL_DENYLIST
    )
    return {
        "field": path,
        "kind": kind,
        "mujoco": _fmt(mj),
        "motrix": _fmt(mx),
        "guard_enforced": guard_enforced,
    }


def audit_tree(tree: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, backends in _discover(tree).items():
        values: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for backend in backends:
            try:
                cfg = _compose(tree, f"{task}/{backend}")
                values[backend] = {p: _select(cfg, p) for p in (DENYLIST + WARNING_LIST)}
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                errors[backend] = f"{type(exc).__name__}: {exc}"

        mj_name, mx_name = PRIMARY_PAIR
        if mj_name in values and mx_name in values:
            mj, mx = values[mj_name], values[mx_name]
            deny = [
                d for p in DENYLIST if (d := _diff_field(p, mj.get(p, ABSENT), mx.get(p, ABSENT)))
            ]
            warn = [
                d
                for p in WARNING_LIST
                if (d := _diff_field(p, mj.get(p, ABSENT), mx.get(p, ABSENT)))
            ]
            verdict = "TRANSFERABLE" if not deny else "BLOCKED"
        else:
            deny, warn, verdict = [], [], "N/A (no mujoco+motrix pair)"

        rows.append(
            {
                "tree": tree,
                "task": task,
                "backends": backends,
                "verdict": verdict,
                "deny_diffs": deny,
                "warn_diffs": warn,
                "errors": errors,
            }
        )
    return rows


def _print_human(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(f"\n### [{row['tree']}] {row['task']}  backends={row['backends']}")
        if row["errors"]:
            print(f"  COMPOSE ERRORS: {row['errors']}")
        print(f"  VERDICT (mujoco<->motrix): {row['verdict']}")
        for diff in row["deny_diffs"]:
            flag = (
                "" if diff["guard_enforced"] else "  [guard-blind-spot: re-check dataclass default]"
            )
            print(
                f"    DENY  {diff['field']} [{diff['kind']}]: "
                f"mujoco={diff['mujoco']}  motrix={diff['motrix']}{flag}"
            )
        for diff in row["warn_diffs"]:
            print(
                f"    warn  {diff['field']} [{diff['kind']}]: "
                f"mujoco={diff['mujoco']}  motrix={diff['motrix']}"
            )

    transferable = [r for r in rows if r["verdict"] == "TRANSFERABLE"]
    blocked = [r for r in rows if r["verdict"] == "BLOCKED"]
    blind = [r for r in blocked if any(not d["guard_enforced"] for d in r["deny_diffs"])]
    print("\n" + "=" * 80)
    print(
        f"TRANSFERABLE: {len(transferable)}   BLOCKED: {len(blocked)}   "
        f"(of which contain a guard-blind-spot field: {len(blind)})"
    )
    if blind:
        print("Tasks with an asymmetric-presence DENYLIST field the guard may NOT enforce:")
        for r in blind:
            fields = [d["field"] for d in r["deny_diffs"] if not d["guard_enforced"]]
            print(f"  - [{r['tree']}] {r['task']}: {fields}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--trees",
        nargs="+",
        default=["ppo", "appo"],
        help="Hydra config trees under conf/ to audit (default: ppo appo).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for tree in args.trees:
        rows.extend(audit_tree(tree))

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        _print_human(rows)


if __name__ == "__main__":
    main()
