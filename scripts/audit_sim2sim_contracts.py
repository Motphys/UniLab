"""Audit cross-backend sim2sim contract divergences across task owner YAMLs.

For every task with >=2 backend YAMLs, hydra-composes each backend's effective config
and compares the DENYLIST / WARNING_LIST fields from ``unilab.utils.sim2sim``.
Off-policy owners now live in separate per-algorithm trees (``sac``, ``td3``,
``flashsac``), so SAC, TD3, and FlashSAC are never compared with one another.

Read-only.

    uv run scripts/audit_sim2sim_contracts.py
    uv run scripts/audit_sim2sim_contracts.py --trees ppo appo sac td3 flashsac
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

from unilab.utils.sim2sim import DENYLIST, ENV_STRUCTURAL_DENYLIST, WARNING_LIST, _normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_ROOT = REPO_ROOT / "conf"
ABSENT = "<absent>"

# Audited backend pairs. mujoco<->motrix is the historical primary contract;
# mujoco<->isaacgym covers the subprocess backend owners.
CONTRACT_PAIRS: tuple[tuple[str, str], ...] = (
    ("mujoco", "motrix"),
    ("mujoco", "isaacgym"),
)


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
    overrides: list[str] = [f"task={task_variant}"]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        return compose("config", overrides=overrides)


def _discover(tree: str) -> dict[str, list[str]]:
    base = CONF_ROOT / tree / "task"
    out: dict[str, list[str]] = {}
    if not base.is_dir():
        return out
    for owner_file in sorted(base.rglob("*.yaml")):
        if owner_file.stem == "base":
            continue
        task_variant = owner_file.parent.relative_to(base).as_posix()
        if task_variant == ".":
            continue
        out.setdefault(task_variant, []).append(owner_file.stem)
    return {task: sorted(backends) for task, backends in sorted(out.items())}


def _diff_field(path: str, left: Any, right: Any) -> dict[str, Any] | None:
    left_absent, right_absent = left is ABSENT, right is ABSENT
    if left_absent and right_absent:
        return None
    if left_absent != right_absent:
        kind = "asymmetric-presence"
    elif _values_equal(left, right):
        return None
    else:
        kind = "value-diff"
    guard_enforced = kind == "value-diff" or (
        kind == "asymmetric-presence" and path in ENV_STRUCTURAL_DENYLIST
    )
    return {
        "field": path,
        "kind": kind,
        "left": _fmt(left),
        "right": _fmt(right),
        "guard_enforced": guard_enforced,
    }


def _audit_pair(
    left_name: str,
    right_name: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    deny = [d for p in DENYLIST if (d := _diff_field(p, left.get(p, ABSENT), right.get(p, ABSENT)))]
    warn = [
        d for p in WARNING_LIST if (d := _diff_field(p, left.get(p, ABSENT), right.get(p, ABSENT)))
    ]
    return {
        "pair": f"{left_name}<->{right_name}",
        "left_name": left_name,
        "right_name": right_name,
        "verdict": "TRANSFERABLE" if not deny else "BLOCKED",
        "deny_diffs": deny,
        "warn_diffs": warn,
    }


def audit_tree(tree: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    discovered = _discover(tree)
    if not discovered:
        raise ValueError(f"No task owner configs discovered under conf/{tree}/task")
    for task, backends in discovered.items():
        values: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for backend in backends:
            try:
                cfg = _compose(tree, f"{task}/{backend}")
                values[backend] = {p: _select(cfg, p) for p in (DENYLIST + WARNING_LIST)}
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                errors[backend] = f"{type(exc).__name__}: {exc}"

        pairs = [
            _audit_pair(left_name, right_name, values[left_name], values[right_name])
            for left_name, right_name in CONTRACT_PAIRS
            if left_name in values and right_name in values
        ]
        if not pairs:
            verdict = "N/A (no audited backend pair)"
        elif any(pair["verdict"] == "BLOCKED" for pair in pairs):
            verdict = "BLOCKED"
        else:
            verdict = "TRANSFERABLE"

        rows.append(
            {
                "tree": tree,
                "task": task,
                "backends": backends,
                "verdict": verdict,
                "pairs": pairs,
                "errors": errors,
            }
        )
    return rows


def _print_human(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(f"\n### [{row['tree']}] {row['task']}  backends={row['backends']}")
        if row["errors"]:
            print(f"  COMPOSE ERRORS: {row['errors']}")
        if not row["pairs"]:
            print(f"  VERDICT: {row['verdict']}")
            continue
        for pair in row["pairs"]:
            print(f"  VERDICT ({pair['pair']}): {pair['verdict']}")
            for diff in pair["deny_diffs"]:
                flag = (
                    ""
                    if diff["guard_enforced"]
                    else "  [guard-blind-spot: re-check dataclass default]"
                )
                print(
                    f"    DENY  {diff['field']} [{diff['kind']}]: "
                    f"{pair['left_name']}={diff['left']}  {pair['right_name']}={diff['right']}{flag}"
                )
            for diff in pair["warn_diffs"]:
                print(
                    f"    warn  {diff['field']} [{diff['kind']}]: "
                    f"{pair['left_name']}={diff['left']}  {pair['right_name']}={diff['right']}"
                )

    transferable = [r for r in rows if r["verdict"] == "TRANSFERABLE"]
    blocked = [r for r in rows if r["verdict"] == "BLOCKED"]
    blind = [
        r
        for r in blocked
        if any(not d["guard_enforced"] for pair in r["pairs"] for d in pair["deny_diffs"])
    ]
    print("\n" + "=" * 80)
    print(
        f"TRANSFERABLE: {len(transferable)}   BLOCKED: {len(blocked)}   "
        f"(of which contain a guard-blind-spot field: {len(blind)})"
    )
    if blind:
        print("Tasks with an asymmetric-presence DENYLIST field the guard may NOT enforce:")
        for r in blind:
            fields = [
                d["field"]
                for pair in r["pairs"]
                for d in pair["deny_diffs"]
                if not d["guard_enforced"]
            ]
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
