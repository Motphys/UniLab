"""Static audit boundary for the state-update-only graph candidate."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "unilab"

# Only the extracted fixed-shape update body is audited. Orchestration, timing,
# semantic logs, host validation, counters, state replacement, reset execution,
# and recorder/metric/curriculum publication intentionally stay outside this
# narrower #2338 boundary.
AUDITED_FUNCTIONS: dict[Path, frozenset[str]] = {
    SOURCE_ROOT / "envs/manager_based_rl_env.py": frozenset(
        {
            "_advance_episode_steps",
            "_compute_termination_and_reset_flags",
            "_compute_rewards",
            "_compute_metrics",
            "_compute_commands",
            "_post_compute_commands",
            "_compute_observations",
            "_publish_step_outputs",
        }
    ),
}

FORBIDDEN_DIRECT_CALLS = frozenset({"nonzero", "masked_select", "item", "tolist", "numpy"})
FORBIDDEN_ATTRIBUTE_CALLS = frozenset({"cpu", "tolist", "numpy", "item"})
FORBIDDEN_ALLOCATION_CALLS = frozenset({"empty", "zeros", "ones", "cat", "stack"})

# Eager execution must remain eager. These calls run only while graph mode is
# disabled; a future graph path must replace them with persistent out= writes.
OUTSIDE_UPDATE_GRAPH_VIOLATIONS: frozenset[tuple[str, int, str]] = frozenset(
    {
        (
            "src/unilab/envs/manager_based_rl_env.py",
            1683,
            "eager_terminated.clone()",
        ),
        (
            "src/unilab/envs/manager_based_rl_env.py",
            1685,
            "eager_truncated.clone()",
        ),
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    function: str
    kind: str
    expression: str


def _qualified_name(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    parts: list[str] = []
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(current.name)
        current = parents.get(id(current))
    return ".".join(reversed(parts))


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return ""


def _source(call: ast.Call) -> str:
    try:
        return ast.unparse(call)
    except Exception:
        return "<unparseable>"


def find_violations() -> list[Violation]:
    violations: list[Violation] = []
    for path, wanted in AUDITED_FUNCTIONS.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent

        for node in ast.walk(tree):
            qualified = _qualified_name(node, parents)
            if qualified.rsplit(".", 1)[-1] not in wanted:
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                name = _call_name(child)
                expression = _source(child)
                has_output_carrier = any(keyword.arg == "out" for keyword in child.keywords)
                if name in FORBIDDEN_DIRECT_CALLS:
                    kind = "dynamic-or-host-call"
                elif name in FORBIDDEN_ATTRIBUTE_CALLS:
                    kind = "host-transfer"
                elif name in FORBIDDEN_ALLOCATION_CALLS and not has_output_carrier:
                    kind = "capture-time-allocation"
                elif name == "clone":
                    kind = "temporary-allocation"
                else:
                    continue
                key = (str(path.relative_to(ROOT)), child.lineno, expression)
                if key in OUTSIDE_UPDATE_GRAPH_VIOLATIONS:
                    continue
                violations.append(
                    Violation(
                        path.relative_to(ROOT),
                        child.lineno,
                        qualified,
                        kind,
                        expression,
                    )
                )
    return violations


def test_env_update_graph_static_audit_is_zero() -> None:
    violations = find_violations()
    actual = {(str(item.path), item.line, item.kind, item.expression) for item in violations}
    expected: set[tuple[str, int, str, str]] = set()
    assert actual == expected, (
        "State-update graph audit changed:\n"
        f"unexpected={sorted(actual - expected)}\n"
        f"missing={sorted(expected - actual)}"
    )


def test_update_audit_scope_remains_explicit_and_narrow() -> None:
    assert AUDITED_FUNCTIONS.keys() == {
        SOURCE_ROOT / "envs/manager_based_rl_env.py",
    }
    assert set(next(iter(AUDITED_FUNCTIONS.values()))) == {
        "_advance_episode_steps",
        "_compute_termination_and_reset_flags",
        "_compute_rewards",
        "_compute_metrics",
        "_compute_commands",
        "_post_compute_commands",
        "_compute_observations",
        "_publish_step_outputs",
    }


def test_outside_graph_classifications_are_explicit() -> None:
    assert OUTSIDE_UPDATE_GRAPH_VIOLATIONS
