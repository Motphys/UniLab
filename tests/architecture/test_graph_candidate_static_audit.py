"""Static audit boundary for the K=1 collector graph candidate hot path."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "unilab"

# Functions reachable from action consumption through post-reset observation
# and transition publication. `publish_validation_verdicts` is intentionally
# absent: it is the fail-closed completion boundary outside capture.
AUDITED_FUNCTIONS: dict[Path, frozenset[str]] = {
    SOURCE_ROOT / "base/torch_env.py": frozenset(
        {
            "step",
            "_validate_action_contract",
            "_record_action_finite",
            "_validate_control",
            "_validate_state",
            "_compute_truncated",
            "_reset_done_envs",
            "_validate_reset_observation_mask",
            "_scatter_reset_info_mask",
        }
    ),
    SOURCE_ROOT / "envs/manager_based_rl_env.py": frozenset(
        {
            "step",
            "apply_action",
            "update_state",
            "_update_state_in_read_phase",
            "_run_env_update_graph_body",
            "_reset_done_envs",
            "reset",
            "_publish_reward_components",
            "_publish_step_outputs",
        }
    ),
    SOURCE_ROOT / "managers/action_manager.py": frozenset(
        {"process_action", "apply_action", "reset"}
    ),
    # Reset-time semantic publication is completion/interval-owned and must
    # not be captured; its audit covers the compute path only.
    SOURCE_ROOT / "managers/reward_manager.py": frozenset({"compute"}),
    SOURCE_ROOT / "managers/termination_manager.py": frozenset({"compute"}),
    SOURCE_ROOT / "managers/command_manager.py": frozenset({"compute"}),
    SOURCE_ROOT / "managers/observation_manager.py": frozenset({"compute", "compute_group"}),
    SOURCE_ROOT / "tasks/motion_tracking/common/manager_terms.py": frozenset(
        {
            "MotionCommand.reset",
            "MotionCommand._resample_command",
            "MotionCommand._update_command",
            "MotionCommand._update_command_reset",
            "MotionCommand.post_compute",
            "MotionCommand.post_compute_reset",
        }
    ),
}

FORBIDDEN_DIRECT_CALLS = frozenset({"nonzero", "masked_select", "item", "tolist", "numpy"})
FORBIDDEN_ATTRIBUTE_CALLS = frozenset({"cpu", "tolist", "numpy", "item"})
FORBIDDEN_ALLOCATION_CALLS = frozenset({"empty", "zeros", "ones", "cat", "stack"})

# Static operations outside the graph-candidate proper are explicitly
# classified rather than silently erased from the audit:
# - non-autoreset mode is rejected by the public graph preflight;
# - public-carrier layout recovery is a fail-cold fallback, while a graph
#   runtime materializes the stable layout before capture;
# - eager invalid-row diagnostics are unreachable after deferred validation
#   returns, with publication owned by the completion boundary.
OUT_OF_CAPTURE_VIOLATIONS: frozenset[tuple[str, int, str]] = frozenset(
    {
        (
            "src/unilab/envs/manager_based_rl_env.py",
            1419,
            "self._manual_reset_pending.nonzero(as_tuple=False)",
        ),
        (
            "src/unilab/envs/manager_based_rl_env.py",
            1419,
            "self._manual_reset_pending.nonzero(as_tuple=False).flatten().detach().cpu()",
        ),
        (
            "src/unilab/envs/manager_based_rl_env.py",
            1419,
            "self._manual_reset_pending.nonzero(as_tuple=False).flatten().detach().cpu().tolist()",
        ),
        (
            "src/unilab/envs/manager_based_rl_env.py",
            2278,
            "eager_terminated.clone()",
        ),
        (
            "src/unilab/envs/manager_based_rl_env.py",
            2280,
            "eager_truncated.clone()",
        ),
        (
            "src/unilab/managers/observation_manager.py",
            635,
            "cast(torch.Tensor, obs).clone()",
        ),
        (
            "src/unilab/managers/observation_manager.py",
            705,
            "torch.cat(\n"
            "                    tensors,\n"
            "                    dim=self._group_obs_concatenate_dim[group_name],\n"
            "                    out=result,\n"
            "                )",
        ),
        (
            "src/unilab/managers/reward_manager.py",
            266,
            "torch.nonzero(~finite[:, term_idx])",
        ),
        (
            "src/unilab/managers/reward_manager.py",
            269,
            "invalid_rows.tolist()",
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


def _function_name(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    return ""


def _qualified_name(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    parts: list[str] = []
    current: ast.AST | None = node
    while current is not None:
        name = _function_name(current)
        if name:
            parts.append(name)
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
            leaf = qualified.rsplit(".", 1)[-1]
            # Method selectors include the class name; accept exact names too
            # so subclasses/local helpers remain auditable when explicitly named.
            if qualified not in wanted and leaf not in wanted:
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                name = _call_name(child)
                expression = _source(child)
                # Calls that write into an explicitly supplied output carrier
                # do not allocate a result even when the operation name also
                # has an allocating overload.
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
                if key in OUT_OF_CAPTURE_VIOLATIONS:
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


def test_graph_candidate_static_audit_is_zero() -> None:
    violations = find_violations()
    actual = {(str(item.path), item.line, item.kind, item.expression) for item in violations}
    expected: set[tuple[str, int, str, str]] = set()
    assert actual == expected, (
        "Graph candidate static audit changed:\n"
        f"unexpected={sorted(actual - expected)}\n"
        f"missing={sorted(expected - actual)}"
    )


def test_static_audit_scope_remains_explicit_and_nonempty() -> None:
    assert AUDITED_FUNCTIONS
    for path, functions in AUDITED_FUNCTIONS.items():
        assert path.is_file(), path
        assert functions


def test_static_audit_out_of_capture_classifications_are_explicit() -> None:
    assert OUT_OF_CAPTURE_VIOLATIONS
