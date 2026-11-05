"""Guard the issue #2583 host-rotation sync boundary on training hot paths.

`EntityData._host_root_link_lin_vel_b` / `_host_root_link_ang_vel_b` rotate
world-frame velocities on the host (NumPy) side. Calling them from the
training hot path (`src/unilab/envs/mdp/`, `src/unilab/tasks/`,
`src/unilab/managers/`) forces a D2H copy plus a CPU synchronization per step;
the device-resident `quat_apply_inverse` tensor primitives are the hot-path
contract. The same risk appears when a `.cpu()`/`.numpy()` transfer is
immediately followed by a quaternion helper call.

Cold reset/startup paths may be exempted per file: register the
(path, line) pair in `_HOST_ROTATION_SYNC_EXEMPTIONS` and mark the source
line with `# unilab-arch: host-rotation-sync allow` so the exemption stays
auditable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "unilab"
HOT_PATH_DIRS = (
    SOURCE_ROOT / "envs" / "mdp",
    SOURCE_ROOT / "tasks",
    SOURCE_ROOT / "managers",
)

HOST_ADAPTER_TOKENS = frozenset(
    {
        "_host_root_link_lin_vel_b",
        "_host_root_link_ang_vel_b",
    }
)

_HOST_TRANSFER_METHODS = frozenset({"cpu", "numpy"})
_QUAT_CALL_PREFIXES = ("np_quat", "quat_", "_quat_")
_HOST_SYNC_WINDOW = 10
_EXEMPTION_MARKER = "unilab-arch: host-rotation-sync allow"

# Per-file allowlist for audited cold (reset/startup) paths only. Keys are
# repo-relative paths; values are the line numbers of the quaternion helper
# call that follows a host transfer. Every entry must carry the exemption
# marker comment on the call line or the line above it, and must match a
# detected violation (stale entries fail the consistency test below).
_HOST_ROTATION_SYNC_EXEMPTIONS: dict[str, frozenset[int]] = {}


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    detail: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.detail}"


def _iter_hot_path_files() -> list[Path]:
    return sorted(
        path
        for directory in HOT_PATH_DIRS
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _is_quat_call(func: ast.expr) -> bool:
    name = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    return name is not None and name.startswith(_QUAT_CALL_PREFIXES)


def find_host_adapter_violations() -> list[Violation]:
    violations: list[Violation] = []
    for path in _iter_hot_path_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in HOST_ADAPTER_TOKENS:
                violations.append(
                    Violation(
                        relative,
                        node.lineno,
                        f"host-side root velocity adapter `{node.attr}` is forbidden on the "
                        "training hot path (D2H + CPU sync); use the device-resident "
                        "tensor rotation primitives instead",
                    )
                )
    return violations


def find_host_rotation_sync_violations() -> list[Violation]:
    violations: list[Violation] = []
    for path in _iter_hot_path_files():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        relative = path.relative_to(ROOT)
        transfer_lines: list[int] = []
        quat_calls: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _HOST_TRANSFER_METHODS:
                transfer_lines.append(node.lineno)
            if _is_quat_call(func):
                name = func.id if isinstance(func, ast.Name) else func.attr  # type: ignore[union-attr]
                quat_calls.append((node.lineno, name))
        if not transfer_lines or not quat_calls:
            continue
        lines = source.splitlines()
        exemptions = _HOST_ROTATION_SYNC_EXEMPTIONS.get(str(relative), frozenset())
        for line, name in quat_calls:
            window = [h for h in transfer_lines if h <= line <= h + _HOST_SYNC_WINDOW]
            if not window:
                continue
            if line in exemptions:
                continue
            violations.append(
                Violation(
                    relative,
                    line,
                    f"quaternion helper `{name}` runs within {_HOST_SYNC_WINDOW} lines of a "
                    f".cpu()/.numpy() transfer (line {window[-1]}); host rotation after a D2H "
                    "transfer synchronizes the training hot path — keep the rotation on device, "
                    "or register an audited cold-path exemption in "
                    "_HOST_ROTATION_SYNC_EXEMPTIONS with the "
                    f"`{_EXEMPTION_MARKER}` marker",
                )
            )
        # Exempted lines must carry the marker comment for auditability.
        for line in exemptions:
            context = lines[max(0, line - 2) : line]
            if not any(_EXEMPTION_MARKER in text for text in context):
                violations.append(
                    Violation(
                        relative,
                        line,
                        f"registered exemption lacks the `{_EXEMPTION_MARKER}` marker "
                        "comment on the call line or the line above it",
                    )
                )
    return violations


def test_hot_paths_do_not_use_host_root_velocity_adapters() -> None:
    violations = find_host_adapter_violations()
    assert violations == [], "\n".join(violation.render() for violation in violations)


def test_hot_paths_do_not_sync_for_host_rotation() -> None:
    violations = find_host_rotation_sync_violations()
    assert violations == [], "\n".join(violation.render() for violation in violations)


def test_host_rotation_sync_exemptions_stay_current() -> None:
    detected: set[tuple[str, int]] = set()
    for path in _iter_hot_path_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = str(path.relative_to(ROOT))
        transfer_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _HOST_TRANSFER_METHODS
        ]
        if not transfer_lines:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_quat_call(node.func):
                if any(h <= node.lineno <= h + _HOST_SYNC_WINDOW for h in transfer_lines):
                    detected.add((relative, node.lineno))
    registered = {
        (path, line) for path, lines in _HOST_ROTATION_SYNC_EXEMPTIONS.items() for line in lines
    }
    stale = registered - detected
    assert not stale, (
        "stale host-rotation-sync exemptions (no matching violation remains): "
        + ", ".join(f"{path}:{line}" for path, line in sorted(stale))
    )
