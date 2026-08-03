"""AST-driven runtime backend and environment contract isolation audit."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_BACKEND_PREFIX = "unilab.base.backend"
_RUNTIME_FILENAMES = frozenset({"backend.py", "batch.py"})
_SHARED_RUNTIME_MODULES = frozenset(
    {
        "base",
        "batch",
        "motrix_camera",
        "mutation",
        "playback_common",
    }
)
_RAW_BACKEND_MEMBERS = frozenset({"model"})
_DIAGNOSTIC_MEMBERS = frozenset({"__class__"})


@dataclass(frozen=True, order=True)
class BackendIsolationViolation:
    """One source-located backend isolation policy violation."""

    path: str
    line: int
    column: int
    code: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.code}: {self.message}"


@dataclass(frozen=True)
class BackendIsolationReport:
    """Complete, deterministic result of one repository isolation audit."""

    backend_packages: tuple[str, ...]
    runtime_modules: tuple[str, ...]
    contract_files: tuple[str, ...]
    violations: tuple[BackendIsolationViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def require_ok(self) -> None:
        if not self.ok:
            raise BackendIsolationAuditError(self)


class BackendIsolationAuditError(RuntimeError):
    """Raised when a backend isolation report is required to pass."""

    def __init__(self, report: BackendIsolationReport):
        self.report = report
        details = "\n".join(violation.format() for violation in report.violations)
        super().__init__(f"backend isolation audit failed:\n{details}")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _violation(
    *,
    root: Path,
    path: Path,
    code: str,
    message: str,
    node: ast.AST | None = None,
) -> BackendIsolationViolation:
    return BackendIsolationViolation(
        path=_relative_path(path, root),
        line=int(getattr(node, "lineno", 1)),
        column=int(getattr(node, "col_offset", 0)) + 1,
        code=code,
        message=message,
    )


def _parse_source(
    path: Path,
    *,
    root: Path,
    violations: list[BackendIsolationViolation],
) -> ast.Module | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        violations.append(
            _violation(
                root=root,
                path=path,
                code="source-read-error",
                message=f"cannot read source: {type(exc).__name__}: {exc}",
            )
        )
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        violations.append(
            BackendIsolationViolation(
                path=_relative_path(path, root),
                line=int(exc.lineno or 1),
                column=int(exc.offset or 1),
                code="source-syntax-error",
                message=exc.msg,
            )
        )
        return None


def _dotted_expression(node: ast.AST) -> str | None:
    parts: list[str] = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return None
    parts.append(cursor.id)
    return ".".join(reversed(parts))


def _resolve_relative_import(module_name: str, level: int, imported: str | None) -> str | None:
    package_parts = module_name.split(".")[:-1]
    ascend = level - 1
    if ascend >= len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - ascend]
    if imported:
        base_parts.extend(imported.split("."))
    return ".".join(base_parts)


def _import_targets(node: ast.Import | ast.ImportFrom, module_name: str) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    if node.level:
        base = _resolve_relative_import(module_name, node.level, node.module)
        if base is None:
            return ()
    else:
        base = node.module or ""

    targets = [base] if base else []
    for alias in node.names:
        if alias.name == "*":
            continue
        targets.append(f"{base}.{alias.name}" if base else alias.name)
    return tuple(targets)


def _import_aliases(node: ast.Import | ast.ImportFrom, module_name: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if isinstance(node, ast.Import):
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            aliases[local_name] = alias.name if alias.asname else local_name
        return aliases

    if node.level:
        base = _resolve_relative_import(module_name, node.level, node.module)
    else:
        base = node.module or ""
    if base is None:
        return aliases
    for alias in node.names:
        if alias.name == "*":
            continue
        local_name = alias.asname or alias.name
        aliases[local_name] = f"{base}.{alias.name}" if base else alias.name
    return aliases


def _expand_alias(dotted: str, aliases: dict[str, str]) -> str:
    first, separator, remainder = dotted.partition(".")
    resolved = aliases.get(first, first)
    return f"{resolved}.{remainder}" if separator else resolved


def _backend_owner(target: str, packages: frozenset[str]) -> str | None:
    prefix = f"{_BACKEND_PREFIX}."
    if not target.startswith(prefix):
        return None
    owner = target[len(prefix) :].split(".", 1)[0]
    return owner if owner in packages else None


def _audit_runtime_module(
    *,
    root: Path,
    path: Path,
    module_name: str,
    owner: str,
    packages: frozenset[str],
    tree: ast.Module,
) -> list[BackendIsolationViolation]:
    violations: list[BackendIsolationViolation] = []
    aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.level:
            resolved = _resolve_relative_import(module_name, node.level, node.module)
            if resolved is None:
                violations.append(
                    _violation(
                        root=root,
                        path=path,
                        node=node,
                        code="invalid-relative-import",
                        message="relative import escapes the unilab package",
                    )
                )
                continue
        aliases.update(_import_aliases(node, module_name))
        for target in _import_targets(node, module_name):
            target_owner = _backend_owner(target, packages)
            if target_owner is not None and target_owner != owner:
                violations.append(
                    _violation(
                        root=root,
                        path=path,
                        node=node,
                        code="sibling-runtime-import",
                        message=f"{owner} runtime imports sibling backend {target_owner}: {target}",
                    )
                )
                continue
            prefix = f"{_BACKEND_PREFIX}."
            if target.startswith(prefix) and target_owner is None:
                shared_name = target[len(prefix) :].split(".", 1)[0]
                if shared_name not in _SHARED_RUNTIME_MODULES:
                    violations.append(
                        _violation(
                            root=root,
                            path=path,
                            node=node,
                            code="unapproved-runtime-dependency",
                            message=f"runtime dependency is not in the shared allowlist: {target}",
                        )
                    )

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                dotted = _dotted_expression(base)
                if dotted is None:
                    continue
                resolved = _expand_alias(dotted, aliases)
                base_owner = _backend_owner(resolved, packages)
                if base_owner is not None and base_owner != owner:
                    violations.append(
                        _violation(
                            root=root,
                            path=path,
                            node=base,
                            code="sibling-backend-inheritance",
                            message=f"{owner} runtime inherits from sibling backend {base_owner}: {resolved}",
                        )
                    )
        if isinstance(node, ast.Attribute):
            dotted = _dotted_expression(node)
            if dotted is None:
                continue
            resolved = _expand_alias(dotted, aliases)
            target_owner = _backend_owner(resolved, packages)
            if target_owner is None or target_owner == owner:
                continue
            suffix = resolved.split(f"{_BACKEND_PREFIX}.{target_owner}.", 1)[-1]
            if any(part.startswith("_") for part in suffix.split(".")):
                violations.append(
                    _violation(
                        root=root,
                        path=path,
                        node=node,
                        code="sibling-private-access",
                        message=f"{owner} runtime accesses sibling private implementation: {resolved}",
                    )
                )

    return violations


def _sim_backend_public_members(
    *,
    root: Path,
    path: Path,
    tree: ast.Module | None,
    violations: list[BackendIsolationViolation],
) -> frozenset[str]:
    if tree is None:
        return frozenset()
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SimBackend"
    ]
    if len(classes) != 1:
        violations.append(
            _violation(
                root=root,
                path=path,
                code="invalid-backend-contract",
                message=f"expected exactly one top-level SimBackend class, found {len(classes)}",
            )
        )
        return frozenset()

    members: set[str] = set()
    for node in classes[0].body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members.add(node.target.id)
        elif isinstance(node, ast.Assign):
            members.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return frozenset(member for member in members if not member.startswith("_"))


class _BackendContractUseVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        root: Path,
        path: Path,
        public_members: frozenset[str],
    ) -> None:
        self._root = root
        self._path = path
        self._public_members = public_members
        self._alias_scopes: list[set[str]] = [{"_backend"}]
        self.violations: list[BackendIsolationViolation] = []

    def _is_backend_expression(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return any(node.id in scope for scope in reversed(self._alias_scopes))
        return isinstance(node, ast.Attribute) and node.attr == "_backend"

    def _add(self, node: ast.AST, code: str, message: str) -> None:
        self.violations.append(
            _violation(
                root=self._root,
                path=self._path,
                node=node,
                code=code,
                message=message,
            )
        )

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._alias_scopes.append(set(self._alias_scopes[-1]))
        for statement in node.body:
            self.visit(statement)
        self._alias_scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        if self._is_backend_expression(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._alias_scopes[-1].add(target.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            if self._is_backend_expression(node.value) and isinstance(node.target, ast.Name):
                self._alias_scopes[-1].add(node.target.id)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and node.args
            and self._is_backend_expression(node.args[0])
        ):
            self._add(
                node,
                "dynamic-backend-probe",
                f"{node.func.id} capability probing bypasses the SimBackend contract",
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._is_backend_expression(node.value):
            member = node.attr
            if member in _DIAGNOSTIC_MEMBERS:
                pass
            elif member.startswith("_"):
                self._add(
                    node,
                    "private-backend-member",
                    f"backend private member access is forbidden: {member}",
                )
            elif member in _RAW_BACKEND_MEMBERS:
                self._add(
                    node,
                    "raw-backend-object",
                    f"env/DR code cannot access raw backend object: {member}",
                )
            elif member not in self._public_members:
                self._add(
                    node,
                    "unknown-backend-member",
                    f"SimBackend does not declare public member: {member}",
                )
        self.generic_visit(node)


def _contract_source_paths(source_root: Path) -> tuple[Path, ...]:
    roots = (source_root / "envs", source_root / "dr")
    paths: list[Path] = []
    for scan_root in roots:
        if scan_root.is_dir():
            paths.extend(sorted(scan_root.rglob("*.py")))
    np_env = source_root / "base" / "np_env.py"
    if np_env.is_file():
        paths.append(np_env)
    return tuple(sorted(set(paths)))


def _missing_path_violation(
    root: Path, path: Path, code: str, message: str
) -> BackendIsolationViolation:
    return _violation(root=root, path=path, code=code, message=message)


def audit_backend_isolation(root: Path) -> BackendIsolationReport:
    """Audit backend runtime imports and env/DR use of ``SimBackend``.

    ``root`` is the repository root containing ``src/unilab``.  The audit is
    fail-closed for missing roots, malformed backend packages, unreadable or
    invalid Python, and an invalid ``SimBackend`` declaration.
    """

    root = root.resolve()
    source_root = root / "src" / "unilab"
    backend_root = source_root / "base" / "backend"
    base_path = backend_root / "base.py"
    required_paths = (
        source_root,
        backend_root,
        source_root / "envs",
        source_root / "dr",
        source_root / "base" / "np_env.py",
        base_path,
    )
    violations: list[BackendIsolationViolation] = []
    for path in required_paths:
        if not path.exists():
            violations.append(
                _missing_path_violation(
                    root,
                    path,
                    "missing-audit-root",
                    "required backend isolation audit root is missing",
                )
            )

    if not backend_root.is_dir():
        return BackendIsolationReport((), (), (), tuple(sorted(set(violations))))

    package_dirs = sorted(
        path
        for path in backend_root.iterdir()
        if path.is_dir() and any((path / filename).is_file() for filename in _RUNTIME_FILENAMES)
    )
    if not package_dirs:
        violations.append(
            _missing_path_violation(
                root,
                backend_root,
                "empty-runtime-root",
                "no runtime backend package containing backend.py was found",
            )
        )

    packages: list[str] = []
    runtime_paths: list[tuple[str, Path]] = []
    for package_dir in package_dirs:
        package_name = package_dir.name
        packages.append(package_name)
        if not (package_dir / "__init__.py").is_file():
            violations.append(
                _missing_path_violation(
                    root,
                    package_dir,
                    "invalid-backend-layout",
                    f"backend package {package_name!r} is missing __init__.py",
                )
            )
        if not (package_dir / "backend.py").is_file():
            violations.append(
                _missing_path_violation(
                    root,
                    package_dir,
                    "invalid-backend-layout",
                    f"backend package {package_name!r} is missing backend.py",
                )
            )
        for filename in sorted(_RUNTIME_FILENAMES):
            path = package_dir / filename
            if path.is_file():
                runtime_paths.append((package_name, path))

    frozen_packages = frozenset(packages)
    runtime_modules: list[str] = []
    for owner, path in runtime_paths:
        module_name = f"{_BACKEND_PREFIX}.{owner}.{path.stem}"
        runtime_modules.append(module_name)
        tree = _parse_source(path, root=root, violations=violations)
        if tree is not None:
            violations.extend(
                _audit_runtime_module(
                    root=root,
                    path=path,
                    module_name=module_name,
                    owner=owner,
                    packages=frozen_packages,
                    tree=tree,
                )
            )

    base_tree = (
        _parse_source(base_path, root=root, violations=violations) if base_path.is_file() else None
    )
    public_members = _sim_backend_public_members(
        root=root,
        path=base_path,
        tree=base_tree,
        violations=violations,
    )

    contract_paths = _contract_source_paths(source_root)
    for path in contract_paths:
        tree = _parse_source(path, root=root, violations=violations)
        if tree is None:
            continue
        visitor = _BackendContractUseVisitor(
            root=root,
            path=path,
            public_members=public_members,
        )
        visitor.visit(tree)
        violations.extend(visitor.violations)

    return BackendIsolationReport(
        backend_packages=tuple(packages),
        runtime_modules=tuple(runtime_modules),
        contract_files=tuple(_relative_path(path, root) for path in contract_paths),
        violations=tuple(sorted(set(violations))),
    )


def format_backend_isolation_report(report: BackendIsolationReport) -> Iterable[str]:
    """Yield a stable human-readable audit summary."""

    yield (
        f"backend isolation: packages={len(report.backend_packages)} "
        f"runtime_modules={len(report.runtime_modules)} contract_files={len(report.contract_files)}"
    )
    if report.ok:
        yield "PASS: runtime backends and env/DR callers respect the SimBackend boundary"
    else:
        yield from (f"FAIL: {violation.format()}" for violation in report.violations)
