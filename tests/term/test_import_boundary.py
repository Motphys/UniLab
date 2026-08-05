import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "unilab" / "term"
BANNED_PREFIXES = (
    "hydra",
    "omegaconf",
    "unilab.algos",
    "unilab.base",
    "unilab.envs",
    "unilab.ipc",
    "unilab.training",
)


def test_term_package_does_not_import_owner_or_runtime_layers() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = [node.module]
            else:
                continue
            for module in imported:
                if module.startswith(BANNED_PREFIXES):
                    violations.append(f"{path.name}:{node.lineno}: {module}")

    assert violations == []
