from __future__ import annotations

import re
from pathlib import Path

from unilab.utils.support_matrix import render_generated_block, replace_generated_block

VALID_HYDRA_KEYS = {
    "task",
    "algo",
    "training",
    "reward",
    "env",
    "teacher",
    "interactive",
    "viser",
    "num_envs",
    "max_iterations",
    "num_steps_per_env",
    "learning_rate",
    "entropy_coef",
    "desired_kl",
    "load_run",
    "play_only",
    "no_play",
    "logger",
    "sim_backend",
    "play_env_num",
    "play_steps",
    "cam_distance",
    "cam_elevation",
    "cam_azimuth",
    "num_timesteps",
    "num_gpus",
    "device",
    "wandb_project",
    "wandb_entity",
    "wandb_group",
    "wandb_name",
    "wandb_tags",
    "wandb_notes",
    "wandb_mode",
}

DOC_PATTERNS = [
    "*.md",
    "docs/**/*.md",
    "src/**/*.md",
    "scripts/**/*.md",
    ".github/ISSUE_TEMPLATE/**/*.yml",
    ".github/ISSUE_TEMPLATE/**/*.yaml",
]

SKIP_PATTERNS = [
    r"(^|/)\.git(/|$)",
    r"(^|/)\.venv(/|$)",
    r"(^|/)__pycache__(/|$)",
    r"(^|/)\.pytest_cache(/|$)",
]

USER_DOC_MIGRATION_PHRASES = [
    "已拆成",
    "本页改成",
    "下沉到参考页",
]

USER_DOC_MAX_LINES = 120

SPHINX_REMOVED_PATH_PATTERNS = [
    (
        r"\.\./\.\./README\.md",
        "Use a MyST `{doc}` link to `/index` instead of `../../README.md`",
    ),
    (
        r"\bdocs/users/(?:en|zh_CN)/",
        "Use `docs/sphinx/source/<lang>/user_guide/` or a MyST `{doc}` link",
    ),
    (
        r"\bdocs/developers/(?:en|zh_CN|adr)/",
        "Use `docs/sphinx/source/<lang>/developer_guide/`, `/adr/`, or a MyST `{doc}` link",
    ),
    (
        r"(?<![\w/-])(?:\.\./)*users/(?:en|zh_CN)/",
        "Use the current `user_guide` Sphinx path or a MyST `{doc}` link",
    ),
    (
        r"(?<![\w/-])(?:\.\./)*developers/(?:en|zh_CN|adr)/",
        "Use the current `developer_guide` / `adr` Sphinx path or a MyST `{doc}` link",
    ),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def should_skip(path: Path) -> bool:
    path_str = str(path)
    return any(re.search(pattern, path_str) for pattern in SKIP_PATTERNS)


def find_docs(root: Path) -> list[Path]:
    docs: list[Path] = []
    for pattern in DOC_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and not should_skip(path):
                docs.append(path)
    issue_template_dir = root / ".github" / "ISSUE_TEMPLATE"
    if issue_template_dir.exists():
        for pattern in ("*.yml", "*.yaml"):
            for path in issue_template_dir.rglob(pattern):
                if path.is_file() and not should_skip(path):
                    docs.append(path)
    return sorted(set(docs))


def check_script_references(content: str, doc_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    script_pattern = r"(?<![\w/.-])(scripts/(?:[A-Za-z0-9_/-]+/)?[A-Za-z0-9_]+\.py)\b"
    for match in re.finditer(script_pattern, content):
        script_path = match.group(1)
        if not (root / script_path).exists():
            errors.append(f"{doc_path}: Script not found: {script_path}")
    return errors


def check_file_paths(content: str, doc_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    path_patterns = [
        r"`(src/[^`]+)`",
        r"`(conf/[^`]+)`",
        r"`(tests/[^`]+)`",
        r"\(src/[^\)]+\)",
        r"\(conf/[^\)]+\)",
        r"\(tests/[^\)]+\)",
    ]

    for pattern in path_patterns:
        for match in re.finditer(pattern, content):
            rel_path = match.group(1)
            if any(token in rel_path for token in ("*", "<", "{", "}", "...")):
                continue
            if "::" in rel_path:
                rel_path = rel_path.split("::", 1)[0]
            if not (root / rel_path).exists():
                errors.append(f"{doc_path}: Path not found: {rel_path}")

    return errors


def check_markdown_links(content: str, doc_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    sphinx_root = root / "docs" / "sphinx"
    try:
        doc_path.relative_to(sphinx_root)
        return errors
    except ValueError:
        pass

    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

    for match in re.finditer(link_pattern, content):
        link_text = match.group(1)
        link_target = match.group(2)
        if link_target.startswith(("http://", "https://", "#", "mailto:")):
            continue

        if link_target.startswith("/"):
            full_path = root / link_target.lstrip("/")
        else:
            full_path = doc_path.parent / link_target

        full_path = Path(str(full_path).split("#")[0])
        if not full_path.exists():
            errors.append(f"{doc_path}: Link not found: {link_target} (text: '{link_text}')")

    return errors


def check_raw_github_repo_urls(content: str, doc_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    url_pattern = r"https://github\.com/unilabsim/UniLab/blob/main/([^\s\"'<>]+)"

    for match in re.finditer(url_pattern, content):
        rel_path = match.group(1).rstrip(").,")
        if not (root / rel_path).exists():
            errors.append(f"{doc_path}: GitHub URL target not found: {rel_path}")

    return errors


def check_markdown_fences(content: str, doc_path: Path, root: Path) -> list[str]:
    del root
    errors: list[str] = []
    fence_start: int | None = None

    for line_no, line in enumerate(content.splitlines(), start=1):
        if not re.match(r"^\s*```", line):
            continue
        if fence_start is None:
            fence_start = line_no
        else:
            fence_start = None

    if fence_start is not None:
        errors.append(f"{doc_path}: Unclosed fenced code block starting at line {fence_start}")

    return errors


def check_canonical_commands(content: str, doc_path: Path, root: Path) -> list[str]:
    del root
    errors: list[str] = []

    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if "uv run python scripts/" in stripped:
            errors.append(
                f"{doc_path}:{line_no}: Use `uv run scripts/...` instead of "
                "`uv run python scripts/...`"
            )
        if stripped.startswith(("unilab train", "unilab eval", "unilab demo")):
            errors.append(
                f"{doc_path}:{line_no}: Use `uv run train`, `uv run eval`, "
                "or `uv run demo` instead of the removed `unilab` subcommand interface"
            )
        if stripped == "source .venv/bin/activate":
            errors.append(
                f"{doc_path}:{line_no}: Use `uv run <command>` instead of activating .venv"
            )

    return errors


def check_hydra_keys(content: str, doc_path: Path, root: Path) -> list[str]:
    del root
    errors: list[str] = []
    hydra_pattern = r"(?:^|\s)([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)="
    command_fence_languages = {"", "bash", "console", "shell", "sh", "text"}
    in_fence = False
    fence_language = ""

    for line in content.splitlines():
        fence_match = re.match(r"^\s*```\s*([A-Za-z0-9_+-]*)", line)
        if fence_match:
            in_fence = not in_fence
            fence_language = fence_match.group(1).lower() if in_fence else ""
            continue
        if in_fence and fence_language not in command_fence_languages:
            continue

        stripped = line.strip()
        is_cli_line = (
            "scripts/train_" in stripped
            or "scripts/play_interactive.py" in stripped
            or "scripts/play_viser.py" in stripped
        )
        is_override_line = bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_.-]*=", stripped))
        if not (is_cli_line or is_override_line):
            continue

        for match in re.finditer(hydra_pattern, stripped):
            full_key = match.group(1)
            parts = full_key.split(".")
            if parts[0] not in VALID_HYDRA_KEYS and parts[0] not in {"python", "uv"}:
                if re.match(r"^[a-z_]+$", parts[0]):
                    errors.append(f"{doc_path}: Unknown config key: {full_key}")

    return errors


def check_argparse_vs_hydra(content: str, doc_path: Path, root: Path) -> list[str]:
    del root
    errors: list[str] = []
    old_flags = [
        (r"--task\s+", "Use Hydra style: task=<value>"),
        (r"--env_num\s+", "Use Hydra style: algo.num_envs=<value>"),
        (r"--play_only\b", "Use Hydra style: training.play_only=true"),
        (r"--no_play\b", "Use Hydra style: training.no_play=true"),
        (r"--load_run\s+", "Use Hydra style: algo.load_run=<value>"),
        (r"--cam_distance\s+", "Use Hydra style: training.cam_distance=<value>"),
        (r"--cam_elevation\s+", "Use Hydra style: training.cam_elevation=<value>"),
        (r"--cam_azimuth\s+", "Use Hydra style: training.cam_azimuth=<value>"),
    ]
    hydra_scripts = [
        "train_rsl_rl.py",
        "train_mlx_ppo.py",
        "train_appo.py",
        "train_offpolicy.py",
    ]

    for pattern, message in old_flags:
        for match in re.finditer(pattern, content):
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 50)
            context = content[start:end]
            if any(script in context for script in hydra_scripts):
                errors.append(
                    f"{doc_path}: Outdated argparse flag '{match.group(0).strip()}': {message}"
                )

    return errors


def check_training_entrypoint_semantics(content: str, doc_path: Path, root: Path) -> list[str]:
    del root
    errors: list[str] = []

    for _match in re.finditer(r"\btraining\.load_run=", content):
        errors.append(
            f"{doc_path}: Deprecated Hydra key 'training.load_run': use algo.load_run=<value>"
        )

    for match in re.finditer(r"logs/rsl_rl_train/", content):
        errors.append(
            f"{doc_path}: Stale log root '{match.group(0)}': describe logs via "
            "logs/<algo.algo_log_name>/<task>/ or the current default algo.algo_log_name"
        )

    task_pattern = r"\btask=([A-Za-z0-9_-]+)(?=$|[\s\"'])"
    for match in re.finditer(task_pattern, content):
        errors.append(
            f"{doc_path}: Task override '{match.group(0)}' is missing the backend segment; "
            "use task=<task>/<backend> or task=<algo>/<task>/<backend>"
        )

    return errors


def check_generated_support_matrix(content: str, doc_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    if (
        doc_path
        != root
        / "docs"
        / "sphinx"
        / "source"
        / "zh_CN"
        / "user_guide"
        / "E-reference"
        / "01-backend-support-matrix.md"
    ):
        return errors

    expected = replace_generated_block(content, render_generated_block(root))
    if expected != content:
        errors.append(
            f"{doc_path}: Generated support matrix is stale; run "
            "`uv run scripts/generate_support_matrix.py --write`"
        )
    return errors


def check_zh_cn_doc_shape(content: str, doc_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    zh_root = root / "docs" / "sphinx" / "source" / "zh_CN"
    try:
        doc_path.relative_to(zh_root)
    except ValueError:
        return errors
    if doc_path.suffix != ".md":
        return errors
    # Section / language landing pages don't carry the language-tag + navigation
    # contract — they have their own toctree-driven layout.
    if doc_path.name == "index.md":
        return errors

    lines = content.splitlines()
    if len(lines) < 3 or lines[2].strip() != "语言: 简体中文":
        errors.append(f"{doc_path}: zh_CN docs must declare `语言: 简体中文` after title")

    if "\n## Navigation\n" not in content:
        errors.append(f"{doc_path}: zh_CN docs must include a `## Navigation` section")

    if not re.search(r"- Index:\s*\[[^\]]+\]\([^)]+index\.md\)", content):
        errors.append(f"{doc_path}: zh_CN docs must link back to a docs index in Navigation")

    return errors


def check_adr_shape(content: str, doc_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    adr_dir = root / "docs" / "sphinx" / "source" / "adr"
    if doc_path.parent != adr_dir or doc_path.suffix != ".md":
        return errors
    if doc_path.name in ("ADR-0000-index.md", "README.md"):
        return errors

    required_tokens = [
        "- Status:",
        "- Date:",
        "- Owners:",
        "- Supersedes:",
        "- Superseded by:",
        "## Alternatives Considered",
        "## Evidence In Repo",
        "## Related Documents",
    ]
    for token in required_tokens:
        if token not in content:
            errors.append(f"{doc_path}: ADR must include `{token}`")

    return errors


def check_user_doc_architecture(content: str, doc_path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    user_root = root / "docs" / "sphinx" / "source" / "zh_CN" / "user_guide"
    try:
        doc_path.relative_to(user_root)
    except ValueError:
        return errors

    exempt_long_docs = {
        user_root / "E-reference" / "01-backend-support-matrix.md",
    }
    if doc_path not in exempt_long_docs and len(content.splitlines()) > USER_DOC_MAX_LINES:
        errors.append(
            f"{doc_path}: user docs should stay concise; split files before exceeding "
            f"{USER_DOC_MAX_LINES} lines"
        )

    for phrase in USER_DOC_MIGRATION_PHRASES:
        if phrase in content:
            errors.append(
                f"{doc_path}: user docs should not expose migration/restructure phrasing: `{phrase}`"
            )

    tasks_index = user_root / "D-tasks" / "01-task-index.md"
    if doc_path == tasks_index:
        for token in ("## 按机器人家族", "## 按任务类型"):
            if token not in content:
                errors.append(f"{doc_path}: task index must include `{token}`")

    backend_overview = user_root / "02-simulation-backends.md"
    if doc_path == backend_overview and "E-reference/01-backend-support-matrix.md" not in content:
        errors.append(
            f"{doc_path}: backend overview must route users to `E-reference/01-backend-support-matrix.md`"
        )

    return errors


def check_sphinx_source_migration_guards(content: str, doc_path: Path, root: Path) -> list[str]:
    warnings: list[str] = []
    sphinx_source = root / "docs" / "sphinx" / "source"
    try:
        relative_path = doc_path.relative_to(sphinx_source)
    except ValueError:
        return warnings
    if doc_path.suffix != ".md":
        return warnings

    for pattern, message in SPHINX_REMOVED_PATH_PATTERNS:
        for match in re.finditer(pattern, content):
            line_no = content.count("\n", 0, match.start()) + 1
            removed_path = match.group(0)
            warnings.append(
                f"{doc_path}:{line_no}: Removed Sphinx doc path `{removed_path}`: {message}"
            )

    if relative_path.parts and relative_path.parts[0] == "en":
        for match in re.finditer(r"(?m)^语言: 简体中文$", content):
            line_no = content.count("\n", 0, match.start()) + 1
            warnings.append(
                f"{doc_path}:{line_no}: English Sphinx pages must not declare `语言: 简体中文`"
            )
        for match in re.finditer(r"(?m)^## Navigation\s*$", content):
            line_no = content.count("\n", 0, match.start()) + 1
            warnings.append(
                f"{doc_path}:{line_no}: English Sphinx pages should use toctree/Furo "
                "navigation instead of a hand-written `## Navigation` section"
            )

    return warnings


def check_document_warnings(doc_path: Path, root: Path) -> list[str]:
    content = doc_path.read_text(encoding="utf-8")
    warnings: list[str] = []
    warnings.extend(check_sphinx_source_migration_guards(content, doc_path, root))
    return warnings


def check_document(doc_path: Path, root: Path) -> list[str]:
    content = doc_path.read_text(encoding="utf-8")
    errors: list[str] = []

    sphinx_root = root / "docs" / "sphinx"
    in_sphinx = False
    try:
        doc_path.relative_to(sphinx_root)
        in_sphinx = True
    except ValueError:
        pass

    if not in_sphinx:
        errors.extend(check_script_references(content, doc_path, root))
        errors.extend(check_file_paths(content, doc_path, root))
        errors.extend(check_markdown_links(content, doc_path, root))
        errors.extend(check_raw_github_repo_urls(content, doc_path, root))
        errors.extend(check_markdown_fences(content, doc_path, root))
        errors.extend(check_canonical_commands(content, doc_path, root))
        errors.extend(check_hydra_keys(content, doc_path, root))
        errors.extend(check_argparse_vs_hydra(content, doc_path, root))
        errors.extend(check_training_entrypoint_semantics(content, doc_path, root))

    errors.extend(check_generated_support_matrix(content, doc_path, root))
    errors.extend(check_zh_cn_doc_shape(content, doc_path, root))
    errors.extend(check_adr_shape(content, doc_path, root))
    errors.extend(check_user_doc_architecture(content, doc_path, root))
    return errors


def collect_doc_errors(root: Path | None = None) -> list[str]:
    resolved_root = root or repo_root()
    errors: list[str] = []
    for doc_path in find_docs(resolved_root):
        errors.extend(check_document(doc_path, resolved_root))
    return errors


def collect_doc_warnings(root: Path | None = None) -> list[str]:
    resolved_root = root or repo_root()
    warnings: list[str] = []
    for doc_path in find_docs(resolved_root):
        warnings.extend(check_document_warnings(doc_path, resolved_root))
    return warnings
