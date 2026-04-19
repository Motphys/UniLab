from __future__ import annotations

import re
from pathlib import Path

from unilab.docs.support_matrix import render_generated_block, replace_generated_block

VALID_HYDRA_KEYS = {
    "task",
    "algo",
    "training",
    "reward",
    "env",
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
    ".github/ISSUE_TEMPLATE/**/*.yml",
    ".github/ISSUE_TEMPLATE/**/*.yaml",
]

SKIP_PATTERNS = [
    r"(^|/)\.git(/|$)",
    r"(^|/)\.venv(/|$)",
    r"(^|/)__pycache__(/|$)",
    r"(^|/)\.pytest_cache(/|$)",
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
    script_pattern = r"(?<![\w/.-])(scripts/[A-Za-z0-9_]+\.py)\b"
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


def check_hydra_keys(content: str, doc_path: Path, root: Path) -> list[str]:
    del root
    errors: list[str] = []
    hydra_pattern = r"(?:^|\s)([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)="
    for line in content.splitlines():
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
    if doc_path != root / "docs" / "users" / "zh_CN" / "02-simulation-backends.md":
        return errors

    expected = replace_generated_block(content, render_generated_block(root))
    if expected != content:
        errors.append(
            f"{doc_path}: Generated support matrix is stale; run "
            "`uv run python scripts/generate_support_matrix.py --write`"
        )
    return errors


def check_document(doc_path: Path, root: Path) -> list[str]:
    content = doc_path.read_text(encoding="utf-8")
    errors: list[str] = []
    errors.extend(check_script_references(content, doc_path, root))
    errors.extend(check_file_paths(content, doc_path, root))
    errors.extend(check_markdown_links(content, doc_path, root))
    errors.extend(check_hydra_keys(content, doc_path, root))
    errors.extend(check_argparse_vs_hydra(content, doc_path, root))
    errors.extend(check_training_entrypoint_semantics(content, doc_path, root))
    errors.extend(check_generated_support_matrix(content, doc_path, root))
    return errors


def collect_doc_errors(root: Path | None = None) -> list[str]:
    resolved_root = root or repo_root()
    errors: list[str] = []
    for doc_path in find_docs(resolved_root):
        errors.extend(check_document(doc_path, resolved_root))
    return errors
