"""Shell completion helper for UniLab training entrypoints."""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from unilab import cli

TRAINING_ENTRYPOINTS = {
    "train": "unilab.cli:train_main",
    "eval": "unilab.cli:eval_main",
}
RUN_PATH_ROOTS = ("benchmark", "scripts")
RUN_PATH_SUFFIXES = (".py", ".sh")
RUN_PATH_IGNORED_PARTS = ("__pycache__", "outputs")
SCRIPT_ASSIGNMENT_PATTERN = re.compile(r'^([A-Za-z0-9_.-]+)\s*=\s*"([^"]+)"\s*(?:#.*)?$')
COMPLETION_BLOCK_START = "# >>> unilab completion >>>"
COMPLETION_BLOCK_END = "# <<< unilab completion <<<"
SUPPORTED_SHELLS = ("bash", "zsh")


@dataclass(frozen=True)
class TaskCompletionEntry:
    algo: str
    task: str
    sim: str
    owner: str


@dataclass(frozen=True)
class CompletionMetadata:
    commands: tuple[str, ...]
    flags: dict[str, tuple[str, ...]]
    choices: dict[str, dict[str, tuple[str, ...]]]
    tasks: tuple[TaskCompletionEntry, ...]
    run_paths: tuple[str, ...] = ()


def _find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _read_project_scripts(pyproject_path: Path) -> dict[str, str]:
    scripts: dict[str, str] = {}
    in_scripts = False
    for line in pyproject_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_scripts = stripped == "[project.scripts]"
            continue
        if not in_scripts:
            continue
        match = SCRIPT_ASSIGNMENT_PATTERN.fullmatch(stripped)
        if match is not None:
            scripts[match.group(1)] = match.group(2)
    return scripts


def _training_commands(scripts: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        command
        for command, target in TRAINING_ENTRYPOINTS.items()
        if scripts.get(command) == target
    )


def _project_commands(scripts: Mapping[str, str]) -> tuple[str, ...]:
    training_commands = _training_commands(scripts)
    commands = [*training_commands]
    commands.extend(command for command in sorted(scripts) if command not in training_commands)
    return tuple(commands)


def _parser_for_command(command: str) -> argparse.ArgumentParser:
    return cli._train_eval_parser(mode=command)


def _parser_flags(command: str) -> tuple[str, ...]:
    parser = _parser_for_command(command)
    flags: list[str] = []
    for action in parser._actions:
        flags.extend(option for option in action.option_strings if option.startswith("--"))
    return tuple(flags)


def _parser_choices(command: str) -> dict[str, tuple[str, ...]]:
    parser = _parser_for_command(command)
    choices: dict[str, tuple[str, ...]] = {}
    for action in parser._actions:
        if action.choices is None:
            continue
        selected = tuple(str(choice) for choice in action.choices)
        for option in action.option_strings:
            if option.startswith("--"):
                choices[option] = selected
    return choices


def _sim_from_owner(owner: str) -> str | None:
    for sim in cli.SUPPORTED_SIMS:
        if owner == sim or owner.startswith(f"{sim}_"):
            return sim
    return None


def _task_entries_for_group(
    root: Path, group: str, algos: Sequence[str]
) -> list[TaskCompletionEntry]:
    entries: list[TaskCompletionEntry] = []
    task_root = root / "conf" / group / "task"
    if not task_root.is_dir():
        return entries
    for task_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
        for owner_yaml in sorted(task_dir.glob("*.yaml")):
            sim = _sim_from_owner(owner_yaml.stem)
            if sim is None:
                continue
            entries.extend(
                TaskCompletionEntry(algo=algo, task=task_dir.name, sim=sim, owner=owner_yaml.stem)
                for algo in algos
            )
    return entries


def _task_entries_for_offpolicy(root: Path) -> list[TaskCompletionEntry]:
    entries: list[TaskCompletionEntry] = []
    task_root = root / "conf" / "offpolicy" / "task"
    if not task_root.is_dir():
        return entries
    for algo in cli.SUPPORTED_ALGOS:
        algo_root = task_root / algo
        if not algo_root.is_dir():
            continue
        for task_dir in sorted(path for path in algo_root.iterdir() if path.is_dir()):
            for owner_yaml in sorted(task_dir.glob("*.yaml")):
                sim = _sim_from_owner(owner_yaml.stem)
                if sim is None:
                    continue
                entries.append(
                    TaskCompletionEntry(
                        algo=algo,
                        task=task_dir.name,
                        sim=sim,
                        owner=owner_yaml.stem,
                    )
                )
    return entries


def _task_entries(root: Path) -> tuple[TaskCompletionEntry, ...]:
    entries = [
        *_task_entries_for_group(root, "ppo", ("ppo", "mlx_ppo")),
        *_task_entries_for_group(root, "appo", ("appo",)),
        *_task_entries_for_offpolicy(root),
    ]
    return tuple(
        sorted(entries, key=lambda entry: (entry.task, entry.algo, entry.sim, entry.owner))
    )


def _is_run_path(path: Path) -> bool:
    return path.is_file() and path.suffix in RUN_PATH_SUFFIXES and path.name != "__init__.py"


def _has_run_path_descendant(path: Path) -> bool:
    return any(
        child.name != "__init__.py" and child.suffix in RUN_PATH_SUFFIXES
        for child in path.rglob("*")
    )


def _run_path_entries(root: Path) -> tuple[str, ...]:
    entries: set[str] = set()
    for path_root in RUN_PATH_ROOTS:
        base = root / path_root
        if not base.is_dir():
            continue
        entries.add(f"{path_root}/")
        for path in base.rglob("*"):
            if path.name.startswith("."):
                continue
            relative_parts = path.relative_to(root).parts
            if any(part in RUN_PATH_IGNORED_PARTS for part in relative_parts):
                continue
            relative_path = path.relative_to(root).as_posix()
            if path.is_dir():
                if _has_run_path_descendant(path):
                    entries.add(f"{relative_path}/")
            elif _is_run_path(path):
                entries.add(relative_path)
    return tuple(sorted(entries))


def build_metadata(root: Path | None = None) -> CompletionMetadata:
    selected_root = root or _find_project_root(Path.cwd()) or cli.repo_root()
    scripts = _read_project_scripts(selected_root / "pyproject.toml")
    training_commands = _training_commands(scripts)
    return CompletionMetadata(
        commands=_project_commands(scripts),
        flags={command: _parser_flags(command) for command in training_commands},
        choices={command: _parser_choices(command) for command in training_commands},
        tasks=_task_entries(selected_root),
        run_paths=_run_path_entries(selected_root),
    )


def _current_word(words: Sequence[str], cword: int) -> str:
    if 0 <= cword < len(words):
        return words[cword]
    return ""


def _previous_word(words: Sequence[str], cword: int) -> str:
    if cword > 0 and cword - 1 < len(words):
        return words[cword - 1]
    return ""


def _matching(candidates: Sequence[str], prefix: str) -> list[str]:
    return [candidate for candidate in candidates if candidate.startswith(prefix)]


def _dedupe(candidates: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(candidates))


def _path_choices(candidates: Sequence[str], prefix: str) -> list[str]:
    choices: set[str] = set()
    for candidate in candidates:
        if not candidate.startswith(prefix):
            continue
        remainder = candidate[len(prefix) :]
        if remainder == "":
            continue
        if "/" in remainder:
            choices.add(f"{prefix}{remainder.split('/', 1)[0]}/")
        else:
            choices.add(candidate)
    return sorted(choices)


def _option_state(words: Sequence[str], cword: int) -> tuple[dict[str, str], set[str]]:
    values: dict[str, str] = {}
    used: set[str] = set()
    index = 3
    limit = min(cword, len(words))
    while index < limit:
        token = words[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            option, value = token.split("=", 1)
            used.add(option)
            values[option] = value
            index += 1
            continue
        used.add(token)
        if index + 1 < limit and not words[index + 1].startswith("--"):
            values[token] = words[index + 1]
            index += 2
            continue
        index += 1
    return values, used


def _available_flags(metadata: CompletionMetadata, command: str, used: set[str]) -> tuple[str, ...]:
    return tuple(flag for flag in metadata.flags.get(command, ()) if flag not in used)


def _task_choices(
    metadata: CompletionMetadata,
    *,
    prefix: str,
    selected_algo: str | None,
    selected_sim: str | None,
    selected_profile: str | None,
) -> list[str]:
    tasks: set[str] = set()
    for entry in metadata.tasks:
        if selected_algo is not None and entry.algo != selected_algo:
            continue
        if selected_sim is not None and entry.sim != selected_sim:
            continue
        if selected_profile is not None:
            if selected_sim is not None and entry.owner != f"{selected_sim}_{selected_profile}":
                continue
            if selected_sim is None and not entry.owner.endswith(f"_{selected_profile}"):
                continue
        tasks.add(entry.task)
    return _matching(tuple(sorted(tasks)), prefix)


def complete_words(
    words: Sequence[str],
    cword: int,
    metadata: CompletionMetadata | None = None,
) -> list[str]:
    selected_metadata = metadata or build_metadata()
    if len(words) < 2 or words[0] != "uv" or words[1] != "run":
        return []

    current = _current_word(words, cword)
    if cword <= 2:
        return _dedupe(
            [
                *_matching(selected_metadata.commands, current),
                *_path_choices(selected_metadata.run_paths, current),
            ]
        )

    command = words[2]
    if command not in selected_metadata.commands:
        return []

    previous = _previous_word(words, cword)
    option_values, used_options = _option_state(words, cword)
    choices = selected_metadata.choices.get(command, {})
    if previous == "--task":
        return _task_choices(
            selected_metadata,
            prefix=current,
            selected_algo=option_values.get("--algo"),
            selected_sim=option_values.get("--sim"),
            selected_profile=option_values.get("--profile"),
        )
    if previous in choices:
        return _matching(choices[previous], current)
    if current == "" or current.startswith("-") or previous == command:
        return _matching(_available_flags(selected_metadata, command, used_options), current)
    return []


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="unilab-complete")
    parser.add_argument("--cword", type=int, required=True)
    parser.add_argument("words", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.words and args.words[0] == "--":
        args.words = args.words[1:]
    return args


def _detect_shell(selected: str) -> str:
    if selected != "auto":
        return selected

    shell_name = Path(os.environ.get("SHELL", "")).name
    if shell_name in SUPPORTED_SHELLS:
        return shell_name
    system_name = platform.system()
    if system_name == "Darwin":
        return "zsh"
    if system_name == "Linux":
        return "bash"
    raise SystemExit("Cannot detect shell for completion install; use --shell bash or --shell zsh.")


def _default_rc_file(shell: str) -> Path:
    if shell == "bash":
        return Path.home() / ".bashrc"
    if shell == "zsh":
        return Path.home() / ".zshrc"
    raise SystemExit(f"Unsupported shell={shell!r}; choose one of: {', '.join(SUPPORTED_SHELLS)}")


def _completion_script_path(shell: str) -> Path:
    root = _find_project_root(Path.cwd()) or cli.repo_root()
    return root / "scripts" / "completions" / f"unilab.{shell}"


def _quote_shell_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _completion_block(script_path: Path) -> str:
    quoted = _quote_shell_path(script_path)
    return (
        f"{COMPLETION_BLOCK_START}\n"
        f'if [ -f "{quoted}" ]; then\n'
        f'    source "{quoted}"\n'
        "fi\n"
        f"{COMPLETION_BLOCK_END}\n"
    )


def _without_completion_block(content: str, *, force: bool = False) -> str:
    start = content.find(COMPLETION_BLOCK_START)
    end = content.find(COMPLETION_BLOCK_END)
    if start == -1 and end == -1:
        return content
    if start == -1 or end == -1 or end < start:
        if force:
            return content
        raise SystemExit(
            "Existing UniLab completion block is malformed; rerun with --force or edit it manually."
        )

    end_line = content.find("\n", end)
    if end_line == -1:
        end_line = len(content)
    else:
        end_line += 1
    return (content[:start].rstrip() + "\n" + content[end_line:].lstrip("\n")).lstrip("\n")


def _write_completion_block(
    *,
    rc_file: Path,
    script_path: Path,
    dry_run: bool,
    force: bool,
) -> None:
    content = rc_file.read_text(encoding="utf-8") if rc_file.exists() else ""
    cleaned = _without_completion_block(content, force=force).rstrip()
    block = _completion_block(script_path).rstrip()
    updated = f"{cleaned}\n\n{block}\n" if cleaned else f"{block}\n"
    if dry_run:
        print(updated, end="")
        return
    rc_file.parent.mkdir(parents=True, exist_ok=True)
    rc_file.write_text(updated, encoding="utf-8")
    print(f"Installed UniLab completion in {rc_file}")
    print(f"Open a new shell or run: source {rc_file}")


def _remove_completion_block(*, rc_file: Path, dry_run: bool, force: bool) -> None:
    if not rc_file.exists():
        print(f"No rc file found: {rc_file}")
        return
    content = rc_file.read_text(encoding="utf-8")
    updated = _without_completion_block(content, force=force)
    if dry_run:
        print(updated, end="")
        return
    rc_file.write_text(updated, encoding="utf-8")
    print(f"Removed UniLab completion from {rc_file}")


def _install_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unilab-complete install")
    parser.add_argument("--shell", choices=("auto", *SUPPORTED_SHELLS), default="auto")
    parser.add_argument("--rc-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    return parser


def install_main(argv: Sequence[str] | None = None) -> int:
    args = _install_parser().parse_args(argv)
    shell = _detect_shell(args.shell)
    rc_file = args.rc_file or _default_rc_file(shell)
    if args.uninstall:
        _remove_completion_block(rc_file=rc_file, dry_run=args.dry_run, force=args.force)
        return 0
    _write_completion_block(
        rc_file=rc_file,
        script_path=_completion_script_path(shell),
        dry_run=args.dry_run,
        force=args.force,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    selected_argv = list(sys.argv[1:] if argv is None else argv)
    if selected_argv and selected_argv[0] == "install":
        return install_main(selected_argv[1:])

    args = _parse_args(selected_argv)
    for candidate in complete_words(args.words, args.cword):
        print(candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
