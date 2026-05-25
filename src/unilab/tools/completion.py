"""Shell completion helper for UniLab training entrypoints."""

from __future__ import annotations

import argparse
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
SCRIPT_ASSIGNMENT_PATTERN = re.compile(r'^([A-Za-z0-9_.-]+)\s*=\s*"([^"]+)"\s*(?:#.*)?$')


@dataclass(frozen=True)
class CompletionMetadata:
    commands: tuple[str, ...]
    flags: dict[str, tuple[str, ...]]
    choices: dict[str, dict[str, tuple[str, ...]]]


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


def build_metadata(root: Path | None = None) -> CompletionMetadata:
    selected_root = root or _find_project_root(Path.cwd()) or cli.repo_root()
    scripts = _read_project_scripts(selected_root / "pyproject.toml")
    commands = _training_commands(scripts)
    return CompletionMetadata(
        commands=commands,
        flags={command: _parser_flags(command) for command in commands},
        choices={command: _parser_choices(command) for command in commands},
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
        return _matching(selected_metadata.commands, current)

    command = words[2]
    if command not in selected_metadata.commands:
        return []

    previous = _previous_word(words, cword)
    choices = selected_metadata.choices.get(command, {})
    if previous in choices:
        return _matching(choices[previous], current)
    if current.startswith("-") or previous == command:
        return _matching(selected_metadata.flags.get(command, ()), current)
    return []


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="unilab-complete")
    parser.add_argument("--cword", type=int, required=True)
    parser.add_argument("words", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.words and args.words[0] == "--":
        args.words = args.words[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for candidate in complete_words(args.words, args.cword):
        print(candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
