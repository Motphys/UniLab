#!/usr/bin/env python3
"""Pre-fetch robot binary assets from Hugging Face into their project paths.

Robot meshes and textures are hosted on Hugging Face rather than committed to git.
They are also downloaded automatically on first use, but this command lets you pull
them ahead of time (e.g. for CI or offline prep) with a single invocation. Files land
under ``src/unilab/assets/robots/<robot>/`` — no manual file moving needed.

Usage:
  uv run unilab-pull-assets               # pull the default robot (x2)
  uv run unilab-pull-assets --robot g1
  uv run unilab-pull-assets --robot microduck
  uv run unilab-pull-assets --robot all   # pull every registered robot
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from unilab.assets.hub import ROBOT_ASSET_SPECS, resolve_robot_asset_dir

_ALL = "all"


@dataclass(frozen=True)
class _AssetSummary:
    target: Path
    count: int
    label: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="x2",
        choices=[*_sorted_robots(), _ALL],
        help="Robot whose binary assets to download, or 'all' (default: x2).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-directory download logs and Hugging Face progress bars.",
    )
    return parser.parse_args(argv)


def _sorted_robots() -> list[str]:
    return sorted(ROBOT_ASSET_SPECS)


def _pull_robot(robot: str, *, show_progress: bool) -> list[_AssetSummary]:
    summaries: list[_AssetSummary] = []
    for directory, marker, pattern, label in ROBOT_ASSET_SPECS[robot]:
        target = resolve_robot_asset_dir(directory, marker=marker, show_progress=show_progress)
        count = sum(1 for path in target.rglob(pattern) if path.is_file())
        summaries.append(_AssetSummary(target=target, count=count, label=label))
    return summaries


def _format_single_robot_summary(robot: str, summaries: Sequence[_AssetSummary]) -> str:
    parts = [f"{summary.count} {summary.label} files at {summary.target}" for summary in summaries]
    return f"{robot} assets ready: {'; '.join(parts)}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    robots = _sorted_robots() if args.robot == _ALL else [args.robot]
    for robot in robots:
        summaries = _pull_robot(robot, show_progress=args.verbose)
        print(_format_single_robot_summary(robot, summaries), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
