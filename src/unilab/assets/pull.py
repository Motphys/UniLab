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
        help="Retained for compatibility; Hugging Face output remains suppressed.",
    )
    return parser.parse_args(argv)


def _sorted_robots() -> list[str]:
    return sorted(ROBOT_ASSET_SPECS)


def _pull_robot(robot: str) -> list[_AssetSummary]:
    summaries: list[_AssetSummary] = []
    for directory, marker, pattern, label in ROBOT_ASSET_SPECS[robot]:
        target = resolve_robot_asset_dir(directory, marker=marker, show_progress=False)
        count = sum(1 for path in target.rglob(pattern) if path.is_file())
        summaries.append(_AssetSummary(target=target, count=count, label=label))
    return summaries


def _format_total_summary(robots: Sequence[str], summaries: Sequence[_AssetSummary]) -> str:
    total_files = sum(summary.count for summary in summaries)
    label_counts: dict[str, int] = {}
    for summary in summaries:
        label_counts[summary.label] = label_counts.get(summary.label, 0) + summary.count
    counts = ", ".join(f"{count} {label} files" for label, count in sorted(label_counts.items()))
    return (
        f"Robot assets ready: {len(robots)} robots, {len(summaries)} directories, "
        f"{total_files} files ({counts})"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    robots = _sorted_robots() if args.robot == _ALL else [args.robot]
    summaries: list[_AssetSummary] = []
    for robot in robots:
        print(f"Downloading {robot} assets ...", flush=True)
        robot_summaries = _pull_robot(robot)
        summaries.extend(robot_summaries)
    print(_format_total_summary(robots, summaries), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
