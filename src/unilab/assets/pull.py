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

from unilab.assets.hub import ROBOT_ASSET_SPECS, resolve_robot_asset_dir

_ALL = "all"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="x2",
        choices=[*_sorted_robots(), _ALL],
        help="Robot whose binary assets to download, or 'all' (default: x2).",
    )
    return parser.parse_args(argv)


def _sorted_robots() -> list[str]:
    return sorted(ROBOT_ASSET_SPECS)


def _pull_robot(robot: str) -> None:
    for directory, marker, pattern, label in ROBOT_ASSET_SPECS[robot]:
        target = resolve_robot_asset_dir(directory, marker=marker)
        count = sum(1 for path in target.rglob(pattern) if path.is_file())
        print(f"{robot} assets ready at {target} ({count} {label} files)")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    robots = _sorted_robots() if args.robot == _ALL else [args.robot]
    for robot in robots:
        _pull_robot(robot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
