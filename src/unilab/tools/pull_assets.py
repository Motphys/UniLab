#!/usr/bin/env python3
"""Pre-fetch robot binary assets (meshes) from Hugging Face into their project paths.

Robot STL meshes are hosted on Hugging Face rather than committed to git. They are
also downloaded automatically on first use, but this command lets you pull them
ahead of time (e.g. for CI or offline prep) with a single invocation. Files land
under ``src/unilab/assets/robots/<robot>/meshes/`` — no manual file moving needed.

Usage:
  uv run unilab-pull-assets               # pull the default robot (x2)
  uv run unilab-pull-assets --robot x2
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from unilab.assets.hub import resolve_robot_asset_dir

# robot name -> (ASSETS_ROOT_PATH-relative dir, completeness marker file)
_ROBOT_ASSETS: dict[str, tuple[str, str]] = {
    "x2": ("robots/x2/meshes", "pelvis.STL"),
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="x2",
        choices=sorted(_ROBOT_ASSETS),
        help="Robot whose meshes to download (default: x2).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    directory, marker = _ROBOT_ASSETS[args.robot]
    target = resolve_robot_asset_dir(directory, marker=marker)
    count = len(list(target.glob("*.STL")))
    print(f"{args.robot} meshes ready at {target} ({count} STL files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
