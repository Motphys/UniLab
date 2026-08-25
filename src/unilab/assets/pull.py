#!/usr/bin/env python3
"""Pre-fetch robot binary assets from Hugging Face into their project paths.

Robot meshes and textures are hosted on Hugging Face rather than committed to git.
They are also downloaded automatically on first use, but this command lets you pull
them ahead of time (e.g. for CI or offline prep) with a single invocation. Files land
under ``src/unilab/assets/robots/<robot>/`` — no manual file moving needed.

Usage:
  uv run unilab-pull-assets               # pull the default robot (x2)
  uv run unilab-pull-assets --robot x2
  uv run unilab-pull-assets --robot t800
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from unilab.assets.hub import resolve_robot_asset_dir

# robot name -> ((ASSETS_ROOT_PATH-relative dir, marker, glob, label), ...)
_ROBOT_ASSETS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "x2": (("robots/x2/meshes", "pelvis.STL", "*.STL", "STL"),),
    "t800": (
        ("robots/t800/assets", "LINK_BASE.obj", "*.obj", "OBJ"),
        ("robots/t800/textures", "LINK_BASE.png", "*.png", "PNG"),
    ),
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="x2",
        choices=sorted(_ROBOT_ASSETS),
        help="Robot whose binary assets to download (default: x2).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    for directory, marker, pattern, label in _ROBOT_ASSETS[args.robot]:
        target = resolve_robot_asset_dir(directory, marker=marker)
        count = len(list(target.glob(pattern)))
        print(f"{args.robot} assets ready at {target} ({count} {label} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
