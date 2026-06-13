from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "src/unilab/base/backend/drake/native/drake_env_pool.cc"
OUTPUT_DIR = REPO_ROOT / "src/unilab/base/backend/drake/native"


def _pkg_config_flags(package: str, option: str) -> list[str]:
    try:
        result = subprocess.run(
            ["pkg-config", option, package],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return result.stdout.split()


def build_command(drake_home: Path, output: Path) -> list[str]:
    python_include = Path(sysconfig.get_paths()["include"])
    pybind_include = drake_home / "include/pybind11"
    include_flags = [
        f"-I{python_include}",
        f"-I{drake_home / 'include'}",
        f"-I{pybind_include}",
        *_pkg_config_flags("eigen3", "--cflags"),
        *_pkg_config_flags("fmt", "--cflags"),
    ]
    lib_flags = [
        *_pkg_config_flags("fmt", "--libs"),
    ]
    lib_dir = drake_home / "lib"
    python_link_flags = []
    if platform.system() == "Darwin":
        python_link_flags = ["-undefined", "dynamic_lookup"]
    return [
        os.environ.get("CXX", sysconfig.get_config_var("CXX") or "c++"),
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-shared",
        *python_link_flags,
        *include_flags,
        str(SOURCE),
        f"-L{lib_dir}",
        "-ldrake",
        *lib_flags,
        "-Wl,-rpath," + str(lib_dir),
        "-o",
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drake-home",
        default=os.environ.get("DRAKE_HOME"),
        help="Drake C++ install prefix containing include/ and lib/libdrake.so.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.drake_home:
        raise SystemExit("Set DRAKE_HOME or pass --drake-home to the Drake C++ install prefix.")
    drake_home = Path(args.drake_home).expanduser().resolve()
    if not (drake_home / "include/drake").is_dir():
        raise SystemExit(f"Drake include directory not found under {drake_home}")
    if not (drake_home / "lib/libdrake.so").exists():
        raise SystemExit(f"Drake shared library not found under {drake_home}")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise SystemExit("Could not determine Python extension suffix")
    output = OUTPUT_DIR / ("_drake_env_pool" + ext_suffix)
    command = build_command(drake_home, output)
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
