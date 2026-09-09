#!/usr/bin/env bash
# Build the locally modified SuperDex source checkout for UniLab development.
#
# This is a temporary source-build path. It deliberately does not publish a
# wheel or install a SuperDex package from PyPI.
#
# Usage:
#   bash scripts/tools/setup_superdex_env.sh \
#     --source /absolute/path/to/project_superdex
#
# Environment:
#   UNISIM_SUPERDEX_HOME   install root (default: ~/.cache/unisim/superdex)
#   SUPERDEX_SOURCE_DIR    source checkout (alternative to --source)
#   SUPERDEX_PYTHON        Python 3.12 executable (default: current python)

set -euo pipefail

usage() {
  sed -n '2,18p' "$0"
}

SOURCE_DIR="${SUPERDEX_SOURCE_DIR:-}"
PREFIX="${UNISIM_SUPERDEX_HOME:-${UNILAB_SUPERDEX_HOME:-$HOME/.cache/unisim/superdex}}"
PYTHON_BIN="${SUPERDEX_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source) [ "$#" -ge 2 ] || { echo "error: --source needs a path" >&2; exit 2; }; SOURCE_DIR="$2"; shift 2 ;;
    --prefix) [ "$#" -ge 2 ] || { echo "error: --prefix needs a path" >&2; exit 2; }; PREFIX="$2"; shift 2 ;;
    --python) [ "$#" -ge 2 ] || { echo "error: --python needs a path" >&2; exit 2; }; PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$SOURCE_DIR" ] || { echo "error: pass --source or set SUPERDEX_SOURCE_DIR" >&2; exit 1; }
SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
[ -f "$SOURCE_DIR/CMakeLists.txt" ] || { echo "error: not a SuperDex source checkout: $SOURCE_DIR" >&2; exit 1; }
[ -x "$PYTHON_BIN" ] || { echo "error: Python executable not found: $PYTHON_BIN" >&2; exit 1; }
command -v cmake >/dev/null || { echo "error: cmake is required" >&2; exit 1; }

BUILD_DIR="$PREFIX/build"
mkdir -p "$PREFIX"
echo "[setup_superdex_env] source=$SOURCE_DIR"
echo "[setup_superdex_env] build=$BUILD_DIR prefix=$PREFIX"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -GNinja \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSUPERDEX_PHYSICS_BUILD_TESTS=OFF \
  -DSUPERDEX_PHYSICS_BUILD_BENCHMARKS=OFF \
  -DMOCHI_BUILD_DEBUGGER=OFF \
  -DMOCHI_BUILD_RENDERER=OFF \
  -DMOCHI_BUILD_MESH_CLI=OFF \
  -DMOCHI_BUILD_SHARED=ON \
  -DMOCHI_USE_PYBIND=ON
cmake --build "$BUILD_DIR" --target mochi_physics_pybind superdex_robotics_pybind -j "${SUPERDEX_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
cmake --install "$BUILD_DIR" --prefix "$PREFIX" --component mochi_physics_pybind || true
cmake --install "$BUILD_DIR" --prefix "$PREFIX" --component superdex_robotics_pybind || true

cat <<EOF

[setup_superdex_env] 完成。当前 shell 执行：
export SUPERDEX_ASSETS_PATH="$SOURCE_DIR/assets"
export SUPERDEX_NATIVE_PATH="$PREFIX"

# 让 Python facade 优先加载本地编译的 native extension：
export PYTHONPATH="$PREFIX:$PYTHONPATH"

# UniLab 本地开发安装：
uv pip install -e "$SOURCE_DIR/superdex_physics/wheels/superdex-physics" \\
  -e "$SOURCE_DIR/superdex_robotics/wheels/superdex-robotics" \\
  -e "$(cd "$(dirname "$0")/../.." && pwd)/../unisim[superdex]" \\
  -e "$(cd "$(dirname "$0")/../.." && pwd)"
EOF
