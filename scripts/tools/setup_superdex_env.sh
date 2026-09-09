#!/usr/bin/env bash
# Build the locally modified SuperDex source checkout for UniLab development.
#
# This is a temporary source-build path. It deliberately does not publish a
# wheel or install a SuperDex package from PyPI.
#
# Usage:
#   bash scripts/tools/setup_superdex_env.sh
#   bash scripts/tools/setup_superdex_env.sh --source /absolute/path/to/project_superdex
#
# Environment:
#   UNISIM_SUPERDEX_HOME   install root (default: ~/.cache/unisim/superdex)
#   SUPERDEX_SOURCE_DIR    source checkout (optional; otherwise cloned automatically)
#   SUPERDEX_REPOSITORY    git URL (default: https://github.com/unilabsim/project_superdex.git)
#   SUPERDEX_REF           git ref (default: dev/issue-2-superdex-throughput)
#   SUPERDEX_PYTHON        Python 3.12 executable (default: current python)

set -euo pipefail

usage() {
  sed -n '2,18p' "$0"
}

SOURCE_DIR="${SUPERDEX_SOURCE_DIR:-}"
REPOSITORY="${SUPERDEX_REPOSITORY:-https://github.com/unilabsim/project_superdex.git}"
SUPERDEX_REF="${SUPERDEX_REF:-dev/issue-2-superdex-throughput}"
PREFIX="${UNISIM_SUPERDEX_HOME:-${UNILAB_SUPERDEX_HOME:-$HOME/.cache/unisim/superdex}}"
PYTHON_BIN="${SUPERDEX_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source) [ "$#" -ge 2 ] || { echo "error: --source needs a path" >&2; exit 2; }; SOURCE_DIR="$2"; shift 2 ;;
    --repository) [ "$#" -ge 2 ] || { echo "error: --repository needs a URL" >&2; exit 2; }; REPOSITORY="$2"; shift 2 ;;
    --ref) [ "$#" -ge 2 ] || { echo "error: --ref needs a git ref" >&2; exit 2; }; SUPERDEX_REF="$2"; shift 2 ;;
    --prefix) [ "$#" -ge 2 ] || { echo "error: --prefix needs a path" >&2; exit 2; }; PREFIX="$2"; shift 2 ;;
    --python) [ "$#" -ge 2 ] || { echo "error: --python needs a path" >&2; exit 2; }; PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$SOURCE_DIR" ]; then
  SOURCE_DIR="$PREFIX/source"
  if [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    command -v git >/dev/null || { echo "error: git is required for automatic checkout" >&2; exit 1; }
    mkdir -p "$PREFIX"
    echo "[setup_superdex_env] cloning $REPOSITORY@$SUPERDEX_REF"
    git clone --branch "$SUPERDEX_REF" --depth 1 "$REPOSITORY" "$SOURCE_DIR"
  fi
fi
if [ -z "${SUPERDEX_PYTHON:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  if command -v uv >/dev/null; then
    mkdir -p "$PREFIX"
    uv venv --python 3.12 "$PREFIX/venv"
    PYTHON_BIN="$PREFIX/venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
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

# Install the public Python facade/runtime into the active environment. The
# native extension remains the locally compiled one exposed through PYTHONPATH.
if command -v uv >/dev/null; then
  uv pip install --python "$PYTHON_BIN" "superdex-physics==1.0.0" "superdex-robotics==1.0.0"
else
  "$PYTHON_BIN" -m pip install "superdex-physics==1.0.0" "superdex-robotics==1.0.0"
fi

ENV_FILE="$PREFIX/env.sh"
cat > "$ENV_FILE" <<EOF
export SUPERDEX_ASSETS_PATH="$SOURCE_DIR/assets"
export SUPERDEX_NATIVE_PATH="$PREFIX"
export PYTHONPATH="$PREFIX:\${PYTHONPATH:-}"
export UV_NO_SYNC=1
EOF

cat <<EOF

[setup_superdex_env] 完成。当前 shell 执行：
source "$ENV_FILE"

# UniLab 本地开发安装：
uv pip install --python "$PYTHON_BIN" \\
  -e "$(cd "$(dirname "$0")/../.." && pwd)/../unisim[superdex]" \\
  -e "$(cd "$(dirname "$0")/../.." && pwd)/../unilab_rl" \\
  -e "$(cd "$(dirname "$0")/../.." && pwd)"
EOF
