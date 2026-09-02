#!/usr/bin/env bash
··# Set up the UniLab Drake batch runtime on Linux or Apple Silicon macOS.
#
# This script mirrors the external-runtime setup flow used by IsaacGym and
# IsaacSim: expensive downloads are resumable, completed download/extract steps
# are marked, and all runtime files live outside the repository.  Drake C++ is
# still an explicit external dependency; pass --download-drake when the
# official Drake tarball should be fetched automatically.

set -euo pipefail

UNILAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP_HOME="${UNILAB_DRAKE_SETUP_HOME:-$HOME/.unilab/drake}"
DEFAULT_SOURCE="${UNILAB_ROOT}/../drake_uni"
DRAKE_UNI_SOURCE="${UNILAB_DRAKE_UNI_SOURCE:-}"
DRAKE_PREFIX="${UNILAB_DRAKE_HOME:-${DRAKE_HOME:-}}"
DRAKE_VERSION="${DRAKE_VERSION:-1.56.0}"
DRAKE_DEPS_ROOT="${UNILAB_DRAKE_DEPS_ROOT:-}"
DOWNLOAD_DRAKE=0

usage() {
  cat <<'EOF'
Usage: setup_drake_env.sh [options]

Install UniLab's Drake batch runtime and build the native DrakeUni extension.
The default is to use an existing Drake C++ prefix.  Pass --download-drake to
fetch the official Drake tarball for the current host into UNILAB_DRAKE_SETUP_HOME.

Options:
  --drake-home <path>       Drake C++ prefix (or set DRAKE_HOME).
  --drake-uni-source <path> DrakeUni checkout; defaults to ../drake_uni when it exists.
  --deps-root <path>        Optional unpacked system deps root (or set
                            UNILAB_DRAKE_DEPS_ROOT).
  --download-drake          Download the official host tarball when no prefix is given.
  --drake-version <version> Version used with --download-drake (default: 1.56.0).
  --drake-platform <name>   Official tarball platform (default: noble on Linux,
                            mac-arm64 on Apple Silicon macOS).
  -h, --help                Show this help.

Environment:
  UNILAB_DRAKE_SETUP_HOME   Download/marker/log root (default: ~/.unilab/drake).
  UNILAB_DRAKE_HOME         Same meaning as DRAKE_HOME.
  UNILAB_DRAKE_UNI_SOURCE   DrakeUni source checkout.
  UNILAB_DRAKE_DEPS_ROOT    Unpacked deps root for Eigen/fmt/spdlog.

The complete path requires Linux x86_64 or Apple Silicon macOS, C++20, Eigen,
fmt, spdlog, and a Python ABI matching the UniLab uv environment. The script
never installs system packages with sudo. On Ubuntu install missing packages with:
  sudo apt-get install build-essential pkg-config libeigen3-dev libfmt-dev libspdlog-dev
On macOS install the runtime libraries used by the official arm64 tarball with:
  brew install fmt gcc
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --drake-home)
      [ "$#" -ge 2 ] || { echo "error: --drake-home requires a path" >&2; exit 2; }
      DRAKE_PREFIX="$2"
      shift 2
      ;;
    --drake-uni-source)
      [ "$#" -ge 2 ] || { echo "error: --drake-uni-source requires a path" >&2; exit 2; }
      DRAKE_UNI_SOURCE="$2"
      shift 2
      ;;
    --deps-root)
      [ "$#" -ge 2 ] || { echo "error: --deps-root requires a path" >&2; exit 2; }
      DRAKE_DEPS_ROOT="$2"
      shift 2
      ;;
    --download-drake)
      DOWNLOAD_DRAKE=1
      shift
      ;;
    --drake-version)
      [ "$#" -ge 2 ] || { echo "error: --drake-version requires a value" >&2; exit 2; }
      DRAKE_VERSION="$2"
      shift 2
      ;;
    --drake-platform)
      [ "$#" -ge 2 ] || { echo "error: --drake-platform requires a value" >&2; exit 2; }
      DRAKE_PLATFORM="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"
case "$HOST_OS/$HOST_ARCH" in
  Linux/x86_64)
    DEFAULT_DRAKE_PLATFORM="noble"
    LIBRARY_PATH_VAR="LD_LIBRARY_PATH"
    ;;
  Darwin/arm64)
    DEFAULT_DRAKE_PLATFORM="mac-arm64"
    LIBRARY_PATH_VAR="DYLD_LIBRARY_PATH"
    ;;
  Darwin/*)
    echo "error: Drake's official macOS runtime is arm64 only; Intel macOS is not supported" >&2
    exit 1
    ;;
  *)
    echo "error: unsupported host $HOST_OS/$HOST_ARCH (supported: Linux x86_64, macOS arm64)" >&2
    exit 1
    ;;
esac
DRAKE_PLATFORM="${DRAKE_PLATFORM:-$DEFAULT_DRAKE_PLATFORM}"
command -v uv >/dev/null || { echo "error: uv is required" >&2; exit 1; }
command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }
command -v c++ >/dev/null || { echo "error: a C++ compiler is required" >&2; exit 1; }

MARKER_DIR="$SETUP_HOME/.markers"
LOG_FILE="$SETUP_HOME/install.log"
mkdir -p "$MARKER_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[setup_drake_env $(date +%H:%M:%S)] $*"; }

run_step() {
  local marker="$1"
  shift
  if [ -f "$MARKER_DIR/$marker" ]; then
    log "SKIP  $marker (already done)"
    return 0
  fi
  log "START $marker: $*"
  "$@"
  touch "$MARKER_DIR/$marker"
  log "DONE  $marker"
}

if [ -z "$DRAKE_UNI_SOURCE" ] && [ -d "$DEFAULT_SOURCE/.git" ]; then
  DRAKE_UNI_SOURCE="$DEFAULT_SOURCE"
fi

if [ -z "$DRAKE_PREFIX" ] && [ "$DOWNLOAD_DRAKE" -eq 1 ]; then
  DRAKE_TARBALL="$SETUP_HOME/drake-${DRAKE_VERSION}-${DRAKE_PLATFORM}.tar.gz"
  DRAKE_PREFIX="$SETUP_HOME/drake-${DRAKE_VERSION}-${DRAKE_PLATFORM}"
  download_drake() {
    local url="https://github.com/RobotLocomotion/drake/releases/download/v${DRAKE_VERSION}/drake-${DRAKE_VERSION}-${DRAKE_PLATFORM}.tar.gz"
    if [ ! -f "$DRAKE_TARBALL" ]; then
      log "downloading Drake ${DRAKE_VERSION} to $DRAKE_TARBALL"
      curl -fL -C - --retry 5 --retry-delay 10 -o "$DRAKE_TARBALL.part" "$url"
      mv "$DRAKE_TARBALL.part" "$DRAKE_TARBALL"
    fi
    tar -tzf "$DRAKE_TARBALL" >/dev/null
  }
  extract_drake() {
    if [ ! -d "$DRAKE_PREFIX" ]; then
      mkdir -p "$DRAKE_PREFIX"
      tar -xzf "$DRAKE_TARBALL" --strip-components=1 -C "$DRAKE_PREFIX"
    fi
  }
  run_step 00_download_drake download_drake
  run_step 01_extract_drake extract_drake
elif [ -z "$DRAKE_PREFIX" ]; then
  echo "error: set DRAKE_HOME/UNILAB_DRAKE_HOME, pass --drake-home, or use --download-drake" >&2
  exit 1
fi

DRAKE_PREFIX="$(cd "$DRAKE_PREFIX" && pwd)"
for required in "$DRAKE_PREFIX/include/drake" "$DRAKE_PREFIX/include/pybind11"; do
  [ -d "$required" ] || {
    echo "error: required Drake directory not found: $required" >&2
    exit 1
  }
done
DRAKE_LIBRARY=""
for candidate in "$DRAKE_PREFIX/lib/libdrake.so" "$DRAKE_PREFIX/lib/libdrake.dylib"; do
  if [ -f "$candidate" ]; then
    DRAKE_LIBRARY="$candidate"
    break
  fi
done
[ -n "$DRAKE_LIBRARY" ] || {
  echo "error: Drake shared library not found under $DRAKE_PREFIX/lib (expected libdrake.so or libdrake.dylib)" >&2
  exit 1
}

# Official tarballs ship Eigen and fmt headers, while fmt is linked from the
# host. Prefer explicit overrides, then discover Homebrew's fmt on macOS.
if [ -z "${EIGEN3_INCLUDE_DIR:-}" ]; then
  if [ -f "$DRAKE_PREFIX/include/eigen3/Eigen/Core" ]; then
    EIGEN3_INCLUDE_DIR="$DRAKE_PREFIX/include/eigen3"
  elif [ -f "$DRAKE_PREFIX/include/Eigen/Core" ]; then
    EIGEN3_INCLUDE_DIR="$DRAKE_PREFIX/include"
  fi
fi
if [ -z "${FMT_INCLUDE_DIR:-}" ] && [ -f "$DRAKE_PREFIX/include/fmt/format.h" ]; then
  FMT_INCLUDE_DIR="$DRAKE_PREFIX/include"
fi
if [ "$HOST_OS" = "Darwin" ] && command -v brew >/dev/null; then
  FMT_PREFIX="$(brew --prefix fmt 2>/dev/null || true)"
  if [ -n "$FMT_PREFIX" ] && [ -d "$FMT_PREFIX/lib" ]; then
    FMT_LIB_DIR="${FMT_LIB_DIR:-$FMT_PREFIX/lib}"
    FMT_INCLUDE_DIR="${FMT_INCLUDE_DIR:-$FMT_PREFIX/include}"
  fi
fi

if [ -n "$DRAKE_DEPS_ROOT" ]; then
  DRAKE_DEPS_ROOT="$(cd "$DRAKE_DEPS_ROOT" && pwd)"
  if [ "$HOST_OS" = "Darwin" ]; then
    EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-$DRAKE_DEPS_ROOT/include}"
    FMT_INCLUDE_DIR="${FMT_INCLUDE_DIR:-$DRAKE_DEPS_ROOT/include}"
    FMT_LIB_DIR="${FMT_LIB_DIR:-$DRAKE_DEPS_ROOT/lib}"
    PKG_CONFIG_PATH="$DRAKE_DEPS_ROOT/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    DYLD_LIBRARY_PATH="$DRAKE_DEPS_ROOT/lib:$DRAKE_PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH
  else
    EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-$DRAKE_DEPS_ROOT/usr/include/eigen3}"
    FMT_INCLUDE_DIR="${FMT_INCLUDE_DIR:-$DRAKE_DEPS_ROOT/usr/include}"
    FMT_LIB_DIR="${FMT_LIB_DIR:-$DRAKE_DEPS_ROOT/usr/lib/x86_64-linux-gnu}"
    PKG_CONFIG_PATH="$DRAKE_DEPS_ROOT/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    LD_LIBRARY_PATH="$DRAKE_DEPS_ROOT/usr/lib/x86_64-linux-gnu:$DRAKE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
  fi
  export EIGEN3_INCLUDE_DIR FMT_INCLUDE_DIR FMT_LIB_DIR PKG_CONFIG_PATH
fi

if [ -z "${EIGEN3_INCLUDE_DIR:-}" ] || [ ! -f "$EIGEN3_INCLUDE_DIR/Eigen/Core" ]; then
  echo "error: Eigen headers not found; install libeigen3-dev (Linux), brew install eigen (macOS), or pass --deps-root" >&2
  exit 1
fi
if [ "$HOST_OS" = "Darwin" ] && {
  [ -z "${FMT_LIB_DIR:-}" ] || ! compgen -G "$FMT_LIB_DIR/libfmt*" >/dev/null;
}; then
  echo "error: fmt library not found; install it with 'brew install fmt' or set FMT_LIB_DIR" >&2
  exit 1
fi
export EIGEN3_INCLUDE_DIR FMT_INCLUDE_DIR FMT_LIB_DIR

log "UniLab root: $UNILAB_ROOT"
log "Drake prefix: $DRAKE_PREFIX"
log "DrakeUni source: ${DRAKE_UNI_SOURCE:-<not resolved>}"
log "Setup home: $SETUP_HOME"

run_step 02_sync_uv uv --directory "$UNILAB_ROOT" sync --extra drake
run_step 03_install_completion uv --directory "$UNILAB_ROOT" run --no-sync unilab-complete install

if [ -z "$DRAKE_UNI_SOURCE" ]; then
  DRAKE_UNI_SOURCE="$SETUP_HOME/drake_uni"
  clone_drake_uni() {
    if [ ! -d "$DRAKE_UNI_SOURCE/.git" ]; then
      git clone --depth 1 --branch "${DRAKE_UNI_VERSION:-main}" \
        https://github.com/unilabsim/drake_uni.git "$DRAKE_UNI_SOURCE"
    fi
  }
  run_step 04_fetch_drake_uni clone_drake_uni
fi
DRAKE_UNI_SOURCE="$(cd "$DRAKE_UNI_SOURCE" && pwd)"
[ -f "$DRAKE_UNI_SOURCE/scripts/build_drake_batch.py" ] || {
  echo "error: build_drake_batch.py not found under $DRAKE_UNI_SOURCE" >&2
  exit 1
}

uv --directory "$UNILAB_ROOT" pip install --python "$UNILAB_ROOT/.venv/bin/python" \
  --force-reinstall --no-deps --no-build-isolation -e "$DRAKE_UNI_SOURCE"

RUNTIME_LIBRARY_PATH="${!LIBRARY_PATH_VAR:-$DRAKE_PREFIX/lib}"
if [ "$HOST_OS" = "Darwin" ] && command -v brew >/dev/null; then
  GCC_PREFIX="$(brew --prefix gcc 2>/dev/null || true)"
  if [ -n "$GCC_PREFIX" ] && [ -f "$GCC_PREFIX/lib/gcc/current/libgfortran.5.dylib" ]; then
    RUNTIME_LIBRARY_PATH="$GCC_PREFIX/lib/gcc/current:$RUNTIME_LIBRARY_PATH"
  fi
fi
export "$LIBRARY_PATH_VAR=$RUNTIME_LIBRARY_PATH"

env "$LIBRARY_PATH_VAR=$RUNTIME_LIBRARY_PATH" \
  uv --directory "$UNILAB_ROOT" run --no-sync python \
  "$DRAKE_UNI_SOURCE/scripts/build_drake_batch.py" --drake-home "$DRAKE_PREFIX"

env "$LIBRARY_PATH_VAR=$RUNTIME_LIBRARY_PATH" \
  uv --directory "$UNILAB_ROOT" run --no-sync python - <<'PY'
import drake_uni
from drake_uni.runtime import batch_diagnostics

diagnostics = batch_diagnostics()
print(f"drake_uni import: {drake_uni.__file__}")
print(f"batch diagnostics: {diagnostics}")
if not diagnostics.batch_available:
    raise SystemExit(diagnostics.batch_import_error or "Drake batch extension is unavailable")
PY

cat <<EOF

[setup_drake_env] 完成。当前 shell 如需直接运行 Drake batch，请设置：

export DRAKE_HOME="$DRAKE_PREFIX"
export UNILAB_DRAKE_HOME="$DRAKE_PREFIX"
export UNILAB_DRAKE_UNI_SOURCE="$DRAKE_UNI_SOURCE"
export $LIBRARY_PATH_VAR="$RUNTIME_LIBRARY_PATH"

然后在 UniLab 根目录运行：
uv run --no-sync pytest tests/base/backend/test_drake_batch_pool.py \
  tests/scripts/test_drake_training_smoke.py -q
EOF
