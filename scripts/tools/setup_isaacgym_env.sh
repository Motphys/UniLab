#!/usr/bin/env bash
# Install an external, self-contained IsaacGym Preview 4 runtime for UniLab.
#
# IsaacGym (NVIDIA Preview 4, EOL) only supports Python 3.6-3.8, so it cannot
# live in the main uv environment (requires-python >= 3.10). Everything is
# installed under UNILAB_ISAACGYM_HOME (default: $HOME/.unilab/isaacgym) and
# located at runtime purely through environment variables; this repo never
# hard-codes machine-local paths.
#
# Layout produced by this script (aligned with the benchmark env vars used by
# scripts/benchmark/physics/benchmark_physics_step_isaacgym.py):
#   $UNILAB_ISAACGYM_HOME/miniconda3                        dedicated miniconda
#   $UNILAB_ISAACGYM_HOME/miniconda3/envs/hsgym             Python 3.8 conda env
#   $UNILAB_ISAACGYM_HOME/isaacgym/python                   unpacked Preview 4
#   $UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz downloaded tarball
#
# The script is idempotent: completed steps are skipped on re-run.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup_isaacgym_env.sh [--tarball <path>] [-h|--help]

Install an external Python 3.8 IsaacGym Preview 4 runtime for UniLab
benchmarks. IsaacGym cannot be installed into the main uv environment.

Options:
  --tarball <path>   Path to IsaacGym_Preview_4_Package.tar.gz.
                     Default: auto-downloaded (no login required) to
                     $UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz
                     from https://developer.nvidia.com/isaac-gym-preview-4 ;
                     use this option to supply a pre-downloaded tarball
                     instead (e.g. on a machine without internet access).
  -h, --help         Show this help.

Environment:
  UNILAB_ISAACGYM_HOME   Install root. Default: $HOME/.unilab/isaacgym
EOF
}

ISAACGYM_HOME="${UNILAB_ISAACGYM_HOME:-$HOME/.unilab/isaacgym}"
TARBALL_ARG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tarball)
      [ "$#" -ge 2 ] || { echo "error: --tarball requires a path argument" >&2; exit 2; }
      TARBALL_ARG="$2"
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

CONDA_ROOT="$ISAACGYM_HOME/miniconda3"
ENV_NAME="hsgym"
ENV_ROOT="$CONDA_ROOT/envs/$ENV_NAME"
HSGYM_PYTHON="$ENV_ROOT/bin/python3.8"
ISAACGYM_DIR="$ISAACGYM_HOME/isaacgym"
TARBALL_PATH="${TARBALL_ARG:-$ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz}"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
LIBSTDCXX_SENTINEL="$ENV_ROOT/.unilab_libstdcxx_installed"

log() {
  echo "[setup_isaacgym_env] $*"
}

mkdir -p "$ISAACGYM_HOME"

# Step 1: dedicated miniconda (skipped if already installed).
if [ -x "$CONDA_ROOT/bin/conda" ]; then
  log "miniconda already present at $CONDA_ROOT, skipping"
else
  log "installing miniconda into $CONDA_ROOT"
  installer="$(mktemp /tmp/unilab_miniconda_XXXXXX.sh)"
  curl -fsSL "$MINICONDA_URL" -o "$installer"
  bash "$installer" -b -u -p "$CONDA_ROOT"
  rm -f "$installer"
fi

# Step 2: Python 3.8 conda env with the Ubuntu 24.04 GLIBCXX fix.
if [ -x "$HSGYM_PYTHON" ]; then
  log "conda env '$ENV_NAME' already present at $ENV_ROOT, skipping create"
else
  log "creating conda env '$ENV_NAME' (python=3.8)"
  "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  "$CONDA_ROOT/bin/conda" install -y -n base -c conda-forge mamba
  MAMBA_ROOT_PREFIX="$CONDA_ROOT" "$CONDA_ROOT/bin/mamba" create -y -n "$ENV_NAME" \
    python=3.8 -c conda-forge --override-channels
fi

# Fix `version GLIBCXX_3.4.32 not found` on Ubuntu 24.04 by shipping a newer
# libstdc++ inside the env.
if [ -f "$LIBSTDCXX_SENTINEL" ]; then
  log "libstdcxx-ng already installed in '$ENV_NAME', skipping"
else
  log "installing libstdcxx-ng into '$ENV_NAME'"
  "$CONDA_ROOT/bin/conda" install -y -n "$ENV_NAME" -c conda-forge libstdcxx-ng
  touch "$LIBSTDCXX_SENTINEL"
fi

# Step 3: fetch the IsaacGym tarball. The NVIDIA download URL redirects to a
# signed URL without requiring a login, so the script can fetch it directly;
# --tarball or a pre-placed file skips the download.
if [ -n "$TARBALL_ARG" ] && [ ! -f "$TARBALL_PATH" ]; then
  echo "[setup_isaacgym_env] error: --tarball file not found: $TARBALL_PATH" >&2
  exit 1
fi
if [ ! -f "$TARBALL_PATH" ]; then
  log "downloading IsaacGym Preview 4 tarball (~200 MB) to $TARBALL_PATH"
  curl -fL --retry 3 "https://developer.nvidia.com/isaac-gym-preview-4" \
    -o "$TARBALL_PATH.part"
  mv "$TARBALL_PATH.part" "$TARBALL_PATH"
fi
if ! tar -tzf "$TARBALL_PATH" >/dev/null 2>&1; then
  echo "[setup_isaacgym_env] error: $TARBALL_PATH is not a valid gzip tarball (download may have failed)" >&2
  exit 1
fi

# Step 4: unpack and pip install -e with the env's own pip.
if [ -d "$ISAACGYM_DIR/python" ]; then
  log "isaacgym already unpacked at $ISAACGYM_DIR, skipping extract"
else
  log "unpacking $TARBALL_PATH into $ISAACGYM_HOME"
  tar -xzf "$TARBALL_PATH" -C "$ISAACGYM_HOME"
fi

if "$ENV_ROOT/bin/pip" show isaacgym >/dev/null 2>&1; then
  log "isaacgym already installed in '$ENV_NAME', skipping pip install"
else
  log "installing isaacgym (pip install -e) into '$ENV_NAME'"
  "$ENV_ROOT/bin/pip" install -e "$ISAACGYM_DIR/python"
fi

# Step 5: self-check the import with the env's lib/ on LD_LIBRARY_PATH. The
# gymtorch import triggers a one-time JIT compile of its C++ extension
# (several minutes on a fresh machine, cached afterwards); doing it here keeps
# the first backend INIT handshake fast. ninja must be reachable via the env's
# bin/ on PATH.
log "running isaacgym import self-check (first run compiles the gymtorch extension)"
PATH="$ENV_ROOT/bin:$PATH" LD_LIBRARY_PATH="$ENV_ROOT/lib" "$HSGYM_PYTHON" \
  -c "from isaacgym import gymapi, gymtorch; print('isaacgym import OK')"

# Step 6: print the exports and verification command for the user's shell rc.
cat <<EOF

[setup_isaacgym_env] 完成。把下面几行写入你的 shell rc（如 ~/.bashrc）：

export UNILAB_BENCHMARK_HOLOSOMA_DEPS="$ISAACGYM_HOME"
export UNILAB_BENCHMARK_HSGYM_PYTHON="$HSGYM_PYTHON"
export UNILAB_BENCHMARK_HSGYM_LIB="$ENV_ROOT/lib"
# URDF 模型树（go1_description/ 等）需自备，指向其根目录：
export UNILAB_BENCHMARK_MODELS_ROOT="<path-to-your-urdf-models-root>"

# 训练后端（task=<task>/isaacgym）默认从 ~/.unilab/isaacgym 自动发现运行时；
# 若用了自定义 UNILAB_ISAACGYM_HOME，训练时同名导出即可：
export UNILAB_ISAACGYM_HOME="$ISAACGYM_HOME"

然后在仓库根目录用 benchmark 脚本验证：

PYTHONPATH="\$UNILAB_BENCHMARK_HOLOSOMA_DEPS/isaacgym/python" \\
LD_LIBRARY_PATH="\$UNILAB_BENCHMARK_HSGYM_LIB" \\
uv run --no-project "\$UNILAB_BENCHMARK_HSGYM_PYTHON" \\
    scripts/benchmark/physics/benchmark_physics_step_isaacgym.py \\
    --tasks g1_walk_flat --batch-sizes 256 --models-root "\$UNILAB_BENCHMARK_MODELS_ROOT"
EOF
