#!/usr/bin/env bash
# Install an external, self-contained IsaacSim + IsaacLab runtime for UniLab.
#
# IsaacSim/IsaacLab require Python 3.11 while the main UniLab uv environment
# uses a newer Python, so (like the IsaacGym worker env) this runtime lives
# outside the repo under UNISIM_ISAACSIM_HOME (default: $HOME/.cache/unisim/isaacsim)
# and is located purely through environment variables; this repo never
# hard-codes machine-local paths.
#
# Layout produced by this script (mirrors setup_isaacgym_env.sh conventions):
#   $UNISIM_ISAACSIM_HOME/venv                 dedicated Python 3.11 venv
#   $UNISIM_ISAACSIM_HOME/IsaacLab             IsaacLab source tree
#   $UNISIM_ISAACSIM_HOME/tools/bin/cmake      official cmake binary (no sudo)
#   $UNISIM_ISAACSIM_HOME/.markers             per-step completion markers
#   $UNISIM_ISAACSIM_HOME/install.log          full install log
#
# Design goals:
#   * Idempotent — every completed step drops a marker file; re-running the
#     script after a failure skips finished steps so multi-GB downloads are
#     never repeated.
#   * Resumable — large downloads use curl -C - so an interrupted transfer
#     continues where it stopped; pip's HTTP cache covers pip downloads.
#   * Isolated — UniLab code, configs, and the UniLab venv are never touched.
#
# Usage:
#   bash scripts/tools/setup_isaacsim_env.sh [--verify-sim]
#
# Environment overrides:
#   UNISIM_ISAACSIM_HOME  install root           (default: ~/.cache/unisim/isaacsim)
#   ISAACLAB_VERSION      IsaacLab git tag       (default: v2.3.0)
#   ISAACSIM_VERSION      isaacsim pip version   (default: 5.1.0)
#   TORCH_VERSION         torch version          (default: 2.7.0)
#   TORCHVISION_VERSION   torchvision version    (default: 0.22.0)
#
# Reference: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
set -euo pipefail

ISAACSIM_HOME=${UNISIM_ISAACSIM_HOME:-${UNILAB_ISAACSIM_HOME:-$HOME/.cache/unisim/isaacsim}}
ISAACLAB_VERSION=${ISAACLAB_VERSION:-v2.3.0}
ISAACSIM_VERSION=${ISAACSIM_VERSION:-5.1.0}
TORCH_VERSION=${TORCH_VERSION:-2.7.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.0}
PYTHON_VERSION=3.11

VENV_DIR=$ISAACSIM_HOME/venv
ISAACLAB_DIR=$ISAACSIM_HOME/IsaacLab
MARKER_DIR=$ISAACSIM_HOME/.markers
LOG_FILE=$ISAACSIM_HOME/install.log

VERIFY_SIM=0
for arg in "$@"; do
  case "$arg" in
    --verify-sim) VERIFY_SIM=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { echo "[setup_isaacsim_env $(date +%H:%M:%S)] $*"; }

# run_step <marker-name> <command...>
# Skips the command if the marker exists; creates the marker on success.
run_step() {
  local marker=$1; shift
  if [[ -f $MARKER_DIR/$marker ]]; then
    log "SKIP  $marker (already done)"
    return 0
  fi
  log "START $marker: $*"
  "$@"
  touch "$MARKER_DIR/$marker"
  log "DONE  $marker"
}

mkdir -p "$MARKER_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log "UNISIM_ISAACSIM_HOME=$ISAACSIM_HOME"
log "isaacsim=$ISAACSIM_VERSION isaaclab=$ISAACLAB_VERSION torch=$TORCH_VERSION+cu128"

# Accept the Omniverse Kit EULA non-interactively (install + any Kit launch)
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-1}

# ---------------------------------------------------------------------------
# Step 0: system build tools (needed by IsaacLab's native extensions, e.g. egl_probe).
# g++ must come from the system; cmake is installed as a standalone binary in
# step 1b so no sudo is required.
# ---------------------------------------------------------------------------
step_system_deps() {
  if ! command -v g++ >/dev/null; then
    echo "g++ (build-essential) is required; install it with: sudo apt-get install -y build-essential" >&2
    return 1
  fi
  log "g++ present: $(g++ --version | head -1)"
}
run_step 00_system_deps step_system_deps

# ---------------------------------------------------------------------------
# Step 1: Python 3.11 venv (IsaacSim/IsaacLab require Python 3.11)
# ---------------------------------------------------------------------------
step_create_venv() {
  # Remove a leftover broken venv (e.g. created without pip by a failed run);
  # venv creation is cheap, no downloads are lost.
  if [[ -d $VENV_DIR && ! -x $VENV_DIR/bin/pip ]]; then
    rm -rf "$VENV_DIR"
  fi
  if command -v uv >/dev/null; then
    uv venv --python "$PYTHON_VERSION" --seed "$VENV_DIR"
  elif command -v "python$PYTHON_VERSION" >/dev/null; then
    "python$PYTHON_VERSION" -m venv "$VENV_DIR"
  else
    echo "Need uv or python$PYTHON_VERSION to create the venv" >&2
    return 1
  fi
  "$VENV_DIR/bin/pip" install --upgrade pip
}
run_step 01_venv step_create_venv

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

# ---------------------------------------------------------------------------
# Step 1b: real cmake binary (avoids sudo; used by egl_probe's native build).
# Do NOT use pip's `cmake` wheel here: its console script does
# `from cmake import cmake`, which breaks inside pip's isolated build envs.
# ---------------------------------------------------------------------------
step_cmake() {
  local ver=3.30.5
  local dest=$ISAACSIM_HOME/tools/cmake-$ver-linux-x86_64
  if [[ ! -x $dest/bin/cmake ]]; then
    mkdir -p "$ISAACSIM_HOME/tools"
    local tarball=$ISAACSIM_HOME/tools/cmake-$ver-linux-x86_64.tar.gz
    curl -fL -C - --retry 5 --retry-delay 10 -o "$tarball" \
      "https://github.com/Kitware/CMake/releases/download/v$ver/cmake-$ver-linux-x86_64.tar.gz"
    tar -xzf "$tarball" -C "$ISAACSIM_HOME/tools"
    rm -f "$tarball"
  fi
  mkdir -p "$ISAACSIM_HOME/tools/bin"
  ln -sfn "$dest/bin/cmake" "$ISAACSIM_HOME/tools/bin/cmake"
  # drop the fragile pip-cmake entry point so it cannot shadow the real binary
  "$PIP" uninstall -y cmake >/dev/null 2>&1 || true
}
run_step 01b_cmake step_cmake

# venv bin + real cmake on PATH for IsaacLab's native builds
export PATH="$ISAACSIM_HOME/tools/bin:$VENV_DIR/bin:$PATH"

# ---------------------------------------------------------------------------
# Step 2: PyTorch (CUDA 12.8 wheels). pip's HTTP cache (~/.cache/pip) makes
# re-runs cheap if the wheel was fetched before.
# ---------------------------------------------------------------------------
run_step 02_torch \
  "$PIP" install -U "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# Step 3: IsaacSim from the NVIDIA PyPI index (multi-GB; cached by pip on re-run)
# ---------------------------------------------------------------------------
step_isaacsim() {
  "$PIP" install pyperclip
  "$PIP" install "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
    --extra-index-url https://pypi.nvidia.com
}
run_step 03_isaacsim step_isaacsim

# ---------------------------------------------------------------------------
# Step 4: IsaacLab source. Tarball + curl -C - so a broken download resumes
# instead of restarting (git clone cannot resume).
# ---------------------------------------------------------------------------
step_fetch_isaaclab() {
  local tarball=$ISAACSIM_HOME/isaaclab-$ISAACLAB_VERSION.tar.gz
  local url=https://github.com/isaac-sim/IsaacLab/archive/refs/tags/$ISAACLAB_VERSION.tar.gz
  if [[ ! -d $ISAACLAB_DIR ]]; then
    if [[ ! -d $ISAACSIM_HOME/IsaacLab-${ISAACLAB_VERSION#v} ]]; then
      curl -fL -C - --retry 5 --retry-delay 10 -o "$tarball" "$url"
      tar -xzf "$tarball" -C "$ISAACSIM_HOME"
    fi
    mv "$ISAACSIM_HOME/IsaacLab-${ISAACLAB_VERSION#v}" "$ISAACLAB_DIR"
    rm -f "$tarball"
  fi
}
run_step 04_isaaclab_src step_fetch_isaaclab

# ---------------------------------------------------------------------------
# Step 5: install IsaacLab extensions into the venv.
# Upstream workarounds:
#   * setuptools>=81 removed pkg_resources -> pin via PIP_BUILD_CONSTRAINT
#   * flatdict 4.0.1 upstream bug -> 4.1.0 (IsaacLab#4576)
#   * egl_probe cmake max-version issue -> CMAKE_POLICY_VERSION_MINIMUM=3.5
#   * torch pinned via PIP_CONSTRAINT so RL-framework deps (e.g. sb3 2.9.x)
#     cannot upgrade torch to a newer CUDA build mid-install
# ---------------------------------------------------------------------------
step_install_isaaclab() {
  cd "$ISAACLAB_DIR"
  "$PIP" install 'setuptools<81'
  echo 'setuptools<81' > build-constraints.txt
  export PIP_BUILD_CONSTRAINT="$ISAACLAB_DIR/build-constraints.txt"
  # Pin torch stack: pip would otherwise upgrade to a newer torch/CUDA build
  # (pulled in by stable_baselines3 2.9.x), replacing the cu128 build
  # IsaacSim 5.1 expects and downloading many GB unnecessarily.
  printf 'torch==%s+cu128\ntorchvision==%s+cu128\ntorchaudio==2.7.0\n' \
    "$TORCH_VERSION" "$TORCHVISION_VERSION" > "$ISAACLAB_DIR/install-constraints.txt"
  export PIP_CONSTRAINT="$ISAACLAB_DIR/install-constraints.txt"
  sed -i 's/flatdict==4\.0\.1/flatdict==4.1.0/' source/isaaclab/setup.py
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  # isaaclab.sh runs `tabs 4` under `set -e`; it aborts on TERM=dumb
  TERM=xterm ./isaaclab.sh --install
  unset PIP_BUILD_CONSTRAINT PIP_CONSTRAINT
}
run_step 05_isaaclab_install step_install_isaaclab

# ---------------------------------------------------------------------------
# Step 6: verification (imports + CUDA visibility)
# ---------------------------------------------------------------------------
step_verify() {
  "$PYTHON" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("torch", torch.__version__, "cuda", torch.version.cuda)
import isaacsim  # noqa: F401
print("isaacsim import OK")
import isaaclab  # noqa: F401
print("isaaclab", isaaclab.__version__)
EOF
}
run_step 06_verify step_verify

# ---------------------------------------------------------------------------
# Optional: bounded headless smoke test (real physics stepping).
# NOTE: the standalone/full IsaacSim app segfaults in librtx.scenedb with
# driver 595.x (IsaacLab#6785); the IsaacLab headless path works, which is
# what the UniLab backend will use. Bounded steps so the test terminates.
# ---------------------------------------------------------------------------
if [[ $VERIFY_SIM -eq 1 ]]; then
  log "Running bounded headless IsaacLab smoke test (first launch may take minutes)"
  "$PYTHON" - <<'EOF'
from isaaclab.app import AppLauncher
simulation_app = AppLauncher({"headless": True}).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  (registers all IsaacLab envs)
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg("Isaac-Cartpole-v0", num_envs=16)
env = gym.make("Isaac-Cartpole-v0", cfg=env_cfg)
obs, _ = env.reset()
assert isinstance(obs, dict), f"obs should be a dict, got {type(obs)}"
with torch.inference_mode():
    for i in range(20):
        actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        obs, rew, terminated, truncated, _ = env.step(actions)
# print BEFORE close: kit swallows buffered stdout on shutdown
print("SMOKE TEST OK: Isaac-Cartpole-v0 stepped 20x with 16 envs", flush=True)
env.close()
simulation_app.close()
EOF
fi

cat <<EOF

[setup_isaacsim_env] 完成。把下面几行写入你的 shell rc（如 ~/.bashrc）：

export UNISIM_ISAACSIM_HOME="$ISAACSIM_HOME"
export OMNI_KIT_ACCEPT_EULA=1
# 交互调试时激活 venv：
source "\$UNISIM_ISAACSIM_HOME/venv/bin/activate"

# Legacy UniLab variable (optional for older launch scripts):
# export UNILAB_ISAACSIM_HOME="$ISAACSIM_HOME"

  venv:     $VENV_DIR
  IsaacLab: $ISAACLAB_DIR
  log:      $LOG_FILE

重跑本脚本是安全的：已完成的步骤会被跳过；加 --verify-sim 可重跑
headless 冒烟测试（Isaac-Cartpole-v0, 16 envs x 20 steps）。
EOF
