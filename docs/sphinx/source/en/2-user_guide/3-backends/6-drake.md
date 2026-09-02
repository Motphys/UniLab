# Drake Backend

Drake is an experimental CPU batch backend. UniLab still owns the task,
reward, observations, and training loop. Rendering uses MuJoCo's native
renderer: Drake advances physics and MuJoCo only draws the current state. The
supported native setup is Linux
x86_64 and Apple Silicon macOS (arm64); Intel macOS has no official Drake
binary.

## Prerequisites

- Python `>=3.10,<3.14` and `uv`.
- Linux: a C++20 toolchain and development packages:

  ```bash
  sudo apt-get update
  sudo apt-get install -y build-essential pkg-config libeigen3-dev libfmt-dev \
    libspdlog-dev curl git
  ```

- Apple Silicon macOS: Apple Clang and Homebrew runtime libraries:

  ```bash
  xcode-select --install  # if clang++ is not available
  brew install fmt gcc
  ```

  `gcc` supplies the `libgfortran` runtime referenced by Drake's macOS build.

## Install

From the UniLab checkout, run:

```bash
make setup-drake
```

The target downloads the host-appropriate Drake 1.56.0 tarball, installs the
`drake-uni` extra, builds the native extension with the active uv Python, and
runs a batch diagnostic. It is resumable; files and logs are kept in
`~/.unilab/drake`.

To use an existing Drake installation instead:

```bash
make setup-drake DRAKE_HOME=/path/to/drake
```

The prefix must contain `include/drake/`, `include/pybind11/`, and
`lib/libdrake.so`. On macOS, the setup script also discovers Homebrew `fmt`
and adds the required `gcc` library directory to `DYLD_LIBRARY_PATH`.

The setup script prints the environment exports needed by later shells. On
macOS they are normally:

```bash
export DRAKE_HOME="$HOME/.unilab/drake/drake-1.56.0-mac-arm64"
export UNILAB_DRAKE_HOME="$DRAKE_HOME"
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/gcc/lib/gcc/current:$DRAKE_HOME/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

## Verify

Run this in a fresh process (before importing `pydrake`):

```bash
uv run --no-sync python - <<'PY'
from drake_uni.runtime import batch_diagnostics

diagnostics = batch_diagnostics()
print(diagnostics)
if not diagnostics.batch_available:
    raise SystemExit(diagnostics.batch_import_error or "Drake batch extension is unavailable")
PY
```

The command must report `batch_available=True`.

## Train Go2

Go2 assets are downloaded on demand. Pull them once before the first run:

```bash
uv run --no-sync unilab-pull-assets --robot go2
```

Select Drake with `--sim drake`; do not override `training.sim_backend` by hand.
The normal PPO command is:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim drake
```

This uses the Drake owner configuration (`1024` environments, `151` iterations,
and CPU training because Drake exposes float64 NumPy buffers). The Drake owner
also uses the scene keyframe reset; floating-root randomization is not exposed
by the current backend contract.

On Apple Silicon macOS, this command completed all 151 iterations locally
(Drake 1.56.0, Python 3.13, 1024 environments) in about 254 seconds.
For an installation probe, temporarily add
`algo.max_iterations=1 algo.num_envs=4 algo.num_steps_per_env=4
training.no_play=true env.drake_nthread=1`; these overrides are not production
settings.

Drake has no separate renderer. Automatic recording and the interactive viewer
use MuJoCo while Drake remains the only physics engine being stepped. Use
`--render-mode none` for headless evaluation; recording and interactive
playback require the MuJoCo extra and visual assets. MuJoCo is not stepped or
used to re-evaluate the checkpoint.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: drake_uni` | Rerun `make setup-drake`. |
| `DrakeEnvPool batch extension has not been built` | Rebuild with the same uv Python and Drake prefix; do not copy an extension from another ABI. |
| `libdrake.so` or `libgfortran` cannot be loaded | Export `DRAKE_HOME` and `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS), including the Homebrew gcc directory. |
| Eigen/fmt link errors | Linux: install the packages above. macOS: run `brew install fmt gcc`. |
