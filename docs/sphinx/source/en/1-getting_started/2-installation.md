# Installation

This page covers dependency setup only. Training commands and playback details
live in the getting-started and algorithm pages.

## Requirements

- Python `>=3.10,<3.14`, from `pyproject.toml`.
- `uv`, used for dependency sync and command execution.
- `cmake`, required by the local setup documented in
  `docs/sphinx/source/zh_CN/1-getting_started/2-installation.md`.

## Clone And Sync

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
```

```bash
brew install cmake
# Ubuntu / Debian:
# sudo apt-get install cmake
```

Choose one sync path:

```bash
make setup
make setup-motrix
```

`make setup` runs `uv sync` and installs shell completion. `make setup-motrix`
runs `uv sync --extra motrix` and installs the same completion entry. If `make`
is unavailable, run the underlying sync directly:

```bash
uv sync
uv sync --extra motrix
```

## Platform Profiles

Linux CUDA and macOS use the default `pyproject.toml`. The default Linux torch
wheel source is the PyTorch `cu128` index configured in `pyproject.toml`.

ROCm and Intel XPU have explicit Makefile targets:

```bash
make sync-rocm
make sync-xpu
```

`make sync-rocm` copies `pyproject.rocm.toml` into `pyproject.toml` and syncs the
ROCm profile. `make sync-xpu` syncs Motrix dependencies without installing the
default torch package, then installs the XPU torch wheel through `uv pip`.

## Package Mirrors

For a local package mirror, set the uv index before syncing:

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Smoke Check

After sync, run a small check through the top-level CLI:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

For Motrix, install the extra first and switch with `--sim`:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

Do not use the `training.sim_backend` field by itself to switch backends; choose
the backend with `--sim`.
