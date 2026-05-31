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

## Conda And Pip

The recommended path is still the in-repo `make setup` / `make setup-motrix` (or
`uv`) workflow. Conda can serve as an outer environment for Python, CUDA, or
system-library isolation, but once the environment is active keep using the
repository's `make` / `uv` commands inside it:

```bash
conda create -n unilab python=3.13
conda activate unilab
pip install uv
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
make setup-motrix
```

Use `make setup` if you do not need Motrix. ROCm and XPU still go through the
platform-specific `make` targets below.

`pip install -e .` and `pip install .` are only for dev verification inside a
source checkout; they do not yet support running training from an arbitrary
directory via a built wheel/sdist. The training entrypoints still depend on the
repository's `conf/` and `scripts/`. The pip-only / out-of-repo / published-wheel
validation path is tracked by issue #360.

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

ROCm notes:

- `make sync-rocm` requires ROCm `>= 7.1` and installs the matching PyTorch wheel
  from the repository's ROCm dependency files.
- It swaps `pyproject.rocm.toml` / `uv.rocm.lock` in as the active
  `pyproject.toml` / `uv.lock`, so afterwards you can run bare `uv run ...`.
- To return to the default CUDA / macOS profile, run
  `git restore -- pyproject.toml uv.lock` and then re-run `make setup-motrix`
  (or `uv sync --extra motrix`); confirm the active profile before committing any
  non-ROCm dependency change.
- The training device field keeps `cuda` semantics; do not set it to `rocm`.

Intel XPU notes:

- Keep using `uv run --no-sync ...` so the default Linux dependencies are not
  synced back in.
- Ubuntu 24.04+ also needs the system driver packages `intel-opencl-icd` and
  `libze-intel-gpu1`.
- Off-policy training can add `training.use_amp=true` as needed.

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
