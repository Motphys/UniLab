![G1 motion tracking overview](docs/assets/g1_readme.png)

# UniLab

Languages: English | [简体中文](docs/users/zh_CN/01-getting-started.md)

Train robot RL without a GPU simulation backend.

UniLab uses **CPU simulation + shared-memory runtime + GPU learning** instead of coupling simulation and learning inside one GPU-resident pipeline.

Start with the `Quick Demo` below to run the primary training command from this repository.

## 🚀 Quick Demo

```bash
# 0. If uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. Install dependencies
uv sync --extra motrix

# 3. Run a first PPO training job
# macOS: 73s on M5Max-128GB, 1min43s on M3Max-48GB, 2.5min on MacBookNeo-8GB
# Linux: 31s on RTX 4090 and R9-9950x3d
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

This is the first-level training entrypoint. It routes to the registered `go2_joystick_flat/motrix` task owner config and keeps backend selection in the CLI flags.

For evaluation and demo playback:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# Demo playback from a local trained checkpoint
uv run demo
```

On macOS / MacBook, the UniLab CLI routes Motrix renderer playback through `mxpython` when needed. Detailed script-level commands are documented under `docs/users/zh_CN/`.

### Interactive Notebooks

Prefer a guided, step-by-step experience? Open the notebooks in Jupyter:

- [Demo Notebook](notebook/demo.ipynb): local checkpoint playback via `uv run demo`
- [PPO Training Walkthrough](notebook/unilab_walkthrough_ppo_go1_joystick_mujoco.ipynb): end-to-end guide from config preview to training to playback, with explanations for beginners

> Notebooks require a local environment (no Colab support) — MuJoCo needs local compute.

## 🏃 Example Runs

These are example repository runs for documented commands and hardware setups. They are useful as concrete entrypoints and reported timings, but they are **not** yet a formal benchmark manifest.

```bash
# Linux SAC (5.5min on RTX 4090 and R9-9950x3d)
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

```bash
# Linux G1 motion tracking (1h35min on RTX 4090 and R9-9950x3d)
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
```

## 🧱 System Layout

```
┌───────────────────┐     Unified Shared Memory     ┌────────────────────┐
│  CPU Physics Sim  │ ───────────────────────────▶  │ GPU Policy Training│
│  mujoco.rollout   │      SharedReplayBuffer       │   PPO / SAC / TD3  │
│ Multithread Step  │    (PyTorch shared tensors)   │     CUDA / MPS     │
└───────────────────┘                               └────────────────────┘
```

## 🎯 Training Entrypoints

Use `uv run train` for training, `uv run eval` for checkpoint playback, and `uv run demo` for the local demo preset. These commands are the first-level training interface and keep algorithm, task, and backend selection explicit.

See [03 Training Guide](docs/users/zh_CN/03-training.md) for the algorithm matrix, log directory layout, Hydra overrides, script-level entrypoints, and demo flags.

## 🐳 Docker

Use Docker when you want an isolated **Linux NVIDIA/CUDA** runtime without changing your local Python environment. The Dockerfile is based on an NVIDIA CUDA Ubuntu image, so Linux training containers require an NVIDIA GPU, a compatible host driver, and NVIDIA Container Toolkit. macOS Docker is not currently supported.

### Build Image

```bash
docker build -t unilab:latest .
```

The image installs the default UniLab runtime, `mujoco-uni`, the `motrix` extra, and the dev/test tools.

### Run Image

```bash
# Check the unified CLI entrypoint
docker run --rm unilab:latest

# Linux NVIDIA training shell with the repo mounted
# Keep .venv in a named volume so the container does not overwrite the host venv
docker run --rm --gpus all -it \
  -v "$(pwd):/workspace/UniLab" \
  -v unilab-venv:/workspace/UniLab/.venv \
  -w /workspace/UniLab \
  unilab:latest bash
```

Using a named volume for `/workspace/UniLab/.venv` avoids mixing a container-created virtual environment with the host checkout. This prevents path mismatches such as `/workspace/UniLab/.venv` vs `.venv` and avoids permission problems when returning to local `uv` or `make test-all` workflows.

## 📚 Documentation

Use [docs/README.md](docs/README.md) as the documentation index. High-signal entrypoints:

- [Getting Started](docs/users/zh_CN/01-getting-started.md): installation, dependency setup, and first-run commands
- [Training Guide](docs/users/zh_CN/03-training.md): training, playback, resume flow, Hydra overrides, and W&B
- [Simulation Backends](docs/users/zh_CN/02-simulation-backends.md): generated MuJoCo / Motrix support matrix
- [Development Standard](docs/developers/zh_CN/development-standard.md): contracts, layering, and validation boundaries
- [ADR Index](docs/developers/adr/README.md): accepted architecture decisions
