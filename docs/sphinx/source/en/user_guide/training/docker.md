# Docker

The checked-in `Dockerfile` is the Linux NVIDIA/CUDA container path. It installs
UniLab runtime dependencies, the Motrix extra, and dev/test tooling.

## Build

```bash
docker build -t unilab:latest .
```

## Run With A Mounted Checkout

```bash
docker run --rm --gpus all -it \
  -v "$(pwd):/workspace/UniLab" \
  -v unilab-venv:/workspace/UniLab/.venv \
  -w /workspace/UniLab \
  unilab:latest bash
```

Inside the container, use the same commands as the host workflow:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

ROCm containers should use AMD's ROCm PyTorch image and the `make sync-rocm`
workflow instead of the repository's CUDA `Dockerfile`.
