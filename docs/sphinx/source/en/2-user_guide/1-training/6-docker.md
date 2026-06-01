# Docker

The checked-in `Dockerfile` is the Linux NVIDIA/CUDA container path. It installs
UniLab runtime dependencies, the Motrix extra, and dev/test tooling. macOS Docker
is not a primary path, and ROCm uses a separate image (see below) rather than
this CUDA `Dockerfile`.

## Build

```bash
docker build -t unilab:latest .
```

Quickly check the image entrypoint (it defaults to `uv run train --help`):

```bash
docker run --rm unilab:latest
```

Check CUDA visibility inside the container:

```bash
docker run --rm --gpus all unilab:latest \
  uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Run With A Mounted Checkout

```bash
docker run --rm --gpus all -it \
  -v "$(pwd):/workspace/UniLab" \
  -v unilab-venv:/workspace/UniLab/.venv \
  -w /workspace/UniLab \
  unilab:latest bash
```

Putting `.venv` in a named volume keeps the container's virtual environment from
overwriting the host repository directory, and leaves the host `uv` workflow
clean when you switch back to it.

Inside the container, use the same commands as the host workflow:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Evaluation and playback work the same way inside the container; `--load-run -1`
picks the latest run:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

Because the checkout is bind-mounted, training artifacts still write back to the
mounted repository's `logs/` directory.

## ROCm Containers

ROCm containers use AMD's ROCm PyTorch image and the `make sync-rocm` workflow
instead of the repository's CUDA `Dockerfile`. Inside the container, run
`make sync-rocm` first to activate the ROCm profile, then use `uv run ...` for
training. The container typically needs the AMD device mounts `/dev/kfd` and
`/dev/dri`, plus `--group-add=video` and `--ipc=host`.

A minimal example using AMD's ROCm PyTorch image:

```bash
docker run --rm -it --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  -v "$(pwd):/workspace/UniLab" -w /workspace/UniLab \
  rocm/pytorch:latest bash
```

Use the ROCm PyTorch image tag that matches your installed ROCm version; see
AMD's ROCm PyTorch images for the available tags.
