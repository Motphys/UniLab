# Docker

语言: 简体中文

本页说明容器运行方式和适用范围。常规本地安装优先看 [安装与环境](01-install.md)。

## 适用范围

- 根目录 `Dockerfile` 面向 Linux NVIDIA / CUDA
- macOS Docker 目前不作为主要路径
- ROCm 容器不复用仓库根目录 `Dockerfile`

## 构建镜像

```bash
docker build -t unilab:latest .
```

快速检查镜像入口：

```bash
docker run --rm unilab:latest
```

## 启动训练容器

```bash
docker run --rm --gpus all -it \
  -v "$(pwd):/workspace/UniLab" \
  -v unilab-venv:/workspace/UniLab/.venv \
  -w /workspace/UniLab \
  unilab:latest bash
```

把 `.venv` 放进 named volume 的作用是避免容器内虚拟环境覆盖宿主机仓库目录，回到本地 `uv` 工作流时也更干净。

## 容器内命令

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
```

快速检查容器内 CUDA：

```bash
docker run --rm --gpus all unilab:latest uv run python -c "import torch; print(torch.cuda.is_available())"
```

## ROCm 说明

- ROCm 请使用 AMD 官方 `rocm/pytorch` 镜像
- 进入容器后先运行 `make sync-rocm` 激活 ROCm profile，训练命令使用 `uv run ...`
- 设备挂载通常至少包括 `/dev/kfd`、`/dev/dri`、`--group-add=video` 和 `--ipc=host`
- 仓库根目录 `Dockerfile` 保持 NVIDIA / CUDA 路径，不直接承担 ROCm 容器职责

最小 ROCm 容器示意：

```bash
docker run --rm -it --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri/renderD128 --group-add=video \
  rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.7.1
```

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [首次运行](02-first-run.md)
