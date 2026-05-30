# Docker

语言: 简体中文

仓库内置的 `Dockerfile` 是 Linux NVIDIA/CUDA 容器路径。它会安装 UniLab 运行时依
赖、Motrix extra 以及 dev/test 工具。

## 构建

```bash
docker build -t unilab:latest .
```

## 挂载本地 checkout 运行

```bash
docker run --rm --gpus all -it \
  -v "$(pwd):/workspace/UniLab" \
  -v unilab-venv:/workspace/UniLab/.venv \
  -w /workspace/UniLab \
  unilab:latest bash
```

在容器内部，使用与宿主机工作流相同的命令：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

ROCm 容器应使用 AMD 的 ROCm PyTorch 镜像以及 `make sync-rocm` 工作流，而不是仓库
的 CUDA `Dockerfile`。

## Navigation

- Index: [文档](0-index.md)
