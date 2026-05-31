# Docker

仓库内置的 `Dockerfile` 是 Linux NVIDIA/CUDA 容器路径。它会安装 UniLab 运行时依
赖、Motrix extra 以及 dev/test 工具。macOS Docker 目前不作为主要路径，ROCm 则使用
另外的镜像（见下文），而不是这个 CUDA `Dockerfile`。

## 构建

```bash
docker build -t unilab:latest .
```

快速检查镜像入口（默认入口是 `uv run train --help`）：

```bash
docker run --rm unilab:latest
```

检查容器内 CUDA 是否可见：

```bash
docker run --rm --gpus all unilab:latest \
  uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 挂载本地 checkout 运行

```bash
docker run --rm --gpus all -it \
  -v "$(pwd):/workspace/UniLab" \
  -v unilab-venv:/workspace/UniLab/.venv \
  -w /workspace/UniLab \
  unilab:latest bash
```

把 `.venv` 放进 named volume 的作用是避免容器内的虚拟环境覆盖宿主机仓库目录，切回
本地 `uv` 工作流时也更干净。

在容器内部，使用与宿主机工作流相同的命令：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

容器内的评估和回放方式相同；`--load-run -1` 会选择最新的 run：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

由于 checkout 是 bind-mount 挂载进来的，训练产物仍会写回挂载后的仓库 `logs/`
目录。

## ROCm 容器

ROCm 容器应使用 AMD 的 ROCm PyTorch 镜像以及 `make sync-rocm` 工作流，而不是仓库
的 CUDA `Dockerfile`。进入容器后先运行 `make sync-rocm` 激活 ROCm profile，再用
`uv run ...` 训练。容器通常需要挂载 AMD 设备 `/dev/kfd` 和 `/dev/dri`，并加上
`--group-add=video` 和 `--ipc=host`。

使用 AMD ROCm PyTorch 镜像的最小示例：

```bash
docker run --rm -it --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  -v "$(pwd):/workspace/UniLab" -w /workspace/UniLab \
  rocm/pytorch:latest bash
```

请使用与本机 ROCm 版本匹配的 ROCm PyTorch 镜像 tag；可用 tag 参见 AMD 的 ROCm
PyTorch 镜像。
