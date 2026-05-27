# 快速开始

语言: 简体中文

本页给出首次阅读的最短路径：先安装，再完成第一次运行，然后进入训练主题。

## 主线顺序

1. 先看 [安装与环境](A-getting-started/01-install.md)
2. 再看 [首次运行](A-getting-started/02-first-run.md)
3. 需要容器时看 [Docker](A-getting-started/03-docker.md)
4. 跑起来以后转到 [训练指南](03-training.md)

## 平台选择

以顶层 `README.md` 为准，平台路径分成三类：

| 平台 | 安装命令 | 运行说明 |
|------|----------|----------|
| Linux CUDA / macOS | `make setup-motrix` | 常规使用 `uv run ...`；底层同步命令是 `uv sync --extra motrix` |
| Linux AMD / ROCm | `make sync-rocm` | 后续命令使用 `uv run ...` |
| Linux Intel Arc / iGPU | `make sync-xpu` | 后续命令使用 `uv run --no-sync ...` |

## 第一条命令

统一 CLI 是默认用户入口：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

回放最近一次训练结果：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1
```

本地 demo：

```bash
uv run demo
```

## 去哪里找细节

- 安装命令、国内镜像、ROCm / XPU 注意事项：见 [安装与环境](A-getting-started/01-install.md)
- conda / pip 用户的当前支持边界：见 [安装与环境](A-getting-started/01-install.md#conda--pip-用户说明)
- `train` / `eval` / `demo` 细节：见 [训练指南](03-training.md)
- backend 差异：见 [仿真后端](02-simulation-backends.md)
- 精确支持状态：见 [后端支持矩阵](E-reference/01-backend-support-matrix.md)
- 任务专属命令：见 [任务索引](D-tasks/01-task-index.md)

## Navigation

- Index: [Documentation](../../index.md)
- Next: [Install](A-getting-started/01-install.md)
