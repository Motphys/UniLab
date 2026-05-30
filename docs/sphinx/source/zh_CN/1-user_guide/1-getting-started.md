# 快速开始

语言: 简体中文

本页给出首次阅读的最短路径：先安装，再完成第一次运行，然后进入训练主题。

## 主线顺序

1. 先看 [安装与环境](1-getting-started/1-install.md)
2. 再看 [首次运行](1-getting-started/2-first-run.md)
3. 需要容器时看 [Docker](1-getting-started/3-docker.md)
4. 跑起来以后转到 [训练指南](3-training.md)

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

本地 demo（首次运行会从 Hugging Face 自动拉取预训练 checkpoint）：

```bash
uv run demo dance
```

可用 demo 名：`teaser`、`dance`、`wallflip`、`boxtracking`、`locomani`、`inhandgrasp`。

> 面向大陆用户：motion、scene 和 demo checkpoint 默认从 `huggingface.co` 拉取。如果官方端点不可达，可在运行 `train` / `eval` / `demo` 前切换到社区镜像：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## 去哪里找细节

- 安装命令、国内镜像、ROCm / XPU 注意事项：见 [安装与环境](1-getting-started/1-install.md)
- conda / pip 用户的当前支持边界：见 [安装与环境](1-getting-started/1-install.md#conda--pip-用户说明)
- `train` / `eval` / `demo` 细节：见 [训练指南](3-training.md)
- backend 差异：见 [仿真后端](2-simulation-backends.md)
- 精确支持状态：见 [后端支持矩阵](5-reference/1-backend-support-matrix.md)
- 任务专属命令：见 [任务索引](4-tasks/1-task-index.md)

## Navigation

- Index: [Documentation](../0-index.md)
- Next: [Install](1-getting-started/1-install.md)
