# 快速演示

语言: 简体中文

本页对应 README 中的快速演示（Quick Demo），使用一级 UniLab CLI：
`uv run train`、`uv run eval` 和 `uv run demo`。

## 克隆与同步

```bash
# 0. 如果尚未安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. 克隆仓库
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
```

为你的平台选择且仅选择一条依赖安装命令：

```bash
# Linux CUDA 或 macOS
make setup-motrix

# 不安装 shell 自动补全：
# uv sync --extra motrix

# 如果未安装 make：
# uv sync --extra motrix && uv run --no-sync unilab-complete install

# Linux AMD / ROCm
# make sync-rocm

# Linux Intel Arc / iGPU
# make sync-xpu
```

## 训练

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

该命令会路由到已注册的 `go2_joystick_flat` 任务，并使用 Motrix 后端。CLI 通过
`--algo`、`--task` 和 `--sim` 让算法、任务和后端的选择保持显式；其内部会组合（compose）出匹配的 owner YAML。

## 评估或演示

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# 面向 Linux/服务器运行的 Motrix 无头（headless）视频导出
uv run eval --algo ppo --task go2_joystick_flat --sim motrix \
  --load-run -1 --render-mode record

# 演示回放（首次运行会从 Hugging Face 拉取预训练检查点）
uv run demo dance
```

可用的 demo 名称：`dance`、`wallflip`、`boxtracking`、`locomani`、`inhandgrasp`。

在 macOS 上，CLI 会在需要时通过 `mxpython` 路由 Motrix 的交互式回放。使用
`--render-mode record` 进行无头视频导出，或使用
`--render-mode none` 跳过渲染。

## 简短冒烟变体

对于 CI 风格的本地检查，保持相同的 CLI 路由，并在各标志之后追加 Hydra 覆盖项（override）：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

## 后续步骤

- 平台细节：{doc}`2-installation`。
- 回放细节：{doc}`3-evaluation_and_playback`。
- 在 {doc}`../2-user_guide/1-training/1-cli_reference` 中了解统一 CLI。
- 在 {doc}`../2-user_guide/1-training/2-hydra_config` 中阅读 Hydra owner YAML 的工作方式。

## Navigation

- Index: [文档](0-index.md)
