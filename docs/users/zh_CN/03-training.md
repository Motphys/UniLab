# 训练指南

语言: 简体中文

本页覆盖训练、回放、恢复训练、Hydra override 和 W&B。

## 统一 CLI 入口

`uv run train`、`uv run eval` 和 `uv run demo` 是 UniLab 的第一等级训练入口。CLI 通过 `--algo`、`--task`、`--sim` 三个参数自动路由到对应训练脚本，同时保持算法、任务和后端选择显式可见。

### uv run train

| 目标 | 主命令 | 日志根目录模式 |
|------|--------|----------------|
| PPO (RSL-RL / torch) | `uv run train --algo ppo --task <task> --sim <backend>` | `logs/rsl_rl_ppo/<task>/` |
| PPO (MLX / macOS) | `uv run train --algo mlx_ppo --task <task> --sim <backend>` | `logs/mlx_rl_train/<task>/` |
| APPO | `uv run train --algo appo --task <task> --sim <backend>` | `logs/appo/<task>/` |
| SAC | `uv run train --algo sac --task <task> --sim <backend>` | `logs/fast_sac/<task>/` |
| FlashSAC | `uv run train --algo flashsac --task <task> --sim <backend>` | `logs/flash_sac/<task>/` |
| TD3 | `uv run train --algo td3 --task <task> --sim <backend>` | `logs/fast_td3/<task>/` |

常用命令：

```bash
# PPO
uv run train --algo ppo --task go2_joystick_flat --sim mujoco

# APPO
uv run train --algo appo --task go2_joystick_flat --sim mujoco

# SAC
uv run train --algo sac --task g1_walk_flat --sim mujoco

# TD3
uv run train --algo td3 --task g1_walk_flat --sim mujoco

# FlashSAC
uv run train --algo flashsac --task g1_walk_flat --sim mujoco

# MLX PPO (macOS only)
uv run train --algo mlx_ppo --task go2_joystick_flat --sim mujoco
```

支持的算法：`ppo`、`mlx_ppo`、`appo`、`sac`、`td3`、`flashsac`
支持的模拟器：`mujoco`、`motrix`

训练命令默认会在训练结束后自动进入回放。

- `mujoco` 会导出 `play_video.mp4`
- `motrix` 会打开交互式窗口渲染
- `training.no_play=true` 可以跳过自动回放

Hydra override 可以直接追加在命令末尾：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco training.max_iterations=10
```

run 目录命名格式是 `YYYY-MM-DD_HH-MM-SS_<sim_backend>`，例如 `2026-03-09_18-30-00_mujoco`。

### uv run eval

评估模式自动设置 `training.play_only=true`，并通过 `--load-run` 指定 checkpoint：

```bash
# 回放最新 run
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

# 回放指定 run
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run 2026-04-24_01-36-01_mujoco
```

### uv run demo

运行本地训练产出的 checkpoint demo，无需手动指定 task 和 checkpoint：

```bash
# 默认 preset（go2_joystick_mujoco_ppo）
uv run demo

# 指定 preset
uv run demo --preset go2_joystick_mujoco_ppo

# 重新生成 demo 目录
uv run demo --refresh

# 指定推理设备
uv run demo --device cpu
```

| 参数 | 说明 |
|------|------|
| `--preset` | demo 预设名称（默认 `go2_joystick_mujoco_ppo`） |
| `--refresh` | 删除已有 demo 目录并重新生成 |
| `--device` | 推理设备（`cpu`、`cuda`、`mps`） |

> **For Developers**: 当前 demo checkpoint 需要本地训练产出后手动放置。尚缺少 checkpoint 网络托管方案（如 CDN / model registry），后续需要补充自动下载机制。

## 低层脚本入口

直接脚本入口主要用于调试、验证单个训练栈，或需要访问底层 Hydra compose 细节的场景。常规训练优先使用上面的统一 CLI。

| 目标 | 入口脚本 | 日志根目录模式 |
|------|----------|----------------|
| PPO (RSL-RL / torch) | `scripts/train_rsl_rl.py` | `logs/<algo.algo_log_name>/<task>/` |
| PPO (MLX / macOS) | `scripts/train_mlx_ppo.py` | `logs/<algo.algo_log_name>/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/<algo.algo_log_name>/<task>/` |
| SAC / TD3 / FlashSAC | `scripts/train_offpolicy.py` | `logs/<algo.algo_log_name>/<task>/` |

实际目录名由 `algo.algo_log_name` 决定；当前默认值分别是 `rsl_rl_ppo`、`mlx_rl_train`、`appo`、`fast_sac`、`flash_sac` 和 `fast_td3`。

```bash
# PPO (RSL-RL)
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco

# PPO (MLX, Apple Silicon)
uv run scripts/train_mlx_ppo.py task=go1_joystick_flat/mujoco

# APPO
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco

# Off-policy
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco

# Hydra override
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco algo.num_envs=2048 algo.max_iterations=1000
```

在 macOS / MacBook 上，统一 CLI 会在需要打开 MotrixSim 原生 renderer 时自动路由到 `mxpython`。如果直接调用脚本，只要命令会打开 MotrixSim 原生 renderer（训练后自动回放或 `training.play_only=true`），就需要用 `uv run mxpython` 启动；不需要可视化的训练仍可使用 `uv run scripts/... training.no_play=true`。

## 脚本回放

常规 checkpoint 回放优先使用 `uv run eval`。下面是直接脚本形式，主要用于调试底层训练入口。

```bash
# 回放最新结果
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco training.play_only=true
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco training.play_only=true

# macOS / MacBook 上的 MotrixSim 原生 renderer
uv run mxpython scripts/train_rsl_rl.py task=go2_joystick_flat/motrix training.play_only=true

# 回放指定 run
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco training.play_only=true algo.load_run="2024-02-04_12-00-00"
```

## 恢复训练

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco algo.load_run="2024-02-04_12-00-00"
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco algo.load_run="2024-02-04_12-00-00"
```

## Hydra Overrides

统一 CLI 和低层训练脚本都由 Hydra 配置驱动。统一 CLI 可以直接在命令末尾追加 Hydra override：

```bash
# 统一 CLI
uv run train --algo ppo --task go2_joystick_flat --sim mujoco training.max_iterations=10 training.no_play=true

# 低层脚本通用形式
uv run scripts/train_*.py [config_group=value] [key.subkey=value]
```

低层脚本常见 config group：

```bash
task=go1_joystick_flat/mujoco
algo=sac
```

常见 Hydra override：

```bash
training.play_only=true
training.no_play=true
algo.load_run="-1"
training.logger=tensorboard
algo.num_envs=2048
algo.max_iterations=1000
```

`task` 是后端选择入口，例如 `task=go1_joystick_flat/motrix`。`training.sim_backend` 由对应的 task owner YAML 设置，只用于标识最终后端；不要用 `training.sim_backend=motrix` 单独切换后端。

查看完整合成配置:

```bash
uv run scripts/train_offpolicy.py --cfg job
```

## W&B

设置 `training.logger=wandb` 后，会自动记录到 Weights & Biases。训练脚本也会在本地 run 目录里写出:

- `run_config.json`
- `run_summary.json`

如果 backend 是 `mujoco` 且训练生成了 `play_video.mp4`，该视频也会上传到当前 W&B run。

```bash
# 基本用法
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco training.logger=wandb

# 共享 project / entity
uv run scripts/train_appo.py \
  task=go1_joystick_flat/mujoco \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_entity=my-team

# 按 task 分组
uv run scripts/train_offpolicy.py \
  algo=sac \
  task=sac/g1_walk_flat/mujoco \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_group=g1_walk_flat
```

常用字段:

- `training.wandb_project`
- `training.wandb_entity`
- `training.wandb_group`
- `training.wandb_name`
- `training.wandb_tags`
- `training.wandb_notes`
- `training.wandb_mode=offline`

自动记录的元数据包括 task、algorithm、backend、device、硬件信息、git 信息、完整配置、总运行时，以及可用时的最终回放视频。

## Docker 中运行训练

当你希望在隔离的 Linux NVIDIA/CUDA 环境中运行训练时，可以直接在容器里执行统一 CLI 或脚本入口。Linux 训练容器需要 NVIDIA 显卡、兼容的宿主机驱动，以及 NVIDIA Container Toolkit。推荐先在仓库根目录构建 image：

```bash
docker build -t unilab:latest .
```

该 image 默认安装 UniLab 运行依赖、`mujoco-uni`、`motrix` extra，以及 dev/test 工具。

最简单的容器训练方式是挂载当前仓库，并通过 `--gpus all` 将宿主机 NVIDIA GPU 暴露给容器：

```bash
docker run --rm --gpus all -it \
  -v "$(pwd):/workspace/UniLab" \
  -v unilab-venv:/workspace/UniLab/.venv \
  -w /workspace/UniLab \
  unilab:latest bash
```

这里延续 [快速开始](01-getting-started.md) 中的做法：将 `.venv` 单独放到 named volume，避免容器内创建的虚拟环境污染宿主机仓库目录。

进入容器后，可以直接执行：

```bash
# 统一 CLI
uv run train --algo ppo --task go2_joystick_flat --sim mujoco

# 直接脚本入口
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

macOS Docker 目前不支持。

Linux AMD / ROCm 容器不使用仓库根目录的 CUDA `Dockerfile`。请使用 ROCm >= 7.1 的 AMD 官方 `rocm/pytorch` 镜像，并按 ROCm Docker 要求挂载 `/dev/kfd`、`/dev/dri`、`--group-add=video`、`--ipc=host` 等设备与 IPC 参数。进入 ROCm 容器后可以运行 `make sync-rocm`，训练命令使用 `uv run --no-sync ...`。

训练日志和产物仍会写入挂载后的仓库目录，例如 `logs/rsl_rl_ppo/<task>/`、`logs/fast_sac/<task>/`。如果只想快速验证 image 是否可用，可以直接运行：

```bash
docker run --rm unilab:latest
```

## Navigation

- Index: [Documentation](../../README.md)
- Previous: [Simulation Backends](02-simulation-backends.md)
- Next: [Algorithms](04-algorithms.md)
