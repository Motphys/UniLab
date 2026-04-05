# Training Guide

本页覆盖训练、回放、恢复训练、Hydra 覆盖参数和 W&B。

## Pick An Entrypoint

| 目标 | 入口脚本 | 默认日志根目录 |
|------|----------|---------------|
| PPO (RSL-RL / torch) | `scripts/train_rsl_rl.py` | `logs/rsl_rl_train/<task>/` |
| PPO (MLX / macOS) | `scripts/train_mlx_ppo.py` | `logs/mlx_rl_train/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/appo/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/fast_sac/<task>/` / `logs/fast_td3/<task>/` |

## Start Training

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py task=go1_joystick

# PPO (MLX, Apple Silicon)
uv run python scripts/train_mlx_ppo.py task=go1_joystick

# APPO
uv run python scripts/train_appo.py task=go1_joystick

# Off-policy
uv run python scripts/train_offpolicy.py algo=sac task=go1_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# CLI 覆盖
uv run python scripts/train_offpolicy.py algo=sac task=g1_sac algo.num_envs=2048 algo.max_iterations=1000
```

默认情况下，训练脚本完成后会进入自动回放阶段。

- `mujoco` 会导出 `play_video.mp4`
- `motrix` 使用交互式窗口渲染
- `training.no_play=true` 可以跳过自动回放

训练日志目录命名为 `YYYY-MM-DD_HH-MM-SS_<sim_backend>`，例如 `2026-03-09_18-30-00_mujoco`。

## Playback

```bash
# 回放最新结果
uv run python scripts/train_rsl_rl.py task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true

# 回放指定 run
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

## Resume Training

```bash
uv run python scripts/train_rsl_rl.py task=go2_joystick training.load_run="2024-02-04_12-00-00"
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.load_run="2024-02-04_12-00-00"
```

## Hydra Overrides

所有训练脚本都基于 Hydra 配置。

```bash
# 通用格式
uv run python scripts/train_*.py [config_group=value] [key.subkey=value]

# 常见参数
task=go1_joystick
algo=sac
training.play_only=true
training.no_play=true
training.load_run="-1"
training.logger=tensorboard
algo.num_envs=2048
algo.max_iterations=1000
```

查看完整配置：

```bash
uv run python scripts/train_offpolicy.py --cfg job
```

## W&B

将 `training.logger=wandb` 打开后，训练脚本会自动记录到 Weights & Biases，并在本地 run 目录中写出：

- `run_config.json`
- `run_summary.json`

如果是 `mujoco` 并且训练后生成了 `play_video.mp4`，也会一并上传到当前 W&B run。

```bash
# 基本用法
uv run python scripts/train_rsl_rl.py task=go1_joystick training.logger=wandb

# 指定共享 project / entity
uv run python scripts/train_appo.py \
  task=go1_joystick \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_entity=my-team

# 按 task 分组
uv run python scripts/train_offpolicy.py \
  algo=sac \
  task=go2_joystick \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_group=go2_joystick
```

常用字段：

- `training.wandb_project`
- `training.wandb_entity`
- `training.wandb_group`
- `training.wandb_name`
- `training.wandb_tags`
- `training.wandb_notes`
- `training.wandb_mode=offline`

默认自动记录的信息包括任务、算法、backend、设备、硬件信息、git 信息、完整配置、整体运行时间、最终摘要指标，以及最终回放视频。

## Navigation

- Previous: [Simulation Backends](02-simulation-backends.md)
- Next: [Algorithms](04-algorithms.md)
