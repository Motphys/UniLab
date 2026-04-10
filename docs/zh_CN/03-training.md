# 训练指南

语言: 简体中文

本页覆盖训练、回放、恢复训练、Hydra override 和 W&B。

## Pick An Entrypoint

| 目标 | 入口脚本 | 日志根目录模式 |
|------|----------|----------------|
| PPO (RSL-RL / torch) | `scripts/train_rsl_rl.py` | `logs/<algo.algo_log_name>/<task>/` |
| PPO (MLX / macOS) | `scripts/train_mlx_ppo.py` | `logs/<algo.algo_log_name>/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/<algo.algo_log_name>/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/<algo.algo_log_name>/<task>/` |

实际目录名由 `algo.algo_log_name` 决定；当前默认值分别是 `rsl_rl_ppo`、`appo`、`fast_sac` 和 `fast_td3`。

## Start Training

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py task=go1_joystick/mujoco

# PPO (MLX, Apple Silicon)
uv run python scripts/train_mlx_ppo.py task=go1_joystick/mujoco

# APPO
uv run python scripts/train_appo.py task=go1_joystick/mujoco

# Off-policy
uv run python scripts/train_offpolicy.py algo=sac task=sac/go1_joystick/mujoco
uv run python scripts/train_offpolicy.py algo=td3 task=td3/go1_joystick/mujoco

# CLI override
uv run python scripts/train_offpolicy.py algo=sac task=sac/g1_sac/mujoco algo.num_envs=2048 algo.max_iterations=1000
```

训练脚本默认会在训练结束后自动进入回放。

- `mujoco` 会导出 `play_video.mp4`
- `motrix` 会打开交互式窗口渲染
- `training.no_play=true` 可以跳过自动回放

run 目录命名格式是 `YYYY-MM-DD_HH-MM-SS_<sim_backend>`，例如 `2026-03-09_18-30-00_mujoco`。

## Playback

```bash
# 回放最新结果
uv run python scripts/train_rsl_rl.py task=go2_joystick/mujoco training.play_only=true
uv run python scripts/train_offpolicy.py algo=sac task=sac/go2_joystick/mujoco training.play_only=true

# 回放指定 run
uv run python scripts/train_offpolicy.py algo=td3 task=td3/go1_joystick/mujoco training.play_only=true algo.load_run="2024-02-04_12-00-00"
```

## Resume Training

```bash
uv run python scripts/train_rsl_rl.py task=go2_joystick/mujoco algo.load_run="2024-02-04_12-00-00"
uv run python scripts/train_offpolicy.py algo=sac task=sac/go2_joystick/mujoco algo.load_run="2024-02-04_12-00-00"
```

## Hydra Overrides

所有训练脚本都由 Hydra 配置驱动。

```bash
# 通用形式
uv run python scripts/train_*.py [config_group=value] [key.subkey=value]

# 常见参数
task=go1_joystick/mujoco
algo=sac
training.play_only=true
training.no_play=true
algo.load_run="-1"
training.logger=tensorboard
algo.num_envs=2048
algo.max_iterations=1000
```

`task` 是后端选择入口，例如 `task=go1_joystick/motrix`。`training.sim_backend` 由对应的 task owner YAML 设置，只用于标识最终后端；不要用 `training.sim_backend=motrix` 单独切换后端。

查看完整合成配置:

```bash
uv run python scripts/train_offpolicy.py --cfg job
```

## W&B

设置 `training.logger=wandb` 后，会自动记录到 Weights & Biases。训练脚本也会在本地 run 目录里写出:

- `run_config.json`
- `run_summary.json`

如果 backend 是 `mujoco` 且训练生成了 `play_video.mp4`，该视频也会上传到当前 W&B run。

```bash
# 基本用法
uv run python scripts/train_rsl_rl.py task=go1_joystick/mujoco training.logger=wandb

# 共享 project / entity
uv run python scripts/train_appo.py \
  task=go1_joystick/mujoco \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_entity=my-team

# 按 task 分组
uv run python scripts/train_offpolicy.py \
  algo=sac \
  task=sac/go2_joystick/mujoco \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_group=go2_joystick
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

## Navigation

- Previous: [Simulation Backends](02-simulation-backends.md)
- Next: [Algorithms](04-algorithms.md)
