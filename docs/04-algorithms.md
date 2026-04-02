# Algorithms

本页只保留算法级说明，训练入口和通用参数见 [03-training.md](03-training.md)。

## APPO

APPO 是基于 V-trace 重要性采样修正的异步 PPO 实现。Collector 子进程负责 CPU 仿真，Learner 进程负责 GPU 训练，通过 ring buffer 并行运行。

### Core Features

| 特性 | 说明 |
|------|------|
| 异步多进程 | Collector 与 Learner 解耦并行 |
| V-trace IS 修正 | 用 `π_target / π_behavior` 修正 off-policy 数据 |
| 4 槽 ring buffer | 最多 4 条 rollout 在飞 |
| Replay queue | Learner 端维护 rollout 消费队列 |
| 日志目录 | `logs/appo/<task>/<timestamp>_mujoco/` |

### Usage

```bash
# 默认训练
uv run python scripts/train_appo.py task=go1_joystick

# 指定环境数量和迭代次数
uv run python scripts/train_appo.py task=go2_joystick algo.num_envs=2048 algo.max_iterations=300

# 调整 replay queue 深度
uv run python scripts/train_appo.py task=go1_joystick training.replay_queue_size=2

# 跳过训练后的自动回放
uv run python scripts/train_appo.py task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true training.load_run="2026-03-16_01-35-12_mujoco"
```

### Key Params

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task` | `go2_joystick` | 任务配置名称 |
| `algo.max_iterations` | 150 | 最大训练迭代次数 |
| `algo.num_envs` | 2048 | 并行环境数量 |
| `algo.steps_per_env` | 24 | 每条 rollout 的步数 |
| `training.replay_queue_size` | 3 | Learner 端 rollout 重放队列深度 |
| `training.device` | 自动检测 | Learner 设备 |
| `training.collector_device` | `cpu` | Collector 设备 |
| `training.logger` | `tensorboard` | 日志后端 |
| `training.play_only` | false | 仅回放 |
| `training.no_play` | false | 跳过自动回放 |
| `training.load_run` | `-1` | 指定 run 目录名或 checkpoint 路径 |
| `algo.save_interval` | 50 | checkpoint 保存间隔 |

### APPO vs PPO

| 维度 | rsl-rl PPO | APPO |
|------|-----------|------|
| 收集方式 | 同步 | 异步 |
| IS 修正 | 无 | V-trace |
| CPU/GPU 利用率 | 交替满载 | 同时满载 |
| 适用场景 | 样本效率优先 | 吞吐量优先 |

## FastSAC And FastTD3

FastSAC / FastTD3 基于异步多进程架构，使用共享内存解耦 CPU 仿真和 GPU 训练。

### Core Features

| 特性 | 说明 |
|------|------|
| 异步多进程 | Collector 与 Learner 独立运行 |
| 统一共享内存 | PyTorch shared tensors 零拷贝传输 |
| 同步 / 异步模式 | 同时支持默认同步收集和异步收集 |
| 自动 Play | 训练后自动回放 |

### Usage

```bash
# 基本训练
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# 异步模式
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.no_sync_collection=true

# 跳过自动回放
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

### Key Params

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algo` | `sac` | 算法选择 |
| `task` | `go1_joystick` | 任务配置名称 |
| `algo.max_iterations` | 500 (SAC) / 5000 (TD3) | 最大训练迭代次数 |
| `algo.num_envs` | 4096 | 并行环境数量 |
| `training.device` | 自动检测 | Learner 设备 |
| `training.sim_backend` | `mujoco` | 仿真后端 |
| `training.no_sync_collection` | false | 启用异步收集 |
| `training.env_steps_per_sync` | 1 | 同步模式下每次收集步数 |
| `training.play_only` | false | 仅回放 |
| `training.no_play` | false | 跳过自动回放 |

## Navigation

- Previous: [Training Guide](03-training.md)
- Next: [G1 Motion Tracking](05-g1-motion-tracking.md)
