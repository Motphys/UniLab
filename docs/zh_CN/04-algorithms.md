# 算法

语言: [English](../en/04-algorithms.md) | 简体中文 | [日本語](../ja/04-algorithms.md) | [한국어](../ko/04-algorithms.md)

本页只保留算法级说明。入口脚本和通用 CLI 参数见 [03-training.md](03-training.md)。

## APPO

APPO 是 UniLab 的异步 PPO 实现，带有 V-trace importance-sampling 修正。collector 子进程负责 CPU 仿真，learner 进程负责 GPU 训练，二者通过 ring-buffer 流水线并行运行。

### Core Features

| 特性 | 说明 |
|------|------|
| 异步多进程 | collector 和 learner 并行运行 |
| V-trace IS 修正 | 用 `pi_target / pi_behavior` 修正 off-policy 数据 |
| 4 槽 ring buffer | 最多 4 条 rollout 可同时在飞 |
| Replay queue | learner 侧缓存待消费 rollout 的队列 |
| 日志目录 | `logs/appo/<task>/<timestamp>_mujoco/` |

### Usage

```bash
# 默认训练
uv run python scripts/train_appo.py task=go1_joystick

# 指定环境数和迭代数
uv run python scripts/train_appo.py task=go2_joystick algo.num_envs=2048 algo.max_iterations=300

# 调整 replay queue 深度
uv run python scripts/train_appo.py task=go1_joystick training.replay_queue_size=2

# 跳过自动回放
uv run python scripts/train_appo.py task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true training.load_run="2026-03-16_01-35-12_mujoco"
```

### Key Parameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task` | `go2_joystick` | task 配置名 |
| `algo.max_iterations` | 150 | 最大训练迭代数 |
| `algo.num_envs` | 2048 | 并行环境数量 |
| `algo.steps_per_env` | 24 | 每个 env 的 rollout 长度 |
| `training.replay_queue_size` | 3 | learner 侧 rollout 重放深度 |
| `training.device` | 自动检测 | learner 设备 |
| `training.collector_device` | `cpu` | collector 设备 |
| `training.logger` | `tensorboard` | 日志后端 |
| `training.play_only` | false | 仅回放 |
| `training.no_play` | false | 跳过自动回放 |
| `training.load_run` | `-1` | run 目录名或 checkpoint 路径 |
| `algo.save_interval` | 50 | checkpoint 保存间隔 |

### APPO vs PPO

| 维度 | rsl-rl PPO | APPO |
|------|------------|------|
| 采集方式 | 同步 | 异步 |
| IS 修正 | 无 | V-trace |
| CPU / GPU 利用率 | 交替满载 | 同时满载 |
| 适用场景 | 样本效率优先 | 吞吐优先 |

## FastSAC And FastTD3

FastSAC 和 FastTD3 使用同一套异步多进程架构，通过 shared memory 将 CPU 仿真和 GPU 训练解耦。

### Core Features

| 特性 | 说明 |
|------|------|
| 异步多进程 | collector 与 learner 独立运行 |
| 统一共享内存 | 使用 PyTorch shared tensors 零拷贝传输 |
| 同步 / 异步模式 | 同时支持默认同步采集和异步采集 |
| 自动回放 | 训练结束后自动进入回放 |

### Usage

```bash
# 基本训练
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# 异步采集模式
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.no_sync_collection=true

# 跳过自动回放
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

### Key Parameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algo` | `sac` | 算法选择 |
| `task` | `go1_joystick` | task 配置名 |
| `algo.max_iterations` | 500 (SAC) / 5000 (TD3) | 最大训练迭代数 |
| `algo.num_envs` | 4096 | 并行环境数量 |
| `training.device` | 自动检测 | learner 设备 |
| `training.sim_backend` | `mujoco` | 仿真后端 |
| `training.no_sync_collection` | false | 启用异步采集 |
| `training.env_steps_per_sync` | 1 | 同步模式下每轮采集步数 |
| `training.play_only` | false | 仅回放 |
| `training.no_play` | false | 跳过自动回放 |

## Navigation

- Previous: [Training Guide](03-training.md)
- Next: [G1 Motion Tracking](05-g1-motion-tracking.md)
