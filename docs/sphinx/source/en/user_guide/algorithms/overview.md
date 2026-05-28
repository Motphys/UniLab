# 算法


本页只保留算法级说明。入口脚本和通用 CLI 参数见 {doc}`Training Guide <../getting_started/training>`。

## APPO

APPO 是 UniLab 的异步 PPO 实现，带有 V-trace importance-sampling 修正。collector 子进程负责 CPU 仿真，learner 进程负责 GPU 训练，二者通过 ring-buffer 流水线并行运行。

### Core Features

| 特性 | 说明 |
|------|------|
| 异步多进程 | collector 和 learner 并行运行 |
| V-trace IS 修正 | 用 `pi_target / pi_behavior` 修正 off-policy 数据 |
| 4 槽 ring buffer | 最多 4 条 rollout 可同时在飞 |
| Replay queue | learner 侧缓存待消费 rollout 的队列 |
| 日志目录 | `logs/<algo.algo_log_name>/<task>/<timestamp>_<sim_backend>/` |

### Usage

```bash
# 默认训练
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco

# 指定环境数和迭代数
uv run scripts/train_appo.py task=go2_joystick_flat/mujoco algo.num_envs=2048 algo.max_iterations=300

# 调整 replay queue 深度
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.replay_queue_size=2

# 跳过自动回放
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.no_play=true
```

### Playback

```bash
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.play_only=true
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.play_only=true algo.load_run="2026-03-16_01-35-12_mujoco"
```

### Key Parameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task` | `go1_joystick_flat/mujoco` | 单个 task 配置入口，内部同时定义 task + backend |
| `algo.max_iterations` | 150 | 最大训练迭代数 |
| `algo.num_envs` | 2048 | 并行环境数量 |
| `algo.steps_per_env` | 24 | 每个 env 的 rollout 长度 |
| `training.replay_queue_size` | 3 | learner 侧 rollout 重放深度 |
| `training.device` | 自动检测 | learner 设备 |
| `training.collector_device` | auto (follows `training.device`) | collector 设备 |
| `training.logger` | `tensorboard` | 日志后端 |
| `training.play_only` | false | 仅回放 |
| `training.no_play` | false | 跳过自动回放 |
| `algo.load_run` | `-1` | run 目录名或 checkpoint 路径 |
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
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco

# 异步采集模式
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco training.no_sync_collection=true

# 跳过自动回放
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco training.no_play=true
```

### Playback

```bash
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco training.play_only=true
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco training.play_only=true algo.load_run="2024-02-04_12-00-00"
```

### Key Parameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algo` | `sac` | 算法选择 |
| `task` | `sac/g1_walk_flat/mujoco` | 单个 task 配置入口，内部同时定义 algo + task + backend |
| `algo.max_iterations` | 500 (SAC) / 5000 (TD3) | 最大训练迭代数 |
| `algo.num_envs` | 4096 | 并行环境数量 |
| `training.device` | 自动检测 | learner 设备 |
| `conf/*/task/...` | - | 唯一 owner 配置入口；reward/env/backend-specific algo 都在这里改 |
| `training.no_sync_collection` | false | 启用异步采集 |
| `training.env_steps_per_sync` | 1 | 同步模式下每轮采集步数 |
| `training.play_only` | false | 仅回放 |
| `training.no_play` | false | 跳过自动回放 |

## FlashSAC

FlashSAC 是基于 FlashAttention 风格 Block 的 off-policy 算法，actor 使用 BatchNorm embedder + 结构化噪声探索，critic 使用 distributional Q（C51 变体）。与 FastSAC 共用同一个训练入口 `train_offpolicy.py`，但网络结构和前向接口不同。

### Usage

```bash
# 基本训练
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco

# 跳过自动回放
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco training.no_play=true
```

### Playback

```bash
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco training.play_only=true
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco training.play_only=true algo.load_run="2026-04-23_14-06-57_mujoco"
```

### Key Parameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algo` | `flashsac` | 算法选择 |
| `task` | `flashsac/g1_walk_flat/mujoco` | 唯一 owner 配置入口；reward/env/algo 都在这里改 |
| `algo.max_iterations` | 5000 | 最大训练迭代数 |
| `algo.num_envs` | 1024 | 并行环境数量 |
| `algo.tau` | 0.01 | target network 软更新系数 |
| `algo.algo_params.actor_num_blocks` | 2 | actor FlashSAC block 层数 |
| `algo.algo_params.critic_num_blocks` | 2 | critic FlashSAC block 层数 |
| `training.play_only` | false | 仅回放 |
| `training.no_play` | false | 跳过自动回放 |
