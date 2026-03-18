# UniLab

## 设计哲学

UniLab 的核心目标是验证一个命题：**robot locomotion RL 不需要依赖 GPU 仿真后端**。

主流框架（Isaac Lab、holosoma 等）将物理仿真、replay buffer、策略训练全部放在 GPU 上，形成紧耦合的全 GPU pipeline。这带来极高的吞吐量，但也引入了强硬件依赖（需要高端 CUDA GPU）和架构复杂度（CUDA graph、warp kernel、GPU 内存管理）。

UniLab 采用**统一内存异构运算架构**：

```
┌───────────────────┐     统一共享内存     ┌────────────────────┐
│   CPU 物理仿真    │ ──────────────────▶  │   GPU 策略训练     │
│  mujoco.rollout   │   SharedReplayBuffer │  PPO / SAC / TD3   │
│    多线程并行     │  (PyTorch shared)    │    CUDA / MPS      │
└───────────────────┘                      └────────────────────┘
```

- **CPU 仿真**：MuJoCo/Motrix 的 CPU 多线程 step，无需 GPU 仿真内核
- **统一内存**：Collector 和 Learner 通过 PyTorch shared tensors 零拷贝通信，运行在独立进程
- **GPU 训练**：策略网络仍在 GPU 上训练，发挥 GPU 的并行计算优势
- **硬件无关**：Mac（MPS）、Linux（CUDA）均可运行

---

## 安装

### 使用 uv（推荐）

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库
git clone https://github.com/TATP-233/UniLab.git
cd UniLab

# 3. 安装系统依赖
brew install cmake  # macOS
# sudo apt-get install cmake  # Ubuntu/Debian

# 4. 同步依赖
# macOS (MPS)
uv sync

# Linux (CUDA 12.4)
uv sync --extra cu124

# 5. 可选：Motrix 后端
uv sync --extra motrix
```

### 国内镜像加速

```bash
# 环境变量
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 或命令行参数
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## TODO

- [x] @czx 增加 Bipedal Locomotion 任务
- [x] @czx 增加 APPO
- [x] @czx 增加 FastTD3 和 FastSAC，torch.mps
- [x] @czx  写轻量版并行采样，跑满cpu 和 gpu @czx
- [x] @yves 优化调度器，测试把 rollout 移动到 cpu 的效率提升
- [x] @yves 兼容 linux 平台
- [x] @yves 优化IPC，做到数值稳定，跨 torch/mlx
- [x] @yves 发布 mujoco-uni
- [x] @yves 统一仿真后端
- [x] @yves G1+off-policy调稳定
- [x] @yves G1 full command space
- [x] @yves 增加G1对称性数据增强
- [x] @yves IPC性能优化：PyTorch shared tensors替换POSIX shm
- [x] @yves GPU利用率分析与优化
- [x] @yves AMP混合精度训练
- [x] @yves 跑通算法迁移: PPO->APPO，but 目前优化不稳定
- [x] @yves 工程化设计 原型级->开发级
- [x] @jdx  增加AC非对称观测
- [ ] @ymr  增加灵巧操作案例
- [ ] @all  off-policy 找到性能最好的 off-policy 算法
- [ ] @czx  适配 mimic/amp 算法，支持人形Whole-Body Tracking
- [ ] @yves mujoco域随机化,需要改 mujoco-uni 源码
- [ ] @jdx  motrixsim域随机化
- [ ] @jdx  写onnx导出和sim2sim
- [ ] @jdx  sim2real
- [ ] @Motphys 算法适配 motrixsim
- [ ] @Motphys motrix step 提速

### 训练状态

#### Mujoco后端

|   算法     | Go1 | Go2 | G1 |
|------------|-----|-----|----|
| ppo(torch) |  ✅ |     | ✅ |
| ppo(mlx)   |  ✅ |     | ✅ |
| sac(torch) |  ✅ | ⚠️  | ✅ |
| td3(torch) |  ⚠️ | ⚠️  | ⚠️ |
| appo(torch)|  ✅ | ✅  | ✅ |

#### Motrix后端

|   算法     | Go1 | Go2 | G1 |
|------------|-----|-----|----|
| ppo(torch) |  ⚠️ |     |    |
| ppo(mlx)   |  ⚠️ |     |    |
| sac(torch) |  ⚠️ |     |    |
| td3(torch) |     |     |    |
| appo(torch)|     |     |    |

**说明**：
- ✅ 已支持
- ⚠️ 开发中

Thirdparty:
   1. https://github.com/mujocolab/mjlab
   2. https://github.com/amazon-far/holosoma
   3. https://github.com/google-deepmind/mujoco
   4. https://github.com/google-deepmind/mujoco_playground/

## 仿真后端 (Simulation Backends)

UniLab 支持两种仿真后端：

- **MuJoCo** (默认)
- **Motrix** (可选)

### 使用 Motrix 后端

```bash
# 训练
uv run python scripts/train_rsl_rl.py task=go1_joystick

# 指定motrix后端
uv run python scripts/train_rsl_rl.py task=go1_joystick training.sim_backend=motrix

# 回放（交互式可视化）
uv run python scripts/train_rsl_rl.py task=go1_joystick training.play_only=true
```

## 训练与回放指南

### 1. 开始训练 (Training)

```bash
# PPO (RSL-RL) - 使用 Hydra 配置
uv run python scripts/train_rsl_rl.py task=go1_joystick

# PPO (MLX - Apple Silicon)
uv run python scripts/train_mlx_ppo.py task=go1_joystick

# APPO（异步 PPO，CPU/GPU 并行）
uv run python scripts/train_appo.py task=go1_joystick

# Off-Policy (SAC/TD3) - 推荐使用统一入口
uv run python scripts/train_offpolicy.py algo=sac task=go1_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# CLI 覆盖配置参数
uv run python scripts/train_offpolicy.py algo=sac task=g1_sac algo.num_envs=2048 algo.max_iterations=1000
```

**注意**：训练脚本默认在训练完成后会进入回放阶段。`mujoco` 后端会导出 `play_video.mp4`，`motrix` 后端为交互式窗口渲染（不导出视频）。使用 `training.no_play=true` 跳过自动回放。

**日志目录命名**：训练 run 目录统一采用 `YYYY-MM-DD_HH-MM-SS_<sim_backend>`，例如 `2026-03-09_18-30-00_mujoco`。

### 2. 回放与渲染视频 (Play / Evaluation)

使用 `training.play_only=true` 参数跳过训练，直接回放。脚本会加载最新 checkpoint；`mujoco` 回放生成 `play_video.mp4`，`motrix` 回放打开交互窗口。

```bash
# 回放最新训练结果
uv run python scripts/train_rsl_rl.py task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true

# 加载特定 checkpoint 回放
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

### 3. 加载特定 Run 继续训练

```bash
# PPO 继续训练
uv run python scripts/train_rsl_rl.py task=go2_joystick training.load_run="2024-02-04_12-00-00"

# OffPolicy (SAC) 继续训练
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.load_run="2024-02-04_12-00-00"
```

### 通用参数说明

所有训练脚本使用 Hydra 配置系统，支持 CLI 覆盖：

```bash
# 基本格式
uv run python scripts/train_*.py [config_group=value] [key.subkey=value]

# 示例
task=go1_joystick                    # 选择任务配置
algo=sac                             # 选择算法配置（offpolicy）
training.play_only=true              # 仅回放模式
training.no_play=true                # 训练后跳过回放
training.load_run="-1"               # 加载运行 ID（-1=最新）
training.logger=tensorboard          # 日志后端（tensorboard/wandb/none）
algo.num_envs=2048                   # 覆盖环境数量
algo.max_iterations=1000             # 覆盖迭代次数
```

查看完整配置：
```bash
uv run python scripts/train_offpolicy.py --cfg job
```

---

## APPO（异步 PPO）

基于 V-trace 重要性采样修正的异步 PPO 实现。Collector 子进程（CPU 仿真）通过 4 槽 ring buffer 持续写入 rollout，Learner 进程（GPU）无需等待即可从 ring buffer 中消费并训练，实现真正的 CPU/GPU 并行。

### 核心特性

| 特性 | 说明 |
|------|------|
| **异步多进程** | Collector（CPU）与 Learner（GPU）完全解耦，各自满负荷运行 |
| **V-trace IS 修正** | 使用重要性采样比 ρ = π_target/π_behavior 修正 off-policy 数据的优势估计 |
| **4 槽 ring buffer** | 支持最多 4 条 rollout 在飞，collector 满时覆盖最旧 slot（无阻塞） |
| **Replay queue** | Learner 端维护 rollout 重放队列，每次迭代消费所有可用 slot |
| **日志目录** | `logs/appo/<task>/<timestamp>_mujoco/` |

### 训练

```bash
# 默认参数训练
uv run python scripts/train_appo.py task=go1_joystick

# 指定环境数量和迭代次数
uv run python scripts/train_appo.py task=go2_joystick algo.num_envs=2048 algo.max_iterations=300

# 自定义 replay queue 深度（越大 staleness 越高，但 GPU 利用率更稳定）
uv run python scripts/train_appo.py task=go1_joystick training.replay_queue_size=2

# 跳过训练后的自动 play
uv run python scripts/train_appo.py task=go1_joystick training.no_play=true
```

### 回放

```bash
# 仅回放（加载最新 checkpoint）
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true

# 加载特定 checkpoint 回放
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true training.load_run="2026-03-16_01-35-12_mujoco"
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task` | `go2_joystick` | 任务配置名称 |
| `algo.max_iterations` | 150 | 最大训练迭代次数 |
| `algo.num_envs` | 2048 | 并行环境数量 |
| `algo.steps_per_env` | 24 | 每条 rollout 的步数 |
| `training.replay_queue_size` | 3 | Learner 端 rollout 重放队列深度 |
| `training.device` | 自动检测 | Learner 设备（`cuda` / `mps` / `cpu`） |
| `training.collector_device` | `cpu` | Collector 设备 |
| `training.logger` | `tensorboard` | 日志后端（`tensorboard` / `wandb` / `none`） |
| `training.play_only` | false | 仅回放，跳过训练 |
| `training.no_play` | false | 训练后跳过自动回放 |
| `training.load_run` | `-1` | 指定 run 目录名或 checkpoint 路径 |
| `algo.save_interval` | 50 | 每隔多少次迭代保存一次 checkpoint |

### 与 PPO 的对比

| 维度 | rsl-rl PPO | APPO |
|------|-----------|------|
| 收集方式 | 同步（训练等收集） | 异步（并行运行） |
| IS 修正 | 无（严格 on-policy） | V-trace（容忍 staleness） |
| CPU/GPU 利用率 | 交替满载 | 同时满载 |
| 适用场景 | 快速收敛、样本效率优先 | 吞吐量优先、大规模并行 |

---

## FastSAC & FastTD3

基于异步多进程架构的 off-policy 算法实现，使用统一共享内存实现 CPU 仿真与 GPU 训练的解耦。

### 核心特性

| 特性 | 说明 |
|------|------|
| **异步多进程** | Collector 进程（CPU 仿真）与 Learner 进程（GPU 训练）独立运行 |
| **统一共享内存** | 通过 PyTorch shared tensors 实现零拷贝 CPU-GPU 数据传输 |
| **同步/异步模式** | 支持同步收集（默认）和异步收集两种模式 |
| **自动 Play** | 训练完成后自动生成回放视频 |

### 训练

```bash
# 基本训练
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# 异步模式（更高吞吐量）
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.no_sync_collection=true

# 跳过训练后的自动 play
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.no_play=true
```

### 回放

```bash
# 仅回放模式
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true

# 加载特定 checkpoint
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algo` | `sac` | 算法选择（`sac` / `td3`） |
| `task` | `go1_joystick` | 任务配置名称 |
| `algo.max_iterations` | 500 (SAC) / 5000 (TD3) | 最大训练迭代次数 |
| `algo.num_envs` | 4096 | 并行环境数量 |
| `training.device` | 自动检测 | Learner 设备 (`cuda` / `mps` / `cpu`) |
| `training.sim_backend` | `mujoco` | 仿真后端（`mujoco` / `motrix`） |
| `training.no_sync_collection` | false | 启用异步收集模式 |
| `training.env_steps_per_sync` | 1 | 同步模式下每次收集的步数 |
| `training.play_only` | false | 仅回放，跳过训练 |
| `training.no_play` | false | 训练后跳过自动回放 |
