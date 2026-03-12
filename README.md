# UniLab

## 设计哲学

UniLab 的核心目标是验证一个命题：**robot locomotion RL 不需要依赖 GPU 仿真后端**。

主流框架（Isaac Lab、holosoma 等）将物理仿真、replay buffer、策略训练全部放在 GPU 上，形成紧耦合的全 GPU pipeline。这带来极高的吞吐量，但也引入了强硬件依赖（需要高端 CUDA GPU）和架构复杂度（CUDA graph、warp kernel、GPU 内存管理）。

UniLab 采用**统一内存异构运算架构**：

```
┌───────────────────┐     统一共享内存     ┌────────────────────┐
│   CPU 物理仿真    │ ──────────────────▶  │   GPU 策略训练     │
│  mujoco.rollout   │   SharedReplayBuffer │  PPO / SAC / TD3   │
│    多线程并行     │     (POSIX shm)      │    CUDA / MPS      │
└───────────────────┘                      └────────────────────┘
```

- **CPU 仿真**：MuJoCo/Motrix 的 CPU 多线程 step，无需 GPU 仿真内核
- **统一内存**：Collector 和 Learner 通过共享内存解耦，运行在独立进程
- **GPU 训练**：策略网络仍在 GPU 上训练，发挥 GPU 的并行计算优势
- **硬件无关**：Mac（MPS）、Linux（CUDA）均可运行

---

## TODO

- [x] 增加 Bipedal Locomotion 任务
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
- [ ] @yves 增加AC非对称观测
- [ ] @ymr  增加灵巧操作案例
- [ ] @czx  把 FastTD3+Go2 调稳定
- [ ] @jdx  适配 mimic/amp 算法，支持人形Whole-Body Tracking
- [ ] @yves mujoco域随机化,需要改 mujoco-uni 源码
- [ ] @jdx  motrixsim域随机化
- [ ] @jdx  写onnx导出和sim2sim
- [ ] @jdx  sim2real
- [ ] @yves 把稳定的 FastTD3 和 FastSAC 迁移到 mlx
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
| appo(torch)|     |     |    |

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

## 安装 (Installation)

1. **克隆仓库并安装**:
   ```bash
   git clone https://github.com/TATP-233/UniLab.git
   cd UniLab
   pip install --extra-index-url https://test.pypi.org/simple/ mujoco-uni==3.5.0.post2
   pip install -e .
   ```

2. **可选：安装 Motrix 后端支持**:
   ```bash
   pip install -e ".[motrix]"
   ```

## 仿真后端 (Simulation Backends)

UniLab 支持两种仿真后端：

- **MuJoCo** (默认)
- **Motrix** (可选)

### 使用 Motrix 后端

```bash
# 训练
python scripts/train_rsl_rl.py --task Go1JoystickFlatTerrain --sim_backend motrix

# 回放（交互式可视化）
python scripts/train_rsl_rl.py --task Go1JoystickFlatTerrain --sim_backend motrix --play_only
```

## 训练与回放指南

### 1. 开始训练 (Training)

```bash
# PPO (RSL-RL)
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain

# PPO (MLX - Apple Silicon)
python scripts/train_mlx_ppo.py --task Go2JoystickFlatTerrain

# Unified OffPolicy entry (recommended)
python scripts/train_offpolicy.py --algo sac --task Go1JoystickFlatTerrain
python scripts/train_offpolicy.py --algo td3 --task Go1JoystickFlatTerrain

# APPO
python scripts/train_appo.py --task Go2JoystickFlatTerrain
```

**注意**：训练脚本默认在训练完成后会进入回放阶段。`mujoco` 后端会导出 `play_video.mp4`，`motrix` 后端为交互式窗口渲染（不导出视频）。使用 `--no_play` 跳过自动回放。

**日志目录命名**：训练 run 目录统一采用 `YYYY-MM-DD_HH-MM-SS_<sim_backend>`，例如 `2026-03-09_18-30-00_mujoco`。

### 2. 回放与渲染视频 (Play / Evaluation)

使用 `--play_only` 参数跳过训练，直接回放。脚本会加载最新 checkpoint；`mujoco` 回放生成 `play_video.mp4`，`motrix` 回放打开交互窗口。

```bash
# 回放最新训练结果
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain --play_only
python scripts/train_offpolicy.py --algo sac --task Go2JoystickFlatTerrain --play_only

# 加载特定 checkpoint 回放
python scripts/train_offpolicy.py --algo td3 --task Go1JoystickFlatTerrain --play_only --load_run "2024-02-04_12-00-00"
```

### 3. 加载特定 Run 继续训练

```bash
# PPO 继续训练
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain --load_run "2024-02-04_12-00-00"

# OffPolicy (SAC) 继续训练
python scripts/train_offpolicy.py --algo sac --task Go2JoystickFlatTerrain --load_run "2024-02-04_12-00-00"
```

### 通用参数说明

*   `--task`: 任务名称（如 `Go2JoystickFlatTerrain`、`Go1JoystickFlatTerrain`）
*   `--play_only`: 仅回放模式，跳过训练
*   `--no_play`: 训练后跳过自动回放
*   `--load_run`: 指定加载的运行 ID，默认 `-1`（最新）
*   `--play_env_num`: 回放时的环境数量（默认 16）
*   `--logger`: 日志后端（`tensorboard` / `wandb` / `none`）

---

## FastSAC & FastTD3

基于异步多进程架构的 off-policy 算法实现，使用统一共享内存实现 CPU 仿真与 GPU 训练的解耦。

### 核心特性

| 特性 | 说明 |
|------|------|
| **异步多进程** | Collector 进程（CPU 仿真）与 Learner 进程（GPU 训练）独立运行 |
| **统一共享内存** | 通过 SharedReplayBuffer (POSIX shm) 实现零拷贝数据传输 |
| **同步/异步模式** | 支持同步收集（默认）和异步收集两种模式 |
| **自动 Play** | 训练完成后自动生成回放视频 |

### 训练

```bash
# Unified entry
python scripts/train_offpolicy.py --algo sac --task Go2JoystickFlatTerrain
python scripts/train_offpolicy.py --algo td3 --task Go1JoystickFlatTerrain

# 异步模式（更高吞吐量）
python scripts/train_offpolicy.py --algo sac --task Go2JoystickFlatTerrain --no_sync_collection

# 跳过训练后的自动 play
python scripts/train_offpolicy.py --algo td3 --task Go1JoystickFlatTerrain --no_play
```

### 回放

```bash
# 仅回放模式
python scripts/train_offpolicy.py --algo sac --task Go2JoystickFlatTerrain --play_only

# 加载特定 checkpoint
python scripts/train_offpolicy.py --algo td3 --task Go1JoystickFlatTerrain --play_only --load_run "2024-02-04_12-00-00"
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_iterations` | 1500 (SAC) / 5000 (TD3) | 最大训练迭代次数 |
| `--num_envs` | 4096 | 并行环境数量 |
| `--device` | 自动检测 | Learner 设备 (`cuda` / `mps` / `cpu`) |
| `--collector_device` | `cpu` | Collector 设备（默认 `cpu`，可手动设为 `mps`/`cuda`） |
| `--sim_backend` | `mujoco` | 仿真后端（`mujoco` / `motrix`） |
| `--no_sync_collection` | False | 启用异步收集模式 |
| `--env_steps_per_sync` | 1 | 同步模式下每次收集的步数 |
| `--play_only` | False | 仅回放，跳过训练 |
| `--no_play` | False | 训练后跳过自动回放 |

## APPO (Asynchronous PPO)

基于 [IMPACT (Luo et al., 2020)](https://arxiv.org/abs/1912.00167) 的异步 PPO 实现，使用原生多进程实现 CPU 物理仿真与 GPU 策略训练的流水线并行。

### 核心特性

| 特性 | 说明 |
|------|------|
| **V-trace IS 修正** | 使用重要性采样比率修正异步收集导致的 off-policy 偏差 |
| **Target Network** | Soft update (`τ`) 平滑更新目标网络，稳定 IS 比率计算 |
| **异步流水线** | CPU 数据收集与 GPU 训练重叠执行，最大化硬件利用率 |
| **MPS 兼容** | 支持 Apple Silicon (MPS)、CUDA 和纯 CPU 训练 |

### 架构

```
┌────────────────┐         ┌──────────────────┐
│ Collector Proc │ CPU     │   APPOLearner    │  GPU/MPS
│  (rollout)     │────────▶│  V-trace + PPO   │
│  π_behavior    │  batch  │  π_current       │
└────────────────┘         │  π_target (EMA)  │
                           └──────────────────┘
```

### 训练

```bash
# 默认训练 (自动检测 GPU/MPS/CPU)
python scripts/train_appo.py --task Go2JoystickFlatTerrain

# 调整并行环境数和 rollout 长度
python scripts/train_appo.py --total_envs 1024 --steps_per_env 24
```

### 回放

```bash
python scripts/train_appo.py --task Go2JoystickFlatTerrain --play_only
```

### APPO 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total_envs` | 1024 | 总环境数（均分到各 worker） |
| `--steps_per_env` | 24 | 每个环境每次迭代的步数 |
| `--max_iterations` | 1500 | 最大训练迭代次数 |
| `--save_interval` | 50 | 每 N 次迭代保存 checkpoint |
| `--device` | 自动检测 | 训练设备 (`cuda:0` / `mps` / `cpu`) |
| `--collector_device` | 自动检测 | Collector 设备 |
| `--play_only` | False | 仅回放 |
| `--no_play` | False | 训练后跳过自动回放 |

算法超参数（在 `locomotion_params.py` 的 `algorithm` 字段中配置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau` | 1.0 | Target network soft update 系数（1.0 = 硬拷贝） |
| `target_update_freq` | 1 | Target network 更新频率 |
| `vtrace_clip_rho` | 1.0 | V-trace ρ 截断阈值 |
| `vtrace_clip_c` | 1.0 | V-trace c 截断阈值 |

