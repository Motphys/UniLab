# UniLab

## TODO

- [x] 增加 Bipedal Locomotion 任务
- [x] 增加 FastTD3 和 FastSAC，torch.mps
- [ ] 把 FastTD3 和 FastSAC 迁移到 mlx
- [ ] 设计统一训练框架，支持不同的算法（ppo、TD3、SAC）和后端（mlx 和 pytorch）
- [ ] 适配 mimic/amp 算法，支持人形Whole-Body Tracking
- [x] (*)用MLX重写 MuJoCo

Thirdparty:
   1. https://github.com/mujocolab/mjlab
   2. https://github.com/amazon-far/holosoma
   3. https://github.com/google-deepmind/mujoco
   4. https://github.com/google-deepmind/mujoco_playground/

## 安装 (Installation)

1. **安装依赖**:
   项目依赖 `mujoco-mlx`，当前从 [TestPyPI](https://test.pypi.org/project/mujoco-mlx/) 获取。在项目根目录下运行：
   ```bash
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mujoco-mlx==3.5.0
   ```

2. **克隆仓库**:
   ```bash
   git clone https://github.com/TATP-233/UniLab.git
   cd UniLab
   pip install -e .
   ```

## 训练与回放指南

### 1. 开始训练 (Training)
默认使用 `Go2JoystickFlatTerrain` 任务：

```bash
# 基本训练
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain
```

### 2. 回放与渲染视频 (Play / Evaluation)
增加 `--play_only` 参数。脚本默认会加载最新的一次 `run`，并在该次 run 的目录中生成 `play_video.mp4`。

```bash
# 加载最新的一次训练结果进行回放并渲染
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain --play_only
```

### 3. 加载特定 Run 继续训练
如果你想从某个特定的检查点继续训练：

```bash
# --load_run 可以是 logs/rsl_rl_train/TaskName 下的文件夹名
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain --load_run "2024-02-04_12-00-00"
```

### 参数说明
*   `--task`: 任务名称（如 `Go2JoystickFlatTerrain` 或 `Go1JoystickFlatTerrain`）。
*   `--play_only`: 仅推理回放模式，不进行训练，会生成并行渲染的视频。
*   `--load_run`: 指定加载的运行 ID (文件夹名)，默认为 `-1` (最新)。
*   `--env_num`: 训练时的环境数量 (默认 1024)。
*   `--play_env_num`: 回放时的环境数量 (默认 16)。

---

## APPO (Asynchronous PPO)

基于 [IMPACT (Luo et al., 2020)](https://arxiv.org/abs/1912.00167) 的异步 PPO 实现，使用 Ray 实现 CPU 物理仿真与 GPU 策略训练的流水线并行。

### 核心特性

| 特性 | 说明 |
|------|------|
| **V-trace IS 修正** | 使用重要性采样比率修正异步收集导致的 off-policy 偏差 |
| **Target Network** | Soft update (`τ`) 平滑更新目标网络，稳定 IS 比率计算 |
| **异步流水线** | CPU 数据收集与 GPU 训练重叠执行，最大化硬件利用率 |
| **MPS 兼容** | 支持 Apple Silicon (MPS)、CUDA 和纯 CPU 训练 |

### 架构

```
┌──────────────┐         ┌──────────────────┐
│  Ray Worker  │  CPU    │   APPOLearner    │  GPU/MPS
│  (rollout)   │────────▶│  V-trace + PPO   │
│  π_behavior  │  batch  │  π_current       │
└──────────────┘         │  π_target (EMA)  │
                         └──────────────────┘
```

### 训练

```bash
# 默认训练 (自动检测 GPU/MPS/CPU)
python scripts/train_appo.py --task Go2JoystickFlatTerrain

# 推荐配置：1 worker 避免 MuJoCo 线程竞争
python scripts/train_appo.py --num_workers 1 --total_envs 1024 --steps_per_env 24

# 多 worker (适用于多核服务器)
python scripts/train_appo.py --num_workers 4 --total_envs 4096
```

### 回放

```bash
python scripts/train_appo.py --task Go2JoystickFlatTerrain --play_only
```

### APPO 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_workers` | 1 | Ray rollout worker 数量 |
| `--total_envs` | 1024 | 总环境数（均分到各 worker） |
| `--steps_per_env` | 24 | 每个环境每次迭代的步数 |
| `--max_iterations` | 1500 | 最大训练迭代次数 |
| `--save_interval` | 50 | 每 N 次迭代保存 checkpoint |
| `--device` | 自动检测 | 训练设备 (`cuda:0` / `mps` / `cpu`) |

算法超参数（在 `locomotion_params.py` 的 `algorithm` 字段中配置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau` | 1.0 | Target network soft update 系数（1.0 = 硬拷贝） |
| `target_update_freq` | 1 | Target network 更新频率 |
| `vtrace_clip_rho` | 1.0 | V-trace ρ 截断阈值 |
| `vtrace_clip_c` | 1.0 | V-trace c 截断阈值 |

