# Go1 + SAC 实现对比：UniLab vs holosoma

**文档日期**: 2026-03-10
**对比版本**: UniLab (dev/gpu_buffer) vs holosoma (third-party)

---

## 目录

1. [核心架构差异](#1-核心架构差异)
2. [FastSAC 算法实现](#2-fastsac-算法实现)
3. [网络架构对比](#3-网络架构对比)
4. [Go1 环境实现](#4-go1-环境实现)
5. [奖励函数对比](#5-奖励函数对比)
6. [Domain Randomization](#6-domain-randomization)
7. [Curriculum Learning](#7-curriculum-learning)
8. [问题总结与修复建议](#8-问题总结与修复建议)

---

## 1. 核心架构差异

### 1.1 仿真后端

| 项目 | holosoma | UniLab |
|------|----------|--------|
| **仿真引擎** | mujoco_warp (GPU) | mujoco.rollout (CPU) |
| **策略训练** | GPU (PyTorch) | GPU (PyTorch) |
| **数据传输** | GPU 内存直接访问 | 共享内存 (POSIX shm) |
| **并行方式** | GPU 并行 | CPU 多线程 + GPU 训练 |

**设计哲学差异**：
- **holosoma**: 全 GPU pipeline，追求极致性能（需要 GPU 仿真支持）
- **UniLab**: CPU/GPU 解耦，统一内存异构运算，硬件灵活部署

**结论**: 这是**架构选择差异**，不是实现缺陷。UniLab 的设计目标是在不依赖 GPU 仿真的情况下实现高效训练。

---

## 2. FastSAC 算法实现

### 2.1 超参数对比

#### 核心超参数

| 参数 | holosoma | UniLab | 状态 | 说明 |
|------|----------|--------|------|------|
| `gamma` | 0.97 | 0.97 | ✅ 一致 | 折扣因子 |
| `tau` | 0.125 | 0.125 | ✅ 一致 | 目标网络软更新系数 |
| `actor_lr` | 3e-4 | 3e-4 | ✅ 一致 | Actor 学习率 |
| `critic_lr` | 3e-4 | 3e-4 | ✅ 一致 | Critic 学习率 |
| `alpha_lr` | 3e-4 | 3e-4 | ✅ 一致 | 熵系数学习率 |
| `alpha_init` | 0.001 | 0.001 | ✅ 一致 | 熵系数初始值 |
| `batch_size` | 8192 | 8192 | ✅ 一致 | 批次大小 |
| `updates_per_step` | 8 | 8 | ✅ 一致 | 每步更新次数 |
| `policy_frequency` | 4 | 4 | ✅ 一致 | 策略更新频率 |
| `replay_buffer_n` | 1024 | 1024 | ✅ 一致 | 每环境 buffer 大小 |
| `num_atoms` | 101 | 101 | ✅ 一致 | 分布式 Q 网络原子数 |
| `v_min` | -20.0 | -20.0 | ✅ 一致 | Q 值下界 |
| `v_max` | 20.0 | 20.0 | ✅ 一致 | Q 值上界 |

#### 关键差异

| 参数 | holosoma | UniLab | 影响 |
|------|----------|--------|------|
| **`target_entropy_ratio`** | 0.0 | 1.0 | ⚠️ **重要差异** |
| **`weight_decay`** | 0.001 | 0.0 | ⚠️ **重要差异** |
| `obs_normalization` | True | True | ✅ 一致 |
| `use_layer_norm` | True | True | ✅ 一致 |
| `max_grad_norm` | 0.0 | 0.0 | ✅ 一致 |

**详细说明**：

1. **`target_entropy_ratio`**:
   - **holosoma**: 0.0 → `target_entropy = 0`（固定熵目标）
   - **UniLab**: 1.0 → `target_entropy = -action_dim`（标准 SAC）
   - **影响**: 控制探索程度，0.0 可能导致策略过早收敛

2. **`weight_decay`**:
   - **holosoma**: 所有优化器使用 0.001
   - **UniLab**: 未设置（默认 0.0）
   - **影响**: 正则化强度，影响网络泛化能力

---

### 2.2 算法实现细节

#### 2.2.1 Actor 网络

**代码位置**：
- holosoma: `src/holosoma/holosoma/agents/fast_sac/fast_sac.py:Actor`
- UniLab: `unilab/algos/torch/fast_sac/learner.py:SACActor`

**架构**：
```python
# 两者完全一致
Input (obs_dim)
  → Linear(hidden_dim=512) → LayerNorm → SiLU
  → Linear(256) → LayerNorm → SiLU
  → Linear(128) → LayerNorm → SiLU
  → fc_mu(action_dim) + fc_logstd(action_dim)
```

**输出头初始化**：
```python
# 两者一致：零初始化
nn.init.constant_(fc_mu.weight, 0.0)
nn.init.constant_(fc_mu.bias, 0.0)
nn.init.constant_(fc_logstd.weight, 0.0)
nn.init.constant_(fc_logstd.bias, 0.0)
```

**动作采样**：
```python
# 两者一致：tanh-squashed Gaussian
raw_action = Normal(mean, std).rsample()
tanh_action = tanh(raw_action)
action = tanh_action * action_scale + action_bias

# Log prob with Jacobian correction
log_prob = dist.log_prob(raw_action)
log_prob -= log(1 - tanh_action^2 + 1e-6)  # tanh correction
log_prob -= log(action_scale + 1e-6)       # scale correction
```

#### 2.2.2 Critic 网络

**代码位置**：
- holosoma: `src/holosoma/holosoma/agents/fast_sac/fast_sac.py:Critic`
- UniLab: `unilab/algos/torch/fast_sac/learner.py:DistributionalQNetwork`

**架构**：
```python
# 两者完全一致：Distributional Q-Network (C51)
Input (obs_dim + action_dim)
  → Linear(hidden_dim=768) → LayerNorm → SiLU
  → Linear(384) → LayerNorm → SiLU
  → Linear(192) → LayerNorm → SiLU
  → Linear(num_atoms=101)
```

**分布式 Bellman 更新**：
- 支持范围：`[v_min=-20, v_max=20]`
- 原子数：101
- 使用 projection 算法进行分布更新

---

## 3. 网络架构对比

### 3.1 完整架构总结

| 组件 | holosoma | UniLab | 一致性 |
|------|----------|--------|--------|
| Actor 隐藏层 | [512, 256, 128] | [512, 256, 128] | ✅ |
| Actor 激活函数 | SiLU | SiLU | ✅ |
| Actor 归一化 | LayerNorm | LayerNorm | ✅ |
| Actor 输出 | Tanh-squashed | Tanh-squashed | ✅ |
| Critic 隐藏层 | [768, 384, 192] | [768, 384, 192] | ✅ |
| Critic 类型 | Distributional (C51) | Distributional (C51) | ✅ |
| Critic 数量 | 2 (ensemble) | 2 (ensemble) | ✅ |
| 目标网络 | Soft update (τ=0.125) | Soft update (τ=0.125) | ✅ |

**结论**: 网络架构完全一致，UniLab 成功复现了 holosoma 的网络设计。

---

## 4. Go1 环境实现

### 4.1 环境配置

**代码位置**：
- holosoma: `src/holosoma/holosoma/envs/locomotion/locomotion_manager.py`
- UniLab: `unilab/envs/locomotion/go1/joystick.py`

#### 基础配置

| 配置项 | holosoma | UniLab | 说明 |
|--------|----------|--------|------|
| 机器人 DOF | 12 | 12 | ✅ 一致 |
| 控制频率 | 50 Hz (0.02s) | 50 Hz (0.02s) | ✅ 一致 |
| Episode 长度 | 20s (1000 steps) | 20s (1000 steps) | ✅ 一致 |
| 初始高度 | ~0.3m | 0.45m | ⚠️ 不同 |
| 默认姿态 | 配置文件定义 | 配置文件定义 | - |

### 4.2 观测空间

**UniLab Go1 观测** (33 维):
```python
obs = [
    linvel (3),           # 本体坐标系线速度
    gyro (3),             # 角速度
    gravity (3),          # 重力向量（负值）
    dof_pos - default (12),  # 关节位置偏差
    dof_vel (12),         # 关节速度
    last_actions (12),    # 上一步动作
    commands (3)          # 速度命令 [vx, vy, vyaw]
]
```

**holosoma 观测**（需查看 ObservationManagerCfg）:
- 可能包含更多信息（如接触力、高度扫描等）
- 使用模块化的 observation manager

---

## 5. 奖励函数对比

### 5.1 UniLab Go1 奖励

**代码**: `unilab/envs/locomotion/go1/joystick.py:64-73`

```python
reward_scales = {
    "tracking_lin_vel": 1.0,      # 线速度跟踪
    "tracking_ang_vel": 0.2,      # 角速度跟踪
    "lin_vel_z": -5.0,            # 惩罚垂直速度
    "ang_vel_xy": -0.1,           # 惩罚 roll/pitch 角速度
    "base_height": -100.0,        # 惩罚高度偏差
    "action_rate": -0.005,        # 惩罚动作变化率
    "similar_to_default": -0.1,   # 惩罚偏离默认姿态
}
```

**奖励函数实现**：

1. **`tracking_lin_vel`** (权重 1.0):
   ```python
   lin_vel_error = sum((cmd[:2] - linvel[:2])^2)
   reward = exp(-lin_vel_error / 0.25)
   ```

2. **`tracking_ang_vel`** (权重 0.2):
   ```python
   ang_vel_error = (cmd[2] - gyro[2])^2
   reward = exp(-ang_vel_error / 0.25)
   ```

3. **`lin_vel_z`** (权重 -5.0):
   ```python
   reward = -linvel_z^2
   ```

4. **`ang_vel_xy`** (权重 -0.1):
   ```python
   reward = -(gyro_x^2 + gyro_y^2)
   ```

5. **`base_height`** (权重 -100.0):
   ```python
   reward = -(base_height - 0.3)^2
   ```

6. **`action_rate`** (权重 -0.005):
   ```python
   reward = -sum((action - last_action)^2)
   ```

7. **`similar_to_default`** (权重 -0.1):
   ```python
   reward = -sum(abs(dof_pos - default_angles))
   ```

### 5.2 holosoma 奖励系统

**架构**: 使用 `RewardManagerCfg` 模块化配置

**典型奖励项**（基于 locomotion_manager.py 推断）:
- `tracking_lin_vel`: 线速度跟踪
- `tracking_ang_vel`: 角速度跟踪
- `lin_vel_z`: 垂直速度惩罚
- `ang_vel_xy`: roll/pitch 惩罚
- `orientation`: 姿态惩罚
- `base_height`: 高度惩罚
- `action_rate`: 动作平滑
- `torques`: 力矩惩罚
- `dof_acc`: 关节加速度惩罚
- `collision`: 碰撞惩罚
- `termination`: 终止惩罚
- **`alive`**: 存活奖励（重要！）

### 5.3 关键差异

| 奖励项 | holosoma | UniLab Go1 | 状态 |
|--------|----------|------------|------|
| `tracking_lin_vel` | ✅ | ✅ | 一致 |
| `tracking_ang_vel` | ✅ | ✅ | 一致 |
| `lin_vel_z` | ✅ | ✅ | 一致 |
| `ang_vel_xy` | ✅ | ✅ | 一致 |
| `base_height` | ✅ | ✅ | 一致 |
| `action_rate` | ✅ | ✅ | 一致 |
| `similar_to_default` | ✅ | ✅ | 一致 |
| **`alive`** | ✅ | ❌ | **缺失** |
| `orientation` | ✅ | ❌ | 缺失 |
| `torques` | ✅ | ❌ | 缺失 |
| `dof_acc` | ✅ | ❌ | 缺失 |
| `collision` | ✅ | ❌ | 缺失 |

**问题**：
1. **缺少 `alive` 奖励**：这是重要的基础奖励，鼓励机器人保持站立
2. 权重可能不同（需要查看 holosoma 的具体配置文件）

---

## 6. Domain Randomization

### 6.1 UniLab Go1 随机化

**代码**: `unilab/envs/locomotion/go1/joystick.py:152-165`

```python
def reset(env_indices):
    # 1. 位置随机化
    dxy = uniform(-0.5, 0.5, size=(N, 2))
    qpos[:, 0:2] += dxy

    # 2. 朝向随机化
    yaw = uniform(-π, π, size=(N,))
    qpos[:, 3:7] = quat_mul(qpos[:, 3:7], yaw_to_quat(yaw))

    # 3. 速度随机化
    qvel[:, 0:6] = uniform(-0.5, 0.5, size=(N, 6))

    # 4. 关节位置随机化（在 base.py 中）
    dof_pos = default_angles * uniform(0.5, 1.5)
```

**随机化范围**：
- 位置: ±0.5m (XY)
- 朝向: ±180°
- 线速度: ±0.5 m/s
- 角速度: ±0.5 rad/s
- 关节位置: 50%-150% 默认值

### 6.2 holosoma 随机化

**架构**: 使用 `RandomizationManagerCfg` 模块化配置

**支持的随机化类型**（基于 locomotion_manager.py）:

1. **Push Randomization** (推力扰动):
   ```python
   def _push_robots(env_ids):
       rand = uniform(-1, 1, size=(N, 2))
       push_vel = rand * max_push_vel
       robot_root_states[env_ids, 7:9] = push_vel
   ```

2. **Action Delay** (动作延迟):
   - 模拟真实机器人的通信延迟

3. **Friction Randomization** (摩擦力随机化):
   - 地面摩擦系数随机化

4. **Mass Randomization** (质量随机化):
   - 机器人质量/惯性随机化

5. **Motor Strength** (电机强度):
   - 模拟电机性能差异

### 6.3 差异总结

| 随机化类型 | holosoma | UniLab Go1 | 状态 |
|------------|----------|------------|------|
| 初始位置 | ✅ | ✅ | 一致 |
| 初始朝向 | ✅ | ✅ | 一致 |
| 初始速度 | ✅ | ✅ | 一致 |
| 关节位置 | ✅ | ✅ | 一致 |
| **推力扰动** | ✅ | ❌ | **缺失** |
| **动作延迟** | ✅ | ❌ | **缺失** |
| **摩擦力** | ✅ | ❌ | **缺失** |
| **质量/惯性** | ✅ | ❌ | **缺失** |
| **电机强度** | ✅ | ❌ | **缺失** |

**问题**: UniLab Go1 的 DR 过于简单，缺少物理参数随机化，可能导致 sim-to-real gap。

---

## 7. Curriculum Learning

### 7.1 holosoma Curriculum

**代码**: `src/holosoma/holosoma/envs/locomotion/locomotion_manager.py:183-236`

**实现**：
1. **Average Episode Length Tracker**:
   ```python
   tracker = curriculum_manager.get_term("average_episode_tracker")
   avg_length = tracker.get_average()
   ```

2. **Penalty Curriculum**:
   - 根据 episode 长度动态调整 penalty 权重
   - 初始 scale 较小，随训练增加
   - 配置: `initial_scale`, `min_scale`, `max_scale`, `degree`

3. **Checkpoint 保存/恢复**:
   ```python
   def get_checkpoint_state():
       return {
           "average_episode_tracker": tracker.state_dict(),
           "reward_penalty_scale": penalty_scale
       }
   ```

### 7.2 UniLab Curriculum

**Go1 环境**: ❌ **未启用 curriculum**

**G1 环境**: ✅ 已实现完整 curriculum（参考 `unilab/envs/curriculum.py`）

**差异**: Go1 环境需要添加 curriculum 支持。

---

## 8. 问题总结与修复建议

### 8.1 算法层面问题

#### 问题 1: `target_entropy_ratio` 不一致

**现状**:
- holosoma: 0.0 (固定熵)
- UniLab: 1.0 (自适应熵)

**影响**:
- 0.0 可能导致策略过早收敛，探索不足
- 1.0 是标准 SAC 设置，但 holosoma 选择 0.0 可能有特殊原因

**建议**:
```python
# 实验对比两种设置
target_entropy_ratio = 0.0  # 先对齐 holosoma
```

#### 问题 2: `weight_decay` 缺失

**现状**:
- holosoma: 0.001 (所有优化器)
- UniLab: 0.0 (未设置)

**影响**: 缺少正则化，可能过拟合

**修复**:
```python
# unilab/algos/torch/fast_sac/learner.py
self.actor_optimizer = optim.Adam(
    self.actor.parameters(),
    lr=actor_lr,
    weight_decay=0.001  # 添加
)
self.critic_optimizer = optim.Adam(
    itertools.chain(self.critic1.parameters(), self.critic2.parameters()),
    lr=critic_lr,
    weight_decay=0.001  # 添加
)
self.alpha_optimizer = optim.Adam(
    [self.log_alpha],
    lr=alpha_lr,
    weight_decay=0.001  # 添加
)
```

---

### 8.2 环境层面问题

#### 问题 3: 命令多样性不足

**现状**:
```python
# UniLab Go1: 固定命令
commands = uniform([0.5, 0.0, 0.0], [0.5, 0.0, 0.0])  # 只前进
```

**问题**: 机器人只学会前进，无法转向/后退

**修复**:
```python
# 修改 unilab/envs/locomotion/go1/joystick.py
@dataclass
class Commands:
    vel_limit = [
        [-0.5, -0.3, -0.8],  # [vx_min, vy_min, vyaw_min]
        [0.8, 0.3, 0.8]      # [vx_max, vy_max, vyaw_max]
    ]
```

#### 问题 4: 缺少 `alive` 奖励

**现状**: Go1 环境没有 `alive` 奖励

**影响**: 缺少基础生存激励

**修复**:
```python
# 添加到 reward_config
reward_scales = {
    "alive": 0.5,  # 新增
    "tracking_lin_vel": 1.0,
    # ... 其他奖励
}

def _reward_alive(self, info, linvel, gyro, dof_pos, qpos):
    return np.ones(self._num_envs, dtype=get_global_dtype())
```

#### 问题 5: Domain Randomization 不足

**现状**: 只有位置/速度随机化

**建议**: 添加物理参数随机化
```python
# 1. 摩擦力随机化
friction = uniform(0.5, 1.5) * default_friction

# 2. 质量随机化
mass = uniform(0.8, 1.2) * default_mass

# 3. 推力扰动（训练中随机施加）
if step % 100 == 0:
    push_vel = uniform(-2.0, 2.0, size=(N, 2))
    apply_push(push_vel)
```

#### 问题 6: Curriculum Learning 未启用

**现状**: Go1 环境未使用 curriculum

**修复**: 参考 G1 实现，添加 `PenaltyCurriculum`
```python
# 在 Go1WalkTask 中添加
from unilab.envs.curriculum import PenaltyCurriculum, EpisodeLengthTracker

def __init__(self, cfg, num_envs, backend_type):
    super().__init__(cfg, backend, num_envs)

    # 添加 curriculum
    self.episode_tracker = EpisodeLengthTracker()
    self.penalty_curriculum = PenaltyCurriculum(
        initial_scale=0.5,
        min_scale=0.5,
        max_scale=1.0,
        degree=0.001
    )
```

---

### 8.3 优先级排序

| 优先级 | 问题 | 类型 | 预期收益 |
|--------|------|------|----------|
| 🔴 P0 | 命令多样性不足 | 环境 | 高 - 学会多方向运动 |
| 🔴 P0 | 缺少 `alive` 奖励 | 环境 | 高 - 基础生存激励 |
| 🟡 P1 | `weight_decay` 缺失 | 算法 | 中 - 防止过拟合 |
| 🟡 P1 | `target_entropy_ratio` | 算法 | 中 - 对齐 holosoma |
| 🟢 P2 | DR 不足 | 环境 | 中 - 提升泛化 |
| 🟢 P2 | Curriculum 未启用 | 环境 | 低 - 加速训练 |

---

### 8.4 修复路线图

**Phase 1: 关键修复** (预计 1-2 天)
1. 添加 `alive` 奖励
2. 修复命令采样范围
3. 添加 `weight_decay`
4. 对齐 `target_entropy_ratio`

**Phase 2: 增强 DR** (预计 2-3 天)
1. 添加摩擦力随机化
2. 添加质量随机化
3. 添加推力扰动

**Phase 3: Curriculum** (预计 1 天)
1. 移植 G1 的 curriculum 到 Go1
2. 配置 penalty scaling

---

## 9. 附录

### 9.1 文件路径索引

**holosoma**:
- 环境: `third-party/holosoma/src/holosoma/holosoma/envs/locomotion/locomotion_manager.py`
- FastSAC: `third-party/holosoma/src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py`
- 网络: `third-party/holosoma/src/holosoma/holosoma/agents/fast_sac/fast_sac.py`
- 配置: `third-party/holosoma/src/holosoma/holosoma/config_types/algo.py`

**UniLab**:
- Go1 环境: `unilab/envs/locomotion/go1/joystick.py`
- Go1 基类: `unilab/envs/locomotion/go1/base.py`
- FastSAC Learner: `unilab/algos/torch/fast_sac/learner.py`
- FastSAC Runner: `unilab/algos/torch/fast_sac/runner.py`
- Curriculum: `unilab/envs/curriculum.py`
- 训练脚本: `scripts/train_offpolicy.py`

### 9.2 参考资料

1. **holosoma 论文**: [待补充]
2. **SAC 原论文**: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications"
3. **Distributional RL**: Bellemare et al., "A Distributional Perspective on RL"
4. **UniLab 设计文档**: `MEMORY.md`

---

**文档维护**: 本文档应随代码更新同步维护。
