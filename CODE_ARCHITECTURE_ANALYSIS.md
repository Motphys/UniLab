# UniLab 代码架构分析与优化建议

## 分析日期
2026-03-07

---

## 一、整体架构评估

### 1.1 模块结构

```
unilab/
├── algos/
│   └── torch/
│       ├── common/           # 共享组件
│       │   ├── async_runner.py
│       │   ├── worker.py
│       │   ├── logger.py
│       │   └── replay_buffer.py
│       ├── fast_sac/          # SAC 实现
│       │   ├── runner.py
│       │   └── learner.py
│       └── fast_td3/          # TD3 实现
│           ├── runner.py
│           └── learner.py
└── ipc/                      # 进程间通信
    ├── shared_buffer.py
    ├── shared_weight_sync.py
    └── async_runner.py
```

### 1.2 架构优点

1. **清晰的分层**：IPC 层、算法层、通用组件层分离明确
2. **异步训练架构**：collector 和 learner 进程分离，提高吞吐量
3. **共享内存优化**：使用 SharedReplayBuffer 和 SharedWeightSync 避免数据拷贝
4. **基类抽象**：AsyncRunner 提供统一的异步训练框架

### 1.3 架构问题

1. **代码重复严重**：SAC 和 TD3 的 runner 有 ~85% 代码重复
2. **组件耦合**：SharedObsNormStats 定义在 SAC runner 中，TD3 需要导入
3. **缺少抽象层**：观测归一化、数值稳定性等通用功能未抽象
4. **配置分散**：超参数硬编码在多处，缺少统一配置管理

---

## 二、SAC vs TD3 实现差异分析

### 2.1 Runner 层差异

| 特性 | SAC | TD3 | 差异说明 |
|------|-----|-----|---------|
| **观测归一化同步** | ✅ 有 SharedObsNormStats | ✅ 有（新增） | TD3 从 SAC 导入 |
| **训练循环结构** | 相同 | 相同 | 100% 重复代码 |
| **Collector 启动** | 相同 | 相同 | 100% 重复代码 |
| **指标收集** | 相同 | 相同 | 100% 重复代码 |
| **权重同步** | 相同 | 相同 | 100% 重复代码 |
| **Logger 集成** | 相同 | 相同 | 100% 重复代码 |

**代码重复率：~320 行 / 370 行 = 86.5%**

### 2.2 Learner 层差异

| 组件 | SAC | TD3 | 关键差异 |
|------|-----|-----|---------|
| **Actor 架构** | SiLU + LayerNorm | ReLU | 激活函数和归一化 |
| **Actor 输出** | 随机策略（Gaussian） | 确定性策略 + 噪声 | 探索机制不同 |
| **Critic 架构** | 分布式 Q (C51) | 分布式 Q (C51) | 相同 |
| **目标网络** | Critic only | Critic only | 相同 |
| **观测归一化** | EmpiricalNormalization | EmpiricalNormalization | 相同实现 |
| **优化器** | AdamW | AdamW | 相同 |
| **学习率调度** | Cosine | Cosine | 相同 |
| **数值稳定性** | ✅ NaN 保护 | ✅ NaN 保护（新增） | SAC 原有，TD3 新增 |
| **梯度裁剪** | ❌ 无 | ✅ 有（新增） | TD3 新增 |
| **熵正则化** | ✅ 自动调整 alpha | ❌ 无 | SAC 独有 |

### 2.3 代码复用情况

**完全重复的代码**：
- `EmpiricalNormalization` 类（~50 行）
- `DistributionalQNetwork` 类（~80 行）
- `Critic` 类（~50 行）
- Runner 的训练循环（~200 行）

**可复用但未复用的代码**：
- 观测归一化同步逻辑
- 数值稳定性检查
- 梯度裁剪逻辑
- Buffer 采样和批处理

---

## 三、具体问题分析

### 3.1 SharedObsNormStats 的设计问题

**当前实现**：
```python
# 在 fast_sac/runner.py 中定义
class SharedObsNormStats:
    def __init__(self, ctx):
        self.q = ctx.Queue(maxsize=2)
        self.last_stats = None
```

**问题**：
1. 定义在算法特定文件中，不是通用组件
2. TD3 需要跨模块导入：`from unilab.algos.torch.fast_sac.runner import SharedObsNormStats`
3. 违反单一职责原则（runner 不应定义 IPC 组件）
4. 如果添加新算法（如 PPO），需要再次导入

### 3.2 EmpiricalNormalization 重复实现

**当前状态**：
- `fast_sac/learner.py` 中有完整实现（~50 行）
- `fast_td3/learner.py` 中有完整实现（~50 行）
- 两份代码 100% 相同

**影响**：
- 维护成本翻倍
- Bug 修复需要同步两处
- 新功能添加容易遗漏

### 3.3 Critic 网络重复实现

**当前状态**：
- `DistributionalQNetwork` 在两个 learner 中完全相同
- `Critic` 类在两个 learner 中完全相同
- 总计 ~130 行重复代码

### 3.4 数值稳定性逻辑分散

**SAC 的 NaN 保护**：
```python
# 在 SACActor.forward() 中
mean = torch.clamp(mean, -10.0, 10.0)
mean = torch.nan_to_num(mean, nan=0.0)
log_std = torch.nan_to_num(log_std, nan=self.log_std_min)
```

**TD3 的 NaN 保护**：
```python
# 在 update_critic() 和 update_actor() 中
if torch.isnan(qf_loss) or torch.isinf(qf_loss):
    return {...}
```

**问题**：
- 两种不同的保护策略
- 没有统一的数值稳定性工具
- 难以确保一致性

### 3.5 训练循环代码重复

**重复的逻辑**（在两个 runner 中）：
1. Buffer 和 WeightSync 初始化（~20 行）
2. Collector 进程启动（~30 行）
3. 同步/异步采集协调（~40 行）
4. 指标收集和日志记录（~50 行）
5. 检查点保存（~10 行）

**总计重复**：~150 行核心训练逻辑

---

## 四、优化建议（按优先级排序）

### 优先级 1：提取共享组件到 common 模块

#### 4.1.1 移动 SharedObsNormStats 到 IPC 层

**目标位置**：`unilab/ipc/shared_obs_stats.py`

**理由**：
- 这是进程间通信组件，应该在 IPC 层
- 所有算法都可能需要观测归一化同步
- 消除跨算法模块的依赖

**影响范围**：
- 创建新文件：`unilab/ipc/shared_obs_stats.py`
- 修改：`fast_sac/runner.py`（删除定义，改为导入）
- 修改：`fast_td3/runner.py`（修改导入路径）

#### 4.1.2 提取 EmpiricalNormalization 到 common

**目标位置**：`unilab/algos/torch/common/normalization.py`

**理由**：
- 两个算法使用完全相同的实现
- 未来算法（PPO、DDPG）也会需要
- 便于统一维护和测试

**影响范围**：
- 创建新文件：`unilab/algos/torch/common/normalization.py`
- 修改：`fast_sac/learner.py`（删除定义，改为导入）
- 修改：`fast_td3/learner.py`（删除定义，改为导入）

#### 4.1.3 提取分布式 Critic 网络到 common

**目标位置**：`unilab/algos/torch/common/networks.py`

**内容**：
- `DistributionalQNetwork`
- `Critic`（双 Q 网络）

**理由**：
- SAC 和 TD3 使用完全相同的 C51 实现
- 减少 ~130 行重复代码
- 便于添加其他分布式 RL 算法

**影响范围**：
- 创建新文件：`unilab/algos/torch/common/networks.py`
- 修改：`fast_sac/learner.py`（删除定义，改为导入）
- 修改：`fast_td3/learner.py`（删除定义，改为导入）

---

### 优先级 2：抽象训练循环到基类

#### 4.2.1 创建 OffPolicyRunner 基类

**目标位置**：`unilab/algos/torch/common/off_policy_runner.py`

**抽象内容**：
```python
class OffPolicyRunner(AsyncRunner):
    """Off-policy 算法的通用训练循环"""

    def learn(self, max_iterations, save_interval, log_dir, logger_type):
        # 1. 初始化 learner（子类实现）
        # 2. 创建 shared buffer
        # 3. 创建 weight sync
        # 4. 创建 obs norm stats（如果需要）
        # 5. 启动 collector
        # 6. 训练循环（模板方法）
        # 7. 保存检查点

    @abstractmethod
    def _build_learner(self):
        """子类实现：构建具体的 learner"""
        pass

    @abstractmethod
    def _update_step(self, learner, batch):
        """子类实现：单步更新逻辑"""
        pass
```

**好处**：
- 消除 ~200 行重复代码
- 统一训练流程，减少 bug
- 新算法只需实现 learner 和 update 逻辑

**影响范围**：
- 创建新文件：`unilab/algos/torch/common/off_policy_runner.py`
- 修改：`fast_sac/runner.py`（继承 OffPolicyRunner，删除重复代码）
- 修改：`fast_td3/runner.py`（继承 OffPolicyRunner，删除重复代码）

---

### 优先级 3：统一数值稳定性工具

#### 4.3.1 创建数值稳定性工具模块

**目标位置**：`unilab/algos/torch/common/numerical_stability.py`

**内容**：
```python
class NumericalStabilityMixin:
    """数值稳定性工具集"""

    @staticmethod
    def safe_forward(loss, metrics_dict):
        """检查 loss 是否为 NaN/Inf，返回安全值"""
        if torch.isnan(loss) or torch.isinf(loss):
            return None, {k: float('nan') for k in metrics_dict}
        return loss, None

    @staticmethod
    def clip_gradients(parameters, max_norm=10.0):
        """梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    @staticmethod
    def safe_tensor(tensor, nan_value=0.0, clamp_range=(-10.0, 10.0)):
        """张量安全化：clamp + nan_to_num"""
        tensor = torch.clamp(tensor, *clamp_range)
        return torch.nan_to_num(tensor, nan=nan_value)
```

**好处**：
- 统一数值稳定性策略
- 可配置的保护级别
- 便于测试和调试

**影响范围**：
- 创建新文件：`unilab/algos/torch/common/numerical_stability.py`
- 修改：`fast_sac/learner.py`（使用 Mixin）
- 修改：`fast_td3/learner.py`（使用 Mixin）

---

### 优先级 4：配置管理优化

#### 4.4.1 创建配置类

**目标位置**：`unilab/algos/torch/common/config.py`

**内容**：
```python
@dataclass
class OffPolicyConfig:
    """Off-policy 算法通用配置"""
    # 环境配置
    env_name: str
    num_envs: int = 4096

    # 训练配置
    max_iterations: int = 5000
    batch_size: int = 8192
    replay_buffer_n: int = 1000
    warmup_steps: int = 50

    # 优化器配置
    gamma: float = 0.97
    tau: float = 0.1
    weight_decay: float = 0.1

    # 数值稳定性
    gradient_clip_norm: float = 10.0
    enable_nan_check: bool = True

    # 观测归一化
    obs_normalization: bool = True

@dataclass
class TD3Config(OffPolicyConfig):
    """TD3 特定配置"""
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    use_cdq: bool = True

@dataclass
class SACConfig(OffPolicyConfig):
    """SAC 特定配置"""
    auto_alpha: bool = True
    target_entropy: float = None
```

**好处**：
- 类型安全的配置
- 便于序列化和加载
- 清晰的默认值管理
- 支持配置继承

---

## 五、重构路线图

### 阶段 1：提取共享组件（1-2 天）

**步骤**：
1. 创建 `unilab/ipc/shared_obs_stats.py`，移动 SharedObsNormStats
2. 创建 `unilab/algos/torch/common/normalization.py`，移动 EmpiricalNormalization
3. 创建 `unilab/algos/torch/common/networks.py`，移动 Critic 相关类
4. 更新所有导入路径
5. 运行测试确保功能不变

**风险**：低（纯代码移动，无逻辑变更）

### 阶段 2：抽象训练循环（2-3 天）

**步骤**：
1. 创建 `OffPolicyRunner` 基类
2. 重构 `FastSACRunner` 继承基类
3. 重构 `FastTD3Runner` 继承基类
4. 删除重复代码
5. 运行完整训练测试

**风险**：中（涉及核心训练逻辑）

### 阶段 3：统一数值稳定性（1 天）

**步骤**：
1. 创建 `NumericalStabilityMixin`
2. 在 learner 中应用 Mixin
3. 统一 SAC 和 TD3 的保护策略
4. 添加单元测试

**风险**：低（增强现有功能）

### 阶段 4：配置管理（1 天）

**步骤**：
1. 创建配置类
2. 更新 runner 和 learner 使用配置对象
3. 添加配置序列化/反序列化
4. 更新训练脚本

**风险**：低（向后兼容）

---

## 六、预期收益

### 6.1 代码质量提升

| 指标 | 当前 | 重构后 | 改善 |
|------|------|--------|------|
| 代码重复率 | ~35% | ~5% | -30% |
| 单个算法代码量 | ~900 行 | ~400 行 | -55% |
| 共享组件数量 | 3 个 | 8 个 | +167% |
| 测试覆盖难度 | 高 | 低 | 显著降低 |

### 6.2 维护成本降低

- **Bug 修复**：从修改 2 处降低到 1 处
- **新功能添加**：在 common 模块添加，所有算法自动受益
- **新算法开发**：从 ~900 行降低到 ~400 行（减少 55% 工作量）

### 6.3 可扩展性提升

- 添加新的 off-policy 算法（DDPG、SAC-Discrete）只需：
  1. 实现 Learner 类（~200 行）
  2. 继承 OffPolicyRunner（~50 行）
  3. 创建配置类（~20 行）

  总计 ~270 行 vs 当前 ~900 行

---

## 七、风险评估与缓解

### 7.1 重构风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 引入新 bug | 中 | 高 | 完整的回归测试套件 |
| 性能下降 | 低 | 中 | 性能基准测试 |
| 破坏现有代码 | 低 | 高 | 保持向后兼容，渐进式重构 |
| 团队学习成本 | 中 | 低 | 详细文档和示例 |

### 7.2 缓解策略

1. **渐进式重构**：按阶段进行，每个阶段独立测试
2. **保持向后兼容**：旧 API 保留，标记为 deprecated
3. **完整测试**：每个阶段后运行完整训练测试
4. **性能监控**：对比重构前后的训练速度和内存使用
5. **文档更新**：同步更新架构文档和使用示例

---

## 八、总结

### 当前架构的核心问题

1. **代码重复严重**：SAC 和 TD3 有 ~35% 代码重复
2. **组件耦合**：SharedObsNormStats 定义在错误的位置
3. **缺少抽象**：训练循环、数值稳定性未抽象
4. **维护困难**：修改需要同步多处

### 优化后的预期效果

1. **代码量减少 55%**：从 ~900 行降到 ~400 行/算法
2. **重复率降低到 5%**：共享组件充分复用
3. **新算法开发加速 70%**：只需实现核心逻辑
4. **维护成本降低 50%**：统一的组件和工具

### 建议执行顺序

**立即执行**（优先级 1）：
- 移动 SharedObsNormStats 到 IPC 层
- 提取 EmpiricalNormalization 到 common

**短期执行**（1-2 周内，优先级 2-3）：
- 抽象 OffPolicyRunner 基类
- 统一数值稳定性工具

**中期执行**（1 个月内，优先级 4）：
- 配置管理优化
- 完善文档和测试

---

## 附录：重构前后代码对比示例

### A.1 使用 EmpiricalNormalization

**重构前**：
```python
# fast_td3/learner.py 中定义 50 行
class EmpiricalNormalization(nn.Module):
    ...

# fast_sac/learner.py 中定义 50 行（完全相同）
class EmpiricalNormalization(nn.Module):
    ...
```

**重构后**：
```python
# common/normalization.py 中定义 50 行
class EmpiricalNormalization(nn.Module):
    ...

# fast_td3/learner.py 和 fast_sac/learner.py
from unilab.algos.torch.common.normalization import EmpiricalNormalization
```

**收益**：减少 50 行重复代码

### A.2 训练循环

**重构前**：
```python
# fast_td3/runner.py 中 200 行训练循环
def learn(self, ...):
    # 初始化 buffer, weight_sync, collector...
    # 训练循环...
    # 保存检查点...

# fast_sac/runner.py 中 200 行训练循环（85% 相同）
def learn(self, ...):
    # 初始化 buffer, weight_sync, collector...
    # 训练循环...
    # 保存检查点...
```

**重构后**：
```python
# common/off_policy_runner.py 中 200 行通用训练循环
class OffPolicyRunner(AsyncRunner):
    def learn(self, ...):
        # 通用训练循环模板
        ...

# fast_td3/runner.py 只需 50 行
class FastTD3Runner(OffPolicyRunner):
    def _build_learner(self): ...
    def _update_step(self, learner, batch): ...

# fast_sac/runner.py 只需 50 行
class FastSACRunner(OffPolicyRunner):
    def _build_learner(self): ...
    def _update_step(self, learner, batch): ...
```

**收益**：减少 ~300 行重复代码，新算法开发加速

