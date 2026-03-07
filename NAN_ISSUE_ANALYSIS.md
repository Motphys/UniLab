# NaN 问题紧急分析

## 当前状态

从 c771df6 到 HEAD 的改动：
- ✅ Buffer 竞态修复（已在 c771df6 中）
- ✅ 代码重构（提取共享组件）
- ✅ TD3 数值稳定性（NaN 检查 + 梯度裁剪）
- ✅ SAC safe_tensor 已回滚

## NaN 问题分析

### 可能原因 1: TD3 的 NaN 检查导致训练中断

**问题代码**（TD3 learner）：
```python
loss, nan_metrics = check_nan_loss(qf_loss, {...})
if loss is None:
    return nan_metrics  # 返回 NaN 指标，但不更新网络
```

**问题**：
- 如果检测到 NaN，直接返回，**不更新网络**
- 但训练循环继续，导致后续全是 NaN
- 原来的代码也有这个问题，但可能触发频率不同

### 可能原因 2: 梯度裁剪的副作用

**问题代码**：
```python
if self.weight_decay > 0:
    clip_gradients(self.qnet.parameters(), max_norm=10.0)
```

**问题**：
- `weight_decay` 默认是 0.1（> 0），所以总是裁剪
- 梯度裁剪可能改变训练动态
- 但理论上应该让训练更稳定，不是更不稳定

### 可能原因 3: 共享组件的微妙差异

**EmpiricalNormalization**：
- 从 learner 移到 common
- 代码完全相同
- 不应该有问题 ✅

**Critic 网络**：
- 从 learner 移到 common
- 代码完全相同
- 不应该有问题 ✅

