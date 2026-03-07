# NaN 问题根本原因

## 🔴 发现的 Bug

### 位置：TD3 learner.py 第 368-386 行

```python
loss, nan_metrics = check_nan_loss(qf_loss, {
    "qf_loss": 0.0,
    "qf_max": 0.0,
    "qf_min": 0.0,
})
if loss is None:
    return nan_metrics  # 提前返回 ✅

self.q_optimizer.zero_grad(set_to_none=True)
loss.backward()
if self.weight_decay > 0:
    clip_gradients(self.qnet.parameters(), max_norm=10.0)
self.q_optimizer.step()

return {
    "qf_loss": qf_loss.item(),  # ❌ BUG: 这里用的是原始 qf_loss
    "qf_max": qf1_next_target_value.max().item(),
    "qf_min": qf1_next_target_value.min().item(),
}
```

### 问题分析

**逻辑错误**：
- 如果 `qf_loss` 是 NaN，提前返回 `nan_metrics` ✅
- 但如果 `qf_loss` 正常，最后返回时用的是 `qf_loss.item()` 而不是 `loss.item()`
- `loss` 和 `qf_loss` 是同一个对象（check_nan_loss 返回原对象）
- 所以这个不是 bug ✅

等等，让我重新检查 check_nan_loss 的实现...

## 重新分析

### check_nan_loss 实现

```python
def check_nan_loss(loss: torch.Tensor, default_metrics: dict):
    if torch.isnan(loss) or torch.isinf(loss):
        nan_metrics = {k: float('nan') for k in default_metrics}
        return None, nan_metrics
    return loss, None  # 返回原始 loss
```

**结论**：
- 如果正常：`loss = qf_loss`, `nan_metrics = None`
- 如果 NaN：`loss = None`, `nan_metrics = {...}`

所以代码逻辑是对的 ✅

## 真正的问题

### 可能原因 1: 梯度裁剪导致的问题

**原始代码**（c771df6）：
```python
self.q_optimizer.zero_grad(set_to_none=True)
qf_loss.backward()
if self.weight_decay > 0:
    torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=10.0)
self.q_optimizer.step()
```

**当前代码**：
```python
self.q_optimizer.zero_grad(set_to_none=True)
loss.backward()  # loss 就是 qf_loss
if self.weight_decay > 0:
    clip_gradients(self.qnet.parameters(), max_norm=10.0)
self.q_optimizer.step()
```

**差异**：无实质差异 ✅

### 可能原因 2: 观测归一化的问题

让我检查 TD3 runner 中观测归一化的同步...

## 需要检查的点

1. **是 TD3 还是 SAC 出现 NaN？**
2. **NaN 出现在第几个 iteration？**
3. **是 qf_loss 还是 actor_loss 先出现 NaN？**
4. **观测归一化是否正常工作？**

## 快速诊断建议

在 TD3 learner 的 update_critic 开始处添加调试：
```python
def update_critic(self, data):
    observations = data["obs"]
    print(f"[DEBUG] obs stats: mean={observations.mean().item():.4f}, std={observations.std().item():.4f}, has_nan={torch.isnan(observations).any().item()}")
    # ... 原有代码
```
